//! HTTP REST API transport for p2a-mcp.
//!
//! Provides a REST API that mirrors the MCP tool interface,
//! allowing web clients to interact with the analytics server.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::audit::AuditLogger;
use crate::config::ServerConfig;
use crate::server::AnalyticsServer;
use crate::session::{SessionError, SessionManager};
use crate::transport::TransportResult;

/// Shared application state for HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    /// Analytics server for tool execution
    pub server: Arc<AnalyticsServer>,
    /// Session manager for multi-user support
    pub session_manager: Arc<SessionManager>,
    /// Persistent session manager for database operations (optional)
    #[cfg(feature = "db")]
    pub persistent_manager: Option<Arc<crate::persistent_session::PersistentSessionManager>>,
    /// Audit logger for tool call logging
    pub audit_logger: Arc<AuditLogger>,
}

/// Start the HTTP transport.
pub async fn start_http_transport(config: &ServerConfig) -> TransportResult<()> {
    let session_manager = Arc::new(SessionManager::new(config.session.clone()));

    // Start background cleanup task (every 10 minutes)
    session_manager.clone().start_cleanup_task(10);

    let server = Arc::new(AnalyticsServer::new());

    // Create audit logger
    let audit_logger = Arc::new(match AuditLogger::new(&config.audit) {
        Ok(logger) => logger,
        Err(e) => {
            tracing::warn!("Failed to initialize audit logger: {}", e);
            AuditLogger::disabled()
        }
    });

    // Create persistent session manager if db feature is enabled
    #[cfg(feature = "db")]
    let persistent_manager = {
        let db_path = config.http.db_path.as_deref();
        match crate::persistent_session::PersistentSessionManager::new(
            db_path,
            config.session.max_sessions,
            config.session.ttl_minutes,
        )
        .await
        {
            Ok(manager) => {
                tracing::info!("Database persistence enabled");
                Some(Arc::new(manager))
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to initialize database, running without persistence: {}",
                    e
                );
                None
            }
        }
    };

    #[cfg(feature = "db")]
    let state = AppState {
        server,
        session_manager,
        persistent_manager: persistent_manager.clone(),
        audit_logger,
    };

    #[cfg(not(feature = "db"))]
    let state = AppState {
        server,
        session_manager,
        audit_logger,
    };

    #[cfg(feature = "db")]
    let app = create_router(state, config, persistent_manager);

    #[cfg(not(feature = "db"))]
    let app = create_router(state, config);

    let addr = config.http.addr;
    tracing::info!("Starting HTTP transport on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        crate::transport::TransportError::Http(format!("Failed to bind to {}: {}", addr, e))
    })?;

    axum::serve(listener, app)
        .await
        .map_err(|e| crate::transport::TransportError::Http(format!("HTTP server error: {}", e)))?;

    Ok(())
}

/// Create the axum router with all routes.
#[cfg(feature = "db")]
pub fn create_router(
    state: AppState,
    config: &ServerConfig,
    persistent_manager: Option<Arc<crate::persistent_session::PersistentSessionManager>>,
) -> Router {
    let router = create_base_router(state, config);

    // Add conversation routes if persistent manager is available

    if let Some(manager) = persistent_manager {
        tracing::info!("Adding conversation routes to /api/*");
        let conv_state = super::conversation::ConversationState {
            session_manager: manager,
        };
        router.nest("/api", super::conversation::conversation_routes(conv_state))
    } else {
        tracing::warn!("Conversation routes NOT added (no persistent manager)");
        router
    }
}

/// Create the axum router with all routes (without db feature).
#[cfg(not(feature = "db"))]
pub fn create_router(state: AppState, config: &ServerConfig) -> Router {
    create_base_router(state, config)
}

/// Create the base router with core routes.
fn create_base_router(state: AppState, config: &ServerConfig) -> Router {
    let cors = if config.http.cors_permissive {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else if !config.http.cors_origins.is_empty() {
        let origins: Vec<_> = config
            .http
            .cors_origins
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
    };

    let router = Router::new()
        // Health check
        .route("/health", get(health_check))
        // Session management
        .route("/api/sessions", post(create_session))
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}", delete(delete_session))
        // Tool discovery
        .route("/api/tools", get(list_tools))
        // Tool execution
        .route("/api/tools/{name}", post(call_tool))
        // File browser
        .route("/api/files", get(list_files));

    // Add WebSocket route if feature is enabled
    #[cfg(feature = "websocket")]
    let router = router.route("/ws", get(super::websocket::ws_handler));

    // Add LLM routes if feature is enabled
    #[cfg(feature = "llm")]
    let router = router
        .route("/api/llm/chat", post(llm_chat))
        .route("/api/llm/chat/stream", post(llm_chat_stream))
        .route("/api/llm/models", get(llm_list_models))
        .route("/api/llm/env-keys", get(llm_env_keys))
        .route("/api/llm/generate-title", post(llm_generate_title));

    router
        // Add middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// Response wrapper for API endpoints.
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

/// Request to create a new session.
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    #[serde(default)]
    pub user_id: Option<String>,
}

/// Response for session creation.
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
}

/// Request to call a tool.
#[derive(Debug, Deserialize)]
pub struct CallToolRequest {
    /// Session ID (required for HTTP transport)
    pub session_id: String,
    /// Tool arguments as JSON
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Tool execution result.
#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub success: bool,
    pub content: Vec<ContentItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Content item in tool result.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentItem {
    Text { text: String },
    Image { data: String, mime_type: String },
}

/// Tool definition for discovery.
#[derive(Debug, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// =============================================================================
// Route Handlers
// =============================================================================

/// Health check endpoint.
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let session_count = state.session_manager.session_count().await;
    Json(serde_json::json!({
        "status": "ok",
        "server": "p2a-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "active_sessions": session_count
    }))
}

/// Create a new session.
async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    // First create the in-memory session
    match state
        .session_manager
        .create_session(request.user_id.clone())
        .await
    {
        Ok(session_id) => {
            // Also register in persistent storage if available (for conversation support)
            #[cfg(feature = "db")]
            if let Some(ref persistent_manager) = state.persistent_manager {
                match persistent_manager
                    .register_session(&session_id, request.user_id)
                    .await
                {
                    Ok(()) => {
                        tracing::info!(
                            "Session {} registered in database for conversation support",
                            session_id
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to register session in database: {}", e);
                        // Continue anyway - session exists in memory, conversations just won't work
                    }
                }
            }

            (
                StatusCode::CREATED,
                Json(ApiResponse::success(CreateSessionResponse { session_id })),
            )
        }
        Err(SessionError::MaxSessionsReached) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiResponse::error("Maximum number of sessions reached")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// List all active sessions.
async fn list_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let sessions = state.session_manager.list_sessions().await;
    Json(ApiResponse::success(sessions))
}

/// Get session information.
async fn get_session(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.session_manager.get_session(&id).await {
        Ok(session) => {
            let info = session.info().await;
            (StatusCode::OK, Json(ApiResponse::success(info)))
        }
        Err(SessionError::NotFound) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Session not found")),
        ),
        Err(SessionError::Expired) => (
            StatusCode::GONE,
            Json(ApiResponse::error("Session expired")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// Delete a session.
async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.delete_session(&id).await {
        Ok(()) => (StatusCode::NO_CONTENT, Json(ApiResponse::success(()))),
        Err(SessionError::NotFound) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Session not found")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// List available tools.
async fn list_tools(State(state): State<AppState>) -> impl IntoResponse {
    let tools = state.server.list_tools();
    Json(ApiResponse::success(tools))
}

/// Query params for file listing.
#[derive(Debug, Deserialize)]
pub struct ListFilesQuery {
    /// Directory path to list (defaults to home directory)
    pub path: Option<String>,
}

/// File entry in directory listing.
#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

/// Response for file listing.
#[derive(Debug, Serialize)]
pub struct ListFilesResponse {
    pub path: String,
    pub parent: Option<String>,
    pub entries: Vec<FileEntry>,
}

/// List files in a directory.
async fn list_files(Query(query): Query<ListFilesQuery>) -> impl IntoResponse {
    use std::path::PathBuf;

    // Default to home directory
    let path = match &query.path {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")),
    };

    // Read directory
    let entries = match std::fs::read_dir(&path) {
        Ok(read_dir) => {
            let mut entries: Vec<FileEntry> = read_dir
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let metadata = entry.metadata().ok()?;
                    let name = entry.file_name().to_string_lossy().to_string();

                    // Skip hidden files
                    if name.starts_with('.') {
                        return None;
                    }

                    let is_dir = metadata.is_dir();

                    // Filter to only show directories and supported data files
                    if !is_dir {
                        let ext = name.split('.').next_back().unwrap_or("").to_lowercase();
                        if !["csv", "parquet", "json", "xlsx", "xls", "dta", "sas7bdat"]
                            .contains(&ext.as_str())
                        {
                            return None;
                        }
                    }

                    Some(FileEntry {
                        name,
                        path: entry.path().to_string_lossy().to_string(),
                        is_dir,
                        size: if is_dir { None } else { Some(metadata.len()) },
                    })
                })
                .collect();

            // Sort: directories first, then files, both alphabetically
            entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            });

            entries
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::<ListFilesResponse>::error(format!(
                    "Cannot read directory: {}",
                    e
                ))),
            );
        }
    };

    let parent = path.parent().map(|p| p.to_string_lossy().to_string());

    (
        StatusCode::OK,
        Json(ApiResponse::success(ListFilesResponse {
            path: path.to_string_lossy().to_string(),
            parent,
            entries,
        })),
    )
}

/// Call a tool.
async fn call_tool(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<CallToolRequest>,
) -> impl IntoResponse {
    // Get the session
    let session = match state.session_manager.get_session(&request.session_id).await {
        Ok(s) => s,
        Err(SessionError::NotFound) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::error("Session not found")),
            );
        }
        Err(SessionError::Expired) => {
            return (
                StatusCode::GONE,
                Json(ApiResponse::error("Session expired")),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            );
        }
    };

    // Execute the tool with session context and timing
    let start = std::time::Instant::now();
    let tool_result = state
        .server
        .call_tool_with_session(&name, request.arguments.clone(), &session)
        .await;
    let duration_ms = start.elapsed().as_millis() as u64;

    match tool_result {
        Ok(result) => {
            // Extract text content from result for logging
            let result_text: String = result
                .content
                .iter()
                .filter_map(|item| match item {
                    ContentItem::Text { text } => Some(text.clone()),
                    ContentItem::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Audit logging (if enabled)
            if state.audit_logger.is_enabled() {
                let entry = AuditLogger::create_entry(
                    &request.session_id,
                    &name,
                    &request.arguments,
                    result.success,
                    duration_ms,
                    &result_text,
                    None, // Client IP would require extracting from request headers
                );
                state.audit_logger.log(entry).await;
            }

            // Capture dataset metadata for load_dataset, upload_dataset, and create_dataset
            #[cfg(feature = "db")]
            if (name == "load_dataset" || name == "upload_dataset" || name == "create_dataset")
                && result.success
            {
                if let Some(ref persistent_manager) = state.persistent_manager {
                    tracing::info!(
                        tool = %name,
                        session_id = %request.session_id,
                        result_len = result_text.len(),
                        "[CALL_TOOL] Capturing dataset metadata for direct tool call"
                    );

                    // Use the same capture function as StreamingToolExecutor
                    #[cfg(feature = "llm")]
                    llm_handlers::StreamingToolExecutor::capture_dataset_metadata_static(
                        persistent_manager.db().clone(),
                        &request.session_id,
                        &name,
                        &request.arguments,
                        &result_text,
                    )
                    .await;
                }
            }

            (StatusCode::OK, Json(ApiResponse::success(result)))
        }
        Err(e) => {
            // Audit log for failures
            if state.audit_logger.is_enabled() {
                let entry = AuditLogger::create_entry(
                    &request.session_id,
                    &name,
                    &request.arguments,
                    false,
                    duration_ms,
                    &e,
                    None,
                );
                state.audit_logger.log(entry).await;
            }

            (StatusCode::BAD_REQUEST, Json(ApiResponse::error(e)))
        }
    }
}

// =============================================================================
// LLM Endpoints (feature-gated)
// =============================================================================

#[cfg(feature = "llm")]
mod llm_handlers {
    use super::*;
    use crate::llm::{
        AnthropicProvider, LlmProvider, Message, OllamaProvider, OpenAIProvider, ProviderConfig,
        ProviderType, ToolExecutor, build_enhanced_dataset_context, get_mcp_tool_definitions,
        get_system_prompt_with_context,
    };
    use crate::session::Session;

    /// Request for LLM chat.
    #[derive(Debug, Deserialize)]
    pub struct LlmChatRequest {
        pub session_id: String,
        pub message: String,
        #[serde(default)]
        pub provider: Option<ProviderConfig>,
        #[serde(default)]
        pub history: Vec<Message>,
        /// Whether to have the LLM interpret tool results (default: true)
        #[serde(default = "default_interpret")]
        pub interpret: bool,
        /// Optional conversation ID for persistence
        #[serde(default)]
        pub conversation_id: Option<String>,
        /// Whether to auto-retrieve history from database when conversation_id
        /// is provided and history is empty (default: true)
        #[serde(default = "default_retrieve_history")]
        pub retrieve_history: bool,
    }

    fn default_retrieve_history() -> bool {
        true
    }

    fn default_interpret() -> bool {
        true
    }

    /// Response for LLM chat.
    #[derive(Debug, Serialize)]
    pub struct LlmChatResponse {
        pub message: Message,
    }

    /// Response for listing models.
    #[derive(Debug, Serialize)]
    pub struct LlmModelsResponse {
        pub provider: String,
        pub models: Vec<String>,
    }

    /// Internal tool executor that uses the analytics server.
    pub struct SessionToolExecutor {
        server: Arc<crate::server::AnalyticsServer>,
        session: Arc<Session>,
    }

    impl SessionToolExecutor {
        pub fn new(server: Arc<crate::server::AnalyticsServer>, session: Arc<Session>) -> Self {
            Self { server, session }
        }
    }

    #[async_trait::async_trait]
    impl ToolExecutor for SessionToolExecutor {
        async fn execute(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Result<String, String> {
            tracing::warn!(">>> SessionToolExecutor::execute called for tool: {}", name);
            tracing::info!(tool = %name, "Executing tool");
            let start = std::time::Instant::now();
            let result = self
                .server
                .call_tool_with_session(name, arguments, &self.session)
                .await;
            let elapsed = start.elapsed();
            tracing::info!(tool = %name, elapsed_ms = %elapsed.as_millis(), "Tool execution completed");
            match result {
                Ok(result) => {
                    // Convert tool result to string
                    let content = result
                        .content
                        .iter()
                        .map(|item| match item {
                            ContentItem::Text { text } => text.clone(),
                            ContentItem::Image { .. } => "[Image output]".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(content)
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Create a provider from config (defaults to Ollama).
    /// If no API key is provided, checks for environment variables:
    /// - OPENAI_API_KEY for OpenAI
    /// - ANTHROPIC_API_KEY for Anthropic
    fn create_provider(config: Option<ProviderConfig>) -> Box<dyn LlmProvider> {
        let mut config = config.unwrap_or_default();

        // Fill in API key from environment variable if not provided
        if config.api_key.is_none()
            || config
                .api_key
                .as_ref()
                .map(|k| k.is_empty())
                .unwrap_or(false)
        {
            let env_key = match config.provider_type {
                ProviderType::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
                ProviderType::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
                ProviderType::Ollama => None, // Ollama doesn't need an API key
            };
            if env_key.is_some() {
                tracing::info!(
                    provider = %config.provider_type,
                    "Using API key from environment variable"
                );
                config.api_key = env_key;
            }
        }

        match config.provider_type {
            ProviderType::Ollama => Box::new(OllamaProvider::new(config)),
            ProviderType::OpenAI => Box::new(OpenAIProvider::new(config)),
            ProviderType::Anthropic => Box::new(AnthropicProvider::new(config)),
        }
    }

    /// LLM chat endpoint.
    pub async fn llm_chat(
        State(state): State<AppState>,
        Json(request): Json<LlmChatRequest>,
    ) -> impl IntoResponse {
        // Get the session
        let session = match state.session_manager.get_session(&request.session_id).await {
            Ok(s) => s,
            Err(crate::session::SessionError::NotFound) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ApiResponse::error("Session not found")),
                );
            }
            Err(crate::session::SessionError::Expired) => {
                return (
                    StatusCode::GONE,
                    Json(ApiResponse::error("Session expired")),
                );
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse::error(e.to_string())),
                );
            }
        };

        // Create provider and tool executor
        let provider = create_provider(request.provider);
        let tool_executor = SessionToolExecutor::new(state.server.clone(), session.clone());
        let tools = get_mcp_tool_definitions();

        // Build enhanced dataset context from currently loaded datasets
        // This includes data types, sample values, and basic statistics
        let dataset_context = {
            let datasets = session.datasets.read().await;
            build_enhanced_dataset_context(&datasets)
        };

        tracing::info!(provider = %provider.provider_type(), num_tools = %tools.len(), has_datasets = dataset_context.is_some(), "Starting LLM chat");

        // Build message history with dataset context
        let system_prompt = get_system_prompt_with_context(dataset_context.as_deref());
        let mut messages = vec![Message::system(system_prompt)];
        messages.extend(request.history);
        messages.push(Message::user(request.message));

        // Execute chat
        let start = std::time::Instant::now();
        let result = provider
            .chat(&messages, &tools, &tool_executor, request.interpret)
            .await;
        let elapsed = start.elapsed();

        match result {
            Ok(response) => {
                tracing::info!(elapsed_ms = %elapsed.as_millis(), "LLM chat completed successfully");
                (
                    StatusCode::OK,
                    Json(ApiResponse::success(LlmChatResponse { message: response })),
                )
            }
            Err(e) => {
                tracing::error!(error = %e, elapsed_ms = %elapsed.as_millis(), "LLM chat failed");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse::error(e.to_string())),
                )
            }
        }
    }

    /// List available LLM models.
    pub async fn llm_list_models(State(_state): State<AppState>) -> impl IntoResponse {
        let provider = create_provider(None);

        match provider.list_models().await {
            Ok(models) => (
                StatusCode::OK,
                Json(ApiResponse::success(LlmModelsResponse {
                    provider: provider.provider_type().to_string(),
                    models,
                })),
            ),
            Err(e) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ApiResponse::error(format!(
                    "LLM provider not available: {}",
                    e
                ))),
            ),
        }
    }

    /// Response for env keys check.
    #[derive(Debug, Serialize)]
    pub struct EnvKeysResponse {
        pub openai: bool,
        pub anthropic: bool,
    }

    /// Check which API keys are available from environment variables.
    pub async fn llm_env_keys() -> impl IntoResponse {
        let openai = std::env::var("OPENAI_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false);
        let anthropic = std::env::var("ANTHROPIC_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false);

        (
            StatusCode::OK,
            Json(ApiResponse::success(EnvKeysResponse { openai, anthropic })),
        )
    }

    /// Request for generating a conversation title.
    #[derive(Debug, Deserialize)]
    pub struct GenerateTitleRequest {
        /// The user's first message to base the title on
        pub user_message: String,
        /// Optional assistant response for more context
        #[serde(default)]
        pub assistant_response: Option<String>,
        /// Provider configuration
        #[serde(default)]
        pub provider: Option<ProviderConfig>,
    }

    /// Response for title generation.
    #[derive(Debug, Serialize)]
    pub struct GenerateTitleResponse {
        pub title: String,
    }

    /// Generate a conversation title using the LLM.
    pub async fn llm_generate_title(
        Json(request): Json<GenerateTitleRequest>,
    ) -> impl IntoResponse {
        let provider = create_provider(request.provider);

        // Build a simple prompt for title generation
        let context = if let Some(ref response) = request.assistant_response {
            format!(
                "User: {}\nAssistant: {}",
                request.user_message,
                // Truncate long responses
                if response.len() > 500 {
                    &response[..500]
                } else {
                    response
                }
            )
        } else {
            format!("User: {}", request.user_message)
        };

        let prompt = format!(
            "Generate a very short title (3-6 words, no quotes) for this conversation:\n\n{}\n\nTitle:",
            context
        );

        let messages = vec![
            Message::system(
                "You are a helpful assistant that generates concise conversation titles. Respond with only the title, no quotes or punctuation.",
            ),
            Message::user(prompt),
        ];

        // Use chat without tools for simple title generation
        match provider
            .chat(&messages, &[], &NoOpToolExecutor, false)
            .await
        {
            Ok(response) => {
                // Clean up the title
                let title = response
                    .content
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();

                // Limit title length
                let title = if title.len() > 50 {
                    format!("{}...", title.chars().take(47).collect::<String>())
                } else {
                    title
                };

                (
                    StatusCode::OK,
                    Json(ApiResponse::success(GenerateTitleResponse { title })),
                )
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to generate title");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse::error(format!(
                        "Failed to generate title: {}",
                        e
                    ))),
                )
            }
        }
    }

    /// No-op tool executor for simple chat without tools.
    struct NoOpToolExecutor;

    #[async_trait::async_trait]
    impl ToolExecutor for NoOpToolExecutor {
        async fn execute(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, String> {
            Err("Tool execution not available".to_string())
        }
    }

    /// Progress event for streaming chat.
    #[derive(Debug, Clone, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ProgressEvent {
        Status {
            message: String,
        },
        ToolStart {
            tool: String,
            arguments: serde_json::Value,
        },
        ToolEnd {
            tool: String,
            elapsed_ms: u64,
            result: Option<String>,
        },
        /// Tool result with images (for viz tools)
        ToolResult {
            tool: String,
            images: Vec<ImageData>,
        },
        Content {
            text: String,
        },
        Done {
            message: Message,
        },
        Error {
            error: String,
        },
    }

    /// Image data for tool results.
    #[derive(Debug, Clone, Serialize)]
    pub struct ImageData {
        pub data: String,
        pub mime_type: String,
    }

    /// Streaming tool executor that sends progress events.
    pub struct StreamingToolExecutor {
        server: Arc<crate::server::AnalyticsServer>,
        session: Arc<Session>,
        sender: tokio::sync::mpsc::Sender<ProgressEvent>,
        /// Database connection for persistence (optional)
        #[cfg(feature = "db")]
        db: Option<Arc<crate::db::DbConnection>>,
        /// Conversation ID for tool call persistence
        #[cfg(feature = "db")]
        conversation_id: Option<String>,
        /// Message ID for tool call association
        #[cfg(feature = "db")]
        message_id: Option<String>,
        /// Session ID for dataset metadata persistence
        #[cfg(feature = "db")]
        session_id: Option<String>,
    }

    impl StreamingToolExecutor {
        pub fn new(
            server: Arc<crate::server::AnalyticsServer>,
            session: Arc<Session>,
            sender: tokio::sync::mpsc::Sender<ProgressEvent>,
        ) -> Self {
            Self {
                server,
                session,
                sender,
                #[cfg(feature = "db")]
                db: None,
                #[cfg(feature = "db")]
                conversation_id: None,
                #[cfg(feature = "db")]
                message_id: None,
                #[cfg(feature = "db")]
                session_id: None,
            }
        }

        #[cfg(feature = "db")]
        pub fn with_persistence(
            mut self,
            db: Arc<crate::db::DbConnection>,
            conversation_id: String,
            message_id: String,
            session_id: String,
        ) -> Self {
            self.db = Some(db);
            self.conversation_id = Some(conversation_id);
            self.message_id = Some(message_id);
            self.session_id = Some(session_id);
            self
        }

        /// Set just db and session_id for dataset metadata tracking
        /// (doesn't require conversation_id/message_id)
        #[cfg(feature = "db")]
        pub fn with_dataset_tracking(
            mut self,
            db: Arc<crate::db::DbConnection>,
            session_id: String,
        ) -> Self {
            self.db = Some(db);
            self.session_id = Some(session_id);
            self
        }

        /// Capture dataset metadata after successful load_dataset or upload_dataset
        /// This is the internal method called by the executor.
        #[cfg(feature = "db")]
        async fn capture_dataset_metadata(
            db: Arc<crate::db::DbConnection>,
            session_id: &str,
            tool_name: &str,
            arguments: &serde_json::Value,
            result_text: &str,
        ) {
            Self::capture_dataset_metadata_static(
                db,
                session_id,
                tool_name,
                arguments,
                result_text,
            )
            .await;
        }

        /// Public static method for capturing dataset metadata.
        /// Can be called from outside the executor (e.g., direct tool calls).
        #[cfg(feature = "db")]
        pub async fn capture_dataset_metadata_static(
            db: Arc<crate::db::DbConnection>,
            session_id: &str,
            tool_name: &str,
            arguments: &serde_json::Value,
            result_text: &str,
        ) {
            use std::path::PathBuf;

            // DEBUG: Log entry point with full context
            tracing::info!(
                tool = tool_name,
                session = session_id,
                result_len = result_text.len(),
                "[METADATA_CAPTURE] Starting metadata capture"
            );

            // DEBUG: Log the full result text (truncated for very long results)
            let result_preview = if result_text.len() > 2000 {
                format!(
                    "{}...[truncated, total {} bytes]",
                    &result_text[..2000],
                    result_text.len()
                )
            } else {
                result_text.to_string()
            };
            tracing::info!(
                result_text = %result_preview,
                "[METADATA_CAPTURE] Full result text"
            );

            // Parse result to extract dataset info
            // Format: "Successfully loaded dataset 'name'\n\nDimensions: X rows x Y columns\n\nColumns:..."
            let first_line = result_text.lines().next();
            tracing::info!(
                first_line = ?first_line,
                "[METADATA_CAPTURE] First line of result"
            );

            let dataset_name = first_line.and_then(|line| {
                let start = line.find('\'');
                let end = line.rfind('\'');
                tracing::info!(
                    line = %line,
                    start_quote_pos = ?start,
                    end_quote_pos = ?end,
                    "[METADATA_CAPTURE] Parsing dataset name from first line"
                );
                match (start, end) {
                    (Some(s), Some(e)) if s < e => {
                        let name = line[s + 1..e].to_string();
                        tracing::info!(
                            extracted_name = %name,
                            "[METADATA_CAPTURE] Extracted dataset name"
                        );
                        Some(name)
                    }
                    _ => {
                        tracing::warn!("[METADATA_CAPTURE] Could not extract dataset name - quote positions invalid");
                        None
                    }
                }
            });

            // Find dimensions line
            let dimensions_line = result_text
                .lines()
                .find(|line| line.trim().starts_with("Dimensions:"));
            tracing::info!(
                dimensions_line = ?dimensions_line,
                "[METADATA_CAPTURE] Found dimensions line"
            );

            let dimensions = dimensions_line.and_then(|line| {
                // "Dimensions: X rows x Y columns"
                let parts: Vec<&str> = line.split_whitespace().collect();
                tracing::info!(
                    parts = ?parts,
                    parts_len = parts.len(),
                    "[METADATA_CAPTURE] Dimensions line parts"
                );
                if parts.len() >= 5 {
                    let rows = parts[1].parse::<i32>().ok();
                    let cols = parts[4].parse::<i32>().ok();
                    tracing::info!(
                        rows = ?rows,
                        cols = ?cols,
                        "[METADATA_CAPTURE] Parsed dimensions"
                    );
                    match (rows, cols) {
                        (Some(r), Some(c)) => Some((r, c)),
                        _ => {
                            tracing::warn!("[METADATA_CAPTURE] Could not parse row/col numbers");
                            None
                        }
                    }
                } else {
                    tracing::warn!(
                        parts_len = parts.len(),
                        "[METADATA_CAPTURE] Dimensions line has insufficient parts (need >= 5)"
                    );
                    None
                }
            });

            // Extract column names from "Columns:" section
            let columns_start = result_text.find("Columns:");
            tracing::info!(
                columns_start_pos = ?columns_start,
                "[METADATA_CAPTURE] Looking for 'Columns:' section"
            );

            let column_names: Vec<String> = if let Some(start) = columns_start {
                let columns_section = &result_text[start..];
                let lines_after_columns: Vec<&str> = columns_section.lines().collect();
                tracing::info!(
                    total_lines_in_section = lines_after_columns.len(),
                    first_5_lines = ?lines_after_columns.iter().take(5).collect::<Vec<_>>(),
                    "[METADATA_CAPTURE] Lines in Columns section"
                );

                let parsed_columns: Vec<String> = columns_section
                    .lines()
                    .skip(1) // Skip "Columns:" line
                    .enumerate()
                    .filter_map(|(idx, line)| {
                        let trimmed_line = line.trim();
                        let starts_with_dash = trimmed_line.starts_with('-');

                        if !starts_with_dash {
                            if !trimmed_line.is_empty() {
                                tracing::debug!(
                                    line_idx = idx,
                                    line = %line,
                                    trimmed = %trimmed_line,
                                    "[METADATA_CAPTURE] Skipping line (doesn't start with '-')"
                                );
                            }
                            return None;
                        }

                        // "  - column_name (dtype): X nulls (Y%)"
                        let after_dash = trimmed_line.trim_start_matches('-').trim();
                        let paren_pos = after_dash.find('(');

                        tracing::info!(
                            line_idx = idx,
                            original_line = %line,
                            after_dash = %after_dash,
                            paren_pos = ?paren_pos,
                            "[METADATA_CAPTURE] Parsing column line"
                        );

                        match paren_pos {
                            Some(pos) => {
                                let col_name = after_dash[..pos].trim().to_string();
                                tracing::info!(
                                    extracted_column = %col_name,
                                    "[METADATA_CAPTURE] Extracted column name"
                                );
                                Some(col_name)
                            }
                            None => {
                                tracing::warn!(
                                    line = %line,
                                    "[METADATA_CAPTURE] Column line has no '(' - cannot extract name"
                                );
                                None
                            }
                        }
                    })
                    .collect();

                tracing::info!(
                    num_columns_parsed = parsed_columns.len(),
                    columns = ?parsed_columns,
                    "[METADATA_CAPTURE] Finished parsing columns"
                );
                parsed_columns
            } else {
                tracing::warn!("[METADATA_CAPTURE] No 'Columns:' section found in result");
                Vec::new()
            };

            // Get source path and file metadata
            let source_path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let file_size_bytes = source_path
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as i64);

            // Determine source type
            let source_type = if tool_name == "upload_dataset" {
                arguments
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            } else if tool_name == "create_dataset" {
                "inline_csv".to_string()
            } else {
                source_path
                    .as_ref()
                    .map(PathBuf::from)
                    .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            };

            tracing::info!(
                dataset_name = ?dataset_name,
                dimensions = ?dimensions,
                num_columns = column_names.len(),
                column_names = ?column_names,
                source_type = %source_type,
                source_path = ?source_path,
                "[METADATA_CAPTURE] Final parsed metadata summary"
            );

            // Build and save metadata
            match (dataset_name.as_ref(), dimensions.as_ref()) {
                (Some(name), Some((rows, cols))) => {
                    let mut meta = crate::db::DatasetMeta::new(
                        session_id.to_string(),
                        name.clone(),
                        source_type.clone(),
                        *rows,
                        *cols,
                        column_names.clone(),
                    );
                    meta.source_path = source_path;
                    meta.file_size_bytes = file_size_bytes;

                    tracing::info!(
                        meta_id = %meta.id_string(),
                        meta_session_id = %meta.session_id,
                        meta_name = %meta.name,
                        meta_source_type = %meta.source_type,
                        meta_row_count = meta.row_count,
                        meta_column_count = meta.column_count,
                        meta_column_names = ?meta.column_names,
                        meta_column_names_len = meta.column_names.len(),
                        "[METADATA_CAPTURE] DatasetMeta object before save"
                    );

                    match db.save_dataset_meta(&meta).await {
                        Ok(saved_meta) => {
                            tracing::info!(
                                dataset = %name,
                                session_id = %session_id,
                                saved_id = %saved_meta.id_string(),
                                saved_column_names = ?saved_meta.column_names,
                                saved_column_names_len = saved_meta.column_names.len(),
                                "[METADATA_CAPTURE] Successfully saved dataset metadata"
                            );

                            // DEBUG: Immediately read back from DB to verify
                            match db.get_dataset_meta(session_id, name).await {
                                Ok(Some(read_back)) => {
                                    tracing::info!(
                                        read_back_id = %read_back.id_string(),
                                        read_back_name = %read_back.name,
                                        read_back_column_names = ?read_back.column_names,
                                        read_back_column_names_len = read_back.column_names.len(),
                                        "[METADATA_CAPTURE] Verified: read back from DB"
                                    );
                                }
                                Ok(None) => {
                                    tracing::error!(
                                        "[METADATA_CAPTURE] ERROR: Just saved but cannot read back - not found!"
                                    );
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "[METADATA_CAPTURE] ERROR: Failed to read back from DB"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "[METADATA_CAPTURE] Failed to save dataset metadata"
                            );
                        }
                    }
                }
                _ => {
                    tracing::warn!(
                        dataset_name = ?dataset_name,
                        dimensions = ?dimensions,
                        "[METADATA_CAPTURE] Could not parse dataset info from result - missing name or dimensions"
                    );
                }
            }
        }
    }

    #[async_trait::async_trait]
    impl ToolExecutor for StreamingToolExecutor {
        async fn execute(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Result<String, String> {
            tracing::warn!(
                ">>> StreamingToolExecutor::execute called for tool: {}",
                name
            );

            // Send tool start event with arguments
            let _ = self
                .sender
                .send(ProgressEvent::ToolStart {
                    tool: name.to_string(),
                    arguments: arguments.clone(),
                })
                .await;

            // Create tool call record in DB if persistence is enabled
            #[cfg(feature = "db")]
            let tool_call_id = if let (Some(db), Some(conv_id), Some(msg_id)) =
                (&self.db, &self.conversation_id, &self.message_id)
            {
                let tool_call = crate::db::ToolCall::new(
                    msg_id.clone(),
                    conv_id.clone(),
                    name.to_string(),
                    serde_json::to_string(&arguments).unwrap_or_default(),
                );
                let tc_id = tool_call.id_string();

                // Mark as running
                let mut running_tool_call = tool_call;
                running_tool_call.status = crate::db::ToolCallStatus::Running;

                match db.create_tool_call(&running_tool_call).await {
                    Ok(_) => {
                        tracing::debug!(tool_call_id = %tc_id, "Created tool call record");
                        Some(tc_id)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create tool call record: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            // Clone arguments for potential metadata capture later
            #[cfg(feature = "db")]
            let args_for_metadata = arguments.clone();

            tracing::info!(tool = %name, "Executing tool");
            let start = std::time::Instant::now();
            let result = self
                .server
                .call_tool_with_session(name, arguments, &self.session)
                .await;
            let elapsed = start.elapsed();
            let duration_ms = elapsed.as_millis() as i32;

            // Extract result text for the ToolEnd event
            let result_text = match &result {
                Ok(r) => {
                    let text: String = r
                        .content
                        .iter()
                        .filter_map(|item| match item {
                            ContentItem::Text { text } => Some(text.clone()),
                            ContentItem::Image { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    // Truncate for the event (keep it reasonable for SSE)
                    if text.len() > 2000 {
                        Some(format!("{}...", &text[..2000]))
                    } else {
                        Some(text)
                    }
                }
                Err(e) => Some(format!("Error: {}", e)),
            };

            // Send tool end event with result
            let _ = self
                .sender
                .send(ProgressEvent::ToolEnd {
                    tool: name.to_string(),
                    elapsed_ms: elapsed.as_millis() as u64,
                    result: result_text,
                })
                .await;

            tracing::info!(tool = %name, elapsed_ms = %elapsed.as_millis(), "Tool execution completed");

            match result {
                Ok(result) => {
                    // Extract result text for DB operations
                    #[cfg(feature = "db")]
                    let result_str = result
                        .content
                        .iter()
                        .filter_map(|item| match item {
                            ContentItem::Text { text } => Some(text.clone()),
                            ContentItem::Image { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    // Update tool call record with success
                    #[cfg(feature = "db")]
                    if let (Some(db), Some(tc_id)) = (&self.db, &tool_call_id) {
                        // Truncate result if too long (max 10KB)
                        let truncated_result = if result_str.len() > 10240 {
                            format!("{}...[truncated]", &result_str[..10240])
                        } else {
                            result_str.clone()
                        };
                        if let Err(e) = db
                            .complete_tool_call(tc_id, &truncated_result, duration_ms)
                            .await
                        {
                            tracing::warn!("Failed to update tool call record: {}", e);
                        }
                    }

                    // Capture dataset metadata for load_dataset, upload_dataset, and create_dataset
                    // This is separate from tool call tracking - it only needs db and session_id
                    #[cfg(feature = "db")]
                    if let Some(db) = &self.db {
                        if (name == "load_dataset"
                            || name == "upload_dataset"
                            || name == "create_dataset")
                            && result.error.is_none()
                        {
                            if let Some(session_id) = &self.session_id {
                                Self::capture_dataset_metadata(
                                    db.clone(),
                                    session_id,
                                    name,
                                    &args_for_metadata,
                                    &result_str,
                                )
                                .await;
                            }
                        }
                    }

                    // Extract images and send them as a separate event
                    let images: Vec<ImageData> = result
                        .content
                        .iter()
                        .filter_map(|item| match item {
                            ContentItem::Image { data, mime_type } => Some(ImageData {
                                data: data.clone(),
                                mime_type: mime_type.clone(),
                            }),
                            _ => None,
                        })
                        .collect();

                    // If there are images, send them to the frontend
                    if !images.is_empty() {
                        let _ = self
                            .sender
                            .send(ProgressEvent::ToolResult {
                                tool: name.to_string(),
                                images,
                            })
                            .await;
                    }

                    // Return text content only (without base64) for the LLM
                    let content = result
                        .content
                        .iter()
                        .map(|item| match item {
                            ContentItem::Text { text } => text.clone(),
                            ContentItem::Image { .. } => {
                                "[Image output - displayed in UI]".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(content)
                }
                Err(e) => {
                    // Update tool call record with error
                    #[cfg(feature = "db")]
                    if let (Some(db), Some(tc_id)) = (&self.db, &tool_call_id) {
                        if let Err(db_err) = db.fail_tool_call(tc_id, &e, duration_ms).await {
                            tracing::warn!("Failed to update tool call record: {}", db_err);
                        }
                    }
                    Err(e)
                }
            }
        }
    }

    /// Streaming LLM chat endpoint using Server-Sent Events.
    pub async fn llm_chat_stream(
        State(state): State<AppState>,
        Json(request): Json<LlmChatRequest>,
    ) -> axum::response::sse::Sse<
        impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
    > {
        use crate::llm::StreamChunk;
        use tokio::sync::mpsc;

        let (tx, mut rx) = mpsc::channel::<ProgressEvent>(100);

        // Spawn the chat task
        let state_clone = state.clone();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            // Send initial status
            let _ = tx_clone
                .send(ProgressEvent::Status {
                    message: "Starting analysis...".to_string(),
                })
                .await;

            tracing::info!(interpret = %request.interpret, conversation_id = ?request.conversation_id, "LLM chat request received");

            // Get the session
            let session = match state_clone
                .session_manager
                .get_session(&request.session_id)
                .await
            {
                Ok(s) => s,
                Err(crate::session::SessionError::NotFound) => {
                    let _ = tx_clone
                        .send(ProgressEvent::Error {
                            error: "Session not found".to_string(),
                        })
                        .await;
                    return;
                }
                Err(e) => {
                    let _ = tx_clone
                        .send(ProgressEvent::Error {
                            error: e.to_string(),
                        })
                        .await;
                    return;
                }
            };

            // Get database connection for persistence/tracking
            #[cfg(feature = "db")]
            let db_connection = state_clone
                .persistent_manager
                .as_ref()
                .map(|pm| pm.db().clone());

            // Normalize conversation_id: treat empty string as None
            #[cfg(feature = "db")]
            let effective_conversation_id = request
                .conversation_id
                .as_ref()
                .filter(|id| !id.is_empty())
                .cloned();

            // Create placeholder assistant message if conversation persistence is enabled
            #[cfg(feature = "db")]
            let message_id = if let (Some(pm), Some(conv_id)) =
                (&state_clone.persistent_manager, &effective_conversation_id)
            {
                // Create placeholder message
                match pm.db().add_message(conv_id, "assistant", "").await {
                    Ok(msg) => {
                        let msg_id = msg.id_string();
                        tracing::debug!(message_id = %msg_id, "Created placeholder assistant message");
                        Some(msg_id)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create placeholder message: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            // Build enhanced dataset context from currently loaded datasets
            // This includes data types, sample values, and basic statistics
            let dataset_context = {
                let datasets = session.datasets.read().await;
                build_enhanced_dataset_context(&datasets)
            };

            // Create provider and streaming tool executor
            let provider = create_provider(request.provider);
            let mut tool_executor =
                StreamingToolExecutor::new(state_clone.server.clone(), session, tx_clone.clone());

            // Track whether we've set up the executor for persistence
            #[cfg(feature = "db")]
            let mut persistence_configured = false;

            // Add full persistence context if conversation tracking is available
            #[cfg(feature = "db")]
            if let (Some(db), Some(conv_id), Some(msg_id)) =
                (&db_connection, &effective_conversation_id, &message_id)
            {
                tool_executor = tool_executor.with_persistence(
                    db.clone(),
                    conv_id.clone(),
                    msg_id.clone(),
                    request.session_id.clone(),
                );
                persistence_configured = true;
            }

            // Fall back to dataset metadata tracking if full persistence wasn't configured
            // This ensures we always capture dataset metadata when db is available
            #[cfg(feature = "db")]
            if !persistence_configured {
                if let Some(db) = &db_connection {
                    tool_executor =
                        tool_executor.with_dataset_tracking(db.clone(), request.session_id.clone());
                }
            }

            let tools = get_mcp_tool_definitions();

            let _ = tx_clone
                .send(ProgressEvent::Status {
                    message: format!("Connecting to {} LLM...", provider.provider_type()),
                })
                .await;

            tracing::info!(
                has_datasets = dataset_context.is_some(),
                "Building system prompt with dataset context"
            );

            // Auto-retrieve history from database if:
            // 1. conversation_id is provided
            // 2. history is empty
            // 3. retrieve_history flag is true (default)
            #[cfg(feature = "db")]
            let retrieved_history: Vec<Message> = if request.retrieve_history
                && request.history.is_empty()
                && effective_conversation_id.is_some()
            {
                if let (Some(db), Some(conv_id)) = (&db_connection, &effective_conversation_id) {
                    match db.get_messages(conv_id).await {
                        Ok(db_messages) => {
                            tracing::info!(
                                conversation_id = %conv_id,
                                message_count = db_messages.len(),
                                "Retrieved conversation history from database"
                            );
                            // Convert DB messages to LLM messages, including tool calls
                            let mut history = Vec::new();
                            for db_msg in db_messages {
                                // Get tool calls for this message
                                let tool_calls_for_msg =
                                    db.get_tool_calls_for_message(&db_msg.id_string()).await;

                                match db_msg.role {
                                    crate::db::MessageRole::User => {
                                        if !db_msg.content.is_empty() {
                                            history.push(Message::user(db_msg.content.clone()));
                                        }
                                    }
                                    crate::db::MessageRole::Assistant => {
                                        // Check if this message has completed tool calls
                                        if let Ok(tool_calls) = &tool_calls_for_msg {
                                            let completed_calls: Vec<_> = tool_calls
                                                .iter()
                                                .filter(|tc| {
                                                    tc.status == crate::db::ToolCallStatus::Success
                                                })
                                                .collect();

                                            if !completed_calls.is_empty() {
                                                // Create tool calls for the assistant message
                                                let llm_tool_calls: Vec<crate::llm::ToolCall> =
                                                    completed_calls
                                                        .iter()
                                                        .map(|tc| crate::llm::ToolCall {
                                                            id: tc.id_string(),
                                                            name: tc.tool_name.clone(),
                                                            arguments: serde_json::from_str(
                                                                &tc.arguments,
                                                            )
                                                            .unwrap_or(serde_json::Value::Null),
                                                        })
                                                        .collect();

                                                // Assistant message with tool calls
                                                history.push(Message::assistant_with_tools(
                                                    db_msg.content.clone(),
                                                    llm_tool_calls,
                                                ));

                                                // Tool result message
                                                let tool_results: Vec<crate::llm::ToolResult> =
                                                    completed_calls
                                                        .iter()
                                                        .map(|tc| {
                                                            let result_content = tc
                                                                .result
                                                                .as_deref()
                                                                .unwrap_or("");
                                                            // Truncate long results
                                                            let truncated =
                                                                if result_content.len() > 2000 {
                                                                    format!(
                                                                        "{}...\n[Truncated]",
                                                                        &result_content[..2000]
                                                                    )
                                                                } else {
                                                                    result_content.to_string()
                                                                };
                                                            crate::llm::ToolResult {
                                                                tool_call_id: tc.id_string(),
                                                                content: truncated,
                                                                is_error: false,
                                                            }
                                                        })
                                                        .collect();

                                                history.push(Message::tool_result(tool_results));
                                            } else if !db_msg.content.is_empty() {
                                                // No tool calls, just content
                                                history
                                                    .push(Message::assistant(db_msg.content.clone()));
                                            }
                                        } else if !db_msg.content.is_empty() {
                                            history.push(Message::assistant(db_msg.content.clone()));
                                        }
                                    }
                                    crate::db::MessageRole::System => {
                                        // Skip system messages in history retrieval
                                    }
                                }
                            }
                            history
                        }
                        Err(e) => {
                            tracing::warn!(
                                conversation_id = %conv_id,
                                error = %e,
                                "Failed to retrieve conversation history, using empty history"
                            );
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            #[cfg(not(feature = "db"))]
            let retrieved_history: Vec<Message> = Vec::new();

            // Determine which history to use: client-provided takes precedence
            let history_to_use = if request.history.is_empty() {
                retrieved_history
            } else {
                request.history.clone()
            };

            // Build message history with dataset context
            let system_prompt = get_system_prompt_with_context(dataset_context.as_deref());
            let mut messages = vec![Message::system(system_prompt)];
            messages.extend(history_to_use);
            messages.push(Message::user(request.message.clone()));

            // Create streaming callback that forwards content chunks
            let tx_for_callback = tx_clone.clone();
            let stream_callback: Box<dyn Fn(StreamChunk) + Send + Sync> = Box::new(move |chunk| {
                match chunk {
                    StreamChunk::Text { content } => {
                        // Use try_send for non-blocking send from sync callback
                        let _ = tx_for_callback.try_send(ProgressEvent::Content { text: content });
                    }
                    StreamChunk::Done => {
                        // Done is handled separately after chat_stream returns
                    }
                    StreamChunk::Error { message } => {
                        let _ = tx_for_callback.try_send(ProgressEvent::Error { error: message });
                    }
                    StreamChunk::ToolCall { .. } | StreamChunk::ToolResult { .. } => {
                        // Tool events are handled by StreamingToolExecutor
                    }
                }
            });

            // Execute streaming chat
            match provider
                .chat_stream(
                    &messages,
                    &tools,
                    &tool_executor,
                    request.interpret,
                    stream_callback,
                )
                .await
            {
                Ok(response) => {
                    // Update message content in database if persistence is enabled
                    #[cfg(feature = "db")]
                    if let (Some(db), Some(msg_id)) = (&db_connection, &message_id) {
                        if let Err(e) = db.update_message_content(msg_id, &response.content).await {
                            tracing::warn!("Failed to update message content: {}", e);
                        } else {
                            tracing::debug!(message_id = %msg_id, "Updated assistant message content");
                        }
                    }

                    // Debug: check if content has newlines
                    let has_newlines = response.content.contains('\n');
                    let content_preview: String = response.content.chars().take(300).collect();
                    tracing::info!(
                        has_newlines = has_newlines,
                        content_preview = %content_preview,
                        "Sending Done event with message content"
                    );
                    let _ = tx_clone
                        .send(ProgressEvent::Done { message: response })
                        .await;
                }
                Err(e) => {
                    // Update message with error content if persistence is enabled
                    #[cfg(feature = "db")]
                    if let (Some(db), Some(msg_id)) = (&db_connection, &message_id) {
                        let error_content = format!("[Error: {}]", e);
                        let _ = db.update_message_content(msg_id, &error_content).await;
                    }

                    let _ = tx_clone
                        .send(ProgressEvent::Error {
                            error: e.to_string(),
                        })
                        .await;
                }
            }
        });

        // Create SSE stream from receiver
        let stream = async_stream::stream! {
            while let Some(event) = rx.recv().await {
                let data = serde_json::to_string(&event).unwrap_or_default();
                yield Ok(axum::response::sse::Event::default().data(data));

                // Stop after Done or Error
                if matches!(event, ProgressEvent::Done { .. } | ProgressEvent::Error { .. }) {
                    break;
                }
            }
        };

        axum::response::sse::Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
    }
}

#[cfg(feature = "llm")]
use llm_handlers::{llm_chat, llm_chat_stream, llm_env_keys, llm_generate_title, llm_list_models};
