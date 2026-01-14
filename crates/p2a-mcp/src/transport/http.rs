//! HTTP REST API transport for p2a-mcp.
//!
//! Provides a REST API that mirrors the MCP tool interface,
//! allowing web clients to interact with the analytics server.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::config::ServerConfig;
use crate::server::AnalyticsServer;
use crate::session::{SessionError, SessionInfo, SessionManager};
use crate::transport::TransportResult;

/// Shared application state for HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    /// Analytics server for tool execution
    pub server: Arc<AnalyticsServer>,
    /// Session manager for multi-user support
    pub session_manager: Arc<SessionManager>,
}

/// Start the HTTP transport.
pub async fn start_http_transport(config: &ServerConfig) -> TransportResult<()> {
    let session_manager = Arc::new(SessionManager::new(config.session.clone()));

    // Start background cleanup task (every 10 minutes)
    session_manager.clone().start_cleanup_task(10);

    let server = Arc::new(AnalyticsServer::new());

    let state = AppState {
        server,
        session_manager,
    };

    let app = create_router(state, config);

    let addr = config.http.addr;
    tracing::info!("Starting HTTP transport on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        crate::transport::TransportError::Http(format!("Failed to bind to {}: {}", addr, e))
    })?;

    axum::serve(listener, app).await.map_err(|e| {
        crate::transport::TransportError::Http(format!("HTTP server error: {}", e))
    })?;

    Ok(())
}

/// Create the axum router with all routes.
fn create_router(state: AppState, config: &ServerConfig) -> Router {
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
        .route("/api/llm/env-keys", get(llm_env_keys));

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
    match state.session_manager.create_session(request.user_id).await {
        Ok(session_id) => (
            StatusCode::CREATED,
            Json(ApiResponse::success(CreateSessionResponse { session_id })),
        ),
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
async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
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
                        let ext = name.split('.').last().unwrap_or("").to_lowercase();
                        if !["csv", "parquet", "json", "xlsx", "xls", "dta", "sas7bdat"].contains(&ext.as_str()) {
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
            entries.sort_by(|a, b| {
                match (a.is_dir, b.is_dir) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                }
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

    // Execute the tool with session context
    match state.server.call_tool_with_session(&name, request.arguments, &session).await {
        Ok(result) => (StatusCode::OK, Json(ApiResponse::success(result))),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::error(e)),
        ),
    }
}

// =============================================================================
// LLM Endpoints (feature-gated)
// =============================================================================

#[cfg(feature = "llm")]
mod llm_handlers {
    use super::*;
    use crate::llm::{
        get_mcp_tool_definitions, get_system_prompt, LlmProvider, Message, OllamaProvider,
        OpenAIProvider, ProviderConfig, ProviderType, ToolExecutor,
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
        if config.api_key.is_none() || config.api_key.as_ref().map(|k| k.is_empty()).unwrap_or(false) {
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
            // TODO: Add Anthropic provider
            ProviderType::Anthropic => Box::new(OllamaProvider::new(config)),
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
        let tool_executor = SessionToolExecutor::new(state.server.clone(), session);
        let tools = get_mcp_tool_definitions();

        tracing::info!(provider = %provider.provider_type(), num_tools = %tools.len(), "Starting LLM chat");

        // Build message history
        let mut messages = vec![Message::system(get_system_prompt())];
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
    pub async fn llm_list_models(
        State(_state): State<AppState>,
    ) -> impl IntoResponse {
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
                Json(ApiResponse::error(format!("LLM provider not available: {}", e))),
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

    /// Progress event for streaming chat.
    #[derive(Debug, Clone, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ProgressEvent {
        Status { message: String },
        ToolStart { tool: String },
        ToolEnd { tool: String, elapsed_ms: u64 },
        /// Tool result with images (for viz tools)
        ToolResult { tool: String, images: Vec<ImageData> },
        Content { text: String },
        Done { message: Message },
        Error { error: String },
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
    }

    impl StreamingToolExecutor {
        pub fn new(
            server: Arc<crate::server::AnalyticsServer>,
            session: Arc<Session>,
            sender: tokio::sync::mpsc::Sender<ProgressEvent>,
        ) -> Self {
            Self { server, session, sender }
        }
    }

    #[async_trait::async_trait]
    impl ToolExecutor for StreamingToolExecutor {
        async fn execute(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Result<String, String> {
            // Send tool start event
            let _ = self.sender.send(ProgressEvent::ToolStart {
                tool: name.to_string()
            }).await;

            tracing::info!(tool = %name, "Executing tool");
            let start = std::time::Instant::now();
            let result = self
                .server
                .call_tool_with_session(name, arguments, &self.session)
                .await;
            let elapsed = start.elapsed();

            // Send tool end event
            let _ = self.sender.send(ProgressEvent::ToolEnd {
                tool: name.to_string(),
                elapsed_ms: elapsed.as_millis() as u64,
            }).await;

            tracing::info!(tool = %name, elapsed_ms = %elapsed.as_millis(), "Tool execution completed");
            match result {
                Ok(result) => {
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
                        let _ = self.sender.send(ProgressEvent::ToolResult {
                            tool: name.to_string(),
                            images,
                        }).await;
                    }

                    // Return text content only (without base64) for the LLM
                    let content = result
                        .content
                        .iter()
                        .map(|item| match item {
                            ContentItem::Text { text } => text.clone(),
                            ContentItem::Image { .. } => "[Image output - displayed in UI]".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(content)
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Streaming LLM chat endpoint using Server-Sent Events.
    pub async fn llm_chat_stream(
        State(state): State<AppState>,
        Json(request): Json<LlmChatRequest>,
    ) -> axum::response::sse::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
        use crate::llm::StreamChunk;
        use tokio::sync::mpsc;

        let (tx, mut rx) = mpsc::channel::<ProgressEvent>(100);

        // Spawn the chat task
        let state_clone = state.clone();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            // Send initial status
            let _ = tx_clone.send(ProgressEvent::Status {
                message: "Starting analysis...".to_string()
            }).await;

            tracing::info!(interpret = %request.interpret, "LLM chat request received");

            // Get the session
            let session = match state_clone.session_manager.get_session(&request.session_id).await {
                Ok(s) => s,
                Err(crate::session::SessionError::NotFound) => {
                    let _ = tx_clone.send(ProgressEvent::Error {
                        error: "Session not found".to_string()
                    }).await;
                    return;
                }
                Err(e) => {
                    let _ = tx_clone.send(ProgressEvent::Error {
                        error: e.to_string()
                    }).await;
                    return;
                }
            };

            // Create provider and streaming tool executor
            let provider = create_provider(request.provider);
            let tool_executor = StreamingToolExecutor::new(
                state_clone.server.clone(),
                session,
                tx_clone.clone(),
            );
            let tools = get_mcp_tool_definitions();

            let _ = tx_clone.send(ProgressEvent::Status {
                message: format!("Connecting to {} LLM...", provider.provider_type())
            }).await;

            // Build message history
            let mut messages = vec![Message::system(get_system_prompt())];
            messages.extend(request.history);
            messages.push(Message::user(request.message));

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
            match provider.chat_stream(&messages, &tools, &tool_executor, request.interpret, stream_callback).await {
                Ok(response) => {
                    // Debug: check if content has newlines
                    let has_newlines = response.content.contains('\n');
                    let content_preview: String = response.content.chars().take(300).collect();
                    tracing::info!(
                        has_newlines = has_newlines,
                        content_preview = %content_preview,
                        "Sending Done event with message content"
                    );
                    let _ = tx_clone.send(ProgressEvent::Done { message: response }).await;
                }
                Err(e) => {
                    let _ = tx_clone.send(ProgressEvent::Error {
                        error: e.to_string()
                    }).await;
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

        axum::response::sse::Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
    }
}

#[cfg(feature = "llm")]
use llm_handlers::{llm_chat, llm_chat_stream, llm_env_keys, llm_list_models};
