//! HTTP REST API transport for p2a-mcp.
//!
//! Provides a REST API that mirrors the MCP tool interface,
//! allowing web clients to interact with the analytics server.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
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
        .route("/api/tools/{name}", post(call_tool));

    // Add WebSocket route if feature is enabled
    #[cfg(feature = "websocket")]
    let router = router.route("/ws", get(super::websocket::ws_handler));

    // Add LLM routes if feature is enabled
    #[cfg(feature = "llm")]
    let router = router
        .route("/api/llm/chat", post(llm_chat))
        .route("/api/llm/models", get(llm_list_models));

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
        ProviderConfig, ProviderType, ToolExecutor,
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
            match self
                .server
                .call_tool_with_session(name, arguments, &self.session)
                .await
            {
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
    fn create_provider(config: Option<ProviderConfig>) -> Box<dyn LlmProvider> {
        let config = config.unwrap_or_default();
        match config.provider_type {
            ProviderType::Ollama => Box::new(OllamaProvider::new(config)),
            // TODO: Add Anthropic and OpenAI providers
            _ => Box::new(OllamaProvider::new(config)),
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

        // Build message history
        let mut messages = vec![Message::system(get_system_prompt())];
        messages.extend(request.history);
        messages.push(Message::user(request.message));

        // Execute chat
        match provider
            .chat(&messages, &tools, &tool_executor)
            .await
        {
            Ok(response) => (
                StatusCode::OK,
                Json(ApiResponse::success(LlmChatResponse { message: response })),
            ),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            ),
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
}

#[cfg(feature = "llm")]
use llm_handlers::{llm_chat, llm_list_models};
