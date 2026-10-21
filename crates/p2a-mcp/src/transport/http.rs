//! HTTP REST API transport for p2a-mcp.
//!
//! Provides a REST API that mirrors the MCP tool interface,
//! allowing web clients to interact with the analytics server.
//!
//! This file holds the shared state and response types, router composition
//! and startup. The route handlers live in `routes_sessions`, `routes_tools`
//! and `llm_routes`; CORS, authentication and the body limit in `middleware`.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::Serialize;

use super::middleware;
use super::routes_sessions::{
    create_session, delete_session, get_session, list_files, list_sessions,
};
use super::routes_tools::{call_tool, list_tools};
use crate::audit::AuditLogger;
use crate::config::{DeploymentMode, ServerConfig};
use crate::exec::{ExecConfig, StuckWatchdog, ToolGovernor};
use crate::server::AnalyticsServer;
use crate::session::SessionManager;
use crate::transport::{TransportError, TransportResult};

pub use super::routes_sessions::{
    CreateSessionRequest, CreateSessionResponse, FileEntry, ListFilesQuery, ListFilesResponse,
};
pub use super::routes_tools::CallToolRequest;

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
    /// Admission control and off-runtime execution for every tool call
    pub governor: Arc<ToolGovernor>,
}

/// Build the shared state and the router for `config`, with the governor
/// settings resolved from it (`ExecConfig::from_config`).
pub async fn build_app(config: &ServerConfig) -> Result<(Router, AppState), TransportError> {
    build_app_with(config, ExecConfig::from_config(config)).await
}

/// Build the shared state and the router for `config` with explicit governor
/// settings.
///
/// This is the one place that creates the session manager (with its
/// 10-minute cleanup task), the analytics server, the audit logger, the tool
/// governor and, with the `db` feature, the persistent session manager. The
/// HTTP transport, the embedded desktop server and the integration tests all
/// start from here. Process-wide policies are installed by [`serve`], not
/// here, so tests can build several apps.
pub async fn build_app_with(
    config: &ServerConfig,
    exec: ExecConfig,
) -> Result<(Router, AppState), TransportError> {
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

    let governor = Arc::new(ToolGovernor::new(exec, session_manager.clone()));

    let state = AppState {
        server,
        session_manager,
        #[cfg(feature = "db")]
        persistent_manager: init_persistence(config).await,
        audit_logger,
        governor,
    };

    Ok((create_router(state.clone(), config), state))
}

/// Create the persistent session manager, or `None` (with a log line) when
/// the database cannot be opened.
#[cfg(feature = "db")]
async fn init_persistence(
    config: &ServerConfig,
) -> Option<Arc<crate::persistent_session::PersistentSessionManager>> {
    let db_path = config.http.db_path.as_deref();
    tracing::info!("Initializing database with path: {:?}", db_path);
    match crate::persistent_session::PersistentSessionManager::new(
        db_path,
        config.session.max_sessions,
        config.session.ttl_minutes,
    )
    .await
    {
        Ok(manager) => {
            tracing::info!("Database persistence enabled - conversation routes will be available");
            Some(Arc::new(manager))
        }
        Err(e) => {
            tracing::warn!(
                "Failed to initialize database, running WITHOUT persistence (conversations disabled): {}",
                e
            );
            None
        }
    }
}

/// Validate `config` and bind its address. Every HTTP start path (the
/// binary, the embedded desktop server) goes through here, so the bind guard
/// (`config::check_bind_policy`, part of `ServerConfig::validate`) cannot be
/// skipped.
pub async fn bind(config: &ServerConfig) -> TransportResult<tokio::net::TcpListener> {
    config
        .validate()
        .map_err(|e| TransportError::Config(e.to_string()))?;
    let addr = config.http.addr;
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| TransportError::Http(format!("Failed to bind to {}: {}", addr, e)))
}

/// Serve the app on `listener` until `shutdown` resolves.
///
/// Validates the configuration again, so no caller can skip the guard, then
/// builds the app, installs the process-wide policies (allocation ceiling,
/// session directories) and serves it with graceful shutdown.
pub async fn serve(
    listener: tokio::net::TcpListener,
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> TransportResult<()> {
    config
        .validate()
        .map_err(|e| TransportError::Config(e.to_string()))?;
    let (app, state) = build_app(&config).await?;
    let exec = state.governor.config().clone();
    crate::exec::install_process_policies(&exec);
    log_startup(&config, &exec, &listener);
    if let Some(limit) = exec.stuck_exit {
        spawn_stuck_watchdog(state.governor.clone(), limit);
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| TransportError::Http(format!("HTTP server error: {}", e)))
}

/// Start the HTTP transport: [`bind`], then [`serve`] until the process ends.
pub async fn start_http_transport(config: &ServerConfig) -> TransportResult<()> {
    let listener = bind(config).await?;
    serve(listener, config.clone(), std::future::pending()).await
}

fn log_startup(config: &ServerConfig, exec: &ExecConfig, listener: &tokio::net::TcpListener) {
    let addr = listener.local_addr().unwrap_or(config.http.addr);
    let mode = config.deployment_mode();
    match mode {
        DeploymentMode::Authenticated => {
            tracing::info!("HTTP access-token authentication is ENABLED for /api routes");
        }
        DeploymentMode::OpenPublic => {
            tracing::warn!(
                "HTTP running OPEN and UNAUTHENTICATED on {addr} (open public mode). \
                 Session-list and file-browser endpoints are disabled and the hosted resource \
                 profile applies; isolation relies on unguessable session IDs. Put an edge \
                 rate-limiter in front for abuse/DoS control."
            );
        }
        DeploymentMode::Local => {
            tracing::warn!(
                "HTTP access-token authentication is DISABLED (local mode: loopback bind); \
                 set P2A_ACCESS_TOKEN before exposing this server"
            );
        }
    }
    if (exec.session_data_bytes.is_some() || exec.total_data_bytes.is_some())
        && exec.session_concurrency > 1
    {
        tracing::warn!(
            "dataset quotas are configured with --session-concurrency > 1; a refused call is \
             undone by restoring the session's datasets, which assumes one call per session"
        );
    }
    tracing::info!(
        "Starting HTTP transport on {addr} (mode {mode:?}, resource profile {:?}, \
         tool concurrency {}, deadline {:?})",
        config.resource_profile(),
        exec.concurrency,
        exec.deadline
    );
}

/// Exit with code 70 when every tool permit stays held by overdue work for
/// `limit`, so the platform restarts the machine instead of leaving it
/// wedged. Set by the hosted profile (300 s) or `--stuck-exit-secs`.
fn spawn_stuck_watchdog(governor: Arc<ToolGovernor>, limit: Duration) {
    tokio::spawn(async move {
        let mut watchdog = StuckWatchdog::new(limit);
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        loop {
            tick.tick().await;
            if watchdog.observe(governor.health().healthy, Instant::now()) {
                tracing::error!(
                    "tool workers stuck past their deadline for {limit:?}; exiting so the \
                     platform restarts the machine"
                );
                std::process::exit(70);
            }
        }
    });
}

/// Create the axum router with all routes.
///
/// With the `db` feature, the conversation routes are nested when
/// `state.persistent_manager` is set.
pub fn create_router(state: AppState, config: &ServerConfig) -> Router {
    #[cfg(feature = "db")]
    let persistent_manager = state.persistent_manager.clone();

    let router = create_base_router(state, config);

    // Add conversation routes if persistent manager is available
    #[cfg(feature = "db")]
    let router = if let Some(manager) = persistent_manager {
        tracing::info!("Adding conversation routes to /api/*");
        let conv_state = super::conversation::ConversationState {
            session_manager: manager,
        };
        router.nest("/api", super::conversation::conversation_routes(conv_state))
    } else {
        tracing::warn!("Conversation routes NOT added (no persistent manager)");
        router
    };

    // Apply global middleware LAST, so it also covers the nested conversation
    // routes (e.g. /api/sessions/{id}/datasets). Applying CORS inside
    // create_base_router would skip routes nested afterwards.
    middleware::apply_global_middleware(router, config)
}

/// Create the base router with core routes.
fn create_base_router(state: AppState, config: &ServerConfig) -> Router {
    // In the intentionally-open public mode (non-loopback bind, no access
    // token), the two enumeration surfaces are omitted entirely: `GET
    // /api/sessions` (which would list every active session ID and defeat the
    // unguessable-session-ID isolation) and `GET /api/files` (a filesystem
    // browser). The web frontend uses neither. They remain available for
    // loopback/desktop use and for authenticated (token-gated) deployments.
    let hide_enumeration = config.http.is_open_unauthenticated();

    let mut router = Router::new()
        // Health check
        .route("/health", get(health_check))
        // Session management
        .route("/api/sessions", post(create_session))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}", delete(delete_session))
        // Tool discovery
        .route("/api/tools", get(list_tools))
        // Tool execution
        .route("/api/tools/{name}", post(call_tool));

    if !hide_enumeration {
        router = router
            .route("/api/sessions", get(list_sessions))
            // File browser
            .route("/api/files", get(list_files));
    }

    // Add WebSocket route if feature is enabled
    #[cfg(feature = "websocket")]
    let router = router.route("/ws", get(super::websocket::ws_handler));

    // Add LLM routes if feature is enabled
    #[cfg(feature = "llm")]
    let router = {
        use super::llm_routes::{
            llm_chat, llm_chat_stream, llm_env_keys, llm_generate_title, llm_list_models,
        };
        router
            .route("/api/llm/chat", post(llm_chat))
            .route("/api/llm/chat/stream", post(llm_chat_stream))
            .route("/api/llm/models", get(llm_list_models))
            .route("/api/llm/env-keys", get(llm_env_keys))
            .route("/api/llm/generate-title", post(llm_generate_title))
    };

    // NOTE: global middleware (CORS, body limit, trace) is applied in
    // `apply_global_middleware` after all routes — including nested conversation
    // routes — are composed in `create_router`.
    router.with_state(state)
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
    /// Machine-readable error code (for example `server_busy`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
    /// Structured error details (for example `{"parameter": "bootstrap"}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            code: None,
            details: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
            code: None,
            details: None,
        }
    }

    /// An error with a machine-readable code and optional details.
    pub fn error_with_code(
        message: impl Into<String>,
        code: &'static str,
        details: Option<serde_json::Value>,
    ) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
            code: Some(code),
            details,
        }
    }
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

/// Health check endpoint: 200 with `"status": "ok"`, or 503 with
/// `"status": "degraded"` while every tool permit is held by work past its
/// deadline. The `governor` object reports in-flight and overdue work.
async fn health_check(State(state): State<AppState>) -> Response {
    let session_count = state.session_manager.session_count().await;
    let governor = state.governor.health();
    let (status, label) = if governor.healthy {
        (StatusCode::OK, "ok")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "degraded")
    };
    (
        status,
        Json(serde_json::json!({
            "status": label,
            "server": "p2a-mcp",
            "version": env!("CARGO_PKG_VERSION"),
            "active_sessions": session_count,
            "governor": governor,
        })),
    )
        .into_response()
}
