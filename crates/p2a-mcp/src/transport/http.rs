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
use crate::config::{DeploymentMode, ServerConfig, SessionConfig, TransportType};
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
/// settings resolved from it (`ExecConfig::from_config`). The configuration
/// is read as an HTTP one whatever its `transport` field says
/// (`ServerConfig::as_http`).
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
/// here, so tests can build several apps. Like [`build_app`], it reads
/// `config` as an HTTP configuration whatever its `transport` field says.
pub async fn build_app_with(
    config: &ServerConfig,
    exec: ExecConfig,
) -> Result<(Router, AppState), TransportError> {
    let config = &config.as_http();
    // Resolved before anything is spawned, so a bad override refuses the
    // start without leaving tasks behind.
    #[cfg(feature = "db")]
    let limits = resolve_limits(config)?;
    #[cfg(feature = "db")]
    if let Some(db_path) = config.http.db_path.as_deref() {
        enforce_db_path_policy(
            config.deployment_mode(),
            std::path::Path::new(db_path),
            crate::path_jail::allowed_data_root(),
        )?;
    }
    // The eviction threshold follows the resource profile here, like the
    // deadlines and quotas, whichever way `config` was built.
    let session_manager = Arc::new(SessionManager::new(SessionConfig {
        evict_idle_minutes: config.resolved_evict_idle_minutes(),
        ..config.session.clone()
    }));

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

    #[cfg(feature = "db")]
    let persistent_manager = init_persistence(
        config,
        session_manager.clone(),
        persistence_limits(&limits, governor.config().rss_soft_limit_bytes),
    )
    .await;

    let state = AppState {
        server,
        session_manager,
        #[cfg(feature = "db")]
        persistent_manager,
        audit_logger,
        governor,
    };

    Ok((create_router(state.clone(), config), state))
}

/// The request limits of `config`'s resource profile with its overrides
/// applied, the values `tools::limits::current()` holds once installed. An
/// invalid override refuses the start.
#[cfg(feature = "db")]
fn resolve_limits(config: &ServerConfig) -> Result<crate::tools::limits::Limits, TransportError> {
    use crate::config::ResourceProfile;
    use crate::tools::limits::Limits;
    match config.resource_profile() {
        ResourceProfile::Off => Limits::off(),
        ResourceProfile::Hosted => Limits::hosted(),
    }
    .with_overrides(&config.limits.overrides)
    .map_err(TransportError::Config)
}

/// The session database must lie outside the path-jail root (INF-1): file
/// tools can write anywhere under it. An authenticated or open public server
/// refuses to start otherwise; a local one (the desktop app keeps its store
/// under the user's data directory, which can be inside a data root that
/// defaults to the working directory) logs a warning and starts. A data root
/// that cannot be resolved is treated the same way.
#[cfg(feature = "db")]
fn enforce_db_path_policy(
    mode: DeploymentMode,
    db_path: &std::path::Path,
    data_root: Result<std::path::PathBuf, String>,
) -> Result<(), TransportError> {
    let verdict = data_root
        .map_err(|e| format!("cannot check P2A_DB_PATH against the data root: {e}"))
        .and_then(|root| {
            crate::persistent_session::check_db_path_outside_data_root(db_path, &root)
        });
    match verdict {
        Ok(()) => Ok(()),
        Err(message) if mode == DeploymentMode::Local => {
            tracing::warn!("{message}");
            Ok(())
        }
        Err(message) => Err(TransportError::Config(message)),
    }
}

/// The conversation-store bounds from the resolved request limits and the
/// governor's resident-memory floor. They are handed to the store, so the db
/// module never reads the process-wide limits.
#[cfg(feature = "db")]
fn persistence_limits(
    limits: &crate::tools::limits::Limits,
    rss_soft_limit_bytes: Option<u64>,
) -> crate::db::PersistenceLimits {
    crate::db::PersistenceLimits {
        message_bytes: Some(usize::try_from(limits.message_bytes).unwrap_or(usize::MAX)),
        conversations_per_session: limits.conversations_per_session,
        messages_per_conversation: limits.messages_per_conversation,
        session_db_bytes: limits.session_db_bytes,
        rss_soft_limit_bytes,
    }
}

/// Create the persistent session manager over the shared live registry and
/// start its background tasks (row deletion on session removal, the
/// periodic sweep), or `None` (with a log line) when the database cannot be
/// opened.
#[cfg(feature = "db")]
async fn init_persistence(
    config: &ServerConfig,
    session_manager: Arc<SessionManager>,
    limits: crate::db::PersistenceLimits,
) -> Option<Arc<crate::persistent_session::PersistentSessionManager>> {
    let db_path = config.http.db_path.as_deref();
    tracing::info!("Initializing database with path: {:?}", db_path);
    match crate::persistent_session::PersistentSessionManager::new(db_path, session_manager, limits)
        .await
    {
        Ok(manager) => {
            tracing::info!("Database persistence enabled - conversation routes will be available");
            let manager = Arc::new(manager);
            // Stored sessions that can still be adopted keep their data
            // directories through the first-use purge of stale ones, at
            // most as many as the live table holds.
            let ttl_minutes = manager.sessions().ttl_minutes();
            let max_kept = config.session.max_sessions;
            if let Err(e) = manager
                .db()
                .keep_adoptable_session_dirs(ttl_minutes, max_kept)
                .await
            {
                tracing::warn!(error = %e, "Cannot list the stored sessions; their data directories count as stale");
            }
            // Delete the rows of removed sessions and sweep stale ones.
            manager.start_background_tasks();
            Some(manager)
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

/// Validate `config` as an HTTP configuration and bind its address. Every
/// HTTP start path (the binary, the embedded desktop server) goes through
/// here, so the bind guard (`config::check_bind_policy`, part of
/// `ServerConfig::validate`) cannot be skipped: it applies whatever
/// `config.transport` says, including the stdio default of
/// `ServerConfig::default()`.
pub async fn bind(config: &ServerConfig) -> TransportResult<tokio::net::TcpListener> {
    config
        .as_http()
        .validate()
        .map_err(|e| TransportError::Config(e.to_string()))?;
    let addr = config.http.addr;
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| TransportError::Http(format!("Failed to bind to {}: {}", addr, e)))
}

/// Serve the app on `listener` until `shutdown` resolves.
///
/// The configuration's address is replaced by the socket's own and its
/// transport set to HTTP, so the deployment mode, resource profile, bind
/// guard and Host check follow the socket actually served, not the address
/// or transport a caller wrote in `config` (the default transport is stdio).
/// Then it validates the configuration again, so no caller can skip the
/// guard, builds the app, installs the process-wide policies (allocation
/// ceiling, session directories) and serves it with graceful shutdown.
/// Before it returns, every audit entry queued by then is written out; a call
/// still running after its deadline answer is logged when its work ends.
/// It does not install the request limits or apply `--limit-override`
/// (`P2A_LIMIT_OVERRIDES`); `transport::start_transport` does that before it
/// calls `start_http_transport`. A caller that uses `serve` directly gets the
/// off profile's request limits, whatever the governor's resource profile,
/// unless it first calls
/// `tools::limits::install(tools::limits::resolve(&config)?)`.
pub async fn serve(
    listener: tokio::net::TcpListener,
    mut config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> TransportResult<()> {
    config.http.addr = listener
        .local_addr()
        .map_err(|e| TransportError::Http(format!("Failed to read the bound address: {e}")))?;
    config.transport = TransportType::Http;
    config
        .validate()
        .map_err(|e| TransportError::Config(e.to_string()))?;
    let (app, state) = build_app(&config).await?;
    let exec = state.governor.config().clone();
    crate::exec::install_process_policies(&exec);
    log_startup(&config, &exec);
    if let Some(limit) = exec.stuck_exit {
        spawn_stuck_watchdog(state.governor.clone(), limit);
    }

    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| TransportError::Http(format!("HTTP server error: {}", e)));
    // Write out the queued audit entries before returning, so a caller that
    // exits right after a graceful shutdown keeps them. The governor's logger
    // holds the entries of every tool call; `state.audit_logger` is opened
    // for the same `config.audit` and is normally the same writer, flushed
    // as well in case an embedder logged through it.
    state.governor.flush_audit().await;
    state.audit_logger.flush().await;
    served
}

/// Start the HTTP transport: [`bind`], then [`serve`] until the process ends.
/// There is no shutdown signal here: the binary serves until it is killed,
/// so its audit entries are as safe as the writer's own flush whenever its
/// queue runs empty (see [`crate::audit`]).
pub async fn start_http_transport(config: &ServerConfig) -> TransportResult<()> {
    let listener = bind(config).await?;
    serve(listener, config.clone(), std::future::pending()).await
}

fn log_startup(config: &ServerConfig, exec: &ExecConfig) {
    let addr = config.http.addr;
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
            if config.http.expose_enumeration {
                tracing::warn!(
                    "--expose-enumeration (P2A_EXPOSE_ENUMERATION) is ignored in open public mode"
                );
            }
        }
        DeploymentMode::Local => {
            tracing::warn!(
                "HTTP access-token authentication is DISABLED (local mode: loopback bind, \
                 requests must be addressed to localhost); set P2A_ACCESS_TOKEN before \
                 exposing this server"
            );
        }
    }
    if (exec.session_data_bytes.is_some() || exec.total_data_bytes.is_some())
        && exec.session_concurrency > 1
    {
        tracing::warn!(
            "dataset quotas are configured with --session-concurrency \
             (P2A_SESSION_CONCURRENCY) above 1: a call refused by a dataset quota restores the \
             session's datasets as they were when it started, which removes datasets that \
             another concurrent call in the same session added and reverts the ones it replaced"
        );
    }
    if exec.rss_soft_limit_bytes.is_some() && p2a_core::get_process_memory().is_none() {
        tracing::warn!(
            "--memory-soft-limit-mb (P2A_MEMORY_SOFT_LIMIT_MB) is set, but this platform does \
             not report the process memory; memory admission is disabled"
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
/// `state.persistent_manager` is set. The routes and middleware follow the
/// HTTP deployment mode of `config` whatever its `transport` field says.
pub fn create_router(state: AppState, config: &ServerConfig) -> Router {
    let config = &config.as_http();
    #[cfg(feature = "db")]
    let persistent_manager = state.persistent_manager.clone();
    let governor = Arc::clone(&state.governor);

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
    //
    // The deployment mode (of the HTTP-forced config) is handed to the
    // handlers that depend on it, e.g. session creation ignores `user_id`
    // in open public mode. The governor goes to the conversation routes
    // whose work it bounds (the dataset reload) without changing their
    // state type.
    middleware::apply_global_middleware(router, config)
        .layer(axum::Extension(config.deployment_mode()))
        .layer(axum::Extension(governor))
}

/// Create the base router with core routes.
fn create_base_router(state: AppState, config: &ServerConfig) -> Router {
    let mode = config.deployment_mode();

    // The two enumeration surfaces, `GET /api/sessions` (every active session
    // ID, which would defeat the unguessable-session-ID isolation) and `GET
    // /api/files` (a filesystem browser), are registered only for an
    // authenticated server or a local one started with --expose-enumeration,
    // and never in open public mode (which includes a loopback bind with
    // --allow-unauthenticated behind a proxy). Neither shipped client uses
    // them.
    let expose_enumeration = match mode {
        DeploymentMode::OpenPublic => false,
        DeploymentMode::Authenticated => true,
        DeploymentMode::Local => config.http.expose_enumeration,
    };

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

    if expose_enumeration {
        router = router
            .route("/api/sessions", get(list_sessions))
            // File browser
            .route("/api/files", get(list_files));
    }

    // WebSocket route: off in open public mode unless --enable-ws
    // (P2A_ENABLE_WS). Its tool messages run under the governor like HTTP
    // calls, and its message and frame sizes are capped like request bodies.
    #[cfg(feature = "websocket")]
    if mode != DeploymentMode::OpenPublic || config.http.enable_ws {
        router = router.route("/ws", get(super::websocket::ws_handler));
    }

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
/// deadline. The `governor` object reports in-flight and overdue work; in
/// open public mode it leaves out the in-flight count and the process memory
/// (`ToolGovernor::health_report`).
async fn health_check(State(state): State<AppState>) -> Response {
    let session_count = state.session_manager.session_count().await;
    let health = state.governor.health();
    let governor = state.governor.health_report(&health);
    let (status, label) = if health.healthy {
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

#[cfg(test)]
mod serve_tests {
    use super::*;
    use serde_json::{Value, json};

    /// One HTTP/1.1 POST to `addr` (`Host: localhost`, the connection closed
    /// after the answer); the JSON body of a 2xx answer.
    async fn post(addr: std::net::SocketAddr, uri: &str, body: &Value) -> Value {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let body = body.to_string();
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "POST {uri} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).await.unwrap();
        let (head, json) = answer.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 2"), "{head}");
        serde_json::from_str(json).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serve_returns_only_after_the_queued_audit_entries_are_written() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("audit.log");
        let mut config = ServerConfig {
            transport: TransportType::Http,
            ..Default::default()
        };
        config.audit = crate::config::AuditConfig {
            enabled: true,
            path: log.to_string_lossy().into_owned(),
        };
        // The writer of this log keeps every entry in its queue until
        // `held` drops.
        let held = crate::audit::gate::hold(&log);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // `serve` installs process-wide policies (the p2a-core allocation
        // ceiling, the session-directory switch) in this test process, which
        // the other lib tests share. Resolved as `serve` resolves them (at
        // the bound address), they must be the defaults.
        let mut bound = config.clone();
        bound.http.addr = addr;
        let exec = crate::exec::ExecConfig::from_config(&bound);
        assert!(
            exec.alloc_ceiling_bytes.is_none() && !exec.session_dirs,
            "serve would install a non-default process policy in the shared test process"
        );
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let mut server = tokio::spawn(serve(listener, config, async {
            let _ = stopped.await;
        }));

        let created = post(addr, "/api/sessions", &json!({})).await;
        let sid = created["data"]["session_id"].as_str().unwrap().to_string();
        let call = json!({"session_id": sid, "arguments": {}});
        for _ in 0..5 {
            post(addr, "/api/tools/list_datasets", &call).await;
        }
        stop.send(()).unwrap();
        // While the entries are held, serve cannot return. With the gate
        // closed this holds however long the wait is, so it cannot fail on a
        // slow machine.
        assert!(
            tokio::time::timeout(Duration::from_millis(200), &mut server)
                .await
                .is_err(),
            "serve returned with its audit entries still queued"
        );
        drop(held);
        server.await.unwrap().unwrap();

        let text = std::fs::read_to_string(&log).unwrap();
        let entries: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(entries.len(), 5, "{text}");
        assert!(
            entries
                .iter()
                .all(|e| e["tool_name"] == "list_datasets" && e["session_id"] == sid.as_str())
        );
    }
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::*;

    #[test]
    fn db_path_policy_by_mode() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path().canonicalize().unwrap();
        let other = tempfile::tempdir().unwrap();
        let inside = r.join("db");
        let outside = other.path().join("db");

        // Local mode (the desktop app) only warns.
        assert!(enforce_db_path_policy(DeploymentMode::Local, &inside, Ok(r.clone())).is_ok());
        assert!(
            enforce_db_path_policy(DeploymentMode::Local, &inside, Err("no root".into())).is_ok()
        );
        for mode in [DeploymentMode::Authenticated, DeploymentMode::OpenPublic] {
            match enforce_db_path_policy(mode, &inside, Ok(r.clone())) {
                Err(TransportError::Config(message)) => {
                    assert!(message.contains("P2A_DB_PATH"), "{message}")
                }
                other => panic!("{mode:?}: expected a config error, got {other:?}"),
            }
            assert!(enforce_db_path_policy(mode, &outside, Ok(r.clone())).is_ok());
            assert!(matches!(
                enforce_db_path_policy(mode, &outside, Err("no root".into())),
                Err(TransportError::Config(_))
            ));
        }
    }
}
