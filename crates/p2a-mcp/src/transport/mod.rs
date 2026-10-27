//! Transport layer abstraction for p2a-mcp.
//!
//! Supports multiple transport mechanisms:
//! - `stdio`: Standard input/output for CLI and desktop integration
//! - `http`: HTTP REST API with optional WebSocket for web clients

#[cfg(feature = "http")]
pub mod http;

#[cfg(feature = "http")]
pub mod middleware;

#[cfg(feature = "http")]
pub mod routes_sessions;

#[cfg(feature = "http")]
pub mod routes_tools;

#[cfg(all(feature = "http", feature = "llm"))]
pub mod llm_routes;

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(all(feature = "http", feature = "db"))]
pub mod conversation;

use crate::config::ServerConfig;

/// Transport error types.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[cfg(feature = "http")]
    #[error("HTTP error: {0}")]
    Http(String),
}

/// Result type for transport operations.
pub type TransportResult<T> = Result<T, TransportError>;

/// Start the appropriate transport based on configuration.
///
/// The request limits of the configured resource profile, with the
/// `--limit-override` (`P2A_LIMIT_OVERRIDES`) entries applied, are resolved
/// first, for every transport; an unknown override key fails startup. The
/// stdio transport installs them here. On the HTTP path
/// `transport::http::serve` resolves them again, from the socket it serves,
/// and installs them. Either way the server holds its limits while it runs,
/// and a start whose limits differ from those of a server still running in
/// this process is refused (`http::serve` says more; without the `http`
/// feature no other server can run).
pub async fn start_transport(config: &ServerConfig) -> TransportResult<()> {
    use crate::config::TransportType;

    let limits = crate::tools::limits::resolve(config).map_err(TransportError::Config)?;

    match config.transport {
        TransportType::Stdio => {
            // Held while the stdio server runs, as `http::serve` holds its
            // limits, so no other server in this process replaces them.
            #[cfg(feature = "http")]
            let _held = http::hold_limits(&limits)?;
            tracing::info!(
                profile = ?config.resource_profile(),
                overrides = ?config.limits.overrides,
                "resource limits"
            );
            crate::tools::limits::install(limits);
            start_stdio_transport(config).await
        }
        // `serve` holds and installs the limits; an install here, before its
        // check, would replace those of a server already running.
        #[cfg(feature = "http")]
        TransportType::Http => http::start_http_transport(config).await,
    }
}

/// Start the stdio transport (existing behavior).
///
/// The per-allocation ceiling of the configured resource profile applies
/// here too (none unless `--limits hosted` or `--max-alloc-mb` is set), and
/// so do the request limits `start_transport` installed (argument, dataset
/// row and response-size checks in `AnalyticsServer`'s `call_tool`);
/// deadlines and quotas are HTTP-only.
async fn start_stdio_transport(config: &ServerConfig) -> TransportResult<()> {
    use crate::server::AnalyticsServer;
    use rmcp::{ServiceExt, transport::stdio};

    p2a_core::resource::set_alloc_ceiling_bytes(config.alloc_ceiling_bytes());

    tracing::info!("Starting stdio transport...");

    let audit = std::sync::Arc::new(
        crate::audit::AuditLogger::new(&config.audit).unwrap_or_else(|e| {
            tracing::warn!("Failed to initialize audit logger: {}", e);
            crate::audit::AuditLogger::disabled()
        }),
    );
    let server = AnalyticsServer::new().with_audit_logger(audit.clone());
    let service = server
        .serve(stdio())
        .await
        .map_err(|e| TransportError::Server(format!("Failed to start stdio transport: {}", e)))?;

    tracing::info!("MCP server running on stdio, waiting for requests...");

    let waited = service
        .waiting()
        .await
        .map_err(|e| TransportError::Server(format!("Stdio transport error: {}", e)));
    // Write out the queued audit entries before the process exits.
    audit.flush().await;
    waited?;

    Ok(())
}
