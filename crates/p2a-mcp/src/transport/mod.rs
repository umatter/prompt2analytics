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
/// `--limit-override` (`P2A_LIMIT_OVERRIDES`) entries applied, are installed
/// first, for every transport; an unknown override key fails startup.
pub async fn start_transport(config: &ServerConfig) -> TransportResult<()> {
    use crate::config::TransportType;

    let limits = crate::tools::limits::resolve(config).map_err(TransportError::Config)?;
    tracing::info!(
        profile = ?config.resource_profile(),
        overrides = ?config.limits.overrides,
        "resource limits"
    );
    crate::tools::limits::install(limits);

    match config.transport {
        TransportType::Stdio => start_stdio_transport(config).await,
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
