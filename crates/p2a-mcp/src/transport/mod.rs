//! Transport layer abstraction for p2a-mcp.
//!
//! Supports multiple transport mechanisms:
//! - `stdio`: Standard input/output for CLI and desktop integration
//! - `http`: HTTP REST API with optional WebSocket for web clients

#[cfg(feature = "http")]
pub mod http;

#[cfg(feature = "websocket")]
pub mod websocket;

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
pub async fn start_transport(config: &ServerConfig) -> TransportResult<()> {
    use crate::config::TransportType;

    match config.transport {
        TransportType::Stdio => {
            start_stdio_transport().await
        }
        #[cfg(feature = "http")]
        TransportType::Http => {
            http::start_http_transport(config).await
        }
    }
}

/// Start the stdio transport (existing behavior).
async fn start_stdio_transport() -> TransportResult<()> {
    use rmcp::{transport::stdio, ServiceExt};
    use crate::server::AnalyticsServer;

    tracing::info!("Starting stdio transport...");

    let server = AnalyticsServer::new();
    let service = server.serve(stdio()).await.map_err(|e| {
        TransportError::Server(format!("Failed to start stdio transport: {}", e))
    })?;

    tracing::info!("MCP server running on stdio, waiting for requests...");

    service.waiting().await.map_err(|e| {
        TransportError::Server(format!("Stdio transport error: {}", e))
    })?;

    Ok(())
}
