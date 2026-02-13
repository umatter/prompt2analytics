//! prompt2analytics MCP Server
//!
//! This is the main entry point for the MCP server that exposes
//! data analytics capabilities via the Model Context Protocol.
//!
//! Supports multiple transports:
//! - `stdio`: Standard input/output for CLI and desktop integration (default)
//! - `http`: HTTP REST API with optional WebSocket for web clients (requires `http` feature)

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod audit;
mod config;
#[cfg(feature = "db")]
pub mod db;
#[cfg(feature = "llm")]
mod llm;
#[cfg(feature = "db")]
pub mod persistent_session;
mod server;
mod session;
mod tools;
mod transport;

use config::{CliArgs, ServerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args = CliArgs::parse();
    let config = ServerConfig::from_args(args)?;

    // Initialize logging to stderr (so it doesn't interfere with stdio MCP transport)
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "p2a_mcp=info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    tracing::info!("Starting prompt2analytics MCP server...");
    tracing::info!("Transport: {:?}", config.transport);

    // Start the appropriate transport
    transport::start_transport(&config).await.map_err(|e| {
        tracing::error!("Transport error: {}", e);
        anyhow::anyhow!("{}", e)
    })?;

    tracing::info!("MCP server shutting down");
    Ok(())
}
