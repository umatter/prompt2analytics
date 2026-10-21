//! prompt2analytics MCP Server
//!
//! This is the main entry point for the MCP server that exposes
//! data analytics capabilities via the Model Context Protocol.
//!
//! Supports multiple transports:
//! - `stdio`: Standard input/output for CLI and desktop integration (default)
//! - `http`: HTTP REST API with optional WebSocket for web clients (requires `http` feature)
//!
//! The binary is a thin wrapper over the `p2a_mcp` library: it parses the
//! configuration, validates it, builds the async runtime explicitly and runs
//! the configured transport.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use p2a_mcp::config::{CliArgs, ServerConfig};

fn main() -> Result<()> {
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

    config.validate()?;

    tracing::info!("Starting prompt2analytics MCP server...");
    tracing::info!("Transport: {:?}", config.transport);

    // At least two workers, so /health and SSE keep-alives are served on a
    // one-CPU machine while tool work runs on the blocking pool.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.worker_threads())
        .enable_all()
        .build()?;

    // Start the appropriate transport
    runtime
        .block_on(p2a_mcp::transport::start_transport(&config))
        .map_err(|e| {
            tracing::error!("Transport error: {}", e);
            anyhow::anyhow!("{}", e)
        })?;

    tracing::info!("MCP server shutting down");
    Ok(())
}
