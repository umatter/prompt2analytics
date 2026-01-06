//! prompt2analytics MCP Server
//!
//! This is the main entry point for the MCP server that exposes
//! data analytics capabilities via the Model Context Protocol.

use anyhow::Result;
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod server;
mod tools;

use server::AnalyticsServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to stderr (so it doesn't interfere with stdio MCP transport)
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "p2a_mcp=info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    tracing::info!("Starting prompt2analytics MCP server...");

    // Create and start the server
    let server = AnalyticsServer::new();
    let service = server.serve(stdio()).await?;

    tracing::info!("MCP server running, waiting for requests...");

    // Wait for the service to complete
    service.waiting().await?;

    tracing::info!("MCP server shutting down");
    Ok(())
}
