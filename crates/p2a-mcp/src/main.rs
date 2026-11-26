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

/// The log filter when `RUST_LOG` is unset: the server's own messages from
/// info up, and warnings from the analytics library (a parquet file loaded
/// after a caught polars panic, for one).
const DEFAULT_LOG_FILTER: &str = "p2a_mcp=info,p2a_core=warn";

fn main() -> Result<()> {
    // Parse command-line arguments
    let args = CliArgs::parse();
    let config = ServerConfig::from_args(args)?;

    // Initialize logging to stderr (so it doesn't interfere with stdio MCP transport)
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| DEFAULT_LOG_FILTER.into()),
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
        .thread_stack_size(p2a_mcp::RUNTIME_STACK_BYTES)
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

#[cfg(test)]
mod tests {
    use super::DEFAULT_LOG_FILTER;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn default_log_filter_shows_core_warnings() {
        // Pre-fix the default filter was "p2a_mcp=info", which drops every
        // event of the p2a_core target.
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_env_filter(tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER))
            .with_writer(move || writer.clone())
            .finish();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../p2a-core/tests/data/arrow_schema_short.parquet");
        let loaded = tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();
            p2a_core::data::DataLoader::load(&path)
        });
        assert!(loaded.is_err(), "the fixture loaded");
        let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("polars panicked reading a parquet file"),
            "nothing logged: {log}"
        );
        assert!(log.contains("WARN"), "{log}");
    }
}
