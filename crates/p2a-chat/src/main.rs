//! `p2a-chat` — Standalone terminal chat client for prompt2analytics
//!
//! Connects to a running p2a-mcp HTTP server and provides an interactive REPL.
//! This binary has no dependency on p2a-core, so it compiles in seconds.

use clap::Parser;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize minimal logger
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .format_timestamp_secs()
        .init();

    let args = p2a_chat::ChatArgs::parse();
    p2a_chat::run(&args).await
}
