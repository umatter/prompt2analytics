//! Dioxus cross-platform frontend for prompt2analytics
//!
//! Supports web (WASM), desktop (Windows/macOS/Linux), and mobile (iOS/Android).
//!
//! - **Web**: Connects to a remote p2a-mcp backend (configured via settings)
//! - **Desktop/Mobile**: Runs an embedded p2a-mcp backend automatically

// Suppress dead code warnings for prototype - many API items are defined for future use
#![allow(dead_code)]

mod api;
mod app;
mod components;
mod platform;
mod state;
mod utils;

use platform::platform_name;

/// Initialize logging based on platform
fn init_logging() {
    #[cfg(target_arch = "wasm32")]
    {
        use tracing::Level;
        use tracing_wasm::WASMLayerConfigBuilder;

        let config = WASMLayerConfigBuilder::default()
            .set_max_level(Level::DEBUG)
            .build();
        tracing_wasm::set_as_global_default_with_config(config);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::EnvFilter;

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,p2a_dioxus=debug,p2a_mcp=info,p2a_core=info"));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
    }
}

/// Start the embedded backend server on native platforms
/// Returns the server handle which must be kept alive for the duration of the app
#[cfg(not(target_arch = "wasm32"))]
fn start_embedded_backend() -> Option<p2a_mcp::EmbeddedServer> {
    use p2a_mcp::EmbeddedServerConfig;

    // Create a tokio runtime for the embedded server
    let rt = tokio::runtime::Runtime::new().ok()?;

    // Configure the embedded server
    let config = EmbeddedServerConfig::default()
        .with_port(8080)
        .with_host("127.0.0.1");

    // Start the server
    let server = rt.block_on(async {
        match p2a_mcp::start_embedded_server(config).await {
            Ok(server) => {
                tracing::info!("Embedded backend started at {}", server.url());
                Some(server)
            }
            Err(e) => {
                tracing::error!("Failed to start embedded backend: {}", e);
                None
            }
        }
    })?;

    // Keep the runtime alive by leaking it (it will live for the app lifetime)
    // This is intentional - we need the runtime to stay alive for the server
    std::mem::forget(rt);

    Some(server)
}

fn main() {
    // Initialize logging for the current platform
    init_logging();

    tracing::info!(
        "Starting prompt2analytics Dioxus frontend on {}",
        platform_name()
    );

    // On native platforms, start the embedded backend
    #[cfg(not(target_arch = "wasm32"))]
    let _server = {
        tracing::info!("Starting embedded p2a-mcp backend...");
        let server = start_embedded_backend();
        if server.is_none() {
            tracing::warn!("Embedded backend failed to start. App may not function correctly.");
        }
        server
    };

    // Launch the Dioxus app
    dioxus::launch(app::App);
}
