//! Dioxus cross-platform frontend for prompt2analytics
//!
//! Supports web (WASM), desktop (Windows/macOS/Linux), and mobile (iOS/Android).
//! It communicates with the p2a-mcp HTTP backend.

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
            .unwrap_or_else(|_| EnvFilter::new("info,p2a_dioxus=debug"));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
    }
}

fn main() {
    // Initialize logging for the current platform
    init_logging();

    tracing::info!(
        "Starting prompt2analytics Dioxus frontend on {}",
        platform_name()
    );

    // Launch the Dioxus app
    dioxus::launch(app::App);
}
