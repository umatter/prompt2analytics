//! Dioxus web frontend for prompt2analytics
//!
//! This is a minimal prototype to validate the Dioxus migration approach.
//! It communicates with the existing p2a-mcp HTTP backend.

// Suppress dead code warnings for prototype - many API items are defined for future use
#![allow(dead_code)]

mod api;
mod app;
mod components;
mod state;
mod utils;

use tracing::Level;
use tracing_wasm::WASMLayerConfigBuilder;

fn main() {
    // Initialize tracing for WASM
    let config = WASMLayerConfigBuilder::default()
        .set_max_level(Level::DEBUG)
        .build();
    tracing_wasm::set_as_global_default_with_config(config);

    tracing::info!("Starting prompt2analytics Dioxus prototype");

    // Launch the Dioxus app
    dioxus::launch(app::App);
}
