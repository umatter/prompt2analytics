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
mod export;
mod platform;
mod state;
mod utils;

use platform::platform_name;

/// Initialize logging based on platform
fn init_logging() {
    #[cfg(feature = "web")]
    {
        use tracing::Level;
        use tracing_wasm::WASMLayerConfigBuilder;

        let config = WASMLayerConfigBuilder::default()
            .set_max_level(Level::DEBUG)
            .build();
        tracing_wasm::set_as_global_default_with_config(config);
    }

    #[cfg(any(feature = "desktop", feature = "mobile"))]
    {
        use tracing_subscriber::EnvFilter;

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,p2a_dioxus=debug,p2a_mcp=info,p2a_core=info"));

        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

/// Start the embedded backend server on native platforms
/// Returns the server handle which must be kept alive for the duration of the app
#[cfg(any(feature = "desktop", feature = "mobile"))]
fn start_embedded_backend() -> Option<p2a_mcp::EmbeddedServer> {
    use p2a_mcp::EmbeddedServerConfig;

    // Create a tokio runtime for the embedded server
    let rt = tokio::runtime::Runtime::new().ok()?;

    // Determine the database path (use user data directory)
    let db_path = dirs::data_dir()
        .map(|d| d.join("p2a").join("data"))
        .map(|p| p.to_string_lossy().to_string());

    if let Some(ref path) = db_path {
        tracing::info!("Database path: {}", path);
    } else {
        tracing::warn!("Could not determine user data directory, using in-memory database");
    }

    // Configure the embedded server
    // Use port 8081 to avoid conflict with dx serve dev server on 8080
    let mut config = EmbeddedServerConfig::default()
        .with_port(8081)
        .with_host("127.0.0.1");

    // Set database path if available
    if let Some(path) = db_path {
        config = config.with_db_path(path);
    }

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
    #[cfg(any(feature = "desktop", feature = "mobile"))]
    let _server = {
        tracing::info!("Starting embedded p2a-mcp backend...");
        let server = start_embedded_backend();
        if server.is_none() {
            tracing::warn!("Embedded backend failed to start. App may not function correctly.");
        }
        server
    };

    // Launch the Dioxus app with platform-specific configuration
    #[cfg(feature = "desktop")]
    {
        use dioxus::desktop::{Config, WindowBuilder};

        // Load the app icon
        let icon = load_icon();

        let mut window_builder = WindowBuilder::new()
            .with_title("prompt2analytics")
            .with_inner_size(dioxus::desktop::tao::dpi::LogicalSize::new(1200.0, 800.0));

        if let Some(icon) = icon {
            window_builder = window_builder.with_window_icon(Some(icon));
        }

        let config = Config::new().with_window(window_builder);

        dioxus::LaunchBuilder::desktop()
            .with_cfg(config)
            .launch(app::App);
    }

    #[cfg(feature = "mobile")]
    {
        dioxus::launch(app::App);
    }

    #[cfg(feature = "web")]
    {
        dioxus::launch(app::App);
    }
}

/// Load the application icon from embedded PNG data
#[cfg(feature = "desktop")]
fn load_icon() -> Option<dioxus::desktop::tao::window::Icon> {
    use dioxus::desktop::tao::window::Icon;

    // Embed the icon at compile time
    let icon_bytes = include_bytes!("../assets/icons/p2a-icon-256.png");

    // Decode PNG to RGBA
    let img = image::load_from_memory(icon_bytes).ok()?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    Icon::from_rgba(rgba.into_raw(), width, height).ok()
}
