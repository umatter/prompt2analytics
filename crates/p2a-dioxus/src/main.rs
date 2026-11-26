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
mod examples;
mod exhibit;
mod export;
mod figure_pdf;
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

/// How long the embedded server keeps an idle session: 30 days.
#[cfg(any(feature = "desktop", feature = "mobile"))]
const DESKTOP_SESSION_TTL_MINUTES: u64 = 30 * 24 * 60;

/// Start the embedded backend server on native platforms.
///
/// Returns the server handle, which must be kept alive for the duration of
/// the app, or the reason the server did not start. The server binds a free
/// loopback port (port 0), so a second instance of the app or a dev server on
/// 8080/8081 never collides with it; the caller passes `server.url()` to the
/// API client.
#[cfg(any(feature = "desktop", feature = "mobile"))]
fn start_embedded_backend() -> Result<p2a_mcp::EmbeddedServer, String> {
    use p2a_mcp::EmbeddedServerConfig;

    // Create a tokio runtime for the embedded server
    let rt = tokio::runtime::Builder::new_multi_thread()
        .thread_stack_size(p2a_mcp::RUNTIME_STACK_BYTES)
        .enable_all()
        .build()
        .map_err(|e| format!("cannot create the async runtime: {e}"))?;

    // Determine the database path (use user data directory)
    let db_path = dirs::data_dir()
        .map(|d| d.join("p2a").join("data"))
        .map(|p| p.to_string_lossy().to_string());

    if let Some(ref path) = db_path {
        tracing::info!("Database path: {}", path);
    } else {
        tracing::warn!("Could not determine user data directory, using in-memory database");
    }

    // Configure the embedded server: a free loopback port chosen by the OS,
    // and sessions that live 30 days from their last use, so the app's
    // conversations survive restarts and idle hours (its session id is kept
    // in the config directory, state::session).
    let mut config = EmbeddedServerConfig::default()
        .with_port(0)
        .with_host("127.0.0.1")
        .with_session_ttl_minutes(DESKTOP_SESSION_TTL_MINUTES);

    // Set database path if available
    if let Some(path) = db_path {
        config = config.with_db_path(path);
    }

    // Start the server; it binds before it returns, so a bind error is an Err.
    let server = rt
        .block_on(p2a_mcp::start_embedded_server(config))
        .map_err(|e| e.to_string())?;
    tracing::info!("Embedded backend started at {}", server.url());

    // Keep the runtime alive by leaking it (it will live for the app lifetime)
    // This is intentional - we need the runtime to stay alive for the server
    std::mem::forget(rt);

    Ok(server)
}

fn main() {
    // Initialize logging for the current platform
    init_logging();

    // On web, WASM has no default panic backtrace: a Rust panic compiles to an
    // `unreachable` trap and the console only shows "RuntimeError: unreachable".
    // Install a hook that logs the panic location + message via tracing (which
    // routes to the browser console) BEFORE the trap aborts execution.
    #[cfg(feature = "web")]
    {
        std::panic::set_hook(Box::new(|info| {
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown location>".to_string());
            tracing::error!("WASM PANIC at {} :: {}", loc, msg);
        }));
    }

    tracing::info!(
        "Starting prompt2analytics Dioxus frontend on {}",
        platform_name()
    );

    // On native platforms, start the embedded backend
    #[cfg(any(feature = "desktop", feature = "mobile"))]
    let _server = {
        tracing::info!("Starting embedded p2a-mcp backend...");
        match start_embedded_backend() {
            Ok(server) => {
                // Every API client and the chat stream talk to this server.
                api::set_backend_base_url(Some(server.url()));
                Some(server)
            }
            Err(e) => {
                tracing::error!("Embedded backend failed to start: {}", e);
                api::set_backend_start_error(e);
                None
            }
        }
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
        // Narrow the static CSP's connect-src to the compiled-in backend
        // before anything renders.
        api::install_backend_csp();
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
