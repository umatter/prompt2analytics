//! p2a-mcp library interface
//!
//! This module exposes the MCP server functionality as a library,
//! allowing it to be embedded in other applications (like p2a-dioxus desktop).

pub mod audit;
pub mod config;
#[cfg(feature = "db")]
pub mod db;
#[cfg(feature = "llm")]
pub mod llm;
pub mod path_jail;
#[cfg(feature = "db")]
pub mod persistent_session;
pub mod server;
pub mod session;
pub mod tools;
pub mod transport;

use tokio::sync::oneshot;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(feature = "http")]
pub use config::HttpConfig;
pub use config::{ServerConfig, SessionConfig, TransportType};
pub use server::AnalyticsServer;
pub use session::SessionManager;

/// Configuration for the embedded server
#[derive(Debug, Clone)]
pub struct EmbeddedServerConfig {
    /// Port to listen on (default: 8080)
    pub port: u16,
    /// Host to bind to (default: 127.0.0.1)
    pub host: String,
    /// Enable CORS for all origins.
    ///
    /// Defaults to `false`. Only opt in for local development with a
    /// loopback-only bind address; enabling this with a non-loopback
    /// host exposes the server to cross-origin requests from any page.
    pub cors_permissive: bool,
    /// Database path for persistence (default: None = in-memory or auto-detect)
    pub db_path: Option<String>,
}

impl Default for EmbeddedServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            cors_permissive: false,
            db_path: None,
        }
    }
}

impl EmbeddedServerConfig {
    /// Create config with custom port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Create config with custom host
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set database path
    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }
}

/// Handle to control the embedded server
pub struct EmbeddedServer {
    shutdown_tx: Option<oneshot::Sender<()>>,
    port: u16,
    host: String,
}

impl EmbeddedServer {
    /// Get the URL the server is listening on
    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Shutdown the server
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for EmbeddedServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Start the embedded server in the background
///
/// Returns an `EmbeddedServer` handle that can be used to get the URL
/// or shut down the server.
///
/// # Example
/// ```ignore
/// let server = p2a_mcp::start_embedded_server(Default::default()).await?;
/// println!("Server running at: {}", server.url());
/// // Server runs until `server` is dropped or shutdown() is called
/// ```
#[cfg(feature = "http")]
pub async fn start_embedded_server(
    config: EmbeddedServerConfig,
) -> Result<EmbeddedServer, Box<dyn std::error::Error + Send + Sync>> {
    use std::net::SocketAddr;

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    // Warn when permissive CORS is combined with a non-loopback bind address.
    if config.cors_permissive && !is_loopback(&config.host) {
        tracing::warn!(
            "cors_permissive=true combined with non-loopback host {:?} exposes the server to \
             cross-origin requests from any web page; set cors_permissive=false in production",
            config.host
        );
    }

    // Build server config
    let server_config = ServerConfig {
        transport: TransportType::Http,
        http: HttpConfig {
            addr,
            cors_permissive: config.cors_permissive,
            cors_origins: vec![],
            // The embedded server is intended for local (loopback) desktop use;
            // no shared token. A non-loopback bind is rejected by the HTTP
            // transport's startup guard unless explicitly opened.
            access_token: None,
            allow_unauthenticated: false,
            #[cfg(feature = "db")]
            db_path: config.db_path.clone(),
        },
        session: SessionConfig {
            ttl_minutes: 60,
            max_sessions: 100,
        },
        #[cfg(feature = "auth")]
        auth: config::AuthConfig {
            enabled: false,
            jwt_secret: None,
        },
        audit: config::AuditConfig {
            enabled: false,
            path: String::new(),
        },
    };

    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    // Spawn the server in a background task
    let host = config.host.clone();
    let port = config.port;

    tokio::spawn(async move {
        if let Err(e) = run_http_server_with_shutdown(server_config, shutdown_rx).await {
            tracing::error!("Embedded server error: {}", e);
        }
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(EmbeddedServer {
        shutdown_tx: Some(shutdown_tx),
        port,
        host,
    })
}

/// Run HTTP server with graceful shutdown support
#[cfg(feature = "http")]
async fn run_http_server_with_shutdown(
    config: ServerConfig,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (app, _state) = crate::transport::http::build_app(&config).await?;

    let addr = config.http.addr;
    tracing::info!("Starting embedded HTTP server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
            tracing::info!("Embedded server shutting down");
        })
        .await?;

    Ok(())
}

#[cfg(feature = "http")]
fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "::1" | "localhost")
}

/// Initialize logging for the embedded server
/// Call this once at application startup if you want server logs
pub fn init_embedded_logging() {
    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "p2a_mcp=info,p2a_core=info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .try_init();
}
