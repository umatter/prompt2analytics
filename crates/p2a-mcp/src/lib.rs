//! p2a-mcp library interface
//!
//! This module exposes the MCP server functionality as a library,
//! allowing it to be embedded in other applications (like p2a-dioxus desktop).

/// Stack of each runtime thread. An unoptimised build's futures (the session database's most of all)
/// need more than the 2 MiB default.
pub const RUNTIME_STACK_BYTES: usize = 16 * 1024 * 1024;

pub mod audit;
pub mod config;
#[cfg(feature = "db")]
pub mod db;
#[cfg(feature = "http")]
pub mod exec;
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

/// Configuration for the embedded server. Build it from
/// [`EmbeddedServerConfig::default`] and the `with_*` methods: it is
/// `#[non_exhaustive]`, so a field added later does not break callers.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Minutes a session may stay idle before it expires, with its
    /// conversations (default: 60)
    pub session_ttl_minutes: u64,
}

impl Default for EmbeddedServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            cors_permissive: false,
            db_path: None,
            session_ttl_minutes: 60,
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

    /// Set the session lifetime: minutes a session may stay idle before it
    /// expires and its conversations are deleted
    pub fn with_session_ttl_minutes(mut self, minutes: u64) -> Self {
        self.session_ttl_minutes = minutes;
        self
    }
}

/// The server configuration the embedded server runs with, at `addr`.
#[cfg(feature = "http")]
fn embedded_server_config(
    config: &EmbeddedServerConfig,
    addr: std::net::SocketAddr,
) -> ServerConfig {
    ServerConfig {
        transport: TransportType::Http,
        http: HttpConfig {
            addr,
            cors_permissive: config.cors_permissive,
            cors_origins: vec![],
            // The embedded server is intended for local (loopback) desktop use
            // with no shared token. `bind` applies the same guard as the
            // binary, so a non-loopback host is refused.
            access_token: None,
            allow_unauthenticated: false,
            expose_enumeration: false,
            enable_ws: false,
            session_dirs: config::SessionDirsSetting::Auto,
            #[cfg(feature = "db")]
            db_path: config.db_path.clone(),
        },
        session: SessionConfig {
            ttl_minutes: config.session_ttl_minutes,
            max_sessions: 100,
            evict_idle_minutes: None,
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
        exec: config::ExecSettings::default(),
        limits: config::LimitsSettings::default(),
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
/// Binds before returning, so a refused or failed bind is reported here and
/// the handle carries the real port (also when `port` is 0). A start next to
/// a server of this process that runs under other request limits or process
/// policies is refused here too (`transport::http::hold_limits`). The
/// embedded server is always a local one (the off profile, no session
/// directories), so it is refused while, for example, an open public server
/// of the same process still serves anything. The server uses the same
/// guarded start path as `p2a-mcp --transport http`.
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
    if config.cors_permissive && !addr.ip().is_loopback() {
        tracing::warn!(
            "cors_permissive=true combined with non-loopback host {:?} exposes the server to \
             cross-origin requests from any web page; set cors_permissive=false in production",
            config.host
        );
    }

    let server_config = embedded_server_config(&config, addr);

    // Bind first, so the guard's refusal or a bind error returns to the caller.
    let listener = transport::http::bind(&server_config).await?;
    let bound = listener.local_addr()?;
    // Take the hold `serve` takes, for the configuration as `serve` reads it
    // (at the bound address, validated), so that an invalid configuration, or
    // limits or process policies that conflict with a server already running
    // in this process, are refused here, not only in the spawned task's log.
    // `serve` shares this hold; the task drops it once `serve` has returned.
    let (held, _, _) = transport::http::hold_for_socket(&mut server_config.clone(), bound)?;
    let port = bound.port();

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tokio::spawn(async move {
        let _held = held;
        let shutdown = async {
            let _ = shutdown_rx.await;
            tracing::info!("Embedded server shutting down");
        };
        if let Err(e) = transport::http::serve(listener, server_config, shutdown).await {
            tracing::error!("Embedded server error: {}", e);
        }
    });

    Ok(EmbeddedServer {
        shutdown_tx: Some(shutdown_tx),
        port,
        host: config.host,
    })
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

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;

    #[test]
    fn embedded_config_carries_the_session_lifetime() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let default = embedded_server_config(&EmbeddedServerConfig::default(), addr);
        assert_eq!(default.session.ttl_minutes, 60);
        let thirty_days = EmbeddedServerConfig::default().with_session_ttl_minutes(30 * 24 * 60);
        assert_eq!(thirty_days.session_ttl_minutes, 43_200);
        let server = embedded_server_config(&thirty_days, addr);
        assert_eq!(server.session.ttl_minutes, 43_200);
        assert_eq!(server.http.addr, addr);
    }
}
