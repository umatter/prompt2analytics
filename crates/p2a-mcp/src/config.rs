//! Server configuration for p2a-mcp.
//!
//! Supports configuration via CLI arguments, environment variables,
//! and configuration files.

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Transport type for the MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TransportType {
    /// Standard input/output (default, for CLI/desktop integration)
    #[default]
    Stdio,
    /// HTTP REST API with optional WebSocket
    #[cfg(feature = "http")]
    Http,
}

/// CLI arguments for the MCP server.
#[derive(Parser, Debug)]
#[command(name = "p2a-mcp")]
#[command(about = "prompt2analytics MCP Server - Natural language data analytics")]
#[command(version)]
pub struct CliArgs {
    /// Transport type to use
    #[arg(
        short,
        long,
        value_enum,
        default_value = "stdio",
        env = "P2A_TRANSPORT"
    )]
    pub transport: TransportType,

    /// Host address for HTTP transport
    #[cfg(feature = "http")]
    #[arg(long, default_value = "127.0.0.1", env = "P2A_HOST")]
    pub host: String,

    /// Port for HTTP transport
    #[cfg(feature = "http")]
    #[arg(short, long, default_value = "8080", env = "P2A_PORT")]
    pub port: u16,

    /// Enable CORS for all origins (development mode)
    #[cfg(feature = "http")]
    #[arg(long, env = "P2A_CORS_PERMISSIVE")]
    pub cors_permissive: bool,

    /// Allowed CORS origins (comma-separated)
    #[cfg(feature = "http")]
    #[arg(long, env = "P2A_CORS_ORIGINS", value_delimiter = ',')]
    pub cors_origins: Vec<String>,

    /// Enable authentication
    #[cfg(feature = "auth")]
    #[arg(long, env = "P2A_AUTH_ENABLED")]
    pub auth_enabled: bool,

    /// JWT secret for authentication (required if auth is enabled)
    #[cfg(feature = "auth")]
    #[arg(long, env = "P2A_JWT_SECRET")]
    pub jwt_secret: Option<String>,

    /// Session timeout in minutes
    #[arg(long, default_value = "60", env = "P2A_SESSION_TTL")]
    pub session_ttl_minutes: u64,

    /// Maximum number of concurrent sessions
    #[arg(long, default_value = "100", env = "P2A_MAX_SESSIONS")]
    pub max_sessions: usize,

    /// Database path for persistence (when db feature is enabled)
    #[cfg(feature = "db")]
    #[arg(long, env = "P2A_DB_PATH")]
    pub db_path: Option<String>,

    /// Enable audit logging for tool calls
    #[arg(long, env = "P2A_AUDIT_LOG")]
    pub audit_log: bool,

    /// Path to audit log file (defaults to p2a-audit.log in current directory)
    #[arg(long, env = "P2A_AUDIT_PATH")]
    pub audit_path: Option<String>,
}

/// Server configuration derived from CLI args and environment.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub transport: TransportType,
    #[cfg(feature = "http")]
    pub http: HttpConfig,
    pub session: SessionConfig,
    #[cfg(feature = "auth")]
    pub auth: AuthConfig,
    pub audit: AuditConfig,
}

/// HTTP transport configuration.
#[cfg(feature = "http")]
#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub addr: SocketAddr,
    pub cors_permissive: bool,
    pub cors_origins: Vec<String>,
    /// Database path for persistence (None = in-memory)
    #[cfg(feature = "db")]
    pub db_path: Option<String>,
}

/// Session management configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub ttl_minutes: u64,
    pub max_sessions: usize,
}

/// Authentication configuration.
#[cfg(feature = "auth")]
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: Option<String>,
}

/// Audit logging configuration.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Whether audit logging is enabled
    pub enabled: bool,
    /// Path to the audit log file
    pub path: String,
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    pub fn from_args(args: CliArgs) -> Self {
        Self {
            transport: args.transport,
            #[cfg(feature = "http")]
            http: HttpConfig {
                addr: format!("{}:{}", args.host, args.port)
                    .parse()
                    .expect("Invalid host:port"),
                cors_permissive: args.cors_permissive,
                cors_origins: args.cors_origins,
                #[cfg(feature = "db")]
                db_path: args.db_path,
            },
            session: SessionConfig {
                ttl_minutes: args.session_ttl_minutes,
                max_sessions: args.max_sessions,
            },
            #[cfg(feature = "auth")]
            auth: AuthConfig {
                enabled: args.auth_enabled,
                jwt_secret: args.jwt_secret,
            },
            audit: AuditConfig {
                enabled: args.audit_log,
                path: args
                    .audit_path
                    .unwrap_or_else(|| "p2a-audit.log".to_string()),
            },
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            transport: TransportType::default(),
            #[cfg(feature = "http")]
            http: HttpConfig {
                addr: "127.0.0.1:8080".parse().unwrap(),
                cors_permissive: false,
                cors_origins: vec![],
                #[cfg(feature = "db")]
                db_path: None,
            },
            session: SessionConfig {
                ttl_minutes: 60,
                max_sessions: 100,
            },
            #[cfg(feature = "auth")]
            auth: AuthConfig {
                enabled: false,
                jwt_secret: None,
            },
            audit: AuditConfig {
                enabled: false,
                path: "p2a-audit.log".to_string(),
            },
        }
    }
}
