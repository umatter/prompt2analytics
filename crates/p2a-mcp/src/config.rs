//! Server configuration for p2a-mcp.
//!
//! Supports configuration via CLI arguments, environment variables,
//! and configuration files.
//!
//! This module also decides the deployment posture (design section 1):
//! [`DeploymentMode`] comes from the access token and the open-mode flag
//! (never from the bind address alone), and [`ResourceProfile`] says whether
//! the hosted resource floor applies.

use anyhow::Context;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Whether an environment variable is set to a truthy value
/// (`1`, `true`, `yes`, `on`, case-insensitive). Empty/unset/other = false.
#[cfg(feature = "http")]
fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

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

/// How the server is exposed, decided from explicit configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentMode {
    /// stdio, or an HTTP server on a loopback address with no token and no
    /// open-mode flag (desktop, development, evaluation runs).
    Local,
    /// An HTTP server with an access token.
    Authenticated,
    /// An HTTP server without a token that is open to the public: either
    /// `--allow-unauthenticated` is set or the bind address is not loopback.
    OpenPublic,
}

/// Resource profile: whether the hosted caps, deadlines and quotas apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceProfile {
    /// No caps, deadlines or quotas beyond explicit knobs.
    Off,
    /// The floor for a public deployment (design section 2.6 and 4).
    Hosted,
}

/// Whether each tool call runs inside a per-session data directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum SessionDirsSetting {
    /// On in open public mode, off otherwise.
    #[default]
    Auto,
    /// Always on.
    On,
    /// Always off.
    Off,
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

    /// Shared bearer token required on all `/api` routes (HTTP transport).
    ///
    /// When set, every request must carry `Authorization: Bearer <token>`;
    /// `/health` and CORS preflight are exempt. A token is REQUIRED for any
    /// non-loopback bind unless `--allow-unauthenticated` is set. Distribute
    /// the token to authorized clients out-of-band.
    #[cfg(feature = "http")]
    #[arg(long, env = "P2A_ACCESS_TOKEN")]
    pub access_token: Option<String>,

    /// Run as an open public server with no access token.
    ///
    /// This is the "open but hardened" deployment mode for a public
    /// bring-your-own-key demo: there is no login, and per-visitor isolation
    /// relies on unguessable, non-enumerable session IDs. On any bind
    /// address, loopback included (for a server behind a same-host reverse
    /// proxy), the flag switches the server to open public mode: the
    /// session-list and file-browser endpoints are NOT exposed, and the hosted
    /// resource profile (caps, deadlines, quotas) applies. Abuse and DoS
    /// protection beyond that is expected to come from an edge layer (a
    /// reverse proxy or CDN rate limit). Without this flag, the server refuses
    /// to bind a non-loopback address that has no access token, to avoid an
    /// accidental open deployment.
    ///
    /// The `P2A_ALLOW_UNAUTHENTICATED` environment variable enables the same
    /// behavior and accepts any truthy value (`1`, `true`, `yes`, `on`). It is
    /// read in `from_args` rather than via clap, because clap's env-backed
    /// bools only accept `true`/`false` (so `=1` would error at startup).
    #[cfg(feature = "http")]
    #[arg(long)]
    pub allow_unauthenticated: bool,

    /// Keep the enumeration routes (`GET /api/sessions`, `GET /api/files`) on
    /// a local server. They are always on with an access token and never on
    /// in open public mode. Env: `P2A_EXPOSE_ENUMERATION` (any truthy value).
    #[cfg(feature = "http")]
    #[arg(long)]
    pub expose_enumeration: bool,

    /// Register the `/ws` route in open public mode, where it is off by
    /// default. Env: `P2A_ENABLE_WS` (any truthy value).
    #[cfg(feature = "http")]
    #[arg(long)]
    pub enable_ws: bool,

    /// Per-session data directories: `auto` (on in open public mode), `on`
    /// or `off`.
    #[cfg(feature = "http")]
    #[arg(long, value_enum, default_value = "auto", env = "P2A_SESSION_DIRS")]
    pub session_dirs: SessionDirsSetting,

    /// Deprecated: superseded by `--access-token` / `P2A_ACCESS_TOKEN`. This
    /// flag no longer enables any access control on its own and is kept only
    /// for backward compatibility.
    #[cfg(feature = "auth")]
    #[arg(long, env = "P2A_AUTH_ENABLED")]
    pub auth_enabled: bool,

    /// Deprecated: unused. JWT authentication is not implemented; use the
    /// shared bearer token (`--access-token`) instead.
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

    /// Resource profile: `off` or `hosted`. Open public mode always uses
    /// `hosted`; `off` is refused there.
    #[arg(long, value_enum, env = "P2A_LIMITS")]
    pub limits: Option<ResourceProfile>,

    /// Raise or lower one request limit, as `key=value` (repeatable;
    /// comma-separated in the environment variable).
    #[arg(
        long = "limit-override",
        env = "P2A_LIMIT_OVERRIDES",
        value_delimiter = ','
    )]
    pub limit_overrides: Vec<String>,

    /// Async runtime worker threads (default: max(2, CPUs)).
    #[arg(long, env = "P2A_WORKER_THREADS")]
    pub worker_threads: Option<usize>,

    /// Tool calls that may run at once, server-wide (default: CPUs).
    #[arg(long, env = "P2A_TOOL_CONCURRENCY")]
    pub tool_concurrency: Option<usize>,

    /// Tool calls that may run at once in one session (default: CPUs, or 1
    /// when a dataset quota applies; hosted: 1).
    #[arg(long, env = "P2A_SESSION_CONCURRENCY")]
    pub session_concurrency: Option<usize>,

    /// Seconds a tool call may wait for a free slot before it is refused
    /// (default: 300; hosted: 10).
    #[arg(long, env = "P2A_TOOL_QUEUE_WAIT_SECS")]
    pub tool_queue_wait_secs: Option<u64>,

    /// Wall-clock limit per tool call in seconds, 0 = none (default: none;
    /// hosted: 60).
    #[arg(long, env = "P2A_TOOL_TIMEOUT_SECS")]
    pub tool_timeout_secs: Option<u64>,

    /// Exit (code 70) when every tool slot has been stuck past its deadline
    /// for this many seconds, 0 = never (default: never; hosted: 300).
    #[arg(long, env = "P2A_STUCK_EXIT_SECS")]
    pub stuck_exit_secs: Option<u64>,

    /// Refuse new tool calls while the process RSS is above this many MiB,
    /// 0 = never (default: never; hosted: 700).
    #[arg(long, env = "P2A_MEMORY_SOFT_LIMIT_MB")]
    pub memory_soft_limit_mb: Option<u64>,

    /// In-memory dataset MiB per session, 0 = unlimited (default: unlimited;
    /// hosted: 200).
    #[arg(long, env = "P2A_SESSION_DATA_MB")]
    pub session_data_mb: Option<u64>,

    /// In-memory dataset MiB over all sessions, 0 = unlimited (default:
    /// unlimited; hosted: 500).
    #[arg(long, env = "P2A_TOTAL_DATA_MB")]
    pub total_data_mb: Option<u64>,

    /// Largest single buffer a computation may allocate, in MiB, 0 =
    /// unlimited (default: unlimited; hosted: 256).
    #[arg(long, env = "P2A_MAX_ALLOC_MB")]
    pub max_alloc_mb: Option<u64>,

    /// When the session table is full, evict the least recently used session
    /// idle for at least this many minutes, 0 = never evict (default: never;
    /// hosted: 10).
    #[arg(long, env = "P2A_SESSION_EVICT_IDLE_MINUTES")]
    pub session_evict_idle_minutes: Option<u64>,
}

/// Raw execution-governor knobs (design section 2.6). `None` means the
/// profile default; for the timeout, watchdog, byte and eviction knobs an
/// explicit `0` means none/off. The governor resolves them
/// (`exec::ExecConfig::from_config`).
#[derive(Debug, Clone, Default)]
pub struct ExecSettings {
    pub worker_threads: Option<usize>,
    pub tool_concurrency: Option<usize>,
    pub session_concurrency: Option<usize>,
    pub tool_queue_wait_secs: Option<u64>,
    pub tool_timeout_secs: Option<u64>,
    pub stuck_exit_secs: Option<u64>,
    pub memory_soft_limit_mb: Option<u64>,
    pub session_data_mb: Option<u64>,
    pub total_data_mb: Option<u64>,
    pub max_alloc_mb: Option<u64>,
    pub session_evict_idle_minutes: Option<u64>,
}

/// Raw request-limit settings: the configured profile (`None` = decided by
/// the deployment mode) and `key=value` overrides, interpreted by the limits
/// module.
#[derive(Debug, Clone, Default)]
pub struct LimitsSettings {
    pub profile: Option<ResourceProfile>,
    pub overrides: Vec<String>,
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
    pub exec: ExecSettings,
    pub limits: LimitsSettings,
}

/// HTTP transport configuration.
#[cfg(feature = "http")]
#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub addr: SocketAddr,
    pub cors_permissive: bool,
    pub cors_origins: Vec<String>,
    /// Shared bearer token required on all `/api` routes (None = no auth,
    /// permitted only for loopback binds or when `allow_unauthenticated`).
    pub access_token: Option<String>,
    /// Open public mode without a token (`--allow-unauthenticated`), on any
    /// bind address; required for a non-loopback bind without a token.
    pub allow_unauthenticated: bool,
    /// Keep the enumeration routes on a local server (`--expose-enumeration`).
    pub expose_enumeration: bool,
    /// Register `/ws` in open public mode (`--enable-ws`).
    pub enable_ws: bool,
    /// Per-session data directories (`--session-dirs`).
    pub session_dirs: SessionDirsSetting,
    /// Database path for persistence (None = in-memory)
    #[cfg(feature = "db")]
    pub db_path: Option<String>,
}

#[cfg(feature = "http")]
impl HttpConfig {
    /// The deployment mode from the token, the open-mode flag and the bind
    /// address: a token means [`DeploymentMode::Authenticated`]; otherwise
    /// `--allow-unauthenticated` or a non-loopback bind means
    /// [`DeploymentMode::OpenPublic`]; otherwise [`DeploymentMode::Local`].
    pub fn deployment_mode(&self) -> DeploymentMode {
        if self.access_token.is_some() {
            DeploymentMode::Authenticated
        } else if self.allow_unauthenticated || !self.addr.ip().is_loopback() {
            DeploymentMode::OpenPublic
        } else {
            DeploymentMode::Local
        }
    }

    /// Kept for source compatibility; now `deployment_mode() == OpenPublic`
    /// (no longer keyed to the bind address alone, MCP-5). In this mode the
    /// enumeration endpoints (`GET /api/sessions`, `GET /api/files`) are not
    /// registered.
    pub fn is_open_unauthenticated(&self) -> bool {
        self.deployment_mode() == DeploymentMode::OpenPublic
    }
}

/// The bind guard shared by every HTTP start path: a non-loopback bind needs
/// an access token or `--allow-unauthenticated`. It compares the parsed IP,
/// never a host string.
#[cfg(feature = "http")]
pub fn check_bind_policy(http: &HttpConfig) -> Result<(), String> {
    let addr = http.addr;
    if !addr.ip().is_loopback() && http.access_token.is_none() && !http.allow_unauthenticated {
        return Err(format!(
            "refusing to bind {addr}: a non-loopback bind requires an access token. \
             Set P2A_ACCESS_TOKEN (or --access-token) for a private/authenticated deployment, \
             or set P2A_ALLOW_UNAUTHENTICATED=1 (--allow-unauthenticated) for an intentional \
             open bring-your-own-key demo, or bind 127.0.0.1 for local development."
        ));
    }
    Ok(())
}

/// Session management configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub ttl_minutes: u64,
    pub max_sessions: usize,
    /// When the table is full, evict the least recently used session idle for
    /// at least this many minutes (`None` = never evict), as the
    /// `SessionManager` applies it. The HTTP app resolves the value it passes
    /// with `ServerConfig::resolved_evict_idle_minutes`: the
    /// `--session-evict-idle-minutes` knob, else a value set here, else the
    /// hosted profile's default.
    pub evict_idle_minutes: Option<u64>,
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

/// Default session eviction threshold in the hosted profile, in minutes.
const HOSTED_EVICT_IDLE_MINUTES: u64 = 10;
/// Default per-allocation ceiling in the hosted profile, in MiB.
const HOSTED_MAX_ALLOC_MB: u64 = 256;
/// Largest accepted value of a governor knob given in seconds (one year).
const MAX_KNOB_SECS: u64 = 31_536_000;
/// Largest accepted session eviction threshold, in minutes (one year).
const MAX_EVICT_IDLE_MINUTES: u64 = 525_600;

/// MiB to bytes as `usize`, saturating.
fn mib_to_usize(mb: u64) -> usize {
    usize::try_from(mb.saturating_mul(1 << 20)).unwrap_or(usize::MAX)
}

/// The number of CPUs available to this process (1 if unknown).
pub(crate) fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    pub fn from_args(args: CliArgs) -> anyhow::Result<Self> {
        let config = Self {
            transport: args.transport,
            #[cfg(feature = "http")]
            http: HttpConfig {
                addr: format!("{}:{}", args.host, args.port)
                    .parse()
                    .context(format!("Invalid host:port '{}:{}'", args.host, args.port))?,
                cors_permissive: args.cors_permissive,
                cors_origins: args.cors_origins,
                access_token: args
                    .access_token
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty()),
                allow_unauthenticated: args.allow_unauthenticated
                    || env_flag_enabled("P2A_ALLOW_UNAUTHENTICATED"),
                expose_enumeration: args.expose_enumeration
                    || env_flag_enabled("P2A_EXPOSE_ENUMERATION"),
                enable_ws: args.enable_ws || env_flag_enabled("P2A_ENABLE_WS"),
                session_dirs: args.session_dirs,
                #[cfg(feature = "db")]
                db_path: args.db_path,
            },
            session: SessionConfig {
                ttl_minutes: args.session_ttl_minutes,
                max_sessions: args.max_sessions,
                evict_idle_minutes: None,
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
            exec: ExecSettings {
                worker_threads: args.worker_threads,
                tool_concurrency: args.tool_concurrency,
                session_concurrency: args.session_concurrency,
                tool_queue_wait_secs: args.tool_queue_wait_secs,
                tool_timeout_secs: args.tool_timeout_secs,
                stuck_exit_secs: args.stuck_exit_secs,
                memory_soft_limit_mb: args.memory_soft_limit_mb,
                session_data_mb: args.session_data_mb,
                total_data_mb: args.total_data_mb,
                max_alloc_mb: args.max_alloc_mb,
                session_evict_idle_minutes: args.session_evict_idle_minutes,
            },
            limits: LimitsSettings {
                profile: args.limits,
                overrides: args.limit_overrides,
            },
        };
        Ok(config)
    }

    /// This configuration with its transport set to HTTP. `ServerConfig`'s
    /// default transport is stdio, which reads as local mode with no bind
    /// guard; so the HTTP start paths (`transport::http::bind`, `serve`, the
    /// app builders and `create_router`) and the governor resolver
    /// (`exec::ExecConfig::from_config`) derive the deployment mode, resource
    /// profile and bind guard from the HTTP settings whatever `transport`
    /// says.
    #[cfg(feature = "http")]
    pub(crate) fn as_http(&self) -> ServerConfig {
        ServerConfig {
            transport: TransportType::Http,
            ..self.clone()
        }
    }

    /// The deployment mode: stdio (or a build without the `http` feature) is
    /// always [`DeploymentMode::Local`]; HTTP follows
    /// `HttpConfig::deployment_mode`.
    pub fn deployment_mode(&self) -> DeploymentMode {
        match self.transport {
            TransportType::Stdio => DeploymentMode::Local,
            #[cfg(feature = "http")]
            TransportType::Http => self.http.deployment_mode(),
        }
    }

    /// The resource profile: always [`ResourceProfile::Hosted`] in open public
    /// mode (fail closed), otherwise the configured profile or
    /// [`ResourceProfile::Off`].
    pub fn resource_profile(&self) -> ResourceProfile {
        if self.deployment_mode() == DeploymentMode::OpenPublic {
            ResourceProfile::Hosted
        } else {
            self.limits.profile.unwrap_or(ResourceProfile::Off)
        }
    }

    /// Startup refusals: the bind guard, `P2A_LIMITS=off` in open public mode,
    /// zero threads or concurrency, and governor knobs too large to apply
    /// (each refusal names the flag, its environment variable and the bound).
    pub fn validate(&self) -> anyhow::Result<()> {
        #[cfg(feature = "http")]
        if self.transport == TransportType::Http {
            check_bind_policy(&self.http).map_err(anyhow::Error::msg)?;
        }
        if self.limits.profile == Some(ResourceProfile::Off)
            && self.deployment_mode() == DeploymentMode::OpenPublic
        {
            anyhow::bail!(
                "P2A_LIMITS=off (--limits off) is not allowed for an open public server \
                 (no access token): the hosted resource profile always applies there. \
                 Raise individual request limits with --limit-override \
                 (P2A_LIMIT_OVERRIDES) and governor limits with their flags instead (for \
                 example --tool-timeout-secs or --session-data-mb)."
            );
        }
        for (flag, value) in [
            (
                "--worker-threads (P2A_WORKER_THREADS)",
                self.exec.worker_threads,
            ),
            (
                "--tool-concurrency (P2A_TOOL_CONCURRENCY)",
                self.exec.tool_concurrency,
            ),
            (
                "--session-concurrency (P2A_SESSION_CONCURRENCY)",
                self.exec.session_concurrency,
            ),
        ] {
            if value == Some(0) {
                anyhow::bail!("{flag} must be at least 1");
            }
        }
        let max_permits = tokio::sync::Semaphore::MAX_PERMITS;
        for (flag, value) in [
            (
                "--tool-concurrency (P2A_TOOL_CONCURRENCY)",
                self.exec.tool_concurrency,
            ),
            (
                "--session-concurrency (P2A_SESSION_CONCURRENCY)",
                self.exec.session_concurrency,
            ),
        ] {
            if value.is_some_and(|v| v > max_permits) {
                anyhow::bail!("{flag} must be at most {max_permits}");
            }
        }
        for (flag, value, max, unit) in [
            (
                "--tool-queue-wait-secs (P2A_TOOL_QUEUE_WAIT_SECS)",
                self.exec.tool_queue_wait_secs,
                MAX_KNOB_SECS,
                "seconds (one year)",
            ),
            (
                "--tool-timeout-secs (P2A_TOOL_TIMEOUT_SECS)",
                self.exec.tool_timeout_secs,
                MAX_KNOB_SECS,
                "seconds (one year; 0 means no limit)",
            ),
            (
                "--stuck-exit-secs (P2A_STUCK_EXIT_SECS)",
                self.exec.stuck_exit_secs,
                MAX_KNOB_SECS,
                "seconds (one year; 0 means never)",
            ),
            (
                "--session-evict-idle-minutes (P2A_SESSION_EVICT_IDLE_MINUTES)",
                self.exec.session_evict_idle_minutes,
                MAX_EVICT_IDLE_MINUTES,
                "minutes (one year; 0 means never evict)",
            ),
        ] {
            if value.is_some_and(|v| v > max) {
                anyhow::bail!("{flag} must be at most {max} {unit}");
            }
        }
        Ok(())
    }

    /// Async runtime worker threads: the explicit setting, else max(2, CPUs),
    /// so `/health` and keep-alives are served on a one-CPU machine while
    /// tool work runs on the blocking pool.
    pub fn worker_threads(&self) -> usize {
        self.exec
            .worker_threads
            .filter(|&n| n > 0)
            .unwrap_or_else(|| available_cpus().max(2))
    }

    /// The per-allocation ceiling for p2a-core: the explicit `--max-alloc-mb`
    /// (`0` = unlimited), else 256 MiB in the hosted profile and unlimited
    /// otherwise. The stdio transport applies it too.
    pub fn alloc_ceiling_bytes(&self) -> Option<usize> {
        match self.exec.max_alloc_mb {
            Some(0) => None,
            Some(mb) => Some(mib_to_usize(mb)),
            None => (self.resource_profile() == ResourceProfile::Hosted)
                .then(|| mib_to_usize(HOSTED_MAX_ALLOC_MB)),
        }
    }

    /// The eviction threshold the HTTP app gives its `SessionManager`: the
    /// explicit `--session-evict-idle-minutes` (`0` = never), else
    /// `session.evict_idle_minutes` when a caller set it, else 10 minutes in
    /// the hosted profile and never otherwise. It is resolved where the app
    /// is built, so a programmatic open public configuration gets the hosted
    /// default too.
    pub(crate) fn resolved_evict_idle_minutes(&self) -> Option<u64> {
        match self.exec.session_evict_idle_minutes {
            Some(0) => None,
            Some(m) => Some(m),
            None => self.session.evict_idle_minutes.or_else(|| {
                (self.resource_profile() == ResourceProfile::Hosted)
                    .then_some(HOSTED_EVICT_IDLE_MINUTES)
            }),
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
                access_token: None,
                allow_unauthenticated: false,
                expose_enumeration: false,
                enable_ws: false,
                session_dirs: SessionDirsSetting::Auto,
                #[cfg(feature = "db")]
                db_path: None,
            },
            session: SessionConfig {
                ttl_minutes: 60,
                max_sessions: 100,
                evict_idle_minutes: None,
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
            exec: ExecSettings::default(),
            limits: LimitsSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_threads_default_is_at_least_two() {
        assert!(ServerConfig::default().worker_threads() >= 2);
    }

    #[test]
    fn stdio_is_local_and_off() {
        let c = ServerConfig::default();
        assert_eq!(c.deployment_mode(), DeploymentMode::Local);
        assert_eq!(c.resource_profile(), ResourceProfile::Off);
        assert_eq!(c.alloc_ceiling_bytes(), None);
        assert!(c.validate().is_ok());
    }
}

#[cfg(all(test, feature = "http"))]
mod open_mode_tests {
    use super::*;

    fn http(addr: &str, token: Option<&str>) -> HttpConfig {
        cfg(addr, token, false).http
    }

    fn cfg(addr: &str, token: Option<&str>, allow: bool) -> ServerConfig {
        let mut c = ServerConfig {
            transport: TransportType::Http,
            ..Default::default()
        };
        c.http.addr = addr.parse().unwrap();
        c.http.access_token = token.map(str::to_string);
        c.http.allow_unauthenticated = allow;
        c
    }

    #[test]
    fn open_only_when_public_and_tokenless() {
        // Public bind, no token -> open (enumeration endpoints hidden).
        assert!(http("0.0.0.0:8080", None).is_open_unauthenticated());
        // Public bind WITH a token -> not "open" (auth-gated).
        assert!(!http("0.0.0.0:8080", Some("secret")).is_open_unauthenticated());
        // Loopback, no token -> trusted local, not "open".
        assert!(!http("127.0.0.1:8080", None).is_open_unauthenticated());
        assert!(!http("[::1]:8080", None).is_open_unauthenticated());
    }

    #[test]
    fn deployment_mode_table() {
        use DeploymentMode::*;
        assert_eq!(cfg("127.0.0.1:8080", None, false).deployment_mode(), Local);
        assert_eq!(
            cfg("127.0.0.1:8080", None, true).deployment_mode(),
            OpenPublic
        ); // MCP-5 repro
        assert_eq!(
            cfg("0.0.0.0:8080", None, true).deployment_mode(),
            OpenPublic
        );
        assert_eq!(
            cfg("0.0.0.0:8080", Some("t"), false).deployment_mode(),
            Authenticated
        );
        assert_eq!(
            cfg("127.0.0.1:8080", Some("t"), true).deployment_mode(),
            Authenticated
        );
        assert_eq!(ServerConfig::default().deployment_mode(), Local); // stdio
        assert!(
            cfg("127.0.0.1:8080", None, true)
                .http
                .is_open_unauthenticated()
        ); // was false
    }

    #[test]
    fn resource_profile_rule() {
        use ResourceProfile::*;
        assert_eq!(cfg("127.0.0.1:8080", None, false).resource_profile(), Off);
        assert_eq!(
            cfg("0.0.0.0:8080", Some("t"), false).resource_profile(),
            Off
        );
        assert_eq!(cfg("0.0.0.0:8080", None, true).resource_profile(), Hosted);
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.limits.profile = Some(Hosted);
        assert_eq!(c.resource_profile(), Hosted);
        let mut c = cfg("0.0.0.0:8080", None, true);
        c.limits.profile = Some(Off);
        assert_eq!(c.resource_profile(), Hosted); // fail closed
        assert!(
            c.validate()
                .unwrap_err()
                .to_string()
                .contains("P2A_LIMITS=off")
        );
    }

    #[test]
    fn bind_policy_uses_parsed_ip() {
        let ok = |a, t, f| check_bind_policy(&cfg(a, t, f).http).is_ok();
        assert!(
            ok("127.0.0.1:80", None, false)
                && ok("127.0.0.2:80", None, false)
                && ok("[::1]:80", None, false)
        );
        assert!(
            !ok("0.0.0.0:80", None, false)
                && !ok("[::]:80", None, false)
                && !ok("192.168.1.5:80", None, false)
        );
        assert!(ok("0.0.0.0:80", Some("t"), false) && ok("0.0.0.0:80", None, true));
        let e = cfg("0.0.0.0:80", None, false)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(e.contains("requires an access token"), "{e}");
    }

    #[test]
    fn zero_concurrency_is_refused() {
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.exec.tool_concurrency = Some(0);
        assert!(
            c.validate()
                .unwrap_err()
                .to_string()
                .contains("tool-concurrency")
        );
    }

    #[test]
    fn cli_parses_new_knobs() {
        let args = CliArgs::try_parse_from([
            "p2a-mcp",
            "--transport",
            "http",
            "--limits",
            "hosted",
            "--tool-timeout-secs",
            "5",
            "--limit-override",
            "replications=20000",
            "--limit-override",
            "folds=10",
            "--session-dirs",
            "on",
            "--expose-enumeration",
            "--enable-ws",
            "--worker-threads",
            "3",
        ])
        .unwrap();
        let c = ServerConfig::from_args(args).unwrap();
        assert_eq!(c.limits.profile, Some(ResourceProfile::Hosted));
        assert_eq!(c.limits.overrides, vec!["replications=20000", "folds=10"]);
        assert_eq!(c.exec.tool_timeout_secs, Some(5));
        assert_eq!(c.http.session_dirs, SessionDirsSetting::On);
        assert!(c.http.expose_enumeration && c.http.enable_ws);
        assert_eq!(c.worker_threads(), 3);
        assert_eq!(c.session.evict_idle_minutes, None); // resolved where the app is built
        assert_eq!(c.resolved_evict_idle_minutes(), Some(10)); // the hosted default
    }

    #[test]
    fn eviction_threshold_resolution() {
        let open = cfg("0.0.0.0:8080", None, true);
        assert_eq!(open.resolved_evict_idle_minutes(), Some(10));
        assert_eq!(
            cfg("127.0.0.1:8080", None, false).resolved_evict_idle_minutes(),
            None
        );
        let mut c = open.clone();
        c.exec.session_evict_idle_minutes = Some(0);
        assert_eq!(c.resolved_evict_idle_minutes(), None);
        let mut c = open.clone();
        c.exec.session_evict_idle_minutes = Some(3);
        c.session.evict_idle_minutes = Some(30);
        assert_eq!(c.resolved_evict_idle_minutes(), Some(3)); // the knob wins
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.session.evict_idle_minutes = Some(30);
        assert_eq!(c.resolved_evict_idle_minutes(), Some(30));
    }

    #[test]
    fn off_refusal_points_at_limit_overrides() {
        let mut c = cfg("0.0.0.0:8080", None, true);
        c.limits.profile = Some(ResourceProfile::Off);
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("--limit-override (P2A_LIMIT_OVERRIDES)"), "{e}");
    }

    #[test]
    fn out_of_range_knobs_are_refused() {
        let max_permits = tokio::sync::Semaphore::MAX_PERMITS;
        type Set = fn(&mut ServerConfig);
        let refused: [(Set, &str, &str); 6] = [
            (
                |c| c.exec.tool_timeout_secs = Some(31_536_001),
                "--tool-timeout-secs (P2A_TOOL_TIMEOUT_SECS)",
                "31536000",
            ),
            (
                |c| c.exec.tool_timeout_secs = Some(u64::MAX),
                "--tool-timeout-secs (P2A_TOOL_TIMEOUT_SECS)",
                "31536000",
            ),
            (
                |c| c.exec.stuck_exit_secs = Some(31_536_001),
                "--stuck-exit-secs (P2A_STUCK_EXIT_SECS)",
                "31536000",
            ),
            (
                |c| c.exec.tool_queue_wait_secs = Some(u64::MAX),
                "--tool-queue-wait-secs (P2A_TOOL_QUEUE_WAIT_SECS)",
                "31536000",
            ),
            (
                |c| c.exec.session_evict_idle_minutes = Some(525_601),
                "--session-evict-idle-minutes (P2A_SESSION_EVICT_IDLE_MINUTES)",
                "525600",
            ),
            (
                |c| c.exec.session_evict_idle_minutes = Some(u64::MAX),
                "--session-evict-idle-minutes (P2A_SESSION_EVICT_IDLE_MINUTES)",
                "525600",
            ),
        ];
        for (set, flag, max) in refused {
            let mut c = cfg("127.0.0.1:8080", None, false);
            set(&mut c);
            let e = c.validate().unwrap_err().to_string();
            assert!(e.contains(flag) && e.contains(max), "{e}");
        }
        let over = Some(max_permits + 1);
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.exec.tool_concurrency = over;
        let e = c.validate().unwrap_err().to_string();
        assert!(
            e.contains("--tool-concurrency (P2A_TOOL_CONCURRENCY)")
                && e.contains(&max_permits.to_string()),
            "{e}"
        );
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.exec.session_concurrency = over;
        let e = c.validate().unwrap_err().to_string();
        assert!(
            e.contains("--session-concurrency (P2A_SESSION_CONCURRENCY)")
                && e.contains(&max_permits.to_string()),
            "{e}"
        );
        // The bounds themselves are accepted.
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.exec.tool_timeout_secs = Some(31_536_000);
        c.exec.stuck_exit_secs = Some(31_536_000);
        c.exec.tool_queue_wait_secs = Some(31_536_000);
        c.exec.session_evict_idle_minutes = Some(525_600);
        c.exec.tool_concurrency = Some(max_permits);
        c.exec.session_concurrency = Some(max_permits);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn alloc_ceiling_follows_profile_and_knob() {
        assert_eq!(
            cfg("0.0.0.0:8080", None, true).alloc_ceiling_bytes(),
            Some(256 << 20)
        );
        let mut c = cfg("0.0.0.0:8080", None, true);
        c.exec.max_alloc_mb = Some(0);
        assert_eq!(c.alloc_ceiling_bytes(), None);
        let mut c = cfg("127.0.0.1:8080", None, false);
        c.exec.max_alloc_mb = Some(64);
        assert_eq!(c.alloc_ceiling_bytes(), Some(64 << 20));
    }
}
