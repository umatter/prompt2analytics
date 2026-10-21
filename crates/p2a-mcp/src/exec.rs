//! Execution governor for HTTP-side tool calls (design section 2).
//!
//! Every HTTP-side tool execution (`POST /api/tools/{name}`, both LLM tool
//! executors and websocket `Tool` messages) goes through
//! [`ToolGovernor::call`], which applies, in order:
//!
//! 1. memory admission (refuse while the process RSS is above a soft limit);
//! 2. a per-session and a global permit, each with a bounded queue wait;
//! 3. execution on tokio's blocking pool, so the async runtime keeps serving
//!    `/health`, SSE keep-alives and other visitors;
//! 4. a per-call [`Budget`] installed for p2a-core's cooperative
//!    [`resource::checkpoint`]s, cancelled when the caller goes away;
//! 5. a wall-clock deadline (504) and panic isolation (500, session intact);
//! 6. per-session and total dataset-byte quotas that undo the call that
//!    crossed them.
//!
//! The permits, the in-flight registry entry and the session's active-call
//! count are moved into the blocking closure, so they are released when the
//! work actually stops, not when the HTTP response is sent. Work that ignores
//! its budget past the deadline stays counted as overdue until it ends; when
//! every permit is held by overdue work, [`ToolGovernor::health`] reports the
//! server unhealthy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use p2a_core::memory::format_bytes;
use p2a_core::resource::{self, Budget, CancelToken};
use rmcp::model::ErrorCode;
use serde::Serialize;
use tokio::sync::Semaphore;
use tracing::Instrument;

use crate::config::{
    DeploymentMode, ResourceProfile, ServerConfig, SessionDirsSetting, available_cpus,
};
use crate::server::AnalyticsServer;
use crate::session::{Session, SessionManager, datasets_bytes};
use crate::transport::http::{ApiResponse, ToolResult};

/// How long past its deadline cooperative work may take to stop before the
/// governor gives up waiting and answers 504.
pub const DEADLINE_GRACE: Duration = Duration::from_millis(250);

const SESSION_BUSY_RETRY_SECS: u64 = 2;
const SERVER_BUSY_RETRY_SECS: u64 = 5;
const MEMORY_PRESSURE_RETRY_SECS: u64 = 15;

/// Resolved governor settings (design section 2.6).
#[derive(Debug, Clone)]
pub struct ExecConfig {
    /// Global permits for tool execution.
    pub concurrency: usize,
    /// Permits per session.
    pub session_concurrency: usize,
    /// Maximum wait for a permit before refusing.
    pub queue_wait: Duration,
    /// Per-call wall clock; `None` = unlimited.
    pub deadline: Option<Duration>,
    /// Exit when every permit has been held by overdue work this long.
    pub stuck_exit: Option<Duration>,
    /// Refuse new calls while the process RSS is above this.
    pub rss_soft_limit_bytes: Option<u64>,
    /// In-memory dataset bytes per session (MCP-10).
    pub session_data_bytes: Option<u64>,
    /// In-memory dataset bytes over all sessions (MCP-10).
    pub total_data_bytes: Option<u64>,
    /// Installed via `p2a_core::resource::set_alloc_ceiling_bytes`.
    pub alloc_ceiling_bytes: Option<usize>,
    /// Enter the `path_jail` session scope per call.
    pub session_dirs: bool,
}

const MIB: u64 = 1 << 20;

impl Default for ExecConfig {
    /// The Off-profile values: no deadline, quotas or admission limit, and
    /// one permit per CPU.
    fn default() -> Self {
        let cpus = available_cpus();
        Self {
            concurrency: cpus,
            session_concurrency: cpus,
            queue_wait: Duration::from_secs(300),
            deadline: None,
            stuck_exit: None,
            rss_soft_limit_bytes: None,
            session_data_bytes: None,
            total_data_bytes: None,
            alloc_ceiling_bytes: None,
            session_dirs: false,
        }
    }
}

/// `Some(0)` means none; `None` takes the profile default.
fn zero_is_none(explicit: Option<u64>, default: Option<u64>) -> Option<u64> {
    match explicit {
        Some(0) => None,
        Some(v) => Some(v),
        None => default,
    }
}

impl ExecConfig {
    /// Resolve the governor settings from the configured knobs and the
    /// resource profile. Explicit knobs override the profile default.
    pub fn from_config(cfg: &ServerConfig) -> Self {
        let hosted = cfg.resource_profile() == ResourceProfile::Hosted;
        let cpus = available_cpus();
        let exec = &cfg.exec;
        let pick = |hosted_value: u64| hosted.then_some(hosted_value);
        Self {
            concurrency: exec
                .tool_concurrency
                .filter(|&n| n > 0)
                .unwrap_or(cpus.max(1)),
            session_concurrency: exec
                .session_concurrency
                .filter(|&n| n > 0)
                .unwrap_or(if hosted { 1 } else { cpus.max(1) }),
            queue_wait: Duration::from_secs(exec.tool_queue_wait_secs.unwrap_or(if hosted {
                10
            } else {
                300
            })),
            deadline: zero_is_none(exec.tool_timeout_secs, pick(60)).map(Duration::from_secs),
            stuck_exit: zero_is_none(exec.stuck_exit_secs, pick(300)).map(Duration::from_secs),
            rss_soft_limit_bytes: zero_is_none(exec.memory_soft_limit_mb, pick(700))
                .map(|mb| mb.saturating_mul(MIB)),
            session_data_bytes: zero_is_none(exec.session_data_mb, pick(200))
                .map(|mb| mb.saturating_mul(MIB)),
            total_data_bytes: zero_is_none(exec.total_data_mb, pick(500))
                .map(|mb| mb.saturating_mul(MIB)),
            alloc_ceiling_bytes: cfg.alloc_ceiling_bytes(),
            session_dirs: match cfg.http.session_dirs {
                SessionDirsSetting::Auto => cfg.deployment_mode() == DeploymentMode::OpenPublic,
                SessionDirsSetting::On => true,
                SessionDirsSetting::Off => false,
            },
        }
    }

    fn deadline_secs(&self) -> u64 {
        self.deadline.map(ceil_secs).unwrap_or(0)
    }
}

fn ceil_secs(d: Duration) -> u64 {
    d.as_secs() + u64::from(d.subsec_nanos() > 0)
}

/// Install the process-wide policies the governor relies on: the p2a-core
/// per-allocation ceiling and the per-session data directories switch.
/// Called once by the HTTP server at startup (not by `build_app`, so tests
/// can build several apps).
pub fn install_process_policies(cfg: &ExecConfig) {
    resource::set_alloc_ceiling_bytes(cfg.alloc_ceiling_bytes);
    crate::path_jail::configure_session_dirs(cfg.session_dirs);
}

/// Where a governed call came from (for logs).
#[derive(Debug, Clone, Copy)]
pub enum CallOrigin {
    /// `POST /api/tools/{name}`.
    Http,
    /// A tool call from the LLM tool loop.
    LlmTool,
    /// A websocket `Tool` message.
    WebSocket,
}

/// A snapshot of the governor's state, reported by `/health`.
#[derive(Debug, Clone, Serialize)]
pub struct GovernorHealth {
    /// Calls whose work is still running (including detached overdue work).
    pub in_flight: usize,
    /// Running calls past their deadline plus [`DEADLINE_GRACE`].
    pub overdue: usize,
    /// Global permits.
    pub capacity: usize,
    /// Process resident set size, where the platform reports it.
    pub rss_bytes: Option<u64>,
    /// False when every permit is held by overdue work.
    pub healthy: bool,
}

/// Why a governed tool call did not produce a tool result.
#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    /// No tool has this name.
    #[error("Unknown tool: {0}")]
    UnknownTool(String),
    /// The arguments do not match the tool's schema.
    #[error("{0}")]
    InvalidArguments(String),
    /// A request limit refused a parameter.
    #[error("{message}")]
    LimitExceeded { parameter: String, message: String },
    /// A dataset-memory quota refused the call; its changes were undone.
    #[error("{0}")]
    QuotaExceeded(String),
    /// This session's permits are all in use.
    #[error("This session is already running a tool call; retry in {retry_after_secs} s.")]
    SessionBusy { retry_after_secs: u64 },
    /// The server's permits are all in use.
    #[error("The server is busy with other requests; retry in {retry_after_secs} s.")]
    ServerBusy { retry_after_secs: u64 },
    /// The process is above its memory soft limit.
    #[error("The server is low on memory; retry in {retry_after_secs} s or use a smaller dataset.")]
    MemoryPressure { retry_after_secs: u64 },
    /// The call ran past the server's time limit and was stopped.
    #[error(
        "The tool call exceeded this server's time limit of {limit_secs} s and was stopped. \
         Use a smaller dataset or fewer iterations or replications."
    )]
    DeadlineExceeded { limit_secs: u64 },
    /// The tool panicked or failed in the server itself.
    #[error("{0}")]
    Internal(String),
}

impl ToolCallError {
    /// The HTTP status for this error.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::UnknownTool(_) | Self::InvalidArguments(_) => StatusCode::BAD_REQUEST,
            Self::LimitExceeded { .. } | Self::QuotaExceeded(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::SessionBusy { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::ServerBusy { .. } | Self::MemoryPressure { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::DeadlineExceeded { .. } => StatusCode::GATEWAY_TIMEOUT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The machine-readable code in the response body.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownTool(_) => "unknown_tool",
            Self::InvalidArguments(_) => "invalid_arguments",
            Self::LimitExceeded { .. } => "limit_exceeded",
            Self::QuotaExceeded(_) => "quota_exceeded",
            Self::SessionBusy { .. } => "session_busy",
            Self::ServerBusy { .. } => "server_busy",
            Self::MemoryPressure { .. } => "memory_pressure",
            Self::DeadlineExceeded { .. } => "deadline_exceeded",
            Self::Internal(_) => "internal",
        }
    }

    /// The `Retry-After` value in seconds, for the refusals worth retrying.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::SessionBusy { retry_after_secs }
            | Self::ServerBusy { retry_after_secs }
            | Self::MemoryPressure { retry_after_secs } => Some(*retry_after_secs),
            _ => None,
        }
    }

    /// Map a handler's `ErrorData`: invalid parameters stay the caller's
    /// problem (400); anything else is a server-side failure (500).
    pub fn from_handler_error(tool: &str, e: rmcp::ErrorData) -> Self {
        if e.code == ErrorCode::INVALID_PARAMS {
            Self::InvalidArguments(e.message.to_string())
        } else {
            Self::Internal(format!("Tool '{tool}' failed: {}", e.message))
        }
    }

    fn details(&self) -> Option<serde_json::Value> {
        match self {
            Self::LimitExceeded { parameter, .. } => {
                Some(serde_json::json!({ "parameter": parameter }))
            }
            Self::DeadlineExceeded { limit_secs } => {
                Some(serde_json::json!({ "limit_secs": limit_secs }))
            }
            _ => None,
        }
    }
}

impl IntoResponse for ToolCallError {
    fn into_response(self) -> Response {
        let body =
            ApiResponse::<()>::error_with_code(self.to_string(), self.code(), self.details());
        let mut response = (self.status(), Json(body)).into_response();
        if let Some(secs) = self.retry_after_secs() {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}

/// Cancels the call's token when the governing future is dropped (client
/// disconnect, aborted SSE task) or returns.
struct CancelOnDrop(CancelToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

type InFlight = Arc<Mutex<HashMap<u64, Option<Instant>>>>;

fn lock(map: &InFlight) -> MutexGuard<'_, HashMap<u64, Option<Instant>>> {
    map.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Registers a call's deadline while its work runs; removes it on drop.
struct InFlightGuard {
    map: InFlight,
    id: u64,
}

impl InFlightGuard {
    fn register(map: &InFlight, id: u64, deadline: Option<Instant>) -> Self {
        lock(map).insert(id, deadline);
        Self {
            map: Arc::clone(map),
            id,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        lock(&self.map).remove(&self.id);
    }
}

/// Admission control and off-runtime execution for tool calls.
pub struct ToolGovernor {
    cfg: ExecConfig,
    sessions: Arc<SessionManager>,
    global: Arc<Semaphore>,
    in_flight: InFlight,
    next_id: AtomicU64,
}

impl std::fmt::Debug for ToolGovernor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolGovernor")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl ToolGovernor {
    /// A governor with `cfg.concurrency` global permits over `sessions`.
    pub fn new(cfg: ExecConfig, sessions: Arc<SessionManager>) -> Self {
        let global = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
        Self {
            cfg,
            sessions,
            global,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// The resolved settings.
    pub fn config(&self) -> &ExecConfig {
        &self.cfg
    }

    /// In-flight and overdue work, capacity and process memory.
    pub fn health(&self) -> GovernorHealth {
        let now = Instant::now();
        let (in_flight, overdue) = {
            let map = lock(&self.in_flight);
            let overdue = map
                .values()
                .filter(|deadline| deadline.is_some_and(|at| now >= at + DEADLINE_GRACE))
                .count();
            (map.len(), overdue)
        };
        let capacity = self.cfg.concurrency.max(1);
        GovernorHealth {
            in_flight,
            overdue,
            capacity,
            rss_bytes: p2a_core::get_process_memory().map(|m| m.rss_bytes as u64),
            healthy: overdue < capacity,
        }
    }

    /// Run one tool call for `session` under admission control, permits, a
    /// budget and deadline, panic isolation and the dataset quotas.
    pub async fn call(
        &self,
        server: Arc<AnalyticsServer>,
        session: Arc<Session>,
        tool: &str,
        args: serde_json::Value,
        origin: CallOrigin,
    ) -> Result<ToolResult, ToolCallError> {
        let span = tracing::info_span!("tool_call", tool, ?origin, session_id = %session.id);
        async {
            let handle = tokio::runtime::Handle::current();
            let sessions = Arc::clone(&self.sessions);
            let quotas = Quotas {
                per_session: self.cfg.session_data_bytes,
                total: self.cfg.total_data_bytes,
            };
            let limit_secs = self.cfg.deadline_secs();
            let tool_name = tool.to_string();
            let s = Arc::clone(&session);
            self.run_blocking(&session, tool, move |budget| {
                let budget = budget.clone();
                handle.block_on(async move {
                    let snapshot = if quotas.any() {
                        Some(s.datasets.read().await.clone())
                    } else {
                        None
                    };
                    let result = server.call_tool_with_session(&tool_name, args, &s).await?;
                    if !result.success && budget.deadline().is_some_and(|d| Instant::now() >= d) {
                        return Err(ToolCallError::DeadlineExceeded { limit_secs });
                    }
                    if let Some(snapshot) = snapshot {
                        quotas.enforce(&s, &sessions, snapshot).await?;
                    }
                    s.touch().await;
                    Ok(result)
                })
            })
            .await
        }
        .instrument(span)
        .await
    }

    /// Run `work` on the blocking pool under the same admission, permits,
    /// budget, deadline and panic handling as a tool call. `work` receives the
    /// call's budget, which is also installed for p2a-core's checkpoints.
    pub async fn run_blocking<R, F>(
        &self,
        session: &Arc<Session>,
        label: &str,
        work: F,
    ) -> Result<R, ToolCallError>
    where
        R: Send + 'static,
        F: FnOnce(&Budget) -> Result<R, ToolCallError> + Send + 'static,
    {
        // 1. Memory admission.
        if let Some(limit) = self.cfg.rss_soft_limit_bytes
            && let Some(mem) = p2a_core::get_process_memory()
            && mem.rss_bytes as u64 > limit
        {
            tracing::warn!(
                rss = %format_bytes(mem.rss_bytes),
                "refusing a tool call: process memory is above the soft limit"
            );
            return Err(ToolCallError::MemoryPressure {
                retry_after_secs: MEMORY_PRESSURE_RETRY_SECS,
            });
        }

        // 2. Permits: the session's first, then a global one.
        let session_slots = session.exec_semaphore(self.cfg.session_concurrency.max(1));
        let session_permit =
            match tokio::time::timeout(self.cfg.queue_wait, session_slots.acquire_owned()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => return Err(ToolCallError::Internal("session is closed".into())),
                Err(_) => {
                    return Err(ToolCallError::SessionBusy {
                        retry_after_secs: SESSION_BUSY_RETRY_SECS,
                    });
                }
            };
        let global_permit = match tokio::time::timeout(
            self.cfg.queue_wait,
            Arc::clone(&self.global).acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => return Err(ToolCallError::Internal("server is closed".into())),
            Err(_) => {
                return Err(ToolCallError::ServerBusy {
                    retry_after_secs: SERVER_BUSY_RETRY_SECS,
                });
            }
        };

        // 3. Budget, cancellation and bookkeeping.
        let started = Instant::now();
        let deadline_at = self.cfg.deadline.map(|d| started + d);
        let token = CancelToken::new();
        let budget = Budget::new(deadline_at, token.clone());
        let _cancel_on_drop = CancelOnDrop(token.clone());
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let in_flight = InFlightGuard::register(&self.in_flight, id, deadline_at);
        let active = session.begin_call();
        let session_id = session.id.clone();
        let span = tracing::Span::current();

        // 4. Off the runtime. Everything held is released when the work stops.
        let mut join = tokio::task::spawn_blocking(move || {
            let _held = (session_permit, global_permit, in_flight, active);
            let _span = span.enter();
            let _scope = crate::path_jail::enter_session_scope(&session_id);
            resource::with_budget(budget.clone(), || work(&budget))
        });

        // 5. Deadline.
        let limit_secs = self.cfg.deadline_secs();
        let outcome = match deadline_at {
            Some(at) => {
                let give_up = tokio::time::Instant::from_std(at + DEADLINE_GRACE);
                tokio::select! {
                    joined = &mut join => joined,
                    () = tokio::time::sleep_until(give_up) => {
                        token.cancel();
                        tracing::warn!(
                            tool = label,
                            limit_secs,
                            "tool call passed its deadline; the work is detached and counted as overdue until it stops"
                        );
                        return Err(ToolCallError::DeadlineExceeded { limit_secs });
                    }
                }
            }
            None => join.await,
        };

        // 6. Outcome.
        match outcome {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => {
                if deadline_at.is_some_and(|at| Instant::now() >= at) {
                    Err(ToolCallError::DeadlineExceeded { limit_secs })
                } else {
                    Err(e)
                }
            }
            Err(join_error) if join_error.is_panic() => {
                tracing::error!(tool = label, "tool call panicked");
                Err(ToolCallError::Internal(format!(
                    "tool '{label}' failed unexpectedly; the session is still usable"
                )))
            }
            Err(join_error) => {
                tracing::error!(tool = label, error = %join_error, "tool call was interrupted");
                Err(ToolCallError::Internal(format!(
                    "tool '{label}' was interrupted before it finished"
                )))
            }
        }
    }
}

/// The dataset-byte quotas of one call.
#[derive(Debug, Clone, Copy)]
struct Quotas {
    per_session: Option<u64>,
    total: Option<u64>,
}

impl Quotas {
    fn any(&self) -> bool {
        self.per_session.is_some() || self.total.is_some()
    }

    /// Refuse a call that grew the session's datasets past a quota, restoring
    /// `snapshot`. A call that does not grow the data is never refused. The
    /// rollback assumes one call per session at a time.
    async fn enforce(
        &self,
        session: &Session,
        sessions: &SessionManager,
        snapshot: HashMap<String, p2a_core::data::Dataset>,
    ) -> Result<(), ToolCallError> {
        let before = datasets_bytes(&snapshot);
        let after = session.refresh_data_bytes().await;
        if after <= before {
            return Ok(());
        }
        let mut refusal = None;
        if let Some(limit) = self.per_session
            && after > limit
        {
            refusal = Some(format!(
                "This session's datasets would use {} (limit {} per session). The change was \
                 undone; use a smaller dataset or start a new session.",
                fmt_bytes(after),
                fmt_bytes(limit)
            ));
        }
        if refusal.is_none()
            && let Some(limit) = self.total
        {
            let total = sessions.total_data_bytes().await;
            if total > limit {
                refusal = Some(format!(
                    "The server's dataset memory is full ({} of {}). The change was undone; \
                     try again later or use a smaller dataset.",
                    fmt_bytes(total),
                    fmt_bytes(limit)
                ));
            }
        }
        match refusal {
            None => Ok(()),
            Some(message) => {
                *session.datasets.write().await = snapshot;
                session.refresh_data_bytes().await;
                tracing::info!(session_id = %session.id, "dataset quota exceeded; change undone");
                Err(ToolCallError::QuotaExceeded(message))
            }
        }
    }
}

fn fmt_bytes(bytes: u64) -> String {
    format_bytes(usize::try_from(bytes).unwrap_or(usize::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransportType;

    fn http_cfg(addr: &str, token: Option<&str>, allow: bool) -> ServerConfig {
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
    fn tool_call_error_table() {
        use ToolCallError::*;
        let cases = [
            (UnknownTool("x".into()), 400, "unknown_tool", None),
            (InvalidArguments("x".into()), 400, "invalid_arguments", None),
            (
                LimitExceeded {
                    parameter: "p".into(),
                    message: "m".into(),
                },
                422,
                "limit_exceeded",
                None,
            ),
            (QuotaExceeded("q".into()), 422, "quota_exceeded", None),
            (
                SessionBusy {
                    retry_after_secs: 2,
                },
                429,
                "session_busy",
                Some(2),
            ),
            (
                ServerBusy {
                    retry_after_secs: 5,
                },
                503,
                "server_busy",
                Some(5),
            ),
            (
                MemoryPressure {
                    retry_after_secs: 15,
                },
                503,
                "memory_pressure",
                Some(15),
            ),
            (
                DeadlineExceeded { limit_secs: 60 },
                504,
                "deadline_exceeded",
                None,
            ),
            (Internal("i".into()), 500, "internal", None),
        ];
        for (e, status, code, retry) in cases {
            assert_eq!(e.status().as_u16(), status, "{e:?}");
            assert_eq!(e.code(), code, "{e:?}");
            assert_eq!(e.retry_after_secs(), retry, "{e:?}");
        }
    }

    async fn body_json(r: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn into_response_sets_retry_after_and_code() {
        let r = ToolCallError::ServerBusy {
            retry_after_secs: 5,
        }
        .into_response();
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(r.headers()[header::RETRY_AFTER], "5");
        let v = body_json(r).await;
        assert_eq!(v["success"], serde_json::json!(false));
        assert_eq!(v["code"], serde_json::json!("server_busy"));
        assert!(v["error"].as_str().unwrap().contains("busy"), "{v}");

        let r = ToolCallError::LimitExceeded {
            parameter: "bootstrap".into(),
            message: "m".into(),
        }
        .into_response();
        assert!(r.headers().get(header::RETRY_AFTER).is_none());
        let v = body_json(r).await;
        assert_eq!(v["details"]["parameter"], serde_json::json!("bootstrap"));

        let v = body_json(ToolCallError::DeadlineExceeded { limit_secs: 60 }.into_response()).await;
        assert_eq!(v["details"]["limit_secs"], serde_json::json!(60));
    }

    #[test]
    fn handler_errors_map_by_code() {
        let e = ToolCallError::from_handler_error(
            "t",
            rmcp::ErrorData::invalid_params("bad column", None),
        );
        assert!(matches!(&e, ToolCallError::InvalidArguments(m) if m == "bad column"));
        let e = ToolCallError::from_handler_error("t", rmcp::ErrorData::internal_error("x", None));
        assert!(matches!(&e, ToolCallError::Internal(m) if m.contains("Tool 't' failed")));
    }

    #[test]
    fn exec_config_hosted_defaults() {
        let c = ExecConfig::from_config(&http_cfg("0.0.0.0:8080", None, true));
        assert_eq!(c.concurrency, available_cpus().max(1));
        assert_eq!(c.session_concurrency, 1);
        assert_eq!(c.queue_wait, Duration::from_secs(10));
        assert_eq!(c.deadline, Some(Duration::from_secs(60)));
        assert_eq!(c.stuck_exit, Some(Duration::from_secs(300)));
        assert_eq!(c.rss_soft_limit_bytes, Some(700 << 20));
        assert_eq!(c.session_data_bytes, Some(200 << 20));
        assert_eq!(c.total_data_bytes, Some(500 << 20));
        assert_eq!(c.alloc_ceiling_bytes, Some(256 << 20));
        assert!(c.session_dirs);
    }

    #[test]
    fn exec_config_off_defaults() {
        for c in [
            ExecConfig::from_config(&http_cfg("127.0.0.1:8080", None, false)),
            ExecConfig::from_config(&http_cfg("0.0.0.0:8080", Some("t"), false)),
            ExecConfig::default(),
        ] {
            assert_eq!(c.concurrency, available_cpus());
            assert_eq!(c.session_concurrency, available_cpus());
            assert_eq!(c.queue_wait, Duration::from_secs(300));
            assert_eq!(c.deadline, None);
            assert_eq!(c.stuck_exit, None);
            assert_eq!(c.rss_soft_limit_bytes, None);
            assert_eq!(c.session_data_bytes, None);
            assert_eq!(c.total_data_bytes, None);
            assert_eq!(c.alloc_ceiling_bytes, None);
            assert!(!c.session_dirs);
        }
    }

    #[test]
    fn exec_config_explicit_overrides() {
        let mut hosted = http_cfg("0.0.0.0:8080", None, true);
        hosted.exec.tool_timeout_secs = Some(0);
        assert_eq!(ExecConfig::from_config(&hosted).deadline, None);
        hosted.exec.tool_timeout_secs = Some(5);
        hosted.exec.memory_soft_limit_mb = Some(0);
        hosted.exec.tool_concurrency = Some(3);
        hosted.exec.session_concurrency = Some(2);
        let c = ExecConfig::from_config(&hosted);
        assert_eq!(c.deadline, Some(Duration::from_secs(5)));
        assert_eq!(c.rss_soft_limit_bytes, None);
        assert_eq!((c.concurrency, c.session_concurrency), (3, 2));

        let mut local = http_cfg("127.0.0.1:8080", None, false);
        local.http.session_dirs = SessionDirsSetting::On;
        local.exec.session_data_mb = Some(1);
        let c = ExecConfig::from_config(&local);
        assert!(c.session_dirs);
        assert_eq!(c.session_data_bytes, Some(1 << 20));
        let mut open = http_cfg("0.0.0.0:8080", None, true);
        open.http.session_dirs = SessionDirsSetting::Off;
        assert!(!ExecConfig::from_config(&open).session_dirs);
    }
}
