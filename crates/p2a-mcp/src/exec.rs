//! Execution governor for HTTP-side tool calls (design section 2).
//!
//! Every HTTP-side tool execution (`POST /api/tools/{name}`, both LLM tool
//! executors and websocket `Tool` messages) goes through
//! [`ToolGovernor::call`], which applies, in order:
//!
//! 1. memory admission (refuse while the process RSS is above a soft limit);
//! 2. a per-session and a global permit, within one bounded queue wait;
//! 3. execution on tokio's blocking pool inside the session's directory scope
//!    (`path_jail::enter_session_scope`; with session directories on, file
//!    tools are refused outside one), so the async runtime keeps serving
//!    `/health`, SSE keep-alives and other visitors;
//! 4. a per-call [`Budget`] installed for p2a-core's cooperative
//!    [`resource::checkpoint`]s, cancelled at the deadline and when the
//!    future driving the call is dropped (for `POST /api/tools/{name}`, when
//!    the client disconnects and the handler future is dropped);
//! 5. a wall-clock deadline (504) and panic isolation (500, session intact);
//! 6. per-session and total dataset-byte quotas that undo the call that
//!    crossed them;
//! 7. one audit entry per call (when auditing is configured), with the call's
//!    origin and outcome, written where the work ends; refusals before the
//!    work starts (memory, session or server busy) are recorded too.
//!
//! The request limits in force when the call is made travel with it onto the
//! blocking thread, where `AnalyticsServer::call_tool_with_session` checks
//! them before dispatch.
//!
//! The streaming chat (`POST /api/llm/chat/stream`) runs its tool loop in a
//! detached task that is not aborted when the SSE client disconnects, so the
//! tool calls of an abandoned stream keep running until they finish or reach
//! their deadline.
//!
//! The permits, the in-flight registry entry and the session's active-call
//! count are moved into the blocking closure, so they are released when the
//! work actually stops, not when the HTTP response is sent. Work that ignores
//! its budget past the deadline is detached, not stopped: its result is
//! discarded, but it keeps running (and may still change the session) and
//! stays counted as overdue until it ends. When every permit is held by
//! overdue work, [`ToolGovernor::health`] reports the server unhealthy.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
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

use crate::audit::{AuditEntry, AuditLogger, ResultSummary, sanitize_arguments};
use crate::config::{
    AuditConfig, DeploymentMode, ResourceProfile, ServerConfig, SessionDirsSetting, available_cpus,
};
use crate::server::AnalyticsServer;
use crate::session::{Session, SessionManager, datasets_bytes};
use crate::tools::limits::{self, LimitViolation};
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
    /// Leave the process memory and the in-flight count out of `/health`
    /// (open public mode, where `/health` is unauthenticated).
    pub redact_health: bool,
    /// Where the governor writes one audit entry per call (`None`: no
    /// audit). The governor shares the process's writer for that file.
    pub audit: Option<AuditConfig>,
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
            redact_health: false,
            audit: None,
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
    ///
    /// `cfg` is read as an HTTP configuration whatever its `transport` says
    /// (`ServerConfig::as_http`), like the router built from it, so the
    /// deployment mode and resource profile here are the ones the router
    /// serves.
    ///
    /// When a dataset quota applies, the session concurrency defaults to 1
    /// in every profile: a call refused by a quota restores the session's
    /// datasets as they were when it started, which would undo what another
    /// concurrent call in the same session did. An explicit
    /// `--session-concurrency` is kept.
    pub fn from_config(cfg: &ServerConfig) -> Self {
        let cfg = &cfg.as_http();
        let hosted = cfg.resource_profile() == ResourceProfile::Hosted;
        let cpus = available_cpus();
        let exec = &cfg.exec;
        let pick = |hosted_value: u64| hosted.then_some(hosted_value);
        let session_data_bytes =
            zero_is_none(exec.session_data_mb, pick(200)).map(|mb| mb.saturating_mul(MIB));
        let total_data_bytes =
            zero_is_none(exec.total_data_mb, pick(500)).map(|mb| mb.saturating_mul(MIB));
        let quotas = session_data_bytes.is_some() || total_data_bytes.is_some();
        Self {
            concurrency: exec
                .tool_concurrency
                .filter(|&n| n > 0)
                .unwrap_or(cpus.max(1)),
            session_concurrency: exec
                .session_concurrency
                .filter(|&n| n > 0)
                .unwrap_or(if hosted || quotas { 1 } else { cpus.max(1) }),
            queue_wait: Duration::from_secs(exec.tool_queue_wait_secs.unwrap_or(if hosted {
                10
            } else {
                300
            })),
            deadline: zero_is_none(exec.tool_timeout_secs, pick(60)).map(Duration::from_secs),
            stuck_exit: zero_is_none(exec.stuck_exit_secs, pick(300)).map(Duration::from_secs),
            rss_soft_limit_bytes: zero_is_none(exec.memory_soft_limit_mb, pick(700))
                .map(|mb| mb.saturating_mul(MIB)),
            session_data_bytes,
            total_data_bytes,
            alloc_ceiling_bytes: cfg.alloc_ceiling_bytes(),
            session_dirs: match cfg.http.session_dirs {
                SessionDirsSetting::Auto => cfg.deployment_mode() == DeploymentMode::OpenPublic,
                SessionDirsSetting::On => true,
                SessionDirsSetting::Off => false,
            },
            redact_health: cfg.deployment_mode() == DeploymentMode::OpenPublic,
            audit: cfg.audit.enabled.then(|| cfg.audit.clone()),
        }
    }

    fn deadline_secs(&self) -> u64 {
        self.deadline.map(ceil_secs).unwrap_or(0)
    }
}

fn ceil_secs(d: Duration) -> u64 {
    d.as_secs().saturating_add(u64::from(d.subsec_nanos() > 0))
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

impl CallOrigin {
    /// The origin as the audit log records it: `http`, `llm_tool` or
    /// `websocket`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::LlmTool => "llm_tool",
            Self::WebSocket => "websocket",
        }
    }
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
    /// A request limit refused a parameter, a dataset's size or the size of
    /// the response.
    #[error("{message}")]
    LimitExceeded {
        /// The parameter, dataset field or budget key that was exceeded.
        parameter: String,
        /// What the request asked for (or the response size).
        requested: String,
        /// The limit that applies.
        limit: String,
        /// The refusal shown to the caller.
        message: String,
    },
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
    /// The call ran past the server's time limit and its result was
    /// discarded. Work that does not check its budget is detached, not
    /// stopped, and may still finish (and change the session) later.
    #[error(
        "The tool call exceeded this server's time limit of {limit_secs} s; its result was \
         discarded, and work already under way may still finish in the background. Use a \
         smaller dataset or fewer iterations or replications."
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

    /// Map a handler's `ErrorData`. The handlers report a caller's mistake
    /// (a missing column, a bad parameter combination) as either
    /// `INVALID_PARAMS` or `INVALID_REQUEST`; both stay the caller's problem
    /// (400 `invalid_arguments`). Any other code is a server-side failure
    /// (500 `internal`).
    pub fn from_handler_error(tool: &str, e: rmcp::ErrorData) -> Self {
        if e.code == ErrorCode::INVALID_PARAMS || e.code == ErrorCode::INVALID_REQUEST {
            Self::InvalidArguments(e.message.to_string())
        } else {
            Self::Internal(format!("Tool '{tool}' failed: {}", e.message))
        }
    }

    /// The error a call reports when its work returned `self` at or after
    /// its deadline. A server-side failure there is most likely the work
    /// giving up on its budget, so it reads as the deadline; a specific
    /// refusal (quota, arguments, unknown tool, limit) is kept.
    fn after_deadline(self, limit_secs: u64) -> Self {
        match self {
            Self::Internal(_) => Self::DeadlineExceeded { limit_secs },
            other => other,
        }
    }

    fn details(&self) -> Option<serde_json::Value> {
        match self {
            Self::LimitExceeded {
                parameter,
                requested,
                limit,
                ..
            } => Some(serde_json::json!({
                "parameter": parameter,
                "requested": requested,
                "limit": limit,
            })),
            Self::DeadlineExceeded { limit_secs } => {
                Some(serde_json::json!({ "limit_secs": limit_secs }))
            }
            _ => None,
        }
    }
}

impl From<LimitViolation> for ToolCallError {
    /// A request-limit refusal (422 `limit_exceeded`). The violation's
    /// `override_key` is for the server log only and is dropped here.
    fn from(v: LimitViolation) -> Self {
        Self::LimitExceeded {
            parameter: v.parameter,
            requested: v.requested,
            limit: v.limit,
            message: v.message,
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

/// Cancels the call's token when the governing future returns or is
/// dropped. A dropped HTTP handler future (client disconnect) drops it; the
/// detached streaming-chat task is not aborted on an SSE disconnect, so its
/// calls are not cancelled that way.
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
    /// Global permits actually created (`cfg.concurrency` clamped to
    /// `1..=Semaphore::MAX_PERMITS`).
    capacity: usize,
    in_flight: InFlight,
    next_id: AtomicU64,
    /// One entry per call (disabled unless `cfg.audit` is set).
    audit: Arc<AuditLogger>,
}

/// Wait for a permit of `slots` until `give_up` (`None` = no limit).
/// `None` means the wait timed out.
async fn acquire_until(
    slots: Arc<Semaphore>,
    give_up: Option<tokio::time::Instant>,
) -> Option<Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError>> {
    match give_up {
        Some(at) => tokio::time::timeout_at(at, slots.acquire_owned())
            .await
            .ok(),
        None => Some(slots.acquire_owned().await),
    }
}

/// Take the session's permit, then a global one, both within the one
/// `queue_wait` (so a call never waits longer than that in total).
async fn acquire_permits(
    session_slots: Arc<Semaphore>,
    global: Arc<Semaphore>,
    queue_wait: Duration,
) -> Result<
    (
        tokio::sync::OwnedSemaphorePermit,
        tokio::sync::OwnedSemaphorePermit,
    ),
    ToolCallError,
> {
    let give_up = tokio::time::Instant::now().checked_add(queue_wait);
    let session_permit = match acquire_until(session_slots, give_up).await {
        Some(Ok(permit)) => permit,
        Some(Err(_)) => return Err(ToolCallError::Internal("session is closed".into())),
        None => {
            return Err(ToolCallError::SessionBusy {
                retry_after_secs: SESSION_BUSY_RETRY_SECS,
            });
        }
    };
    let global_permit = match acquire_until(global, give_up).await {
        Some(Ok(permit)) => permit,
        Some(Err(_)) => return Err(ToolCallError::Internal("server is closed".into())),
        None => {
            return Err(ToolCallError::ServerBusy {
                retry_after_secs: SERVER_BUSY_RETRY_SECS,
            });
        }
    };
    Ok((session_permit, global_permit))
}

impl std::fmt::Debug for ToolGovernor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolGovernor")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl ToolGovernor {
    /// A governor with `cfg.concurrency` global permits over `sessions`
    /// (at least 1, at most `Semaphore::MAX_PERMITS`).
    pub fn new(cfg: ExecConfig, sessions: Arc<SessionManager>) -> Self {
        let capacity = cfg.concurrency.clamp(1, Semaphore::MAX_PERMITS);
        let audit = Arc::new(match &cfg.audit {
            Some(audit) => AuditLogger::new(audit).unwrap_or_else(|e| {
                tracing::warn!("Failed to initialize the governor's audit logger: {}", e);
                AuditLogger::disabled()
            }),
            None => AuditLogger::disabled(),
        });
        Self {
            audit,
            cfg,
            sessions,
            global: Arc::new(Semaphore::new(capacity)),
            capacity,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// The resolved settings.
    pub fn config(&self) -> &ExecConfig {
        &self.cfg
    }

    /// In-flight and overdue work, capacity and process memory. With
    /// `redact_health` (open public mode) `rss_bytes` is `None`.
    pub fn health(&self) -> GovernorHealth {
        let now = Instant::now();
        let (in_flight, overdue) = {
            let map = lock(&self.in_flight);
            let overdue = map
                .values()
                .filter(|deadline| {
                    deadline
                        .and_then(|at| at.checked_add(DEADLINE_GRACE))
                        .is_some_and(|late| now >= late)
                })
                .count();
            (map.len(), overdue)
        };
        let rss_bytes = if self.cfg.redact_health {
            None
        } else {
            p2a_core::get_process_memory().map(|m| m.rss_bytes as u64)
        };
        GovernorHealth {
            in_flight,
            overdue,
            capacity: self.capacity,
            rss_bytes,
            healthy: overdue < self.capacity,
        }
    }

    /// The `governor` object `/health` publishes for `health`. In open public
    /// mode (`redact_health`) it leaves out the in-flight count and reports no
    /// process memory, so an unauthenticated visitor cannot poll them to time
    /// saturation or memory-pressure attempts; `overdue`, `capacity` and
    /// `healthy` stay.
    pub fn health_report(&self, health: &GovernorHealth) -> serde_json::Value {
        let mut report = serde_json::to_value(health).unwrap_or(serde_json::Value::Null);
        if self.cfg.redact_health
            && let Some(fields) = report.as_object_mut()
        {
            fields.remove("in_flight");
            fields.insert("rss_bytes".into(), serde_json::Value::Null);
        }
        report
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
            // Sanitised before the arguments move into the call, and only when
            // auditing. One copy is for a refusal before the work starts.
            let audit = self
                .audit
                .is_enabled()
                .then(|| (Arc::clone(&self.audit), sanitize_arguments(&args)));
            let refusal_arguments = audit.as_ref().map(|(_, arguments)| arguments.clone());
            let outcome = self
                .run_blocking(&session, tool, move |budget| {
                    let budget = budget.clone();
                    let started = Instant::now();
                    // The entry is written where the work ends, also when the
                    // deadline answer was already sent or the work panicked.
                    let ran = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        handle.block_on(async {
                            let snapshot = if quotas.any() {
                                Some(s.datasets.read().await.clone())
                            } else {
                                None
                            };
                            let result =
                                match server.call_tool_with_session(&tool_name, args, &s).await {
                                    Ok(result) => result,
                                    Err(e) => {
                                        // A result refused for its size comes from work
                                        // that ran: undo dataset growth past a quota
                                        // before reporting the refusal.
                                        if let Some(snapshot) = snapshot {
                                            quotas.enforce(&s, &sessions, snapshot).await?;
                                        }
                                        return Err(e);
                                    }
                                };
                            if !result.success
                                && budget.deadline().is_some_and(|d| Instant::now() >= d)
                            {
                                return Err(ToolCallError::DeadlineExceeded { limit_secs });
                            }
                            if let Some(snapshot) = snapshot {
                                quotas.enforce(&s, &sessions, snapshot).await?;
                            }
                            s.touch().await;
                            Ok(result)
                        })
                    }));
                    if let Some((audit, arguments)) = audit {
                        let (outcome, summary) = match &ran {
                            Ok(Ok(result)) => (
                                result_outcome(result, &budget),
                                ResultSummary::of_tool_result(result),
                            ),
                            Ok(Err(e)) => (e.code(), ResultSummary::default()),
                            Err(_) => ("internal", ResultSummary::default()),
                        };
                        let entry = AuditEntry::new(
                            &s.id,
                            origin.as_str(),
                            &tool_name,
                            arguments,
                            outcome,
                            elapsed_ms(started),
                            summary,
                        );
                        handle.block_on(audit.log(entry));
                    }
                    match ran {
                        Ok(result) => result,
                        Err(panic) => std::panic::resume_unwind(panic),
                    }
                })
                .await;
            // Refusals before the work started never reached the closure.
            if let (Err(e), Some(arguments)) = (&outcome, refusal_arguments)
                && matches!(
                    e,
                    ToolCallError::MemoryPressure { .. }
                        | ToolCallError::SessionBusy { .. }
                        | ToolCallError::ServerBusy { .. }
                )
            {
                let entry = AuditEntry::new(
                    &session.id,
                    origin.as_str(),
                    tool,
                    arguments,
                    e.code(),
                    0,
                    ResultSummary::default(),
                );
                self.audit.log(entry).await;
            }
            outcome
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

        // 2. Permits: the session's first, then a global one, within one wait.
        let (session_permit, global_permit) = acquire_permits(
            session.exec_semaphore(self.cfg.session_concurrency),
            Arc::clone(&self.global),
            self.cfg.queue_wait,
        )
        .await?;

        // 3. Budget, cancellation and bookkeeping. A deadline too far away to
        // represent is no deadline.
        let started = Instant::now();
        let deadline_at = self.cfg.deadline.and_then(|d| started.checked_add(d));
        let token = CancelToken::new();
        let budget = Budget::new(deadline_at, token.clone());
        let _cancel_on_drop = CancelOnDrop(token.clone());
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let in_flight = InFlightGuard::register(&self.in_flight, id, deadline_at);
        let active = session.begin_call();
        let session_id = session.id.clone();
        let span = tracing::Span::current();
        // The request limits in force here (the installed ones, or a test's
        // thread-local override) travel with the call onto the blocking
        // thread, where the tool's limit checks and handlers read them.
        let limits = limits::current();

        // 4. Off the runtime. Everything held is released when the work stops.
        let mut join = tokio::task::spawn_blocking(move || {
            let _held = (session_permit, global_permit, in_flight, active);
            let _span = span.enter();
            let _scope = crate::path_jail::enter_session_scope(&session_id);
            limits::with_limits((*limits).clone(), || {
                resource::with_budget(budget.clone(), || work(&budget))
            })
        });

        // 5. Deadline.
        let limit_secs = self.cfg.deadline_secs();
        let give_up = deadline_at
            .and_then(|at| at.checked_add(DEADLINE_GRACE))
            .map(tokio::time::Instant::from_std);
        let outcome = match give_up {
            Some(give_up) => {
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
                    Err(e.after_deadline(limit_secs))
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

/// The audit outcome of a call that returned a tool result: `ok`, or for a
/// failed result `deadline_exceeded`/`cancelled` when its budget had run out
/// (the budget has no getter for its token, so a spent budget before the
/// deadline means cancellation), else `tool_error`.
fn result_outcome(result: &ToolResult, budget: &Budget) -> &'static str {
    if result.success {
        "ok"
    } else if budget.check().is_err() {
        if budget.deadline().is_some_and(|d| Instant::now() >= d) {
            "deadline_exceeded"
        } else {
            "cancelled"
        }
    } else {
        "tool_error"
    }
}

/// Milliseconds since `started`, saturating.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
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
    /// rollback assumes one call per session at a time, which is why
    /// `ExecConfig::from_config` defaults the session concurrency to 1 when a
    /// quota applies.
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

/// Exits the process when the governor stays unhealthy (every permit held by
/// overdue work) for `limit` without a break.
#[derive(Debug)]
pub struct StuckWatchdog {
    limit: Duration,
    unhealthy_since: Option<Instant>,
}

impl StuckWatchdog {
    /// A watchdog that fires after `limit` of continuous unhealth.
    pub fn new(limit: Duration) -> Self {
        Self {
            limit,
            unhealthy_since: None,
        }
    }

    /// Record one observation; true once the governor has been unhealthy
    /// for at least `limit` without a healthy observation in between.
    pub fn observe(&mut self, healthy: bool, now: Instant) -> bool {
        if healthy {
            self.unhealthy_since = None;
            return false;
        }
        let since = *self.unhealthy_since.get_or_insert(now);
        now.saturating_duration_since(since) >= self.limit
    }
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
                    requested: "2".into(),
                    limit: "1".into(),
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
            requested: "20000".into(),
            limit: "10000".into(),
            message: "m".into(),
        }
        .into_response();
        assert!(r.headers().get(header::RETRY_AFTER).is_none());
        let v = body_json(r).await;
        assert_eq!(v["details"]["parameter"], serde_json::json!("bootstrap"));

        let v = body_json(ToolCallError::DeadlineExceeded { limit_secs: 60 }.into_response()).await;
        assert_eq!(v["details"]["limit_secs"], serde_json::json!(60));
    }

    #[tokio::test]
    async fn limit_violation_maps_to_422_with_details() {
        let v = LimitViolation {
            tool: "mediation_analysis".into(),
            parameter: "bootstrap".into(),
            requested: "1000000000000".into(),
            limit: "10000".into(),
            override_key: "replications",
            message: "Parameter `bootstrap` = 1000000000000 exceeds this server's limit of 10000 \
                      (replications) for `mediation_analysis`. Use a smaller value."
                .into(),
        };
        let e = ToolCallError::from(v.clone());
        assert_eq!(e.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(e.code(), "limit_exceeded");
        assert_eq!(e.to_string(), v.message);
        let body = body_json(e.into_response()).await;
        assert_eq!(body["code"], "limit_exceeded");
        assert_eq!(body["error"], v.message.as_str());
        assert_eq!(
            body["details"],
            serde_json::json!({
                "parameter": "bootstrap",
                "requested": "1000000000000",
                "limit": "10000"
            })
        );
        // The override key is for the server log only.
        assert!(!body.to_string().contains("replications\""), "{body}");
    }

    #[test]
    fn handler_errors_map_by_code() {
        let e = ToolCallError::from_handler_error(
            "t",
            rmcp::ErrorData::invalid_params("bad column", None),
        );
        assert!(matches!(&e, ToolCallError::InvalidArguments(m) if m == "bad column"));
        // Handlers also report caller mistakes as INVALID_REQUEST.
        let e = ToolCallError::from_handler_error(
            "t",
            rmcp::ErrorData::invalid_request("filter requires a column", None),
        );
        assert!(
            matches!(&e, ToolCallError::InvalidArguments(m) if m == "filter requires a column"),
            "{e:?}"
        );
        let e = ToolCallError::from_handler_error("t", rmcp::ErrorData::internal_error("x", None));
        assert!(matches!(&e, ToolCallError::Internal(m) if m.contains("Tool 't' failed")));
    }

    #[test]
    fn deadline_only_masks_internal_errors() {
        use ToolCallError::*;
        assert!(matches!(
            Internal("stopped".into()).after_deadline(60),
            DeadlineExceeded { limit_secs: 60 }
        ));
        // A specific refusal the work returned late keeps its own code.
        for e in [
            QuotaExceeded("q".into()),
            InvalidArguments("a".into()),
            UnknownTool("t".into()),
            LimitExceeded {
                parameter: "p".into(),
                requested: "2".into(),
                limit: "1".into(),
                message: "m".into(),
            },
        ] {
            let code = e.code();
            assert_eq!(e.after_deadline(60).code(), code);
        }
    }

    #[test]
    fn deadline_text_does_not_claim_the_work_stopped() {
        let m = ToolCallError::DeadlineExceeded { limit_secs: 60 }.to_string();
        assert!(m.contains("60 s") && m.contains("discarded"), "{m}");
        assert!(!m.contains("was stopped"), "{m}");
    }

    #[tokio::test]
    async fn queue_wait_bounds_both_permits_together() {
        // The session permit frees after half the wait; the global one only
        // after the whole wait. With one shared deadline the call is refused;
        // with a fresh wait per permit it would have got the global permit.
        //
        // The margins are deliberate. Shared deadline at 4 s, global release
        // at 5 s, fresh per-permit deadline at 6 s: a runtime stalled for up
        // to a second (a loaded CI runner) cannot let the 4 s timeout and the
        // 5 s release fire in one timer tick, where tokio's Timeout would
        // poll the acquire first and get the permit.
        let wait = Duration::from_secs(4);
        let session_slots = Arc::new(Semaphore::new(1));
        let global = Arc::new(Semaphore::new(1));
        let session_held = Arc::clone(&session_slots).acquire_owned().await.unwrap();
        let global_held = Arc::clone(&global).acquire_owned().await.unwrap();
        tokio::spawn(async move {
            tokio::time::sleep(wait / 2).await;
            drop(session_held);
        });
        tokio::spawn(async move {
            tokio::time::sleep(wait + wait / 4).await;
            drop(global_held);
        });
        let r = acquire_permits(session_slots, global, wait).await;
        assert!(
            matches!(
                r,
                Err(ToolCallError::ServerBusy {
                    retry_after_secs: SERVER_BUSY_RETRY_SECS
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn health_report_redacts_in_open_public_mode() {
        let sessions = Arc::new(SessionManager::new(crate::config::SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let open = ExecConfig::from_config(&http_cfg("0.0.0.0:8080", None, true));
        assert!(open.redact_health);
        let gov = ToolGovernor::new(open, sessions.clone());
        let h = gov.health();
        assert_eq!(h.rss_bytes, None);
        let report = gov.health_report(&h);
        assert!(report.get("in_flight").is_none(), "{report}");
        assert_eq!(report["rss_bytes"], serde_json::Value::Null);
        for key in ["overdue", "capacity", "healthy"] {
            assert!(report.get(key).is_some(), "{report}");
        }

        let local = ExecConfig::from_config(&http_cfg("127.0.0.1:8080", None, false));
        assert!(!local.redact_health);
        let gov = ToolGovernor::new(local, sessions);
        let report = gov.health_report(&gov.health());
        assert!(report.get("in_flight").is_some(), "{report}");
    }

    /// A governor over a fresh session manager, and one session in it.
    async fn governor_and_session(cfg: ExecConfig) -> (ToolGovernor, Arc<Session>) {
        let sessions = Arc::new(SessionManager::new(crate::config::SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let id = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&id).await.unwrap();
        (ToolGovernor::new(cfg, sessions), session)
    }

    #[tokio::test]
    async fn huge_settings_do_not_panic() {
        // Values an embedder might pass to mean "unlimited": no deadline
        // arithmetic, semaphore size or queue deadline may panic.
        let (gov, session) = governor_and_session(ExecConfig {
            concurrency: usize::MAX,
            session_concurrency: usize::MAX,
            queue_wait: Duration::MAX,
            deadline: Some(Duration::MAX),
            ..Default::default()
        })
        .await;
        let r = gov.run_blocking(&session, "x", |_| Ok(7)).await;
        assert!(matches!(r, Ok(7)), "{r:?}");
        let h = gov.health();
        assert_eq!((h.in_flight, h.overdue), (0, 0));
        assert!(h.healthy);
    }

    #[test]
    fn watchdog_fires_only_after_continuous_unhealth() {
        let t0 = Instant::now();
        let mut w = StuckWatchdog::new(Duration::from_secs(300));
        assert!(!w.observe(true, t0));
        assert!(!w.observe(false, t0));
        assert!(!w.observe(false, t0 + Duration::from_secs(299)));
        assert!(!w.observe(true, t0 + Duration::from_secs(299))); // resets
        assert!(!w.observe(false, t0 + Duration::from_secs(300)));
        assert!(w.observe(false, t0 + Duration::from_secs(600)));
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
        assert!(c.redact_health);
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
            assert!(!c.redact_health);
        }
    }

    #[test]
    fn exec_config_reads_the_config_as_http() {
        // The default transport is stdio, which on its own reads as local
        // mode; the governor resolves the HTTP deployment mode, as the router
        // built from the same config does.
        let mut c = ServerConfig::default();
        c.http.addr = "0.0.0.0:8080".parse().unwrap();
        c.http.allow_unauthenticated = true;
        let e = ExecConfig::from_config(&c);
        assert_eq!(e.deadline, Some(Duration::from_secs(60)));
        assert_eq!(e.session_data_bytes, Some(200 << 20));
        assert!(e.redact_health);
    }

    #[test]
    fn a_dataset_quota_defaults_to_one_call_per_session() {
        let mut c = http_cfg("127.0.0.1:8080", None, false);
        c.exec.session_data_mb = Some(1);
        assert_eq!(ExecConfig::from_config(&c).session_concurrency, 1);
        let mut c = http_cfg("127.0.0.1:8080", None, false);
        c.exec.total_data_mb = Some(1);
        assert_eq!(ExecConfig::from_config(&c).session_concurrency, 1);
        // An explicit value is kept.
        c.exec.session_concurrency = Some(4);
        assert_eq!(ExecConfig::from_config(&c).session_concurrency, 4);
        // An explicit 0 is no quota, so the Off default stays.
        let mut c = http_cfg("127.0.0.1:8080", None, false);
        c.exec.session_data_mb = Some(0);
        c.exec.total_data_mb = Some(0);
        assert_eq!(
            ExecConfig::from_config(&c).session_concurrency,
            available_cpus().max(1)
        );
    }

    #[test]
    fn exec_config_carries_the_audit_config() {
        let mut c = http_cfg("127.0.0.1:8080", None, false);
        assert!(ExecConfig::from_config(&c).audit.is_none());
        assert!(ExecConfig::default().audit.is_none());
        c.audit.enabled = true;
        c.audit.path = "audit-test.log".into();
        let audit = ExecConfig::from_config(&c).audit.unwrap();
        assert!(audit.enabled);
        assert_eq!(audit.path, "audit-test.log");
    }

    #[test]
    fn call_origin_names() {
        assert_eq!(CallOrigin::Http.as_str(), "http");
        assert_eq!(CallOrigin::LlmTool.as_str(), "llm_tool");
        assert_eq!(CallOrigin::WebSocket.as_str(), "websocket");
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
