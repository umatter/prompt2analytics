//! Session management for multi-user HTTP transport.
//!
//! Each session maintains isolated dataset storage and state,
//! allowing multiple users to work independently.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::sync::{RwLock, Semaphore, broadcast};
use uuid::Uuid;

use p2a_core::data::Dataset;

use crate::config::SessionConfig;

/// A unique session identifier.
pub type SessionId = String;

/// Session state for a single user connection.
pub struct Session {
    /// Unique session identifier
    pub id: SessionId,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was last accessed
    pub last_accessed: RwLock<DateTime<Utc>>,
    /// Datasets loaded in this session
    pub datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    /// Global random seed for ML reproducibility
    pub global_seed: Arc<RwLock<Option<u64>>>,
    /// Optional user ID (when authentication is enabled)
    pub user_id: Option<String>,
    /// Per-session execution permits, created on first use.
    exec_slots: OnceLock<Arc<Semaphore>>,
    /// Tool calls currently running in this session.
    active_calls: AtomicUsize,
    /// Cached in-memory size of `datasets`, in bytes.
    data_bytes: AtomicU64,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("created_at", &self.created_at)
            .field("user_id", &self.user_id)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Create a new session with the given ID.
    pub fn new(id: SessionId, user_id: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id,
            created_at: now,
            last_accessed: RwLock::new(now),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
            user_id,
            exec_slots: OnceLock::new(),
            active_calls: AtomicUsize::new(0),
            data_bytes: AtomicU64::new(0),
        }
    }

    /// Update the last accessed timestamp.
    pub async fn touch(&self) {
        let mut last_accessed = self.last_accessed.write().await;
        *last_accessed = Utc::now();
    }

    /// Check if the session has expired.
    pub async fn is_expired(&self, ttl_minutes: u64) -> bool {
        let last_accessed = *self.last_accessed.read().await;
        is_past_ttl(last_accessed, Utc::now(), ttl_minutes)
    }

    /// Get session info for API responses.
    pub async fn info(&self) -> SessionInfo {
        let datasets = self.datasets.read().await;
        let last_accessed = self.last_accessed.read().await;
        SessionInfo {
            id: self.id.clone(),
            created_at: self.created_at,
            last_accessed: *last_accessed,
            dataset_count: datasets.len(),
            user_id: self.user_id.clone(),
        }
    }

    /// The session's execution semaphore. It is created with `permits`
    /// (clamped to `1..=Semaphore::MAX_PERMITS`) on the first call; later
    /// calls return the same semaphore whatever they pass.
    pub fn exec_semaphore(&self, permits: usize) -> Arc<Semaphore> {
        self.exec_slots
            .get_or_init(|| Arc::new(Semaphore::new(permits.clamp(1, Semaphore::MAX_PERMITS))))
            .clone()
    }

    /// Tool calls currently running in this session.
    pub fn active_calls(&self) -> usize {
        self.active_calls.load(Ordering::SeqCst)
    }

    /// Count a running call until the returned guard is dropped. A session
    /// with a running call is never evicted or expired by the cleanup task.
    pub fn begin_call(self: &Arc<Self>) -> ActiveCallGuard {
        self.active_calls.fetch_add(1, Ordering::SeqCst);
        ActiveCallGuard {
            session: Arc::clone(self),
        }
    }

    /// The cached in-memory size of this session's datasets, in bytes (as of
    /// the last [`refresh_data_bytes`](Self::refresh_data_bytes)).
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes.load(Ordering::SeqCst)
    }

    /// Recompute the size of this session's datasets, cache it and return it.
    pub async fn refresh_data_bytes(&self) -> u64 {
        let bytes = datasets_bytes(&*self.datasets.read().await);
        self.data_bytes.store(bytes, Ordering::SeqCst);
        bytes
    }
}

/// Keeps a session's active-call count raised while it lives.
#[derive(Debug)]
pub struct ActiveCallGuard {
    session: Arc<Session>,
}

impl Drop for ActiveCallGuard {
    fn drop(&mut self) {
        self.session.active_calls.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The in-memory size of a dataset map: the sum of each DataFrame's
/// `estimated_size()` (which, unlike `p2a_core::estimate_dataset_memory`,
/// counts string buffers).
pub fn datasets_bytes(map: &HashMap<String, Dataset>) -> u64 {
    map.values()
        .map(|d| d.df().estimated_size() as u64)
        .fold(0u64, u64::saturating_add)
}

/// `bytes` as the dataset quota refusals print them: the tool governor's
/// (`exec`) and a dataset reload's (`persistent_session`), one format for
/// both.
#[cfg(any(feature = "http", feature = "db"))]
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    p2a_core::memory::format_bytes(usize::try_from(bytes).unwrap_or(usize::MAX))
}

/// Whether a session last accessed at `last_accessed` has expired at `now`
/// under `ttl_minutes`: more whole minutes have elapsed than the TTL. The
/// comparison is signed, so a `last_accessed` in the future (clock skew, a
/// stored row) is not expired. The live table and the persisted rows use the
/// same rule.
pub(crate) fn is_past_ttl(
    last_accessed: DateTime<Utc>,
    now: DateTime<Utc>,
    ttl_minutes: u64,
) -> bool {
    let elapsed = now.signed_duration_since(last_accessed);
    elapsed.num_minutes() > i64::try_from(ttl_minutes).unwrap_or(i64::MAX)
}

/// The longest `user_id` a session accepts, in bytes.
const MAX_USER_ID_BYTES: usize = 128;

/// Check a client-supplied `user_id`: at most 128 bytes, each an ASCII
/// letter or digit or one of `.`, `_`, `@` and `-`. An empty id is `None`.
pub fn validate_user_id(raw: Option<String>) -> Result<Option<String>, String> {
    let Some(id) = raw.filter(|id| !id.is_empty()) else {
        return Ok(None);
    };
    let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'@' | b'-');
    if id.len() <= MAX_USER_ID_BYTES && id.bytes().all(allowed) {
        Ok(Some(id))
    } else {
        Err(format!(
            "user_id must be at most {MAX_USER_ID_BYTES} characters from A-Z, a-z, 0-9, '.', \
             '_', '@' and '-'"
        ))
    }
}

/// Whether `id` has the form of a session id: a UUID in the hyphenated
/// lower-case form `SessionManager::create_session` produces.
pub(crate) fn is_session_id(id: &str) -> bool {
    Uuid::parse_str(id).is_ok_and(|u| u.hyphenated().encode_lower(&mut Uuid::encode_buffer()) == id)
}

/// Session information for API responses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: SessionId,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub dataset_count: usize,
    pub user_id: Option<String>,
}

/// Why a session left the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalReason {
    /// Deleted by its client.
    Deleted,
    /// Idle longer than the TTL.
    Expired,
    /// Evicted to make room for a new session.
    Evicted,
}

/// Capacity of the removal channel. A receiver that falls further behind
/// sees `RecvError::Lagged` and is expected to resynchronise (the persistent
/// manager runs a sweep).
const REMOVAL_CHANNEL_CAPACITY: usize = 1024;

/// Remove a removed session's data directory, logging a failure.
fn remove_session_dir_logged(id: &str) {
    if let Err(e) = crate::path_jail::remove_session_dir(id) {
        tracing::warn!(session_id = %id, error = %e, "Failed to remove session directory");
    }
}

/// Manages all active sessions.
///
/// This is the only registry of live sessions and the only owner of their
/// `last_accessed` and TTL; the persistent manager holds a shared reference
/// to it and makes stored sessions live again with [`SessionManager::adopt`].
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions keyed by session ID
    sessions: Arc<RwLock<HashMap<SessionId, Arc<Session>>>>,
    /// Configuration for session management
    config: SessionConfig,
    /// Every removal (delete, expiry, cleanup, eviction) is sent here.
    removals: broadcast::Sender<SessionId>,
}

impl SessionManager {
    /// Create a new session manager with the given configuration.
    pub fn new(config: SessionConfig) -> Self {
        let (removals, _) = broadcast::channel(REMOVAL_CHANNEL_CAPACITY);
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            removals,
        }
    }

    /// Subscribe to session removals. Every session that leaves the table
    /// (deleted, expired on access or by cleanup, evicted) is sent once,
    /// after it has left. Removals before the call are not replayed.
    pub fn subscribe_removals(&self) -> broadcast::Receiver<SessionId> {
        self.removals.subscribe()
    }

    /// The idle time after which a session expires, in minutes.
    pub fn ttl_minutes(&self) -> u64 {
        self.config.ttl_minutes
    }

    /// A snapshot of every live session's id and last access.
    pub async fn live_last_accessed(&self) -> Vec<(SessionId, DateTime<Utc>)> {
        // Collect first so no last_accessed lock is awaited under the map lock.
        let sessions: Vec<Arc<Session>> = self.sessions.read().await.values().cloned().collect();
        let mut out = Vec::with_capacity(sessions.len());
        for session in sessions {
            let last = *session.last_accessed.read().await;
            out.push((session.id.clone(), last));
        }
        out
    }

    /// Create a new session and return its ID.
    ///
    /// The capacity check and the insert happen under one write lock, so
    /// concurrent creations cannot overshoot `max_sessions`. When the table is
    /// full and `evict_idle_minutes` is set, the least recently used session
    /// that has no running call and has been idle at least that long is
    /// evicted to make room; otherwise this returns `MaxSessionsReached`.
    pub async fn create_session(&self, user_id: Option<String>) -> Result<SessionId, SessionError> {
        let mut sessions = self.sessions.write().await;
        let id = Uuid::new_v4().to_string();
        let session = Arc::new(Session::new(id.clone(), user_id));
        self.admit(&mut sessions, session).await?;

        tracing::info!(session_id = %id, "Created new session");
        Ok(id)
    }

    /// Make a stored session live again under its own id (a persisted session
    /// after a restart). `id` must be a UUID in the hyphenated lower-case form
    /// `create_session` produces; anything else is `NotFound`. A session that
    /// is already live is returned as it is. Otherwise a fresh session
    /// (`last_accessed` = now, no datasets) goes through the same capacity and
    /// eviction rules as `create_session`.
    pub async fn adopt(
        &self,
        id: &str,
        user_id: Option<String>,
    ) -> Result<Arc<Session>, SessionError> {
        if !is_session_id(id) {
            return Err(SessionError::NotFound);
        }
        let mut sessions = self.sessions.write().await;
        if let Some(existing) = sessions.get(id) {
            return Ok(existing.clone());
        }
        let session = Arc::new(Session::new(id.to_string(), user_id));
        self.admit(&mut sessions, session.clone()).await?;
        tracing::info!(session_id = %id, "Adopted stored session");
        Ok(session)
    }

    /// Insert `session` into the write-locked table, evicting the least
    /// recently used idle session first when the table is full and eviction
    /// is on (see [`create_session`](Self::create_session)).
    async fn admit(
        &self,
        sessions: &mut HashMap<SessionId, Arc<Session>>,
        session: Arc<Session>,
    ) -> Result<(), SessionError> {
        if sessions.len() >= self.config.max_sessions {
            // A threshold too large for a chrono::Duration means never evict.
            let Some(threshold) = self
                .config
                .evict_idle_minutes
                .and_then(|m| i64::try_from(m).ok())
                .and_then(chrono::Duration::try_minutes)
            else {
                return Err(SessionError::MaxSessionsReached);
            };
            let now = Utc::now();
            // Lock order: the sessions map, then a session's last_accessed.
            let mut victim: Option<(SessionId, DateTime<Utc>)> = None;
            for (id, session) in sessions.iter() {
                if session.active_calls() > 0 {
                    continue;
                }
                let last = *session.last_accessed.read().await;
                if now.signed_duration_since(last) < threshold {
                    continue;
                }
                if victim.as_ref().is_none_or(|(_, oldest)| last < *oldest) {
                    victim = Some((id.clone(), last));
                }
            }
            let Some((victim_id, _)) = victim else {
                return Err(SessionError::MaxSessionsReached);
            };
            sessions.remove(&victim_id);
            tracing::info!(session_id = %victim_id, "Evicted least recently used idle session");
            self.session_removed(&victim_id, RemovalReason::Evicted);
        }
        sessions.insert(session.id.clone(), session);
        Ok(())
    }

    /// Get a session by ID, updating its last accessed time.
    pub async fn get_session(&self, id: &str) -> Result<Arc<Session>, SessionError> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(id).cloned().ok_or(SessionError::NotFound)?;
        drop(sessions);

        // Check if expired (a session with a running call is not)
        if session.active_calls() == 0 && session.is_expired(self.config.ttl_minutes).await {
            // Remove expired session
            let removed = self.sessions.write().await.remove(id);
            if removed.is_some() {
                tracing::info!(session_id = %id, "Removed expired session");
                self.session_removed(id, RemovalReason::Expired);
            }
            return Err(SessionError::Expired);
        }

        // Update last accessed
        session.touch().await;
        Ok(session)
    }

    /// Delete a session by ID.
    pub async fn delete_session(&self, id: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id).ok_or(SessionError::NotFound)?;
        drop(sessions);
        tracing::info!(session_id = %id, "Deleted session");
        self.session_removed(id, RemovalReason::Deleted);
        Ok(())
    }

    /// List all active sessions.
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        // Collect first so no dataset lock is awaited under the map lock.
        let sessions: Vec<Arc<Session>> = self.sessions.read().await.values().cloned().collect();
        let mut infos = Vec::with_capacity(sessions.len());
        for session in sessions {
            infos.push(session.info().await);
        }
        infos
    }

    /// Clean up expired sessions (call periodically). Sessions with a running
    /// call are skipped.
    pub async fn cleanup_expired(&self) -> usize {
        let sessions = self.sessions.read().await;
        let mut expired_ids = Vec::new();

        for (id, session) in sessions.iter() {
            if session.active_calls() == 0 && session.is_expired(self.config.ttl_minutes).await {
                expired_ids.push(id.clone());
            }
        }
        drop(sessions);

        let mut count = 0;
        if !expired_ids.is_empty() {
            let mut sessions = self.sessions.write().await;
            for id in &expired_ids {
                if sessions.remove(id).is_some() {
                    count += 1;
                    self.session_removed(id, RemovalReason::Expired);
                }
            }
            tracing::info!(count = count, "Cleaned up expired sessions");
        }

        count
    }

    /// The single hook every removal path (delete, expiry, cleanup, eviction)
    /// calls once the session has left the table. It tells the removal
    /// subscribers (the persistent manager deletes the session's rows) and
    /// removes the session's data directory: on tokio's blocking pool inside
    /// a runtime, inline outside one. Every async method of the manager only
    /// awaits tokio's runtime-agnostic locks, so any executor can poll it;
    /// `start_cleanup_task`, a plain function that spawns a tokio task, is the
    /// only method that needs a runtime.
    fn session_removed(&self, id: &str, reason: RemovalReason) {
        tracing::debug!(session_id = %id, ?reason, "Session removed");
        // An error only means that nobody is subscribed.
        let _ = self.removals.send(id.to_string());
        let id = id.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(move || remove_session_dir_logged(&id));
            }
            Err(_) => remove_session_dir_logged(&id),
        }
    }

    /// Start background cleanup task.
    pub fn start_cleanup_task(self: Arc<Self>, interval_minutes: u64) {
        let manager = self.clone();
        tokio::spawn(async move {
            let interval = tokio::time::Duration::from_secs(interval_minutes * 60);
            loop {
                tokio::time::sleep(interval).await;
                manager.cleanup_expired().await;
            }
        });
    }

    /// Get session count.
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// The cached dataset bytes of every session, summed. This takes only the
    /// map's read lock and never a session's dataset lock, so a quota check
    /// does not wait behind another session's long write.
    pub async fn total_data_bytes(&self) -> u64 {
        self.sessions
            .read()
            .await
            .values()
            .map(|s| s.data_bytes())
            .fold(0u64, u64::saturating_add)
    }
}

/// Session management errors.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Session not found")]
    NotFound,
    #[error("Session expired")]
    Expired,
    #[error("Maximum number of sessions reached")]
    MaxSessionsReached,
}

/// A wrapper that provides dataset storage for a session.
/// This is used to make tool handlers work with both stdio (single global store)
/// and HTTP (per-session store) transports.
#[derive(Clone)]
pub struct DatasetStore {
    datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    global_seed: Arc<RwLock<Option<u64>>>,
}

impl DatasetStore {
    /// Create a new dataset store (for stdio transport - single global store).
    pub fn new() -> Self {
        Self {
            datasets: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
        }
    }

    /// Create from a session (for HTTP transport).
    pub fn from_session(session: &Session) -> Self {
        Self {
            datasets: session.datasets.clone(),
            global_seed: session.global_seed.clone(),
        }
    }

    /// Get the datasets map.
    pub fn datasets(&self) -> &Arc<RwLock<HashMap<String, Dataset>>> {
        &self.datasets
    }

    /// Get the global seed.
    pub fn global_seed(&self) -> &Arc<RwLock<Option<u64>>> {
        &self.global_seed
    }
}

impl Default for DatasetStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple in-memory session store for testing and development.
/// In production, you might want to use Redis or another external store.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: RwLock<HashMap<SessionId, Arc<Session>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, session: Arc<Session>) {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session);
    }

    pub async fn get(&self, id: &str) -> Option<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }

    pub async fn remove(&self, id: &str) -> Option<Arc<Session>> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn cfg(max: usize, evict: Option<u64>) -> SessionConfig {
        SessionConfig {
            ttl_minutes: 60,
            max_sessions: max,
            evict_idle_minutes: evict,
        }
    }

    async fn backdate(m: &SessionManager, id: &str, minutes: i64) {
        let s = m.get_session(id).await.unwrap();
        *s.last_accessed.write().await = Utc::now() - chrono::Duration::minutes(minutes);
    }

    async fn futures_count(handles: Vec<tokio::task::JoinHandle<bool>>) -> usize {
        let mut ok = 0;
        for h in handles {
            if h.await.unwrap() {
                ok += 1;
            }
        }
        ok
    }

    #[tokio::test]
    async fn full_manager_evicts_least_recently_used_idle_session() {
        let m = SessionManager::new(cfg(2, Some(10)));
        let a = m.create_session(None).await.unwrap();
        let b = m.create_session(None).await.unwrap();
        backdate(&m, &a, 11).await;
        backdate(&m, &b, 12).await;
        let c = m.create_session(None).await.unwrap();
        assert!(matches!(
            m.get_session(&b).await,
            Err(SessionError::NotFound)
        ));
        assert!(m.get_session(&a).await.is_ok() && m.get_session(&c).await.is_ok());
        assert_eq!(m.session_count().await, 2);
    }

    #[tokio::test]
    async fn full_manager_refuses_when_nobody_is_idle_long_enough() {
        let m = SessionManager::new(cfg(2, Some(10)));
        let a = m.create_session(None).await.unwrap();
        m.create_session(None).await.unwrap();
        backdate(&m, &a, 9).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn eviction_off_keeps_refusing() {
        let m = SessionManager::new(cfg(1, None));
        let a = m.create_session(None).await.unwrap();
        backdate(&m, &a, 59).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn huge_eviction_threshold_means_never_evict() {
        // A threshold no chrono::Duration can hold must not panic in
        // create_session; it means "never evict".
        for minutes in [u64::MAX, i64::MAX as u64, (i64::MAX / 60) as u64 + 1] {
            let m = SessionManager::new(cfg(1, Some(minutes)));
            let a = m.create_session(None).await.unwrap();
            backdate(&m, &a, 59).await;
            assert!(
                matches!(
                    m.create_session(None).await,
                    Err(SessionError::MaxSessionsReached)
                ),
                "{minutes}"
            );
        }
    }

    #[test]
    fn exec_semaphore_clamps_its_permits() {
        let s = Session::new("s".into(), None);
        assert_eq!(
            s.exec_semaphore(usize::MAX).available_permits(),
            Semaphore::MAX_PERMITS
        );
        let s = Session::new("s".into(), None);
        assert_eq!(s.exec_semaphore(0).available_permits(), 1);
    }

    #[tokio::test]
    async fn busy_session_is_never_evicted() {
        let m = SessionManager::new(cfg(1, Some(10)));
        let a = m.create_session(None).await.unwrap();
        let s = m.get_session(&a).await.unwrap();
        let _g = s.begin_call();
        backdate(&m, &a, 30).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn busy_session_survives_expiry_cleanup() {
        let m = SessionManager::new(cfg(10, None));
        let a = m.create_session(None).await.unwrap();
        let s = m.get_session(&a).await.unwrap();
        let g = s.begin_call();
        backdate(&m, &a, 61).await;
        assert_eq!(m.cleanup_expired().await, 0);
        drop(g);
        assert_eq!(s.active_calls(), 0);
        assert_eq!(m.cleanup_expired().await, 1);
        assert!(matches!(
            m.get_session(&a).await,
            Err(SessionError::NotFound)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn capacity_holds_under_concurrent_creation() {
        // invariant; the old read-then-write check could overshoot
        let m = Arc::new(SessionManager::new(cfg(5, None)));
        let hs: Vec<_> = (0..40)
            .map(|_| {
                let m = m.clone();
                tokio::spawn(async move { m.create_session(None).await.is_ok() })
            })
            .collect();
        let ok = futures_count(hs).await;
        assert_eq!(ok, 5);
        assert_eq!(m.session_count().await, 5);
    }

    #[tokio::test]
    async fn adopt_respects_capacity_and_ids() {
        let m = SessionManager::new(cfg(1, None));
        let a = m.create_session(None).await.unwrap();
        // A fixed id with letters a-f, so its uppercase form below differs
        // from it; a random UUID can be all digits, which made that variant
        // adopt the live session.
        let id2 = "6f1c0b9e-3a4d-4c2b-9e7f-0a1b2c3d4e5f".to_string();
        assert!(matches!(
            m.adopt(&id2, None).await,
            Err(SessionError::MaxSessionsReached)
        ));
        m.delete_session(&a).await.unwrap();
        let adopted = m.adopt(&id2, Some("u1".into())).await.unwrap();
        assert_eq!(adopted.user_id.as_deref(), Some("u1"));
        assert!(Arc::ptr_eq(&adopted, &m.get_session(&id2).await.unwrap()));
        // Adopting a live id returns the live session, not a second one.
        assert!(Arc::ptr_eq(&adopted, &m.adopt(&id2, None).await.unwrap()));
        assert_eq!(m.session_count().await, 1);
        for bad in [
            "not-a-uuid".to_string(),
            id2.to_uppercase(),
            id2.replace('-', ""),
            format!("{{{id2}}}"),
        ] {
            assert!(
                matches!(m.adopt(&bad, None).await, Err(SessionError::NotFound)),
                "{bad}"
            );
        }
    }

    #[tokio::test]
    async fn removals_are_broadcast() {
        use tokio::sync::broadcast::error::TryRecvError;
        let m = SessionManager::new(cfg(1, Some(10)));
        let mut rx = m.subscribe_removals();

        let a = m.create_session(None).await.unwrap();
        m.delete_session(&a).await.unwrap();
        assert_eq!(rx.try_recv().unwrap(), a);

        let b = m.create_session(None).await.unwrap();
        backdate(&m, &b, 65).await;
        assert!(matches!(
            m.get_session(&b).await,
            Err(SessionError::Expired)
        ));
        assert_eq!(rx.try_recv().unwrap(), b);

        let c = m.create_session(None).await.unwrap();
        backdate(&m, &c, 65).await;
        assert_eq!(m.cleanup_expired().await, 1);
        assert_eq!(rx.try_recv().unwrap(), c);

        let d = m.create_session(None).await.unwrap();
        backdate(&m, &d, 11).await;
        m.create_session(None).await.unwrap(); // evicts d
        assert_eq!(rx.try_recv().unwrap(), d);

        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn validate_user_id_rules() {
        assert_eq!(validate_user_id(None), Ok(None));
        assert_eq!(validate_user_id(Some(String::new())), Ok(None));
        let max = "a".repeat(128);
        assert_eq!(validate_user_id(Some(max.clone())), Ok(Some(max)));
        assert!(validate_user_id(Some("a".repeat(129))).is_err());
        for ok in ["a.b", "a_b", "a@b", "a-b", "Z9", "alice.o-k_1@example.org"] {
            assert_eq!(
                validate_user_id(Some(ok.into())),
                Ok(Some(ok.into())),
                "{ok}"
            );
        }
        for bad in ["has space", "semi;colon", "ü", "a/b", "a\nb", "a+b", "<x>"] {
            let err = validate_user_id(Some(bad.into())).unwrap_err();
            assert!(err.contains("128"), "{err}");
        }
    }

    #[test]
    fn ttl_comparison_is_signed() {
        let now = Utc::now();
        // A last access in the future (clock skew, a stored row) is not expired.
        assert!(!is_past_ttl(now + chrono::Duration::minutes(90), now, 60));
        assert!(!is_past_ttl(now - chrono::Duration::minutes(60), now, 60));
        assert!(is_past_ttl(now - chrono::Duration::minutes(61), now, 60));
        assert!(!is_past_ttl(
            now - chrono::Duration::days(365),
            now,
            u64::MAX
        ));
    }

    #[tokio::test]
    async fn live_last_accessed_snapshots_every_session() {
        let m = SessionManager::new(cfg(10, None));
        assert_eq!(m.ttl_minutes(), 60);
        let a = m.create_session(None).await.unwrap();
        let b = m.create_session(None).await.unwrap();
        backdate(&m, &a, 7).await;
        let mut snap = m.live_last_accessed().await;
        snap.sort_by_key(|(_, t)| *t);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].0, a);
        assert_eq!(snap[1].0, b);
        assert!(Utc::now().signed_duration_since(snap[0].1).num_minutes() >= 7);
    }

    #[tokio::test]
    async fn total_data_bytes_sums_cached_session_bytes() {
        let m = SessionManager::new(cfg(10, None));
        let (a, b) = (
            m.create_session(None).await.unwrap(),
            m.create_session(None).await.unwrap(),
        );
        let ds = |n: usize| {
            Dataset::new(df! {"x" => (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>()}.unwrap())
        };
        let (sa, sb) = (
            m.get_session(&a).await.unwrap(),
            m.get_session(&b).await.unwrap(),
        );
        sa.datasets.write().await.insert("d".into(), ds(1000));
        sb.datasets.write().await.insert("d".into(), ds(3000));
        let (ba, bb) = (sa.refresh_data_bytes().await, sb.refresh_data_bytes().await);
        assert!(ba >= 8000 && bb >= 24000);
        assert_eq!(sa.data_bytes(), ba);
        assert_eq!(m.total_data_bytes().await, ba + bb);
    }

    /// Drive `f` on this thread without a Tokio runtime. Uncontended tokio
    /// locks are ready on their first poll, so one poll with a no-op waker
    /// is enough; a future that is not ready then would never be woken, so
    /// the helper panics instead of spinning.
    fn block_on_without_runtime<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!(
                "future was not ready on its first poll; it needs a runtime or hit a contended lock"
            ),
        }
    }

    #[test]
    fn removal_paths_work_outside_a_tokio_runtime() {
        assert!(tokio::runtime::Handle::try_current().is_err());
        let m = SessionManager::new(cfg(1, Some(10)));
        let mut removals = m.subscribe_removals();
        // Deletion.
        let a = block_on_without_runtime(m.create_session(None)).unwrap();
        block_on_without_runtime(m.delete_session(&a)).unwrap();
        assert_eq!(removals.try_recv().unwrap(), a);
        // Eviction when the table is full.
        let b = block_on_without_runtime(m.create_session(None)).unwrap();
        block_on_without_runtime(backdate(&m, &b, 11));
        let c = block_on_without_runtime(m.create_session(None)).unwrap();
        assert_eq!(removals.try_recv().unwrap(), b);
        assert!(block_on_without_runtime(m.get_session(&c)).is_ok());
    }
}
