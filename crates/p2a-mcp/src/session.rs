//! Session management for multi-user HTTP transport.
//!
//! Each session maintains isolated dataset storage and state,
//! allowing multiple users to work independently.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
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
        }
    }

    /// Update the last accessed timestamp.
    pub async fn touch(&self) {
        let mut last_accessed = self.last_accessed.write().await;
        *last_accessed = Utc::now();
    }

    /// Check if the session has expired.
    pub async fn is_expired(&self, ttl_minutes: u64) -> bool {
        let last_accessed = self.last_accessed.read().await;
        let elapsed = Utc::now().signed_duration_since(*last_accessed);
        elapsed.num_minutes() as u64 > ttl_minutes
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

/// Manages all active sessions.
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions keyed by session ID
    sessions: Arc<RwLock<HashMap<SessionId, Arc<Session>>>>,
    /// Configuration for session management
    config: SessionConfig,
}

impl SessionManager {
    /// Create a new session manager with the given configuration.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create a new session and return its ID.
    pub async fn create_session(&self, user_id: Option<String>) -> Result<SessionId, SessionError> {
        let sessions = self.sessions.read().await;
        if sessions.len() >= self.config.max_sessions {
            return Err(SessionError::MaxSessionsReached);
        }
        drop(sessions);

        let id = Uuid::new_v4().to_string();
        let session = Arc::new(Session::new(id.clone(), user_id));

        let mut sessions = self.sessions.write().await;
        sessions.insert(id.clone(), session);

        tracing::info!(session_id = %id, "Created new session");
        Ok(id)
    }

    /// Get a session by ID, updating its last accessed time.
    pub async fn get_session(&self, id: &str) -> Result<Arc<Session>, SessionError> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(id).cloned().ok_or(SessionError::NotFound)?;
        drop(sessions);

        // Check if expired
        if session.is_expired(self.config.ttl_minutes).await {
            // Remove expired session
            self.delete_session(id).await?;
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
        tracing::info!(session_id = %id, "Deleted session");
        Ok(())
    }

    /// List all active sessions.
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        let mut infos = Vec::with_capacity(sessions.len());
        for session in sessions.values() {
            infos.push(session.info().await);
        }
        infos
    }

    /// Clean up expired sessions (call periodically).
    pub async fn cleanup_expired(&self) -> usize {
        let sessions = self.sessions.read().await;
        let mut expired_ids = Vec::new();

        for (id, session) in sessions.iter() {
            if session.is_expired(self.config.ttl_minutes).await {
                expired_ids.push(id.clone());
            }
        }
        drop(sessions);

        let count = expired_ids.len();
        if count > 0 {
            let mut sessions = self.sessions.write().await;
            for id in &expired_ids {
                sessions.remove(id);
            }
            tracing::info!(count = count, "Cleaned up expired sessions");
        }

        count
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
