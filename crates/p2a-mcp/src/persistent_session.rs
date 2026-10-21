//! Persistent session management using SurrealDB.
//!
//! This module adds database persistence (session rows, conversations,
//! messages, tool calls, settings and dataset metadata) to the live session
//! registry. [`SessionManager`] stays the only registry of live sessions and
//! the only owner of `last_accessed` and the TTL; the persistent manager holds
//! a shared reference to it, makes a stored session live again after a
//! restart, and never deletes rows on a read path.

#[cfg(feature = "db")]
use crate::db::{
    Conversation, ConversationWithMessages, DbConnection, DbError, DbSession, Message,
    PersistenceLimits, Settings,
};
use crate::session::{
    DatasetStore, Session, SessionError, SessionId, SessionManager, is_past_ttl, is_session_id,
};
use std::sync::Arc;

/// A session manager that persists sessions to SurrealDB.
///
/// Live sessions (with their in-memory datasets) belong to the shared
/// [`SessionManager`]; this type adds the database rows for conversations,
/// settings and session metadata.
#[cfg(feature = "db")]
pub struct PersistentSessionManager {
    /// The live session registry, shared with the rest of the server
    sessions: Arc<SessionManager>,
    /// Database connection for persistence
    db: Arc<DbConnection>,
}

#[cfg(feature = "db")]
impl PersistentSessionManager {
    /// Create a persistent session manager over a RocksDB store at `db_path`
    /// (in memory when `None`), sharing the live registry `sessions` and
    /// applying `limits` to every write.
    pub async fn new(
        db_path: Option<&str>,
        sessions: Arc<SessionManager>,
        limits: PersistenceLimits,
    ) -> Result<Self, DbError> {
        let db = match db_path {
            Some(path) => DbConnection::connect_rocksdb(path).await?,
            None => DbConnection::connect_memory().await?,
        };
        Ok(Self::from_connection(db, sessions, limits))
    }

    /// Create a persistent session manager over an open connection.
    pub fn from_connection(
        db: DbConnection,
        sessions: Arc<SessionManager>,
        limits: PersistenceLimits,
    ) -> Self {
        Self {
            sessions,
            db: Arc::new(db.with_limits(limits)),
        }
    }

    /// The live session registry this manager shares.
    pub fn sessions(&self) -> &Arc<SessionManager> {
        &self.sessions
    }

    /// Create a new live session and persist its row.
    pub async fn create_session(
        &self,
        user_id: Option<String>,
    ) -> Result<SessionId, PersistentSessionError> {
        let id = self.sessions.create_session(user_id.clone()).await?;
        if let Err(e) = self.register_session(&id, user_id).await {
            let _ = self.sessions.delete_session(&id).await;
            return Err(e);
        }
        tracing::info!(session_id = %id, "Created new persistent session");
        Ok(id)
    }

    /// Store the row of a live session (created by the shared
    /// `SessionManager`) unless it already exists.
    pub async fn register_session(
        &self,
        session_id: &str,
        user_id: Option<String>,
    ) -> Result<(), PersistentSessionError> {
        if self.db.get_session(session_id).await?.is_some() {
            tracing::debug!(session_id = %session_id, "Session already registered in database");
            return Ok(());
        }
        let db_session = DbSession::new(session_id.to_string(), user_id);
        self.db.upsert_session(&db_session).await?;
        tracing::info!(session_id = %session_id, "Registered existing session in database");
        Ok(())
    }

    /// Get a live session by ID, touching it.
    ///
    /// A session that is not live but whose row was accessed within the TTL
    /// (a RocksDB store after a restart) is adopted into the live registry,
    /// subject to its capacity. A row older than the TTL gives `Expired`, no
    /// row gives `NotFound`. This never deletes rows: the removal listener and
    /// the sweep do that.
    pub async fn get_session(&self, id: &str) -> Result<Arc<Session>, PersistentSessionError> {
        match self.sessions.get_session(id).await {
            Ok(session) => return Ok(session),
            Err(SessionError::NotFound) => {}
            Err(e) => return Err(e.into()),
        }
        // Only ids of the form `create_session` produces can have rows.
        if !is_session_id(id) {
            return Err(SessionError::NotFound.into());
        }
        let Some(row) = self.db.get_session(id).await? else {
            return Err(SessionError::NotFound.into());
        };
        if is_past_ttl(
            row.last_accessed_chrono(),
            chrono::Utc::now(),
            self.sessions.ttl_minutes(),
        ) {
            return Err(SessionError::Expired.into());
        }
        Ok(self.sessions.adopt(id, row.user_id).await?)
    }

    /// Get the dataset store for a session.
    pub async fn get_dataset_store(
        &self,
        session_id: &str,
    ) -> Result<DatasetStore, PersistentSessionError> {
        let session = self.get_session(session_id).await?;
        Ok(DatasetStore::from_session(&session))
    }

    /// End a session: delete all of its rows and remove it from the live
    /// registry. `NotFound` only when it was neither live nor stored.
    ///
    /// The rows go first, so this call's delete never runs concurrently
    /// with the one the removal listener starts when the live session goes;
    /// that second delete also catches a row written in between.
    pub async fn delete_session(&self, id: &str) -> Result<(), PersistentSessionError> {
        let was_stored = self.db.get_session(id).await?.is_some();
        self.db.delete_session(id).await?;
        let was_live = match self.sessions.delete_session(id).await {
            Ok(()) => true,
            Err(SessionError::NotFound) => false,
            Err(e) => return Err(e.into()),
        };
        if !was_live && !was_stored {
            return Err(SessionError::NotFound.into());
        }
        tracing::info!(session_id = %id, "Deleted persistent session");
        Ok(())
    }

    /// List all sessions (from database).
    pub async fn list_sessions(&self) -> Result<Vec<DbSession>, PersistentSessionError> {
        Ok(self.db.list_sessions().await?)
    }

    // ==================== Conversation Operations ====================

    /// Create a new conversation for a session.
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, PersistentSessionError> {
        // The session must be live; its row must exist so the conversation
        // is never an orphan (registration can have failed at creation).
        let session = self.get_session(session_id).await?;
        self.register_session(session_id, session.user_id.clone())
            .await?;

        // Create conversation
        let conv = self.db.create_conversation(session_id, title).await?;

        // Set as active conversation
        self.db
            .set_active_conversation(session_id, Some(&conv.id_string()))
            .await?;

        tracing::info!(
            session_id = %session_id,
            conversation_id = %conv.id_string(),
            "Created new conversation"
        );
        Ok(conv)
    }

    /// The conversation `conversation_id` if the live session `session_id`
    /// owns it. The session is looked up (and touched) first; a conversation
    /// that is missing or belongs to another session is the same
    /// `NotFound("Conversation not found")`, without the id.
    pub async fn authorize_conversation(
        &self,
        session_id: &str,
        conversation_id: &str,
    ) -> Result<Conversation, PersistentSessionError> {
        self.get_session(session_id).await?;
        match self.db.get_conversation(conversation_id).await {
            Ok(conv) if conv.session_id == session_id => Ok(conv),
            Ok(_) | Err(DbError::NotFound(_)) => {
                Err(DbError::NotFound(CONVERSATION_NOT_FOUND.to_string()).into())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The message `message_id` if the live session `session_id` owns its
    /// conversation; otherwise `NotFound("Message not found")`, the same for
    /// a missing and a foreign message.
    pub async fn authorize_message(
        &self,
        session_id: &str,
        message_id: &str,
    ) -> Result<Message, PersistentSessionError> {
        self.get_session(session_id).await?;
        let not_found =
            || PersistentSessionError::from(DbError::NotFound(MESSAGE_NOT_FOUND.to_string()));
        let message = match self.db.get_message(message_id).await {
            Ok(message) => message,
            Err(DbError::NotFound(_)) => return Err(not_found()),
            Err(e) => return Err(e.into()),
        };
        match self.db.get_conversation(&message.conversation_id).await {
            Ok(conv) if conv.session_id == session_id => Ok(message),
            Ok(_) | Err(DbError::NotFound(_)) => Err(not_found()),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a conversation by ID.
    pub async fn get_conversation(&self, id: &str) -> Result<Conversation, PersistentSessionError> {
        Ok(self.db.get_conversation(id).await?)
    }

    /// Get a conversation with all its messages.
    pub async fn get_conversation_with_messages(
        &self,
        id: &str,
    ) -> Result<ConversationWithMessages, PersistentSessionError> {
        Ok(self.db.get_conversation_with_messages(id).await?)
    }

    /// List all conversations for a session.
    pub async fn list_conversations(
        &self,
        session_id: &str,
    ) -> Result<Vec<Conversation>, PersistentSessionError> {
        Ok(self.db.list_conversations(session_id).await?)
    }

    /// Update a conversation's title.
    pub async fn update_conversation_title(
        &self,
        id: &str,
        title: &str,
    ) -> Result<Conversation, PersistentSessionError> {
        Ok(self.db.update_conversation_title(id, title).await?)
    }

    /// Archive or unarchive a conversation.
    pub async fn set_conversation_archived(
        &self,
        id: &str,
        is_archived: bool,
    ) -> Result<Conversation, PersistentSessionError> {
        Ok(self.db.set_conversation_archived(id, is_archived).await?)
    }

    /// Delete a conversation and all its messages.
    pub async fn delete_conversation(&self, id: &str) -> Result<(), PersistentSessionError> {
        self.db.delete_conversation(id).await?;
        tracing::info!(conversation_id = %id, "Deleted conversation");
        Ok(())
    }

    // ==================== Message Operations ====================

    /// Add a message to a conversation.
    pub async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<Message, PersistentSessionError> {
        Ok(self.db.add_message(conversation_id, role, content).await?)
    }

    /// Get all messages for a conversation.
    pub async fn get_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<Message>, PersistentSessionError> {
        Ok(self.db.get_messages(conversation_id).await?)
    }

    /// Clear all messages in a conversation.
    pub async fn clear_messages(
        &self,
        conversation_id: &str,
    ) -> Result<u32, PersistentSessionError> {
        Ok(self.db.clear_messages(conversation_id).await?)
    }

    // ==================== Settings Operations ====================

    /// Get or create settings for a live session. No row is created for a
    /// session that is not live.
    pub async fn get_settings(&self, session_id: &str) -> Result<Settings, PersistentSessionError> {
        self.get_session(session_id).await?;
        Ok(self.db.get_or_create_settings(session_id).await?)
    }

    /// Update settings for a live session.
    pub async fn update_settings(
        &self,
        settings: &Settings,
    ) -> Result<Settings, PersistentSessionError> {
        self.get_session(&settings.session_id).await?;
        Ok(self.db.update_settings(settings).await?)
    }

    /// Patch specific settings fields of a live session, creating its
    /// settings row first when it has none.
    pub async fn patch_settings(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<i32>,
    ) -> Result<Settings, PersistentSessionError> {
        self.get_session(session_id).await?;
        self.db.get_or_create_settings(session_id).await?;
        Ok(self
            .db
            .patch_settings(session_id, provider, model, temperature, max_tokens)
            .await?)
    }

    // ==================== Database Stats ====================

    /// Get database statistics.
    pub async fn db_stats(&self) -> Result<crate::db::DbStats, PersistentSessionError> {
        Ok(self.db.stats().await?)
    }

    /// Run database health check.
    pub async fn db_health_check(&self) -> Result<bool, PersistentSessionError> {
        Ok(self.db.health_check().await?)
    }

    /// Get the database connection (for advanced operations).
    pub fn db(&self) -> &Arc<DbConnection> {
        &self.db
    }

    // ==================== Cleanup ====================

    /// Bring the store in line with the live registry:
    ///
    /// 1. write each live session's own `last_accessed` to its row (creating
    ///    a missing row);
    /// 2. delete the rows of every session that is not live and whose row was
    ///    last accessed longer ago than the TTL (the whole-minute rule the
    ///    live table uses);
    /// 3. delete orphaned rows (see `DbConnection::delete_orphans`).
    pub async fn sweep(&self) -> Result<SweepStats, PersistentSessionError> {
        let mut stats = SweepStats::default();
        for (id, at) in self.sessions.live_last_accessed().await {
            self.db.set_session_last_accessed(&id, at).await?;
            stats.refreshed += 1;
        }

        // A row is past the TTL once more whole minutes than the TTL have
        // elapsed (`is_past_ttl`), i.e. at least ttl + 1 minutes. A TTL too
        // large for a date never expires a row.
        let cutoff = i64::try_from(self.sessions.ttl_minutes())
            .ok()
            .and_then(|ttl| ttl.checked_add(1))
            .and_then(chrono::Duration::try_minutes)
            .and_then(|ttl| chrono::Utc::now().checked_sub_signed(ttl));
        if let Some(cutoff) = cutoff {
            let live: std::collections::HashSet<SessionId> = self
                .sessions
                .live_last_accessed()
                .await
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for id in self.db.session_ids_last_accessed_before(cutoff).await? {
                if live.contains(&id) {
                    continue;
                }
                self.db.delete_session(&id).await?;
                stats.expired_deleted += 1;
            }
        }

        stats.orphans_deleted = self.db.delete_orphans().await?;
        if stats.expired_deleted > 0 || stats.orphans_deleted > 0 {
            tracing::info!(
                refreshed = stats.refreshed,
                expired_deleted = stats.expired_deleted,
                orphans_deleted = stats.orphans_deleted,
                "Swept the conversation store"
            );
        }
        Ok(stats)
    }

    /// Start the two tasks that keep the store in line with the live
    /// registry. Both hold only a weak reference and stop once the manager
    /// is dropped.
    ///
    /// - A listener on [`SessionManager::subscribe_removals`] deletes the
    ///   rows of every session that leaves the registry (deleted, expired,
    ///   evicted). When it falls behind it runs a [`sweep`](Self::sweep).
    /// - A timer runs a sweep at start and then every 30 s per TTL minute,
    ///   clamped to 60 s ..= 600 s.
    pub fn start_background_tasks(self: &Arc<Self>) {
        use tokio::sync::broadcast::error::RecvError;

        let mut removals = self.sessions.subscribe_removals();
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                let event = removals.recv().await;
                let Some(pm) = weak.upgrade() else { break };
                match event {
                    Ok(id) => {
                        if let Err(e) = pm.db.delete_session(&id).await {
                            tracing::warn!(session_id = %id, error = %e, "Failed to delete the rows of a removed session");
                        }
                    }
                    Err(RecvError::Lagged(missed)) => {
                        tracing::warn!(
                            missed,
                            "Session removal listener fell behind; sweeping the store"
                        );
                        if let Err(e) = pm.sweep().await {
                            tracing::warn!(error = %e, "Conversation-store sweep failed");
                        }
                    }
                    Err(RecvError::Closed) => break,
                }
            }
        });

        let weak = Arc::downgrade(self);
        let period = sweep_period(self.sessions.ttl_minutes());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let Some(pm) = weak.upgrade() else { break };
                if let Err(e) = pm.sweep().await {
                    tracing::warn!(error = %e, "Conversation-store sweep failed");
                }
            }
        });
    }

    // ==================== Tool Call Operations ====================

    /// Get tool calls for a conversation.
    pub async fn get_tool_calls_for_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<crate::db::ToolCall>, PersistentSessionError> {
        Ok(self
            .db
            .get_tool_calls_for_conversation(conversation_id)
            .await?)
    }

    /// Get tool calls for a specific message.
    pub async fn get_tool_calls_for_message(
        &self,
        message_id: &str,
    ) -> Result<Vec<crate::db::ToolCall>, PersistentSessionError> {
        Ok(self.db.get_tool_calls_for_message(message_id).await?)
    }

    // ==================== Dataset Meta Operations ====================

    /// Get all dataset metadata for a session.
    pub async fn get_datasets_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::db::DatasetMeta>, PersistentSessionError> {
        Ok(self.db.get_datasets_for_session(session_id).await?)
    }

    /// Save dataset metadata.
    pub async fn save_dataset_meta(
        &self,
        meta: &crate::db::DatasetMeta,
    ) -> Result<crate::db::DatasetMeta, PersistentSessionError> {
        Ok(self.db.save_dataset_meta(meta).await?)
    }

    /// Reload datasets for a session from their original source paths.
    ///
    /// The datasets go into the live session (adopted if necessary). Every
    /// stored path is validated through the path jail inside the session's
    /// scope, and the files are loaded on the blocking pool without holding
    /// the session's dataset lock. Failure texts are fixed and never contain
    /// server paths; the details are logged at debug level.
    pub async fn reload_session_datasets(
        &self,
        session_id: &str,
    ) -> Result<ReloadResult, PersistentSessionError> {
        use p2a_core::DataLoader;

        let session = self.get_session(session_id).await?;
        let metas = self.db.get_datasets_for_session(session_id).await?;

        let mut result = ReloadResult::default();
        let mut to_load = Vec::new();
        for meta in metas {
            match meta.source_path {
                Some(source_path) => to_load.push((meta.name, source_path)),
                // e.g. uploaded datasets without a file path
                None => result.skipped.push(ReloadSkip {
                    name: meta.name,
                    reason: RELOAD_NO_SOURCE.to_string(),
                }),
            }
        }

        let scope_id = session.id.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            let _scope = crate::path_jail::enter_session_scope(&scope_id);
            to_load
                .into_iter()
                .map(|(name, source_path)| {
                    let outcome = match crate::path_jail::validate_data_path(&source_path) {
                        Err(e) => {
                            tracing::debug!(dataset = %name, error = %e, "Reload path rejected by the path jail");
                            Err(RELOAD_OUTSIDE_JAIL)
                        }
                        Ok(path) => DataLoader::load(&path).map_err(|e| {
                            tracing::debug!(dataset = %name, error = %e, "Failed to reload dataset");
                            RELOAD_LOAD_FAILED
                        }),
                    };
                    (name, source_path, outcome)
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| {
            tracing::warn!(session_id = %session_id, error = %e, "Dataset reload task failed");
            PersistentSessionError::Internal("dataset reload failed".to_string())
        })?;

        let mut datasets = session.datasets.write().await;
        for (name, source_path, outcome) in loaded {
            match outcome {
                Ok(dataset) => {
                    tracing::info!(dataset = %name, "Reloaded dataset from file");
                    datasets.insert(name.clone(), dataset);
                    result.succeeded.push(name);
                }
                Err(error) => result.failed.push(ReloadFailure {
                    name,
                    source_path,
                    error: error.to_string(),
                }),
            }
        }
        drop(datasets);
        session.refresh_data_bytes().await;

        tracing::info!(
            session_id = %session_id,
            succeeded = result.succeeded.len(),
            failed = result.failed.len(),
            skipped = result.skipped.len(),
            "Dataset reload completed"
        );

        Ok(result)
    }
}

/// The answer for a conversation or message the caller does not own; the
/// same whether it exists or not.
const CONVERSATION_NOT_FOUND: &str = "Conversation not found";
const MESSAGE_NOT_FOUND: &str = "Message not found";

/// Reload texts. They are fixed so that no server path reaches a caller.
const RELOAD_NO_SOURCE: &str = "no source path stored";
const RELOAD_OUTSIDE_JAIL: &str = "path is not inside this session's data directory";
const RELOAD_LOAD_FAILED: &str = "the file could not be loaded";

/// What one [`PersistentSessionManager::sweep`] did.
#[derive(Debug, Default, Clone, Copy)]
pub struct SweepStats {
    /// Live sessions whose row got their `last_accessed`.
    pub refreshed: usize,
    /// Sessions past the TTL, not live, whose rows were deleted.
    pub expired_deleted: usize,
    /// Orphaned rows deleted.
    pub orphans_deleted: usize,
}

/// How often the background sweep runs: 30 s per TTL minute, clamped to
/// 60 s ..= 600 s.
fn sweep_period(ttl_minutes: u64) -> std::time::Duration {
    std::time::Duration::from_secs(ttl_minutes.saturating_mul(30).clamp(60, 600))
}

/// Refuse a database path inside the path-jail root, where file tools can
/// write (and so corrupt the store).
///
/// A relative `db_path` is taken against the current directory. Its deepest
/// existing ancestor is canonicalised (resolving symlinks) and the rest is
/// appended with `..` and `.` resolved lexically, so a path that does not
/// exist yet is judged by where it would be created. `data_root` is
/// canonicalised when it exists. The error is an operator startup message
/// and names both paths.
pub fn check_db_path_outside_data_root(
    db_path: &std::path::Path,
    data_root: &std::path::Path,
) -> Result<(), String> {
    use std::path::{Component, PathBuf};

    let absolute = if db_path.is_absolute() {
        db_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot resolve P2A_DB_PATH against the current directory: {e}"))?
            .join(db_path)
    };
    // Split into the deepest existing ancestor and the components below it.
    let mut base = absolute.as_path();
    let mut rest = Vec::new();
    let mut resolved = loop {
        if let Ok(canonical) = base.canonicalize() {
            break canonical;
        }
        match (base.parent(), base.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(Component::Normal(name));
                base = parent;
            }
            // `..` or `.` as the last component of a missing path.
            (Some(parent), None) => {
                rest.push(base.components().next_back().unwrap_or(Component::CurDir));
                base = parent;
            }
            (None, _) => break PathBuf::from(base),
        }
    };
    for component in rest.into_iter().rev() {
        match component {
            Component::ParentDir => {
                resolved.pop();
            }
            Component::CurDir => {}
            other => resolved.push(other.as_os_str()),
        }
    }
    let root = data_root
        .canonicalize()
        .unwrap_or_else(|_| data_root.to_path_buf());
    if resolved.starts_with(&root) {
        return Err(format!(
            "P2A_DB_PATH ({}) is inside the data root ({}); file tools can write there. Put \
             the database outside P2A_DATA_ROOT, e.g. /var/lib/p2a/db",
            db_path.display(),
            root.display()
        ));
    }
    Ok(())
}

/// Result of reloading session datasets.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ReloadResult {
    /// Names of successfully reloaded datasets
    pub succeeded: Vec<String>,
    /// Datasets that failed to reload
    pub failed: Vec<ReloadFailure>,
    /// Datasets that were skipped
    pub skipped: Vec<ReloadSkip>,
}

/// A dataset that failed to reload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReloadFailure {
    /// Dataset name
    pub name: String,
    /// Original source path
    pub source_path: String,
    /// Error message
    pub error: String,
}

/// A dataset that was skipped during reload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReloadSkip {
    /// Dataset name
    pub name: String,
    /// Reason for skipping
    pub reason: String,
}

/// Errors from persistent session management.
#[derive(Debug, thiserror::Error)]
pub enum PersistentSessionError {
    #[error("Session error: {0}")]
    Session(#[from] SessionError),

    #[cfg(feature = "db")]
    #[error("Database error: {0}")]
    Database(#[from] DbError),

    /// A failure inside the server (a panicked worker); the text is generic.
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
#[cfg(feature = "db")]
mod tests {
    use super::*;
    use crate::config::SessionConfig;

    fn registry(ttl_minutes: u64) -> Arc<SessionManager> {
        Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes,
            max_sessions: 100,
            evict_idle_minutes: None,
        }))
    }

    async fn manager() -> PersistentSessionManager {
        PersistentSessionManager::new(None, registry(60), PersistenceLimits::default())
            .await
            .expect("Failed to create manager")
    }

    /// Two managers over one store with separate live registries: the second
    /// one sees the store as a restarted server would.
    async fn restarted_pair() -> (
        DbConnection,
        PersistentSessionManager,
        PersistentSessionManager,
        Arc<SessionManager>,
    ) {
        let db = DbConnection::connect_memory().await.unwrap();
        let pm1 = PersistentSessionManager::from_connection(
            db.clone(),
            registry(60),
            PersistenceLimits::default(),
        );
        let sm2 = registry(60);
        let pm2 = PersistentSessionManager::from_connection(
            db.clone(),
            sm2.clone(),
            PersistenceLimits::default(),
        );
        (db, pm1, pm2, sm2)
    }

    #[tokio::test]
    async fn adopts_db_session_within_ttl_after_restart() {
        let (_db, pm1, pm2, sm2) = restarted_pair().await;
        let sid = pm1.create_session(Some("u1".into())).await.unwrap();
        pm1.create_conversation(&sid, "kept").await.unwrap();

        let adopted = pm2.get_session(&sid).await.unwrap();
        assert!(Arc::ptr_eq(&adopted, &sm2.get_session(&sid).await.unwrap()));
        assert_eq!(adopted.user_id.as_deref(), Some("u1"));
        assert_eq!(pm2.list_conversations(&sid).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn stale_db_session_is_expired_not_deleted() {
        let (db, pm1, pm2, _sm2) = restarted_pair().await;
        let sid = pm1.create_session(None).await.unwrap();
        let conv = pm1.create_conversation(&sid, "kept").await.unwrap();
        db.set_session_last_accessed(&sid, chrono::Utc::now() - chrono::Duration::hours(2))
            .await
            .unwrap();

        assert!(matches!(
            pm2.get_session(&sid).await,
            Err(PersistentSessionError::Session(SessionError::Expired))
        ));
        assert!(db.get_session(&sid).await.unwrap().is_some());
        assert!(db.get_conversation(&conv.id_string()).await.is_ok());
    }

    #[tokio::test]
    async fn read_path_never_deletes() {
        let pm = manager().await;
        let sid = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&sid, "kept").await.unwrap();
        pm.add_message(&conv.id_string(), "user", "hello")
            .await
            .unwrap();
        let live = pm.sessions().get_session(&sid).await.unwrap();
        *live.last_accessed.write().await = chrono::Utc::now() - chrono::Duration::minutes(65);

        assert!(matches!(
            pm.get_session(&sid).await,
            Err(PersistentSessionError::Session(SessionError::Expired))
        ));
        assert!(pm.db().get_session(&sid).await.unwrap().is_some());
        assert_eq!(
            pm.db().get_messages(&conv.id_string()).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn non_session_ids_are_not_looked_up_or_adopted() {
        let pm = manager().await;
        for id in ["random-xyz-1", "", "../x"] {
            assert!(matches!(
                pm.get_session(id).await,
                Err(PersistentSessionError::Session(SessionError::NotFound))
            ));
        }
        assert_eq!(pm.sessions().session_count().await, 0);
    }

    #[tokio::test]
    async fn delete_session_ends_the_live_session_and_its_rows() {
        let pm = manager().await;
        let sid = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&sid, "gone").await.unwrap();
        pm.delete_session(&sid).await.unwrap();
        assert!(matches!(
            pm.sessions().get_session(&sid).await,
            Err(SessionError::NotFound)
        ));
        assert!(pm.db().get_session(&sid).await.unwrap().is_none());
        assert!(pm.db().get_conversation(&conv.id_string()).await.is_err());
        assert!(matches!(
            pm.delete_session(&sid).await,
            Err(PersistentSessionError::Session(SessionError::NotFound))
        ));
    }

    #[tokio::test]
    async fn authorize_hides_foreign_and_missing_objects() {
        let pm = manager().await;
        let s1 = pm.create_session(None).await.unwrap();
        let s2 = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&s1, "mine").await.unwrap();
        let msg = pm
            .add_message(&conv.id_string(), "user", "private")
            .await
            .unwrap();
        let (c, m) = (conv.id_string(), msg.id_string());

        assert!(pm.authorize_conversation(&s1, &c).await.is_ok());
        assert!(pm.authorize_message(&s1, &m).await.is_ok());
        let text = |r: Result<(), PersistentSessionError>| match r {
            Err(PersistentSessionError::Database(DbError::NotFound(t))) => t,
            other => panic!("expected NotFound, got {other:?}"),
        };
        let foreign = text(pm.authorize_conversation(&s2, &c).await.map(|_| ()));
        let missing = text(
            pm.authorize_conversation(&s2, "no-such-id")
                .await
                .map(|_| ()),
        );
        assert_eq!(foreign, missing);
        assert!(!foreign.contains(&c));
        let foreign = text(pm.authorize_message(&s2, &m).await.map(|_| ()));
        let missing = text(pm.authorize_message(&s2, "no-such-id").await.map(|_| ()));
        assert_eq!(foreign, missing);
        // A caller that is not live is a session error, not an object error.
        assert!(matches!(
            pm.authorize_conversation("random-xyz-1", &c).await,
            Err(PersistentSessionError::Session(SessionError::NotFound))
        ));
    }

    #[tokio::test]
    async fn settings_rows_only_for_live_sessions() {
        let pm = manager().await;
        assert!(matches!(
            pm.get_settings("random-xyz-1").await,
            Err(PersistentSessionError::Session(SessionError::NotFound))
        ));
        assert!(
            pm.patch_settings("random-xyz-1", None, Some("m"), None, None)
                .await
                .is_err()
        );
        let row: Option<Settings> = pm
            .db()
            .db()
            .select(("settings", "random-xyz-1"))
            .await
            .unwrap();
        assert!(row.is_none());
        // A live session can patch before it ever read its settings.
        let sid = pm.create_session(None).await.unwrap();
        let patched = pm
            .patch_settings(&sid, None, Some("m"), None, None)
            .await
            .unwrap();
        assert_eq!(patched.model, "m");
    }

    /// Poll `f` until it returns true, up to 250 x 20 ms. No timing assertion.
    async fn wait_until<F, Fut>(f: F) -> bool
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        for _ in 0..250 {
            if f().await {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        false
    }

    /// A stored session row (with one conversation) that is not live.
    async fn stored_only(
        db: &DbConnection,
        last_accessed_ago: chrono::Duration,
    ) -> (String, String) {
        let sid = uuid::Uuid::new_v4().to_string();
        db.set_session_last_accessed(&sid, chrono::Utc::now() - last_accessed_ago)
            .await
            .unwrap();
        let conv = db.create_conversation(&sid, "stored").await.unwrap();
        (sid, conv.id_string())
    }

    #[tokio::test]
    async fn sweep_deletes_stale_non_live_rows() {
        let pm = manager().await;
        let (stale, stale_conv) = stored_only(pm.db(), chrono::Duration::hours(2)).await;
        let (recent, recent_conv) = stored_only(pm.db(), chrono::Duration::minutes(5)).await;

        let stats = pm.sweep().await.unwrap();
        assert_eq!(stats.expired_deleted, 1);
        assert!(pm.db().get_session(&stale).await.unwrap().is_none());
        assert!(pm.db().get_conversation(&stale_conv).await.is_err());
        // Within the TTL: kept, still adoptable after a restart.
        assert!(pm.db().get_session(&recent).await.unwrap().is_some());
        assert!(pm.db().get_conversation(&recent_conv).await.is_ok());
    }

    #[tokio::test]
    async fn sweep_keeps_and_refreshes_live_sessions() {
        let pm = manager().await;
        let sid = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&sid, "live").await.unwrap();
        pm.db()
            .set_session_last_accessed(&sid, chrono::Utc::now() - chrono::Duration::hours(2))
            .await
            .unwrap();

        let stats = pm.sweep().await.unwrap();
        assert_eq!((stats.refreshed, stats.expired_deleted), (1, 0));
        let row = pm.db().get_session(&sid).await.unwrap().expect("row kept");
        let live = *pm
            .sessions()
            .get_session(&sid)
            .await
            .unwrap()
            .last_accessed
            .read()
            .await;
        let drift = live.signed_duration_since(row.last_accessed_chrono());
        assert!(drift.num_seconds().abs() < 1, "{drift}");
        assert!(pm.db().get_conversation(&conv.id_string()).await.is_ok());
    }

    #[tokio::test]
    async fn sweep_removes_orphans() {
        let pm = manager().await;
        let db = pm.db();
        // A message whose conversation does not exist.
        let ghost = Message::user("ghost".into(), "orphan".into());
        let ghost_id = ghost.id_string();
        let _: Option<Message> = db
            .db()
            .create(ghost.id.clone())
            .content(ghost)
            .await
            .unwrap();
        // Settings and a conversation of a session without a row.
        db.get_or_create_settings("no-row-session").await.unwrap();
        let orphan_conv = db.create_conversation("no-row-session", "t").await.unwrap();
        db.add_message(&orphan_conv.id_string(), "user", "x")
            .await
            .unwrap();
        // A live session's rows stay.
        let sid = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&sid, "kept").await.unwrap();
        pm.add_message(&conv.id_string(), "user", "kept")
            .await
            .unwrap();
        pm.get_settings(&sid).await.unwrap();

        let stats = pm.sweep().await.unwrap();
        assert!(stats.orphans_deleted >= 3, "{stats:?}");
        assert!(matches!(
            db.get_message(&ghost_id).await,
            Err(DbError::NotFound(_))
        ));
        let row: Option<Settings> = db
            .db()
            .select(("settings", "no-row-session"))
            .await
            .unwrap();
        assert!(row.is_none());
        assert!(db.get_conversation(&orphan_conv.id_string()).await.is_err());
        assert!(
            db.get_messages(&orphan_conv.id_string())
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(db.get_messages(&conv.id_string()).await.unwrap().len(), 1);
        let row: Option<Settings> = db.db().select(("settings", sid.as_str())).await.unwrap();
        assert!(row.is_some());
        // A second sweep finds nothing left to do.
        assert_eq!(pm.sweep().await.unwrap().orphans_deleted, 0);
    }

    #[tokio::test]
    async fn removal_notification_deletes_rows() {
        let pm = Arc::new(manager().await);
        pm.start_background_tasks();
        let sid = pm.create_session(None).await.unwrap();
        let conv = pm.create_conversation(&sid, "gone").await.unwrap();
        pm.get_settings(&sid).await.unwrap();

        pm.sessions().delete_session(&sid).await.unwrap();
        let (db, c) = (pm.db().clone(), conv.id_string());
        assert!(
            wait_until(|| {
                let (db, sid, c) = (db.clone(), sid.clone(), c.clone());
                async move {
                    db.get_session(&sid).await.unwrap().is_none()
                        && db.get_conversation(&c).await.is_err()
                }
            })
            .await,
            "the removed session's rows were not deleted"
        );
    }

    #[test]
    fn db_path_inside_the_data_root_is_refused() {
        let root_dir = tempfile::tempdir().unwrap();
        let r = root_dir.path().canonicalize().unwrap();
        for inside in [r.join("db"), r.clone(), r.join("a/../db"), r.join("./x/db")] {
            let err = check_db_path_outside_data_root(&inside, &r).unwrap_err();
            assert!(
                err.contains("P2A_DB_PATH") && err.contains("data root"),
                "{err}"
            );
        }
        let other = tempfile::tempdir().unwrap();
        for outside in [other.path().join("db"), r.join("../p2a-outside-db")] {
            assert!(
                check_db_path_outside_data_root(&outside, &r).is_ok(),
                "{}",
                outside.display()
            );
        }
        // A symlink outside the root that points into it does not hide the
        // path the store would really be created at.
        #[cfg(unix)]
        {
            let link = other.path().join("link-into-root");
            std::os::unix::fs::symlink(&r, &link).unwrap();
            assert!(check_db_path_outside_data_root(&link.join("db"), &r).is_err());
        }
    }

    #[test]
    fn sweep_period_follows_the_ttl() {
        assert_eq!(sweep_period(1).as_secs(), 60);
        assert_eq!(sweep_period(10).as_secs(), 300);
        assert_eq!(sweep_period(60).as_secs(), 600);
        assert_eq!(sweep_period(u64::MAX).as_secs(), 600);
    }

    #[tokio::test]
    async fn test_create_session() {
        let manager = manager().await;

        let session_id = manager
            .create_session(None)
            .await
            .expect("Failed to create session");

        assert!(!session_id.is_empty());

        // Verify we can get the session
        let session = manager
            .get_session(&session_id)
            .await
            .expect("Failed to get session");
        assert_eq!(session.id, session_id);
    }

    #[tokio::test]
    async fn test_conversation_workflow() {
        let manager = manager().await;

        // Create session
        let session_id = manager
            .create_session(Some("test-user".to_string()))
            .await
            .expect("Failed to create session");

        // Create conversation
        let conv = manager
            .create_conversation(&session_id, "Test Chat")
            .await
            .expect("Failed to create conversation");

        assert_eq!(conv.title, "Test Chat");
        assert_eq!(conv.session_id, session_id);

        // Add messages
        let conv_id = conv.id_string();
        let _msg1 = manager
            .add_message(&conv_id, "user", "Hello!")
            .await
            .expect("Failed to add user message");

        let _msg2 = manager
            .add_message(&conv_id, "assistant", "Hi there!")
            .await
            .expect("Failed to add assistant message");

        // Get messages
        let messages = manager
            .get_messages(&conv_id)
            .await
            .expect("Failed to get messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello!");
        assert_eq!(messages[1].content, "Hi there!");

        // Get conversation with messages
        let full = manager
            .get_conversation_with_messages(&conv_id)
            .await
            .expect("Failed to get conversation with messages");

        assert_eq!(full.conversation.id_string(), conv_id);
        assert_eq!(full.messages.len(), 2);

        // Delete conversation
        manager
            .delete_conversation(&conv_id)
            .await
            .expect("Failed to delete conversation");

        // Verify deleted
        let result = manager.get_conversation(&conv_id).await;
        assert!(result.is_err());
    }
}
