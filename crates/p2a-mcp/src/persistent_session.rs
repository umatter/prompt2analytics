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

    /// End a session: remove it from the live registry and delete all of its
    /// rows. `NotFound` only when it was neither live nor stored.
    pub async fn delete_session(&self, id: &str) -> Result<(), PersistentSessionError> {
        let was_live = match self.sessions.delete_session(id).await {
            Ok(()) => true,
            Err(SessionError::NotFound) => false,
            Err(e) => return Err(e.into()),
        };
        let was_stored = self.db.get_session(id).await?.is_some();
        self.db.delete_session(id).await?;
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

    /// Get or create settings for a session.
    pub async fn get_settings(&self, session_id: &str) -> Result<Settings, PersistentSessionError> {
        Ok(self.db.get_or_create_settings(session_id).await?)
    }

    /// Update settings for a session.
    pub async fn update_settings(
        &self,
        settings: &Settings,
    ) -> Result<Settings, PersistentSessionError> {
        Ok(self.db.update_settings(settings).await?)
    }

    /// Patch specific settings fields.
    pub async fn patch_settings(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<i32>,
    ) -> Result<Settings, PersistentSessionError> {
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

/// Reload texts. They are fixed so that no server path reaches a caller.
const RELOAD_NO_SOURCE: &str = "no source path stored";
const RELOAD_OUTSIDE_JAIL: &str = "path is not inside this session's data directory";
const RELOAD_LOAD_FAILED: &str = "the file could not be loaded";

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
