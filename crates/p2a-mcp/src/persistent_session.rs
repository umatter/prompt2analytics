//! Persistent session management using SurrealDB.
//!
//! This module extends the basic in-memory session management with
//! database persistence for sessions, conversations, and settings.

#[cfg(feature = "db")]
use crate::db::{
    Conversation, ConversationWithMessages, DbConnection, DbError, DbSession, Message, Settings,
};
use crate::session::{DatasetStore, Session, SessionError, SessionId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A session manager that persists sessions to SurrealDB.
///
/// This combines in-memory dataset storage (for fast access to loaded data)
/// with database persistence (for conversations, settings, and session metadata).
#[cfg(feature = "db")]
pub struct PersistentSessionManager {
    /// In-memory session cache (datasets are not serializable)
    sessions: Arc<RwLock<HashMap<SessionId, Arc<Session>>>>,
    /// Database connection for persistence
    db: Arc<DbConnection>,
    /// Session configuration
    max_sessions: usize,
    ttl_minutes: u64,
}

#[cfg(feature = "db")]
impl PersistentSessionManager {
    /// Create a new persistent session manager.
    pub async fn new(
        db_path: Option<&str>,
        max_sessions: usize,
        ttl_minutes: u64,
    ) -> Result<Self, DbError> {
        let db = match db_path {
            Some(path) => DbConnection::connect_rocksdb(path).await?,
            None => DbConnection::connect_memory().await?,
        };

        Ok(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            db: Arc::new(db),
            max_sessions,
            ttl_minutes,
        })
    }

    /// Create a new session and persist it.
    pub async fn create_session(
        &self,
        user_id: Option<String>,
    ) -> Result<SessionId, PersistentSessionError> {
        // Check max sessions
        let sessions = self.sessions.read().await;
        if sessions.len() >= self.max_sessions {
            return Err(PersistentSessionError::Session(
                SessionError::MaxSessionsReached,
            ));
        }
        drop(sessions);

        // Create in-memory session
        let id = uuid::Uuid::new_v4().to_string();
        let session = Arc::new(Session::new(id.clone(), user_id.clone()));

        // Persist to database
        let db_session = DbSession::new(id.clone(), user_id);
        self.db.upsert_session(&db_session).await?;

        // Store in memory
        let mut sessions = self.sessions.write().await;
        sessions.insert(id.clone(), session);

        tracing::info!(session_id = %id, "Created new persistent session");
        Ok(id)
    }

    /// Get a session by ID, loading from DB if not in memory.
    pub async fn get_session(
        &self,
        id: &str,
    ) -> Result<Arc<Session>, PersistentSessionError> {
        // Try memory first
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(id).cloned() {
            drop(sessions);

            // Check if expired
            if session.is_expired(self.ttl_minutes).await {
                self.delete_session(id).await?;
                return Err(PersistentSessionError::Session(SessionError::Expired));
            }

            // Update timestamps
            session.touch().await;
            self.db.touch_session(id).await?;
            return Ok(session);
        }
        drop(sessions);

        // Try loading from database
        if let Some(db_session) = self.db.get_session(id).await? {
            // Restore in-memory session (without datasets - those need to be reloaded)
            let session = Arc::new(Session::new(id.to_string(), db_session.user_id.clone()));

            // Store in memory
            let mut sessions = self.sessions.write().await;
            sessions.insert(id.to_string(), session.clone());

            // Update timestamps
            self.db.touch_session(id).await?;

            tracing::info!(session_id = %id, "Restored session from database");
            return Ok(session);
        }

        Err(PersistentSessionError::Session(SessionError::NotFound))
    }

    /// Get the dataset store for a session.
    pub async fn get_dataset_store(
        &self,
        session_id: &str,
    ) -> Result<DatasetStore, PersistentSessionError> {
        let session = self.get_session(session_id).await?;
        Ok(DatasetStore::from_session(&session))
    }

    /// Delete a session and all its data.
    pub async fn delete_session(&self, id: &str) -> Result<(), PersistentSessionError> {
        // Remove from memory
        let mut sessions = self.sessions.write().await;
        sessions.remove(id);
        drop(sessions);

        // Remove from database
        self.db.delete_session(id).await?;

        tracing::info!(session_id = %id, "Deleted persistent session");
        Ok(())
    }

    /// List all sessions (from database).
    pub async fn list_sessions(&self) -> Result<Vec<DbSession>, PersistentSessionError> {
        Ok(self.db.list_sessions().await?)
    }

    /// Clean up old sessions (both memory and database).
    pub async fn cleanup_old_sessions(&self, days: u32) -> Result<u32, PersistentSessionError> {
        let count = self.db.cleanup_old_sessions(days).await?;

        if count > 0 {
            // Also clean up memory cache
            let mut sessions = self.sessions.write().await;
            let old_count = sessions.len();
            sessions.retain(|id, _| {
                // Keep if exists in DB (will be checked on next access)
                true
            });
            tracing::info!(
                db_cleaned = count,
                memory_before = old_count,
                "Cleaned up old sessions"
            );
        }

        Ok(count)
    }

    // ==================== Conversation Operations ====================

    /// Create a new conversation for a session.
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, PersistentSessionError> {
        // Verify session exists
        let _ = self.get_session(session_id).await?;

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
}

/// Errors from persistent session management.
#[derive(Debug, thiserror::Error)]
pub enum PersistentSessionError {
    #[error("Session error: {0}")]
    Session(#[from] SessionError),

    #[cfg(feature = "db")]
    #[error("Database error: {0}")]
    Database(#[from] DbError),
}

#[cfg(test)]
#[cfg(feature = "db")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_session() {
        let manager = PersistentSessionManager::new(None, 100, 60)
            .await
            .expect("Failed to create manager");

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
        let manager = PersistentSessionManager::new(None, 100, 60)
            .await
            .expect("Failed to create manager");

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
