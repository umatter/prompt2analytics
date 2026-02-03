//! Session persistence operations

use chrono::Utc;
use surrealdb::RecordId;
use surrealdb::sql::Datetime;

use super::connection::{DbConnection, DbError};
use super::models::{DbSession, Settings};

impl DbConnection {
    // ==================== Session Operations ====================

    /// Create or update a session in the database
    pub async fn upsert_session(&self, session: &DbSession) -> Result<DbSession, DbError> {
        let result: Option<DbSession> = self
            .db()
            .upsert(session.id.clone())
            .content(session.clone())
            .await?;

        result.ok_or_else(|| DbError::Query("Failed to upsert session".to_string()))
    }

    /// Get a session by ID
    pub async fn get_session(&self, id: &str) -> Result<Option<DbSession>, DbError> {
        let session: Option<DbSession> = self.db().select(("sessions", id)).await?;
        Ok(session)
    }

    /// Update session's last_accessed timestamp
    pub async fn touch_session(&self, id: &str) -> Result<(), DbError> {
        let id_owned = id.to_string();
        self.db()
            .query("UPDATE sessions SET last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", id_owned.as_str()))))
            .await?;
        Ok(())
    }

    /// Update session's active conversation
    pub async fn set_active_conversation(
        &self,
        session_id: &str,
        conversation_id: Option<&str>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET active_conversation_id = $conv_id, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("conv_id", conversation_id.map(|s| s.to_string())))
            .await?;
        Ok(())
    }

    /// Update session's dataset names
    pub async fn update_session_datasets(
        &self,
        session_id: &str,
        dataset_names: Vec<String>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET dataset_names = $names, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("names", dataset_names))
            .await?;
        Ok(())
    }

    /// Update session's global seed
    pub async fn update_session_seed(
        &self,
        session_id: &str,
        seed: Option<i64>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET global_seed = $seed, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("seed", seed))
            .await?;
        Ok(())
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<DbSession>, DbError> {
        let mut result = self
            .db()
            .query("SELECT * FROM sessions ORDER BY last_accessed DESC")
            .await?;

        let sessions: Vec<DbSession> = result.take(0)?;
        Ok(sessions)
    }

    /// Delete a session and all associated data
    pub async fn delete_session(&self, id: &str) -> Result<(), DbError> {
        // Get all conversations for this session
        let conversations = self.list_conversations(id).await?;

        // Delete each conversation (which also deletes messages and tool calls)
        for conv in conversations {
            self.delete_conversation(&conv.id_string()).await?;
        }

        // Delete settings
        let _: Option<Settings> = self.db().delete(("settings", id)).await?;

        // Delete dataset metadata
        let id_owned = id.to_string();
        self.db()
            .query("DELETE FROM dataset_meta WHERE session_id = $id")
            .bind(("id", id_owned))
            .await?;

        // Delete session
        let _: Option<DbSession> = self.db().delete(("sessions", id)).await?;

        Ok(())
    }

    /// Clean up sessions older than the given number of days
    pub async fn cleanup_old_sessions(&self, days: u32) -> Result<u32, DbError> {
        let cutoff = Datetime::from(Utc::now() - chrono::Duration::days(days as i64));

        // Get old sessions
        let mut result = self
            .db()
            .query("SELECT id FROM sessions WHERE last_accessed < $cutoff")
            .bind(("cutoff", cutoff))
            .await?;

        #[derive(serde::Deserialize)]
        struct IdOnly {
            id: RecordId,
        }

        let old_sessions: Vec<IdOnly> = result.take(0)?;
        let count = old_sessions.len() as u32;

        // Delete each old session
        for session in old_sessions {
            let id_str = session.id.key().to_string();
            if let Err(e) = self.delete_session(&id_str).await {
                tracing::warn!("Failed to delete old session {}: {}", id_str, e);
            }
        }

        if count > 0 {
            tracing::info!("Cleaned up {} old sessions", count);
        }

        Ok(count)
    }

    // ==================== Settings Operations ====================

    /// Get or create settings for a session
    pub async fn get_or_create_settings(&self, session_id: &str) -> Result<Settings, DbError> {
        // Try to get existing settings
        let existing: Option<Settings> = self.db().select(("settings", session_id)).await?;

        if let Some(settings) = existing {
            return Ok(settings);
        }

        // Create default settings
        let settings = Settings::default_for_session(session_id.to_string());
        let created: Option<Settings> = self
            .db()
            .create(settings.id.clone())
            .content(settings)
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to create settings".to_string()))
    }

    /// Update settings for a session
    pub async fn update_settings(&self, settings: &Settings) -> Result<Settings, DbError> {
        let mut updated_settings = settings.clone();
        updated_settings.updated_at = Datetime::from(Utc::now());
        let session_id = settings.session_id.clone();

        let result: Option<Settings> = self
            .db()
            .update(("settings", &session_id))
            .content(updated_settings)
            .await?;

        result.ok_or_else(|| DbError::NotFound("Settings not found".to_string()))
    }

    /// Update specific settings fields
    pub async fn patch_settings(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<i32>,
    ) -> Result<Settings, DbError> {
        // Build update SET clauses dynamically
        let mut set_parts = vec!["updated_at = time::now()".to_string()];

        if provider.is_some() {
            set_parts.push("provider = $provider".to_string());
        }
        if model.is_some() {
            set_parts.push("model = $model".to_string());
        }
        if temperature.is_some() {
            set_parts.push("temperature = $temperature".to_string());
        }
        if max_tokens.is_some() {
            set_parts.push("max_tokens = $max_tokens".to_string());
        }

        let query = format!(
            "UPDATE settings SET {} WHERE id = $id RETURN AFTER",
            set_parts.join(", ")
        );

        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query(&query)
            .bind((
                "id",
                RecordId::from(("settings", session_id_owned.as_str())),
            ))
            .bind(("provider", provider.map(|s| s.to_string())))
            .bind(("model", model.map(|s| s.to_string())))
            .bind(("temperature", temperature))
            .bind(("max_tokens", max_tokens))
            .await?;

        let settings: Option<Settings> = result.take(0)?;
        settings.ok_or_else(|| DbError::NotFound("Settings not found".to_string()))
    }
}
