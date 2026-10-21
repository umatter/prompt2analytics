//! Session persistence operations

use chrono::{DateTime, Utc};
use surrealdb::RecordId;
use surrealdb::sql::Datetime;

use super::connection::{DbConnection, DbError};
use super::models::{Conversation, DbSession, Settings};

/// One row of `SELECT count() ... GROUP ALL`.
#[derive(serde::Deserialize)]
struct CountRow {
    count: i64,
}

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

    /// Set a session row's `last_accessed` to `at` (the live session's own
    /// value, flushed by the sweep). A missing row is created.
    pub async fn set_session_last_accessed(
        &self,
        id: &str,
        at: DateTime<Utc>,
    ) -> Result<(), DbError> {
        let mut result = self
            .db()
            .query("UPDATE $id SET last_accessed = $at RETURN AFTER")
            .bind(("id", RecordId::from(("sessions", id))))
            .bind(("at", Datetime::from(at)))
            .await?;
        let updated: Vec<DbSession> = result.take(0)?;
        if updated.is_empty() {
            let mut row = DbSession::new(id.to_string(), None);
            row.last_accessed = Datetime::from(at);
            self.upsert_session(&row).await?;
        }
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

        // Delete each conversation (which also deletes messages and tool
        // calls). One that is already gone was deleted concurrently (the
        // removal listener and DELETE /api/sessions/{id} can both run).
        for conv in conversations {
            match self.delete_conversation(&conv.id_string()).await {
                Ok(()) | Err(DbError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
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
        self.forget_session_bytes(id);

        Ok(())
    }

    /// Ids of the session rows whose `last_accessed` is before `cutoff`.
    pub async fn session_ids_last_accessed_before(
        &self,
        cutoff: DateTime<Utc>,
    ) -> Result<Vec<String>, DbError> {
        let mut result = self
            .db()
            .query("SELECT * FROM sessions WHERE last_accessed < $cutoff")
            .bind(("cutoff", Datetime::from(cutoff)))
            .await?;
        let rows: Vec<DbSession> = result.take(0)?;
        Ok(rows.iter().map(DbSession::id_string).collect())
    }

    /// Number of conversations the session holds.
    pub async fn count_conversations(&self, session_id: &str) -> Result<usize, DbError> {
        let mut result = self
            .db()
            .query("SELECT count() FROM conversations WHERE session_id = $session_id GROUP ALL")
            .bind(("session_id", session_id.to_string()))
            .await?;
        let count: Option<CountRow> = result.take(0)?;
        Ok(count.map_or(0, |c| usize::try_from(c.count).unwrap_or(0)))
    }

    /// Bytes the session's conversations store: titles, message contents
    /// and tool-call arguments, results and errors, as UTF-8 bytes (summed
    /// here; SurrealQL's `string::len` counts characters). Seeds the
    /// `session_db_bytes` counter.
    pub async fn session_stored_bytes(&self, session_id: &str) -> Result<u64, DbError> {
        #[derive(serde::Deserialize)]
        struct Content {
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct ToolText {
            arguments: String,
            result: Option<String>,
            error: Option<String>,
        }
        let len = |s: &str| s.len() as u64;

        let conversations = self.list_conversations(session_id).await?;
        let ids: Vec<String> = conversations.iter().map(|c| c.id_string()).collect();
        let mut total = conversations
            .iter()
            .map(|c| len(&c.title))
            .fold(0u64, u64::saturating_add);
        if ids.is_empty() {
            return Ok(total);
        }
        let mut result = self
            .db()
            .query("SELECT content FROM messages WHERE conversation_id INSIDE $ids")
            .query(
                "SELECT arguments, result, error FROM tool_calls WHERE conversation_id INSIDE $ids",
            )
            .bind(("ids", ids))
            .await?;
        let contents: Vec<Content> = result.take(0)?;
        let tool_texts: Vec<ToolText> = result.take(1)?;
        total = contents
            .iter()
            .map(|m| len(&m.content))
            .fold(total, u64::saturating_add);
        total = tool_texts
            .iter()
            .map(|t| {
                len(&t.arguments)
                    .saturating_add(t.result.as_deref().map_or(0, len))
                    .saturating_add(t.error.as_deref().map_or(0, len))
            })
            .fold(total, u64::saturating_add);
        Ok(total)
    }

    /// Delete the rows that no session or conversation owns any more:
    /// conversations, settings and dataset metadata whose session row is
    /// missing, and messages and tool calls whose conversation is missing.
    /// Rows written after the call started are left alone, so a session or
    /// conversation being created concurrently cannot lose its first rows.
    /// Returns the number of rows deleted.
    pub async fn delete_orphans(&self) -> Result<usize, DbError> {
        #[derive(serde::Deserialize)]
        struct SessionRef {
            session_id: String,
        }

        let started = Datetime::from(Utc::now());
        let session_ids: std::collections::HashSet<String> = {
            let mut result = self.db().query("SELECT * FROM sessions").await?;
            let rows: Vec<DbSession> = result.take(0)?;
            rows.iter().map(DbSession::id_string).collect()
        };
        let mut deleted = 0usize;

        // Conversations (with their messages and tool calls).
        let mut result = self
            .db()
            .query("SELECT * FROM conversations WHERE created_at < $started")
            .bind(("started", started.clone()))
            .await?;
        let conversations: Vec<Conversation> = result.take(0)?;
        for conv in conversations
            .iter()
            .filter(|c| !session_ids.contains(&c.session_id))
        {
            match self.delete_conversation(&conv.id_string()).await {
                Ok(()) => deleted += 1,
                Err(DbError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
            self.forget_session_bytes(&conv.session_id);
        }

        let known: Vec<String> = session_ids.into_iter().collect();
        // Settings and dataset metadata of sessions without a row.
        for (table, stamp) in [("settings", "updated_at"), ("dataset_meta", "loaded_at")] {
            let mut result = self
                .db()
                .query(format!(
                    "SELECT session_id FROM {table} \
                     WHERE session_id NOTINSIDE $known AND {stamp} < $started"
                ))
                .query(format!(
                    "DELETE FROM {table} WHERE session_id NOTINSIDE $known AND {stamp} < $started"
                ))
                .bind(("known", known.clone()))
                .bind(("started", started.clone()))
                .await?
                .check()?;
            let rows: Vec<SessionRef> = result.take(0)?;
            deleted += rows.len();
        }

        // Messages and tool calls of conversations that no longer exist.
        let conversation_ids: Vec<String> = {
            let mut result = self.db().query("SELECT * FROM conversations").await?;
            let rows: Vec<Conversation> = result.take(0)?;
            rows.iter().map(|c| c.id_string()).collect()
        };
        for (table, stamp) in [("messages", "created_at"), ("tool_calls", "started_at")] {
            let mut result = self
                .db()
                .query(format!(
                    "SELECT count() FROM {table} \
                     WHERE conversation_id NOTINSIDE $known AND {stamp} < $started GROUP ALL"
                ))
                .query(format!(
                    "DELETE FROM {table} \
                     WHERE conversation_id NOTINSIDE $known AND {stamp} < $started"
                ))
                .bind(("known", conversation_ids.clone()))
                .bind(("started", started.clone()))
                .await?
                .check()?;
            let count: Option<CountRow> = result.take(0)?;
            deleted += count.map_or(0, |c| usize::try_from(c.count).unwrap_or(0));
        }

        if deleted > 0 {
            tracing::info!(deleted, "Deleted orphaned conversation-store rows");
        }
        Ok(deleted)
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
        for (what, value) in [
            ("provider", Some(&settings.provider)),
            ("model", Some(&settings.model)),
            ("base_url", settings.base_url.as_ref()),
            ("system_prompt", settings.system_prompt.as_ref()),
        ] {
            self.check_message_bytes(what, "setting", value.map_or(0, String::len))?;
        }
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
        for (what, value) in [("provider", provider), ("model", model)] {
            self.check_message_bytes(what, "setting", value.map_or(0, str::len))?;
        }
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
