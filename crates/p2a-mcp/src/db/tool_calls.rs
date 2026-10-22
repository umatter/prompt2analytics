//! Tool call CRUD operations

use surrealdb::RecordId;

use super::connection::{DbConnection, DbError};
use super::models::{ToolCall, ToolCallStatus};

impl DbConnection {
    // ==================== Tool Call Operations ====================

    /// Create a new tool call record. Its conversation must exist
    /// (`NotFound`), and it is refused with `MemoryPressure` above the RSS
    /// soft limit. Its texts are cut at a char boundary to `message_bytes`
    /// and to the bytes the session has left under `session_db_bytes`, never
    /// refused.
    pub async fn create_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        self.check_memory()?;
        let conv = self.get_conversation(&tool_call.conversation_id).await?;
        let tool_call = self
            .fit_tool_call_texts(&conv.session_id, tool_call.clone())
            .await?;
        let created: Option<ToolCall> = self
            .db()
            .create(tool_call.id.clone())
            .content(tool_call)
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to create tool call".to_string()))
    }

    /// Update an existing tool call (typically to set result/error/status).
    /// Its texts are cut to fit, like those of a new one.
    pub async fn update_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        let id_str = tool_call.id_string();
        let session_id = self.tool_call_session(&tool_call.conversation_id).await?;
        let tool_call = self
            .fit_tool_call_texts(&session_id, tool_call.clone())
            .await?;

        let result: Option<ToolCall> = self
            .db()
            .update(("tool_calls", id_str.as_str()))
            .content(tool_call)
            .await?;

        result.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id_str)))
    }

    /// Server-produced tool-call text (arguments, a result or an error) cut
    /// at a char boundary to `message_bytes` and, with a `session_db_bytes`
    /// budget, to the bytes session `session_id` has left, where it is then
    /// counted. Never refused.
    async fn fit_tool_call_text(&self, session_id: &str, text: &str) -> Result<String, DbError> {
        let text = self.truncate_server_text(text);
        self.fit_session_bytes(session_id, text).await
    }

    /// [`fit_tool_call_text`](Self::fit_tool_call_text) for every text field
    /// of `tool_call`.
    async fn fit_tool_call_texts(
        &self,
        session_id: &str,
        mut tool_call: ToolCall,
    ) -> Result<ToolCall, DbError> {
        tool_call.arguments = self
            .fit_tool_call_text(session_id, &tool_call.arguments)
            .await?;
        if let Some(result) = tool_call.result.take() {
            tool_call.result = Some(self.fit_tool_call_text(session_id, &result).await?);
        }
        if let Some(error) = tool_call.error.take() {
            tool_call.error = Some(self.fit_tool_call_text(session_id, &error).await?);
        }
        Ok(tool_call)
    }

    /// The session that owns conversation `conversation_id`, read only when
    /// the `session_db_bytes` budget needs it (the empty string otherwise).
    async fn tool_call_session(&self, conversation_id: &str) -> Result<String, DbError> {
        if !self.tracks_session_bytes() {
            return Ok(String::new());
        }
        Ok(self.get_conversation(conversation_id).await?.session_id)
    }

    /// The session that owns tool call `id`, read only when the
    /// `session_db_bytes` budget needs it (the empty string otherwise).
    async fn tool_call_owner(&self, id: &str) -> Result<String, DbError> {
        if !self.tracks_session_bytes() {
            return Ok(String::new());
        }
        let tool_call = self.get_tool_call(id).await?;
        self.tool_call_session(&tool_call.conversation_id).await
    }

    /// Mark a tool call as running
    pub async fn mark_tool_call_running(&self, id: &str) -> Result<ToolCall, DbError> {
        let id_owned = id.to_string();
        let mut result = self
            .db()
            .query("UPDATE tool_calls SET status = $status WHERE id = $id RETURN AFTER")
            .bind(("id", RecordId::from(("tool_calls", id_owned.as_str()))))
            .bind(("status", ToolCallStatus::Running))
            .await?;

        let updated: Option<ToolCall> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))
    }

    /// Mark a tool call as completed successfully. The result is cut to
    /// fit, never refused.
    pub async fn complete_tool_call(
        &self,
        id: &str,
        result_str: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
        let session_id = self.tool_call_owner(id).await?;
        let result_str = self.fit_tool_call_text(&session_id, result_str).await?;
        let id_owned = id.to_string();
        let mut result = self
            .db()
            .query(
                r#"UPDATE tool_calls SET
                    status = $status,
                    result = $result,
                    completed_at = time::now(),
                    duration_ms = $duration_ms
                WHERE id = $id RETURN AFTER"#,
            )
            .bind(("id", RecordId::from(("tool_calls", id_owned.as_str()))))
            .bind(("status", ToolCallStatus::Success))
            .bind(("result", result_str))
            .bind(("duration_ms", duration_ms))
            .await?;

        let updated: Option<ToolCall> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))
    }

    /// Mark a tool call as failed. The error text is cut to fit, never
    /// refused.
    pub async fn fail_tool_call(
        &self,
        id: &str,
        error: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
        let session_id = self.tool_call_owner(id).await?;
        let error = self.fit_tool_call_text(&session_id, error).await?;
        let id_owned = id.to_string();
        let mut result = self
            .db()
            .query(
                r#"UPDATE tool_calls SET
                    status = $status,
                    error = $error,
                    completed_at = time::now(),
                    duration_ms = $duration_ms
                WHERE id = $id RETURN AFTER"#,
            )
            .bind(("id", RecordId::from(("tool_calls", id_owned.as_str()))))
            .bind(("status", ToolCallStatus::Error))
            .bind(("error", error))
            .bind(("duration_ms", duration_ms))
            .await?;

        let updated: Option<ToolCall> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))
    }

    /// Get all tool calls for a message
    pub async fn get_tool_calls_for_message(
        &self,
        message_id: &str,
    ) -> Result<Vec<ToolCall>, DbError> {
        let message_id_owned = message_id.to_string();
        let mut result = self
            .db()
            .query(
                "SELECT * FROM tool_calls WHERE message_id = $message_id ORDER BY started_at ASC",
            )
            .bind(("message_id", message_id_owned))
            .await?;

        let tool_calls: Vec<ToolCall> = result.take(0)?;
        Ok(tool_calls)
    }

    /// Get all tool calls for a conversation
    pub async fn get_tool_calls_for_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ToolCall>, DbError> {
        let conv_id_owned = conversation_id.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM tool_calls WHERE conversation_id = $conversation_id ORDER BY started_at ASC")
            .bind(("conversation_id", conv_id_owned))
            .await?;

        let tool_calls: Vec<ToolCall> = result.take(0)?;
        Ok(tool_calls)
    }

    /// Get tool call by ID
    pub async fn get_tool_call(&self, id: &str) -> Result<ToolCall, DbError> {
        let tool_call: Option<ToolCall> = self.db().select(("tool_calls", id)).await?;
        tool_call.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))
    }

    /// Delete all tool calls for a message
    pub async fn delete_tool_calls_for_message(&self, message_id: &str) -> Result<u32, DbError> {
        let message_id_owned = message_id.to_string();

        // Count before deletion
        let mut result = self
            .db()
            .query("SELECT count() FROM tool_calls WHERE message_id = $message_id GROUP ALL")
            .bind(("message_id", message_id_owned.clone()))
            .await?;

        #[derive(serde::Deserialize)]
        struct CountResult {
            count: i64,
        }

        let count: Option<CountResult> = result.take(0)?;
        let deleted_count = count.map(|c| c.count as u32).unwrap_or(0);

        // Delete tool calls
        self.db()
            .query("DELETE FROM tool_calls WHERE message_id = $message_id")
            .bind(("message_id", message_id_owned))
            .await?;

        Ok(deleted_count)
    }
}
