//! Tool call CRUD operations

use surrealdb::RecordId;

use super::connection::{DbConnection, DbError};
use super::models::{ToolCall, ToolCallStatus};

impl DbConnection {
    // ==================== Tool Call Operations ====================

    /// Create a new tool call record
    pub async fn create_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        let created: Option<ToolCall> = self
            .db()
            .create(tool_call.id.clone())
            .content(tool_call.clone())
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to create tool call".to_string()))
    }

    /// Update an existing tool call (typically to set result/error/status)
    pub async fn update_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        let id_str = tool_call.id_string();

        let result: Option<ToolCall> = self
            .db()
            .update(("tool_calls", id_str.as_str()))
            .content(tool_call.clone())
            .await?;

        result.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id_str)))
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

    /// Mark a tool call as completed successfully
    pub async fn complete_tool_call(
        &self,
        id: &str,
        result_str: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
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
            .bind(("result", result_str.to_string()))
            .bind(("duration_ms", duration_ms))
            .await?;

        let updated: Option<ToolCall> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))
    }

    /// Mark a tool call as failed
    pub async fn fail_tool_call(
        &self,
        id: &str,
        error: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
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
            .bind(("error", error.to_string()))
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
