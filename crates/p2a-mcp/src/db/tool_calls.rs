//! Tool call CRUD operations
//!
//! A tool call's create, update and status changes run in its conversation's
//! turn, like the writes of its messages and of the conversation row (see
//! [`create_tool_call`](DbConnection::create_tool_call)). Tool calls are
//! deleted with their conversation's messages, by a clear or delete of the
//! conversation in its turn, or, when left behind for a deleted
//! conversation, by the orphan sweeps.

use surrealdb::RecordId;

use super::connection::{DbConnection, DbError, run_to_end};
use super::models::{ToolCall, ToolCallStatus};
use super::sessions::tool_call_texts_len;

impl DbConnection {
    // ==================== Tool Call Operations ====================

    /// Create a new tool call record. Its conversation must exist, and its
    /// message must exist in that conversation (`NotFound` otherwise), and it
    /// is refused with `MemoryPressure` above the RSS soft limit. Its texts
    /// are cut at a char boundary to `message_bytes` and to the bytes the
    /// session has left under `session_db_bytes`, never refused.
    ///
    /// The existence checks, the bytes and the write run in the
    /// conversation's [`StoreTurn`](super::connection::StoreTurn), in a task
    /// a dropped caller does not stop ([`run_to_end`]), like a tool call's
    /// update and status changes; a call dropped while it waits for the turn
    /// does nothing. A delete or clear of the conversation runs wholly
    /// before such a write or wholly after it, never alongside it: on
    /// RocksDB, which validates an optimistic transaction's writes when it
    /// commits, a tool-call write committed on a row a clear's transaction
    /// deletes would fail the clear whole on a write conflict. A write that
    /// comes after a delete finds no conversation, and one after a clear
    /// finds no message or, for an update or status change, no row: it is
    /// `NotFound`, and nothing is counted or stored.
    pub async fn create_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        self.check_memory()?;
        let db = self.clone();
        let tool_call = tool_call.clone();
        let turn = db.conversation_turn(&tool_call.conversation_id).await;
        run_to_end(turn, async move {
            let conv = db.get_conversation(&tool_call.conversation_id).await?;
            // A clear that ran first left the conversation but took the
            // message; a row stored for it would only be counted, never
            // shown, until the next clear or delete.
            let message = db.get_message(&tool_call.message_id).await?;
            if message.conversation_id != tool_call.conversation_id {
                return Err(DbError::NotFound(format!(
                    "Message not found in conversation {}: {}",
                    tool_call.conversation_id, tool_call.message_id
                )));
            }
            let tool_call = db
                .fit_tool_call_texts(&conv.session_id, tool_call, 0)
                .await?;
            let created: Option<ToolCall> = db
                .db()
                .create(tool_call.id.clone())
                .content(tool_call)
                .await?;

            created.ok_or_else(|| DbError::Query("Failed to create tool call".to_string()))
        })
        .await
    }

    /// Update an existing tool call (typically to set result/error/status).
    /// It must be stored for `tool_call.conversation_id` and
    /// `tool_call.message_id` (`NotFound` otherwise, as when it does not
    /// exist): an update never moves a call to another message, which could
    /// be missing or in another conversation, as a create would refuse. Its
    /// texts are cut to fit, like
    /// those of a new one, with the bytes of the stored texts they replace as
    /// room: with a `session_db_bytes` budget only what they add counts. Runs
    /// in the conversation's turn, like
    /// [`create_tool_call`](Self::create_tool_call).
    pub async fn update_tool_call(&self, tool_call: &ToolCall) -> Result<ToolCall, DbError> {
        let db = self.clone();
        let tool_call = tool_call.clone();
        let turn = db.conversation_turn(&tool_call.conversation_id).await;
        run_to_end(turn, async move {
            let id = tool_call.id_string();
            let stored = db.get_tool_call(&id).await?;
            if stored.conversation_id != tool_call.conversation_id {
                return Err(DbError::NotFound(format!(
                    "Tool call not found in conversation {}: {}",
                    tool_call.conversation_id, id
                )));
            }
            if stored.message_id != tool_call.message_id {
                return Err(DbError::NotFound(format!(
                    "Tool call {} not found for message {}: it is stored for message {}",
                    id, tool_call.message_id, stored.message_id
                )));
            }
            let session_id = db.tool_call_session(&stored.conversation_id).await?;
            let replaced = tool_call_texts_len(
                &stored.arguments,
                stored.result.as_deref(),
                stored.error.as_deref(),
            );
            let tool_call = db
                .fit_tool_call_texts(&session_id, tool_call, replaced)
                .await?;

            let result: Option<ToolCall> = db
                .db()
                .update(("tool_calls", id.as_str()))
                .content(tool_call)
                .await?;
            let updated =
                result.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))?;
            db.release_session_bytes(&session_id, replaced);
            Ok(updated)
        })
        .await
    }

    /// Every text field of `tool_call` cut at a char boundary to
    /// `message_bytes` and fitted, arguments first, into what session
    /// `session_id` has left under `session_db_bytes`, with the `replaced`
    /// bytes of the stored texts the write replaces as room
    /// ([`fit_session_texts`](Self::fit_session_texts)). Never refused.
    async fn fit_tool_call_texts(
        &self,
        session_id: &str,
        mut tool_call: ToolCall,
        replaced: u64,
    ) -> Result<ToolCall, DbError> {
        let mut texts: Vec<&mut String> = std::iter::once(&mut tool_call.arguments)
            .chain(tool_call.result.as_mut())
            .chain(tool_call.error.as_mut())
            .collect();
        for text in texts.iter_mut() {
            **text = self.truncate_server_text(text.as_str());
        }
        self.fit_session_texts(session_id, &mut texts, replaced)
            .await?;
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

    /// Change the status of tool call `id`, and for a finished call store its
    /// result or error, in its conversation's turn, like
    /// [`create_tool_call`](Self::create_tool_call). The conversation is the
    /// stored row's (`NotFound` when there is none). The text is cut at a
    /// char boundary to `message_bytes` and to what the session has left
    /// under `session_db_bytes`, with the bytes of the stored text it
    /// replaces as room, never refused.
    async fn change_tool_call_status(
        &self,
        id: &str,
        change: StatusChange,
    ) -> Result<ToolCall, DbError> {
        let conversation_id = self.get_tool_call(id).await?.conversation_id;
        let db = self.clone();
        let id = id.to_string();
        let turn = db.conversation_turn(&conversation_id).await;
        run_to_end(turn, async move {
            let (status, mut text, duration_ms) = match change {
                StatusChange::Running => (ToolCallStatus::Running, None, None),
                StatusChange::Finished {
                    status,
                    field,
                    text,
                    duration_ms,
                } => (
                    status,
                    Some((field, db.truncate_server_text(&text))),
                    Some(duration_ms),
                ),
            };
            let mut session_id = String::new();
            let mut replaced = 0;
            if let Some((field, text)) = &mut text
                && db.tracks_session_bytes()
            {
                // Read again in the turn: a delete or clear that ran before
                // it took the row, and no bytes may be counted for a write
                // that finds none.
                let stored = db.get_tool_call(&id).await?;
                session_id = db.tool_call_session(&stored.conversation_id).await?;
                replaced = field.stored(&stored).map_or(0, |t| t.len() as u64);
                db.fit_session_texts(&session_id, &mut [text], replaced)
                    .await?;
            }

            let query = match &text {
                None => "UPDATE tool_calls SET status = $status WHERE id = $id RETURN AFTER",
                Some((TextField::Result, _)) => {
                    r#"UPDATE tool_calls SET
                        status = $status,
                        result = $text,
                        completed_at = time::now(),
                        duration_ms = $duration_ms
                    WHERE id = $id RETURN AFTER"#
                }
                Some((TextField::Error, _)) => {
                    r#"UPDATE tool_calls SET
                        status = $status,
                        error = $text,
                        completed_at = time::now(),
                        duration_ms = $duration_ms
                    WHERE id = $id RETURN AFTER"#
                }
            };
            let mut result = db
                .db()
                .query(query)
                .bind(("id", RecordId::from(("tool_calls", id.as_str()))))
                .bind(("status", status))
                .bind(("text", text.map(|(_, text)| text)))
                .bind(("duration_ms", duration_ms))
                .await?;

            let updated: Option<ToolCall> = result.take(0)?;
            let updated =
                updated.ok_or_else(|| DbError::NotFound(format!("Tool call not found: {}", id)))?;
            db.release_session_bytes(&session_id, replaced);
            Ok(updated)
        })
        .await
    }

    /// Mark a tool call as running, in its conversation's turn, like
    /// [`create_tool_call`](Self::create_tool_call).
    pub async fn mark_tool_call_running(&self, id: &str) -> Result<ToolCall, DbError> {
        self.change_tool_call_status(id, StatusChange::Running)
            .await
    }

    /// Mark a tool call as completed successfully, in its conversation's
    /// turn. The result is cut to fit, with the bytes of a stored result as
    /// room, never refused.
    pub async fn complete_tool_call(
        &self,
        id: &str,
        result_str: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
        let change = StatusChange::Finished {
            status: ToolCallStatus::Success,
            field: TextField::Result,
            text: result_str.to_string(),
            duration_ms,
        };
        self.change_tool_call_status(id, change).await
    }

    /// Mark a tool call as failed, in its conversation's turn. The error
    /// text is cut to fit, with the bytes of a stored error as room, never
    /// refused.
    pub async fn fail_tool_call(
        &self,
        id: &str,
        error: &str,
        duration_ms: i32,
    ) -> Result<ToolCall, DbError> {
        let change = StatusChange::Finished {
            status: ToolCallStatus::Error,
            field: TextField::Error,
            text: error.to_string(),
            duration_ms,
        };
        self.change_tool_call_status(id, change).await
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
}

/// A change of an existing tool call's status
/// ([`change_tool_call_status`](DbConnection::change_tool_call_status)).
enum StatusChange {
    /// It is running.
    Running,
    /// It finished with `status`, storing `text` in `field`, after
    /// `duration_ms`.
    Finished {
        status: ToolCallStatus,
        field: TextField,
        text: String,
        duration_ms: i32,
    },
}

/// The text field a finished tool call stores.
#[derive(Clone, Copy)]
enum TextField {
    /// `result`, of a call that succeeded.
    Result,
    /// `error`, of a call that failed.
    Error,
}

impl TextField {
    /// This field's text in the stored row `tool_call`.
    fn stored(self, tool_call: &ToolCall) -> Option<&str> {
        match self {
            TextField::Result => tool_call.result.as_deref(),
            TextField::Error => tool_call.error.as_deref(),
        }
    }
}
