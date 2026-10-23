//! Conversation CRUD operations

use chrono::Utc;
use surrealdb::RecordId;
use surrealdb::sql::Datetime;

use super::connection::{DbConnection, DbError};
use super::models::{Conversation, ConversationWithMessages, Message, MessageRole};
use super::sessions::{MessageText, ToolCallText, message_text_bytes, tool_call_text_bytes};

// Note: Datetime is used in add_message for creating messages
// RecordId is used throughout for proper SurrealDB record identification

impl DbConnection {
    // ==================== Conversation Operations ====================

    /// Create a new conversation.
    ///
    /// Refused with `MemoryPressure` above the RSS soft limit, `TooLarge`
    /// for a title longer than `message_bytes`, and `QuotaExceeded` when the
    /// session already holds `conversations_per_session` conversations or
    /// the title would pass its `session_db_bytes` budget.
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, DbError> {
        self.check_memory()?;
        self.check_message_bytes("title", "title", title.len())?;
        // Creations in one session take turns from the count to the write,
        // so concurrent ones cannot pass the cap together.
        let _turn = self.session_turn(session_id).await;
        if let Some(max) = self.limits().conversations_per_session {
            let n = self.count_conversations(session_id).await?;
            if n >= max {
                return Err(DbError::QuotaExceeded(format!(
                    "this session already has {n} conversations, the most this server keeps; \
                     delete one first"
                )));
            }
        }
        self.reserve_session_bytes(session_id, title.len() as u64)
            .await?;

        let conv = Conversation::new(session_id.to_string(), title.to_string());

        let created: Option<Conversation> = self.db().create(conv.id.clone()).content(conv).await?;

        created.ok_or_else(|| DbError::Query("Failed to create conversation".to_string()))
    }

    /// Get a conversation by ID
    pub async fn get_conversation(&self, id: &str) -> Result<Conversation, DbError> {
        let conv: Option<Conversation> = self.db().select(("conversations", id)).await?;
        conv.ok_or_else(|| DbError::NotFound(format!("Conversation not found: {}", id)))
    }

    /// Get a conversation with all its messages
    pub async fn get_conversation_with_messages(
        &self,
        id: &str,
    ) -> Result<ConversationWithMessages, DbError> {
        let conversation = self.get_conversation(id).await?;
        let messages = self.get_messages(id).await?;

        Ok(ConversationWithMessages {
            conversation,
            messages,
        })
    }

    /// List all conversations for a session
    pub async fn list_conversations(&self, session_id: &str) -> Result<Vec<Conversation>, DbError> {
        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM conversations WHERE session_id = $session_id ORDER BY updated_at DESC")
            .bind(("session_id", session_id_owned))
            .await?;

        let conversations: Vec<Conversation> = result.take(0)?;
        Ok(conversations)
    }

    /// Update a conversation's title. A title longer than `message_bytes` is
    /// `TooLarge`; with a `session_db_bytes` budget, the bytes the new title
    /// adds count against it (`QuotaExceeded` when they would pass it).
    pub async fn update_conversation_title(
        &self,
        id: &str,
        title: &str,
    ) -> Result<Conversation, DbError> {
        self.check_message_bytes("title", "title", title.len())?;
        if self.tracks_session_bytes() {
            let conv = self.get_conversation(id).await?;
            let added = title.len().saturating_sub(conv.title.len());
            self.reserve_session_bytes(&conv.session_id, added as u64)
                .await?;
        }
        let id_owned = id.to_string();
        let mut result = self
            .db()
            .query("UPDATE conversations SET title = $title, updated_at = time::now() WHERE id = $id RETURN AFTER")
            .bind(("id", RecordId::from(("conversations", id_owned.as_str()))))
            .bind(("title", title.to_string()))
            .await?;

        let updated: Option<Conversation> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Conversation not found: {}", id)))
    }

    /// Archive or unarchive a conversation
    pub async fn set_conversation_archived(
        &self,
        id: &str,
        is_archived: bool,
    ) -> Result<Conversation, DbError> {
        let id_owned = id.to_string();
        let mut result = self
            .db()
            .query("UPDATE conversations SET is_archived = $archived, updated_at = time::now() WHERE id = $id RETURN AFTER")
            .bind(("id", RecordId::from(("conversations", id_owned.as_str()))))
            .bind(("archived", is_archived))
            .await?;

        let updated: Option<Conversation> = result.take(0)?;
        updated.ok_or_else(|| DbError::NotFound(format!("Conversation not found: {}", id)))
    }

    /// Delete a conversation and all its messages. With a `session_db_bytes`
    /// budget, the bytes this call removed are released from the session's
    /// counter (also when a later step fails). Deletes and clears of one
    /// conversation take turns.
    pub async fn delete_conversation(&self, id: &str) -> Result<(), DbError> {
        let _turn = self.delete_turn(id).await;
        // A write racing this delete, or a step below that fails, can leave
        // rows behind; the next orphan sweeps look for them.
        self.note_conversation_deleted(id);
        let owner = self.counted_owner(id).await?;
        let mut removed = Removed::default();
        let outcome = self
            .delete_conversation_and_rows(id, owner.is_some(), &mut removed)
            .await;
        if let Some(session_id) = &owner {
            self.release_session_bytes(session_id, removed.bytes);
        }
        if outcome? {
            Ok(())
        } else {
            Err(DbError::NotFound(format!("Conversation not found: {}", id)))
        }
    }

    /// Delete conversation `id`'s tool calls, messages and row, in that
    /// order, adding what each step removed to `removed` (the title counts
    /// only when the row was there to delete). Whether the row existed.
    async fn delete_conversation_and_rows(
        &self,
        id: &str,
        measure: bool,
        removed: &mut Removed,
    ) -> Result<bool, DbError> {
        self.delete_conversation_rows(id, measure, removed).await?;
        let deleted: Option<Conversation> = self.db().delete(("conversations", id)).await?;
        if let Some(conv) = &deleted {
            removed.bytes = removed.bytes.saturating_add(conv.title.len() as u64);
        }
        Ok(deleted.is_some())
    }

    /// Delete the tool calls and then the messages of conversation `id`,
    /// adding what each statement removed to `removed` once it committed.
    ///
    /// With `measure`, each delete returns the rows it removed (`RETURN
    /// BEFORE`) and their text bytes and messages are counted; a delete that
    /// runs after another one finds nothing. Callers hold the conversation's
    /// [`DeleteTurn`](super::connection::DeleteTurn), so two deletes never run
    /// at the same time. Without `measure` nothing is returned or counted.
    async fn delete_conversation_rows(
        &self,
        id: &str,
        measure: bool,
        removed: &mut Removed,
    ) -> Result<(), DbError> {
        let id = || ("id", id.to_string());
        if !measure {
            self.db()
                .query("DELETE FROM tool_calls WHERE conversation_id = $id")
                .bind(id())
                .await?
                .check()?;
            self.db()
                .query("DELETE FROM messages WHERE conversation_id = $id")
                .bind(id())
                .await?
                .check()?;
            return Ok(());
        }
        let tool_calls: Vec<ToolCallText> = self
            .db()
            .query("DELETE FROM tool_calls WHERE conversation_id = $id RETURN BEFORE")
            .bind(id())
            .await?
            .take(0)?;
        removed.bytes = removed
            .bytes
            .saturating_add(tool_call_text_bytes(&tool_calls));
        let messages: Vec<MessageText> = self
            .db()
            .query("DELETE FROM messages WHERE conversation_id = $id RETURN BEFORE")
            .bind(id())
            .await?
            .take(0)?;
        removed.messages = messages.len();
        removed.bytes = removed.bytes.saturating_add(message_text_bytes(&messages));
        Ok(())
    }

    /// The session whose `session_db_bytes` counter a delete in conversation
    /// `id` lowers. `None` without the budget, when the conversation does not
    /// exist, or when the session's counter is not seeded when the delete
    /// starts (seeding then reads what is left, so nothing is released).
    /// Call it during the delete's [`DeleteTurn`](super::connection::DeleteTurn).
    async fn counted_owner(&self, id: &str) -> Result<Option<String>, DbError> {
        if !self.tracks_session_bytes() {
            return Ok(None);
        }
        let conv = match self.get_conversation(id).await {
            Ok(conv) => conv,
            Err(DbError::NotFound(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(self
            .has_session_bytes(&conv.session_id)
            .then_some(conv.session_id))
    }

    /// The conversation a client-posted message of `content_len` bytes goes
    /// to, after the store's checks: memory, the conversation must exist
    /// (`NotFound`), `message_bytes` (`TooLarge`), `messages_per_conversation`
    /// and `session_db_bytes` (`QuotaExceeded`; the bytes are counted).
    ///
    /// Call it during the conversation's turn
    /// ([`delete_turn`](DbConnection::delete_turn)) and keep the turn until
    /// the message is written and counted, so that concurrent adds cannot
    /// pass `messages_per_conversation` together.
    async fn admit_message(
        &self,
        conversation_id: &str,
        content_len: usize,
    ) -> Result<Conversation, DbError> {
        self.check_memory()?;
        let conv = self.get_conversation(conversation_id).await?;
        self.check_message_bytes("message", "message", content_len)?;
        if let Some(max) = self.limits().messages_per_conversation {
            let n = usize::try_from(conv.message_count).unwrap_or(0);
            if n >= max {
                return Err(DbError::QuotaExceeded(format!(
                    "this conversation already has {n} messages, the most this server keeps; \
                     start a new conversation"
                )));
            }
        }
        self.reserve_session_bytes(&conv.session_id, content_len as u64)
            .await?;
        Ok(conv)
    }

    /// Update conversation metadata after adding a message
    pub(crate) async fn update_conversation_after_message(
        &self,
        conversation_id: &str,
        message_preview: &str,
    ) -> Result<(), DbError> {
        let conv_id_owned = conversation_id.to_string();

        // Get current message count
        let mut result = self
            .db()
            .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
            .bind(("id", conv_id_owned))
            .await?;

        #[derive(serde::Deserialize)]
        struct CountResult {
            count: i64,
        }

        let count: Option<CountResult> = result.take(0)?;
        let message_count = count.map(|c| c.count as i32).unwrap_or(0);

        // Truncate preview if needed
        let preview = if message_preview.len() > 100 {
            format!("{}...", &message_preview[..97])
        } else {
            message_preview.to_string()
        };

        // Update conversation
        let conv_id = conversation_id.to_string();
        self.db()
            .query("UPDATE conversations SET message_count = $count, last_message_preview = $preview, updated_at = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("conversations", conv_id.as_str()))))
            .bind(("count", message_count))
            .bind(("preview", preview))
            .await?;

        Ok(())
    }

    // ==================== Message Operations ====================

    /// Add a message to a conversation.
    ///
    /// The conversation must exist (`NotFound`). Refused with
    /// `MemoryPressure` above the RSS soft limit, `TooLarge` for content
    /// longer than `message_bytes`, and `QuotaExceeded` when the conversation
    /// is full or the content would pass the session's `session_db_bytes`.
    /// Adds, deletes and clears of one conversation take turns.
    pub async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<Message, DbError> {
        let role = match role {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            _ => {
                return Err(DbError::Query(format!("Invalid message role: {}", role)));
            }
        };
        let _turn = self.delete_turn(conversation_id).await;
        self.admit_message(conversation_id, content.len()).await?;

        let msg_id = uuid::Uuid::new_v4().to_string();
        let message = Message {
            id: RecordId::from(("messages", msg_id.as_str())),
            conversation_id: conversation_id.to_string(),
            role,
            content: content.to_string(),
            created_at: Datetime::from(Utc::now()),
            token_count: None,
            model: None,
            finish_reason: None,
        };

        let created: Option<Message> = self
            .db()
            .create(message.id.clone())
            .content(message)
            .await?;

        // Update conversation metadata
        self.update_conversation_after_message(conversation_id, content)
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to create message".to_string()))
    }

    /// Add a full message object, with the checks of
    /// [`add_message`](Self::add_message).
    pub async fn save_message(&self, message: &Message) -> Result<Message, DbError> {
        let _turn = self.delete_turn(&message.conversation_id).await;
        self.admit_message(&message.conversation_id, message.content.len())
            .await?;
        let created: Option<Message> = self
            .db()
            .create(message.id.clone())
            .content(message.clone())
            .await?;

        // Update conversation metadata
        self.update_conversation_after_message(&message.conversation_id, &message.content)
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to save message".to_string()))
    }

    /// Get all messages for a conversation
    pub async fn get_messages(&self, conversation_id: &str) -> Result<Vec<Message>, DbError> {
        let conv_id_owned = conversation_id.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM messages WHERE conversation_id = $id ORDER BY created_at ASC")
            .bind(("id", conv_id_owned))
            .await?;

        let messages: Vec<Message> = result.take(0)?;
        Ok(messages)
    }

    /// Update the content of an existing message (the server's final
    /// answer). The content is never refused for size: it is cut at a char
    /// boundary to `message_bytes` and, with a `session_db_bytes` budget, to
    /// the bytes the session has left (possibly to the empty string).
    pub async fn update_message_content(
        &self,
        message_id: &str,
        content: &str,
    ) -> Result<Message, DbError> {
        let mut content = self.truncate_server_text(content);
        if self.tracks_session_bytes() {
            let message = self.get_message(message_id).await?;
            let conv = self.get_conversation(&message.conversation_id).await?;
            content = self.fit_session_bytes(&conv.session_id, content).await?;
        }
        let msg_id_owned = message_id.to_string();
        let mut result = self
            .db()
            .query("UPDATE messages SET content = $content WHERE id = $id RETURN AFTER")
            .bind(("id", RecordId::from(("messages", msg_id_owned.as_str()))))
            .bind(("content", content.clone()))
            .await?;

        let updated: Option<Message> = result.take(0)?;

        if let Some(msg) = &updated {
            // Update conversation metadata with new preview
            self.update_conversation_after_message(&msg.conversation_id, &content)
                .await?;
        }

        updated.ok_or_else(|| DbError::NotFound(format!("Message not found: {}", message_id)))
    }

    /// Get a message by ID
    pub async fn get_message(&self, message_id: &str) -> Result<Message, DbError> {
        let msg: Option<Message> = self.db().select(("messages", message_id)).await?;
        msg.ok_or_else(|| DbError::NotFound(format!("Message not found: {}", message_id)))
    }

    /// Delete all messages in a conversation; returns how many. With a
    /// `session_db_bytes` budget, the bytes this call removed are released
    /// from the session's counter (also when a later step fails). Deletes
    /// and clears of one conversation take turns.
    pub async fn clear_messages(&self, conversation_id: &str) -> Result<u32, DbError> {
        let _turn = self.delete_turn(conversation_id).await;
        let deleted_count = match self.counted_owner(conversation_id).await? {
            Some(session_id) => {
                let mut removed = Removed::default();
                let outcome = self
                    .delete_conversation_rows(conversation_id, true, &mut removed)
                    .await;
                self.release_session_bytes(&session_id, removed.bytes);
                outcome?;
                u32::try_from(removed.messages).unwrap_or(u32::MAX)
            }
            None => {
                // Count messages before deletion
                #[derive(serde::Deserialize)]
                struct CountResult {
                    count: i64,
                }
                let mut result = self
                    .db()
                    .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
                    .bind(("id", conversation_id.to_string()))
                    .await?;
                let count: Option<CountResult> = result.take(0)?;
                self.delete_conversation_rows(conversation_id, false, &mut Removed::default())
                    .await?;
                count.map(|c| c.count as u32).unwrap_or(0)
            }
        };

        // Reset conversation counters
        let conv_id = conversation_id.to_string();
        self.db()
            .query("UPDATE conversations SET message_count = 0, last_message_preview = NONE, updated_at = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("conversations", conv_id.as_str()))))
            .await?;

        Ok(deleted_count)
    }
}

/// What deleting a conversation's rows removed: the messages and the text
/// bytes (contents, tool-call texts and, for a whole conversation, its
/// title) that the `session_db_bytes` budget counts.
#[derive(Debug, Default)]
struct Removed {
    messages: usize,
    bytes: u64,
}
