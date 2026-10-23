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
    /// Runs in the conversation's turn, like every write of its row.
    pub async fn update_conversation_title(
        &self,
        id: &str,
        title: &str,
    ) -> Result<Conversation, DbError> {
        self.check_message_bytes("title", "title", title.len())?;
        // In the turn, the write cannot conflict with a message add's
        // counter update, and a delete cannot run between the bytes and the
        // write.
        let _turn = self.conversation_turn(id).await;
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

    /// Archive or unarchive a conversation, in its turn (like every write of
    /// its row, so that it cannot conflict with a message add's counter
    /// update).
    pub async fn set_conversation_archived(
        &self,
        id: &str,
        is_archived: bool,
    ) -> Result<Conversation, DbError> {
        let _turn = self.conversation_turn(id).await;
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

    /// Delete a conversation and all its messages and tool calls, in one
    /// transaction (see [`delete_conversation_rows`](Self::delete_conversation_rows)):
    /// when the store fails it, or the process stops during it, nothing is
    /// deleted. With a `session_db_bytes` budget, the bytes this call
    /// removed are released from the session's counter. Deletes and clears
    /// of one conversation take turns.
    pub async fn delete_conversation(&self, id: &str) -> Result<(), DbError> {
        let _turn = self.conversation_turn(id).await;
        // A write racing this delete can leave rows behind; the next orphan
        // sweeps look for them.
        self.note_conversation_deleted(id);
        let owner = self.counted_owner(id).await?;
        let removed = self
            .delete_conversation_rows(id, Finish::DeleteConversation, owner.is_some())
            .await?;
        if let Some(session_id) = &owner {
            self.release_session_bytes(session_id, removed.bytes);
        }
        if removed.conversation {
            Ok(())
        } else {
            Err(DbError::NotFound(format!("Conversation not found: {}", id)))
        }
    }

    /// Delete the tool calls and then the messages of conversation `id` and
    /// then, in the same transaction, `finish` it: reset its counters (a
    /// clear) or delete its row (a conversation delete). A call stopped part
    /// way, by the process dying or by its caller being dropped while the
    /// store runs the query, has therefore done all of it or none of it, and
    /// the conversation's `message_count` never outlives its messages. An
    /// error means that nothing was deleted, unless it came from reading back
    /// the rows of a transaction that committed.
    ///
    /// With `measure`, the deletes return the rows they removed (`RETURN
    /// BEFORE`) and their messages and text bytes are counted (the title's
    /// too when the conversation row is deleted); without it only whether the
    /// conversation row was deleted is returned. Callers hold the
    /// conversation's [`StoreTurn`](super::connection::StoreTurn), so two of
    /// these never run on one conversation at the same time, and the later
    /// one finds nothing.
    async fn delete_conversation_rows(
        &self,
        id: &str,
        finish: Finish,
        measure: bool,
    ) -> Result<Removed, DbError> {
        let rows = if measure {
            "RETURN BEFORE"
        } else {
            "RETURN NONE"
        };
        let finish_statement = match finish {
            Finish::ResetCounters => {
                "UPDATE conversations SET message_count = 0, last_message_preview = NONE, \
                 updated_at = time::now() WHERE id = $conv RETURN NONE"
            }
            Finish::DeleteConversation => "DELETE $conv RETURN BEFORE",
        };
        let mut response = self
            .db()
            .query(format!(
                "BEGIN TRANSACTION;
                 DELETE FROM tool_calls WHERE conversation_id = $id {rows};
                 DELETE FROM messages WHERE conversation_id = $id {rows};
                 {finish_statement};
                 COMMIT TRANSACTION;"
            ))
            .bind(("id", id.to_string()))
            .bind(("conv", RecordId::from(("conversations", id))))
            .await?
            .check()?;

        // Committed: BEGIN and COMMIT return nothing, so the three
        // statements' results are 0, 1 and 2.
        let mut removed = Removed::default();
        if measure {
            let tool_calls: Vec<ToolCallText> = response.take(0)?;
            let messages: Vec<MessageText> = response.take(1)?;
            removed.messages = messages.len();
            removed.bytes =
                tool_call_text_bytes(&tool_calls).saturating_add(message_text_bytes(&messages));
        }
        if let Finish::DeleteConversation = finish {
            let deleted: Option<Conversation> = response.take(2)?;
            if let Some(conv) = deleted {
                removed.conversation = true;
                if measure {
                    removed.bytes = removed.bytes.saturating_add(conv.title.len() as u64);
                }
            }
        }
        Ok(removed)
    }

    /// The session whose `session_db_bytes` counter a delete in conversation
    /// `id` lowers. `None` without the budget, when the conversation does not
    /// exist, or when the session's counter is not seeded when the delete
    /// starts (seeding then reads what is left, so nothing is released).
    /// Call it during the delete's [`StoreTurn`](super::connection::StoreTurn).
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
    /// The `messages_per_conversation` check counts the conversation's
    /// message rows ([`count_messages`](Self::count_messages)); the stored
    /// `message_count` is only what the conversation list shows. Call it
    /// during the conversation's turn
    /// ([`conversation_turn`](DbConnection::conversation_turn)) and keep the
    /// turn until the message is written, so that concurrent adds cannot
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
            let n = self.count_messages(conversation_id).await?;
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

    /// The number of message rows conversation `conversation_id` holds, as
    /// the store counts them now; a failed count is an error.
    pub(crate) async fn count_messages(&self, conversation_id: &str) -> Result<usize, DbError> {
        #[derive(serde::Deserialize)]
        struct CountResult {
            count: i64,
        }
        let mut result = self
            .db()
            .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
            .bind(("id", conversation_id.to_string()))
            .await?
            .check()?;
        let count: Option<CountResult> = result.take(0)?;
        Ok(count.map_or(0, |c| usize::try_from(c.count).unwrap_or(0)))
    }

    /// Update conversation metadata after a message was added or changed:
    /// the stored `message_count` (a recount) and the preview. Call it
    /// during the conversation's turn
    /// ([`conversation_turn`](DbConnection::conversation_turn)), so that the
    /// recount and its write are not interleaved with another call's; a
    /// failed recount or write is an error.
    pub(crate) async fn update_conversation_after_message(
        &self,
        conversation_id: &str,
        message_preview: &str,
    ) -> Result<(), DbError> {
        let message_count =
            i32::try_from(self.count_messages(conversation_id).await?).unwrap_or(i32::MAX);

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
            .await?
            .check()?;

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
    ///
    /// An error from the conversation's counter and preview update comes
    /// after the message row was written: the message is stored, and only
    /// the conversation's `message_count` and preview are stale.
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
        let _turn = self.conversation_turn(conversation_id).await;
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
    ///
    /// An error from the conversation's counter and preview update comes
    /// after the message row was written: the message is stored, and only
    /// the conversation's `message_count` and preview are stale.
    pub async fn save_message(&self, message: &Message) -> Result<Message, DbError> {
        let _turn = self.conversation_turn(&message.conversation_id).await;
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
    ///
    /// The bytes, the update and the conversation's recount run in the
    /// conversation's turn, so a delete or clear of the conversation runs
    /// before them (the message is `NotFound` and nothing is counted) or
    /// after them (it releases the new content's bytes, and a clear resets
    /// the preview).
    /// An error from the conversation's counter and preview update comes
    /// after the message row was written: the new content is stored, and
    /// only the conversation's `message_count` and preview are stale.
    pub async fn update_message_content(
        &self,
        message_id: &str,
        content: &str,
    ) -> Result<Message, DbError> {
        let mut content = self.truncate_server_text(content);
        let conversation_id = self.get_message(message_id).await?.conversation_id;
        let _turn = self.conversation_turn(&conversation_id).await;
        if self.tracks_session_bytes() {
            // Read again in the turn: a clear that ran before it took the
            // message, and its bytes must not be counted for a write that
            // finds no row.
            self.get_message(message_id).await?;
            let conv = self.get_conversation(&conversation_id).await?;
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
        let Some(msg) = updated else {
            return Err(DbError::NotFound(format!(
                "Message not found: {}",
                message_id
            )));
        };
        self.update_conversation_after_message(&msg.conversation_id, &content)
            .await?;
        Ok(msg)
    }

    /// Get a message by ID
    pub async fn get_message(&self, message_id: &str) -> Result<Message, DbError> {
        let msg: Option<Message> = self.db().select(("messages", message_id)).await?;
        msg.ok_or_else(|| DbError::NotFound(format!("Message not found: {}", message_id)))
    }

    /// Delete all messages (and tool calls) in a conversation and reset its
    /// `message_count` and preview, in one transaction (see
    /// [`delete_conversation_rows`](Self::delete_conversation_rows)): when
    /// the store fails it, or the process stops during it, nothing is
    /// deleted. Returns how many messages went. With a
    /// `session_db_bytes` budget, the bytes this call removed are released
    /// from the session's counter. Deletes and clears of one conversation
    /// take turns.
    pub async fn clear_messages(&self, conversation_id: &str) -> Result<u32, DbError> {
        let _turn = self.conversation_turn(conversation_id).await;
        let owner = self.counted_owner(conversation_id).await?;
        // Unmeasured, the delete returns no rows: count them first (in the
        // turn, so no add of this process lands in between).
        let counted = match owner {
            Some(_) => None,
            None => Some(self.count_messages(conversation_id).await?),
        };
        let removed = self
            .delete_conversation_rows(conversation_id, Finish::ResetCounters, owner.is_some())
            .await?;
        if let Some(session_id) = &owner {
            self.release_session_bytes(session_id, removed.bytes);
        }
        Ok(u32::try_from(counted.unwrap_or(removed.messages)).unwrap_or(u32::MAX))
    }
}

/// What [`delete_conversation_rows`](DbConnection::delete_conversation_rows)
/// does after deleting a conversation's tool calls and messages, in the same
/// transaction.
#[derive(Debug, Clone, Copy)]
enum Finish {
    /// Reset the conversation's `message_count` and preview (a clear).
    ResetCounters,
    /// Delete the conversation row (a conversation delete).
    DeleteConversation,
}

/// What deleting a conversation's rows removed: the messages and the text
/// bytes (contents, tool-call texts and, for a whole conversation, its
/// title) that the `session_db_bytes` budget counts, and whether the
/// conversation row was deleted.
#[derive(Debug, Default)]
struct Removed {
    messages: usize,
    bytes: u64,
    conversation: bool,
}
