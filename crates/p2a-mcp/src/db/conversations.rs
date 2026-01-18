//! Conversation CRUD operations

use chrono::Utc;
use surrealdb::sql::Datetime;
use surrealdb::RecordId;

use super::connection::{DbConnection, DbError};
use super::models::{Conversation, ConversationWithMessages, Message, MessageRole};

// Note: Datetime is used in add_message for creating messages
// RecordId is used throughout for proper SurrealDB record identification

impl DbConnection {
    // ==================== Conversation Operations ====================

    /// Create a new conversation
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, DbError> {
        let conv = Conversation::new(session_id.to_string(), title.to_string());

        let created: Option<Conversation> = self
            .db()
            .create(conv.id.clone())
            .content(conv)
            .await?;

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

    /// Update a conversation's title
    pub async fn update_conversation_title(
        &self,
        id: &str,
        title: &str,
    ) -> Result<Conversation, DbError> {
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

    /// Delete a conversation and all its messages
    pub async fn delete_conversation(&self, id: &str) -> Result<(), DbError> {
        let id_owned = id.to_string();

        // Delete tool calls first
        self.db()
            .query("DELETE FROM tool_calls WHERE conversation_id = $id")
            .bind(("id", id_owned.clone()))
            .await?;

        // Delete messages
        self.db()
            .query("DELETE FROM messages WHERE conversation_id = $id")
            .bind(("id", id_owned.clone()))
            .await?;

        // Delete conversation
        let deleted: Option<Conversation> = self.db().delete(("conversations", id)).await?;

        if deleted.is_some() {
            Ok(())
        } else {
            Err(DbError::NotFound(format!("Conversation not found: {}", id)))
        }
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

    /// Add a message to a conversation
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

    /// Add a full message object
    pub async fn save_message(&self, message: &Message) -> Result<Message, DbError> {
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
            .query(
                "SELECT * FROM messages WHERE conversation_id = $id ORDER BY created_at ASC",
            )
            .bind(("id", conv_id_owned))
            .await?;

        let messages: Vec<Message> = result.take(0)?;
        Ok(messages)
    }

    /// Update the content of an existing message
    pub async fn update_message_content(
        &self,
        message_id: &str,
        content: &str,
    ) -> Result<Message, DbError> {
        let msg_id_owned = message_id.to_string();
        let mut result = self
            .db()
            .query("UPDATE messages SET content = $content WHERE id = $id RETURN AFTER")
            .bind(("id", RecordId::from(("messages", msg_id_owned.as_str()))))
            .bind(("content", content.to_string()))
            .await?;

        let updated: Option<Message> = result.take(0)?;

        if let Some(msg) = &updated {
            // Update conversation metadata with new preview
            self.update_conversation_after_message(&msg.conversation_id, content)
                .await?;
        }

        updated.ok_or_else(|| DbError::NotFound(format!("Message not found: {}", message_id)))
    }

    /// Get a message by ID
    pub async fn get_message(&self, message_id: &str) -> Result<Message, DbError> {
        let msg: Option<Message> = self.db().select(("messages", message_id)).await?;
        msg.ok_or_else(|| DbError::NotFound(format!("Message not found: {}", message_id)))
    }

    /// Delete all messages in a conversation
    pub async fn clear_messages(&self, conversation_id: &str) -> Result<u32, DbError> {
        let conv_id_owned = conversation_id.to_string();

        // Delete tool calls first
        self.db()
            .query("DELETE FROM tool_calls WHERE conversation_id = $id")
            .bind(("id", conv_id_owned.clone()))
            .await?;

        // Count messages before deletion
        let mut result = self
            .db()
            .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
            .bind(("id", conv_id_owned.clone()))
            .await?;

        #[derive(serde::Deserialize)]
        struct CountResult {
            count: i64,
        }

        let count: Option<CountResult> = result.take(0)?;
        let deleted_count = count.map(|c| c.count as u32).unwrap_or(0);

        // Delete messages
        self.db()
            .query("DELETE FROM messages WHERE conversation_id = $id")
            .bind(("id", conv_id_owned))
            .await?;

        // Reset conversation counters
        let conv_id = conversation_id.to_string();
        self.db()
            .query("UPDATE conversations SET message_count = 0, last_message_preview = NONE, updated_at = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("conversations", conv_id.as_str()))))
            .await?;

        Ok(deleted_count)
    }
}
