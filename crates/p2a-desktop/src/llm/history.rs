//! Conversation history persistence using SQLite.
//!
//! This module provides storage and retrieval of LLM conversations,
//! including messages, tool calls, and user settings.

use super::{LlmError, Message, MessageRole, ProviderConfig, ProviderType, ToolCall, ToolResult};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

/// A stored conversation with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub provider: ProviderType,
    pub model: String,
}

/// A stored message within a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: String,
    pub conversation_id: String,
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_results: Option<Vec<ToolResult>>,
    pub timestamp: DateTime<Utc>,
}

impl StoredMessage {
    /// Convert to the provider Message type.
    pub fn to_message(&self) -> Message {
        Message {
            role: self.role,
            content: self.content.clone(),
            tool_calls: self.tool_calls.clone(),
            tool_results: self.tool_results.clone(),
        }
    }
}

/// SQLite-backed history store.
pub struct HistoryStore {
    conn: Mutex<Connection>,
}

impl HistoryStore {
    /// Create a new history store, initializing the database at the given path.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self, LlmError> {
        let conn = Connection::open(db_path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory history store (for testing).
    pub fn in_memory() -> Result<Self, LlmError> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<(), LlmError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                tool_results TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id);
            "#,
        )?;
        Ok(())
    }

    // ========== Conversation Methods ==========

    /// Create a new conversation.
    pub fn create_conversation(
        &self,
        title: &str,
        provider: ProviderType,
        model: &str,
    ) -> Result<Conversation, LlmError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, provider, model) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &id,
                title,
                now.to_rfc3339(),
                now.to_rfc3339(),
                provider.to_string(),
                model
            ],
        )?;

        Ok(Conversation {
            id,
            title: title.to_string(),
            created_at: now,
            updated_at: now,
            provider,
            model: model.to_string(),
        })
    }

    /// List all conversations, ordered by most recently updated.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>, LlmError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, title, created_at, updated_at, provider, model 
             FROM conversations 
             ORDER BY updated_at DESC",
        )?;

        let conversations = stmt
            .query_map([], |row| {
                let provider_str: String = row.get(4)?;
                let provider = match provider_str.as_str() {
                    "ollama" => ProviderType::Ollama,
                    "anthropic" => ProviderType::Anthropic,
                    "openai" => ProviderType::OpenAI,
                    _ => ProviderType::Ollama,
                };

                Ok(Conversation {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    provider,
                    model: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(conversations)
    }

    /// Get a specific conversation by ID.
    pub fn get_conversation(&self, id: &str) -> Result<Option<Conversation>, LlmError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, title, created_at, updated_at, provider, model 
             FROM conversations WHERE id = ?1",
        )?;

        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let provider_str: String = row.get(4)?;
            let provider = match provider_str.as_str() {
                "ollama" => ProviderType::Ollama,
                "anthropic" => ProviderType::Anthropic,
                "openai" => ProviderType::OpenAI,
                _ => ProviderType::Ollama,
            };

            Ok(Some(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                    .unwrap()
                    .with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                    .unwrap()
                    .with_timezone(&Utc),
                provider,
                model: row.get(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Update conversation title.
    pub fn update_conversation_title(&self, id: &str, title: &str) -> Result<(), LlmError> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now();
        conn.execute(
            "UPDATE conversations SET title = ?1, updated_at = ?2 WHERE id = ?3",
            params![title, now.to_rfc3339(), id],
        )?;
        Ok(())
    }

    /// Touch conversation (update updated_at timestamp).
    pub fn touch_conversation(&self, id: &str) -> Result<(), LlmError> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now();
        conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now.to_rfc3339(), id],
        )?;
        Ok(())
    }

    /// Delete a conversation and all its messages.
    pub fn delete_conversation(&self, id: &str) -> Result<(), LlmError> {
        let conn = self.conn.lock().unwrap();
        // Delete messages first (foreign key cascade should handle this, but be explicit)
        conn.execute("DELETE FROM messages WHERE conversation_id = ?1", params![id])?;
        conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ========== Message Methods ==========

    /// Add a message to a conversation.
    pub fn add_message(
        &self,
        conversation_id: &str,
        message: &Message,
    ) -> Result<StoredMessage, LlmError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let tool_calls_json = message
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap());
        let tool_results_json = message
            .tool_results
            .as_ref()
            .map(|tr| serde_json::to_string(tr).unwrap());

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_results, timestamp) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &id,
                conversation_id,
                message.role.to_string(),
                &message.content,
                tool_calls_json,
                tool_results_json,
                now.to_rfc3339()
            ],
        )?;

        // Touch the conversation
        drop(conn);
        self.touch_conversation(conversation_id)?;

        Ok(StoredMessage {
            id,
            conversation_id: conversation_id.to_string(),
            role: message.role,
            content: message.content.clone(),
            tool_calls: message.tool_calls.clone(),
            tool_results: message.tool_results.clone(),
            timestamp: now,
        })
    }

    /// Get all messages for a conversation, ordered by timestamp.
    pub fn get_messages(&self, conversation_id: &str) -> Result<Vec<StoredMessage>, LlmError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, role, content, tool_calls, tool_results, timestamp 
             FROM messages 
             WHERE conversation_id = ?1 
             ORDER BY timestamp ASC",
        )?;

        let messages = stmt
            .query_map(params![conversation_id], |row| {
                let role_str: String = row.get(2)?;
                let role = match role_str.as_str() {
                    "system" => MessageRole::System,
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    "tool" => MessageRole::Tool,
                    _ => MessageRole::User,
                };

                let tool_calls: Option<Vec<ToolCall>> = row
                    .get::<_, Option<String>>(4)?
                    .map(|s| serde_json::from_str(&s).unwrap_or_default());
                let tool_results: Option<Vec<ToolResult>> = row
                    .get::<_, Option<String>>(5)?
                    .map(|s| serde_json::from_str(&s).unwrap_or_default());

                Ok(StoredMessage {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role,
                    content: row.get(3)?,
                    tool_calls,
                    tool_results,
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(messages)
    }

    // ========== Settings Methods ==========

    /// Get a setting value.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>, LlmError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;

        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// Set a setting value.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<(), LlmError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Get the LLM provider configuration from settings.
    pub fn get_provider_config(&self) -> Result<ProviderConfig, LlmError> {
        if let Some(json) = self.get_setting("provider_config")? {
            Ok(serde_json::from_str(&json)?)
        } else {
            Ok(ProviderConfig::default())
        }
    }

    /// Save the LLM provider configuration to settings.
    pub fn set_provider_config(&self, config: &ProviderConfig) -> Result<(), LlmError> {
        let json = serde_json::to_string(config)?;
        self.set_setting("provider_config", &json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_list_conversations() {
        let store = HistoryStore::in_memory().unwrap();

        let conv = store
            .create_conversation("Test Chat", ProviderType::Ollama, "llama3.2")
            .unwrap();
        assert_eq!(conv.title, "Test Chat");

        let conversations = store.list_conversations().unwrap();
        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].id, conv.id);
    }

    #[test]
    fn test_add_and_get_messages() {
        let store = HistoryStore::in_memory().unwrap();

        let conv = store
            .create_conversation("Test", ProviderType::Ollama, "llama3.2")
            .unwrap();

        let msg1 = Message::user("Hello");
        let msg2 = Message::assistant("Hi there!");

        store.add_message(&conv.id, &msg1).unwrap();
        store.add_message(&conv.id, &msg2).unwrap();

        let messages = store.get_messages(&conv.id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "Hi there!");
    }

    #[test]
    fn test_settings() {
        let store = HistoryStore::in_memory().unwrap();

        store.set_setting("test_key", "test_value").unwrap();
        let value = store.get_setting("test_key").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        let missing = store.get_setting("nonexistent").unwrap();
        assert_eq!(missing, None);
    }

    #[test]
    fn test_provider_config() {
        let store = HistoryStore::in_memory().unwrap();

        let mut config = ProviderConfig::default();
        config.provider_type = ProviderType::Anthropic;
        config.model = "claude-3-sonnet".to_string();
        config.api_key = Some("test-key".to_string());

        store.set_provider_config(&config).unwrap();

        let loaded = store.get_provider_config().unwrap();
        assert_eq!(loaded.provider_type, ProviderType::Anthropic);
        assert_eq!(loaded.model, "claude-3-sonnet");
        assert_eq!(loaded.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_delete_conversation() {
        let store = HistoryStore::in_memory().unwrap();

        let conv = store
            .create_conversation("To Delete", ProviderType::Ollama, "llama3.2")
            .unwrap();
        store.add_message(&conv.id, &Message::user("test")).unwrap();

        store.delete_conversation(&conv.id).unwrap();

        let conversations = store.list_conversations().unwrap();
        assert!(conversations.is_empty());

        let messages = store.get_messages(&conv.id).unwrap();
        assert!(messages.is_empty());
    }
}
