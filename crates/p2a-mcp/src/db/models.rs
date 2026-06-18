//! Database models for SurrealDB persistence
//!
//! These structs map to SurrealDB tables for storing conversations,
//! messages, tool calls, sessions, and settings.
//!
//! Uses SurrealDB's native types for proper database serialization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;
use surrealdb::sql::Datetime;

/// A conversation (chat session)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Unique conversation ID
    pub id: RecordId,
    /// Session this conversation belongs to
    pub session_id: String,
    /// User-visible title
    pub title: String,
    /// When the conversation was created
    pub created_at: Datetime,
    /// When the conversation was last updated
    pub updated_at: Datetime,
    /// Whether the conversation is archived
    pub is_archived: bool,
    /// Number of messages in this conversation
    pub message_count: i32,
    /// Preview of the last message (for UI display)
    pub last_message_preview: Option<String>,
}

impl Conversation {
    /// Create a new conversation
    pub fn new(session_id: String, title: String) -> Self {
        let now = Datetime::from(Utc::now());
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("conversations", id.as_str())),
            session_id,
            title,
            created_at: now.clone(),
            updated_at: now,
            is_archived: false,
            message_count: 0,
            last_message_preview: None,
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        // RecordIdKey.to_string() may wrap string IDs in angle brackets ⟨⟩
        // We need to extract the raw string value
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }

    /// Get created_at as chrono DateTime
    pub fn created_at_chrono(&self) -> DateTime<Utc> {
        DateTime::from(self.created_at.clone())
    }

    /// Get updated_at as chrono DateTime
    pub fn updated_at_chrono(&self) -> DateTime<Utc> {
        DateTime::from(self.updated_at.clone())
    }
}

/// A chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: RecordId,
    /// Conversation this message belongs to
    pub conversation_id: String,
    /// Message role (user, assistant, system)
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// When the message was created
    pub created_at: Datetime,
    /// Token count (if available)
    pub token_count: Option<i32>,
    /// Model used (for assistant messages)
    pub model: Option<String>,
    /// Finish reason (for assistant messages)
    pub finish_reason: Option<String>,
}

/// Message role enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl Message {
    /// Create a new user message
    pub fn user(conversation_id: String, content: String) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("messages", id.as_str())),
            conversation_id,
            role: MessageRole::User,
            content,
            created_at: Datetime::from(Utc::now()),
            token_count: None,
            model: None,
            finish_reason: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant(conversation_id: String, content: String, model: Option<String>) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("messages", id.as_str())),
            conversation_id,
            role: MessageRole::Assistant,
            content,
            created_at: Datetime::from(Utc::now()),
            token_count: None,
            model,
            finish_reason: Some("stop".to_string()),
        }
    }

    /// Create a new system message
    pub fn system(conversation_id: String, content: String) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("messages", id.as_str())),
            conversation_id,
            role: MessageRole::System,
            content,
            created_at: Datetime::from(Utc::now()),
            token_count: None,
            model: None,
            finish_reason: None,
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }

    /// Get created_at as chrono DateTime
    pub fn created_at_chrono(&self) -> DateTime<Utc> {
        DateTime::from(self.created_at.clone())
    }
}

/// A tool call made by the assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique tool call ID
    pub id: RecordId,
    /// Message this tool call belongs to
    pub message_id: String,
    /// Conversation this tool call belongs to
    pub conversation_id: String,
    /// Name of the tool
    pub tool_name: String,
    /// Tool arguments as JSON string
    pub arguments: String,
    /// Tool result as JSON string (if completed)
    pub result: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Tool call status
    pub status: ToolCallStatus,
    /// When the tool call started
    pub started_at: Datetime,
    /// When the tool call completed
    pub completed_at: Option<Datetime>,
    /// Duration in milliseconds
    pub duration_ms: Option<i32>,
}

/// Tool call status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallStatus {
    Pending,
    Running,
    Success,
    Error,
}

impl ToolCall {
    /// Create a new pending tool call
    pub fn new(
        message_id: String,
        conversation_id: String,
        tool_name: String,
        arguments: String,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("tool_calls", id.as_str())),
            message_id,
            conversation_id,
            tool_name,
            arguments,
            result: None,
            error: None,
            status: ToolCallStatus::Pending,
            started_at: Datetime::from(Utc::now()),
            completed_at: None,
            duration_ms: None,
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }

    /// Mark the tool call as completed successfully
    pub fn complete(&mut self, result: String) {
        self.result = Some(result);
        self.status = ToolCallStatus::Success;
        let completed = Datetime::from(Utc::now());
        let started: DateTime<Utc> = DateTime::from(self.started_at.clone());
        let completed_chrono: DateTime<Utc> = DateTime::from(completed.clone());
        self.duration_ms = Some((completed_chrono - started).num_milliseconds() as i32);
        self.completed_at = Some(completed);
    }

    /// Mark the tool call as failed
    pub fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.status = ToolCallStatus::Error;
        let completed = Datetime::from(Utc::now());
        let started: DateTime<Utc> = DateTime::from(self.started_at.clone());
        let completed_chrono: DateTime<Utc> = DateTime::from(completed.clone());
        self.duration_ms = Some((completed_chrono - started).num_milliseconds() as i32);
        self.completed_at = Some(completed);
    }
}

/// Persisted session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbSession {
    /// Session ID
    pub id: RecordId,
    /// Optional user ID (for authenticated users)
    pub user_id: Option<String>,
    /// When the session was created
    pub created_at: Datetime,
    /// When the session was last accessed
    pub last_accessed: Datetime,
    /// Global random seed for ML reproducibility
    pub global_seed: Option<i64>,
    /// Currently active conversation ID
    pub active_conversation_id: Option<String>,
    /// List of loaded dataset names (for session restoration)
    pub dataset_names: Vec<String>,
}

impl DbSession {
    /// Create a new session record
    pub fn new(id: String, user_id: Option<String>) -> Self {
        let now = Datetime::from(Utc::now());
        Self {
            id: RecordId::from(("sessions", id.as_str())),
            user_id,
            created_at: now.clone(),
            last_accessed: now,
            global_seed: None,
            active_conversation_id: None,
            dataset_names: Vec::new(),
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }

    /// Get created_at as chrono DateTime
    pub fn created_at_chrono(&self) -> DateTime<Utc> {
        DateTime::from(self.created_at.clone())
    }

    /// Get last_accessed as chrono DateTime
    pub fn last_accessed_chrono(&self) -> DateTime<Utc> {
        DateTime::from(self.last_accessed.clone())
    }
}

/// User settings stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Settings ID (same as session_id for 1:1 mapping)
    pub id: RecordId,
    /// Session this settings belongs to
    pub session_id: String,
    /// LLM provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Base URL for provider (optional)
    pub base_url: Option<String>,
    /// Temperature setting
    pub temperature: f64,
    /// Max tokens
    pub max_tokens: i32,
    /// Custom system prompt (optional)
    pub system_prompt: Option<String>,
    /// When settings were last updated
    pub updated_at: Datetime,
}

impl Settings {
    /// Create default settings for a session
    pub fn default_for_session(session_id: String) -> Self {
        Self {
            id: RecordId::from(("settings", session_id.as_str())),
            session_id: session_id.clone(),
            provider: "ollama".to_string(),
            model: "llama3.1".to_string(),
            base_url: None,
            temperature: 0.7,
            max_tokens: 4096,
            system_prompt: None,
            updated_at: Datetime::from(Utc::now()),
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }
}

/// Metadata about a loaded dataset (not the actual data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    /// Unique ID
    pub id: RecordId,
    /// Session this dataset belongs to
    pub session_id: String,
    /// Dataset name (as used in commands)
    pub name: String,
    /// Original file path (if loaded from file)
    pub source_path: Option<String>,
    /// Source type (csv, parquet, json, etc.)
    pub source_type: String,
    /// Number of rows
    pub row_count: i32,
    /// Number of columns
    pub column_count: i32,
    /// Column names
    pub column_names: Vec<String>,
    /// When the dataset was loaded
    pub loaded_at: Datetime,
    /// File size in bytes (if known)
    pub file_size_bytes: Option<i64>,
}

impl DatasetMeta {
    /// Create metadata for a dataset
    pub fn new(
        session_id: String,
        name: String,
        source_type: String,
        row_count: i32,
        column_count: i32,
        column_names: Vec<String>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: RecordId::from(("dataset_meta", id.as_str())),
            session_id,
            name,
            source_path: None,
            source_type,
            row_count,
            column_count,
            column_names,
            loaded_at: Datetime::from(Utc::now()),
            file_size_bytes: None,
        }
    }

    /// Get the ID as a string (just the key part, without SurrealDB formatting)
    pub fn id_string(&self) -> String {
        let key_str = self.id.key().to_string();
        key_str
            .trim_start_matches('⟨')
            .trim_end_matches('⟩')
            .to_string()
    }
}

/// Response type for conversation with messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationWithMessages {
    pub conversation: Conversation,
    pub messages: Vec<Message>,
}
