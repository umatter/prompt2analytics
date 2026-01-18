//! API request/response types for communicating with p2a-mcp backend

use serde::{Deserialize, Serialize};

/// Generic API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(default)]
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// Session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: String,
    pub last_accessed: String,
    pub dataset_count: u32,
    #[serde(default)]
    pub user_id: Option<String>,
}

/// Session creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    #[serde(default)]
    pub user_id: Option<String>,
}

/// Session creation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
}

/// Tool call information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool result from execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Content item in tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub success: bool,
    pub content: Vec<ContentItem>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub session_id: String,
    pub arguments: serde_json::Value,
}

/// Chat message (for history)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_results: Option<Vec<ToolResult>>,
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_type: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// LLM chat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmChatRequest {
    pub session_id: String,
    pub message: String,
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
    #[serde(default)]
    pub history: Option<Vec<Message>>,
    #[serde(default = "default_interpret")]
    pub interpret: bool,
    /// Optional conversation ID for persistence
    #[serde(default)]
    pub conversation_id: Option<String>,
}

fn default_interpret() -> bool {
    true
}

/// Image data from tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub data: String,
    pub mime_type: String,
}

/// SSE stream event types
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// Status update (e.g., "Calling tool...")
    #[serde(rename = "status")]
    Status { message: String },

    /// Tool execution started
    #[serde(rename = "tool_start")]
    ToolStart { tool: String },

    /// Tool execution completed
    #[serde(rename = "tool_end")]
    ToolEnd { tool: String, elapsed_ms: u64 },

    /// Tool result with potential images
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        images: Option<Vec<ImageData>>,
    },

    /// Streaming content chunk
    #[serde(rename = "content")]
    Content { text: String },

    /// Streaming complete with final message
    #[serde(rename = "done")]
    Done { message: Message },

    /// Error occurred
    #[serde(rename = "error")]
    Error { error: String },
}

/// Tool definition from the backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub server: String,
    pub version: String,
    pub active_sessions: u32,
}

// === Conversation types ===

/// Conversation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub session_id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub is_archived: bool,
    #[serde(default)]
    pub message_count: i32,
    #[serde(default)]
    pub last_message_preview: Option<String>,
}

/// Conversation message (stored in DB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub id: String,
    pub conversation_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
    #[serde(default)]
    pub token_count: Option<i32>,
    #[serde(default)]
    pub model: Option<String>,
}

/// Conversation with all messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationWithMessages {
    pub conversation: Conversation,
    pub messages: Vec<ConversationMessage>,
}

/// Create conversation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateConversationRequest {
    pub title: String,
}

/// Update conversation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConversationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_archived: Option<bool>,
}

/// Add message request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddMessageRequest {
    pub role: String,
    pub content: String,
}

// === Persisted Tool Call types ===

/// Persisted tool call (from database)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistedToolCall {
    pub id: String,
    pub message_id: String,
    pub conversation_id: String,
    pub tool_name: String,
    pub arguments: String,
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    pub status: String,
    pub started_at: String,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<i32>,
}

// === Dataset Metadata types ===

/// Dataset metadata for persistence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetMeta {
    pub id: String,
    pub session_id: String,
    pub name: String,
    #[serde(default)]
    pub source_path: Option<String>,
    pub source_type: String,
    pub row_count: i32,
    pub column_count: i32,
    pub column_names: Vec<String>,
    pub loaded_at: String,
    #[serde(default)]
    pub file_size_bytes: Option<i64>,
}

/// Result of reloading session datasets
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReloadResult {
    /// Names of successfully reloaded datasets
    pub succeeded: Vec<String>,
    /// Datasets that failed to reload
    pub failed: Vec<ReloadFailure>,
    /// Datasets that were skipped
    pub skipped: Vec<ReloadSkip>,
}

/// A dataset that failed to reload
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReloadFailure {
    /// Dataset name
    pub name: String,
    /// Original source path
    pub source_path: String,
    /// Error message
    pub error: String,
}

/// A dataset that was skipped during reload
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReloadSkip {
    /// Dataset name
    pub name: String,
    /// Reason for skipping
    pub reason: String,
}
