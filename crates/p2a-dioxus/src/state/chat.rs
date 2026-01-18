//! Chat state management

use crate::api::{ConversationMessage, Message, PersistedToolCall, ToolCall};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// A chat message in the UI
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Unique message ID
    pub id: String,
    /// Message role (user or assistant)
    pub role: String,
    /// Message content (may be streaming)
    pub content: String,
    /// Tool calls made by the assistant
    pub tool_calls: Vec<ToolCallInfo>,
    /// Base64-encoded images from tool results
    pub images: Vec<String>,
    /// Whether this message is currently streaming
    pub is_streaming: bool,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            is_streaming: false,
            timestamp: Utc::now(),
        }
    }

    /// Create a new assistant message (streaming)
    pub fn assistant_streaming() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            is_streaming: true,
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant message from API response
    pub fn from_api_message(msg: &Message) -> Self {
        let tool_calls: Vec<ToolCallInfo> = msg
            .tool_calls
            .as_ref()
            .map(|calls| calls.iter().map(ToolCallInfo::from).collect::<Vec<_>>())
            .unwrap_or_default();

        Self {
            id: Uuid::new_v4().to_string(),
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls,
            images: Vec::new(),
            is_streaming: false,
            timestamp: Utc::now(),
        }
    }

    /// Create a chat message from a persisted conversation message
    pub fn from_conversation_message(msg: &ConversationMessage) -> Self {
        // Parse the timestamp from the stored string
        let timestamp = chrono::DateTime::parse_from_rfc3339(&msg.created_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Self {
            id: msg.id.clone(),
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls: Vec::new(), // Tool calls will be loaded separately
            images: Vec::new(),
            is_streaming: false,
            timestamp,
        }
    }

    /// Create a chat message from a persisted conversation message with tool calls
    pub fn from_conversation_message_with_tools(
        msg: &ConversationMessage,
        tool_calls: &[PersistedToolCall],
    ) -> Self {
        let mut chat_msg = Self::from_conversation_message(msg);
        chat_msg.tool_calls = tool_calls.iter().map(ToolCallInfo::from).collect();
        chat_msg
    }

    /// Set tool calls on this message
    pub fn set_tool_calls(&mut self, tool_calls: Vec<ToolCallInfo>) {
        self.tool_calls = tool_calls;
    }
}

/// Information about a tool call
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool arguments as JSON
    pub arguments: serde_json::Value,
    /// Tool result (if completed)
    pub result: Option<String>,
    /// Whether the tool call succeeded
    pub success: Option<bool>,
    /// Whether the tool call is expanded in the UI
    pub is_expanded: bool,
}

impl From<&ToolCall> for ToolCallInfo {
    fn from(tc: &ToolCall) -> Self {
        Self {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            result: None,
            success: None,
            is_expanded: false,
        }
    }
}

impl From<&PersistedToolCall> for ToolCallInfo {
    fn from(tc: &PersistedToolCall) -> Self {
        // Parse arguments from string
        let arguments = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);

        // Determine success from status
        let success = match tc.status.as_str() {
            "success" => Some(true),
            "error" => Some(false),
            _ => None,
        };

        // Use result or error as the result text
        let result = tc.result.clone().or_else(|| tc.error.clone());

        Self {
            id: tc.id.clone(),
            name: tc.tool_name.clone(),
            arguments,
            result,
            success,
            is_expanded: false,
        }
    }
}

/// Active tool call being processed
#[derive(Debug, Clone)]
pub struct ActiveToolCall {
    pub name: String,
    pub started_at: DateTime<Utc>,
}

/// Chat state
#[derive(Clone)]
pub struct ChatState {
    /// All messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Current input text
    pub input: String,
    /// Whether a request is being processed
    pub is_processing: bool,
    /// Current status message
    pub status: Option<String>,
    /// Current error message
    pub error: Option<String>,
    /// Prompt history for arrow-key navigation
    pub prompt_history: Vec<String>,
    /// Current history index (-1 = not navigating)
    pub history_index: i32,
    /// Currently active tool call (being processed)
    pub active_tool: Option<ActiveToolCall>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatState {
    /// Create a new chat state
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            is_processing: false,
            status: None,
            error: None,
            prompt_history: Vec::new(),
            history_index: -1,
            active_tool: None,
        }
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: &str) {
        let message = ChatMessage::user(content);
        self.messages.push(message);

        // Add to prompt history
        if !content.trim().is_empty() {
            self.prompt_history.push(content.to_string());
        }
    }

    /// Add a streaming assistant message
    pub fn add_streaming_message(&mut self) {
        let message = ChatMessage::assistant_streaming();
        self.messages.push(message);
    }

    /// Append content to the current streaming message
    pub fn append_content(&mut self, text: &str) {
        if let Some(last) = self.messages.last_mut() {
            if last.is_streaming {
                last.content.push_str(text);
            }
        }
    }

    /// Finalize the streaming message with full data
    pub fn finalize_message(&mut self, msg: Message) {
        if let Some(last) = self.messages.last_mut() {
            if last.is_streaming {
                last.content = msg.content;
                last.is_streaming = false;

                // Update tool calls
                if let Some(calls) = msg.tool_calls {
                    last.tool_calls = calls.iter().map(ToolCallInfo::from).collect::<Vec<_>>();
                }
            }
        }
    }

    /// Add an image to the current streaming message
    pub fn add_image(&mut self, base64_data: &str) {
        if let Some(last) = self.messages.last_mut() {
            last.images.push(base64_data.to_string());
        }
    }

    /// Set the current status
    pub fn set_status(&mut self, status: Option<String>) {
        self.status = status;
    }

    /// Set error message
    pub fn set_error(&mut self, error: Option<String>) {
        self.error = error;
    }

    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.error = None;
        self.status = None;
    }

    /// Start processing
    pub fn start_processing(&mut self) {
        self.is_processing = true;
        self.error = None;
    }

    /// Stop processing
    pub fn stop_processing(&mut self) {
        self.is_processing = false;
        self.status = None;
        self.active_tool = None;
    }

    /// Set active tool call
    pub fn set_active_tool(&mut self, name: String) {
        self.active_tool = Some(ActiveToolCall {
            name,
            started_at: Utc::now(),
        });
    }

    /// Clear active tool call
    pub fn clear_active_tool(&mut self) {
        self.active_tool = None;
    }

    /// Navigate history up (older messages)
    pub fn navigate_history_up(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }

        if self.history_index == -1 {
            // Starting navigation, go to most recent
            self.history_index = (self.prompt_history.len() - 1) as i32;
        } else if self.history_index > 0 {
            self.history_index -= 1;
        }

        if let Some(content) = self.prompt_history.get(self.history_index as usize) {
            self.input = content.clone();
        }
    }

    /// Navigate history down (newer messages)
    pub fn navigate_history_down(&mut self) {
        if self.history_index == -1 {
            return;
        }

        self.history_index += 1;

        if self.history_index >= self.prompt_history.len() as i32 {
            // Gone past the end, clear input
            self.history_index = -1;
            self.input.clear();
        } else if let Some(content) = self.prompt_history.get(self.history_index as usize) {
            self.input = content.clone();
        }
    }

    /// Reset history navigation
    pub fn reset_history_index(&mut self) {
        self.history_index = -1;
    }

    /// Build message history for API request
    pub fn build_history(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| !m.is_streaming)
            .map(|m| Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: if m.tool_calls.is_empty() {
                    None
                } else {
                    Some(
                        m.tool_calls
                            .iter()
                            .map(|tc| crate::api::types::ToolCall {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            })
                            .collect(),
                    )
                },
                tool_results: None,
            })
            .collect()
    }
}
