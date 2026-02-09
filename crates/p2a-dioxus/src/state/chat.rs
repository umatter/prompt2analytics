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
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            last.content.push_str(text);
        }
    }

    /// Finalize the streaming message with full data
    pub fn finalize_message(&mut self, msg: Message) {
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            last.content = msg.content;
            last.is_streaming = false;

            // Update tool calls from backend response (has full details)
            // This replaces the streaming-tracked tool calls with the authoritative ones
            if let Some(calls) = msg.tool_calls {
                let new_calls: Vec<ToolCallInfo> = calls
                    .iter()
                    .map(|tc| {
                        ToolCallInfo {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            result: None, // Will be filled from tool_results if present
                            success: Some(true), // If we got them in Done, they succeeded
                            is_expanded: false,
                        }
                    })
                    .collect();

                // If we have tool results, match them to tool calls
                if let Some(ref results) = msg.tool_results {
                    let mut final_calls = new_calls;
                    for tc in final_calls.iter_mut() {
                        if let Some(result) = results.iter().find(|r| r.tool_call_id == tc.id) {
                            tc.result = Some(result.content.clone());
                            tc.success = Some(!result.is_error);
                        }
                    }
                    last.tool_calls = final_calls;
                } else {
                    last.tool_calls = new_calls;
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

    /// Set active tool call and add to current message's tool calls
    pub fn set_active_tool(&mut self, name: String, arguments: serde_json::Value) {
        self.active_tool = Some(ActiveToolCall {
            name: name.clone(),
            started_at: Utc::now(),
        });

        // Add a pending tool call to the current streaming message
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            // Only add if not already present (avoid duplicates on re-renders)
            if !last
                .tool_calls
                .iter()
                .any(|tc| tc.name == name && tc.success.is_none())
            {
                last.tool_calls.push(ToolCallInfo {
                    id: uuid::Uuid::new_v4().to_string(),
                    name,
                    arguments,
                    result: None,
                    success: None, // None means still running
                    is_expanded: false,
                });
            }
        }
    }

    /// Clear active tool call and mark it as complete with result
    pub fn clear_active_tool(&mut self, result: Option<String>) {
        if let Some(active) = self.active_tool.take() {
            // Mark the tool call as successful in the current message
            if let Some(last) = self.messages.last_mut()
                && last.is_streaming
                && let Some(tc) = last
                    .tool_calls
                    .iter_mut()
                    .find(|tc| tc.name == active.name && tc.success.is_none())
            {
                tc.success = Some(true);
                tc.result = result;
            }
        }
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

    /// Build message history for API request.
    ///
    /// This method now includes tool calls and their results in the history,
    /// which is critical for multi-turn conversation accuracy. The LLM needs
    /// to know what tools were previously called and their results to handle
    /// follow-up requests like "now run a t-test on those results."
    ///
    /// For OpenAI-compatible APIs: assistant messages with tool_calls are
    /// followed by tool result messages with the corresponding tool_results.
    pub fn build_history(&self) -> Vec<Message> {
        let mut history = Vec::new();

        for msg in self.messages.iter().filter(|m| !m.is_streaming) {
            if msg.role == "user" {
                // User messages are included as-is
                if !msg.content.is_empty() {
                    history.push(Message {
                        role: "user".to_string(),
                        content: msg.content.clone(),
                        tool_calls: None,
                        tool_results: None,
                    });
                }
            } else if msg.role == "assistant" {
                // Check if this assistant message has tool calls with results
                let completed_tool_calls: Vec<_> = msg
                    .tool_calls
                    .iter()
                    .filter(|tc| tc.result.is_some())
                    .collect();

                if !completed_tool_calls.is_empty() {
                    // Create tool calls for the assistant message
                    let tool_calls: Vec<crate::api::ToolCall> = completed_tool_calls
                        .iter()
                        .map(|tc| crate::api::ToolCall {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        })
                        .collect();

                    // Assistant message with tool calls
                    // Note: For OpenAI format, the content can be empty when there are tool calls
                    history.push(Message {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                        tool_calls: Some(tool_calls),
                        tool_results: None,
                    });

                    // Tool result message (required by OpenAI API format)
                    let tool_results: Vec<crate::api::ToolResult> = completed_tool_calls
                        .iter()
                        .map(|tc| {
                            // Truncate very long results to avoid context overflow
                            let result_content = tc.result.as_deref().unwrap_or("");
                            let truncated = if result_content.len() > 2000 {
                                format!(
                                    "{}...\n[Result truncated, {} chars total]",
                                    &result_content[..2000],
                                    result_content.len()
                                )
                            } else {
                                result_content.to_string()
                            };

                            crate::api::ToolResult {
                                tool_call_id: tc.id.clone(),
                                content: truncated,
                                is_error: tc.success == Some(false),
                            }
                        })
                        .collect();

                    history.push(Message {
                        role: "tool".to_string(),
                        content: String::new(),
                        tool_calls: None,
                        tool_results: Some(tool_results),
                    });
                } else if !msg.content.is_empty() {
                    // Assistant message without tool calls - just include content
                    history.push(Message {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                        tool_calls: None,
                        tool_results: None,
                    });
                }
            }
        }

        history
    }

    /// Build a compact summary of conversation context for the system prompt.
    ///
    /// This provides a fallback for LLMs that don't support tool messages in history,
    /// or when we need to reduce context size. It summarizes what tools were called
    /// and their key results.
    pub fn build_tool_context_summary(&self) -> Option<String> {
        let tool_summaries: Vec<String> = self
            .messages
            .iter()
            .filter(|m| !m.is_streaming && m.role == "assistant")
            .flat_map(|m| m.tool_calls.iter())
            .filter(|tc| tc.result.is_some())
            .map(|tc| {
                let result_preview = tc
                    .result
                    .as_deref()
                    .map(|r| {
                        if r.len() > 200 {
                            format!("{}...", &r[..200])
                        } else {
                            r.to_string()
                        }
                    })
                    .unwrap_or_default();

                let status = if tc.success == Some(true) {
                    "✓"
                } else if tc.success == Some(false) {
                    "✗"
                } else {
                    "?"
                };

                format!(
                    "- {} {}({}): {}",
                    status,
                    tc.name,
                    serde_json::to_string(&tc.arguments)
                        .unwrap_or_default()
                        .chars()
                        .take(100)
                        .collect::<String>(),
                    result_preview
                )
            })
            .collect();

        if tool_summaries.is_empty() {
            None
        } else {
            Some(format!(
                "## Previous Tool Calls in This Conversation\n\n{}",
                tool_summaries.join("\n")
            ))
        }
    }
}
