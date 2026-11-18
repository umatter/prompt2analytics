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

impl ToolCallInfo {
    /// Whether the call has ended: it has an outcome or a result. A finished
    /// call with `success == None` ended with an unknown outcome (a server
    /// older than 0.6.0 sends no status).
    pub fn is_finished(&self) -> bool {
        self.success.is_some() || self.result.is_some()
    }
}

/// A `tool_end` status as a call's outcome: `"success"` and `"error"` (the
/// 0.6.0 server's values), anything else or none unknown.
pub fn tool_outcome(status: Option<&str>) -> Option<bool> {
    match status {
        Some("success") => Some(true),
        Some("error") => Some(false),
        _ => None,
    }
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

/// The rows of a turn the client stores itself: the user's prompt, when the
/// turn has a conversation, posted before the stream starts. The server
/// stores the assistant's reply with its tool calls (API_ENDPOINTS.md, chat
/// routes), so the client posts no assistant row.
pub fn turn_persistence(conversation_id: &str) -> &'static [&'static str] {
    if conversation_id.is_empty() {
        &[]
    } else {
        &["user"]
    }
}

/// Active tool call being processed
#[derive(Debug, Clone)]
pub struct ActiveToolCall {
    pub name: String,
    pub started_at: DateTime<Utc>,
    /// Index of the call its start added to the streaming message's calls.
    pub index: Option<usize>,
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
    /// The prompt Retry sends when the transcript was cleared (the session
    /// ended); kept by `clear_messages`.
    pub retry_prompt: Option<String>,
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
            retry_prompt: None,
        }
    }

    /// Whether an answer is being written (a send runs).
    pub fn is_busy(&self) -> bool {
        self.is_processing
    }

    /// Take the prompt Retry should send after the transcript was cleared.
    pub fn take_retry_prompt(&mut self) -> Option<String> {
        self.retry_prompt.take()
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
                            result: None,  // Will be filled from tool_results if present
                            success: None, // Unknown unless a result says otherwise
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

    /// End the trailing streaming message without a Done: its partial text
    /// is kept and it stops streaming; an empty one shows `note` instead.
    /// Calls still running keep an unknown outcome.
    pub fn end_interrupted(&mut self, note: &str) {
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            last.is_streaming = false;
            if last.content.is_empty() {
                last.content = note.to_string();
            }
        }
        self.active_tool = None;
    }

    /// Stop processing
    pub fn stop_processing(&mut self) {
        self.is_processing = false;
        self.status = None;
        self.active_tool = None;
    }

    /// Set the active tool call and add a running call to the streaming
    /// message. The server runs one call at a time, so the next `tool_end`
    /// belongs to this call.
    pub fn set_active_tool(&mut self, name: String, arguments: serde_json::Value) {
        let mut index = None;
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            index = Some(last.tool_calls.len());
            last.tool_calls.push(ToolCallInfo {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.clone(),
                arguments,
                result: None,
                success: None,
                is_expanded: false,
            });
        }
        self.active_tool = Some(ActiveToolCall {
            name,
            started_at: Utc::now(),
            index,
        });
    }

    /// End the active tool call with its result and its `tool_end` status
    /// (`"success"`, `"error"`, or none from an older server: unknown).
    pub fn clear_active_tool(&mut self, result: Option<String>, status: Option<&str>) {
        let Some(active) = self.active_tool.take() else {
            return;
        };
        let Some(last) = self.messages.last_mut().filter(|m| m.is_streaming) else {
            return;
        };
        let at_index = active.index.filter(|&i| {
            last.tool_calls
                .get(i)
                .is_some_and(|tc| tc.name == active.name && !tc.is_finished())
        });
        // A stale index (the calls were replaced): the last running call of that name.
        let target = at_index.or_else(|| {
            last.tool_calls
                .iter()
                .rposition(|tc| tc.name == active.name && !tc.is_finished())
        });
        if let Some(tc) = target.and_then(|i| last.tool_calls.get_mut(i)) {
            tc.success = tool_outcome(status);
            tc.result = result;
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

    /// Index of the user message that opened the turn in flight: the message just before the
    /// trailing streaming placeholder(s), when it is a user message. `None` when the transcript
    /// does not end in a streaming message.
    fn in_flight_prompt(&self) -> Option<usize> {
        let streaming = self
            .messages
            .iter()
            .rev()
            .take_while(|m| m.is_streaming)
            .count();
        if streaming == 0 {
            return None;
        }
        let i = self.messages.len().checked_sub(streaming + 1)?;
        self.messages.get(i).filter(|m| m.role == "user").map(|_| i)
    }

    /// Build message history for API request: the transcript *before* the turn being sent.
    ///
    /// ChatPanel::handle_send adds the user's message and a streaming placeholder, then calls
    /// this. That message is the request's `message`, which the backend appends after the history,
    /// so it is left out here.
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

        let in_flight = self.in_flight_prompt();
        for (i, msg) in self.messages.iter().enumerate() {
            if msg.is_streaming || Some(i) == in_flight {
                continue;
            }
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
                                    crate::utils::truncate_on_char_boundary(result_content, 2000),
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
                            format!("{}...", crate::utils::truncate_on_char_boundary(r, 200))
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

#[cfg(test)]
mod tests {
    //! ChatState as ChatPanel::handle_send drives it.
    use super::*;
    use crate::api::ToolResult;
    use serde_json::json;

    fn reply(content: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_results: None,
        }
    }

    /// The first half of a turn, as handle_send runs it: processing starts, the prompt and a
    /// streaming placeholder are added, then the history for the request is built.
    fn send(chat: &mut ChatState, prompt: &str) -> Vec<Message> {
        chat.start_processing();
        chat.add_user_message(prompt);
        chat.add_streaming_message();
        chat.build_history()
    }

    /// The end of a turn: the Done event, then stop_processing.
    fn finish(chat: &mut ChatState, done: Message) {
        chat.finalize_message(done);
        chat.stop_processing();
    }

    /// interpret = false: Done carries the calls and the backend's is_error for each result. A
    /// failed call is marked failed and goes back to the model as is_error = true.
    #[test]
    fn test_failed_tool_results_stay_failed_in_history() {
        let mut chat = ChatState::new();
        send(&mut chat, "fit both models");
        chat.set_active_tool("regression_ols".to_string(), json!({"dataset": "sales"}));
        chat.clear_active_tool(Some("OLS Regression Results".to_string()), Some("success"));
        chat.set_active_tool(
            "panel_fixed_effects".to_string(),
            json!({"dataset": "sales"}),
        );
        chat.clear_active_tool(
            Some("Error: column 'firm' not found".to_string()),
            Some("error"),
        );
        finish(
            &mut chat,
            Message {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: Some(vec![
                    ToolCall {
                        id: "call_1".into(),
                        name: "regression_ols".into(),
                        arguments: json!({"dataset": "sales"}),
                    },
                    ToolCall {
                        id: "call_2".into(),
                        name: "panel_fixed_effects".into(),
                        arguments: json!({"dataset": "sales"}),
                    },
                ]),
                tool_results: Some(vec![
                    ToolResult {
                        tool_call_id: "call_1".into(),
                        content: "OLS Regression Results".into(),
                        is_error: false,
                    },
                    ToolResult {
                        tool_call_id: "call_2".into(),
                        content: "Error: column 'firm' not found".into(),
                        is_error: true,
                    },
                ]),
            },
        );
        let last = chat.messages.last().expect("the reply");
        assert!(!last.is_streaming);
        let status: Vec<(&str, Option<bool>)> = last
            .tool_calls
            .iter()
            .map(|tc| (tc.id.as_str(), tc.success))
            .collect();
        assert_eq!(status, [("call_1", Some(true)), ("call_2", Some(false))]);

        let history = chat.build_history();
        let roles: Vec<&str> = history.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool"]);
        let ids: Vec<&str> = history[1]
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(ids, ["call_1", "call_2"]);
        let results: Vec<(&str, bool)> = history[2]
            .tool_results
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|r| (r.tool_call_id.as_str(), r.is_error))
            .collect();
        assert_eq!(results, [("call_1", false), ("call_2", true)]);
    }

    /// interpret = true: Done carries no tool calls. Each ToolEnd result lands on its own running
    /// call, even for two calls of one tool, and the streamed calls stay.
    #[test]
    fn test_tool_end_results_attach_to_the_running_call() {
        let mut chat = ChatState::new();
        send(&mut chat, "summarise price and sqft");
        chat.set_active_tool("summary_stats".to_string(), json!({"column": "price"}));
        chat.clear_active_tool(Some("price: mean 312.4".to_string()), Some("success"));
        chat.set_active_tool("summary_stats".to_string(), json!({"column": "sqft"}));
        chat.clear_active_tool(Some("sqft: mean 1840".to_string()), Some("success"));
        finish(&mut chat, reply("Price averages 312.4, sqft 1840."));
        let last = chat.messages.last().expect("the reply");
        assert_eq!(last.content, "Price averages 312.4, sqft 1840.");
        let calls: Vec<(serde_json::Value, Option<&str>)> = last
            .tool_calls
            .iter()
            .map(|tc| (tc.arguments.clone(), tc.result.as_deref()))
            .collect();
        assert_eq!(
            calls,
            [
                (json!({"column": "price"}), Some("price: mean 312.4")),
                (json!({"column": "sqft"}), Some("sqft: mean 1840")),
            ]
        );
        assert!(chat.active_tool.is_none());
        let status: Vec<Option<bool>> = last.tool_calls.iter().map(|tc| tc.success).collect();
        assert_eq!(status, [Some(true), Some(true)]);
    }

    /// The UI-13 scenario: a ToolEnd with status "error" marks the call failed, and it goes back
    /// to the model as a failed result (interpret = true: Done carries no calls).
    #[test]
    fn test_tool_end_error_status_marks_the_call_failed() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe the data");
        chat.set_active_tool(
            "describe_dataset".to_string(),
            json!({"dataset": "does_not_exist"}),
        );
        chat.clear_active_tool(
            Some("Dataset 'does_not_exist' not found. Load it first with load_dataset.".into()),
            Some("error"),
        );
        finish(&mut chat, reply("The dataset is not loaded."));
        let last = chat.messages.last().expect("the reply");
        assert_eq!(last.tool_calls.len(), 1);
        assert_eq!(last.tool_calls[0].success, Some(false));
        assert!(last.tool_calls[0].is_finished());
        let history = chat.build_history();
        let results = history[2].tool_results.as_deref().unwrap_or_default();
        assert!(results[0].is_error, "{history:?}");
    }

    /// An older server sends no status: the call is finished, its outcome unknown, and it is not
    /// reported to the model as an error.
    #[test]
    fn test_tool_end_without_status_is_unknown() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.set_active_tool("describe_dataset".to_string(), json!({"dataset": "sales"}));
        chat.clear_active_tool(Some("100 rows".into()), None);
        let call = &chat.messages.last().unwrap().tool_calls[0];
        assert_eq!(call.success, None);
        assert!(call.is_finished());
        finish(&mut chat, reply("Sales has 100 rows."));
        let history = chat.build_history();
        let results = history[2].tool_results.as_deref().unwrap_or_default();
        assert!(!results[0].is_error);
    }

    /// Two calls of one tool, the first failed: each end lands on its own call, and both cards
    /// stay (a finished call with an unknown outcome is not taken for a running one).
    #[test]
    fn test_two_calls_of_one_tool_each_get_their_end() {
        let mut chat = ChatState::new();
        send(&mut chat, "load both files");
        chat.set_active_tool("load_dataset".to_string(), json!({"path": "a.csv"}));
        assert!(!chat.messages.last().unwrap().tool_calls[0].is_finished());
        chat.clear_active_tool(Some("Error: a.csv not found".into()), Some("error"));
        chat.set_active_tool("load_dataset".to_string(), json!({"path": "b.csv"}));
        chat.clear_active_tool(Some("Loaded b".into()), Some("success"));
        let calls = &chat.messages.last().unwrap().tool_calls;
        let got: Vec<(serde_json::Value, Option<bool>)> = calls
            .iter()
            .map(|tc| (tc.arguments.clone(), tc.success))
            .collect();
        assert_eq!(
            got,
            [
                (json!({"path": "a.csv"}), Some(false)),
                (json!({"path": "b.csv"}), Some(true)),
            ]
        );

        // Two calls that each end without a status are still two cards.
        let mut chat = ChatState::new();
        send(&mut chat, "load both files");
        for path in ["a.csv", "b.csv"] {
            chat.set_active_tool("load_dataset".to_string(), json!({"path": path}));
            chat.clear_active_tool(Some(format!("Loaded {path}")), None);
        }
        assert_eq!(chat.messages.last().unwrap().tool_calls.len(), 2);
    }

    /// interpret = false: a call Done lists without a matching result has an unknown outcome,
    /// not a success.
    #[test]
    fn test_done_call_without_result_is_unknown() {
        let mut chat = ChatState::new();
        send(&mut chat, "fit it");
        finish(
            &mut chat,
            Message {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    name: "regression_ols".into(),
                    arguments: json!({"dataset": "sales"}),
                }]),
                tool_results: Some(vec![]),
            },
        );
        assert_eq!(chat.messages.last().unwrap().tool_calls[0].success, None);
    }

    /// The client stores the user's row of a turn, before the stream; the server stores the
    /// assistant's reply (with its tool calls), so the client posts none.
    #[test]
    fn turn_persistence_posts_only_the_user_row() {
        assert_eq!(turn_persistence("e104f3c9"), ["user"]);
        assert!(turn_persistence("").is_empty());
    }

    #[test]
    fn end_interrupted_keeps_partial_text() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.append_content("The data has");
        chat.set_active_tool("describe_dataset".into(), json!({"dataset": "sales"}));
        chat.end_interrupted("Stopped.");
        let last = chat.messages.last().unwrap();
        assert_eq!(last.content, "The data has");
        assert!(!last.is_streaming);
        assert_eq!(last.tool_calls[0].success, None);
        assert!(chat.active_tool.is_none());

        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.end_interrupted("Stopped.");
        assert_eq!(chat.messages.last().unwrap().content, "Stopped.");
    }

    #[test]
    fn retry_prompt_survives_clear() {
        let mut chat = ChatState::new();
        chat.retry_prompt = Some("regress price on sqft".into());
        chat.clear_messages();
        assert_eq!(chat.retry_prompt.as_deref(), Some("regress price on sqft"));
        assert_eq!(
            chat.take_retry_prompt().as_deref(),
            Some("regress price on sqft")
        );
        assert_eq!(chat.retry_prompt, None);
    }

    /// A result over 2,000 bytes is cut on a char boundary in the history.
    #[test]
    fn test_history_cuts_long_results_on_a_char_boundary() {
        let mut chat = ChatState::new();
        send(&mut chat, "show the labels");
        chat.set_active_tool("head_dataset".to_string(), json!({"dataset": "labels"}));
        // 'a' then 1500 x 'é' (2 bytes each): 3001 bytes. Byte 2000 falls inside an 'é' (bytes
        // 1999..2001), so the cut is at 1999: "a" + 999 x 'é'.
        chat.clear_active_tool(Some(format!("a{}", "é".repeat(1500))), Some("success"));
        finish(&mut chat, reply("Here are the labels."));
        let history = chat.build_history();
        assert_eq!(history.len(), 3, "{history:?}");
        let content = &history[2].tool_results.as_deref().unwrap_or_default()[0].content;
        assert!(
            content.starts_with(&format!("a{}...\n[Result truncated", "é".repeat(999))),
            "{content}"
        );
    }

    fn turns(history: &[Message]) -> Vec<(String, String)> {
        history
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect()
    }

    /// The prompt being sent is the request's `message`, and the backend appends it after
    /// `history`, so it must not be in the history too.
    #[test]
    fn test_build_history_leaves_out_the_prompt_being_sent() {
        let mut chat = ChatState::new();
        assert_eq!(turns(&send(&mut chat, "load sales.csv")), pairs(&[]));
        finish(&mut chat, reply("Loaded sales: 100 rows, 2 columns."));
        assert_eq!(
            turns(&send(&mut chat, "regress price on sqft")),
            pairs(&[
                ("user", "load sales.csv"),
                ("assistant", "Loaded sales: 100 rows, 2 columns.")
            ])
        );
    }

    /// Only the prompt of the trailing turn is left out. An earlier turn whose stream stopped
    /// without Done keeps its prompt; its unfinished reply stays out, as before.
    #[test]
    fn test_build_history_skips_only_the_trailing_prompt() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.append_content("The data has");
        chat.stop_processing();
        assert_eq!(
            turns(&send(&mut chat, "describe sales again")),
            pairs(&[("user", "describe sales")])
        );
    }
}

/// Rule-5 timing of tool start/end pairs against verbatim copies of the
/// base `set_active_tool`/`clear_active_tool` (a scan over the message's calls).
#[cfg(test)]
mod timing {
    use super::*;
    use std::time::Instant;

    fn base_set_active_tool(s: &mut ChatState, name: String, arguments: serde_json::Value) {
        s.active_tool = Some(ActiveToolCall {
            name: name.clone(),
            started_at: Utc::now(),
            index: None,
        });
        if let Some(last) = s.messages.last_mut()
            && last.is_streaming
        {
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
                    success: None,
                    is_expanded: false,
                });
            }
        }
    }

    fn base_clear_active_tool(s: &mut ChatState, result: Option<String>) {
        if let Some(active) = s.active_tool.take() {
            if let Some(last) = s.messages.last_mut()
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

    fn fresh() -> ChatState {
        let mut s = ChatState::new();
        s.add_user_message("q");
        s.add_streaming_message();
        s
    }

    fn run_base(pairs: usize) -> usize {
        let mut s = fresh();
        for _ in 0..pairs {
            base_set_active_tool(&mut s, "regression_ols".into(), serde_json::Value::Null);
            base_clear_active_tool(&mut s, Some("ok".into()));
        }
        s.messages.last().unwrap().tool_calls.len()
    }

    fn run_head(pairs: usize) -> usize {
        let mut s = fresh();
        for _ in 0..pairs {
            s.set_active_tool("regression_ols".into(), serde_json::Value::Null);
            s.clear_active_tool(Some("ok".into()), Some("success"));
        }
        s.messages.last().unwrap().tool_calls.len()
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    #[test]
    #[ignore = "benchmark; times tool start/end pairs against the base scan"]
    fn time_tool_pairs() {
        for pairs in [100, 1_000, 10_000] {
            for _ in 0..3 {
                assert_eq!(run_base(pairs), run_head(pairs));
            }
            let (mut b, mut h) = (Vec::new(), Vec::new());
            for _ in 0..31 {
                let t = Instant::now();
                std::hint::black_box(run_base(pairs));
                b.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                std::hint::black_box(run_head(pairs));
                h.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (mb, mh) = (median(b), median(h));
            println!(
                "TIMING tool pairs {pairs}: base {mb:.4} ms, head {mh:.4} ms, head/base {:.3}",
                mh / mb
            );
        }
    }
}
