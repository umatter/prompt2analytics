//! Chat state management

use crate::api::{ConversationMessage, Message, PersistedToolCall, ToolCall};
use chrono::{DateTime, Utc};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// The source of `ChatMessage::tool_rev`: one counter for the process, so
/// no two states of any messages share a revision, not even those of a
/// message rebuilt with the same id.
static NEXT_TOOL_REV: AtomicU64 = AtomicU64::new(1);

/// A tool-call revision never handed out before.
fn next_rev() -> u64 {
    NEXT_TOOL_REV.fetch_add(1, Ordering::Relaxed)
}

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
    /// Whether the answer ended before it was finished (stopped, a stream
    /// that closed or went silent, a session that ended)
    pub interrupted: bool,
    /// A fresh revision ([`next_rev`], unique in the process) whenever
    /// `tool_calls` change (a call added, ended or replaced) and at every
    /// construction, so the props comparison sees the change in constant
    /// time. Write the calls through `set_tool_calls` or the ChatState
    /// methods.
    pub(crate) tool_rev: u64,
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
            interrupted: false,
            tool_rev: next_rev(),
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
            interrupted: false,
            tool_rev: next_rev(),
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
            interrupted: false,
            tool_rev: next_rev(),
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
            interrupted: false,
            tool_rev: next_rev(),
            timestamp,
        }
    }

    /// Create a chat message from a persisted conversation message with tool calls
    pub fn from_conversation_message_with_tools(
        msg: &ConversationMessage,
        tool_calls: &[PersistedToolCall],
    ) -> Self {
        let mut chat_msg = Self::from_conversation_message(msg);
        chat_msg.set_tool_calls(tool_calls.iter().map(ToolCallInfo::from).collect());
        chat_msg
    }

    /// Set tool calls on this message
    pub fn set_tool_calls(&mut self, tool_calls: Vec<ToolCallInfo>) {
        self.tool_calls = tool_calls;
        self.tool_rev = next_rev();
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

/// The banner when the stream closed before the answer finished.
pub const CLOSED_EARLY: &str =
    "The connection closed before the answer finished. Retry to ask again.";

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
    /// The live transcripts (plots and tool results included) of the
    /// conversations left in this page session, least recently left first;
    /// at most `LIVE_CACHE_CONVERSATIONS`.
    live: Vec<(String, Vec<ChatMessage>)>,
}

/// How many left conversations keep their live transcript.
pub const LIVE_CACHE_CONVERSATIONS: usize = 8;

/// What the chat panel's sync effect reads: the conversation shown and the
/// one the transcript was last synced to, the stored rows of the current
/// conversation now and at the last sync, and the transcript's state.
#[derive(Debug, Clone, Copy)]
pub struct SyncInput<'a> {
    pub current: Option<&'a str>,
    pub synced: Option<&'a str>,
    pub stored_len: usize,
    pub prev_stored_len: usize,
    /// The transcript holds messages (after the left one was stashed)
    pub chat_has_messages: bool,
    pub busy: bool,
    /// The current conversation has a live transcript in the cache
    pub has_live: bool,
    /// The transcript shown was restored from the cache for the current
    /// conversation
    pub restored: bool,
}

/// The sync tracking while a send runs: `(synced id, synced stored-row
/// count)`. The conversation a send created is tracked with no rows; rows of
/// the tracked conversation that arrive meanwhile (a select's load still
/// running when the send started) are counted, so the end of the send does
/// not read them as newly arrived and rebuild over the answer just written.
pub fn busy_tracking(
    current: Option<&str>,
    synced: Option<&str>,
    stored_len: usize,
    prev_stored_len: usize,
) -> (Option<String>, usize) {
    if current != synced {
        (current.map(str::to_string), 0)
    } else {
        (synced.map(str::to_string), stored_len.max(prev_stored_len))
    }
}

/// What the sync effect does with the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncAction {
    /// Leave it
    Skip,
    /// Leave it and track the current conversation (no stored rows yet)
    Track,
    /// Show the current conversation's live transcript
    Restore,
    /// Rebuild it from the stored rows
    Rebuild,
    /// Empty it (the stored rows, if any, arrive later)
    Clear,
}

/// The sync effect's decision, as a pure function.
pub fn decide_sync(i: &SyncInput) -> SyncAction {
    if i.busy {
        return SyncAction::Skip;
    }
    let Some(current) = i.current else {
        return if i.synced.is_some() {
            SyncAction::Clear
        } else {
            SyncAction::Skip
        };
    };
    if i.synced == Some(current) {
        let rows_arrived = i.stored_len > 0 && i.prev_stored_len == 0;
        return if rows_arrived && !i.restored {
            SyncAction::Rebuild
        } else {
            SyncAction::Skip
        };
    }
    if i.has_live {
        SyncAction::Restore
    } else if i.stored_len > 0 {
        SyncAction::Rebuild
    } else if i.chat_has_messages {
        SyncAction::Clear
    } else {
        SyncAction::Track
    }
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
            live: Vec::new(),
        }
    }

    /// Keep the transcript of `conv_id`, which is being left, for this page
    /// session (moved, not copied); the transcript is empty afterwards. The
    /// least recently left conversation beyond the cache's bound is dropped.
    pub fn stash_live(&mut self, conv_id: &str) {
        let messages = std::mem::take(&mut self.messages);
        self.live.retain(|(id, _)| id != conv_id);
        self.live.push((conv_id.to_string(), messages));
        if self.live.len() > LIVE_CACHE_CONVERSATIONS {
            let excess = self.live.len() - LIVE_CACHE_CONVERSATIONS;
            self.live.drain(..excess);
        }
    }

    /// Show the live transcript of `conv_id`, taken out of the cache;
    /// false (and nothing changed) when there is none.
    pub fn restore_live(&mut self, conv_id: &str) -> bool {
        let Some(pos) = self.live.iter().position(|(id, _)| id == conv_id) else {
            return false;
        };
        self.messages = self.live.remove(pos).1;
        true
    }

    /// Whether `conv_id` has a live transcript in the cache.
    pub fn has_live(&self, conv_id: &str) -> bool {
        self.live.iter().any(|(id, _)| id == conv_id)
    }

    /// The cached live transcript of `conv_id`, if any (a late tool-call
    /// load fills it in).
    pub fn live_messages_mut(&mut self, conv_id: &str) -> Option<&mut Vec<ChatMessage>> {
        self.live
            .iter_mut()
            .find(|(id, _)| id == conv_id)
            .map(|(_, m)| m)
    }

    /// Drop the live transcript of `conv_id` (deleted).
    pub fn forget_live(&mut self, conv_id: &str) {
        self.live.retain(|(id, _)| id != conv_id);
    }

    /// Drop every live transcript (the session and its conversations ended).
    pub fn clear_live(&mut self) {
        self.live.clear();
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
                    last.set_tool_calls(final_calls);
                } else {
                    last.set_tool_calls(new_calls);
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

    /// Start processing unless a send already runs: the gate `handle_send`
    /// passes synchronously, before its task is spawned, so a second click
    /// cannot start a second send.
    pub fn try_start_processing(&mut self) -> bool {
        if self.is_busy() {
            return false;
        }
        self.start_processing();
        true
    }

    /// End the trailing streaming message without a Done: its partial text
    /// is kept, it stops streaming and is marked interrupted; an empty one
    /// shows `note` instead. Calls still running keep an unknown outcome.
    pub fn end_interrupted(&mut self, note: &str) {
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            last.is_streaming = false;
            last.interrupted = true;
            if last.content.is_empty() {
                last.content = note.to_string();
            }
        }
        self.active_tool = None;
    }

    /// The send of `prompt` failed before its turn was set up (no session
    /// could be had): the turn is added and ended as not sent, `error` is
    /// shown and processing ends, so Retry sends `prompt` again (and takes
    /// the unsent turn out) rather than the prompt before it.
    pub fn fail_before_turn(&mut self, prompt: &str, error: String) {
        self.add_user_message(prompt);
        self.add_streaming_message();
        self.end_interrupted("Not sent.");
        self.set_error(Some(error));
        self.stop_processing();
    }

    /// Run once after the stream returned, whatever its result: an answer
    /// still streaming is ended as interrupted (its partial text kept), and
    /// when no Done came (`saw_done` false) and no error is shown, the
    /// banner says the stream closed early. An answer already finished or
    /// ended (a Done, a stop) is left as it is.
    pub fn finish_stream(&mut self, saw_done: bool) {
        let streaming = self.messages.last().is_some_and(|m| m.is_streaming);
        if !streaming {
            return;
        }
        self.end_interrupted("");
        if !saw_done && self.error.is_none() {
            self.error = Some(CLOSED_EARLY.to_string());
        }
    }

    /// The prompt Retry sends, with the turn it replaces taken out: after an
    /// answer that is empty or interrupted, that answer and its prompt are
    /// removed (the send adds the prompt again); after a finished answer the
    /// last prompt is asked again and nothing is removed. With no prompt in
    /// the transcript (the session ended and it was cleared), the prompt that
    /// was being sent. That stashed prompt is taken either way, so a stale
    /// one is never sent later.
    pub fn take_retry_turn(&mut self) -> Option<String> {
        let stashed = self.take_retry_prompt();
        if let Some(last) = self.messages.last()
            && last.role == "assistant"
            && !last.is_streaming
            && (last.interrupted || last.content.is_empty())
            && let Some(before) = self.messages.len().checked_sub(2)
            && self.messages[before].role == "user"
        {
            self.messages.pop();
            return self.messages.pop().map(|m| m.content);
        }
        self.messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.clone())
            .or(stashed)
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
            last.tool_rev = next_rev();
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
            last.tool_rev = next_rev();
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
                    // Note: For OpenAI format, the content can be empty when there are tool calls.
                    // An interrupted answer's partial text is not sent; its tools ran.
                    history.push(Message {
                        role: "assistant".to_string(),
                        content: if msg.interrupted {
                            String::new()
                        } else {
                            msg.content.clone()
                        },
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
                } else if !msg.content.is_empty() && !msg.interrupted {
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

    #[test]
    fn rows_that_arrive_during_a_send_do_not_rebuild_after_it() {
        // A select's load lands while the send runs on that conversation.
        let (id, len) = super::busy_tracking(Some("b"), Some("b"), 4, 0);
        assert_eq!((id.as_deref(), len), (Some("b"), 4));
        // After the send, the same rows no longer look newly arrived.
        let action = super::decide_sync(&super::SyncInput {
            current: Some("b"),
            synced: Some("b"),
            stored_len: 4,
            prev_stored_len: len,
            chat_has_messages: true,
            busy: false,
            has_live: false,
            restored: false,
        });
        assert_eq!(action, super::SyncAction::Skip);
        // A conversation the send created is tracked with no rows.
        let (id, len) = super::busy_tracking(Some("new"), None, 0, 0);
        assert_eq!((id.as_deref(), len), (Some("new"), 0));
    }
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

    /// A stream that closed before Done: the partial answer is finished and
    /// marked, and the banner says so (an SSE error already shown is kept).
    #[test]
    fn finish_stream_ends_an_answer_cut_short() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.append_content("half");
        chat.finish_stream(false);
        let last = chat.messages.last().unwrap();
        assert!(!last.is_streaming);
        assert!(last.interrupted);
        assert_eq!(last.content, "half");
        assert_eq!(chat.error.as_deref(), Some(CLOSED_EARLY));
        // A second call changes nothing.
        let before = chat.messages.len();
        chat.set_error(None);
        chat.finish_stream(false);
        assert_eq!(chat.messages.len(), before);
        assert_eq!(chat.error, None);

        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.set_error(Some("Provider error (upstream_error)".into()));
        chat.finish_stream(false);
        assert_eq!(
            chat.error.as_deref(),
            Some("Provider error (upstream_error)")
        );
        assert!(chat.messages.last().unwrap().interrupted);
    }

    #[test]
    fn finish_stream_leaves_a_finished_or_stopped_answer() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.finalize_message(reply("All done."));
        chat.finish_stream(true);
        let last = chat.messages.last().unwrap();
        assert!(!last.interrupted);
        assert_eq!(last.content, "All done.");
        assert_eq!(chat.error, None);

        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.end_interrupted("Stopped.");
        chat.finish_stream(false);
        assert_eq!(chat.error, None, "a stop is not an error");
        assert_eq!(chat.messages.last().unwrap().content, "Stopped.");
    }

    #[test]
    fn end_interrupted_marks_the_message() {
        let mut chat = ChatState::new();
        send(&mut chat, "describe sales");
        chat.append_content("The data has");
        chat.end_interrupted("Stopped.");
        assert!(chat.messages.last().unwrap().interrupted);
        assert!(!chat.messages[0].interrupted);
    }

    fn contents(chat: &ChatState) -> Vec<(String, String)> {
        chat.messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect()
    }

    #[test]
    fn take_retry_turn_replaces_an_interrupted_turn() {
        let mut chat = ChatState::new();
        send(&mut chat, "a");
        chat.append_content("half");
        chat.finish_stream(false);
        chat.stop_processing();
        assert_eq!(chat.take_retry_turn().as_deref(), Some("a"));
        assert!(chat.messages.is_empty());

        // An empty placeholder is replaced the same way.
        let mut chat = ChatState::new();
        send(&mut chat, "a");
        chat.end_interrupted("");
        chat.stop_processing();
        assert_eq!(chat.take_retry_turn().as_deref(), Some("a"));
        assert!(chat.messages.is_empty());
    }

    #[test]
    fn take_retry_turn_reasks_after_a_finished_answer() {
        let mut chat = ChatState::new();
        send(&mut chat, "a");
        finish(&mut chat, reply("done"));
        assert_eq!(chat.take_retry_turn().as_deref(), Some("a"));
        assert_eq!(
            contents(&chat),
            pairs(&[("user", "a"), ("assistant", "done")])
        );
    }

    #[test]
    fn take_retry_turn_uses_the_prompt_of_an_ended_session() {
        let mut chat = ChatState::new();
        chat.retry_prompt = Some("regress price on sqft".into());
        assert_eq!(
            chat.take_retry_turn().as_deref(),
            Some("regress price on sqft")
        );
        assert_eq!(chat.retry_prompt, None);
        // A stale one is dropped when the transcript has a prompt.
        let mut chat = ChatState::new();
        send(&mut chat, "a");
        finish(&mut chat, reply("done"));
        chat.retry_prompt = Some("old".into());
        assert_eq!(chat.take_retry_turn().as_deref(), Some("a"));
        assert_eq!(chat.retry_prompt, None);
    }

    #[test]
    fn retry_after_a_send_that_had_no_session_sends_that_prompt() {
        let mut chat = ChatState::new();
        send(&mut chat, "a");
        finish(&mut chat, reply("done"));
        assert!(chat.try_start_processing());
        chat.fail_before_turn("b", "Session error: HTTP error: 503".into());
        assert!(!chat.is_processing);
        assert_eq!(
            chat.error.as_deref(),
            Some("Session error: HTTP error: 503")
        );
        // Retry sends the prompt that failed, not the one before it, and
        // the turn it replaces is taken out.
        assert_eq!(chat.take_retry_turn().as_deref(), Some("b"));
        assert_eq!(
            contents(&chat),
            pairs(&[("user", "a"), ("assistant", "done")])
        );
    }

    /// The tools of an interrupted answer ran on the server: their calls go
    /// back to the model, its partial text does not.
    #[test]
    fn build_history_keeps_the_tools_of_an_interrupted_answer() {
        let mut chat = ChatState::new();
        send(&mut chat, "load and describe");
        chat.set_active_tool("load_dataset".into(), json!({"name": "sales"}));
        chat.clear_active_tool(Some("Loaded sales".into()), Some("success"));
        chat.append_content("The data has");
        chat.finish_stream(false);
        chat.stop_processing();
        let history = send(&mut chat, "describe sales");
        assert_eq!(
            turns(&history),
            pairs(&[
                ("user", "load and describe"),
                ("assistant", ""),
                ("tool", "")
            ])
        );
        assert_eq!(
            history[2].tool_results.as_ref().unwrap()[0].content,
            "Loaded sales"
        );

        let mut chat = ChatState::new();
        send(&mut chat, "load and describe");
        chat.append_content("The data has");
        chat.finish_stream(false);
        chat.stop_processing();
        let history = send(&mut chat, "describe sales");
        assert_eq!(turns(&history), pairs(&[("user", "load and describe")]));
    }

    fn input<'a>(current: Option<&'a str>, synced: Option<&'a str>) -> SyncInput<'a> {
        SyncInput {
            current,
            synced,
            stored_len: 0,
            prev_stored_len: 0,
            chat_has_messages: false,
            busy: false,
            has_live: false,
            restored: false,
        }
    }

    #[test]
    fn decide_sync_table() {
        use SyncAction::*;
        let a = Some("a");
        let b = Some("b");
        // An answer streams: nothing is touched.
        let i = SyncInput {
            busy: true,
            has_live: true,
            ..input(b, a)
        };
        assert_eq!(decide_sync(&i), Skip);
        // Same conversation, no stored rows came.
        assert_eq!(decide_sync(&input(a, a)), Skip);
        let i = SyncInput {
            stored_len: 4,
            prev_stored_len: 4,
            ..input(a, a)
        };
        assert_eq!(decide_sync(&i), Skip, "rows already synced");
        // Stored rows arrived: rebuilt, unless the view was restored live.
        let i = SyncInput {
            stored_len: 4,
            ..input(a, a)
        };
        assert_eq!(decide_sync(&i), Rebuild);
        let i = SyncInput {
            restored: true,
            ..i
        };
        assert_eq!(decide_sync(&i), Skip);
        // A switch to a conversation with a live transcript.
        let i = SyncInput {
            has_live: true,
            stored_len: 4,
            ..input(b, a)
        };
        assert_eq!(decide_sync(&i), Restore);
        // A switch without one: rebuilt from rows already there, else wait.
        let i = SyncInput {
            stored_len: 4,
            ..input(b, a)
        };
        assert_eq!(decide_sync(&i), Rebuild);
        let i = SyncInput {
            chat_has_messages: true,
            ..input(b, None)
        };
        assert_eq!(decide_sync(&i), Clear);
        assert_eq!(decide_sync(&input(b, a)), Track);
        // No conversation any more; none before and none now keeps the chat.
        assert_eq!(decide_sync(&input(None, a)), Clear);
        let i = SyncInput {
            chat_has_messages: true,
            ..input(None, None)
        };
        assert_eq!(decide_sync(&i), Skip);
    }

    #[test]
    fn stash_and_restore_move_the_transcript() {
        let mut chat = ChatState::new();
        send(&mut chat, "plot sales");
        chat.add_image("iVBORw0KGgo=");
        chat.set_active_tool("plot".into(), json!({"dataset": "sales"}));
        chat.clear_active_tool(Some("ok".into()), Some("success"));
        finish(&mut chat, reply("Here it is."));
        let ptr = chat.messages.as_ptr();
        let before: Vec<(String, String)> = contents(&chat);

        chat.stash_live("a");
        assert!(chat.messages.is_empty());
        assert!(chat.has_live("a"));
        assert!(!chat.has_live("b"));
        assert!(chat.restore_live("a"));
        assert_eq!(chat.messages.as_ptr(), ptr, "moved, not cloned");
        assert_eq!(contents(&chat), before);
        assert_eq!(chat.messages[1].images, vec!["iVBORw0KGgo=".to_string()]);
        assert_eq!(chat.messages[1].tool_calls.len(), 1);
        assert!(!chat.has_live("a"), "restored entries leave the cache");
        assert!(!chat.restore_live("a"));
    }

    #[test]
    fn the_live_cache_keeps_the_most_recent_conversations() {
        let mut chat = ChatState::new();
        for i in 0..=LIVE_CACHE_CONVERSATIONS {
            chat.add_user_message(&format!("q{i}"));
            chat.stash_live(&format!("c{i}"));
        }
        assert!(!chat.has_live("c0"), "the least recently stashed went");
        for i in 1..=LIVE_CACHE_CONVERSATIONS {
            assert!(chat.has_live(&format!("c{i}")), "c{i}");
        }
        chat.forget_live("c3");
        assert!(!chat.has_live("c3"));
        chat.clear_live();
        assert!(!chat.has_live("c4"));
    }

    /// A second chip click while the first send runs is refused.
    #[test]
    fn try_start_processing_admits_one_send() {
        let mut chat = ChatState::new();
        chat.set_error(Some("old".into()));
        assert!(chat.try_start_processing());
        assert!(chat.is_processing);
        assert_eq!(chat.error, None, "starting clears the error");
        assert!(!chat.try_start_processing(), "busy: refused");
        chat.stop_processing();
        assert!(chat.try_start_processing());
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

    /// 3 warm-ups, then 31 interleaved runs; medians in ms.
    fn time_pair(mut base: impl FnMut() -> usize, mut head: impl FnMut() -> usize) -> (f64, f64) {
        for _ in 0..3 {
            std::hint::black_box(base());
            std::hint::black_box(head());
        }
        let (mut b, mut h) = (Vec::new(), Vec::new());
        for _ in 0..31 {
            let t = Instant::now();
            std::hint::black_box(base());
            b.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(head());
            h.push(t.elapsed().as_secs_f64() * 1e3);
        }
        (median(b), median(h))
    }

    fn stored_rows(n: usize) -> Vec<ConversationMessage> {
        (0..n)
            .map(|i| ConversationMessage {
                id: format!("m{i}"),
                conversation_id: "a".into(),
                role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                content: format!("message {i} ").repeat(20),
                created_at: "2026-11-18T10:00:00Z".into(),
                token_count: None,
                model: None,
            })
            .collect()
    }

    #[test]
    #[ignore = "benchmark; times a switch away and back: base rebuild against the live cache"]
    fn time_conversation_switch() {
        for n in [100, 1_000, 10_000] {
            let rows = stored_rows(n);
            let mut base = ChatState::new();
            base.messages = rows
                .iter()
                .map(ChatMessage::from_conversation_message)
                .collect();
            let mut head = base.clone();
            let (mb, mh) = time_pair(
                || {
                    // base: clear and rebuild from the stored rows
                    base.clear_messages();
                    base.messages
                        .extend(rows.iter().map(ChatMessage::from_conversation_message));
                    base.messages.len()
                },
                || {
                    head.stash_live("a");
                    head.restore_live("a");
                    head.messages.len()
                },
            );
            println!(
                "TIMING switch away and back {n} messages: base {mb:.4} ms, head {mh:.4} ms, head/base {:.3}",
                mh / mb
            );
        }
    }
}
