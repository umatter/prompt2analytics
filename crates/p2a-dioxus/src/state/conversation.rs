//! Conversation state management
//!
//! Manages the list of conversations and the currently active conversation.

use crate::api::{Conversation, ConversationMessage, api};

/// Why the current conversation cannot be switched now, if it cannot: an
/// answer streams into it, or a conversation call runs.
pub fn switch_blocked(chat: &super::ChatState, conv: &ConversationState) -> Option<&'static str> {
    if chat.is_busy() {
        Some("An answer is being written. Stop it first to switch conversations.")
    } else if conv.is_operating {
        Some("A conversation change is still running; try again in a moment.")
    } else {
        None
    }
}

/// Whether `conversation_id` still has the default title (and gets a
/// generated one after its first turn).
pub fn needs_title(conversations: &[Conversation], conversation_id: &str) -> bool {
    conversations
        .iter()
        .find(|c| c.id == conversation_id)
        .is_some_and(|c| c.title == "New Conversation")
}

/// Conversation state for managing conversation history
#[derive(Clone)]
pub struct ConversationState {
    /// List of all conversations for the current session
    pub conversations: Vec<Conversation>,
    /// Currently selected conversation ID
    pub current_conversation_id: Option<String>,
    /// Whether conversations are being loaded
    pub is_loading: bool,
    /// Whether a conversation operation is in progress
    pub is_operating: bool,
    /// Error message if operation failed
    pub error: Option<String>,
    /// Messages for the current conversation (cached)
    pub current_messages: Vec<ConversationMessage>,
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationState {
    /// Create a new conversation state
    pub fn new() -> Self {
        Self {
            conversations: Vec::new(),
            current_conversation_id: None,
            is_loading: false,
            is_operating: false,
            error: None,
            current_messages: Vec::new(),
        }
    }

    /// Load all conversations for a session
    pub async fn load_conversations(&self, session_id: &str) -> Result<Vec<Conversation>, String> {
        let client = api();
        match client.list_conversations(session_id).await {
            Ok(conversations) => {
                tracing::info!("Loaded {} conversations", conversations.len());
                Ok(conversations)
            }
            Err(e) => {
                tracing::error!("Failed to load conversations: {}", e);
                Err(e)
            }
        }
    }

    /// Create a new conversation
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, String> {
        let client = api();
        match client.create_conversation(session_id, title).await {
            Ok(conversation) => {
                tracing::info!("Created conversation: {}", conversation.id);
                Ok(conversation)
            }
            Err(e) => {
                tracing::error!("Failed to create conversation: {}", e);
                Err(e)
            }
        }
    }

    /// Update a conversation's title
    pub async fn update_conversation_title(
        &self,
        conversation_id: &str,
        title: &str,
    ) -> Result<Conversation, String> {
        let client = api();
        match client
            .update_conversation(conversation_id, Some(title), None)
            .await
        {
            Ok(conversation) => {
                tracing::info!("Updated conversation title: {}", conversation_id);
                Ok(conversation)
            }
            Err(e) => {
                tracing::error!("Failed to update conversation: {}", e);
                Err(e)
            }
        }
    }

    /// Archive or unarchive a conversation
    pub async fn set_conversation_archived(
        &self,
        conversation_id: &str,
        is_archived: bool,
    ) -> Result<Conversation, String> {
        let client = api();
        match client
            .update_conversation(conversation_id, None, Some(is_archived))
            .await
        {
            Ok(conversation) => {
                tracing::info!(
                    "Set conversation {} archived: {}",
                    conversation_id,
                    is_archived
                );
                Ok(conversation)
            }
            Err(e) => {
                tracing::error!("Failed to update conversation: {}", e);
                Err(e)
            }
        }
    }

    /// Delete a conversation
    pub async fn delete_conversation(&self, conversation_id: &str) -> Result<(), String> {
        let client = api();
        match client.delete_conversation(conversation_id).await {
            Ok(()) => {
                tracing::info!("Deleted conversation: {}", conversation_id);
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to delete conversation: {}", e);
                Err(e)
            }
        }
    }

    /// Load messages for a conversation
    pub async fn load_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, String> {
        let client = api();
        match client.get_messages(conversation_id).await {
            Ok(messages) => {
                tracing::info!(
                    "Loaded {} messages for conversation {}",
                    messages.len(),
                    conversation_id
                );
                Ok(messages)
            }
            Err(e) => {
                tracing::error!("Failed to load messages: {}", e);
                Err(e)
            }
        }
    }

    /// Add a message to a conversation
    pub async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<ConversationMessage, String> {
        let client = api();
        match client.add_message(conversation_id, role, content).await {
            Ok(message) => {
                tracing::debug!("Added message to conversation {}", conversation_id);
                Ok(message)
            }
            Err(e) => {
                tracing::error!("Failed to add message: {}", e);
                Err(e)
            }
        }
    }

    /// Clear all messages in a conversation
    pub async fn clear_messages(&self, conversation_id: &str) -> Result<u32, String> {
        let client = api();
        match client.clear_messages(conversation_id).await {
            Ok(count) => {
                tracing::info!(
                    "Cleared {} messages from conversation {}",
                    count,
                    conversation_id
                );
                Ok(count)
            }
            Err(e) => {
                tracing::error!("Failed to clear messages: {}", e);
                Err(e)
            }
        }
    }

    // === State mutation methods (for use with signals) ===

    /// Set the list of conversations
    pub fn set_conversations(&mut self, conversations: Vec<Conversation>) {
        self.conversations = conversations;
        self.error = None;
    }

    /// Set the current conversation
    pub fn set_current_conversation(&mut self, conversation_id: Option<String>) {
        self.current_conversation_id = conversation_id;
        // Clear cached messages when switching conversations
        self.current_messages.clear();
    }

    /// Set the cached messages of `conversation_id`, if it is still the
    /// current conversation; a load that returns after another conversation
    /// was selected is dropped (returns false).
    pub fn set_current_messages(
        &mut self,
        conversation_id: &str,
        messages: Vec<ConversationMessage>,
    ) -> bool {
        if self.current_conversation_id.as_deref() != Some(conversation_id) {
            return false;
        }
        self.current_messages = messages;
        true
    }

    /// Add a conversation to the list
    pub fn add_conversation(&mut self, conversation: Conversation) {
        // Insert at the beginning (most recent first)
        self.conversations.insert(0, conversation);
    }

    /// Remove a conversation from the list
    pub fn remove_conversation(&mut self, conversation_id: &str) {
        self.conversations.retain(|c| c.id != conversation_id);

        // If the removed conversation was current, clear the selection
        if self.current_conversation_id.as_deref() == Some(conversation_id) {
            self.current_conversation_id = None;
            self.current_messages.clear();
        }
    }

    /// Update a conversation in the list
    pub fn update_conversation(&mut self, updated: Conversation) {
        if let Some(conv) = self.conversations.iter_mut().find(|c| c.id == updated.id) {
            *conv = updated;
        }
    }

    /// Get the current conversation
    pub fn get_current_conversation(&self) -> Option<&Conversation> {
        self.current_conversation_id
            .as_ref()
            .and_then(|id| self.conversations.iter().find(|c| &c.id == id))
    }

    /// Get non-archived conversations
    pub fn get_active_conversations(&self) -> Vec<&Conversation> {
        self.conversations
            .iter()
            .filter(|c| !c.is_archived)
            .collect()
    }

    /// Get archived conversations
    pub fn get_archived_conversations(&self) -> Vec<&Conversation> {
        self.conversations
            .iter()
            .filter(|c| c.is_archived)
            .collect()
    }

    /// Check if there are any conversations
    pub fn has_conversations(&self) -> bool {
        !self.conversations.is_empty()
    }

    /// Check if a conversation is selected
    pub fn has_current_conversation(&self) -> bool {
        self.current_conversation_id.is_some()
    }

    /// Set loading state
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }

    /// Set operating state (for create/update/delete operations)
    pub fn set_operating(&mut self, operating: bool) {
        self.is_operating = operating;
    }

    /// Set error state
    pub fn set_error(&mut self, error: Option<String>) {
        self.error = error;
        self.is_loading = false;
        self.is_operating = false;
    }

    /// Clear error state
    pub fn clear_error(&mut self) {
        self.error = None;
    }

    /// Add a message to the cached messages
    pub fn add_cached_message(&mut self, message: ConversationMessage) {
        self.current_messages.push(message);
    }

    /// Clear cached messages
    pub fn clear_cached_messages(&mut self) {
        self.current_messages.clear();
    }

    /// Forget every conversation (their session ended).
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ChatState;

    fn conversation(id: &str, title: &str) -> Conversation {
        Conversation {
            id: id.into(),
            session_id: "s".into(),
            title: title.into(),
            created_at: "2026-11-18T10:00:00Z".into(),
            updated_at: "2026-11-18T10:00:00Z".into(),
            is_archived: false,
            message_count: 0,
            last_message_preview: None,
        }
    }

    fn message(conversation_id: &str, content: &str) -> ConversationMessage {
        ConversationMessage {
            id: format!("m-{content}"),
            conversation_id: conversation_id.into(),
            role: "user".into(),
            content: content.into(),
            created_at: "2026-11-18T10:00:00Z".into(),
            token_count: None,
            model: None,
        }
    }

    /// A quick A-then-B selection: A's messages arrive after B became current and are dropped.
    #[test]
    fn stale_message_load_is_ignored() {
        let mut state = ConversationState::new();
        state.set_current_conversation(Some("B".into()));
        assert!(!state.set_current_messages("A", vec![message("A", "from A")]));
        assert!(state.current_messages.is_empty());
        assert!(state.set_current_messages("B", vec![message("B", "from B")]));
        assert_eq!(state.current_messages[0].content, "from B");
        // A load for a conversation that is no longer current does not replace B's.
        assert!(!state.set_current_messages("A", vec![message("A", "late")]));
        assert_eq!(state.current_messages[0].content, "from B");
    }

    #[test]
    fn switch_blocked_while_processing_or_operating() {
        let mut chat = ChatState::new();
        let mut conv = ConversationState::new();
        assert_eq!(switch_blocked(&chat, &conv), None);
        chat.start_processing();
        assert!(chat.is_busy());
        let why = switch_blocked(&chat, &conv).expect("blocked while streaming");
        assert!(why.contains("answer is being written"), "{why}");
        chat.stop_processing();
        conv.set_operating(true);
        let why = switch_blocked(&chat, &conv).expect("blocked while operating");
        assert!(why.contains("conversation"), "{why}");
    }

    /// The title goes to the conversation the turn ran in, not to the one current when it ends.
    #[test]
    fn title_target_is_the_turns_conversation() {
        let convs = vec![
            conversation("new", "New Conversation"),
            conversation("turn", "Sales analysis"),
        ];
        assert!(!needs_title(&convs, "turn"));
        assert!(needs_title(&convs, "new"));
        assert!(!needs_title(&convs, "gone"));
    }
}
