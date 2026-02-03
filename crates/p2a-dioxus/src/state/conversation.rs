//! Conversation state management
//!
//! Manages the list of conversations and the currently active conversation.

use crate::api::{Conversation, ConversationMessage, api};

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

    /// Set cached messages for the current conversation
    pub fn set_current_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.current_messages = messages;
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
}
