//! LLM Service - orchestrates LLM providers, tool execution, and history.
//!
//! This module provides the main service layer that:
//! - Manages the active LLM provider
//! - Wraps the MCP client for tool execution
//! - Handles conversation history persistence

use super::{
    get_mcp_tool_definitions, get_system_prompt_with_context, AnthropicProvider, Conversation,
    HistoryStore, LlmError, LlmProvider, Message, OllamaProvider, OpenAIProvider, ProviderConfig,
    ProviderType, StreamChunk, ToolExecutor,
};
use crate::mcp::McpClient;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tool executor that wraps the MCP client.
pub struct McpToolExecutor {
    mcp_client: Arc<McpClient>,
}

impl McpToolExecutor {
    pub fn new(mcp_client: Arc<McpClient>) -> Self {
        Self { mcp_client }
    }
}

#[async_trait]
impl ToolExecutor for McpToolExecutor {
    async fn execute(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        match self.mcp_client.call_tool(name, arguments).await {
            Ok(result) => {
                if result.success {
                    Ok(result.content)
                } else {
                    Err(result.error.unwrap_or_else(|| "Unknown error".to_string()))
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

/// The main LLM service that orchestrates providers, tools, and history.
pub struct LlmService {
    provider: RwLock<Option<Box<dyn LlmProvider>>>,
    config: RwLock<ProviderConfig>,
    history: Arc<HistoryStore>,
    tool_executor: Arc<McpToolExecutor>,
}

impl LlmService {
    /// Create a new LLM service.
    pub fn new(mcp_client: Arc<McpClient>, history: Arc<HistoryStore>) -> Self {
        let tool_executor = Arc::new(McpToolExecutor::new(mcp_client));

        Self {
            provider: RwLock::new(None),
            config: RwLock::new(ProviderConfig::default()),
            history,
            tool_executor,
        }
    }

    /// Initialize the service with saved configuration.
    pub async fn init(&self) -> Result<(), LlmError> {
        let config = self.history.get_provider_config()?;
        self.set_config(config).await?;
        Ok(())
    }

    /// Get the current provider configuration.
    pub async fn get_config(&self) -> ProviderConfig {
        self.config.read().await.clone()
    }

    /// Set the provider configuration and create the appropriate provider.
    pub async fn set_config(&self, config: ProviderConfig) -> Result<(), LlmError> {
        // Save to history
        self.history.set_provider_config(&config)?;

        // Create the appropriate provider
        let provider: Box<dyn LlmProvider> = match config.provider_type {
            ProviderType::Ollama => Box::new(OllamaProvider::new(config.clone())),
            ProviderType::Anthropic => Box::new(AnthropicProvider::new(config.clone())),
            ProviderType::OpenAI => Box::new(OpenAIProvider::new(config.clone())),
        };

        *self.config.write().await = config;
        *self.provider.write().await = Some(provider);

        Ok(())
    }

    /// Check if the current provider is available.
    pub async fn is_provider_available(&self) -> Result<bool, LlmError> {
        let provider = self.provider.read().await;
        match provider.as_ref() {
            Some(p) => p.is_available().await,
            None => Ok(false),
        }
    }

    /// List available models for the current provider.
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let provider = self.provider.read().await;
        match provider.as_ref() {
            Some(p) => p.list_models().await,
            None => Err(LlmError::NotAvailable("No provider configured".to_string())),
        }
    }

    /// Get the history store.
    pub fn history(&self) -> &HistoryStore {
        &self.history
    }

    /// Get current dataset context for the system prompt.
    ///
    /// Calls the list_datasets tool to get currently loaded datasets
    /// and formats them for inclusion in the system prompt.
    async fn get_dataset_context(&self) -> Option<String> {
        match self.tool_executor.execute("list_datasets", serde_json::json!({})).await {
            Ok(result) if !result.is_empty() && !result.contains("No datasets") => Some(result),
            _ => None,
        }
    }

    /// Send a message in a conversation and get a response.
    ///
    /// If `conversation_id` is None, creates a new conversation.
    /// Returns the conversation ID and the assistant's response message.
    pub async fn send_message(
        &self,
        conversation_id: Option<&str>,
        user_message: &str,
    ) -> Result<(String, Message), LlmError> {
        let provider = self.provider.read().await;
        let provider = provider
            .as_ref()
            .ok_or_else(|| LlmError::NotAvailable("No provider configured".to_string()))?;

        let config = self.config.read().await;

        // Get or create conversation
        let conv_id = match conversation_id {
            Some(id) => id.to_string(),
            None => {
                // Create new conversation with first few words as title
                let title = user_message
                    .chars()
                    .take(50)
                    .collect::<String>()
                    .split_whitespace()
                    .take(6)
                    .collect::<Vec<_>>()
                    .join(" ");
                let title = if title.len() < user_message.len() {
                    format!("{}...", title)
                } else {
                    title
                };

                let conv = self.history.create_conversation(
                    &title,
                    config.provider_type,
                    &config.model,
                )?;
                conv.id
            }
        };

        // Load existing messages for the conversation
        let stored_messages = self.history.get_messages(&conv_id)?;

        // Get dataset context for system prompt
        let dataset_context = self.get_dataset_context().await;
        let system_prompt = get_system_prompt_with_context(dataset_context.as_deref());
        let mut messages: Vec<Message> = vec![Message::system(system_prompt)];

        // Add stored messages
        for sm in &stored_messages {
            messages.push(sm.to_message());
        }

        // Add the new user message
        let user_msg = Message::user(user_message);
        messages.push(user_msg.clone());

        // Save user message to history
        self.history.add_message(&conv_id, &user_msg)?;

        // Get tool definitions
        let tools = get_mcp_tool_definitions();

        // Call the LLM with tool execution loop
        let response = provider
            .chat(&messages, &tools, self.tool_executor.as_ref())
            .await?;

        // Save assistant response to history
        self.history.add_message(&conv_id, &response)?;

        Ok((conv_id, response))
    }

    /// Send a message with streaming response.
    ///
    /// Returns the conversation ID, with streaming chunks delivered via callback.
    pub async fn send_message_stream(
        &self,
        conversation_id: Option<&str>,
        user_message: &str,
        callback: Box<dyn Fn(StreamChunk) + Send + Sync>,
    ) -> Result<(String, Message), LlmError> {
        let provider = self.provider.read().await;
        let provider = provider
            .as_ref()
            .ok_or_else(|| LlmError::NotAvailable("No provider configured".to_string()))?;

        let config = self.config.read().await;

        // Get or create conversation
        let conv_id = match conversation_id {
            Some(id) => id.to_string(),
            None => {
                let title = user_message
                    .chars()
                    .take(50)
                    .collect::<String>()
                    .split_whitespace()
                    .take(6)
                    .collect::<Vec<_>>()
                    .join(" ");
                let title = if title.len() < user_message.len() {
                    format!("{}...", title)
                } else {
                    title
                };

                let conv = self.history.create_conversation(
                    &title,
                    config.provider_type,
                    &config.model,
                )?;
                conv.id
            }
        };

        // Load existing messages
        let stored_messages = self.history.get_messages(&conv_id)?;

        // Get dataset context for system prompt
        let dataset_context = self.get_dataset_context().await;
        let system_prompt = get_system_prompt_with_context(dataset_context.as_deref());
        let mut messages: Vec<Message> = vec![Message::system(system_prompt)];

        for sm in &stored_messages {
            messages.push(sm.to_message());
        }

        let user_msg = Message::user(user_message);
        messages.push(user_msg.clone());

        // Save user message
        self.history.add_message(&conv_id, &user_msg)?;

        // Get tools
        let tools = get_mcp_tool_definitions();

        // Call with streaming
        let response = provider
            .chat_stream(&messages, &tools, self.tool_executor.as_ref(), callback)
            .await?;

        // Save response
        self.history.add_message(&conv_id, &response)?;

        Ok((conv_id, response))
    }

    /// Get all conversations.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>, LlmError> {
        self.history.list_conversations()
    }

    /// Get messages for a conversation.
    pub fn get_conversation_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<Message>, LlmError> {
        let stored = self.history.get_messages(conversation_id)?;
        Ok(stored.into_iter().map(|sm| sm.to_message()).collect())
    }

    /// Delete a conversation.
    pub fn delete_conversation(&self, conversation_id: &str) -> Result<(), LlmError> {
        self.history.delete_conversation(conversation_id)
    }

    /// Rename a conversation.
    pub fn rename_conversation(
        &self,
        conversation_id: &str,
        new_title: &str,
    ) -> Result<(), LlmError> {
        self.history.update_conversation_title(conversation_id, new_title)
    }

    /// Get a conversation by ID.
    pub fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>, LlmError> {
        self.history.get_conversation(conversation_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Mock MCP client for testing
    fn create_test_service() -> LlmService {
        let mcp_client = Arc::new(McpClient::new(PathBuf::from("/nonexistent")));
        let history = Arc::new(HistoryStore::in_memory().unwrap());
        LlmService::new(mcp_client, history)
    }

    #[tokio::test]
    async fn test_default_config() {
        let service = create_test_service();
        let config = service.get_config().await;
        assert_eq!(config.provider_type, ProviderType::Ollama);
    }

    #[tokio::test]
    async fn test_set_config() {
        let service = create_test_service();

        let mut config = ProviderConfig::default();
        config.provider_type = ProviderType::Anthropic;
        config.model = "claude-sonnet-4-20250514".to_string();
        config.api_key = Some("test-key".to_string());

        service.set_config(config.clone()).await.unwrap();

        let loaded = service.get_config().await;
        assert_eq!(loaded.provider_type, ProviderType::Anthropic);
        assert_eq!(loaded.model, "claude-sonnet-4-20250514");
    }
}
