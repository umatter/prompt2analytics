//! LLM integration module for prompt2analytics desktop app.
//!
//! This module provides:
//! - Multi-provider LLM support (Ollama, Anthropic, OpenAI)
//! - Conversation history persistence
//! - Tool execution loop for MCP integration

mod anthropic;
mod history;
mod ollama;
mod openai;
mod provider;
mod service;
mod tools;

pub use anthropic::AnthropicProvider;
pub use history::{Conversation, HistoryStore, StoredMessage};
pub use ollama::OllamaProvider;
pub use openai::OpenAIProvider;
pub use provider::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
pub use service::LlmService;
pub use tools::{get_mcp_tool_definitions, get_system_prompt, get_system_prompt_with_context};
