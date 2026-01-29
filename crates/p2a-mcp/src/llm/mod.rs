//! LLM integration for the MCP server.
//!
//! Provides multi-provider LLM support (Ollama, Anthropic, OpenAI) with
//! server-side tool execution for HTTP/WebSocket clients.

mod ollama;
mod openai;
mod provider;
mod tools;

pub use ollama::OllamaProvider;
pub use openai::OpenAIProvider;
pub use provider::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
pub use tools::{get_mcp_tool_definitions, get_system_prompt_with_context};
