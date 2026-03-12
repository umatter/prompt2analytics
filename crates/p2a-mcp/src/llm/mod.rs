//! LLM integration for the MCP server.
//!
//! Provides multi-provider LLM support (Ollama, Anthropic, OpenAI) with
//! server-side tool execution for HTTP/WebSocket clients.

mod anthropic;
mod context;
mod ollama;
mod openai;
mod provider;
mod retry;
mod tokens;
mod tools;

pub use anthropic::AnthropicProvider;
pub use context::{
    ContextConfig, ContextManager, ProcessedContext, build_enhanced_dataset_context,
    summarize_tool_call, truncate_tool_result,
};
pub use ollama::OllamaProvider;
pub use openai::OpenAIProvider;
pub use provider::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
pub use tokens::{
    ContextBudget, estimate_message_tokens, estimate_tokens, estimate_tool_definition_tokens,
    get_model_context_size,
};
pub use retry::{RetryConfig, send_with_retry};
pub use tools::{get_system_prompt_with_context, TIER1_TOOLS, INTERNAL_TOOLS};
