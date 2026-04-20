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

/// Build a `reqwest::Client` with timeouts appropriate for LLM calls.
///
/// The connect timeout (10 s) fails fast on network errors; the overall
/// request timeout (5 min) accommodates long completions but ensures a
/// stalled endpoint cannot pin the SSE stream forever.
pub(crate) fn build_http_client(provider_name: &'static str) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to build {} HTTP client with timeouts ({}); falling back to default client",
                provider_name,
                e
            );
            reqwest::Client::new()
        })
}

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
pub use retry::{RetryConfig, send_with_retry};
pub use tokens::{
    ContextBudget, estimate_message_tokens, estimate_tokens, estimate_tool_definition_tokens,
    get_model_context_size,
};
pub use tools::{INTERNAL_TOOLS, TIER1_TOOLS, get_system_prompt_with_context};
