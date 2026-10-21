//! LLM integration for the MCP server.
//!
//! Provides multi-provider LLM support (Ollama, Anthropic, OpenAI) with
//! server-side tool execution for HTTP/WebSocket clients.

mod anthropic;
mod context;
pub mod egress;
mod ollama;
mod openai;
mod provider;
mod retry;
mod tokens;
mod tools;

use std::sync::Arc;
use std::time::Duration;

use crate::tools::limits::{self, Limits};

/// The LLM budgets of the active resource profile, read in one place.
///
/// Built from WP02's limits catalogue (`llm_*` keys, overridable with
/// `P2A_LIMIT_OVERRIDES`) and the egress policy.
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// `llm_timeout_secs`: total time for a non-streaming provider call, and
    /// the idle time allowed between reads of a streaming one.
    pub timeout: Duration,
    /// `llm_max_tool_iterations`: the clamp for a request's
    /// `max_tool_iterations`.
    pub max_tool_iterations: usize,
    /// `llm_context_bytes`: the dataset block in the system prompt.
    pub context_bytes: usize,
    /// `llm_history_bytes`: the conversation history sent to the model, and
    /// the tool results kept in one tool loop.
    pub history_bytes: usize,
    /// `llm_streams_per_session`: concurrent chat requests per session
    /// (`None` = unlimited).
    pub streams_per_session: Option<usize>,
    /// `llm_server_keys`: whether the server's own provider keys may be used.
    pub server_keys: bool,
    /// Which provider endpoints the server connects to.
    pub egress: Arc<egress::EgressPolicy>,
}

impl LlmSettings {
    /// Settings from `l` and the raw `P2A_LLM_ALLOWED_HOSTS` value.
    pub fn from_limits(l: &Limits, env_allowlist: Option<&str>) -> Self {
        Self {
            timeout: Duration::from_secs(l.llm_timeout_secs),
            max_tool_iterations: l.llm_max_tool_iterations,
            context_bytes: usize::try_from(l.llm_context_bytes).unwrap_or(usize::MAX),
            history_bytes: usize::try_from(l.llm_history_bytes).unwrap_or(usize::MAX),
            streams_per_session: l.llm_streams_per_session,
            server_keys: l.llm_server_keys,
            egress: Arc::new(egress::EgressPolicy::from_parts(env_allowlist, l)),
        }
    }

    /// Settings from [`limits::current`] and the environment.
    pub fn current() -> Self {
        Self::from_limits(&limits::current(), egress::allowlist_from_env().as_deref())
    }
}

/// Build the `reqwest::Client` for one provider.
///
/// Connections go through [`egress::GuardedResolver`] (a host that resolves
/// to any internal address is refused; loopback is admitted only for Ollama
/// when the profile allows it), redirects are never followed and system
/// proxies are ignored. The connect timeout is 10 s. `llm_timeout_secs`
/// bounds each non-streaming call in total (set per request by the
/// providers) and each read of a streaming body (the idle timeout set here);
/// a timed-out call is never retried. The bound is configurable with
/// `P2A_LIMIT_OVERRIDES=llm_timeout_secs=<n>` (hosted 120 s, off 900 s).
///
/// Returns `None`, after logging, when the client cannot be built. There is
/// no fallback to a default client, which would drop every guard.
pub(crate) fn build_http_client(
    provider: ProviderType,
    settings: &LlmSettings,
) -> Option<reqwest::Client> {
    let loopback_ok = provider == ProviderType::Ollama && settings.egress.allow_loopback;
    let resolver = egress::GuardedResolver::new(settings.egress.clone(), loopback_ok);
    match egress::guarded_client_builder(resolver)
        .read_timeout(settings.timeout)
        .build()
    {
        Ok(client) => Some(client),
        Err(e) => {
            tracing::error!(%provider, error = %e, "Failed to build the LLM HTTP client");
            None
        }
    }
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
    ToolCall, ToolDefinition, ToolExecutor, ToolResult, resolve_provider_config,
    server_key_availability, validate_base_url, validate_provider_config,
};
pub use retry::{RetryConfig, send_with_retry};
pub use tokens::{
    ContextBudget, estimate_message_tokens, estimate_tokens, estimate_tool_definition_tokens,
    get_model_context_size,
};
pub use tools::{INTERNAL_TOOLS, TIER1_TOOLS, get_system_prompt_with_context};

#[cfg(test)]
mod tests {
    use super::egress::EgressPolicy;
    use super::*;

    #[test]
    fn settings_from_profiles() {
        let off = LlmSettings::from_limits(&Limits::off(), None);
        assert_eq!(
            (
                off.timeout,
                off.max_tool_iterations,
                off.context_bytes,
                off.history_bytes,
                off.streams_per_session,
                off.server_keys
            ),
            (Duration::from_secs(900), 50, 65_536, 524_288, None, true)
        );
        assert_eq!(
            *off.egress,
            EgressPolicy {
                allowed_hosts: None,
                allow_loopback: true
            }
        );
        let h = LlmSettings::from_limits(&Limits::hosted(), None);
        assert_eq!(
            (
                h.timeout,
                h.max_tool_iterations,
                h.context_bytes,
                h.history_bytes,
                h.streams_per_session,
                h.server_keys
            ),
            (
                Duration::from_secs(120),
                25,
                24_576,
                131_072,
                Some(1),
                false
            )
        );
        assert!(!h.egress.allow_loopback && h.egress.allowed_hosts.is_some());
    }
}
