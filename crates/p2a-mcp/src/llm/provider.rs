//! LLM provider trait and common types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Default maximum number of tool execution iterations per request.
pub const DEFAULT_MAX_TOOL_ITERATIONS: usize = 25;

/// Configuration for an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Maximum number of tool execution iterations (default: 25)
    #[serde(default)]
    pub max_tool_iterations: Option<usize>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::Ollama,
            api_key: None,
            base_url: None,
            model: "llama3.2".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
            max_tool_iterations: None,
        }
    }
}

/// Supported LLM provider types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Ollama,
    Anthropic,
    OpenAI,
}

/// Validate a user-supplied provider `base_url` before the server issues any
/// request to it.
///
/// The `base_url` field arrives from the untrusted request body, and the
/// server would otherwise make a server-side request to it carrying the
/// caller's bearer token — a classic SSRF sink (cloud metadata endpoints,
/// internal admin panels, loopback services). This applies defence-in-depth:
///
/// - The scheme must be `https`. The one exception is Ollama, which is
///   commonly a local `http://localhost:11434`, so `http` to a loopback host
///   is permitted only for [`ProviderType::Ollama`].
/// - IP-literal hosts in private, loopback, link-local, or unspecified ranges
///   are rejected (again except loopback for Ollama). This blocks the
///   `http://169.254.169.254/` metadata vector and direct-to-internal-IP SSRF.
/// - If `P2A_LLM_ALLOWED_HOSTS` is set (comma-separated), the host must match
///   one of its entries — a hard allowlist operators can use to lock the proxy
///   to known provider hosts.
///
/// Returns `Ok(())` when `base_url` is `None` (the provider uses its built-in
/// default, which is trusted).
pub fn validate_base_url(provider_type: ProviderType, base_url: &str) -> Result<(), String> {
    use std::net::IpAddr;

    let url = reqwest::Url::parse(base_url).map_err(|e| format!("invalid base_url: {e}"))?;
    let host = url
        .host_str()
        .ok_or_else(|| "base_url has no host".to_string())?;

    // `host_str` serializes IPv6 literals with brackets (e.g. "[::1]"), which do
    // not parse as `IpAddr`; strip them for the IP-range check.
    let host_ip = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
        .parse::<IpAddr>()
        .ok();

    let is_ollama_loopback = provider_type == ProviderType::Ollama
        && (host.eq_ignore_ascii_case("localhost")
            || host_ip.map(|ip| ip.is_loopback()).unwrap_or(false));

    match url.scheme() {
        "https" => {}
        "http" if is_ollama_loopback => {}
        other => {
            return Err(format!(
                "base_url scheme `{other}` is not allowed; use https \
                 (http is permitted only for a local Ollama endpoint)"
            ));
        }
    }

    if let Some(ip) = host_ip
        && is_disallowed_ip(ip)
        && !is_ollama_loopback
    {
        return Err(format!(
            "base_url host {ip} is in a private, loopback, or link-local range \
             and is not allowed"
        ));
    }

    if let Ok(list) = std::env::var("P2A_LLM_ALLOWED_HOSTS") {
        let allowed: Vec<&str> = list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !allowed.is_empty() && !allowed.iter().any(|h| h.eq_ignore_ascii_case(host)) {
            return Err(format!(
                "base_url host `{host}` is not in the P2A_LLM_ALLOWED_HOSTS allowlist"
            ));
        }
    }

    Ok(())
}

/// True if an IP literal is one the server must never be pointed at from an
/// untrusted `base_url` (private, loopback, link-local, or unspecified).
fn is_disallowed_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_unspecified()
                || v4.octets()[0] == 0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // fc00::/7 unique-local
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // fe80::/10 link-local
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Validate the optional `base_url` on a provider config (no-op when unset).
pub fn validate_provider_config(config: &ProviderConfig) -> Result<(), String> {
    match &config.base_url {
        Some(url) if !url.trim().is_empty() => validate_base_url(config.provider_type, url),
        _ => Ok(()),
    }
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Ollama => write!(f, "ollama"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::OpenAI => write!(f, "openai"),
        }
    }
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_results: Option<Vec<ToolResult>>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
            tool_results: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
            tool_results: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_results: None,
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_results: None,
        }
    }

    pub fn tool_result(tool_results: Vec<ToolResult>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: String::new(),
            tool_calls: None,
            tool_results: Some(tool_results),
        }
    }
}

/// Role of a message in the conversation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
        }
    }
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Tool definition for LLM function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Streaming response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    Text { content: String },
    ToolCall { tool_call: ToolCall },
    ToolResult { tool_result: ToolResult },
    Done,
    Error { message: String },
}

/// Error type for LLM operations.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Provider not available: {0}")]
    NotAvailable(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),
}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        LlmError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        LlmError::SerializationError(err.to_string())
    }
}

/// Trait for executing tools (implemented by the server).
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, name: &str, arguments: serde_json::Value) -> Result<String, String>;
}

/// The main LLM provider trait.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Get the provider type.
    fn provider_type(&self) -> ProviderType;

    /// Check if the provider is available.
    async fn is_available(&self) -> Result<bool, LlmError>;

    /// List available models for this provider.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;

    /// Send a message and get a complete response (with tool execution loop).
    ///
    /// When `interpret` is true (default), the LLM will interpret and synthesize tool results.
    /// When `interpret` is false, tool results are returned directly without LLM interpretation.
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
    ) -> Result<Message, LlmError>;

    /// Send a message and stream the response.
    ///
    /// When `interpret` is true (default), the LLM will interpret and synthesize tool results.
    /// When `interpret` is false, tool results are returned directly without LLM interpretation.
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
        callback: Box<dyn Fn(StreamChunk) + Send + Sync>,
    ) -> Result<Message, LlmError>;
}

#[cfg(test)]
mod base_url_tests {
    use super::{ProviderType, validate_base_url};

    #[test]
    fn accepts_https_provider_hosts() {
        assert!(validate_base_url(ProviderType::OpenAI, "https://api.openai.com/v1").is_ok());
        assert!(
            validate_base_url(ProviderType::Anthropic, "https://api.anthropic.com").is_ok()
        );
    }

    #[test]
    fn rejects_plain_http_for_cloud_providers() {
        assert!(validate_base_url(ProviderType::OpenAI, "http://api.openai.com").is_err());
    }

    #[test]
    fn rejects_ssrf_targets() {
        // Cloud metadata endpoint and loopback/private ranges.
        assert!(validate_base_url(ProviderType::OpenAI, "http://169.254.169.254/").is_err());
        assert!(validate_base_url(ProviderType::OpenAI, "https://169.254.169.254/").is_err());
        assert!(validate_base_url(ProviderType::OpenAI, "https://127.0.0.1:8080").is_err());
        assert!(validate_base_url(ProviderType::OpenAI, "https://10.0.0.5").is_err());
        assert!(validate_base_url(ProviderType::OpenAI, "https://192.168.1.1").is_err());
        assert!(validate_base_url(ProviderType::Anthropic, "https://[::1]/").is_err());
    }

    #[test]
    fn allows_local_ollama_only() {
        assert!(validate_base_url(ProviderType::Ollama, "http://localhost:11434").is_ok());
        assert!(validate_base_url(ProviderType::Ollama, "http://127.0.0.1:11434").is_ok());
        // The same loopback endpoint is not acceptable for a cloud provider.
        assert!(validate_base_url(ProviderType::OpenAI, "http://localhost:11434").is_err());
    }

    #[test]
    fn rejects_malformed_url() {
        assert!(validate_base_url(ProviderType::OpenAI, "not a url").is_err());
    }
}
