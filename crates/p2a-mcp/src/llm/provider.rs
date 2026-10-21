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

impl ProviderType {
    /// The provider's built-in endpoint, used when a config has no
    /// `base_url`.
    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderType::Ollama => "http://localhost:11434",
            ProviderType::Anthropic => "https://api.anthropic.com",
            ProviderType::OpenAI => "https://api.openai.com/v1",
        }
    }
}

/// Validate a user-supplied provider `base_url` before the server issues any
/// request to it.
///
/// The `base_url` field arrives from the untrusted request body, and the
/// server would otherwise make a server-side request to it carrying the
/// caller's bearer token: a classic SSRF sink (cloud metadata endpoints,
/// internal admin panels, loopback services). The check is
/// [`EgressPolicy::check_base_url`](super::egress::EgressPolicy::check_base_url)
/// under the server's current policy:
///
/// - If a host allowlist applies (`P2A_LLM_ALLOWED_HOSTS`, or the hosted
///   profile's default provider list), the host must be on it.
/// - The scheme must be `https`, except `http` to a loopback host for
///   [`ProviderType::Ollama`] when the profile allows loopback.
/// - IP-literal hosts in private, loopback, link-local, CGNAT, reserved or
///   documentation ranges, including the IPv4 address embedded in an IPv6
///   literal, are rejected (again except loopback for Ollama).
///
/// Host names are checked again when they are resolved: the LLM HTTP client
/// refuses a name that resolves to any such address.
pub fn validate_base_url(provider_type: ProviderType, base_url: &str) -> Result<(), String> {
    super::egress::EgressPolicy::current()
        .check_base_url(provider_type, base_url)
        .map_err(|e| e.0)
}

/// Validate the endpoint a provider config will use: its `base_url`, or the
/// provider's built-in default when none is set (the default Ollama endpoint
/// is checked like any other).
pub fn validate_provider_config(config: &ProviderConfig) -> Result<(), String> {
    super::egress::EgressPolicy::current()
        .check_provider_config(config)
        .map_err(|e| e.0)
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

    /// The egress policy refused the provider host.
    #[error("{0}")]
    EgressDenied(String),

    /// The provider answered with a non-success status. The body is never
    /// part of the error.
    #[error("{provider} returned HTTP {status}; {}", upstream_hint(*.status))]
    Upstream { provider: &'static str, status: u16 },

    /// The provider did not answer within `llm_timeout_secs`.
    #[error("The LLM provider did not respond in time")]
    Timeout,

    /// The provider's response body exceeded the size cap.
    #[error("The LLM provider's response exceeded {0} bytes")]
    ResponseTooLarge(usize),
}

/// What a caller can do about an upstream status.
fn upstream_hint(status: u16) -> &'static str {
    match status {
        300..=399 => {
            "the provider redirected the request, and redirects are not followed; set base_url \
             to the final endpoint"
        }
        400 => {
            "the provider rejected the request; the conversation or the dataset context may be \
             too long for this model"
        }
        401 | 403 => "check the API key",
        404 => "check the model name and base_url",
        408 => "the provider timed out; try again",
        413 => "the request is too large for this provider",
        422 => "the provider rejected the request parameters",
        429 => "the provider is rate limiting requests; try again later",
        500..=599 => "the provider is unavailable; try again later",
        _ => "unexpected response from the provider",
    }
}

impl From<reqwest::Error> for LlmError {
    /// An egress refusal anywhere in the source chain, a timeout after
    /// connecting, or a network error without the URL.
    fn from(err: reqwest::Error) -> Self {
        if let Some(denied) = super::egress::find_egress_error(&err) {
            return LlmError::EgressDenied(denied.0.clone());
        }
        if err.is_timeout() && !err.is_connect() {
            return LlmError::Timeout;
        }
        LlmError::NetworkError(err.without_url().to_string())
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
        assert!(validate_base_url(ProviderType::Anthropic, "https://api.anthropic.com").is_ok());
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
        // IPv4-mapped IPv6 and CGNAT literals.
        assert!(validate_base_url(ProviderType::OpenAI, "https://[::ffff:127.0.0.1]/").is_err());
        assert!(validate_base_url(ProviderType::OpenAI, "https://100.64.0.1/").is_err());
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

    #[test]
    fn upstream_error_display() {
        use super::LlmError;
        assert_eq!(
            LlmError::Upstream {
                provider: "OpenAI",
                status: 401
            }
            .to_string(),
            "OpenAI returned HTTP 401; check the API key"
        );
        assert_eq!(
            LlmError::Upstream {
                provider: "Anthropic",
                status: 400
            }
            .to_string(),
            "Anthropic returned HTTP 400; the provider rejected the request; the conversation \
             or the dataset context may be too long for this model"
        );
    }
}
