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

/// The tool-loop iterations a chat request may run: the request's value,
/// else the provider config's, else [`DEFAULT_MAX_TOOL_ITERATIONS`], clamped
/// to `[1, cap]` (`llm_max_tool_iterations`).
pub fn effective_max_tool_iterations(
    request: Option<usize>,
    provider: Option<usize>,
    cap: usize,
) -> usize {
    request
        .or(provider)
        .unwrap_or(DEFAULT_MAX_TOOL_ITERATIONS)
        .clamp(1, cap.max(1))
}

/// The provider's name in messages.
fn provider_name(t: ProviderType) -> &'static str {
    match t {
        ProviderType::Ollama => "Ollama",
        ProviderType::Anthropic => "Anthropic",
        ProviderType::OpenAI => "OpenAI",
    }
}

/// The environment variable holding the server's own key for `t`.
fn server_key_var(t: ProviderType) -> Option<&'static str> {
    match t {
        ProviderType::OpenAI => Some("OPENAI_API_KEY"),
        ProviderType::Anthropic => Some("ANTHROPIC_API_KEY"),
        ProviderType::Ollama => None,
    }
}

/// Whether `base_url` (unset, or after trimming one trailing `/`) is the
/// provider's built-in endpoint.
fn is_default_endpoint(t: ProviderType, base_url: Option<&str>) -> bool {
    match base_url {
        None => true,
        Some(url) => url.strip_suffix('/').unwrap_or(url) == t.default_base_url(),
    }
}

/// The provider config a chat request runs with, or the message of a 400.
///
/// 1. A missing config is the default (local Ollama); a blank `base_url` or
///    `api_key` counts as unset.
/// 2. `max_tool_iterations` is the request's value, else the config's, else
///    the default, clamped to `settings.max_tool_iterations`.
/// 3. The effective endpoint must pass the egress policy.
/// 4. Without a caller key, OpenAI and Anthropic get the server's key only
///    when server keys are enabled and the endpoint is the provider's
///    built-in one; otherwise the caller must supply `api_key`. The server's
///    key is never sent to a host the caller picked. Ollama needs no key.
///
/// `env` reads an environment variable (a test seam for `std::env::var`).
pub fn resolve_provider_config(
    cfg: Option<ProviderConfig>,
    request_max_tool_iterations: Option<usize>,
    settings: &super::LlmSettings,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<ProviderConfig, String> {
    let mut cfg = cfg.unwrap_or_default();
    cfg.base_url = cfg
        .base_url
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty());
    cfg.api_key = cfg.api_key.filter(|k| !k.trim().is_empty());
    cfg.max_tool_iterations = Some(effective_max_tool_iterations(
        request_max_tool_iterations,
        cfg.max_tool_iterations,
        settings.max_tool_iterations,
    ));

    settings
        .egress
        .check_provider_config(&cfg)
        .map_err(|e| format!("Rejected LLM provider base_url: {e}"))?;

    if cfg.api_key.is_none()
        && let Some(var) = server_key_var(cfg.provider_type)
    {
        let name = provider_name(cfg.provider_type);
        if !settings.server_keys {
            return Err(format!(
                "Provide an API key for {name} in the provider settings (api_key); this server \
                 does not supply one."
            ));
        }
        if !is_default_endpoint(cfg.provider_type, cfg.base_url.as_deref()) {
            return Err(format!(
                "Provide an API key for {name} (api_key): this server's own key is used only \
                 with the default {name} endpoint."
            ));
        }
        match env(var).filter(|k| !k.trim().is_empty()) {
            Some(key) => {
                tracing::info!(
                    provider = %cfg.provider_type,
                    "Using the server's API key for the provider's default endpoint"
                );
                cfg.api_key = Some(key);
            }
            None => {
                return Err(format!(
                    "Provide an API key for {name} in the provider settings (api_key)."
                ));
            }
        }
    }
    Ok(cfg)
}

/// Whether the server would supply its own key for (OpenAI, Anthropic):
/// server keys are enabled and the environment holds a non-empty key.
pub fn server_key_availability(
    settings: &super::LlmSettings,
    env: &dyn Fn(&str) -> Option<String>,
) -> (bool, bool) {
    let has = |var: &str| settings.server_keys && env(var).is_some_and(|k| !k.trim().is_empty());
    (has("OPENAI_API_KEY"), has("ANTHROPIC_API_KEY"))
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

impl ToolCall {
    /// The arguments of a tool call that a provider sends as text (OpenAI's
    /// `function.arguments`, Anthropic's streamed input): the JSON value the
    /// text holds, or the empty object when it is empty or blank (a call
    /// without parameters). Text that is not JSON, such as a reply cut off
    /// at its token limit, stays a string: the tool call path refuses it as
    /// not a JSON object, so an unfinished call never runs with the tool's
    /// defaults.
    pub(crate) fn parse_arguments(raw: &str) -> serde_json::Value {
        if raw.trim().is_empty() {
            return serde_json::Value::Object(serde_json::Map::new());
        }
        serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
    }

    /// The arguments for a provider API that takes them as a JSON object
    /// when the call is sent back to it (Anthropic's `tool_use.input`,
    /// Ollama's `function.arguments`): the arguments when they are an object,
    /// the empty object otherwise. Arguments that are not an object (text
    /// that was not JSON, a positional array) belong to a call the tool call
    /// path refused for that, and its tool result says so; `null` is
    /// arguments left out, which that path takes as the empty object. Sent as
    /// they are, they would make the API refuse the whole request, in this
    /// turn's next iteration and in every later turn rebuilt from the stored
    /// history, so the model would never read the tool result.
    pub(crate) fn arguments_object(&self) -> serde_json::Value {
        match &self.arguments {
            serde_json::Value::Object(_) => self.arguments.clone(),
            _ => serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// The arguments for a provider API that takes them as text when the
    /// call is sent back to it (OpenAI's `function.arguments`): the text the
    /// model sent when it was not JSON (kept as a string by
    /// [`parse_arguments`](Self::parse_arguments)), the JSON text of any other
    /// value.
    pub(crate) fn arguments_text(&self) -> String {
        match &self.arguments {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        }
    }
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
pub(crate) mod tool_call_tests {
    use super::{Message, MessageRole, ToolCall};
    use serde_json::{Value, json};

    /// The text of a tool call cut off at the model's token limit.
    pub(crate) const CUT: &str = r#"{"lag_max": 5"#;

    /// An assistant message with four calls of `timeseries_arma_acf` (ids
    /// `c0` to `c3`): arguments by name, the text of a cut-off call (which
    /// [`ToolCall::parse_arguments`] keeps as a string) and a positional
    /// array, which the tool call path refuses, and `null`, which it takes as
    /// arguments left out.
    pub(crate) fn calls_with_every_argument_shape() -> Message {
        let arguments = [json!({"lag_max": 5}), json!(CUT), json!([5]), Value::Null];
        Message {
            role: MessageRole::Assistant,
            content: String::new(),
            tool_calls: Some(
                arguments
                    .into_iter()
                    .enumerate()
                    .map(|(i, arguments)| ToolCall {
                        id: format!("c{i}"),
                        name: "timeseries_arma_acf".into(),
                        arguments,
                    })
                    .collect(),
            ),
            tool_results: None,
        }
    }

    #[test]
    fn arguments_are_sent_back_in_the_shape_each_api_takes() {
        let calls = calls_with_every_argument_shape().tool_calls.unwrap();
        let objects: Vec<Value> = calls.iter().map(ToolCall::arguments_object).collect();
        assert_eq!(
            objects,
            [json!({"lag_max": 5}), json!({}), json!({}), json!({})]
        );
        let texts: Vec<String> = calls.iter().map(ToolCall::arguments_text).collect();
        assert_eq!(texts, [r#"{"lag_max":5}"#, CUT, "[5]", "null"]);
    }

    #[test]
    fn argument_text_that_is_not_json_stays_a_string() {
        // Empty or blank: a call without parameters.
        for blank in ["", " ", "\n\t "] {
            assert_eq!(ToolCall::parse_arguments(blank), json!({}), "{blank:?}");
        }
        assert_eq!(
            ToolCall::parse_arguments(r#"{"lag_max": 5}"#),
            json!({"lag_max": 5})
        );
        // Not JSON (cut off, or prose): kept as the text, never `null`.
        for text in [r#"{"lag_max": 5"#, "lag_max = 5", "{"] {
            assert_eq!(
                ToolCall::parse_arguments(text),
                Value::String(text.to_string())
            );
        }
        // JSON that is not an object is the value; the tool call path refuses it.
        assert_eq!(ToolCall::parse_arguments("[1, 2]"), json!([1, 2]));
        assert_eq!(ToolCall::parse_arguments("null"), Value::Null);
    }
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

#[cfg(test)]
mod key_policy_tests {
    use super::*;
    use crate::llm::LlmSettings;
    use crate::tools::limits::Limits;

    fn env(k: &str) -> Option<String> {
        match k {
            "OPENAI_API_KEY" => Some("SENTINEL_OPENAI".into()),
            "ANTHROPIC_API_KEY" => Some("SENTINEL_ANTHROPIC".into()),
            _ => None,
        }
    }

    fn cfg(t: ProviderType, url: Option<&str>, key: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            provider_type: t,
            base_url: url.map(Into::into),
            api_key: key.map(Into::into),
            model: "m".into(),
            ..ProviderConfig::default()
        }
    }

    #[test]
    fn server_key_only_for_default_endpoint() {
        let off = LlmSettings::from_limits(&Limits::off(), None);
        let c = resolve_provider_config(
            Some(cfg(ProviderType::OpenAI, None, None)),
            None,
            &off,
            &env,
        )
        .unwrap();
        assert_eq!(c.api_key.as_deref(), Some("SENTINEL_OPENAI"));
        let c = resolve_provider_config(
            Some(cfg(
                ProviderType::OpenAI,
                Some("https://api.openai.com/v1/"),
                None,
            )),
            None,
            &off,
            &env,
        )
        .unwrap();
        assert_eq!(c.api_key.as_deref(), Some("SENTINEL_OPENAI"));
        let c = resolve_provider_config(
            Some(cfg(
                ProviderType::Anthropic,
                Some("https://api.anthropic.com"),
                Some(""),
            )),
            None,
            &off,
            &env,
        )
        .unwrap();
        assert_eq!(c.api_key.as_deref(), Some("SENTINEL_ANTHROPIC"));
        for (t, url) in [
            (ProviderType::OpenAI, "https://openrouter.ai/api/v1"),
            (ProviderType::Anthropic, "https://llm.example.com"),
        ] {
            let e = resolve_provider_config(Some(cfg(t, Some(url), None)), None, &off, &env)
                .unwrap_err();
            assert!(e.contains("api_key") && !e.contains("SENTINEL"), "{e}");
        }
        let c = resolve_provider_config(
            Some(cfg(
                ProviderType::OpenAI,
                Some("https://openrouter.ai/api/v1"),
                Some("caller"),
            )),
            None,
            &off,
            &env,
        )
        .unwrap();
        assert_eq!(c.api_key.as_deref(), Some("caller"));
        // A blank base_url means the default endpoint.
        let c = resolve_provider_config(
            Some(cfg(ProviderType::OpenAI, Some("  "), None)),
            None,
            &off,
            &env,
        )
        .unwrap();
        assert_eq!(
            (c.base_url, c.api_key.as_deref()),
            (None, Some("SENTINEL_OPENAI"))
        );
    }

    #[test]
    fn max_tool_iterations_are_clamped() {
        assert_eq!(effective_max_tool_iterations(Some(1000), None, 25), 25);
        assert_eq!(effective_max_tool_iterations(Some(0), None, 25), 1);
        assert_eq!(effective_max_tool_iterations(None, Some(40), 50), 40);
        assert_eq!(effective_max_tool_iterations(None, None, 50), 25);
        // The request level wins.
        assert_eq!(effective_max_tool_iterations(Some(10), Some(40), 50), 10);
        let s = LlmSettings::from_limits(&Limits::hosted(), None);
        let c = resolve_provider_config(
            Some(ProviderConfig {
                provider_type: ProviderType::OpenAI,
                api_key: Some("k".into()),
                ..ProviderConfig::default()
            }),
            Some(1000),
            &s,
            &|_| None,
        )
        .unwrap();
        assert_eq!(c.max_tool_iterations, Some(25));
    }

    #[test]
    fn hosted_profile_never_uses_server_keys() {
        let h = LlmSettings::from_limits(&Limits::hosted(), None);
        let e =
            resolve_provider_config(Some(cfg(ProviderType::OpenAI, None, None)), None, &h, &env)
                .unwrap_err();
        // The refusal names the parameter, like the other key refusals.
        assert!(e.contains("api_key") && !e.contains("SENTINEL"), "{e}");
        assert!(
            resolve_provider_config(
                Some(cfg(ProviderType::OpenAI, None, Some("caller"))),
                None,
                &h,
                &env
            )
            .is_ok()
        );
    }

    #[test]
    fn env_keys_endpoint_reflects_policy() {
        assert_eq!(
            server_key_availability(&LlmSettings::from_limits(&Limits::off(), None), &env),
            (true, true)
        );
        assert_eq!(
            server_key_availability(&LlmSettings::from_limits(&Limits::hosted(), None), &env),
            (false, false)
        );
        assert_eq!(
            server_key_availability(&LlmSettings::from_limits(&Limits::off(), None), &|_| Some(
                String::new()
            )),
            (false, false)
        );
    }
}
