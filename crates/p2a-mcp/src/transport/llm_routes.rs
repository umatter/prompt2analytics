//! LLM chat routes (`/api/llm/*`) and the two tool executors the LLM tool
//! loop uses. This was the `llm_handlers` module of `transport/http.rs`.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::http::{ApiResponse, AppState, ContentItem};
use crate::exec::{CallOrigin, ToolCallError, ToolGovernor};
use crate::llm::{
    AnthropicProvider, ChatSlot, ContextManager, LlmError, LlmProvider, LlmSettings, Message,
    OllamaProvider, OpenAIProvider, ProviderConfig, ProviderType, ToolDefinition, ToolExecutor,
    build_dataset_context_bounded, check_llm_tool_call, get_system_prompt_with_context,
    resolve_provider_config, server_key_availability, truncate_str, truncate_tool_result,
    try_acquire_chat_slot,
};
use crate::session::Session;

/// Request for LLM chat.
#[derive(Debug, Deserialize)]
pub struct LlmChatRequest {
    pub session_id: String,
    pub message: String,
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
    #[serde(default)]
    pub history: Vec<Message>,
    /// Whether to have the LLM interpret tool results (default: true)
    #[serde(default = "default_interpret")]
    pub interpret: bool,
    /// Optional conversation ID for persistence
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Whether to auto-retrieve history from database when conversation_id
    /// is provided and history is empty (default: true)
    #[serde(default = "default_retrieve_history")]
    pub retrieve_history: bool,
    /// Maximum number of tool execution iterations (default: 25)
    #[serde(default)]
    pub max_tool_iterations: Option<usize>,
}

fn default_retrieve_history() -> bool {
    true
}

fn default_interpret() -> bool {
    true
}

/// Response for LLM chat.
#[derive(Debug, Serialize)]
pub struct LlmChatResponse {
    pub message: Message,
}

/// Response for listing models.
#[derive(Debug, Serialize)]
pub struct LlmModelsResponse {
    pub provider: String,
    pub models: Vec<String>,
}

/// Internal tool executor that runs the LLM's tool calls through the
/// governor. A refusal or failure becomes the tool's error text, so the
/// model can retry with smaller parameters.
pub struct SessionToolExecutor {
    governor: Arc<ToolGovernor>,
    server: Arc<crate::server::AnalyticsServer>,
    session: Arc<Session>,
}

impl SessionToolExecutor {
    pub fn new(
        governor: Arc<ToolGovernor>,
        server: Arc<crate::server::AnalyticsServer>,
        session: Arc<Session>,
    ) -> Self {
        Self {
            governor,
            server,
            session,
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for SessionToolExecutor {
    async fn execute(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        refuse_unless_callable(&self.server, name)?;
        tracing::warn!(">>> SessionToolExecutor::execute called for tool: {}", name);
        tracing::info!(tool = %name, "Executing tool");
        let start = std::time::Instant::now();
        let result = self
            .governor
            .call(
                self.server.clone(),
                self.session.clone(),
                name,
                arguments,
                CallOrigin::LlmTool,
            )
            .await
            .map_err(|e| e.to_string());
        let elapsed = start.elapsed();
        tracing::info!(tool = %name, elapsed_ms = %elapsed.as_millis(), "Tool execution completed");
        match result {
            Ok(result) => {
                // Convert tool result to string
                let content = result
                    .content
                    .iter()
                    .map(|item| match item {
                        ContentItem::Text { text } => text.clone(),
                        ContentItem::Image { .. } => "[Image output]".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(content)
            }
            Err(e) => Err(e),
        }
    }
}

/// The names of every registered tool. The tool set is fixed per binary,
/// so it is collected once.
fn llm_callable_tools(server: &crate::server::AnalyticsServer) -> &'static HashSet<String> {
    static TOOLS: OnceLock<HashSet<String>> = OnceLock::new();
    TOOLS.get_or_init(|| server.list_tools().into_iter().map(|t| t.name).collect())
}

/// Refuse a model-emitted tool name the assistant may not call: an internal
/// tool, or one no handler registers. The refusal is the tool's error text.
fn refuse_unless_callable(
    server: &crate::server::AnalyticsServer,
    name: &str,
) -> Result<(), String> {
    check_llm_tool_call(name, |n| llm_callable_tools(server).contains(n)).inspect_err(|_| {
        tracing::warn!(
            tool = %truncate_str(name, 64),
            "Refused a tool call the assistant may not make"
        );
    })
}

/// Read a server environment variable (the key lookup of
/// `resolve_provider_config` and `server_key_availability`).
fn server_env(var: &str) -> Option<String> {
    std::env::var(var).ok()
}

/// Build the provider for a config that `resolve_provider_config` accepted:
/// its endpoint passed the egress policy and its key follows the key policy.
fn build_provider(cfg: ProviderConfig, settings: &LlmSettings) -> Box<dyn LlmProvider> {
    match cfg.provider_type {
        ProviderType::Ollama => Box::new(OllamaProvider::with_settings(cfg, settings)),
        ProviderType::OpenAI => Box::new(OpenAIProvider::with_settings(cfg, settings)),
        ProviderType::Anthropic => Box::new(AnthropicProvider::with_settings(cfg, settings)),
    }
}

/// The HTTP status of a failed LLM call: the provider's own failure is a
/// 502, its silence a 504, a refused endpoint or key the caller's 400. The
/// message is the error's text, which never carries an upstream body.
fn llm_error_status(e: &LlmError) -> StatusCode {
    match e {
        LlmError::Upstream { .. } => StatusCode::BAD_GATEWAY,
        LlmError::Timeout => StatusCode::GATEWAY_TIMEOUT,
        LlmError::EgressDenied(_) | LlmError::InvalidApiKey => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// The prompt of a title request: the user's message (at most 2,000 bytes)
/// and the assistant's reply (at most 500), cut on char boundaries.
fn title_prompt(user: &str, assistant: Option<&str>) -> String {
    let context = match assistant {
        Some(a) => format!(
            "User: {}\nAssistant: {}",
            truncate_str(user, 2_000),
            truncate_str(a, 500)
        ),
        None => format!("User: {}", truncate_str(user, 2_000)),
    };
    format!(
        "Generate a very short title (3-6 words, no quotes) for this conversation:\n\n{context}\n\nTitle:"
    )
}

/// The result preview of a `ToolEnd` event: at most 2,000 bytes plus "...".
fn tool_end_preview(text: &str) -> String {
    if text.len() > 2_000 {
        format!("{}...", truncate_str(text, 2_000))
    } else {
        text.to_string()
    }
}

/// The size of a dataset's source file for its stored metadata, read only
/// through the path jail inside the session's scope: `None` for a path
/// outside the data root or a file that cannot be read. Synchronous, so the
/// scope is never held across an `.await`.
#[cfg_attr(not(feature = "db"), allow(dead_code))]
fn jailed_file_size(session_id: &str, path: &str) -> Option<i64> {
    let _scope = crate::path_jail::enter_session_scope(session_id);
    let checked = crate::path_jail::validate_data_path(path).ok()?;
    let len = std::fs::metadata(checked).ok()?.len();
    Some(i64::try_from(len).unwrap_or(i64::MAX))
}

/// Human-readable provider name for UI status messages. OpenRouter is
/// routed through the OpenAI-compatible client, so the provider_type
/// alone would say "openai" even when the user picked OpenRouter.
fn provider_display_name(config: Option<&ProviderConfig>) -> String {
    let Some(cfg) = config else {
        return "Ollama".to_string();
    };
    if cfg.provider_type == ProviderType::OpenAI
        && cfg
            .base_url
            .as_deref()
            .is_some_and(|url| url.contains("openrouter.ai"))
    {
        return "OpenRouter".to_string();
    }
    match cfg.provider_type {
        ProviderType::OpenAI => "OpenAI".to_string(),
        ProviderType::Anthropic => "Anthropic".to_string(),
        ProviderType::Ollama => "Ollama".to_string(),
    }
}

/// Build the LLM's dataset context for `session` off the async runtime,
/// under the governor's admission control, permits and deadline:
/// summarizing every column of every dataset (unique counts, null masks,
/// numeric ranges) is real work on a large session, and it runs before any
/// provider or key is contacted. The datasets are cloned first (cheap: the
/// columns are shared), so no dataset lock is held while it runs. A session
/// without datasets needs no work and takes no permit. The block is at most
/// `context_bytes` long (`llm_context_bytes`).
async fn governed_dataset_context(
    governor: &ToolGovernor,
    session: &Arc<Session>,
    context_bytes: usize,
) -> Result<Option<String>, ToolCallError> {
    let datasets = session.datasets.read().await.clone();
    if datasets.is_empty() {
        return Ok(None);
    }
    governor
        .run_blocking(session, "llm_context", move |_budget| {
            Ok(build_dataset_context_bounded(&datasets, context_bytes))
        })
        .await
}

/// The messages of one chat request: the system prompt with the bounded
/// dataset block, the history fitted by [`ContextManager`] to the byte cap
/// (`llm_history_bytes`) and, for a model with a known window, its token
/// budget, then the user's message, which is never compacted.
async fn build_chat_messages(
    governor: &ToolGovernor,
    session: &Arc<Session>,
    history: Vec<Message>,
    user_message: String,
    tools: &[ToolDefinition],
    cfg: &ProviderConfig,
    settings: &LlmSettings,
) -> Result<Vec<Message>, ToolCallError> {
    let dataset_context =
        governed_dataset_context(governor, session, settings.context_bytes).await?;
    let system_prompt = get_system_prompt_with_context(dataset_context.as_deref(), tools);
    let reserved_for_response = cfg.max_tokens.unwrap_or(4096) as usize;
    let processed = ContextManager::for_request(
        &cfg.model,
        cfg.provider_type,
        settings.history_bytes,
        reserved_for_response,
    )
    .with_tools(tools)
    .process_history(history, &system_prompt);
    tracing::info!(
        has_datasets = dataset_context.is_some(),
        history_messages = processed.messages.len(),
        turns_summarized = processed.turns_summarized,
        "Built the chat context"
    );

    let mut messages = Vec::with_capacity(processed.messages.len() + 2);
    messages.push(Message::system(system_prompt));
    messages.extend(processed.messages);
    messages.push(Message::user(user_message));
    Ok(messages)
}

/// LLM chat endpoint.
pub async fn llm_chat(
    State(state): State<AppState>,
    Json(request): Json<LlmChatRequest>,
) -> Response {
    // At most llm_streams_per_session chat requests run per session; the
    // non-streaming chat runs the same tool loop, so it takes a slot too.
    let settings = LlmSettings::current();
    let _slot = match acquire_chat_slot_or_busy(&request.session_id, settings.streams_per_session) {
        Ok(slot) => slot,
        Err(busy) => return *busy,
    };

    // Resolve the provider (egress policy, key policy, iteration clamp)
    // before anything else, so a refused config costs nothing.
    let provider_config = match resolve_provider_config(
        request.provider,
        request.max_tool_iterations,
        &settings,
        &server_env,
    ) {
        Ok(cfg) => cfg,
        Err(msg) => {
            return (StatusCode::BAD_REQUEST, Json(ApiResponse::<()>::error(msg))).into_response();
        }
    };

    // Get the session
    let session = match state.session_manager.get_session(&request.session_id).await {
        Ok(s) => s,
        Err(crate::session::SessionError::NotFound) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::<()>::error("Session not found")),
            )
                .into_response();
        }
        Err(crate::session::SessionError::Expired) => {
            return (
                StatusCode::GONE,
                Json(ApiResponse::<()>::error("Session expired")),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response();
        }
    };

    let tools = state.server.list_tools_for_llm();

    // The bounded dataset block (built under the governor), the history
    // fitted to the request's budget, and the user's message.
    let messages = match build_chat_messages(
        &state.governor,
        &session,
        request.history,
        request.message,
        &tools,
        &provider_config,
        &settings,
    )
    .await
    {
        Ok(messages) => messages,
        Err(e) => return e.into_response(),
    };

    // Create provider and tool executor
    let provider = build_provider(provider_config, &settings);
    let tool_executor = SessionToolExecutor::new(
        state.governor.clone(),
        state.server.clone(),
        session.clone(),
    );

    tracing::info!(provider = %provider.provider_type(), num_tools = %tools.len(), "Starting LLM chat");

    // Execute chat
    let start = std::time::Instant::now();
    let result = provider
        .chat(&messages, &tools, &tool_executor, request.interpret)
        .await;
    let elapsed = start.elapsed();

    match result {
        Ok(response) => {
            tracing::info!(elapsed_ms = %elapsed.as_millis(), "LLM chat completed successfully");
            (
                StatusCode::OK,
                Json(ApiResponse::success(LlmChatResponse { message: response })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, elapsed_ms = %elapsed.as_millis(), "LLM chat failed");
            (
                llm_error_status(&e),
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response()
        }
    }
}

/// List available LLM models.
pub async fn llm_list_models(State(_state): State<AppState>) -> impl IntoResponse {
    // The model list comes from the default local Ollama endpoint, which the
    // egress policy may not allow (hosted profile, explicit allowlist).
    let settings = LlmSettings::current();
    let config = ProviderConfig::default();
    if settings.egress.check_provider_config(&config).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiResponse::error(
                "LLM provider not available: the default local Ollama endpoint is not enabled \
                 on this server",
            )),
        );
    }
    let provider = build_provider(config, &settings);

    match provider.list_models().await {
        Ok(models) => (
            StatusCode::OK,
            Json(ApiResponse::success(LlmModelsResponse {
                provider: provider.provider_type().to_string(),
                models,
            })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiResponse::error(format!(
                "LLM provider not available: {}",
                e
            ))),
        ),
    }
}

/// Response for env keys check.
#[derive(Debug, Serialize)]
pub struct EnvKeysResponse {
    pub openai: bool,
    pub anthropic: bool,
}

/// Report for which providers the server would supply its own API key:
/// false for both when server keys are disabled (hosted profile).
pub async fn llm_env_keys() -> impl IntoResponse {
    let (openai, anthropic) = server_key_availability(&LlmSettings::current(), &server_env);

    (
        StatusCode::OK,
        Json(ApiResponse::success(EnvKeysResponse { openai, anthropic })),
    )
}

/// Request for generating a conversation title.
#[derive(Debug, Deserialize)]
pub struct GenerateTitleRequest {
    /// The user's first message to base the title on
    pub user_message: String,
    /// Optional assistant response for more context
    #[serde(default)]
    pub assistant_response: Option<String>,
    /// Provider configuration
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
}

/// Response for title generation.
#[derive(Debug, Serialize)]
pub struct GenerateTitleResponse {
    pub title: String,
}

/// Generate a conversation title using the LLM.
pub async fn llm_generate_title(Json(request): Json<GenerateTitleRequest>) -> impl IntoResponse {
    // Resolve the provider (egress policy, key policy) before any request.
    let settings = LlmSettings::current();
    let provider_config =
        match resolve_provider_config(request.provider, None, &settings, &server_env) {
            Ok(cfg) => cfg,
            Err(msg) => return (StatusCode::BAD_REQUEST, Json(ApiResponse::error(msg))),
        };

    let provider = build_provider(provider_config, &settings);

    // Build a simple prompt for title generation
    let prompt = title_prompt(&request.user_message, request.assistant_response.as_deref());

    let messages = vec![
        Message::system(
            "You are a helpful assistant that generates concise conversation titles. Respond with only the title, no quotes or punctuation.",
        ),
        Message::user(prompt),
    ];

    // Use chat without tools for simple title generation
    match provider
        .chat(&messages, &[], &NoOpToolExecutor, false)
        .await
    {
        Ok(response) => {
            // Clean up the title
            let title = response
                .content
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();

            // Limit title length
            let title = if title.len() > 50 {
                format!("{}...", title.chars().take(47).collect::<String>())
            } else {
                title
            };

            (
                StatusCode::OK,
                Json(ApiResponse::success(GenerateTitleResponse { title })),
            )
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to generate title");
            (
                llm_error_status(&e),
                Json(ApiResponse::error(format!(
                    "Failed to generate title: {}",
                    e
                ))),
            )
        }
    }
}

/// No-op tool executor for simple chat without tools.
struct NoOpToolExecutor;

#[async_trait::async_trait]
impl ToolExecutor for NoOpToolExecutor {
    async fn execute(&self, _name: &str, _arguments: serde_json::Value) -> Result<String, String> {
        Err("Tool execution not available".to_string())
    }
}

/// Progress event for streaming chat.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    Status {
        message: String,
    },
    ToolStart {
        tool: String,
        arguments: serde_json::Value,
    },
    ToolEnd {
        tool: String,
        elapsed_ms: u64,
        result: Option<String>,
    },
    /// Tool result with images (for viz tools)
    ToolResult {
        tool: String,
        images: Vec<ImageData>,
    },
    Content {
        text: String,
    },
    Done {
        message: Message,
    },
    Error {
        error: String,
    },
}

/// Image data for tool results.
#[derive(Debug, Clone, Serialize)]
pub struct ImageData {
    pub data: String,
    pub mime_type: String,
}

/// Streaming tool executor that sends progress events. Tool calls run
/// through the governor, like [`SessionToolExecutor`].
pub struct StreamingToolExecutor {
    governor: Arc<ToolGovernor>,
    server: Arc<crate::server::AnalyticsServer>,
    session: Arc<Session>,
    sender: tokio::sync::mpsc::Sender<ProgressEvent>,
    /// Database connection for persistence (optional)
    #[cfg(feature = "db")]
    db: Option<Arc<crate::db::DbConnection>>,
    /// Conversation ID for tool call persistence
    #[cfg(feature = "db")]
    conversation_id: Option<String>,
    /// Message ID for tool call association
    #[cfg(feature = "db")]
    message_id: Option<String>,
    /// Session ID for dataset metadata persistence
    #[cfg(feature = "db")]
    session_id: Option<String>,
}

impl StreamingToolExecutor {
    pub fn new(
        governor: Arc<ToolGovernor>,
        server: Arc<crate::server::AnalyticsServer>,
        session: Arc<Session>,
        sender: tokio::sync::mpsc::Sender<ProgressEvent>,
    ) -> Self {
        Self {
            governor,
            server,
            session,
            sender,
            #[cfg(feature = "db")]
            db: None,
            #[cfg(feature = "db")]
            conversation_id: None,
            #[cfg(feature = "db")]
            message_id: None,
            #[cfg(feature = "db")]
            session_id: None,
        }
    }

    #[cfg(feature = "db")]
    pub fn with_persistence(
        mut self,
        db: Arc<crate::db::DbConnection>,
        conversation_id: String,
        message_id: String,
        session_id: String,
    ) -> Self {
        self.db = Some(db);
        self.conversation_id = Some(conversation_id);
        self.message_id = Some(message_id);
        self.session_id = Some(session_id);
        self
    }

    /// Set just db and session_id for dataset metadata tracking
    /// (doesn't require conversation_id/message_id)
    #[cfg(feature = "db")]
    pub fn with_dataset_tracking(
        mut self,
        db: Arc<crate::db::DbConnection>,
        session_id: String,
    ) -> Self {
        self.db = Some(db);
        self.session_id = Some(session_id);
        self
    }

    /// Capture dataset metadata after successful load_dataset or upload_dataset
    /// This is the internal method called by the executor.
    #[cfg(feature = "db")]
    async fn capture_dataset_metadata(
        db: Arc<crate::db::DbConnection>,
        session_id: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
        result_text: &str,
    ) {
        Self::capture_dataset_metadata_static(db, session_id, tool_name, arguments, result_text)
            .await;
    }

    /// Public static method for capturing dataset metadata.
    /// Can be called from outside the executor (e.g., direct tool calls).
    #[cfg(feature = "db")]
    pub async fn capture_dataset_metadata_static(
        db: Arc<crate::db::DbConnection>,
        session_id: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
        result_text: &str,
    ) {
        use std::path::PathBuf;

        // DEBUG: Log entry point with full context
        tracing::info!(
            tool = tool_name,
            session = session_id,
            result_len = result_text.len(),
            "[METADATA_CAPTURE] Starting metadata capture"
        );

        // DEBUG: Log the full result text (truncated for very long results)
        let result_preview = if result_text.len() > 2000 {
            format!(
                "{}...[truncated, total {} bytes]",
                truncate_str(result_text, 2_000),
                result_text.len()
            )
        } else {
            result_text.to_string()
        };
        tracing::info!(
            result_text = %result_preview,
            "[METADATA_CAPTURE] Full result text"
        );

        // Parse result to extract dataset info
        // Format: "Successfully loaded dataset 'name'\n\nDimensions: X rows x Y columns\n\nColumns:..."
        let first_line = result_text.lines().next();
        tracing::info!(
            first_line = ?first_line,
            "[METADATA_CAPTURE] First line of result"
        );

        let dataset_name = first_line.and_then(|line| {
            let start = line.find('\'');
            let end = line.rfind('\'');
            tracing::info!(
                line = %line,
                start_quote_pos = ?start,
                end_quote_pos = ?end,
                "[METADATA_CAPTURE] Parsing dataset name from first line"
            );
            match (start, end) {
                (Some(s), Some(e)) if s < e => {
                    let name = line[s + 1..e].to_string();
                    tracing::info!(
                        extracted_name = %name,
                        "[METADATA_CAPTURE] Extracted dataset name"
                    );
                    Some(name)
                }
                _ => {
                    tracing::warn!("[METADATA_CAPTURE] Could not extract dataset name - quote positions invalid");
                    None
                }
            }
        });

        // Find dimensions line
        let dimensions_line = result_text
            .lines()
            .find(|line| line.trim().starts_with("Dimensions:"));
        tracing::info!(
            dimensions_line = ?dimensions_line,
            "[METADATA_CAPTURE] Found dimensions line"
        );

        let dimensions = dimensions_line.and_then(|line| {
            // "Dimensions: X rows x Y columns"
            let parts: Vec<&str> = line.split_whitespace().collect();
            tracing::info!(
                parts = ?parts,
                parts_len = parts.len(),
                "[METADATA_CAPTURE] Dimensions line parts"
            );
            if parts.len() >= 5 {
                let rows = parts[1].parse::<i32>().ok();
                let cols = parts[4].parse::<i32>().ok();
                tracing::info!(
                    rows = ?rows,
                    cols = ?cols,
                    "[METADATA_CAPTURE] Parsed dimensions"
                );
                match (rows, cols) {
                    (Some(r), Some(c)) => Some((r, c)),
                    _ => {
                        tracing::warn!("[METADATA_CAPTURE] Could not parse row/col numbers");
                        None
                    }
                }
            } else {
                tracing::warn!(
                    parts_len = parts.len(),
                    "[METADATA_CAPTURE] Dimensions line has insufficient parts (need >= 5)"
                );
                None
            }
        });

        // Extract column names from "Columns:" section
        let columns_start = result_text.find("Columns:");
        tracing::info!(
            columns_start_pos = ?columns_start,
            "[METADATA_CAPTURE] Looking for 'Columns:' section"
        );

        let column_names: Vec<String> = if let Some(start) = columns_start {
            let columns_section = &result_text[start..];
            let lines_after_columns: Vec<&str> = columns_section.lines().collect();
            tracing::info!(
                total_lines_in_section = lines_after_columns.len(),
                first_5_lines = ?lines_after_columns.iter().take(5).collect::<Vec<_>>(),
                "[METADATA_CAPTURE] Lines in Columns section"
            );

            let parsed_columns: Vec<String> = columns_section
                .lines()
                .skip(1) // Skip "Columns:" line
                .enumerate()
                .filter_map(|(idx, line)| {
                    let trimmed_line = line.trim();
                    let starts_with_dash = trimmed_line.starts_with('-');

                    if !starts_with_dash {
                        if !trimmed_line.is_empty() {
                            tracing::debug!(
                                line_idx = idx,
                                line = %line,
                                trimmed = %trimmed_line,
                                "[METADATA_CAPTURE] Skipping line (doesn't start with '-')"
                            );
                        }
                        return None;
                    }

                    // "  - column_name (dtype): X nulls (Y%)"
                    let after_dash = trimmed_line.trim_start_matches('-').trim();
                    let paren_pos = after_dash.find('(');

                    tracing::info!(
                        line_idx = idx,
                        original_line = %line,
                        after_dash = %after_dash,
                        paren_pos = ?paren_pos,
                        "[METADATA_CAPTURE] Parsing column line"
                    );

                    match paren_pos {
                        Some(pos) => {
                            let col_name = after_dash[..pos].trim().to_string();
                            tracing::info!(
                                extracted_column = %col_name,
                                "[METADATA_CAPTURE] Extracted column name"
                            );
                            Some(col_name)
                        }
                        None => {
                            tracing::warn!(
                                line = %line,
                                "[METADATA_CAPTURE] Column line has no '(' - cannot extract name"
                            );
                            None
                        }
                    }
                })
                .collect();

            tracing::info!(
                num_columns_parsed = parsed_columns.len(),
                columns = ?parsed_columns,
                "[METADATA_CAPTURE] Finished parsing columns"
            );
            parsed_columns
        } else {
            tracing::warn!("[METADATA_CAPTURE] No 'Columns:' section found in result");
            Vec::new()
        };

        // Get source path and file metadata
        let source_path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let file_size_bytes = source_path
            .as_deref()
            .and_then(|p| jailed_file_size(session_id, p));

        // Determine source type
        let source_type = if tool_name == "upload_dataset" {
            arguments
                .get("filename")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
                .unwrap_or_else(|| "unknown".to_string())
        } else if tool_name == "create_dataset" {
            "inline_csv".to_string()
        } else {
            source_path
                .as_ref()
                .map(PathBuf::from)
                .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
                .unwrap_or_else(|| "unknown".to_string())
        };

        tracing::info!(
            dataset_name = ?dataset_name,
            dimensions = ?dimensions,
            num_columns = column_names.len(),
            column_names = ?column_names,
            source_type = %source_type,
            source_path = ?source_path,
            "[METADATA_CAPTURE] Final parsed metadata summary"
        );

        // Build and save metadata
        match (dataset_name.as_ref(), dimensions.as_ref()) {
            (Some(name), Some((rows, cols))) => {
                let mut meta = crate::db::DatasetMeta::new(
                    session_id.to_string(),
                    name.clone(),
                    source_type.clone(),
                    *rows,
                    *cols,
                    column_names.clone(),
                );
                meta.source_path = source_path;
                meta.file_size_bytes = file_size_bytes;

                tracing::info!(
                    meta_id = %meta.id_string(),
                    meta_session_id = %meta.session_id,
                    meta_name = %meta.name,
                    meta_source_type = %meta.source_type,
                    meta_row_count = meta.row_count,
                    meta_column_count = meta.column_count,
                    meta_column_names = ?meta.column_names,
                    meta_column_names_len = meta.column_names.len(),
                    "[METADATA_CAPTURE] DatasetMeta object before save"
                );

                match db.save_dataset_meta(&meta).await {
                    Ok(saved_meta) => {
                        tracing::info!(
                            dataset = %name,
                            session_id = %session_id,
                            saved_id = %saved_meta.id_string(),
                            saved_column_names = ?saved_meta.column_names,
                            saved_column_names_len = saved_meta.column_names.len(),
                            "[METADATA_CAPTURE] Successfully saved dataset metadata"
                        );

                        // DEBUG: Immediately read back from DB to verify
                        match db.get_dataset_meta(session_id, name).await {
                            Ok(Some(read_back)) => {
                                tracing::info!(
                                    read_back_id = %read_back.id_string(),
                                    read_back_name = %read_back.name,
                                    read_back_column_names = ?read_back.column_names,
                                    read_back_column_names_len = read_back.column_names.len(),
                                    "[METADATA_CAPTURE] Verified: read back from DB"
                                );
                            }
                            Ok(None) => {
                                tracing::error!(
                                    "[METADATA_CAPTURE] ERROR: Just saved but cannot read back - not found!"
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    error = %e,
                                    "[METADATA_CAPTURE] ERROR: Failed to read back from DB"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "[METADATA_CAPTURE] Failed to save dataset metadata"
                        );
                    }
                }
            }
            _ => {
                tracing::warn!(
                    dataset_name = ?dataset_name,
                    dimensions = ?dimensions,
                    "[METADATA_CAPTURE] Could not parse dataset info from result - missing name or dimensions"
                );
            }
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for StreamingToolExecutor {
    async fn execute(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        // Before any ToolStart event, DB tool-call row or governor call.
        refuse_unless_callable(&self.server, name)?;
        tracing::warn!(
            ">>> StreamingToolExecutor::execute called for tool: {}",
            name
        );

        // Send tool start event with arguments
        let _ = self
            .sender
            .send(ProgressEvent::ToolStart {
                tool: name.to_string(),
                arguments: arguments.clone(),
            })
            .await;

        // Create tool call record in DB if persistence is enabled
        #[cfg(feature = "db")]
        let tool_call_id = if let (Some(db), Some(conv_id), Some(msg_id)) =
            (&self.db, &self.conversation_id, &self.message_id)
        {
            let tool_call = crate::db::ToolCall::new(
                msg_id.clone(),
                conv_id.clone(),
                name.to_string(),
                serde_json::to_string(&arguments).unwrap_or_default(),
            );
            let tc_id = tool_call.id_string();

            // Mark as running
            let mut running_tool_call = tool_call;
            running_tool_call.status = crate::db::ToolCallStatus::Running;

            match db.create_tool_call(&running_tool_call).await {
                Ok(_) => {
                    tracing::debug!(tool_call_id = %tc_id, "Created tool call record");
                    Some(tc_id)
                }
                Err(e) => {
                    tracing::warn!("Failed to create tool call record: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Clone arguments for potential metadata capture later
        #[cfg(feature = "db")]
        let args_for_metadata = arguments.clone();

        tracing::info!(tool = %name, "Executing tool");
        let start = std::time::Instant::now();
        let result = self
            .governor
            .call(
                self.server.clone(),
                self.session.clone(),
                name,
                arguments,
                CallOrigin::LlmTool,
            )
            .await
            .map_err(|e| e.to_string());
        let elapsed = start.elapsed();
        let duration_ms = elapsed.as_millis() as i32;

        // Extract result text for the ToolEnd event
        let result_text = match &result {
            Ok(r) => {
                let text: String = r
                    .content
                    .iter()
                    .filter_map(|item| match item {
                        ContentItem::Text { text } => Some(text.clone()),
                        ContentItem::Image { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                // Truncate for the event (keep it reasonable for SSE)
                Some(tool_end_preview(&text))
            }
            Err(e) => Some(format!("Error: {}", e)),
        };

        // Send tool end event with result
        let _ = self
            .sender
            .send(ProgressEvent::ToolEnd {
                tool: name.to_string(),
                elapsed_ms: elapsed.as_millis() as u64,
                result: result_text,
            })
            .await;

        tracing::info!(tool = %name, elapsed_ms = %elapsed.as_millis(), "Tool execution completed");

        match result {
            Ok(result) => {
                // Extract result text for DB operations
                #[cfg(feature = "db")]
                let result_str = result
                    .content
                    .iter()
                    .filter_map(|item| match item {
                        ContentItem::Text { text } => Some(text.clone()),
                        ContentItem::Image { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                // Update tool call record with success
                #[cfg(feature = "db")]
                if let (Some(db), Some(tc_id)) = (&self.db, &tool_call_id) {
                    // Truncate result if too long (max 10KB)
                    let truncated_result = if result_str.len() > 10240 {
                        format!("{}...[truncated]", truncate_str(&result_str, 10_240))
                    } else {
                        result_str.clone()
                    };
                    if let Err(e) = db
                        .complete_tool_call(tc_id, &truncated_result, duration_ms)
                        .await
                    {
                        tracing::warn!("Failed to update tool call record: {}", e);
                    }
                }

                // Capture dataset metadata for load_dataset, upload_dataset, and create_dataset
                // This is separate from tool call tracking - it only needs db and session_id
                #[cfg(feature = "db")]
                if let Some(db) = &self.db {
                    if (name == "load_dataset"
                        || name == "upload_dataset"
                        || name == "create_dataset")
                        && result.error.is_none()
                    {
                        if let Some(session_id) = &self.session_id {
                            Self::capture_dataset_metadata(
                                db.clone(),
                                session_id,
                                name,
                                &args_for_metadata,
                                &result_str,
                            )
                            .await;
                        }
                    }
                }

                // Extract images and send them as a separate event
                let images: Vec<ImageData> = result
                    .content
                    .iter()
                    .filter_map(|item| match item {
                        ContentItem::Image { data, mime_type } => Some(ImageData {
                            data: data.clone(),
                            mime_type: mime_type.clone(),
                        }),
                        _ => None,
                    })
                    .collect();

                // If there are images, send them to the frontend
                if !images.is_empty() {
                    let _ = self
                        .sender
                        .send(ProgressEvent::ToolResult {
                            tool: name.to_string(),
                            images,
                        })
                        .await;
                }

                // Return text content only (without base64) for the LLM
                let content = result
                    .content
                    .iter()
                    .map(|item| match item {
                        ContentItem::Text { text } => text.clone(),
                        ContentItem::Image { .. } => "[Image output - displayed in UI]".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(content)
            }
            Err(e) => {
                // Update tool call record with error
                #[cfg(feature = "db")]
                if let (Some(db), Some(tc_id)) = (&self.db, &tool_call_id) {
                    if let Err(db_err) = db.fail_tool_call(tc_id, &e, duration_ms).await {
                        tracing::warn!("Failed to update tool call record: {}", db_err);
                    }
                }
                Err(e)
            }
        }
    }
}

/// Take a chat slot of the session, or the 429 `session_busy` answer when
/// `limit` chat requests are already running for it.
fn acquire_chat_slot_or_busy(
    session_id: &str,
    limit: Option<usize>,
) -> Result<ChatSlot, Box<Response>> {
    try_acquire_chat_slot(session_id, limit).ok_or_else(|| {
        let mut busy = (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ApiResponse::<()>::error_with_code(
                "Another chat request is still running for this session; wait for it to finish.",
                "session_busy",
                None,
            )),
        )
            .into_response();
        busy.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("2"),
        );
        Box::new(busy)
    })
}

/// Run `fut` for as long as the receiver of `tx` (the SSE stream) is alive.
/// `None` when the receiver went away first: `fut` is then dropped, and with
/// it everything it was awaiting.
async fn run_until_client_gone<F: std::future::Future>(
    tx: &tokio::sync::mpsc::Sender<ProgressEvent>,
    fut: F,
) -> Option<F::Output> {
    tokio::select! {
        biased;
        out = fut => Some(out),
        () = tx.closed() => None,
    }
}

/// Streaming LLM chat endpoint using Server-Sent Events.
///
/// The chat runs in its own task for as long as the client listens: when
/// the SSE stream is dropped (the client disconnected), the chat future is
/// dropped, and with it the provider request and any in-flight governed
/// tool call, whose guard cancels the tool; the stored message and tool
/// calls are closed as stopped ([`run_chat_task`]). The session's chat
/// slot is held until the task ends.
pub async fn llm_chat_stream(
    State(state): State<AppState>,
    Json(request): Json<LlmChatRequest>,
) -> Response {
    let settings = LlmSettings::current();
    let slot = match acquire_chat_slot_or_busy(&request.session_id, settings.streams_per_session) {
        Ok(slot) => slot,
        Err(busy) => return *busy,
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(100);

    #[cfg(feature = "db")]
    let db = state.persistent_manager.as_ref().map(|pm| pm.db().clone());
    let placeholder: Arc<OnceLock<String>> = Arc::default();
    tokio::spawn(async move {
        let _slot = slot;
        let chat = run_chat_stream(state, request, settings, tx.clone(), placeholder.clone());
        run_chat_task(
            &tx,
            chat,
            #[cfg(feature = "db")]
            db,
            &placeholder,
        )
        .await;
    });

    // Create SSE stream from receiver
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_default();
            yield Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().data(data),
            );

            // Stop after Done or Error
            if matches!(event, ProgressEvent::Done { .. } | ProgressEvent::Error { .. }) {
                break;
            }
        }
    };

    axum::response::sse::Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// The task of [`llm_chat_stream`]: `chat` runs for as long as the client
/// listens on `tx`. When the client disconnects first, `chat` is dropped
/// and, in db builds, what it left unfinished is closed in `db`: the
/// assistant message whose id is in `placeholder` is marked stopped, and
/// each of its tool calls still pending or running is failed, since the
/// dropped future will never finish it.
async fn run_chat_task<F: std::future::Future<Output = ()>>(
    tx: &tokio::sync::mpsc::Sender<ProgressEvent>,
    chat: F,
    #[cfg(feature = "db")] db: Option<Arc<crate::db::DbConnection>>,
    placeholder: &OnceLock<String>,
) {
    if run_until_client_gone(tx, chat).await.is_some() {
        return;
    }
    tracing::info!("The chat stream's client disconnected; the chat was stopped");
    #[cfg(feature = "db")]
    if let (Some(db), Some(msg_id)) = (&db, placeholder.get()) {
        stop_unfinished_chat(db, msg_id).await;
    }
    #[cfg(not(feature = "db"))]
    let _ = placeholder;
}

/// Error text of a tool call the client's disconnect stopped.
#[cfg(feature = "db")]
const STOPPED_TOOL_CALL: &str = "stopped: the client disconnected";

/// Mark assistant message `msg_id` stopped and fail its tool calls that are
/// still pending or running (see [`run_chat_task`]). Failures are logged.
#[cfg(feature = "db")]
async fn stop_unfinished_chat(db: &crate::db::DbConnection, msg_id: &str) {
    use crate::db::ToolCallStatus;
    if let Err(e) = db
        .update_message_content(msg_id, "[Stopped: the client disconnected]")
        .await
    {
        tracing::warn!("Failed to mark the stopped message: {}", e);
    }
    let calls = match db.get_tool_calls_for_message(msg_id).await {
        Ok(calls) => calls,
        Err(e) => {
            tracing::warn!("Failed to read the stopped message's tool calls: {}", e);
            return;
        }
    };
    for tc in calls
        .iter()
        .filter(|tc| matches!(tc.status, ToolCallStatus::Pending | ToolCallStatus::Running))
    {
        let started: chrono::DateTime<chrono::Utc> = tc.started_at.clone().into();
        let elapsed_ms = (chrono::Utc::now() - started).num_milliseconds().max(0);
        let duration_ms = i32::try_from(elapsed_ms).unwrap_or(i32::MAX);
        if let Err(e) = db
            .fail_tool_call(&tc.id_string(), STOPPED_TOOL_CALL, duration_ms)
            .await
        {
            tracing::warn!("Failed to mark a stopped tool call: {}", e);
        }
    }
}

/// The streaming chat: everything [`llm_chat_stream`] runs in its task,
/// ending with a `Done` or an `Error` event on `tx`. `placeholder` receives
/// the id of the assistant message it stores (db builds).
async fn run_chat_stream(
    state: AppState,
    request: LlmChatRequest,
    settings: LlmSettings,
    tx: tokio::sync::mpsc::Sender<ProgressEvent>,
    placeholder: Arc<OnceLock<String>>,
) {
    use crate::llm::StreamChunk;

    // Send initial status
    let _ = tx
        .send(ProgressEvent::Status {
            message: "Starting analysis...".to_string(),
        })
        .await;

    tracing::info!(interpret = %request.interpret, conversation_id = ?request.conversation_id, "LLM chat request received");

    // Get the session
    let session = match state.session_manager.get_session(&request.session_id).await {
        Ok(s) => s,
        Err(crate::session::SessionError::NotFound) => {
            let _ = tx
                .send(ProgressEvent::Error {
                    error: "Session not found".to_string(),
                })
                .await;
            return;
        }
        Err(e) => {
            let _ = tx
                .send(ProgressEvent::Error {
                    error: e.to_string(),
                })
                .await;
            return;
        }
    };

    // Compute a human-readable provider name for the status message
    // before `request.provider` is moved into the resolver — OpenRouter
    // is routed through the OpenAI client with a custom base_url, so
    // "openai" alone would be misleading in the connecting banner.
    let provider_display = provider_display_name(request.provider.as_ref());

    // Resolve the provider (egress policy, key policy, iteration clamp)
    // before any database write or provider request.
    let provider_config = match resolve_provider_config(
        request.provider,
        request.max_tool_iterations,
        &settings,
        &server_env,
    ) {
        Ok(cfg) => cfg,
        Err(msg) => {
            let _ = tx.send(ProgressEvent::Error { error: msg }).await;
            return;
        }
    };

    // Get database connection for persistence/tracking
    #[cfg(feature = "db")]
    let db_connection = state.persistent_manager.as_ref().map(|pm| pm.db().clone());

    // Normalize conversation_id: treat empty string as None
    #[cfg(feature = "db")]
    let effective_conversation_id = request
        .conversation_id
        .as_ref()
        .filter(|id| !id.is_empty())
        .cloned();

    // A named conversation must belong to this session. A missing and a
    // foreign conversation get the same answer, so the check is not an
    // existence oracle; it runs before the placeholder is written, stored
    // history is read or the provider is contacted.
    #[cfg(feature = "db")]
    if let (Some(db), Some(conv_id)) = (&db_connection, &effective_conversation_id) {
        let owned = matches!(
            db.get_conversation(conv_id).await,
            Ok(c) if c.session_id == request.session_id
        );
        if !owned {
            let _ = tx
                .send(ProgressEvent::Error {
                    error: "Conversation not found".to_string(),
                })
                .await;
            return;
        }
    }

    // Create placeholder assistant message if conversation persistence is enabled
    #[cfg(feature = "db")]
    let message_id = if let (Some(pm), Some(conv_id)) =
        (&state.persistent_manager, &effective_conversation_id)
    {
        // Create placeholder message
        match pm.db().add_message(conv_id, "assistant", "").await {
            Ok(msg) => {
                let msg_id = msg.id_string();
                tracing::debug!(message_id = %msg_id, "Created placeholder assistant message");
                Some(msg_id)
            }
            Err(e) => {
                tracing::warn!("Failed to create placeholder message: {}", e);
                None
            }
        }
    } else {
        None
    };
    #[cfg(feature = "db")]
    if let Some(id) = &message_id {
        let _ = placeholder.set(id.clone());
    }
    #[cfg(not(feature = "db"))]
    let _ = placeholder;

    // Create streaming tool executor (the provider is built once the
    // messages are, below)
    // Only the db build reconfigures the executor below.
    #[cfg_attr(not(feature = "db"), allow(unused_mut))]
    let mut tool_executor = StreamingToolExecutor::new(
        state.governor.clone(),
        state.server.clone(),
        session.clone(),
        tx.clone(),
    );

    // Track whether we've set up the executor for persistence
    #[cfg(feature = "db")]
    let mut persistence_configured = false;

    // Add full persistence context if conversation tracking is available
    #[cfg(feature = "db")]
    if let (Some(db), Some(conv_id), Some(msg_id)) =
        (&db_connection, &effective_conversation_id, &message_id)
    {
        tool_executor = tool_executor.with_persistence(
            db.clone(),
            conv_id.clone(),
            msg_id.clone(),
            request.session_id.clone(),
        );
        persistence_configured = true;
    }

    // Fall back to dataset metadata tracking if full persistence wasn't configured
    // This ensures we always capture dataset metadata when db is available
    #[cfg(feature = "db")]
    if !persistence_configured {
        if let Some(db) = &db_connection {
            tool_executor =
                tool_executor.with_dataset_tracking(db.clone(), request.session_id.clone());
        }
    }

    let tools = state.server.list_tools_for_llm();

    let _ = tx
        .send(ProgressEvent::Status {
            message: format!("Connecting to {provider_display}..."),
        })
        .await;

    // Auto-retrieve history from database if:
    // 1. conversation_id is provided
    // 2. history is empty
    // 3. retrieve_history flag is true (default)
    #[cfg(feature = "db")]
    let retrieved_history: Vec<Message> = if request.retrieve_history
        && request.history.is_empty()
        && effective_conversation_id.is_some()
    {
        if let (Some(db), Some(conv_id)) = (&db_connection, &effective_conversation_id) {
            match db.get_messages(conv_id).await {
                Ok(db_messages) => {
                    tracing::info!(
                        conversation_id = %conv_id,
                        message_count = db_messages.len(),
                        "Retrieved conversation history from database"
                    );
                    // Convert DB messages to LLM messages, including tool calls
                    let mut history = Vec::new();
                    for db_msg in db_messages {
                        // Get tool calls for this message
                        let tool_calls_for_msg =
                            db.get_tool_calls_for_message(&db_msg.id_string()).await;

                        match db_msg.role {
                            crate::db::MessageRole::User => {
                                if !db_msg.content.is_empty() {
                                    history.push(Message::user(db_msg.content.clone()));
                                }
                            }
                            crate::db::MessageRole::Assistant => {
                                // Check if this message has completed tool calls
                                if let Ok(tool_calls) = &tool_calls_for_msg {
                                    let completed_calls: Vec<_> = tool_calls
                                        .iter()
                                        .filter(|tc| {
                                            tc.status == crate::db::ToolCallStatus::Success
                                        })
                                        .collect();

                                    if !completed_calls.is_empty() {
                                        // Create tool calls for the assistant message
                                        let llm_tool_calls: Vec<crate::llm::ToolCall> =
                                            completed_calls
                                                .iter()
                                                .map(|tc| crate::llm::ToolCall {
                                                    id: tc.id_string(),
                                                    name: tc.tool_name.clone(),
                                                    arguments: serde_json::from_str(&tc.arguments)
                                                        .unwrap_or(serde_json::Value::Null),
                                                })
                                                .collect();

                                        // Assistant message with tool calls
                                        history.push(Message::assistant_with_tools(
                                            db_msg.content.clone(),
                                            llm_tool_calls,
                                        ));

                                        // Tool result message
                                        let tool_results: Vec<crate::llm::ToolResult> =
                                            completed_calls
                                                .iter()
                                                .map(|tc| {
                                                    let result_content =
                                                        tc.result.as_deref().unwrap_or("");
                                                    // Truncate long results
                                                    let truncated =
                                                        truncate_tool_result(result_content, 2_000);
                                                    crate::llm::ToolResult {
                                                        tool_call_id: tc.id_string(),
                                                        content: truncated,
                                                        is_error: false,
                                                    }
                                                })
                                                .collect();

                                        history.push(Message::tool_result(tool_results));
                                    } else if !db_msg.content.is_empty() {
                                        // No tool calls, just content
                                        history.push(Message::assistant(db_msg.content.clone()));
                                    }
                                } else if !db_msg.content.is_empty() {
                                    history.push(Message::assistant(db_msg.content.clone()));
                                }
                            }
                            crate::db::MessageRole::System => {
                                // Skip system messages in history retrieval
                            }
                        }
                    }
                    history
                }
                Err(e) => {
                    tracing::warn!(
                        conversation_id = %conv_id,
                        error = %e,
                        "Failed to retrieve conversation history, using empty history"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    #[cfg(not(feature = "db"))]
    let retrieved_history: Vec<Message> = Vec::new();

    // Determine which history to use: client-provided takes precedence.
    // The request's history and message move into the context; neither is
    // used again.
    let history_to_use = if request.history.is_empty() {
        retrieved_history
    } else {
        request.history
    };

    // The bounded dataset block (built under the governor), the history
    // fitted to the request's budget, and the user's message.
    let messages = match build_chat_messages(
        &state.governor,
        &session,
        history_to_use,
        request.message,
        &tools,
        &provider_config,
        &settings,
    )
    .await
    {
        Ok(messages) => messages,
        Err(e) => {
            #[cfg(feature = "db")]
            if let (Some(db), Some(msg_id)) = (&db_connection, &message_id) {
                let _ = db
                    .update_message_content(msg_id, &format!("[Error: {e}]"))
                    .await;
            }
            let _ = tx
                .send(ProgressEvent::Error {
                    error: e.to_string(),
                })
                .await;
            return;
        }
    };
    let provider = build_provider(provider_config, &settings);

    // Create streaming callback that forwards content chunks
    let tx_for_callback = tx.clone();
    let stream_callback: Box<dyn Fn(StreamChunk) + Send + Sync> = Box::new(move |chunk| {
        match chunk {
            StreamChunk::Text { content } => {
                // Use try_send for non-blocking send from sync callback
                let _ = tx_for_callback.try_send(ProgressEvent::Content { text: content });
            }
            StreamChunk::Done => {
                // Done is handled separately after chat_stream returns
            }
            StreamChunk::Error { message } => {
                let _ = tx_for_callback.try_send(ProgressEvent::Error { error: message });
            }
            StreamChunk::ToolCall { .. } | StreamChunk::ToolResult { .. } => {
                // Tool events are handled by StreamingToolExecutor
            }
        }
    });

    // Execute streaming chat
    match provider
        .chat_stream(
            &messages,
            &tools,
            &tool_executor,
            request.interpret,
            stream_callback,
        )
        .await
    {
        Ok(response) => {
            // Update message content in database if persistence is enabled
            #[cfg(feature = "db")]
            if let (Some(db), Some(msg_id)) = (&db_connection, &message_id) {
                if let Err(e) = db.update_message_content(msg_id, &response.content).await {
                    tracing::warn!("Failed to update message content: {}", e);
                } else {
                    tracing::debug!(message_id = %msg_id, "Updated assistant message content");
                }
            }

            // Debug: check if content has newlines
            let has_newlines = response.content.contains('\n');
            let content_preview: String = response.content.chars().take(300).collect();
            tracing::info!(
                has_newlines = has_newlines,
                content_preview = %content_preview,
                "Sending Done event with message content"
            );
            let _ = tx.send(ProgressEvent::Done { message: response }).await;
        }
        Err(e) => {
            // Update message with error content if persistence is enabled
            #[cfg(feature = "db")]
            if let (Some(db), Some(msg_id)) = (&db_connection, &message_id) {
                let error_content = format!("[Error: {}]", e);
                let _ = db.update_message_content(msg_id, &error_content).await;
            }

            let _ = tx
                .send(ProgressEvent::Error {
                    error: e.to_string(),
                })
                .await;
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::config::SessionConfig;
    use crate::exec::ExecConfig;
    use crate::session::SessionManager;

    /// A governor that refuses every call on memory admission (RSS > 1 byte).
    async fn governed_parts() -> (
        Arc<ToolGovernor>,
        Arc<crate::server::AnalyticsServer>,
        Arc<Session>,
    ) {
        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let id = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&id).await.unwrap();
        let governor = Arc::new(ToolGovernor::new(
            ExecConfig {
                rss_soft_limit_bytes: Some(1),
                ..Default::default()
            },
            sessions,
        ));
        (
            governor,
            Arc::new(crate::server::AnalyticsServer::new()),
            session,
        )
    }

    #[tokio::test]
    async fn dataset_context_runs_under_the_governor() {
        let (governor, _server, session) = governed_parts().await;
        // No datasets: nothing to summarize, no permit taken, no refusal.
        assert!(matches!(
            governed_dataset_context(&governor, &session, 65_536).await,
            Ok(None)
        ));
        let df = polars::prelude::df! {
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        .unwrap();
        session
            .datasets
            .write()
            .await
            .insert("d".into(), p2a_core::data::Dataset::new(df));
        // The refusing governor's memory admission applies to the context.
        let r = governed_dataset_context(&governor, &session, 65_536).await;
        assert!(
            matches!(r, Err(ToolCallError::MemoryPressure { .. })),
            "{r:?}"
        );
        // An admitting governor builds it off the runtime.
        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let admitting = ToolGovernor::new(ExecConfig::default(), sessions);
        let context = governed_dataset_context(&admitting, &session, 65_536)
            .await
            .unwrap()
            .expect("a context for one dataset");
        assert!(context.contains('y') && context.contains('x'), "{context}");
        assert_eq!(admitting.health().in_flight, 0);
    }

    #[tokio::test]
    async fn session_tool_executor_is_governed() {
        let (governor, server, session) = governed_parts().await;
        let executor = SessionToolExecutor::new(governor, server, session);
        let err = executor
            .execute("list_datasets", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.contains("low on memory"), "{err}");
    }

    #[tokio::test]
    async fn streaming_tool_executor_is_governed() {
        let (governor, server, session) = governed_parts().await;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let executor = StreamingToolExecutor::new(governor, server, session, tx);
        let err = executor
            .execute("list_datasets", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.contains("low on memory"), "{err}");
        let mut tool_end = None;
        while let Ok(event) = rx.try_recv() {
            if let ProgressEvent::ToolEnd { result, .. } = event {
                tool_end = result;
            }
        }
        let text = tool_end.expect("a ToolEnd event with a result");
        assert!(text.starts_with("Error:"), "{text}");
    }
}

/// The chat task's disconnect path against an in-memory database.
#[cfg(all(test, feature = "db"))]
mod disconnect_tests {
    use super::*;
    use crate::config::SessionConfig;
    use crate::db::{DbConnection, ToolCallStatus};
    use crate::exec::ExecConfig;
    use crate::session::SessionManager;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Sets its flag when dropped, so the permit holder below always ends,
    /// even when an assertion fails first.
    struct Release(Arc<AtomicBool>);

    impl Drop for Release {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn disconnect_mid_tool_fails_the_running_tool_call() {
        let db = Arc::new(DbConnection::connect_memory().await.unwrap());
        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let sid = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&sid).await.unwrap();
        let conv = db.create_conversation(&sid, "t").await.unwrap().id_string();
        let msg = db
            .add_message(&conv, "assistant", "")
            .await
            .unwrap()
            .id_string();
        // A finished call of the same message stays as it is.
        let done = db
            .create_tool_call(&crate::db::ToolCall::new(
                msg.clone(),
                conv.clone(),
                "list_datasets".into(),
                "{}".into(),
            ))
            .await
            .unwrap()
            .id_string();
        db.complete_tool_call(&done, "ok", 3).await.unwrap();

        // One permit, held, so the governed call waits in the queue with
        // its row already Running.
        let governor = Arc::new(ToolGovernor::new(
            ExecConfig {
                concurrency: 1,
                session_concurrency: 1,
                queue_wait: Duration::from_secs(300),
                ..Default::default()
            },
            sessions,
        ));
        let release = Arc::new(AtomicBool::new(false));
        let _release_on_exit = Release(release.clone());
        let rel = release.clone();
        let (g, s) = (governor.clone(), session.clone());
        let holder = tokio::spawn(async move {
            g.run_blocking(&s, "hold", move |_budget| {
                while !rel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(())
            })
            .await
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<ProgressEvent>(100);
        let executor = StreamingToolExecutor::new(
            governor.clone(),
            Arc::new(crate::server::AnalyticsServer::new()),
            session,
            tx.clone(),
        )
        .with_persistence(db.clone(), conv.clone(), msg.clone(), sid.clone());
        let placeholder = Arc::new(OnceLock::new());
        placeholder.set(msg.clone()).unwrap();
        let (task_db, task_placeholder) = (db.clone(), placeholder.clone());
        let task = tokio::spawn(async move {
            let chat = async move {
                let _ = executor
                    .execute("list_datasets", serde_json::json!({}))
                    .await;
            };
            run_chat_task(&tx, chat, Some(task_db), &task_placeholder).await;
        });

        // Wait (bounded) until the governed call's row is Running.
        let running = |calls: &[crate::db::ToolCall]| {
            calls
                .iter()
                .filter(|tc| tc.status == ToolCallStatus::Running)
                .count()
        };
        let mut waited = 0;
        while running(&db.get_tool_calls_for_message(&msg).await.unwrap()) == 0 {
            assert!(waited < 500, "the tool call never started");
            waited += 1;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // The client goes away mid-tool.
        drop(rx);
        tokio::time::timeout(Duration::from_secs(30), task)
            .await
            .expect("the chat task ends once the client is gone")
            .unwrap();

        let calls = db.get_tool_calls_for_message(&msg).await.unwrap();
        assert_eq!(calls.len(), 2);
        assert!(
            calls
                .iter()
                .all(|tc| !matches!(tc.status, ToolCallStatus::Pending | ToolCallStatus::Running)),
            "{calls:?}"
        );
        let stopped: Vec<_> = calls.iter().filter(|tc| tc.id_string() != done).collect();
        assert_eq!(stopped[0].status, ToolCallStatus::Error);
        assert_eq!(stopped[0].error.as_deref(), Some(STOPPED_TOOL_CALL));
        assert!(stopped[0].duration_ms.is_some_and(|ms| ms >= 0));
        let finished = calls.iter().find(|tc| tc.id_string() == done).unwrap();
        assert_eq!(
            (&finished.status, finished.result.as_deref()),
            (&ToolCallStatus::Success, Some("ok"))
        );
        assert_eq!(
            db.get_message(&msg).await.unwrap().content,
            "[Stopped: the client disconnected]"
        );

        release.store(true, Ordering::SeqCst);
        assert!(holder.await.unwrap().is_ok());
    }
}

/// Tests that run on every platform (the governor tests above are
/// Linux-only because they rely on RSS readings).
#[cfg(test)]
mod chat_tests {
    use super::*;
    use crate::config::SessionConfig;
    use crate::exec::ExecConfig;
    use crate::session::SessionManager;
    use crate::tools::limits::Limits;

    /// An admitting governor and one live session.
    async fn admitting_parts() -> (Arc<ToolGovernor>, Arc<Session>) {
        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let id = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&id).await.unwrap();
        let governor = Arc::new(ToolGovernor::new(ExecConfig::default(), sessions));
        (governor, session)
    }

    #[tokio::test]
    async fn chat_messages_use_the_request_budgets() {
        let (governor, session) = admitting_parts().await;
        let cols: Vec<polars::prelude::Column> = (0..3000)
            .map(|j| {
                let v: Vec<f64> = (0..3).map(|i| i as f64 * 0.37 + j as f64 * 0.011).collect();
                polars::prelude::Column::new(format!("c{j}").into(), v)
            })
            .collect();
        let df = polars::prelude::DataFrame::new(cols).unwrap();
        session
            .datasets
            .write()
            .await
            .insert("wide".into(), p2a_core::data::Dataset::new(df));
        let settings = LlmSettings {
            context_bytes: 4_096,
            history_bytes: 20_000,
            ..LlmSettings::from_limits(&Limits::off(), None)
        };
        let history: Vec<Message> = (0..30)
            .flat_map(|i| {
                [
                    Message::user(format!("u{i:02}:{}", "x".repeat(996))),
                    Message::assistant(format!("a{i:02}:{}", "y".repeat(996))),
                ]
            })
            .collect();
        let last = history.last().unwrap().content.clone();
        // Ollama: only the byte cap applies to the history.
        let cfg = ProviderConfig::default();
        let msgs = build_chat_messages(
            &governor,
            &session,
            history,
            "now".to_string(),
            &[],
            &cfg,
            &settings,
        )
        .await
        .unwrap();

        let sys = &msgs[0].content;
        assert_eq!(msgs[0].role, crate::llm::MessageRole::System);
        let block = &sys[sys.find("<<<DATASETS").unwrap()..sys.find("DATASETS>>>").unwrap()];
        assert!(block.len() <= 4_096 + 32, "{}", block.len());
        assert!(block.contains("\"c0\""));

        let hist = &msgs[1..msgs.len() - 1];
        let bytes: usize = hist.iter().map(|m| m.content.len()).sum();
        assert!(bytes <= 20_000, "{bytes}");
        assert!(
            hist[0]
                .content
                .starts_with("[Conversation context from earlier in this session:]")
        );
        assert_eq!(hist.last().unwrap().content, last);
        assert_eq!(msgs.last().unwrap().content, "now");
    }

    #[tokio::test]
    async fn chat_future_is_dropped_when_receiver_closes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (tx, rx) = tokio::sync::mpsc::channel::<ProgressEvent>(1);
        let dropped = Arc::new(AtomicBool::new(false));
        struct SetOnDrop(Arc<AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst)
            }
        }
        let g = SetOnDrop(dropped.clone());
        drop(rx);
        let out = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_until_client_gone(&tx, async move {
                let _g = g;
                std::future::pending::<()>().await
            }),
        )
        .await
        .expect("returns once the receiver is gone");
        assert!(out.is_none());
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn chat_future_completes_while_receiver_alive() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<ProgressEvent>(1);
        assert_eq!(run_until_client_gone(&tx, async { 7 }).await, Some(7));
    }

    #[tokio::test]
    async fn second_chat_in_a_limited_session_is_busy() {
        let held = acquire_chat_slot_or_busy("chat-busy-test", Some(1)).unwrap();
        let busy = acquire_chat_slot_or_busy("chat-busy-test", Some(1)).unwrap_err();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(busy.headers()[axum::http::header::RETRY_AFTER], "2");
        let body = axum::body::to_bytes(busy.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["code"], "session_busy");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("Another chat request is still running")
        );
        // Unlimited sessions are never busy; a released slot is free again.
        assert!(acquire_chat_slot_or_busy("chat-busy-test", None).is_ok());
        drop(held);
        assert!(acquire_chat_slot_or_busy("chat-busy-test", Some(1)).is_ok());
    }

    #[test]
    fn title_prompt_truncates_on_char_boundary() {
        // Byte 500 falls inside a char.
        let resp = format!("a{}", "é".repeat(400));
        let p = title_prompt(&format!("b{}", "ü".repeat(3000)), Some(&resp));
        assert!(p.contains("Assistant: a") && p.len() < 4_000, "{}", p.len());
        assert!(title_prompt("hi", None).contains("User: hi"));
    }

    #[test]
    fn tool_end_preview_is_char_safe() {
        let s = tool_end_preview(&format!("a{}", "é".repeat(1500)));
        assert!(s.len() <= 2_003 && s.ends_with("..."));
        assert_eq!(tool_end_preview("short"), "short");
    }

    #[test]
    fn metadata_file_size_goes_through_the_jail() {
        assert_eq!(
            jailed_file_size("00000000-0000-4000-8000-000000000000", "/etc/passwd"),
            None
        );
    }

    #[test]
    fn llm_errors_map_to_http_statuses() {
        let cases = [
            (
                LlmError::Upstream {
                    provider: "OpenAI",
                    status: 401,
                },
                StatusCode::BAD_GATEWAY,
            ),
            (LlmError::Timeout, StatusCode::GATEWAY_TIMEOUT),
            (LlmError::EgressDenied("no".into()), StatusCode::BAD_REQUEST),
            (LlmError::InvalidApiKey, StatusCode::BAD_REQUEST),
            (
                LlmError::NetworkError("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (e, want) in cases {
            assert_eq!(llm_error_status(&e), want, "{e}");
        }
    }
}
