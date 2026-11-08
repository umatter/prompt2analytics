//! Ollama LLM provider implementation.
//!
//! Connects to a local Ollama instance for LLM inference with tool calling support. This module
//! is message conversion and transport; the tool-calling loop is [`super::tool_loop`].

use super::LlmSettings;
use super::retry::{MAX_UPSTREAM_BODY_BYTES, count_streamed, read_json_capped, upstream_error};
use super::tool_loop::{
    AssistantTurn, StreamSink, ToolLoopConfig, ToolLoopProvider, run_tool_loop,
};
use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, build_http_client,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// The name used in upstream error messages.
const PROVIDER: &str = "Ollama";

/// Ollama provider for local LLM inference.
pub struct OllamaProvider {
    /// `None` when the guarded client could not be built; every call then
    /// fails with [`LlmError::NotAvailable`].
    client: Option<Client>,
    config: ProviderConfig,
    /// Total time allowed for one non-streaming call.
    timeout: std::time::Duration,
    /// Cap on the tool-result text re-sent to the model in one tool loop
    /// (`llm_history_bytes`).
    loop_tool_bytes: usize,
    /// Upper bound on tool-loop iterations (`llm_max_tool_iterations`).
    max_iterations_cap: usize,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the server's current LLM settings.
    pub fn new(config: ProviderConfig) -> Self {
        Self::with_settings(config, &LlmSettings::current())
    }

    /// Create a new Ollama provider with explicit LLM settings (egress
    /// policy, timeout).
    pub fn with_settings(config: ProviderConfig, settings: &LlmSettings) -> Self {
        Self {
            client: build_http_client(ProviderType::Ollama, settings),
            config,
            timeout: settings.timeout,
            loop_tool_bytes: settings.history_bytes,
            max_iterations_cap: settings.max_tool_iterations,
        }
    }

    /// The HTTP client, or `NotAvailable` when it could not be built.
    fn client(&self) -> Result<&Client, LlmError> {
        self.client.as_ref().ok_or_else(|| {
            LlmError::NotAvailable("the HTTP client could not be initialised".to_string())
        })
    }

    /// Get the base URL for the Ollama API.
    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or(ProviderType::Ollama.default_base_url())
    }

    /// Convert our Message format to Ollama's message format.
    ///
    /// Tool results attached to an assistant message follow it (LLM-17); a tool-role message
    /// directly after it that repeats one is not sent twice.
    fn to_ollama_messages(&self, messages: &[Message]) -> Vec<OllamaMessage> {
        let mut result = Vec::new();
        // The ids of the results sent right after the latest assistant message.
        let mut sent: HashSet<&str> = HashSet::new();

        for msg in messages {
            if msg.role != MessageRole::Tool {
                sent.clear();
            }
            match msg.role {
                MessageRole::System | MessageRole::User | MessageRole::Assistant => {
                    let mut ollama_msg = OllamaMessage {
                        role: msg.role.to_string(),
                        content: msg.content.clone(),
                        tool_calls: None,
                    };

                    // Add tool calls if present (for assistant messages)
                    if let Some(ref tool_calls) = msg.tool_calls {
                        ollama_msg.tool_calls = Some(
                            tool_calls
                                .iter()
                                .map(|tc| OllamaToolCall {
                                    function: OllamaFunction {
                                        name: tc.name.clone(),
                                        arguments: tc.arguments_object(),
                                    },
                                })
                                .collect(),
                        );
                    }

                    result.push(ollama_msg);

                    for tr in msg.attached_tool_results() {
                        sent.insert(tr.tool_call_id.as_str());
                        result.push(OllamaMessage {
                            role: "tool".to_string(),
                            content: tr.content.clone(),
                            tool_calls: None,
                        });
                    }
                }
                MessageRole::Tool => {
                    // Tool results are sent as separate tool messages
                    if let Some(ref tool_results) = msg.tool_results {
                        for tr in tool_results
                            .iter()
                            .filter(|tr| !sent.contains(tr.tool_call_id.as_str()))
                        {
                            result.push(OllamaMessage {
                                role: "tool".to_string(),
                                content: tr.content.clone(),
                                tool_calls: None,
                            });
                        }
                    }
                }
            }
        }

        result
    }

    /// Convert tool definitions to Ollama's format.
    fn to_ollama_tools(&self, tools: &[ToolDefinition]) -> Vec<OllamaTool> {
        tools
            .iter()
            .map(|t| OllamaTool {
                r#type: "function".to_string(),
                function: OllamaToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    /// Make a single API call to Ollama.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<AssistantTurn, LlmError> {
        let url = format!("{}/api/chat", self.base_url());

        let ollama_messages = self.to_ollama_messages(messages);
        let ollama_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_ollama_tools(tools))
        };

        let request_body = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: ollama_messages,
            tools: ollama_tools,
            stream: false,
            options: Some(OllamaOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens.map(|n| n as i32),
            }),
        };

        let retry_config = super::retry::RetryConfig::default();
        let client = self.client()?;
        let timeout = self.timeout;
        let response = super::retry::send_with_retry(
            || client.post(&url).timeout(timeout).json(&request_body),
            &retry_config,
        )
        .await?;

        if !response.status().is_success() {
            return Err(upstream_error(PROVIDER, response).await);
        }

        let chat_response: OllamaChatResponse =
            read_json_capped(response, MAX_UPSTREAM_BODY_BYTES).await?;

        // Convert response to our Message format
        let tool_calls = chat_response
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(OllamaToolCall::into_call)
            .collect();

        Ok(AssistantTurn {
            content: chat_response.message.content,
            tool_calls,
        })
    }

    /// Make a streaming API call to Ollama.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &StreamSink,
    ) -> Result<AssistantTurn, LlmError> {
        let url = format!("{}/api/chat", self.base_url());

        let ollama_messages = self.to_ollama_messages(messages);
        let ollama_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_ollama_tools(tools))
        };

        let request_body = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: ollama_messages,
            tools: ollama_tools,
            stream: true,
            options: Some(OllamaOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens.map(|n| n as i32),
            }),
        };

        // No total timeout: a stream is bounded by the client's per-read
        // (idle) timeout and the response-size cap.
        let retry_config = super::retry::RetryConfig::default();
        let client = self.client()?;
        let response =
            super::retry::send_with_retry(|| client.post(&url).json(&request_body), &retry_config)
                .await?;

        if !response.status().is_success() {
            return Err(upstream_error(PROVIDER, response).await);
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut received = 0usize;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            count_streamed(&mut received, chunk.len())?;
            let text = String::from_utf8_lossy(&chunk);

            // Ollama streams JSON objects, one per line
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                if let Ok(chunk_response) = serde_json::from_str::<OllamaChatResponse>(line) {
                    // Append content if present
                    if !chunk_response.message.content.is_empty() {
                        full_content.push_str(&chunk_response.message.content);
                        callback(StreamChunk::Text {
                            content: chunk_response.message.content.clone(),
                        });
                    }

                    // Handle tool calls if present
                    if let Some(tcs) = chunk_response.message.tool_calls {
                        tool_calls = tcs.into_iter().map(OllamaToolCall::into_call).collect();
                    }

                    // Check if done
                    if chunk_response.done {
                        break;
                    }
                }
            }
        }

        Ok(AssistantTurn {
            content: full_content,
            tool_calls,
        })
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Ollama
    }

    async fn is_available(&self) -> Result<bool, LlmError> {
        let url = format!("{}/api/tags", self.base_url());
        let Ok(client) = self.client() else {
            return Ok(false);
        };
        match client.get(&url).timeout(self.timeout).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = format!("{}/api/tags", self.base_url());
        let response = self
            .client()?
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::NotAvailable(
                "Ollama is not running or not reachable".to_string(),
            ));
        }

        let tags: OllamaTagsResponse = read_json_capped(response, MAX_UPSTREAM_BODY_BYTES).await?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
    ) -> Result<Message, LlmError> {
        let config = ToolLoopConfig::for_provider(
            &self.config,
            self.max_iterations_cap,
            self.loop_tool_bytes,
            interpret,
        );
        run_tool_loop(self, messages, tools, tool_executor, &config, None).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
        callback: Box<dyn Fn(StreamChunk) + Send + Sync>,
    ) -> Result<Message, LlmError> {
        let config = ToolLoopConfig::for_provider(
            &self.config,
            self.max_iterations_cap,
            self.loop_tool_bytes,
            interpret,
        );
        run_tool_loop(
            self,
            messages,
            tools,
            tool_executor,
            &config,
            Some(callback.as_ref()),
        )
        .await
    }
}

#[async_trait]
impl ToolLoopProvider for OllamaProvider {
    fn convert(&self, messages: &[Message]) -> Result<Value, LlmError> {
        let mut wire = serde_json::Map::new();
        wire.insert(
            "messages".into(),
            serde_json::to_value(self.to_ollama_messages(messages))?,
        );
        Ok(Value::Object(wire))
    }

    async fn call(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: Option<&StreamSink>,
    ) -> Result<AssistantTurn, LlmError> {
        match stream {
            Some(sink) => self.call_api_stream(messages, tools, sink).await,
            None => self.call_api(messages, tools).await,
        }
    }
}

// ========== Ollama API Types ==========

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
}

#[derive(Debug, Serialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaFunction,
}

impl OllamaToolCall {
    /// As a [`ToolCall`]. Ollama's calls carry no id; the tool loop numbers them.
    fn into_call(self) -> ToolCall {
        ToolCall {
            id: String::new(),
            name: self.function.name,
            arguments: self.function.arguments,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaFunction {
    name: String,
    arguments: Value,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    r#type: String,
    function: OllamaToolFunction,
}

#[derive(Debug, Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    #[serde(default)]
    done: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolResult;

    /// The message /api/llm/chat returns with interpret=false (calls and results on one assistant
    /// message), replayed as history between two user turns.
    fn replayed() -> Vec<Message> {
        let call = ToolCall {
            id: "call_r1_1".into(),
            name: "list_datasets".into(),
            arguments: serde_json::json!({}),
        };
        let result = ToolResult {
            tool_call_id: "call_r1_1".into(),
            content: "No datasets".into(),
            is_error: false,
        };
        vec![
            Message::system("sys"),
            Message::user("list"),
            Message {
                role: MessageRole::Assistant,
                content: "**list_datasets**:\nNo datasets".into(),
                tool_calls: Some(vec![call]),
                tool_results: Some(vec![result]),
            },
            Message::user("next"),
        ]
    }

    /// LLM-17: each attached result follows its call as a `tool` message. Hand-derived from the
    /// serde attributes of OllamaMessage / OllamaToolCall (no ids on the wire).
    #[test]
    fn assistant_tool_results_follow_their_calls() {
        let p = OllamaProvider::new(ProviderConfig::default());
        let got = serde_json::to_value(p.to_ollama_messages(&replayed())).unwrap();
        assert_eq!(
            got,
            serde_json::json!([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "list"},
                {"role": "assistant", "content": "**list_datasets**:\nNo datasets",
                 "tool_calls": [{"function": {"name": "list_datasets", "arguments": {}}}]},
                {"role": "tool", "content": "No datasets"},
                {"role": "user", "content": "next"}
            ])
        );
    }

    /// A tool-role message right after the assistant message that repeats an attached result is not
    /// sent twice; a result naming none of the message's calls is not sent (no provider accepts it).
    #[test]
    fn tool_results_are_sent_once_and_only_for_their_calls() {
        let p = OllamaProvider::new(ProviderConfig::default());
        let mut h = replayed();
        let again = h[2].tool_results.clone().unwrap();
        h.insert(3, Message::tool_result(again));
        let once = serde_json::to_value(p.to_ollama_messages(&replayed())).unwrap();
        assert_eq!(
            serde_json::to_value(p.to_ollama_messages(&h)).unwrap(),
            once
        );
        let mut orphan = replayed();
        orphan[2].tool_results.as_mut().unwrap()[0].tool_call_id = "someone_else".into();
        let sent = serde_json::to_value(p.to_ollama_messages(&orphan)).unwrap();
        let roles: Vec<&str> = sent
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
    }

    #[test]
    fn test_provider_type() {
        let config = ProviderConfig::default();
        let provider = OllamaProvider::new(config);
        assert_eq!(provider.provider_type(), ProviderType::Ollama);
    }

    #[test]
    fn test_base_url_default() {
        let config = ProviderConfig::default();
        let provider = OllamaProvider::new(config);
        assert_eq!(provider.base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_base_url_custom() {
        let mut config = ProviderConfig::default();
        config.base_url = Some("http://192.168.1.100:11434".to_string());
        let provider = OllamaProvider::new(config);
        assert_eq!(provider.base_url(), "http://192.168.1.100:11434");
    }

    #[test]
    fn test_message_conversion() {
        let config = ProviderConfig::default();
        let provider = OllamaProvider::new(config);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let ollama_messages = provider.to_ollama_messages(&messages);
        assert_eq!(ollama_messages.len(), 3);
        assert_eq!(ollama_messages[0].role, "system");
        assert_eq!(ollama_messages[1].role, "user");
        assert_eq!(ollama_messages[2].role, "assistant");
    }

    #[test]
    fn function_arguments_are_always_an_object() {
        // Ollama takes `function.arguments` as an object; a call refused for
        // its arguments (also one stored by another provider in the same
        // conversation) goes back with the empty one.
        use crate::llm::provider::tool_call_tests::calls_with_every_argument_shape;
        let provider = OllamaProvider::new(ProviderConfig::default());
        let sent = provider.to_ollama_messages(&[calls_with_every_argument_shape()]);
        let arguments: Vec<&Value> = sent[0]
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|tc| &tc.function.arguments)
            .collect();
        let empty = serde_json::json!({});
        assert_eq!(
            arguments,
            [&serde_json::json!({"lag_max": 5}), &empty, &empty, &empty]
        );
    }
}
