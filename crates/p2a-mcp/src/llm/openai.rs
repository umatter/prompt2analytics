//! OpenAI LLM provider implementation.
//!
//! Connects to OpenAI's Chat Completions API for LLM inference with tool calling support.

use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// OpenAI provider for LLM inference.
pub struct OpenAIProvider {
    client: Client,
    config: ProviderConfig,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Get the base URL for the OpenAI API.
    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    }

    /// Get the API key.
    fn api_key(&self) -> Result<&str, LlmError> {
        self.config
            .api_key
            .as_deref()
            .ok_or(LlmError::InvalidApiKey)
    }

    /// Convert our Message format to OpenAI's message format.
    fn to_openai_messages(&self, messages: &[Message]) -> Vec<OpenAIMessage> {
        let mut result = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    result.push(OpenAIMessage {
                        role: "system".to_string(),
                        content: Some(msg.content.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                MessageRole::User => {
                    result.push(OpenAIMessage {
                        role: "user".to_string(),
                        content: Some(msg.content.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                MessageRole::Assistant => {
                    let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                        tcs.iter()
                            .map(|tc| OpenAIToolCall {
                                id: tc.id.clone(),
                                r#type: "function".to_string(),
                                function: OpenAIFunctionCall {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect()
                    });

                    result.push(OpenAIMessage {
                        role: "assistant".to_string(),
                        content: if msg.content.is_empty() {
                            None
                        } else {
                            Some(msg.content.clone())
                        },
                        tool_calls,
                        tool_call_id: None,
                    });
                }
                MessageRole::Tool => {
                    // Tool results in OpenAI format
                    if let Some(ref tool_results) = msg.tool_results {
                        for tr in tool_results {
                            result.push(OpenAIMessage {
                                role: "tool".to_string(),
                                content: Some(tr.content.clone()),
                                tool_calls: None,
                                tool_call_id: Some(tr.tool_call_id.clone()),
                            });
                        }
                    }
                }
            }
        }

        result
    }

    /// Convert tool definitions to OpenAI's format.
    fn to_openai_tools(&self, tools: &[ToolDefinition]) -> Vec<OpenAITool> {
        tools
            .iter()
            .map(|t| OpenAITool {
                r#type: "function".to_string(),
                function: OpenAIToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    /// Execute the tool calling loop until the LLM produces a final response.
    ///
    /// When `interpret` is true, the loop continues until the LLM produces a final text response.
    /// When `interpret` is false, the loop returns raw tool output after the first tool execution.
    async fn execute_tool_loop(
        &self,
        messages: &mut Vec<Message>,
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        max_iterations: usize,
        interpret: bool,
    ) -> Result<Message, LlmError> {
        let mut iterations = 0;
        let mut tool_call_history: Vec<u64> = Vec::new();
        let mut tool_name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut last_text_content = String::new();

        loop {
            iterations += 1;
            if iterations > max_iterations {
                tracing::warn!("Maximum tool execution iterations ({}) exceeded, returning partial results", max_iterations);
                let fallback = if last_text_content.is_empty() {
                    "Analysis reached the maximum number of tool calls. Here are the results gathered so far.".to_string()
                } else {
                    last_text_content
                };
                return Ok(Message {
                    role: MessageRole::Assistant,
                    content: fallback,
                    tool_calls: None,
                    tool_results: None,
                });
            }

            // Make the API call
            let response = self.call_api(messages, tools).await?;

            // Track last text content for graceful degradation
            if !response.content.is_empty() {
                last_text_content = response.content.clone();
            }

            // Check if there are tool calls to execute
            if let Some(ref tool_calls) = response.tool_calls {
                if !tool_calls.is_empty() {
                    // Loop detection: hash (tool_name, arguments) for each call
                    let mut iteration_hash = std::hash::DefaultHasher::new();
                    for tc in tool_calls {
                        tc.name.hash(&mut iteration_hash);
                        tc.arguments.to_string().hash(&mut iteration_hash);
                    }
                    let hash = Hasher::finish(&iteration_hash);

                    // Check for exact repeat of previous iteration
                    if tool_call_history.last() == Some(&hash) {
                        tracing::warn!("Loop detected: exact repeat of previous tool calls, breaking");
                        let fallback = if last_text_content.is_empty() {
                            "Analysis detected a repeated tool call pattern and stopped. Please try rephrasing your request.".to_string()
                        } else {
                            last_text_content
                        };
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: fallback,
                            tool_calls: None,
                            tool_results: None,
                        });
                    }
                    tool_call_history.push(hash);

                    // Check for same tool called too many times
                    for tc in tool_calls {
                        let count = tool_name_counts.entry(tc.name.clone()).or_insert(0);
                        *count += 1;
                        if *count > 3 {
                            tracing::warn!("Loop detected: tool '{}' called {} times, breaking", tc.name, count);
                            let fallback = if last_text_content.is_empty() {
                                format!("Analysis stopped: tool '{}' was called repeatedly. Here are the results gathered so far.", tc.name)
                            } else {
                                last_text_content
                            };
                            return Ok(Message {
                                role: MessageRole::Assistant,
                                content: fallback,
                                tool_calls: None,
                                tool_results: None,
                            });
                        }
                    }

                    // Add the assistant message with tool calls
                    messages.push(response.clone());

                    // Execute each tool and collect results
                    let mut tool_results = Vec::new();
                    let mut tool_outputs = Vec::new();
                    for tc in tool_calls {
                        let result = tool_executor.execute(&tc.name, tc.arguments.clone()).await;
                        let is_error = result.is_err();
                        let content = result.unwrap_or_else(|e| format!("Error: {}", e));
                        tool_outputs.push(format!("**{}**:\n{}", tc.name, content));
                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content,
                            is_error,
                        });
                    }

                    // Add tool results as a new message
                    messages.push(Message::tool_result(tool_results.clone()));

                    // If interpret is false, return raw tool output without calling LLM again
                    if !interpret {
                        tracing::info!(
                            "interpret=false, returning raw tool output without LLM interpretation"
                        );
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: tool_outputs.join("\n\n"),
                            tool_calls: Some(tool_calls.clone()),
                            tool_results: Some(tool_results),
                        });
                    }
                    tracing::info!("interpret=true, continuing to LLM for interpretation");

                    // Continue the loop to get the next response (interpretation)
                    continue;
                }
            }

            // No tool calls - this is the final response
            return Ok(response);
        }
    }

    /// Make a single API call to OpenAI.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let url = format!("{}/chat/completions", self.base_url());
        let api_key = self.api_key()?;

        let openai_messages = self.to_openai_messages(messages);
        let openai_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_openai_tools(tools))
        };

        let request_body = OpenAIChatRequest {
            model: self.config.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            stream: None,
        };

        let retry_config = super::retry::RetryConfig::default();
        let client = &self.client;
        let auth_header = format!("Bearer {}", api_key);
        let response = super::retry::send_with_retry(
            || {
                client
                    .post(&url)
                    .header("Authorization", &auth_header)
                    .header("Content-Type", "application/json")
                    .json(&request_body)
            },
            &retry_config,
        )
        .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let chat_response: OpenAIChatResponse = response.json().await?;

        // Get the first choice
        let choice = chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError("No response from OpenAI".to_string()))?;

        // Convert response to our Message format
        let tool_calls = choice.message.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| {
                    // Parse arguments from JSON string
                    let arguments: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                    ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments,
                    }
                })
                .collect()
        });

        Ok(Message {
            role: MessageRole::Assistant,
            content: choice.message.content.unwrap_or_default(),
            tool_calls,
            tool_results: None,
        })
    }

    /// Make a streaming API call to OpenAI.
    /// Returns the complete message and calls the callback for each text chunk.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
        let url = format!("{}/chat/completions", self.base_url());
        let api_key = self.api_key()?;

        let openai_messages = self.to_openai_messages(messages);
        let openai_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_openai_tools(tools))
        };

        let request_body = OpenAIChatRequest {
            model: self.config.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            stream: Some(true),
        };

        let retry_config = super::retry::RetryConfig::default();
        let client = &self.client;
        let auth_header = format!("Bearer {}", api_key);
        let response = super::retry::send_with_retry(
            || {
                client
                    .post(&url)
                    .header("Authorization", &auth_header)
                    .header("Content-Type", "application/json")
                    .json(&request_body)
            },
            &retry_config,
        )
        .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        // Process SSE stream
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut content = String::new();

        // For accumulating tool calls across chunks
        let mut tool_calls_map: HashMap<usize, (String, String, String)> = HashMap::new(); // index -> (id, name, arguments)

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| LlmError::NetworkError(e.to_string()))?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(chunk_response) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                        if let Some(choice) = chunk_response.choices.first() {
                            // Handle content delta
                            if let Some(ref delta_content) = choice.delta.content {
                                content.push_str(delta_content);
                                callback(StreamChunk::Text {
                                    content: delta_content.clone(),
                                });
                            }

                            // Handle tool call deltas
                            if let Some(ref tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let entry =
                                        tool_calls_map.entry(tc.index).or_insert_with(|| {
                                            (String::new(), String::new(), String::new())
                                        });

                                    if let Some(ref id) = tc.id {
                                        entry.0 = id.clone();
                                    }
                                    if let Some(ref func) = tc.function {
                                        if let Some(ref name) = func.name {
                                            entry.1 = name.clone();
                                        }
                                        if let Some(ref args) = func.arguments {
                                            entry.2.push_str(args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert accumulated tool calls to our format
        let tool_calls = if tool_calls_map.is_empty() {
            None
        } else {
            let mut calls: Vec<(usize, ToolCall)> = tool_calls_map
                .into_iter()
                .map(|(idx, (id, name, arguments))| {
                    let args: Value = serde_json::from_str(&arguments).unwrap_or(Value::Null);
                    (
                        idx,
                        ToolCall {
                            id,
                            name,
                            arguments: args,
                        },
                    )
                })
                .collect();
            calls.sort_by_key(|(idx, _)| *idx);
            Some(calls.into_iter().map(|(_, tc)| tc).collect())
        };

        Ok(Message {
            role: MessageRole::Assistant,
            content,
            tool_calls,
            tool_results: None,
        })
    }

    /// Execute the streaming tool calling loop.
    async fn execute_tool_loop_stream(
        &self,
        messages: &mut Vec<Message>,
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        max_iterations: usize,
        interpret: bool,
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
        let mut iterations = 0;
        let mut tool_call_history: Vec<u64> = Vec::new();
        let mut tool_name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut last_text_content = String::new();

        loop {
            iterations += 1;
            if iterations > max_iterations {
                tracing::warn!("Maximum tool execution iterations ({}) exceeded, returning partial results", max_iterations);
                let fallback = if last_text_content.is_empty() {
                    "Analysis reached the maximum number of tool calls. Here are the results gathered so far.".to_string()
                } else {
                    last_text_content
                };
                return Ok(Message {
                    role: MessageRole::Assistant,
                    content: fallback,
                    tool_calls: None,
                    tool_results: None,
                });
            }

            // Make the streaming API call
            let response = self.call_api_stream(messages, tools, callback).await?;

            // Track last text content for graceful degradation
            if !response.content.is_empty() {
                last_text_content = response.content.clone();
            }

            // Check if there are tool calls to execute
            if let Some(ref tool_calls) = response.tool_calls {
                if !tool_calls.is_empty() {
                    // Loop detection: hash (tool_name, arguments) for each call
                    let mut iteration_hash = std::hash::DefaultHasher::new();
                    for tc in tool_calls {
                        tc.name.hash(&mut iteration_hash);
                        tc.arguments.to_string().hash(&mut iteration_hash);
                    }
                    let hash = Hasher::finish(&iteration_hash);

                    // Check for exact repeat of previous iteration
                    if tool_call_history.last() == Some(&hash) {
                        tracing::warn!("Loop detected: exact repeat of previous tool calls, breaking");
                        let fallback = if last_text_content.is_empty() {
                            "Analysis detected a repeated tool call pattern and stopped. Please try rephrasing your request.".to_string()
                        } else {
                            last_text_content
                        };
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: fallback,
                            tool_calls: None,
                            tool_results: None,
                        });
                    }
                    tool_call_history.push(hash);

                    // Check for same tool called too many times
                    for tc in tool_calls {
                        let count = tool_name_counts.entry(tc.name.clone()).or_insert(0);
                        *count += 1;
                        if *count > 3 {
                            tracing::warn!("Loop detected: tool '{}' called {} times, breaking", tc.name, count);
                            let fallback = if last_text_content.is_empty() {
                                format!("Analysis stopped: tool '{}' was called repeatedly. Here are the results gathered so far.", tc.name)
                            } else {
                                last_text_content
                            };
                            return Ok(Message {
                                role: MessageRole::Assistant,
                                content: fallback,
                                tool_calls: None,
                                tool_results: None,
                            });
                        }
                    }

                    // Add the assistant message with tool calls
                    messages.push(response.clone());

                    // Execute each tool and collect results
                    let mut tool_results = Vec::new();
                    let mut tool_outputs = Vec::new();
                    for tc in tool_calls {
                        // Notify about tool call
                        callback(StreamChunk::ToolCall {
                            tool_call: tc.clone(),
                        });

                        let result = tool_executor.execute(&tc.name, tc.arguments.clone()).await;
                        let is_error = result.is_err();
                        let content = result.unwrap_or_else(|e| format!("Error: {}", e));
                        tool_outputs.push(format!("**{}**:\n{}", tc.name, content));

                        let tool_result = ToolResult {
                            tool_call_id: tc.id.clone(),
                            content,
                            is_error,
                        };

                        // Notify about tool result
                        callback(StreamChunk::ToolResult {
                            tool_result: tool_result.clone(),
                        });

                        tool_results.push(tool_result);
                    }

                    // Add tool results as a new message
                    messages.push(Message::tool_result(tool_results.clone()));

                    // If interpret is false, return raw tool output without calling LLM again
                    if !interpret {
                        tracing::info!(
                            "interpret=false, returning raw tool output without LLM interpretation"
                        );
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: tool_outputs.join("\n\n"),
                            tool_calls: Some(tool_calls.clone()),
                            tool_results: Some(tool_results),
                        });
                    }
                    tracing::info!("interpret=true, continuing to LLM for interpretation");

                    // Continue the loop to get the next response (interpretation)
                    continue;
                }
            }

            // No tool calls - this is the final response
            return Ok(response);
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn is_available(&self) -> Result<bool, LlmError> {
        // Check if API key is set
        Ok(self.config.api_key.is_some())
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = format!("{}/models", self.base_url());
        let api_key = self.api_key()?;

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let models_response: OpenAIModelsResponse = response.json().await?;

        // Filter to chat models only
        let chat_models: Vec<String> = models_response
            .data
            .into_iter()
            .filter(|m| m.id.starts_with("gpt-"))
            .map(|m| m.id)
            .collect();

        Ok(chat_models)
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
    ) -> Result<Message, LlmError> {
        let mut conversation = messages.to_vec();
        let max_iterations = self.config.max_tool_iterations.unwrap_or(super::provider::DEFAULT_MAX_TOOL_ITERATIONS);
        self.execute_tool_loop(&mut conversation, tools, tool_executor, max_iterations, interpret)
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
        callback: Box<dyn Fn(StreamChunk) + Send + Sync>,
    ) -> Result<Message, LlmError> {
        let mut conversation = messages.to_vec();
        let max_iterations = self.config.max_tool_iterations.unwrap_or(super::provider::DEFAULT_MAX_TOOL_ITERATIONS);
        self.execute_tool_loop_stream(
            &mut conversation,
            tools,
            tool_executor,
            max_iterations,
            interpret,
            callback.as_ref(),
        )
        .await
    }
}

// ========== OpenAI API Types ==========

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Clone)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIToolFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModel {
    id: String,
}

// ========== OpenAI Streaming Types ==========

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIStreamFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let mut config = ProviderConfig::default();
        config.provider_type = ProviderType::OpenAI;
        config.model = "gpt-4o".to_string();
        let provider = OpenAIProvider::new(config);
        assert_eq!(provider.provider_type(), ProviderType::OpenAI);
    }

    #[test]
    fn test_base_url_default() {
        let config = ProviderConfig::default();
        let provider = OpenAIProvider::new(config);
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn test_message_conversion() {
        let config = ProviderConfig::default();
        let provider = OpenAIProvider::new(config);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
        ];

        let openai_messages = provider.to_openai_messages(&messages);
        assert_eq!(openai_messages.len(), 2);
        assert_eq!(openai_messages[0].role, "system");
        assert_eq!(openai_messages[1].role, "user");
    }
}
