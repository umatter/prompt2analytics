//! Ollama LLM provider implementation.
//!
//! Connects to a local Ollama instance for LLM inference with tool calling support.

use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ollama provider for local LLM inference.
pub struct OllamaProvider {
    client: Client,
    config: ProviderConfig,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Get the base URL for the Ollama API.
    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("http://localhost:11434")
    }

    /// Convert our Message format to Ollama's message format.
    fn to_ollama_messages(&self, messages: &[Message]) -> Vec<OllamaMessage> {
        let mut result = Vec::new();

        for msg in messages {
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
                                        arguments: tc.arguments.clone(),
                                    },
                                })
                                .collect(),
                        );
                    }

                    result.push(ollama_msg);
                }
                MessageRole::Tool => {
                    // Tool results are sent as separate tool messages
                    if let Some(ref tool_results) = msg.tool_results {
                        for tr in tool_results {
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

        loop {
            iterations += 1;
            if iterations > max_iterations {
                return Err(LlmError::ApiError(
                    "Maximum tool execution iterations exceeded".to_string(),
                ));
            }

            // Make the API call
            let response = self.call_api(messages, tools).await?;

            // Check if there are tool calls to execute
            if let Some(ref tool_calls) = response.tool_calls {
                if !tool_calls.is_empty() {
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
                        tracing::info!("interpret=false, returning raw tool output without LLM interpretation");
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: tool_outputs.join("\n\n"),
                            tool_calls: Some(tool_calls.clone()),
                            tool_results: Some(tool_results),
                        });
                    }
                    tracing::info!("interpret=true, continuing to LLM for interpretation");

                    // Continue the loop to get the next response
                    continue;
                }
            }

            // No tool calls - this is the final response
            return Ok(response);
        }
    }

    /// Make a single API call to Ollama.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
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

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, body
            )));
        }

        let chat_response: OllamaChatResponse = response.json().await?;

        // Convert response to our Message format
        let tool_calls = chat_response.message.tool_calls.map(|tcs| {
            tcs.into_iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{}", i),
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                })
                .collect()
        });

        Ok(Message {
            role: MessageRole::Assistant,
            content: chat_response.message.content,
            tool_calls,
            tool_results: None,
        })
    }

    /// Make a streaming API call to Ollama.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
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

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, body
            )));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
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
                        let converted: Vec<ToolCall> = tcs
                            .into_iter()
                            .enumerate()
                            .map(|(i, tc)| ToolCall {
                                id: format!("call_{}", i),
                                name: tc.function.name,
                                arguments: tc.function.arguments,
                            })
                            .collect();

                        for tc in &converted {
                            callback(StreamChunk::ToolCall {
                                tool_call: tc.clone(),
                            });
                        }
                        tool_calls = Some(converted);
                    }

                    // Check if done
                    if chunk_response.done {
                        callback(StreamChunk::Done);
                        break;
                    }
                }
            }
        }

        Ok(Message {
            role: MessageRole::Assistant,
            content: full_content,
            tool_calls,
            tool_results: None,
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
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = format!("{}/api/tags", self.base_url());
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(LlmError::NotAvailable(
                "Ollama is not running or not reachable".to_string(),
            ));
        }

        let tags: OllamaTagsResponse = response.json().await?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        interpret: bool,
    ) -> Result<Message, LlmError> {
        let mut conversation = messages.to_vec();
        self.execute_tool_loop(&mut conversation, tools, tool_executor, 10, interpret)
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
        let mut iterations = 0;
        let max_iterations = 10;

        loop {
            iterations += 1;
            if iterations > max_iterations {
                return Err(LlmError::ApiError(
                    "Maximum tool execution iterations exceeded".to_string(),
                ));
            }

            // Make streaming API call
            let response = self
                .call_api_stream(&conversation, tools, callback.as_ref())
                .await?;

            // Check if there are tool calls to execute
            if let Some(ref tool_calls) = response.tool_calls {
                if !tool_calls.is_empty() {
                    // Add the assistant message with tool calls
                    conversation.push(response.clone());

                    // Execute each tool and collect results
                    let mut tool_results = Vec::new();
                    let mut tool_outputs = Vec::new();
                    for tc in tool_calls {
                        let result = tool_executor.execute(&tc.name, tc.arguments.clone()).await;
                        let content = result
                            .clone()
                            .unwrap_or_else(|e| format!("Error: {}", e));
                        let tool_result = ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: content.clone(),
                            is_error: result.is_err(),
                        };

                        // Notify about tool result
                        callback(StreamChunk::ToolResult {
                            tool_result: tool_result.clone(),
                        });

                        tool_outputs.push(format!("**{}**:\n{}", tc.name, content));
                        tool_results.push(tool_result);
                    }

                    // Add tool results as a new message
                    conversation.push(Message::tool_result(tool_results.clone()));

                    // If interpret is false, return raw tool output without calling LLM again
                    if !interpret {
                        return Ok(Message {
                            role: MessageRole::Assistant,
                            content: tool_outputs.join("\n\n"),
                            tool_calls: Some(tool_calls.clone()),
                            tool_results: Some(tool_results),
                        });
                    }

                    // Continue the loop
                    continue;
                }
            }

            // No tool calls - this is the final response
            return Ok(response);
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
}
