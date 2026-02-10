//! Anthropic LLM provider implementation.
//!
//! Connects to Anthropic's Messages API for LLM inference with tool calling support.

use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic provider for LLM inference.
pub struct AnthropicProvider {
    client: Client,
    config: ProviderConfig,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Get the base URL for the Anthropic API.
    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com")
    }

    /// Get the API key.
    fn api_key(&self) -> Result<&str, LlmError> {
        self.config
            .api_key
            .as_deref()
            .ok_or(LlmError::InvalidApiKey)
    }

    /// Extract system message and convert remaining messages to Anthropic's format.
    fn to_anthropic_messages(
        &self,
        messages: &[Message],
    ) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt = None;
        let mut result = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    // Anthropic takes system as a separate parameter
                    system_prompt = Some(msg.content.clone());
                }
                MessageRole::User => {
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
                MessageRole::Assistant => {
                    // Check if this message has tool calls
                    if let Some(ref tool_calls) = msg.tool_calls {
                        let mut content_blocks = Vec::new();

                        // Add text content if present
                        if !msg.content.is_empty() {
                            content_blocks.push(AnthropicContentBlock::Text {
                                text: msg.content.clone(),
                            });
                        }

                        // Add tool_use blocks
                        for tc in tool_calls {
                            content_blocks.push(AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            });
                        }

                        result.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(content_blocks),
                        });
                    } else {
                        result.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Text(msg.content.clone()),
                        });
                    }
                }
                MessageRole::Tool => {
                    // Tool results go in a user message with tool_result content blocks
                    if let Some(ref tool_results) = msg.tool_results {
                        let content_blocks: Vec<AnthropicContentBlock> = tool_results
                            .iter()
                            .map(|tr| AnthropicContentBlock::ToolResult {
                                tool_use_id: tr.tool_call_id.clone(),
                                content: tr.content.clone(),
                                is_error: if tr.is_error { Some(true) } else { None },
                            })
                            .collect();

                        result.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(content_blocks),
                        });
                    }
                }
            }
        }

        (system_prompt, result)
    }

    /// Convert tool definitions to Anthropic's format.
    fn to_anthropic_tools(&self, tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect()
    }

    /// Execute the tool calling loop until the LLM produces a final response.
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

    /// Make a single API call to Anthropic.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let url = format!("{}/v1/messages", self.base_url());
        let api_key = self.api_key()?;

        let (system_prompt, anthropic_messages) = self.to_anthropic_messages(messages);
        let anthropic_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_anthropic_tools(tools))
        };

        let request_body = AnthropicChatRequest {
            model: self.config.model.clone(),
            messages: anthropic_messages,
            system: system_prompt,
            tools: anthropic_tools,
            max_tokens: self.config.max_tokens.unwrap_or(4096),
            stream: None,
        };

        tracing::debug!(
            model = %self.config.model,
            message_count = %messages.len(),
            tool_count = %tools.len(),
            "Sending request to Anthropic API"
        );

        let response = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        let chat_response: AnthropicChatResponse = response.json().await?;

        // Parse the response content blocks
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for block in chat_response.content {
            match block {
                AnthropicResponseBlock::Text { text } => {
                    text_content.push_str(&text);
                }
                AnthropicResponseBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
            }
        }

        Ok(Message {
            role: MessageRole::Assistant,
            content: text_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_results: None,
        })
    }

    /// Make a streaming API call to Anthropic.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
        let url = format!("{}/v1/messages", self.base_url());
        let api_key = self.api_key()?;

        let (system_prompt, anthropic_messages) = self.to_anthropic_messages(messages);
        let anthropic_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_anthropic_tools(tools))
        };

        let request_body = AnthropicChatRequest {
            model: self.config.model.clone(),
            messages: anthropic_messages,
            system: system_prompt,
            tools: anthropic_tools,
            max_tokens: self.config.max_tokens.unwrap_or(4096),
            stream: Some(true),
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        // Process SSE stream
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut text_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| LlmError::NetworkError(e.to_string()))?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                        match event {
                            AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                                match content_block {
                                    AnthropicStreamContentBlock::Text { .. } => {}
                                    AnthropicStreamContentBlock::ToolUse { id, name } => {
                                        current_tool_id = id;
                                        current_tool_name = name;
                                        current_tool_input.clear();
                                    }
                                }
                            }
                            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => match delta {
                                AnthropicStreamDelta::TextDelta { text } => {
                                    text_content.push_str(&text);
                                    callback(StreamChunk::Text { content: text });
                                }
                                AnthropicStreamDelta::InputJsonDelta { partial_json } => {
                                    current_tool_input.push_str(&partial_json);
                                }
                            },
                            AnthropicStreamEvent::ContentBlockStop { .. } => {
                                // If we were building a tool call, finalize it
                                if !current_tool_id.is_empty() {
                                    let input: Value = serde_json::from_str(&current_tool_input)
                                        .unwrap_or(Value::Null);
                                    tool_calls.push(ToolCall {
                                        id: current_tool_id.clone(),
                                        name: current_tool_name.clone(),
                                        arguments: input,
                                    });
                                    current_tool_id.clear();
                                    current_tool_name.clear();
                                    current_tool_input.clear();
                                }
                            }
                            AnthropicStreamEvent::MessageStop => {
                                // End of message
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(Message {
            role: MessageRole::Assistant,
            content: text_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
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

        loop {
            iterations += 1;
            if iterations > max_iterations {
                return Err(LlmError::ApiError(
                    "Maximum tool execution iterations exceeded".to_string(),
                ));
            }

            // Make the streaming API call
            let response = self.call_api_stream(messages, tools, callback).await?;

            // Check if there are tool calls to execute
            if let Some(ref tool_calls) = response.tool_calls {
                if !tool_calls.is_empty() {
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
impl LlmProvider for AnthropicProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    async fn is_available(&self) -> Result<bool, LlmError> {
        // Check if API key is set
        Ok(self.config.api_key.is_some())
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Anthropic doesn't have a models list endpoint, return known models
        Ok(vec![
            "claude-opus-4-5-20251101".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ])
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
        self.execute_tool_loop_stream(
            &mut conversation,
            tools,
            tool_executor,
            10,
            interpret,
            callback.as_ref(),
        )
        .await
    }
}

// ========== Anthropic API Types ==========

#[derive(Debug, Serialize)]
struct AnthropicChatRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicChatResponse {
    content: Vec<AnthropicResponseBlock>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

// ========== Anthropic Streaming Types ==========

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart {
        #[allow(dead_code)]
        message: Value,
    },
    ContentBlockStart {
        #[allow(dead_code)]
        index: usize,
        content_block: AnthropicStreamContentBlock,
    },
    ContentBlockDelta {
        #[allow(dead_code)]
        index: usize,
        delta: AnthropicStreamDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: usize,
    },
    MessageDelta {
        #[allow(dead_code)]
        delta: Value,
    },
    MessageStop,
    Ping,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamContentBlock {
    Text {
        #[allow(dead_code)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let mut config = ProviderConfig::default();
        config.provider_type = ProviderType::Anthropic;
        config.model = "claude-3-5-sonnet-20241022".to_string();
        let provider = AnthropicProvider::new(config);
        assert_eq!(provider.provider_type(), ProviderType::Anthropic);
    }

    #[test]
    fn test_base_url_default() {
        let config = ProviderConfig::default();
        let provider = AnthropicProvider::new(config);
        assert_eq!(provider.base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn test_message_conversion() {
        let config = ProviderConfig::default();
        let provider = AnthropicProvider::new(config);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
        ];

        let (system, anthropic_messages) = provider.to_anthropic_messages(&messages);
        assert_eq!(system, Some("You are a helpful assistant".to_string()));
        assert_eq!(anthropic_messages.len(), 1);
        assert_eq!(anthropic_messages[0].role, "user");
    }
}
