//! Anthropic Claude LLM provider implementation.
//!
//! Connects to the Anthropic API for Claude model inference with tool calling support.

use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Claude provider.
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

    /// Get the API key.
    fn api_key(&self) -> Result<&str, LlmError> {
        self.config
            .api_key
            .as_deref()
            .ok_or_else(|| LlmError::InvalidApiKey)
    }

    /// Get the API URL.
    fn api_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(ANTHROPIC_API_URL)
    }

    /// Convert our Message format to Anthropic's message format.
    /// Returns (system_prompt, messages).
    fn to_anthropic_messages(&self, messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt = None;
        let mut result = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system_prompt = Some(msg.content.clone());
                }
                MessageRole::User => {
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
                MessageRole::Assistant => {
                    // Build content blocks for assistant message
                    let mut content_blocks: Vec<AnthropicContentBlock> = Vec::new();

                    // Add text content if present
                    if !msg.content.is_empty() {
                        content_blocks.push(AnthropicContentBlock::Text {
                            text: msg.content.clone(),
                        });
                    }

                    // Add tool use blocks if present
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            content_blocks.push(AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            });
                        }
                    }

                    if !content_blocks.is_empty() {
                        result.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(content_blocks),
                        });
                    }
                }
                MessageRole::Tool => {
                    // Tool results go in a user message with tool_result blocks
                    if let Some(ref tool_results) = msg.tool_results {
                        let blocks: Vec<AnthropicContentBlock> = tool_results
                            .iter()
                            .map(|tr| AnthropicContentBlock::ToolResult {
                                tool_use_id: tr.tool_call_id.clone(),
                                content: tr.content.clone(),
                                is_error: if tr.is_error { Some(true) } else { None },
                            })
                            .collect();

                        result.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(blocks),
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

    /// Parse an Anthropic response into our Message format.
    fn parse_response(&self, response: &AnthropicResponse) -> Message {
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in &response.content {
            match block {
                AnthropicContentBlock::Text { text } => {
                    content.push_str(text);
                }
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
                _ => {}
            }
        }

        Message {
            role: MessageRole::Assistant,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_results: None,
        }
    }

    /// Make a single API call to Anthropic.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let api_key = self.api_key()?;
        let (system, anthropic_messages) = self.to_anthropic_messages(messages);

        let anthropic_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_anthropic_tools(tools))
        };

        let request_body = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens.unwrap_or(4096),
            system,
            messages: anthropic_messages,
            tools: anthropic_tools,
            temperature: self.config.temperature,
            stream: false,
        };

        let response = self
            .client
            .post(self.api_url())
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(match status.as_u16() {
                401 => LlmError::InvalidApiKey,
                429 => LlmError::RateLimited,
                _ => LlmError::ApiError(format!("Anthropic API error ({}): {}", status, body)),
            });
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        Ok(self.parse_response(&anthropic_response))
    }

    /// Execute the tool calling loop until the LLM produces a final response.
    async fn execute_tool_loop(
        &self,
        messages: &mut Vec<Message>,
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
        max_iterations: usize,
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
                    for tc in tool_calls {
                        let result = tool_executor.execute(&tc.name, tc.arguments.clone()).await;
                        let is_error = result.is_err();
                        tool_results.push(ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: result.unwrap_or_else(|e| format!("Error: {}", e)),
                            is_error,
                        });
                    }

                    // Add tool results as a new message
                    messages.push(Message::tool_result(tool_results));

                    // Continue the loop to get the next response
                    continue;
                }
            }

            // No tool calls - check stop reason
            // If stop_reason is "end_turn" or "stop_sequence", we're done
            return Ok(response);
        }
    }

    /// Make a streaming API call to Anthropic.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
        let api_key = self.api_key()?;
        let (system, anthropic_messages) = self.to_anthropic_messages(messages);

        let anthropic_tools = if tools.is_empty() {
            None
        } else {
            Some(self.to_anthropic_tools(tools))
        };

        let request_body = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens.unwrap_or(4096),
            system,
            messages: anthropic_messages,
            tools: anthropic_tools,
            temperature: self.config.temperature,
            stream: true,
        };

        let response = self
            .client
            .post(self.api_url())
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(match status.as_u16() {
                401 => LlmError::InvalidApiKey,
                429 => LlmError::RateLimited,
                _ => LlmError::ApiError(format!("Anthropic API error ({}): {}", status, body)),
            });
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let text = String::from_utf8_lossy(&chunk);

            // Parse SSE events
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        break;
                    }

                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                        match event {
                            AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                                match content_block {
                                    AnthropicContentBlock::ToolUse { id, name, .. } => {
                                        current_tool_id = id;
                                        current_tool_name = name;
                                        current_tool_input.clear();
                                    }
                                    _ => {}
                                }
                            }
                            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                                match delta {
                                    AnthropicDelta::TextDelta { text } => {
                                        full_content.push_str(&text);
                                        callback(StreamChunk::Text { content: text });
                                    }
                                    AnthropicDelta::InputJsonDelta { partial_json } => {
                                        current_tool_input.push_str(&partial_json);
                                    }
                                }
                            }
                            AnthropicStreamEvent::ContentBlockStop { .. } => {
                                // If we were building a tool call, finalize it
                                if !current_tool_id.is_empty() && !current_tool_name.is_empty() {
                                    let arguments: Value =
                                        serde_json::from_str(&current_tool_input)
                                            .unwrap_or(Value::Object(Default::default()));
                                    let tc = ToolCall {
                                        id: current_tool_id.clone(),
                                        name: current_tool_name.clone(),
                                        arguments,
                                    };
                                    callback(StreamChunk::ToolCall {
                                        tool_call: tc.clone(),
                                    });
                                    tool_calls.push(tc);
                                    current_tool_id.clear();
                                    current_tool_name.clear();
                                    current_tool_input.clear();
                                }
                            }
                            AnthropicStreamEvent::MessageStop => {
                                callback(StreamChunk::Done);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(Message {
            role: MessageRole::Assistant,
            content: full_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_results: None,
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    async fn is_available(&self) -> Result<bool, LlmError> {
        // Check if API key is configured
        Ok(self.config.api_key.is_some())
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Anthropic doesn't have a list models endpoint, so return known models
        Ok(vec![
            "claude-sonnet-4-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ])
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
    ) -> Result<Message, LlmError> {
        let mut conversation = messages.to_vec();
        self.execute_tool_loop(&mut conversation, tools, tool_executor, 10)
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_executor: &dyn ToolExecutor,
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
                    for tc in tool_calls {
                        let result = tool_executor.execute(&tc.name, tc.arguments.clone()).await;
                        let is_error = result.is_err();
                        let tool_result = ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: result.unwrap_or_else(|e| format!("Error: {}", e)),
                            is_error,
                        };

                        // Notify about tool result
                        callback(StreamChunk::ToolResult {
                            tool_result: tool_result.clone(),
                        });

                        tool_results.push(tool_result);
                    }

                    // Add tool results as a new message
                    conversation.push(Message::tool_result(tool_results));

                    // Continue the loop
                    continue;
                }
            }

            // No tool calls - this is the final response
            return Ok(response);
        }
    }
}

// ========== Anthropic API Types ==========

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: AnthropicContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicError,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
    model: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let config = ProviderConfig {
            provider_type: ProviderType::Anthropic,
            api_key: Some("test-key".to_string()),
            base_url: None,
            model: "claude-sonnet-4-20250514".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        let provider = AnthropicProvider::new(config);
        assert_eq!(provider.provider_type(), ProviderType::Anthropic);
    }

    #[test]
    fn test_message_conversion() {
        let config = ProviderConfig {
            provider_type: ProviderType::Anthropic,
            api_key: Some("test-key".to_string()),
            base_url: None,
            model: "claude-sonnet-4-20250514".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        let provider = AnthropicProvider::new(config);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let (system, anthropic_messages) = provider.to_anthropic_messages(&messages);
        assert_eq!(system, Some("You are a helpful assistant".to_string()));
        assert_eq!(anthropic_messages.len(), 2); // System is separate
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
    }

    #[test]
    fn test_tool_definition_conversion() {
        let config = ProviderConfig {
            provider_type: ProviderType::Anthropic,
            api_key: Some("test-key".to_string()),
            base_url: None,
            model: "claude-sonnet-4-20250514".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        let provider = AnthropicProvider::new(config);

        let tools = vec![ToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                }
            }),
        }];

        let anthropic_tools = provider.to_anthropic_tools(&tools);
        assert_eq!(anthropic_tools.len(), 1);
        assert_eq!(anthropic_tools[0].name, "test_tool");
    }
}
