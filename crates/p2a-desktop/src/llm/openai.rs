//! OpenAI GPT LLM provider implementation.
//!
//! Uses the async-openai crate for OpenAI API integration with function calling support.

use super::{
    LlmError, LlmProvider, Message, MessageRole, ProviderConfig, ProviderType, StreamChunk,
    ToolCall, ToolDefinition, ToolExecutor, ToolResult,
};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTool, ChatCompletionToolType, CreateChatCompletionRequestArgs, FunctionCall,
        FunctionObjectArgs,
    },
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

/// OpenAI GPT provider.
pub struct OpenAIProvider {
    config: ProviderConfig,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
    }

    /// Get a configured client.
    fn client(&self) -> Result<Client<OpenAIConfig>, LlmError> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or(LlmError::InvalidApiKey)?;

        let mut config = OpenAIConfig::new().with_api_key(api_key);

        if let Some(ref base_url) = self.config.base_url {
            config = config.with_api_base(base_url);
        }

        Ok(Client::with_config(config))
    }

    /// Convert our Message format to OpenAI's message format.
    fn to_openai_messages(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ChatCompletionRequestMessage>, LlmError> {
        let mut result = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    result.push(
                        ChatCompletionRequestSystemMessageArgs::default()
                            .content(msg.content.clone())
                            .build()
                            .map_err(|e| LlmError::SerializationError(e.to_string()))?
                            .into(),
                    );
                }
                MessageRole::User => {
                    result.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(msg.content.clone())
                            .build()
                            .map_err(|e| LlmError::SerializationError(e.to_string()))?
                            .into(),
                    );
                }
                MessageRole::Assistant => {
                    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();

                    if !msg.content.is_empty() {
                        builder.content(msg.content.clone());
                    }

                    if let Some(ref tool_calls) = msg.tool_calls {
                        let openai_tool_calls: Vec<ChatCompletionMessageToolCall> = tool_calls
                            .iter()
                            .map(|tc| ChatCompletionMessageToolCall {
                                id: tc.id.clone(),
                                r#type: ChatCompletionToolType::Function,
                                function: FunctionCall {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                },
                            })
                            .collect();
                        builder.tool_calls(openai_tool_calls);
                    }

                    result.push(
                        builder
                            .build()
                            .map_err(|e| LlmError::SerializationError(e.to_string()))?
                            .into(),
                    );
                }
                MessageRole::Tool => {
                    // Tool results as separate tool messages
                    if let Some(ref tool_results) = msg.tool_results {
                        for tr in tool_results {
                            result.push(
                                ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(&tr.tool_call_id)
                                    .content(tr.content.clone())
                                    .build()
                                    .map_err(|e| LlmError::SerializationError(e.to_string()))?
                                    .into(),
                            );
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Convert tool definitions to OpenAI's format.
    fn to_openai_tools(&self, tools: &[ToolDefinition]) -> Result<Vec<ChatCompletionTool>, LlmError> {
        tools
            .iter()
            .map(|t| {
                let function = FunctionObjectArgs::default()
                    .name(&t.name)
                    .description(&t.description)
                    .parameters(t.parameters.clone())
                    .build()
                    .map_err(|e| LlmError::SerializationError(e.to_string()))?;

                Ok(ChatCompletionTool {
                    r#type: ChatCompletionToolType::Function,
                    function,
                })
            })
            .collect()
    }

    /// Make a single API call to OpenAI.
    async fn call_api(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let client = self.client()?;
        let openai_messages = self.to_openai_messages(messages)?;

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.config.model)
            .messages(openai_messages);

        if let Some(temp) = self.config.temperature {
            request_builder.temperature(temp);
        }

        if let Some(max_tokens) = self.config.max_tokens {
            request_builder.max_tokens(max_tokens as u16);
        }

        if !tools.is_empty() {
            let openai_tools = self.to_openai_tools(tools)?;
            request_builder.tools(openai_tools);
        }

        let request = request_builder
            .build()
            .map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let response = client
            .chat()
            .create(request)
            .await
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("401") || err_str.contains("Invalid API Key") {
                    LlmError::InvalidApiKey
                } else if err_str.contains("429") {
                    LlmError::RateLimited
                } else {
                    LlmError::ApiError(err_str)
                }
            })?;

        // Parse the first choice
        let choice = response.choices.first().ok_or_else(|| {
            LlmError::ApiError("No response choices returned".to_string())
        })?;

        let content = choice
            .message
            .content
            .clone()
            .unwrap_or_default();

        let tool_calls = choice.message.tool_calls.as_ref().map(|tcs| {
            tcs.iter()
                .map(|tc| {
                    let arguments: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                    ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments,
                    }
                })
                .collect()
        });

        Ok(Message {
            role: MessageRole::Assistant,
            content,
            tool_calls,
            tool_results: None,
        })
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

            // No tool calls - this is the final response
            return Ok(response);
        }
    }

    /// Make a streaming API call to OpenAI.
    async fn call_api_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        callback: &(dyn Fn(StreamChunk) + Send + Sync),
    ) -> Result<Message, LlmError> {
        let client = self.client()?;
        let openai_messages = self.to_openai_messages(messages)?;

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.config.model)
            .messages(openai_messages);

        if let Some(temp) = self.config.temperature {
            request_builder.temperature(temp);
        }

        if let Some(max_tokens) = self.config.max_tokens {
            request_builder.max_tokens(max_tokens as u16);
        }

        if !tools.is_empty() {
            let openai_tools = self.to_openai_tools(tools)?;
            request_builder.tools(openai_tools);
        }

        let request = request_builder
            .build()
            .map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let mut stream = client
            .chat()
            .create_stream(request)
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // Track partial tool calls being built
        let mut partial_tool_calls: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    for choice in response.choices {
                        // Handle content delta
                        if let Some(content) = choice.delta.content {
                            full_content.push_str(&content);
                            callback(StreamChunk::Text { content });
                        }

                        // Handle tool call deltas
                        if let Some(tcs) = choice.delta.tool_calls {
                            for tc_delta in tcs {
                                let index = tc_delta.index as usize;
                                let entry = partial_tool_calls.entry(index).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });

                                if let Some(id) = tc_delta.id {
                                    entry.0 = id;
                                }
                                if let Some(function) = tc_delta.function {
                                    if let Some(name) = function.name {
                                        entry.1 = name;
                                    }
                                    if let Some(args) = function.arguments {
                                        entry.2.push_str(&args);
                                    }
                                }
                            }
                        }

                        // Check for finish reason
                        if let Some(finish_reason) = choice.finish_reason {
                            if finish_reason == async_openai::types::FinishReason::ToolCalls {
                                // Finalize tool calls
                                for (_, (id, name, args)) in partial_tool_calls.drain() {
                                    let arguments: Value =
                                        serde_json::from_str(&args).unwrap_or(Value::Null);
                                    let tc = ToolCall {
                                        id,
                                        name,
                                        arguments,
                                    };
                                    callback(StreamChunk::ToolCall {
                                        tool_call: tc.clone(),
                                    });
                                    tool_calls.push(tc);
                                }
                            }
                            callback(StreamChunk::Done);
                        }
                    }
                }
                Err(e) => {
                    callback(StreamChunk::Error {
                        message: e.to_string(),
                    });
                    return Err(LlmError::ApiError(e.to_string()));
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
impl LlmProvider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn is_available(&self) -> Result<bool, LlmError> {
        // Check if API key is configured
        Ok(self.config.api_key.is_some())
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Try to list models from the API
        let client = self.client()?;

        match client.models().list().await {
            Ok(response) => {
                let models: Vec<String> = response
                    .data
                    .into_iter()
                    .filter(|m| m.id.starts_with("gpt-"))
                    .map(|m| m.id)
                    .collect();
                Ok(models)
            }
            Err(_) => {
                // Return known models as fallback
                Ok(vec![
                    "gpt-4o".to_string(),
                    "gpt-4o-mini".to_string(),
                    "gpt-4-turbo".to_string(),
                    "gpt-4".to_string(),
                    "gpt-3.5-turbo".to_string(),
                ])
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let config = ProviderConfig {
            provider_type: ProviderType::OpenAI,
            api_key: Some("test-key".to_string()),
            base_url: None,
            model: "gpt-4o".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        let provider = OpenAIProvider::new(config);
        assert_eq!(provider.provider_type(), ProviderType::OpenAI);
    }

    #[test]
    fn test_message_conversion() {
        let config = ProviderConfig {
            provider_type: ProviderType::OpenAI,
            api_key: Some("test-key".to_string()),
            base_url: None,
            model: "gpt-4o".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        let provider = OpenAIProvider::new(config);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let openai_messages = provider.to_openai_messages(&messages).unwrap();
        assert_eq!(openai_messages.len(), 3);
    }
}
