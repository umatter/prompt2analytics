//! HTTP client for communicating with p2a-mcp backend

use super::types::*;
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Default API base URL
const DEFAULT_BASE_URL: &str = "http://localhost:8080";

/// API client for p2a-mcp backend
#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
}

impl Default for ApiClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiClient {
    /// Create a new API client with default base URL
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create a new API client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Perform a GET request
    async fn get<T>(&self, endpoint: &str) -> Result<ApiResponse<T>, String>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let response = self.fetch(&url, "GET", None).await?;
        self.parse_response(response).await
    }

    /// Perform a POST request with JSON body
    async fn post<T, B>(&self, endpoint: &str, body: &B) -> Result<ApiResponse<T>, String>
    where
        T: for<'de> serde::Deserialize<'de>,
        B: serde::Serialize,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let body_json = serde_json::to_string(body).map_err(|e| e.to_string())?;
        let response = self.fetch(&url, "POST", Some(&body_json)).await?;
        self.parse_response(response).await
    }

    /// Perform a DELETE request
    async fn delete(&self, endpoint: &str) -> Result<(), String> {
        let url = format!("{}{}", self.base_url, endpoint);
        let _response = self.fetch(&url, "DELETE", None).await?;
        Ok(())
    }

    /// Internal fetch implementation using web-sys
    async fn fetch(
        &self,
        url: &str,
        method: &str,
        body: Option<&str>,
    ) -> Result<Response, String> {
        let opts = RequestInit::new();
        opts.set_method(method);
        opts.set_mode(RequestMode::Cors);

        if let Some(body_str) = body {
            opts.set_body(&JsValue::from_str(body_str));
        }

        let request = Request::new_with_str_and_init(url, &opts).map_err(|e| format!("{e:?}"))?;

        request
            .headers()
            .set("Content-Type", "application/json")
            .map_err(|e| format!("{e:?}"))?;

        let window = web_sys::window().ok_or("No window object")?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| format!("Fetch error: {e:?}"))?;

        let response: Response = resp_value
            .dyn_into()
            .map_err(|_| "Response is not a Response object")?;

        if !response.ok() {
            return Err(format!(
                "HTTP error: {} {}",
                response.status(),
                response.status_text()
            ));
        }

        Ok(response)
    }

    /// Parse JSON response
    async fn parse_response<T>(&self, response: Response) -> Result<ApiResponse<T>, String>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let text = JsFuture::from(response.text().map_err(|e| format!("{e:?}"))?)
            .await
            .map_err(|e| format!("Failed to get response text: {e:?}"))?;

        let text_str = text.as_string().ok_or("Response is not a string")?;

        serde_json::from_str(&text_str).map_err(|e| format!("JSON parse error: {e}"))
    }

    // === Session endpoints ===

    /// Create a new session
    pub async fn create_session(&self) -> Result<String, String> {
        let request = CreateSessionRequest { user_id: None };
        let response: ApiResponse<CreateSessionResponse> =
            self.post("/api/sessions", &request).await?;

        if response.success {
            response
                .data
                .map(|d| d.session_id)
                .ok_or_else(|| "No session ID in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Get session info
    pub async fn get_session(&self, session_id: &str) -> Result<SessionInfo, String> {
        let response: ApiResponse<SessionInfo> =
            self.get(&format!("/api/sessions/{session_id}")).await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No session info in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Delete a session
    pub async fn delete_session(&self, session_id: &str) -> Result<(), String> {
        self.delete(&format!("/api/sessions/{session_id}")).await
    }

    // === Tool endpoints ===

    /// List available tools
    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>, String> {
        let response: ApiResponse<Vec<ToolDefinition>> = self.get("/api/tools").await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No tools in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Call a tool
    pub async fn call_tool(
        &self,
        session_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecutionResult, String> {
        let request = ToolCallRequest {
            session_id: session_id.to_string(),
            arguments,
        };

        let response: ApiResponse<ToolExecutionResult> =
            self.post(&format!("/api/tools/{tool_name}"), &request).await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No tool result in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // === Health endpoint ===

    /// Check server health
    pub async fn health(&self) -> Result<HealthResponse, String> {
        let url = format!("{}/health", self.base_url);
        let response = self.fetch(&url, "GET", None).await?;

        let text = JsFuture::from(response.text().map_err(|e| format!("{e:?}"))?)
            .await
            .map_err(|e| format!("Failed to get response text: {e:?}"))?;

        let text_str = text.as_string().ok_or("Response is not a string")?;

        serde_json::from_str(&text_str).map_err(|e| format!("JSON parse error: {e}"))
    }

    /// Get the base URL for SSE streaming
    pub fn streaming_url(&self, endpoint: &str) -> String {
        format!("{}{}", self.base_url, endpoint)
    }

    // === Conversation endpoints ===

    /// List all conversations for a session
    pub async fn list_conversations(&self, session_id: &str) -> Result<Vec<Conversation>, String> {
        let response: ApiResponse<Vec<Conversation>> = self
            .get(&format!("/api/sessions/{session_id}/conversations"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversations in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Create a new conversation
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, String> {
        let request = CreateConversationRequest {
            title: title.to_string(),
        };

        let response: ApiResponse<Conversation> = self
            .post(
                &format!("/api/sessions/{session_id}/conversations"),
                &request,
            )
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Get a conversation by ID
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Conversation, String> {
        let response: ApiResponse<Conversation> = self
            .get(&format!("/api/conversations/{conversation_id}"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Get a conversation with all messages
    pub async fn get_conversation_with_messages(
        &self,
        conversation_id: &str,
    ) -> Result<ConversationWithMessages, String> {
        let response: ApiResponse<ConversationWithMessages> = self
            .get(&format!("/api/conversations/{conversation_id}/full"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Update a conversation's title
    pub async fn update_conversation(
        &self,
        conversation_id: &str,
        title: Option<&str>,
        is_archived: Option<bool>,
    ) -> Result<Conversation, String> {
        let request = UpdateConversationRequest {
            title: title.map(|s| s.to_string()),
            is_archived,
        };

        let response: ApiResponse<Conversation> = self
            .post(
                &format!("/api/conversations/{conversation_id}"),
                &request,
            )
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Delete a conversation
    pub async fn delete_conversation(&self, conversation_id: &str) -> Result<(), String> {
        self.delete(&format!("/api/conversations/{conversation_id}"))
            .await
    }

    /// Get messages for a conversation
    pub async fn get_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, String> {
        let response: ApiResponse<Vec<ConversationMessage>> = self
            .get(&format!("/api/conversations/{conversation_id}/messages"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No messages in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Add a message to a conversation
    pub async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<ConversationMessage, String> {
        let request = AddMessageRequest {
            role: role.to_string(),
            content: content.to_string(),
        };

        let response: ApiResponse<ConversationMessage> = self
            .post(
                &format!("/api/conversations/{conversation_id}/messages"),
                &request,
            )
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No message in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Clear all messages in a conversation
    pub async fn clear_messages(&self, conversation_id: &str) -> Result<u32, String> {
        #[derive(Deserialize)]
        struct ClearResponse {
            deleted_count: u32,
        }

        let response: ApiResponse<ClearResponse> = self
            .post(
                &format!("/api/conversations/{conversation_id}/messages/clear"),
                &serde_json::json!({}),
            )
            .await?;

        if response.success {
            response
                .data
                .map(|d| d.deleted_count)
                .ok_or_else(|| "No count in response".to_string())
        } else {
            Err(response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }
}

/// Global API client instance
pub fn api() -> ApiClient {
    ApiClient::new()
}
