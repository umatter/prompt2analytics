//! HTTP client for communicating with p2a-mcp backend
//!
//! Uses platform abstraction to work on web (fetch API) and native (reqwest)

use super::types::*;
use crate::platform::{HttpClient, create_http_client};
use serde::Deserialize;

/// Default API base URL.
///
/// Resolved at *compile time*: a hosted web build bakes in the public backend
/// URL via the `P2A_BACKEND_URL` env var (set by the Pages deploy workflow);
/// every other build (local dev, desktop) leaves it unset and falls back to the
/// localhost backend documented in CLAUDE.md, so behavior is unchanged.
const DEFAULT_BASE_URL: &str = match option_env!("P2A_BACKEND_URL") {
    Some(url) => url,
    None => "http://localhost:8080",
};

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
        let client = create_http_client();
        let response = client.get(&url).await.map_err(|e| e.to_string())?;

        if !response.is_ok() {
            return Err(format!("HTTP error: {}", response.status));
        }

        serde_json::from_str(&response.body).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Perform a POST request with JSON body
    async fn post<T, B>(&self, endpoint: &str, body: &B) -> Result<ApiResponse<T>, String>
    where
        T: for<'de> serde::Deserialize<'de>,
        B: serde::Serialize,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let body_json = serde_json::to_string(body).map_err(|e| e.to_string())?;
        let client = create_http_client();
        let response = client
            .post(&url, &body_json)
            .await
            .map_err(|e| e.to_string())?;

        if !response.is_ok() {
            return Err(format!("HTTP error: {}", response.status));
        }

        serde_json::from_str(&response.body).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Perform a PUT request with JSON body
    async fn put<T, B>(&self, endpoint: &str, body: &B) -> Result<ApiResponse<T>, String>
    where
        T: for<'de> serde::Deserialize<'de>,
        B: serde::Serialize,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let body_json = serde_json::to_string(body).map_err(|e| e.to_string())?;
        let client = create_http_client();
        let response = client
            .put(&url, &body_json)
            .await
            .map_err(|e| e.to_string())?;

        if !response.is_ok() {
            return Err(format!("HTTP error: {}", response.status));
        }

        serde_json::from_str(&response.body).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Perform a DELETE request
    async fn delete(&self, endpoint: &str) -> Result<(), String> {
        let url = format!("{}{}", self.base_url, endpoint);
        let client = create_http_client();
        let response = client.delete(&url).await.map_err(|e| e.to_string())?;

        if !response.is_ok() {
            return Err(format!("HTTP error: {}", response.status));
        }

        Ok(())
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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

        let response: ApiResponse<ToolExecutionResult> = self
            .post(&format!("/api/tools/{tool_name}"), &request)
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No tool result in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // === Health endpoint ===

    /// Check server health
    pub async fn health(&self) -> Result<HealthResponse, String> {
        let url = format!("{}/health", self.base_url);
        let client = create_http_client();
        let response = client.get(&url).await.map_err(|e| e.to_string())?;

        if !response.is_ok() {
            return Err(format!("HTTP error: {}", response.status));
        }

        serde_json::from_str(&response.body).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Get the base URL for SSE streaming
    pub fn streaming_url(&self, endpoint: &str) -> String {
        format!("{}{}", self.base_url, endpoint)
    }

    /// Get the configured backend base URL. Used by callers that talk
    /// directly to the platform streaming layer (which takes a `&str`
    /// rather than an `ApiClient`), so the URL doesn't have to be
    /// duplicated.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    // === Conversation endpoints ===

    /// List all conversations for a session
    pub async fn list_conversations(&self, session_id: &str) -> Result<Vec<Conversation>, String> {
        let response: ApiResponse<Vec<Conversation>> = self
            .get(&format!("/api/conversations?session_id={session_id}"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversations in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Create a new conversation
    pub async fn create_conversation(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<Conversation, String> {
        let request = CreateConversationRequest {
            session_id: session_id.to_string(),
            title: title.to_string(),
        };

        let response: ApiResponse<Conversation> = self.post("/api/conversations", &request).await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            .put(&format!("/api/conversations/{conversation_id}"), &request)
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No conversation in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
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
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // === LLM endpoints ===

    /// Generate a conversation title using the LLM
    pub async fn generate_title(
        &self,
        user_message: &str,
        assistant_response: Option<&str>,
        provider: Option<ProviderConfig>,
    ) -> Result<String, String> {
        let request = GenerateTitleRequest {
            user_message: user_message.to_string(),
            assistant_response: assistant_response.map(|s| s.to_string()),
            provider,
        };

        let response: ApiResponse<GenerateTitleResponse> =
            self.post("/api/llm/generate-title", &request).await?;

        if response.success {
            response
                .data
                .map(|r| r.title)
                .ok_or_else(|| "No title in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // === Tool Call History endpoints ===

    /// Get all tool calls for a conversation
    pub async fn get_conversation_tool_calls(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<PersistedToolCall>, String> {
        let response: ApiResponse<Vec<PersistedToolCall>> = self
            .get(&format!("/api/conversations/{conversation_id}/tool-calls"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No tool calls in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Get tool calls for a specific message
    pub async fn get_message_tool_calls(
        &self,
        message_id: &str,
    ) -> Result<Vec<PersistedToolCall>, String> {
        let response: ApiResponse<Vec<PersistedToolCall>> = self
            .get(&format!("/api/messages/{message_id}/tool-calls"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No tool calls in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // === Dataset Metadata endpoints ===

    /// List all dataset metadata for a session
    pub async fn list_session_datasets(
        &self,
        session_id: &str,
    ) -> Result<Vec<DatasetMeta>, String> {
        let response: ApiResponse<Vec<DatasetMeta>> = self
            .get(&format!("/api/sessions/{session_id}/datasets"))
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No datasets in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Reload all datasets for a session from their original source paths
    pub async fn reload_session_datasets(&self, session_id: &str) -> Result<ReloadResult, String> {
        let response: ApiResponse<ReloadResult> = self
            .post(
                &format!("/api/sessions/{session_id}/datasets/reload"),
                &serde_json::json!({}),
            )
            .await?;

        if response.success {
            response
                .data
                .ok_or_else(|| "No reload result in response".to_string())
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "Unknown error".to_string()))
        }
    }
}

/// Global API client instance
pub fn api() -> ApiClient {
    ApiClient::new()
}

/// The origin (`scheme://host[:port]`) of a backend base URL, for use as a
/// CSP source expression.
///
/// Returns `None` unless the URL parses with an `http` or `https` scheme and
/// its serialized origin consists only of `[A-Za-z0-9.:/\-\[\]]`, so a
/// malformed `P2A_BACKEND_URL` can never inject extra CSP tokens or
/// directives.
pub fn backend_origin(base_url: &str) -> Option<String> {
    let url = reqwest::Url::parse(base_url.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let origin = url.origin().ascii_serialization();
    let safe = !origin.is_empty()
        && origin
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '/' | '-' | '[' | ']'));
    safe.then_some(origin)
}

/// The runtime `connect-src` directive: `'self'` plus the backend origin
/// when it is a valid http(s) origin.
pub fn connect_src_policy(base_url: &str) -> String {
    match backend_origin(base_url) {
        Some(origin) => format!("connect-src 'self' {origin}"),
        None => "connect-src 'self'".to_string(),
    }
}

/// Narrow `connect-src` to the compiled-in backend.
///
/// `index.html` ships a static CSP whose `connect-src` must also allow the
/// backend, which is only known at compile time (`P2A_BACKEND_URL`). This
/// appends a second CSP `<meta>` with `connect-src 'self' <backend origin>`;
/// browsers enforce every policy, so the effective `connect-src` is the
/// intersection. Call it before `dioxus::launch`, so nothing untrusted has
/// rendered yet. On non-wasm targets it is a no-op.
pub fn install_backend_csp() {
    #[cfg(target_arch = "wasm32")]
    {
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            tracing::warn!("CSP: no document; runtime connect-src narrowing skipped");
            return;
        };
        match document.query_selector("meta[data-p2a-csp]") {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("CSP: meta lookup failed: {e:?}");
                return;
            }
        }
        let head = match document.query_selector("head") {
            Ok(Some(head)) => head,
            Ok(None) => {
                tracing::warn!("CSP: no <head>; runtime connect-src narrowing skipped");
                return;
            }
            Err(e) => {
                tracing::warn!("CSP: <head> lookup failed: {e:?}");
                return;
            }
        };
        let meta = match document.create_element("meta") {
            Ok(meta) => meta,
            Err(e) => {
                tracing::warn!("CSP: could not create <meta>: {e:?}");
                return;
            }
        };
        let policy = connect_src_policy(DEFAULT_BASE_URL);
        for (name, value) in [
            ("http-equiv", "Content-Security-Policy"),
            ("content", policy.as_str()),
            ("data-p2a-csp", "backend"),
        ] {
            if let Err(e) = meta.set_attribute(name, value) {
                tracing::warn!("CSP: could not set {name}: {e:?}");
                return;
            }
        }
        if let Err(e) = head.append_child(&meta) {
            tracing::warn!("CSP: could not append <meta>: {e:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_origin_of_base_urls() {
        assert_eq!(
            backend_origin("https://p2a-api.qamelab.org").as_deref(),
            Some("https://p2a-api.qamelab.org")
        );
        assert_eq!(
            backend_origin("https://p2a-api.qamelab.org/").as_deref(),
            Some("https://p2a-api.qamelab.org")
        );
        assert_eq!(
            backend_origin("http://localhost:8080").as_deref(),
            Some("http://localhost:8080")
        );
        assert_eq!(
            backend_origin("https://example.org:443/api").as_deref(),
            Some("https://example.org")
        );
        assert_eq!(backend_origin("not a url"), None);
        assert_eq!(backend_origin("javascript:alert(1)"), None);
    }

    #[test]
    fn connect_src_policy_names_only_self_and_backend() {
        assert_eq!(
            connect_src_policy("https://p2a-api.qamelab.org"),
            "connect-src 'self' https://p2a-api.qamelab.org"
        );
        assert_eq!(
            connect_src_policy("http://localhost:8080/"),
            "connect-src 'self' http://localhost:8080"
        );
        assert_eq!(connect_src_policy("not a url"), "connect-src 'self'");
        for hostile in [
            "https://a.example;x",
            "https://a.example; script-src *",
            "https://a example",
        ] {
            let p = connect_src_policy(hostile);
            assert!(
                p.starts_with("connect-src 'self'") && !p.contains(';'),
                "{p}"
            );
        }
    }

    #[test]
    fn index_html_ships_a_restrictive_csp() {
        let html = include_str!("../../index.html");
        let at = html
            .find("http-equiv=\"Content-Security-Policy\"")
            .expect("CSP meta present");
        assert!(
            at < html.find("<link").unwrap(),
            "CSP must precede the stylesheet link"
        );
        let content = &html[at..];
        let start = content.find("content=\"").unwrap() + "content=\"".len();
        let policy = &content[start..start + content[start..].find('"').unwrap()];
        let d = |name: &str| {
            policy
                .split(';')
                .map(str::trim)
                .find(|x| x.split_whitespace().next() == Some(name))
                .unwrap_or_else(|| panic!("missing {name}"))
                .to_string()
        };
        assert_eq!(d("default-src"), "default-src 'self'");
        assert_eq!(d("script-src"), "script-src 'self' 'wasm-unsafe-eval'");
        assert_eq!(d("img-src"), "img-src 'self' data:");
        assert_eq!(d("object-src"), "object-src 'none'");
        assert_eq!(d("base-uri"), "base-uri 'none'");
        assert!(d("connect-src").starts_with("connect-src 'self'"));
    }
}
