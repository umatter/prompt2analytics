//! Platform-agnostic SSE streaming abstraction
//!
//! - Web: Uses web_sys ReadableStream
//! - Native: Uses reqwest with streaming response

use crate::api::types::{LlmChatRequest, Message, ProviderConfig, StreamEvent};

/// Streaming error type
#[derive(Debug, Clone)]
pub struct StreamError(pub String);

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StreamError {}

/// Parse a single SSE line into a StreamEvent
fn parse_sse_line(line: &str) -> Option<StreamEvent> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return None;
    }

    // Parse data lines
    if let Some(data) = line.strip_prefix("data: ") {
        // Try to parse as JSON
        match serde_json::from_str::<StreamEvent>(data) {
            Ok(event) => return Some(event),
            Err(e) => {
                tracing::warn!("Failed to parse SSE event: {} - data: {}", e, data);
                return None;
            }
        }
    }

    None
}

// ============================================================================
// Web implementation (WASM with ReadableStream)
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub mod web {
    use super::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    /// Stream chat with the LLM backend (web version)
    pub async fn stream_chat<F>(
        base_url: &str,
        session_id: &str,
        message: &str,
        history: Vec<Message>,
        provider: ProviderConfig,
        interpret: bool,
        conversation_id: Option<String>,
        mut on_event: F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(StreamEvent),
    {
        tracing::debug!("[SSE] stream_chat started");
        let url = format!("{}/api/llm/chat/stream", base_url.trim_end_matches('/'));
        tracing::debug!("[SSE] URL: {}", url);

        let request_body = LlmChatRequest {
            session_id: session_id.to_string(),
            message: message.to_string(),
            provider: Some(provider),
            history: Some(history),
            interpret,
            conversation_id,
        };

        let body_json =
            serde_json::to_string(&request_body).map_err(|e| StreamError(e.to_string()))?;
        tracing::debug!("[SSE] Request body length: {}", body_json.len());

        // Create fetch request
        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body_json));

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| StreamError(format!("{e:?}")))?;

        request
            .headers()
            .set("Content-Type", "application/json")
            .map_err(|e| StreamError(format!("{e:?}")))?;

        tracing::debug!("[SSE] Executing fetch");
        // Execute fetch
        let window =
            web_sys::window().ok_or_else(|| StreamError("No window object".to_string()))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| StreamError(format!("Fetch error: {e:?}")))?;
        tracing::debug!("[SSE] Got response");

        let response: Response = resp_value
            .dyn_into()
            .map_err(|_| StreamError("Response is not a Response object".to_string()))?;

        tracing::debug!("[SSE] Response status: {}", response.status());
        if !response.ok() {
            return Err(StreamError(format!(
                "HTTP error: {} {}",
                response.status(),
                response.status_text()
            )));
        }

        // Get the body as a ReadableStream and read it
        let body = response
            .body()
            .ok_or_else(|| StreamError("Response has no body".to_string()))?;
        tracing::debug!("[SSE] Got body, getting reader");

        // Use the BYOB reader approach with JS interop
        let reader = body.get_reader();

        let mut buffer = String::new();
        tracing::debug!("[SSE] Starting read loop");

        loop {
            // Read from the stream
            let read_promise = js_sys::Reflect::get(&reader, &JsValue::from_str("read"))
                .map_err(|e| StreamError(format!("Failed to get read method: {e:?}")))?;

            let read_fn: js_sys::Function = read_promise
                .dyn_into()
                .map_err(|_| StreamError("read is not a function".to_string()))?;

            let read_result = read_fn
                .call0(&reader)
                .map_err(|e| StreamError(format!("Failed to call read: {e:?}")))?;

            let read_promise: js_sys::Promise = read_result
                .dyn_into()
                .map_err(|_| StreamError("read result is not a promise".to_string()))?;

            let chunk_result = JsFuture::from(read_promise)
                .await
                .map_err(|e| StreamError(format!("Read error: {e:?}")))?;

            // Check if done
            let done = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("done"))
                .map_err(|_| StreamError("Failed to get done".to_string()))?
                .as_bool()
                .unwrap_or(true);

            if done {
                tracing::debug!("[SSE] Stream done");
                break;
            }

            // Get the value
            let value = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("value"))
                .map_err(|_| StreamError("Failed to get value".to_string()))?;

            if value.is_undefined() {
                continue;
            }

            // Convert Uint8Array to string
            let array: js_sys::Uint8Array = value
                .dyn_into()
                .map_err(|_| StreamError("Value is not Uint8Array".to_string()))?;

            let bytes = array.to_vec();
            let text = String::from_utf8_lossy(&bytes);
            tracing::debug!("[SSE] Got chunk: {} bytes", bytes.len());
            buffer.push_str(&text);

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if let Some(event) = parse_sse_line(&line) {
                    tracing::debug!("[SSE] Parsed event, calling callback");
                    on_event(event);
                }
            }
        }

        // Process remaining buffer
        if !buffer.is_empty() {
            if let Some(event) = parse_sse_line(&buffer) {
                on_event(event);
            }
        }

        tracing::debug!("[SSE] stream_chat completed");
        Ok(())
    }

    /// Abort controller wrapper for cancelling requests
    #[derive(Clone)]
    pub struct StreamAbortController {
        controller: web_sys::AbortController,
    }

    impl StreamAbortController {
        /// Create a new abort controller
        pub fn new() -> Result<Self, StreamError> {
            let controller = web_sys::AbortController::new()
                .map_err(|e| StreamError(format!("Failed to create AbortController: {e:?}")))?;
            Ok(Self { controller })
        }

        /// Get the abort signal for use with fetch
        pub fn signal(&self) -> web_sys::AbortSignal {
            self.controller.signal()
        }

        /// Abort the request
        pub fn abort(&self) {
            self.controller.abort();
        }
    }

    impl Default for StreamAbortController {
        fn default() -> Self {
            Self::new().expect("Failed to create AbortController")
        }
    }
}

// ============================================================================
// Native implementation (reqwest with streaming)
// ============================================================================

#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "desktop", feature = "mobile")
))]
pub mod native {
    use super::*;
    use futures::StreamExt;

    /// Stream chat with the LLM backend (native version)
    pub async fn stream_chat<F>(
        base_url: &str,
        session_id: &str,
        message: &str,
        history: Vec<Message>,
        provider: ProviderConfig,
        interpret: bool,
        conversation_id: Option<String>,
        mut on_event: F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(StreamEvent),
    {
        tracing::debug!("[SSE] stream_chat started");
        let url = format!("{}/api/llm/chat/stream", base_url.trim_end_matches('/'));
        tracing::debug!("[SSE] URL: {}", url);

        let request_body = LlmChatRequest {
            session_id: session_id.to_string(),
            message: message.to_string(),
            provider: Some(provider),
            history: Some(history),
            interpret,
            conversation_id,
        };

        let body_json =
            serde_json::to_string(&request_body).map_err(|e| StreamError(e.to_string()))?;
        tracing::debug!("[SSE] Request body length: {}", body_json.len());

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body_json)
            .send()
            .await
            .map_err(|e| StreamError(format!("Fetch error: {}", e)))?;

        tracing::debug!("[SSE] Response status: {}", response.status());
        if !response.status().is_success() {
            return Err(StreamError(format!("HTTP error: {}", response.status())));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        tracing::debug!("[SSE] Starting read loop");

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| StreamError(format!("Read error: {}", e)))?;
            let text = String::from_utf8_lossy(&bytes);
            tracing::debug!("[SSE] Got chunk: {} bytes", bytes.len());
            buffer.push_str(&text);

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if let Some(event) = parse_sse_line(&line) {
                    tracing::debug!("[SSE] Parsed event, calling callback");
                    on_event(event);
                }
            }
        }

        // Process remaining buffer
        if !buffer.is_empty()
            && let Some(event) = parse_sse_line(&buffer)
        {
            on_event(event);
        }

        tracing::debug!("[SSE] stream_chat completed");
        Ok(())
    }

    /// Abort controller placeholder for native (uses tokio cancellation)
    #[derive(Clone, Default)]
    pub struct StreamAbortController {
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl StreamAbortController {
        pub fn new() -> Result<Self, StreamError> {
            Ok(Self {
                cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            })
        }

        pub fn abort(&self) {
            self.cancelled
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        pub fn is_cancelled(&self) -> bool {
            self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
}

// ============================================================================
// Platform-specific re-exports
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub use web::stream_chat;

#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "desktop", feature = "mobile")
))]
pub use native::stream_chat;
