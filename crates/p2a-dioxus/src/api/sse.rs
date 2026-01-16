//! Server-Sent Events (SSE) client for streaming LLM responses
//!
//! This module handles streaming chat responses from the p2a-mcp backend.

use super::types::{LlmChatRequest, Message, ProviderConfig, StreamEvent};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Stream chat with the LLM backend
///
/// This function initiates a streaming chat request and calls the callback
/// for each event received from the server.
pub async fn stream_chat<F>(
    base_url: &str,
    session_id: &str,
    message: &str,
    history: Vec<Message>,
    provider: ProviderConfig,
    interpret: bool,
    mut on_event: F,
) -> Result<(), String>
where
    F: FnMut(StreamEvent),
{
    log("[SSE] stream_chat started");
    let url = format!("{}/api/llm/chat/stream", base_url.trim_end_matches('/'));
    log(&format!("[SSE] URL: {}", url));

    let request_body = LlmChatRequest {
        session_id: session_id.to_string(),
        message: message.to_string(),
        provider: Some(provider),
        history: Some(history),
        interpret,
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| e.to_string())?;
    log(&format!("[SSE] Request body length: {}", body_json.len()));

    // Create fetch request
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(&body_json));

    let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;

    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(|e| format!("{e:?}"))?;

    log("[SSE] Executing fetch");
    // Execute fetch
    let window = web_sys::window().ok_or("No window object")?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch error: {e:?}"))?;
    log("[SSE] Got response");

    let response: Response = resp_value
        .dyn_into()
        .map_err(|_| "Response is not a Response object")?;

    log(&format!("[SSE] Response status: {}", response.status()));
    if !response.ok() {
        return Err(format!(
            "HTTP error: {} {}",
            response.status(),
            response.status_text()
        ));
    }

    // Get the body as a ReadableStream and read it
    let body = response.body().ok_or("Response has no body")?;
    log("[SSE] Got body, getting reader");

    // Use the BYOB reader approach with JS interop
    let reader = body.get_reader();

    let mut buffer = String::new();
    log("[SSE] Starting read loop");

    loop {
        // Read from the stream
        let read_promise = js_sys::Reflect::get(&reader, &JsValue::from_str("read"))
            .map_err(|e| format!("Failed to get read method: {e:?}"))?;

        let read_fn: js_sys::Function = read_promise
            .dyn_into()
            .map_err(|_| "read is not a function")?;

        let read_result = read_fn
            .call0(&reader)
            .map_err(|e| format!("Failed to call read: {e:?}"))?;

        let read_promise: js_sys::Promise = read_result
            .dyn_into()
            .map_err(|_| "read result is not a promise")?;

        let chunk_result = JsFuture::from(read_promise)
            .await
            .map_err(|e| format!("Read error: {e:?}"))?;

        // Check if done
        let done = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("done"))
            .map_err(|_| "Failed to get done")?
            .as_bool()
            .unwrap_or(true);

        if done {
            log("[SSE] Stream done");
            break;
        }

        // Get the value
        let value = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("value"))
            .map_err(|_| "Failed to get value")?;

        if value.is_undefined() {
            continue;
        }

        // Convert Uint8Array to string
        let array: js_sys::Uint8Array = value
            .dyn_into()
            .map_err(|_| "Value is not Uint8Array")?;

        let bytes = array.to_vec();
        let text = String::from_utf8_lossy(&bytes);
        log(&format!("[SSE] Got chunk: {} bytes", bytes.len()));
        buffer.push_str(&text);

        // Process complete lines
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if let Some(event) = parse_sse_line(&line) {
                log("[SSE] Parsed event, calling callback");
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

    log("[SSE] stream_chat completed");
    Ok(())
}

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

/// Abort controller wrapper for cancelling requests
#[derive(Clone)]
pub struct StreamAbortController {
    controller: web_sys::AbortController,
}

impl StreamAbortController {
    /// Create a new abort controller
    pub fn new() -> Result<Self, String> {
        let controller = web_sys::AbortController::new()
            .map_err(|e| format!("Failed to create AbortController: {e:?}"))?;
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
