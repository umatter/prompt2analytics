//! Platform-agnostic SSE streaming abstraction
//!
//! - Web: Uses web_sys ReadableStream, aborted through an AbortController
//! - Native: Uses reqwest with streaming response, dropped to abort
//!
//! Both read through one loop, [`read_events`]: it ends on the stream's end,
//! on the caller's stop ([`stop_pair`]) or when no data arrived for
//! [`IDLE_TIMEOUT_SECS`] (the server sends a keep-alive every 15 s).

use std::future::Future;

use futures::channel::oneshot;
use futures::future::{Either, FutureExt, Shared, select};
use futures::{Stream, StreamExt};

use crate::api::types::{LlmChatRequest, Message, ProviderConfig, StreamEvent};

/// Seconds without any data (events or keep-alives) after which a stream is
/// given up: six of the server's 15 s keep-alives.
pub const IDLE_TIMEOUT_SECS: u64 = 90;

/// How a stream failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamErrorKind {
    /// The request or the read failed, or the server refused it.
    Failed,
    /// The user stopped it.
    Stopped,
    /// No data for [`IDLE_TIMEOUT_SECS`].
    Idle,
}

/// Streaming error type
#[derive(Debug, Clone)]
pub struct StreamError {
    kind: StreamErrorKind,
    message: String,
}

impl StreamError {
    /// A failed request or read.
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            kind: StreamErrorKind::Failed,
            message: message.into(),
        }
    }

    /// The user stopped the answer.
    pub fn stopped() -> Self {
        Self {
            kind: StreamErrorKind::Stopped,
            message: "Answer stopped.".to_string(),
        }
    }

    /// The stream went silent.
    pub fn idle() -> Self {
        Self {
            kind: StreamErrorKind::Idle,
            message: format!(
                "No data from the server for {IDLE_TIMEOUT_SECS} s; the answer was given up."
            ),
        }
    }

    pub fn kind(&self) -> StreamErrorKind {
        self.kind
    }
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for StreamError {}

/// How a stream that was read to its end ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamOutcome {
    /// Whether a final event (`done` or `error`) arrived; a stream that ends
    /// without one was cut off.
    pub done: bool,
}

/// The shared state of a stop handle and its signal.
struct StopInner {
    fired: std::cell::Cell<bool>,
    tx: std::cell::RefCell<Option<oneshot::Sender<()>>>,
}

/// The handle that stops a stream (the Stop button holds it). Cloning
/// shares it; dropping every clone does not stop the stream.
#[derive(Clone)]
pub struct StreamStop(std::rc::Rc<StopInner>);

impl StreamStop {
    /// Stop the stream (once; later calls do nothing).
    pub fn stop(&self) {
        self.0.fired.set(true);
        if let Some(tx) = self.0.tx.borrow_mut().take() {
            let _ = tx.send(());
        }
    }
}

impl PartialEq for StreamStop {
    fn eq(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.0, &other.0)
    }
}

/// The stream's side of a [`StreamStop`].
#[derive(Clone)]
pub struct StopSignal {
    inner: std::rc::Rc<StopInner>,
    rx: Shared<oneshot::Receiver<()>>,
}

impl StopSignal {
    /// Resolves when the stop is fired; never when the handle is dropped.
    pub async fn fired(&self) {
        if self.rx.clone().await.is_err() {
            futures::future::pending::<()>().await;
        }
    }

    /// Whether the stop has been fired already (no polling needed).
    pub fn is_fired(&self) -> bool {
        self.inner.fired.get()
    }
}

/// A connected stop handle and signal.
pub fn stop_pair() -> (StreamStop, StopSignal) {
    let (tx, rx) = oneshot::channel();
    let inner = std::rc::Rc::new(StopInner {
        fired: std::cell::Cell::new(false),
        tx: std::cell::RefCell::new(Some(tx)),
    });
    (
        StreamStop(inner.clone()),
        StopSignal {
            inner,
            rx: rx.shared(),
        },
    )
}

/// One step of [`read_events`].
enum ReadStep {
    Chunk(Option<Result<Vec<u8>, StreamError>>),
    Stopped,
    Idle,
}

/// Read SSE events from `bytes` and pass each to `on_event`, until the
/// stream ends, `stop` fires (`Stopped`) or the stream waits longer than a
/// timer from `next_timer` (`Idle`). A timer is armed for each wait for
/// data, so a stream that delivers without waiting costs no timer and no
/// stop poll; the stop is then seen after each chunk. A line is decoded
/// only once complete, so a UTF-8 character split across chunks stays
/// intact.
pub async fn read_events<S, T, TF>(
    mut bytes: S,
    stop: StopSignal,
    mut next_timer: TF,
    mut on_event: impl FnMut(StreamEvent),
) -> Result<StreamOutcome, StreamError>
where
    S: Stream<Item = Result<Vec<u8>, StreamError>> + Unpin,
    TF: FnMut() -> T,
    T: Future<Output = ()>,
{
    use std::task::Poll;

    let mut outcome = StreamOutcome::default();
    let mut deliver = |event: StreamEvent, outcome: &mut StreamOutcome| {
        if matches!(event, StreamEvent::Done { .. } | StreamEvent::Error { .. }) {
            outcome.done = true;
        }
        on_event(event);
    };
    let stopped = stop.fired();
    futures::pin_mut!(stopped);
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        if stop.is_fired() {
            return Err(StreamError::stopped());
        }
        let mut timer: Option<std::pin::Pin<Box<T>>> = None;
        let step = futures::future::poll_fn(|cx| {
            if let Poll::Ready(item) = bytes.poll_next_unpin(cx) {
                return Poll::Ready(ReadStep::Chunk(item));
            }
            if stopped.as_mut().poll(cx).is_ready() {
                return Poll::Ready(ReadStep::Stopped);
            }
            let timer = timer.get_or_insert_with(|| Box::pin(next_timer()));
            if timer.as_mut().poll(cx).is_ready() {
                return Poll::Ready(ReadStep::Idle);
            }
            Poll::Pending
        })
        .await;
        match step {
            ReadStep::Stopped => return Err(StreamError::stopped()),
            ReadStep::Idle => return Err(StreamError::idle()),
            ReadStep::Chunk(None) => break,
            ReadStep::Chunk(Some(chunk)) => {
                let chunk = chunk?;
                drain_complete_lines(&mut buffer, &chunk, |line| {
                    if let Some(event) = parse_sse_line(&line) {
                        deliver(event, &mut outcome);
                    }
                });
            }
        }
    }
    // A trailing line with no final newline.
    if let Some(line) = decode_trailing(&buffer)
        && let Some(event) = parse_sse_line(&line)
    {
        deliver(event, &mut outcome);
    }
    Ok(outcome)
}

/// Headers for the chat-stream request: `Content-Type: application/json`,
/// plus `Authorization: Bearer <token>` when a backend access token is set.
/// Header values are never logged.
fn stream_request_headers(token: Option<&str>) -> Vec<(String, String)> {
    let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
    headers.extend(crate::platform::authorization_header(token));
    headers
}

/// The chat-stream request body.
fn request_body(
    session_id: &str,
    message: &str,
    history: Vec<Message>,
    provider: ProviderConfig,
    interpret: bool,
    conversation_id: Option<String>,
) -> Result<String, StreamError> {
    let request_body = LlmChatRequest {
        session_id: session_id.to_string(),
        message: message.to_string(),
        provider: Some(provider),
        history: Some(history),
        interpret,
        conversation_id,
        retrieve_history: true,
    };
    serde_json::to_string(&request_body).map_err(|e| StreamError::failed(e.to_string()))
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

/// Append `chunk` to the raw byte buffer and invoke `on_line` for each complete
/// newline-terminated line.
///
/// Decoding is deferred until a full line is present, so a multi-byte UTF-8
/// character split across two network chunks is never decoded until all of its
/// bytes have arrived. This avoids the corruption caused by calling
/// `String::from_utf8_lossy` on each raw chunk independently (which would
/// replace the split character's leading bytes with U+FFFD).
fn drain_complete_lines(buf: &mut Vec<u8>, chunk: &[u8], mut on_line: impl FnMut(String)) {
    buf.extend_from_slice(chunk);
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
        // Strip the trailing '\n' and an optional preceding '\r'.
        let mut end = line_bytes.len() - 1;
        if end > 0 && line_bytes[end - 1] == b'\r' {
            end -= 1;
        }
        on_line(String::from_utf8_lossy(&line_bytes[..end]).into_owned());
    }
}

/// Decode any bytes left in the buffer after the stream ends (a final line with
/// no trailing newline).
fn decode_trailing(buf: &[u8]) -> Option<String> {
    if buf.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(buf).into_owned())
    }
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

    /// The idle timer of the web build.
    fn idle_timer() -> gloo_timers::future::TimeoutFuture {
        gloo_timers::future::TimeoutFuture::new((IDLE_TIMEOUT_SECS * 1000) as u32)
    }

    /// Read one chunk from the body reader: `None` at the end.
    async fn read_chunk(reader: &JsValue) -> Option<Result<Vec<u8>, StreamError>> {
        let step = async {
            let read_fn: js_sys::Function =
                js_sys::Reflect::get(reader, &JsValue::from_str("read"))
                    .map_err(|e| StreamError::failed(format!("Failed to get read method: {e:?}")))?
                    .dyn_into()
                    .map_err(|_| StreamError::failed("read is not a function"))?;
            let read_promise: js_sys::Promise = read_fn
                .call0(reader)
                .map_err(|e| StreamError::failed(format!("Failed to call read: {e:?}")))?
                .dyn_into()
                .map_err(|_| StreamError::failed("read result is not a promise"))?;
            let chunk_result = JsFuture::from(read_promise)
                .await
                .map_err(|e| StreamError::failed(format!("Read error: {e:?}")))?;
            let done = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("done"))
                .map_err(|_| StreamError::failed("Failed to get done"))?
                .as_bool()
                .unwrap_or(true);
            if done {
                return Ok(None);
            }
            let value = js_sys::Reflect::get(&chunk_result, &JsValue::from_str("value"))
                .map_err(|_| StreamError::failed("Failed to get value"))?;
            if value.is_undefined() {
                return Ok(Some(Vec::new()));
            }
            let array: js_sys::Uint8Array = value
                .dyn_into()
                .map_err(|_| StreamError::failed("Value is not Uint8Array"))?;
            Ok(Some(array.to_vec()))
        };
        match step.await {
            Ok(Some(bytes)) => Some(Ok(bytes)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Stream chat with the LLM backend (web version)
    pub async fn stream_chat<F>(
        base_url: &str,
        session_id: &str,
        message: &str,
        history: Vec<Message>,
        provider: ProviderConfig,
        interpret: bool,
        conversation_id: Option<String>,
        stop: StopSignal,
        on_event: F,
    ) -> Result<StreamOutcome, StreamError>
    where
        F: FnMut(StreamEvent),
    {
        let url = format!("{}/api/llm/chat/stream", base_url.trim_end_matches('/'));
        let body_json = request_body(
            session_id,
            message,
            history,
            provider,
            interpret,
            conversation_id,
        )?;
        tracing::debug!("[SSE] Request body length: {}", body_json.len());

        // The controller aborts the fetch and the body read on stop or idle.
        let controller = web_sys::AbortController::new()
            .map_err(|e| StreamError::failed(format!("Failed to create AbortController: {e:?}")))?;

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body_json));
        opts.set_signal(Some(&controller.signal()));

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| StreamError::failed(format!("{e:?}")))?;

        // The token is read at request time, so call sites need not pass it.
        let token = crate::platform::backend_access_token();
        for (name, value) in stream_request_headers(token.as_deref()) {
            request
                .headers()
                .set(&name, &value)
                .map_err(|e| StreamError::failed(format!("Failed to set header {name}: {e:?}")))?;
        }

        let window = web_sys::window().ok_or_else(|| StreamError::failed("No window object"))?;
        let fetch = JsFuture::from(window.fetch_with_request(&request));
        futures::pin_mut!(fetch);
        let fetch_stop = stop.clone();
        let stopped = fetch_stop.fired();
        futures::pin_mut!(stopped);
        let waiting = select(stopped, idle_timer());
        let resp_value = match select(fetch, waiting).await {
            Either::Left((resp, _)) => resp.map_err(|e| {
                tracing::warn!("fetch failed: {e:?}");
                StreamError::failed(crate::platform::http::UNREACHABLE_TEXT)
            })?,
            Either::Right((Either::Left(_), _)) => {
                controller.abort();
                return Err(StreamError::stopped());
            }
            Either::Right((Either::Right(_), _)) => {
                controller.abort();
                return Err(StreamError::idle());
            }
        };

        let response: Response = resp_value
            .dyn_into()
            .map_err(|_| StreamError::failed("Response is not a Response object"))?;

        tracing::debug!("[SSE] Response status: {}", response.status());
        if !response.ok() {
            // The server's error body (429 session_busy, 404 not_found, ...).
            let body = match response.text() {
                Ok(promise) => JsFuture::from(promise)
                    .await
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_default(),
                Err(_) => String::new(),
            };
            return Err(StreamError::failed(String::from(
                crate::platform::api_error(response.status(), &body, token.is_some()),
            )));
        }

        let body = response
            .body()
            .ok_or_else(|| StreamError::failed("Response has no body"))?;
        let reader: JsValue = body.get_reader().into();
        let chunks = futures::stream::unfold(reader, |reader| async move {
            read_chunk(&reader).await.map(|chunk| (chunk, reader))
        });
        futures::pin_mut!(chunks);

        let result = read_events(chunks, stop, idle_timer, on_event).await;
        if result.is_err() {
            controller.abort();
        }
        result
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

    /// Seconds to wait for the connection to the backend.
    const CONNECT_TIMEOUT_SECS: u64 = 10;

    fn idle_timer() -> tokio::time::Sleep {
        tokio::time::sleep(std::time::Duration::from_secs(IDLE_TIMEOUT_SECS))
    }

    /// Stream chat with the LLM backend (native version)
    pub async fn stream_chat<F>(
        base_url: &str,
        session_id: &str,
        message: &str,
        history: Vec<Message>,
        provider: ProviderConfig,
        interpret: bool,
        conversation_id: Option<String>,
        stop: StopSignal,
        on_event: F,
    ) -> Result<StreamOutcome, StreamError>
    where
        F: FnMut(StreamEvent),
    {
        let url = format!("{}/api/llm/chat/stream", base_url.trim_end_matches('/'));
        let body_json = request_body(
            session_id,
            message,
            history,
            provider,
            interpret,
            conversation_id,
        )?;
        tracing::debug!("[SSE] Request body length: {}", body_json.len());

        // The token is read at request time, so call sites need not pass it.
        let token = crate::platform::backend_access_token();
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| StreamError::failed(format!("Cannot create the HTTP client: {e}")))?;
        let mut request = client.post(&url);
        for (name, value) in stream_request_headers(token.as_deref()) {
            request = request.header(name, value);
        }
        let send = request.body(body_json).send();
        futures::pin_mut!(send);
        let fetch_stop = stop.clone();
        let stopped = fetch_stop.fired();
        futures::pin_mut!(stopped);
        let waiting = select(stopped, Box::pin(idle_timer()));
        let response = match select(send, waiting).await {
            Either::Left((resp, _)) => {
                resp.map_err(|e| StreamError::failed(format!("Fetch error: {}", e)))?
            }
            // Dropping the request future closes the connection.
            Either::Right((Either::Left(_), _)) => return Err(StreamError::stopped()),
            Either::Right((Either::Right(_), _)) => return Err(StreamError::idle()),
        };

        tracing::debug!("[SSE] Response status: {}", response.status());
        if !response.status().is_success() {
            // The server's error body (429 session_busy, 404 not_found, ...).
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(StreamError::failed(String::from(
                crate::platform::api_error(status, &body, token.is_some()),
            )));
        }

        // Dropping the body stream on stop or idle closes the connection; the
        // server then ends the tool loop.
        let chunks = response.bytes_stream().map(|chunk| {
            chunk
                .map(|b| b.to_vec())
                .map_err(|e| StreamError::failed(format!("Read error: {}", e)))
        });
        futures::pin_mut!(chunks);
        read_events(chunks, stop, idle_timer, on_event).await
    }
}

// ============================================================================
// Stub implementation (fallback for non-platform builds)
// ============================================================================

#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
pub mod stub {
    use super::*;

    /// Stub stream_chat that returns an error (for builds without platform features)
    pub async fn stream_chat<F>(
        _base_url: &str,
        _session_id: &str,
        _message: &str,
        _history: Vec<Message>,
        _provider: ProviderConfig,
        _interpret: bool,
        _conversation_id: Option<String>,
        _stop: StopSignal,
        _on_event: F,
    ) -> Result<StreamOutcome, StreamError>
    where
        F: FnMut(StreamEvent),
    {
        Err(StreamError::failed(
            "Streaming not available without platform features",
        ))
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

#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
pub use stub::stream_chat;

#[cfg(test)]
mod request_header_tests {
    use super::stream_request_headers;

    #[test]
    fn stream_request_carries_bearer_token() {
        let h = stream_request_headers(Some("s3cret"));
        assert!(h.contains(&("Authorization".to_string(), "Bearer s3cret".to_string())));
        assert!(h.contains(&("Content-Type".to_string(), "application/json".to_string())));
        assert!(
            !stream_request_headers(None)
                .iter()
                .any(|(k, _)| k == "Authorization")
        );
    }
}

#[cfg(test)]
mod line_buffer_tests {
    use super::{decode_trailing, drain_complete_lines};

    #[test]
    fn reassembles_multibyte_char_split_across_chunks() {
        // "data: héllo\n" — the 'é' is 0xC3 0xA9. Split the stream mid-character.
        let full = "data: héllo\n".as_bytes();
        let split = full.iter().position(|&b| b == 0xC3).unwrap() + 1; // between 0xC3 and 0xA9
        let (a, b) = full.split_at(split);

        let mut buf = Vec::new();
        let mut lines = Vec::new();
        drain_complete_lines(&mut buf, a, |l| lines.push(l));
        assert!(lines.is_empty(), "no complete line yet");
        drain_complete_lines(&mut buf, b, |l| lines.push(l));

        assert_eq!(lines, vec!["data: héllo".to_string()]);
        assert!(buf.is_empty());
    }

    #[test]
    fn handles_crlf_and_multiple_lines_and_trailing() {
        let mut buf = Vec::new();
        let mut lines = Vec::new();
        drain_complete_lines(&mut buf, b"one\r\ntwo\ntrail", |l| lines.push(l));
        assert_eq!(lines, vec!["one".to_string(), "two".to_string()]);
        assert_eq!(decode_trailing(&buf), Some("trail".to_string()));
    }
}

#[cfg(test)]
mod read_loop_tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;
    use std::cell::Cell;
    use std::rc::Rc;

    fn chunks(parts: &[&[u8]]) -> Vec<Result<Vec<u8>, StreamError>> {
        parts.iter().map(|p| Ok(p.to_vec())).collect()
    }

    fn texts(events: &[StreamEvent]) -> Vec<String> {
        events
            .iter()
            .map(|e| match e {
                StreamEvent::Content { text } => text.clone(),
                StreamEvent::Done { message } => format!("done:{}", message.content),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn read_events_parses_lines_in_order() {
        let full = "data: {\"type\":\"content\",\"text\":\"héllo\"}\r\n";
        let split = full.as_bytes().iter().position(|&b| b == 0xC3).unwrap() + 1;
        let (a, b) = full.as_bytes().split_at(split);
        let input = chunks(&[
            a,
            b,
            b": keep-alive\n",
            b"data: {\"type\":\"content\",\"text\":\" world\"}\n",
            b"data: {\"type\":\"done\",\"message\":{\"role\":\"assistant\",\"content\":\"hi\"}}",
        ]);
        let (_stop, signal) = stop_pair();
        let mut events = Vec::new();
        let outcome = futures::executor::block_on(read_events(
            stream::iter(input),
            signal,
            futures::future::pending::<()>,
            |e| events.push(e),
        ))
        .expect("the stream ends normally");
        assert_eq!(texts(&events), ["héllo", " world", "done:hi"]);
        assert!(outcome.done);
    }

    #[test]
    fn stop_ends_the_read() {
        let first = chunks(&[b"data: {\"type\":\"content\",\"text\":\"partial\"}\n"]);
        let input = stream::iter(first).chain(stream::pending());
        let (stop, signal) = stop_pair();
        let mut events = Vec::new();
        let err = futures::executor::block_on(read_events(
            input,
            signal,
            futures::future::pending::<()>,
            |e| {
                events.push(e);
                stop.stop();
            },
        ))
        .expect_err("stopped");
        assert_eq!(err.kind(), StreamErrorKind::Stopped);
        assert_eq!(texts(&events), ["partial"]);
    }

    #[test]
    fn idle_timer_ends_a_silent_stream() {
        let (_stop, signal) = stop_pair();
        let err = futures::executor::block_on(read_events(
            stream::pending::<Result<Vec<u8>, StreamError>>(),
            signal,
            || futures::future::ready(()),
            |_| {},
        ))
        .expect_err("idle");
        assert_eq!(err.kind(), StreamErrorKind::Idle);
        assert!(
            err.to_string().contains("No data from the server for 90 s"),
            "{err}"
        );
    }

    #[test]
    fn a_stop_before_the_stream_starts_is_seen() {
        let (stop, signal) = stop_pair();
        assert!(!signal.is_fired());
        stop.stop();
        assert!(signal.is_fired());
    }

    /// A stream that waits (returns Pending once) before each chunk and
    /// before its end.
    fn waiting_stream(
        items: Vec<Result<Vec<u8>, StreamError>>,
    ) -> impl Stream<Item = Result<Vec<u8>, StreamError>> + Unpin {
        let mut items = items.into_iter();
        let mut waited = false;
        futures::stream::poll_fn(move |cx| {
            if !waited {
                waited = true;
                cx.waker().wake_by_ref();
                return std::task::Poll::Pending;
            }
            waited = false;
            std::task::Poll::Ready(items.next())
        })
    }

    #[test]
    fn the_idle_timer_is_armed_for_every_wait() {
        let armed = Rc::new(Cell::new(0));
        let counter = armed.clone();
        let input = chunks(&[
            b"data: {\"type\":\"content\",\"text\":\"a\"}\n",
            b"data: {\"type\":\"content\",\"text\":\"b\"}\n",
            b"data: {\"type\":\"content\",\"text\":\"c\"}\n",
        ]);
        let (_stop, signal) = stop_pair();
        let mut n = 0;
        futures::executor::block_on(read_events(
            waiting_stream(input),
            signal,
            move || {
                counter.set(counter.get() + 1);
                futures::future::pending::<()>()
            },
            |_| n += 1,
        ))
        .expect("ends normally");
        assert_eq!(n, 3);
        // One timer per wait: three chunks and the end of the stream.
        assert_eq!(armed.get(), 4);
    }

    #[test]
    fn a_stop_during_a_flowing_stream_ends_it() {
        // Data never waits, so only the check after each chunk sees the stop.
        let input = stream::iter(std::iter::repeat_with(|| {
            Ok(b"data: {\"type\":\"content\",\"text\":\"a\"}\n".to_vec())
        }));
        let (stop, signal) = stop_pair();
        let mut n = 0;
        let err = futures::executor::block_on(read_events(
            input,
            signal,
            futures::future::pending::<()>,
            |_| {
                n += 1;
                if n == 3 {
                    stop.stop();
                }
            },
        ))
        .expect_err("stopped");
        assert_eq!(err.kind(), StreamErrorKind::Stopped);
        assert_eq!(n, 3);
    }

    #[test]
    fn a_dropped_stop_handle_does_not_stop_the_stream() {
        let (stop, signal) = stop_pair();
        drop(stop);
        let input = chunks(&[b"data: {\"type\":\"content\",\"text\":\"a\"}\n"]);
        let mut n = 0;
        futures::executor::block_on(read_events(
            stream::iter(input),
            signal,
            futures::future::pending::<()>,
            |_| n += 1,
        ))
        .expect("ends normally");
        assert_eq!(n, 1);
    }
}

/// Rule-5 timing of the read loop against a verbatim copy of the base loop
/// (base: `while let Some(chunk) = stream.next().await`, drain each chunk).
#[cfg(test)]
mod timing {
    use super::*;
    use futures::stream;
    use std::time::Instant;

    async fn base_loop<S>(
        mut s: S,
        mut on_event: impl FnMut(StreamEvent),
    ) -> Result<(), StreamError>
    where
        S: Stream<Item = Result<Vec<u8>, StreamError>> + Unpin,
    {
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk_result) = s.next().await {
            let bytes = chunk_result?;
            drain_complete_lines(&mut buffer, &bytes, |line| {
                if let Some(event) = parse_sse_line(&line) {
                    on_event(event);
                }
            });
        }
        if let Some(line) = decode_trailing(&buffer)
            && let Some(event) = parse_sse_line(&line)
        {
            on_event(event);
        }
        Ok(())
    }

    fn content_chunks(events: usize) -> Vec<Vec<u8>> {
        // 40-byte content events, five per chunk.
        let line = b"data: {\"type\":\"content\",\"text\":\"abcde\"}\n";
        assert_eq!(line.len(), 40);
        (0..events.div_ceil(5))
            .map(|i| {
                let n = (events - i * 5).min(5);
                line.repeat(n)
            })
            .collect()
    }

    fn done_chunks(bytes: usize) -> Vec<Vec<u8>> {
        let prefix = "data: {\"type\":\"done\",\"message\":{\"role\":\"assistant\",\"content\":\"";
        let suffix = "\"}}\n";
        let body = "x".repeat(bytes - prefix.len() - suffix.len());
        let line = format!("{prefix}{body}{suffix}").into_bytes();
        line.chunks(1000).map(|c| c.to_vec()).collect()
    }

    fn run_base(chunks: &[Vec<u8>]) -> usize {
        let mut n = 0;
        let s = stream::iter(chunks.iter().cloned().map(Ok));
        futures::executor::block_on(base_loop(s, |_| n += 1)).unwrap();
        n
    }

    fn run_head(chunks: &[Vec<u8>]) -> usize {
        let mut n = 0;
        let s = stream::iter(chunks.iter().cloned().map(Ok));
        let (_stop, signal) = stop_pair();
        futures::executor::block_on(read_events(
            s,
            signal,
            futures::future::pending::<()>,
            |_| n += 1,
        ))
        .unwrap();
        n
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// 3 warm-ups, then 31 interleaved base/head runs; medians in ms.
    fn compare(label: &str, chunks: &[Vec<u8>]) {
        for _ in 0..3 {
            assert_eq!(run_base(chunks), run_head(chunks));
        }
        let (mut b, mut h) = (Vec::new(), Vec::new());
        for _ in 0..31 {
            let t = Instant::now();
            std::hint::black_box(run_base(chunks));
            b.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(run_head(chunks));
            h.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (mb, mh) = (median(b), median(h));
        println!(
            "TIMING {label}: base {mb:.4} ms, head {mh:.4} ms, head/base {:.3}",
            mh / mb
        );
    }

    #[test]
    #[ignore = "benchmark; times the SSE read loop against the base loop"]
    fn time_read_loop() {
        for events in [100, 1_000, 10_000, 100_000] {
            compare(
                &format!("sse content events {events}"),
                &content_chunks(events),
            );
        }
        for bytes in [1_000, 10_000, 100_000] {
            compare(&format!("sse done line {bytes} bytes"), &done_chunks(bytes));
        }
    }
}
