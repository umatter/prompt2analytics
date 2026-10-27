//! WebSocket transport for bidirectional streaming communication.
//!
//! A connection handles one client message at a time, in the order the
//! messages arrive, and keeps reading the socket while it handles one:
//!
//! - a close frame, a read error or the end of the stream ends the connection
//!   at once and drops the message being handled. A tool call it was running
//!   is cancelled, as a disconnect cancels `POST /api/tools/{name}`: the
//!   governor cancels the call's budget when its future is dropped, and
//!   cooperative work stops at its next checkpoint;
//! - protocol pings are answered as the socket is read, and a `ping` message
//!   is answered at once;
//! - other messages wait their turn, up to [`MAX_WAITING_MESSAGES`] of them and
//!   the message cap in bytes; one more is refused with an `error` whose code
//!   is `connection_busy`.
//!
//! An answer echoes at most [`MAX_ID_BYTES`] of a message's `id`: a message
//! with a longer one is refused, and that refusal, like any answer that has
//! no id to echo, carries the id `system`.
//! A parse error, and a tool's refusal of arguments that do not deserialise,
//! give serde's message, which quotes the offending value, cut to
//! [`MAX_QUOTE_BYTES`](crate::server::MAX_QUOTE_BYTES) of it
//! ([`quote_cut`]). Nothing else is cut here: a
//! tool's own error or result can still quote argument values, such as a
//! dataset or column name, in full (a result can hold such a quote twice,
//! in its content and in its `error`), up to the size of the message and,
//! for a result where a `response_bytes` cap is set, that cap.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{
        State,
        ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures::{
    sink::SinkExt,
    stream::{Stream, StreamExt},
};
use p2a_core::memory::format_bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::exec::{CallOrigin, ToolGovernor};
use crate::server::{AnalyticsServer, quote_cut};
use crate::session::{Session, SessionError, SessionManager};
use crate::transport::http::ToolResult;

/// WebSocket message types from client to server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Authenticate with a session ID
    Auth { session_id: String },
    /// Execute a tool
    Tool {
        id: String,
        name: String,
        #[serde(default)]
        args: serde_json::Value,
    },
    /// Chat message for LLM (Phase 3)
    Chat {
        id: String,
        message: String,
        #[serde(default)]
        stream: bool,
    },
    /// Ping for keepalive
    Ping,
}

impl ClientMessage {
    /// The id the client gave this message, for the types that have one.
    fn id(&self) -> Option<&str> {
        match self {
            Self::Tool { id, .. } | Self::Chat { id, .. } => Some(id),
            Self::Auth { .. } | Self::Ping => None,
        }
    }
}

/// The longest `id` a client message may have, in bytes. Every answer to a
/// message carries its id, so this keeps the id part of every answer short.
const MAX_ID_BYTES: usize = 256;

/// The id an answer to a message carries: the message's own when it has
/// one of at most [`MAX_ID_BYTES`], `system` otherwise.
fn answer_id(id: Option<&str>) -> String {
    match id {
        Some(id) if id.len() <= MAX_ID_BYTES => id.to_string(),
        _ => "system".to_string(),
    }
}

/// The refusal of a message whose id is longer than [`MAX_ID_BYTES`].
fn id_too_long(id: &str) -> String {
    format!(
        "Invalid message: its id is {} long, and an id may be at most {MAX_ID_BYTES} bytes.",
        format_bytes(id.len())
    )
}

/// The refusal of a message that is not a client message, with serde's
/// message cut by [`quote_cut`]: the end it keeps says what serde expected
/// and where it stopped.
fn invalid_format(e: &serde_json::Error) -> String {
    format!("Invalid message format: {}", quote_cut(e))
}

/// WebSocket message types from server to client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Authentication result
    AuthResult {
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Streaming text chunk (for LLM responses)
    Text { id: String, content: String },
    /// Tool call indication (LLM wants to call a tool)
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    /// Tool execution result
    ToolResult {
        id: String,
        success: bool,
        result: ToolResult,
    },
    /// Request completed
    Done { id: String },
    /// Error occurred. `code` is the machine-readable code of a governor or
    /// tool-call refusal (for example `server_busy`), when there is one.
    Error {
        id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
    /// Pong response
    Pong,
}

/// State for a WebSocket connection.
struct WsState {
    session: Option<Arc<Session>>,
    server: Arc<AnalyticsServer>,
    session_manager: Arc<SessionManager>,
    governor: Arc<ToolGovernor>,
}

impl WsState {
    fn new(
        server: Arc<AnalyticsServer>,
        session_manager: Arc<SessionManager>,
        governor: Arc<ToolGovernor>,
    ) -> Self {
        Self {
            session: None,
            server,
            session_manager,
            governor,
        }
    }

    fn is_authenticated(&self) -> bool {
        self.session.is_some()
    }
}

/// WebSocket upgrade handler.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<super::http::AppState>,
) -> impl IntoResponse {
    // Bound messages and frames like HTTP request bodies, and the messages
    // waiting behind the one being handled to the same size in total.
    let cap = super::middleware::max_request_body_bytes();
    ws.max_message_size(cap)
        .max_frame_size(cap)
        .on_upgrade(move |socket| {
            handle_socket(
                socket,
                state.server,
                state.session_manager,
                state.governor,
                cap,
            )
        })
}

/// The most messages that wait while a connection handles another one.
const MAX_WAITING_MESSAGES: usize = 16;

/// The variable that sets the message cap, the HTTP body cap of
/// [`super::middleware::max_request_body_bytes`]; the messages waiting on a
/// connection may take that much together.
const MESSAGE_CAP_VAR: &str = "P2A_MAX_HTTP_BODY_MB";

/// How long a connection that ended may take to send what it still has and
/// its close frame before the socket is dropped.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// Handle a WebSocket connection; messages waiting behind the one being
/// handled may take up to `waiting_bytes` in total.
async fn handle_socket(
    socket: WebSocket,
    server: Arc<AnalyticsServer>,
    session_manager: Arc<SessionManager>,
    governor: Arc<ToolGovernor>,
    waiting_bytes: usize,
) {
    let (mut sender, mut receiver) = socket.split();

    // Create a channel for sending messages
    let (tx, mut rx) = mpsc::channel::<ServerMessage>(32);

    // Spawn a task to forward messages from the channel to the WebSocket
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
        // Answer the client's close frame, or send one when the connection
        // ended otherwise.
        let _ = sender.close().await;
    });

    let mut ws_state = WsState::new(server, session_manager, governor);
    let mut waiting = Waiting::new(waiting_bytes);
    loop {
        let text = match waiting.pop() {
            Some(text) => text,
            None => match Incoming::of(receiver.next().await) {
                Incoming::Text(text) => text,
                Incoming::Other => continue,
                Incoming::Gone => break,
            },
        };
        let handling = handle_text(&mut ws_state, &text, &tx);
        if !read_while_handling(handling, &mut receiver, &mut waiting, &tx).await {
            break;
        }
    }

    // Clean up
    drop(tx);
    if tokio::time::timeout(CLOSE_GRACE, &mut send_task)
        .await
        .is_err()
    {
        send_task.abort();
    }
    tracing::debug!("WebSocket connection closed");
}

/// What one read of the socket brought.
enum Incoming {
    Text(Utf8Bytes),
    /// A ping, pong or binary frame: nothing to handle. Reading the socket
    /// answers a ping.
    Other,
    /// A close frame, a read error or the end of the stream.
    Gone,
}

impl Incoming {
    fn of(frame: Option<Result<Message, axum::Error>>) -> Self {
        match frame {
            Some(Ok(Message::Text(text))) => Self::Text(text),
            Some(Ok(Message::Close(_)) | Err(_)) | None => Self::Gone,
            Some(Ok(_)) => Self::Other,
        }
    }
}

/// Parse and handle one client message; failures are sent as `error`s.
async fn handle_text(state: &mut WsState, text: &str, tx: &mpsc::Sender<ServerMessage>) {
    let error = match serde_json::from_str::<ClientMessage>(text) {
        Ok(client_msg) => match handle_client_message(state, client_msg, tx).await {
            Ok(()) => return,
            Err(e) => e,
        },
        Err(e) => invalid_format(&e),
    };
    let _ = tx
        .send(ServerMessage::Error {
            id: "system".to_string(),
            message: error,
            code: None,
        })
        .await;
}

/// Drive `handling` (one client message) to its end while reading the
/// socket: what arrives meanwhile goes to [`Waiting::admit`]. `false` when
/// the client went first (or the writer stopped): `handling` is dropped
/// then, and with it a governed tool call it was running, whose guard
/// cancels the call's budget.
async fn read_while_handling<R>(
    handling: impl Future<Output = ()>,
    receiver: &mut R,
    waiting: &mut Waiting,
    tx: &mpsc::Sender<ServerMessage>,
) -> bool
where
    R: Stream<Item = Result<Message, axum::Error>> + Unpin,
{
    let mut handling = std::pin::pin!(handling);
    loop {
        tokio::select! {
            biased;
            () = &mut handling => return true,
            () = tx.closed() => return false,
            frame = receiver.next() => match Incoming::of(frame) {
                Incoming::Text(text) => {
                    if let Some(answer) = waiting.admit(text) {
                        let _ = tx.send(answer).await;
                    }
                }
                Incoming::Other => {}
                Incoming::Gone => return false,
            },
        }
    }
}

/// The `type` of a client message, read without building the rest.
#[derive(Deserialize)]
struct MessageType<'a> {
    #[serde(rename = "type", borrow)]
    kind: Option<Cow<'a, str>>,
}

/// The `id` of a client message, read without building the rest.
#[derive(Deserialize)]
struct MessageId<'a> {
    #[serde(borrow)]
    id: Option<Cow<'a, str>>,
}

/// Messages that arrived while the connection handles another one, in
/// order: at most [`MAX_WAITING_MESSAGES`], and at most `max_bytes` in total.
struct Waiting {
    messages: VecDeque<Utf8Bytes>,
    bytes: usize,
    max_bytes: usize,
}

impl Waiting {
    fn new(max_bytes: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            bytes: 0,
            max_bytes,
        }
    }

    /// The message that has waited longest.
    fn pop(&mut self) -> Option<Utf8Bytes> {
        let text = self.messages.pop_front()?;
        self.bytes -= text.len();
        Some(text)
    }

    /// Take `text`, which arrived while another message is handled; the
    /// answer to send at once, if any: the `pong` to a `ping`, or the
    /// `connection_busy` refusal when the message does not fit.
    fn admit(&mut self, text: Utf8Bytes) -> Option<ServerMessage> {
        if serde_json::from_str::<MessageType>(&text)
            .is_ok_and(|message| message.kind.as_deref() == Some("ping"))
        {
            return Some(ServerMessage::Pong);
        }
        let bytes = self.bytes.saturating_add(text.len());
        // The refusal names the bound that applied.
        let refusal = if self.messages.len() >= MAX_WAITING_MESSAGES {
            Some(format!(
                "This connection is still handling an earlier message, and \
                 {MAX_WAITING_MESSAGES} messages, the most that may wait behind it, are \
                 already waiting; send this one again once their answers arrive."
            ))
        } else if bytes > self.max_bytes {
            Some(format!(
                "This connection is still handling an earlier message, and this one ({}) \
                 does not fit behind the messages already waiting: together they may take \
                 at most the message cap, {} ({MESSAGE_CAP_VAR}), and those take {}; send \
                 it again once their answers arrive.",
                format_bytes(text.len()),
                format_bytes(self.max_bytes),
                format_bytes(self.bytes)
            ))
        } else {
            None
        };
        if let Some(message) = refusal {
            let id = serde_json::from_str::<MessageId>(&text)
                .ok()
                .and_then(|message| message.id);
            return Some(ServerMessage::Error {
                id: answer_id(id.as_deref()),
                message,
                code: Some("connection_busy".to_string()),
            });
        }
        self.bytes = bytes;
        self.messages.push_back(text);
        None
    }
}

/// Handle a client message. Every answer to it carries its id, so a message
/// whose id is longer than [`MAX_ID_BYTES`] is refused (as `system`) before
/// it is handled.
async fn handle_client_message(
    state: &mut WsState,
    msg: ClientMessage,
    tx: &mpsc::Sender<ServerMessage>,
) -> Result<(), String> {
    if let Some(id) = msg.id().filter(|id| id.len() > MAX_ID_BYTES) {
        return Err(id_too_long(id));
    }
    match msg {
        ClientMessage::Auth { session_id } => {
            match state.session_manager.get_session(&session_id).await {
                Ok(session) => {
                    state.session = Some(session);
                    tx.send(ServerMessage::AuthResult {
                        success: true,
                        error: None,
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                }
                Err(SessionError::NotFound) => {
                    tx.send(ServerMessage::AuthResult {
                        success: false,
                        error: Some("Session not found".to_string()),
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                }
                Err(SessionError::Expired) => {
                    tx.send(ServerMessage::AuthResult {
                        success: false,
                        error: Some("Session expired".to_string()),
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                }
                Err(e) => {
                    tx.send(ServerMessage::AuthResult {
                        success: false,
                        error: Some(e.to_string()),
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                }
            }
        }

        ClientMessage::Tool { id, name, args } => {
            let Some(auth_session_id) = state.session.as_ref().map(|s| s.id.clone()) else {
                tx.send(ServerMessage::Error {
                    id: id.clone(),
                    message: "Not authenticated. Send auth message first.".to_string(),
                    code: None,
                })
                .await
                .map_err(|_| "Channel closed".to_string())?;
                return Ok(());
            };

            // Resolve the session again for every call, like an HTTP tool
            // call: a session deleted, evicted or expired since `Auth` is no
            // longer in the table (or in its dataset accounting) and must not
            // keep running tools. A successful lookup also refreshes its
            // last-accessed time.
            let session = match state.session_manager.get_session(&auth_session_id).await {
                Ok(session) => session,
                Err(e) => {
                    state.session = None;
                    let code = match e {
                        SessionError::Expired => "session_expired",
                        _ => "session_not_found",
                    };
                    tx.send(ServerMessage::Error {
                        id,
                        message: format!("{e}. Create a session and authenticate again."),
                        code: Some(code.to_string()),
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                    return Ok(());
                }
            };

            // Execute the tool under the governor, like an HTTP tool call
            match state
                .governor
                .call(
                    state.server.clone(),
                    session,
                    &name,
                    args,
                    CallOrigin::WebSocket,
                )
                .await
            {
                Ok(result) => {
                    tx.send(ServerMessage::ToolResult {
                        id: id.clone(),
                        success: result.success,
                        result,
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;

                    tx.send(ServerMessage::Done { id })
                        .await
                        .map_err(|_| "Channel closed".to_string())?;
                }
                Err(e) => {
                    tx.send(ServerMessage::Error {
                        id,
                        message: e.to_string(),
                        code: Some(e.code().to_string()),
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;
                }
            }
        }

        ClientMessage::Chat {
            id,
            message,
            stream,
        } => {
            if !state.is_authenticated() {
                tx.send(ServerMessage::Error {
                    id: id.clone(),
                    message: "Not authenticated. Send auth message first.".to_string(),
                    code: None,
                })
                .await
                .map_err(|_| "Channel closed".to_string())?;
                return Ok(());
            }

            // LLM chat over WebSocket is not implemented. Return an explicit
            // error so clients don't treat a stub text response as a real
            // completion.
            let _ = (message, stream);
            tx.send(ServerMessage::Error {
                id,
                message: "LLM chat over WebSocket is not implemented; use the HTTP \
                         /api/llm/chat or /api/llm/chat/stream endpoint instead."
                    .to_string(),
                code: None,
            })
            .await
            .map_err(|_| "Channel closed".to_string())?;
        }

        ClientMessage::Ping => {
            tx.send(ServerMessage::Pong)
                .await
                .map_err(|_| "Channel closed".to_string())?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::MAX_QUOTE_BYTES;

    #[test]
    fn test_client_message_parsing() {
        let auth = r#"{"type": "auth", "session_id": "123"}"#;
        let msg: ClientMessage = serde_json::from_str(auth).unwrap();
        match msg {
            ClientMessage::Auth { session_id } => assert_eq!(session_id, "123"),
            _ => panic!("Expected Auth message"),
        }

        let tool = r#"{"type": "tool", "id": "req-1", "name": "list_datasets", "args": {}}"#;
        let msg: ClientMessage = serde_json::from_str(tool).unwrap();
        match msg {
            ClientMessage::Tool { id, name, .. } => {
                assert_eq!(id, "req-1");
                assert_eq!(name, "list_datasets");
            }
            _ => panic!("Expected Tool message"),
        }

        let chat = r#"{"type": "chat", "id": "req-2", "message": "Hello", "stream": true}"#;
        let msg: ClientMessage = serde_json::from_str(chat).unwrap();
        match msg {
            ClientMessage::Chat {
                id,
                message,
                stream,
            } => {
                assert_eq!(id, "req-2");
                assert_eq!(message, "Hello");
                assert!(stream);
            }
            _ => panic!("Expected Chat message"),
        }
    }

    #[test]
    fn test_server_message_serialization() {
        let auth = ServerMessage::AuthResult {
            success: true,
            error: None,
        };
        let json = serde_json::to_string(&auth).unwrap();
        assert!(json.contains("\"type\":\"auth_result\""));
        assert!(json.contains("\"success\":true"));

        let text = ServerMessage::Text {
            id: "1".to_string(),
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn error_message_carries_code() {
        let with = ServerMessage::Error {
            id: "1".into(),
            message: "m".into(),
            code: Some("server_busy".into()),
        };
        let json = serde_json::to_string(&with).unwrap();
        assert!(json.contains("\"code\":\"server_busy\""), "{json}");
        let without = ServerMessage::Error {
            id: "1".into(),
            message: "m".into(),
            code: None,
        };
        let json = serde_json::to_string(&without).unwrap();
        assert!(!json.contains("\"code\""), "{json}");
    }

    #[tokio::test]
    async fn tool_message_after_the_session_is_gone_is_refused() {
        use crate::config::SessionConfig;
        use crate::exec::ExecConfig;

        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let session_id = sessions.create_session(None).await.unwrap();
        let governor = Arc::new(ToolGovernor::new(ExecConfig::default(), sessions.clone()));
        let mut state = WsState::new(Arc::new(AnalyticsServer::new()), sessions.clone(), governor);
        let (tx, mut rx) = mpsc::channel(8);
        handle_client_message(
            &mut state,
            ClientMessage::Auth {
                session_id: session_id.clone(),
            },
            &tx,
        )
        .await
        .unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(ServerMessage::AuthResult { success: true, .. })
        ));
        // Deleted (or evicted, or expired) while the socket stays open.
        sessions.delete_session(&session_id).await.unwrap();
        handle_client_message(
            &mut state,
            ClientMessage::Tool {
                id: "t1".into(),
                name: "list_datasets".into(),
                args: serde_json::json!({}),
            },
            &tx,
        )
        .await
        .unwrap();
        match rx.recv().await {
            Some(ServerMessage::Error { id, code, .. }) => {
                assert_eq!(id, "t1");
                assert_eq!(code.as_deref(), Some("session_not_found"));
            }
            other => panic!("expected a coded error, got {other:?}"),
        }
    }

    /// A connection state whose governor runs on `exec`, authenticated to a
    /// new session; returns the state, that session and the message
    /// channel's two ends.
    async fn authenticated(
        exec: crate::exec::ExecConfig,
    ) -> (
        WsState,
        Arc<Session>,
        mpsc::Sender<ServerMessage>,
        mpsc::Receiver<ServerMessage>,
    ) {
        use crate::config::SessionConfig;

        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let session_id = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&session_id).await.unwrap();
        let governor = Arc::new(ToolGovernor::new(exec, sessions.clone()));
        let mut state = WsState::new(Arc::new(AnalyticsServer::new()), sessions, governor);
        let (tx, mut rx) = mpsc::channel(8);
        handle_client_message(&mut state, ClientMessage::Auth { session_id }, &tx)
            .await
            .unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(ServerMessage::AuthResult { success: true, .. })
        ));
        (state, session, tx, rx)
    }

    #[tokio::test]
    async fn positional_arguments_are_refused() {
        // `set_seed` by position: a derived request struct would take the
        // array, past the request limits and the audit redaction, which read
        // parameters by name.
        let (mut state, session, tx, mut rx) = authenticated(Default::default()).await;
        let tool = ClientMessage::Tool {
            id: "t1".into(),
            name: "set_seed".into(),
            args: serde_json::json!([42]),
        };
        handle_client_message(&mut state, tool, &tx).await.unwrap();
        match rx.recv().await {
            Some(ServerMessage::Error { id, code, message }) => {
                assert_eq!(id, "t1");
                assert_eq!(code.as_deref(), Some("invalid_arguments"));
                assert!(
                    message.starts_with("The arguments of `set_seed` are an array"),
                    "{message}"
                );
            }
            other => panic!("expected a coded error, got {other:?}"),
        }
        assert_eq!(*session.global_seed.read().await, None, "set_seed ran");
    }

    /// The tool name of a message is the caller's string, up to the message
    /// cap (32 MiB by default): the refusal echoes it cut.
    #[tokio::test]
    async fn a_long_tool_name_is_echoed_cut() {
        let (mut state, _session, tx, mut rx) = authenticated(Default::default()).await;
        let tool = ClientMessage::Tool {
            id: "t1".into(),
            name: "x".repeat(1_000_000),
            args: serde_json::json!({}),
        };
        handle_client_message(&mut state, tool, &tx).await.unwrap();
        match rx.recv().await {
            Some(ServerMessage::Error { code, message, .. }) => {
                assert_eq!(code.as_deref(), Some("unknown_tool"));
                assert_eq!(message, format!("Unknown tool: {}", "x".repeat(64)));
            }
            other => panic!("expected a coded error, got {other:?}"),
        }
    }

    /// A shared buffer the log lines of a test subscriber go to.
    #[derive(Clone, Default)]
    struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The governor's `tool_call` span carries the tool name into every log
    /// line inside the call, such as a memory-pressure refusal (a soft
    /// limit of one byte refuses every call): the name is recorded cut.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_long_tool_name_is_logged_cut() {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        // While this subscriber is the only dispatcher registered, tracing
        // takes the interest of a callsite that another thread registers
        // from that thread's default alone (none, so disabled for good),
        // such as the refusal's when `tool_message_is_governed` runs at the
        // same time. A second registered dispatcher makes every registration
        // ask all of them, this subscriber included.
        let _second = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _guard = tracing::subscriber::set_default(subscriber);
        let exec = crate::exec::ExecConfig {
            rss_soft_limit_bytes: Some(1),
            ..Default::default()
        };
        let (mut state, _session, tx, mut rx) = authenticated(exec).await;
        let tool = ClientMessage::Tool {
            id: "t1".into(),
            name: "x".repeat(1_000_000),
            args: serde_json::json!({}),
        };
        // A test on another thread may have registered the span's callsite
        // while only the global no-op subscriber existed, caching it as
        // disabled: recompute every callsite's interest with this
        // subscriber in place.
        tracing::callsite::rebuild_interest_cache();
        handle_client_message(&mut state, tool, &tx).await.unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(ServerMessage::Error { code: Some(code), .. }) if code == "memory_pressure"
        ));
        let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        let refusal = log
            .lines()
            .find(|l| l.contains("process memory is above the soft limit"))
            .unwrap_or_else(|| panic!("no refusal logged: {log}"));
        assert!(
            refusal.contains(&format!("tool=\"{}\"", "x".repeat(64))),
            "{refusal}"
        );
        assert!(refusal.len() < 1_000, "{} bytes", refusal.len());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn tool_message_is_governed() {
        use crate::config::SessionConfig;
        use crate::exec::ExecConfig;

        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 10,
            evict_idle_minutes: None,
        }));
        let session_id = sessions.create_session(None).await.unwrap();
        let governor = Arc::new(ToolGovernor::new(
            ExecConfig {
                rss_soft_limit_bytes: Some(1),
                ..Default::default()
            },
            sessions.clone(),
        ));
        let mut state = WsState::new(Arc::new(AnalyticsServer::new()), sessions, governor);
        let (tx, mut rx) = mpsc::channel(8);
        handle_client_message(&mut state, ClientMessage::Auth { session_id }, &tx)
            .await
            .unwrap();
        handle_client_message(
            &mut state,
            ClientMessage::Tool {
                id: "t1".into(),
                name: "list_datasets".into(),
                args: serde_json::json!({}),
            },
            &tx,
        )
        .await
        .unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(ServerMessage::AuthResult { success: true, .. })
        ));
        match rx.recv().await {
            Some(ServerMessage::Error { id, code, .. }) => {
                assert_eq!(id, "t1");
                assert_eq!(code.as_deref(), Some("memory_pressure"));
            }
            other => panic!("expected a coded error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // A connection that goes stops its call; it is read while a call runs
    // -----------------------------------------------------------------------

    use serde_json::json;
    use tokio_tungstenite::tungstenite::Message as Frame;

    fn tool_text(id: &str, padding: usize) -> Utf8Bytes {
        let message = json!({"type": "tool", "id": id, "name": "list_datasets",
            "args": {}, "padding": "x".repeat(padding)});
        message.to_string().into()
    }

    #[test]
    fn waiting_messages_are_bounded_by_count() {
        let mut waiting = Waiting::new(usize::MAX);
        for i in 0..MAX_WAITING_MESSAGES {
            assert!(waiting.admit(tool_text(&format!("m{i}"), 0)).is_none());
        }
        match waiting.admit(tool_text("over", 0)) {
            Some(ServerMessage::Error { id, code, message }) => {
                assert_eq!(id, "over");
                assert_eq!(code.as_deref(), Some("connection_busy"));
                // The refusal names the bound that applied, not the other.
                assert!(message.contains("16 messages"), "{message}");
                assert!(!message.contains(&format_bytes(usize::MAX)), "{message}");
                assert!(!message.contains(MESSAGE_CAP_VAR), "{message}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A ping is answered even when the queue is full, and does not wait.
        assert!(matches!(
            waiting.admit(r#"{"type": "ping"}"#.into()),
            Some(ServerMessage::Pong)
        ));
        // The first that arrived is handled first; that makes room for one.
        let first = waiting.pop().unwrap();
        assert!(
            first.as_str().contains(r#""id":"m0""#),
            "{}",
            first.as_str()
        );
        assert!(waiting.admit(tool_text("next", 0)).is_none());
        assert!(waiting.admit(tool_text("over", 0)).is_some());
    }

    #[test]
    fn waiting_messages_are_bounded_by_bytes() {
        let one = tool_text("a", 100).len();
        // Room for `a` and `b`, and a little more, but not for `c` too: the
        // bound and what waits are two different numbers.
        let headroom = 10;
        assert!(tool_text("c", 0).len() > headroom);
        let mut waiting = Waiting::new(2 * one + headroom);
        assert!(waiting.admit(tool_text("a", 100)).is_none());
        assert!(waiting.admit(tool_text("b", 100)).is_none());
        match waiting.admit(tool_text("c", 0)) {
            Some(ServerMessage::Error { id, code, message }) => {
                assert_eq!(id, "c");
                assert_eq!(code.as_deref(), Some("connection_busy"));
                // The refusal names the bound that applied, by its value and
                // its variable, this message's size and what already waits,
                // not the count bound.
                assert!(
                    message.contains(&format!(
                        "at most the message cap, {} bytes ({MESSAGE_CAP_VAR})",
                        2 * one + headroom
                    )),
                    "{message}"
                );
                assert!(
                    message.contains(&format!("({} bytes)", tool_text("c", 0).len())),
                    "{message}"
                );
                assert!(
                    message.contains(&format!("take {} bytes", 2 * one)),
                    "{message}"
                );
                assert!(!message.contains("16 messages"), "{message}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        waiting.pop().unwrap();
        assert!(waiting.admit(tool_text("c", 0)).is_none());
        // A refused message whose id cannot be read is answered as `system`.
        match waiting.admit("not json, and too long to wait".repeat(10).into()) {
            Some(ServerMessage::Error { id, .. }) => assert_eq!(id, "system"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        waiting.pop().unwrap();
        waiting.pop().unwrap();
        assert!(waiting.pop().is_none());
        assert_eq!(waiting.bytes, 0);
    }

    /// A refusal holds no more of the refused message than a short id.
    #[test]
    fn a_busy_refusal_echoes_only_a_short_id() {
        // Nothing fits, so every message is refused.
        let mut waiting = Waiting::new(0);
        let longest = "i".repeat(MAX_ID_BYTES);
        match waiting.admit(tool_text(&longest, 0)) {
            Some(ServerMessage::Error { id, .. }) => assert_eq!(id, longest),
            other => panic!("expected a refusal, got {other:?}"),
        }
        match waiting.admit(tool_text(&format!("{longest}i"), 0)) {
            Some(ServerMessage::Error { id, code, .. }) => {
                assert_eq!(id, "system");
                assert_eq!(code.as_deref(), Some("connection_busy"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// Every answer to a message echoes its id: one longer than the bound is
    /// refused before the message is handled, and the refusal does not echo
    /// it.
    #[tokio::test]
    async fn a_message_whose_id_is_too_long_is_refused_unechoed() {
        let (mut state, session, tx, mut rx) = authenticated(Default::default()).await;
        let longest = "i".repeat(MAX_ID_BYTES);
        let set_seed = |id: &str, seed: u64| {
            json!({"type": "tool", "id": id, "name": "set_seed", "args": {"seed": seed}})
                .to_string()
        };
        handle_text(&mut state, &set_seed(&longest, 42), &tx).await;
        match rx.recv().await {
            Some(ServerMessage::ToolResult { id, success, .. }) => {
                assert_eq!(id, longest);
                assert!(success);
            }
            other => panic!("expected a tool result, got {other:?}"),
        }
        assert!(matches!(rx.recv().await, Some(ServerMessage::Done { id }) if id == longest));
        let longer = format!("{longest}i");
        let chat = json!({"type": "chat", "id": longer, "message": "hi"}).to_string();
        for text in [set_seed(&longer, 43), chat] {
            handle_text(&mut state, &text, &tx).await;
            match rx.recv().await {
                Some(ServerMessage::Error { id, message, code }) => {
                    assert_eq!(id, "system");
                    assert_eq!(code, None);
                    assert_eq!(
                        message,
                        format!(
                            "Invalid message: its id is {} bytes long, and an id may be at \
                             most {MAX_ID_BYTES} bytes.",
                            MAX_ID_BYTES + 1
                        )
                    );
                }
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
        assert_eq!(
            *session.global_seed.read().await,
            Some(42),
            "the refused set_seed ran"
        );
        assert!(
            rx.try_recv().is_err(),
            "a refused message was answered twice"
        );
    }

    /// serde quotes an unknown `type` in full; the parse error sent back cuts
    /// the quote (here between two-byte characters) and keeps its end, which
    /// says what was expected and where.
    #[tokio::test]
    async fn a_parse_error_quotes_the_message_cut() {
        let (mut state, _session, tx, mut rx) = authenticated(Default::default()).await;
        let text = json!({"type": "é".repeat(100_000)}).to_string();
        handle_text(&mut state, &text, &tx).await;
        match rx.recv().await {
            Some(ServerMessage::Error { id, message, code }) => {
                assert_eq!(id, "system");
                assert_eq!(code, None);
                assert!(
                    message.starts_with("Invalid message format: unknown variant `éé"),
                    "{message}"
                );
                assert!(message.contains(" bytes cut ...]éé"), "{message}");
                let (said, column) = message
                    .rsplit_once(" at line 1 column ")
                    .unwrap_or_else(|| panic!("no position: {message}"));
                assert!(
                    said.ends_with("éé`, expected one of `auth`, `tool`, `chat`, `ping`"),
                    "{message}"
                );
                assert!(column.parse::<usize>().is_ok(), "{message}");
                assert!(
                    message.len() < 2 * MAX_QUOTE_BYTES,
                    "{} bytes",
                    message.len()
                );
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
        // A short one is quoted whole.
        handle_text(&mut state, r#"{"type": "nonsense"}"#, &tx).await;
        match rx.recv().await {
            Some(ServerMessage::Error { message, .. }) => {
                assert!(
                    message.starts_with("Invalid message format: unknown variant `nonsense`"),
                    "{message}"
                );
                assert!(!message.contains(" cut ...]"), "{message}");
                assert!(message.contains(" at line 1 column "), "{message}");
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    /// A tool message whose argument does not deserialise is refused with
    /// serde's message, which quotes the value: the refusal gives it cut.
    #[tokio::test]
    async fn an_argument_parse_error_quotes_the_value_cut() {
        let (mut state, session, tx, mut rx) = authenticated(Default::default()).await;
        let text = json!({"type": "tool", "id": "t1", "name": "set_seed",
            "args": {"seed": "7".repeat(100_000)}})
        .to_string();
        handle_text(&mut state, &text, &tx).await;
        match rx.recv().await {
            Some(ServerMessage::Error { id, message, code }) => {
                assert_eq!(id, "t1");
                assert_eq!(code.as_deref(), Some("invalid_arguments"));
                assert!(
                    message.starts_with(
                        "Failed to parse arguments for 'set_seed': invalid type: string \"777"
                    ),
                    "{message}"
                );
                assert!(message.ends_with("777\", expected u64"), "{message}");
                assert!(
                    message.len() < 2 * MAX_QUOTE_BYTES,
                    "{} bytes",
                    message.len()
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(*session.global_seed.read().await, None, "set_seed ran");
    }

    type Client = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    /// How long a test waits for what takes milliseconds on a correct server;
    /// the abandoned work these tests guard against runs far longer.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// A local server on a free loopback port whose governor audits to a file
    /// of its own, with `deadline` per call.
    struct Served {
        addr: std::net::SocketAddr,
        state: crate::transport::http::AppState,
        audit: std::path::PathBuf,
        _dir: tempfile::TempDir,
    }

    impl Served {
        async fn start(deadline: Duration) -> Self {
            use crate::config::{AuditConfig, ServerConfig, TransportType};
            let dir = tempfile::tempdir().unwrap();
            let audit = dir.path().join("audit.log");
            let exec = crate::exec::ExecConfig {
                deadline: Some(deadline),
                audit: Some(AuditConfig {
                    enabled: true,
                    path: audit.to_string_lossy().into_owned(),
                }),
                ..Default::default()
            };
            let cfg = ServerConfig {
                transport: TransportType::Http,
                ..Default::default()
            };
            let (app, state) = crate::transport::http::build_app_with(&cfg, exec)
                .await
                .unwrap();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await });
            Self {
                addr,
                state,
                audit,
                _dir: dir,
            }
        }

        /// A server whose deadline only ends the work of a regression, where
        /// the call outlives its client.
        async fn start_local() -> Self {
            Self::start(Duration::from_secs(60)).await
        }

        fn in_flight(&self) -> usize {
            self.state.governor.health().in_flight
        }

        async fn wait_for_in_flight(&self, n: usize, what: &str) {
            let give_up = std::time::Instant::now() + PATIENCE;
            while self.in_flight() != n {
                assert!(
                    std::time::Instant::now() < give_up,
                    "timed out waiting for: {what}"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }

        /// A connection authenticated to a new session that holds dataset
        /// `d`: 200 rows of noisy `y` and `x`.
        async fn connect(&self) -> Client {
            let session_id = self
                .state
                .session_manager
                .create_session(None)
                .await
                .unwrap();
            let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/ws", self.addr))
                .await
                .unwrap();
            send_json(&mut ws, json!({"type": "auth", "session_id": session_id})).await;
            assert_eq!(
                next_json(&mut ws).await,
                json!({"type": "auth_result", "success": true})
            );
            let columns = json!([
                {"name": "y", "distribution": {"type": "normal", "mean": 0.0, "std": 1.0}},
                {"name": "x", "distribution": {"type": "normal", "mean": 0.0, "std": 1.0}}
            ]);
            send_json(
                &mut ws,
                json!({"type": "tool", "id": "data", "name": "generate_random_data", "args":
                    {"n_rows": 200, "columns": columns, "seed": 7, "name": "d"}}),
            )
            .await;
            let result = next_json(&mut ws).await;
            assert_eq!(result["success"], json!(true), "{result}");
            assert_eq!(
                next_json(&mut ws).await,
                json!({"type": "done", "id": "data"})
            );
            ws
        }

        /// Start a bootstrap of five million replications, which runs far
        /// longer than any test, and wait until it runs.
        async fn start_long_call(&self, ws: &mut Client) {
            send_json(
                ws,
                json!({"type": "tool", "id": "long", "name": "regression_bootstrap_cov", "args":
                    {"dataset": "d", "y": "y", "x": ["x"], "n_boot": 5_000_000, "seed": 1}}),
            )
            .await;
            self.wait_for_in_flight(1, "the long call to start").await;
        }

        /// The outcome the audit recorded for the long call, once its work
        /// has stopped.
        async fn long_call_outcome(&self) -> String {
            self.wait_for_in_flight(0, "the long call to stop").await;
            self.state.governor.flush_audit().await;
            let log = std::fs::read_to_string(&self.audit).unwrap();
            let entry = log
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .find(|entry| entry["tool_name"] == "regression_bootstrap_cov")
                .unwrap_or_else(|| panic!("no audit entry for the long call: {log}"));
            assert_eq!(entry["origin"], json!("websocket"), "{entry}");
            entry["outcome"].as_str().unwrap().to_string()
        }
    }

    async fn send_json(ws: &mut Client, message: serde_json::Value) {
        ws.send(Frame::text(message.to_string())).await.unwrap();
    }

    /// The server's next frame, within [`PATIENCE`].
    async fn next_frame(
        ws: &mut Client,
    ) -> Option<Result<Frame, tokio_tungstenite::tungstenite::Error>> {
        tokio::time::timeout(PATIENCE, ws.next())
            .await
            .expect("the server did not answer in time")
    }

    /// The server's next JSON message; control frames are skipped.
    async fn next_json(ws: &mut Client) -> serde_json::Value {
        loop {
            match next_frame(ws).await {
                Some(Ok(Frame::Text(text))) => return serde_json::from_str(text.as_str()).unwrap(),
                Some(Ok(Frame::Ping(_) | Frame::Pong(_))) => {}
                other => panic!("expected a message, got {other:?}"),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_close_frame_cancels_the_running_call() {
        let served = Served::start_local().await;
        let mut ws = served.connect().await;
        served.start_long_call(&mut ws).await;
        ws.close(None).await.unwrap();
        // The server answers the close handshake at once, not when the call
        // ends.
        match next_frame(&mut ws).await {
            Some(Ok(Frame::Close(_))) => {}
            other => panic!("expected the server's close frame, got {other:?}"),
        }
        assert_eq!(served.long_call_outcome().await, "cancelled");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_connection_cancels_the_running_call() {
        let served = Served::start_local().await;
        let mut ws = served.connect().await;
        served.start_long_call(&mut ws).await;
        drop(ws); // the TCP connection closes without a close frame
        assert_eq!(served.long_call_outcome().await, "cancelled");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pings_are_answered_while_a_call_runs() {
        let served = Served::start_local().await;
        let mut ws = served.connect().await;
        served.start_long_call(&mut ws).await;
        // A protocol ping, as a client library's keepalive sends it.
        ws.send(Frame::Ping(vec![7, 7].into())).await.unwrap();
        match next_frame(&mut ws).await {
            Some(Ok(Frame::Pong(data))) => assert_eq!(&data[..], [7, 7]),
            other => panic!("expected a pong frame, got {other:?}"),
        }
        // The protocol's own keepalive message.
        send_json(&mut ws, json!({"type": "ping"})).await;
        assert_eq!(next_json(&mut ws).await, json!({"type": "pong"}));
        assert_eq!(served.in_flight(), 1, "the call ended before the pongs");
        drop(ws);
        assert_eq!(served.long_call_outcome().await, "cancelled");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn messages_beyond_the_waiting_bound_are_refused_at_once() {
        let served = Served::start_local().await;
        let mut ws = served.connect().await;
        served.start_long_call(&mut ws).await;
        for i in 0..=MAX_WAITING_MESSAGES {
            send_json(
                &mut ws,
                json!({"type": "tool", "id": format!("m{i}"), "name": "list_datasets"}),
            )
            .await;
        }
        let refusal = next_json(&mut ws).await;
        assert_eq!(refusal["type"], json!("error"), "{refusal}");
        assert_eq!(
            refusal["id"],
            json!(format!("m{MAX_WAITING_MESSAGES}")),
            "{refusal}"
        );
        assert_eq!(refusal["code"], json!("connection_busy"), "{refusal}");
        let message = refusal["message"].as_str().unwrap();
        assert!(
            message.contains(&format!("{MAX_WAITING_MESSAGES} messages")),
            "{message}"
        );
        assert_eq!(served.in_flight(), 1, "the call ended before the refusal");
        drop(ws);
        assert_eq!(served.long_call_outcome().await, "cancelled");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn messages_sent_during_a_call_are_handled_after_it_in_order() {
        // A short deadline ends the long call; what was sent meanwhile is then
        // handled in the order it was sent.
        let served = Served::start(Duration::from_secs(2)).await;
        let mut ws = served.connect().await;
        served.start_long_call(&mut ws).await;
        send_json(
            &mut ws,
            json!({"type": "tool", "id": "a", "name": "list_datasets"}),
        )
        .await;
        send_json(&mut ws, json!({"type": "nonsense"})).await;
        send_json(
            &mut ws,
            json!({"type": "tool", "id": "b", "name": "list_datasets"}),
        )
        .await;
        let ended = next_json(&mut ws).await;
        assert_eq!(ended["id"], json!("long"), "{ended}");
        assert_eq!(ended["code"], json!("deadline_exceeded"), "{ended}");
        let order: Vec<serde_json::Value> = {
            let mut seen = Vec::new();
            for _ in 0..5 {
                let m = next_json(&mut ws).await;
                seen.push(json!([m["type"], m["id"]]));
            }
            seen
        };
        assert_eq!(
            order,
            vec![
                json!(["tool_result", "a"]),
                json!(["done", "a"]),
                json!(["error", "system"]),
                json!(["tool_result", "b"]),
                json!(["done", "b"]),
            ]
        );
        assert_eq!(served.long_call_outcome().await, "deadline_exceeded");
    }
}
