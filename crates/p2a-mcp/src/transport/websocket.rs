//! WebSocket transport for bidirectional streaming communication.
//!
//! Provides real-time streaming responses for LLM chat and tool execution.

use std::sync::Arc;

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::exec::{CallOrigin, ToolGovernor};
use crate::server::AnalyticsServer;
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
    // Bound messages and frames like HTTP request bodies.
    let cap = super::middleware::max_request_body_bytes();
    ws.max_message_size(cap)
        .max_frame_size(cap)
        .on_upgrade(move |socket| {
            handle_socket(socket, state.server, state.session_manager, state.governor)
        })
}

/// Handle a WebSocket connection.
async fn handle_socket(
    socket: WebSocket,
    server: Arc<AnalyticsServer>,
    session_manager: Arc<SessionManager>,
    governor: Arc<ToolGovernor>,
) {
    let (mut sender, mut receiver) = socket.split();

    // Create a channel for sending messages
    let (tx, mut rx) = mpsc::channel::<ServerMessage>(32);

    // Spawn a task to forward messages from the channel to the WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
    });

    // Connection state
    let mut ws_state = WsState::new(server, session_manager, governor);

    // Process incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let text_str = text.to_string();
                match serde_json::from_str::<ClientMessage>(&text_str) {
                    Ok(client_msg) => {
                        if let Err(e) = handle_client_message(&mut ws_state, client_msg, &tx).await
                        {
                            let _ = tx
                                .send(ServerMessage::Error {
                                    id: "system".to_string(),
                                    message: e,
                                    code: None,
                                })
                                .await;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(ServerMessage::Error {
                                id: "system".to_string(),
                                message: format!("Invalid message format: {}", e),
                                code: None,
                            })
                            .await;
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(data)) => {
                // Axum handles pong automatically for us
                let _ = data;
            }
            Err(_) => break,
            _ => {}
        }
    }

    // Clean up
    drop(tx);
    let _ = send_task.await;
    tracing::debug!("WebSocket connection closed");
}

/// Handle a client message.
async fn handle_client_message(
    state: &mut WsState,
    msg: ClientMessage,
    tx: &mpsc::Sender<ServerMessage>,
) -> Result<(), String> {
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
}
