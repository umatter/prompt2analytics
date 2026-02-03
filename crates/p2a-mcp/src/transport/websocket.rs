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
    /// Error occurred
    Error { id: String, message: String },
    /// Pong response
    Pong,
}

/// State for a WebSocket connection.
struct WsState {
    session: Option<Arc<Session>>,
    server: Arc<AnalyticsServer>,
    session_manager: Arc<SessionManager>,
}

impl WsState {
    fn new(server: Arc<AnalyticsServer>, session_manager: Arc<SessionManager>) -> Self {
        Self {
            session: None,
            server,
            session_manager,
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
    ws.on_upgrade(move |socket| handle_socket(socket, state.server, state.session_manager))
}

/// Handle a WebSocket connection.
async fn handle_socket(
    socket: WebSocket,
    server: Arc<AnalyticsServer>,
    session_manager: Arc<SessionManager>,
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
    let mut ws_state = WsState::new(server, session_manager);

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
                                })
                                .await;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(ServerMessage::Error {
                                id: "system".to_string(),
                                message: format!("Invalid message format: {}", e),
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
            if !state.is_authenticated() {
                tx.send(ServerMessage::Error {
                    id: id.clone(),
                    message: "Not authenticated. Send auth message first.".to_string(),
                })
                .await
                .map_err(|_| "Channel closed".to_string())?;
                return Ok(());
            }

            let session = state.session.as_ref().unwrap();

            // Execute the tool
            match state
                .server
                .call_tool_with_session(&name, args.clone(), session)
                .await
            {
                Ok(result) => {
                    tx.send(ServerMessage::ToolResult {
                        id: id.clone(),
                        success: true,
                        result,
                    })
                    .await
                    .map_err(|_| "Channel closed".to_string())?;

                    tx.send(ServerMessage::Done { id })
                        .await
                        .map_err(|_| "Channel closed".to_string())?;
                }
                Err(e) => {
                    tx.send(ServerMessage::Error { id, message: e })
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
                })
                .await
                .map_err(|_| "Channel closed".to_string())?;
                return Ok(());
            }

            // LLM chat will be implemented in Phase 3
            // For now, just acknowledge the message
            tx.send(ServerMessage::Text {
                id: id.clone(),
                content: format!(
                    "Chat functionality coming in Phase 3. Received: {} (stream: {})",
                    message, stream
                ),
            })
            .await
            .map_err(|_| "Channel closed".to_string())?;

            tx.send(ServerMessage::Done { id })
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
}
