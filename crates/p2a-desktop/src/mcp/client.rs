//! MCP client for communicating with p2a-mcp subprocess via JSON-RPC over stdio.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use tokio::sync::oneshot;

use super::protocol::*;

/// Error type for MCP client operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Failed to spawn MCP server: {0}")]
    SpawnError(#[from] std::io::Error),

    #[error("Failed to serialize request: {0}")]
    SerializeError(#[from] serde_json::Error),

    #[error("MCP server not running")]
    NotRunning,

    #[error("Request timed out")]
    Timeout,

    #[error("Response channel closed")]
    ChannelClosed,

    #[error("JSON-RPC error {code}: {message}")]
    RpcError { code: i32, message: String },

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// MCP client that manages a subprocess and communicates via JSON-RPC.
pub struct McpClient {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    pending_requests: Arc<RwLock<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    next_id: AtomicU64,
    mcp_binary_path: PathBuf,
}

impl McpClient {
    /// Create a new MCP client with the path to the p2a-mcp binary.
    pub fn new(mcp_binary_path: PathBuf) -> Self {
        Self {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            mcp_binary_path,
        }
    }

    /// Spawn the MCP server subprocess.
    pub async fn spawn(&self) -> Result<(), McpError> {
        {
            let child_guard = self.child.lock().unwrap();
            if child_guard.is_some() {
                return Ok(()); // Already running
            }
        }

        let mut child = Command::new(&self.mcp_binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Show server logs
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin should be piped");
        let stdout = child.stdout.take().expect("stdout should be piped");

        {
            let mut child_guard = self.child.lock().unwrap();
            *child_guard = Some(child);
        }
        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            *stdin_guard = Some(stdin);
        }

        // Spawn reader thread for stdout
        self.spawn_reader(stdout);

        // Send initialize request (required by MCP protocol)
        self.initialize().await?;

        Ok(())
    }

    /// Spawn a thread to read responses from stdout.
    fn spawn_reader(&self, stdout: ChildStdout) {
        let pending = Arc::clone(&self.pending_requests);

        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let id = response.id;
                            // Use std::sync::RwLock - works from any thread
                            let mut pending = pending.write().unwrap();
                            if let Some(tx) = pending.remove(&id) {
                                let _ = tx.send(response);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading from MCP server: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });
    }

    /// Send the MCP initialize request.
    async fn initialize(&self) -> Result<(), McpError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "p2a-desktop",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let _response = self.send_request("initialize", Some(params)).await?;

        // Send initialized notification
        self.send_notification("notifications/initialized", None)?;

        Ok(())
    }

    /// Send a JSON-RPC request and wait for response.
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);

        // Set up response channel
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending_requests.write().unwrap();
            pending.insert(id, tx);
        }

        // Send request
        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            let stdin = stdin_guard.as_mut().ok_or(McpError::NotRunning)?;
            serde_json::to_writer(&mut *stdin, &request)?;
            writeln!(stdin)?;
            stdin.flush()?;
        }

        // Wait for response
        let response = rx.await.map_err(|_| McpError::ChannelClosed)?;

        if let Some(error) = response.error {
            return Err(McpError::RpcError {
                code: error.code,
                message: error.message,
            });
        }

        response
            .result
            .ok_or_else(|| McpError::InvalidResponse("Missing result".to_string()))
    }

    /// Send a JSON-RPC notification (no response expected).
    fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(serde_json::Value::Null)
        });

        let mut stdin_guard = self.stdin.lock().unwrap();
        let stdin = stdin_guard.as_mut().ok_or(McpError::NotRunning)?;
        serde_json::to_writer(&mut *stdin, &notification)?;
        writeln!(stdin)?;
        stdin.flush()?;

        Ok(())
    }

    /// Call an MCP tool.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.send_request("tools/call", Some(params)).await?;

        let call_result: ToolCallResult = serde_json::from_value(result)
            .map_err(|e| McpError::InvalidResponse(e.to_string()))?;

        Ok(ToolResult::from_call_result(call_result, name))
    }

    /// List available tools.
    pub async fn list_tools(&self) -> Result<Vec<serde_json::Value>, McpError> {
        let result = self.send_request("tools/list", None).await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        Ok(tools)
    }

    /// Shutdown the MCP server.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        // Try to send graceful shutdown
        let _ = self.send_notification("exit", None);

        // Kill the process
        let mut child_guard = self.child.lock().unwrap();
        if let Some(mut child) = child_guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        *self.stdin.lock().unwrap() = None;

        Ok(())
    }

    /// Check if the MCP server is running.
    pub fn is_running(&self) -> bool {
        self.child.lock().unwrap().is_some()
    }
}
