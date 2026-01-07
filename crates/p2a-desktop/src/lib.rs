//! p2a-desktop library - Tauri desktop application for prompt2analytics.

pub mod commands;
pub mod mcp;

use mcp::McpClient;
use std::path::PathBuf;
use std::sync::Arc;

/// Application state shared across all Tauri commands.
pub struct AppState {
    mcp_client: Arc<McpClient>,
}

impl AppState {
    /// Create new AppState with the path to the MCP binary.
    pub fn new(mcp_binary_path: PathBuf) -> Self {
        Self {
            mcp_client: Arc::new(McpClient::new(mcp_binary_path)),
        }
    }

    /// Get a reference to the MCP client.
    pub fn mcp_client(&self) -> &McpClient {
        &self.mcp_client
    }
}

/// Find the path to the p2a-mcp binary.
///
/// Looks in the following locations:
/// 1. Same directory as the desktop binary
/// 2. ../p2a-mcp/ relative to desktop binary
/// 3. ./target/release/p2a-mcp (development)
/// 4. PATH environment variable
pub fn find_mcp_binary() -> Option<PathBuf> {
    // Try relative to current exe
    if let Ok(exe_path) = std::env::current_exe() {
        let exe_dir = exe_path.parent()?;

        // Same directory
        let candidate = exe_dir.join("p2a-mcp");
        if candidate.exists() {
            return Some(candidate);
        }

        // Windows executable
        let candidate = exe_dir.join("p2a-mcp.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Try development path
    let dev_path = PathBuf::from("./target/release/p2a-mcp");
    if dev_path.exists() {
        return Some(dev_path);
    }

    // Try debug path
    let debug_path = PathBuf::from("./target/debug/p2a-mcp");
    if debug_path.exists() {
        return Some(debug_path);
    }

    // Try workspace root paths
    let workspace_release = PathBuf::from("../../target/release/p2a-mcp");
    if workspace_release.exists() {
        return Some(workspace_release);
    }

    // Check PATH
    if let Ok(path) = which::which("p2a-mcp") {
        return Some(path);
    }

    None
}
