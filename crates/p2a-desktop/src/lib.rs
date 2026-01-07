//! p2a-desktop library - Tauri desktop application for prompt2analytics.

pub mod commands;
pub mod llm;
pub mod mcp;

use llm::{HistoryStore, LlmService};
use mcp::McpClient;
use std::path::PathBuf;
use std::sync::Arc;

/// Application state shared across all Tauri commands.
pub struct AppState {
    mcp_client: Arc<McpClient>,
    llm_service: Arc<LlmService>,
}

impl AppState {
    /// Create new AppState with the path to the MCP binary and data directory.
    pub fn new(mcp_binary_path: PathBuf, data_dir: PathBuf) -> Self {
        let mcp_client = Arc::new(McpClient::new(mcp_binary_path));

        // Create data directory if it doesn't exist
        std::fs::create_dir_all(&data_dir).ok();

        // Initialize history store
        let history_path = data_dir.join("conversations.db");
        let history = Arc::new(
            HistoryStore::new(&history_path).expect("Failed to initialize history store"),
        );

        // Initialize LLM service
        let llm_service = Arc::new(LlmService::new(mcp_client.clone(), history));

        Self {
            mcp_client,
            llm_service,
        }
    }

    /// Get a reference to the MCP client.
    pub fn mcp_client(&self) -> &McpClient {
        &self.mcp_client
    }

    /// Get a reference to the LLM service.
    pub fn llm_service(&self) -> &LlmService {
        &self.llm_service
    }
}

/// Get the application data directory.
///
/// Returns platform-specific data directory:
/// - Linux: ~/.local/share/p2a-desktop
/// - macOS: ~/Library/Application Support/p2a-desktop
/// - Windows: %APPDATA%/p2a-desktop
pub fn get_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("p2a-desktop")
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
