//! Analytics-related Tauri commands.

use crate::mcp::ToolResult;
use crate::AppState;
use tauri::State;

/// Invoke an MCP tool by name with arguments.
#[tauri::command]
pub async fn invoke_tool(
    state: State<'_, AppState>,
    tool_name: String,
    arguments: serde_json::Value,
) -> Result<ToolResult, String> {
    let client = state.mcp_client();

    // Ensure MCP server is running
    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    client
        .call_tool(&tool_name, arguments)
        .await
        .map_err(|e| e.to_string())
}

/// List all available MCP tools.
#[tauri::command]
pub async fn list_tools(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
    let client = state.mcp_client();

    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    client.list_tools().await.map_err(|e| e.to_string())
}
