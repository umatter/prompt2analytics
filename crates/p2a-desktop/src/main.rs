//! Tauri desktop application entry point for prompt2analytics.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use p2a_desktop_lib::{find_mcp_binary, AppState};
use tauri::Manager;

fn main() {
    // Find the MCP binary
    let mcp_binary = find_mcp_binary().unwrap_or_else(|| {
        eprintln!("Warning: p2a-mcp binary not found. Analytics will not work.");
        eprintln!("Build with: cargo build --release -p p2a-mcp");
        std::path::PathBuf::from("p2a-mcp")
    });

    println!("Using MCP binary: {:?}", mcp_binary);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(move |app| {
            // Create app state with MCP client
            let state = AppState::new(mcp_binary.clone());
            app.manage(state);

            // Spawn MCP server in background
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state: tauri::State<'_, AppState> = app_handle.state();
                if let Err(e) = state.mcp_client().spawn().await {
                    eprintln!("Failed to spawn MCP server: {}", e);
                } else {
                    println!("MCP server started successfully");
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // Gracefully shutdown MCP server
                let app = window.app_handle();
                let state: tauri::State<'_, AppState> = app.state();
                let client = state.mcp_client();

                // Spawn async shutdown
                let client: &'static p2a_desktop_lib::mcp::McpClient =
                    unsafe { std::mem::transmute(client) };
                tauri::async_runtime::spawn(async move {
                    let _ = client.shutdown().await;
                });
            }
        })
        .invoke_handler(tauri::generate_handler![
            // Analytics commands
            p2a_desktop_lib::commands::invoke_tool,
            p2a_desktop_lib::commands::list_tools,
            // Dataset commands
            p2a_desktop_lib::commands::list_datasets,
            p2a_desktop_lib::commands::load_dataset,
            p2a_desktop_lib::commands::get_dataset_preview,
            p2a_desktop_lib::commands::describe_dataset,
            // File commands
            p2a_desktop_lib::commands::pick_file,
            p2a_desktop_lib::commands::pick_files,
            p2a_desktop_lib::commands::pick_directory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
