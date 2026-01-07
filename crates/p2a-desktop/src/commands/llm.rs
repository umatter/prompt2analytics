//! LLM-related Tauri commands.

use crate::llm::{Conversation, Message, ProviderConfig, ProviderType, StreamChunk};
use crate::AppState;
use tauri::{Emitter, State};

/// Response from sending a message.
#[derive(serde::Serialize)]
pub struct SendMessageResponse {
    pub conversation_id: String,
    pub message: Message,
}

/// Send a message to the LLM and get a response.
#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    conversation_id: Option<String>,
    message: String,
) -> Result<SendMessageResponse, String> {
    let llm_service = state.llm_service();

    // Ensure MCP server is running for tool execution
    let mcp_client = state.mcp_client();
    if !mcp_client.is_running() {
        mcp_client.spawn().await.map_err(|e| e.to_string())?;
    }

    let (conv_id, response) = llm_service
        .send_message(conversation_id.as_deref(), &message)
        .await
        .map_err(|e| e.to_string())?;

    Ok(SendMessageResponse {
        conversation_id: conv_id,
        message: response,
    })
}

/// Send a message with streaming response.
/// Emits "llm-stream" events to the frontend.
#[tauri::command]
pub async fn send_message_stream(
    state: State<'_, AppState>,
    app: tauri::AppHandle,
    conversation_id: Option<String>,
    message: String,
) -> Result<SendMessageResponse, String> {
    let llm_service = state.llm_service();

    // Ensure MCP server is running
    let mcp_client = state.mcp_client();
    if !mcp_client.is_running() {
        mcp_client.spawn().await.map_err(|e| e.to_string())?;
    }

    // Create callback that emits events to frontend
    let app_handle = app.clone();
    let callback = Box::new(move |chunk: StreamChunk| {
        let _ = app_handle.emit("llm-stream", &chunk);
    });

    let (conv_id, response) = llm_service
        .send_message_stream(conversation_id.as_deref(), &message, callback)
        .await
        .map_err(|e| e.to_string())?;

    Ok(SendMessageResponse {
        conversation_id: conv_id,
        message: response,
    })
}

/// List all conversations.
#[tauri::command]
pub async fn list_conversations(
    state: State<'_, AppState>,
) -> Result<Vec<Conversation>, String> {
    let llm_service = state.llm_service();
    llm_service
        .list_conversations()
        .map_err(|e| e.to_string())
}

/// Get messages for a conversation.
#[tauri::command]
pub async fn get_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<Vec<Message>, String> {
    let llm_service = state.llm_service();
    llm_service
        .get_conversation_messages(&conversation_id)
        .map_err(|e| e.to_string())
}

/// Delete a conversation.
#[tauri::command]
pub async fn delete_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<(), String> {
    let llm_service = state.llm_service();
    llm_service
        .delete_conversation(&conversation_id)
        .map_err(|e| e.to_string())
}

/// Get current LLM settings.
#[tauri::command]
pub async fn get_llm_settings(state: State<'_, AppState>) -> Result<ProviderConfig, String> {
    let llm_service = state.llm_service();
    Ok(llm_service.get_config().await)
}

/// Update LLM settings.
#[tauri::command]
pub async fn update_llm_settings(
    state: State<'_, AppState>,
    config: ProviderConfig,
) -> Result<(), String> {
    let llm_service = state.llm_service();
    llm_service
        .set_config(config)
        .await
        .map_err(|e| e.to_string())
}

/// Check if current provider is available.
#[tauri::command]
pub async fn check_provider(state: State<'_, AppState>) -> Result<bool, String> {
    let llm_service = state.llm_service();
    llm_service
        .is_provider_available()
        .await
        .map_err(|e| e.to_string())
}

/// List available models for current provider.
#[tauri::command]
pub async fn list_provider_models(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let llm_service = state.llm_service();
    llm_service
        .list_models()
        .await
        .map_err(|e| e.to_string())
}

/// Get list of supported provider types.
#[tauri::command]
pub fn list_provider_types() -> Vec<String> {
    vec![
        ProviderType::Ollama.to_string(),
        ProviderType::Anthropic.to_string(),
        ProviderType::OpenAI.to_string(),
    ]
}

/// Rename a conversation.
#[tauri::command]
pub async fn rename_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
    new_title: String,
) -> Result<(), String> {
    let llm_service = state.llm_service();
    llm_service
        .rename_conversation(&conversation_id, &new_title)
        .map_err(|e| e.to_string())
}

/// Export conversation data structure.
#[derive(serde::Serialize)]
pub struct ExportedConversation {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub provider: String,
    pub model: String,
    pub messages: Vec<Message>,
}

/// Export a conversation.
#[tauri::command]
pub async fn export_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<ExportedConversation, String> {
    let llm_service = state.llm_service();

    // Get conversation metadata
    let conversation = llm_service
        .get_conversation(&conversation_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Conversation not found".to_string())?;

    // Get messages
    let messages = llm_service
        .get_conversation_messages(&conversation_id)
        .map_err(|e| e.to_string())?;

    Ok(ExportedConversation {
        id: conversation.id,
        title: conversation.title,
        created_at: conversation.created_at.to_rfc3339(),
        updated_at: conversation.updated_at.to_rfc3339(),
        provider: conversation.provider.to_string(),
        model: conversation.model,
        messages,
    })
}
