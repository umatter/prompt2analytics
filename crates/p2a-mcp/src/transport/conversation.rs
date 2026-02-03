//! Conversation management HTTP routes.
//!
//! Provides REST API endpoints for managing conversations and messages
//! with persistent storage via SurrealDB.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post, put},
};
use serde::{Deserialize, Serialize};
use tracing;

use crate::db::{Conversation, ConversationWithMessages, DatasetMeta, Message, Settings, ToolCall};
use crate::persistent_session::{PersistentSessionError, PersistentSessionManager, ReloadResult};

/// Shared state for conversation routes.
#[derive(Clone)]
pub struct ConversationState {
    pub session_manager: Arc<PersistentSessionManager>,
}

/// Create conversation routes.
pub fn conversation_routes(state: ConversationState) -> Router {
    Router::new()
        // Conversation CRUD
        .route("/conversations", post(create_conversation))
        .route("/conversations", get(list_conversations))
        .route("/conversations/{id}", get(get_conversation))
        .route("/conversations/{id}", put(update_conversation))
        .route("/conversations/{id}", delete(delete_conversation))
        .route("/conversations/{id}/archive", post(archive_conversation))
        // Messages
        .route("/conversations/{id}/messages", get(get_messages))
        .route("/conversations/{id}/messages", post(add_message))
        .route("/conversations/{id}/messages", delete(clear_messages))
        // Tool calls
        .route(
            "/conversations/{id}/tool-calls",
            get(get_conversation_tool_calls),
        )
        .route("/messages/{id}/tool-calls", get(get_message_tool_calls))
        // Settings
        .route("/sessions/{session_id}/settings", get(get_settings))
        .route("/sessions/{session_id}/settings", put(update_settings))
        // Datasets
        .route(
            "/sessions/{session_id}/datasets",
            get(list_session_datasets),
        )
        .route(
            "/sessions/{session_id}/datasets/reload",
            post(reload_session_datasets),
        )
        // Database health
        .route("/db/health", get(db_health))
        .route("/db/stats", get(db_stats))
        .with_state(state)
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

// =============================================================================
// API Response DTOs (for serialization without SurrealDB types)
// =============================================================================

/// Dataset metadata for API responses (converts RecordId to String)
#[derive(Debug, Serialize)]
pub struct DatasetMetaResponse {
    pub id: String,
    pub session_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    pub source_type: String,
    pub row_count: i32,
    pub column_count: i32,
    pub column_names: Vec<String>,
    pub loaded_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size_bytes: Option<i64>,
}

impl From<DatasetMeta> for DatasetMetaResponse {
    fn from(meta: DatasetMeta) -> Self {
        Self {
            id: meta.id_string(),
            session_id: meta.session_id,
            name: meta.name,
            source_path: meta.source_path,
            source_type: meta.source_type,
            row_count: meta.row_count,
            column_count: meta.column_count,
            column_names: meta.column_names,
            loaded_at: chrono::DateTime::<chrono::Utc>::from(meta.loaded_at).to_rfc3339(),
            file_size_bytes: meta.file_size_bytes,
        }
    }
}

/// Conversation for API responses (converts RecordId/Datetime to String)
#[derive(Debug, Serialize)]
pub struct ConversationResponse {
    pub id: String,
    pub session_id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub is_archived: bool,
    pub message_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_message_preview: Option<String>,
}

impl From<Conversation> for ConversationResponse {
    fn from(conv: Conversation) -> Self {
        // Extract datetime strings first (before moving owned fields)
        let id = conv.id_string();
        let created_at = conv.created_at_chrono().to_rfc3339();
        let updated_at = conv.updated_at_chrono().to_rfc3339();
        Self {
            id,
            session_id: conv.session_id,
            title: conv.title,
            created_at,
            updated_at,
            is_archived: conv.is_archived,
            message_count: conv.message_count,
            last_message_preview: conv.last_message_preview,
        }
    }
}

/// Message for API responses (converts RecordId/Datetime to String)
#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub id: String,
    pub conversation_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl From<Message> for MessageResponse {
    fn from(msg: Message) -> Self {
        // Extract datetime and id strings first (before moving owned fields)
        let id = msg.id_string();
        let created_at = msg.created_at_chrono().to_rfc3339();
        let role = format!("{:?}", msg.role).to_lowercase();
        Self {
            id,
            conversation_id: msg.conversation_id,
            role,
            content: msg.content,
            created_at,
            token_count: msg.token_count,
            model: msg.model,
        }
    }
}

/// Conversation with messages for API responses
#[derive(Debug, Serialize)]
pub struct ConversationWithMessagesResponse {
    pub conversation: ConversationResponse,
    pub messages: Vec<MessageResponse>,
}

impl From<ConversationWithMessages> for ConversationWithMessagesResponse {
    fn from(cwm: ConversationWithMessages) -> Self {
        Self {
            conversation: cwm.conversation.into(),
            messages: cwm.messages.into_iter().map(Into::into).collect(),
        }
    }
}

// =============================================================================
// Request types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateConversationRequest {
    pub session_id: String,
    pub title: String,
}

#[derive(Debug, Deserialize)]
pub struct ListConversationsQuery {
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateConversationRequest {
    pub title: String,
}

#[derive(Debug, Deserialize)]
pub struct ArchiveConversationRequest {
    pub is_archived: bool,
}

#[derive(Debug, Deserialize)]
pub struct AddMessageRequest {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSettingsRequest {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
}

// =============================================================================
// Route handlers
// =============================================================================

fn error_response(err: PersistentSessionError) -> (StatusCode, Json<ApiResponse<()>>) {
    let status = match &err {
        PersistentSessionError::Session(crate::session::SessionError::NotFound) => {
            StatusCode::NOT_FOUND
        }
        PersistentSessionError::Session(crate::session::SessionError::Expired) => StatusCode::GONE,
        PersistentSessionError::Session(crate::session::SessionError::MaxSessionsReached) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        PersistentSessionError::Database(crate::db::DbError::NotFound(_)) => StatusCode::NOT_FOUND,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ApiResponse::error(err.to_string())))
}

/// Create a new conversation.
async fn create_conversation(
    State(state): State<ConversationState>,
    Json(request): Json<CreateConversationRequest>,
) -> impl IntoResponse {
    match state
        .session_manager
        .create_conversation(&request.session_id, &request.title)
        .await
    {
        Ok(conv) => {
            let response: ConversationResponse = conv.into();
            (StatusCode::CREATED, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<ConversationResponse>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// List conversations for a session.
async fn list_conversations(
    State(state): State<ConversationState>,
    axum::extract::Query(query): axum::extract::Query<ListConversationsQuery>,
) -> impl IntoResponse {
    match state
        .session_manager
        .list_conversations(&query.session_id)
        .await
    {
        Ok(convs) => {
            let response: Vec<ConversationResponse> = convs.into_iter().map(Into::into).collect();
            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Vec<ConversationResponse>>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Get a conversation by ID.
async fn get_conversation(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state
        .session_manager
        .get_conversation_with_messages(&id)
        .await
    {
        Ok(conv) => {
            let response: ConversationWithMessagesResponse = conv.into();
            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<ConversationWithMessagesResponse>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Update a conversation's title.
async fn update_conversation(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateConversationRequest>,
) -> impl IntoResponse {
    match state
        .session_manager
        .update_conversation_title(&id, &request.title)
        .await
    {
        Ok(conv) => {
            let response: ConversationResponse = conv.into();
            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<ConversationResponse>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Delete a conversation.
async fn delete_conversation(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.delete_conversation(&id).await {
        Ok(()) => (StatusCode::NO_CONTENT, Json(ApiResponse::success(()))),
        Err(e) => error_response(e),
    }
}

/// Archive or unarchive a conversation.
async fn archive_conversation(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
    Json(request): Json<ArchiveConversationRequest>,
) -> impl IntoResponse {
    match state
        .session_manager
        .set_conversation_archived(&id, request.is_archived)
        .await
    {
        Ok(conv) => {
            let response: ConversationResponse = conv.into();
            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<ConversationResponse>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Get messages for a conversation.
async fn get_messages(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.get_messages(&id).await {
        Ok(messages) => {
            let response: Vec<MessageResponse> = messages.into_iter().map(Into::into).collect();
            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Vec<MessageResponse>>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Add a message to a conversation.
async fn add_message(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
    Json(request): Json<AddMessageRequest>,
) -> impl IntoResponse {
    match state
        .session_manager
        .add_message(&id, &request.role, &request.content)
        .await
    {
        Ok(msg) => {
            let response: MessageResponse = msg.into();
            (StatusCode::CREATED, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<MessageResponse>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Clear all messages in a conversation.
async fn clear_messages(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.clear_messages(&id).await {
        Ok(count) => (
            StatusCode::OK,
            Json(ApiResponse::success(
                serde_json::json!({ "deleted_count": count }),
            )),
        ),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<serde_json::Value>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Get settings for a session.
async fn get_settings(
    State(state): State<ConversationState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.get_settings(&session_id).await {
        Ok(settings) => (StatusCode::OK, Json(ApiResponse::success(settings))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Settings>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Update settings for a session.
async fn update_settings(
    State(state): State<ConversationState>,
    Path(session_id): Path<String>,
    Json(request): Json<UpdateSettingsRequest>,
) -> impl IntoResponse {
    match state
        .session_manager
        .patch_settings(
            &session_id,
            request.provider.as_deref(),
            request.model.as_deref(),
            request.temperature,
            request.max_tokens,
        )
        .await
    {
        Ok(settings) => (StatusCode::OK, Json(ApiResponse::success(settings))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Settings>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Database health check.
async fn db_health(State(state): State<ConversationState>) -> impl IntoResponse {
    match state.session_manager.db_health_check().await {
        Ok(healthy) => (
            StatusCode::OK,
            Json(ApiResponse::success(
                serde_json::json!({ "healthy": healthy }),
            )),
        ),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<serde_json::Value>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Database statistics.
async fn db_stats(State(state): State<ConversationState>) -> impl IntoResponse {
    match state.session_manager.db_stats().await {
        Ok(stats) => (StatusCode::OK, Json(ApiResponse::success(stats))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<crate::db::DbStats>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

// =============================================================================
// Tool Call Routes
// =============================================================================

/// Get tool calls for a conversation.
async fn get_conversation_tool_calls(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state
        .session_manager
        .get_tool_calls_for_conversation(&id)
        .await
    {
        Ok(tool_calls) => (StatusCode::OK, Json(ApiResponse::success(tool_calls))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Vec<ToolCall>>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Get tool calls for a specific message.
async fn get_message_tool_calls(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.get_tool_calls_for_message(&id).await {
        Ok(tool_calls) => (StatusCode::OK, Json(ApiResponse::success(tool_calls))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Vec<ToolCall>>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

// =============================================================================
// Dataset Routes
// =============================================================================

/// List all dataset metadata for a session.
async fn list_session_datasets(
    State(state): State<ConversationState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    tracing::info!(
        session_id = %session_id,
        "[LIST_DATASETS] Fetching datasets for session"
    );

    match state
        .session_manager
        .get_datasets_for_session(&session_id)
        .await
    {
        Ok(datasets) => {
            // DEBUG: Log each dataset from the database
            tracing::info!(
                num_datasets = datasets.len(),
                "[LIST_DATASETS] Retrieved datasets from database"
            );
            for (idx, ds) in datasets.iter().enumerate() {
                tracing::info!(
                    idx = idx,
                    id = %ds.id_string(),
                    name = %ds.name,
                    column_count = ds.column_count,
                    column_names_len = ds.column_names.len(),
                    column_names = ?ds.column_names,
                    "[LIST_DATASETS] Dataset from DB"
                );
            }

            // Convert to API response type (RecordId -> String)
            let response: Vec<DatasetMetaResponse> = datasets.into_iter().map(Into::into).collect();

            // DEBUG: Log response being sent to frontend
            tracing::info!(
                num_datasets = response.len(),
                "[LIST_DATASETS] Sending response to frontend"
            );
            for (idx, ds) in response.iter().enumerate() {
                tracing::info!(
                    idx = idx,
                    id = %ds.id,
                    name = %ds.name,
                    column_count = ds.column_count,
                    column_names_len = ds.column_names.len(),
                    column_names = ?ds.column_names,
                    "[LIST_DATASETS] Dataset in API response"
                );
            }

            (StatusCode::OK, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "[LIST_DATASETS] Failed to fetch datasets"
            );
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<Vec<DatasetMetaResponse>>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}

/// Reload all datasets for a session from their original source paths.
async fn reload_session_datasets(
    State(state): State<ConversationState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state
        .session_manager
        .reload_session_datasets(&session_id)
        .await
    {
        Ok(result) => (StatusCode::OK, Json(ApiResponse::success(result))),
        Err(e) => {
            let (status, json) = error_response(e);
            (
                status,
                Json(ApiResponse::<ReloadResult>::error(
                    json.0.error.unwrap_or_default(),
                )),
            )
        }
    }
}
