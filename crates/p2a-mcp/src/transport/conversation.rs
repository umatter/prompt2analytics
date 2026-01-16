//! Conversation management HTTP routes.
//!
//! Provides REST API endpoints for managing conversations and messages
//! with persistent storage via SurrealDB.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::db::{Conversation, ConversationWithMessages, Message, Settings};
use crate::persistent_session::{PersistentSessionError, PersistentSessionManager};

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
        // Settings
        .route("/sessions/{session_id}/settings", get(get_settings))
        .route("/sessions/{session_id}/settings", put(update_settings))
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
        Ok(conv) => (StatusCode::CREATED, Json(ApiResponse::success(conv))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Conversation>::error(json.0.error.unwrap_or_default())))
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
        Ok(convs) => (StatusCode::OK, Json(ApiResponse::success(convs))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Vec<Conversation>>::error(json.0.error.unwrap_or_default())))
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
        Ok(conv) => (StatusCode::OK, Json(ApiResponse::success(conv))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<ConversationWithMessages>::error(json.0.error.unwrap_or_default())))
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
        Ok(conv) => (StatusCode::OK, Json(ApiResponse::success(conv))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Conversation>::error(json.0.error.unwrap_or_default())))
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
        Ok(conv) => (StatusCode::OK, Json(ApiResponse::success(conv))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Conversation>::error(json.0.error.unwrap_or_default())))
        }
    }
}

/// Get messages for a conversation.
async fn get_messages(
    State(state): State<ConversationState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.get_messages(&id).await {
        Ok(messages) => (StatusCode::OK, Json(ApiResponse::success(messages))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Vec<Message>>::error(json.0.error.unwrap_or_default())))
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
        Ok(msg) => (StatusCode::CREATED, Json(ApiResponse::success(msg))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<Message>::error(json.0.error.unwrap_or_default())))
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
            Json(ApiResponse::success(serde_json::json!({ "deleted_count": count }))),
        ),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<serde_json::Value>::error(json.0.error.unwrap_or_default())))
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
            (status, Json(ApiResponse::<Settings>::error(json.0.error.unwrap_or_default())))
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
            (status, Json(ApiResponse::<Settings>::error(json.0.error.unwrap_or_default())))
        }
    }
}

/// Database health check.
async fn db_health(State(state): State<ConversationState>) -> impl IntoResponse {
    match state.session_manager.db_health_check().await {
        Ok(healthy) => (
            StatusCode::OK,
            Json(ApiResponse::success(serde_json::json!({ "healthy": healthy }))),
        ),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<serde_json::Value>::error(json.0.error.unwrap_or_default())))
        }
    }
}

/// Database statistics.
async fn db_stats(State(state): State<ConversationState>) -> impl IntoResponse {
    match state.session_manager.db_stats().await {
        Ok(stats) => (StatusCode::OK, Json(ApiResponse::success(stats))),
        Err(e) => {
            let (status, json) = error_response(e);
            (status, Json(ApiResponse::<crate::db::DbStats>::error(json.0.error.unwrap_or_default())))
        }
    }
}
