//! Conversation management HTTP routes.
//!
//! Provides REST API endpoints for managing conversations and messages
//! with persistent storage via SurrealDB.
//!
//! Every conversation, message and tool-call route takes the caller's session
//! id ([`SESSION_HEADER`], or the `session_id` query parameter as a fallback)
//! and answers 404 unless that session is live and owns the object; a missing
//! object and someone else's object get the same answer. The
//! `/sessions/{session_id}/...` routes treat the path id as the caller.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Extension, Json, Router,
    extract::{FromRequestParts, Path, Query, State},
    http::{HeaderMap, StatusCode, Uri, header, request::Parts},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
};
use serde::{Deserialize, Serialize};
use tracing;

use super::http::ApiResponse;
use super::routes_sessions::capacity_response;
use crate::db::{Conversation, ConversationWithMessages, DatasetMeta, DbError, Message};
use crate::exec::{ToolCallError, ToolGovernor};
use crate::persistent_session::{
    DatasetQuotas, PersistentSessionError, PersistentSessionManager, ReloadFailure, ReloadPlan,
    ReloadResult, insert_reloaded_dataset, load_reload_source,
};
use crate::session::SessionError;

/// The header that carries the caller's session id on conversation, message
/// and tool-call routes.
pub const SESSION_HEADER: &str = "x-p2a-session-id";

/// Session ids are UUIDs; a longer id is answered without a lookup.
const MAX_SESSION_ID_BYTES: usize = 128;

/// Shared state for conversation routes.
#[derive(Clone)]
pub struct ConversationState {
    pub session_manager: Arc<PersistentSessionManager>,
}

/// Create conversation routes.
///
/// The dataset reload runs under the tool governor, which the router must
/// provide as an `Extension<Arc<ToolGovernor>>` layer (`create_router` adds
/// it).
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
// The caller's session
// =============================================================================

/// The caller's session id: the `X-P2A-Session-Id` header (trimmed, non-empty),
/// else the `session_id` query parameter. Neither gives 400; an id longer
/// than any session id gives 404 without a lookup. Whether the session is
/// live and owns the object is checked by the handler.
pub(crate) struct CallerSession(pub String);

impl<S: Send + Sync> FromRequestParts<S> for CallerSession {
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let id = header_session(&parts.headers).or_else(|| query_session(&parts.uri));
        let Some(id) = id else {
            return Err(error_with(
                StatusCode::BAD_REQUEST,
                "send the session id in the X-P2A-Session-Id header",
                "invalid_arguments",
            ));
        };
        if id.len() > MAX_SESSION_ID_BYTES {
            return Err(error_with(
                StatusCode::NOT_FOUND,
                "Session not found",
                "not_found",
            ));
        }
        Ok(CallerSession(id))
    }
}

/// The session id in the `X-P2A-Session-Id` header, if one is sent.
fn header_session(headers: &HeaderMap) -> Option<String> {
    headers
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The session id in the `session_id` query parameter, if one is sent.
fn query_session(uri: &Uri) -> Option<String> {
    Query::<HashMap<String, String>>::try_from_uri(uri)
        .ok()
        .and_then(|Query(q)| q.get("session_id").map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
}

/// 404 when a header names a different session than the path of a
/// `/sessions/{session_id}/...` route, which is the caller.
fn path_session_mismatch(headers: &HeaderMap, path_id: &str) -> Option<Response> {
    match header_session(headers) {
        Some(h) if h != path_id => Some(error_with(
            StatusCode::NOT_FOUND,
            "Session not found",
            "not_found",
        )),
        _ => None,
    }
}

// =============================================================================
// Responses
// =============================================================================

fn ok<T: Serialize>(status: StatusCode, data: T) -> Response {
    (status, Json(ApiResponse::success(data))).into_response()
}

fn error_with(status: StatusCode, message: impl Into<String>, code: &'static str) -> Response {
    (
        status,
        Json(ApiResponse::<()>::error_with_code(message, code, None)),
    )
        .into_response()
}

/// Seconds a client should wait before retrying a write refused for memory.
const MEMORY_PRESSURE_RETRY_AFTER_SECS: &str = "15";

/// The one mapping from store errors to responses: 404 `not_found`, 410
/// `session_expired`, 413 `payload_too_large`, 422 `quota_exceeded`, 503
/// `capacity` and `memory_pressure` (with `Retry-After`), and 500 `internal`
/// with a generic text for everything else (the details are logged).
fn error_response(err: PersistentSessionError) -> Response {
    match err {
        PersistentSessionError::Session(SessionError::NotFound) => {
            error_with(StatusCode::NOT_FOUND, "Session not found", "not_found")
        }
        PersistentSessionError::Session(SessionError::Expired) => {
            error_with(StatusCode::GONE, "Session expired", "session_expired")
        }
        PersistentSessionError::Session(SessionError::MaxSessionsReached) => capacity_response(),
        PersistentSessionError::Database(DbError::NotFound(message)) => {
            error_with(StatusCode::NOT_FOUND, message, "not_found")
        }
        PersistentSessionError::Database(DbError::TooLarge(message)) => {
            error_with(StatusCode::PAYLOAD_TOO_LARGE, message, "payload_too_large")
        }
        PersistentSessionError::Database(DbError::QuotaExceeded(message)) => {
            error_with(StatusCode::UNPROCESSABLE_ENTITY, message, "quota_exceeded")
        }
        PersistentSessionError::Database(e @ DbError::MemoryPressure) => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, MEMORY_PRESSURE_RETRY_AFTER_SECS)],
            Json(ApiResponse::<()>::error_with_code(
                e.to_string(),
                "memory_pressure",
                None,
            )),
        )
            .into_response(),
        other => {
            tracing::warn!(error = %other, "Conversation route failed");
            error_with(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal error",
                "internal",
            )
        }
    }
}

/// `Ok(data)` as `status` with the data, `Err` through [`error_response`].
fn respond<T: Serialize>(
    status: StatusCode,
    result: Result<T, PersistentSessionError>,
) -> Response {
    match result {
        Ok(data) => ok(status, data),
        Err(e) => error_response(e),
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

/// The roles a client may post.
const MESSAGE_ROLES: [&str; 3] = ["user", "assistant", "system"];

/// Create a new conversation. The body's `session_id` is the caller; a
/// header naming a different session is a 400.
async fn create_conversation(
    State(state): State<ConversationState>,
    headers: HeaderMap,
    Json(request): Json<CreateConversationRequest>,
) -> Response {
    if header_session(&headers).is_some_and(|h| h != request.session_id) {
        return error_with(
            StatusCode::BAD_REQUEST,
            "the X-P2A-Session-Id header and the body's session_id name different sessions",
            "invalid_arguments",
        );
    }
    let result = state
        .session_manager
        .create_conversation(&request.session_id, &request.title)
        .await
        .map(ConversationResponse::from);
    respond(StatusCode::CREATED, result)
}

/// List the caller's conversations. The caller ([`CallerSession`]: the
/// header, else the `session_id` parameter) must be live; a header and a
/// parameter that name different sessions are a 400.
async fn list_conversations(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    uri: Uri,
) -> Response {
    if query_session(&uri).is_some_and(|q| q != caller) {
        return error_with(
            StatusCode::BAD_REQUEST,
            "the X-P2A-Session-Id header and the session_id parameter name different sessions",
            "invalid_arguments",
        );
    }
    let pm = &state.session_manager;
    if let Err(e) = pm.get_session(&caller).await {
        return error_response(e);
    }
    let result = pm.list_conversations(&caller).await.map(|convs| {
        convs
            .into_iter()
            .map(ConversationResponse::from)
            .collect::<Vec<_>>()
    });
    respond(StatusCode::OK, result)
}

/// Get a conversation with its messages.
async fn get_conversation(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    let conversation = match pm.authorize_conversation(&caller, &id).await {
        Ok(conv) => conv,
        Err(e) => return error_response(e),
    };
    let result = pm.get_messages(&id).await.map(|messages| {
        ConversationWithMessagesResponse::from(ConversationWithMessages {
            conversation,
            messages,
        })
    });
    respond(StatusCode::OK, result)
}

/// Update a conversation's title.
async fn update_conversation(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
    Json(request): Json<UpdateConversationRequest>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    let result = pm
        .update_conversation_title(&id, &request.title)
        .await
        .map(ConversationResponse::from);
    respond(StatusCode::OK, result)
}

/// Delete a conversation.
async fn delete_conversation(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    respond(StatusCode::NO_CONTENT, pm.delete_conversation(&id).await)
}

/// Archive or unarchive a conversation.
async fn archive_conversation(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
    Json(request): Json<ArchiveConversationRequest>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    let result = pm
        .set_conversation_archived(&id, request.is_archived)
        .await
        .map(ConversationResponse::from);
    respond(StatusCode::OK, result)
}

/// Get messages for a conversation.
async fn get_messages(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    let result = pm.get_messages(&id).await.map(|messages| {
        messages
            .into_iter()
            .map(MessageResponse::from)
            .collect::<Vec<_>>()
    });
    respond(StatusCode::OK, result)
}

/// Add a message to a conversation.
async fn add_message(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
    Json(request): Json<AddMessageRequest>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    if !MESSAGE_ROLES.contains(&request.role.as_str()) {
        return error_with(
            StatusCode::BAD_REQUEST,
            "role must be one of user, assistant and system",
            "invalid_arguments",
        );
    }
    let result = pm
        .add_message(&id, &request.role, &request.content)
        .await
        .map(MessageResponse::from);
    respond(StatusCode::CREATED, result)
}

/// Clear all messages in a conversation.
async fn clear_messages(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    let result = pm
        .clear_messages(&id)
        .await
        .map(|count| serde_json::json!({ "deleted_count": count }));
    respond(StatusCode::OK, result)
}

/// Get settings for a session; the path id is the caller and must be live.
async fn get_settings(
    State(state): State<ConversationState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Some(mismatch) = path_session_mismatch(&headers, &session_id) {
        return mismatch;
    }
    respond(
        StatusCode::OK,
        state.session_manager.get_settings(&session_id).await,
    )
}

/// Update settings for a session; the path id is the caller and must be live.
async fn update_settings(
    State(state): State<ConversationState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(request): Json<UpdateSettingsRequest>,
) -> Response {
    if let Some(mismatch) = path_session_mismatch(&headers, &session_id) {
        return mismatch;
    }
    let result = state
        .session_manager
        .patch_settings(
            &session_id,
            request.provider.as_deref(),
            request.model.as_deref(),
            request.temperature,
            request.max_tokens,
        )
        .await;
    respond(StatusCode::OK, result)
}

/// Database health check.
async fn db_health(State(state): State<ConversationState>) -> Response {
    let result = state
        .session_manager
        .db_health_check()
        .await
        .map(|healthy| serde_json::json!({ "healthy": healthy }));
    respond(StatusCode::OK, result)
}

/// Database statistics.
async fn db_stats(State(state): State<ConversationState>) -> Response {
    respond(StatusCode::OK, state.session_manager.db_stats().await)
}

// =============================================================================
// Tool Call Routes
// =============================================================================

/// Get tool calls for a conversation.
async fn get_conversation_tool_calls(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_conversation(&caller, &id).await {
        return error_response(e);
    }
    respond(
        StatusCode::OK,
        pm.get_tool_calls_for_conversation(&id).await,
    )
}

/// Get tool calls for a specific message.
async fn get_message_tool_calls(
    State(state): State<ConversationState>,
    CallerSession(caller): CallerSession,
    Path(id): Path<String>,
) -> Response {
    let pm = &state.session_manager;
    if let Err(e) = pm.authorize_message(&caller, &id).await {
        return error_response(e);
    }
    respond(StatusCode::OK, pm.get_tool_calls_for_message(&id).await)
}

// =============================================================================
// Dataset Routes
// =============================================================================

/// List all dataset metadata for a session; the path id is the caller and
/// must be live.
async fn list_session_datasets(
    State(state): State<ConversationState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Some(mismatch) = path_session_mismatch(&headers, &session_id) {
        return mismatch;
    }
    let pm = &state.session_manager;
    if let Err(e) = pm.get_session(&session_id).await {
        return error_response(e);
    }
    match pm.get_datasets_for_session(&session_id).await {
        Ok(datasets) => {
            tracing::debug!(
                session_id = %session_id,
                num_datasets = datasets.len(),
                "[LIST_DATASETS] Retrieved datasets from database"
            );
            let response: Vec<DatasetMetaResponse> = datasets.into_iter().map(Into::into).collect();
            ok(StatusCode::OK, response)
        }
        Err(e) => error_response(e),
    }
}

/// Reload all datasets for a session from their original source paths; the
/// path id is the caller and must be live (or adoptable).
///
/// The loading runs as one governed call (`ToolGovernor::run_blocking`):
/// memory admission, the session's and a global permit, the deadline and
/// panic isolation, on the blocking pool inside the session's path-jail
/// scope. Each loaded dataset is inserted only if it fits the governor's
/// dataset quotas; one that does not is reported failed. A refusal by the
/// governor answers as a refused tool call does (503 `server_busy` or
/// `memory_pressure`, 429 `session_busy`, 504 `deadline_exceeded`, with
/// `Retry-After` where retrying helps).
async fn reload_session_datasets(
    State(state): State<ConversationState>,
    Extension(governor): Extension<Arc<ToolGovernor>>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Some(mismatch) = path_session_mismatch(&headers, &session_id) {
        return mismatch;
    }
    let pm = &state.session_manager;
    let ReloadPlan {
        session,
        sources,
        skipped,
    } = match pm.reload_plan(&session_id).await {
        Ok(plan) => plan,
        Err(e) => return error_response(e),
    };
    let quotas = DatasetQuotas {
        per_session: governor.config().session_data_bytes,
        total: governor.config().total_data_bytes,
    };
    let sessions = Arc::clone(pm.sessions());
    let target = Arc::clone(&session);
    let handle = tokio::runtime::Handle::current();
    let outcome = governor
        .run_blocking(&session, "reload_session_datasets", move |budget| {
            // Past the deadline (or once the caller is gone) nothing more is
            // loaded or inserted.
            let in_budget = || {
                budget.check().map_err(|_| {
                    ToolCallError::Internal("the dataset reload was stopped".to_string())
                })
            };
            let mut result = ReloadResult {
                skipped,
                ..ReloadResult::default()
            };
            for source in sources {
                in_budget()?;
                let loaded = match load_reload_source(&source) {
                    Ok(dataset) => {
                        in_budget()?;
                        handle.block_on(insert_reloaded_dataset(
                            &target,
                            &sessions,
                            source.name.clone(),
                            dataset,
                            quotas,
                        ))
                    }
                    Err(error) => Err(error),
                };
                match loaded {
                    Ok(()) => result.succeeded.push(source.name),
                    Err(error) => result.failed.push(ReloadFailure {
                        name: source.name,
                        source_path: source.source_path,
                        error: error.to_string(),
                    }),
                }
            }
            Ok(result)
        })
        .await;
    match outcome {
        Ok(result) => {
            tracing::info!(
                session_id = %session_id,
                succeeded = result.succeeded.len(),
                failed = result.failed.len(),
                skipped = result.skipped.len(),
                "Dataset reload completed"
            );
            ok(StatusCode::OK, result)
        }
        Err(e) => e.into_response(),
    }
}
