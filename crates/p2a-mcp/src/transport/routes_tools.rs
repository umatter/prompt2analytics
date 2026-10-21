//! Tool routes: discovery (`GET /api/tools`) and execution
//! (`POST /api/tools/{name}`).

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;

use super::http::{ApiResponse, AppState, ContentItem};
use crate::audit::AuditLogger;
use crate::session::SessionError;

/// Request to call a tool.
#[derive(Debug, Deserialize)]
pub struct CallToolRequest {
    /// Session ID (required for HTTP transport)
    pub session_id: String,
    /// Tool arguments as JSON
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// List available tools.
pub(crate) async fn list_tools(State(state): State<AppState>) -> impl IntoResponse {
    let tools = state.server.list_tools();
    Json(ApiResponse::success(tools))
}

/// Call a tool.
pub(crate) async fn call_tool(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<CallToolRequest>,
) -> impl IntoResponse {
    // Get the session
    let session = match state.session_manager.get_session(&request.session_id).await {
        Ok(s) => s,
        Err(SessionError::NotFound) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::error("Session not found")),
            );
        }
        Err(SessionError::Expired) => {
            return (
                StatusCode::GONE,
                Json(ApiResponse::error("Session expired")),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            );
        }
    };

    // Execute the tool with session context and timing
    let start = std::time::Instant::now();
    let tool_result = state
        .server
        .call_tool_with_session(&name, request.arguments.clone(), &session)
        .await;
    let duration_ms = start.elapsed().as_millis() as u64;

    match tool_result {
        Ok(result) => {
            // Extract text content from result for logging
            let result_text: String = result
                .content
                .iter()
                .filter_map(|item| match item {
                    ContentItem::Text { text } => Some(text.clone()),
                    ContentItem::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Audit logging (if enabled)
            if state.audit_logger.is_enabled() {
                let entry = AuditLogger::create_entry(
                    &request.session_id,
                    &name,
                    &request.arguments,
                    result.success,
                    duration_ms,
                    &result_text,
                    None, // Client IP would require extracting from request headers
                );
                state.audit_logger.log(entry).await;
            }

            // Capture dataset metadata for load_dataset, upload_dataset, and create_dataset
            #[cfg(feature = "db")]
            if (name == "load_dataset" || name == "upload_dataset" || name == "create_dataset")
                && result.success
            {
                if let Some(ref persistent_manager) = state.persistent_manager {
                    tracing::info!(
                        tool = %name,
                        session_id = %request.session_id,
                        result_len = result_text.len(),
                        "[CALL_TOOL] Capturing dataset metadata for direct tool call"
                    );

                    // Use the same capture function as StreamingToolExecutor
                    #[cfg(feature = "llm")]
                    super::llm_routes::StreamingToolExecutor::capture_dataset_metadata_static(
                        persistent_manager.db().clone(),
                        &request.session_id,
                        &name,
                        &request.arguments,
                        &result_text,
                    )
                    .await;
                }
            }

            (StatusCode::OK, Json(ApiResponse::success(result)))
        }
        Err(e) => {
            // Audit log for failures
            if state.audit_logger.is_enabled() {
                let entry = AuditLogger::create_entry(
                    &request.session_id,
                    &name,
                    &request.arguments,
                    false,
                    duration_ms,
                    &e,
                    None,
                );
                state.audit_logger.log(entry).await;
            }

            (StatusCode::BAD_REQUEST, Json(ApiResponse::error(e)))
        }
    }
}
