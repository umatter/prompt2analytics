//! Tool routes: discovery (`GET /api/tools`) and execution
//! (`POST /api/tools/{name}`, through the tool governor).

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use super::http::{ApiResponse, AppState};
use crate::exec::CallOrigin;
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

/// Call a tool. The call runs under the tool governor; its refusals and
/// failures answer with their own status, code and `Retry-After`. The
/// governor writes the audit entry (origin `http`).
pub(crate) async fn call_tool(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<CallToolRequest>,
) -> Response {
    // Get the session
    let session = match state.session_manager.get_session(&request.session_id).await {
        Ok(s) => s,
        Err(SessionError::NotFound) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::<()>::error("Session not found")),
            )
                .into_response();
        }
        Err(SessionError::Expired) => {
            return (
                StatusCode::GONE,
                Json(ApiResponse::<()>::error("Session expired")),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response();
        }
    };

    let tool_result = state
        .governor
        .call(
            state.server.clone(),
            session,
            &name,
            request.arguments.clone(),
            CallOrigin::Http,
        )
        .await;

    match tool_result {
        Ok(result) => {
            // Capture dataset metadata for load_dataset, upload_dataset, and create_dataset
            #[cfg(feature = "db")]
            if (name == "load_dataset" || name == "upload_dataset" || name == "create_dataset")
                && result.success
            {
                if let Some(ref persistent_manager) = state.persistent_manager {
                    let result_text: String = result
                        .content
                        .iter()
                        .filter_map(|item| match item {
                            super::http::ContentItem::Text { text } => Some(text.clone()),
                            super::http::ContentItem::Image { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
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

            (StatusCode::OK, Json(ApiResponse::success(result))).into_response()
        }
        Err(e) => e.into_response(),
    }
}
