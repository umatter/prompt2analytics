//! Session routes (`/api/sessions*`) and the file browser (`/api/files`).

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};

use super::http::{ApiResponse, AppState};
use crate::session::SessionError;

/// Request to create a new session.
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    #[serde(default)]
    pub user_id: Option<String>,
}

/// Response for session creation.
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
}

/// Create a new session.
pub(crate) async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    // First create the in-memory session
    match state
        .session_manager
        .create_session(request.user_id.clone())
        .await
    {
        Ok(session_id) => {
            // Also register in persistent storage if available (for conversation support)
            #[cfg(feature = "db")]
            if let Some(ref persistent_manager) = state.persistent_manager {
                match persistent_manager
                    .register_session(&session_id, request.user_id)
                    .await
                {
                    Ok(()) => {
                        tracing::info!(
                            "Session {} registered in database for conversation support",
                            session_id
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to register session in database: {}", e);
                        // Continue anyway - session exists in memory, conversations just won't work
                    }
                }
            }

            (
                StatusCode::CREATED,
                Json(ApiResponse::success(CreateSessionResponse { session_id })),
            )
        }
        Err(SessionError::MaxSessionsReached) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiResponse::error("Maximum number of sessions reached")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// List all active sessions.
pub(crate) async fn list_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let sessions = state.session_manager.list_sessions().await;
    Json(ApiResponse::success(sessions))
}

/// Get session information.
pub(crate) async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.get_session(&id).await {
        Ok(session) => {
            let info = session.info().await;
            (StatusCode::OK, Json(ApiResponse::success(info)))
        }
        Err(SessionError::NotFound) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Session not found")),
        ),
        Err(SessionError::Expired) => (
            StatusCode::GONE,
            Json(ApiResponse::error("Session expired")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// Delete a session.
pub(crate) async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.session_manager.delete_session(&id).await {
        Ok(()) => (StatusCode::NO_CONTENT, Json(ApiResponse::success(()))),
        Err(SessionError::NotFound) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Session not found")),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string())),
        ),
    }
}

/// Query params for file listing.
#[derive(Debug, Deserialize)]
pub struct ListFilesQuery {
    /// Directory path to list (defaults to home directory)
    pub path: Option<String>,
}

/// File entry in directory listing.
#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

/// Response for file listing.
#[derive(Debug, Serialize)]
pub struct ListFilesResponse {
    pub path: String,
    pub parent: Option<String>,
    pub entries: Vec<FileEntry>,
}

/// List files in a directory.
///
/// The path is validated against the process-wide data root jail (see
/// `crate::path_jail`). Requests outside the jail return 403.
pub(crate) async fn list_files(Query(query): Query<ListFilesQuery>) -> impl IntoResponse {
    use crate::path_jail;

    // Default to the configured jail root when no path is provided.
    let path = match query
        .path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(requested) => match path_jail::validate_data_path(requested) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::FORBIDDEN,
                    Json(ApiResponse::<ListFilesResponse>::error(format!(
                        "Path rejected by data root jail: {}",
                        e
                    ))),
                );
            }
        },
        None => match path_jail::allowed_data_root() {
            Ok(root) => root,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse::<ListFilesResponse>::error(format!(
                        "Cannot resolve data root: {}",
                        e
                    ))),
                );
            }
        },
    };

    // Read directory
    let entries = match std::fs::read_dir(&path) {
        Ok(read_dir) => {
            let mut entries: Vec<FileEntry> = read_dir
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let metadata = entry.metadata().ok()?;
                    let name = entry.file_name().to_string_lossy().to_string();

                    // Skip hidden files
                    if name.starts_with('.') {
                        return None;
                    }

                    let is_dir = metadata.is_dir();

                    // Filter to only show directories and supported data files
                    if !is_dir {
                        let ext = name.split('.').next_back().unwrap_or("").to_lowercase();
                        if !["csv", "parquet", "json", "xlsx", "xls", "dta", "sas7bdat"]
                            .contains(&ext.as_str())
                        {
                            return None;
                        }
                    }

                    Some(FileEntry {
                        name,
                        path: entry.path().to_string_lossy().to_string(),
                        is_dir,
                        size: if is_dir { None } else { Some(metadata.len()) },
                    })
                })
                .collect();

            // Sort: directories first, then files, both alphabetically
            entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            });

            entries
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::<ListFilesResponse>::error(format!(
                    "Cannot read directory: {}",
                    e
                ))),
            );
        }
    };

    // Only expose a parent link if the parent is still inside the jail; this
    // prevents the UI from offering a "go up" button that escapes the root.
    let jail_root = crate::path_jail::allowed_data_root().ok();
    let parent = path.parent().and_then(|p| match &jail_root {
        Some(root) if p.starts_with(root) => Some(p.to_string_lossy().to_string()),
        _ => None,
    });

    (
        StatusCode::OK,
        Json(ApiResponse::success(ListFilesResponse {
            path: path.to_string_lossy().to_string(),
            parent,
            entries,
        })),
    )
}
