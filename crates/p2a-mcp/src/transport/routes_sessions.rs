//! Session routes (`/api/sessions*`) and the file browser (`/api/files`).

use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::http::{ApiResponse, AppState};
use crate::config::DeploymentMode;
use crate::session::{SessionError, validate_user_id};

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

/// Seconds a client should wait before retrying when the session table is full.
const CAPACITY_RETRY_AFTER_SECS: &str = "60";

/// The answer when the session table is full: 503 with code `capacity` and
/// `Retry-After`.
pub(crate) fn capacity_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, CAPACITY_RETRY_AFTER_SECS)],
        Json(ApiResponse::<()>::error_with_code(
            "Maximum number of sessions reached",
            "capacity",
            None,
        )),
    )
        .into_response()
}

/// A JSON error with a machine-readable `code`.
fn error_with(status: StatusCode, message: &str, code: &'static str) -> Response {
    (
        status,
        Json(ApiResponse::<()>::error_with_code(message, code, None)),
    )
        .into_response()
}

/// Create a new session. A full session table answers 503 with code
/// `capacity` and `Retry-After`. The `user_id` must pass
/// [`validate_user_id`] (400 `invalid_arguments` otherwise); in open public
/// mode it has no meaning and is not stored.
pub(crate) async fn create_session(
    State(state): State<AppState>,
    Extension(mode): Extension<DeploymentMode>,
    Json(request): Json<CreateSessionRequest>,
) -> Response {
    let user_id = if mode == DeploymentMode::OpenPublic {
        None
    } else {
        match validate_user_id(request.user_id) {
            Ok(user_id) => user_id,
            Err(message) => {
                return error_with(StatusCode::BAD_REQUEST, &message, "invalid_arguments");
            }
        }
    };
    // First create the in-memory session
    match state.session_manager.create_session(user_id.clone()).await {
        Ok(session_id) => {
            // Also register in persistent storage if available (for conversation support)
            #[cfg(feature = "db")]
            if let Some(ref persistent_manager) = state.persistent_manager {
                match persistent_manager
                    .register_session(&session_id, user_id)
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
                .into_response()
        }
        Err(SessionError::MaxSessionsReached) => capacity_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// List all active sessions.
pub(crate) async fn list_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let sessions = state.session_manager.list_sessions().await;
    Json(ApiResponse::success(sessions))
}

/// Why [`live_session`] found no session.
#[derive(Debug)]
enum LookupError {
    /// The registry's answer: not found, expired, or full (adoption).
    Session(SessionError),
    /// The store failed (logged). The session may still exist, so this is
    /// not a "not found".
    #[cfg_attr(not(feature = "db"), allow(dead_code))]
    Internal,
}

/// Look up a live session. With persistence, a stored session still within
/// its TTL is adopted (a RocksDB server after a restart), subject to the
/// registry's capacity.
async fn live_session(
    state: &AppState,
    id: &str,
) -> Result<std::sync::Arc<crate::session::Session>, LookupError> {
    #[cfg(feature = "db")]
    if let Some(pm) = &state.persistent_manager {
        use crate::persistent_session::PersistentSessionError;
        return pm.get_session(id).await.map_err(|e| match e {
            PersistentSessionError::Session(e) => LookupError::Session(e),
            other => {
                tracing::warn!(session_id = %id, error = %other, "Session lookup failed");
                LookupError::Internal
            }
        });
    }
    state
        .session_manager
        .get_session(id)
        .await
        .map_err(LookupError::Session)
}

/// Get session information: 404 `not_found`, 410 `session_expired`, 503
/// `capacity` (with `Retry-After`) when adopting a stored session finds the
/// registry full, and 500 `internal` when the store fails.
pub(crate) async fn get_session(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match live_session(&state, &id).await {
        Ok(session) => {
            let info = session.info().await;
            (StatusCode::OK, Json(ApiResponse::success(info))).into_response()
        }
        Err(LookupError::Session(SessionError::NotFound)) => {
            error_with(StatusCode::NOT_FOUND, "Session not found", "not_found")
        }
        Err(LookupError::Session(SessionError::Expired)) => {
            error_with(StatusCode::GONE, "Session expired", "session_expired")
        }
        Err(LookupError::Session(SessionError::MaxSessionsReached)) => capacity_response(),
        Err(LookupError::Internal) => error_with(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal error",
            "internal",
        ),
    }
}

/// Delete a session: 204, 404 `not_found` when it does not exist, and 500
/// `internal` with a generic text when deleting fails (the details are
/// logged). With persistence, its rows are deleted before the answer (a
/// stored session that is no longer live is deleted too).
pub(crate) async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    let not_found = || error_with(StatusCode::NOT_FOUND, "Session not found", "not_found");
    let internal = |error: &dyn std::fmt::Display| {
        tracing::warn!(session_id = %id, error = %error, "Failed to delete session");
        error_with(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal error",
            "internal",
        )
    };
    let deleted = || (StatusCode::NO_CONTENT, Json(ApiResponse::success(()))).into_response();
    #[cfg(feature = "db")]
    if let Some(pm) = &state.persistent_manager {
        use crate::persistent_session::PersistentSessionError;
        return match pm.delete_session(&id).await {
            Ok(()) => deleted(),
            Err(PersistentSessionError::Session(SessionError::NotFound)) => not_found(),
            Err(e) => internal(&e),
        };
    }
    match state.session_manager.delete_session(&id).await {
        Ok(()) => deleted(),
        Err(SessionError::NotFound) => not_found(),
        Err(e) => internal(&e),
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

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use tower::ServiceExt;

    /// Deleting a session that does not exist answers 404 with the code
    /// `not_found` that `GET /api/sessions/{id}` uses, with persistence (the
    /// `db` feature) and without.
    #[tokio::test]
    async fn deleting_an_unknown_session_is_not_found_with_a_code() {
        let (app, state) = super::super::http::build_app(&crate::config::ServerConfig::default())
            .await
            .unwrap();
        // With the db feature, the persistent branch is the one under test:
        // build_app only logs when persistence fails to start.
        #[cfg(feature = "db")]
        assert!(state.persistent_manager.is_some());
        #[cfg(not(feature = "db"))]
        let _ = state;
        let req = Request::delete(format!("/api/sessions/{}", uuid::Uuid::new_v4()))
            .header(header::HOST, "localhost")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "not_found", "{body}");
    }
}
