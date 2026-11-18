//! Session routes (`/api/sessions*`) and the file browser (`/api/files`).

use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::http::{ApiResponse, AppState, JsonBody};
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
    JsonBody(request): JsonBody<CreateSessionRequest>,
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
pub(crate) enum LookupError {
    /// The registry's answer: not found, expired, or full (adoption).
    Session(SessionError),
    /// The store failed (logged). The session may still exist, so this is
    /// not a "not found".
    #[cfg_attr(not(feature = "db"), allow(dead_code))]
    Internal,
}

/// Look up a live session. With persistence, a stored session still within
/// its TTL is adopted (a persistent SurrealKV store after a restart), subject to the
/// registry's capacity.
pub(crate) async fn live_session(
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

/// The answer of the tool and chat routes when [`live_session`] finds no session: 404
/// `not_found`, 410 `session_expired`, 503 `capacity` (adoption found the registry full) or 500
/// `internal`.
pub(crate) fn lookup_error_response(error: LookupError) -> Response {
    match error {
        LookupError::Session(SessionError::NotFound) => {
            error_with(StatusCode::NOT_FOUND, "Session not found", "not_found")
        }
        LookupError::Session(SessionError::Expired) => {
            error_with(StatusCode::GONE, "Session expired", "session_expired")
        }
        LookupError::Session(SessionError::MaxSessionsReached) => capacity_response(),
        LookupError::Internal => error_with(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal error",
            "internal",
        ),
    }
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
/// `crate::path_jail`). Errors: 403 `forbidden` outside the jail, 400
/// `invalid_arguments` for a path that cannot be read as a directory, 500
/// `internal` when the data root cannot be resolved (the detail is logged).
pub(crate) async fn list_files(Query(query): Query<ListFilesQuery>) -> Response {
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
                return error_with(
                    StatusCode::FORBIDDEN,
                    &format!("Path rejected by data root jail: {e}"),
                    "forbidden",
                );
            }
        },
        None => match path_jail::allowed_data_root() {
            Ok(root) => root,
            Err(e) => {
                tracing::warn!(error = %e, "Cannot resolve the data root");
                return data_root_error();
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
                    if !is_dir && !is_loadable_data_file(&name) {
                        return None;
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
            return error_with(
                StatusCode::BAD_REQUEST,
                &format!("Cannot read directory: {e}"),
                "invalid_arguments",
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
        .into_response()
}

/// The file browser's answer when the data root cannot be resolved: a
/// generic 500 `internal` (the detail is logged, not returned).
fn data_root_error() -> Response {
    error_with(
        StatusCode::INTERNAL_SERVER_ERROR,
        "Cannot resolve the data root",
        "internal",
    )
}

/// Whether the file browser lists a file: exactly the extensions `DataLoader::load` reads.
fn is_loadable_data_file(name: &str) -> bool {
    name.rsplit_once('.').is_some_and(|(_, ext)| {
        p2a_core::DataLoader::supported_extensions().contains(&ext.to_ascii_lowercase().as_str())
    })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use tower::ServiceExt;

    #[test]
    fn test_file_browser_lists_loadable_extensions() {
        for name in [
            "a.jsonl",
            "b.NDJSON",
            "c.pq",
            "d.csv",
            "e.parquet",
            "f.dta",
            "g.sas7bdat",
            "h.json",
        ] {
            assert!(super::is_loadable_data_file(name), "{name}");
        }
        for name in ["i.txt", "csv", "j.csv.bak", "k."] {
            assert!(!super::is_loadable_data_file(name), "{name}");
        }
        for ext in p2a_core::DataLoader::supported_extensions() {
            assert!(super::is_loadable_data_file(&format!("x.{ext}")), "{ext}");
        }
    }

    async fn body_of(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_tool_route_unknown_session_is_not_found_with_a_code() {
        let (app, _state) = super::super::http::build_app(&crate::config::ServerConfig::default())
            .await
            .unwrap();
        let body = serde_json::json!({
            "session_id": uuid::Uuid::new_v4().to_string(),
            "arguments": {}
        });
        let req = Request::post("/api/tools/load_dataset")
            .header(header::HOST, "localhost")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_of(resp).await;
        assert_eq!(body["code"], "not_found", "{body}");
        assert_eq!(body["error"], "Session not found", "{body}");
    }

    /// A JSON body a route cannot read answers the server's JSON error body
    /// with code `invalid_arguments`, the status axum gives the rejection,
    /// and the parser's quote cut as the websocket cuts it.
    #[tokio::test]
    async fn an_unreadable_json_body_is_refused_in_the_error_shape() {
        let (app, _state) = super::super::http::build_app(&crate::config::ServerConfig::default())
            .await
            .unwrap();
        let send = |uri: &str, content_type: Option<&str>, body: String| {
            let mut req = Request::post(uri).header(header::HOST, "localhost");
            if let Some(ct) = content_type {
                req = req.header(header::CONTENT_TYPE, ct);
            }
            app.clone().oneshot(req.body(Body::from(body)).unwrap())
        };
        // A value of the wrong type: 422 in the error shape.
        for (uri, body) in [
            (
                "/api/tools/load_dataset",
                serde_json::json!({"session_id": ["not", "a", "string"], "arguments": {}}),
            ),
            ("/api/sessions", serde_json::json!({"user_id": [1, 2]})),
        ] {
            let resp = send(uri, Some("application/json"), body.to_string())
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY, "{uri}");
            let json = body_of(resp).await;
            assert_eq!(json["success"], false, "{uri}: {json}");
            assert_eq!(json["code"], "invalid_arguments", "{uri}: {json}");
        }
        // serde quotes a mistyped string value and an unknown field name in
        // full: the refusal keeps the quote's two ends around the cut note.
        #[cfg(feature = "llm")]
        {
            use crate::server::MAX_QUOTE_BYTES;
            let long = "x".repeat(10_000);
            let mut unknown_field = serde_json::Map::new();
            unknown_field.insert(long.clone(), serde_json::json!(1));
            for (uri, body) in [
                (
                    "/api/llm/chat",
                    serde_json::json!({"session_id": "s", "message": "m", "interpret": &long}),
                ),
                (
                    "/api/llm/chat/stream",
                    serde_json::json!({"session_id": "s", "message": "m", "interpret": &long}),
                ),
                ("/api/llm/models", serde_json::Value::Object(unknown_field)),
            ] {
                let resp = send(uri, Some("application/json"), body.to_string())
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY, "{uri}");
                let json = body_of(resp).await;
                assert_eq!(json["code"], "invalid_arguments", "{uri}: {json}");
                let error = json["error"].as_str().unwrap();
                let note_start = error.find("[... ").expect("no cut note");
                let note_len = error[note_start..]
                    .find(" bytes cut ...]")
                    .expect("no cut note")
                    + " bytes cut ...]".len();
                assert!(
                    error.len() <= MAX_QUOTE_BYTES + note_len,
                    "{uri}: {} bytes",
                    error.len()
                );
                assert!(!error.contains(&long), "{uri}: the quote is whole");
            }
        }
        // Malformed JSON keeps 400, a missing content type 415, both with the JSON body.
        for (content_type, body, status) in [
            (
                Some("application/json"),
                "{not json".to_string(),
                StatusCode::BAD_REQUEST,
            ),
            (None, "{}".to_string(), StatusCode::UNSUPPORTED_MEDIA_TYPE),
        ] {
            #[allow(unused_mut)]
            let mut uris = vec!["/api/tools/load_dataset", "/api/sessions"];
            #[cfg(feature = "llm")]
            uris.extend([
                "/api/llm/generate-title",
                "/api/llm/chat",
                "/api/llm/chat/stream",
                "/api/llm/models",
            ]);
            for uri in uris {
                let resp = send(uri, content_type, body.clone()).await.unwrap();
                assert_eq!(resp.status(), status, "{uri}");
                let json = body_of(resp).await;
                assert_eq!(json["code"], "invalid_arguments", "{uri}: {json}");
            }
        }
    }

    /// The mapping the tool and chat routes use: an expired session is 410
    /// `session_expired`, an unknown one 404 `not_found`.
    #[tokio::test]
    async fn lookup_errors_carry_codes() {
        use super::{LookupError, lookup_error_response};
        use crate::session::SessionError;
        for (error, status, code) in [
            (SessionError::Expired, StatusCode::GONE, "session_expired"),
            (SessionError::NotFound, StatusCode::NOT_FOUND, "not_found"),
        ] {
            let resp = lookup_error_response(LookupError::Session(error));
            assert_eq!(resp.status(), status);
            assert_eq!(body_of(resp).await["code"], code);
        }
        let resp = lookup_error_response(LookupError::Internal);
        assert_eq!(body_of(resp).await["code"], "internal");
    }

    #[tokio::test]
    async fn test_list_files_errors_have_codes() {
        use super::{ListFilesQuery, list_files};
        use axum::extract::Query;
        let outside = tempfile::tempdir().unwrap();
        let resp = list_files(Query(ListFilesQuery {
            path: Some(outside.path().to_str().unwrap().to_string()),
        }))
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(body_of(resp).await["code"], "forbidden");

        let inside = tempfile::Builder::new()
            .prefix(".p2a-wp31-")
            .tempdir_in(crate::path_jail::allowed_data_root().unwrap())
            .unwrap();
        let file = inside.path().join("f.csv");
        std::fs::write(&file, "x\n1\n").unwrap();
        let resp = list_files(Query(ListFilesQuery {
            path: Some(file.to_str().unwrap().to_string()),
        }))
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_of(resp).await["code"], "invalid_arguments");

        let resp = super::data_root_error();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_of(resp).await;
        assert_eq!(body["code"], "internal");
        assert_eq!(body["error"], "Cannot resolve the data root");
    }

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

    /// With persistence, a session whose rows cannot be deleted answers 500
    /// `internal` with the generic text, and the session stays live.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn a_failed_persistent_delete_is_internal_with_a_code() {
        let (app, state) = super::super::http::build_app(&crate::config::ServerConfig::default())
            .await
            .unwrap();
        let pm = state.persistent_manager.clone().expect("persistence");
        let id = pm.sessions().create_session(None).await.unwrap();
        pm.db()
            .set_session_last_accessed(&id, chrono::Utc::now())
            .await
            .unwrap();
        let meta = crate::db::DatasetMeta::new(
            id.clone(),
            "d".into(),
            "csv".into(),
            3,
            1,
            vec!["x".into()],
        );
        pm.db().save_dataset_meta(&meta).await.unwrap();
        pm.db()
            .db()
            .query(
                "DEFINE EVENT refuse_meta ON TABLE dataset_meta WHEN $event = 'DELETE' \
                 THEN { THROW 'refused' }",
            )
            .await
            .unwrap()
            .check()
            .unwrap();

        let req = Request::delete(format!("/api/sessions/{id}"))
            .header(header::HOST, "localhost")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_of(resp).await;
        assert_eq!(body["code"], "internal", "{body}");
        assert_eq!(body["error"], "internal error", "{body}");
        assert!(pm.sessions().get_session(&id).await.is_ok());
    }
}
