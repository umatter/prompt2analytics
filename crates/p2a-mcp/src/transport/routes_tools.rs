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

/// Tools whose successful result the conversation store reads dataset
/// metadata from (with the `db` and `llm` features and an open store).
#[cfg(all(feature = "db", feature = "llm"))]
const METADATA_TOOLS: [&str; 3] = ["load_dataset", "upload_dataset", "create_dataset"];

/// Store the dataset metadata of a successful `tool` call in `session_id`,
/// read from its `result_text`, on the blocking pool, and wait until it is
/// stored, so that a dataset listing made after the call's answer shows it.
///
/// The parsing walks every column line of the result, so its work is
/// linear in the call's output (bounded by `response_bytes` where request
/// limits apply); it runs on a blocking thread (`Handle::block_on` there,
/// as in `reload_session_datasets`), not on a runtime worker. The store
/// write is not moved by this: the embedded engine runs every query in a
/// task on the runtime that opened the store, whoever awaits it, which is
/// why the row it writes keeps a bounded list of names
/// (`DbConnection::save_dataset_meta`).
///
/// The capture passes no admission of the tool governor. A capture starts
/// at most once per finished governed call; captures take no governor
/// permit and may overlap later calls, each bounded by its output and by
/// the stored-row cap. A second queue would let other visitors' calls,
/// queued behind the call, hold back its answer and then refuse its row,
/// which the dataset listing and the reload after a restart read. Once
/// spawned the capture runs to the end, also when the client goes away. A
/// capture that fails is logged; the call's answer is left as it is.
#[cfg(all(feature = "db", feature = "llm"))]
pub(super) async fn capture_dataset_metadata_off_runtime(
    db: std::sync::Arc<crate::db::DbConnection>,
    session_id: String,
    tool: String,
    arguments: serde_json::Value,
    result_text: String,
) {
    let handle = tokio::runtime::Handle::current();
    let (log_session, log_tool) = (session_id.clone(), tool.clone());
    let joined = tokio::task::spawn_blocking(move || {
        #[cfg(test)]
        capture_threads::record(&session_id);
        handle.block_on(
            super::llm_routes::StreamingToolExecutor::capture_dataset_metadata_static(
                db,
                &session_id,
                &tool,
                &arguments,
                &result_text,
            ),
        );
    })
    .await;
    if let Err(e) = joined {
        tracing::warn!(
            tool = %log_tool,
            session_id = %log_session,
            error = %e,
            "[CALL_TOOL] Dataset metadata was not captured"
        );
    }
}

/// The thread each session's dataset-metadata capture ran on, for the tests
/// that check that it is not a runtime worker.
#[cfg(all(test, feature = "db", feature = "llm"))]
mod capture_threads {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    static THREADS: Mutex<Vec<(String, ThreadId)>> = Mutex::new(Vec::new());

    pub(super) fn record(session_id: &str) {
        THREADS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((session_id.to_string(), std::thread::current().id()));
    }

    /// The threads the captures in `session_id` ran on, in order.
    pub(super) fn of(session_id: &str) -> Vec<ThreadId> {
        THREADS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|(id, _)| id == session_id)
            .map(|(_, thread)| *thread)
            .collect()
    }
}

/// The two arguments the dataset-metadata capture reads, `path` and
/// `filename`, copied without the rest of the call's payload. A missing
/// field becomes null, which the capture reads as absent. The LLM tool loop
/// (`llm_routes`) can use it for the copy it keeps for its own capture.
#[cfg(any(test, all(feature = "db", feature = "llm")))]
pub(super) fn capture_fields(arguments: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "path": arguments.get("path"),
        "filename": arguments.get("filename"),
    })
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
    Json(CallToolRequest {
        session_id,
        arguments,
    }): Json<CallToolRequest>,
) -> Response {
    // Get the session
    let session = match state.session_manager.get_session(&session_id).await {
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

    // The arguments move into the call. The metadata capture below is the
    // only reader after it, so only a call it will capture keeps a copy, and
    // only of the fields it reads (a create_dataset csv_content or an
    // upload_dataset content can be the whole request body).
    #[cfg(all(feature = "db", feature = "llm"))]
    let capture_arguments = (state.persistent_manager.is_some()
        && METADATA_TOOLS.contains(&name.as_str()))
    .then(|| capture_fields(&arguments));

    let tool_result = state
        .governor
        .call(
            state.server.clone(),
            session,
            &name,
            arguments,
            CallOrigin::Http,
        )
        .await;

    match tool_result {
        Ok(result) => {
            // Capture dataset metadata for load_dataset, upload_dataset and
            // create_dataset, with the same parser as the LLM tool loop.
            #[cfg(all(feature = "db", feature = "llm"))]
            if result.success
                && let (Some(arguments), Some(persistent_manager)) =
                    (capture_arguments, &state.persistent_manager)
            {
                let result_text: String = result
                    .content
                    .iter()
                    .filter_map(|item| match item {
                        super::http::ContentItem::Text { text } => Some(text.as_str()),
                        super::http::ContentItem::Image { .. } => None,
                    })
                    .collect::<Vec<&str>>()
                    .join("\n");
                tracing::info!(
                    tool = %name,
                    session_id = %session_id,
                    result_len = result_text.len(),
                    "[CALL_TOOL] Capturing dataset metadata for direct tool call"
                );
                capture_dataset_metadata_off_runtime(
                    persistent_manager.db().clone(),
                    session_id,
                    name,
                    arguments,
                    result_text,
                )
                .await;
            }

            (StatusCode::OK, Json(ApiResponse::success(result))).into_response()
        }
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::capture_fields;
    use serde_json::{Value, json};

    #[test]
    fn the_capture_copy_holds_only_path_and_filename() {
        let upload = json!({"name": "d", "filename": "d.csv", "content": "x".repeat(1 << 20)});
        assert_eq!(
            capture_fields(&upload),
            json!({"path": null, "filename": "d.csv"})
        );
        let load = json!({"path": "data/a.csv", "name": "a"});
        let copy = capture_fields(&load);
        assert_eq!(copy, json!({"path": "data/a.csv", "filename": null}));
        // Read as the capture reads it: null is the same as absent.
        assert_eq!(copy.get("filename").and_then(Value::as_str), None);
        assert_eq!(
            capture_fields(&Value::Null),
            json!({"path": null, "filename": null})
        );
    }

    /// The dataset-metadata capture after a call runs on the blocking pool
    /// and needs no permit of its own: the call answers once its row is
    /// stored, whatever other work holds the governor's permits.
    #[cfg(all(feature = "db", feature = "llm"))]
    mod metadata_capture {
        use std::sync::Arc;
        use std::sync::mpsc;
        use std::time::Duration;

        use axum::Router;
        use axum::body::Body;
        use axum::http::{Request, StatusCode, header};
        use serde_json::{Value, json};
        use tower::ServiceExt;

        use super::super::capture_threads;
        use crate::config::{ServerConfig, TransportType};
        use crate::exec::ExecConfig;
        use crate::session::Session;
        use crate::transport::http::{AppState, build_app_with};

        async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
            let request = Request::post(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            (
                status,
                serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            )
        }

        async fn new_session(app: &Router, state: &AppState) -> Arc<Session> {
            let (status, body) = post(app, "/api/sessions", json!({})).await;
            assert_eq!(status, StatusCode::CREATED, "{body}");
            let id = body["data"]["session_id"].as_str().unwrap();
            state.session_manager.get_session(id).await.unwrap()
        }

        /// Wait (at most 30 s) until `done` holds.
        async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
            for _ in 0..6_000 {
                if done() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            panic!("timed out waiting for {what}");
        }

        /// Governed work in `session` that runs until `release` is sent to or
        /// dropped; `started` is sent once it runs.
        fn hold(
            state: &AppState,
            session: &Arc<Session>,
        ) -> (
            mpsc::Sender<()>,
            tokio::sync::oneshot::Receiver<()>,
            tokio::task::JoinHandle<()>,
        ) {
            let (release, released) = mpsc::channel::<()>();
            let (started, running) = tokio::sync::oneshot::channel();
            let (governor, session) = (state.governor.clone(), session.clone());
            let task = tokio::spawn(async move {
                governor
                    .run_blocking(&session, "hold", move |_budget| {
                        let _ = started.send(());
                        let _ = released.recv();
                        Ok(())
                    })
                    .await
                    .unwrap();
            });
            (release, running, task)
        }

        /// A two-column `create_dataset` call in `session`.
        fn create_body(session: &Session) -> Value {
            json!({
                "session_id": session.id,
                "arguments": {"name": "d", "csv_content": "a,b\n1.5,2.25\n2.75,3.5\n"},
            })
        }

        /// With one global permit, other work queued behind a
        /// `create_dataset` call takes the permit when the call's work is
        /// done, and keeps it for longer than the queue wait. The call still
        /// answers, and its metadata row is stored, while that work holds
        /// the only permit: the capture does not queue for one.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn the_capture_needs_no_permit() {
            let config = ServerConfig {
                transport: TransportType::Http,
                ..Default::default()
            };
            let exec = ExecConfig {
                concurrency: 1,
                session_concurrency: 1,
                queue_wait: Duration::from_secs(10),
                ..Default::default()
            };
            let (app, state) = build_app_with(&config, exec).await.unwrap();
            let store = state
                .persistent_manager
                .clone()
                .expect("an in-memory store");
            let first = new_session(&app, &state).await;
            let second = new_session(&app, &state).await;
            let caller = new_session(&app, &state).await;

            // The only permit is held, so the call queues for it.
            let (release_first, first_running, first_task) = hold(&state, &first);
            first_running.await.unwrap();
            let body = create_body(&caller);
            let mut call = tokio::spawn({
                let app = app.clone();
                async move { post(&app, "/api/tools/create_dataset", body).await }
            });
            // The call holds the caller's permit once it queues for the global one.
            let caller_slots = caller.exec_semaphore(1);
            wait_until("the call to queue", || {
                caller_slots.available_permits() == 0
            })
            .await;
            // Work queued after the call gets the permit when the call's work
            // is done (the semaphores are first in, first out).
            let (release_second, second_running, second_task) = hold(&state, &second);
            let second_slots = second.exec_semaphore(1);
            wait_until("the hold to queue", || {
                second_slots.available_permits() == 0
            })
            .await;
            release_first.send(()).unwrap();
            second_running.await.unwrap();
            assert!(
                caller.datasets.read().await.contains_key("d"),
                "the hold ran before the call"
            );

            // The hold keeps the only permit until it is released below; the
            // call answers meanwhile, with its row stored. A capture that
            // queued for the permit would answer only after the queue wait,
            // with no row.
            let answered = tokio::time::timeout(Duration::from_secs(30), &mut call).await;
            let (status, body) = answered
                .expect("the call did not answer while the hold had the permit")
                .unwrap();
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["data"]["success"], json!(true), "{body}");
            let rows = store
                .db()
                .get_datasets_for_session(&caller.id)
                .await
                .unwrap();
            assert_eq!(rows.len(), 1, "no metadata row for the call's dataset");
            assert_eq!((rows[0].row_count, rows[0].column_count), (2, 2));
            assert_eq!(rows[0].column_names, ["a", "b"]);
            assert!(!second_task.is_finished(), "the hold ended early");
            assert_eq!(state.governor.health().in_flight, 1);

            release_second.send(()).unwrap();
            first_task.await.unwrap();
            second_task.await.unwrap();
            assert_eq!(state.governor.health().in_flight, 0);
        }

        /// The capture's parsing runs on a blocking thread, not on the
        /// runtime's thread (here the test's own, the only worker of a
        /// current-thread runtime).
        #[tokio::test]
        async fn the_capture_runs_off_the_runtime() {
            let config = ServerConfig {
                transport: TransportType::Http,
                ..Default::default()
            };
            let (app, state) = build_app_with(&config, ExecConfig::default())
                .await
                .unwrap();
            let store = state
                .persistent_manager
                .clone()
                .expect("an in-memory store");
            let caller = new_session(&app, &state).await;
            let (status, body) =
                post(&app, "/api/tools/create_dataset", create_body(&caller)).await;
            assert_eq!(status, StatusCode::OK, "{body}");

            let threads = capture_threads::of(&caller.id);
            assert_eq!(threads.len(), 1, "captures: {threads:?}");
            assert_ne!(threads[0], std::thread::current().id());
            let rows = store
                .db()
                .get_datasets_for_session(&caller.id)
                .await
                .unwrap();
            assert_eq!(rows.len(), 1);
        }

        /// A session's metadata rows stay within its `session_db_bytes`
        /// budget when its datasets leave its map after each call (the rows
        /// stay): once the rows fill the budget, the capture is refused
        /// (and logged) and the call still answers. Before, the rows were
        /// not counted.
        #[tokio::test]
        async fn metadata_rows_stop_at_the_session_budget() {
            let budget: u64 = 3_000;
            let config = ServerConfig {
                transport: TransportType::Http,
                limits: crate::config::LimitsSettings {
                    profile: None,
                    overrides: vec![format!("session_db_bytes={budget}")],
                },
                ..Default::default()
            };
            let (app, state) = build_app_with(&config, ExecConfig::default())
                .await
                .unwrap();
            let store = state
                .persistent_manager
                .clone()
                .expect("an in-memory store");
            let caller = new_session(&app, &state).await;
            let cycles = 40;
            for i in 0..cycles {
                let body = json!({
                    "session_id": caller.id,
                    "arguments": {"name": format!("d{i:02}"), "csv_content": "a,b\n1.5,2.25\n2.75,3.5\n"},
                });
                let (status, body) = post(&app, "/api/tools/create_dataset", body).await;
                assert_eq!(status, StatusCode::OK, "{body}");
                assert_eq!(body["data"]["success"], json!(true), "{body}");
                // The dataset leaves the session's map; its row stays.
                caller.datasets.write().await.clear();
            }

            let db = store.db();
            let rows = db.get_datasets_for_session(&caller.id).await.unwrap();
            assert!(
                !rows.is_empty() && rows.len() < cycles,
                "{} metadata rows after {cycles} new names",
                rows.len()
            );
            assert!(db.session_stored_bytes(&caller.id).await.unwrap() <= budget);
        }

        /// A wide dataset's metadata row keeps its first
        /// `MAX_STORED_COLUMN_NAMES` column names and its full width.
        #[tokio::test]
        async fn a_wide_dataset_keeps_its_first_column_names() {
            use crate::db::MAX_STORED_COLUMN_NAMES as MAX;
            let config = ServerConfig {
                transport: TransportType::Http,
                ..Default::default()
            };
            let (app, state) = build_app_with(&config, ExecConfig::default())
                .await
                .unwrap();
            let store = state
                .persistent_manager
                .clone()
                .expect("an in-memory store");
            let caller = new_session(&app, &state).await;
            let width = MAX + 500;
            let names: Vec<String> = (0..width).map(|i| format!("c{i:05}")).collect();
            let row = |r: f64| {
                (0..width)
                    .map(|i| format!("{:.3}", r + 0.37 * (i % 11) as f64))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let csv = format!("{}\n{}\n{}\n", names.join(","), row(1.1), row(2.3));
            let body = json!({
                "session_id": caller.id,
                "arguments": {"name": "wide", "csv_content": csv},
            });
            let (status, body) = post(&app, "/api/tools/create_dataset", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["data"]["success"], json!(true), "{body}");

            let rows = store
                .db()
                .get_datasets_for_session(&caller.id)
                .await
                .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].column_count, i32::try_from(width).unwrap());
            assert_eq!(rows[0].column_names.len(), MAX);
            assert!(rows[0].column_names == names[..MAX], "not the first names");
        }
    }
}
