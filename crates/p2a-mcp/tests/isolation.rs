//! End-to-end tests for per-session data directories and the path jail
//! (T0-WP09): cross-session reads and overwrites, relative paths, symlinks,
//! the disk quota, session export and import, the stale-directory purge and
//! directory removal.
//!
//! The server is built in-process with `build_app` and session directories
//! switched on, and driven with `tower::ServiceExt::oneshot`. Every test in
//! this binary shares one data root (set up once in [`init`]); each test uses
//! its own sessions, so the tests can run in parallel.
#![cfg(feature = "http")]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use p2a_mcp::config::{ServerConfig, SessionDirsSetting};
use p2a_mcp::tools::limits::{self, Limits};
use serde_json::{Value, json};
use tower::ServiceExt;

static ROOT: OnceLock<PathBuf> = OnceLock::new();

/// A session directory left behind by an earlier process; the first scoped
/// file access in this process must remove it.
const STALE: &str = "0e1d2c3b-4a59-4687-9a0b-1c2d3e4f5a6b";

/// The budgets every test in this binary runs under.
fn overrides() -> Vec<String> {
    // load_file_bytes=2048: every other test in this file loads files of a few dozen bytes.
    vec![
        "session_disk_bytes=65536".to_string(),
        "export_bytes=32768".to_string(),
        "load_file_bytes=2048".to_string(),
    ]
}

/// The shared data root, set up once per test binary.
fn init() -> &'static Path {
    ROOT.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("p2a-isolation-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("sessions").join(STALE)).unwrap();
        std::fs::write(root.join("sessions").join(STALE).join("old.csv"), "x\n1\n").unwrap();
        // In the global root: must be unreachable from any session.
        std::fs::write(root.join("shared.csv"), "secret\n1\n").unwrap();
        // SAFETY: runs inside OnceLock::get_or_init, which every test calls first and which
        // blocks the other test threads until it returns, so no thread reads the environment
        // concurrently.
        unsafe { std::env::set_var("P2A_DATA_ROOT", &root) };
        p2a_mcp::path_jail::configure_session_dirs(true);
        limits::install(Limits::off().with_overrides(&overrides()).unwrap());
        root.canonicalize().unwrap()
    })
}

/// A fresh app over the shared root: loopback, no token (Local mode), session
/// directories on.
async fn app() -> Router {
    init();
    let mut cfg = ServerConfig::default();
    cfg.http.session_dirs = SessionDirsSetting::On;
    cfg.limits.overrides = overrides();
    p2a_mcp::transport::http::build_app(&cfg).await.unwrap().0
}

fn root_str() -> &'static str {
    init().to_str().unwrap()
}

/// The directory of session `sid`.
fn session_dir(sid: &str) -> PathBuf {
    init().join("sessions").join(sid)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn session(app: &Router) -> String {
    let (status, body) = send(app, post_json("/api/sessions", json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    body["data"]["session_id"].as_str().unwrap().to_string()
}

/// Calls `tool` in session `sid`: whether the call and the tool succeeded,
/// and the tool's text (or the error).
async fn call(app: &Router, sid: &str, tool: &str, args: Value) -> (bool, String) {
    let (_, body) = send(
        app,
        post_json(
            &format!("/api/tools/{tool}"),
            json!({"session_id": sid, "arguments": args}),
        ),
    )
    .await;
    let ok =
        body["success"].as_bool() == Some(true) && body["data"]["success"].as_bool() == Some(true);
    let text = body["data"]["content"][0]["text"]
        .as_str()
        .or_else(|| body["data"]["error"].as_str())
        .or_else(|| body["error"].as_str())
        .unwrap_or_default()
        .to_string();
    (ok, text)
}

async fn delete(app: &Router, sid: &str) {
    let req = Request::delete(format!("/api/sessions/{sid}"))
        .header(header::HOST, "localhost")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{status} {body}");
}

/// Names of leftover temporary files in `dir`.
fn temp_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".p2a-tmp"))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_sessions_cannot_read_or_overwrite_each_others_exports() {
    let app = app().await;
    let root = init();
    let (a, b) = (session(&app).await, session(&app).await);
    assert!(
        call(
            &app,
            &a,
            "create_dataset",
            json!({"name": "private", "csv_content": "patient,score\nalice,3\nbob,5\n"})
        )
        .await
        .0
    );
    let (ok, msg) = call(
        &app,
        &a,
        "export_dataset",
        json!({"dataset": "private", "path": "cleaned_data.csv"}),
    )
    .await;
    assert!(ok, "{msg}");
    assert!(
        msg.contains("cleaned_data.csv") && !msg.contains(root_str()),
        "{msg}"
    );
    let a_file = session_dir(&a).join("cleaned_data.csv");
    assert!(a_file.is_file());
    let a_abs = a_file.to_str().unwrap().to_string();
    for path in [
        "cleaned_data.csv".to_string(),
        a_abs.clone(),
        format!("../{a}/cleaned_data.csv"),
        "../../shared.csv".into(),
        root.join("shared.csv").to_str().unwrap().into(),
    ] {
        let (ok, msg) = call(&app, &b, "load_dataset", json!({"path": path})).await;
        assert!(!ok, "session B loaded {path}");
        assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    }
    assert!(
        call(
            &app,
            &b,
            "create_dataset",
            json!({"name": "evil", "csv_content": "x\n1\n"})
        )
        .await
        .0
    );
    for path in [format!("../{a}/cleaned_data.csv"), a_abs] {
        let (ok, msg) = call(
            &app,
            &b,
            "export_dataset",
            json!({"dataset": "evil", "path": path}),
        )
        .await;
        assert!(!ok, "session B wrote {path}");
        assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    }
    assert_eq!(
        std::fs::read_to_string(&a_file).unwrap(),
        "patient,score\nalice,3\nbob,5\n"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relative_paths_resolve_in_the_session_directory() {
    let app = app().await;
    let s = session(&app).await;
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "d", "csv_content": "x,y\n1,1.1\n2,1.9\n"})
        )
        .await
        .0
    );
    let (ok, msg) = call(
        &app,
        &s,
        "export_dataset",
        json!({"dataset": "d", "path": "out.csv"}),
    )
    .await;
    assert!(ok, "{msg}");
    let (ok, msg) = call(
        &app,
        &s,
        "load_dataset",
        json!({"path": "out.csv", "name": "again"}),
    )
    .await;
    assert!(ok, "{msg}");
    assert!(msg.contains("2 rows"), "{msg}");
    let own = session_dir(&s).join("out.csv");
    let (ok, msg) = call(
        &app,
        &s,
        "load_dataset",
        json!({"path": own.to_str().unwrap(), "name": "absolute"}),
    )
    .await;
    assert!(ok, "{msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn calls_without_a_scope_cannot_touch_files() {
    init();
    let err = p2a_mcp::path_jail::validate_data_path("anything.csv").unwrap_err();
    assert!(err.contains("needs a session"), "{err}");
    let err = p2a_mcp::path_jail::validate_data_path_for_write("anything.csv").unwrap_err();
    assert!(err.contains("needs a session"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_session_dirs_are_purged_on_first_use() {
    let app = app().await;
    let s = session(&app).await;
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "d", "csv_content": "x\n1\n"})
        )
        .await
        .0
    );
    assert!(
        call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "d", "path": "d.csv"})
        )
        .await
        .0
    );
    assert!(!init().join("sessions").join(STALE).exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleting_a_session_removes_its_directory() {
    let app = app().await;
    let s = session(&app).await;
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "d", "csv_content": "x\n1\n"})
        )
        .await
        .0
    );
    assert!(
        call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "d", "path": "d.csv"})
        )
        .await
        .0
    );
    let dir = session_dir(&s);
    assert!(dir.is_dir());
    delete(&app, &s).await;
    // The directory is removed off the runtime; poll for it (no timing assertion).
    for _ in 0..50 {
        if !dir.exists() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    panic!("the session directory was not removed");
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn symlinks_in_a_session_directory_are_not_followed() {
    let app = app().await;
    let s = session(&app).await;
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "d", "csv_content": "x\n1\n"})
        )
        .await
        .0
    );
    // The first export creates the session directory.
    assert!(
        call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "d", "path": "first.csv"})
        )
        .await
        .0
    );
    let outside = tempfile::tempdir().unwrap();
    let escaped = outside.path().join("escaped.csv");
    std::os::unix::fs::symlink(&escaped, session_dir(&s).join("link.csv")).unwrap();
    let (ok, msg) = call(
        &app,
        &s,
        "export_dataset",
        json!({"dataset": "d", "path": "link.csv"}),
    )
    .await;
    assert!(!ok, "export followed a symlink");
    assert!(msg.contains("symbolic link"), "{msg}");
    assert!(!escaped.exists());
    let (ok, msg) = call(&app, &s, "load_dataset", json!({"path": "link.csv"})).await;
    assert!(!ok, "load followed a symlink");
    assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    assert!(temp_files(&session_dir(&s)).is_empty());
}

/// `"i,j\n"` plus `"{i},{2i}\n"` for `i` in `0..n`.
fn numbered_csv(n: usize) -> String {
    let rows: String = (0..n).map(|i| format!("{i},{}\n", 2 * i)).collect();
    format!("i,j\n{rows}")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_disk_quota_refuses_exports_past_the_limit() {
    let app = app().await;
    let s = session(&app).await;
    // About 24 KB per export; the quota is 64 KiB.
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "n", "csv_content": numbered_csv(2500)})
        )
        .await
        .0
    );
    for name in ["a.csv", "b.csv"] {
        let (ok, msg) = call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "n", "path": name}),
        )
        .await;
        assert!(ok, "{name}: {msg}");
    }
    let (ok, msg) = call(
        &app,
        &s,
        "export_dataset",
        json!({"dataset": "n", "path": "c.csv"}),
    )
    .await;
    assert!(!ok, "the third export passed the quota");
    assert!(msg.contains("storage limit"), "{msg}");
    assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    assert!(!session_dir(&s).join("c.csv").exists());
    assert!(temp_files(&session_dir(&s)).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_messages_show_relative_paths() {
    let app = app().await;
    let s = session(&app).await;
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "d", "csv_content": "x,y\n1,1.1\n2,1.9\n"})
        )
        .await
        .0
    );
    // `sub` does not exist.
    let (ok, msg) = call(
        &app,
        &s,
        "export_dataset",
        json!({"dataset": "d", "path": "sub/x.csv"}),
    )
    .await;
    assert!(!ok, "{msg}");
    assert!(msg.contains("sub") && !msg.contains(root_str()), "{msg}");
    for path in ["x.csv", "x.parquet", "x.json"] {
        let (ok, msg) = call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "d", "path": path}),
        )
        .await;
        assert!(ok, "{path}: {msg}");
        assert!(msg.contains(path) && !msg.contains(root_str()), "{msg}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_dataset_refuses_files_above_the_load_limit() {
    let app = app().await;
    let s = session(&app).await;
    // about 2.8 KB on disk
    let rows: String = (0..400).map(|i| format!("{i},{}.5\n", i % 7)).collect();
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "big", "csv_content": format!("x,y\n{rows}")})
        )
        .await
        .0
    );
    assert!(
        call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "big", "path": "big.csv"})
        )
        .await
        .0
    );
    let (ok, msg) = call(
        &app,
        &s,
        "load_dataset",
        json!({"path": "big.csv", "name": "again"}),
    )
    .await;
    assert!(!ok, "a file above load_file_bytes was loaded");
    assert!(
        msg.contains("(load_file_bytes)") && msg.contains("2048"),
        "{msg}"
    );
    assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "small", "csv_content": "x,y\n1,1.1\n2,1.9\n"})
        )
        .await
        .0
    );
    assert!(
        call(
            &app,
            &s,
            "export_dataset",
            json!({"dataset": "small", "path": "small.csv"})
        )
        .await
        .0
    );
    let (ok, msg) = call(
        &app,
        &s,
        "load_dataset",
        json!({"path": "small.csv", "name": "again"}),
    )
    .await;
    assert!(ok, "{msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_dataset_respects_export_bytes() {
    let app = app().await;
    let s = session(&app).await;
    // About 48 KB as CSV, more than export_bytes = 32,768.
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "n", "csv_content": numbered_csv(5000)})
        )
        .await
        .0
    );
    let (ok, msg) = call(
        &app,
        &s,
        "export_dataset",
        json!({"dataset": "n", "path": "n.csv"}),
    )
    .await;
    assert!(!ok, "an export above export_bytes was written");
    assert!(
        msg.contains("export limit") && msg.contains("32768"),
        "{msg}"
    );
    assert!(!session_dir(&s).join("n.csv").exists());
    assert!(temp_files(&session_dir(&s)).is_empty());
}

/// A session holding the two-row `private` dataset.
async fn private_session(app: &Router) -> String {
    let s = session(app).await;
    assert!(
        call(
            app,
            &s,
            "create_dataset",
            json!({"name": "private", "csv_content": "patient,score\nalice,3\nbob,5\n"})
        )
        .await
        .0
    );
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_session_streams_to_file_and_respects_export_bytes() {
    let app = app().await;
    let s = private_session(&app).await;
    let (ok, msg) = call(&app, &s, "export_session", json!({"file_path": "s.json"})).await;
    assert!(ok, "{msg}");
    assert!(msg.contains("s.json") && !msg.contains(root_str()), "{msg}");
    let saved: Value =
        serde_json::from_slice(&std::fs::read(session_dir(&s).join("s.json")).unwrap()).unwrap();
    assert_eq!(
        saved["datasets"]["private"]["data"]["patient"],
        json!(["alice", "bob"])
    );
    // About 48 KB as session JSON, more than export_bytes = 32,768 (the
    // 2,500-row set is only about 23 KB, too small).
    assert!(
        call(
            &app,
            &s,
            "create_dataset",
            json!({"name": "big", "csv_content": numbered_csv(5000)})
        )
        .await
        .0
    );
    let (ok, msg) = call(&app, &s, "export_session", json!({"file_path": "big.json"})).await;
    assert!(!ok, "a session export above export_bytes was written");
    assert!(msg.contains("export limit"), "{msg}");
    assert!(!session_dir(&s).join("big.json").exists());
    assert!(temp_files(&session_dir(&s)).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_then_import_round_trips_in_a_session() {
    let app = app().await;
    let s = private_session(&app).await;
    let (ok, msg) = call(&app, &s, "export_session", json!({"file_path": "s.json"})).await;
    assert!(ok, "{msg}");
    let (ok, msg) = call(
        &app,
        &s,
        "import_session",
        json!({"file_path": "s.json", "merge": false}),
    )
    .await;
    assert!(ok, "{msg}");
    assert!(msg.contains("Datasets imported: 1"), "{msg}");
    assert!(msg.contains("s.json") && !msg.contains(root_str()), "{msg}");
    let (ok, msg) = call(&app, &s, "head_dataset", json!({"dataset": "private"})).await;
    assert!(ok, "{msg}");
    assert!(msg.contains("alice") && msg.contains("bob"), "{msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_session_refuses_files_above_the_import_limit() {
    let app = app().await;
    let s = private_session(&app).await;
    // The first export creates the session directory; then a 40 KB session
    // file (more than export_bytes = 32,768) is dropped straight into it.
    assert!(
        call(&app, &s, "export_session", json!({"file_path": "s.json"}))
            .await
            .0
    );
    let values: Vec<String> = (0..8_000).map(|i| i.to_string()).collect();
    let json_text = format!(
        r#"{{"datasets":{{"big":{{"data":{{"x":[{}]}}}}}}}}"#,
        values.join(",")
    );
    assert!(json_text.len() > 32_768 && json_text.len() < 65_536);
    std::fs::write(session_dir(&s).join("big.json"), json_text).unwrap();
    let (ok, msg) = call(
        &app,
        &s,
        "import_session",
        json!({"file_path": "big.json", "merge": false}),
    )
    .await;
    assert!(!ok, "a session file above the import limit was imported");
    assert!(msg.contains("import limit"), "{msg}");
    assert!(!msg.contains(root_str()), "leaks the root: {msg}");
    // Refused before anything was replaced.
    let (ok, msg) = call(&app, &s, "head_dataset", json!({"dataset": "private"})).await;
    assert!(ok, "{msg}");
}
