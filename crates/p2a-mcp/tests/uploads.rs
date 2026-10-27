//! Where `upload_dataset` keeps an uploaded Parquet file (PR-23): nowhere.
//! The file is decoded from the request's bytes, so an upload needs no
//! folder and stores nothing: not in the session's directory, whose storage
//! quota would then count it, and not in the server's temporary folder,
//! which lies outside the data root and every quota.
//!
//! The server is built in-process with session directories on, as in the
//! open public mode, and driven with `tower::ServiceExt::oneshot`. The
//! process's temporary folder is a regular file, so nothing can be created
//! in it. Every test in this binary shares that environment (set up once in
//! [`init`]) and uses its own sessions.
#![cfg(feature = "http")]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use base64::Engine;
use p2a_mcp::config::{ServerConfig, SessionDirsSetting};
use p2a_mcp::tools::limits::{self, Limits};
use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};
use serde_json::{Value, json};
use tower::ServiceExt;

static ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The budgets every test in this binary runs under: a session storage
/// quota, as in the hosted profile.
fn overrides() -> Vec<String> {
    vec!["session_disk_bytes=65536".to_string()]
}

/// The shared data root, set up once per test binary.
fn init() -> &'static Path {
    ROOT.get_or_init(|| {
        let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("p2a-uploads-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(&root).unwrap();
        // A file where the temporary folder should be: nothing can be
        // created in it.
        let not_a_folder = base.join("tmp");
        std::fs::write(&not_a_folder, "not a folder\n").unwrap();
        // SAFETY: runs inside OnceLock::get_or_init, which every test calls first and which
        // blocks the other test threads until it returns, so no thread reads the environment
        // concurrently.
        unsafe {
            std::env::set_var("P2A_DATA_ROOT", &root);
            std::env::set_var("TMPDIR", &not_a_folder);
        }
        p2a_mcp::path_jail::configure_session_dirs(true);
        limits::install(Limits::off().with_overrides(&overrides()).unwrap());
        root.canonicalize().unwrap()
    })
}

/// A fresh app over the shared root: loopback, no token (Local mode),
/// session directories on.
async fn app() -> Router {
    init();
    let mut cfg = ServerConfig::default();
    cfg.http.session_dirs = SessionDirsSetting::On;
    cfg.limits.overrides = overrides();
    p2a_mcp::transport::http::build_app(&cfg).await.unwrap().0
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

/// A Parquet file of `rows` rows: an integer, a noisy float and a label.
fn parquet_bytes(rows: usize) -> Vec<u8> {
    let x: Vec<i64> = (0..rows as i64).collect();
    let y: Vec<f64> = (0..rows)
        .map(|i| i as f64 * 1.1 + (i % 7) as f64 * 0.3)
        .collect();
    let label: Vec<String> = (0..rows).map(|i| format!("row-{i:04}")).collect();
    let mut df = DataFrame::new(vec![
        Series::new("x".into(), x).into(),
        Series::new("y".into(), y).into(),
        Series::new("label".into(), label).into(),
    ])
    .unwrap();
    let mut out = Vec::new();
    ParquetWriter::new(&mut out).finish(&mut df).unwrap();
    out
}

/// Every regular file under `dir`, relative to it, sorted.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn a_parquet_upload_is_decoded_from_the_request_and_writes_no_file() {
    // Pre-fix the upload was first written to the server's temporary
    // folder, outside the data root, the session's directory and its
    // storage quota, and loaded from there; where that folder could not be
    // written, every Parquet upload failed ("Failed to write temp file").
    let root = init();
    assert!(std::env::temp_dir().is_file());
    let app = app().await;
    let sid = session(&app).await;
    let content = base64::engine::general_purpose::STANDARD.encode(parquet_bytes(200));
    let before = files_under(root);

    let (ok, text) = call(
        &app,
        &sid,
        "upload_dataset",
        json!({"content": content, "filename": "small.parquet", "name": "up"}),
    )
    .await;
    assert!(ok, "{text}");
    assert!(text.contains("200 rows x 3 columns"), "{text}");
    assert_eq!(files_under(root), before, "the upload stored a file");

    let (ok, head) = call(&app, &sid, "head_dataset", json!({"dataset": "up", "n": 3})).await;
    assert!(ok, "{head}");
    assert!(head.contains("row-0002"), "{head}");
    assert!(head.contains("2.8"), "{head}");
}

#[tokio::test]
async fn bytes_that_are_not_parquet_are_refused_without_a_file() {
    // The loader's error comes from the bytes alone: it names no file.
    let root = init();
    let app = app().await;
    let sid = session(&app).await;
    let content = base64::engine::general_purpose::STANDARD.encode(b"x,y\n1,2\n");
    let before = files_under(root);

    let (ok, text) = call(
        &app,
        &sid,
        "upload_dataset",
        json!({"content": content, "filename": "not.parquet"}),
    )
    .await;
    assert!(!ok, "{text}");
    assert!(text.starts_with("Failed to parse Parquet"), "{text}");
    assert!(!text.contains(".parquet"), "{text}");
    assert!(!text.contains(root.to_str().unwrap()), "{text}");
    assert_eq!(files_under(root), before, "the upload stored a file");
}
