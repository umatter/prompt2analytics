//! Per-session data directories across a restart with a RocksDB session
//! store: the stale-directory purge (T0-WP09) and the adoption of stored
//! sessions (T0-WP07) together.
//!
//! What the "earlier process" left is set up before this process uses any
//! path: session directories with files under the data root, and the rows of
//! the store, which a child process of this binary writes ([`earlier_process`])
//! because a RocksDB store stays locked until the process that opened it
//! exits. The server is then built in-process with `build_app` over that
//! store, session directories on, and driven with
//! `tower::ServiceExt::oneshot`. The purge runs once per process, at the first
//! scoped file access, so the whole sequence is one test.
#![cfg(all(feature = "http", feature = "db"))]

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use chrono::Utc;
use p2a_mcp::config::{ServerConfig, SessionDirsSetting};
use p2a_mcp::db::{DatasetMeta, DbConnection};
use serde_json::{Value, json};
use tower::ServiceExt;

/// A stored session within the TTL that a client comes back for: adopted,
/// and its datasets reload from its directory.
const KEPT: &str = "5a0c9e2b-7d14-4f36-8b21-3c9d0e4f6a71";
/// A stored session within the TTL that nobody comes back for: its directory
/// stays until its row ages out and the sweep deletes it.
const LATER: &str = "6b1d0f3c-8e25-4a47-9c32-4d0e1f5a7b82";
/// A stored session within the TTL that is deleted without being adopted.
const DELETED: &str = "7c2e1a4d-9f36-4b58-8d43-5e1f2a6b8c93";
/// A stored session past the TTL: it cannot be adopted, so its directory is
/// stale.
const EXPIRED: &str = "8d3f2b5e-a047-4c69-9e54-6f2a3b7c9da4";
/// A directory without a stored session.
const STALE: &str = "9e4a3c6f-b158-4d7a-8f65-7a3b4c8daeb5";

/// The temporary file an export into `KEPT` left when the earlier process
/// stopped while writing it.
const LEFTOVER: &str = ".out.csv.0b5c4d7e-c269-4e8b-a076-8b4c5d9ebfc6.p2a-tmp";

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, "localhost");
    let req = match body {
        Some(b) => req
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    }
    .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// The rows the earlier process left in the store at `db_path`.
async fn seed_store(db_path: &Path) {
    let db = DbConnection::connect_rocksdb(db_path).await.unwrap();
    let now = Utc::now();
    for id in [KEPT, LATER, DELETED] {
        db.set_session_last_accessed(id, now).await.unwrap();
    }
    db.set_session_last_accessed(EXPIRED, now - chrono::Duration::hours(3))
        .await
        .unwrap();
    let mut meta = DatasetMeta::new(
        KEPT.to_string(),
        "mine".to_string(),
        "csv".to_string(),
        3,
        2,
        vec!["x".to_string(), "y".to_string()],
    );
    meta.source_path = Some("mine.csv".to_string());
    db.save_dataset_meta(&meta).await.unwrap();
}

/// Names the store [`earlier_process`] seeds, in the child process only.
const SEED_DB_VAR: &str = "P2A_TEST_RESTART_SEED_DB";

/// The earlier process: run by the main test as a child process of this
/// binary, because a RocksDB store opened in a process stays locked until
/// that process exits. Does nothing when the binary's tests run normally.
#[test]
fn earlier_process() {
    let Some(db_path) = std::env::var_os(SEED_DB_VAR) else {
        return;
    };
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(seed_store(Path::new(&db_path)));
}

/// Run [`earlier_process`] in a child process over `db_path`.
fn run_earlier_process(db_path: &Path) {
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "earlier_process", "--test-threads=1"])
        .env(SEED_DB_VAR, db_path)
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success(), "the earlier process failed: {status}");
}

/// The session directories the earlier process left under `root`.
fn seed_dirs(root: &Path) {
    for id in [KEPT, LATER, DELETED, EXPIRED, STALE] {
        let dir = root.join("sessions").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mine.csv"), "x,y\n1,2.1\n2,3.9\n3,6.2\n").unwrap();
    }
    let leftover = root.join("sessions").join(KEPT).join(LEFTOVER);
    std::fs::write(&leftover, "x,y\n1,2.1\n2,").unwrap();
    let an_hour_ago = SystemTime::now() - Duration::from_secs(3600);
    std::fs::File::options()
        .write(true)
        .open(&leftover)
        .unwrap()
        .set_modified(an_hour_ago)
        .unwrap();
}

fn exists(path: PathBuf) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stored_sessions_keep_their_directories_across_a_restart() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("p2a-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let root = base.join("data");
    let db_path = base.join("db");
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    seed_dirs(&root);
    run_earlier_process(&db_path);

    // SAFETY: std serialises its own reads and writes of the environment;
    // the other test in this binary only reads one variable through std, and
    // nothing in this process has started a thread that reads it otherwise.
    unsafe { std::env::set_var("P2A_DATA_ROOT", &root) };
    p2a_mcp::path_jail::configure_session_dirs(true);
    let mut cfg = ServerConfig::default();
    cfg.http.session_dirs = SessionDirsSetting::On;
    cfg.http.db_path = Some(db_path.to_string_lossy().into_owned());
    let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
    let pm = state
        .persistent_manager
        .clone()
        .expect("persistent manager");
    let dir = |id: &str| root.join("sessions").join(id);

    // A stored session deleted before anyone came back for it takes its
    // directory with it.
    let (status, body) = send(&app, "DELETE", &format!("/api/sessions/{DELETED}"), None).await;
    assert!(status.is_success(), "{status} {body}");
    assert!(
        !exists(dir(DELETED)),
        "the deleted stored session's directory is still there"
    );

    // The client comes back: the session is adopted and its dataset reloads
    // from its directory (the first scoped file access, which purges).
    let (status, body) = send(&app, "GET", &format!("/api/sessions/{KEPT}"), None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send(
        &app,
        "POST",
        &format!("/api/sessions/{KEPT}/datasets/reload"),
        Some(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"]["succeeded"], json!(["mine"]), "{body}");
    assert!(exists(dir(KEPT).join("mine.csv")));
    assert!(
        !exists(dir(KEPT).join(LEFTOVER)),
        "the earlier process's temporary file is still in the adopted session's directory"
    );

    // What cannot come back was purged; a session that can is kept.
    assert!(!exists(dir(STALE)), "a directory without a stored session");
    assert!(!exists(dir(EXPIRED)), "a stored session past the TTL");
    assert!(exists(dir(LATER).join("mine.csv")));

    // Once the unadopted session's row ages out, the sweep deletes the row
    // and the directory goes with it.
    let long_ago = Utc::now() - chrono::Duration::hours(3);
    assert!(
        pm.db()
            .refresh_session_last_accessed(LATER, long_ago)
            .await
            .unwrap()
    );
    pm.sweep().await.unwrap();
    assert!(pm.db().get_session(LATER).await.unwrap().is_none());
    assert!(
        !exists(dir(LATER)),
        "the expired stored session's directory is still there"
    );

    // The adopted session is untouched by all of this.
    assert!(exists(dir(KEPT).join("mine.csv")));
    let _ = std::fs::remove_dir_all(&base);
}
