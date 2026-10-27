//! Integration tests for the session registry, conversation ownership and the
//! bounded conversation store (T0-WP07).
//!
//! Servers are built in-process with `build_app` and driven with
//! `tower::ServiceExt::oneshot`; nothing binds a socket. Every configuration
//! here keeps per-session data directories off.
#![cfg(all(feature = "http", feature = "db"))]

use std::future::Future;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use p2a_mcp::config::ServerConfig;
use p2a_mcp::db::DatasetMeta;
use p2a_mcp::path_jail::allowed_data_root;
use p2a_mcp::persistent_session::{PersistentSessionManager, RELOAD_LABEL};
use p2a_mcp::tools::limits;
use p2a_mcp::transport::http::{AppState, build_app};
use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};
use polars_parquet_format::FileMetaData;
use polars_parquet_format::thrift::protocol::{TCompactInputProtocol, TCompactOutputProtocol};
use serde_json::{Value, json};
use tower::ServiceExt;

async fn app(cfg: ServerConfig) -> (Router, AppState) {
    build_app(&cfg).await.expect("build_app")
}

/// Send one request; `session` sets the `X-P2A-Session-Id` header.
async fn call(
    r: &Router,
    method: &str,
    uri: &str,
    session: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let (status, _, body) = call_with_headers(r, method, uri, session, body).await;
    (status, body)
}

/// [`call`], also returning the response headers.
async fn call_with_headers(
    r: &Router,
    method: &str,
    uri: &str,
    session: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, HeaderMap, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(sid) = session {
        req = req.header("x-p2a-session-id", sid);
    }
    let req = match body {
        Some(b) => req
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    }
    .unwrap();
    let resp = r.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 << 20)
        .await
        .unwrap();
    (
        status,
        headers,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn new_session(r: &Router) -> String {
    let (st, body) = call(r, "POST", "/api/sessions", None, Some(json!({}))).await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    body["data"]["session_id"].as_str().unwrap().to_string()
}

async fn new_conversation(r: &Router, sid: &str, title: &str) -> String {
    let (st, body) = call(
        r,
        "POST",
        "/api/conversations",
        Some(sid),
        Some(json!({"session_id": sid, "title": title})),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    body["data"]["id"].as_str().unwrap().to_string()
}

async fn add_msg(r: &Router, sid: &str, conv: &str, content: &str) -> (StatusCode, Value) {
    call(
        r,
        "POST",
        &format!("/api/conversations/{conv}/messages"),
        Some(sid),
        Some(json!({"role": "user", "content": content})),
    )
    .await
}

fn pm(s: &AppState) -> &Arc<PersistentSessionManager> {
    s.persistent_manager.as_ref().expect("persistent manager")
}

/// Poll `f` until it returns true, up to 250 x 20 ms. No timing assertion.
async fn wait_until<F, Fut>(f: F) -> bool
where
    F: Fn() -> Fut,
    Fut: Future<Output = bool>,
{
    for _ in 0..250 {
        if f().await {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    false
}

/// What the process logs at INFO and above, from the first call on. The
/// subscriber is the global default because a governed call logs on a
/// blocking-pool thread, which a test thread's own default never reaches.
/// Every test in this binary writes here once it is installed, so look a
/// line up by something only one test logs, such as its session id.
fn process_logs() -> &'static ProcessLogs {
    static LOGS: std::sync::OnceLock<ProcessLogs> = std::sync::OnceLock::new();
    LOGS.get_or_init(|| {
        let logs = ProcessLogs::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("no other global subscriber in this test binary");
        logs
    })
}

#[derive(Clone, Default)]
struct ProcessLogs(Arc<std::sync::Mutex<Vec<u8>>>);

impl ProcessLogs {
    /// The lines logged so far that contain `needle`.
    fn lines_with(&self, needle: &str) -> Vec<String> {
        String::from_utf8_lossy(&self.0.lock().unwrap())
            .lines()
            .filter(|l| l.contains(needle))
            .map(str::to_string)
            .collect()
    }
}

impl std::io::Write for ProcessLogs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// =============================================================================
// Task 1: one session registry (MCP-2; server half of MCP-8)
// =============================================================================

/// The copy of the session the persistent manager used to hold went stale
/// while tools kept the live session fresh; the next conversation then
/// deleted every row of the session and answered 410.
#[tokio::test]
async fn new_conversation_after_registration_ttl_keeps_history() {
    let mut cfg = ServerConfig::default();
    cfg.session.ttl_minutes = 1;
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;
    let c1 = new_conversation(&r, &sid, "first").await;
    assert_eq!(add_msg(&r, &sid, &c1, "hello").await.0, StatusCode::CREATED);
    let live = state.session_manager.get_session(&sid).await.unwrap();
    let held = pm(&state).get_session(&sid).await.unwrap();
    *held.last_accessed.write().await = chrono::Utc::now() - chrono::Duration::minutes(5);
    live.touch().await; // tool activity refreshes the live session
    let (st, body) = call(
        &r,
        "POST",
        "/api/conversations",
        Some(&sid),
        Some(json!({"session_id": sid, "title": "second"})),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let (st, body) = call(
        &r,
        "GET",
        &format!("/api/conversations/{c1}/messages"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert!(pm(&state).db().get_session(&sid).await.unwrap().is_some());
}

/// Reload loads into the live session and validates every stored path
/// through the path jail; failure texts carry no server paths.
#[tokio::test]
async fn reload_restores_datasets_into_the_live_session() {
    let (r, state) = app(ServerConfig::default()).await;
    let sid = new_session(&r).await;

    let root = allowed_data_root().unwrap();
    let inside = tempfile::Builder::new()
        .prefix(".p2a-wp07-")
        .tempdir_in(&root)
        .unwrap();
    let inside_csv = inside.path().join("d.csv");
    std::fs::write(&inside_csv, "x,y\n1,1.1\n2,1.9\n3,3.2\n").unwrap();
    // Removed when `outside` drops at the end of the test.
    let outside = tempfile::Builder::new()
        .prefix("p2a-wp07-outside-")
        .suffix(".csv")
        .tempfile_in(std::env::temp_dir())
        .unwrap();
    let outside_csv = outside.path().to_path_buf();
    std::fs::write(&outside_csv, "x,y\n1,1.1\n2,1.9\n3,3.2\n").unwrap();
    assert!(!outside_csv.canonicalize().unwrap().starts_with(&root));

    for (name, path) in [("d", &inside_csv), ("o", &outside_csv)] {
        let mut meta = DatasetMeta::new(
            sid.clone(),
            name.to_string(),
            "csv".to_string(),
            3,
            2,
            vec!["x".to_string(), "y".to_string()],
        );
        meta.source_path = Some(path.to_string_lossy().into_owned());
        pm(&state).db().save_dataset_meta(&meta).await.unwrap();
    }

    let (st, body) = call(
        &r,
        "POST",
        &format!("/api/sessions/{sid}/datasets/reload"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert_eq!(body["data"]["succeeded"], json!(["d"]), "{body}");
    assert_eq!(body["data"]["failed"][0]["name"], "o", "{body}");
    assert!(
        !body
            .to_string()
            .contains(&*allowed_data_root().unwrap().to_string_lossy()),
        "{body}"
    );
    assert!(
        state
            .session_manager
            .get_session(&sid)
            .await
            .unwrap()
            .datasets
            .read()
            .await
            .contains_key("d")
    );
}

/// A CSV of `rows` rows with two float columns, y = 2x plus noise.
fn noisy_csv(rows: usize, seed: usize) -> String {
    let mut csv = String::from("x,y\n");
    for i in 0..rows {
        let x = i as f64 + 0.25;
        let noise = ((i * 7919 + seed * 104_729) % 1000) as f64 / 1000.0 - 0.5;
        csv.push_str(&format!("{x},{}\n", 2.0 * x + noise));
    }
    csv
}

/// Store a dataset_meta row for `name` loaded from `path`, `age_minutes`
/// old (a reload goes through the rows newest first).
async fn store_meta(
    state: &AppState,
    sid: &str,
    name: &str,
    path: &std::path::Path,
    age_minutes: i64,
) {
    let mut meta = DatasetMeta::new(
        sid.to_string(),
        name.to_string(),
        "csv".to_string(),
        0,
        2,
        vec!["x".to_string(), "y".to_string()],
    );
    meta.source_path = Some(path.to_string_lossy().into_owned());
    meta.loaded_at = (chrono::Utc::now() - chrono::Duration::minutes(age_minutes)).into();
    pm(state).db().save_dataset_meta(&meta).await.unwrap();
}

/// A reload runs under the governor's dataset quotas: with two stored
/// datasets that together pass the session's quota, the second is reported
/// failed and the session stays within the quota.
#[tokio::test]
async fn reload_respects_the_session_dataset_quota() {
    let quota: u64 = 1 << 20;
    let mut cfg = ServerConfig::default();
    cfg.exec.session_data_mb = Some(1);
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;

    let dir = tempfile::Builder::new()
        .prefix(".p2a-wp07-")
        .tempdir_in(allowed_data_root().unwrap())
        .unwrap();
    // 40,000 rows x 2 float columns: 640,000 bytes in memory each, so one
    // fits the 1 MiB quota and two do not.
    for (age, name) in [(0, "first"), (1, "second")] {
        let path = dir.path().join(format!("{name}.csv"));
        std::fs::write(&path, noisy_csv(40_000, age as usize)).unwrap();
        store_meta(&state, &sid, name, &path, age).await;
    }

    let (st, body) = call(
        &r,
        "POST",
        &format!("/api/sessions/{sid}/datasets/reload"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert_eq!(body["data"]["succeeded"], json!(["first"]), "{body}");
    assert_eq!(body["data"]["failed"][0]["name"], "second", "{body}");
    // The refusal names the per-session limit.
    let error = body["data"]["failed"][0]["error"]
        .as_str()
        .unwrap_or_default();
    let limit = p2a_core::memory::format_bytes(1 << 20);
    assert!(
        error.contains(&format!("(limit {limit} per session)")),
        "{body}"
    );
    let session = state.session_manager.get_session(&sid).await.unwrap();
    assert!(!session.datasets.read().await.contains_key("second"));
    let bytes = session.refresh_data_bytes().await;
    assert!(bytes > 0 && bytes <= quota, "{bytes}");
}

/// A reload checks each source file against `load_file_bytes` before it
/// parses it, as `load_dataset` does: a file that grew past the budget after
/// it was loaded is reported failed with its size and the limit (and no
/// server path), and the session does not get it. The refusal's log line
/// carries the session id, so an operator can tie it to the session.
#[test]
fn reload_refuses_a_source_above_load_file_bytes() {
    let logs = process_logs();
    let big_csv = noisy_csv(200, 3);
    let size = big_csv.len() as u64;
    let capped = limits::Limits::off()
        .with_overrides(&[format!("load_file_bytes={}", size - 1)])
        .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // The limits are this thread's (`with_limits`, not a process-wide
    // install); the current-thread runtime runs the route on it, and the
    // governor hands them to the blocking thread that loads the files.
    limits::with_limits(capped, || {
        rt.block_on(async {
            let (r, state) = app(ServerConfig::default()).await;
            let sid = new_session(&r).await;
            let root = allowed_data_root().unwrap();
            let dir = tempfile::Builder::new()
                .prefix(".p2a-pr12-")
                .tempdir_in(&root)
                .unwrap();
            let (small, big) = (dir.path().join("small.csv"), dir.path().join("big.csv"));
            std::fs::write(&small, noisy_csv(3, 1)).unwrap();
            std::fs::write(&big, &big_csv).unwrap();
            store_meta(&state, &sid, "small", &small, 0).await;
            store_meta(&state, &sid, "big", &big, 1).await;

            let (st, body) = call(
                &r,
                "POST",
                &format!("/api/sessions/{sid}/datasets/reload"),
                Some(&sid),
                None,
            )
            .await;
            assert_eq!(st, StatusCode::OK, "{body}");
            assert_eq!(body["data"]["succeeded"], json!(["small"]), "{body}");
            assert_eq!(body["data"]["failed"][0]["name"], "big", "{body}");
            let error = body["data"]["failed"][0]["error"]
                .as_str()
                .unwrap_or_default();
            assert!(
                error.contains("(load_file_bytes)")
                    && error.contains(&size.to_string())
                    && error.contains(&(size - 1).to_string()),
                "{body}"
            );
            assert!(!error.contains(&*root.to_string_lossy()), "{body}");
            let session = state.session_manager.get_session(&sid).await.unwrap();
            assert!(session.datasets.read().await.contains_key("small"));
            assert!(!session.datasets.read().await.contains_key("big"));

            // The session id must come from the reload span's own field, not
            // from anywhere in the line (the HTTP trace span's URI holds it
            // too, should that span ever reach this subscriber's level).
            let reload_span = format!("{RELOAD_LABEL}{{session_id={sid}}}");
            let refusals = logs.lines_with("request refused by a named budget");
            assert!(
                refusals
                    .iter()
                    .any(|l| l.contains("dataset=big") && l.contains(&reload_span)),
                "no refusal logged in span {reload_span}: {refusals:#?}"
            );
        })
    });
}

/// Write a parquet file of `rows` rows of one float column at `path`, the
/// values cycling through 97 noisy levels: dictionary-encoded, so a few
/// kilobytes on disk and 8 bytes a row once decoded.
fn levels_parquet(path: &std::path::Path, rows: usize) {
    let values: Vec<f64> = (0..rows).map(|i| (i % 97) as f64 * 0.37 + 0.1).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    ParquetWriter::new(std::fs::File::create(path).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// POST the reload route for `sid`; the route answers 200 with the result.
async fn reload(r: &Router, sid: &str) -> Value {
    let (st, body) = call(
        r,
        "POST",
        &format!("/api/sessions/{sid}/datasets/reload"),
        Some(sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    body
}

/// A reload runs the check `load_dataset` runs before it decodes a parquet
/// file (PR-H2). A stored parquet path now holding a file whose decoded size
/// exceeds the session's quota is reported failed with `load_dataset`'s
/// refusal before anything is decoded ("Nothing was loaded"), with no server
/// path; the other stored dataset still reloads, and the dataset the file
/// held when it was last loaded stays as it was. The refusal's log line
/// carries its text, so an operator sees the size and the limit.
#[tokio::test]
async fn reload_refuses_a_parquet_source_past_the_quota_before_decoding_it() {
    let logs = process_logs();
    let mut cfg = ServerConfig::default();
    cfg.exec.session_data_mb = Some(8);
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;
    let root = allowed_data_root().unwrap();
    let dir = tempfile::Builder::new()
        .prefix(".p2a-prh2-")
        .tempdir_in(&root)
        .unwrap();
    let (small, stored) = (dir.path().join("small.csv"), dir.path().join("p.parquet"));
    std::fs::write(&small, noisy_csv(50, 4)).unwrap();
    // 1,000 rows when it is first loaded.
    levels_parquet(&stored, 1_000);
    store_meta(&state, &sid, "small", &small, 0).await;
    store_meta(&state, &sid, "p", &stored, 1).await;
    let body = reload(&r, &sid).await;
    assert_eq!(body["data"]["succeeded"], json!(["small", "p"]), "{body}");

    // Replaced at the stored path by 4,000,000 rows: 32 MB once decoded.
    // Pre-fix the reload decoded it in full and only then refused it at the
    // insert ("this session's datasets would use 30.53 MB ...").
    levels_parquet(&stored, 4_000_000);
    let body = reload(&r, &sid).await;
    assert_eq!(body["data"]["succeeded"], json!(["small"]), "{body}");
    assert_eq!(body["data"]["failed"][0]["name"], "p", "{body}");
    let error = body["data"]["failed"][0]["error"]
        .as_str()
        .unwrap_or_default();
    let limit = p2a_core::memory::format_bytes(8 << 20);
    assert!(
        error.starts_with("Loading this file takes at least")
            && error.contains("Nothing was loaded")
            && error.contains(&format!("(limit {limit} per session)")),
        "{body}"
    );
    assert!(!error.contains(&*root.to_string_lossy()), "{body}");
    let session = state.session_manager.get_session(&sid).await.unwrap();
    let datasets = session.datasets.read().await;
    assert_eq!(datasets["p"].nrows(), 1_000);
    assert_eq!(datasets["small"].nrows(), 50);

    // Pre-fix (round 2) the line named the dataset but not the refusal.
    let reload_span = format!("{RELOAD_LABEL}{{session_id={sid}}}");
    let refusals = logs.lines_with("reload source refused by the dataset quotas before decoding");
    assert!(
        refusals.iter().any(|l| l.contains(&reload_span)
            && l.contains("dataset=p")
            && l.contains(&format!("refusal={error}"))),
        "no refusal with its text logged in span {reload_span}: {refusals:#?}"
    );
}

/// Write a parquet file of `rows` rows of one noisy float column at `path`,
/// in row groups of `group_rows` rows.
fn grouped_parquet(path: &std::path::Path, rows: usize, group_rows: usize) {
    let values: Vec<f64> = (0..rows).map(|i| (i % 31) as f64 * 0.5 + 0.25).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    ParquetWriter::new(std::fs::File::create(path).unwrap())
        .with_row_group_size(Some(group_rows))
        .finish(&mut df)
        .unwrap();
}

/// Rewrite the footer of the parquet file at `path` with `edit`, leaving the
/// pages where they are (the helper of the same name in
/// tests/dataset_quotas.rs).
fn rewrite_footer(path: &std::path::Path, edit: impl FnOnce(&mut FileMetaData)) {
    let bytes = std::fs::read(path).unwrap();
    let n = bytes.len();
    assert_eq!(&bytes[n - 4..], b"PAR1");
    let len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let start = n - 8 - len;
    let mut input = TCompactInputProtocol::new(&bytes[start..n - 8], usize::MAX);
    let mut footer = FileMetaData::read_from_in_protocol(&mut input).unwrap();
    edit(&mut footer);
    let mut written = Vec::new();
    footer
        .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut written))
        .unwrap();
    let mut out = bytes[..start].to_vec();
    out.extend_from_slice(&written);
    out.extend_from_slice(&u32::try_from(written.len()).unwrap().to_le_bytes());
    out.extend_from_slice(b"PAR1");
    std::fs::write(path, out).unwrap();
}

/// A reload refuses from its footer a stored parquet file that the check
/// before decoding will not read, as `load_dataset` does, with or without
/// dataset quotas (PR-H2, round-1 follow-up). The stored path is reloaded
/// once and then replaced by a file whose row groups point at the same
/// column chunk, which no writer produces: it is reported in `failed` with
/// the check's reason and no server path, the other stored dataset still
/// reloads, and the dataset the file held when it was last loaded stays.
/// Pre-fix the reload handed the file to the loader, which read the shared
/// chunk once per row group and replaced the dataset with 4,000 rows.
async fn reload_refuses_a_parquet_source_it_will_not_read_under(cfg: ServerConfig) {
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;
    let root = allowed_data_root().unwrap();
    let dir = tempfile::Builder::new()
        .prefix(".p2a-prh2-")
        .tempdir_in(&root)
        .unwrap();
    let (small, stored) = (dir.path().join("small.csv"), dir.path().join("p.parquet"));
    std::fs::write(&small, noisy_csv(50, 5)).unwrap();
    levels_parquet(&stored, 1_000);
    store_meta(&state, &sid, "small", &small, 0).await;
    store_meta(&state, &sid, "p", &stored, 1).await;
    let body = reload(&r, &sid).await;
    assert_eq!(body["data"]["succeeded"], json!(["small", "p"]), "{body}");

    grouped_parquet(&stored, 4_000, 1_000);
    rewrite_footer(&stored, |footer| {
        assert_eq!(footer.row_groups.len(), 4);
        let shared = footer.row_groups[0].columns[0].meta_data.clone();
        for group in footer.row_groups.iter_mut().skip(1) {
            group.columns[0].meta_data = shared.clone();
        }
    });
    let body = reload(&r, &sid).await;
    assert_eq!(body["data"]["succeeded"], json!(["small"]), "{body}");
    assert_eq!(body["data"]["failed"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["data"]["failed"][0]["name"], "p", "{body}");
    let error = body["data"]["failed"][0]["error"]
        .as_str()
        .unwrap_or_default();
    assert_eq!(
        error, "the file could not be loaded: two column chunks of the file share bytes",
        "{body}"
    );
    assert!(!error.contains(&*root.to_string_lossy()), "{body}");
    let session = state.session_manager.get_session(&sid).await.unwrap();
    let datasets = session.datasets.read().await;
    assert_eq!(datasets["p"].nrows(), 1_000);
    assert_eq!(datasets["small"].nrows(), 50);
}

#[tokio::test]
async fn reload_refuses_a_parquet_source_it_will_not_read_without_quotas() {
    reload_refuses_a_parquet_source_it_will_not_read_under(ServerConfig::default()).await;
}

#[tokio::test]
async fn reload_refuses_a_parquet_source_it_will_not_read_under_a_quota() {
    let mut cfg = ServerConfig::default();
    cfg.exec.session_data_mb = Some(8);
    reload_refuses_a_parquet_source_it_will_not_read_under(cfg).await;
}

/// What a reload's check before decoding reserves against the total quota
/// is released once the file is in or refused. A regression guard, not
/// evidence of the check: without it nothing is reserved. Five reloads of a
/// 300,000-row parquet file (2.4 MB decoded, admitted at 3.2 MB with what
/// decoding it allocates) under a 6 MiB total each replace the dataset by
/// name; a reservation the first left behind would refuse the second.
#[tokio::test]
async fn reload_releases_what_its_check_before_decoding_reserves() {
    let mut cfg = ServerConfig::default();
    cfg.exec.total_data_mb = Some(6);
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;
    let dir = tempfile::Builder::new()
        .prefix(".p2a-prh2-")
        .tempdir_in(allowed_data_root().unwrap())
        .unwrap();
    let stored = dir.path().join("p.parquet");
    levels_parquet(&stored, 300_000);
    store_meta(&state, &sid, "p", &stored, 0).await;
    for round in 0..5 {
        let body = reload(&r, &sid).await;
        assert_eq!(
            body["data"]["succeeded"],
            json!(["p"]),
            "round {round}: {body}"
        );
    }
}

/// Only `load_dataset` records the file it loaded as a dataset's source. A
/// `path` argument to `upload_dataset` or `create_dataset`, which neither
/// tool reads, is not stored, so a reload skips those datasets instead of
/// replacing them with that file.
#[cfg(feature = "llm")]
#[tokio::test]
async fn only_load_dataset_records_a_source_path() {
    use base64::Engine;

    let (r, state) = app(ServerConfig::default()).await;
    let sid = new_session(&r).await;
    let root = allowed_data_root().unwrap();
    let dir = tempfile::Builder::new()
        .prefix(".p2a-pr12-")
        .tempdir_in(&root)
        .unwrap();
    std::fs::write(dir.path().join("file.csv"), noisy_csv(50, 2)).unwrap();
    let file = dir
        .path()
        .strip_prefix(&root)
        .unwrap()
        .join("file.csv")
        .to_string_lossy()
        .into_owned();
    let small = noisy_csv(3, 5);
    let calls = [
        ("load_dataset", json!({"path": file, "name": "loaded"})),
        (
            "upload_dataset",
            json!({
                "content": base64::engine::general_purpose::STANDARD.encode(&small),
                "filename": "small.csv",
                "name": "uploaded",
                "path": file,
            }),
        ),
        (
            "create_dataset",
            json!({"csv_content": small, "name": "created", "path": file}),
        ),
    ];
    for (tool, arguments) in calls {
        let (st, body) = call(
            &r,
            "POST",
            &format!("/api/tools/{tool}"),
            Some(&sid),
            Some(json!({"session_id": sid, "arguments": arguments})),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{tool}: {body}");
        assert_eq!(body["data"]["success"], true, "{tool}: {body}");
    }

    let (st, body) = call(
        &r,
        "GET",
        &format!("/api/sessions/{sid}/datasets"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let source = |name: &str| {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["name"] == name)
            .map(|d| d["source_path"].clone())
    };
    assert_eq!(source("loaded"), Some(json!(file)), "{body}");
    assert_eq!(source("uploaded"), Some(Value::Null), "{body}");
    assert_eq!(source("created"), Some(Value::Null), "{body}");

    let (st, body) = call(
        &r,
        "POST",
        &format!("/api/sessions/{sid}/datasets/reload"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert_eq!(body["data"]["succeeded"], json!(["loaded"]), "{body}");
    let mut skipped: Vec<&str> = body["data"]["skipped"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["name"].as_str())
        .collect();
    skipped.sort_unstable();
    assert_eq!(skipped, ["created", "uploaded"], "{body}");
    let session = state.session_manager.get_session(&sid).await.unwrap();
    let datasets = session.datasets.read().await;
    for name in ["uploaded", "created"] {
        assert_eq!(datasets[name].nrows(), 3, "{name}");
    }
}

/// A reload is admitted like a tool call: above the memory soft limit it is
/// refused with the governor's 503.
#[tokio::test]
async fn reload_is_refused_under_memory_pressure() {
    if p2a_core::get_process_memory().is_none() {
        return; // the platform does not report the process memory
    }
    let mut cfg = ServerConfig::default();
    cfg.exec.memory_soft_limit_mb = Some(1);
    let (r, _state) = app(cfg).await;
    let sid = new_session(&r).await;
    let (st, headers, body) = call_with_headers(
        &r,
        "POST",
        &format!("/api/sessions/{sid}/datasets/reload"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["code"], "memory_pressure", "{body}");
    assert!(headers.contains_key(header::RETRY_AFTER));
}

/// `GET /api/sessions/{id}` adopts a stored session through the persistent
/// manager; a full registry answers 503 `capacity` with `Retry-After`, as
/// session creation does.
#[tokio::test]
async fn get_session_answers_capacity_when_adoption_finds_the_registry_full() {
    let mut cfg = ServerConfig::default();
    cfg.session.max_sessions = 1;
    let (r, state) = app(cfg).await;
    new_session(&r).await;
    let stored = uuid::Uuid::new_v4().to_string();
    // A row accessed now, so it is within the TTL.
    pm(&state)
        .db()
        .upsert_session(&p2a_mcp::db::DbSession::new(stored.clone(), None))
        .await
        .unwrap();

    let (st, headers, body) =
        call_with_headers(&r, "GET", &format!("/api/sessions/{stored}"), None, None).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["code"], "capacity", "{body}");
    assert_eq!(headers[header::RETRY_AFTER], "60");
}

/// A store failure while looking a session up is a 500 `internal` with a
/// generic text, not a 404 that tells the client to abandon the session.
#[tokio::test]
async fn get_session_answers_internal_on_a_store_failure() {
    let (r, state) = app(ServerConfig::default()).await;
    let broken = uuid::Uuid::new_v4().to_string();
    // A session row the store cannot read back as a session.
    pm(&state)
        .db()
        .db()
        .query("REMOVE TABLE sessions")
        .query("CREATE type::thing('sessions', $id) SET last_accessed = 'not-a-date'")
        .bind(("id", broken.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();

    let (st, body) = call(&r, "GET", &format!("/api/sessions/{broken}"), None, None).await;
    assert_eq!(st, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(body["code"], "internal", "{body}");
    assert_eq!(body["error"], "internal error", "{body}");
}

// =============================================================================
// Task 2: the owning session on every conversation route (MCP-11; MCP-7 settings)
// =============================================================================

/// Every conversation, message and tool-call route answers 404 to a session
/// that does not own the object, identically for missing and foreign ids.
#[tokio::test]
async fn conversation_routes_answer_404_to_other_sessions() {
    let (r, state) = app(ServerConfig::default()).await;
    let s1 = new_session(&r).await;
    let s2 = new_session(&r).await;
    let c1 = new_conversation(&r, &s1, "mine").await;
    let (st, body) = add_msg(&r, &s1, &c1, "S1 PRIVATE").await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let m1 = body["data"]["id"].as_str().unwrap().to_string();

    let requests: Vec<(&str, String, Option<Value>)> = vec![
        ("GET", format!("/api/conversations/{c1}"), None),
        (
            "PUT",
            format!("/api/conversations/{c1}"),
            Some(json!({"title": "x"})),
        ),
        (
            "POST",
            format!("/api/conversations/{c1}/archive"),
            Some(json!({"is_archived": true})),
        ),
        ("GET", format!("/api/conversations/{c1}/messages"), None),
        (
            "POST",
            format!("/api/conversations/{c1}/messages"),
            Some(json!({"role": "user", "content": "intruder"})),
        ),
        ("DELETE", format!("/api/conversations/{c1}/messages"), None),
        ("GET", format!("/api/conversations/{c1}/tool-calls"), None),
        ("GET", format!("/api/messages/{m1}/tool-calls"), None),
        ("DELETE", format!("/api/conversations/{c1}"), None),
    ];
    for (method, uri, body) in requests {
        let (st, body) = call(&r, method, &uri, Some(&s2), body).await;
        assert_eq!(st, StatusCode::NOT_FOUND, "{method} {uri}: {body}");
        assert!(!body.to_string().contains("S1 PRIVATE"), "{method} {uri}");
    }

    let msgs = pm(&state).db().get_messages(&c1).await.unwrap();
    assert_eq!((msgs.len(), msgs[0].content.as_str()), (1, "S1 PRIVATE"));
    assert_eq!(
        pm(&state).db().get_conversation(&c1).await.unwrap().title,
        "mine"
    );
    let foreign = call(
        &r,
        "GET",
        &format!("/api/conversations/{c1}"),
        Some(&s2),
        None,
    )
    .await
    .1;
    let missing = call(
        &r,
        "GET",
        &format!("/api/conversations/{}", uuid::Uuid::new_v4()),
        Some(&s2),
        None,
    )
    .await
    .1;
    assert_eq!(foreign["error"], missing["error"]); // indistinguishable
    assert_eq!(
        call(
            &r,
            "GET",
            &format!("/api/conversations/{c1}"),
            Some(&s1),
            None
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(
            &r,
            "GET",
            &format!("/api/conversations/{c1}?session_id={s1}"),
            None,
            None
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(&r, "GET", &format!("/api/conversations/{c1}"), None, None)
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        call(
            &r,
            "GET",
            "/api/conversations?session_id=not-live",
            None,
            None
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    // Listing takes the caller like every other conversation route: the
    // header alone is enough, and a query naming another session is a 400.
    let (st, body) = call(&r, "GET", "/api/conversations", Some(&s1), None).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 1, "{body}");
    let (st, body) = call(
        &r,
        "GET",
        &format!("/api/conversations?session_id={s2}"),
        Some(&s1),
        None,
    )
    .await;
    assert_eq!(
        (st, body["code"].as_str()),
        (StatusCode::BAD_REQUEST, Some("invalid_arguments"))
    );
    let (st, body) = call(&r, "GET", "/api/conversations", None, None).await;
    assert_eq!(
        (st, body["code"].as_str()),
        (StatusCode::BAD_REQUEST, Some("invalid_arguments"))
    );
}

/// Settings rows are created only for live sessions, and the path id is the
/// caller.
#[tokio::test]
async fn settings_require_a_live_session() {
    let (r, state) = app(ServerConfig::default()).await;
    assert_eq!(
        call(&r, "GET", "/api/sessions/random-xyz-1/settings", None, None)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let row: Option<p2a_mcp::db::Settings> = pm(&state)
        .db()
        .db()
        .select(("settings", "random-xyz-1"))
        .await
        .unwrap();
    assert!(row.is_none());
    assert_eq!(
        call(
            &r,
            "PUT",
            "/api/sessions/random-xyz-1/settings",
            None,
            Some(json!({"model": "m"}))
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    let sid = new_session(&r).await;
    assert_eq!(
        call(
            &r,
            "GET",
            &format!("/api/sessions/{sid}/settings"),
            Some(&sid),
            None
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(
            &r,
            "PUT",
            &format!("/api/sessions/{sid}/settings"),
            Some(&sid),
            Some(json!({"model": "m"}))
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(
            &r,
            "GET",
            &format!("/api/sessions/{sid}/settings"),
            Some("other"),
            None
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Task 3: bounded persistence and deletion when a session ends (MCP-7)
// =============================================================================

/// A message to a conversation that does not exist is refused on the route
/// and on the store call the chat stream uses; nothing is stored.
#[tokio::test]
async fn messages_to_unknown_conversations_are_refused() {
    let (r, state) = app(ServerConfig::default()).await;
    let sid = new_session(&r).await;
    let before = pm(&state).db().stats().await.unwrap().message_count;
    let (st, body) = call(
        &r,
        "POST",
        "/api/conversations/does-not-exist-123/messages",
        Some(&sid),
        Some(json!({"role": "user", "content": "x"})),
    )
    .await;
    assert_eq!(st, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(pm(&state).db().stats().await.unwrap().message_count, before);
    // The chat-stream path writes through the store directly.
    assert!(matches!(
        pm(&state)
            .db()
            .add_message("does-not-exist-123", "assistant", "")
            .await,
        Err(p2a_mcp::db::DbError::NotFound(_))
    ));
    assert_eq!(pm(&state).db().stats().await.unwrap().message_count, before);
}

/// Message size, messages per conversation and conversations per session
/// are bounded by the resolved limits.
#[tokio::test]
async fn message_size_and_counts_are_bounded() {
    let mut cfg = ServerConfig::default();
    cfg.limits.overrides = vec![
        "message_bytes=1024".into(),
        "messages_per_conversation=2".into(),
        "conversations_per_session=1".into(),
    ];
    let (r, _state) = app(cfg).await;
    let sid = new_session(&r).await;
    let c = new_conversation(&r, &sid, "t").await;

    let (st, body) = add_msg(&r, &sid, &c, &"a".repeat(1025)).await;
    assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["code"], "payload_too_large");
    assert!(body["error"].as_str().unwrap().contains("1024"), "{body}");
    assert_eq!(
        add_msg(&r, &sid, &c, &"a".repeat(1024)).await.0,
        StatusCode::CREATED
    );
    assert_eq!(add_msg(&r, &sid, &c, "second").await.0, StatusCode::CREATED);
    let (st, body) = add_msg(&r, &sid, &c, "third").await;
    assert_eq!(
        (st, body["code"].as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some("quota_exceeded")),
        "{body}"
    );
    let (st, body) = call(
        &r,
        "POST",
        "/api/conversations",
        Some(&sid),
        Some(json!({"session_id": sid, "title": "another"})),
    )
    .await;
    assert_eq!(
        (st, body["code"].as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some("quota_exceeded")),
        "{body}"
    );
}

/// A session removed for expiry loses its rows (through the removal
/// notification), not only its live state.
#[tokio::test]
async fn expired_sessions_lose_their_rows() {
    let mut cfg = ServerConfig::default();
    cfg.session.ttl_minutes = 1;
    let (r, state) = app(cfg).await;
    let sid = new_session(&r).await;
    let c = new_conversation(&r, &sid, "t").await;
    assert_eq!(add_msg(&r, &sid, &c, "hello").await.0, StatusCode::CREATED);
    assert_eq!(
        call(
            &r,
            "GET",
            &format!("/api/sessions/{sid}/settings"),
            Some(&sid),
            None
        )
        .await
        .0,
        StatusCode::OK
    );

    let live = state.session_manager.get_session(&sid).await.unwrap();
    *live.last_accessed.write().await = chrono::Utc::now() - chrono::Duration::minutes(5);
    assert_eq!(state.session_manager.cleanup_expired().await, 1);
    let db = pm(&state).db().clone();
    assert!(
        wait_until(|| {
            let (db, sid, c) = (db.clone(), sid.clone(), c.clone());
            async move {
                db.get_session(&sid).await.unwrap().is_none()
                    && db.get_conversation(&c).await.is_err()
            }
        })
        .await,
        "rows of the expired session were not deleted"
    );
    assert!(db.get_messages(&c).await.unwrap().is_empty());
}

/// DELETE /api/sessions/{id} deletes the session's rows before it answers.
#[tokio::test]
async fn deleting_a_session_deletes_its_rows() {
    let (r, state) = app(ServerConfig::default()).await;
    let sid = new_session(&r).await;
    let c = new_conversation(&r, &sid, "t").await;
    assert_eq!(add_msg(&r, &sid, &c, "hello").await.0, StatusCode::CREATED);

    assert_eq!(
        call(
            &r,
            "DELETE",
            &format!("/api/sessions/{sid}"),
            Some(&sid),
            None
        )
        .await
        .0,
        StatusCode::NO_CONTENT
    );
    // Synchronous: no wait.
    assert!(pm(&state).db().get_session(&sid).await.unwrap().is_none());
    assert!(pm(&state).db().get_conversation(&c).await.is_err());
    assert_eq!(
        call(
            &r,
            "POST",
            "/api/conversations",
            Some(&sid),
            Some(json!({"session_id": sid, "title": "x"}))
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
}

/// The bytes one session's conversation history holds are capped by
/// session_db_bytes; the budget is per session, and deleting a conversation
/// frees its bytes.
#[tokio::test]
async fn session_db_bytes_caps_a_sessions_history() {
    let mut cfg = ServerConfig::default();
    cfg.limits.overrides = vec!["session_db_bytes=2048".into()];
    let (r, _state) = app(cfg).await;

    let s1 = new_session(&r).await;
    let c1 = new_conversation(&r, &s1, "t").await; // 1 byte
    assert_eq!(
        add_msg(&r, &s1, &c1, &"a".repeat(1500)).await.0,
        StatusCode::CREATED
    );
    // 1 + 1500 + 600 > 2048
    let (st, body) = add_msg(&r, &s1, &c1, &"b".repeat(600)).await;
    assert_eq!(
        (st, body["code"].as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some("quota_exceeded")),
        "{body}"
    );
    assert!(body["error"].as_str().unwrap().contains("2048"), "{body}");
    // Exactly 2048.
    assert_eq!(
        add_msg(&r, &s1, &c1, &"c".repeat(547)).await.0,
        StatusCode::CREATED
    );

    // The budget is per session.
    let s2 = new_session(&r).await;
    let c2 = new_conversation(&r, &s2, "t").await;
    assert_eq!(
        add_msg(&r, &s2, &c2, &"d".repeat(1500)).await.0,
        StatusCode::CREATED
    );

    assert_eq!(
        call(
            &r,
            "DELETE",
            &format!("/api/conversations/{c1}"),
            Some(&s1),
            None
        )
        .await
        .0,
        StatusCode::NO_CONTENT
    );
    // Deleting the conversation freed its bytes.
    let c3 = new_conversation(&r, &s1, "t").await;
    assert_eq!(
        add_msg(&r, &s1, &c3, &"e".repeat(1500)).await.0,
        StatusCode::CREATED
    );
}

// =============================================================================
// Task 4: user_id validation (MCP-17)
// =============================================================================

/// `user_id` is at most 128 bytes of `[A-Za-z0-9._@-]`.
#[tokio::test]
async fn user_id_is_validated() {
    let (r, _state) = app(ServerConfig::default()).await;
    for bad in [
        "a".repeat(129),
        "has space".to_string(),
        "semi;colon".to_string(),
        "ü".to_string(),
    ] {
        let (st, body) = call(
            &r,
            "POST",
            "/api/sessions",
            None,
            Some(json!({"user_id": bad})),
        )
        .await;
        assert_eq!(
            (st, body["code"].as_str()),
            (StatusCode::BAD_REQUEST, Some("invalid_arguments")),
            "{bad}: {body}"
        );
    }
    assert_eq!(
        call(
            &r,
            "POST",
            "/api/sessions",
            None,
            Some(json!({"user_id": "a".repeat(128)}))
        )
        .await
        .0,
        StatusCode::CREATED
    );
    let (_, body) = call(
        &r,
        "POST",
        "/api/sessions",
        None,
        Some(json!({"user_id": "alice.o-k_1@example.org"})),
    )
    .await;
    let sid = body["data"]["session_id"].as_str().unwrap();
    assert_eq!(
        call(&r, "GET", &format!("/api/sessions/{sid}"), None, None)
            .await
            .1["data"]["user_id"],
        "alice.o-k_1@example.org"
    );
}

/// In open public mode the `user_id` means nothing and is not stored.
#[tokio::test]
async fn open_public_mode_ignores_user_id() {
    let mut cfg = ServerConfig::default();
    cfg.http.allow_unauthenticated = true;
    cfg.http.session_dirs = p2a_mcp::config::SessionDirsSetting::Off;
    let (r, state) = app(cfg).await;
    let (st, body) = call(
        &r,
        "POST",
        "/api/sessions",
        None,
        Some(json!({"user_id": "x".repeat(4096)})),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let sid = body["data"]["session_id"].as_str().unwrap();
    assert_eq!(
        state
            .session_manager
            .get_session(sid)
            .await
            .unwrap()
            .user_id,
        None
    );
    assert_eq!(
        pm(&state)
            .db()
            .get_session(sid)
            .await
            .unwrap()
            .unwrap()
            .user_id,
        None
    );
}

// =============================================================================
// Task 5: the session database outside the data root (INF-1)
// =============================================================================

/// An authenticated server refuses to start with its database inside the
/// path-jail root, where file tools can write.
#[tokio::test]
async fn database_inside_the_data_root_is_refused() {
    let root = allowed_data_root().unwrap();
    let tmp = tempfile::Builder::new()
        .prefix(".p2a-wp07-")
        .tempdir_in(&root)
        .unwrap();
    let mut cfg = ServerConfig::default();
    cfg.http.access_token = Some("test-token".into()); // Authenticated mode
    cfg.http.db_path = Some(tmp.path().join("db").to_string_lossy().into_owned());
    let err = build_app(&cfg)
        .await
        .err()
        .expect("startup must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("P2A_DB_PATH") && msg.contains("data root"),
        "{msg}"
    );
    assert!(!tmp.path().join("db").exists());
}
