//! The governor's dataset quotas against what datasets really cost (PR-03):
//! wide frames, many small frames and categorical labels.
//!
//! Servers are built in-process and driven with `tower::ServiceExt::oneshot`.
#![cfg(feature = "http")]

use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use base64::Engine;
use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::exec::ExecConfig;
use p2a_mcp::path_jail::allowed_data_root;
use p2a_mcp::transport::http::{AppState, build_app_with};
use polars::prelude::{
    CategoricalPhysical, Categories, DataFrame, DataType, NamedFrom, ParquetWriter, Series,
};
use serde_json::{Value, json};
use tower::ServiceExt;

/// The default local HTTP configuration.
fn local() -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = "127.0.0.1:8080".parse().unwrap();
    c
}

/// A local app with a per-session dataset quota of `per_session` bytes.
async fn app_with_quota(per_session: u64) -> (Router, AppState) {
    let exec = ExecConfig {
        session_data_bytes: Some(per_session),
        ..Default::default()
    };
    build_app_with(&local(), exec).await.unwrap()
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
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn new_session(app: &Router) -> String {
    let (s, b) = send(app, post_json("/api/sessions", json!({}))).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    b["data"]["session_id"].as_str().unwrap().to_string()
}

async fn call(app: &Router, sid: &str, tool: &str, args: Value) -> (StatusCode, Value) {
    send(
        app,
        post_json(
            &format!("/api/tools/{tool}"),
            json!({"session_id": sid, "arguments": args}),
        ),
    )
    .await
}

/// The number of a session's calls, out of `max_calls`, that succeed before
/// the first `quota_exceeded`; panics on any other refusal.
async fn calls_until_quota(
    max_calls: usize,
    mut call_number: impl AsyncFnMut(usize) -> (StatusCode, Value),
) -> Option<usize> {
    for i in 0..max_calls {
        let (status, body) = call_number(i).await;
        match status {
            StatusCode::OK => assert_eq!(body["data"]["success"], json!(true), "{body}"),
            StatusCode::UNPROCESSABLE_ENTITY if body["code"] == json!("quota_exceeded") => {
                return Some(i);
            }
            _ => panic!("call {i}: {status} {body}"),
        }
    }
    None
}

const MIB: u64 = 1 << 20;

// RSS measured per column on a release build of the hosted server (PR-03):
// 760 B for 1 x 500 generated frames, 1,094 B for 1 x 12,001 pivots with
// 99-byte names, 1,250 B for 1 x 12,000 CSV uploads.
const RSS_PER_GENERATED_COLUMN: u64 = 760;
const RSS_PER_PIVOT_COLUMN: u64 = 1094;
const RSS_PER_CSV_COLUMN: u64 = 1250;

/// The most calls of `columns` columns at `rss_per_column` each that fit
/// `quota`; a quota that bounds memory refuses the call after them.
fn calls_that_fit(quota: u64, columns: u64, rss_per_column: u64) -> usize {
    usize::try_from(quota / (columns * rss_per_column)).unwrap()
}

#[tokio::test]
async fn wide_generated_frames_reach_the_session_quota() {
    // Pre-fix each 1 x 500 frame was charged its 4 kB of values, so about 500
    // of them fit a 2 MiB quota while their columns took about 190 MB.
    let (app, _) = app_with_quota(2 * MIB).await;
    let sid = new_session(&app).await;
    let columns: Vec<Value> = (0..500)
        .map(|i| json!({"name": format!("c{i}"), "distribution": {"type": "uniform", "min": 0.0, "max": 1.0}}))
        .collect();
    let fit = calls_that_fit(2 * MIB, 500, RSS_PER_GENERATED_COLUMN);
    let accepted = calls_until_quota(fit + 1, async |i| {
        call(
            &app,
            &sid,
            "generate_random_data",
            json!({"n_rows": 1, "columns": columns, "seed": 3, "name": format!("w{i}")}),
        )
        .await
    })
    .await;
    assert!(
        accepted.is_some_and(|n| n <= fit),
        "accepted {accepted:?} of {} frames",
        fit + 1
    );
}

#[tokio::test]
async fn pivot_fan_out_reaches_the_session_quota() {
    // 2,000 distinct 99-byte keys pivot to 1 x 2,001 frames: about 2.4 MB of
    // RSS each, charged about 16 kB each pre-fix.
    let (app, _) = app_with_quota(2 * MIB).await;
    let sid = new_session(&app).await;
    let mut csv = String::from("idx,key,val\n");
    for i in 0..2000 {
        csv.push_str(&format!("0,{}{i:06},{}\n", "k".repeat(93), i * 3 + 1));
    }
    let (s, b) = call(
        &app,
        &sid,
        "create_dataset",
        json!({"name": "long", "csv_content": csv}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let fit = calls_that_fit(2 * MIB, 2001, RSS_PER_PIVOT_COLUMN);
    let accepted = calls_until_quota(fit + 2, async |i| {
        call(
            &app,
            &sid,
            "munge_pivot",
            json!({"dataset": "long", "index": ["idx"], "on": "key", "values": "val",
                   "result_name": format!("w{i}")}),
        )
        .await
    })
    .await;
    assert!(
        accepted.is_some_and(|n| n <= fit),
        "accepted {accepted:?} pivots"
    );
}

#[tokio::test]
async fn wide_csv_uploads_reach_the_session_quota() {
    // A 1 x 2,000 CSV is 18 kB of text and about 2.6 MB of RSS once parsed.
    let (app, _) = app_with_quota(4 * MIB).await;
    let sid = new_session(&app).await;
    let header: Vec<String> = (0..2000).map(|j| format!("c{j:05}")).collect();
    let row: Vec<String> = (0..2000).map(|j| format!("{}.5", j % 10)).collect();
    let csv = format!("{}\n{}\n", header.join(","), row.join(","));
    let content = base64::engine::general_purpose::STANDARD.encode(&csv);
    let fit = calls_that_fit(4 * MIB, 2000, RSS_PER_CSV_COLUMN);
    let accepted = calls_until_quota(fit + 2, async |i| {
        call(
            &app,
            &sid,
            "upload_dataset",
            json!({"content": content, "filename": "wide.csv", "name": format!("u{i}")}),
        )
        .await
    })
    .await;
    assert!(
        accepted.is_some_and(|n| n <= fit),
        "accepted {accepted:?} uploads"
    );
}

/// Write `df` as a parquet file at `path`.
fn write_parquet(df: &mut DataFrame, path: &Path) {
    let file = std::fs::File::create(path).unwrap();
    ParquetWriter::new(file).finish(df).unwrap();
}

/// A folder inside the data root for one test, removed when it drops.
fn data_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(".p2a-pr03-")
        .tempdir_in(allowed_data_root().unwrap())
        .unwrap()
}

/// A parquet file whose column `k` is a dictionary of `labels` distinct
/// labels of 1,000 bytes starting with `tag`, one row each.
fn categorical_parquet(path: &Path, tag: &str, labels: usize) {
    let values: Vec<String> = (0..labels).map(|i| format!("{tag}{i:0999}")).collect();
    let dtype = DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
    let k = Series::new("k".into(), values).cast(&dtype).unwrap();
    let mut df = DataFrame::new(vec![k.into()]).unwrap();
    write_parquet(&mut df, path);
}

#[tokio::test]
async fn categorical_labels_count_against_the_session_quota() {
    // Each file holds 2,500 labels of 1,000 bytes (2.5 MB) as a dictionary
    // column, which loads as Categorical. Pre-fix each was charged its 10 kB
    // of codes, so a 6 MiB quota held any number of them.
    let dir = data_dir();
    let (first, second) = (dir.path().join("a.parquet"), dir.path().join("b.parquet"));
    categorical_parquet(&first, "a", 2500);
    categorical_parquet(&second, "b", 2500);

    let (app, _) = app_with_quota(6 * MIB).await;
    let sid = new_session(&app).await;
    let (status, body) = call(
        &app,
        &sid,
        "load_dataset",
        json!({"path": first.to_string_lossy(), "name": "a"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.to_string().contains("(cat)"), "{body}");
    let (status, body) = call(
        &app,
        &sid,
        "load_dataset",
        json!({"path": second.to_string_lossy(), "name": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
}
