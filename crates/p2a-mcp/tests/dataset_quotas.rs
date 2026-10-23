//! The governor's dataset quotas against what datasets really cost (PR-03):
//! wide frames, many small frames, categorical labels, and parquet files
//! whose decoded size the footer gives away before anything is decoded.
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
use polars_parquet_format::thrift::protocol::{TCompactInputProtocol, TCompactOutputProtocol};
use polars_parquet_format::{CompressionCodec, FileMetaData, PageHeader, PageType};
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

/// A parquet file of `rows` copies of one float: a few kilobytes on disk,
/// 8 bytes a row once decoded.
fn constant_parquet(path: &Path, rows: usize) {
    let mut df = DataFrame::new(vec![Series::new("x".into(), vec![1.5f64; rows]).into()]).unwrap();
    write_parquet(&mut df, path);
}

/// A folder inside the data root for one test, removed when it drops.
fn data_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(".p2a-pr03-")
        .tempdir_in(allowed_data_root().unwrap())
        .unwrap()
}

/// `body`'s refusal text.
fn error_text(body: &Value) -> &str {
    body["error"].as_str().unwrap_or_default()
}

#[tokio::test]
async fn a_parquet_upload_past_the_quota_is_refused_before_it_is_decoded() {
    // 4,000,000 rows of one float: a small file, 32 MB once decoded. Pre-fix
    // the upload was decoded in full and only then undone by the quota
    // ("The change was undone"), so a larger file of the same kind could
    // exhaust the server's memory before the quota saw it.
    let dir = data_dir();
    let path = dir.path().join("bomb.parquet");
    constant_parquet(&path, 4_000_000);
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() < 512 * 1024, "{} bytes", bytes.len());

    let (app, _) = app_with_quota(8 * MIB).await;
    let sid = new_session(&app).await;
    let content = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let (status, body) = call(
        &app,
        &sid,
        "upload_dataset",
        json!({"content": content, "filename": "bomb.parquet"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
    assert!(error_text(&body).contains("Nothing was loaded"), "{body}");
    let (_, listed) = call(&app, &sid, "list_datasets", json!({})).await;
    assert!(!listed.to_string().contains("bomb"), "{listed}");
}

#[tokio::test]
async fn a_parquet_load_past_the_quota_is_refused_before_it_is_decoded() {
    let dir = data_dir();
    let path = dir.path().join("bomb.parquet");
    constant_parquet(&path, 4_000_000);

    let (app, _) = app_with_quota(8 * MIB).await;
    let sid = new_session(&app).await;
    let (status, body) = call(
        &app,
        &sid,
        "load_dataset",
        json!({"path": path.to_string_lossy(), "name": "bomb"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
    assert!(error_text(&body).contains("Nothing was loaded"), "{body}");
}

#[tokio::test]
async fn parquet_files_inside_the_quota_load_and_replace_by_name() {
    // 300,000 noisy rows: 2.4 MB decoded. Two fit a 6 MiB quota, a third
    // does not, and loading one again under a name already held replaces it.
    // A regression guard, not evidence of the check before decoding: without
    // that check the files load too and the quota undoes the third after it.
    // It keeps the check from refusing a file that replaces one by name.
    let dir = data_dir();
    let path = dir.path().join("fits.parquet");
    let values: Vec<f64> = (0..300_000).map(|i| (i % 97) as f64 * 0.37 + 0.1).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    write_parquet(&mut df, &path);

    let (app, _) = app_with_quota(6 * MIB).await;
    let sid = new_session(&app).await;
    let load = |name: &'static str| {
        call(
            &app,
            &sid,
            "load_dataset",
            json!({"path": path.to_string_lossy(), "name": name}),
        )
    };
    for name in ["a", "b", "a", "b"] {
        let (status, body) = load(name).await;
        assert_eq!(status, StatusCode::OK, "{name}: {body}");
        assert_eq!(body["data"]["success"], json!(true), "{name}: {body}");
    }
    let (status, body) = load("c").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
}

/// A parquet file of one row whose column `x` is a list of `values` copies
/// of one categorical label. The values are dictionary-encoded, so the
/// file is small and so are its pages uncompressed; decoded they take a
/// 4-byte code each.
fn one_row_list_parquet(path: &Path, values: usize) {
    let dtype = DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
    let inner = Series::new("".into(), ["same label"])
        .cast(&dtype)
        .unwrap()
        .new_from_index(0, values);
    let x = Series::new("x".into(), [inner]);
    assert!(matches!(x.dtype(), DataType::List(inner) if inner.is_categorical()));
    let mut df = DataFrame::new(vec![x.into()]).unwrap();
    write_parquet(&mut df, path);
}

#[tokio::test]
async fn a_one_row_list_parquet_past_the_quota_is_refused_before_it_is_decoded() {
    // One row holding a list of 4,000,000 values: a file of about a
    // kilobyte, its pages hardly more uncompressed, 16 MB once decoded. Pre-fix
    // the check before decoding charged a list column 8 bytes a row, so this
    // file was decoded in full and only then undone by the quota ("The
    // change was undone"); a longer list could exhaust the server's memory
    // before the quota saw it.
    let dir = data_dir();
    let path = dir.path().join("list.parquet");
    one_row_list_parquet(&path, 4_000_000);
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() < 512 * 1024, "{} bytes", bytes.len());

    let (app, _) = app_with_quota(8 * MIB).await;
    let sid = new_session(&app).await;
    let content = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let (status, body) = call(
        &app,
        &sid,
        "upload_dataset",
        json!({"content": content, "filename": "list.parquet"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
    assert!(error_text(&body).contains("Nothing was loaded"), "{body}");

    let (status, body) = call(
        &app,
        &sid,
        "load_dataset",
        json!({"path": path.to_string_lossy(), "name": "list"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"), "{body}");
    assert!(error_text(&body).contains("Nothing was loaded"), "{body}");
    let (_, listed) = call(&app, &sid, "list_datasets", json!({})).await;
    assert!(
        listed.to_string().contains("No datasets currently loaded"),
        "{listed}"
    );
}

/// Rewrite the footer of the parquet file at `path` with `edit`. The pages
/// stay where they are, so the file decodes as before wherever the decoder
/// does not follow the edited fields.
fn rewrite_footer(path: &Path, edit: impl FnOnce(&mut FileMetaData)) {
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

/// Upload the parquet file at `path` and load it from its path, each time
/// into a fresh session of a server with an 8 MiB session quota, and check
/// that both are refused before anything is decoded.
async fn refused_before_decoding_by_upload_and_load(path: &Path) {
    let content = base64::engine::general_purpose::STANDARD.encode(std::fs::read(path).unwrap());
    let (app, _) = app_with_quota(8 * MIB).await;
    let calls = [
        (
            "upload_dataset",
            json!({"content": content, "filename": "lies.parquet"}),
        ),
        (
            "load_dataset",
            json!({"path": path.to_string_lossy(), "name": "lies"}),
        ),
    ];
    for (tool, args) in calls {
        let sid = new_session(&app).await;
        let (status, body) = call(&app, &sid, tool, args).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{tool}: {body}");
        assert_eq!(body["code"], json!("quota_exceeded"), "{tool}: {body}");
        assert!(
            error_text(&body).contains("Nothing was loaded"),
            "{tool}: {body}"
        );
    }
}

#[tokio::test]
async fn a_parquet_footer_that_understates_its_rows_is_refused_before_decoding() {
    // The file-level row count says one row; polars decodes the rows each
    // row group declares (4,000,000, 32 MB). Pre-fix the check before
    // decoding read the file-level count, so the file was decoded in full
    // and only then undone ("The change was undone").
    let dir = data_dir();
    let path = dir.path().join("rows.parquet");
    constant_parquet(&path, 4_000_000);
    rewrite_footer(&path, |footer| footer.num_rows = 1);
    refused_before_decoding_by_upload_and_load(&path).await;
}

#[tokio::test]
async fn a_parquet_footer_that_understates_a_chunks_entries_is_refused_before_decoding() {
    // One row holding a list of 4,000,000 values (16 MB decoded), whose
    // column chunk says it holds one entry. polars reads a chunk's pages
    // until their headers have given it that many entries, a page at a
    // time, so it still decodes the first page and all 4,000,000 values in
    // it. Pre-fix the check before decoding took the chunk's count, so the
    // file was decoded in full and only then undone.
    let dir = data_dir();
    let path = dir.path().join("entries.parquet");
    one_row_list_parquet(&path, 4_000_000);
    rewrite_footer(&path, |footer| {
        for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
            chunk.meta_data.as_mut().unwrap().num_values = 1;
        }
    });
    refused_before_decoding_by_upload_and_load(&path).await;
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
    // of codes, so a quota held any number of them. Held, each is charged
    // about 4.2 MB, so two pass a 7.5 MiB quota. Loading one takes about
    // 7.3 MB by the check before decoding (the labels, and the decompressed
    // dictionary page and its decoded copy while it loads), which the quota
    // admits.
    let dir = data_dir();
    let (first, second) = (dir.path().join("a.parquet"), dir.path().join("b.parquet"));
    categorical_parquet(&first, "a", 2500);
    categorical_parquet(&second, "b", 2500);

    let (app, _) = app_with_quota(15 * MIB / 2).await;
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

/// Rebuild the parquet file at `path` with `edit` applied to every page
/// header, in file order. The pages are copied as they are, the column
/// chunks laid out again one after another, and the footer's offsets and
/// sizes rewritten to match (its page indexes dropped), so the file reads as
/// before wherever the decoder does not follow the edited fields.
fn rewrite_pages(path: &Path, mut edit: impl FnMut(&mut PageHeader)) {
    let bytes = std::fs::read(path).unwrap();
    let n = bytes.len();
    let len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let mut input = TCompactInputProtocol::new(&bytes[n - 8 - len..n - 8], usize::MAX);
    let mut footer = FileMetaData::read_from_in_protocol(&mut input).unwrap();
    let mut out = b"PAR1".to_vec();
    for group in &mut footer.row_groups {
        let mut group_bytes = 0;
        for chunk in &mut group.columns {
            let meta = chunk.meta_data.as_mut().unwrap();
            assert_ne!(meta.codec, CompressionCodec::UNCOMPRESSED);
            let from = meta.dictionary_page_offset.unwrap_or(meta.data_page_offset) as usize;
            let to = from + meta.total_compressed_size as usize;
            let start = out.len();
            let mut first_data_page = None;
            let mut dictionary_page = None;
            let mut at = from;
            while at < to {
                let mut rest = &bytes[at..to];
                let mut header = PageHeader::read_from_in_protocol(
                    &mut TCompactInputProtocol::new(&mut rest, usize::MAX),
                )
                .unwrap();
                let header_len = (to - at) - rest.len();
                let body = &rest[..header.compressed_page_size as usize];
                if header.type_ == PageType::DICTIONARY_PAGE {
                    dictionary_page.get_or_insert(out.len() as i64);
                } else {
                    first_data_page.get_or_insert(out.len() as i64);
                }
                edit(&mut header);
                header
                    .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut out))
                    .unwrap();
                out.extend_from_slice(body);
                at += header_len + body.len();
            }
            // A writer may leave the dictionary page's offset out and start
            // the chunk's data at it; the rebuilt chunk states both.
            meta.dictionary_page_offset = dictionary_page;
            meta.data_page_offset = first_data_page.unwrap();
            meta.total_compressed_size = (out.len() - start) as i64;
            meta.index_page_offset = None;
            meta.bloom_filter_offset = None;
            meta.bloom_filter_length = None;
            chunk.file_offset = meta.data_page_offset;
            chunk.offset_index_offset = None;
            chunk.offset_index_length = None;
            chunk.column_index_offset = None;
            chunk.column_index_length = None;
            group_bytes += meta.total_compressed_size;
        }
        group.file_offset = None;
        group.total_compressed_size = Some(group_bytes);
    }
    let mut written = Vec::new();
    footer
        .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut written))
        .unwrap();
    out.extend_from_slice(&written);
    out.extend_from_slice(&u32::try_from(written.len()).unwrap().to_le_bytes());
    out.extend_from_slice(b"PAR1");
    std::fs::write(path, out).unwrap();
}

/// Bytes a test page header declares it decompresses to: far past the
/// 8 MiB quota, and small enough that a server that decodes the file
/// anyway survives the test.
const DECLARED_PAGE_BYTES: i32 = 64 << 20;

#[tokio::test]
async fn a_float_data_page_that_declares_a_large_size_is_refused_before_decoding() {
    // 100,000 noisy floats: 800 kB decoded, in compressed data pages whose
    // headers are rewritten to declare 64 MiB each. polars decompresses a
    // page into a zero-filled buffer of the declared size before it checks
    // the page. Pre-fix the check before decoding counted a fixed-width
    // column's values and not its pages, so the file was admitted.
    let dir = data_dir();
    let path = dir.path().join("pages.parquet");
    let values: Vec<f64> = (0..100_000).map(|i| i as f64 * 0.731 + 0.1).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    write_parquet(&mut df, &path);
    let mut data_pages = 0;
    rewrite_pages(&path, |header| {
        if header.type_ == PageType::DATA_PAGE || header.type_ == PageType::DATA_PAGE_V2 {
            header.uncompressed_page_size = DECLARED_PAGE_BYTES;
            data_pages += 1;
        }
    });
    assert!(data_pages > 0);
    refused_before_decoding_by_upload_and_load(&path).await;
}

#[tokio::test]
async fn an_integer_dictionary_page_that_declares_a_large_size_is_refused_before_decoding() {
    // 100,000 integers of 50 values, which polars writes as a dictionary
    // page and pages of codes; the dictionary page's header is rewritten to
    // declare 64 MiB. polars decompresses the dictionary page into a buffer
    // of that size before it reads a row.
    let dir = data_dir();
    let path = dir.path().join("dict.parquet");
    let values: Vec<i64> = (0..100_000).map(|i| (i * 7) % 50 + 3).collect();
    let mut df = DataFrame::new(vec![Series::new("k".into(), values).into()]).unwrap();
    write_parquet(&mut df, &path);
    let mut dictionaries = 0;
    rewrite_pages(&path, |header| {
        if header.type_ == PageType::DICTIONARY_PAGE {
            header.uncompressed_page_size = DECLARED_PAGE_BYTES;
            dictionaries += 1;
        }
    });
    assert!(dictionaries > 0);
    refused_before_decoding_by_upload_and_load(&path).await;
}

#[tokio::test]
async fn a_dictionary_under_a_chunk_that_claims_no_entries_is_refused_before_decoding() {
    // 10,000 labels of 1,000 bytes (10 MB) in a dictionary page, under a
    // column chunk whose footer entry says it holds no entries and one byte
    // of pages. polars reads and decodes a chunk's dictionary page before
    // it looks at the chunk's count. Pre-fix the check before decoding read
    // no page of such a chunk and took its footer size, so the file was
    // admitted and its labels decoded.
    let dir = data_dir();
    let path = dir.path().join("labels.parquet");
    categorical_parquet(&path, "c", 10_000);
    rewrite_footer(&path, |footer| {
        for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
            let meta = chunk.meta_data.as_mut().unwrap();
            meta.num_values = 0;
            meta.total_uncompressed_size = 1;
        }
    });
    refused_before_decoding_by_upload_and_load(&path).await;
}

#[tokio::test]
async fn a_row_group_that_overstates_its_rows_cannot_hide_another_groups_values() {
    // Two row groups: one row holding an empty list, its count rewritten to
    // 100,000 rows, and one row holding a list of 60,000 structs of 20
    // floats (9.6 MB decoded). polars decodes each row group on its own, the
    // second in full. Pre-fix the check before decoding summed rows and
    // entries over the file, so the first group's claimed rows cancelled the
    // second group's values and the file was admitted.
    use polars::prelude::{IntoSeries, ListChunked, StructChunked};
    let dir = data_dir();
    let path = dir.path().join("groups.parquet");
    let fields: Vec<Series> = (0..20)
        .map(|f| {
            Series::new(
                format!("f{f}").into(),
                (0..60_000)
                    .map(|i| ((i + f) % 9) as f64 * 0.5 + 0.25)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let wide = StructChunked::from_series("".into(), 60_000, fields.iter())
        .unwrap()
        .into_series();
    let lists: ListChunked = [Some(wide.clear()), Some(wide)].into_iter().collect();
    let mut df = DataFrame::new(vec![lists.into_series().with_name("l".into()).into()]).unwrap();
    ParquetWriter::new(std::fs::File::create(&path).unwrap())
        .with_row_group_size(Some(1))
        .finish(&mut df)
        .unwrap();
    rewrite_footer(&path, |footer| {
        assert_eq!(footer.row_groups.len(), 2);
        footer.row_groups[0].num_rows = 100_000;
        footer.num_rows = 100_001;
    });
    refused_before_decoding_by_upload_and_load(&path).await;
}

#[tokio::test]
async fn a_parquet_file_with_shared_column_chunks_is_not_loaded() {
    // Row groups whose column chunks point at the same bytes: no writer
    // produces them, and checking them would read those page headers once
    // per chunk. They are refused as unreadable, not loaded, and not
    // answered as a server failure.
    let dir = data_dir();
    let path = dir.path().join("shared.parquet");
    let values: Vec<f64> = (0..4_000).map(|i| (i % 31) as f64 * 0.5 + 0.25).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    ParquetWriter::new(std::fs::File::create(&path).unwrap())
        .with_row_group_size(Some(1_000))
        .finish(&mut df)
        .unwrap();
    rewrite_footer(&path, |footer| {
        let shared = footer.row_groups[0].columns[0].meta_data.clone();
        for group in footer.row_groups.iter_mut().skip(1) {
            group.columns[0].meta_data = shared.clone();
        }
    });
    let (app, _) = app_with_quota(8 * MIB).await;
    let sid = new_session(&app).await;
    let (status, body) = call(
        &app,
        &sid,
        "load_dataset",
        json!({"path": path.to_string_lossy(), "name": "shared"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"]["success"], json!(false), "{body}");
    assert!(body.to_string().contains("share bytes"), "{body}");
    let (_, listed) = call(&app, &sid, "list_datasets", json!({})).await;
    assert!(!listed.to_string().contains("shared"), "{listed}");
}
