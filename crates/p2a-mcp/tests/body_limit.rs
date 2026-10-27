//! The HTTP request-body limits (PR-H8).
//!
//! `P2A_MAX_HTTP_BODY_MB` (default 32 MiB) caps every request body: a body
//! declared above it answers 413 with the plain-text body `length limit
//! exceeded` before it is read, and one without a declared length is cut off
//! at it. The routes that parse a JSON body accept one up to that cap,
//! not only up to axum's own 2 MiB extractor default, except in the hosted
//! profile, where they keep 2 MiB (or the cap, when it is lower): a body is
//! parsed before the governor's memory admission sees the call.
//!
//! The router reads the cap from the environment when it is built, and
//! `serve` installs process-wide limits, so each case runs in a child process
//! of this test binary ([`run_case`]) with its own environment.
#![cfg(feature = "http")]

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use base64::Engine;
use p2a_mcp::config::{ResourceProfile, ServerConfig, TransportType};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

const MIB: usize = 1 << 20;

/// axum's default limit for the `Json`, `Bytes` and `String` extractors.
const AXUM_DEFAULT_LIMIT: usize = 2_097_152;

/// The variable the server reads its body cap from, in MiB.
const CAP_VAR: &str = "P2A_MAX_HTTP_BODY_MB";

/// Set by [`run_case`] for the child process: which case [`case`] runs.
const CASE_VAR: &str = "P2A_BODY_LIMIT_TEST_CASE";

/// The plain-text body of a refusal from the declared length.
const CAP_REFUSAL: &str = "length limit exceeded";

/// The plain-text body of a refusal while a JSON body is read.
const READ_REFUSAL: &str = "Failed to buffer the request body: length limit exceeded";

/// Run [`case`] for `name` in a child process of this binary, with the body
/// cap variable set to `cap_mb` (unset for `None`) and a data root of its own.
/// The child's output is shown only when it fails.
fn run_case(name: &str, cap_mb: Option<&str>) {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("p2a-body-limit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", "case", "--test-threads=1"])
        .env(CASE_VAR, name)
        .env("P2A_DATA_ROOT", &root);
    match cap_mb {
        Some(mb) => child.env(CAP_VAR, mb),
        None => child.env_remove(CAP_VAR),
    };
    let out = child.output().unwrap();
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        out.status.success(),
        "case {name} failed: {}\nstdout:\n{}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn local_server_parses_bodies_up_to_the_default_cap() {
    run_case("local", None);
}

#[test]
fn local_server_parses_bodies_up_to_a_lowered_cap() {
    run_case("local-lowered", Some("4"));
}

#[test]
fn hosted_server_keeps_parsed_bodies_to_2_mib_under_the_default_cap() {
    run_case("hosted", None);
}

#[test]
fn hosted_server_applies_a_cap_lowered_below_2_mib() {
    run_case("hosted-lowered", Some("1"));
}

/// What one case serves and expects.
struct Case {
    config: ServerConfig,
    /// The body cap (`P2A_MAX_HTTP_BODY_MB`).
    cap: usize,
    /// The largest JSON body the routes parse.
    parsed: usize,
}

/// One case, in the child process [`run_case`] starts; nothing otherwise.
#[test]
fn case() {
    let Ok(name) = std::env::var(CASE_VAR) else {
        return;
    };
    let case = match name.as_str() {
        "local" => Case {
            config: local(),
            cap: 32 * MIB,
            parsed: 32 * MIB,
        },
        "local-lowered" => Case {
            config: local(),
            cap: 4 * MIB,
            parsed: 4 * MIB,
        },
        "hosted" => Case {
            config: open_public(),
            cap: 32 * MIB,
            parsed: 2 * MIB,
        },
        "hosted-lowered" => Case {
            config: open_public(),
            cap: MIB,
            parsed: MIB,
        },
        other => panic!("unknown case {other}"),
    };
    let hosted = case.config.resource_profile() == ResourceProfile::Hosted;
    assert_eq!(hosted, name.starts_with("hosted"), "{name}");
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            check_served_limits(&case).await;
            check_built_app_limits(&case).await;
        });
}

/// A local configuration (loopback, no token): the resource profile is off.
fn local() -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = "127.0.0.1:0".parse().unwrap();
    c
}

/// An open public configuration (no token, `--allow-unauthenticated`): the
/// hosted profile.
fn open_public() -> ServerConfig {
    let mut c = local();
    c.http.allow_unauthenticated = true;
    c
}

/// A CSV of three noisy numeric columns at least `bytes` long.
fn csv_of_at_least(bytes: usize) -> String {
    let mut csv = String::from("x,y,z\n");
    let mut i = 0u64;
    while csv.len() < bytes {
        let noise = ((i * 7919) % 1000) as f64 / 1000.0 - 0.5;
        csv.push_str(&format!(
            "{i},{:.4},{}\n",
            0.5 * i as f64 + noise,
            (i * 31) % 97
        ));
        i += 1;
    }
    csv
}

/// The JSON body of a `create_dataset` call in `sid`, padded with trailing
/// whitespace (which JSON allows) to exactly `bytes`.
fn create_dataset_body(sid: &str, bytes: usize) -> Vec<u8> {
    let body = |csv: &str| {
        json!({"session_id": sid, "arguments": {"name": "inline", "csv_content": csv}}).to_string()
    };
    // Each row's newline is escaped as two bytes, so stop well short.
    let overhead = body("").len();
    let csv = csv_of_at_least(bytes.saturating_sub(overhead + 64) * 9 / 10);
    pad(body(&csv), bytes)
}

/// The JSON body of an `upload_dataset` call in `sid` (a base64-encoded CSV
/// file), padded with trailing whitespace to exactly `bytes`.
fn upload_dataset_body(sid: &str, bytes: usize) -> Vec<u8> {
    let csv = csv_of_at_least(bytes.saturating_sub(512) * 3 / 4);
    let content = base64::engine::general_purpose::STANDARD.encode(csv.as_bytes());
    let body = json!({
        "session_id": sid,
        "arguments": {"content": content, "filename": "upload.csv", "name": "uploaded"}
    });
    pad(body.to_string(), bytes)
}

fn pad(body: String, bytes: usize) -> Vec<u8> {
    assert!(body.len() <= bytes, "{} > {bytes}", body.len());
    let mut body = body.into_bytes();
    body.resize(bytes, b' ');
    body
}

/// An answer read from a socket: status, `Content-Type` and body.
struct Answer {
    status: u16,
    content_type: String,
    body: Vec<u8>,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Send one HTTP/1.1 POST to `addr` declaring `Content-Length: length`,
/// write `body` (which may be shorter: the server can answer from the
/// headers), and read the answer by its own `Content-Length`. The body is
/// written while the answer is read, so an answer the server sends before it
/// has read the whole body (a refusal) is read, not lost to a write error.
async fn post(addr: SocketAddr, uri: &str, length: usize, body: &[u8]) -> Answer {
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut stream, mut writer) = stream.into_split();
    let mut request = format!(
        "POST {uri} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {length}\r\nConnection: close\r\n\r\n"
    )
    .into_bytes();
    request.extend_from_slice(body);
    let written = tokio::spawn(async move {
        // A server that answers early may close before the body is written.
        let _ = writer.write_all(&request).await;
        writer
    });
    let mut buf = Vec::new();
    let end = loop {
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break end;
        }
        let mut chunk = [0u8; 8192];
        let n = stream
            .read(&mut chunk)
            .await
            .unwrap_or_else(|e| panic!("{uri}: no answer to a {length}-byte body: {e}"));
        assert!(n > 0, "connection closed before the answer's headers");
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.lines();
    let status = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {head:?}"));
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    let mut body = buf[end + 4..].to_vec();
    let length: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no Content-Length in {head:?}"));
    while body.len() < length {
        let mut chunk = [0u8; 8192];
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "connection closed inside the answer's body");
        body.extend_from_slice(&chunk[..n]);
    }
    written.abort();
    Answer {
        status,
        content_type: header("content-type").unwrap_or_default(),
        body,
    }
}

async fn post_all(addr: SocketAddr, uri: &str, body: &[u8]) -> Answer {
    post(addr, uri, body.len(), body).await
}

/// Asserts that `answer` is a plain-text 413 whose body is `text`.
fn assert_refused(answer: &Answer, text: &str, what: &str) {
    assert_eq!(answer.status, 413, "{what}: {}", answer.text());
    assert_eq!(answer.text(), text, "{what}");
    assert!(
        answer.content_type.starts_with("text/plain"),
        "{what}: {}",
        answer.content_type
    );
}

/// Asserts that `answer` is a successful tool call.
fn assert_ran(answer: &Answer, what: &str) {
    assert_eq!(answer.status, 200, "{what}: {}", answer.text());
    let json = answer.json();
    assert_eq!(json["data"]["success"], json!(true), "{what}: {json}");
}

/// The sizes a case sends a JSON body of and expects to run: one between
/// axum's default and the parse limit where there is room, and the parse
/// limit itself where that is cheap (up to 4 MiB).
fn accepted_sizes(case: &Case) -> Vec<usize> {
    let mut sizes = Vec::new();
    if AXUM_DEFAULT_LIMIT + MIB < case.parsed {
        sizes.push(AXUM_DEFAULT_LIMIT + MIB);
    }
    if case.parsed <= 4 * MIB {
        sizes.push(case.parsed);
    }
    sizes
}

/// A server started through `bind` + `serve`, as the binary does, runs a
/// JSON body up to the case's parse limit on the JSON route
/// (`create_dataset`) and on the upload route (`upload_dataset`), refuses one
/// byte more, and refuses a body declared above the cap from its headers.
async fn check_served_limits(case: &Case) {
    let listener = p2a_mcp::transport::http::bind(&case.config).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(p2a_mcp::transport::http::serve(
        listener,
        case.config.clone(),
        async {
            let _ = stopped.await;
        },
    ));

    let created = post_all(addr, "/api/sessions", b"{}").await;
    assert_eq!(created.status, 201, "{}", created.text());
    let sid = created.json()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    for size in accepted_sizes(case) {
        let body = create_dataset_body(&sid, size);
        let answer = post_all(addr, "/api/tools/create_dataset", &body).await;
        assert_ran(&answer, &format!("create_dataset, {size} bytes"));
        let body = upload_dataset_body(&sid, size);
        let answer = post_all(addr, "/api/tools/upload_dataset", &body).await;
        assert_ran(&answer, &format!("upload_dataset, {size} bytes"));
    }

    for uri in ["/api/tools/create_dataset", "/api/tools/upload_dataset"] {
        // Declared above the cap: refused from the headers alone.
        let answer = post(addr, uri, case.cap + 1, b"").await;
        assert_refused(&answer, CAP_REFUSAL, uri);
        // Within the cap but above the parse limit: refused while it is read,
        // each route with a body of its own shape.
        if case.parsed < case.cap {
            let body = if uri.ends_with("/upload_dataset") {
                upload_dataset_body(&sid, case.parsed + 1)
            } else {
                create_dataset_body(&sid, case.parsed + 1)
            };
            let answer = post_all(addr, uri, &body).await;
            assert_refused(&answer, READ_REFUSAL, uri);
        }
    }

    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
}

/// The router of `build_app` (the entry point of an embedder that serves it
/// itself) applies the same limits, also to a body that declares no length.
async fn check_built_app_limits(case: &Case) {
    let (app, _state) = p2a_mcp::transport::http::build_app(&case.config)
        .await
        .unwrap();
    let send = |req: Request<Body>| {
        let app = app.clone();
        async move {
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            (status, String::from_utf8_lossy(&bytes).into_owned())
        }
    };
    let (status, created) = send(
        Request::post("/api/sessions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let created: Value = serde_json::from_str(&created).unwrap();
    let sid = created["data"]["session_id"].as_str().unwrap().to_string();

    // A streamed body: no Content-Length.
    let streamed = |body: Vec<u8>| {
        let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
            body.chunks(64 * 1024).map(|c| Ok(c.to_vec())).collect();
        Request::post("/api/tools/create_dataset")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from_stream(futures::stream::iter(chunks)))
            .unwrap()
    };
    let largest = accepted_sizes(case)[0];
    let (status, answer) = send(streamed(create_dataset_body(&sid, largest))).await;
    assert_eq!(status, StatusCode::OK, "{largest} bytes: {answer}");
    let answer: Value = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["data"]["success"], json!(true), "{answer}");

    // One byte above the parse limit is cut off: 413.
    let mut body = create_dataset_body(&sid, largest);
    body.resize(case.parsed + 1, b' ');
    let (status, answer) = send(streamed(body)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{answer}");
    assert_eq!(answer, READ_REFUSAL);
}
