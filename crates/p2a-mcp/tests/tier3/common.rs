//! Shared helpers of the Tier 3 MCP-path tests (T3-WP05; T3 design 2.5).
//!
//! Three ways in, each the real entry point:
//! - direct dispatch: a [`Session`] holding the test's datasets, and tools run through
//!   `AnalyticsServer::call_tool_with_session`, the dispatcher the HTTP API uses (request
//!   limits, the response cap and the session's datasets included);
//! - HTTP: the app of `transport::http::build_app` in Local mode (loopback, no token,
//!   limits off), driven with `tower::ServiceExt::oneshot`;
//! - stdio: the rmcp service over an in-memory duplex stream, speaking newline-delimited
//!   JSON-RPC as the stdio transport does over stdin/stdout.
//!
//! Every read of a helper is bounded by a 30-second timeout that panics with a message: a
//! guard against a hang, not a timing assertion.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use p2a_core::data::{DataLoader, Dataset};
use p2a_mcp::AnalyticsServer;
use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::exec::ToolCallError;
use p2a_mcp::session::Session;
use p2a_mcp::transport::http::{AppState, ContentItem, ToolResult};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Session ids of this test binary: unique without a clock or an RNG.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// How long a helper waits for one answer before it fails the test.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Awaits `fut`, panicking with `what` if it takes longer than [`READ_TIMEOUT`].
async fn bounded<T>(what: &str, fut: impl Future<Output = T>) -> T {
    match tokio::time::timeout(READ_TIMEOUT, fut).await {
        Ok(v) => v,
        Err(_) => panic!("{what}: no answer within {READ_TIMEOUT:?}"),
    }
}

/// A session holding the case's dataset under its file stem (e.g. "two_sample_na"), and a
/// dispatcher.
pub fn session_with(case_id: &str) -> (Session, AnalyticsServer) {
    let case = p2a_validation::Case::load(case_id);
    let path = case.dataset_path();
    let df = DataLoader::load_csv(&path)
        .unwrap_or_else(|e| panic!("case {case_id}: cannot load {}: {e}", path.display()));
    let stem = std::path::Path::new(&case.dataset)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("case {case_id}: dataset {} has no file stem", case.dataset))
        .to_string();
    session_with_datasets(vec![(stem.as_str(), Dataset::new(df))])
}

/// A session holding the given datasets (for Rust-built fixtures), and a dispatcher.
///
/// The session's cached byte size is not refreshed: the default limits charge no session
/// quota. A test that selects `Limits::hosted()` calls `session.refresh_data_bytes().await`
/// first.
pub fn session_with_datasets(datasets: Vec<(&str, Dataset)>) -> (Session, AnalyticsServer) {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let session = Session::new(format!("tier3-{id}"), None);
    {
        let mut map = session.datasets.try_write().expect("fresh session");
        for (name, ds) in datasets {
            map.insert(name.to_string(), ds);
        }
    }
    let server = AnalyticsServer::with_session(&session);
    (session, server)
}

/// `server.call_tool_with_session(tool, args, session)`.
pub async fn try_call(
    server: &AnalyticsServer,
    session: &Session,
    tool: &str,
    args: Value,
) -> Result<ToolResult, ToolCallError> {
    server.call_tool_with_session(tool, args, session).await
}

/// [`try_call`]; panics with the tool name on a dispatch error.
pub async fn call(
    server: &AnalyticsServer,
    session: &Session,
    tool: &str,
    args: Value,
) -> ToolResult {
    match try_call(server, session, tool, args).await {
        Ok(r) => r,
        Err(e) => panic!("{tool}: dispatch failed: {e:?}"),
    }
}

/// Concatenated text items of a result, joined with "\n".
pub fn text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ContentItem::Text { text } => Some(text.as_str()),
            ContentItem::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Local mode: HTTP transport on loopback, no token, resource profile off.
pub fn local_config() -> ServerConfig {
    ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    }
}

/// The HTTP app and its state for [`local_config`].
pub async fn http_app() -> (Router, AppState) {
    p2a_mcp::transport::http::build_app(&local_config())
        .await
        .expect("build_app in Local mode")
}

/// POSTs `body` as JSON to `uri` (with `Host: localhost`, which Local mode requires) and
/// returns the status and the JSON body (`Value::Null` for an empty or non-JSON body).
pub async fn post_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    use tower::ServiceExt;
    let req = Request::post(uri)
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = bounded(&format!("POST {uri}"), app.clone().oneshot(req))
        .await
        .expect("infallible service");
    let status = resp.status();
    let bytes = bounded(
        &format!("POST {uri} body"),
        axum::body::to_bytes(resp.into_body(), usize::MAX),
    )
    .await
    .expect("response body");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, value)
}

/// `POST /api/tools/{tool}` with `{"session_id": sid, "arguments": args}`.
pub async fn post_tool(app: &Router, sid: &str, tool: &str, args: Value) -> (StatusCode, Value) {
    post_json(
        app,
        &format!("/api/tools/{tool}"),
        json!({"session_id": sid, "arguments": args}),
    )
    .await
}

/// `POST /api/sessions {}`: the new session's id.
pub async fn new_session(app: &Router) -> String {
    let (status, body) = post_json(app, "/api/sessions", json!({})).await;
    assert!(status.is_success(), "POST /api/sessions: {status} {body}");
    body["data"]["session_id"]
        .as_str()
        .unwrap_or_else(|| panic!("no data.session_id in {body}"))
        .to_string()
}

/// The dataset `name` of session `sid`, if both exist.
pub async fn session_dataset(state: &AppState, sid: &str, name: &str) -> Option<Dataset> {
    state
        .session_manager
        .get_session(sid)
        .await
        .ok()?
        .datasets
        .read()
        .await
        .get(name)
        .cloned()
}

/// Serves `router` on 127.0.0.1 (an ephemeral port) on a background task, for loopback
/// stubs of LLM providers; returns its address.
pub async fn spawn_stub(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// Runs `calls` (tool, arguments) over an in-memory stdio transport (newline-delimited
/// JSON-RPC), in order in one session, each after the previous one was answered, so a later
/// call sees the datasets an earlier one made. Returns each call's JSON-RPC response.
pub async fn stdio_calls(server: AnalyticsServer, calls: &[(&str, Value)]) -> Vec<Value> {
    let (client, server_end) = tokio::io::duplex(1 << 20);
    let (r, w) = tokio::io::split(server_end);
    let service = tokio::spawn(async move {
        let running = rmcp::ServiceExt::serve(server, (r, w))
            .await
            .expect("serve");
        let _ = running.waiting().await;
    });
    let (cr, mut cw) = tokio::io::split(client);
    let mut lines = BufReader::new(cr).lines();

    async fn send<W: AsyncWriteExt + Unpin>(w: &mut W, msg: Value) {
        let mut line = msg.to_string();
        line.push('\n');
        w.write_all(line.as_bytes()).await.expect("write");
        w.flush().await.expect("flush");
    }
    async fn read_until_id<R: AsyncBufReadExt + Unpin>(
        lines: &mut tokio::io::Lines<R>,
        id: u64,
    ) -> Value {
        bounded(&format!("stdio response {id}"), async {
            loop {
                let line = lines
                    .next_line()
                    .await
                    .expect("read")
                    .expect("server closed the stream");
                let msg: Value = serde_json::from_str(&line).expect("JSON-RPC line");
                if msg["id"] == id {
                    return msg;
                }
            }
        })
        .await
    }

    send(
        &mut cw,
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "tier3", "version": "0"}}}),
    )
    .await;
    read_until_id(&mut lines, 0).await;
    send(
        &mut cw,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    .await;
    let mut out = Vec::with_capacity(calls.len());
    for (id, (tool, args)) in (1u64..).zip(calls) {
        send(
            &mut cw,
            json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}),
        )
        .await;
        out.push(read_until_id(&mut lines, id).await);
    }
    drop(cw);
    service.abort();
    out
}
