//! Integration tests for the HTTP transport: routes, the tool governor,
//! deployment modes and middleware (T0-WP01).
//!
//! Servers are built in-process and driven with `tower::ServiceExt::oneshot`;
//! nothing binds a socket except the embedded-server tests.
#![cfg(feature = "http")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::exec::{ExecConfig, ToolCallError};
use p2a_mcp::session::Session;
use p2a_mcp::transport::http::{AppState, build_app, build_app_with};
use serde_json::{Value, json};
use tower::ServiceExt;

/// An HTTP server configuration with the given bind address, token and
/// open-mode flag; everything else at its default.
fn http_cfg(addr: &str, token: Option<&str>, allow_unauth: bool) -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = addr.parse().unwrap();
    c.http.access_token = token.map(str::to_string);
    c.http.allow_unauthenticated = allow_unauth;
    c
}

async fn app(cfg: &ServerConfig) -> (Router, AppState) {
    build_app(cfg).await.unwrap()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, body)
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn new_session(app: &Router) -> String {
    let (s, _, b) = send(app, post_json("/api/sessions", json!({}))).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    b["data"]["session_id"].as_str().unwrap().to_string()
}

async fn call(app: &Router, sid: &str, tool: &str, args: Value) -> (StatusCode, HeaderMap, Value) {
    send(
        app,
        post_json(
            &format!("/api/tools/{tool}"),
            json!({"session_id": sid, "arguments": args}),
        ),
    )
    .await
}

/// Poll `cond` every 20 ms for up to 10 s; panic if it never holds.
#[allow(dead_code)]
async fn wait_until(mut cond: impl FnMut() -> bool, what: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for: {what}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Sets its flag when dropped. A test that holds a spinning closure keeps
/// one alive, so the closure gets its exit even when an assertion fails
/// first: dropping a runtime waits for its blocking tasks.
struct Release(Arc<AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// The default local configuration.
fn local() -> ServerConfig {
    http_cfg("127.0.0.1:8080", None, false)
}

#[tokio::test]
async fn characterization_core_routes() {
    let (app, _) = app(&http_cfg("127.0.0.1:8080", None, false)).await;
    assert_eq!(send(&app, get("/health")).await.0, StatusCode::OK);
    let sid = new_session(&app).await;
    assert_eq!(
        send(&app, get(&format!("/api/sessions/{sid}"))).await.0,
        StatusCode::OK
    );
    assert_eq!(send(&app, get("/api/tools")).await.0, StatusCode::OK);
    let (s, _, b) = call(&app, &sid, "list_datasets", json!({})).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["data"]["success"], json!(true));
    assert_eq!(
        call(&app, &sid, "no_such_tool", json!({})).await.0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        send(&app, delete(&format!("/api/sessions/{sid}"))).await.0,
        StatusCode::NO_CONTENT
    );
}

// ---------------------------------------------------------------------------
// One guarded start path (MCP-24)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embedded_server_refuses_non_loopback_bind() {
    let r = p2a_mcp::start_embedded_server(
        p2a_mcp::EmbeddedServerConfig::default()
            .with_host("0.0.0.0")
            .with_port(0),
    )
    .await;
    let e = r
        .err()
        .expect("a public bind without a token must be refused");
    assert!(e.to_string().contains("requires an access token"), "{e}");
}

#[tokio::test]
async fn embedded_server_binds_before_returning_and_reports_real_port() {
    let s = p2a_mcp::start_embedded_server(p2a_mcp::EmbeddedServerConfig::default().with_port(0))
        .await
        .unwrap();
    assert_ne!(s.port(), 0);
    s.shutdown();
}

// ---------------------------------------------------------------------------
// The tool governor (MCP-1, MCP-10)
// ---------------------------------------------------------------------------

async fn app_with(cfg: &ServerConfig, exec: ExecConfig) -> (Router, AppState) {
    build_app_with(cfg, exec).await.unwrap()
}

async fn session_of(state: &AppState, sid: &str) -> Arc<Session> {
    state.session_manager.get_session(sid).await.unwrap()
}

async fn generate(
    app: &Router,
    sid: &str,
    name: &str,
    rows: usize,
) -> (StatusCode, HeaderMap, Value) {
    call(
        app,
        sid,
        "generate_random_data",
        json!({
            "n_rows": rows,
            "columns": [
                {"name": "y", "distribution": {"type": "normal", "mean": 0.0, "std": 1.0}},
                {"name": "x", "distribution": {"type": "normal", "mean": 0.0, "std": 1.0}}
            ],
            "seed": 7,
            "name": name
        }),
    )
    .await
}

#[tokio::test(flavor = "current_thread")] // one runtime thread, like the 1-vCPU deploy
async fn health_answers_while_a_tool_runs_on_one_thread() {
    let (app, _) = build_app(&local()).await.unwrap();
    let sid = new_session(&app).await;
    assert_eq!(generate(&app, &sid, "d", 500).await.0, StatusCode::OK);
    let done = Arc::new(AtomicBool::new(false));
    let (app2, sid2, done2) = (app.clone(), sid.clone(), done.clone());
    let tool = tokio::spawn(async move {
        let r = call(
            &app2,
            &sid2,
            "regression_bootstrap_cov",
            json!({"dataset": "d", "y": "y", "x": ["x"], "n_boot": 20000, "seed": 1}),
        )
        .await;
        done2.store(true, Ordering::SeqCst);
        r
    });
    for _ in 0..3 {
        tokio::task::yield_now().await;
    }
    assert_eq!(send(&app, get("/health")).await.0, StatusCode::OK);
    assert!(
        !done.load(Ordering::SeqCst),
        "/health must answer while the tool is still running"
    );
    let (s, _, b) = tool.await.unwrap();
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["data"]["success"], json!(true), "{b}");
}

#[tokio::test]
async fn cooperative_work_stops_at_deadline() {
    let exec = ExecConfig {
        deadline: Some(Duration::from_millis(200)),
        ..Default::default()
    };
    let (app, state) = app_with(&local(), exec).await;
    let sid = new_session(&app).await;
    let gov = state.governor.clone();
    let r = gov
        .run_blocking(&session_of(&state, &sid).await, "loop", |_budget| {
            loop {
                if p2a_core::resource::checkpoint().is_err() {
                    return Err::<(), _>(ToolCallError::Internal("stopped".into()));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        })
        .await;
    assert!(
        matches!(r, Err(ToolCallError::DeadlineExceeded { .. })),
        "{r:?}"
    );
    wait_until(|| gov.health().in_flight == 0, "the work to end").await;
    assert_eq!(gov.health().overdue, 0);
}

#[tokio::test]
async fn uncooperative_work_is_reported_overdue() {
    let exec = ExecConfig {
        concurrency: 1,
        deadline: Some(Duration::from_millis(200)),
        ..Default::default()
    };
    let (app, state) = app_with(&local(), exec).await;
    let sid = new_session(&app).await;
    let gov = state.governor.clone();
    let release = Arc::new(AtomicBool::new(false));
    let _release_on_exit = Release(release.clone());
    let rel = release.clone();
    let r = gov
        .run_blocking(&session_of(&state, &sid).await, "spin", move |_budget| {
            while !rel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        })
        .await;
    assert!(
        matches!(r, Err(ToolCallError::DeadlineExceeded { .. })),
        "{r:?}"
    );
    let h = gov.health();
    assert_eq!(
        (h.in_flight, h.overdue, h.capacity, h.healthy),
        (1, 1, 1, false)
    );
    let (s, _, body) = send(&app, get("/health")).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["status"], json!("degraded"));
    assert_eq!(body["governor"]["overdue"], json!(1));
    assert_eq!(body["governor"]["capacity"], json!(1));
    release.store(true, Ordering::SeqCst);
    wait_until(|| gov.health().in_flight == 0, "the work to end").await;
    assert!(gov.health().healthy);
    let (s, _, body) = send(&app, get("/health")).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(body["status"], json!("ok"));
}

#[tokio::test]
async fn dropping_the_call_cancels_the_work() {
    let (app, state) = app_with(&local(), ExecConfig::default()).await;
    let sid = new_session(&app).await;
    let gov = state.governor.clone();
    let session = session_of(&state, &sid).await;
    let observed = Arc::new(AtomicBool::new(false));
    let seen = observed.clone();
    let g = gov.clone();
    let task = tokio::spawn(async move {
        g.run_blocking(&session, "wait", move |_budget| {
            while p2a_core::resource::checkpoint().is_ok() {
                std::thread::sleep(Duration::from_millis(5));
            }
            seen.store(true, Ordering::SeqCst);
            Ok(())
        })
        .await
    });
    wait_until(|| gov.health().in_flight == 1, "the work to start").await;
    task.abort();
    wait_until(
        || observed.load(Ordering::SeqCst),
        "the work to see the cancel",
    )
    .await;
    wait_until(|| gov.health().in_flight == 0, "the work to end").await;
}

#[tokio::test]
async fn saturation_refuses_with_retry_after() {
    let exec = ExecConfig {
        concurrency: 1,
        session_concurrency: 1,
        queue_wait: Duration::from_millis(100),
        ..Default::default()
    };
    let (app, state) = app_with(&local(), exec).await;
    let (a, b) = (new_session(&app).await, new_session(&app).await);
    let gov = state.governor.clone();
    let (sa, sb) = (session_of(&state, &a).await, session_of(&state, &b).await);
    let release = Arc::new(AtomicBool::new(false));
    let _release_on_exit = Release(release.clone());
    let rel = release.clone();
    let (g, s) = (gov.clone(), sa.clone());
    let holder = tokio::spawn(async move {
        g.run_blocking(&s, "hold", move |_budget| {
            while !rel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        })
        .await
    });
    wait_until(|| gov.health().in_flight == 1, "the holder to start").await;

    let r = gov.run_blocking(&sa, "second", |_| Ok(())).await;
    assert!(
        matches!(
            r,
            Err(ToolCallError::SessionBusy {
                retry_after_secs: 2
            })
        ),
        "{r:?}"
    );
    let r = gov.run_blocking(&sb, "other", |_| Ok(())).await;
    assert!(
        matches!(
            r,
            Err(ToolCallError::ServerBusy {
                retry_after_secs: 5
            })
        ),
        "{r:?}"
    );

    let (s, h, body) = call(&app, &b, "list_datasets", json!({})).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(h[header::RETRY_AFTER], "5");
    assert_eq!(body["code"], json!("server_busy"));
    let (s, h, body) = call(&app, &a, "list_datasets", json!({})).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(h[header::RETRY_AFTER], "2");
    assert_eq!(body["code"], json!("session_busy"));

    release.store(true, Ordering::SeqCst);
    assert!(holder.await.unwrap().is_ok());
}

#[tokio::test]
async fn panic_leaves_the_session_usable() {
    let (app, state) = app_with(&local(), ExecConfig::default()).await;
    let sid = new_session(&app).await;
    let gov = state.governor.clone();
    let session = session_of(&state, &sid).await;
    let handle = tokio::runtime::Handle::current();
    let s = session.clone();
    let r = gov
        .run_blocking(
            &session,
            "boom",
            move |_budget| -> Result<(), ToolCallError> {
                let _guard = handle.block_on(s.datasets.write());
                panic!("boom")
            },
        )
        .await;
    match r {
        Err(ToolCallError::Internal(m)) => assert!(m.contains("failed unexpectedly"), "{m}"),
        other => panic!("expected Internal, got {other:?}"),
    }
    let (s, _, b) = call(&app, &sid, "list_datasets", json!({})).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["data"]["success"], json!(true));
    wait_until(|| gov.health().in_flight == 0, "the work to end").await;
}

#[tokio::test]
async fn session_quota_rolls_back_the_new_dataset() {
    let exec = ExecConfig {
        session_data_bytes: Some(1_000_000),
        ..Default::default()
    };
    let (app, _) = app_with(&local(), exec).await;
    let sid = new_session(&app).await;
    assert_eq!(generate(&app, &sid, "small", 1000).await.0, StatusCode::OK);
    let (s, _, b) = generate(&app, &sid, "big", 100_000).await; // about 1.6 MB
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{b}");
    assert_eq!(b["code"], json!("quota_exceeded"));
    let msg = b["error"].as_str().unwrap();
    assert!(
        msg.contains("per session") && msg.contains("undone"),
        "{msg}"
    );
    let (_, _, b) = call(&app, &sid, "list_datasets", json!({})).await;
    let listing = b.to_string();
    assert!(
        listing.contains("**small**") && !listing.contains("**big**"),
        "{listing}"
    );
}

#[tokio::test]
async fn total_quota_spans_sessions() {
    let exec = ExecConfig {
        total_data_bytes: Some(1_000_000),
        ..Default::default()
    };
    let (app, _) = app_with(&local(), exec).await;
    let (a, b) = (new_session(&app).await, new_session(&app).await);
    assert_eq!(generate(&app, &a, "first", 40_000).await.0, StatusCode::OK); // about 640 kB
    let (s, _, body) = generate(&app, &b, "second", 40_000).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["code"], json!("quota_exceeded"));
    let (_, _, listing_b) = call(&app, &b, "list_datasets", json!({})).await;
    assert!(!listing_b.to_string().contains("**second**"), "{listing_b}");
    let (_, _, listing_a) = call(&app, &a, "list_datasets", json!({})).await;
    assert!(listing_a.to_string().contains("**first**"), "{listing_a}");
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn every_http_entry_is_governed() {
    let exec = ExecConfig {
        rss_soft_limit_bytes: Some(1),
        ..Default::default()
    };
    let (app, _) = app_with(&local(), exec).await;
    let sid = new_session(&app).await;
    let (s, h, b) = call(&app, &sid, "list_datasets", json!({})).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(b["code"], json!("memory_pressure"));
    assert_eq!(h[header::RETRY_AFTER], "15");
}

#[tokio::test]
async fn unknown_tool_and_bad_arguments_have_codes() {
    let (app, _) = build_app(&local()).await.unwrap();
    let sid = new_session(&app).await;
    let (s, _, b) = call(&app, &sid, "no_such_tool", json!({})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(b["code"], json!("unknown_tool"));
    let (s, _, b) = call(&app, &sid, "generate_random_data", json!({"n_rows": "x"})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(b["code"], json!("invalid_arguments"));
}

#[tokio::test]
async fn full_session_table_answers_503_with_code() {
    let mut cfg = local();
    cfg.session.max_sessions = 1;
    let (app, _) = build_app(&cfg).await.unwrap();
    new_session(&app).await;
    let (s, h, b) = send(&app, post_json("/api/sessions", json!({}))).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(b["code"], json!("capacity"));
    assert!(h.contains_key(header::RETRY_AFTER));
}
// ---------------------------------------------------------------------------
// Governor health (MCP-1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_reports_governor_fields() {
    let (app, _) = build_app(&local()).await.unwrap();
    let (s, _, body) = send(&app, get("/health")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["status"], json!("ok"));
    for key in ["in_flight", "overdue", "capacity", "healthy"] {
        assert!(
            body["governor"].get(key).is_some(),
            "missing governor.{key}: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// Deployment modes: enumeration routes and the Host check (MCP-5)
// ---------------------------------------------------------------------------

fn with_bearer(mut req: Request<Body>, bearer: Option<&str>) -> Request<Body> {
    if let Some(t) = bearer {
        req.headers_mut().insert(
            header::AUTHORIZATION,
            format!("Bearer {t}").parse().unwrap(),
        );
    }
    req
}

/// Status of `GET /api/sessions` and `GET /api/files`.
async fn enum_status(cfg: &ServerConfig, bearer: Option<&str>) -> (StatusCode, StatusCode) {
    let (app, _) = build_app(cfg).await.unwrap();
    let sessions = send(&app, with_bearer(get("/api/sessions"), bearer))
        .await
        .0;
    let files = send(&app, with_bearer(get("/api/files"), bearer)).await.0;
    (sessions, files)
}

#[tokio::test]
async fn enumeration_routes_follow_the_mode() {
    // Local default: hidden (405 = POST-only path, 404 = unregistered)
    assert_eq!(
        enum_status(&http_cfg("127.0.0.1:8080", None, false), None).await,
        (StatusCode::METHOD_NOT_ALLOWED, StatusCode::NOT_FOUND)
    );
    // MCP-5 repro: loopback + P2A_ALLOW_UNAUTHENTICATED -> OpenPublic -> hidden (was 200, 200)
    assert_eq!(
        enum_status(&http_cfg("127.0.0.1:8080", None, true), None).await,
        (StatusCode::METHOD_NOT_ALLOWED, StatusCode::NOT_FOUND)
    );
    let mut c = http_cfg("127.0.0.1:8080", None, false);
    c.http.expose_enumeration = true;
    assert_eq!(
        enum_status(&c, None).await,
        (StatusCode::OK, StatusCode::OK)
    );
    let mut c = http_cfg("0.0.0.0:8080", None, true);
    c.http.expose_enumeration = true; // never in OpenPublic
    assert_eq!(
        enum_status(&c, None).await,
        (StatusCode::METHOD_NOT_ALLOWED, StatusCode::NOT_FOUND)
    );
    assert_eq!(
        enum_status(&http_cfg("0.0.0.0:8080", Some("t"), false), Some("t")).await,
        (StatusCode::OK, StatusCode::OK)
    );
}

#[tokio::test]
async fn local_mode_rejects_foreign_host_headers() {
    let (app, _) = build_app(&local()).await.unwrap();
    let with_host = |h: &str| {
        Request::get("/health")
            .header(header::HOST, h)
            .body(Body::empty())
            .unwrap()
    };
    for ok in [
        "localhost:8081",
        "LOCALHOST",
        "127.0.0.1:8080",
        "[::1]:8080",
        "127.0.0.5",
    ] {
        assert_eq!(send(&app, with_host(ok)).await.0, StatusCode::OK, "{ok}");
    }
    for bad in [
        "evil.example",
        "localhost.evil.example",
        "0.0.0.0:8080",
        "127.0.0.1.nip.io",
    ] {
        let (s, _, b) = send(&app, with_host(bad)).await;
        assert_eq!(s, StatusCode::MISDIRECTED_REQUEST, "{bad}");
        assert_eq!(b["code"], json!("misdirected"));
    }
    assert_eq!(send(&app, get("/health")).await.0, StatusCode::OK); // no Host header: not a browser
    let (app, _) = build_app(&http_cfg("0.0.0.0:8080", None, true))
        .await
        .unwrap();
    assert_eq!(
        send(&app, with_host("p2a-api.qamelab.org")).await.0,
        StatusCode::OK
    );
}
