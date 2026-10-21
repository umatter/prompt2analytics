//! Integration tests for the HTTP transport: routes, the tool governor,
//! deployment modes and middleware (T0-WP01).
//!
//! Servers are built in-process and driven with `tower::ServiceExt::oneshot`;
//! nothing binds a socket except the embedded-server tests.
#![cfg(feature = "http")]

use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::transport::http::{AppState, build_app};
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
