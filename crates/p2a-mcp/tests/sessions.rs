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
use axum::http::{Request, StatusCode, header};
use p2a_mcp::config::ServerConfig;
use p2a_mcp::db::DatasetMeta;
use p2a_mcp::path_jail::allowed_data_root;
use p2a_mcp::persistent_session::PersistentSessionManager;
use p2a_mcp::transport::http::{AppState, build_app};
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
    let bytes = axum::body::to_bytes(resp.into_body(), 16 << 20)
        .await
        .unwrap();
    (
        status,
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
#[allow(dead_code)]
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
    let outside_csv = std::env::temp_dir().join("p2a-wp07-outside.csv");
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
