//! The persistence protocol of a streamed chat turn, driven as the web UI drives it
//! (ChatPanel::handle_send): the client stores the user's message, then streams the turn with the
//! transcript before it as `history`; the backend stores the reply. Left out on purpose: the
//! client's own POST of the reply on Done (chat_panel.rs), which stores a second assistant row.
//! db + llm builds only (an in-memory SurrealDB and a loopback Ollama stand-in). Owned by T3-WP54.
#![cfg(all(feature = "db", feature = "llm"))]

use crate::common::{http_app, new_session, spawn_stub};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

/// One request: the status and the whole body as text (an SSE stream is read to its end).
async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    session: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, String) {
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header("host", "localhost");
    if let Some(s) = session {
        req = req.header("x-p2a-session-id", s);
    }
    let req = match body {
        Some(b) => req
            .header("content-type", "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .unwrap_or_else(|_| panic!("{method} {uri}: no response within 30 s"))
        .unwrap();
    let status = resp.status();
    let bytes = tokio::time::timeout(
        Duration::from_secs(30),
        axum::body::to_bytes(resp.into_body(), usize::MAX),
    )
    .await
    .unwrap_or_else(|_| panic!("{method} {uri}: no complete body within 30 s"))
    .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn send_json(
    app: &axum::Router,
    method: &str,
    uri: &str,
    session: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let (status, text) = send(app, method, uri, session, body).await;
    (
        status,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    )
}

/// An Ollama stand-in on loopback: answers request n with `answers[n]` as one NDJSON line (the
/// last answer repeats) and records every request body.
async fn ollama(answers: Vec<Value>) -> (String, Arc<Mutex<Vec<Value>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let (b, a) = (bodies.clone(), Arc::new(answers));
    let router = axum::Router::new().route(
        "/api/chat",
        axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let (b, a) = (b.clone(), a.clone());
            async move {
                let n = {
                    let mut g = b.lock().unwrap();
                    g.push(body);
                    g.len() - 1
                };
                (
                    [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
                    format!("{}\n", a[n.min(a.len() - 1)]),
                )
            }
        }),
    );
    (format!("http://{}", spawn_stub(router).await), bodies)
}

fn says(text: &str) -> Value {
    json!({"message": {"role": "assistant", "content": text}, "done": true})
}

fn calls(tool: &str) -> Value {
    json!({"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": tool, "arguments": {}}}]}, "done": true})
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn role_content(rows: &[Value]) -> Vec<(String, String)> {
    rows.iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap_or_default().to_string(),
                m["content"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// The SSE events of a chat-stream body (`data: {json}` lines; keep-alive comments skipped).
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("SSE data {d:?}: {e}")))
        .collect()
}

/// The stored messages of `cid`, in order.
async fn stored(app: &axum::Router, sid: &str, cid: &str) -> Vec<Value> {
    let (s, b) = send_json(
        app,
        "GET",
        &format!("/api/conversations/{cid}/messages"),
        Some(sid),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    b["data"].as_array().cloned().unwrap_or_default()
}

/// One turn as ChatPanel::handle_send sends it: the client stores the user row, then streams the
/// turn (retrieve_history true, as platform/streaming.rs always sends). Returns the SSE events.
async fn turn(
    app: &axum::Router,
    sid: &str,
    cid: &str,
    base_url: &str,
    history: Value,
    message: &str,
) -> Vec<Value> {
    let (s, b) = send_json(
        app,
        "POST",
        &format!("/api/conversations/{cid}/messages"),
        Some(sid),
        Some(json!({"role": "user", "content": message})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    let (s, body) = send(
        app,
        "POST",
        "/api/llm/chat/stream",
        None,
        Some(json!({
            "session_id": sid, "message": message, "history": history, "conversation_id": cid,
            "retrieve_history": true, "interpret": true,
            "provider": {"provider_type": "ollama", "base_url": base_url, "model": "stub"}})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let ev = events(&body);
    assert_eq!(
        ev.last().and_then(|e| e["type"].as_str()),
        Some("done"),
        "{body}"
    );
    ev
}

/// Each streamed turn stores exactly one assistant row, holding the final reply, and the turn's
/// tool call is attached to that row. With `history` the transcript before the prompt (what
/// ChatState::build_history sends), the model gets each earlier turn once and the prompt once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_streamed_turns_store_one_assistant_row_each() {
    let (app, _state) = http_app().await;
    let sid = new_session(&app).await;
    let (s, conv) = send_json(
        &app,
        "POST",
        "/api/conversations",
        None,
        Some(json!({"session_id": sid, "title": "New Conversation"})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{conv}");
    let cid = conv["data"]["id"]
        .as_str()
        .expect("conversation id")
        .to_string();
    let (base_url, bodies) = ollama(vec![
        says("Loaded."),
        calls("list_datasets"),
        says("No datasets are loaded yet."),
    ])
    .await;

    // Turn 1: nothing before it. (Request 0's turns are not asserted: with an empty history the
    // backend loads the stored rows, including the user row just posted, and appends `message`
    // again, an open defect of its own.)
    let ev = turn(&app, &sid, &cid, &base_url, json!([]), "load the data").await;
    assert_eq!(ev.last().unwrap()["message"]["content"], "Loaded.");
    assert_eq!(
        role_content(&stored(&app, &sid, &cid).await),
        pairs(&[("user", "load the data"), ("assistant", "Loaded.")])
    );

    // Turn 2: the transcript before the prompt as history; the model calls a tool, then answers.
    let history = json!([
        {"role": "user", "content": "load the data", "tool_calls": null, "tool_results": null},
        {"role": "assistant", "content": "Loaded.", "tool_calls": null, "tool_results": null}
    ]);
    let ev = turn(
        &app,
        &sid,
        &cid,
        &base_url,
        history,
        "how many datasets are loaded?",
    )
    .await;
    let tool_events: Vec<(&str, &str)> = ev
        .iter()
        .filter_map(|e| Some((e["type"].as_str()?, e["tool"].as_str()?)))
        .collect();
    assert_eq!(
        tool_events,
        [
            ("tool_start", "list_datasets"),
            ("tool_end", "list_datasets")
        ],
        "{ev:?}"
    );
    assert_eq!(
        ev.last().unwrap()["message"]["content"],
        "No datasets are loaded yet."
    );

    let rows = stored(&app, &sid, &cid).await;
    assert_eq!(
        role_content(&rows),
        pairs(&[
            ("user", "load the data"),
            ("assistant", "Loaded."),
            ("user", "how many datasets are loaded?"),
            ("assistant", "No datasets are loaded yet."),
        ])
    );

    let (s, tc) = send_json(
        &app,
        "GET",
        &format!("/api/conversations/{cid}/tool-calls"),
        Some(&sid),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{tc}");
    let tc = tc["data"].as_array().cloned().unwrap_or_default();
    assert_eq!(tc.len(), 1, "{tc:?}");
    assert_eq!(
        (tc[0]["tool_name"].as_str(), tc[0]["status"].as_str()),
        (Some("list_datasets"), Some("success"))
    );
    assert_eq!(
        tc[0]["message_id"], rows[3]["id"],
        "the call belongs to turn 2's reply row"
    );

    // Request 1 is turn 2's first model call: each earlier turn once, then the prompt once.
    let first_call_of_turn_2 = bodies.lock().unwrap()[1].clone();
    let model_turns: Vec<(String, String)> = first_call_of_turn_2["messages"]
        .as_array()
        .expect("messages")[1..]
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap_or_default().to_string(),
                m["content"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    assert_eq!(
        model_turns,
        pairs(&[
            ("user", "load the data"),
            ("assistant", "Loaded."),
            ("user", "how many datasets are loaded?"),
        ])
    );
}
