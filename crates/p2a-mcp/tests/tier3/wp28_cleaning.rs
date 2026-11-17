//! T4-WP28 MCP-path tests: cleaning-session undo, suggestions that apply, previews of the
//! operations apply runs. Owned by T4-WP28.
use crate::common::*;
use p2a_core::data::Dataset;
use p2a_core::polars::prelude::*;
use serde_json::{Value, json};

/// The id printed by cleaning_session_start ("Session ID: <id>").
fn session_id(t: &str) -> String {
    t.lines()
        .find_map(|l| l.trim().strip_prefix("Session ID: "))
        .expect("session id")
        .trim()
        .to_string()
}

/// The "Rows: n" a rollback reply reports.
fn rows(t: &str) -> usize {
    t.lines()
        .find_map(|l| l.trim().strip_prefix("Rows: "))
        .unwrap_or_else(|| panic!("no rows in {t}"))
        .trim()
        .parse()
        .unwrap()
}

/// Six rows of noisy x.
fn xs() -> Dataset {
    Dataset::new(df! { "x" => [1.5, 2.25, 3.1, 4.8, 5.05, 6.7] }.unwrap())
}

async fn start(
    server: &p2a_mcp::server::AnalyticsServer,
    session: &p2a_mcp::session::Session,
    dataset: &str,
) -> String {
    let r = call(
        server,
        session,
        "cleaning_session_start",
        json!({ "dataset": dataset }),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    session_id(&text(&r))
}

async fn apply(
    server: &p2a_mcp::server::AnalyticsServer,
    session: &p2a_mcp::session::Session,
    args: Value,
) -> String {
    let r = call(server, session, "cleaning_session_apply", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    t
}

/// DATA-31: after an undo and a new operation, a second undo restored the undone operation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleaning_session_undo_follows_the_branch() {
    let (session, server) = session_with_datasets(vec![("xs", xs())]);
    let sid = start(&server, &session, "xs").await;
    let filter = |op: &str, v: &str| json!({"session_id": sid, "operation": "filter", "columns": ["x"], "operator": op, "filter_value": v});
    apply(&server, &session, filter(">", "2")).await; // A: 5 rows
    apply(&server, &session, filter("<", "6")).await; // B: 4 rows
    let undo = || json!({ "session_id": sid });
    let r = call(&server, &session, "cleaning_rollback", undo()).await;
    assert_eq!(rows(&text(&r)), 5, "{}", text(&r)); // at A
    apply(&server, &session, filter("<", "5")).await; // C on A: 3 rows
    let r = call(&server, &session, "cleaning_rollback", undo()).await;
    assert_eq!(rows(&text(&r)), 5, "back at A, not B: {}", text(&r));
    let r = call(&server, &session, "cleaning_rollback", undo()).await;
    assert_eq!(rows(&text(&r)), 6, "{}", text(&r));
    let r = call(&server, &session, "cleaning_rollback", undo()).await;
    assert!(!r.success);
    assert!(
        text(&r).contains("Already at initial state"),
        "{}",
        text(&r)
    );
}

