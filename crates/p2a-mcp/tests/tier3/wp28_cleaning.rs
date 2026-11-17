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

/// A frame with a 75%-missing column, a constant column, outliers on both sides, empty strings
/// and a text-number column.
fn messy() -> Dataset {
    let n = 20;
    let mostly_missing: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 4 == 0).then_some(i as f64 * 0.7 + 0.1))
        .collect();
    let mut wide: Vec<f64> = (0..n)
        .map(|i| 50.0 + (i as f64 * 1.3).sin() * 4.0)
        .collect();
    wide[3] = -400.0;
    wide[11] = 900.0;
    let texts: Vec<&str> = (0..n)
        .map(|i| match i % 5 {
            0 => "",
            1 => "12.5",
            2 => "7",
            3 => "2024-01-15",
            _ => "abc",
        })
        .collect();
    Dataset::new(
        df! {
            "mostly_missing" => mostly_missing,
            "constant" => vec![3i64; n],
            "wide" => wide,
            "texts" => texts,
        }
        .unwrap(),
    )
}

/// The `Arguments:` blocks of a suggest_cleaning reply as request objects ("- key: value"
/// lines; `columns` is JSON). A fill value shown for reference only is left out.
fn suggested_arguments(t: &str) -> Vec<serde_json::Map<String, Value>> {
    let mut out = Vec::new();
    let mut lines = t.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "Arguments:" {
            continue;
        }
        let mut m = serde_json::Map::new();
        for arg in lines.by_ref() {
            let Some((k, v)) = arg
                .trim()
                .strip_prefix("- ")
                .and_then(|a| a.split_once(": "))
            else {
                break;
            };
            if v.contains("(shown for reference") {
                continue;
            }
            let v = if k == "columns" {
                serde_json::from_str(v).unwrap()
            } else {
                Value::from(v)
            };
            m.insert(k.to_string(), v);
        }
        out.push(m);
    }
    out
}

/// DATA-52: suggestions named operations apply does not have (drop_column, standardize, cast) or
/// left out what it needs (a filter's operator, a replace's old value).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_suggestion_applies() {
    let (session, server) = session_with_datasets(vec![("messy", messy())]);
    let r = call(
        &server,
        &session,
        "suggest_cleaning",
        json!({ "dataset": "messy" }),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let all = suggested_arguments(&t);
    assert!(!all.is_empty(), "{t}");
    let mut applied = Vec::new();
    for args in all {
        let op = args["operation"].as_str().unwrap().to_string();
        if op == "none" {
            assert_eq!(args.len(), 1, "{args:?}");
            continue;
        }
        let sid = start(&server, &session, "messy").await;
        let mut args = args.clone();
        args.insert("session_id".into(), Value::from(sid));
        apply(&server, &session, Value::Object(args)).await;
        applied.push(op);
    }
    for op in ["drop_column", "filter", "replace"] {
        assert!(
            applied.iter().any(|a| a == op),
            "{op} not applied: {applied:?}\n{t}"
        );
    }
    assert_eq!(applied.iter().filter(|a| *a == "filter").count(), 2, "{t}");
}

/// drop_column is an operation of preview_cleaning and cleaning_session_apply; rollback restores
/// the column.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preview_and_apply_drop_column() {
    let (session, server) = session_with_datasets(vec![("messy", messy())]);
    let r = call(
        &server,
        &session,
        "preview_cleaning",
        json!({ "dataset": "messy", "operation": "drop_column", "columns": ["constant"] }),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(t.contains("Drop columns: constant"), "{t}");
    assert!(t.contains("Rows affected: 0 (0.0%)"), "{t}");
    let r = call(
        &server,
        &session,
        "preview_cleaning",
        json!({ "dataset": "messy", "operation": "drop_column", "columns": ["nope"] }),
    )
    .await;
    assert!(text(&r).contains("Column 'nope' not found"), "{}", text(&r));

    let sid = start(&server, &session, "messy").await;
    let t = apply(
        &server,
        &session,
        json!({ "session_id": sid, "operation": "drop_column", "columns": ["constant"],
                "output_dataset": "after" }),
    )
    .await;
    assert!(t.contains("drop_column constant"), "{t}");
    let r = call(
        &server,
        &session,
        "cleaning_rollback",
        json!({ "session_id": sid, "output_dataset": "back" }),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let names = |ds: &Dataset| -> Vec<String> {
        ds.df()
            .get_column_names()
            .iter()
            .map(|c| c.to_string())
            .collect()
    };
    let datasets = session.datasets.read().await;
    assert_eq!(
        names(&datasets["after"]),
        ["mostly_missing", "wide", "texts"]
    );
    assert_eq!(
        names(&datasets["back"]),
        ["mostly_missing", "constant", "wide", "texts"]
    );
}
