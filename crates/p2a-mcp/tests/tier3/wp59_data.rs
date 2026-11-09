//! T3-WP59 MCP-path tests: fill strategies, value counts, replace, describe_dataset, dendrogram input,
//! option parsing and chart outputs (data surface). Owned by T3-WP59.
use crate::common::*;
use p2a_core::data::Dataset;
use p2a_core::polars::prelude::*;
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

/// The session's dataset `name`.
async fn dataset(session: &p2a_mcp::session::Session, name: &str) -> Dataset {
    let map = session.datasets.read().await;
    map.get(name).cloned().unwrap_or_else(|| {
        panic!(
            "no dataset '{name}'; have {:?}",
            map.keys().collect::<Vec<_>>()
        )
    })
}
/// A numeric or boolean column as f64, missing as NaN (as Case::vector reads JSON null).
fn nums(ds: &Dataset, c: &str) -> Vec<f64> {
    ds.df()
        .column(c)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect()
}
/// A text column, missing as "" (the names of the harness's .strings() values).
fn labels(ds: &Dataset, c: &str) -> Vec<String> {
    ds.df()
        .column(c)
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect()
}
/// The id printed by cleaning_session_start ("Session ID: <id>").
fn session_id(t: &str) -> String {
    t.lines()
        .find_map(|l| l.trim().strip_prefix("Session ID: "))
        .expect("session id")
        .trim()
        .to_string()
}

/// DATA-28 / LLM-12: 'mode' did not exist. munge_fill_na wrote the literal text "mode" into text cells
/// (and failed to parse it in numeric ones); the cleaning tools filled "0" and previewed
/// "<strategy value>". R: names(which.max(table(x))) (ties to the first sorted level), then
/// tidyr::replace_na (validation/r/cases/data_munging_t3.R, case data_munging_fill_mode, T3-WP50).
/// (The architect's test_mcp_fill_na_mode, named test_validate_* because it is R-pinned.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_fill_na_mode() {
    let case = Case::load("data_munging_fill_mode");
    let (session, server) = session_with("data_munging_fill_mode"); // dataset "munging_fill"
    let r = call(&server, &session, "munge_fill_na",
        json!({"dataset": "munging_fill", "columns": ["s"], "strategy": "mode", "result_name": "s_only"})).await;
    assert!(r.success, "{}", text(&r));
    assert_eq!(
        labels(&dataset(&session, "s_only").await, "s"),
        case.names("s_filled")
    ); // before: "mode" at rows 2, 6
    let r = call(&server, &session, "munge_fill_na",
        json!({"dataset": "munging_fill", "columns": ["k", "f", "s"], "strategy": "MODE", "result_name": "all"})).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(
        t.contains(
            "Filled 6 missing values in 3 columns: k (2, with 1), f (2, with 2.5), s (2, with 'a')"
        ),
        "{t}"
    );
    assert!(t.contains("Saved as 'all'."), "{t}");
    let all = dataset(&session, "all").await;
    assert_eq!(all.df().column("k").unwrap().dtype(), &DataType::Int64); // the mode keeps the integer type
    Check::new(&case)
        .vector("k_filled", &nums(&all, "k"), Tol::exact())
        .vector("f_filled", &nums(&all, "f"), Tol::exact())
        .finish();
    assert_eq!(labels(&all, "s"), case.names("s_filled"));
    // the cleaning tools: the preview shows the real value, the session fills it
    let p = text(&call(&server, &session, "preview_cleaning",
        json!({"dataset": "munging_fill", "operation": "fill_na", "columns": ["s"], "strategy": "mode"})).await);
    assert!(
        p.contains("| 2 | s | null | a |") && p.contains("| 6 | s | null | a |"),
        "{p}"
    );
    let sid = session_id(&text(
        &call(
            &server,
            &session,
            "cleaning_session_start",
            json!({"dataset": "munging_fill"}),
        )
        .await,
    ));
    let a = call(
        &server,
        &session,
        "cleaning_session_apply",
        json!({"session_id": sid, "operation": "fill_na", "columns": ["s"], "strategy": "Mode"}),
    )
    .await;
    assert!(
        a.success && text(&a).contains("Filled 2 missing values in 1 column: s (2, with 'a')"),
        "{}",
        text(&a)
    );
}

/// DATA-28: a constant was taken from `strategy` and lowercased ('Unknown' became 'unknown'), and with
/// no columns named a text constant failed on the first integer column. The constant now comes from
/// `value` with its case kept; columns that cannot take it are skipped and listed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_fill_constant_keeps_case() {
    let p = Dataset::new(
        df!("name" => [Some("Ann"), None, Some("Bo")], "age" => [None, Some(30i64), Some(41)])
            .unwrap(),
    );
    let (session, server) = session_with_datasets(vec![("p", p)]);
    let r = call(&server, &session, "munge_fill_na",
        json!({"dataset": "p", "columns": ["name"], "strategy": "constant", "value": "Unknown", "result_name": "p1"})).await;
    assert!(r.success, "{}", text(&r));
    assert_eq!(
        labels(&dataset(&session, "p1").await, "name"),
        ["Ann", "Unknown", "Bo"]
    );
    let r = call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "p", "strategy": "Constant", "value": "Unknown", "result_name": "p2"}),
    )
    .await;
    let t = text(&r);
    assert!(
        r.success
            && t.contains("name (1, with 'Unknown')")
            && t.contains("skipped 1 column: age ("),
        "{t}"
    );
    let p2 = dataset(&session, "p2").await;
    assert_eq!(
        (
            labels(&p2, "name")[1].as_str(),
            p2.df().column("age").unwrap().null_count()
        ),
        ("Unknown", 1)
    );
    // the old form (the constant in `strategy`) is refused with directions; nothing is written
    let r = call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "p", "columns": ["name"], "strategy": "Unknown", "result_name": "p3"}),
    )
    .await;
    let t = text(&r);
    assert!(
        !r.success
            && t.contains("unknown fill strategy 'Unknown'")
            && t.contains("use strategy 'constant' and put the fill value in 'value'"),
        "{t}"
    );
    assert!(session.datasets.read().await.get("p3").is_none());
    // the cleaning preview keeps the case too
    let t = text(&call(&server, &session, "preview_cleaning",
        json!({"dataset": "p", "operation": "fill_na", "columns": ["name"], "strategy": "constant", "value": "Unknown"})).await);
    assert!(t.contains("| 1 | name | null | Unknown |"), "{t}");
}

/// LLM-12: the cleaning tools mapped every strategy they did not know ('mode', 'Mean', 'interpolate', a
/// typo, or none) to the constant "0"; munge_fill_na wrote unknown strategy names into text cells.
/// Keywords are case-insensitive; anything else is a tool error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_fill_unknown_strategy_is_error() {
    const UNKNOWN: &str = "unknown fill strategy 'interpolate'; expected one of: mean, median, mode, forward, backward, zero, constant";
    let q = Dataset::new(
        df!("bracket" => [Some("a"), None, Some("b")], "income" => [Some(1.5), None, Some(2.5)])
            .unwrap(),
    );
    let (session, server) = session_with_datasets(vec![("q", q)]);
    let r = call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "q", "columns": ["bracket"], "strategy": "interpolate"}),
    )
    .await;
    assert!(!r.success && text(&r).contains(UNKNOWN), "{}", text(&r)); // before: "interpolate" written, success
    assert_eq!(
        dataset(&session, "q")
            .await
            .df()
            .column("bracket")
            .unwrap()
            .null_count(),
        1
    );
    let r = call(
        &server,
        &session,
        "preview_cleaning",
        json!({"dataset": "q", "operation": "fill_na", "strategy": "interpolate"}),
    )
    .await;
    assert!(!r.success && text(&r).contains(UNKNOWN), "{}", text(&r));
    let sid = session_id(&text(
        &call(
            &server,
            &session,
            "cleaning_session_start",
            json!({"dataset": "q"}),
        )
        .await,
    ));
    let r = call(&server, &session, "cleaning_session_apply",
        json!({"session_id": sid, "operation": "fill_na", "columns": ["income"], "strategy": "interpolate"})).await;
    assert!(!r.success && text(&r).contains(UNKNOWN), "{}", text(&r)); // before: income filled with 0
    // R: mean(c(1.5, 2.5)) = 2 ('Mean' was the constant "0" before)
    let r = call(&server, &session, "cleaning_session_apply",
        json!({"session_id": sid, "operation": "fill_na", "columns": ["income"], "strategy": "Mean"})).await;
    assert!(
        r.success && text(&r).contains("income (1, with 2)"),
        "{}",
        text(&r)
    );
    let r = call(
        &server,
        &session,
        "cleaning_session_apply",
        json!({"session_id": sid, "operation": "fill_na", "columns": ["bracket"]}),
    )
    .await;
    assert!(!r.success && text(&r).contains("fill_na needs a strategy (mean | median | mode | forward | backward | zero | constant) or a fill value in 'value'"), "{}", text(&r));
    let r = call(
        &server,
        &session,
        "cleaning_session_apply",
        json!({"session_id": sid, "operation": "fill_na", "columns": ["bracket"], "value": "z"}),
    )
    .await;
    assert!(
        r.success && text(&r).contains("bracket (1, with 'z')"),
        "{}",
        text(&r)
    );
    // suggest_cleaning's value for a computed strategy is a reference, not an argument
    let t = text(
        &call(
            &server,
            &session,
            "suggest_cleaning",
            json!({"dataset": "q"}),
        )
        .await,
    );
    assert!(
        t.contains("(shown for reference; pass only strategy 'median' to fill_na)"),
        "{t}"
    );
}

/// DATA-65: normalize = true returned counts (the field was never read), and values were printed with
/// Debug (`Int64(2)`). R: dplyr::count(d, g, sort = TRUE) with the NA group and n / sum(n)
/// (data_munging_t3.R, case data_munging_value_counts, T3-WP50). (The architect's
/// test_mcp_value_counts_normalize, named test_validate_* because it is R-pinned.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_value_counts_normalize() {
    let case = Case::load("data_munging_value_counts");
    let (session, server) = session_with("data_munging_value_counts"); // dataset "munging_counts"
    let r = call(
        &server,
        &session,
        "munge_value_counts",
        json!({"dataset": "munging_counts", "column": "g", "normalize": true, "result_name": "p"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let p = dataset(&session, "p").await;
    let names: Vec<String> = p
        .df()
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .collect();
    assert_eq!(names, ["g", "proportion"]); // before: ["g", "count"]
    Check::new(&case)
        .vector("vc_g", &nums(&p, "g"), Tol::exact())
        .vector("vc_proportion", &nums(&p, "proportion"), Tol::exact())
        .finish();
    for line in [
        "  2: 0.4 (40.0%)\n",
        "  7: 0.3 (30.0%)\n",
        "  NA (missing): 0.2 (20.0%)\n",
        "  5: 0.1 (10.0%)\n",
    ] {
        assert!(t.contains(line), "{line:?} missing:\n{t}");
    }
    assert!(
        t.contains("Full result saved as 'p' (columns: g, proportion)"),
        "{t}"
    );
    let r = call(
        &server,
        &session,
        "munge_value_counts",
        json!({"dataset": "munging_counts", "column": "g", "result_name": "c"}),
    )
    .await;
    let t = text(&r);
    Check::new(&case)
        .vector(
            "vc_n",
            &nums(&dataset(&session, "c").await, "count"),
            Tol::exact(),
        )
        .finish();
    assert!(
        t.contains("  2: 4\n") && t.contains("  NA (missing): 2\n") && !t.contains("Int64("),
        "{t}"
    );
}
