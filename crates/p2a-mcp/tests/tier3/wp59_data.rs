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

/// T3 design 5.1 (the MCP half of DATA-9, T3-WP50): new_value was a required string, so the documented
/// `null` was refused before the handler ran. R: dplyr::na_if (data_munging_t3.R, case
/// data_munging_na_if, T3-WP50).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_str_replace_value_null() {
    let case = Case::load("data_munging_na_if");
    let (session, server) = session_with("data_munging_na_if"); // dataset "munging_fill"
    let r = call(&server, &session, "str_replace_value",
        json!({"dataset": "munging_fill", "column": "k", "old_value": "3", "new_value": null, "result_name": "r1"})).await;
    assert!(
        r.success && text(&r).contains("Replaced 2 values '3' with missing in column 'k'"),
        "{}",
        text(&r)
    );
    // new_value omitted: missing too
    let r = call(
        &server,
        &session,
        "str_replace_value",
        json!({"dataset": "r1", "column": "f", "old_value": "1.25", "result_name": "r2"}),
    )
    .await;
    assert!(
        r.success && text(&r).contains("Replaced 2 values '1.25' with missing in column 'f'"),
        "{}",
        text(&r)
    );
    let r = call(&server, &session, "str_replace_value",
        json!({"dataset": "r2", "column": "s", "old_value": "c", "new_value": null, "result_name": "r3"})).await;
    assert!(r.success, "{}", text(&r));
    let r3 = dataset(&session, "r3").await;
    Check::new(&case)
        .vector("k_na_if_3", &nums(&r3, "k"), Tol::exact())
        .vector("f_na_if_1_25", &nums(&r3, "f"), Tol::exact())
        .finish();
    assert_eq!(labels(&r3, "s"), case.names("s_na_if_c"));
    // a text replacement is still literal
    let r = call(&server, &session, "str_replace_value",
        json!({"dataset": "munging_fill", "column": "s", "old_value": "d", "new_value": "D", "result_name": "r4"})).await;
    assert!(r.success, "{}", text(&r));
    assert_eq!(labels(&dataset(&session, "r4").await, "s")[8], "D");
}

/// The parts of a describe_dataset line: "missing" (the nulls count), "n", "distinct", "top" and every
/// "key = value" statistic.
fn describe_line(t: &str, col: &str) -> std::collections::HashMap<String, String> {
    let prefix = format!("- {col} (");
    let line = t
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no line for {col}:\n{t}"));
    let (_, rest) = line.split_once("): ").expect("name (dtype): ...");
    let mut parts = rest.split("; ");
    let mut m = std::collections::HashMap::new();
    m.insert(
        "missing".to_string(),
        parts
            .next()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .to_string(),
    );
    for p in parts {
        if let Some(top) = p.strip_prefix("top: ") {
            m.insert("top".into(), top.into());
        } else if let Some(d) = p.strip_suffix(" distinct") {
            m.insert("distinct".into(), d.into());
        } else {
            for kv in p.split(", ") {
                let (k, v) = kv
                    .split_once(" = ")
                    .unwrap_or_else(|| panic!("{kv:?} in {line}"));
                m.insert(k.into(), v.into());
            }
        }
    }
    m
}
fn stat(m: &std::collections::HashMap<String, String>, k: &str) -> f64 {
    match m.get(k).map(String::as_str) {
        Some("NA") => f64::NAN,
        Some(v) => v.parse().unwrap(),
        None => panic!("no {k} in {m:?}"),
    }
}
/// "'east' (4), 'west' (4), ..., and 2 more" -> (labels, counts)
fn top_levels(m: &std::collections::HashMap<String, String>) -> (Vec<String>, Vec<f64>) {
    m["top"]
        .split(", ")
        .filter(|e| !e.starts_with("and "))
        .map(|e| {
            let (label, count) = e.rsplit_once(" (").unwrap();
            (
                label.trim_matches('\'').to_string(),
                count.trim_end_matches(')').parse::<f64>().unwrap(),
            )
        })
        .unzip()
}

/// LLM-6: describe_dataset promised summary statistics (and the system prompt sends the model to it for a
/// mean) but printed only types and null counts. R: summary(), sd(), table() (mcp_data_t3.R, case
/// data_describe_columns); the text has 7 significant digits (relative 5e-7). (The architect's
/// test_mcp_describe_dataset_statistics, named test_validate_* because it is R-pinned.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_describe_dataset_statistics() {
    let case = Case::load("data_describe_columns");
    let (session, server) = session_with("data_describe_columns"); // dataset "describe_columns"
    let r = call(
        &server,
        &session,
        "describe_dataset",
        json!({"dataset": "describe_columns"}),
    )
    .await;
    let t = text(&r);
    assert!(
        r.success && t.contains("Dimensions: 15 rows x 7 columns"),
        "{t}"
    );
    let names = [
        "n", "missing", "mean", "sd", "min", "q1", "median", "q3", "max",
    ];
    let mut check = Check::new(&case);
    for col in ["id", "income", "age"] {
        let m = describe_line(&t, col);
        let v: Vec<f64> = names.iter().map(|k| stat(&m, k)).collect(); // before: no "mean" (panics)
        check
            .named_subset(col, &["n", "missing"], &names, &v, Tol::exact())
            .named_subset(
                col,
                &["mean", "sd", "min", "q1", "median", "q3", "max"],
                &names,
                &v,
                Tol::new(5e-7, 0.0),
            ); // 7 significant digits
    }
    for col in ["region", "grade", "member", "empty"] {
        let m = describe_line(&t, col);
        check.named(
            &format!("{col}_counts"),
            &["n", "missing", "distinct"],
            &[stat(&m, "n"), stat(&m, "missing"), stat(&m, "distinct")],
            Tol::exact(),
        );
        if col != "empty" {
            let (l, c) = top_levels(&m);
            assert_eq!(l, case.names(&format!("{col}_top")), "{col}");
            check.named(&format!("{col}_top"), &l, &c, Tol::exact());
        }
    }
    check.finish();
    assert!(t.contains("  - grade (str): 0 nulls (0.0%); n = 15; 7 distinct; top: 'A' (3), 'B' (3), 'C' (2), 'D' (2), 'E' (2), and 2 more"), "{t}");
    assert!(
        t.contains(
            "  - member (bool): 1 nulls (6.7%); n = 14; 2 distinct; top: TRUE (9), FALSE (5)"
        ),
        "{t}"
    );
}

/// DATA-25: viz_dendrogram cast JSON numbers with `as usize` (negative and NaN became 0, 0.5 became 0) and
/// dropped short rows, so malformed linkages drew a wrong tree and reported success. (The out-of-range id
/// panic of the finding was fixed by T0-WP20, 812a148; its error is checked here as a regression guard.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_dendrogram_bad_linkage() {
    let (session, server) = session_with_datasets(vec![]);
    for (args, needle) in [
        (
            json!({"linkage_matrix": [[-1, 1, 1.0, 2]]}),
            "cluster1 -1 is negative",
        ),
        (
            json!({"linkage_matrix": [[0.5, 1, 1.0, 2]]}),
            "cluster1 0.5 is not a whole number",
        ),
        (
            json!({"linkage_matrix": [[0, 1, 1.0, 2], [2, 3]]}),
            "row 1 has 2 numbers",
        ),
        (
            json!({"linkage_matrix": [[0, 1, 1.0, 2, 9]]}),
            "row 0 has 5 numbers",
        ),
        (
            json!({"linkage_matrix": [[0, 1, 1.0, 1.5]]}),
            "size 1.5 must be a whole number",
        ),
        (
            json!({"linkage_matrix": [[5, 6, 1.0, 2]]}),
            "refers to cluster 5, which does not exist yet",
        ), // core, T0-WP20
        (
            json!({"linkage_matrix": [[0, 1, 1.0, 2], [2, 3, 1.5, 3]], "labels": ["a", "b"]}),
            "labels has 2 entries, but a linkage matrix with 2 rows has 3 samples",
        ),
    ] {
        let r = call(&server, &session, "viz_dendrogram", args.clone()).await;
        assert!(
            !r.success && text(&r).contains(needle),
            "{args}: {}",
            text(&r)
        );
    }
    let ok = call(
        &server,
        &session,
        "viz_dendrogram",
        json!({"linkage_matrix": [[0, 1, 1.0, 2], [2, 3, 1.5, 3]], "labels": ["a", "b", "c"]}),
    )
    .await;
    assert!(
        ok.success && text(&ok).contains("Samples: 3"),
        "{}",
        text(&ok)
    );
}

/// DATA-25: the documented conversion of R's hclust()$merge (negative = sample, 1-based merges) gives a
/// linkage the tool accepts, with R's sample count and maximum height; the raw R matrix is refused with
/// the conversion rule. R: hclust(dist(x)) (mcp_data_t3.R, case viz_dendrogram_hclust).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_dendrogram_hclust() {
    let case = Case::load("viz_dendrogram_hclust");
    let (session, server) = session_with("viz_dendrogram_hclust");
    let n = case.scalar("n");
    let conv = |m: f64| if m < 0.0 { -m - 1.0 } else { n + m - 1.0 };
    let (l, r, h, s) = (
        case.vector("merge_left"),
        case.vector("merge_right"),
        case.vector("height"),
        case.vector("size"),
    );
    let rows: Vec<[f64; 4]> = (0..h.len())
        .map(|i| [conv(l[i]), conv(r[i]), h[i], s[i]])
        .collect();
    let labels: Vec<String> = (1..=7usize).map(|i| i.to_string()).collect();
    let res = call(
        &server,
        &session,
        "viz_dendrogram",
        json!({"linkage_matrix": rows, "labels": labels}),
    )
    .await;
    let t = text(&res);
    assert!(
        res.success && t.contains(&format!("Merge steps: {}", h.len())),
        "{t}"
    );
    let samples: f64 = t
        .lines()
        .find_map(|x| x.strip_prefix("Samples: "))
        .unwrap()
        .parse()
        .unwrap();
    let max: f64 = t
        .lines()
        .find_map(|x| x.strip_prefix("Max distance: "))
        .unwrap()
        .parse()
        .unwrap();
    Check::new(&case)
        .scalar("n", samples, Tol::exact())
        .finish();
    assert!(Tol::printed(4).allows(max, h[h.len() - 1]), "{max}"); // "8.6740" vs 8.6739999999999995
    // the unconverted R matrix (negative samples, 1-based merges) is refused with the conversion rule
    let raw: Vec<[f64; 4]> = (0..h.len()).map(|i| [l[i], r[i], h[i], s[i]]).collect();
    let bad = call(
        &server,
        &session,
        "viz_dendrogram",
        json!({"linkage_matrix": raw}),
    )
    .await;
    assert!(
        !bad.success && text(&bad).contains("R's hclust()$merge"),
        "{}",
        text(&bad)
    );
}
