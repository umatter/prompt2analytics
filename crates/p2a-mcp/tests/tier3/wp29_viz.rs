//! T4-WP29 MCP-path tests: chart tools (box-plot statistics against R). Owned by T4-WP29.
use crate::common::*;
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

/// The numbers after `label` up to the next comma or line end ("Lower whisker: 1.0000, ...").
fn number_after(block: &str, label: &str) -> f64 {
    let rest = &block[block
        .find(label)
        .unwrap_or_else(|| panic!("no {label:?} in {block}"))
        + label.len()..];
    let end = rest.find([',', '\n', ')']).unwrap_or(rest.len());
    rest[..end].trim().parse().unwrap()
}

/// viz_boxplot per group of case `viz_boxplot_groups` (grDevices::boxplot.stats, coef = 1.5): whiskers,
/// hinges and median at the four decimals printed, n and the outliers. A box plot whose only column
/// has no values is refused in-band (the case's `allna_*` values are pinned by the core test, as are
/// `inf6_*`: a CSV cannot carry the Inf through R's reader).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_boxplot_groups() {
    let case = Case::load("viz_boxplot_groups");
    let (session, server) = session_with("viz_boxplot_groups");
    let mut check = Check::new(&case);
    for g in ["data62", "n4", "n12", "n7", "norm37", "const3", "allna"] {
        let f = call(&server, &session, "munge_filter",
            json!({"dataset": "viz_boxplot_groups", "column": "group", "op": "eq", "value": g, "result_name": g}))
        .await;
        assert!(f.success, "{}", text(&f));
        let r = call(
            &server,
            &session,
            "viz_boxplot",
            json!({"dataset": g, "columns": ["value"]}),
        )
        .await;
        let t = text(&r);
        if g == "allna" {
            assert!(!r.success && t.contains("No valid data"), "{t}");
            continue;
        }
        assert!(r.success, "{g}: {t}");
        let n: f64 = number_after(&t, "value (n = ");
        check.scalar(&format!("{g}_n"), n, Tol::exact());
        let stats = [
            number_after(&t, "Lower whisker: "),
            number_after(&t, "Q1 (lower hinge): "),
            number_after(&t, "Median: "),
            number_after(&t, "Q3 (upper hinge): "),
            number_after(&t, "Upper whisker: "),
        ];
        check.vector(&format!("{g}_stats"), &stats, Tol::printed(4));
        let outliers = t
            .lines()
            .find_map(|l| l.trim().strip_prefix("Outliers: "))
            .unwrap_or_else(|| panic!("{t}"));
        let out: Vec<f64> = if outliers == "none" {
            Vec::new()
        } else {
            outliers.split(", ").map(|v| v.parse().unwrap()).collect()
        };
        check.scalar(&format!("{g}_n_out"), out.len() as f64, Tol::exact());
        if !out.is_empty() {
            check.vector(&format!("{g}_out"), &out, Tol::printed(4));
        }
    }
    check.finish();
}
