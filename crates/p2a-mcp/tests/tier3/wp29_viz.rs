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
/// has no values is refused in-band; beside a column with values it is drawn with n = 0 and its rows
/// reported as dropped (`allna_n`; `inf6_*` are pinned by the core test: a CSV cannot carry the Inf
/// through R's reader).
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
            // Beside a column that has values, the empty column is drawn as n = 0 and its rows
            // are reported as dropped.
            let e = call(
                &server,
                &session,
                "munge_one_hot_encode",
                json!({"dataset": g, "column": "group", "result_name": "allna_wide"}),
            )
            .await;
            assert!(e.success, "{}", text(&e));
            let r = call(
                &server,
                &session,
                "viz_boxplot",
                json!({"dataset": "allna_wide", "columns": ["value", "group_allna"]}),
            )
            .await;
            let t = text(&r);
            assert!(r.success, "{t}");
            assert!(t.contains("value (n = 0): no values"), "{t}");
            assert!(
                t.contains(
                    "Missing values: 2 of 2 rows dropped because of missing values (value: 2)"
                ),
                "{t}"
            );
            let n: f64 = number_after(&t, "value (n = ");
            check.scalar("allna_n", n, Tol::exact());
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
        // printed with 7 significant digits (format_significant)
        let seven = Tol::new(5e-7, 1e-12);
        check.vector(&format!("{g}_stats"), &stats, seven);
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
            check.vector(&format!("{g}_out"), &out, seven);
        }
    }
    check.finish();
}
