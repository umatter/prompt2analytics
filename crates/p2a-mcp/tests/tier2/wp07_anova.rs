//! T2-WP07 MCP-path tests (ANOVA, MANOVA, Wilcoxon and rank tests). Owned by T2-WP07.
//!
//! The ANOVA tools through `call_tool_with_session`, on the R cases of
//! validation/r/cases/stats_anova.R (unbalanced and incomplete layouts, missing responses and
//! labels, Int64 columns).

use crate::common::*;
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

/// The numbers on the line whose first whitespace token is `label` (significance stars and
/// other words skipped). Panics with the text when no such line exists.
fn row(text: &str, label: &str) -> Vec<f64> {
    let line = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(label))
        .unwrap_or_else(|| panic!("no line starting with {label:?} in:\n{text}"));
    line.split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

#[tokio::test]
async fn test_validate_mcp_anova_two_way_unbalanced() {
    let case = Case::load("stat_anova_two_way_unbalanced");
    let (session, server) = session_with("stat_anova_two_way_unbalanced");
    let r = call(
        &server,
        &session,
        "anova_two_way",
        json!({"dataset": "anova_two_way_unbalanced", "response": "y",
               "factor_a": "fa", "factor_b": "fb"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let (p4, x) = (Tol::printed(4), Tol::exact());
    let mut check = Check::new(&case);
    for (label, key) in [("fa", "int_fa"), ("fb", "int_fb"), ("fa:fb", "int_fa_x_fb")] {
        let v = row(&t, label);
        assert_eq!(v.len(), 5, "{label}: {v:?}\n{t}");
        check
            .scalar(&format!("{key}_ss"), v[0], p4)
            .scalar(&format!("{key}_df"), v[1], x)
            .scalar(&format!("{key}_f"), v[3], p4)
            .scalar(&format!("{key}_p"), v[4], p4);
    }
    let resid = row(&t, "Residuals");
    check
        .scalar("int_resid_ss", resid[0], p4)
        .scalar("int_resid_df", resid[1], x)
        .finish();
    assert!(t.contains("Sums of squares: Type I"), "{t}");
    assert!(t.contains("Missing values: 1 of 13 rows dropped"), "{t}");
    assert!(t.contains("Levels: 2 × 2"), "{t}");
}
