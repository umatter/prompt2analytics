//! T2-WP12 MCP-path tests (Callaway-Sant'Anna and the DiD family). Owned by T2-WP12.

use crate::common::{call, session_with, session_with_datasets, text};
use p2a_core::data::{DataLoader, Dataset};
use p2a_core::polars::prelude::{Column, DataType};
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

/// The first line of `t` containing `needle`; panics with the text otherwise.
fn line_with<'a>(t: &'a str, needle: &str) -> &'a str {
    t.lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing {needle:?} in:\n{t}"))
}

/// The number right after `marker` in `line`, up to the first character that is not
/// part of a float.
fn number_after(line: &str, marker: &str) -> f64 {
    let start = line
        .find(marker)
        .unwrap_or_else(|| panic!("no {marker:?} in {line:?}"))
        + marker.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|e| panic!("{:?} after {marker:?} in {line:?}: {e}", &rest[..end]))
}

fn args(dataset: &str, comparison: &str, bootstrap: u64) -> serde_json::Value {
    json!({"dataset": dataset, "outcome": "y", "treatment_time": "first_treat", "time_col": "time",
           "unit_col": "unit", "comparison_group": comparison, "bootstrap": bootstrap})
}

/// The text shows "ATT = 2.5335 (SE = 0.2034, 95% CI [2.1349, 2.9321])" and
/// "Chi-squared(9) = 7.26, p-value = 0.6105" (R: sprintf of the case values).
#[tokio::test]
async fn test_validate_mcp_staggered_did_analytic() {
    let case = Case::load("did_cs_analytic");
    let (session, server) = session_with("did_cs_analytic");
    let res = call(
        &server,
        &session,
        "staggered_did",
        args("cs_panel", "never_treated", 0),
    )
    .await;
    let t = text(&res);
    assert!(res.success, "{t}");
    let ov = line_with(&t, "ATT = ");
    let pt = line_with(&t, "Chi-squared(");
    Check::new(&case)
        .scalar(
            "never_simple_att",
            number_after(ov, "ATT = "),
            Tol::printed(4),
        )
        .scalar(
            "never_simple_se",
            number_after(ov, "SE = "),
            Tol::printed(4),
        )
        .scalar(
            "never_pretrend_w",
            number_after(pt, ") = "),
            Tol::printed(2),
        )
        .scalar(
            "never_pretrend_p",
            number_after(pt, "p-value = "),
            Tol::printed(4),
        )
        .finish();
    assert!(t.contains("Standard Errors:    analytic"), "{t}");
    assert!(t.contains("Chi-squared(9) = "), "{t}");
}

#[tokio::test]
async fn test_validate_mcp_staggered_did_string_unit_ids() {
    let case = Case::load("did_cs_analytic");
    let mut df = DataLoader::load_csv(case.dataset_path()).unwrap();
    let units = df.column("unit").unwrap().cast(&DataType::Int64).unwrap();
    let labels: Vec<String> = units
        .i64()
        .unwrap()
        .iter()
        .map(|u| format!("state_{}", u.unwrap()))
        .collect();
    df.with_column(Column::new("unit".into(), labels)).unwrap();
    let (session, server) = session_with_datasets(vec![("cs_panel_str", Dataset::new(df))]);
    let res = call(
        &server,
        &session,
        "staggered_did",
        args("cs_panel_str", "not_yet_treated", 0),
    )
    .await;
    let t = text(&res);
    assert!(res.success, "{t}");
    let ov = line_with(&t, "ATT = ");
    let pt = line_with(&t, "Chi-squared(");
    Check::new(&case)
        .scalar(
            "notyet_simple_att",
            number_after(ov, "ATT = "),
            Tol::printed(4),
        )
        .scalar(
            "notyet_simple_se",
            number_after(ov, "SE = "),
            Tol::printed(4),
        )
        .scalar(
            "notyet_pretrend_w",
            number_after(pt, ") = "),
            Tol::printed(2),
        )
        .finish();
}

#[tokio::test]
async fn test_mcp_staggered_did_rejects_small_bootstrap() {
    let (session, server) = session_with("did_cs_analytic");
    let res = call(
        &server,
        &session,
        "staggered_did",
        args("cs_panel", "never_treated", 20),
    )
    .await;
    let t = text(&res);
    assert!(!res.success, "{t}");
    let err = res.error.as_deref().unwrap_or_default();
    assert!(err.contains("at least 50"), "{err}");
    assert!(err.contains("0 for analytic"), "{err}");
}
