//! T2-WP13 MCP-path tests (mediation, weighting/matching, TMLE, gsynth). Owned by T2-WP13.

use crate::common::{call, session_with, session_with_datasets, text};
use p2a_core::data::{DataLoader, Dataset};
use p2a_validation::{Case, Check, Tol, text::labelled};
use serde_json::json;

/// The numbers on the first line of `text` that contains `needle`, tokenised on
/// whitespace with `(`, `)` and `,` stripped ("0.6132 (SE = 0.0594)" gives 0.6132 and
/// 0.0594).
fn numbers_on_line(text: &str, needle: &str) -> Vec<f64> {
    let line = text
        .lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing {needle:?} in:\n{text}"));
    line.split_whitespace()
        .filter_map(|tok| {
            tok.trim_matches(|c| c == '(' || c == ')' || c == ',')
                .parse::<f64>()
                .ok()
        })
        .collect()
}

const MISSING_LINE: &str =
    "Missing values: 50 of 400 rows dropped because of missing values (y: 15, m: 25, x1: 10)";

#[tokio::test]
async fn test_validate_mcp_mediation_analysis_causal_na() {
    let case = Case::load("causal_mediation_na");
    let (session, server) = session_with("causal_mediation_na");
    let res = call(
        &server,
        &session,
        "mediation_analysis",
        json!({"dataset": "causal_na", "outcome": "y", "treatment": "d", "mediator": "m",
               "covariates": ["x1", "x2"], "trim": 0.05, "bootstrap": 20}),
    )
    .await;
    let t = text(&res);
    assert!(res.success, "{t}");
    assert!(t.contains("N = 280 (trimmed: 70)"), "{t}");
    assert!(t.contains(MISSING_LINE), "{t}");
    let total = labelled(&t, "Total (ATE)").unwrap_or_else(|| panic!("no total in:\n{t}"));
    Check::new(&case)
        .scalar("ipw_total_trim05", total, Tol::printed(4))
        .finish();
}

#[tokio::test]
async fn test_validate_mcp_natural_effects_mediation_causal_na() {
    let case = Case::load("causal_mediation_na");
    let (session, server) = session_with("causal_mediation_na");
    let res = call(
        &server,
        &session,
        "natural_effects_mediation",
        json!({"dataset": "causal_na", "outcome": "y", "treatment": "d", "mediator": "m",
               "confounders": ["x1", "x2"], "allow_interaction": false, "n_bootstrap": 0}),
    )
    .await;
    let t = text(&res);
    assert!(res.success, "{t}");
    assert!(t.contains("N = 350 (treated: 197, control: 153)"), "{t}");
    assert!(t.contains(MISSING_LINE), "{t}");
    let nums = numbers_on_line(&t, "Treatment -> Mediator:");
    assert_eq!(nums.len(), 2, "{nums:?}");
    Check::new(&case)
        .named_subset("med_coef", &["d"], &["d"], &nums[..1], Tol::printed(4))
        .named_subset("med_se", &["d"], &["d"], &nums[1..], Tol::printed(4))
        .finish();
}

#[tokio::test]
async fn test_validate_mcp_treatment_ipw_null_outcome_is_named() {
    let (session, server) = session_with("causal_mediation_na");
    let res = call(
        &server,
        &session,
        "treatment_ipw",
        json!({"dataset": "causal_na", "outcome": "y", "treatment": "d",
               "covariates": ["x2"], "bootstrap": 0}),
    )
    .await;
    let t = text(&res);
    assert!(!res.success, "{t}");
    assert!(t.contains("Column 'y' contains 15 missing value(s)"), "{t}");
    assert!(!t.contains("not found"), "{t}");
}

#[tokio::test]
async fn test_mcp_gsynth_unbalanced_panel_is_explained() {
    // units C1-C4 (never treated), T1 (treated from t = 7), T2 (from t = 8); (C2, 5) absent
    let mut csv = String::from("unit,time,y,D\n");
    let units = ["C1", "C2", "C3", "C4", "T1", "T2"];
    for (i, unit) in units.iter().enumerate() {
        let u = (i + 1) as f64;
        for t in 1..=10 {
            if *unit == "C2" && t == 5 {
                continue;
            }
            let d = match *unit {
                "T1" => u8::from(t >= 7),
                "T2" => u8::from(t >= 8),
                _ => 0,
            };
            let tf = f64::from(t);
            let y = 10.0 + tf + 0.5 * u + 0.3 * (1.3 * (10.0 * u + tf)).sin() + 3.0 * f64::from(d);
            csv.push_str(&format!("{unit},{t},{y},{d}\n"));
        }
    }
    let df = DataLoader::from_csv_string(&csv).unwrap();
    let (session, server) = session_with_datasets(vec![("panel", Dataset::new(df))]);
    let res = call(
        &server,
        &session,
        "gsynth",
        json!({"dataset": "panel", "outcome": "y", "treatment": "D", "unit_col": "unit",
               "time_col": "time", "n_factors": 1}),
    )
    .await;
    let t = text(&res);
    assert!(!res.success, "{t}");
    assert!(
        t.contains(
            "gsynth needs a balanced panel: 1 (unit, time) cell(s) are missing \
             (first: unit 'C2', time 5)"
        ),
        "{t}"
    );
}
