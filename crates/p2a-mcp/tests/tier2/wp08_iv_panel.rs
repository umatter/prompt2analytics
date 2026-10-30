//! T2-WP08 MCP-path tests (IV first stage and panel extraction). Owned by T2-WP08.
//!
//! Each test runs one tool on an R case's dataset through the dispatcher and compares
//! the printed numbers with the case at print precision (`Tol::printed(4)`).

use p2a_validation::{Case, Check, Tol};
use serde_json::json;

use crate::common::{call, session_with, text};

/// The number after `prefix` on the first line whose trimmed start begins with `prefix`
/// (up to white space or a comma).
fn number_after(text: &str, prefix: &str) -> f64 {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting with {prefix:?} in:\n{text}"));
    let rest = line[prefix.len()..].trim_start();
    let token: String = rest
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != ',')
        .collect();
    token
        .parse()
        .unwrap_or_else(|e| panic!("{token:?} after {prefix:?} in {line:?}: {e}"))
}

/// The numbers after the first white-space token `name` of a table row, with trailing
/// non-numeric characters (significance stars, the dagger) dropped from each token.
fn row(text: &str, name: &str) -> Vec<f64> {
    let line = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{text}"));
    line.split_whitespace()
        .skip(1)
        .map(|t| {
            let t = t.trim_end_matches(|c: char| !c.is_ascii_digit());
            t.parse()
                .unwrap_or_else(|e| panic!("{t:?} in row {line:?}: {e}"))
        })
        .collect()
}

/// The last non-empty line of a text.
fn last_line(text: &str) -> &str {
    text.lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
}

#[tokio::test]
async fn test_validate_mcp_iv_2sls_weak_controls() {
    let case = Case::load("iv_weak_controls");
    let (session, server) = session_with("iv_weak_controls");
    let args = json!({
        "dataset": "iv_weak_controls", "y": "y", "x_exog": ["w"], "x_endog": ["x"],
        "instruments": ["z1", "z2"], "robust": false
    });
    let r = call(&server, &session, "iv_2sls", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(
        t.contains("First stage (partial F of the excluded instruments):"),
        "{t}"
    );
    assert!(t.contains("on (2, 496) DF"), "{t}");
    assert!(t.contains("[CRITICAL]") && t.contains("partial F"), "{t}");
    assert!(!t.contains("First-stage F:"), "{t}");
    let classical = number_after(&t, "x: F =");

    // robust omitted: the default (true) reports the HC1 Wald F
    let args = json!({
        "dataset": "iv_weak_controls", "y": "y", "x_exog": ["w"], "x_endog": ["x"],
        "instruments": ["z1", "z2"]
    });
    let r = call(&server, &session, "iv_2sls", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(t.contains("HC1 Wald F of the excluded instruments"), "{t}");
    Check::new(&case)
        .vector("weak_f", &[classical], Tol::printed(4))
        .vector("weak_f_hc1", &[number_after(&t, "x: F =")], Tol::printed(4))
        .finish();
}

#[tokio::test]
async fn test_validate_mcp_iv_first_stage_weak_controls() {
    let case = Case::load("iv_weak_controls");
    let (session, server) = session_with("iv_weak_controls");
    let args = json!({
        "dataset": "iv_weak_controls", "endogenous_var": "x",
        "instruments": ["z1", "z2"], "controls": ["w"]
    });
    let r = call(&server, &session, "iv_first_stage", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    Check::new(&case)
        .scalar(
            "fs_partial_f",
            number_after(&t, "F-statistic:"),
            Tol::printed(4),
        )
        .finish();
    assert!(t.contains("Controls: w"), "{t}");
    assert!(t.contains("WEAK (F < 5)"), "{t}");
}

#[tokio::test]
async fn test_validate_mcp_iv_sargan_na_rows() {
    let case = Case::load("iv_na_rows");
    let (session, server) = session_with("iv_na_rows");
    let args = json!({
        "dataset": "iv_na_rows", "y": "y", "x_exog": ["w", "k"], "x_endog": ["x"],
        "instruments": ["z1", "z2"]
    });
    let r = call(&server, &session, "iv_sargan_test", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    Check::new(&case)
        .scalar("sargan", number_after(&t, "J statistic:"), Tol::printed(4))
        .finish();
    assert_eq!(
        last_line(&t),
        "Missing values: 6 of 300 rows dropped because of missing values \
         (y: 2, w: 1, k: 1, x: 1, z1: 1, z2: 1)",
        "{t}"
    );
}

#[tokio::test]
async fn test_validate_mcp_panel_fixed_effects_grunfeld_na() {
    let case = Case::load("panel_grunfeld_na");
    let (session, server) = session_with("panel_grunfeld_na");
    let args = json!({
        "dataset": "grunfeld_na", "y": "inv", "x": ["value", "capital"], "entity_var": "firm_id"
    });
    let r = call(&server, &session, "panel_fixed_effects", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let names = ["value", "capital"];
    let coef: Vec<f64> = names.iter().map(|n| row(&t, n)[0]).collect();
    let se: Vec<f64> = names.iter().map(|n| row(&t, n)[1]).collect();
    Check::new(&case)
        .named("fe_coef", &names, &coef, Tol::printed(4))
        .named("fe_se", &names, &se, Tol::printed(4))
        .finish();
    assert!(t.contains("No. Observations: 195"), "{t}");
    assert!(t.contains("No. Groups: 10"), "{t}");
    assert_eq!(
        last_line(&t),
        "Missing values: 5 of 200 rows dropped because of missing values \
         (inv: 2, value: 1, firm_id: 2)",
        "{t}"
    );
}

#[tokio::test]
async fn test_validate_mcp_panel_hdfe_grunfeld_na() {
    let case = Case::load("panel_grunfeld_na");
    let (session, server) = session_with("panel_grunfeld_na");
    let args = json!({
        "dataset": "grunfeld_na", "y": "inv", "x": ["value", "capital"],
        "fe": ["firm_id", "year"], "se_type": "standard"
    });
    let r = call(&server, &session, "panel_hdfe", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let names = ["value", "capital"];
    let coef: Vec<f64> = names.iter().map(|n| row(&t, n)[0]).collect();
    let se: Vec<f64> = names.iter().map(|n| row(&t, n)[1]).collect();
    Check::new(&case)
        .named("hdfe_coef", &names, &coef, Tol::printed(4))
        .named("hdfe_se_iid", &names, &se, Tol::printed(4))
        .finish();
    assert!(t.contains("No. Observations: 193"), "{t}");
    assert!(t.contains("firm_id: 10 levels"), "{t}");
    assert!(t.contains("year: 20 levels"), "{t}");
    assert_eq!(
        last_line(&t),
        "Missing values: 7 of 200 rows dropped because of missing values \
         (inv: 2, value: 1, firm_id: 2, year: 2)",
        "{t}"
    );
}

#[tokio::test]
async fn test_validate_mcp_panel_pmg_grunfeld_na() {
    let case = Case::load("panel_grunfeld_na");
    let (session, server) = session_with("panel_grunfeld_na");
    let args = json!({
        "dataset": "grunfeld_na", "y": "inv", "x": ["value", "capital"], "entity_var": "firm_id"
    });
    let r = call(&server, &session, "panel_pmg", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    Check::new(&case)
        .scalar(
            "n_complete_fe",
            number_after(&t, "No. Observations:"),
            Tol::exact(),
        )
        .scalar("n_groups", number_after(&t, "No. Entities:"), Tol::exact())
        .finish();
    assert!(
        t.contains(
            "Missing values: 5 of 200 rows dropped because of missing values \
             (inv: 2, value: 1, firm_id: 2)"
        ),
        "{t}"
    );
}
