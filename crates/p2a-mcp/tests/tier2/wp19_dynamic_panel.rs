//! T2-WP19 MCP-path tests (Arellano-Bond GMM and panel unit-root tests). Owned by T2-WP19.
//!
//! The registered tests run a tool on an R case's dataset through the dispatcher and
//! compare the printed numbers with the case at print precision (`Tol::printed(4)`); the
//! others check refusals and warnings on the committed CSVs and load no case.

use p2a_core::data::{DataLoader, Dataset};
use p2a_validation::{Case, Check, Tol, repo_path};
use serde_json::{Value, json};

use crate::common::{call, session_with, session_with_datasets, text};

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

/// The number right after the first occurrence of `label` (e.g. `"chi2(20) = "`,
/// `"AR(2): z = "`), up to white space or a comma.
fn after(text: &str, label: &str) -> f64 {
    let at = text
        .find(label)
        .unwrap_or_else(|| panic!("no {label:?} in:\n{text}"));
    let token: String = text[at + label.len()..]
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != ',')
        .collect();
    token
        .parse()
        .unwrap_or_else(|e| panic!("{token:?} after {label:?}: {e}"))
}

/// The panel_gmm arguments for ab_balanced, with `extra` merged in.
fn gmm_args(dataset: &str, extra: Value) -> Value {
    let mut args = json!({
        "dataset": dataset, "y": "y", "x": ["x"], "entity_var": "id", "time_var": "year"
    });
    if let (Some(a), Some(e)) = (args.as_object_mut(), extra.as_object()) {
        a.extend(e.clone());
    }
    args
}

#[tokio::test]
async fn test_validate_mcp_panel_gmm_ab_balanced() {
    let case = Case::load("dpd_ab_balanced");
    let (session, server) = session_with("dpd_ab_balanced");
    let names = ["lag(y, 1)", "x"];
    let p4 = Tol::printed(4);
    let mut c = Check::new(&case);

    // Defaults: two-step, Windmeijer-corrected SEs.
    let r = call(
        &server,
        &session,
        "panel_gmm",
        gmm_args("ab_balanced", json!({})),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(
        t.contains("Std. errors: two-step, Windmeijer (2005) corrected"),
        "{t}"
    );
    let (lag, x) = (row(&t, "L1.y"), row(&t, "x"));
    c.named("twostep_coef", &names, &[lag[0], x[0]], p4)
        .named("twostep_se_robust", &names, &[lag[1], x[1]], p4)
        .scalar(
            "twostep_n_obs",
            after(&t, "No. Observations: "),
            Tol::exact(),
        )
        .scalar(
            "twostep_n_instruments_nonzero",
            after(&t, "No. Instruments: "),
            Tol::exact(),
        )
        .scalar("twostep_sargan", after(&t, "chi2(20) = "), p4)
        .scalar("twostep_ar1", after(&t, "AR(1): z = "), p4)
        .scalar("twostep_ar2", after(&t, "AR(2): z = "), p4);

    // One-step with Arellano-Bond (1991) classical SEs and the homoskedastic Sargan test.
    let args = gmm_args("ab_balanced", json!({"step": "onestep", "robust": false}));
    let r = call(&server, &session, "panel_gmm", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let (lag, x) = (row(&t, "L1.y"), row(&t, "x"));
    c.named("onestep_coef", &names, &[lag[0], x[0]], p4)
        .named("onestep_se_textbook", &names, &[lag[1], x[1]], p4)
        .scalar(
            "onestep_sargan_homoskedastic",
            after(&t, "one-step weights): chi2(20) = "),
            p4,
        );
    assert!(
        t.contains("Sargan test (homoskedastic, one-step weights): chi2(20) = 28.2550"),
        "{t}"
    );
    c.finish();
}

/// Text unit ids, a year no unit observes and missing cells. Before: "Panel unit root
/// test failed: Empty dataset".
#[tokio::test]
async fn test_validate_mcp_panel_unit_root_text_ids() {
    let case = Case::load("panel_pur_extraction");
    let (session, server) = session_with("panel_pur_extraction");
    let args = json!({
        "dataset": "ab_common_gap", "variable": "y", "unit_col": "unit", "time_col": "year",
        "test": "ips", "lags": 1
    });
    let r = call(&server, &session, "panel_unit_root", args).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    for needle in [
        "N panels:   100",
        "Warnings:",
        "have gaps in time",
        "10 of 700 rows dropped because of missing values (y: 10)",
    ] {
        assert!(t.contains(needle), "{needle:?} not in:\n{t}");
    }
    let mut c = Check::new(&case);
    c.scalar("n_panels", after(&t, "N panels:   "), Tol::exact())
        .scalar("avg_t", after(&t, "Avg T:      "), Tol::printed(1))
        .scalar(
            "n_rows_dropped",
            after(&t, "Missing values: "),
            Tol::exact(),
        );
    c.finish();
}

/// ab_balanced with its row 2 (id 1, 2002) appended once more.
fn balanced_with_row_2_twice() -> Dataset {
    let path = repo_path("validation/datasets/generated/ab_balanced.csv");
    let mut df = DataLoader::load_csv(&path).unwrap();
    let extra = df.slice(2, 1);
    df.vstack_mut(&extra).unwrap();
    Dataset::new(df)
}

#[tokio::test]
async fn test_mcp_panel_gmm_duplicate_row_is_a_tool_error() {
    let (session, server) = session_with_datasets(vec![("dup", balanced_with_row_2_twice())]);
    let r = call(&server, &session, "panel_gmm", gmm_args("dup", json!({}))).await;
    let t = format!("{} {}", text(&r), r.error.clone().unwrap_or_default());
    assert!(!r.success, "{t}");
    assert!(t.contains("has 2 rows for time 2002"), "{t}");
}

#[tokio::test]
async fn test_mcp_panel_gmm_system_warns() {
    let path = repo_path("validation/datasets/generated/ab_balanced.csv");
    let ds = Dataset::new(DataLoader::load_csv(&path).unwrap());
    let (session, server) = session_with_datasets(vec![("ab_balanced", ds)]);
    let args = gmm_args("ab_balanced", json!({"transform": "system"}));
    let r = call(&server, &session, "panel_gmm", args).await;
    let t = text(&r);
    // T3-WP15 implemented system GMM; T3-WP58 renames/moves this test with the panel_gmm description
    assert!(r.success, "{t}");
    assert!(t.contains("System GMM (Blundell-Bond) (Two-step)"), "{t}");
    assert!(!t.contains("ECON-30"), "{t}");
    assert!(!t.contains("Difference GMM (Arellano-Bond)"), "{t}");
    assert!(t.contains("No. Observations: 1300"), "{t}");
}
