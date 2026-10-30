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

/// The JSON object a tool returned as its only text item.
fn json_of(r: &p2a_mcp::transport::http::ToolResult) -> serde_json::Value {
    let t = text(r);
    assert!(r.success, "tool error: {t}");
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{e}: {t}"))
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[tokio::test]
async fn test_validate_mcp_anova_one_way_na_labels() {
    let case = Case::load("stat_oneway_na_labels");
    let (session, server) = session_with("stat_oneway_na_labels");
    let r = call(
        &server,
        &session,
        "anova_one_way",
        json!({"dataset": "oneway_na_labels", "response": "y", "factor": "g"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let (p4, x) = (Tol::printed(4), Tol::exact());
    let between = row(&t, "Between");
    let within = row(&t, "Within");
    let labels = ["a", "b", "c"];
    let sizes: Vec<f64> = labels.iter().map(|g| row(&t, g)[0]).collect();
    Check::new(&case)
        .scalar("aov_ss_between", between[0], p4)
        .scalar("aov_df_between", between[1], x)
        .scalar("aov_f", between[3], p4)
        .scalar("aov_p", between[4], p4)
        .scalar("aov_ss_within", within[0], p4)
        .scalar("aov_df_within", within[1], x)
        .named("aov_group_n", &labels, &sizes, x)
        .finish();
    assert!(
        !t.lines().any(|l| {
            let l = l.trim_start();
            l.starts_with("Null") || l.starts_with("null") || l.starts_with('"')
        }),
        "{t}"
    );
    assert!(t.contains("N = 5  |  Groups = 3"), "{t}");
    assert!(t.contains("Missing values: 4 of 9 rows dropped"), "{t}");
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_kruskal_wallis_na_labels() {
    let case = Case::load("stat_oneway_na_labels");
    let (session, server) = session_with("stat_oneway_na_labels");
    let v = json_of(
        &call(
            &server,
            &session,
            "hypothesis_kruskal_wallis",
            json!({"dataset": "oneway_na_labels", "value": "v9", "group": "g9"}),
        )
        .await,
    );
    let groups = v["group_statistics"].as_array().unwrap();
    let names: Vec<&str> = groups
        .iter()
        .map(|g| g["group"].as_str().unwrap())
        .collect();
    let sizes: Vec<f64> = groups.iter().map(|g| num(&g["n"])).collect();
    Check::new(&case)
        .scalar("kw_chisq", num(&v["statistic"]), Tol::stat(6))
        .scalar("kw_df", num(&v["df"]), Tol::exact())
        .scalar("kw_p", num(&v["p_value"]), Tol::p_value(6))
        .named("kw_group_n", &names, &sizes, Tol::exact())
        .finish();
    assert_eq!(names, ["a", "b"]);
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_oneway_na_labels() {
    let case = Case::load("stat_oneway_na_labels");
    let (session, server) = session_with("stat_oneway_na_labels");
    let v = json_of(
        &call(
            &server,
            &session,
            "hypothesis_oneway",
            json!({"dataset": "oneway_na_labels", "value": "v9", "group": "g9"}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("welch_f", num(&v["statistic"]), Tol::stat(6))
        .scalar("welch_df1", num(&v["df_numerator"]), Tol::stat(6))
        .scalar("welch_df2", num(&v["df_denominator"]), Tol::stat(6))
        .scalar("welch_p", num(&v["p_value"]), Tol::p_value(6))
        .finish();
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_wilcoxon_paired_na() {
    let case = Case::load("stat_wilcoxon_na");
    let (session, server) = session_with("stat_wilcoxon_na");
    let r = call(
        &server,
        &session,
        "hypothesis_wilcoxon",
        json!({"dataset": "wilcoxon_na", "x": "x", "y": "y", "paired": true, "exact": false}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    // The Display text, a blank line, then the JSON object.
    let start = t.find("\n{").unwrap_or_else(|| panic!("no JSON in:\n{t}"));
    let v: serde_json::Value =
        serde_json::from_str(&t[start + 1..]).unwrap_or_else(|e| panic!("{e}: {t}"));
    Check::new(&case)
        .scalar("paired_v", num(&v["statistic"]), Tol::exact())
        .scalar("paired_p_normal", num(&v["p_value"]), Tol::p_value(8))
        .scalar("paired_z", num(&v["z_score"]), Tol::stat(8))
        .scalar("n_pairs", num(&v["n1"]), Tol::exact())
        .finish();
    assert!(t.contains("Missing values: 2 of 10 rows dropped"), "{t}");
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_friedman_na() {
    use p2a_core::data::{DataLoader, Dataset};
    use p2a_core::polars::prelude::*;

    let case = Case::load("stat_friedman_quade");
    let (session, server) = session_with("stat_friedman_quade");
    let args = |value: &str| {
        json!({"dataset": "friedman_quade", "value": value, "treatment": "treat",
               "block": "block"})
    };
    let v = json_of(&call(&server, &session, "hypothesis_friedman", args("y")).await);
    let treatments: Vec<&str> = v["treatment_statistics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["treatment"].as_str().unwrap())
        .collect();
    assert_eq!(treatments, ["A", "B", "C"]);
    let na = json_of(&call(&server, &session, "hypothesis_friedman", args("y_na")).await);
    Check::new(&case)
        .scalar("friedman_q", num(&v["statistic"]), Tol::stat(24))
        .scalar("friedman_df", num(&v["df"]), Tol::exact())
        .scalar("friedman_p", num(&v["p_value"]), Tol::p_value(24))
        .scalar("na_friedman_q", num(&na["statistic"]), Tol::stat(24))
        .scalar("na_n_blocks", num(&na["n_blocks"]), Tol::exact())
        .finish();

    // A replicated (block, treatment) cell is refused, as friedman.test refuses it.
    let df = DataLoader::load_csv(case.dataset_path()).unwrap();
    let extra = df! {
        "y" => [99i64], "y_na" => [Some(99i64)], "treat" => ["A"], "block" => ["S1"],
    }
    .unwrap();
    let dup = Dataset::new(df.vstack(&extra).unwrap());
    let (session, server) = session_with_datasets(vec![("friedman_dup", dup)]);
    let r = call(
        &server,
        &session,
        "hypothesis_friedman",
        json!({"dataset": "friedman_dup", "value": "y", "treatment": "treat", "block": "block"}),
    )
    .await;
    let t = format!("{}\n{}", text(&r), r.error.clone().unwrap_or_default());
    assert!(!r.success, "{t}");
    assert!(
        t.contains("not an unreplicated complete block design"),
        "{t}"
    );
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_quade_na() {
    let case = Case::load("stat_friedman_quade");
    let (session, server) = session_with("stat_friedman_quade");
    let args = |value: &str| {
        json!({"dataset": "friedman_quade", "value": value, "treatment": "treat",
               "block": "block"})
    };
    let v = json_of(&call(&server, &session, "hypothesis_quade", args("y")).await);
    let na = json_of(&call(&server, &session, "hypothesis_quade", args("y_na")).await);
    Check::new(&case)
        .scalar("quade_f", num(&v["statistic"]), Tol::stat(24))
        .scalar("quade_df1", num(&v["df1"]), Tol::exact())
        .scalar("quade_df2", num(&v["df2"]), Tol::exact())
        .scalar("quade_p", num(&v["p_value"]), Tol::p_value(24))
        .scalar("na_quade_complete_f", num(&na["statistic"]), Tol::stat(24))
        .scalar("na_quade_complete_df2", num(&na["df2"]), Tol::exact())
        .scalar("na_quade_complete_p", num(&na["p_value"]), Tol::p_value(24))
        .finish();
}
