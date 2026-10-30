//! T2-WP06 MCP-path tests (Shapiro-Wilk and the one/two-sample wrappers). Owned by T2-WP06.
//!
//! Shapiro-Wilk, Bartlett, pairwise t-tests and Mahalanobis distances through the MCP tools,
//! on the R cases of validation/r/cases/stats_onesample.R (NA-padded and Int64 columns).

use crate::common::*;
use p2a_mcp::transport::http::{ContentItem, ToolResult};
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

/// The JSON object a tool returned as its first text item.
fn json_of(r: &ToolResult) -> Value {
    assert!(r.success, "tool error: {:?}", r.error);
    match &r.content[0] {
        ContentItem::Text { text } => {
            serde_json::from_str(text).unwrap_or_else(|e| panic!("{e}: {text}"))
        }
        other => panic!("unexpected content {other:?}"),
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_shapiro_wilk() {
    let case = Case::load("stat_shapiro_wilk");
    let (session, server) = session_with("stat_shapiro_wilk");
    let mut check = Check::new(&case);
    // NA-padded Float64 columns first, then Int64 ones
    for col in ["n5", "n20_rnorm", "n3_123", "n25_int"] {
        let v = json_of(
            &call(
                &server,
                &session,
                "hypothesis_shapiro_wilk",
                json!({"dataset": "shapiro_cases", "column": col}),
            )
            .await,
        );
        let n = v["n"].as_u64().unwrap() as usize;
        check
            .scalar(&format!("{col}_w"), f(&v["w_statistic"]), Tol::stat(n))
            .scalar(&format!("{col}_p"), f(&v["p_value"]), Tol::p_value(n))
            .scalar(&format!("{col}_n"), n as f64, Tol::exact());
        assert_eq!(
            v["reject_normality"],
            json!(case.scalar(&format!("{col}_p")) < 0.05),
            "{col}"
        );
    }
    check.finish();
}

#[tokio::test]
async fn test_validate_mcp_hypothesis_bartlett_test() {
    let case = Case::load("stat_two_sample_na");
    let (session, server) = session_with("stat_two_sample_na");
    let v = json_of(
        &call(
            &server,
            &session,
            "hypothesis_bartlett_test",
            json!({"dataset": "two_sample_na", "response": "xi", "factor": "g"}),
        )
        .await,
    );
    let groups = v["group_statistics"].as_array().unwrap();
    let names: Vec<&str> = groups
        .iter()
        .map(|g| g["group"].as_str().unwrap())
        .collect();
    let vars: Vec<f64> = groups.iter().map(|g| f(&g["variance"])).collect();
    Check::new(&case)
        .scalar("bartlett_k2", f(&v["statistic"]), Tol::stat(20))
        .scalar("bartlett_df", f(&v["df"]), Tol::exact())
        .scalar("bartlett_p", f(&v["p_value"]), Tol::p_value(20))
        .scalar(
            "bartlett_pooled_var",
            f(&v["pooled_variance"]),
            Tol::stat(20),
        )
        .named("bartlett_group_var", &names, &vars, Tol::stat(20))
        .scalar("n_xi_g", f(&v["n_obs"]), Tol::exact())
        .finish();
}
