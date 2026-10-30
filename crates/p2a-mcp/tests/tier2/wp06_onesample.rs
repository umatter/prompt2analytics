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

#[tokio::test]
async fn test_validate_mcp_hypothesis_pairwise_t_test() {
    let case = Case::load("stat_pairwise_mahalanobis_na");
    let (session, server) = session_with("stat_pairwise_mahalanobis_na");
    // the tool's default is pool_sd: false; R's default is pooled
    let v = json_of(
        &call(
            &server,
            &session,
            "hypothesis_pairwise_t_test",
            json!({"dataset": "pairwise_mahalanobis_na", "response": "y", "factor": "g", "pool_sd": true}),
        )
        .await,
    );
    let adj = &v["p_values_adjusted"];
    let lower: Vec<f64> = [(1, 0), (2, 0), (2, 1)]
        .iter()
        .map(|&(i, j)| f(&adj[i][j]))
        .collect();
    let nums = |key: &str| -> Vec<f64> {
        v[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is not an array: {v}"))
            .iter()
            .map(f)
            .collect()
    };
    let names: Vec<&str> = v["group_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    Check::new(&case)
        .vector("pairwise_p", &lower, Tol::p_value(28))
        .vector("pairwise_p_raw", &nums("p_values_raw"), Tol::p_value(28))
        .vector("pairwise_t", &nums("t_statistics"), Tol::stat(28))
        .scalar("pairwise_pooled_sd", f(&v["pooled_sd"]), Tol::stat(28))
        .named(
            "pairwise_group_n",
            &names,
            &nums("group_sizes"),
            Tol::exact(),
        )
        .named(
            "pairwise_group_mean",
            &names,
            &nums("group_means"),
            Tol::coef(28),
        )
        .finish();
}

#[tokio::test]
async fn test_validate_mcp_multivariate_mahalanobis() {
    let case = Case::load("stat_pairwise_mahalanobis_na");
    let (session, server) = session_with("stat_pairwise_mahalanobis_na");
    let v = json_of(
        &call(
            &server,
            &session,
            "multivariate_mahalanobis",
            json!({"dataset": "pairwise_mahalanobis_na", "columns": ["a", "b"]}),
        )
        .await,
    );
    let parsed = |s: &Value| -> f64 {
        let text = s.as_str().unwrap_or_else(|| panic!("not a string: {s}"));
        text.parse()
            .unwrap_or_else(|e| panic!("{text:?} is not a number: {e}"))
    };
    let center: Vec<f64> = v["center"].as_array().unwrap().iter().map(f).collect();
    let first: Vec<f64> = v["first_10_distances"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| parsed(&d["distance_squared"]))
        .collect();
    let r_dist = case.vector("mahalanobis_complete");
    let r_mean = r_dist.iter().sum::<f64>() / r_dist.len() as f64;
    let mut check = Check::new(&case);
    check
        .scalar("n_complete", f(&v["n_obs"]), Tol::exact())
        .named("mahalanobis_center", &["a", "b"], &center, Tol::coef(29));
    assert_eq!(first.len(), 10);
    for (i, d) in first.iter().enumerate() {
        assert!(
            Tol::printed(4).allows(*d, r_dist[i]),
            "first_10_distances[{i}]: {d} vs R {}",
            r_dist[i]
        );
    }
    let mean = parsed(&v["summary"]["mean_distance"]);
    assert!(
        Tol::printed(4).allows(mean, r_mean),
        "mean_distance {mean} vs R {r_mean}"
    );
    check.finish();
}
