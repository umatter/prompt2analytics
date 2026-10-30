//! T2-WP10 MCP-path tests (ordered logit/probit and count models). Owned by T2-WP10.

use p2a_validation::text::labelled;
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

use crate::common::{call, session_with, text};

/// The first line whose trimmed start begins with `start`.
fn line<'a>(t: &'a str, start: &str) -> &'a str {
    t.lines()
        .find(|l| l.trim_start().starts_with(start))
        .unwrap_or_else(|| panic!("no line starting with {start:?} in:\n{t}"))
}

/// The whitespace tokens of `l` that parse as numbers once `(),:;` are trimmed.
fn numbers(l: &str) -> Vec<f64> {
    l.split_whitespace()
        .filter_map(|tok| {
            tok.trim_matches(|c| matches!(c, '(' | ')' | ',' | ':' | ';'))
                .parse::<f64>()
                .ok()
        })
        .collect()
}

/// The four numbers (coefficient, SE, z, p) of the table row whose first token is `name`.
fn row(t: &str, name: &str) -> Vec<f64> {
    let l = t
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{t}"));
    let v = numbers(l);
    assert_eq!(v.len(), 4, "row {name:?}: {l:?}");
    v
}

/// The `i`-th number of each named row, in `names` order.
fn column(t: &str, names: &[&str], i: usize) -> Vec<f64> {
    names.iter().map(|n| row(t, n)[i]).collect()
}

/// The printed polr fit of `t` against the case keys with prefix `p`.
fn check_printed_polr(case: &Case, p: &str, t: &str) {
    let xs = ["x1", "x2"];
    let tol = Tol::printed(4);
    let k = |s: &str| format!("{p}{s}");
    let cut1 = numbers(line(t, "1|2:"));
    let cut2 = numbers(line(t, "2|3:"));
    Check::new(case)
        .named(&k("coef"), &xs, &column(t, &xs, 0), tol)
        .named(&k("se_coef"), &xs, &column(t, &xs, 1), tol)
        .named(&k("z"), &xs, &column(t, &xs, 2), tol)
        .named(&k("p"), &xs, &column(t, &xs, 3), tol)
        .named(&k("zeta"), &["1|2", "2|3"], &[cut1[0], cut2[0]], tol)
        .named(&k("se_zeta"), &["1|2", "2|3"], &[cut1[1], cut2[1]], tol)
        .scalar(&k("loglik"), labelled(t, "Log-Likelihood:").unwrap(), tol)
        .finish();
}

/// `ordered_model` prints polr's fit (before: sign-reversed estimates, SE 0.1000).
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_mcp_ordered_model_disc_ordered() {
    let case = Case::load("disc_ordered");
    let (session, server) = session_with("disc_ordered");
    for (model_type, p) in [("logit", "logit_"), ("probit", "probit_")] {
        let r = call(
            &server,
            &session,
            "ordered_model",
            json!({"dataset": "ordered", "y": "y", "x": ["x1", "x2"], "model_type": model_type}),
        )
        .await;
        let t = text(&r);
        assert!(r.success, "{t}");
        assert!(t.contains("Converged: true"), "{t}");
        check_printed_polr(&case, p, &t);
    }
}

/// A text outcome without an order is a tool error that asks for `levels`.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_ordered_model_text_outcome_asks_for_levels() {
    let (session, server) = session_with("disc_ordered");
    let r = call(
        &server,
        &session,
        "ordered_model",
        json!({"dataset": "ordered", "y": "y_label", "x": ["x1", "x2"]}),
    )
    .await;
    let t = text(&r);
    assert!(!r.success, "{t}");
    assert!(t.contains("levels"), "{t}");
}

/// `negbin` prints glm.nb's fit (before: theta stopped below the MLE and SE(theta) was
/// 0.1 theta).
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_mcp_negbin_disc_negbin() {
    let case = Case::load("disc_negbin");
    let (session, server) = session_with("disc_negbin");
    let r = call(
        &server,
        &session,
        "negbin",
        json!({"dataset": "negbin", "y": "y", "x": ["x1", "x2"]}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(t.contains("Converged: true"), "{t}");
    let names = ["(Intercept)", "x1", "x2"];
    let tol = Tol::printed(4);
    let theta = numbers(line(&t, "Theta:"));
    Check::new(&case)
        .named("coef", &names, &column(&t, &names, 0), tol)
        .named("se", &names, &column(&t, &names, 1), tol)
        .named("z", &names, &column(&t, &names, 2), tol)
        .named("p", &names, &column(&t, &names, 3), tol)
        .scalar("theta", theta[0], tol)
        .scalar("se_theta", theta[1], tol)
        .scalar("loglik", labelled(&t, "Log-Likelihood:").unwrap(), tol)
        .finish();
}

/// Rows with missing values are dropped and reported; a bad `init_theta` is refused.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_negbin_reports_dropped_rows_and_rejects_bad_init_theta() {
    let (session, server) = session_with("disc_negbin");
    let r = call(
        &server,
        &session,
        "negbin",
        json!({"dataset": "negbin", "y": "y_na", "x": ["x1_na", "x2"]}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(t.contains("Missing values: 5 of 400 rows dropped"), "{t}");
    let r = call(
        &server,
        &session,
        "negbin",
        json!({"dataset": "negbin", "y": "y", "x": ["x1", "x2"], "init_theta": -1.0}),
    )
    .await;
    let t = text(&r);
    assert!(!r.success, "{t}");
    assert!(t.contains("init_theta"), "{t}");
}

/// `zeroinfl` prints pscl's fit for both distributions (before: SE 0.1 after two EM
/// passes reported as converged, and a clamped moment theta for ZINB).
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_mcp_zeroinfl_disc_zeroinfl() {
    let case = Case::load("disc_zeroinfl");
    let (session, server) = session_with("disc_zeroinfl");
    let names = ["(Intercept)", "x1", "x2"];
    let tol = Tol::printed(4);
    for (y, dist, p) in [("y", "poisson", "zip_"), ("y_nb", "negbin", "zinb_")] {
        let r = call(
            &server,
            &session,
            "zeroinfl",
            json!({"dataset": "zeroinfl", "y": y, "x": ["x1", "x2"], "z": ["x1", "x2"], "dist": dist}),
        )
        .await;
        let t = text(&r);
        assert!(r.success, "{t}");
        assert!(t.contains("Converged: true"), "{t}");
        let (count, zero) = t
            .split_once("Zero-Inflation Model")
            .unwrap_or_else(|| panic!("no zero-inflation table in:\n{t}"));
        let k = |s: &str| format!("{p}{s}");
        let mut check = Check::new(&case);
        check
            .named(&k("count"), &names, &column(count, &names, 0), tol)
            .vector(&k("se_count"), &column(count, &names, 1), tol)
            .vector(&k("z_count"), &column(count, &names, 2), tol)
            .vector(&k("p_count"), &column(count, &names, 3), tol)
            .named(&k("zero"), &names, &column(zero, &names, 0), tol)
            .vector(&k("se_zero"), &column(zero, &names, 1), tol)
            .vector(&k("z_zero"), &column(zero, &names, 2), tol)
            .vector(&k("p_zero"), &column(zero, &names, 3), tol)
            .scalar(&k("loglik"), labelled(&t, "Log-Likelihood:").unwrap(), tol);
        if dist == "negbin" {
            let theta = numbers(line(&t, "Theta:"));
            assert_eq!(theta.len(), 4, "{t}");
            check
                .scalar("zinb_theta", theta[0], tol)
                .scalar("zinb_se_theta", theta[1], tol)
                .scalar("zinb_log_theta", theta[2], tol)
                .scalar("zinb_se_log_theta", theta[3], tol);
        }
        check.finish();
    }
}
