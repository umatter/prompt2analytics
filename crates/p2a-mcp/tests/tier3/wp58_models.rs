//! T3-WP58 MCP-path tests: the regression, discrete, panel and survival tools through
//! call_tool_with_session, pinned to the wave-2 R cases. Owned by T3-WP58.
#[path = "wp58_text.rs"]
mod checks;
use crate::common::*;
use checks::*;
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

const X8: [&str; 8] = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"];
/// The MCP glmnet tools run at glmnet's default thresh (1e-7). An R port of p2a's coordinate
/// descent at that thresh lies at most 1.66e-4 (lasso), 1.55e-4 (elastic net), 1.41e-4 (ridge)
/// and 1.53e-4 (binomial) from the exact minimiser on these fits, so 1e-3 absolute.
const MCP_THRESH: Tol = Tol::new(0.0, 1e-3);

/// [intercept, coefficients in `vars` order] of one path entry; every variable must be listed.
fn entry(fit: &Value, vars: &[&str]) -> Vec<f64> {
    let listed = fit["coefficients"].as_array().expect("coefficients");
    let mut out = vec![fit["intercept"].as_f64().expect("intercept")];
    for v in vars {
        out.push(
            listed
                .iter()
                .find(|c| c["variable"] == *v)
                .and_then(|c| c["estimate"].as_f64())
                .unwrap_or_else(|| panic!("{v} is not listed: {fit}")),
        );
    }
    out
}

fn path_of(j: &Value) -> &Vec<Value> {
    j["path"].as_array().expect("path")
}

/// glmnet_paths (reg_glmnet.R): lasso, ridge and binomial lasso at user lambdas; the p > 20
/// listing rule.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_glmnet_returns_coefficients() {
    let case = Case::load("glmnet_paths");
    let (session, server) = session_with("glmnet_paths"); // dataset "glmnet_sim"
    let lams = [1.0, 0.3, 0.1, 0.03];
    // Fails before: the text "Glmnet Result: ..." is not JSON.
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_glmnet",
            json!({"dataset": "glmnet_sim", "y": "y", "x": X8, "lambda": lams}),
        )
        .await,
    );
    assert_eq!(
        (
            j["model"].as_str(),
            j["family"].as_str(),
            j["alpha"].as_f64()
        ),
        (Some("glmnet"), Some("gaussian"), Some(1.0))
    );
    assert_eq!(j["variables"], json!(X8));
    assert_eq!(j["coefficients_listed"], "all");
    let path = path_of(&j);
    assert_eq!(path.len(), 4);
    let flat: Vec<f64> = path.iter().flat_map(|f| entry(f, &X8)).collect();
    let dev: Vec<f64> = path
        .iter()
        .map(|f| f["dev_ratio"].as_f64().unwrap())
        .collect();
    Check::new(&case)
        .vector("a10_std_coef_tight", &flat, MCP_THRESH)
        .vector("a10_std_dev_ratio_tight", &dev, Tol::new(0.0, 1e-5)) // port: 6.3e-7
        .finish();
    let tight = case.vector("a10_std_coef_tight");
    for (k, f) in path.iter().enumerate() {
        assert_eq!(f["lambda"].as_f64(), Some(lams[k]));
        // the lasso zeros are exact (soft thresholding); the port gives glmnet's zero pattern
        let block = &tight[9 * k..9 * k + 9];
        assert_eq!(
            f["df"].as_u64(),
            Some(block[1..].iter().filter(|c| **c != 0.0).count() as u64)
        );
        for (a, t) in entry(f, &X8).iter().zip(block) {
            assert_eq!(*a == 0.0, *t == 0.0, "zero pattern at lambda {}", lams[k]);
        }
    }
    // the JSON carries the library's numbers unchanged (marshalling only)
    let ds = session
        .datasets
        .read()
        .await
        .get("glmnet_sim")
        .cloned()
        .unwrap();
    let mut cfg = p2a_core::regression::GlmnetConfig::default();
    cfg.lambda = Some(lams.to_vec());
    let core = p2a_core::regression::run_glmnet(&ds, "y", &X8, &cfg).unwrap();
    for (f, c) in path.iter().zip(&core.fits) {
        let mut want = vec![c.intercept];
        want.extend(&c.coefficients);
        assert_eq!(entry(f, &X8), want);
    }
    // ridge: glmnet(alpha = 0)
    let r = json_of(
        &call(
            &server,
            &session,
            "regression_ridge",
            json!({"dataset": "glmnet_sim", "y": "y", "x": X8, "lambda": lams}),
        )
        .await,
    );
    let flat: Vec<f64> = path_of(&r).iter().flat_map(|f| entry(f, &X8)).collect();
    Check::new(&case)
        .vector("a0_std_coef_tight", &flat, MCP_THRESH)
        .finish();
    assert_eq!(r["alpha"].as_f64(), Some(0.0));
    // binomial lasso; the family is parsed (any case, R's and common spellings)
    let b = json_of(
        &call(
            &server,
            &session,
            "regression_glmnet",
            json!({"dataset": "glmnet_sim", "y": "yb", "x": X8, "family": "Logistic",
                   "lambda": [0.1, 0.03, 0.01]}),
        )
        .await,
    );
    assert_eq!(b["family"], "binomial");
    let flat: Vec<f64> = path_of(&b).iter().flat_map(|f| entry(f, &X8)).collect();
    Check::new(&case)
        .vector("binom_coef_tight", &flat, MCP_THRESH)
        .finish();
    // p > 20: the 20 largest |estimate| per lambda and the count of the other non-zero ones
    let k30: Vec<String> = (1..=30).map(|j| format!("K{j}")).collect();
    let k = json_of(
        &call(
            &server,
            &session,
            "regression_glmnet",
            json!({"dataset": "glmnet_sim", "y": "yk", "x": k30,
                   "lambda": [0.0009602538655657498]}),
        )
        .await,
    );
    assert_eq!(
        k["coefficients_listed"],
        "the 20 largest |estimate| per lambda"
    );
    let fit = &path_of(&k)[0];
    let listed = fit["coefficients"].as_array().unwrap();
    let df = fit["df"].as_u64().unwrap();
    assert!(df > 20, "{df}"); // 29 in the port at thresh 1e-7 (30 at the minimiser)
    assert_eq!(
        (listed.len() as u64, fit["more_nonzero"].as_u64()),
        (20, Some(df - 20))
    );
    let mags: Vec<f64> = listed
        .iter()
        .map(|c| c["estimate"].as_f64().unwrap().abs())
        .collect();
    assert!(
        mags.windows(2).all(|w| w[0] >= w[1]) && mags[19] > 0.0,
        "{mags:?}"
    );
    // an unknown family is a tool error naming the accepted ones (before: silently gaussian)
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_glmnet",
            json!({"dataset": "glmnet_sim", "y": "y", "x": X8, "family": "poisson"}),
        )
        .await,
    );
    assert!(
        e.contains("poisson") && e.contains("gaussian, binomial"),
        "{e}"
    );
}

/// glmnet_paths: the default lasso path (61 lambdas, fdev stop) through regression_lasso.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_lasso_default_path() {
    let case = Case::load("glmnet_paths");
    let (session, server) = session_with("glmnet_paths");
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_lasso",
            json!({"dataset": "glmnet_sim", "y": "y", "x": X8}),
        )
        .await,
    );
    let path = path_of(&j);
    let lambda: Vec<f64> = path.iter().map(|f| f["lambda"].as_f64().unwrap()).collect();
    let df: Vec<f64> = path.iter().map(|f| f["df"].as_f64().unwrap()).collect();
    Check::new(&case)
        .vector("path_lambda", &lambda, Tol::coef(150))
        .vector("path_df", &df, Tol::exact())
        .finish();
    assert!(
        path.iter()
            .all(|f| f["coefficients"].as_array().unwrap().len() == 8)
    );
}

/// cv.glmnet: coefficients at lambda.min / lambda.1se equal the library's for the same seed;
/// the seed is the request's, else the session's, else CV_GLMNET_DEFAULT_SEED.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_cv_glmnet_returns_coefficients_and_uses_session_seed() {
    let case = Case::load("glmnet_paths");
    let (session, server) = session_with("glmnet_paths");
    let _ = &case;
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_cv_glmnet",
            json!({"dataset": "glmnet_sim", "y": "y", "x": X8, "nfolds": 5, "seed": 7}),
        )
        .await,
    ); // fails before: text
    let ds = session
        .datasets
        .read()
        .await
        .get("glmnet_sim")
        .cloned()
        .unwrap();
    let core = p2a_core::regression::run_cv_glmnet(
        &ds,
        "y",
        &X8,
        &p2a_core::regression::GlmnetConfig::default(),
        5,
        Some(7),
    )
    .unwrap();
    assert_eq!(
        (j["seed"].as_u64(), j["nfolds"].as_u64()),
        (Some(7), Some(5))
    );
    assert_eq!(j["lambda_min"].as_f64(), Some(core.lambda_min));
    assert_eq!(
        j["index_min"].as_u64(),
        Some(core.lambda_min_index as u64 + 1)
    );
    assert_eq!(j["lambda_1se"].as_f64(), Some(core.lambda_1se));
    assert_eq!(
        j["index_1se"].as_u64(),
        Some(core.lambda_1se_index as u64 + 1)
    );
    for (key, fit) in [
        ("at_lambda_min", core.fit_min()),
        ("at_lambda_1se", core.fit_1se()),
    ] {
        let mut want = vec![fit.intercept];
        want.extend(&fit.coefficients);
        assert_eq!(entry(&j[key], &X8), want, "{key}");
        assert_eq!(j[key]["lambda"].as_f64(), Some(fit.lambda));
    }
    assert_eq!(j["cv_curve"].as_array().unwrap().len(), core.lambda.len());
    let no_seed = json!({"dataset": "glmnet_sim", "y": "y", "x": X8, "nfolds": 5});
    let d = json_of(&call(&server, &session, "regression_cv_glmnet", no_seed.clone()).await);
    assert_eq!(
        d["seed"].as_u64(),
        Some(p2a_core::regression::glmnet::CV_GLMNET_DEFAULT_SEED)
    );
    assert!(
        call(&server, &session, "set_seed", json!({"seed": 7}))
            .await
            .success
    );
    let s = json_of(&call(&server, &session, "regression_cv_glmnet", no_seed).await);
    assert_eq!(s["seed"].as_u64(), Some(7));
    assert_eq!(s["at_lambda_min"], j["at_lambda_min"]);
}

/// The chat loop keeps only the head of a tool result (MAX_TOOL_RESULT_BYTES_IN_LOOP): the
/// warnings and the selection summary come before the path, so the model sees which
/// predictors the lasso keeps even when the full path is cut.
#[cfg(feature = "llm")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_glmnet_reply_survives_the_chat_loop_cut() {
    let case = Case::load("glmnet_paths");
    let _ = &case;
    let (session, server) = session_with("glmnet_paths");
    let k30: Vec<String> = (1..=30).map(|j| format!("K{j}")).collect();
    let x8: Vec<String> = X8.iter().map(|s| s.to_string()).collect();
    let ds = session
        .datasets
        .read()
        .await
        .get("glmnet_sim")
        .cloned()
        .unwrap();
    for (y, x) in [("y", x8), ("yk", k30)] {
        let r = call(
            &server,
            &session,
            "regression_lasso",
            json!({"dataset": "glmnet_sim", "y": y, "x": x}),
        )
        .await;
        let t = text(&r);
        let cut =
            p2a_mcp::llm::truncate_tool_result(&t, p2a_mcp::llm::MAX_TOOL_RESULT_BYTES_IN_LOOP);
        assert!(cut.contains("\"warnings\":"), "{y}: no warnings in the cut");
        let more = t.find("\"more_selected\":").expect("more_selected");
        assert!(more < cut.len(), "{y}: the selection summary is cut");
        let path_at = t.find("\"path\":").expect("path");
        assert!(path_at < 4096, "{y}: the path starts at byte {path_at}");
        // marshalling only: each listed entry index is the library's first non-zero fit
        let x_refs: Vec<&str> = x.iter().map(String::as_str).collect();
        let core = p2a_core::regression::run_lasso(&ds, y, &x_refs, None).unwrap();
        let j = json_of(&r);
        let selection = j["selection"].as_array().expect("selection");
        assert!(!selection.is_empty() && selection.len() <= 20, "{y}");
        for s in selection {
            let v = s["variable"].as_str().unwrap();
            let col = x_refs.iter().position(|n| *n == v).unwrap();
            let first = core
                .fits
                .iter()
                .position(|f| f.coefficients[col] != 0.0)
                .unwrap();
            assert_eq!(
                s["enters_at_index"].as_u64(),
                Some(first as u64 + 1),
                "{y} {v}"
            );
        }
    }
}

/// (term names, estimates, standard errors) of a GLS JSON result.
fn gls_terms(j: &Value) -> (Vec<String>, Vec<f64>, Vec<f64>) {
    let rows = j["coefficients"].as_array().expect("coefficients");
    (
        rows.iter()
            .map(|r| r["term"].as_str().unwrap().to_string())
            .collect(),
        rows.iter()
            .map(|r| r["estimate"].as_f64().unwrap())
            .collect(),
        rows.iter()
            .map(|r| r["std_error"].as_f64().unwrap())
            .collect(),
    )
}

fn num(j: &Value, k: &str) -> f64 {
    j[k].as_f64().unwrap_or_else(|| panic!("{k}: {j}"))
}

/// gls_compsymm (regression_smoothers_t3.R): nlme::gls(y ~ x + z, correlation =
/// corCompSymm(form = ~ 1 | g)), REML and ML; pinned to nlme's profile likelihood maximised
/// tightly (the *_profile_* keys).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gls_compsymm() {
    let case = Case::load("gls_compsymm");
    let n = case.scalar("n") as usize; // 150, 30 groups of 5
    let (session, server) = session_with("gls_compsymm"); // dataset "gls_cs"
    for (method, m) in [(None, "reml"), (Some("ML"), "ml")] {
        let mut a = json!({"dataset": "gls_cs", "y": "y", "x": ["x", "z"],
                           "correlation": "compound_symmetry", "group": "g"});
        if let Some(mm) = method {
            a["method"] = json!(mm);
        }
        // fails before: unknown field `group`
        let j = json_of(&call(&server, &session, "regression_gls", a).await);
        let (names, est, se) = gls_terms(&j);
        let p = |s: &str| format!("{m}_profile_{s}");
        Check::new(&case)
            .scalar(
                &p("rho"),
                num(&j, "correlation_param"),
                Tol::iterative(1e-6),
            )
            .named(&p("coef"), &names, &est, Tol::coef(n))
            .named(&p("se"), &names, &se, Tol::se(n))
            .scalar(&p("sigma"), num(&j, "sigma"), Tol::stat(n))
            .scalar(&p("loglik"), num(&j, "log_likelihood"), Tol::stat(n))
            .scalar(&format!("{m}_aic"), num(&j, "aic"), Tol::stat(n))
            .scalar(&format!("{m}_bic"), num(&j, "bic"), Tol::stat(n))
            .scalar("n_groups", num(&j, "n_groups"), Tol::exact())
            .finish();
        let source = if m == "reml" {
            "estimated (REML)"
        } else {
            "estimated (ML)"
        };
        assert_eq!(j["correlation_param_source"], source);
        assert_eq!(
            (j["method"].as_str(), j["group"].as_str()),
            (Some(m.to_uppercase().as_str()), Some("g"))
        );
        assert_eq!(j["converged"], true);
    }
    // a user rho is labelled as such and not optimised (corCompSymm(0.3, fixed = TRUE))
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_gls",
            json!({"dataset": "gls_cs", "y": "y", "x": ["x", "z"], "correlation": "cs",
                   "group": "g", "rho": 0.3}),
        )
        .await,
    );
    let (names, est, se) = gls_terms(&j);
    Check::new(&case)
        .named("cs03_reml_coef", &names, &est, Tol::coef(n))
        .named("cs03_reml_se", &names, &se, Tol::se(n))
        .scalar("cs03_reml_sigma", num(&j, "sigma"), Tol::stat(n))
        .scalar("cs03_reml_loglik", num(&j, "log_likelihood"), Tol::stat(n))
        .finish();
    assert_eq!(j["correlation_param_source"], "user-supplied");
    assert!(j.get("converged").is_none(), "{j}");
}

/// gls_ar1: the default AR(1) without rho is nlme's REML (phi 0.68198), not the lag-1
/// residual correlation (0.6542); ML on request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gls_ar1_reml() {
    let case = Case::load("gls_ar1");
    let n = case.scalar("n_obs") as usize; // 50
    let (session, server) = session_with("gls_ar1");
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_gls",
            json!({"dataset": "gls_ar1", "y": "y", "x": ["x"]}),
        )
        .await,
    );
    let (names, est, se) = gls_terms(&j);
    Check::new(&case)
        // fails before: 0.6541921822266757
        .scalar("phi", num(&j, "correlation_param"), Tol::iterative(1e-6))
        .named("coef", &names, &est, Tol::coef(n))
        .named("se", &names, &se, Tol::se(n))
        .scalar("log_lik_reml", num(&j, "log_likelihood"), Tol::stat(n))
        .finish();
    assert_eq!(j["correlation_param_source"], "estimated (REML)");
    let ml = json_of(
        &call(
            &server,
            &session,
            "regression_gls",
            json!({"dataset": "gls_ar1", "y": "y", "x": ["x"], "method": "ml"}),
        )
        .await,
    );
    let (names, est, _) = gls_terms(&ml);
    Check::new(&case)
        .scalar(
            "phi_ml",
            num(&ml, "correlation_param"),
            Tol::iterative(1e-6),
        )
        .named("coef_ml", &names, &est, Tol::coef(n))
        .scalar("log_lik_ml", num(&ml, "log_likelihood"), Tol::stat(n))
        .finish();
}

/// Compound symmetry needs groups; unknown structures and methods are errors.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gls_options_and_errors() {
    let case = Case::load("gls_compsymm");
    let _ = &case;
    let (session, server) = session_with("gls_compsymm");
    let base = json!({"dataset": "gls_cs", "y": "y", "x": ["x", "z"]});
    let with = |k: &str, v: Value| {
        let mut a = base.clone();
        a[k] = v;
        a
    };
    // fails before: success with correlation_param 0.5
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_gls",
            with("correlation", json!("compound_symmetry")),
        )
        .await,
    );
    assert!(e.contains("rho") && e.contains("not identified"), "{e}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_gls",
            with("correlation", json!("toeplitz")),
        )
        .await,
    );
    assert!(
        e.contains("toeplitz") && e.contains("compound_symmetry") && e.contains("ar1"),
        "{e}"
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_gls",
            with("method", json!("GMM")),
        )
        .await,
    );
    assert!(
        e.contains("GMM") && e.contains("reml") && e.contains("ml"),
        "{e}"
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_gls",
            json!({"dataset": "gls_cs", "y": "y", "x": ["x"], "correlation": "cs", "group": "nope"}),
        )
        .await,
    );
    assert!(e.contains("nope"), "{e}");
    // identity: no correlation parameter
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_gls",
            with("correlation", json!("identity")),
        )
        .await,
    );
    assert!(
        j["correlation_param"].is_null() && j["correlation_param_source"] == "not applicable",
        "{j}"
    );
}
