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

/// rq_br_se (reg_quantreg_br.R): rq(method = "br") and summary.rq(se = nid | iid | ker) at
/// tau 0.25, 0.5, 0.75.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_quantreg_br_se() {
    let case = Case::load("rq_br_se");
    let (session, server) = session_with("rq_br_se"); // dataset "quantreg_hetero"
    let base = json!({"dataset": "quantreg_hetero", "y": "y", "x": ["x"], "tau": 0.25});
    for (se, key) in [
        (None, "nid"),
        (Some("iid"), "iid"),
        (Some("ker"), "ker"),
        (Some("Powell"), "ker"),
    ] {
        let mut a = base.clone();
        if let Some(s) = se {
            a["se"] = json!(s); // fails before: unknown field `se`
        }
        let t = text(&call(&server, &session, "regression_quantreg", a).await);
        for v in ["(Intercept)", "x"] {
            assert_row(
                &format!("tau 0.25 {v} se {key}"),
                row(&t, v),
                &[
                    case.named("tau25_coef", v),
                    case.named(&format!("tau25_se_{key}"), v),
                ],
                Tol::printed(4),
            );
        }
        // the default method 'auto' runs the simplex at this n
        assert!(t.contains("Barrodale-Roberts simplex (br)"), "{t}");
    }
    let t = text(
        &call(
            &server,
            &session,
            "regression_quantreg",
            json!({"dataset": "quantreg_hetero", "y": "y", "x": ["x"], "taus": [0.25, 0.5, 0.75]}),
        )
        .await,
    );
    let blocks: Vec<&str> = t.split("\n---\n").collect();
    assert_eq!(blocks.len(), 3, "{t}");
    for (b, tag) in blocks.iter().zip(["tau25", "tau50", "tau75"]) {
        assert_row(
            tag,
            row(b, "x"),
            &[
                case.named(&format!("{tag}_coef"), "x"),
                case.named(&format!("{tag}_se_nid"), "x"),
            ],
            Tol::printed(4),
        );
    }
    let mut a = base.clone();
    a["method"] = json!("simplex");
    assert!(
        text(&call(&server, &session, "regression_quantreg", a).await)
            .contains("Barrodale-Roberts simplex (br)")
    );
    let mut a = base.clone();
    a["method"] = json!("IRLS");
    assert!(
        text(&call(&server, &session, "regression_quantreg", a).await)
            .contains("IRLS approximation (irls)")
    );
    let mut a = base.clone();
    a["se"] = json!("hc1");
    let e = tool_error(&call(&server, &session, "regression_quantreg", a).await);
    assert!(
        e.contains("hc1") && e.contains("nid, iid, ker, boot"),
        "{e}"
    );
    let mut a = base.clone();
    a["method"] = json!("lasso");
    let e = tool_error(&call(&server, &session, "regression_quantreg", a).await);
    assert!(
        e.contains("lasso") && e.contains("auto, br, fn, irls"),
        "{e}"
    );
}

/// An empty taus list, tau together with taus, and tau on the boundary are tool errors.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_quantreg_empty_taus_is_error() {
    let case = Case::load("rq_br_se");
    let _ = &case;
    let (session, server) = session_with("rq_br_se");
    let q = |extra: Value| {
        let mut a = json!({"dataset": "quantreg_hetero", "y": "y", "x": ["x"]});
        for (k, v) in extra.as_object().unwrap() {
            a[k] = v.clone();
        }
        a
    };
    // before: an empty success
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_quantreg",
            q(json!({"taus": []})),
        )
        .await,
    );
    assert!(e.to_lowercase().contains("tau"), "{e}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_quantreg",
            q(json!({"tau": 0.5, "taus": [0.25]})),
        )
        .await,
    );
    assert!(e.contains("tau") && e.contains("taus"), "{e}");
    for tau in [0.0, 1.0] {
        let e = tool_error(
            &call(
                &server,
                &session,
                "regression_quantreg",
                q(json!({"tau": tau})),
            )
            .await,
        );
        assert!(e.contains("tau"), "tau = {tau}: {e}");
    }
}

/// ols_aliased (reg_rank_qr.R): the dummy trap y ~ x1 + da + db + dc; lm reports dc as NA
/// (rank 4 of 5).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_ols_aliased() {
    let case = Case::load("ols_aliased");
    let (session, server) = session_with("ols_aliased");
    let t = text(
        &call(
            &server,
            &session,
            "regression_ols",
            json!({"dataset": "ols_aliased", "y": "y", "x": ["x1", "da", "db", "dc"],
                   "cov_type": "standard"}),
        )
        .await,
    );
    for v in ["(Intercept)", "x1", "da", "db"] {
        assert_row(
            v,
            row(&t, v),
            &[case.named("trap_coef", v), case.named("trap_se", v)],
            Tol::printed(4),
        );
    }
    assert!(
        printed(row(&t, "dc")).is_empty(),
        "the aliased row shows no number:\n{t}"
    );
    assert!(
        t.contains("Coefficients not defined because of singularities: dc"),
        "{t}"
    );
}

/// lmtest cases and reset_princomp through the MCP diagnostics tools.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_diagnostics_tests() {
    let (dg, bg, rp, wd, hv) = (
        Case::load("ols_diagnostics_grunfeld"),
        Case::load("diag_bgtest_ar1"),
        Case::load("reset_princomp"),
        Case::load("diag_wald_sim"),
        Case::load("diag_harvtest_nonlin"),
    );
    let (session, server) = session_with_datasets(vec![
        ("grunfeld", dataset_of(&dg)),
        ("bgtest_ar1", dataset_of(&bg)),
        ("reset_two", dataset_of(&rp)),
        ("wald_sim", dataset_of(&wd)),
        ("harvtest_nonlin", dataset_of(&hv)),
    ]);
    let t = text(
        &call(
            &server,
            &session,
            "regression_diagnostics",
            json!({"dataset": "grunfeld", "y": "inv", "x": ["value", "capital"]}),
        )
        .await,
    );
    let printed4 = Tol::printed(4);
    assert_shows(
        "JB",
        after(section(&t, "Jarque-Bera"), "Statistic:"),
        dg.scalar("jarque_bera"),
        printed4,
    );
    assert_shows(
        "BP",
        after(section(&t, "Breusch-Pagan"), "Statistic:"),
        dg.scalar("breusch_pagan"),
        printed4,
    );
    assert_shows(
        "DW",
        after(section(&t, "Durbin-Watson"), "Statistic:"),
        dg.scalar("durbin_watson"),
        printed4,
    );
    assert_shows(
        "kappa",
        after(section(&t, "Condition Number"), "Value:"),
        dg.scalar("condition_number"),
        Tol::printed(2),
    );
    let t = text(
        &call(
            &server,
            &session,
            "regression_bgtest",
            json!({"dataset": "bgtest_ar1", "y": "y", "x": ["x"], "order": 4, "test_type": "F"}),
        )
        .await,
    );
    assert_shows(
        "bgtest F",
        after(&t, "F Statistic:"),
        bg.scalar("bgtest4_f"),
        printed4,
    );
    assert_row("bgtest df", after(&t, "df:"), &[4.0, 94.0], Tol::exact());
    let t = text(
        &call(
            &server,
            &session,
            "regression_resettest",
            json!({"dataset": "reset_two", "y": "y", "x": ["x1", "x2"], "reset_type": "princomp"}),
        )
        .await,
    );
    assert_shows(
        "RESET princomp",
        after(&t, "F Statistic:"),
        rp.scalar("princomp_stat"),
        printed4,
    );
    assert_row(
        "RESET df",
        after(&t, "df:"),
        &rp.vector("princomp_df"),
        Tol::exact(),
    );
    let t = text(
        &call(
            &server,
            &session,
            "regression_waldtest",
            json!({"dataset": "wald_sim", "y": "y", "x_unrestricted": ["x1", "x2", "x3"],
                   "x_restricted": ["x1"]}),
        )
        .await,
    );
    assert_shows(
        "Wald F",
        after(&t, "F Statistic:"),
        wd.scalar("wald_drop_x2_x3_f"),
        printed4,
    );
    assert_row("Wald df", after(&t, "df:"), &[2.0, 96.0], Tol::exact());
    let t = text(
        &call(
            &server,
            &session,
            "regression_harvtest",
            json!({"dataset": "harvtest_nonlin", "y": "y", "x": ["x"]}),
        )
        .await,
    );
    // lmtest::harvtest reports |t|; p2a keeps the sign of the mean recursive residual
    let signed = hv.scalar("harvey_collier") * hv.scalar("recresid_mean").signum();
    assert_shows(
        "Harvey-Collier",
        after(&t, "Test statistic (t):"),
        signed,
        printed4,
    );
    assert_shows(
        "HC df",
        after(&t, "Degrees of freedom:"),
        hv.scalar("harvey_collier_df"),
        Tol::exact(),
    );
}

/// hac_ar1 and dk_panel: Parzen kernel at bandwidth 4 as sandwich::kernHAC; Driscoll-Kraay
/// coefficients and the number of periods (its SEs are REG-24, Tier 4: not pinned).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_hac_dk() {
    let (hac, dk) = (Case::load("hac_ar1"), Case::load("dk_panel"));
    let (session, server) = session_with_datasets(vec![
        ("hac_ar1", dataset_of(&hac)),
        ("dk_panel", dataset_of(&dk)),
    ]);
    let t = text(
        &call(
            &server,
            &session,
            "regression_hac",
            json!({"dataset": "hac_ar1", "y": "y", "x": ["x"], "kernel": "parzen", "bandwidth": 4}),
        )
        .await,
    );
    for v in ["(Intercept)", "x"] {
        assert_row(
            v,
            row(&t, v),
            &[hac.named("coef", v), hac.named("kernhac_parzen_bw4", v)],
            Tol::printed(4),
        );
    }
    assert!(
        call(
            &server,
            &session,
            "regression_hac",
            json!({"dataset": "hac_ar1", "y": "y", "x": ["x"], "kernel": "Quadratic Spectral"}),
        )
        .await
        .success
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_hac",
            json!({"dataset": "hac_ar1", "y": "y", "x": ["x"], "kernel": "gaussian"}),
        )
        .await,
    );
    assert!(
        e.contains("gaussian") && e.to_lowercase().contains("bartlett"),
        "{e}"
    );
    let t = text(
        &call(
            &server,
            &session,
            "regression_driscoll_kraay",
            json!({"dataset": "dk_panel", "y": "y", "x": ["x"], "time_col": "time"}),
        )
        .await,
    );
    for v in ["(Intercept)", "x"] {
        assert_row(v, row(&t, v), &[dk.named("coef", v)], Tol::printed(4));
    }
    assert_shows("T", after(&t, "T ="), dk.scalar("n_periods"), Tol::exact());
}

/// Option strings of the regression tools are parsed; unknown values are errors that list
/// the accepted ones.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_options_reject_unknown() {
    let (bg, st, nl) = (
        Case::load("diag_bgtest_ar1"),
        Case::load("step_legacy"),
        Case::load("nls_expdecay"),
    );
    let (session, server) = session_with_datasets(vec![
        ("bgtest_ar1", dataset_of(&bg)),
        ("step_legacy", dataset_of(&st)),
        ("nls_expdecay", dataset_of(&nl)),
    ]);
    let reg = json!({"dataset": "bgtest_ar1", "y": "y", "x": ["x"]});
    let with = |base: &Value, k: &str, v: Value| {
        let mut a = base.clone();
        a[k] = v;
        a
    };
    // before: chisq
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_bgtest",
            with(&reg, "test_type", json!("wald")),
        )
        .await,
    );
    assert!(e.contains("'wald'") && e.contains("chisq, f"), "{e}");
    // before: fitted
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_resettest",
            with(&reg, "reset_type", json!("squares")),
        )
        .await,
    );
    assert!(
        e.contains("squares") && e.contains("fitted, regressor, princomp"),
        "{e}"
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_bootstrap_cov",
            with(&reg, "bootstrap_type", json!("jackknife")),
        )
        .await,
    );
    assert!(
        e.contains("jackknife") && e.to_lowercase().contains("pairs"),
        "{e}"
    );
    let step = json!({"dataset": "step_legacy", "response": "y", "predictors": ["x1", "x2", "x3"]});
    // before: AIC
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_step",
            with(&step, "criterion", json!("cp")),
        )
        .await,
    );
    assert!(e.contains("'cp'") && e.contains("aic, bic"), "{e}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_step",
            with(&step, "direction", json!("sideways")),
        )
        .await,
    );
    assert!(e.contains("sideways"), "{e}");
    let nls = json!({"dataset": "nls_expdecay", "x": "x", "y": "y", "model": "Exp-Decay",
                     "start": [8.0, 0.3, 1.0]});
    // before: "Unknown model 'Exp-Decay'"
    let r = call(
        &server,
        &session,
        "regression_nls",
        with(&nls, "algorithm", json!("LM")),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_nls",
            with(&nls, "algorithm", json!("newton")),
        )
        .await,
    );
    assert!(
        e.contains("newton") && e.contains("levenberg_marquardt, gauss_newton"),
        "{e}"
    );
}

/// The bootstrap is seeded by the request, else the session seed, else 42 (before: entropy).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_bootstrap_cov_is_seeded() {
    let case = Case::load("ols_mtcars");
    let (session, server) = session_with_datasets(vec![("mtcars", dataset_of(&case))]);
    let a = json!({"dataset": "mtcars", "y": "mpg", "x": ["wt"], "n_boot": 200});
    let run = |extra: Option<u64>| {
        let mut x = a.clone();
        if let Some(s) = extra {
            x["seed"] = json!(s);
        }
        x
    };
    let first = text(&call(&server, &session, "regression_bootstrap_cov", run(None)).await);
    // fails before: two unseeded calls differ
    assert_eq!(
        first,
        text(&call(&server, &session, "regression_bootstrap_cov", run(None)).await)
    );
    assert_eq!(
        first,
        text(&call(&server, &session, "regression_bootstrap_cov", run(Some(42))).await)
    );
    assert!(
        call(&server, &session, "set_seed", json!({"seed": 5}))
            .await
            .success
    );
    let five = text(&call(&server, &session, "regression_bootstrap_cov", run(None)).await);
    assert_eq!(
        five,
        text(&call(&server, &session, "regression_bootstrap_cov", run(Some(5))).await)
    );
    assert_ne!(five, first);
}

/// loess_x10_direct and smoothers_gcv_periodic_loess (regression_smoothers_t3.R):
/// loess(surface = "direct") on x = 1:10; the 60% mass point at 0 gives lowess's tie mean with
/// a zero-width warning in the JSON.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_loess_x10_and_mass_point() {
    let (x10, sm) = (
        Case::load("loess_x10_direct"),
        Case::load("smoothers_gcv_periodic_loess"),
    );
    let (session, server) = session_with_datasets(vec![
        ("loess_x10", dataset_of(&x10)),
        ("smoothers_t3", dataset_of(&sm)),
    ]);
    for (span, degree, key) in [
        (0.75, 2, "fit_s075_d2"),
        (0.75, 1, "fit_s075_d1"),
        (1.2, 2, "fit_s12_d2"),
    ] {
        let j = json_of(
            &call(
                &server,
                &session,
                "regression_loess",
                json!({"dataset": "loess_x10", "y": "y", "x": "x", "span": span, "degree": degree}),
            )
            .await,
        );
        Check::new(&x10)
            .vector(key, &nums(&j["fitted_values"]), Tol::coef(10))
            .finish();
    }
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_loess",
            json!({"dataset": "smoothers_t3", "y": "ym", "x": "xm", "span": 0.5, "degree": 1}),
        )
        .await,
    );
    Check::new(&sm)
        .vector(
            "loess_masspoint_expected",
            &nums(&j["fitted_values"]),
            Tol::coef(100),
        )
        .finish();
    let w = j["warnings"].as_array().expect("warnings"); // fails before: no "warnings" key
    assert!(
        w.iter().any(|s| s.as_str().unwrap().contains("zero-width")),
        "{w:?}"
    );
}

/// supsmu_sine and smoothers_gcv_periodic_loess: fixed span, weights and periodic = TRUE as
/// R's supsmu.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_supsmu_span_and_periodic() {
    let (ss, sm) = (
        Case::load("supsmu_sine"),
        Case::load("smoothers_gcv_periodic_loess"),
    );
    let (session, server) = session_with_datasets(vec![
        ("supsmu_sine", dataset_of(&ss)),
        ("smoothers_t3", dataset_of(&sm)),
    ]);
    let head = |c: &Case, k: &str| c.vector(k).into_iter().take(20).collect::<Vec<f64>>();
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_supsmu",
            json!({"dataset": "supsmu_sine", "x": "x", "y": "y", "span": 0.3}),
        )
        .await,
    );
    assert_eq!(nums(&j["x_sample"]), head(&ss, "cv_x")); // the sorted unique x, exactly
    assert_eq!(nums(&j["y_smoothed_sample"]).len(), 20);
    for (i, (a, e)) in nums(&j["y_smoothed_sample"])
        .iter()
        .zip(head(&ss, "span03_y"))
        .enumerate()
    {
        assert!(
            Tol::coef(100).allows(*a, e),
            "span 0.3 point {i}: {a} vs {e}"
        );
    }
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_supsmu",
            json!({"dataset": "supsmu_sine", "x": "x", "y": "y", "weights": "w"}),
        )
        .await,
    );
    for (a, e) in nums(&j["y_smoothed_sample"])
        .iter()
        .zip(head(&ss, "weighted_y"))
    {
        assert!(Tol::coef(100).allows(*a, e), "weighted: {a} vs {e}");
    }
    for (periodic, key) in [(true, "supsmu_periodic_y"), (false, "supsmu_plain_y")] {
        let j = json_of(
            &call(
                &server,
                &session,
                "regression_supsmu",
                json!({"dataset": "smoothers_t3", "x": "x", "y": "y", "periodic": periodic}),
            )
            .await,
        );
        for (a, e) in nums(&j["y_smoothed_sample"]).iter().zip(head(&sm, key)) {
            assert!(Tol::coef(100).allows(*a, e), "{key}: {a} vs {e}");
        }
    }
    // periodic smoothing needs x in [0, 1] (R stops); supsmu_sine's x runs to 2 pi
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_supsmu",
            json!({"dataset": "supsmu_sine", "x": "x", "y": "y", "periodic": true}),
        )
        .await,
    );
    assert!(e.contains("periodic"), "{e}");
}

/// smooth.spline's criterion through MCP. Its numbers are REG-29/REG-30 (Tier 4) and are not
/// R-pinned: the tool must equal the library with the same criterion, and GCV and CV must give
/// different fits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_smooth_spline_criterion() {
    use p2a_core::regression::{SmoothSplineConfig, smooth_spline, smooth_spline::SplineCriterion};
    let case = Case::load("smoothers_gcv_periodic_loess");
    let ds = dataset_of(&case);
    let col = |n: &str| {
        p2a_core::data::columns::numeric_column(ds.df(), n, p2a_core::data::columns::NaPolicy::Fail)
            .unwrap()
            .0
    };
    let (x, y) = (col("x"), col("y"));
    let (session, server) = session_with_datasets(vec![("smoothers_t3", ds.clone())]);
    let mut fits = Vec::new();
    for (c, crit, label) in [
        (None, SplineCriterion::Gcv, "gcv"),
        (Some("cv"), SplineCriterion::Cv, "cv"),
        (Some("LOOCV"), SplineCriterion::Cv, "cv"),
    ] {
        let mut a = json!({"dataset": "smoothers_t3", "x": "x", "y": "y"});
        if let Some(s) = c {
            a["criterion"] = json!(s); // fails before: unknown field `criterion`
        }
        let j = json_of(&call(&server, &session, "regression_smooth_spline", a).await);
        let lib = smooth_spline(&x, &y, SmoothSplineConfig::with_criterion(crit)).unwrap();
        assert_eq!(j["criterion"], label);
        assert_eq!(
            (
                j["df"].as_f64(),
                j["lambda"].as_f64(),
                j["cv_crit"].as_f64()
            ),
            (Some(lib.df), Some(lib.lambda), lib.cv_crit)
        );
        fits.push(j["df"].as_f64().unwrap());
    }
    assert_ne!(fits[0], fits[1], "GCV and CV choose different smoothing");
    let j = json_of(
        &call(
            &server,
            &session,
            "regression_smooth_spline",
            json!({"dataset": "smoothers_t3", "x": "x", "y": "y", "df": 6.0}),
        )
        .await,
    );
    assert!(j["criterion"].is_null(), "{j}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "regression_smooth_spline",
            json!({"dataset": "smoothers_t3", "x": "x", "y": "y", "criterion": "aic"}),
        )
        .await,
    );
    assert!(e.contains("aic") && e.contains("gcv, cv"), "{e}");
}

/// step_legacy (regression.R): R's step() selections and printed criteria through MCP.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_step() {
    let case = Case::load("step_legacy");
    let n = 100;
    let (session, server) = session_with("step_legacy");
    for (direction, criterion, key) in [
        ("forward", "aic", "forward_aic"),
        ("backward", "BIC", "backward_bic"),
    ] {
        let j = json_of(
            &call(
                &server,
                &session,
                "regression_step",
                json!({"dataset": "step_legacy", "response": "y", "predictors": ["x1", "x2", "x3"],
                       "direction": direction, "criterion": criterion}),
            )
            .await,
        );
        assert_eq!(
            j["final_model"]["variables"],
            json!(["x1", "x2"]),
            "{direction}"
        );
        let last = j["steps"].as_array().unwrap().last().unwrap()["criterion"]
            .as_f64()
            .unwrap();
        Check::new(&case).scalar(key, last, Tol::stat(n)).finish();
    }
}

/// nls_expdecay (regression_smoothers_t3.R): stats::nls(y ~ a exp(-b x) + c), start (8, 0.3, 1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_regression_nls() {
    let case = Case::load("nls_expdecay");
    let (session, server) = session_with("nls_expdecay");
    for algorithm in ["LM", "gauss_newton"] {
        let t = text(
            &call(
                &server,
                &session,
                "regression_nls",
                json!({"dataset": "nls_expdecay", "x": "x", "y": "y", "model": "exponential_decay",
                       "start": [8.0, 0.3, 1.0], "algorithm": algorithm}),
            )
            .await,
        );
        for p in ["a", "b", "c"] {
            assert_row(
                &format!("{algorithm} {p}"),
                row(&t, p),
                &[case.named("coef", p), case.named("se", p)],
                Tol::printed(4),
            );
        }
    }
}

/// disc_mlogit_fishing (discrete_mlogit_asc.R): mlogit(mode ~ price + catch | income) with
/// J - 1 constants, the model without constants (asc = false) and reflevel = "pier"; pinned to
/// the tight MLE (*_tight).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_mlogit_asc() {
    let case = Case::load("disc_mlogit_fishing");
    let (session, server) = session_with("disc_mlogit_fishing"); // dataset "mlogit_fishing"
    let base = json!({"dataset": "mlogit_fishing", "choice_id": "id", "alt_id": "alt",
                      "choice": "chosen", "alt_specific": ["price", "catch"],
                      "ind_specific": ["income"]});
    let t = text(&call(&server, &session, "mlogit", base.clone()).await);
    assert!(t.contains("Alternative-specific constants: yes"), "{t}");
    for v in ["price", "catch"] {
        assert_row(
            v,
            row(&t, v),
            &[
                case.named("asc_coef_tight", v),
                case.named("asc_se_tight", v),
            ],
            Tol::printed(4),
        );
    }
    for alt in ["boat", "charter", "pier"] {
        let block = section(&t, &format!("{alt} vs beach"));
        let k = format!("(Intercept):{alt}");
        assert_row(
            &k,
            row(block, "(Intercept)"),
            &[
                case.named("asc_coef_tight", &k),
                case.named("asc_se_tight", &k),
            ],
            Tol::printed(4),
        );
    }
    assert_shows(
        "logLik",
        after(&t, "Log-Likelihood:"),
        case.scalar("asc_loglik"),
        Tol::printed(4),
    );
    let mut a = base.clone();
    a["asc"] = json!(false); // fails before: unknown field `asc`
    let t = text(&call(&server, &session, "mlogit", a).await);
    assert!(t.contains("Alternative-specific constants: no"), "{t}");
    for v in ["price", "catch"] {
        assert_row(
            v,
            row(&t, v),
            &[
                case.named("noasc_coef_tight", v),
                case.named("noasc_se_tight", v),
            ],
            Tol::printed(4),
        );
    }
    assert_shows(
        "logLik",
        after(&t, "Log-Likelihood:"),
        case.scalar("noasc_loglik"),
        Tol::printed(4),
    );
    let mut a = base.clone();
    a["reflevel"] = json!("pier");
    let t = text(&call(&server, &session, "mlogit", a).await);
    let k = "(Intercept):beach";
    assert_row(
        k,
        row(section(&t, "beach vs pier"), "(Intercept)"),
        &[
            case.named("pier_coef_tight", k),
            case.named("pier_se_tight", k),
        ],
        Tol::printed(4),
    );
    let mut a = base.clone();
    a["reflevel"] = json!("pier");
    a["reference"] = json!("boat");
    let e = tool_error(&call(&server, &session, "mlogit", a).await);
    assert!(e.contains("reflevel") && e.contains("reference"), "{e}");
}

/// disc_binary_glm: glm logit and probit, an aliased regressor (NA) and an outcome outside
/// {0, 1}. Logit pins the MLE to 1e-13 (logit_*); probit the tight MLE (probit_*_tight).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_logit_probit_binary_glm() {
    let case = Case::load("disc_binary_glm");
    let (session, server) = session_with("disc_binary_glm"); // dataset "binary_glm"; y, d Int64
    let x = ["x1", "x2", "d"];
    for (tool, coef, se) in [
        ("logit", "logit_coef", "logit_se"),
        ("probit", "probit_coef_tight", "probit_se_tight"),
    ] {
        let t = text(
            &call(
                &server,
                &session,
                tool,
                json!({"dataset": "binary_glm", "y": "y", "x": x}),
            )
            .await,
        );
        for v in ["(Intercept)", "x1", "x2", "d"] {
            assert_row(
                &format!("{tool} {v}"),
                row(&t, v),
                &[case.named(coef, v), case.named(se, v)],
                Tol::printed(4),
            );
        }
        assert!(t.contains("Converged: yes ("), "{t}");
    }
    let t = text(
        &call(
            &server,
            &session,
            "logit",
            json!({"dataset": "binary_glm", "y": "y", "x": ["x1", "x1b", "x2", "d"]}),
        )
        .await,
    );
    assert!(printed(row(&t, "x1b")).is_empty(), "{t}");
    assert!(
        t.contains("Coefficients not defined because of singularities: x1b"),
        "{t}"
    );
    assert_row(
        "x1",
        row(&t, "x1"),
        &[case.named("aliased_coef", "x1")],
        Tol::printed(4),
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "logit",
            json!({"dataset": "binary_glm", "y": "y012", "x": ["x1"]}),
        )
        .await,
    );
    assert!(e.contains("0/1"), "{e}");
}

/// disc_multinom_overlap: nnet::multinom on three overlapping categories (tight MLE);
/// reference A (default) and B.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_multinom_overlap() {
    let case = Case::load("disc_multinom_overlap");
    let (session, server) = session_with("disc_multinom_overlap"); // dataset "multinom_overlap"
    let t = text(
        &call(
            &server,
            &session,
            "multinom",
            json!({"dataset": "multinom_overlap", "y": "y", "x": ["x1", "x2"]}),
        )
        .await,
    );
    for cat in ["B", "C"] {
        let block = section(&t, &format!("--- {cat} vs A ---"));
        for v in ["(Intercept)", "x1", "x2"] {
            let k = format!("{cat}:{v}");
            assert_row(
                &k,
                row(block, v),
                &[case.named("coef_tight", &k), case.named("se_tight", &k)],
                Tol::printed(4),
            );
        }
    }
    let t = text(
        &call(
            &server,
            &session,
            "multinom",
            json!({"dataset": "multinom_overlap", "y": "y", "x": ["x1", "x2"], "reference": "B"}),
        )
        .await,
    );
    let k = "A:x1";
    assert_row(
        k,
        row(section(&t, "--- A vs B ---"), "x1"),
        &[
            case.named("refB_coef_tight", k),
            case.named("refB_se_tight", k),
        ],
        Tol::printed(4),
    );
}

/// disc_hurdle_zero: the binary part of pscl::hurdle is glm(I(y > 0) ~ x1 + x2, binomial).
/// The count part is ECON-16 (Tier 4) and is not pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_hurdle_zero_part() {
    let case = Case::load("disc_hurdle_zero");
    let (session, server) = session_with("disc_hurdle_zero"); // dataset "zeroinfl"
    let t = text(
        &call(
            &server,
            &session,
            "hurdle_model",
            json!({"dataset": "zeroinfl", "y": "y", "x": ["x1", "x2"]}),
        )
        .await,
    );
    let binary = section(&t, "Binary Part");
    for v in ["(Intercept)", "x1", "x2"] {
        assert_row(
            v,
            row(binary, v),
            &[case.named("zero_coef", v), case.named("zero_se_tight", v)],
            Tol::printed(4),
        );
    }
}

/// feglm_families (econ_panel_t3.R): fixest::feglm gaussian with id and time effects
/// (coefficients only; its SEs and deviance are ECON-20, Tier 4). The family is parsed
/// case-insensitively (before: "Gaussian" was "Unknown GLM family").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_feglm_gaussian() {
    let case = Case::load("feglm_families");
    let (session, server) = session_with("feglm_families"); // dataset "feglm_panel"
    let t = text(
        &call(
            &server,
            &session,
            "feglm",
            json!({"dataset": "feglm_panel", "y": "y_gauss", "x": ["x", "x2"],
                   "fe": ["id", "time"], "family": "Gaussian"}),
        )
        .await,
    );
    for v in ["x", "x2"] {
        assert_row(
            v,
            row(&t, v),
            &[case.named("gauss_coef", v)],
            Tol::printed(4),
        );
    }
}

/// Option strings and boundary parameters of the discrete tools.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_discrete_options_and_boundaries() {
    let (ml, zi, bg, fe) = (
        Case::load("disc_mixed_logit"),
        Case::load("disc_zeroinfl"),
        Case::load("disc_binary_glm"),
        Case::load("feglm_families"),
    );
    let (session, server) = session_with_datasets(vec![
        ("mixed_logit", dataset_of(&ml)),
        ("zeroinfl", dataset_of(&zi)),
        ("binary_glm", dataset_of(&bg)),
        ("feglm_panel", dataset_of(&fe)),
    ]);
    let mixed = json!({"dataset": "mixed_logit", "choice_id": "id", "alt_id": "alt",
                       "choice": "choice", "variables": ["x", "w"], "random_vars": ["x"],
                       "n_draws": 50});
    let with = |b: &Value, k: &str, v: Value| {
        let mut a = b.clone();
        a[k] = v;
        a
    };
    let e = tool_error(
        &call(
            &server,
            &session,
            "mixed_logit",
            with(&mixed, "n_draws", json!(0)),
        )
        .await,
    );
    assert!(e.contains("n_draws"), "{e}");
    // before: normal
    let e = tool_error(
        &call(
            &server,
            &session,
            "mixed_logit",
            with(&mixed, "distribution", json!("gamma")),
        )
        .await,
    );
    assert!(
        e.contains("gamma") && e.contains("normal, lognormal, triangular, uniform, fixed"),
        "{e}"
    );
    let r = call(
        &server,
        &session,
        "mixed_logit",
        with(&mixed, "distribution", json!("Normal")),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let ord = json!({"dataset": "binary_glm", "y": "y", "x": ["x1"]});
    let e = tool_error(
        &call(
            &server,
            &session,
            "ordered_model",
            with(&ord, "model_type", json!("cloglog")),
        )
        .await,
    );
    assert!(e.contains("cloglog") && e.contains("logit, probit"), "{e}");
    let zip = json!({"dataset": "zeroinfl", "y": "y", "x": ["x1", "x2"]});
    let e = tool_error(
        &call(
            &server,
            &session,
            "zeroinfl",
            with(&zip, "dist", json!("geometric")),
        )
        .await,
    );
    assert!(
        e.contains("geometric") && e.contains("poisson, negbin"),
        "{e}"
    );
    // "NB" is a new alias (before: "Invalid dist"); y_nb is the column of the pinned ZINB fit
    let nb = json!({"dataset": "zeroinfl", "y": "y_nb", "x": ["x1", "x2"], "dist": "NB"});
    let r = call(&server, &session, "zeroinfl", nb).await;
    assert!(r.success, "{}", text(&r));
    let e = tool_error(
        &call(
            &server,
            &session,
            "hurdle_model",
            with(&zip, "dist", json!("binomial")),
        )
        .await,
    );
    assert!(
        e.contains("binomial") && e.contains("poisson, negbin"),
        "{e}"
    );
    let fg = json!({"dataset": "feglm_panel", "y": "y_bin", "x": ["x"], "fe": ["id"]});
    let e = tool_error(
        &call(
            &server,
            &session,
            "feglm",
            with(&fg, "family", json!("gamma")),
        )
        .await,
    );
    assert!(
        e.contains("gamma") && e.contains("logit, probit, poisson, gaussian"),
        "{e}"
    );
}

/// Pseudo-random mixed-logit draws follow the session seed (before: always 42).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_mixed_logit_uses_session_seed() {
    let case = Case::load("disc_mixed_logit");
    let (session, server) = session_with("disc_mixed_logit"); // dataset "mixed_logit"
    let a = json!({"dataset": "mixed_logit", "choice_id": "id", "alt_id": "alt",
                   "choice": "choice", "variables": ["x", "w"], "random_vars": ["x"],
                   "n_draws": 20, "halton": false});
    let _ = &case;
    let mut runs = Vec::new();
    for seed in [5, 5, 6] {
        assert!(
            call(&server, &session, "set_seed", json!({"seed": seed}))
                .await
                .success
        );
        runs.push(text(
            &call(&server, &session, "mixed_logit", a.clone()).await,
        ));
    }
    assert_eq!(runs[0], runs[1]);
    assert_ne!(runs[0], runs[2]);
}

const COX_X: [&str; 3] = ["age", "sex", "ph.ecog"];

/// surv_cox_robust (surv_cox_aft.R): coxph Efron with robust (Lin-Wei) SEs, cluster(inst), and
/// Breslow ties, on lung (integer time, status 1/2, one missing ph.ecog and one missing inst).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_cox_robust() {
    let case = Case::load("surv_cox_robust");
    let (session, server) = session_with("surv_cox_robust"); // dataset "lung"
    let base = json!({"dataset": "lung", "time": "time", "event": "status", "covariates": COX_X});
    let with = |k: &str, v: Value| {
        let mut a = base.clone();
        a[k] = v;
        a
    };
    // fails before: unknown field `robust_se`
    let t = text(&call(&server, &session, "cox_ph", with("robust_se", json!(true))).await);
    for v in COX_X {
        assert_row(
            v,
            row(&t, v),
            &[
                case.named("efron_coef", v),
                case.named("efron_se", v),
                case.named("efron_se_robust", v),
            ],
            Tol::printed(4),
        );
    }
    assert!(
        t.contains("Standard errors: robust (Lin-Wei sandwich)"),
        "{t}"
    );
    assert_shows(
        "rscore",
        after(&t, "Robust score test ="),
        case.scalar("efron_rscore_robust"),
        Tol::printed(2),
    );
    let t = text(&call(&server, &session, "cox_ph", with("cluster", json!("inst"))).await);
    for v in COX_X {
        assert_row(
            v,
            row(&t, v),
            &[
                case.named("efron_coef_cluster", v),
                case.named("efron_se_model_cluster", v),
                case.named("efron_se_cluster", v),
            ],
            Tol::printed(4),
        );
    }
    assert!(t.contains("clustered by inst (18 clusters"), "{t}");
    // before: Efron
    let t = text(
        &call(
            &server,
            &session,
            "cox_ph",
            with("ties_method", json!("Breslow")),
        )
        .await,
    );
    for v in COX_X {
        assert_row(
            v,
            row(&t, v),
            &[case.named("breslow_coef", v), case.named("breslow_se", v)],
            Tol::printed(4),
        );
    }
}

/// Contradictory or invalid Cox settings are tool errors.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_cox_options_are_checked() {
    let case = Case::load("surv_cox_robust");
    let _ = &case;
    let (session, server) = session_with("surv_cox_robust");
    let base = json!({"dataset": "lung", "time": "time", "event": "status", "covariates": COX_X});
    let with = |pairs: &[(&str, Value)]| {
        let mut a = base.clone();
        for (k, v) in pairs {
            a[*k] = v.clone();
        }
        a
    };
    let e = tool_error(
        &call(
            &server,
            &session,
            "cox_ph",
            with(&[("robust_se", json!(false)), ("cluster", json!("inst"))]),
        )
        .await,
    );
    assert!(e.contains("cluster") && e.contains("robust"), "{e}");
    // before: Efron
    let e = tool_error(
        &call(
            &server,
            &session,
            "cox_ph",
            with(&[("ties_method", json!("exact"))]),
        )
        .await,
    );
    assert!(e.contains("exact") && e.contains("efron, breslow"), "{e}");
    let e = tool_error(&call(&server, &session, "cox_ph", with(&[("max_iter", json!(0))])).await);
    assert!(e.contains("max_iter"), "{e}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "cox_ph",
            with(&[("cluster", json!("nope"))]),
        )
        .await,
    );
    assert!(e.contains("nope"), "{e}");
}

/// surv_aft: survreg weibull, exponential (scale fixed), lognormal and loglogistic on lung
/// (227 complete rows).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_aft() {
    let case = Case::load("surv_aft");
    let (session, server) = session_with("surv_aft"); // dataset "lung"
    for (dist, tag) in [
        (None, "weibull"),
        (Some("exponential"), "exponential"),
        (Some("Log-Normal"), "lognormal"),
        (Some("log_logistic"), "loglogistic"),
    ] {
        let mut a =
            json!({"dataset": "lung", "time": "time", "event": "status", "covariates": COX_X});
        if let Some(d) = dist {
            a["distribution"] = json!(d);
        }
        let t = text(&call(&server, &session, "aft", a).await);
        for v in ["(Intercept)", "age", "sex", "ph.ecog"] {
            assert_row(
                &format!("{tag} {v}"),
                row(&t, v),
                &[
                    case.named(&format!("{tag}_coef"), v),
                    case.named(&format!("{tag}_se"), v),
                ],
                Tol::printed(4),
            );
        }
        assert_shows(
            "logLik",
            after(&t, "Log-Likelihood:"),
            case.scalar(&format!("{tag}_loglik")),
            Tol::printed(4),
        );
        if tag == "exponential" {
            assert!(t.contains("Scale fixed at 1"), "{t}");
        } else {
            assert_shows(
                "scale",
                after(&t, "Scale ="),
                case.scalar(&format!("{tag}_scale")),
                Tol::printed(4),
            );
        }
    }
    // before: Weibull
    let e = tool_error(
        &call(
            &server,
            &session,
            "aft",
            json!({"dataset": "lung", "time": "time", "event": "status", "covariates": COX_X,
                   "distribution": "gaussian"}),
        )
        .await,
    );
    assert!(
        e.contains("gaussian") && e.contains("weibull, exponential, lognormal, loglogistic"),
        "{e}"
    );
}

/// dpd_sys_balanced (econ_sysgmm.R): plm::pgmm(transformation = "ld") two-step with
/// Windmeijer SEs, one-step robust; one-step classical SEs are refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_panel_gmm_system() {
    let case = Case::load("dpd_sys_balanced");
    let (session, server) = session_with("dpd_sys_balanced"); // dataset "ab_balanced"
    let base = json!({"dataset": "ab_balanced", "y": "y", "x": ["x"], "entity_var": "id",
                      "time_var": "year", "transform": "system"});
    let with = |pairs: &[(&str, Value)]| {
        let mut a = base.clone();
        for (k, v) in pairs {
            a[*k] = v.clone();
        }
        a
    };
    let t = text(&call(&server, &session, "panel_gmm", base.clone()).await);
    let (c2, s2) = (
        case.vector("twostep_coef"),
        case.vector("twostep_se_robust"),
    );
    assert_row("L1.y", row(&t, "L1.y"), &[c2[0], s2[0]], Tol::printed(4));
    assert_row("x", row(&t, "x"), &[c2[1], s2[1]], Tol::printed(4));
    assert_shows(
        "n",
        after(&t, "No. Observations:"),
        case.scalar("twostep_n_obs"),
        Tol::exact(),
    );
    assert!(t.contains("600 differenced, 700 level"), "{t}");
    assert_shows(
        "Sargan",
        after(&t, "chi2(27) ="),
        case.scalar("twostep_sargan"),
        Tol::printed(4),
    );
    assert!(
        !t.contains("ECON-30") && !t.contains("Difference GMM (Arellano-Bond)"),
        "{t}"
    );
    let t = text(
        &call(
            &server,
            &session,
            "panel_gmm",
            with(&[
                ("transform", json!("ld")),
                ("step", json!("one-step")),
                ("robust", json!(true)),
            ]),
        )
        .await,
    );
    let (c1, s1) = (
        case.vector("onestep_coef"),
        case.vector("onestep_se_robust"),
    );
    assert_row(
        "L1.y one-step",
        row(&t, "L1.y"),
        &[c1[0], s1[0]],
        Tol::printed(4),
    );
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_gmm",
            with(&[("step", json!("onestep")), ("robust", json!(false))]),
        )
        .await,
    );
    assert!(e.contains("robust = true"), "{e}");
    // before: difference
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_gmm",
            with(&[("transform", json!("levels"))]),
        )
        .await,
    );
    assert!(
        e.contains("levels") && e.contains("difference, system"),
        "{e}"
    );
    // before: two-step
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_gmm",
            with(&[("step", json!("three"))]),
        )
        .await,
    );
    assert!(e.contains("three") && e.contains("onestep, twostep"), "{e}");
}

/// purtest_lag_selection (econ_purtest.R): per-unit AIC and SIC lags up to pmax = 4 as
/// plm::purtest; the IPS/LLC/MW statistics are ECON-41/43/44 (Tier 4) and are not pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_panel_unit_root_aic() {
    let case = Case::load("purtest_lag_selection");
    let (session, server) = session_with("purtest_lag_selection"); // dataset "purtest_panel"
    let base = json!({"dataset": "purtest_panel", "variable": "y", "unit_col": "id",
                      "time_col": "t", "test": "ips", "pmax": 4});
    let with = |pairs: &[(&str, Value)]| {
        let mut a = base.clone();
        for (k, v) in pairs {
            a[*k] = v.clone();
        }
        a
    };
    for (rule, key, label) in [
        ("aic", "ips_aic_lags", "AIC per unit, pmax = 4"),
        ("SIC", "ips_sic_lags", "SIC per unit, pmax = 4"),
    ] {
        // fails before: unknown field `lag_selection`
        let t = text(
            &call(
                &server,
                &session,
                "panel_unit_root",
                with(&[("lag_selection", json!(rule))]),
            )
            .await,
        );
        let lags = case.vector(key);
        let (min, max) = (
            lags.iter().cloned().fold(f64::INFINITY, f64::min),
            lags.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let mean = lags.iter().sum::<f64>() / lags.len() as f64; // 2.00 (AIC), 1.08 (SIC)
        assert!(t.contains(label), "{t}");
        assert_row(
            rule,
            after(&t, "per unit:"),
            &[min, max, mean],
            Tol::printed(2),
        );
    }
    let r = call(
        &server,
        &session,
        "panel_unit_root",
        with(&[("test", json!("madwu")), ("lag_selection", json!("hall"))]),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_unit_root",
            with(&[("lag_selection", json!("aic")), ("lags", json!(2))]),
        )
        .await,
    );
    assert!(e.contains("lags fixes the lag order"), "{e}");
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_unit_root",
            with(&[("lag_selection", json!("aic")), ("pmax", json!(0))]),
        )
        .await,
    );
    assert!(e.contains("pmax must be at least 1"), "{e}");
    // before: intercept
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_unit_root",
            with(&[("model", json!("constant"))]),
        )
        .await,
    );
    assert!(
        e.contains("constant") && e.contains("none, intercept, trend"),
        "{e}"
    );
    // before: LLC
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_unit_root",
            with(&[("test", json!("llc2"))]),
        )
        .await,
    );
    assert!(
        e.contains("llc2") && e.contains("llc, ips, madwu, hadri"),
        "{e}"
    );
}

/// pvcm_pmg_sim and pggls_sim (econ_panel_t3.R): Swamy random coefficients, mean group and
/// pooled FGLS coefficients (the MG and pooled-GLS SEs are ECON-34 and ECON-33, Tier 4).
/// panel_pmg accepts only model 'within'.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_panel_heterogeneous_and_gls() {
    let (pv, gl) = (Case::load("pvcm_pmg_sim"), Case::load("pggls_sim"));
    let (session, server) = session_with_datasets(vec![
        ("pvcm_panel", dataset_of(&pv)),
        ("pggls_panel", dataset_of(&gl)),
    ]);
    let pvcm = json!({"dataset": "pvcm_panel", "y": "y", "x": ["x1", "x2"], "entity_var": "id"});
    let with = |b: &Value, k: &str, v: Value| {
        let mut a = b.clone();
        a[k] = v;
        a
    };
    let names = ["(Intercept)", "x1", "x2"];
    let t = text(
        &call(
            &server,
            &session,
            "panel_pvcm",
            with(&pvcm, "model", json!("Random")),
        )
        .await,
    );
    for v in names {
        assert_row(
            v,
            row(&t, v),
            &[pv.named("random_coef", v), pv.named("random_se", v)],
            Tol::printed(4),
        );
    }
    let t = text(&call(&server, &session, "panel_pmg", pvcm.clone()).await);
    for v in names {
        assert_row(v, row(&t, v), &[pv.named("mg_coef", v)], Tol::printed(4));
    }
    // panel_pmg is the mean-group estimator; a model it does not run is refused (before: ignored)
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_pmg",
            with(&pvcm, "model", json!("random")),
        )
        .await,
    );
    assert!(e.contains("panel_pvcm"), "{e}");
    let within = text(
        &call(
            &server,
            &session,
            "panel_pmg",
            with(&pvcm, "model", json!("within")),
        )
        .await,
    );
    assert_eq!(within, t);
    // before: within
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_pvcm",
            with(&pvcm, "model", json!("pooling")),
        )
        .await,
    );
    assert!(e.contains("pooling") && e.contains("within, random"), "{e}");
    let g = json!({"dataset": "pggls_panel", "y": "y", "x": ["x1", "x2"], "entity_var": "id",
                   "time_var": "t"});
    let t = text(
        &call(
            &server,
            &session,
            "panel_gls",
            with(&g, "model", json!("pooling")),
        )
        .await,
    );
    for v in names {
        assert_row(
            v,
            row(&t, v),
            &[gl.named("pooling_coef", v)],
            Tol::printed(4),
        );
    }
    // before: fixed effects
    let e = tool_error(
        &call(
            &server,
            &session,
            "panel_gls",
            with(&g, "model", json!("re")),
        )
        .await,
    );
    assert!(e.contains("'re'") && e.contains("fe, pooling, fd"), "{e}");
}

/// panel_absorbed_regressors (econ_panel_t3.R): a time-invariant regressor is dropped by FE as
/// plm, and the MCP text shows the warning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_panel_fe_drops_absorbed() {
    let case = Case::load("panel_absorbed_regressors");
    let (session, server) = session_with("panel_absorbed_regressors"); // dataset "panel_absorbed"
    let t = text(
        &call(
            &server,
            &session,
            "panel_fixed_effects",
            json!({"dataset": "panel_absorbed", "y": "y", "x": ["x1", "x2", "z"], "entity_var": "id"}),
        )
        .await,
    );
    for v in ["x1", "x2"] {
        assert_row(
            v,
            row(&t, v),
            &[case.named("fe_coef", v), case.named("fe_se", v)],
            Tol::printed(4),
        );
    }
    assert!(
        t.contains("'z' dropped: no variation left after removing the 'id' fixed effects"),
        "{t}"
    );
    // before: "Unknown SE type 'classical'"
    let r = call(
        &server,
        &session,
        "panel_hdfe",
        json!({"dataset": "panel_absorbed", "y": "y", "x": ["x1", "x2"], "fe": ["id", "t"],
               "se_type": "classical"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
}
