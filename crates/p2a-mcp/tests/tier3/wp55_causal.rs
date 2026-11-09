//! T3-WP55 MCP-path tests (causal surface). Owned by T3-WP55.
//! Each test drives `call_tool_with_session`. An R pin computes the library result for the configuration the
//! request must map to, checks it against the case (Tier 1 tolerances), and requires the tool text to be that
//! result's text.
#![allow(dead_code, unused_imports)]
use crate::common::*;
use ndarray::{Array1, Array2};
use p2a_core::data::columns::{NaPolicy, numeric_column};
use p2a_core::data::{DataLoader, Dataset};
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

fn case_ds(case: &Case) -> Dataset {
    Dataset::new(DataLoader::load_csv(case.dataset_path()).expect("case dataset"))
}
fn col(ds: &Dataset, name: &str) -> Array1<f64> {
    Array1::from(numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0)
}
async fn ok(server: &AnalyticsServer, session: &Session, tool: &str, args: Value) -> String {
    let r = call(server, session, tool, args).await;
    let t = text(&r);
    assert!(r.success, "{tool} failed: {t}");
    t
}
async fn refused(server: &AnalyticsServer, session: &Session, tool: &str, args: Value) -> String {
    let r = call(server, session, tool, args).await;
    let t = text(&r);
    assert!(!r.success, "{tool} should be refused, got: {t}");
    t
}
fn with(base: &Value, pairs: &[(&str, Value)]) -> Value {
    let mut a = base.clone();
    for (k, v) in pairs {
        a[*k] = v.clone();
    }
    a
}

/// One row per option field of the causal tools: the tool, a request on the missing dataset `nope`
/// with one unknown option value, and what the refusal must name (the option and canonical spellings).
fn unknown_option_rows() -> Vec<(&'static str, Value, Vec<&'static str>)> {
    let did = json!({"dataset": "nope", "outcome": "y", "treatment_time": "g", "time_col": "t", "unit_col": "i"});
    let et = json!({"dataset": "nope", "outcome": "y", "unit_col": "i", "time_col": "t", "treatment": "d",
                    "first_treat": "g"});
    let mte = json!({"dataset": "nope", "y": "y", "d": "d", "z": "z"});
    let otc = json!({"dataset": "nope", "outcome": "y", "treatment": "d", "covariates": ["x"]});
    let tc = json!({"dataset": "nope", "treatment": "d", "covariates": ["x"]});
    let gf = json!({"dataset": "nope", "outcome": "y", "baseline_covariates": ["b"], "time_varying_covariates": ["l"],
                    "treatment_cols": ["a"], "time_points": 1});
    let ne = json!({"dataset": "nope", "outcome": "y", "treatment": "d", "mediator": "m"});
    let sc = json!({"dataset": "nope", "outcome": "y", "unit_col": "i", "time_col": "t", "treated_unit": "1",
                    "treatment_time": 5, "predictors": [{"column": "x"}]});
    let gs = json!({"dataset": "nope", "outcome": "y", "treatment": "d", "unit_col": "i", "time_col": "t"});
    let scpi = json!({"dataset": "nope", "outcome": "y", "unit_col": "i", "time_col": "t", "treated_unit": "1",
                      "treatment_time": 5});
    let rd = json!({"dataset": "nope", "outcome": "y", "running_var": "x"});
    let rdf = json!({"dataset": "nope", "outcome": "y", "running_var": "x", "treatment": "d"});
    let rdm = json!({"dataset": "nope", "outcome": "y", "running_var": "x", "cutoffs": [0.0, 1.0]});
    let gmm = json!({"dataset": "nope", "y": "y", "x": ["x"], "z": ["z"]});
    let lt =
        json!({"dataset": "nope", "outcomes": ["y"], "treatments": ["a"], "covariates": ["l"]});
    let me = json!({"dataset": "nope", "y": "y", "x": ["x"]});
    let mut sc_bad = sc.clone();
    sc_bad["predictors"] = json!([{"column": "x", "aggregation": "median"}]);
    vec![
        (
            "staggered_did",
            with(&did, &[("comparison_group", json!("sometimes"))]),
            vec!["comparison_group", "never_treated", "not_yet_treated"],
        ),
        (
            "staggered_did",
            with(&did, &[("estimation_method", json!("tmle"))]),
            vec!["estimation_method", "outcome_regression", "doubly_robust"],
        ),
        (
            "etwfe",
            with(&et, &[("cgroup", json!("always"))]),
            vec!["cgroup", "notyet", "never"],
        ),
        (
            "iv_mte",
            with(&mte, &[("propensity_model", json!("tobit"))]),
            vec!["probit", "logit"],
        ),
        (
            "treatment_ipw",
            with(&otc, &[("estimand", json!("atc"))]),
            vec!["estimand", "ate", "att"],
        ),
        (
            "treatment_doubly_robust",
            with(&otc, &[("method", json!("tmle"))]),
            vec!["aipw", "regression"],
        ),
        (
            "treatment_doubly_robust",
            with(&otc, &[("estimand", json!("ato"))]),
            vec!["ate", "att"],
        ),
        (
            "treatment_double_ml",
            with(&otc, &[("model_type", json!("plm"))]),
            vec!["plr", "iivm"],
        ),
        (
            "treatment_cbps",
            with(&tc, &[("method", json!("npcbps"))]),
            vec!["over", "exact", "logit"],
        ),
        (
            "treatment_weightit",
            with(&tc, &[("method", json!("cbps"))]),
            vec!["logistic", "energy"],
        ),
        (
            "treatment_weightit",
            with(&tc, &[("estimand", json!("ato"))]),
            vec!["ate", "atc"],
        ),
        (
            "treatment_sbw",
            with(&tc, &[("estimand", json!("ato"))]),
            vec!["att", "atc"],
        ),
        (
            "treatment_twang",
            with(&tc, &[("stop_method", json!("ks.median"))]),
            vec!["stop_method", "es.mean", "ks.max"],
        ),
        (
            "treatment_twang",
            with(&tc, &[("estimand", json!("ato"))]),
            vec!["estimand", "att"],
        ),
        (
            "propensity_matching",
            with(&tc, &[("method", json!("genetic"))]),
            vec!["nearest", "subclass"],
        ),
        (
            "propensity_matching",
            with(&tc, &[("distance", json!("gam"))]),
            vec!["logit", "mahalanobis"],
        ),
        (
            "treatment_tmle",
            with(&otc, &[("q_model", json!("poisson"))]),
            vec!["q_model", "logistic", "linear"],
        ),
        (
            "collaborative_tmle",
            with(&otc, &[("stopping_rule", json!("aic"))]),
            vec!["cv_minimum", "one_se"],
        ),
        (
            "gformula",
            with(&gf, &[("intervention", json!("sometimes"))]),
            vec!["always_treat", "natural_course"],
        ),
        (
            "gformula",
            with(&gf, &[("outcome_type", json!("count"))]),
            vec!["outcome_type", "survival"],
        ),
        (
            "regression_standardization",
            with(&otc, &[("model_type", json!("gamma"))]),
            vec!["linear", "poisson"],
        ),
        (
            "regression_standardization",
            with(&otc, &[("estimand", json!("ato"))]),
            vec!["ate", "levels"],
        ),
        (
            "regression_standardization",
            with(&otc, &[("se_method", json!("jackknife"))]),
            vec!["bootstrap", "sandwich"],
        ),
        (
            "natural_effects_mediation",
            with(&ne, &[("scale", json!("log"))]),
            vec!["difference", "odds_ratio"],
        ),
        (
            "synthetic_control",
            with(&sc, &[("v_method", json!("random"))]),
            vec!["datadriven", "equal"],
        ),
        ("synthetic_control", sc_bad, vec!["mean", "sum"]),
        (
            "gsynth",
            with(&gs, &[("estimator", json!("em"))]),
            vec!["ife", "mc"],
        ),
        (
            "gsynth",
            with(&gs, &[("force", json!("three-way"))]),
            vec!["unit", "two_way"],
        ),
        (
            "scpi",
            with(&scpi, &[("constraint", json!("elastic"))]),
            vec!["simplex", "lasso_simplex"],
        ),
        (
            "scpi",
            with(&scpi, &[("variance_method", json!("bootstrap"))]),
            vec!["subgaussian", "kfold_cv"],
        ),
        (
            "rd_estimate",
            with(&rd, &[("kernel", json!("gaussian"))]),
            vec!["kernel", "triangular"],
        ),
        (
            "rd_estimate",
            with(&rd, &[("bwselect", json!("ik"))]),
            vec!["bwselect", "mserd", "cercomb2"],
        ),
        (
            "rd_bw",
            with(&rd, &[("bwselect", json!("ik"))]),
            vec!["bwselect", "msetwo"],
        ),
        (
            "rd_fuzzy",
            with(&rdf, &[("kernel", json!("cosine"))]),
            vec!["kernel", "epanechnikov"],
        ),
        (
            "rd_multi",
            with(&rdm, &[("pooling_weights", json!("median"))]),
            vec!["pooling_weights", "sample_size", "inverse_variance"],
        ),
        (
            "gmm_iv",
            with(&gmm, &[("method", json!("gel"))]),
            vec!["twostep", "cue"],
        ),
        (
            "gmm_iv",
            with(&gmm, &[("vcov", json!("cluster"))]),
            vec!["hac", "mds"],
        ),
        (
            "gmm_iv",
            with(&gmm, &[("kernel", json!("gauss"))]),
            vec!["bartlett", "quadratic_spectral"],
        ),
        (
            "ltmle",
            with(&lt, &[("q_model", json!("probit"))]),
            vec!["logistic", "linear"],
        ),
        (
            "marginal_effects",
            with(&me, &[("model", json!("tobit"))]),
            vec!["model", "ols", "probit"],
        ),
        (
            "evalue",
            json!({"effect_type": "hr2", "point": 1.5}),
            vec!["effect_type", "rr", "smd"],
        ),
    ]
}

/// CAUS-97: the R (did, etwfe), CLI and MCP spellings give the same fit, and anything else is refused before
/// the data are read, naming the option and the accepted values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_causal_option_aliases_and_unknown() {
    let (session, server) = session_with("did_cs_covariates"); // dataset `cs_cov_panel`
    let base = json!({"dataset": "cs_cov_panel", "outcome": "y", "treatment_time": "first_treat", "time_col": "time",
                      "unit_col": "unit", "covariates": ["x1", "x2"], "bootstrap": 0});
    let nyt = ok(
        &server,
        &session,
        "staggered_did",
        with(&base, &[("comparison_group", json!("not_yet_treated"))]),
    )
    .await;
    for s in ["notyettreated", "Not-Yet-Treated", "notyet", "nyt"] {
        assert_eq!(
            ok(
                &server,
                &session,
                "staggered_did",
                with(&base, &[("comparison_group", json!(s))])
            )
            .await,
            nyt,
            "{s}"
        );
    }
    assert_ne!(
        ok(
            &server,
            &session,
            "staggered_did",
            with(&base, &[("comparison_group", json!("never"))])
        )
        .await,
        nyt
    );
    let dr = ok(
        &server,
        &session,
        "staggered_did",
        with(&base, &[("estimation_method", json!("dr"))]),
    )
    .await;
    for s in ["DR", "doubly-robust", "doubly_robust", "aipw", "drdid"] {
        assert_eq!(
            ok(
                &server,
                &session,
                "staggered_did",
                with(&base, &[("estimation_method", json!(s))])
            )
            .await,
            dr,
            "{s}"
        );
    }
    assert_ne!(
        ok(
            &server,
            &session,
            "staggered_did",
            with(&base, &[("estimation_method", json!("reg"))])
        )
        .await,
        dr
    );
    let (es, ev) = session_with("etwfe_staggered"); // dataset `etwfe_panel`
    let eb = json!({"dataset": "etwfe_panel", "outcome": "y", "unit_col": "id", "time_col": "year",
                    "treatment": "treat", "first_treat": "first_treat"});
    let never = ok(&ev, &es, "etwfe", with(&eb, &[("cgroup", json!("never"))])).await;
    for s in ["never_treated", "NeverTreated", "Never Treated"] {
        assert_eq!(
            ok(&ev, &es, "etwfe", with(&eb, &[("cgroup", json!(s))])).await,
            never,
            "{s}"
        );
    }
    assert_ne!(ok(&ev, &es, "etwfe", eb.clone()).await, never);
    for (tool, args, needles) in unknown_option_rows() {
        let t = refused(&server, &session, tool, args).await;
        assert!(
            !t.contains("not found"),
            "{tool}: the option must be refused before the dataset lookup: {t}"
        );
        for n in needles {
            assert!(t.contains(n), "{tool}: {n:?} missing in {t:?}");
        }
    }
}

/// did::att_gt(xformla = ~ x1 + x2, control_group = "notyettreated", est_method = "dr", anticipation = 1,
/// bstrap = FALSE) as written out in the did_cs_covariates case, through the MCP spellings of R.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_staggered_did() {
    use p2a_core::econometrics::staggered_did::{
        AttEstimationMethod, ComparisonGroup, StaggeredDidConfig, run_staggered_did,
    };
    let case = Case::load("did_cs_covariates");
    let (session, server) = session_with("did_cs_covariates");
    let t = ok(&server, &session, "staggered_did", json!({"dataset": "cs_cov_panel", "outcome": "y",
        "treatment_time": "first_treat", "time_col": "time", "unit_col": "unit", "covariates": ["x1", "x2"],
        "comparison_group": "notyettreated", "estimation_method": "DR", "anticipation": 1, "bootstrap": 0})).await;
    let ds = case_ds(&case);
    let mut c = StaggeredDidConfig::default();
    c.comparison_group = ComparisonGroup::NotYetTreated;
    c.estimation_method = AttEstimationMethod::DoublyRobust;
    c.anticipation = 1;
    c.bootstrap = 0;
    c.base_period = -1;
    let lib = run_staggered_did(
        &ds,
        "y",
        "first_treat",
        "time",
        "unit",
        Some(&["x1", "x2"]),
        c,
    )
    .unwrap();
    // The DR estimate contains a propensity IRLS: the written-out transcription is the tight reference.
    Check::new(&case)
        .scalar(
            "notyet_ant1_simple_att_dr",
            lib.overall_att.att,
            Tol::iterative(1e-8),
        ) // 1.9729398081306126
        .scalar(
            "notyet_ant1_simple_se_dr",
            lib.overall_att.std_error,
            Tol::iterative(1e-8),
        ) // 0.16520145652426557
        .finish();
    assert!(
        t.starts_with(&lib.to_string()),
        "tool text is not the result's Display:\n{t}"
    );
    assert!(t.contains("Anticipation:       1 period(s)"), "{t}");
    let e = refused(&server, &session, "staggered_did", json!({"dataset": "cs_cov_panel", "outcome": "y",
        "treatment_time": "first_treat", "time_col": "time", "unit_col": "unit", "anticipation": 8, "bootstrap": 0})).await;
    assert!(
        e.contains("anticipation") && e.contains("smaller than the number of periods (8)"),
        "{e}"
    );
}
