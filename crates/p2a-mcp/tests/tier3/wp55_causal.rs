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
            "etwfe",
            with(&et, &[("vcov", json!("hc3"))]),
            vec!["vcov", "cluster", "iid"],
        ),
        (
            "etwfe",
            with(&et, &[("aggregation", json!("cohort_period"))]),
            vec!["aggregation", "simple", "calendar"],
        ),
        (
            "iv_mte",
            with(&mte, &[("propensity_model", json!("tobit"))]),
            vec!["probit", "logit"],
        ),
        (
            "iv_mte",
            with(&mte, &[("estimand", json!("ite"))]),
            vec!["mte_curve", "prte"],
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
            "treatment_double_ml",
            with(&otc, &[("model", json!("plm"))]),
            vec!["model", "plr", "pliv"],
        ),
        (
            "treatment_double_ml",
            with(&otc, &[("ml_method", json!("forest"))]),
            vec!["ml_method", "ols", "lasso", "ridge"],
        ),
        (
            "treatment_cbps",
            with(&tc, &[("method", json!("npcbps"))]),
            vec!["over", "exact", "logit"],
        ),
        (
            "treatment_cbps",
            with(&tc, &[("estimand", json!("ATC"))]),
            vec!["ATE", "ATT"],
        ),
        (
            "treatment_entropy_balance",
            with(&tc, &[("estimand", json!("ato"))]),
            vec!["ate", "atc"],
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
            "propensity_matching",
            with(&tc, &[("m_order", json!("closest"))]),
            vec!["m_order", "largest", "random"],
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
            "collaborative_tmle",
            with(&otc, &[("cv_criterion", json!("aic"))]),
            vec!["cv_criterion", "rss", "varic"],
        ),
        (
            "collaborative_tmle",
            with(&otc, &[("order", json!("random"))]),
            vec!["order", "forward", "prespecified"],
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
            "natural_effects_mediation",
            with(&ne, &[("family", json!("gamma"))]),
            vec!["family", "binomial"],
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
            "rd_multi",
            with(&rdm, &[("bwselect", json!("ik"))]),
            vec!["bwselect", "mserd"],
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
            "ltmle",
            with(&lt, &[("regime", json!("sometimes"))]),
            vec!["always", "dynamic"],
        ),
        (
            "ltmle",
            with(&lt, &[("contrast", json!("sometimes"))]),
            vec!["contrast", "none"],
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

/// etwfe::etwfe(ivar = id, cgroup, vcov) + emfx (the etwfe_staggered case) through the MCP.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_etwfe() {
    use p2a_core::econometrics::etwfe::{
        ControlGroup, EtwfeAggregation, EtwfeConfig, EtwfeVcov, run_etwfe,
    };
    let case = Case::load("etwfe_staggered");
    let (session, server) = session_with("etwfe_staggered");
    let ds = case_ds(&case);
    let base = json!({"dataset": "etwfe_panel", "outcome": "y", "unit_col": "id", "time_col": "year",
                      "treatment": "treat", "first_treat": "first_treat"});
    let lib = |cg: ControlGroup, v: EtwfeVcov| {
        let mut c = EtwfeConfig::default();
        c.cgroup = cg;
        c.vcov = v;
        run_etwfe(
            &ds,
            "y",
            "id",
            "year",
            "treat",
            "first_treat",
            None,
            Some(c),
        )
        .unwrap()
    };
    let dflt = lib(ControlGroup::NotYet, EtwfeVcov::Cluster);
    assert_eq!(
        ok(&server, &session, "etwfe", base.clone()).await,
        dflt.to_string()
    );
    let never = lib(ControlGroup::Never, EtwfeVcov::Cluster);
    let ev = never.aggregation(EtwfeAggregation::Event);
    let t = ok(
        &server,
        &session,
        "etwfe",
        with(
            &base,
            &[
                ("cgroup", json!("never_treated")),
                ("aggregation", json!("event")),
            ],
        ),
    )
    .await;
    assert!(
        t.starts_with(&never.aggregation_text(EtwfeAggregation::Event)),
        "{t}"
    );
    let iid = lib(ControlGroup::NotYet, EtwfeVcov::Iid);
    assert_eq!(
        ok(
            &server,
            &session,
            "etwfe",
            with(&base, &[("vcov", json!("IID"))])
        )
        .await,
        iid.to_string()
    );
    Check::new(&case)
        .vector(
            "notyet_emfx_simple_estimate",
            &[dflt.overall.estimate],
            Tol::coef(960),
        ) // 1.563937356319881
        .vector(
            "notyet_delta_simple_se",
            &[dflt.overall.std_error],
            Tol::se(960),
        ) // 0.1493020807661981
        .vector(
            "never_emfx_event_estimate",
            &ev.iter().map(|a| a.estimate).collect::<Vec<_>>(),
            Tol::coef(960),
        )
        .vector(
            "never_delta_event_se",
            &ev.iter().map(|a| a.std_error).collect::<Vec<_>>(),
            Tol::se(960),
        )
        .vector(
            "notyet_iid_delta_simple_se",
            &[iid.overall.std_error],
            Tol::se(960),
        ) // 0.1266790513180721
        .finish();
    let e = refused(
        &server,
        &session,
        "etwfe",
        with(&base, &[("tref", json!(42))]),
    )
    .await;
    assert!(e.contains("tref 42 is not a period of 'year'"), "{e}");
}

/// rdrobust::rdbwselect / rdrobust with selected bandwidths (the rd_bwselect cases) through rd_estimate, rd_bw,
/// rd_fuzzy and rd_multi; rdrobust's spellings and p2a's aliases.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_rd_bandwidth_selection() {
    use p2a_core::econometrics::rd::{
        BandwidthMethod, KernelType, RdConfig, rd_bandwidth_with, run_fuzzy_rd, run_rd,
    };
    let sharp = Case::load("rd_bwselect_sharp");
    let fuzzy = Case::load("rd_bwselect_fuzzy");
    let (session, server) = session_with("rd_bwselect_sharp"); // dataset `rd_sim`
    let ds = case_ds(&sharp);
    let base = json!({"dataset": "rd_sim", "outcome": "y", "running_var": "x", "cutoff": 0.0});
    let cfg = |m: BandwidthMethod, k: KernelType, sr: f64| {
        let mut c = RdConfig::default();
        c.bwselect = m;
        c.kernel = k;
        c.scaleregul = sr;
        c
    };
    let auto = run_rd(
        &ds,
        "y",
        "x",
        0.0,
        cfg(BandwidthMethod::MseRd, KernelType::Triangular, 1.0),
    )
    .unwrap();
    assert!(
        ok(&server, &session, "rd_estimate", base.clone())
            .await
            .starts_with(&auto.to_string())
    );
    let cer = run_rd(
        &ds,
        "y",
        "x",
        0.0,
        cfg(BandwidthMethod::CerRd, KernelType::Triangular, 1.0),
    )
    .unwrap();
    assert!(
        ok(
            &server,
            &session,
            "rd_estimate",
            with(&base, &[("bwselect", json!("CER"))])
        )
        .await
        .starts_with(&cer.to_string())
    );
    let noreg = run_rd(
        &ds,
        "y",
        "x",
        0.0,
        cfg(BandwidthMethod::MseRd, KernelType::Triangular, 0.0),
    )
    .unwrap();
    assert!(
        ok(
            &server,
            &session,
            "rd_estimate",
            with(&base, &[("scaleregul", json!(0.0))])
        )
        .await
        .starts_with(&noreg.to_string())
    );
    let two = rd_bandwidth_with(
        &ds,
        "y",
        "x",
        None,
        0.0,
        &cfg(BandwidthMethod::MseTwo, KernelType::Triangular, 1.0),
    )
    .unwrap();
    assert_eq!(
        ok(
            &server,
            &session,
            "rd_bw",
            with(
                &base,
                &[("bwselect", json!("mse-two")), ("kernel", json!("TRI"))]
            )
        )
        .await,
        two.to_string()
    );
    let epa = rd_bandwidth_with(
        &ds,
        "y",
        "x",
        None,
        0.0,
        &cfg(BandwidthMethod::CerRd, KernelType::Epanechnikov, 1.0),
    )
    .unwrap();
    assert_eq!(
        ok(
            &server,
            &session,
            "rd_bw",
            with(
                &base,
                &[("bwselect", json!("cerrd")), ("kernel", json!("epa"))]
            )
        )
        .await,
        epa.to_string()
    );
    let fz = run_fuzzy_rd(&ds, "yf", "x", "d", 0.0, RdConfig::default()).unwrap();
    assert!(
        ok(
            &server,
            &session,
            "rd_fuzzy",
            json!({"dataset": "rd_sim", "outcome": "yf", "running_var": "x", "treatment": "d"})
        )
        .await
        .starts_with(&fz.to_string())
    );
    let m = ok(&server, &session, "rd_multi",
               json!({"dataset": "rd_sim", "outcome": "y", "running_var": "x", "cutoffs": [0.0], "bwselect": "cerrd"})).await;
    assert!(
        m.contains(&format!(
            "{:>10.4} {:>10.4}",
            sharp.scalar("cerrd_coef_bc"),
            sharp.scalar("cerrd_se_robust")
        )),
        "{m}"
    );
    let n = 1000;
    Check::new(&sharp)
        .vector("auto_h", &[auto.h_left, auto.h_right], Tol::coef(n))
        .vector("auto_b", &[auto.b_left, auto.b_right], Tol::coef(n))
        .scalar("auto_coef_conv", auto.tau_conventional, Tol::coef(n))
        .scalar("auto_coef_bc", auto.tau_bc, Tol::coef(n))
        .scalar("auto_se_conv", auto.se_conventional, Tol::se(n))
        .scalar("auto_se_robust", auto.se_robust, Tol::se(n))
        .vector("cerrd_h", &[cer.h_left, cer.h_right], Tol::coef(n))
        .scalar("cerrd_coef_bc", cer.tau_bc, Tol::coef(n))
        .scalar("cerrd_se_robust", cer.se_robust, Tol::se(n))
        .vector("noreg_auto_h", &[noreg.h_left, noreg.h_right], Tol::coef(n))
        .scalar("noreg_auto_coef_bc", noreg.tau_bc, Tol::coef(n))
        .vector(
            "tri_p1_nn_msetwo",
            &[two.h_left, two.h_right, two.b_left, two.b_right],
            Tol::coef(n),
        )
        .vector(
            "epa_p1_nn_cerrd",
            &[epa.h_left, epa.h_right, epa.b_left, epa.b_right],
            Tol::coef(n),
        )
        .finish();
    Check::new(&fuzzy)
        .vector(
            "fuzzy_auto_h",
            &[fz.outcome_rd.h_left, fz.outcome_rd.h_right],
            Tol::coef(n),
        )
        .vector(
            "fuzzy_auto_b",
            &[fz.outcome_rd.b_left, fz.outcome_rd.b_right],
            Tol::coef(n),
        )
        .scalar("fuzzy_auto_coef_conv", fz.tau_fuzzy, Tol::coef(n))
        .scalar("fuzzy_auto_coef_bc", fz.tau_fuzzy_bc, Tol::coef(n))
        .scalar("fuzzy_auto_se_robust", fz.se_fuzzy_robust, Tol::se(n))
        .finish();
}

/// TOOL-4: x = [z2] enters the probit propensity and the second stage (the mte_mcp_covariate case); before,
/// x was dropped and the tool returned the single-instrument fit without it (`nox_*`). The R pins are the
/// first-stage fits at epsilon 1e-14 (`*_tight`); the tool runs the default tolerance, whose text is the
/// library Display of the same call and whose estimands lie within Tol::iterative(1e-6) of the tight fit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_iv_mte_uses_covariates() {
    use p2a_core::econometrics::ivmte::{IVMTEConfig, TreatmentEffectEstimate, run_ivmte};
    let case = Case::load("mte_mcp_covariate");
    let (session, server) = session_with("mte_mcp_covariate"); // dataset `mte_sim`
    let ds = case_ds(&case);
    let (y, d, z1) = (col(&ds, "y"), col(&ds, "d"), col(&ds, "z1"));
    let n = y.len();
    let x = Array2::from_shape_vec((n, 1), col(&ds, "z2").to_vec()).unwrap();
    let args = json!({"dataset": "mte_sim", "y": "y", "d": "d", "z": "z1", "x": ["z2"], "mte_degree": 3, "n_grid": 1001});
    let t = ok(&server, &session, "iv_mte", args.clone()).await;
    let mut c = IVMTEConfig::default();
    c.mte_degree = 3;
    c.n_grid = 1001;
    let without = run_ivmte(&y.view(), &d.view(), &z1.view(), None, c.clone()).unwrap();
    let mut cx = c.clone();
    cx.propensity_covariates = true;
    let lib = run_ivmte(
        &y.view(),
        &d.view(),
        &z1.view(),
        Some(&x.view()),
        cx.clone(),
    )
    .unwrap();
    assert!(t.starts_with(&lib.to_string()), "{t}");
    assert!(!t.starts_with(&without.to_string()));
    // the tight fits (glm epsilon 1e-14)
    let mut ct = cx.clone();
    ct.tolerance = 1e-14;
    let tight = run_ivmte(&y.view(), &d.view(), &z1.view(), Some(&x.view()), ct).unwrap();
    let mut c0t = c.clone();
    c0t.tolerance = 1e-14;
    let tight0 = run_ivmte(&y.view(), &d.view(), &z1.view(), None, c0t).unwrap();
    let te = |t: &TreatmentEffectEstimate| [t.estimate, t.se];
    let mut check = Check::new(&case);
    check
        .vector(
            "fs_coef_tight",
            &tight.first_stage.coefficients,
            Tol::iterative(1e-10),
        )
        // The two probit solutions agree to about 2e-11 relative; the cubic in P is nearly collinear and
        // amplifies that to about 5e-9 in the P^3 coefficient, while the estimands (combinations along
        // the well-determined directions) hold at Tol::coef. A documented iterative tolerance.
        .vector(
            "mte_coef_tight",
            &tight.second_stage_coefficients,
            Tol::iterative(1e-8),
        )
        .vector(
            "nox_fs_coef_tight",
            &tight0.first_stage.coefficients,
            Tol::iterative(1e-10),
        )
        .vector(
            "nox_mte_coef_tight",
            &tight0.second_stage_coefficients,
            Tol::iterative(1e-8),
        );
    for (k, e) in [
        ("ate_tight", &tight.ate),
        ("att_tight", &tight.att),
        ("atu_tight", &tight.atu),
        ("late_support_tight", &tight.late),
        ("nox_ate_tight", &tight0.ate),
        ("nox_att_tight", &tight0.att),
        ("nox_atu_tight", &tight0.atu),
        ("nox_late_support_tight", &tight0.late),
    ] {
        check
            .named_subset(k, &["est"], &["est", "se"], &te(e), Tol::coef(n))
            .named_subset(k, &["se"], &["est", "se"], &te(e), Tol::se(n));
    }
    // the tool's default first-stage tolerance (1e-8) against the tight fit; under HC1 the R default-control
    // fit is 2.0e-8 relative from it in the ATE estimate and 2.8e-7 in its SE
    for (k, e) in [
        ("ate_tight", &lib.ate),
        ("att_tight", &lib.att),
        ("atu_tight", &lib.atu),
        ("late_support_tight", &lib.late),
        ("nox_ate_tight", &without.ate),
        ("nox_att_tight", &without.att),
        ("nox_atu_tight", &without.atu),
        ("nox_late_support_tight", &without.late),
    ] {
        check.named(k, &["est", "se"], &te(e), Tol::iterative(1e-6));
    }
    check.finish();
    let e = refused(
        &server,
        &session,
        "iv_mte",
        with(&args, &[("x", json!(["no_such_column"]))]),
    )
    .await;
    assert!(e.contains("no_such_column"), "{e}");
    let r = refused(
        &server,
        &session,
        "iv_mte",
        with(&args, &[("x", json!(["z1"]))]),
    )
    .await;
    assert!(r.contains("x must not repeat y, d or z"), "{r}");
    let g = refused(
        &server,
        &session,
        "iv_mte",
        with(&args, &[("prte_weights", json!([1.0, 2.0]))]),
    )
    .await;
    assert!(g.contains("prte_weights"), "{g}");
    let b1 = ok(
        &server,
        &session,
        "iv_mte",
        with(
            &args,
            &[("bootstrap_reps", json!(20)), ("estimand", json!("att"))],
        ),
    )
    .await;
    let b2 = ok(
        &server,
        &session,
        "iv_mte",
        with(
            &args,
            &[("bootstrap_reps", json!(20)), ("estimand", json!("ATT"))],
        ),
    )
    .await;
    assert_eq!(b1, b2); // seeded
    assert!(b1.to_lowercase().contains("bootstrap"), "{b1}");
}

/// gmm::gmm(y ~ x1 + x2 + w, ~ z1 + z2 + z3 + w) (the gmm_linear_iv case) through the MCP.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gmm_iv() {
    use p2a_core::econometrics::general_gmm::{GeneralGmmConfig, GmmMethod, GmmVcov, run_gmm_iv};
    let case = Case::load("gmm_linear_iv");
    let (session, server) = session_with("gmm_linear_iv"); // dataset `gmm_iv`
    let ds = case_ds(&case);
    let (x, z) = (["x1", "x2", "w"], ["z1", "z2", "z3", "w"]);
    let base = json!({"dataset": "gmm_iv", "y": "y", "x": x, "z": z});
    let fit = |c: GeneralGmmConfig| run_gmm_iv(&ds, "y", &x, &z, Some(c)).unwrap();
    let cfg = |m: GmmMethod, v: GmmVcov| {
        let mut c = GeneralGmmConfig::default();
        c.method = m;
        c.vcov = v;
        c
    };
    let hac = fit(cfg(GmmMethod::TwoStep, GmmVcov::HAC));
    assert_eq!(
        ok(&server, &session, "gmm_iv", base.clone()).await,
        hac.to_string()
    );
    let mds = fit(cfg(GmmMethod::TwoStep, GmmVcov::MDS));
    assert_eq!(
        ok(
            &server,
            &session,
            "gmm_iv",
            with(&base, &[("vcov", json!("MDS"))])
        )
        .await,
        mds.to_string()
    );
    let cue = fit(cfg(GmmMethod::CUE, GmmVcov::MDS));
    assert_eq!(
        ok(
            &server,
            &session,
            "gmm_iv",
            with(
                &base,
                &[("method", json!("cue")), ("vcov", json!("robust"))]
            )
        )
        .await,
        cue.to_string()
    );
    // W = (Z'Z/n)^-1, Z = [1, z1, z2, z3, w]: gmm(weightsMatrix = W) is 2SLS with the sandwich variance
    let n = 500usize;
    let cols: Vec<Array1<f64>> = z.iter().map(|c| col(&ds, c)).collect();
    let zm = Array2::from_shape_fn((n, 5), |(i, j)| if j == 0 { 1.0 } else { cols[j - 1][i] });
    let w = p2a_core::linalg::solve::sym_inverse(
        (zm.t().dot(&zm) / n as f64).view(),
        p2a_core::linalg::solve::SingularPolicy::Error,
        "Z'Z/n",
    )
    .unwrap()
    .inverse;
    let mut fc = cfg(GmmMethod::TwoStep, GmmVcov::MDS);
    fc.weight_matrix = Some(w.clone());
    let fixed = fit(fc);
    let tf = ok(
        &server,
        &session,
        "gmm_iv",
        with(
            &base,
            &[
                ("vcov", json!("mds")),
                (
                    "weight_matrix",
                    json!(w.iter().copied().collect::<Vec<f64>>()),
                ),
            ],
        ),
    )
    .await;
    assert_eq!(tf, fixed.to_string());
    assert!(tf.contains("One-Step"), "{tf}");
    Check::new(&case)
        .vector("twostep_hac_bartlett_coef", &hac.coefficients, Tol::coef(n))
        .vector("twostep_hac_bartlett_se", &hac.std_errors, Tol::se(n))
        .vector("twostep_mds_coef", &mds.coefficients, Tol::coef(n))
        .vector("twostep_mds_se", &mds.std_errors, Tol::se(n))
        .scalar("twostep_mds_J", mds.j_stat, Tol::stat(n))
        .vector("cue_mds_coef", &cue.coefficients, Tol::iterative(1e-8))
        .vector("fixedw_mds_coef", &fixed.coefficients, Tol::coef(n))
        .vector("fixedw_mds_se", &fixed.std_errors, Tol::se(n))
        .finish();
    let e = refused(
        &server,
        &session,
        "gmm_iv",
        with(&base, &[("weight_matrix", json!(vec![0.0; 16]))]),
    )
    .await;
    assert!(e.contains("weight_matrix") && e.contains("5 x 5"), "{e}");
}

/// CBPS::CBPS(method, ATT) (the cbps_sim_methods tight refit) through the MCP; R's names.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_treatment_cbps() {
    use p2a_core::econometrics::cbps::{CbpsConfig, CbpsEstimand, CbpsMethod, run_cbps};
    let case = Case::load("cbps_sim_methods");
    let (session, server) = session_with("cbps_sim_methods"); // dataset `cbps_sim`
    let ds = case_ds(&case);
    let base =
        json!({"dataset": "cbps_sim", "treatment": "treat", "covariates": ["x1", "x2", "x3"]});
    let fit = |m: CbpsMethod, e: CbpsEstimand| {
        let mut c = CbpsConfig::default();
        c.method = m;
        c.estimand = e;
        c.balance_threshold = 0.1;
        c.tolerance = 1e-8;
        c.max_iter = 100;
        run_cbps(&ds, "treat", &["x1", "x2", "x3"], Some(c)).unwrap()
    };
    let over = fit(CbpsMethod::Over, CbpsEstimand::Ate);
    assert_eq!(
        ok(&server, &session, "treatment_cbps", base.clone()).await,
        over.to_string()
    );
    let exact = fit(CbpsMethod::Exact, CbpsEstimand::Att);
    assert_eq!(
        ok(
            &server,
            &session,
            "treatment_cbps",
            with(
                &base,
                &[("method", json!("exact")), ("estimand", json!("att"))]
            )
        )
        .await,
        exact.to_string()
    );
    Check::new(&case)
        .vector(
            "over_ate_tight_coef",
            &over.coefficients,
            Tol::iterative(1e-6),
        )
        .vector(
            "exact_att_tight_coef",
            &exact.coefficients,
            Tol::iterative(1e-6),
        )
        .finish();
}

/// Entropy balancing (exact dual), AIPW and the IPW M-estimation SE through the MCP; SBW refuses its unused
/// penalty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_weighting() {
    use p2a_core::econometrics::treatment::{
        DoublyRobustConfig, IpwConfig, run_doubly_robust, run_ipw_treatment,
    };
    use p2a_core::econometrics::weightit::{
        EntropyBalanceConfig, WeightEstimand, entropy_balance_with,
    };
    let ebal = Case::load("weights_ebal");
    let aipw = Case::load("ipw_aipw_sim");
    let (session, server) = session_with("weights_ebal"); // dataset `weighting_sim`
    let ds = case_ds(&ebal);
    let covs = ["x1", "x2", "x3"];
    let mut ec = EntropyBalanceConfig::default();
    ec.estimand = WeightEstimand::ATE;
    let eb = entropy_balance_with(&ds, "treat", &covs, &ec).unwrap();
    assert_eq!(ok(&server, &session, "treatment_entropy_balance",
                  json!({"dataset": "weighting_sim", "treatment": "treat", "covariates": covs, "estimand": "ATE"})).await,
               eb.to_string());
    let mut dc = DoublyRobustConfig::default();
    dc.trim = 0.0;
    dc.bootstrap = 999;
    dc.seed = None; // the tool's defaults
    let dr = run_doubly_robust(&ds, "y", "treat", &covs, dc).unwrap();
    let t = ok(&server, &session, "treatment_doubly_robust",
               json!({"dataset": "weighting_sim", "outcome": "y", "treatment": "treat", "covariates": covs, "trim": 0.0})).await;
    assert_eq!(t, dr.to_string());
    assert!(t.contains("analytic influence function"), "{t}");
    let mut ic = IpwConfig::default();
    ic.trim = 0.0;
    ic.bootstrap = 0;
    ic.normalized = true;
    let ipw = run_ipw_treatment(&ds, "y", "treat", &covs, ic).unwrap();
    let ti = ok(&server, &session, "treatment_ipw",
                json!({"dataset": "weighting_sim", "outcome": "y", "treatment": "treat", "covariates": covs, "trim": 0.0})).await;
    assert!(ti.starts_with(&ipw.to_string()), "{ti}");
    assert!(ti.contains("M-estimation"), "{ti}");
    let s = refused(&server, &session, "treatment_sbw",
                    json!({"dataset": "weighting_sim", "treatment": "treat", "covariates": covs, "balance_penalty": 50.0})).await;
    assert!(s.contains("balance_penalty"), "{s}");
    let s = ok(
        &server,
        &session,
        "treatment_sbw",
        json!({"dataset": "weighting_sim", "treatment": "treat", "covariates": covs}),
    )
    .await;
    assert!(!s.contains("balance_penalty"), "{s}");
    Check::new(&ebal)
        .scalar("ate_exact_ess_t", eb.ess_treated, Tol::stat(300))
        .scalar("ate_exact_ess_c", eb.ess_control, Tol::stat(300))
        .finish();
    // the propensity is an IRLS fit
    Check::new(&aipw)
        .scalar("aipw_ate", dr.effect, Tol::iterative(1e-8))
        .scalar("aipw_ate_se", dr.std_error, Tol::iterative(1e-8))
        .scalar("ipw_ate_hajek", ipw.effect, Tol::iterative(1e-8))
        .scalar("ipw_ate_se_mest", ipw.std_error, Tol::se(300))
        .finish();
}

/// MatchIt's defaults through the MCP: optimal full matching without bounds and CEM with Sturges bins
/// (the matchit_full_cem case).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_propensity_matching() {
    use p2a_core::econometrics::matching::{MatchMethod, MatchOptions, match_it_with};
    let case = Case::load("matchit_full_cem");
    let (session, server) = session_with("matchit_full_cem"); // dataset `matching_sim`
    let ds = case_ds(&case);
    let covs = ["x1", "x2", "x3"];
    let base = json!({"dataset": "matching_sim", "treatment": "treat", "covariates": covs});
    // the tool passes a distance only when one is given
    let full = match_it_with(
        &ds,
        "treat",
        &covs,
        MatchMethod::Full {
            min_ratio: 0.0,
            max_ratio: f64::INFINITY,
        },
        None,
        &MatchOptions::default(),
    )
    .unwrap();
    let tf = ok(
        &server,
        &session,
        "propensity_matching",
        with(
            &base,
            &[("method", json!("optimal")), ("output_name", json!("fm"))],
        ),
    )
    .await;
    assert!(tf.starts_with(&full.to_string()), "{tf}");
    let cem = match_it_with(
        &ds,
        "treat",
        &covs,
        MatchMethod::CoarsenedExact {
            cutpoints: None,
            n_bins: None,
        },
        None,
        &MatchOptions::default(),
    )
    .unwrap();
    let tc = ok(
        &server,
        &session,
        "propensity_matching",
        with(&base, &[("method", json!("CEM")), ("ratio", json!(3))]),
    )
    .await;
    assert!(
        tc.starts_with(&cem.to_string()) && tc.contains("Sturges bins"),
        "{tc}"
    );
    assert!(
        tc.contains("Warning: ratio is not used with method 'cem'"),
        "{tc}"
    );
    Check::new(&case)
        .scalar(
            "full_tight_total_distance",
            full.total_distance.unwrap(),
            Tol::iterative(1e-8),
        ) // 0.8659137461118286
        .scalar(
            "cem_n_matched_t",
            cem.n_matched_treated as f64,
            Tol::exact(),
        ) // 68
        .scalar(
            "cem_n_matched_c",
            cem.n_matched_control as f64,
            Tol::exact(),
        ) // 90
        .scalar(
            "cem_n_strata",
            cem.subclasses.as_ref().unwrap().len() as f64,
            Tol::exact(),
        ) // 33
        .finish();
    // MatchIt's tol-0.001 value 0.8700895944690001
    assert!(full.total_distance.unwrap() < case.scalar("full_total_distance"));
}

/// TOOL-15: the matched sample is registered as MatchIt::match.data builds it (matched units with distance,
/// weights, subclass), and the text holds both balance tables (the matchit_mcp_output case).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_matching_output_dataset() {
    let case = Case::load("matchit_mcp_output");
    let nearest = Case::load("matchit_nearest");
    let (session, server) = session_with("matchit_mcp_output"); // dataset `matching_sim`
    let base =
        json!({"dataset": "matching_sim", "treatment": "treat", "covariates": ["x1", "x2", "x3"]});
    let t = ok(&server, &session, "propensity_matching", base.clone()).await;
    for want in [
        "Balance Before Matching",
        "Balance After Matching",
        "eCDF Max",
        "Matching order: largest",
        "Matched sample saved as 'matching_sim_matched': 178 of 240 rows",
    ] {
        assert!(t.contains(want), "missing {want:?} in\n{t}");
    }
    assert!(!t.contains("KS statistics"));
    let m = session
        .datasets
        .read()
        .await
        .get("matching_sim_matched")
        .cloned()
        .expect("matched sample registered");
    let df = m.df();
    let keep: Vec<usize> = case
        .vector("r1_weights")
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(df.height(), keep.len()); // 178
    let get = |c: &str| numeric_column(df, c, NaPolicy::Fail).unwrap().0;
    let sub_r: Vec<f64> = keep
        .iter()
        .map(|&i| case.vector("r1_subclass")[i])
        .collect();
    let ps_r: Vec<f64> = keep
        .iter()
        .map(|&i| nearest.vector("logit_ps_tight")[i])
        .collect();
    assert_eq!(get("weights"), vec![1.0; keep.len()]);
    assert_eq!(get("subclass"), sub_r); // 71, 87, 1, 2, 77, 3, 48, 4, 7, 41, ...
    for (a, b) in get("distance").iter().zip(&ps_r) {
        assert!(Tol::iterative(1e-8).allows(*a, *b), "{a} vs {b}");
    }
    assert_eq!(get("id").len(), keep.len()); // input columns kept
    // 1:2 rounds: every unit is matched; MatchIt's weights (0.848314606741573, 1, 1.696629213483146)
    ok(
        &server,
        &session,
        "propensity_matching",
        with(&base, &[("ratio", json!(2)), ("output_name", json!("m2"))]),
    )
    .await;
    let m2 = session.datasets.read().await.get("m2").cloned().unwrap();
    let w2 = numeric_column(m2.df(), "weights", NaPolicy::Fail)
        .unwrap()
        .0;
    let s2 = numeric_column(m2.df(), "subclass", NaPolicy::Fail)
        .unwrap()
        .0;
    Check::new(&case)
        .vector("r2_weights", &w2, Tol::coef(240))
        .vector("r2_subclass", &s2, Tol::exact())
        .finish();
    let e = refused(
        &server,
        &session,
        "propensity_matching",
        with(&base, &[("output_name", json!("matching_sim"))]),
    )
    .await;
    assert!(e.contains("output_name"), "{e}");
}

/// medflex::neImpute + neModel on the odds-ratio scale with A:M (the medflex_scales case, odds_int), robust
/// sandwich SEs (n_bootstrap = 0).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_natural_effects_mediation() {
    use p2a_core::econometrics::medflex::{
        EffectScale, MedflexConfig, MedflexFamily, run_medflex_dataset,
    };
    let case = Case::load("medflex_scales");
    let (session, server) = session_with("medflex_scales"); // dataset `medflex_sim`
    let ds = case_ds(&case);
    let t = ok(
        &server,
        &session,
        "natural_effects_mediation",
        json!({"dataset": "medflex_sim", "outcome": "yb",
        "treatment": "a", "mediator": "m", "confounders": ["c1", "c2"], "allow_interaction": true,
        "scale": "OR", "family": "binomial", "n_bootstrap": 0}),
    )
    .await;
    let mut c = MedflexConfig::default();
    c.allow_interaction = true;
    c.bootstrap_ci = false;
    c.n_bootstrap = 0;
    c.scale = EffectScale::OddsRatio;
    c.family = Some(MedflexFamily::Binomial);
    c.confidence_level = 0.95;
    let lib = run_medflex_dataset(&ds, "yb", "a", "m", &["c1", "c2"], c).unwrap();
    assert_eq!(t, lib.to_string());
    // IRLS working model and quasi-GLM natural effect model
    Check::new(&case)
        .scalar(
            "odds_int_nde",
            lib.natural_direct_effect,
            Tol::iterative(1e-8),
        )
        .scalar(
            "odds_int_nie",
            lib.natural_indirect_effect,
            Tol::iterative(1e-8),
        )
        .scalar("odds_int_te", lib.total_effect, Tol::iterative(1e-8))
        .finish();
    assert!(t.contains("robust"), "{t}");
    let args = |k: &str| {
        json!({"dataset": "medflex_sim", "outcome": "yc", "treatment": "a", "mediator": "m",
                                "covariates": ["c1", "c2"], k: 60})
    };
    let w = ok(&server, &session, "mediation_analysis", args("bootstrap")).await;
    assert_eq!(
        w,
        ok(&server, &session, "mediation_analysis", args("n_bootstrap")).await
    ); // alias, same seed
}

/// ltmle's defaults written out (the ltmle_two_period case): static always/never, the dynamic rule
/// A_t = 1{L_{t-1} > 0}. The tool runs the default configuration, pinned against the tight values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ltmle() {
    use p2a_core::econometrics::ltmle::{LtmleConfig, LtmleData, Regime, run_ltmle};
    let case = Case::load("ltmle_two_period");
    let (session, server) = session_with("ltmle_two_period"); // dataset `ltmle_sim`
    let ds = case_ds(&case);
    let base = json!({"dataset": "ltmle_sim", "outcomes": ["Y", "Y"], "treatments": ["A1", "A2"], "covariates": ["L0", "L1"]});
    let data = LtmleData::from_dataset(&ds, "Y", &["A1", "A2"], &[&["L0"], &["L1"]]).unwrap();
    let lib = run_ltmle(&data, LtmleConfig::default()).unwrap(); // always vs never, logistic Q, gbounds (0.01, 1)
    assert_eq!(
        ok(&server, &session, "ltmle", base.clone()).await,
        lib.to_string()
    );
    assert!(lib.converged);
    let mut dc = LtmleConfig::default();
    dc.intervention = Regime::threshold_rules(&["L0".to_string(), "L1".to_string()], 0.0);
    let dyn_ = run_ltmle(&data, dc).unwrap();
    assert_eq!(
        ok(
            &server,
            &session,
            "ltmle",
            with(
                &base,
                &[
                    ("regime", json!("dynamic")),
                    ("dynamic_columns", json!(["L0", "L1"])),
                    ("dynamic_threshold", json!(0.0))
                ]
            )
        )
        .await,
        dyn_.to_string()
    );
    Check::new(&case)
        .scalar("psi_always", lib.psi_treated, Tol::iterative(1e-8))
        .scalar("psi_never", lib.psi_control, Tol::iterative(1e-8))
        .scalar("ate_se", lib.se, Tol::iterative(1e-8))
        .scalar("psi_dynamic", dyn_.psi_treated, Tol::iterative(1e-8)) // 0.45392016719826284
        .scalar("ate_dynamic_vs_never", dyn_.ate, Tol::iterative(1e-8)) // 0.12833200065831912
        .finish();
    let none = ok(
        &server,
        &session,
        "ltmle",
        with(&base, &[("contrast", json!("none"))]),
    )
    .await;
    assert!(!none.contains("Contrast:"), "{none}");
    let e = refused(
        &server,
        &session,
        "ltmle",
        with(&base, &[("dynamic_columns", json!(["L0", "L1"]))]),
    )
    .await;
    assert!(e.contains("dynamic_columns"), "{e}");
    let same = refused(
        &server,
        &session,
        "ltmle",
        with(&base, &[("contrast", json!("always"))]),
    )
    .await;
    assert!(same.contains("regime and contrast are the same"), "{same}");
    let y_cov = refused(
        &server,
        &session,
        "ltmle",
        with(&base, &[("covariates", json!(["L0", "L1,Y"]))]),
    )
    .await;
    assert!(
        y_cov.contains("'Y'") && y_cov.contains("outcome"),
        "{y_cov}"
    );
    let a_cov = refused(
        &server,
        &session,
        "ltmle",
        with(&base, &[("covariates", json!(["L0,A1", "L1"]))]),
    )
    .await;
    assert!(a_cov.contains("'A1'"), "{a_cov}");
}

/// The pooled logistic hazard g-formula (causal_gformula.R, gformula_hazard): point treatment A replicated
/// over 5 periods, time-fixed L and Lb. The tool runs the hazard model at its default control; the pins are
/// the R solution (`risk_always`, `risk_never`, `hazard_coef`), from which the default control is within
/// 7.8e-10 relative.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gformula() {
    use p2a_core::econometrics::gformula::{
        GFormulaConfig, GFormulaData, GFormulaIntervention, GFormulaOutcomeType, run_gformula,
    };
    let case = Case::load("gformula_hazard");
    let (session, server) = session_with("gformula_hazard"); // dataset `gformula_surv`
    let ds = case_ds(&case);
    let n = ds.df().height();
    let args = |iv: &str| {
        json!({"dataset": "gformula_surv", "outcome": "event", "baseline_covariates": ["L", "Lb"],
        "time_varying_covariates": [], "treatment_cols": ["A", "A", "A", "A", "A"], "time_points": 5,
        "outcome_type": "survival", "time_col": "time", "intervention": iv, "n_bootstrap": 20, "seed": 7})
    };
    let base = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            col(&ds, "L")[i]
        } else {
            col(&ds, "Lb")[i]
        }
    });
    let a = col(&ds, "A");
    let data = GFormulaData::new_survival(
        base,
        vec![Array2::zeros((n, 0)); 5],
        vec![a; 5],
        col(&ds, "time"),
        col(&ds, "event"),
    )
    .unwrap()
    .with_names(&["L", "Lb"], &[], "A")
    .unwrap();
    let run = |treat_all: bool| {
        let mut c = GFormulaConfig::default();
        c.time_points = 5;
        c.outcome_type = GFormulaOutcomeType::Survival;
        c.n_bootstrap = 20;
        c.seed = Some(7);
        c.intervention = GFormulaIntervention::Static { treat_all };
        c.max_iter = 100;
        c.tolerance = 1e-8;
        run_gformula(&data, c).unwrap()
    };
    let (r1, r0) = (run(true), run(false));
    assert_eq!(
        ok(&server, &session, "gformula", args("always")).await,
        r1.to_string()
    );
    assert_eq!(
        ok(&server, &session, "gformula", args("never_treat")).await,
        r0.to_string()
    );
    assert!(r1.all_models_converged && r0.all_models_converged);
    let hm = r1.outcome_model.as_ref().unwrap();
    Check::new(&case)
        .vector(
            "risk_always",
            &r1.risk_curve_intervention,
            Tol::iterative(1e-8),
        )
        .vector(
            "risk_never",
            &r0.risk_curve_intervention,
            Tol::iterative(1e-8),
        )
        .named(
            "hazard_coef",
            &hm.names,
            &hm.coefficients,
            Tol::iterative(1e-8),
        )
        .finish();
    let e = refused(
        &server,
        &session,
        "gformula",
        with(&args("always"), &[("outcome_type", json!("binary"))]),
    )
    .await;
    assert!(e.contains("time_col"), "{e}");
    let mut no_time = args("always");
    no_time.as_object_mut().unwrap().remove("time_col");
    let e = refused(&server, &session, "gformula", no_time).await;
    assert!(e.contains("needs time_col"), "{e}");
}

/// C-TMLE with a real 5-fold CV risk (causal_stdreg.R, ctmle_cv), VarIC and RSS criteria. The MCP runs
/// `CTmleConfig::default()` (glm epsilon 1e-8), the case epsilon 1e-14: iterative tolerances.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_collaborative_tmle() {
    use p2a_core::econometrics::ctmle::{
        CTmleConfig, CTmleQModel, CVCriterion, SelectionOrder, ctmle,
    };
    let case = Case::load("ctmle_cv");
    let (session, server) = session_with("ctmle_cv"); // dataset `ctmle_sim`
    let ds = case_ds(&case);
    let w = ["w1", "w2", "w3", "w4"];
    let base = json!({"dataset": "ctmle_sim", "outcome": "yb", "treatment": "a", "covariates": w, "q_model": "binary"});
    let fit = |cr: CVCriterion, order: SelectionOrder| {
        let mut c = CTmleConfig::default();
        c.q_model = CTmleQModel::Logistic;
        c.cv_criterion = cr;
        c.order = order;
        c.gbound = (0.025, 0.975);
        ctmle(&ds, "yb", "a", &w, c).unwrap()
    };
    let (varic, rss) = (
        fit(CVCriterion::VarIC, SelectionOrder::Forward),
        fit(CVCriterion::RSS, SelectionOrder::Forward),
    );
    assert_eq!(
        ok(
            &server,
            &session,
            "collaborative_tmle",
            with(&base, &[("cv_criterion", json!("var-ic"))])
        )
        .await,
        varic.to_string()
    );
    assert_eq!(
        ok(
            &server,
            &session,
            "collaborative_tmle",
            with(
                &base,
                &[("cv_criterion", json!("rss")), ("order", json!("fwd"))]
            )
        )
        .await,
        rss.to_string()
    );
    let back = fit(CVCriterion::RSS, SelectionOrder::Backward);
    assert_eq!(
        ok(
            &server,
            &session,
            "collaborative_tmle",
            with(&base, &[("order", json!("backward"))])
        )
        .await,
        back.to_string()
    );
    let pre = fit(CVCriterion::RSS, SelectionOrder::Prespecified(Vec::new()));
    assert_eq!(
        ok(
            &server,
            &session,
            "collaborative_tmle",
            with(&base, &[("order", json!("prespecified"))])
        )
        .await,
        pre.to_string()
    );
    Check::new(&case)
        .scalar("logit_varic_ate", varic.ate, Tol::iterative(1e-7))
        .scalar("logit_varic_se", varic.se, Tol::iterative(1e-6))
        .scalar("logit_rss_ate", rss.ate, Tol::iterative(1e-7))
        .scalar("logit_rss_se", rss.se, Tol::iterative(1e-6))
        .finish();
    assert!(
        varic.selected_covariates.is_empty(),
        "{:?}",
        varic.selected_covariate_names
    ); // mask [0,0,0,0]
    assert_eq!(
        (
            rss.selected_covariates.clone(),
            rss.selected_covariate_names.clone()
        ),
        (vec![0], vec!["w1".to_string()])
    ); // [1,0,0,0]
}

/// DoubleML scores with the case's fixed folds (causal_doubleml.R, doubleml_models): PLIV with OLS
/// learners, PLR with lasso.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_treatment_double_ml() {
    use p2a_core::econometrics::doubleml::{
        DMLModelType, DoubleMLConfig, MLMethod, run_double_ml, run_double_ml_iv,
    };
    let case = Case::load("doubleml_models");
    let (session, server) = session_with("doubleml_models"); // dataset `doubleml_sim`
    let ds = case_ds(&case);
    let n = ds.df().height();
    let xs = ["X1", "X2", "X3", "X4", "X5", "X6"];
    let x = Array2::from_shape_fn((n, 6), |(i, j)| col(&ds, xs[j])[i]);
    let folds: Vec<usize> = col(&ds, "fold").iter().map(|v| *v as usize).collect();
    let cfg = |m: DMLModelType, l: MLMethod| {
        let mut c = DoubleMLConfig::default();
        c.model_type = m;
        c.ml_method = l;
        c.folds = Some(folds.clone());
        c
    };
    let pliv = run_double_ml_iv(
        &col(&ds, "yz").view(),
        &col(&ds, "dz").view(),
        &col(&ds, "z").view(),
        &x.view(),
        cfg(DMLModelType::PLIV, MLMethod::OLS),
    )
    .unwrap();
    let base =
        json!({"dataset": "doubleml_sim", "outcome": "yz", "treatment": "dz", "covariates": xs});
    assert_eq!(
        ok(
            &server,
            &session,
            "treatment_double_ml",
            with(
                &base,
                &[
                    ("model", json!("partially_linear_iv")),
                    ("instrument", json!("z")),
                    ("fold_column", json!("fold"))
                ]
            )
        )
        .await,
        pliv.to_string()
    );
    let mut lc = cfg(DMLModelType::PLR, MLMethod::Lasso);
    lc.lambda = 0.05;
    let lasso = run_double_ml(
        &col(&ds, "y").view(),
        &col(&ds, "dcont").view(),
        &x.view(),
        lc,
    )
    .unwrap();
    assert_eq!(
        ok(
            &server,
            &session,
            "treatment_double_ml",
            json!({"dataset": "doubleml_sim", "outcome": "y", "treatment": "dcont", "covariates": xs,
                   "model_type": "plr", "ml_method": "L1", "lambda": 0.05, "fold_column": "fold"})
        )
        .await,
        lasso.to_string()
    );
    Check::new(&case)
        .scalar("pliv_theta", pliv.theta, Tol::coef(n))
        .scalar("pliv_se", pliv.se, Tol::se(n))
        .scalar("plr_lasso_theta_tight", lasso.theta, Tol::new(1e-9, 1e-12))
        .scalar("plr_lasso_se_tight", lasso.se, Tol::new(1e-9, 1e-12))
        .finish();
    let e = refused(
        &server,
        &session,
        "treatment_double_ml",
        with(
            &base,
            &[("model", json!("pliv")), ("fold_column", json!("X1"))],
        ),
    )
    .await;
    assert!(e.contains("needs instrument"), "{e}");
    let e = refused(
        &server,
        &session,
        "treatment_double_ml",
        with(&base, &[("instrument", json!("z"))]),
    )
    .await;
    assert!(e.contains("only used with pliv and iivm"), "{e}");
    let e = refused(
        &server,
        &session,
        "treatment_double_ml",
        with(
            &base,
            &[
                ("model", json!("pliv")),
                ("instrument", json!("z")),
                ("fold_column", json!("X1")),
            ],
        ),
    )
    .await;
    assert!(
        e.contains("fold_column 'X1' must hold non-negative whole numbers"),
        "{e}"
    );
}

/// gsynth 1.4.0 (causal_gsynth.R, gsynth_simdata): IFE r = 2 without covariates (closed form), the MCP
/// default force 'unit', and the leave-one-out CV choice of r with covariates.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_gsynth() {
    use p2a_core::econometrics::synth::{GsynthConfig, GsynthForce, run_gsynth};
    let case = Case::load("gsynth_simdata");
    let (session, server) = session_with("gsynth_simdata"); // dataset `gsynth_simdata`
    let ds = case_ds(&case);
    let base = json!({"dataset": "gsynth_simdata", "outcome": "Y", "treatment": "D", "unit_col": "id", "time_col": "time"});
    let fit = |f: GsynthForce, x: &[&str], cv: bool| {
        let mut c = GsynthConfig::default();
        c.n_factors = if cv { 0 } else { 2 };
        c.max_factors = 5;
        c.cross_validate = cv;
        c.force = f;
        run_gsynth(&ds, "Y", "D", "id", "time", x, c).unwrap()
    };
    let tw = fit(GsynthForce::TwoWay, &[], false);
    assert_eq!(
        ok(
            &server,
            &session,
            "gsynth",
            with(&base, &[("force", json!("two-way"))])
        )
        .await,
        tw.to_string()
    );
    let unit = fit(GsynthForce::Unit, &[], false);
    assert_eq!(
        ok(&server, &session, "gsynth", base.clone()).await,
        unit.to_string()
    );
    let cv = fit(GsynthForce::TwoWay, &["X1", "X2"], true);
    assert_eq!(
        ok(
            &server,
            &session,
            "gsynth",
            with(
                &base,
                &[
                    ("force", json!("twoWay")),
                    ("covariates", json!(["X1", "X2"])),
                    ("n_factors", json!(0)),
                    ("cv", json!(true)),
                    ("r_max", json!(5))
                ]
            )
        )
        .await,
        cv.to_string()
    );
    Check::new(&case)
        .scalar("ife_r2_nocov_att_avg", tw.att, Tol::coef(1500)) // 4.639592652031497
        .scalar("ife_r2_unit_att_avg", unit.att, Tol::coef(1500)) // 4.87256592159725
        // the core cross-validates leave-one-out (fect cv.method = "loo"); gsynth 1.4's random-fold
        // choice (`ife_cv_cov_r`, 1) is recorded in the case, not pinned
        .scalar("ife_loo_cov_r", cv.n_factors as f64, Tol::exact()) // 2
        .finish();
    let e = refused(
        &server,
        &session,
        "gsynth",
        with(
            &base,
            &[
                ("bootstrap_se", json!(false)),
                ("bootstrap_reps", json!(20)),
            ],
        ),
    )
    .await;
    assert!(
        e.contains("bootstrap_se = false contradicts bootstrap_reps = 20"),
        "{e}"
    );
    let e = refused(
        &server,
        &session,
        "gsynth",
        with(&base, &[("r_max", json!(5)), ("max_factors", json!(4))]),
    )
    .await;
    assert!(e.contains("former name of r_max"), "{e}");
}

/// TOOL-4 duplicate: contour_data returns sensemakr's adjusted-estimate grid of the classical lm fit
/// (causal_mcp_surface.R, sensemakr_mcp_contour).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_sensemakr_contour_data() {
    let case = Case::load("sensemakr_mcp_contour");
    let (session, server) = session_with("sensemakr_mcp_contour"); // dataset `sensemakr_darfur`
    let args = json!({"dataset": "sensemakr_darfur", "y": "peacefactor", "treatment": "directlyharmed",
        "covariates": ["age", "farmer_dar", "herder_dar", "pastvoted", "hhsize_darfur", "female"], "contour_data": true});
    let t = ok(&server, &session, "sensemakr", args.clone()).await;
    let block = t.split("Contour data").nth(1).expect("contour section");
    let json_start = block.find('{').unwrap();
    let v: Value = serde_json::from_str(block[json_start..].lines().next().unwrap()).unwrap();
    let flat: Vec<f64> = v["adjusted_estimates"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
        .collect();
    let grid: Vec<f64> = v["r2dz_x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let grid_y: Vec<f64> = v["r2yz_dx"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    Check::new(&case)
        .vector("grid", &grid, Tol::coef(1276))
        .vector("grid", &grid_y, Tol::coef(1276))
        .vector("adjusted_estimates", &flat, Tol::coef(1276))
        .finish();
    let off = ok(
        &server,
        &session,
        "sensemakr",
        with(&args, &[("contour_data", json!(false))]),
    )
    .await;
    assert!(!off.contains("Contour data"), "{off}");
}

/// TOOL-14: impossible parameter values are tool errors naming the parameter, never successful
/// estimates. The core validates; the MCP returns the message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_causal_degenerate_parameters_are_tool_errors() {
    let (rs, rv) = session_with("rd_bwselect_sharp"); // dataset `rd_sim`
    let rd = json!({"dataset": "rd_sim", "outcome": "y", "running_var": "x", "cutoff": 0.0});
    for (tool, extra, needle) in [
        (
            "rd_estimate",
            json!({"h": -1.0}),
            "Main bandwidth h must be a finite number greater than 0",
        ),
        (
            "rd_estimate",
            json!({"h": 0.3, "b": 0.0}),
            "Bias bandwidth b must be a finite number greater than 0",
        ),
        (
            "rd_estimate",
            json!({"scaleregul": -1.0}),
            "scaleregul (the regularisation scale of the bandwidth selector) must be a finite",
        ),
        (
            "rd_fuzzy",
            json!({"treatment": "d", "h": -0.5}),
            "Main bandwidth h must be",
        ),
        (
            "rd_multi",
            json!({"cutoffs": [0.0, 0.5], "bandwidths": [0.3, -1.0]}),
            "The bandwidth for cutoff 2 must be",
        ),
        (
            "rd_multi",
            json!({"cutoffs": [0.0], "bandwidth": 0.0}),
            "The global bandwidth must be",
        ),
        ("rd_multi", json!({"cutoffs": []}), "cutoff"),
        (
            "rd_estimate",
            json!({"level": 1.5}),
            "Confidence level must be strictly between 0 and 1",
        ),
    ] {
        let mut a = rd.clone();
        if tool == "rd_multi" {
            // rd_multi takes `cutoffs` only
            a.as_object_mut().unwrap().remove("cutoff");
        }
        for (k, v) in extra.as_object().unwrap() {
            a[k] = v.clone();
        }
        let t = refused(&rv, &rs, tool, a).await;
        assert!(t.contains(needle), "{tool}: {needle:?} not in {t:?}");
    }
    let (ms, mv) = session_with("mte_local_iv"); // dataset `mte_sim`
    for g in [0, 1] {
        let t = refused(
            &mv,
            &ms,
            "iv_mte",
            json!({"dataset": "mte_sim", "y": "y", "d": "d", "z": "z1", "n_grid": g}),
        )
        .await;
        assert!(t.contains("n_grid"), "n_grid = {g}: {t}");
    }
    let (es, ev) = session_with("etwfe_staggered");
    let t = refused(
        &ev,
        &es,
        "etwfe",
        json!({"dataset": "etwfe_panel", "outcome": "y", "unit_col": "id", "time_col": "year",
               "treatment": "treat", "first_treat": "first_treat", "gref": 5}),
    )
    .await;
    assert!(t.contains("gref"), "{t}");
    let (ls, lv) = session_with("ltmle_two_period");
    let t = refused(
        &lv,
        &ls,
        "ltmle",
        json!({"dataset": "ltmle_sim", "outcomes": ["Y", "Y"], "treatments": ["A1", "A2"],
               "covariates": ["L0", "L1"], "ps_lower": 0.5, "ps_upper": 0.4}),
    )
    .await;
    assert!(t.contains("gbounds") || t.contains("ps_lower"), "{t}");
    // no dataset: a missing baseline risk is a tool error, not a protocol error
    let t = refused(
        &lv,
        &ls,
        "evalue",
        json!({"effect_type": "rd", "point": 0.1, "se": 0.02}),
    )
    .await;
    assert!(t.contains("baseline_risk"), "{t}");
}

/// Drift guard: every option field's schema description lists each canonical spelling of the core
/// table it is parsed with, quoted as 'name'.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_causal_option_descriptions_list_accepted_values() {
    use p2a_core::econometrics::*;
    let (_session, server) = session_with_datasets(vec![]);
    let tools = server.router_tools();
    let desc = |tool: &str, field: &str| -> String {
        let t = tools
            .iter()
            .find(|t| t.name == tool)
            .unwrap_or_else(|| panic!("{tool}"));
        t.input_schema["properties"][field]["description"]
            .as_str()
            .unwrap_or_else(|| panic!("{tool}.{field}"))
            .to_string()
    };
    let rows: Vec<(&str, &str, Vec<&str>)> = vec![
        (
            "staggered_did",
            "comparison_group",
            staggered_did::ComparisonGroup::choices().names(),
        ),
        (
            "staggered_did",
            "estimation_method",
            staggered_did::AttEstimationMethod::choices().names(),
        ),
        ("etwfe", "cgroup", etwfe::ControlGroup::choices().names()),
        ("etwfe", "vcov", etwfe::EtwfeVcov::choices().names()),
        (
            "etwfe",
            "aggregation",
            etwfe::EtwfeAggregation::choices().names(),
        ),
        (
            "rd_estimate",
            "bwselect",
            rd::BandwidthMethod::choices().names(),
        ),
        (
            "rd_multi",
            "bwselect",
            rd::BandwidthMethod::choices().names(),
        ),
        ("rd_estimate", "kernel", rd::KernelType::choices().names()),
        ("iv_mte", "estimand", ivmte::MTEEstimand::choices().names()),
        (
            "iv_mte",
            "propensity_model",
            ivmte::PropensityModel::choices().names(),
        ),
        ("gmm_iv", "vcov", general_gmm::GmmVcov::choices().names()),
        (
            "treatment_cbps",
            "method",
            cbps::CbpsMethod::choices().names(),
        ),
        (
            "treatment_cbps",
            "estimand",
            cbps::CbpsEstimand::choices().names(),
        ),
        (
            "treatment_weightit",
            "method",
            weightit::WeightMethod::choices().names(),
        ),
        (
            "treatment_doubly_robust",
            "method",
            treatment::DRMethod::choices().names(),
        ),
        (
            "propensity_matching",
            "method",
            matching::MatchMethodKind::choices().names(),
        ),
        (
            "propensity_matching",
            "distance",
            matching::DistanceMethod::choices().names(),
        ),
        (
            "propensity_matching",
            "m_order",
            matching::MOrder::choices().names(),
        ),
        (
            "natural_effects_mediation",
            "scale",
            medflex::EffectScale::choices().names(),
        ),
        (
            "natural_effects_mediation",
            "family",
            medflex::MedflexFamily::choices().names(),
        ),
        ("ltmle", "regime", ltmle::RegimeKind::choices().names()),
        (
            "gformula",
            "outcome_type",
            gformula::GFormulaOutcomeType::choices().names(),
        ),
        (
            "gformula",
            "intervention",
            gformula::GFormulaRegime::choices().names(),
        ),
        (
            "collaborative_tmle",
            "cv_criterion",
            ctmle::CVCriterion::choices().names(),
        ),
        (
            "collaborative_tmle",
            "q_model",
            ctmle::CTmleQModel::choices().names(),
        ),
        (
            "treatment_double_ml",
            "model",
            doubleml::DMLModelType::choices().names(),
        ),
        (
            "treatment_double_ml",
            "ml_method",
            doubleml::MLMethod::choices().names(),
        ),
        (
            "gsynth",
            "estimator",
            synth::GsynthEstimator::choices().names(),
        ),
        ("gsynth", "force", synth::GsynthForce::choices().names()),
        (
            "scpi",
            "variance_method",
            scpi::VarianceMethod::choices().names(),
        ),
    ];
    let mut missing = Vec::new();
    for (tool, field, names) in rows {
        let d = desc(tool, field);
        for n in names {
            if !d.contains(&format!("'{n}'")) {
                missing.push(format!("{tool}.{field}: '{n}' missing in {d:?}"));
            }
        }
    }
    assert!(missing.is_empty(), "{}", missing.join("\n"));
    let pm = tools
        .iter()
        .find(|t| t.name == "propensity_matching")
        .unwrap();
    assert!(
        !pm.description
            .as_deref()
            .unwrap_or("")
            .contains("KS statistics")
    );
    let cb = tools.iter().find(|t| t.name == "treatment_cbps").unwrap();
    assert!(
        !cb.description
            .as_deref()
            .unwrap_or("")
            .contains("overidentified)")
    );
}
