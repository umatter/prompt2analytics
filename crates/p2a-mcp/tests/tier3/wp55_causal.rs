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
