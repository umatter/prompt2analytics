//! Tier 4: the causal effect commands through the p2a CLI (clustered DiD, seeded bootstraps,
//! trimmed IPW units).

#[path = "tier3/common.rs"]
mod common;

use common::*;
use serde_json::Value;

/// The JSON field `key` of `v` as a number.
fn number(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("`{key}` is not a number: {v}"))
}

/// `causal did ... -x tenure --cluster employee_id` matches lm + sandwich::vcovCL (CR1) for the
/// interaction (case did_2x2_training). Fails before: the command had no --cluster flag.
#[test]
fn test_validate_cli_causal_did_cluster() {
    let case = Case::load("did_2x2_training");
    let s = temp_session();
    s.load_as(&case.dataset_path(), "training", &[]);
    let args = [
        "causal",
        "did",
        "training",
        "-y",
        "performance",
        "--treat",
        "training_group",
        "--post",
        "post_training",
        "-x",
        "tenure",
        "--cluster",
        "employee_id",
    ];
    let j = s.json(&args);
    let term = ["training_group:post_training"];
    let n = case.scalar("tenure_df_resid") as usize + 5;
    Check::new(&case)
        .named_subset(
            "tenure_coef",
            &term,
            &term,
            &[number(&j, "att")],
            Tol::coef(n),
        )
        .named_subset(
            "tenure_se_cl",
            &term,
            &term,
            &[number(&j, "std_error")],
            Tol::se(n),
        )
        .scalar("n_clusters", number(&j, "n_clusters"), Tol::exact())
        .finish();
    assert_eq!(j["cluster_var"], "employee_id", "{j}");
    let txt = s.ok(&args);
    let nc = case.scalar("n_clusters");
    assert!(
        txt.contains(&format!("Clustered SEs: employee_id ({nc} clusters)")),
        "{txt}"
    );
    // without --cluster, no cluster is reported
    let j = s.json(&args[..args.len() - 2]);
    assert!(
        j["cluster_var"].is_null() && j["n_clusters"].is_null(),
        "{j}"
    );
}

/// Every bootstrapped causal command is seeded: two runs without --seed agree (default 42),
/// and --seed changes the draws. Fails before: stdreg and staggered-did drew unseeded random
/// numbers, and ipw, doubly-robust and mediation (whose core default seed was already 42) had
/// no --seed flag.
#[test]
fn test_cli_causal_seed_flags() {
    let s = temp_session();
    let generated = |f: &str| repo_path(&format!("validation/datasets/generated/{f}"));
    s.load_as(&generated("weighting_sim.csv"), "w", &[]);
    s.load_as(&generated("mediation_huber_sim.csv"), "m", &[]);
    s.load_as(&generated("stdreg_sim.csv"), "s", &[]);
    s.load_as(&generated("cs_panel.csv"), "cs", &[]);
    let w = ["-y", "y", "-t", "treat", "-x", "x1", "x2", "x3"];
    let mut ipw = vec!["causal", "ipw", "w"];
    ipw.extend_from_slice(&w);
    ipw.extend_from_slice(&["--bootstrap", "50"]);
    let mut dr = vec!["causal", "doubly-robust", "w"];
    dr.extend_from_slice(&w);
    dr.extend_from_slice(&["--method", "ipw", "--bootstrap", "50"]);
    let mediation = vec![
        "causal",
        "mediation",
        "m",
        "-y",
        "y",
        "-t",
        "d",
        "-m",
        "m",
        "-x",
        "x1",
        "x2",
        "--bootstrap",
        "50",
    ];
    let stdreg = vec![
        "causal",
        "stdreg",
        "s",
        "-y",
        "yc",
        "-t",
        "a",
        "-x",
        "x1",
        "x2",
        "--bootstrap",
        "50",
    ];
    let staggered = vec![
        "causal",
        "staggered-did",
        "cs",
        "-y",
        "y",
        "--treatment-time",
        "first_treat",
        "--time",
        "time",
        "--unit",
        "unit",
        "--bootstrap",
        "50",
    ];
    for cmd in [ipw, dr, mediation, stdreg, staggered] {
        let first = s.ok(&cmd);
        assert_eq!(first, s.ok(&cmd), "{cmd:?}");
        let mut default = cmd.clone();
        default.extend_from_slice(&["--seed", "42"]);
        assert_eq!(first, s.ok(&default), "{cmd:?}");
        let mut other = cmd.clone();
        other.extend_from_slice(&["--seed", "7"]);
        assert_ne!(first, s.ok(&other), "{cmd:?}");
    }
}

/// `causal mediation` JSON carries the reverse decomposition (direct effect on the treated,
/// indirect effect on the controls) with SEs and p-values, and both decompositions add up
/// to the total effect.
#[test]
fn test_cli_causal_mediation_json_has_the_reverse_decomposition() {
    let s = temp_session();
    let path = repo_path("validation/datasets/generated/mediation_huber_sim.csv");
    s.load_as(&path, "m", &[]);
    let j = s.json(&[
        "causal",
        "mediation",
        "m",
        "-y",
        "y",
        "-t",
        "d",
        "-m",
        "m",
        "-x",
        "x1",
        "x2",
        "--bootstrap",
        "50",
    ]);
    for key in [
        "direct_effect_treated",
        "indirect_effect_control",
        "se_direct_treated",
        "se_indirect_control",
        "p_direct_treated",
        "p_indirect_control",
    ] {
        assert!(number(&j, key).is_finite(), "{key}: {j}");
    }
    let total = number(&j, "total_effect");
    let forward = number(&j, "direct_effect") + number(&j, "indirect_effect");
    let reverse = number(&j, "direct_effect_treated") + number(&j, "indirect_effect_control");
    assert!(
        (forward - total).abs() <= 1e-12 * total.abs().max(1.0),
        "{j}"
    );
    assert!(
        (reverse - total).abs() <= 1e-12 * total.abs().max(1.0),
        "{j}"
    );
}

/// `causal ipw` reports the units its trimming dropped, in JSON and text. Fails before:
/// neither output had the count.
#[test]
fn test_cli_causal_ipw_reports_trimmed() {
    let s = temp_session();
    let path = repo_path("validation/datasets/generated/weighting_sim.csv");
    s.load_as(&path, "w", &[]);
    let base = [
        "causal", "ipw", "w", "-y", "y", "-t", "treat", "-x", "x1", "x2", "x3",
    ];
    let mut trimmed = base.to_vec();
    trimmed.extend_from_slice(&["--trim", "0.2"]);
    let j = s.json(&trimmed);
    let n = j["n_trimmed"].as_u64().unwrap_or_else(|| panic!("{j}"));
    assert!(n > 0, "{j}");
    assert!(s.ok(&trimmed).contains(&format!("trimmed: {n})")));
    let mut untrimmed = base.to_vec();
    untrimmed.extend_from_slice(&["--trim", "0"]);
    assert_eq!(s.json(&untrimmed)["n_trimmed"], 0);
}

/// `causal iv` names its standard errors and reports ivreg's adjusted R-squared (case
/// cli_iv_report: AER::ivreg, classical and HC1 via coeftest). Fails before: no --robust flag, no
/// `se_type`, and the R-squared was printed again as the adjusted value.
#[test]
fn test_validate_cli_iv_report() {
    let case = Case::load("cli_iv_report");
    let n = case.scalar("n") as usize;
    let s = temp_session();
    s.load_as(&case.dataset_path(), "ivw", &[]);
    let args = [
        "causal",
        "iv",
        "ivw",
        "-y",
        "y",
        "--endog",
        "x",
        "--instruments",
        "z1",
        "z2",
        "--exog",
        "w",
    ];
    for (robust, se_key, p_key, se_type) in [
        (None, "se_hc1", "p_hc1", "HC1"),
        (Some("false"), "se_classical", "p_classical", "classical"),
    ] {
        let mut a: Vec<&str> = args.to_vec();
        if let Some(r) = robust {
            a.extend(["--robust", r]);
        }
        let v = s.json(&a);
        let t = json_table(&v, "t_value");
        Check::new(&case)
            .named("coef", &t.names, &t.coef, Tol::coef(n))
            .named(se_key, &t.names, &t.se, Tol::se(n))
            .named(p_key, &t.names, &t.p, Tol::p_value(n))
            .scalar("r_squared", number(&v, "r_squared"), Tol::stat(n))
            .scalar("adj_r_squared", number(&v, "adj_r_squared"), Tol::stat(n))
            .finish();
        assert_eq!(v["se_type"], se_type, "{v}");
        let text = s.ok(&a);
        let want = if robust.is_none() {
            "Standard errors: HC1 (heteroskedasticity-robust)"
        } else {
            "Standard errors: classical"
        };
        assert!(text.lines().any(|l| l == want), "{text}");
    }
}

/// CLI-33 (gsynth part): with no --force and no --n-factors the CLI fits gsynth's defaults, unit
/// fixed effects with the number of factors chosen by cross-validation over 0..=5 (case
/// cli_gsynth_defaults: gsynth at fixed r, CV = FALSE, and gsynth's default CV run, whose
/// choice of r differs from core's: deviation CLI-33). Fails before: the
/// default was two-way effects searched over 0..=10.
#[test]
fn test_validate_cli_gsynth_defaults() {
    let case = Case::load("cli_gsynth_defaults");
    let s = temp_session();
    s.load_as(&case.dataset_path(), "gs", &[]);
    let base = [
        "causal", "gsynth", "gs", "-y", "Y", "-d", "D", "--unit", "id", "--time", "time", "-x", "X",
    ];
    // p2a's EM stops before R's converged fit at r >= 1: 5e-4 relative, the tolerance of the
    // estimator's own validation and of test_validate_cli_gsynth_r_spellings (other.md)
    let tol = Tol::iterative(5e-4);
    let j = s.json(&base);
    assert_eq!(j["force"], "unit", "{j}");
    assert_eq!(j["r_max"], 5, "{j}");
    // deviation CLI-33: gsynth's (deterministic) cross-validation chooses r = 0 on this panel
    // (`default_r`, ATT `default_att_avg` 6.948); core's chooses r = 1, ATT 2.967, so both keys
    // stay out of the Check. The default run's ATT is pinned at the r core chose, which is
    // still R's fixed-r fit.
    let r = number(&j, "n_factors") as usize;
    assert_eq!(
        r, 1,
        "core's CV choice changed: re-check deviation CLI-33 against default_r = 0"
    );
    Check::new(&case)
        .scalar(&format!("unit_r{r}_att_avg"), number(&j, "att"), tol)
        .finish();
    // fixed numbers of factors, both fixed-effect choices; r = 0 is the search over 0..=0
    for (force, r) in [("unit", 0), ("unit", 1), ("two_way", 0), ("two_way", 1)] {
        let fixed = if r == 0 {
            ["--r-max", "0"]
        } else {
            ["--n-factors", "1"]
        };
        let j = s.json(&argv(&argv(&base, &["--force", force]), &fixed));
        assert_eq!(number(&j, "n_factors"), f64::from(r), "{j}");
        let tag = if force == "unit" { "unit" } else { "twoway" };
        Check::new(&case)
            .scalar(&format!("{tag}_r{r}_att_avg"), number(&j, "att"), tol)
            .finish();
        assert!(
            j["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .all(|w| !w.as_str().unwrap().contains("--r-max")),
            "{j}"
        );
    }
    // --r-max with a fixed --n-factors has no effect, and the output says so
    let fixed = argv(&base, &["--n-factors", "1", "--r-max", "3"]);
    let j = s.json(&fixed);
    let warnings = j["warnings"].as_array().unwrap();
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().unwrap().contains("--r-max 3 has no effect")),
        "{j}"
    );
    let text = s.ok(&fixed);
    assert!(text.contains("--r-max 3 has no effect"), "{text}");
    // also when the value given is the default's
    let j = s.json(&argv(&base, &["--n-factors", "1", "--r-max", "5"]));
    assert!(
        j["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("--r-max 5 has no effect")),
        "{j}"
    );
}

/// causal staggered-did takes did's base_period (--base-period-type universal), passes
/// --min-obs-per-cell to core, and reports the simple overall ATT and the units dropped as
/// treated in the first period (cases did_cs_varying, did_cs_small_cohorts; analytic SEs).
/// Fails before: no such flags or fields.
#[test]
fn test_validate_cli_staggered_universal_simple() {
    let case = Case::load("did_cs_varying");
    let s = temp_session();
    s.load_as(&case.dataset_path(), "cs", &[]);
    let base = [
        "causal",
        "staggered-did",
        "cs",
        "-y",
        "y",
        "--treatment-time",
        "first_treat",
        "--time",
        "time",
        "--unit",
        "unit",
        "--bootstrap",
        "0",
    ];
    let j = s.json(&base);
    let n = number(&j, "n_obs") as usize;
    assert_eq!(j["base_period_type"], "varying", "{j}");
    Check::new(&case)
        .scalar(
            "never_varying_simple_overall_att",
            number(&j["overall_simple"], "att"),
            Tol::coef(n),
        )
        .scalar(
            "never_varying_simple_overall_se",
            number(&j["overall_simple"], "std_error"),
            Tol::se(n),
        )
        .finish();
    let u = s.json(&argv(&base, &["--base-period-type", "universal"]));
    assert_eq!(u["base_period_type"], "universal", "{u}");
    Check::new(&case)
        .scalar(
            "never_universal_group_overall_att",
            number(&u["overall_att"], "att"),
            Tol::coef(n),
        )
        .scalar(
            "never_universal_group_overall_se",
            number(&u["overall_att"], "std_error"),
            Tol::se(n),
        )
        .finish();
    let text = s.ok(&base);
    assert!(text.to_lowercase().contains("simple"), "{text}");

    let small = Case::load("did_cs_small_cohorts");
    s.load_as(&small.dataset_path(), "cs_small", &[]);
    let mut sm = base.to_vec();
    sm[2] = "cs_small";
    let j = s.json(&sm);
    Check::new(&small)
        .scalar(
            "n_dropped_first_period",
            number(&j, "n_dropped_first_period"),
            Tol::exact(),
        )
        .finish();
    // a cell minimum above every cell's size leaves no cell to estimate
    // (core's refusal, quoting its skip reason: the flag reached the estimator)
    let out = s.run(&argv(&base, &["--min-obs-per-cell", "100000"]));
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(
        err.contains("No valid group-time ATT estimates") && err.contains("small cell size"),
        "{err}"
    );
    // 0 units per cell means nothing: refused by the parser, naming the flag and the bound
    let out = s.run(&argv(&base, &["--min-obs-per-cell", "0"]));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(
        err.contains("--min-obs-per-cell") && err.contains("1.."),
        "{err}"
    );
}
