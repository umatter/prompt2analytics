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
