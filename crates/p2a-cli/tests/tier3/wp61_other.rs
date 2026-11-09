//! T3-WP61: causal, time-series, ML, spatial and viz commands through the p2a CLI.
//! Option values are typed (CLI-14), documented flags act (CLI-23, CLI-10, CLI-11), notes and the
//! wave-2 fields reach text and JSON (SPAT-25, CLI-17 causal part). Every numeric pin is an R case.
use crate::common::*;
use serde_json::Value;

/// `p2a <args>` without a session: clap validates option values while parsing, before any
/// dataset lookup. Asserts a usage error (exit 2) without a panic; returns stderr.
fn usage_error(args: &[&str]) -> String {
    let out = p2a().args(args).output().expect("run p2a");
    assert_no_panic(&out, &args.join(" "));
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(
        out.status.code(),
        Some(2),
        "{args:?}: want exit 2, got {:?}\nstdout: {}\nstderr: {stderr}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
    stderr
}

/// Each row must be a usage error whose message contains the needle.
fn assert_usage_errors(rows: &[(Vec<&str>, &str)]) {
    for (args, needle) in rows {
        let err = usage_error(args);
        assert!(
            err.contains(needle),
            "{args:?}: stderr lacks {needle:?}:\n{err}"
        );
    }
}

/// `v[key]`, with the document in the panic message when absent.
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("no key {key:?} in {v:#}"))
}

/// serde's `Vec<(String, f64)>` ([[name, value], ..]) as (names, values); null -> NaN.
fn pairs(v: &Value) -> (Vec<String>, Vec<f64>) {
    let a = v.as_array().unwrap_or_else(|| panic!("not an array: {v}"));
    a.iter()
        .map(|p| {
            (
                p[0].as_str().unwrap().to_string(),
                p[1].as_f64().unwrap_or(f64::NAN),
            )
        })
        .unzip()
}

/// Rows of numbers after the line equal (trimmed) to `header`, skipping `skip` title lines, up
/// to a blank line.
fn table_rows(text: &str, header: &str, skip: usize) -> Vec<Vec<f64>> {
    text.lines()
        .skip_while(|l| l.trim() != header)
        .skip(1 + skip)
        .take_while(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| {
                    t.parse::<f64>()
                        .unwrap_or_else(|_| panic!("{t:?} in {l:?}"))
                })
                .collect()
        })
        .collect()
}

/// True when two label vectors induce the same partition (labels may be numbered differently).
fn same_partition(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && (0..a.len()).all(|i| (0..a.len()).all(|j| (a[i] == a[j]) == (b[i] == b[j])))
}

/// Every key of `keys` is present.
fn assert_keys(v: &Value, keys: &[&str]) {
    for k in keys {
        assert!(v.get(*k).is_some(), "no key {k:?} in {v:#}")
    }
}

/// Names of a JSON array of strings.
fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|s| {
            s.as_str()
                .unwrap_or_else(|| panic!("not a string: {s}"))
                .to_string()
        })
        .collect()
}

/// Loads a committed dataset (path relative to the repository root) under `name`, for tests
/// that need a dataset but pin no R value.
fn load_repo_csv(s: &TempSession, rel: &str, name: &str) -> String {
    s.load_as(&repo_path(rel), name, &[]);
    name.to_string()
}

// ---------------------------------------------------------------------------------------------
// causal rd, fuzzy-rd
// ---------------------------------------------------------------------------------------------

/// CLI-14: an unknown kernel or selector ran the triangular/mserd default with exit 0; later it
/// was refused only after the session opened (exit 1). It must be a clap usage error (exit 2)
/// listing the values.
#[test]
fn test_cli_unknown_kernel_is_rejected() {
    let rd = ["causal", "rd", "d", "-y", "y", "-r", "x", "-c", "0"];
    let fz = [
        "causal", "fuzzy-rd", "d", "-y", "y", "-r", "x", "-t", "t", "-c", "0",
    ];
    let err = usage_error(&argv(&rd, &["--kernel", "gaussian"]));
    // the core message (kept verbatim), wrapped by clap's "invalid value"
    assert!(
        err.contains("invalid value 'gaussian'")
            && err.contains("Unknown kernel 'gaussian'; use \"triangular\"")
            && err.contains("epanechnikov")
            && err.contains("uniform"),
        "{err}"
    );
    assert_usage_errors(&[
        (argv(&rd, &["--bwselect", "cct"]), "Unknown bwselect 'cct'"),
        (argv(&rd, &["--vce", "robust"]), "hc1"),
        (argv(&fz, &["--kernel", "gaussian"]), "epanechnikov"),
        (argv(&fz, &["--bwselect", "ik"]), "cercomb2"),
    ]);
}

/// CLI-14: `--bwselect certwo` silently ran mserd at v0.1.0; the JSON and text still did not
/// name the selector or the bandwidth source.
/// R: rdbwselect(y, x, c = 0, bwselect = "certwo"), case rd_bwselect_sharp key tri_p1_nn_certwo.
#[test]
fn test_validate_cli_rd_bwselect_certwo() {
    let case = Case::load("rd_bwselect_sharp");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "causal",
        "rd",
        d.as_str(),
        "-y",
        "y",
        "-r",
        "x",
        "-c",
        "0",
        "--bwselect",
        "certwo",
    ];
    let j = s.json(&args);
    let bw = [
        num(field(&j, "h_left")),
        num(field(&j, "h_right")),
        num(field(&j, "b_left")),
        num(field(&j, "b_right")),
    ];
    Check::new(&case)
        .vector("tri_p1_nn_certwo", &bw, Tol::coef(1000))
        .finish();
    assert_eq!(field(&j, "bwselect"), "certwo");
    assert!(
        field(&j, "bandwidth_source")
            .as_str()
            .unwrap()
            .starts_with("h and b selected by certwo"),
        "{j:#}"
    );
    assert_keys(&j, &["warnings", "diagnostics", "kernel", "vce"]);
    let text = s.ok(&args);
    assert!(
        text.contains("Bandwidth source: h and b selected by certwo")
            && text.contains("Kernel: Triangular"),
        "{text}"
    );
}

/// CLI-14: the rdrobust alias `epa` ran the triangular kernel at v0.1.0: R pin of the selected
/// Epanechnikov bandwidths. R: rdbwselect(.., kernel = "epanechnikov").
#[test]
fn test_validate_cli_rd_kernel_alias() {
    let case = Case::load("rd_bwselect_sharp");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal", "rd", &d, "-y", "y", "-r", "x", "-c", "0", "--kernel", "epa",
    ]);
    let bw = [
        num(field(&j, "h_left")),
        num(field(&j, "h_right")),
        num(field(&j, "b_left")),
        num(field(&j, "b_right")),
    ];
    Check::new(&case)
        .vector("epa_p1_nn_mserd", &bw, Tol::coef(1000))
        .finish();
    assert_eq!(field(&j, "kernel"), "Epanechnikov"); // KernelType Display, as T2's JSON
}

/// CLI-14: fuzzy RD had no --bwselect (it always ran mserd).
/// R: rdbwselect(yf, x, c = 0, fuzzy = d, bwselect = "certwo"), case rd_bwselect_fuzzy.
#[test]
fn test_validate_cli_fuzzy_rd_bwselect() {
    let case = Case::load("rd_bwselect_fuzzy");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "causal",
        "fuzzy-rd",
        d.as_str(),
        "-y",
        "yf",
        "-r",
        "x",
        "-t",
        "d",
        "-c",
        "0",
        "--bwselect",
        "cer_two",
    ];
    let j = s.json(&args);
    let bw = [
        num(field(&j, "h_left")),
        num(field(&j, "h_right")),
        num(field(&j, "b_left")),
        num(field(&j, "b_right")),
    ];
    Check::new(&case)
        .vector("fuzzy_tri_p1_nn_certwo", &bw, Tol::coef(1000))
        .finish();
    assert_eq!(field(&j, "bwselect"), "certwo");
    assert_eq!(field(&j, "kernel"), "Triangular");
    assert_keys(&j, &["warnings", "diagnostics", "bandwidth_source"]);
    let text = s.ok(&args);
    assert!(
        text.contains("Bandwidth source: h and b selected by certwo")
            && text.contains("Kernel: Triangular"),
        "{text}"
    );
}

// ---------------------------------------------------------------------------------------------
// causal staggered-did, did
// ---------------------------------------------------------------------------------------------

/// CLI-14: the MCP spelling not_yet_treated ran the never-treated comparison.
/// R: did_cs_analytic notyet_simple_att / _se (analytic influence-function SEs, bstrap = FALSE).
#[test]
fn test_validate_cli_staggered_accepts_mcp_spelling() {
    let case = Case::load("did_cs_analytic");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal",
        "staggered-did",
        &d,
        "-y",
        "y",
        "--treatment-time",
        "first_treat",
        "--time",
        "time",
        "--unit",
        "unit",
        "--comparison",
        "not_yet_treated",
        "--bootstrap",
        "0",
    ]);
    let o = field(&j, "overall_att");
    Check::new(&case)
        .scalar("notyet_simple_att", num(&o["att"]), Tol::coef(720))
        .scalar("notyet_simple_se", num(&o["std_error"]), Tol::se(720))
        .finish();
    assert_eq!(field(&j, "comparison_group"), "Not-Yet-Treated");
    assert_keys(
        &j,
        &["warnings", "diagnostics", "anticipation", "covariates"],
    );
    let base = [
        "causal",
        "staggered-did",
        "d",
        "-y",
        "y",
        "--treatment-time",
        "g",
        "--time",
        "t",
        "--unit",
        "u",
    ];
    assert_usage_errors(&[
        (argv(&base, &["--comparison", "later"]), "not_yet_treated"),
        (argv(&base, &["--method", "drdid2"]), "doubly_robust"),
    ]);
}

/// did's anticipation = 1 with DRDID's doubly robust cells and covariates.
/// R: did_cs_covariates never_ant1_simple_att_dr / _se_dr (glm IRLS propensity).
#[test]
fn test_validate_cli_staggered_anticipation() {
    let case = Case::load("did_cs_covariates");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal",
        "staggered-did",
        &d,
        "-y",
        "y",
        "--treatment-time",
        "first_treat",
        "--time",
        "time",
        "--unit",
        "unit",
        "-x",
        "x1",
        "x2",
        "--method",
        "DR",
        "--anticipation",
        "1",
        "--bootstrap",
        "0",
    ]);
    let o = field(&j, "overall_att");
    Check::new(&case)
        .scalar(
            "never_ant1_simple_att_dr",
            num(&o["att"]),
            Tol::iterative(1e-8),
        )
        .scalar(
            "never_ant1_simple_se_dr",
            num(&o["std_error"]),
            Tol::iterative(1e-8),
        )
        .finish();
    assert_eq!(num(field(&j, "anticipation")), 1.0);
    assert_eq!(field(&j, "covariates"), &serde_json::json!(["x1", "x2"]));
}

/// CLI-32: the help called dr the default while `or` runs (the hand-written text was fixed by
/// 6d139e07; this test guards the typed arg's help and `--anticipation`).
#[test]
fn test_cli_staggered_help_names_the_default() {
    let out = p2a()
        .args(["causal", "staggered-did", "--help"])
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{help}");
    assert!(!help.contains("doubly robust, default"), "{help}");
    let line = help
        .lines()
        .skip_while(|l| !l.contains("--method <"))
        .take(6)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        line.contains("outcome_regression") && line.contains("[default: or]"),
        "{line}"
    );
    assert!(help.contains("--anticipation"), "{help}");
}

/// The 2x2 DiD JSON carries `aliased`, `df`, the warnings and the diagnostics (lm drops
/// years_since_hire, collinear with tenure).
/// R: did_2x2_training aliased_coef / aliased_se_hc1 for training_group:post_training.
#[test]
fn test_validate_cli_did_aliased() {
    let case = Case::load("did_2x2_training");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "causal",
        "did",
        d.as_str(),
        "-y",
        "performance",
        "--treat",
        "training_group",
        "--post",
        "post_training",
        "-x",
        "tenure",
        "years_since_hire",
    ];
    let j = s.json(&args);
    Check::new(&case)
        .scalar("aliased_df_resid", num(field(&j, "df")), Tol::exact())
        .finish();
    let att = num(field(&j, "att"));
    let se = num(field(&j, "std_error"));
    let coef = case.named("aliased_coef", "training_group:post_training");
    let se_r = case.named("aliased_se_hc1", "training_group:post_training");
    assert!(Tol::coef(1200).allows(att, coef), "{att} vs {coef}");
    assert!(Tol::se(1200).allows(se, se_r), "{se} vs {se_r}");
    assert_eq!(
        field(&j, "aliased"),
        &serde_json::json!(["years_since_hire"])
    );
    assert!(
        field(&j, "warnings")
            .to_string()
            .contains("years_since_hire"),
        "{j:#}"
    );
    assert_keys(&j, &["diagnostics"]);
    let text = s.ok(&args);
    assert!(
        text.contains("Warning: Coefficients not defined because of singularities")
            && text.contains("years_since_hire"),
        "{text}"
    );
}

/// `causal staggered-did -h` shows a one-line help for every flag (clap prints it after the
/// value name, or on the next line when another flag's help is long).
#[test]
fn test_cli_staggered_short_help_lists_flags() {
    let out = p2a()
        .args(["causal", "staggered-did", "-h"])
        .output()
        .unwrap();
    let h = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = h.lines().collect();
    for flag in [
        "--treatment-time <",
        "--comparison <",
        "--method <",
        "--bootstrap <",
        "--anticipation <",
    ] {
        let i = lines
            .iter()
            .position(|l| l.trim_start().starts_with(flag))
            .unwrap_or_else(|| panic!("{flag} missing:\n{h}"));
        let same = lines[i]
            .split(flag)
            .nth(1)
            .and_then(|r| r.split_once('>'))
            .map(|(_, d)| d.trim());
        let help = match same {
            Some(d) if !d.is_empty() => d,
            _ => lines.get(i + 1).map(|l| l.trim()).unwrap_or(""),
        };
        assert!(
            !help.is_empty() && !help.starts_with("--"),
            "{flag} has no one-line help:\n{h}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// causal ipw, doubly-robust, tmle, stdreg, iv, mediation
// ---------------------------------------------------------------------------------------------

/// CLI-14 (`--estimand atc` gave ATE) and the SE method and warnings of IPW and AIPW.
/// R: ipw_aipw_sim (glm propensity, Hajek IPW with the M-estimation SE; AIPW with OLS outcome
/// models), trim 0.
#[test]
fn test_validate_cli_ipw_dr_se_method() {
    let case = Case::load("ipw_aipw_sim");
    let s = temp_session();
    let d = s.load(&case);
    let ipw_args = [
        "causal",
        "ipw",
        d.as_str(),
        "-y",
        "y",
        "-t",
        "treat",
        "-x",
        "x1",
        "x2",
        "x3",
        "--trim",
        "0",
    ];
    let ipw = s.json(&ipw_args); // default --bootstrap 0: the analytic M-estimation SE
    let dr = s.json(&[
        "causal",
        "doubly-robust",
        &d,
        "-y",
        "y",
        "-t",
        "treat",
        "-x",
        "x1",
        "x2",
        "x3",
        "--trim",
        "0",
        "--method",
        "dr",
    ]);
    Check::new(&case)
        .scalar("ipw_ate_hajek", num(field(&ipw, "effect")), Tol::coef(300))
        .scalar(
            "ipw_ate_se_mest",
            num(field(&ipw, "std_error")),
            Tol::se(300),
        )
        .scalar("ipw_ate_se_eif", num(field(&ipw, "se_eif")), Tol::se(300))
        .scalar("aipw_ate", num(field(&dr, "effect")), Tol::coef(300))
        .scalar("aipw_ate_se", num(field(&dr, "std_error")), Tol::se(300))
        .finish();
    assert_eq!(
        field(&ipw, "se_method"),
        "M-estimation (propensity estimated)"
    );
    assert_eq!(field(&dr, "se_method"), "analytic influence function");
    assert_keys(
        &ipw,
        &["warnings", "diagnostics", "se_mest", "se_naive", "se_eif"],
    );
    assert_keys(&dr, &["warnings"]);
    let text = s.ok(&ipw_args);
    assert!(
        text.contains("SE method: M-estimation (propensity estimated)"),
        "{text}"
    );
    let base = ["causal", "ipw", "d", "-y", "y", "-t", "t", "-x", "a"];
    let drb = [
        "causal",
        "doubly-robust",
        "d",
        "-y",
        "y",
        "-t",
        "t",
        "-x",
        "a",
    ];
    assert_usage_errors(&[
        (
            argv(&base, &["--estimand", "atc"]),
            "unknown estimand 'atc'; expected one of: ate, att",
        ),
        (argv(&drb, &["--method", "tmle"]), "aipw"),
    ]);
}

/// CLI-14: `--qmodel gaussian` (a QModel alias) ran the logistic Q model.
/// R: tmle_glm linear_ate, linear_se (lm Q on yc, glm g, linear fluctuation).
#[test]
fn test_validate_cli_tmle_qmodel_alias() {
    let case = Case::load("tmle_glm");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal", "tmle", &d, "-y", "yc", "-t", "a", "-x", "w1", "w2", "w3", "w4", "--qmodel",
        "gaussian",
    ]);
    Check::new(&case)
        .scalar("linear_ate", num(field(&j, "ate")), Tol::coef(300))
        .scalar("linear_se", num(field(&j, "ate_se")), Tol::se(300))
        .finish();
    assert_keys(
        &j,
        &[
            "warnings",
            "fluctuation_coef_control",
            "targeting_iterations",
        ],
    );
    assert_usage_errors(&[(
        vec![
            "causal", "tmle", "d", "-y", "y", "-t", "t", "-x", "a", "--qmodel", "probit",
        ],
        "logistic",
    )]);
}

/// CLI-14: R's family name `binomial` ran a linear outcome model.
/// R: stdreg_glm logit_ate, logit_ey1, logit_ey0 (glm(yb ~ x1 + x2 + a, binomial), tight fit).
#[test]
fn test_validate_cli_stdreg_r_spellings() {
    let case = Case::load("stdreg_glm");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal",
        "stdreg",
        &d,
        "-y",
        "yb",
        "-t",
        "a",
        "-x",
        "x1",
        "x2",
        "--model",
        "binomial",
        "--se-method",
        "robust",
    ]);
    Check::new(&case)
        .scalar("logit_ate", num(field(&j, "ate")), Tol::iterative(1e-8))
        .scalar("logit_ey1", num(field(&j, "ey1")), Tol::iterative(1e-8))
        .scalar("logit_ey0", num(field(&j, "ey0")), Tol::iterative(1e-8))
        .finish();
    assert_eq!(field(&j, "converged"), true);
    assert_keys(&j, &["warnings", "iterations"]);
    let b = ["causal", "stdreg", "d", "-y", "y", "-t", "t", "-x", "a"];
    assert_usage_errors(&[
        (argv(&b, &["--model", "probit"]), "poisson"),
        (argv(&b, &["--estimand", "ato"]), "levels"),
        (argv(&b, &["--se-method", "jackknife"]), "sandwich"),
    ]);
}

/// CLI-17 (causal part): IV and mediation JSON carried no warnings, IV no diagnostics (the IV
/// first stage and both commands' `missing` are kept).
#[test]
fn test_cli_causal_json_carries_notes() {
    let s = temp_session();
    let iv = load_repo_csv(
        &s,
        "validation/datasets/generated/iv_weak_controls.csv",
        "ivw",
    );
    let j = s.json(&[
        "causal",
        "iv",
        &iv,
        "-y",
        "y",
        "--exog",
        "w",
        "--endog",
        "x",
        "--instruments",
        "z1",
        "z2",
    ]);
    assert_keys(&j, &["warnings", "diagnostics", "first_stage"]);
    let med = load_repo_csv(&s, "validation/datasets/generated/causal_na.csv", "cna");
    let j = s.json(&[
        "causal",
        "mediation",
        &med,
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
    assert_keys(&j, &["warnings", "missing"]);
}

// ---------------------------------------------------------------------------------------------
// causal matching, synth, gsynth
// ---------------------------------------------------------------------------------------------

/// MatchIt spellings (nearest, glm), m.order and ESS. R: matchit_nearest ratio2_ess
/// (matchit(ratio = 2), m.order "largest"): control ESS 136.5689655172414, treated 89.
#[test]
fn test_validate_cli_matching_options() {
    let case = Case::load("matchit_nearest");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "causal",
        "matching",
        d.as_str(),
        "-t",
        "treat",
        "-x",
        "x1",
        "x2",
        "x3",
        "--method",
        "nearest",
        "--ratio",
        "2",
        "--distance",
        "glm",
    ];
    let j = s.json(&args);
    Check::new(&case)
        .vector(
            "ratio2_ess",
            &[num(field(&j, "ess_control")), num(field(&j, "ess_treated"))],
            Tol::coef(240),
        )
        .finish();
    assert_eq!(field(&j, "m_order"), "largest");
    assert_keys(
        &j,
        &["warnings", "diagnostics", "total_distance", "caliper_abs"],
    );
    let text = s.ok(&args);
    assert!(!text.contains("Detailed Balance After Matching"), "{text}");
    let b = ["causal", "matching", "d", "-t", "t", "-x", "a"];
    assert_usage_errors(&[
        (argv(&b, &["--method", "genetic"]), "nearest"),
        (argv(&b, &["--distance", "rf"]), "mahalanobis"),
        (argv(&b, &["--m-order", "closest"]), "largest"),
    ]);
}

/// CLI-14: `--v-method uniform` (an alias of equal V) ran the data-driven V search.
/// R: synth_weights_qp equal_w and equal_predictor_loss (the exact simplex QP with V = I on the
/// standardised predictors).
#[test]
fn test_validate_cli_synth_v_method_alias() {
    let case = Case::load("synth_weights_qp");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal",
        "synth",
        &d,
        "-y",
        "y",
        "--unit",
        "unit",
        "--time",
        "time",
        "--treated",
        "TR",
        "--treatment-time",
        "15",
        "-p",
        "x1",
        "x2",
        "x3",
        "--v-method",
        "uniform",
    ]);
    let (names, w) = pairs(field(&j, "all_unit_weights"));
    Check::new(&case)
        .named("equal_w", &names, &w, Tol::coef(14))
        .scalar(
            "equal_predictor_loss",
            num(field(&j, "predictor_loss")),
            Tol::stat(14),
        )
        .finish();
    assert_keys(&j, &["warnings"]);
    assert_usage_errors(&[(
        vec![
            "causal",
            "synth",
            "d",
            "-y",
            "y",
            "--unit",
            "u",
            "--time",
            "t",
            "--treated",
            "T",
            "--treatment-time",
            "5",
            "-p",
            "a",
            "--v-method",
            "ml",
        ],
        "equal",
    )]);
}

/// CLI-14: gsynth spellings, Display names in JSON, the covariate coefficients.
/// R: gsynth_simdata ife_r2_cov_att_avg and ife_r2_cov_beta (gsynth 1.4, r = 2, X1 X2, converged);
/// 5e-4 relative, p2a's default stopping rule.
#[test]
fn test_validate_cli_gsynth_r_spellings() {
    let case = Case::load("gsynth_simdata");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "causal",
        "gsynth",
        &d,
        "-y",
        "Y",
        "-d",
        "D",
        "--unit",
        "id",
        "--time",
        "time",
        "-x",
        "X1",
        "X2",
        "--n-factors",
        "2",
        "--estimator",
        "gsynth",
        "--force",
        "two-way",
    ]);
    Check::new(&case)
        .scalar(
            "ife_r2_cov_att_avg",
            num(field(&j, "att")),
            Tol::new(5e-4, 0.0),
        )
        .vector(
            "ife_r2_cov_beta",
            &f64s(field(&j, "beta")),
            Tol::new(5e-4, 0.0),
        )
        .finish();
    assert_eq!(field(&j, "estimator"), "ife");
    assert_eq!(field(&j, "force"), "two_way");
    assert_keys(
        &j,
        &[
            "warnings",
            "covariate_names",
            "cv_mspe",
            "sigma2",
            "dynamic_counts",
            "bootstrap",
        ],
    );
    let b = [
        "causal", "gsynth", "d", "-y", "y", "-d", "t", "--unit", "u", "--time", "t",
    ];
    assert_usage_errors(&[
        (argv(&b, &["--force", "both"]), "two_way"),
        (argv(&b, &["--estimator", "em"]), "mc"),
    ]);
}

// ---------------------------------------------------------------------------------------------
// ts option parsing and notes
// ---------------------------------------------------------------------------------------------

/// CLI-14 time-series part: unknown values ran mean/additive/Yule-Walker/additive/level with exit
/// 0, and `--spans 3,x` became [3]. Each must be a usage error listing the values.
#[test]
fn test_cli_ts_rejects_unknown_option_values() {
    let mut rows: Vec<(Vec<&str>, &str)> = vec![
        (
            vec![
                "ts",
                "changepoint",
                "d",
                "--col",
                "y",
                "--change-type",
                "trend",
            ],
            "meanvar",
        ),
        (
            vec![
                "ts",
                "holt-winters",
                "d",
                "--col",
                "y",
                "--period",
                "12",
                "--seasonal",
                "damped",
            ],
            "multiplicative",
        ),
        (
            vec!["ts", "ar", "d", "--col", "y", "--method", "mle"],
            "yule-walker",
        ),
        (
            vec![
                "ts",
                "decompose",
                "d",
                "--col",
                "y",
                "--period",
                "12",
                "--decompose-type",
                "stl",
            ],
            "multiplicative",
        ),
        (
            vec![
                "ts",
                "struct-ts",
                "d",
                "--col",
                "y",
                "--model-type",
                "arima",
            ],
            "BSM",
        ),
    ];
    if cfg!(feature = "spectral-analysis") {
        rows.push((
            vec!["ts", "spectrum", "d", "--col", "y", "--spans", "3,x"],
            "--spans",
        ));
    }
    assert_usage_errors(&rows);
}

/// Granger reports the core's notes. R: granger_short lag2 (grangertest(y ~ x, order = 2)).
#[test]
fn test_validate_cli_granger_notes() {
    let case = Case::load("granger_short");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&["ts", "granger", &d, "--x", "x", "--y", "y", "--lags", "2"]);
    Check::new(&case)
        .scalar("lag2_F", num(field(&j, "f_statistic")), Tol::stat(20))
        .scalar("lag2_p", num(field(&j, "p_value")), Tol::p_value(20))
        .scalar("lag2_df2", num(field(&j, "df2")), Tol::exact())
        .finish();
    assert_keys(&j, &["warnings"]);
}

/// StructTS reports its convergence and warnings; the model name is the R spelling.
/// R: structts_nile_ukgas ukgas_*_tight (StructTS(log10(UKgas), "BSM") at the tight optimum).
/// The BSM level variance is 0 at the optimum (a boundary) and the slope variance sits next to
/// it, so only the seasonal and observation variances are compared here.
#[test]
fn test_validate_cli_struct_ts_notes() {
    let case = Case::load("structts_nile_ukgas");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "ts",
        "struct-ts",
        d.as_str(),
        "--col",
        "ukgas",
        "--model-type",
        "bsm",
        "--period",
        "4",
    ];
    let j = s.json(&args);
    let c = field(&j, "coefficients");
    Check::new(&case)
        .scalar(
            "ukgas_loglik_tight",
            num(field(&j, "log_likelihood")),
            Tol::stat(108),
        )
        .named_subset(
            "ukgas_coef_tight",
            &["seas", "epsilon"],
            &["seas", "epsilon"],
            &[num(&c["seasonal"]), num(&c["epsilon"])],
            Tol::iterative(1e-6),
        )
        .finish();
    assert_eq!(field(&j, "model_type"), "BSM");
    assert_keys(&j, &["converged", "iterations", "warnings"]);
    assert!(s.ok(&args).contains("Converged: "), "notes line");
}

/// `ts ar` JSON carries the warnings and the canonical method name; the changepoint penalty help
/// names changepoint's MBIC default.
#[test]
fn test_cli_ar_notes_and_changepoint_help() {
    let s = temp_session();
    let d = load_repo_csv(&s, "validation/datasets/generated/granger_short.csv", "gs");
    let j = s.json(&[
        "ts", "ar", &d, "--col", "x", "--method", "OLS", "--order", "2",
    ]);
    assert_eq!(field(&j, "method"), "ols");
    assert_keys(&j, &["warnings"]);
    let out = p2a()
        .args(["ts", "changepoint", "--help"])
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        help.contains("MBIC") && !help.contains("automatic BIC"),
        "{help}"
    );
}
