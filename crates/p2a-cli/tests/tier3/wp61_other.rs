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

/// The `t+i:` rows of a forecast block: the numbers after the colon (brackets and commas dropped).
fn forecast_rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| l.trim_start().starts_with("t+"))
        .map(|l| {
            l.split_once(':')
                .unwrap()
                .1
                .replace(['[', ']', ','], " ")
                .split_whitespace()
                .map(|t| t.parse().unwrap_or_else(|_| panic!("{t:?} in {l:?}")))
                .collect()
        })
        .collect()
}

fn column(rows: &[Vec<f64>], i: usize) -> Vec<f64> {
    rows.iter().map(|r| r[i]).collect()
}

/// CLI-23: --horizon was parsed and dropped. R: ts_holtwinters_forecast
/// airpass_mult_{fit,resvar} (HoltWinters(AirPassengers, 0.3, 0.05, 0.4, "multiplicative"))
/// and airpass_notrend_{fit,lwr,upr} (HoltWinters(AirPassengers, 0.3, FALSE, 0.2, "additive");
/// predict(n.ahead = 12, prediction.interval = TRUE)).
#[test]
fn test_validate_cli_holt_winters_horizon_prints_forecasts() {
    let case = Case::load("ts_holtwinters_forecast");
    let s = temp_session();
    let d = s.load(&case);

    // multiplicative: point forecasts; R's interval formula for this branch is wrong, so the
    // intervals are null and a warning says why.
    let mult = [
        "ts",
        "holt-winters",
        d.as_str(),
        "--col",
        "airpass",
        "--period",
        "12",
        "--seasonal",
        "multiplicative",
        "--alpha",
        "0.3",
        "--beta",
        "0.05",
        "--gamma",
        "0.4",
        "--horizon",
        "12",
    ];
    let j = s.json(&mult);
    let f = field(&j, "forecast");
    Check::new(&case)
        .vector("airpass_mult_fit", &f64s(&f["mean"]), Tol::coef(144))
        .scalar(
            "airpass_mult_resvar",
            num(&f["residual_variance"]),
            Tol::stat(144),
        )
        // deviation TS-22: airpass_mult_lwr / airpass_mult_upr are not compared
        .finish();
    assert_eq!(num(&f["horizon"]), 12.0);
    for k in ["level", "std_errors", "lower", "upper"] {
        assert!(f[k].is_null(), "{k}: {f:#}");
    }
    let warnings = strings(field(&j, "warnings"));
    assert!(
        warnings.iter().any(|w| w.contains("prediction intervals")),
        "{warnings:?}"
    );
    let text = s.ok(&mult);
    assert!(
        text.contains(
            "Forecasts (h = 12; prediction intervals are implemented for additive and non-seasonal fits):"
        ),
        "{text}"
    );
    let rows = forecast_rows(&text);
    Check::new(&case)
        .vector("airpass_mult_fit", &column(&rows, 0), Tol::printed(4))
        .finish();

    // additive, no trend: forecasts and 95% limits
    let add = [
        "ts",
        "holt-winters",
        d.as_str(),
        "--col",
        "airpass",
        "--period",
        "12",
        "--seasonal",
        "additive",
        "--alpha",
        "0.3",
        "--beta",
        "0",
        "--gamma",
        "0.2",
        "--horizon",
        "12",
    ];
    let j = s.json(&add);
    let f = field(&j, "forecast");
    Check::new(&case)
        .vector("airpass_notrend_fit", &f64s(&f["mean"]), Tol::coef(144))
        .vector("airpass_notrend_lwr", &f64s(&f["lower"]), Tol::stat(144))
        .vector("airpass_notrend_upr", &f64s(&f["upper"]), Tol::stat(144))
        .finish();
    assert_eq!((num(&f["horizon"]), num(&f["level"])), (12.0, 0.95));
    let text = s.ok(&add);
    assert!(
        text.contains("Forecasts (h = 12, 95% prediction interval):"),
        "{text}"
    );
    let rows = forecast_rows(&text);
    Check::new(&case)
        .vector("airpass_notrend_fit", &column(&rows, 0), Tol::printed(4))
        .vector("airpass_notrend_lwr", &column(&rows, 1), Tol::printed(4))
        .vector("airpass_notrend_upr", &column(&rows, 2), Tol::printed(4))
        .finish();

    // optimised parameters (the optimiser is not pinned, TS-24): the CLI prints what the
    // library predicts
    let j6 = s.json(&[
        "ts",
        "holt-winters",
        &d,
        "--col",
        "airpass",
        "--period",
        "12",
        "--horizon",
        "6",
        "--level",
        "0.8",
    ]);
    let ds = case_dataset(&case);
    let fit = p2a_core::run_holt_winters(
        &ds,
        "airpass",
        12,
        p2a_core::SeasonalType::Additive,
        None,
        None,
        None,
    )
    .unwrap();
    let p = fit.predict(6, 0.8).unwrap();
    let f6 = field(&j6, "forecast");
    assert_eq!(f64s(&f6["mean"]), p.forecast);
    assert_eq!(f64s(&f6["lower"]), p.lower);
    assert!(
        s.ok(&[
            "ts",
            "holt-winters",
            &d,
            "--col",
            "airpass",
            "--period",
            "12",
            "--horizon",
            "2",
            "--level",
            "0.975"
        ])
        .contains("Forecasts (h = 2, 97.5% prediction interval):")
    );
}

/// CLI-10: "a -> b" printed irf[h][a][b], the response of a to a shock in b (the Cholesky zero
/// for a -> b), for two variables and five horizons. R: var_canada irf_ortho_<impulse>
/// (vars::irf(VAR(Canada, p = 2), n.ahead = 12, ortho = TRUE)), 13 x 4 column-major.
#[test]
fn test_validate_cli_irf_labels_as_vars() {
    let case = Case::load("var_canada");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "ts",
        "irf",
        d.as_str(),
        "--cols",
        "e",
        "prod",
        "rw",
        "U",
        "--lags",
        "2",
        "--steps",
        "13",
    ];
    let text = s.ok(&args);
    let mut c = Check::new(&case);
    for imp in ["e", "prod", "rw", "U"] {
        let rows = table_rows(&text, &format!("Impulse: {imp}"), 1);
        assert_eq!(rows.len(), 13, "Impulse: {imp} table in\n{text}");
        let col_major: Vec<f64> = (1..=4)
            .flat_map(|r| rows.iter().map(move |row| row[r]))
            .collect();
        c.vector(&format!("irf_ortho_{imp}"), &col_major, Tol::printed(6));
    }
    c.finish();
    let j = s.json(&args);
    assert_eq!(
        field(&j, "axes"),
        &serde_json::json!(["horizon", "response", "impulse"])
    );
    let irf = field(&j, "irf");
    let r = case.vector("irf_ortho_e");
    assert!(
        Tol::stat(82).allows(num(&irf[0][1][0]), r[13]),
        "response prod, impulse e, h = 0"
    );
    assert_eq!(
        num(&irf[0][0][1]),
        0.0,
        "response e to an impulse in prod at h = 0 is the Cholesky zero"
    );
    let prod_on_e = f64s(&field(&j, "impulse_responses")["e"]["prod"]);
    assert_eq!(prod_on_e.len(), 13);
    for (h, v) in prod_on_e.iter().enumerate() {
        assert!(
            Tol::stat(82).allows(*v, r[13 + h]),
            "h = {h}: {v} vs {}",
            r[13 + h]
        )
    }
}

/// `ts var` JSON gains the coefficient table, SEs and residual df; the text is the core report.
/// R: var_canada (VAR(Canada, p = 2, type = "const")).
#[test]
fn test_validate_cli_var_json() {
    let case = Case::load("var_canada");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "ts",
        "var",
        d.as_str(),
        "--cols",
        "e",
        "prod",
        "rw",
        "U",
        "--lags",
        "2",
    ];
    let j = s.json(&args);
    let names = strings(field(&j, "coef_names"));
    Check::new(&case)
        .named(
            "coef_e",
            &names,
            &f64s(&field(&j, "coefficients")[0]),
            Tol::coef(82),
        )
        .named(
            "se_e",
            &names,
            &f64s(&field(&j, "std_errors")[0]),
            Tol::se(82),
        )
        .named(
            "p_U",
            &names,
            &f64s(&field(&j, "p_values")[3]),
            Tol::p_value(82),
        )
        .scalar("loglik", num(field(&j, "log_likelihood")), Tol::stat(82))
        .scalar(
            "sigma_ml_logdet",
            num(field(&j, "log_det_sigma_ml")),
            Tol::stat(82),
        )
        .scalar("df_resid", num(field(&j, "df_residual")), Tol::exact())
        .finish();
    assert_keys(&j, &["t_values", "sigma_u_ml"]);
    assert!(s.ok(&args).contains("Residual df per equation: 73"));
}

/// CLI-11: --horizon printed the last h in-sample variances in reverse order as t+1..t+h; the
/// JSON ignored it. R: garch_normal_t norm_forecast_var_tight (GARCH(1,1), normal, with mean;
/// the GARCH recursion from the likelihood optimum).
#[test]
fn test_validate_cli_garch_horizon_forecasts() {
    let case = Case::load("garch_normal_t");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "ts",
        "garch",
        d.as_str(),
        "--col",
        "r_norm",
        "-p",
        "1",
        "-q",
        "1",
        "--horizon",
        "10",
    ];
    let j = s.json(&args);
    let f = field(&j, "forecast");
    let var = f64s(&f["variance"]);
    Check::new(&case)
        .vector("norm_forecast_var_tight", &var, Tol::new(1e-8, 0.0))
        .finish();
    for (v, vol) in var.iter().zip(f64s(&f["volatility"])) {
        assert!(
            (vol - v.sqrt()).abs() <= 1e-15 * v.sqrt(),
            "{vol} vs sqrt({v})"
        )
    }
    assert_eq!(num(&f["horizon"]), 10.0);
    assert_keys(
        &j,
        &[
            "param_names",
            "estimates",
            "std_errors",
            "p_values",
            "shape",
            "distribution",
            "converged",
            "iterations",
            "warnings",
        ],
    );
    assert_eq!(field(&j, "distribution"), "norm");
    let text = s.ok(&args);
    let printed: Vec<f64> = text
        .lines()
        .filter(|l| l.trim_start().starts_with("t+"))
        .map(|l| {
            l.split("variance ")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap()
        })
        .collect();
    Check::new(&case)
        .vector("norm_forecast_var_tight", &printed, Tol::new(1e-6, 0.0))
        .finish();
}

/// --distribution std (fGarch's standardised t). R: garch_normal_t std_par_tight[shape],
/// std_forecast_var_tight.
#[test]
fn test_validate_cli_garch_student_t() {
    let case = Case::load("garch_normal_t");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "ts",
        "garch",
        &d,
        "--col",
        "r_t",
        "-p",
        "1",
        "-q",
        "1",
        "--distribution",
        "t",
        "--horizon",
        "10",
    ]);
    let shape = num(field(&j, "shape"));
    assert!(
        Tol::new(1e-8, 0.0).allows(shape, case.named("std_par_tight", "shape")),
        "{shape}"
    );
    Check::new(&case)
        .vector(
            "std_forecast_var_tight",
            &f64s(&field(&j, "forecast")["variance"]),
            Tol::new(1e-8, 0.0),
        )
        .finish();
    assert_eq!(field(&j, "distribution"), "std");
    assert_usage_errors(&[(
        vec!["ts", "garch", "d", "--col", "y", "--distribution", "ged"],
        "std",
    )]);
}

/// `ts varma --long-var-order`; the JSON carries the step-2 table. R: varma_hr (Hannan-Rissanen
/// VARMA(1,1), long VAR m = 8 and the default m).
#[test]
fn test_validate_cli_varma_long_var_order() {
    let case = Case::load("varma_hr");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "ts",
        "varma",
        &d,
        "--cols",
        "y1",
        "y2",
        "-p",
        "1",
        "-q",
        "1",
        "--long-var-order",
        "8",
    ]);
    let names = strings(field(&j, "regressor_names"));
    Check::new(&case)
        .scalar(
            "long_var_order",
            num(field(&j, "long_var_order")),
            Tol::exact(),
        )
        .scalar("n_eff", num(field(&j, "n_obs")), Tol::exact())
        .scalar("loglik", num(field(&j, "log_likelihood")), Tol::stat(391))
        .scalar("df_residual", num(field(&j, "df_residual")), Tol::exact())
        .named(
            "coef_y1",
            &names,
            &f64s(&field(&j, "coefficients")[0]),
            Tol::coef(391),
        )
        .finish();
    let args = [
        "ts", "varma", &d, "--cols", "y1", "y2", "-p", "1", "-q", "1",
    ];
    let j = s.json(&args);
    Check::new(&case)
        .scalar(
            "default_long_var_order",
            num(field(&j, "long_var_order")),
            Tol::exact(),
        )
        .scalar(
            "default_loglik",
            num(field(&j, "log_likelihood")),
            Tol::stat(388),
        )
        .finish();
    assert_keys(&j, &["warnings", "std_errors", "df_residual", "estimator"]);
    assert!(
        s.ok(&args).contains("e.y1.l1"),
        "the text prints the step-2 table"
    );
}

/// `ts vecm` defaults to K = 2 and the trace-test rank; --ecdet as ca.jo. R: vecm_denmark
/// (urca::ca.jo(K = 2), ecdet none and const; rank by the trace test at 5%).
#[test]
fn test_validate_cli_vecm_options() {
    let case = Case::load("vecm_denmark");
    let s = temp_session();
    let d = s.load(&case);
    let cols = ["--cols", "LRM", "LRY", "IBO", "IDE"];
    // default --lags 2, no --rank
    let j = s.json(&argv(
        &["ts", "vecm", &d],
        &argv(&cols, &["--ecdet", "none"]),
    ));
    let mut trace = f64s(field(&j, "trace_stat"));
    trace.reverse(); // urca order
    let beta: Vec<f64> = field(&j, "beta")
        .as_array()
        .unwrap()
        .iter()
        .map(|r| num(&r[0]))
        .collect();
    Check::new(&case)
        .vector("none_trace_stat", &trace, Tol::stat(55))
        .vector("none_beta", &beta, Tol::coef(55))
        .finish();
    assert_eq!(
        (
            num(field(&j, "selected_rank")),
            num(field(&j, "rank")),
            num(field(&j, "lags"))
        ),
        (1.0, 1.0, 2.0)
    );
    assert_eq!(field(&j, "method"), "VECM");
    let j0 = s.json(&argv(
        &["ts", "vecm", &d],
        &argv(&cols, &["--ecdet", "constant"]),
    ));
    let mut trace0 = f64s(field(&j0, "trace_stat"));
    trace0.reverse();
    Check::new(&case)
        .vector("trace_stat", &trace0, Tol::stat(55))
        .finish();
    assert_eq!(num(field(&j0, "selected_rank")), 0.0);
    assert!(
        field(&j0, "warnings")
            .to_string()
            .contains("does not reject rank 0"),
        "{j0:#}"
    );
    let text = s.ok(&argv(
        &["ts", "vecm", &d],
        &argv(&cols, &["--ecdet", "const", "--rank", "1"]),
    ));
    assert!(
        text.contains("Rank chosen by the trace test at 5%: 0"),
        "{text}"
    );
    assert_usage_errors(&[(
        vec![
            "ts",
            "vecm",
            "d",
            "--cols",
            "a",
            "b",
            "--ecdet",
            "quadratic",
        ],
        "trend",
    )]);
}

// ---------------------------------------------------------------------------------------------
// spatial
// ---------------------------------------------------------------------------------------------

/// SPAT-25: the CLI JSON impacts object had no warnings; the text table printed none.
/// R: spatial_sar_sdm_sem sar_rho, sar_impact_* (spatialreg formulas written out; 4-NN
/// row-standardised W).
#[test]
fn test_validate_cli_spatial_impacts_warnings_in_json() {
    let case = Case::load("spatial_sar_sdm_sem");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "spatial",
        "sar",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "--coord-x",
        "cx",
        "--coord-y",
        "cy",
        "-k",
        "4",
        "--impacts",
    ];
    let j = s.json(&args);
    let imp = field(&j, "impacts");
    assert_eq!(field(imp, "warnings"), &serde_json::json!([]));
    let names = strings(&imp["var_names"]);
    Check::new(&case)
        .scalar("sar_rho", num(field(&j, "rho")), Tol::coef(80))
        .named(
            "sar_impact_direct",
            &names,
            &f64s(&imp["direct"]),
            Tol::coef(80),
        )
        .named(
            "sar_impact_indirect",
            &names,
            &f64s(&imp["indirect"]),
            Tol::coef(80),
        )
        .named(
            "sar_impact_total",
            &names,
            &f64s(&imp["total"]),
            Tol::coef(80),
        )
        .finish();
    assert_keys(
        &j,
        &[
            "converged",
            "iterations",
            "warnings",
            "weight_style",
            "k_neighbors",
        ],
    );
    assert_eq!(field(&j, "weight_style"), "W");
    assert!(s.ok(&args).contains("Spatial Impacts:"));
}

/// CLI-14: an unknown style ran row standardisation. On a 4-NN graph of n = 80 rows W = B / 4
/// and U = B / 320 = W / 80, so rho_B = rho_W / 4 and rho_U = 80 rho_W, all with
/// W's logLik (R: spatial_sar_sdm_sem sar_rho, sar_loglik; spdep's styles reparameterise rho).
#[test]
fn test_validate_cli_spatial_weight_style_b() {
    let case = Case::load("spatial_sar_sdm_sem");
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "spatial",
        "sar",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "--coord-x",
        "cx",
        "--coord-y",
        "cy",
        "-k",
        "4",
    ];
    let rho_w = case.scalar("sar_rho");
    for (style, canonical, scale) in [("binary", "B", 0.25), ("unit_sum", "U", 80.0)] {
        let j = s.json(&argv(&base, &["--weight-style", style]));
        assert!(
            Tol::coef(80).allows(num(field(&j, "rho")), rho_w * scale),
            "{style}: {j:#}"
        );
        Check::new(&case)
            .scalar(
                "sar_loglik",
                num(field(&j, "log_likelihood")),
                Tol::stat(80),
            )
            .finish();
        assert_eq!(field(&j, "weight_style"), canonical);
    }
    let sar = [
        "spatial",
        "sar",
        "d",
        "-y",
        "y",
        "-x",
        "a",
        "--coord-x",
        "cx",
        "--coord-y",
        "cy",
    ];
    assert_usage_errors(&[
        (argv(&sar, &["--weight-style", "rowstdd"]), "minmax"),
        (
            vec![
                "spatial",
                "moran",
                "d",
                "-y",
                "y",
                "--coord-x",
                "cx",
                "--coord-y",
                "cy",
                "--alternative",
                "bigger",
            ],
            "two.sided",
        ),
    ]);
}

/// CLI-14: Moran's alternative by R's names. R: spatial_weights_moran moran (moran.test on the
/// 4-NN W, x1).
#[test]
fn test_validate_cli_moran_alternative() {
    let case = Case::load("spatial_weights_moran");
    let s = temp_session();
    let d = s.load(&case);
    let m = [
        "spatial",
        "moran",
        d.as_str(),
        "-y",
        "x1",
        "--coord-x",
        "cx",
        "--coord-y",
        "cy",
        "-k",
        "4",
    ];
    let two = s.json(&argv(&m, &["--alternative", "two.sided"]));
    let gt = s.json(&argv(&m, &["--alternative", "greater"]));
    Check::new(&case)
        .named_subset(
            "moran",
            &["I"],
            &["I"],
            &[num(field(&two, "moran_i"))],
            Tol::coef(80),
        )
        .named_subset(
            "moran",
            &["p_two_sided"],
            &["p_two_sided"],
            &[num(field(&two, "p_value"))],
            Tol::p_value(80),
        )
        .named_subset(
            "moran",
            &["p_rand"],
            &["p_rand"],
            &[num(field(&gt, "p_value"))],
            Tol::p_value(80),
        )
        .finish();
    assert_eq!(field(&two, "alternative"), "two.sided");
    assert_eq!(field(&gt, "alternative"), "greater");
    assert_keys(&two, &["n_islands", "weight_style"]);
}

// ---------------------------------------------------------------------------------------------
// ml
// ---------------------------------------------------------------------------------------------

/// CLI-23: --transform computed the scores and never printed them. R: cli_pca_scores
/// (prcomp(hclust_60)$x, centred, unscaled).
#[test]
fn test_validate_cli_pca_transform() {
    let case = Case::load("cli_pca_scores");
    let s = temp_session();
    let d = s.load(&case);
    let csv = s.dir().join("scores.csv");
    let j = s.json(&[
        "ml",
        "pca",
        &d,
        "--cols",
        "a",
        "b",
        "c",
        "--scores-file",
        csv.to_str().unwrap(),
    ]);
    let rows: Vec<Vec<f64>> = field(&j, "scores")
        .as_array()
        .unwrap()
        .iter()
        .map(f64s)
        .collect();
    assert_eq!((rows.len(), rows[0].len()), (60, 3));
    let r = case.vector("scores"); // column-major 60 x 3
    let mut aligned = Vec::new();
    for k in 0..3 {
        let dot: f64 = (0..60).map(|i| rows[i][k] * r[k * 60 + i]).sum();
        // a component's sign is arbitrary
        aligned.extend((0..60).map(|i| rows[i][k] * dot.signum()));
    }
    let sdev: Vec<f64> = f64s(field(&j, "explained_variance"))
        .iter()
        .map(|v| v.sqrt())
        .collect();
    Check::new(&case)
        .vector("scores", &aligned, Tol::coef(60))
        .vector("sdev", &sdev, Tol::stat(60))
        .finish();
    let file = std::fs::read_to_string(&csv).unwrap();
    assert_eq!(file.lines().next(), Some("PC1,PC2,PC3"));
    assert_eq!(file.lines().count(), 61);
    let first: Vec<f64> = file
        .lines()
        .nth(1)
        .unwrap()
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(first, rows[0], "the file holds the JSON scores");
    let text = s.ok(&["ml", "pca", &d, "--cols", "a", "b", "c", "--transform"]);
    assert!(
        text.contains("Scores (first 10 of 60 observations"),
        "{text}"
    );
}

/// CLI-14: ml hierarchical knew four linkages (exit 1 otherwise). R: hclust_methods
/// average_cutree3 and average_height (hclust(dist(hclust_60), "average")).
#[test]
fn test_validate_cli_linkage_names() {
    let case = Case::load("hclust_methods");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&[
        "ml",
        "hierarchical",
        &d,
        "--cols",
        "a",
        "b",
        "c",
        "-n",
        "3",
        "--linkage",
        "upgma",
    ]);
    assert!(
        same_partition(&f64s(field(&j, "labels")), &case.vector("average_cutree3")),
        "{j:#}"
    );
    Check::new(&case)
        .vector(
            "average_height",
            &f64s(field(&j, "merge_distances")),
            Tol::coef(60),
        )
        .finish();
    assert_eq!(field(&j, "linkage"), "average");
    assert_usage_errors(&[
        (
            vec![
                "ml",
                "hierarchical",
                "d",
                "--cols",
                "a",
                "b",
                "--linkage",
                "ward3",
            ],
            "mcquitty",
        ),
        (
            vec![
                "ml",
                "random-forest",
                "d",
                "--cols",
                "a",
                "-y",
                "y",
                "--max-features",
                "0.33",
            ],
            "log2",
        ),
    ]);
}

/// k-means reports its convergence (Lloyd's stop rule) and warnings.
#[test]
fn test_cli_kmeans_notes() {
    let s = temp_session();
    let d = load_repo_csv(&s, "validation/datasets/generated/hclust_60.csv", "h60");
    let args = [
        "ml", "kmeans", &d, "--cols", "a", "b", "c", "-k", "3", "--seed", "1",
    ];
    let j = s.json(&args);
    assert_keys(&j, &["converged", "iterations", "warnings"]);
    assert!(s.ok(&args).contains("Converged: "));
}

// ---------------------------------------------------------------------------------------------
// viz
// ---------------------------------------------------------------------------------------------

/// DATA-24 CLI part: viz residuals passed no leverage, so the core used the one-predictor formula
/// for a three-regressor model. R: cli_viz_residuals (lm(mpg ~ wt + hp + qsec): hatvalues,
/// rstandard, cooks.distance).
#[test]
fn test_validate_cli_viz_residuals_hat_values() {
    let case = Case::load("cli_viz_residuals");
    let s = temp_session();
    let d = s.load(&case);
    let png = s.dir().join("r.png");
    let j = s.json(&[
        "viz",
        "residuals",
        &d,
        "-y",
        "mpg",
        "-x",
        "wt",
        "hp",
        "qsec",
        "-f",
        png.to_str().unwrap(),
    ]);
    Check::new(&case)
        .vector("hat", &f64s(field(&j, "leverage")), Tol::coef(32))
        .vector(
            "rstandard",
            &f64s(field(&j, "standardized_residuals")),
            Tol::stat(32),
        )
        .vector("cooks", &f64s(field(&j, "cooks_distance")), Tol::stat(32))
        .scalar("sigma", num(field(&j, "sigma")), Tol::stat(32))
        .scalar("rank", num(field(&j, "n_params")), Tol::exact())
        .scalar("n", num(field(&j, "n_obs")), Tol::exact())
        .finish();
    for suffix in ["", "_qq", "_scale_location", "_leverage"] {
        assert!(
            s.dir().join(format!("r{suffix}.png")).exists(),
            "r{suffix}.png"
        );
    }
    assert_eq!(strings(field(&j, "files")).len(), 4);
    assert_keys(&j, &["warnings", "message"]);
}

/// CLI-14: viz dendrogram ran Ward for any linkage it did not know; its JSON names the linkage.
#[test]
fn test_cli_viz_dendrogram_linkage() {
    let s = temp_session();
    let d = load_repo_csv(&s, "validation/datasets/generated/hclust_60.csv", "h60");
    let png = s.dir().join("tree.png");
    let jd = s.json(&[
        "viz",
        "dendrogram",
        &d,
        "--cols",
        "a",
        "b",
        "c",
        "--linkage",
        "upgmc",
        "-f",
        png.to_str().unwrap(),
    ]);
    assert_eq!(field(&jd, "linkage"), "centroid");
    assert!(png.exists());
    assert_usage_errors(&[(
        vec![
            "viz",
            "dendrogram",
            "d",
            "--cols",
            "a",
            "b",
            "--linkage",
            "ward3",
            "-f",
            "t.png",
        ],
        "centroid",
    )]);
}
