//! T3-WP60: regression, discrete, panel, survival, data, munge and script commands through the p2a CLI.

use crate::common::*;
use serde_json::{Value, json};

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| {
            x.as_str()
                .unwrap_or_else(|| panic!("not a string: {x}"))
                .to_string()
        })
        .collect()
}

fn warnings(v: &Value) -> Vec<String> {
    strs(&v["warnings"])
}

/// "<category>:<variable>" names and values of a multinomial-logit JSON matrix; categories[0] is the
/// reference.
fn multinom_named(v: &Value, key: &str) -> (Vec<String>, Vec<f64>) {
    let (cats, vars) = (strs(&v["categories"]), strs(&v["variables"]));
    let mut out = (Vec::new(), Vec::new());
    for (j, row) in v[key].as_array().unwrap().iter().enumerate() {
        for (k, x) in f64s(row).into_iter().enumerate() {
            out.0.push(format!("{}:{}", cats[j + 1], vars[k]));
            out.1.push(x);
        }
    }
    out
}

// =================================================================================================
// Discrete choice
// =================================================================================================

/// CLI-17: logit/probit print glm.fit's convergence and every core warning, in text and JSON
/// (disc_binary_glm: ysep is separated by x1 + x2 and by neither alone; x1b = 2 x1 is aliased).
#[test]
fn test_cli_logit_prints_converged_and_warnings() {
    let case = Case::load("disc_binary_glm");
    let s = temp_session();
    let d = s.load(&case);
    let sep = [
        "discrete",
        "logit",
        d.as_str(),
        "-y",
        "ysep",
        "-x",
        "x1",
        "x2",
    ];
    let text = s.ok(&sep);
    // a separated fit has no MLE: the iteration count is a path value, so only the line is asserted
    assert_eq!(
        text.lines()
            .filter(|l| l.starts_with("Converged: "))
            .count(),
        1,
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l == "Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred"),
        "{text}"
    );
    let sep_line = text
        .lines()
        .find(|l| {
            l.starts_with("Warning: Separation: fitted probabilities are numerically 0 or 1 for ")
        })
        .unwrap_or_else(|| panic!("no separation warning:\n{text}"));
    // most fitted values reach glm's clamp before its criterion stops the fit
    let k: usize = sep_line
        .split("for ")
        .nth(1)
        .and_then(|r| r.split(" of 400 observations").next())
        .and_then(|k| k.trim().parse().ok())
        .unwrap_or_else(|| panic!("no count in {sep_line:?}"));
    assert!(k >= 200, "{sep_line}");
    let v = s.json(&sep);
    assert!(v["converged"].is_boolean(), "{v}");
    assert!(v["iterations"].is_u64(), "{v}");
    assert_eq!(v["separation"], true);
    let w = warnings(&v);
    assert!(
        w.iter()
            .any(|x| x == "glm.fit: fitted probabilities numerically 0 or 1 occurred"),
        "{w:?}"
    );
    for x in &w {
        assert!(text.contains(x.as_str()), "text lacks {x:?}");
    }
    // aliased x1b: NA (null) coefficient and R's warning
    let al = [
        "discrete",
        "logit",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x1b",
        "x2",
        "d",
    ];
    let v = s.json(&al);
    assert_eq!(v["aliased"], json!(["x1b"]));
    let t = json_table(&v, "t_value");
    Check::new(&case)
        .named("aliased_coef", &t.names, &t.coef, Tol::coef(400))
        .finish();
    assert!(
        s.ok(&al)
            .contains("Warning: Coefficients not defined because of singularities: x1b")
    );
    // a probit that converges quietly: the Converged line, no Warning line, "warnings": []
    let pr = [
        "discrete",
        "probit",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "d",
    ];
    let text = s.ok(&pr);
    assert!(
        text.lines().any(|l| l.starts_with("Converged: yes (")),
        "{text}"
    );
    assert!(!text.contains("Warning:"), "{text}");
    assert_eq!(s.json(&pr)["warnings"], json!([]));
}

/// logit/probit JSON against glm (disc_binary_glm). The logit is the MLE to 1e-13 at p2a's default
/// control. The probit is compared with the tight MLE: p2a's default control (epsilon 1e-10, maxit
/// 100), tighter than R's glm.control (1e-8, 25): registry deviation ECON-60.
#[test]
fn test_validate_cli_logit_probit_glm() {
    let case = Case::load("disc_binary_glm");
    let n = case.scalar("n") as usize;
    let s = temp_session();
    let d = s.load(&case);
    for model in ["logit", "probit"] {
        let v = s.json(&[
            "discrete",
            model,
            d.as_str(),
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "d",
        ]);
        let t = json_table(&v, "t_value"); // the z statistic, under T0's key
        let mut c = Check::new(&case);
        if model == "logit" {
            c.named("logit_coef", &t.names, &t.coef, Tol::coef(n))
                .named("logit_se", &t.names, &t.se, Tol::se(n))
                .named("logit_p", &t.names, &t.p, Tol::p_value(n))
                .named("logit_z", &t.names, &t.stat, Tol::stat(n))
                .scalar("logit_loglik", num(&v["log_likelihood"]), Tol::stat(n))
                .scalar("logit_aic", num(&v["aic"]), Tol::stat(n))
                .scalar(
                    "logit_null_loglik",
                    num(&v["log_likelihood_null"]),
                    Tol::stat(n),
                );
        } else {
            c.named("probit_coef_tight", &t.names, &t.coef, Tol::iterative(1e-6))
                .named("probit_se_tight", &t.names, &t.se, Tol::iterative(1e-6))
                .scalar(
                    "probit_loglik_tight",
                    num(&v["log_likelihood"]),
                    Tol::stat(n),
                )
                .scalar("probit_aic_tight", num(&v["aic"]), Tol::stat(n));
        }
        c.finish();
        assert_eq!(v["aliased"], json!([]));
        assert_eq!(v["separation"], false);
        assert_eq!(v["converged"], true);
    }
}

/// CLI-23 (mlogit part): --base reaches run_multinom (disc_multinom_overlap; nnet::multinom re-solved
/// by Newton).
#[test]
fn test_validate_cli_mlogit_base() {
    let case = Case::load("disc_multinom_overlap");
    let n = case.scalar("n") as usize; // 300
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "discrete",
        "mlogit",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
    ];
    let v = s.json(&argv(&base, &["--base", "B"]));
    assert_eq!(v["reference_category"], "B");
    let (names, coef) = multinom_named(&v, "coefficients");
    let (_, se) = multinom_named(&v, "std_errors");
    Check::new(&case)
        .named("refB_coef_tight", &names, &coef, Tol::iterative(1e-8))
        .named("refB_se_tight", &names, &se, Tol::iterative(1e-6))
        .scalar("loglik", num(&v["log_likelihood"]), Tol::stat(n))
        .finish();
    assert_eq!(v["converged"], true);
    assert_eq!(v["separation"], false);
    let text = s.ok(&argv(&base, &["--base", "B"]));
    assert!(text.contains("Reference category: B"), "{text}");
    assert!(
        text.contains("--- Category: A (vs B) ---") && text.contains("--- Category: C (vs B) ---"),
        "{text}"
    );
    let v0 = s.json(&base); // nnet's default: the first level
    assert_eq!(v0["reference_category"], "A");
    let (n0, c0) = multinom_named(&v0, "coefficients");
    Check::new(&case)
        .named("coef_tight", &n0, &c0, Tol::iterative(1e-8))
        .finish();
    let e = s.err(&argv(&base, &["--base", "Z"]));
    assert!(e.contains("'Z'"), "{e}");
}

/// WP18 surface: clogit has alternative-specific constants by default (mlogit), --asc false is
/// `| 0`, --reflevel (alias of --reference) moves the constants; every coefficient is printed
/// (disc_mlogit_fishing).
#[test]
fn test_validate_cli_clogit_asc_reflevel() {
    let case = Case::load("disc_mlogit_fishing");
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "discrete",
        "clogit",
        d.as_str(),
        "--choice-id",
        "id",
        "--alt-id",
        "alt",
        "-y",
        "chosen",
        "-x",
        "price",
        "catch",
    ];
    let v = s.json(&base);
    let t = json_table(&v, "z_value");
    assert_eq!(
        t.names,
        [
            "(Intercept):boat",
            "(Intercept):charter",
            "(Intercept):pier",
            "price",
            "catch"
        ]
    );
    Check::new(&case)
        .named(
            "generic_asc_coef_tight",
            &t.names,
            &t.coef,
            Tol::iterative(1e-9),
        )
        .named(
            "generic_asc_se_tight",
            &t.names,
            &t.se,
            Tol::iterative(1e-8),
        )
        .scalar(
            "generic_asc_loglik",
            num(&v["log_likelihood"]),
            Tol::stat(1182),
        )
        .finish();
    assert_eq!(
        (v["asc"].clone(), v["reference_alternative"].clone()),
        (json!(true), json!("beach"))
    );
    let v0 = s.json(&argv(&base, &["--asc", "false"]));
    let t0 = json_table(&v0, "z_value");
    Check::new(&case)
        .named(
            "generic_noasc_coef_tight",
            &t0.names,
            &t0.coef,
            Tol::iterative(1e-9),
        )
        .named(
            "generic_noasc_se_tight",
            &t0.names,
            &t0.se,
            Tol::iterative(1e-8),
        )
        .scalar(
            "generic_noasc_loglik",
            num(&v0["log_likelihood"]),
            Tol::stat(1182),
        )
        .finish();
    // reference pier: the same optimum reparametrised (constants minus the pier constant)
    let vp = s.json(&argv(&base, &["--reflevel", "pier"]));
    let tp = json_table(&vp, "z_value");
    let b = |nm: &str| case.named("generic_asc_coef_tight", nm);
    let pier = b("(Intercept):pier");
    let want = [
        ("(Intercept):beach", -pier),
        ("(Intercept):boat", b("(Intercept):boat") - pier),
        ("(Intercept):charter", b("(Intercept):charter") - pier),
        ("price", b("price")),
        ("catch", b("catch")),
    ];
    for (nm, w) in want {
        let i = tp
            .names
            .iter()
            .position(|x| x == nm)
            .unwrap_or_else(|| panic!("{nm} missing: {:?}", tp.names));
        assert!(
            Tol::iterative(1e-9).allows(tp.coef[i], w),
            "{nm}: {} vs {w}",
            tp.coef[i]
        );
    }
    assert!(Tol::stat(1182).allows(
        num(&vp["log_likelihood"]),
        case.scalar("generic_asc_loglik")
    ));
    let text = s.ok(&base);
    for nm in [
        "(Intercept):boat",
        "(Intercept):charter",
        "(Intercept):pier",
        "price",
        "catch",
    ] {
        assert!(
            text.lines().any(|l| l.starts_with(nm)),
            "{nm} row missing:\n{text}"
        );
    }
    assert!(
        text.contains("Alternative-specific constants: yes"),
        "{text}"
    );
}

/// CLI-3: `discrete poisson` fits glm(family = poisson) (cli_discrete_poisson). Before: every call
/// failed in run_feglm, which refuses an empty fixed-effect list.
#[test]
fn test_validate_cli_discrete_poisson() {
    let case = Case::load("cli_discrete_poisson");
    let n = case.scalar("n") as usize; // 400
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/generated/negbin.csv"),
        "negbin",
        &[],
    );
    let v = s.json(&["discrete", "poisson", "negbin", "-y", "y", "-x", "x1", "x2"]);
    let names = strs(&v["variables"]);
    assert_eq!(names, ["(Intercept)", "x1", "x2"]);
    Check::new(&case)
        .named("coef", &names, &f64s(&v["coefficients"]), Tol::coef(n))
        .named("se", &names, &f64s(&v["std_errors"]), Tol::se(n))
        .named("z", &names, &f64s(&v["z_stats"]), Tol::stat(n))
        .named("p", &names, &f64s(&v["p_values"]), Tol::p_value(n))
        .scalar("loglik", num(&v["log_likelihood"]), Tol::stat(n))
        .scalar("aic", num(&v["aic"]), Tol::stat(n))
        .scalar("deviance", num(&v["deviance"]), Tol::stat(n))
        .scalar("null_deviance", num(&v["null_deviance"]), Tol::stat(n))
        .scalar("n", num(&v["n_obs"]), Tol::exact())
        .finish();
    assert_eq!(v["converged"], true);
    assert!(v["warnings"].as_array().is_some(), "{v}");
    // missing cells: rows dropped jointly (glm's na.omit), reported as T2's missing note
    let w = s.json(&[
        "discrete", "poisson", "negbin", "-y", "y_na", "-x", "x1_na", "x2",
    ]);
    let names = strs(&w["variables"]);
    Check::new(&case)
        .named("na_coef", &names, &f64s(&w["coefficients"]), Tol::coef(n))
        .named("na_se", &names, &f64s(&w["std_errors"]), Tol::se(n))
        .scalar(
            "na_n_dropped",
            num(&w["missing"]["n_dropped"]),
            Tol::exact(),
        )
        .finish();
    let text = s.ok(&["discrete", "poisson", "negbin", "-y", "y", "-x", "x1", "x2"]);
    assert!(
        text.lines().any(|l| l.starts_with("Converged: yes (")),
        "{text}"
    );
}

// =================================================================================================
// Survival
// =================================================================================================

/// WP19 surface: --ties as coxph (Efron, Breslow), robust (Lin-Wei) and cluster(inst) SEs
/// (surv_cox_robust, lung).
#[test]
fn test_validate_cli_cox_ties_robust_cluster() {
    let case = Case::load("surv_cox_robust");
    let n = case.scalar("n") as usize; // 227
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "survival",
        "cox",
        d.as_str(),
        "-t",
        "time",
        "-e",
        "status",
        "-x",
        "age",
        "sex",
        "ph.ecog",
    ];
    for ties in ["efron", "breslow"] {
        let v = s.json(&argv(&base, &["--ties", ties]));
        let names = strs(&v["variables"]);
        assert_eq!(
            (v["ties"].clone(), v["se_type"].clone()),
            (json!(ties), json!("model"))
        );
        Check::new(&case)
            .named(
                &format!("{ties}_coef"),
                &names,
                &f64s(&v["coefficients"]),
                Tol::coef(n),
            )
            .named(
                &format!("{ties}_se"),
                &names,
                &f64s(&v["std_errors"]),
                Tol::se(n),
            )
            .named(
                &format!("{ties}_z"),
                &names,
                &f64s(&v["z_stats"]),
                Tol::stat(n),
            )
            .named(
                &format!("{ties}_p"),
                &names,
                &f64s(&v["p_values"]),
                Tol::p_value(n),
            )
            .scalar(
                &format!("{ties}_loglik"),
                num(&v["log_likelihood"]),
                Tol::stat(n),
            )
            .scalar("n", num(&v["n_obs"]), Tol::exact())
            .finish();
        assert_eq!(v["converged"], true);
    }
    // the default is coxph's (Efron)
    assert_eq!(s.json(&base)["ties"], "efron");
    let rob = argv(&base, &["--ties", "efron", "--robust"]);
    let v = s.json(&rob);
    let names = strs(&v["variables"]);
    assert_eq!(v["se_type"], "robust");
    Check::new(&case)
        .named(
            "efron_se_robust",
            &names,
            &f64s(&v["std_errors"]),
            Tol::se(n),
        )
        .named(
            "efron_se",
            &names,
            &f64s(&v["naive_std_errors"]),
            Tol::se(n),
        )
        .named("efron_z_robust", &names, &f64s(&v["z_stats"]), Tol::stat(n))
        .named(
            "efron_p_robust",
            &names,
            &f64s(&v["p_values"]),
            Tol::p_value(n),
        )
        .scalar(
            "efron_rscore_robust",
            num(&v["robust_score_test"]),
            Tol::stat(n),
        )
        .scalar(
            "efron_rscore_p_robust",
            num(&v["robust_score_p_value"]),
            Tol::p_value(n),
        )
        .finish();
    let text = s.ok(&rob);
    assert!(
        text.contains("Standard errors: robust (Lin-Wei sandwich)"),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l.starts_with("Variable") && l.contains("Robust SE")),
        "{text}"
    );
    let v = s.json(&argv(&base, &["--ties", "efron", "--cluster", "inst"]));
    let names = strs(&v["variables"]);
    assert_eq!(v["se_type"], "cluster");
    Check::new(&case)
        .named(
            "efron_coef_cluster",
            &names,
            &f64s(&v["coefficients"]),
            Tol::coef(n),
        )
        .named(
            "efron_se_model_cluster",
            &names,
            &f64s(&v["naive_std_errors"]),
            Tol::se(n),
        )
        .named(
            "efron_se_cluster",
            &names,
            &f64s(&v["std_errors"]),
            Tol::se(n),
        )
        .named(
            "efron_z_cluster",
            &names,
            &f64s(&v["z_stats"]),
            Tol::stat(n),
        )
        .named(
            "efron_p_cluster",
            &names,
            &f64s(&v["p_values"]),
            Tol::p_value(n),
        )
        .scalar(
            "efron_rscore_cluster",
            num(&v["robust_score_test"]),
            Tol::stat(n),
        )
        .scalar("n_clusters", num(&v["n_clusters"]), Tol::exact())
        .scalar("n_cluster_obs", num(&v["n_obs"]), Tol::exact())
        .finish();
}

/// WP19 surface: --dist as survreg (surv_aft, lung): coefficients, SEs, scale, Log(scale), logLik,
/// AIC, BIC.
#[test]
fn test_validate_cli_aft_distributions() {
    let case = Case::load("surv_aft");
    let n = case.scalar("n") as usize; // 227
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "survival",
        "aft",
        d.as_str(),
        "-t",
        "time",
        "-e",
        "status",
        "-x",
        "age",
        "sex",
        "ph.ecog",
    ];
    for dist in ["weibull", "exponential", "lognormal", "loglogistic"] {
        let v = s.json(&argv(&base, &["--dist", dist]));
        let names = strs(&v["variables"]);
        let mut c = Check::new(&case);
        c.named(
            &format!("{dist}_coef"),
            &names,
            &f64s(&v["coefficients"]),
            Tol::coef(n),
        )
        .named(
            &format!("{dist}_se"),
            &names,
            &f64s(&v["std_errors"]),
            Tol::se(n),
        )
        .named(
            &format!("{dist}_z"),
            &names,
            &f64s(&v["z_stats"]),
            Tol::stat(n),
        )
        .named(
            &format!("{dist}_p"),
            &names,
            &f64s(&v["p_values"]),
            Tol::p_value(n),
        )
        .scalar(&format!("{dist}_scale"), num(&v["scale"]), Tol::stat(n))
        .scalar(
            &format!("{dist}_loglik"),
            num(&v["log_likelihood"]),
            Tol::stat(n),
        )
        .scalar(&format!("{dist}_aic"), num(&v["aic"]), Tol::stat(n))
        .scalar(&format!("{dist}_bic"), num(&v["bic"]), Tol::stat(n));
        if dist == "exponential" {
            assert_eq!(v["scale_fixed"], true);
            assert!(v["log_scale_se"].is_null(), "{v}");
        } else {
            assert_eq!(v["scale_fixed"], false);
            c.scalar(
                &format!("{dist}_se_log_scale"),
                num(&v["log_scale_se"]),
                Tol::se(n),
            );
        }
        c.finish();
        assert_eq!(v["converged"], true);
    }
    assert!(
        s.ok(&argv(&base, &["--dist", "exponential"]))
            .contains("Scale fixed at 1")
    );
    let text = s.ok(&argv(&base, &["--dist", "weibull"]));
    let row: Vec<&str> = text
        .lines()
        .find(|l| l.starts_with("Log(scale)"))
        .unwrap_or_else(|| panic!("{text}"))
        .split_whitespace()
        .collect();
    assert!(
        Tol::printed(4).allows(row[1].parse().unwrap(), case.scalar("weibull_scale").ln()),
        "{row:?}"
    );
    assert!(
        Tol::printed(4).allows(row[3].parse().unwrap(), case.scalar("weibull_z_log_scale")),
        "{row:?}"
    );
}

/// CLI-23 (competing-risks part): CIF SEs and intervals at --conf, as survfit (cli_competing_risks).
#[test]
fn test_validate_cli_competing_risks_ci() {
    let case = Case::load("cli_competing_risks");
    let n = case.scalar("n") as usize; // 200
    let s = temp_session();
    let d = s.load(&case); // "cr_sim"
    let base = [
        "survival",
        "competing-risks",
        d.as_str(),
        "-t",
        "time",
        "-e",
        "event_type",
    ];
    for (extra, tag, level) in [(&[][..], "95", 0.95), (&["--conf", "0.5"][..], "50", 0.5)] {
        let v = s.json(&argv(&base, extra));
        assert_eq!(num(&v["conf_level"]), level);
        assert_eq!(v["se_method"], "delta method (as survfit)");
        let cifs = v["cumulative_incidence"].as_array().unwrap();
        assert_eq!(cifs.len(), 2);
        for (i, cif) in cifs.iter().enumerate() {
            let k = i + 1;
            assert_eq!(cif["event_type"], k);
            Check::new(&case)
                .vector("times", &f64s(&cif["times"]), Tol::new(1e-15, 0.0)) // JSON float parsing may move 1 ulp
                .vector(&format!("cif_{k}"), &f64s(&cif["incidence"]), Tol::coef(n))
                .vector(
                    &format!("survfit_se_{k}"),
                    &f64s(&cif["std_errors"]),
                    Tol::se(n),
                )
                .vector(
                    &format!("survfit_ci{tag}_lower_{k}"),
                    &f64s(&cif["ci_lower"]),
                    Tol::stat(n),
                )
                .vector(
                    &format!("survfit_ci{tag}_upper_{k}"),
                    &f64s(&cif["ci_upper"]),
                    Tol::stat(n),
                )
                .finish();
        }
        assert_eq!(v["n_events_by_type"], json!({"1": 80, "2": 65}));
    }
    let text = s.ok(&base);
    assert!(
        text.contains("Standard errors: delta method (as survfit); intervals on the log scale"),
        "{text}"
    );
    let mut lines = text.lines().skip_while(|l| !l.starts_with("Event type 1"));
    // header, "Time CIF SE Lower Upper", dashes, then the first event time 0.033633
    let first: Vec<f64> = lines
        .nth(3)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let want = [
        case.vector("times")[0],
        case.vector("cif_1")[0],
        case.vector("survfit_se_1")[0],
        case.vector("survfit_ci95_lower_1")[0],
        case.vector("survfit_ci95_upper_1")[0],
    ];
    for (a, e) in first.iter().zip(want) {
        assert!(Tol::printed(4).allows(*a, e), "{first:?} vs {want:?}");
    }
}

// =================================================================================================
// Table tests: every computing command prints one JSON document with its notes; option values
// parse through the cores' spellings
// =================================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Notes {
    Plain,
    Iterative,
    Report,
    ReportIterative,
}

struct Cmd {
    args: &'static [&'static str],
    notes: Notes,
}

const fn c(args: &'static [&'static str], notes: Notes) -> Cmd {
    Cmd { args, notes }
}

#[allow(unused_imports)]
use Notes::{Iterative as It, Plain as Pl, Report as Rp, ReportIterative as RpIt};

const DATASETS: &[&str] = &[
    "validation/datasets/grunfeld.csv",
    "validation/datasets/generated/binary_glm.csv",
    "validation/datasets/generated/multinom_overlap.csv",
    "validation/datasets/generated/mlogit_fishing.csv",
    "validation/datasets/generated/ordered.csv",
    "validation/datasets/generated/negbin.csv",
    "validation/datasets/generated/zeroinfl.csv",
    "validation/datasets/generated/mixed_logit.csv",
    "validation/datasets/generated/lung.csv",
    "validation/datasets/generated/cr_sim.csv",
];

const DISCRETE: &[Cmd] = &[
    c(
        &[
            "discrete",
            "logit",
            "binary_glm",
            "-y",
            "ysep",
            "-x",
            "x1",
            "x2",
        ],
        It,
    ),
    c(
        &[
            "discrete",
            "probit",
            "binary_glm",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "d",
        ],
        It,
    ),
    c(
        &[
            "discrete", "o-logit", "ordered", "-y", "y", "-x", "x1", "x2",
        ],
        It,
    ),
    c(
        &[
            "discrete", "o-probit", "ordered", "-y", "y", "-x", "x1", "x2",
        ],
        It,
    ),
    c(
        &[
            "discrete",
            "mlogit",
            "multinom_overlap",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
        ],
        It,
    ),
    c(
        &["discrete", "negbin", "negbin", "-y", "y", "-x", "x1", "x2"],
        It,
    ),
    c(
        &["discrete", "poisson", "negbin", "-y", "y", "-x", "x1", "x2"],
        It,
    ),
    c(
        &["discrete", "zip", "zeroinfl", "-y", "y", "-x", "x1", "x2"],
        It,
    ),
    c(
        &[
            "discrete", "zinb", "zeroinfl", "-y", "y_nb", "-x", "x1", "x2",
        ],
        It,
    ),
    c(
        &[
            "discrete", "hurdle", "zeroinfl", "-y", "y", "-x", "x1", "x2",
        ],
        It,
    ),
    c(
        &[
            "discrete",
            "clogit",
            "mlogit_fishing",
            "--choice-id",
            "id",
            "--alt-id",
            "alt",
            "-y",
            "chosen",
            "-x",
            "price",
            "catch",
        ],
        It,
    ),
    c(
        &[
            "discrete",
            "mixed-logit",
            "mixed_logit",
            "--choice-id",
            "id",
            "--alt-id",
            "alt",
            "-y",
            "choice",
            "-x",
            "x",
            "w",
            "--random-vars",
            "x",
            "--n-draws",
            "50",
        ],
        It,
    ),
];

const DISCRETE_OK: &[&[&str]] = &[
    &[
        "discrete", "hurdle", "zeroinfl", "-y", "y", "-x", "x1", "x2", "--dist", "nb",
    ],
    &[
        "discrete",
        "clogit",
        "mlogit_fishing",
        "--choice-id",
        "id",
        "--alt-id",
        "alt",
        "-y",
        "chosen",
        "-x",
        "price",
        "catch",
        "--asc",
        "false",
    ],
];

const DISCRETE_REFUSED: &[(&[&str], &[&str])] = &[
    (
        &[
            "discrete", "hurdle", "zeroinfl", "-y", "y", "-x", "x1", "--dist", "zip",
        ],
        &["poisson", "negbin"],
    ),
    (
        &[
            "discrete",
            "clogit",
            "mlogit_fishing",
            "--choice-id",
            "id",
            "--alt-id",
            "alt",
            "-y",
            "chosen",
            "-x",
            "price",
            "--asc",
            "maybe",
        ],
        // WP06's boolean-value parser (BoolishValueParser) names the value, not the spellings
        &["'maybe'", "not a boolean"],
    ),
];

const SURVIVAL: &[Cmd] = &[
    c(
        &["survival", "km", "lung", "-t", "time", "-e", "status"],
        Pl,
    ),
    c(
        &[
            "survival", "log-rank", "lung", "-t", "time", "-e", "status", "-g", "sex",
        ],
        Pl,
    ),
    c(
        &[
            "survival", "cox", "lung", "-t", "time", "-e", "status", "-x", "age", "sex", "ph.ecog",
        ],
        It,
    ),
    c(
        &[
            "survival", "aft", "lung", "-t", "time", "-e", "status", "-x", "age", "sex", "ph.ecog",
        ],
        It,
    ),
    c(
        &[
            "survival",
            "competing-risks",
            "cr_sim",
            "-t",
            "time",
            "-e",
            "event_type",
        ],
        Pl,
    ),
];

const SURVIVAL_OK: &[&[&str]] = &[
    &[
        "survival", "cox", "lung", "-t", "time", "-e", "status", "-x", "age", "sex", "--ties",
        "BRESLOW",
    ],
    &[
        "survival",
        "aft",
        "lung",
        "-t",
        "time",
        "-e",
        "status",
        "-x",
        "age",
        "sex",
        "--dist",
        "log-normal",
    ],
];

const SURVIVAL_REFUSED: &[(&[&str], &[&str])] = &[
    (
        &[
            "survival", "cox", "lung", "-t", "time", "-e", "status", "-x", "age", "--ties", "exact",
        ],
        &["efron", "breslow"],
    ),
    (
        &[
            "survival", "aft", "lung", "-t", "time", "-e", "status", "-x", "age", "--dist", "gamma",
        ],
        &["weibull", "exponential", "lognormal", "loglogistic"],
    ),
];

const ALL: &[&[Cmd]] = &[DISCRETE, SURVIVAL];
const OPTIONS_OK: &[&[&[&str]]] = &[DISCRETE_OK, SURVIVAL_OK];
const OPTIONS_REFUSED: &[&[(&[&str], &[&str])]] = &[DISCRETE_REFUSED, SURVIVAL_REFUSED];

/// A session with every dataset the table tests use, each named after its file stem.
fn table_session() -> TempSession {
    let s = temp_session();
    for rel in DATASETS {
        let p = repo_path(rel);
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        s.load_as(&p, &name, &[]);
    }
    s
}

/// CLI-17, CLI-29: every computing command of these files prints one JSON document carrying
/// "warnings" (an array; T1-WP05's RegressionReport commands only when non-empty) and, for
/// iterative fits, "converged" and "iterations"; the text output shows each of those warnings and,
/// for iterative fits outside the report commands, a "Converged:" line. No command of these files
/// has an identification-diagnostics report: those belong to the causal commands.
#[test]
fn test_cli_json_has_warnings_and_diagnostics() {
    let s = table_session();
    let mut problems = Vec::new();
    for cmd in ALL.iter().flat_map(|g| g.iter()) {
        let what = cmd.args.join(" ");
        let out = s.run(&argv(&["-F", "json"], cmd.args));
        assert_no_panic(&out, &what);
        if !out.status.success() {
            problems.push(format!(
                "{what}: exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ));
            continue;
        }
        let v: Value = match serde_json::from_slice::<Value>(&out.stdout) {
            Ok(v) if v.is_object() => v,
            Ok(v) => {
                problems.push(format!("{what}: JSON is not an object: {v}"));
                continue;
            }
            Err(e) => {
                problems.push(format!("{what}: stdout is not one JSON document ({e})"));
                continue;
            }
        };
        let report = matches!(cmd.notes, Notes::Report | Notes::ReportIterative);
        let ws = match v.get("warnings") {
            Some(Value::Array(a)) if a.iter().all(Value::is_string) => strs(&v["warnings"]),
            None if report => Vec::new(),
            other => {
                problems.push(format!("{what}: \"warnings\" is {other:?}"));
                continue;
            }
        };
        if matches!(cmd.notes, Notes::Iterative | Notes::ReportIterative)
            && !(v["converged"].is_boolean() && v["iterations"].is_u64())
        {
            problems.push(format!("{what}: no converged/iterations"));
        }
        let t = s.run(cmd.args);
        assert_no_panic(&t, &what);
        let text = String::from_utf8_lossy(&t.stdout).into_owned();
        if !t.status.success() {
            problems.push(format!("{what}: text run failed"));
            continue;
        }
        for w in &ws {
            if !text.contains(w.as_str()) {
                problems.push(format!("{what}: text lacks warning {w:?}"));
            }
        }
        if cmd.notes == Notes::Iterative && !text.contains("Converged:") {
            problems.push(format!("{what}: text has no Converged: line"));
        }
    }
    assert!(
        problems.is_empty(),
        "{} problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// R1 / CLI-14 (this package's part): option values parse with their aliases, unknown ones exit 2
/// naming the accepted spellings.
#[test]
fn test_cli_option_values_parse_aliases_and_reject_unknown() {
    let s = table_session();
    for args in OPTIONS_OK.iter().flat_map(|g| g.iter()) {
        let out = s.run(args);
        assert_no_panic(&out, &args.join(" "));
        assert!(
            out.status.success(),
            "{}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    for (args, names) in OPTIONS_REFUSED.iter().flat_map(|g| g.iter()) {
        let out = s.run(args);
        let e = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{}: {e}", args.join(" "));
        for n in names.iter() {
            assert!(e.contains(n), "{}: {n} not listed in {e}", args.join(" "));
        }
    }
}
