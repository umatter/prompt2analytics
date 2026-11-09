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
fn test_validate_cli_logit_prints_converged_and_warnings() {
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
// Panel
// =================================================================================================

/// WP17 surface: --lag-selection aic|sic|hall per unit as plm::purtest (purtest_lag_selection;
/// dfcor = TRUE).
#[test]
fn test_validate_cli_unit_root_lag_selection() {
    let case = Case::load("purtest_lag_selection");
    let n = case.scalar("n_obs") as usize; // 480
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "panel",
        "unit-root",
        d.as_str(),
        "-v",
        "y",
        "--unit",
        "id",
        "--time",
        "t",
        "--test",
        "ips",
        "--pmax",
        "4",
    ];
    for rule in ["aic", "sic", "hall"] {
        let v = s.json(&argv(&base, &["--lag-selection", rule]));
        Check::new(&case)
            .vector(
                &format!("ips_{rule}_dfcor_lags"),
                &f64s(&v["unit_lags"]),
                Tol::exact(),
            )
            .vector(
                &format!("trho_{rule}_dfcor"),
                &f64s(&v["unit_statistics"]),
                Tol::stat(n),
            )
            .finish();
        assert!(v["warnings"].is_array(), "{v}");
    }
    let v = s.json(&argv(
        &base,
        &["--lag-selection", "aic", "--model", "trend"],
    ));
    Check::new(&case)
        .vector("trend_lags_aic_dfcor", &f64s(&v["unit_lags"]), Tol::exact())
        .vector(
            "trend_trho_aic_dfcor",
            &f64s(&v["unit_statistics"]),
            Tol::stat(n),
        )
        .finish();
    let want = case
        .vector("ips_aic_dfcor_lags")
        .iter()
        .map(|x| (*x as usize).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let text = s.ok(&argv(&base, &["--lag-selection", "aic"]));
    assert!(text.contains(&format!("Unit lags: {want}")), "{text}");
    let e = s.err(&argv(&base, &["--lag-selection", "fixed"]));
    assert!(e.contains("lags"), "{e}");
}

/// WP17 surface: regressors without within variation are dropped and named
/// (panel_absorbed_regressors, N 30, T 6).
#[test]
fn test_validate_cli_panel_absorbed_regressors() {
    let case = Case::load("panel_absorbed_regressors");
    let n = 180;
    let s = temp_session();
    let d = s.load(&case);
    let fe = [
        "panel",
        "fe",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "z",
        "--entity",
        "id",
    ];
    let v = s.json(&fe);
    let t = json_table(&v, "t_value");
    assert_eq!(t.names, ["x1", "x2"]);
    assert_eq!(strs(&v["dropped"]), ["z"]);
    Check::new(&case)
        .named("fe_coef", &t.names, &t.coef, Tol::coef(n))
        .named("fe_se", &t.names, &t.se, Tol::se(n))
        .finish();
    let w = warnings(&v);
    assert!(w.iter().any(|x| x.contains('z')), "{w:?}");
    let text = s.ok(&fe);
    for x in &w {
        assert!(text.contains(x.as_str()), "text lacks {x:?}");
    }
    let v = s.json(&[
        "panel",
        "hdfe",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "z",
        "w",
        "--fe",
        "id",
        "t",
    ]);
    let t = json_table(&v, "t_value");
    assert_eq!(t.names, ["x1", "x2"]);
    assert_eq!(strs(&v["dropped"]), ["z", "w"]);
    Check::new(&case)
        .named("hdfe2_coef", &t.names, &t.coef, Tol::coef(n))
        .finish();
    assert_eq!(v["converged"], true);
    assert!(v["iterations"].as_u64().unwrap() >= 1);
}

// =================================================================================================
// Regression
// =================================================================================================

/// WP10 surface: --se nid|iid|ker as summary.rq, BR coefficients (rq_br_se, tau 0.5, n 100).
#[test]
fn test_validate_cli_quantreg_se() {
    let case = Case::load("rq_br_se");
    let n = 100;
    let s = temp_session();
    let d = s.load(&case);
    for se in ["nid", "iid", "ker"] {
        let v = s.json(&[
            "reg",
            "quantreg",
            d.as_str(),
            "-y",
            "y",
            "-x",
            "x",
            "-t",
            "0.5",
            "--se",
            se,
        ]);
        let rows = v["coefficients"].as_array().unwrap();
        let names: Vec<String> = rows
            .iter()
            .map(|r| r["name"].as_str().unwrap().to_string())
            .collect();
        let col = |k: &str| {
            rows.iter()
                .map(|r| if r[k].is_null() { f64::NAN } else { num(&r[k]) })
                .collect::<Vec<f64>>()
        };
        Check::new(&case)
            .named("tau50_coef", &names, &col("estimate"), Tol::coef(n))
            .named(
                &format!("tau50_se_{se}"),
                &names,
                &col("std_error"),
                Tol::se(n),
            )
            .named(
                &format!("tau50_t_{se}"),
                &names,
                &col("t_value"),
                Tol::stat(n),
            )
            .named(
                &format!("tau50_p_{se}"),
                &names,
                &col("p_value"),
                Tol::p_value(n),
            )
            .named(
                &format!("tau50_ci_lower_{se}"),
                &names,
                &col("ci_lower_95"),
                Tol::stat(n),
            )
            .named(
                &format!("tau50_ci_upper_{se}"),
                &names,
                &col("ci_upper_95"),
                Tol::stat(n),
            )
            .scalar("tau50_objective", num(&v["objective"]), Tol::stat(n))
            .finish();
        assert_eq!(v["se_type"], se);
        assert!(v["se_method"].as_str().unwrap().starts_with(se), "{v}");
        assert_eq!(v["converged"], true);
    }
    let args = ["reg", "quantreg", d.as_str(), "-y", "y", "-x", "x"];
    let v = s.json(&args);
    let w = warnings(&v);
    assert!(!w.iter().any(|x| x.contains("rank")), "no rank note: {w:?}");
    let text = s.ok(&args);
    for x in &w {
        assert!(text.contains(&format!("Warning: {x}")), "{text}");
    }
    assert!(
        text.contains("Algorithm: Barrodale-Roberts simplex (br)"),
        "{text}"
    );
    assert!(
        s.json(&argv(&args, &["--method", "fn"]))["algorithm"]
            .as_str()
            .unwrap()
            .ends_with("(fn)")
    );
}

/// WP11 surface: compound symmetry within --group, rho by REML or ML (gls_compsymm, 30 groups of 5).
#[test]
fn test_validate_cli_gls_compsymm_group() {
    let case = Case::load("gls_compsymm");
    let n = 150;
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "reg",
        "gls",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x",
        "z",
        "--correlation",
        "cs",
        "--group",
        "g",
    ];
    for k in ["reml", "ml"] {
        let v = s.json(&argv(&base, &["--method", k]));
        let t = json_arrays(&v);
        Check::new(&case)
            .named(
                &format!("{k}_profile_coef"),
                &t.names,
                &t.coef,
                Tol::coef(n),
            )
            .named(&format!("{k}_profile_se"), &t.names, &t.se, Tol::se(n))
            .scalar(
                &format!("{k}_profile_rho"),
                num(&v["correlation_param"]),
                Tol::iterative(1e-6), // flat maximum
            )
            .scalar(
                &format!("{k}_profile_sigma"),
                num(&v["sigma"]),
                Tol::stat(n),
            )
            .scalar(
                &format!("{k}_profile_loglik"),
                num(&v["log_likelihood"]),
                Tol::stat(n),
            )
            // cross-check: nlme's own fit stops up to 3e-8 relative short in rho; coefficients
            // within 1e-6
            .named(
                &format!("{k}_coef"),
                &t.names,
                &t.coef,
                Tol::iterative(1e-6),
            )
            .scalar(&format!("{k}_aic"), num(&v["aic"]), Tol::stat(n))
            .scalar(&format!("{k}_bic"), num(&v["bic"]), Tol::stat(n))
            .scalar("n_groups", num(&v["n_groups"]), Tol::exact())
            .finish();
        assert_eq!(v["rho_estimated"], true);
        assert_eq!(v["method"].as_str().unwrap().to_lowercase(), k);
    }
    let v = s.json(&argv(&base, &["--rho", "0.3"]));
    let t = json_arrays(&v);
    Check::new(&case)
        .named("cs03_reml_coef", &t.names, &t.coef, Tol::coef(n))
        .named("cs03_reml_se", &t.names, &t.se, Tol::se(n))
        .finish();
    assert_eq!(v["rho_estimated"], false);
    // one group: compound symmetry is not identified
    let e = s.err(&[
        "reg",
        "gls",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x",
        "z",
        "--correlation",
        "cs",
    ]);
    assert!(e.to_lowercase().contains("group"), "{e}");
}

/// WP11 surface: the zero-width-neighbourhood warning reaches text and JSON
/// (smoothers_gcv_periodic_loess).
#[test]
fn test_validate_cli_loess_masspoint_warns() {
    let case = Case::load("smoothers_gcv_periodic_loess");
    let s = temp_session();
    let d = s.load(&case); // "smoothers_t3"
    let args = [
        "reg",
        "loess",
        d.as_str(),
        "-y",
        "ym",
        "-x",
        "xm",
        "--span",
        "0.5",
        "--degree",
        "1",
    ];
    let v = s.json(&args);
    let fit = f64s(&v["fitted_values_sample"]);
    let want = case.vector("loess_masspoint_expected");
    for i in 0..10 {
        assert!(
            Tol::coef(100).allows(fit[i], want[i]),
            "{i}: {} vs {}",
            fit[i],
            want[i]
        );
    }
    let w = warnings(&v);
    assert!(
        w.iter()
            .any(|x| x.contains("zero-width neighbourhood at 60 of 100 points")),
        "{w:?}"
    );
    let text = s.ok(&args);
    for x in &w {
        assert!(text.contains(&format!("Warning: {x}")), "{text}");
    }
}

/// WP11 surface: --criterion gcv (default, R's cv = FALSE) | cv, equal to the core in process. No R
/// pin: p2a's smoother is not R's yet.
#[test]
fn test_cli_smooth_spline_criterion() {
    use p2a_core::data::columns::{NaPolicy, numeric_column};
    use p2a_core::regression::smooth_spline::{SmoothSplineConfig, SplineCriterion, smooth_spline};
    let p = repo_path("validation/datasets/generated/smoothers_t3.csv");
    let s = temp_session();
    s.load_as(&p, "sm", &[]);
    let ds = p2a_core::Dataset::new(p2a_core::DataLoader::load_csv(&p).unwrap());
    let x = numeric_column(ds.df(), "x", NaPolicy::Fail).unwrap().0;
    let y = numeric_column(ds.df(), "y", NaPolicy::Fail).unwrap().0;
    let base = ["reg", "smooth-spline", "sm", "-y", "y", "-x", "x"];
    for (extra, crit, name) in [
        (&[][..], SplineCriterion::Gcv, "gcv"),
        (&["--criterion", "cv"][..], SplineCriterion::Cv, "cv"),
        (&["--criterion", "LOOCV"][..], SplineCriterion::Cv, "cv"),
    ] {
        let v = s.json(&argv(&base, extra));
        let r = smooth_spline(&x, &y, SmoothSplineConfig::with_criterion(crit)).unwrap();
        assert_eq!(v["criterion"], name);
        assert!(
            Tol::new(1e-15, 0.0).allows(num(&v["df"]), r.df)
                && Tol::new(1e-15, 0.0).allows(num(&v["lambda"]), r.lambda),
            "{v}"
        );
    }
    assert!(s.json(&argv(&base, &["--df", "5"]))["criterion"].is_null());
    assert!(s.ok(&base).contains("Criterion: GCV"));
}

/// WP09 surface: reg hac on an aliased design: x2 null, t and p on df = n - rank = 117
/// (cli_hac_aliased).
#[test]
fn test_validate_cli_hac_aliased_df() {
    let case = Case::load("cli_hac_aliased");
    let n = 120;
    let s = temp_session();
    let d = s.load(&case); // "ols_aliased"
    let v = s.json(&[
        "reg",
        "hac",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "x3",
        "--lags",
        "4",
    ]);
    let t = json_arrays(&v);
    assert_eq!(t.names, ["(Intercept)", "x1", "x2", "x3"]);
    assert!(t.coef[2].is_nan() && t.se[2].is_nan());
    let keep = ["(Intercept)", "x1", "x3"];
    Check::new(&case)
        .named_subset("coef", &keep, &t.names, &t.coef, Tol::coef(n))
        .named_subset("se_nw", &keep, &t.names, &t.se, Tol::se(n))
        .named_subset("t_nw", &keep, &t.names, &t.stat, Tol::stat(n))
        .named_subset("p_nw", &keep, &t.names, &t.p, Tol::p_value(n))
        .scalar("df_resid", num(&v["df_resid"]), Tol::exact())
        .scalar("r2", num(&v["r_squared"]), Tol::stat(n))
        .finish();
    assert_eq!(v["kernel"], "Bartlett (Newey-West)");
    assert!(v["warnings"].is_array());
}

/// exponential_decay is y = a exp(-b x) + c with three start values, all required (nls_expdecay).
#[test]
fn test_validate_cli_nls_expdecay() {
    let case = Case::load("nls_expdecay");
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "reg",
        "nls",
        d.as_str(),
        "-y",
        "y",
        "-x",
        "x",
        "--model",
        "exponential_decay",
    ];
    let v = s.json(&argv(&base, &["--start", "8,0.3,1"]));
    let ps = v["parameters"].as_array().unwrap();
    let names: Vec<String> = ps
        .iter()
        .map(|p| p["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["a", "b", "c"]);
    let est: Vec<f64> = ps.iter().map(|p| num(&p["estimate"])).collect();
    Check::new(&case)
        .named("coef", &names, &est, Tol::iterative(1e-6))
        .named("se", &names, &f64s(&v["std_errors"]), Tol::iterative(1e-5))
        .scalar("rss", num(&v["rss"]), Tol::iterative(1e-8))
        .finish();
    assert_eq!(v["converged"], true);
    let e = s.err(&argv(&base, &["--start", "1,0.1"]));
    assert!(e.contains("3 starting values"), "{e}");
    assert_eq!(s.run(&base).status.code(), Some(2), "--start is required");
}

// =================================================================================================
// Stats
// =================================================================================================

/// STAT-77 (CLI pp-test): `p2a stats pp-test` through the binary on pp_t3 (n = 200), pinned to R's
/// PP.test internals (case stat_pp_t3, rw_short_*: lshort = TRUE, lag trunc(4 (n/100)^0.25) = 4).
/// `--lshort` is passed because the CLI's default is lshort = false. The Z(tau) statistic and its
/// p-value are not pinned (registry deviations).
#[test]
fn test_validate_cli_pp_test_internals() {
    let case = Case::load("stat_pp_t3");
    let s = temp_session();
    let d = s.load(&case);
    let j = s.json(&["stats", "pp-test", d.as_str(), "--col", "rw", "--lshort"]);
    Check::new(&case)
        .scalar("rw_short_gamma", num(&j["gamma_hat"]), Tol::coef(200))
        .scalar("rw_short_t", num(&j["t_statistic"]), Tol::stat(200))
        .scalar("rw_short_sigma2", num(&j["sigma_squared"]), Tol::stat(200))
        .scalar(
            "rw_short_lambda2",
            num(&j["lambda_squared"]),
            Tol::stat(200),
        )
        .scalar("rw_short_lag", num(&j["truncation_lag"]), Tol::exact())
        .finish();
}

// =================================================================================================
// Derived datasets
// =================================================================================================

/// CLI-4: every invocation is a new process; a munge result and a generated dataset are there in
/// the next one.
#[test]
fn test_validate_cli_munge_persists_across_invocations() {
    let case = Case::load("cli_munge_ols_grunfeld");
    let n = case.scalar("sel_n") as usize; // 200
    let s = temp_session();
    let g = s.load(&case); // "grunfeld"
    s.ok(&[
        "munge",
        "select",
        g.as_str(),
        "--columns",
        "inv",
        "value",
        "--name",
        "sub",
    ]);
    let v = s.json(&["reg", "ols", "sub", "-y", "inv", "-x", "value"]);
    let t = json_table(&v, "t_value");
    Check::new(&case)
        .named("sel_coef", &t.names, &t.coef, Tol::coef(n))
        .named("sel_se_hc1", &t.names, &t.se, Tol::se(n))
        .named("sel_t_hc1", &t.names, &t.stat, Tol::stat(n))
        .named("sel_p_hc1", &t.names, &t.p, Tol::p_value(n))
        .scalar("sel_r2", num(&v["r_squared"]), Tol::stat(n))
        .scalar("sel_adj_r2", num(&v["adj_r_squared"]), Tol::stat(n))
        .scalar("sel_n", num(&v["n_observations"]), Tol::exact())
        .finish();
    s.ok(&["munge", "sort", g.as_str(), "--by", "inv"]); // default name grunfeld_sorted
    let rows = s.json(&["data", "head", "grunfeld_sorted", "-n", "5"]);
    let col = |k: &str| -> Vec<f64> {
        rows.as_array()
            .unwrap()
            .iter()
            .map(|r| r[k].to_string().trim_matches('"').parse().unwrap())
            .collect()
    };
    assert_eq!(col("inv"), case.vector("sorted_inv_head"));
    assert_eq!(col("firm"), case.vector("sorted_firm_head"));
    assert_eq!(col("year"), case.vector("sorted_year_head"));
    s.ok(&[
        "data",
        "generate",
        "-n",
        "50",
        "-d",
        "sim",
        "-s",
        "7",
        "-c",
        r#"[{"name": "x", "distribution": {"type": "normal", "mean": 0, "std": 1}}, {"name": "y", "distribution": {"type": "uniform", "min": 0, "max": 1}}]"#,
    ]);
    assert_eq!(s.json_docs(&["data", "describe", "sim"])[0]["rows"], 50);
    s.ok(&["reg", "ols", "sim", "-y", "y", "-x", "x"]);
    // in place: the munged version replaces the loaded one for later invocations
    s.ok(&[
        "munge",
        "drop",
        g.as_str(),
        "--columns",
        "capital",
        "--name",
        g.as_str(),
    ]);
    assert_eq!(
        s.json_docs(&["data", "describe", g.as_str()])[0]["columns"],
        4
    );
    let sess: Value = serde_json::from_str(&std::fs::read_to_string(s.path()).unwrap()).unwrap();
    for (name, file) in [
        ("sub", "sub.parquet"),
        ("grunfeld_sorted", "grunfeld_sorted.parquet"),
        ("sim", "sim.parquet"),
        ("grunfeld", "grunfeld.parquet"),
    ] {
        let m = &sess["datasets"][name];
        assert_eq!(
            (m["derived"].clone(), m["format"].clone()),
            (json!(true), json!("parquet")),
            "{name}: {m}"
        );
        assert_eq!(m["source_path"], format!("session.json.datasets/{file}"));
        assert!(
            s.dir().join("session.json.datasets").join(file).exists(),
            "{file}"
        );
    }
    // the session directory moves with its sidecar
    let other = tempfile::tempdir().unwrap();
    std::fs::copy(s.path(), other.path().join("session.json")).unwrap();
    std::fs::create_dir(other.path().join("session.json.datasets")).unwrap();
    for e in std::fs::read_dir(s.dir().join("session.json.datasets")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(
            e.path(),
            other
                .path()
                .join("session.json.datasets")
                .join(e.file_name()),
        )
        .unwrap();
    }
    let out = p2a()
        .arg("--session")
        .arg(other.path().join("session.json"))
        .args(["data", "describe", "sub"])
        .current_dir(other.path())
        .output()
        .unwrap();
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).contains("Rows: 200"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// CLI-4: odd names stay inside the sidecar; a dataset loaded over a derived one drops the stale
/// file; a result that cannot be stored is an error, not "Created dataset".
#[test]
fn test_validate_cli_derived_dataset_names_replacement_and_errors() {
    let case = Case::load("cli_munge_ols_grunfeld");
    let s = temp_session();
    let g = s.load(&case);
    s.ok(&[
        "munge",
        "select",
        g.as_str(),
        "--columns",
        "inv",
        "--name",
        "my data/ü",
    ]);
    assert_eq!(
        s.json_docs(&["data", "describe", "my data/ü"])[0]["rows"],
        200
    );
    assert!(
        s.dir()
            .join("session.json.datasets/my%20data%2F%C3%BC.parquet")
            .exists()
    );
    s.ok(&[
        "munge",
        "select",
        g.as_str(),
        "--columns",
        "inv",
        "--name",
        "sub",
    ]);
    s.load_as(&case.dataset_path(), "sub", &[]);
    assert!(!s.dir().join("session.json.datasets/sub.parquet").exists());
    // a regular file where the sidecar directory should be: munge reports the failure
    let t = temp_session();
    let g = t.load(&case);
    std::fs::write(t.dir().join("session.json.datasets"), "not a directory").unwrap();
    let e = t.err(&[
        "munge",
        "select",
        g.as_str(),
        "--columns",
        "inv",
        "--name",
        "sub",
    ]);
    assert!(
        e.contains("Could not store dataset 'sub' in the session"),
        "{e}"
    );
    let sess: Value = serde_json::from_str(&std::fs::read_to_string(t.path()).unwrap()).unwrap();
    assert!(sess["datasets"].get("sub").is_none());
}

/// -F json munge bin / cast print one document (they printed the "Created dataset" object and a
/// second message object).
#[test]
fn test_cli_munge_bin_and_cast_print_one_json_document() {
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/generated/bin_cut.csv"),
        "b",
        &[],
    );
    let docs = s.json_docs(&[
        "munge", "bin", "b", "--column", "x", "--method", "custom", "--edges", "0,1,3,5",
    ]);
    assert_eq!(docs.len(), 1, "{docs:?}");
    assert!(
        docs[0]["details"]
            .as_str()
            .is_some_and(|d| d.contains("Counts: ")),
        "{docs:?}"
    );
    let c = s.dir().join("c.csv");
    std::fs::write(&c, "v\n\"1,234\"\n7\nx\n").unwrap();
    s.load_as(&c, "c", &[]);
    let docs = s.json_docs(&["munge", "cast", "c", "--column", "v", "--dtype", "int"]);
    assert_eq!(docs.len(), 1, "{docs:?}");
    assert!(
        docs[0]["warnings"][0]
            .as_str()
            .unwrap()
            .contains("could not be converted"),
        "{docs:?}"
    );
}

/// CLI-29 / CLI-4: a session with a munge step exports a script that replays from another working
/// directory and fits the same model; the replay leaves nothing behind there.
#[cfg(unix)]
#[test]
fn test_validate_cli_exported_script_with_munge_runs() {
    let case = Case::load("cli_munge_ols_grunfeld");
    let n = case.scalar("post_n") as usize; // 100
    let s = temp_session();
    let g = s.load(&case); // absolute path recorded
    s.ok(&[
        "munge",
        "filter",
        g.as_str(),
        "-c",
        "year",
        "-o",
        "ge",
        "-v",
        "1945",
        "-n",
        "post",
    ]);
    let fit = s.ok(&["reg", "ols", "post", "-y", "inv", "-x", "value", "capital"]);
    let t = text_table(&fit);
    Check::new(&case)
        .named("post_coef", &t.names, &t.coef, Tol::printed(6))
        .named("post_se_hc1", &t.names, &t.se, Tol::printed(6))
        .named("post_t_hc1", &t.names, &t.stat, Tol::printed(3))
        .named("post_p_hc1", &t.names, &t.p, Tol::printed(4))
        .finish();
    assert_eq!(case.scalar("post_rows") as usize, n);
    let script = s.dir().join("replay.sh");
    let out = p2a()
        .args([
            "script",
            "export",
            s.path().to_str().unwrap(),
            "-o",
            script.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::fs::read_to_string(&script)
            .unwrap()
            .contains("rm -rf -- \"$SESSION_FILE.datasets\"")
    );
    let other = tempfile::tempdir().unwrap();
    let bin = std::path::Path::new(env!("CARGO_BIN_EXE_p2a"))
        .parent()
        .unwrap()
        .to_path_buf();
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = std::process::Command::new("bash")
        .arg(&script)
        .current_dir(other.path())
        .env("PATH", path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "replay failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains(&format!("Created dataset 'post' ({n} rows x 5 columns)"))
            && stdout.contains("Script completed successfully"),
        "{stdout}"
    );
    assert_eq!(
        coefficient_rows(&stdout),
        coefficient_rows(&fit),
        "the replay fits the same model"
    );
    let left: Vec<_> = std::fs::read_dir(other.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "left behind: {left:?}");
}

/// fill-na --method mode (R's names(which.max(table(x))): ties go to the first level), a text
/// constant, the strategy rules; value-counts --normalize (dplyr::count, then n / sum(n)).
#[test]
fn test_cli_munge_fill_na_mode_and_value_counts_normalize() {
    let s = temp_session();
    let csv = s.dir().join("fill.csv");
    std::fs::write(
        &csv,
        "k,s,f,g\n3,b,1.5,x\n1,a,,y\n2,b,2.5,x\n2,,4.5,\n,a,,x\n1,c,0.5,y\n",
    )
    .unwrap();
    s.load_as(&csv, "fill", &[]);
    // `data head` JSON prints every cell as a string and quotes text cells inside it (CLI-15):
    // compare the text without either layer of quotes
    let cell = |v: &Value| -> String {
        v.as_str()
            .map_or_else(|| v.to_string(), str::to_string)
            .trim_matches('"')
            .to_string()
    };
    let out = s.ok(&[
        "munge",
        "fill-na",
        "fill",
        "--method",
        "mode",
        "--columns",
        "k",
        "s",
        "--name",
        "filled",
    ]);
    assert!(
        out.contains("Filled 2 missing values in 2 columns: k (1, with 1), s (1, with 'a')"),
        "{out}"
    );
    let col = |r: &Value, k: &str| -> Vec<String> {
        r.as_array().unwrap().iter().map(|x| cell(&x[k])).collect()
    };
    let rows = s.json(&["data", "head", "filled", "-n", "6"]);
    assert_eq!(col(&rows, "k"), ["3", "1", "2", "2", "1", "1"]);
    assert_eq!(col(&rows, "s"), ["b", "a", "b", "a", "a", "c"]);
    s.ok(&[
        "munge",
        "fill-na",
        "fill",
        "--method",
        "constant",
        "--value",
        "missing",
        "--columns",
        "s",
        "--name",
        "c1",
    ]);
    assert_eq!(
        col(&s.json(&["data", "head", "c1", "-n", "6"]), "s")[3],
        "missing"
    );
    assert!(
        s.err(&[
            "munge",
            "fill-na",
            "fill",
            "--method",
            "constant",
            "--columns",
            "f"
        ])
        .contains("needs a value")
    );
    assert!(
        s.err(&[
            "munge",
            "fill-na",
            "fill",
            "--method",
            "mean",
            "--value",
            "3",
            "--columns",
            "f",
        ])
        .contains("only used with strategy 'constant'")
    );
    s.ok(&[
        "munge",
        "value-counts",
        "fill",
        "-c",
        "g",
        "--normalize",
        "--name",
        "counts",
    ]);
    let rows = s.json(&["data", "head", "counts", "-n", "5"]);
    // g is x, y, x, NA, x, y: dplyr::count(g, sort = TRUE) |> mutate(proportion = n / sum(n))
    // gives x 3 1/2, y 2 1/3, NA 1 1/6 (the table prints six decimals)
    let mut got: Vec<(String, f64)> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (cell(&r["g"]), cell(&r["proportion"]).parse().unwrap()))
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    let want = [("null", 1.0 / 6.0), ("x", 0.5), ("y", 1.0 / 3.0)];
    assert_eq!(got.len(), want.len(), "{got:?}");
    for ((g, p), (wg, wp)) in got.iter().zip(want) {
        assert_eq!(g, wg);
        assert!((p - wp).abs() < 1e-6, "{g}: {p} vs {wp}");
    }
}

/// INF-9: data load reads JSON Lines through DataLoader::load, and the next invocation reloads it.
#[test]
fn test_cli_data_load_jsonl_persists() {
    let s = temp_session();
    let body = "{\"a\": 1, \"b\": 2.5, \"c\": \"x\"}\n{\"a\": 2, \"b\": null, \"c\": \"y\"}\n{\"a\": 3, \"b\": -1, \"c\": null}\n";
    for ext in ["jsonl", "ndjson"] {
        let p = s.dir().join(format!("t.{ext}"));
        std::fs::write(&p, body).unwrap();
        s.load_as(&p, ext, &[]);
        let types: Vec<(String, String)> = s.json_docs(&["data", "describe", ext])[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["name"].as_str().unwrap().to_string(),
                    c["dtype"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            types,
            [
                ("a".to_string(), "Int64".to_string()),
                ("b".into(), "Float64".into()),
                ("c".into(), "String".into())
            ]
        );
        let rows = s.json(&["data", "head", ext, "-n", "3"]);
        assert_eq!(
            (rows[1]["b"].clone(), rows[2]["c"].clone()),
            (json!("null"), json!("null"))
        );
    }
    let p = s.dir().join("t.xyz");
    std::fs::write(&p, body).unwrap();
    let e = s.err(&["data", "load", p.to_str().unwrap(), "--name", "bad"]);
    assert!(
        e.contains("jsonl"),
        "the message lists the readable extensions: {e}"
    );
}

/// data save --output x.json writes text cells as plain JSON strings, so data load reads back the
/// text itself (quotes, non-ASCII and missing cells included) and the numbers bit for bit.
#[test]
fn test_cli_data_save_json_round_trip() {
    let s = temp_session();
    let csv = s.dir().join("t.csv");
    // the last row has a comma in a quoted field
    std::fs::write(
        &csv,
        "x,k,s\n0.1,1,plain\n-2.718281828459045,-7,\"say \"\"hi\"\"\"\n1e-300,3,\n33.3,4,\"é,ü\"\n",
    )
    .unwrap();
    s.load_as(&csv, "t", &[]);
    let out = s.dir().join("t.json");
    s.ok(&["data", "save", "t", "--output", out.to_str().unwrap()]);
    let written: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let col = |k: &str| -> Vec<Value> {
        written
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[k].clone())
            .collect()
    };
    assert_eq!(
        col("s"),
        [
            json!("plain"),
            json!("say \"hi\""),
            json!(null),
            json!("é,ü")
        ]
    );
    assert_eq!(col("k"), [json!(1), json!(-7), json!(3), json!(4)]);
    let x: Vec<f64> = col("x").iter().map(|v| v.as_f64().unwrap()).collect();
    let want = [0.1, -std::f64::consts::E, 1e-300, 33.3];
    for (g, w) in x.iter().zip(want) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    s.load_as(&out, "back", &[]);
    let mut types: Vec<(String, String)> = s.json_docs(&["data", "describe", "back"])[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["name"].as_str().unwrap().to_string(),
                c["dtype"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    types.sort();
    assert_eq!(
        types,
        [
            ("k".to_string(), "Int64".to_string()),
            ("s".into(), "String".into()),
            ("x".into(), "Float64".into())
        ]
    );
    // the reloaded copy saves to the same JSON
    let again = s.dir().join("again.json");
    s.ok(&["data", "save", "back", "--output", again.to_str().unwrap()]);
    let reread: Value = serde_json::from_str(&std::fs::read_to_string(&again).unwrap()).unwrap();
    assert_eq!(reread, written);
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
    "validation/datasets/generated/feglm_panel.csv",
    "validation/datasets/generated/pvcm_panel.csv",
    "validation/datasets/generated/pggls_panel.csv",
    "validation/datasets/generated/ab_balanced.csv",
    "validation/datasets/generated/purtest_panel.csv",
    "validation/datasets/generated/panel_absorbed.csv",
    "validation/datasets/generated/quantreg_hetero.csv",
    "validation/datasets/generated/smoothers_t3.csv",
    "validation/datasets/generated/gls_cs.csv",
    "validation/datasets/generated/nls_expdecay.csv",
    "validation/datasets/generated/line_outliers.csv",
    "validation/datasets/generated/supsmu_sine.csv",
    "validation/datasets/generated/step_legacy.csv",
    "validation/datasets/generated/hac_ar1.csv",
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

const PANEL: &[Cmd] = &[
    c(
        &[
            "panel", "fe", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
        ],
        Rp,
    ),
    c(
        &[
            "panel", "re", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
        ],
        Rp,
    ),
    c(
        &[
            "panel", "hausman", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity",
            "firm",
        ],
        Pl,
    ),
    c(
        &[
            "panel", "hdfe", "grunfeld", "-y", "inv", "-x", "value", "capital", "--fe", "firm",
            "year",
        ],
        RpIt,
    ),
    c(
        &[
            "panel",
            "feglm",
            "feglm_panel",
            "-y",
            "y_bin",
            "-x",
            "x",
            "x2",
            "--fe",
            "id",
            "time",
            "--family",
            "logit",
        ],
        It,
    ),
    c(
        &[
            "panel",
            "gmm",
            "ab_balanced",
            "-y",
            "y",
            "-x",
            "x",
            "--entity",
            "id",
            "--time",
            "year",
        ],
        Pl,
    ),
    c(
        &[
            "panel",
            "pvcm",
            "pvcm_panel",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "--entity",
            "id",
        ],
        Pl,
    ),
    c(
        &[
            "panel",
            "pmg",
            "pvcm_panel",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "--entity",
            "id",
        ],
        Pl,
    ),
    c(
        &[
            "panel",
            "gls",
            "pggls_panel",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "--entity",
            "id",
            "--time",
            "t",
            "--model",
            "fe",
        ],
        Pl,
    ),
    c(
        &[
            "panel",
            "unit-root",
            "purtest_panel",
            "-v",
            "y",
            "--unit",
            "id",
            "--time",
            "t",
            "--test",
            "ips",
            "--lag-selection",
            "aic",
            "--pmax",
            "4",
        ],
        Pl,
    ),
];

const PANEL_OK: &[&[&str]] = &[
    &[
        "panel",
        "feglm",
        "feglm_panel",
        "-y",
        "y_bin",
        "-x",
        "x",
        "x2",
        "--fe",
        "id",
        "time",
        "--family",
        "binomial",
    ],
    &[
        "panel",
        "pvcm",
        "pvcm_panel",
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "--entity",
        "id",
        "--model",
        "swamy",
    ],
    &[
        "panel",
        "gls",
        "pggls_panel",
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "--entity",
        "id",
        "--time",
        "t",
        "--model",
        "first-difference",
    ],
    &[
        "panel",
        "unit-root",
        "purtest_panel",
        "-v",
        "y",
        "--unit",
        "id",
        "--time",
        "t",
        "--test",
        "levinlin",
        "--model",
        "drift",
        "--lag-selection",
        "BIC",
        "--pmax",
        "4",
    ],
];

const PANEL_REFUSED: &[(&[&str], &[&str])] = &[
    (
        &[
            "panel",
            "feglm",
            "feglm_panel",
            "-y",
            "y_bin",
            "-x",
            "x",
            "--fe",
            "id",
            "--family",
            "gamma",
        ],
        &["logit", "probit", "poisson", "gaussian"],
    ),
    (
        &[
            "panel",
            "pvcm",
            "pvcm_panel",
            "-y",
            "y",
            "-x",
            "x1",
            "--entity",
            "id",
            "--model",
            "pooling",
        ],
        &["within", "random"],
    ),
    (
        &[
            "panel",
            "gls",
            "pggls_panel",
            "-y",
            "y",
            "-x",
            "x1",
            "--entity",
            "id",
            "--time",
            "t",
            "--model",
            "random",
        ],
        &["fe", "fd", "pooling"],
    ),
    (
        &[
            "panel",
            "unit-root",
            "purtest_panel",
            "-v",
            "y",
            "--unit",
            "id",
            "--time",
            "t",
            "--test",
            "kpss",
        ],
        &["llc", "ips", "madwu", "hadri"],
    ),
    (
        &[
            "panel",
            "unit-root",
            "purtest_panel",
            "-v",
            "y",
            "--unit",
            "id",
            "--time",
            "t",
            "--model",
            "quadratic",
        ],
        &["none", "intercept", "trend"],
    ),
    (
        &[
            "panel",
            "unit-root",
            "purtest_panel",
            "-v",
            "y",
            "--unit",
            "id",
            "--time",
            "t",
            "--lag-selection",
            "hq",
        ],
        &["fixed", "aic", "sic", "hall"],
    ),
    (
        &[
            "panel",
            "gmm",
            "ab_balanced",
            "-y",
            "y",
            "-x",
            "x",
            "--entity",
            "id",
            "--time",
            "year",
            "--step",
            "three",
        ],
        &["onestep", "twostep"],
    ),
    (
        &[
            "panel",
            "gmm",
            "ab_balanced",
            "-y",
            "y",
            "-x",
            "x",
            "--entity",
            "id",
            "--time",
            "year",
            "--vcov",
            "hc9",
        ],
        &["robust", "classical"],
    ),
];

const REGRESSION: &[Cmd] = &[
    c(
        &[
            "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
        ],
        Rp,
    ),
    c(
        &[
            "reg",
            "clustered",
            "grunfeld",
            "-y",
            "inv",
            "-x",
            "value",
            "capital",
            "--cluster",
            "firm",
        ],
        Rp,
    ),
    c(
        &[
            "reg",
            "diagnostics",
            "grunfeld",
            "-y",
            "inv",
            "-x",
            "value",
            "capital",
        ],
        Pl,
    ),
    c(
        &["reg", "quantreg", "quantreg_hetero", "-y", "y", "-x", "x"],
        It,
    ),
    c(
        &[
            "reg",
            "loess",
            "smoothers_t3",
            "-y",
            "ym",
            "-x",
            "xm",
            "--span",
            "0.5",
            "--degree",
            "1",
        ],
        Pl,
    ),
    c(
        &[
            "reg",
            "nls",
            "nls_expdecay",
            "-y",
            "y",
            "-x",
            "x",
            "--model",
            "exponential_decay",
            "--start",
            "8,0.3,1",
        ],
        It,
    ),
    c(
        &[
            "reg",
            "gls",
            "gls_cs",
            "-y",
            "y",
            "-x",
            "x",
            "z",
            "--correlation",
            "cs",
            "--group",
            "g",
        ],
        It,
    ),
    c(
        &[
            "reg",
            "step",
            "step_legacy",
            "-y",
            "y",
            "-x",
            "x1",
            "x2",
            "x3",
        ],
        Pl,
    ),
    c(
        &["reg", "smooth-spline", "smoothers_t3", "-y", "y", "-x", "x"],
        Pl,
    ),
    c(&["reg", "supsmu", "supsmu_sine", "-y", "y", "-x", "x"], Pl),
    c(&["reg", "line", "line_outliers", "-y", "y", "-x", "x"], Pl),
    c(&["reg", "hac", "hac_ar1", "-y", "y", "-x", "x"], Pl),
];

const REGRESSION_OK: &[&[&str]] = &[
    &[
        "reg",
        "quantreg",
        "quantreg_hetero",
        "-y",
        "y",
        "-x",
        "x",
        "--se",
        "Koenker-Bassett",
        "--method",
        "simplex",
    ],
    &[
        "reg",
        "gls",
        "gls_cs",
        "-y",
        "y",
        "-x",
        "x",
        "z",
        "--correlation",
        "exchangeable",
        "--group",
        "g",
        "--method",
        "restricted",
    ],
    &[
        "reg",
        "smooth-spline",
        "smoothers_t3",
        "-y",
        "y",
        "-x",
        "x",
        "--criterion",
        "LOOCV",
    ],
    &[
        "reg",
        "step",
        "step_legacy",
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
        "x3",
        "--direction",
        "Forwards",
    ],
];

const REGRESSION_REFUSED: &[(&[&str], &[&str])] = &[
    (
        &[
            "reg",
            "quantreg",
            "quantreg_hetero",
            "-y",
            "y",
            "-x",
            "x",
            "--se",
            "rank",
        ],
        &["nid", "iid", "ker", "boot"],
    ),
    (
        &[
            "reg",
            "quantreg",
            "quantreg_hetero",
            "-y",
            "y",
            "-x",
            "x",
            "--method",
            "lasso",
        ],
        &["br", "fn", "irls"],
    ),
    (
        &[
            "reg", "gls", "gls_cs", "-y", "y", "-x", "x", "--method", "gmm",
        ],
        &["reml", "ml"],
    ),
    (
        &[
            "reg",
            "gls",
            "gls_cs",
            "-y",
            "y",
            "-x",
            "x",
            "--correlation",
            "arma",
        ],
        &["identity", "ar1", "compound_symmetry", "ar1_auto"],
    ),
    (
        &[
            "reg",
            "smooth-spline",
            "smoothers_t3",
            "-y",
            "y",
            "-x",
            "x",
            "--criterion",
            "aic",
        ],
        &["gcv", "cv"],
    ),
    (
        &[
            "reg",
            "step",
            "step_legacy",
            "-y",
            "y",
            "-x",
            "x1",
            "--direction",
            "sideways",
        ],
        &["forward", "backward", "both"],
    ),
];

const STATS: &[Cmd] = &[
    c(&["stats", "t-test-one", "grunfeld", "-c", "inv"], Pl),
    c(
        &[
            "stats",
            "t-test-two",
            "grunfeld",
            "--col1",
            "inv",
            "--col2",
            "capital",
        ],
        Pl,
    ),
    c(
        &[
            "stats",
            "t-test-paired",
            "grunfeld",
            "--col1",
            "inv",
            "--col2",
            "capital",
        ],
        Pl,
    ),
    c(&["stats", "chisq-gof", "binary_glm", "-o", "d"], Pl),
    c(
        &[
            "stats",
            "chisq-indep",
            "binary_glm",
            "--row-var",
            "y",
            "--col-var",
            "d",
        ],
        Pl,
    ),
    c(
        &[
            "stats",
            "fisher",
            "binary_glm",
            "--row-var",
            "y",
            "--col-var",
            "d",
        ],
        Pl,
    ),
    c(&["stats", "shapiro", "grunfeld", "-c", "inv"], Pl),
    c(
        &[
            "stats", "wilcoxon", "grunfeld", "--col1", "inv", "--col2", "capital",
        ],
        Pl,
    ),
    c(
        &["stats", "bartlett", "grunfeld", "-c", "inv", "-g", "firm"],
        Pl,
    ),
    c(
        &["stats", "kruskal", "grunfeld", "-c", "inv", "-g", "firm"],
        Pl,
    ),
    c(&["stats", "fivenum", "grunfeld", "-c", "inv"], Pl),
    c(&["stats", "iqr", "grunfeld", "-c", "inv"], Pl),
    c(&["stats", "mad", "grunfeld", "-c", "inv"], Pl),
    c(&["stats", "acf", "grunfeld", "-c", "inv"], Pl),
    c(
        &["stats", "anova", "grunfeld", "-r", "inv", "-f", "firm"],
        Pl,
    ),
    c(
        &[
            "stats",
            "anova-two",
            "grunfeld",
            "-r",
            "inv",
            "--factor-a",
            "firm",
            "--factor-b",
            "year",
        ],
        Pl,
    ),
    c(
        &["stats", "tukey", "grunfeld", "-r", "inv", "-f", "firm"],
        Pl,
    ),
    c(
        &[
            "stats", "friedman", "grunfeld", "-v", "inv", "-g", "firm", "-b", "year",
        ],
        Pl,
    ),
    c(&["stats", "pacf", "grunfeld", "-c", "inv"], Pl),
    c(
        &["stats", "ccf", "grunfeld", "--x", "inv", "--y", "value"],
        Pl,
    ),
    c(&["stats", "box-test", "grunfeld", "-c", "inv"], Pl),
    c(
        &[
            "stats", "ks-test", "grunfeld", "--x", "inv", "--y", "capital",
        ],
        Pl,
    ),
    c(&["stats", "pp-test", "grunfeld", "-c", "inv"], Pl),
    c(&["stats", "power-t", "--n", "50", "--delta", "0.5"], Pl),
];

const STATS_REFUSED: &[(&[&str], &[&str])] = &[(
    &[
        "stats",
        "anova-two",
        "grunfeld",
        "-r",
        "inv",
        "--factor-a",
        "firm",
        "--factor-b",
        "year",
        "--ss-type",
        "IV",
    ],
    &["'IV'", "I", "II", "III"],
)];

const MUNGE_REFUSED: &[(&[&str], &[&str])] = &[(
    &["munge", "fill-na", "grunfeld", "--method", "interpolate"],
    &["mean", "median", "mode"],
)];

const ALL: &[&[Cmd]] = &[DISCRETE, SURVIVAL, PANEL, REGRESSION, STATS];
const OPTIONS_OK: &[&[&[&str]]] = &[DISCRETE_OK, SURVIVAL_OK, PANEL_OK, REGRESSION_OK];
const OPTIONS_REFUSED: &[&[(&[&str], &[&str])]] = &[
    DISCRETE_REFUSED,
    SURVIVAL_REFUSED,
    PANEL_REFUSED,
    REGRESSION_REFUSED,
    STATS_REFUSED,
    MUNGE_REFUSED,
];

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

/// An option the chosen settings would ignore is refused, the message naming both flags: --rho
/// with ar1_auto (rho estimated) or identity, --seed without --se boot, --pmax without a lag
/// criterion.
#[test]
fn test_cli_ignored_options_are_refused() {
    let s = temp_session();
    for stem in ["gls_ar1", "quantreg_hetero", "purtest_panel"] {
        let p = repo_path(&format!("validation/datasets/generated/{stem}.csv"));
        s.load_as(&p, stem, &[]);
    }
    let gls = ["reg", "gls", "gls_ar1", "-y", "y", "-x", "x"];
    let qr = ["reg", "quantreg", "quantreg_hetero", "-y", "y", "-x", "x"];
    let ur = [
        "panel",
        "unit-root",
        "purtest_panel",
        "-v",
        "y",
        "--unit",
        "id",
        "--time",
        "t",
    ];
    let refused: &[(Vec<&str>, [&str; 2])] = &[
        (argv(&gls, &["--rho", "0.3"]), ["--rho", "ar1"]),
        (
            argv(&gls, &["--correlation", "ar1_auto", "--rho", "0.3"]),
            ["--rho", "ar1_auto"],
        ),
        (
            argv(&gls, &["--correlation", "identity", "--rho", "0.3"]),
            ["--rho", "identity"],
        ),
        (argv(&qr, &["--seed", "7"]), ["--seed", "--se boot"]),
        (
            argv(&qr, &["--se", "ker", "--seed", "7"]),
            ["--seed", "--se boot"],
        ),
        (argv(&ur, &["--pmax", "4"]), ["--pmax", "--lag-selection"]),
        (
            argv(
                &ur,
                &["--lags", "1", "--lag-selection", "fixed", "--pmax", "4"],
            ),
            ["--pmax", "--lag-selection"],
        ),
    ];
    for (args, names) in refused {
        let e = s.err(args);
        for n in names {
            assert!(e.contains(n), "{}: {n} not named in {e}", args.join(" "));
        }
    }
    // the combinations the options apply to still run
    for args in [
        argv(&gls, &["--correlation", "ar1", "--rho", "0.3"]),
        argv(&qr, &["--se", "boot", "--seed", "7"]),
        argv(&ur, &["--lag-selection", "sic", "--pmax", "4"]),
    ] {
        s.ok(&args);
    }
}

/// Datasets load lazily; `data list` reads each one, so a dataset whose source file is gone is
/// warned about and listed apart from the loaded ones (JSON key "unavailable").
#[test]
fn test_cli_data_list_marks_unreadable_datasets() {
    let s = temp_session();
    let a = s.dir().join("a.csv");
    let b = s.dir().join("b.csv");
    std::fs::write(&a, "x,y\n1,2.5\n2,3.5\n").unwrap();
    std::fs::write(&b, "x\n1\n").unwrap();
    s.load_as(&a, "a", &[]);
    s.load_as(&b, "b", &[]);
    assert_eq!(s.json(&["data", "list"]), json!({ "datasets": ["a", "b"] }));
    std::fs::remove_file(&b).unwrap();
    let out = s.run(&["data", "list"]);
    assert_no_panic(&out, "data list");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    let (loaded, rest) = text.split_once("could not be read").expect(&text);
    assert!(loaded.contains("Loaded datasets:\n  - a\n"), "{text}");
    assert!(!loaded.contains("- b"), "{text}");
    assert!(rest.contains("- b"), "{text}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("Could not reload dataset 'b'"),
        "the failed reload is warned about: {err}"
    );
    assert_eq!(
        s.json(&["data", "list"]),
        json!({ "datasets": ["a"], "unavailable": ["b"] })
    );
}
