//! T3-WP06: boolean flags that can be turned off (CLI-5), `reg clustered --cluster2` (CLI-18), the global
//! `--max-memory` warning (DATA-30) and the no-intercept statistics they expose (REG-19), through the p2a CLI.

use crate::common::*;

#[test]
fn test_tier3_cli_helpers() {
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/grunfeld.csv"),
        "grunfeld",
        &[],
    );
    let args = [
        "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
    ];
    let t = text_table(&s.ok(&args));
    let j = json_table(&s.json(&args), "t_value");
    assert_eq!(t.names, ["(Intercept)", "value", "capital"]);
    assert_eq!(j.names, t.names);
    assert!(
        t.coef
            .iter()
            .zip(&j.coef)
            .all(|(a, b)| Tol::printed(6).allows(*a, *b))
    );
    assert_eq!(argv(&["a"], &["b", "c"]), ["a", "b", "c"]);
}

/// Two-way CR1 (sandwich::vcovCL(~firm + year, HC1)): SEs from T1's ols_grunfeld, t and p from ols_twoway_cluster_grunfeld.
#[test]
fn test_validate_cli_clustered_two_way() {
    let r = Case::load("ols_grunfeld");
    let case = Case::load("ols_twoway_cluster_grunfeld");
    let n = r.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&r);
    let one = [
        "reg",
        "clustered",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--cluster",
        "firm",
    ];
    let two = argv(&one, &["--cluster2", "year"]);
    let out = s.ok(&two);
    let t = text_table(&out);
    Check::new(&r)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_cluster_firm_year_hc1", &t.names, &t.se, Tol::printed(6))
        .finish();
    Check::new(&case)
        .named(
            "t_cluster_firm_year_hc1",
            &t.names,
            &t.stat,
            Tol::printed(3),
        )
        .named("p_cluster_firm_year_hc1", &t.names, &t.p, Tol::printed(4))
        .finish();
    assert!(
        out.contains(
            "\nStandard errors: clustered by firm and year (two-way CR1, G = 10 and 20)\n"
        ),
        "{out}"
    );
    let v = s.json(&two);
    let j = json_table(&v, "t_value");
    Check::new(&r)
        .named("coef", &j.names, &j.coef, Tol::coef(n))
        .named("se_cluster_firm_year_hc1", &j.names, &j.se, Tol::se(n))
        .finish();
    Check::new(&case)
        .named("t_cluster_firm_year_hc1", &j.names, &j.stat, Tol::stat(n))
        .named("p_cluster_firm_year_hc1", &j.names, &j.p, Tol::p_value(n))
        .scalar("n_clusters_firm", num(&v["n_clusters"]), Tol::exact())
        .scalar("n_clusters_year", num(&v["n_clusters_2"]), Tol::exact())
        .finish();
    assert_eq!(v["se_type"], "cluster:firm+year");
    assert_eq!(v["method"], "Clustered OLS (clusters: firm, year)");
    // without --cluster2: T1's one-way result, unchanged
    let o = json_table(&s.json(&one), "t_value");
    Check::new(&r)
        .named("se_cluster_firm_hc1", &o.names, &o.se, Tol::se(n))
        .finish();
}

/// reg ols --intercept false = lm(inv ~ 0 + value + capital) (CLI-5, REG-19); the help example's spelling, `=` and R's FALSE.
#[test]
fn test_validate_cli_ols_no_intercept() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&case);
    let args = ["reg", "ols", &g, "-y", "inv", "-x", "value", "capital"];
    let out = s.ok(&argv(&args, &["--intercept", "false"]));
    let t = text_table(&out);
    assert_eq!(t.names, ["value", "capital"], "no (Intercept) row:\n{out}");
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_hc1", &t.names, &t.se, Tol::printed(6))
        .named("t_hc1", &t.names, &t.stat, Tol::printed(3))
        .named("p_hc1", &t.names, &t.p, Tol::printed(4))
        .scalar(
            "r2",
            labelled(&out, "R-squared (uncentered):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "adj_r2",
            labelled(&out, "Adj. R-squared (uncentered):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .finish();
    let v = s.json(&argv(&args, &["--intercept=FALSE", "--robust", "standard"]));
    let j = json_table(&v, "t_value");
    Check::new(&case)
        .named("coef", &j.names, &j.coef, Tol::coef(n))
        .named("se_classical", &j.names, &j.se, Tol::se(n))
        .named("t_classical", &j.names, &j.stat, Tol::stat(n))
        .named("p_classical", &j.names, &j.p, Tol::p_value(n))
        .scalar("r2", num(&v["r_squared"]), Tol::stat(n))
        .scalar("adj_r2", num(&v["adj_r_squared"]), Tol::stat(n))
        .finish();
    assert_eq!(v["r_squared_type"], "uncentered");
    let with = json_table(&s.json(&args), "t_value");
    assert_eq!(with.names, ["(Intercept)", "value", "capital"]);
    assert_eq!(
        json_table(&s.json(&argv(&args, &["--intercept"])), "t_value"),
        with
    );
}

/// reg clustered --intercept false, one- and two-way: vcovCL on lm(inv ~ 0 + value + capital).
#[test]
fn test_validate_cli_clustered_no_intercept() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&case);
    let one = [
        "reg",
        "clustered",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--cluster",
        "firm",
        "--intercept",
        "false",
    ];
    for (extra, key) in [
        (&[][..], "cluster_firm_hc1"),
        (&["--cluster2", "year"][..], "cluster_firm_year_hc1"),
    ] {
        let v = s.json(&argv(&one, extra));
        let j = json_table(&v, "t_value");
        assert_eq!(j.names, ["value", "capital"]);
        Check::new(&case)
            .named("coef", &j.names, &j.coef, Tol::coef(n))
            .named(&format!("se_{key}"), &j.names, &j.se, Tol::se(n))
            .named(&format!("p_{key}"), &j.names, &j.p, Tol::p_value(n))
            .scalar("r2", num(&v["r_squared"]), Tol::stat(n))
            .scalar("adj_r2", num(&v["adj_r_squared"]), Tol::stat(n))
            .finish();
    }
    assert!(!s.ok(&one).contains("(Intercept)"));
}

/// reg hac --intercept false = NeweyWest(lm(inv ~ 0 + value + capital), lag = 4, prewhite = FALSE).
#[test]
fn test_validate_cli_hac_no_intercept() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&case);
    let args = [
        "reg",
        "hac",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--lags",
        "4",
        "--intercept",
        "false",
    ];
    let v = s.json(&args);
    let j = json_arrays(&v);
    assert_eq!(j.names, ["value", "capital"]);
    assert_eq!(v["bandwidth"], 4);
    Check::new(&case)
        .named("coef", &j.names, &j.coef, Tol::coef(n))
        .named("se_nw_lag4", &j.names, &j.se, Tol::se(n))
        .named("t_nw_lag4", &j.names, &j.stat, Tol::stat(n))
        .named("p_nw_lag4", &j.names, &j.p, Tol::p_value(n))
        .scalar("r2", num(&v["r_squared"]), Tol::stat(n))
        .finish();
    let out = s.ok(&args);
    assert!(
        !out.contains("(Intercept)") && out.contains("R-squared (uncentered):"),
        "{out}"
    );
}

/// reg gls --correlation identity --intercept false is lm(inv ~ 0 + value + capital) with classical SEs.
#[test]
fn test_validate_cli_gls_identity_no_intercept() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&case);
    let v = s.json(&[
        "reg",
        "gls",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--correlation",
        "identity",
        "--intercept",
        "false",
    ]);
    let j = json_arrays(&v);
    assert_eq!(j.names, ["value", "capital"]);
    Check::new(&case)
        .named("coef", &j.names, &j.coef, Tol::coef(n))
        .named("se_classical", &j.names, &j.se, Tol::se(n))
        .named("t_classical", &j.names, &j.stat, Tol::stat(n))
        .named("p_classical", &j.names, &j.p, Tol::p_value(n))
        .scalar("r2", num(&v["r_squared"]), Tol::stat(n))
        .scalar("adj_r2", num(&v["adj_r_squared"]), Tol::stat(n))
        .scalar("sigma", num(&v["sigma"]), Tol::stat(n))
        .finish();
}

/// reg step --direction backward --intercept false = step(lm(inv ~ 0 + value + capital)); --use-bic true = k = log(n).
#[test]
fn test_validate_cli_step_no_intercept() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&case);
    let args = [
        "reg",
        "step",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--direction",
        "backward",
        "--intercept",
        "false",
    ];
    let v = s.json(&args);
    assert_eq!(
        v["final_variables"],
        serde_json::json!(["value", "capital"])
    );
    // only the initial model: no term is dropped (`n_steps` counts the search rounds, here 1)
    assert_eq!(v["steps"].as_array().unwrap().len(), 1);
    assert!(v["steps"][0]["action"].is_null(), "{v}");
    Check::new(&case)
        .scalar(
            "extract_aic",
            num(&v["steps"][0]["criterion"]),
            Tol::stat(n),
        )
        .scalar("r2", num(&v["final_r_squared"]), Tol::stat(n))
        .scalar("aic", num(&v["final_aic"]), Tol::stat(n))
        .finish();
    let b = s.json(&argv(&args, &["--use-bic", "true"]));
    assert_eq!(b["criterion"], "BIC");
    Check::new(&case)
        .scalar(
            "extract_bic",
            num(&b["steps"][0]["criterion"]),
            Tol::stat(n),
        )
        .finish();
    assert_eq!(
        s.json(&argv(&args, &["--use-bic"])),
        b,
        "bare --use-bic (the help example) still means true"
    );
}

/// --intercept false reaches quantreg(). Coefficients are compared with p2a-core in process: R's exact
/// rq(tau = 0.5) is recorded (rq_coef) but not pinned, REG-52 (T3-WP10 implements Barrodale-Roberts).
#[test]
fn test_cli_quantreg_no_intercept_reaches_core() {
    let case = Case::load("ols_no_intercept_grunfeld");
    let s = temp_session();
    let g = s.load(&case);
    let v = s.json(&[
        "reg",
        "quantreg",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--intercept",
        "false",
    ]);
    let rows = v["coefficients"].as_array().unwrap();
    let names: Vec<&str> = rows.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["value", "capital"]);
    let mut cfg = p2a_core::regression::QuantRegConfig::default();
    cfg.intercept = false;
    let want =
        p2a_core::regression::quantreg(&case_dataset(&case), "inv", &["value", "capital"], &cfg)
            .unwrap();
    assert_eq!(rows.len(), want.coefficients.len());
    for (row, c) in rows.iter().zip(&want.coefficients) {
        assert!(
            Tol::new(1e-12, 1e-15).allows(num(&row["estimate"]), c.estimate),
            "{row} vs {}",
            c.estimate
        );
    }
}

/// ts ar: AIC selection (default) = ar(method = "yule-walker"); --aic false = ar(aic = FALSE), order 20.
/// var_pred is not pinned (TS-15).
#[test]
fn test_validate_cli_ts_ar_aic_flag() {
    let case = Case::load("ts_ar_yw_aic");
    let n = case.scalar("n_obs") as usize;
    let s = temp_session();
    let d = s.load(&case);
    let base = ["ts", "ar", &d, "--col", "y_ar"];
    let v = s.json(&base);
    Check::new(&case)
        .scalar("aic_order", num(&v["order"]), Tol::exact())
        .vector("aic_ar", &f64s(&v["ar_coefficients"]), Tol::coef(n))
        .vector("aic_rel", &f64s(&v["aic"]), Tol::stat(n))
        .scalar("x_mean", num(&v["x_mean"]), Tol::coef(n))
        .finish();
    let v0 = s.json(&argv(&base, &["--aic", "false"]));
    Check::new(&case)
        .scalar("noaic_order", num(&v0["order"]), Tol::exact())
        .vector("noaic_ar", &f64s(&v0["ar_coefficients"]), Tol::coef(n))
        .finish();
    assert!(v0["aic"].is_null());
    assert_eq!(
        s.json(&argv(&base, &["--aic=FALSE"]))["ar_coefficients"],
        v0["ar_coefficients"]
    );
    let v2 = s.json(&argv(&base, &["--order", "2", "--aic", "false"]));
    Check::new(&case)
        .vector("order2_ar", &f64s(&v2["ar_coefficients"]), Tol::coef(n))
        .finish();
}

/// ts stl --robust true (the help example) reaches run_stl; not R-pinned (TS-26: f32 STL, default windows).
#[test]
fn test_cli_ts_stl_robust_flag() {
    let case = Case::load("ts_ar_yw_aic");
    let s = temp_session();
    let d = s.load(&case);
    let args = ["ts", "stl", &d, "--col", "y_seas", "--period", "12"];
    let robust = s.json(&argv(&args, &["--robust", "true"]));
    let plain = s.json(&argv(&args, &["--robust", "false"]));
    assert_eq!(
        (robust["robust"].as_bool(), plain["robust"].as_bool()),
        (Some(true), Some(false))
    );
    assert_eq!(s.json(&args), plain, "the default is non-robust");
    assert_eq!(
        s.json(&argv(&args, &["--robust"])),
        robust,
        "bare --robust means true"
    );
    let want = p2a_core::run_stl(&case_dataset(&case), "y_seas", 12, true).unwrap();
    assert_eq!(f64s(&robust["remainder_sample"]), want.remainder[..10]);
    assert_ne!(
        robust["remainder_sample"], plain["remainder_sample"],
        "the outlier at t = 6 is down-weighted"
    );
}

/// ts spectrum --detrend false reaches run_spectrum; not R-pinned (STAT-68: periodogram scale and taper).
#[cfg(feature = "spectral-analysis")]
#[test]
fn test_cli_ts_spectrum_detrend_flag() {
    let case = Case::load("ts_ar_yw_aic");
    let s = temp_session();
    let d = s.load(&case);
    let args = ["ts", "spectrum", &d, "--col", "y_seas"];
    let off = s.json(&argv(&args, &["--detrend", "false"]));
    let on = s.json(&args);
    assert_eq!(
        (off["detrend"].as_bool(), on["detrend"].as_bool()),
        (Some(false), Some(true))
    );
    let mut cfg = p2a_core::SpectrumConfig::default();
    cfg.detrend = false;
    let want = p2a_core::run_spectrum(&case_dataset(&case), "y_seas", cfg).unwrap();
    assert_eq!(f64s(&off["spec_sample"]), want.spec[..10]);
    assert_ne!(
        off["spec_sample"], on["spec_sample"],
        "the trend dominates the low frequencies"
    );
}

/// script export --comments false drops the per-command comment lines only.
#[test]
fn test_cli_script_export_comments_flag() {
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/grunfeld.csv"),
        "grunfeld",
        &[],
    );
    s.ok(&[
        "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
    ]);
    let session = s.path().to_str().unwrap().to_owned();
    let script = s.dir().join("x.sh");
    let export = |extra: &[&str]| {
        s.ok(&argv(
            &["script", "export", &session, "-o", script.to_str().unwrap()],
            extra,
        ));
        std::fs::read_to_string(&script).unwrap()
    };
    let comments = |sc: &str| {
        sc.lines()
            .filter(|l| l.starts_with("# data load") || l.starts_with("# reg ols"))
            .count()
    };
    let commands = |sc: &str| {
        sc.lines()
            .filter(|l| l.starts_with("p2a --session"))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let (default, on, off) = (
        export(&[]),
        export(&["--comments", "true"]),
        export(&["--comments", "false"]),
    );
    assert_eq!(
        (comments(&default), comments(&on), comments(&off)),
        (2, 2, 0)
    );
    assert_eq!(commands(&off), commands(&default));
    assert_eq!(commands(&off).len(), 2);
    assert!(off.starts_with("#!/bin/bash\n# p2a analytics script\n"));
}
