//! T3-WP56 MCP-path tests: statistics, hypothesis and time-series tools through
//! call_tool_with_session on R cases with missing cells and Int64 columns. Owned by T3-WP56.
use crate::common::*;
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

/// The JSON block of a tool result: the whole text when it starts with '{', else what follows
/// "JSON:\n", else the last top-level "{" line (Display text followed by pretty JSON).
fn json_of(text: &str) -> Value {
    let start = if text.trim_start().starts_with('{') {
        text.find('{').unwrap()
    } else if let Some(i) = text.find("JSON:\n") {
        i + "JSON:\n".len()
    } else {
        text.rfind("\n{\n")
            .map(|i| i + 1)
            .unwrap_or_else(|| panic!("no JSON block in\n{text}"))
    };
    serde_json::from_str(&text[start..]).unwrap_or_else(|e| panic!("bad JSON ({e}) in\n{text}"))
}

fn f(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or(f64::NAN)
}

fn fs(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(f)
        .collect()
}

fn number_after(text: &str, needle: &str) -> f64 {
    let i = text
        .find(needle)
        .unwrap_or_else(|| panic!("no {needle:?} in\n{text}"))
        + needle.len();
    let tok: String = text[i..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .collect();
    tok.parse()
        .unwrap_or_else(|_| panic!("no number after {needle:?} in\n{text}"))
}

async fn ok(s: &AnalyticsServer, se: &Session, tool: &str, args: Value) -> String {
    let r = call(s, se, tool, args).await;
    let t = text(&r);
    assert!(r.success, "{tool}: {t}");
    t
}

async fn refused(s: &AnalyticsServer, se: &Session, tool: &str, args: Value) -> String {
    let r = call(s, se, tool, args).await;
    let t = text(&r);
    assert!(!r.success, "{tool} must be a tool error: {t}");
    t
}

/// TOOL-27: R/CLI spellings of `alternative` are accepted, unknown ones refused. Before:
/// "right"/"left" ran the two-sided test (t-test, Mood, Poisson), "Greater" was refused by
/// Fisher and ran two-sided in CMH.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_hypothesis_alternative_aliases() {
    // t.test(x, y, alternative = "greater", conf.level = 0.99) on ttest_levels_t3 (Welch)
    let tl = Case::load("stat_ttest_levels_t3");
    let (se, s) = session_with("stat_ttest_levels_t3");
    let args = |alt: &str| json!({"dataset": "ttest_levels_t3", "x": "x", "y": "y", "alternative": alt, "conf_level": 0.99});
    let right = ok(&s, &se, "hypothesis_t_test", args("right")).await;
    assert_eq!(
        right,
        ok(&s, &se, "hypothesis_t_test", args("greater")).await
    );
    let mut c = Check::new(&tl);
    // 6 decimals printed
    c.scalar(
        "welch99_greater_p",
        number_after(&right, "p-value = "),
        Tol::printed(6),
    );
    c.finish();
    assert!(
        refused(&s, &se, "hypothesis_t_test", args("bigger"))
            .await
            .contains("expected one of")
    );
    // mood.test(s1, s2, alternative = "less") on mcp_groups_t3: z -0.029266061720005017,
    // p 0.488326197062813
    let g = Case::load("mcp_groups_t3");
    let (ge, gs) = session_with("mcp_groups_t3");
    let m = json_of(
        &ok(
            &gs,
            &ge,
            "hypothesis_mood_test",
            json!({"dataset": "mcp_groups_t3", "x": "s1", "y": "s2", "alternative": "left"}),
        )
        .await,
    );
    Check::new(&g)
        .scalar("mood_less_p", f(&m["p_value"]), Tol::p_value(64))
        .scalar("mood_less_z", f(&m["z_score"]), Tol::stat(64))
        .finish();
    // poisson.test(10, 5, alternative = "less"): p 0.986304731401617, CI [0, 3.392443847144379]
    let p = Case::load("stat_poisson_t3");
    let (pe, ps) = session_with_datasets(vec![]);
    let j = json_of(
        &ok(
            &ps,
            &pe,
            "hypothesis_poisson",
            json!({"x": [10], "t": [5.0], "alternative": "left"}),
        )
        .await,
    );
    // x10_less_ci is not pinned: p2a-core's one-sample poisson_test returns the two-sided
    // interval for every alternative (R: [0, 3.392443847144379] for "less"). New observation,
    // reported to the core owners; the handler passes the alternative through.
    Check::new(&p)
        .scalar("x10_less_p", f(&j["p_value"]), Tol::p_value(10))
        .finish();
    assert!(
        refused(
            &ps,
            &pe,
            "hypothesis_poisson",
            json!({"x": [10], "t": [5.0], "alternative": "up"})
        )
        .await
        .contains("two.sided")
    );
    // Fisher and CMH: an alias gives the canonical output; an unknown value is refused
    for (case, tool, args) in [
        (
            "stat_fisher_rows_2000",
            "hypothesis_fisher_exact",
            json!({"dataset": "fisher_rows_2000", "row_var": "treat", "col_var": "outcome"}),
        ),
        (
            "stat_mantelhaen_na",
            "hypothesis_mantelhaen",
            json!({"dataset": "mantelhaen_na", "row_var": "treat", "col_var": "outcome", "stratum_var": "stratum"}),
        ),
    ] {
        let (e, sv) = session_with(case);
        let with = |alt: &str| {
            let mut a = args.clone();
            a["alternative"] = json!(alt);
            a
        };
        assert_eq!(
            ok(&sv, &e, tool, with("Greater")).await,
            ok(&sv, &e, tool, with("greater")).await,
            "{tool}"
        );
        assert_ne!(
            ok(&sv, &e, tool, with("right")).await,
            ok(&sv, &e, tool, with("two.sided")).await,
            "{tool}"
        );
        assert!(
            refused(&sv, &e, tool, with("bigger"))
                .await
                .contains("expected one of: two.sided, greater, less"),
            "{tool}"
        );
    }
    // pairwise p_adjust_method and cor.test method through the core enums
    let pw = json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g", "p_adjust_method": "Bonf", "exact": false});
    assert!(
        ok(&gs, &ge, "hypothesis_pairwise_wilcox", pw)
            .await
            .contains("\"p_adjust_method\": \"bonferroni\"")
    );
    assert!(
        refused(
            &gs,
            &ge,
            "hypothesis_pairwise_t_test",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g", "p_adjust_method": "sidak"})
        )
        .await
        .contains("holm")
    );
    assert!(
        refused(
            &gs,
            &ge,
            "hypothesis_cor_test",
            json!({"dataset": "mcp_groups_t3", "x": "s1", "y": "s2", "method": "distance"})
        )
        .await
        .contains("pearson")
    );
}

/// TOOL-27: the KS reference distribution takes R's names and parameters and is named in the
/// output. Before: distribution "punif" tested N(0, 1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_ks_distribution_aliases() {
    // ks.test(xu, "punif", -1.5, 3, exact = FALSE): D 0.24555555555555558,
    // p 0.09809220328878654 (n = 25); alternative "greater": D+ 0.15522222222222223,
    // p 0.29978284585485815
    let case = Case::load("stat_ks_t3");
    let (se, s) = session_with("stat_ks_t3");
    let base = json!({"dataset": "ks_t3", "x": "xu", "distribution": "punif", "a": -1.5, "b": 3.0});
    let j = json_of(&ok(&s, &se, "hypothesis_ks_test", base.clone()).await);
    assert_eq!(j["distribution"], "punif(min = -1.5, max = 3)");
    assert_eq!(j["n"], 25);
    let mut gt = base.clone();
    gt["alternative"] = json!("right");
    let g = json_of(&ok(&s, &se, "hypothesis_ks_test", gt).await);
    Check::new(&case)
        .scalar("unif_d", f(&j["statistic_D"]), Tol::stat(25))
        .scalar("unif_p_asym", f(&j["p_value"]), Tol::p_value(25))
        .scalar("unif_dplus", f(&g["statistic_D"]), Tol::stat(25))
        .scalar("unif_p_greater_asym", f(&g["p_value"]), Tol::p_value(25))
        .finish();
    // "Uniform" and "unif" are the same distribution; pexp names its rate
    for d in ["Uniform", "unif"] {
        let mut a = base.clone();
        a["distribution"] = json!(d);
        assert_eq!(
            json_of(&ok(&s, &se, "hypothesis_ks_test", a).await)["statistic_D"],
            j["statistic_D"]
        );
    }
    let e = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_ks_test",
            json!({"dataset": "ks_t3", "x": "xe", "distribution": "pexp", "rate": 0.8}),
        )
        .await,
    );
    assert_eq!(e["distribution"], "pexp(rate = 0.8)");
    Check::new(&case)
        .scalar("exp_d", f(&e["statistic_D"]), Tol::stat(40))
        .finish();
    // refusals: an unknown distribution, a parameter of another distribution, parameters on a
    // two-sample test, an invalid parameter
    for (args, needle) in [
        (
            json!({"dataset": "ks_t3", "x": "xu", "distribution": "gamma"}),
            "expected one of: normal, uniform, exponential",
        ),
        (
            json!({"dataset": "ks_t3", "x": "xu", "distribution": "uniform", "rate": 2.0}),
            "rate",
        ),
        (
            json!({"dataset": "ks_t3", "x": "xa", "y": "ya", "distribution": "normal"}),
            "one-sample",
        ),
        (
            json!({"dataset": "ks_t3", "x": "xn", "distribution": "normal", "sd": 0.0}),
            "sd",
        ),
    ] {
        assert!(
            refused(&s, &se, "hypothesis_ks_test", args)
                .await
                .contains(needle),
            "{needle}"
        );
    }
}

/// TOOL-27: power options with the CLI's kebab-case and R's spellings. Before:
/// 'one-sample'/'one-sided' ran the two-sample, two-sided calculation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_power_option_spellings() {
    // power.t.test(n = 20, delta = 0.8, sd = 1.2, type = .., alternative = ..)$power
    // (stat_power_solve); the port reproduces pt/pnt to <= 1.8e-14 relative, so 1e-10 is the
    // documented margin.
    let case = Case::load("stat_power_solve");
    let (se, s) = session_with_datasets(vec![]);
    let tight = Tol::new(1e-10, 0.0);
    let t = |ty: &str, alt: &str| json!({"n": 20.0, "delta": 0.8, "sd": 1.2, "test_type": ty, "alternative": alt});
    let mut c = Check::new(&case);
    for (ty, alt, key) in [
        ("one-sample", "one-sided", "t_onesample_onesided_power"),
        ("one.sample", "onesided", "t_onesample_onesided_power"),
        ("Paired", "two-sided", "t_paired_twosided_power"),
        ("two_sample", "two.sided", "t_twosample_twosided_power"),
    ] {
        let j = json_of(&ok(&s, &se, "power_t_test", t(ty, alt)).await);
        c.scalar(key, f(&j["power"]), tight);
    }
    // power.prop.test(n = 50, p1 = 0.5, p2 = 0.75, alternative = "one.sided")$power
    let j = json_of(
        &ok(
            &s,
            &se,
            "power_prop_test",
            json!({"n": 50.0, "p1": 0.5, "p2": 0.75, "alternative": "one-sided"}),
        )
        .await,
    );
    c.scalar("prop_onesided_power", f(&j["power"]), tight);
    c.finish();
    assert_eq!(j["alternative"], "one.sided");
    assert!(
        refused(&s, &se, "power_t_test", t("student", "two.sided"))
            .await
            .contains("expected one of: two.sample, one.sample, paired")
    );
    assert!(
        refused(
            &s,
            &se,
            "power_prop_test",
            json!({"n": 50.0, "p1": 0.5, "p2": 0.75, "alternative": "less"})
        )
        .await
        .contains("one.sided")
    );
}

/// TOOL-27: options of the descriptive and multivariate tools. Before: MAD centre 'trimmed'
/// silently used the median, cov.wt 'mle' and 'biased' silently used the unbiased divisor,
/// MANOVA `test` was read nowhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_stats_option_aliases() {
    // mad(x1, center = mean(x1)) (mcp_misc_t3, n = 25); cov.wt(.., method = "ML")
    // (stat_robust_weighted_t3)
    let misc = Case::load("mcp_misc_t3");
    let rw = Case::load("stat_robust_weighted_t3");
    let (se, s) = session_with("mcp_misc_t3"); // dataset robust_weighted_t3
    let m = json_of(
        &ok(
            &s,
            &se,
            "stats_mad",
            json!({"dataset": "robust_weighted_t3", "column": "x1", "center": "Mean"}),
        )
        .await,
    );
    let cw = json_of(
        &ok(
            &s,
            &se,
            "stats_cov_wt",
            json!({"dataset": "robust_weighted_t3", "columns": ["x1", "x2"], "weights": "w", "method": "mle"}),
        )
        .await,
    );
    let cov: Vec<f64> = cw["covariance_matrix"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(fs)
        .collect();
    Check::new(&misc)
        .scalar("mad_x1_center_mean", f(&m["mad"]), Tol::coef(25))
        .scalar("mean_x1", f(&m["center_used"]), Tol::coef(25))
        .finish();
    Check::new(&rw)
        .vector("covwt_ml", &cov, Tol::stat(25))
        .finish();
    assert_eq!(
        (m["center"].as_str(), cw["method"].as_str()),
        (Some("mean"), Some("ML"))
    );
    for (tool, args, needle) in [
        (
            "stats_mad",
            json!({"dataset": "robust_weighted_t3", "column": "x1", "center": "trimmed"}),
            "unknown center 'trimmed'",
        ),
        (
            "stats_cov_wt",
            json!({"dataset": "robust_weighted_t3", "columns": ["x1", "x2"], "weights": "w", "method": "biased"}),
            "unknown method 'biased'",
        ),
        (
            "stats_density",
            json!({"dataset": "robust_weighted_t3", "column": "x1", "kernel": "optcosine"}),
            "gaussian",
        ),
        (
            "stats_spline",
            json!({"dataset": "robust_weighted_t3", "x": "x1", "y": "x2", "method": "akima"}),
            "natural",
        ),
        (
            "stats_approx",
            json!({"dataset": "robust_weighted_t3", "x": "x1", "y": "x2", "xout": [8.0], "rule": "extrapolate"}),
            "nearest",
        ),
        (
            "multivariate_factanal",
            json!({"dataset": "robust_weighted_t3", "columns": ["x1", "x2", "w"], "n_factors": 1, "rotation": "oblimin"}),
            "varimax",
        ),
    ] {
        assert!(
            refused(&s, &se, tool, args).await.contains(needle),
            "{tool}: {needle}"
        );
    }
    // aliases give the canonical result
    let d =
        |k: &str| json!({"dataset": "robust_weighted_t3", "column": "x1", "kernel": k, "n": 64});
    assert_eq!(
        ok(&s, &se, "stats_density", d("normal")).await,
        ok(&s, &se, "stats_density", d("gaussian")).await
    );
    // MANOVA: the selected test is reported and summarised (all four are computed, as
    // summary.manova(test = ..))
    let (ge, gs) = session_with("mcp_groups_t3");
    let w = json_of(
        &ok(
            &gs,
            &ge,
            "anova_manova",
            json!({"dataset": "mcp_groups_t3", "response_vars": ["vi", "v"], "factor": "g", "test": "Wilks"}),
        )
        .await,
    );
    assert_eq!(w["selected_test"], "wilks");
    assert!(
        w["interpretation"]
            .as_str()
            .unwrap()
            .starts_with("Wilks' Lambda"),
        "{w}"
    );
    assert!(
        refused(
            &gs,
            &ge,
            "anova_manova",
            json!({"dataset": "mcp_groups_t3", "response_vars": ["vi", "v"], "factor": "g", "test": "lambda"})
        )
        .await
        .contains("pillai")
    );
    let mt = json_of(
        &ok(
            &gs,
            &ge,
            "stats_model_tables",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factors": ["g"], "table_type": "effect"}),
        )
        .await,
    );
    let gc = Case::load("mcp_groups_t3");
    Check::new(&gc)
        .named(
            "effects",
            &["ctl", "trtA", "trtB"],
            &fs(&mt["one_way_table"]["values"]),
            Tol::coef(30),
        )
        .finish();
    assert!(
        ok(
            &gs,
            &ge,
            "stats_se_contrast",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g", "contrast_type": "contr.helmert"})
        )
        .await
        .contains("standard_errors")
    );
}

/// TOOL-27: ACF types. Before: acf_type 'autocovariance' returned correlations (lag 0 = 1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_acf_type_aliases() {
    // acf(x, type = "covariance") (19 lags, lag.max 18 = floor(10 log10 64)); acf(x); pacf(x)
    // (stat_ts_acf_spline_t3)
    let case = Case::load("stat_ts_acf_spline_t3");
    let (se, s) = session_with("stat_ts_acf_spline_t3");
    let run = |t: &str| json!({"dataset": "ts_acf_spline_t3", "column": "x", "acf_type": t});
    let cov = json_of(&ok(&s, &se, "timeseries_acf", run("autocovariance")).await);
    let acvf = json_of(&ok(&s, &se, "timeseries_acf", run("ACVF")).await);
    let cor = json_of(
        &ok(
            &s,
            &se,
            "timeseries_acf",
            json!({"dataset": "ts_acf_spline_t3", "column": "x"}),
        )
        .await,
    );
    let pacf = json_of(&ok(&s, &se, "timeseries_acf", run("pacf")).await);
    Check::new(&case)
        .vector("acvf", &fs(&cov["values"]), Tol::stat(64))
        .vector("acf", &fs(&cor["values"]), Tol::stat(64))
        .vector("pacf", &fs(&pacf["values"]), Tol::stat(64))
        .scalar(
            "acf_ci_bound",
            f(&cor["confidence_bound_95"]),
            Tol::coef(64),
        )
        .finish();
    assert_eq!(cov["values"], acvf["values"]);
    assert!(
        refused(&s, &se, "timeseries_acf", run("spectral"))
            .await
            .contains("correlation")
    );
}

/// TOOL-27: the other time-series options. Before: 'box_pierce' ran Ljung-Box, 'mle' fitted
/// Yule-Walker, 'Multiplicative' (decompose) ran additive, unknown spectrum methods ran the
/// periodogram, an unknown StructTS model ran the local level, 'var' (changepoint) detected
/// mean changes, and 'Recursive'/'Keep' were refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_timeseries_option_aliases() {
    let sc = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    // Box.test(w, lag = 10, type = "Box-Pierce", fitdf = 2)
    let bp = json_of(
        &ok(
            &s,
            &se,
            "timeseries_box_test",
            json!({"dataset": "mcp_series_t3", "column": "w", "lag": 10, "test_type": "box_pierce", "fitdf": 2}),
        )
        .await,
    );
    // ar(z, method = "yule-walker")
    let yw = json_of(
        &ok(
            &s,
            &se,
            "timeseries_ar",
            json!({"dataset": "mcp_series_t3", "column": "z", "method": "yw"}),
        )
        .await,
    );
    // decompose(ts(y, frequency = 12), "multiplicative")$figure (printed with 4 decimals)
    let dm = json_of(
        &ok(
            &s,
            &se,
            "timeseries_decompose",
            json!({"dataset": "mcp_series_t3", "column": "y", "period": 12, "decompose_type": "Multiplicative"}),
        )
        .await,
    );
    let fig: Vec<f64> = dm["seasonal_figure"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| f(&e["value"]))
        .collect();
    // runmed(z, 5, endrule = "keep")[1:20]; filter(z, c(0.5, -0.2), method = "recursive")[1:20]
    let rk = json_of(
        &ok(
            &s,
            &se,
            "timeseries_runmed",
            json!({"dataset": "mcp_series_t3", "column": "z", "k": 5, "endrule": "Keep"}),
        )
        .await,
    );
    let fr = json_of(
        &ok(
            &s,
            &se,
            "timeseries_filter",
            json!({"dataset": "mcp_series_t3", "column": "z", "filter": [0.5, -0.2], "method": "Recursive"}),
        )
        .await,
    );
    Check::new(&sc)
        .scalar("bp_stat", f(&bp["statistic"]), Tol::stat(96))
        .scalar("bp_df", f(&bp["df"]), Tol::exact())
        .scalar("bp_p", f(&bp["p_value"]), Tol::p_value(96))
        .scalar("yw_order", f(&yw["order"]), Tol::exact())
        .vector("yw_ar", &fs(&yw["ar_coefficients"]), Tol::coef(96))
        .scalar("yw_var", f(&yw["prediction_variance"]), Tol::stat(96))
        .scalar("yw_mean", f(&yw["x_mean"]), Tol::coef(96))
        .vector("mult_figure", &fig, Tol::printed(4))
        .vector("runmed_keep_head", &fs(&rk["values_sample"]), Tol::coef(96))
        .vector(
            "filter_recursive_head",
            &fs(&fr["values_sample"]),
            Tol::coef(96),
        )
        .finish();
    assert_eq!(
        (
            bp["test_type"].as_str(),
            dm["decompose_type"].as_str(),
            fr["method"].as_str(),
            rk["endrule"].as_str()
        ),
        (
            Some("Box-Pierce"),
            Some("multiplicative"),
            Some("recursive"),
            Some("keep")
        )
    );
    // changepoint 'var' on lev: cpt.var(lev, PELT, MBIC) = 84, 151 (ts_filter_ar_cpt)
    let cp = Case::load("ts_filter_ar_cpt");
    let (ce, cs) = session_with("ts_filter_ar_cpt");
    let t = ok(
        &cs,
        &ce,
        "ts_changepoint",
        json!({"dataset": "ts_utils", "column": "lev", "change_type": "var"}),
    )
    .await;
    assert!(t.contains("Changepoint Positions: [84, 151]"), "{t}");
    Check::new(&cp)
        .scalar(
            "cpt_var_cost",
            number_after(&t, "Total Cost:"),
            Tol::printed(4),
        )
        .finish();
    for (tool, args, needle) in [
        (
            "timeseries_box_test",
            json!({"dataset": "mcp_series_t3", "column": "w", "test_type": "portmanteau"}),
            "expected one of: Ljung-Box, Box-Pierce",
        ),
        (
            "timeseries_ar",
            json!({"dataset": "mcp_series_t3", "column": "z", "method": "mle"}),
            "expected one of: yule-walker, burg, ols",
        ),
        (
            "timeseries_decompose",
            json!({"dataset": "mcp_series_t3", "column": "y", "period": 12, "decompose_type": "log"}),
            "additive",
        ),
        (
            "timeseries_spectrum",
            json!({"dataset": "mcp_series_t3", "column": "w", "method": "welch"}),
            "expected one of: pgram, ar",
        ),
        (
            "timeseries_structts",
            json!({"dataset": "mcp_series_t3", "column": "w", "model_type": "arima"}),
            "level",
        ),
        (
            "ts_holt_winters",
            json!({"dataset": "mcp_series_t3", "column": "y", "period": 12, "seasonal": "damped"}),
            "additive",
        ),
        (
            "ts_changepoint",
            json!({"dataset": "mcp_series_t3", "column": "w", "method": "segneigh"}),
            "pelt",
        ),
        (
            "ts_changepoint",
            json!({"dataset": "mcp_series_t3", "column": "w", "method": "binary", "change_type": "variance"}),
            "binary",
        ),
        (
            "timeseries_runmed",
            json!({"dataset": "mcp_series_t3", "column": "z", "k": 5, "endrule": "tukey"}),
            "keep",
        ),
        (
            "timeseries_filter",
            json!({"dataset": "mcp_series_t3", "column": "z", "filter": [0.5], "method": "fir"}),
            "convolution",
        ),
    ] {
        assert!(
            refused(&s, &se, tool, args).await.contains(needle),
            "{tool}: {needle}"
        );
    }
}

/// TOOL-12: R's ccf(x, y): the value at lag k is cor(x[t+k], y[t]); a peak at k < 0 means x
/// leads y. Before: "y leads x by 1 periods".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ccf_direction() {
    let case = Case::load("stat_ts_acf_spline_t3");
    let (se, s) = session_with("stat_ts_acf_spline_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_ccf",
            json!({"dataset": "ts_acf_spline_t3", "x": "x", "y": "y", "lag_max": 5}),
        )
        .await,
    );
    Check::new(&case)
        .vector("ccf", &fs(&j["values"]), Tol::stat(64))
        .vector("ccf_lags", &fs(&j["lags"]), Tol::exact())
        .scalar("acf_ci_bound", f(&j["confidence_bound_95"]), Tol::coef(64))
        .finish();
    assert_eq!(j["max_correlation"]["lag"], -1);
    assert_eq!(
        j["max_correlation"]["interpretation"],
        "x leads y by 1 period"
    );
    // swapping the series swaps the reading: ccf(y, x) peaks at +1
    let r = json_of(
        &ok(
            &s,
            &se,
            "timeseries_ccf",
            json!({"dataset": "ts_acf_spline_t3", "x": "y", "y": "x", "lag_max": 5}),
        )
        .await,
    );
    assert_eq!(
        (
            r["max_correlation"]["lag"].as_i64(),
            r["max_correlation"]["interpretation"].as_str()
        ),
        (Some(1), Some("x leads y by 1 period"))
    );
    let desc = AnalyticsServer::new()
        .router_tools()
        .into_iter()
        .find(|t| t.name == "timeseries_ccf")
        .unwrap()
        .description
        .unwrap()
        .to_string();
    assert!(
        desc.contains("a peak at a negative lag k means x leads y")
            && !desc.contains("Positive lag k means x leads y"),
        "{desc}"
    );
}

/// TOOL-11: NaN or ±Inf input is a tool error naming the column (R: na.fail); before, the
/// handler panicked at `partial_cmp(..).unwrap()`. (This is the one inline fixture with a NaN
/// cell.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_ccf_nan_no_panic() {
    use p2a_core::polars::prelude::*;
    let x = [1.1, 2.3, f64::NAN, 3.2, 2.0, 1.4, 0.9, 1.7, 2.6, 1.2];
    let y = [0.4, 1.0, 2.1, 1.9, 3.1, 1.8, 1.3, 0.8, 1.9, 2.4];
    let inf = [1.1, 2.3, f64::INFINITY, 3.2, 2.0, 1.4, 0.9, 1.7, 2.6, 1.2];
    let ds = p2a_core::data::Dataset::new(df!("x" => x, "y" => y, "xi" => inf).unwrap());
    let (se, s) = session_with_datasets(vec![("nan", ds)]);
    let r = try_call(
        &s,
        &se,
        "timeseries_ccf",
        json!({"dataset": "nan", "x": "x", "y": "y"}),
    )
    .await
    .expect("dispatch");
    assert!(
        !r.success && text(&r).contains("missing value"),
        "{}",
        text(&r)
    );
    let r = try_call(
        &s,
        &se,
        "timeseries_ccf",
        json!({"dataset": "nan", "x": "xi", "y": "y"}),
    )
    .await
    .expect("dispatch");
    assert!(!r.success && text(&r).contains("infinite"), "{}", text(&r));
}

/// TOOL-25: the default is R's n = 3 x (complete pairs), capped at 200; `n` is R's n; the
/// description promises points, not coefficients. Before: `n` was an unknown field and the
/// description said "returns the spline coefficients".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_stats_spline_default_n() {
    let case = Case::load("stat_ts_acf_spline_t3");
    let (se, s) = session_with("stat_ts_acf_spline_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "stats_spline",
            json!({"dataset": "ts_acf_spline_t3", "x": "kx", "y": "ky", "method": "natural"}),
        )
        .await,
    );
    let one = json_of(
        &ok(
            &s,
            &se,
            "stats_spline",
            json!({"dataset": "ts_acf_spline_t3", "x": "kx", "y": "ky", "method": "natural", "n": 1}),
        )
        .await,
    );
    Check::new(&case)
        .vector("spline_x", &fs(&j["x"]), Tol::coef(10))
        .vector("spline_y", &fs(&j["y"]), Tol::coef(10))
        .scalar("spline_n", f(&j["n_points"]), Tol::exact())
        .scalar("spline1_x", fs(&one["x"])[0], Tol::coef(10))
        .scalar("spline1_y", fs(&one["y"])[0], Tol::coef(10))
        .finish();
    assert_eq!(
        (j["n_pairs"].as_u64(), j["missing"]["n_dropped"].as_u64()),
        (Some(10), Some(54))
    );
    let tool = AnalyticsServer::new()
        .router_tools()
        .into_iter()
        .find(|t| t.name == "stats_spline")
        .unwrap();
    let xout = tool.input_schema["properties"]["xout"]["description"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        !xout.contains("coefficients") && xout.contains("3 times"),
        "{xout}"
    );
    assert!(tool.input_schema["properties"]["n"].is_object());
}

/// The MCP default number of spline points is capped at 200 (R's 3 x pairs would be 300 here);
/// an explicit n is honoured.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_stats_spline_default_cap() {
    use p2a_core::polars::prelude::*;
    let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.37).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v.sin() + 0.01 * ((i * 7 % 11) as f64))
        .collect();
    let ds = p2a_core::data::Dataset::new(df!("x" => x, "y" => y).unwrap());
    let (se, s) = session_with_datasets(vec![("pairs", ds)]);
    let j = json_of(
        &ok(
            &s,
            &se,
            "stats_spline",
            json!({"dataset": "pairs", "x": "x", "y": "y"}),
        )
        .await,
    );
    assert_eq!(
        (j["n_points"].as_u64(), j["n_pairs"].as_u64()),
        (Some(200), Some(100))
    );
    let k = json_of(
        &ok(
            &s,
            &se,
            "stats_spline",
            json!({"dataset": "pairs", "x": "x", "y": "y", "n": 300}),
        )
        .await,
    );
    assert_eq!(k["n_points"].as_u64(), Some(300));
}

/// The first `got.len()` entries of a case vector (`Check::vector` compares whole values only).
fn check_prefix(case: &Case, key: &str, got: &[f64], tol: Tol) {
    let want = case.vector(key);
    assert!(
        !got.is_empty() && got.len() <= want.len(),
        "case {}: {key}: {} values against {}",
        case.id,
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            tol.allows(*g, *w),
            "case {}: {key}[{i}] rust={g} R={w}",
            case.id
        );
    }
}

/// TOOL-26: the scores are returned (the first rows, with a note) or stored as columns under
/// output_name. Before: only scores_computed and scores_shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_factanal_scores_returned() {
    let case = Case::load("mcp_factanal_t3");
    let (se, s) = session_with("mcp_factanal_t3");
    let cols = json!(["f1", "f2", "f3", "f4", "f5", "f6"]);
    // factanal rule; the reference is the varimax optimum (tight L-BFGS-B and rotation)
    let fa = Tol::iterative(1e-4);
    for (method, key) in [
        ("bartlett", "bartlett_rows"),
        ("Thomson", "regression_rows"),
    ] {
        let args = json!({"dataset": "mcp_factanal_t3", "columns": cols.clone(), "n_factors": 2, "scores": method});
        let j = json_of(&ok(&s, &se, "multivariate_factanal", args).await);
        let sc = &j["scores"];
        assert_eq!(
            (
                sc["n_rows"].as_u64(),
                sc["rows_returned"].as_u64(),
                sc["columns"].clone()
            ),
            (Some(1200), Some(50), json!(["Factor1", "Factor2"]))
        );
        assert!(sc["note"].as_str().unwrap().contains("50 of 1200"), "{sc}");
        // R rows 1, 2, 3 = the first 6 entries of the row-major case vector (rows 999+ are not
        // returned inline)
        let got: Vec<f64> = [0usize, 1, 2]
            .iter()
            .flat_map(|&i| fs(&sc["rows"][i]))
            .collect();
        check_prefix(&case, key, &got, fa);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_factanal_scores_output_name() {
    let case = Case::load("mcp_factanal_t3");
    let (se, s) = session_with("mcp_factanal_t3");
    let cols = json!(["f1", "f2", "f3", "f4", "f5", "f6"]);
    for (method, key, name) in [
        ("bartlett", "bartlett_rows", "fa_bartlett"),
        ("regression", "regression_rows", "fa_regression"),
    ] {
        let args = json!({"dataset": "mcp_factanal_t3", "columns": cols.clone(), "n_factors": 2,
                          "scores": method, "output_name": name});
        let j = json_of(&ok(&s, &se, "multivariate_factanal", args).await);
        assert_eq!(
            (j["scores"]["saved_as"].as_str(), j["scores"].get("rows")),
            (Some(name), None)
        );
        let ds = se.datasets.read().await[name].clone();
        assert_eq!((ds.df().height(), ds.df().width()), (1200, 8));
        // na-ok: columns the handler built from a finite score matrix
        let col = |c: &str| {
            ds.df()
                .column(c)
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>()
        };
        let (f1, f2) = (col("Factor1"), col("Factor2"));
        let got: Vec<f64> = [0usize, 1, 2, 998, 999, 1000, 1198, 1199]
            .iter()
            .flat_map(|&i| [f1[i], f2[i]])
            .collect();
        check_prefix(&case, key, &got, Tol::iterative(1e-4)); // all 8 rows, 16 values
    }
    // output_name without scores, and a name that collides with an input column, are refused
    assert!(
        refused(
            &s,
            &se,
            "multivariate_factanal",
            json!({"dataset": "mcp_factanal_t3", "columns": cols.clone(), "n_factors": 2, "output_name": "x"})
        )
        .await
        .contains("scores")
    );
    assert!(
        refused(
            &s,
            &se,
            "multivariate_factanal",
            json!({"dataset": "fa_bartlett", "columns": cols, "n_factors": 2, "scores": "bartlett", "output_name": "again"})
        )
        .await
        .contains("already has a column 'Factor1'")
    );
}

/// Root tolerance against R's power.*.test with its default uniroot tolerance: the core's
/// zeroin follows R's iteration, so the roots agree to a few ulps plus the root tolerance.
fn root_tol() -> Tol {
    Tol::new(
        8.0 * f64::EPSILON,
        2.0 * p2a_core::stats::power::R_UNIROOT_TOL,
    )
}

/// Power surface (STAT-61's MCP half): sd and sig_level default to 1 and 0.05; the omitted
/// parameter is solved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_power_solve_defaults() {
    let case = Case::load("stat_power_solve");
    let (se, s) = session_with_datasets(vec![]);
    let p = json_of(&ok(&s, &se, "power_t_test", json!({"n": 30.0, "delta": 0.5})).await);
    let n = json_of(&ok(&s, &se, "power_t_test", json!({"delta": 0.5, "power": 0.8})).await);
    Check::new(&case)
        .scalar("t_n30_d05_power", f(&p["power"]), Tol::new(1e-10, 0.0))
        .scalar("t_d05_p08_n_pkg", f(&n["n"]), root_tol())
        .finish();
    assert_eq!(
        (n["solved_for"].as_str(), p["solved_for"].as_str()),
        (Some("n"), Some("power"))
    );
    assert!(n["root"]["iterations"].as_u64().unwrap() >= 1);
}

/// Power surface: solve_for and the anova/prop solvers. Before: solve_for was an unknown
/// field, power_anova_test assumed within_var = 1 when it was omitted, and required
/// between_var and groups.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_power_solve_for() {
    let case = Case::load("stat_power_solve");
    let (se, s) = session_with_datasets(vec![]);
    let root = root_tol();
    let power = Tol::new(1e-10, 0.0);
    let sd = json_of(
        &ok(
            &s,
            &se,
            "power_t_test",
            json!({"n": 20.0, "delta": 0.8, "power": 0.9, "solve_for": "sd"}),
        )
        .await,
    );
    let sd2 = json_of(
        &ok(
            &s,
            &se,
            "power_t_test",
            json!({"n": 20.0, "delta": 0.8, "power": 0.9, "sig_level": 0.05}),
        )
        .await,
    );
    let pn = json_of(
        &ok(
            &s,
            &se,
            "power_prop_test",
            json!({"p1": 0.3, "p2": 0.5, "power": 0.8}),
        )
        .await,
    );
    let pp = json_of(
        &ok(
            &s,
            &se,
            "power_prop_test",
            json!({"n": 100.0, "p1": 0.3, "p2": 0.5}),
        )
        .await,
    );
    let ap = json_of(
        &ok(
            &s,
            &se,
            "power_anova_test",
            json!({"groups": 4, "n": 5.0, "between_var": 1.0, "within_var": 3.0}),
        )
        .await,
    );
    let ag = json_of(
        &ok(
            &s,
            &se,
            "power_anova_test",
            json!({"n": 5.0, "between_var": 1.0, "within_var": 3.0, "power": 0.8}),
        )
        .await,
    );
    let aw = json_of(
        &ok(
            &s,
            &se,
            "power_anova_test",
            json!({"groups": 4, "n": 5.0, "between_var": 1.0, "power": 0.8}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("t_twosample_twosided_sd_pkg", f(&sd["sd"]), root)
        .scalar("t_twosample_twosided_sd_pkg", f(&sd2["sd"]), root)
        .scalar("prop_p03_p05_n_pkg", f(&pn["n"]), root)
        .scalar("prop_n100_power", f(&pp["power"]), power)
        .scalar("anova_power", f(&ap["power"]), power)
        .scalar("anova_groups_pkg", f(&ag["groups"]), root)
        .scalar("anova_within_pkg", f(&aw["within_var"]), root)
        .finish();
    assert_eq!(
        (sd["solved_for"].as_str(), aw["solved_for"].as_str()),
        (Some("sd"), Some("within.var"))
    );
    // Cohen's f from the core: sqrt((groups - 1) / groups * between / within) = sqrt(3/4 * 1/3)
    assert!((f(&ap["effect_size_f"]) - 0.5).abs() < 1e-12, "{ap}");
    for (tool, args, needle) in [
        (
            "power_t_test",
            json!({"n": 20.0, "delta": 0.8, "sd": 1.0, "power": 0.9, "solve_for": "sd"}),
            "omit it",
        ),
        (
            "power_t_test",
            json!({"n": 20.0}),
            "exactly one of n, delta, power",
        ),
        (
            "power_t_test",
            json!({"n": 20.0, "delta": 0.8, "power": 0.9, "solve_for": "effect"}),
            "expected one of",
        ),
        (
            "power_anova_test",
            json!({"groups": 4, "n": 5.0}),
            "exactly one of",
        ),
    ] {
        assert!(
            refused(&s, &se, tool, args).await.contains(needle),
            "{tool}: {needle}"
        );
    }
}

/// Mauchly's test against a user Sigma (R's mauchly.test(Sigma = ..)). The p-value is
/// Anderson's second-order p with the corrected coefficient, pinned to the case's
/// `*_p_anderson`; R's p (with its typo, STAT-82) is a cross-check. Before: sigma was an
/// unknown field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_stats_mauchly_sigma() {
    let case = Case::load("stat_mauchly_sigma");
    let (se, s) = session_with("stat_mauchly_sigma");
    let cols = json!(["V1", "V2", "V3", "V4"]);
    // diag(1, 2, 3, 4)
    let sigma = json!([
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0
    ]);
    let w = json_of(
        &ok(
            &s,
            &se,
            "stats_mauchly_test",
            json!({"dataset": "mauchly_sigma", "columns": cols, "sigma": sigma}),
        )
        .await,
    );
    let i = json_of(
        &ok(
            &s,
            &se,
            "stats_mauchly_test",
            json!({"dataset": "mauchly_sigma", "columns": cols}),
        )
        .await,
    );
    let cross = Tol::new(1e-3, 1e-12); // deviation STAT-82: R's p is a cross-check only
    Check::new(&case)
        .scalar("psi_contrasts_W", f(&w["w_statistic"]), Tol::stat(15))
        .scalar("psi_contrasts_chisq", f(&w["chi_squared"]), Tol::stat(15))
        .scalar("psi_contrasts_df", f(&w["df"]), Tol::exact())
        .scalar(
            "psi_contrasts_p_anderson",
            f(&w["p_value"]),
            Tol::p_value(15),
        )
        .scalar("psi_contrasts_p", f(&w["p_value"]), cross)
        .scalar(
            "psi_contrasts_p_first_order",
            f(&w["p_value_first_order"]),
            Tol::p_value(15),
        )
        .scalar(
            "psi_contrasts_gg",
            f(&w["epsilon_corrections"]["greenhouse_geisser"]),
            Tol::stat(15),
        )
        .scalar("contrasts_W", f(&i["w_statistic"]), Tol::stat(15))
        .scalar("contrasts_p_anderson", f(&i["p_value"]), Tol::p_value(15))
        .scalar("contrasts_p", f(&i["p_value"]), cross)
        .finish();
    assert_eq!(
        (w["sigma"].as_str(), i["sigma"].as_str()),
        (Some("user-supplied (4 x 4, row-major)"), Some("identity"))
    );
    let bad = json!({"dataset": "mauchly_sigma", "columns": cols, "sigma": [1.0, 0.0, 0.0, 1.0]});
    assert!(
        refused(&s, &se, "stats_mauchly_test", bad)
            .await
            .contains("16")
    );
}

/// STAT-28's MCP half: the non-decomposable loglin model iterates to eps and reports
/// non-convergence. Before: no warnings key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_stats_loglin_nondecomposable() {
    let case = Case::load("stat_loglin_nodecomp");
    let (se, s) = session_with("stat_loglin_nodecomp");
    let base = json!({"dataset": "loglin_2x3x4", "count_column": "Freq", "factor_columns": ["Var1", "Var2", "Var3"],
                      "margins": [[0, 1], [0, 2], [1, 2]], "eps": 1e-12, "max_iter": 1000});
    let j = json_of(&ok(&s, &se, "stats_loglin", base.clone()).await);
    let mut short = base.clone();
    short["max_iter"] = json!(3);
    let k = json_of(&ok(&s, &se, "stats_loglin", short).await);
    let gof = &j["goodness_of_fit"];
    // The iteration count and the 3-iteration statistic are not pinned: they depend on the
    // rounding of each IPF step near eps.
    Check::new(&case)
        .scalar(
            "lrt",
            f(&gof["likelihood_ratio_test"]["statistic"]),
            Tol::stat(720),
        )
        .scalar(
            "pearson",
            f(&gof["pearson_chi_squared"]["statistic"]),
            Tol::stat(720),
        )
        .scalar("df", f(&gof["likelihood_ratio_test"]["df"]), Tol::exact())
        .finish();
    assert_eq!(
        (
            j["convergence"]["converged"].as_bool(),
            j["warnings"].as_array().map(Vec::len)
        ),
        (Some(true), Some(0))
    );
    assert!(j["convergence"]["max_deviation"].as_f64().unwrap() <= 1e-12);
    assert_eq!(k["convergence"]["converged"].as_bool(), Some(false));
    assert!(
        k["warnings"][0]
            .as_str()
            .unwrap()
            .contains("did not converge"),
        "{k}"
    );
}

/// STAT-38's MCP surface: an AR spectrum has no chi-square interval; the periodogram's is R's
/// spec.ci at the reported df. Before: confidence_interval_95 always carried two multipliers
/// (a NaN pair for AR).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_timeseries_spectrum_ci() {
    let case = Case::load("stat_spectrum_t3");
    let (se, s) = session_with("stat_spectrum_t3");
    let p = json_of(
        &ok(
            &s,
            &se,
            "timeseries_spectrum",
            json!({"dataset": "spectrum_t3", "column": "x", "spans": [3, 3]}),
        )
        .await,
    );
    let ci = &p["confidence_interval_95"];
    // deviation STAT-39: p2a's df for spans c(3, 3) is 18 where spec.pgram reports
    // 6.552102376599634; the multipliers are pinned to R's spec.ci at p2a's df (ci_df[2] = 18)
    assert_eq!(f(&ci["degrees_of_freedom"]), case.vector("ci_df")[2]);
    let want = case.vector("ci95_at_df");
    for (got, w) in [
        (f(&ci["multiplier_lower"]), want[4]),
        (f(&ci["multiplier_upper"]), want[5]),
    ] {
        // the core's spec.ci reproduces R to 1e-10 relative
        assert!(Tol::new(1e-10, 0.0).allows(got, w), "rust={got} R={w}");
    }
    Check::new(&case)
        .scalar("peak_freq", f(&p["peak"]["frequency"]), Tol::coef(96))
        .finish();
    let a = json_of(
        &ok(
            &s,
            &se,
            "timeseries_spectrum",
            json!({"dataset": "spectrum_t3", "column": "x", "method": "ar"}),
        )
        .await,
    );
    assert!(
        a["confidence_interval_95"].is_null()
            && a["confidence_interval_note"]
                .as_str()
                .unwrap()
                .contains("spec.ar"),
        "{a}"
    );
}

/// STAT-54's MCP half (closes TOOL-10): the outlier threshold is qchisq(0.95, rank), and a
/// singular covariance is refused with its rank. Before: "8.1000" (2.7 p) as a string.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_mahalanobis_threshold() {
    let case = Case::load("stat_mahalanobis_t3");
    let misc = Case::load("mcp_misc_t3");
    let (se, s) = session_with("stat_mahalanobis_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "multivariate_mahalanobis",
            json!({"dataset": "mahalanobis_t3", "columns": ["f1", "f2", "f3"]}),
        )
        .await,
    );
    let od = &j["outlier_detection"];
    Check::new(&misc)
        .scalar(
            "qchisq95_df3",
            f(&od["chi_squared_threshold_95"]),
            Tol::coef(30),
        )
        .finish();
    let d10: Vec<f64> = j["first_10_distances"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| f(&e["distance_squared"]))
        .collect();
    let want = case.vector("full_d2");
    for (g, w) in d10.iter().zip(&want) {
        assert!(Tol::printed(4).allows(*g, *w), "rust={g} R={w}");
    }
    assert_eq!(
        (
            j["rank"].as_u64(),
            od["df"].as_u64(),
            od["n_potential_outliers"].as_u64()
        ),
        (Some(3), Some(3), Some(0))
    );
    // b = 2 a: R's mahalanobis refuses (ab_r_error 1); rank 1 of 2
    let e = refused(
        &s,
        &se,
        "multivariate_mahalanobis",
        json!({"dataset": "mahalanobis_t3", "columns": ["a", "b"]}),
    )
    .await;
    assert!(e.contains("rank 1 of 2"), "{e}");
}
