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

const CANADA: [&str; 4] = ["e", "prod", "rw", "U"];

/// ECON-49's MCP half: ts_var returns coefficients, SEs, t and p per equation as JSON.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_var_coefficients() {
    let case = Case::load("var_canada");
    let (se, s) = session_with("var_canada");
    let t = ok(
        &s,
        &se,
        "ts_var",
        json!({"dataset": "var_canada", "columns": CANADA, "lags": 2}),
    )
    .await;
    assert!(t.starts_with("Vector Autoregression VAR(2)"), "{t}"); // the Display stays first
    let j = json_of(&t);
    let names: Vec<String> = j["coef_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut c = Check::new(&case);
    for (k, eq) in CANADA.iter().enumerate() {
        c.named(
            &format!("coef_{eq}"),
            &names,
            &fs(&j["coefficients"][k]),
            Tol::coef(82),
        )
        .named(
            &format!("se_{eq}"),
            &names,
            &fs(&j["std_errors"][k]),
            Tol::se(82),
        )
        .named(
            &format!("t_{eq}"),
            &names,
            &fs(&j["t_values"][k]),
            Tol::stat(82),
        )
        .named(
            &format!("p_{eq}"),
            &names,
            &fs(&j["p_values"][k]),
            Tol::p_value(82),
        );
    }
    let sigma: Vec<f64> = j["sigma_u"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(fs)
        .collect();
    c.vector("sigma_df", &sigma, Tol::stat(82))
        .scalar("loglik", f(&j["log_likelihood"]), Tol::stat(82))
        .scalar("aic_stats", f(&j["aic"]), Tol::stat(82))
        .scalar("bic_stats", f(&j["bic"]), Tol::stat(82))
        .scalar("df_resid", f(&j["df_residual"]), Tol::exact())
        .scalar("n_obs", f(&j["n_obs"]), Tol::exact())
        .finish();
}

/// Every impulse-response horizon as JSON. `steps` counts horizons including h = 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_var_irf() {
    let case = Case::load("var_canada");
    let (se, s) = session_with("var_canada");
    let j = json_of(
        &ok(
            &s,
            &se,
            "ts_var_irf",
            json!({"dataset": "var_canada", "columns": CANADA, "lags": 2, "steps": 13}),
        )
        .await,
    );
    assert_eq!(
        (j["steps"].as_u64(), j["orthogonalized"].as_bool()),
        (Some(13), Some(true))
    );
    let mut c = Check::new(&case);
    for (i, imp) in CANADA.iter().enumerate() {
        // R irf_ortho_<impulse>: 13 x 4 column-major (response r at horizon h = index h + 13 r)
        let got: Vec<f64> = (0..4)
            .flat_map(|r| (0..13).map(move |h| (h, r)))
            .map(|(h, r)| f(&j["irf"][h][r][i]))
            .collect();
        c.vector(&format!("irf_ortho_{imp}"), &got, Tol::coef(82));
    }
    c.finish();
    let steps = AnalyticsServer::new()
        .router_tools()
        .into_iter()
        .find(|t| t.name == "ts_var_irf")
        .unwrap()
        .input_schema["properties"]["steps"]["description"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        steps.contains("h = 0") && steps.contains("n.ahead + 1"),
        "{steps}"
    );
}

/// ECON-53's MCP half: the Granger F with df and the capped default order with its warning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_granger() {
    let case = Case::load("granger_short");
    let (se, s) = session_with("granger_short");
    let j = json_of(
        &ok(
            &s,
            &se,
            "ts_granger",
            json!({"dataset": "granger_short", "dependent": "y", "cause": "x", "lags": 2}),
        )
        .await,
    );
    let d = json_of(
        &ok(
            &s,
            &se,
            "ts_granger",
            json!({"dataset": "granger_short", "dependent": "y", "cause": "x"}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("lag2_F", f(&j["f_statistic"]), Tol::stat(20))
        .scalar("lag2_p", f(&j["p_value"]), Tol::p_value(20))
        .scalar("lag2_df1", f(&j["df1"]), Tol::exact())
        .scalar("lag2_df2", f(&j["df2"]), Tol::exact())
        .scalar("lag2_n_used", f(&j["n_obs"]), Tol::exact())
        .scalar("default_lag", f(&d["lags"]), Tol::exact())
        .scalar("lag6_F", f(&d["f_statistic"]), Tol::stat(20))
        .finish();
    assert!(!d["warnings"].as_array().unwrap().is_empty(), "{d}");
}

/// Hannan-Rissanen VARMA(1, 1) at a given and at the default long-VAR order, and VARMA(1, 0)
/// with its exact OLS standard errors. Before: long_var_order was an unknown field and the
/// output was text only. The second step of a q > 0 fit regresses on estimated residuals, so
/// no standard errors are reported for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_varma() {
    let case = Case::load("varma_hr");
    let (se, s) = session_with("varma_hr");
    let run = |p: usize, q: usize, m: Option<usize>| {
        let mut a = json!({"dataset": "varma_sim", "columns": ["y1", "y2"], "p": p, "q": q});
        if let Some(m) = m {
            a["long_var_order"] = json!(m);
        }
        a
    };
    let j = json_of(&ok(&s, &se, "ts_varma", run(1, 1, Some(8))).await);
    let d = json_of(&ok(&s, &se, "ts_varma", run(1, 1, None)).await);
    let a = json_of(&ok(&s, &se, "ts_varma", run(1, 0, Some(8))).await);
    let names = |v: &Value| -> Vec<String> {
        v["regressor_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    };
    let n = case.scalar("n_eff") as usize;
    let nd = case.scalar("default_n_eff") as usize;
    let na = case.scalar("p1q0_lv8_n_eff") as usize;
    let mut c = Check::new(&case);
    for (k, eq) in ["y1", "y2"].iter().enumerate() {
        c.named(
            &format!("coef_{eq}"),
            &names(&j),
            &fs(&j["coefficients"][k]),
            Tol::coef(n),
        )
        .named(
            &format!("p1q0_lv8_coef_{eq}"),
            &names(&a),
            &fs(&a["coefficients"][k]),
            Tol::coef(na),
        )
        .named(
            &format!("p1q0_lv8_se_{eq}"),
            &names(&a),
            &fs(&a["std_errors"][k]),
            Tol::se(na),
        )
        .named(
            &format!("p1q0_lv8_p_{eq}"),
            &names(&a),
            &fs(&a["p_values"][k]),
            Tol::p_value(na),
        );
    }
    c.scalar("loglik", f(&j["log_likelihood"]), Tol::stat(n))
        .scalar("aic", f(&j["aic"]), Tol::stat(n))
        .scalar("bic", f(&j["bic"]), Tol::stat(n))
        .scalar("n_eff", f(&j["n_obs"]), Tol::exact())
        .scalar("df_residual", f(&j["df_residual"]), Tol::exact())
        .scalar("n_params", f(&j["n_params"]), Tol::exact())
        .scalar("long_var_order", f(&j["long_var_order"]), Tol::exact())
        .named(
            "default_coef_y1",
            &names(&d),
            &fs(&d["coefficients"][0]),
            Tol::coef(nd),
        )
        .scalar(
            "default_long_var_order",
            f(&d["long_var_order"]),
            Tol::exact(),
        )
        .scalar("default_loglik", f(&d["log_likelihood"]), Tol::stat(nd))
        .scalar("p1q0_lv8_loglik", f(&a["log_likelihood"]), Tol::stat(na))
        .scalar("p1q0_lv8_n_eff", f(&a["n_obs"]), Tol::exact())
        .finish();
    assert!(
        j["std_errors"].as_array().is_none_or(|v| v.is_empty()),
        "{}",
        j["std_errors"]
    );
}

/// Johansen VECM options. Before: ecdet was an unknown field, rank was required and the
/// output was text only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_vecm() {
    let case = Case::load("vecm_denmark");
    let (se, s) = session_with("vecm_denmark");
    let cols = json!(["LRM", "LRY", "IBO", "IDE"]);
    let run = |extra: Value| {
        let mut a = json!({"dataset": "vecm_denmark", "columns": cols, "lags": 2});
        for (k, v) in extra.as_object().unwrap() {
            a[k] = v.clone();
        }
        a
    };
    // urca lists H0 r <= k-1 first; p2a r = 0 first
    let rev = |key: &str| {
        let mut v = case.vector(key);
        v.reverse();
        v
    };
    let close = |got: &[f64], want: &[f64], tol: Tol, what: &str| {
        assert_eq!(got.len(), want.len(), "{what}");
        for (g, w) in got.iter().zip(want) {
            assert!(tol.allows(*g, *w), "{what}: rust={g} R={w}");
        }
    };
    let n = 53;
    let c = json_of(
        &ok(
            &s,
            &se,
            "ts_vecm",
            run(json!({"ecdet": "Constant", "rank": 1})),
        )
        .await,
    );
    close(
        &fs(&c["trace_stat"]),
        &rev("trace_stat"),
        Tol::stat(n),
        "trace_stat",
    );
    close(
        &c["trace_cv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| f(&r[1]))
            .collect::<Vec<_>>(),
        &rev("trace_cv5"),
        Tol::exact(),
        "trace cv 5%",
    );
    close(
        &fs(&c["eigen_stat"]),
        &rev("eigen_stat"),
        Tol::stat(n),
        "eigen_stat",
    );
    close(
        &fs(&c["eigenvalues"])[..4],
        &case.vector("eigenvalues")[..4],
        Tol::stat(n),
        "eigenvalues",
    );
    close(
        &c["beta"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| f(&r[0]))
            .collect::<Vec<_>>(),
        &case.vector("beta"),
        Tol::coef(n),
        "beta",
    );
    // coef(cajorls(..)$rlm): 5 rows (ect1, 4 x .dl1) x 4 equations, column-major
    let r = case.vector("rlm_coef");
    close(
        &c["alpha"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| f(&a[0]))
            .collect::<Vec<_>>(),
        &[r[0], r[5], r[10], r[15]],
        Tol::coef(n),
        "alpha",
    );
    for i in 0..4 {
        close(
            &fs(&c["gamma"][i]),
            &r[5 * i + 1..5 * i + 5],
            Tol::coef(n),
            "gamma",
        );
    }
    assert_eq!(
        (
            c["deterministic"].as_str(),
            c["selected_rank"].as_u64(),
            c["rank"].as_u64()
        ),
        (Some("const"), Some(0), Some(1))
    );
    // ecdet none: the trace test selects r = 1 when rank is omitted
    let none = json_of(&ok(&s, &se, "ts_vecm", run(json!({}))).await);
    close(
        &fs(&none["trace_stat"]),
        &rev("none_trace_stat"),
        Tol::stat(n),
        "none trace",
    );
    assert_eq!(
        (none["selected_rank"].as_u64(), none["rank"].as_u64()),
        (Some(1), Some(1))
    );
    close(
        &none["beta"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| f(&r[0]))
            .collect::<Vec<_>>(),
        &case.vector("none_beta"),
        Tol::coef(n),
        "none beta",
    );
    // const without a rank: the trace test selects 0, the restricted VECM is not estimated, a
    // warning says so
    let zero = json_of(&ok(&s, &se, "ts_vecm", run(json!({"ecdet": "const"}))).await);
    assert!(
        zero["beta"].as_array().unwrap().is_empty()
            && !zero["warnings"].as_array().unwrap().is_empty(),
        "{zero}"
    );
    for (extra, needle) in [
        (
            json!({"ecdet": "linear"}),
            "expected one of: none, const, trend",
        ),
        (json!({"lags": 1}), "at least 2"),
    ] {
        assert!(
            refused(&s, &se, "ts_vecm", run(extra))
                .await
                .contains(needle),
            "{needle}"
        );
    }
}

/// GARCH with Student-t errors and variance forecasts on the MCP surface. Before:
/// distribution and horizon were unknown fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_garch_fit_t() {
    const PAR: Tol = Tol::new(1e-8, 0.0);
    const SE: Tol = Tol::new(1e-6, 0.0);
    let case = Case::load("garch_normal_t");
    let (se, s) = session_with("garch_normal_t");
    let names = |v: &Value| -> Vec<String> {
        v["param_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    };
    let n = json_of(
        &ok(
            &s,
            &se,
            "ts_garch_fit",
            json!({"dataset": "garch_returns", "column": "r_norm", "horizon": 10}),
        )
        .await,
    );
    let t = json_of(
        &ok(
            &s,
            &se,
            "ts_garch_fit",
            json!({"dataset": "garch_returns", "column": "r_t", "distribution": "t", "horizon": 10}),
        )
        .await,
    );
    Check::new(&case)
        .named("norm_par_tight", &names(&n), &fs(&n["estimates"]), PAR)
        .named("norm_se_tight", &names(&n), &fs(&n["std_errors"]), SE)
        .scalar(
            "norm_loglik_tight",
            f(&n["log_likelihood"]),
            Tol::stat(1500),
        )
        .vector(
            "norm_forecast_var_tight",
            &fs(&n["forecast"]["variance"]),
            PAR,
        )
        .named("std_par_tight", &names(&t), &fs(&t["estimates"]), PAR)
        .named("std_se_tight", &names(&t), &fs(&t["std_errors"]), SE)
        .scalar("std_loglik_tight", f(&t["log_likelihood"]), Tol::stat(1500))
        .vector(
            "std_forecast_var_tight",
            &fs(&t["forecast"]["variance"]),
            PAR,
        )
        .finish();
    assert_eq!(
        (n["distribution"].as_str(), t["distribution"].as_str()),
        (Some("norm"), Some("std"))
    );
    let vol = fs(&t["forecast"]["volatility"]);
    assert!((vol[0] - fs(&t["forecast"]["variance"])[0].sqrt()).abs() < 1e-15);
    assert!(
        refused(
            &s,
            &se,
            "ts_garch_fit",
            json!({"dataset": "garch_returns", "column": "r_t", "distribution": "ged"})
        )
        .await
        .contains("expected one of: norm, std")
    );
}

/// Reads the `t+h:` forecast rows of `ts_holt_winters`: the point forecast, then the
/// interval limits when the row has them.
fn hw_rows(t: &str, horizon: usize) -> Vec<Vec<f64>> {
    (1..=horizon)
        .map(|h| {
            let line = t
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("t+{h}:")))
                .unwrap_or_else(|| panic!("t+{h} in\n{t}"));
            line.split([':', '[', ']', ','])
                .skip(1)
                .filter_map(|x| x.trim().parse().ok())
                .collect()
        })
        .collect()
}

/// Holt-Winters forecasts: predict.HoltWinters' 95% intervals for an additive fit, point
/// forecasts and a note for a multiplicative one (R's multiplicative interval formula is
/// wrong). Before: point forecasts only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ts_holt_winters_forecast() {
    let mult_case = Case::load("ts_holtwinters_forecast");
    let add_case = Case::load("mcp_hw_additive_t3");
    let (se, s) = session_with("ts_holtwinters_forecast");
    let call = |seasonal: &str| {
        json!({"dataset": "holtwinters_series", "column": "airpass", "period": 12,
            "seasonal": seasonal, "alpha": 0.3, "beta": 0.05, "gamma": 0.4, "horizon": 12})
    };
    let m = ok(&s, &se, "ts_holt_winters", call("multiplicative")).await;
    let rows = hw_rows(&m, 12);
    assert!(rows.iter().all(|r| r.len() == 1), "{m}");
    let fit: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    Check::new(&mult_case)
        .vector("airpass_mult_fit", &fit, Tol::printed(4))
        .scalar(
            "airpass_mult_sse",
            number_after(&m, "Sum of Squared Errors:"),
            Tol::printed(4),
        )
        .finish();
    assert!(
        m.contains("prediction intervals are implemented for additive and non-seasonal fits only"),
        "{m}"
    );

    let a = ok(&s, &se, "ts_holt_winters", call("additive")).await;
    let rows = hw_rows(&a, 12);
    assert!(rows.iter().all(|r| r.len() == 3), "{a}");
    let col = |i: usize| -> Vec<f64> { rows.iter().map(|r| r[i]).collect() };
    Check::new(&add_case)
        .vector("hw_add_fit", &col(0), Tol::printed(4))
        .vector("hw_add_lwr", &col(1), Tol::printed(4))
        .vector("hw_add_upr", &col(2), Tol::printed(4))
        .scalar(
            "hw_add_sse",
            number_after(&a, "Sum of Squared Errors:"),
            Tol::printed(4),
        )
        .finish();
    assert!(a.contains("95% prediction intervals"), "{a}");
}

/// cpgram's band and the Stephens statistic on a white-noise series (missing rows
/// dropped, as R's cpgram). Before: no critical_value key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_timeseries_cpgram() {
    let case = Case::load("ts_filter_ar_cpt");
    let (se, s) = session_with("ts_filter_ar_cpt");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_cpgram",
            json!({"dataset": "ts_utils", "column": "wn"}),
        )
        .await,
    );
    let wn = &j["white_noise_test"];
    let freq = case.vector("cpgram_wn_freq");
    let cum = case.vector("cpgram_wn_cpgram");
    for (got, want) in [
        (fs(&j["freq_sample"]), &freq[..20]),
        (fs(&j["cpgram_sample"]), &cum[..20]),
    ] {
        assert_eq!(got.len(), 20);
        for (g, w) in got.iter().zip(want) {
            assert!(Tol::stat(100).allows(*g, *w), "rust={g} R={w}");
        }
    }
    Check::new(&case)
        .scalar("cpgram_wn_crit", f(&wn["critical_value"]), Tol::stat(100))
        .scalar("cpgram_wn_stat", f(&wn["ks_statistic"]), Tol::stat(100))
        .scalar("cpgram_wn_p", f(&wn["ks_p_value"]), Tol::p_value(100))
        .scalar("cpgram_wn_max_dev", f(&wn["max_deviation"]), Tol::stat(100))
        .finish();
    assert_eq!(
        (
            wn["is_white_noise"].as_bool(),
            j["missing"]["n_dropped"].as_u64()
        ),
        (Some(true), Some(110))
    );
}

/// StructTS variances as numbers, with the convergence notes. Before: 6-decimal strings.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_timeseries_structts_bsm() {
    let case = Case::load("structts_nile_ukgas");
    let (se, s) = session_with("structts_nile_ukgas");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_structts",
            json!({"dataset": "structts_series", "column": "ukgas", "model_type": "BSM", "period": 4}),
        )
        .await,
    );
    let v = &j["variance_coefficients"];
    assert!(v["level_variance"].is_number(), "{v}");
    let got = [
        f(&v["level_variance"]),
        f(&v["slope_variance"]),
        f(&v["seasonal_variance"]),
        f(&v["observation_variance"]),
    ];
    Check::new(&case)
        .named(
            "ukgas_coef_tight",
            &["level", "slope", "seas", "epsilon"],
            &got,
            Tol::iterative(1e-6),
        )
        .scalar(
            "ukgas_loglik_tight",
            f(&j["fit_statistics"]["log_likelihood"]),
            Tol::stat(108),
        )
        .finish();
    assert_eq!(
        (
            j["converged"].as_bool(),
            j["warnings"].as_array().map(Vec::len)
        ),
        (Some(true), Some(0))
    );
}

/// Clipped CausalImpact periods are reported. No R reference (CausalImpact is not
/// installed): a property test. Before: no warnings key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_causal_impact_reports_clipped_periods() {
    // holtwinters_series: t = 1..144, airpass
    let (se, s) = session_with("ts_holtwinters_forecast");
    let j = json_of(
        &ok(
            &s,
            &se,
            "causal_impact_analysis",
            json!({"dataset": "holtwinters_series", "response_col": "airpass", "time_col": "t",
                "pre_period_start": -5, "pre_period_end": 100,
                "post_period_start": 101, "post_period_end": 500}),
        )
        .await,
    );
    let w: Vec<&str> = j["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert!(
        w.iter()
            .any(|x| x.contains("pre_period start -5 is before the first time value 1")),
        "{w:?}"
    );
    assert!(
        w.iter()
            .any(|x| x.contains("post_period end 500 is after the last time value 144")),
        "{w:?}"
    );
    assert_eq!(
        (
            j["periods"]["pre_period"]["n_obs"].as_u64(),
            j["periods"]["post_period"]["n_obs"].as_u64()
        ),
        (Some(100), Some(44))
    );
    assert!(
        refused(
            &s,
            &se,
            "causal_impact_analysis",
            json!({"dataset": "holtwinters_series", "response_col": "airpass", "time_col": "t",
                "pre_period_start": 1, "pre_period_end": 100,
                "post_period_start": 120, "post_period_end": 110})
        )
        .await
        .contains("post_period")
    );
}

/// AR warnings in the JSON; the changepoint penalty, cpgram and filter descriptions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_timeseries_wp37_texts() {
    let (se, s) = session_with("mcp_series_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_ar",
            json!({"dataset": "mcp_series_t3", "column": "z", "method": "burg"}),
        )
        .await,
    );
    assert!(j["warnings"].is_array(), "{j}");
    let tools = s.router_tools();
    let tool = |name: &str| tools.iter().find(|t| t.name == name).unwrap();
    let desc = |name: &str| tool(name).description.clone().unwrap().to_string();
    let prop = |name: &str, p: &str| {
        tool(name).input_schema["properties"][p]["description"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let penalty = prop("ts_changepoint", "penalty");
    assert!(
        penalty.contains("MBIC") && !penalty.contains("BIC (log(n))"),
        "{penalty}"
    );
    assert!(desc("timeseries_cpgram").contains("1.358"));
    assert!(prop("timeseries_filter", "sides").contains("centred"));
}

/// Asserts `got` against element `i` of the case vector `key`.
fn check_elem(case: &Case, key: &str, i: usize, got: f64, tol: Tol) {
    let want = case.vector(key)[i];
    assert!(
        tol.allows(got, want),
        "case {}: {key}[{i}] rust={got} R={want}",
        case.id
    );
}

/// The case value `key` as an integer count (exact comparison).
fn case_count(case: &Case, key: &str) -> u64 {
    let v = case.vector(key);
    assert_eq!(v.len(), 1, "case {}: {key} is not a scalar", case.id);
    v[0] as u64
}

/// STAT-46/STAT-77: anova_tukey_hsd through the dispatch path, pinned to TukeyHSD(aov())
/// (mcp_groups_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_tukey_hsd() {
    let case = Case::load("mcp_groups_t3");
    let (se, s) = session_with("mcp_groups_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "anova_tukey_hsd",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g"}),
        )
        .await,
    );
    let cmp = j["comparisons"].as_array().unwrap();
    let pairs: Vec<(&str, &str)> = cmp
        .iter()
        .map(|c| (c["group1"].as_str().unwrap(), c["group2"].as_str().unwrap()))
        .collect();
    assert_eq!(
        pairs,
        [("trtA", "ctl"), ("trtB", "ctl"), ("trtB", "trtA")],
        "{j}"
    );
    let col = |k: &str| -> Vec<f64> { cmp.iter().map(|c| f(&c[k])).collect() };
    Check::new(&case)
        .vector("tukey_diff", &col("diff"), Tol::coef(30))
        .vector("tukey_lwr", &col("ci_lower"), Tol::stat(30))
        .vector("tukey_upr", &col("ci_upper"), Tol::stat(30))
        .vector("tukey_padj", &col("p_adj"), Tol::p_value(30))
        .finish();
    check_elem(&case, "aov_ms", 1, f(&j["mse"]), Tol::stat(30));
    assert_eq!(j["df"].as_u64(), Some(27));
}

/// STAT-46/STAT-77: hypothesis_pairwise_wilcox through the dispatch path, pinned to
/// pairwise.wilcox.test(p.adjust.method = "holm", exact = FALSE) (mcp_groups_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_pairwise_wilcox() {
    let case = Case::load("mcp_groups_t3");
    let (se, s) = session_with("mcp_groups_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_pairwise_wilcox",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g",
                "p_adjust_method": "holm", "exact": false}),
        )
        .await,
    );
    let p = &j["p_values_adjusted"];
    let got = [f(&p[1][0]), f(&p[2][0]), f(&p[2][1])];
    Check::new(&case)
        .vector("pairwise_wilcox_holm", &got, Tol::p_value(30))
        .finish();
    assert_eq!(j["group_names"], json!(["ctl", "trtA", "trtB"]));
    assert_eq!(j["missing"]["n_dropped"].as_u64(), Some(3), "{j}");
}

/// STAT-46/STAT-77: hypothesis_wilcoxon (two samples) through the dispatch path, pinned to
/// wilcox.test(x, y, exact = FALSE, correct = TRUE) (mcp_groups_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_wilcoxon_two_sample() {
    let case = Case::load("mcp_groups_t3");
    let (se, s) = session_with("mcp_groups_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_wilcoxon",
            json!({"dataset": "mcp_groups_t3", "x": "s1", "y": "s2", "exact": false, "correct": true}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("wilcox_p", f(&j["p_value"]), Tol::p_value(64))
        .finish();
    let (n1, n2) = (j["n1"].as_u64().unwrap(), j["n2"].as_u64().unwrap());
    assert_eq!(
        (n1, n2),
        (
            case_count(&case, "wilcox_n1"),
            case_count(&case, "wilcox_n2")
        )
    );
    // deviation STAT-72: the tool reports the rank sum R1, R's W is R1 - n1 (n1 + 1) / 2
    let w = f(&j["statistic"]) - (n1 * (n1 + 1) / 2) as f64;
    assert_eq!(w, case.vector("wilcox_W")[0], "{j}");
}

/// STAT-46/STAT-77: hypothesis_mcnemar through the dispatch path, pinned to
/// mcnemar.test(matrix(c(., 15, 5, .))) with and without the correction (mcp_misc_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_mcnemar() {
    let case = Case::load("mcp_misc_t3");
    let (se, s) = session_with("mcp_misc_t3");
    let c = json_of(&ok(&s, &se, "hypothesis_mcnemar", json!({"b": 15, "c": 5})).await);
    let u = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_mcnemar",
            json!({"b": 15, "c": 5, "correct": false}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("mcnemar_b15_c5_stat", f(&c["statistic"]), Tol::stat(20))
        .scalar("mcnemar_b15_c5_p", f(&c["p_value"]), Tol::p_value(20))
        .scalar(
            "mcnemar_b15_c5_uncorrected_stat",
            f(&u["statistic"]),
            Tol::stat(20),
        )
        .scalar(
            "mcnemar_b15_c5_uncorrected_p",
            f(&u["p_value"]),
            Tol::p_value(20),
        )
        .finish();
}

/// STAT-46/STAT-77: hypothesis_poisson one-sided through the dispatch path, pinned to
/// poisson.test (stat_poisson_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_poisson_one_sided() {
    let case = Case::load("stat_poisson_t3");
    let (se, s) = session_with("mcp_misc_t3");
    let g = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_poisson",
            json!({"x": [10], "t": [5], "alternative": "greater"}),
        )
        .await,
    );
    let l = json_of(
        &ok(
            &s,
            &se,
            "hypothesis_poisson",
            json!({"x": [11, 21], "t": [800, 3011], "alternative": "less"}),
        )
        .await,
    );
    // deviation STAT-58: the two-sided p-value is not pinned here.
    // deviation new observation (core): the one-sided calls report the two-sided interval
    // (R: x10_greater_ci, rr_less_ci), so the interval bounds are not pinned.
    Check::new(&case)
        .scalar("x10_greater_p", f(&g["p_value"]), Tol::p_value(10))
        .scalar("rr_less_p", f(&l["p_value"]), Tol::p_value(10))
        .scalar("rr_est", f(&l["estimate"]["value"]), Tol::coef(10))
        .finish();
}

/// STAT-46/STAT-77: stats_model_tables (one way, means) through the dispatch path, pinned
/// to model.tables(aov(), "means") (mcp_groups_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_model_tables_one_way() {
    let case = Case::load("mcp_groups_t3");
    let (se, s) = session_with("mcp_groups_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "stats_model_tables",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factors": ["g"]}),
        )
        .await,
    );
    let t = &j["one_way_table"];
    let levels: Vec<String> = t["levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    // deviation STAT-75: the standard errors are not pinned.
    Check::new(&case)
        .named("means", &levels, &fs(&t["values"]), Tol::coef(30))
        .scalar("grand_mean", f(&j["grand_mean"]), Tol::coef(30))
        .vector("group_n", &fs(&t["n"]), Tol::exact())
        .finish();
    assert_eq!(levels, ["ctl", "trtA", "trtB"]);
}

/// STAT-46/STAT-77: stats_se_contrast through the dispatch path, pinned to
/// se.contrast(aov(), list(g == "ctl", g == "trtA")) (mcp_groups_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_se_contrast() {
    let case = Case::load("mcp_groups_t3");
    let (se, s) = session_with("mcp_groups_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "stats_se_contrast",
            json!({"dataset": "mcp_groups_t3", "response": "vi", "factor": "g", "contrasts": [[1, -1, 0]]}),
        )
        .await,
    );
    Check::new(&case)
        .scalar(
            "contrast_ctl_trtA_se",
            f(&j["standard_errors"][0]),
            Tol::se(30),
        )
        .finish();
    check_elem(&case, "aov_ms", 1, f(&j["mse"]), Tol::stat(30));
    assert_eq!(f(&j["df_mse"]), 27.0, "{j}");
}

/// STAT-46/STAT-77: compute_correlation through the dispatch path, pinned to cor()
/// (mcp_misc_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_compute_correlation() {
    let case = Case::load("mcp_misc_t3");
    let (se, s) = session_with("mcp_misc_t3");
    let t = ok(
        &s,
        &se,
        "compute_correlation",
        json!({"dataset": "robust_weighted_t3"}),
    )
    .await;
    let mut got = Vec::new();
    for label in ["x1", "x2", "w"] {
        // the data row, not the header (whose first token is a column name too)
        let row: Vec<f64> = t
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(label))
            .find_map(|l| {
                l.split_whitespace()
                    .skip(1)
                    .map(|x| x.parse().ok())
                    .collect::<Option<Vec<f64>>>()
            })
            .unwrap_or_else(|| panic!("row {label} in\n{t}"));
        assert_eq!(row.len(), 3, "{t}");
        got.extend(row);
    }
    Check::new(&case)
        .vector("cor_x1_x2_w", &got, Tol::printed(4))
        .finish();
}

/// STAT-46/STAT-77: multivariate_cancor through the dispatch path, pinned to cancor()'s
/// centres (stat_cancor_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_cancor_centres() {
    let case = Case::load("stat_cancor_t3");
    let (se, s) = session_with("stat_cancor_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "multivariate_cancor",
            json!({"dataset": "cancor_t3", "x_columns": ["x1", "x2", "x3"], "y_columns": ["y1", "y2"]}),
        )
        .await,
    );
    // deviation STAT-16: the correlations and coefficients are not pinned here.
    Check::new(&case)
        .named(
            "xcenter",
            &["x1", "x2", "x3"],
            &fs(&j["x_center"]),
            Tol::coef(100),
        )
        .named(
            "ycenter",
            &["y1", "y2"],
            &fs(&j["y_center"]),
            Tol::coef(100),
        )
        .finish();
    assert_eq!(
        (j["n_obs"].as_u64(), j["n_canonical"].as_u64()),
        (Some(100), Some(2))
    );
}

/// STAT-46/STAT-77: stats_iqr with quantile type 6 through the dispatch path, pinned to
/// IQR(type = 6) (stat_quantile_types).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_stats_iqr_type6() {
    let case = Case::load("stat_quantile_types");
    let (se, s) = session_with("stat_quantile_types");
    let j = json_of(
        &ok(
            &s,
            &se,
            "stats_iqr",
            json!({"dataset": "quantile_sample", "column": "x", "qtype": 6}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("x_iqr_type6", f(&j["iqr"]), Tol::coef(28))
        .finish();
    assert_eq!(j["quantile_type"].as_u64(), Some(6));
    let r = refused(
        &s,
        &se,
        "stats_iqr",
        json!({"dataset": "quantile_sample", "column": "x", "qtype": 12}),
    )
    .await;
    assert!(r.contains("1 to 9"), "{r}");
}

/// STAT-46/STAT-77: timeseries_pp_test through the dispatch path, pinned to PP.test and
/// its internals (stat_pp_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_pp_test() {
    let case = Case::load("stat_pp_t3");
    let (se, s) = session_with("stat_pp_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_pp_test",
            json!({"dataset": "pp_t3", "column": "rw"}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("rw_short_gamma", f(&j["gamma_hat"]), Tol::stat(200))
        .scalar("rw_short_t", f(&j["t_statistic"]), Tol::stat(200))
        .scalar("rw_short_sigma2", f(&j["sigma_squared"]), Tol::stat(200))
        .scalar("rw_short_lambda2", f(&j["lambda_squared"]), Tol::stat(200))
        .scalar("rw_short_stat", f(&j["statistic"]), Tol::stat(200))
        .scalar("rw_short_p", f(&j["p_value"]), Tol::p_value(200))
        .finish();
    assert_eq!(
        j["truncation_lag"].as_u64(),
        Some(case_count(&case, "rw_short_lag"))
    );
}

/// STAT-46/STAT-77: timeseries_box_test (Ljung-Box) through the dispatch path, pinned to
/// Box.test(type = "Ljung-Box") (mcp_series_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_box_test_ljung() {
    let case = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_box_test",
            json!({"dataset": "mcp_series_t3", "column": "w", "lag": 10, "test_type": "Ljung-Box"}),
        )
        .await,
    );
    Check::new(&case)
        .scalar("lb_stat", f(&j["statistic"]), Tol::stat(96))
        .scalar("lb_p", f(&j["p_value"]), Tol::p_value(96))
        .finish();
    assert_eq!(j["df"].as_u64(), Some(case_count(&case, "lb_df")));
}

/// STAT-46/STAT-77: timeseries_decompose (additive) through the dispatch path, pinned to
/// decompose() (mcp_series_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_decompose_additive() {
    let case = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_decompose",
            json!({"dataset": "mcp_series_t3", "column": "y", "period": 12}),
        )
        .await,
    );
    let figure: Vec<f64> = j["seasonal_figure"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f(&x["value"]))
        .collect();
    Check::new(&case)
        .vector("add_figure", &figure, Tol::printed(4))
        .scalar(
            "add_trend_mean",
            f(&j["summary"]["trend_mean"]),
            Tol::printed(4),
        )
        .scalar(
            "add_random_popvar",
            f(&j["summary"]["random_variance"]),
            Tol::printed(4),
        )
        .vector(
            "add_seasonal_head",
            &fs(&j["components_sample"]["first_5_seasonal"]),
            Tol::printed(4),
        )
        .finish();
    assert_eq!(
        j["components_sample"]["first_5_trend"],
        json!(["NA", "NA", "NA", "NA", "NA"])
    );
}

/// STAT-46/STAT-77: timeseries_ar (Burg) through the dispatch path, pinned to
/// ar(method = "burg") (mcp_series_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ar_burg() {
    let case = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_ar",
            json!({"dataset": "mcp_series_t3", "column": "z", "method": "burg"}),
        )
        .await,
    );
    assert_eq!(j["order"].as_u64(), Some(case_count(&case, "burg_order")));
    Check::new(&case)
        .vector("burg_ar", &fs(&j["ar_coefficients"]), Tol::coef(96))
        .scalar("burg_var", f(&j["prediction_variance"]), Tol::stat(96))
        .finish();
}

/// STAT-46/STAT-77: timeseries_runmed (constant end rule) through the dispatch path,
/// pinned to runmed(endrule = "constant") (mcp_series_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_runmed_constant() {
    let case = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    let j = json_of(
        &ok(
            &s,
            &se,
            "timeseries_runmed",
            json!({"dataset": "mcp_series_t3", "column": "z", "k": 5, "endrule": "constant"}),
        )
        .await,
    );
    // deviation TS-34: endrule 'median' is not pinned (R's runmed_median_head is recorded).
    Check::new(&case)
        .vector(
            "runmed_constant_head",
            &fs(&j["values_sample"]),
            Tol::coef(96),
        )
        .finish();
    assert_eq!(j["n"].as_u64(), Some(96));
}

/// STAT-46/STAT-77: embed, window, filter, diffinv, toeplitz, ARMAtoMA and acf2AR through
/// the dispatch path, pinned to R's functions of the same names (mcp_series_t3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_series_utilities() {
    let case = Case::load("mcp_series_t3");
    let (se, s) = session_with("mcp_series_t3");
    let z = |extra: Value| {
        let mut a = json!({"dataset": "mcp_series_t3", "column": "z"});
        a.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        a
    };
    let flat = |v: &Value, rows: usize| -> Vec<f64> {
        v.as_array().unwrap()[..rows].iter().flat_map(fs).collect()
    };
    let em = json_of(&ok(&s, &se, "timeseries_embed", z(json!({"dimension": 3}))).await);
    let wi = json_of(
        &ok(
            &s,
            &se,
            "timeseries_window",
            z(json!({"start": 10, "end": 20})),
        )
        .await,
    );
    let fi = json_of(
        &ok(
            &s,
            &se,
            "timeseries_filter",
            z(json!({"filter": [0.25, 0.25, 0.25, 0.25]})),
        )
        .await,
    );
    let di = json_of(&ok(&s, &se, "timeseries_diffinv", z(json!({"xi": [1.5]}))).await);
    let ts = json_of(
        &ok(
            &s,
            &se,
            "linalg_toeplitz",
            json!({"column": [1, 0.5, 0.25]}),
        )
        .await,
    );
    let ta = json_of(
        &ok(
            &s,
            &se,
            "linalg_toeplitz",
            json!({"column": [1, 2, 3], "row": [1, 4, 5]}),
        )
        .await,
    );
    let ma = json_of(
        &ok(
            &s,
            &se,
            "timeseries_arma_to_ma",
            json!({"ar": [0.5, -0.2], "ma": [0.4], "lag_max": 10}),
        )
        .await,
    );
    let ar = json_of(
        &ok(
            &s,
            &se,
            "timeseries_acf_to_ar",
            json!({"acf": [1, 0.6, 0.3, 0.1]}),
        )
        .await,
    );
    assert!(fi["values_sample"][0].is_null(), "{fi}");
    // acf2AR's rows, unpadded (R pads with zeros)
    let rows = ar["ar_matrix"].as_array().unwrap();
    let mut padded = Vec::new();
    for r in rows {
        let mut row = fs(r);
        row.resize(rows.len(), 0.0);
        padded.extend(row);
    }
    Check::new(&case)
        .vector("embed3_head", &flat(&em["matrix_sample"], 10), Tol::exact())
        .vector("window_11_20", &fs(&wi["values"]), Tol::exact())
        .vector("filter4_head", &fs(&fi["values_sample"]), Tol::coef(96))
        .vector("diffinv_head", &fs(&di["values_sample"]), Tol::coef(96))
        .vector("toeplitz_sym", &flat(&ts["matrix"], 3), Tol::exact())
        .vector("toeplitz_asym", &flat(&ta["matrix"], 3), Tol::exact())
        .vector("arma_to_ma", &fs(&ma["psi_weights"]), Tol::coef(10))
        .vector("acf2ar", &padded, Tol::coef(10))
        .vector("acf2ar_pacf", &fs(&ar["partial_acf"]), Tol::coef(10))
        .finish();
    assert_eq!(
        (em["n_rows"].as_u64(), di["n"].as_u64()),
        (
            Some(case_count(&case, "embed3_rows")),
            Some(case_count(&case, "diffinv_n"))
        )
    );
}

/// STAT-46/STAT-77: timeseries_arma_acf through the dispatch path, pinned to ARMAacf()
/// (ts_filter_ar_cpt).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_arma_acf() {
    let case = Case::load("ts_filter_ar_cpt");
    let (se, s) = session_with("ts_filter_ar_cpt");
    let ar2 = json_of(
        &ok(
            &s,
            &se,
            "timeseries_arma_acf",
            json!({"ar": [0.5, -0.3], "lag_max": 10}),
        )
        .await,
    );
    let pacf = json_of(
        &ok(
            &s,
            &se,
            "timeseries_arma_acf",
            json!({"ar": [0.5, -0.3], "lag_max": 10, "pacf": true}),
        )
        .await,
    );
    let ma2 = json_of(
        &ok(
            &s,
            &se,
            "timeseries_arma_acf",
            json!({"ma": [0.4, 0.25], "lag_max": 6}),
        )
        .await,
    );
    // The tool's PACF starts at lag 0 (1.0, as its ACF does); R's starts at lag 1.
    let pacf_values = fs(&pacf["values"]);
    assert_eq!((pacf_values[0], pacf["lags"][0].as_u64()), (1.0, Some(0)));
    // deviation TS-14: ARMA(1, 1) is not pinned.
    Check::new(&case)
        .vector("armaacf_ar2", &fs(&ar2["values"]), Tol::coef(11))
        .vector("armaacf_ar2_pacf", &pacf_values[1..], Tol::coef(11))
        .vector("armaacf_ma2", &fs(&ma2["values"]), Tol::coef(11))
        .finish();
}

/// STAT-46/STAT-77: timeseries_lag, ts_mstl and timeseries_arima_sim through the dispatch
/// path. No values are pinned (deviations TS-29, TS-26, TOOL-28).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_lag_mstl_arima_sim() {
    let (se, s) = session_with("mcp_series_t3");
    // deviation TS-29: the values are x[t + 3]; R's dplyr::lag gives lag_dplyr3_head.
    let lag = json_of(
        &ok(
            &s,
            &se,
            "timeseries_lag",
            json!({"dataset": "mcp_series_t3", "column": "z", "k": 3}),
        )
        .await,
    );
    assert_eq!(
        (lag["lag"].as_u64(), lag["n"].as_u64()),
        (Some(3), Some(93))
    );
    // deviation TS-26: the MSTL components are not pinned.
    let m = ok(
        &s,
        &se,
        "ts_mstl",
        json!({"dataset": "mcp_series_t3", "column": "y", "periods": [12]}),
    )
    .await;
    assert!(
        m.contains("Observations: 96") && m.contains("Seasonal Periods: [12]"),
        "{m}"
    );
    let (ge, gs) = session_with("mcp_groups_t3");
    let r = refused(
        &gs,
        &ge,
        "ts_mstl",
        json!({"dataset": "mcp_groups_t3", "column": "vi", "periods": [3]}),
    )
    .await;
    assert!(r.contains("missing value"), "{r}");
    // TOOL-28: sd is not asserted.
    let args = json!({"n": 50, "ar": [0.5], "seed": 42});
    let a = json_of(&ok(&s, &se, "timeseries_arima_sim", args.clone()).await);
    let b = json_of(&ok(&s, &se, "timeseries_arima_sim", args).await);
    assert_eq!(a["values_sample"], b["values_sample"]);
    assert_eq!(a["n"].as_u64(), Some(50));
    assert_eq!(fs(&a["values_sample"]).len(), 50);
}
