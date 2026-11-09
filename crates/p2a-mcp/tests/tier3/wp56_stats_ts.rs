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
