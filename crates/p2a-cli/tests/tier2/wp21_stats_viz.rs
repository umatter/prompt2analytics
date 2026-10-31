//! T2-WP21: `p2a stats` and `p2a viz` on the shared NA-aware extractor (CLI-8, CLI-9, TOOL-33 CLI part),
//! pinned to the R cases of T2-WP05/WP06/WP07 and validation/r/cases/cli_stats.R.
use crate::common::*;
use serde_json::{Value, json};

/// The first number after the first occurrence of `label` in `text`.
fn num_after(text: &str, label: &str) -> f64 {
    let i = text
        .find(label)
        .unwrap_or_else(|| panic!("{label:?} not in output:\n{text}"));
    let rest = text[i + label.len()..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|e| panic!("{label:?}: {e} in {rest:?}"))
}

/// `[lo, hi]` right after `label` (which ends with "[").
fn interval_after(text: &str, label: &str) -> (f64, f64) {
    let i = text
        .find(label)
        .unwrap_or_else(|| panic!("{label:?} not in output:\n{text}"))
        + label.len();
    let inner = &text[i..i + text[i..].find(']').unwrap()];
    let (lo, hi) = inner.split_once(',').unwrap();
    (lo.trim().parse().unwrap(), hi.trim().parse().unwrap())
}

/// Tokens after the label of the first line whose first whitespace token is `label`.
fn row<'a>(text: &'a str, label: &str) -> Vec<&'a str> {
    text.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .find(|t| t.first() == Some(&label))
        .map(|t| t[1..].to_vec())
        .unwrap_or_else(|| panic!("no row {label:?} in output:\n{text}"))
}

/// Numbers of a row, "(2.7692)" read as 2.7692.
fn nums(tokens: &[&str]) -> Vec<f64> {
    tokens
        .iter()
        .map(|t| {
            t.trim_matches(|c| c == '(' || c == ')')
                .parse()
                .unwrap_or_else(|e| panic!("{t}: {e}"))
        })
        .collect()
}

/// The `Note:` line about dropped rows, if any.
fn note(text: &str) -> Option<&str> {
    text.lines().find(|l| l.starts_with("Note: "))
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is not a number in {v}"))
}

fn num_vec(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is not an array in {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn missing(n_total: u64, n_used: u64, by_column: Value) -> Value {
    json!({"n_total": n_total, "n_used": n_used, "n_dropped": n_total - n_used, "by_column": by_column})
}

/// CLI-9: a missing cell was read as 0 (t = 3.0939 on 21 df); t.test drops it (case stat_two_sample_na).
#[test]
fn test_validate_cli_t_test_one_na() {
    let case = Case::load("stat_two_sample_na");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    let out = s.ok(&["stats", "t-test-one", d, "--col", "x"]);
    let (lo, hi) = interval_after(&out, "CI: [");
    Check::new(&case)
        .scalar("one_t", num_after(&out, "t = "), Tol::printed(4))
        .scalar("one_df", num_after(&out, "df = "), Tol::printed(1))
        .scalar("one_p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector("one_ci", &[lo, hi], Tol::printed(4))
        .scalar(
            "one_mean",
            num_after(&out, "Sample mean: "),
            Tol::printed(4),
        )
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 15 of 22 rows dropped because of missing values (x: 15)"),
        "{out}"
    );
    let j = s.json(&["stats", "t-test-one", d, "--col", "x"]);
    Check::new(&case)
        .scalar("one_t", num(&j, "t_statistic"), Tol::stat(7))
        .scalar("one_df", num(&j, "df"), Tol::exact())
        .scalar("one_p", num(&j, "p_value"), Tol::p_value(7))
        .vector(
            "one_ci",
            &[num(&j, "ci_lower"), num(&j, "ci_upper")],
            Tol::stat(7),
        )
        .scalar("one_mean", num(&j, "mean"), Tol::coef(7))
        .scalar("one_se", num(&j, "std_error"), Tol::se(7))
        .scalar("n_x", num(&j, "n"), Tol::exact())
        .finish();
    assert_eq!(j["missing"], missing(22, 7, json!({"x": 15})));
}

/// CLI-8: an Int64 column was "must be numeric" (case cli_stat_descriptive_na: t.test(xi, mu = 48, conf.level = 0.9)).
#[test]
fn test_validate_cli_t_test_one_int64() {
    let case = Case::load("cli_stat_descriptive_na");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "stats",
        "t-test-one",
        d.as_str(),
        "--col",
        "xi",
        "--mu",
        "48",
        "--conf-level",
        "0.9",
    ];
    let out = s.ok(&args);
    let (lo, hi) = interval_after(&out, "90% CI: [");
    Check::new(&case)
        .scalar("int_one_t", num_after(&out, "t = "), Tol::printed(4))
        .scalar("int_one_df", num_after(&out, "df = "), Tol::printed(1))
        .scalar("int_one_p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector("int_one_ci90", &[lo, hi], Tol::printed(4))
        .scalar(
            "int_one_mean",
            num_after(&out, "Sample mean: "),
            Tol::printed(4),
        )
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 2 of 22 rows dropped because of missing values (xi: 2)"),
        "{out}"
    );
    let j = s.json(&args);
    Check::new(&case)
        .scalar("int_one_t", num(&j, "t_statistic"), Tol::stat(20))
        .scalar("int_one_df", num(&j, "df"), Tol::exact())
        .scalar("int_one_p", num(&j, "p_value"), Tol::p_value(20))
        .vector(
            "int_one_ci90",
            &[num(&j, "ci_lower"), num(&j, "ci_upper")],
            Tol::stat(20),
        )
        .scalar("int_one_mean", num(&j, "mean"), Tol::coef(20))
        .scalar("int_one_se", num(&j, "std_error"), Tol::se(20))
        .scalar("int_one_n", num(&j, "n"), Tol::exact())
        .finish();
    assert_eq!(j["missing"], missing(22, 20, json!({"xi": 2})));
}

/// CLI-9: pairs with a missing cell entered as (x, 0) / (0, y) (t = 0.8931 on 21 df); t.test(paired) keeps complete pairs.
#[test]
fn test_validate_cli_t_test_paired_na() {
    let case = Case::load("stat_two_sample_na");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "stats",
        "t-test-paired",
        d.as_str(),
        "--col1",
        "x",
        "--col2",
        "y",
    ];
    let out = s.ok(&args);
    let (lo, hi) = interval_after(&out, "CI: [");
    Check::new(&case)
        .scalar("paired_t", num_after(&out, "t = "), Tol::printed(4))
        .scalar("paired_df", num_after(&out, "df = "), Tol::printed(1))
        .scalar("paired_p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector("paired_ci", &[lo, hi], Tol::printed(4))
        .scalar(
            "paired_est",
            num_after(&out, "Mean difference: "),
            Tol::printed(4),
        )
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 16 of 22 rows dropped because of missing values (x: 15, y: 15)"),
        "{out}"
    );
    let j = s.json(&args);
    Check::new(&case)
        .scalar("paired_t", num(&j, "t_statistic"), Tol::stat(6))
        .scalar("paired_df", num(&j, "df"), Tol::exact())
        .scalar("paired_p", num(&j, "p_value"), Tol::p_value(6))
        .vector(
            "paired_ci",
            &[num(&j, "ci_lower"), num(&j, "ci_upper")],
            Tol::stat(6),
        )
        .scalar("paired_est", num(&j, "mean_difference"), Tol::coef(6))
        .scalar("paired_se", num(&j, "std_error"), Tol::se(6))
        .scalar("n_pairs", num(&j, "n"), Tol::exact())
        .finish();
    assert_eq!(j["missing"], missing(22, 6, json!({"x": 15, "y": 15})));
}

/// CLI-8/CLI-9 and the mislabelled difference: Welch and pooled on x/y (NA per sample), Welch on Int64 xi/yi.
#[test]
fn test_validate_cli_t_test_two_na() {
    let case = Case::load("stat_two_sample_na");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    let est = case.vector("welch_est"); // [mean x, mean y]
    let out = s.ok(&["stats", "t-test-two", d, "--col1", "x", "--col2", "y"]);
    let (lo, hi) = interval_after(&out, "difference: [");
    Check::new(&case)
        .scalar("welch_t", num_after(&out, "t = "), Tol::printed(4))
        .scalar("welch_df", num_after(&out, "df = "), Tol::printed(2))
        .scalar("welch_p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector("welch_ci", &[lo, hi], Tol::printed(4))
        .vector(
            "welch_est",
            &[num_after(&out, "Means: x = "), num_after(&out, ", y = ")],
            Tol::printed(4),
        )
        .finish();
    // derived from R: the difference of t.test's two estimates (the command printed
    // mean(x) = 2.0500 as the difference)
    assert!(
        Tol::printed(4).allows(num_after(&out, "Mean difference: "), est[0] - est[1]),
        "{out}"
    );
    assert_eq!(
        note(&out),
        Some("Note: 30 of 44 rows dropped because of missing values (x: 15, y: 15)"),
        "{out}"
    );
    let j = s.json(&["stats", "t-test-two", d, "--col1", "x", "--col2", "y"]);
    Check::new(&case)
        .scalar("welch_t", num(&j, "t_statistic"), Tol::stat(14))
        .scalar("welch_df", num(&j, "df"), Tol::stat(14))
        .scalar("welch_p", num(&j, "p_value"), Tol::p_value(14))
        .vector(
            "welch_ci",
            &[num(&j, "ci_lower"), num(&j, "ci_upper")],
            Tol::stat(14),
        )
        .vector(
            "welch_est",
            &[num(&j, "mean_x"), num(&j, "mean_y")],
            Tol::coef(14),
        )
        .scalar("welch_se", num(&j, "std_error"), Tol::se(14))
        .scalar("n_x", num(&j, "n_x"), Tol::exact())
        .scalar("n_y", num(&j, "n_y"), Tol::exact())
        .finish();
    assert!(
        Tol::coef(14).allows(num(&j, "mean_difference"), est[0] - est[1]),
        "{j}"
    );
    assert_eq!(j["missing"], missing(44, 14, json!({"x": 15, "y": 15})));
    let jp = s.json(&[
        "stats",
        "t-test-two",
        d,
        "--col1",
        "x",
        "--col2",
        "y",
        "--equal-var",
    ]);
    Check::new(&case)
        .scalar("pooled_t", num(&jp, "t_statistic"), Tol::stat(14))
        .scalar("pooled_df", num(&jp, "df"), Tol::exact())
        .scalar("pooled_p", num(&jp, "p_value"), Tol::p_value(14))
        .vector(
            "pooled_ci",
            &[num(&jp, "ci_lower"), num(&jp, "ci_upper")],
            Tol::stat(14),
        )
        .scalar("pooled_se", num(&jp, "std_error"), Tol::se(14))
        .finish();
    // Int64 columns (they were "must be numeric")
    let ji = s.json(&["stats", "t-test-two", d, "--col1", "xi", "--col2", "yi"]);
    Check::new(&case)
        .scalar("int_welch_t", num(&ji, "t_statistic"), Tol::stat(38))
        .scalar("int_welch_df", num(&ji, "df"), Tol::stat(38))
        .scalar("int_welch_p", num(&ji, "p_value"), Tol::p_value(38))
        .vector(
            "int_welch_ci",
            &[num(&ji, "ci_lower"), num(&ji, "ci_upper")],
            Tol::stat(38),
        )
        .vector(
            "int_welch_est",
            &[num(&ji, "mean_x"), num(&ji, "mean_y")],
            Tol::coef(38),
        )
        .scalar("int_welch_se", num(&ji, "std_error"), Tol::se(38))
        .scalar("n_xi", num(&ji, "n_x"), Tol::exact())
        .scalar("n_yi", num(&ji, "n_y"), Tol::exact())
        .finish();
    assert_eq!(ji["missing"], missing(44, 38, json!({"xi": 2, "yi": 4})));
}

/// CLI-9: Wilcoxon read the missing cells as 0 (V = 45, p = 0.0840); wilcox.test drops them (case stat_wilcoxon_na).
#[test]
fn test_validate_cli_wilcoxon_na() {
    let case = Case::load("stat_wilcoxon_na");
    let s = temp_session();
    let d = s.load(&case);
    let paired = [
        "stats",
        "wilcoxon",
        d.as_str(),
        "--col1",
        "x",
        "--col2",
        "y",
        "--paired",
    ];
    let out = s.ok(&paired);
    Check::new(&case)
        .scalar("paired_v", num_after(&out, "W = "), Tol::exact()) // the CLI labels V as W (STAT-72, Tier 4)
        .scalar(
            "paired_p_exact",
            num_after(&out, "p-value = "),
            Tol::printed(6),
        )
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 2 of 10 rows dropped because of missing values (x: 1, y: 1)"),
        "{out}"
    );
    let j = s.json(&paired);
    Check::new(&case)
        .scalar("paired_v", num(&j, "statistic"), Tol::exact())
        .scalar("paired_p_exact", num(&j, "p_value"), Tol::p_value(8)) // default config: exact for 8 pairs without ties
        .finish();
    assert_eq!(j["missing"], missing(10, 8, json!({"x": 1, "y": 1})));
    let r = s.json(&[
        "stats",
        "wilcoxon",
        d.as_str(),
        "--col1",
        "x",
        "--col2",
        "y",
    ]);
    // p2a's rank-sum statistic is the rank sum R1 (R's W = R1 - n_x(n_x+1)/2, STAT-72); R1 itself
    // is the case key. Not pinned: the exact rank-sum p (no R value in a Tier 2 case; the exact
    // path is STAT-73, Tier 4).
    Check::new(&case)
        .scalar("ranksum_r1", num(&r, "statistic"), Tol::exact())
        .finish();
    assert_eq!(r["missing"], missing(20, 18, json!({"x": 1, "y": 1}))); // each sample on its own
}

/// CLI-9/CLI-8: fivenum/IQR/MAD drop missing cells (fivenum; IQR/mad na.rm = TRUE) and accept
/// Int64 (case cli_stat_descriptive_na).
#[test]
fn test_validate_cli_descriptive_na() {
    let case = Case::load("cli_stat_descriptive_na");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    let out = s.ok(&["stats", "fivenum", d, "--col", "x"]);
    let five: Vec<f64> = ["Min:", "Q1:", "Median:", "Q3:", "Max:"]
        .iter()
        .map(|l| nums(&row(&out, l))[0])
        .collect();
    Check::new(&case)
        .vector("fivenum_x", &five, Tol::printed(4))
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 15 of 22 rows dropped because of missing values (x: 15)"),
        "{out}"
    );
    for (col, n, k) in [("x", 7_u64, 15_u64), ("xi", 20, 2)] {
        let mut by = serde_json::Map::new();
        by.insert(col.to_string(), k.into());
        let want = missing(22, n, Value::Object(by));
        let j = s.json(&["stats", "fivenum", d, "--col", col]);
        let got = ["min", "q1", "median", "q3", "max"].map(|key| num(&j, key));
        Check::new(&case)
            .vector(&format!("fivenum_{col}"), &got, Tol::coef(n as usize))
            .scalar(&format!("n_{col}"), num(&j, "n"), Tol::exact())
            .finish();
        assert_eq!(j["missing"], want);
        let iqr = s.json(&["stats", "iqr", d, "--col", col]);
        let mad = s.json(&["stats", "mad", d, "--col", col]);
        Check::new(&case)
            .scalar(
                &format!("iqr_{col}"),
                num(&iqr, "iqr"),
                Tol::coef(n as usize),
            )
            .scalar(
                &format!("mad_{col}"),
                num(&mad, "mad"),
                Tol::coef(n as usize),
            )
            .finish();
        assert_eq!((&iqr["missing"], &mad["missing"]), (&want, &want));
    }
    let text = s.ok(&["stats", "mad", d, "--col", "xi"]);
    Check::new(&case)
        .scalar("mad_xi", num_after(&text, "MAD(xi): "), Tol::printed(4))
        .finish();
    assert_eq!(
        note(&text),
        Some("Note: 2 of 22 rows dropped because of missing values (xi: 2)")
    );
}

/// The values of the `Lag k: v` lines, in order.
fn lag_values(text: &str) -> Vec<f64> {
    text.lines()
        .filter(|l| l.trim_start().starts_with("Lag "))
        .map(|l| {
            l.split_once(':')
                .unwrap()
                .1
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

/// CLI-8: an Int64 series was "must be numeric" (case cli_stat_acf_int: pacf(y, lag.max = 5)).
#[test]
fn test_validate_cli_acf_pacf_int64() {
    let case = Case::load("cli_stat_acf_int");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    let p = s.json(&["stats", "pacf", d, "--col", "y", "--lag-max", "5"]);
    let partial = s.json(&[
        "stats",
        "acf",
        d,
        "--col",
        "y",
        "--lag-max",
        "5",
        "--partial",
    ]);
    Check::new(&case)
        .vector("pacf", &num_vec(&p, "values"), Tol::stat(24))
        .scalar("n", num(&p, "n_obs"), Tol::exact())
        .vector("pacf", &num_vec(&partial, "values"), Tol::stat(24))
        .finish();
    assert_eq!(
        (p["lags"].clone(), partial["lags"].clone()),
        (json!([1, 2, 3, 4, 5]), json!([1, 2, 3, 4, 5]))
    );
    // Not pinned: p["confidence_bound"] is 1.96 / sqrt(n); R's acf object has no bound
    // (plot.acf draws qnorm(0.975) / sqrt(n)).
    let text = s.ok(&["stats", "pacf", d, "--col", "y", "--lag-max", "5"]);
    Check::new(&case)
        .vector("pacf", &lag_values(&text), Tol::printed(4))
        .finish();
    // stats acf (not partial): the CLI passes adjusted = true (n - k denominators, CLI-22,
    // Tier 4), so its values are R's acf times n / (n - k), derived from the case key `acf`;
    // lag 0 is 1.
    let a = s.json(&["stats", "acf", d, "--col", "y", "--lag-max", "5"]);
    let values = num_vec(&a, "values");
    assert_eq!(
        (a["lags"].clone(), values[0]),
        (json!([0, 1, 2, 3, 4, 5]), 1.0)
    );
    let r_acf = case.vector("acf");
    for k in 1..=5 {
        let want = r_acf[k - 1] * 24.0 / (24.0 - k as f64);
        assert!(
            Tol::stat(24).allows(values[k], want),
            "lag {k}: rust={} derived R={want}",
            values[k]
        );
    }
}

/// `data load validation/<rel> --name <file stem>` for tests that compare no R value (and so
/// load no case); returns the name.
fn load_dataset(s: &TempSession, rel: &str) -> String {
    let path = repo_path(&format!("validation/{rel}"));
    let name = path.file_stem().unwrap().to_str().unwrap().to_string();
    s.load_as(&path, &name, &[]);
    name
}

/// CLI-9: a missing value in the series was read as 0; R's acf/pacf refuse it
/// (na.action = na.fail, as case cli_stat_acf_int checks in R). A refusal test: it compares no
/// R number and is not registered.
#[test]
fn test_cli_acf_refuses_missing() {
    let s = temp_session();
    let d = load_dataset(&s, "datasets/generated/friedman_quade.csv");
    for args in [
        &[
            "stats",
            "acf",
            d.as_str(),
            "--col",
            "y_na",
            "--lag-max",
            "5",
        ][..],
        &[
            "stats",
            "acf",
            d.as_str(),
            "--col",
            "y_na",
            "--lag-max",
            "5",
            "--partial",
        ][..],
        &[
            "stats",
            "pacf",
            d.as_str(),
            "--col",
            "y_na",
            "--lag-max",
            "5",
        ][..],
    ] {
        let e = s.err(args);
        assert!(
            e.contains("Column 'y_na' contains 1 missing value(s)") && e.contains("na.fail"),
            "{args:?}: {e}"
        );
    }
}

/// Observed counts and expected counts of the rows `labels` of a "obs (exp)" table.
fn obs_exp(text: &str, labels: &[&str]) -> (Vec<f64>, Vec<f64>) {
    let cells: Vec<f64> = labels.iter().flat_map(|l| nums(&row(text, l))).collect();
    let obs = cells.iter().step_by(2).copied().collect();
    let exp = cells.iter().skip(1).step_by(2).copied().collect();
    (obs, exp)
}

/// TOOL-33: level labels, R's small-expected warning, the Yates line and dropped rows
/// (cases stat_chisq_small_expected, stat_contingency_na_codes).
#[test]
fn test_validate_cli_chisq_indep_labels_warning() {
    let small = Case::load("stat_chisq_small_expected");
    let codes = Case::load("stat_contingency_na_codes");
    let s = temp_session();
    let (d1, d2) = (s.load(&small), s.load(&codes));
    let a1 = [
        "stats",
        "chisq-indep",
        d1.as_str(),
        "--row-var",
        "x",
        "--col-var",
        "y",
    ];
    let out = s.ok(&a1);
    let (obs, exp) = obs_exp(&out, &["p", "q", "r"]);
    Check::new(&small)
        .scalar("chisq", num_after(&out, "X-squared = "), Tol::printed(4))
        .scalar("df", num_after(&out, "df = "), Tol::exact())
        .scalar("p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector("observed", &obs, Tol::exact())
        .vector("expected", &exp, Tol::printed(4))
        .finish();
    assert!(
        out.lines()
            .any(|l| l.split_whitespace().collect::<Vec<_>>() == ["s", "t"]),
        "header row: {out}"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("Warning: Chi-squared approximation may be incorrect")),
        "{out}"
    );
    assert!(
        !out.contains("Continuity correction") && note(&out).is_none(),
        "{out}"
    );
    let j = s.json(&a1);
    assert_eq!(
        (j["row_levels"].clone(), j["col_levels"].clone()),
        (json!(["p", "q", "r"]), json!(["s", "t"]))
    );
    Check::new(&small)
        .scalar("chisq", num(&j, "chi_squared"), Tol::stat(13))
        .scalar("df", num(&j, "df"), Tol::exact())
        .scalar("p", num(&j, "p_value"), Tol::p_value(13))
        .vector("observed", &num_vec(&j, "observed"), Tol::exact())
        .vector("expected", &num_vec(&j, "expected"), Tol::stat(13))
        .vector("residuals", &num_vec(&j, "residuals"), Tol::stat(13))
        .finish();
    assert!(
        j["warnings"][0]
            .as_str()
            .unwrap()
            .starts_with("Chi-squared approximation may be incorrect"),
        "{j}"
    );
    assert_eq!(
        (j["yates_correction"].clone(), j.get("missing")),
        (json!(false), None)
    );
    // 2 x 2 after dropping the 3 rows with a missing h: Yates, labels a/b x u/v, note
    let a2 = [
        "stats",
        "chisq-indep",
        d2.as_str(),
        "--row-var",
        "g",
        "--col-var",
        "h",
    ];
    let out2 = s.ok(&a2);
    let (obs2, exp2) = obs_exp(&out2, &["a", "b"]);
    Check::new(&codes)
        .scalar(
            "gh_chisq_yates",
            num_after(&out2, "X-squared = "),
            Tol::printed(4),
        )
        .scalar("gh_df", num_after(&out2, "df = "), Tol::exact())
        .scalar(
            "gh_p_yates",
            num_after(&out2, "p-value = "),
            Tol::printed(6),
        )
        .vector("gh_observed", &obs2, Tol::exact())
        .vector("gh_expected", &exp2, Tol::printed(4))
        .finish();
    assert!(out2.contains("Continuity correction: Yates"), "{out2}");
    assert_eq!(
        note(&out2),
        Some("Note: 3 of 12 rows dropped because of missing values (h: 3)"),
        "{out2}"
    );
    let j2 = s.json(&a2);
    Check::new(&codes)
        .scalar("gh_chisq_yates", num(&j2, "chi_squared"), Tol::stat(9))
        .scalar("gh_p_yates", num(&j2, "p_value"), Tol::p_value(9))
        .vector("gh_observed", &num_vec(&j2, "observed"), Tol::exact())
        .vector("gh_expected", &num_vec(&j2, "expected"), Tol::stat(9))
        .vector("gh_residuals", &num_vec(&j2, "residuals"), Tol::stat(9))
        .scalar("gh_n", num(&j2, "n"), Tol::exact())
        .finish();
    assert_eq!(
        (j2["row_levels"].clone(), j2["col_levels"].clone()),
        (json!(["a", "b"]), json!(["u", "v"]))
    );
    assert_eq!(j2["yates_correction"], json!(true));
    assert_eq!(j2["missing"], missing(12, 9, json!({"h": 3})));
}

/// TOOL-33 (GOF): categories in R's order with observed and expected counts; the table is left
/// out for many categories.
#[test]
fn test_validate_cli_chisq_gof_categories() {
    let case = Case::load("stat_chisq_gof");
    let s = temp_session();
    let d = s.load(&case);
    let args = ["stats", "chisq-gof", d.as_str(), "--observed", "cat"];
    let out = s.ok(&args);
    let n = case.scalar("n_used");
    let rows: Vec<Vec<f64>> = ["a", "b", "c"]
        .iter()
        .map(|l| nums(&row(&out, l)))
        .collect(); // [observed, expected]
    Check::new(&case)
        .scalar(
            "uniform_chisq",
            num_after(&out, "X-squared = "),
            Tol::printed(4),
        )
        .scalar("uniform_df", num_after(&out, "df = "), Tol::exact())
        .scalar("uniform_p", num_after(&out, "p-value = "), Tol::printed(6))
        .vector(
            "observed",
            &rows.iter().map(|r| r[0]).collect::<Vec<_>>(),
            Tol::exact(),
        )
        .finish();
    // uniform expected counts, derived from R's n_used: n / 3
    assert!(
        rows.iter().all(|r| Tol::printed(4).allows(r[1], n / 3.0)),
        "{out}"
    );
    assert!(!out.contains("Warning:"), "{out}");
    assert_eq!(
        note(&out),
        Some("Note: 2 of 52 rows dropped because of missing values (cat: 2)"),
        "{out}"
    );
    let j = s.json(&args);
    assert_eq!(j["categories"], json!(["a", "b", "c"]));
    Check::new(&case)
        .scalar("uniform_chisq", num(&j, "chi_squared"), Tol::stat(50))
        .scalar("uniform_df", num(&j, "df"), Tol::exact())
        .scalar("uniform_p", num(&j, "p_value"), Tol::p_value(50))
        .vector("observed", &num_vec(&j, "observed"), Tol::exact())
        .scalar("n_used", num(&j, "n"), Tol::exact())
        .finish();
    assert!(
        num_vec(&j, "expected")
            .iter()
            .all(|&e| Tol::stat(50).allows(e, n / 3.0)),
        "{j}"
    );
    assert_eq!(j["missing"], missing(52, 50, json!({"cat": 2})));
    // 52 categories (the id column): the table is left out of the text, kept in the JSON
    let many = s.ok(&["stats", "chisq-gof", d.as_str(), "--observed", "id"]);
    assert!(
        many.contains("Table not shown: 52 categories (more than 50); use -F json for the counts."),
        "{many}"
    );
    assert!(
        many.lines()
            .any(|l| l.starts_with("Warning: Chi-squared approximation may be incorrect")),
        "{many}"
    );
    let jm = s.json(&["stats", "chisq-gof", d.as_str(), "--observed", "id"]);
    assert_eq!(jm["categories"].as_array().unwrap().len(), 52);
}

/// TOOL-33 (Fisher): R's estimate is the conditional MLE; labelled counts; exact limits; dropped
/// rows (cases stat_fisher_rows_2000, stat_contingency_na_codes; tight values = fisher.test's
/// equations re-solved with uniroot(tol = 1e-15)).
#[test]
fn test_validate_cli_fisher_conditional_mle() {
    const TIGHT: Tol = Tol::new(1e-9, 1e-12);
    let big = Case::load("stat_fisher_rows_2000");
    let codes = Case::load("stat_contingency_na_codes");
    let s = temp_session();
    let (d1, d2) = (s.load(&big), s.load(&codes));
    let a1 = [
        "stats",
        "fisher",
        d1.as_str(),
        "--row-var",
        "treat",
        "--col-var",
        "outcome",
    ];
    let out = s.ok(&a1);
    let (lo, hi) = interval_after(&out, "95% CI: [");
    Check::new(&big)
        .scalar(
            "p_two_sided",
            num_after(&out, "p-value = "),
            Tol::printed(6),
        )
        .scalar(
            "estimate_cond_mle_tight",
            num_after(&out, "Odds ratio (conditional MLE) = "),
            Tol::printed(4),
        )
        .scalar("ci_lower_two_sided_tight", lo, Tol::printed(4))
        .scalar("ci_upper_two_sided_tight", hi, Tol::printed(4))
        .scalar(
            "sample_odds_ratio",
            num_after(&out, "Sample odds ratio = "),
            Tol::printed(4),
        )
        .finish();
    assert_eq!(
        (nums(&row(&out, "control")), nums(&row(&out, "treated"))),
        (vec![520.0, 480.0], vec![480.0, 520.0])
    );
    assert!(note(&out).is_none(), "{out}");
    let j = s.json(&a1);
    Check::new(&big)
        .scalar("p_two_sided", num(&j, "p_value"), Tol::p_value(2000))
        .scalar("estimate_cond_mle_tight", num(&j, "conditional_mle"), TIGHT)
        .scalar("ci_lower_two_sided_tight", num(&j, "ci_lower"), TIGHT)
        .scalar("ci_upper_two_sided_tight", num(&j, "ci_upper"), TIGHT)
        .scalar(
            "sample_odds_ratio",
            num(&j, "sample_odds_ratio"),
            Tol::coef(2000),
        )
        .scalar("sample_odds_ratio", num(&j, "odds_ratio"), Tol::coef(2000)) // unchanged meaning of the old key
        .finish();
    // the case asserts table(treat, outcome) == [520, 480, 480, 520] (stopifnot); levels in R
    // factor order
    assert_eq!(
        (
            j["table"].clone(),
            j["row_levels"].clone(),
            j["col_levels"].clone()
        ),
        (
            json!([520.0, 480.0, 480.0, 520.0]),
            json!(["control", "treated"]),
            json!(["no", "yes"])
        )
    );
    // numeric codes 2 < 10 (STAT-50 order) with one missing code
    let a2 = [
        "stats",
        "fisher",
        d2.as_str(),
        "--row-var",
        "code",
        "--col-var",
        "out",
    ];
    let out2 = s.ok(&a2);
    let (lo2, hi2) = interval_after(&out2, "95% CI: [");
    Check::new(&codes)
        .scalar(
            "code_out_fisher_p",
            num_after(&out2, "p-value = "),
            Tol::printed(6),
        )
        .scalar(
            "code_out_fisher_estimate_tight",
            num_after(&out2, "Odds ratio (conditional MLE) = "),
            Tol::printed(4),
        )
        .scalar("code_out_fisher_ci_lower_tight", lo2, Tol::printed(4))
        .scalar("code_out_fisher_ci_upper_tight", hi2, Tol::printed(4))
        .scalar(
            "code_out_sample_or",
            num_after(&out2, "Sample odds ratio = "),
            Tol::printed(4),
        )
        .vector(
            "code_out_table",
            &[nums(&row(&out2, "2")), nums(&row(&out2, "10"))].concat(),
            Tol::exact(),
        )
        .finish();
    assert_eq!(
        note(&out2),
        Some("Note: 1 of 12 rows dropped because of missing values (code: 1)"),
        "{out2}"
    );
    let j2 = s.json(&a2);
    Check::new(&codes)
        .scalar("code_out_fisher_p", num(&j2, "p_value"), Tol::p_value(11))
        .scalar(
            "code_out_fisher_estimate_tight",
            num(&j2, "conditional_mle"),
            TIGHT,
        )
        .scalar(
            "code_out_fisher_ci_lower_tight",
            num(&j2, "ci_lower"),
            TIGHT,
        )
        .scalar(
            "code_out_fisher_ci_upper_tight",
            num(&j2, "ci_upper"),
            TIGHT,
        )
        .scalar(
            "code_out_sample_or",
            num(&j2, "sample_odds_ratio"),
            Tol::coef(11),
        )
        .vector("code_out_table", &num_vec(&j2, "table"), Tol::exact())
        .finish();
    assert_eq!(
        (j2["row_levels"].clone(), j2["col_levels"].clone()),
        (json!(["2", "10"]), json!(["0", "1"]))
    );
    assert_eq!(j2["missing"], missing(12, 11, json!({"code": 1})));
}
