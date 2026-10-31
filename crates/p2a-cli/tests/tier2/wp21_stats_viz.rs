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
        Some(
            "Note: 30 of 44 values dropped because of missing values (two samples of 22; x: \
             15, y: 15)"
        ),
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
    let rt = s.ok(&[
        "stats",
        "wilcoxon",
        d.as_str(),
        "--col1",
        "x",
        "--col2",
        "y",
    ]);
    assert_eq!(
        note(&rt),
        Some(
            "Note: 2 of 20 values dropped because of missing values (two samples of 10; x: 1, \
             y: 1)"
        ),
        "{rt}"
    );
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

/// One-way ANOVA (aov drops rows missing y or g) and Kruskal-Wallis (labels missing) (case
/// stat_oneway_na_labels).
#[test]
fn test_validate_cli_anova_kruskal_na() {
    let case = Case::load("stat_oneway_na_labels");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    let out = s.ok(&["stats", "anova", d, "--response", "y", "--factor", "g"]);
    let (b, w) = (nums(&row(&out, "Between")), nums(&row(&out, "Within"))); // SS, DF, MS[, F, p]
    Check::new(&case)
        .scalar("aov_ss_between", b[0], Tol::printed(4))
        .scalar("aov_df_between", b[1], Tol::exact())
        .scalar("aov_f", b[3], Tol::printed(4))
        .scalar("aov_p", b[4], Tol::printed(6))
        .scalar("aov_ss_within", w[0], Tol::printed(4))
        .scalar("aov_df_within", w[1], Tol::exact())
        .scalar("aov_n_used", num_after(&out, "N = "), Tol::exact())
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 4 of 9 rows dropped because of missing values (y: 3, g: 2)"),
        "{out}"
    );
    let j = s.json(&["stats", "anova", d, "--response", "y", "--factor", "g"]);
    Check::new(&case)
        .scalar("aov_f", num(&j, "f_statistic"), Tol::stat(5))
        .scalar("aov_p", num(&j, "p_value"), Tol::p_value(5))
        .scalar("aov_ss_between", num(&j, "ss_between"), Tol::stat(5))
        .scalar("aov_ss_within", num(&j, "ss_within"), Tol::stat(5))
        .scalar("aov_df_between", num(&j, "df_between"), Tol::exact())
        .scalar("aov_df_within", num(&j, "df_within"), Tol::exact())
        .scalar("aov_n_used", num(&j, "n_obs"), Tol::exact())
        .finish();
    assert_eq!(j["missing"], missing(9, 5, json!({"y": 3, "g": 2})));
    let k = s.ok(&["stats", "kruskal", d, "--col", "v9", "--group", "g9"]);
    Check::new(&case)
        .scalar("kw_chisq", num_after(&k, "H = "), Tol::printed(4))
        .scalar("kw_df", num_after(&k, "df = "), Tol::exact())
        .scalar("kw_p", num_after(&k, "p-value = "), Tol::printed(6))
        .finish();
    assert_eq!(
        note(&k),
        Some("Note: 3 of 9 rows dropped because of missing values (g9: 3)"),
        "{k}"
    );
    let kj = s.json(&["stats", "kruskal", d, "--col", "v9", "--group", "g9"]);
    Check::new(&case)
        .scalar("kw_chisq", num(&kj, "statistic"), Tol::stat(6))
        .scalar("kw_p", num(&kj, "p_value"), Tol::p_value(6))
        .finish();
    assert_eq!(kj["missing"], missing(9, 6, json!({"g9": 3})));
}

/// Tukey HSD after aov with two missing responses (case stat_pairwise_mahalanobis_na: group means
/// and pooled SD).
#[test]
fn test_validate_cli_tukey_na() {
    let case = Case::load("stat_pairwise_mahalanobis_na");
    let s = temp_session();
    let d = s.load(&case);
    let args = [
        "stats",
        "tukey",
        d.as_str(),
        "--response",
        "y",
        "--factor",
        "g",
    ];
    let out = s.ok(&args);
    let psd = case.scalar("pairwise_pooled_sd");
    // derived: MSE = pooled SD^2
    assert!(
        Tol::printed(4).allows(num_after(&out, "MSE = "), psd * psd),
        "{out}"
    );
    Check::new(&case)
        .scalar("pairwise_df", num_after(&out, ", df = "), Tol::exact())
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 2 of 30 rows dropped because of missing values (y: 2)"),
        "{out}"
    );
    let j = s.json(&args);
    assert!(Tol::stat(28).allows(num(&j, "mse"), psd * psd), "{j}");
    Check::new(&case)
        .scalar("pairwise_df", num(&j, "df"), Tol::exact())
        .finish();
    for c in j["comparisons"].as_array().unwrap() {
        let (g1, g2) = (c["group1"].as_str().unwrap(), c["group2"].as_str().unwrap());
        // p2a: diff = mean(group1) - mean(group2)
        let want = case.named("pairwise_group_mean", g1) - case.named("pairwise_group_mean", g2);
        assert!(Tol::coef(28).allows(num(c, "diff"), want), "{g1}-{g2}: {c}");
    }
    assert_eq!(j["comparisons"].as_array().unwrap().len(), 3);
    // Not pinned: p adj and the intervals (R's TukeyHSD values are in no Tier 2 case).
    assert_eq!(j["missing"], missing(30, 28, json!({"y": 2})));
}

/// Friedman: a missing response drops its block (case stat_friedman_quade).
#[test]
fn test_validate_cli_friedman_na() {
    let case = Case::load("stat_friedman_quade");
    let s = temp_session();
    let d = s.load(&case);
    let na = [
        "stats",
        "friedman",
        d.as_str(),
        "--value",
        "y_na",
        "--group",
        "treat",
        "--block",
        "block",
    ];
    let out = s.ok(&na);
    Check::new(&case)
        .scalar(
            "na_friedman_q",
            num_after(&out, "Chi-squared = "),
            Tol::printed(4),
        )
        .scalar(
            "na_friedman_p",
            num_after(&out, "p-value = "),
            Tol::printed(6),
        )
        .scalar("na_n_blocks", num_after(&out, "Blocks: "), Tol::exact())
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 3 of 24 rows dropped because of missing values (y_na: 1)"),
        "{out}"
    );
    let j = s.json(&na);
    Check::new(&case)
        .scalar("na_friedman_q", num(&j, "statistic"), Tol::stat(21))
        .scalar("na_friedman_p", num(&j, "p_value"), Tol::p_value(21))
        .scalar("na_n_blocks", num(&j, "n_blocks"), Tol::exact())
        .finish();
    assert_eq!(j["missing"], missing(24, 21, json!({"y_na": 1})));
    let full = s.ok(&[
        "stats",
        "friedman",
        d.as_str(),
        "--value",
        "y",
        "--group",
        "treat",
        "--block",
        "block",
    ]);
    Check::new(&case)
        .scalar(
            "friedman_q",
            num_after(&full, "Chi-squared = "),
            Tol::printed(4),
        )
        .scalar("friedman_df", num_after(&full, "df = "), Tol::exact())
        .scalar(
            "friedman_p",
            num_after(&full, "p-value = "),
            Tol::printed(6),
        )
        .finish();
    assert!(note(&full).is_none(), "nothing dropped, no note: {full}");
}

/// Shapiro-Wilk on a padded Int64 column, Bartlett value ~ group, KS on two Int64 columns (cases
/// stat_shapiro_wilk, stat_two_sample_na).
#[test]
fn test_validate_cli_shapiro_bartlett_ks_na() {
    let sw = Case::load("stat_shapiro_wilk");
    let two = Case::load("stat_two_sample_na");
    let s = temp_session();
    let (d1, d2) = (s.load(&sw), s.load(&two));
    let out = s.ok(&["stats", "shapiro", d1.as_str(), "--col", "n25_int"]);
    Check::new(&sw)
        .scalar("n25_int_w", num_after(&out, "W = "), Tol::printed(6))
        .scalar("n25_int_p", num_after(&out, "p-value = "), Tol::printed(6))
        .scalar("n25_int_n", num_after(&out, "n = "), Tol::exact())
        .finish();
    assert_eq!(
        note(&out),
        Some("Note: 4975 of 5000 rows dropped because of missing values (n25_int: 4975)"),
        "{out}"
    );
    let j = s.json(&["stats", "shapiro", d1.as_str(), "--col", "n25_int"]);
    Check::new(&sw)
        .scalar("n25_int_w", num(&j, "W"), Tol::stat(25))
        .scalar("n25_int_p", num(&j, "p_value"), Tol::p_value(25))
        .finish();
    assert_eq!(j["missing"], missing(5000, 25, json!({"n25_int": 4975})));
    let b = s.ok(&[
        "stats",
        "bartlett",
        d2.as_str(),
        "--col",
        "xi",
        "--group",
        "g",
    ]);
    Check::new(&two)
        .scalar(
            "bartlett_k2",
            num_after(&b, "K-squared = "),
            Tol::printed(4),
        )
        .scalar("bartlett_df", num_after(&b, "df = "), Tol::exact())
        .scalar("bartlett_p", num_after(&b, "p-value = "), Tol::printed(6))
        .finish();
    assert_eq!(
        note(&b),
        Some("Note: 2 of 22 rows dropped because of missing values (xi: 2, g: 1)"),
        "{b}"
    );
    let bj = s.json(&[
        "stats",
        "bartlett",
        d2.as_str(),
        "--col",
        "xi",
        "--group",
        "g",
    ]);
    Check::new(&two)
        .scalar("bartlett_k2", num(&bj, "statistic"), Tol::stat(20))
        .scalar("bartlett_p", num(&bj, "p_value"), Tol::p_value(20))
        .finish();
    assert_eq!(bj["missing"], missing(22, 20, json!({"xi": 2, "g": 1})));
    let ks = s.json(&["stats", "ks-test", d2.as_str(), "--x", "xi", "--y", "yi"]);
    // Not pinned: the KS p-value (asymptotic in p2a, exact with ties in R: STAT-27, Tier 4,
    // recorded in stats_onesample.json).
    Check::new(&two)
        .scalar("ks_d", num(&ks, "statistic"), Tol::stat(38))
        .scalar("n_xi", num(&ks, "n"), Tol::exact())
        .scalar("n_yi", num(&ks, "n_2"), Tol::exact())
        .finish();
    assert_eq!(ks["missing"], missing(44, 38, json!({"xi": 2, "yi": 4})));
    let kst = s.ok(&["stats", "ks-test", d2.as_str(), "--x", "xi", "--y", "yi"]);
    assert_eq!(
        note(&kst),
        Some(
            "Note: 6 of 44 values dropped because of missing values (two samples of 22; xi: \
             2, yi: 4)"
        ),
        "{kst}"
    );
}

/// STAT-4's least-squares two-way ANOVA through the CLI: Type I (aov) and Type II (car::Anova),
/// with the dropped row (case stat_anova_two_way_unbalanced).
#[test]
fn test_validate_cli_anova_two_ss_type() {
    let case = Case::load("stat_anova_two_way_unbalanced");
    let s = temp_session();
    let d = s.load(&case);
    let base = [
        "stats",
        "anova-two",
        d.as_str(),
        "--response",
        "y",
        "--factor-a",
        "fa",
        "--factor-b",
        "fb",
        "--interaction",
    ];
    let out = s.ok(&base);
    // SS DF MS F p | Error: SS DF MS | Total: SS DF
    let (a, b, ab, e, t) = (
        nums(&row(&out, "fa")),
        nums(&row(&out, "fb")),
        nums(&row(&out, "Interaction")),
        nums(&row(&out, "Error")),
        nums(&row(&out, "Total")),
    );
    Check::new(&case)
        .scalar("int_fa_ss", a[0], Tol::printed(4))
        .scalar("int_fa_df", a[1], Tol::exact())
        .scalar("int_fa_f", a[3], Tol::printed(4))
        .scalar("int_fa_p", a[4], Tol::printed(6))
        .scalar("int_fb_ss", b[0], Tol::printed(4))
        .scalar("int_fb_f", b[3], Tol::printed(4))
        .scalar("int_fb_p", b[4], Tol::printed(6))
        .scalar("int_fa_x_fb_ss", ab[0], Tol::printed(4))
        .scalar("int_fa_x_fb_df", ab[1], Tol::exact())
        .scalar("int_fa_x_fb_f", ab[3], Tol::printed(4))
        .scalar("int_fa_x_fb_p", ab[4], Tol::printed(6))
        .scalar("int_resid_ss", e[0], Tol::printed(4))
        .scalar("int_resid_df", e[1], Tol::exact())
        .scalar("n_used", num_after(&out, "N = "), Tol::exact())
        .finish();
    // derived: Type I sums of squares add up to the total SS on the 12 kept rows (df 11)
    let total = ["int_fa_ss", "int_fb_ss", "int_fa_x_fb_ss", "int_resid_ss"]
        .iter()
        .map(|k| case.scalar(k))
        .sum::<f64>();
    assert!(Tol::printed(4).allows(t[0], total) && t[1] == 11.0, "{out}");
    assert!(
        out.contains("Sums of squares: Type I (sequential, as R aov)"),
        "{out}"
    );
    assert_eq!(
        note(&out),
        Some("Note: 1 of 13 rows dropped because of missing values (y: 1, fa: 1)"),
        "{out}"
    );
    let j = s.json(&base);
    Check::new(&case)
        .scalar("int_fa_ss", num(&j, "ss_a"), Tol::stat(12))
        .scalar("int_fa_df", num(&j, "df_a"), Tol::exact())
        .scalar("int_fa_f", num(&j, "f_a"), Tol::stat(12))
        .scalar("int_fa_p", num(&j, "p_a"), Tol::p_value(12))
        .scalar("int_fb_ss", num(&j, "ss_b"), Tol::stat(12))
        .scalar("int_fb_p", num(&j, "p_b"), Tol::p_value(12))
        .scalar("int_fa_x_fb_ss", num(&j, "ss_ab"), Tol::stat(12))
        .scalar("int_fa_x_fb_df", num(&j, "df_ab"), Tol::exact())
        .scalar("int_fa_x_fb_p", num(&j, "p_ab"), Tol::p_value(12))
        .scalar("int_resid_ss", num(&j, "ss_error"), Tol::stat(12))
        .scalar("int_resid_df", num(&j, "df_error"), Tol::exact())
        .scalar("n_used", num(&j, "n_obs"), Tol::exact())
        .finish();
    assert_eq!(
        (j["ss_type"].clone(), j["missing"].clone()),
        (json!("I"), missing(13, 12, json!({"y": 1, "fa": 1})))
    );
    let ii: Vec<&str> = base.iter().copied().chain(["--ss-type", "II"]).collect();
    let out2 = s.ok(&ii);
    let a2 = nums(&row(&out2, "fa"));
    Check::new(&case)
        .scalar("type2_fa_ss", a2[0], Tol::printed(4))
        .scalar("type2_fa_f", a2[3], Tol::printed(4))
        .scalar("type2_fa_p", a2[4], Tol::printed(6))
        .finish();
    assert!(
        out2.contains("Sums of squares: Type II (as car::Anova(type = 2))"),
        "{out2}"
    );
    let j2 = s.json(&ii);
    Check::new(&case)
        .scalar("type2_fa_ss", num(&j2, "ss_a"), Tol::stat(12))
        .scalar("type2_fa_f", num(&j2, "f_a"), Tol::stat(12))
        .scalar("type2_fa_p", num(&j2, "p_a"), Tol::p_value(12))
        .scalar("type2_fb_ss", num(&j2, "ss_b"), Tol::stat(12))
        .scalar("type2_int_ss", num(&j2, "ss_ab"), Tol::stat(12))
        .scalar("type2_resid_ss", num(&j2, "ss_error"), Tol::stat(12))
        .scalar("type2_resid_df", num(&j2, "df_error"), Tol::exact())
        .finish();
    assert_eq!(j2["ss_type"], json!("II"));
    let bad: Vec<&str> = base.iter().copied().chain(["--ss-type", "IV"]).collect();
    assert!(s.err(&bad).contains("Unknown sum-of-squares type 'IV'"));
    // Not pinned: Type III is refused when the model has aliased coefficients (the empty
    // (C, L) cell of y3 ~ fa3 * fb3), as car::Anova(type = 3) is.
    let iii = [
        "stats",
        "anova-two",
        d.as_str(),
        "--response",
        "y3",
        "--factor-a",
        "fa3",
        "--factor-b",
        "fb3",
        "--interaction",
        "--ss-type",
        "III",
    ];
    let e3 = s.err(&iii);
    assert!(
        e3.contains("Two-way ANOVA failed") && e3.contains("aliased"),
        "{e3}"
    );
}

fn png(s: &TempSession, name: &str) -> String {
    s.dir().join(name).to_str().unwrap().to_string()
}

fn is_png(path: &str) -> bool {
    std::fs::read(path)
        .map(|b| b.starts_with(b"\x89PNG"))
        .unwrap_or(false)
}

/// CLI-9: scatter, line and heatmap paired the columns with zeros for missing cells; now complete
/// rows only (cases stat_two_sample_na: n_pairs; cli_stat_descriptive_na: n_complete).
#[test]
fn test_validate_na_cli_viz_xy_plots() {
    let case = Case::load("stat_two_sample_na");
    let desc = Case::load("cli_stat_descriptive_na");
    let s = temp_session();
    let d = s.load(&case);
    let d = d.as_str();
    for kind in ["scatter", "line"] {
        let f = png(&s, &format!("{kind}.png"));
        let out = s.ok(&["viz", kind, d, "-x", "x", "-y", "y", "-f", f.as_str()]);
        assert_eq!(
            note(&out),
            Some("Note: 16 of 22 rows dropped because of missing values (x: 15, y: 15)"),
            "{kind}: {out}"
        );
        assert!(is_png(&f), "{kind}");
        let j = s.json(&["viz", kind, d, "-x", "x", "-y", "y", "-f", f.as_str()]);
        Check::new(&case)
            .scalar("n_pairs", num(&j["missing"], "n_used"), Tol::exact())
            .finish();
        assert_eq!(j["missing"], missing(22, 6, json!({"x": 15, "y": 15})));
    }
    let f = png(&s, "heat.png");
    let j = s.json(&[
        "viz",
        "heatmap",
        d,
        "--cols",
        "x",
        "y",
        "xi",
        "-f",
        f.as_str(),
    ]);
    Check::new(&desc)
        .scalar("n_complete", num(&j["missing"], "n_used"), Tol::exact())
        .finish();
    assert!(is_png(&f));
}

/// CLI-8: an Int64 group column was refused ("Group column must be string"); the row with a
/// missing code is dropped. Compares no R number (the groups are pinned by the unit test
/// commands::viz::tests::test_validate_cli_viz_box_groups), so it is not registered.
#[test]
fn test_cli_viz_box_int64_group() {
    let s = temp_session();
    let d = load_dataset(&s, "datasets/generated/contingency_na_codes.csv");
    let f = png(&s, "box.png");
    let out = s.ok(&[
        "viz",
        "box",
        d.as_str(),
        "-y",
        "out",
        "-g",
        "code",
        "-f",
        f.as_str(),
    ]);
    assert!(out.contains("Box plot saved to") && is_png(&f), "{out}");
    assert_eq!(
        note(&out),
        Some("Note: 1 of 12 rows dropped because of missing values (code: 1)"),
        "{out}"
    );
    let j = s.json(&[
        "viz",
        "box",
        d.as_str(),
        "-y",
        "out",
        "-g",
        "code",
        "-f",
        f.as_str(),
    ]);
    assert_eq!(j["missing"], missing(12, 11, json!({"code": 1})));
}

/// CLI-8/CLI-9: histogram of an Int64 column with missing cells; dendrogram of an Int64 column,
/// and its refusal of missing values; residual plots of an Int64 response. Not registered (no R
/// number).
#[test]
fn test_cli_viz_histogram_dendrogram_residuals() {
    let s = temp_session();
    let d2 = load_dataset(&s, "datasets/generated/two_sample_na.csv");
    let dq = load_dataset(&s, "datasets/generated/friedman_quade.csv");
    let dp = load_dataset(&s, "datasets/generated/pairwise_mahalanobis_na.csv");
    let h = png(&s, "hist.png");
    let out = s.ok(&[
        "viz",
        "histogram",
        d2.as_str(),
        "--col",
        "xi",
        "-f",
        h.as_str(),
    ]);
    assert_eq!(
        note(&out),
        Some("Note: 2 of 22 rows dropped because of missing values (xi: 2)"),
        "{out}"
    );
    assert!(is_png(&h));
    let t = png(&s, "tree.png");
    s.ok(&[
        "viz",
        "dendrogram",
        dq.as_str(),
        "--cols",
        "y",
        "-f",
        t.as_str(),
    ]);
    assert!(is_png(&t));
    let e = s.err(&[
        "viz",
        "dendrogram",
        d2.as_str(),
        "--cols",
        "xi",
        "yi",
        "-f",
        t.as_str(),
    ]);
    assert!(e.contains("Column 'xi' contains 2 missing value(s)"), "{e}");
    let r = png(&s, "resid.png");
    s.ok(&[
        "viz",
        "residuals",
        dp.as_str(),
        "-y",
        "id",
        "-x",
        "b",
        "-f",
        r.as_str(),
    ]);
    for suffix in ["", "_qq", "_scale_location", "_leverage"] {
        assert!(
            is_png(&png(&s, &format!("resid{suffix}.png"))),
            "resid{suffix}.png"
        );
    }
}

/// CLI-9: event-study and IRF rows with a missing cell were plotted at 0; an IRF band column that
/// cannot be read was silently ignored. Not registered (no R number).
#[test]
fn test_cli_viz_event_study_irf() {
    let s = temp_session();
    let d = load_dataset(&s, "datasets/generated/two_sample_na.csv");
    let d = d.as_str();
    let f = png(&s, "event.png");
    let out = s.ok(&[
        "viz",
        "event-study",
        d,
        "--time-col",
        "id",
        "--estimate-col",
        "x",
        "--ci-lower-col",
        "y",
        "--ci-upper-col",
        "xi",
        "-f",
        f.as_str(),
    ]);
    assert_eq!(
        note(&out),
        Some("Note: 16 of 22 rows dropped because of missing values (x: 15, y: 15, xi: 2)"),
        "{out}"
    );
    assert!(is_png(&f));
    let i = png(&s, "irf.png");
    let out = s.ok(&[
        "viz",
        "irf",
        d,
        "--horizon-col",
        "id",
        "--response-col",
        "xi",
        "-f",
        i.as_str(),
    ]);
    assert_eq!(
        note(&out),
        Some("Note: 2 of 22 rows dropped because of missing values (xi: 2)"),
        "{out}"
    );
    let e = s.err(&[
        "viz",
        "irf",
        d,
        "--horizon-col",
        "id",
        "--response-col",
        "xi",
        "--ci-lower-col",
        "nope",
        "--ci-upper-col",
        "x",
        "-f",
        i.as_str(),
    ]);
    assert!(e.contains("Column 'nope' not found"), "{e}");
}
