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
