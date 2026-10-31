//! T2-WP22: the X-2 / X-8 reviewer repro through the p2a CLI: empty cells are dropped as R
//! drops NA, with a note, and integer columns are numeric.
use crate::common::*;
use serde_json::Value;

/// A JSON number; `null` (how the CLI prints NaN) and anything else is NaN.
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// `v[k]` as a number.
fn f(v: &Value, k: &str) -> f64 {
    num(&v[k])
}

/// X-2: the one-sample and paired t-tests drop the empty cells of x and y, as R does (they read
/// them as 0 before Tier 2), and say so.
#[test]
fn test_validate_cli_x2_reviewer_missing_cells() {
    let case = Case::load("na_x2_x8_reviewer");
    let s = temp_session();
    let ds = s.load(&case); // "na_reviewer"
    let one = s.json(&["stats", "t-test-one", &ds, "-c", "x"]);
    let pr = s.json(&["stats", "t-test-paired", &ds, "--col1", "x", "--col2", "y"]);
    let txt = s.ok(&["stats", "t-test-one", &ds, "-c", "x"]);
    let rows = 8.0;
    let n = 8;
    Check::new(&case)
        .scalar("one_t", f(&one, "t_statistic"), Tol::stat(n))
        .scalar("one_df", f(&one, "df"), Tol::exact())
        .scalar("one_p", f(&one, "p_value"), Tol::p_value(n))
        .scalar("one_mean", f(&one, "mean"), Tol::coef(n))
        .vector(
            "one_ci",
            &[f(&one, "ci_lower"), f(&one, "ci_upper")],
            Tol::stat(n),
        )
        .scalar(
            "one_n",
            rows - f(&one["missing"], "n_dropped"),
            Tol::exact(),
        )
        .scalar("paired_t", f(&pr, "t_statistic"), Tol::stat(n))
        .scalar("paired_df", f(&pr, "df"), Tol::exact())
        .scalar("paired_p", f(&pr, "p_value"), Tol::p_value(n))
        .scalar("paired_mean_diff", f(&pr, "mean_difference"), Tol::coef(n))
        .scalar(
            "paired_n",
            rows - f(&pr["missing"], "n_dropped"),
            Tol::exact(),
        )
        .finish();
    assert!(
        txt.contains("Note: 1 of 8 rows dropped because of missing values (x: 1)"),
        "{txt}"
    );
}

/// X-8: an integer column in the one-sample t-test, and integer survival time and status in
/// Kaplan-Meier (they were refused as non-numeric before Tier 2).
#[test]
fn test_validate_cli_x8_reviewer_integer_columns() {
    let case = Case::load("na_x2_x8_reviewer");
    let s = temp_session();
    let ds = s.load(&case);
    let ti = s.json(&["stats", "t-test-one", &ds, "-c", "i"]);
    let km = s.json(&["survival", "km", &ds, "-t", "time", "-e", "event"]);
    let times: Vec<f64> = km["times"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let at = |t: f64| {
        times
            .iter()
            .position(|&u| u == t)
            .unwrap_or_else(|| panic!("time {t} missing: {times:?}"))
    };
    // R reports the event times only.
    let idx: Vec<usize> = case.vector("km_time").into_iter().map(at).collect();
    let pick = |k: &str| idx.iter().map(|&j| num(&km[k][j])).collect::<Vec<_>>();
    let n = 8;
    Check::new(&case)
        .scalar("int_t", f(&ti, "t_statistic"), Tol::stat(n))
        .scalar("int_df", f(&ti, "df"), Tol::exact())
        .scalar("int_p", f(&ti, "p_value"), Tol::p_value(n))
        .scalar("int_mean", f(&ti, "mean"), Tol::coef(n))
        .vector("km_surv", &pick("survival"), Tol::stat(n))
        .vector("km_std_err", &pick("std_errors"), Tol::se(n)) // the last is NaN in both (S = 0)
        .vector("km_n_risk", &pick("n_at_risk"), Tol::exact())
        .vector("km_n_event", &pick("n_events"), Tol::exact())
        .scalar("km_median", f(&km, "median_survival"), Tol::exact())
        .finish();
    assert!(ti.get("missing").is_none(), "{ti}");
    assert!(km.get("missing").is_none(), "{km}");
}
