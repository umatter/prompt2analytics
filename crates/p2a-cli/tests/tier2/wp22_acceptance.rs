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

/// The last non-empty line of a command's text output.
fn last_line(text: &str) -> &str {
    text.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("")
}

/// causal iv reports the rows it drops (text `Note:` line, JSON `missing`) and names its
/// first-stage statistic with its df: the CLI fits robust SEs, so it is the HC1 Wald F of the
/// excluded instruments, R's on the complete rows (case iv_na_rows).
#[test]
fn test_validate_cli_causal_iv_reports_missing_rows() {
    let case = Case::load("iv_na_rows");
    let s = temp_session();
    let ds = s.load(&case);
    let args = [
        "causal",
        "iv",
        ds.as_str(),
        "-y",
        "y",
        "--exog",
        "w",
        "k",
        "--endog",
        "x",
        "--instruments",
        "z1",
        "z2",
    ];
    let txt = s.ok(&args);
    let j = s.json(&args);
    let fs = &j["first_stage"];
    let n = 294;
    Check::new(&case)
        .scalar("n_obs", f(&j, "n_observations"), Tol::exact())
        .scalar("n_dropped", f(&j["missing"], "n_dropped"), Tol::exact())
        .vector("weak_f_hc1", &[num(&fs["f_stats"][0])], Tol::stat(n))
        .vector("weak_df1", &[num(&fs["df"][0][0])], Tol::exact())
        .vector("weak_df2", &[num(&fs["df"][0][1])], Tol::exact())
        .finish();
    assert_eq!(
        fs["f_type"], "HC1 Wald F of the excluded instruments",
        "{j}"
    );
    assert!(
        txt.contains("First-stage diagnostics (HC1 Wald F of the excluded instruments):")
            && txt.contains("x F-statistic: 88.0139 on 2 and 289 DF"),
        "{txt}"
    );
    assert!(
        last_line(&txt).starts_with("Note: 6 of 300 rows dropped because of missing values ("),
        "{txt}"
    );
}

/// causal mediation reports the rows it drops: R's complete rows of the model columns (case
/// causal_mediation_na), and after trimming at 0.05 the units R's IPW keeps.
#[test]
fn test_validate_cli_causal_mediation_reports_missing_rows() {
    let case = Case::load("causal_mediation_na");
    let s = temp_session();
    let ds = s.load(&case); // "causal_na"
    let args = [
        "causal",
        "mediation",
        ds.as_str(),
        "-y",
        "y",
        "-t",
        "d",
        "-m",
        "m",
        "-x",
        "x1",
        "x2",
        "--bootstrap",
        "50",
    ];
    let txt = s.ok(&args);
    let j = s.json(&args);
    Check::new(&case)
        .scalar("n_complete", f(&j["missing"], "n_used"), Tol::exact())
        .scalar("n_dropped", f(&j["missing"], "n_dropped"), Tol::exact())
        .scalar(
            "missing_y",
            f(&j["missing"]["by_column"], "y"),
            Tol::exact(),
        )
        .scalar(
            "missing_m",
            f(&j["missing"]["by_column"], "m"),
            Tol::exact(),
        )
        .scalar(
            "missing_x1",
            f(&j["missing"]["by_column"], "x1"),
            Tol::exact(),
        )
        .scalar("ipw_kept_trim05", f(&j, "n_obs"), Tol::exact())
        .finish();
    assert_eq!(
        last_line(&txt),
        "Note: 50 of 400 rows dropped because of missing values (y: 15, m: 25, x1: 10)",
        "{txt}"
    );
}

/// causal staggered-did -F json carries the pre-trend test's method, R's analytic Wald
/// statistic (case did_cs_analytic, never-treated controls, outcome regression), and the rows
/// dropped when a cell is missing.
#[test]
fn test_validate_cli_staggered_did_json_reports_pretrend_method() {
    let case = Case::load("did_cs_analytic");
    let s = temp_session();
    let ds = s.load(&case); // "cs_panel"
    let did = |ds: &str| {
        s.json(&[
            "causal",
            "staggered-did",
            ds,
            "-y",
            "y",
            "--treatment-time",
            "first_treat",
            "--time",
            "time",
            "--unit",
            "unit",
            "--bootstrap",
            "0",
        ])
    };
    let j = did(&ds);
    let pt = &j["pretrend_test"];
    let n = j["n_obs"].as_u64().unwrap() as usize;
    Check::new(&case)
        .scalar("never_pretrend_w", f(pt, "chi2"), Tol::stat(n))
        .scalar("never_pretrend_df", f(pt, "df"), Tol::exact())
        .scalar("never_pretrend_p", f(pt, "p_value"), Tol::p_value(n))
        .finish();
    let method = pt["method"].as_str().unwrap_or_default();
    assert!(method.contains("influence-function"), "{j}");
    assert!(j.get("missing").is_none(), "{j}");
    // One y cell blanked: the row is dropped and reported.
    let text = std::fs::read_to_string(case.dataset_path()).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let row = lines[5].rsplit_once(',').unwrap().0.to_string();
    lines[5] = format!("{row},");
    let gap = s.dir().join("cs_gap.csv");
    std::fs::write(&gap, lines.join("\n") + "\n").unwrap();
    s.load_as(&gap, "cs_gap", &[]);
    let g = did("cs_gap");
    assert_eq!(
        (
            g["missing"]["n_dropped"].as_u64(),
            g["missing"]["by_column"]["y"].as_u64()
        ),
        (Some(1), Some(1)),
        "{g}"
    );
}
