//! T2-WP19: Arellano-Bond GMM and panel unit-root tests through the p2a CLI.
//!
//! The registered tests compare `-F json` numbers with R cases at full tolerance and
//! printed numbers at print precision; the error test loads no case.
use crate::common::*;
use serde_json::Value;

/// The numbers of a JSON array.
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("not a number: {x}")))
        .collect()
}

/// A JSON number (null reads as NaN).
fn number(v: &Value) -> f64 {
    if v.is_null() {
        return f64::NAN;
    }
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// `panel gmm ab_balanced -y y -x x --entity id --time year` followed by `extra`.
fn gmm_argv<'a>(dataset: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut argv = vec![
        "panel", "gmm", dataset, "-y", "y", "-x", "x", "--entity", "id", "--time", "year",
    ];
    argv.extend_from_slice(extra);
    argv
}

/// The coefficient and SE of the table row whose first token is `name`.
fn coef_and_se(text: &str, name: &str) -> (f64, f64) {
    let line = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{text}"));
    let v: Vec<f64> = line
        .split_whitespace()
        .skip(1)
        .take(2)
        .map(|t| {
            t.parse()
                .unwrap_or_else(|e| panic!("{t:?} in {line:?}: {e}"))
        })
        .collect();
    (v[0], v[1])
}

/// pgmm's numbers in `-F json`: the default (two-step, Windmeijer), one-step classical,
/// two lags and collapsed instruments. Before: exit 2 on `--step`; the default gave
/// L1.y -0.32.
#[test]
fn test_validate_cli_panel_gmm_ab_balanced() {
    let s = temp_session();
    let case = Case::load("dpd_ab_balanced");
    s.load(&case);
    let names = ["lag(y, 1)", "x"];
    let exact = Tol::exact();
    let mut c = Check::new(&case);

    let j = s.json(&gmm_argv("ab_balanced", &[]));
    c.named(
        "twostep_coef",
        &names,
        &numbers(&j["coefficients"]),
        Tol::coef(600),
    )
    .named(
        "twostep_se_robust",
        &names,
        &numbers(&j["std_errors"]),
        Tol::se(600),
    )
    .scalar(
        "twostep_sargan",
        number(&j["sargan_statistic"]),
        Tol::stat(600),
    )
    .scalar("twostep_sargan_df_nonzero", number(&j["sargan_df"]), exact)
    .scalar("twostep_n_obs", number(&j["n_obs"]), exact)
    .scalar(
        "twostep_n_instruments_nonzero",
        number(&j["n_instruments"]),
        exact,
    )
    .scalar("twostep_ar2", number(&j["ar2_statistic"]), Tol::stat(600));
    assert_eq!(j["se_type"], "two-step, Windmeijer (2005) corrected");
    assert_eq!(j["sargan_homoskedastic"], Value::Null);
    assert_eq!(
        (j["lags"].as_u64(), j["min_lag"].as_u64()),
        (Some(1), Some(2))
    );

    let j = s.json(&gmm_argv(
        "ab_balanced",
        &["--step", "onestep", "--vcov", "classical"],
    ));
    c.named(
        "onestep_coef",
        &names,
        &numbers(&j["coefficients"]),
        Tol::coef(600),
    )
    .named(
        "onestep_se_textbook",
        &names,
        &numbers(&j["std_errors"]),
        Tol::se(600),
    )
    .scalar("onestep_sigma2", number(&j["sigma2"]), Tol::se(600))
    .scalar(
        "onestep_sargan_homoskedastic",
        number(&j["sargan_homoskedastic"]),
        Tol::stat(600),
    );
    assert_eq!(j["se_type"], "one-step, homoskedastic (Arellano-Bond 1991)");

    let j = s.json(&gmm_argv("ab_balanced", &["--lags", "2"]));
    c.named(
        "lags2_twostep_coef",
        &["lag(y, 1:2)1", "lag(y, 1:2)2", "x"],
        &numbers(&j["coefficients"]),
        Tol::coef(500),
    );
    assert_eq!(j["variables"], serde_json::json!(["L1.y", "L2.y", "x"]));

    let j = s.json(&gmm_argv("ab_balanced", &["--collapse"]));
    c.named(
        "collapse_twostep_coef",
        &names,
        &numbers(&j["coefficients"]),
        Tol::coef(600),
    );
    assert_eq!(j["collapse"], Value::Bool(true));
    c.finish();
}

/// The text output names the SE type and the Sargan/Hansen test; the table at 6 decimals.
#[test]
fn test_validate_cli_panel_gmm_text() {
    let s = temp_session();
    let case = Case::load("dpd_ab_balanced");
    s.load(&case);
    let t = s.ok(&gmm_argv("ab_balanced", &[]));
    assert!(
        t.contains("Std. errors: two-step, Windmeijer (2005) corrected"),
        "{t}"
    );
    assert!(t.contains("Sargan/Hansen test: chi2(20) = 26.5932"), "{t}");
    let (lag, lag_se) = coef_and_se(&t, "L1.y");
    let (x, x_se) = coef_and_se(&t, "x");
    let names = ["lag(y, 1)", "x"];
    let mut c = Check::new(&case);
    c.named("twostep_coef", &names, &[lag, x], Tol::printed(6))
        .named(
            "twostep_se_robust",
            &names,
            &[lag_se, x_se],
            Tol::printed(6),
        );
    c.finish();
}

/// Text unit ids, a year no unit observes, missing cells. Before: "Panel unit root test
/// failed: Empty dataset" (exit 1).
#[test]
fn test_validate_cli_panel_unit_root_text_ids() {
    let s = temp_session();
    let case = Case::load("panel_pur_extraction");
    let name = s.load(&case);
    assert_eq!(name, "ab_common_gap");
    let argv = [
        "panel",
        "unit-root",
        "ab_common_gap",
        "-v",
        "y",
        "--unit",
        "unit",
        "--time",
        "year",
        "--test",
        "ips",
        "--lags",
        "1",
    ];
    let j = s.json(&argv);
    let mut c = Check::new(&case);
    c.scalar("n_panels", number(&j["n_panels"]), Tol::exact())
        .scalar("avg_t", number(&j["avg_time_periods"]), Tol::exact())
        .vector("unit_tau", &numbers(&j["unit_statistics"]), Tol::stat(690))
        .scalar(
            "n_rows_dropped",
            number(&j["missing"]["n_dropped"]),
            Tol::exact(),
        );
    c.finish();
    let warnings = j["warnings"].as_array().expect("warnings array");
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().is_some_and(|w| w.contains("have gaps in time"))),
        "{warnings:?}"
    );
    let t = s.ok(&argv);
    assert!(t.contains("Warnings:"), "{t}");
    assert!(
        t.contains("10 of 700 rows dropped because of missing values (y: 10)"),
        "{t}"
    );
}

#[test]
fn test_cli_panel_gmm_duplicate_row_error() {
    let s = temp_session();
    let csv = std::fs::read_to_string(repo_path("validation/datasets/generated/ab_balanced.csv"))
        .unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    let dup = format!("{}\n{}\n", lines.join("\n"), lines[3]);
    let path = s.dir().join("dup.csv");
    std::fs::write(&path, dup).unwrap();
    s.load_as(&path, "dup", &[]);
    let e = s.err(&gmm_argv("dup", &[]));
    assert!(e.contains("has 2 rows for time 2002"), "{e}");
}

/// panel hdfe, feglm, pmg, pvcm and gls report the rows they drop: a text note and a JSON
/// `missing` object. The counts of rows used are R's: the two-way HDFE and FEGLM use the
/// rows complete in every column (`n_complete`), PMG and PVCM those complete in (inv,
/// value, capital, firm) (`n_complete_fe`). Panel GLS needs at least as many entities as
/// periods, so it runs on the first ten years (4 of their 101 rows have a missing value).
#[test]
fn test_validate_cli_panel_missing_notes() {
    let s = temp_session();
    let case = Case::load("panel_grunfeld_na");
    assert_eq!(s.load(&case), "grunfeld_na");
    let model = ["grunfeld_na", "-y", "inv", "-x", "value", "capital"];
    let argv = |cmd: &str, extra: &[&'static str]| -> Vec<String> {
        let mut v = vec!["panel".to_string(), cmd.to_string()];
        v.extend(model.iter().map(|a| a.to_string()));
        v.extend(extra.iter().map(|a| a.to_string()));
        v
    };
    let run_json = |cmd: &str, extra: &[&'static str]| {
        let args = argv(cmd, extra);
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        s.json(&args)
    };
    let hdfe_flags = ["--fe", "firm_name", "year"];
    let hdfe_args = argv("hdfe", &hdfe_flags);
    let t = s.ok(&hdfe_args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(
        t.lines()
            .any(|l| l.starts_with("Note: 7 of 200 rows dropped because of missing values")),
        "{t}"
    );
    let hdfe = run_json("hdfe", &hdfe_flags);
    let feglm = run_json(
        "feglm",
        &["--fe", "firm_name", "year", "--family", "gaussian"],
    );
    let pmg = run_json("pmg", &["--entity", "firm_name"]);
    let pvcm = run_json("pvcm", &["--entity", "firm_name"]);
    let mut c = Check::new(&case);
    c.scalar(
        "hdfe_n_removed",
        number(&hdfe["missing"]["n_dropped"]),
        Tol::exact(),
    )
    .scalar(
        "n_complete",
        number(&feglm["missing"]["n_used"]),
        Tol::exact(),
    )
    .scalar(
        "n_complete_fe",
        number(&pmg["missing"]["n_used"]),
        Tol::exact(),
    )
    .scalar(
        "n_complete_fe",
        number(&pvcm["missing"]["n_used"]),
        Tol::exact(),
    );
    c.finish();

    // Panel GLS on 1935-1944 and the rows with a missing year.
    let csv = std::fs::read_to_string(case.dataset_path()).unwrap();
    let mut lines = csv.lines();
    let mut early = vec![lines.next().unwrap().to_string()];
    early.extend(
        lines
            .filter(|l| {
                let year = l.split(',').nth(3).unwrap_or("");
                year.is_empty() || year.parse::<i64>().is_ok_and(|y| y <= 1944)
            })
            .map(str::to_string),
    );
    let path = s.dir().join("grunfeld_na_early.csv");
    std::fs::write(&path, early.join("\n") + "\n").unwrap();
    s.load_as(&path, "early", &[]);
    let gls = [
        "panel",
        "gls",
        "early",
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--entity",
        "firm_name",
        "--time",
        "year",
    ];
    let t = s.ok(&gls);
    assert!(
        t.lines().any(|l| l
            == "Note: 4 of 101 rows dropped because of missing values (inv: 2, year: 2)"),
        "{t}"
    );
    let j = s.json(&gls);
    assert_eq!(j["missing"]["n_dropped"].as_u64(), Some(4), "{j}");
}
