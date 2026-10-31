//! T2-WP16: ARIMA fit and forecasts through the p2a CLI.

use crate::common::*;

/// The case, a scratch session with the case's dataset loaded, and the dataset's name.
fn arima_session(id: &str) -> (Case, TempSession, String) {
    let case = Case::load(id);
    let s = temp_session();
    let name = s.load(&case);
    (case, s, name)
}

/// The six numbers after ':' on every trimmed line that starts with "t+".
fn forecast_rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("t+"))
        .map(|l| {
            let (_, numbers) = l.split_once(':').expect("a forecast row has a ':'");
            numbers
                .split_whitespace()
                .map(|v| v.parse().expect("a number"))
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// The second token of the first trimmed line whose first token is `name`.
fn coef_value(text: &str, name: &str) -> f64 {
    text.lines()
        .find_map(|l| {
            let mut tokens = l.split_whitespace();
            (tokens.next() == Some(name)).then(|| tokens.next()?.parse().ok())?
        })
        .unwrap_or_else(|| panic!("no line starting with `{name}` in\n{text}"))
}

fn numbers(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("not a number: {x}")))
        .collect()
}

/// R: forecast::forecast(Arima(a, c(1, 0, 1), method = "CSS"), h = 5) at the exact CSS
/// optimum, through `p2a -F json ts arima`.
#[test]
fn test_validate_cli_ts_arima_json() {
    let (case, s, name) = arima_session("arima_css_forecast");
    let j = s.json(&[
        "ts",
        "arima",
        &name,
        "--col",
        "a",
        "-p",
        "1",
        "-d",
        "0",
        "-q",
        "1",
        "--horizon",
        "5",
    ]);
    let n = 300;
    let names = ["ar1", "ma1", "intercept"];
    let coefs: Vec<f64> = names
        .iter()
        .map(|k| {
            j["coefficients"][k]
                .as_f64()
                .unwrap_or_else(|| panic!("{k}: {j}"))
        })
        .collect();
    let fc = &j["forecast"];
    Check::new(&case)
        .named("a_101_coef_tight", &names, &coefs, Tol::iterative(1e-6))
        .scalar(
            "a_101_sigma2_tight",
            j["sigma2"].as_f64().unwrap(),
            Tol::stat(n),
        )
        .scalar(
            "a_101_loglik_tight",
            j["loglik"].as_f64().unwrap(),
            Tol::stat(n),
        )
        .scalar(
            "a_101_aic_css_tight",
            j["aic"].as_f64().unwrap(),
            Tol::stat(n),
        )
        .vector("a_101_mean_tight", &numbers(&fc["mean"]), Tol::stat(n))
        .vector("a_101_se_tight", &numbers(&fc["std_errors"]), Tol::stat(n))
        .vector(
            "a_101_lower80_tight",
            &numbers(&fc["lower_80"]),
            Tol::stat(n),
        )
        .vector(
            "a_101_upper80_tight",
            &numbers(&fc["upper_80"]),
            Tol::stat(n),
        )
        .vector(
            "a_101_lower95_tight",
            &numbers(&fc["lower_95"]),
            Tol::stat(n),
        )
        .vector(
            "a_101_upper95_tight",
            &numbers(&fc["upper_95"]),
            Tol::stat(n),
        )
        .finish();
    assert_eq!(j["n_cond"], 1, "{j}");
    assert_eq!(j["converged"], true, "{j}");
}

/// R: forecast::forecast(Arima(dr, c(0, 1, 1), include.drift = TRUE, method = "CSS"), h = 5)
/// at the exact CSS optimum, through the text output of `p2a ts arima --drift`.
#[test]
fn test_validate_cli_ts_arima_text_drift() {
    let (case, s, name) = arima_session("arima_css_forecast");
    let t = s.ok(&[
        "ts",
        "arima",
        &name,
        "--col",
        "dr",
        "-p",
        "0",
        "-d",
        "1",
        "-q",
        "1",
        "--drift",
        "--horizon",
        "5",
    ]);
    let rows = forecast_rows(&t);
    assert_eq!(rows.len(), 5, "{t}");
    let col = |j: usize| rows.iter().map(|row| row[j]).collect::<Vec<_>>();
    Check::new(&case)
        .vector("dr_011_drift_mean_tight", &col(0), Tol::printed(4))
        .vector("dr_011_drift_se_tight", &col(1), Tol::printed(4))
        .vector("dr_011_drift_lower80_tight", &col(2), Tol::printed(4))
        .vector("dr_011_drift_upper80_tight", &col(3), Tol::printed(4))
        .vector("dr_011_drift_lower95_tight", &col(4), Tol::printed(4))
        .vector("dr_011_drift_upper95_tight", &col(5), Tol::printed(4))
        .named_subset(
            "dr_011_drift_coef_tight",
            &["drift"],
            &["drift"],
            &[coef_value(&t, "drift")],
            Tol::printed(6),
        )
        .finish();
}

/// Option and data errors: include_mean with d > 0, missing values (with the fill hint),
/// text; an integer column fits.
#[test]
fn test_cli_ts_arima_errors() {
    let (_, s, name) = arima_session("arima_css_forecast");
    let arima = |col: &'static str, d: &'static str, extra: &[&'static str]| {
        let mut args = vec![
            "ts",
            "arima",
            name.as_str(),
            "--col",
            col,
            "-p",
            if col == "c" { "0" } else { "1" },
            "-d",
            d,
            "-q",
            "1",
        ];
        args.extend_from_slice(extra);
        args
    };
    let e = s.err(&arima("c", "1", &["--include-mean", "true"]));
    assert!(e.contains("include_mean"), "{e}");
    let e = s.err(&arima("a_na", "0", &[]));
    assert!(
        e.contains("Column 'a_na' contains 2 missing value(s)")
            && e.contains("munge fill-na")
            && e.contains("impute rather than drop"),
        "{e}"
    );
    let e = s.err(&arima("lab", "0", &[]));
    assert!(e.contains("non-numeric"), "{e}");
    let t = s.ok(&arima("ai", "0", &[]));
    assert!(t.contains("intercept"), "{t}");
}

/// ts changepoint, ar and garch read integer columns and refuse missing values.
#[test]
fn test_cli_ts_extraction() {
    let (_, s, name) = arima_session("arima_css_forecast");
    s.ok(&["ts", "changepoint", &name, "--col", "ai"]);
    // `--aic` is a flag that is always on (clap SetTrue with default "true"), so `--aic false`
    // does not parse; the order is chosen by AIC here.
    s.ok(&[
        "ts", "ar", &name, "--col", "ai", "--method", "ols", "--order", "2",
    ]);
    // Only the dtype is asserted for GARCH; TS-21 owns its numbers.
    let out = s.run(&["ts", "garch", &name, "--col", "ai"]);
    assert_no_panic(&out, "ts garch --col ai");
    let stderr = String::from_utf8_lossy(&out.stderr);
    for word in ["not numeric", "non-numeric", "empty"] {
        assert!(!stderr.contains(word), "{stderr}");
    }
    for cmd in ["changepoint", "ar", "garch"] {
        let e = s.err(&["ts", cmd, &name, "--col", "a_na"]);
        assert!(e.contains("missing value"), "ts {cmd}: {e}");
    }
}
