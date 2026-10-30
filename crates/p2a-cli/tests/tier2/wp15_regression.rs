//! T2-WP15: regression helpers through the p2a CLI.
use crate::common::*;
use p2a_core::data::columns::numeric_options;
use serde_json::{Value, json};

/// The complete (x, y) pairs of a case dataset, in row order.
fn complete_pairs(case: &Case) -> (Vec<f64>, Vec<f64>) {
    let df = p2a_core::DataLoader::load_csv(case.dataset_path()).unwrap();
    let x = numeric_options(&df, "x").unwrap();
    let y = numeric_options(&df, "y").unwrap();
    x.iter().zip(&y).filter_map(|(a, b)| a.zip(*b)).unzip()
}

/// The `missing` object of reg_smoothers_na (2 missing x, 2 missing y).
fn smoothers_missing() -> Value {
    json!({"n_total": 60, "n_used": 56, "n_dropped": 4, "by_column": {"x": 2, "y": 2}})
}

const SMOOTHERS_NOTE: &str = "Note: 4 of 60 rows dropped because of missing values (x: 2, y: 2)";

/// REG-21: rows with a missing cluster id are dropped and the model re-estimated;
/// the text shows the drop as a warning, `-F json` as a `missing` object.
#[test]
fn test_validate_cli_reg_clustered_na() {
    let case = Case::load("reg_cluster_na");
    let s = temp_session();
    let name = s.load(&case);
    let args = [
        "reg",
        "clustered",
        &name,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--cluster",
        "firm_label",
    ];
    let text = s.ok(&args);
    assert!(text.contains("Observations: 197\n"), "{text}");
    assert!(
        text.contains("  - 3 of 200 rows dropped because of missing values (firm_label: 3)\n"),
        "{text}"
    );
    assert!(!text.contains("Note:"), "the drop is printed once\n{text}");
    let (names, values): (Vec<String>, Vec<Vec<f64>>) =
        p2a_validation::text::coefficient_rows(&text)
            .into_iter()
            .unzip();
    let coef: Vec<f64> = values.iter().map(|v| v[0]).collect();
    let se: Vec<f64> = values.iter().map(|v| v[1]).collect();
    Check::new(&case)
        .named("coef", &names, &coef, Tol::printed(6))
        .named("se_cluster_hc1", &names, &se, Tol::printed(6))
        .finish();

    let j = s.json(&args);
    let rows = j["coefficients"].as_array().unwrap();
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    let pick = |key: &str| -> Vec<f64> { rows.iter().map(|r| r[key].as_f64().unwrap()).collect() };
    Check::new(&case)
        .named("coef", &names, &pick("coefficient"), Tol::coef(197))
        .named("se_cluster_hc1", &names, &pick("std_error"), Tol::se(197))
        .scalar(
            "n_used",
            j["n_observations"].as_f64().unwrap(),
            Tol::exact(),
        )
        .finish();
    assert_eq!(j["n_clusters"], 10);
    assert_eq!(
        j["missing"],
        json!({"n_total": 200, "n_used": 197, "n_dropped": 3, "by_column": {"firm_label": 3}})
    );
}

/// REG-13: `reg line` drops the incomplete pairs (R's line uses complete pairs) and
/// prints the note; the fit equals `line` on the complete pairs.
#[test]
fn test_validate_cli_reg_line_na() {
    let case = Case::load("reg_smoothers_na");
    let s = temp_session();
    let name = s.load(&case);
    let args = ["reg", "line", &name, "-y", "y", "-x", "x"];
    let text = s.ok(&args);
    assert!(text.contains("Observations: 56\n"), "{text}");
    assert!(text.trim_end().ends_with(SMOOTHERS_NOTE), "{text}");

    let j = s.json(&args);
    Check::new(&case)
        .scalar("n_complete", j["n"].as_f64().unwrap(), Tol::exact())
        .finish();
    let (xc, yc) = complete_pairs(&case);
    let fit = p2a_core::regression::line::line(&xc, &yc, Some(1)).unwrap();
    assert_eq!(j["intercept"].as_f64().unwrap(), fit.intercept);
    assert_eq!(j["slope"].as_f64().unwrap(), fit.slope);
    assert_eq!(j["missing"], smoothers_missing());
}

/// REG-13: `reg supsmu` drops the incomplete pairs; its x grid is R's grid after the
/// deletion, and the smooth equals `supsmu` on the complete pairs.
#[test]
fn test_validate_cli_reg_supsmu_na() {
    let case = Case::load("reg_smoothers_na");
    let s = temp_session();
    let name = s.load(&case);
    let args = ["reg", "supsmu", &name, "-y", "y", "-x", "x"];
    let text = s.ok(&args);
    assert!(text.contains("Observations: 56\n"), "{text}");
    assert!(text.trim_end().ends_with(SMOOTHERS_NOTE), "{text}");
    let points: Vec<(f64, f64)> = text
        .lines()
        .filter_map(|l| {
            let rest = l.trim_start().split_once(": x=")?.1;
            let (x, y) = rest.split_once(", y=")?;
            Some((x.parse().unwrap(), y.parse().unwrap()))
        })
        .collect();
    assert_eq!(points.len(), 10, "{text}");
    let r_x = case.vector("supsmu_x");
    for (i, (x, y)) in points.iter().enumerate() {
        assert!(
            Tol::printed(4).allows(*x, r_x[i]),
            "x[{i}] {x} vs R {}",
            r_x[i]
        );
        assert!(y.is_finite(), "y[{i}] = {y}");
    }

    let j = s.json(&args);
    assert_eq!(j["n"], 56);
    let (xc, yc) = complete_pairs(&case);
    let fit = p2a_core::regression::supsmu::supsmu(&xc, &yc, None, None, false, 0.0).unwrap();
    let fitted: Vec<f64> = j["fitted_sample"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    // Same doubles in (the x grid above is R's, and reg line's residuals match the
    // in-process fit exactly), but the p2a binary and this test binary round the
    // smoother's boundary fits differently in the last bits (1-2 ulp observed), so this
    // cross-process comparison allows 1e-12 relative; it is not an R pin (REG-31).
    assert_eq!(fitted.len(), 10);
    for (a, e) in fitted.iter().zip(&fit.y[..10]) {
        assert!(Tol::new(1e-12, 0.0).allows(*a, *e), "{a} vs {e}");
    }
    assert_eq!(j["missing"], smoothers_missing());
}

/// REG-25 and REG-13: Int64 columns (mtcars hp) are numbers in reg quantreg, loess,
/// smooth-spline, line and supsmu; nothing is dropped, so no output has `missing`.
#[test]
fn test_validate_cli_reg_integer_columns() {
    let case = Case::load("ols_mtcars");
    let s = temp_session();
    let name = s.load(&case);
    let q = s.json(&["reg", "quantreg", &name, "-y", "hp", "-x", "wt"]);
    assert_eq!(q["n_obs"], 32);
    let df = p2a_core::DataLoader::load_csv(case.dataset_path()).unwrap();
    let ds = p2a_core::Dataset::new(df.clone());
    let fit = p2a_core::run_quantreg(&ds, "hp", &["wt"], 0.5).unwrap();
    let estimates: Vec<f64> = q["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["estimate"].as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = fit.coefficients.iter().map(|c| c.estimate).collect();
    // The p2a process and this test process can differ in the last bits of the IRLS
    // fit (2 ulp on the intercept were observed), so this cross-process comparison
    // allows 1e-12 relative; it is not an R pin.
    assert_eq!(estimates.len(), expected.len());
    for (a, e) in estimates.iter().zip(&expected) {
        assert!(Tol::new(1e-12, 0.0).allows(*a, *e), "{a} vs {e}");
    }
    let loess = s.json(&["reg", "loess", &name, "-y", "mpg", "-x", "hp"]);
    assert_eq!(loess["n_obs"], 32);
    let spline = s.json(&[
        "reg",
        "smooth-spline",
        &name,
        "-y",
        "mpg",
        "-x",
        "hp",
        "--df",
        "4",
    ]);
    assert_eq!(spline["n_obs"], 32);
    let line = s.json(&["reg", "line", &name, "-y", "mpg", "-x", "hp"]);
    assert_eq!(line["n"], 32);
    // supsmu's n counts the distinct x values (hp has ties).
    let supsmu = s.json(&["reg", "supsmu", &name, "-y", "mpg", "-x", "hp"]);
    let column = |c: &str| -> Vec<f64> {
        numeric_options(&df, c)
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect()
    };
    let smooth =
        p2a_core::regression::supsmu::supsmu(&column("hp"), &column("mpg"), None, None, false, 0.0)
            .unwrap();
    assert_eq!(supsmu["n"].as_u64().unwrap() as usize, smooth.n);
    for (what, v) in [
        ("quantreg", &q),
        ("loess", &loess),
        ("smooth-spline", &spline),
        ("line", &line),
        ("supsmu", &supsmu),
    ] {
        assert!(v.get("missing").is_none(), "{what}: {v}");
    }
}

/// REG-25: smooth.spline refuses missing values, as R does; the error names the column.
#[test]
fn test_validate_cli_reg_smooth_spline_missing_is_error() {
    let case = Case::load("reg_smoothers_na");
    let s = temp_session();
    let name = s.load(&case);
    let err = s.err(&["reg", "smooth-spline", &name, "-y", "y", "-x", "x"]);
    assert!(
        err.contains("Column 'x' contains 2 missing value(s)"),
        "{err}"
    );
}

/// REG-25: reg loess and reg quantreg drop the incomplete rows (R's na.omit), print the
/// note as their last text line and add `missing` to `-F json`.
#[test]
fn test_validate_cli_reg_loess_quantreg_missing_note() {
    let case = Case::load("reg_smoothers_na");
    let s = temp_session();
    let name = s.load(&case);
    for (args, note) in [
        (
            ["reg", "loess", &name, "-y", "y", "-x", "x"],
            SMOOTHERS_NOTE,
        ),
        (
            ["reg", "quantreg", &name, "-y", "y", "-x", "x"],
            "Note: 4 of 60 rows dropped because of missing values (y: 2, x: 2)",
        ),
    ] {
        let text = s.ok(&args);
        assert!(text.contains("Observations: 56\n"), "{text}");
        assert!(text.trim_end().ends_with(note), "{text}");
        let j = s.json(&args);
        Check::new(&case)
            .scalar("n_complete", j["n_obs"].as_f64().unwrap(), Tol::exact())
            .finish();
        assert_eq!(j["missing"], smoothers_missing(), "{j}");
    }
}
