//! T2-WP14: spatial econometrics through the p2a CLI (integer columns, missing values).
//!
//! Coordinates, outcomes and regressors are read through the shared extractor with
//! `NaPolicy::Fail`: integer columns give the same results as their float copies, and a
//! missing or text cell is an error naming the column (exit 1).
use crate::common::*;
use std::path::{Path, PathBuf};

/// Writes the 25 integer points of moran_int_points.R (plan T2-WP14, Appendix A.3) to
/// `dir/name` as CSV with columns gx, gy, v, for i = 0..24:
/// gx = (40 i^2 + 3 i) mod 101, gy = (2 i^2 + 7 i + 5) mod 97,
/// v = (gx + gy) div 10 + (3 i) mod 7 (no kNN ties for k = 4).
/// `float` writes "43.0" so that polars infers Float64; `blank` = (column, row) is left
/// empty (a null); `text` = (column, row) is written as "a".
fn write_points(
    dir: &Path,
    name: &str,
    float: bool,
    blank: Option<(&str, usize)>,
    text: Option<(&str, usize)>,
) -> PathBuf {
    let mut csv = String::from("gx,gy,v\n");
    for i in 0..25i64 {
        let gx = (40 * i * i + 3 * i) % 101;
        let gy = (2 * i * i + 7 * i + 5) % 97;
        let v = (gx + gy) / 10 + (3 * i) % 7;
        let row = i as usize;
        let cells: Vec<String> = [("gx", gx), ("gy", gy), ("v", v)]
            .into_iter()
            .map(|(col, value)| {
                if blank == Some((col, row)) {
                    String::new()
                } else if text == Some((col, row)) {
                    "a".to_string()
                } else if float {
                    format!("{value}.0")
                } else {
                    value.to_string()
                }
            })
            .collect();
        csv.push_str(&cells.join(","));
        csv.push('\n');
    }
    let path = dir.join(name);
    std::fs::write(&path, csv).unwrap();
    path
}

/// `spatial moran <dataset> -y v --coord-x gx --coord-y gy -k 4`.
fn moran(dataset: &str) -> Vec<&str> {
    vec![
        "spatial",
        "moran",
        dataset,
        "-y",
        "v",
        "--coord-x",
        "gx",
        "--coord-y",
        "gy",
        "-k",
        "4",
    ]
}

/// `spatial <model> <dataset> -y v -x gx --coord-x gx --coord-y gy -k 4 [--durbin]`.
fn model<'a>(cmd: &'a str, dataset: &'a str, durbin: bool) -> Vec<&'a str> {
    let mut args = vec![
        "spatial",
        cmd,
        dataset,
        "-y",
        "v",
        "-x",
        "gx",
        "--coord-x",
        "gx",
        "--coord-y",
        "gy",
        "-k",
        "4",
    ];
    if durbin {
        args.push("--durbin");
    }
    args
}

/// R: moran_int_points.R (plan T2-WP14, Appendix A.3; base R, because spdep is not
/// installed): Moran's I with the randomisation variance on kNN-4 row-standardised
/// weights.
#[test]
fn test_cli_spatial_moran_int64_matches_r() {
    let ts = temp_session();
    ts.load_as(
        &write_points(ts.dir(), "pts.csv", false, None, None),
        "pts",
        &[],
    );
    let j = ts.json(&moran("pts"));
    let n = 25;
    for (key, r_value) in [
        ("moran_i", 0.388_188_976_377_952_8),
        ("expected_i", -0.041_666_666_666_666_664),
        ("variance", 0.012_700_853_696_201_244),
        ("z_score", 3.814_223_835_409_538),
    ] {
        let v = j[key].as_f64().unwrap_or_else(|| panic!("{key} in {j}"));
        assert!(Tol::stat(n).allows(v, r_value), "{key} {v} vs R {r_value}");
    }
    // two-sided, the CLI default
    let p = j["p_value"].as_f64().unwrap();
    assert!(
        Tol::p_value(n).allows(p, 0.000_136_611_607_587_379_17),
        "p {p}"
    );
    assert_eq!(j["n"], 25);
}

#[test]
fn test_cli_spatial_models_int64_equal_float64() {
    let ts = temp_session();
    ts.load_as(
        &write_points(ts.dir(), "pts.csv", false, None, None),
        "pts",
        &[],
    );
    ts.load_as(
        &write_points(ts.dir(), "ptsf.csv", true, None, None),
        "ptsf",
        &[],
    );
    for (cmd, durbin) in [("sar", false), ("sar", true), ("sem", false)] {
        let (int, flt) = (model(cmd, "pts", durbin), model(cmd, "ptsf", durbin));
        assert_eq!(ts.json(&int), ts.json(&flt), "{cmd} durbin={durbin}");
        assert_eq!(ts.ok(&int), ts.ok(&flt), "{cmd} durbin={durbin}");
    }
    assert_eq!(ts.json(&moran("pts")), ts.json(&moran("ptsf")));
}

#[test]
fn test_na_cli_spatial_missing_values_refused() {
    let ts = temp_session();
    ts.load_as(
        &write_points(ts.dir(), "gx_na.csv", false, Some(("gx", 9)), None),
        "gx_na",
        &[],
    );
    ts.load_as(
        &write_points(ts.dir(), "v_na.csv", false, Some(("v", 2)), None),
        "v_na",
        &[],
    );
    for args in [moran("gx_na"), model("sar", "gx_na", false)] {
        let e = ts.err(&args);
        assert!(
            e.contains("Failed to build spatial weights")
                && e.contains("Column 'gx' contains 1 missing value(s)"),
            "{e}"
        );
    }
    for args in [
        moran("v_na"),
        model("sar", "v_na", false),
        model("sem", "v_na", false),
        model("sac", "v_na", false),
    ] {
        let e = ts.err(&args);
        assert!(e.contains("Column 'v' contains 1 missing value(s)"), "{e}");
    }
}

#[test]
fn test_na_cli_spatial_text_coordinate() {
    let ts = temp_session();
    ts.load_as(
        &write_points(ts.dir(), "gx_text.csv", false, None, Some(("gx", 5))),
        "gx_text",
        &[],
    );
    let e = ts.err(&moran("gx_text"));
    assert!(e.contains("Column 'gx' contains non-numeric values"), "{e}");
}
