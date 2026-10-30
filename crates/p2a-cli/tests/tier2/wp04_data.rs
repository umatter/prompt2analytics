//! T2-WP04: munge bin, cast, mutate and fill-na, and interactive charts with a missing group,
//! through the p2a CLI.
use crate::common::*;
use std::path::PathBuf;

/// Writes `body` to `<dir>/<name>` and returns the path.
fn write(s: &TempSession, name: &str, body: &str) -> PathBuf {
    let p = s.dir().join(name);
    std::fs::write(&p, body).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    p
}

/// The `Counts:` line `munge bin` prints for the R bin codes `codes` (0-based bin, -1 = NA)
/// of `k` bins, `n_missing` of the NA rows being missing values.
fn counts_line(codes: &[f64], k: usize, n_missing: usize) -> String {
    let bins: Vec<String> = (0..k)
        .map(|j| {
            let n = codes.iter().filter(|c| **c == j as f64).count();
            format!("bin_{j}: {n}")
        })
        .collect();
    let na = codes.iter().filter(|c| **c < 0.0).count();
    format!(
        "Counts: {}; NA: {na} ({n_missing} missing, {} outside the edges)",
        bins.join(", "),
        na - n_missing
    )
}

#[test]
fn test_validate_cli_munge_bin_cut() {
    // validation/r/cases/data_quality.R, case data_bin_cut: cut(x, breaks = c(0, 1, 3, 5)),
    // include.lowest = TRUE, right = FALSE. R: sum(is.na(x)) = 2 (rows 10 and 12); the other
    // NA rows are outside the edges.
    let case = Case::load("data_bin_cut");
    let s = temp_session();
    s.load_as(&case.dataset_path(), "b", &[]);
    for (key, flag) in [
        ("right_codes", None),
        ("include_lowest_codes", Some("--include-lowest")),
        ("left_closed_codes", Some("--left-closed")),
    ] {
        let mut args = vec![
            "munge", "bin", "b", "--column", "x", "--method", "custom", "--edges", "0,1,3,5",
        ];
        args.extend(flag);
        let stdout = s.ok(&args);
        let want = counts_line(&case.vector(key), 3, 2);
        assert!(stdout.contains(&want), "{key}: want {want:?}\n{stdout}");
    }
    // The R codes behind the lines above, so the helper cannot drift from them.
    assert_eq!(
        counts_line(&case.vector("include_lowest_codes"), 3, 2),
        "Counts: bin_0: 3, bin_1: 3, bin_2: 2; NA: 4 (2 missing, 2 outside the edges)"
    );
}

#[test]
fn test_cli_munge_fill_na_all_missing_is_an_error() {
    // DATA-35. An all-empty CSV column has no observed value, so its mean is undefined
    // (R: mean(c(NA, NA, NA), na.rm = TRUE) is NaN); it was filled with 0.
    let s = temp_session();
    let e = write(&s, "e.csv", "id,z\n1,\n2,\n3,\n");
    s.load_as(&e, "e", &[]);
    let stderr = s.err(&[
        "munge",
        "fill-na",
        "e",
        "--method",
        "mean",
        "--columns",
        "z",
    ]);
    assert!(stderr.contains("no observed values"), "{stderr}");
}

#[test]
fn test_cli_munge_cast_reports_failed_values() {
    // DATA-37. R: as.integer(c("1,234", "7", "x")) is NA 7 NA with a coercion warning.
    let s = temp_session();
    let c = write(&s, "c.csv", "v\n\"1,234\"\n7\nx\n");
    s.load_as(&c, "c", &[]);
    let stdout = s.ok(&["munge", "cast", "c", "--column", "v", "--dtype", "int"]);
    assert!(
        stdout.contains(
            "Warning: column 'v': 2 values could not be converted to int and are now missing \
             (first: '1,234', 'x')"
        ),
        "{stdout}"
    );
    let stderr = s.err(&[
        "munge", "cast", "c", "--column", "v", "--dtype", "int", "--strict",
    ]);
    assert!(
        stderr.contains("2 values could not be converted"),
        "{stderr}"
    );
}

#[test]
fn test_cli_munge_mutate_refuses_text() {
    // DATA-37. R: c("$5", "10", "2,000") * c(6, 3, 1.5) is an error.
    let s = temp_session();
    let m = write(&s, "m.csv", "p,q\n$5,6\n10,3\n\"2,000\",1.5\n");
    s.load_as(&m, "m", &[]);
    let stderr = s.err(&[
        "munge",
        "mutate",
        "m",
        "--new-col",
        "pq",
        "--expr",
        "mul:p:q",
    ]);
    assert!(stderr.contains("numeric"), "{stderr}");
}

#[test]
fn test_cli_viz_interactive_with_missing_group() {
    // DATA-27: a missing group label aborted the CLI (exit 101); it is the group "NA" now.
    let s = temp_session();
    let n = write(&s, "n.csv", "x,y,g\n1,2.1,a\n2,3.9,\n3,6.2,b\n4,7.8,a\n");
    s.load_as(&n, "n", &[]);
    for (args, file) in [
        (
            vec![
                "viz",
                "scatter-interactive",
                "n",
                "-x",
                "x",
                "-y",
                "y",
                "-g",
                "g",
                "-f",
                "s.html",
            ],
            "s.html",
        ),
        (
            vec![
                "viz",
                "histogram-interactive",
                "n",
                "--col",
                "x",
                "-g",
                "g",
                "-f",
                "h.html",
            ],
            "h.html",
        ),
    ] {
        s.ok(&args);
        let html = std::fs::read_to_string(s.dir().join(file)).unwrap();
        assert!(html.contains("\"NA\""), "{file} has no group \"NA\"");
    }
}
