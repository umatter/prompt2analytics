//! `p2a munge lag|lead|diff|fill-na` on panel data: `--group-by` keeps the shift or the fill
//! within each unit, and `--time` shifts by the integer time index (period t - k of the same
//! unit, missing when that period is not in the data) instead of by row position.

use assert_cmd::Command;
use std::path::Path;

fn p2a() -> Command {
    Command::cargo_bin("p2a").expect("Failed to find p2a binary")
}

/// Two units; unit 1 has no period 3.
const PANEL: &str = "id,t,x\n1,1,1.1\n1,2,2.3\n1,4,3.9\n2,1,10.2\n2,2,11.7\n2,3,13.4\n";

/// Loads `csv` as dataset `p` into a fresh session and returns the session path.
fn session_with(dir: &Path, csv: &str) -> String {
    let session = dir.join("s.json").to_str().unwrap().to_string();
    let path = dir.join("p.csv");
    std::fs::write(&path, csv).unwrap();
    p2a()
        .args(["--session", &session, "data", "load"])
        .arg(&path)
        .args(["--name", "p"])
        .assert()
        .success();
    session
}

/// Runs `munge <args> -n out`, saves `out` and returns the column `col` (None for an empty cell).
fn munge_column(dir: &Path, session: &str, args: &[&str], col: &str) -> Vec<Option<f64>> {
    p2a()
        .args(["--session", session, "munge"])
        .args(args)
        .args(["-n", "out"])
        .assert()
        .success();
    let out = dir.join("out.csv");
    p2a()
        .args(["--session", session, "data", "save", "out", "--output"])
        .arg(&out)
        .assert()
        .success();
    let text = std::fs::read_to_string(&out).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let j = header
        .iter()
        .position(|h| *h == col)
        .unwrap_or_else(|| panic!("no column {col} in {header:?}"));
    lines
        .map(|l| {
            let cell = l.split(',').nth(j).unwrap();
            (!cell.is_empty()).then(|| cell.parse().unwrap())
        })
        .collect()
}

fn assert_column(got: &[Option<f64>], want: &[Option<f64>]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        match (g, w) {
            (Some(g), Some(w)) => assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}"),
            (None, None) => {}
            _ => panic!("{got:?} vs {want:?}"),
        }
    }
}

/// A grouped difference never subtracts the last value of one unit from the first of the next.
#[test]
fn diff_group_by_stays_within_units() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), PANEL);
    let got = munge_column(
        tmp.path(),
        &session,
        &["diff", "p", "-c", "x", "--group-by", "id"],
        "x_diff1",
    );
    assert_column(
        &got,
        &[
            None,
            Some(2.3 - 1.1),
            Some(3.9 - 2.3),
            None,
            Some(11.7 - 10.2),
            Some(13.4 - 11.7),
        ],
    );
}

/// With `--time`, the difference of period 4 is missing when period 3 is not in the data.
#[test]
fn diff_time_uses_the_time_index() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), PANEL);
    let got = munge_column(
        tmp.path(),
        &session,
        &["diff", "p", "-c", "x", "--group-by", "id", "--time", "t"],
        "x_diff1",
    );
    assert_column(
        &got,
        &[
            None,
            Some(2.3 - 1.1),
            None,
            None,
            Some(11.7 - 10.2),
            Some(13.4 - 11.7),
        ],
    );
}

/// The grouped, time-indexed percent change has the same missing pattern as the difference.
#[test]
fn pct_change_takes_group_by_and_time() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), PANEL);
    let got = munge_column(
        tmp.path(),
        &session,
        &[
            "diff",
            "p",
            "-c",
            "x",
            "--pct",
            "--group-by",
            "id",
            "--time",
            "t",
        ],
        "x_pct_change1",
    );
    assert_column(
        &got,
        &[
            None,
            Some((2.3 - 1.1) / 1.1),
            None,
            None,
            Some((11.7 - 10.2) / 10.2),
            Some((13.4 - 11.7) / 11.7),
        ],
    );
}

/// `lag --time`: the lag of period 4 is period 3, which unit 1 does not have.
#[test]
fn lag_time_uses_the_time_index() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), PANEL);
    let got = munge_column(
        tmp.path(),
        &session,
        &["lag", "p", "-c", "x", "--group-by", "id", "--time", "t"],
        "x_lag1",
    );
    assert_column(&got, &[None, Some(1.1), None, None, Some(10.2), Some(11.7)]);
}

/// `lead --time`: the lead of period 2 of unit 1 is period 3, which it does not have.
#[test]
fn lead_time_uses_the_time_index() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), PANEL);
    let got = munge_column(
        tmp.path(),
        &session,
        &["lead", "p", "-c", "x", "--group-by", "id", "--time", "t"],
        "x_lead1",
    );
    assert_column(&got, &[Some(2.3), None, None, Some(11.7), Some(13.4), None]);
}

/// A grouped forward fill does not carry unit 1's last value into unit 2.
#[test]
fn fill_na_group_by_stays_within_units() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), "id,x\n1,1.5\n1,\n2,\n2,4.2\n2,\n");
    let got = munge_column(
        tmp.path(),
        &session,
        &[
            "fill-na",
            "p",
            "-m",
            "forward",
            "-c",
            "x",
            "--group-by",
            "id",
        ],
        "x",
    );
    assert_column(&got, &[Some(1.5), Some(1.5), None, Some(4.2), Some(4.2)]);
}

/// `--group-by` applies only to forward and backward fills; another method is refused.
#[test]
fn fill_na_group_by_refuses_other_methods() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with(tmp.path(), "id,x\n1,1.5\n1,\n2,\n2,4.2\n");
    p2a()
        .args(["--session", &session, "munge", "fill-na", "p", "-m", "mean"])
        .args(["--group-by", "id", "-n", "out"])
        .assert()
        .code(1)
        .stderr(predicates::str::contains("Fill NA failed: "))
        .stderr(predicates::str::contains("forward and backward"));
}
