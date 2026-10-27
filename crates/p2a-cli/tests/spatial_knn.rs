//! `p2a spatial`: the k-nearest-neighbour weights refuse a `k` outside
//! 1..=n-1 (and non-finite coordinates) with exit status 1, as the MCP
//! `spatial_neighbors` tool does, instead of clamping `k` to n - 1.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

fn p2a() -> Command {
    Command::cargo_bin("p2a").expect("Failed to find p2a binary")
}

/// A session holding `pts`: six points (`cx`, `cy`) with an outcome `v` and a
/// regressor `x`, all with noise.
fn session_with_points(dir: &Path, csv: &str) -> String {
    let session = dir.join("s.json").to_str().unwrap().to_string();
    let path = dir.join("pts.csv");
    std::fs::write(&path, csv).unwrap();
    p2a()
        .args(["--session", &session, "data", "load"])
        .arg(&path)
        .args(["--name", "pts"])
        .assert()
        .success();
    session
}

const POINTS: &str = "cx,cy,v,x\n\
                      0.1,0.2,1.3,0.5\n\
                      1.2,0.1,2.1,1.1\n\
                      0.3,1.1,1.7,0.9\n\
                      1.1,1.3,2.9,1.6\n\
                      2.2,0.4,2.4,2.3\n\
                      2.1,1.6,3.8,2.8\n";

/// The arguments of each spatial command up to `-k`, for dataset `pts`.
fn commands() -> [Vec<&'static str>; 4] {
    let coords = ["--coord-x", "cx", "--coord-y", "cy"];
    let model = |name| {
        let mut v = vec!["spatial", name, "pts", "-y", "v", "-x", "x"];
        v.extend(coords);
        v
    };
    let mut moran = vec!["spatial", "moran", "pts", "-y", "v"];
    moran.extend(coords);
    [model("sar"), model("sem"), model("sac"), moran]
}

/// k = n, k = n + 1 and k = 0 are refused by every spatial command, in text
/// and JSON output, with the core's message and exit status 1. Before, k = n
/// was clamped to n - 1 without a word and k = 0 gave every point no
/// neighbours.
#[test]
fn k_outside_one_to_n_minus_one_is_refused_with_exit_1() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with_points(tmp.path(), POINTS);
    for args in commands() {
        for k in ["6", "7", "0"] {
            let want = format!("k must be between 1 and n - 1 = 5, got {k}");
            p2a()
                .args(["--session", &session])
                .args(&args)
                .args(["-k", k])
                .assert()
                .code(1)
                .stderr(predicate::str::contains("Failed to build spatial weights"))
                .stderr(predicate::str::contains(want.as_str()));
            p2a()
                .args(["--session", &session, "--format", "json"])
                .args(&args)
                .args(["-k", k])
                .assert()
                .code(1)
                .stderr(predicate::str::starts_with("{\"error\":"))
                .stderr(predicate::str::contains(want.as_str()));
        }
    }
}

/// k = n - 1, the largest valid value, still runs.
#[test]
fn k_of_n_minus_one_still_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with_points(tmp.path(), POINTS);
    p2a()
        .args(["--session", &session, "--format", "json"])
        .args(&commands()[3])
        .args(["-k", "5"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"moran_i\""));
}

/// The `-k` help of every spatial command gives the valid range and says
/// what its n is.
#[test]
fn k_help_names_the_range_and_what_n_is() {
    for cmd in ["sar", "sem", "sac", "moran"] {
        p2a()
            .args(["spatial", cmd, "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains(
                "1 to n - 1 where n is the number of rows",
            ));
    }
}

/// A coordinate that is not finite is refused, as in MCP; before, the legacy
/// constructor fell back to an all-pairs search over the NaN distances.
#[test]
fn a_non_finite_coordinate_is_refused_with_exit_1() {
    let tmp = tempfile::tempdir().unwrap();
    let csv = POINTS.replace("2.2,0.4", "NaN,0.4");
    let session = session_with_points(tmp.path(), &csv);
    p2a()
        .args(["--session", &session])
        .args(&commands()[3])
        .args(["-k", "2"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("Failed to build spatial weights"))
        .stderr(predicate::str::contains(
            "coordinates of observation 5 are not finite",
        ));
}
