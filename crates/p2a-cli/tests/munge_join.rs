//! `p2a munge join`: argument parsing, error text and exit status.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

fn p2a() -> Command {
    Command::cargo_bin("p2a").expect("Failed to find p2a binary")
}

/// A session holding `cs` (id, x, y) and `cs2` (id, z, code).
fn session_with_two_datasets(dir: &Path) -> String {
    let session = dir.join("s.json").to_str().unwrap().to_string();
    let files = [
        ("cs", "id,x,y\n1,1.1,2.3\n2,1.9,3.8\n3,3.2,6.1\n4,3.8,8.2\n"),
        ("cs2", "id,z,code\n1,0.4,a\n2,0.9,b\n3,1.6,c\n5,2.2,d\n"),
    ];
    for (name, csv) in files {
        let path = dir.join(format!("{name}.csv"));
        std::fs::write(&path, csv).unwrap();
        p2a()
            .args(["--session", &session, "data", "load"])
            .arg(&path)
            .args(["--name", name])
            .assert()
            .success();
    }
    session
}

/// Without `--on` there is nothing to join on: clap refuses the command with
/// its usage error (exit 2) naming the flag, before any dataset is touched.
#[test]
fn join_without_on_is_a_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with_two_datasets(tmp.path());
    p2a()
        .args([
            "--session",
            &session,
            "munge",
            "join",
            "cs",
            "cs2",
            "-n",
            "o",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--on"))
        .stderr(predicate::str::contains("left_on").not());
}

/// A refused join says "Join failed:" once, in text and JSON output, and the
/// process exits 1. A key refusal names the CLI's flags and the datasets the
/// user typed, never the core's `left_on`/`right_on` parameters.
#[test]
fn join_errors_say_join_failed_once_and_exit_nonzero() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with_two_datasets(tmp.path());
    // (keys, expected text, whether the CLI words the refusal itself)
    let cases: [(&[&str], &str, bool); 5] = [
        (
            &["--on", "id", "--right-on", "id", "z"],
            "Join failed: --on has 1 columns but --right-on has 2 columns",
            true,
        ),
        // A missing key names its flag and the dataset that lacks it.
        (
            &["--on", "nope"],
            "Join failed: --on column 'nope' is not in dataset 'cs'",
            true,
        ),
        (
            &["--on", "id", "--right-on", "nope"],
            "Join failed: --right-on column 'nope' is not in dataset 'cs2'",
            true,
        ),
        (
            &["--on", "x"],
            "Join failed: --on column 'x' is not in dataset 'cs2' (no --right-on is given, \
             so the --on names are used for both datasets)",
            true,
        ),
        // polars refuses keys of different types; the core wraps its error,
        // whose wording is polars' own.
        (
            &["--on", "id", "--right-on", "code"],
            "Join failed: ",
            false,
        ),
    ];
    for (keys, want, cli_worded) in cases {
        for format in ["text", "json"] {
            let mut assert = p2a()
                .args(["--format", format, "--session", &session])
                .args(["munge", "join", "cs", "cs2", "-n", "o"])
                .args(keys)
                .assert()
                .code(1);
            if cli_worded {
                assert = assert
                    .stderr(predicate::str::contains("left_on").not())
                    .stderr(predicate::str::contains("right_on").not());
            }
            let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
            assert!(stderr.contains(want), "{keys:?} {format}: {stderr}");
            assert_eq!(
                stderr.matches("Join failed").count(),
                1,
                "{keys:?} {format}: {stderr}"
            );
        }
    }
}

/// Control: a valid join still succeeds.
#[test]
fn join_on_a_shared_key_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let session = session_with_two_datasets(tmp.path());
    p2a()
        .args(["--session", &session, "munge", "join", "cs", "cs2"])
        .args(["--on", "id", "-t", "inner", "-n", "o"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Created dataset 'o' (3 rows x 5 columns)",
        ));
}
