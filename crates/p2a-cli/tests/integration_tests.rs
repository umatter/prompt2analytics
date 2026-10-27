//! Integration tests for p2a CLI.
//!
//! These tests verify end-to-end functionality of the CLI binary.

use assert_cmd::Command;
use predicates::prelude::*;

/// Get a Command instance for the p2a CLI binary.
fn p2a() -> Command {
    Command::cargo_bin("p2a").expect("Failed to find p2a binary")
}

/// Every non-comment line must be one of the fixed script lines or one replayed command.
fn assert_only_intended_lines(script: &str, expected_commands: usize) {
    assert!(
        script.chars().all(|c| c == '\n' || !c.is_control()),
        "raw control character:\n{script}"
    );
    let mut commands = 0;
    for line in script.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t.starts_with("p2a --session \"$SESSION_FILE\" ") {
            commands += 1;
            continue;
        }
        assert!(
            matches!(
                t,
                "set -euo pipefail"
                    | "rm -f \"$SESSION_FILE\""
                    | "echo \"Script completed successfully\""
            ) || t.starts_with("SESSION_FILE=\".p2a_session_"),
            "unexpected executable line {line:?} in:\n{script}"
        );
    }
    assert_eq!(commands, expected_commands, "{script}");
}

/// Replay an exported script under bash with a stub `p2a` that logs its argv (NUL-separated).
#[cfg(unix)]
fn replay_with_stub(dir: &std::path::Path, script: &std::path::Path) -> Vec<String> {
    use std::os::unix::fs::PermissionsExt;
    let bin = dir.join("stub_bin");
    std::fs::create_dir_all(&bin).unwrap();
    let stub = bin.join("p2a");
    std::fs::write(
        &stub,
        "#!/bin/bash\nprintf '%s\\0' \"$@\" >> \"$P2A_STUB_LOG\"\n",
    )
    .unwrap();
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    let log = dir.join("stub.log");
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let status = std::process::Command::new("bash")
        .arg(script)
        .current_dir(dir)
        .env("PATH", path)
        .env("P2A_STUB_LOG", &log)
        .status()
        .expect("bash");
    assert!(status.success());
    let raw = std::fs::read(&log).unwrap_or_default();
    raw.split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect()
}

/// Test that the help command works.
#[test]
fn test_help_command() {
    p2a()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Command-line interface for prompt2analytics",
        ))
        .stdout(predicate::str::contains("EXAMPLES"));
}

/// Test that subcommand help works.
#[test]
fn test_subcommand_help() {
    p2a()
        .args(["data", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Data loading and inspection"));
}

/// Test that the version command works.
#[test]
fn test_version_command() {
    p2a()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("p2a"));
}

/// Test the smoke-test command.
#[test]
fn test_smoke_test_command() {
    p2a()
        .arg("smoke-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke test PASSED"));
}

/// Test smoke-test with JSON format.
#[test]
fn test_smoke_test_json_output() {
    p2a()
        .args(["--format", "json", "smoke-test"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"status\": \"success\""));
}

/// Test invalid command fails gracefully.
#[test]
fn test_invalid_command() {
    p2a()
        .arg("nonexistent-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

/// Test regression subcommand help.
#[test]
fn test_regression_help() {
    p2a()
        .args(["regression", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ols"))
        .stdout(predicate::str::contains("Regression analysis"));
}

/// Test stats subcommand help.
#[test]
fn test_stats_help() {
    p2a()
        .args(["stats", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Statistical tests"));
}

/// Test panel subcommand help.
#[test]
fn test_panel_help() {
    p2a()
        .args(["panel", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Panel data estimation"));
}

/// Test causal subcommand help.
#[test]
fn test_causal_help() {
    p2a()
        .args(["causal", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Causal inference"));
}

/// Test ml subcommand help.
#[test]
fn test_ml_help() {
    p2a()
        .args(["ml", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Machine learning"));
}

/// Test data load command with non-existent file.
#[test]
fn test_data_load_nonexistent_file() {
    p2a()
        .args(["data", "load", "nonexistent_file.csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error"));
}

/// Test data load with valid CSV file.
#[test]
fn test_data_load_valid_csv() {
    // Create a temporary CSV file with .csv suffix
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let csv_path = temp_dir.path().join("test_data.csv");

    std::fs::write(&csv_path, "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n")
        .expect("Failed to write CSV");

    p2a()
        .args(["data", "load", csv_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Rows: 3"))
        .stdout(predicate::str::contains("Columns: 3"));
}

/// A parquet file whose schema the core loader refuses is an error naming
/// the column, with exit code 1, not a panic (exit code 101). The file is
/// p2a-core's fixture of a list of int32-coded dictionaries carrying
/// polars' Categorical metadata, which polars' nested decoder leaves
/// unimplemented (see `a_nested_dictionary_with_polars_metadata_and_signed_codes_is_refused`
/// in p2a-core's loader).
#[test]
fn test_data_load_parquet_polars_cannot_decode_is_an_error() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../p2a-core/tests/data/polars_metadata_signed_list.parquet");
    p2a()
        .args(["data", "load", fixture.to_str().unwrap()])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("Cannot read column 'k'"))
        .stderr(predicate::str::contains("panicked").not());
}

/// A parquet file the core loader's schema check passes but polars panics
/// on is an error too, with exit code 1, not a panic (exit code 101): the
/// loader catches the unwind. The file is p2a-core's fixture of a float
/// column whose ARROW:schema is too short to hold its own header (see
/// `a_panic_in_polars_no_schema_check_rules_out_is_an_error` in p2a-core's
/// loader). The default panic hook still prints polars' panic to stderr
/// before the error, so stderr is not checked for its absence.
#[test]
fn test_data_load_parquet_polars_panics_on_is_an_error() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../p2a-core/tests/data/arrow_schema_short.parquet");
    p2a()
        .args(["data", "load", fixture.to_str().unwrap()])
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "polars failed while reading the parquet file",
        ));
}

/// Test data head command.
#[test]
fn test_data_head_command() {
    // Create a temporary CSV file with .csv suffix
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let csv_path = temp_dir.path().join("head_test.csv");
    let session_path = temp_dir.path().join("session.json");

    let mut content = String::from("a,b\n");
    for i in 0..20 {
        content.push_str(&format!("{},{}\n", i, i * 2));
    }
    std::fs::write(&csv_path, content).expect("Failed to write CSV");

    // `data head` operates on a session-loaded dataset by name, so load first,
    // then head the loaded dataset (both within the same session).
    p2a()
        .args([
            "--session",
            session_path.to_str().unwrap(),
            "data",
            "load",
            csv_path.to_str().unwrap(),
            "--name",
            "head_data",
        ])
        .assert()
        .success();

    p2a()
        .args([
            "--session",
            session_path.to_str().unwrap(),
            "data",
            "head",
            "head_data",
            "-n",
            "5",
        ])
        .assert()
        .success();
}

/// A "soft" usage error (referencing a nonexistent dataset) must exit non-zero
/// so scripts and CI can detect the failure.
#[test]
fn test_soft_error_exits_nonzero() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let session_path = temp_dir.path().join("session.json");
    p2a()
        .args([
            "--session",
            session_path.to_str().unwrap(),
            "data",
            "describe",
            "does_not_exist",
        ])
        .assert()
        .failure();
}

/// Commands run under `--session` are recorded and reproduced by `script export`.
#[test]
fn test_session_records_and_exports() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let csv_path = temp_dir.path().join("rec.csv");
    std::fs::write(&csv_path, "x,y\n1,2\n3,4\n5,6\n").expect("write csv");
    let session_path = temp_dir.path().join("session.json");
    let script_path = temp_dir.path().join("out.sh");

    p2a()
        .args([
            "--session",
            session_path.to_str().unwrap(),
            "data",
            "load",
            csv_path.to_str().unwrap(),
            "--name",
            "rec",
        ])
        .assert()
        .success();
    p2a()
        .args([
            "--session",
            session_path.to_str().unwrap(),
            "data",
            "describe",
            "rec",
        ])
        .assert()
        .success();

    p2a()
        .args([
            "script",
            "export",
            session_path.to_str().unwrap(),
            "-o",
            script_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let exported = std::fs::read_to_string(&script_path).expect("read exported script");
    assert!(
        exported.contains("data load"),
        "exported script should contain the recorded load command:\n{exported}"
    );
    assert!(
        exported.contains("data describe rec"),
        "exported script should contain the recorded describe command:\n{exported}"
    );
}

/// A recorded argument containing a newline must stay inside its comment and its quoted argument.
#[test]
fn test_script_export_keeps_newline_argument_inert() {
    let tmp = tempfile::tempdir().unwrap();
    let session = tmp.path().join("s.json");
    let script = tmp.path().join("out.sh");
    let payload = "nodata\ntouch injected_marker #";
    p2a()
        .args([
            "--session",
            session.to_str().unwrap(),
            "data",
            "describe",
            payload,
        ])
        .assert()
        .failure(); // unknown dataset; still recorded
    p2a()
        .args([
            "script",
            "export",
            session.to_str().unwrap(),
            "-o",
            script.to_str().unwrap(),
        ])
        .assert()
        .success();
    let text = std::fs::read_to_string(&script).unwrap();
    assert_only_intended_lines(&text, 1);
    assert!(
        text.contains("# data describe 'nodata\\ntouch injected_marker #'\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "p2a --session \"$SESSION_FILE\" data describe $'nodata\\ntouch injected_marker #'\n"
        ),
        "{text}"
    );
    #[cfg(unix)]
    {
        let args = replay_with_stub(tmp.path(), &script);
        assert!(!tmp.path().join("injected_marker").exists());
        assert_eq!(args[2..], ["data", "describe", payload]); // the argument round-trips exactly
    }
}

/// Title, version, comment and legacy category/arguments from a crafted session file stay inert.
#[test]
fn test_script_export_neutralises_crafted_session_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let session = tmp.path().join("crafted.json");
    let script = tmp.path().join("out.sh");
    let json = serde_json::json!({
        "id": "crafted", "title": "t\ntouch injected_title",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "working_dir": "/tmp", "datasets": {}, "version": "0.1.3\ntouch injected_version",
        "commands": [{
            "id": "c1", "timestamp": "2026-01-01T00:00:00Z",
            "command_line": "legacy\ntouch injected_comment",
            "category": "data;touch injected_category;", "subcommand": "describe",
            "arguments": {"dataset": "d\ntouch injected_arg"},
            "dataset_refs": [], "output_files": [], "success": true, "error": null, "duration_ms": 0
        }]
    });
    std::fs::write(&session, serde_json::to_string(&json).unwrap()).unwrap();
    p2a()
        .args([
            "script",
            "export",
            session.to_str().unwrap(),
            "-o",
            script.to_str().unwrap(),
        ])
        .assert()
        .success();
    let text = std::fs::read_to_string(&script).unwrap();
    assert_only_intended_lines(&text, 1);
    assert!(text.contains("# t\\ntouch injected_title\n"));
    assert!(text.contains("# p2a version: 0.1.3\\ntouch injected_version\n"));
    assert!(text.contains("# legacy\\ntouch injected_comment\n"));
    assert!(
        text.contains(
            "p2a --session \"$SESSION_FILE\" 'data;touch injected_category;' describe $'d\\ntouch injected_arg'\n"
        ),
        "{text}"
    );
    #[cfg(unix)]
    {
        let args = replay_with_stub(tmp.path(), &script);
        for m in [
            "injected_title",
            "injected_version",
            "injected_comment",
            "injected_category",
            "injected_arg",
        ] {
            assert!(!tmp.path().join(m).exists(), "{m} was created");
        }
        assert_eq!(
            args[2..],
            [
                "data;touch injected_category;",
                "describe",
                "d\ntouch injected_arg"
            ]
        );
    }
}

/// Test that output formats can be specified.
#[test]
fn test_output_format_json() {
    p2a()
        .args(["--format", "json", "smoke-test"])
        .assert()
        .success()
        .stdout(predicate::str::contains("{"));
}

/// Test that output formats can be specified (table).
#[test]
fn test_output_format_table() {
    p2a()
        .args(["--format", "table", "smoke-test"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke test PASSED"));
}

/// Test munge subcommand help.
#[test]
fn test_munge_help() {
    p2a()
        .args(["munge", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Data munging"));
}

/// Test viz subcommand help.
#[test]
fn test_viz_help() {
    p2a()
        .args(["viz", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Visualization"));
}

/// Test timeseries subcommand help.
#[test]
fn test_timeseries_help() {
    p2a()
        .args(["timeseries", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Time series"));
}

/// Test spatial subcommand help.
#[test]
fn test_spatial_help() {
    p2a()
        .args(["spatial", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Spatial econometrics"));
}

/// Test discrete subcommand help.
#[test]
fn test_discrete_help() {
    p2a()
        .args(["discrete", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Discrete choice"));
}

/// Test survival subcommand help.
#[test]
fn test_survival_help() {
    p2a()
        .args(["survival", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Survival analysis"));
}

/// Test that the short format flag works.
#[test]
fn test_short_format_flag() {
    p2a()
        .args(["-F", "json", "smoke-test"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"status\": \"success\""));
}

/// Test the script subcommand help.
#[test]
fn test_script_help() {
    p2a()
        .args(["script", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Script generation"));
}
