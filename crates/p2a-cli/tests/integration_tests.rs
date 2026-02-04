//! Integration tests for p2a CLI.
//!
//! These tests verify end-to-end functionality of the CLI binary.

use assert_cmd::Command;
use predicates::prelude::*;

/// Get a Command instance for the p2a CLI binary.
fn p2a() -> Command {
    Command::cargo_bin("p2a").expect("Failed to find p2a binary")
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

/// Test data head command.
#[test]
fn test_data_head_command() {
    // Create a temporary CSV file with .csv suffix
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let csv_path = temp_dir.path().join("head_test.csv");

    let mut content = String::from("a,b\n");
    for i in 0..20 {
        content.push_str(&format!("{},{}\n", i, i * 2));
    }
    std::fs::write(&csv_path, content).expect("Failed to write CSV");

    p2a()
        .args(["data", "head", csv_path.to_str().unwrap(), "-n", "5"])
        .assert()
        .success();
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
