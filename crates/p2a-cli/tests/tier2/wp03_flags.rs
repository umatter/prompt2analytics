//! T2-WP03: global flags, subcommand short flags, session recording, data save/load through the p2a CLI.
use crate::common::*;

/// `p2a --session <path> -F json ts arima <args…>`: asserts success and parses the JSON object.
///
/// Skips stdout lines before the object: the arima crate's L-BFGS (liblbfgs 0.1.0,
/// `line.rs:228`) prints a line-search error to stdout on this series. T2-WP16 replaces that
/// optimiser and pins the ARIMA output with the strict `TempSession::json`; here only the
/// flag binding matters.
fn arima_json(s: &TempSession, args: &[&str]) -> serde_json::Value {
    let mut all = vec!["-F", "json", "ts", "arima"];
    all.extend_from_slice(args);
    let stdout = s.ok(&all);
    let start = if stdout.starts_with('{') {
        0
    } else {
        stdout.find("\n{").map_or(0, |i| i + 1)
    };
    serde_json::from_str(&stdout[start..])
        .unwrap_or_else(|e| panic!("{all:?}: no JSON object on stdout ({e})\n{stdout}"))
}

#[test]
fn test_tier2_cli_helpers() {
    let out = p2a().arg("--version").output().unwrap();
    assert!(out.status.success());
    let s = temp_session();
    assert_eq!(s.json(&["data", "list"])["message"], "No datasets loaded");
    assert!(
        s.path().exists(),
        "the session file is written by the first command"
    );
}

/// CLI-12: these nine subcommands panicked (exit 101) in every debug build, even for --help.
#[test]
fn test_cli_short_flags_bind_to_subcommands_in_debug_build() {
    const HELP: [(&[&str], &str); 10] = [
        (&["munge", "filter"], "-v, --value <VALUE>"),
        (&["munge", "mutate"], "-N, --name <NAME>"),
        (&["munge", "mutate"], "-n, --new-col <NEW_COL>"),
        (&["munge", "pivot"], "-v, --values <VALUES>"),
        (&["munge", "melt"], "-v, --value-vars <VALUE_VARS>"),
        (&["panel", "unit-root"], "-v, --var <VAR>"),
        (&["stats", "friedman"], "-v, --value <VALUE>"),
        (&["ts", "arima"], "-q, --ma <MA>"),
        (&["ts", "varma"], "-q, --ma <MA>"),
        (&["ts", "garch"], "-q, --garch <GARCH>"),
    ];
    for (cmd, line) in HELP {
        let out = p2a().args(cmd).arg("--help").output().unwrap();
        assert_no_panic(&out, &cmd.join(" "));
        let help = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && help.contains(line),
            "{cmd:?}: missing {line:?}\n{help}"
        );
        assert!(
            help.contains("--quiet")
                && !help.contains("-q, --quiet")
                && !help.contains("-v, --verbose"),
            "{help}"
        );
    }
    // A real fit with the MA order given as -q, in a debug build.
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/gdp_quarterly.csv"),
        "gdp",
        &[],
    );
    let fit = arima_json(
        &s,
        &[
            "gdp",
            "--col",
            "gdp_growth",
            "-p",
            "1",
            "-d",
            "0",
            "-q",
            "2",
        ],
    );
    assert_eq!(fit["model"], "ARIMA(1,0,2)"); // 2 is not the default (1), so -q bound to --ma
}

/// CLI-2: every `data save` panicked (exit 101, in release builds too) because the subcommand's
/// `format` arg shared its id with the global --format.
#[test]
fn test_cli_data_save_file_format() {
    let s = temp_session();
    let g = repo_path("validation/datasets/grunfeld.csv");
    s.load_as(&g, "grunfeld", &[]);
    let p = |f: &str| s.dir().join(f);
    s.ok(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g.dat").to_str().unwrap(),
        "--file-format",
        "csv",
    ]);
    assert_same_csv(&p("g.dat"), &g);
    s.ok(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g.csv").to_str().unwrap(),
    ]); // format from the extension
    assert_same_csv(&p("g.csv"), &g);
    // -F after the subcommand is the global output format (was "unexpected argument '-F'")
    let out = s.run(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g2.csv").to_str().unwrap(),
        "-F",
        "json",
    ]);
    assert_no_panic(&out, "data save -F json");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    // R: nrow(read.csv("grunfeld.csv")) = 200, ncol = 5
    assert_eq!(
        (
            j["format"].as_str(),
            j["rows"].as_u64(),
            j["columns"].as_u64()
        ),
        (Some("CSV"), Some(200), Some(5))
    );
    for bad in ["g.unknown", "g_no_extension"] {
        let e = s.err(&[
            "data",
            "save",
            "grunfeld",
            "--output",
            p(bad).to_str().unwrap(),
        ]);
        assert!(e.contains("--file-format"), "{bad}: {e}");
    }
}
