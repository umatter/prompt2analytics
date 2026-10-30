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
    // the format name and the extension are case-insensitive, as in `data load`
    s.ok(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g3.dat").to_str().unwrap(),
        "--file-format",
        "CSV",
    ]);
    assert_same_csv(&p("g3.dat"), &g);
    s.ok(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g4.CSV").to_str().unwrap(),
    ]);
    assert_same_csv(&p("g4.CSV"), &g);
    // clap refuses an unknown --file-format (usage error, exit 2) and lists the allowed values
    let out = s.run(&[
        "data",
        "save",
        "grunfeld",
        "--output",
        p("g.xlsx").to_str().unwrap(),
        "--file-format",
        "xlsx",
    ]);
    assert_no_panic(&out, "data save --file-format xlsx");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.code() == Some(2) && stderr.contains("[possible values: csv, parquet, json]"),
        "{:?}\n{stderr}",
        out.status.code()
    );
    assert!(!p("g.xlsx").exists());
}

/// CLI-12: a recorded command keeps its subcommand flags (-q, -v) and a global typed after the
/// subcommand (--quiet), and the exported script replays with the real binary.
#[cfg(unix)]
#[test]
fn test_cli_session_replays_subcommand_flags() {
    let s = temp_session();
    let gdp_path = repo_path("validation/datasets/gdp_quarterly.csv");
    let gdp = gdp_path.to_str().unwrap();
    let copy_path = s.dir().join("gdp_copy.dat");
    let copy = copy_path.to_str().unwrap();
    s.ok(&["data", "load", gdp, "--name", "gdp"]);
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
            "--quiet",
        ],
    );
    assert_eq!(fit["model"], "ARIMA(1,0,2)");
    let filt = s.json(&[
        "munge", "filter", "gdp", "-c", "year", "-o", "ge", "-v", "2000", "-n", "recent",
    ]);
    // R: sum(read.csv("gdp_quarterly.csv")$year >= 2000) = 96, ncol = 4
    assert_eq!(
        filt["message"],
        "Created dataset 'recent' (96 rows x 4 columns)"
    );
    s.ok(&[
        "data",
        "save",
        "gdp",
        "--output",
        copy,
        "--file-format",
        "csv",
    ]);
    let want: Vec<Vec<String>> = [
        vec!["data", "load", gdp, "--name", "gdp"],
        vec![
            "ts",
            "arima",
            "gdp",
            "--col",
            "gdp_growth",
            "-p",
            "1",
            "-d",
            "0",
            "-q",
            "2",
            "--quiet",
        ],
        vec![
            "munge", "filter", "gdp", "-c", "year", "-o", "ge", "-v", "2000", "-n", "recent",
        ],
        vec![
            "data",
            "save",
            "gdp",
            "--output",
            copy,
            "--file-format",
            "csv",
        ],
    ]
    .iter()
    .map(|v| v.iter().map(|t| t.to_string()).collect())
    .collect();
    assert_eq!(s.recorded_argv(), want);
    std::fs::remove_file(&copy_path).unwrap();
    let script = s.dir().join("replay.sh");
    let out = p2a()
        .args([
            "script",
            "export",
            s.path().to_str().unwrap(),
            "-o",
            script.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = s.replay(&script);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "replay failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("Created dataset 'recent' (96 rows x 4 columns)")
            && stdout.contains("Script completed successfully"),
        "{stdout}"
    );
    assert_same_csv(&copy_path, &gdp_path); // the replayed `data save` wrote the file again
}

/// DATA-32 (CLI): "NA" is missing by default (R write.csv), --na-strings replaces the marker list,
/// and the markers persist: every call below is a new process that reloads the session's datasets.
#[test]
fn test_validate_cli_data_load_na_strings() {
    use std::collections::BTreeMap;
    let case = Case::load("columns_conventions");
    let s = temp_session();
    let canon = case.dataset_path();
    let text = std::fs::read_to_string(&canon).unwrap();
    let derive = |marker: &str, file: &str| {
        let mut out = String::new();
        for line in text.lines() {
            let f: Vec<&str> = line
                .split(',')
                .map(|c| if c.is_empty() { marker } else { c })
                .collect();
            assert_eq!(f.len(), 5, "case dataset changed shape: {line:?}");
            out.push_str(&f.join(","));
            out.push('\n');
        }
        let p = s.dir().join(file);
        std::fs::write(&p, out).unwrap();
        p
    };
    let (na, dot) = (derive("NA", "cc_na.csv"), derive(".", "cc_dot.csv"));
    s.load_as(&canon, "canon", &[]);
    s.load_as(&na, "na", &[]);
    s.load_as(&dot, "dot", &["--na-strings", "."]);
    s.load_as(&dot, "raw", &[]);
    let dtypes = |name: &str| -> BTreeMap<String, String> {
        s.json_docs(&["data", "describe", name])[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["name"].as_str().unwrap().to_string(),
                    c["dtype"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    };
    let map = |pairs: [(&str, &str); 5]| {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect::<BTreeMap<_, _>>()
    };
    // R classes integer, numeric, integer, character, logical
    let typed = map([
        ("id", "Int64"),
        ("y", "Float64"),
        ("n", "Int64"),
        ("s", "String"),
        ("flag", "Boolean"),
    ]);
    for name in ["canon", "na", "dot"] {
        assert_eq!(dtypes(name), typed, "{name}");
    }
    // R read.csv() without na.strings: "." is text, columns are character
    assert_eq!(
        dtypes("raw"),
        map([
            ("id", "Int64"),
            ("y", "String"),
            ("n", "String"),
            ("s", "String"),
            ("flag", "String")
        ])
    );
    let head = |name: &str| s.json(&["data", "head", name, "-n", "10"]);
    let rows = head("canon");
    assert_eq!(head("na"), rows); // R: identical(d_na, canon)
    assert_eq!(head("dot"), rows); // R: identical(d_dot, canon)
    let nulls = |rows: &serde_json::Value, col: &str| -> Vec<usize> {
        rows.as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, r)| r[col] == "null")
            .map(|(i, _)| i)
            .collect()
    };
    let r_rows = |key: &str, missing: fn(f64) -> bool| -> Vec<usize> {
        case.vector(key)
            .iter()
            .enumerate()
            .filter(|(_, v)| missing(**v))
            .map(|(i, _)| i)
            .collect()
    };
    assert_eq!(nulls(&rows, "n"), r_rows("n_codes", |v| v == -1.0));
    assert_eq!(nulls(&rows, "s"), r_rows("s_codes", |v| v == -1.0));
    assert_eq!(nulls(&rows, "flag"), r_rows("flag_numeric", f64::is_nan));
    assert_eq!(nulls(&rows, "y").len() as f64, case.scalar("missing_y"));
    let complete: Vec<f64> = (0..10)
        .filter(|&i| ["y", "n", "s"].iter().all(|c| rows[i][*c] != "null"))
        .map(|i| i as f64)
        .collect();
    assert_eq!(complete, case.vector("complete_rows"));
    let raw = head("raw");
    for col in ["id", "y", "n", "s", "flag"] {
        assert!(nulls(&raw, col).is_empty(), "{col}");
    }
    // the recorded command keeps the flag for script export
    let dot_s = dot.to_str().unwrap();
    assert!(
        s.recorded_argv().contains(
            &["data", "load", dot_s, "--name", "dot", "--na-strings", "."]
                .map(String::from)
                .to_vec()
        )
    );
    // --na-strings is a CSV option; other formats store missing values natively
    let pq = s.dir().join("canon.parquet");
    s.ok(&["data", "save", "canon", "--output", pq.to_str().unwrap()]);
    let e = s.err(&[
        "data",
        "load",
        pq.to_str().unwrap(),
        "--name",
        "pq",
        "--na-strings",
        ".",
    ]);
    assert!(e.contains("--na-strings") && e.contains("CSV"), "{e}");
}
