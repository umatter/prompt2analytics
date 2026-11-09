//! T2-WP03: global flags, subcommand short flags, session recording, data save/load through the p2a CLI.
use crate::common::*;

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
    let fit = s.json(&[
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
    ]);
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
    // grunfeld.csv has 200 rows and 5 columns
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
    let fit = s.json(&[
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
    ]);
    assert_eq!(fit["model"], "ARIMA(1,0,2)");
    let filt = s.json(&[
        "munge", "filter", "gdp", "-c", "year", "-o", "ge", "-v", "2000", "-n", "recent",
    ]);
    // 96 rows of gdp_quarterly.csv have year >= 2000; it has 4 columns
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

/// R's class of a column, coded as the case `cli_na_strings` codes it (integer 0, numeric 1,
/// character 2, logical 3), for the polars dtype p2a reports; NaN for any other dtype.
fn r_class_code(dtype: &str) -> f64 {
    match dtype {
        "Int64" => 0.0,
        "Float64" => 1.0,
        "String" => 2.0,
        "Boolean" => 3.0,
        _ => f64::NAN,
    }
}

#[test]
fn test_cli_na_strings_refused_for_any_non_csv_file() {
    // The --na-strings guard covers every extension but csv, also one the loader does not
    // read, so the refusal names the flag rather than the format.
    let s = temp_session();
    let txt = s.dir().join("t.txt");
    std::fs::write(&txt, "x\n1\n.\n").unwrap();
    let e = s.err(&[
        "data",
        "load",
        txt.to_str().unwrap(),
        "--name",
        "t",
        "--na-strings",
        ".",
    ]);
    assert!(e.contains("--na-strings") && e.contains("CSV"), "{e}");
    assert!(e.contains("'txt'"), "{e}");
    // without the flag the format itself is refused, with the extensions the build reads
    let e = s.err(&["data", "load", txt.to_str().unwrap(), "--name", "t"]);
    assert!(
        e.contains("Unsupported file format") && e.contains("'txt'") && e.contains("csv"),
        "{e}"
    );
}

/// DATA-32 (CLI): "NA" is missing by default (R write.csv), --na-strings replaces the marker list,
/// and the markers persist: every call below is a new process that reloads the session's datasets.
/// The case `cli_na_strings` reads the same "NA"- and "."-coded copies of the case file in R.
#[test]
fn test_validate_cli_data_load_na_strings() {
    let case = Case::load("columns_conventions");
    let read = Case::load("cli_na_strings");
    let n = case.vector("n_codes").len();
    let s = temp_session();
    let canon = case.dataset_path();
    let text = std::fs::read_to_string(&canon).unwrap();
    // The case file with every empty field replaced by `marker`, as the R case derives it.
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
    s.load_as(&dot, "raw", &[]); // default markers: "." is text
    // (column name, dtype) in column order, and the first n rows, of a loaded dataset
    let dtypes = |name: &str| -> Vec<(String, String)> {
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
    let head = |name: &str| {
        let rows = s.json(&["data", "head", name, "-n", &n.to_string()]);
        assert_eq!(rows.as_array().unwrap().len(), n, "{name}: {rows}");
        rows
    };
    let nulls = |rows: &serde_json::Value, col: &str| -> Vec<usize> {
        rows.as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, r)| r[col] == "null")
            .map(|(i, _)| i)
            .collect()
    };
    let loads: Vec<(&str, Vec<(String, String)>, serde_json::Value)> =
        ["canon", "na", "dot", "raw"]
            .into_iter()
            .map(|name| (name, dtypes(name), head(name)))
            .collect();
    let (_, canon_types, rows) = &loads[0];

    // R: column classes and missing cells per column of each read, and identical() to the case
    // file for the coded copies (p2a: the same dtypes and the same cells as `canon`).
    let mut c = Check::new(&read);
    for (name, types, head_rows) in &loads {
        let cols: Vec<&str> = types.iter().map(|(col, _)| col.as_str()).collect();
        let classes: Vec<f64> = types.iter().map(|(_, t)| r_class_code(t)).collect();
        let missing: Vec<f64> = cols
            .iter()
            .map(|col| nulls(head_rows, col).len() as f64)
            .collect();
        c.named(&format!("{name}_class"), &cols, &classes, Tol::exact())
            .named(&format!("{name}_missing"), &cols, &missing, Tol::exact());
        if matches!(*name, "na" | "dot") {
            let same = types == canon_types && head_rows == rows;
            if !same {
                eprintln!(
                    "{name} differs from canon:\n{types:?}\n{head_rows}\ncanon:\n{canon_types:?}\n{rows}"
                );
            }
            c.scalar(
                &format!("{name}_identical"),
                f64::from(u8::from(same)),
                Tol::exact(),
            );
        }
    }
    c.finish();

    // R: the rows complete.cases(y, n, s) keeps and the missing cells of y, n and s, for the
    // case file and both coded copies.
    for (name, _, head_rows) in &loads[..3] {
        let complete: Vec<f64> = (0..n)
            .filter(|&i| {
                ["y", "n", "s"]
                    .iter()
                    .all(|col| head_rows[i][*col] != "null")
            })
            .map(|i| i as f64)
            .collect();
        eprintln!("columns_conventions on the load `{name}`:");
        let mut c = Check::new(&case);
        c.vector("complete_rows", &complete, Tol::exact()).scalar(
            "n_dropped",
            (n - complete.len()) as f64,
            Tol::exact(),
        );
        for col in ["y", "n", "s"] {
            c.scalar(
                &format!("missing_{col}"),
                nulls(head_rows, col).len() as f64,
                Tol::exact(),
            );
        }
        c.finish();
    }
    // R: the 0-based rows of the missing cells of each column of the case file
    let mut c = Check::new(&read);
    for col in ["y", "n", "s", "flag"] {
        let positions: Vec<f64> = nulls(rows, col).into_iter().map(|i| i as f64).collect();
        c.vector(&format!("canon_na_rows_{col}"), &positions, Tol::exact());
    }
    c.finish();

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
