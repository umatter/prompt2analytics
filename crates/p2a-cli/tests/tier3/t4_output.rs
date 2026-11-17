//! Tier 4 CLI output conventions: one JSON document per command, the shared `Warnings:` block,
//! R's significance marks, typed JSON cells.

use crate::common::*;
use std::path::Path;

/// A session with the datasets the commands below use, each named after its file stem.
fn session(datasets: &[&str]) -> TempSession {
    let s = temp_session();
    for rel in datasets {
        let p = repo_path(rel);
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        s.load_as(&p, &name, &[]);
    }
    s
}

const DATASETS: &[&str] = &[
    "validation/datasets/grunfeld.csv",
    "validation/datasets/generated/binary_glm.csv",
    "validation/datasets/generated/zeroinfl.csv",
    "validation/datasets/generated/lung.csv",
    "validation/datasets/generated/iv_weak_controls.csv",
    "validation/datasets/generated/ts_acf_spline_t3.csv",
    "validation/datasets/generated/hac_ar1.csv",
];

/// Commands across the categories, each of which must print exactly one JSON document.
const ONE_DOC: &[&[&str]] = &[
    &["data", "head", "grunfeld"],
    &["data", "describe", "grunfeld"],
    &["data", "list"],
    &[
        "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
    ],
    &["reg", "hac", "hac_ar1", "-y", "y", "-x", "x", "--lags", "2"],
    &[
        "panel", "fe", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
    ],
    &[
        "discrete",
        "logit",
        "binary_glm",
        "-y",
        "y",
        "-x",
        "x1",
        "x2",
    ],
    &["discrete", "zip", "zeroinfl", "-y", "y", "-x", "x1", "x2"],
    &[
        "causal",
        "iv",
        "iv_weak_controls",
        "-y",
        "y",
        "--endog",
        "x",
        "--instruments",
        "z1",
        "z2",
        "--exog",
        "w",
    ],
    &["stats", "fivenum", "grunfeld", "-c", "inv"],
    &["stats", "acf", "ts_acf_spline_t3", "-c", "x"],
    &["stats", "chisq-gof", "binary_glm", "-o", "d"],
    &[
        "survival", "km", "lung", "--time", "time", "--event", "status", "--group", "sex",
    ],
    &[
        "survival", "km", "lung", "--time", "time", "--event", "status",
    ],
];

/// Every command in [`ONE_DOC`] prints one JSON value on stdout.
#[test]
fn test_cli_commands_print_one_json_document() {
    let s = session(DATASETS);
    let mut problems = Vec::new();
    for args in ONE_DOC {
        let docs = s.json_docs(args);
        if docs.len() != 1 {
            problems.push(format!("{}: {} JSON documents", args.join(" "), docs.len()));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Strips comments, string and char literals from Rust source, keeping line breaks, so braces
/// can be counted.
fn strip(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if c == 'r' && (b.get(i + 1) == Some(&'"') || b.get(i + 1) == Some(&'#')) {
            let mut j = i + 1;
            let mut hashes = 0;
            while b.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if b.get(j) != Some(&'"') {
                out.push(c);
                i += 1;
                continue;
            }
            j += 1;
            loop {
                if j >= b.len() {
                    break;
                }
                if b[j] == '"' && (0..hashes).all(|k| b.get(j + 1 + k) == Some(&'#')) {
                    j += 1 + hashes;
                    break;
                }
                if b[j] == '\n' {
                    out.push('\n');
                }
                j += 1;
            }
            out.push_str("\"\"");
            i = j;
        } else if c == '"' {
            i += 1;
            while i < b.len() && b[i] != '"' {
                if b[i] == '\\' {
                    i += 1;
                }
                if b.get(i) == Some(&'\n') {
                    out.push('\n');
                }
                i += 1;
            }
            out.push_str("\"\"");
            i += 1;
        } else if c == '\''
            && b.get(i + 2) == Some(&'\'')
            && b.get(i + 1).is_some_and(|c| "{}".contains(*c))
        {
            out.push_str("' '");
            i += 3;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The JSON print sites of `src` (lines that print `to_string_pretty`, `print_json` or a
/// `format_dataset_summary`), each with the stack of blocks it sits in (block = the line where
/// the brace opened) and whether one of them is a loop. Two sites of which one's block stack is
/// a prefix of the other's can both run in one invocation; a site in a loop can run twice.
fn json_problems(file: &str, raw: &str) -> Vec<String> {
    let src = strip(raw);
    let raw_lines: Vec<&str> = raw.lines().collect();
    let mut problems = Vec::new();
    // (line opened, is loop)
    let mut stack: Vec<(usize, bool)> = Vec::new();
    // sites of the current top-level fn: (line, stack)
    let mut sites: Vec<(usize, Vec<(usize, bool)>)> = Vec::new();
    let mut fn_depth: Option<usize> = None;
    let mut fn_name = String::new();
    for (ln, line) in src.lines().enumerate() {
        let t = line.trim_start();
        let annotated = |l: usize| {
            raw_lines.get(l).is_some_and(|r| r.contains("json-ok:"))
                || (l > 0 && raw_lines[l - 1].contains("json-ok:"))
        };
        if fn_depth.is_none()
            && let Some(rest) = t
                .strip_prefix("pub fn ")
                .or_else(|| t.strip_prefix("fn "))
                .or_else(|| t.strip_prefix("pub(crate) fn "))
        {
            fn_name = rest.split(['(', '<']).next().unwrap_or("").to_string();
            fn_depth = Some(stack.len());
            sites.clear();
        }
        // a printer's own declaration (`fn print_json(`) is not a call
        let is_decl = ["fn ", "pub fn ", "pub(crate) fn "]
            .iter()
            .any(|p| t.starts_with(p));
        let pretty = (t.contains("println!") || t.starts_with("serde_json::to_string_pretty"))
            && line.contains("to_string_pretty");
        let prints_json = !is_decl
            && (pretty || t.contains("print_json(") || t.contains("format_dataset_summary("));
        if prints_json && fn_depth.is_some() && !annotated(ln) {
            sites.push((ln + 1, stack.clone()));
        }
        for c in line.chars() {
            match c {
                '{' => {
                    let looped = t.starts_with("for ")
                        || t.starts_with("while ")
                        || t.starts_with("loop ")
                        || t.contains(".for_each(");
                    stack.push((ln, looped));
                }
                '}' => {
                    stack.pop();
                    if let Some(d) = fn_depth
                        && stack.len() == d
                    {
                        let base = d + 1;
                        for (k, (l, st)) in sites.iter().enumerate() {
                            if st[base.min(st.len())..].iter().any(|(_, lp)| *lp) {
                                problems
                                    .push(format!("{file}:{l} {fn_name}: JSON printed in a loop"));
                            }
                            for (l2, st2) in &sites[k + 1..] {
                                if st2.starts_with(st) || st.starts_with(st2) {
                                    problems.push(format!(
                                        "{file}:{l} and :{l2} {fn_name}: two JSON documents on one path"
                                    ));
                                }
                            }
                        }
                        fn_depth = None;
                        sites.clear();
                    }
                }
                _ => {}
            }
        }
    }
    problems
}

/// No function of the command modules prints two JSON documents on one path or one in a loop
/// (a heuristic on the source; a deliberate exception carries `// json-ok: <reason>`).
#[test]
fn no_second_json_document_in_one_function() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
    let mut problems = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        problems.extend(json_problems(&name, &src));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The scan flags sequential prints and prints in a loop, not exclusive branches.
#[test]
fn json_scan_finds_sequential_and_looped_prints() {
    let src = r#"
fn a(x: bool) {
    if x {
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
    } else {
        println!("{}", serde_json::to_string_pretty(&w).unwrap());
    }
}
fn b() {
    println!("{}", format_dataset_summary("n", 1, 1, &[], &f));
    if json {
        println!("{}", serde_json::to_string_pretty(&w).unwrap());
    }
}
fn c() {
    for g in gs {
        println!("{}", serde_json::to_string_pretty(&g).unwrap());
    }
}
"#;
    let p = json_problems("x.rs", src);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].contains(" b: two JSON documents"), "{p:?}");
    assert!(p[1].contains(" c: JSON printed in a loop"), "{p:?}");
}

/// A `discrete` command with a warning prints the shared block: a blank line, `Warnings:`, then
/// `  - <warning>`.
#[test]
fn test_cli_print_convergence_warnings_block() {
    // z = 2 x is aliased, so the negative binomial fit warns "Coefficients not defined because
    // of singularities: z" and keeps x.
    let s = temp_session();
    let p = s.dir().join("alias.csv");
    let mut csv = String::from("y,x,z\n");
    for i in 0..30 {
        let y = (i * 7) % 11 / 2;
        let x = i as f64 / 10.0 + 0.5;
        csv.push_str(&format!("{y},{x},{}\n", 2.0 * x));
    }
    std::fs::write(&p, csv).unwrap();
    s.load_as(&p, "alias", &[]);
    let out = s.ok(&["discrete", "negbin", "alias", "-y", "y", "-x", "x", "z"]);
    let lines: Vec<&str> = out.lines().collect();
    let at = lines
        .iter()
        .position(|l| *l == "Warnings:")
        .unwrap_or_else(|| panic!("no Warnings: line\n{out}"));
    assert_eq!(lines[at - 1], "", "no blank line before Warnings:\n{out}");
    assert!(lines[at + 1].starts_with("  - "), "{out}");
}

/// R's `.` mark for 0.05 <= p < 0.1 in a coefficient table whose marks used to stop at `*`.
/// `discrete poisson zeroinfl -y y -x x1 x2`: the intercept's p-value is 0.0827 (R:
/// `glm(y ~ x1 + x2, poisson)`), so its row ends in ` .`; before, it ended in a blank.
#[test]
fn test_cli_marks_weak_significance_with_a_dot() {
    let s = session(&["validation/datasets/generated/zeroinfl.csv"]);
    let args = [
        "discrete", "poisson", "zeroinfl", "-y", "y", "-x", "x1", "x2",
    ];
    let mut json_args = vec!["-F", "json"];
    json_args.extend_from_slice(&args);
    let j: serde_json::Value = serde_json::from_str(&s.ok(&json_args)).unwrap();
    let names: Vec<&str> = j["variables"]
        .as_array()
        .unwrap_or_else(|| panic!("no variables\n{j}"))
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let p: Vec<f64> = j["p_values"]
        .as_array()
        .unwrap_or_else(|| panic!("no p_values\n{j}"))
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let weak: Vec<&str> = names
        .iter()
        .zip(&p)
        .filter(|&(_, &p)| (0.05..0.1).contains(&p))
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(weak.len(), 1, "one coefficient with 0.05 <= p < 0.1\n{j}");
    let text = s.ok(&args);
    let row = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(weak[0]))
        .unwrap_or_else(|| panic!("no row for {}\n{text}", weak[0]));
    assert!(row.ends_with(" ."), "{row:?}\n{text}");
}

/// `data head -F json` prints typed cells: integers and floats as numbers, a missing cell as
/// null, text as a string, booleans as booleans and dates as ISO strings.
#[test]
fn test_cli_data_head_json_is_typed() {
    use polars::prelude::*;
    let s = temp_session();
    let p = s.dir().join("typed.parquet");
    let mut df = df! {
        "i" => [3i64, 4],
        "f" => [Some(2.5f64), None],
        "t" => ["abc", "de"],
        "b" => [true, false],
        "d" => ["2026-11-17", "2026-11-18"],
    }
    .unwrap();
    df.apply("d", |c| c.cast(&DataType::Date).unwrap()).unwrap();
    ParquetWriter::new(std::fs::File::create(&p).unwrap())
        .finish(&mut df)
        .unwrap();
    s.load_as(&p, "typed", &[]);
    let rows = s.json(&["data", "head", "typed"]);
    let r0 = &rows[0];
    assert_eq!(r0["i"], serde_json::json!(3), "{rows}");
    assert_eq!(r0["f"], serde_json::json!(2.5), "{rows}");
    assert_eq!(r0["t"], serde_json::json!("abc"), "{rows}");
    assert_eq!(r0["b"], serde_json::json!(true), "{rows}");
    assert_eq!(r0["d"], serde_json::json!("2026-11-17"), "{rows}");
    assert!(rows[1]["f"].is_null(), "{rows}");
}

/// `data describe -F json` prints one object with the column types inside it.
#[test]
fn test_cli_data_describe_json_is_one_object() {
    let s = session(&["validation/datasets/grunfeld.csv"]);
    let docs = s.json_docs(&["data", "describe", "grunfeld"]);
    assert_eq!(docs.len(), 1, "{docs:?}");
    let v = &docs[0];
    assert_eq!(v["name"], "grunfeld");
    assert_eq!(v["rows"], 200);
    let names = v["column_names"].as_array().unwrap();
    let types = v["column_types"].as_array().unwrap();
    assert_eq!(v["columns"].as_u64().unwrap() as usize, names.len());
    assert_eq!(types.len(), names.len());
    for (t, n) in types.iter().zip(names) {
        assert_eq!(&t["name"], n);
        assert!(t["dtype"].is_string(), "{t}");
    }
}

/// `smoke-test -F json` prints one JSON document on stdout; the progress lines go to stderr.
#[test]
fn test_cli_smoke_test_json_is_one_document() {
    let out = p2a().args(["-F", "json", "smoke-test"]).output().unwrap();
    assert!(out.status.success(), "{out:?}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not one JSON document ({e}):\n{}",
            String::from_utf8_lossy(&out.stdout)
        )
    });
    assert_eq!(v["status"], "success");
    assert!(String::from_utf8_lossy(&out.stderr).contains("Running p2a smoke test"));
}

/// `data list --help` says the command reads every dataset.
#[test]
fn test_cli_data_list_help_says_it_reads_datasets() {
    let out = p2a().args(["data", "list", "--help"]).output().unwrap();
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("reads every dataset"), "{help}");
}

/// `discrete hurdle --dist negbin` reports theta's SE and log(theta)'s SE, as pscl's summary
/// (case disc_hurdle_t4, y_nb); the Poisson hurdle has neither.
#[test]
fn test_validate_cli_hurdle_theta_se() {
    let case = Case::load("disc_hurdle_t4");
    let s = temp_session();
    s.load_as(&case.dataset_path(), "zi", &[]);
    let args = [
        "discrete", "hurdle", "zi", "-y", "y_nb", "-x", "x1", "x2", "--dist", "negbin",
    ];
    let j = s.json(&args);
    Check::new(&case)
        .scalar("hnb_theta", num(&j["theta"]), Tol::iterative(1e-7))
        .scalar(
            "hnb_se_theta",
            num(&j["theta_std_error"]),
            Tol::iterative(1e-5),
        )
        .scalar(
            "hnb_se_log_theta",
            num(&j["log_theta_std_error"]),
            Tol::iterative(1e-5),
        )
        .finish();
    let text = s.ok(&args);
    assert!(
        text.lines()
            .any(|l| l.starts_with("Theta (dispersion): ") && l.contains("log(theta) SE: ")),
        "{text}"
    );
    let p = s.json(&[
        "discrete", "hurdle", "zi", "-y", "y", "-x", "x1", "x2", "--dist", "poisson",
    ]);
    assert!(p["theta_std_error"].is_null(), "{p}");
}
