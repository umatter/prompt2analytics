//! The paper's CLI exhibits, run and checked against R.
//!
//! Every test here runs one command of the paper's `generate_outputs.sh`
//! (`paper/archive/code/generate_outputs.sh` in the paper repository) exactly as
//! that script does: in a fresh session that first loads
//! `validation/datasets/grunfeld.csv` as `grunfeld`, with stdout and stderr
//! captured. The output is then checked in two steps:
//!
//! 1. Its numbers are parsed and compared with the committed R reference case
//!    (`validation/expected/<id>.json`): printed text at print precision
//!    ([`Tol::printed`]), `-F json` output at the `validation/README.md`
//!    tolerances.
//! 2. Only then is the whole output compared with its golden copy,
//!    `tests/golden/<exhibit>.txt`, named after the paper exhibit
//!    (`paper/archive/generated/<exhibit>.tex`), so a regenerated capture can be
//!    diffed against the golden directly.
//!
//! The order matters: `P2A_UPDATE_GOLDEN=1` rewrites a golden file, and since the
//! R checks run first, a golden copy can never hold a number that disagrees
//! with R.

use p2a_core::regression::{CovarianceType, OlsCoefficient};
use p2a_core::traits::t_test_p_value;
use p2a_core::{
    ClusterAdjustment, DataLoader, Dataset, PanelVcov, run_hausman_test, run_ols,
    run_ols_clustered, run_random_effects, run_vcov_hac,
};
use p2a_validation::golden::assert_golden;
use p2a_validation::text::{coefficient_rows, labelled};
use p2a_validation::{Case, Check, Tol};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// =============================================================================
// Harness
// =============================================================================

/// The paper's dataset; every exhibit case must be computed on it.
const GRUNFELD: &str = "validation/datasets/grunfeld.csv";

/// A temporary session file holding the dataset `grunfeld`.
struct Session {
    _dir: tempfile::TempDir,
    file: PathBuf,
}

/// A fresh session in which `dataset` is loaded as `grunfeld`.
fn session(dataset: &Path) -> Session {
    let dir = tempfile::tempdir().expect("temporary directory");
    let file = dir.path().join("s.json");
    let s = Session { _dir: dir, file };
    let out = p2a(
        &s,
        &[
            "data",
            "load",
            dataset.to_str().expect("UTF-8 dataset path"),
            "--name",
            "grunfeld",
        ],
    );
    assert!(
        out.status.success(),
        "data load {} failed: {}\n{}",
        dataset.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    s
}

/// A session for an exhibit case, which must be computed on the paper's dataset.
fn grunfeld_session(case: &Case) -> Session {
    assert_eq!(
        case.dataset, GRUNFELD,
        "case {} is not on the paper's dataset",
        case.id
    );
    session(&case.dataset_path())
}

/// Runs `p2a --session <file> <args...>`. Variables that change how polars
/// formats tables (`POLARS_*`, `COLUMNS`) are removed from its environment.
fn p2a(s: &Session, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_p2a"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("POLARS_") {
            cmd.env_remove(&key);
        }
    }
    cmd.env_remove("COLUMNS");
    cmd.arg("--session").arg(&s.file).args(args);
    cmd.output().expect("p2a runs")
}

/// stdout of a command that must succeed with an empty stderr, so that it equals
/// what `generate_outputs.sh` captures with `2>&1`.
fn run(s: &Session, args: &[&str]) -> String {
    let out = p2a(s, args);
    let stdout = String::from_utf8(out.stdout).expect("UTF-8 stdout");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "p2a {args:?} exited with {}:\n{stdout}\n{stderr}",
        out.status
    );
    assert!(stderr.is_empty(), "p2a {args:?} wrote to stderr:\n{stderr}");
    stdout
}

/// `tests/golden/<name>.txt`.
fn golden(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/")).join(format!("{name}.txt"))
}

/// The longest float at the start of `s`, `[+-]?digits[.digits][(e|E)[+-]digits]`,
/// or `NaN`, `inf`, `-inf`, with its length in bytes.
fn float_prefix(s: &str) -> Option<(f64, usize)> {
    for (word, value) in [
        ("NaN", f64::NAN),
        ("-inf", f64::NEG_INFINITY),
        ("+inf", f64::INFINITY),
        ("inf", f64::INFINITY),
    ] {
        if s.starts_with(word) {
            return Some((value, word.len()));
        }
    }
    let b = s.as_bytes();
    let digits = |mut i: usize| {
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let int_end = digits(i);
    if int_end == i {
        return None;
    }
    i = int_end;
    if i < b.len() && b[i] == b'.' {
        let frac_end = digits(i + 1);
        if frac_end > i + 1 {
            i = frac_end;
        }
    }
    if i < b.len() && matches!(b[i], b'e' | b'E') {
        let mut j = i + 1;
        if j < b.len() && matches!(b[j], b'+' | b'-') {
            j += 1;
        }
        let exp_end = digits(j);
        if exp_end > j {
            i = exp_end;
        }
    }
    s[..i].parse::<f64>().ok().map(|v| (v, i))
}

/// The number after the first `key` that follows `anchor` (an empty anchor is
/// the start of the text), spaces skipped. Panics with the line if there is none.
fn num_after(text: &str, anchor: &str, key: &str) -> f64 {
    let start = text
        .find(anchor)
        .unwrap_or_else(|| panic!("anchor `{anchor}` not found in\n{text}"));
    let at = start
        + text[start..]
            .find(key)
            .unwrap_or_else(|| panic!("`{key}` not found after `{anchor}` in\n{text}"));
    let after = text[at + key.len()..].trim_start_matches(' ');
    float_prefix(after).map(|(v, _)| v).unwrap_or_else(|| {
        let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
        let line = text[line_start..].lines().next().unwrap_or("");
        panic!("no number after `{key}` on the line `{line}`")
    })
}

/// Every number on the rest of the first line containing `key`. A number starts
/// at a word boundary; a `-` directly after a digit separates two numbers, so
/// `range 0.8415-0.8459` gives both bounds.
fn nums_on_line(text: &str, key: &str) -> Vec<f64> {
    let line = text
        .lines()
        .find(|l| l.contains(key))
        .unwrap_or_else(|| panic!("no line contains `{key}` in\n{text}"));
    let rest = &line[line.find(key).unwrap_or(0) + key.len()..];
    let mut out = Vec::new();
    let mut prev: Option<char> = None;
    let mut i = 0;
    while i < rest.len() {
        let c = rest[i..].chars().next().unwrap_or(' ');
        let after_digit = prev.is_some_and(|p| p.is_ascii_digit());
        let word_char = |p: char| p.is_alphanumeric() || p == '_' || p == '.';
        let at_boundary = prev.is_none_or(|p| !word_char(p));
        if !(c == '-' && after_digit) && at_boundary {
            if let Some((v, len)) = float_prefix(&rest[i..]) {
                out.push(v);
                prev = rest[..i + len].chars().last();
                i += len;
                continue;
            }
        }
        prev = Some(c);
        i += c.len_utf8();
    }
    out
}

/// A parsed coefficient table.
struct Table {
    names: Vec<String>,
    coef: Vec<f64>,
    se: Vec<f64>,
    stat: Vec<f64>,
    p: Vec<f64>,
}

/// The coefficient table of a text report (`format_regression_report` layout).
fn table(text: &str) -> Table {
    let mut t = Table {
        names: Vec::new(),
        coef: Vec::new(),
        se: Vec::new(),
        stat: Vec::new(),
        p: Vec::new(),
    };
    for (name, v) in coefficient_rows(text) {
        assert_eq!(v.len(), 4, "row `{name}` needs 4 numbers: {v:?}");
        t.names.push(name);
        t.coef.push(v[0]);
        t.se.push(v[1]);
        t.stat.push(v[2]);
        t.p.push(v[3]);
    }
    t
}

/// A JSON number, `null` as NaN; panics on anything else.
fn json_f64(v: &Value) -> f64 {
    match v {
        Value::Null => f64::NAN,
        Value::Number(n) => n.as_f64().expect("finite JSON number"),
        other => panic!("expected a number or null, got {other}"),
    }
}

/// The `coefficients` array of a JSON report; `stat_key` is `t_value` or `z_value`.
fn json_table(v: &Value, stat_key: &str) -> Table {
    let rows = v["coefficients"]
        .as_array()
        .unwrap_or_else(|| panic!("no `coefficients` array in {v}"));
    let mut t = Table {
        names: Vec::new(),
        coef: Vec::new(),
        se: Vec::new(),
        stat: Vec::new(),
        p: Vec::new(),
    };
    for row in rows {
        t.names.push(
            row["name"]
                .as_str()
                .unwrap_or_else(|| panic!("coefficient without a name: {row}"))
                .to_string(),
        );
        t.coef.push(json_f64(&row["coefficient"]));
        t.se.push(json_f64(&row["std_error"]));
        t.stat.push(json_f64(&row[stat_key]));
        t.p.push(json_f64(&row["p_value"]));
    }
    t
}

/// A std-only CSV reader: the header and the data rows, split on commas with
/// surrounding quotes stripped. The committed numeric CSVs have no quoted commas.
fn csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let split = |line: &str| -> Vec<String> {
        line.split(',')
            .map(|c| c.trim().trim_matches('"').to_string())
            .collect()
    };
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = split(lines.next().expect("CSV header"));
    let rows = lines.map(split).collect();
    (header, rows)
}

/// The number of distinct values in column `col` of a CSV file.
fn csv_levels(path: &Path, col: &str) -> usize {
    let (header, rows) = csv(path);
    let j = header
        .iter()
        .position(|h| h == col)
        .unwrap_or_else(|| panic!("no column `{col}` in {}", path.display()));
    rows.iter()
        .map(|r| r[j].clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// The JSON text with every number token outside strings replaced by `#`.
fn mask_numbers(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut chars = json.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            match c {
                '\\' => {
                    if let Some(e) = chars.next() {
                        out.push(e);
                    }
                }
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if c == '-' || c.is_ascii_digit() {
            while chars
                .peek()
                .is_some_and(|n| n.is_ascii_digit() || matches!(n, '.' | 'e' | 'E' | '+' | '-'))
            {
                chars.next();
            }
            out.push('#');
        } else {
            out.push(c);
        }
    }
    out
}

/// Compares two JSON values: same structure and strings, integers exactly,
/// floats within `tol`. Differences are appended to `diffs` with their path.
fn json_diff(path: &str, actual: &Value, expected: &Value, tol: Tol, diffs: &mut Vec<String>) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            if a.is_f64() != e.is_f64() {
                diffs.push(format!("{path}: {a} vs golden {e} (integer vs float)"));
            } else if a.is_f64() {
                let (a, e) = (
                    a.as_f64().unwrap_or(f64::NAN),
                    e.as_f64().unwrap_or(f64::NAN),
                );
                if !tol.allows(a, e) {
                    diffs.push(format!("{path}: {a:e} vs golden {e:e}"));
                }
            } else if a != e {
                diffs.push(format!("{path}: {a} vs golden {e}"));
            }
        }
        (Value::Array(a), Value::Array(e)) if a.len() == e.len() => {
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                json_diff(&format!("{path}[{i}]"), a, e, tol, diffs);
            }
        }
        (Value::Object(a), Value::Object(e))
            if a.keys().collect::<Vec<_>>() == e.keys().collect::<Vec<_>>() =>
        {
            for (k, a) in a {
                json_diff(&format!("{path}.{k}"), a, &e[k], tol, diffs);
            }
        }
        (a, e) if a == e => {}
        (a, e) => diffs.push(format!("{path}: {a} vs golden {e}")),
    }
}

/// Golden comparison of JSON output. With `P2A_UPDATE_GOLDEN=1` the file is
/// written ([`assert_golden`]). Otherwise the texts must be identical once every
/// number is masked (same keys, order, indentation and strings), and the numbers
/// must agree: integers exactly, floats within `Tol::new(1e-9, 1e-14)`. The
/// tolerance absorbs last-bit differences between CPUs (BLAS and faer reductions
/// differ between AVX2 and AVX-512); the R checks, not this comparison, carry the
/// numerical accuracy.
fn assert_json_golden(path: &Path, actual: &str) {
    if std::env::var_os("P2A_UPDATE_GOLDEN").is_some_and(|v| v == "1") {
        assert_golden(path, actual);
        return;
    }
    let expected = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "golden file {} cannot be read ({e}); check the output against R, \
             then create it with P2A_UPDATE_GOLDEN=1",
            path.display()
        )
    });
    let expected = expected.replace("\r\n", "\n");
    let (masked_a, masked_e) = (mask_numbers(actual), mask_numbers(&expected));
    if masked_a != masked_e {
        let line = masked_a
            .lines()
            .zip(masked_e.lines())
            .position(|(a, e)| a != e)
            .map_or_else(
                || masked_a.lines().count().min(masked_e.lines().count()) + 1,
                |i| i + 1,
            );
        panic!(
            "JSON golden {} differs in layout at line {line}:\nactual:\n{actual}\ngolden:\n{expected}",
            path.display()
        );
    }
    let a: Value = serde_json::from_str(actual).expect("actual output is JSON");
    let e: Value = serde_json::from_str(&expected).expect("golden file is JSON");
    let mut diffs = Vec::new();
    json_diff("$", &a, &e, Tol::new(1e-9, 1e-14), &mut diffs);
    assert!(
        diffs.is_empty(),
        "JSON golden {} differs:\n{}",
        path.display(),
        diffs.join("\n")
    );
}

#[test]
fn harness_parsers() {
    assert_eq!(float_prefix("-1.5e-3x"), Some((-1.5e-3, 7)));
    assert_eq!(float_prefix("12."), Some((12.0, 2)));
    assert!(float_prefix("NaN").is_some_and(|(v, n)| v.is_nan() && n == 3));
    assert_eq!(float_prefix("-inf"), Some((f64::NEG_INFINITY, 4)));
    assert_eq!(float_prefix("x1"), None);
    let text = "a\nsigma_u: 83.1138, sigma_e: 55.0085, theta (mean): 0.8420, range 0.8415-0.8459\n";
    assert_eq!(
        nums_on_line(text, "theta (mean):"),
        vec![0.8420, 0.8415, 0.8459]
    );
    assert_eq!(
        nums_on_line(
            "Wald chi-squared: 657.6739 on 2 DF (p-value: 0.0000)",
            "Wald chi-squared:"
        ),
        vec![657.6739, 2.0, 0.0]
    );
    assert_eq!(num_after(text, "sigma_u", "sigma_e:"), 55.0085);
    assert_eq!(
        mask_numbers("{\"a1\": -1.5e-3, \"b\": [2, null], \"c\": \"x 3\"}"),
        "{\"a1\": #, \"b\": [#, null], \"c\": \"x 3\"}"
    );
}

// =============================================================================
// data describe, data head (exhibits cli_data_describe, cli_data_head)
// =============================================================================

/// `Int64` when every CSV value of the column parses as an integer, else `Float64`.
fn csv_types(rows: &[Vec<String>], ncols: usize) -> Vec<&'static str> {
    (0..ncols)
        .map(|j| {
            if rows.iter().all(|r| r[j].parse::<i64>().is_ok()) {
                "Int64"
            } else {
                "Float64"
            }
        })
        .collect()
}

#[test]
fn test_validate_cli_data_describe() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &["data", "describe", "grunfeld"]);

    let (header, rows) = csv(&case.dataset_path());
    assert!(
        Tol::exact().allows(num_after(&out, "", "Rows:"), case.scalar("n_obs")),
        "Rows: vs R's n_obs {}\n{out}",
        case.scalar("n_obs")
    );
    assert_eq!(rows.len() as f64, case.scalar("n_obs"));
    assert!(
        out.contains(&format!("\n  Columns: {}\n", header.len())),
        "{out}"
    );
    assert!(
        out.contains(&format!("\n  Column names: {}\n", header.join(", "))),
        "{out}"
    );
    for (col, ty) in header.iter().zip(csv_types(&rows, header.len())) {
        assert!(
            out.contains(&format!("\n  {col}: {ty}\n")),
            "missing `  {col}: {ty}` in\n{out}"
        );
    }
    assert_golden(&golden("cli_data_describe"), &out);
}

#[test]
fn test_validate_cli_data_head() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &["data", "head", "grunfeld", "--n", "5"]);

    let (header, rows) = csv(&case.dataset_path());
    let cells = |line: &str| -> Vec<String> {
        line.trim()
            .trim_matches('│')
            .split(['│', '┆'])
            .map(|c| c.trim().to_string())
            .collect()
    };
    let lines: Vec<&str> = out.lines().collect();
    let body: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| l.starts_with('│'))
        .collect();
    assert!(body.len() >= 3, "no table in\n{out}");
    assert_eq!(cells(body[0]), header, "header row\n{out}");
    let dtypes: Vec<String> = csv_types(&rows, header.len())
        .iter()
        .map(|t| if *t == "Int64" { "i64" } else { "f64" }.to_string())
        .collect();
    assert_eq!(cells(body[2]), dtypes, "dtype row\n{out}");
    let sep = lines
        .iter()
        .position(|l| l.starts_with('╞'))
        .unwrap_or_else(|| panic!("no ╞ line in\n{out}"));
    let data: Vec<&str> = lines[sep + 1..]
        .iter()
        .copied()
        .take_while(|l| l.starts_with('│'))
        .collect();
    assert_eq!(data.len(), 5, "{out}");
    for (i, line) in data.iter().enumerate() {
        let printed: Vec<f64> = cells(line)
            .iter()
            .map(|c| {
                c.parse::<f64>()
                    .unwrap_or_else(|_| panic!("cell `{c}` in `{line}`"))
            })
            .collect();
        let expected: Vec<f64> = rows[i]
            .iter()
            .map(|c| c.parse::<f64>().expect("numeric CSV"))
            .collect();
        assert_eq!(printed, expected, "row {i}\n{out}");
    }
    assert_golden(&golden("cli_data_head"), &out);
}

// =============================================================================
// reg ols, reg clustered, reg hac (exhibits cli_ols, cli_ols_json)
// =============================================================================

/// The case's dataset as the core loads it, for in-process comparisons.
fn dataset(path: &Path) -> Dataset {
    Dataset::new(
        DataLoader::load_csv(path).unwrap_or_else(|e| panic!("load {}: {e}", path.display())),
    )
}

/// Parsed `-F json` output.
fn json(out: &str) -> Value {
    serde_json::from_str(out).unwrap_or_else(|e| panic!("not JSON ({e}):\n{out}"))
}

/// The lines under `Warnings:` (`  - <text>`), without the prefix.
fn warnings_block(text: &str) -> Vec<String> {
    text.lines()
        .skip_while(|l| *l != "Warnings:")
        .skip(1)
        .take_while(|l| l.starts_with("  - "))
        .map(|l| l["  - ".len()..].to_string())
        .collect()
}

/// The whitespace-split tokens of the rows of a report's coefficient table.
fn row_tokens(text: &str) -> Vec<Vec<String>> {
    let rule = "-".repeat(60);
    text.lines()
        .skip_while(|l| !l.starts_with("Variable "))
        .skip(2)
        .take_while(|l| *l != rule)
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// A p-value token: the value at 4 decimals plus at most its stars.
fn assert_p_token(tok: &str, p: f64, line: &[String]) {
    let printed = format!("{p:.4}");
    let rest = tok.strip_prefix(&printed);
    assert!(
        rest.is_some_and(|r| ["", ".", "*", "**", "***"].contains(&r)),
        "p token `{tok}` is not {printed} plus stars in {line:?}"
    );
}

/// Each printed row is the in-process coefficient at print precision: a NaN
/// prints `NaN`, never `0.000`.
fn assert_rows_match(text: &str, coefficients: &[OlsCoefficient]) {
    let rows = row_tokens(text);
    assert_eq!(rows.len(), coefficients.len(), "{text}");
    for (tokens, c) in rows.iter().zip(coefficients) {
        assert_eq!(tokens.len(), 5, "{tokens:?}");
        assert_eq!(tokens[0], c.name, "{tokens:?}");
        assert_eq!(tokens[1], format!("{:.6}", c.estimate), "{tokens:?}");
        assert_eq!(tokens[2], format!("{:.6}", c.std_error), "{tokens:?}");
        assert_eq!(tokens[3], format!("{:.3}", c.t_value), "{tokens:?}");
        assert_p_token(&tokens[4], c.p_value, tokens);
    }
}

/// Printed and in-process value of the same computation. The CLI prints the
/// shortest representation that reads back exactly, but serde_json's default
/// float parser (no `float_roundtrip` feature) is not correctly rounded and can
/// be an ulp or two off, so the comparison allows a few ulps.
const SAME_COMPUTATION: Tol = Tol::new(1e-15, 0.0);

/// A JSON number equals `expected`, `null` standing for NaN.
fn assert_json_num(v: &Value, expected: f64, what: &str) {
    let actual = json_f64(v);
    assert!(
        SAME_COMPUTATION.allows(actual, expected),
        "{what}: JSON {v} vs in-process {expected:e}"
    );
}

/// Each JSON coefficient object holds the in-process numbers.
fn assert_json_rows_match(v: &Value, coefficients: &[OlsCoefficient]) {
    let rows = v["coefficients"].as_array().expect("coefficients array");
    assert_eq!(rows.len(), coefficients.len(), "{v}");
    for (row, c) in rows.iter().zip(coefficients) {
        assert_eq!(row["name"], c.name.as_str());
        assert_json_num(&row["coefficient"], c.estimate, &c.name);
        assert_json_num(&row["std_error"], c.std_error, &c.name);
        assert_json_num(&row["t_value"], c.t_value, &c.name);
        assert_json_num(&row["p_value"], c.p_value, &c.name);
    }
}

/// The JSON `warnings` key: the list when `expected` is non-empty, else absent.
fn assert_json_warnings(v: &Value, expected: &[String]) {
    if expected.is_empty() {
        assert!(v.get("warnings").is_none(), "{v}");
    } else {
        assert_eq!(v["warnings"], serde_json::json!(expected), "{v}");
    }
}

const OLS: [&str; 8] = [
    "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
];

#[test]
fn test_validate_cli_ols() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &[&OLS[..], &["--robust", "hc1"]].concat());
    let t = table(&out);
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_hc1", &t.names, &t.se, Tol::printed(6))
        .named("t_hc1", &t.names, &t.stat, Tol::printed(3))
        .named("p_hc1", &t.names, &t.p, Tol::printed(4))
        .scalar("r2", labelled(&out, "R-squared:").unwrap(), Tol::printed(6))
        .scalar(
            "adj_r2",
            labelled(&out, "Adj. R-squared:").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .finish();
    assert!(
        out.contains("\nStandard errors: HC1 (heteroskedasticity-robust)\n"),
        "{out}"
    );
    assert!(
        out.contains("t-value") && !out.contains("Warnings:"),
        "{out}"
    );
    assert_golden(&golden("cli_ols"), &out);
}

#[test]
fn test_validate_cli_ols_json() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(
        &s,
        &[&["-F", "json"][..], &OLS, &["--robust", "hc1"]].concat(),
    );
    let v = json(&out);
    let t = json_table(&v, "t_value");
    let n = 200;
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se_hc1", &t.names, &t.se, Tol::se(n))
        .named("t_hc1", &t.names, &t.stat, Tol::stat(n))
        .named("p_hc1", &t.names, &t.p, Tol::p_value(n))
        .scalar("r2", json_f64(&v["r_squared"]), Tol::stat(n))
        .scalar("adj_r2", json_f64(&v["adj_r_squared"]), Tol::stat(n))
        .scalar("n_obs", json_f64(&v["n_observations"]), Tol::exact())
        .finish();
    assert_eq!(v["se_type"], "HC1");
    assert_eq!(v["r_squared_type"], "ordinary");
    assert_eq!(v["statistic"], "t");
    assert_eq!(v["method"], "OLS Regression");
    assert!(v.get("warnings").is_none(), "{out}");
    assert_json_golden(&golden("cli_ols_json"), &out);
}

#[test]
fn test_validate_cli_ols_standard() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &[&OLS[..], &["--robust", "standard"]].concat());
    let t = table(&out);
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_classical", &t.names, &t.se, Tol::printed(6))
        .named("t_classical", &t.names, &t.stat, Tol::printed(3))
        .named("p_classical", &t.names, &t.p, Tol::printed(4))
        .finish();
    assert!(out.contains("\nStandard errors: classical\n"), "{out}");

    let out = run(
        &s,
        &[&["-F", "json"][..], &OLS, &["--robust", "standard"]].concat(),
    );
    let v = json(&out);
    let t = json_table(&v, "t_value");
    let n = 200;
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se_classical", &t.names, &t.se, Tol::se(n))
        .named("t_classical", &t.names, &t.stat, Tol::stat(n))
        .named("p_classical", &t.names, &t.p, Tol::p_value(n))
        .finish();
    assert_eq!(v["se_type"], "classical");
}

/// X-20: the default is HC1, and the output says so.
#[test]
fn test_cli_ols_default_is_hc1() {
    let s = session(&p2a_validation::repo_path(GRUNFELD));
    let default = run(&s, &OLS);
    let hc1 = run(&s, &[&OLS[..], &["--robust", "hc1"]].concat());
    assert_eq!(default, hc1);
    assert!(
        default.contains("\nStandard errors: HC1 (heteroskedasticity-robust)\n"),
        "{default}"
    );
}

#[test]
fn test_validate_cli_clustered() {
    let case = Case::load("ols_grunfeld");
    let s = grunfeld_session(&case);
    let args = [
        "reg",
        "clustered",
        "grunfeld",
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--cluster",
        "firm",
    ];
    let out = run(&s, &args);
    let t = table(&out);
    let n = 200;
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_cluster_firm_hc1", &t.names, &t.se, Tol::printed(6))
        .named("p_cluster_firm_hc1", &t.names, &t.p, Tol::printed(4))
        .scalar(
            "adj_r2",
            labelled(&out, "Adj. R-squared:").unwrap(),
            Tol::printed(6),
        )
        .finish();
    // R has no t key for the clustered fit: t = coef / se, derived from the case.
    let coef = case.by_names("coef", &t.names);
    let se = case.by_names("se_cluster_firm_hc1", &t.names);
    for (i, name) in t.names.iter().enumerate() {
        let expected = coef[i] / se[i];
        assert!(
            Tol::printed(3).allows(t.stat[i], expected),
            "t[{name}] printed {} vs R {expected}",
            t.stat[i]
        );
    }
    let g = csv_levels(&case.dataset_path(), "firm");
    assert!(
        out.contains(&format!(
            "\nStandard errors: clustered by firm (CR1, G = {g})\n"
        )),
        "{out}"
    );
    let r = run_ols_clustered(
        &dataset(&case.dataset_path()),
        "inv",
        &["value", "capital"],
        "firm",
        None,
    )
    .unwrap();
    assert!(
        !r.ols.warnings.is_empty(),
        "G = 10 clusters should be warned about"
    );
    assert_eq!(warnings_block(&out), r.ols.warnings, "{out}");

    let out = run(&s, &[&["-F", "json"][..], &args].concat());
    let v = json(&out);
    let t = json_table(&v, "t_value");
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se_cluster_firm_hc1", &t.names, &t.se, Tol::se(n))
        .named("p_cluster_firm_hc1", &t.names, &t.p, Tol::p_value(n))
        .scalar("adj_r2", json_f64(&v["adj_r_squared"]), Tol::stat(n))
        .finish();
    let coef = case.by_names("coef", &t.names);
    let se = case.by_names("se_cluster_firm_hc1", &t.names);
    for (i, name) in t.names.iter().enumerate() {
        let expected = coef[i] / se[i];
        assert!(
            Tol::stat(n).allows(t.stat[i], expected),
            "t[{name}] {} vs R {expected}",
            t.stat[i]
        );
    }
    assert_eq!(v["se_type"], "cluster:firm");
    assert_eq!(v["n_clusters"], g);
    assert!(v.get("n_clusters_2").is_none(), "{out}");
    assert_eq!(v["warnings"], serde_json::json!(r.ols.warnings));
}

/// CLI-6: an aliased regressor (a column given twice) must not abort the CLI;
/// the table shows the core's numbers, NaN included, and its warnings.
#[test]
fn test_cli_ols_aliased_regressor_no_panic() {
    let path = p2a_validation::repo_path(GRUNFELD);
    let s = session(&path);
    let args = [
        "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital", "value",
    ];
    let r = run_ols(
        &dataset(&path),
        "inv",
        &["value", "capital", "value"],
        true,
        CovarianceType::HC1,
    )
    .unwrap();
    let names: Vec<&str> = r.coefficients.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["(Intercept)", "value", "capital", "value"]);

    let out = p2a(&s, &args);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(out.status.success(), "{}\n{stdout}\n{stderr}", out.status);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_rows_match(&stdout, &r.coefficients);
    for w in &r.warnings {
        assert!(
            stdout.contains(&format!("\n  - {w}\n")),
            "missing `{w}` in\n{stdout}"
        );
    }

    let out = p2a(&s, &[&["-F", "json"][..], &args].concat());
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(out.status.success(), "{}\n{stdout}\n{stderr}", out.status);
    assert!(!stderr.contains("panicked"), "{stderr}");
    let v = json(&stdout);
    assert_json_rows_match(&v, &r.coefficients);
    assert_json_warnings(&v, &r.warnings);
}

/// CLI-6 for `reg clustered`. The core's cluster-robust standard errors of an
/// aliased design are not reproducible from run to run (the cluster meat is a
/// parallel sum over a `HashMap` of clusters, and the singular bread turns its
/// last-bit noise into NaN in one run and a finite value in the next), so the
/// SEs are checked for what the CLI does with them, not against an in-process
/// run: a NaN SE gives a NaN t and p, a finite one t = c / s.
#[test]
fn test_cli_clustered_aliased_regressor_no_panic() {
    let path = p2a_validation::repo_path(GRUNFELD);
    let s = session(&path);
    let args = [
        "reg",
        "clustered",
        "grunfeld",
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "value",
        "--cluster",
        "firm",
    ];
    let r = run_ols_clustered(
        &dataset(&path),
        "inv",
        &["value", "capital", "value"],
        "firm",
        None,
    )
    .unwrap();

    let out = p2a(&s, &args);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(out.status.success(), "{}\n{stdout}\n{stderr}", out.status);
    assert!(!stderr.contains("panicked"), "{stderr}");
    let rows = row_tokens(&stdout);
    assert_eq!(rows.len(), r.ols.coefficients.len(), "{stdout}");
    for (tokens, c) in rows.iter().zip(&r.ols.coefficients) {
        assert_eq!(tokens.len(), 5, "{tokens:?}");
        assert_eq!(tokens[0], c.name, "{tokens:?}");
        assert_eq!(tokens[1], format!("{:.6}", c.estimate), "{tokens:?}");
        if tokens[2] == "NaN" {
            assert_eq!(tokens[3..], ["NaN", "NaN"], "{tokens:?}");
        }
    }
    assert_eq!(warnings_block(&stdout), r.ols.warnings, "{stdout}");

    let out = p2a(&s, &[&["-F", "json"][..], &args].concat());
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(out.status.success(), "{}\n{stdout}\n{stderr}", out.status);
    assert!(!stderr.contains("panicked"), "{stderr}");
    let v = json(&stdout);
    let rows = v["coefficients"].as_array().expect("coefficients array");
    assert_eq!(rows.len(), r.ols.coefficients.len(), "{v}");
    for (row, c) in rows.iter().zip(&r.ols.coefficients) {
        assert_eq!(row["name"], c.name.as_str());
        assert_json_num(&row["coefficient"], c.estimate, &c.name);
        let (b, se) = (json_f64(&row["coefficient"]), json_f64(&row["std_error"]));
        if se.is_nan() {
            assert!(
                row["t_value"].is_null() && row["p_value"].is_null(),
                "{row}"
            );
        } else {
            let expected = b / se;
            if expected.is_finite() {
                let t = json_f64(&row["t_value"]);
                assert!(
                    Tol::new(1e-12, 0.0).allows(t, expected),
                    "t {t} vs c/s {expected}"
                );
            } else {
                assert!(row["t_value"].is_null(), "{row}");
            }
        }
    }
    assert_json_warnings(&v, &r.ols.warnings);
}

/// The rows of the `reg hac` table: name, coef, HAC SE, t, p (stars dropped).
fn hac_rows(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .skip_while(|l| !l.starts_with("Variable "))
        .skip(2)
        .take_while(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().take(5).map(str::to_string).collect())
        .collect()
}

/// CLI-6 (`reg hac`): t is c / s, also when the HAC SE is zero, and the
/// estimator's warnings are printed.
#[test]
fn test_cli_hac_t_ratio_and_warnings() {
    let path = p2a_validation::repo_path(GRUNFELD);
    let s = session(&path);
    let ds = dataset(&path);

    // (a) A duplicated regressor: the second `value` absorbs nothing and its
    // HAC SE can be exactly 0, so c / s is infinite, never 0.
    let h = run_vcov_hac(
        &ds,
        "inv",
        &["value", "capital", "value"],
        None,
        Some("bartlett"),
        false,
    )
    .unwrap();
    let args = [
        "reg", "hac", "grunfeld", "-y", "inv", "-x", "value", "capital", "value",
    ];
    let out = run(&s, &args);
    let rows = hac_rows(&out);
    assert_eq!(rows.len(), h.coefficients.len(), "{out}");
    let df = (h.n_obs - h.n_params) as f64;
    for (i, row) in rows.iter().enumerate() {
        let t = h.coefficients[i] / h.std_errors[i];
        assert_eq!(row[3], format!("{t:.4}"), "{row:?}\n{out}");
        assert_eq!(
            row[4],
            format!("{:.4}", t_test_p_value(t, df)),
            "{row:?}\n{out}"
        );
    }
    assert_eq!(warnings_block(&out), h.warnings, "{out}");
    let v = json(&run(&s, &[&["-F", "json"][..], &args].concat()));
    let t_json = v["t_values"].as_array().expect("t_values");
    for (i, tj) in t_json.iter().enumerate() {
        let t = h.coefficients[i] / h.std_errors[i];
        if t.is_finite() {
            assert_json_num(tj, t, "t_values");
        } else {
            assert!(tj.is_null(), "non-finite t {t} must be null: {tj}");
        }
    }

    // (b) A bandwidth above n/4 is warned about (T0-WP14).
    let h = run_vcov_hac(
        &ds,
        "inv",
        &["value", "capital"],
        Some(60),
        Some("bartlett"),
        false,
    )
    .unwrap();
    assert!(!h.warnings.is_empty(), "--lags 60 of 200 should warn");
    let args = [
        "reg", "hac", "grunfeld", "-y", "inv", "-x", "value", "capital", "--lags", "60",
    ];
    let out = run(&s, &args);
    assert_eq!(warnings_block(&out), h.warnings, "{out}");
    let v = json(&run(&s, &[&["-F", "json"][..], &args].concat()));
    assert_eq!(v["warnings"], serde_json::json!(h.warnings), "{v}");
}

// =============================================================================
// reg diagnostics (exhibit cli_diagnostics)
// =============================================================================

#[test]
fn test_validate_cli_diagnostics() {
    let case = Case::load("ols_diagnostics_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(
        &s,
        &[
            "reg",
            "diagnostics",
            "grunfeld",
            "-y",
            "inv",
            "-x",
            "value",
            "capital",
        ],
    );
    // The VIF lines `  <name>: <v>` between the header and the next empty line.
    let (names, vifs): (Vec<String>, Vec<f64>) = out
        .lines()
        .skip_while(|l| *l != "Variance Inflation Factors:")
        .skip(1)
        .take_while(|l| !l.trim().is_empty())
        .map(|l| {
            let (name, v) = l
                .trim()
                .rsplit_once(": ")
                .unwrap_or_else(|| panic!("VIF line `{l}`"));
            (name.to_string(), v.parse::<f64>().unwrap_or(f64::NAN))
        })
        .unzip();
    Check::new(&case)
        .scalar(
            "jarque_bera",
            num_after(&out, "Jarque-Bera Test", "Statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "jarque_bera_p",
            num_after(&out, "Jarque-Bera Test", "P-value:"),
            Tol::printed(4),
        )
        .scalar(
            "breusch_pagan",
            num_after(&out, "Breusch-Pagan Test", "Statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "breusch_pagan_p",
            num_after(&out, "Breusch-Pagan Test", "P-value:"),
            Tol::printed(4),
        )
        .scalar(
            "durbin_watson",
            num_after(&out, "", "Durbin-Watson Statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "condition_number",
            num_after(&out, "", "Condition Number:"),
            Tol::printed(4),
        )
        .named("vif", &names, &vifs, Tol::printed(4))
        .finish();
    assert!(!out.contains("(Intercept):"), "{out}");
    assert_golden(&golden("cli_diagnostics"), &out);
}

// =============================================================================
// panel fe, panel re, panel hdfe (exhibits cli_fe, cli_re, cli_hdfe)
// =============================================================================

const FE: [&str; 10] = [
    "panel", "fe", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
];
const RE: [&str; 10] = [
    "panel", "re", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
];
const HDFE: [&str; 11] = [
    "panel", "hdfe", "grunfeld", "-y", "inv", "-x", "value", "capital", "--fe", "firm", "year",
];

/// `args` with `extra` appended.
fn with(args: &[&'static str], extra: &[&'static str]) -> Vec<&'static str> {
    [args, extra].concat()
}

/// `-F json` and `args`.
fn as_json(args: &[&'static str]) -> Vec<&'static str> {
    [&["-F", "json"][..], args].concat()
}

/// `case[coef] / case[se]` for `names` (R reports no t for a sandwich covariance).
fn derived_t(case: &Case, se_key: &str, names: &[String]) -> Vec<f64> {
    let coef = case.by_names("coef", names);
    let se = case.by_names(se_key, names);
    coef.iter().zip(&se).map(|(b, s)| b / s).collect()
}

/// Every `actual` within `tol` of `expected`.
fn assert_all(what: &str, actual: &[f64], expected: &[f64], tol: Tol) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(tol.allows(*a, *e), "{what}[{i}]: {a} vs R {e}");
    }
}

#[test]
fn test_validate_cli_fe() {
    let case = Case::load("panel_fe_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &FE);
    let t = table(&out);
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se", &t.names, &t.se, Tol::printed(6))
        .named("t", &t.names, &t.stat, Tol::printed(3))
        .named("p", &t.names, &t.p, Tol::printed(4))
        .scalar(
            "r2",
            labelled(&out, "R-squared (within):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "adj_r2",
            labelled(&out, "Adj. R-squared (within):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .scalar(
            "n_groups",
            labelled(&out, "Number of groups:").unwrap(),
            Tol::exact(),
        )
        .scalar(
            "f_stat",
            num_after(&out, "", "F-statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "f_p",
            num_after(&out, "F-statistic:", "(p-value:"),
            Tol::printed(4),
        )
        .finish();
    assert!(
        out.contains(&format!("\nStandard errors: {}\n", PanelVcov::Classical)),
        "{out}"
    );
    assert!(out.contains("t-value") && !out.contains("z-value"), "{out}");
    assert_golden(&golden("cli_fe"), &out);
}

#[test]
fn test_validate_cli_fe_json() {
    let case = Case::load("panel_fe_grunfeld");
    let s = grunfeld_session(&case);
    let v = json(&run(&s, &as_json(&FE)));
    let t = json_table(&v, "t_value");
    let n = 200;
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se", &t.names, &t.se, Tol::se(n))
        .named("t", &t.names, &t.stat, Tol::stat(n))
        .named("p", &t.names, &t.p, Tol::p_value(n))
        .scalar("r2", json_f64(&v["r_squared"]), Tol::stat(n))
        .scalar("adj_r2", json_f64(&v["adj_r_squared"]), Tol::stat(n))
        .scalar("n_obs", json_f64(&v["n_observations"]), Tol::exact())
        .scalar("n_groups", json_f64(&v["n_groups"]), Tol::exact())
        .scalar("f_stat", json_f64(&v["f_statistic"]), Tol::stat(n))
        .scalar("f_p", json_f64(&v["f_p_value"]), Tol::p_value(n))
        .finish();
    assert_eq!(v["r_squared_type"], "within");
    assert_eq!(v["se_type"], "classical");
    assert_eq!(v["statistic"], "t");
}

#[test]
fn test_validate_cli_fe_vcov() {
    let case = Case::load("panel_fe_grunfeld");
    let s = grunfeld_session(&case);
    let n = 200;
    for (flag, adj, json_se) in [
        ("cluster", "sss", "cluster:firm"),
        ("cluster-hc1", "hc1", "cluster-hc1:firm"),
        ("cluster-hc0", "hc0", "cluster-hc0:firm"),
    ] {
        let v = json(&run(&s, &as_json(&with(&FE, &["--vcov", flag]))));
        let t = json_table(&v, "t_value");
        let se_key = format!("se_arellano_{adj}");
        Check::new(&case)
            .named("coef", &t.names, &t.coef, Tol::coef(n))
            .named(&se_key, &t.names, &t.se, Tol::se(n))
            .scalar(
                &format!("f_stat_arellano_{adj}"),
                json_f64(&v["f_statistic"]),
                Tol::stat(n),
            )
            .scalar(
                &format!("f_p_arellano_{adj}"),
                json_f64(&v["f_p_value"]),
                Tol::p_value(n),
            )
            .finish();
        assert_all(
            &format!("t ({flag})"),
            &t.stat,
            &derived_t(&case, &se_key, &t.names),
            Tol::stat(n),
        );
        assert_eq!(v["se_type"], json_se, "{flag}");
    }

    let out = run(&s, &with(&FE, &["--vcov", "cluster"]));
    let t = table(&out);
    Check::new(&case)
        .named("se_arellano_sss", &t.names, &t.se, Tol::printed(6))
        .scalar(
            "f_stat_arellano_sss",
            num_after(&out, "", "F-statistic (Wald, clustered):"),
            Tol::printed(4),
        )
        .scalar(
            "f_df2_arellano",
            nums_on_line(&out, "F-statistic (Wald, clustered):")[2],
            Tol::exact(),
        )
        .finish();
    assert!(
        out.contains(&format!(
            "\nStandard errors: {}\n",
            PanelVcov::Arellano(ClusterAdjustment::Stata)
        )),
        "{out}"
    );
    assert_eq!(run(&s, &with(&FE, &["--vcov", "classical"])), run(&s, &FE));
}

#[test]
fn test_validate_cli_re() {
    let case = Case::load("panel_re_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &RE);
    let header = out
        .lines()
        .find(|l| l.starts_with("Variable "))
        .unwrap_or_else(|| panic!("no table header in\n{out}"));
    assert!(
        header.contains("z-value") && !header.contains("t-value"),
        "{header}"
    );
    let t = table(&out);
    let wald = nums_on_line(&out, "Wald chi-squared:");
    assert_eq!(wald.len(), 3, "{wald:?}");
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se", &t.names, &t.se, Tol::printed(6))
        .named("z", &t.names, &t.stat, Tol::printed(3))
        .named("p", &t.names, &t.p, Tol::printed(4))
        .scalar("r2", labelled(&out, "R-squared:").unwrap(), Tol::printed(6))
        .scalar(
            "adj_r2",
            labelled(&out, "Adj. R-squared:").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .scalar(
            "n_groups",
            labelled(&out, "Number of groups:").unwrap(),
            Tol::exact(),
        )
        .scalar("wald_chisq", wald[0], Tol::printed(4))
        .scalar("wald_df", wald[1], Tol::exact())
        .scalar("wald_p", wald[2], Tol::printed(4))
        .finish();
    for (key, expected) in [
        ("sigma_u:", case.scalar("sigma2_id").sqrt()),
        ("sigma_e:", case.scalar("sigma2_idios").sqrt()),
        ("theta:", case.scalar("theta_mean")),
    ] {
        let printed = num_after(&out, "", key);
        assert!(
            Tol::printed(4).allows(printed, expected),
            "{key} {printed} vs R {expected}"
        );
    }
    assert!(out.contains("\nStandard errors: classical\n"), "{out}");
    assert!(!out.contains("F-statistic:"), "{out}");
    assert_golden(&golden("cli_re"), &out);
}

#[test]
fn test_validate_cli_re_json() {
    let case = Case::load("panel_re_grunfeld");
    let s = grunfeld_session(&case);
    let v = json(&run(&s, &as_json(&RE)));
    let t = json_table(&v, "z_value");
    let n = 200;
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se", &t.names, &t.se, Tol::se(n))
        .named("z", &t.names, &t.stat, Tol::stat(n))
        .named("p", &t.names, &t.p, Tol::p_value(n))
        .scalar("r2", json_f64(&v["r_squared"]), Tol::stat(n))
        .scalar("adj_r2", json_f64(&v["adj_r_squared"]), Tol::stat(n))
        .scalar("wald_chisq", json_f64(&v["wald_chi2"]), Tol::stat(n))
        .scalar("wald_df", json_f64(&v["wald_df"]), Tol::exact())
        .scalar("wald_p", json_f64(&v["wald_p_value"]), Tol::p_value(n))
        .scalar("wald_f", json_f64(&v["f_statistic"]), Tol::stat(n))
        .scalar("wald_f_p", json_f64(&v["f_p_value"]), Tol::p_value(n))
        .scalar("sigma2_id", json_f64(&v["sigma_u"]).powi(2), Tol::stat(n))
        .scalar(
            "sigma2_idios",
            json_f64(&v["sigma_e"]).powi(2),
            Tol::stat(n),
        )
        .scalar("theta_mean", json_f64(&v["theta"]), Tol::stat(n))
        .scalar("n_groups", json_f64(&v["n_groups"]), Tol::exact())
        .scalar("n_obs", json_f64(&v["n_observations"]), Tol::exact())
        .finish();
    for row in v["coefficients"].as_array().unwrap() {
        assert!(row.get("t_value").is_none(), "{row}");
    }
    assert_eq!(v["statistic"], "z");
    assert_eq!(v["se_type"], "classical");
    assert_eq!(v["r_squared_type"], "cor(y*, X*b)^2 (plm)");
    assert!(
        v.get("theta_min").is_none() && v.get("theta_max").is_none(),
        "{v}"
    );
}

#[test]
fn test_validate_cli_re_vcov_cluster() {
    let case = Case::load("panel_re_grunfeld");
    let s = grunfeld_session(&case);
    let n = 200;
    let out = run(&s, &with(&RE, &["--vcov", "cluster"]));
    let t = table(&out);
    let wald = nums_on_line(&out, "Wald chi-squared (clustered):");
    assert_eq!(wald.len(), 3, "{wald:?}");
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_arellano_sss", &t.names, &t.se, Tol::printed(6))
        .scalar("wald_chisq_arellano_sss", wald[0], Tol::printed(4))
        .scalar("wald_df", wald[1], Tol::exact())
        .scalar("wald_p_arellano_sss", wald[2], Tol::printed(4))
        .finish();
    assert_all(
        "z (cluster)",
        &t.stat,
        &derived_t(&case, "se_arellano_sss", &t.names),
        Tol::printed(3),
    );
    assert!(
        out.contains(&format!(
            "\nStandard errors: {}\n",
            PanelVcov::Arellano(ClusterAdjustment::Stata)
        )),
        "{out}"
    );
    for (flag, adj) in [("cluster-hc0", "hc0"), ("cluster-hc1", "hc1")] {
        let v = json(&run(&s, &as_json(&with(&RE, &["--vcov", flag]))));
        let t = json_table(&v, "z_value");
        Check::new(&case)
            .named(&format!("se_arellano_{adj}"), &t.names, &t.se, Tol::se(n))
            .scalar(
                &format!("wald_chisq_arellano_{adj}"),
                json_f64(&v["wald_chi2"]),
                Tol::stat(n),
            )
            .scalar(
                &format!("wald_p_arellano_{adj}"),
                json_f64(&v["wald_p_value"]),
                Tol::p_value(n),
            )
            .scalar(
                &format!("wald_f_arellano_{adj}"),
                json_f64(&v["f_statistic"]),
                Tol::stat(n),
            )
            .scalar(
                &format!("wald_f_p_arellano_{adj}"),
                json_f64(&v["f_p_value"]),
                Tol::p_value(n),
            )
            .finish();
        assert_eq!(v["se_type"], format!("{flag}:firm"));
    }
}

#[test]
fn test_validate_cli_re_unbalanced() {
    let case = Case::load("panel_re_grunfeld_unbalanced");
    let s = session(&case.dataset_path());
    let out = run(&s, &RE);
    let t = table(&out);
    let theta = nums_on_line(&out, "theta (mean):");
    assert_eq!(theta.len(), 3, "{theta:?}");
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se", &t.names, &t.se, Tol::printed(6))
        .named("z", &t.names, &t.stat, Tol::printed(3))
        .named("p", &t.names, &t.p, Tol::printed(4))
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .scalar(
            "n_groups",
            labelled(&out, "Number of groups:").unwrap(),
            Tol::exact(),
        )
        .scalar("theta_mean", theta[0], Tol::printed(4))
        .scalar("theta_min", theta[1], Tol::printed(4))
        .scalar("theta_max", theta[2], Tol::printed(4))
        .finish();
    for (key, expected) in [
        ("sigma_u:", case.scalar("sigma2_id").sqrt()),
        ("sigma_e:", case.scalar("sigma2_idios").sqrt()),
    ] {
        let printed = num_after(&out, "", key);
        assert!(
            Tol::printed(4).allows(printed, expected),
            "{key} {printed} vs R {expected}"
        );
    }

    let v = json(&run(&s, &as_json(&RE)));
    let n = 171;
    Check::new(&case)
        .scalar("theta_mean", json_f64(&v["theta"]), Tol::stat(n))
        .scalar("theta_min", json_f64(&v["theta_min"]), Tol::stat(n))
        .scalar("theta_max", json_f64(&v["theta_max"]), Tol::stat(n))
        .scalar("n_obs", json_f64(&v["n_observations"]), Tol::exact())
        .finish();
}

#[test]
fn test_validate_cli_re_no_effect() {
    let case = Case::load("panel_re_sim_no_effect");
    let s = session(&case.dataset_path());
    let out = run(&s, &RE);
    let t = table(&out);
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se", &t.names, &t.se, Tol::printed(6))
        .finish();
    for (key, expected) in [
        ("sigma_u:", case.scalar("sigma2_id").sqrt()),
        ("theta:", case.scalar("theta_mean")),
    ] {
        let printed = num_after(&out, "", key);
        assert!(
            Tol::printed(4).allows(printed, expected),
            "{key} {printed} vs R {expected}"
        );
    }
    let r = run_random_effects(
        &dataset(&case.dataset_path()),
        "inv",
        &["value", "capital"],
        "firm",
    )
    .unwrap();
    assert!(
        !r.warnings.is_empty(),
        "the negative variance estimate should be warned about"
    );
    assert_eq!(warnings_block(&out), r.warnings, "{out}");
    let v = json(&run(&s, &as_json(&RE)));
    assert_eq!(v["warnings"], serde_json::json!(r.warnings), "{v}");
}

#[test]
fn test_validate_cli_hdfe() {
    let case = Case::load("panel_hdfe_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &HDFE);
    let t = table(&out);
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_hetero", &t.names, &t.se, Tol::printed(6))
        .named("t_hetero", &t.names, &t.stat, Tol::printed(3))
        .named("p_hetero", &t.names, &t.p, Tol::printed(4))
        .scalar(
            "r2_within",
            labelled(&out, "R-squared (within):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "adj_r2_within_plm",
            labelled(&out, "Adj. R-squared (within):").unwrap(),
            Tol::printed(6),
        )
        .scalar(
            "n_obs",
            labelled(&out, "Observations:").unwrap(),
            Tol::exact(),
        )
        .scalar(
            "within_f",
            num_after(&out, "", "F-statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "within_f_p",
            num_after(&out, "F-statistic:", "(p-value:"),
            Tol::printed(4),
        )
        .finish();
    for fe in ["firm", "year"] {
        let levels = csv_levels(&case.dataset_path(), fe);
        assert!(
            out.contains(&format!("\n  {fe}: {levels} levels\n")),
            "{out}"
        );
    }
    assert!(
        out.contains("\nStandard errors: HC1 (heteroskedasticity-robust)\n"),
        "{out}"
    );
    assert_golden(&golden("cli_hdfe"), &out);
}

#[test]
fn test_validate_cli_hdfe_json() {
    let case = Case::load("panel_hdfe_grunfeld");
    let s = grunfeld_session(&case);
    let v = json(&run(&s, &as_json(&HDFE)));
    let t = json_table(&v, "t_value");
    let n = 200;
    // The MAP demeaning converges to ~1e-15 on this balanced panel, so the
    // standard tolerances apply (no Tol::iterative).
    Check::new(&case)
        .named("coef", &t.names, &t.coef, Tol::coef(n))
        .named("se_hetero", &t.names, &t.se, Tol::se(n))
        .named("t_hetero", &t.names, &t.stat, Tol::stat(n))
        .named("p_hetero", &t.names, &t.p, Tol::p_value(n))
        .scalar("r2_within", json_f64(&v["r_squared"]), Tol::stat(n))
        .scalar(
            "adj_r2_within_plm",
            json_f64(&v["adj_r_squared"]),
            Tol::stat(n),
        )
        .scalar("within_f", json_f64(&v["f_statistic"]), Tol::stat(n))
        .scalar("within_f_p", json_f64(&v["f_p_value"]), Tol::p_value(n))
        .scalar("n_obs", json_f64(&v["n_observations"]), Tol::exact())
        .finish();
    let path = case.dataset_path();
    let (firm, year) = (csv_levels(&path, "firm"), csv_levels(&path, "year"));
    assert_eq!(v["n_groups"], firm);
    assert_eq!(
        v["fe_dimensions"],
        serde_json::json!([{"levels": firm, "name": "firm"}, {"levels": year, "name": "year"}])
    );
    assert_eq!(v["r_squared_type"], "within");
    assert_eq!(v["se_type"], "HC1");
    assert_eq!(v["statistic"], "t");
}

#[test]
fn test_cli_panel_vcov_rejects_unknown() {
    let s = session(&p2a_validation::repo_path(GRUNFELD));
    for cmd in [&FE, &RE] {
        let out = p2a(&s, &with(cmd, &["--vcov", "robust"]));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("invalid value 'robust'"), "{stderr}");
        for allowed in ["classical", "cluster", "cluster-hc0", "cluster-hc1"] {
            assert!(stderr.contains(allowed), "{allowed} missing in {stderr}");
        }
    }
}

// =============================================================================
// panel hausman (exhibit cli_hausman)
// =============================================================================

const HAUSMAN: [&str; 10] = [
    "panel", "hausman", "grunfeld", "-y", "inv", "-x", "value", "capital", "--entity", "firm",
];

#[test]
fn test_validate_cli_hausman() {
    let case = Case::load("panel_hausman_grunfeld");
    let s = grunfeld_session(&case);
    let out = run(&s, &HAUSMAN);
    Check::new(&case)
        .scalar(
            "chisq",
            num_after(&out, "", "Chi-squared statistic:"),
            Tol::printed(4),
        )
        .scalar(
            "df",
            num_after(&out, "", "Degrees of freedom:"),
            Tol::exact(),
        )
        .scalar("p", num_after(&out, "", "P-value:"), Tol::printed(4))
        .finish();
    let h = run_hausman_test(
        &dataset(&case.dataset_path()),
        "inv",
        &["value", "capital"],
        "firm",
    )
    .unwrap();
    let lines: Vec<&str> = out.lines().collect();
    let p_line = lines
        .iter()
        .position(|l| l.starts_with("P-value:"))
        .unwrap_or_else(|| panic!("no P-value line in\n{out}"));
    assert_eq!(
        lines.get(p_line + 1).copied(),
        Some(format!("Covariance: {}", h.vcov_method).as_str()),
        "{out}"
    );
    assert!(
        out.contains(&format!("\nRecommendation: {}\n", h.recommendation)),
        "{out}"
    );
    assert!(h.warnings.is_empty(), "{:?}", h.warnings);
    assert!(!out.contains("Warnings:"), "{out}");
    assert_golden(&golden("cli_hausman"), &out);
}

#[test]
fn test_validate_cli_hausman_json() {
    let case = Case::load("panel_hausman_grunfeld");
    let s = grunfeld_session(&case);
    let v = json(&run(&s, &as_json(&HAUSMAN)));
    let n = 200;
    Check::new(&case)
        .scalar("chisq", json_f64(&v["chi2_statistic"]), Tol::stat(n))
        .scalar("p", json_f64(&v["p_value"]), Tol::p_value(n))
        .scalar("df", json_f64(&v["df"]), Tol::exact())
        .finish();
    let h = run_hausman_test(
        &dataset(&case.dataset_path()),
        "inv",
        &["value", "capital"],
        "firm",
    )
    .unwrap();
    assert_eq!(v["vcov_method"], h.vcov_method.as_str());
    assert_eq!(v["warnings"], serde_json::json!([]));
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "chi2_statistic",
            "df",
            "p_value",
            "recommendation",
            "test",
            "vcov_method",
            "warnings"
        ]
    );
}

#[test]
fn test_validate_cli_hausman_sim_corr() {
    let case = Case::load("panel_hausman_sim_corr");
    let s = session(&case.dataset_path());
    let out = run(&s, &HAUSMAN);
    Check::new(&case)
        .scalar(
            "chisq",
            num_after(&out, "", "Chi-squared statistic:"),
            Tol::printed(4),
        )
        .scalar("p", num_after(&out, "", "P-value:"), Tol::printed(4))
        .finish();
    let h = run_hausman_test(
        &dataset(&case.dataset_path()),
        "inv",
        &["value", "capital"],
        "firm",
    )
    .unwrap();
    assert!(
        !h.warnings.is_empty(),
        "Var(b_FE) - Var(b_RE) is indefinite here and should be warned about"
    );
    let after_recommendation = out
        .split_once("\nRecommendation: ")
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("no recommendation in\n{out}"));
    assert_eq!(warnings_block(after_recommendation), h.warnings, "{out}");

    let v = json(&run(&s, &as_json(&HAUSMAN)));
    Check::new(&case)
        .scalar("chisq", json_f64(&v["chi2_statistic"]), Tol::stat(300))
        .finish();
    assert_eq!(v["warnings"], serde_json::json!(h.warnings), "{v}");
}
