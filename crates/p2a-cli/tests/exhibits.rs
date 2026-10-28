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

use p2a_validation::golden::assert_golden;
use p2a_validation::text::coefficient_rows;
use p2a_validation::{Case, Tol};
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
