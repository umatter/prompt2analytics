//! Number, row and JSON checks for the MCP-path tests of T3-WP58, shared by the tier3 module wp58_models and
//! tests/integration_tests.rs (included there with #[path]). Owned by T3-WP58.
#![allow(dead_code)]
use p2a_mcp::transport::http::{ContentItem, ToolResult};
use p2a_validation::{Case, Tol};
use serde_json::Value;

/// The text items of a tool result, joined with "\n".
pub fn result_text(r: &ToolResult) -> String {
    r.content
        .iter()
        .filter_map(|c| match c {
            ContentItem::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Numbers printed in `s`, each with half a unit of its last printed digit. Decimal or scientific notation with an
/// optional sign; whitespace and , ; : = ( ) [ ] | separate tokens; trailing '*', '%' and '.' are dropped; tokens
/// that do not parse as finite numbers (names, NaN, NA, -) are skipped.
pub fn printed(s: &str) -> Vec<(f64, f64)> {
    s.split(|c: char| c.is_whitespace() || ",;:=()[]|".contains(c))
        .filter_map(|raw| {
            let tok = raw.trim_end_matches(['*', '%', '.']);
            if !tok.chars().any(|c| c.is_ascii_digit()) {
                return None;
            }
            let v: f64 = tok.parse().ok().filter(|v: &f64| v.is_finite())?;
            let (mantissa, exp) = match tok.find(['e', 'E']) {
                Some(i) => (&tok[..i], tok[i + 1..].parse::<i32>().ok()?),
                None => (tok, 0),
            };
            let decimals = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i32);
            Some((v, 0.5 * 10f64.powi(exp - decimals)))
        })
        .collect()
}

/// A printed value `v` shows `expected` when it is within half a unit of its last digit plus 1e-6 relative (p2a and R
/// may differ by ~1e-8 and land on two sides of a rounding boundary), or within `tol`.
fn close(v: f64, half: f64, expected: f64, tol: Tol) -> bool {
    (v - expected).abs() <= half + 1e-6 * expected.abs() + 1e-12 || tol.allows(v, expected)
}

/// `expected` is shown in `s`: some printed number lies within its half-unit (or `tol`) of it.
#[track_caller]
pub fn assert_shows(what: &str, s: &str, expected: f64, tol: Tol) {
    assert!(
        printed(s).iter().any(|&(v, h)| close(v, h, expected, tol)),
        "{what}: {expected} is not shown in:\n{s}"
    );
}

/// Every expected value is shown on `line`, left to right (so a coefficient precedes its SE).
#[track_caller]
pub fn assert_row(what: &str, line: &str, expected: &[f64], tol: Tol) {
    let nums = printed(line);
    let mut from = 0;
    for &e in expected {
        match nums[from..].iter().position(|&(v, h)| close(v, h, e, tol)) {
            Some(i) => from += i + 1,
            None => panic!("{what}: {e} (of {expected:?}, in order) is not shown on:\n{line}"),
        }
    }
}

/// The first line whose trimmed text starts with `label` followed by whitespace.
#[track_caller]
pub fn row<'a>(text: &'a str, label: &str) -> &'a str {
    text.lines()
        .find(|l| {
            l.trim_start()
                .strip_prefix(label)
                .is_some_and(|rest| rest.starts_with(char::is_whitespace))
        })
        .unwrap_or_else(|| panic!("no row '{label}' in:\n{text}"))
}

/// The text after the line that contains the first `marker`.
#[track_caller]
pub fn section<'a>(text: &'a str, marker: &str) -> &'a str {
    let i = text
        .find(marker)
        .unwrap_or_else(|| panic!("no '{marker}' in:\n{text}"));
    let rest = &text[i..];
    rest.find('\n').map_or("", |j| &rest[j + 1..])
}

/// The rest of the line after the first `label`.
#[track_caller]
pub fn after<'a>(text: &'a str, label: &str) -> &'a str {
    let i = text
        .find(label)
        .unwrap_or_else(|| panic!("no '{label}' in:\n{text}"));
    text[i + label.len()..].split('\n').next().unwrap_or("")
}

/// The JSON document of a tool's text: the whole text, or the block after "JSON:\n".
#[track_caller]
pub fn json_of(r: &ToolResult) -> Value {
    let t = result_text(r);
    let body = t.split_once("JSON:\n").map_or(t.as_str(), |(_, j)| j);
    serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}):\n{t}"))
}

/// The error text of a tool error (panics when the tool succeeded).
#[track_caller]
pub fn tool_error(r: &ToolResult) -> String {
    assert!(
        !r.success,
        "expected a tool error, got:\n{}",
        result_text(r)
    );
    r.error.clone().unwrap_or_else(|| result_text(r))
}

/// A JSON array of numbers (null -> NaN).
pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

/// The case's analysis dataset, read as the server reads CSV files.
pub fn dataset_of(case: &Case) -> p2a_core::data::Dataset {
    p2a_core::data::Dataset::new(
        p2a_core::data::DataLoader::load_csv(case.dataset_path()).expect("case dataset"),
    )
}
