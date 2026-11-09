//! T3-WP57 MCP-path tests (machine-learning and spatial surface). Owned by T3-WP57.
//! Every test drives AnalyticsServer::call_tool_with_session through the shared helpers and pins
//! the result to an R case (validation/methods/mcp_ml_spatial.json) or to a stated property.
use crate::common::*;
use p2a_core::polars::prelude::*;
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_mcp::transport::http::ToolResult;
use p2a_validation::{Case, Tol};
use serde_json::{Value, json};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn case_df(id: &str) -> (Case, DataFrame) {
    let case = Case::load(id);
    let df = p2a_core::DataLoader::load_csv(&case.dataset_path()).expect("case dataset loads");
    (case, df)
}

fn dataset(df: DataFrame) -> p2a_core::Dataset {
    p2a_core::Dataset::new(df)
}

/// Text of a successful call (panics with the message otherwise).
fn ok(r: &ToolResult) -> String {
    assert!(r.success, "tool error: {} {:?}", text(r), r.error);
    text(r)
}

/// Message of a tool error (panics on success).
fn err(r: &ToolResult) -> String {
    assert!(!r.success, "expected a tool error, got:\n{}", text(r));
    let t = text(r);
    if t.is_empty() {
        r.error.clone().unwrap_or_default()
    } else {
        t
    }
}

fn json_block(t: &str) -> Value {
    let j = t
        .split("JSON Summary:\n")
        .nth(1)
        .unwrap_or_else(|| panic!("no JSON Summary in:\n{t}"));
    serde_json::from_str(j).unwrap_or_else(|e| panic!("JSON Summary does not parse ({e}):\n{j}"))
}

/// The first line containing `needle`.
fn line<'a>(t: &'a str, needle: &str) -> &'a str {
    t.lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line with {needle:?} in:\n{t}"))
}

/// The line after the first line containing `header`.
fn line_after<'a>(t: &'a str, header: &str) -> &'a str {
    let mut it = t.lines().skip_while(|l| !l.contains(header));
    it.next()
        .unwrap_or_else(|| panic!("no {header:?} in:\n{t}"));
    it.next()
        .unwrap_or_else(|| panic!("nothing after {header:?} in:\n{t}"))
}

/// The number right after `key` in `l`.
fn num_after(l: &str, key: &str) -> f64 {
    let at = l
        .find(key)
        .unwrap_or_else(|| panic!("{key:?} not in {l:?}"))
        + key.len();
    let tok: String = l[at..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
        .collect();
    tok.parse()
        .unwrap_or_else(|_| panic!("no number after {key:?} in {l:?}"))
}

/// The numbers after the name on the first line whose first token is `name` (significance
/// stars, parentheses and commas stripped).
fn row(t: &str, name: &str) -> Vec<f64> {
    let l = t
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{t}"));
    l.split_whitespace()
        .skip(1)
        .filter_map(|w| {
            w.trim_end_matches('*')
                .trim_matches(|c| c == '(' || c == ')' || c == ',')
                .parse()
                .ok()
        })
        .collect()
}

fn printed(actual: f64, r: f64, decimals: u32, what: &str) {
    assert!(
        Tol::printed(decimals).allows(actual, r),
        "{what}: printed {actual}, R {r}"
    );
}

fn close(actual: f64, r: f64, tol: Tol, what: &str) {
    assert!(tol.allows(actual, r), "{what}: {actual} vs R {r}");
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn with(base: &Value, extra: Value) -> Value {
    let mut a = base.clone();
    for (k, v) in extra.as_object().unwrap() {
        a[k] = v.clone();
    }
    a
}

/// spatial_neighbors on `ds` with coordinates cx, cy and `extra` request fields; returns the text.
async fn neighbors(
    server: &AnalyticsServer,
    s: &Session,
    ds: &str,
    name: &str,
    extra: Value,
) -> String {
    let a = with(
        &json!({"dataset": ds, "x_coord": "cx", "y_coord": "cy", "weights_name": name}),
        extra,
    );
    ok(&call(server, s, "spatial_neighbors", a).await)
}

/// a and b describe the same partition (a bijection of labels).
fn same_partition(a: &[i64], b: &[f64]) -> bool {
    use std::collections::HashMap;
    let (mut ab, mut ba) = (HashMap::new(), HashMap::new());
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            let y = y as i64;
            *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x
        })
}

fn int_column(s: &Session, ds: &str, col: &str) -> Vec<Option<i64>> {
    let map = s.datasets.try_read().expect("no lock held after the call");
    map[ds]
        .df()
        .column(col)
        .unwrap_or_else(|_| panic!("{ds} has no column {col}"))
        .i64()
        .expect("Int64")
        .into_iter()
        .collect()
}

fn float_column(s: &Session, ds: &str, col: &str) -> Vec<Option<f64>> {
    let map = s.datasets.try_read().expect("no lock held after the call");
    map[ds]
        .df()
        .column(col)
        .unwrap_or_else(|_| panic!("{ds} has no column {col}"))
        .f64()
        .expect("Float64")
        .into_iter()
        .collect()
}

/// The description of `field` in the input schema of `tool`.
fn field_description(tool: &str, field: &str) -> String {
    let tools = AnalyticsServer::new().router_tools();
    let t = tools
        .iter()
        .find(|t| t.name == tool)
        .unwrap_or_else(|| panic!("{tool}"));
    Value::Object((*t.input_schema).clone())["properties"][field]["description"]
        .as_str()
        .unwrap_or_else(|| panic!("{tool}.{field}"))
        .to_string()
}

fn tool_description(tool: &str) -> String {
    let tools = AnalyticsServer::new().router_tools();
    tools
        .iter()
        .find(|t| t.name == tool)
        .unwrap_or_else(|| panic!("{tool}"))
        .description
        .clone()
        .unwrap_or_default()
        .to_string()
}

/// Every name in `names` appears quoted in the description of `tool.field`.
fn lists_every_value(tool: &str, field: &str, names: Vec<&str>) {
    let d = field_description(tool, field);
    for n in names {
        assert!(
            d.contains(&format!("'{n}'")),
            "{tool}.{field} lacks '{n}': {d}"
        );
    }
}

/// Column `c` of `df` as f64 (complete case data).
fn col_f64(df: &DataFrame, c: &str) -> Vec<f64> {
    df.column(c)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.expect("complete case data"))
        .collect()
}
