//! Helpers shared by the Tier 3 CLI test modules. The Tier 2 helpers are reused verbatim (`p2a()`,
//! `temp_session()`, `TempSession`, `assert_no_panic`, `assert_same_csv`, and the `p2a_validation`
//! re-exports `Case`, `Check`, `Tol`, `repo_path`); this file adds coefficient tables and in-process
//! loading.

#[path = "../tier2/common.rs"]
mod tier2;

pub use p2a_validation::text::{coefficient_rows, labelled};
pub use tier2::*;

use serde_json::Value;
use std::process::Output;

/// A coefficient table: names and the four numeric columns (estimate, SE, t or z, p).
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub names: Vec<String>,
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    pub stat: Vec<f64>,
    pub p: Vec<f64>,
}

impl Table {
    fn from_rows(rows: impl IntoIterator<Item = (String, [f64; 4])>) -> Table {
        let mut t = Table {
            names: Vec::new(),
            coef: Vec::new(),
            se: Vec::new(),
            stat: Vec::new(),
            p: Vec::new(),
        };
        for (name, [b, se, stat, p]) in rows {
            t.names.push(name);
            t.coef.push(b);
            t.se.push(se);
            t.stat.push(stat);
            t.p.push(p);
        }
        t
    }
}

/// The coefficient table of a `format_regression_report` text output; panics unless every row has
/// four numbers.
pub fn text_table(text: &str) -> Table {
    Table::from_rows(coefficient_rows(text).into_iter().map(|(name, v)| {
        let n: [f64; 4] = v
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("row {name} does not have 4 numbers: {v:?}"));
        (name, n)
    }))
}

/// A number, `null` read as NaN; panics on anything else.
fn num_or_nan(v: &Value) -> f64 {
    if v.is_null() {
        f64::NAN
    } else {
        v.as_f64()
            .unwrap_or_else(|| panic!("not a number or null: {v}"))
    }
}

/// `v["coefficients"][i]{name, coefficient, std_error, <stat_key>, p_value}`; `null` -> NaN.
pub fn json_table(v: &Value, stat_key: &str) -> Table {
    let rows = v["coefficients"]
        .as_array()
        .unwrap_or_else(|| panic!("no coefficients array in {v}"));
    Table::from_rows(rows.iter().map(|r| {
        let name = r["name"]
            .as_str()
            .unwrap_or_else(|| panic!("coefficient row without a name: {r}"))
            .to_string();
        (
            name,
            [
                num_or_nan(&r["coefficient"]),
                num_or_nan(&r["std_error"]),
                num_or_nan(&r[stat_key]),
                num_or_nan(&r["p_value"]),
            ],
        )
    }))
}

/// `v["names"]`, `v["coefficients"]`, `v["std_errors"]`, `v["t_values"]`, `v["p_values"]` as a table;
/// panics with the JSON when `names` is absent.
pub fn json_arrays(v: &Value) -> Table {
    let names: Vec<String> = v["names"]
        .as_array()
        .unwrap_or_else(|| panic!("no names array in {v}"))
        .iter()
        .map(|n| n.as_str().expect("names are strings").to_string())
        .collect();
    Table {
        coef: f64s(&v["coefficients"]),
        se: f64s(&v["std_errors"]),
        stat: f64s(&v["t_values"]),
        p: f64s(&v["p_values"]),
        names,
    }
}

/// A JSON array of numbers, `null` read as NaN; panics otherwise.
pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(num_or_nan)
        .collect()
}

/// A JSON number; panics unless it is one.
pub fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// The case's dataset loaded in process, as `data load` reads a CSV file.
pub fn case_dataset(case: &Case) -> p2a_core::Dataset {
    let path = case.dataset_path();
    let df =
        p2a_core::DataLoader::load_csv(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    p2a_core::Dataset::new(df)
}

/// The lines of the command's stderr.
pub fn stderr_lines(out: &Output) -> Vec<String> {
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .map(str::to_owned)
        .collect()
}

/// `args` followed by `extra`.
pub fn argv<'a>(args: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    let mut all = args.to_vec();
    all.extend_from_slice(extra);
    all
}
