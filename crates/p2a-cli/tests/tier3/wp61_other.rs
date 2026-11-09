//! T3-WP61: causal, time-series, ML, spatial and viz commands through the p2a CLI.
//! Option values are typed (CLI-14), documented flags act (CLI-23, CLI-10, CLI-11), notes and the
//! wave-2 fields reach text and JSON (SPAT-25, CLI-17 causal part). Every numeric pin is an R case.
use crate::common::*;
use serde_json::Value;

/// `p2a <args>` without a session: clap validates option values while parsing, before any
/// dataset lookup. Asserts a usage error (exit 2) without a panic; returns stderr.
fn usage_error(args: &[&str]) -> String {
    let out = p2a().args(args).output().expect("run p2a");
    assert_no_panic(&out, &args.join(" "));
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(
        out.status.code(),
        Some(2),
        "{args:?}: want exit 2, got {:?}\nstdout: {}\nstderr: {stderr}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
    stderr
}

/// Each row must be a usage error whose message contains the needle.
fn assert_usage_errors(rows: &[(Vec<&str>, &str)]) {
    for (args, needle) in rows {
        let err = usage_error(args);
        assert!(
            err.contains(needle),
            "{args:?}: stderr lacks {needle:?}:\n{err}"
        );
    }
}

/// `v[key]`, with the document in the panic message when absent.
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("no key {key:?} in {v:#}"))
}

/// serde's `Vec<(String, f64)>` ([[name, value], ..]) as (names, values); null -> NaN.
fn pairs(v: &Value) -> (Vec<String>, Vec<f64>) {
    let a = v.as_array().unwrap_or_else(|| panic!("not an array: {v}"));
    a.iter()
        .map(|p| {
            (
                p[0].as_str().unwrap().to_string(),
                p[1].as_f64().unwrap_or(f64::NAN),
            )
        })
        .unzip()
}

/// Rows of numbers after the line equal (trimmed) to `header`, skipping `skip` title lines, up
/// to a blank line.
fn table_rows(text: &str, header: &str, skip: usize) -> Vec<Vec<f64>> {
    text.lines()
        .skip_while(|l| l.trim() != header)
        .skip(1 + skip)
        .take_while(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| {
                    t.parse::<f64>()
                        .unwrap_or_else(|_| panic!("{t:?} in {l:?}"))
                })
                .collect()
        })
        .collect()
}

/// True when two label vectors induce the same partition (labels may be numbered differently).
fn same_partition(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && (0..a.len()).all(|i| (0..a.len()).all(|j| (a[i] == a[j]) == (b[i] == b[j])))
}

/// Every key of `keys` is present.
fn assert_keys(v: &Value, keys: &[&str]) {
    for k in keys {
        assert!(v.get(*k).is_some(), "no key {k:?} in {v:#}")
    }
}

/// Names of a JSON array of strings.
fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|s| {
            s.as_str()
                .unwrap_or_else(|| panic!("not a string: {s}"))
                .to_string()
        })
        .collect()
}

/// Loads a committed dataset (path relative to the repository root) under `name`, for tests
/// that need a dataset but pin no R value.
fn load_repo_csv(s: &TempSession, rel: &str, name: &str) -> String {
    s.load_as(&repo_path(rel), name, &[]);
    name.to_string()
}
