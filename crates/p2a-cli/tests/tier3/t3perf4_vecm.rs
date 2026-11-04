//! T3-PERF4: `p2a ts vecm` without `--lags`.
//!
//! `run_vecm` takes the lag order of the levels VAR, which must be at least 2 (`urca::ca.jo`'s
//! `K >= 2`); the CLI's default `--lags` has to be a value it accepts.

use crate::common::temp_session;
use serde_json::Value;
use std::fmt::Write as _;

/// A small noisy cointegrated pair: x a random walk, y = 0.8 x + stationary noise.
fn cointegrated_csv() -> String {
    let mut s = String::from("x,y\n");
    let (mut x, mut state) = (0.0_f64, 0x9E37_79B9_7F4A_7C15_u64);
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for _ in 0..80 {
        x += noise();
        let y = 0.8 * x + 0.3 * noise();
        writeln!(s, "{x:.10},{y:.10}").unwrap();
    }
    s
}

/// Fails before: the default `--lags 1` was refused by `run_vecm` (lag order below 2).
#[test]
fn test_ts_vecm_default_lags_is_accepted() {
    let s = temp_session();
    let csv = s.dir().join("coint.csv");
    std::fs::write(&csv, cointegrated_csv()).unwrap();
    s.load_as(&csv, "coint", &[]);
    let out = s.ok(&[
        "--format", "json", "ts", "vecm", "coint", "--cols", "x", "y",
    ]);
    let v: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    assert_eq!(v["method"], "VECM", "{out}");
    assert_eq!(v["lags"], 2, "{out}");
    assert_eq!(v["rank"], 1, "{out}");
    assert_eq!(v["n_vars"], 2, "{out}");
}
