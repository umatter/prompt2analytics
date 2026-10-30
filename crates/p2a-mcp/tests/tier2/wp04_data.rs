//! T2-WP04 MCP-path tests (data layer: quality profile, binning, casts, missing-value fills,
//! mutate operands, interactive charts with a missing group). Owned by T2-WP04.

use crate::common::{self, text};
use p2a_core::data::Dataset;
use p2a_validation::{Case, Check, Tol};
use polars::prelude::*;
use serde_json::json;

/// The bin index of each row of a `bin_<k>` label column, -1 for a null (NA) label: the
/// coding of the R case `data_bin_cut`.
fn codes_of(ds: &Dataset, col: &str) -> Vec<f64> {
    ds.df()
        .column(col)
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|label| match label {
            None => -1.0,
            Some(l) => l
                .strip_prefix("bin_")
                .and_then(|k| k.parse::<f64>().ok())
                .unwrap_or_else(|| panic!("label {l:?} is not bin_<k>")),
        })
        .collect()
}

#[tokio::test]
async fn test_mcp_data_quality_profile_counts_nan() {
    // WT-9. R: y <- c(1, NaN, NaN, 4, NaN); sum(is.na(y)) = 3; median(y, na.rm = TRUE) = 2.5
    let y = Dataset::new(df!("y" => [1.0, f64::NAN, f64::NAN, 4.0, f64::NAN]).unwrap());
    let (session, server) = common::session_with_datasets(vec![("q", y)]);
    let r = common::call(
        &server,
        &session,
        "data_quality_profile",
        json!({"dataset": "q"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(
        t.contains("Null: 3 (60.0%)")
            && t.contains("Median: 2.50")
            && t.contains("Completeness: 40.0%"),
        "{t}"
    );
}

#[tokio::test]
async fn test_validate_mcp_munge_bin_cut() {
    // validation/r/cases/data_quality.R, case data_bin_cut: cut(x, breaks = c(0, 1, 3, 5))
    // with right = TRUE / include.lowest = TRUE / right = FALSE.
    let case = Case::load("data_bin_cut");
    let (session, server) = common::session_with("data_bin_cut");
    for (key, right, include_lowest) in [
        ("right_codes", true, false),
        ("include_lowest_codes", true, true),
        ("left_closed_codes", false, false),
    ] {
        let r = common::call(
            &server,
            &session,
            "munge_bin",
            json!({"dataset": "bin_cut", "column": "x", "strategy": "custom",
                   "bins": [0, 1, 3, 5], "right": right, "include_lowest": include_lowest,
                   "result_name": key}),
        )
        .await;
        let t = text(&r);
        assert!(r.success, "{key}: {t}");
        if key == "right_codes" {
            // R: sum(is.na(cut(x, c(0, 1, 3, 5)))) = 5, of which sum(is.na(x)) = 2
            assert!(t.contains("NA: 5 (2 missing, 3 outside the edges)"), "{t}");
        }
        let ds = session.datasets.read().await.get(key).unwrap().clone();
        Check::new(&case)
            .vector(key, &codes_of(&ds, "x_binned"), Tol::exact())
            .finish();
    }
}

#[tokio::test]
async fn test_mcp_munge_fill_na_all_missing_is_an_error() {
    // DATA-35. R: mean(c(NA_real_, NA_real_, NA_real_), na.rm = TRUE) is NaN, not 0.
    let e = Dataset::new(df!("id" => [1i64, 2, 3], "z" => [None::<f64>, None, None]).unwrap());
    let (session, server) = common::session_with_datasets(vec![("e", e)]);
    let r = common::call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "e", "columns": ["z"], "strategy": "mean"}),
    )
    .await;
    let t = text(&r);
    assert!(!r.success && t.contains("no observed values"), "{t}");
}

#[tokio::test]
async fn test_mcp_munge_cast_reports_failed_values() {
    // DATA-37. R: as.integer(c("1,234", "7", "x")) is NA 7 NA with a coercion warning.
    let c = Dataset::new(df!("v" => ["1,234", "7", "x"]).unwrap());
    let (session, server) = common::session_with_datasets(vec![("c", c)]);
    let args = |strict: bool| {
        json!({"dataset": "c", "column": "v", "dtype": "int", "strict": strict,
               "result_name": "c2"})
    };
    let r = common::call(&server, &session, "munge_cast", args(false)).await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(
        t.contains(
            "Warning: column 'v': 2 values could not be converted to int and are now missing \
             (first: '1,234', 'x')"
        ),
        "{t}"
    );
    let r = common::call(&server, &session, "munge_cast", args(true)).await;
    let t = text(&r);
    assert!(
        !r.success && t.contains("2 values could not be converted to int"),
        "{t}"
    );
}

#[tokio::test]
async fn test_mcp_munge_mutate_refuses_text() {
    // DATA-37. R: c("$5", "10", "2,000") * c(6, 3, 1.5) is an error
    // ("non-numeric argument to binary operator").
    let m = Dataset::new(df!("p" => ["$5", "10", "2,000"], "q" => [6.0, 3.0, 1.5]).unwrap());
    let (session, server) = common::session_with_datasets(vec![("m", m)]);
    let r = common::call(
        &server,
        &session,
        "munge_mutate",
        json!({"dataset": "m", "new_column": "pq", "expr_type": "arithmetic",
               "left": "p", "operator": "*", "right": "q"}),
    )
    .await;
    let t = text(&r);
    assert!(!r.success && t.contains("numeric"), "{t}");
}
