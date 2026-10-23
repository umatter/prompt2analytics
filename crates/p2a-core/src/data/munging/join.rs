//! Join operations for combining datasets.
//!
//! This module provides various join operations (left, right, inner, full, anti, semi)
//! as well as concatenation operations (vertical and horizontal).
//!
//! # Join Types
//!
//! - **Left Join**: Keep all rows from left, matching rows from right
//! - **Right Join**: Keep all rows from right, matching rows from left
//! - **Inner Join**: Keep only rows that match in both datasets
//! - **Full Join**: Keep all rows from both datasets
//! - **Anti Join**: Keep rows from left that have no match in right
//! - **Semi Join**: Keep rows from left that have a match in right (no right columns)
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Join two datasets
//! let result = left_join(&orders, &customers, &["customer_id"], None, Some("_right"))?;
//!
//! // Concatenate datasets vertically
//! let combined = concat(&[&df1, &df2, &df3])?;
//!
//! // Concatenate datasets horizontally
//! let wide = hconcat(&[&df1, &df2])?;
//! ```

use super::CHECKPOINT_EVERY;
use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;

/// Type of join operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Keep all rows from left, matching rows from right
    Left,
    /// Keep all rows from right, matching rows from left
    Right,
    /// Keep only rows that match in both datasets
    Inner,
    /// Keep all rows from both datasets
    Full,
    /// Keep rows from left that have no match in right
    Anti,
    /// Keep rows from left that have a match in right (no right columns)
    Semi,
}

impl JoinType {
    /// Convert to Polars JoinType
    fn to_polars(&self) -> polars::prelude::JoinType {
        match self {
            JoinType::Left => polars::prelude::JoinType::Left,
            JoinType::Right => polars::prelude::JoinType::Right,
            JoinType::Inner => polars::prelude::JoinType::Inner,
            JoinType::Full => polars::prelude::JoinType::Full,
            JoinType::Anti => polars::prelude::JoinType::Anti,
            JoinType::Semi => polars::prelude::JoinType::Semi,
        }
    }
}

/// Join two datasets on specified columns.
///
/// # Arguments
///
/// * `left` - Left dataset
/// * `right` - Right dataset
/// * `left_on` - Column names to join on from left dataset; at least one (an
///   empty list is refused with [`MungeError::JoinError`])
/// * `right_on` - Column names to join on from right dataset (if None, uses
///   left_on; must not be empty when given, so `Some(&[])` is refused)
/// * `join_type` - Type of join (Left, Right, Inner, Full, Anti, Semi)
/// * `suffix` - Suffix for duplicate column names from right dataset
///
/// # Errors
///
/// [`MungeError::JoinError`] for an empty `left_on`, a `right_on` of
/// `Some(&[])`, or `left_on` and `right_on` of different lengths;
/// [`MungeError::ColumnNotFound`] for a key missing from its dataset.
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{join, JoinType};
///
/// // Inner join on customer_id
/// let result = join(&orders, &customers, &["customer_id"], None, JoinType::Inner, Some("_right"))?;
///
/// // Left join with different column names
/// let result = join(&orders, &customers, &["cust_id"], Some(&["customer_id"]), JoinType::Left, None)?;
/// ```
pub fn join(
    left: &Dataset,
    right: &Dataset,
    left_on: &[&str],
    right_on: Option<&[&str]>,
    join_type: JoinType,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join_with(
        left,
        right,
        left_on,
        right_on,
        join_type,
        suffix,
        &JoinOptions::default(),
    )
}

/// Opt-in output ceiling for joins.
///
/// `Default` means no ceiling, which is what [`join`] and [`cross_join`] use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JoinOptions {
    /// Refuse a join whose result would have more rows than this.
    pub max_output_rows: Option<u64>,
}

/// Validate the key columns of a join and return the right-hand keys
/// (`right_on`, or `left_on` when it is `None`).
///
/// An empty key list is refused: polars panics when it joins on zero keys.
fn join_keys<'a>(
    left: &Dataset,
    right: &Dataset,
    left_on: &[&'a str],
    right_on: Option<&[&'a str]>,
) -> MungeResult<Vec<&'a str>> {
    if left_on.is_empty() {
        return Err(MungeError::JoinError(
            "a join needs at least one key column, but left_on is empty".to_string(),
        ));
    }
    if matches!(right_on, Some([])) {
        return Err(MungeError::JoinError(
            "a join needs at least one key column, but right_on is empty; omit it to join on \
             the left_on names"
                .to_string(),
        ));
    }

    // Validate left_on columns exist
    for col in left_on {
        if left.df().column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Determine right_on columns
    let right_on_cols: Vec<&str> = match right_on {
        Some(cols) => cols.to_vec(),
        None => left_on.to_vec(),
    };

    // Validate right_on columns exist
    for col in &right_on_cols {
        if right.df().column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Check column count matches
    if left_on.len() != right_on_cols.len() {
        return Err(MungeError::JoinError(format!(
            "left_on has {} columns but right_on has {} columns",
            left_on.len(),
            right_on_cols.len()
        )));
    }

    Ok(right_on_cols)
}

/// The number of rows `join(left, right, left_on, right_on, join_type, _)`
/// would return, computed from per-key row counts on both sides (two hash
/// group-bys and a join of the two count tables; the data are not joined).
///
/// Null keys never match, as in [`join`]. The count saturates at `u64::MAX`.
///
/// # Errors
///
/// The keys are validated as in [`join`]: an empty `left_on`, or a
/// `right_on` of `Some(&[])`, returns [`MungeError::JoinError`].
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{estimate_join_rows, JoinType};
///
/// let rows = estimate_join_rows(&orders, &customers, &["customer_id"], None, JoinType::Left)?;
/// ```
pub fn estimate_join_rows(
    left: &Dataset,
    right: &Dataset,
    left_on: &[&str],
    right_on: Option<&[&str]>,
    join_type: JoinType,
) -> MungeResult<u64> {
    const LEFT_N: &str = "__p2a_left_n";
    const RIGHT_N: &str = "__p2a_right_n";

    let right_keys = join_keys(left, right, left_on, right_on)?;
    let join_err = |e: PolarsError| MungeError::JoinError(e.to_string());

    // Per-key row counts on each side, with eager group-bys: polars runs a
    // lazy group-by of this shape on its streaming engine, which blocks on a
    // tokio runtime of its own and therefore panics when it is called from
    // inside another one, as every MCP tool call is.
    let key_counts = |ds: &Dataset, keys: &[&str], name: &str| -> MungeResult<DataFrame> {
        let groups = ds.df().group_by(keys.iter().copied()).map_err(join_err)?;
        let mut columns = groups.keys();
        let counts = groups.get_groups().group_count().with_name(name.into());
        columns.push(counts.into_series().into_column());
        DataFrame::new(columns).map_err(join_err)
    };
    let left_counts = key_counts(left, left_on, LEFT_N)?;
    let right_counts = key_counts(right, &right_keys, RIGHT_N)?;
    // Null keys never match (the JoinArgs default), as in `join`.
    let matched = left_counts
        .join(
            &right_counts,
            left_on.iter().copied(),
            right_keys.iter().copied(),
            JoinArgs::new(polars::prelude::JoinType::Inner),
            None,
        )
        .map_err(join_err)?;

    let counts = |name: &str| -> MungeResult<UInt64Chunked> {
        let column = matched
            .column(name)
            .and_then(|c| c.cast(&DataType::UInt64))
            .map_err(|e| MungeError::JoinError(e.to_string()))?;
        column
            .u64()
            .cloned()
            .map_err(|e| MungeError::JoinError(e.to_string()))
    };
    let (left_n, right_n) = (counts(LEFT_N)?, counts(RIGHT_N)?);

    // inner = sum of l * r over matching keys; ml / mr = rows of each side
    // that have a match.
    let (mut inner, mut ml, mut mr) = (0u64, 0u64, 0u64);
    for (i, (l, r)) in left_n.into_iter().zip(right_n.into_iter()).enumerate() {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(|e| MungeError::JoinError(e.to_string()))?;
        }
        let (l, r) = (l.unwrap_or(0), r.unwrap_or(0));
        inner = inner.saturating_add(l.saturating_mul(r));
        ml = ml.saturating_add(l);
        mr = mr.saturating_add(r);
    }
    let unmatched_left = (left.nrows() as u64).saturating_sub(ml);
    let unmatched_right = (right.nrows() as u64).saturating_sub(mr);

    Ok(match join_type {
        JoinType::Inner => inner,
        JoinType::Left => inner.saturating_add(unmatched_left),
        JoinType::Right => inner.saturating_add(unmatched_right),
        JoinType::Full => inner
            .saturating_add(unmatched_left)
            .saturating_add(unmatched_right),
        JoinType::Semi => ml,
        JoinType::Anti => unmatched_left,
    })
}

/// [`join`] with an optional ceiling on the number of result rows.
///
/// When `opts.max_output_rows` is set, the result size is predicted with
/// [`estimate_join_rows`] first and a join above the ceiling is refused with
/// [`MungeError::JoinError`] naming both numbers. Without a ceiling this is
/// exactly [`join`].
///
/// # Errors
///
/// The keys are validated as in [`join`] before any estimate or join runs:
/// an empty `left_on`, or a `right_on` of `Some(&[])`, returns
/// [`MungeError::JoinError`].
pub fn join_with(
    left: &Dataset,
    right: &Dataset,
    left_on: &[&str],
    right_on: Option<&[&str]>,
    join_type: JoinType,
    suffix: Option<&str>,
    opts: &JoinOptions,
) -> MungeResult<Dataset> {
    let right_on_cols = join_keys(left, right, left_on, right_on)?;

    if let Some(max) = opts.max_output_rows {
        let rows = estimate_join_rows(left, right, left_on, right_on, join_type)?;
        if rows > max {
            return Err(MungeError::JoinError(format!(
                "the join would return {rows} rows, more than the allowed {max}; join on more \
                 specific keys or remove duplicate keys first"
            )));
        }
    }

    // Convert to owned strings for Polars (PlSmallStr needs owned data)
    let left_on_str: Vec<String> = left_on.iter().map(|s| s.to_string()).collect();
    let right_on_str: Vec<String> = right_on_cols.iter().map(|s| s.to_string()).collect();

    // Set up join arguments
    let suffix_str: PlSmallStr = suffix.unwrap_or("_right").into();

    // Perform the join
    let result = left
        .df()
        .join(
            right.df(),
            left_on_str.as_slice(),
            right_on_str.as_slice(),
            JoinArgs::new(join_type.to_polars()).with_suffix(Some(suffix_str)),
            None, // JoinTypeOptions
        )
        .map_err(|e| MungeError::JoinError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Left join: Keep all rows from left, matching rows from right.
///
/// # Example
///
/// ```ignore
/// let result = left_join(&orders, &customers, &["customer_id"], None, Some("_cust"))?;
/// ```
pub fn left_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Left, suffix)
}

/// Right join: Keep all rows from right, matching rows from left.
///
/// # Example
///
/// ```ignore
/// let result = right_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn right_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Right, suffix)
}

/// Inner join: Keep only rows that match in both datasets.
///
/// # Example
///
/// ```ignore
/// let result = inner_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn inner_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Inner, suffix)
}

/// Full (outer) join: Keep all rows from both datasets.
///
/// # Example
///
/// ```ignore
/// let result = full_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn full_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Full, suffix)
}

/// Anti join: Keep rows from left that have NO match in right.
///
/// # Example
///
/// ```ignore
/// // Find orders without matching customers
/// let orphan_orders = anti_join(&orders, &customers, &["customer_id"], None)?;
/// ```
pub fn anti_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Anti, None)
}

/// Semi join: Keep rows from left that have a match in right (no right columns added).
///
/// # Example
///
/// ```ignore
/// // Find orders that have matching customers (but don't add customer columns)
/// let matched_orders = semi_join(&orders, &customers, &["customer_id"], None)?;
/// ```
pub fn semi_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Semi, None)
}

/// Concatenate datasets vertically (stack rows).
///
/// All datasets must have the same columns. Missing columns are filled with nulls
/// if `fill_null` is true.
///
/// # Arguments
///
/// * `datasets` - Slice of datasets to concatenate
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::concat;
///
/// let combined = concat(&[&jan_data, &feb_data, &mar_data])?;
/// ```
pub fn concat(datasets: &[&Dataset]) -> MungeResult<Dataset> {
    if datasets.is_empty() {
        return Err(MungeError::EmptyDataset);
    }

    if datasets.len() == 1 {
        return Ok(datasets[0].clone());
    }

    // Get DataFrames
    let dfs: Vec<DataFrame> = datasets.iter().map(|ds| ds.df().clone()).collect();

    // Use diagonal concat to handle missing columns (fills with nulls)
    let result = polars::functions::concat_df_diagonal(&dfs)
        .map_err(|e| MungeError::JoinError(format!("Concat failed: {}", e)))?;

    Ok(Dataset::new(result))
}

/// Concatenate datasets horizontally (add columns side by side).
///
/// All datasets must have the same number of rows.
///
/// # Arguments
///
/// * `datasets` - Slice of datasets to concatenate horizontally
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::hconcat;
///
/// let wide = hconcat(&[&demographics, &survey_responses])?;
/// ```
pub fn hconcat(datasets: &[&Dataset]) -> MungeResult<Dataset> {
    if datasets.is_empty() {
        return Err(MungeError::EmptyDataset);
    }

    if datasets.len() == 1 {
        return Ok(datasets[0].clone());
    }

    // Check all have the same number of rows
    let nrows = datasets[0].nrows();
    for ds in datasets.iter().skip(1) {
        if ds.nrows() != nrows {
            return Err(MungeError::RowCountMismatch {
                expected: nrows,
                found: ds.nrows(),
            });
        }
    }

    // Get DataFrames
    let dfs: Vec<DataFrame> = datasets.iter().map(|ds| ds.df().clone()).collect();

    // Use horizontal concat
    let result = polars::functions::concat_df_horizontal(&dfs, true)
        .map_err(|e| MungeError::JoinError(format!("hconcat failed: {}", e)))?;

    Ok(Dataset::new(result))
}

/// Cross join: Cartesian product of two datasets.
///
/// Every row from left is paired with every row from right.
/// Result has left.nrows() * right.nrows() rows.
///
/// # Arguments
///
/// * `left` - Left dataset
/// * `right` - Right dataset
/// * `suffix` - Suffix for duplicate column names from right dataset
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::cross_join;
///
/// // Create all combinations of products and stores
/// let combos = cross_join(&products, &stores, Some("_store"))?;
/// ```
pub fn cross_join(left: &Dataset, right: &Dataset, suffix: Option<&str>) -> MungeResult<Dataset> {
    cross_join_with(left, right, suffix, &JoinOptions::default())
}

/// [`cross_join`] with an optional ceiling on the number of result rows.
///
/// The result has `left.nrows() * right.nrows()` rows; a product that
/// overflows is an error, and one above `opts.max_output_rows` is refused
/// with [`MungeError::JoinError`] naming both numbers.
pub fn cross_join_with(
    left: &Dataset,
    right: &Dataset,
    suffix: Option<&str>,
    opts: &JoinOptions,
) -> MungeResult<Dataset> {
    use polars_ops::frame::join::MaintainOrderJoin;

    let rows = (left.nrows() as u64)
        .checked_mul(right.nrows() as u64)
        .ok_or_else(|| {
            MungeError::JoinError(format!(
                "a cross join of {} and {} rows is too large",
                left.nrows(),
                right.nrows()
            ))
        })?;
    if let Some(max) = opts.max_output_rows
        && rows > max
    {
        return Err(MungeError::JoinError(format!(
            "the cross join would return {rows} rows, more than the allowed {max}"
        )));
    }

    let suffix_str: PlSmallStr = suffix.unwrap_or("_right").into();

    let result = left
        .df()
        .cross_join(right.df(), Some(suffix_str), None, MaintainOrderJoin::None)
        .map_err(|e| MungeError::JoinError(format!("Cross join failed: {}", e)))?;

    Ok(Dataset::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    /// The error text of a call that must fail (`Dataset` is not `Debug`).
    fn err_text(r: MungeResult<Dataset>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn estimated_rows_equal_actual_join_heights() {
        let l = Dataset::new(
            df!("k" => [Some(1i64), Some(1), Some(2), Some(3), None],
                "lv" => [0.5, 1.7, 2.2, 3.9, 4.1])
            .unwrap(),
        );
        let r = Dataset::new(
            df!("k" => [Some(1i64), Some(1), Some(1), Some(2), Some(4), None],
                "rv" => [1.1, 2.3, 3.2, 4.4, 5.6, 6.5])
            .unwrap(),
        );
        for (jt, want) in [
            (JoinType::Inner, 7u64),
            (JoinType::Left, 9),
            (JoinType::Right, 9),
            (JoinType::Full, 11),
            (JoinType::Semi, 3),
            (JoinType::Anti, 2),
        ] {
            assert_eq!(
                estimate_join_rows(&l, &r, &["k"], None, jt).unwrap(),
                want,
                "{jt:?}"
            );
            assert_eq!(
                join(&l, &r, &["k"], None, jt, None).unwrap().nrows() as u64,
                want,
                "{jt:?}"
            );
        }
        let l2 = Dataset::new(df!("a" => [1i64, 1, 1, 2], "b" => ["x", "x", "y", "x"]).unwrap());
        let r2 =
            Dataset::new(df!("c" => [1i64, 1, 2, 2, 3], "d" => ["x", "x", "x", "x", "z"]).unwrap());
        for (jt, want) in [
            (JoinType::Inner, 6u64),
            (JoinType::Left, 7),
            (JoinType::Right, 7),
            (JoinType::Full, 8),
            (JoinType::Semi, 3),
            (JoinType::Anti, 1),
        ] {
            let right_on: &[&str] = &["c", "d"];
            assert_eq!(
                estimate_join_rows(&l2, &r2, &["a", "b"], Some(right_on), jt).unwrap(),
                want,
                "{jt:?}"
            );
            assert_eq!(
                join(&l2, &r2, &["a", "b"], Some(right_on), jt, None)
                    .unwrap()
                    .nrows() as u64,
                want
            );
        }
        assert!(matches!(
            estimate_join_rows(&l, &r, &["nope"], None, JoinType::Inner),
            Err(MungeError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn estimate_join_rows_runs_inside_a_tokio_runtime() {
        // MCP tool calls run inside a tokio runtime. A lazy group-by on at
        // least 1000 rows (15 with debug assertions) with few distinct keys
        // runs on polars' streaming engine, which blocks on a runtime of its
        // own and panics there.
        let k: Vec<i64> = (0..2_000).map(|i| i % 10).collect();
        let v: Vec<f64> = (0..2_000).map(|i| (i as f64 * 0.37).sin()).collect();
        let d = Dataset::new(df!("k" => k, "v" => v).unwrap());
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let rows = rt.block_on(async {
            (
                estimate_join_rows(&d, &d, &["k"], None, JoinType::Inner),
                estimate_join_rows(&d, &d, &["k"], None, JoinType::Full),
            )
        });
        assert_eq!(rows.0.unwrap(), 400_000); // 10 keys x 200 x 200
        assert_eq!(rows.1.unwrap(), 400_000);
    }

    #[test]
    fn join_with_refuses_output_above_limit() {
        let v: Vec<f64> = (0..50).map(|i| i as f64 * 0.31 + 0.05).collect();
        let d = Dataset::new(df!("k" => vec![7i64; 50], "v" => v).unwrap());
        let cap = |n| JoinOptions {
            max_output_rows: Some(n),
        };
        let e = err_text(join_with(
            &d,
            &d,
            &["k"],
            None,
            JoinType::Inner,
            None,
            &cap(2_499),
        ));
        assert!(e.contains("2500") && e.contains("2499"), "{e}");
        assert_eq!(
            join_with(&d, &d, &["k"], None, JoinType::Inner, None, &cap(2_500))
                .unwrap()
                .nrows(),
            2_500
        );
        assert_eq!(
            join(&d, &d, &["k"], None, JoinType::Inner, None)
                .unwrap()
                .nrows(),
            2_500
        );
        let s = Dataset::new(df!("p" => [1.5, 2.5, 3.5]).unwrap());
        assert!(err_text(cross_join_with(&s, &s, None, &cap(8))).contains('9'));
        assert_eq!(cross_join_with(&s, &s, None, &cap(9)).unwrap().nrows(), 9);
        assert_eq!(cross_join(&s, &s, None).unwrap().nrows(), 9);
    }

    #[test]
    fn join_refuses_an_empty_key_list() {
        // polars panics when it is asked to join on zero keys, so an empty key
        // list must be refused before it gets there, on every path.
        let l =
            Dataset::new(df!("id" => [1i64, 2, 3, 4], "x" => [0.41, 1.37, 2.62, 3.18]).unwrap());
        let r = Dataset::new(df!("id" => [1i64, 2, 3, 5], "z" => [9.1, 8.4, 6.6, 4.9]).unwrap());
        let none: &[&str] = &[];
        let cap = JoinOptions {
            max_output_rows: Some(1_000),
        };
        let names_side = |e: &str, side: &str| {
            assert!(
                e.contains("at least one key column") && e.contains(&format!("{side} is empty")),
                "{side}: {e}"
            );
        };
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            for right_on in [None, Some(none), Some(&["id"][..])] {
                names_side(&err_text(join(&l, &r, none, right_on, jt, None)), "left_on");
                names_side(
                    &err_text(join_with(&l, &r, none, right_on, jt, None, &cap)),
                    "left_on",
                );
                match estimate_join_rows(&l, &r, none, right_on, jt) {
                    Err(MungeError::JoinError(e)) => names_side(&e, "left_on"),
                    other => panic!("{jt:?}: expected a JoinError, got {other:?}"),
                }
            }
            names_side(
                &err_text(join(&l, &r, &["id"], Some(none), jt, None)),
                "right_on",
            );
            match estimate_join_rows(&l, &r, &["id"], Some(none), jt) {
                Err(MungeError::JoinError(e)) => names_side(&e, "right_on"),
                other => panic!("{jt:?}: expected a JoinError, got {other:?}"),
            }
        }
        // The named wrappers go through the same check.
        names_side(&err_text(left_join(&l, &r, none, None, None)), "left_on");
        names_side(&err_text(right_join(&l, &r, none, None, None)), "left_on");
        names_side(&err_text(inner_join(&l, &r, none, None, None)), "left_on");
        names_side(&err_text(full_join(&l, &r, none, None, None)), "left_on");
        names_side(&err_text(anti_join(&l, &r, none, None)), "left_on");
        names_side(&err_text(semi_join(&l, &r, none, None)), "left_on");
        // One named key still joins.
        assert_eq!(
            join(&l, &r, &["id"], None, JoinType::Inner, None)
                .unwrap()
                .nrows(),
            3
        );
    }

    fn make_orders() -> Dataset {
        let df = df! {
            "order_id" => [1, 2, 3, 4, 5],
            "customer_id" => [101, 102, 101, 103, 104],
            "amount" => [100.0, 200.0, 150.0, 300.0, 50.0],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn make_customers() -> Dataset {
        let df = df! {
            "customer_id" => [101, 102, 103],
            "name" => ["Alice", "Bob", "Charlie"],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_inner_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = inner_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // Only orders with matching customers (order 5 has customer 104 which doesn't exist)
        assert_eq!(result.nrows(), 4);
        assert!(result.df().column("name").is_ok());
        assert!(result.df().column("order_id").is_ok());
    }

    #[test]
    fn test_left_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = left_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All orders kept, even those without matching customers
        assert_eq!(result.nrows(), 5);
        assert!(result.df().column("name").is_ok());

        // Check that customer 104's order has null name
        let name_col = result.df().column("name").unwrap();
        let names: Vec<Option<&str>> = name_col.str().unwrap().into_iter().collect();
        // The last row (customer_id 104) should have null name
        assert!(names.iter().any(|n| n.is_none()));
    }

    #[test]
    fn test_right_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = right_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All customers kept
        // Customer 101 has 2 orders, 102 has 1, 103 has 1 = 4 rows
        assert_eq!(result.nrows(), 4);
    }

    #[test]
    fn test_full_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = full_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All orders + all customers (5 orders, but customer 101 appears twice)
        // Orders: 1(101), 2(102), 3(101), 4(103), 5(104)
        // Full join keeps all 5 orders
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_anti_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = anti_join(&orders, &customers, &["customer_id"], None).unwrap();

        // Only order 5 (customer 104) has no matching customer
        assert_eq!(result.nrows(), 1);

        let order_ids: Vec<i32> = result
            .df()
            .column("order_id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(order_ids, vec![5]);
    }

    #[test]
    fn test_semi_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = semi_join(&orders, &customers, &["customer_id"], None).unwrap();

        // Orders with matching customers, but no customer columns added
        assert_eq!(result.nrows(), 4);
        assert!(result.df().column("name").is_err()); // name column not added
        assert!(result.df().column("order_id").is_ok());
    }

    #[test]
    fn test_join_with_different_column_names() {
        let df1 = df! {
            "id" => [1, 2, 3],
            "value" => [10, 20, 30],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "ref_id" => [1, 2, 4],
            "label" => ["A", "B", "D"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = left_join(&ds1, &ds2, &["id"], Some(&["ref_id"]), None).unwrap();

        assert_eq!(result.nrows(), 3);
        assert!(result.df().column("label").is_ok());
    }

    #[test]
    fn test_concat() {
        let df1 = df! {
            "x" => [1, 2],
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "x" => [3, 4],
            "y" => ["c", "d"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = concat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_concat_with_missing_columns() {
        let df1 = df! {
            "x" => [1, 2],
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "x" => [3, 4],
            "z" => [10.0, 20.0],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        // Diagonal concat fills missing columns with nulls
        let result = concat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 3); // x, y, z
    }

    #[test]
    fn test_hconcat() {
        let df1 = df! {
            "x" => [1, 2, 3],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "y" => ["a", "b", "c"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = hconcat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert!(result.df().column("x").is_ok());
        assert!(result.df().column("y").is_ok());
    }

    #[test]
    fn test_hconcat_row_mismatch() {
        let df1 = df! {
            "x" => [1, 2, 3],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = hconcat(&[&ds1, &ds2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_join() {
        let df1 = df! {
            "product" => ["A", "B"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "store" => [1, 2, 3],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = cross_join(&ds1, &ds2, None).unwrap();

        // 2 products x 3 stores = 6 rows
        assert_eq!(result.nrows(), 6);
        assert!(result.df().column("product").is_ok());
        assert!(result.df().column("store").is_ok());
    }

    #[test]
    fn test_join_column_not_found() {
        let orders = make_orders();
        let customers = make_customers();

        let result = inner_join(&orders, &customers, &["nonexistent"], None, None);
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_join_with_suffix() {
        let df1 = df! {
            "id" => [1, 2],
            "value" => [10, 20],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "id" => [1, 2],
            "value" => [100, 200],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = inner_join(&ds1, &ds2, &["id"], None, Some("_other")).unwrap();

        assert!(result.df().column("value").is_ok());
        assert!(result.df().column("value_other").is_ok());
    }
}
