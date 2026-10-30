//! Design matrix construction for regression analysis.
//!
//! Provides utilities for building design matrices from datasets,
//! with support for intercept terms and column selection.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use thiserror::Error;

/// Get all column names from a DataFrame as a Vec<String>.
///
/// Useful for providing available columns in error messages.
#[inline]
pub fn get_column_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Error type for design matrix construction.
#[derive(Debug, Error)]
pub enum DesignError {
    #[error("Column '{0}' not found in dataset")]
    ColumnNotFound(String),

    #[error("Column '{0}' is not numeric (contains non-float values)")]
    NonNumericColumn(String),

    /// Missing cells (null, or NaN in a float column); `first_rows` holds the
    /// first 5 0-based row indices at most.
    #[error(
        "Column '{column}' contains {count} missing value(s) (first at 0-based rows {first_rows:?})"
    )]
    NullValues {
        column: String,
        count: usize,
        first_rows: Vec<usize>,
    },

    /// `±Inf` cells; `first_rows` holds the first 5 0-based row indices at most.
    #[error(
        "Column '{column}' contains {count} infinite value(s) (first at 0-based rows {first_rows:?})"
    )]
    NonFinite {
        column: String,
        count: usize,
        first_rows: Vec<usize>,
    },

    #[error("Empty dataset: no rows to process")]
    EmptyDataset,

    #[error("No columns specified for design matrix")]
    NoColumns,

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

/// A design matrix with metadata.
#[derive(Debug, Clone)]
pub struct DesignMatrix {
    /// The matrix data (n_samples x n_features)
    pub data: Array2<f64>,
    /// Column names (including "(Intercept)" if applicable)
    pub column_names: Vec<String>,
    /// Whether an intercept column was added
    pub has_intercept: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features (including intercept if present)
    pub n_features: usize,
}

impl DesignMatrix {
    /// Build a design matrix from a DataFrame.
    ///
    /// # Arguments
    /// * `df` - The source DataFrame
    /// * `columns` - Column names to include
    /// * `intercept` - Whether to add an intercept column (column of 1s)
    ///
    /// # Returns
    /// A DesignMatrix with the specified columns extracted and converted to f64.
    pub fn from_dataframe(
        df: &DataFrame,
        columns: &[&str],
        intercept: bool,
    ) -> Result<Self, DesignError> {
        if df.height() == 0 {
            return Err(DesignError::EmptyDataset);
        }

        if columns.is_empty() && !intercept {
            return Err(DesignError::NoColumns);
        }

        let n_obs = df.height();
        let n_cols = columns.len() + if intercept { 1 } else { 0 };

        let mut data = Array2::zeros((n_obs, n_cols));
        let mut column_names = Vec::with_capacity(n_cols);

        let mut col_idx = 0;

        // Add intercept column if requested
        if intercept {
            for i in 0..n_obs {
                data[[i, col_idx]] = 1.0;
            }
            column_names.push("(Intercept)".to_string());
            col_idx += 1;
        }

        // Extract each requested column
        for &col_name in columns {
            let series = df
                .column(col_name)
                .map_err(|_| DesignError::ColumnNotFound(col_name.to_string()))?;

            let values = extract_f64_values(series, col_name)?;

            for (i, &val) in values.iter().enumerate() {
                data[[i, col_idx]] = val;
            }
            column_names.push(col_name.to_string());
            col_idx += 1;
        }

        Ok(DesignMatrix {
            data,
            column_names,
            has_intercept: intercept,
            n_obs,
            n_features: n_cols,
        })
    }

    /// Extract a single column as an Array1.
    pub fn extract_column(df: &DataFrame, col_name: &str) -> Result<Array1<f64>, DesignError> {
        let series = df
            .column(col_name)
            .map_err(|_| DesignError::ColumnNotFound(col_name.to_string()))?;

        let values = extract_f64_values(series, col_name)?;
        Ok(Array1::from_vec(values))
    }

    /// Get the degrees of freedom (n_obs - n_features).
    pub fn df(&self) -> usize {
        self.n_obs.saturating_sub(self.n_features)
    }

    /// Get a view of the data matrix.
    pub fn view(&self) -> ndarray::ArrayView2<'_, f64> {
        self.data.view()
    }

    /// Get the column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.column_names.iter().position(|n| n == name)
    }

    /// Get data for a specific column by name.
    pub fn column_data(&self, name: &str) -> Option<Array1<f64>> {
        let idx = self.column_index(name)?;
        Some(self.data.column(idx).to_owned())
    }
}

/// The most row indices a [`DesignError::NullValues`] or [`DesignError::NonFinite`]
/// carries, so an error on a large column stays a short message.
const FIRST_ROWS: usize = 5;

/// A count of cells and the first [`FIRST_ROWS`] of their row indices.
#[derive(Default)]
struct Hits {
    count: usize,
    first_rows: Vec<usize>,
}

impl Hits {
    fn add(&mut self, row: usize) {
        self.count += 1;
        if self.first_rows.len() < FIRST_ROWS {
            self.first_rows.push(row);
        }
    }
}

/// Extract f64 values from a column with the shared strict numeric rule
/// (`crate::data::columns`): text, date and other non-numeric columns are
/// `NonNumericColumn`, null and NaN are missing (`NullValues`), and `±Inf` is
/// `NonFinite`. Missing values are reported before infinite ones.
fn extract_f64_values(series: &Column, col_name: &str) -> Result<Vec<f64>, DesignError> {
    let ca = crate::data::columns::float64_strict(series, true)
        .ok_or_else(|| DesignError::NonNumericColumn(col_name.to_string()))?;

    let mut missing = Hits::default();
    let mut infinite = Hits::default();
    let mut values = Vec::with_capacity(ca.len());
    for (i, v) in ca.iter().enumerate() {
        match v {
            Some(x) if x.is_nan() => missing.add(i),
            Some(x) if x.is_infinite() => infinite.add(i),
            Some(x) => values.push(x),
            None => missing.add(i),
        }
    }

    if missing.count > 0 {
        return Err(DesignError::NullValues {
            column: col_name.to_string(),
            count: missing.count,
            first_rows: missing.first_rows,
        });
    }
    if infinite.count > 0 {
        return Err(DesignError::NonFinite {
            column: col_name.to_string(),
            count: infinite.count,
            first_rows: infinite.first_rows,
        });
    }
    Ok(values)
}

/// Extract unique groups/entities from a column.
/// Returns a mapping from group label to row indices.
///
/// The keys are the column's factor labels (`crate::data::columns::Factor`: text
/// as stored, integers in decimal, booleans "FALSE"/"TRUE", dates "YYYY-MM-DD"). A
/// missing label is [`DesignError::NullValues`]; a column that cannot be a factor
/// is a [`DesignError::PolarsError`] naming it.
///
/// `DesignError` has no resource variant: an allocation refused under the
/// [`crate::resource`] ceiling is also returned as [`DesignError::PolarsError`], whose
/// text carries the "Resource limit" message, and `econ_error_from_design` turns any
/// `PolarsError` into `EconError::Internal`. A caller that must report a refusal as
/// `EconError::ResourceLimit` uses `crate::data::columns::factor_options` directly.
pub fn extract_groups(
    df: &DataFrame,
    group_col: &str,
) -> Result<std::collections::HashMap<String, Vec<usize>>, DesignError> {
    use crate::errors::EconError;

    let (codes, levels) =
        crate::data::columns::factor_options(df, group_col).map_err(|e| match e {
            EconError::ColumnNotFound { column, .. } => DesignError::ColumnNotFound(column),
            other => DesignError::PolarsError(PolarsError::ComputeError(other.to_string().into())),
        })?;

    let mut missing = Hits::default();
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); levels.len()];
    for (i, code) in codes.iter().enumerate() {
        match code {
            Some(c) => rows[*c as usize].push(i),
            None => missing.add(i),
        }
    }
    if missing.count > 0 {
        return Err(DesignError::NullValues {
            column: group_col.to_string(),
            count: missing.count,
            first_rows: missing.first_rows,
        });
    }

    Ok(levels.into_iter().zip(rows).collect())
}

/// Demean data within groups (for fixed effects).
/// Returns demeaned X and y arrays.
pub fn demean_within_groups(
    x: &Array2<f64>,
    y: &Array1<f64>,
    groups: &std::collections::HashMap<String, Vec<usize>>,
) -> (Array2<f64>, Array1<f64>) {
    let (_n, k) = x.dim();
    let mut x_demeaned = x.clone();
    let mut y_demeaned = y.clone();

    for indices in groups.values() {
        // Compute group means for X
        let mut x_means = vec![0.0; k];
        for &i in indices {
            for j in 0..k {
                x_means[j] += x[[i, j]];
            }
        }
        let n_group = indices.len() as f64;
        for j in 0..k {
            x_means[j] /= n_group;
        }

        // Compute group mean for y
        let y_mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n_group;

        // Demean
        for &i in indices {
            for j in 0..k {
                x_demeaned[[i, j]] -= x_means[j];
            }
            y_demeaned[i] -= y_mean;
        }
    }

    (x_demeaned, y_demeaned)
}

/// Quasi-demean data for random effects.
/// Uses theta = 1 - sqrt(sigma_e^2 / (T * sigma_u^2 + sigma_e^2))
pub fn quasi_demean_within_groups(
    x: &Array2<f64>,
    y: &Array1<f64>,
    groups: &std::collections::HashMap<String, Vec<usize>>,
    theta: f64,
) -> (Array2<f64>, Array1<f64>) {
    let (_n, k) = x.dim();
    let mut x_transformed = x.clone();
    let mut y_transformed = y.clone();

    for indices in groups.values() {
        // Compute group means for X
        let mut x_means = vec![0.0; k];
        for &i in indices {
            for j in 0..k {
                x_means[j] += x[[i, j]];
            }
        }
        let n_group = indices.len() as f64;
        for j in 0..k {
            x_means[j] /= n_group;
        }

        // Compute group mean for y
        let y_mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n_group;

        // Quasi-demean: x_it - theta * x_bar_i
        for &i in indices {
            for j in 0..k {
                x_transformed[[i, j]] -= theta * x_means[j];
            }
            y_transformed[i] -= theta * y_mean;
        }
    }

    (x_transformed, y_transformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_df() -> DataFrame {
        df! {
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2" => [2.0, 4.0, 6.0, 8.0, 10.0],
            "y" => [3.0, 5.0, 7.0, 9.0, 11.0],
            "group" => ["A", "A", "B", "B", "B"]
        }
        .unwrap()
    }

    #[test]
    fn test_design_matrix_basic() {
        let df = create_test_df();
        let dm = DesignMatrix::from_dataframe(&df, &["x1", "x2"], false).unwrap();

        assert_eq!(dm.n_obs, 5);
        assert_eq!(dm.n_features, 2);
        assert!(!dm.has_intercept);
        assert_eq!(dm.column_names, vec!["x1", "x2"]);
    }

    #[test]
    fn test_design_matrix_with_intercept() {
        let df = create_test_df();
        let dm = DesignMatrix::from_dataframe(&df, &["x1"], true).unwrap();

        assert_eq!(dm.n_features, 2);
        assert!(dm.has_intercept);
        assert_eq!(dm.column_names, vec!["(Intercept)", "x1"]);

        // Check intercept column is all 1s
        for i in 0..dm.n_obs {
            assert_eq!(dm.data[[i, 0]], 1.0);
        }
    }

    #[test]
    fn test_extract_column() {
        let df = create_test_df();
        let y = DesignMatrix::extract_column(&df, "y").unwrap();

        assert_eq!(y.len(), 5);
        assert_eq!(y[0], 3.0);
        assert_eq!(y[4], 11.0);
    }

    #[test]
    fn test_extract_groups() {
        let df = create_test_df();
        let groups = extract_groups(&df, "group").unwrap();

        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_column_not_found() {
        let df = create_test_df();
        let result = DesignMatrix::from_dataframe(&df, &["nonexistent"], false);

        assert!(matches!(result, Err(DesignError::ColumnNotFound(_))));
    }

    /// 100 000 rows, x1 missing in every even row (50 000 nulls), noisy y.
    fn half_null_frame() -> DataFrame {
        let x1: Vec<Option<f64>> = (0..100_000usize)
            .map(|i| (i % 2 == 1).then(|| i as f64 * 0.5 + ((i * 7) % 11) as f64 * 0.1))
            .collect();
        let y: Vec<f64> = (0..100_000usize)
            .map(|i| i as f64 * 0.25 + ((i * 13) % 17) as f64 * 0.05)
            .collect();
        df! { "x1" => x1, "y" => y }.unwrap()
    }

    #[test]
    fn test_reg32_design_nan_is_missing() {
        let df = df! {
            "x1" => [1.0, 2.0, f64::NAN, 4.0, 5.5],
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1],
        }
        .unwrap();
        let expected = "Column 'x1' contains 1 missing value(s) (first at 0-based rows [2])";
        let err =
            DesignMatrix::from_dataframe(&df, &["x1"], true).expect_err("NaN must be reported");
        assert_eq!(err.to_string(), expected);
        let err = DesignMatrix::extract_column(&df, "x1").expect_err("NaN must be reported");
        assert_eq!(err.to_string(), expected);
    }

    #[test]
    fn test_reg32_design_inf_is_refused() {
        let df = df! { "x1" => [1.0, 2.0, 3.0, f64::INFINITY, f64::NEG_INFINITY] }.unwrap();
        let err =
            DesignMatrix::from_dataframe(&df, &["x1"], false).expect_err("Inf must be reported");
        assert_eq!(
            err.to_string(),
            "Column 'x1' contains 2 infinite value(s) (first at 0-based rows [3, 4])"
        );
    }

    #[test]
    fn test_reg32_design_null_message_is_bounded() {
        let df = half_null_frame();
        let msg = DesignMatrix::from_dataframe(&df, &["x1"], true)
            .expect_err("nulls must be reported")
            .to_string();
        assert_eq!(
            msg,
            "Column 'x1' contains 50000 missing value(s) (first at 0-based rows [0, 2, 4, 6, 8])"
        );
        assert!(msg.len() < 120, "{} bytes", msg.len());
    }

    #[test]
    fn test_reg32_harvtest_null_error_is_bounded() {
        let ds = crate::data::Dataset::new(half_null_frame());
        let msg = crate::regression::harvey_collier_test(&ds, "y", &["x1"])
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Design matrix error: Column 'x1' contains 50000 missing value(s) \
             (first at 0-based rows [0, 2, 4, 6, 8])"
        );
    }

    #[test]
    fn test_reg32_design_text_column_is_non_numeric() {
        let df = df! { "s" => ["1.5", "2", "3"], "t" => ["1", "2", "x"] }.unwrap();
        for name in ["s", "t"] {
            match DesignMatrix::from_dataframe(&df, &[name], false) {
                Err(DesignError::NonNumericColumn(c)) => {
                    assert_eq!(c, name);
                    assert_eq!(
                        DesignError::NonNumericColumn(c).to_string(),
                        format!("Column '{name}' is not numeric (contains non-float values)")
                    );
                }
                other => panic!("{name}: expected NonNumericColumn, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_design_extract_groups_uses_factor_labels() {
        let df = create_test_df();
        let groups = extract_groups(&df, "group").unwrap();
        let mut keys: Vec<&String> = groups.keys().collect();
        keys.sort();
        assert_eq!(keys, ["A", "B"]);
        assert_eq!(groups["A"], [0, 1]);
        assert_eq!(groups["B"], [2, 3, 4]);

        let ints = df! { "g" => [10i64, 2, 10] }.unwrap();
        let groups = extract_groups(&ints, "g").unwrap();
        let mut keys: Vec<&String> = groups.keys().collect();
        keys.sort();
        assert_eq!(keys, ["10", "2"]);
        assert_eq!(groups["10"], [0, 2]);

        let missing = df! { "group" => [Some("A"), None] }.unwrap();
        let err = extract_groups(&missing, "group").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Column 'group' contains 1 missing value(s) (first at 0-based rows [1])"
        );
    }
}
