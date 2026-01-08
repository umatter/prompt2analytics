//! Design matrix construction for regression analysis.
//!
//! Provides utilities for building design matrices from datasets,
//! with support for intercept terms and column selection.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use thiserror::Error;

/// Error type for design matrix construction.
#[derive(Debug, Error)]
pub enum DesignError {
    #[error("Column '{0}' not found in dataset")]
    ColumnNotFound(String),

    #[error("Column '{0}' is not numeric (contains non-float values)")]
    NonNumericColumn(String),

    #[error("Column '{0}' contains null values at indices: {1:?}")]
    NullValues(String, Vec<usize>),

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

/// Extract f64 values from a Polars Series.
fn extract_f64_values(series: &Column, col_name: &str) -> Result<Vec<f64>, DesignError> {
    // Try to cast to Float64
    let float_series = series
        .cast(&DataType::Float64)
        .map_err(|_| DesignError::NonNumericColumn(col_name.to_string()))?;

    let ca = float_series
        .f64()
        .map_err(|_| DesignError::NonNumericColumn(col_name.to_string()))?;

    // Check for null values
    let null_indices: Vec<usize> = ca
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if v.is_none() { Some(i) } else { None })
        .collect();

    if !null_indices.is_empty() {
        return Err(DesignError::NullValues(col_name.to_string(), null_indices));
    }

    // Extract values (we've already checked for nulls)
    let values: Vec<f64> = ca.iter().map(|v| v.unwrap()).collect();

    Ok(values)
}

/// Extract unique groups/entities from a column.
/// Returns a mapping from group ID to row indices.
pub fn extract_groups(
    df: &DataFrame,
    group_col: &str,
) -> Result<std::collections::HashMap<String, Vec<usize>>, DesignError> {
    let series = df
        .column(group_col)
        .map_err(|_| DesignError::ColumnNotFound(group_col.to_string()))?;

    let mut groups: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();

    // Convert to string representation for grouping
    let str_values: Vec<String> = (0..series.len())
        .map(|i| format!("{:?}", series.get(i).unwrap()))
        .collect();

    for (i, key) in str_values.iter().enumerate() {
        groups.entry(key.clone()).or_default().push(i);
    }

    Ok(groups)
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
    use polars::prelude::*;

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
}
