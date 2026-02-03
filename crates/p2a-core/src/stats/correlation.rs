//! Correlation analysis for datasets.

use crate::data::Dataset;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Check if a DataType is numeric.
fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Result of a correlation matrix computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

impl CorrelationMatrix {
    /// Format the correlation matrix as a readable string.
    pub fn to_string_table(&self) -> String {
        let mut result = String::new();

        // Header row
        result.push_str(&format!("{:>12}", ""));
        for col in &self.columns {
            result.push_str(&format!("{:>12}", truncate(col, 10)));
        }
        result.push('\n');

        // Data rows
        for (i, row) in self.matrix.iter().enumerate() {
            result.push_str(&format!("{:>12}", truncate(&self.columns[i], 10)));
            for val in row {
                result.push_str(&format!("{:>12.4}", val));
            }
            result.push('\n');
        }

        result
    }
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

/// Compute the correlation matrix for numeric columns in a dataset.
pub fn correlation_matrix(dataset: &Dataset) -> PolarsResult<CorrelationMatrix> {
    let df = dataset.df();

    // Select only numeric columns
    let numeric_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col| is_numeric_dtype(col.dtype()))
        .map(|col| col.name().to_string())
        .collect();

    if numeric_cols.is_empty() {
        return Ok(CorrelationMatrix {
            columns: vec![],
            matrix: vec![],
        });
    }

    let numeric_df = df.select(numeric_cols.iter().map(|s| s.as_str()))?;
    let n = numeric_cols.len();

    // Cast all columns to f64 for correlation computation
    let float_cols: Vec<Series> = numeric_df
        .get_columns()
        .iter()
        .map(|col| col.as_materialized_series().cast(&DataType::Float64))
        .collect::<PolarsResult<Vec<_>>>()?;

    // Compute correlation matrix
    let mut matrix = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0;
            } else if j > i {
                let corr = pearson_correlation(&float_cols[i], &float_cols[j])?;
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
    }

    Ok(CorrelationMatrix {
        columns: numeric_cols,
        matrix,
    })
}

/// Compute Pearson correlation between two series.
fn pearson_correlation(x: &Series, y: &Series) -> PolarsResult<f64> {
    let x_f64 = x.f64()?;
    let y_f64 = y.f64()?;

    let n = x_f64.len();
    if n == 0 {
        return Ok(f64::NAN);
    }

    let x_mean = x_f64.mean().unwrap_or(f64::NAN);
    let y_mean = y_f64.mean().unwrap_or(f64::NAN);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut valid_count = 0;

    for (x_opt, y_opt) in x_f64.iter().zip(y_f64.iter()) {
        if let (Some(x_val), Some(y_val)) = (x_opt, y_opt) {
            let dx = x_val - x_mean;
            let dy = y_val - y_mean;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
            valid_count += 1;
        }
    }

    if valid_count == 0 || sum_x2 == 0.0 || sum_y2 == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(sum_xy / (sum_x2.sqrt() * sum_y2.sqrt()))
}
