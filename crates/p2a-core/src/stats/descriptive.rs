//! Descriptive statistics for datasets.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use crate::data::Dataset;

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

/// Descriptive statistics for an entire dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub nrows: usize,
    pub ncols: usize,
    pub columns: Vec<ColumnStats>,
}

/// Descriptive statistics for a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub name: String,
    pub dtype: String,
    pub count: usize,
    pub null_count: usize,
    pub null_pct: f64,
    /// Statistics for numeric columns
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric: Option<NumericStats>,
    /// Statistics for string columns
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categorical: Option<CategoricalStats>,
}

/// Statistics specific to numeric columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
}

/// Statistics specific to categorical/string columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalStats {
    pub unique_count: usize,
}

impl DescriptiveStats {
    /// Compute descriptive statistics for a dataset.
    pub fn compute(dataset: &Dataset) -> PolarsResult<Self> {
        let df = dataset.df();
        let columns = df
            .get_columns()
            .iter()
            .map(|col| Self::compute_column_stats(col))
            .collect::<PolarsResult<Vec<_>>>()?;

        Ok(Self {
            nrows: dataset.nrows(),
            ncols: dataset.ncols(),
            columns,
        })
    }

    /// Compute descriptive statistics for a single column.
    fn compute_column_stats(col: &Column) -> PolarsResult<ColumnStats> {
        let name = col.name().to_string();
        let dtype = col.dtype().to_string();
        let count = col.len();
        let null_count = col.null_count();
        let null_pct = if count > 0 {
            (null_count as f64 / count as f64) * 100.0
        } else {
            0.0
        };

        let numeric = if is_numeric_dtype(col.dtype()) {
            Some(Self::compute_numeric_stats(col)?)
        } else {
            None
        };

        let categorical = if matches!(col.dtype(), DataType::String | DataType::Categorical(_, _)) {
            Some(Self::compute_categorical_stats(col)?)
        } else {
            None
        };

        Ok(ColumnStats {
            name,
            dtype,
            count,
            null_count,
            null_pct,
            numeric,
            categorical,
        })
    }

    /// Compute numeric statistics for a column.
    fn compute_numeric_stats(col: &Column) -> PolarsResult<NumericStats> {
        let series = col.as_materialized_series();
        let float_series = series.cast(&DataType::Float64)?;

        let mean = float_series.mean().unwrap_or(f64::NAN);
        let std = float_series.std(1).unwrap_or(f64::NAN);

        // Compute min/max
        let min = float_series
            .min::<f64>()
            .ok()
            .flatten()
            .unwrap_or(f64::NAN);
        let max = float_series
            .max::<f64>()
            .ok()
            .flatten()
            .unwrap_or(f64::NAN);

        // For quantiles, use a simpler approach - compute from sorted values
        let (q25, median, q75) = compute_quantiles(&float_series);

        Ok(NumericStats {
            mean,
            std,
            min,
            q25,
            median,
            q75,
            max,
        })
    }

    /// Compute categorical statistics for a column.
    fn compute_categorical_stats(col: &Column) -> PolarsResult<CategoricalStats> {
        let series = col.as_materialized_series();
        let unique_count = series.n_unique()?;

        Ok(CategoricalStats {
            unique_count,
        })
    }
}

/// Compute quantiles from a float series.
fn compute_quantiles(series: &Series) -> (f64, f64, f64) {
    let f64_series = match series.f64() {
        Ok(s) => s,
        Err(_) => return (f64::NAN, f64::NAN, f64::NAN),
    };

    // Collect non-null values
    let mut values: Vec<f64> = f64_series.into_iter().filter_map(|v| v).collect();

    if values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len();

    let q25_idx = (n as f64 * 0.25).floor() as usize;
    let q50_idx = (n as f64 * 0.50).floor() as usize;
    let q75_idx = (n as f64 * 0.75).floor() as usize;

    let q25 = values.get(q25_idx.min(n - 1)).copied().unwrap_or(f64::NAN);
    let median = values.get(q50_idx.min(n - 1)).copied().unwrap_or(f64::NAN);
    let q75 = values.get(q75_idx.min(n - 1)).copied().unwrap_or(f64::NAN);

    (q25, median, q75)
}

impl std::fmt::Display for DescriptiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset: {} rows x {} columns\n", self.nrows, self.ncols)?;

        for col in &self.columns {
            writeln!(f, "Column: {} ({})", col.name, col.dtype)?;
            writeln!(f, "  Count: {}, Nulls: {} ({:.1}%)", col.count, col.null_count, col.null_pct)?;

            if let Some(num) = &col.numeric {
                writeln!(f, "  Mean: {:.4}, Std: {:.4}", num.mean, num.std)?;
                writeln!(f, "  Min: {:.4}, 25%: {:.4}, Median: {:.4}, 75%: {:.4}, Max: {:.4}",
                    num.min, num.q25, num.median, num.q75, num.max)?;
            }

            if let Some(cat) = &col.categorical {
                writeln!(f, "  Unique: {}", cat.unique_count)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
