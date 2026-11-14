//! Descriptive statistics for datasets.

use crate::data::Dataset;
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};
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
    /// Present values: cells that are neither null nor (in a numeric column) NaN.
    pub count: usize,
    /// Missing cells: nulls, and NaN cells of a numeric column.
    pub null_count: usize,
    /// Missing cells as a percentage of the rows.
    pub null_pct: f64,
    /// Statistics for numeric columns
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric: Option<NumericStats>,
    /// Statistics for string columns
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categorical: Option<CategoricalStats>,
}

/// Statistics specific to numeric columns, over the present values: mean, sd (n - 1), min, max,
/// and the quartiles and median of R's type 7 (`quantile()`, `summary()`). NaN when no value is
/// present.
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
            .map(Self::compute_column_stats)
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
        let rows = col.len();

        let (numeric, missing) = if is_numeric_dtype(col.dtype()) {
            let (stats, missing) = Self::compute_numeric_stats(col)?;
            (Some(stats), missing)
        } else {
            (None, col.null_count())
        };
        let null_pct = if rows > 0 {
            (missing as f64 / rows as f64) * 100.0
        } else {
            0.0
        };

        let categorical = if matches!(col.dtype(), DataType::String | DataType::Categorical(_, _)) {
            Some(Self::compute_categorical_stats(col)?)
        } else {
            None
        };

        Ok(ColumnStats {
            name,
            dtype,
            count: rows - missing,
            null_count: missing,
            null_pct,
            numeric,
            categorical,
        })
    }

    /// Numeric statistics over the present values of a column, and the number of missing cells
    /// (null or NaN).
    fn compute_numeric_stats(col: &Column) -> PolarsResult<(NumericStats, usize)> {
        let cast = col.as_materialized_series().cast(&DataType::Float64)?;
        let ca = cast.f64()?;
        let mut values: Vec<f64> = resource::try_with_capacity(ca.len(), "describe column")
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        values.extend(ca.iter().flatten().filter(|v| !v.is_nan()));
        let missing = ca.len() - values.len();
        values.sort_unstable_by(f64::total_cmp);

        let n = values.len();
        let nan = f64::NAN;
        let stats = if n == 0 {
            NumericStats {
                mean: nan,
                std: nan,
                min: nan,
                q25: nan,
                median: nan,
                q75: nan,
                max: nan,
            }
        } else {
            let mean = values.iter().sum::<f64>() / n as f64;
            let std = if n > 1 {
                let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
                (ss / (n - 1) as f64).sqrt()
            } else {
                nan
            };
            let q = |p| quantile_sorted(&values, p, QuantileType::T7);
            NumericStats {
                mean,
                std,
                min: values[0],
                q25: q(0.25),
                median: q(0.5),
                q75: q(0.75),
                max: values[n - 1],
            }
        };
        Ok((stats, missing))
    }

    /// Compute categorical statistics for a column.
    fn compute_categorical_stats(col: &Column) -> PolarsResult<CategoricalStats> {
        let series = col.as_materialized_series();
        let unique_count = series.n_unique()?;

        Ok(CategoricalStats { unique_count })
    }
}

impl std::fmt::Display for DescriptiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset: {} rows x {} columns\n", self.nrows, self.ncols)?;

        for col in &self.columns {
            writeln!(f, "Column: {} ({})", col.name, col.dtype)?;
            writeln!(
                f,
                "  Count: {}, Nulls: {} ({:.1}%)",
                col.count, col.null_count, col.null_pct
            )?;

            if let Some(num) = &col.numeric {
                writeln!(f, "  Mean: {:.4}, Std: {:.4}", num.mean, num.std)?;
                writeln!(
                    f,
                    "  Min: {:.4}, 25%: {:.4}, Median: {:.4}, 75%: {:.4}, Max: {:.4}",
                    num.min, num.q25, num.median, num.q75, num.max
                )?;
            }

            if let Some(cat) = &col.categorical {
                writeln!(f, "  Unique: {}", cat.unique_count)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_describe_quantiles_skip_nan_cells() {
        // X-6: a NaN cell used to count towards n and enter a sort with no total order, so the
        // quantile indices fell on other elements (or on the NaN itself).
        let df = df!("x" => [3.0, f64::NAN, 1.0, 2.0, 5.0]).unwrap();
        let stats = DescriptiveStats::compute(&Dataset::new(df)).unwrap();
        let col = &stats.columns[0];
        let num = col.numeric.as_ref().unwrap();
        // R's type 7 on the four present values [1, 2, 3, 5]; the NaN cell is missing
        assert_eq!((num.q25, num.median, num.q75), (1.75, 2.5, 3.5));
        assert_eq!(num.mean, 2.75);
        assert_eq!((col.count, col.null_count), (4, 1));
        assert_eq!(col.null_pct, 20.0);
    }

    /// count, mean, sd, min, type-7 quartiles and median, max over the present values of a
    /// double column, one with a missing cell and an Int64 column with two (case
    /// stat_describe_t4). Fails before: floor(n p) quartiles, count including nulls.
    #[test]
    fn test_validate_describe_quantiles_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_describe_t4");
        let stats = DescriptiveStats::compute(&ds).unwrap();
        let mut check = Check::new(&case);
        for name in ["e", "o", "ni"] {
            let col = stats.columns.iter().find(|c| c.name == name).unwrap();
            let num = col.numeric.as_ref().unwrap();
            let n = col.count;
            assert_eq!(col.null_count, 12 - n);
            let got = [
                n as f64, num.mean, num.std, num.min, num.q25, num.median, num.q75, num.max,
            ];
            // every entry at the coefficient tolerance (sd too: tighter than Tol::se)
            check.vector(name, &got, Tol::coef(n));
        }
        let g = stats.columns.iter().find(|c| c.name == "g").unwrap();
        assert!(g.numeric.is_none() && g.categorical.is_some());
        let four = df!("v" => [1.0, 2.0, 3.0, 4.0]).unwrap();
        let s4 = DescriptiveStats::compute(&Dataset::new(four)).unwrap();
        let num = s4.columns[0].numeric.as_ref().unwrap();
        check
            .vector("four", &[num.q25, num.median, num.q75], Tol::coef(4))
            .finish();
    }
}
