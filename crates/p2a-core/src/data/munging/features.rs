//! Feature engineering operations for data preparation.
//!
//! This module provides operations for creating new features from existing data,
//! including lag/lead operations, rolling statistics, scaling, and encoding.
//!
//! # Operations
//!
//! - **Lag/Lead**: Create shifted versions of columns
//! - **Diff/PctChange**: Calculate differences and percentage changes
//! - **Standardize/Normalize**: Scale features
//! - **Bin**: Discretize continuous variables
//! - **OneHot**: Create dummy variables from categorical columns
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Create lagged variable
//! let with_lag = lag(&data, "price", 1, None)?;
//!
//! // Standardize features
//! let standardized = standardize(&data, &["x1", "x2"])?;
//! ```

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;
use polars_ops::prelude::RankOptions;

/// Create a lagged version of a column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Name of the column to lag
/// * `periods` - Number of periods to lag (positive = look back)
/// * `group_by` - Optional column(s) to group by for panel data
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::lag;
///
/// // Simple lag
/// let with_lag = lag(&data, "price", 1, None)?;
///
/// // Panel data with grouping
/// let panel_lag = lag(&data, "price", 1, Some(&["firm_id"]))?;
/// ```
pub fn lag(
    dataset: &Dataset,
    column: &str,
    periods: usize,
    group_by: Option<&[&str]>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let lagged_name = format!("{}_lag{}", column, periods);

    let result = if let Some(groups) = group_by {
        // Validate group columns
        for g in groups {
            if df.column(*g).is_err() {
                return Err(MungeError::ColumnNotFound(g.to_string()));
            }
        }

        // Group by and apply shift
        let group_exprs: Vec<Expr> = groups.iter().map(|g| col(*g)).collect();
        df.clone()
            .lazy()
            .with_column(
                col(column)
                    .shift(lit(periods as i64))
                    .over(group_exprs)
                    .alias(&lagged_name)
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    } else {
        // Simple shift without grouping
        df.clone()
            .lazy()
            .with_column(
                col(column)
                    .shift(lit(periods as i64))
                    .alias(&lagged_name)
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    };

    Ok(Dataset::new(result))
}

/// Create a lead (forward-shifted) version of a column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Name of the column to lead
/// * `periods` - Number of periods to lead (positive = look forward)
/// * `group_by` - Optional column(s) to group by for panel data
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::lead;
///
/// let with_lead = lead(&data, "price", 1, None)?;
/// ```
pub fn lead(
    dataset: &Dataset,
    column: &str,
    periods: usize,
    group_by: Option<&[&str]>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let lead_name = format!("{}_lead{}", column, periods);

    let result = if let Some(groups) = group_by {
        // Validate group columns
        for g in groups {
            if df.column(*g).is_err() {
                return Err(MungeError::ColumnNotFound(g.to_string()));
            }
        }

        // Group by and apply negative shift (lead)
        let group_exprs: Vec<Expr> = groups.iter().map(|g| col(*g)).collect();
        df.clone()
            .lazy()
            .with_column(
                col(column)
                    .shift(lit(-(periods as i64)))
                    .over(group_exprs)
                    .alias(&lead_name)
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    } else {
        // Simple negative shift without grouping
        df.clone()
            .lazy()
            .with_column(
                col(column)
                    .shift(lit(-(periods as i64)))
                    .alias(&lead_name)
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    };

    Ok(Dataset::new(result))
}

/// Calculate the difference between consecutive values.
///
/// Computes: value[i] - value[i - periods]
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Name of the column to difference
/// * `periods` - Number of periods for the difference
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::diff;
///
/// let with_diff = diff(&data, "price", 1)?;
/// ```
pub fn diff(
    dataset: &Dataset,
    column: &str,
    periods: usize,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let diff_name = format!("{}_diff{}", column, periods);

    // Calculate difference as: current - lagged
    let result = df.clone()
        .lazy()
        .with_column(
            (col(column) - col(column).shift(lit(periods as i64)))
                .alias(&diff_name)
        )
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Calculate percentage change between consecutive values.
///
/// Computes: (value[i] - value[i - periods]) / value[i - periods]
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Name of the column to calculate percentage change
/// * `periods` - Number of periods for the percentage change
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::pct_change;
///
/// let with_pct = pct_change(&data, "price", 1)?;
/// ```
pub fn pct_change(
    dataset: &Dataset,
    column: &str,
    periods: usize,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let pct_name = format!("{}_pct_change{}", column, periods);

    // Calculate (current - lagged) / lagged
    let lagged = col(column).shift(lit(periods as i64));
    let result = df.clone()
        .lazy()
        .with_column(
            ((col(column) - lagged.clone()) / lagged)
                .alias(&pct_name)
        )
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Standardize columns to have mean 0 and standard deviation 1.
///
/// Computes: (value - mean) / std
///
/// # Arguments
///
/// * `dataset` - Dataset containing the columns
/// * `columns` - Columns to standardize
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::standardize;
///
/// let standardized = standardize(&data, &["x1", "x2"])?;
/// ```
pub fn standardize(dataset: &Dataset, columns: &[&str]) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns exist
    for col_name in columns {
        if df.column(*col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }

    // Build expressions for standardization
    let mut exprs: Vec<Expr> = Vec::new();
    for col_name in columns {
        let standardized_name = format!("{}_standardized", col_name);
        let c = col(*col_name);
        let mean = c.clone().mean();
        let std = c.clone().std(1);
        exprs.push(((c - mean) / std).alias(&standardized_name));
    }

    let result = df.clone()
        .lazy()
        .with_columns(exprs)
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Normalize columns to range [0, 1].
///
/// Computes: (value - min) / (max - min)
///
/// # Arguments
///
/// * `dataset` - Dataset containing the columns
/// * `columns` - Columns to normalize
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::normalize;
///
/// let normalized = normalize(&data, &["x1", "x2"])?;
/// ```
pub fn normalize(dataset: &Dataset, columns: &[&str]) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns exist
    for col_name in columns {
        if df.column(*col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }

    // Build expressions for normalization
    let mut exprs: Vec<Expr> = Vec::new();
    for col_name in columns {
        let normalized_name = format!("{}_normalized", col_name);
        let c = col(*col_name);
        let min = c.clone().min();
        let max = c.clone().max();
        exprs.push(((c - min.clone()) / (max - min)).alias(&normalized_name));
    }

    let result = df.clone()
        .lazy()
        .with_columns(exprs)
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Binning strategy for discretization.
#[derive(Debug, Clone)]
pub enum BinStrategy {
    /// Equal width bins with specified number of bins
    EqualWidth(usize),
    /// Equal frequency bins (quantiles) with specified number of bins
    Quantile(usize),
    /// Custom bin edges
    Custom(Vec<f64>),
}

/// Discretize a continuous column into bins.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Column to bin
/// * `strategy` - Binning strategy
/// * `labels` - Optional labels for the bins
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{bin, BinStrategy};
///
/// let binned = bin(&data, "age", BinStrategy::EqualWidth(5), None)?;
/// ```
pub fn bin(
    dataset: &Dataset,
    column: &str,
    strategy: BinStrategy,
    labels: Option<&[&str]>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    let col_data = df.column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;

    let float_col = col_data.cast(&DataType::Float64)
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;
    let float_ca = float_col.f64()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    // Calculate bin edges based on strategy
    let edges = match &strategy {
        BinStrategy::EqualWidth(n) => {
            let min = float_ca.min().unwrap_or(0.0);
            let max = float_ca.max().unwrap_or(1.0);
            let step = (max - min) / (*n as f64);
            (0..=*n).map(|i| min + (i as f64) * step).collect::<Vec<_>>()
        }
        BinStrategy::Quantile(n) => {
            let mut edges = vec![f64::NEG_INFINITY];
            for i in 1..*n {
                let q = (i as f64) / (*n as f64);
                let quantile = float_ca
                    .quantile(q, QuantileMethod::Linear)
                    .map_err(|e| MungeError::FeatureError(e.to_string()))?
                    .unwrap_or(0.0);
                edges.push(quantile);
            }
            edges.push(f64::INFINITY);
            edges
        }
        BinStrategy::Custom(edges) => edges.clone(),
    };

    // Validate labels match number of bins
    let n_bins = edges.len().saturating_sub(1);
    if let Some(lbls) = labels {
        if lbls.len() != n_bins {
            return Err(MungeError::FeatureError(format!(
                "Expected {} labels for {} bins, got {}",
                n_bins, n_bins, lbls.len()
            )));
        }
    }

    // Create bin assignments
    let bin_name = format!("{}_binned", column);
    let mut bin_values: Vec<String> = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        let val = float_ca.get(i);
        let bin_idx = match val {
            Some(v) => {
                // Find bin index
                let mut idx = 0;
                for j in 1..edges.len() {
                    if v > edges[j - 1] && v <= edges[j] {
                        idx = j - 1;
                        break;
                    }
                }
                idx
            }
            None => 0, // Assign nulls to first bin
        };

        let label = if let Some(lbls) = labels {
            lbls.get(bin_idx).copied().unwrap_or("unknown").to_string()
        } else {
            // Use auto-generated labels
            format!("bin_{}", bin_idx)
        };
        bin_values.push(label);
    }

    let bin_col = Column::new(bin_name.into(), bin_values);
    let mut result = df.clone();
    result.with_column(bin_col)
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Create one-hot encoded dummy variables from a categorical column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Categorical column to encode
/// * `drop_first` - Whether to drop the first category (to avoid multicollinearity)
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::one_hot_encode;
///
/// let encoded = one_hot_encode(&data, "region", true)?;
/// ```
pub fn one_hot_encode(
    dataset: &Dataset,
    column: &str,
    drop_first: bool,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    let col_data = df.column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;

    // Get unique values as strings
    let unique = col_data.unique()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    // Convert to string representation
    let unique_str: Vec<String> = (0..unique.len())
        .map(|i| {
            unique.get(i)
                .map(|v| format!("{}", v))
                .unwrap_or_else(|_| "null".to_string())
        })
        .collect();

    let mut result = df.clone();

    // Skip first category if drop_first is true
    let start_idx = if drop_first { 1 } else { 0 };

    for cat in unique_str.iter().skip(start_idx) {
        let dummy_name = format!("{}_{}", column, cat);
        let mut dummy_values: Vec<i32> = Vec::with_capacity(df.height());

        for row_idx in 0..df.height() {
            let val = col_data.get(row_idx)
                .map(|v| format!("{}", v))
                .unwrap_or_else(|_| "null".to_string());
            dummy_values.push(if val == *cat { 1 } else { 0 });
        }

        let dummy_col = Column::new(dummy_name.into(), dummy_values);
        result.with_column(dummy_col)
            .map_err(|e| MungeError::FeatureError(e.to_string()))?;
    }

    Ok(Dataset::new(result))
}

/// Calculate cumulative sum of a column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Column to cumsum
/// * `reverse` - Whether to compute in reverse order
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::cumsum;
///
/// let with_cumsum = cumsum(&data, "value", false)?;
/// ```
pub fn cumsum(
    dataset: &Dataset,
    column: &str,
    reverse: bool,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let cumsum_name = format!("{}_cumsum", column);

    let result = df.clone()
        .lazy()
        .with_column(
            col(column).cum_sum(reverse).alias(&cumsum_name)
        )
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Calculate cumulative product of a column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Column to cumprod
/// * `reverse` - Whether to compute in reverse order
pub fn cumprod(
    dataset: &Dataset,
    column: &str,
    reverse: bool,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let cumprod_name = format!("{}_cumprod", column);

    let result = df.clone()
        .lazy()
        .with_column(
            col(column).cum_prod(reverse).alias(&cumprod_name)
        )
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Rank values in a column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Column to rank
/// * `descending` - Whether to rank in descending order
pub fn rank(
    dataset: &Dataset,
    column: &str,
    descending: bool,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let rank_name = format!("{}_rank", column);

    let opts = RankOptions {
        descending,
        ..Default::default()
    };

    let result = df.clone()
        .lazy()
        .with_column(
            col(column).rank(opts, None).alias(&rank_name)
        )
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    #[test]
    fn test_lag() {
        let df = df! {
            "id" => [1, 1, 1, 2, 2, 2],
            "value" => [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = lag(&ds, "value", 1, None).unwrap();

        assert!(result.df().column("value_lag1").is_ok());
        let lag_col = result.df().column("value_lag1").unwrap();
        // First value should be null
        assert!(lag_col.f64().unwrap().get(0).is_none());
        // Second value should be first original value
        assert_eq!(lag_col.f64().unwrap().get(1), Some(10.0));
    }

    #[test]
    fn test_lag_with_groupby() {
        let df = df! {
            "id" => [1, 1, 1, 2, 2, 2],
            "value" => [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = lag(&ds, "value", 1, Some(&["id"])).unwrap();

        let lag_col = result.df().column("value_lag1").unwrap();
        // First value of each group should be null
        assert!(lag_col.f64().unwrap().get(0).is_none());
        assert!(lag_col.f64().unwrap().get(3).is_none());
    }

    #[test]
    fn test_lead() {
        let df = df! {
            "value" => [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = lead(&ds, "value", 1, None).unwrap();

        assert!(result.df().column("value_lead1").is_ok());
        let lead_col = result.df().column("value_lead1").unwrap();
        // First value should be second original value
        assert_eq!(lead_col.f64().unwrap().get(0), Some(20.0));
        // Last value should be null
        assert!(lead_col.f64().unwrap().get(4).is_none());
    }

    #[test]
    fn test_diff() {
        let df = df! {
            "value" => [10.0, 15.0, 25.0, 40.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = diff(&ds, "value", 1).unwrap();

        let diff_col = result.df().column("value_diff1").unwrap();
        // First value should be null
        assert!(diff_col.f64().unwrap().get(0).is_none());
        // Second value should be 15 - 10 = 5
        assert_eq!(diff_col.f64().unwrap().get(1), Some(5.0));
        // Third value should be 25 - 15 = 10
        assert_eq!(diff_col.f64().unwrap().get(2), Some(10.0));
    }

    #[test]
    fn test_pct_change() {
        let df = df! {
            "value" => [100.0, 110.0, 99.0, 121.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = pct_change(&ds, "value", 1).unwrap();

        let pct_col = result.df().column("value_pct_change1").unwrap();
        // First value should be null
        assert!(pct_col.f64().unwrap().get(0).is_none());
        // Second value should be (110 - 100) / 100 = 0.1
        let val = pct_col.f64().unwrap().get(1).unwrap();
        assert!((val - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_standardize() {
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = standardize(&ds, &["x"]).unwrap();

        let std_col = result.df().column("x_standardized").unwrap();
        let std_ca = std_col.f64().unwrap();

        // Mean should be approximately 0
        let mean: f64 = std_ca.into_no_null_iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let df = df! {
            "x" => [0.0, 25.0, 50.0, 75.0, 100.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = normalize(&ds, &["x"]).unwrap();

        let norm_col = result.df().column("x_normalized").unwrap();
        let norm_ca = norm_col.f64().unwrap();

        // Values should be in [0, 1]
        assert_eq!(norm_ca.get(0), Some(0.0));
        assert_eq!(norm_ca.get(4), Some(1.0));
        assert_eq!(norm_ca.get(2), Some(0.5));
    }

    #[test]
    fn test_bin_equal_width() {
        let df = df! {
            "value" => [5.0, 15.0, 25.0, 35.0, 45.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = bin(&ds, "value", BinStrategy::EqualWidth(5), None).unwrap();

        assert!(result.df().column("value_binned").is_ok());
    }

    #[test]
    fn test_one_hot_encode() {
        let df = df! {
            "category" => ["a", "b", "a", "c", "b"],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = one_hot_encode(&ds, "category", false).unwrap();

        // Should have one column per category
        assert!(result.df().column("category_\"a\"").is_ok() ||
                result.df().column("category_\"b\"").is_ok() ||
                result.df().column("category_\"c\"").is_ok());
    }

    #[test]
    fn test_one_hot_encode_drop_first() {
        let df = df! {
            "category" => ["a", "b", "a", "c", "b"],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = one_hot_encode(&ds, "category", true).unwrap();

        // Should have one fewer column than categories (3 - 1 = 2)
        let n_dummy_cols = result.df().get_columns().iter()
            .filter(|c| c.name().starts_with("category_"))
            .count();
        assert_eq!(n_dummy_cols, 2);
    }

    #[test]
    fn test_cumsum() {
        let df = df! {
            "value" => [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = cumsum(&ds, "value", false).unwrap();

        let cumsum_col = result.df().column("value_cumsum").unwrap();
        let cumsum_ca = cumsum_col.f64().unwrap();

        assert_eq!(cumsum_ca.get(0), Some(1.0));
        assert_eq!(cumsum_ca.get(1), Some(3.0));
        assert_eq!(cumsum_ca.get(2), Some(6.0));
        assert_eq!(cumsum_ca.get(4), Some(15.0));
    }

    #[test]
    fn test_rank() {
        let df = df! {
            "value" => [30.0, 10.0, 20.0, 50.0, 40.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = rank(&ds, "value", false).unwrap();

        assert!(result.df().column("value_rank").is_ok());
    }

    #[test]
    fn test_column_not_found() {
        let df = df! {
            "x" => [1.0, 2.0, 3.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        assert!(matches!(
            lag(&ds, "nonexistent", 1, None),
            Err(MungeError::ColumnNotFound(_))
        ));
    }
}
