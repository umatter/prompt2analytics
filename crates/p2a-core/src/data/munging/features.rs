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

use super::CHECKPOINT_EVERY;
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
            if df.column(g).is_err() {
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
                    .alias(&lagged_name),
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    } else {
        // Simple shift without grouping
        df.clone()
            .lazy()
            .with_column(col(column).shift(lit(periods as i64)).alias(&lagged_name))
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
            if df.column(g).is_err() {
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
                    .alias(&lead_name),
            )
            .collect()
            .map_err(|e| MungeError::FeatureError(e.to_string()))?
    } else {
        // Simple negative shift without grouping
        df.clone()
            .lazy()
            .with_column(col(column).shift(lit(-(periods as i64))).alias(&lead_name))
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
pub fn diff(dataset: &Dataset, column: &str, periods: usize) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let diff_name = format!("{}_diff{}", column, periods);

    // Calculate difference as: current - lagged
    let result = df
        .clone()
        .lazy()
        .with_column((col(column) - col(column).shift(lit(periods as i64))).alias(&diff_name))
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
pub fn pct_change(dataset: &Dataset, column: &str, periods: usize) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let pct_name = format!("{}_pct_change{}", column, periods);

    // Calculate (current - lagged) / lagged
    let lagged = col(column).shift(lit(periods as i64));
    let result = df
        .clone()
        .lazy()
        .with_column(((col(column) - lagged.clone()) / lagged).alias(&pct_name))
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
        if df.column(col_name).is_err() {
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

    let result = df
        .clone()
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
        if df.column(col_name).is_err() {
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

    let result = df
        .clone()
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
    let col_data = df
        .column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;

    let float_col = col_data
        .cast(&DataType::Float64)
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;
    let float_ca = float_col
        .f64()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    // Calculate bin edges based on strategy
    let edges = match &strategy {
        BinStrategy::EqualWidth(n) => {
            let n = *n;
            check_bin_count(n)?;
            let min = float_ca.min().unwrap_or(0.0);
            let max = float_ca.max().unwrap_or(1.0);
            if !min.is_finite() || !max.is_finite() {
                return Err(MungeError::FeatureError(format!(
                    "cannot compute equal-width bins: column `{column}` contains non-finite values"
                )));
            }
            if !(max - min).is_finite() {
                return Err(MungeError::FeatureError(format!(
                    "the range of `{column}` is too wide to divide into equal-width bins"
                )));
            }
            let step = (max - min) / (n as f64);
            let mut edges = edge_buffer(n)?;
            for i in 0..=n {
                if i.is_multiple_of(CHECKPOINT_EVERY) {
                    crate::resource::checkpoint().map_err(feature_err)?;
                }
                edges.push(min + (i as f64) * step);
            }
            edges
        }
        BinStrategy::Quantile(n) => {
            check_bin_count(*n)?;
            quantile_edges(float_ca, *n)?
        }
        BinStrategy::Custom(edges) => {
            if edges.len() < 2 {
                return Err(MungeError::FeatureError(format!(
                    "custom bins need at least two edges, got {}",
                    edges.len()
                )));
            }
            if edges.iter().any(|e| e.is_nan()) {
                return Err(MungeError::FeatureError(
                    "custom bin edges must not contain NaN".to_string(),
                ));
            }
            if !edges.windows(2).all(|w| w[0] <= w[1]) {
                return Err(MungeError::FeatureError(
                    "custom bin edges must be in non-decreasing order".to_string(),
                ));
            }
            edges.clone()
        }
    };

    // Validate labels match number of bins
    let n_bins = edges.len().saturating_sub(1);
    if let Some(lbls) = labels {
        if lbls.len() != n_bins {
            return Err(MungeError::FeatureError(format!(
                "Expected {} labels for {} bins, got {}",
                n_bins,
                n_bins,
                lbls.len()
            )));
        }
    }

    // Create bin assignments
    let bin_name = format!("{}_binned", column);
    let mut bin_values: Vec<String> = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(feature_err)?;
        }
        let val = float_ca.get(i);
        let bin_idx = match val {
            // For sorted edges, `partition_point` is the unique j with
            // edges[j-1] < v <= edges[j]; values at or below the first edge,
            // above the last one, or NaN go to the first bin.
            Some(v) => match edges.partition_point(|e| *e < v) {
                j if j >= 1 && j < edges.len() => j - 1,
                _ => 0,
            },
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
    result
        .with_column(bin_col)
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

fn feature_err(e: crate::errors::EconError) -> MungeError {
    MungeError::FeatureError(e.to_string())
}

fn check_bin_count(n: usize) -> MungeResult<()> {
    if n == 0 {
        return Err(MungeError::FeatureError(
            "the number of bins must be at least 1".to_string(),
        ));
    }
    Ok(())
}

/// An empty, fallibly allocated buffer for the `n + 1` edges of `n` bins.
fn edge_buffer(n: usize) -> MungeResult<Vec<f64>> {
    let len = n.checked_add(1).ok_or_else(|| {
        MungeError::FeatureError(format!("{n} bins are too many: the edge count overflows"))
    })?;
    crate::resource::try_with_capacity::<f64>(len, "bin edges").map_err(feature_err)
}

/// All `n + 1` edges of `n` equal-frequency bins: `-inf`, the interior
/// quantiles `i / n` for `i` in `1..n`, and `+inf`.
///
/// The interior edges are bit-identical to polars'
/// `quantile(i / n, QuantileMethod::Linear)` (R's `quantile(type = 7)`), but
/// the column is sorted once instead of once per edge. An all-null or empty
/// column gives interior edges of `0.0`, as the per-edge `unwrap_or(0.0)` did.
fn quantile_edges(ca: &Float64Chunked, n: usize) -> MungeResult<Vec<f64>> {
    check_bin_count(n)?;
    let mut edges = edge_buffer(n)?;
    edges.push(f64::NEG_INFINITY);

    let len = ca.len();
    let nc = ca.null_count();
    let m = len - nc;
    // Nulls first, the order polars' generic quantile path uses.
    let sorted = if m > 0 { Some(ca.sort(false)) } else { None };

    for i in 1..n {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(feature_err)?;
        }
        let edge = match &sorted {
            None => 0.0,
            Some(sorted) => {
                // polars-core 0.52 `quantile_idx` and `linear_interpol`.
                let q = i as f64 / n as f64;
                let fi = (m as f64 - 1.0) * q + nc as f64;
                let lo = (fi as usize).clamp(0, len - 1);
                let top = fi.ceil() as usize;
                let lower = sorted.get(lo).unwrap_or(0.0);
                if lo == top {
                    lower
                } else {
                    let upper = sorted.get(lo + 1).unwrap_or(lower);
                    if lower == upper {
                        lower
                    } else {
                        (fi - lo as f64) * (upper - lower) + lower
                    }
                }
            }
        };
        edges.push(edge);
    }

    edges.push(f64::INFINITY);
    Ok(edges)
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
pub fn one_hot_encode(dataset: &Dataset, column: &str, drop_first: bool) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    let col_data = df
        .column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;

    // Get unique values as strings
    let unique = col_data
        .unique()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    // Convert to string representation
    let unique_str: Vec<String> = (0..unique.len())
        .map(|i| {
            unique
                .get(i)
                .map(|v| format!("{}", v))
                .unwrap_or_else(|_| "null".to_string())
        })
        .collect();

    // Skip first category if drop_first is true
    let start_idx = if drop_first { 1 } else { 0 };

    // The k x n dummy cells, checked before anything is built
    let height = df.height();
    let n_dummies = unique_str.len().saturating_sub(start_idx);
    let cells = crate::resource::checked_len(&[n_dummies, height], "one-hot dummy columns")
        .map_err(feature_err)?;
    crate::resource::check_alloc::<i32>(cells, "one-hot dummy columns").map_err(feature_err)?;

    // Each distinct display string maps to the index of its first occurrence,
    // so categories whose display strings coincide get the same dummy values,
    // as with the per-cell string comparison.
    let mut category_ids: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::with_capacity(unique_str.len());
    for (i, cat) in unique_str.iter().enumerate() {
        // u32::MAX marks a row without a category below.
        let id = u32::try_from(i)
            .ok()
            .filter(|&id| id != u32::MAX)
            .ok_or_else(|| {
                MungeError::FeatureError(format!(
                    "column `{column}` has too many distinct values to one-hot encode"
                ))
            })?;
        category_ids.entry(cat.as_str()).or_insert(id);
    }

    // Format every cell once and record its category (u32::MAX if none)
    let mut row_ids: Vec<u32> =
        crate::resource::try_with_capacity(height, "one-hot row categories")
            .map_err(feature_err)?;
    for row_idx in 0..height {
        if row_idx.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(feature_err)?;
        }
        let val = col_data
            .get(row_idx)
            .map(|v| format!("{}", v))
            .unwrap_or_else(|_| "null".to_string());
        row_ids.push(category_ids.get(val.as_str()).copied().unwrap_or(u32::MAX));
    }

    let mut result = df.clone();

    for cat in unique_str.iter().skip(start_idx) {
        crate::resource::checkpoint().map_err(feature_err)?;
        let id = category_ids[cat.as_str()];
        let dummy_name = format!("{}_{}", column, cat);
        let dummy_values: Vec<i32> = row_ids.iter().map(|&r| i32::from(r == id)).collect();

        let dummy_col = Column::new(dummy_name.into(), dummy_values);
        result
            .with_column(dummy_col)
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
pub fn cumsum(dataset: &Dataset, column: &str, reverse: bool) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let cumsum_name = format!("{}_cumsum", column);

    let result = df
        .clone()
        .lazy()
        .with_column(col(column).cum_sum(reverse).alias(&cumsum_name))
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
pub fn cumprod(dataset: &Dataset, column: &str, reverse: bool) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let cumprod_name = format!("{}_cumprod", column);

    let result = df
        .clone()
        .lazy()
        .with_column(col(column).cum_prod(reverse).alias(&cumprod_name))
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
pub fn rank(dataset: &Dataset, column: &str, descending: bool) -> MungeResult<Dataset> {
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

    let result = df
        .clone()
        .lazy()
        .with_column(col(column).rank(opts, None).alias(&rank_name))
        .collect()
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    Ok(Dataset::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;
    use rand::{Rng, SeedableRng};

    fn ten_values() -> Dataset {
        Dataset::new(df!("x" => [2.3, 1.1, 5.7, 3.3, 4.8, 0.4, 9.1, 6.6, 7.2, 3.9]).unwrap())
    }

    fn labels_of(ds: &Dataset, col: &str) -> Vec<String> {
        ds.df()
            .column(col)
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect()
    }

    /// The error text of a call that must fail (`Dataset` is not `Debug`).
    fn err_text(r: MungeResult<Dataset>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    fn idx(v: &[usize]) -> Vec<String> {
        v.iter().map(|i| format!("bin_{i}")).collect()
    }

    /// The pre-fix assignment: first j with edges[j-1] < v <= edges[j], else 0 (nulls 0).
    fn bin_reference(values: &[Option<f64>], edges: &[f64]) -> Vec<String> {
        values
            .iter()
            .map(|val| {
                let bin_idx = match val {
                    Some(v) => {
                        let mut idx = 0;
                        for j in 1..edges.len() {
                            if *v > edges[j - 1] && *v <= edges[j] {
                                idx = j - 1;
                                break;
                            }
                        }
                        idx
                    }
                    None => 0,
                };
                format!("bin_{bin_idx}")
            })
            .collect()
    }

    #[test]
    fn bin_matches_r_cut_for_equal_width_and_type7_quantiles() {
        // R 4.x:
        // x <- c(2.3, 1.1, 5.7, 3.3, 4.8, 0.4, 9.1, 6.6, 7.2, 3.9)
        // e <- min(x) + (0:4) * ((max(x) - min(x)) / 4)
        // cut(x, e, right = TRUE, include.lowest = TRUE, labels = FALSE) - 1     # 0 0 2 1 2 0 3 2 3 1
        // q <- quantile(x, c(.25, .5, .75), type = 7)   # 2.5499999999999998 4.3499999999999996 6.3749999999999991
        // cut(x, c(-Inf, q, Inf), labels = FALSE) - 1                             # 0 0 2 1 2 0 3 3 3 1
        // cut(x, c(-Inf, quantile(x, (1:4)/5, type = 7), Inf), labels = FALSE) - 1 # 1 0 3 1 2 0 4 3 4 2
        // cut(x, c(-Inf, 2, 5, Inf), labels = FALSE) - 1                          # 1 0 2 1 1 0 2 2 2 1
        let ds = ten_values();
        let ew = bin(&ds, "x", BinStrategy::EqualWidth(4), None).unwrap();
        assert_eq!(
            labels_of(&ew, "x_binned"),
            idx(&[0, 0, 2, 1, 2, 0, 3, 2, 3, 1])
        );
        let q4 = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        assert_eq!(
            labels_of(&q4, "x_binned"),
            idx(&[0, 0, 2, 1, 2, 0, 3, 3, 3, 1])
        );
        let q5 = bin(&ds, "x", BinStrategy::Quantile(5), None).unwrap();
        assert_eq!(
            labels_of(&q5, "x_binned"),
            idx(&[1, 0, 3, 1, 2, 0, 4, 3, 4, 2])
        );
        let c = bin(
            &ds,
            "x",
            BinStrategy::Custom(vec![f64::NEG_INFINITY, 2.0, 5.0, f64::INFINITY]),
            None,
        )
        .unwrap();
        assert_eq!(
            labels_of(&c, "x_binned"),
            idx(&[1, 0, 2, 1, 1, 0, 2, 2, 2, 1])
        );
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        let e = quantile_edges(&ca, 4).unwrap();
        assert_eq!(e.len(), 5);
        assert!(e[0] == f64::NEG_INFINITY && e[4] == f64::INFINITY);
        for (got, r) in
            e[1..4]
                .iter()
                .zip([2.5499999999999998, 4.3499999999999996, 6.3749999999999991])
        {
            assert!((got - r).abs() < 1e-12, "{got} vs R {r}");
        }
    }

    #[test]
    fn quantile_edges_equal_polars_linear_quantiles() {
        // Covers polars' slice path (no nulls, one chunk) and generic path (nulls; two chunks), with ties.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let tied: Vec<Option<f64>> = (0..1_000)
            .map(|_| {
                if rng.gen_bool(0.05) {
                    None
                } else {
                    Some((rng.gen_range(-50.0..50.0_f64) * 2.0).round() / 2.0)
                }
            })
            .collect();
        let smooth: Vec<f64> = (0..1_000).map(|_| rng.gen_range(-3.0..3.0_f64)).collect();
        let mut two_chunks = Float64Chunked::from_slice("x".into(), &smooth[..400]);
        two_chunks
            .append(&Float64Chunked::from_slice("x".into(), &smooth[400..]))
            .unwrap();
        let cases = [
            Float64Chunked::from_iter_options("x".into(), tied.iter().copied()),
            Float64Chunked::from_slice("x".into(), &smooth),
            two_chunks,
        ];
        for ca in &cases {
            for n in 1..=12usize {
                let got = quantile_edges(ca, n).unwrap();
                let want: Vec<f64> = (1..n)
                    .map(|i| {
                        ca.quantile(i as f64 / n as f64, QuantileMethod::Linear)
                            .unwrap()
                            .unwrap_or(0.0)
                    })
                    .collect();
                assert_eq!(&got[1..n], &want[..], "n = {n}");
            }
        }
        // Labels equal the old linear scan for every strategy.
        let ds = Dataset::new(df!("x" => tied.clone()).unwrap());
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        for (s, edges) in [
            (BinStrategy::Quantile(7), quantile_edges(&ca, 7).unwrap()),
            (
                BinStrategy::Custom(vec![-20.0, -5.0, -5.0, 0.0, 12.5, 30.0]),
                vec![-20.0, -5.0, -5.0, 0.0, 12.5, 30.0],
            ),
        ] {
            let out = bin(&ds, "x", s, None).unwrap();
            assert_eq!(labels_of(&out, "x_binned"), bin_reference(&tied, &edges));
        }
        let (mn, mx) = (ca.min().unwrap(), ca.max().unwrap());
        let step = (mx - mn) / 7.0;
        let ew: Vec<f64> = (0..=7).map(|i| mn + i as f64 * step).collect();
        let out = bin(&ds, "x", BinStrategy::EqualWidth(7), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), bin_reference(&tied, &ew));
    }

    #[test]
    fn bin_rejects_degenerate_requests() {
        let ds = ten_values();
        for s in [BinStrategy::EqualWidth(0), BinStrategy::Quantile(0)] {
            let e = err_text(bin(&ds, "x", s, None));
            assert!(e.contains("at least 1"), "{e}");
        }
        for edges in [
            vec![],
            vec![1.0],
            vec![0.0, f64::NAN, 2.0],
            vec![3.0, 1.0, 2.0],
        ] {
            assert!(
                bin(&ds, "x", BinStrategy::Custom(edges.clone()), None).is_err(),
                "{edges:?}"
            );
        }
        let inf = Dataset::new(df!("x" => [1.0, f64::INFINITY, 2.0]).unwrap());
        let e = err_text(bin(&inf, "x", BinStrategy::EqualWidth(3), None));
        assert!(e.contains("finite"), "{e}");
        let wide = Dataset::new(df!("x" => [-1e308, 0.0, 1e308]).unwrap());
        let e = err_text(bin(&wide, "x", BinStrategy::EqualWidth(3), None));
        assert!(e.contains("too wide"), "{e}");
    }

    #[test]
    fn bin_huge_count_is_an_error_not_an_abort() {
        // Keep this order: before the fix the first call aborts the test binary (8e18-byte edge vector);
        // the Quantile line would hang instead.
        let ds = ten_values();
        assert!(
            bin(
                &ds,
                "x",
                BinStrategy::EqualWidth(1_000_000_000_000_000_000),
                None
            )
            .is_err()
        );
        assert!(bin(&ds, "x", BinStrategy::EqualWidth(usize::MAX), None).is_err());
        assert!(
            bin(
                &ds,
                "x",
                BinStrategy::Quantile(1_000_000_000_000_000_000),
                None
            )
            .is_err()
        );
        let e = err_text(crate::resource::with_alloc_ceiling(Some(1 << 20), || {
            bin(&ds, "x", BinStrategy::EqualWidth(1_000_000), None)
        }));
        assert!(e.contains("limit"), "{e}");
    }

    /// The pre-fix algorithm, verbatim, as the reference (formats every cell
    /// once per category).
    fn one_hot_reference(ds: &Dataset, column: &str, drop_first: bool) -> DataFrame {
        let df = ds.df();
        let col_data = df.column(column).unwrap();
        let unique = col_data.unique().unwrap();
        let unique_str: Vec<String> = (0..unique.len())
            .map(|i| {
                unique
                    .get(i)
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|_| "null".to_string())
            })
            .collect();
        let mut result = df.clone();
        let start_idx = if drop_first { 1 } else { 0 };
        for cat in unique_str.iter().skip(start_idx) {
            let dummy_name = format!("{}_{}", column, cat);
            let mut dummy_values: Vec<i32> = Vec::with_capacity(df.height());
            for row_idx in 0..df.height() {
                let val = col_data
                    .get(row_idx)
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|_| "null".to_string());
                dummy_values.push(if val == *cat { 1 } else { 0 });
            }
            let dummy_col = Column::new(dummy_name.into(), dummy_values);
            result.with_column(dummy_col).unwrap();
        }
        result
    }

    #[test]
    fn one_hot_matches_per_cell_reference() {
        let ds = Dataset::new(
            df!(
                "region" => [Some("north"), Some("south"), None, Some("west"), Some("north"), Some("east")],
                "g" => [Some(1i64), Some(2), Some(2), None, Some(3), Some(1)],
                "h" => [1.5, 3.0, 1.5, 2.25, 3.0, 1.5],
            )
            .unwrap(),
        );
        for col in ["region", "g", "h"] {
            let got = one_hot_encode(&ds, col, false).unwrap();
            let want = one_hot_reference(&ds, col, false);
            assert_eq!(got.df().width(), want.width(), "{col}");
            for c in want.get_columns() {
                let g = got.df().column(c.name().as_str()).unwrap();
                assert!(
                    g.as_materialized_series()
                        .equals_missing(c.as_materialized_series()),
                    "{col}: {}",
                    c.name()
                );
            }
        }
        // 5 levels including null; drop_first removes one (which one depends
        // on polars' unique order).
        assert_eq!(
            one_hot_encode(&ds, "region", true).unwrap().df().width(),
            3 + 4
        );
    }

    #[test]
    fn one_hot_output_size_is_checked() {
        let ds = Dataset::new(df!("id" => (0..600i64).collect::<Vec<_>>()).unwrap());
        // 600 x 600 x 4 bytes > 1 MiB
        let r =
            crate::resource::with_alloc_ceiling(Some(1 << 20), || one_hot_encode(&ds, "id", false));
        assert!(err_text(r).contains("limit"));
        assert_eq!(one_hot_encode(&ds, "id", false).unwrap().df().width(), 601);
    }

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
        assert!(
            result.df().column("category_\"a\"").is_ok()
                || result.df().column("category_\"b\"").is_ok()
                || result.df().column("category_\"c\"").is_ok()
        );
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
        let n_dummy_cols = result
            .df()
            .get_columns()
            .iter()
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
