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

use std::fmt::Write as _;

use super::CHECKPOINT_EVERY;
use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;
use polars_ops::prelude::RankOptions;
use serde::{Deserialize, Serialize};

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

/// Discretize a numeric column into bins, as R's `cut()` does.
///
/// Adds the string column `{column}_binned`: `bin_0`, `bin_1`, ... or the
/// given `labels`, and null (NA) for a missing value (null or NaN) or a value
/// outside the edges. Intervals are right-closed, `(a, b]`, and a value equal
/// to the lowest edge is NA: [`bin_with`] takes R's `right` and
/// `include.lowest` options and reports the edges and counts.
///
/// Edges: `EqualWidth(n)` are `cut(x, n)`'s, the range of the present values
/// cut into `n` equal parts and widened by 0.1% of it at both ends, so that the
/// minimum and the maximum are in the first and last bin. `Quantile(n)` are
/// `-inf`, the type-7 quantiles `i / n` of the present values (R's
/// `quantile(type = 7, na.rm = TRUE)`) and `+inf`; an edge that interpolates
/// from a `-inf` lower neighbour is `-inf` (R's value next to a finite
/// neighbour), and a present `-inf` (`+inf`) that no interval holds goes into
/// the first (last) bin, so a column with a few `-inf` values, from `log(0)` say,
/// is binned. Interior edges must differ, as `cut()` refuses duplicated breaks:
/// when the `-inf` (`+inf`) values are frequent enough that two interior edges
/// are `-inf` (`+inf`), the call is refused (use fewer bins or custom edges).
/// `Custom(edges)` are used as given.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the column
/// * `column` - Numeric column to bin (integers, floats, decimals; text is
///   refused, as `cut()` refuses it)
/// * `strategy` - Binning strategy
/// * `labels` - Optional labels for the bins
///
/// # Errors
///
/// Returns an error when the column is missing or not numeric; when the
/// number of bins is zero or its edges cannot be allocated; when custom edges
/// are fewer than two, contain NaN or are not strictly increasing (R refuses
/// duplicated breaks; R sorts unsorted ones, but labels follow the given order
/// here, so they are refused); when an equal-width or quantile column has no
/// observed value; when the minimum or maximum of an equal-width column is not
/// finite or its range overflows; when the finite values of a quantile column
/// span a range that overflows or two interior quantile edges are equal; when
/// `labels` does not have one label per bin; and when the computation is
/// cancelled.
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
    bin_with(dataset, column, strategy, labels, &BinOptions::default()).map(|o| o.dataset)
}

/// How [`bin_with`] closes the intervals (R's `cut(right =, include.lowest =)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinOptions {
    /// `(a, b]` when true (R's default), `[a, b)` when false.
    pub right: bool,
    /// A value equal to the lowest edge goes into the first bin (the highest
    /// edge into the last bin when `right` is false). R's default is false.
    pub include_lowest: bool,
}

impl Default for BinOptions {
    /// R's defaults: `right = TRUE`, `include.lowest = FALSE`.
    fn default() -> Self {
        BinOptions {
            right: true,
            include_lowest: false,
        }
    }
}

/// Result of [`bin_with`].
#[derive(Clone)]
pub struct BinOutcome {
    /// Input dataset plus the new String column (null = NA).
    pub dataset: Dataset,
    /// Name of the new column, `{column}_binned`.
    pub column: String,
    /// Edges used, non-decreasing (strictly increasing except for repeated
    /// infinite quantile ends; EqualWidth: R `cut(x, n)` edges, the range
    /// widened by 0.1%).
    pub edges: Vec<f64>,
    /// (label, rows) per bin, in edge order.
    pub counts: Vec<(String, usize)>,
    /// Rows whose input was missing (null or NaN).
    pub n_missing: usize,
    /// Rows with a present value outside the edges.
    pub n_out_of_range: usize,
    /// The interval options used.
    pub options: BinOptions,
}

impl std::fmt::Debug for BinOutcome {
    /// Everything but the dataset, which is not `Debug`; its shape instead.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinOutcome")
            .field("dataset", &(self.dataset.nrows(), self.dataset.ncols()))
            .field("column", &self.column)
            .field("edges", &self.edges)
            .field("counts", &self.counts)
            .field("n_missing", &self.n_missing)
            .field("n_out_of_range", &self.n_out_of_range)
            .field("options", &self.options)
            .finish()
    }
}

impl BinOutcome {
    /// Two lines, e.g.
    /// `Edges: 0, 1, 3, 5; intervals (a, b]` and
    /// `Counts: bin_0: 2, bin_1: 3, bin_2: 2; NA: 5 (2 missing, 3 outside the edges)`.
    ///
    /// The interval text is `(a, b]` or `[a, b)`, followed by `, lowest edge
    /// included` / `, highest edge included` when `include_lowest`; the `; NA:
    /// ...` part is left out when no row is NA. Edges print in Rust's shortest
    /// round-trip form (`0`, `1.5`, `-inf`).
    pub fn summary(&self) -> String {
        let edges: Vec<String> = self.edges.iter().map(|e| e.to_string()).collect();
        let intervals = match (self.options.right, self.options.include_lowest) {
            (true, false) => "(a, b]",
            (true, true) => "(a, b], lowest edge included",
            (false, false) => "[a, b)",
            (false, true) => "[a, b), highest edge included",
        };
        let counts: Vec<String> = self
            .counts
            .iter()
            .map(|(label, n)| format!("{label}: {n}"))
            .collect();
        let mut text = format!(
            "Edges: {}; intervals {intervals}\nCounts: {}",
            edges.join(", "),
            counts.join(", ")
        );
        let na = self.n_missing + self.n_out_of_range;
        if na > 0 {
            text.push_str(&format!(
                "; NA: {na} ({} missing, {} outside the edges)",
                self.n_missing, self.n_out_of_range
            ));
        }
        text
    }
}

/// [`bin`] with R's interval options, reporting the edges used, the rows per
/// bin and the NA rows (missing inputs and values outside the edges).
///
/// A value `v` goes to bin `j` when `edges[j] < v <= edges[j + 1]` (`right`)
/// or `edges[j] <= v < edges[j + 1]` (not `right`); `include_lowest` closes the
/// first interval at the lowest edge (the last at the highest edge when not
/// `right`). This is R's `.bincode`. See [`bin`] for the edges and the errors.
pub fn bin_with(
    dataset: &Dataset,
    column: &str,
    strategy: BinStrategy,
    labels: Option<&[&str]>,
    options: &BinOptions,
) -> MungeResult<BinOutcome> {
    let df = dataset.df();
    // Booleans are refused, as R's cut(TRUE, 2) refuses them.
    let cells = super::numeric_cells(df, column, false)?;
    let no_values = || {
        MungeError::FeatureError(format!(
            "column '{column}' has no observed values; cannot compute bins"
        ))
    };

    // Calculate bin edges based on strategy
    let edges = match &strategy {
        BinStrategy::EqualWidth(n) => {
            let n = *n;
            check_bin_count(n)?;
            let (min, max) = cells
                .iter()
                .flatten()
                .fold(None, |acc: Option<(f64, f64)>, &v| {
                    Some(match acc {
                        None => (v, v),
                        Some((lo, hi)) => (lo.min(v), hi.max(v)),
                    })
                })
                .ok_or_else(no_values)?;
            if !min.is_finite() || !max.is_finite() {
                return Err(MungeError::FeatureError(format!(
                    "cannot compute equal-width bins: column `{column}` contains non-finite values"
                )));
            }
            let dx = max - min;
            if !dx.is_finite() {
                return Err(MungeError::FeatureError(format!(
                    "the range of `{column}` is too wide to divide into equal-width bins"
                )));
            }
            // R's cut.default: seq.int(min, max, length.out = n + 1) with the
            // ends moved out by dx / 1000; a constant column spans
            // min -/+ |min| / 1000 (1 / 1000 around 0) instead.
            let (from, to, step) = if dx == 0.0 {
                let d = if min != 0.0 { min.abs() } else { 1.0 };
                let (from, to) = (min - d / 1000.0, max + d / 1000.0);
                (from, to, (to - from) / (n as f64))
            } else {
                (min - dx / 1000.0, max + dx / 1000.0, dx / (n as f64))
            };
            let origin = if dx == 0.0 { from } else { min };
            let mut edges = edge_buffer(n)?;
            edges.push(from);
            for i in 1..n {
                if i.is_multiple_of(CHECKPOINT_EVERY) {
                    crate::resource::checkpoint().map_err(feature_err)?;
                }
                edges.push(origin + (i as f64) * step);
            }
            edges.push(to);
            edges
        }
        BinStrategy::Quantile(n) => {
            let n = *n;
            check_bin_count(n)?;
            let n_present = cells.iter().filter(|v| v.is_some()).count();
            if n_present == 0 {
                return Err(no_values());
            }
            let mut present = crate::resource::try_with_capacity::<f64>(n_present, "bin values")
                .map_err(feature_err)?;
            present.extend(cells.iter().flatten());
            let ca = Float64Chunked::from_slice(column.into(), &present);
            drop(present);
            let edges = quantile_edges(&ca, column, n)?;
            // R's cut refuses duplicated breaks. The outer edges are -inf and
            // +inf, and an interior edge may equal one of them when the data
            // hold infinite values (see `bin`); interior edges must differ.
            if let Some(w) = edges[1..n].windows(2).find(|w| w[0] == w[1]) {
                return Err(MungeError::FeatureError(format!(
                    "quantile edges of '{column}' are not unique ({} appears more than once); \
                     use fewer bins or custom edges",
                    w[0]
                )));
            }
            edges
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
            if !edges.windows(2).all(|w| w[0] < w[1]) {
                return Err(MungeError::FeatureError(
                    "custom bin edges must be strictly increasing (R: 'breaks' are not unique)"
                        .to_string(),
                ));
            }
            edges.clone()
        }
    };

    // `partition_point` below needs sorted edges. Every strategy produces
    // them: equal-width edges by construction, custom edges by the checks
    // above, quantile edges by `quantile_edges`' range check and its `-inf`
    // rule. This defensive check keeps an unsorted or NaN edge (`<=` rejects
    // both) from ever reaching the assignment; no input is known to trip it.
    if !edges.windows(2).all(|w| w[0] <= w[1]) {
        let kind = match strategy {
            BinStrategy::EqualWidth(_) => "equal-width",
            BinStrategy::Quantile(_) => "quantile",
            BinStrategy::Custom(_) => "custom",
        };
        return Err(MungeError::FeatureError(format!(
            "the {kind} bin edges computed for column `{column}` are not in non-decreasing order"
        )));
    }

    // Validate labels match number of bins
    let n_bins = edges.len() - 1;
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

    // R's .bincode: the bin of a present value, None outside the edges.
    let k = n_bins;
    let quantile = matches!(strategy, BinStrategy::Quantile(_));
    let bin_of = |v: f64| -> Option<usize> {
        let j = if options.right {
            edges.partition_point(|e| *e < v)
        } else {
            edges.partition_point(|e| *e <= v)
        };
        let found = match j {
            0 => (options.right && options.include_lowest && v == edges[0]).then_some(0),
            j if j <= k => Some(j - 1),
            _ => (!options.right && options.include_lowest && v == edges[k]).then_some(k - 1),
        };
        // The quantile bins are open at both ends: an infinite value no
        // interval holds goes to the outer bin on its side.
        found.or(if !quantile {
            None
        } else if v == f64::NEG_INFINITY {
            Some(0)
        } else if v == f64::INFINITY {
            Some(k - 1)
        } else {
            None
        })
    };

    // Create bin assignments. Each row's label goes straight into the
    // column's builder (an auto label through one reused buffer), so the
    // labels take no allocation per row beyond the column itself.
    let bin_name = format!("{}_binned", column);
    let mut builder = StringChunkedBuilder::new(bin_name.as_str().into(), df.height());
    let mut auto_label = String::new();
    let mut counts = crate::resource::try_vec(k, 0usize, "bin counts").map_err(feature_err)?;
    let (mut n_missing, mut n_out_of_range) = (0usize, 0usize);

    for (i, val) in cells.iter().enumerate() {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(feature_err)?;
        }
        let bin_idx = match *val {
            None => {
                n_missing += 1;
                builder.append_null();
                continue;
            }
            Some(v) => match bin_of(v) {
                Some(b) => b,
                None => {
                    n_out_of_range += 1;
                    builder.append_null();
                    continue;
                }
            },
        };
        counts[bin_idx] += 1;
        if let Some(lbls) = labels {
            builder.append_value(lbls[bin_idx]);
        } else {
            // Use auto-generated labels
            auto_label.clear();
            write!(auto_label, "bin_{bin_idx}")
                .map_err(|e| MungeError::FeatureError(e.to_string()))?;
            builder.append_value(auto_label.as_str());
        }
    }
    drop(cells);

    let bin_col = builder.finish().into_column();
    let mut result = df.clone();
    result
        .with_column(bin_col)
        .map_err(|e| MungeError::FeatureError(e.to_string()))?;

    let counts = counts
        .into_iter()
        .enumerate()
        .map(|(j, n)| {
            let label = match labels {
                Some(lbls) => lbls[j].to_string(),
                None => format!("bin_{j}"),
            };
            (label, n)
        })
        .collect();

    Ok(BinOutcome {
        dataset: Dataset::new(result),
        column: bin_name,
        edges,
        counts,
        n_missing,
        n_out_of_range,
        options: *options,
    })
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

/// All `n + 1` edges of `n` equal-frequency bins of the column `column`
/// (`ca`): `-inf`, the interior quantiles `i / n` for `i` in `1..n`, and
/// `+inf`.
///
/// NaN values are skipped like nulls. The interior edges are then
/// bit-identical to polars' `quantile(i / n, QuantileMethod::Linear)` (R's
/// `quantile(type = 7)`) of the column with its NaN values made null, but the
/// column is sorted once instead of once per edge. One exception: an edge
/// that interpolates from a `-inf` lower neighbour is `-inf`. For a finite
/// upper neighbour that is R's `(1 - h) * lower + h * upper` with `h < 1`,
/// where polars' formula gives NaN (`-inf + inf`); for a `+inf` upper
/// neighbour R and polars both give NaN, and this function still returns
/// `-inf`. A column without a value that is neither null nor NaN, or an empty
/// one, gives interior edges of `0.0`, as the per-edge `unwrap_or(0.0)` did
/// for an all-null column.
///
/// A column whose finite values span a range that overflows `f64` is
/// refused, as `EqualWidth` refuses it, whatever `n`: an interpolation
/// between two of its values could overflow to `+inf` and misplace or
/// unsort the edges. With that check and the `-inf` rule the edges are
/// non-decreasing.
fn quantile_edges(ca: &Float64Chunked, column: &str, n: usize) -> MungeResult<Vec<f64>> {
    check_bin_count(n)?;
    let mut edges = edge_buffer(n)?;
    edges.push(f64::NEG_INFINITY);

    let len = ca.len();
    let nc = ca.null_count();
    let mut n_nan = 0usize;
    let mut finite_range: Option<(f64, f64)> = None;
    for v in ca.into_iter().flatten() {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_finite() {
            finite_range = Some(match finite_range {
                None => (v, v),
                Some((lo_v, hi_v)) => (lo_v.min(v), hi_v.max(v)),
            });
        }
    }
    // Two finite values differ by at most this range, so once it is finite
    // no interpolation between finite values overflows.
    if finite_range.is_some_and(|(lo_v, hi_v)| !(hi_v - lo_v).is_finite()) {
        return Err(MungeError::FeatureError(format!(
            "the range of `{column}` is too wide to divide into quantile bins"
        )));
    }
    // `m` values are neither null nor NaN. polars sorts nulls first (the order
    // its generic quantile path uses) and NaN last, so the sorted entries
    // `nc..nc + m` are exactly those values and the interpolation below never
    // reads a NaN.
    let m = len - nc - n_nan;
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
                    } else if lower == f64::NEG_INFINITY {
                        // polars' formula gives `inf + -inf` = NaN here.
                        f64::NEG_INFINITY
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

    /// The labels of a binned column; `None` is NA.
    fn labels_of(ds: &Dataset, col: &str) -> Vec<Option<String>> {
        ds.df()
            .column(col)
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }

    /// The bin index of each row of a `bin_<k>` column, -1 for NA: the coding of the R
    /// case `data_bin_cut`.
    fn codes_of(ds: &Dataset, col: &str) -> Vec<f64> {
        labels_of(ds, col)
            .into_iter()
            .map(|l| match l {
                None => -1.0,
                Some(l) => l["bin_".len()..].parse().unwrap(),
            })
            .collect()
    }

    /// The error text of a call that must fail (`Dataset` is not `Debug`).
    fn err_text(r: MungeResult<Dataset>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    fn idx(v: &[usize]) -> Vec<Option<String>> {
        v.iter().map(|i| Some(format!("bin_{i}"))).collect()
    }

    /// R's right-closed `cut` by a linear scan: `bin_{j-1}` for the j with
    /// `edges[j-1] < v <= edges[j]`; NA (`None`) for null, NaN and values outside the edges.
    fn cut_reference(values: &[Option<f64>], edges: &[f64]) -> Vec<Option<String>> {
        values
            .iter()
            .map(|val| {
                let v = (*val)?;
                (1..edges.len())
                    .find(|&j| v > edges[j - 1] && v <= edges[j])
                    .map(|j| format!("bin_{}", j - 1))
            })
            .collect()
    }

    /// [`cut_reference`] with the Quantile strategy's rule for infinite values: a present
    /// `-inf` that no interval holds is in the first bin, a present `+inf` in the last.
    fn quantile_reference(values: &[Option<f64>], edges: &[f64]) -> Vec<Option<String>> {
        cut_reference(values, edges)
            .into_iter()
            .zip(values)
            .map(|(label, v)| {
                label.or_else(|| match v {
                    Some(v) if *v == f64::NEG_INFINITY => Some("bin_0".to_string()),
                    Some(v) if *v == f64::INFINITY => Some(format!("bin_{}", edges.len() - 2)),
                    _ => None,
                })
            })
            .collect()
    }

    #[test]
    fn bin_matches_r_cut_for_equal_width_and_type7_quantiles() {
        // R 4.x:
        // x <- c(2.3, 1.1, 5.7, 3.3, 4.8, 0.4, 9.1, 6.6, 7.2, 3.9)
        // cut(x, 4, labels = FALSE) - 1                                           # 0 0 2 1 2 0 3 2 3 1
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
        let e = quantile_edges(&ca, "x", 4).unwrap();
        assert_eq!(e.len(), 5);
        assert!(e[0] == f64::NEG_INFINITY && e[4] == f64::INFINITY);
        // R's 17-digit values above, written in their shortest round-trip form.
        for (got, r) in e[1..4].iter().zip([2.55, 4.35, 6.374999999999999]) {
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
                let got = quantile_edges(ca, "x", n).unwrap();
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
        // Labels equal R's cut by a linear scan for every strategy: the quantile bins
        // hold every present value (open outer bins), custom edges leave values outside
        // them NA, and the equal-width edges are cut(x, 7)'s.
        let ds = Dataset::new(df!("x" => tied.clone()).unwrap());
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        for (s, edges) in [
            (
                BinStrategy::Quantile(7),
                quantile_edges(&ca, "x", 7).unwrap(),
            ),
            (
                BinStrategy::Custom(vec![-20.0, -5.0, 0.0, 12.5, 30.0]),
                vec![-20.0, -5.0, 0.0, 12.5, 30.0],
            ),
        ] {
            let out = bin(&ds, "x", s, None).unwrap();
            assert_eq!(labels_of(&out, "x_binned"), cut_reference(&tied, &edges));
        }
        let (mn, mx) = (ca.min().unwrap(), ca.max().unwrap());
        let dx = mx - mn;
        let mut ew: Vec<f64> = (0..=7).map(|i| mn + i as f64 * (dx / 7.0)).collect();
        ew[0] = mn - dx / 1000.0;
        ew[7] = mx + dx / 1000.0;
        let out = bin(&ds, "x", BinStrategy::EqualWidth(7), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), cut_reference(&tied, &ew));
    }

    /// The binned column is the String column of one label per row that
    /// the per-row assignment gives (auto labels, given labels, null for a
    /// null input), from a column of several chunks.
    #[test]
    fn bin_column_holds_each_rows_label() {
        let values: Vec<Option<f64>> = (0..300)
            .map(|i| (i % 7 != 3).then(|| (i as f64 * 0.61).sin() * 9.0 + 0.2))
            .collect();
        let mut x = Float64Chunked::from_iter_options("x".into(), values[..120].iter().copied());
        x.append(&Float64Chunked::from_iter_options(
            "x".into(),
            values[120..].iter().copied(),
        ))
        .unwrap();
        let ds = Dataset::new(DataFrame::new(vec![x.into_column()]).unwrap());
        assert!(ds.df().max_n_chunks() > 1);
        let edges = vec![-10.0, -3.0, 0.0, 4.5, 10.0];
        let long = "a label longer than a view holds inline";
        let given = ["neg", long, "", "pos"];
        let want_auto = cut_reference(&values, &edges);
        let want_given: Vec<Option<String>> = want_auto
            .iter()
            .map(|l| {
                l.as_ref()
                    .map(|l| given[l["bin_".len()..].parse::<usize>().unwrap()].to_string())
            })
            .collect();
        for (labels, want) in [(None, &want_auto), (Some(&given[..]), &want_given)] {
            let out = bin(&ds, "x", BinStrategy::Custom(edges.clone()), labels).unwrap();
            let got = out.df().column("x_binned").unwrap();
            assert_eq!(got.dtype(), &DataType::String);
            assert_eq!(got.null_count(), 43);
            let expected = Column::new("x_binned".into(), want.clone());
            assert!(
                got.as_materialized_series()
                    .equals_missing(expected.as_materialized_series()),
                "{labels:?}"
            );
            assert!(
                out.df()
                    .column("x")
                    .unwrap()
                    .equals_missing(ds.df().column("x").unwrap())
            );
        }
        // Quantile bins name their bins by the same rule.
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        let q = quantile_edges(&ca, "x", 6).unwrap();
        let out = bin(&ds, "x", BinStrategy::Quantile(6), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), quantile_reference(&values, &q));
    }

    #[test]
    fn bin_rejects_degenerate_requests() {
        let ds = ten_values();
        for s in [BinStrategy::EqualWidth(0), BinStrategy::Quantile(0)] {
            let e = err_text(bin(&ds, "x", s, None));
            assert!(e.contains("at least 1"), "{e}");
        }
        // R: cut(1:5, c(0, 2, 2, 5)) is an error ("'breaks' are not unique").
        for edges in [
            vec![],
            vec![1.0],
            vec![0.0, f64::NAN, 2.0],
            vec![3.0, 1.0, 2.0],
            vec![0.0, 2.0, 2.0, 5.0],
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
    fn quantile_bins_treat_nan_like_null() {
        // R 4.x:
        // x <- c(1, 2, 3, NaN)
        // q <- quantile(x, c(.25, .5, .75), type = 7, na.rm = TRUE)   # 1.5 2 2.5
        // cut(x, c(-Inf, q, Inf), labels = FALSE) - 1                 # 0 1 3 NA
        let ds = Dataset::new(df!("x" => [1.0, 2.0, 3.0, f64::NAN]).unwrap());
        let out = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        let mut want = idx(&[0, 1, 3]);
        want.push(None);
        assert_eq!(labels_of(&out, "x_binned"), want);

        // A NaN anywhere gives the edges of the same column with that NaN made null.
        let with_nan = Float64Chunked::from_iter_options(
            "x".into(),
            [4.2, f64::NAN, -9.0, -1.3, 0.7, f64::NAN, 2.9]
                .into_iter()
                .map(|v| (v != -9.0).then_some(v)),
        );
        let as_null = Float64Chunked::from_iter_options(
            "x".into(),
            [4.2, f64::NAN, -9.0, -1.3, 0.7, f64::NAN, 2.9]
                .into_iter()
                .map(|v| (v != -9.0 && !v.is_nan()).then_some(v)),
        );
        for n in 1..=6usize {
            assert_eq!(
                quantile_edges(&with_nan, "x", n).unwrap(),
                quantile_edges(&as_null, "x", n).unwrap(),
                "n = {n}"
            );
        }
        // An all-NaN column has the interior edges of an all-null one.
        let all_nan = Float64Chunked::from_slice("x".into(), &[f64::NAN, f64::NAN]);
        assert_eq!(
            quantile_edges(&all_nan, "x", 2).unwrap(),
            vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]
        );
    }

    #[test]
    fn quantile_edges_next_to_infinite_values_stay_sorted() {
        const NEG: f64 = f64::NEG_INFINITY;
        const POS: f64 = f64::INFINITY;
        // R 4.x:
        // quantile(c(-Inf, 1, 2, 3), c(.25, .5, .75), type = 7)   # -Inf 1.5 2.25
        // polars' formula gives NaN for the first edge (-inf + inf); bin() used to refuse the column.
        let ds = Dataset::new(df!("x" => [NEG, 1.0, 2.0, 3.0]).unwrap());
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        assert_eq!(
            quantile_edges(&ca, "x", 4).unwrap(),
            vec![NEG, NEG, 1.5, 2.25, POS]
        );
        let out = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), idx(&[0, 1, 2, 3]));
        // Edges [-inf, -inf, 0.5, inf, inf].
        let ds = Dataset::new(df!("x" => [NEG, 0.5, POS]).unwrap());
        let out = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), idx(&[0, 1, 2]));
        // The one edge that interpolates between -inf and +inf. R gives NaN for it
        // (quantile(c(-Inf, Inf), .5, type = 7) is NaN), as does polars' formula; -inf is the
        // documented choice, and with NaN bin() would refuse the column through its edge guard.
        let ds = Dataset::new(df!("x" => [NEG, POS]).unwrap());
        let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
        assert_eq!(quantile_edges(&ca, "x", 2).unwrap(), vec![NEG, NEG, POS]);
        assert_eq!(
            labels_of(
                &bin(&ds, "x", BinStrategy::Quantile(2), None).unwrap(),
                "x_binned"
            ),
            idx(&[0, 1])
        );
        // +inf only pushes the top edges to +inf ([-inf, 1.5, 2, inf, inf]), which keeps them sorted.
        let ds = Dataset::new(df!("x" => [1.0, 2.0, POS]).unwrap());
        let out = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), idx(&[0, 1, 2]));

        // Whether a column with -inf values (log(0)) bins no longer depends on their count: with
        // 100 of 1000 rows, the first decile edge (index 99.9) sat between the last -inf and the
        // first finite value and came out NaN; with 101 it lay between two -inf values.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        for n_neg in [100usize, 101] {
            let x: Vec<Option<f64>> = (0..1_000)
                .map(|i| {
                    if i < n_neg {
                        NEG
                    } else {
                        rng.gen_range(-5.0..5.0)
                    }
                })
                .map(Some)
                .collect();
            let ds = Dataset::new(df!("x" => x.clone()).unwrap());
            let ca = ds.df().column("x").unwrap().f64().unwrap().clone();
            let edges = quantile_edges(&ca, "x", 10).unwrap();
            assert_eq!(edges[1], NEG, "n_neg = {n_neg}");
            assert!(edges[2..].iter().all(|e| e.is_finite() || *e == POS));
            assert!(edges.windows(2).all(|w| w[0] <= w[1]), "{edges:?}");
            let labels = labels_of(
                &bin(&ds, "x", BinStrategy::Quantile(10), None).unwrap(),
                "x_binned",
            );
            assert_eq!(labels, quantile_reference(&x, &edges), "n_neg = {n_neg}");
            assert!(
                labels[..n_neg]
                    .iter()
                    .all(|l| l.as_deref() == Some("bin_0"))
            );
            assert!(
                labels[n_neg..]
                    .iter()
                    .all(|l| l.is_some() && l.as_deref() != Some("bin_0"))
            );
        }
    }

    #[test]
    fn quantile_bins_refuse_a_finite_range_that_overflows() {
        // `upper - lower` overflows to +inf. [-1e308, 1e308] gave the edges [-inf, inf, inf] and
        // put both rows in bin_0 (R's median is 0: bins 0 and 1); [-1e308, 1e308, 1.5e308] gave
        // the unsorted edges [-inf, inf, 1e308, 1.25e308, inf]. The refusal does not depend on the
        // number of bins, and an infinite value does not hide the overflowing finite range.
        for (x, n) in [
            (vec![-1e308, 1e308], 2),
            (vec![-1e308, 1e308, 1.5e308], 4),
            (vec![-1e308, 1e308], 1),
            (vec![f64::NEG_INFINITY, -1e308, 1e308, 1.5e308], 4),
        ] {
            let ds = Dataset::new(df!("x" => x.clone()).unwrap());
            let e = err_text(bin(&ds, "x", BinStrategy::Quantile(n), None));
            assert!(
                e.contains("`x`") && e.contains("too wide"),
                "{x:?}, {n}: {e}"
            );
        }
        // A wide but finite range bins, and infinite values do not count towards the range:
        // edges [-inf, -inf, 0, inf, inf].
        let ds =
            Dataset::new(df!("x" => [f64::NEG_INFINITY, -1e307, 1e307, f64::INFINITY]).unwrap());
        let out = bin(&ds, "x", BinStrategy::Quantile(4), None).unwrap();
        assert_eq!(labels_of(&out, "x_binned"), idx(&[0, 1, 2, 2]));
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

    #[test]
    fn test_validate_bin_cut_right_closed() {
        use p2a_validation::{Check, Tol};
        // validation/r/cases/data_quality.R, case data_bin_cut: cut(x, breaks = c(0, 1, 3, 5))
        let (case, ds) = crate::test_support::load_case("data_bin_cut");
        let out = bin(
            &ds,
            "x",
            BinStrategy::Custom(vec![0.0, 1.0, 3.0, 5.0]),
            None,
        )
        .unwrap();
        let c = codes_of(&out, "x_binned");
        Check::new(&case)
            .vector("right_codes", &c, Tol::exact())
            .scalar(
                "n_na_right",
                c.iter().filter(|v| **v < 0.0).count() as f64,
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_bin_cut_options() {
        use p2a_validation::{Check, Tol};
        // Same case: include.lowest = TRUE closes [0, 1]; right = FALSE gives [0,1) [1,3) [3,5).
        let (case, ds) = crate::test_support::load_case("data_bin_cut");
        let edges = || BinStrategy::Custom(vec![0.0, 1.0, 3.0, 5.0]);
        let right = bin_with(&ds, "x", edges(), None, &BinOptions::default()).unwrap();
        let low = bin_with(
            &ds,
            "x",
            edges(),
            None,
            &BinOptions {
                right: true,
                include_lowest: true,
            },
        )
        .unwrap();
        let left = bin_with(
            &ds,
            "x",
            edges(),
            None,
            &BinOptions {
                right: false,
                include_lowest: false,
            },
        )
        .unwrap();
        Check::new(&case)
            .vector(
                "right_codes",
                &codes_of(&right.dataset, "x_binned"),
                Tol::exact(),
            )
            .vector(
                "include_lowest_codes",
                &codes_of(&low.dataset, "x_binned"),
                Tol::exact(),
            )
            .vector(
                "left_closed_codes",
                &codes_of(&left.dataset, "x_binned"),
                Tol::exact(),
            )
            .scalar(
                "n_na_right",
                (right.n_missing + right.n_out_of_range) as f64,
                Tol::exact(),
            )
            .finish();
        assert_eq!(
            (right.n_missing, right.n_out_of_range, low.n_out_of_range),
            (2, 3, 2)
        );
        assert_eq!(right.edges, vec![0.0, 1.0, 3.0, 5.0]);
        assert_eq!(right.column, "x_binned");
        assert_eq!(
            right.summary(),
            "Edges: 0, 1, 3, 5; intervals (a, b]\n\
             Counts: bin_0: 2, bin_1: 3, bin_2: 2; NA: 5 (2 missing, 3 outside the edges)"
        );
        assert_eq!(
            low.summary(),
            "Edges: 0, 1, 3, 5; intervals (a, b], lowest edge included\n\
             Counts: bin_0: 3, bin_1: 3, bin_2: 2; NA: 4 (2 missing, 2 outside the edges)"
        );
        let left_low = bin_with(
            &ds,
            "x",
            edges(),
            Some(&["lo", "mid", "hi"]),
            &BinOptions {
                right: false,
                include_lowest: true,
            },
        )
        .unwrap();
        // R: cut(x, c(0, 1, 3, 5), right = FALSE, include.lowest = TRUE) puts 5 in [3, 5].
        assert_eq!(
            left_low.summary(),
            "Edges: 0, 1, 3, 5; intervals [a, b), highest edge included\n\
             Counts: lo: 2, mid: 3, hi: 3; NA: 4 (2 missing, 2 outside the edges)"
        );
    }

    #[test]
    fn test_bin_missing_nan_and_out_of_range_values_are_na() {
        // DATA-12 repro. R:
        // cut(c(-5, 0, 5, 15, 25, NA, NaN), c(0, 10, 20), labels = c("low", "high"))
        // <NA> <NA> low high <NA> <NA> <NA>
        let ds = Dataset::new(
            df!("x" => [Some(-5.0), Some(0.0), Some(5.0), Some(15.0), Some(25.0), None,
                        Some(f64::NAN)])
            .unwrap(),
        );
        let out = bin(
            &ds,
            "x",
            BinStrategy::Custom(vec![0.0, 10.0, 20.0]),
            Some(&["low", "high"]),
        )
        .unwrap();
        assert_eq!(
            labels_of(&out, "x_binned"),
            vec![
                None,
                None,
                Some("low".into()),
                Some("high".into()),
                None,
                None,
                None
            ]
        );
        let o = bin_with(
            &ds,
            "x",
            BinStrategy::Custom(vec![0.0, 10.0, 20.0]),
            None,
            &BinOptions::default(),
        )
        .unwrap();
        assert_eq!((o.n_missing, o.n_out_of_range), (2, 3));
        assert_eq!(
            o.counts,
            vec![("bin_0".to_string(), 1), ("bin_1".to_string(), 1)]
        );
    }

    #[test]
    fn test_bin_equal_width_follows_cut_n() {
        // R: x10 <- c(2.3, 1.1, 5.7, 3.3, 4.8, 0.4, 9.1, 6.6, 7.2, 3.9); cut(x10, 7)
        // breaks 0.39130000000000004 ... 9.1086999999999989, codes - 1: 1 0 4 2 3 0 6 4 5 2
        let o = bin_with(
            &ten_values(),
            "x",
            BinStrategy::EqualWidth(7),
            None,
            &BinOptions::default(),
        )
        .unwrap();
        assert_eq!(
            codes_of(&o.dataset, "x_binned"),
            vec![1., 0., 4., 2., 3., 0., 6., 4., 5., 2.]
        );
        assert_eq!(
            o.edges,
            [
                0.39130000000000004,
                1.6428571428571428,
                2.8857142857142852,
                4.1285714285714281,
                5.371428571428571,
                6.6142857142857139,
                7.8571428571428559,
                9.1086999999999989
            ]
        );
        // R: cut(c(3, 3, 3), 2) has breaks 2.997 3 3.003, codes 0 0 0.
        let c = bin_with(
            &Dataset::new(df!("x" => [3.0, 3.0, 3.0]).unwrap()),
            "x",
            BinStrategy::EqualWidth(2),
            None,
            &BinOptions::default(),
        )
        .unwrap();
        assert_eq!(
            (codes_of(&c.dataset, "x_binned"), c.edges.clone()),
            (
                vec![0., 0., 0.],
                vec![2.9969999999999999, 3.0, 3.0030000000000001]
            )
        );
        // R: cut(c(0, 0), 3) codes 1 1.
        let z = bin(
            &Dataset::new(df!("x" => [0.0, 0.0]).unwrap()),
            "x",
            BinStrategy::EqualWidth(3),
            None,
        )
        .unwrap();
        assert_eq!(codes_of(&z, "x_binned"), vec![1., 1.]);
    }

    #[test]
    fn test_na_bin_fixtures() {
        // Int64 accepted (R cut(1:4, 2): 0 0 1 1); text refused (R: cut("1.5", 2)
        // is an error, "'x' must be numeric"); an absent column is named; a column without an
        // observed value has no equal-width or quantile edges.
        let i = bin(
            &Dataset::new(df!("x" => [1i64, 2, 3, 4]).unwrap()),
            "x",
            BinStrategy::EqualWidth(2),
            None,
        )
        .unwrap();
        assert_eq!(codes_of(&i, "x_binned"), vec![0., 0., 1., 1.]);
        let e = err_text(bin(
            &Dataset::new(df!("x" => ["1.5", "2.5"]).unwrap()),
            "x",
            BinStrategy::EqualWidth(2),
            None,
        ));
        assert!(e.contains("'x'") && e.contains("numeric"), "{e}");
        assert!(matches!(
            bin(&ten_values(), "nope", BinStrategy::EqualWidth(2), None),
            Err(MungeError::ColumnNotFound(_))
        ));
        let empty = Dataset::new(df!("x" => [None::<f64>, None]).unwrap());
        for s in [BinStrategy::EqualWidth(2), BinStrategy::Quantile(2)] {
            let e = err_text(bin(&empty, "x", s, None));
            assert!(e.contains("no observed values"), "{e}");
        }
        // Custom edges need no observed value: every row is NA.
        let all_na = bin(&empty, "x", BinStrategy::Custom(vec![0.0, 1.0]), None).unwrap();
        assert_eq!(codes_of(&all_na, "x_binned"), vec![-1., -1.]);
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
