//! Partial Dependence Plots (PDP) for model interpretability.
//!
//! Computes the marginal effect of features on model predictions, enabling
//! visualization and understanding of how features influence predictions.
//!
//! ## Overview
//!
//! Partial dependence shows the average predicted outcome when a feature is set
//! to a particular value, marginalizing over all other features:
//!
//! PDP(x_s) = E_X_c[f(x_s, X_c)] = (1/n) * sum_i f(x_s, x_c_i)
//!
//! where:
//! - x_s is the feature(s) of interest
//! - X_c are the complement features
//! - f is the prediction function
//!
//! ## Example
//!
//! ```rust
//! use p2a_core::ml::{PdpConfig, partial_dependence, random_forest_with_trees};
//! use ndarray::Array2;
//!
//! // 60 observations: the target depends on x0 alone, x1 is a nuisance feature.
//! let x = Array2::from_shape_fn((60, 2), |(i, j)| {
//!     if j == 0 { i as f64 / 10.0 } else { (i % 5) as f64 }
//! });
//! let y = x.column(0).mapv(|x0| 2.0 * x0);
//! let model =
//!     random_forest_with_trees(x.view(), y.view(), Some(25), None, None, None, Some(42), None)
//!         .unwrap();
//!
//! // A forest prediction is the mean over its trees, so this closure is the
//! // fitted model's prediction function for an arbitrary design matrix.
//! let predict = |data: &Array2<f64>| -> Vec<f64> {
//!     data.rows()
//!         .into_iter()
//!         .map(|row| {
//!             model.trees.iter().map(|t| t.predict_one(&row)).sum::<f64>()
//!                 / model.trees.len() as f64
//!         })
//!         .collect()
//! };
//!
//! // Marginal effect of feature 0, averaging over the observed values of x1.
//! // `PdpConfig` controls the grid; the default is pdp's: min(distinct values, 51) uniform points.
//! let config = PdpConfig {
//!     grid_resolution: Some(25),
//!     ..Default::default()
//! };
//! let pdp = partial_dependence(&x, 0, predict, &config).unwrap();
//!
//! assert_eq!(pdp.grid_values.len(), 25);
//! assert_eq!(pdp.n_obs, 60);
//! // y increases in x0, so the partial dependence rises across the grid.
//! assert!(*pdp.pdp_values.last().unwrap() > pdp.pdp_values[0]);
//! ```
//!
//! ## The engine
//!
//! Every partial-dependence and ICE entry point of the crate runs on two functions of this module, so the same
//! model gets the same grid and the same checks whichever one is called: [`crate::ml::compute_ice_curves`] and
//! the model wrappers in [`crate::ml::ice`], and [`crate::ml::gbm_partial_dependence`] and
//! [`crate::ml::cart_partial_dependence`].
//!
//! [`feature_grid`] builds a grid with the rules of `pdp::partial`'s `pred_grid` (pdp 0.8.3):
//! - [`GridSpec::Uniform`] is `seq(min(x), max(x), length.out = n)` (`grid.resolution = n`). With no
//!   resolution set, the configs use pdp's default `n = min(length(unique(x)), 51)`, counting non-missing
//!   values (pdp counts a missing value as one more);
//! - [`GridSpec::Quantiles`] is `quantile(x, probs, type = 7, na.rm = TRUE)` (`quantiles = TRUE, probs`);
//! - [`GridSpec::Values`] is a user grid (`pred.grid`), sorted.
//!
//! Exact duplicate grid values are listed once (pdp repeats the row with the same `yhat`). NaN feature values
//! are ignored when a grid is built; infinite values are an error.
//!
//! The evaluation loop checks the length of every prediction vector, returns a fallible model's errors and stops
//! when the call's budget is cancelled. A NaN prediction propagates into the mean, unlike pdp's
//! `mean(na.rm = TRUE)`: averaging over a subset of the rows would change the estimand without saying so.
//!
//! ## References
//!
//! - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine".
//!   Annals of Statistics, 29(5), 1189-1232.
//!   <https://doi.org/10.1214/aos/1013203451>
//!
//! - R package `pdp`: Greenwell, B. M. (2017). "pdp: An R Package for Constructing
//!   Partial Dependence Plots". The R Journal, 9(1), 421-436.
//!   <https://journal.r-project.org/archive/2017/RJ-2017-016/>
//!
//! - Python library `scikit-learn`: sklearn.inspection.partial_dependence
//!   <https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html>

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_with_capacity};
use crate::stats::robust::{QuantileType, quantile_sorted};

/// pdp's cap on its default grid, `min(length(unique(x)), 51)`.
const DEFAULT_MAX_GRID_POINTS: usize = 51;

/// How the grid of one feature is built: the rules of `pdp::partial`'s `pred_grid` (pdp 0.8.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GridSpec {
    /// `seq(min(x), max(x), length.out = n)` over the non-missing values (pdp `grid.resolution = n`).
    Uniform(usize),
    /// `quantile(x, probs, type = 7, na.rm = TRUE)` (pdp `quantiles = TRUE, probs`).
    Quantiles(Vec<f64>),
    /// User grid values (pdp `pred.grid`).
    Values(Vec<f64>),
}

/// Configuration for Partial Dependence Plot computation.
///
/// The default is `pdp::partial`'s: uniform points, `min(length(unique(x)), 51)` of them, uncentered.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PdpConfig {
    /// Number of grid points, at least 1. With `use_quantiles` the quantiles at
    /// `seq(0, 1, length.out = grid_resolution)`, else `seq(min, max, length.out = grid_resolution)`.
    ///
    /// `None` (default) is pdp's choice: `min(length(unique(x)), 51)` uniform points, or the quantiles at
    /// `1:9/10` with `use_quantiles`.
    pub grid_resolution: Option<usize>,

    /// Use quantiles for grid points instead of uniform spacing (default: false), type 7, as
    /// `pdp::partial(quantiles = TRUE)`.
    ///
    /// When true, grid points are placed at quantiles of the feature distribution,
    /// which better captures the data density. When false, uses uniform spacing
    /// between min and max values.
    pub use_quantiles: bool,

    /// Explicit probabilities (pdp's `probs`), each in [0, 1]; override `grid_resolution` and `use_quantiles`.
    pub quantiles: Option<Vec<f64>>,

    /// PD minus its mean over the grid (default: false). `pdp::partial` has no PD centring; for ICE see
    /// [`crate::ml::IceConfig::center`].
    pub center: bool,

    /// Number of grid points for the second feature of [`partial_dependence_2d`] (default:
    /// `grid_resolution`). [`partial_dependence`] ignores it.
    pub grid_resolution_2: Option<usize>,
}

impl PdpConfig {
    /// The grid specification for a feature with these values: `quantiles` when set -> `Quantiles(probs)`;
    /// else `use_quantiles` -> `Quantiles(seq(0, 1, length.out = grid_resolution))` (`1:9/10` without a
    /// resolution); else `Uniform(grid_resolution)` (`Uniform(min(length(unique(x)), 51))` without one).
    ///
    /// # Errors
    ///
    /// `InvalidSpecification` when `grid_resolution` is `Some(0)` and used; `ResourceLimit` for a size that
    /// cannot be allocated.
    pub fn grid_spec(&self, values: &[f64]) -> EconResult<GridSpec> {
        self.spec_with(self.grid_resolution, "grid_resolution", values)
    }

    /// The second feature of the 2-D PD: the same rule with `grid_resolution_2`, or `grid_resolution` when that
    /// is `None`.
    ///
    /// # Errors
    ///
    /// As [`PdpConfig::grid_spec`], naming `grid_resolution_2` when that is `Some(0)`.
    pub fn grid_spec_2(&self, values: &[f64]) -> EconResult<GridSpec> {
        match self.grid_resolution_2 {
            Some(n) => self.spec_with(Some(n), "grid_resolution_2", values),
            None => self.grid_spec(values),
        }
    }

    fn spec_with(
        &self,
        resolution: Option<usize>,
        name: &str,
        values: &[f64],
    ) -> EconResult<GridSpec> {
        if let Some(probs) = &self.quantiles {
            return Ok(GridSpec::Quantiles(probs.clone()));
        }
        if resolution == Some(0) {
            return Err(EconError::InvalidSpecification {
                message: format!("{name} must be at least 1, got 0"),
            });
        }
        Ok(match (self.use_quantiles, resolution) {
            (true, Some(n)) => GridSpec::Quantiles(seq_length_out(0.0, 1.0, n, "quantile grid")?),
            (true, None) => GridSpec::Quantiles((1..=9).map(|k| k as f64 / 10.0).collect()),
            (false, Some(n)) => GridSpec::Uniform(n),
            (false, None) => GridSpec::Uniform(default_grid_points(values)?),
        })
    }
}

/// Result from Partial Dependence Plot computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdpResult {
    /// Grid values at which PDP was computed
    pub grid_values: Vec<f64>,

    /// Partial dependence values (averaged predictions)
    pub pdp_values: Vec<f64>,

    /// Feature index used for PDP
    pub feature_index: usize,

    /// Feature name (if provided)
    pub feature_name: Option<String>,

    /// Number of observations used
    pub n_obs: usize,

    /// Whether the PDP is centered
    pub centered: bool,

    /// Standard deviation of predictions at each grid point (optional)
    pub pdp_std: Option<Vec<f64>>,

    /// Model the prediction function came from ("GBM", "CART"); None for a caller's closure.
    #[serde(default)]
    pub model: Option<String>,
}

impl std::fmt::Display for PdpResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Partial Dependence Plot Results")?;
        writeln!(f, "================================")?;

        if let Some(ref name) = self.feature_name {
            writeln!(f, "Feature: {} (index {})", name, self.feature_index)?;
        } else {
            writeln!(f, "Feature index: {}", self.feature_index)?;
        }

        if let Some(ref model) = self.model {
            writeln!(f, "Model: {model}")?;
        }
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Grid points: {}", self.grid_values.len())?;
        writeln!(f, "Centered: {}", self.centered)?;
        writeln!(f)?;

        // Show summary statistics
        let pdp_min = self
            .pdp_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let pdp_max = self
            .pdp_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let pdp_range = pdp_max - pdp_min;

        writeln!(f, "PDP Range:")?;
        writeln!(f, "  Min: {:.4}", pdp_min)?;
        writeln!(f, "  Max: {:.4}", pdp_max)?;
        writeln!(f, "  Range: {:.4}", pdp_range)?;
        writeln!(f)?;

        // Show first few grid points
        let n_show = self.grid_values.len().min(10);
        writeln!(f, "Grid Values (first {}):", n_show)?;
        for i in 0..n_show {
            write!(
                f,
                "  x={:.4} -> PDP={:.4}",
                self.grid_values[i], self.pdp_values[i]
            )?;
            if let Some(ref std) = self.pdp_std {
                write!(f, " (std={:.4})", std[i])?;
            }
            writeln!(f)?;
        }

        if self.grid_values.len() > 10 {
            writeln!(f, "  ... ({} more points)", self.grid_values.len() - 10)?;
        }

        Ok(())
    }
}

/// Result from 2D Partial Dependence Plot computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pdp2dResult {
    /// Grid values for first feature
    pub grid_values_1: Vec<f64>,

    /// Grid values for second feature
    pub grid_values_2: Vec<f64>,

    /// Partial dependence matrix (grid_1 x grid_2)
    ///
    /// Element [i][j] is the PDP value at (grid_values_1[i], grid_values_2[j])
    pub pdp_values: Vec<Vec<f64>>,

    /// First feature index
    pub feature_index_1: usize,

    /// Second feature index
    pub feature_index_2: usize,

    /// Feature names (if provided)
    pub feature_names: Option<(String, String)>,

    /// Number of observations used
    pub n_obs: usize,

    /// Whether the PDP is centered
    pub centered: bool,
}

impl std::fmt::Display for Pdp2dResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "2D Partial Dependence Plot Results")?;
        writeln!(f, "===================================")?;

        if let Some(ref names) = self.feature_names {
            writeln!(
                f,
                "Features: {} (idx {}) x {} (idx {})",
                names.0, self.feature_index_1, names.1, self.feature_index_2
            )?;
        } else {
            writeln!(
                f,
                "Feature indices: {} x {}",
                self.feature_index_1, self.feature_index_2
            )?;
        }

        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(
            f,
            "Grid: {} x {} = {} points",
            self.grid_values_1.len(),
            self.grid_values_2.len(),
            self.grid_values_1.len() * self.grid_values_2.len()
        )?;
        writeln!(f, "Centered: {}", self.centered)?;
        writeln!(f)?;

        // Find min/max PDP values
        let mut pdp_min = f64::INFINITY;
        let mut pdp_max = f64::NEG_INFINITY;
        for row in &self.pdp_values {
            for &val in row {
                pdp_min = pdp_min.min(val);
                pdp_max = pdp_max.max(val);
            }
        }

        writeln!(f, "PDP Range: [{:.4}, {:.4}]", pdp_min, pdp_max)?;

        Ok(())
    }
}

/// Compute Partial Dependence Plot for a single feature.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `feature_index` - Index of the feature to compute PDP for
/// * `predict_fn` - Function that takes feature matrix and returns predictions
/// * `config` - PDP configuration
///
/// # Returns
///
/// PdpResult containing grid values and averaged predictions
///
/// # Example
///
/// ```rust
/// use p2a_core::ml::partial_dependence;
/// use ndarray::Array2;
///
/// // Row i is [3i/100, (3i+1)/100, (3i+2)/100], so x0 runs 0.00..=2.97
/// // and the mean of x1 is 1.495.
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64 / 100.0).collect()).unwrap();
///
/// // Simple linear prediction function for example
/// let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
///     x.rows().into_iter().map(|row| row[0] + 2.0 * row[1]).collect()
/// };
///
/// // `&Default::default()` is a `PdpConfig`: pdp's min(distinct values, 51) uniform points, uncentered.
/// let result = partial_dependence(&data, 0, predict_fn, &Default::default()).unwrap();
///
/// assert_eq!(result.grid_values.len(), 51);
/// assert_eq!(result.n_obs, 100);
///
/// // For a linear predictor the PDP is exact: PDP(g) = g + 2 * mean(x1) = g + 2.99,
/// // i.e. a line through 2.99 with slope equal to the coefficient on x0.
/// assert!((result.pdp_values[0] - 2.99).abs() < 1e-9); // grid starts at x0 = 0
/// let slope = (result.pdp_values[50] - result.pdp_values[0])
///     / (result.grid_values[50] - result.grid_values[0]);
/// assert!((slope - 1.0).abs() < 1e-9);
/// ```
///
/// # References
///
/// - Friedman (2001), Equation 10.51
/// - R pdp package: `pdp::partial()`
pub fn partial_dependence<F>(
    data: &Array2<f64>,
    feature_index: usize,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<PdpResult>
where
    F: Fn(&Array2<f64>) -> Vec<f64>,
{
    try_partial_dependence(
        data,
        feature_index,
        |m: &Array2<f64>| Ok(predict_fn(m)),
        config,
    )
}

/// [`partial_dependence`] for a prediction function that can fail: its errors are returned unchanged, at the
/// first grid point that fails.
pub fn try_partial_dependence<F>(
    data: &Array2<f64>,
    feature_index: usize,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<PdpResult>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
{
    pd_1d(data.view(), feature_index, predict_fn, config)
}

/// Compute Partial Dependence Plot with feature name.
///
/// Same as `partial_dependence` but accepts a feature name for display.
pub fn partial_dependence_named<F>(
    data: &Array2<f64>,
    feature_index: usize,
    feature_name: &str,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<PdpResult>
where
    F: Fn(&Array2<f64>) -> Vec<f64>,
{
    let mut result = partial_dependence(data, feature_index, predict_fn, config)?;
    result.feature_name = Some(feature_name.to_string());
    Ok(result)
}

/// Compute 2D Partial Dependence Plot for feature interactions.
///
/// Shows the joint marginal effect of two features on predictions.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `feature_index_1` - Index of the first feature
/// * `feature_index_2` - Index of the second feature
/// * `predict_fn` - Function that takes feature matrix and returns predictions
/// * `config` - PDP configuration: the first feature's grid as in [`partial_dependence`], the second with
///   `grid_resolution_2`
///
/// # Returns
///
/// Pdp2dResult containing grid values and the PDP matrix: `pdp_values[i][j]` is the PD at
/// `(grid_values_1[i], grid_values_2[j])`
///
/// # References
///
/// - Friedman (2001), Section 10.13.2
/// - R pdp package: `pdp::partial()` with multiple features
pub fn partial_dependence_2d<F>(
    data: &Array2<f64>,
    feature_index_1: usize,
    feature_index_2: usize,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<Pdp2dResult>
where
    F: Fn(&Array2<f64>) -> Vec<f64>,
{
    try_partial_dependence_2d(
        data,
        feature_index_1,
        feature_index_2,
        |m: &Array2<f64>| Ok(predict_fn(m)),
        config,
    )
}

/// [`partial_dependence_2d`] for a prediction function that can fail: its errors are returned unchanged, at the
/// first grid point that fails.
pub fn try_partial_dependence_2d<F>(
    data: &Array2<f64>,
    feature_index_1: usize,
    feature_index_2: usize,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<Pdp2dResult>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "2D PDP computation requires data".to_string(),
        });
    }

    if feature_index_1 >= n_features || feature_index_2 >= n_features {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Feature indices ({}, {}) out of range (data has {} features)",
                feature_index_1, feature_index_2, n_features
            ),
        });
    }

    if feature_index_1 == feature_index_2 {
        return Err(EconError::InvalidSpecification {
            message: "2D PDP requires two different features".to_string(),
        });
    }

    let values_1 = data.column(feature_index_1).to_vec();
    let values_2 = data.column(feature_index_2).to_vec();
    let spec_1 = config.grid_spec(&values_1)?;
    let spec_2 = config.grid_spec_2(&values_2)?;
    let grid_values_1 = feature_grid(&values_1, &spec_1, &format!("feature {feature_index_1}"))?;
    let grid_values_2 = feature_grid(&values_2, &spec_2, &format!("feature {feature_index_2}"))?;

    let n_grid_1 = grid_values_1.len();
    let n_grid_2 = grid_values_2.len();
    let total = checked_len(&[n_grid_1, n_grid_2], "2D partial dependence")?;

    let mut pdp_values = try_with_capacity::<Vec<f64>>(n_grid_1, "2D partial dependence")?;
    for _ in 0..n_grid_1 {
        pdp_values.push(try_vec(n_grid_2, 0.0, "2D partial dependence")?);
    }

    evaluate_grid(
        data.view(),
        &[feature_index_1, feature_index_2],
        &[grid_values_1.as_slice(), grid_values_2.as_slice()],
        predict_fn,
        |idx, predictions| {
            pdp_values[idx[0]][idx[1]] = predictions.iter().sum::<f64>() / n_samples as f64;
            Ok(())
        },
    )?;

    // Center if requested
    let centered = config.center;
    if centered {
        let sum: f64 = pdp_values.iter().flat_map(|row| row.iter()).sum();
        let mean = sum / total as f64;
        for row in &mut pdp_values {
            for val in row {
                *val -= mean;
            }
        }
    }

    Ok(Pdp2dResult {
        grid_values_1,
        grid_values_2,
        pdp_values,
        feature_index_1,
        feature_index_2,
        feature_names: None,
        n_obs: n_samples,
        centered,
    })
}

/// Compute 2D Partial Dependence Plot with feature names.
pub fn partial_dependence_2d_named<F>(
    data: &Array2<f64>,
    feature_index_1: usize,
    feature_index_2: usize,
    feature_name_1: &str,
    feature_name_2: &str,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<Pdp2dResult>
where
    F: Fn(&Array2<f64>) -> Vec<f64>,
{
    let mut result =
        partial_dependence_2d(data, feature_index_1, feature_index_2, predict_fn, config)?;
    result.feature_names = Some((feature_name_1.to_string(), feature_name_2.to_string()));
    Ok(result)
}

// ---------------------------------------------------------------------------
// The engine: one grid function, one evaluation loop.
// ---------------------------------------------------------------------------

/// `seq(from, to, length.out = n)` for doubles, as R 4.3's `seq.default`: the endpoints are exact, and a range
/// whose difference overflows is scaled by 4 (no Inf, no NaN).
fn seq_length_out(from: f64, to: f64, n: usize, what: &str) -> EconResult<Vec<f64>> {
    let mut out = try_with_capacity::<f64>(n, what)?;
    match n {
        0 => {}
        1 => out.push(from),
        2 => {
            out.push(from);
            out.push(to);
        }
        _ if from == to => out.resize(n, from),
        _ => {
            let n1 = (n - 1) as f64;
            let del = to - from;
            if del.is_finite() {
                let by = del / n1;
                out.push(from);
                out.extend((1..n - 1).map(|i| from + i as f64 * by));
                out.push(to);
            } else {
                let (f4, t4) = (from / 4.0, to / 4.0);
                let by = (t4 - f4) / n1;
                out.push(f4 * 4.0);
                out.extend((1..n - 1).map(|i| (f4 + i as f64 * by) * 4.0));
                out.push(t4 * 4.0);
            }
        }
    }
    Ok(out)
}

/// pdp's default grid size for one feature: `min(length(unique(x)), 51)`, counted over the non-missing values,
/// at least 1 (a feature with no finite value is then refused by [`feature_grid`]).
pub(crate) fn default_grid_points(values: &[f64]) -> EconResult<usize> {
    let mut distinct = try_with_capacity::<f64>(values.len(), "partial-dependence grid")?;
    distinct.extend(values.iter().copied().filter(|v| !v.is_nan()));
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    Ok(distinct.len().clamp(1, DEFAULT_MAX_GRID_POINTS))
}

/// Refuse infinite feature values and a feature without any finite value (pdp: `na.rm = TRUE`).
fn check_feature_values(values: &[f64], feature: &str) -> EconResult<()> {
    if values.iter().any(|v| v.is_infinite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "{feature} contains infinite values; partial dependence and ICE need finite feature values"
            ),
        });
    }
    if values.iter().all(|v| v.is_nan()) {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: format!("{feature} has no non-missing values"),
        });
    }
    Ok(())
}

/// The grid of one feature, ascending, each value once (exact equality).
///
/// NaN in `values` is ignored (pdp's `na.rm = TRUE`). `feature` names the feature in error messages.
///
/// # Errors
///
/// `InvalidSpecification` for a `Uniform(0)` grid, an empty or out-of-range probability list, an empty or
/// non-finite list of user values, and infinite feature values; `InsufficientData` for a feature with no
/// non-missing value; `ResourceLimit` for a grid that cannot be allocated.
///
/// # Example
///
/// ```
/// use p2a_core::ml::pdp::{GridSpec, feature_grid};
/// // pdp::partial(grid.resolution = 5): seq(min, max, length.out = 5) over the non-missing values
/// let x = [3.0, f64::NAN, 1.0, 2.0, 5.0];
/// assert_eq!(feature_grid(&x, &GridSpec::Uniform(5), "x").unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// // quantile(x, c(0.25, 0.5), type = 7, na.rm = TRUE)
/// assert_eq!(feature_grid(&x, &GridSpec::Quantiles(vec![0.25, 0.5]), "x").unwrap(), vec![1.75, 2.5]);
/// ```
pub fn feature_grid(values: &[f64], spec: &GridSpec, feature: &str) -> EconResult<Vec<f64>> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    let mut grid = match spec {
        GridSpec::Values(user) => {
            if user.is_empty() {
                return Err(invalid(format!(
                    "{feature}: grid_values must contain at least one value"
                )));
            }
            if let Some(bad) = user.iter().find(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "{feature}: grid_values must be finite, got {bad}"
                )));
            }
            let mut grid = try_with_capacity::<f64>(user.len(), "partial-dependence grid")?;
            grid.extend_from_slice(user);
            grid
        }
        GridSpec::Uniform(n) => {
            if *n == 0 {
                return Err(invalid(format!(
                    "{feature}: grid_resolution must be at least 1, got 0"
                )));
            }
            check_feature_values(values, feature)?;
            let (min, max) = values
                .iter()
                .filter(|v| !v.is_nan())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            seq_length_out(min, max, *n, "partial-dependence grid")?
        }
        GridSpec::Quantiles(probs) => {
            if probs.is_empty() {
                return Err(invalid(format!(
                    "{feature}: the quantile grid needs at least one probability"
                )));
            }
            if let Some(bad) = probs.iter().find(|p| !(**p >= 0.0 && **p <= 1.0)) {
                return Err(invalid(format!(
                    "{feature}: quantile probabilities must be in [0, 1], got {bad}"
                )));
            }
            check_feature_values(values, feature)?;
            let mut sorted = try_with_capacity::<f64>(values.len(), "partial-dependence grid")?;
            sorted.extend(values.iter().copied().filter(|v| !v.is_nan()));
            sorted.sort_by(f64::total_cmp);
            let mut grid = try_with_capacity::<f64>(probs.len(), "partial-dependence grid")?;
            grid.extend(
                probs
                    .iter()
                    .map(|&p| quantile_sorted(&sorted, p, QuantileType::T7)),
            );
            grid
        }
    };
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    Ok(grid)
}

/// Evaluate `predict` on `data` with `features[k]` set to `grids[k][idx[k]]` for every point of the Cartesian
/// product (row-major: the last feature varies fastest), calling `sink(&idx, &predictions)` once per point in
/// that order.
///
/// A prediction vector whose length is not `data.nrows()` is `EconError::Computation`; `predict`'s and `sink`'s
/// errors are returned unchanged. `resource::checkpoint()` runs before every point.
pub(crate) fn evaluate_grid<F, S>(
    data: ArrayView2<'_, f64>,
    features: &[usize],
    grids: &[&[f64]],
    predict: F,
    mut sink: S,
) -> EconResult<()>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
    S: FnMut(&[usize], &[f64]) -> EconResult<()>,
{
    if features.is_empty() || features.len() != grids.len() || grids.iter().any(|g| g.is_empty()) {
        return Err(EconError::InvalidSpecification {
            message: "partial dependence needs one non-empty grid per feature".to_string(),
        });
    }
    let n = data.nrows();
    let dims: Vec<usize> = grids.iter().map(|g| g.len()).collect();
    let total = checked_len(&dims, "partial-dependence grid")?;
    let mut work = data.to_owned();
    let mut idx = vec![0usize; features.len()];
    for point in 0..total {
        checkpoint()?;
        let mut rem = point;
        for k in (0..features.len()).rev() {
            idx[k] = rem % dims[k];
            rem /= dims[k];
        }
        for (k, &f) in features.iter().enumerate() {
            work.column_mut(f).fill(grids[k][idx[k]]);
        }
        let predictions = predict(&work)?;
        if predictions.len() != n {
            return Err(EconError::Computation(format!(
                "prediction function returned {} values for {} rows (grid point {} of {})",
                predictions.len(),
                n,
                point + 1,
                total
            )));
        }
        sink(&idx, &predictions)?;
    }
    Ok(())
}

/// The 1-D partial dependence of every entry point.
pub(crate) fn pd_1d<F>(
    data: ArrayView2<'_, f64>,
    feature_index: usize,
    predict: F,
    config: &PdpConfig,
) -> EconResult<PdpResult>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "PDP computation requires data".to_string(),
        });
    }

    if feature_index >= n_features {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Feature index {} out of range (data has {} features)",
                feature_index, n_features
            ),
        });
    }

    let values = data.column(feature_index).to_vec();
    let spec = config.grid_spec(&values)?;
    let grid_values = feature_grid(&values, &spec, &format!("feature {feature_index}"))?;

    let mut pdp_values = try_with_capacity::<f64>(grid_values.len(), "partial dependence")?;
    let mut pdp_std = try_with_capacity::<f64>(grid_values.len(), "partial dependence")?;

    // Average predictions (Monte Carlo integration over X_c; Friedman 2001, Section 8.2)
    evaluate_grid(
        data,
        &[feature_index],
        &[grid_values.as_slice()],
        predict,
        |_, predictions| {
            let mean = predictions.iter().sum::<f64>() / n_samples as f64;
            let variance =
                predictions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / n_samples as f64;
            pdp_values.push(mean);
            pdp_std.push(variance.sqrt());
            Ok(())
        },
    )?;

    // Center if requested (c-PDP)
    let centered = config.center;
    if centered {
        let pdp_mean: f64 = pdp_values.iter().sum::<f64>() / pdp_values.len() as f64;
        for val in &mut pdp_values {
            *val -= pdp_mean;
        }
    }

    Ok(PdpResult {
        grid_values,
        pdp_values,
        feature_index,
        feature_name: None,
        n_obs: n_samples,
        centered,
        pdp_std: Some(pdp_std),
        model: None,
    })
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_pdp_basic() {
        // Simple linear relationship: y = x0 + 2*x1
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];

        let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
            x.rows()
                .into_iter()
                .map(|row| row[0] + 2.0 * row[1])
                .collect()
        };

        let config = PdpConfig {
            grid_resolution: Some(5),
            use_quantiles: false,
            ..Default::default()
        };

        let result = partial_dependence(&data, 0, predict_fn, &config).unwrap();

        assert_eq!(result.feature_index, 0);
        assert_eq!(result.n_obs, 5);
        assert!(!result.centered);

        // Grid should span from 1 to 5
        assert_eq!(result.grid_values.len(), 5);
        assert!((result.grid_values[0] - 1.0).abs() < 1e-10);
        assert!((result.grid_values[4] - 5.0).abs() < 1e-10);

        // PDP for x0 should show linear increase
        // At x0=1, predictions are [1+2*1, 1+2*2, ..., 1+2*5] -> mean = 1 + 2*3 = 7
        // At x0=5, predictions are [5+2*1, 5+2*2, ..., 5+2*5] -> mean = 5 + 2*3 = 11
        assert!((result.pdp_values[0] - 7.0).abs() < 1e-10);
        assert!((result.pdp_values[4] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_pdp_centered() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];

        let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
            x.rows()
                .into_iter()
                .map(|row| row[0] + 2.0 * row[1])
                .collect()
        };

        let config = PdpConfig {
            grid_resolution: Some(5),
            use_quantiles: false,
            center: true,
            ..Default::default()
        };

        let result = partial_dependence(&data, 0, predict_fn, &config).unwrap();

        assert!(result.centered);

        // Centered PDP should have mean 0
        let mean: f64 = result.pdp_values.iter().sum::<f64>() / result.pdp_values.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_pdp_quantiles() {
        // Data with non-uniform distribution
        let data = array![[1.0, 1.0], [1.0, 2.0], [2.0, 3.0], [10.0, 4.0], [10.0, 5.0],];

        let predict_fn =
            |x: &Array2<f64>| -> Vec<f64> { x.rows().into_iter().map(|row| row[0]).collect() };

        let config = PdpConfig {
            grid_resolution: Some(3),
            use_quantiles: true,
            ..Default::default()
        };

        let result = partial_dependence(&data, 0, predict_fn, &config).unwrap();

        // With quantiles, grid should reflect data distribution
        // Q0 = 1, Q0.5 = 2, Q1 = 10
        assert!((result.grid_values[0] - 1.0).abs() < 1e-10);
        assert!((result.grid_values[result.grid_values.len() - 1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_pdp_2d() {
        let data = array![
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0],
        ];

        let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
            x.rows()
                .into_iter()
                .map(|row| row[0] * row[1]) // Interaction effect
                .collect()
        };

        let config = PdpConfig {
            grid_resolution: Some(3),
            use_quantiles: false,
            ..Default::default()
        };

        let result = partial_dependence_2d(&data, 0, 1, predict_fn, &config).unwrap();

        assert_eq!(result.feature_index_1, 0);
        assert_eq!(result.feature_index_2, 1);
        assert_eq!(result.n_obs, 4);

        // Check dimensions
        assert_eq!(result.grid_values_1.len(), 3);
        assert_eq!(result.grid_values_2.len(), 3);
        assert_eq!(result.pdp_values.len(), 3);
        assert_eq!(result.pdp_values[0].len(), 3);
    }

    #[test]
    fn test_pdp_errors() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        let predict_fn =
            |x: &Array2<f64>| -> Vec<f64> { x.rows().into_iter().map(|row| row[0]).collect() };

        let config = PdpConfig::default();

        // Out of range feature index
        let result = partial_dependence(&data, 5, predict_fn, &config);
        assert!(result.is_err());

        // Empty data
        let empty_data: Array2<f64> = Array2::zeros((0, 2));
        let result = partial_dependence(&empty_data, 0, predict_fn, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_grid_uniform_rule() {
        let grid = feature_grid(&[0.0, 1.0, 2.0, 3.0, 4.0], &GridSpec::Uniform(5), "x").unwrap();
        assert_eq!(grid, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_display() {
        let result = PdpResult {
            grid_values: vec![1.0, 2.0, 3.0],
            pdp_values: vec![2.0, 4.0, 6.0],
            feature_index: 0,
            feature_name: Some("feature_x".to_string()),
            n_obs: 100,
            centered: false,
            pdp_std: Some(vec![0.1, 0.2, 0.3]),
            model: None,
        };

        let output = format!("{}", result);
        assert!(output.contains("Partial Dependence Plot"));
        assert!(output.contains("feature_x"));
        assert!(output.contains("100"));
        assert!(!output.contains("Model:"));

        let with_model = PdpResult {
            model: Some("GBM".to_string()),
            ..result
        };
        assert!(format!("{with_model}").contains("Model: GBM"));
    }

    fn lm_closure(b: &[f64]) -> impl Fn(&Array2<f64>) -> Vec<f64> + Copy + '_ {
        move |m: &Array2<f64>| m.rows().into_iter().map(|r| pdp_lm_value(b, r)).collect()
    }

    /// pdp::partial's grid rules on pdp_lm: grids bit-identical to R (they use R's own operations, seq.default and
    /// quantile.default type 7), PD values at PD_TOL.
    #[test]
    fn test_validate_pdp_grid_rules() {
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let b = pdp_lm_coef(&case);
        let f = lm_closure(&b);
        // pdp's default: min(length(unique(x)), 51) uniform points
        let default = PdpConfig::default();
        let d1 = partial_dependence(&x, 0, f, &default).unwrap();
        let d3 = partial_dependence(&x, 2, f, &default).unwrap();
        let uniform = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(25),
            ..Default::default()
        };
        let pu = partial_dependence(&x, 0, f, &uniform).unwrap();
        let q = PdpConfig {
            use_quantiles: true,
            quantiles: Some(case.vector("pdp_default_probs")),
            ..Default::default()
        };
        let pq = partial_dependence(&x, 0, f, &q).unwrap();
        let q20 = PdpConfig {
            use_quantiles: true,
            grid_resolution: Some(20),
            ..Default::default()
        };
        let pd = partial_dependence(&x, 0, f, &q20).unwrap();
        // deviation ML-137: x3 has ties; pdp repeats them, p2a lists each once
        let pt = partial_dependence(&x, 2, f, &q).unwrap();
        assert_eq!(case.vector("pd_ties_grid_r").len(), 9);
        Check::new(&case)
            .vector("pd_default_grid", &d1.grid_values, Tol::exact())
            .vector("pd_default_yhat", &d1.pdp_values, PD_TOL)
            .vector("pd_default_x3_grid", &d3.grid_values, Tol::exact())
            .vector("pd_default_x3_yhat", &d3.pdp_values, PD_TOL)
            .vector("pd_uniform_grid", &pu.grid_values, Tol::exact())
            .vector("pd_uniform_yhat", &pu.pdp_values, PD_TOL)
            .vector("pd_quantile_grid", &pq.grid_values, Tol::exact())
            .vector("pd_quantile_yhat", &pq.pdp_values, PD_TOL)
            .vector("pd_quantile20_grid", &pd.grid_values, Tol::exact())
            .vector("pd_quantile20_yhat", &pd.pdp_values, PD_TOL)
            .vector("pd_ties_grid_unique", &pt.grid_values, Tol::exact())
            .vector("pd_ties_yhat_unique", &pt.pdp_values, PD_TOL)
            .finish();
    }

    /// Two predictors on 5 x 4 uniform points; grid_resolution_2 sizes the second grid (WT-18 doc).
    #[test]
    fn test_validate_pdp_lm_2d() {
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let b = pdp_lm_coef(&case);
        let config = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(5),
            grid_resolution_2: Some(4),
            ..Default::default()
        };
        let r = partial_dependence_2d(&x, 0, 1, lm_closure(&b), &config).unwrap();
        let flat: Vec<f64> = r.pdp_values.iter().flatten().copied().collect(); // [i][j], x1 slowest
        Check::new(&case)
            .vector("pd2_grid1", &r.grid_values_1, Tol::exact())
            .vector("pd2_grid2", &r.grid_values_2, Tol::exact())
            .vector("pd2_yhat", &flat, PD_TOL)
            .finish();
    }

    /// A feature on the 1e-12 scale keeps its 9 distinct grid points (no absolute tolerance merges them).
    #[test]
    fn test_validate_pdp_small_scale() {
        let (case, ds) = crate::test_support::load_case("pdp_small_scale");
        let x = matrix(&ds, &["x", "z"]);
        let b = case.by_names("coef", &["(Intercept)", "x", "z"]);
        let f = |m: &Array2<f64>| -> Vec<f64> {
            m.rows()
                .into_iter()
                .map(|r| small_scale_value(&b, r))
                .collect()
        };
        let q = PdpConfig {
            use_quantiles: true,
            quantiles: Some((1..=9).map(|k| k as f64 / 10.0).collect()), // R's 1:9/10, the same divisions
            ..Default::default()
        };
        let u = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(9),
            ..Default::default()
        };
        let pq = partial_dependence(&x, 0, f, &q).unwrap();
        let pu = partial_dependence(&x, 0, f, &u).unwrap();
        Check::new(&case)
            .vector("pd_quantile_grid", &pq.grid_values, Tol::exact())
            .vector("pd_quantile_yhat", &pq.pdp_values, PD_TOL)
            .vector("pd_uniform_grid", &pu.grid_values, Tol::exact())
            .vector("pd_uniform_yhat", &pu.pdp_values, PD_TOL)
            .finish();
    }

    fn noisy_rows() -> Array2<f64> {
        Array2::from_shape_fn((10, 2), |(i, j)| {
            i as f64 * 0.7 + j as f64 * ((i * 3) % 5) as f64 * 0.1 + 0.01 * (i * i) as f64
        })
    }

    fn linear(m: &Array2<f64>) -> Vec<f64> {
        m.rows().into_iter().map(|r| r[0] + 2.0 * r[1]).collect()
    }

    /// WT-18: a prediction vector of the wrong length is an error in 1-D and 2-D (2-D returned a mean over the
    /// wrong count: 9 or 11 values divided by 10).
    #[test]
    fn test_pdp_2d_rejects_wrong_prediction_length() {
        let x = noisy_rows();
        let short = |m: &Array2<f64>| -> Vec<f64> { linear(m).into_iter().skip(1).collect() };
        let long = |m: &Array2<f64>| -> Vec<f64> {
            let mut v = linear(m);
            v.push(0.0);
            v
        };
        let config = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(3),
            ..Default::default()
        };
        let is_len_err = |e: &EconError, got: usize| matches!(e, EconError::Computation(m) if m.contains(&format!("returned {got} values for 10 rows")));
        let e = partial_dependence_2d(&x, 0, 1, short, &config).unwrap_err();
        assert!(is_len_err(&e, 9), "{e}");
        let e = partial_dependence_2d(&x, 0, 1, long, &config).unwrap_err();
        assert!(is_len_err(&e, 11), "{e}");
        let e = partial_dependence(&x, 0, short, &config).unwrap_err();
        assert!(is_len_err(&e, 9), "{e}");
    }

    /// Grid sizes and probabilities are validated, not silently changed.
    #[test]
    fn test_pdp_config_rejects_invalid_grids() {
        let x = noisy_rows();
        let spec_msg = |c: &PdpConfig| match partial_dependence(&x, 0, linear, c) {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        let c = PdpConfig {
            grid_resolution: Some(0), // was silently 2 points
            ..Default::default()
        };
        assert!(spec_msg(&c).contains("grid_resolution must be at least 1, got 0"));
        let c = PdpConfig {
            quantiles: Some(vec![0.2, 1.5]), // was clamped to 1
            ..Default::default()
        };
        assert!(spec_msg(&c).contains("quantile probabilities must be in [0, 1], got 1.5"));
        let c = PdpConfig {
            quantiles: Some(vec![]), // was an empty PD
            ..Default::default()
        };
        assert!(spec_msg(&c).contains("needs at least one probability"));
        let c = PdpConfig {
            grid_resolution_2: Some(0), // was silently 2 points
            ..Default::default()
        };
        let e = partial_dependence_2d(&x, 0, 1, linear, &c).unwrap_err();
        assert!(
            e.to_string()
                .contains("grid_resolution_2 must be at least 1, got 0"),
            "{e}"
        );
        // one point is pdp's seq(min, max, length.out = 1) = min
        let c = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(1),
            ..Default::default()
        };
        assert_eq!(
            partial_dependence(&x, 0, linear, &c).unwrap().grid_values,
            vec![x[[0, 0]]]
        );
    }

    /// An impossible grid is a ResourceLimit error.
    #[test]
    fn test_pdp_huge_grid_is_a_resource_error() {
        let x = noisy_rows();
        for use_quantiles in [false, true] {
            let c = PdpConfig {
                grid_resolution: Some(usize::MAX),
                use_quantiles,
                ..Default::default()
            };
            assert!(matches!(
                partial_dependence(&x, 0, linear, &c),
                Err(EconError::ResourceLimit(_))
            ));
        }
    }

    /// A cancelled budget stops the grid loop (was: ran to the end).
    #[test]
    fn test_pdp_stops_when_cancelled() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let x = noisy_rows();
        let token = CancelToken::new();
        token.cancel();
        let r = with_budget(Budget::new(None, token), || {
            partial_dependence(&x, 0, linear, &PdpConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
    }

    /// The default is pdp's grid, min(length(unique(x)), 51) uniform points per feature; quantile grids use pdp's
    /// default probs 1:9/10 and R's seq(0, 1, length.out = n), bit for bit.
    #[test]
    fn test_validate_pdp_config_grid_spec() {
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let x1 = x.column(0).to_vec();
        let x3 = x.column(2).to_vec();
        let default = PdpConfig::default();
        let n1 = case.vector("pd_default_grid").len();
        let n3 = case.vector("pd_default_x3_grid").len();
        assert_eq!(default.grid_spec(&x1).unwrap(), GridSpec::Uniform(n1));
        assert_eq!(default.grid_spec(&x3).unwrap(), GridSpec::Uniform(n3));
        assert_eq!(default.grid_spec_2(&x3).unwrap(), GridSpec::Uniform(n3));
        let q = PdpConfig {
            use_quantiles: true,
            ..Default::default()
        };
        assert_eq!(
            q.grid_spec(&x1).unwrap(),
            GridSpec::Quantiles(case.vector("pdp_default_probs"))
        );
        let q20 = PdpConfig {
            use_quantiles: true,
            grid_resolution: Some(20),
            ..Default::default()
        };
        assert_eq!(
            q20.grid_spec(&x1).unwrap(),
            GridSpec::Quantiles(case.vector("quantile20_probs"))
        );
        let c = PdpConfig {
            grid_resolution: Some(20),
            grid_resolution_2: Some(4),
            ..Default::default()
        };
        assert_eq!(c.grid_spec(&x1).unwrap(), GridSpec::Uniform(20));
        assert_eq!(c.grid_spec_2(&x3).unwrap(), GridSpec::Uniform(4));
        let c = PdpConfig {
            grid_resolution: Some(20),
            ..Default::default()
        };
        assert_eq!(c.grid_spec_2(&x3).unwrap(), GridSpec::Uniform(20));
        let c = PdpConfig {
            quantiles: Some(vec![0.5]),
            ..c
        };
        assert_eq!(c.grid_spec(&x1).unwrap(), GridSpec::Quantiles(vec![0.5]));
    }

    #[test]
    fn test_feature_grid_rules() {
        let x = [3.0, f64::NAN, 1.0, 2.0, 5.0];
        // NaN ignored
        assert_eq!(
            feature_grid(&x, &GridSpec::Uniform(5), "x").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        // sorted
        assert_eq!(
            feature_grid(&x, &GridSpec::Quantiles(vec![0.5, 0.25]), "x").unwrap(),
            vec![1.75, 2.5]
        );
        // rep(2, 10), once
        assert_eq!(
            feature_grid(&[2.0; 4], &GridSpec::Uniform(10), "x").unwrap(),
            vec![2.0]
        );
        assert_eq!(
            feature_grid(&[], &GridSpec::Values(vec![3.0, 1.0, 1.0, 2.0]), "x").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        // R's overflow branch of seq: no Inf or NaN
        assert_eq!(
            feature_grid(&[-1e308, 1e308], &GridSpec::Uniform(5), "x").unwrap(),
            vec![-1e308, -5e307, 0.0, 5e307, 1e308]
        );
        let spec_err = |v: &[f64], s: GridSpec| match feature_grid(v, &s, "feature 3") {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        assert!(
            spec_err(&[1.0, f64::INFINITY], GridSpec::Uniform(3))
                .contains("feature 3 contains infinite values")
        );
        assert!(
            spec_err(&[1.0, f64::INFINITY], GridSpec::Quantiles(vec![0.5]))
                .contains("feature 3 contains infinite values")
        );
        assert!(
            spec_err(&[1.0], GridSpec::Uniform(0)).contains("grid_resolution must be at least 1")
        );
        assert!(
            spec_err(&[1.0], GridSpec::Quantiles(vec![f64::NAN]))
                .contains("must be in [0, 1], got NaN")
        );
        assert!(
            spec_err(&[1.0], GridSpec::Values(vec![]))
                .contains("grid_values must contain at least one value")
        );
        assert!(
            spec_err(&[1.0], GridSpec::Values(vec![1.0, f64::NAN]))
                .contains("grid_values must be finite")
        );
        assert!(matches!(
            feature_grid(&[f64::NAN, f64::NAN], &GridSpec::Uniform(3), "feature 3"),
            Err(EconError::InsufficientData { .. })
        ));
        // the default grid counts non-missing values; a feature with none is refused by the grid, not the count
        let default = PdpConfig::default();
        assert_eq!(default.grid_spec(&x).unwrap(), GridSpec::Uniform(4));
        let empty = default.grid_spec(&[f64::NAN]).unwrap();
        assert!(matches!(
            feature_grid(&[f64::NAN], &empty, "feature 3"),
            Err(EconError::InsufficientData { .. })
        ));
    }

    /// Errors of a fallible model are returned unchanged, at the first failing grid point.
    #[test]
    fn test_pdp_prediction_error_propagates() {
        let x = noisy_rows();
        let calls = std::cell::Cell::new(0);
        let failing = |m: &Array2<f64>| -> EconResult<Vec<f64>> {
            calls.set(calls.get() + 1);
            if calls.get() == 3 {
                Err(EconError::Computation("model failed".into()))
            } else {
                Ok(linear(m))
            }
        };
        let c = PdpConfig {
            grid_resolution: Some(5),
            ..Default::default()
        };
        let e = try_partial_dependence(&x, 0, failing, &c).unwrap_err();
        assert!(matches!(&e, EconError::Computation(m) if m == "model failed"));
        assert_eq!(calls.get(), 3);
        calls.set(0);
        let e = try_partial_dependence_2d(&x, 0, 1, failing, &c).unwrap_err();
        assert!(matches!(&e, EconError::Computation(m) if m == "model failed"));
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! Case data and the case fits' prediction functions, shared by the pdp, ice and evaluation tests.
    use crate::data::Dataset;
    use crate::data::columns::{NaPolicy, numeric_column};
    use ndarray::{Array1, Array2, ArrayView1};

    /// PD and ICE values pinned to pdp::partial on closed-form lm predictions: only the summation order differs
    /// from R (|diff| <= 8.9e-16 * max(|R|, 1) on these cases), so 1e-12 is three orders above the noise.
    pub(crate) const PD_TOL: p2a_validation::Tol = p2a_validation::Tol::new(1e-12, 1e-12);

    /// Case columns as an n x k matrix.
    pub(crate) fn matrix(ds: &Dataset, cols: &[&str]) -> Array2<f64> {
        let c: Vec<Vec<f64>> = cols
            .iter()
            .map(|name| numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0)
            .collect();
        Array2::from_shape_fn((c[0].len(), c.len()), |(i, j)| c[j][i])
    }

    pub(crate) fn column(ds: &Dataset, name: &str) -> Array1<f64> {
        Array1::from(numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0)
    }

    /// The pdp_lm data, for tests that pin properties and no R value.
    pub(crate) fn pdp_lm_data() -> (Array2<f64>, Array1<f64>) {
        let path = p2a_validation::repo_path("validation/datasets/generated/pdp_lm.csv");
        let df = crate::data::DataLoader::load_csv(&path).unwrap();
        let ds = Dataset::new(df);
        (matrix(&ds, &["x1", "x2", "x3"]), column(&ds, "y"))
    }

    /// pdp_lm: lm(y ~ x1 * x2 + I(x1^2) + x3), coefficients in R's order.
    pub(crate) fn pdp_lm_coef(case: &p2a_validation::Case) -> Vec<f64> {
        case.by_names(
            "coef",
            &["(Intercept)", "x1", "x2", "I(x1^2)", "x3", "x1:x2"],
        )
    }

    /// The pdp_lm fit at one row [x1, x2, x3].
    pub(crate) fn pdp_lm_value(b: &[f64], r: ArrayView1<f64>) -> f64 {
        b[0] + b[1] * r[0] + b[2] * r[1] + b[3] * r[0] * r[0] + b[4] * r[2] + b[5] * r[0] * r[1]
    }

    /// pdp_small_scale: lm(y ~ x + z) at one row [x, z].
    pub(crate) fn small_scale_value(b: &[f64], r: ArrayView1<f64>) -> f64 {
        b[0] + b[1] * r[0] + b[2] * r[1]
    }

    /// 40 noisy rows of two features; y depends mostly on column 1, so every fitted tree splits on it first.
    pub(crate) fn two_feature_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| {
            if j == 0 {
                ((i * 7) % 11) as f64 * 0.3 + 0.01 * i as f64
            } else {
                ((i * 13) % 40) as f64 / 10.0
            }
        });
        let y = Array1::from_shape_fn(40, |i| {
            3.0 * x[[i, 1]] + 0.2 * x[[i, 0]] + ((i * 7919) % 17) as f64 * 0.05
        });
        (x, y)
    }
}
