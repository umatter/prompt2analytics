//! Individual Conditional Expectation (ICE) curves for model interpretation.
//!
//! ICE curves visualize how predictions change as a single feature varies,
//! holding all other features constant. Each observation gets its own curve,
//! revealing heterogeneous effects that Partial Dependence Plots (PDP) may obscure.
//!
//! # Algorithm
//!
//! For each observation i and grid point x_s:
//! 1. Create a modified copy where feature j = x_s
//! 2. Predict: ICE_i(x_s) = f(x_s, x_{-j}^i)
//! 3. Repeat across the grid
//!
//! The PDP is simply the mean of all ICE curves: PDP(x_s) = (1/n) * sum_i ICE_i(x_s)
//!
//! # Centered ICE (c-ICE)
//!
//! Centering reveals heterogeneity by subtracting the prediction at the smallest grid value:
//! c-ICE_i(x_s) = ICE_i(x_s) - ICE_i(min grid value)
//!
//! # Grids
//!
//! Grids and centring as `pdp::partial` (pdp 0.8.3); computed by the engine in [`crate::ml::pdp`]. By default
//! `min(length(unique(x)), 51)` uniform points over the sampled rows' non-missing values.
//!
//! # References
//!
//! - Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
//!   "Peeking Inside the Black Box: Visualizing Statistical Learning with Plots
//!   of Individual Conditional Expectation."
//!   Journal of Computational and Graphical Statistics, 24(1), 44-65.
//!   <https://arxiv.org/abs/1309.6392>
//!
//! - R package ICEbox: Goldstein et al. (2015).
//!   <https://cran.r-project.org/package=ICEbox>
//!
//! # Example
//!
//! ```rust
//! use ndarray::{ArrayView2, array};
//! use p2a_core::ml::{IceConfig, compute_ice_curves};
//!
//! // Stands in for a fitted model's prediction function: f(x0, x1) = x0 * x1.
//! // The interaction means the effect of x0 depends on each observation's x1.
//! fn predict(x: ArrayView2<f64>) -> Vec<f64> {
//!     (0..x.nrows()).map(|i| x[[i, 0]] * x[[i, 1]]).collect()
//! }
//!
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
//!
//! // Compute centered ICE curves (c-ICE) for feature 0
//! let config = IceConfig {
//!     feature_index: 0,
//!     grid_resolution: Some(5),
//!     center: true,
//!     ..Default::default()
//! };
//!
//! let result = compute_ice_curves(x.view(), predict, &config).unwrap();
//!
//! // The grid spans the observed range of feature 0: 1.0, 2.0, ..., 5.0
//! assert_eq!(result.n_grid, 5);
//! assert!((result.grid_values[0] - 1.0).abs() < 1e-10);
//! assert!((result.grid_values[4] - 5.0).abs() < 1e-10);
//!
//! // Centering anchors every curve at the first grid point, so each starts at 0.
//! // From there c-ICE_i(x_s) = (x_s - 1) * x1_i, i.e. a slope of x1_i: the curves
//! // fan out instead of running parallel, which is the interaction made visible.
//! assert!(result.ice_curves[[0, 0]].abs() < 1e-10);
//! assert!((result.ice_curves[[0, 4]] - 8.0).abs() < 1e-10); // 4 * x1 = 4 * 2
//! assert!((result.ice_curves[[4, 4]] - 24.0).abs() < 1e-10); // 4 * x1 = 4 * 6
//!
//! // The PDP is the average of the ICE curves (mean x1 = 4), and on its own it
//! // shows none of that spread.
//! assert!((result.pdp_curve[4] - 16.0).abs() < 1e-10);
//!
//! // `ice_spread` quantifies what the PDP hides: max(std) / mean(|PDP|),
//! // here (4 * sqrt(2)) / 8 = 1 / sqrt(2) ~= 0.707, well above the 0.5 at which
//! // the Display impl flags interaction effects.
//! assert!(
//!     (result.ice_spread.heterogeneity_index - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10
//! );
//! ```

use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::pdp::{self, GridSpec};
use crate::errors::{EconError, EconResult};
use crate::resource::{try_vec, try_with_capacity, try_zeros2};
use crate::stats::robust::{QuantileType, quantile_sorted};

/// Configuration for ICE curve computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceConfig {
    /// Index of the feature to compute ICE for (0-indexed).
    pub feature_index: usize,

    /// `seq(min, max, length.out = grid_resolution)` over the feature's non-missing values, at least 1.
    /// `None` (default) is pdp's choice, `min(length(unique(x)), 51)` points.
    pub grid_resolution: Option<usize>,

    /// Subtract each curve's value at the smallest grid value (c-ICE, `pdp::partial(center = TRUE)`).
    /// Default: false.
    pub center: bool,

    /// Fraction of observations to use, in (0, 1]; the grid is built from the sampled rows. Default: 1.0 (all).
    /// Set to e.g. 0.1 to randomly sample 10% of observations.
    pub frac_to_plot: f64,

    /// Random seed for sampling when frac_to_plot < 1.0 (default 42).
    pub seed: Option<u64>,

    /// User grid (pdp's `pred.grid`): overrides `grid_resolution`; non-empty and finite; sorted ascending, each
    /// value once.
    pub grid_values: Option<Vec<f64>>,

    /// Feature name (for display purposes).
    pub feature_name: Option<String>,
}

impl Default for IceConfig {
    fn default() -> Self {
        IceConfig {
            feature_index: 0,
            grid_resolution: None,
            center: false,
            frac_to_plot: 1.0,
            seed: None,
            grid_values: None,
            feature_name: None,
        }
    }
}

impl IceConfig {
    /// The grid specification for a feature with these values: `grid_values` -> `Values`; else
    /// `Uniform(grid_resolution)`, with `min(length(unique(x)), 51)` points when `grid_resolution` is `None`.
    ///
    /// # Errors
    ///
    /// `InvalidSpecification` for empty `grid_values` or `grid_resolution` `Some(0)`.
    pub fn grid_spec(&self, values: &[f64]) -> EconResult<GridSpec> {
        match (&self.grid_values, self.grid_resolution) {
            (Some(grid), _) if grid.is_empty() => Err(EconError::InvalidSpecification {
                message: "grid_values must contain at least one value".to_string(),
            }),
            (Some(grid), _) => Ok(GridSpec::Values(grid.clone())),
            (None, Some(0)) => Err(EconError::InvalidSpecification {
                message: "grid_resolution must be at least 1, got 0".to_string(),
            }),
            (None, Some(n)) => Ok(GridSpec::Uniform(n)),
            (None, None) => Ok(GridSpec::Uniform(pdp::default_grid_points(values)?)),
        }
    }
}

/// Result of ICE curve computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceResult {
    /// Grid values along the feature axis.
    pub grid_values: Vec<f64>,

    /// ICE curves: n_obs x n_grid matrix.
    /// Each row is one observation's ICE curve.
    #[serde(skip)]
    pub ice_curves: Array2<f64>,

    /// Partial Dependence Plot curve (average of all ICE curves).
    pub pdp_curve: Vec<f64>,

    /// Feature index that was analyzed.
    pub feature_index: usize,

    /// Feature name (if provided).
    pub feature_name: Option<String>,

    /// Number of observations used.
    pub n_obs: usize,

    /// Number of grid points.
    pub n_grid: usize,

    /// Whether curves are centered (c-ICE).
    pub centered: bool,

    /// Original feature values for each observation.
    pub original_feature_values: Vec<f64>,

    /// ICE curve data as nested Vec (for JSON serialization).
    pub ice_curves_vec: Vec<Vec<f64>>,

    /// Summary statistics of ICE spread at each grid point.
    pub ice_spread: IceSpread,
}

/// Summary statistics of ICE curve spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceSpread {
    /// Standard deviation of ICE values at each grid point.
    pub std_dev: Vec<f64>,

    /// Range (max - min) at each grid point.
    pub range: Vec<f64>,

    /// Interquartile range at each grid point.
    pub iqr: Vec<f64>,

    /// Indicates heterogeneity: max(std_dev) / mean(|pdp|).
    pub heterogeneity_index: f64,
}

impl std::fmt::Display for IceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Individual Conditional Expectation (ICE) Results")?;
        writeln!(f, "=================================================")?;

        if let Some(ref name) = self.feature_name {
            writeln!(f, "Feature: {} (index {})", name, self.feature_index)?;
        } else {
            writeln!(f, "Feature index: {}", self.feature_index)?;
        }

        writeln!(f, "Number of observations: {}", self.n_obs)?;
        writeln!(f, "Grid resolution: {} points", self.n_grid)?;
        writeln!(
            f,
            "Grid range: [{:.4}, {:.4}]",
            self.grid_values.first().unwrap_or(&0.0),
            self.grid_values.last().unwrap_or(&0.0)
        )?;
        writeln!(f, "Centered (c-ICE): {}", self.centered)?;

        writeln!(f)?;
        writeln!(f, "Partial Dependence Plot (PDP) Summary:")?;
        let pdp_min = self.pdp_curve.iter().cloned().fold(f64::INFINITY, f64::min);
        let pdp_max = self
            .pdp_curve
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let pdp_mean = self.pdp_curve.iter().sum::<f64>() / self.pdp_curve.len() as f64;
        writeln!(
            f,
            "  Min: {:.4}, Max: {:.4}, Mean: {:.4}",
            pdp_min, pdp_max, pdp_mean
        )?;
        writeln!(f, "  Range: {:.4}", pdp_max - pdp_min)?;

        writeln!(f)?;
        writeln!(f, "Heterogeneity Analysis:")?;
        writeln!(
            f,
            "  Heterogeneity index: {:.4}",
            self.ice_spread.heterogeneity_index
        )?;
        writeln!(
            f,
            "  Max ICE spread (std): {:.4}",
            self.ice_spread
                .std_dev
                .iter()
                .cloned()
                .fold(0.0f64, f64::max)
        )?;
        writeln!(
            f,
            "  Mean ICE spread (std): {:.4}",
            self.ice_spread.std_dev.iter().sum::<f64>() / self.ice_spread.std_dev.len() as f64
        )?;

        if self.ice_spread.heterogeneity_index > 0.5 {
            writeln!(f)?;
            writeln!(
                f,
                "Note: High heterogeneity index suggests interaction effects."
            )?;
            writeln!(
                f,
                "      The feature's effect varies substantially across observations."
            )?;
        }

        Ok(())
    }
}

/// Compute Individual Conditional Expectation (ICE) curves with full analysis.
///
/// This is an enhanced version of ICE curves that includes heterogeneity analysis,
/// sampling for large datasets, and detailed spread statistics. For a simpler
/// version that reports only the averaged curve, see
/// [`crate::ml::partial_dependence`] in the pdp module.
///
/// ICE curves show how each observation's prediction changes as a single feature
/// varies across its range. This reveals heterogeneous effects that may be hidden
/// in the average Partial Dependence Plot (PDP).
///
/// # Arguments
///
/// * `data` - Feature matrix (n_obs x n_features)
/// * `predict_fn` - Function that takes a feature matrix and returns predictions
/// * `config` - Configuration for ICE computation
///
/// # Returns
///
/// `IceResult` containing:
/// - `ice_curves`: n_obs x n_grid matrix of individual predictions
/// - `pdp_curve`: Average of all ICE curves (the PDP)
/// - `grid_values`: Feature values used for the x-axis
/// - `ice_spread`: Heterogeneity statistics
///
/// # Algorithm
///
/// For each observation i and grid value x_s (Goldstein et al. 2015, Eq. 1):
///
/// ICE_i(x_s) = f(x_s, x_{-j}^i)
///
/// where f is the prediction function, x_s is the grid value for feature j,
/// and x_{-j}^i are all other features for observation i.
///
/// # References
///
/// - Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
///   "Peeking Inside the Black Box: Visualizing Statistical Learning with Plots
///   of Individual Conditional Expectation."
///   Journal of Computational and Graphical Statistics, 24(1), 44-65.
pub fn compute_ice_curves<F>(
    data: ArrayView2<f64>,
    predict_fn: F,
    config: &IceConfig,
) -> EconResult<IceResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    ice_core(data, |m: &Array2<f64>| Ok(predict_fn(m.view())), config)
}

/// [`compute_ice_curves`] for a prediction function that can fail: its errors are returned unchanged, at the
/// first grid point that fails.
pub fn try_compute_ice_curves<F>(
    data: ArrayView2<f64>,
    predict_fn: F,
    config: &IceConfig,
) -> EconResult<IceResult>
where
    F: Fn(ArrayView2<f64>) -> EconResult<Vec<f64>>,
{
    ice_core(data, |m: &Array2<f64>| predict_fn(m.view()), config)
}

/// ICE on the engine of [`crate::ml::pdp`], for every entry point of this module.
fn ice_core<F>(data: ArrayView2<'_, f64>, predict: F, config: &IceConfig) -> EconResult<IceResult>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
{
    let n_obs = data.nrows();
    let n_features = data.ncols();

    // Validate inputs
    if n_obs == 0 {
        return Err(EconError::EmptyDataset);
    }

    if config.feature_index >= n_features {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Feature index {} is out of bounds (data has {} features)",
                config.feature_index, n_features
            ),
        });
    }

    let frac = config.frac_to_plot;
    if !(frac > 0.0 && frac <= 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("frac_to_plot must be in (0, 1], got {frac}"),
        });
    }

    // Sample observations if needed
    let sampled_data = if frac < 1.0 {
        let n_sample = ((n_obs as f64) * frac).ceil() as usize;
        let n_sample = n_sample.max(1).min(n_obs);

        let mut rng_state = config.seed.unwrap_or(42);
        let mut indices: Vec<usize> = (0..n_obs).collect();

        // Fisher-Yates shuffle
        for i in 0..n_sample.min(indices.len()) {
            let j = i + lcg_random(&mut rng_state) % (indices.len() - i);
            indices.swap(i, j);
        }
        indices.truncate(n_sample);
        indices.sort_unstable(); // Sort for cache-friendly access

        select_rows(&data, &indices)
    } else {
        data.to_owned()
    };

    let n_sampled = sampled_data.nrows();
    let feature_index = config.feature_index;

    // The original feature values and the grid built from them
    let original_feature_values = sampled_data.column(feature_index).to_vec();
    let spec = config.grid_spec(&original_feature_values)?;
    let grid_values = pdp::feature_grid(
        &original_feature_values,
        &spec,
        &format!("feature {feature_index}"),
    )?;
    let n_grid = grid_values.len();

    // Compute ICE curves: one prediction call per grid point
    let mut ice_curves = try_zeros2(n_sampled, n_grid, "ICE curves")?;
    pdp::evaluate_grid(
        sampled_data.view(),
        &[feature_index],
        &[grid_values.as_slice()],
        predict,
        |g, predictions| {
            ice_curves
                .column_mut(g[0])
                .assign(&ArrayView1::from(predictions));
            Ok(())
        },
    )?;

    // Apply centering if requested (Goldstein et al. 2015, Section 3.1):
    // the grid is sorted, so column 0 is the smallest grid value
    let centered = config.center;
    if centered {
        let anchor = ice_curves.column(0).to_owned();
        for mut column in ice_curves.columns_mut() {
            column -= &anchor;
        }
    }

    // Compute PDP (average of ICE curves)
    let mut pdp_curve = try_with_capacity::<f64>(n_grid, "ICE curves")?;
    pdp_curve.extend(
        ice_curves
            .columns()
            .into_iter()
            .map(|column| column.iter().sum::<f64>() / n_sampled as f64),
    );

    // Compute heterogeneity statistics
    let ice_spread = compute_ice_spread(ice_curves.view(), &pdp_curve)?;

    // Convert ice_curves to Vec for JSON serialization
    let ice_curves_vec: Vec<Vec<f64>> =
        (0..n_sampled).map(|i| ice_curves.row(i).to_vec()).collect();

    Ok(IceResult {
        grid_values,
        ice_curves,
        pdp_curve,
        feature_index,
        feature_name: config.feature_name.clone(),
        n_obs: n_sampled,
        n_grid,
        centered,
        original_feature_values,
        ice_curves_vec,
        ice_spread,
    })
}

/// Compute ICE spread statistics to quantify heterogeneity: population SD, range and `IQR()` (type 7) per grid
/// point; a grid point with a NaN prediction gets NaN for all three.
fn compute_ice_spread(ice_curves: ArrayView2<'_, f64>, pdp_curve: &[f64]) -> EconResult<IceSpread> {
    let n_grid = ice_curves.ncols();
    let n_obs = ice_curves.nrows();

    let mut std_dev = try_vec(n_grid, f64::NAN, "ICE spread")?;
    let mut range = try_vec(n_grid, f64::NAN, "ICE spread")?;
    let mut iqr = try_vec(n_grid, f64::NAN, "ICE spread")?;
    let mut sorted = try_with_capacity::<f64>(n_obs, "ICE spread")?;

    for (g_idx, col) in ice_curves.columns().into_iter().enumerate() {
        if col.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean = col.iter().sum::<f64>() / n_obs as f64;
        let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_obs as f64;
        std_dev[g_idx] = variance.sqrt();

        sorted.clear();
        sorted.extend(col.iter().copied());
        sorted.sort_unstable_by(f64::total_cmp);
        range[g_idx] = sorted[n_obs - 1] - sorted[0];
        iqr[g_idx] = quantile_sorted(&sorted, 0.75, QuantileType::T7)
            - quantile_sorted(&sorted, 0.25, QuantileType::T7);
    }

    // Heterogeneity index: max(std) / mean(|pdp|)
    let max_std = std_dev.iter().cloned().fold(0.0f64, f64::max);
    let mean_abs_pdp = pdp_curve.iter().map(|v| v.abs()).sum::<f64>() / pdp_curve.len() as f64;
    let heterogeneity_index = if mean_abs_pdp > 1e-10 {
        max_std / mean_abs_pdp
    } else {
        max_std // If PDP is near zero, just use max std
    };

    Ok(IceSpread {
        std_dev,
        range,
        iqr,
        heterogeneity_index,
    })
}

use super::lcg_random;

/// Select rows from a 2D array by indices.
fn select_rows(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_cols));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Compute ICE curves using a GBM model.
///
/// # Arguments
///
/// * `data` - Feature matrix used for computing ICE, with the columns the model was fitted on
/// * `gbm_result` - Fitted GBM model
/// * `feature_index` - Index of feature to analyze
/// * `grid_resolution` - Number of grid points (default: `min(length(unique(x)), 51)`, as pdp)
/// * `center` - Whether to compute centered ICE (default: false)
///
/// # Errors
///
/// `InvalidSpecification` for a model without trees or data with another column count; the model's prediction
/// errors are returned unchanged.
pub fn ice_curves_gbm(
    data: ArrayView2<f64>,
    gbm_result: &crate::ml::GbmResult,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    let config = IceConfig {
        feature_index,
        grid_resolution,
        center: center.unwrap_or(false),
        feature_name: gbm_result
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    ice_core(data, pdp::gbm_predictor(gbm_result, data.ncols())?, &config)
}

/// ICE curves of a random forest fitted with [`crate::ml::random_forest_with_trees`] (the forest's mean tree
/// prediction). Arguments and errors as [`ice_curves_gbm`].
pub fn ice_curves_rf(
    data: ArrayView2<f64>,
    model: &crate::ml::RandomForestModel,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    let config = IceConfig {
        feature_index,
        grid_resolution,
        center: center.unwrap_or(false),
        feature_name: model
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    ice_core(data, pdp::forest_predictor(model, data.ncols())?, &config)
}

/// Compute ICE curves using a CART model. Arguments and errors as [`ice_curves_gbm`].
pub fn ice_curves_cart(
    data: ArrayView2<f64>,
    cart_result: &crate::ml::CartResult,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    let config = IceConfig {
        feature_index,
        grid_resolution,
        center: center.unwrap_or(false),
        feature_name: cart_result
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    ice_core(
        data,
        pdp::cart_predictor(cart_result, data.ncols())?,
        &config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::pdp::test_fixtures::*;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    /// Simple linear model for testing: y = 2*x0 + 3*x1
    fn linear_predict(x: ArrayView2<f64>) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| 2.0 * x[[i, 0]] + 3.0 * x[[i, 1]])
            .collect()
    }

    /// Model with interaction: y = x0 * x1
    fn interaction_predict(x: ArrayView2<f64>) -> Vec<f64> {
        (0..x.nrows()).map(|i| x[[i, 0]] * x[[i, 1]]).collect()
    }

    #[test]
    fn test_ice_basic() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        assert_eq!(result.n_obs, 5);
        assert_eq!(result.n_grid, 5);
        assert_eq!(result.feature_index, 0);
        assert!(!result.centered);

        // Grid should span from 1.0 to 5.0
        assert!((result.grid_values[0] - 1.0).abs() < 1e-10);
        assert!((result.grid_values[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ice_linear_model_parallel_curves() {
        // For a linear model without interactions, ICE curves should be parallel
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        // In a linear model, when we vary x0, the effect is constant across observations
        // However, each observation has a different x1, so curves are shifted vertically
        // But the SLOPE should be the same (parallel curves)

        // Compute slopes for each curve
        let slopes: Vec<f64> = (0..result.n_obs)
            .map(|i| {
                let y1 = result.ice_curves[[i, 0]];
                let y2 = result.ice_curves[[i, 4]];
                let x1 = result.grid_values[0];
                let x2 = result.grid_values[4];
                (y2 - y1) / (x2 - x1)
            })
            .collect();

        // All slopes should be equal to 2.0 (coefficient of x0)
        for slope in slopes {
            assert!(
                (slope - 2.0).abs() < 1e-10,
                "Expected slope 2.0, got {}",
                slope
            );
        }
    }

    #[test]
    fn test_ice_centered() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            center: true,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        assert!(result.centered);

        // All curves should start at 0 after centering
        for i in 0..result.n_obs {
            assert!(
                result.ice_curves[[i, 0]].abs() < 1e-10,
                "Centered curve {} should start at 0, got {}",
                i,
                result.ice_curves[[i, 0]]
            );
        }
    }

    #[test]
    fn test_ice_interaction_heterogeneity() {
        // With interactions, ICE curves should NOT be parallel
        // Use data with variance in feature 0, different x1 values across observations
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), interaction_predict, &config).unwrap();

        // With interaction model (y = x0 * x1), when we vary x0,
        // the slope at each observation depends on x1 value
        // Slopes should differ because each obs has different x1

        // Heterogeneity index should be high due to interaction
        assert!(
            result.ice_spread.heterogeneity_index > 0.01,
            "Expected some heterogeneity due to interaction, got {}",
            result.ice_spread.heterogeneity_index
        );

        // ICE spread should be non-zero (curves aren't parallel)
        let max_std = result
            .ice_spread
            .std_dev
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        assert!(
            max_std > 0.0,
            "Expected non-zero ICE spread, got max_std = {}",
            max_std
        );
    }

    #[test]
    fn test_ice_pdp_is_average() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        // PDP should be the average of ICE curves at each grid point
        for g_idx in 0..result.n_grid {
            let expected_pdp: f64 = (0..result.n_obs)
                .map(|i| result.ice_curves[[i, g_idx]])
                .sum::<f64>()
                / result.n_obs as f64;

            assert!(
                (result.pdp_curve[g_idx] - expected_pdp).abs() < 1e-10,
                "PDP mismatch at grid point {}: expected {}, got {}",
                g_idx,
                expected_pdp,
                result.pdp_curve[g_idx]
            );
        }
    }

    #[test]
    fn test_ice_custom_grid() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(50), // Ignored when custom grid provided
            grid_values: Some(vec![0.0, 2.5, 5.0, 7.5, 10.0]),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        assert_eq!(result.n_grid, 5);
        assert_eq!(result.grid_values, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_ice_sampling() {
        // Create larger dataset
        let mut data_vec = Vec::new();
        for i in 0..100 {
            data_vec.push(i as f64);
            data_vec.push((i as f64) * 2.0);
        }
        let data = Array2::from_shape_vec((100, 2), data_vec).unwrap();

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(10),
            frac_to_plot: 0.1, // Use 10% of data
            seed: Some(42),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        // Should have ~10 observations
        assert!(result.n_obs <= 15, "Expected ~10 obs, got {}", result.n_obs);
        assert!(result.n_obs >= 5, "Expected ~10 obs, got {}", result.n_obs);
    }

    #[test]
    fn test_ice_error_invalid_feature() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let config = IceConfig {
            feature_index: 5, // Out of bounds
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ice_display() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(5),
            feature_name: Some("test_feature".to_string()),
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), linear_predict, &config).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Individual Conditional Expectation"));
        assert!(display.contains("test_feature"));
        assert!(display.contains("5 points"));
    }

    #[test]
    fn test_ice_spread_computation() {
        // Create data where ICE curves have known spread
        // Need variance in feature 0 for the analysis
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: Some(3),
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), interaction_predict, &config).unwrap();

        // Should have non-zero spread due to interaction (y = x0 * x1)
        // Different x1 values mean different slopes when varying x0
        assert!(result.ice_spread.std_dev.iter().any(|&s| s > 0.0));
        assert!(result.ice_spread.range.iter().any(|&r| r > 0.0));
    }

    fn lm_view_closure(b: &[f64]) -> impl Fn(ArrayView2<f64>) -> Vec<f64> + Copy + '_ {
        move |m: ArrayView2<f64>| m.rows().into_iter().map(|r| pdp_lm_value(b, r)).collect()
    }

    fn flat(a: &Array2<f64>) -> Vec<f64> {
        a.iter().copied().collect() // standard layout: row-major, observation slowest (as the case stores it)
    }

    /// pdp::partial(ice = TRUE[, center = TRUE]) on an unsorted user grid: the grid is sorted (order_grid) and the
    /// centred curves start at the smallest grid value. Before: the grid kept the user's order, anchored at 1.3.
    #[test]
    fn test_validate_pdp_lm_ice_custom_grid() {
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let b = pdp_lm_coef(&case);
        let config = IceConfig {
            grid_values: Some(case.vector("ice_user_grid")), // 1.3, -1.7, 0.25, -0.6, 1.9
            ..Default::default()
        };
        let ice = compute_ice_curves(x.view(), lm_view_closure(&b), &config).unwrap();
        let config = IceConfig {
            center: true,
            ..config
        };
        let cice = compute_ice_curves(x.view(), lm_view_closure(&b), &config).unwrap();
        Check::new(&case)
            .vector("ice_sorted_grid", &ice.grid_values, Tol::exact())
            .vector("ice_custom_yhat", &flat(&ice.ice_curves), PD_TOL)
            .vector("ice_custom_pd", &ice.pdp_curve, PD_TOL)
            .vector("ice_sorted_grid", &cice.grid_values, Tol::exact())
            .vector("cice_custom_yhat", &flat(&cice.ice_curves), PD_TOL)
            .vector("cice_custom_pd", &cice.pdp_curve, PD_TOL)
            .finish();
    }

    /// The 8-point uniform ICE grid and the spread statistics: population SD, range, IQR type 7.
    #[test]
    fn test_validate_pdp_lm_ice_uniform() {
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let b = pdp_lm_coef(&case);
        let config = IceConfig {
            grid_resolution: Some(8),
            ..Default::default()
        };
        let r = compute_ice_curves(x.view(), lm_view_closure(&b), &config).unwrap();
        let d = compute_ice_curves(x.view(), lm_view_closure(&b), &IceConfig::default()).unwrap();
        Check::new(&case)
            .vector("ice_uniform_grid", &r.grid_values, Tol::exact())
            .vector("ice_uniform_yhat", &flat(&r.ice_curves), PD_TOL)
            .vector("ice_uniform_pd", &r.pdp_curve, PD_TOL)
            .vector("ice_uniform_sd_pop", &r.ice_spread.std_dev, PD_TOL)
            .vector("ice_uniform_range", &r.ice_spread.range, PD_TOL)
            .vector("ice_uniform_iqr7", &r.ice_spread.iqr, PD_TOL)
            // the default grid: pdp's min(length(unique(x)), 51) points
            .vector("pd_default_grid", &d.grid_values, Tol::exact())
            .vector("pd_default_yhat", &d.pdp_curve, PD_TOL)
            .finish();
    }

    /// Centred ICE for a feature on the 1e-12 scale (was Err "Feature 0 has no variance").
    #[test]
    fn test_validate_pdp_small_scale_cice() {
        let (case, ds) = crate::test_support::load_case("pdp_small_scale");
        let x = matrix(&ds, &["x", "z"]);
        let b = case.by_names("coef", &["(Intercept)", "x", "z"]);
        let config = IceConfig {
            grid_resolution: Some(6),
            center: true,
            ..Default::default()
        };
        let f = |m: ArrayView2<f64>| -> Vec<f64> {
            m.rows()
                .into_iter()
                .map(|r| small_scale_value(&b, r))
                .collect()
        };
        let r = compute_ice_curves(x.view(), f, &config).unwrap();
        Check::new(&case)
            .vector("cice_uniform_grid", &r.grid_values, Tol::exact())
            .vector("cice_uniform_yhat", &flat(&r.ice_curves), PD_TOL)
            .finish();
    }

    /// An empty or non-finite custom grid is an error (it reached the centring step, or gave an empty PDP).
    #[test]
    fn test_ice_empty_custom_grid_is_error() {
        let x = array![[1.0, 2.0], [2.0, 3.5], [3.0, 3.9], [4.0, 5.2]];
        for center in [true, false] {
            let config = IceConfig {
                grid_values: Some(vec![]),
                center,
                ..Default::default()
            };
            let e = compute_ice_curves(x.view(), linear_predict, &config).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message }
                    if message.contains("grid_values must contain at least one value")),
                "{e}"
            );
        }
        let config = IceConfig {
            grid_values: Some(vec![1.0, f64::NAN]),
            ..Default::default()
        };
        let e = compute_ice_curves(x.view(), linear_predict, &config).unwrap_err();
        assert!(e.to_string().contains("grid_values must be finite"), "{e}");
    }

    /// A GBM whose prediction fails is an error, not all-zero curves.
    #[test]
    fn test_ice_gbm_prediction_error_propagates() {
        let (x, y) = two_feature_data();
        let gc = crate::ml::GbmConfig {
            n_trees: 5,
            seed: Some(7),
            ..Default::default()
        };
        let mut fit = crate::ml::gbm(x.view(), y.view(), &gc).unwrap();
        fit.trees.clear(); // a GBM without fitted trees: gbm_predict errs
        let e = ice_curves_gbm(x.view(), &fit, 0, Some(5), None).unwrap_err();
        assert!(e.to_string().contains("no fitted trees"), "{e}");
    }

    /// Data with fewer columns than the model was fitted on is an error, not an out-of-bounds index in the
    /// trees' traversal (every tree here splits on column 1 first).
    #[test]
    fn test_ice_model_wrappers_refuse_column_mismatch() {
        let (x, y) = two_feature_data();
        let narrow = x.slice(ndarray::s![.., 0..1]);
        let gc = crate::ml::GbmConfig {
            n_trees: 5,
            seed: Some(7),
            ..Default::default()
        };
        let gfit = crate::ml::gbm(x.view(), y.view(), &gc).unwrap();
        let cfit = crate::ml::cart(x.view(), y.view(), &crate::ml::CartConfig::default()).unwrap();
        for e in [
            ice_curves_gbm(narrow, &gfit, 0, Some(4), None).unwrap_err(),
            ice_curves_cart(narrow, &cfit, 0, Some(4), None).unwrap_err(),
        ] {
            assert!(
                e.to_string()
                    .contains("was fitted on 2 features; the data has 1 columns"),
                "{e}"
            );
        }
    }

    /// frac_to_plot NaN refused (was silently 1); grid_resolution 0 refused, 1 accepted (pdp's
    /// seq(length.out = 1)); a constant feature gives a one-point grid (was Err "has no variance"), as pdp.
    #[test]
    fn test_ice_parameter_rules() {
        let x = array![[1.0, 2.0], [2.0, 3.5], [3.0, 3.9], [4.0, 5.2]];
        let c = IceConfig {
            frac_to_plot: f64::NAN,
            ..Default::default()
        };
        assert!(
            compute_ice_curves(x.view(), linear_predict, &c)
                .unwrap_err()
                .to_string()
                .contains("frac_to_plot must be in (0, 1]")
        );
        let c = IceConfig {
            grid_resolution: Some(0),
            ..Default::default()
        };
        assert!(
            compute_ice_curves(x.view(), linear_predict, &c)
                .unwrap_err()
                .to_string()
                .contains("grid_resolution must be at least 1")
        );
        let c = IceConfig {
            grid_resolution: Some(1),
            ..Default::default()
        };
        assert_eq!(
            compute_ice_curves(x.view(), linear_predict, &c)
                .unwrap()
                .grid_values,
            vec![1.0]
        );
        let flat_x = array![[1.0, 2.0], [1.0, 3.5], [1.0, 3.9]];
        let c = IceConfig {
            center: true,
            ..Default::default()
        };
        let r = compute_ice_curves(flat_x.view(), linear_predict, &c).unwrap();
        assert_eq!(r.grid_values, vec![1.0]);
        assert!(r.ice_curves.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_ice_prediction_error_propagates() {
        let x = array![[1.0, 2.0], [2.0, 3.5], [3.0, 3.9], [4.0, 5.2]];
        let calls = std::cell::Cell::new(0);
        let failing = |m: ArrayView2<f64>| -> EconResult<Vec<f64>> {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                Err(EconError::Computation("model failed".into()))
            } else {
                Ok(linear_predict(m))
            }
        };
        let e = try_compute_ice_curves(x.view(), failing, &IceConfig::default()).unwrap_err();
        assert!(matches!(&e, EconError::Computation(m) if m == "model failed"));
        assert_eq!(calls.get(), 2);
    }

    /// A NaN prediction gives NaN spread statistics at its grid point and leaves the others alone.
    #[test]
    fn test_ice_spread_nan_column() {
        let x = array![[1.0, 2.0], [2.0, 3.5], [3.0, 3.9], [4.0, 5.2]];
        let f = |m: ArrayView2<f64>| -> Vec<f64> {
            (0..m.nrows())
                .map(|i| {
                    if m[[i, 0]] == 2.0 && i == 1 {
                        f64::NAN
                    } else {
                        linear_predict(m)[i]
                    }
                })
                .collect()
        };
        let c = IceConfig {
            grid_values: Some(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        };
        let r = compute_ice_curves(x.view(), f, &c).unwrap();
        for spread in [
            &r.ice_spread.std_dev,
            &r.ice_spread.range,
            &r.ice_spread.iqr,
        ] {
            assert!(spread[1].is_nan());
            assert!(spread[0].is_finite() && spread[2].is_finite());
        }
    }

    /// ice_curves_rf on a RandomForestModel (it always returned an error): every ICE value is the forest's tree
    /// average on the modified row, and at an observation's own x1 it is that observation's fitted value, bit for
    /// bit. A property, so it reads the data directly and pins no R value.
    #[test]
    fn test_ice_curves_rf_uses_forest_predictions() {
        let (x, y) = pdp_lm_data();
        let model = crate::ml::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(20),
            Some(6),
            None,
            None,
            Some(11),
            None,
        )
        .unwrap();
        let r = ice_curves_rf(x.view(), &model, 0, Some(6), None).unwrap();
        for i in 0..x.nrows() {
            for (g, &v) in r.grid_values.iter().enumerate() {
                let mut row = x.row(i).to_owned();
                row[0] = v;
                let want = model
                    .trees
                    .iter()
                    .map(|t| t.predict_one(&row.view()))
                    .sum::<f64>()
                    / model.trees.len() as f64;
                assert_eq!(r.ice_curves[[i, g]].to_bits(), want.to_bits());
            }
        }
        let own = IceConfig {
            grid_values: Some(x.column(0).to_vec()),
            ..Default::default()
        };
        let f = crate::ml::pdp::forest_predictor(&model, 3).unwrap();
        let r = try_compute_ice_curves(x.view(), |v| f(&v.to_owned()), &own).unwrap();
        for i in 0..x.nrows() {
            let g = r.grid_values.iter().position(|&v| v == x[[i, 0]]).unwrap();
            assert_eq!(
                r.ice_curves[[i, g]].to_bits(),
                model.predictions[i].to_bits()
            );
        }
        // wrong column count
        let e =
            ice_curves_rf(x.slice(ndarray::s![.., 0..2]), &model, 0, Some(6), None).unwrap_err();
        assert!(
            e.to_string()
                .contains("was fitted on 3 features; the data has 2 columns"),
            "{e}"
        );
    }
}
