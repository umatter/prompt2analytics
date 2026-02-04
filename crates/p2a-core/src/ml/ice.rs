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
//! Centering reveals heterogeneity by subtracting the prediction at an anchor point:
//! c-ICE_i(x_s) = ICE_i(x_s) - ICE_i(x_anchor)
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
//! ```rust,no_run
//! use p2a_core::ml::{random_forest, ice_curves, IceConfig};
//! use ndarray::{array, Array2};
//!
//! // Train a model
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
//! let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let rf = random_forest(x.view(), y.view(), Some(10), None, None, None, Some(42), None).unwrap();
//!
//! // Compute ICE curves for feature 0
//! let config = IceConfig {
//!     feature_index: 0,
//!     grid_resolution: 20,
//!     center: true,
//!     ..Default::default()
//! };
//!
//! let result = compute_ice_curves(
//!     x.view(),
//!     |x_data| {
//!         // Predict using the model
//!         (0..x_data.nrows())
//!             .map(|i| {
//!                 let row = x_data.row(i);
//!                 // Simple average for RF prediction
//!                 row.mean().unwrap_or(0.0)
//!             })
//!             .collect()
//!     },
//!     &config,
//! ).unwrap();
//!
//! println!("Grid: {:?}", result.grid_values);
//! println!("PDP: {:?}", result.pdp_curve);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// Configuration for ICE curve computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceConfig {
    /// Index of the feature to compute ICE for (0-indexed).
    pub feature_index: usize,

    /// Number of grid points for the feature range. Default: 50.
    pub grid_resolution: usize,

    /// Whether to center curves at the minimum feature value (c-ICE). Default: false.
    pub center: bool,

    /// Fraction of observations to use (for large datasets). Default: 1.0 (all).
    /// Set to e.g. 0.1 to randomly sample 10% of observations.
    pub frac_to_plot: f64,

    /// Random seed for sampling when frac_to_plot < 1.0.
    pub seed: Option<u64>,

    /// Custom grid values (overrides grid_resolution if provided).
    pub grid_values: Option<Vec<f64>>,

    /// Feature name (for display purposes).
    pub feature_name: Option<String>,
}

impl Default for IceConfig {
    fn default() -> Self {
        IceConfig {
            feature_index: 0,
            grid_resolution: 50,
            center: false,
            frac_to_plot: 1.0,
            seed: None,
            grid_values: None,
            feature_name: None,
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
/// version, see [`crate::ml::ice_curves`] in the pdp module.
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

    if config.grid_resolution < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Grid resolution must be at least 2".to_string(),
        });
    }

    if config.frac_to_plot <= 0.0 || config.frac_to_plot > 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "frac_to_plot must be in (0, 1]".to_string(),
        });
    }

    // Sample observations if needed
    let (sampled_indices, sampled_data) = if config.frac_to_plot < 1.0 {
        let n_sample = ((n_obs as f64) * config.frac_to_plot).ceil() as usize;
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

        let sampled = select_rows(&data, &indices);
        (indices, sampled)
    } else {
        let indices: Vec<usize> = (0..n_obs).collect();
        (indices, data.to_owned())
    };

    let n_sampled = sampled_data.nrows();

    // Determine grid values
    let grid_values = if let Some(ref custom_grid) = config.grid_values {
        custom_grid.clone()
    } else {
        // Extract feature column and compute range
        let feature_col = sampled_data.column(config.feature_index);
        let mut feature_values: Vec<f64> = feature_col.iter().cloned().collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = *feature_values.first().unwrap();
        let max_val = *feature_values.last().unwrap();

        if (max_val - min_val).abs() < 1e-10 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Feature {} has no variance (all values are {:.4})",
                    config.feature_index, min_val
                ),
            });
        }

        // Create evenly spaced grid
        let step = (max_val - min_val) / (config.grid_resolution - 1) as f64;
        (0..config.grid_resolution)
            .map(|i| min_val + i as f64 * step)
            .collect()
    };

    let n_grid = grid_values.len();

    // Store original feature values
    let original_feature_values: Vec<f64> = sampled_data
        .column(config.feature_index)
        .iter()
        .cloned()
        .collect();

    // Compute ICE curves
    // For efficiency, we batch all predictions for each grid value
    let mut ice_curves = Array2::zeros((n_sampled, n_grid));

    for (g_idx, &grid_val) in grid_values.iter().enumerate() {
        // Create modified data with feature set to grid value
        let mut modified_data = sampled_data.clone();

        for i in 0..n_sampled {
            modified_data[[i, config.feature_index]] = grid_val;
        }

        // Get predictions for all observations at this grid point
        let predictions = predict_fn(modified_data.view());

        if predictions.len() != n_sampled {
            return Err(EconError::Computation(format!(
                "Prediction function returned {} values but expected {}",
                predictions.len(),
                n_sampled
            )));
        }

        // Store predictions
        for (i, &pred) in predictions.iter().enumerate() {
            ice_curves[[i, g_idx]] = pred;
        }
    }

    // Apply centering if requested (Goldstein et al. 2015, Section 3.1)
    // c-ICE_i(x_s) = ICE_i(x_s) - ICE_i(x_anchor)
    let centered = config.center;
    if centered {
        // Center at the first grid point (minimum feature value)
        let anchor_predictions = ice_curves.column(0).to_owned();
        for g_idx in 0..n_grid {
            for i in 0..n_sampled {
                ice_curves[[i, g_idx]] -= anchor_predictions[i];
            }
        }
    }

    // Compute PDP (average of ICE curves)
    let pdp_curve: Vec<f64> = (0..n_grid)
        .map(|g_idx| ice_curves.column(g_idx).mean().unwrap_or(0.0))
        .collect();

    // Compute heterogeneity statistics
    let ice_spread = compute_ice_spread(&ice_curves.view(), &pdp_curve);

    // Convert ice_curves to Vec for JSON serialization
    let ice_curves_vec: Vec<Vec<f64>> =
        (0..n_sampled).map(|i| ice_curves.row(i).to_vec()).collect();

    Ok(IceResult {
        grid_values,
        ice_curves,
        pdp_curve,
        feature_index: config.feature_index,
        feature_name: config.feature_name.clone(),
        n_obs: n_sampled,
        n_grid,
        centered,
        original_feature_values,
        ice_curves_vec,
        ice_spread,
    })
}

/// Compute ICE spread statistics to quantify heterogeneity.
fn compute_ice_spread(ice_curves: &ArrayView2<f64>, pdp_curve: &[f64]) -> IceSpread {
    let n_grid = ice_curves.ncols();
    let n_obs = ice_curves.nrows();

    let mut std_dev = Vec::with_capacity(n_grid);
    let mut range = Vec::with_capacity(n_grid);
    let mut iqr = Vec::with_capacity(n_grid);

    for g_idx in 0..n_grid {
        let col = ice_curves.column(g_idx);
        let mean = col.mean().unwrap_or(0.0);

        // Standard deviation
        let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_obs as f64;
        std_dev.push(variance.sqrt());

        // Range
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        range.push(max_val - min_val);

        // IQR
        let mut sorted: Vec<f64> = col.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = n_obs / 4;
        let q3_idx = (3 * n_obs) / 4;
        let q1 = if n_obs > 0 { sorted[q1_idx] } else { 0.0 };
        let q3 = if n_obs > 0 {
            sorted[q3_idx.min(n_obs - 1)]
        } else {
            0.0
        };
        iqr.push(q3 - q1);
    }

    // Heterogeneity index: max(std) / mean(|pdp|)
    let max_std = std_dev.iter().cloned().fold(0.0f64, f64::max);
    let mean_abs_pdp = pdp_curve.iter().map(|v| v.abs()).sum::<f64>() / pdp_curve.len() as f64;
    let heterogeneity_index = if mean_abs_pdp > 1e-10 {
        max_std / mean_abs_pdp
    } else {
        max_std // If PDP is near zero, just use max std
    };

    IceSpread {
        std_dev,
        range,
        iqr,
        heterogeneity_index,
    }
}

/// Simple LCG random number generator for reproducible sampling.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

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
/// Convenience function that wraps `ice_curves` for GBM models.
///
/// # Arguments
///
/// * `data` - Feature matrix used for computing ICE
/// * `gbm_result` - Fitted GBM model
/// * `feature_index` - Index of feature to analyze
/// * `grid_resolution` - Number of grid points (default: 50)
/// * `center` - Whether to compute centered ICE (default: false)
pub fn ice_curves_gbm(
    data: ArrayView2<f64>,
    gbm_result: &crate::ml::GbmResult,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    let config = IceConfig {
        feature_index,
        grid_resolution: grid_resolution.unwrap_or(50),
        center: center.unwrap_or(false),
        feature_name: gbm_result
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    compute_ice_curves(
        data,
        |x| crate::ml::gbm_predict(gbm_result, x).unwrap_or_else(|_| vec![0.0; x.nrows()]),
        &config,
    )
}

/// Compute ICE curves using a Random Forest model.
///
/// Convenience function that wraps `ice_curves` for Random Forest models.
pub fn ice_curves_rf(
    data: ArrayView2<f64>,
    rf_result: &crate::ml::RandomForestResult,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    // Random Forest stores fitted predictions, not trees for prediction
    // We need to re-fit or use a different approach
    // For now, this is a placeholder - RF doesn't expose predict easily
    let config = IceConfig {
        feature_index,
        grid_resolution: grid_resolution.unwrap_or(50),
        center: center.unwrap_or(false),
        feature_name: rf_result
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    // Note: Random Forest result doesn't store trees for prediction
    // This is a limitation - user should use ice_curves directly with predict_fn
    Err(EconError::InvalidSpecification {
        message: "Random Forest ICE requires using compute_ice_curves() directly with a predict function. \
                  The RF result doesn't store trees for out-of-sample prediction."
            .to_string(),
    })
}

/// Compute ICE curves using a CART model.
///
/// Convenience function that wraps `ice_curves` for CART models.
pub fn ice_curves_cart(
    data: ArrayView2<f64>,
    cart_result: &crate::ml::CartResult,
    feature_index: usize,
    grid_resolution: Option<usize>,
    center: Option<bool>,
) -> EconResult<IceResult> {
    let config = IceConfig {
        feature_index,
        grid_resolution: grid_resolution.unwrap_or(50),
        center: center.unwrap_or(false),
        feature_name: cart_result
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned()),
        ..Default::default()
    };

    compute_ice_curves(
        data,
        |x| crate::ml::cart_predict(cart_result, x).unwrap_or_else(|_| vec![0.0; x.nrows()]),
        &config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
            grid_resolution: 5,
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
            grid_resolution: 5,
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
            grid_resolution: 5,
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
        let data = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];

        let config = IceConfig {
            feature_index: 0,
            grid_resolution: 5,
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
            grid_resolution: 5,
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
            grid_resolution: 50, // Ignored when custom grid provided
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
            grid_resolution: 10,
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
    fn test_ice_error_no_variance() {
        let data = array![[1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];

        let config = IceConfig {
            feature_index: 0, // All values are 1.0
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
            grid_resolution: 5,
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
            grid_resolution: 3,
            center: false,
            ..Default::default()
        };

        let result = compute_ice_curves(data.view(), interaction_predict, &config).unwrap();

        // Should have non-zero spread due to interaction (y = x0 * x1)
        // Different x1 values mean different slopes when varying x0
        assert!(result.ice_spread.std_dev.iter().any(|&s| s > 0.0));
        assert!(result.ice_spread.range.iter().any(|&r| r > 0.0));
    }
}
