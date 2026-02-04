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
//! ```rust,no_run
//! use p2a_core::ml::{random_forest, partial_dependence, PdpConfig};
//! use ndarray::{array, Array2};
//!
//! // Fit a model
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
//! let y = array![1.0, 2.0, 3.0, 4.0];
//! let rf = random_forest(x.view(), y.view(), Some(10), None, None, None, None, None).unwrap();
//!
//! // Compute PDP for feature 0
//! let config = PdpConfig::default();
//! let pdp = partial_dependence(
//!     &x,
//!     0,
//!     |data| {
//!         // Predict function using the model
//!         data.rows().into_iter()
//!             .map(|row| rf.predictions.iter().sum::<f64>() / rf.predictions.len() as f64)
//!             .collect()
//!     },
//!     &config,
//! ).unwrap();
//! println!("Grid values: {:?}", pdp.grid_values);
//! println!("PDP values: {:?}", pdp.pdp_values);
//! ```
//!
//! ## References
//!
//! - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine".
//!   Annals of Statistics, 29(5), 1189-1232.
//!   https://doi.org/10.1214/aos/1013203451
//!
//! - R package `pdp`: Greenwell, B. M. (2017). "pdp: An R Package for Constructing
//!   Partial Dependence Plots". The R Journal, 9(1), 421-436.
//!   https://journal.r-project.org/archive/2017/RJ-2017-016/
//!
//! - Python library `scikit-learn`: sklearn.inspection.partial_dependence
//!   https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html

use ndarray::{Array1, Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// Configuration for Partial Dependence Plot computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdpConfig {
    /// Number of grid points to evaluate (default: 20)
    pub grid_resolution: usize,

    /// Use quantiles for grid points instead of uniform spacing (default: true)
    ///
    /// When true, grid points are placed at quantiles of the feature distribution,
    /// which better captures the data density. When false, uses uniform spacing
    /// between min and max values.
    pub use_quantiles: bool,

    /// Quantile probabilities for grid points (default: evenly spaced)
    ///
    /// If `use_quantiles` is true and this is None, quantiles are automatically
    /// computed based on `grid_resolution`.
    pub quantiles: Option<Vec<f64>>,

    /// Center the PDP values by subtracting the mean (default: false)
    ///
    /// Centered PDPs (c-PDP) show relative effects rather than absolute predictions.
    pub center: bool,

    /// Grid points for 2D PDP (second feature)
    ///
    /// If provided along with a second feature index, computes a 2D interaction PDP.
    pub grid_resolution_2: Option<usize>,
}

impl Default for PdpConfig {
    fn default() -> Self {
        PdpConfig {
            grid_resolution: 20,
            use_quantiles: true,
            quantiles: None,
            center: false,
            grid_resolution_2: None,
        }
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
/// ```rust,no_run
/// use p2a_core::ml::{partial_dependence, PdpConfig};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64 / 100.0).collect()).unwrap();
///
/// // Simple linear prediction function for example
/// let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
///     x.rows().into_iter().map(|row| row[0] + 2.0 * row[1]).collect()
/// };
///
/// let result = partial_dependence(&data, 0, predict_fn, &PdpConfig::default()).unwrap();
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

    // Extract feature values
    let feature_values: Vec<f64> = data.column(feature_index).iter().cloned().collect();

    // Compute grid points
    let grid_values = compute_grid_points(&feature_values, config)?;
    let n_grid = grid_values.len();

    // Compute PDP values: for each grid point, set feature to that value
    // for all observations and average predictions
    // Algorithm: Friedman (2001), Section 10.13.2
    let mut pdp_values = Vec::with_capacity(n_grid);
    let mut pdp_std = Vec::with_capacity(n_grid);

    for &grid_val in &grid_values {
        // Create modified data with feature set to grid value
        let mut modified_data = data.clone();
        for i in 0..n_samples {
            modified_data[[i, feature_index]] = grid_val;
        }

        // Get predictions
        let predictions = predict_fn(&modified_data);

        if predictions.len() != n_samples {
            return Err(EconError::Computation(format!(
                "Prediction function returned {} values, expected {}",
                predictions.len(),
                n_samples
            )));
        }

        // Average predictions (Monte Carlo integration over X_c)
        let mean: f64 = predictions.iter().sum::<f64>() / n_samples as f64;
        pdp_values.push(mean);

        // Compute standard deviation
        let variance: f64 =
            predictions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / n_samples as f64;
        pdp_std.push(variance.sqrt());
    }

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
    })
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
/// * `config` - PDP configuration (uses grid_resolution for both features)
///
/// # Returns
///
/// Pdp2dResult containing grid values and PDP matrix
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

    // Extract feature values
    let feature_values_1: Vec<f64> = data.column(feature_index_1).iter().cloned().collect();
    let feature_values_2: Vec<f64> = data.column(feature_index_2).iter().cloned().collect();

    // Compute grid points
    let grid_values_1 = compute_grid_points(&feature_values_1, config)?;
    let grid_values_2 = if let Some(res) = config.grid_resolution_2 {
        let config_2 = PdpConfig {
            grid_resolution: res,
            ..config.clone()
        };
        compute_grid_points(&feature_values_2, &config_2)?
    } else {
        compute_grid_points(&feature_values_2, config)?
    };

    let n_grid_1 = grid_values_1.len();
    let n_grid_2 = grid_values_2.len();

    // Compute 2D PDP values
    let mut pdp_values = vec![vec![0.0; n_grid_2]; n_grid_1];

    for (i, &grid_val_1) in grid_values_1.iter().enumerate() {
        for (j, &grid_val_2) in grid_values_2.iter().enumerate() {
            // Create modified data with both features set to grid values
            let mut modified_data = data.clone();
            for k in 0..n_samples {
                modified_data[[k, feature_index_1]] = grid_val_1;
                modified_data[[k, feature_index_2]] = grid_val_2;
            }

            // Get predictions and average
            let predictions = predict_fn(&modified_data);
            let mean: f64 = predictions.iter().sum::<f64>() / n_samples as f64;
            pdp_values[i][j] = mean;
        }
    }

    // Center if requested
    let centered = config.center;
    if centered {
        let total: f64 = pdp_values.iter().flat_map(|row| row.iter()).sum();
        let mean = total / (n_grid_1 * n_grid_2) as f64;
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

/// Compute grid points for PDP.
///
/// Either uses quantiles (default) or uniform spacing.
fn compute_grid_points(values: &[f64], config: &PdpConfig) -> EconResult<Vec<f64>> {
    if values.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Cannot compute grid points from empty values".to_string(),
        });
    }

    // Handle explicit quantiles
    if let Some(ref quantiles) = config.quantiles {
        return compute_quantile_values(values, quantiles);
    }

    let n_grid = config.grid_resolution.max(2);

    if config.use_quantiles {
        // Compute evenly spaced quantiles
        let probs: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        compute_quantile_values(values, &probs)
    } else {
        // Uniform spacing between min and max
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted[0];
        let max_val = sorted[sorted.len() - 1];

        if (max_val - min_val).abs() < 1e-10 {
            // All values are the same
            return Ok(vec![min_val]);
        }

        let step = (max_val - min_val) / (n_grid - 1) as f64;
        let grid: Vec<f64> = (0..n_grid).map(|i| min_val + i as f64 * step).collect();

        Ok(grid)
    }
}

/// Compute quantile values.
fn compute_quantile_values(values: &[f64], probs: &[f64]) -> EconResult<Vec<f64>> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut result = Vec::with_capacity(probs.len());

    for &p in probs {
        let p = p.clamp(0.0, 1.0);
        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let frac = index - lower as f64;

        let value = if lower == upper || upper >= n {
            sorted[lower.min(n - 1)]
        } else {
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        };
        result.push(value);
    }

    // Remove duplicates while preserving order
    let mut unique: Vec<f64> = Vec::with_capacity(result.len());
    for val in result {
        if unique.is_empty() || (val - *unique.last().unwrap()).abs() > 1e-10 {
            unique.push(val);
        }
    }

    Ok(unique)
}

/// Compute Individual Conditional Expectation (ICE) curves.
///
/// ICE curves show how each individual's prediction changes across the grid,
/// while PDP shows the average. ICE reveals heterogeneity in feature effects.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `feature_index` - Index of the feature to compute ICE for
/// * `predict_fn` - Function that takes feature matrix and returns predictions
/// * `config` - PDP configuration
///
/// # Returns
///
/// (grid_values, ice_curves) where ice_curves[i] is the curve for observation i
///
/// # References
///
/// - Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
///   "Peeking Inside the Black Box: Visualizing Statistical Learning With Plots
///   of Individual Conditional Expectation". Journal of Computational and
///   Graphical Statistics, 24(1), 44-65.
pub fn ice_curves<F>(
    data: &Array2<f64>,
    feature_index: usize,
    predict_fn: F,
    config: &PdpConfig,
) -> EconResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(&Array2<f64>) -> Vec<f64>,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "ICE computation requires data".to_string(),
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

    // Extract feature values and compute grid
    let feature_values: Vec<f64> = data.column(feature_index).iter().cloned().collect();
    let grid_values = compute_grid_points(&feature_values, config)?;
    let n_grid = grid_values.len();

    // Compute ICE curve for each observation
    let mut ice_curves = vec![vec![0.0; n_grid]; n_samples];

    for (g, &grid_val) in grid_values.iter().enumerate() {
        // Create modified data with feature set to grid value
        let mut modified_data = data.clone();
        for i in 0..n_samples {
            modified_data[[i, feature_index]] = grid_val;
        }

        // Get predictions
        let predictions = predict_fn(&modified_data);

        // Store in ICE curves
        for (i, &pred) in predictions.iter().enumerate() {
            ice_curves[i][g] = pred;
        }
    }

    // Center if requested (c-ICE)
    if config.center {
        for curve in &mut ice_curves {
            let first = curve[0];
            for val in curve.iter_mut() {
                *val -= first;
            }
        }
    }

    Ok((grid_values, ice_curves))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
            grid_resolution: 5,
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
            grid_resolution: 5,
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
            grid_resolution: 3,
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
            grid_resolution: 3,
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
    fn test_ice_curves() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let predict_fn = |x: &Array2<f64>| -> Vec<f64> {
            x.rows().into_iter().map(|row| row[0] + row[1]).collect()
        };

        let config = PdpConfig {
            grid_resolution: 3,
            use_quantiles: false,
            ..Default::default()
        };

        let (grid, curves) = ice_curves(&data, 0, predict_fn, &config).unwrap();

        // Should have 3 curves (one per observation)
        assert_eq!(curves.len(), 3);

        // Each curve should have 3 points
        for curve in &curves {
            assert_eq!(curve.len(), 3);
        }

        // ICE curve for first observation (x1=1):
        // At x0=1: 1+1=2, at x0=2: 2+1=3, at x0=3: 3+1=4
        assert!((curves[0][0] - 2.0).abs() < 1e-10);
        assert!((curves[0][1] - 3.0).abs() < 1e-10);
        assert!((curves[0][2] - 4.0).abs() < 1e-10);
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
    fn test_grid_points_uniform() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let config = PdpConfig {
            grid_resolution: 5,
            use_quantiles: false,
            ..Default::default()
        };

        let grid = compute_grid_points(&values, &config).unwrap();

        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 0.0).abs() < 1e-10);
        assert!((grid[4] - 4.0).abs() < 1e-10);
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
        };

        let output = format!("{}", result);
        assert!(output.contains("Partial Dependence Plot"));
        assert!(output.contains("feature_x"));
        assert!(output.contains("100"));
    }
}
