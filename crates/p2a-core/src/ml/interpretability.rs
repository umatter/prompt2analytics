//! Model interpretability methods.
//!
//! Implements ICE curves, LIME, and SHAP for explaining model predictions.
//!
//! ## Methods
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **ICE** | [`ice_curves`] | Individual Conditional Expectation curves |
//! | **LIME** | [`lime`] | Local Interpretable Model-agnostic Explanations |
//! | **SHAP** | [`shap_values`] | SHapley Additive exPlanations |
//!
//! ## References
//!
//! - Goldstein, A., et al. (2015). "Peeking Inside the Black Box: Visualizing
//!   Statistical Learning With Plots of Individual Conditional Expectation."
//!   *Journal of Computational and Graphical Statistics*, 24(1), 44-65.
//! - Ribeiro, M. T., et al. (2016). "Why Should I Trust You? Explaining the
//!   Predictions of Any Classifier." *KDD 2016*.
//! - Lundberg, S. M., & Lee, S. (2017). "A Unified Approach to Interpreting
//!   Model Predictions." *NeurIPS 2017*.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

// =============================================================================
// ICE Curves (Individual Conditional Expectation)
// =============================================================================

/// Result from ICE curve analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCurvesResult {
    /// Feature name
    pub feature_name: String,
    /// Grid values for the feature
    pub grid_values: Vec<f64>,
    /// ICE curves - each row is one observation's curve
    /// Shape: (n_samples, grid_resolution)
    pub ice_curves: Vec<Vec<f64>>,
    /// Partial dependence (mean of ICE curves)
    pub pd_values: Vec<f64>,
    /// Centered ICE curves (c-ICE) - each curve centered at its initial value
    pub centered_ice: Vec<Vec<f64>>,
    /// Standard deviation of predictions at each grid point
    pub std_values: Vec<f64>,
    /// Number of observations
    pub n_samples: usize,
}

impl std::fmt::Display for IceCurvesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ICE Curves Analysis")?;
        writeln!(f, "===================")?;
        writeln!(f, "Feature: {}", self.feature_name)?;
        writeln!(f, "Samples: {}", self.n_samples)?;
        writeln!(f, "Grid points: {}", self.grid_values.len())?;
        writeln!(f)?;

        writeln!(f, "Partial Dependence (mean of ICE curves):")?;
        writeln!(f, "{:<12} {:>12} {:>12}", "Value", "PD", "Std")?;
        writeln!(f, "{:-<38}", "")?;

        let n_display = self.grid_values.len().min(10);
        for i in 0..n_display {
            writeln!(
                f,
                "{:<12.4} {:>12.4} {:>12.4}",
                self.grid_values[i], self.pd_values[i], self.std_values[i]
            )?;
        }

        if self.grid_values.len() > 10 {
            writeln!(f, "... ({} more points)", self.grid_values.len() - 10)?;
        }

        Ok(())
    }
}

/// Compute Individual Conditional Expectation (ICE) curves.
///
/// ICE plots show how the predicted response changes for each individual observation
/// as a feature varies, unlike partial dependence which shows the average effect.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `predict_fn` - Function that takes a feature matrix and returns predictions
/// * `feature_idx` - Index of the feature to analyze
/// * `feature_name` - Name of the feature
/// * `grid_resolution` - Number of grid points (default: 50)
/// * `sample_frac` - Fraction of samples to use for ICE curves (default: 1.0)
/// * `seed` - Random seed for sampling
///
/// # Returns
///
/// ICE curves result containing individual curves and the average (PD).
///
/// # References
///
/// Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
/// "Peeking Inside the Black Box: Visualizing Statistical Learning With Plots
/// of Individual Conditional Expectation." *J. Comp. Graph. Stat.*, 24(1), 44-65.
pub fn ice_curves<F>(
    data: ArrayView2<f64>,
    predict_fn: F,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: Option<usize>,
    sample_frac: Option<f64>,
    seed: Option<u64>,
) -> EconResult<IceCurvesResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if feature_idx >= n_features {
        return Err(EconError::Computation(format!(
            "Feature index {} out of bounds ({})",
            feature_idx, n_features
        )));
    }

    if n_samples == 0 {
        return Err(EconError::EmptyDataset);
    }

    let grid_res = grid_resolution.unwrap_or(50);
    let frac = sample_frac.unwrap_or(1.0).clamp(0.0, 1.0);

    // Select samples to use
    let sample_indices: Vec<usize> = if frac < 1.0 {
        let mut rng = seed.unwrap_or(42);
        let n_use = ((n_samples as f64) * frac).max(1.0) as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
        indices.truncate(n_use);
        indices
    } else {
        (0..n_samples).collect()
    };

    let n_use = sample_indices.len();

    // Generate grid values for the feature
    let col = data.column(feature_idx);
    let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let step = if grid_res > 1 {
        (max_val - min_val) / (grid_res - 1) as f64
    } else {
        0.0
    };

    let grid_values: Vec<f64> = (0..grid_res).map(|i| min_val + step * i as f64).collect();

    // Compute ICE curves
    let mut ice_curves: Vec<Vec<f64>> = Vec::with_capacity(n_use);
    let mut pd_values = vec![0.0; grid_res];
    let mut sq_values = vec![0.0; grid_res];

    for &sample_idx in &sample_indices {
        let mut curve = Vec::with_capacity(grid_res);

        for (grid_idx, &grid_val) in grid_values.iter().enumerate() {
            // Create modified data point
            let mut modified = Array2::zeros((1, n_features));
            for j in 0..n_features {
                modified[[0, j]] = if j == feature_idx {
                    grid_val
                } else {
                    data[[sample_idx, j]]
                };
            }

            let pred = predict_fn(modified.view());
            let pred_val = pred.first().copied().unwrap_or(0.0);
            curve.push(pred_val);

            pd_values[grid_idx] += pred_val;
            sq_values[grid_idx] += pred_val * pred_val;
        }

        ice_curves.push(curve);
    }

    // Average for PD
    let n_use_f = n_use as f64;
    for i in 0..grid_res {
        pd_values[i] /= n_use_f;
    }

    // Standard deviation
    let std_values: Vec<f64> = (0..grid_res)
        .map(|i| {
            let variance = sq_values[i] / n_use_f - pd_values[i].powi(2);
            variance.max(0.0).sqrt()
        })
        .collect();

    // Centered ICE (c-ICE): center each curve at its initial value
    let centered_ice: Vec<Vec<f64>> = ice_curves
        .iter()
        .map(|curve| {
            let initial = curve.first().copied().unwrap_or(0.0);
            curve.iter().map(|&v| v - initial).collect()
        })
        .collect();

    Ok(IceCurvesResult {
        feature_name: feature_name.to_string(),
        grid_values,
        ice_curves,
        pd_values,
        centered_ice,
        std_values,
        n_samples: n_use,
    })
}

// =============================================================================
// LIME (Local Interpretable Model-agnostic Explanations)
// =============================================================================

/// Result from LIME explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimeResult {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Local feature importance (coefficients of the local linear model)
    pub importance: Vec<f64>,
    /// Standard errors of importance estimates
    pub std_errors: Vec<f64>,
    /// Intercept of the local model
    pub intercept: f64,
    /// Local R-squared (how well the local model fits)
    pub local_r2: f64,
    /// Prediction for the explained instance
    pub prediction: f64,
    /// Explained instance index
    pub instance_idx: usize,
    /// Number of perturbed samples used
    pub n_samples: usize,
}

impl std::fmt::Display for LimeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LIME Explanation")?;
        writeln!(f, "================")?;
        writeln!(f, "Instance: {}", self.instance_idx)?;
        writeln!(f, "Prediction: {:.4}", self.prediction)?;
        writeln!(f, "Local R-squared: {:.4}", self.local_r2)?;
        writeln!(f, "Perturbed samples: {}", self.n_samples)?;
        writeln!(f)?;

        writeln!(f, "Feature Importances:")?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12}",
            "Feature", "Importance", "Std.Err"
        )?;
        writeln!(f, "{:-<46}", "")?;

        // Sort by absolute importance
        let mut indexed: Vec<(usize, f64)> = self
            .importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (idx, imp) in indexed.iter().take(10) {
            let name = self
                .feature_names
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("X{}", idx));
            let se = self.std_errors.get(*idx).copied().unwrap_or(0.0);
            writeln!(f, "{:<20} {:>12.4} {:>12.4}", name, imp, se)?;
        }

        Ok(())
    }
}

/// LIME: Local Interpretable Model-agnostic Explanations.
///
/// Explains individual predictions by fitting a locally-weighted linear model
/// around the instance of interest.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `predict_fn` - Function that takes a feature matrix and returns predictions
/// * `instance_idx` - Index of the instance to explain
/// * `n_samples` - Number of perturbed samples to generate (default: 1000)
/// * `kernel_width` - Width of the exponential kernel (default: auto)
/// * `feature_names` - Optional feature names
/// * `seed` - Random seed
///
/// # Returns
///
/// LIME result with local feature importances.
///
/// # References
///
/// Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
/// "Why Should I Trust You? Explaining the Predictions of Any Classifier."
/// *KDD 2016*.
pub fn lime<F>(
    data: ArrayView2<f64>,
    predict_fn: F,
    instance_idx: usize,
    n_samples: Option<usize>,
    kernel_width: Option<f64>,
    feature_names: Option<Vec<String>>,
    seed: Option<u64>,
) -> EconResult<LimeResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    let n_obs = data.nrows();
    let n_features = data.ncols();

    if instance_idx >= n_obs {
        return Err(EconError::Computation(format!(
            "Instance index {} out of bounds ({})",
            instance_idx, n_obs
        )));
    }

    let n_perturb = n_samples.unwrap_or(1000);
    let mut rng = seed.unwrap_or(42);

    // Get the instance to explain
    let instance = data.row(instance_idx);
    let original_pred = predict_fn(instance.to_owned().insert_axis(Axis(0)).view());
    let original_pred_val = original_pred.first().copied().unwrap_or(0.0);

    // Compute feature statistics for perturbation
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    for j in 0..n_features {
        let col = data.column(j);
        let mean: f64 = col.iter().sum::<f64>() / n_obs as f64;
        let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_obs as f64;
        means[j] = mean;
        stds[j] = variance.sqrt().max(1e-10);
    }

    // Default kernel width is sqrt(n_features) * 0.75 (common heuristic)
    let sigma = kernel_width.unwrap_or_else(|| (n_features as f64).sqrt() * 0.75);

    // Generate perturbed samples
    let mut perturbed = Array2::zeros((n_perturb, n_features));
    let mut distances = Vec::with_capacity(n_perturb);

    for i in 0..n_perturb {
        let mut sq_dist = 0.0;
        for j in 0..n_features {
            // Box-Muller transform for normal distribution
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (rng as f64) / (u64::MAX as f64);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng as f64) / (u64::MAX as f64);

            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Perturb around the original instance
            let perturbed_val = instance[j] + z * stds[j];
            perturbed[[i, j]] = perturbed_val;

            // Normalized distance
            let diff = (perturbed_val - instance[j]) / stds[j];
            sq_dist += diff * diff;
        }
        distances.push(sq_dist.sqrt());
    }

    // Get predictions for perturbed samples
    let predictions = predict_fn(perturbed.view());

    // Compute kernel weights
    let weights: Vec<f64> = distances
        .iter()
        .map(|&d| (-d * d / (2.0 * sigma * sigma)).exp())
        .collect();

    // Fit weighted linear regression
    // Normalize features for the local model
    let mut x_normalized = Array2::zeros((n_perturb, n_features + 1));
    for i in 0..n_perturb {
        x_normalized[[i, 0]] = 1.0; // Intercept
        for j in 0..n_features {
            x_normalized[[i, j + 1]] = (perturbed[[i, j]] - instance[j]) / stds[j];
        }
    }

    // Weighted least squares: (X'WX)^-1 X'Wy
    let y: Array1<f64> = Array1::from_vec(predictions);

    // Apply weights
    let sqrt_weights: Vec<f64> = weights.iter().map(|&w| w.sqrt()).collect();

    let mut x_weighted = Array2::zeros((n_perturb, n_features + 1));
    let mut y_weighted = Array1::zeros(n_perturb);

    for i in 0..n_perturb {
        let sw = sqrt_weights[i];
        y_weighted[i] = y[i] * sw;
        for j in 0..=n_features {
            x_weighted[[i, j]] = x_normalized[[i, j]] * sw;
        }
    }

    // Solve via normal equations
    let xtx = x_weighted.t().dot(&x_weighted);
    let xty = x_weighted.t().dot(&y_weighted);

    // Simple matrix inverse using Cramer's rule / Gaussian elimination
    let coefficients = solve_linear_system(&xtx, &xty)?;

    let intercept = coefficients[0];
    let importance: Vec<f64> = coefficients[1..].to_vec();

    // Compute fitted values and R-squared
    let y_pred: Array1<f64> = x_weighted.dot(&Array1::from_vec(coefficients.clone()));
    let y_mean = y_weighted.sum() / n_perturb as f64;

    let ss_tot: f64 = y_weighted.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y_weighted
        .iter()
        .zip(y_pred.iter())
        .map(|(&yi, &ypi)| (yi - ypi).powi(2))
        .sum();

    let local_r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // Compute standard errors (approximate)
    let dof = n_perturb.saturating_sub(n_features + 1);
    let mse = if dof > 0 { ss_res / dof as f64 } else { 0.0 };

    // Diagonal of (X'WX)^-1 for standard errors
    let xtx_inv = invert_matrix(&xtx)?;
    let std_errors: Vec<f64> = (1..=n_features)
        .map(|j| (xtx_inv[[j, j]].abs() * mse).sqrt())
        .collect();

    let names =
        feature_names.unwrap_or_else(|| (0..n_features).map(|i| format!("X{}", i)).collect());

    Ok(LimeResult {
        feature_names: names,
        importance,
        std_errors,
        intercept,
        local_r2,
        prediction: original_pred_val,
        instance_idx,
        n_samples: n_perturb,
    })
}

// =============================================================================
// SHAP Values (SHapley Additive exPlanations)
// =============================================================================

/// Result from SHAP value computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapResult {
    /// Feature names
    pub feature_names: Vec<String>,
    /// SHAP values for each feature (for a single instance or averaged)
    pub shap_values: Vec<f64>,
    /// Base value (expected prediction over the background data)
    pub base_value: f64,
    /// Prediction for the explained instance(s)
    pub prediction: f64,
    /// Number of samples used for background
    pub n_background: usize,
    /// Instance index (if single instance) or -1 for global
    pub instance_idx: i64,
}

impl std::fmt::Display for ShapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Values")?;
        writeln!(f, "===========")?;
        if self.instance_idx >= 0 {
            writeln!(f, "Instance: {}", self.instance_idx)?;
        } else {
            writeln!(f, "Global (averaged)")?;
        }
        writeln!(f, "Base value: {:.4}", self.base_value)?;
        writeln!(f, "Prediction: {:.4}", self.prediction)?;
        writeln!(
            f,
            "Sum of SHAP + base: {:.4}",
            self.base_value + self.shap_values.iter().sum::<f64>()
        )?;
        writeln!(f)?;

        writeln!(f, "Feature Contributions:")?;
        writeln!(f, "{:<20} {:>12}", "Feature", "SHAP")?;
        writeln!(f, "{:-<34}", "")?;

        // Sort by absolute SHAP value
        let mut indexed: Vec<(usize, f64)> = self
            .shap_values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (idx, shap) in indexed.iter().take(10) {
            let name = self
                .feature_names
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("X{}", idx));
            writeln!(f, "{:<20} {:>12.4}", name, shap)?;
        }

        Ok(())
    }
}

/// Compute SHAP values using Kernel SHAP (model-agnostic).
///
/// SHAP values provide a theoretically grounded way to explain predictions
/// based on game-theoretic Shapley values.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples x n_features)
/// * `predict_fn` - Function that takes a feature matrix and returns predictions
/// * `instance_idx` - Index of the instance to explain
/// * `n_background` - Number of background samples (default: min(100, n_samples))
/// * `n_samples` - Number of coalition samples for SHAP estimation (default: 2048)
/// * `feature_names` - Optional feature names
/// * `seed` - Random seed
///
/// # Returns
///
/// SHAP result with feature contributions.
///
/// # Note
///
/// This is a simplified Kernel SHAP implementation. For tree models,
/// dedicated TreeSHAP would be more efficient.
///
/// # References
///
/// Lundberg, S. M., & Lee, S. (2017).
/// "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*.
pub fn shap_values<F>(
    data: ArrayView2<f64>,
    predict_fn: F,
    instance_idx: usize,
    n_background: Option<usize>,
    n_samples: Option<usize>,
    feature_names: Option<Vec<String>>,
    seed: Option<u64>,
) -> EconResult<ShapResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    let n_obs = data.nrows();
    let n_features = data.ncols();

    if instance_idx >= n_obs {
        return Err(EconError::Computation(format!(
            "Instance index {} out of bounds ({})",
            instance_idx, n_obs
        )));
    }

    let n_bg = n_background.unwrap_or_else(|| n_obs.min(100));
    let n_coal = n_samples.unwrap_or(2048);
    let mut rng = seed.unwrap_or(42);

    // Get the instance to explain
    let instance = data.row(instance_idx);
    let original_pred = predict_fn(instance.to_owned().insert_axis(Axis(0)).view());
    let original_pred_val = original_pred.first().copied().unwrap_or(0.0);

    // Sample background data
    let bg_indices: Vec<usize> = if n_bg >= n_obs {
        (0..n_obs).collect()
    } else {
        let mut indices: Vec<usize> = (0..n_obs).collect();
        for i in (1..n_obs).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
        indices.truncate(n_bg);
        indices
    };

    let n_bg_actual = bg_indices.len();

    // Compute base value (expected prediction over background)
    let bg_data: Array2<f64> =
        Array2::from_shape_fn((n_bg_actual, n_features), |(i, j)| data[[bg_indices[i], j]]);
    let bg_preds = predict_fn(bg_data.view());
    let base_value: f64 = bg_preds.iter().sum::<f64>() / n_bg_actual as f64;

    // Kernel SHAP: sample coalitions and fit weighted regression
    // For each coalition S, compute E[f(x) | x_S = x_S^instance]
    // Weight by Shapley kernel: (M-1) / (C(M, |S|) * |S| * (M - |S|))

    let mut coalition_matrix = Array2::zeros((n_coal, n_features));
    let mut coalition_preds = Vec::with_capacity(n_coal);
    let mut kernel_weights = Vec::with_capacity(n_coal);

    let m = n_features;
    let m_f = m as f64;

    for i in 0..n_coal {
        // Generate random coalition (subset of features)
        let mut coalition_size = 0;
        let mut coalition = vec![false; n_features];

        for j in 0..n_features {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 63) == 1 {
                coalition[j] = true;
                coalition_size += 1;
            }
        }

        // Ensure we have at least one feature in or out
        if coalition_size == 0 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % n_features;
            coalition[j] = true;
            coalition_size = 1;
        } else if coalition_size == n_features {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % n_features;
            coalition[j] = false;
            coalition_size = n_features - 1;
        }

        // Store coalition as binary vector
        for j in 0..n_features {
            coalition_matrix[[i, j]] = if coalition[j] { 1.0 } else { 0.0 };
        }

        // Compute kernel weight
        // w(z) = (M-1) / (C(M,|z|) * |z| * (M-|z|))
        let s = coalition_size;
        let weight = if s > 0 && s < m {
            (m_f - 1.0) / (binomial(m, s) as f64 * s as f64 * (m - s) as f64)
        } else {
            1.0
        };
        kernel_weights.push(weight);

        // Compute expected prediction for this coalition
        // Average over background samples, replacing non-coalition features with background values
        let mut coalition_pred = 0.0;
        for &bg_idx in &bg_indices {
            let mut sample = Array2::zeros((1, n_features));
            for j in 0..n_features {
                sample[[0, j]] = if coalition[j] {
                    instance[j]
                } else {
                    data[[bg_idx, j]]
                };
            }
            let pred = predict_fn(sample.view());
            coalition_pred += pred.first().copied().unwrap_or(0.0);
        }
        coalition_pred /= n_bg_actual as f64;
        coalition_preds.push(coalition_pred);
    }

    // Fit weighted linear regression to get SHAP values
    // Target: coalition_preds - base_value
    // Features: coalition_matrix
    // Weights: kernel_weights

    let y: Array1<f64> = Array1::from_iter(coalition_preds.iter().map(|&p| p - base_value));

    // Apply weights
    let sqrt_weights: Vec<f64> = kernel_weights.iter().map(|&w| w.sqrt()).collect();

    let mut x_weighted = Array2::zeros((n_coal, n_features));
    let mut y_weighted = Array1::zeros(n_coal);

    for i in 0..n_coal {
        let sw = sqrt_weights[i];
        y_weighted[i] = y[i] * sw;
        for j in 0..n_features {
            x_weighted[[i, j]] = coalition_matrix[[i, j]] * sw;
        }
    }

    // Solve via normal equations
    let xtx = x_weighted.t().dot(&x_weighted);
    let xty = x_weighted.t().dot(&y_weighted);

    let shap_values = solve_linear_system(&xtx, &xty)?;

    let names =
        feature_names.unwrap_or_else(|| (0..n_features).map(|i| format!("X{}", i)).collect());

    Ok(ShapResult {
        feature_names: names,
        shap_values,
        base_value,
        prediction: original_pred_val,
        n_background: n_bg_actual,
        instance_idx: instance_idx as i64,
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Solve linear system Ax = b using Cholesky decomposition
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> EconResult<Vec<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(EconError::Computation("Dimension mismatch".to_string()));
    }

    // Add small regularization for numerical stability
    let mut a_reg = a.clone();
    for i in 0..n {
        a_reg[[i, i]] += 1e-8;
    }

    // Cholesky decomposition A = L * L^T
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag = a_reg[[j, j]] - sum;
                if diag <= 0.0 {
                    return Err(EconError::Computation(
                        "Matrix not positive definite".to_string(),
                    ));
                }
                l[[j, j]] = diag.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a_reg[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Solve L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / l[[i, i]];
    }

    // Solve L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    Ok(x)
}

/// Invert matrix using Cholesky decomposition
fn invert_matrix(a: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = a.nrows();

    // Add small regularization
    let mut a_reg = a.clone();
    for i in 0..n {
        a_reg[[i, i]] += 1e-8;
    }

    let mut inv = Array2::zeros((n, n));

    // Solve for each column of identity
    for col in 0..n {
        let mut b = Array1::zeros(n);
        b[col] = 1.0;

        let x = solve_linear_system(&a_reg, &b)?;
        for row in 0..n {
            inv[[row, col]] = x[row];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ice_curves_basic() {
        // Simple data where y = x[0] + noise
        let data = array![[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5],];

        // Simple linear predictor
        let predict_fn = |x: ArrayView2<f64>| -> Vec<f64> {
            (0..x.nrows())
                .map(|i| x[[i, 0]] + 0.1 * x[[i, 1]])
                .collect()
        };

        let result =
            ice_curves(data.view(), predict_fn, 0, "x0", Some(10), None, Some(42)).unwrap();

        assert_eq!(result.feature_name, "x0");
        assert_eq!(result.grid_values.len(), 10);
        assert_eq!(result.ice_curves.len(), 5);
        assert_eq!(result.pd_values.len(), 10);

        // PD should increase with x0 (since y = x0 + 0.1*x1)
        assert!(result.pd_values.last().unwrap() > result.pd_values.first().unwrap());
    }

    #[test]
    fn test_lime_basic() {
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0],];

        // y = 2*x0 + x1
        let predict_fn = |x: ArrayView2<f64>| -> Vec<f64> {
            (0..x.nrows())
                .map(|i| 2.0 * x[[i, 0]] + x[[i, 1]])
                .collect()
        };

        let result = lime(data.view(), predict_fn, 2, Some(100), None, None, Some(42)).unwrap();

        assert_eq!(result.instance_idx, 2);
        assert_eq!(result.importance.len(), 2);
        // For a linear model, LIME should recover approximately the true coefficients
        // (allowing for some noise due to the local approximation)
    }

    #[test]
    fn test_shap_values_basic() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];

        // y = x0 + x1
        let predict_fn = |x: ArrayView2<f64>| -> Vec<f64> {
            (0..x.nrows()).map(|i| x[[i, 0]] + x[[i, 1]]).collect()
        };

        let result = shap_values(
            data.view(),
            predict_fn,
            2,
            Some(5),
            Some(500),
            None,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.instance_idx, 2);
        assert_eq!(result.shap_values.len(), 2);

        // SHAP values should sum to prediction - base_value (approximately)
        let shap_sum: f64 = result.shap_values.iter().sum();
        let expected_diff = result.prediction - result.base_value;
        assert!(
            (shap_sum - expected_diff).abs() < 1.0,
            "SHAP sum ({}) should be close to pred - base ({})",
            shap_sum,
            expected_diff
        );
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 3), 120);
    }

    #[test]
    fn test_centered_ice() {
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0],];

        let predict_fn =
            |x: ArrayView2<f64>| -> Vec<f64> { (0..x.nrows()).map(|i| x[[i, 0]] * 2.0).collect() };

        let result = ice_curves(data.view(), predict_fn, 0, "x", Some(5), None, Some(42)).unwrap();

        // Centered ICE curves should start at 0
        for curve in &result.centered_ice {
            assert!(curve[0].abs() < 1e-10, "Centered ICE should start at 0");
        }
    }
}
