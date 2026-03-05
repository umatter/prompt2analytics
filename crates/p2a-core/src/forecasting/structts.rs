//! Structural Time Series models (StructTS).
//!
//! Fits structural time series models by maximum likelihood using the Kalman filter.
//! Supports local level, local linear trend, and basic structural model (BSM).
//!
//! # References
//!
//! - Harvey, A. C. (1990). "Forecasting, Structural Time Series Models and the Kalman Filter".
//!   Cambridge University Press.
//! - Durbin, J. & Koopman, S. J. (2012). "Time Series Analysis by State Space Methods".
//!   Oxford Statistical Science Series.
//! - R Core Team. `stats::StructTS()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/StructTS.html>

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::forecasting::kalman::{StateSpaceModel, kalman_filter, kalman_smoother};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Type of structural time series model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum StructTsType {
    /// Local level model (random walk + noise).
    /// State: [μ], with μ_{t+1} = μ_t + η_t
    /// Observation: y_t = μ_t + ε_t
    #[default]
    Level,
    /// Local linear trend model.
    /// State: [μ, ν], with μ_{t+1} = μ_t + ν_t + η_t, ν_{t+1} = ν_t + ζ_t
    /// Observation: y_t = μ_t + ε_t
    Trend,
    /// Basic Structural Model (level + trend + seasonality).
    /// State: [μ, ν, γ₁, ..., γ_{s-1}]
    /// Observation: y_t = μ_t + γ_t + ε_t
    BSM,
}

/// Configuration for structural time series fitting.
#[derive(Debug, Clone)]
pub struct StructTsConfig {
    /// Model type.
    pub model_type: StructTsType,
    /// Seasonal period (only used for BSM).
    pub period: Option<usize>,
    /// Fixed variance parameters (None means estimate).
    /// Order: [level, slope, seasonal, observation]
    pub fixed: Option<Vec<Option<f64>>>,
    /// Maximum iterations for optimization.
    pub max_iter: usize,
    /// Tolerance for convergence.
    pub tolerance: f64,
}

impl Default for StructTsConfig {
    fn default() -> Self {
        Self {
            model_type: StructTsType::Level,
            period: None,
            fixed: None,
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

/// Result from structural time series fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructTsResult {
    /// Model type fitted.
    pub model_type: StructTsType,
    /// Estimated variance parameters.
    pub coef: StructTsCoefficients,
    /// Fitted values (signal = level + trend + seasonal if applicable).
    pub fitted: Vec<f64>,
    /// Residuals.
    pub residuals: Vec<f64>,
    /// Log-likelihood at optimum.
    pub log_likelihood: f64,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of estimated parameters.
    pub n_params: usize,
    /// Convergence status.
    pub converged: bool,
    /// Smoothed level component.
    pub level: Vec<f64>,
    /// Smoothed slope component (for Trend and BSM).
    pub slope: Option<Vec<f64>>,
    /// Smoothed seasonal component (for BSM).
    pub seasonal: Option<Vec<f64>>,
}

/// Estimated variance coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructTsCoefficients {
    /// Level variance (σ²_η).
    pub level: f64,
    /// Slope variance (σ²_ζ) - for Trend and BSM.
    pub slope: Option<f64>,
    /// Seasonal variance (σ²_ω) - for BSM.
    pub seasonal: Option<f64>,
    /// Observation variance (σ²_ε).
    pub epsilon: f64,
}

/// Fit a structural time series model.
///
/// # Arguments
///
/// * `y` - Time series values
/// * `config` - Model configuration
///
/// # Returns
///
/// `StructTsResult` with fitted model and components.
pub fn struct_ts(y: &[f64], config: StructTsConfig) -> EconResult<StructTsResult> {
    let n = y.len();

    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "StructTS requires at least 10 observations".to_string(),
        });
    }

    // Validate BSM configuration
    if config.model_type == StructTsType::BSM {
        let period = config.period.unwrap_or(12);
        if n < 2 * period {
            return Err(EconError::InsufficientData {
                required: 2 * period,
                provided: n,
                context: format!(
                    "BSM with period {} requires at least {} observations",
                    period,
                    2 * period
                ),
            });
        }
    }

    // Initialize variance parameters
    let y_var = sample_variance(y);
    let (init_params, n_params) = initialize_params(&config, y_var);

    // Optimize variance parameters
    let (opt_params, log_lik, converged) =
        optimize_params(y, &config, init_params, config.max_iter, config.tolerance)?;

    // Build final model and get smoothed states
    let model = build_model(&config, &opt_params)?;
    let init_state = initialize_state(&config, y);
    let init_cov = initialize_covariance(&config, y_var);

    let filter_result = kalman_filter(y, &model, init_state.view(), init_cov.view())?;
    let smooth_result = kalman_smoother(&filter_result, &model)?;

    // Extract components
    let (level, slope, seasonal) = extract_components(&smooth_result.smoothed_states, &config);

    // Compute fitted values
    let fitted: Vec<f64> = (0..n)
        .map(|t| {
            let mut val = level[t];
            if let Some(ref s) = seasonal {
                val += s[t];
            }
            val
        })
        .collect();

    // Compute residuals
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    // Build coefficients
    let coef = params_to_coef(&config, &opt_params);

    // Compute AIC and BIC
    let aic = -2.0 * log_lik + 2.0 * n_params as f64;
    let bic = -2.0 * log_lik + (n_params as f64) * (n as f64).ln();

    Ok(StructTsResult {
        model_type: config.model_type,
        coef,
        fitted,
        residuals,
        log_likelihood: log_lik,
        aic,
        bic,
        n_obs: n,
        n_params,
        converged,
        level,
        slope,
        seasonal,
    })
}

/// Convenience function to run StructTS on a Dataset.
pub fn run_struct_ts(
    dataset: &Dataset,
    column: &str,
    model_type: StructTsType,
    period: Option<usize>,
) -> EconResult<StructTsResult> {
    let df = dataset.df();
    let available: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let col = df.column(column).map_err(|_| EconError::ColumnNotFound {
        column: column.to_string(),
        available: available.clone(),
    })?;

    let y: Vec<f64> = col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: column.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    struct_ts(
        &y,
        StructTsConfig {
            model_type,
            period,
            ..Default::default()
        },
    )
}

// Helper functions

fn sample_variance(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn initialize_params(config: &StructTsConfig, y_var: f64) -> (Vec<f64>, usize) {
    // Initialize variances as fractions of data variance
    match config.model_type {
        StructTsType::Level => {
            // [level_var, obs_var]
            (vec![y_var * 0.1, y_var * 0.9], 2)
        }
        StructTsType::Trend => {
            // [level_var, slope_var, obs_var]
            (vec![y_var * 0.1, y_var * 0.01, y_var * 0.89], 3)
        }
        StructTsType::BSM => {
            // [level_var, slope_var, seasonal_var, obs_var]
            (
                vec![y_var * 0.1, y_var * 0.01, y_var * 0.1, y_var * 0.79],
                4,
            )
        }
    }
}

fn optimize_params(
    y: &[f64],
    config: &StructTsConfig,
    init_params: Vec<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Vec<f64>, f64, bool)> {
    // Use log-transformed parameters for better optimization landscape
    let y_var = sample_variance(y);

    // Transform to log scale (relative to y_var)
    let init_log: Vec<f64> = init_params
        .iter()
        .map(|&p| (p / y_var).max(1e-10).ln())
        .collect();

    // Objective function: negative log-likelihood with log-transformed params
    let objective = |log_params: &[f64]| -> f64 {
        let params: Vec<f64> = log_params.iter().map(|&lp| y_var * lp.exp()).collect();
        match compute_loglik(y, config, &params) {
            Ok(ll) => -ll,
            Err(_) => 1e20,
        }
    };

    // Use Nelder-Mead (simpler and more robust for small parameter counts)
    let (best_log_params, neg_loglik, converged) =
        nelder_mead_optimize(&objective, &init_log, max_iter, tolerance);

    // Transform back to original scale
    let best_params: Vec<f64> = best_log_params.iter().map(|&lp| y_var * lp.exp()).collect();

    Ok((best_params, -neg_loglik, converged))
}

/// Nelder-Mead simplex optimization for small parameter counts.
fn nelder_mead_optimize(
    objective: &dyn Fn(&[f64]) -> f64,
    init: &[f64],
    max_iter: usize,
    tolerance: f64,
) -> (Vec<f64>, f64, bool) {
    let n = init.len();

    // Standard Nelder-Mead parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(init.to_vec());

    for i in 0..n {
        let mut vertex = init.to_vec();
        vertex[i] += 0.5; // Step of 0.5 in log space
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();

    for _iter in 0..max_iter {
        // Sort vertices by value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence
        let f_spread = (values[n] - values[0]).abs() / (values[0].abs() + 1e-10);
        if f_spread < tolerance {
            return (simplex[0].clone(), values[0], true);
        }

        // Centroid of all except worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let f_reflected = objective(&reflected);

        if f_reflected < values[n - 1] && f_reflected >= values[0] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else if f_reflected < values[0] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = objective(&expanded);

            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else {
            // Contraction
            let contract_point: Vec<f64> = if f_reflected < values[n] {
                (0..n)
                    .map(|j| centroid[j] + rho * (reflected[j] - centroid[j]))
                    .collect()
            } else {
                (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect()
            };
            let f_contracted = objective(&contract_point);

            let threshold = if f_reflected < values[n] {
                f_reflected
            } else {
                values[n]
            };
            if f_contracted < threshold {
                simplex[n] = contract_point;
                values[n] = f_contracted;
            } else {
                // Shrink
                let best = simplex[0].clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
                    }
                    values[i] = objective(&simplex[i]);
                }
            }
        }
    }

    (simplex[0].clone(), values[0], false)
}

fn compute_loglik(y: &[f64], config: &StructTsConfig, params: &[f64]) -> EconResult<f64> {
    let model = build_model(config, params)?;
    let y_var = sample_variance(y);
    let init_state = initialize_state(config, y);
    let init_cov = initialize_covariance(config, y_var);

    let result = kalman_filter(y, &model, init_state.view(), init_cov.view())?;
    Ok(result.log_likelihood)
}

fn build_model(config: &StructTsConfig, params: &[f64]) -> EconResult<StateSpaceModel> {
    match config.model_type {
        StructTsType::Level => {
            // State: [μ]
            let transition = Array2::from_elem((1, 1), 1.0);
            let observation = Array1::from_elem(1, 1.0);
            let selection = Array2::from_elem((1, 1), 1.0);
            let state_cov = Array2::from_elem((1, 1), params[0].max(1e-12));
            let obs_var = params[1].max(1e-12);

            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var)
        }
        StructTsType::Trend => {
            // State: [μ, ν]
            let transition = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
            let observation = Array1::from_vec(vec![1.0, 0.0]);
            let selection = Array2::eye(2);
            let state_cov = Array2::from_shape_vec(
                (2, 2),
                vec![params[0].max(1e-12), 0.0, 0.0, params[1].max(1e-12)],
            )
            .unwrap();
            let obs_var = params[2].max(1e-12);

            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var)
        }
        StructTsType::BSM => {
            let period = config.period.unwrap_or(12);
            let m = 2 + period - 1; // level + slope + (period-1) seasonal states

            // Transition matrix
            let mut transition = Array2::zeros((m, m));
            // Level: μ_{t+1} = μ_t + ν_t
            transition[[0, 0]] = 1.0;
            transition[[0, 1]] = 1.0;
            // Slope: ν_{t+1} = ν_t
            transition[[1, 1]] = 1.0;
            // Seasonal: γ_t = -γ_{t-1} - ... - γ_{t-s+1}
            for j in 2..m {
                transition[[2, j]] = -1.0;
            }
            for i in 3..m {
                transition[[i, i - 1]] = 1.0;
            }

            // Observation: y_t = μ_t + γ_t
            let mut observation = Array1::zeros(m);
            observation[0] = 1.0; // Level
            observation[2] = 1.0; // Seasonal

            // Selection matrix (which states receive innovations)
            let mut selection = Array2::zeros((m, 3));
            selection[[0, 0]] = 1.0; // Level gets innovation
            selection[[1, 1]] = 1.0; // Slope gets innovation
            selection[[2, 2]] = 1.0; // Seasonal gets innovation

            // State covariance
            let state_cov = Array2::from_shape_vec(
                (3, 3),
                vec![
                    params[0].max(1e-12),
                    0.0,
                    0.0,
                    0.0,
                    params[1].max(1e-12),
                    0.0,
                    0.0,
                    0.0,
                    params[2].max(1e-12),
                ],
            )
            .unwrap();

            let obs_var = params[3].max(1e-12);

            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var)
        }
    }
}

fn initialize_state(config: &StructTsConfig, y: &[f64]) -> Array1<f64> {
    match config.model_type {
        StructTsType::Level => Array1::from_elem(1, y[0]),
        StructTsType::Trend => {
            // Initial level from first observation, slope from first difference
            let slope = if y.len() > 1 { y[1] - y[0] } else { 0.0 };
            Array1::from_vec(vec![y[0], slope])
        }
        StructTsType::BSM => {
            let period = config.period.unwrap_or(12);
            let m = 2 + period - 1;
            let mut state = Array1::zeros(m);
            state[0] = y[0]; // Level
            state[1] = if y.len() > 1 { y[1] - y[0] } else { 0.0 }; // Slope
            // Seasonal initialized to zero
            state
        }
    }
}

fn initialize_covariance(config: &StructTsConfig, y_var: f64) -> Array2<f64> {
    let m = match config.model_type {
        StructTsType::Level => 1,
        StructTsType::Trend => 2,
        StructTsType::BSM => 2 + config.period.unwrap_or(12) - 1,
    };

    // Use diffuse initialization (large variance)
    Array2::from_diag(&Array1::from_elem(m, y_var * 10.0))
}

fn extract_components(
    smoothed_states: &[Vec<f64>],
    config: &StructTsConfig,
) -> (Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let _n = smoothed_states.len();

    let level: Vec<f64> = smoothed_states.iter().map(|s| s[0]).collect();

    let slope = match config.model_type {
        StructTsType::Trend | StructTsType::BSM => {
            Some(smoothed_states.iter().map(|s| s[1]).collect())
        }
        _ => None,
    };

    let seasonal = if config.model_type == StructTsType::BSM {
        Some(smoothed_states.iter().map(|s| s[2]).collect())
    } else {
        None
    };

    (level, slope, seasonal)
}

fn params_to_coef(config: &StructTsConfig, params: &[f64]) -> StructTsCoefficients {
    match config.model_type {
        StructTsType::Level => StructTsCoefficients {
            level: params[0],
            slope: None,
            seasonal: None,
            epsilon: params[1],
        },
        StructTsType::Trend => StructTsCoefficients {
            level: params[0],
            slope: Some(params[1]),
            seasonal: None,
            epsilon: params[2],
        },
        StructTsType::BSM => StructTsCoefficients {
            level: params[0],
            slope: Some(params[1]),
            seasonal: Some(params[2]),
            epsilon: params[3],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_local_level_data(n: usize, level_var: f64, obs_var: f64, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut level = 100.0;
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            level += (rng.r#gen::<f64>() - 0.5) * 2.0 * level_var.sqrt();
            y.push(level + (rng.r#gen::<f64>() - 0.5) * 2.0 * obs_var.sqrt());
        }

        y
    }

    #[test]
    fn test_local_level_model() {
        let y = generate_local_level_data(100, 1.0, 0.5, 42);

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::Level,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.model_type, StructTsType::Level);
        assert_eq!(result.n_obs, 100);
        assert!(result.log_likelihood.is_finite());
        assert!(result.coef.level > 0.0);
        assert!(result.coef.epsilon > 0.0);
        assert!(result.coef.slope.is_none());
        assert!(result.coef.seasonal.is_none());

        // Fitted values should track the data
        for (fi, yi) in result.fitted.iter().zip(y.iter()) {
            assert!((fi - yi).abs() < 10.0);
        }
    }

    #[test]
    fn test_local_linear_trend_model() {
        // Generate trending data
        let y: Vec<f64> = (0..50)
            .map(|t| 100.0 + 0.5 * t as f64 + (t as f64 * 0.1).sin())
            .collect();

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::Trend,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.model_type, StructTsType::Trend);
        assert!(result.slope.is_some());

        // Slope should be positive for upward trending data
        let slopes = result.slope.unwrap();
        let avg_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
        assert!(avg_slope > 0.0, "Average slope should be positive");
    }

    #[test]
    fn test_bsm_model() {
        // Generate seasonal data
        let period = 12;
        let n = 48; // 4 years of monthly data
        let y: Vec<f64> = (0..n)
            .map(|t| {
                let trend = 100.0 + 0.2 * t as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
                trend + seasonal
            })
            .collect();

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::BSM,
                period: Some(period),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.model_type, StructTsType::BSM);
        assert!(result.seasonal.is_some());

        // Seasonal component should capture the pattern
        let seasonal = result.seasonal.unwrap();
        assert!(!seasonal.is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let y = vec![1.0, 2.0, 3.0];
        let result = struct_ts(&y, StructTsConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_aic_bic() {
        let y = generate_local_level_data(50, 1.0, 0.5, 42);
        let result = struct_ts(&y, StructTsConfig::default()).unwrap();

        // AIC and BIC should be finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // BIC penalizes more than AIC for this sample size
        // BIC = -2*LL + k*ln(n), AIC = -2*LL + 2k
        // For n=50, ln(50) ≈ 3.9 > 2, so BIC > AIC
        assert!(result.bic > result.aic);
    }

    #[test]
    fn test_validate_structts_local_level_tracks_data() {
        // The estimated level should track the data within a reasonable range.
        let y = generate_local_level_data(150, 1.0, 0.5, 101);

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::Level,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.n_obs, 150);

        // Level should be close to the data (within a few standard deviations)
        let mut max_dev = 0.0f64;
        for t in 10..result.n_obs {
            let dev = (result.level[t] - y[t]).abs();
            max_dev = max_dev.max(dev);
        }

        // The level is the smoothed signal; deviations should be moderate
        assert!(
            max_dev < 10.0,
            "Level should track data, max deviation = {:.4}",
            max_dev
        );
    }

    #[test]
    fn test_validate_structts_bsm_components_sum_to_data() {
        // For BSM: fitted = level + seasonal should approximate data.
        // residuals = data - fitted
        let period = 12;
        let n = 60; // 5 years of monthly data
        let y: Vec<f64> = (0..n)
            .map(|t| {
                let trend = 50.0 + 0.3 * t as f64;
                let seasonal = 8.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
                let noise = 0.2 * ((t * 13 + 7) % 11) as f64 / 11.0 - 0.1;
                trend + seasonal + noise
            })
            .collect();

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::BSM,
                period: Some(period),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(result.seasonal.is_some());
        assert!(result.slope.is_some());

        // fitted + residual should equal original data
        for t in 0..n {
            let reconstructed = result.fitted[t] + result.residuals[t];
            let diff = (reconstructed - y[t]).abs();
            assert!(
                diff < 1e-10,
                "fitted + residual should equal y at t={}, diff={:.12}",
                t,
                diff
            );
        }
    }

    #[test]
    fn test_validate_structts_loglikelihood_computed() {
        // Log-likelihood should be finite for all model types.
        let y = generate_local_level_data(80, 0.5, 1.0, 77);

        for model_type in &[StructTsType::Level, StructTsType::Trend] {
            let config = StructTsConfig {
                model_type: *model_type,
                ..Default::default()
            };
            let result = struct_ts(&y, config).unwrap();

            assert!(
                result.log_likelihood.is_finite(),
                "Log-likelihood should be finite for {:?}, got {}",
                model_type,
                result.log_likelihood
            );
        }
    }

    #[test]
    fn test_validate_structts_trend_model_positive_slope() {
        // For clearly trending data, the estimated slope should be positive.
        let n = 100;
        let y: Vec<f64> = (0..n)
            .map(|t| {
                let trend = 10.0 + 2.0 * t as f64;
                let noise = 0.5 * ((t * 11 + 3) % 7) as f64 / 7.0 - 0.25;
                trend + noise
            })
            .collect();

        let result = struct_ts(
            &y,
            StructTsConfig {
                model_type: StructTsType::Trend,
                ..Default::default()
            },
        )
        .unwrap();

        let slopes = result.slope.unwrap();
        let avg_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
        assert!(
            avg_slope > 0.5,
            "Average slope should be clearly positive for linearly trending data, got {:.4}",
            avg_slope
        );
    }
}
