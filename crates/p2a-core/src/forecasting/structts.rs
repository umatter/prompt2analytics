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
use crate::forecasting::kalman::{
    StateSpaceModel, admit_filter_and_smoother_history, kalman_filter, kalman_loglik,
    kalman_smoother,
};
use crate::resource::{checkpoint, try_zeros2, with_remedy};
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
    /// Seasonal period, only used for BSM (12 when unset): at least 2, and at
    /// most half the series, since StructTS needs two full seasons.
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
/// `StructTsResult` with fitted model and components, or
/// [`EconError::Cancelled`] when the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline (checked
/// once per optimiser iteration and within each Kalman-filter pass).
///
/// The likelihood evaluations keep no Kalman history; the final pass holds
/// the filter's and the smoother's at once (see [`kalman_filter`] and
/// [`kalman_smoother`]), m = 1, 2, or period + 1 for BSM. The two are
/// admitted as one figure before the optimiser starts, about 24·n·m² bytes
/// (n·(24m² + 144m + 256) once m ≥ 2, 472 bytes per observation at m = 1):
/// [`EconError::ResourceLimit`] when it exceeds the per-allocation ceiling,
/// naming `period` for BSM. The BSM's transition matrix and initial state
/// covariance (m × m) are allocated fallibly: a [`EconError::ResourceLimit`]
/// naming `period` when they cannot be.
///
/// A BSM `period` (12 when unset) must be at least 2, as in R ("frequency
/// must be a positive integer >= 2 for BSM"), and the series must hold two
/// full seasons, 2 · period observations:
/// [`EconError::InvalidSpecification`] for a period below 2 or one whose two
/// seasons overflow, [`EconError::InsufficientData`] for a series shorter
/// than two seasons. The level and trend models ignore `period`.
pub fn struct_ts(y: &[f64], config: StructTsConfig) -> EconResult<StructTsResult> {
    let n = y.len();

    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "StructTS requires at least 10 observations".to_string(),
        });
    }

    // Validate BSM configuration: 2 <= period <= n / 2, before anything is
    // sized by it.
    if config.model_type == StructTsType::BSM {
        check_bsm_period(bsm_period(&config), n)?;
    }

    // The final pass holds the filter's and the smoother's histories at once.
    // Admit them as one figure before the optimiser, whose likelihood passes
    // keep none, so that a history above the ceiling is refused before any
    // fitting work.
    let m = state_dim(&config);
    admit_filter_and_smoother_history(n, m)
        .map_err(|e| with_remedy(e, &history_remedy(&config, m)))?;

    // Initialize variance parameters
    let y_var = sample_variance(y);
    let (init_params, n_params) = initialize_params(&config, y_var);

    // Optimize variance parameters
    let (opt_params, log_lik, converged) =
        optimize_params(y, &config, init_params, config.max_iter, config.tolerance)?;

    // Build final model and get smoothed states
    let model = build_model(&config, &opt_params)?;
    let init_state = initialize_state(&config, y);
    let init_cov = initialize_covariance(&config, y_var)?;

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
///
/// Errors as [`struct_ts`] does, including [`EconError::Cancelled`] on a
/// stopped budget.
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

/// The BSM seasonal period when [`StructTsConfig::period`] is unset.
const DEFAULT_BSM_PERIOD: usize = 12;

/// The BSM seasonal period of `config`: its `period`, or
/// [`DEFAULT_BSM_PERIOD`] when unset. The one place the default is applied,
/// so that the validation, the state dimension and the remedies agree.
fn bsm_period(config: &StructTsConfig) -> usize {
    config.period.unwrap_or(DEFAULT_BSM_PERIOD)
}

/// `Ok` when a BSM with seasonal `period` can be fitted to `n` observations:
/// a cycle of at least two seasons, and two full cycles in the series. The
/// product 2 · period is checked, so a huge period is refused rather than
/// overflowing (a panic in a debug build, a wrap that passes in release).
fn check_bsm_period(period: usize, n: usize) -> EconResult<()> {
    if period < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "BSM period must be at least 2, got {period}: a seasonal cycle has at least \
                 two seasons; use the trend model for a series without seasonality"
            ),
        });
    }
    match period.checked_mul(2) {
        Some(required) if n >= required => Ok(()),
        Some(required) => Err(EconError::InsufficientData {
            required,
            provided: n,
            context: format!(
                "BSM with period {period} requires at least {required} observations; with {n} \
                 observations period must be at most {}",
                n / 2
            ),
        }),
        None => Err(EconError::InvalidSpecification {
            message: format!(
                "BSM period {period} is too large: StructTS needs two full seasons, \
                 2 · period observations, so with {n} observations period must be at most {}",
                n / 2
            ),
        }),
    }
}

/// The state dimension of the model for `config`, the one figure that
/// [`build_model`], [`initialize_state`], [`initialize_covariance`] and the
/// history admission in [`struct_ts`] all use. [`struct_ts`] checks the BSM
/// period first; saturating all the same, so that no use can panic on it.
fn state_dim(config: &StructTsConfig) -> usize {
    match config.model_type {
        StructTsType::Level => 1,
        StructTsType::Trend => 2,
        // level + slope + (period - 1) seasonal states
        StructTsType::BSM => bsm_period(config).saturating_add(1),
    }
}

/// What to change when the Kalman history of an `m`-dimensional state is
/// refused: for BSM the state dimension follows from `period`.
fn history_remedy(config: &StructTsConfig, m: usize) -> String {
    match config.model_type {
        StructTsType::BSM => format!(
            "{}; use a smaller period or a shorter series",
            bsm_dim_note(config, m)
        ),
        StructTsType::Level | StructTsType::Trend => "use a shorter series".to_string(),
    }
}

/// How `config`'s BSM `period` sets the state dimension `m`, for a remedy.
/// The one wording [`history_remedy`] and [`bsm_square`] share, with the
/// period read through [`bsm_period`], so the two remedies cannot drift apart.
fn bsm_dim_note(config: &StructTsConfig, m: usize) -> String {
    format!(
        "with period {} the state dimension is period + 1 = {m}",
        bsm_period(config)
    )
}

/// An m × m zero matrix of the BSM, `what` it is, allocated fallibly: a
/// [`EconError::ResourceLimit`] that names `period` and `m` when its size
/// overflows, exceeds the per-allocation ceiling or cannot be allocated.
/// `what` is static and the remedy is formatted only on a refusal, so the
/// likelihood passes of the search allocate no text.
fn bsm_square(config: &StructTsConfig, m: usize, what: &'static str) -> EconResult<Array2<f64>> {
    try_zeros2(m, m, what).map_err(|e| {
        with_remedy(
            e,
            &format!("{}; use a smaller period", bsm_dim_note(config, m)),
        )
    })
}

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

    // Objective function: negative log-likelihood with log-transformed params.
    // A parameter vector the model cannot use scores 1e20. A stopped budget
    // stops the fit instead of scoring the vertex, and so would a
    // ResourceLimit, which the likelihood pass raises only when its m × m
    // transition or initial covariance cannot be allocated (it keeps no
    // Kalman history; struct_ts admits the final pass's before the search).
    let objective = |log_params: &[f64]| -> EconResult<f64> {
        let params: Vec<f64> = log_params.iter().map(|&lp| y_var * lp.exp()).collect();
        match compute_loglik(y, config, &params) {
            Ok(ll) => Ok(-ll),
            Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => Err(e),
            Err(_) => Ok(1e20),
        }
    };

    // Use Nelder-Mead (simpler and more robust for small parameter counts)
    let (best_log_params, neg_loglik, converged) =
        nelder_mead_optimize(&objective, &init_log, max_iter, tolerance)?;

    // Transform back to original scale
    let best_params: Vec<f64> = best_log_params.iter().map(|&lp| y_var * lp.exp()).collect();

    Ok((best_params, -neg_loglik, converged))
}

/// Nelder-Mead simplex optimization for small parameter counts.
///
/// Each evaluation of `objective` is a Kalman-filter pass over the series, so
/// the budget is checked once per iteration (and within each pass by
/// [`kalman_loglik`]); an `Err` from `objective` stops the search. CausalImpact
/// fits its BSTS variances with this optimiser too.
pub(crate) fn nelder_mead_optimize(
    objective: &dyn Fn(&[f64]) -> EconResult<f64>,
    init: &[f64],
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Vec<f64>, f64, bool)> {
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

    let mut values: Vec<f64> = simplex
        .iter()
        .map(|v| objective(v))
        .collect::<EconResult<_>>()?;

    for _iter in 0..max_iter {
        checkpoint()?;
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
            return Ok((simplex[0].clone(), values[0], true));
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
        let f_reflected = objective(&reflected)?;

        if f_reflected < values[n - 1] && f_reflected >= values[0] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else if f_reflected < values[0] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = objective(&expanded)?;

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
            let f_contracted = objective(&contract_point)?;

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
                    values[i] = objective(&simplex[i])?;
                }
            }
        }
    }

    Ok((simplex[0].clone(), values[0], false))
}

fn compute_loglik(y: &[f64], config: &StructTsConfig, params: &[f64]) -> EconResult<f64> {
    let model = build_model(config, params)?;
    let y_var = sample_variance(y);
    let init_state = initialize_state(config, y);
    let init_cov = initialize_covariance(config, y_var)?;

    kalman_loglik(y, &model, init_state.view(), init_cov.view())
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
            let m = state_dim(config); // level + slope + (period-1) seasonal states

            // Transition matrix
            let mut transition = bsm_square(config, m, "the BSM transition matrix")?;
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
            let mut state = Array1::zeros(state_dim(config));
            state[0] = y[0]; // Level
            state[1] = if y.len() > 1 { y[1] - y[0] } else { 0.0 }; // Slope
            // Seasonal initialized to zero
            state
        }
    }
}

fn initialize_covariance(config: &StructTsConfig, y_var: f64) -> EconResult<Array2<f64>> {
    // Use diffuse initialization (large variance)
    let m = state_dim(config);
    let mut cov = match config.model_type {
        StructTsType::BSM => bsm_square(config, m, "the BSM initial state covariance")?,
        StructTsType::Level | StructTsType::Trend => Array2::zeros((m, m)),
    };
    cov.diag_mut().fill(y_var * 10.0);
    Ok(cov)
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

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn struct_ts_optimiser_stops_when_the_budget_is_cancelled() {
        // Before the fix the Nelder-Mead loop over Kalman-filter likelihoods
        // never looked at the budget and returned Ok under a cancelled one.
        let y = generate_local_level_data(100, 1.0, 0.5, 7);
        for model_type in [StructTsType::Level, StructTsType::Trend] {
            let config = StructTsConfig {
                model_type,
                ..Default::default()
            };
            assert!(matches!(
                cancelled(|| struct_ts(&y, config.clone())),
                Err(EconError::Cancelled(_))
            ));
            // Without a budget nothing changes.
            assert!(struct_ts(&y, config).is_ok());
        }
    }

    #[test]
    fn struct_ts_optimiser_itself_stops_when_the_budget_is_cancelled() {
        // struct_ts would also stop at its final kalman_filter call, so the
        // test above passes on the filter's own checkpoint. Call the optimiser
        // directly (no final pass): with neither the per-iteration checkpoint
        // nor the Cancelled pass-through in the objective, every vertex scores
        // 1e20 and the search returns Ok as converged.
        let y = generate_local_level_data(48, 1.0, 0.5, 11);
        for (model_type, period) in [
            (StructTsType::Level, None),
            (StructTsType::Trend, None),
            (StructTsType::BSM, Some(4)),
        ] {
            let config = StructTsConfig {
                model_type,
                period,
                ..Default::default()
            };
            let (init, _) = initialize_params(&config, sample_variance(&y));
            assert!(
                matches!(
                    cancelled(|| optimize_params(
                        &y,
                        &config,
                        init.clone(),
                        config.max_iter,
                        config.tolerance
                    )),
                    Err(EconError::Cancelled(_))
                ),
                "{model_type:?}"
            );
            // With no iterations the loop's checkpoint never runs, so this
            // pins the objective's pass-through alone: the initial simplex
            // must not score a cancelled likelihood as 1e20.
            assert!(
                matches!(
                    cancelled(|| optimize_params(&y, &config, init.clone(), 0, config.tolerance)),
                    Err(EconError::Cancelled(_))
                ),
                "{model_type:?}"
            );
            // Without a budget nothing changes.
            assert!(optimize_params(&y, &config, init, config.max_iter, config.tolerance).is_ok());
        }
    }

    #[test]
    fn nelder_mead_optimize_checkpoints_each_iteration() {
        // The objective never looks at the budget, so only the loop's own
        // checkpoint can stop the search.
        let objective = |x: &[f64]| -> EconResult<f64> { Ok(x.iter().map(|v| v * v).sum()) };
        let init = [1.0, -2.0];
        assert!(matches!(
            cancelled(|| nelder_mead_optimize(&objective, &init, 200, 1e-12)),
            Err(EconError::Cancelled(_))
        ));
        let (x, f, _) = nelder_mead_optimize(&objective, &init, 200, 1e-12).unwrap();
        assert!(f < 1e-6 && x.iter().all(|v| v.abs() < 1e-3), "{x:?} {f}");

        // A budget cancelled mid-search. The objective cancels the live token
        // on its CANCEL_AT-th evaluation. With a check on every iteration, at
        // most one iteration's evaluations follow the cancel (a reflection, an
        // expansion or contraction, and a shrink of n vertices). A check made
        // only before the loop would let the search run on and return Ok.
        const CANCEL_AT: usize = 20;
        let bound = CANCEL_AT + init.len() + 2;
        let free_evals = std::cell::Cell::new(0usize);
        let free = |x: &[f64]| -> EconResult<f64> {
            free_evals.set(free_evals.get() + 1);
            objective(x)
        };
        nelder_mead_optimize(&free, &init, 200, 1e-12).unwrap();
        // The uncancelled search needs more evaluations than the bound allows,
        // so the bound below can only hold if the search was stopped.
        assert!(free_evals.get() > bound, "{} evaluations", free_evals.get());

        let token = crate::resource::CancelToken::new();
        let evals = std::cell::Cell::new(0usize);
        let cancelling = |x: &[f64]| -> EconResult<f64> {
            evals.set(evals.get() + 1);
            if evals.get() == CANCEL_AT {
                token.cancel();
            }
            objective(x)
        };
        let budget = crate::resource::Budget::new(None, token.clone());
        let stopped = crate::resource::with_budget(budget, || {
            nelder_mead_optimize(&cancelling, &init, 200, 1e-12)
        });
        assert!(
            matches!(stopped, Err(EconError::Cancelled(_))),
            "{stopped:?}"
        );
        assert!(evals.get() <= bound, "{} evaluations", evals.get());
    }

    fn seasonal_series(n: usize, period: usize, seed: u64) -> Vec<f64> {
        let noise = generate_local_level_data(n, 0.2, 0.5, seed);
        (0..n)
            .map(|t| {
                let season = 6.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
                noise[t] + 0.1 * t as f64 + season
            })
            .collect()
    }

    #[test]
    fn struct_ts_refuses_a_kalman_history_above_the_allocation_ceiling() {
        // The Kalman filter's and smoother's histories grow with n·m²
        // (m = period + 1 for BSM), so with a large state dimension (a
        // seasonal model with a long period) memory grew without bound.
        // struct_ts now admits both, as one figure, before the optimiser, and
        // the refusal names the period that sets m. Before the fix this
        // returned Ok under the ceiling.
        let (n, period) = (48, 12);
        let y = seasonal_series(n, period, 3);
        let config = StructTsConfig {
            model_type: StructTsType::BSM,
            period: Some(period),
            ..Default::default()
        };
        // Filter and smoother together: 48 · (24·13² + 144·13 + 256) =
        // 296,832 bytes > 64 KiB.
        let ceiling = Some(64 << 10);
        let refused =
            crate::resource::with_alloc_ceiling(ceiling, || struct_ts(&y, config.clone()));
        match &refused {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("Kalman filter's and smoother's")
                    && msg.contains("with period 12 the state dimension is period + 1 = 13")
                    && msg.ends_with("; use a smaller period or a shorter series")
                    && msg.matches("; use ").count() == 1,
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // The refusal comes before the optimiser: under a stopped budget it
        // is still the ResourceLimit, so no likelihood pass ran first.
        let refused_first = cancelled(|| {
            crate::resource::with_alloc_ceiling(ceiling, || struct_ts(&y, config.clone()))
        });
        assert!(
            matches!(refused_first, Err(EconError::ResourceLimit(_))),
            "{refused_first:?}"
        );
        // The likelihood passes keep no history, so the optimiser itself runs
        // under the ceiling and finds exactly what it finds without one. Had
        // they run kalman_filter, which admits its history (198,144 bytes
        // here), every evaluation would be ResourceLimit under this ceiling.
        let (init, _) = initialize_params(&config, sample_variance(&y));
        let bounded = crate::resource::with_alloc_ceiling(ceiling, || {
            optimize_params(&y, &config, init.clone(), config.max_iter, config.tolerance)
        });
        let unbounded = optimize_params(&y, &config, init, config.max_iter, config.tolerance);
        assert_eq!(bounded.unwrap(), unbounded.unwrap());
        // The hosted ceiling (256 MiB) leaves an ordinary monthly BSM alone.
        let fitted = crate::resource::with_alloc_ceiling(Some(256 << 20), || {
            struct_ts(&seasonal_series(240, period, 4), config.clone())
        });
        assert_eq!(fitted.unwrap().n_obs, 240);
    }

    #[test]
    fn struct_ts_admits_a_small_state_history_by_its_bytes() {
        // A local level model keeps a one-dimensional state: 48 bytes of
        // values per filter step and 16 per smoother step, but its nested
        // Vecs hold several times that, so counting values admitted a long
        // series' history far past the ceiling. Before the fix both
        // refusals below returned Ok.
        let y = generate_local_level_data(1000, 1.0, 0.5, 17);
        let fit = |model_type, ceiling: usize, y: &[f64]| {
            let config = StructTsConfig {
                model_type,
                ..Default::default()
            };
            crate::resource::with_alloc_ceiling(Some(ceiling), || struct_ts(y, config))
        };
        // Filter and smoother together. Level: 1,000 · 472 = 472,000 bytes
        // > 64 KiB (the values: 64,000). Trend: 1,000 · (24·2² + 144·2 +
        // 256) = 640,000 bytes > 256 KiB (the values: 160,000).
        for (model_type, ceiling, m) in [
            (StructTsType::Level, 64 << 10, 1),
            (StructTsType::Trend, 256 << 10, 2),
        ] {
            match fit(model_type, ceiling, &y) {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.contains(&format!("1000 observations and a {m}-dimensional state"))
                        && msg.ends_with("; use a shorter series"),
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit for {model_type:?}, got {other:?}"),
            }
        }
        // 100 rows (100 · 472 = 47,200 bytes) fit under 64 KiB.
        assert_eq!(
            fit(StructTsType::Level, 64 << 10, &y[..100]).unwrap().n_obs,
            100
        );
    }

    #[test]
    fn struct_ts_admits_the_final_pass_as_one_figure() {
        // The final pass holds the filter's history while the smoother builds
        // its own. Level, 150 rows: 48,000 and 22,800 bytes each fit 64 KiB,
        // 70,800 together do not. Before the fix, which admits the two as one
        // figure, this returned Ok; one at a time, a final pass could hold up
        // to about 1.5 times the ceiling.
        let y = generate_local_level_data(150, 1.0, 0.5, 19);
        let config = StructTsConfig::default();
        let refused =
            crate::resource::with_alloc_ceiling(Some(64 << 10), || struct_ts(&y, config.clone()));
        match refused {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("Kalman filter's and smoother's")
                    && msg.contains("150 observations and a 1-dimensional state")
                    && msg.ends_with("; use a shorter series"),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // 130 rows (61,360 bytes) fit.
        let fitted = crate::resource::with_alloc_ceiling(Some(64 << 10), || {
            struct_ts(&y[..130], config.clone())
        });
        assert_eq!(fitted.unwrap().n_obs, 130);
        // Without a ceiling nothing changes.
        assert_eq!(struct_ts(&y, config).unwrap().n_obs, 150);
    }

    #[test]
    fn state_dim_is_the_dimension_of_the_model_that_runs() {
        // The admission sizes the history by state_dim; the filter runs the
        // model build_model builds, from initialize_state and
        // initialize_covariance. All four must agree.
        let y = seasonal_series(48, 12, 23);
        let y_var = sample_variance(&y);
        for (model_type, period, m) in [
            (StructTsType::Level, None, 1),
            (StructTsType::Trend, None, 2),
            (StructTsType::BSM, Some(4), 5),
            (StructTsType::BSM, Some(12), 13),
            (StructTsType::BSM, None, 13),
        ] {
            let config = StructTsConfig {
                model_type,
                period,
                ..Default::default()
            };
            let (params, _) = initialize_params(&config, y_var);
            assert_eq!(state_dim(&config), m, "{model_type:?} {period:?}");
            assert_eq!(build_model(&config, &params).unwrap().state_dim(), m);
            assert_eq!(initialize_state(&config, &y).len(), m);
            assert_eq!(initialize_covariance(&config, y_var).unwrap().dim(), (m, m));
        }
    }

    fn bsm(period: usize) -> StructTsConfig {
        StructTsConfig {
            model_type: StructTsType::BSM,
            period: Some(period),
            ..Default::default()
        }
    }

    #[test]
    fn struct_ts_refuses_a_bsm_period_below_two() {
        // A seasonal cycle has at least two seasons, as R's StructTS requires
        // ("frequency must be a positive integer >= 2 for BSM"). Period 0
        // and 1 passed the n >= 2 · period check, and build_model then
        // indexed past its matrices: [0, 1] of a 1 × 1 transition for period
        // 0, observation[2] of a 2-vector for period 1. Before the fix both
        // panicked.
        let y = seasonal_series(48, 12, 29);
        for period in [0, 1] {
            match struct_ts(&y, bsm(period)) {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.starts_with(&format!("BSM period must be at least 2, got {period}")),
                    "{message}"
                ),
                other => panic!("expected InvalidSpecification for period {period}, got {other:?}"),
            }
        }
        // Period 2 is a model.
        assert_eq!(struct_ts(&y, bsm(2)).unwrap().n_obs, 48);
        // The level and trend models have no seasonal component and ignore
        // period, so period 0 is not refused there.
        for model_type in [StructTsType::Level, StructTsType::Trend] {
            let config = StructTsConfig {
                model_type,
                period: Some(0),
                ..Default::default()
            };
            assert_eq!(struct_ts(&y, config).unwrap().n_obs, 48, "{model_type:?}");
        }
    }

    #[test]
    fn struct_ts_refuses_a_bsm_period_whose_two_seasons_overflow() {
        // StructTS needs two full seasons, 2 · period observations. That
        // product was unchecked: for period > usize::MAX / 2 it panicked in a
        // debug build, and in a release build it wrapped to a small number
        // that a short series passes. Before the fix this panicked.
        let y = seasonal_series(48, 12, 31);
        let period = usize::MAX / 2 + 1;
        match struct_ts(&y, bsm(period)) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.starts_with(&format!("BSM period {period} is too large"))
                    && message.ends_with("with 48 observations period must be at most 24"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        // A period that needs more than the series holds is refused as before,
        // and the refusal now names the bound on period, as the overflow's
        // does. Before the fix the context stopped at the observations.
        match struct_ts(&y, bsm(25)) {
            Err(EconError::InsufficientData {
                required,
                provided,
                context,
            }) => {
                assert_eq!((required, provided), (50, 48));
                assert_eq!(
                    context,
                    "BSM with period 25 requires at least 50 observations; with 48 \
                     observations period must be at most 24"
                );
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
        // The largest period whose two seasons fit in usize takes the same
        // InsufficientData refusal, not the overflow's.
        match struct_ts(&y, bsm(usize::MAX / 2)) {
            Err(EconError::InsufficientData { context, .. }) => assert!(
                context.ends_with("; with 48 observations period must be at most 24"),
                "{context}"
            ),
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn bsm_model_matrices_are_allocated_fallibly() {
        // build_model allocated the m × m transition with Array2::zeros, and
        // initialize_covariance the m × m prior with from_diag (m = period +
        // 1). Both are infallible: a matrix the allocator cannot provide
        // aborted the process, and one whose size overflows panicked. With
        // no ceiling (CLI, library) period <= n / 2 still lets a long series
        // ask for an m × m beyond memory. Both are now a ResourceLimit that
        // names period. Before the fix the refusal below returned Ok and the
        // overflow panicked.
        let y_var = 4.0;
        let (params, _) = initialize_params(&bsm(12), y_var);
        let remedy =
            "; with period 12 the state dimension is period + 1 = 13; use a smaller period";
        // 8 · 13² = 1,352 bytes > 1 KiB.
        let ceiling = Some(1 << 10);
        match crate::resource::with_alloc_ceiling(ceiling, || build_model(&bsm(12), &params)) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.starts_with("the BSM transition matrix would need")
                    && msg.ends_with(remedy)
                    && msg.matches("; use ").count() == 1,
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        match crate::resource::with_alloc_ceiling(ceiling, || {
            initialize_covariance(&bsm(12), y_var)
        }) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.starts_with("the BSM initial state covariance would need")
                    && msg.ends_with(remedy),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // An unset period is the default 12 in the remedy too.
        let default_period = StructTsConfig {
            model_type: StructTsType::BSM,
            ..Default::default()
        };
        match crate::resource::with_alloc_ceiling(ceiling, || {
            initialize_covariance(&default_period, y_var)
        }) {
            Err(EconError::ResourceLimit(msg)) => assert!(msg.ends_with(remedy), "{msg}"),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // An m × m whose size overflows is refused before any allocation.
        let huge = bsm(usize::MAX / 4);
        assert!(matches!(
            build_model(&huge, &params),
            Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
        ));
        assert!(matches!(
            initialize_covariance(&huge, y_var),
            Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
        ));
        // Without a ceiling the prior is the diagonal it was.
        let prior = initialize_covariance(&bsm(12), y_var).unwrap();
        assert_eq!(
            prior,
            Array2::from_diag(&Array1::from_elem(13, y_var * 10.0))
        );
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
