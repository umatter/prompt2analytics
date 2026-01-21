//! Kalman filtering, smoothing, and forecasting for state-space models.
//!
//! Implements the Kalman filter for univariate time series in state-space form:
//!
//! State equation:     α_{t+1} = T α_t + R η_t,  η_t ~ N(0, Q)
//! Observation eq:     y_t = Z' α_t + ε_t,       ε_t ~ N(0, H)
//!
//! # References
//!
//! - Durbin, J. & Koopman, S. J. (2012). "Time Series Analysis by State Space Methods".
//!   Oxford Statistical Science Series.
//! - Harvey, A. C. (1990). "Forecasting, Structural Time Series Models and the Kalman Filter".
//!   Cambridge University Press.
//! - R Core Team. Kalman filtering functions in stats package.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/KalmanLike.html>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;

/// State-space model specification.
///
/// Represents a linear Gaussian state-space model:
/// - State equation: α_{t+1} = T α_t + c + R η_t
/// - Observation eq: y_t = Z' α_t + d + ε_t
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// Transition matrix T (m × m)
    pub transition: Array2<f64>,
    /// Observation vector Z (m × 1) - coefficients linking state to observation
    pub observation: Array1<f64>,
    /// State covariance selection matrix R (m × r)
    pub selection: Array2<f64>,
    /// State innovation covariance Q (r × r)
    pub state_cov: Array2<f64>,
    /// Observation variance H (scalar for univariate)
    pub obs_var: f64,
    /// State intercept c (m × 1), optional
    pub state_intercept: Option<Array1<f64>>,
    /// Observation intercept d (scalar), optional
    pub obs_intercept: Option<f64>,
}

impl StateSpaceModel {
    /// Create a new state-space model.
    pub fn new(
        transition: Array2<f64>,
        observation: Array1<f64>,
        selection: Array2<f64>,
        state_cov: Array2<f64>,
        obs_var: f64,
    ) -> EconResult<Self> {
        let m = transition.nrows();

        if transition.ncols() != m {
            return Err(EconError::InvalidSpecification {
                message: format!("Transition matrix must be square, got {}x{}", m, transition.ncols()),
            });
        }

        if observation.len() != m {
            return Err(EconError::InvalidSpecification {
                message: format!("Observation vector length ({}) must match state dimension ({})",
                    observation.len(), m),
            });
        }

        if selection.nrows() != m {
            return Err(EconError::InvalidSpecification {
                message: format!("Selection matrix rows ({}) must match state dimension ({})",
                    selection.nrows(), m),
            });
        }

        let r = selection.ncols();
        if state_cov.dim() != (r, r) {
            return Err(EconError::InvalidSpecification {
                message: format!("State covariance dimensions ({:?}) must be ({}, {})",
                    state_cov.dim(), r, r),
            });
        }

        Ok(Self {
            transition,
            observation,
            selection,
            state_cov,
            obs_var,
            state_intercept: None,
            obs_intercept: None,
        })
    }

    /// Get the state dimension.
    pub fn state_dim(&self) -> usize {
        self.transition.nrows()
    }

    /// Compute R Q R' (state disturbance covariance)
    pub fn state_disturbance_cov(&self) -> Array2<f64> {
        let rq = self.selection.dot(&self.state_cov);
        rq.dot(&self.selection.t())
    }
}

/// Result from Kalman filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilterResult {
    /// Filtered state estimates α_{t|t}
    pub filtered_states: Vec<Vec<f64>>,
    /// Filtered state covariances P_{t|t}
    pub filtered_covs: Vec<Vec<Vec<f64>>>,
    /// Predicted state estimates α_{t|t-1}
    pub predicted_states: Vec<Vec<f64>>,
    /// Predicted state covariances P_{t|t-1}
    pub predicted_covs: Vec<Vec<Vec<f64>>>,
    /// Prediction errors (innovations) v_t = y_t - Z' α_{t|t-1}
    pub innovations: Vec<f64>,
    /// Innovation variances F_t
    pub innovation_vars: Vec<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
    /// State dimension
    pub state_dim: usize,
}

/// Result from Kalman smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanSmootherResult {
    /// Smoothed state estimates α_{t|n}
    pub smoothed_states: Vec<Vec<f64>>,
    /// Smoothed state covariances P_{t|n}
    pub smoothed_covs: Vec<Vec<Vec<f64>>>,
    /// Number of observations
    pub n_obs: usize,
    /// State dimension
    pub state_dim: usize,
}

/// Result from Kalman forecasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanForecastResult {
    /// Forecasted observations
    pub forecasts: Vec<f64>,
    /// Forecast variances
    pub variances: Vec<f64>,
    /// Forecast standard errors
    pub std_errors: Vec<f64>,
    /// Forecasted states
    pub state_forecasts: Vec<Vec<f64>>,
    /// Forecast horizon
    pub horizon: usize,
}

/// Run the Kalman filter on a time series.
///
/// Implements the standard Kalman filter recursions:
/// - Prediction: α_{t|t-1} = T α_{t-1|t-1}, P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'
/// - Update: α_{t|t} = α_{t|t-1} + K_t v_t, P_{t|t} = P_{t|t-1} - K_t F_t K_t'
///
/// # Arguments
///
/// * `y` - Observed time series (may contain NaN for missing values)
/// * `model` - State-space model specification
/// * `init_state` - Initial state estimate α_0
/// * `init_cov` - Initial state covariance P_0
///
/// # Returns
///
/// `KalmanFilterResult` containing filtered states, innovations, and log-likelihood.
pub fn kalman_filter(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
) -> EconResult<KalmanFilterResult> {
    let n = y.len();
    let m = model.state_dim();

    if init_state.len() != m {
        return Err(EconError::InvalidSpecification {
            message: format!("Initial state length ({}) must match state dimension ({})",
                init_state.len(), m),
        });
    }

    if init_cov.dim() != (m, m) {
        return Err(EconError::InvalidSpecification {
            message: format!("Initial covariance dimensions ({:?}) must be ({}, {})",
                init_cov.dim(), m, m),
        });
    }

    // Storage
    let mut filtered_states = Vec::with_capacity(n);
    let mut filtered_covs = Vec::with_capacity(n);
    let mut predicted_states = Vec::with_capacity(n);
    let mut predicted_covs = Vec::with_capacity(n);
    let mut innovations = Vec::with_capacity(n);
    let mut innovation_vars = Vec::with_capacity(n);

    // Pre-compute R Q R'
    let state_dist_cov = model.state_disturbance_cov();

    // Initialize
    let mut state = init_state.to_owned();
    let mut cov = init_cov.to_owned();

    let mut log_lik = 0.0;
    let mut n_valid = 0usize;

    for t in 0..n {
        // Prediction step
        let pred_state = model.transition.dot(&state);
        let pred_cov = model.transition.dot(&cov).dot(&model.transition.t()) + &state_dist_cov;

        predicted_states.push(pred_state.to_vec());
        predicted_covs.push(array2_to_nested_vec(&pred_cov));

        // Update step (if observation is not missing)
        if y[t].is_nan() {
            // Missing observation: no update
            state = pred_state;
            cov = pred_cov;
            innovations.push(f64::NAN);
            innovation_vars.push(f64::NAN);
        } else {
            // Innovation
            let pred_obs = model.observation.dot(&pred_state);
            let v = y[t] - pred_obs;

            // Innovation variance: F = Z' P Z + H
            let f = model.observation.dot(&pred_cov.dot(&model.observation)) + model.obs_var;

            innovations.push(v);
            innovation_vars.push(f);

            if f > 1e-12 {
                // Kalman gain: K = P Z / F
                let kalman_gain = pred_cov.dot(&model.observation) / f;

                // State update
                state = &pred_state + &(&kalman_gain * v);

                // Covariance update (Joseph form for numerical stability)
                let identity = Array2::eye(m);
                let k_z_t = outer_product(&kalman_gain, &model.observation);
                let factor = &identity - &k_z_t;
                cov = factor.dot(&pred_cov).dot(&factor.t())
                    + outer_product(&kalman_gain, &kalman_gain) * model.obs_var;

                // Log-likelihood contribution
                log_lik -= 0.5 * (f.ln() + v * v / f);
                n_valid += 1;
            } else {
                state = pred_state;
                cov = pred_cov;
            }
        }

        filtered_states.push(state.to_vec());
        filtered_covs.push(array2_to_nested_vec(&cov));
    }

    // Add constant term to log-likelihood
    log_lik -= 0.5 * n_valid as f64 * (2.0 * std::f64::consts::PI).ln();

    Ok(KalmanFilterResult {
        filtered_states,
        filtered_covs,
        predicted_states,
        predicted_covs,
        innovations,
        innovation_vars,
        log_likelihood: log_lik,
        n_obs: n,
        state_dim: m,
    })
}

/// Run the Kalman smoother (backward pass).
///
/// Computes smoothed state estimates using all observations.
///
/// # Arguments
///
/// * `filter_result` - Result from kalman_filter
/// * `model` - State-space model specification
///
/// # Returns
///
/// `KalmanSmootherResult` with smoothed states and covariances.
pub fn kalman_smoother(
    filter_result: &KalmanFilterResult,
    model: &StateSpaceModel,
) -> EconResult<KalmanSmootherResult> {
    let n = filter_result.n_obs;
    let m = filter_result.state_dim;

    let mut smoothed_states = vec![vec![0.0; m]; n];
    let mut smoothed_covs = vec![vec![vec![0.0; m]; m]; n];

    // Initialize with final filtered values
    smoothed_states[n - 1] = filter_result.filtered_states[n - 1].clone();
    smoothed_covs[n - 1] = filter_result.filtered_covs[n - 1].clone();

    // Backward recursion
    for t in (0..n - 1).rev() {
        let filtered_state = vec_to_array1(&filter_result.filtered_states[t]);
        let filtered_cov = nested_vec_to_array2(&filter_result.filtered_covs[t]);
        let predicted_state_next = vec_to_array1(&filter_result.predicted_states[t + 1]);
        let predicted_cov_next = nested_vec_to_array2(&filter_result.predicted_covs[t + 1]);
        let smoothed_state_next = vec_to_array1(&smoothed_states[t + 1]);
        let smoothed_cov_next = nested_vec_to_array2(&smoothed_covs[t + 1]);

        // Compute smoother gain
        // L = P_{t|t} T' P_{t+1|t}^{-1}
        let (pred_cov_inv, _) = safe_inverse(&predicted_cov_next.view())
            .map_err(|e| EconError::InvalidSpecification {
                message: format!("Failed to invert predicted covariance: {}", e),
            })?;

        let smoother_gain = filtered_cov.dot(&model.transition.t()).dot(&pred_cov_inv);

        // Smoothed state: α_{t|n} = α_{t|t} + L (α_{t+1|n} - α_{t+1|t})
        let state_diff = &smoothed_state_next - &predicted_state_next;
        let smoothed_state = &filtered_state + &smoother_gain.dot(&state_diff);

        // Smoothed covariance: P_{t|n} = P_{t|t} + L (P_{t+1|n} - P_{t+1|t}) L'
        let cov_diff = &smoothed_cov_next - &predicted_cov_next;
        let smoothed_cov = &filtered_cov + &smoother_gain.dot(&cov_diff).dot(&smoother_gain.t());

        smoothed_states[t] = smoothed_state.to_vec();
        smoothed_covs[t] = array2_to_nested_vec(&smoothed_cov);
    }

    Ok(KalmanSmootherResult {
        smoothed_states,
        smoothed_covs,
        n_obs: n,
        state_dim: m,
    })
}

/// Generate forecasts using the Kalman filter.
///
/// # Arguments
///
/// * `filter_result` - Result from kalman_filter
/// * `model` - State-space model specification
/// * `horizon` - Number of steps to forecast ahead
///
/// # Returns
///
/// `KalmanForecastResult` with forecasts and prediction intervals.
pub fn kalman_forecast(
    filter_result: &KalmanFilterResult,
    model: &StateSpaceModel,
    horizon: usize,
) -> EconResult<KalmanForecastResult> {
    if horizon == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Forecast horizon must be positive".to_string(),
        });
    }

    let n = filter_result.n_obs;
    let m = filter_result.state_dim;

    // Start from final filtered state
    let mut state = vec_to_array1(&filter_result.filtered_states[n - 1]);
    let mut cov = nested_vec_to_array2(&filter_result.filtered_covs[n - 1]);

    let state_dist_cov = model.state_disturbance_cov();

    let mut forecasts = Vec::with_capacity(horizon);
    let mut variances = Vec::with_capacity(horizon);
    let mut std_errors = Vec::with_capacity(horizon);
    let mut state_forecasts = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        // Predict state
        state = model.transition.dot(&state);
        cov = model.transition.dot(&cov).dot(&model.transition.t()) + &state_dist_cov;

        // Predict observation
        let forecast = model.observation.dot(&state);
        let variance = model.observation.dot(&cov.dot(&model.observation)) + model.obs_var;

        forecasts.push(forecast);
        variances.push(variance);
        std_errors.push(variance.sqrt());
        state_forecasts.push(state.to_vec());
    }

    Ok(KalmanForecastResult {
        forecasts,
        variances,
        std_errors,
        state_forecasts,
        horizon,
    })
}

/// Compute the log-likelihood of observations given a state-space model.
///
/// This is useful for maximum likelihood estimation of model parameters.
pub fn kalman_loglik(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
) -> EconResult<f64> {
    let result = kalman_filter(y, model, init_state, init_cov)?;
    Ok(result.log_likelihood)
}

// Helper functions

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let m = a.len();
    let n = b.len();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

fn array2_to_nested_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    let (rows, cols) = arr.dim();
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(arr[[i, j]]);
        }
        result.push(row);
    }
    result
}

fn vec_to_array1(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

fn nested_vec_to_array2(v: &[Vec<f64>]) -> Array2<f64> {
    let rows = v.len();
    let cols = if rows > 0 { v[0].len() } else { 0 };
    let mut arr = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = v[i][j];
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_local_level_model() {
        // Local level model: α_{t+1} = α_t + η_t, y_t = α_t + ε_t
        // This is equivalent to ARIMA(0,1,1)

        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.1]];  // σ²_η
        let obs_var = 0.2;  // σ²_ε

        let model = StateSpaceModel::new(
            transition, observation, selection, state_cov, obs_var
        ).unwrap();

        // Generate some data
        let y = vec![1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 2.8, 3.2];

        // Initial state
        let init_state = array![y[0]];
        let init_cov = array![[1.0]];

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.state_dim, 1);
        assert!(result.log_likelihood.is_finite());

        // Filtered states should track the data
        for t in 0..result.n_obs {
            assert!((result.filtered_states[t][0] - y[t]).abs() < 1.0);
        }
    }

    #[test]
    fn test_kalman_smoother() {
        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.1]];
        let obs_var = 0.2;

        let model = StateSpaceModel::new(
            transition, observation, selection, state_cov, obs_var
        ).unwrap();

        let y = vec![1.0, 1.5, 2.0, 2.2, 2.5];
        let init_state = array![y[0]];
        let init_cov = array![[1.0]];

        let filter_result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();
        let smooth_result = kalman_smoother(&filter_result, &model).unwrap();

        assert_eq!(smooth_result.n_obs, 5);

        // Smoothed states should have smaller variance than filtered
        // (they use more information)
        for t in 0..smooth_result.n_obs {
            // Just check that values are reasonable
            assert!(smooth_result.smoothed_states[t][0].is_finite());
        }
    }

    #[test]
    fn test_kalman_forecast() {
        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.1]];
        let obs_var = 0.2;

        let model = StateSpaceModel::new(
            transition, observation, selection, state_cov, obs_var
        ).unwrap();

        let y = vec![1.0, 1.5, 2.0, 2.2, 2.5];
        let init_state = array![y[0]];
        let init_cov = array![[1.0]];

        let filter_result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();
        let forecast_result = kalman_forecast(&filter_result, &model, 3).unwrap();

        assert_eq!(forecast_result.horizon, 3);
        assert_eq!(forecast_result.forecasts.len(), 3);

        // Forecasts should be around the last observation for local level
        for f in &forecast_result.forecasts {
            assert!((f - 2.5).abs() < 1.0);
        }

        // Variance should increase with horizon
        assert!(forecast_result.variances[2] > forecast_result.variances[0]);
    }

    #[test]
    fn test_missing_values() {
        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.1]];
        let obs_var = 0.2;

        let model = StateSpaceModel::new(
            transition, observation, selection, state_cov, obs_var
        ).unwrap();

        // Data with missing value
        let y = vec![1.0, f64::NAN, 2.0, 2.2, 2.5];
        let init_state = array![y[0]];
        let init_cov = array![[1.0]];

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert_eq!(result.n_obs, 5);
        assert!(result.innovations[1].is_nan());  // Missing observation
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_local_linear_trend() {
        // Local linear trend model:
        // μ_{t+1} = μ_t + ν_t + ξ_t
        // ν_{t+1} = ν_t + ζ_t
        // y_t = μ_t + ε_t

        let transition = array![
            [1.0, 1.0],
            [0.0, 1.0]
        ];
        let observation = array![1.0, 0.0];  // Observe level only
        let selection = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let state_cov = array![
            [0.1, 0.0],
            [0.0, 0.01]
        ];
        let obs_var = 0.2;

        let model = StateSpaceModel::new(
            transition, observation, selection, state_cov, obs_var
        ).unwrap();

        // Generate trending data
        let y: Vec<f64> = (0..10).map(|t| 1.0 + 0.5 * t as f64 + 0.1 * (t as f64).sin()).collect();

        let init_state = array![y[0], 0.5];  // level = y[0], slope = 0.5
        let init_cov = array![
            [1.0, 0.0],
            [0.0, 0.1]
        ];

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert_eq!(result.state_dim, 2);
        assert!(result.log_likelihood.is_finite());

        // Slope estimate (second state) should be positive for trending data
        let final_slope = result.filtered_states[result.n_obs - 1][1];
        assert!(final_slope > 0.0);
    }
}
