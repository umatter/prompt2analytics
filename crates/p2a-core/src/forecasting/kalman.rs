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

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::resource::{
    check_alloc, checked_len, checkpoint, checkpoint_stride, overflow, try_with_capacity,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

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
                message: format!(
                    "Transition matrix must be square, got {}x{}",
                    m,
                    transition.ncols()
                ),
            });
        }

        if observation.len() != m {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Observation vector length ({}) must match state dimension ({})",
                    observation.len(),
                    m
                ),
            });
        }

        if selection.nrows() != m {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Selection matrix rows ({}) must match state dimension ({})",
                    selection.nrows(),
                    m
                ),
            });
        }

        let r = selection.ncols();
        if state_cov.dim() != (r, r) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "State covariance dimensions ({:?}) must be ({}, {})",
                    state_cov.dim(),
                    r,
                    r
                ),
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
/// `KalmanFilterResult` containing filtered states, innovations, and log-likelihood,
/// or [`EconError::Cancelled`] when the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline (checked
/// about every 65,536 operations of the O(m³) steps).
///
/// The result keeps every step's predicted and filtered state and covariance,
/// the innovation and its variance, in nested `Vec`s whose every state vector
/// and covariance row is a heap buffer of its own. Counted with glibc's
/// per-buffer overhead, that is n·(16m² + 96m + 176) bytes for n observations
/// and an m-dimensional state once m ≥ 2, and 320 bytes per step for m = 1,
/// whose values are only 48. That history is admitted as a whole before the
/// first step: it is [`EconError::ResourceLimit`] when its size overflows or
/// exceeds the per-allocation ceiling
/// ([`crate::resource::set_alloc_ceiling_bytes`]). [`kalman_loglik`] runs the
/// same recursions without keeping any of it.
pub fn kalman_filter(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
) -> EconResult<KalmanFilterResult> {
    let n = y.len();
    let m = model.state_dim();

    check_initial_values(m, init_state, init_cov)?;

    // Every step stores P_{t|t-1} and P_{t|t}, both state vectors, the
    // innovation and its variance. Admit the whole history before any work:
    // the nested Vecs grow one m-vector at a time, so no single allocation
    // would ever reach the per-allocation ceiling while the total grows
    // without bound.
    admit_filter_history(n, m)?;
    let what = filter_history_what(n, m);
    let mut history = FilterHistory {
        filtered_states: try_with_capacity(n, &what)?,
        filtered_covs: try_with_capacity(n, &what)?,
        predicted_states: try_with_capacity(n, &what)?,
        predicted_covs: try_with_capacity(n, &what)?,
        innovations: try_with_capacity(n, &what)?,
        innovation_vars: try_with_capacity(n, &what)?,
    };

    let log_likelihood = filter_pass(y, model, init_state, init_cov, Some(&mut history))?;

    Ok(KalmanFilterResult {
        filtered_states: history.filtered_states,
        filtered_covs: history.filtered_covs,
        predicted_states: history.predicted_states,
        predicted_covs: history.predicted_covs,
        innovations: history.innovations,
        innovation_vars: history.innovation_vars,
        log_likelihood,
        n_obs: n,
        state_dim: m,
    })
}

/// The per-step values a [`KalmanFilterResult`] keeps.
struct FilterHistory {
    filtered_states: Vec<Vec<f64>>,
    filtered_covs: Vec<Vec<Vec<f64>>>,
    predicted_states: Vec<Vec<f64>>,
    predicted_covs: Vec<Vec<Vec<f64>>>,
    innovations: Vec<f64>,
    innovation_vars: Vec<f64>,
}

/// `InvalidSpecification` unless the initial state is an `m`-vector and the
/// initial covariance `m × m`.
fn check_initial_values(
    m: usize,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
) -> EconResult<()> {
    if init_state.len() != m {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Initial state length ({}) must match state dimension ({})",
                init_state.len(),
                m
            ),
        });
    }

    if init_cov.dim() != (m, m) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Initial covariance dimensions ({:?}) must be ({}, {})",
                init_cov.dim(),
                m,
                m
            ),
        });
    }
    Ok(())
}

/// The Kalman filter recursions over `y`, returning the log-likelihood and,
/// when `history` is given, pushing every step's states, covariances,
/// innovation and innovation variance onto it. The initial values must have
/// been checked with [`check_initial_values`].
fn filter_pass(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
    mut history: Option<&mut FilterHistory>,
) -> EconResult<f64> {
    let n = y.len();
    let m = model.state_dim();

    // Pre-compute R Q R'
    let state_dist_cov = model.state_disturbance_cov();

    // Initialize
    let mut state = init_state.to_owned();
    let mut cov = init_cov.to_owned();

    let mut log_lik = 0.0;
    let mut n_valid = 0usize;

    // Each step costs O(m³): check the budget about every 65,536 operations.
    let stride = checkpoint_stride(m.saturating_mul(m).saturating_mul(m));
    for t in 0..n {
        if t % stride == 0 {
            checkpoint()?;
        }
        // Prediction step
        let pred_state = model.transition.dot(&state);
        let pred_cov = model.transition.dot(&cov).dot(&model.transition.t()) + &state_dist_cov;

        if let Some(h) = history.as_deref_mut() {
            h.predicted_states.push(pred_state.to_vec());
            h.predicted_covs.push(array2_to_nested_vec(&pred_cov));
        }

        // Update step (if observation is not missing)
        if y[t].is_nan() {
            // Missing observation: no update
            state = pred_state;
            cov = pred_cov;
            if let Some(h) = history.as_deref_mut() {
                h.innovations.push(f64::NAN);
                h.innovation_vars.push(f64::NAN);
            }
        } else {
            // Innovation
            let pred_obs = model.observation.dot(&pred_state);
            let v = y[t] - pred_obs;

            // Innovation variance: F = Z' P Z + H
            let f = model.observation.dot(&pred_cov.dot(&model.observation)) + model.obs_var;

            if let Some(h) = history.as_deref_mut() {
                h.innovations.push(v);
                h.innovation_vars.push(f);
            }

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

        if let Some(h) = history.as_deref_mut() {
            h.filtered_states.push(state.to_vec());
            h.filtered_covs.push(array2_to_nested_vec(&cov));
        }
    }

    // Add constant term to log-likelihood
    log_lik -= 0.5 * n_valid as f64 * (2.0 * std::f64::consts::PI).ln();

    Ok(log_lik)
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
/// `KalmanSmootherResult` with smoothed states and covariances, or
/// [`EconError::Cancelled`] when the budget is cancelled or past its deadline
/// (checked as in [`kalman_filter`]). Its states and covariances, nested and
/// counted as the filter's are (n·(8m² + 48m + 80) bytes once m ≥ 2, 152
/// bytes per step for m = 1), are admitted before any work:
/// [`EconError::ResourceLimit`] when their size overflows or exceeds the
/// per-allocation ceiling.
pub fn kalman_smoother(
    filter_result: &KalmanFilterResult,
    model: &StateSpaceModel,
) -> EconResult<KalmanSmootherResult> {
    let n = filter_result.n_obs;
    let m = filter_result.state_dim;

    // An empty filter result has no final state to seed the backward pass;
    // indexing `[n - 1]` would underflow/panic. Reject it cleanly instead.
    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Kalman smoother requires a non-empty filter result".to_string(),
        });
    }

    // α_{t|n} and P_{t|n} for every step: admit them before any work.
    admit_smoother_history(n, m)?;
    let what = smoother_history_what(n, m);

    // Built backward from t = n - 1 and reversed at the end, so nothing of
    // size n·m² is allocated or zero-filled before the first checkpoint.
    let mut smoothed_states: Vec<Vec<f64>> = try_with_capacity(n, &what)?;
    let mut smoothed_covs: Vec<Vec<Vec<f64>>> = try_with_capacity(n, &what)?;

    // Initialize with final filtered values
    smoothed_states.push(filter_result.filtered_states[n - 1].clone());
    smoothed_covs.push(filter_result.filtered_covs[n - 1].clone());

    // Backward recursion; each step costs O(m³) (a matrix inverse and products)
    let stride = checkpoint_stride(m.saturating_mul(m).saturating_mul(m));
    for (step, t) in (0..n - 1).rev().enumerate() {
        if step % stride == 0 {
            checkpoint()?;
        }
        let filtered_state = vec_to_array1(&filter_result.filtered_states[t]);
        let filtered_cov = nested_vec_to_array2(&filter_result.filtered_covs[t]);
        let predicted_state_next = vec_to_array1(&filter_result.predicted_states[t + 1]);
        let predicted_cov_next = nested_vec_to_array2(&filter_result.predicted_covs[t + 1]);
        // α_{t+1|n} and P_{t+1|n} were pushed last, at index `step`.
        let smoothed_state_next = vec_to_array1(&smoothed_states[step]);
        let smoothed_cov_next = nested_vec_to_array2(&smoothed_covs[step]);

        // Compute smoother gain
        // L = P_{t|t} T' P_{t+1|t}^{-1}
        let (pred_cov_inv, _) = safe_inverse(&predicted_cov_next.view()).map_err(|e| {
            EconError::InvalidSpecification {
                message: format!("Failed to invert predicted covariance: {}", e),
            }
        })?;

        let smoother_gain = filtered_cov.dot(&model.transition.t()).dot(&pred_cov_inv);

        // Smoothed state: α_{t|n} = α_{t|t} + L (α_{t+1|n} - α_{t+1|t})
        let state_diff = &smoothed_state_next - &predicted_state_next;
        let smoothed_state = &filtered_state + &smoother_gain.dot(&state_diff);

        // Smoothed covariance: P_{t|n} = P_{t|t} + L (P_{t+1|n} - P_{t+1|t}) L'
        let cov_diff = &smoothed_cov_next - &predicted_cov_next;
        let smoothed_cov = &filtered_cov + &smoother_gain.dot(&cov_diff).dot(&smoother_gain.t());

        smoothed_states.push(smoothed_state.to_vec());
        smoothed_covs.push(array2_to_nested_vec(&smoothed_cov));
    }

    // Back to time order: index t holds α_{t|n} and P_{t|n}.
    smoothed_states.reverse();
    smoothed_covs.reverse();

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
/// `KalmanForecastResult` with forecasts and prediction intervals, or
/// [`EconError::Cancelled`] when the budget is cancelled or past its deadline
/// (checked every step). Its state forecasts, nested as the filter's states
/// are, and its forecasts, variances and standard errors (8m + 64 bytes a
/// step once m ≥ 2, 80 bytes for m = 1, whose values are only 32) are
/// admitted before any work: [`EconError::ResourceLimit`] when their size
/// overflows or exceeds the per-allocation ceiling.
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

    const WHAT: &str = "the Kalman forecast (horizon × state dimension)";

    let n = filter_result.n_obs;
    let m = filter_result.state_dim;

    // Admit what the forecast keeps, counted by the bytes its nested Vecs
    // hold, before any work.
    admit_bytes(&[history_bytes(horizon, m, FORECAST_STEP, WHAT)?], WHAT)?;

    // Forecasting starts from the final filtered state; an empty filter result
    // has none, so indexing `[n - 1]` would underflow/panic.
    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Kalman forecast requires a non-empty filter result".to_string(),
        });
    }

    // Start from final filtered state
    let mut state = vec_to_array1(&filter_result.filtered_states[n - 1]);
    let mut cov = nested_vec_to_array2(&filter_result.filtered_covs[n - 1]);

    let state_dist_cov = model.state_disturbance_cov();

    let mut forecasts = try_with_capacity(horizon, WHAT)?;
    let mut variances = try_with_capacity(horizon, WHAT)?;
    let mut std_errors = try_with_capacity(horizon, WHAT)?;
    let mut state_forecasts: Vec<Vec<f64>> = try_with_capacity(horizon, WHAT)?;

    for _ in 0..horizon {
        // Each step costs O(m³).
        checkpoint()?;
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
///
/// It runs the recursions of [`kalman_filter`] and returns the same value,
/// but keeps no per-step history, so its memory does not grow with the series
/// and it is never refused for the history's size. Errors as
/// [`kalman_filter`] does otherwise, including [`EconError::Cancelled`] on a
/// stopped budget.
pub fn kalman_loglik(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
) -> EconResult<f64> {
    check_initial_values(model.state_dim(), init_state, init_cov)?;
    filter_pass(y, model, init_state, init_cov, None)
}

// History admission

/// Bytes glibc's malloc (the allocator of the Linux builds, the deployed
/// image among them) may add to a request: an 8-byte chunk header, then
/// rounding up to 16 bytes, so a buffer of a multiple of 8 bytes, as every
/// buffer here is, takes at most 16 bytes more than it asks for.
const MALLOC_CHUNK_OVERHEAD: usize = 16;
/// glibc's smallest chunk: no buffer takes less, however small.
const MALLOC_MIN_CHUNK: usize = 32;

/// What one step of a Kalman pass keeps.
#[derive(Clone, Copy, Debug)]
struct Step {
    /// m × m covariances, each a `Vec<Vec<f64>>`.
    covs: usize,
    /// m-vector states, each a `Vec<f64>`.
    states: usize,
    /// Single `f64` values, each a slot in an outer `Vec<f64>`.
    scalars: usize,
}

/// A [`kalman_filter`] step: the predicted and the filtered covariance and
/// state, the innovation and its variance.
const FILTER_STEP: Step = Step {
    covs: 2,
    states: 2,
    scalars: 2,
};
/// A [`kalman_smoother`] step: the smoothed covariance and state.
const SMOOTHER_STEP: Step = Step {
    covs: 1,
    states: 1,
    scalars: 0,
};
/// A [`kalman_forecast`] step: the state forecast, and the forecast, its
/// variance and its standard error.
const FORECAST_STEP: Step = Step {
    covs: 0,
    states: 1,
    scalars: 3,
};

fn filter_history_what(n: usize, m: usize) -> String {
    format!(
        "the Kalman filter's state and covariance history for {n} observations \
         and a {m}-dimensional state"
    )
}

fn smoother_history_what(n: usize, m: usize) -> String {
    format!(
        "the Kalman smoother's state and covariance history for {n} observations \
         and a {m}-dimensional state"
    )
}

/// `ResourceLimit` unless the byte counts in `parts`, summed, fit the
/// per-allocation ceiling; `what` names them all.
fn admit_bytes(parts: &[usize], what: &str) -> EconResult<()> {
    let total = parts
        .iter()
        .try_fold(0usize, |total, &bytes| total.checked_add(bytes))
        .ok_or_else(|| overflow(what))?;
    check_alloc::<u8>(total, what)
}

/// Check that the history [`kalman_filter`] keeps for `n` observations of an
/// `m`-dimensional state ([`history_bytes`] of a [`FILTER_STEP`]) fits the
/// per-allocation ceiling: `ResourceLimit` when it overflows or does not.
fn admit_filter_history(n: usize, m: usize) -> EconResult<()> {
    let what = filter_history_what(n, m);
    admit_bytes(&[history_bytes(n, m, FILTER_STEP, &what)?], &what)
}

/// [`admit_filter_history`] for the history [`kalman_smoother`] keeps.
fn admit_smoother_history(n: usize, m: usize) -> EconResult<()> {
    let what = smoother_history_what(n, m);
    admit_bytes(&[history_bytes(n, m, SMOOTHER_STEP, &what)?], &what)
}

/// Check that the histories [`kalman_filter`] and then [`kalman_smoother`]
/// keep for `n` observations of an `m`-dimensional state fit the
/// per-allocation ceiling as one figure, since a caller that smooths the
/// filter's result holds both at once: n·(24m² + 144m + 256) bytes once
/// m ≥ 2, 472 bytes per step for m = 1. `ResourceLimit` when the sum
/// overflows or does not fit. Callers that run the passes after other work
/// (an optimiser) call this first, so that a refusal stops them before that
/// work; the passes then admit their own histories again.
pub(crate) fn admit_filter_and_smoother_history(n: usize, m: usize) -> EconResult<()> {
    let what = format!(
        "the Kalman filter's and smoother's state and covariance histories for {n} \
         observations and a {m}-dimensional state"
    );
    admit_bytes(
        &[
            history_bytes(n, m, FILTER_STEP, &what)?,
            history_bytes(n, m, SMOOTHER_STEP, &what)?,
        ],
        &what,
    )
}

/// [`admit_filter_and_smoother_history`] for a caller that holds the history
/// [`kalman_filter`] keeps for `n` observations together with a
/// [`kalman_forecast`] of `horizon` steps from it (8m + 64 bytes a step once
/// m ≥ 2, 80 bytes for m = 1).
pub(crate) fn admit_filter_history_and_forecast(
    n: usize,
    m: usize,
    horizon: usize,
) -> EconResult<()> {
    let what = format!(
        "the Kalman filter's state and covariance history for {n} observations \
         and a {m}-dimensional state, with its {horizon}-step forecast"
    );
    admit_bytes(
        &[
            history_bytes(n, m, FILTER_STEP, &what)?,
            history_bytes(horizon, m, FORECAST_STEP, &what)?,
        ],
        &what,
    )
}

/// The heap bytes a buffer of `bytes` bytes occupies (see
/// [`MALLOC_CHUNK_OVERHEAD`] and [`MALLOC_MIN_CHUNK`]).
fn heap_chunk(bytes: usize, what: &str) -> EconResult<usize> {
    bytes
        .checked_add(MALLOC_CHUNK_OVERHEAD)
        .map(|b| b.max(MALLOC_MIN_CHUNK))
        .ok_or_else(|| overflow(what))
}

/// The bytes a Kalman pass keeps over `n` steps of an `m`-dimensional state
/// when each step stores what `step` says, counted as it is stored: a state
/// is a `Vec<f64>`, its header a slot in the outer buffer and its m values a
/// heap buffer of their own; a covariance is a `Vec<Vec<f64>>`, its header a
/// slot in the outer buffer, a heap buffer of m row headers and m row buffers
/// of m values each; a scalar is a slot in an outer `Vec<f64>`. Every heap
/// buffer is charged as [`heap_chunk`] does (the outer buffers' own overhead,
/// at most a page each, aside). `ResourceLimit` naming `what` when the count
/// overflows.
fn history_bytes(n: usize, m: usize, step: Step, what: &str) -> EconResult<usize> {
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(|| overflow(what));
    let row = heap_chunk(checked_len(&[m, size_of::<f64>()], what)?, what)?;
    let state = add(size_of::<Vec<f64>>(), row)?;
    let mut per_step = add(
        checked_len(&[step.states, state], what)?,
        checked_len(&[step.scalars, size_of::<f64>()], what)?,
    )?;
    // Only a pass that keeps covariances is charged for one, so that a
    // forecast's count does not overflow on an m × m it never stores.
    if step.covs > 0 {
        let row_headers = heap_chunk(checked_len(&[m, size_of::<Vec<f64>>()], what)?, what)?;
        let cov = add(
            add(size_of::<Vec<Vec<f64>>>(), row_headers)?,
            checked_len(&[m, row], what)?,
        )?;
        per_step = add(per_step, checked_len(&[step.covs, cov], what)?)?;
    }
    checked_len(&[n, per_step], what)
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
        let state_cov = array![[0.1]]; // σ²_η
        let obs_var = 0.2; // σ²_ε

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

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

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

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

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

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

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

        // Data with missing value
        let y = vec![1.0, f64::NAN, 2.0, 2.2, 2.5];
        let init_state = array![y[0]];
        let init_cov = array![[1.0]];

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert_eq!(result.n_obs, 5);
        assert!(result.innovations[1].is_nan()); // Missing observation
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_local_linear_trend() {
        // Local linear trend model:
        // μ_{t+1} = μ_t + ν_t + ξ_t
        // ν_{t+1} = ν_t + ζ_t
        // y_t = μ_t + ε_t

        let transition = array![[1.0, 1.0], [0.0, 1.0]];
        let observation = array![1.0, 0.0]; // Observe level only
        let selection = array![[1.0, 0.0], [0.0, 1.0]];
        let state_cov = array![[0.1, 0.0], [0.0, 0.01]];
        let obs_var = 0.2;

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

        // Generate trending data
        let y: Vec<f64> = (0..10)
            .map(|t| 1.0 + 0.5 * t as f64 + 0.1 * (t as f64).sin())
            .collect();

        let init_state = array![y[0], 0.5]; // level = y[0], slope = 0.5
        let init_cov = array![[1.0, 0.0], [0.0, 0.1]];

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert_eq!(result.state_dim, 2);
        assert!(result.log_likelihood.is_finite());

        // Slope estimate (second state) should be positive for trending data
        let final_slope = result.filtered_states[result.n_obs - 1][1];
        assert!(final_slope > 0.0);
    }

    #[test]
    fn test_validate_kalman_local_level_tracks_signal() {
        // Generate data from a local level model with known variances.
        // Signal: mu_t = mu_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta)
        // Obs:    y_t  = mu_t + eps_t,     eps_t ~ N(0, sigma2_eps)
        //
        // The filter should track the true signal.
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let sigma2_eta: f64 = 0.5; // state noise variance
        let sigma2_eps: f64 = 1.0; // observation noise variance
        let n = 200;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut true_signal = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut mu = 10.0;

        for _ in 0..n {
            mu += (rng.r#gen::<f64>() - 0.5) * 2.0 * sigma2_eta.sqrt();
            true_signal.push(mu);
            y.push(mu + (rng.r#gen::<f64>() - 0.5) * 2.0 * sigma2_eps.sqrt());
        }

        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[sigma2_eta]];
        let obs_var = sigma2_eps;

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

        let init_state = array![y[0]];
        let init_cov = array![[10.0]]; // diffuse

        let result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        // Filtered states should be closer to the true signal than the raw observations.
        let mut filter_mse = 0.0;
        let mut obs_mse = 0.0;
        for t in 10..n {
            // skip first 10 for burn-in
            let filter_err = result.filtered_states[t][0] - true_signal[t];
            let obs_err = y[t] - true_signal[t];
            filter_mse += filter_err * filter_err;
            obs_mse += obs_err * obs_err;
        }

        assert!(
            filter_mse < obs_mse,
            "Filtered MSE ({:.4}) should be less than observation MSE ({:.4})",
            filter_mse,
            obs_mse
        );
    }

    #[test]
    fn test_validate_kalman_loglikelihood_finite_negative() {
        // Log-likelihood should be finite and negative for any valid model.
        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.5]];
        let obs_var = 1.0;

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

        let y: Vec<f64> = (0..50)
            .map(|t| (t as f64) * 0.3 + 0.5 * (t as f64 * 0.2).sin())
            .collect();
        let init_state = array![y[0]];
        let init_cov = array![[10.0]];

        let ll = kalman_loglik(&y, &model, init_state.view(), init_cov.view()).unwrap();

        assert!(
            ll.is_finite(),
            "Log-likelihood should be finite, got {}",
            ll
        );
        assert!(ll < 0.0, "Log-likelihood should be negative, got {}", ll);
    }

    #[test]
    fn test_validate_kalman_smoother_tighter_than_filter() {
        // Smoother uses all data (backward pass), so smoothed state covariance
        // should be <= filtered covariance at each time point.
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let n = 100;
        let mut rng = ChaCha8Rng::seed_from_u64(88);

        let transition = array![[1.0]];
        let observation = array![1.0];
        let selection = array![[1.0]];
        let state_cov = array![[0.3]];
        let obs_var = 0.5;

        let model =
            StateSpaceModel::new(transition, observation, selection, state_cov, obs_var).unwrap();

        let mut mu = 5.0;
        let y: Vec<f64> = (0..n)
            .map(|_| {
                mu += (rng.r#gen::<f64>() - 0.5) * 1.0;
                mu + (rng.r#gen::<f64>() - 0.5) * 1.5
            })
            .collect();

        let init_state = array![y[0]];
        let init_cov = array![[5.0]];

        let filter_result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();
        let smooth_result = kalman_smoother(&filter_result, &model).unwrap();

        // Check that smoothed covariance <= filtered covariance for interior points.
        // The last point is identical by construction, so skip it.
        let mut count_tighter = 0;
        for t in 0..(n - 1) {
            let filter_var = filter_result.filtered_covs[t][0][0];
            let smooth_var = smooth_result.smoothed_covs[t][0][0];
            if smooth_var <= filter_var + 1e-10 {
                count_tighter += 1;
            }
        }

        // Allow for small numerical issues: at least 90% of points should satisfy the property
        let ratio = count_tighter as f64 / (n - 1) as f64;
        assert!(
            ratio > 0.9,
            "Smoothed covariance should be <= filtered covariance in at least 90% of points, got {:.1}%",
            ratio * 100.0
        );
    }

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    fn local_level() -> (StateSpaceModel, KalmanFilterResult) {
        let model = StateSpaceModel::new(
            array![[1.0]],
            array![1.0],
            array![[1.0]],
            array![[0.1]],
            0.2,
        )
        .unwrap();
        let y = vec![1.0, 1.5, 2.0, 2.2, 2.5];
        let f = kalman_filter(&y, &model, array![y[0]].view(), array![[1.0]].view()).unwrap();
        (model, f)
    }

    #[test]
    fn kalman_forecast_refuses_absurd_horizon_and_honours_cancellation() {
        let (model, f) = local_level();
        assert!(matches!(
            kalman_forecast(&f, &model, 1usize << 50),
            Err(EconError::ResourceLimit(_))
        ));
        assert!(matches!(
            cancelled(|| kalman_forecast(&f, &model, 3)),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn kalman_filter_and_smoother_honour_cancellation() {
        // Each step costs O(m³); StructTS calls the filter once per likelihood
        // evaluation, so a long series or a large state stops at the budget.
        // Before the fix both ran every step and returned Ok.
        let (model, f) = local_level();
        let y = vec![1.0, 1.5, 2.0, 2.2, 2.5];
        assert!(matches!(
            cancelled(|| kalman_filter(&y, &model, array![y[0]].view(), array![[1.0]].view())),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            cancelled(|| kalman_smoother(&f, &model)),
            Err(EconError::Cancelled(_))
        ));
        assert!(kalman_smoother(&f, &model).is_ok());
    }

    /// `m` independent random-walk states whose sum is observed with noise.
    fn summed_random_walks(m: usize) -> StateSpaceModel {
        StateSpaceModel::new(
            Array2::eye(m),
            Array1::from_elem(m, 1.0),
            Array2::eye(m),
            Array2::eye(m) * 0.1,
            0.5,
        )
        .unwrap()
    }

    fn noisy_walk(n: usize, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut level = 3.0;
        (0..n)
            .map(|_| {
                level += rng.r#gen::<f64>() - 0.5;
                level + 0.7 * (rng.r#gen::<f64>() - 0.5)
            })
            .collect()
    }

    fn filter_summed_walks(y: &[f64], m: usize) -> EconResult<KalmanFilterResult> {
        let model = summed_random_walks(m);
        let init_state = Array1::zeros(m);
        let init_cov = Array2::eye(m) * 10.0;
        kalman_filter(y, &model, init_state.view(), init_cov.view())
    }

    #[test]
    fn kalman_filter_admits_its_covariance_history_before_the_loop() {
        // The filter keeps P_{t|t-1} and P_{t|t} for every step, 2·n·m² values,
        // in nested Vecs that no allocation guard saw, so with a large state
        // dimension (a seasonal model with a long period) memory grew without
        // bound. Under a per-allocation ceiling the history is now admitted
        // (or refused) as a whole before the first step; before the fix this
        // returned Ok.
        let (n, m) = (50, 20);
        let y = noisy_walk(n, 5);
        // 50 · (16·20² + 96·20 + 176) = 424,800 bytes > 64 KiB.
        let refused =
            crate::resource::with_alloc_ceiling(Some(64 << 10), || filter_summed_walks(&y, m));
        match refused {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("Kalman filter")
                    && msg.contains("50 observations")
                    && msg.contains("20-dimensional state"),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // A history under the ceiling (5 · 8,496 = 42,480 bytes) runs.
        let short =
            crate::resource::with_alloc_ceiling(Some(64 << 10), || filter_summed_walks(&y[..5], m));
        assert_eq!(short.unwrap().filtered_covs.len(), 5);
        // Without a ceiling nothing changes.
        assert!(filter_summed_walks(&y, m).is_ok());
    }

    #[test]
    fn kalman_smoother_admits_its_covariance_history_before_allocating() {
        // The smoother zero-filled n·m² values with `vec!` before its first
        // step. Before the fix this returned Ok under the ceiling.
        let (n, m) = (50, 20);
        let model = summed_random_walks(m);
        let f = filter_summed_walks(&noisy_walk(n, 9), m).unwrap();
        // 50 · (8·20² + 48·20 + 80) = 212,000 bytes > 64 KiB, < 256 KiB.
        let refused =
            crate::resource::with_alloc_ceiling(Some(64 << 10), || kalman_smoother(&f, &model));
        match refused {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("Kalman smoother") && msg.contains("50 observations"),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        assert!(
            crate::resource::with_alloc_ceiling(Some(256 << 10), || kalman_smoother(&f, &model))
                .is_ok()
        );
    }

    #[test]
    fn kalman_histories_are_admitted_by_the_bytes_they_hold() {
        // A one-dimensional state (the local level model) keeps 48 bytes of
        // values a filter step, but every state and covariance row is a heap
        // buffer of its own: 320 bytes a step are charged now, and glibc holds
        // about 300. Counting values admitted histories several times the
        // ceiling. Before the fix every refusal below returned Ok.
        let y = noisy_walk(2000, 41);
        let filter = |ceiling: usize, n: usize| {
            crate::resource::with_alloc_ceiling(Some(ceiling), || filter_summed_walks(&y[..n], 1))
        };
        // 1,000 · 320 = 320,000 bytes > 64 KiB (the values: 48,000 bytes);
        // 2,000 · 320 = 640,000 bytes > 256 KiB (the values: 96,000 bytes).
        for (ceiling, n) in [(64 << 10, 1000), (256 << 10, 2000)] {
            match filter(ceiling, n) {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.contains("Kalman filter")
                        && msg.contains(&format!("{n} observations"))
                        && msg.contains("1-dimensional state"),
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit for n = {n}, got {other:?}"),
            }
        }
        // 200 · 320 = 64,000 bytes < 64 KiB: admitted.
        assert_eq!(filter(64 << 10, 200).unwrap().filtered_covs.len(), 200);

        // The smoother keeps 16 bytes of values a step and is charged 152:
        // 2,000 · 152 = 304,000 bytes > 256 KiB (the values: 32,000 bytes).
        let level = summed_random_walks(1);
        let f = filter_summed_walks(&y, 1).unwrap();
        for ceiling in [64 << 10, 256 << 10] {
            match crate::resource::with_alloc_ceiling(Some(ceiling), || kalman_smoother(&f, &level))
            {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.contains("Kalman smoother") && msg.contains("2000 observations"),
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit under {ceiling} bytes, got {other:?}"),
            }
        }
        // 400 · 152 = 60,800 bytes < 64 KiB: admitted.
        let short = filter_summed_walks(&y[..400], 1).unwrap();
        assert!(
            crate::resource::with_alloc_ceiling(Some(64 << 10), || kalman_smoother(&short, &level))
                .is_ok()
        );
    }

    #[test]
    fn history_bytes_charges_every_heap_buffer() {
        let bytes = |n, m, step| history_bytes(n, m, step, "t").unwrap();
        // m = 1: a state is 24 + 32 bytes, a covariance 24 + 40 + 32.
        assert_eq!(bytes(1, 1, FILTER_STEP), 320);
        assert_eq!(bytes(1, 1, SMOOTHER_STEP), 152);
        assert_eq!(bytes(1, 1, FORECAST_STEP), 80);
        assert_eq!(bytes(1000, 1, FILTER_STEP), 320_000);
        // m ≥ 2: the closed forms the kalman_filter, kalman_smoother and
        // kalman_forecast docs give, and the sum that
        // admit_filter_and_smoother_history admits.
        for m in [2, 13, 20, 1001] {
            let filter = bytes(1, m, FILTER_STEP);
            let smoother = bytes(1, m, SMOOTHER_STEP);
            assert_eq!(filter, 16 * m * m + 96 * m + 176, "m = {m}");
            assert_eq!(smoother, 8 * m * m + 48 * m + 80, "m = {m}");
            assert_eq!(filter + smoother, 24 * m * m + 144 * m + 256, "m = {m}");
            assert_eq!(bytes(1, m, FORECAST_STEP), 8 * m + 64, "m = {m}");
        }
        // For a large state the overhead is small: within 1% of the values.
        let values = 8 * (2 * (1001 * 1001 + 1001) + 2);
        let charged = bytes(1, 1001, FILTER_STEP);
        assert!(charged > values && (charged as f64) < 1.01 * values as f64);
        // A size that overflows is a ResourceLimit, not a panic or a wrap.
        for (n, m) in [(usize::MAX, 1), (2, usize::MAX / 4), (2, 1 << 31)] {
            assert!(matches!(
                history_bytes(n, m, FILTER_STEP, "t"),
                Err(EconError::ResourceLimit(_))
            ));
        }
        // A forecast keeps no covariance, so an m whose m × m overflows is
        // still counted by its states alone.
        let m = 1usize << 40;
        assert!(history_bytes(1, m, FILTER_STEP, "t").is_err());
        assert_eq!(bytes(1, m, FORECAST_STEP), 8 * m + 64);
    }

    #[test]
    fn admissions_sum_the_passes_a_caller_holds_at_once() {
        let under = |ceiling: usize, f: &dyn Fn() -> EconResult<()>| {
            crate::resource::with_alloc_ceiling(Some(ceiling), f)
        };
        // Filter and smoother, m = 1, n = 150: 48,000 and 22,800 bytes each
        // fit 64 KiB, but a caller that smooths holds both, 70,800 bytes.
        // Admitted one at a time, the pair could hold up to about 1.5 times
        // the ceiling.
        assert!(under(64 << 10, &|| admit_filter_history(150, 1)).is_ok());
        assert!(under(64 << 10, &|| admit_smoother_history(150, 1)).is_ok());
        match under(64 << 10, &|| admit_filter_and_smoother_history(150, 1)) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("Kalman filter's and smoother's")
                    && msg.contains("150 observations and a 1-dimensional state"),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // 130 · 472 = 61,360 bytes fit.
        assert!(under(64 << 10, &|| admit_filter_and_smoother_history(130, 1)).is_ok());

        // Filter and forecast, m = 1: 100 · 320 = 32,000 and 500 · 80 =
        // 40,000 bytes each fit 64 KiB; together, 72,000 do not.
        assert!(under(64 << 10, &|| admit_filter_history(100, 1)).is_ok());
        match under(64 << 10, &|| admit_filter_history_and_forecast(100, 1, 500)) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("100 observations and a 1-dimensional state")
                    && msg.contains("500-step forecast"),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // 32,000 + 300 · 80 = 56,000 bytes fit.
        assert!(under(64 << 10, &|| admit_filter_history_and_forecast(100, 1, 300)).is_ok());
        // A sum that overflows is a ResourceLimit, not a panic or a wrap.
        assert!(matches!(
            admit_filter_history_and_forecast(1, 1, usize::MAX / 80),
            Err(EconError::ResourceLimit(_))
        ));
    }

    #[test]
    fn kalman_forecast_is_admitted_by_the_bytes_it_holds() {
        // At m = 1 a forecast step keeps 32 bytes of values (the state and
        // three scalars), but the state is a heap buffer of its own: 80 bytes
        // are charged. The old admission counted horizon × m values, 8 bytes
        // a step, so it let through a forecast ten times its figure.
        let (model, f) = local_level();
        // 1,000 · 80 = 80,000 bytes > 64 KiB (the old count: 8,000 bytes).
        match crate::resource::with_alloc_ceiling(Some(64 << 10), || {
            kalman_forecast(&f, &model, 1000)
        }) {
            Err(EconError::ResourceLimit(msg)) => {
                assert!(msg.contains("Kalman forecast"), "{msg}")
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // 800 · 80 = 64,000 bytes fit.
        let fits = crate::resource::with_alloc_ceiling(Some(64 << 10), || {
            kalman_forecast(&f, &model, 800)
        });
        assert_eq!(fits.unwrap().state_forecasts.len(), 800);
        // Without a ceiling nothing changes.
        assert_eq!(
            kalman_forecast(&f, &model, 1000).unwrap().forecasts.len(),
            1000
        );
    }

    #[test]
    fn kalman_loglik_keeps_no_history() {
        // An optimiser needs only the likelihood. Had kalman_loglik run
        // kalman_filter, which admits its history (424,800 bytes here), the
        // call under this ceiling would be ResourceLimit; it keeps none and
        // returns the same value. A guard against it going back to the
        // filter, not a test of the history admission itself.
        let (n, m) = (50, 20);
        let mut y = noisy_walk(n, 13);
        y[4] = f64::NAN;
        let model = summed_random_walks(m);
        let init_state = Array1::zeros(m);
        let init_cov = Array2::eye(m) * 10.0;
        let want = kalman_filter(&y, &model, init_state.view(), init_cov.view())
            .unwrap()
            .log_likelihood;
        let got = crate::resource::with_alloc_ceiling(Some(64 << 10), || {
            kalman_loglik(&y, &model, init_state.view(), init_cov.view())
        });
        assert_eq!(got.unwrap().to_bits(), want.to_bits());
        // It still checks the initial values and stops on a stopped budget.
        assert!(matches!(
            kalman_loglik(&y, &model, Array1::zeros(m - 1).view(), init_cov.view()),
            Err(EconError::InvalidSpecification { .. })
        ));
        assert!(matches!(
            cancelled(|| kalman_loglik(&y, &model, init_state.view(), init_cov.view())),
            Err(EconError::Cancelled(_))
        ));
    }

    /// The smoother as it was before it built its output backward: forward
    /// indexed into zero-filled n·m² buffers.
    fn reference_smoother(f: &KalmanFilterResult, model: &StateSpaceModel) -> KalmanSmootherResult {
        let (n, m) = (f.n_obs, f.state_dim);
        let mut states = vec![vec![0.0; m]; n];
        let mut covs = vec![vec![vec![0.0; m]; m]; n];
        states[n - 1] = f.filtered_states[n - 1].clone();
        covs[n - 1] = f.filtered_covs[n - 1].clone();
        for t in (0..n - 1).rev() {
            let filtered_state = vec_to_array1(&f.filtered_states[t]);
            let filtered_cov = nested_vec_to_array2(&f.filtered_covs[t]);
            let predicted_state_next = vec_to_array1(&f.predicted_states[t + 1]);
            let predicted_cov_next = nested_vec_to_array2(&f.predicted_covs[t + 1]);
            let smoothed_state_next = vec_to_array1(&states[t + 1]);
            let smoothed_cov_next = nested_vec_to_array2(&covs[t + 1]);
            let (pred_cov_inv, _) = safe_inverse(&predicted_cov_next.view()).unwrap();
            let gain = filtered_cov.dot(&model.transition.t()).dot(&pred_cov_inv);
            let state =
                &filtered_state + &gain.dot(&(&smoothed_state_next - &predicted_state_next));
            let cov = &filtered_cov
                + &gain
                    .dot(&(&smoothed_cov_next - &predicted_cov_next))
                    .dot(&gain.t());
            states[t] = state.to_vec();
            covs[t] = array2_to_nested_vec(&cov);
        }
        KalmanSmootherResult {
            smoothed_states: states,
            smoothed_covs: covs,
            n_obs: n,
            state_dim: m,
        }
    }

    #[test]
    fn kalman_smoother_matches_the_forward_indexed_reference() {
        // The smoother now builds its output backward and reverses it instead
        // of zero-filling n·m² values up front; the numbers must not change.
        let trend = StateSpaceModel::new(
            array![[1.0, 1.0], [0.0, 1.0]],
            array![1.0, 0.0],
            Array2::eye(2),
            array![[0.1, 0.0], [0.0, 0.01]],
            0.2,
        )
        .unwrap();
        let mut y = noisy_walk(40, 21);
        y[7] = f64::NAN;
        y[8] = f64::NAN;
        let trend_f = kalman_filter(
            &y,
            &trend,
            array![y[0], 0.0].view(),
            (Array2::eye(2) * 5.0).view(),
        )
        .unwrap();
        let walks = summed_random_walks(6);
        let walks_f = filter_summed_walks(&noisy_walk(30, 22), 6).unwrap();
        let single_f = filter_summed_walks(&[1.3], 1).unwrap();
        for (f, model) in [
            (&trend_f, &trend),
            (&walks_f, &walks),
            (&single_f, &summed_random_walks(1)),
        ] {
            let got = kalman_smoother(f, model).unwrap();
            let want = reference_smoother(f, model);
            assert_eq!((got.n_obs, got.state_dim), (want.n_obs, want.state_dim));
            assert_eq!(got.smoothed_states.len(), want.smoothed_states.len());
            assert_eq!(got.smoothed_covs.len(), want.smoothed_covs.len());
            for (a, b) in got.smoothed_states.iter().zip(&want.smoothed_states) {
                for (x, z) in a.iter().zip(b) {
                    assert!((x - z).abs() <= 1e-12, "{x} vs {z}");
                }
            }
            for (a, b) in got.smoothed_covs.iter().zip(&want.smoothed_covs) {
                for (ra, rb) in a.iter().zip(b) {
                    for (x, z) in ra.iter().zip(rb) {
                        assert!((x - z).abs() <= 1e-12, "{x} vs {z}");
                    }
                }
            }
        }
    }
}
