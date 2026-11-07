//! Kalman filtering, smoothing, and forecasting for state-space models.
//!
//! Implements the Kalman filter for univariate time series in state-space form:
//!
//! State equation:     α_{t+1} = T α_t + c + R η_t,  η_t ~ N(0, Q)
//! Observation eq:     y_t = Z' α_t + d + ε_t,       ε_t ~ N(0, H)
//!
//! The filter predicts first: `init_state` is a_0 and `init_cov` is P_0, and the
//! first step predicts a_1 = T a_0 + c, as R's `KalmanRun(nit = -1)` does. The
//! covariance update is P = P⁻ − M M' / F with M = P⁻ Z, computed from the sparsity
//! of T and Z without an allocation per step; the smoother is the Durbin-Koopman
//! state smoother, which needs no inverse of a predicted covariance, so a singular
//! one (a zero variance) is an ordinary case.
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
use crate::resource::{
    check_alloc, checked_len, checkpoint, checkpoint_stride, overflow, try_vec, try_with_capacity,
    try_zeros2,
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

    /// Set the state intercept c (length m) and the observation intercept d,
    /// and check the model ([`Self::validate`]).
    pub fn with_intercepts(
        mut self,
        state_intercept: Option<Array1<f64>>,
        obs_intercept: Option<f64>,
    ) -> EconResult<Self> {
        self.state_intercept = state_intercept;
        self.obs_intercept = obs_intercept;
        self.validate()?;
        Ok(self)
    }

    /// `InvalidSpecification` unless the state dimension is at least 1 and the
    /// dimensions are those [`Self::new`] requires, every matrix entry is finite, `obs_var` and the diagonal of
    /// `state_cov` are at least 0, `state_intercept` has length m and is
    /// finite, and `obs_intercept` is finite. The filter, the smoother and the
    /// forecast call it, because the fields are public.
    pub fn validate(&self) -> EconResult<()> {
        let invalid = |message: String| Err(EconError::InvalidSpecification { message });
        let m = self.transition.nrows();
        if m == 0 {
            return invalid("the state dimension must be at least 1, got 0".to_string());
        }
        if self.transition.ncols() != m {
            return invalid(format!(
                "Transition matrix must be square, got {}x{}",
                m,
                self.transition.ncols()
            ));
        }
        if self.observation.len() != m {
            return invalid(format!(
                "Observation vector length ({}) must match state dimension ({})",
                self.observation.len(),
                m
            ));
        }
        if self.selection.nrows() != m {
            return invalid(format!(
                "Selection matrix rows ({}) must match state dimension ({})",
                self.selection.nrows(),
                m
            ));
        }
        let r = self.selection.ncols();
        if self.state_cov.dim() != (r, r) {
            return invalid(format!(
                "State covariance dimensions ({:?}) must be ({}, {})",
                self.state_cov.dim(),
                r,
                r
            ));
        }
        for (name, all_finite) in [
            ("transition", self.transition.iter().all(|v| v.is_finite())),
            (
                "observation",
                self.observation.iter().all(|v| v.is_finite()),
            ),
            ("selection", self.selection.iter().all(|v| v.is_finite())),
            ("state_cov", self.state_cov.iter().all(|v| v.is_finite())),
        ] {
            if !all_finite {
                return invalid(format!("{name} contains a non-finite entry"));
            }
        }
        if !self.obs_var.is_finite() || self.obs_var < 0.0 {
            return invalid(format!(
                "obs_var must be finite and >= 0, got {}",
                self.obs_var
            ));
        }
        if let Some(&v) = self.state_cov.diag().iter().find(|v| **v < 0.0) {
            return invalid(format!("the diagonal of state_cov must be >= 0, got {v}"));
        }
        if let Some(c) = &self.state_intercept {
            if c.len() != m {
                return invalid(format!(
                    "state_intercept has length {} but the state dimension is {m}",
                    c.len()
                ));
            }
            if c.iter().any(|v| !v.is_finite()) {
                return invalid("state_intercept contains a non-finite entry".to_string());
            }
        }
        if let Some(d) = self.obs_intercept {
            if !d.is_finite() {
                return invalid(format!("obs_intercept must be finite, got {d}"));
            }
        }
        Ok(())
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
/// - Prediction: α_{t|t-1} = T α_{t-1|t-1} + c, P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'
/// - Innovation: v_t = y_t - Z' α_{t|t-1} - d, F_t = Z' P_{t|t-1} Z + H
/// - Update: α_{t|t} = α_{t|t-1} + M_t v_t / F_t, P_{t|t} = P_{t|t-1} - M_t M_t' / F_t,
///   with M_t = P_{t|t-1} Z
///
/// `init_state` is a_0 and `init_cov` P_0; the first step predicts a_1 = T a_0 + c,
/// as R's `KalmanRun(nit = -1)`.
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
/// [`EconError::InvalidSpecification`] for a model [`StateSpaceModel::validate`]
/// refuses or an infinite observation; [`EconError::Computation`] when an
/// innovation variance is not positive and finite (every variance zero, say);
/// [`EconError::Cancelled`] when the budget installed with
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

    model.validate()?;
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

/// The nonzero entries of a matrix, row by row.
fn matrix_nonzeros(a: ArrayView2<f64>, what: &'static str) -> EconResult<Vec<(usize, usize, f64)>> {
    let nnz = a.iter().filter(|v| **v != 0.0).count();
    let mut out = try_with_capacity(nnz, what)?;
    for ((i, j), &v) in a.indexed_iter() {
        if v != 0.0 {
            out.push((i, j, v));
        }
    }
    Ok(out)
}

/// The nonzero entries of a vector.
fn vector_nonzeros(a: ArrayView1<f64>, what: &'static str) -> EconResult<Vec<(usize, f64)>> {
    let nnz = a.iter().filter(|v| **v != 0.0).count();
    let mut out = try_with_capacity(nnz, what)?;
    for (i, &v) in a.indexed_iter() {
        if v != 0.0 {
            out.push((i, v));
        }
    }
    Ok(out)
}

/// A [`StateSpaceModel`] in the form the recursions use: the nonzeros of T, Z,
/// R Q R' and c, computed once per pass, so that no step allocates. Vectors are
/// slices of length m, matrices row-major slices of length m². A dense T costs
/// what a dense product does; a structural model's T X T' costs O(nnz(T) m).
struct StepModel {
    m: usize,
    t: Vec<(usize, usize, f64)>,
    z: Vec<(usize, f64)>,
    rqr: Vec<(usize, usize, f64)>,
    c: Vec<(usize, f64)>,
    d: f64,
    h: f64,
}

impl StepModel {
    /// The model must have passed [`StateSpaceModel::validate`].
    fn new(model: &StateSpaceModel) -> EconResult<Self> {
        const WHAT: &str = "the Kalman step model";
        let m = model.state_dim();
        let r = model.selection.ncols();
        // R Q R' from the nonzeros of R and Q.
        let mut rq = try_zeros2(m, r, WHAT)?;
        let selection = matrix_nonzeros(model.selection.view(), WHAT)?;
        for &(i, a, v) in &selection {
            for b in 0..r {
                rq[[i, b]] += v * model.state_cov[[a, b]];
            }
        }
        let mut rqr = try_zeros2(m, m, WHAT)?;
        for &(l, b, v) in &selection {
            for i in 0..m {
                rqr[[i, l]] += rq[[i, b]] * v;
            }
        }
        Ok(Self {
            m,
            t: matrix_nonzeros(model.transition.view(), WHAT)?,
            z: vector_nonzeros(model.observation.view(), WHAT)?,
            rqr: matrix_nonzeros(rqr.view(), WHAT)?,
            c: match &model.state_intercept {
                Some(c) => vector_nonzeros(c.view(), WHAT)?,
                None => Vec::new(),
            },
            d: model.obs_intercept.unwrap_or(0.0),
            h: model.obs_var,
        })
    }

    /// out = T x.
    #[inline]
    fn t_times(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for &(i, j, v) in &self.t {
            out[i] += v * x[j];
        }
    }

    /// out = T x + c: the state prediction.
    #[inline]
    fn predict_state(&self, a: &[f64], out: &mut [f64]) {
        self.t_times(a, out);
        for &(i, v) in &self.c {
            out[i] += v;
        }
    }

    /// out = T X T'. `scratch` holds m² values.
    #[inline]
    fn t_x_tt(&self, x: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        let m = self.m;
        scratch.fill(0.0);
        for &(i, j, v) in &self.t {
            let source = &x[j * m..(j + 1) * m];
            let target = &mut scratch[i * m..(i + 1) * m];
            for (t, s) in target.iter_mut().zip(source) {
                *t += v * s;
            }
        }
        out.fill(0.0);
        for (row_out, row_s) in out.chunks_exact_mut(m).zip(scratch.chunks_exact(m)) {
            for &(l, j, v) in &self.t {
                row_out[l] += row_s[j] * v;
            }
        }
    }

    /// out = T P T' + R Q R': the covariance prediction.
    #[inline]
    fn predict_cov(&self, p: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        self.t_x_tt(p, out, scratch);
        for &(i, j, v) in &self.rqr {
            out[i * self.m + j] += v;
        }
    }

    /// Z'x.
    #[inline]
    fn z_dot(&self, x: &[f64]) -> f64 {
        self.z.iter().map(|&(i, v)| v * x[i]).sum()
    }

    /// out = X Z.
    #[inline]
    fn x_times_z(&self, x: &[f64], out: &mut [f64]) {
        for (o, row) in out.iter_mut().zip(x.chunks_exact(self.m)) {
            *o = self.z.iter().map(|&(j, v)| row[j] * v).sum();
        }
    }

    /// out = T' x.
    #[inline]
    fn t_transpose_times(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for &(i, j, v) in &self.t {
            out[j] += v * x[i];
        }
    }

    /// The innovation variance of observation `t` (0-based), `Computation`
    /// unless positive and finite.
    #[inline]
    fn innovation_var(&self, m_vec: &[f64], t: usize) -> EconResult<f64> {
        let f = self.z_dot(m_vec) + self.h;
        if f > 0.0 && f.is_finite() {
            Ok(f)
        } else {
            Err(innovation_var_error(t, f))
        }
    }

    /// a = a⁻ + M v / F, P = P⁻ - M M' / F (symmetric: the upper triangle is
    /// computed and copied), both from the predictions. R's `KalmanLike` update
    /// form: O(m²) against the Joseph form's O(m³), and the form the
    /// differentiated filter differentiates.
    #[inline]
    fn update(
        &self,
        an: &[f64],
        pn: &[f64],
        m_vec: &[f64],
        v: f64,
        f: f64,
        a: &mut [f64],
        p: &mut [f64],
    ) {
        let m = self.m;
        let gain = v / f;
        for ((o, x), g) in a.iter_mut().zip(an).zip(m_vec) {
            *o = x + g * gain;
        }
        for i in 0..m {
            let mi = m_vec[i] / f;
            for l in i..m {
                let value = pn[i * m + l] - mi * m_vec[l];
                p[i * m + l] = value;
                p[l * m + i] = value;
            }
        }
    }
}

#[cold]
fn innovation_var_error(t: usize, f: f64) -> EconError {
    EconError::Computation(format!(
        "Kalman filter: the innovation variance at observation {} is {f}; it must be positive \
         and finite (are all variances zero?)",
        t + 1
    ))
}

#[cold]
fn infinite_observation_error(t: usize) -> EconError {
    EconError::InvalidSpecification {
        message: format!("y[{t}] is infinite; use NaN for a missing value"),
    }
}

/// The Kalman filter recursions over `y`, returning the log-likelihood and,
/// when `history` is given, pushing every step's states, covariances,
/// innovation and innovation variance onto it. The model must have passed
/// [`StateSpaceModel::validate`] and the initial values
/// [`check_initial_values`].
fn filter_pass(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
    mut history: Option<&mut FilterHistory>,
) -> EconResult<f64> {
    const WHAT: &str = "the Kalman filter's workspace";
    let n = y.len();
    let m = model.state_dim();
    let step = StepModel::new(model)?;
    let mm = checked_len(&[m, m], WHAT)?;

    // One workspace per pass; no step allocates except the history it keeps.
    let mut a: Vec<f64> = init_state.iter().copied().collect();
    let mut p: Vec<f64> = init_cov.iter().copied().collect();
    let mut an = try_vec(m, 0.0, WHAT)?;
    let mut pn = try_vec(mm, 0.0, WHAT)?;
    let mut m_vec = try_vec(m, 0.0, WHAT)?;
    let mut scratch = try_vec(mm, 0.0, WHAT)?;

    let mut log_lik = 0.0;
    let mut n_valid = 0usize;

    // Each step costs O(m³) at most: check the budget about every 65,536 operations.
    let stride = checkpoint_stride(m.saturating_mul(m).saturating_mul(m));
    for t in 0..n {
        if t % stride == 0 {
            checkpoint()?;
        }
        step.predict_state(&a, &mut an);
        step.predict_cov(&p, &mut pn, &mut scratch);

        if let Some(h) = history.as_deref_mut() {
            h.predicted_states.push(an.clone());
            h.predicted_covs.push(flat_to_nested(&pn, m));
        }

        if y[t].is_nan() {
            // Missing observation: no update
            a.copy_from_slice(&an);
            p.copy_from_slice(&pn);
            if let Some(h) = history.as_deref_mut() {
                h.innovations.push(f64::NAN);
                h.innovation_vars.push(f64::NAN);
            }
        } else {
            if y[t].is_infinite() {
                return Err(infinite_observation_error(t));
            }
            let v = y[t] - step.z_dot(&an) - step.d;
            step.x_times_z(&pn, &mut m_vec);
            let f = step.innovation_var(&m_vec, t)?;

            if let Some(h) = history.as_deref_mut() {
                h.innovations.push(v);
                h.innovation_vars.push(f);
            }

            step.update(&an, &pn, &m_vec, v, f, &mut a, &mut p);

            log_lik -= 0.5 * (f.ln() + v * v / f);
            n_valid += 1;
        }

        if let Some(h) = history.as_deref_mut() {
            h.filtered_states.push(a.clone());
            h.filtered_covs.push(flat_to_nested(&p, m));
        }
    }

    // Add constant term to log-likelihood
    log_lik -= 0.5 * n_valid as f64 * (2.0 * std::f64::consts::PI).ln();

    Ok(log_lik)
}

/// Derivative of a model with respect to one parameter: d(R Q R') (m × m) and
/// dH. c, d, T, Z, a_0 and P_0 are constant in the parameter.
pub(crate) struct ModelDerivative {
    pub(crate) state_dist_cov: Array2<f64>,
    pub(crate) obs_var: f64,
}

/// Sums over the non-missing observations of the predict-first filter, and
/// their derivatives.
#[derive(Debug)]
pub(crate) struct LikelihoodTerms {
    /// Σ v_t² / F_t
    pub(crate) ssq: f64,
    /// Σ ln F_t
    pub(crate) sumlog: f64,
    /// Non-missing observations.
    pub(crate) n_used: usize,
    /// ∂ssq/∂θ_j
    pub(crate) d_ssq: Vec<f64>,
    /// ∂sumlog/∂θ_j
    pub(crate) d_sumlog: Vec<f64>,
}

/// The likelihood terms of `model` and their exact gradient: the Kalman
/// recursions differentiated (Harvey 1990, sec. 3.4.6), one derivative per
/// entry of `derivatives`. It keeps no history, and allocates one workspace
/// per call and nothing per step; a step costs O(nnz(T) m) per derivative. The
/// budget is checked about every 65,536 operations.
pub(crate) fn kalman_likelihood_terms(
    y: &[f64],
    model: &StateSpaceModel,
    init_state: ArrayView1<f64>,
    init_cov: ArrayView2<f64>,
    derivatives: &[ModelDerivative],
) -> EconResult<LikelihoodTerms> {
    const WHAT: &str = "the Kalman likelihood's workspace";
    let m = model.state_dim();
    model.validate()?;
    check_initial_values(m, init_state, init_cov)?;
    let k = derivatives.len();
    let mut d_cov = try_with_capacity(k, WHAT)?;
    for d in derivatives {
        if d.state_dist_cov.dim() != (m, m) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "a model derivative's state_dist_cov has dimensions {:?}, not ({m}, {m})",
                    d.state_dist_cov.dim()
                ),
            });
        }
        d_cov.push(matrix_nonzeros(d.state_dist_cov.view(), WHAT)?);
    }
    let step = StepModel::new(model)?;
    let mm = checked_len(&[m, m], WHAT)?;

    let mut a: Vec<f64> = init_state.iter().copied().collect();
    let mut p: Vec<f64> = init_cov.iter().copied().collect();
    let mut an = try_vec(m, 0.0, WHAT)?;
    let mut pn = try_vec(mm, 0.0, WHAT)?;
    let mut m_vec = try_vec(m, 0.0, WHAT)?;
    let mut dm = try_vec(m, 0.0, WHAT)?;
    let mut scratch = try_vec(mm, 0.0, WHAT)?;
    // Per derivative: da, da⁻ (m each) and dP, dP⁻ (m² each), all starting at 0
    // (a_0 and P_0 do not depend on the parameter).
    let mut da = try_vec(checked_len(&[k, m], WHAT)?, 0.0, WHAT)?;
    let mut dan = try_vec(checked_len(&[k, m], WHAT)?, 0.0, WHAT)?;
    let mut dp = try_vec(checked_len(&[k, mm], WHAT)?, 0.0, WHAT)?;
    let mut dpn = try_vec(checked_len(&[k, mm], WHAT)?, 0.0, WHAT)?;

    let mut out = LikelihoodTerms {
        ssq: 0.0,
        sumlog: 0.0,
        n_used: 0,
        d_ssq: try_vec(k, 0.0, WHAT)?,
        d_sumlog: try_vec(k, 0.0, WHAT)?,
    };

    let per_step = m
        .saturating_mul(m)
        .saturating_mul(m)
        .saturating_mul(k.saturating_add(1));
    let stride = checkpoint_stride(per_step);
    for (t, &yt) in y.iter().enumerate() {
        if t % stride == 0 {
            checkpoint()?;
        }
        step.predict_state(&a, &mut an);
        step.predict_cov(&p, &mut pn, &mut scratch);
        for j in 0..k {
            step.t_times(&da[j * m..(j + 1) * m], &mut dan[j * m..(j + 1) * m]);
            let dpn_j = &mut dpn[j * mm..(j + 1) * mm];
            step.t_x_tt(&dp[j * mm..(j + 1) * mm], dpn_j, &mut scratch);
            for &(i, l, v) in &d_cov[j] {
                dpn_j[i * m + l] += v;
            }
        }

        if yt.is_nan() {
            a.copy_from_slice(&an);
            p.copy_from_slice(&pn);
            da.copy_from_slice(&dan);
            dp.copy_from_slice(&dpn);
            continue;
        }
        if yt.is_infinite() {
            return Err(infinite_observation_error(t));
        }

        let v = yt - step.z_dot(&an) - step.d;
        step.x_times_z(&pn, &mut m_vec);
        let f = step.innovation_var(&m_vec, t)?;
        out.ssq += v * v / f;
        out.sumlog += f.ln();
        out.n_used += 1;

        for j in 0..k {
            let dan_j = &dan[j * m..(j + 1) * m];
            let dpn_j = &dpn[j * mm..(j + 1) * mm];
            let dv = -step.z_dot(dan_j);
            step.x_times_z(dpn_j, &mut dm);
            let df = step.z_dot(&dm) + derivatives[j].obs_var;
            out.d_ssq[j] += 2.0 * v * dv / f - v * v * df / (f * f);
            out.d_sumlog[j] += df / f;

            let (c_v, c_dv, c_df) = (v / f, dv / f, v * df / (f * f));
            for (i, o) in da[j * m..(j + 1) * m].iter_mut().enumerate() {
                *o = dan_j[i] + dm[i] * c_v + m_vec[i] * (c_dv - c_df);
            }
            let dp_j = &mut dp[j * mm..(j + 1) * mm];
            let f2 = df / (f * f);
            for i in 0..m {
                for l in i..m {
                    let value = dpn_j[i * m + l] - (dm[i] * m_vec[l] + m_vec[i] * dm[l]) / f
                        + m_vec[i] * m_vec[l] * f2;
                    dp_j[i * m + l] = value;
                    dp_j[l * m + i] = value;
                }
            }
        }
        step.update(&an, &pn, &m_vec, v, f, &mut a, &mut p);
    }
    Ok(out)
}

/// Run the Kalman smoother (backward pass).
///
/// Computes smoothed state estimates using all observations, by the
/// Durbin-Koopman state smoother (Durbin & Koopman 2012, eq. 4.44): backward from
/// r = 0, N = 0, with L = T - K Z' and K = T P Z / F,
/// r ← Z v / F + L' r and N ← Z Z' / F + L' N L at an observation (T' r and
/// T' N T at a missing one), then a + P r and P - P N P. It inverts nothing, so a
/// singular predicted covariance needs no special treatment.
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

    model.validate()?;
    check_filter_result(filter_result, model)?;

    // An empty filter result has no final state to seed the backward pass.
    // Reject it cleanly.
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

    const WHAT: &str = "the Kalman smoother's workspace";
    let step = StepModel::new(model)?;
    let mm = checked_len(&[m, m], WHAT)?;
    let mut r = try_vec(m, 0.0, WHAT)?;
    let mut r_next = try_vec(m, 0.0, WHAT)?;
    let mut n_mat = try_vec(mm, 0.0, WHAT)?;
    let mut n_next = try_vec(mm, 0.0, WHAT)?;
    let mut p_t = try_vec(mm, 0.0, WHAT)?;
    let mut m_vec = try_vec(m, 0.0, WHAT)?;
    let mut k_vec = try_vec(m, 0.0, WHAT)?;
    let mut n_k = try_vec(m, 0.0, WHAT)?;
    let mut k_w = try_vec(m, 0.0, WHAT)?;
    let mut w = try_vec(mm, 0.0, WHAT)?;
    let mut scratch = try_vec(mm, 0.0, WHAT)?;

    // Each step costs O(m³) (P N P); nothing here allocates except what it keeps.
    let stride = checkpoint_stride(m.saturating_mul(m).saturating_mul(m));
    for (done, t) in (0..n).rev().enumerate() {
        if done % stride == 0 {
            checkpoint()?;
        }
        for (i, row) in filter_result.predicted_covs[t].iter().enumerate() {
            p_t[i * m..(i + 1) * m].copy_from_slice(row);
        }
        let v = filter_result.innovations[t];
        let f = filter_result.innovation_vars[t];

        // r_{t-1} and N_{t-1} from r_t and N_t (held in r_next, n_next).
        std::mem::swap(&mut r, &mut r_next);
        std::mem::swap(&mut n_mat, &mut n_next);
        // W = N T
        w.fill(0.0);
        for (row_w, row_n) in w.chunks_exact_mut(m).zip(n_next.chunks_exact(m)) {
            for &(l, j, value) in &step.t {
                row_w[j] += row_n[l] * value;
            }
        }
        step.t_transpose_times(&r_next, &mut r);
        if f.is_finite() && v.is_finite() {
            step.x_times_z(&p_t, &mut m_vec);
            step.t_times(&m_vec, &mut k_vec);
            for x in k_vec.iter_mut() {
                *x /= f;
            }
            // r ← Z v / F + T' r - Z (K' r)
            let k_r: f64 = k_vec.iter().zip(&r_next).map(|(a, b)| a * b).sum();
            for &(i, z) in &step.z {
                r[i] += z * (v / f - k_r);
            }
            // N L = W - (N K) Z'
            for (o, row) in n_k.iter_mut().zip(n_next.chunks_exact(m)) {
                *o = row.iter().zip(&k_vec).map(|(a, b)| a * b).sum();
            }
            for (row_w, nk) in w.chunks_exact_mut(m).zip(&n_k) {
                for &(j, z) in &step.z {
                    row_w[j] -= nk * z;
                }
            }
            // k_w = K' (N L)
            k_w.fill(0.0);
            for (row_w, kv) in w.chunks_exact(m).zip(&k_vec) {
                for (o, x) in k_w.iter_mut().zip(row_w) {
                    *o += kv * x;
                }
            }
        }
        // N ← T' (N L) ...
        n_mat.fill(0.0);
        for &(i, j, value) in &step.t {
            let source = &w[i * m..(i + 1) * m];
            let target = &mut n_mat[j * m..(j + 1) * m];
            for (o, x) in target.iter_mut().zip(source) {
                *o += value * x;
            }
        }
        if f.is_finite() && v.is_finite() {
            // ... - Z K' (N L) + Z Z' / F
            for &(i, z) in &step.z {
                for (o, x) in n_mat[i * m..(i + 1) * m].iter_mut().zip(&k_w) {
                    *o -= z * x;
                }
                for &(j, z2) in &step.z {
                    n_mat[i * m + j] += z * z2 / f;
                }
            }
        }
        // N is symmetric: copy the upper triangle, as the filter does for P.
        for i in 0..m {
            for l in (i + 1)..m {
                n_mat[l * m + i] = n_mat[i * m + l];
            }
        }

        // α_{t|n} = a_t + P_t r_{t-1}
        let a_t = &filter_result.predicted_states[t];
        let mut state = try_with_capacity(m, &what)?;
        for (row, a) in p_t.chunks_exact(m).zip(a_t) {
            let pr: f64 = row.iter().zip(&r).map(|(p, x)| p * x).sum();
            state.push(a + pr);
        }
        smoothed_states.push(state);

        // P_{t|n} = P_t - P_t N_{t-1} P_t; P N P is symmetric, so the upper
        // triangle is computed and copied.
        for (row_s, row_p) in scratch.chunks_exact_mut(m).zip(p_t.chunks_exact(m)) {
            row_s.fill(0.0);
            for (pv, row_n) in row_p.iter().zip(n_mat.chunks_exact(m)) {
                for (o, x) in row_s.iter_mut().zip(row_n) {
                    *o += pv * x;
                }
            }
        }
        let mut cov: Vec<Vec<f64>> = try_with_capacity(m, &what)?;
        for i in 0..m {
            let row_s = &scratch[i * m..(i + 1) * m];
            let mut row: Vec<f64> = try_with_capacity(m, &what)?;
            for l in 0..m {
                if l < i {
                    // The mirror of an entry already computed in row l.
                    row.push(cov[l][i]);
                } else {
                    let psp: f64 = row_s
                        .iter()
                        .zip(p_t.chunks_exact(m).map(|c| c[l]))
                        .map(|(a, b)| a * b)
                        .sum();
                    row.push(p_t[i * m + l] - psp);
                }
            }
            cov.push(row);
        }
        smoothed_covs.push(cov);
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

/// `InvalidSpecification` unless a filter result's dimensions are those of
/// `model` and of its `n_obs` and `state_dim`: the fields are public.
fn check_filter_result(
    filter_result: &KalmanFilterResult,
    model: &StateSpaceModel,
) -> EconResult<()> {
    let (n, m) = (filter_result.n_obs, filter_result.state_dim);
    if model.state_dim() != m {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the model's state dimension is {} but the filter result's is {m}",
                model.state_dim()
            ),
        });
    }
    let bad = filter_result.predicted_states.len() != n
        || filter_result.predicted_covs.len() != n
        || filter_result.filtered_states.len() != n
        || filter_result.filtered_covs.len() != n
        || filter_result.innovations.len() != n
        || filter_result.innovation_vars.len() != n
        || filter_result.predicted_states.iter().any(|s| s.len() != m)
        || filter_result.filtered_states.iter().any(|s| s.len() != m)
        || filter_result
            .predicted_covs
            .iter()
            .chain(&filter_result.filtered_covs)
            .any(|c| c.len() != m || c.iter().any(|row| row.len() != m));
    if bad {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the filter result's per-step vectors must hold {n} steps of a \
                 {m}-dimensional state"
            ),
        });
    }
    Ok(())
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

    model.validate()?;
    check_filter_result(filter_result, model)?;

    // Admit what the forecast keeps, counted by the bytes its nested Vecs
    // hold, before any work.
    admit_bytes(&[history_bytes(horizon, m, FORECAST_STEP, WHAT)?], WHAT)?;

    // Forecasting starts from the final filtered state; an empty filter result
    // has none.
    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Kalman forecast requires a non-empty filter result".to_string(),
        });
    }

    // Start from final filtered state
    let step = StepModel::new(model)?;
    let mm = checked_len(&[m, m], WHAT)?;
    let mut state = filter_result.filtered_states[n - 1].clone();
    let mut cov = try_vec(mm, 0.0, WHAT)?;
    for (i, row) in filter_result.filtered_covs[n - 1].iter().enumerate() {
        cov[i * m..(i + 1) * m].copy_from_slice(row);
    }
    let mut state_next = try_vec(m, 0.0, WHAT)?;
    let mut cov_next = try_vec(mm, 0.0, WHAT)?;
    let mut scratch = try_vec(mm, 0.0, WHAT)?;
    let mut m_vec = try_vec(m, 0.0, WHAT)?;

    let mut forecasts = try_with_capacity(horizon, WHAT)?;
    let mut variances = try_with_capacity(horizon, WHAT)?;
    let mut std_errors = try_with_capacity(horizon, WHAT)?;
    let mut state_forecasts: Vec<Vec<f64>> = try_with_capacity(horizon, WHAT)?;

    for _ in 0..horizon {
        // Each step costs O(m³).
        checkpoint()?;
        // Predict state and covariance
        step.predict_state(&state, &mut state_next);
        step.predict_cov(&cov, &mut cov_next, &mut scratch);
        std::mem::swap(&mut state, &mut state_next);
        std::mem::swap(&mut cov, &mut cov_next);

        // Predict observation
        let forecast = step.z_dot(&state) + step.d;
        step.x_times_z(&cov, &mut m_vec);
        let variance = step.z_dot(&m_vec) + step.h;

        forecasts.push(forecast);
        variances.push(variance);
        std_errors.push(variance.sqrt());
        state_forecasts.push(state.clone());
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
    model.validate()?;
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

/// A row-major m × m matrix as the nested rows the results keep.
fn flat_to_nested(x: &[f64], m: usize) -> Vec<Vec<f64>> {
    x.chunks_exact(m).map(<[f64]>::to_vec).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix_ops::safe_inverse;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    fn vec_to_array1(v: &[f64]) -> Array1<f64> {
        Array1::from_vec(v.to_vec())
    }

    fn nested_vec_to_array2(v: &[Vec<f64>]) -> Array2<f64> {
        let rows = v.len();
        let cols = if rows > 0 { v[0].len() } else { 0 };
        Array2::from_shape_fn((rows, cols), |(i, j)| v[i][j])
    }

    fn array2_to_nested_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
        arr.outer_iter().map(|row| row.to_vec()).collect()
    }

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
    fn test_kalman_local_level_tracks_signal_recovers_dgp() {
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
    fn test_kalman_smoother_variance_not_above_filter() {
        // The smoother conditions on more data than the filter, so at every t
        // the smoothed variance is at most the filtered one.
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

        for t in 0..n {
            let filter_var = filter_result.filtered_covs[t][0][0];
            let smooth_var = smooth_result.smoothed_covs[t][0][0];
            assert!(
                smooth_var <= filter_var * (1.0 + 1e-12) + 1e-15,
                "t = {t}: smoothed {smooth_var} above filtered {filter_var}"
            );
        }
    }

    /// Called as `intercept_case("kalman_intercepts")`: the case id is a literal in each test.
    fn intercept_case(id: &str) -> (p2a_validation::Case, Vec<f64>) {
        let (case, ds) = crate::test_support::load_case(id);
        let (y, _) = crate::data::columns::numeric_column(
            ds.df(),
            "y",
            crate::data::columns::NaPolicy::Fail,
        )
        .unwrap();
        assert_eq!(y.len(), 50);
        (case, y)
    }

    fn llt(q_slope: f64) -> StateSpaceModel {
        StateSpaceModel::new(
            array![[1.0, 1.0], [0.0, 1.0]],
            array![1.0, 0.0],
            Array2::eye(2),
            array![[0.3, 0.0], [0.0, q_slope]],
            0.5,
        )
        .unwrap()
    }

    /// Sets the public fields directly.
    fn with_cd(mut m: StateSpaceModel) -> StateSpaceModel {
        m.state_intercept = Some(array![0.1, 0.0]);
        m.obs_intercept = Some(2.0);
        m
    }

    fn col(states: &[Vec<f64>], j: usize) -> Vec<f64> {
        states.iter().map(|s| s[j]).collect()
    }

    fn filter_llt(y: &[f64], model: &StateSpaceModel) -> KalmanFilterResult {
        let a0 = Array1::zeros(2);
        let p0 = Array2::eye(2) * 10.0;
        kalman_filter(y, model, a0.view(), p0.view()).unwrap()
    }

    #[test]
    fn test_validate_kalman_intercepts() {
        let (case, y) = intercept_case("kalman_intercepts");
        let model = with_cd(llt(0.01));
        let f = filter_llt(&y, &model);
        let fc = kalman_forecast(&f, &model, 5).unwrap();
        Check::new(&case)
            .scalar("loglik", f.log_likelihood, Tol::stat(50))
            .vector("filtered_level", &col(&f.filtered_states, 0), Tol::coef(50))
            .vector("filtered_slope", &col(&f.filtered_states, 1), Tol::coef(50))
            .vector("innovations", &f.innovations, Tol::coef(50))
            .vector("innovation_var", &f.innovation_vars, Tol::coef(50))
            .vector("forecast", &fc.forecasts, Tol::coef(50))
            .vector("forecast_var", &fc.variances, Tol::coef(50))
            .finish();
    }

    #[test]
    fn test_validate_kalman_smoother_intercepts() {
        let (case, y) = intercept_case("kalman_intercepts");
        let model = with_cd(llt(0.01));
        let s = kalman_smoother(&filter_llt(&y, &model), &model).unwrap();
        let level_var: Vec<f64> = s.smoothed_covs.iter().map(|c| c[0][0]).collect();
        Check::new(&case)
            .vector("smoothed_level", &col(&s.smoothed_states, 0), Tol::coef(50))
            .vector("smoothed_slope", &col(&s.smoothed_states, 1), Tol::coef(50))
            .vector("smoothed_level_var", &level_var, Tol::coef(50))
            .finish();
    }

    #[test]
    fn test_validate_kalman_intercepts_change_the_fit() {
        // Without intercepts the filter pins R's value of the plain model (the
        // anchor); with them the fit must differ.
        let (case, y) = intercept_case("kalman_intercepts");
        let without = filter_llt(&y, &llt(0.01)).log_likelihood;
        Check::new(&case)
            .scalar("loglik_no_intercept", without, Tol::stat(50))
            .finish();
        let with = filter_llt(&y, &with_cd(llt(0.01))).log_likelihood;
        assert!((with - without).abs() > 0.1, "{with} vs {without}");
    }

    #[test]
    fn test_validate_kalman_smoother_singular_covariance() {
        // A deterministic slope (no slope variance, no prior slope variance)
        // leaves every P_{t+1|t} singular; the state smoother inverts none.
        let (case, y) = intercept_case("kalman_intercepts");
        let model = llt(0.0);
        let (a0, p0) = (array![0.0, -0.1], array![[10.0, 0.0], [0.0, 0.0]]);
        let f = kalman_filter(&y, &model, a0.view(), p0.view()).unwrap();
        let s = kalman_smoother(&f, &model).unwrap();
        Check::new(&case)
            .vector(
                "smoothed_level_fixed_slope",
                &col(&s.smoothed_states, 0),
                Tol::coef(50),
            )
            .finish();
        for state in &s.smoothed_states {
            assert!((state[1] + 0.1).abs() < 1e-12, "{}", state[1]);
        }
    }

    #[test]
    fn test_kalman_rejects_bad_intercepts() {
        let mut bad = llt(0.01);
        bad.state_intercept = Some(array![0.1, 0.0, 0.0]);
        let y = vec![1.0, 2.1, 2.9];
        let (a0, p0) = (Array1::zeros(2), Array2::eye(2) * 10.0);
        let invalid = |r: EconResult<()>, needle: &str| match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        invalid(
            kalman_filter(&y, &bad, a0.view(), p0.view()).map(|_| ()),
            "state_intercept",
        );
        let good = llt(0.01);
        let f = kalman_filter(&y, &good, a0.view(), p0.view()).unwrap();
        invalid(kalman_forecast(&f, &bad, 2).map(|_| ()), "state_intercept");
        invalid(kalman_smoother(&f, &bad).map(|_| ()), "state_intercept");
    }

    #[test]
    fn test_kalman_with_intercepts_checks_them() {
        // A model with no state has nothing to filter.
        let empty = StateSpaceModel::new(
            Array2::zeros((0, 0)),
            Array1::zeros(0),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            1.0,
        )
        .unwrap();
        assert!(matches!(
            empty.validate(),
            Err(EconError::InvalidSpecification { .. })
        ));
        match llt(0.01).with_intercepts(Some(array![0.1, 0.0, 0.0]), None) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("state_intercept") && message.contains('3'),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        match llt(0.01).with_intercepts(None, Some(f64::NAN)) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("obs_intercept"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        let set = llt(0.01)
            .with_intercepts(Some(array![0.1, 0.0]), Some(2.0))
            .unwrap();
        assert_eq!(set.obs_intercept, Some(2.0));
    }

    #[test]
    fn test_validate_kalman_likelihood_terms_match_filter_and_differences() {
        let (case, y) = intercept_case("kalman_intercepts");
        let (a0, p0) = (Array1::zeros(2), Array2::eye(2) * 10.0);
        let e = |i: usize| {
            let mut d = Array2::<f64>::zeros((2, 2));
            d[[i, i]] = 1.0;
            d
        };
        let derivatives = [
            ModelDerivative {
                state_dist_cov: e(0),
                obs_var: 0.0,
            },
            ModelDerivative {
                state_dist_cov: e(1),
                obs_var: 0.0,
            },
            ModelDerivative {
                state_dist_cov: Array2::zeros((2, 2)),
                obs_var: 1.0,
            },
        ];
        let model = |q0: f64, q1: f64, h: f64| {
            let mut m = with_cd(llt(q1));
            m.state_cov[[0, 0]] = q0;
            m.obs_var = h;
            m
        };
        let terms = |q0, q1, h| {
            kalman_likelihood_terms(&y, &model(q0, q1, h), a0.view(), p0.view(), &derivatives)
                .unwrap()
        };
        let at = [0.3, 0.01, 0.5];
        let t = terms(at[0], at[1], at[2]);
        assert_eq!(t.n_used, 50);
        let two_pi = (2.0 * std::f64::consts::PI).ln();
        let ll = -0.5 * (t.ssq + t.sumlog) - 0.5 * t.n_used as f64 * two_pi;
        // The same recursion as the filter, and R's value.
        let filter = filter_llt(&y, &model(at[0], at[1], at[2])).log_likelihood;
        assert!(
            (ll - filter).abs() <= 1e-12 * filter.abs(),
            "{ll} vs {filter}"
        );
        Check::new(&case)
            .scalar("loglik", ll, Tol::stat(50))
            .finish();
        // Each gradient component is the central difference of ssq + sumlog.
        for j in 0..3 {
            let step = 1e-6 * at[j];
            let (mut up, mut down) = (at, at);
            up[j] += step;
            down[j] -= step;
            let f = |x: [f64; 3]| {
                let t = terms(x[0], x[1], x[2]);
                t.ssq + t.sumlog
            };
            let numeric = (f(up) - f(down)) / (2.0 * step);
            let exact = t.d_ssq[j] + t.d_sumlog[j];
            assert!(
                (exact - numeric).abs() <= 1e-6 * exact.abs(),
                "{j}: {exact} vs {numeric}"
            );
        }
    }

    #[test]
    fn test_kalman_likelihood_terms_skip_missing_values_and_check_their_input() {
        // A missing observation carries the derivative through the prediction.
        let y = [1.0, 2.2, f64::NAN, f64::NAN, 3.1, 2.8, 3.9, 4.2];
        let (a0, p0) = (Array1::zeros(2), Array2::eye(2) * 10.0);
        let derivative = |h: f64| {
            let mut d = Array2::<f64>::zeros((2, 2));
            d[[0, 0]] = 1.0 - h;
            ModelDerivative {
                state_dist_cov: d,
                obs_var: h,
            }
        };
        let model = |q0: f64, h: f64| {
            let mut m = llt(0.01);
            m.state_cov[[0, 0]] = q0;
            m.obs_var = h;
            m
        };
        let terms = |q0, h, d: &[ModelDerivative]| {
            kalman_likelihood_terms(&y, &model(q0, h), a0.view(), p0.view(), d).unwrap()
        };
        let t = terms(0.3, 0.5, &[derivative(0.0), derivative(1.0)]);
        assert_eq!(t.n_used, 6);
        let f = |q0, h| {
            let t = terms(q0, h, &[]);
            t.ssq + t.sumlog
        };
        let numeric_q = (f(0.3 + 1e-6, 0.5) - f(0.3 - 1e-6, 0.5)) / 2e-6;
        let numeric_h = (f(0.3, 0.5 + 1e-6) - f(0.3, 0.5 - 1e-6)) / 2e-6;
        let exact_q = t.d_ssq[0] + t.d_sumlog[0];
        let exact_h = t.d_ssq[1] + t.d_sumlog[1];
        assert!((exact_q - numeric_q).abs() <= 1e-6 * exact_q.abs());
        assert!((exact_h - numeric_h).abs() <= 1e-6 * exact_h.abs());

        let wrong = ModelDerivative {
            state_dist_cov: Array2::zeros((3, 3)),
            obs_var: 0.0,
        };
        assert!(matches!(
            kalman_likelihood_terms(&y, &model(0.3, 0.5), a0.view(), p0.view(), &[wrong]),
            Err(EconError::InvalidSpecification { .. })
        ));
        assert!(matches!(
            cancelled(|| kalman_likelihood_terms(
                &y,
                &model(0.3, 0.5),
                a0.view(),
                p0.view(),
                &[derivative(0.0)]
            )),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn test_kalman_filter_rejects_nonpositive_innovation_variance() {
        let model = StateSpaceModel::new(
            array![[1.0]],
            array![1.0],
            array![[1.0]],
            array![[0.0]],
            0.0,
        )
        .unwrap();
        let (a0, p0) = (array![1.0], array![[0.0]]);
        match kalman_filter(&[1.0, 2.0, 3.0], &model, a0.view(), p0.view()) {
            Err(EconError::Computation(msg)) => assert!(
                msg.contains("innovation variance") && msg.contains("observation 1"),
                "{msg}"
            ),
            other => panic!("expected Computation, got {other:?}"),
        }
        let noisy = StateSpaceModel::new(
            array![[1.0]],
            array![1.0],
            array![[1.0]],
            array![[0.1]],
            0.2,
        )
        .unwrap();
        match kalman_filter(&[1.0, f64::INFINITY], &noisy, a0.view(), p0.view()) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("infinite"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_kalman_filter_scale_invariant() {
        // The filter has no absolute threshold: data on the scale 1e-9, with
        // every variance rescaled, is the same model. The reference's data keep
        // the check honest about what is being rescaled.
        let (case, y) = intercept_case("kalman_intercepts");
        let plain = filter_llt(&y, &llt(0.01));
        Check::new(&case)
            .scalar("loglik_no_intercept", plain.log_likelihood, Tol::stat(50))
            .finish();
        let c = 1e-9_f64;
        let scaled_y: Vec<f64> = y.iter().map(|v| v * c).collect();
        let c2 = c * c;
        let scaled = StateSpaceModel::new(
            array![[1.0, 1.0], [0.0, 1.0]],
            array![1.0, 0.0],
            Array2::eye(2),
            array![[0.3 * c2, 0.0], [0.0, 0.01 * c2]],
            0.5 * c2,
        )
        .unwrap();
        let (a0, p0) = (Array1::zeros(2), Array2::eye(2) * (10.0 * c2));
        let f = kalman_filter(&scaled_y, &scaled, a0.view(), p0.view()).unwrap();
        let want = plain.log_likelihood - 50.0 * c.ln();
        assert!(
            Tol::stat(50).allows(f.log_likelihood, want),
            "{} vs {want}",
            f.log_likelihood
        );
        for (a, b) in f.filtered_states.iter().zip(&plain.filtered_states) {
            assert!(
                Tol::coef(50).allows(a[0] / c, b[0]),
                "{} vs {}",
                a[0] / c,
                b[0]
            );
        }
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
