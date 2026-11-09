//! Structural Time Series models (StructTS).
//!
//! Fits structural time series models by maximum likelihood using the Kalman filter.
//! Supports local level, local linear trend, and basic structural model (BSM).
//!
//! The fit is that of R's `stats::StructTS`:
//! - the prior a_0 = (y_1, 0, ...) and P_0 = 1e6 · vx in **every** entry, with
//!   vx = var(y)/100 (R: `Z$P[] <- 1e+06 * vx`), which has rank one;
//! - the objective 0.5 (Σ v²/F + Σ ln F) / n_obs with every observation in it,
//!   over the variances in units of vx, starting at 1;
//! - `fixed` with R's length and order.
//!
//! The optimiser is not R's: a projected Newton on p ≥ 0 with the exact gradient
//! (the differentiated Kalman filter), so a variance reaches 0 exactly, and it
//! stops at the optimum of R's objective, where R's L-BFGS-B stops short of it.
//! With missing values the log-likelihood is the observed-data one, not R's value,
//! which scales the data term by length(y)/n_obs; the estimates are the same.
//! `validation/forecasting/structts.md` gives the pinned values.
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
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::forecasting::kalman::{
    ModelDerivative, StateSpaceModel, admit_filter_and_smoother_history, kalman_filter,
    kalman_likelihood_terms, kalman_smoother,
};
use crate::linalg::matrix_ops::{cholesky, eig_symmetric};
use crate::options::Choices;
use crate::resource::{checkpoint, overflow, try_zeros1, try_zeros2, with_remedy};
use crate::traits::convergence::{Convergence, ResultNotes};
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

// The accepted spellings (StructTS(type = )); the first of each entry is canonical.
crate::option_enum!(
    StructTsType,
    Choices {
        option: "model_type",
        values: &[
            (&["level", "local_level"], StructTsType::Level),
            (
                &["trend", "local_trend", "llt", "local_linear_trend"],
                StructTsType::Trend,
            ),
            (
                &["BSM", "basic_structural", "basic_structural_model"],
                StructTsType::BSM,
            ),
        ],
    },
    no_display
);

/// Configuration for structural time series fitting.
#[derive(Debug, Clone)]
pub struct StructTsConfig {
    /// Model type.
    pub model_type: StructTsType,
    /// Seasonal period, ignored unless BSM (12 when unset): at least 2, and at
    /// most half the series, since StructTS needs two full seasons.
    pub period: Option<usize>,
    /// Fixed variances, in R's order: (level, epsilon), (level, slope,
    /// epsilon) or (level, slope, seas, epsilon). Absolute variances, finite
    /// and at least 0; `None` means estimate. At least one must be `None`.
    pub fixed: Option<Vec<Option<f64>>>,
    /// Maximum projected Newton steps (at least 1; R's maxit default is 100).
    pub max_iter: usize,
    /// Tolerance on the scaled projected gradient (finite and positive).
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
    /// Log-likelihood at optimum: the Gaussian log-likelihood of the observed
    /// values, R's `loglik` when none is missing.
    pub log_likelihood: f64,
    /// AIC, from the estimated (free) variances.
    pub aic: f64,
    /// BIC, from the estimated (free) variances and the observed values.
    pub bic: f64,
    /// Number of observations (the length of the series).
    pub n_obs: usize,
    /// Number of estimated (free) variances.
    pub n_params: usize,
    /// Convergence status, `convergence.converged`.
    pub converged: bool,
    /// Smoothed level component.
    pub level: Vec<f64>,
    /// Smoothed slope component (for Trend and BSM).
    pub slope: Option<Vec<f64>>,
    /// Smoothed seasonal component (for BSM).
    pub seasonal: Option<Vec<f64>>,
    /// Filtered level (R's `fitted(fit)[, "level"]`).
    #[serde(default)]
    pub filtered_level: Vec<f64>,
    /// Filtered slope (R's `fitted(fit)[, "slope"]`; for Trend and BSM).
    #[serde(default)]
    pub filtered_slope: Option<Vec<f64>>,
    /// Filtered seasonal (R's `fitted(fit)[, "sea"]`; for BSM).
    #[serde(default)]
    pub filtered_seasonal: Option<Vec<f64>>,
    /// R's `residuals(fit)`: v_t / sqrt(F_t), NaN where the value is missing.
    #[serde(default)]
    pub standardized_residuals: Vec<f64>,
    /// R's names of the variances held at `fixed` ("level", "slope", "seas", "epsilon").
    #[serde(default)]
    pub fixed_parameters: Vec<String>,
    /// The optimiser's convergence record.
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings: the convergence note, a failed line search.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for StructTsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

/// Estimated variance coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructTsCoefficients {
    /// Level variance (σ²_η); exactly 0 where the bound is active, as every variance here.
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
/// * `y` - Time series values; NaN is a missing value, which the Kalman filter
///   passes through, as R does. The first value must not be missing, as in R, and an
///   infinite value is an error
/// * `config` - Model configuration
///
/// # Returns
///
/// `StructTsResult` with fitted model and components, or
/// [`EconError::Cancelled`] when the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline (checked
/// once per optimiser iteration and within each Kalman-filter pass).
///
/// The convergence record reports a stop at `max_iter` or a failed line search,
/// as `converged`, [`StructTsResult::convergence`] and a warning; the estimates are
/// then what the search reached.
/// [`EconError::InvalidSpecification`] for `max_iter` below 1, a `tolerance` that is
/// not finite and positive, a `fixed` of the wrong length or with a negative or
/// non-finite value, every variance fixed, or a constant series.
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
    let spec = StructuralSpec::from_type(config.model_type, config.period);
    if let Some(period) = spec.seasonal {
        check_bsm_period(period, n)?;
    }

    // The final pass holds the filter's and the smoother's histories at once.
    // Admit them as one figure before the optimiser, whose likelihood passes
    // keep none, so that a history above the ceiling is refused before any
    // fitting work.
    let m = spec.state_dim()?;
    admit_filter_and_smoother_history(n, m)
        .map_err(|e| with_remedy(e, &history_remedy(&spec, m)))?;

    let fit = fit_structural(
        y,
        spec,
        config.fixed.as_deref(),
        config.max_iter,
        config.tolerance,
    )?;

    let filter_result = kalman_filter(y, &fit.model, fit.init_state.view(), fit.init_cov.view())?;
    let smooth_result = kalman_smoother(&filter_result, &fit.model)?;

    // The filtered states (R's fitted()) and standardised residuals (residuals()).
    let q = spec.seasonal_index();
    let column =
        |states: &[Vec<f64>], j: usize| -> Vec<f64> { states.iter().map(|s| s[j]).collect() };
    let filtered_level = column(&filter_result.filtered_states, 0);
    let filtered_slope = spec
        .slope
        .then(|| column(&filter_result.filtered_states, 1));
    let filtered_seasonal = spec
        .seasonal
        .map(|_| column(&filter_result.filtered_states, q));
    let standardized_residuals: Vec<f64> = filter_result
        .innovations
        .iter()
        .zip(&filter_result.innovation_vars)
        .map(|(v, f)| v / f.sqrt())
        .collect();

    // The smoothed components (R's tsSmooth()).
    let level = column(&smooth_result.smoothed_states, 0);
    let slope = spec
        .slope
        .then(|| column(&smooth_result.smoothed_states, 1));
    let seasonal = spec
        .seasonal
        .map(|_| column(&smooth_result.smoothed_states, q));

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

    let coef = spec.coefficients(&fit.variances);
    let names = spec.parameter_names();
    let fixed_parameters: Vec<String> = names
        .iter()
        .zip(&fit.free)
        .filter(|(_, free)| !**free)
        .map(|(name, _)| name.to_string())
        .collect();
    let n_params = fit.free.iter().filter(|free| **free).count();

    let log_lik = fit.log_likelihood;
    let aic = -2.0 * log_lik + 2.0 * n_params as f64;
    let bic = -2.0 * log_lik + (n_params as f64) * (fit.n_used as f64).ln();

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
        converged: fit.convergence.converged,
        level,
        slope,
        seasonal,
        filtered_level,
        filtered_slope,
        filtered_seasonal,
        standardized_residuals,
        fixed_parameters,
        convergence: fit.convergence,
        warnings: fit.warnings,
    })
}

/// Convenience function to run StructTS on a Dataset.
///
/// Errors as [`struct_ts`] does, including [`EconError::Cancelled`] on a
/// stopped budget.
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
pub fn run_struct_ts(
    dataset: &Dataset,
    column: &str,
    model_type: StructTsType,
    period: Option<usize>,
) -> EconResult<StructTsResult> {
    let (y, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

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

/// What to change when the Kalman history of an `m`-dimensional state is
/// refused: for BSM the state dimension follows from `period`.
fn history_remedy(spec: &StructuralSpec, m: usize) -> String {
    match spec.seasonal {
        Some(period) => format!(
            "{}; use a smaller period or a shorter series",
            bsm_dim_note(period, m)
        ),
        None => "use a shorter series".to_string(),
    }
}

/// How a BSM's `period` sets the state dimension `m`, for a remedy. The one
/// wording [`history_remedy`] and the matrices' remedy share, so the two cannot
/// drift apart.
fn bsm_dim_note(period: usize, m: usize) -> String {
    format!("with period {period} the state dimension is period + 1 = {m}")
}

/// The remedy of a refused m × m matrix of a BSM.
fn bsm_square_remedy(spec: &StructuralSpec, m: usize) -> String {
    format!(
        "{}; use a smaller period",
        bsm_dim_note(spec.seasonal.unwrap_or(DEFAULT_BSM_PERIOD), m)
    )
}

/// How a caller of the engine names its model and checks its seasonal period.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecNaming {
    /// The method named in errors and warnings.
    pub(crate) method: &'static str,
    /// What the m × m transition matrix is called in a refusal.
    pub(crate) transition: &'static str,
    /// What the m × m prior covariance is called in a refusal.
    pub(crate) prior: &'static str,
    /// The remedy of a refused m × m matrix.
    pub(crate) remedy: fn(&StructuralSpec, usize) -> String,
    /// `Ok` when the seasonal period fits this many observations.
    pub(crate) check_period: fn(usize, usize) -> EconResult<()>,
}

const STRUCT_TS_NAMING: SpecNaming = SpecNaming {
    method: "StructTS",
    transition: "the BSM transition matrix",
    prior: "the BSM initial state covariance",
    remedy: bsm_square_remedy,
    check_period: check_bsm_period,
};

/// A structural model: a level always, an optional slope, an optional dummy
/// seasonal with `seasonal` seasons. The layout is that of R's `makeLevel`,
/// `makeTrend` and `makeBSM`; `slope = false` with a seasonal is CausalImpact's
/// level + seasonal model. The one source of the state dimension.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StructuralSpec {
    pub(crate) slope: bool,
    pub(crate) seasonal: Option<usize>,
    pub(crate) naming: SpecNaming,
}

impl StructuralSpec {
    /// The StructTS model of `model_type`; BSM's period is `period`, 12 when unset.
    pub(crate) fn from_type(model_type: StructTsType, period: Option<usize>) -> Self {
        let (slope, seasonal) = match model_type {
            StructTsType::Level => (false, None),
            StructTsType::Trend => (true, None),
            StructTsType::BSM => (true, Some(period.unwrap_or(DEFAULT_BSM_PERIOD))),
        };
        Self {
            slope,
            seasonal,
            naming: STRUCT_TS_NAMING,
        }
    }

    /// 1 + slope + (seasons - 1): checked, so a huge period cannot wrap.
    pub(crate) fn state_dim(&self) -> EconResult<usize> {
        let base = 1 + usize::from(self.slope);
        match self.seasonal {
            Some(s) => base
                .checked_add(s.saturating_sub(1))
                .ok_or_else(|| overflow("the state dimension")),
            None => Ok(base),
        }
    }

    /// The index of the first seasonal state.
    pub(crate) fn seasonal_index(&self) -> usize {
        1 + usize::from(self.slope)
    }

    /// R's names of the variances, in R's order.
    pub(crate) fn parameter_names(&self) -> Vec<&'static str> {
        let mut names = vec!["level"];
        if self.slope {
            names.push("slope");
        }
        if self.seasonal.is_some() {
            names.push("seas");
        }
        names.push("epsilon");
        names
    }

    /// The state index each disturbance enters: level, [slope], [seasonal].
    fn disturbance_states(&self) -> Vec<usize> {
        let mut states = vec![0];
        if self.slope {
            states.push(1);
        }
        if self.seasonal.is_some() {
            states.push(self.seasonal_index());
        }
        states
    }

    /// An m × m zero matrix, `what` it is, allocated fallibly: a
    /// [`EconError::ResourceLimit`] with the caller's remedy when its size
    /// overflows, exceeds the per-allocation ceiling or cannot be allocated.
    pub(crate) fn square(&self, m: usize, what: &'static str) -> EconResult<Array2<f64>> {
        try_zeros2(m, m, what).map_err(|e| with_remedy(e, &(self.naming.remedy)(self, m)))
    }

    /// The model at the absolute `variances`, in R's order.
    pub(crate) fn model(&self, variances: &[f64]) -> EconResult<StateSpaceModel> {
        let m = self.state_dim()?;
        let mut transition = self.square(m, self.naming.transition)?;
        transition[[0, 0]] = 1.0;
        if self.slope {
            transition[[0, 1]] = 1.0;
            transition[[1, 1]] = 1.0;
        }
        let mut observation = try_zeros1(m, self.naming.transition)?;
        observation[0] = 1.0;
        if self.seasonal.is_some() {
            // γ_{t+1} = -γ_t - ... - γ_{t-s+2}, and the older seasons shift down.
            let q = self.seasonal_index();
            for j in q..m {
                transition[[q, j]] = -1.0;
            }
            for i in (q + 1)..m {
                transition[[i, i - 1]] = 1.0;
            }
            observation[q] = 1.0;
        }
        let states = self.disturbance_states();
        let r = states.len();
        let mut selection = try_zeros2(m, r, self.naming.transition)?;
        let mut state_cov = Array2::zeros((r, r));
        for (a, &state) in states.iter().enumerate() {
            selection[[state, a]] = 1.0;
            state_cov[[a, a]] = variances[a];
        }
        StateSpaceModel::new(transition, observation, selection, state_cov, variances[r])
    }

    /// R's prior: a_0 = (y_1, 0, ...), P_0 = 1e6 · vx in every entry, and
    /// vx = var(y)/100 over the non-missing values.
    pub(crate) fn prior(&self, y: &[f64]) -> EconResult<(Array1<f64>, Array2<f64>, f64)> {
        let method = self.naming.method;
        let m = self.state_dim()?;
        let mut init_cov = self.square(m, self.naming.prior)?;
        let mut init_state = try_zeros1(m, self.naming.prior)?;
        if y.first().is_some_and(|v| v.is_nan()) {
            return Err(EconError::InvalidSpecification {
                message: format!("{method}: the first value of the series must not be missing"),
            });
        }
        let (mut count, mut mean, mut sum2) = (0usize, 0.0f64, 0.0f64);
        for &v in y.iter().filter(|v| !v.is_nan()) {
            count += 1;
            let delta = v - mean;
            mean += delta / count as f64;
            sum2 += delta * (v - mean);
        }
        if count < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: count,
                context: format!("{method} requires at least 2 non-missing observations"),
            });
        }
        let vx = sum2 / (count - 1) as f64 / 100.0;
        if !vx.is_finite() || vx <= 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "{method}: the series is constant (variance 0); there is no variance to estimate"
                ),
            });
        }
        init_state[0] = y[0];
        init_cov.fill(1e6 * vx);
        Ok((init_state, init_cov, vx))
    }

    /// The variances as [`StructTsCoefficients`].
    pub(crate) fn coefficients(&self, variances: &[f64]) -> StructTsCoefficients {
        let mut next = 1;
        let mut take = |present: bool| {
            present.then(|| {
                let v = variances[next];
                next += 1;
                v
            })
        };
        let slope = take(self.slope);
        let seasonal = take(self.seasonal.is_some());
        StructTsCoefficients {
            level: variances[0],
            slope,
            seasonal,
            epsilon: variances[variances.len() - 1],
        }
    }
}

/// A fitted structural model.
pub(crate) struct StructuralFit {
    /// Absolute variances, in R's order, at least 0.
    pub(crate) variances: Vec<f64>,
    pub(crate) free: Vec<bool>,
    pub(crate) log_likelihood: f64,
    /// Non-missing observations.
    pub(crate) n_used: usize,
    /// The model at the estimate.
    pub(crate) model: StateSpaceModel,
    pub(crate) init_state: Array1<f64>,
    pub(crate) init_cov: Array2<f64>,
    pub(crate) convergence: Convergence,
    pub(crate) warnings: Vec<String>,
}

/// StructTS's objective over the free variances, in units of vx: f =
/// 0.5 (Σ v²/F + Σ ln F) / n_obs and its exact gradient. The model and the
/// derivatives are built once; an evaluation sets the variances and runs one
/// differentiated filter pass, which keeps no history.
struct StructuralObjective<'a> {
    y: &'a [f64],
    /// Variances in units of vx; the free entries are overwritten per evaluation.
    p: Vec<f64>,
    free_index: Vec<usize>,
    vx: f64,
    model: StateSpaceModel,
    init_state: Array1<f64>,
    init_cov: Array2<f64>,
    derivatives: Vec<ModelDerivative>,
    disturbances: usize,
    n_used: usize,
}

impl<'a> StructuralObjective<'a> {
    fn new(y: &'a [f64], spec: StructuralSpec, fixed: Option<&[Option<f64>]>) -> EconResult<Self> {
        let names = spec.parameter_names();
        let k = names.len();
        if let Some(fixed) = fixed {
            let invalid = |message: String| Err(EconError::InvalidSpecification { message });
            if fixed.len() != k {
                return invalid(format!(
                    "fixed must have {k} entries ({}) for this model, got {}",
                    names.join(", "),
                    fixed.len()
                ));
            }
            for (name, value) in names.iter().zip(fixed) {
                if let Some(v) = value {
                    if !v.is_finite() || *v < 0.0 {
                        return invalid(format!(
                            "fixed: the {name} variance must be finite and at least 0, got {v}"
                        ));
                    }
                }
            }
            if fixed.iter().all(Option::is_some) {
                return invalid("all parameters were fixed".to_string());
            }
        }

        let (init_state, init_cov, vx) = spec.prior(y)?;
        let value_of = |i: usize| fixed.and_then(|f| f[i]);
        let p: Vec<f64> = (0..k).map(|i| value_of(i).unwrap_or(0.0) / vx).collect();
        let free_index: Vec<usize> = (0..k).filter(|&i| value_of(i).is_none()).collect();
        let n_used = y.iter().filter(|v| !v.is_nan()).count();

        let m = spec.state_dim()?;
        let states = spec.disturbance_states();
        let mut derivatives = Vec::with_capacity(free_index.len());
        for &i in &free_index {
            let mut state_dist_cov = spec.square(m, spec.naming.transition)?;
            let obs_var = if i < states.len() {
                state_dist_cov[[states[i], states[i]]] = vx;
                0.0
            } else {
                vx
            };
            derivatives.push(ModelDerivative {
                state_dist_cov,
                obs_var,
            });
        }
        let model = spec.model(&vec![0.0; k])?;
        Ok(Self {
            y,
            p,
            free_index,
            vx,
            model,
            init_state,
            init_cov,
            derivatives,
            disturbances: states.len(),
            n_used,
        })
    }

    /// Set the free variances to `x`, in units of vx.
    fn set_free(&mut self, x: &[f64]) {
        for (&i, &value) in self.free_index.iter().zip(x) {
            self.p[i] = value;
        }
        for a in 0..self.disturbances {
            self.model.state_cov[[a, a]] = self.p[a] * self.vx;
        }
        self.model.obs_var = self.p[self.disturbances] * self.vx;
    }

    /// Objective, gradient and scoring matrix at `x`. With every variance 0 the
    /// filter has no positive innovation variance; the value is 1000, as R's `getLike`.
    fn eval(&mut self, x: &[f64]) -> EconResult<Evaluation> {
        self.set_free(x);
        if self.p.iter().all(|v| *v == 0.0) {
            return Ok(Evaluation {
                f: 1000.0,
                g: vec![0.0; x.len()],
                scoring: None,
            });
        }
        let terms = kalman_likelihood_terms(
            self.y,
            &self.model,
            self.init_state.view(),
            self.init_cov.view(),
            &self.derivatives,
        )?;
        let n = terms.n_used as f64;
        let f = 0.5 * (terms.ssq + terms.sumlog) / n;
        let g = terms
            .d_ssq
            .iter()
            .zip(&terms.d_sumlog)
            .map(|(a, b)| 0.5 * (a + b) / n)
            .collect();
        let scoring = terms.info.iter().map(|v| v / n).collect();
        Ok(Evaluation {
            f,
            g,
            scoring: Some(scoring),
        })
    }
}

/// Fit the model of `spec` to `y` by maximum likelihood: the prior of
/// [`StructuralSpec::prior`], the variances in units of vx starting at 1,
/// [`minimize_nonnegative`] on StructTS's objective. `fixed` holds absolute
/// variances in R's order. `y` may hold NaN (missing), not an infinite value.
pub(crate) fn fit_structural(
    y: &[f64],
    spec: StructuralSpec,
    fixed: Option<&[Option<f64>]>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<StructuralFit> {
    let method = spec.naming.method;
    if max_iter < 1 {
        return Err(EconError::InvalidSpecification {
            message: format!("max_iter must be at least 1, got {max_iter}"),
        });
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be finite and positive, got {tolerance}"),
        });
    }
    if let Some(period) = spec.seasonal {
        (spec.naming.check_period)(period, y.len())?;
    }
    if let Some(t) = y.iter().position(|v| v.is_infinite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("y[{t}] is infinite; use NaN for a missing value"),
        });
    }

    let mut objective = StructuralObjective::new(y, spec, fixed)?;
    let x0 = vec![1.0; objective.free_index.len()];
    let minimum =
        minimize_nonnegative(&mut |x: &[f64]| objective.eval(x), &x0, max_iter, tolerance)?;

    objective.set_free(&minimum.x);
    let variances: Vec<f64> = objective
        .p
        .iter()
        .map(|p| (p * objective.vx).max(0.0))
        .collect();
    let free: Vec<bool> = (0..variances.len())
        .map(|i| objective.free_index.contains(&i))
        .collect();
    let n_used = objective.n_used;
    let n = n_used as f64;
    let log_likelihood = -n * minimum.f - 0.5 * n * (2.0 * std::f64::consts::PI).ln();

    let convergence = Convergence::new(
        minimum.converged,
        minimum.iterations,
        max_iter,
        "scaled projected gradient",
        minimum.crit,
        tolerance,
    );
    let mut warnings: Vec<String> = convergence.note(method).into_iter().collect();
    if minimum.line_search_failed {
        warnings.push(format!(
            "{method}: the line search could not reduce the objective further; the estimates \
             may not be at the optimum"
        ));
    }

    Ok(StructuralFit {
        variances,
        free,
        log_likelihood,
        n_used,
        model: objective.model,
        init_state: objective.init_state,
        init_cov: objective.init_cov,
        convergence,
        warnings,
    })
}

/// One evaluation of the objective for [`minimize_nonnegative`].
struct Evaluation {
    f: f64,
    g: Vec<f64>,
    /// The expected Hessian at this point, row-major, when the objective has one.
    scoring: Option<Vec<f64>>,
}

impl Evaluation {
    fn is_finite(&self) -> bool {
        self.f.is_finite() && self.g.iter().all(|v| v.is_finite())
    }
}

/// What [`minimize_nonnegative`] returns.
struct Minimum {
    x: Vec<f64>,
    f: f64,
    iterations: usize,
    converged: bool,
    /// The scaled projected gradient at `x`.
    crit: f64,
    line_search_failed: bool,
}

/// The forward-difference step of the Hessian, relative to max(x, 1).
const HESSIAN_STEP: f64 = 1e-6;
/// The Armijo constant of the projected line search.
const ARMIJO: f64 = 1e-4;
/// A change in the objective no double-precision search can rely on.
const STALL_F: f64 = 1e-14;
/// A predicted reduction of the objective, relative to max(|f|, 1), within the
/// rounding of an objective summed over a series.
const NOISE_F: f64 = 1e-10;
/// The reduction in the objective below which scoring gives way to Newton steps.
const SCORING_UNTIL: f64 = 1e-6;
/// Line-search halvings before the search gives up.
const MAX_HALVINGS: usize = 60;

/// A trial point the search may treat as failed: F ≤ 0 somewhere in the filter.
/// A stopped budget or an allocation refusal stops the search instead: an
/// objective that scores every error as a bad point keeps a stopped search running.
fn is_failed_trial(e: &EconError) -> bool {
    matches!(e, EconError::Computation(_))
}

/// `max_{i∈F} |g_i| max(x_i, 1)` over the free set F = complement of
/// {i : x_i = 0, g_i ≥ 0}: scale-free, since x is in units of vx and the
/// objective only shifts by a constant when the data are rescaled.
fn projected_gradient(x: &[f64], g: &[f64]) -> f64 {
    x.iter()
        .zip(g)
        .filter(|&(&xi, &gi)| !(xi == 0.0 && gi >= 0.0))
        .fold(0.0, |crit, (&xi, &gi)| {
            f64::max(crit, gi.abs() * xi.max(1.0))
        })
}

/// A Newton direction on the free set: `free` indexes x, `h` is the free block
/// of the Hessian, symmetrised. A Levenberg shift makes it positive definite when
/// it is not; a failed factorisation or an ascent direction is steepest descent.
fn newton_direction(h: &Array2<f64>, g_free: &[f64]) -> Vec<f64> {
    let k = g_free.len();
    let descent: Vec<f64> = g_free.iter().map(|gi| -gi).collect();
    let Ok((values, _)) = eig_symmetric(&h.view()) else {
        return descent;
    };
    let lambda_max = values.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let lambda_min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let shift = if lambda_max == 0.0 {
        1.0
    } else if lambda_min <= 1e-12 * lambda_max {
        -lambda_min + 1e-6 * lambda_max
    } else {
        0.0
    };
    let mut shifted = h.clone();
    for i in 0..k {
        shifted[[i, i]] += shift;
    }
    let Ok(l) = cholesky(&shifted.view()) else {
        return descent;
    };
    // L z = -g, then L' d = z.
    let mut d = descent.clone();
    for i in 0..k {
        for j in 0..i {
            d[i] -= l[[i, j]] * d[j];
        }
        d[i] /= l[[i, i]];
    }
    for i in (0..k).rev() {
        for j in (i + 1)..k {
            d[i] -= l[[j, i]] * d[j];
        }
        d[i] /= l[[i, i]];
    }
    let slope: f64 = d.iter().zip(g_free).map(|(a, b)| a * b).sum();
    if d.iter().all(|v| v.is_finite()) && slope < 0.0 {
        d
    } else {
        descent
    }
}

/// Minimise `fg` over x ≥ 0 by a projected Newton method: the Hessian of the free
/// set from forward differences of the exact gradient, a Levenberg shift, a
/// projected Armijo search. While the objective supplies a scoring matrix and each
/// step still reduces it by more than [`SCORING_UNTIL`], that matrix stands in for the
/// Hessian, which saves a gradient per free coordinate; the search then finishes on
/// Newton steps, whose rate the stopping tolerance needs. A coordinate reaches its
/// bound 0 exactly. It stops when
/// the scaled projected gradient is at most `tolerance`, when no further reduction
/// is possible in double precision, or at `max_iter`. An `Err` other than a failed
/// trial ([`is_failed_trial`]) stops it at once; a failed trial at the start is an
/// error. The budget is checked once per iteration.
fn minimize_nonnegative(
    fg: &mut dyn FnMut(&[f64]) -> EconResult<Evaluation>,
    x0: &[f64],
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Minimum> {
    let mut x: Vec<f64> = x0.iter().map(|v| v.max(0.0)).collect();
    let start = fg(&x)?;
    if !start.f.is_finite() {
        return Err(EconError::Computation(format!(
            "the objective is {} at the starting point",
            start.f
        )));
    }
    let Evaluation {
        mut f,
        mut g,
        mut scoring,
    } = start;
    let mut scoring_phase = true;
    // Scoring converges linearly: a stop on the criterion after scoring steps waits
    // for one Newton step taken from within it, whose quadratic rate puts the
    // estimate well inside the tolerance.
    let (mut scored, mut polished) = (false, false);
    let mut iterations = 0usize;
    let mut converged = false;
    let mut line_search_failed = false;

    loop {
        checkpoint()?;
        let at_tolerance = projected_gradient(&x, &g) <= tolerance;
        if at_tolerance && (!scored || polished) {
            converged = true;
            break;
        }
        if at_tolerance {
            scoring_phase = false;
        }
        if iterations >= max_iter {
            break;
        }
        let free: Vec<usize> = (0..x.len())
            .filter(|&i| !(x[i] == 0.0 && g[i] >= 0.0))
            .collect();
        let g_free: Vec<f64> = free.iter().map(|&i| g[i]).collect();

        // The Hessian of the free block: the scoring matrix, or forward differences of
        // the gradient.
        let k = free.len();
        let mut h = Array2::zeros((k, k));
        let mut usable = true;
        let scoring_step = scoring_phase && scoring.is_some();
        scored |= scoring_step;
        if let Some(info) = scoring.as_deref().filter(|_| scoring_step) {
            for (row, &i) in free.iter().enumerate() {
                for (col, &j) in free.iter().enumerate() {
                    h[[row, col]] = info[i * x.len() + j];
                }
            }
        } else {
            for (col, &j) in free.iter().enumerate() {
                let step = HESSIAN_STEP * x[j].max(1.0);
                let mut trial = x.clone();
                trial[j] += step;
                match fg(&trial) {
                    Ok(at) if at.is_finite() => {
                        for (row, &i) in free.iter().enumerate() {
                            h[[row, col]] = (at.g[i] - g[i]) / step;
                        }
                    }
                    Ok(_) => {
                        usable = false;
                        break;
                    }
                    Err(e) if is_failed_trial(&e) => {
                        usable = false;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        let direction = if usable {
            let symmetric = (&h + &h.t()) * 0.5;
            newton_direction(&symmetric, &g_free)
        } else {
            let scale = g_free.iter().fold(1.0f64, |a, v| a.max(v.abs()));
            g_free.iter().map(|v| -v / scale).collect()
        };

        // A Newton step that predicts a reduction within the rounding of f cannot be
        // judged by f: it is judged by the gradient, which is exact to far below that.
        let predicted: f64 = direction.iter().zip(&g_free).map(|(d, gi)| -d * gi).sum();
        let below_noise = !scoring_step && usable && predicted <= NOISE_F * f.abs().max(1.0);
        let crit = projected_gradient(&x, &g);

        // The projected Armijo search.
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..MAX_HALVINGS {
            let mut trial = x.clone();
            for (&i, d) in free.iter().zip(&direction) {
                trial[i] = (x[i] + alpha * d).max(0.0);
            }
            if trial == x {
                break;
            }
            match fg(&trial) {
                Ok(at) if at.is_finite() => {
                    let decrease: f64 = trial
                        .iter()
                        .zip(&x)
                        .zip(&g)
                        .map(|((t, xi), gi)| gi * (t - xi))
                        .sum();
                    let by_gradient =
                        below_noise && alpha == 1.0 && projected_gradient(&trial, &at.g) < crit;
                    if by_gradient || at.f <= f + ARMIJO * decrease {
                        accepted = Some((trial, at));
                        break;
                    }
                }
                Ok(_) => {}
                Err(e) if is_failed_trial(&e) => {}
                Err(e) => return Err(e),
            }
            alpha *= 0.5;
        }
        let Some((next, at)) = accepted else {
            if scoring_step {
                // The scoring direction failed; the Newton one may not.
                scoring_phase = false;
                continue;
            }
            // A Newton step predicting a reduction within the rounding of f that
            // reduces neither f nor the gradient: the search is at the optimum as far
            // as double precision resolves it.
            if at_tolerance || below_noise {
                converged = true;
            } else {
                line_search_failed = true;
            }
            break;
        };

        let change_f = (f - at.f).abs();
        let change_x = next
            .iter()
            .zip(&x)
            .fold(0.0f64, |a, (n, o)| a.max((n - o).abs() / n.max(1.0)));
        x = next;
        f = at.f;
        g = at.g;
        scoring = at.scoring;
        iterations += 1;
        polished |= at_tolerance;
        if change_f <= SCORING_UNTIL {
            scoring_phase = false;
        }
        if change_f <= STALL_F && change_x <= 1e-12 {
            // No further reduction is possible in double precision.
            converged = true;
            break;
        }
    }

    Ok(Minimum {
        crit: projected_gradient(&x, &g),
        x,
        f,
        iterations,
        converged,
        line_search_failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_struct_ts_type_parses_aliases_and_rejects_unknown() {
        StructTsType::choices().validate().unwrap();
        assert_eq!("bsm".parse::<StructTsType>().unwrap(), StructTsType::BSM);
        assert_eq!(
            "local linear trend".parse::<StructTsType>().unwrap(),
            StructTsType::Trend
        );
        assert_eq!("LLT".parse::<StructTsType>().unwrap(), StructTsType::Trend);
        assert_eq!(
            "local-level".parse::<StructTsType>().unwrap(),
            StructTsType::Level
        );
        match "arima".parse::<StructTsType>() {
            Err(crate::errors::EconError::InvalidSpecification { message }) => {
                assert_eq!(
                    message,
                    "unknown model_type 'arima'; expected one of: level, trend, BSM"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The search on a plain objective, which has no scoring matrix: Newton throughout.
    fn minimize_nonnegative(
        fg: &mut dyn FnMut(&[f64]) -> EconResult<(f64, Vec<f64>)>,
        x0: &[f64],
        max_iter: usize,
        tolerance: f64,
    ) -> EconResult<Minimum> {
        super::minimize_nonnegative(
            &mut |x: &[f64]| {
                fg(x).map(|(f, g)| Evaluation {
                    f,
                    g,
                    scoring: None,
                })
            },
            x0,
            max_iter,
            tolerance,
        )
    }

    /// X-2: an Int64 column fits as its Float64 copy; missing values, text and absent
    /// columns are refused by name.
    #[test]
    fn test_na_struct_ts_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let run = |col: &str| run_struct_ts(&ds, col, StructTsType::Level, None);
        let int = run("ai").unwrap();
        let flt = run("ai_f64").unwrap();
        assert_eq!(int.log_likelihood.to_bits(), flt.log_likelihood.to_bits());
        assert_same(&int.level, &flt.level);
        assert_null_values(&run("a_na").unwrap_err(), "a_na", 2);
        assert_null_values(&run("ai_na").unwrap_err(), "ai_na", 2);
        assert_non_numeric(&run("lab").unwrap_err(), "lab");
        assert_not_found(&run("nope").unwrap_err(), "nope");
    }

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
    fn test_insufficient_data() {
        let y = vec![1.0, 2.0, 3.0];
        let result = struct_ts(&y, StructTsConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_aic_bic() {
        let y = generate_local_level_data(50, 1.0, 0.5, 42);
        let result = struct_ts(&y, StructTsConfig::default()).unwrap();
        assert_eq!(
            result.aic,
            -2.0 * result.log_likelihood + 2.0 * result.n_params as f64
        );

        // AIC and BIC should be finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // BIC penalizes more than AIC for this sample size
        // BIC = -2*LL + k*ln(n), AIC = -2*LL + 2k
        // For n=50, ln(50) ≈ 3.9 > 2, so BIC > AIC
        assert!(result.bic > result.aic);
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
        // test above passes on the filter's own checkpoint. Call the engine
        // directly (no final pass): an objective that scored a Cancelled error
        // as a failed trial would leave the search running.
        let y = generate_local_level_data(48, 1.0, 0.5, 11);
        for (model_type, period) in [
            (StructTsType::Level, None),
            (StructTsType::Trend, None),
            (StructTsType::BSM, Some(4)),
        ] {
            let spec = StructuralSpec::from_type(model_type, period);
            assert!(
                matches!(
                    cancelled(|| fit_structural(&y, spec, None, 100, 1e-8)),
                    Err(EconError::Cancelled(_))
                ),
                "{model_type:?}"
            );
            // With no iterations the loop's work never runs, so this pins the
            // pass-through at the starting point alone.
            let mut objective = StructuralObjective::new(&y, spec, None).unwrap();
            let x0 = vec![1.0; objective.free_index.len()];
            assert!(
                matches!(
                    cancelled(|| minimize_nonnegative(
                        &mut |x: &[f64]| objective.eval(x).map(|e| (e.f, e.g)),
                        &x0,
                        0,
                        1e-8
                    )),
                    Err(EconError::Cancelled(_))
                ),
                "{model_type:?}"
            );
            // Without a budget nothing changes.
            assert!(fit_structural(&y, spec, None, 100, 1e-8).is_ok());
        }
    }

    #[test]
    fn minimize_nonnegative_checkpoints_each_iteration() {
        // A quartic: Newton's rate on it is 2/3, so the free run takes many
        // iterations. The objective never looks at the budget, so only the
        // loop's own checkpoint can stop the search.
        fn quartic(x: &[f64]) -> EconResult<(f64, Vec<f64>)> {
            Ok((
                x.iter().map(|v| v.powi(4)).sum(),
                x.iter().map(|v| 4.0 * v.powi(3)).collect(),
            ))
        }
        let init = [3.0, 4.0];
        const TOL: f64 = 1e-13;
        assert!(matches!(
            cancelled(|| minimize_nonnegative(&mut quartic, &init, 500, TOL)),
            Err(EconError::Cancelled(_))
        ));

        // A budget cancelled mid-search. The objective cancels the live token
        // on its CANCEL_AT-th evaluation. With a check on every iteration, at
        // most one iteration's evaluations follow the cancel (the Hessian's k
        // columns and the line search's halvings). A check made only before the
        // loop would let the search run on and return Ok.
        const CANCEL_AT: usize = 20;
        let bound = CANCEL_AT + init.len() + 60;
        let mut free_evals = 0usize;
        let free = minimize_nonnegative(
            &mut |x: &[f64]| {
                free_evals += 1;
                quartic(x)
            },
            &init,
            500,
            TOL,
        )
        .unwrap();
        assert!(free.converged, "{}", free.crit);
        // The uncancelled search needs more evaluations than the bound allows,
        // so the bound below can only hold if the search was stopped.
        assert!(free_evals > bound, "{free_evals} evaluations");

        let token = crate::resource::CancelToken::new();
        let mut evals = 0usize;
        let budget = crate::resource::Budget::new(None, token.clone());
        let stopped = crate::resource::with_budget(budget, || {
            minimize_nonnegative(
                &mut |x: &[f64]| {
                    evals += 1;
                    if evals == CANCEL_AT {
                        token.cancel();
                    }
                    quartic(x)
                },
                &init,
                500,
                TOL,
            )
        });
        assert!(matches!(stopped, Err(EconError::Cancelled(_))));
        assert!(evals <= bound, "{evals} evaluations");
    }

    #[test]
    fn minimize_nonnegative_judges_a_step_below_the_rounding_of_the_objective_by_its_gradient() {
        // f is a quadratic with a jump at every other point, as rounding noise, so
        // every trial looks worse by f. A Newton step from just off the optimum
        // predicts a reduction within that noise and is judged by the gradient: it is
        // taken and the search converges. From far off it is judged by f: a failed
        // search. With a gradient as noisy as f nothing is gained either way, and the
        // search stops as converged where f cannot resolve the step.
        let run = |offset: f64, gradient_noise: f64| {
            let x0 = [1.0 + offset];
            let mut noisy = move |x: &[f64]| -> EconResult<(f64, Vec<f64>)> {
                let moved = x[0] != x0[0];
                let noise = if moved { 1.0 } else { 0.0 };
                let g = x[0] - 1.0 + if moved { gradient_noise } else { 0.0 };
                Ok((0.5 * (x[0] - 1.0).powi(2) + noise, vec![g]))
            };
            minimize_nonnegative(&mut noisy, &x0, 100, 1e-12).unwrap()
        };
        let near = run(1e-8, 0.0);
        assert!(near.converged && !near.line_search_failed, "{}", near.crit);
        assert!((near.x[0] - 1.0).abs() < 1e-15, "{}", near.x[0]);
        let flat = run(1e-8, 1.0);
        assert!(flat.converged && !flat.line_search_failed);
        assert_eq!(flat.x[0], 1.0 + 1e-8);
        let far = run(1.0, 0.0);
        assert!(!far.converged && far.line_search_failed, "{}", far.crit);
    }

    #[test]
    fn minimize_nonnegative_treats_only_a_failed_filter_as_a_failed_trial() {
        // A Computation error in a line-search trial halves the step; a stopped
        // budget or an allocation refusal stops the search at once.
        let mut calls = 0usize;
        let found = minimize_nonnegative(
            &mut |x: &[f64]| {
                calls += 1;
                if x[0] > 2.0 {
                    return Err(EconError::Computation("F <= 0".into()));
                }
                Ok(((x[0] - 1.5).powi(2), vec![2.0 * (x[0] - 1.5)]))
            },
            &[0.1],
            100,
            1e-10,
        )
        .unwrap();
        assert!(
            found.converged && (found.x[0] - 1.5).abs() < 1e-8,
            "{:?}",
            found.x
        );
        let mut after = 0usize;
        let stopped = minimize_nonnegative(
            &mut |x: &[f64]| {
                if x[0] == 1.0 {
                    return Ok((x[0] * x[0], vec![2.0 * x[0]]));
                }
                after += 1;
                Err(EconError::ResourceLimit("refused".into()))
            },
            &[1.0],
            100,
            1e-10,
        );
        assert!(matches!(stopped, Err(EconError::ResourceLimit(_))));
        assert_eq!(after, 1);
        // The bound is reached exactly and the gradient there may stay positive.
        let bound = minimize_nonnegative(
            &mut |x: &[f64]| Ok(((x[0] + 1.0).powi(2), vec![2.0 * (x[0] + 1.0)])),
            &[1.0],
            100,
            1e-10,
        )
        .unwrap();
        assert_eq!(bound.x[0], 0.0);
        assert!(bound.converged);
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
        let spec = StructuralSpec::from_type(config.model_type, config.period);
        let bounded = crate::resource::with_alloc_ceiling(ceiling, || {
            fit_structural(&y, spec, None, config.max_iter, config.tolerance)
        })
        .unwrap();
        let unbounded = fit_structural(&y, spec, None, config.max_iter, config.tolerance).unwrap();
        assert_eq!(bounded.variances, unbounded.variances);
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
        // The admission sizes the history by the spec's state_dim; the filter
        // runs the model the spec builds, from its prior. All must agree.
        let y = seasonal_series(48, 12, 23);
        for (model_type, period, m) in [
            (StructTsType::Level, None, 1),
            (StructTsType::Trend, None, 2),
            (StructTsType::BSM, Some(4), 5),
            (StructTsType::BSM, Some(12), 13),
            (StructTsType::BSM, None, 13),
        ] {
            let spec = StructuralSpec::from_type(model_type, period);
            let variances = vec![1.0; spec.parameter_names().len()];
            assert_eq!(spec.state_dim().unwrap(), m, "{model_type:?} {period:?}");
            assert_eq!(spec.model(&variances).unwrap().state_dim(), m);
            let (a0, p0, _) = spec.prior(&y).unwrap();
            assert_eq!(a0.len(), m);
            assert_eq!(p0.dim(), (m, m));
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
        // The m × m transition and prior (m = period + 1) are allocated fallibly:
        // a matrix the allocator cannot provide is a ResourceLimit that names
        // period, and one whose size overflows is refused before any allocation.
        let y = seasonal_series(48, 12, 29);
        let spec = |period| StructuralSpec::from_type(StructTsType::BSM, period);
        let variances = [1.0, 1.0, 1.0, 1.0];
        let remedy =
            "; with period 12 the state dimension is period + 1 = 13; use a smaller period";
        // 8 · 13² = 1,352 bytes > 1 KiB.
        let ceiling = Some(1 << 10);
        match crate::resource::with_alloc_ceiling(ceiling, || spec(Some(12)).model(&variances)) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.starts_with("the BSM transition matrix would need")
                    && msg.ends_with(remedy)
                    && msg.matches("; use ").count() == 1,
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        match crate::resource::with_alloc_ceiling(ceiling, || spec(Some(12)).prior(&y)) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.starts_with("the BSM initial state covariance would need")
                    && msg.ends_with(remedy),
                "{msg}"
            ),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // An unset period is the default 12 in the remedy too.
        match crate::resource::with_alloc_ceiling(ceiling, || spec(None).prior(&y)) {
            Err(EconError::ResourceLimit(msg)) => assert!(msg.ends_with(remedy), "{msg}"),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        // An m × m whose size overflows is refused before any allocation.
        let huge = spec(Some(usize::MAX / 4));
        assert!(matches!(
            huge.model(&variances),
            Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
        ));
        assert!(matches!(
            huge.prior(&y),
            Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
        ));
        // Without a ceiling the prior is R's: 1e6 · var(y)/100 in every entry.
        let (a0, p0, vx) = spec(Some(12)).prior(&y).unwrap();
        assert!(p0.iter().all(|v| *v == 1e6 * vx));
        assert_eq!(a0[0], y[0]);
        assert!(a0.iter().skip(1).all(|v| *v == 0.0));
    }

    /// First n values of a case column; missing cells as NaN.
    fn case_series(ds: &crate::data::Dataset, col: &str, n: usize) -> Vec<f64> {
        let v = crate::data::columns::numeric_options(ds.df(), col).unwrap();
        v[..n].iter().map(|x| x.unwrap_or(f64::NAN)).collect()
    }

    /// The same column read from the generated CSV, for the property tests.
    fn repo_series(col: &str, n: usize) -> Vec<f64> {
        let ds = crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/structts_series.csv",
        );
        case_series(&ds, col, n)
    }

    fn fit(
        y: &[f64],
        t: StructTsType,
        period: Option<usize>,
        fixed: Option<Vec<Option<f64>>>,
    ) -> StructTsResult {
        struct_ts(
            y,
            StructTsConfig {
                model_type: t,
                period,
                fixed,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// R's order.
    fn coef_vec(r: &StructTsResult) -> Vec<f64> {
        let c = &r.coef;
        [Some(c.level), c.slope, c.seasonal, Some(c.epsilon)]
            .into_iter()
            .flatten()
            .collect()
    }

    fn col(r: &Option<Vec<f64>>) -> Vec<f64> {
        r.clone().unwrap()
    }

    // States and residuals cross zero, and the parameters agree to about 1e-8.
    fn state_tol() -> Tol {
        Tol::new(1e-6, 1e-8)
    }

    #[test]
    fn test_validate_structts_nile() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let r = fit(
            &case_series(&ds, "nile", 100),
            StructTsType::Level,
            None,
            None,
        );
        let names = ["level", "epsilon"];
        Check::new(&case)
            .named(
                "nile_coef_tight",
                &names,
                &coef_vec(&r),
                Tol::iterative(1e-6),
            )
            .scalar("nile_loglik_tight", r.log_likelihood, Tol::stat(100))
            .scalar("nile_loglik", r.log_likelihood, Tol::stat(100))
            // R's L-BFGS-B stops early: its measured gap to the optimum is 1.66e-5.
            .named("nile_coef", &names, &coef_vec(&r), Tol::iterative(1e-4))
            .vector("nile_filtered_level_tight", &r.filtered_level, state_tol())
            .vector("nile_smoothed_level_tight", &r.level, state_tol())
            .vector(
                "nile_std_resid_tight",
                &r.standardized_residuals,
                state_tol(),
            )
            .scalar(
                "nile_level_last",
                *r.filtered_level.last().unwrap(),
                Tol::iterative(1e-5),
            )
            .finish();
        assert!(r.converged && r.convergence.converged);
        assert_eq!(r.n_params, 2);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_structts_nile_trend() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let r = fit(
            &case_series(&ds, "nile", 100),
            StructTsType::Trend,
            None,
            None,
        );
        let names = ["level", "slope", "epsilon"];
        Check::new(&case)
            .named(
                "nile_trend_coef_tight",
                &names,
                &coef_vec(&r),
                Tol::iterative(1e-6),
            )
            .scalar("nile_trend_loglik_tight", r.log_likelihood, Tol::stat(100))
            .scalar("nile_trend_loglik", r.log_likelihood, Tol::stat(100))
            // R's L-BFGS-B stops early: its measured gap to the optimum is 7.55e-5.
            .named(
                "nile_trend_coef",
                &names,
                &coef_vec(&r),
                Tol::iterative(1e-4),
            )
            .vector("nile_trend_smoothed_level_tight", &r.level, state_tol())
            .vector(
                "nile_trend_smoothed_slope_tight",
                &col(&r.slope),
                state_tol(),
            )
            .finish();
        // The bound is reached exactly, as R.
        assert_eq!(r.coef.slope, Some(0.0));
    }

    #[test]
    fn test_validate_structts_ukgas() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let r = fit(
            &case_series(&ds, "ukgas", 108),
            StructTsType::BSM,
            Some(4),
            None,
        );
        let names = ["level", "slope", "seas", "epsilon"];
        Check::new(&case)
            .named(
                "ukgas_coef_tight",
                &names,
                &coef_vec(&r),
                Tol::iterative(1e-6),
            )
            .scalar("ukgas_loglik_tight", r.log_likelihood, Tol::stat(108))
            .scalar("ukgas_loglik", r.log_likelihood, Tol::stat(108))
            // R's L-BFGS-B stops early: its measured gaps to the optimum reach 1.50e-3.
            .named("ukgas_coef", &names, &coef_vec(&r), Tol::iterative(2e-3))
            .vector("ukgas_filtered_level_tight", &r.filtered_level, state_tol())
            .vector(
                "ukgas_filtered_slope_tight",
                &col(&r.filtered_slope),
                state_tol(),
            )
            .vector(
                "ukgas_filtered_sea_tight",
                &col(&r.filtered_seasonal),
                state_tol(),
            )
            .vector("ukgas_smoothed_level_tight", &r.level, state_tol())
            .vector("ukgas_smoothed_slope_tight", &col(&r.slope), state_tol())
            .vector("ukgas_smoothed_sea_tight", &col(&r.seasonal), state_tol())
            .vector(
                "ukgas_std_resid_tight",
                &r.standardized_residuals,
                state_tol(),
            )
            .finish();
        assert_eq!(r.coef.level, 0.0);
        assert!(r.converged);
    }

    #[test]
    fn test_validate_structts_fixed() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let y = case_series(&ds, "nile", 100);
        let r = fit(
            &y,
            StructTsType::Level,
            None,
            Some(vec![Some(1500.0), None]),
        );
        let names = ["level", "epsilon"];
        // The slope's bound, fixed instead of reached, is the same optimum.
        let slope_fixed = fit(
            &y,
            StructTsType::Trend,
            None,
            Some(vec![None, Some(0.0), None]),
        );
        Check::new(&case)
            .named(
                "nile_fixed_coef_tight",
                &names,
                &coef_vec(&r),
                Tol::iterative(1e-6),
            )
            .scalar("nile_fixed_loglik_tight", r.log_likelihood, Tol::stat(100))
            .scalar("nile_fixed_loglik", r.log_likelihood, Tol::stat(100))
            .named(
                "nile_trend_coef_tight",
                &["level", "slope", "epsilon"],
                &coef_vec(&slope_fixed),
                Tol::iterative(1e-6),
            )
            .finish();
        assert_eq!(r.coef.level, 1500.0);
        assert_eq!(r.n_params, 1);
        assert_eq!(r.fixed_parameters, ["level"]);
        assert_eq!(slope_fixed.fixed_parameters, ["slope"]);
    }

    #[test]
    fn test_validate_structts_missing() {
        // NaN at rows 21-25 and 61 (1-based): the filter passes through them.
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let r = fit(
            &case_series(&ds, "nile_na", 100),
            StructTsType::Level,
            None,
            None,
        );
        // deviation TS-27: nile_na_loglik and nile_na_loglik_tight are R's values, which
        // scale the data term by length(y)/n_obs; p2a reports the observed-data one.
        Check::new(&case)
            .named(
                "nile_na_coef_tight",
                &["level", "epsilon"],
                &coef_vec(&r),
                Tol::iterative(1e-6),
            )
            .scalar("nile_na_loglik_obs_tight", r.log_likelihood, Tol::stat(100))
            .vector(
                "nile_na_std_resid_tight",
                &r.standardized_residuals,
                state_tol(),
            )
            .finish();
    }

    #[test]
    fn test_validate_structts_fixed_changes_the_fit() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let y = case_series(&ds, "nile", 100);
        let free = fit(&y, StructTsType::Level, None, None);
        let fixed = fit(
            &y,
            StructTsType::Level,
            None,
            Some(vec![Some(1500.0), None]),
        );
        let names = ["level", "epsilon"];
        Check::new(&case)
            .named(
                "nile_coef_tight",
                &names,
                &coef_vec(&free),
                Tol::iterative(1e-6),
            )
            .named(
                "nile_fixed_coef_tight",
                &names,
                &coef_vec(&fixed),
                Tol::iterative(1e-6),
            )
            .finish();
        assert!(
            (free.coef.epsilon - fixed.coef.epsilon).abs() > 10.0,
            "{} vs {}",
            free.coef.epsilon,
            fixed.coef.epsilon
        );
        assert_eq!((free.n_params, fixed.n_params), (2, 1));
        for r in [&free, &fixed] {
            assert_eq!(r.aic, -2.0 * r.log_likelihood + 2.0 * r.n_params as f64);
        }
        assert!(free.fixed_parameters.is_empty());
    }

    #[test]
    fn test_structts_rejects_degenerate_inputs() {
        let noisy = generate_local_level_data(20, 1.0, 0.5, 3);
        let refused =
            |y: &[f64], config: StructTsConfig, needles: &[&str]| match struct_ts(y, config) {
                Err(EconError::InvalidSpecification { message }) => {
                    for needle in needles {
                        assert!(message.contains(needle), "{needle:?} not in {message}");
                    }
                }
                other => panic!("expected InvalidSpecification for {needles:?}, got {other:?}"),
            };
        let config = StructTsConfig::default;
        refused(
            &noisy,
            StructTsConfig {
                max_iter: 0,
                ..config()
            },
            &["max_iter"],
        );
        for tolerance in [0.0, -1.0, f64::NAN] {
            refused(
                &noisy,
                StructTsConfig {
                    tolerance,
                    ..config()
                },
                &["tolerance"],
            );
        }
        refused(&[3.5; 20], config(), &["constant"]);
        let mut leading = noisy.clone();
        leading[0] = f64::NAN;
        refused(&leading, config(), &["first value"]);
        let mut infinite = noisy.clone();
        infinite[7] = f64::INFINITY;
        refused(&infinite, config(), &["infinite"]);
        let fixed = |values: Vec<Option<f64>>| StructTsConfig {
            fixed: Some(values),
            ..config()
        };
        refused(&noisy, fixed(vec![None, None, None]), &["fixed", "2"]);
        refused(&noisy, fixed(vec![Some(-1.0), None]), &["fixed"]);
        refused(&noisy, fixed(vec![Some(f64::NAN), None]), &["fixed"]);
        refused(
            &noisy,
            fixed(vec![Some(1.0), Some(2.0)]),
            &["all parameters were fixed"],
        );
        // Every value missing but the first: one observation is no variance.
        let mut sparse = vec![f64::NAN; 20];
        sparse[0] = 2.0;
        assert!(matches!(
            struct_ts(&sparse, config()),
            Err(EconError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_validate_structts_scale_invariant() {
        let (case, ds) = crate::test_support::load_case("structts_nile_ukgas");
        let y = case_series(&ds, "ukgas", 108);
        let base = fit(&y, StructTsType::BSM, Some(4), None);
        assert!(base.converged);
        let want = coef_vec(&base);
        Check::new(&case)
            .named(
                "ukgas_coef_tight",
                &["level", "slope", "seas", "epsilon"],
                &want,
                Tol::iterative(1e-6),
            )
            .finish();
        for c in [1e-6, 1e6] {
            let scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
            let r = fit(&scaled, StructTsType::BSM, Some(4), None);
            assert!(r.converged, "c = {c}");
            for (got, expected) in coef_vec(&r).iter().zip(&want) {
                let unscaled = got / (c * c);
                assert!(
                    Tol::iterative(1e-6).allows(unscaled, *expected),
                    "c = {c}: {unscaled} vs {expected}"
                );
            }
            let ll = base.log_likelihood - 108.0 * f64::ln(c);
            assert!(Tol::stat(108).allows(r.log_likelihood, ll), "c = {c}");
            assert!(
                r.convergence
                    .iterations
                    .abs_diff(base.convergence.iterations)
                    <= 2,
                "c = {c}: {} vs {} iterations",
                r.convergence.iterations,
                base.convergence.iterations
            );
        }
    }

    #[test]
    fn test_structts_reports_iteration_limit() {
        let r = struct_ts(
            &repo_series("nile", 100),
            StructTsConfig {
                max_iter: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!r.converged);
        assert_eq!(r.convergence.iterations, 2);
        assert!(
            r.warnings[0].starts_with("StructTS did not converge after 2 iterations"),
            "{:?}",
            r.warnings
        );
        assert_eq!(r.notes_warnings(), r.warnings);
        assert_eq!(r.notes_convergence().unwrap().iterations, 2);
    }

    fn normal_noise(n: usize, sd: f64, seed: u64) -> Vec<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let z: f64 = StandardNormal.sample(&mut rng);
                sd * z
            })
            .collect()
    }

    #[test]
    fn test_structts_trend_recovers_dgp() {
        let noise = normal_noise(100, 1.0, 71);
        let y: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(t, e)| 10.0 + 2.0 * t as f64 + e)
            .collect();
        let r = fit(&y, StructTsType::Trend, None, None);
        assert!(r.converged, "{:?}", r.warnings);
        let slopes = col(&r.slope);
        let mean = slopes.iter().sum::<f64>() / slopes.len() as f64;
        assert!((mean - 2.0).abs() < 0.1, "{mean}");
    }

    #[test]
    fn test_structts_bsm_fitted_plus_residuals_is_data() {
        let noise = normal_noise(60, 0.5, 73);
        let y: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(t, e)| {
                let season = 8.0 * (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin();
                50.0 + 0.3 * t as f64 + season + e
            })
            .collect();
        let r = fit(&y, StructTsType::BSM, Some(12), None);
        assert!(r.converged, "{:?}", r.warnings);
        assert!(r.seasonal.is_some() && r.slope.is_some());
        for t in 0..y.len() {
            assert!(
                (r.fitted[t] + r.residuals[t] - y[t]).abs() < 1e-10,
                "t = {t}"
            );
        }
    }
}
