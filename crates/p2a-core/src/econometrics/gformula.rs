//! Parametric G-Formula for causal inference with time-varying treatments.
//!
//! The parametric g-formula (generalized formula) uses Monte Carlo simulation to
//! estimate causal effects under different treatment regimes. It is particularly
//! useful for:
//! - Time-varying treatments and confounders
//! - Longitudinal/panel data
//! - Counterfactual "what-if" scenarios
//!
//! # Algorithm
//!
//! The g-formula proceeds in three main stages:
//!
//! **Stage 1: Model Fitting**
//! 1. Fit models for time-varying confounders: L_t ~ f(L_{t-1}, A_{t-1}, baseline)
//! 2. Fit treatment model (for natural course): A_t ~ g(L_t, A_{t-1}, baseline)
//! 3. Fit outcome model: Y_t ~ h(L_t, A_t, baseline)
//!
//! **Stage 2: Monte Carlo Simulation**
//! For each simulation:
//! 1. Sample baseline values from observed data (with replacement)
//! 2. At each time point t:
//!    - Simulate L_t from fitted confounder models
//!    - Assign A_t according to the intervention (or sample from treatment model)
//!    - Compute outcome probability/expected value
//! 3. Record the final outcome
//!
//! **Stage 3: Estimation and Inference**
//! 1. Average simulated outcomes for risk under each regime
//! 2. Compute risk difference, risk ratio
//! 3. Bootstrap for standard errors and confidence intervals
//!
//! # References
//!
//! - Robins, J.M. (1986). A new approach to causal inference in mortality studies
//!   with a sustained exposure period - application to control of the healthy worker
//!   survivor effect. *Mathematical Modelling*, 7(9-12), 1393-1512.
//!   https://doi.org/10.1016/0270-0255(86)90088-6
//!
//! - Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*.
//!   Chapman & Hall/CRC. Chapter 21. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
//!
//! - McGrath, S., et al. (2020). gfoRmula: An R Package for Estimating the Effects
//!   of Sustained Treatment Strategies via the Parametric g-formula.
//!   *Patterns*, 1(3), 100008. https://doi.org/10.1016/j.patter.2020.100008
//!
//! - Keil, A.P., et al. (2014). The parametric g-formula for time-to-event data:
//!   intuition and a worked example. *Epidemiology*, 25(6), 889-897.
//!   https://doi.org/10.1097/EDE.0000000000000160
//!
//! R equivalent: `gfoRmula::gformula()`, `gfoRmula::gformula_binary_eof()`

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use rand_distr::{Distribution, StandardNormal};

use crate::econometrics::glm::{GlmSpec, fit_glm, plogis};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource::{self, Budget};
use crate::traits::estimator::{SignificanceLevel, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of intervention in the g-formula.
#[derive(Clone, Serialize, Deserialize, Default)]
pub enum GFormulaIntervention {
    /// Natural course: observe what happens under actual treatment patterns
    #[default]
    NaturalCourse,

    /// Static intervention: always treat (true) or never treat (false)
    Static {
        /// Whether to always treat (true) or never treat (false)
        treat_all: bool,
    },

    /// Threshold-based dynamic intervention
    /// Treat if the specified variable crosses a threshold
    Threshold {
        /// Index of the covariate to check (0-indexed into time-varying covariates)
        variable_idx: usize,
        /// Threshold value
        cutoff: f64,
        /// Treat if above threshold (true) or below (false)
        above: bool,
    },
}

impl fmt::Debug for GFormulaIntervention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaturalCourse => write!(f, "NaturalCourse"),
            Self::Static { treat_all } => f
                .debug_struct("Static")
                .field("treat_all", treat_all)
                .finish(),
            Self::Threshold {
                variable_idx,
                cutoff,
                above,
            } => f
                .debug_struct("Threshold")
                .field("variable_idx", variable_idx)
                .field("cutoff", cutoff)
                .field("above", above)
                .finish(),
        }
    }
}

/// Outcome type for the g-formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GFormulaOutcomeType {
    /// Continuous outcome (linear model)
    #[default]
    Continuous,
    /// Binary outcome (logistic model)
    Binary,
    /// Survival/time-to-event (hazard model)
    Survival,
}

impl fmt::Display for GFormulaOutcomeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Continuous => write!(f, "Continuous (Linear)"),
            Self::Binary => write!(f, "Binary (Logistic)"),
            Self::Survival => write!(f, "Survival (Discrete Hazard)"),
        }
    }
}

crate::option_enum!(
    GFormulaOutcomeType,
    crate::options::Choices {
        option: "outcome_type",
        values: &[
            (
                &["continuous", "linear", "gaussian"],
                GFormulaOutcomeType::Continuous
            ),
            (
                &["binary", "logistic", "binomial"],
                GFormulaOutcomeType::Binary
            ),
            (
                &["survival", "hazard", "discrete_hazard", "pooled_logistic"],
                GFormulaOutcomeType::Survival
            ),
        ],
    },
    no_display
);

/// Intervention kinds as option strings (MCP, CLI); turned into a
/// [`GFormulaIntervention`] by [`GFormulaIntervention::from_regime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GFormulaRegime {
    /// Treat every subject in every period.
    AlwaysTreat,
    /// Treat no subject in any period.
    NeverTreat,
    /// Treatment drawn from the fitted treatment model.
    NaturalCourse,
    /// Treat when a time-varying covariate crosses a cutoff.
    Threshold,
}

crate::option_enum!(
    GFormulaRegime,
    crate::options::Choices {
        option: "intervention",
        values: &[
            (
                &["always_treat", "always", "treat_all"],
                GFormulaRegime::AlwaysTreat
            ),
            (
                &["never_treat", "never", "treat_none", "control"],
                GFormulaRegime::NeverTreat
            ),
            (
                &["natural_course", "natural", "observed"],
                GFormulaRegime::NaturalCourse
            ),
            (&["threshold", "dynamic"], GFormulaRegime::Threshold),
        ],
    }
);

impl GFormulaIntervention {
    /// The intervention of `regime`. `Threshold` needs `threshold_variable` (0-based index
    /// into the time-varying covariates) and a finite `threshold_cutoff`;
    /// `threshold_above` defaults to true. Threshold fields with another regime are an
    /// error.
    pub fn from_regime(
        regime: GFormulaRegime,
        threshold_variable: Option<usize>,
        threshold_cutoff: Option<f64>,
        threshold_above: Option<bool>,
    ) -> EconResult<Self> {
        let any_threshold_field =
            threshold_variable.is_some() || threshold_cutoff.is_some() || threshold_above.is_some();
        match regime {
            GFormulaRegime::Threshold => {
                let variable_idx = threshold_variable.ok_or_else(|| {
                    spec_err("threshold_variable is required for intervention 'threshold'".into())
                })?;
                let cutoff = threshold_cutoff.ok_or_else(|| {
                    spec_err("threshold_cutoff is required for intervention 'threshold'".into())
                })?;
                check_cutoff(cutoff)?;
                Ok(Self::Threshold {
                    variable_idx,
                    cutoff,
                    above: threshold_above.unwrap_or(true),
                })
            }
            _ if any_threshold_field => Err(spec_err(
                "threshold_variable, threshold_cutoff and threshold_above apply only to \
                 intervention 'threshold'"
                    .into(),
            )),
            GFormulaRegime::AlwaysTreat => Ok(Self::Static { treat_all: true }),
            GFormulaRegime::NeverTreat => Ok(Self::Static { treat_all: false }),
            GFormulaRegime::NaturalCourse => Ok(Self::NaturalCourse),
        }
    }
}

fn check_cutoff(cutoff: f64) -> EconResult<()> {
    if cutoff.is_finite() {
        Ok(())
    } else {
        Err(spec_err(format!(
            "threshold cutoff must be finite, got {cutoff}"
        )))
    }
}

/// Configuration for the parametric g-formula.
#[derive(Debug, Clone)]
pub struct GFormulaConfig {
    /// Number of Monte Carlo simulations (at least 1; a count whose outcome buffer
    /// cannot be allocated is refused with `EconError::ResourceLimit`)
    pub n_simulations: usize,
    /// Number of time points in the analysis
    pub time_points: usize,
    /// Intervention to evaluate
    pub intervention: GFormulaIntervention,
    /// Type of outcome variable
    pub outcome_type: GFormulaOutcomeType,
    /// Number of bootstrap samples for standard errors: 0 (no bootstrap; the standard
    /// errors are NaN) or at least 2. A count whose result buffer cannot be allocated is
    /// refused with `EconError::ResourceLimit`.
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Maximum iterations for logistic regression (at least 1)
    pub max_iter: usize,
    /// Convergence tolerance (finite and > 0)
    pub tolerance: f64,
}

impl Default for GFormulaConfig {
    fn default() -> Self {
        Self {
            n_simulations: 1000,
            time_points: 2,
            intervention: GFormulaIntervention::Static { treat_all: true },
            outcome_type: GFormulaOutcomeType::Binary,
            n_bootstrap: 200,
            seed: Some(42),
            confidence_level: 0.95,
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// Discrete-time survival outcome, one entry per subject.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvivalOutcome {
    /// Last period the subject is observed in, 1-based (1..=T).
    pub time: Vec<usize>,
    /// true: the event occurred in period `time`; false: event-free through `time` (lost to
    /// follow-up after `time`, or event-free at the end of follow-up when `time == T`).
    pub event: Vec<bool>,
}

impl SurvivalOutcome {
    /// Number of subjects with an event.
    pub fn n_events(&self) -> usize {
        self.event.iter().filter(|&&e| e).count()
    }
}

/// Input data for the parametric g-formula.
///
/// Subjects are rows; covariates and treatments are given per period (wide layout).
/// End-of-follow-up data ([`GFormulaData::new`]) observe every subject in every period and
/// the outcome at the end of follow-up. Survival data ([`GFormulaData::new_survival`]) carry
/// each subject's last observed period and event indicator; cells of later periods are
/// not used.
#[derive(Debug, Clone)]
pub struct GFormulaData {
    /// Baseline (time-invariant) covariates: n_subjects x n_baseline_vars
    pub baseline_covariates: Array2<f64>,

    /// Time-varying covariates: one matrix per time point
    /// Each matrix is n_subjects x n_time_varying_vars
    pub time_varying_covariates: Vec<Array2<f64>>,

    /// Treatment at each time point: one vector per time point, coded 0/1
    pub treatments: Vec<Array1<f64>>,

    /// End-of-follow-up outcome: n_subjects x 1 (binary: 0/1; continuous: any finite
    /// value); for survival data the event indicator (0/1).
    pub outcome: Array1<f64>,

    /// Follow-up times and event indicators; `Some` for data built with
    /// [`GFormulaData::new_survival`].
    pub survival: Option<SurvivalOutcome>,

    /// Names of the baseline covariates (model tables, warnings); default `B1`, `B2`, ...
    pub baseline_names: Vec<String>,
    /// Names of the time-varying covariates; default `L1`, `L2`, ...
    pub time_varying_names: Vec<String>,
    /// Name of the treatment; default `A`.
    pub treatment_name: String,
}

fn spec_err(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn no_subjects() -> EconError {
    EconError::InsufficientData {
        required: 1,
        provided: 0,
        context: "g-formula data have no subjects (rows)".into(),
    }
}

/// Row and period counts of every input agree (also run on data built by literal).
fn check_shapes(
    baseline_covariates: &Array2<f64>,
    time_varying_covariates: &[Array2<f64>],
    treatments: &[Array1<f64>],
    outcome_len: usize,
) -> EconResult<()> {
    let n = baseline_covariates.nrows();
    if outcome_len != n {
        return Err(spec_err(format!(
            "Outcome length ({outcome_len}) does not match number of subjects ({n})"
        )));
    }
    for (t, tvs) in time_varying_covariates.iter().enumerate() {
        if tvs.nrows() != n {
            return Err(spec_err(format!(
                "Time-varying covariates at t={} have {} rows, expected {}",
                t,
                tvs.nrows(),
                n
            )));
        }
    }
    for (t, a) in treatments.iter().enumerate() {
        if a.len() != n {
            return Err(spec_err(format!(
                "Treatment at t={} has length {}, expected {}",
                t,
                a.len(),
                n
            )));
        }
    }
    if treatments.len() != time_varying_covariates.len() {
        return Err(spec_err(format!(
            "Number of treatment time points ({}) does not match \
             number of time-varying covariate time points ({})",
            treatments.len(),
            time_varying_covariates.len()
        )));
    }
    if n == 0 {
        return Err(no_subjects());
    }
    if let Some(first) = time_varying_covariates.first() {
        let k0 = first.ncols();
        for (t, tvs) in time_varying_covariates.iter().enumerate().skip(1) {
            if tvs.ncols() != k0 {
                return Err(spec_err(format!(
                    "g-formula data: time-varying covariates in period {} have {} columns, \
                     period 1 has {k0}",
                    t + 1,
                    tvs.ncols()
                )));
            }
        }
    }
    Ok(())
}

fn default_names(prefix: &str, k: usize) -> Vec<String> {
    (1..=k).map(|j| format!("{prefix}{j}")).collect()
}

/// Number of periods a subject is followed through: `time` for survival data, every
/// period otherwise.
fn followed_through(survival: Option<&SurvivalOutcome>, i: usize, t_max: usize) -> usize {
    survival.map_or(t_max, |s| s.time[i])
}

/// Finite covariate and outcome cells and 0/1 treatments. For survival data only the
/// periods a subject is followed through are read (`survival` gives them).
fn check_cells(data: &GFormulaData) -> EconResult<()> {
    let survival = data.survival.as_ref();
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let surv_note = |i: usize| match survival {
        Some(s) => format!(" (the subject is followed through period {})", s.time[i]),
        None => String::new(),
    };
    let prefix = if survival.is_some() {
        "g-formula survival data"
    } else {
        "g-formula data"
    };
    for (j, col) in data.baseline_covariates.columns().into_iter().enumerate() {
        if let Some(i) = col.iter().position(|v| !v.is_finite()) {
            return Err(spec_err(format!(
                "g-formula data: baseline covariate {} ({}) is not finite in row {}",
                j + 1,
                data.baseline_names[j],
                i + 1
            )));
        }
    }
    for t in 0..t_max {
        let l = &data.time_varying_covariates[t];
        for i in 0..n {
            if t >= followed_through(survival, i, t_max) {
                continue;
            }
            for j in 0..l.ncols() {
                if !l[[i, j]].is_finite() {
                    return Err(spec_err(format!(
                        "{prefix}: time-varying covariate {} ({}) is not finite in row {}, \
                         period {}{}",
                        j + 1,
                        data.time_varying_names[j],
                        i + 1,
                        t + 1,
                        surv_note(i)
                    )));
                }
            }
            let a = data.treatments[t][i];
            if a != 0.0 && a != 1.0 {
                return Err(spec_err(format!(
                    "{prefix}: the treatment in period {} must be 0 or 1; row {} has {a}{}",
                    t + 1,
                    i + 1,
                    surv_note(i)
                )));
            }
        }
    }
    if survival.is_none()
        && let Some(i) = data.outcome.iter().position(|v| !v.is_finite())
    {
        return Err(spec_err(format!(
            "g-formula data: the outcome is not finite in row {}",
            i + 1
        )));
    }
    Ok(())
}

impl GFormulaData {
    /// Create end-of-follow-up g-formula data.
    ///
    /// Refuses mismatched dimensions, zero subjects (`InsufficientData`), time-varying
    /// matrices of unequal width, non-finite covariate or outcome cells and treatments
    /// other than 0/1 (`InvalidSpecification` naming the column, row and period, 1-based).
    pub fn new(
        baseline_covariates: Array2<f64>,
        time_varying_covariates: Vec<Array2<f64>>,
        treatments: Vec<Array1<f64>>,
        outcome: Array1<f64>,
    ) -> EconResult<Self> {
        check_shapes(
            &baseline_covariates,
            &time_varying_covariates,
            &treatments,
            outcome.len(),
        )?;
        let data = Self::assemble(
            baseline_covariates,
            time_varying_covariates,
            treatments,
            outcome,
            None,
        );
        check_cells(&data)?;
        Ok(data)
    }

    fn assemble(
        baseline_covariates: Array2<f64>,
        time_varying_covariates: Vec<Array2<f64>>,
        treatments: Vec<Array1<f64>>,
        outcome: Array1<f64>,
        survival: Option<SurvivalOutcome>,
    ) -> Self {
        let n_tv = time_varying_covariates.first().map_or(0, |m| m.ncols());
        Self {
            baseline_names: default_names("B", baseline_covariates.ncols()),
            time_varying_names: default_names("L", n_tv),
            treatment_name: "A".to_string(),
            baseline_covariates,
            time_varying_covariates,
            treatments,
            outcome,
            survival,
        }
    }

    /// Names for reports. Lengths must match the data; names must be non-empty and
    /// distinct, and none may be `(Intercept)` or `t` (reserved by the hazard model).
    pub fn with_names(
        mut self,
        baseline: &[&str],
        time_varying: &[&str],
        treatment: &str,
    ) -> EconResult<Self> {
        if baseline.len() != self.n_baseline_covars() {
            return Err(spec_err(format!(
                "g-formula data: expected {} baseline names, got {}",
                self.n_baseline_covars(),
                baseline.len()
            )));
        }
        if time_varying.len() != self.n_time_varying_covars() {
            return Err(spec_err(format!(
                "g-formula data: expected {} time-varying names, got {}",
                self.n_time_varying_covars(),
                time_varying.len()
            )));
        }
        let all: Vec<&str> = baseline
            .iter()
            .chain(time_varying.iter())
            .copied()
            .chain(std::iter::once(treatment))
            .collect();
        if all.iter().any(|s| s.trim().is_empty()) {
            return Err(spec_err("g-formula data: names must not be empty".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for s in &all {
            if *s == "(Intercept)" || *s == "t" || !seen.insert(*s) {
                return Err(spec_err(format!(
                    "g-formula data: duplicate or reserved name '{s}' \
                     ('(Intercept)' and 't' are reserved)"
                )));
            }
        }
        self.baseline_names = baseline.iter().map(|s| s.to_string()).collect();
        self.time_varying_names = time_varying.iter().map(|s| s.to_string()).collect();
        self.treatment_name = treatment.to_string();
        Ok(self)
    }

    /// Whether the data hold a survival outcome (built with [`GFormulaData::new_survival`]).
    pub fn is_survival(&self) -> bool {
        self.survival.is_some()
    }

    /// Get the number of subjects.
    pub fn n_subjects(&self) -> usize {
        self.baseline_covariates.nrows()
    }

    /// Get the number of time points.
    pub fn n_time_points(&self) -> usize {
        self.treatments.len()
    }

    /// Get the number of baseline covariates.
    pub fn n_baseline_covars(&self) -> usize {
        self.baseline_covariates.ncols()
    }

    /// Get the number of time-varying covariates.
    pub fn n_time_varying_covars(&self) -> usize {
        if self.time_varying_covariates.is_empty() {
            0
        } else {
            self.time_varying_covariates[0].ncols()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// How a regime's risk was standardised over the baseline covariates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GFormulaStandardization {
    /// `n_simulations` simulated subjects, baseline rows drawn with replacement.
    #[default]
    MonteCarlo,
    /// Each observed baseline row once (a static regime without time-varying covariates:
    /// nothing in the simulation is random).
    Empirical,
}

/// The fitted end-of-follow-up outcome model or discrete-time hazard, as reported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GFormulaModelSummary {
    /// "linear", "logistic" or "pooled logistic hazard".
    pub kind: String,
    /// Column names, starting with "(Intercept)".
    pub names: Vec<String>,
    /// Coefficients; `NaN` for aliased columns.
    pub coefficients: Vec<f64>,
    /// Standard errors; `NaN` for aliased columns.
    pub std_errors: Vec<f64>,
    /// Observations: subjects, or person-periods for the hazard.
    pub n_obs: usize,
    /// The fit converged (always true for "linear").
    pub converged: bool,
    /// IRLS iterations (0 for "linear").
    pub iterations: usize,
}

fn nan() -> f64 {
    f64::NAN
}

/// Result from the parametric g-formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GFormulaResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Main Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Risk (probability of outcome) under natural course
    pub risk_natural: f64,
    /// Risk under the specified intervention
    pub risk_intervention: f64,
    /// Causal risk difference: P(Y=1|intervention) - P(Y=1|natural)
    pub risk_difference: f64,
    /// Causal risk ratio: P(Y=1|intervention) / P(Y=1|natural)
    pub risk_ratio: f64,
    /// Odds ratio (for binary outcomes)
    pub odds_ratio: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Standard Errors and Confidence Intervals
    // ═══════════════════════════════════════════════════════════════════════
    /// Standard error of risk difference (from bootstrap)
    pub se_risk_difference: f64,
    /// Standard error of log(risk ratio) (from bootstrap)
    pub se_log_risk_ratio: f64,

    /// Lower bound of CI for risk difference
    pub ci_lower_rd: f64,
    /// Upper bound of CI for risk difference
    pub ci_upper_rd: f64,
    /// Lower bound of CI for risk ratio
    pub ci_lower_rr: f64,
    /// Upper bound of CI for risk ratio
    pub ci_upper_rr: f64,

    /// P-value for risk difference (H0: RD = 0)
    pub p_value_rd: f64,
    /// Significance level for risk difference
    pub significance_rd: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Simulation Details
    // ═══════════════════════════════════════════════════════════════════════
    /// Number of Monte Carlo simulations performed
    pub n_simulations: usize,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Number of time points in the analysis
    pub n_time_points: usize,
    /// Number of subjects in the original data
    pub n_subjects: usize,
    /// Confidence level used
    pub confidence_level: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Diagnostics
    // ═══════════════════════════════════════════════════════════════════════
    /// Type of outcome model used
    pub outcome_type: GFormulaOutcomeType,
    /// Intervention description
    pub intervention_description: String,
    /// Whether all models converged
    pub all_models_converged: bool,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════
    // Detailed Output (optional, for diagnostics)
    // ═══════════════════════════════════════════════════════════════════════
    /// Simulated outcomes under natural course (for diagnostics)
    #[serde(skip)]
    pub simulated_natural: Vec<f64>,
    /// Simulated outcomes under intervention (for diagnostics)
    #[serde(skip)]
    pub simulated_intervention: Vec<f64>,
    /// Bootstrap estimates of risk difference
    #[serde(skip)]
    pub bootstrap_rd: Vec<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Survival curves, observed risk, models
    // ═══════════════════════════════════════════════════════════════════════
    /// Survival outcome: risk `1 - S(t)` by period `t = 1..T` under the natural course;
    /// empty for end-of-follow-up outcomes.
    #[serde(default)]
    pub risk_curve_natural: Vec<f64>,
    /// Survival outcome: risk by period under the intervention; empty otherwise.
    #[serde(default)]
    pub risk_curve_intervention: Vec<f64>,
    /// Observed risk: the mean outcome, or for survival data the discrete-time
    /// Kaplan-Meier risk at the last period.
    #[serde(default = "nan")]
    pub observed_risk: f64,
    /// Survival data: discrete-time Kaplan-Meier risk `1 - S(t)` by period; empty otherwise.
    #[serde(default)]
    pub observed_risk_curve: Vec<f64>,
    /// How the intervention's risk was standardised (the natural course is always
    /// Monte Carlo).
    #[serde(default)]
    pub standardization: GFormulaStandardization,
    /// The treatment never changes within a subject's follow-up and was modelled as a
    /// point treatment (decided in period 1, carried forward).
    #[serde(default)]
    pub point_treatment: bool,
    /// The fitted outcome model (end of follow-up) or pooled logistic hazard.
    #[serde(default)]
    pub outcome_model: Option<GFormulaModelSummary>,
}

impl fmt::Display for GFormulaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Parametric G-Formula Results")?;
        writeln!(f, "============================")?;
        writeln!(f)?;
        writeln!(f, "Intervention: {}", self.intervention_description)?;
        writeln!(f, "Outcome Type: {}", self.outcome_type)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Subjects:      {}", self.n_subjects)?;
        writeln!(f, "  Time Points:   {}", self.n_time_points)?;
        writeln!(f, "  Simulations:   {}", self.n_simulations)?;
        writeln!(f, "  Bootstrap:     {}", self.n_bootstrap)?;
        writeln!(f)?;
        writeln!(f, "Risks:")?;
        writeln!(f, "  Natural Course:  {:.4}", self.risk_natural)?;
        writeln!(f, "  Intervention:    {:.4}", self.risk_intervention)?;
        writeln!(f)?;
        writeln!(f, "Causal Effects:")?;
        writeln!(
            f,
            "  Risk Difference: {:.4} (SE: {:.4})",
            self.risk_difference, self.se_risk_difference
        )?;
        writeln!(
            f,
            "    {:.0}% CI: [{:.4}, {:.4}]",
            self.confidence_level * 100.0,
            self.ci_lower_rd,
            self.ci_upper_rd
        )?;
        writeln!(
            f,
            "    p-value: {:.4}{}",
            self.p_value_rd,
            self.significance_rd.stars()
        )?;
        writeln!(f)?;
        writeln!(f, "  Risk Ratio:      {:.4}", self.risk_ratio)?;
        writeln!(
            f,
            "    {:.0}% CI: [{:.4}, {:.4}]",
            self.confidence_level * 100.0,
            self.ci_lower_rr,
            self.ci_upper_rr
        )?;

        if self.odds_ratio.is_finite() && self.odds_ratio > 0.0 {
            writeln!(f)?;
            writeln!(f, "  Odds Ratio:      {:.4}", self.odds_ratio)?;
        }

        if !self.all_models_converged {
            writeln!(f)?;
            writeln!(f, "Warning: Some models did not converge")?;
        }

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Model Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// A fitted covariate, treatment, outcome or hazard model, ready for prediction.
#[derive(Debug, Clone)]
struct PredictModel {
    /// Coefficients as fitted, `NaN` for aliased columns (reports, tests).
    fitted: Vec<f64>,
    /// `fitted` with `NaN` replaced by 0: R's `predict` on a rank-deficient fit uses the
    /// identified columns only.
    coef: Vec<f64>,
    std_errors: Vec<f64>,
    names: Vec<String>,
    /// Residual SD `sqrt(RSS / df_residual)` of a linear model (`sigma(lm)`); 1 for logistic.
    sigma: f64,
    /// Logistic (Bernoulli draws) or linear (Normal draws).
    binary: bool,
    n_obs: usize,
    converged: bool,
    iterations: usize,
}

impl PredictModel {
    /// Linear predictor of a design row.
    fn eta(&self, x: &[f64]) -> f64 {
        x.iter().zip(&self.coef).map(|(a, b)| a * b).sum()
    }

    fn summary(&self, kind: &str) -> GFormulaModelSummary {
        GFormulaModelSummary {
            kind: kind.to_string(),
            names: self.names.clone(),
            coefficients: self.fitted.clone(),
            std_errors: self.std_errors.clone(),
            n_obs: self.n_obs,
            converged: self.converged,
            iterations: self.iterations,
        }
    }
}

/// Every model of one g-formula fit.
#[derive(Debug, Clone)]
struct FittedModels {
    /// `[period][covariate]`.
    covariates: Vec<Vec<PredictModel>>,
    /// One per period, or one (period 1) under a point treatment.
    treatment: Vec<PredictModel>,
    /// End-of-follow-up outcome model, or the pooled logistic hazard for survival.
    outcome: PredictModel,
    point_treatment: bool,
    /// Subjects at risk at the start of each period.
    n_at_risk: Vec<usize>,
    /// Every logistic fit converged.
    all_converged: bool,
}

/// One regime's simulation.
#[derive(Debug, Clone)]
struct SimOutput {
    /// Per simulated (or empirical) unit: the outcome's expectation, or the risk by the
    /// end of follow-up for survival.
    per_unit: Vec<f64>,
    mean: f64,
    /// Survival: mean risk by period; empty otherwise.
    curve: Vec<f64>,
}

/// The units a regime is standardised over.
#[derive(Debug, Clone, Copy)]
enum Units {
    /// Each observed baseline row once, in order.
    Empirical,
    /// This many units, baseline rows drawn with replacement.
    MonteCarlo(usize),
}

/// The fitted models and both regimes' simulations.
struct Estimate {
    models: FittedModels,
    natural: SimOutput,
    intervention: SimOutput,
    standardization: GFormulaStandardization,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the parametric g-formula.
///
/// Fits the covariate, treatment and outcome (or hazard) models, simulates the natural
/// course and the intervention from them (see the module documentation for the model
/// specification and the simulation), and bootstraps subjects for standard errors and
/// confidence intervals.
///
/// # Errors
/// `InvalidSpecification` for invalid controls, a threshold on a covariate the data do not
/// have, or an outcome type that does not match the data; `InsufficientData` for data
/// without subjects or a period nobody is at risk in; `ResourceLimit` and `Cancelled` from
/// the resource guards.
///
/// # References
///
/// - Robins (1986), Mathematical Modelling
/// - Hernan & Robins (2020), Causal Inference: What If, Chapters 17 and 21
/// - McGrath et al. (2020), Patterns 1(3), 100008
pub fn run_gformula(data: &GFormulaData, config: GFormulaConfig) -> EconResult<GFormulaResult> {
    validate_inputs(data, &config)?;
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let mut warnings = Vec::new();

    if n < 10 {
        warnings.push(format!(
            "Small sample size (n={}). Results may be unreliable.",
            n
        ));
    }

    let seed = config.seed.unwrap_or(42);

    // Models and both regimes' simulations (common random numbers).
    let est = estimate_once(data, &config, config.n_simulations, seed, &mut warnings)?;
    let risk_natural = est.natural.mean;
    let risk_intervention = est.intervention.mean;

    let risk_difference = risk_intervention - risk_natural;
    let risk_ratio = if risk_natural > 0.0 {
        risk_intervention / risk_natural
    } else {
        f64::INFINITY
    };

    let odds_ratio = if risk_natural > 0.0
        && risk_natural < 1.0
        && risk_intervention > 0.0
        && risk_intervention < 1.0
    {
        let odds_int = risk_intervention / (1.0 - risk_intervention);
        let odds_nat = risk_natural / (1.0 - risk_natural);
        odds_int / odds_nat
    } else {
        f64::NAN
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Bootstrap for Standard Errors
    // ═══════════════════════════════════════════════════════════════════════

    let bootstrap_results = bootstrap_gformula(data, &config, config.n_bootstrap, seed)?;

    let (se_rd, ci_lower_rd, ci_upper_rd) =
        compute_bootstrap_ci(&bootstrap_results, config.confidence_level);

    // SE for log(RR)
    let bootstrap_log_rr: Vec<f64> = bootstrap_results
        .iter()
        .map(|(_, int, nat)| {
            if *nat > 0.0 && *int > 0.0 {
                (int / nat).ln()
            } else {
                f64::NAN
            }
        })
        .filter(|x| x.is_finite())
        .collect();

    let se_log_rr = if bootstrap_log_rr.len() > 1 {
        let mean: f64 = bootstrap_log_rr.iter().sum::<f64>() / bootstrap_log_rr.len() as f64;
        let var: f64 = bootstrap_log_rr
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (bootstrap_log_rr.len() - 1) as f64;
        var.sqrt()
    } else {
        f64::NAN
    };

    // CI for RR (on log scale, then exponentiate)
    let z_crit = compute_z_critical(config.confidence_level);
    let log_rr = if risk_ratio > 0.0 {
        risk_ratio.ln()
    } else {
        f64::NAN
    };
    let ci_lower_rr = (log_rr - z_crit * se_log_rr).exp();
    let ci_upper_rr = (log_rr + z_crit * se_log_rr).exp();

    // P-value for RD (two-sided, normal approximation)
    let z_stat = if se_rd > 0.0 && se_rd.is_finite() {
        risk_difference / se_rd
    } else {
        0.0
    };
    let p_value_rd = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    let significance_rd = SignificanceLevel::from_p_value(p_value_rd);

    let intervention_description = describe(&config.intervention, data);
    let (observed_risk, observed_risk_curve) = observed_risk(data);

    Ok(GFormulaResult {
        risk_natural,
        risk_intervention,
        risk_difference,
        risk_ratio,
        odds_ratio,
        se_risk_difference: se_rd,
        se_log_risk_ratio: se_log_rr,
        ci_lower_rd,
        ci_upper_rd,
        ci_lower_rr,
        ci_upper_rr,
        p_value_rd,
        significance_rd,
        n_simulations: config.n_simulations,
        n_bootstrap: config.n_bootstrap,
        n_time_points: t_max,
        n_subjects: n,
        confidence_level: config.confidence_level,
        outcome_type: config.outcome_type,
        intervention_description,
        all_models_converged: est.models.all_converged,
        warnings,
        simulated_natural: est.natural.per_unit,
        simulated_intervention: est.intervention.per_unit,
        bootstrap_rd: bootstrap_results.iter().map(|(rd, _, _)| *rd).collect(),
        risk_curve_natural: est.natural.curve,
        risk_curve_intervention: est.intervention.curve,
        observed_risk,
        observed_risk_curve,
        standardization: est.standardization,
        point_treatment: est.models.point_treatment,
        outcome_model: Some(
            est.models
                .outcome
                .summary(outcome_kind(config.outcome_type)),
        ),
    })
}

/// The reported kind of the outcome model.
fn outcome_kind(outcome_type: GFormulaOutcomeType) -> &'static str {
    match outcome_type {
        GFormulaOutcomeType::Continuous => "linear",
        GFormulaOutcomeType::Binary => "logistic",
        GFormulaOutcomeType::Survival => "pooled logistic hazard",
    }
}

/// The observed risk: the mean outcome, or for survival data the Kaplan-Meier risk at
/// the last period with its curve.
fn observed_risk(data: &GFormulaData) -> (f64, Vec<f64>) {
    match &data.survival {
        Some(s) => {
            let curve = kaplan_meier_risk(s, data.n_time_points());
            (curve.last().copied().unwrap_or(f64::NAN), curve)
        }
        None => (
            data.outcome.iter().sum::<f64>() / data.n_subjects() as f64,
            Vec::new(),
        ),
    }
}

/// Discrete-time Kaplan-Meier risk `1 - prod_{s <= t} (1 - d_s / n_s)` at `t = 1..t_max`,
/// with `n_s = #{time >= s}` and `d_s = #{time == s, event}`.
fn kaplan_meier_risk(s: &SurvivalOutcome, t_max: usize) -> Vec<f64> {
    let mut at_risk = vec![0usize; t_max + 1];
    let mut events = vec![0usize; t_max + 1];
    for (&t, &e) in s.time.iter().zip(&s.event) {
        at_risk[t] += 1;
        if e {
            events[t] += 1;
        }
    }
    // n_s = subjects whose time is >= s
    let mut n_s = s.time.len();
    let mut surv = 1.0;
    let mut out = Vec::with_capacity(t_max);
    for t in 1..=t_max {
        if n_s > 0 {
            surv *= 1.0 - events[t] as f64 / n_s as f64;
        }
        out.push(1.0 - surv);
        n_s -= at_risk[t];
    }
    out
}

/// Fit the models and simulate the natural course and the intervention from one seeded
/// stream (common random numbers). The intervention is standardised exactly over the
/// observed baseline rows when it is static and there is no time-varying covariate.
fn estimate_once(
    data: &GFormulaData,
    config: &GFormulaConfig,
    n_sim: usize,
    seed: u64,
    warnings: &mut Vec<String>,
) -> EconResult<Estimate> {
    let models = fit_models(data, config, warnings)?;
    let natural = simulate(
        data,
        &models,
        &GFormulaIntervention::NaturalCourse,
        Units::MonteCarlo(n_sim),
        config.outcome_type,
        seed,
    )?;
    let exact = data.n_time_varying_covars() == 0
        && matches!(config.intervention, GFormulaIntervention::Static { .. });
    let (units, standardization) = if exact {
        (Units::Empirical, GFormulaStandardization::Empirical)
    } else {
        (
            Units::MonteCarlo(n_sim),
            GFormulaStandardization::MonteCarlo,
        )
    };
    let intervention = simulate(
        data,
        &models,
        &config.intervention,
        units,
        config.outcome_type,
        seed,
    )?;
    Ok(Estimate {
        models,
        natural,
        intervention,
        standardization,
    })
}

/// Configuration and data checks run before any model is fitted.
fn validate_inputs(data: &GFormulaData, config: &GFormulaConfig) -> EconResult<()> {
    if config.n_simulations == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Number of simulations must be at least 1".to_string(),
        });
    }
    if config.n_bootstrap == 1 {
        return Err(EconError::InvalidSpecification {
            message: "n_bootstrap must be 0 (no bootstrap) or at least 2; got 1".to_string(),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1; got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be finite and > 0; got {}", config.tolerance),
        });
    }
    if data.n_time_points() == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Data must have at least one time point".to_string(),
        });
    }
    // A literal bypasses the constructors: re-check the shapes (cheap) and the subject count.
    check_shapes(
        &data.baseline_covariates,
        &data.time_varying_covariates,
        &data.treatments,
        data.outcome.len(),
    )?;
    if data.baseline_names.len() != data.n_baseline_covars()
        || data.time_varying_names.len() != data.n_time_varying_covars()
    {
        return Err(spec_err(format!(
            "g-formula data: {} baseline and {} time-varying names for {} and {} covariates",
            data.baseline_names.len(),
            data.time_varying_names.len(),
            data.n_baseline_covars(),
            data.n_time_varying_covars()
        )));
    }
    if let Some(s) = &data.survival
        && (s.time.len() != data.n_subjects() || s.event.len() != data.n_subjects())
    {
        return Err(spec_err(format!(
            "g-formula survival data: {} times and {} events for {} subjects",
            s.time.len(),
            s.event.len(),
            data.n_subjects()
        )));
    }
    if let GFormulaIntervention::Threshold {
        variable_idx,
        cutoff,
        ..
    } = &config.intervention
    {
        let n_tv = data.n_time_varying_covars();
        if n_tv == 0 {
            return Err(spec_err(
                "threshold intervention needs a time-varying covariate; the data have no \
                 time-varying covariate"
                    .into(),
            ));
        }
        if *variable_idx >= n_tv {
            let names: Vec<String> = data
                .time_varying_names
                .iter()
                .enumerate()
                .map(|(j, name)| format!("{name} = index {j}"))
                .collect();
            return Err(spec_err(format!(
                "threshold intervention: covariate index {variable_idx} is out of range; the \
                 data have {n_tv} time-varying covariate(s): {}",
                names.join(", ")
            )));
        }
        check_cutoff(*cutoff)?;
    }
    Ok(())
}

/// The text of `intervention` in the result, naming the threshold covariate.
fn describe(intervention: &GFormulaIntervention, data: &GFormulaData) -> String {
    match intervention {
        GFormulaIntervention::NaturalCourse => {
            "Natural course (observed treatment patterns)".to_string()
        }
        GFormulaIntervention::Static { treat_all: true } => {
            "Always treat (A=1 at all time points)".to_string()
        }
        GFormulaIntervention::Static { treat_all: false } => {
            "Never treat (A=0 at all time points)".to_string()
        }
        GFormulaIntervention::Threshold {
            variable_idx,
            cutoff,
            above,
        } => format!(
            "Treat if {} {} {}",
            data.time_varying_names[*variable_idx],
            if *above { ">" } else { "<=" },
            cutoff
        ),
    }
}

/// Convenience function with default configuration.
///
/// Uses static "always treat" intervention by default.
pub fn gformula(data: &GFormulaData) -> EconResult<GFormulaResult> {
    run_gformula(data, GFormulaConfig::default())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Model Fitting
// ═══════════════════════════════════════════════════════════════════════════════

/// The note pushed when rule P applies to data with more than one period.
const POINT_TREATMENT_NOTE: &str = "the treatment never changes within a subject's follow-up: \
     it is modelled as a point treatment decided in period 1 and carried forward under every \
     regime";

/// Covariate model row of period `t` (0-based): `[1, baseline, L_(t-1), A_(t-1)]`, the lags
/// only for `t > 0`.
fn cov_row(buf: &mut Vec<f64>, t: usize, b: &[f64], l_prev: &[f64], a_prev: f64) {
    buf.clear();
    buf.push(1.0);
    buf.extend_from_slice(b);
    if t > 0 {
        buf.extend_from_slice(l_prev);
        buf.push(a_prev);
    }
}

/// Treatment model row of period `t`: `[1, baseline, L_t, A_(t-1)]`, the lag only for
/// `t > 0` and not under a point treatment.
fn trt_row(buf: &mut Vec<f64>, t: usize, b: &[f64], l_t: &[f64], a_prev: f64, point: bool) {
    buf.clear();
    buf.push(1.0);
    buf.extend_from_slice(b);
    buf.extend_from_slice(l_t);
    if t > 0 && !point {
        buf.push(a_prev);
    }
}

/// End-of-follow-up outcome row: `[1, baseline, L_(T-1), A_0, ..., A_(T-1)]`, or one
/// treatment column under a point treatment.
fn eof_row(buf: &mut Vec<f64>, b: &[f64], l_last: &[f64], hist: &[f64], point: bool) {
    buf.clear();
    buf.push(1.0);
    buf.extend_from_slice(b);
    buf.extend_from_slice(l_last);
    if point {
        buf.push(hist[0]);
    } else {
        buf.extend_from_slice(hist);
    }
}

/// A design matrix of `rows.len()` rows filled by `row(i, buf)` for each subject `i`.
fn build_design(
    rows: &[usize],
    k: usize,
    what: &str,
    mut row: impl FnMut(usize, &mut Vec<f64>),
) -> EconResult<Array2<f64>> {
    let mut x = resource::try_zeros2(rows.len(), k, what)?;
    let mut buf = Vec::with_capacity(k);
    for (r, &i) in rows.iter().enumerate() {
        row(i, &mut buf);
        debug_assert_eq!(buf.len(), k);
        for (c, v) in buf.iter().enumerate() {
            x[[r, c]] = *v;
        }
    }
    Ok(x)
}

/// Fit a linear (`lm`) or logistic (`glm`, binomial) model; warnings are prefixed with `what`.
fn fit_predict_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    names: Vec<String>,
    binary: bool,
    what: &str,
    config: &GFormulaConfig,
    warnings: &mut Vec<String>,
) -> EconResult<PredictModel> {
    let rows = x.nrows();
    let (fitted, std_errors, aliased, sigma, converged, iterations) = if binary {
        let mut spec = GlmSpec::logit();
        spec.epsilon = config.tolerance;
        spec.max_iter = config.max_iter;
        let fit = fit_glm(x.view(), y.view(), &spec)?;
        for w in &fit.warnings {
            warnings.push(format!("{what}: {w}"));
        }
        (
            fit.coefficients.to_vec(),
            fit.std_errors.to_vec(),
            fit.aliased,
            1.0,
            fit.converged,
            fit.iterations,
        )
    } else {
        let fit = solve_ols(x.view(), y.view())?;
        let df = rows.saturating_sub(fit.rank);
        if df == 0 {
            return Err(EconError::InsufficientData {
                required: fit.rank + 1,
                provided: rows,
                context: format!(
                    "{what}: the residual SD needs at least one residual degree of freedom"
                ),
            });
        }
        let sigma = (fit.residuals.iter().map(|r| r * r).sum::<f64>() / df as f64).sqrt();
        let se = fit
            .unscaled
            .diag()
            .iter()
            .map(|v| sigma * v.sqrt())
            .collect();
        (fit.beta.to_vec(), se, fit.aliased, sigma, true, 0)
    };
    if let Some(m) = aliasing_message(&aliased, &names) {
        warnings.push(format!("{what}: {m}"));
    }
    let coef = fitted
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    Ok(PredictModel {
        fitted,
        coef,
        std_errors,
        names,
        sigma,
        binary,
        n_obs: rows,
        converged,
        iterations,
    })
}

/// Fit every model of the g-formula.
///
/// Rows of period `t` are the subjects at risk at its start (`R_t`): every subject for
/// end-of-follow-up data, those with `time > t` (0-based `t`) for survival data. In period
/// `t` each time-varying covariate is regressed on `[1, baseline, L_(t-1), A_(t-1)]`
/// (logistic when its values on `R_t` are all 0 or 1, linear otherwise) and the treatment
/// on `[1, baseline, L_t, A_(t-1)]` (logistic). The outcome model is fitted on
/// `[1, baseline, L_(T-1), A_0, ..., A_(T-1)]` (linear or logistic). A treatment that never
/// changes within any subject's follow-up is a point treatment: one treatment model
/// (period 1) and one treatment column in the outcome model.
fn fit_models(
    data: &GFormulaData,
    config: &GFormulaConfig,
    warnings: &mut Vec<String>,
) -> EconResult<FittedModels> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_b = data.n_baseline_covars();
    let n_tv = data.n_time_varying_covars();

    // Risk sets, all checked before any model is fitted.
    let mut risk: Vec<Vec<usize>> = Vec::with_capacity(t_max);
    for t in 0..t_max {
        let rows: Vec<usize> = match &data.survival {
            Some(s) => (0..n).filter(|&i| s.time[i] > t).collect(),
            None => (0..n).collect(),
        };
        if rows.is_empty() {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!(
                    "no subject is at risk at the start of period {} (every follow-up ended \
                     earlier); use fewer periods",
                    t + 1
                ),
            });
        }
        risk.push(rows);
    }

    // Rule P: the treatment never changes within a subject's follow-up.
    let point = t_max == 1
        || (1..t_max).all(|t| {
            risk[t]
                .iter()
                .all(|&i| data.treatments[t][i] == data.treatments[0][i])
        });
    if point && t_max > 1 {
        warnings.push(POINT_TREATMENT_NOTE.to_string());
    }

    let mut b = Vec::with_capacity(n_b);
    let mut l_prev = Vec::with_capacity(n_tv);
    let mut l_cur = Vec::with_capacity(n_tv);
    let fill = |dst: &mut Vec<f64>, row: ndarray::ArrayView1<'_, f64>| {
        dst.clear();
        dst.extend(row.iter());
    };

    let mut covariates = Vec::with_capacity(t_max);
    let mut treatment = Vec::with_capacity(if point { 1 } else { t_max });
    let mut all_converged = true;
    for t in 0..t_max {
        resource::checkpoint()?;
        let rows = &risk[t];
        let l_t = &data.time_varying_covariates[t];

        // Covariate models
        let mut models_t = Vec::with_capacity(n_tv);
        if n_tv > 0 {
            let k = 1 + n_b + if t > 0 { n_tv + 1 } else { 0 };
            let x = build_design(rows, k, "g-formula design", |i, buf| {
                fill(&mut b, data.baseline_covariates.row(i));
                let a_prev = if t > 0 {
                    fill(&mut l_prev, data.time_varying_covariates[t - 1].row(i));
                    data.treatments[t - 1][i]
                } else {
                    0.0
                };
                cov_row(buf, t, &b, &l_prev, a_prev);
            })?;
            let mut names = vec!["(Intercept)".to_string()];
            names.extend(data.baseline_names.iter().cloned());
            if t > 0 {
                names.extend(
                    data.time_varying_names
                        .iter()
                        .map(|s| format!("{s}_t{}", t - 1)),
                );
                names.push(format!("{}_t{}", data.treatment_name, t - 1));
            }
            for j in 0..n_tv {
                let y: Array1<f64> = rows.iter().map(|&i| l_t[[i, j]]).collect();
                let binary = y.iter().all(|&v| v == 0.0 || v == 1.0);
                let what = format!(
                    "covariate model for {} in period {}",
                    data.time_varying_names[j],
                    t + 1
                );
                let m = fit_predict_model(&x, &y, names.clone(), binary, &what, config, warnings)?;
                all_converged &= m.converged;
                models_t.push(m);
            }
        }
        covariates.push(models_t);

        // Treatment model
        if t == 0 || !point {
            let k = 1 + n_b + n_tv + usize::from(t > 0 && !point);
            let x = build_design(rows, k, "g-formula design", |i, buf| {
                fill(&mut b, data.baseline_covariates.row(i));
                fill(&mut l_cur, l_t.row(i));
                let a_prev = if t > 0 {
                    data.treatments[t - 1][i]
                } else {
                    0.0
                };
                trt_row(buf, t, &b, &l_cur, a_prev, point);
            })?;
            let mut names = vec!["(Intercept)".to_string()];
            names.extend(data.baseline_names.iter().cloned());
            names.extend(data.time_varying_names.iter().map(|s| format!("{s}_t{t}")));
            if t > 0 && !point {
                names.push(format!("{}_t{}", data.treatment_name, t - 1));
            }
            let y: Array1<f64> = rows.iter().map(|&i| data.treatments[t][i]).collect();
            let what = format!("treatment model in period {}", t + 1);
            let m = fit_predict_model(&x, &y, names, true, &what, config, warnings)?;
            all_converged &= m.converged;
            treatment.push(m);
        }
    }

    // End-of-follow-up outcome model on every subject.
    let rows: Vec<usize> = (0..n).collect();
    let k = 1 + n_b + n_tv + if point { 1 } else { t_max };
    let mut hist = Vec::with_capacity(t_max);
    let x = build_design(&rows, k, "g-formula design", |i, buf| {
        fill(&mut b, data.baseline_covariates.row(i));
        fill(&mut l_cur, data.time_varying_covariates[t_max - 1].row(i));
        hist.clear();
        hist.extend(data.treatments.iter().map(|a| a[i]));
        eof_row(buf, &b, &l_cur, &hist, point);
    })?;
    let mut names = vec!["(Intercept)".to_string()];
    names.extend(data.baseline_names.iter().cloned());
    names.extend(data.time_varying_names.iter().cloned());
    if point {
        names.push(data.treatment_name.clone());
    } else {
        names.extend((0..t_max).map(|t| format!("{}_t{t}", data.treatment_name)));
    }
    let binary = config.outcome_type != GFormulaOutcomeType::Continuous;
    let outcome = fit_predict_model(
        &x,
        &data.outcome,
        names,
        binary,
        "outcome model",
        config,
        warnings,
    )?;
    all_converged &= outcome.converged;

    Ok(FittedModels {
        covariates,
        treatment,
        outcome,
        point_treatment: point,
        n_at_risk: risk.iter().map(|r| r.len()).collect(),
        all_converged,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Monte Carlo Simulation
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulated subjects between two cancellation checks.
const SIMULATION_CHECK_EVERY: usize = 256;

/// The simulation stream of `seed`: stream 1 of ChaCha8, so it never overlaps the bootstrap
/// resampling stream (stream 0) of the same seed.
fn sim_rng(seed: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(1);
    rng
}

/// Simulate one regime.
///
/// Every regime starts from `sim_rng(seed)` and consumes the same draws (a row per Monte
/// Carlo unit, one draw per time-varying covariate and one treatment draw per period,
/// whether the regime uses it or not), so regimes are compared under common random numbers.
/// Continuous covariates are drawn from `Normal(eta, sigma)`, binary ones from
/// `Bernoulli(plogis(eta))`.
fn simulate(
    data: &GFormulaData,
    models: &FittedModels,
    regime: &GFormulaIntervention,
    units: Units,
    outcome_type: GFormulaOutcomeType,
    seed: u64,
) -> EconResult<SimOutput> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_tv = data.n_time_varying_covars();
    let point = models.point_treatment;
    let mut rng = sim_rng(seed);
    let n_units = match units {
        Units::Empirical => n,
        Units::MonteCarlo(m) => m,
    };
    let mut per_unit =
        resource::try_with_capacity::<f64>(n_units, "Monte Carlo simulations (n_simulations)")?;

    let mut b: Vec<f64> = Vec::with_capacity(data.n_baseline_covars());
    let mut l_prev = vec![0.0; n_tv];
    let mut l_cur = vec![0.0; n_tv];
    let mut hist = Vec::with_capacity(t_max);
    let mut buf = Vec::new();
    for k in 0..n_units {
        if k % SIMULATION_CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }
        let i = match units {
            Units::Empirical => k,
            Units::MonteCarlo(_) => rng.gen_range(0..n),
        };
        b.clear();
        b.extend(data.baseline_covariates.row(i).iter());
        l_prev.iter_mut().for_each(|v| *v = 0.0);
        let mut a_prev = 0.0;
        hist.clear();
        for t in 0..t_max {
            for j in 0..n_tv {
                let m = &models.covariates[t][j];
                cov_row(&mut buf, t, &b, &l_prev, a_prev);
                let lp = m.eta(&buf);
                l_cur[j] = if m.binary {
                    let u: f64 = rng.gen_range(0.0..1.0);
                    f64::from(u8::from(u < plogis(lp)))
                } else {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    lp + m.sigma * z
                };
            }
            // Always consumed, so the regimes' streams stay aligned.
            let u_a: f64 = rng.gen_range(0.0..1.0);
            let a = if point && t > 0 {
                a_prev
            } else {
                match regime {
                    GFormulaIntervention::NaturalCourse => {
                        trt_row(&mut buf, t, &b, &l_cur, a_prev, point);
                        f64::from(u8::from(u_a < plogis(models.treatment[t].eta(&buf))))
                    }
                    GFormulaIntervention::Static { treat_all } => f64::from(u8::from(*treat_all)),
                    GFormulaIntervention::Threshold {
                        variable_idx,
                        cutoff,
                        above,
                    } => {
                        // validate_inputs checked the index
                        let v = l_cur[*variable_idx];
                        f64::from(u8::from(if *above { v > *cutoff } else { v <= *cutoff }))
                    }
                }
            };
            std::mem::swap(&mut l_prev, &mut l_cur);
            a_prev = a;
            hist.push(a);
        }
        // l_prev now holds L_(T-1)
        eof_row(&mut buf, &b, &l_prev, &hist, point);
        let lp = models.outcome.eta(&buf);
        per_unit.push(match outcome_type {
            GFormulaOutcomeType::Continuous => lp,
            GFormulaOutcomeType::Binary | GFormulaOutcomeType::Survival => plogis(lp),
        });
    }
    let mean = per_unit.iter().sum::<f64>() / n_units as f64;
    Ok(SimOutput {
        per_unit,
        mean,
        curve: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════════════

/// Bootstrap replicates evaluated in parallel per chunk; the cancellation budget is checked
/// between chunks and inside every replicate.
const BOOTSTRAP_CHUNK: usize = 64;

/// Runs `f` with `budget` installed on the current thread (a rayon worker does not inherit
/// the caller's thread-local budget), or plainly when there is none.
fn with_optional_budget<R>(budget: &Option<Budget>, f: impl FnOnce() -> R) -> R {
    match budget {
        Some(b) => resource::with_budget(b.clone(), f),
        None => f(),
    }
}

/// Bootstrap the g-formula to obtain standard errors.
/// Returns Vec of (risk_difference, risk_intervention, risk_natural) tuples.
///
/// Replicates run in parallel in chunks of [`BOOTSTRAP_CHUNK`], each worker running under
/// the caller's cooperative-cancellation budget. Replicate `b` has the seed
/// `boot_seed = seed + 1000 b` (wrapping): subjects are resampled from stream 0 of that
/// seed and simulated from stream 1 ([`sim_rng`]) with `n_simulations / 2` (at least 100)
/// units, so results do not depend on the chunking or the thread count. A failed replicate
/// is dropped; a cancellation or resource-limit error stops the bootstrap.
fn bootstrap_gformula(
    data: &GFormulaData,
    config: &GFormulaConfig,
    n_bootstrap: usize,
    seed: u64,
) -> EconResult<Vec<(f64, f64, f64)>> {
    let n = data.n_subjects();

    let mut successful_results =
        resource::try_with_capacity::<(f64, f64, f64)>(n_bootstrap, "bootstrap replications")?;
    let budget = resource::current_budget();
    let n_sim = (config.n_simulations / 2).max(100);

    let replicate = |b: usize| -> EconResult<(f64, f64, f64)> {
        let boot_seed = seed.wrapping_add((b as u64).wrapping_mul(1000));
        let mut rng = ChaCha8Rng::seed_from_u64(boot_seed);
        let boot_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = resample_data(data, &boot_indices)?;
        // Replicate warnings are not reported.
        let est = estimate_once(&boot_data, config, n_sim, boot_seed, &mut Vec::new())?;
        let (risk_int, risk_nat) = (est.intervention.mean, est.natural.mean);
        Ok((risk_int - risk_nat, risk_int, risk_nat))
    };

    // Run bootstrap in parallel, one bounded chunk at a time
    for b0 in (0..n_bootstrap).step_by(BOOTSTRAP_CHUNK) {
        resource::checkpoint()?;
        let b1 = b0.saturating_add(BOOTSTRAP_CHUNK).min(n_bootstrap);
        let results: Vec<EconResult<(f64, f64, f64)>> = (b0..b1)
            .into_par_iter()
            .map(|b| with_optional_budget(&budget, || replicate(b)))
            .collect();

        // Collect successful results in replicate order
        for res in results {
            match res {
                Ok(r) => successful_results.push(r),
                Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
                Err(_) => {}
            }
        }
    }

    if successful_results.len() < n_bootstrap / 2 {
        return Err(EconError::ConvergenceFailure {
            iterations: n_bootstrap,
            last_change: 0.0,
            suggestion: "More than half of bootstrap samples failed. Check model specification."
                .to_string(),
        });
    }

    Ok(successful_results)
}

/// Resample data by indices. The source data are valid, so the copy is assembled directly
/// (names and survival times carried along) instead of being re-validated.
fn resample_data(data: &GFormulaData, indices: &[usize]) -> EconResult<GFormulaData> {
    let n = indices.len();
    let take_rows = |m: &Array2<f64>| -> EconResult<Array2<f64>> {
        let mut out = resource::try_zeros2(n, m.ncols(), "g-formula bootstrap sample")?;
        for (new_i, &old_i) in indices.iter().enumerate() {
            out.row_mut(new_i).assign(&m.row(old_i));
        }
        Ok(out)
    };
    let take = |v: &Array1<f64>| -> Array1<f64> { indices.iter().map(|&i| v[i]).collect() };
    let baseline_covariates = take_rows(&data.baseline_covariates)?;
    let time_varying_covariates = data
        .time_varying_covariates
        .iter()
        .map(take_rows)
        .collect::<EconResult<Vec<_>>>()?;
    let treatments = data.treatments.iter().map(take).collect();
    let survival = data.survival.as_ref().map(|s| SurvivalOutcome {
        time: indices.iter().map(|&i| s.time[i]).collect(),
        event: indices.iter().map(|&i| s.event[i]).collect(),
    });
    Ok(GFormulaData {
        baseline_covariates,
        time_varying_covariates,
        treatments,
        outcome: take(&data.outcome),
        survival,
        baseline_names: data.baseline_names.clone(),
        time_varying_names: data.time_varying_names.clone(),
        treatment_name: data.treatment_name.clone(),
    })
}

/// Compute bootstrap confidence interval using percentile method.
fn compute_bootstrap_ci(
    bootstrap_results: &[(f64, f64, f64)],
    confidence_level: f64,
) -> (f64, f64, f64) {
    if bootstrap_results.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let mut rd_values: Vec<f64> = bootstrap_results.iter().map(|(rd, _, _)| *rd).collect();
    rd_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = rd_values.len();
    let alpha = 1.0 - confidence_level;

    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let ci_lower = rd_values.get(lower_idx).copied().unwrap_or(rd_values[0]);
    let ci_upper_idx = upper_idx.min(n - 1);
    let ci_upper = rd_values
        .get(ci_upper_idx)
        .copied()
        .unwrap_or(rd_values[n - 1]);

    // Standard error from bootstrap SD
    let mean: f64 = rd_values.iter().sum::<f64>() / n as f64;
    let denominator = (n - 1).max(1) as f64;
    let var: f64 = rd_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denominator;
    let se = var.sqrt();

    (se, ci_lower, ci_upper)
}

/// Compute z critical value for given confidence level.
fn compute_z_critical(confidence_level: f64) -> f64 {
    // Common values
    if (confidence_level - 0.95).abs() < 1e-10 {
        1.96
    } else if (confidence_level - 0.99).abs() < 1e-10 {
        2.576
    } else if (confidence_level - 0.90).abs() < 1e-10 {
        1.645
    } else {
        // General case using approximation
        let alpha = 1.0 - confidence_level;
        // Rational approximation to inverse normal
        let p = 1.0 - alpha / 2.0;
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    fn expect_invalid_spec(err: EconError, needle: &str) {
        match err {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains(needle),
                    "message `{message}` does not mention `{needle}`"
                )
            }
            other => panic!("expected InvalidSpecification mentioning `{needle}`, got {other:?}"),
        }
    }

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn base_config() -> GFormulaConfig {
        GFormulaConfig {
            n_simulations: 200,
            n_bootstrap: 20,
            seed: Some(42),
            outcome_type: GFormulaOutcomeType::Continuous,
            ..Default::default()
        }
    }

    #[test]
    fn test_gformula_counts_are_errors_not_aborts() {
        for cfg in [
            GFormulaConfig {
                n_simulations: usize::MAX,
                ..base_config()
            },
            GFormulaConfig {
                n_bootstrap: usize::MAX,
                ..base_config()
            },
        ] {
            let err = run_gformula(&create_test_data(), cfg).unwrap_err();
            assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
        }
    }

    #[test]
    fn test_gformula_rejects_bad_controls() {
        let data = create_test_data();
        let run = |cfg| run_gformula(&data, cfg);
        expect_invalid_spec(
            run(GFormulaConfig {
                n_bootstrap: 1,
                ..base_config()
            })
            .unwrap_err(),
            "n_bootstrap",
        );
        expect_invalid_spec(
            run(GFormulaConfig {
                max_iter: 0,
                ..base_config()
            })
            .unwrap_err(),
            "max_iter",
        );
        expect_invalid_spec(
            run(GFormulaConfig {
                tolerance: f64::NAN,
                ..base_config()
            })
            .unwrap_err(),
            "tolerance",
        );
        // 0 = no bootstrap, as before
        assert!(
            run(GFormulaConfig {
                n_bootstrap: 0,
                ..base_config()
            })
            .is_ok()
        );
    }

    #[test]
    fn test_gformula_honours_cancellation() {
        let r = resource::with_budget(cancelled_budget(), || {
            run_gformula(
                &create_test_data(),
                GFormulaConfig {
                    n_simulations: 200,
                    n_bootstrap: 20,
                    outcome_type: GFormulaOutcomeType::Continuous,
                    ..Default::default()
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    #[test]
    fn test_budget_reaches_rayon_workers() {
        let budget = Some(cancelled_budget());
        let out: Vec<EconResult<()>> = (0..8)
            .into_par_iter()
            .map(|_| with_optional_budget(&budget, resource::checkpoint))
            .collect();
        assert!(
            out.iter()
                .all(|r| matches!(r, Err(EconError::Cancelled(_))))
        );
    }

    /// Create a simple test dataset with known treatment effect.
    fn create_test_data() -> GFormulaData {
        let n = 100;

        // Baseline covariates (1 covariate: X)
        let mut baseline = Array2::zeros((n, 1));
        for i in 0..n {
            baseline[[i, 0]] = (i as f64) / (n as f64);
        }

        // Time-varying covariates (2 time points, 1 covariate each)
        let mut l0 = Array2::zeros((n, 1));
        let mut l1 = Array2::zeros((n, 1));

        // Treatments
        let mut a0 = Array1::zeros(n);
        let mut a1 = Array1::zeros(n);

        // Outcome
        let mut y = Array1::zeros(n);

        for i in 0..n {
            let x = baseline[[i, 0]];
            l0[[i, 0]] = 0.3 * x + 0.1 * ((i % 3) as f64 - 1.0) * 0.1;
            a0[i] = if l0[[i, 0]] > 0.15 { 1.0 } else { 0.0 };
            l1[[i, 0]] = 0.2 * l0[[i, 0]] + 0.1 * a0[i] + 0.05 * ((i % 5) as f64 - 2.0) * 0.1;
            a1[i] = if l1[[i, 0]] > 0.05 { 1.0 } else { 0.0 };
            y[i] = 0.2 * x
                + 0.3 * l1[[i, 0]]
                + 0.4 * a0[i]
                + 0.3 * a1[i]
                + 0.1 * ((i % 7) as f64 - 3.0) * 0.1;
            y[i] = y[i].max(0.0).min(1.0);
        }

        GFormulaData::new(baseline, vec![l0, l1], vec![a0, a1], y).unwrap()
    }

    #[test]
    fn test_gformula_basic() {
        let data = create_test_data();

        let config = GFormulaConfig {
            n_simulations: 200,
            time_points: 2,
            intervention: GFormulaIntervention::Static { treat_all: true },
            outcome_type: GFormulaOutcomeType::Continuous,
            n_bootstrap: 20,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_gformula(&data, config).unwrap();

        assert_eq!(result.n_subjects, 100);
        assert_eq!(result.n_time_points, 2);
        assert!(result.risk_natural >= 0.0 && result.risk_natural <= 1.0);
        assert!(result.risk_intervention >= 0.0 && result.risk_intervention <= 1.0);
        assert!(result.risk_ratio > 0.0);
        assert!(result.se_risk_difference > 0.0 && result.se_risk_difference.is_finite());
    }

    fn expect_insufficient(err: EconError, provided: usize) {
        assert!(
            matches!(err, EconError::InsufficientData { provided: p, .. } if p == provided),
            "got {err:?}"
        );
    }

    fn expect_invalid(err: EconError, needles: &[&str]) {
        let msg = err.to_string();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. }),
            "got {err:?}"
        );
        for n in needles {
            assert!(msg.contains(n), "message {msg:?} lacks {n:?}");
        }
    }

    #[test]
    fn test_gformula_rejects_zero_rows() {
        // CAUS-65: before, new() accepted 0 rows and run_gformula panicked in gen_range(0..0).
        let r = GFormulaData::new(
            Array2::zeros((0, 1)),
            vec![Array2::zeros((0, 1))],
            vec![Array1::zeros(0)],
            Array1::zeros(0),
        );
        expect_insufficient(r.unwrap_err(), 0);
    }

    #[test]
    fn test_run_gformula_guards_zero_rows() {
        // A literal bypasses new(); run_gformula must still refuse it (no panic).
        let data = GFormulaData {
            baseline_covariates: Array2::zeros((0, 1)),
            time_varying_covariates: vec![Array2::zeros((0, 1))],
            treatments: vec![Array1::zeros(0)],
            outcome: Array1::zeros(0),
            survival: None,
            baseline_names: vec!["B1".into()],
            time_varying_names: vec!["L1".into()],
            treatment_name: "A".into(),
        };
        let mut cfg = GFormulaConfig::default();
        cfg.n_simulations = 10;
        cfg.n_bootstrap = 0;
        expect_insufficient(run_gformula(&data, cfg).unwrap_err(), 0);
    }

    #[test]
    fn test_gformula_data_rejects_malformed_cells() {
        let n = 6;
        let ok_b = || Array2::from_shape_fn((n, 2), |(i, j)| (i * 3 + j) as f64 * 0.37 - 1.1);
        let ok_l = || Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.61).sin());
        let ok_a = || Array1::from(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        let ok_y = || Array1::from(vec![0.3, 1.2, 0.8, -0.4, 1.9, 0.1]);
        let mut b = ok_b();
        b[[3, 1]] = f64::NAN;
        expect_invalid(
            GFormulaData::new(b, vec![ok_l()], vec![ok_a()], ok_y()).unwrap_err(),
            &["B2", "row 4", "not finite"],
        );
        let mut a = ok_a();
        a[4] = 0.5;
        expect_invalid(
            GFormulaData::new(ok_b(), vec![ok_l()], vec![a], ok_y()).unwrap_err(),
            &["treatment", "period 1", "0 or 1", "row 5"],
        );
        let mut y = ok_y();
        y[0] = f64::INFINITY;
        expect_invalid(
            GFormulaData::new(ok_b(), vec![ok_l()], vec![ok_a()], y).unwrap_err(),
            &["outcome", "row 1"],
        );
        expect_invalid(
            GFormulaData::new(
                ok_b(),
                vec![ok_l(), Array2::zeros((n, 2))],
                vec![ok_a(), ok_a()],
                ok_y(),
            )
            .unwrap_err(),
            &["period 2", "2 columns", "period 1 has 1"],
        );
        assert!(GFormulaData::new(ok_b(), vec![ok_l()], vec![ok_a()], ok_y()).is_ok());
    }

    /// Noisy two-period DGP: one baseline and one time-varying covariate; the treatment
    /// varies within subjects (no point treatment).
    fn create_tv_test_data(n: usize, seed: u64, binary: bool) -> GFormulaData {
        use rand_distr::{Distribution, StandardNormal};
        let plogis = |x: f64| 1.0 / (1.0 + (-x).exp());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (mut b, mut l0, mut l1) = (
            Array2::zeros((n, 1)),
            Array2::zeros((n, 1)),
            Array2::zeros((n, 1)),
        );
        let (mut a0, mut a1, mut y) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
        for i in 0..n {
            // seven draws per row, always in this order
            let z0: f64 = StandardNormal.sample(&mut rng);
            let z1: f64 = StandardNormal.sample(&mut rng);
            let u0: f64 = rng.gen_range(0.0..1.0);
            let z2: f64 = StandardNormal.sample(&mut rng);
            let u1: f64 = rng.gen_range(0.0..1.0);
            let e: f64 = StandardNormal.sample(&mut rng);
            let u2: f64 = rng.gen_range(0.0..1.0);
            b[[i, 0]] = z0;
            l0[[i, 0]] = 0.5 * z0 + z1;
            a0[i] = f64::from(u0 < plogis(0.4 * l0[[i, 0]]));
            l1[[i, 0]] = 0.3 * z0 + 0.6 * l0[[i, 0]] - 0.4 * a0[i] + z2;
            a1[i] = f64::from(u1 < plogis(0.4 * l1[[i, 0]] + 0.8 * a0[i]));
            let eta = -0.5 + 0.5 * z0 + 0.8 * l1[[i, 0]] + 0.7 * a0[i] + 0.4 * a1[i];
            y[i] = if binary {
                f64::from(u2 < plogis(eta))
            } else {
                eta + e
            };
        }
        GFormulaData::new(b, vec![l0, l1], vec![a0, a1], y).unwrap()
    }

    fn small_tv_data() -> GFormulaData {
        create_tv_test_data(200, 11, false)
    }

    #[test]
    fn test_gformula_rejects_unknown_threshold_covariate() {
        // CAUS-105: before, index 5 with one covariate silently used L = 0.0 (always/never treat).
        let mut cfg = GFormulaConfig::default();
        cfg.n_simulations = 200;
        cfg.n_bootstrap = 0;
        cfg.outcome_type = GFormulaOutcomeType::Continuous;
        cfg.intervention = GFormulaIntervention::Threshold {
            variable_idx: 5,
            cutoff: -1.0,
            above: true,
        };
        expect_invalid(
            run_gformula(&small_tv_data(), cfg.clone()).unwrap_err(),
            &["threshold", "index 5", "1 time-varying covariate", "L1"],
        );
        // no time-varying covariate at all
        let n = 40;
        let data = GFormulaData::new(
            Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.7).cos()),
            vec![Array2::zeros((n, 0))],
            vec![Array1::from_shape_fn(n, |i| (i % 2) as f64)],
            Array1::from_shape_fn(n, |i| (i as f64 * 0.3).sin()),
        )
        .unwrap();
        cfg.intervention = GFormulaIntervention::Threshold {
            variable_idx: 0,
            cutoff: 0.5,
            above: true,
        };
        expect_invalid(
            run_gformula(&data, cfg.clone()).unwrap_err(),
            &["threshold", "no time-varying covariate"],
        );
        cfg.intervention = GFormulaIntervention::Threshold {
            variable_idx: 0,
            cutoff: f64::NAN,
            above: true,
        };
        expect_invalid(
            run_gformula(&small_tv_data(), cfg.clone()).unwrap_err(),
            &["cutoff", "finite"],
        );
        // a valid index works and the description names the covariate
        cfg.intervention = GFormulaIntervention::Threshold {
            variable_idx: 0,
            cutoff: -1.0,
            above: true,
        };
        let r = run_gformula(&small_tv_data(), cfg).unwrap();
        assert_eq!(r.intervention_description, "Treat if L1 > -1");
    }

    #[test]
    fn test_gformula_outcome_type_parses_aliases_and_rejects_unknown() {
        GFormulaOutcomeType::choices().validate().unwrap();
        for (s, v) in [
            ("continuous", GFormulaOutcomeType::Continuous),
            ("Linear", GFormulaOutcomeType::Continuous),
            ("gaussian", GFormulaOutcomeType::Continuous),
            ("binary", GFormulaOutcomeType::Binary),
            ("logistic", GFormulaOutcomeType::Binary),
            ("Survival", GFormulaOutcomeType::Survival),
            ("hazard", GFormulaOutcomeType::Survival),
            ("discrete-hazard", GFormulaOutcomeType::Survival),
        ] {
            assert_eq!(s.parse::<GFormulaOutcomeType>().unwrap(), v, "{s}");
        }
        expect_invalid(
            "cox".parse::<GFormulaOutcomeType>().unwrap_err(),
            &["outcome_type", "continuous", "binary", "survival"],
        );
    }

    #[test]
    fn test_gformula_regime_parses_aliases_and_rejects_unknown() {
        GFormulaRegime::choices().validate().unwrap();
        for (s, v) in [
            ("always_treat", GFormulaRegime::AlwaysTreat),
            ("always", GFormulaRegime::AlwaysTreat),
            ("never-treat", GFormulaRegime::NeverTreat),
            ("control", GFormulaRegime::NeverTreat),
            ("natural", GFormulaRegime::NaturalCourse),
            ("observed", GFormulaRegime::NaturalCourse),
            ("Threshold", GFormulaRegime::Threshold),
            ("dynamic", GFormulaRegime::Threshold),
        ] {
            assert_eq!(s.parse::<GFormulaRegime>().unwrap(), v, "{s}");
        }
        expect_invalid(
            "sometimes".parse::<GFormulaRegime>().unwrap_err(),
            &["intervention", "always_treat", "threshold"],
        );
    }

    #[test]
    fn test_gformula_intervention_from_regime() {
        use GFormulaRegime::*;
        assert!(matches!(
            GFormulaIntervention::from_regime(AlwaysTreat, None, None, None).unwrap(),
            GFormulaIntervention::Static { treat_all: true }
        ));
        assert!(matches!(
            GFormulaIntervention::from_regime(Threshold, Some(1), Some(0.25), None).unwrap(),
            GFormulaIntervention::Threshold {
                variable_idx: 1,
                above: true,
                ..
            }
        ));
        expect_invalid(
            GFormulaIntervention::from_regime(Threshold, None, Some(0.2), None).unwrap_err(),
            &["threshold_variable", "required"],
        );
        expect_invalid(
            GFormulaIntervention::from_regime(Threshold, Some(0), None, None).unwrap_err(),
            &["threshold_cutoff", "required"],
        );
        expect_invalid(
            GFormulaIntervention::from_regime(NeverTreat, Some(0), None, None).unwrap_err(),
            &["apply only to intervention 'threshold'"],
        );
    }

    fn case_col(ds: &crate::data::Dataset, name: &str) -> Array1<f64> {
        crate::data::columns::numeric_options(ds.df(), name)
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect()
    }

    fn nc_data(ds: &crate::data::Dataset, outcome: &str) -> GFormulaData {
        use ndarray::Axis;
        GFormulaData::new(
            case_col(ds, "B").insert_axis(Axis(1)),
            vec![case_col(ds, "L_t0").insert_axis(Axis(1))],
            vec![case_col(ds, "A_t0")],
            case_col(ds, outcome),
        )
        .unwrap()
        .with_names(&["B"], &["L"], "A")
        .unwrap()
    }

    fn mc_se(v: &[f64]) -> f64 {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64 / v.len() as f64)
            .sqrt()
    }

    #[test]
    fn test_validate_gformula_natural_course_risks() {
        // CAUS-16. Exact model-implied risks from validation/r/cases/causal_gformula.R
        // (120-point Gauss-Hermite over Normal(mu_i, sigma)); the Monte Carlo estimate must be
        // within 5 Monte Carlo SEs of them (MC tolerance, seeded). Before: covariates drawn
        // from Uniform(-sigma/2, sigma/2) gave natural 0.166 / always 0.197 against
        // 0.28397 / 0.30555 (> 100 SEs).
        let (case, ds) = crate::test_support::load_case("gformula_natural_course");
        let run = |outcome: &str, ty: GFormulaOutcomeType, treat_all: bool, n_sim: usize| {
            let mut c = GFormulaConfig::default();
            c.outcome_type = ty;
            c.n_simulations = n_sim;
            c.n_bootstrap = 0;
            c.seed = Some(20261209);
            c.intervention = GFormulaIntervention::Static { treat_all };
            run_gformula(&nc_data(&ds, outcome), c).unwrap()
        };
        let near = |key: &str, est: f64, se: f64| {
            let r = case.scalar(key);
            assert!(
                (est - r).abs() <= 5.0 * se,
                "{key}: p2a {est} R {r} |diff| {} > 5 MC SE {}",
                (est - r).abs(),
                5.0 * se
            );
        };
        let b1 = run("Y", GFormulaOutcomeType::Binary, true, 200_000);
        near(
            "natural_binary",
            b1.risk_natural,
            mc_se(&b1.simulated_natural),
        );
        near(
            "always_binary",
            b1.risk_intervention,
            mc_se(&b1.simulated_intervention),
        );
        let b0 = run("Y", GFormulaOutcomeType::Binary, false, 200_000);
        near(
            "never_binary",
            b0.risk_intervention,
            mc_se(&b0.simulated_intervention),
        );
        // natural course reproduces the observed mean (sampling gap 0.0005 + 5 MC SE 0.004)
        assert!((b1.risk_natural - case.scalar("observed_mean_binary")).abs() < 0.01);
        let c1 = run("Yc", GFormulaOutcomeType::Continuous, true, 1_000_000);
        near(
            "natural_cont",
            c1.risk_natural,
            mc_se(&c1.simulated_natural),
        );
        near(
            "always_cont",
            c1.risk_intervention,
            mc_se(&c1.simulated_intervention),
        );
        let c0 = run("Yc", GFormulaOutcomeType::Continuous, false, 1_000_000);
        near(
            "never_cont",
            c0.risk_intervention,
            mc_se(&c0.simulated_intervention),
        );
        assert!((c1.risk_natural - case.scalar("observed_mean_cont")).abs() < 0.02);
    }

    #[test]
    fn test_validate_gformula_natural_course_models() {
        // The covariate, treatment and outcome models against lm / glm (epsilon 1e-14) on the
        // same CSV.
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("gformula_natural_course");
        let n = ds.df().height();
        let mut cfg = GFormulaConfig::default();
        cfg.tolerance = 1e-14;
        cfg.outcome_type = GFormulaOutcomeType::Binary;
        let mut w = Vec::new();
        let mb = fit_models(&nc_data(&ds, "Y"), &cfg, &mut w).unwrap();
        cfg.outcome_type = GFormulaOutcomeType::Continuous;
        let mc = fit_models(&nc_data(&ds, "Yc"), &cfg, &mut w).unwrap();
        assert!(mb.point_treatment && !mb.covariates[0][0].binary);
        Check::new(&case)
            .vector("cov_coef", &mb.covariates[0][0].fitted, Tol::coef(n))
            .scalar("cov_sigma", mb.covariates[0][0].sigma, Tol::stat(n))
            .vector("trt_coef", &mb.treatment[0].fitted, Tol::coef(n))
            .vector("outcome_coef_binary", &mb.outcome.fitted, Tol::coef(n))
            .vector("outcome_coef_cont", &mc.outcome.fitted, Tol::coef(n))
            .scalar("outcome_sigma_cont", mc.outcome.sigma, Tol::stat(n))
            .finish();
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn test_gformula_natural_vs_natural_is_exactly_zero() {
        // Common random numbers: both regimes replay one stream. Before, the intervention
        // continued the natural course's stream, so RD was Monte Carlo noise.
        let mut c = GFormulaConfig::default();
        c.outcome_type = GFormulaOutcomeType::Continuous;
        c.n_simulations = 500;
        c.n_bootstrap = 0;
        c.intervention = GFormulaIntervention::NaturalCourse;
        let r = run_gformula(&create_tv_test_data(300, 5, false), c).unwrap();
        assert_eq!(r.risk_difference, 0.0);
        assert_eq!(r.risk_natural.to_bits(), r.risk_intervention.to_bits());
    }

    #[test]
    fn test_gformula_static_contrast_is_exact_under_common_random_numbers() {
        // CAUS-66 part: replaces the tautology `diff > 0.0 || diff < 1.0`. Linear outcome and
        // covariate models with shared draws: E[Y^1] - E[Y^0] = b_A0 + b_A1 + b_L * g_A
        // exactly (b: outcome [1, B, L_t1, A_t0, A_t1]; g: covariate model of L_t1
        // [1, B, L_t0, A_t0]).
        let data = create_tv_test_data(2_000, 5, false);
        let cfg = |treat_all| {
            let mut c = GFormulaConfig::default();
            c.outcome_type = GFormulaOutcomeType::Continuous;
            c.n_simulations = 2_000;
            c.n_bootstrap = 0;
            c.intervention = GFormulaIntervention::Static { treat_all };
            c
        };
        let r1 = run_gformula(&data, cfg(true)).unwrap();
        let r0 = run_gformula(&data, cfg(false)).unwrap();
        let m = fit_models(&data, &cfg(true), &mut Vec::new()).unwrap();
        assert!(!m.point_treatment);
        let (b, g) = (&m.outcome.fitted, &m.covariates[1][0].fitted);
        let formula = b[3] + b[4] + b[2] * g[3];
        assert!(
            (r1.risk_intervention - r0.risk_intervention - formula).abs() < 1e-12,
            "{} vs {formula}",
            r1.risk_intervention - r0.risk_intervention
        );
        // the DGP's total effect is 0.7 + 0.4 - 0.8 * 0.4 = 0.78 (SE about 0.06 at n = 2000)
        assert!(formula > 0.3);
    }

    #[test]
    fn test_gformula_threshold_extreme_cutoffs_equal_static() {
        let data = create_tv_test_data(300, 5, true);
        let run = |iv| {
            let mut c = GFormulaConfig::default();
            c.n_simulations = 1_000;
            c.n_bootstrap = 0;
            c.intervention = iv;
            run_gformula(&data, c).unwrap()
        };
        let always = run(GFormulaIntervention::Static { treat_all: true });
        let never = run(GFormulaIntervention::Static { treat_all: false });
        let lo = run(GFormulaIntervention::Threshold {
            variable_idx: 0,
            cutoff: -1e300,
            above: true,
        });
        let hi = run(GFormulaIntervention::Threshold {
            variable_idx: 0,
            cutoff: 1e300,
            above: true,
        });
        assert_eq!(
            lo.risk_intervention.to_bits(),
            always.risk_intervention.to_bits()
        );
        assert_eq!(
            hi.risk_intervention.to_bits(),
            never.risk_intervention.to_bits()
        );
    }

    #[test]
    fn test_gformula_covariate_scale_invariance() {
        // R10 scale test: glm.fit's relative deviance criterion and the least-squares solves
        // are scale-free, so rescaling a baseline covariate by 1e6 or 1e-6 leaves the
        // predictions unchanged. The old absolute gradient-norm criterion was not scale-free.
        let base = create_tv_test_data(300, 9, true);
        let mut c = GFormulaConfig::default();
        c.n_simulations = 2_000;
        c.n_bootstrap = 0;
        let r = run_gformula(&base, c.clone()).unwrap();
        for s in [1e6, 1e-6] {
            let mut d = base.clone();
            d.baseline_covariates.mapv_inplace(|v| v * s);
            let rs = run_gformula(&d, c.clone()).unwrap();
            assert!(rs.all_models_converged);
            assert!(
                (rs.risk_natural - r.risk_natural).abs() <= 1e-9 * r.risk_natural.abs(),
                "scale {s}"
            );
            assert!(
                (rs.risk_intervention - r.risk_intervention).abs()
                    <= 1e-9 * r.risk_intervention.abs(),
                "scale {s}"
            );
        }
    }

    #[test]
    fn test_gformula_bootstrap_is_reproducible_and_thread_invariant() {
        // Replaces T0-WP12's captured-constant pin (CAUS-16 changes the simulated values by
        // design): the chunked parallel bootstrap (70 replicates > one chunk of 64) gives
        // bit-identical draws with one thread and many.
        let cfg = || {
            let mut c = GFormulaConfig::default();
            c.n_simulations = 200;
            c.n_bootstrap = 70;
            c.seed = Some(42);
            c.outcome_type = GFormulaOutcomeType::Continuous;
            c
        };
        let data = create_tv_test_data(150, 4, false);
        let a = run_gformula(&data, cfg()).unwrap();
        let b = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run_gformula(&data, cfg()).unwrap());
        assert_eq!(a.bootstrap_rd.len(), 70);
        assert_eq!(a.bootstrap_rd, b.bootstrap_rd);
        assert_eq!(
            a.se_risk_difference.to_bits(),
            b.se_risk_difference.to_bits()
        );
        assert!(a.se_risk_difference > 0.0 && a.ci_lower_rd < a.ci_upper_rd);
    }

    #[test]
    fn test_gformula_data_validation() {
        let baseline = Array2::zeros((10, 2));
        let l0 = Array2::zeros((10, 1));
        let l1 = Array2::zeros((8, 1)); // Wrong size
        let a0 = Array1::zeros(10);
        let a1 = Array1::zeros(10);
        let y = Array1::zeros(10);

        let result = GFormulaData::new(baseline, vec![l0, l1], vec![a0, a1], y);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_ci() {
        let bootstrap_results = vec![
            (0.1, 0.6, 0.5),
            (0.15, 0.65, 0.5),
            (0.05, 0.55, 0.5),
            (0.12, 0.62, 0.5),
            (0.08, 0.58, 0.5),
        ];

        let (se, ci_lower, ci_upper) = compute_bootstrap_ci(&bootstrap_results, 0.95);

        assert!(se > 0.0 && se.is_finite());
        assert!(ci_lower < ci_upper);
    }

    #[test]
    fn test_z_critical() {
        let z_95 = compute_z_critical(0.95);
        assert!((z_95 - 1.96).abs() < 0.01);

        let z_99 = compute_z_critical(0.99);
        assert!((z_99 - 2.576).abs() < 0.01);
    }
}
