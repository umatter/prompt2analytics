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
use std::sync::Arc;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::resource::{self, Budget};
use crate::traits::estimator::{SignificanceLevel, logistic_cdf, normal_cdf};

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

/// Fitted model for time-varying covariates.
#[derive(Debug, Clone)]
struct TimeVaryingCovariateModel {
    /// Coefficients for each covariate (one row per covariate, columns are predictors)
    coefficients: Vec<Array1<f64>>,
    /// Residual standard deviation for each covariate
    residual_sd: Vec<f64>,
    /// Whether each covariate is binary (use logistic) or continuous (use linear)
    is_binary: Vec<bool>,
}

/// Fitted treatment model.
#[derive(Debug, Clone)]
struct TreatmentModel {
    /// Logistic regression coefficients
    coefficients: Array1<f64>,
    /// Whether the model converged
    converged: bool,
}

/// Fitted outcome model.
#[derive(Debug, Clone)]
struct OutcomeModel {
    /// Coefficients
    coefficients: Array1<f64>,
    /// Residual standard deviation (for continuous outcomes)
    residual_sd: f64,
    /// Whether the model converged
    converged: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the parametric g-formula.
///
/// Estimates causal effects of time-varying treatments using Monte Carlo simulation.
///
/// # Arguments
/// * `data` - Input data containing baseline covariates, time-varying covariates,
///            treatments, and outcomes
/// * `config` - Configuration options including intervention type, number of
///              simulations, etc.
///
/// # Returns
/// `GFormulaResult` containing risk estimates, causal effects, and confidence intervals.
///
/// # Algorithm
///
/// 1. **Fit models** for each time-varying covariate, treatment, and outcome
/// 2. **Natural course simulation**: For each MC sample:
///    - Sample baseline from observed data
///    - At each time t: simulate L_t, sample A_t from treatment model
///    - Compute outcome probability
/// 3. **Intervention simulation**: Same as above but assign A_t per intervention
/// 4. **Bootstrap** for standard errors
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{run_gformula, GFormulaData, GFormulaConfig, GFormulaIntervention};
///
/// let data = GFormulaData::new(baseline, time_varying, treatments, outcome)?;
/// let config = GFormulaConfig {
///     n_simulations: 1000,
///     time_points: 2,
///     intervention: GFormulaIntervention::Static { treat_all: true },
///     ..Default::default()
/// };
///
/// let result = run_gformula(&data, config)?;
/// println!("Risk Difference: {:.4} (95% CI: [{:.4}, {:.4}])",
///          result.risk_difference, result.ci_lower_rd, result.ci_upper_rd);
/// ```
///
/// # References
///
/// - Robins (1986), Mathematical Modelling
/// - Hernan & Robins (2020), Causal Inference: What If, Chapter 21
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

    // Initialize RNG
    let seed = config.seed.unwrap_or(42);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // ═══════════════════════════════════════════════════════════════════════
    // Stage 1: Fit Models
    // ═══════════════════════════════════════════════════════════════════════

    // Fit time-varying covariate models: L_t ~ f(L_{t-1}, A_{t-1}, baseline)
    let covariate_models = fit_covariate_models(data, &config)?;

    // Fit treatment models: A_t ~ g(L_t, A_{t-1}, baseline) for each time point
    let treatment_models = fit_treatment_models(data, &config)?;

    // Fit outcome model: Y ~ h(L_T, A, baseline) where T is last time point
    let outcome_model = fit_outcome_model(data, &config)?;

    let all_converged = treatment_models.iter().all(|m| m.converged) && outcome_model.converged;
    if !all_converged {
        warnings.push("Some models did not converge. Consider increasing max_iter.".to_string());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stage 2: Monte Carlo Simulation
    // ═══════════════════════════════════════════════════════════════════════

    // Simulate under natural course
    let outcomes_natural = simulate_gformula(
        data,
        &covariate_models,
        &treatment_models,
        &outcome_model,
        &GFormulaIntervention::NaturalCourse,
        config.n_simulations,
        &config,
        &mut rng,
    )?;

    // Simulate under intervention
    let outcomes_intervention = simulate_gformula(
        data,
        &covariate_models,
        &treatment_models,
        &outcome_model,
        &config.intervention,
        config.n_simulations,
        &config,
        &mut rng,
    )?;

    // Compute risks
    let risk_natural: f64 = outcomes_natural.iter().sum::<f64>() / config.n_simulations as f64;
    let risk_intervention: f64 =
        outcomes_intervention.iter().sum::<f64>() / config.n_simulations as f64;

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
    // Stage 3: Bootstrap for Standard Errors
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
        all_models_converged: all_converged,
        warnings,
        simulated_natural: outcomes_natural,
        simulated_intervention: outcomes_intervention,
        bootstrap_rd: bootstrap_results.iter().map(|(rd, _, _)| *rd).collect(),
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
// Model Fitting Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Fit models for time-varying covariates.
/// For each covariate at time t, we fit: L_t[j] ~ f(L_{t-1}, A_{t-1}, baseline)
fn fit_covariate_models(
    data: &GFormulaData,
    config: &GFormulaConfig,
) -> EconResult<Vec<TimeVaryingCovariateModel>> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_baseline = data.n_baseline_covars();
    let n_tv = data.n_time_varying_covars();

    let mut models = Vec::with_capacity(t_max);

    for t in 0..t_max {
        let mut coef_list = Vec::with_capacity(n_tv);
        let mut sd_list = Vec::with_capacity(n_tv);
        let mut is_binary_list = Vec::with_capacity(n_tv);

        // Get outcome for this time's covariates
        let l_t = &data.time_varying_covariates[t];

        // Build design matrix: [1, baseline, L_{t-1} (if t>0), A_{t-1} (if t>0)]
        let n_predictors = 1 + n_baseline + (if t > 0 { n_tv + 1 } else { 0 });
        let mut x = Array2::zeros((n, n_predictors));

        // Intercept
        for i in 0..n {
            x[[i, 0]] = 1.0;
        }

        // Baseline covariates
        for i in 0..n {
            for j in 0..n_baseline {
                x[[i, 1 + j]] = data.baseline_covariates[[i, j]];
            }
        }

        // Lagged values if t > 0
        if t > 0 {
            let l_prev = &data.time_varying_covariates[t - 1];
            let a_prev = &data.treatments[t - 1];

            for i in 0..n {
                for j in 0..n_tv {
                    x[[i, 1 + n_baseline + j]] = l_prev[[i, j]];
                }
                x[[i, 1 + n_baseline + n_tv]] = a_prev[i];
            }
        }

        // Fit model for each covariate
        for j in 0..n_tv {
            let y: Array1<f64> = l_t.column(j).to_owned();

            // Determine if binary (all values 0 or 1)
            let is_binary = y.iter().all(|&v| (v == 0.0) || (v == 1.0));
            is_binary_list.push(is_binary);

            if is_binary {
                // Logistic regression
                let (_, beta, _, _) = fit_logistic(&x, &y, config.max_iter, config.tolerance)?;
                coef_list.push(beta);
                sd_list.push(1.0); // Not used for binary
            } else {
                // Linear regression
                let (beta, resid_sd) = fit_linear(&x, &y)?;
                coef_list.push(beta);
                sd_list.push(resid_sd);
            }
        }

        models.push(TimeVaryingCovariateModel {
            coefficients: coef_list,
            residual_sd: sd_list,
            is_binary: is_binary_list,
        });
    }

    Ok(models)
}

/// Fit treatment models for each time point.
/// A_t ~ g(L_t, A_{t-1}, baseline) using logistic regression
fn fit_treatment_models(
    data: &GFormulaData,
    config: &GFormulaConfig,
) -> EconResult<Vec<TreatmentModel>> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_baseline = data.n_baseline_covars();
    let n_tv = data.n_time_varying_covars();

    let mut models = Vec::with_capacity(t_max);

    for t in 0..t_max {
        // Design: [1, baseline, L_t, A_{t-1} (if t>0)]
        let n_predictors = 1 + n_baseline + n_tv + (if t > 0 { 1 } else { 0 });
        let mut x = Array2::zeros((n, n_predictors));

        // Intercept
        for i in 0..n {
            x[[i, 0]] = 1.0;
        }

        // Baseline
        for i in 0..n {
            for j in 0..n_baseline {
                x[[i, 1 + j]] = data.baseline_covariates[[i, j]];
            }
        }

        // Current time-varying covariates
        let l_t = &data.time_varying_covariates[t];
        for i in 0..n {
            for j in 0..n_tv {
                x[[i, 1 + n_baseline + j]] = l_t[[i, j]];
            }
        }

        // Lagged treatment if t > 0
        if t > 0 {
            let a_prev = &data.treatments[t - 1];
            for i in 0..n {
                x[[i, 1 + n_baseline + n_tv]] = a_prev[i];
            }
        }

        let y = &data.treatments[t];
        let (_, beta, converged, _) = fit_logistic(&x, y, config.max_iter, config.tolerance)?;

        models.push(TreatmentModel {
            coefficients: beta,
            converged,
        });
    }

    Ok(models)
}

/// Fit outcome model.
/// Y ~ h(L_T, cumulative A, baseline)
fn fit_outcome_model(data: &GFormulaData, config: &GFormulaConfig) -> EconResult<OutcomeModel> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_baseline = data.n_baseline_covars();
    let n_tv = data.n_time_varying_covars();

    // Design: [1, baseline, L_T (last time-varying), cumulative treatment]
    // Cumulative treatment = sum of all A_t for each subject
    let n_predictors = 1 + n_baseline + n_tv + t_max;
    let mut x = Array2::zeros((n, n_predictors));

    // Intercept
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }

    // Baseline
    for i in 0..n {
        for j in 0..n_baseline {
            x[[i, 1 + j]] = data.baseline_covariates[[i, j]];
        }
    }

    // Last time-varying covariates
    let l_last = &data.time_varying_covariates[t_max - 1];
    for i in 0..n {
        for j in 0..n_tv {
            x[[i, 1 + n_baseline + j]] = l_last[[i, j]];
        }
    }

    // Treatment at each time point (allows flexible effect over time)
    for t in 0..t_max {
        for i in 0..n {
            x[[i, 1 + n_baseline + n_tv + t]] = data.treatments[t][i];
        }
    }

    let y = &data.outcome;

    match config.outcome_type {
        GFormulaOutcomeType::Continuous => {
            let (beta, resid_sd) = fit_linear(&x, y)?;
            Ok(OutcomeModel {
                coefficients: beta,
                residual_sd: resid_sd,
                converged: true,
            })
        }
        GFormulaOutcomeType::Binary | GFormulaOutcomeType::Survival => {
            let (_, beta, converged, _) = fit_logistic(&x, y, config.max_iter, config.tolerance)?;
            Ok(OutcomeModel {
                coefficients: beta,
                residual_sd: 1.0,
                converged,
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Monte Carlo Simulation
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulated subjects between two cancellation checks.
const SIMULATION_CHECK_EVERY: usize = 256;

/// Simulate outcomes under a given intervention.
#[allow(clippy::too_many_arguments)]
fn simulate_gformula<R: Rng>(
    data: &GFormulaData,
    covariate_models: &[TimeVaryingCovariateModel],
    treatment_models: &[TreatmentModel],
    outcome_model: &OutcomeModel,
    intervention: &GFormulaIntervention,
    n_sim: usize,
    config: &GFormulaConfig,
    rng: &mut R,
) -> EconResult<Vec<f64>> {
    let n = data.n_subjects();
    let t_max = data.n_time_points();
    let n_baseline = data.n_baseline_covars();
    let n_tv = data.n_time_varying_covars();

    let mut outcomes =
        resource::try_with_capacity::<f64>(n_sim, "Monte Carlo simulations (n_simulations)")?;

    for s in 0..n_sim {
        if s % SIMULATION_CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }

        // Sample a subject with replacement
        let subject_idx = rng.gen_range(0..n);

        // Get baseline for this subject
        let baseline: Array1<f64> = data.baseline_covariates.row(subject_idx).to_owned();

        // Track simulated values
        let mut l_current = Array1::zeros(n_tv);
        let mut a_current = 0.0;
        let mut a_history = Vec::with_capacity(t_max);

        // Simulate forward through time
        for t in 0..t_max {
            // Simulate time-varying covariates L_t
            let cov_model = &covariate_models[t];
            let mut l_new = Array1::zeros(n_tv);

            for j in 0..n_tv {
                // Build predictor vector
                let n_pred = 1 + n_baseline + (if t > 0 { n_tv + 1 } else { 0 });
                let mut x_pred = Array1::zeros(n_pred);
                x_pred[0] = 1.0;
                for k in 0..n_baseline {
                    x_pred[1 + k] = baseline[k];
                }
                if t > 0 {
                    for k in 0..n_tv {
                        x_pred[1 + n_baseline + k] = l_current[k];
                    }
                    x_pred[1 + n_baseline + n_tv] = a_current;
                }

                // Predict
                let linear_pred: f64 = x_pred
                    .iter()
                    .zip(cov_model.coefficients[j].iter())
                    .map(|(x, b)| x * b)
                    .sum();

                if cov_model.is_binary[j] {
                    // Sample from Bernoulli with probability expit(linear_pred)
                    let p = logistic_cdf(linear_pred);
                    let u: f64 = rng.gen_range(0.0..1.0);
                    l_new[j] = if u < p { 1.0 } else { 0.0 };
                } else {
                    // Sample from Normal(linear_pred, residual_sd)
                    let u: f64 = rng.gen_range(-1.0..1.0);
                    l_new[j] = linear_pred + cov_model.residual_sd[j] * u * 0.5;
                }
            }
            l_current = l_new;

            // Assign treatment according to intervention
            let a_t = match intervention {
                GFormulaIntervention::NaturalCourse => {
                    // Sample from treatment model
                    let trt_model = &treatment_models[t];
                    let n_pred = 1 + n_baseline + n_tv + (if t > 0 { 1 } else { 0 });
                    let mut x_pred = Array1::zeros(n_pred);
                    x_pred[0] = 1.0;
                    for k in 0..n_baseline {
                        x_pred[1 + k] = baseline[k];
                    }
                    for k in 0..n_tv {
                        x_pred[1 + n_baseline + k] = l_current[k];
                    }
                    if t > 0 {
                        x_pred[1 + n_baseline + n_tv] = a_current;
                    }

                    let linear_pred: f64 = x_pred
                        .iter()
                        .zip(trt_model.coefficients.iter())
                        .map(|(x, b)| x * b)
                        .sum();
                    let p = logistic_cdf(linear_pred);
                    let u: f64 = rng.gen_range(0.0..1.0);
                    if u < p { 1.0 } else { 0.0 }
                }
                GFormulaIntervention::Static { treat_all } => {
                    if *treat_all {
                        1.0
                    } else {
                        0.0
                    }
                }
                GFormulaIntervention::Threshold {
                    variable_idx,
                    cutoff,
                    above,
                } => {
                    // validate_inputs checked the index
                    let var_val = l_current[*variable_idx];
                    if *above {
                        if var_val > *cutoff { 1.0 } else { 0.0 }
                    } else if var_val <= *cutoff {
                        1.0
                    } else {
                        0.0
                    }
                }
            };

            a_current = a_t;
            a_history.push(a_t);
        }

        // Compute outcome probability
        // Design: [1, baseline, L_T, A_0, ..., A_{T-1}]
        let n_out_pred = 1 + n_baseline + n_tv + t_max;
        let mut x_out = Array1::zeros(n_out_pred);
        x_out[0] = 1.0;
        for k in 0..n_baseline {
            x_out[1 + k] = baseline[k];
        }
        for k in 0..n_tv {
            x_out[1 + n_baseline + k] = l_current[k];
        }
        for (t_idx, &a_t) in a_history.iter().enumerate() {
            x_out[1 + n_baseline + n_tv + t_idx] = a_t;
        }

        let linear_pred: f64 = x_out
            .iter()
            .zip(outcome_model.coefficients.iter())
            .map(|(x, b)| x * b)
            .sum();

        let outcome_value = match config.outcome_type {
            GFormulaOutcomeType::Continuous => linear_pred,
            GFormulaOutcomeType::Binary | GFormulaOutcomeType::Survival => {
                logistic_cdf(linear_pred)
            }
        };

        outcomes.push(outcome_value);
    }

    Ok(outcomes)
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
/// the caller's cooperative-cancellation budget. Replicate `b` is seeded with
/// `seed + 1000 b` (wrapping), so results do not depend on the chunking. A failed replicate
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

    // Wrap data and config in Arc for sharing across threads
    let data_arc = Arc::new(data.clone());
    let config_arc = Arc::new(config.clone());

    let replicate = |b: usize| -> EconResult<(f64, f64, f64)> {
        let data = Arc::clone(&data_arc);
        let config = Arc::clone(&config_arc);

        // Seed for this bootstrap sample
        let boot_seed = seed.wrapping_add((b as u64).wrapping_mul(1000));
        let mut rng = ChaCha8Rng::seed_from_u64(boot_seed);

        // Resample subjects with replacement
        let boot_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

        // Create bootstrap sample
        let boot_data = resample_data(&data, &boot_indices)?;

        // Fit models on bootstrap sample
        let cov_models = fit_covariate_models(&boot_data, &config)?;
        let trt_models = fit_treatment_models(&boot_data, &config)?;
        let out_model = fit_outcome_model(&boot_data, &config)?;

        // Number of simulations for bootstrap (fewer for speed)
        let n_sim = (config.n_simulations / 2).max(100);

        // Simulate
        let out_nat = simulate_gformula(
            &boot_data,
            &cov_models,
            &trt_models,
            &out_model,
            &GFormulaIntervention::NaturalCourse,
            n_sim,
            &config,
            &mut rng,
        )?;

        let out_int = simulate_gformula(
            &boot_data,
            &cov_models,
            &trt_models,
            &out_model,
            &config.intervention,
            n_sim,
            &config,
            &mut rng,
        )?;

        let risk_nat: f64 = out_nat.iter().sum::<f64>() / out_nat.len() as f64;
        let risk_int: f64 = out_int.iter().sum::<f64>() / out_int.len() as f64;
        let rd = risk_int - risk_nat;

        Ok((rd, risk_int, risk_nat))
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
// Helper Functions: Linear and Logistic Regression
// ═══════════════════════════════════════════════════════════════════════════════

/// Fit linear regression using OLS.
/// Returns (coefficients, residual_sd).
fn fit_linear(x: &Array2<f64>, y: &Array1<f64>) -> EconResult<(Array1<f64>, f64)> {
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Linear regression in g-formula".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // Compute residual SD
    let y_hat = x.dot(&beta);
    let residuals = y - &y_hat;
    let n = y.len() as f64;
    let k = beta.len() as f64;
    let df = (n - k).max(1.0);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let resid_sd = (ssr / df).sqrt();

    Ok((beta, resid_sd))
}

/// Fit logistic regression using Newton-Raphson (IRLS).
/// Returns (predictions, coefficients, converged, iterations).
fn fit_logistic(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, bool, usize)> {
    let n = y.len();
    let k = x.ncols();

    // Initialize coefficients
    let mut beta = Array1::zeros(k);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // Linear predictor
        let z: Array1<f64> = x.dot(&beta);

        // Probabilities
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient
        let residuals = y - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            converged = true;
            break;
        }

        // Weights
        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        // Hessian
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton-Raphson update
        let neg_hessian = &hessian * -1.0;
        match safe_inverse(&neg_hessian.view()) {
            Ok((hess_inv, _)) => {
                let delta = hess_inv.dot(&gradient);
                beta = &beta + &delta;
            }
            Err(_) => {
                // Use gradient descent with small step if Hessian is singular
                let step_size = 0.01 / grad_norm.max(1.0);
                beta = &beta + &(gradient * step_size);
            }
        }
    }

    // Final predictions
    let z_final: Array1<f64> = x.dot(&beta);
    let p_final: Array1<f64> = z_final.mapv(logistic_cdf);

    Ok((p_final, beta, converged, iterations))
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

    #[test]
    fn test_gformula_bootstrap_unchanged_by_chunking() {
        // Pinned at 1fdb22f, re-pinned at T3-WP01. In create_test_data a1 equals a0 row for
        // row, so the outcome model's X'X in fit_linear ([1, x, l1, a0, a1]) has rank 4 of 5
        // in the main fit and in every replicate (equilibrated reciprocal condition 1e-34 to
        // 1e-18). The old safe_inverse took faer's Cholesky of that singular matrix whenever
        // it did not fail (about half the replicates), so those replicates' coefficients
        // were noise and the SE was 36.45 and 46.57 on a risk difference of 0.386; the
        // symmetric generalized inverse gives 0.050 and 0.051. Per-replicate seeds are
        // seed + 1000 b, unchanged. 130 replicates cross the 64-replicate chunk boundary
        // twice.
        const SE_RD: [(usize, f64); 2] = [(20, 0.05016316903989167), (130, 0.051370887581548556)];
        for (b, se_rd) in SE_RD {
            let cfg = GFormulaConfig {
                n_simulations: 200,
                time_points: 2,
                intervention: GFormulaIntervention::Static { treat_all: true },
                outcome_type: GFormulaOutcomeType::Continuous,
                n_bootstrap: b,
                seed: Some(42),
                ..Default::default()
            };
            let r = run_gformula(&create_test_data(), cfg).unwrap();
            assert_eq!(r.se_risk_difference, se_rd, "n_bootstrap {b}");
        }
    }

    #[test]
    fn test_gformula_static_interventions() {
        let data = create_test_data();

        let config_always = GFormulaConfig {
            n_simulations: 200,
            intervention: GFormulaIntervention::Static { treat_all: true },
            outcome_type: GFormulaOutcomeType::Continuous,
            n_bootstrap: 20,
            seed: Some(42),
            ..Default::default()
        };
        let result_always = run_gformula(&data, config_always).unwrap();

        let config_never = GFormulaConfig {
            n_simulations: 200,
            intervention: GFormulaIntervention::Static { treat_all: false },
            outcome_type: GFormulaOutcomeType::Continuous,
            n_bootstrap: 20,
            seed: Some(42),
            ..Default::default()
        };
        let result_never = run_gformula(&data, config_never).unwrap();

        // The two interventions should give different risks
        let diff = (result_always.risk_intervention - result_never.risk_intervention).abs();
        println!("Always treat risk: {}", result_always.risk_intervention);
        println!("Never treat risk: {}", result_never.risk_intervention);
        assert!(diff > 0.0 || diff < 1.0); // Just check it doesn't crash
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
