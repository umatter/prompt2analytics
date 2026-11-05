//! Treatment effect estimation using inverse probability weighting (IPW) and doubly robust methods.
//!
//! This module provides causal inference methods for estimating treatment effects:
//! - **IPW (Inverse Probability Weighting)**: Estimates ATE/ATT using propensity score weighting
//! - **AIPW (Augmented IPW)**: Doubly robust estimation combining IPW with outcome regression
//!
//! # References
//!
//! - Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement
//!   from a Finite Universe." *Journal of the American Statistical Association*, 47(260), 663-685.
//! - Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients
//!   When Some Regressors Are Not Always Observed." *JASA*, 89(427), 846-866.
//! - Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal
//!   Inference Models." *Biometrics*, 61(4), 962-973.
//! - Implementation inspired by R package `causalweight` (Bodory & Huber, 2018).
//!   Source: <https://cran.r-project.org/package=causalweight>

use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmLink, fit_propensity};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::options::Choices;
use crate::resource;
use crate::stats::robust::{QuantileType, percentile_interval, quantile_sorted};
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Target estimand for treatment effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Estimand {
    /// Average Treatment Effect (ATE): E[Y(1) - Y(0)]
    /// Effect averaged over the entire population.
    #[default]
    ATE,
    /// Average Treatment Effect on the Treated (ATT): E[Y(1) - Y(0) | D=1]
    /// Effect averaged over those who received treatment.
    ATT,
}

impl fmt::Display for Estimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Estimand::ATE => write!(f, "ATE (Average Treatment Effect)"),
            Estimand::ATT => write!(f, "ATT (Average Treatment Effect on Treated)"),
        }
    }
}

crate::option_enum!(
    Estimand,
    Choices {
        option: "estimand",
        values: &[(&["ate"], Estimand::ATE), (&["att"], Estimand::ATT)],
    },
    no_display
);

/// Doubly robust estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DRMethod {
    /// Augmented Inverse Probability Weighting (AIPW) - doubly robust
    #[default]
    AIPW,
    /// Inverse Probability Weighting only (not doubly robust)
    IPW,
    /// Outcome regression only (not doubly robust)
    Regression,
}

impl fmt::Display for DRMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DRMethod::AIPW => write!(f, "AIPW (Augmented IPW - Doubly Robust)"),
            DRMethod::IPW => write!(f, "IPW (Inverse Probability Weighting)"),
            DRMethod::Regression => write!(f, "Regression Adjustment"),
        }
    }
}

crate::option_enum!(
    DRMethod,
    Choices {
        option: "method",
        values: &[
            (&["aipw", "dr", "doubly_robust"], DRMethod::AIPW),
            (&["ipw"], DRMethod::IPW),
            (
                &["regression", "reg", "ra", "outcome_regression"],
                DRMethod::Regression,
            ),
        ],
    },
    no_display
);

/// Configuration for IPW treatment effect estimation.
#[derive(Debug, Clone)]
pub struct IpwConfig {
    /// Target estimand (ATE or ATT)
    pub estimand: Estimand,
    /// Trimming threshold for propensity scores (default: 0.05), in [0, 0.5).
    /// Observations with p(X) < trim or p(X) > 1-trim are excluded
    pub trim: f64,
    /// Number of bootstrap replications for standard errors (default: 0 = analytic).
    /// When 0, `std_error` is the M-estimation SE that accounts for the estimated logit
    /// propensity (`se_mest`). Set to e.g. 999 to use the bootstrap instead. Must be 0 or at least 2; a count whose
    /// result buffer cannot be allocated is refused with `EconError::ResourceLimit`.
    pub bootstrap: usize,
    /// Use normalized (Hajek) weights (default: true)
    /// If false, uses Horvitz-Thompson weights
    pub normalized: bool,
    /// Random seed for the bootstrap; `None` uses [`DEFAULT_BOOTSTRAP_SEED`] (42), so
    /// results are reproducible.
    pub seed: Option<u64>,
}

/// Seed used by the IPW and doubly-robust bootstraps when `seed` is `None`.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 42;

impl Default for IpwConfig {
    fn default() -> Self {
        Self {
            estimand: Estimand::ATE,
            trim: 0.05,
            bootstrap: 0,
            normalized: true,
            seed: None,
        }
    }
}

/// Configuration for doubly robust estimation.
#[derive(Debug, Clone)]
pub struct DoublyRobustConfig {
    /// Estimation method
    pub method: DRMethod,
    /// Target estimand (ATE or ATT)
    pub estimand: Estimand,
    /// Trimming threshold for propensity scores, in [0, 0.5)
    pub trim: f64,
    /// Number of bootstrap replications. Methods `IPW` and `Regression` take their
    /// standard errors from the bootstrap and need at least 2; `AIPW` uses the analytic
    /// influence function and ignores this field. A count whose result buffer cannot be
    /// allocated is refused with `EconError::ResourceLimit`.
    pub bootstrap: usize,
    /// Random seed for the bootstrap; `None` uses [`DEFAULT_BOOTSTRAP_SEED`] (42), so
    /// results are reproducible.
    pub seed: Option<u64>,
}

impl Default for DoublyRobustConfig {
    fn default() -> Self {
        Self {
            method: DRMethod::AIPW,
            estimand: Estimand::ATE,
            trim: 0.05,
            bootstrap: 999,
            seed: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Summary statistics for propensity scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityScoreSummary {
    /// Mean propensity score
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum
    pub min: f64,
    /// Maximum
    pub max: f64,
    /// Median
    pub median: f64,
    /// 10th percentile
    pub p10: f64,
    /// 90th percentile
    pub p90: f64,
}

/// Result from IPW treatment effect estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpwResult {
    /// Target estimand (ATE or ATT)
    pub estimand: Estimand,
    /// Estimated treatment effect
    pub effect: f64,
    /// Standard error: the M-estimation SE (`se_mest`) when `bootstrap_reps = 0`, else the
    /// bootstrap SD; `se_method` names it
    pub std_error: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// t-statistic (effect / std_error)
    pub t_stat: f64,
    /// p-value (two-tailed)
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Number of observations used (after trimming)
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of observations trimmed
    pub n_trimmed: usize,
    /// Propensity score summary statistics
    pub ps_summary: PropensityScoreSummary,
    /// Mean outcome in treated group
    pub mean_y_treated: f64,
    /// Mean outcome in control group
    pub mean_y_control: f64,
    /// Whether normalized (Hajek) weights were used
    pub normalized: bool,
    /// Trimming threshold used
    pub trim: f64,
    /// Number of bootstrap replications (0 when the SE is analytic)
    pub bootstrap_reps: usize,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
    /// How `std_error` was computed: "M-estimation (propensity estimated)", "efficient
    /// influence function", "propensity treated as known (outcome models unavailable)" or
    /// "bootstrap (B replicates)"
    #[serde(default)]
    pub se_method: String,
    /// M-estimation SE of the stacked logit-score and weighted-mean equations (the
    /// propensity estimation is accounted for; with trimming, the trimmed set is held
    /// fixed); `None` when the propensity information matrix is singular
    #[serde(default)]
    pub se_mest: Option<f64>,
    /// SE with the propensity treated as known (Hajek and ATT: HC0 of the weighted lm; HT:
    /// its influence function)
    #[serde(default)]
    pub se_naive: Option<f64>,
    /// Efficient-influence-function SE (outcome models per arm); `None` when an outcome
    /// model cannot be fitted
    #[serde(default)]
    pub se_eif: Option<f64>,
}

impl ResultNotes for IpwResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl ResultNotes for DoublyRobustResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for IpwResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IPW Treatment Effect Estimation")?;
        writeln!(f, "================================")?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Treatment Effect:")?;
        writeln!(f, "  Effect:     {:>12.4}", self.effect)?;
        writeln!(
            f,
            "  Std. Error: {:>12.4} ({})",
            self.std_error, self.se_method
        )?;
        writeln!(f, "  t-stat:     {:>12.2}", self.t_stat)?;
        writeln!(
            f,
            "  p-value:    {:>12.4}{}",
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI:     [{:.4}, {:.4}]",
            self.ci_lower, self.ci_upper
        )?;
        let fmt_se = |v: Option<f64>| v.map_or("not computed".to_string(), |v| format!("{v:.4}"));
        writeln!(
            f,
            "  Other SEs: efficient influence function {}; propensity treated as known {}",
            fmt_se(self.se_eif),
            fmt_se(self.se_naive)
        )?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Observations:  {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f, "  Trimmed:       {}", self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        writeln!(
            f,
            "  Mean:   {:.4}  Std.Dev: {:.4}",
            self.ps_summary.mean, self.ps_summary.std_dev
        )?;
        writeln!(
            f,
            "  Min:    {:.4}  Max:     {:.4}",
            self.ps_summary.min, self.ps_summary.max
        )?;
        writeln!(
            f,
            "  p10:    {:.4}  p90:     {:.4}",
            self.ps_summary.p10, self.ps_summary.p90
        )?;
        writeln!(f)?;
        writeln!(f, "Settings:")?;
        writeln!(
            f,
            "  Weights:    {}",
            if self.normalized {
                "Normalized (Hajek)"
            } else {
                "Horvitz-Thompson"
            }
        )?;
        writeln!(
            f,
            "  Trim:       {:.2}  Bootstrap: {} reps",
            self.trim, self.bootstrap_reps
        )?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        write!(f, "{}", notes_text(self))
    }
}

/// Result from doubly robust treatment effect estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoublyRobustResult {
    /// Estimation method used
    pub method: DRMethod,
    /// Target estimand
    pub estimand: Estimand,
    /// Estimated treatment effect
    pub effect: f64,
    /// Standard error: AIPW the analytic influence-function SE sd(IF)/sqrt(n); IPW and
    /// Regression the bootstrap SD; `se_method` names it
    pub std_error: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated
    pub n_treated: usize,
    /// Number of control
    pub n_control: usize,
    /// Number trimmed
    pub n_trimmed: usize,
    /// Propensity score summary
    pub ps_summary: PropensityScoreSummary,
    /// Outcome model R² for treated group
    pub outcome_r2_treated: f64,
    /// Outcome model R² for control group
    pub outcome_r2_control: f64,
    /// Trimming threshold
    pub trim: f64,
    /// Bootstrap replications (0 when the SE is analytic)
    pub bootstrap_reps: usize,
    /// Warnings
    pub warnings: Vec<String>,
    /// How `std_error` was computed: "analytic influence function" or "bootstrap (B
    /// replicates)"
    #[serde(default)]
    pub se_method: String,
}

impl fmt::Display for DoublyRobustResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Doubly Robust Treatment Effect Estimation")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Method:   {}", self.method)?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Treatment Effect:")?;
        writeln!(f, "  Effect:     {:>12.4}", self.effect)?;
        writeln!(
            f,
            "  Std. Error: {:>12.4} ({})",
            self.std_error, self.se_method
        )?;
        writeln!(f, "  t-stat:     {:>12.2}", self.t_stat)?;
        writeln!(
            f,
            "  p-value:    {:>12.4}{}",
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI:     [{:.4}, {:.4}]",
            self.ci_lower, self.ci_upper
        )?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Observations:  {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f, "  Trimmed:       {}", self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Model Fit:")?;
        writeln!(f, "  Outcome R² (Treated):  {:.4}", self.outcome_r2_treated)?;
        writeln!(f, "  Outcome R² (Control):  {:.4}", self.outcome_r2_control)?;
        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        writeln!(
            f,
            "  Mean: {:.4}  Range: [{:.4}, {:.4}]",
            self.ps_summary.mean, self.ps_summary.min, self.ps_summary.max
        )?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        write!(f, "{}", notes_text(self))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPW Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run IPW (Inverse Probability Weighting) treatment effect estimation.
///
/// Estimates ATE or ATT using propensity score weighting with optional trimming.
///
/// With `bootstrap = 0` (the default) `std_error` is the M-estimation SE of the stacked
/// logit-score and weighted-mean equations (`se_mest`; the propensity estimation is
/// accounted for, and a trimmed set is held fixed); the result also carries the
/// efficient-influence-function SE (`se_eif`) and the SE with the propensity treated as known
/// (`se_naive`). With `bootstrap >= 2` it is the bootstrap SD.
///
/// # Arguments
/// * `dataset` - The dataset containing outcome, treatment, and covariates
/// * `outcome` - Name of the outcome variable column
/// * `treatment` - Name of the binary treatment variable (0/1)
/// * `covariates` - Names of covariate columns for propensity score estimation
/// * `config` - Configuration options
///
/// # Model
///
/// For ATE with normalized (Hajek) weights:
/// ```text
/// ATE = Σ[D·Y/p(X)] / Σ[D/p(X)] - Σ[(1-D)·Y/(1-p(X))] / Σ[(1-D)/(1-p(X))]
/// ```
///
/// For ATT:
/// ```text
/// ATT = Σ[D·Y] / Σ[D] - Σ[(1-D)·p(X)·Y/(1-p(X))] / Σ[(1-D)·p(X)/(1-p(X))]
/// ```
///
/// # References
///
/// - Horvitz & Thompson (1952) for the original weighting estimator
/// - Hirano, Imbens & Ridder (2003) for efficient estimation of ATE
///
/// # Example
/// ```ignore
/// let config = IpwConfig::default();
/// let result = run_ipw_treatment(&dataset, "outcome", "treatment", &["x1", "x2"], config)?;
/// println!("{}", result);
/// ```
pub fn run_ipw_treatment(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    covariates: &[&str],
    config: IpwConfig,
) -> EconResult<IpwResult> {
    let mut warnings = Vec::new();

    validate_trim(config.trim)?;
    if config.bootstrap == 1 {
        return Err(EconError::InvalidSpecification {
            message: "bootstrap must be 0 (analytic influence-function SE) or at least 2; got 1"
                .to_string(),
        });
    }

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract outcome variable
    let y = DesignMatrix::extract_column(dataset.df(), outcome)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    // Extract treatment variable
    let d = DesignMatrix::extract_column(dataset.df(), treatment)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = y.len();

    // Validate treatment is binary
    let n_treated_orig: usize = d.iter().filter(|&&v| v >= 0.5).count();
    let n_control_orig = n - n_treated_orig;
    if n_treated_orig == 0 || n_control_orig == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must have both treated (1) and control (0) observations. Found {} treated, {} control.",
                treatment, n_treated_orig, n_control_orig
            ),
        });
    }

    // Build design matrix for propensity score model
    let design = DesignMatrix::from_dataframe(dataset.df(), covariates, true)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;

    // Estimate propensity scores using logit
    let ps_fit = fit_propensity(x.view(), d.view(), GlmLink::Logit)?;
    let ps = ps_fit.scores.mapv(|p| p.clamp(1e-10, 1.0 - 1e-10));
    for w in &ps_fit.fit.warnings {
        warnings.push(format!("propensity model: {w}"));
    }

    // Apply trimming
    let trim_lower = config.trim;
    let trim_upper = 1.0 - config.trim;

    let mut keep_idx: Vec<usize> = Vec::new();
    let mut n_trimmed = 0;

    for i in 0..n {
        let ps_i = ps[i];
        let should_keep = match config.estimand {
            Estimand::ATE => ps_i >= trim_lower && ps_i <= trim_upper,
            // For ATT, only trim the upper tail for treated, and both tails for control
            Estimand::ATT => {
                if d[i] >= 0.5 {
                    ps_i <= trim_upper
                } else {
                    ps_i >= trim_lower && ps_i <= trim_upper
                }
            }
        };

        if should_keep {
            keep_idx.push(i);
        } else {
            n_trimmed += 1;
        }
    }

    if keep_idx.len() < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: keep_idx.len(),
            context: "Too many observations trimmed".to_string(),
        });
    }

    // Create trimmed arrays
    let y_trim: Array1<f64> = keep_idx.iter().map(|&i| y[i]).collect();
    let d_trim: Array1<f64> = keep_idx.iter().map(|&i| d[i]).collect();
    let ps_trim: Array1<f64> = keep_idx.iter().map(|&i| ps[i]).collect();
    let x_trim = Array2::from_shape_fn((keep_idx.len(), x.ncols()), |(i, j)| x[[keep_idx[i], j]]);

    let n_trim = y_trim.len();
    let n_treated = d_trim.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n_trim - n_treated;

    // Check for sufficient overlap
    if n_treated < 5 || n_control < 5 {
        warnings.push(format!(
            "Small sample after trimming: {} treated, {} control",
            n_treated, n_control
        ));
    }

    // Compute point estimate
    let effect = compute_ipw_effect(
        &y_trim,
        &d_trim,
        &ps_trim,
        config.estimand,
        config.normalized,
    );

    // Compute propensity score summary
    let ps_summary = compute_ps_summary(&ps_trim);

    // Check for extreme propensity scores
    if ps_summary.min < 0.01 || ps_summary.max > 0.99 {
        warnings.push(format!(
            "Extreme propensity scores detected: min={:.4}, max={:.4}. Consider stronger trimming.",
            ps_summary.min, ps_summary.max
        ));
    }

    // Standard errors. All three analytic SEs are computed on every path.
    let keep_mask: Vec<bool> = {
        let mut m = vec![false; n];
        for &i in &keep_idx {
            m[i] = true;
        }
        m
    };
    let ps_cols: Vec<usize> = (0..x.ncols())
        .filter(|j| !ps_fit.fit.aliased.contains(j))
        .collect();
    let x_ps = x.select(ndarray::Axis(1), &ps_cols);
    let kind = IpwKind::of(config.estimand, config.normalized);
    let se_naive = ipw_naive_se(&y, &d, &ps, &keep_mask, kind, effect);
    let se_mest = match ipw_mest_se(x_ps.view(), &y, &d, &ps, &keep_mask, kind, effect) {
        Ok(v) => Some(v),
        Err(e @ EconError::SingularMatrix { .. }) => {
            warnings.push(format!(
                "M-estimation SE not computed: {e}; the efficient-influence-function SE is reported"
            ));
            None
        }
        Err(e) => return Err(e),
    };
    // Efficient influence function (Lunceford & Davidian 2004; Hahn 1998): the
    // efficiency-bound form, with mu_d(X) = E[Y | D = d, X] by OLS within each arm. With a
    // parametric logit the M-estimation SE is se_mest.
    let n_f = n_trim as f64;
    let x_names = design_names(covariates);
    let se_eif = {
        let mu1_fit = fit_outcome_model(&x_trim, &y_trim, &d_trim, true, &x_names);
        let mu0_fit = fit_outcome_model(&x_trim, &y_trim, &d_trim, false, &x_names);
        match (config.estimand, mu1_fit, mu0_fit) {
            (Estimand::ATE, Ok((mu1, _, _)), Ok((mu0, _, _))) => {
                let mut sum_eif_sq = 0.0;
                for i in 0..n_trim {
                    let ei = ps_trim[i];
                    let ipw_term = if d_trim[i] >= 0.5 {
                        (y_trim[i] - mu1[i]) / ei
                    } else {
                        -(y_trim[i] - mu0[i]) / (1.0 - ei)
                    };
                    let eif_i = ipw_term + mu1[i] - mu0[i] - effect;
                    sum_eif_sq += eif_i * eif_i;
                }
                Some((sum_eif_sq / (n_f * n_f)).sqrt())
            }
            (Estimand::ATT, _, Ok((mu0, _, _))) => {
                let p1 = n_treated as f64 / n_f;
                let mut sum_eif_sq = 0.0;
                for i in 0..n_trim {
                    let ei = ps_trim[i];
                    let resid0 = y_trim[i] - mu0[i];
                    let eif_i = if d_trim[i] >= 0.5 {
                        (resid0 - effect) / p1
                    } else {
                        -(ei / (1.0 - ei)) * resid0 / p1
                    };
                    sum_eif_sq += eif_i * eif_i;
                }
                Some((sum_eif_sq / (n_f * n_f)).sqrt())
            }
            (_, r1, r0) => {
                let e = r1
                    .err()
                    .or(r0.err())
                    .map(|e| e.to_string())
                    .unwrap_or_default();
                warnings.push(format!(
                    "efficient-influence-function SE not computed: an outcome model could not be fitted ({e})"
                ));
                None
            }
        }
    };

    let (std_error, ci_lower, ci_upper, se_method) = if config.bootstrap == 0 {
        let (se, method) = match (se_mest, se_eif) {
            (Some(v), _) => (v, "M-estimation (propensity estimated)".to_string()),
            (None, Some(v)) => (v, "efficient influence function".to_string()),
            (None, None) => {
                warnings.push(
                    "the standard error treats the propensity scores as known (neither the M-estimation nor the efficient-influence-function SE could be computed)".to_string(),
                );
                (
                    se_naive,
                    "propensity treated as known (outcome models unavailable)".to_string(),
                )
            }
        };
        let z = z_critical(0.95)?;
        (se, effect - z * se, effect + z * se, method)
    } else {
        // Bootstrap for standard errors using cached propensity scores (parallel via rayon)
        let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));

        let estimand = config.estimand;
        let normalized = config.normalized;
        let boot_effects: Vec<f64> = streamed_bootstrap(
            &mut rng,
            n_trim,
            config.bootstrap,
            boot_chunk_len(n_trim),
            |indices| {
                let y_boot: Array1<f64> = indices.iter().map(|&i| y_trim[i]).collect();
                let d_boot: Array1<f64> = indices.iter().map(|&i| d_trim[i]).collect();
                let ps_boot: Array1<f64> = indices.iter().map(|&i| ps_trim[i]).collect();

                let effect_boot =
                    compute_ipw_effect(&y_boot, &d_boot, &ps_boot, estimand, normalized);

                if effect_boot.is_finite() {
                    Some(effect_boot)
                } else {
                    None
                }
            },
        )?;

        if boot_effects.len() < config.bootstrap / 2 {
            warnings.push(
                "Many bootstrap iterations failed; standard errors may be unreliable".to_string(),
            );
        }
        let (se, lo, hi) = bootstrap_summary(&boot_effects, effect)?;
        (
            se,
            lo,
            hi,
            format!("bootstrap ({} replicates)", config.bootstrap),
        )
    };

    let t_stat = if std_error > 0.0 && std_error.is_finite() {
        effect / std_error
    } else {
        0.0
    };

    let p_value = normal_p_value_two_sided(t_stat);
    let significance = SignificanceLevel::from_p_value(p_value);

    // Compute group means
    let mean_y_treated = {
        let sum: f64 = y_trim
            .iter()
            .zip(d_trim.iter())
            .filter(|(_, di)| **di >= 0.5)
            .map(|(yi, _)| *yi)
            .sum();
        sum / n_treated as f64
    };

    let mean_y_control = {
        let sum: f64 = y_trim
            .iter()
            .zip(d_trim.iter())
            .filter(|(_, di)| **di < 0.5)
            .map(|(yi, _)| *yi)
            .sum();
        sum / n_control as f64
    };

    Ok(IpwResult {
        estimand: config.estimand,
        effect,
        std_error,
        ci_lower,
        ci_upper,
        t_stat,
        p_value,
        significance,
        n_obs: n_trim,
        n_treated,
        n_control,
        n_trimmed,
        ps_summary,
        mean_y_treated,
        mean_y_control,
        normalized: config.normalized,
        trim: config.trim,
        bootstrap_reps: config.bootstrap,
        warnings,
        se_method,
        se_mest,
        se_naive: Some(se_naive),
        se_eif,
    })
}

/// Refuses a trim outside [0, 0.5).
fn validate_trim(trim: f64) -> EconResult<()> {
    if trim.is_finite() && (0.0..0.5).contains(&trim) {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!(
                "trim must be in [0, 0.5) (the share trimmed from each tail of the propensity score); got {trim}"
            ),
        })
    }
}

/// Names of the propensity and outcome design columns (intercept first).
fn design_names(covariates: &[&str]) -> Vec<String> {
    std::iter::once("(Intercept)".to_string())
        .chain(covariates.iter().map(|c| c.to_string()))
        .collect()
}

/// Bootstrap SD and 95% interval: R's type-7 percentile interval when at least 20
/// replicates succeeded, else the normal interval.
fn bootstrap_summary(boot: &[f64], effect: f64) -> EconResult<(f64, f64, f64)> {
    let se = if boot.len() >= 2 {
        let mean = boot.iter().sum::<f64>() / boot.len() as f64;
        (boot.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (boot.len() - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    let (lo, hi) = if boot.len() >= 20 {
        percentile_interval(boot, 0.95)?
    } else {
        let z = z_critical(0.95)?;
        (effect - z * se, effect + z * se)
    };
    Ok((se, lo, hi))
}

/// The IPW estimator: Hajek ATE, Horvitz-Thompson ATE, or ATT (control weights e/(1-e)).
#[derive(Clone, Copy, PartialEq, Eq)]
enum IpwKind {
    Hajek,
    HorvitzThompson,
    Att,
}

impl IpwKind {
    fn of(estimand: Estimand, normalized: bool) -> Self {
        match (estimand, normalized) {
            (Estimand::ATT, _) => IpwKind::Att,
            (Estimand::ATE, true) => IpwKind::Hajek,
            (Estimand::ATE, false) => IpwKind::HorvitzThompson,
        }
    }

    /// Unit weight of the weighted-mean estimators.
    fn weight(self, d: f64, e: f64) -> f64 {
        let treated = d >= 0.5;
        match self {
            IpwKind::Att => {
                if treated {
                    1.0
                } else {
                    e / (1.0 - e)
                }
            }
            _ => {
                if treated {
                    1.0 / e
                } else {
                    1.0 / (1.0 - e)
                }
            }
        }
    }
}

/// Weighted arm means over the kept units: (mu1, mu0, sum of treated weights, of control
/// weights).
fn weighted_arm_means(
    y: &Array1<f64>,
    d: &Array1<f64>,
    e: &Array1<f64>,
    keep: &[bool],
    kind: IpwKind,
) -> (f64, f64, f64, f64) {
    let (mut s1, mut sy1, mut s0, mut sy0) = (0.0, 0.0, 0.0, 0.0);
    for i in (0..y.len()).filter(|&i| keep[i]) {
        let w = kind.weight(d[i], e[i]);
        if d[i] >= 0.5 {
            s1 += w;
            sy1 += w * y[i];
        } else {
            s0 += w;
            sy0 += w * y[i];
        }
    }
    (sy1 / s1, sy0 / s0, s1, s0)
}

/// SE with the propensity treated as known, over the kept units. Hajek and ATT: HC0 of the
/// weighted regression of y on (1, D); HT: sqrt(sum IF^2) / n_kept.
fn ipw_naive_se(
    y: &Array1<f64>,
    d: &Array1<f64>,
    e: &Array1<f64>,
    keep: &[bool],
    kind: IpwKind,
    effect: f64,
) -> f64 {
    let idx = (0..y.len()).filter(|&i| keep[i]);
    if kind == IpwKind::HorvitzThompson {
        let n_kept = keep.iter().filter(|&&k| k).count() as f64;
        let ss: f64 = idx
            .map(|i| {
                let h = if d[i] >= 0.5 {
                    y[i] / e[i]
                } else {
                    -y[i] / (1.0 - e[i])
                };
                (h - effect).powi(2)
            })
            .sum();
        return ss.sqrt() / n_kept;
    }
    let (mu1, mu0, s1, s0) = weighted_arm_means(y, d, e, keep, kind);
    let (mut v1, mut v0) = (0.0, 0.0);
    for i in idx {
        let w = kind.weight(d[i], e[i]);
        if d[i] >= 0.5 {
            v1 += (w * (y[i] - mu1)).powi(2);
        } else {
            v0 += (w * (y[i] - mu0)).powi(2);
        }
    }
    (v1 / (s1 * s1) + v0 / (s0 * s0)).sqrt()
}

/// M-estimation SE of the stacked logit-score and IPW estimating equations (the exact
/// sandwich; WeightIt::lm_weightit(vcov = "asympt") for Hajek and ATT). The logit score runs
/// over all n units, the IPW equations over the kept units (the trimmed set is held fixed).
/// `x` is the propensity design without its aliased columns and `e` the clipped scores.
fn ipw_mest_se(
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    d: &Array1<f64>,
    e: &Array1<f64>,
    keep: &[bool],
    kind: IpwKind,
    effect: f64,
) -> EconResult<f64> {
    let (n, p) = x.dim();
    let nf = n as f64;
    // I_bb = (1/n) sum e(1-e) x x'
    let mut info = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let v = e[i] * (1.0 - e[i]) / nf;
        for a in 0..p {
            let xa = x[[i, a]] * v;
            for b in 0..=a {
                info[[a, b]] += xa * x[[i, b]];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            info[[b, a]] = info[[a, b]];
        }
    }
    let inv = sym_inverse(
        info.view(),
        SingularPolicy::Error,
        "propensity information matrix",
    )?
    .inverse;
    // a . U_i with U_i = I_bb^-1 x_i (D_i - e_i), from ia = I_bb^-1 a (I_bb is symmetric).
    let u_dot = |i: usize, ia: &Array1<f64>| -> f64 { ia.dot(&x.row(i)) * (d[i] - e[i]) };
    let sum_if_sq = match kind {
        IpwKind::HorvitzThompson => {
            let n_kept = keep.iter().filter(|&&k| k).count() as f64;
            let mut a = Array1::<f64>::zeros(p);
            for i in (0..n).filter(|&i| keep[i]) {
                let c = if d[i] >= 0.5 {
                    -y[i] * (1.0 - e[i]) / e[i]
                } else {
                    -y[i] * e[i] / (1.0 - e[i])
                };
                a.scaled_add(c / nf, &x.row(i));
            }
            let s = n_kept / nf;
            let ia = inv.dot(&a);
            (0..n)
                .map(|i| {
                    let h = if keep[i] {
                        let hi = if d[i] >= 0.5 {
                            y[i] / e[i]
                        } else {
                            -y[i] / (1.0 - e[i])
                        };
                        hi - effect
                    } else {
                        0.0
                    };
                    ((h + u_dot(i, &ia)) / s).powi(2)
                })
                .sum::<f64>()
        }
        _ => {
            let (mu1, mu0, sw1, sw0) = weighted_arm_means(y, d, e, keep, kind);
            let (s1, s0) = (sw1 / nf, sw0 / nf);
            let mut a1 = Array1::<f64>::zeros(p);
            let mut a0 = Array1::<f64>::zeros(p);
            for i in (0..n).filter(|&i| keep[i]) {
                // d w_i / d beta = x_i * c
                if d[i] >= 0.5 {
                    let c = match kind {
                        IpwKind::Att => 0.0,
                        _ => -(1.0 - e[i]) / e[i],
                    };
                    a1.scaled_add(c * (y[i] - mu1) / nf, &x.row(i));
                } else {
                    let c = e[i] / (1.0 - e[i]);
                    a0.scaled_add(c * (y[i] - mu0) / nf, &x.row(i));
                }
            }
            let ia1 = inv.dot(&a1);
            let ia0 = inv.dot(&a0);
            (0..n)
                .map(|i| {
                    let (m1, m0) = if keep[i] {
                        let w = kind.weight(d[i], e[i]);
                        if d[i] >= 0.5 {
                            (w * (y[i] - mu1), 0.0)
                        } else {
                            (0.0, w * (y[i] - mu0))
                        }
                    } else {
                        (0.0, 0.0)
                    };
                    let f = (m1 + u_dot(i, &ia1)) / s1 - (m0 + u_dot(i, &ia0)) / s0;
                    f * f
                })
                .sum::<f64>()
        }
    };
    let se = sum_if_sq.sqrt() / nf;
    if se.is_finite() {
        Ok(se)
    } else {
        Err(EconError::Computation(
            "the M-estimation standard error is not finite".to_string(),
        ))
    }
}

/// At most this many bootstrap indices (8 MiB) exist at once.
const BOOT_CHUNK_INDICES: usize = 1 << 20;
/// At most this many bootstrap index sets exist at once.
const BOOT_CHUNK_MAX: usize = 256;

/// Number of bootstrap index sets of size `n` drawn per chunk.
fn boot_chunk_len(n: usize) -> usize {
    (BOOT_CHUNK_INDICES / n.max(1)).clamp(1, BOOT_CHUNK_MAX)
}

/// Draws `reps` bootstrap index sets of size `n` from `rng`, in replicate order and with the
/// same draws as building all sets up front, holding at most `chunk` sets at a time. `stat`
/// is evaluated in parallel within a chunk. Returns the `Some` values in replicate order.
///
/// The result buffer is reserved fallibly (an unallocatable `reps` is
/// `EconError::ResourceLimit`), and the installed cooperative-cancellation budget is checked
/// before every chunk and before every replicate (on the rayon worker).
fn streamed_bootstrap<F>(
    rng: &mut StdRng,
    n: usize,
    reps: usize,
    chunk: usize,
    stat: F,
) -> EconResult<Vec<f64>>
where
    F: Fn(&[usize]) -> Option<f64> + Sync,
{
    let mut out = resource::try_with_capacity::<f64>(reps, "bootstrap replications")?;
    let budget = resource::current_budget();
    let chunk = chunk.max(1);
    let mut done = 0;
    while done < reps {
        resource::checkpoint()?;
        let m = chunk.min(reps - done);
        // Sequential draws from the single stream keep seeded results identical to
        // materialising every index set first.
        let sets: Vec<Vec<usize>> = (0..m)
            .map(|_| (0..n).map(|_| rng.gen_range(0..n)).collect())
            .collect();
        let values = sets
            .par_iter()
            .map(|s| {
                if let Some(b) = &budget {
                    b.check()?;
                }
                Ok(stat(s))
            })
            .collect::<EconResult<Vec<Option<f64>>>>()?;
        out.extend(values.into_iter().flatten());
        done += m;
    }
    Ok(out)
}

/// Compute IPW treatment effect estimate.
///
/// Uses Hajek (normalized) or Horvitz-Thompson weights.
fn compute_ipw_effect(
    y: &Array1<f64>,
    d: &Array1<f64>,
    ps: &Array1<f64>,
    estimand: Estimand,
    normalized: bool,
) -> f64 {
    let n = y.len();

    match estimand {
        Estimand::ATE => {
            // ATE = E[Y(1)] - E[Y(0)]
            // Using IPW: weighted mean of treated - weighted mean of control

            let mut sum_treated_y = 0.0;
            let mut sum_treated_w = 0.0;
            let mut sum_control_y = 0.0;
            let mut sum_control_w = 0.0;

            for i in 0..n {
                let di = d[i];
                let yi = y[i];
                let ps_i = ps[i].max(1e-10).min(1.0 - 1e-10);

                if di >= 0.5 {
                    // Treated: weight = 1/p(X)
                    let w = 1.0 / ps_i;
                    sum_treated_y += w * yi;
                    sum_treated_w += w;
                } else {
                    // Control: weight = 1/(1-p(X))
                    let w = 1.0 / (1.0 - ps_i);
                    sum_control_y += w * yi;
                    sum_control_w += w;
                }
            }

            if normalized {
                // Hajek estimator: normalize by sum of weights
                let mean_treated = if sum_treated_w > 0.0 {
                    sum_treated_y / sum_treated_w
                } else {
                    0.0
                };
                let mean_control = if sum_control_w > 0.0 {
                    sum_control_y / sum_control_w
                } else {
                    0.0
                };
                mean_treated - mean_control
            } else {
                // Horvitz-Thompson: normalize by sample size
                (sum_treated_y - sum_control_y) / n as f64
            }
        }
        Estimand::ATT => {
            // ATT = E[Y(1) - Y(0) | D=1]
            // = E[Y | D=1] - E[Y(0) | D=1]
            // The second term is estimated using control group weighted by p(X)/(1-p(X))

            let mut sum_treated_y = 0.0;
            let mut n_treated = 0.0;
            let mut sum_control_y = 0.0;
            let mut sum_control_w = 0.0;

            for i in 0..n {
                let di = d[i];
                let yi = y[i];
                let ps_i = ps[i].max(1e-10).min(1.0 - 1e-10);

                if di >= 0.5 {
                    // Treated: simple average
                    sum_treated_y += yi;
                    n_treated += 1.0;
                } else {
                    // Control: weight = p(X)/(1-p(X))
                    let w = ps_i / (1.0 - ps_i);
                    sum_control_y += w * yi;
                    sum_control_w += w;
                }
            }

            let mean_treated = if n_treated > 0.0 {
                sum_treated_y / n_treated
            } else {
                0.0
            };
            let mean_control_weighted = if sum_control_w > 0.0 {
                sum_control_y / sum_control_w
            } else {
                0.0
            };

            mean_treated - mean_control_weighted
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Doubly Robust Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run doubly robust (AIPW) treatment effect estimation.
///
/// Combines inverse probability weighting with outcome regression for double robustness.
/// The estimator is consistent if either the propensity score model or the outcome model
/// is correctly specified (but not necessarily both).
///
/// # Arguments
/// * `dataset` - The dataset
/// * `outcome` - Name of the outcome variable
/// * `treatment` - Name of the binary treatment variable
/// * `covariates` - Names of covariate columns
/// * `config` - Configuration options
///
/// # Model (AIPW for ATE)
///
/// ```text
/// τ_AIPW = (1/n) Σᵢ [
///     μ̂⁽¹⁾(Xᵢ) - μ̂⁽⁰⁾(Xᵢ)
///     + Dᵢ/ê(Xᵢ) · (Yᵢ - μ̂⁽¹⁾(Xᵢ))
///     - (1-Dᵢ)/(1-ê(Xᵢ)) · (Yᵢ - μ̂⁽⁰⁾(Xᵢ))
/// ]
/// ```
///
/// # References
///
/// - Robins, Rotnitzky & Zhao (1994) for the AIPW estimator
/// - Bang & Robins (2005) for doubly robust estimation
pub fn run_doubly_robust(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    covariates: &[&str],
    config: DoublyRobustConfig,
) -> EconResult<DoublyRobustResult> {
    let mut warnings = Vec::new();

    validate_trim(config.trim)?;
    if matches!(config.method, DRMethod::IPW | DRMethod::Regression) && config.bootstrap < 2 {
        let method = if config.method == DRMethod::IPW {
            "ipw"
        } else {
            "regression"
        };
        return Err(EconError::InvalidSpecification {
            message: format!(
                "bootstrap must be at least 2 for method {method} (its standard errors come from the bootstrap); got {}",
                config.bootstrap
            ),
        });
    }

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract variables
    let y = DesignMatrix::extract_column(dataset.df(), outcome)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let d = DesignMatrix::extract_column(dataset.df(), treatment)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = y.len();

    // Validate treatment
    let n_treated_orig: usize = d.iter().filter(|&&v| v >= 0.5).count();
    let n_control_orig = n - n_treated_orig;
    if n_treated_orig == 0 || n_control_orig == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable must have both treated and control observations. Found {} treated, {} control.",
                n_treated_orig, n_control_orig
            ),
        });
    }

    // Build design matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), covariates, true)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;

    // Estimate propensity scores
    let ps_fit = fit_propensity(x.view(), d.view(), GlmLink::Logit)?;
    let ps = ps_fit.scores.mapv(|p| p.clamp(1e-10, 1.0 - 1e-10));
    for w in &ps_fit.fit.warnings {
        warnings.push(format!("propensity model: {w}"));
    }

    // Apply trimming
    let trim_lower = config.trim;
    let trim_upper = 1.0 - config.trim;

    let mut keep_idx: Vec<usize> = Vec::new();
    let mut n_trimmed = 0;

    for i in 0..n {
        if ps[i] >= trim_lower && ps[i] <= trim_upper {
            keep_idx.push(i);
        } else {
            n_trimmed += 1;
        }
    }

    if keep_idx.len() < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: keep_idx.len(),
            context: "Too many observations trimmed".to_string(),
        });
    }

    // Create trimmed arrays
    let y_trim: Array1<f64> = keep_idx.iter().map(|&i| y[i]).collect();
    let d_trim: Array1<f64> = keep_idx.iter().map(|&i| d[i]).collect();
    let ps_trim: Array1<f64> = keep_idx.iter().map(|&i| ps[i]).collect();

    let n_trim = y_trim.len();

    // Build trimmed design matrix
    let mut x_trim = Array2::zeros((n_trim, x.ncols()));
    for (new_i, &old_i) in keep_idx.iter().enumerate() {
        x_trim.row_mut(new_i).assign(&x.row(old_i));
    }

    let n_treated = d_trim.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n_trim - n_treated;

    // Fit outcome models for treated and control separately
    let x_names = design_names(covariates);
    let (mu_1, r2_treated, alias1) = fit_outcome_model(&x_trim, &y_trim, &d_trim, true, &x_names)?;
    let (mu_0, r2_control, alias0) = fit_outcome_model(&x_trim, &y_trim, &d_trim, false, &x_names)?;
    for (group, a) in [("treated", alias1), ("control", alias0)] {
        if let Some(text) = a {
            warnings.push(format!("outcome model ({group}): {text}"));
        }
    }

    // Compute point estimate based on method
    let effect = match config.method {
        DRMethod::AIPW => {
            compute_aipw_effect(&y_trim, &d_trim, &ps_trim, &mu_1, &mu_0, config.estimand)
        }
        DRMethod::IPW => compute_ipw_effect(&y_trim, &d_trim, &ps_trim, config.estimand, true),
        DRMethod::Regression => {
            // Simple difference in predicted means
            let mean_mu1: f64 = mu_1.iter().sum::<f64>() / n_trim as f64;
            let mean_mu0: f64 = mu_0.iter().sum::<f64>() / n_trim as f64;
            mean_mu1 - mean_mu0
        }
    };

    // Propensity score summary
    let ps_summary = compute_ps_summary(&ps_trim);

    // Compute standard error, CI, and inference.
    // For AIPW: use analytic influence function variance (no bootstrap needed).
    // For IPW/Regression: use parallel bootstrap with cached propensity scores.
    let (std_error, ci_lower, ci_upper, se_method, bootstrap_reps) = match config.method {
        DRMethod::AIPW => {
            // Analytic influence-function SE sd(IF) / sqrt(n) (n - 1 denominator, as the R
            // AIPW and tmle packages; Robins, Rotnitzky & Zhao 1994). ATE: IF = psi; ATT:
            // IF_i = (phi_i - D_i tau) n / n1 with phi = D (Y - m0) - (1-D) e/(1-e) (Y - m0).
            let n_f = n_trim as f64;
            let n1 = n_treated as f64;
            let infl: Vec<f64> = (0..n_trim)
                .map(|i| {
                    let (di, yi, ei) = (d_trim[i], y_trim[i], ps_trim[i]);
                    match config.estimand {
                        Estimand::ATE => {
                            let t = if di >= 0.5 {
                                (yi - mu_1[i]) / ei
                            } else {
                                -(yi - mu_0[i]) / (1.0 - ei)
                            };
                            mu_1[i] - mu_0[i] + t
                        }
                        Estimand::ATT => {
                            let phi = if di >= 0.5 {
                                yi - mu_0[i]
                            } else {
                                -ei / (1.0 - ei) * (yi - mu_0[i])
                            };
                            let dt = if di >= 0.5 { effect } else { 0.0 };
                            (phi - dt) * n_f / n1
                        }
                    }
                })
                .collect();
            let mean = infl.iter().sum::<f64>() / n_f;
            let var = infl.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
            let se = (var / n_f).sqrt();
            let z = z_critical(0.95)?;
            (
                se,
                effect - z * se,
                effect + z * se,
                "analytic influence function".to_string(),
                0,
            )
        }
        DRMethod::IPW | DRMethod::Regression => {
            // Parallel bootstrap with cached propensity scores for IPW/Regression.
            // Index sets are drawn sequentially (one bounded chunk at a time) for
            // reproducibility, then evaluated in parallel reusing cached ps/outcome models.
            let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));

            let method = config.method;
            let estimand = config.estimand;

            let boot_effects: Vec<f64> = streamed_bootstrap(
                &mut rng,
                n_trim,
                config.bootstrap,
                boot_chunk_len(n_trim),
                |indices| {
                    let y_boot: Array1<f64> = indices.iter().map(|&i| y_trim[i]).collect();
                    let d_boot: Array1<f64> = indices.iter().map(|&i| d_trim[i]).collect();

                    // Check bootstrap sample has both treated and control
                    let n_t_boot = d_boot.iter().filter(|&&v| v >= 0.5).count();
                    if n_t_boot == 0 || n_t_boot == n_trim {
                        return None;
                    }

                    // The outer match arm restricts `method` to IPW or Regression;
                    // an `if/else` keeps that contract explicit and removes the
                    // `unreachable!()` AIPW arm that used to be a maintenance trap.
                    let effect_boot = if method == DRMethod::IPW {
                        // Reuse cached propensity scores (resample them).
                        let ps_boot: Array1<f64> = indices.iter().map(|&i| ps_trim[i]).collect();
                        compute_ipw_effect(&y_boot, &d_boot, &ps_boot, estimand, true)
                    } else {
                        // Reuse cached outcome model predictions (resample them).
                        let mu_1_boot: Array1<f64> = indices.iter().map(|&i| mu_1[i]).collect();
                        let mu_0_boot: Array1<f64> = indices.iter().map(|&i| mu_0[i]).collect();
                        let mean_mu1: f64 = mu_1_boot.iter().sum::<f64>() / n_trim as f64;
                        let mean_mu0: f64 = mu_0_boot.iter().sum::<f64>() / n_trim as f64;
                        mean_mu1 - mean_mu0
                    };

                    if effect_boot.is_finite() {
                        Some(effect_boot)
                    } else {
                        None
                    }
                },
            )?;

            if boot_effects.len() < config.bootstrap / 2 {
                warnings.push(
                    "Many bootstrap iterations failed; standard errors may be unreliable"
                        .to_string(),
                );
            }

            let (se, lo, hi) = bootstrap_summary(&boot_effects, effect)?;
            (
                se,
                lo,
                hi,
                format!("bootstrap ({} replicates)", config.bootstrap),
                config.bootstrap,
            )
        }
    };

    let t_stat = if std_error > 0.0 && std_error.is_finite() {
        effect / std_error
    } else {
        0.0
    };

    let p_value = normal_p_value_two_sided(t_stat);
    let significance = SignificanceLevel::from_p_value(p_value);

    Ok(DoublyRobustResult {
        method: config.method,
        estimand: config.estimand,
        effect,
        std_error,
        ci_lower,
        ci_upper,
        t_stat,
        p_value,
        significance,
        n_obs: n_trim,
        n_treated,
        n_control,
        n_trimmed,
        ps_summary,
        outcome_r2_treated: r2_treated,
        outcome_r2_control: r2_control,
        trim: config.trim,
        bootstrap_reps,
        warnings,
        se_method,
    })
}

/// Compute AIPW (Augmented IPW) treatment effect estimate.
///
/// Combines IPW with outcome regression for double robustness.
fn compute_aipw_effect(
    y: &Array1<f64>,
    d: &Array1<f64>,
    ps: &Array1<f64>,
    mu_1: &Array1<f64>,
    mu_0: &Array1<f64>,
    estimand: Estimand,
) -> f64 {
    let n = y.len();

    match estimand {
        Estimand::ATE => {
            // AIPW for ATE:
            // τ = (1/n) Σ [μ̂₁(X) - μ̂₀(X) + D(Y - μ̂₁(X))/p(X) - (1-D)(Y - μ̂₀(X))/(1-p(X))]
            let mut sum = 0.0;

            for i in 0..n {
                let di = d[i];
                let yi = y[i];
                let ps_i = ps[i].max(1e-10).min(1.0 - 1e-10);
                let mu1_i = mu_1[i];
                let mu0_i = mu_0[i];

                // Outcome model component
                let outcome_term = mu1_i - mu0_i;

                // IPW augmentation terms
                let ipw_treated = if di >= 0.5 { (yi - mu1_i) / ps_i } else { 0.0 };

                let ipw_control = if di < 0.5 {
                    (yi - mu0_i) / (1.0 - ps_i)
                } else {
                    0.0
                };

                sum += outcome_term + ipw_treated - ipw_control;
            }

            sum / n as f64
        }
        Estimand::ATT => {
            // AIPW for ATT:
            // τ = (1/n₁) Σ [D(Y - μ̂₀(X)) - (1-D)p(X)(Y - μ̂₀(X))/(1-p(X))]
            let n_treated: f64 = d.iter().filter(|&&v| v >= 0.5).count() as f64;
            if n_treated == 0.0 {
                return 0.0;
            }

            let mut sum = 0.0;

            for i in 0..n {
                let di = d[i];
                let yi = y[i];
                let ps_i = ps[i].max(1e-10).min(1.0 - 1e-10);
                let mu0_i = mu_0[i];

                if di >= 0.5 {
                    sum += yi - mu0_i;
                } else {
                    sum -= ps_i * (yi - mu0_i) / (1.0 - ps_i);
                }
            }

            sum / n_treated
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Fit outcome model for a treatment group (OLS regression by pivoted QR, `lm.fit`).
///
/// Returns predicted outcomes for every row (aliased columns count as 0, as
/// `stats::predict.lm`), the group's R² and the aliasing warning, if any.
fn fit_outcome_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    d: &Array1<f64>,
    treated: bool,
    names: &[String],
) -> EconResult<(Array1<f64>, f64, Option<String>)> {
    let n = y.len();
    let indices: Vec<usize> = (0..n)
        .filter(|&i| if treated { d[i] >= 0.5 } else { d[i] < 0.5 })
        .collect();

    if indices.len() < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: indices.len(),
            context: format!(
                "Not enough observations in {} group for outcome model",
                if treated { "treated" } else { "control" }
            ),
        });
    }

    let x_group = x.select(ndarray::Axis(0), &indices);
    let y_group: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
    let fit = least_squares(x_group.view(), y_group.view())?;
    let beta = fit.coefficients.mapv(|b| if b.is_nan() { 0.0 } else { b });
    let mu: Array1<f64> = x.dot(&beta);

    // R² for the group
    let y_group_mean = y_group.mean().unwrap_or(0.0);
    let sst: f64 = y_group.iter().map(|&yi| (yi - y_group_mean).powi(2)).sum();
    let ssr: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    Ok((mu, r2, fit.aliasing_warning(names)))
}

/// Compute propensity score summary statistics.
fn compute_ps_summary(ps: &Array1<f64>) -> PropensityScoreSummary {
    let n = ps.len();
    if n == 0 {
        return PropensityScoreSummary {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p10: 0.0,
            p90: 0.0,
        };
    }

    let mean = ps.mean().unwrap_or(0.0);

    let var: f64 = ps.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let std_dev = var.sqrt();

    let mut sorted: Vec<f64> = ps.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);

    let min = sorted[0];
    let max = sorted[n - 1];
    // R quantile() / median(), type 7
    let median = quantile_sorted(&sorted, 0.5, QuantileType::T7);
    let p10 = quantile_sorted(&sorted, 0.1, QuantileType::T7);
    let p90 = quantile_sorted(&sorted, 0.9, QuantileType::T7);

    PropensityScoreSummary {
        mean,
        std_dev,
        min,
        max,
        median,
        p10,
        p90,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use polars::prelude::*;

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

    fn spec_message(err: EconError) -> String {
        match err {
            EconError::InvalidSpecification { message } => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_estimand_parses_aliases_and_rejects_unknown() {
        Estimand::choices().validate().unwrap();
        assert_eq!("ATE".parse::<Estimand>().unwrap(), Estimand::ATE);
        assert_eq!("att".parse::<Estimand>().unwrap(), Estimand::ATT);
        let m = spec_message("atc".parse::<Estimand>().unwrap_err());
        assert!(m.contains("estimand") && m.contains("ate, att"), "{m}");
    }

    #[test]
    fn test_dr_method_parses_aliases_and_rejects_unknown() {
        DRMethod::choices().validate().unwrap();
        for (s, m) in [
            ("AIPW", DRMethod::AIPW),
            ("dr", DRMethod::AIPW),
            ("doubly-robust", DRMethod::AIPW),
            ("ipw", DRMethod::IPW),
            ("reg", DRMethod::Regression),
            ("Regression", DRMethod::Regression),
            ("outcome_regression", DRMethod::Regression),
        ] {
            assert_eq!(s.parse::<DRMethod>().unwrap(), m, "{s}");
        }
        let m = spec_message("tmle".parse::<DRMethod>().unwrap_err());
        assert!(
            m.contains("method") && m.contains("aipw, ipw, regression"),
            "{m}"
        );
    }

    #[test]
    fn test_streamed_bootstrap_matches_materialised_draws() {
        // stat rejects some sets so the filtering path is covered
        let stat = |idx: &[usize]| {
            if idx[0].is_multiple_of(5) {
                None
            } else {
                Some(idx.iter().map(|&i| ((i * 7919) % 101) as f64).sum::<f64>())
            }
        };
        let mut rng = StdRng::seed_from_u64(7);
        let sets: Vec<Vec<usize>> = (0..101)
            .map(|_| (0..37).map(|_| rng.gen_range(0..37)).collect())
            .collect();
        let reference: Vec<f64> = sets.iter().filter_map(|s| stat(s)).collect();
        assert!(
            reference.len() < 101,
            "precondition: some sets are rejected"
        );
        for chunk in [1, 8, 101, 1000] {
            let mut rng = StdRng::seed_from_u64(7);
            assert_eq!(
                streamed_bootstrap(&mut rng, 37, 101, chunk, stat).unwrap(),
                reference,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn test_bootstrap_count_is_an_error_not_an_abort() {
        let ds = create_treatment_dataset();
        let err = run_ipw_treatment(
            &ds,
            "y",
            "treatment",
            &["x1", "x2"],
            IpwConfig {
                trim: 0.01,
                bootstrap: usize::MAX,
                seed: Some(42),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
        for method in [DRMethod::IPW, DRMethod::Regression] {
            let err = run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method,
                    estimand: Estimand::ATE,
                    trim: 0.01,
                    bootstrap: usize::MAX,
                    seed: Some(42),
                },
            )
            .unwrap_err();
            assert!(
                matches!(err, EconError::ResourceLimit(_)),
                "{method:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn test_bootstrap_needs_two_replicates() {
        let ds = create_treatment_dataset();
        let err = run_ipw_treatment(
            &ds,
            "y",
            "treatment",
            &["x1", "x2"],
            IpwConfig {
                trim: 0.01,
                bootstrap: 1,
                seed: Some(42),
                ..Default::default()
            },
        )
        .unwrap_err();
        expect_invalid_spec(err, "bootstrap");
        for method in [DRMethod::IPW, DRMethod::Regression] {
            for b in [0, 1] {
                let err = run_doubly_robust(
                    &ds,
                    "y",
                    "treatment",
                    &["x1", "x2"],
                    DoublyRobustConfig {
                        method,
                        estimand: Estimand::ATE,
                        trim: 0.01,
                        bootstrap: b,
                        seed: Some(42),
                    },
                )
                .unwrap_err();
                expect_invalid_spec(err, "bootstrap");
            }
        }
        // AIPW uses the influence function and ignores `bootstrap`
        assert!(
            run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method: DRMethod::AIPW,
                    estimand: Estimand::ATE,
                    trim: 0.01,
                    bootstrap: 0,
                    seed: None,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn test_bootstrap_honours_cancellation() {
        let ds = create_treatment_dataset();
        let ipw = resource::with_budget(cancelled_budget(), || {
            run_ipw_treatment(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                IpwConfig {
                    trim: 0.01,
                    bootstrap: 50,
                    seed: Some(42),
                    ..Default::default()
                },
            )
        });
        assert!(matches!(ipw, Err(EconError::Cancelled(_))), "got {ipw:?}");
        let dr = resource::with_budget(cancelled_budget(), || {
            run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method: DRMethod::Regression,
                    estimand: Estimand::ATE,
                    trim: 0.01,
                    bootstrap: 50,
                    seed: Some(42),
                },
            )
        });
        assert!(matches!(dr, Err(EconError::Cancelled(_))), "got {dr:?}");
    }

    /// Create a dataset with known treatment effect and good overlap.
    ///
    /// DGP designed to have overlapping propensity scores:
    /// - Treatment assignment is stochastic (not perfectly determined by X)
    /// - Y = 0.5 * D + 0.3 * x1 + noise
    /// - True ATE ≈ 0.5
    fn create_treatment_dataset() -> Dataset {
        // Dataset with overlapping covariate distributions for treated/control
        // This ensures propensity scores aren't at extreme values (0 or 1)
        let df = df! {
            // Outcome: treated group has ~0.5 higher mean after controlling for x
            "y" => [
                // Treated group (D=1): mean ~1.0 + 0.5 treatment effect
                1.5, 1.8, 1.2, 2.1, 1.7, 2.3, 1.4, 2.0, 1.6, 1.9,
                2.2, 2.5, 1.8, 2.7, 2.0, 2.4, 1.9, 2.6, 2.1, 2.3,
                // Control group (D=0): similar x distribution but lower y
                0.8, 1.2, 0.9, 1.5, 1.1, 1.4, 0.7, 1.3, 1.0, 1.2,
                1.6, 1.9, 1.3, 2.0, 1.5, 1.8, 1.4, 2.1, 1.6, 1.7
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // Overlapping covariate distribution (treated slightly higher on average)
            "x1" => [
                0.3, 0.5, 0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.35, 0.55,
                0.9, 1.1, 0.6, 1.2, 0.75, 1.0, 0.65, 1.15, 0.85, 0.95,
                // Control has overlapping range (slightly lower on average)
                0.1, 0.4, 0.2, 0.6, 0.3, 0.5, 0.0, 0.45, 0.25, 0.35,
                0.7, 0.9, 0.5, 1.0, 0.6, 0.8, 0.55, 0.95, 0.65, 0.75
            ],
            "x2" => [
                0.5, 0.6, 0.4, 0.7, 0.55, 0.75, 0.45, 0.65, 0.5, 0.6,
                0.8, 0.9, 0.7, 0.95, 0.75, 0.85, 0.72, 0.92, 0.78, 0.82,
                0.4, 0.55, 0.45, 0.65, 0.5, 0.6, 0.35, 0.58, 0.48, 0.52,
                0.7, 0.85, 0.6, 0.9, 0.68, 0.78, 0.62, 0.88, 0.72, 0.75
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_ipw_ate_basic() {
        let dataset = create_treatment_dataset();
        let config = IpwConfig {
            estimand: Estimand::ATE,
            trim: 0.01,     // Lower trim for test data
            bootstrap: 100, // Fewer for faster tests
            normalized: true,
            seed: Some(42),
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Check basic structure (40 observations, may have some trimmed)
        assert!(
            result.n_obs >= 30,
            "Should have at least 30 obs, got {}",
            result.n_obs
        );
        assert!(
            result.n_treated >= 15,
            "Should have at least 15 treated, got {}",
            result.n_treated
        );
        assert!(
            result.n_control >= 15,
            "Should have at least 15 control, got {}",
            result.n_control
        );

        // Treatment effect should be positive (around 0.5)
        assert!(
            result.effect > 0.0,
            "ATE should be positive, got {}",
            result.effect
        );
        assert!(
            result.effect < 2.0,
            "ATE should be reasonable, got {}",
            result.effect
        );

        // Standard error should be positive and finite
        assert!(result.std_error > 0.0 && result.std_error.is_finite());
    }

    #[test]
    fn test_ipw_att_basic() {
        let dataset = create_treatment_dataset();
        let config = IpwConfig {
            estimand: Estimand::ATT,
            trim: 0.01,
            bootstrap: 100,
            normalized: true,
            seed: Some(42),
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // ATT should also be positive for this dataset
        assert!(
            result.effect > 0.0,
            "ATT should be positive, got {}",
            result.effect
        );
    }

    #[test]
    fn test_doubly_robust_aipw() {
        let dataset = create_treatment_dataset();
        let config = DoublyRobustConfig {
            method: DRMethod::AIPW,
            estimand: Estimand::ATE,
            trim: 0.01,
            bootstrap: 100,
            seed: Some(42),
        };

        let result = run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Check structure
        assert!(result.n_obs >= 30, "Should have at least 30 obs");
        assert_eq!(result.method, DRMethod::AIPW);

        // Effect should be positive
        assert!(
            result.effect > 0.0,
            "AIPW effect should be positive, got {}",
            result.effect
        );

        // Outcome model R² should be reasonable
        assert!(result.outcome_r2_treated >= 0.0 && result.outcome_r2_treated <= 1.0);
        assert!(result.outcome_r2_control >= 0.0 && result.outcome_r2_control <= 1.0);
    }

    #[test]
    fn test_doubly_robust_ipw_only() {
        let dataset = create_treatment_dataset();
        let config = DoublyRobustConfig {
            method: DRMethod::IPW,
            estimand: Estimand::ATE,
            trim: 0.01,
            bootstrap: 100,
            seed: Some(42),
        };

        let result = run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();
        assert_eq!(result.method, DRMethod::IPW);
        assert!(result.effect > 0.0);
    }

    #[test]
    fn test_doubly_robust_regression_only() {
        let dataset = create_treatment_dataset();
        let config = DoublyRobustConfig {
            method: DRMethod::Regression,
            estimand: Estimand::ATE,
            trim: 0.01,
            bootstrap: 100,
            seed: Some(42),
        };

        let result = run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();
        assert_eq!(result.method, DRMethod::Regression);
        assert!(result.effect > 0.0);
    }

    #[test]
    fn test_ipw_refuses_non_binary_treatment() {
        // the shared propensity fit refuses a treatment not coded 0/1 (0.7 counted as treated)
        let ds = create_treatment_dataset();
        let mut df = ds.df().clone();
        let mut t = DesignMatrix::extract_column(&df, "treatment")
            .unwrap()
            .to_vec();
        t[3] = 0.7;
        df.with_column(Series::new("treatment".into(), t)).unwrap();
        let ds = Dataset::new(df);
        expect_invalid_spec(
            run_ipw_treatment(&ds, "y", "treatment", &["x1", "x2"], IpwConfig::default())
                .unwrap_err(),
            "0/1",
        );
        expect_invalid_spec(
            run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig::default(),
            )
            .unwrap_err(),
            "0/1",
        );
    }

    #[test]
    fn test_seeded_bootstrap_unchanged() {
        // Pinned from the pre-streaming implementation at 1fdb22f, seed 42. The propensity
        // scores from glm::fit_propensity since T3-WP23 differ at 1e-13 from that fit's, so
        // the standard errors are compared at 1e-9 relative. The interval is the type-7
        // percentile interval of the same draws since T3-WP23 (CAUS-111; it was effect
        // +- qnorm(0.975) SE).
        const IPW_SE: f64 = 0.11590718872111475;
        const IPW_CI: (f64, f64) = (0.25515833348085837, 0.6746370453016588);
        const DR_IPW_SE: f64 = 0.11590718872111475;
        const DR_REG_SE: f64 = 0.011033127368161828;
        let ds = create_treatment_dataset();
        let ipw = run_ipw_treatment(
            &ds,
            "y",
            "treatment",
            &["x1", "x2"],
            IpwConfig {
                estimand: Estimand::ATE,
                trim: 0.01,
                bootstrap: 200,
                normalized: true,
                seed: Some(42),
            },
        )
        .unwrap();
        assert!(((ipw.std_error - IPW_SE) / IPW_SE).abs() < 1e-9);
        assert!(((ipw.ci_lower - IPW_CI.0) / IPW_CI.0).abs() < 1e-9);
        assert!(((ipw.ci_upper - IPW_CI.1) / IPW_CI.1).abs() < 1e-9);
        for (method, se) in [
            (DRMethod::IPW, DR_IPW_SE),
            (DRMethod::Regression, DR_REG_SE),
        ] {
            let dr = run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method,
                    estimand: Estimand::ATE,
                    trim: 0.01,
                    bootstrap: 200,
                    seed: Some(42),
                },
            )
            .unwrap();
            assert!(((dr.std_error - se) / se).abs() < 1e-9, "{method:?}");
        }
    }

    #[test]
    fn test_propensity_score_summary() {
        let ps = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let summary = compute_ps_summary(&ps);

        assert!((summary.mean - 0.5).abs() < 0.01);
        assert!((summary.min - 0.1).abs() < 0.01);
        assert!((summary.max - 0.9).abs() < 0.01);
        assert!((summary.median - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_missing_column_error() {
        let dataset = create_treatment_dataset();
        let config = IpwConfig::default();

        let result = run_ipw_treatment(&dataset, "nonexistent", "treatment", &["x1"], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_ipw_result() {
        let dataset = create_treatment_dataset();
        let config = IpwConfig {
            bootstrap: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("IPW Treatment Effect"));
        assert!(output.contains("Effect:"));
        assert!(output.contains("Std. Error:"));
    }

    #[test]
    fn test_display_dr_result() {
        let dataset = create_treatment_dataset();
        let config = DoublyRobustConfig {
            bootstrap: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Doubly Robust"));
        assert!(output.contains("AIPW"));
    }

    // =========================================================================
    // R Validation Tests (Phase 5)
    // =========================================================================

    /// Simple LCG for deterministic random numbers
    fn lcg_rand(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Create validation dataset with known treatment effect.
    /// True ATE is designed to be approximately 0.75.
    fn create_ipw_validation_dataset() -> Dataset {
        let n = 500;
        let mut seed: u64 = 42;

        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate covariates ~ Normal(0, 1) via Box-Muller
            let u1 = lcg_rand(&mut seed).max(1e-10);
            let u2 = lcg_rand(&mut seed);
            let z1 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            let z2 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).sin();
            x1.push(z1);
            x2.push(z2);

            // Propensity score model
            let ps_true = 1.0 / (1.0 + (-(-0.5 + 0.6 * z1 + 0.3 * z2)).exp());
            let t = if lcg_rand(&mut seed) < ps_true {
                1.0
            } else {
                0.0
            };
            treatment.push(t);

            // Outcome model with treatment effect of 0.75
            let u3 = lcg_rand(&mut seed).max(1e-10);
            let u4 = lcg_rand(&mut seed);
            let noise =
                0.5 * ((-2.0_f64 * u3.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u4).cos();
            let y_i = 1.0 + 0.5 * z1 - 0.3 * z2 + 0.75 * t + noise;
            y.push(y_i);
        }

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_ipw_propensity_summary_in_range() {
        // Validate propensity score estimation
        let dataset = create_ipw_validation_dataset();
        let config = IpwConfig {
            bootstrap: 50,
            seed: Some(42),
            trim: 0.01,
            ..Default::default()
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // PS summary should have sensible values
        let ps = &result.ps_summary;
        assert!(
            ps.mean > 0.2 && ps.mean < 0.8,
            "PS mean {:.4} seems extreme",
            ps.mean
        );
        assert!(
            ps.std_dev > 0.0 && ps.std_dev < 0.5,
            "PS SD {:.4} seems extreme",
            ps.std_dev
        );
        assert!(ps.min >= 0.0 && ps.min < 0.5);
        assert!(ps.max > 0.5 && ps.max <= 1.0);
        assert!(ps.p10 < ps.p90);
    }

    #[test]
    fn test_dr_ipw_matches_ipw() {
        // With equal trims, the DR IPW (Hajek) effect is the IPW effect
        let dataset = create_ipw_validation_dataset();

        let ipw_config = IpwConfig {
            estimand: Estimand::ATE,
            bootstrap: 200,
            seed: Some(42),
            trim: 0.01,
            ..Default::default()
        };
        let ipw_result =
            run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], ipw_config).unwrap();

        let dr_config = DoublyRobustConfig {
            method: DRMethod::IPW,
            estimand: Estimand::ATE,
            bootstrap: 200,
            seed: Some(42),
            trim: 0.01,
        };
        let dr_result =
            run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], dr_config).unwrap();

        assert!(
            (ipw_result.effect - dr_result.effect).abs() <= 1e-12,
            "IPW {:.4} and DR-IPW {:.4} should be similar",
            ipw_result.effect,
            dr_result.effect
        );
    }

    #[test]
    fn test_dr_regression_recovers_dgp() {
        // Validate regression-only approach
        let dataset = create_ipw_validation_dataset();

        let config = DoublyRobustConfig {
            method: DRMethod::Regression,
            estimand: Estimand::ATE,
            bootstrap: 200,
            seed: Some(42),
            trim: 0.01,
        };

        let result = run_doubly_robust(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Regression-only should also estimate effect reasonably
        assert!(
            (result.effect - 0.75).abs() < 0.35,
            "Regression ATE {:.4} should be close to 0.75",
            result.effect
        );

        assert_eq!(result.method, DRMethod::Regression);
    }

    #[test]
    fn test_ipw_trimming_reduces_sample() {
        // Test that trimming affects results appropriately
        let dataset = create_ipw_validation_dataset();

        // Without trimming
        let config_no_trim = IpwConfig {
            trim: 0.0,
            bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };
        let result_no_trim =
            run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config_no_trim).unwrap();

        // With aggressive trimming
        let config_trim = IpwConfig {
            trim: 0.1,
            bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };
        let result_trim =
            run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config_trim).unwrap();

        // Trimmed version should have fewer observations
        assert!(result_trim.n_trimmed >= result_no_trim.n_trimmed);
        assert!(result_trim.n_obs <= result_no_trim.n_obs);

        // Both should give reasonable estimates
        assert!((result_no_trim.effect - 0.75).abs() < 0.5);
        assert!((result_trim.effect - 0.75).abs() < 0.5);
    }

    #[test]
    fn test_ipw_t_stat_and_p_value_consistent() {
        // t = effect / se and p = 2 Phi(-|t|), unconditionally.
        let dataset = create_ipw_validation_dataset();
        let config = IpwConfig {
            bootstrap: 300,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        assert!(result.std_error > 0.0 && result.std_error.is_finite());
        let expected_t = result.effect / result.std_error;
        assert!((result.t_stat - expected_t).abs() <= 1e-12 * expected_t.abs().max(1.0));
        let want = 2.0
            * statrs::distribution::Normal::new(0.0, 1.0)
                .unwrap()
                .sf(expected_t.abs());
        assert!(
            (result.p_value - want).abs() <= 1e-12,
            "p {} want {want}",
            result.p_value
        );
    }

    // ── T2-WP13: the R case causal_mediation_na and the shared fixtures ──

    #[test]
    fn test_validate_causal_na_fixture() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let df = ds.df();
        // the dtypes the T2-WP13 tests rely on
        assert_eq!(df.column("d").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("d_na").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("d_txt").unwrap().dtype(), &DataType::String);
        assert_eq!(df.column("x1").unwrap().dtype(), &DataType::Float64);
        let cc = na_fixtures::complete_rows(&case, &ds);
        Check::new(&case)
            .scalar("n_complete", cc.df().height() as f64, Tol::exact())
            .scalar(
                "missing_y",
                df.column("y").unwrap().null_count() as f64,
                Tol::exact(),
            )
            .scalar(
                "missing_m",
                df.column("m").unwrap().null_count() as f64,
                Tol::exact(),
            )
            .scalar(
                "missing_x1",
                df.column("x1").unwrap().null_count() as f64,
                Tol::exact(),
            )
            .scalar(
                "missing_d_na",
                df.column("d_na").unwrap().null_count() as f64,
                Tol::exact(),
            )
            .finish();
        for c in ["y", "d", "m", "x1", "x2"] {
            assert_eq!(cc.df().column(c).unwrap().null_count(), 0, "{c}");
        }
    }

    #[test]
    fn test_validate_na_ipw_columns() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, y: &str, d: &str, x: &[&str]| {
            run_ipw_treatment(ds, y, d, x, IpwConfig::default())
        };
        na_fixtures::assert_refusals(&case, &ds, true, run);
        // the effect and SE are compared across treatment dtypes here; the R pins are the
        // test_validate_ipw_* tests
        na_fixtures::assert_treatment_dtypes_agree(&case, &ds, run, |r| {
            vec![r.effect, r.std_error]
        });
    }

    #[test]
    fn test_validate_na_doubly_robust_columns() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, y: &str, d: &str, x: &[&str]| {
            run_doubly_robust(ds, y, d, x, DoublyRobustConfig::default())
        };
        na_fixtures::assert_refusals(&case, &ds, true, run);
        na_fixtures::assert_treatment_dtypes_agree(&case, &ds, run, |r| {
            vec![r.effect, r.std_error]
        });
    }

    #[test]
    fn test_validate_na_ipw_nan_outcome_is_missing() {
        // A NaN cell is missing, like a null.
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let data = na_fixtures::with_nan(&na_fixtures::complete_rows(&case, &ds), "y", 0);
        match run_ipw_treatment(&data, "y", "d", &["x1", "x2"], IpwConfig::default()) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("y", 1));
            }
            other => panic!("NaN outcome: expected NullValues {{ y, 1 }}, got {other:?}"),
        }
    }

    // ── T3-WP23 Task 6: IPW and AIPW standard errors ─────────────────────────────────

    use p2a_validation::{Check, Tol};
    use statrs::distribution::ContinuousCDF;

    const COVS: [&str; 3] = ["x1", "x2", "x3"];
    const Z975: f64 = 1.9599639845400536; // R qnorm(0.975)

    /// The case is loaded in each test body.
    fn ipw_fit(ds: &Dataset, est: Estimand, normalized: bool) -> IpwResult {
        let c = IpwConfig {
            estimand: est,
            normalized,
            trim: 0.0, // R uses every unit; bootstrap 0 (analytic)
            ..Default::default()
        };
        run_ipw_treatment(ds, "y", "treat", &COVS, c).unwrap()
    }

    fn assert_inference(effect: f64, se: f64, lo: f64, hi: f64, t: f64, p: f64) {
        assert!(
            (lo - (effect - Z975 * se)).abs() < 1e-12 && (hi - (effect + Z975 * se)).abs() < 1e-12
        );
        assert!((t - effect / se).abs() < 1e-12);
        let want = 2.0
            * statrs::distribution::Normal::new(0.0, 1.0)
                .unwrap()
                .sf(t.abs());
        // the crate's normal tail (chi-square form) and statrs agree to ~5e-11 relative far
        // in the tail (p ~ 1e-31)
        assert!(
            (p - want).abs() <= 1e-9 * want.max(1e-300),
            "p {p} want {want}"
        );
    }

    #[test]
    fn test_validate_ipw_hajek_se() {
        // R: glm propensity; Hajek IPW; HC0 of the weighted lm; exact M-estimation sandwich =
        // WeightIt::lm_weightit(vcov = "asympt") (case ipw_aipw_sim). std_error is the
        // M-estimation SE; the EIF value is se_eif.
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let r = ipw_fit(&ds, Estimand::ATE, true);
        let n = r.n_obs;
        assert_eq!(r.se_method, "M-estimation (propensity estimated)");
        assert_eq!(r.se_mest, Some(r.std_error));
        Check::new(&case)
            .scalar("ipw_ate_hajek", r.effect, Tol::coef(n))
            .scalar("ipw_ate_se_mest", r.std_error, Tol::se(n))
            .scalar("ipw_ate_se_eif", r.se_eif.unwrap(), Tol::se(n))
            .scalar("ipw_ate_se_naive", r.se_naive.unwrap(), Tol::se(n))
            .scalar("ipw_ate_se_mest_weightit", r.std_error, Tol::new(1e-6, 0.0))
            .finish();
        assert_inference(
            r.effect,
            r.std_error,
            r.ci_lower,
            r.ci_upper,
            r.t_stat,
            r.p_value,
        );
    }

    #[test]
    fn test_validate_ipw_ht_se() {
        // R: Horvitz-Thompson IPW, its stacked M-estimation sandwich, the EIF and naive SEs
        // (case ipw_aipw_sim). Fails before: std_error was the EIF value.
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let r = ipw_fit(&ds, Estimand::ATE, false);
        let n = r.n_obs;
        Check::new(&case)
            .scalar("ipw_ate_ht", r.effect, Tol::coef(n))
            .scalar("ipw_ate_ht_se_mest", r.std_error, Tol::se(n))
            .scalar("ipw_ate_ht_se_eif", r.se_eif.unwrap(), Tol::se(n))
            .scalar("ipw_ate_ht_se_naive", r.se_naive.unwrap(), Tol::se(n))
            .finish();
        assert_inference(
            r.effect,
            r.std_error,
            r.ci_lower,
            r.ci_upper,
            r.t_stat,
            r.p_value,
        );
    }

    #[test]
    fn test_validate_ipw_att_se() {
        // R: ATT IPW (control weights e/(1-e)); M-estimation = lm_weightit(vcov = "asympt")
        // (case ipw_aipw_sim). Fails before: std_error was the EIF value.
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let r = ipw_fit(&ds, Estimand::ATT, true);
        let n = r.n_obs;
        Check::new(&case)
            .scalar("ipw_att", r.effect, Tol::coef(n))
            .scalar("ipw_att_se_mest", r.std_error, Tol::se(n))
            .scalar("ipw_att_se_eif", r.se_eif.unwrap(), Tol::se(n))
            .scalar("ipw_att_se_naive", r.se_naive.unwrap(), Tol::se(n))
            .scalar("ipw_att_se_mest_weightit", r.std_error, Tol::new(1e-6, 0.0))
            .finish();
        assert_inference(
            r.effect,
            r.std_error,
            r.ci_lower,
            r.ci_upper,
            r.t_stat,
            r.p_value,
        );
    }

    #[test]
    fn test_ipw_mest_se_with_a_fixed_trimmed_set() {
        // The M-estimation SE holds the trimmed set fixed (Crump et al. 2009): a trim that
        // drops nothing changes nothing; a trim that drops units still gives se_mest.
        let ds = create_treatment_dataset();
        let fit = |trim: f64| {
            run_ipw_treatment(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                IpwConfig {
                    trim,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (r0, r1) = (fit(0.0), fit(1e-9));
        assert_eq!(r1.n_trimmed, 0);
        assert!(((r1.std_error - r0.std_error) / r0.std_error).abs() < 1e-15);
        let r = fit(0.1);
        assert!(r.n_trimmed > 0);
        assert_eq!(r.se_mest, Some(r.std_error));
        assert!(r.std_error.is_finite() && r.std_error > 0.0);
        assert!(r.se_eif.is_some());
    }

    #[test]
    fn test_validate_aipw() {
        // AIPW with OLS outcome models per arm; SE = sd(psi)/sqrt(n). Fails before: the SE
        // lacked the sqrt(n/(n-1)) factor and bootstrap_reps reported the configured 999.
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let c = DoublyRobustConfig {
            trim: 0.0, // AIPW; bootstrap 999 is ignored
            ..Default::default()
        };
        let r = run_doubly_robust(&ds, "y", "treat", &COVS, c).unwrap();
        let n = r.n_obs;
        assert_eq!(
            (r.bootstrap_reps, r.se_method.as_str()),
            (0, "analytic influence function")
        );
        Check::new(&case)
            .scalar("aipw_ate", r.effect, Tol::coef(n))
            .scalar("aipw_ate_se", r.std_error, Tol::se(n))
            .scalar("outcome_r2_treated", r.outcome_r2_treated, Tol::stat(n))
            .scalar("outcome_r2_control", r.outcome_r2_control, Tol::stat(n))
            .finish();
        assert_inference(
            r.effect,
            r.std_error,
            r.ci_lower,
            r.ci_upper,
            r.t_stat,
            r.p_value,
        );
    }

    #[test]
    fn test_validate_aipw_att() {
        // AIPW ATT, SE = sd(IF)/sqrt(n) with IF_i = (phi_i - D_i tau) n/n1 (case ipw_aipw_sim)
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let c = DoublyRobustConfig {
            estimand: Estimand::ATT,
            trim: 0.0,
            ..Default::default()
        };
        let r = run_doubly_robust(&ds, "y", "treat", &COVS, c).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("aipw_att", r.effect, Tol::coef(n))
            .scalar("aipw_att_se", r.std_error, Tol::se(n))
            .finish();
        assert_inference(
            r.effect,
            r.std_error,
            r.ci_lower,
            r.ci_upper,
            r.t_stat,
            r.p_value,
        );
    }

    #[test]
    fn test_validate_dr_regression() {
        // Regression adjustment mean(m1 - m0) (case ipw_aipw_sim); bootstrap SE, not pinned
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let c = DoublyRobustConfig {
            method: DRMethod::Regression,
            bootstrap: 200,
            seed: Some(42),
            trim: 0.0,
            ..Default::default()
        };
        let r = run_doubly_robust(&ds, "y", "treat", &COVS, c).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("ra_ate", r.effect, Tol::coef(n))
            .finish();
        assert_eq!(r.se_method, "bootstrap (200 replicates)");
        assert_eq!(r.bootstrap_reps, 200);
        assert!(r.std_error.is_finite());
    }

    #[test]
    fn test_dr_reports_se_method() {
        let ds = create_treatment_dataset();
        let dr = |method, bootstrap| {
            run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method,
                    bootstrap,
                    seed: Some(42),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let a = dr(DRMethod::AIPW, 999);
        assert_eq!(
            (a.bootstrap_reps, a.se_method.as_str()),
            (0, "analytic influence function")
        );
        let b = dr(DRMethod::IPW, 50);
        assert_eq!(b.se_method, "bootstrap (50 replicates)");
        let ipw = |trim| {
            run_ipw_treatment(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                IpwConfig {
                    trim,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let i = ipw(0.05);
        assert_eq!(i.se_method, "M-estimation (propensity estimated)");
        assert!(i.se_mest.is_some() && i.se_eif.is_some() && i.se_naive.is_some());
        let t = ipw(0.4);
        assert!(t.n_trimmed > 0);
        assert_eq!(t.se_mest, Some(t.std_error));
        assert!(t.se_eif.is_some() && t.se_naive.is_some());
    }

    #[test]
    fn test_unseeded_bootstrap_is_reproducible() {
        // R13: seed None uses DEFAULT_BOOTSTRAP_SEED. Fails before (OS entropy).
        let ds = create_treatment_dataset();
        let ipw = |seed| {
            run_ipw_treatment(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                IpwConfig {
                    bootstrap: 100,
                    seed,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (a, b, c) = (ipw(None), ipw(None), ipw(Some(DEFAULT_BOOTSTRAP_SEED)));
        for r in [&b, &c] {
            assert_eq!(
                (r.std_error, r.ci_lower, r.ci_upper),
                (a.std_error, a.ci_lower, a.ci_upper)
            );
        }
        let dr = |seed| {
            run_doubly_robust(
                &ds,
                "y",
                "treatment",
                &["x1", "x2"],
                DoublyRobustConfig {
                    method: DRMethod::Regression,
                    bootstrap: 100,
                    seed,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (a, b, c) = (dr(None), dr(None), dr(Some(DEFAULT_BOOTSTRAP_SEED)));
        for r in [&b, &c] {
            assert_eq!(
                (r.std_error, r.ci_lower, r.ci_upper),
                (a.std_error, a.ci_lower, a.ci_upper)
            );
        }
    }

    #[test]
    fn test_ipw_rejects_bad_trim() {
        // R3
        let ds = create_treatment_dataset();
        for trim in [-0.1, 0.5, f64::NAN] {
            let c = IpwConfig {
                trim,
                ..Default::default()
            };
            expect_invalid_spec(
                run_ipw_treatment(&ds, "y", "treatment", &["x1", "x2"], c).unwrap_err(),
                "trim",
            );
            let c = DoublyRobustConfig {
                trim,
                ..Default::default()
            };
            expect_invalid_spec(
                run_doubly_robust(&ds, "y", "treatment", &["x1", "x2"], c).unwrap_err(),
                "trim",
            );
        }
    }
}

/// Test-only fixtures shared by the T2-WP13 tests of the mediation, weighting, matching,
/// TMLE and standardisation modules. They read the R case `causal_mediation_na`
/// (`validation/r/cases/causal_mediation_na.R`): 400 rows with missing `y`, `m` and `x1`
/// cells, an Int64 0/1 treatment `d`, a treatment with missing values `d_na` and a text
/// treatment `d_txt`.
#[cfg(test)]
pub(super) mod na_fixtures {
    use crate::data::Dataset;
    use crate::data::columns::MissingSummary;
    use crate::errors::{EconError, EconResult};
    use p2a_validation::{Case, Tol};
    use polars::prelude::*;

    /// The rows of case value `complete_rows` (R's `complete.cases` over y, d, m, x1, x2),
    /// in order.
    pub(crate) fn complete_rows(case: &Case, ds: &Dataset) -> Dataset {
        let idx = IdxCa::from_vec(
            "rows".into(),
            case.vector("complete_rows")
                .iter()
                .map(|&r| r as IdxSize)
                .collect(),
        );
        Dataset::new(ds.df().take(&idx).expect("complete_rows are row indices"))
    }

    /// `ds` with column `col` cast to `dtype`.
    pub(crate) fn with_cast(ds: &Dataset, col: &str, dtype: DataType) -> Dataset {
        let mut df = ds.df().clone();
        let cast = df
            .column(col)
            .expect("fixture column")
            .cast(&dtype)
            .expect("fixture cast");
        df.with_column(cast).expect("same height");
        Dataset::new(df)
    }

    /// `ds` with column `col` as Float64 and a NaN at `row` (the one Rust-built NaN fixture:
    /// the R harness cannot write a NaN).
    pub(crate) fn with_nan(ds: &Dataset, col: &str, row: usize) -> Dataset {
        let mut df = ds.df().clone();
        let values: Vec<Option<f64>> = df
            .column(col)
            .expect("fixture column")
            .cast(&DataType::Float64)
            .expect("numeric fixture column")
            .f64()
            .expect("Float64 after the cast")
            .iter()
            .enumerate()
            .map(|(i, v)| if i == row { Some(f64::NAN) } else { v })
            .collect();
        df.with_column(Column::new(col.into(), values))
            .expect("same height");
        Dataset::new(df)
    }

    /// `ds` with its rows in reverse order.
    pub(crate) fn reversed(ds: &Dataset) -> Dataset {
        Dataset::new(ds.df().reverse())
    }

    /// The missing-cell count of `col` in `m`; 0 when the column has none.
    pub(crate) fn by_column(m: &MissingSummary, col: &str) -> f64 {
        m.by_column
            .iter()
            .find(|(c, _)| c == col)
            .map_or(0.0, |(_, k)| *k as f64)
    }

    /// The bit patterns of `v`, for bit-identity assertions.
    pub(crate) fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn count(case: &Case, key: &str) -> usize {
        case.scalar(key) as usize
    }

    fn expect_null<T: std::fmt::Debug>(step: &str, r: EconResult<T>, col: &str, k: usize) {
        match r {
            Err(EconError::NullValues { column, count }) if column == col && count == k => {}
            other => panic!(
                "{step}: expected NullValues {{ column: {col:?}, count: {k} }}, got {other:?}"
            ),
        }
    }

    /// The refusals every Fail-policy estimator of T2-WP13 must give on the case data,
    /// in this order (each fails at extraction, so no estimation runs):
    /// 1. if `has_outcome`: outcome `y` -> `NullValues { "y", missing_y }`;
    /// 2. treatment `d_na` -> `NullValues { "d_na", missing_d_na }`;
    /// 3. covariates `x1, x2` -> `NullValues { "x1", missing_x1 }`;
    /// 4. treatment `d_txt` -> `NonNumericColumn { "d_txt" }`;
    /// 5. treatment `dd` -> `ColumnNotFound { "dd", available }` with `d` available.
    ///
    /// `run(ds, outcome, treatment, covariates)`; estimators without an outcome ignore it.
    pub(crate) fn assert_refusals<T: std::fmt::Debug>(
        case: &Case,
        ds: &Dataset,
        has_outcome: bool,
        run: impl Fn(&Dataset, &str, &str, &[&str]) -> EconResult<T>,
    ) {
        if has_outcome {
            expect_null(
                "step 1 (outcome y)",
                run(ds, "y", "d", &["x2"]),
                "y",
                count(case, "missing_y"),
            );
        }
        expect_null(
            "step 2 (treatment d_na)",
            run(ds, "x2", "d_na", &["x2"]),
            "d_na",
            count(case, "missing_d_na"),
        );
        expect_null(
            "step 3 (covariate x1)",
            run(ds, "x2", "d", &["x1", "x2"]),
            "x1",
            count(case, "missing_x1"),
        );
        match run(ds, "x2", "d_txt", &["x2"]) {
            Err(EconError::NonNumericColumn { column }) if column == "d_txt" => {}
            other => panic!(
                "step 4 (text treatment d_txt): expected NonNumericColumn {{ d_txt }}, \
                 got {other:?}"
            ),
        }
        match run(ds, "x2", "dd", &["x2"]) {
            Err(EconError::ColumnNotFound { column, available })
                if column == "dd" && available.iter().any(|c| c == "d") => {}
            other => panic!(
                "step 5 (absent treatment dd): expected ColumnNotFound {{ dd }} listing d, \
                 got {other:?}"
            ),
        }
    }

    /// On the complete rows, `run(.., "y", "d", ["x1", "x2"])` with `d` as loaded (Int64),
    /// cast to Float64 and cast to Boolean gives the same `key` within
    /// `Tol::new(1e-12, 1e-14)` (not bit equality: matching and the bootstraps use rayon).
    pub(crate) fn assert_treatment_dtypes_agree<T>(
        case: &Case,
        ds: &Dataset,
        run: impl Fn(&Dataset, &str, &str, &[&str]) -> EconResult<T>,
        key: impl Fn(&T) -> Vec<f64>,
    ) {
        let cc = complete_rows(case, ds);
        assert_eq!(cc.df().column("d").unwrap().dtype(), &DataType::Int64);
        let fit = |data: &Dataset, label: &str| match run(data, "y", "d", &["x1", "x2"]) {
            Ok(r) => key(&r),
            Err(e) => panic!("treatment d as {label}: {e:?}"),
        };
        let base = fit(&cc, "Int64");
        assert!(!base.is_empty(), "key gave no values");
        let tol = Tol::new(1e-12, 1e-14);
        for dtype in [DataType::Float64, DataType::Boolean] {
            let label = format!("{dtype}");
            let other = fit(&with_cast(&cc, "d", dtype), &label);
            assert_eq!(other.len(), base.len(), "{label}: key length");
            for (i, (&a, &b)) in other.iter().zip(&base).enumerate() {
                assert!(
                    tol.allows(a, b),
                    "{label}: key[{i}] = {a:?}, Int64 gave {b:?}"
                );
            }
        }
    }
}
