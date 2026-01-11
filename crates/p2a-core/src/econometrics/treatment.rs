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

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{logistic_cdf, normal_cdf, SignificanceLevel};

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

/// Configuration for IPW treatment effect estimation.
#[derive(Debug, Clone)]
pub struct IpwConfig {
    /// Target estimand (ATE or ATT)
    pub estimand: Estimand,
    /// Trimming threshold for propensity scores (default: 0.05)
    /// Observations with p(X) < trim or p(X) > 1-trim are excluded
    pub trim: f64,
    /// Number of bootstrap replications for standard errors (default: 999)
    pub bootstrap: usize,
    /// Use normalized (Hajek) weights (default: true)
    /// If false, uses Horvitz-Thompson weights
    pub normalized: bool,
    /// Random seed for bootstrap (optional, for reproducibility)
    pub seed: Option<u64>,
}

impl Default for IpwConfig {
    fn default() -> Self {
        Self {
            estimand: Estimand::ATE,
            trim: 0.05,
            bootstrap: 999,
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
    /// Trimming threshold for propensity scores
    pub trim: f64,
    /// Number of bootstrap replications
    pub bootstrap: usize,
    /// Random seed for bootstrap
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
    /// Standard error (via bootstrap)
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
    /// Number of bootstrap replications
    pub bootstrap_reps: usize,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
}

impl fmt::Display for IpwResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IPW Treatment Effect Estimation")?;
        writeln!(f, "================================")?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Treatment Effect:")?;
        writeln!(f, "  Effect:     {:>12.4}", self.effect)?;
        writeln!(f, "  Std. Error: {:>12.4}", self.std_error)?;
        writeln!(f, "  t-stat:     {:>12.2}", self.t_stat)?;
        writeln!(f, "  p-value:    {:>12.4}{}", self.p_value, self.significance.stars())?;
        writeln!(f, "  95% CI:     [{:.4}, {:.4}]", self.ci_lower, self.ci_upper)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Observations:  {} (Treated: {}, Control: {})",
                 self.n_obs, self.n_treated, self.n_control)?;
        writeln!(f, "  Trimmed:       {}", self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        writeln!(f, "  Mean:   {:.4}  Std.Dev: {:.4}", self.ps_summary.mean, self.ps_summary.std_dev)?;
        writeln!(f, "  Min:    {:.4}  Max:     {:.4}", self.ps_summary.min, self.ps_summary.max)?;
        writeln!(f, "  p10:    {:.4}  p90:     {:.4}", self.ps_summary.p10, self.ps_summary.p90)?;
        writeln!(f)?;
        writeln!(f, "Settings:")?;
        writeln!(f, "  Weights:    {}", if self.normalized { "Normalized (Hajek)" } else { "Horvitz-Thompson" })?;
        writeln!(f, "  Trim:       {:.2}  Bootstrap: {} reps", self.trim, self.bootstrap_reps)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        Ok(())
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
    /// Standard error (via bootstrap)
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
    /// Bootstrap replications
    pub bootstrap_reps: usize,
    /// Warnings
    pub warnings: Vec<String>,
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
        writeln!(f, "  Std. Error: {:>12.4}", self.std_error)?;
        writeln!(f, "  t-stat:     {:>12.2}", self.t_stat)?;
        writeln!(f, "  p-value:    {:>12.4}{}", self.p_value, self.significance.stars())?;
        writeln!(f, "  95% CI:     [{:.4}, {:.4}]", self.ci_lower, self.ci_upper)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Observations:  {} (Treated: {}, Control: {})",
                 self.n_obs, self.n_treated, self.n_control)?;
        writeln!(f, "  Trimmed:       {}", self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Model Fit:")?;
        writeln!(f, "  Outcome R² (Treated):  {:.4}", self.outcome_r2_treated)?;
        writeln!(f, "  Outcome R² (Control):  {:.4}", self.outcome_r2_control)?;
        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        writeln!(f, "  Mean: {:.4}  Range: [{:.4}, {:.4}]",
                 self.ps_summary.mean, self.ps_summary.min, self.ps_summary.max)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPW Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run IPW (Inverse Probability Weighting) treatment effect estimation.
///
/// Estimates ATE or ATT using propensity score weighting with optional trimming.
/// Standard errors are computed via bootstrap.
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

    // Extract outcome variable
    let y = DesignMatrix::extract_column(dataset.df(), outcome).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract treatment variable
    let d = DesignMatrix::extract_column(dataset.df(), treatment).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

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
    let design = DesignMatrix::from_dataframe(dataset.df(), covariates, true)?;
    let x = design.data;

    // Estimate propensity scores using logit
    let ps = estimate_propensity_scores(&x, &d)?;

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
    let effect = compute_ipw_effect(&y_trim, &d_trim, &ps_trim, config.estimand, config.normalized);

    // Compute propensity score summary
    let ps_summary = compute_ps_summary(&ps_trim);

    // Check for extreme propensity scores
    if ps_summary.min < 0.01 || ps_summary.max > 0.99 {
        warnings.push(format!(
            "Extreme propensity scores detected: min={:.4}, max={:.4}. Consider stronger trimming.",
            ps_summary.min, ps_summary.max
        ));
    }

    // Bootstrap for standard errors
    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut boot_effects: Vec<f64> = Vec::with_capacity(config.bootstrap);

    for _ in 0..config.bootstrap {
        // Resample with replacement
        let indices: Vec<usize> = (0..n_trim).map(|_| rng.gen_range(0..n_trim)).collect();

        let y_boot: Array1<f64> = indices.iter().map(|&i| y_trim[i]).collect();
        let d_boot: Array1<f64> = indices.iter().map(|&i| d_trim[i]).collect();
        let ps_boot: Array1<f64> = indices.iter().map(|&i| ps_trim[i]).collect();

        let effect_boot = compute_ipw_effect(&y_boot, &d_boot, &ps_boot, config.estimand, config.normalized);

        if effect_boot.is_finite() {
            boot_effects.push(effect_boot);
        }
    }

    if boot_effects.len() < config.bootstrap / 2 {
        warnings.push("Many bootstrap iterations failed; standard errors may be unreliable".to_string());
    }

    // Compute standard error and confidence intervals
    let std_error = if !boot_effects.is_empty() {
        let mean_boot: f64 = boot_effects.iter().sum::<f64>() / boot_effects.len() as f64;
        let var_boot: f64 = boot_effects.iter().map(|&e| (e - mean_boot).powi(2)).sum::<f64>()
            / (boot_effects.len() - 1).max(1) as f64;
        var_boot.sqrt()
    } else {
        f64::NAN
    };

    // Sort for percentile CI
    boot_effects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ci_lower = if boot_effects.len() >= 20 {
        let idx = (boot_effects.len() as f64 * 0.025).floor() as usize;
        boot_effects[idx]
    } else {
        effect - 1.96 * std_error
    };

    let ci_upper = if boot_effects.len() >= 20 {
        let idx = (boot_effects.len() as f64 * 0.975).floor() as usize;
        boot_effects[idx.min(boot_effects.len() - 1)]
    } else {
        effect + 1.96 * std_error
    };

    let t_stat = if std_error > 0.0 && std_error.is_finite() {
        effect / std_error
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    // Compute group means
    let mean_y_treated = {
        let sum: f64 = y_trim.iter().zip(d_trim.iter()).filter(|(_, di)| **di >= 0.5).map(|(yi, _)| *yi).sum();
        sum / n_treated as f64
    };

    let mean_y_control = {
        let sum: f64 = y_trim.iter().zip(d_trim.iter()).filter(|(_, di)| **di < 0.5).map(|(yi, _)| *yi).sum();
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
    })
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
                let mean_treated = if sum_treated_w > 0.0 { sum_treated_y / sum_treated_w } else { 0.0 };
                let mean_control = if sum_control_w > 0.0 { sum_control_y / sum_control_w } else { 0.0 };
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

            let mean_treated = if n_treated > 0.0 { sum_treated_y / n_treated } else { 0.0 };
            let mean_control_weighted = if sum_control_w > 0.0 { sum_control_y / sum_control_w } else { 0.0 };

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

    // Extract variables
    let y = DesignMatrix::extract_column(dataset.df(), outcome).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let d = DesignMatrix::extract_column(dataset.df(), treatment).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

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
    let design = DesignMatrix::from_dataframe(dataset.df(), covariates, true)?;
    let x = design.data;

    // Estimate propensity scores
    let ps = estimate_propensity_scores(&x, &d)?;

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
    let (mu_1, r2_treated) = fit_outcome_model(&x_trim, &y_trim, &d_trim, true)?;
    let (mu_0, r2_control) = fit_outcome_model(&x_trim, &y_trim, &d_trim, false)?;

    // Compute point estimate based on method
    let effect = match config.method {
        DRMethod::AIPW => {
            compute_aipw_effect(&y_trim, &d_trim, &ps_trim, &mu_1, &mu_0, config.estimand)
        }
        DRMethod::IPW => {
            compute_ipw_effect(&y_trim, &d_trim, &ps_trim, config.estimand, true)
        }
        DRMethod::Regression => {
            // Simple difference in predicted means
            let mean_mu1: f64 = mu_1.iter().sum::<f64>() / n_trim as f64;
            let mean_mu0: f64 = mu_0.iter().sum::<f64>() / n_trim as f64;
            mean_mu1 - mean_mu0
        }
    };

    // Propensity score summary
    let ps_summary = compute_ps_summary(&ps_trim);

    // Bootstrap for standard errors
    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut boot_effects: Vec<f64> = Vec::with_capacity(config.bootstrap);

    for _ in 0..config.bootstrap {
        let indices: Vec<usize> = (0..n_trim).map(|_| rng.gen_range(0..n_trim)).collect();

        let y_boot: Array1<f64> = indices.iter().map(|&i| y_trim[i]).collect();
        let d_boot: Array1<f64> = indices.iter().map(|&i| d_trim[i]).collect();

        // Check bootstrap sample has both treated and control
        let n_t_boot = d_boot.iter().filter(|&&v| v >= 0.5).count();
        if n_t_boot == 0 || n_t_boot == n_trim {
            continue;
        }

        let mut x_boot = Array2::zeros((n_trim, x_trim.ncols()));
        for (new_i, &old_i) in indices.iter().enumerate() {
            x_boot.row_mut(new_i).assign(&x_trim.row(old_i));
        }

        // Re-estimate propensity scores on bootstrap sample
        let ps_boot = match estimate_propensity_scores(&x_boot, &d_boot) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Re-fit outcome models
        let (mu_1_boot, _) = match fit_outcome_model(&x_boot, &y_boot, &d_boot, true) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let (mu_0_boot, _) = match fit_outcome_model(&x_boot, &y_boot, &d_boot, false) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let effect_boot = match config.method {
            DRMethod::AIPW => {
                compute_aipw_effect(&y_boot, &d_boot, &ps_boot, &mu_1_boot, &mu_0_boot, config.estimand)
            }
            DRMethod::IPW => {
                compute_ipw_effect(&y_boot, &d_boot, &ps_boot, config.estimand, true)
            }
            DRMethod::Regression => {
                let mean_mu1: f64 = mu_1_boot.iter().sum::<f64>() / n_trim as f64;
                let mean_mu0: f64 = mu_0_boot.iter().sum::<f64>() / n_trim as f64;
                mean_mu1 - mean_mu0
            }
        };

        if effect_boot.is_finite() {
            boot_effects.push(effect_boot);
        }
    }

    if boot_effects.len() < config.bootstrap / 2 {
        warnings.push("Many bootstrap iterations failed; standard errors may be unreliable".to_string());
    }

    // Compute SE and CI
    let std_error = if !boot_effects.is_empty() {
        let mean_boot: f64 = boot_effects.iter().sum::<f64>() / boot_effects.len() as f64;
        let var_boot: f64 = boot_effects.iter().map(|&e| (e - mean_boot).powi(2)).sum::<f64>()
            / (boot_effects.len() - 1).max(1) as f64;
        var_boot.sqrt()
    } else {
        f64::NAN
    };

    boot_effects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ci_lower = if boot_effects.len() >= 20 {
        let idx = (boot_effects.len() as f64 * 0.025).floor() as usize;
        boot_effects[idx]
    } else {
        effect - 1.96 * std_error
    };

    let ci_upper = if boot_effects.len() >= 20 {
        let idx = (boot_effects.len() as f64 * 0.975).floor() as usize;
        boot_effects[idx.min(boot_effects.len() - 1)]
    } else {
        effect + 1.96 * std_error
    };

    let t_stat = if std_error > 0.0 && std_error.is_finite() {
        effect / std_error
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
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
        bootstrap_reps: config.bootstrap,
        warnings,
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
                let ipw_treated = if di >= 0.5 {
                    (yi - mu1_i) / ps_i
                } else {
                    0.0
                };

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

/// Estimate propensity scores using logistic regression.
///
/// Uses Newton-Raphson MLE to estimate P(D=1|X).
fn estimate_propensity_scores(x: &Array2<f64>, d: &Array1<f64>) -> EconResult<Array1<f64>> {
    let n = d.len();
    let k = x.ncols();

    // Newton-Raphson for logistic regression
    let mut beta = Array1::zeros(k);
    let max_iter = 50;
    let tolerance = 1e-8;

    for _ in 0..max_iter {
        // Linear predictor
        let z: Array1<f64> = x.dot(&beta);

        // Probabilities
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient: X'(d - p)
        let residuals = d - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Weights: p(1-p)
        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        // Hessian: -X'WX
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Invert -H
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) = safe_inverse(&neg_hessian.view()).map_err(|e| {
            EconError::SingularMatrix {
                context: "Propensity score estimation".to_string(),
                suggestion: format!("Check for multicollinearity in covariates: {:?}", e),
            }
        })?;

        // Update
        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    // Final propensity scores
    let z_final: Array1<f64> = x.dot(&beta);
    let ps: Array1<f64> = z_final.mapv(logistic_cdf);

    // Clip to avoid extreme values
    Ok(ps.mapv(|p| p.max(1e-10).min(1.0 - 1e-10)))
}

/// Fit outcome model for a treatment group (OLS regression).
///
/// Returns predicted outcomes and R².
fn fit_outcome_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    d: &Array1<f64>,
    treated: bool,
) -> EconResult<(Array1<f64>, f64)> {
    let n = y.len();

    // Filter to the relevant treatment group
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

    let n_group = indices.len();
    let k = x.ncols();

    // Build group-specific arrays
    let mut x_group = Array2::zeros((n_group, k));
    let mut y_group = Array1::zeros(n_group);

    for (new_i, &old_i) in indices.iter().enumerate() {
        x_group.row_mut(new_i).assign(&x.row(old_i));
        y_group[new_i] = y[old_i];
    }

    // OLS: β = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x_group.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| {
        EconError::SingularMatrix {
            context: format!("Outcome model for {} group", if treated { "treated" } else { "control" }),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        }
    })?;

    let xty_vec = xty(&x_group.view(), &y_group);
    let beta = xtx_inv.dot(&xty_vec);

    // Predictions for entire sample
    let mu: Array1<f64> = x.dot(&beta);

    // R² for the group
    let y_group_mean = y_group.mean().unwrap_or(0.0);
    let sst: f64 = y_group.iter().map(|&yi| (yi - y_group_mean).powi(2)).sum();

    let mut ssr = 0.0;
    for (new_i, &old_i) in indices.iter().enumerate() {
        let residual = y_group[new_i] - mu[old_i];
        ssr += residual * residual;
    }

    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    Ok((mu, r2))
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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[n - 1];
    let median = sorted[n / 2];

    let p10_idx = (n as f64 * 0.10).floor() as usize;
    let p90_idx = (n as f64 * 0.90).floor() as usize;
    let p10 = sorted[p10_idx.min(n - 1)];
    let p90 = sorted[p90_idx.min(n - 1)];

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
    use polars::prelude::*;

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
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_ipw_ate_basic() {
        let dataset = create_treatment_dataset();
        let config = IpwConfig {
            estimand: Estimand::ATE,
            trim: 0.01,  // Lower trim for test data
            bootstrap: 100, // Fewer for faster tests
            normalized: true,
            seed: Some(42),
        };

        let result = run_ipw_treatment(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Check basic structure (40 observations, may have some trimmed)
        assert!(result.n_obs >= 30, "Should have at least 30 obs, got {}", result.n_obs);
        assert!(result.n_treated >= 15, "Should have at least 15 treated, got {}", result.n_treated);
        assert!(result.n_control >= 15, "Should have at least 15 control, got {}", result.n_control);

        // Treatment effect should be positive (around 0.5)
        assert!(result.effect > 0.0, "ATE should be positive, got {}", result.effect);
        assert!(result.effect < 2.0, "ATE should be reasonable, got {}", result.effect);

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
        assert!(result.effect > 0.0, "ATT should be positive, got {}", result.effect);
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
        assert!(result.effect > 0.0, "AIPW effect should be positive, got {}", result.effect);

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
}
