//! Flexible inverse probability weighting (WeightIt) module.
//!
//! Provides multiple methods for computing propensity score and balancing weights
//! for causal inference, including:
//!
//! - **Logistic (PS)**: Standard propensity score weights from logistic regression
//! - **Entropy Balancing (ebal)**: Exact mean balance via entropy minimization
//! - **Energy Balancing**: Minimize energy distance between weighted distributions, solved
//!   exactly as a dense quadratic program for up to [`ENERGY_MAX_UNITS`] reweighted units
//! - **Stable Balancing**: minimum-variance weights with exact mean balance (Zubizarreta 2015;
//!   WeightIt optweight with tols = 0); approximate balance is `sbw()` with `balance_tol > 0`
//!
//! # Mathematical Background
//!
//! ## Inverse Probability Weighting (IPW)
//!
//! For ATE estimation:
//! ```text
//! w_i = D_i / e(X_i) + (1 - D_i) / (1 - e(X_i))
//! ```
//!
//! For ATT estimation:
//! ```text
//! w_i = D_i + (1 - D_i) * e(X_i) / (1 - e(X_i))
//! ```
//!
//! ## Entropy Balancing
//!
//! Minimizes the Kullback-Leibler divergence from uniform weights subject to
//! exact moment balance constraints:
//!
//! ```text
//! min Σ w_i log(w_i / q_i)
//! s.t. Σ w_i c_r(X_i) = m_r  for r = 1, ..., R
//!      Σ w_i = 1
//!      w_i ≥ 0
//! ```
//!
//! The solution has the form: w_i ∝ exp(λ' c(X_i))
//!
//! # References
//!
//! - Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without
//!   Replacement from a Finite Universe". *JASA*, 47(260), 663-685.
//!
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). "The Central Role of the Propensity
//!   Score in Observational Studies for Causal Effects". *Biometrika*, 70(1), 41-55.
//!
//! - Hainmueller, J. (2012). "Entropy Balancing for Causal Effects: A Multivariate
//!   Reweighting Method to Produce Balanced Samples in Observational Studies".
//!   *Political Analysis*, 20(1), 25-46. https://doi.org/10.1093/pan/mpr025
//!
//! - Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score".
//!   *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
//!
//! - Zubizarreta, J.R. (2015). "Stable Weights that Balance Covariates for
//!   Estimation with Incomplete Outcome Data". *JASA*, 110(511), 910-922.
//!
//! - R package WeightIt: Greifer, N. (2024). WeightIt: Weighting for Covariate Balance
//!   in Observational Studies. https://ngreifer.github.io/WeightIt/

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmFit, GlmSpec, fit_logit_lean};
use crate::econometrics::sbw::{SBWConfig, SBWEstimand, run_sbw_inner};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::options::Choices;
use crate::regression::ols::certainly_full_rank;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

/// Energy balancing refuses more reweighted units than this (decision WP23-A): the QP is
/// dense (two m x m buffers, O(m^3) factorisation, O(m^2) per active-set step). Measured on
/// one core (release build, medians of 3): 6-12 s for 2,000 reweighted units with ATT, 2-4 s
/// with ATE; 34-37 s for 3,000 with ATT. The cap keeps an at-cap solve well inside half of
/// the hosted 60 s deadline.
pub const ENERGY_MAX_UNITS: usize = 2_000;

// ═══════════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════════

/// Weighting method for propensity score estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightMethod {
    /// Inverse propensity weights from a logistic regression (WeightIt `method = "glm"`).
    #[default]
    Logistic,
    /// Entropy balancing (Hainmueller 2012; WeightIt `method = "ebal"`).
    Entropy,
    /// Energy balancing (Huling & Mak 2022; WeightIt `method = "energy"` at lambda = 0),
    /// solved exactly as a quadratic program; at most [`ENERGY_MAX_UNITS`] reweighted units.
    Energy,
    /// Stable balancing weights with exact balance (Zubizarreta 2015; WeightIt
    /// `method = "optweight"` with `tols = 0`), through [`crate::econometrics::sbw`].
    Stable,
}

impl fmt::Display for WeightMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightMethod::Logistic => write!(f, "Logistic (Propensity Score)"),
            WeightMethod::Entropy => write!(f, "Entropy Balancing"),
            WeightMethod::Energy => write!(f, "Energy Balancing"),
            WeightMethod::Stable => write!(f, "Stable Balancing"),
        }
    }
}

crate::option_enum!(
    WeightMethod,
    Choices {
        option: "method",
        values: &[
            (&["logistic", "glm", "ps", "logit"], WeightMethod::Logistic),
            (
                &["entropy", "ebal", "entropy_balancing"],
                WeightMethod::Entropy
            ),
            (&["energy"], WeightMethod::Energy),
            (
                &["stable", "optweight", "sbw", "stable_balancing"],
                WeightMethod::Stable,
            ),
        ],
    },
    no_display
);

/// Target estimand for treatment effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightEstimand {
    /// Average Treatment Effect (population)
    #[default]
    ATE,
    /// Average Treatment Effect on the Treated
    ATT,
    /// Average Treatment Effect on the Control
    ATC,
}

impl fmt::Display for WeightEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightEstimand::ATE => write!(f, "ATE (Average Treatment Effect)"),
            WeightEstimand::ATT => write!(f, "ATT (Effect on Treated)"),
            WeightEstimand::ATC => write!(f, "ATC (Effect on Control)"),
        }
    }
}

crate::option_enum!(
    WeightEstimand,
    Choices {
        option: "estimand",
        values: &[
            (&["ate"], WeightEstimand::ATE),
            (&["att"], WeightEstimand::ATT),
            (&["atc"], WeightEstimand::ATC),
        ],
    },
    no_display
);

/// Configuration for WeightIt estimation.
#[derive(Debug, Clone)]
pub struct WeightItConfig {
    /// Weighting method
    pub method: WeightMethod,
    /// Target estimand
    pub estimand: WeightEstimand,
    /// Include intercept in covariate matrix (default: true)
    pub intercept: bool,
    /// Stabilize weights (default: false); logistic weights only (ignored with a warning
    /// for the balancing methods, as in WeightIt). Multiplies weights by P(D=1) for treated
    /// and P(D=0) for control
    pub stabilize: bool,
    /// Trim extreme weights at this quantile (e.g., 0.99 trims at 1st and 99th percentile);
    /// must be in (0.5, 1]. Set to 1.0 for no trimming (default: 1.0)
    pub trim_quantile: f64,
    /// Maximum iterations for iterative methods (default: 200)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8)
    pub tolerance: f64,
}

impl Default for WeightItConfig {
    fn default() -> Self {
        Self {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATE,
            intercept: true,
            stabilize: false,
            trim_quantile: 1.0,
            max_iter: 200,
            tolerance: 1e-8,
        }
    }
}

/// Balance statistics for a single covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightItCovariateBalance {
    /// Variable name
    pub variable: String,
    /// Mean in treated group
    pub mean_treated: f64,
    /// Mean in control group
    pub mean_control: f64,
    /// Standardized mean difference: (mean_t - mean_c) / sqrt((var_t + var_c) / 2)
    pub std_diff: f64,
    /// Variance ratio: var_t / var_c
    pub var_ratio: f64,
}

/// Balance table showing before/after weighting statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightItBalanceTable {
    /// Balance statistics for each covariate
    pub covariates: Vec<WeightItCovariateBalance>,
    /// Maximum absolute standardized difference
    pub max_std_diff: f64,
    /// Mean absolute standardized difference
    pub mean_std_diff: f64,
}

impl fmt::Display for WeightItBalanceTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Variable", "Mean Treat", "Mean Ctrl", "Std.Diff", "Var.Ratio"
        )?;
        writeln!(f, "{}", "-".repeat(68))?;

        for cov in &self.covariates {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                cov.variable, cov.mean_treated, cov.mean_control, cov.std_diff, cov.var_ratio
            )?;
        }

        writeln!(f, "{}", "-".repeat(68))?;
        writeln!(
            f,
            "Max |Std.Diff|: {:.4}  Mean |Std.Diff|: {:.4}",
            self.max_std_diff, self.mean_std_diff
        )?;

        Ok(())
    }
}

/// Result from WeightIt estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightItResult {
    /// Estimated weights for each observation
    pub weights: Vec<f64>,
    /// Propensity scores (if applicable)
    pub propensity_scores: Option<Vec<f64>>,
    /// Weighting method used
    pub method: WeightMethod,
    /// Target estimand
    pub estimand: WeightEstimand,
    /// Covariate balance before weighting
    pub balance_before: WeightItBalanceTable,
    /// Covariate balance after weighting
    pub balance_after: WeightItBalanceTable,
    /// Effective sample size: ESS = (Σw)² / Σw²
    /// Lower ESS indicates higher weight variability
    pub effective_sample_size: f64,
    /// ESS for treated group
    pub ess_treated: f64,
    /// ESS for control group
    pub ess_control: f64,
    /// Maximum weight
    pub max_weight: f64,
    /// Minimum weight
    pub min_weight: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated
    pub n_treated: usize,
    /// Number of control
    pub n_control: usize,
    /// Whether algorithm converged (for iterative methods)
    pub converged: bool,
    /// Number of iterations (for iterative methods)
    pub iterations: usize,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
    /// Convergence record of the fit (the propensity GLM for `Logistic`).
    #[serde(default)]
    pub convergence: Option<Convergence>,
}

impl ResultNotes for WeightItResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl fmt::Display for WeightItResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "WeightIt: Weighting for Covariate Balance")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Method:   {}", self.method)?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Total:    {}  (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f)?;
        writeln!(f, "Weight Summary:")?;
        writeln!(
            f,
            "  Range:    [{:.4}, {:.4}]",
            self.min_weight, self.max_weight
        )?;
        writeln!(
            f,
            "  ESS:      {:.1} (Treated: {:.1}, Control: {:.1})",
            self.effective_sample_size, self.ess_treated, self.ess_control
        )?;
        writeln!(f)?;
        writeln!(f, "Balance Before Weighting:")?;
        writeln!(
            f,
            "  Max |Std.Diff|:  {:.4}",
            self.balance_before.max_std_diff
        )?;
        writeln!(
            f,
            "  Mean |Std.Diff|: {:.4}",
            self.balance_before.mean_std_diff
        )?;
        writeln!(f)?;
        writeln!(f, "Balance After Weighting:")?;
        writeln!(
            f,
            "  Max |Std.Diff|:  {:.4}",
            self.balance_after.max_std_diff
        )?;
        writeln!(
            f,
            "  Mean |Std.Diff|: {:.4}",
            self.balance_after.mean_std_diff
        )?;
        write!(f, "{}", notes_text(self))
    }
}

/// Options of [`entropy_balance_with()`].
#[derive(Debug, Clone)]
pub struct EntropyBalanceConfig {
    /// Target estimand: ATT (the default; the controls are reweighted to the treated means),
    /// ATC (the treated to the control means) or ATE (both groups to the pooled means, as
    /// `WeightIt::weightit(method = "ebal", estimand = "ATE")`).
    pub estimand: WeightEstimand,
    /// Means every reweighted group is balanced to (one finite value per covariate); `None`
    /// uses the estimand's target-population means.
    pub targets: Option<Vec<f64>>,
    /// Newton iteration cap, at least 1 (default 200).
    pub max_iter: usize,
    /// Convergence tolerance on `max_j |weighted mean_j - target_j| / sd_j`, finite and
    /// positive (default 1e-8).
    pub tolerance: f64,
}

impl Default for EntropyBalanceConfig {
    fn default() -> Self {
        Self {
            estimand: WeightEstimand::ATT,
            targets: None,
            max_iter: 200,
            tolerance: 1e-8,
        }
    }
}

fn default_att() -> WeightEstimand {
    WeightEstimand::ATT
}

/// Result from entropy balancing.
///
/// Weights follow WeightIt's `ebal` scale: target units get weight 1 and each reweighted
/// group sums to the size of the target population (ATT: the controls sum to n1; ATC: the
/// treated sum to n0; ATE: each group sums to n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyBalanceResult {
    /// Weights of all units (in data order).
    pub weights: Vec<f64>,
    /// Lagrange multipliers (original covariate scale) of the reweighted control group
    /// (ATT, ATE) or treated group (ATC); 0 for a dropped or constant covariate.
    pub lambda: Vec<f64>,
    /// Balance achieved for each covariate
    pub balance: WeightItBalanceTable,
    /// ESS `(sum w)^2 / sum w^2` of the reweighted group (ATE: the sum of both groups' ESS).
    pub effective_sample_size: f64,
    /// Maximum weight among the reweighted units
    pub max_weight: f64,
    /// Minimum positive weight among the reweighted units
    pub min_weight: f64,
    /// Whether every group's fit converged
    pub converged: bool,
    /// Newton iterations (the larger of the two groups' for ATE)
    pub iterations: usize,
    /// Final `max_j |weighted mean_j - target_j| / sd_j`
    pub constraint_tolerance: f64,
    /// Number of observations reweighted (ATE: n)
    pub n_reweighted: usize,
    /// Target estimand
    #[serde(default = "default_att")]
    pub estimand: WeightEstimand,
    /// ESS of the treated group
    #[serde(default)]
    pub ess_treated: f64,
    /// ESS of the control group
    #[serde(default)]
    pub ess_control: f64,
    /// ATE only: the treated group's multipliers (original covariate scale)
    #[serde(default)]
    pub lambda_treated: Option<Vec<f64>>,
    /// Covariates dropped as linearly dependent on the others in a reweighted group
    #[serde(default)]
    pub dropped_covariates: Vec<String>,
    /// Convergence record of the Newton iteration
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// Warnings
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for EntropyBalanceResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl fmt::Display for EntropyBalanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Entropy Balancing Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f, "Observations Reweighted: {}", self.n_reweighted)?;
        writeln!(f, "Constraint Tolerance: {:.2e}", self.constraint_tolerance)?;
        writeln!(f)?;
        writeln!(f, "Weight Summary:")?;
        writeln!(
            f,
            "  Range: [{:.4}, {:.4}]",
            self.min_weight, self.max_weight
        )?;
        writeln!(
            f,
            "  ESS:   treated {:.1}, control {:.1}",
            self.ess_treated, self.ess_control
        )?;
        writeln!(f)?;
        writeln!(f, "Balance After Weighting:")?;
        write!(f, "{}", self.balance)?;
        write!(f, "{}", notes_text(self))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main WeightIt Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute balancing weights for causal inference using multiple methods.
///
/// This function implements flexible inverse probability weighting with support
/// for various weighting methods including logistic propensity scores, entropy
/// balancing, energy balancing, and stable balancing weights.
///
/// # Arguments
///
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column (0/1)
/// * `covariate_cols` - Names of covariate columns for balance
/// * `config` - Configuration options including method and estimand
///
/// # Returns
///
/// `WeightItResult` containing weights, balance diagnostics, and effective sample size.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{weightit, WeightItConfig, WeightMethod, WeightEstimand};
///
/// let config = WeightItConfig {
///     method: WeightMethod::Entropy,
///     estimand: WeightEstimand::ATT,
///     ..Default::default()
/// };
///
/// let result = weightit(&dataset, "treatment", &["age", "income", "education"], config)?;
/// println!("ESS: {:.1}", result.effective_sample_size);
/// ```
///
/// # References
///
/// - Greifer, N. (2024). WeightIt: Weighting for Covariate Balance in Observational
///   Studies. R package. https://ngreifer.github.io/WeightIt/
pub fn weightit(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: WeightItConfig,
) -> EconResult<WeightItResult> {
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1; got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be finite and positive; got {}",
                config.tolerance
            ),
        });
    }
    if !(config.trim_quantile.is_finite()
        && config.trim_quantile > 0.5
        && config.trim_quantile <= 1.0)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "trim_quantile must be in (0.5, 1] (1 = no trimming; 0.99 trims at the 1st and 99th percentiles); got {}",
                config.trim_quantile
            ),
        });
    }
    let mut warnings = Vec::new();
    if config.stabilize && config.method != WeightMethod::Logistic {
        let label = match config.method {
            WeightMethod::Entropy => "entropy balancing",
            WeightMethod::Energy => "energy balancing",
            _ => "stable balancing",
        };
        warnings.push(format!(
            "stabilize cannot be used with {label} weights and was ignored (it applies to the logistic propensity-score weights only, as in WeightIt)"
        ));
    }

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract treatment variable
    let d = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = d.len();

    // Validate treatment is binary
    let n_treated: usize = d.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment must have both treated and control observations. Found {} treated, {} control.",
                n_treated, n_control
            ),
        });
    }

    // Build covariate matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, config.intercept)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;

    // Variable names for balance table
    let mut var_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();
    if config.intercept {
        var_names.insert(0, "Intercept".to_string());
    }

    // Compute balance before weighting
    let balance_before = compute_balance_table(&x, &d, &var_names, None);

    // Compute weights based on method
    let mut convergence = None;
    let (weights, propensity_scores, converged, iterations) = match config.method {
        WeightMethod::Logistic => {
            let (w, ps, fit) =
                logistic_weights(&x, &d, config.estimand, config.stabilize, config.max_iter)?;
            let conv = Convergence::new(
                fit.converged,
                fit.iterations,
                config.max_iter,
                "relative change in deviance",
                f64::NAN,
                fit.epsilon,
            );
            for w in &fit.warnings {
                warnings.push(format!("propensity model: {w}"));
            }
            if let Some(note) = conv.note("Propensity model") {
                warnings.push(note);
            }
            convergence = Some(conv);
            (w, Some(ps), fit.converged, fit.iterations)
        }
        WeightMethod::Entropy => {
            let x_cov = covariates_only(&x, config.intercept);
            let names: Vec<String> = covariate_cols.iter().map(|c| c.to_string()).collect();
            let (w, fits) = ebal_groups(
                &x_cov,
                &d,
                config.estimand,
                None,
                &names,
                config.max_iter,
                config.tolerance,
            )?;
            let (conv, _) = ebal_convergence(&fits, config.max_iter, config.tolerance);
            for fit in &fits {
                warnings.extend(fit.warnings.iter().cloned());
            }
            if let Some(note) = conv.note("Entropy balancing") {
                warnings.push(note);
            }
            let (c, it) = (conv.converged, conv.iterations);
            convergence = Some(conv);
            (w, None, c, it)
        }
        WeightMethod::Energy => {
            let x_cov = covariates_only(&x, config.intercept);
            let fit = energy_weights(
                x_cov.view(),
                d.as_slice().unwrap_or(&d.to_vec()),
                config.estimand,
            )?;
            warnings.extend(fit.warnings.iter().cloned());
            // WP23-B: each reweighted group sums to the target population size.
            let nf = n as f64;
            let (n1, n0) = (n_treated as f64, n_control as f64);
            let w: Vec<f64> = fit
                .weights
                .iter()
                .zip(d.iter())
                .map(|(&w, &di)| {
                    let treated = di >= 0.5;
                    match config.estimand {
                        WeightEstimand::ATT if !treated => w * n1 / n0,
                        WeightEstimand::ATC if treated => w * n0 / n1,
                        WeightEstimand::ATE => w * nf / if treated { n1 } else { n0 },
                        _ => w,
                    }
                })
                .collect();
            let conv = Convergence::new(
                true,
                fit.steps,
                fit.max_steps,
                "active-set steps (exact QP)",
                fit.max_violation,
                1e-12,
            );
            convergence = Some(conv);
            (w, None, true, fit.steps)
        }
        WeightMethod::Stable => {
            let x_cov = covariates_only(&x, config.intercept);
            let names: Vec<String> = covariate_cols.iter().map(|c| c.to_string()).collect();
            let mut cfg = SBWConfig::default();
            cfg.estimand = match config.estimand {
                WeightEstimand::ATT => SBWEstimand::ATT,
                WeightEstimand::ATC => SBWEstimand::ATC,
                WeightEstimand::ATE => SBWEstimand::ATE,
            };
            cfg.balance_tol = 0.0;
            cfg.normalize_to_n = false;
            cfg.max_iter = config.max_iter;
            cfg.tolerance = config.tolerance;
            let r = run_sbw_inner(&d.view(), &x_cov.view(), &names, cfg).map_err(|e| match e {
                EconError::InvalidSpecification { message } => EconError::InvalidSpecification {
                    message: format!(
                        "stable balancing weights with exact balance are infeasible: {message}; use treatment_sbw (sbw()) with balance_tol > 0 for approximate balance"
                    ),
                },
                other => other,
            })?;
            // WP23-B: the SBW groups sum to 1; scale each to the target population size.
            let nf = n as f64;
            let (n1, n0) = (n_treated as f64, n_control as f64);
            let w: Vec<f64> = r
                .weights
                .iter()
                .zip(d.iter())
                .map(|(&w, &di)| {
                    let treated = di >= 0.5;
                    match config.estimand {
                        WeightEstimand::ATT if !treated => w * n1,
                        WeightEstimand::ATC if treated => w * n0,
                        WeightEstimand::ATE => w * nf,
                        _ => w,
                    }
                })
                .collect();
            warnings.extend(r.warnings.iter().cloned());
            convergence = r.convergence.clone();
            (w, None, r.converged, r.iterations)
        }
    };

    // Apply weight trimming if requested
    let weights = if config.trim_quantile < 1.0 {
        trim_weights(&weights, config.trim_quantile)
    } else {
        weights
    };

    // Compute balance after weighting
    let balance_after = compute_balance_table(&x, &d, &var_names, Some(&weights));

    // Compute effective sample sizes
    let (ess_total, ess_treated, ess_control) = compute_ess(&weights, &d);

    // Weight summary statistics
    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_weight = weights
        .iter()
        .filter(|&&w| w > 0.0)
        .cloned()
        .fold(f64::INFINITY, f64::min);

    // Warnings for extreme weights
    if max_weight / min_weight.max(1e-10) > 100.0 {
        warnings.push(format!(
            "High weight variability detected (max/min = {:.1}). Consider stabilizing weights.",
            max_weight / min_weight.max(1e-10)
        ));
    }

    if ess_total < n as f64 * 0.3 {
        warnings.push(format!(
            "Low effective sample size ({:.1} of {} = {:.0}%). Weights may be too variable.",
            ess_total,
            n,
            100.0 * ess_total / n as f64
        ));
    }

    Ok(WeightItResult {
        weights,
        propensity_scores,
        method: config.method,
        estimand: config.estimand,
        balance_before,
        balance_after,
        effective_sample_size: ess_total,
        ess_treated,
        ess_control,
        max_weight,
        min_weight,
        n_obs: n,
        n_treated,
        n_control,
        converged,
        iterations,
        warnings,
        convergence,
    })
}

/// Entropy balancing (Hainmueller 2012) of the control group to the treated means (ATT).
///
/// Equivalent to [`entropy_balance_with()`] with the estimand ATT, `targets`, at most 200
/// Newton iterations and tolerance 1e-8.
///
/// # Arguments
///
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of covariate columns to balance
/// * `targets` - Optional target means (defaults to the treated group's means)
///
/// # References
///
/// - Hainmueller, J. (2012). "Entropy Balancing for Causal Effects".
///   *Political Analysis*, 20(1), 25-46.
pub fn entropy_balance(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    targets: Option<&[f64]>,
) -> EconResult<EntropyBalanceResult> {
    entropy_balance_with(
        dataset,
        treatment_col,
        covariate_cols,
        &EntropyBalanceConfig {
            targets: targets.map(<[f64]>::to_vec),
            ..Default::default()
        },
    )
}

/// Entropy balancing (Hainmueller 2012) for the ATT, ATC or ATE.
///
/// Each reweighted group gets the weights closest to uniform in Kullback-Leibler divergence
/// that reproduce the target means exactly:
///
/// ```text
/// min sum_i p_i log(m p_i)   s.t.  sum_i p_i x_i = t,  sum_i p_i = 1,  p_i > 0
/// ```
///
/// solved through its convex dual `min_lambda log sum_i exp(lambda'(x_i - t))` by Newton's
/// method with an Armijo search, on scores in standard-deviation units (scale-free). The
/// dual minimum is the entropy of the optimal weights, so a negative dual value certifies
/// that the target lies outside the convex hull of the group's covariates; a target outside
/// a covariate's range is refused before the solve. Both are `InvalidSpecification` errors
/// (R's WeightIt returns zero weights with a warning). Covariates that are linearly dependent
/// on the others within a reweighted group are dropped, as WeightIt's `make_full_rank`
/// does; their balance is checked after the fit.
///
/// Weights follow WeightIt's `ebal` scale (see [`EntropyBalanceResult`]).
pub fn entropy_balance_with(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: &EntropyBalanceConfig,
) -> EconResult<EntropyBalanceResult> {
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1; got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be finite and positive; got {}",
                config.tolerance
            ),
        });
    }
    if let Some(t) = &config.targets {
        if t.len() != covariate_cols.len() {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "targets must have one mean per covariate: got {} targets for {} covariates",
                    t.len(),
                    covariate_cols.len()
                ),
            });
        }
        if let Some(v) = t.iter().find(|v| !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!("targets must be finite; got {v}"),
            });
        }
    }

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    let d = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;
    let names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    let (weights, fits) = ebal_groups(
        &x,
        &d,
        config.estimand,
        config.targets.as_deref(),
        &names,
        config.max_iter,
        config.tolerance,
    )?;
    let (convergence, residual) = ebal_convergence(&fits, config.max_iter, config.tolerance);
    let mut warnings: Vec<String> = fits.iter().flat_map(|f| f.warnings.clone()).collect();
    if let Some(note) = convergence.note("Entropy balancing") {
        warnings.push(note);
    }
    let mut dropped: Vec<String> = Vec::new();
    for fit in &fits {
        for &j in &fit.dropped {
            if !dropped.contains(&names[j]) {
                dropped.push(names[j].clone());
            }
        }
    }

    let balance = compute_balance_table(&x, &d, &names, Some(&weights));
    let (_, ess_treated, ess_control) = compute_ess(&weights, &d);
    let reweighted = |i: usize| match config.estimand {
        WeightEstimand::ATT => d[i] < 0.5,
        WeightEstimand::ATC => d[i] >= 0.5,
        WeightEstimand::ATE => true,
    };
    let rw: Vec<f64> = (0..d.len())
        .filter(|&i| reweighted(i))
        .map(|i| weights[i])
        .collect();
    let max_weight = rw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_weight = rw
        .iter()
        .filter(|&&w| w > 0.0)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let effective_sample_size = match config.estimand {
        WeightEstimand::ATT => ess_control,
        WeightEstimand::ATC => ess_treated,
        WeightEstimand::ATE => ess_treated + ess_control,
    };
    // fits are (control, treated) for ATE, the single reweighted group otherwise
    let lambda = fits[0].lambda.clone();
    let lambda_treated = (config.estimand == WeightEstimand::ATE).then(|| fits[1].lambda.clone());

    Ok(EntropyBalanceResult {
        weights,
        lambda,
        balance,
        effective_sample_size,
        max_weight,
        min_weight,
        converged: convergence.converged,
        iterations: convergence.iterations,
        constraint_tolerance: residual,
        n_reweighted: rw.len(),
        estimand: config.estimand,
        ess_treated,
        ess_control,
        lambda_treated,
        dropped_covariates: dropped,
        convergence: Some(convergence),
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weighting Method Implementations
// ═══════════════════════════════════════════════════════════════════════════════

/// Logistic propensity-score weights: the shared binomial GLM (`glm.fit`'s IRLS) of the
/// treatment on `x`, then the weights of `estimand`. Returns (weights, scores, the GLM fit).
fn logistic_weights(
    x: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    stabilize: bool,
    max_iter: usize,
) -> EconResult<(Vec<f64>, Vec<f64>, GlmFit)> {
    if let Some((i, v)) = d.iter().enumerate().find(|(_, v)| **v != 0.0 && **v != 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("propensity treatment must be coded 0/1 (row {i} is {v})"),
        });
    }
    let n = d.len();
    let mut spec = GlmSpec::logit();
    spec.max_iter = max_iter;
    let fit = fit_logit_lean(x.view(), d.view(), &spec)?;
    // Clipped only to keep the weights finite.
    let ps: Vec<f64> = fit
        .fitted
        .iter()
        .map(|&p| p.clamp(1e-10, 1.0 - 1e-10))
        .collect();
    let p_treat = d.iter().filter(|&&di| di == 1.0).count() as f64 / n as f64;

    let weights: Vec<f64> = (0..n)
        .map(|i| {
            let treated = d[i] == 1.0;
            let e = ps[i];
            match estimand {
                WeightEstimand::ATE if stabilize => {
                    if treated {
                        p_treat / e
                    } else {
                        (1.0 - p_treat) / (1.0 - e)
                    }
                }
                WeightEstimand::ATE => {
                    if treated {
                        1.0 / e
                    } else {
                        1.0 / (1.0 - e)
                    }
                }
                // ATT and ATC weights are already stable (WeightIt leaves them unchanged).
                WeightEstimand::ATT => {
                    if treated {
                        1.0
                    } else {
                        e / (1.0 - e)
                    }
                }
                WeightEstimand::ATC => {
                    if treated {
                        (1.0 - e) / e
                    } else {
                        1.0
                    }
                }
            }
        })
        .collect();

    Ok((weights, ps, fit))
}

/// The covariate columns of `x` without the intercept column `weightit()` may have added.
fn covariates_only(x: &Array2<f64>, intercept: bool) -> Array2<f64> {
    if intercept {
        x.slice(ndarray::s![.., 1..]).to_owned()
    } else {
        x.clone()
    }
}

/// One group's entropy-balancing fit.
#[derive(Debug, Clone)]
struct EbalFit {
    /// Weights of the group's units (sum 1).
    weights: Vec<f64>,
    /// Multipliers in the original covariate scale (0 for dropped or constant columns).
    lambda: Vec<f64>,
    converged: bool,
    iterations: usize,
    /// Final `max_j |weighted mean_j - target_j| / sd_j` over the solved columns.
    residual: f64,
    /// Columns dropped as linearly dependent on the others.
    dropped: Vec<usize>,
    warnings: Vec<String>,
}

fn ebal_infeasible(message: String) -> EconError {
    EconError::InvalidSpecification {
        message: format!("Entropy balancing is infeasible: {message}"),
    }
}

/// Entropy balancing of one group (`x`, m x k) to `target`: the convex dual
/// `min_lambda f(lambda) = log sum_i exp(C_i . lambda)` with `C_ij = (x_ij - t_j) / sd_j`,
/// minimised by Newton's method with an Armijo search. `group` names the group in messages.
fn ebal_solve(
    x: ArrayView2<f64>,
    target: &[f64],
    names: &[String],
    group: &str,
    max_iter: usize,
    tol: f64,
) -> EconResult<EbalFit> {
    let (m, k) = x.dim();
    if m == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("entropy balancing needs at least one {group} unit"),
        });
    }
    // 1. Range checks; constant columns that already match their target are skipped.
    let mut solved: Vec<usize> = Vec::with_capacity(k);
    for j in 0..k {
        let col = x.column(j);
        if col.iter().any(|v| !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!("covariate {} must be finite", names[j]),
            });
        }
        let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t = target[j];
        if hi == lo {
            if (t - lo).abs() <= 1e-12 * lo.abs().max(1.0) {
                continue;
            }
            return Err(ebal_infeasible(format!(
                "{} is constant ({lo}) in the {group} group but its target mean is {t}",
                names[j]
            )));
        }
        if t <= lo || t >= hi {
            return Err(ebal_infeasible(format!(
                "the target mean of {} ({t:.6}) is outside the {group} group's range [{lo:.6}, {hi:.6}]; positive weights cannot reach it",
                names[j]
            )));
        }
        solved.push(j);
    }

    // 2. Linearly dependent columns (WeightIt's make_full_rank).
    let mut warnings = Vec::new();
    let mut dropped = Vec::new();
    if !solved.is_empty() {
        let mut centred = Array2::<f64>::zeros((m, solved.len()));
        for (c, &j) in solved.iter().enumerate() {
            let mean = x.column(j).sum() / m as f64;
            for i in 0..m {
                centred[[i, c]] = x[[i, j]] - mean;
            }
        }
        // The normal equations' certificate of full rank; only a design without it needs the QR.
        let aliased = if certainly_full_rank(centred.view())? {
            Vec::new()
        } else {
            Qr::decompose(centred.view(), LM_QR_TOL)?.aliased()
        };
        if !aliased.is_empty() {
            dropped = aliased.iter().map(|&c| solved[c]).collect();
            let list: Vec<&str> = dropped.iter().map(|&j| names[j].as_str()).collect();
            warnings.push(format!(
                "Entropy balancing: {} dropped as linearly dependent on the other covariates in the {group} group (as WeightIt's make_full_rank); their balance is checked after the fit",
                list.join(", ")
            ));
            solved.retain(|j| !dropped.contains(j));
        }
    }

    // 3. Scores in SD units.
    let kk = solved.len();
    let sd: Vec<f64> = solved
        .iter()
        .map(|&j| {
            let col = x.column(j);
            let mean = col.sum() / m as f64;
            (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0).max(1.0)).sqrt()
        })
        .collect();
    // row-major m x kk
    let mut c = resource::try_vec(m * kk, 0.0, "entropy-balancing scores")?;
    for (cj, &j) in solved.iter().enumerate() {
        for i in 0..m {
            c[i * kk + cj] = (x[[i, j]] - target[j]) / sd[cj];
        }
    }
    let row = |i: usize| &c[i * kk..(i + 1) * kk];

    // f(lambda), with the normalised weights written into p.
    let eval = |lam: &[f64], p: &mut [f64]| -> f64 {
        for (i, zi) in p.iter_mut().enumerate() {
            *zi = row(i).iter().zip(lam).map(|(u, v)| u * v).sum();
        }
        let zmax = p.iter().cloned().max_by(f64::total_cmp).unwrap_or(0.0);
        let mut sum = 0.0;
        for pi in p.iter_mut() {
            *pi = (*pi - zmax).exp();
            sum += *pi;
        }
        for pi in p.iter_mut() {
            *pi /= sum;
        }
        zmax + sum.ln()
    };

    // 4. Newton with Armijo backtracking. The weights live in two buffers, swapped on a step.
    let mut lambda = vec![0.0; kk];
    let mut trial = vec![0.0; kk];
    let mut g = vec![0.0; kk];
    let mut dev = vec![0.0; kk];
    let mut h = Array2::<f64>::zeros((kk, kk));
    let mut p = resource::try_vec(m, 0.0, "entropy-balancing weights")?;
    let mut p_new = resource::try_vec(m, 0.0, "entropy-balancing weights")?;
    let mut iterations = 0usize;
    let mut converged = false;
    let mut residual;
    let mut f = eval(&lambda, &mut p);
    loop {
        resource::checkpoint()?;
        if f < -1e-10 {
            return Err(ebal_infeasible(format!(
                "the {group} group cannot be reweighted to the target means (the target lies outside the convex hull of its covariates)"
            )));
        }
        g.fill(0.0);
        for (i, &pi) in p.iter().enumerate() {
            for (ga, ca) in g.iter_mut().zip(row(i)) {
                *ga += pi * ca;
            }
        }
        residual = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if residual.is_nan() {
            return Err(EconError::Computation(
                "entropy balancing produced a non-finite balance residual".to_string(),
            ));
        }
        if residual <= tol {
            converged = true;
            break;
        }
        if iterations == max_iter {
            break;
        }
        h.fill(0.0);
        for (i, &pi) in p.iter().enumerate() {
            for ((d, ca), ga) in dev.iter_mut().zip(row(i)).zip(&g) {
                *d = ca - ga;
            }
            for a in 0..kk {
                let wa = pi * dev[a];
                for b in 0..=a {
                    h[[a, b]] += wa * dev[b];
                }
            }
        }
        for a in 0..kk {
            for b in 0..a {
                h[[b, a]] = h[[a, b]];
            }
        }
        let inv = match sym_inverse(h.view(), SingularPolicy::Error, "entropy-balancing Hessian") {
            Ok(inv) => inv.inverse,
            Err(EconError::SingularMatrix { .. }) => {
                return Err(ebal_infeasible(format!(
                    "the {group} group's weights collapse onto a lower-dimensional face (the target lies on the boundary of its covariate hull)"
                )));
            }
            Err(e) => return Err(e),
        };
        let delta = -inv.dot(&ndarray::ArrayView1::from(&g[..]));
        let slope: f64 = g.iter().zip(delta.iter()).map(|(u, v)| u * v).sum();
        let mut t = 1.0;
        let mut step_to = |t: f64, p_new: &mut [f64]| -> f64 {
            for ((v, l), d) in trial.iter_mut().zip(&lambda).zip(delta.iter()) {
                *v = l + d * t;
            }
            eval(&trial, p_new)
        };
        let mut f_new = step_to(t, &mut p_new);
        if -slope > 1e-12 * (1.0 + f.abs()) {
            while (f_new.is_nan() || f_new > f + 1e-4 * t * slope) && t > 1e-12 {
                t *= 0.5;
                f_new = step_to(t, &mut p_new);
            }
            if t <= 1e-12 {
                break;
            }
        }
        lambda.copy_from_slice(&trial);
        f = f_new;
        std::mem::swap(&mut p, &mut p_new);
        iterations += 1;
    }

    // 5. Weights.
    if p.iter().any(|w| !w.is_finite()) {
        return Err(EconError::Computation(
            "entropy balancing produced non-finite weights".to_string(),
        ));
    }
    // 6. Dropped columns must be balanced by the fit too.
    for &j in &dropped {
        let col = x.column(j);
        let mean = col.sum() / m as f64;
        let sdj = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0).max(1.0))
            .sqrt();
        let achieved: f64 = col.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
        if (achieved - target[j]).abs() / sdj > tol.max(1e-8) {
            return Err(ebal_infeasible(format!(
                "{} is linearly dependent on the other covariates in the {group} group but its target mean is not",
                names[j]
            )));
        }
    }
    // 7. Multipliers in the original scale.
    let mut lam_full = vec![0.0; k];
    for (cj, &j) in solved.iter().enumerate() {
        lam_full[j] = lambda[cj] / sd[cj];
    }
    Ok(EbalFit {
        weights: p,
        lambda: lam_full,
        converged,
        iterations,
        residual,
        dropped,
        warnings,
    })
}

/// Entropy balancing for `estimand`: the reweighted groups' fits and the weights of all units
/// in WeightIt's `ebal` scale (target units 1; each reweighted group sums to the target
/// population size). The fits are (control) for ATT, (treated) for ATC and (control,
/// treated) for ATE.
fn ebal_groups(
    x_cov: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    targets: Option<&[f64]>,
    names: &[String],
    max_iter: usize,
    tol: f64,
) -> EconResult<(Vec<f64>, Vec<EbalFit>)> {
    let n = d.len();
    let treated: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
    let control: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();
    let (nt, nc) = (treated.len(), control.len());
    if nt == 0 || nc == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "entropy balancing needs both treated and control units; found {nt} treated and {nc} control"
            ),
        });
    }
    let rows = |idx: &[usize]| x_cov.select(ndarray::Axis(0), idx);
    let means = |idx: &[usize]| -> Vec<f64> {
        (0..x_cov.ncols())
            .map(|j| idx.iter().map(|&i| x_cov[[i, j]]).sum::<f64>() / idx.len() as f64)
            .collect()
    };
    let target = match targets {
        Some(t) => t.to_vec(),
        None => match estimand {
            WeightEstimand::ATT => means(&treated),
            WeightEstimand::ATC => means(&control),
            WeightEstimand::ATE => means(&(0..n).collect::<Vec<_>>()),
        },
    };
    let mut weights = vec![1.0; n];
    let mut fits = Vec::with_capacity(2);
    let mut fit_group = |idx: &[usize], label: &str, total: usize| -> EconResult<()> {
        let fit = ebal_solve(rows(idx).view(), &target, names, label, max_iter, tol)?;
        for (&i, &p) in idx.iter().zip(&fit.weights) {
            weights[i] = p * total as f64;
        }
        fits.push(fit);
        Ok(())
    };
    match estimand {
        WeightEstimand::ATT => fit_group(&control, "control", nt)?,
        WeightEstimand::ATC => fit_group(&treated, "treated", nc)?,
        WeightEstimand::ATE => {
            fit_group(&control, "control", n)?;
            fit_group(&treated, "treated", n)?;
        }
    }
    Ok((weights, fits))
}

/// The combined convergence record of the group fits and the largest final residual.
fn ebal_convergence(fits: &[EbalFit], max_iter: usize, tol: f64) -> (Convergence, f64) {
    let converged = fits.iter().all(|f| f.converged);
    let iterations = fits.iter().map(|f| f.iterations).max().unwrap_or(0);
    let residual = fits.iter().map(|f| f.residual).fold(0.0, f64::max);
    (
        Convergence::new(
            converged,
            iterations,
            max_iter,
            "max |weighted mean - target| / sd",
            residual,
            tol,
        ),
        residual,
    )
}

/// Result of [`energy_weights`].
struct EnergyFit {
    /// Mean 1 within each reweighted group; target units 1.
    weights: Vec<f64>,
    /// Active-set steps (constraints added plus dropped).
    steps: usize,
    max_steps: usize,
    max_violation: f64,
    warnings: Vec<String>,
}

/// Constraints `W_g >= 0` of the energy QP in the Householder basis of each block's sum
/// constraint: `W = c + Z u`, with only the reflectors stored.
struct EnergyConstraints<'a> {
    blocks: &'a [EnergyBlock],
    n_vars: usize,
}

/// One reweighted group's unique rows `offset .. offset + k` (in the merged order) and its
/// reflector `I - beta v v'`, `v = e_1 - 1/sqrt(k)`; variables `var_offset .. + k - 1`.
struct EnergyBlock {
    offset: usize,
    k: usize,
    var_offset: usize,
    v: Vec<f64>,
    beta: f64,
    c: f64,
}

impl EnergyBlock {
    /// `H y` for `y` of length k (in place).
    fn reflect(&self, y: &mut [f64]) {
        if self.k < 2 {
            return;
        }
        let vy: f64 = self.v.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        for (yi, vi) in y.iter_mut().zip(&self.v) {
            *yi -= self.beta * vi * vy;
        }
    }
}

impl EnergyConstraints<'_> {
    fn locate(&self, i: usize) -> (&EnergyBlock, usize) {
        let b = self
            .blocks
            .iter()
            .find(|b| i >= b.offset && i < b.offset + b.k)
            .expect("constraint index within the blocks");
        (b, i - b.offset)
    }

    /// W = c + Z u for every unique row.
    fn totals(&self, x: &[f64]) -> Vec<f64> {
        let k: usize = self.blocks.iter().map(|b| b.k).sum();
        let mut w = vec![0.0; k];
        for b in self.blocks {
            let mut y = vec![0.0; b.k];
            y[1..].copy_from_slice(&x[b.var_offset..b.var_offset + b.k - 1]);
            b.reflect(&mut y);
            for (g, yg) in y.iter().enumerate() {
                w[b.offset + g] = b.c + yg;
            }
        }
        w
    }
}

impl dense_qp::Constraints for EnergyConstraints<'_> {
    fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.k).sum()
    }

    fn n_eq(&self) -> usize {
        0
    }

    fn normal(&self, i: usize, out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        let (b, g) = self.locate(i);
        if b.k < 2 {
            return;
        }
        // Row g of Z: (e_g - beta v_g v) restricted to the block's columns 2..k.
        for col in 1..b.k {
            let delta = if col == g { 1.0 } else { 0.0 };
            out[b.var_offset + col - 1] = delta - b.beta * b.v[g] * b.v[col];
        }
        debug_assert!(out.len() == self.n_vars);
    }

    fn rhs(&self, i: usize) -> f64 {
        -self.locate(i).0.c
    }

    fn normal_norm(&self, i: usize) -> f64 {
        let k = self.locate(i).0.k as f64;
        (1.0 - 1.0 / k).max(0.0).sqrt()
    }

    fn slacks(&self, x: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&self.totals(x));
    }
}

/// Energy balancing weights (Huling & Mak 2022) by an exact QP: ATT reweights the controls
/// towards the treated sample, ATC the treated towards the controls, ATE both groups
/// (WeightIt `weightit2energy`, improved, lambda = 0), on covariates standardised over all
/// units. Identical standardised rows within a group are merged and share their weight.
fn energy_weights(
    x_cov: ArrayView2<f64>,
    d: &[f64],
    estimand: WeightEstimand,
) -> EconResult<EnergyFit> {
    let (n, p) = x_cov.dim();
    let nf = n as f64;
    // 1. Standardise over all units; drop constant columns.
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        let col = x_cov.column(j);
        let mean = col.sum() / nf;
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0)).sqrt();
        if sd > 0.0 && sd.is_finite() {
            cols.push(col.iter().map(|v| (v - mean) / sd).collect());
        }
    }
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect();
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let treated: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
    let control: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();
    let (n1, n0) = (treated.len() as f64, control.len() as f64);

    // 2. Blocks: (is_treated, units).
    let groups: Vec<(bool, &[usize])> = match estimand {
        WeightEstimand::ATT => vec![(false, &control)],
        WeightEstimand::ATC => vec![(true, &treated)],
        WeightEstimand::ATE => vec![(false, &control), (true, &treated)],
    };
    let m: usize = groups.iter().map(|(_, u)| u.len()).sum();
    if m > ENERGY_MAX_UNITS {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "energy balancing reweights at most {ENERGY_MAX_UNITS} units (it solves a dense {m} x {m} quadratic program); use method entropy or stable for larger samples"
            ),
        });
    }

    // 3. Merge identical rows within each block.
    struct Unique {
        row: Vec<f64>,
        units: Vec<usize>,
        treated: bool,
    }
    let mut uniq: Vec<Unique> = Vec::new();
    let mut blocks: Vec<EnergyBlock> = Vec::new();
    let mut var_offset = 0usize;
    for &(is_t, units) in &groups {
        resource::checkpoint()?;
        let mut idx: Vec<usize> = units.to_vec();
        idx.sort_by(|&a, &b| {
            rows[a]
                .iter()
                .zip(&rows[b])
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let offset = uniq.len();
        for &i in &idx {
            let in_block = uniq.len() > offset;
            match uniq.last_mut() {
                Some(u) if in_block && u.row == rows[i] => u.units.push(i),
                _ => uniq.push(Unique {
                    row: rows[i].clone(),
                    units: vec![i],
                    treated: is_t,
                }),
            }
        }
        let k = uniq.len() - offset;
        let ub = 1.0 / (k as f64).sqrt();
        let mut v = vec![-ub; k];
        v[0] += 1.0;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let beta = if k >= 2 { 2.0 / vv } else { 0.0 };
        blocks.push(EnergyBlock {
            offset,
            k,
            var_offset,
            v,
            beta,
            c: units.len() as f64 / k as f64,
        });
        var_offset += k.saturating_sub(1);
    }
    let k = uniq.len();
    let n_vars = var_offset;
    let group_n = |t: bool| if t { n1 } else { n0 };

    // 4-6. Build the reduced QP and solve it; one ridge retry when not positive definite.
    let mut warnings = Vec::new();
    let mut solution: Option<dense_qp::QpSolution> = None;
    let max_steps = 10 * (k + n_vars);
    let cons = EnergyConstraints {
        blocks: &blocks,
        n_vars,
    };
    if n_vars > 0 {
        for ridge in [false, true] {
            // G~ = 2P and q over the unique rows.
            let mut gt = resource::try_zeros2(k, k, "energy-distance matrix")?;
            let mut q = vec![0.0; k];
            for g in 0..k {
                resource::checkpoint()?;
                let ug = &uniq[g];
                for h in 0..=g {
                    let uh = &uniq[h];
                    let dgh = dist(&ug.row, &uh.row);
                    let pgh = match estimand {
                        WeightEstimand::ATT => -dgh / (n0 * n0),
                        WeightEstimand::ATC => -dgh / (n1 * n1),
                        WeightEstimand::ATE => match (ug.treated, uh.treated) {
                            (false, false) => -2.0 * dgh / (n0 * n0),
                            (true, true) => -2.0 * dgh / (n1 * n1),
                            _ => dgh / (n0 * n1),
                        },
                    };
                    gt[[g, h]] = 2.0 * pgh;
                    gt[[h, g]] = 2.0 * pgh;
                }
                q[g] = match estimand {
                    WeightEstimand::ATT => {
                        2.0 / (n0 * n1)
                            * treated
                                .iter()
                                .map(|&j| dist(&ug.row, &rows[j]))
                                .sum::<f64>()
                    }
                    WeightEstimand::ATC => {
                        2.0 / (n0 * n1)
                            * control
                                .iter()
                                .map(|&j| dist(&ug.row, &rows[j]))
                                .sum::<f64>()
                    }
                    WeightEstimand::ATE => {
                        let s: f64 = uniq
                            .iter()
                            .map(|u| u.units.len() as f64 * dist(&ug.row, &u.row))
                            .sum();
                        2.0 / nf * s / group_n(ug.treated)
                    }
                };
                if ridge {
                    let ng = group_n(ug.treated);
                    gt[[g, g]] += 1e-4 / (ng * ng * ug.units.len() as f64);
                }
            }
            // Linear term H (G~ c + q), before G~ is transformed.
            let mut c_full = vec![0.0; k];
            for b in &blocks {
                for g in 0..b.k {
                    c_full[b.offset + g] = b.c;
                }
            }
            let mut lin: Vec<f64> = (0..k)
                .map(|g| (0..k).map(|h| gt[[g, h]] * c_full[h]).sum::<f64>() + q[g])
                .collect();
            for b in &blocks {
                b.reflect(&mut lin[b.offset..b.offset + b.k]);
            }
            // B = H G~ H by rank-one updates per block (rows, then columns).
            for b in &blocks {
                resource::checkpoint()?;
                if b.k < 2 {
                    continue;
                }
                let rows = b.offset..b.offset + b.k;
                let mut vm = vec![0.0; k];
                for (r, vr) in rows.clone().zip(&b.v) {
                    for (c, acc) in vm.iter_mut().enumerate() {
                        *acc += vr * gt[[r, c]];
                    }
                }
                for (r, vr) in rows.clone().zip(&b.v) {
                    for (c, val) in vm.iter().enumerate() {
                        gt[[r, c]] -= b.beta * vr * val;
                    }
                }
                for r in 0..k {
                    let mv: f64 = rows.clone().zip(&b.v).map(|(c, vc)| gt[[r, c]] * vc).sum();
                    for (c, vc) in rows.clone().zip(&b.v) {
                        gt[[r, c]] -= b.beta * mv * vc;
                    }
                }
            }
            // Keep the rows and columns that are not a block's first.
            let keep: Vec<usize> = blocks
                .iter()
                .flat_map(|b| (b.offset + 1)..(b.offset + b.k))
                .collect();
            let mut bmat = resource::try_zeros2(n_vars, n_vars, "energy QP Hessian")?;
            for (i, &r) in keep.iter().enumerate() {
                for (j, &c) in keep.iter().enumerate() {
                    bmat[[i, j]] = gt[[r, c]];
                }
            }
            drop(gt);
            let a: Vec<f64> = keep.iter().map(|&g| lin[g]).collect();
            match dense_qp::solve_dense_qp(bmat, &a, &cons, max_steps)? {
                dense_qp::QpOutcome::Solved(sol) => {
                    if ridge {
                        warnings.push("energy balancing: the energy-distance form is numerically singular (near-duplicate covariate rows); WeightIt's ridge lambda = 1e-4 was added".to_string());
                    }
                    solution = Some(sol);
                    break;
                }
                dense_qp::QpOutcome::Infeasible => {
                    return Err(EconError::Computation(
                        "energy balancing: the quadratic program was reported infeasible, which uniform weights rule out".to_string(),
                    ));
                }
                dense_qp::QpOutcome::NotPositiveDefinite => continue,
            }
        }
        if solution.is_none() {
            return Err(EconError::Computation(
                "energy balancing: the energy-distance form is not positive definite even with WeightIt's ridge lambda = 1e-4".to_string(),
            ));
        }
    }

    // 7. Unit weights, mean 1 within each group.
    let (u, steps, max_violation) = match &solution {
        Some(s) => (s.x.clone(), s.added + s.dropped, s.max_violation),
        None => (Vec::new(), 0, 0.0),
    };
    let totals = cons.totals(&u);
    let mut weights = vec![1.0; n];
    for (g, ug) in uniq.iter().enumerate() {
        let mut w = totals[g] / ug.units.len() as f64;
        if w.abs() < 1e-10 {
            w = 0.0;
        }
        for &i in &ug.units {
            weights[i] = w;
        }
    }
    Ok(EnergyFit {
        weights,
        steps,
        max_steps,
        max_violation,
        warnings,
    })
}

/// Dense strictly convex quadratic programs by the dual active-set method of Goldfarb &
/// Idnani (1983), "A numerically stable dual method for solving strictly convex quadratic
/// programs", *Mathematical Programming* 27, 1-33. Written from the published algorithm; no
/// code from the GPL quadprog packages.
pub(crate) mod dense_qp {
    use ndarray::Array2;

    use crate::errors::{EconError, EconResult};
    use crate::linalg::matrix_ops::cholesky;
    use crate::resource;

    /// The constraints `n_i'x >= b_i`, `i < len()`; the first `n_eq()` are equalities.
    /// Normals are produced on demand so a structured family needs no p x n matrix.
    pub(crate) trait Constraints {
        fn len(&self) -> usize;
        fn n_eq(&self) -> usize;
        fn normal(&self, i: usize, out: &mut [f64]);
        fn rhs(&self, i: usize) -> f64;
        /// `||n_i||`
        fn normal_norm(&self, i: usize) -> f64;
        /// `n_i'x - b_i` for every i
        fn slacks(&self, x: &[f64], out: &mut [f64]);
    }

    /// Explicit constraints: rows of `normals` (p x n), `rhs` (p), the first `meq`
    /// equalities.
    pub(crate) struct DenseConstraints {
        pub normals: Array2<f64>,
        pub rhs: Vec<f64>,
        pub meq: usize,
    }

    impl Constraints for DenseConstraints {
        fn len(&self) -> usize {
            self.rhs.len()
        }
        fn n_eq(&self) -> usize {
            self.meq
        }
        fn normal(&self, i: usize, out: &mut [f64]) {
            for (o, v) in out.iter_mut().zip(self.normals.row(i)) {
                *o = *v;
            }
        }
        fn rhs(&self, i: usize) -> f64 {
            self.rhs[i]
        }
        fn normal_norm(&self, i: usize) -> f64 {
            self.normals
                .row(i)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        }
        fn slacks(&self, x: &[f64], out: &mut [f64]) {
            for (i, o) in out.iter_mut().enumerate() {
                let nx: f64 = self.normals.row(i).iter().zip(x).map(|(a, b)| a * b).sum();
                *o = nx - self.rhs[i];
            }
        }
    }

    pub(crate) struct QpSolution {
        pub x: Vec<f64>,
        /// Per constraint, 0 if inactive.
        pub multipliers: Vec<f64>,
        pub active: Vec<usize>,
        pub added: usize,
        pub dropped: usize,
        pub max_violation: f64,
    }

    pub(crate) enum QpOutcome {
        Solved(QpSolution),
        Infeasible,
        NotPositiveDefinite,
    }

    /// Givens rotation (c, s) taking (a, b) to (hypot(a, b), 0).
    fn givens(a: f64, b: f64) -> (f64, f64) {
        let h = a.hypot(b);
        if h == 0.0 { (1.0, 0.0) } else { (a / h, b / h) }
    }

    /// Rotates rows `i < j` of the row-major n x n buffer `m` (columns `i` and `j` of J, which
    /// is stored transposed): (m_i, m_j) <- (c m_i + s m_j, -s m_i + c m_j).
    fn rotate_rows(m: &mut [f64], n: usize, i: usize, j: usize, c: f64, s: f64) {
        debug_assert!(i < j);
        let (head, tail) = m.split_at_mut(j * n);
        let ri = &mut head[i * n..(i + 1) * n];
        let rj = &mut tail[..n];
        for (a, b) in ri.iter_mut().zip(rj.iter_mut()) {
            let (va, vb) = (*a, *b);
            *a = c * va + s * vb;
            *b = -s * va + c * vb;
        }
    }

    /// Minimise `a'x + 1/2 x'Gx` subject to `c`; `g` (n x n, symmetric) is consumed.
    pub(crate) fn solve_dense_qp(
        g: Array2<f64>,
        a: &[f64],
        c: &dyn Constraints,
        max_steps: usize,
    ) -> EconResult<QpOutcome> {
        let n = a.len();
        let p = c.len();
        let meq = c.n_eq();
        resource::checkpoint()?;
        // 1. G = L L', J = L^{-T}.
        let l = match cholesky(&g.view()) {
            Ok(l) => l,
            Err(_) => return Ok(QpOutcome::NotPositiveDefinite),
        };
        drop(g);
        if l.diag().iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Ok(QpOutcome::NotPositiveDefinite);
        }
        // J is stored transposed (`jt` = J' = L^{-1}, row-major), so every column of J the
        // solver reads or rotates is a contiguous row of `jt`. Row i of L^{-1} is
        // (e_i - sum_{k<i} L[i,k] row_k(L^{-1})) / L[i,i], nonzero in 0..=i.
        let mut jt_arr = resource::try_zeros2(n, n, "QP inverse factor")?;
        let jt = jt_arr
            .as_slice_mut()
            .expect("try_zeros2 returns a standard-layout array");
        for i in 0..n {
            if i % 64 == 0 {
                resource::checkpoint()?;
            }
            let (done, rest) = jt.split_at_mut(i * n);
            let row_i = &mut rest[..n];
            let l_row = l.row(i);
            for k in 0..i {
                let lik = l_row[k];
                if lik != 0.0 {
                    let row_k = &done[k * n..k * n + k + 1];
                    for (o, v) in row_i[..=k].iter_mut().zip(row_k) {
                        *o -= lik * v;
                    }
                }
            }
            row_i[i] += 1.0;
            let inv = 1.0 / l_row[i];
            for o in row_i[..=i].iter_mut() {
                *o *= inv;
            }
        }
        drop(l);
        let row = |c2: usize| c2 * n..(c2 + 1) * n;
        let dot =
            |a2: &[f64], b2: &[f64]| -> f64 { a2.iter().zip(b2).map(|(u2, v2)| u2 * v2).sum() };
        // x = -J (J' a)
        let mut x = vec![0.0; n];
        for c2 in 0..n {
            let jc = &jt[row(c2)];
            let jta = dot(jc, a);
            for (xi, v) in x.iter_mut().zip(jc) {
                *xi -= jta * v;
            }
        }

        let mut r_mat = resource::try_zeros2(n, n, "QP active-set factor")?;
        let mut active: Vec<usize> = Vec::new();
        let mut is_active = vec![false; p];
        let mut sign: Vec<f64> = Vec::new(); // orientation of each active constraint
        let mut u: Vec<f64> = Vec::new();
        let mut q = 0usize;
        let b_max = (0..p).fold(0.0_f64, |m, i| m.max(c.rhs(i).abs()));
        let tol = 1e-12 * (1.0 + b_max);
        let mut slack = vec![0.0; p];
        let mut np = vec![0.0; n];
        let (mut added, mut dropped, mut steps) = (0usize, 0usize, 0usize);
        let bump = |steps: &mut usize| -> EconResult<()> {
            *steps += 1;
            if *steps > max_steps {
                return Err(EconError::Computation(format!(
                    "dense QP: the active-set method did not terminate in {max_steps} steps"
                )));
            }
            resource::checkpoint()
        };

        loop {
            bump(&mut steps)?;
            c.slacks(&x, &mut slack);
            // 2. Choose the constraint to add.
            let (pc, orient) = if q < meq {
                let e = q;
                (e, if slack[e] > 0.0 { -1.0 } else { 1.0 })
            } else {
                let mut best: Option<(usize, f64)> = None;
                for i in meq..p {
                    if is_active[i] {
                        continue;
                    }
                    let nn = c.normal_norm(i);
                    let v = if nn > 0.0 {
                        slack[i] / nn
                    } else if slack[i] >= -tol {
                        continue;
                    } else {
                        f64::NEG_INFINITY
                    };
                    if best.is_none_or(|(_, bv)| v < bv) {
                        best = Some((i, v));
                    }
                }
                match best {
                    Some((i, v)) if v < -tol => (i, 1.0),
                    _ => break,
                }
            };
            c.normal(pc, &mut np);
            for v in np.iter_mut() {
                *v *= orient;
            }
            let rhs_p = orient * c.rhs(pc);
            let mut u_plus = u.clone();
            u_plus.push(0.0);
            // Inner loop: steps towards satisfying constraint pc.
            loop {
                // d = J' n_p; z = J[:, q..] d[q..]; r = R^{-1} d[..q]
                let d: Vec<f64> = (0..n).map(|c2| dot(&jt[row(c2)], &np)).collect();
                let mut z = vec![0.0; n];
                for c2 in q..n {
                    let dc = d[c2];
                    for (zi, v) in z.iter_mut().zip(&jt[row(c2)]) {
                        *zi += dc * v;
                    }
                }
                let mut r = vec![0.0; q];
                for i in (0..q).rev() {
                    let mut sum = d[i];
                    for k in (i + 1)..q {
                        sum -= r_mat[[i, k]] * r[k];
                    }
                    r[i] = sum / r_mat[[i, i]];
                }
                // Partial step length.
                let mut t1 = f64::INFINITY;
                let mut kd = None;
                for (pos, &ci) in active.iter().enumerate() {
                    if ci >= meq && r[pos] > 0.0 {
                        let v = u_plus[pos] / r[pos];
                        if v < t1 {
                            t1 = v;
                            kd = Some(pos);
                        }
                    }
                }
                // Full step length.
                let z_norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
                let np_norm = np.iter().map(|v| v * v).sum::<f64>().sqrt();
                let t2 = if z_norm <= 1e-12 * (1.0 + np_norm) {
                    f64::INFINITY
                } else {
                    let s_p: f64 = np.iter().zip(&x).map(|(a2, b2)| a2 * b2).sum::<f64>() - rhs_p;
                    let zn: f64 = z.iter().zip(&np).map(|(a2, b2)| a2 * b2).sum();
                    -s_p / zn
                };
                let t = t1.min(t2);
                if !t.is_finite() {
                    return Ok(QpOutcome::Infeasible);
                }
                if t2.is_finite() {
                    for (xi, zi) in x.iter_mut().zip(&z) {
                        *xi += t * zi;
                    }
                }
                for (pos, rv) in r.iter().enumerate() {
                    u_plus[pos] -= t * rv;
                }
                u_plus[q] += t;
                if t2 <= t1 {
                    // Full step: add pc.
                    let mut dv = d;
                    for jj in ((q + 1)..n).rev() {
                        let (cc, ss) = givens(dv[jj - 1], dv[jj]);
                        dv[jj - 1] = cc * dv[jj - 1] + ss * dv[jj];
                        dv[jj] = 0.0;
                        rotate_rows(jt, n, jj - 1, jj, cc, ss);
                    }
                    for i in 0..=q {
                        r_mat[[i, q]] = dv[i];
                    }
                    active.push(pc);
                    is_active[pc] = true;
                    sign.push(orient);
                    u = u_plus;
                    q += 1;
                    added += 1;
                    break;
                }
                // Partial step: drop active constraint kd.
                let kd = kd.expect("t1 finite has an index");
                is_active[active[kd]] = false;
                active.remove(kd);
                sign.remove(kd);
                u_plus.remove(kd);
                for col in kd..(q - 1) {
                    for i in 0..q {
                        r_mat[[i, col]] = r_mat[[i, col + 1]];
                    }
                }
                for i in 0..q {
                    r_mat[[i, q - 1]] = 0.0;
                }
                for jj in kd..q.saturating_sub(1) {
                    let (cc, ss) = givens(r_mat[[jj, jj]], r_mat[[jj + 1, jj]]);
                    for col in jj..(q - 1) {
                        let (a2, b2) = (r_mat[[jj, col]], r_mat[[jj + 1, col]]);
                        r_mat[[jj, col]] = cc * a2 + ss * b2;
                        r_mat[[jj + 1, col]] = -ss * a2 + cc * b2;
                    }
                    r_mat[[jj + 1, jj]] = 0.0;
                    rotate_rows(jt, n, jj, jj + 1, cc, ss);
                }
                for col in 0..n {
                    r_mat[[q - 1, col]] = 0.0;
                }
                q -= 1;
                dropped += 1;
                bump(&mut steps)?;
            }
        }

        // 3. Solution.
        c.slacks(&x, &mut slack);
        let mut max_violation = 0.0_f64;
        for (i, s) in slack.iter().enumerate() {
            let v = if i < meq { s.abs() } else { -s };
            max_violation = max_violation.max(v);
        }
        let mut multipliers = vec![0.0; p];
        for (pos, &ci) in active.iter().enumerate() {
            multipliers[ci] = sign[pos] * u[pos];
        }
        Ok(QpOutcome::Solved(QpSolution {
            x,
            multipliers,
            active,
            added,
            dropped,
            max_violation,
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute balance table for covariates.
fn compute_balance_table(
    x: &Array2<f64>,
    d: &Array1<f64>,
    var_names: &[String],
    weights: Option<&[f64]>,
) -> WeightItBalanceTable {
    let n = d.len();
    let k = x.ncols();

    let mut covariates = Vec::with_capacity(k);

    for j in 0..k {
        // Compute weighted means and variances for treated and control
        let mut sum_t = 0.0;
        let mut sum_c = 0.0;
        let mut w_sum_t = 0.0;
        let mut w_sum_c = 0.0;

        for i in 0..n {
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            let x_ij = x[[i, j]];

            if d[i] >= 0.5 {
                sum_t += w * x_ij;
                w_sum_t += w;
            } else {
                sum_c += w * x_ij;
                w_sum_c += w;
            }
        }

        let mean_t = if w_sum_t > 0.0 { sum_t / w_sum_t } else { 0.0 };
        let mean_c = if w_sum_c > 0.0 { sum_c / w_sum_c } else { 0.0 };

        // Compute weighted variances
        let mut var_t = 0.0;
        let mut var_c = 0.0;

        for i in 0..n {
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            let x_ij = x[[i, j]];

            if d[i] >= 0.5 {
                var_t += w * (x_ij - mean_t).powi(2);
            } else {
                var_c += w * (x_ij - mean_c).powi(2);
            }
        }

        var_t = if w_sum_t > 1.0 {
            var_t / (w_sum_t - 1.0)
        } else {
            0.0
        };
        var_c = if w_sum_c > 1.0 {
            var_c / (w_sum_c - 1.0)
        } else {
            0.0
        };

        // Standardized difference
        let pooled_sd = ((var_t + var_c) / 2.0).sqrt().max(1e-10);
        let std_diff = (mean_t - mean_c) / pooled_sd;

        // Variance ratio
        let var_ratio = if var_c > 1e-10 { var_t / var_c } else { 1.0 };

        covariates.push(WeightItCovariateBalance {
            variable: var_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("X{}", j)),
            mean_treated: mean_t,
            mean_control: mean_c,
            std_diff,
            var_ratio,
        });
    }

    // Summary statistics
    let max_std_diff = covariates
        .iter()
        .map(|c| c.std_diff.abs())
        .fold(0.0, f64::max);
    let mean_std_diff = covariates.iter().map(|c| c.std_diff.abs()).sum::<f64>() / k as f64;

    WeightItBalanceTable {
        covariates,
        max_std_diff,
        mean_std_diff,
    }
}

/// Compute effective sample size.
///
/// ESS = (Σ w)² / Σ w²
///
/// Returns (total ESS, treated ESS, control ESS).
fn compute_ess(weights: &[f64], d: &Array1<f64>) -> (f64, f64, f64) {
    let n = d.len();

    let mut w_sum_all = 0.0;
    let mut w_sq_sum_all = 0.0;
    let mut w_sum_t = 0.0;
    let mut w_sq_sum_t = 0.0;
    let mut w_sum_c = 0.0;
    let mut w_sq_sum_c = 0.0;

    for i in 0..n {
        let w = weights[i];
        w_sum_all += w;
        w_sq_sum_all += w * w;

        if d[i] >= 0.5 {
            w_sum_t += w;
            w_sq_sum_t += w * w;
        } else {
            w_sum_c += w;
            w_sq_sum_c += w * w;
        }
    }

    let ess_all = if w_sq_sum_all > 0.0 {
        w_sum_all * w_sum_all / w_sq_sum_all
    } else {
        0.0
    };
    let ess_t = if w_sq_sum_t > 0.0 {
        w_sum_t * w_sum_t / w_sq_sum_t
    } else {
        0.0
    };
    let ess_c = if w_sq_sum_c > 0.0 {
        w_sum_c * w_sum_c / w_sq_sum_c
    } else {
        0.0
    };

    (ess_all, ess_t, ess_c)
}

/// Trim extreme weights at specified quantile.
fn trim_weights(weights: &[f64], quantile: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    let lower_idx = ((1.0 - quantile) * n as f64).floor() as usize;
    let upper_idx = (quantile * n as f64).ceil() as usize;

    let lower_bound = sorted[lower_idx.min(n - 1)];
    let upper_bound = sorted[upper_idx.min(n - 1)];

    weights
        .iter()
        .map(|&w| w.max(lower_bound).min(upper_bound))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn expect_spec(err: EconError, needles: &[&str]) {
        match err {
            EconError::InvalidSpecification { message } => {
                for n in needles {
                    assert!(
                        message.contains(n),
                        "message `{message}` does not mention `{n}`"
                    )
                }
            }
            other => panic!("expected InvalidSpecification mentioning {needles:?}, got {other:?}"),
        }
    }

    use p2a_validation::{Check, Tol};

    const COVS: [&str; 3] = ["x1", "x2", "x3"];

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        DesignMatrix::extract_column(ds.df(), name)
            .unwrap()
            .to_vec()
    }

    /// The entries of `v` whose unit is in the treated (true) or control (false) group, in row order.
    fn group(v: &[f64], d: &[f64], treated: bool) -> Vec<f64> {
        v.iter()
            .zip(d)
            .filter(|&(_, &di)| (di >= 0.5) == treated)
            .map(|(&x, _)| x)
            .collect()
    }

    fn scaled(v: &[f64], f: f64) -> Vec<f64> {
        v.iter().map(|x| x * f).collect()
    }

    fn wmean(v: &[f64], w: &[f64]) -> f64 {
        v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
    }

    fn effect(y: &[f64], d: &[f64], w: &[f64]) -> f64 {
        wmean(&group(y, d, true), &group(w, d, true))
            - wmean(&group(y, d, false), &group(w, d, false))
    }

    fn n_zero(v: &[f64]) -> f64 {
        v.iter().filter(|&&x| x == 0.0).count() as f64
    }

    /// `validation/datasets/generated/weighting_sim.csv` (the data of the weighting cases) for
    /// the property tests, which pin no R value and so do not load a case.
    fn weighting_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/weighting_sim.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    #[test]
    fn test_validate_weightit_logistic() {
        // R: WeightIt 2.0.0 weightit(method = "glm", estimand = ATE | ATT | ATC, stabilize)
        // (case ipw_aipw_sim)
        let (case, ds) = crate::test_support::load_case("ipw_aipw_sim");
        let run = |est: WeightEstimand, stab: bool| {
            let mut c = WeightItConfig::default();
            c.estimand = est;
            c.stabilize = stab;
            weightit(&ds, "treat", &COVS, c).unwrap()
        };
        let (ate, att, atc, stab) = (
            run(WeightEstimand::ATE, false),
            run(WeightEstimand::ATT, false),
            run(WeightEstimand::ATC, false),
            run(WeightEstimand::ATE, true),
        );
        let n = ate.n_obs;
        Check::new(&case)
            .vector(
                "ps_head",
                &ate.propensity_scores.as_ref().unwrap()[..10],
                Tol::coef(n),
            )
            .vector("glm_ate_weights_head", &ate.weights[..12], Tol::coef(n))
            .vector("glm_att_weights_head", &att.weights[..12], Tol::coef(n))
            .vector("glm_atc_weights_head", &atc.weights[..12], Tol::coef(n))
            .vector(
                "glm_ate_stab_weights_head",
                &stab.weights[..12],
                Tol::coef(n),
            )
            .finish();
        assert!(ate.converged && ate.iterations >= 1);
        assert!(ate.convergence.as_ref().unwrap().converged);
    }

    #[test]
    fn test_weightit_logistic_reports_nonconvergence() {
        // CAUS-113: the logistic path used to hard-code converged = true, iterations = 0.
        // R: glm(maxit = 1) on this data reports converged FALSE after 1 iteration.
        let ds = weighting_sim();
        let mut c = WeightItConfig::default();
        c.max_iter = 1;
        let r = weightit(&ds, "treat", &COVS, c).unwrap();
        assert!(
            !r.converged && r.iterations == 1,
            "converged {} iterations {}",
            r.converged,
            r.iterations
        );
        assert!(
            r.warnings.iter().any(|w| w.contains("did not converge")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().ends_with(&notes_text(&r)));
    }

    #[test]
    fn test_weightit_logistic_reports_separation() {
        // R glm(epsilon = 1e-10, maxit = 100) warns "glm.fit: fitted probabilities
        // numerically 0 or 1 occurred"
        let df = df! { "treatment" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "x" => [-2.1, -1.2, -0.7, -0.3, 0.4, 0.9, 1.6, 2.2] }
        .unwrap();
        let r = weightit(
            &Dataset::new(df),
            "treatment",
            &["x"],
            WeightItConfig::default(),
        )
        .unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("fitted probabilities numerically 0 or 1")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_weightit_stabilize_ignored_outside_logistic_is_reported() {
        // R2; R WeightIt 2.0.0 ebal with stabilize = TRUE warns and returns the
        // unstabilised weights.
        let ds = weighting_sim();
        let run = |stab: bool| {
            let mut c = WeightItConfig::default();
            c.method = WeightMethod::Entropy;
            c.stabilize = stab;
            weightit(&ds, "treat", &COVS, c).unwrap()
        };
        let (a, b) = (run(true), run(false));
        assert_eq!(a.weights, b.weights);
        let tag = "stabilize cannot be used with entropy balancing weights";
        assert!(
            a.warnings.iter().any(|w| w.starts_with(tag)),
            "{:?}",
            a.warnings
        );
        assert!(
            !b.warnings.iter().any(|w| w.starts_with(tag)),
            "{:?}",
            b.warnings
        );
    }

    #[test]
    fn test_weightit_rejects_bad_iteration_controls() {
        // R3
        let ds = weighting_sim();
        let mut c = WeightItConfig::default();
        c.max_iter = 0;
        expect_spec(weightit(&ds, "treat", &COVS, c).unwrap_err(), &["max_iter"]);
        for t in [0.0, -1.0, f64::NAN] {
            let mut c = WeightItConfig::default();
            c.tolerance = t;
            expect_spec(
                weightit(&ds, "treat", &COVS, c).unwrap_err(),
                &["tolerance"],
            );
        }
    }

    #[test]
    fn test_weightit_rejects_bad_trim_quantile() {
        // A trim_quantile at or below 0.5 inverted the bounds (every weight collapsed to one
        // value), NaN skipped trimming silently, 0 clamped every weight to the maximum.
        let ds = weighting_sim();
        for q in [f64::NAN, f64::INFINITY, 0.5, 0.3, 0.0, -0.1, 1.01] {
            let mut c = WeightItConfig::default();
            c.trim_quantile = q;
            let err = weightit(&ds, "treat", &COVS, c).unwrap_err();
            expect_spec(err, &["trim_quantile", "(0.5, 1]"]);
        }
        for q in [0.99, 1.0] {
            let mut c = WeightItConfig::default();
            c.trim_quantile = q;
            assert!(
                weightit(&ds, "treat", &COVS, c).is_ok(),
                "trim_quantile {q}"
            );
        }
    }

    fn ebal(ds: &Dataset, est: WeightEstimand) -> WeightItResult {
        let mut c = WeightItConfig::default();
        c.method = WeightMethod::Entropy;
        c.estimand = est;
        c.tolerance = 1e-12;
        weightit(ds, "treat", &COVS, c).unwrap()
    }

    #[test]
    fn test_validate_ebal_ate() {
        // R: WeightIt 2.0.0 ebal ATE (cross-check) and the exact dual, Newton to |grad| < 1e-14
        // (case weights_ebal). Fails before (CAUS-49): ATE ran as ATT, so every treated
        // weight was 1.
        let (case, ds) = crate::test_support::load_case("weights_ebal");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = ebal(&ds, WeightEstimand::ATE);
        let n = r.n_obs;
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc, nf) = (wt.len() as f64, wc.len() as f64, n as f64);
        assert!(r.converged && r.estimand == WeightEstimand::ATE);
        // WP23-B: each reweighted group sums to the target population size
        assert!((wt.iter().sum::<f64>() - nf).abs() < 1e-9);
        assert!((wc.iter().sum::<f64>() - nf).abs() < 1e-9);
        Check::new(&case)
            .vector(
                "ate_exact_weights_t_head",
                &scaled(&wt[..8], nt / nf),
                Tol::coef(n),
            )
            .vector(
                "ate_exact_weights_c_head",
                &scaled(&wc[..8], nc / nf),
                Tol::coef(n),
            )
            .scalar("ate_exact_ess_t", r.ess_treated, Tol::stat(n))
            .scalar("ate_exact_ess_c", r.ess_control, Tol::stat(n))
            .scalar("ate_exact_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .vector("ate_weights_head", &r.weights[..12], Tol::new(1e-4, 0.0))
            .scalar(
                "ate_effect",
                effect(&y, &d, &r.weights),
                Tol::new(1e-4, 0.0),
            )
            .scalar(
                "ate_wmean_t_x1",
                wmean(&group(&col(&ds, "x1"), &d, true), &wt),
                Tol::new(1e-4, 1e-6),
            )
            .finish();
        for c in COVS {
            // exact balance: both groups reproduce the pooled means
            let x = col(&ds, c);
            let pooled = x.iter().sum::<f64>() / nf;
            assert!(
                (wmean(&group(&x, &d, true), &wt) - pooled).abs() < 1e-10,
                "{c}"
            );
            assert!(
                (wmean(&group(&x, &d, false), &wc) - pooled).abs() < 1e-10,
                "{c}"
            );
        }
    }

    #[test]
    fn test_validate_ebal_att() {
        // R: WeightIt 2.0.0 ebal ATT (cross-check) and the exact dual (case weights_ebal).
        // Fails before: the controls summed to n0 = 149 instead of n1 = 151 (WP23-B).
        let (case, ds) = crate::test_support::load_case("weights_ebal");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = ebal(&ds, WeightEstimand::ATT);
        let n = r.n_obs;
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc) = (wt.len() as f64, wc.len() as f64);
        assert!(r.converged && wt.iter().all(|&w| w == 1.0));
        assert!((wc.iter().sum::<f64>() - nt).abs() < 1e-9);
        Check::new(&case)
            .vector(
                "att_exact_weights_c_head",
                &scaled(&wc[..8], nc / nt),
                Tol::coef(n),
            )
            .scalar("att_exact_ess_c", r.ess_control, Tol::stat(n))
            .scalar("att_exact_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .vector("att_weights_head", &r.weights[..12], Tol::new(1e-4, 0.0))
            .scalar(
                "att_effect",
                effect(&y, &d, &r.weights),
                Tol::new(1e-4, 0.0),
            )
            .finish();
    }

    #[test]
    fn test_validate_ebal_atc() {
        // R: WeightIt 2.0.0 ebal ATC (cross-check) and the exact dual (case weights_ebal).
        // Fails before: ATC treated weights summed to n1 instead of n0 (WP23-B).
        let (case, ds) = crate::test_support::load_case("weights_ebal");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = ebal(&ds, WeightEstimand::ATC);
        let n = r.n_obs;
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc) = (wt.len() as f64, wc.len() as f64);
        assert!(r.converged && wc.iter().all(|&w| w == 1.0));
        assert!((wt.iter().sum::<f64>() - nc).abs() < 1e-9);
        Check::new(&case)
            .vector(
                "atc_exact_weights_t_head",
                &scaled(&wt[..8], nt / nc),
                Tol::coef(n),
            )
            .scalar("atc_exact_ess_t", r.ess_treated, Tol::stat(n))
            .scalar("atc_exact_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .vector("atc_weights_head", &r.weights[..12], Tol::new(1e-4, 0.0))
            .finish();
    }

    #[test]
    fn test_validate_entropy_balance_ess() {
        // CAUS-94: ESS was multiplied by n_control (86.72 x 149 = 12921.7); R: control ESS
        // of the exact dual (case weights_ebal).
        let (case, ds) = crate::test_support::load_case("weights_ebal");
        let d = col(&ds, "treat");
        let r = entropy_balance(&ds, "treat", &COVS, None).unwrap();
        let n = d.len();
        Check::new(&case)
            .scalar("att_exact_ess_c", r.effective_sample_size, Tol::stat(n))
            .scalar("att_exact_ess_c", r.ess_control, Tol::stat(n))
            .finish();
        assert_eq!(r.ess_treated, 151.0);
        assert_eq!(r.estimand, WeightEstimand::ATT);
        assert!((group(&r.weights, &d, false).iter().sum::<f64>() - 151.0).abs() < 1e-9);
    }

    #[test]
    fn test_entropy_balance_with_matches_weightit() {
        let ds = weighting_sim();
        for est in [
            WeightEstimand::ATE,
            WeightEstimand::ATT,
            WeightEstimand::ATC,
        ] {
            let c = EntropyBalanceConfig {
                estimand: est,
                tolerance: 1e-12,
                ..Default::default()
            };
            let a = entropy_balance_with(&ds, "treat", &COVS, &c).unwrap();
            let b = ebal(&ds, est);
            assert_eq!(a.estimand, est);
            for (x, y) in a.weights.iter().zip(&b.weights) {
                assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "{est:?}")
            }
        }
    }

    #[test]
    fn test_ebal_infeasible_is_error() {
        // CAUS-95 repro: treated x = 0, 0.5, 1; controls 2..6. Before: converged = true, NaN
        // control weights. R WeightIt returns all-zero control weights with a warning; p2a
        // refuses.
        let df = df! { "treatment" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x" => [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }
        .unwrap();
        let ds = Dataset::new(df);
        let mut c = WeightItConfig::default();
        c.method = WeightMethod::Entropy;
        c.estimand = WeightEstimand::ATT;
        expect_spec(
            weightit(&ds, "treatment", &["x"], c).unwrap_err(),
            &["infeasible", "x"],
        );
        expect_spec(
            entropy_balance(&ds, "treatment", &["x"], None).unwrap_err(),
            &["infeasible", "x"],
        );
        // target inside each coordinate's range but outside the controls' convex hull
        // (x1 + x2 <= 1):
        let df2 = df! { "treatment" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "x1" => [0.0, 1.0, 0.0, 0.2, 0.3, 0.1, 0.9, 0.8, 1.0],
        "x2" => [0.0, 0.0, 1.0, 0.2, 0.1, 0.3, 0.9, 1.0, 0.8] }
        .unwrap();
        expect_spec(
            entropy_balance(&Dataset::new(df2), "treatment", &["x1", "x2"], None).unwrap_err(),
            &["infeasible"],
        );
        // an empty group (entropy_balance had no check; weightit had one)
        let df3 = df! { "treatment" => [1.0, 1.0, 1.0], "x" => [0.1, 0.4, 0.3] }.unwrap();
        expect_spec(
            entropy_balance(&Dataset::new(df3), "treatment", &["x"], None).unwrap_err(),
            &["control"],
        );
    }

    #[test]
    fn test_entropy_on_infeasible_test_data_is_error() {
        // create_test_dataset(): the treated mean of x2 (0.5585) exceeds every control x2
        // (max 0.55), so no positive control weights balance it.
        let ds = create_test_dataset();
        let mut c = WeightItConfig::default();
        c.method = WeightMethod::Entropy;
        c.estimand = WeightEstimand::ATT;
        expect_spec(
            weightit(&ds, "treatment", &["x1", "x2"], c).unwrap_err(),
            &["infeasible", "x2"],
        );
        expect_spec(
            entropy_balance(&ds, "treatment", &["x1", "x2"], None).unwrap_err(),
            &["infeasible", "x2"],
        );
    }

    #[test]
    fn test_ebal_scale_invariant() {
        // Tier 3 convention 5: the scaled criterion converges in similar iteration counts at
        // any data scale.
        let ds = weighting_sim();
        let base = ebal(&ds, WeightEstimand::ATT);
        for s in [1e-6, 1e6] {
            let df = df! { "treat" => col(&ds, "treat"),
            "x1" => scaled(&col(&ds, "x1"), s), "x2" => scaled(&col(&ds, "x2"), s),
            "x3" => scaled(&col(&ds, "x3"), s) }
            .unwrap();
            let r = ebal(&Dataset::new(df), WeightEstimand::ATT);
            assert!(
                (r.iterations as i64 - base.iterations as i64).abs() <= 1,
                "scale {s}: {} vs {}",
                r.iterations,
                base.iterations
            );
            for (a, b) in r.weights.iter().zip(&base.weights) {
                assert!((a - b).abs() <= 1e-10 * b.abs(), "scale {s}")
            }
        }
    }

    #[test]
    fn test_ebal_drops_collinear_covariate() {
        // WeightIt drops linearly dependent covariates (make_full_rank); balance on them
        // follows
        let ds = weighting_sim();
        let x1 = col(&ds, "x1");
        let df = df! { "treat" => col(&ds, "treat"), "x1" => x1.clone(), "x2" => col(&ds, "x2"),
        "x3" => col(&ds, "x3"),
        "x4" => x1.iter().map(|v| 2.0 * v + 1.0).collect::<Vec<_>>() }
        .unwrap();
        let c = EntropyBalanceConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let r = entropy_balance_with(&Dataset::new(df), "treat", &["x1", "x2", "x3", "x4"], &c)
            .unwrap();
        let b = ebal(&ds, WeightEstimand::ATT);
        assert_eq!(r.dropped_covariates, vec!["x4".to_string()]);
        assert!(r.warnings.iter().any(|w| w.contains("x4")));
        for (a, b) in r.weights.iter().zip(&b.weights) {
            assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0))
        }
    }

    #[test]
    fn test_entropy_balance_with_rejects_bad_controls() {
        // R3
        let ds = weighting_sim();
        let bad = [
            (
                EntropyBalanceConfig {
                    max_iter: 0,
                    ..Default::default()
                },
                "max_iter",
            ),
            (
                EntropyBalanceConfig {
                    tolerance: f64::NAN,
                    ..Default::default()
                },
                "tolerance",
            ),
            (
                EntropyBalanceConfig {
                    targets: Some(vec![0.0, 0.0]),
                    ..Default::default()
                },
                "target",
            ),
            (
                EntropyBalanceConfig {
                    targets: Some(vec![0.0, f64::INFINITY, 0.5]),
                    ..Default::default()
                },
                "target",
            ),
        ];
        for (c, needle) in bad {
            expect_spec(
                entropy_balance_with(&ds, "treat", &COVS, &c).unwrap_err(),
                &[needle],
            );
        }
    }

    #[test]
    fn test_weight_method_parses_aliases_and_rejects_unknown() {
        WeightMethod::choices().validate().unwrap();
        for (s, m) in [
            ("logistic", WeightMethod::Logistic),
            ("GLM", WeightMethod::Logistic),
            ("ps", WeightMethod::Logistic),
            ("ebal", WeightMethod::Entropy),
            ("Entropy-Balancing", WeightMethod::Entropy),
            ("energy", WeightMethod::Energy),
            ("optweight", WeightMethod::Stable),
            ("SBW", WeightMethod::Stable),
        ] {
            assert_eq!(s.parse::<WeightMethod>().unwrap(), m, "{s}");
        }
        let err = "kernel".parse::<WeightMethod>().unwrap_err();
        expect_spec(err, &["method", "logistic", "entropy", "energy", "stable"]);
    }

    #[test]
    fn test_weight_estimand_parses_aliases_and_rejects_unknown() {
        WeightEstimand::choices().validate().unwrap();
        for (s, e) in [
            ("ATE", WeightEstimand::ATE),
            ("att", WeightEstimand::ATT),
            (" Atc ", WeightEstimand::ATC),
        ] {
            assert_eq!(s.parse::<WeightEstimand>().unwrap(), e);
        }
        expect_spec(
            "atu".parse::<WeightEstimand>().unwrap_err(),
            &["estimand", "ate", "att", "atc"],
        );
    }

    /// Create test dataset with imbalance but overlapping support.
    /// Overlap is required for propensity score methods to work correctly.
    fn create_test_dataset() -> Dataset {
        // Treated group has higher means, but ranges overlap with control
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // Treated mean ~0.65, control mean ~0.45, OVERLAPPING range [0.25, 0.95]
            "x1" => [
                0.5, 0.6, 0.55, 0.75, 0.6, 0.7, 0.5, 0.8, 0.65, 0.6,
                0.8, 0.85, 0.65, 0.75, 0.55, 0.7, 0.6, 0.9, 0.65, 0.68,
                0.3, 0.4, 0.35, 0.6, 0.25, 0.5, 0.28, 0.7, 0.35, 0.45,
                0.5, 0.55, 0.32, 0.75, 0.38, 0.45, 0.3, 0.65, 0.42, 0.52
            ],
            // Treated mean ~0.55, control mean ~0.35, OVERLAPPING range [0.18, 0.75]
            "x2" => [
                0.4, 0.5, 0.45, 0.65, 0.5, 0.6, 0.42, 0.72, 0.55, 0.5,
                0.62, 0.68, 0.52, 0.62, 0.48, 0.58, 0.5, 0.75, 0.55, 0.58,
                0.2, 0.3, 0.25, 0.45, 0.18, 0.38, 0.2, 0.5, 0.3, 0.28,
                0.35, 0.42, 0.22, 0.55, 0.28, 0.32, 0.2, 0.48, 0.32, 0.38
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_weightit_logistic_ate() {
        // create_test_dataset() is completely separated in (x1, x2): glm.fit drives the
        // propensity scores to 0/1 and reports it (all ATE weights 1). The properties are
        // checked on weighting_sim instead.
        let separated = weightit(
            &create_test_dataset(),
            "treatment",
            &["x1", "x2"],
            WeightItConfig::default(),
        )
        .unwrap();
        assert!(
            separated
                .warnings
                .iter()
                .any(|w| w.contains("fitted probabilities numerically 0 or 1"))
        );

        let dataset = weighting_sim();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATE,
            ..Default::default()
        };
        let result = weightit(&dataset, "treat", &COVS, config).unwrap();

        assert_eq!(result.n_obs, 300);
        assert_eq!(result.n_treated, 151);
        assert_eq!(result.n_control, 149);
        assert_eq!(result.weights.len(), 300);
        // ESS = (sum w)^2 / sum w^2 is at most n
        assert!(result.effective_sample_size > 0.0);
        assert!(result.effective_sample_size <= 300.0 * (1.0 + 1e-12));
        assert!(result.balance_after.max_std_diff < result.balance_before.max_std_diff);
    }

    #[test]
    fn test_weightit_logistic_att() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Treated weights should all be 1.0 for ATT
        for i in 0..40 {
            let is_treated = i < 20;
            if is_treated {
                assert!(
                    (result.weights[i] - 1.0).abs() < 1e-10,
                    "Treated weight should be 1.0, got {}",
                    result.weights[i]
                );
            }
        }
    }

    #[test]
    fn test_weightit_energy() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Energy,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };
        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();
        // Energy weights can be exactly 0; treated units keep weight 1.
        assert_eq!(result.weights.len(), 40);
        assert!(result.weights.iter().all(|&w| w >= 0.0 && w.is_finite()));
        let d = col(&dataset, "treatment");
        assert!(group(&result.weights, &d, true).iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_weightit_stable_on_infeasible_test_data_is_error() {
        // create_test_dataset's treated mean of x2 exceeds every control x2: exact balance is
        // infeasible, and the error names the column.
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Stable,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };
        expect_spec(
            weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap_err(),
            &["balance", "x2"],
        );
    }

    #[test]
    fn test_ess_computation() {
        let weights = vec![1.0, 1.0, 1.0, 1.0]; // Uniform weights
        let d = Array1::from(vec![1.0, 1.0, 0.0, 0.0]);

        let (ess_all, ess_t, ess_c) = compute_ess(&weights, &d);

        // Uniform weights should give ESS = n
        assert!((ess_all - 4.0).abs() < 1e-10);
        assert!((ess_t - 2.0).abs() < 1e-10);
        assert!((ess_c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_table() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.5, 1.2, 0.6, 1.1, 0.55, 0.5, 0.2, 0.6, 0.3, 0.55, 0.25,
            ],
        )
        .unwrap();
        let d = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let var_names = vec!["x1".to_string(), "x2".to_string()];

        let balance = compute_balance_table(&x, &d, &var_names, None);

        // Treated has higher means
        assert!(balance.covariates[0].mean_treated > balance.covariates[0].mean_control);
        assert!(balance.covariates[1].mean_treated > balance.covariates[1].mean_control);

        // Standardized difference should be positive (treated > control)
        assert!(balance.covariates[0].std_diff > 0.0);
    }

    #[test]
    fn test_weight_trimming() {
        let weights = vec![0.1, 0.5, 1.0, 2.0, 10.0];
        let trimmed = trim_weights(&weights, 0.8);

        // Extreme values should be trimmed
        assert!(trimmed.iter().all(|&w| (0.1..=10.0).contains(&w)));
    }

    #[test]
    fn test_display_result() {
        let dataset = create_test_dataset();
        let config = WeightItConfig::default();

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("WeightIt"));
        assert!(output.contains("ESS"));
        assert!(output.contains("Balance"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Validation tests against DGP-implied properties
    // ═══════════════════════════════════════════════════════════════════════════

    /// Create a larger dataset with known DGP for validation.
    ///
    /// DGP:
    ///   x1 ~ Uniform(0, 1), x2 ~ Uniform(0, 1)
    ///   P(T=1|X) = logistic(-0.5 + 1.0*x1 + 0.8*x2)
    ///   Treatment assigned based on propensity with deterministic threshold.
    ///   Y = 2.0 + 0.75*T + 1.5*x1 + 1.0*x2 + noise
    ///
    /// True ATE = 0.75
    fn create_validation_dataset() -> Dataset {
        let n = 100;
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);

        // Simple LCG-like deterministic sequence for reproducibility
        let mut seed: u64 = 42;
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..n {
            let v1 = lcg(&mut seed);
            let v2 = lcg(&mut seed);
            let noise = (lcg(&mut seed) - 0.5) * 0.4;

            let lp = -0.5 + 1.0 * v1 + 0.8 * v2;
            let ps = 1.0 / (1.0 + (-lp).exp());

            let t = if ps + (lcg(&mut seed) - 0.5) * 0.3 > 0.5 {
                1.0
            } else {
                0.0
            };

            let y = 2.0 + 0.75 * t + 1.5 * v1 + 1.0 * v2 + noise;

            x1.push(v1);
            x2.push(v2);
            treatment.push(t);
            outcome.push(y);
        }

        let df = df! {
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2,
            "outcome" => outcome
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_entropy_weights_positive_finite() {
        // ATT on the validation dataset (feasible): weights are positive and finite, and the
        // controls sum to n1 (WP23-B).
        let dataset = create_validation_dataset();

        let result = entropy_balance(&dataset, "treatment", &["x1", "x2"], None).unwrap();

        assert!(
            result.weights.iter().all(|&w| w > 0.0 && w.is_finite()),
            "Entropy balance weights should be positive and finite"
        );
        let df = dataset.df();
        let d = df
            .column("treatment")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();
        let n1 = d.into_iter().flatten().filter(|&t| t >= 0.5).count() as f64;
        let control_sum: f64 = d
            .into_iter()
            .enumerate()
            .filter(|(_, t)| t.unwrap() < 0.5)
            .map(|(i, _)| result.weights[i])
            .sum();
        assert!(
            (control_sum - n1).abs() < 1e-8 * n1,
            "controls sum to {control_sum}, want n1 = {n1}"
        );
    }

    #[test]
    fn test_logistic_weights_improve_balance() {
        // Validate: logistic IPW should improve balance relative to raw data.
        let dataset = create_validation_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATE,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        assert!(
            result.balance_after.max_std_diff < result.balance_before.max_std_diff,
            "Logistic IPW should improve balance: before={:.4}, after={:.4}",
            result.balance_before.max_std_diff,
            result.balance_after.max_std_diff
        );
    }

    #[test]
    fn test_att_treated_weights_are_one() {
        // Validate: for ATT estimation, treated units should have weight = 1.0
        // (they serve as the reference distribution). This holds for all methods.
        let dataset = create_validation_dataset();

        let df = dataset.df();
        let t_col = df
            .column("treatment")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();

        for method in [WeightMethod::Logistic, WeightMethod::Entropy] {
            let config = WeightItConfig {
                method,
                estimand: WeightEstimand::ATT,
                max_iter: 500,
                tolerance: 1e-4,
                ..Default::default()
            };

            let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

            for i in 0..result.n_obs {
                let t = t_col.get(i).unwrap_or(0.0);
                if t >= 0.5 {
                    assert!(
                        (result.weights[i] - 1.0).abs() < 1e-8,
                        "ATT treated weight at index {} should be 1.0, got {} (method: {:?})",
                        i,
                        result.weights[i],
                        method
                    );
                }
            }
        }
    }

    #[test]
    fn test_weighted_ate_recovers_dgp() {
        // Validate: using IPW weights to compute weighted difference in means
        // should recover an ATE estimate close to the true value (0.75).
        let dataset = create_validation_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATE,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        let df = dataset.df();
        let t_col = df
            .column("treatment")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();
        let y_col = df
            .column("outcome")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();

        let mut sum_wy_t = 0.0;
        let mut sum_w_t = 0.0;
        let mut sum_wy_c = 0.0;
        let mut sum_w_c = 0.0;

        for i in 0..result.n_obs {
            let t = t_col.get(i).unwrap();
            let y = y_col.get(i).unwrap();
            let w = result.weights[i];

            if t >= 0.5 {
                sum_wy_t += w * y;
                sum_w_t += w;
            } else {
                sum_wy_c += w * y;
                sum_w_c += w;
            }
        }

        let ate_hat = sum_wy_t / sum_w_t - sum_wy_c / sum_w_c;

        // True ATE = 0.75. With n=100, allow tolerance of 0.5.
        assert!(
            (ate_hat - 0.75).abs() < 0.5,
            "Weighted ATE ({:.4}) should be within 0.5 of true ATE (0.75)",
            ate_hat
        );
    }

    #[test]
    fn test_ess_properties() {
        // ESS = (sum w)^2 / (sum w^2) is in (0, n] for every method; weights are >= 0 and
        // finite. On weighting_sim, where every balancing problem is feasible.
        let dataset = weighting_sim();
        for method in [
            WeightMethod::Logistic,
            WeightMethod::Entropy,
            WeightMethod::Energy,
            WeightMethod::Stable,
        ] {
            let config = WeightItConfig {
                method,
                estimand: WeightEstimand::ATE,
                ..Default::default()
            };
            let result = weightit(&dataset, "treat", &COVS, config).unwrap();
            assert!(
                result.effective_sample_size > 0.0
                    && result.effective_sample_size <= result.n_obs as f64 * (1.0 + 1e-12),
                "ESS {} for {method:?}",
                result.effective_sample_size
            );
            assert!(
                result.weights.iter().all(|&w| w >= 0.0 && w.is_finite()),
                "weights for {method:?}"
            );
        }
    }

    #[test]
    fn test_entropy_balance_at_least_as_good_as_logistic() {
        // ATT on weighting_sim (feasible): entropy balancing balances the means exactly, so its
        // largest standardised difference is at most that of the logistic weights.
        let dataset = weighting_sim();
        let run = |method| {
            weightit(
                &dataset,
                "treat",
                &COVS,
                WeightItConfig {
                    method,
                    estimand: WeightEstimand::ATT,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let entropy = run(WeightMethod::Entropy);
        let logistic = run(WeightMethod::Logistic);
        assert!(
            entropy.balance_after.max_std_diff <= 1e-8,
            "entropy max_std_diff {}",
            entropy.balance_after.max_std_diff
        );
        assert!(1e-8 <= logistic.balance_after.max_std_diff + 1e-8);
        assert!(entropy.balance_after.max_std_diff <= logistic.balance_after.max_std_diff + 1e-8);
    }

    #[test]
    fn test_atc_control_weights_are_one() {
        // Validate: for ATC estimation, control units should have weight = 1.0.
        let dataset = create_validation_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATC,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        let df = dataset.df();
        let t_col = df
            .column("treatment")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();

        for i in 0..result.n_obs {
            let t = t_col.get(i).unwrap_or(1.0);
            if t < 0.5 {
                assert!(
                    (result.weights[i] - 1.0).abs() < 1e-8,
                    "ATC control weight at index {} should be 1.0, got {}",
                    i,
                    result.weights[i]
                );
            }
        }
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_weightit_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| {
            weightit(ds, d, x, WeightItConfig::default())
        };
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.weights.clone());
    }

    #[test]
    fn test_validate_na_entropy_balance_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| entropy_balance(ds, d, x, None);
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.weights.clone());
    }

    // ── Task 5: dense QP, energy and stable ─────────────────────────────────────────────

    fn cancelled_budget() -> resource::Budget {
        let t = resource::CancelToken::new();
        t.cancel();
        resource::Budget::new(None, t)
    }

    fn small_qp(last: ([f64; 3], f64)) -> (Array2<f64>, Vec<f64>, dense_qp::DenseConstraints) {
        let g = ndarray::array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let normals = ndarray::array![
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            last.0
        ];
        (
            g,
            vec![-1.0, -2.0, 0.5],
            dense_qp::DenseConstraints {
                normals,
                rhs: vec![1.0, 0.0, 0.0, 0.0, last.1],
                meq: 1,
            },
        )
    }

    #[test]
    fn test_dense_qp_matches_quadprog() {
        // R: quadprog::solve.QP(G, -a, N, b, meq = 1): x = (0.2, 0.8, 0), value -0.6, active
        // {equality, x3 >= 0}, multipliers 0.6 and 0.7
        let (g, a, c) = small_qp(([1.0, 0.0, -1.0], 0.1));
        let dense_qp::QpOutcome::Solved(s) = dense_qp::solve_dense_qp(g, &a, &c, 100).unwrap()
        else {
            panic!("not solved")
        };
        for (got, want) in s.x.iter().zip([0.2, 0.8, 0.0]) {
            assert!((got - want).abs() < 1e-12)
        }
        let mut act = s.active.clone();
        act.sort();
        assert_eq!(act, vec![0, 3]);
        assert!((s.multipliers[0] - 0.6).abs() < 1e-12 && (s.multipliers[3] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_dense_qp_detects_infeasibility() {
        // x1 + x2 + x3 = 1, x >= 0, x1 >= 1.5: R solve.QP "constraints are inconsistent"
        let (g, a, c) = small_qp(([1.0, 0.0, 0.0], 1.5));
        assert!(matches!(
            dense_qp::solve_dense_qp(g, &a, &c, 100).unwrap(),
            dense_qp::QpOutcome::Infeasible
        ));
    }

    #[test]
    fn test_dense_qp_kkt_on_seeded_problems() {
        // Property test (no R values): 60 seeded strictly convex QPs that are feasible by
        // construction (x0 satisfies every constraint), so every outcome must be Solved and
        // satisfy the KKT conditions.
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(2026);
        for case in 0..60 {
            let n = rng.gen_range(2..=12);
            let p = rng.gen_range(1..=20);
            let meq = rng.gen_range(0..=2usize.min(n - 1).min(p));
            let m = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0..1.0));
            let g = m.t().dot(&m) + Array2::<f64>::eye(n) * 0.1;
            let a: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let normals = Array2::from_shape_fn((p, n), |_| rng.gen_range(-1.0..1.0));
            let x0 = Array1::from_shape_fn(n, |_| rng.gen_range(-1.0..1.0));
            let at_x0 = normals.dot(&x0);
            let rhs: Vec<f64> = (0..p)
                .map(|i| {
                    if i < meq {
                        at_x0[i]
                    } else {
                        at_x0[i] - rng.gen_range(0.0..1.0)
                    }
                })
                .collect();
            let c = dense_qp::DenseConstraints {
                normals: normals.clone(),
                rhs: rhs.clone(),
                meq,
            };
            let dense_qp::QpOutcome::Solved(s) =
                dense_qp::solve_dense_qp(g.clone(), &a, &c, 1000).unwrap()
            else {
                panic!("case {case}: a feasible problem was not solved")
            };
            let (x, u) = (
                Array1::from(s.x.clone()),
                Array1::from(s.multipliers.clone()),
            );
            let stat = g.dot(&x) + Array1::from(a.clone()) - normals.t().dot(&u);
            assert!(
                stat.iter().all(|v| v.abs() < 1e-10),
                "case {case}: stationarity {stat:?}"
            );
            let slack = normals.dot(&x) - Array1::from(rhs);
            for i in 0..p {
                if i < meq {
                    assert!(slack[i].abs() < 1e-10, "case {case}: equality {i}")
                } else {
                    assert!(
                        slack[i] > -1e-10 && u[i] > -1e-12,
                        "case {case}: inequality {i}"
                    )
                }
                assert!(
                    (u[i] * slack[i]).abs() < 1e-10,
                    "case {case}: complementarity {i}"
                );
            }
        }
    }

    fn wfit(ds: &Dataset, m: WeightMethod, est: WeightEstimand) -> WeightItResult {
        let c = WeightItConfig {
            method: m,
            estimand: est,
            ..Default::default()
        };
        weightit(ds, "treat", &COVS, c).unwrap()
    }

    type Rows = Vec<Vec<f64>>;

    /// Rows of the covariates standardised over all n units ((x - mean)/sd, sd with n - 1,
    /// R's scale()): (treated rows, control rows, all rows), each in data order.
    fn standardised_rows(ds: &Dataset, d: &[f64]) -> (Rows, Rows, Rows) {
        let cols: Vec<Vec<f64>> = COVS
            .iter()
            .map(|c| {
                let x = col(ds, c);
                let n = x.len() as f64;
                let m = x.iter().sum::<f64>() / n;
                let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
                x.iter().map(|v| (v - m) / sd).collect()
            })
            .collect();
        let all: Rows = (0..d.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        let pick = |t: bool| -> Rows {
            all.iter()
                .zip(d)
                .filter(|&(_, &di)| (di >= 0.5) == t)
                .map(|(r, _)| r.clone())
                .collect()
        };
        (pick(true), pick(false), all.clone())
    }

    fn euclid(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Energy distance between samples a and b with mean-1 weights (the case's .edist).
    fn edist(xa: &[Vec<f64>], wa: &[f64], xb: &[Vec<f64>], wb: &[f64]) -> f64 {
        let quad = |x: &[Vec<f64>], w: &[f64], y: &[Vec<f64>], v: &[f64]| -> f64 {
            x.iter()
                .zip(w)
                .map(|(r, &wi)| {
                    y.iter()
                        .zip(v)
                        .map(|(s, &vj)| wi * vj * euclid(r, s))
                        .sum::<f64>()
                })
                .sum()
        };
        let (na, nb) = (xa.len() as f64, xb.len() as f64);
        2.0 / (na * nb) * quad(xa, wa, xb, wb)
            - quad(xa, wa, xa, wa) / (na * na)
            - quad(xb, wb, xb, wb) / (nb * nb)
    }

    #[test]
    fn test_validate_energy_att() {
        // R: Huling & Mak (2022) energy QP solved with quadprog (case weights_energy).
        // Fails before (CAUS-50): weights were 1/(||x - treated mean|| + 0.1).
        let (case, ds) = crate::test_support::load_case("weights_energy");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = wfit(&ds, WeightMethod::Energy, WeightEstimand::ATT);
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc, n) = (wt.len() as f64, wc.len() as f64, d.len());
        let wc1 = scaled(&wc, nc / nt); // R: mean-1 weights; p2a: controls sum to n1
        let means: Vec<f64> = COVS
            .iter()
            .map(|c| wmean(&group(&col(&ds, c), &d, false), &wc))
            .collect();
        let (xs_t, xs_c, _xs_all) = standardised_rows(&ds, &d);
        Check::new(&case)
            .vector("att_weights_head", &wc1[..12], Tol::new(1e-8, 1e-10))
            .vector("att_weighted_means", &means, Tol::new(1e-8, 1e-12))
            .scalar("att_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .scalar("att_n_zero", n_zero(&wc), Tol::exact())
            .scalar(
                "att_energy",
                edist(&xs_c, &wc1, &xs_t, &vec![1.0; wt.len()]),
                Tol::coef(n),
            )
            .finish();
        assert!(wt.iter().all(|&w| w == 1.0) && r.converged);
    }

    #[test]
    fn test_validate_energy_atc() {
        // R: the treated reweighted towards the controls (case weights_energy)
        let (case, ds) = crate::test_support::load_case("weights_energy");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = wfit(&ds, WeightMethod::Energy, WeightEstimand::ATC);
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc, n) = (wt.len() as f64, wc.len() as f64, d.len());
        let wt1 = scaled(&wt, nt / nc);
        let means: Vec<f64> = COVS
            .iter()
            .map(|c| wmean(&group(&col(&ds, c), &d, true), &wt))
            .collect();
        let (xs_t, xs_c, _xs_all) = standardised_rows(&ds, &d);
        Check::new(&case)
            .vector("atc_weights_head", &wt1[..12], Tol::new(1e-8, 1e-10))
            .vector("atc_weighted_means", &means, Tol::new(1e-8, 1e-12))
            .scalar("atc_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .scalar("atc_n_zero", n_zero(&wt), Tol::exact())
            .scalar(
                "atc_energy",
                edist(&xs_t, &wt1, &xs_c, &vec![1.0; wc.len()]),
                Tol::coef(n),
            )
            .finish();
        assert!(wc.iter().all(|&w| w == 1.0) && r.converged);
    }

    #[test]
    fn test_validate_energy_ate() {
        // R: WeightIt weightit2energy's improved ATE at lambda 0, solved with quadprog (case
        // weights_energy)
        let (case, ds) = crate::test_support::load_case("weights_energy");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let r = wfit(&ds, WeightMethod::Energy, WeightEstimand::ATE);
        let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
        let (nt, nc, n) = (wt.len() as f64, wc.len() as f64, d.len());
        let nf = n as f64;
        let head: Vec<f64> = (0..12)
            .map(|i| {
                let ng = if d[i] >= 0.5 { nt } else { nc };
                r.weights[i] * ng / nf
            })
            .collect();
        let means_t: Vec<f64> = COVS
            .iter()
            .map(|c| wmean(&group(&col(&ds, c), &d, true), &wt))
            .collect();
        let means_c: Vec<f64> = COVS
            .iter()
            .map(|c| wmean(&group(&col(&ds, c), &d, false), &wc))
            .collect();
        let (xs_t, xs_c, xs_all) = standardised_rows(&ds, &d);
        let (wt1, wc1) = (scaled(&wt, nt / nf), scaled(&wc, nc / nf));
        let ones = vec![1.0; n];
        let energy = edist(&xs_t, &wt1, &xs_all, &ones)
            + edist(&xs_c, &wc1, &xs_all, &ones)
            + edist(&xs_t, &wt1, &xs_c, &wc1);
        Check::new(&case)
            .vector("ate_weights_head", &head, Tol::new(1e-8, 1e-10))
            .vector("ate_weighted_means_t", &means_t, Tol::new(1e-8, 1e-12))
            .vector("ate_weighted_means_c", &means_c, Tol::new(1e-8, 1e-12))
            .scalar("ate_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .scalar("ate_n_zero", n_zero(&r.weights), Tol::exact())
            .scalar("ate_energy", energy, Tol::coef(n))
            .finish();
        assert!((wt.iter().sum::<f64>() - nf).abs() < 1e-9);
        assert!((wc.iter().sum::<f64>() - nf).abs() < 1e-9);
        assert!(r.converged);
    }

    #[test]
    fn test_energy_estimands_differ() {
        // convention 2; fails before: ATE and ATC returned the ATT weights
        let ds = weighting_sim();
        let att = wfit(&ds, WeightMethod::Energy, WeightEstimand::ATT);
        for est in [WeightEstimand::ATE, WeightEstimand::ATC] {
            let r = wfit(&ds, WeightMethod::Energy, est);
            assert!(
                r.weights
                    .iter()
                    .zip(&att.weights)
                    .any(|(a, b)| (a - b).abs() > 1e-6),
                "{est:?}"
            );
        }
    }

    #[test]
    fn test_energy_merges_duplicate_rows() {
        // Two identical control rows (x = 1.1) share one merged weight; hand-sized R reference
        // (merged QP with quadprog): control weights in the p2a scale (sum n1 = 3) are 2/9,
        // 13/18, 8/9, 4/9, 13/18. The unmerged Hessian is singular.
        let df = df! { "treat" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x" => [0.9, 1.3, 2.1, 0.2, 1.1, 1.7, 2.6, 1.1] }
        .unwrap();
        let c = WeightItConfig {
            method: WeightMethod::Energy,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };
        let r = weightit(&Dataset::new(df), "treat", &["x"], c).unwrap();
        let want = [
            1.0,
            1.0,
            1.0,
            2.0 / 9.0,
            13.0 / 18.0,
            8.0 / 9.0,
            4.0 / 9.0,
            13.0 / 18.0,
        ];
        for (i, (got, w)) in r.weights.iter().zip(want).enumerate() {
            assert!((got - w).abs() < 1e-12, "unit {i}: {got}")
        }
        assert!(
            !r.warnings.iter().any(|w| w.contains("ridge")),
            "exact duplicates are merged, not ridged: {:?}",
            r.warnings
        );
    }

    #[test]
    fn test_energy_refuses_too_many_units() {
        // decision WP23-A: one control more than the cap
        let n = ENERGY_MAX_UNITS + 3;
        let df =
            df! { "treat" => (0..n).map(|i| if i < 2 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(),
            "x" => (0..n).map(|i| (i as f64 * 0.37).sin()).collect::<Vec<f64>>() }
            .unwrap();
        let c = WeightItConfig {
            method: WeightMethod::Energy,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };
        expect_spec(
            weightit(&Dataset::new(df), "treat", &["x"], c).unwrap_err(),
            &["energy", "at most 2000 units"],
        );
    }

    #[test]
    fn test_energy_honours_cancellation() {
        let ds = weighting_sim();
        let c = WeightItConfig {
            method: WeightMethod::Energy,
            estimand: WeightEstimand::ATT,
            ..Default::default()
        };
        let r = resource::with_budget(cancelled_budget(), || {
            weightit(&ds, "treat", &COVS, c.clone())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    #[test]
    fn test_validate_stable_weights() {
        // WeightIt 'stable' = the SBW QP with exact balance (optweight, tols = 0); case
        // weights_sbw. Fails before (CAUS-50): penalised gradient descent.
        let (case, ds) = crate::test_support::load_case("weights_sbw");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let n = d.len();
        let nf = n as f64;
        let att = wfit(&ds, WeightMethod::Stable, WeightEstimand::ATT);
        let atc = wfit(&ds, WeightMethod::Stable, WeightEstimand::ATC);
        let ate = wfit(&ds, WeightMethod::Stable, WeightEstimand::ATE);
        let (nt, nc) = (
            group(&d, &d, true).len() as f64,
            group(&d, &d, false).len() as f64,
        );
        Check::new(&case)
            .vector(
                "att_exact_weights_head",
                &scaled(&group(&att.weights, &d, false)[..12], 1.0 / nt),
                Tol::new(1e-8, 1e-12),
            )
            .scalar(
                "att_exact_effect",
                effect(&y, &d, &att.weights),
                Tol::coef(n),
            )
            .vector(
                "atc_exact_weights_head",
                &scaled(&group(&atc.weights, &d, true)[..12], 1.0 / nc),
                Tol::new(1e-8, 1e-12),
            )
            .scalar(
                "atc_exact_effect",
                effect(&y, &d, &atc.weights),
                Tol::coef(n),
            )
            .vector(
                "ate_exact_weights_t_head",
                &scaled(&group(&ate.weights, &d, true)[..8], 1.0 / nf),
                Tol::new(1e-8, 1e-12),
            )
            .vector(
                "ate_exact_weights_c_head",
                &scaled(&group(&ate.weights, &d, false)[..8], 1.0 / nf),
                Tol::new(1e-8, 1e-12),
            )
            .scalar(
                "ate_exact_effect",
                effect(&y, &d, &ate.weights),
                Tol::coef(n),
            )
            .finish();
    }
}
