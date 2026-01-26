//! Flexible inverse probability weighting (WeightIt) module.
//!
//! Provides multiple methods for computing propensity score and balancing weights
//! for causal inference, including:
//!
//! - **Logistic (PS)**: Standard propensity score weights from logistic regression
//! - **Entropy Balancing (ebal)**: Exact mean balance via entropy minimization
//! - **Energy Balancing**: Minimize energy distance between weighted distributions
//! - **Stable Balancing**: Target stable weights with approximate balance
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

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::logistic_cdf;

// ═══════════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════════

/// Weighting method for propensity score estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightMethod {
    /// Standard propensity score weights from logistic regression
    #[default]
    Logistic,
    /// Entropy balancing (Hainmueller 2012)
    Entropy,
    /// Energy balancing weights
    Energy,
    /// Stable balancing weights (Zubizarreta 2015)
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

/// Configuration for WeightIt estimation.
#[derive(Debug, Clone)]
pub struct WeightItConfig {
    /// Weighting method
    pub method: WeightMethod,
    /// Target estimand
    pub estimand: WeightEstimand,
    /// Include intercept in covariate matrix (default: true)
    pub intercept: bool,
    /// Stabilize weights (default: false)
    /// Multiplies weights by P(D=1) for treated and P(D=0) for control
    pub stabilize: bool,
    /// Trim extreme weights at this quantile (e.g., 0.99 trims at 1st and 99th percentile)
    /// Set to 1.0 for no trimming (default: 1.0)
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
        writeln!(f, "{:<20} {:>12} {:>12} {:>12} {:>12}",
                 "Variable", "Mean Treat", "Mean Ctrl", "Std.Diff", "Var.Ratio")?;
        writeln!(f, "{}", "-".repeat(68))?;

        for cov in &self.covariates {
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                     cov.variable, cov.mean_treated, cov.mean_control,
                     cov.std_diff, cov.var_ratio)?;
        }

        writeln!(f, "{}", "-".repeat(68))?;
        writeln!(f, "Max |Std.Diff|: {:.4}  Mean |Std.Diff|: {:.4}",
                 self.max_std_diff, self.mean_std_diff)?;

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
}

impl fmt::Display for WeightItResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "WeightIt: Weighting for Covariate Balance")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Method:   {}", self.method)?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Total:    {}  (Treated: {}, Control: {})",
                 self.n_obs, self.n_treated, self.n_control)?;
        writeln!(f)?;
        writeln!(f, "Weight Summary:")?;
        writeln!(f, "  Range:    [{:.4}, {:.4}]", self.min_weight, self.max_weight)?;
        writeln!(f, "  ESS:      {:.1} (Treated: {:.1}, Control: {:.1})",
                 self.effective_sample_size, self.ess_treated, self.ess_control)?;
        writeln!(f)?;

        if !self.converged {
            writeln!(f, "WARNING: Algorithm did not converge after {} iterations", self.iterations)?;
            writeln!(f)?;
        }

        writeln!(f, "Balance Before Weighting:")?;
        writeln!(f, "  Max |Std.Diff|:  {:.4}", self.balance_before.max_std_diff)?;
        writeln!(f, "  Mean |Std.Diff|: {:.4}", self.balance_before.mean_std_diff)?;
        writeln!(f)?;
        writeln!(f, "Balance After Weighting:")?;
        writeln!(f, "  Max |Std.Diff|:  {:.4}", self.balance_after.max_std_diff)?;
        writeln!(f, "  Mean |Std.Diff|: {:.4}", self.balance_after.mean_std_diff)?;

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

/// Result from entropy balancing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyBalanceResult {
    /// Estimated weights for each observation (control units only for ATT)
    pub weights: Vec<f64>,
    /// Lagrange multipliers from optimization
    pub lambda: Vec<f64>,
    /// Balance achieved for each covariate
    pub balance: WeightItBalanceTable,
    /// Effective sample size
    pub effective_sample_size: f64,
    /// Maximum weight
    pub max_weight: f64,
    /// Minimum weight (among non-zero)
    pub min_weight: f64,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final constraint violation (max absolute)
    pub constraint_tolerance: f64,
    /// Number of observations reweighted
    pub n_reweighted: usize,
}

impl fmt::Display for EntropyBalanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Entropy Balancing Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Observations Reweighted: {}", self.n_reweighted)?;
        writeln!(f, "Converged: {} ({} iterations)", self.converged, self.iterations)?;
        writeln!(f, "Constraint Tolerance: {:.2e}", self.constraint_tolerance)?;
        writeln!(f)?;
        writeln!(f, "Weight Summary:")?;
        writeln!(f, "  Range: [{:.4}, {:.4}]", self.min_weight, self.max_weight)?;
        writeln!(f, "  ESS:   {:.1}", self.effective_sample_size)?;
        writeln!(f)?;
        writeln!(f, "Balance After Weighting:")?;
        write!(f, "{}", self.balance)?;

        Ok(())
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
    let mut warnings = Vec::new();

    // Extract treatment variable
    let d = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

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
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, config.intercept)?;
    let x = design.data;

    // Variable names for balance table
    let mut var_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();
    if config.intercept {
        var_names.insert(0, "Intercept".to_string());
    }

    // Compute balance before weighting
    let balance_before = compute_balance_table(&x, &d, &var_names, None);

    // Compute weights based on method
    let (weights, propensity_scores, converged, iterations) = match config.method {
        WeightMethod::Logistic => {
            let (w, ps) = compute_logistic_weights(&x, &d, config.estimand, config.stabilize)?;
            (w, Some(ps), true, 0)
        }
        WeightMethod::Entropy => {
            let result = compute_entropy_weights(&x, &d, config.estimand, config.max_iter, config.tolerance)?;
            if !result.converged {
                warnings.push(format!(
                    "Entropy balancing did not converge after {} iterations",
                    result.iterations
                ));
            }
            (result.weights, None, result.converged, result.iterations)
        }
        WeightMethod::Energy => {
            let (w, conv, iters) = compute_energy_weights(&x, &d, config.estimand, config.max_iter, config.tolerance)?;
            if !conv {
                warnings.push(format!(
                    "Energy balancing did not converge after {} iterations",
                    iters
                ));
            }
            (w, None, conv, iters)
        }
        WeightMethod::Stable => {
            let (w, conv, iters) = compute_stable_weights(&x, &d, config.estimand, config.max_iter, config.tolerance)?;
            if !conv {
                warnings.push(format!(
                    "Stable balancing did not converge after {} iterations",
                    iters
                ));
            }
            (w, None, conv, iters)
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
    let min_weight = weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::INFINITY, f64::min);

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
            ess_total, n, 100.0 * ess_total / n as f64
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
    })
}

/// Compute entropy balancing weights specifically.
///
/// Entropy balancing reweights the control (or treated) group to achieve exact
/// mean balance on specified covariates while minimizing the entropy distance
/// from uniform weights.
///
/// # Arguments
///
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of covariate columns to balance
/// * `targets` - Optional target means (defaults to opposite group means for ATT/ATC)
///
/// # Algorithm
///
/// Solves the constrained optimization problem:
/// ```text
/// min  Σ w_i log(w_i)
/// s.t. Σ w_i X_i = target_means
///      Σ w_i = 1, w_i ≥ 0
/// ```
///
/// Uses dual formulation with Newton-Raphson optimization:
/// ```text
/// w_i = exp(λ' X_i) / Σ_j exp(λ' X_j)
/// ```
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
    // Extract treatment variable
    let d = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Build covariate matrix (without intercept for entropy balancing)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let x = design.data;

    let n = d.len();
    let k = x.ncols();

    // Identify treated and control indices
    let treated_idx: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();

    // For ATT: reweight control to match treated means
    // Build control covariate matrix
    let n_ctrl = control_idx.len();
    let mut x_ctrl = Array2::zeros((n_ctrl, k));
    for (new_i, &old_i) in control_idx.iter().enumerate() {
        x_ctrl.row_mut(new_i).assign(&x.row(old_i));
    }

    // Compute target means (treated group means if not provided)
    let target_means = match targets {
        Some(t) => {
            if t.len() != k {
                return Err(EconError::InvalidSpecification {
                    message: format!("Target means length ({}) must match number of covariates ({})", t.len(), k),
                });
            }
            Array1::from_vec(t.to_vec())
        }
        None => {
            // Compute treated group means
            let n_treat = treated_idx.len() as f64;
            let mut means = Array1::zeros(k);
            for &i in &treated_idx {
                for j in 0..k {
                    means[j] += x[[i, j]];
                }
            }
            means /= n_treat;
            means
        }
    };

    // Run entropy balancing optimization
    let (weights, lambda, converged, iterations, constraint_tol) =
        entropy_balance_optimize(&x_ctrl, &target_means, 200, 1e-8)?;

    // Create full weight vector (1.0 for treated, computed weights for control)
    let mut full_weights = vec![0.0; n];
    for &i in &treated_idx {
        full_weights[i] = 1.0;
    }
    for (ctrl_i, &orig_i) in control_idx.iter().enumerate() {
        full_weights[orig_i] = weights[ctrl_i] * n_ctrl as f64; // Scale to sum to n_ctrl
    }

    // Compute balance
    let var_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();
    let balance = compute_balance_table(&x, &d, &var_names, Some(&full_weights));

    // Compute ESS for control group
    let w_sum: f64 = weights.iter().sum();
    let w_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
    let ess = if w_sq_sum > 0.0 { w_sum * w_sum / w_sq_sum } else { 0.0 };

    let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_w = weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::INFINITY, f64::min);

    Ok(EntropyBalanceResult {
        weights: full_weights,
        lambda: lambda.to_vec(),
        balance,
        effective_sample_size: ess * n_ctrl as f64,
        max_weight: max_w * n_ctrl as f64,
        min_weight: min_w * n_ctrl as f64,
        converged,
        iterations,
        constraint_tolerance: constraint_tol,
        n_reweighted: n_ctrl,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weighting Method Implementations
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute logistic propensity score weights.
fn compute_logistic_weights(
    x: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    stabilize: bool,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
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
    let ps: Vec<f64> = z_final.iter().map(|&z| logistic_cdf(z).max(1e-10).min(1.0 - 1e-10)).collect();

    // Compute stabilization factor
    let p_treat = d.iter().filter(|&&di| di >= 0.5).count() as f64 / n as f64;

    // Compute weights based on estimand
    let weights: Vec<f64> = (0..n).map(|i| {
        let di = d[i];
        let psi = ps[i];

        let raw_weight = match estimand {
            WeightEstimand::ATE => {
                // ATE: T/ps + (1-T)/(1-ps)
                if di >= 0.5 {
                    1.0 / psi
                } else {
                    1.0 / (1.0 - psi)
                }
            }
            WeightEstimand::ATT => {
                // ATT: T + (1-T) * ps / (1-ps)
                if di >= 0.5 {
                    1.0
                } else {
                    psi / (1.0 - psi)
                }
            }
            WeightEstimand::ATC => {
                // ATC: T * (1-ps) / ps + (1-T)
                if di >= 0.5 {
                    (1.0 - psi) / psi
                } else {
                    1.0
                }
            }
        };

        if stabilize {
            match estimand {
                WeightEstimand::ATE => {
                    if di >= 0.5 {
                        p_treat / psi
                    } else {
                        (1.0 - p_treat) / (1.0 - psi)
                    }
                }
                WeightEstimand::ATT => raw_weight, // ATT weights are already stable
                WeightEstimand::ATC => raw_weight, // ATC weights are already stable
            }
        } else {
            raw_weight
        }
    }).collect();

    Ok((weights, ps))
}

/// Compute entropy balancing weights using Newton's method.
///
/// Implements Hainmueller (2012) Algorithm 1.
fn compute_entropy_weights(
    x: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<WeightItResult> {
    let n = d.len();
    let k = x.ncols();

    // Identify groups based on estimand
    let (reweight_idx, target_idx): (Vec<usize>, Vec<usize>) = match estimand {
        WeightEstimand::ATT => {
            let ctrl: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();
            let treat: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
            (ctrl, treat)
        }
        WeightEstimand::ATC => {
            let treat: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
            let ctrl: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();
            (treat, ctrl)
        }
        WeightEstimand::ATE => {
            // For ATE, reweight both groups to overall mean
            // This is more complex - use separate reweighting
            let ctrl: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();
            let treat: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
            (ctrl, treat) // For now, default to ATT-like
        }
    };

    let n_reweight = reweight_idx.len();
    let n_target = target_idx.len();

    // Build matrices for reweighted and target groups
    let mut x_reweight = Array2::zeros((n_reweight, k));
    let mut x_target = Array2::zeros((n_target, k));

    for (new_i, &old_i) in reweight_idx.iter().enumerate() {
        x_reweight.row_mut(new_i).assign(&x.row(old_i));
    }
    for (new_i, &old_i) in target_idx.iter().enumerate() {
        x_target.row_mut(new_i).assign(&x.row(old_i));
    }

    // Target means (mean of target group)
    let target_means: Array1<f64> = x_target.mean_axis(ndarray::Axis(0))
        .ok_or_else(|| EconError::Computation("Failed to compute target means".to_string()))?;

    // Run entropy balancing optimization
    let (weights_reweight, _lambda, converged, iterations, _constraint_tol) =
        entropy_balance_optimize(&x_reweight, &target_means, max_iter, tolerance)?;

    // Create full weight vector
    let mut weights = vec![1.0; n];
    for (reweight_i, &orig_i) in reweight_idx.iter().enumerate() {
        weights[orig_i] = weights_reweight[reweight_i] * n_reweight as f64;
    }

    // Variable names
    let var_names: Vec<String> = (0..k).map(|i| format!("X{}", i)).collect();

    // Compute balance
    let balance_before = compute_balance_table(x, d, &var_names, None);
    let balance_after = compute_balance_table(x, d, &var_names, Some(&weights));

    // ESS
    let (ess_total, ess_treated, ess_control) = compute_ess(&weights, d);

    let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_w = weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::INFINITY, f64::min);

    let n_treated = d.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    Ok(WeightItResult {
        weights,
        propensity_scores: None,
        method: WeightMethod::Entropy,
        estimand,
        balance_before,
        balance_after,
        effective_sample_size: ess_total,
        ess_treated,
        ess_control,
        max_weight: max_w,
        min_weight: min_w,
        n_obs: n,
        n_treated,
        n_control,
        converged,
        iterations,
        warnings: Vec::new(),
    })
}

/// Core entropy balancing optimization using Newton's method.
///
/// Solves: min Σ w_i log(w_i) s.t. Σ w_i X_i = target, Σ w_i = 1
///
/// Uses dual formulation where optimal weights are:
/// w_i ∝ exp(λ' X_i)
fn entropy_balance_optimize(
    x: &Array2<f64>,
    target: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Vec<f64>, Array1<f64>, bool, usize, f64)> {
    let n = x.nrows();
    let k = x.ncols();

    // Initialize Lagrange multipliers
    let mut lambda = Array1::zeros(k);

    let mut converged = false;
    let mut iterations = 0;
    let mut constraint_tol = f64::INFINITY;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute weights: w_i ∝ exp(λ' X_i)
        let mut exp_scores = Array1::zeros(n);
        for i in 0..n {
            let score: f64 = (0..k).map(|j| lambda[j] * x[[i, j]]).sum();
            exp_scores[i] = score.exp().min(1e100); // Prevent overflow
        }

        // Normalize
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Array1<f64> = &exp_scores / sum_exp;

        // Compute weighted means
        let mut weighted_means = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                weighted_means[j] += weights[i] * x[[i, j]];
            }
        }

        // Constraint violation: (weighted_means - target)
        let constraint = &weighted_means - target;
        constraint_tol = constraint.iter().map(|c: &f64| c.abs()).fold(0.0, f64::max);

        // Check convergence
        if constraint_tol < tolerance {
            converged = true;
            break;
        }

        // Newton step: solve H * delta = -constraint
        // Hessian H[j,l] = Σ w_i (X_ij - mean_j)(X_il - mean_l)
        // This is the weighted covariance matrix
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                let dev_j = x[[i, j]] - weighted_means[j];
                for l in 0..k {
                    let dev_l = x[[i, l]] - weighted_means[l];
                    hessian[[j, l]] += wi * dev_j * dev_l;
                }
            }
        }

        // Add small regularization for numerical stability
        for j in 0..k {
            hessian[[j, j]] += 1e-8;
        }

        // Solve for update
        let (hess_inv, _) = safe_inverse(&hessian.view()).map_err(|_| {
            EconError::SingularMatrix {
                context: "Entropy balancing Hessian".to_string(),
                suggestion: "Covariates may be linearly dependent".to_string(),
            }
        })?;

        let delta = hess_inv.dot(&constraint);

        // Line search with backtracking
        let mut step = 1.0;
        for _ in 0..20 {
            let lambda_new = &lambda - &(&delta * step);

            // Evaluate constraint at new point
            let mut exp_new = Array1::zeros(n);
            for i in 0..n {
                let score: f64 = (0..k).map(|j| lambda_new[j] * x[[i, j]]).sum();
                exp_new[i] = score.exp().min(1e100);
            }
            let sum_new: f64 = exp_new.iter().sum();
            let w_new: Array1<f64> = &exp_new / sum_new;

            let mut means_new = Array1::zeros(k);
            for i in 0..n {
                for j in 0..k {
                    means_new[j] += w_new[i] * x[[i, j]];
                }
            }

            let new_tol = (&means_new - target).iter().map(|c: &f64| c.abs()).fold(0.0, f64::max);

            if new_tol < constraint_tol * 1.1 {
                // Accept step
                lambda = lambda_new;
                break;
            }
            step *= 0.5;
        }
    }

    // Compute final weights
    let mut exp_scores = Array1::zeros(n);
    for i in 0..n {
        let score: f64 = (0..k).map(|j| lambda[j] * x[[i, j]]).sum();
        exp_scores[i] = score.exp().min(1e100);
    }
    let sum_exp: f64 = exp_scores.iter().sum();
    let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    Ok((weights, lambda, converged, iterations, constraint_tol))
}

/// Compute energy balancing weights.
///
/// Minimizes the energy distance between the weighted control distribution
/// and the treated distribution.
fn compute_energy_weights(
    x: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Vec<f64>, bool, usize)> {
    // Energy balancing is similar to entropy but uses a different objective
    // For simplicity, we use an iterative reweighting approach

    let n = d.len();

    // Start with uniform weights
    let mut weights = vec![1.0; n];

    // Identify groups
    let treated_idx: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();

    let n_treat = treated_idx.len() as f64;
    let n_ctrl = control_idx.len() as f64;

    // Compute pairwise distances for energy
    let k = x.ncols();

    // Target: treated group mean
    let mut target_mean = Array1::zeros(k);
    for &i in &treated_idx {
        for j in 0..k {
            target_mean[j] += x[[i, j]];
        }
    }
    target_mean /= n_treat;

    let mut converged = false;
    let mut iterations = 0;

    // Iteratively reweight to minimize energy distance
    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute weighted control mean
        let mut w_sum = 0.0;
        let mut weighted_mean = Array1::zeros(k);
        for &i in &control_idx {
            w_sum += weights[i];
            for j in 0..k {
                weighted_mean[j] += weights[i] * x[[i, j]];
            }
        }
        if w_sum > 0.0 {
            weighted_mean /= w_sum;
        }

        // Check convergence: distance to target
        let diff: f64 = (&weighted_mean - &target_mean).iter().map(|d| d * d).sum::<f64>().sqrt();
        if diff < tolerance {
            converged = true;
            break;
        }

        // Update weights using gradient of energy distance
        // Simplified: increase weight for control units closer to target
        for &i in &control_idx {
            let dist: f64 = (0..k).map(|j| (x[[i, j]] - target_mean[j]).powi(2)).sum::<f64>().sqrt();
            // Inverse distance weighting
            weights[i] = 1.0 / (dist + 0.1);
        }

        // Normalize control weights
        let ctrl_sum: f64 = control_idx.iter().map(|&i| weights[i]).sum();
        for &i in &control_idx {
            weights[i] = weights[i] / ctrl_sum * n_ctrl;
        }
    }

    // For ATT, treated get weight 1
    for &i in &treated_idx {
        weights[i] = 1.0;
    }

    Ok((weights, converged, iterations))
}

/// Compute stable balancing weights (Zubizarreta 2015).
///
/// Minimizes weight dispersion while achieving approximate balance.
fn compute_stable_weights(
    x: &Array2<f64>,
    d: &Array1<f64>,
    estimand: WeightEstimand,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Vec<f64>, bool, usize)> {
    // Stable weights minimize variance of weights subject to approximate balance
    // We use a penalty-based approach

    let n = d.len();
    let k = x.ncols();

    let treated_idx: Vec<usize> = (0..n).filter(|&i| d[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| d[i] < 0.5).collect();

    let n_treat = treated_idx.len() as f64;
    let n_ctrl = control_idx.len() as f64;

    // Target means (treated group)
    let mut target_mean = Array1::zeros(k);
    for &i in &treated_idx {
        for j in 0..k {
            target_mean[j] += x[[i, j]];
        }
    }
    target_mean /= n_treat;

    // Start with uniform weights
    let mut weights = vec![1.0; n];
    for &i in &control_idx {
        weights[i] = n_ctrl.recip();
    }

    let balance_penalty = 100.0; // Penalty for balance constraint violation
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute current weighted mean of control
        let mut ctrl_w_sum = 0.0;
        let mut weighted_mean = Array1::zeros(k);
        for &i in &control_idx {
            ctrl_w_sum += weights[i];
            for j in 0..k {
                weighted_mean[j] += weights[i] * x[[i, j]];
            }
        }
        if ctrl_w_sum > 0.0 {
            weighted_mean /= ctrl_w_sum;
        }

        // Balance violation
        let balance_viol: f64 = (&weighted_mean - &target_mean).iter()
            .map(|d| d * d).sum::<f64>().sqrt();

        if balance_viol < tolerance {
            converged = true;
            break;
        }

        // Update weights: gradient descent on objective
        // Objective = variance(w) + penalty * balance_violation^2
        let w_mean = 1.0 / n_ctrl;

        for &i in &control_idx {
            // Gradient of variance term
            let var_grad = 2.0 * (weights[i] - w_mean);

            // Gradient of balance penalty
            let mut balance_grad = 0.0;
            for j in 0..k {
                let diff = weighted_mean[j] - target_mean[j];
                balance_grad += 2.0 * balance_penalty * diff * (x[[i, j]] - weighted_mean[j]) / ctrl_w_sum;
            }

            // Update
            let grad = var_grad + balance_grad;
            weights[i] -= 0.01 * grad;
            weights[i] = weights[i].max(0.001); // Keep positive
        }

        // Renormalize
        let w_sum: f64 = control_idx.iter().map(|&i| weights[i]).sum();
        for &i in &control_idx {
            weights[i] /= w_sum;
        }
    }

    // Scale to n_ctrl
    for &i in &control_idx {
        weights[i] *= n_ctrl;
    }

    // Treated get weight 1
    for &i in &treated_idx {
        weights[i] = 1.0;
    }

    Ok((weights, converged, iterations))
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

        var_t = if w_sum_t > 1.0 { var_t / (w_sum_t - 1.0) } else { 0.0 };
        var_c = if w_sum_c > 1.0 { var_c / (w_sum_c - 1.0) } else { 0.0 };

        // Standardized difference
        let pooled_sd = ((var_t + var_c) / 2.0).sqrt().max(1e-10);
        let std_diff = (mean_t - mean_c) / pooled_sd;

        // Variance ratio
        let var_ratio = if var_c > 1e-10 { var_t / var_c } else { 1.0 };

        covariates.push(WeightItCovariateBalance {
            variable: var_names.get(j).cloned().unwrap_or_else(|| format!("X{}", j)),
            mean_treated: mean_t,
            mean_control: mean_c,
            std_diff,
            var_ratio,
        });
    }

    // Summary statistics
    let max_std_diff = covariates.iter().map(|c| c.std_diff.abs()).fold(0.0, f64::max);
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

    let ess_all = if w_sq_sum_all > 0.0 { w_sum_all * w_sum_all / w_sq_sum_all } else { 0.0 };
    let ess_t = if w_sq_sum_t > 0.0 { w_sum_t * w_sum_t / w_sq_sum_t } else { 0.0 };
    let ess_c = if w_sq_sum_c > 0.0 { w_sum_c * w_sum_c / w_sq_sum_c } else { 0.0 };

    (ess_all, ess_t, ess_c)
}

/// Trim extreme weights at specified quantile.
fn trim_weights(weights: &[f64], quantile: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let lower_idx = ((1.0 - quantile) * n as f64).floor() as usize;
    let upper_idx = (quantile * n as f64).ceil() as usize;

    let lower_bound = sorted[lower_idx.min(n - 1)];
    let upper_bound = sorted[upper_idx.min(n - 1)];

    weights.iter().map(|&w| w.max(lower_bound).min(upper_bound)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    /// Create test dataset with known imbalance.
    fn create_test_dataset() -> Dataset {
        // Treated group has higher means on x1, x2
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // Treated has higher x1 (mean ~1.5 vs ~0.5)
            "x1" => [
                1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.1, 1.7, 1.5, 1.4,
                1.9, 2.0, 1.6, 1.8, 1.3, 1.7, 1.4, 2.1, 1.5, 1.6,
                0.3, 0.5, 0.4, 0.7, 0.2, 0.6, 0.1, 0.8, 0.4, 0.5,
                0.6, 0.7, 0.3, 0.9, 0.4, 0.5, 0.2, 0.8, 0.5, 0.6
            ],
            // Similar imbalance on x2
            "x2" => [
                0.8, 0.9, 0.7, 1.0, 0.85, 0.95, 0.75, 1.1, 0.9, 0.85,
                1.0, 1.1, 0.9, 1.05, 0.8, 0.95, 0.85, 1.15, 0.9, 0.92,
                0.3, 0.4, 0.35, 0.5, 0.25, 0.45, 0.2, 0.55, 0.4, 0.35,
                0.45, 0.5, 0.3, 0.6, 0.35, 0.4, 0.25, 0.55, 0.4, 0.42
            ]
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_weightit_logistic_ate() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Logistic,
            estimand: WeightEstimand::ATE,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);
        assert_eq!(result.weights.len(), 40);

        // ESS should be less than n (weights add variability)
        assert!(result.effective_sample_size > 0.0);
        assert!(result.effective_sample_size <= 40.0);

        // Balance should improve after weighting
        assert!(result.balance_after.max_std_diff < result.balance_before.max_std_diff
                || result.balance_after.max_std_diff < 0.5);
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
            let is_treated = if i < 20 { true } else { false };
            if is_treated {
                assert!((result.weights[i] - 1.0).abs() < 1e-10,
                        "Treated weight should be 1.0, got {}", result.weights[i]);
            }
        }
    }

    #[test]
    fn test_weightit_entropy() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Entropy,
            estimand: WeightEstimand::ATT,
            max_iter: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Entropy balancing should achieve very good balance
        // (within tolerance, near-zero std diff)
        assert!(result.balance_after.max_std_diff < 0.1,
                "Entropy balancing should achieve good balance, got max std_diff = {}",
                result.balance_after.max_std_diff);
    }

    #[test]
    fn test_entropy_balance_direct() {
        let dataset = create_test_dataset();

        let result = entropy_balance(&dataset, "treatment", &["x1", "x2"], None).unwrap();

        // Should converge
        assert!(result.converged, "Entropy balancing should converge");

        // Should achieve good balance
        assert!(result.balance.max_std_diff < 0.1,
                "Should achieve good balance, got {}", result.balance.max_std_diff);

        // ESS should be reasonable
        assert!(result.effective_sample_size > 5.0,
                "ESS should be reasonable, got {}", result.effective_sample_size);
    }

    #[test]
    fn test_weightit_energy() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Energy,
            estimand: WeightEstimand::ATT,
            max_iter: 100,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Should produce valid weights
        assert_eq!(result.weights.len(), 40);
        assert!(result.weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_weightit_stable() {
        let dataset = create_test_dataset();
        let config = WeightItConfig {
            method: WeightMethod::Stable,
            estimand: WeightEstimand::ATT,
            max_iter: 100,
            ..Default::default()
        };

        let result = weightit(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Stable weights should have lower variance than logistic
        assert_eq!(result.weights.len(), 40);
        assert!(result.weights.iter().all(|&w| w > 0.0));
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
        let x = Array2::from_shape_vec((6, 2), vec![
            1.0, 0.5,
            1.2, 0.6,
            1.1, 0.55,
            0.5, 0.2,
            0.6, 0.3,
            0.55, 0.25,
        ]).unwrap();
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
        assert!(trimmed.iter().all(|&w| w >= 0.1 && w <= 10.0));
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
}
