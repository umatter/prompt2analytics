//! Stable Balancing Weights (SBW) module.
//!
//! Implements the stable balancing weights method of Zubizarreta (2015) for
//! causal inference. SBW directly optimizes for covariate balance rather than
//! modeling the propensity score, finding weights that minimize variance while
//! achieving exact or approximate balance on covariate moments.
//!
//! # Mathematical Background
//!
//! ## Quadratic Programming Formulation
//!
//! For ATT estimation with control weights w_i, we solve:
//!
//! ```text
//! minimize:    (1/2) * w'Hw + c'w  (variance of weights)
//! subject to:  A * w = b           (balance constraints)
//!              w_i >= l            (lower bound on weights)
//! ```
//!
//! Where:
//! - H = I (identity matrix) for variance minimization
//! - c = 0 (no linear term for pure variance minimization)
//! - A = covariate matrix with normalization constraint
//! - b = target means (treated group means for ATT)
//! - l = minimum weight (default 0 for non-negativity)
//!
//! ## Exact Balance: Dual Semismooth Newton
//!
//! With H = I the exact problem (balance_tol = 0) is the Euclidean projection of the
//! origin onto {w : A w = b, w >= l}. Writing a_i = [x_i, 1] / n for the columns of A
//! (n reweighted units, k covariates; b = [target; 1], so the weights sum to n), its dual
//! has only k + 1 variables mu:
//!
//! ```text
//! w_i(mu) = max(l, a_i . mu)
//! g(mu)   = b . mu + sum_i h(a_i . mu),   h(z) = -z^2/2 (z >= l),   l^2/2 - l z (z < l)
//! grad g  = r(mu) = b - sum_i w_i(mu) a_i          (the balance residual)
//! ```
//!
//! g is concave and piecewise quadratic, and is maximised by a semismooth Newton method:
//!
//! ```text
//! mu_0 = (A A')^-1 b                   Schur complement: w = A'(A A')^-1 b, the unbounded
//!                                      optimum (the old dense KKT solution when no bound binds)
//! S    = {i : a_i . mu > l}            units off the bound
//! M    = sum_{i in S} a_i a_i'         (k+1) x (k+1); pseudo-inverse if singular
//! mu  <- mu + t M^-1 r(mu)             t = 1 when the step keeps S (exact on that piece),
//!                                      otherwise Armijo backtracking on g
//! ```
//!
//! Each step costs one (k+1) x (k+1) solve and O(n k^2) work; nothing of size n x n is
//! formed. The bound w >= l holds exactly and exact balance is reached to rounding. The
//! iteration stops when ||r||_inf <= 1e-14 (1 + ||b||_inf), or <= tolerance (1 + ||b||_inf)
//! after a full step that kept S. If no feasible weights exist the dual is unbounded: no
//! ascent direction, a failed line search, diverging multipliers or max_iter steps end the
//! solve with `EconError::InvalidSpecification` suggesting approximate balance. With
//! min_weight >= 0 a target mean outside the reweighted group's range is refused up front.
//!
//! `lambda` in the result is -mu, the multiplier of the KKT system
//!
//! ```text
//! [I    A'][w]   [0]
//! [A    0 ][λ] = [b]
//! ```
//!
//! which the dual solution satisfies on the free units.
//!
//! ## Approximate Balance
//!
//! When exact balance is infeasible, we relax to:
//!
//! ```text
//! minimize:    (1/2) * w'Hw + c'w + ρ * ||A*w - b||²
//! subject to:  w_i >= l
//! ```
//!
//! Where ρ is a penalty parameter for balance constraint violation.
//!
//! # References
//!
//! - Zubizarreta, J.R. (2015). "Stable Weights that Balance Covariates for
//!   Estimation with Incomplete Outcome Data". *Journal of the American
//!   Statistical Association*, 110(511), 910-922.
//!   DOI: 10.1080/01621459.2015.1023805
//!
//! - Zubizarreta, J.R., Cerdeiro, D.A., & Kelz, R.R. (2020). sbw: Stable
//!   Balancing Weights for Causal Inference and Estimation with Incomplete
//!   Outcome Data. R package. https://cran.r-project.org/package=sbw
//!
//! - Chan, K.C.G., Yam, S.C.P., & Zhang, Z. (2016). "Globally Efficient
//!   Non-parametric Inference of Average Treatment Effects by Empirical
//!   Balancing Calibration Weighting". *JRSS-B*, 78(3), 673-700.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx};
use crate::resource;

// ═══════════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════════

/// Target estimand for stable balancing weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SBWEstimand {
    /// Average Treatment Effect on the Treated.
    /// Control units are reweighted to match treated distribution.
    #[default]
    ATT,
    /// Average Treatment Effect (population).
    /// Both groups are reweighted to match overall distribution.
    ATE,
    /// Average Treatment Effect on the Control.
    /// Treated units are reweighted to match control distribution.
    ATC,
}

impl fmt::Display for SBWEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SBWEstimand::ATT => write!(f, "ATT (Average Treatment Effect on Treated)"),
            SBWEstimand::ATE => write!(f, "ATE (Average Treatment Effect)"),
            SBWEstimand::ATC => write!(f, "ATC (Average Treatment Effect on Control)"),
        }
    }
}

/// Configuration for stable balancing weights estimation.
#[derive(Debug, Clone)]
pub struct SBWConfig {
    /// Target estimand (ATT, ATE, ATC)
    pub estimand: SBWEstimand,
    /// Tolerance for approximate balance (0 = exact balance)
    /// Constraints: |Σw_i X_ij / Σw_i - target_j| <= balance_tol
    pub balance_tol: f64,
    /// Minimum weight allowed (default 0 for non-negativity)
    pub min_weight: f64,
    /// Normalize weights to sum to n (true) or 1 (false)
    pub normalize_to_n: bool,
    /// Maximum iterations for approximate balance solver
    pub max_iter: usize,
    /// Convergence tolerance for optimization
    pub tolerance: f64,
    /// Penalty parameter for approximate balance (higher = stricter balance)
    pub balance_penalty: f64,
}

impl Default for SBWConfig {
    fn default() -> Self {
        Self {
            estimand: SBWEstimand::ATT,
            balance_tol: 0.0, // Exact balance by default
            min_weight: 0.0,  // Non-negativity
            normalize_to_n: true,
            max_iter: 1000,
            tolerance: 1e-8,
            balance_penalty: 1000.0,
        }
    }
}

/// Balance statistics showing covariate balance before and after weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceStats {
    /// Covariate names
    pub covariate_names: Vec<String>,
    /// Standardized mean differences for each covariate
    pub std_diff: Vec<f64>,
    /// Maximum absolute standardized difference
    pub max_std_diff: f64,
    /// Mean absolute standardized difference
    pub mean_std_diff: f64,
    /// Variance ratios (treated/control) for each covariate
    pub var_ratio: Vec<f64>,
    /// Weighted mean in target group for each covariate
    pub mean_target: Vec<f64>,
    /// Weighted mean in reweighted group for each covariate
    pub mean_reweighted: Vec<f64>,
}

impl fmt::Display for BalanceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Covariate", "Target", "Reweighted", "Std.Diff", "Var.Ratio"
        )?;
        writeln!(f, "{}", "-".repeat(68))?;

        for i in 0..self.covariate_names.len() {
            let balanced = if self.std_diff[i].abs() < 0.1 {
                " "
            } else {
                "*"
            };
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>11.4}{} {:>12.4}",
                self.covariate_names[i],
                self.mean_target[i],
                self.mean_reweighted[i],
                self.std_diff[i],
                balanced,
                self.var_ratio[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(68))?;
        writeln!(
            f,
            "Max |Std.Diff|: {:.4}   Mean |Std.Diff|: {:.4}",
            self.max_std_diff, self.mean_std_diff
        )?;

        Ok(())
    }
}

/// Result from stable balancing weights estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SBWResult {
    /// Optimal weights for each observation
    pub weights: Vec<f64>,
    /// Covariate balance before weighting
    pub balance_before: BalanceStats,
    /// Covariate balance after weighting
    pub balance_after: BalanceStats,
    /// Effective sample size: ESS = (Σw)² / Σw²
    pub effective_sample_size: f64,
    /// ESS for target group (treated for ATT)
    pub ess_target: f64,
    /// ESS for reweighted group (control for ATT)
    pub ess_reweighted: f64,
    /// Maximum weight among reweighted units
    pub max_weight: f64,
    /// Minimum weight among reweighted units
    pub min_weight: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final objective function value (variance of weights)
    pub objective_value: f64,
    /// Maximum balance constraint violation
    pub max_constraint_violation: f64,
    /// Target estimand used
    pub estimand: SBWEstimand,
    /// Number of observations
    pub n_obs: usize,
    /// Number of target units (treated for ATT)
    pub n_target: usize,
    /// Number of reweighted units (control for ATT)
    pub n_reweighted: usize,
    /// Lagrange multipliers from QP solution
    pub lambda: Vec<f64>,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
}

impl fmt::Display for SBWResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Stable Balancing Weights (SBW)")?;
        writeln!(f, "==============================")?;
        writeln!(f, "Estimand: {}", self.estimand)?;
        writeln!(f)?;

        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Total:      {}  (Target: {}, Reweighted: {})",
            self.n_obs, self.n_target, self.n_reweighted
        )?;
        writeln!(f)?;

        writeln!(f, "Optimization:")?;
        writeln!(
            f,
            "  Converged:  {} ({} iterations)",
            if self.converged { "Yes" } else { "No" },
            self.iterations
        )?;
        writeln!(
            f,
            "  Objective:  {:.6} (variance of weights)",
            self.objective_value
        )?;
        writeln!(f, "  Max violation: {:.6}", self.max_constraint_violation)?;
        writeln!(f)?;

        writeln!(f, "Weight Summary (reweighted group):")?;
        writeln!(
            f,
            "  Range:      [{:.4}, {:.4}]",
            self.min_weight, self.max_weight
        )?;
        writeln!(
            f,
            "  ESS:        {:.1} of {} ({:.1}%)",
            self.ess_reweighted,
            self.n_reweighted,
            100.0 * self.ess_reweighted / self.n_reweighted as f64
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
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute stable balancing weights for causal inference.
///
/// This function solves a quadratic programming problem to find weights that
/// minimize variance while achieving exact or approximate covariate balance.
///
/// # Arguments
///
/// * `treatment` - Treatment indicator array (1 = treated, 0 = control)
/// * `covariates` - Covariate matrix (n x k)
/// * `config` - Configuration options
///
/// # Returns
///
/// `SBWResult` containing optimal weights and balance diagnostics.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::sbw::{run_sbw, SBWConfig, SBWEstimand};
/// use ndarray::Array1;
///
/// let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
/// let covariates = ndarray::array![[1.0, 0.5], [1.2, 0.6], [0.5, 0.2], [0.4, 0.3], [0.6, 0.25]];
///
/// let config = SBWConfig {
///     estimand: SBWEstimand::ATT,
///     ..Default::default()
/// };
///
/// let result = run_sbw(&treatment.view(), &covariates.view(), config)?;
/// println!("ESS: {:.1}", result.effective_sample_size);
/// ```
///
/// # References
///
/// - Zubizarreta, J.R. (2015). "Stable Weights that Balance Covariates for
///   Estimation with Incomplete Outcome Data". *JASA*, 110(511), 910-922.
pub fn run_sbw(
    treatment: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    config: SBWConfig,
) -> EconResult<SBWResult> {
    let cov_names: Vec<String> = (0..covariates.ncols())
        .map(|i| format!("X{}", i + 1))
        .collect();
    run_sbw_inner(treatment, covariates, &cov_names, config)
}

/// Checks the configuration before any work.
fn validate_sbw_config(config: &SBWConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if !(config.balance_tol.is_finite() && config.balance_tol >= 0.0) {
        return invalid(format!(
            "balance_tol must be finite and >= 0 (0 = exact balance); got {}",
            config.balance_tol
        ));
    }
    if !config.min_weight.is_finite() {
        return invalid(format!(
            "min_weight must be finite; got {}",
            config.min_weight
        ));
    }
    if config.balance_tol == 0.0 && config.min_weight > 1.0 {
        return invalid(format!(
            "min_weight must be <= 1 for exact balance (the weights average 1 in the n scale); got {}",
            config.min_weight
        ));
    }
    if config.max_iter == 0 {
        return invalid("max_iter must be at least 1; got 0".to_string());
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return invalid(format!(
            "tolerance must be finite and > 0; got {}",
            config.tolerance
        ));
    }
    if config.balance_tol > 0.0
        && !(config.balance_penalty.is_finite() && config.balance_penalty > 0.0)
    {
        return invalid(format!(
            "balance_penalty must be finite and > 0 for approximate balance; got {}",
            config.balance_penalty
        ));
    }
    Ok(())
}

/// [`run_sbw`] with the covariate names used in balance tables and error messages.
fn run_sbw_inner(
    treatment: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    cov_names: &[String],
    config: SBWConfig,
) -> EconResult<SBWResult> {
    let n = treatment.len();
    let mut warnings = Vec::new();

    // Validate inputs
    validate_sbw_config(&config)?;
    if n != covariates.nrows() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment length ({}) must match covariate rows ({})",
                n,
                covariates.nrows()
            ),
        });
    }
    if treatment.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "treatment values must be finite (no NaN or infinite values)".to_string(),
        });
    }
    for (col, name) in covariates.columns().into_iter().zip(cov_names) {
        if col.iter().any(|v| !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "covariate {name} contains NaN or infinite values; covariates must be finite"
                ),
            });
        }
    }

    // Identify target and reweight groups based on estimand
    let (target_idx, reweight_idx): (Vec<usize>, Vec<usize>) = match config.estimand {
        SBWEstimand::ATT => {
            let treated: Vec<usize> = (0..n).filter(|&i| treatment[i] >= 0.5).collect();
            let control: Vec<usize> = (0..n).filter(|&i| treatment[i] < 0.5).collect();
            (treated, control)
        }
        SBWEstimand::ATC => {
            let control: Vec<usize> = (0..n).filter(|&i| treatment[i] < 0.5).collect();
            let treated: Vec<usize> = (0..n).filter(|&i| treatment[i] >= 0.5).collect();
            (control, treated)
        }
        SBWEstimand::ATE => {
            // For ATE, we reweight both groups to the overall distribution
            // For now, implement as ATT (reweight control to treated)
            // Full ATE would require reweighting both groups
            let treated: Vec<usize> = (0..n).filter(|&i| treatment[i] >= 0.5).collect();
            let control: Vec<usize> = (0..n).filter(|&i| treatment[i] < 0.5).collect();
            (treated, control)
        }
    };

    let n_target = target_idx.len();
    let n_reweight = reweight_idx.len();

    if n_target == 0 || n_reweight == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Both groups must have observations. Found {} target, {} to reweight.",
                n_target, n_reweight
            ),
        });
    }

    // Extract covariate matrices for each group
    let x_target = extract_rows(covariates, &target_idx);
    let x_reweight = extract_rows(covariates, &reweight_idx);

    // Compute target means (means of target group)
    let target_means = x_target
        .mean_axis(Axis(0))
        .ok_or_else(|| EconError::Computation("Failed to compute target means".to_string()))?;

    // Exact balance with non-negative weights reweights to a convex combination of the
    // reweighted units, so every target mean must lie within that group's range.
    if config.balance_tol == 0.0 && config.min_weight >= 0.0 {
        for (j, name) in cov_names.iter().enumerate() {
            let col = x_reweight.column(j);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let t = target_means[j];
            if t < lo || t > hi {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Exact balance is infeasible: the target mean of covariate {name} ({t:.4}) lies outside the reweighted group's range [{lo:.4}, {hi:.4}], which weights >= {} cannot reach. Use approximate balance (balance_tol > 0) or another estimand.",
                        config.min_weight
                    ),
                });
            }
        }
    }

    // Compute balance before weighting
    let balance_before =
        compute_balance_stats(&x_target.view(), &x_reweight.view(), cov_names, None);

    // Solve the quadratic programming problem
    let (weights, lambda, converged, iterations, objective, max_violation) =
        if config.balance_tol == 0.0 {
            // Exact balance: dual semismooth Newton in the constraint dimension
            solve_sbw_exact(&x_reweight.view(), &target_means.view(), &config)?
        } else {
            // Approximate balance: use penalized QP
            solve_sbw_approximate(&x_reweight.view(), &target_means.view(), &config)?
        };

    // Scale weights for normalization
    let mut final_weights = weights.clone();
    if config.normalize_to_n {
        let w_sum: f64 = final_weights.iter().sum();
        if w_sum > 0.0 {
            for w in &mut final_weights {
                *w *= n_reweight as f64 / w_sum;
            }
        }
    }

    // Create full weight vector
    let mut full_weights = vec![0.0; n];
    for &i in &target_idx {
        full_weights[i] = 1.0;
    }
    for (j, &i) in reweight_idx.iter().enumerate() {
        full_weights[i] = final_weights[j];
    }

    // Compute balance after weighting
    let balance_after = compute_balance_stats(
        &x_target.view(),
        &x_reweight.view(),
        cov_names,
        Some(&final_weights),
    );

    // Compute effective sample sizes
    let (ess_total, ess_target, ess_reweight) = compute_ess(&full_weights, treatment);

    // Weight statistics
    let max_weight = final_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_weight = final_weights
        .iter()
        .filter(|&&w| w > 0.0)
        .cloned()
        .fold(f64::INFINITY, f64::min);

    // Check for warnings
    if max_weight / min_weight.max(1e-10) > 100.0 {
        warnings.push(format!(
            "High weight variability (max/min = {:.1}). Consider approximate balance.",
            max_weight / min_weight.max(1e-10)
        ));
    }

    if ess_reweight < n_reweight as f64 * 0.3 {
        warnings.push(format!(
            "Low effective sample size ({:.1} of {}, {:.0}%). Weights are highly variable.",
            ess_reweight,
            n_reweight,
            100.0 * ess_reweight / n_reweight as f64
        ));
    }

    if !converged {
        warnings.push(format!(
            "Optimization did not converge after {} iterations. Results may be unreliable.",
            iterations
        ));
    }

    if max_violation > 0.01 {
        warnings.push(format!(
            "Balance constraints not fully satisfied (max violation = {:.4}).",
            max_violation
        ));
    }

    Ok(SBWResult {
        weights: full_weights,
        balance_before,
        balance_after,
        effective_sample_size: ess_total,
        ess_target,
        ess_reweighted: ess_reweight,
        max_weight,
        min_weight,
        converged,
        iterations,
        objective_value: objective,
        max_constraint_violation: max_violation,
        estimand: config.estimand,
        n_obs: n,
        n_target,
        n_reweighted: n_reweight,
        lambda,
        warnings,
    })
}

/// Run SBW on a dataset with column names.
///
/// # Arguments
///
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration options
pub fn sbw(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: SBWConfig,
) -> EconResult<SBWResult> {
    // Extract treatment variable
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Build covariate matrix (without intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let covariates = design.data;

    // Run SBW with the real column names
    let cov_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();
    run_sbw_inner(&treatment.view(), &covariates.view(), &cov_names, config)
}

// ═══════════════════════════════════════════════════════════════════════════════
// QP Solvers
// ═══════════════════════════════════════════════════════════════════════════════

/// Residual below which the dual iteration stops at once (relative to `1 + ||b||_inf`).
const SBW_EXACT_RESIDUAL: f64 = 1e-14;
/// Multipliers growing beyond this factor of the first step mean the dual is unbounded.
const SBW_DUAL_DIVERGENCE: f64 = 1e10;
/// Armijo sufficient-increase constant and smallest step of the backtracking line search.
const SBW_ARMIJO_C: f64 = 1e-4;
const SBW_MIN_STEP: f64 = 1e-12;
/// Rounding allowance of a dual-objective comparison (relative to `1 + |g|`).
const SBW_DUAL_ROUNDING: f64 = 1e-12;

/// The (k+1)-dimensional dual of the exact SBW problem, evaluated row by row over the
/// reweighted group's covariates `x` without materialising the constraint matrix.
///
/// ```text
/// a_i = [x_i, 1] / n,   b = [target; 1],   z_i(mu) = a_i . mu,   w_i(mu) = max(l, z_i)
/// g(mu) = b . mu + sum_i h(z_i),   h(z) = -z^2 / 2 (z >= l),   l^2 / 2 - l z (z < l)
/// grad g(mu) = r(mu) = b - sum_i w_i(mu) a_i
/// ```
struct SbwDual<'a> {
    x: ArrayView2<'a, f64>,
    b: Array1<f64>,
    lower: f64,
    n_f: f64,
}

impl SbwDual<'_> {
    fn z(&self, i: usize, mu: &Array1<f64>) -> f64 {
        let k = self.x.ncols();
        let mut s = mu[k];
        for j in 0..k {
            s += self.x[[i, j]] * mu[j];
        }
        s / self.n_f
    }

    /// Residual r(mu) (compensated sums, so its rounding does not grow with n) and the set
    /// S = {i : z_i > l} of units off the bound.
    fn residual(&self, mu: &Array1<f64>) -> (Array1<f64>, Vec<bool>) {
        let (n, k) = self.x.dim();
        let m = k + 1;
        let mut sum = vec![0.0; m];
        let mut comp = vec![0.0; m];
        let mut free = vec![false; n];
        let mut add = |p: usize, v: f64| {
            let t = sum[p] + v;
            if sum[p].abs() >= v.abs() {
                comp[p] += (sum[p] - t) + v;
            } else {
                comp[p] += (v - t) + sum[p];
            }
            sum[p] = t;
        };
        for (i, is_free) in free.iter_mut().enumerate() {
            let z = self.z(i, mu);
            let w = if z > self.lower {
                *is_free = true;
                z
            } else {
                self.lower
            };
            for j in 0..k {
                add(j, w * self.x[[i, j]] / self.n_f);
            }
            add(k, w / self.n_f);
        }
        let r = Array1::from_shape_fn(m, |p| self.b[p] - (sum[p] + comp[p]));
        (r, free)
    }

    /// The set S at `mu` only.
    fn free_set(&self, mu: &Array1<f64>) -> Vec<bool> {
        (0..self.x.nrows())
            .map(|i| self.z(i, mu) > self.lower)
            .collect()
    }

    /// Dual objective g(mu).
    fn value(&self, mu: &Array1<f64>) -> f64 {
        let l = self.lower;
        let mut g = self.b.dot(mu);
        for i in 0..self.x.nrows() {
            let z = self.z(i, mu);
            g += if z >= l {
                -0.5 * z * z
            } else {
                0.5 * l * l - l * z
            };
        }
        g
    }

    /// M = sum over the selected units of a_i a_i' ((k+1) x (k+1)); all units when `free` is None.
    fn gram(&self, free: Option<&[bool]>) -> Array2<f64> {
        let (n, k) = self.x.dim();
        let m = k + 1;
        let mut g = Array2::<f64>::zeros((m, m));
        let mut row = vec![0.0; m];
        for i in 0..n {
            if free.is_some_and(|f| !f[i]) {
                continue;
            }
            for j in 0..k {
                row[j] = self.x[[i, j]] / self.n_f;
            }
            row[k] = 1.0 / self.n_f;
            for p in 0..m {
                for q in 0..=p {
                    g[[p, q]] += row[p] * row[q];
                }
            }
        }
        for p in 0..m {
            for q in 0..p {
                g[[q, p]] = g[[p, q]];
            }
        }
        g
    }
}

fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// Solve SBW with exact balance constraints through the problem's dual.
///
/// The exact problem in the n scale (weights sum to n) is
///
/// ```text
/// minimise (1/2) ||w||^2   subject to   sum_i w_i [x_i, 1] / n = [target; 1],   w_i >= l
/// ```
///
/// i.e. the Euclidean projection of the origin onto {w : A w = b, w >= l}. Its dual has
/// only k+1 variables mu, with w_i = max(l, a_i . mu) (see [`SbwDual`]); it is solved by a
/// semismooth Newton method:
///
/// ```text
/// mu_0     = (A A')^-1 b                      (Schur complement: w = A'(AA')^-1 b, no bound)
/// repeat   S = {i : a_i . mu > l},  M = sum_{i in S} a_i a_i',  delta = M^-1 r(mu)
///          mu <- mu + t delta                 (t = 1, or Armijo backtracking on g when S changes)
/// until    ||r||_inf <= 1e-14 (1 + ||b||_inf), or <= tolerance (1 + ||b||_inf) after a
///          full step that kept S
/// ```
///
/// Each step is one (k+1) x (k+1) solve and O(n k^2) work; memory is O(n + k^2). When no
/// bound binds, the first step is the old dense-KKT solution. The bound w >= l holds
/// exactly. A dual that does not converge (no ascent direction, a failed line search,
/// diverging multipliers or `max_iter` steps) means no feasible weights exist, and is
/// `EconError::InvalidSpecification` suggesting approximate balance.
///
/// Returns weights summing to 1, `lambda = -mu` (the sign convention of the KKT system
/// `[I A'; A 0][w; lambda] = [0; b]`), convergence, the number of steps (mu_0 counts as
/// one), the objective and the maximum balance violation.
fn solve_sbw_exact(
    x: &ArrayView2<f64>,
    target_means: &ArrayView1<f64>,
    config: &SBWConfig,
) -> EconResult<(Vec<f64>, Vec<f64>, bool, usize, f64, f64)> {
    let n = x.nrows();
    let k = x.ncols();
    let l = config.min_weight;

    let mut b = Array1::<f64>::zeros(k + 1);
    for j in 0..k {
        b[j] = target_means[j];
    }
    b[k] = 1.0; // weights sum to n
    let dual = SbwDual {
        x: x.view(),
        b,
        lower: l,
        n_f: n as f64,
    };
    let b_scale = 1.0 + inf_norm(&dual.b);

    let infeasible = || EconError::InvalidSpecification {
        message: format!(
            "Exact balance could not be reached: no weights >= {l} on the {n} reweighted units reproduce the target covariate means (the targets may lie outside the reweighted group's range, or covariates may be collinear). Use approximate balance (balance_tol > 0)."
        ),
    };

    // First step: the Schur-complement closed form of the unbounded problem.
    let (m_all_inv, _) =
        safe_inverse(&dual.gram(None).view()).map_err(|e| EconError::SingularMatrix {
            context: "SBW balance constraints".to_string(),
            suggestion: format!(
                "Balance constraints may be infeasible or redundant. Error: {:?}. \
                 Try using approximate balance (balance_tol > 0).",
                e
            ),
        })?;
    let mut mu = m_all_inv.dot(&dual.b);
    let mu0_scale = 1.0 + inf_norm(&mu);
    let mut iterations = 1;
    let mut full_step_kept_set = false;

    loop {
        resource::checkpoint()?;

        let (r, free) = dual.residual(&mu);
        let r_inf = inf_norm(&r);
        let close = r_inf <= config.tolerance * b_scale;
        if r_inf <= SBW_EXACT_RESIDUAL * b_scale || (close && full_step_kept_set) {
            break;
        }
        if iterations >= config.max_iter || inf_norm(&mu) > SBW_DUAL_DIVERGENCE * mu0_scale {
            return Err(infeasible());
        }

        // Newton direction on the current piece (pseudo-inverse when |S| < k + 1 or the
        // free units' covariates are collinear).
        let (m_inv, _) = safe_inverse(&dual.gram(Some(&free)).view()).map_err(|_| infeasible())?;
        let delta = m_inv.dot(&r);
        let slope = r.dot(&delta);
        if !(slope.is_finite() && slope > 0.0) {
            // No ascent direction: stagnation at a solution within tolerance, or infeasible.
            if close {
                break;
            }
            return Err(infeasible());
        }

        // On a piece the dual is an exact quadratic, so a full step that keeps S is its
        // maximiser; otherwise backtrack on g.
        let full = &mu + &delta;
        let full_free = dual.free_set(&full);
        if full_free == free {
            mu = full;
            full_step_kept_set = true;
        } else {
            let g0 = dual.value(&mu);
            let slack = SBW_DUAL_ROUNDING * (1.0 + g0.abs());
            let mut t = 1.0;
            let mut candidate = full;
            while dual.value(&candidate) < g0 + SBW_ARMIJO_C * t * slope - slack {
                t *= 0.5;
                if t <= SBW_MIN_STEP {
                    break;
                }
                candidate = &mu + &(&delta * t);
            }
            if t <= SBW_MIN_STEP {
                if close {
                    break;
                }
                return Err(infeasible());
            }
            mu = candidate;
            full_step_kept_set = false;
        }
        iterations += 1;
    }

    // Weights on the bound-respecting piece, renormalised to sum 1.
    let weights: Vec<f64> = (0..n).map(|i| dual.z(i, &mu).max(l)).collect();
    let w_sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = if w_sum > 0.0 {
        weights.iter().map(|&w| w / w_sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };
    let lambda: Vec<f64> = mu.iter().map(|v| -v).collect();

    // Compute constraint violation
    let max_violation = compute_constraint_violation(x, &weights, target_means);

    // Compute objective value (variance of weights)
    let w_mean = 1.0 / n as f64;
    let objective: f64 = weights.iter().map(|w| (w - w_mean).powi(2)).sum();

    Ok((weights, lambda, true, iterations, objective, max_violation))
}

/// Solve SBW with approximate balance using penalized QP.
///
/// Minimizes: (1/2) * w'w + ρ * ||A*w - b||²
/// Subject to: w >= l
///
/// Uses iterative projected gradient descent.
fn solve_sbw_approximate(
    x: &ArrayView2<f64>,
    target_means: &ArrayView1<f64>,
    config: &SBWConfig,
) -> EconResult<(Vec<f64>, Vec<f64>, bool, usize, f64, f64)> {
    let n = x.nrows();
    let k = x.ncols();
    let rho = config.balance_penalty;

    // Initialize with uniform weights
    let mut weights: Vec<f64> = vec![1.0 / n as f64; n];

    // Precompute X'X for gradient (unused in simplified version)
    let _xtx_mat = xtx(x);

    // Gradient descent with projection
    let step_size = 0.01 / (1.0 + rho);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // Compute gradient
        // grad = w + ρ * X' * (X*w/n - target) * (1/n)
        // where the balance constraint is Σw_i X_ij / Σw_i = target_j

        let w_arr = Array1::from(weights.clone());
        let w_sum: f64 = weights.iter().sum();

        // Weighted mean: X*w / Σw
        let weighted_mean: Array1<f64> = if w_sum > 0.0 {
            let xw = x.t().dot(&w_arr);
            xw / w_sum
        } else {
            Array1::zeros(k)
        };

        // Residual from target
        let residual = &weighted_mean - target_means;

        // Gradient of balance penalty
        // d/dw_i [ρ * ||X*w/Σw - target||²]
        // = ρ * 2 * (X*w/Σw - target)' * d/dw_i[X*w/Σw]
        // = ρ * 2 * residual' * [X_i/Σw - X*w/(Σw)² ]
        // = ρ * 2/Σw * residual' * [X_i - X*w/Σw]
        // = ρ * 2/Σw * residual' * [X_i - weighted_mean]

        let mut gradient = Array1::zeros(n);
        for i in 0..n {
            // Variance term gradient
            gradient[i] = 2.0 * (weights[i] - 1.0 / n as f64);

            // Balance penalty gradient
            if w_sum > 0.0 {
                let mut balance_grad = 0.0;
                for j in 0..k {
                    balance_grad += residual[j] * (x[[i, j]] - weighted_mean[j]);
                }
                gradient[i] += 2.0 * rho * balance_grad / w_sum;
            }

            // Normalization constraint gradient (soft constraint)
            gradient[i] += 2.0 * rho * (w_sum - 1.0) / n as f64;
        }

        // Update with gradient descent
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        for i in 0..n {
            weights[i] -= step_size * gradient[i];
            weights[i] = weights[i].max(config.min_weight);
        }

        // Renormalize
        let w_sum: f64 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in &mut weights {
                *w /= w_sum;
            }
        }

        // Check convergence
        let max_violation = compute_constraint_violation(x, &weights, target_means);
        if grad_norm < config.tolerance && max_violation < config.balance_tol + 0.001 {
            converged = true;
            break;
        }
    }

    // Compute final statistics
    let max_violation = compute_constraint_violation(x, &weights, target_means);
    let w_mean = 1.0 / n as f64;
    let objective: f64 = weights.iter().map(|w| (w - w_mean).powi(2)).sum();

    // Lambda not available for approximate solution
    let lambda = vec![0.0; k + 1];

    Ok((
        weights,
        lambda,
        converged,
        iterations,
        objective,
        max_violation,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract rows from a matrix given indices.
fn extract_rows(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let k = x.ncols();
    let n = indices.len();
    let mut result = Array2::zeros((n, k));

    for (new_i, &old_i) in indices.iter().enumerate() {
        for j in 0..k {
            result[[new_i, j]] = x[[old_i, j]];
        }
    }

    result
}

/// Compute maximum constraint violation.
fn compute_constraint_violation(
    x: &ArrayView2<f64>,
    weights: &[f64],
    target_means: &ArrayView1<f64>,
) -> f64 {
    let n = x.nrows();
    let k = x.ncols();

    let w_sum: f64 = weights.iter().sum();
    if w_sum == 0.0 {
        return f64::INFINITY;
    }

    let mut max_violation = 0.0_f64;
    for j in 0..k {
        let weighted_mean: f64 = (0..n).map(|i| weights[i] * x[[i, j]]).sum::<f64>() / w_sum;
        let violation = (weighted_mean - target_means[j]).abs();
        max_violation = f64::max(max_violation, violation);
    }

    max_violation
}

/// Compute balance statistics.
fn compute_balance_stats(
    x_target: &ArrayView2<f64>,
    x_reweight: &ArrayView2<f64>,
    cov_names: &[String],
    weights: Option<&[f64]>,
) -> BalanceStats {
    let k = x_target.ncols();
    let n_target = x_target.nrows();
    let n_reweight = x_reweight.nrows();

    let mut std_diff = Vec::with_capacity(k);
    let mut var_ratio = Vec::with_capacity(k);
    let mut mean_target = Vec::with_capacity(k);
    let mut mean_reweighted = Vec::with_capacity(k);

    for j in 0..k {
        // Target group mean (unweighted)
        let target_mean: f64 = x_target.column(j).iter().sum::<f64>() / n_target as f64;

        // Reweight group weighted mean
        let reweight_mean: f64 = if let Some(w) = weights {
            let w_sum: f64 = w.iter().sum();
            if w_sum > 0.0 {
                (0..n_reweight)
                    .map(|i| w[i] * x_reweight[[i, j]])
                    .sum::<f64>()
                    / w_sum
            } else {
                x_reweight.column(j).iter().sum::<f64>() / n_reweight as f64
            }
        } else {
            x_reweight.column(j).iter().sum::<f64>() / n_reweight as f64
        };

        // Target group variance
        let target_var: f64 = x_target
            .column(j)
            .iter()
            .map(|&x| (x - target_mean).powi(2))
            .sum::<f64>()
            / (n_target - 1).max(1) as f64;

        // Reweight group weighted variance
        let reweight_var: f64 = if let Some(w) = weights {
            let w_sum: f64 = w.iter().sum();
            if w_sum > 0.0 {
                let weighted_sq_diff: f64 = (0..n_reweight)
                    .map(|i| w[i] * (x_reweight[[i, j]] - reweight_mean).powi(2))
                    .sum();
                weighted_sq_diff / w_sum
            } else {
                x_reweight
                    .column(j)
                    .iter()
                    .map(|&x| (x - reweight_mean).powi(2))
                    .sum::<f64>()
                    / (n_reweight - 1).max(1) as f64
            }
        } else {
            x_reweight
                .column(j)
                .iter()
                .map(|&x| (x - reweight_mean).powi(2))
                .sum::<f64>()
                / (n_reweight - 1).max(1) as f64
        };

        // Pooled standard deviation
        let pooled_sd = ((target_var + reweight_var) / 2.0).sqrt().max(1e-10);

        // Standardized difference
        let sd = (target_mean - reweight_mean) / pooled_sd;

        // Variance ratio
        let vr = if reweight_var > 1e-10 {
            target_var / reweight_var
        } else {
            1.0
        };

        std_diff.push(sd);
        var_ratio.push(vr);
        mean_target.push(target_mean);
        mean_reweighted.push(reweight_mean);
    }

    let max_std_diff = std_diff.iter().map(|d| d.abs()).fold(0.0, f64::max);
    let mean_std_diff = std_diff.iter().map(|d| d.abs()).sum::<f64>() / k as f64;

    BalanceStats {
        covariate_names: cov_names.to_vec(),
        std_diff,
        max_std_diff,
        mean_std_diff,
        var_ratio,
        mean_target,
        mean_reweighted,
    }
}

/// Compute effective sample size.
/// ESS = (Σw)² / Σw²
fn compute_ess(weights: &[f64], treatment: &ArrayView1<f64>) -> (f64, f64, f64) {
    let n = treatment.len();

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

        if treatment[i] >= 0.5 {
            w_sum_t += w;
            w_sq_sum_t += w * w;
        } else {
            w_sum_c += w;
            w_sq_sum_c += w * w;
        }
    }

    let ess_all = if w_sq_sum_all > 0.0 {
        w_sum_all.powi(2) / w_sq_sum_all
    } else {
        0.0
    };
    let ess_t = if w_sq_sum_t > 0.0 {
        w_sum_t.powi(2) / w_sq_sum_t
    } else {
        0.0
    };
    let ess_c = if w_sq_sum_c > 0.0 {
        w_sum_c.powi(2) / w_sq_sum_c
    } else {
        0.0
    };

    (ess_all, ess_t, ess_c)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use ndarray::array;
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

    fn lcg(seed: &mut u64) -> f64 {
        *seed = 1_103_515_245u64.wrapping_mul(*seed).wrapping_add(12_345) % 2_147_483_648;
        *seed as f64 / 2_147_483_648.0
    }

    /// `rows` x `cols` matrix of LCG draws, filled row-major.
    fn lcg_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let values: Vec<f64> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        Array2::from_shape_vec((rows, cols), values).unwrap()
    }

    /// The 240 x 3 control matrix shared by the dense-KKT and certificate tests, and the
    /// target `column means + shift`.
    fn lcg_controls_and_target(shift: [f64; 3]) -> (Array2<f64>, Array1<f64>) {
        let xc = lcg_matrix(240, 3, 2026);
        let target = xc.mean_axis(Axis(0)).unwrap() + &Array1::from(shift.to_vec());
        (xc, target)
    }

    /// Verbatim copy of the pre-change `solve_sbw_exact` (dense (n+k+1)^2 KKT build,
    /// `safe_inverse`, clip at `min_weight = 0`, renormalise), returning (weights, lambda).
    fn dense_kkt_reference(
        x: &ArrayView2<f64>,
        target_means: &ArrayView1<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let config = SBWConfig::default();
        let n = x.nrows();
        let k = x.ncols();
        let n_constraints = k + 1;
        let system_size = n + n_constraints;
        let mut kkt = Array2::<f64>::zeros((system_size, system_size));
        for i in 0..n {
            kkt[[i, i]] = 1.0;
        }
        for i in 0..n {
            for j in 0..k {
                let val = x[[i, j]] / n as f64;
                kkt[[i, n + j]] = val;
                kkt[[n + j, i]] = val;
            }
            kkt[[i, n + k]] = 1.0 / n as f64;
            kkt[[n + k, i]] = 1.0 / n as f64;
        }
        let mut rhs = Array1::<f64>::zeros(system_size);
        for j in 0..k {
            rhs[n + j] = target_means[j];
        }
        rhs[n + k] = 1.0;
        let (kkt_inv, _cond) = safe_inverse(&kkt.view()).unwrap();
        let solution = kkt_inv.dot(&rhs);
        let weights: Vec<f64> = solution.iter().take(n).cloned().collect();
        let lambda: Vec<f64> = solution.iter().skip(n).cloned().collect();
        let weights: Vec<f64> = weights.iter().map(|&w| w.max(config.min_weight)).collect();
        let w_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = if w_sum > 0.0 {
            weights.iter().map(|&w| w / w_sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
        (weights, lambda)
    }

    /// Outcome for create_test_data(): y = 1 + 2 x1 - x2 + 0.5 d + e (e fixed, 2 decimals).
    const SBW_Y: [f64; 30] = [
        2.32, 2.22, 2.30, 2.29, 2.29, 2.37, 2.04, 2.55, 2.22, 2.27, 1.21, 1.64, 1.56, 2.15, 1.48,
        1.98, 1.38, 2.56, 1.38, 1.61, 1.50, 1.68, 1.42, 2.20, 1.45, 2.01, 1.41, 2.27, 1.51, 1.59,
    ];

    #[test]
    fn test_sbw_exact_matches_r_sbw() {
        // R reference (R 4.3.3, quadprog 1.5.8; the sbw call is the plan's sbw 1.2 record):
        //   Xc <- cbind(x1[d==0], x2[d==0]); tm <- c(mean(x1[d==1]), mean(x2[d==1])); n <- 20
        //   qp <- quadprog::solve.QP(diag(n), rep(0, n), cbind(t(rbind(t(Xc), 1)) / n, diag(n)),
        //                            c(tm, 1, rep(0, n)), meq = 3)      # exact QP, n scale
        //   qp$solution * 409  ->  2464 0 500 2288 0 0 0 0 500 0 0 1096 0 1008 0 0 0 324 0 0
        //   # signed multipliers from the active set S = {w > 0} (quadprog reports |mu|):
        //   A <- rbind(t(Xc), 1) / n; mu <- solve(A[, S] %*% t(A[, S]), c(tm, 1)); lambda <- -mu
        //   s <- sbw::sbw(dat, ind = "d", out = "y", bal = list(bal_cov = c("x1","x2"), bal_alg = FALSE,
        //                 bal_tol = 1e-8, bal_std = "group"), sol = list(sol_nam = "quadprog"),
        //                 par = list(par_est = "att"))                  # sbw::estimate(s): ATT 0.555337388135859
        let (treatment, covariates) = create_test_data();
        let r = run_sbw(&treatment.view(), &covariates.view(), SBWConfig::default()).unwrap();
        let expected = [
            2464.0, 0.0, 500.0, 2288.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 1096.0, 0.0, 1008.0,
            0.0, 0.0, 0.0, 324.0, 0.0, 0.0,
        ];
        for i in 0..10 {
            assert!((r.weights[i] - 1.0).abs() < 1e-12);
        }
        for (j, e) in expected.iter().enumerate() {
            assert!(
                (r.weights[10 + j] - e / 409.0).abs() < 1e-9,
                "control {j}: {} vs {}",
                r.weights[10 + j],
                e / 409.0
            );
        }
        assert!(r.converged);
        assert!(r.max_constraint_violation < 1e-10);
        let lambda = [3172.6161369150, -4424.4498777446, 255.0611246939];
        for (got, want) in r.lambda.iter().zip(lambda) {
            assert!((got - want).abs() < 1e-8 * want.abs(), "{got} vs {want}");
        }
        let (sw, swy) = (10..30).fold((0.0, 0.0), |(a, b), i| {
            (a + r.weights[i], b + r.weights[i] * SBW_Y[i])
        });
        let att = SBW_Y[..10].iter().sum::<f64>() / 10.0 - swy / sw;
        assert!((att - 0.555337408312916).abs() < 1e-10, "ATT {att}"); // exact QP optimum
        assert!((att - 0.555337388135859).abs() < 1e-6, "ATT {att}"); // R sbw 1.2 (bal_tol 1e-8)
    }

    #[test]
    fn test_sbw_exact_matches_dense_kkt_when_no_bound_binds() {
        // target = column means + [0.02, -0.01, 0.015]; R: unconstrained weights in [0.7938, 1.2560].
        let (xc, target) = lcg_controls_and_target([0.02, -0.01, 0.015]);
        // port check: the first three draws (R gives the same with exact integer arithmetic)
        assert_eq!(
            [xc[[0, 0]], xc[[0, 1]], xc[[0, 2]]],
            [
                0.089137417729943991,
                0.36493223533034325,
                0.078967132139950991
            ]
        );
        let (w_ref, lambda_ref) = dense_kkt_reference(&xc.view(), &target.view());
        assert!(
            w_ref.iter().all(|&w| w > 0.0),
            "precondition: no bound binds"
        );
        let (w, lambda, converged, iterations, _, viol) =
            solve_sbw_exact(&xc.view(), &target.view(), &SBWConfig::default()).unwrap();
        assert!(
            converged && iterations <= 2 && viol < 1e-12,
            "converged {converged}, iterations {iterations}, violation {viol}"
        );
        for (a, b) in w.iter().zip(&w_ref) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        for (a, b) in lambda.iter().zip(&lambda_ref) {
            assert!((a - b).abs() < 1e-8 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn test_sbw_exact_kkt_certificate_with_binding_bounds() {
        // target = column means + [0.15, -0.12, 0.10]: 24 unconstrained weights are negative.
        // R quadprog 1.5.8 optimum (n scale, sum = 240), solve.QP(diag(240), 0,
        //   cbind(cbind(xc, 1) / 240, diag(240)), c(target, 1, rep(l, 240)), meq = 4):
        //   l = 0:    max w 3.249634429310, sum w^2 385.061327595962, 34 weights at the bound
        //   l = 0.1:  max w 3.348946038565, sum w^2 388.293680009016, 40 at the bound
        //   l = -0.5: max w 3.058753620112, sum w^2 380.430452018262,  6 at the bound
        let (xc, target) = lcg_controls_and_target([0.15, -0.12, 0.10]);
        for (l, max_w, ssq, at_bound) in [
            (0.0, 3.249634429310, 385.061327595962, 34),
            (0.1, 3.348946038565, 388.293680009016, 40),
            (-0.5, 3.058753620112, 380.430452018262, 6),
        ] {
            let cfg = SBWConfig {
                min_weight: l,
                ..Default::default()
            };
            let (w, lambda, converged, _, _, viol) =
                solve_sbw_exact(&xc.view(), &target.view(), &cfg).unwrap();
            let wn: Vec<f64> = w.iter().map(|v| v * 240.0).collect();
            assert!(converged && viol < 1e-10, "l {l}: violation {viol}");
            // optimality certificate: w_i = max(l, z_i), z_i = -(lambda . [x_i, 1]) / n
            for i in 0..240 {
                let z = -(lambda[0] * xc[[i, 0]]
                    + lambda[1] * xc[[i, 1]]
                    + lambda[2] * xc[[i, 2]]
                    + lambda[3])
                    / 240.0;
                assert!((wn[i] - z.max(l)).abs() < 1e-9, "l {l}, unit {i}");
            }
            assert!((wn.iter().cloned().fold(f64::MIN, f64::max) - max_w).abs() < 1e-8);
            assert!((wn.iter().map(|v| v * v).sum::<f64>() - ssq).abs() < 1e-7);
            assert_eq!(
                wn.iter().filter(|&&v| (v - l).abs() < 1e-9).count(),
                at_bound,
                "l {l}"
            );
        }
    }

    #[test]
    fn test_sbw_exact_infeasible_is_an_error() {
        // ATC on create_test_data(): the control mean of x2 (0.474) is below every treated x2
        // (min 0.5), so no non-negative treated weights reach it. R sbw also refuses it.
        let (treatment, covariates) = create_test_data();
        let err = run_sbw(
            &treatment.view(),
            &covariates.view(),
            SBWConfig {
                estimand: SBWEstimand::ATC,
                ..Default::default()
            },
        )
        .unwrap_err();
        expect_invalid_spec(err, "balance_tol");
        // the same through the dataset entry point names the real column
        let ds = create_test_dataset();
        let err = sbw(
            &ds,
            "treatment",
            &["x1", "x2"],
            SBWConfig {
                estimand: SBWEstimand::ATC,
                ..Default::default()
            },
        )
        .unwrap_err();
        expect_invalid_spec(err, "covariate x2");
    }

    #[test]
    fn test_sbw_exact_infeasible_without_range_violation_is_an_error() {
        // Each target mean lies inside its covariate's range, but no convex combination of
        // the four corner points reaches the pair (0.9, 0.9): the range pre-check passes and
        // the Newton solver must detect infeasibility itself.
        let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let covariates = array![
            [0.9, 0.9],
            [0.9, 0.9],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5]
        ];
        let err = run_sbw(&treatment.view(), &covariates.view(), SBWConfig::default()).unwrap_err();
        expect_invalid_spec(err, "balance_tol");
    }

    #[test]
    fn test_sbw_atc_with_swapped_labels_matches_att() {
        // swapping labels turns the ATT problem into an ATC one: same weights
        let (treatment, covariates) = create_test_data();
        let swapped = treatment.mapv(|d| 1.0 - d);
        let att = run_sbw(&treatment.view(), &covariates.view(), SBWConfig::default()).unwrap();
        let atc = run_sbw(
            &swapped.view(),
            &covariates.view(),
            SBWConfig {
                estimand: SBWEstimand::ATC,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(atc.n_target, 10);
        assert_eq!(atc.n_reweighted, 20);
        for (a, b) in att.weights.iter().zip(&atc.weights) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sbw_rejects_invalid_input() {
        let (t, x) = create_test_data();
        let d = SBWConfig::default;
        let cases = [
            (
                SBWConfig {
                    balance_tol: f64::NAN,
                    ..d()
                },
                "balance_tol",
            ),
            (
                SBWConfig {
                    balance_tol: -0.1,
                    ..d()
                },
                "balance_tol",
            ),
            (
                SBWConfig {
                    tolerance: 0.0,
                    ..d()
                },
                "tolerance",
            ),
            (SBWConfig { max_iter: 0, ..d() }, "max_iter"),
            (
                SBWConfig {
                    min_weight: f64::NAN,
                    ..d()
                },
                "min_weight",
            ),
            (
                SBWConfig {
                    min_weight: 1.5,
                    ..d()
                },
                "min_weight",
            ),
            (
                SBWConfig {
                    balance_tol: 0.1,
                    balance_penalty: 0.0,
                    ..d()
                },
                "balance_penalty",
            ),
        ];
        for (cfg, needle) in cases {
            expect_invalid_spec(run_sbw(&t.view(), &x.view(), cfg).unwrap_err(), needle);
        }
        let mut x_nan = x.clone();
        x_nan[[12, 1]] = f64::NAN;
        expect_invalid_spec(
            run_sbw(&t.view(), &x_nan.view(), SBWConfig::default()).unwrap_err(),
            "finite",
        );
        let mut t_inf = t.clone();
        t_inf[3] = f64::INFINITY;
        expect_invalid_spec(
            run_sbw(&t_inf.view(), &x.view(), SBWConfig::default()).unwrap_err(),
            "finite",
        );
    }

    #[test]
    fn test_sbw_honours_cancellation() {
        let (t, x) = create_test_data();
        let r = resource::with_budget(cancelled_budget(), || {
            run_sbw(&t.view(), &x.view(), SBWConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
        let r = resource::with_budget(cancelled_budget(), || {
            run_sbw(
                &t.view(),
                &x.view(),
                SBWConfig {
                    balance_tol: 0.1,
                    ..Default::default()
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    #[test]
    fn test_sbw_exact_large_n_is_tractable() {
        // 2,000 treated + 20,000 controls, 3 lcg covariates, treated shifted by +0.3 in x0
        // (with uniform covariates the unconstrained weights are about 1 + 3.6 (x0 - 0.5), so
        // the bound binds; a +0.15 shift would leave every weight above 0.1). Do not run before
        // the fix: the old path allocates a 20,004^2 dense matrix (3.2 GB) and inverts it.
        // No timing assertion.
        let (n_t, n_c) = (2_000, 20_000);
        let mut x = lcg_matrix(n_t + n_c, 3, 7);
        for i in 0..n_t {
            x[[i, 0]] += 0.3;
        }
        let t = Array1::from_shape_fn(n_t + n_c, |i| if i < n_t { 1.0 } else { 0.0 });
        let r = run_sbw(&t.view(), &x.view(), SBWConfig::default()).unwrap();
        assert!(r.converged);
        assert!(r.iterations <= 50, "iterations {}", r.iterations);
        assert!(
            r.max_constraint_violation < 1e-9,
            "violation {}",
            r.max_constraint_violation
        );
        assert!(r.weights.iter().all(|&w| w >= 0.0));
        assert!(
            r.weights[n_t..].iter().filter(|&&w| w == 0.0).count() > 0,
            "precondition: the bound binds"
        );
    }

    /// Create test data with known imbalance but overlapping support.
    /// SBW requires overlap: control covariate range must contain treated means.
    fn create_test_data() -> (Array1<f64>, Array2<f64>) {
        // Treatment indicator: 10 treated, 20 control
        let treatment = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        // Covariates with imbalance BUT OVERLAPPING SUPPORT
        // Treated mean ~0.65 for X1, control range 0.3-1.2 includes 0.65
        let covariates = array![
            // Treated (mean ~0.65 for X1, ~0.55 for X2)
            [0.6, 0.5],
            [0.7, 0.6],
            [0.65, 0.55],
            [0.75, 0.6],
            [0.6, 0.5],
            [0.7, 0.55],
            [0.55, 0.5],
            [0.8, 0.65],
            [0.65, 0.55],
            [0.6, 0.5],
            // Control (lower mean ~0.5 for X1, but range 0.3-1.2 covers treated mean)
            [0.3, 0.3],
            [0.5, 0.4],
            [0.4, 0.35],
            [1.0, 0.8],
            [0.35, 0.3],
            [0.9, 0.7],
            [0.3, 0.25],
            [1.2, 0.9],
            [0.4, 0.35],
            [0.5, 0.4],
            [0.35, 0.3],
            [0.6, 0.5],
            [0.45, 0.38],
            [0.95, 0.75],
            [0.32, 0.28],
            [0.85, 0.65],
            [0.28, 0.22],
            [1.1, 0.85],
            [0.42, 0.35],
            [0.55, 0.45],
        ];

        (treatment, covariates)
    }

    /// Create test dataset with overlapping support for SBW.
    fn create_test_dataset() -> Dataset {
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "x1" => [
                // Treated (mean ~0.65)
                0.6, 0.7, 0.65, 0.75, 0.6, 0.7, 0.55, 0.8, 0.65, 0.6,
                // Control with overlapping range (0.28-1.2 covers treated mean 0.65)
                0.3, 0.5, 0.4, 1.0, 0.35, 0.9, 0.3, 1.2, 0.4, 0.5,
                0.35, 0.6, 0.45, 0.95, 0.32, 0.85, 0.28, 1.1, 0.42, 0.55
            ],
            "x2" => [
                // Treated (mean ~0.55)
                0.5, 0.6, 0.55, 0.6, 0.5, 0.55, 0.5, 0.65, 0.55, 0.5,
                // Control with overlapping range
                0.3, 0.4, 0.35, 0.8, 0.3, 0.7, 0.25, 0.9, 0.35, 0.4,
                0.3, 0.5, 0.38, 0.75, 0.28, 0.65, 0.22, 0.85, 0.35, 0.45
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_sbw_exact_balance_att() {
        let (treatment, covariates) = create_test_data();

        let config = SBWConfig {
            estimand: SBWEstimand::ATT,
            balance_tol: 0.0, // Exact balance
            ..Default::default()
        };

        let result = run_sbw(&treatment.view(), &covariates.view(), config).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 30);
        assert_eq!(result.n_target, 10);
        assert_eq!(result.n_reweighted, 20);
        assert_eq!(result.weights.len(), 30);

        // Treated units should have weight 1.0
        for i in 0..10 {
            assert!(
                (result.weights[i] - 1.0).abs() < 1e-6,
                "Treated weight should be 1.0, got {}",
                result.weights[i]
            );
        }

        // Exact balance is achieved
        assert!(
            result.max_constraint_violation < 1e-10,
            "max constraint violation {}",
            result.max_constraint_violation
        );
        assert!(
            result.balance_after.max_std_diff < 1e-8,
            "SBW should achieve exact balance, got max_std_diff = {}",
            result.balance_after.max_std_diff
        );

        // Balance should improve from initial imbalance
        assert!(
            result.balance_after.max_std_diff < result.balance_before.max_std_diff,
            "Balance should improve: before={}, after={}",
            result.balance_before.max_std_diff,
            result.balance_after.max_std_diff
        );
    }

    #[test]
    fn test_sbw_approximate_balance() {
        let (treatment, covariates) = create_test_data();

        let config = SBWConfig {
            estimand: SBWEstimand::ATT,
            balance_tol: 0.1, // Approximate balance
            balance_penalty: 100.0,
            ..Default::default()
        };

        let result = run_sbw(&treatment.view(), &covariates.view(), config).unwrap();

        // Should still achieve reasonable balance
        assert!(
            result.balance_after.max_std_diff < 0.2,
            "Approximate SBW should achieve reasonable balance, got {}",
            result.balance_after.max_std_diff
        );

        // Weights should be positive
        assert!(result.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_sbw_dataset_interface() {
        let dataset = create_test_dataset();

        let config = SBWConfig {
            estimand: SBWEstimand::ATT,
            ..Default::default()
        };

        let result = sbw(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        // Check that column names are used
        assert_eq!(result.balance_after.covariate_names, vec!["x1", "x2"]);

        // Should achieve reasonable balance (<0.3 std diff is acceptable)
        assert!(
            result.balance_after.max_std_diff < 0.3,
            "Balance should improve, got max_std_diff = {}",
            result.balance_after.max_std_diff
        );
    }

    #[test]
    fn test_ess_computation() {
        // Uniform weights should give ESS = n
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0]);

        let (ess_all, ess_t, ess_c) = compute_ess(&weights, &treatment.view());

        assert!((ess_all - 4.0).abs() < 1e-10);
        assert!((ess_t - 2.0).abs() < 1e-10);
        assert!((ess_c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ess_with_varying_weights() {
        // Weights [2, 2, 1, 1] for control
        // ESS = (2+2+1+1)² / (4+4+1+1) = 36/10 = 3.6
        let weights = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
        let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let (_, _, ess_c) = compute_ess(&weights, &treatment.view());

        // Control: (2+2+1+1)² / (4+4+1+1) = 36/10 = 3.6
        assert!((ess_c - 3.6).abs() < 1e-10);
    }

    #[test]
    fn test_balance_stats() {
        let x_target = array![[1.0, 0.5], [1.2, 0.6], [1.1, 0.55]];
        let x_reweight = array![[0.5, 0.2], [0.6, 0.3], [0.55, 0.25]];
        let names = vec!["x1".to_string(), "x2".to_string()];

        let stats = compute_balance_stats(&x_target.view(), &x_reweight.view(), &names, None);

        // Target has higher means, so std_diff should be positive
        assert!(stats.std_diff[0] > 0.0);
        assert!(stats.std_diff[1] > 0.0);

        // There should be imbalance
        assert!(stats.max_std_diff > 1.0);
    }

    #[test]
    fn test_display_result() {
        let dataset = create_test_dataset();
        let config = SBWConfig::default();

        let result = sbw(&dataset, "treatment", &["x1", "x2"], config).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Stable Balancing Weights"));
        assert!(output.contains("ESS"));
        assert!(output.contains("Balance"));
    }

    #[test]
    fn test_sbw_empty_group() {
        let treatment = Array1::from(vec![1.0, 1.0, 1.0]); // All treated
        let covariates = array![[1.0], [1.2], [1.1]];

        let config = SBWConfig::default();
        let result = run_sbw(&treatment.view(), &covariates.view(), config);

        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_violation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let weights = vec![0.5, 0.3, 0.2]; // Sum = 1
        let target = Array1::from(vec![1.5, 2.5]); // Exact match would need weights

        let violation = compute_constraint_violation(&x.view(), &weights, &target.view());

        // Weighted mean: [0.5*1 + 0.3*2 + 0.2*3, 0.5*2 + 0.3*3 + 0.2*4]
        //              = [0.5 + 0.6 + 0.6, 1.0 + 0.9 + 0.8] = [1.7, 2.7]
        // Violation: max(|1.7-1.5|, |2.7-2.5|) = max(0.2, 0.2) = 0.2
        assert!((violation - 0.2).abs() < 1e-10);
    }
}
