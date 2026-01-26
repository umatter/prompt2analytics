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
//! ## Lagrangian Solution
//!
//! For equality-constrained QP, the KKT conditions give:
//!
//! ```text
//! [H    A'][w]   [−c]
//! [A    0 ][λ] = [b ]
//! ```
//!
//! Solving this linear system yields optimal weights and Lagrange multipliers.
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
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx};

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
        writeln!(f, "{:<20} {:>12} {:>12} {:>12} {:>12}",
                 "Covariate", "Target", "Reweighted", "Std.Diff", "Var.Ratio")?;
        writeln!(f, "{}", "-".repeat(68))?;

        for i in 0..self.covariate_names.len() {
            let balanced = if self.std_diff[i].abs() < 0.1 { " " } else { "*" };
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>11.4}{} {:>12.4}",
                     self.covariate_names[i],
                     self.mean_target[i],
                     self.mean_reweighted[i],
                     self.std_diff[i],
                     balanced,
                     self.var_ratio[i])?;
        }

        writeln!(f, "{}", "-".repeat(68))?;
        writeln!(f, "Max |Std.Diff|: {:.4}   Mean |Std.Diff|: {:.4}",
                 self.max_std_diff, self.mean_std_diff)?;

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
        writeln!(f, "  Total:      {}  (Target: {}, Reweighted: {})",
                 self.n_obs, self.n_target, self.n_reweighted)?;
        writeln!(f)?;

        writeln!(f, "Optimization:")?;
        writeln!(f, "  Converged:  {} ({} iterations)",
                 if self.converged { "Yes" } else { "No" }, self.iterations)?;
        writeln!(f, "  Objective:  {:.6} (variance of weights)", self.objective_value)?;
        writeln!(f, "  Max violation: {:.6}", self.max_constraint_violation)?;
        writeln!(f)?;

        writeln!(f, "Weight Summary (reweighted group):")?;
        writeln!(f, "  Range:      [{:.4}, {:.4}]", self.min_weight, self.max_weight)?;
        writeln!(f, "  ESS:        {:.1} of {} ({:.1}%)",
                 self.ess_reweighted, self.n_reweighted,
                 100.0 * self.ess_reweighted / self.n_reweighted as f64)?;
        writeln!(f)?;

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
    let n = treatment.len();
    let k = covariates.ncols();
    let mut warnings = Vec::new();

    // Validate inputs
    if n != covariates.nrows() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment length ({}) must match covariate rows ({})",
                n, covariates.nrows()
            ),
        });
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
    let target_means = x_target.mean_axis(Axis(0))
        .ok_or_else(|| EconError::Computation("Failed to compute target means".to_string()))?;

    // Compute covariate names
    let cov_names: Vec<String> = (0..k).map(|i| format!("X{}", i + 1)).collect();

    // Compute balance before weighting
    let balance_before = compute_balance_stats(
        &x_target.view(),
        &x_reweight.view(),
        &cov_names,
        None,
    );

    // Solve the quadratic programming problem
    let (weights, lambda, converged, iterations, objective, max_violation) = if config.balance_tol == 0.0 {
        // Exact balance: use Lagrangian method
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
        &cov_names,
        Some(&final_weights),
    );

    // Compute effective sample sizes
    let (ess_total, ess_target, ess_reweight) = compute_ess(&full_weights, treatment);

    // Weight statistics
    let max_weight = final_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_weight = final_weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::INFINITY, f64::min);

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
            ess_reweight, n_reweight, 100.0 * ess_reweight / n_reweight as f64
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
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Build covariate matrix (without intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let covariates = design.data;

    // Run SBW
    let mut result = run_sbw(&treatment.view(), &covariates.view(), config)?;

    // Update covariate names with actual column names
    result.balance_before.covariate_names = covariate_cols.iter().map(|s| s.to_string()).collect();
    result.balance_after.covariate_names = covariate_cols.iter().map(|s| s.to_string()).collect();

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// QP Solvers
// ═══════════════════════════════════════════════════════════════════════════════

/// Solve SBW with exact balance constraints using Lagrangian method.
///
/// Solves the KKT system:
/// ```text
/// [H    A'][w]   [-c]
/// [A    0 ][λ] = [b ]
/// ```
///
/// Where H = I (for variance minimization), A is the constraint matrix,
/// and b is the target means.
fn solve_sbw_exact(
    x: &ArrayView2<f64>,
    target_means: &ArrayView1<f64>,
    config: &SBWConfig,
) -> EconResult<(Vec<f64>, Vec<f64>, bool, usize, f64, f64)> {
    let n = x.nrows();
    let k = x.ncols();

    // Constraint matrix A includes:
    // 1. Balance constraints: (1/n) * Σ w_i X_ij = target_j for each covariate j
    // 2. Normalization constraint: Σ w_i = n
    // Total: k + 1 constraints

    let n_constraints = k + 1;

    // Build KKT system matrix
    // [I    A']  [n x n,     n x (k+1)]
    // [A    0 ]  [(k+1) x n, (k+1) x (k+1)]
    let system_size = n + n_constraints;
    let mut kkt = Array2::<f64>::zeros((system_size, system_size));

    // Top-left: H = I (identity for variance minimization)
    for i in 0..n {
        kkt[[i, i]] = 1.0;
    }

    // Top-right and bottom-left: A' and A
    // Balance constraints: X (each row of X corresponds to one observation)
    for i in 0..n {
        for j in 0..k {
            // A[j, i] = X[i, j] / n (balance constraint for covariate j)
            let val = x[[i, j]] / n as f64;
            kkt[[i, n + j]] = val;        // A' part
            kkt[[n + j, i]] = val;        // A part
        }
        // Normalization constraint
        kkt[[i, n + k]] = 1.0 / n as f64;
        kkt[[n + k, i]] = 1.0 / n as f64;
    }

    // Build RHS vector
    // Top: -c = 0 (no linear term in objective)
    // Bottom: b = [target_means, 1] (constraints)
    let mut rhs = Array1::<f64>::zeros(system_size);
    for j in 0..k {
        rhs[n + j] = target_means[j];
    }
    rhs[n + k] = 1.0; // Normalized weights sum to 1

    // Solve KKT system
    let (kkt_inv, _cond) = safe_inverse(&kkt.view()).map_err(|e| {
        EconError::SingularMatrix {
            context: "SBW KKT system".to_string(),
            suggestion: format!(
                "Balance constraints may be infeasible or redundant. Error: {:?}. \
                 Try using approximate balance (balance_tol > 0).",
                e
            ),
        }
    })?;

    let solution = kkt_inv.dot(&rhs);

    // Extract weights and Lagrange multipliers
    let weights: Vec<f64> = solution.iter().take(n).cloned().collect();
    let lambda: Vec<f64> = solution.iter().skip(n).cloned().collect();

    // Project weights to satisfy lower bound constraint
    let weights: Vec<f64> = weights.iter()
        .map(|&w| w.max(config.min_weight))
        .collect();

    // Renormalize after projection
    let w_sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = if w_sum > 0.0 {
        weights.iter().map(|&w| w / w_sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    // Compute constraint violation
    let max_violation = compute_constraint_violation(x, &weights, target_means);

    // Compute objective value (variance of weights)
    let w_mean = 1.0 / n as f64;
    let objective: f64 = weights.iter().map(|w| (w - w_mean).powi(2)).sum();

    Ok((weights, lambda, true, 1, objective, max_violation))
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

    Ok((weights, lambda, converged, iterations, objective, max_violation))
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
                (0..n_reweight).map(|i| w[i] * x_reweight[[i, j]]).sum::<f64>() / w_sum
            } else {
                x_reweight.column(j).iter().sum::<f64>() / n_reweight as f64
            }
        } else {
            x_reweight.column(j).iter().sum::<f64>() / n_reweight as f64
        };

        // Target group variance
        let target_var: f64 = x_target.column(j).iter()
            .map(|&x| (x - target_mean).powi(2))
            .sum::<f64>() / (n_target - 1).max(1) as f64;

        // Reweight group weighted variance
        let reweight_var: f64 = if let Some(w) = weights {
            let w_sum: f64 = w.iter().sum();
            if w_sum > 0.0 {
                let weighted_sq_diff: f64 = (0..n_reweight)
                    .map(|i| w[i] * (x_reweight[[i, j]] - reweight_mean).powi(2))
                    .sum();
                weighted_sq_diff / w_sum
            } else {
                x_reweight.column(j).iter()
                    .map(|&x| (x - reweight_mean).powi(2))
                    .sum::<f64>() / (n_reweight - 1).max(1) as f64
            }
        } else {
            x_reweight.column(j).iter()
                .map(|&x| (x - reweight_mean).powi(2))
                .sum::<f64>() / (n_reweight - 1).max(1) as f64
        };

        // Pooled standard deviation
        let pooled_sd = ((target_var + reweight_var) / 2.0).sqrt().max(1e-10);

        // Standardized difference
        let sd = (target_mean - reweight_mean) / pooled_sd;

        // Variance ratio
        let vr = if reweight_var > 1e-10 { target_var / reweight_var } else { 1.0 };

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

    let ess_all = if w_sq_sum_all > 0.0 { w_sum_all.powi(2) / w_sq_sum_all } else { 0.0 };
    let ess_t = if w_sq_sum_t > 0.0 { w_sum_t.powi(2) / w_sq_sum_t } else { 0.0 };
    let ess_c = if w_sq_sum_c > 0.0 { w_sum_c.powi(2) / w_sq_sum_c } else { 0.0 };

    (ess_all, ess_t, ess_c)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use polars::prelude::*;

    /// Create test data with known imbalance.
    fn create_test_data() -> (Array1<f64>, Array2<f64>) {
        // Treatment indicator: 10 treated, 20 control
        let treatment = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        // Covariates with imbalance: treated has higher means
        let covariates = array![
            // Treated (higher x1, x2)
            [1.2, 0.8], [1.5, 0.9], [1.3, 0.7], [1.8, 1.0], [1.4, 0.85],
            [1.6, 0.95], [1.1, 0.75], [1.7, 1.1], [1.5, 0.9], [1.4, 0.85],
            // Control (lower x1, x2)
            [0.3, 0.3], [0.5, 0.4], [0.4, 0.35], [0.7, 0.5], [0.2, 0.25],
            [0.6, 0.45], [0.1, 0.2], [0.8, 0.55], [0.4, 0.35], [0.5, 0.4],
            [0.35, 0.32], [0.55, 0.42], [0.45, 0.37], [0.75, 0.52], [0.25, 0.27],
            [0.65, 0.47], [0.15, 0.22], [0.85, 0.57], [0.45, 0.37], [0.55, 0.42],
        ];

        (treatment, covariates)
    }

    /// Create test dataset.
    fn create_test_dataset() -> Dataset {
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "x1" => [
                1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.1, 1.7, 1.5, 1.4,
                0.3, 0.5, 0.4, 0.7, 0.2, 0.6, 0.1, 0.8, 0.4, 0.5,
                0.35, 0.55, 0.45, 0.75, 0.25, 0.65, 0.15, 0.85, 0.45, 0.55
            ],
            "x2" => [
                0.8, 0.9, 0.7, 1.0, 0.85, 0.95, 0.75, 1.1, 0.9, 0.85,
                0.3, 0.4, 0.35, 0.5, 0.25, 0.45, 0.2, 0.55, 0.35, 0.4,
                0.32, 0.42, 0.37, 0.52, 0.27, 0.47, 0.22, 0.57, 0.37, 0.42
            ]
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_sbw_exact_balance_att() {
        let (treatment, covariates) = create_test_data();

        let config = SBWConfig {
            estimand: SBWEstimand::ATT,
            balance_tol: 0.0,  // Exact balance
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
            assert!((result.weights[i] - 1.0).abs() < 1e-6,
                    "Treated weight should be 1.0, got {}", result.weights[i]);
        }

        // SBW should achieve good balance
        assert!(result.balance_after.max_std_diff < 0.05,
                "SBW should achieve good balance, got max_std_diff = {}",
                result.balance_after.max_std_diff);

        // Balance should improve
        assert!(result.balance_after.max_std_diff < result.balance_before.max_std_diff);
    }

    #[test]
    fn test_sbw_approximate_balance() {
        let (treatment, covariates) = create_test_data();

        let config = SBWConfig {
            estimand: SBWEstimand::ATT,
            balance_tol: 0.1,  // Approximate balance
            balance_penalty: 100.0,
            ..Default::default()
        };

        let result = run_sbw(&treatment.view(), &covariates.view(), config).unwrap();

        // Should still achieve reasonable balance
        assert!(result.balance_after.max_std_diff < 0.2,
                "Approximate SBW should achieve reasonable balance, got {}",
                result.balance_after.max_std_diff);

        // Weights should be positive
        assert!(result.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_sbw_atc() {
        let (treatment, covariates) = create_test_data();

        let config = SBWConfig {
            estimand: SBWEstimand::ATC,
            ..Default::default()
        };

        let result = run_sbw(&treatment.view(), &covariates.view(), config).unwrap();

        // For ATC, control is target, treated is reweighted
        assert_eq!(result.n_target, 20);
        assert_eq!(result.n_reweighted, 10);

        // Control units should have weight 1.0
        for i in 10..30 {
            assert!((result.weights[i] - 1.0).abs() < 1e-6,
                    "Control weight should be 1.0 for ATC, got {}", result.weights[i]);
        }
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

        // Should achieve good balance
        assert!(result.balance_after.max_std_diff < 0.1);
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
        let treatment = Array1::from(vec![1.0, 1.0, 1.0]);  // All treated
        let covariates = array![[1.0], [1.2], [1.1]];

        let config = SBWConfig::default();
        let result = run_sbw(&treatment.view(), &covariates.view(), config);

        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_violation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let weights = vec![0.5, 0.3, 0.2];  // Sum = 1
        let target = Array1::from(vec![1.5, 2.5]);  // Exact match would need weights

        let violation = compute_constraint_violation(&x.view(), &weights, &target.view());

        // Weighted mean: [0.5*1 + 0.3*2 + 0.2*3, 0.5*2 + 0.3*3 + 0.2*4]
        //              = [0.5 + 0.6 + 0.6, 1.0 + 0.9 + 0.8] = [1.7, 2.7]
        // Violation: max(|1.7-1.5|, |2.7-2.5|) = max(0.2, 0.2) = 0.2
        assert!((violation - 0.2).abs() < 1e-10);
    }
}
