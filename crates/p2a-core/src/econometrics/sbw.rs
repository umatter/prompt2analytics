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
//! formed. The bound w >= l holds exactly and exact balance is reached to rounding; r is
//! accumulated with compensated (Neumaier) sums, so its rounding does not grow with n.
//!
//! Stopping rule, with tol = tolerance (1 + ||b||_inf):
//!
//! - Converged when ||r||_inf <= 1e-14 (1 + ||b||_inf), or ||r||_inf <= tol after a full
//!   step that kept S.
//! - If no ascent direction exists, the line search fails or max_iter steps are used up
//!   while ||r||_inf <= tol, the solve stops there as converged. Only outside that tolerance
//!   do these cases end it with an error.
//! - No ascent direction or a failed line search outside tol, and diverging multipliers
//!   (||mu||_inf > 1e10 (1 + ||mu_0||_inf)), mean that no feasible weights exist (the dual of
//!   an infeasible problem is unbounded): `EconError::InvalidSpecification` suggesting
//!   approximate balance.
//! - max_iter steps outside tol: `EconError::ConvergenceFailure` with
//!   `iterations = max_iter` and `last_change = ||r||_inf`, since running out of steps says
//!   nothing about feasibility or about the specification.
//! - A b, Gram matrix or multiplier that is not finite (magnitudes beyond f64):
//!   `EconError::Computation`, never a converged result.
//! - A (k+1) x (k+1) inversion that fails outright (a singular M gets the pseudo-inverse, so
//!   only a failed decomposition): `EconError::SingularMatrix` at the first step, the
//!   infeasibility error at later steps.
//!
//! Before the solve, with min_weight >= 0 a target mean outside the reweighted group's range
//! is refused, and in every configuration a target mean that overflows is refused.
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
//! ## Approximate Balance: Band QP by Proximal Newton
//!
//! With balance_tol > 0 each covariate mean must lie in a band around its target,
//! delta_j = balance_tol (`Raw`) or balance_tol sd_j (`StdDev`, sd_j the full-sample standard
//! deviation, Zubizarreta 2015):
//!
//! ```text
//! minimise (1/2) ||w||^2   subject to   (1/m) sum_i w_i = 1,
//!                                       |(1/m) sum_i w_i x_ij - t_j| <= delta_j,   w_i >= l
//! ```
//!
//! (m reweighted units, the n scale). With covariates centred on the band and scaled by
//! their group standard deviation, a_i = [xs_i, 1] / m and band half-widths r_j, the dual in
//! mu = (y, nu) (k + 1 variables) is
//!
//! ```text
//! F(mu) = sum_i h(a_i . mu) - sum_j r_j |y_j| + nu,      w_i(mu) = max(l, a_i . mu)
//! ```
//!
//! concave and piecewise quadratic plus an l1 term. -F is minimised by proximal Newton (Lee,
//! Sun & Saunders 2014): each step minimises the quadratic model plus the l1 term exactly by
//! the feature-sign method (Lee, Battle, Raina & Ng 2007), with an Armijo search; the
//! (k+1) x (k+1) Hessian sum_{i in S} a_i a_i' is regularised only when singular. Memory is
//! O(n k). The solve stops when the KKT residual is below 1e-14 (1 + max r_j), or below
//! tolerance (1 + max r_j) after an unregularised full step that kept S. When max_iter steps
//! are used up, a residual within tolerance (1 + max r_j) stops as converged (as the exact
//! solver rules); outside it the result is unconverged, with a note.
//!
//! Infeasibility: a band that misses a covariate's range (l >= 0), or a dual value above the
//! largest primal objective on {w >= l, sum w = m} (weak duality), is an
//! `EconError::InvalidSpecification` asking for a larger balance_tol.
//!
//! ## Estimands
//!
//! ATT reweights the controls to the treated means and ATC the treated to the control means
//! (the other group keeps weight 1); ATE solves one problem per group, each reweighted to the
//! pooled means.
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
use crate::data::columns::econ_error_from_design;
use crate::econometrics::weightit::{
    BalanceScale, SdDenominator, finite_abs_mean_max, group_mean_var,
};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{eig_symmetric, safe_inverse};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::options::Choices;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

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

crate::option_enum!(
    SBWEstimand,
    Choices {
        option: "estimand",
        values: &[
            (&["att"], SBWEstimand::ATT),
            (&["ate"], SBWEstimand::ATE),
            (&["atc"], SBWEstimand::ATC),
        ],
    },
    no_display
);

/// The unit in which [`SBWConfig::balance_tol`] is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BalanceTolScale {
    /// `balance_tol` in the covariates' own units.
    #[default]
    Raw,
    /// `balance_tol` in full-sample standard deviations of each covariate (Zubizarreta 2015;
    /// sd with the n - 1 denominator).
    StdDev,
}

crate::option_enum!(
    BalanceTolScale,
    Choices {
        option: "balance_tol_scale",
        values: &[
            (&["raw", "absolute"], BalanceTolScale::Raw),
            (
                &["sd", "std", "std_dev", "standardized", "standardised"],
                BalanceTolScale::StdDev,
            ),
        ],
    }
);

/// Configuration for stable balancing weights estimation.
#[derive(Debug, Clone)]
pub struct SBWConfig {
    /// Target estimand (ATT, ATE, ATC)
    pub estimand: SBWEstimand,
    /// Balance tolerance; must be finite and >= 0. `0` (the default) selects exact balance,
    /// solved by the dual Newton method of the module docs. A positive value selects
    /// approximate balance, |Σw_i X_ij / Σw_i - target_j| <= delta_j with delta_j = balance_tol
    /// in the unit of `balance_tol_scale`, solved exactly as a quadratic program.
    pub balance_tol: f64,
    /// Lower bound l on each reweighted unit's weight (default 0, non-negative weights);
    /// finite and at most 1. Both solvers hold w_i >= l exactly in the n scale (the weights
    /// average 1), and with l >= 0 a target mean (exact) or band (approximate) outside the
    /// reweighted group's range is refused.
    pub min_weight: f64,
    /// Normalize weights to sum to n (true) or 1 (false)
    pub normalize_to_n: bool,
    /// Iteration cap, at least 1. Approximate balance: proximal Newton steps; running out
    /// within `tolerance` stops as converged, otherwise it returns an unconverged result with
    /// a note. Exact balance: dual Newton steps, the
    /// closed-form first step counting as one; running out within `tolerance` stops as
    /// converged, otherwise it is an `EconError::ConvergenceFailure` with
    /// `iterations = max_iter`.
    pub max_iter: usize,
    /// Convergence tolerance; must be finite and > 0. Exact balance stops once the balance
    /// residual satisfies ||r||_inf <= tolerance (1 + ||b||_inf) after a full Newton step
    /// that kept the set of units off the bound (b = [target means; 1]; see the module docs
    /// for the other exits). Approximate balance stops once the KKT residual is below
    /// tolerance (1 + max r_j) after an unregularised full step that kept that set.
    pub tolerance: f64,
    /// Not used since T3; only the default 1000 is accepted. Approximate balance is solved
    /// exactly as a quadratic program, which has no penalty parameter; any other value
    /// (including a non-finite one) is an `EconError::InvalidSpecification`.
    pub balance_penalty: f64,
    /// Unit of `balance_tol`: `Raw` (the default, the covariates' own units) or `StdDev`
    /// (each covariate's full-sample standard deviation, n - 1 denominator).
    pub balance_tol_scale: BalanceTolScale,
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
            balance_tol_scale: BalanceTolScale::Raw,
        }
    }
}

/// Balance statistics showing covariate balance before and after weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceStats {
    /// Covariate names
    pub covariate_names: Vec<String>,
    /// Standardized mean differences for each covariate: (target - reweighted mean) / SD of the
    /// unweighted sample (treated SD for ATT, control SD for ATC, pooled for ATE; binary
    /// covariates p (1 - p); cobalt's s.d.denom)
    pub std_diff: Vec<f64>,
    /// Maximum absolute standardized difference over the finite values
    pub max_std_diff: f64,
    /// Mean absolute standardized difference over the finite values
    pub mean_std_diff: f64,
    /// Variance ratios (target / reweighted) of cobalt's weighted variances, NaN for binary
    /// covariates
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
    /// Number of solver steps (ATE: the larger of the two group solves)
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
    /// ESS of the treated group
    #[serde(default)]
    pub ess_treated: f64,
    /// ESS of the control group
    #[serde(default)]
    pub ess_control: f64,
    /// Objective `sum (w - 1/m)^2` (weights summing to 1) of the treated group's problem
    /// (ATC, ATE)
    #[serde(default)]
    pub objective_treated: Option<f64>,
    /// Objective of the control group's problem (ATT, ATE)
    #[serde(default)]
    pub objective_control: Option<f64>,
    /// ATE only: the treated problem's multipliers (`lambda` holds the control problem's)
    #[serde(default)]
    pub lambda_treated: Option<Vec<f64>>,
    /// Convergence record of the solver
    #[serde(default)]
    pub convergence: Option<Convergence>,
}

impl ResultNotes for SBWResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
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
            "  ESS:        treated {:.1}, control {:.1}",
            self.ess_treated, self.ess_control
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
/// ```
/// use ndarray::{Array1, array};
/// use p2a_core::econometrics::{SBWConfig, SBWEstimand, run_sbw};
///
/// // The treated means (1.1, 0.55) are a convex combination of the controls (weights 0.5,
/// // 0.5, 0), so exact balance (the default, balance_tol = 0) is feasible with non-negative
/// // weights. Lying inside each control column's range is not enough on its own, but a
/// // target mean outside that range is never feasible.
/// let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
/// let covariates = array![[1.0, 0.5], [1.2, 0.6], [0.9, 0.4], [1.3, 0.7], [1.0, 0.6]];
///
/// let config = SBWConfig {
///     estimand: SBWEstimand::ATT,
///     ..Default::default()
/// };
///
/// let result = run_sbw(&treatment.view(), &covariates.view(), config)?;
/// assert!(result.converged && result.max_constraint_violation < 1e-10);
/// println!("ESS: {:.1}", result.effective_sample_size);
/// # Ok::<(), p2a_core::EconError>(())
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
    if config.min_weight > 1.0 {
        return invalid(format!(
            "min_weight must be at most 1 (the mean weight in the n scale); got {}",
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
    if config.balance_penalty != 1000.0 {
        return invalid(
            "balance_penalty is not used: approximate balance is solved exactly as a quadratic program (Zubizarreta 2015), which has no penalty parameter; omit it".to_string(),
        );
    }
    Ok(())
}

/// [`run_sbw`] with the covariate names used in balance tables and error messages.
pub(crate) fn run_sbw_inner(
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

    let treated: Vec<usize> = (0..n).filter(|&i| treatment[i] >= 0.5).collect();
    let control: Vec<usize> = (0..n).filter(|&i| treatment[i] < 0.5).collect();
    let (nt, nc) = (treated.len(), control.len());
    if nt == 0 || nc == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Both groups must have observations. Found {nt} treated and {nc} control units."
            ),
        });
    }
    let x_t = extract_rows(covariates, &treated);
    let x_c = extract_rows(covariates, &control);

    // Finite covariates can still overflow when summed; an infinite target would make every
    // relative convergence threshold infinite.
    let means = |x: &ArrayView2<f64>, group: &str| -> EconResult<Array1<f64>> {
        let m = x
            .mean_axis(Axis(0))
            .ok_or_else(|| EconError::Computation("Failed to compute target means".to_string()))?;
        for (t, name) in m.iter().zip(cov_names) {
            if !t.is_finite() {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "the {group} mean of covariate {name} is not finite (its values are too large in magnitude to sum in f64); rescale the covariate"
                    ),
                });
            }
        }
        Ok(m)
    };

    // Problems: (reweighted group is the treated, its indices, its covariates, target means).
    let problems: Vec<(bool, &[usize], &Array2<f64>, Array1<f64>)> = match config.estimand {
        SBWEstimand::ATT => vec![(false, &control, &x_c, means(&x_t.view(), "treated")?)],
        SBWEstimand::ATC => vec![(true, &treated, &x_t, means(&x_c.view(), "control")?)],
        SBWEstimand::ATE => {
            let pooled = means(covariates, "pooled")?;
            vec![
                (true, &treated, &x_t, pooled.clone()),
                (false, &control, &x_c, pooled),
            ]
        }
    };

    let exact = config.balance_tol == 0.0;
    // Band half-widths in the covariates' units.
    let delta: Vec<f64> = (0..covariates.ncols())
        .map(|j| match config.balance_tol_scale {
            BalanceTolScale::Raw => config.balance_tol,
            BalanceTolScale::StdDev => {
                let col = covariates.column(j);
                let mean = col.sum() / n as f64;
                let var =
                    col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
                config.balance_tol * var.sqrt()
            }
        })
        .collect();

    let mut full_weights = vec![1.0; n];
    let mut group_weights: Vec<(bool, Vec<f64>)> = Vec::with_capacity(2);
    let mut lambda = Vec::new();
    let mut lambda_treated = None;
    let mut converged = true;
    let mut iterations = 0usize;
    let mut final_residual = 0.0_f64;
    let mut max_violation = 0.0_f64;
    let (mut objective_treated, mut objective_control) = (None, None);
    for (is_treated, idx, x_g, target) in &problems {
        if exact && config.min_weight >= 0.0 {
            // Exact balance with non-negative weights reweights to a convex combination of
            // the reweighted units, so every target mean must lie within that group's range.
            for (j, name) in cov_names.iter().enumerate() {
                let col = x_g.column(j);
                let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let t = target[j];
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
        let (w, lam, conv, it, obj, viol, resid) = if exact {
            solve_sbw_exact(&x_g.view(), &target.view(), &config)?
        } else {
            solve_sbw_approximate(&x_g.view(), &target.view(), &delta, cov_names, &config)?
        };
        // Solver weights sum to 1; normalize_to_n scales the group to its own size.
        let scale = if config.normalize_to_n {
            idx.len() as f64
        } else {
            1.0
        };
        let w: Vec<f64> = w.iter().map(|v| v * scale).collect();
        for (&i, &wi) in idx.iter().zip(&w) {
            full_weights[i] = wi;
        }
        converged &= conv;
        // ATE solves two groups: report the larger step count and residual (as
        // ebal_convergence does), so neither exceeds what one solve may use.
        iterations = iterations.max(it);
        final_residual = final_residual.max(resid);
        max_violation = max_violation.max(viol);
        if *is_treated {
            objective_treated = Some(obj);
            if config.estimand == SBWEstimand::ATE {
                lambda_treated = Some(lam);
            } else {
                lambda = lam;
            }
        } else {
            objective_control = Some(obj);
            lambda = lam;
        }
        group_weights.push((*is_treated, w));
    }
    let objective = objective_treated.unwrap_or(0.0) + objective_control.unwrap_or(0.0);
    let weights_of = |treated_group: bool| -> Option<&[f64]> {
        group_weights
            .iter()
            .find(|(t, _)| *t == treated_group)
            .map(|(_, w)| w.as_slice())
    };

    // Balance before and after weighting ("target" = the treated for ATT and ATE), with the
    // SMD denominators fixed from the unweighted sample (cobalt's s.d.denom).
    let scale = balance_scale(covariates, treatment, config.estimand);
    let (balance_before, balance_after) = match config.estimand {
        SBWEstimand::ATT => (
            compute_balance_stats(&x_t.view(), &x_c.view(), cov_names, None, &scale),
            compute_balance_stats(
                &x_t.view(),
                &x_c.view(),
                cov_names,
                weights_of(false),
                &scale,
            ),
        ),
        SBWEstimand::ATC => (
            compute_balance_stats(&x_c.view(), &x_t.view(), cov_names, None, &scale),
            compute_balance_stats(
                &x_c.view(),
                &x_t.view(),
                cov_names,
                weights_of(true),
                &scale,
            ),
        ),
        SBWEstimand::ATE => (
            compute_balance_stats_two(&x_t.view(), None, &x_c.view(), None, cov_names, &scale),
            compute_balance_stats_two(
                &x_t.view(),
                weights_of(true),
                &x_c.view(),
                weights_of(false),
                cov_names,
                &scale,
            ),
        ),
    };

    // Effective sample sizes
    let (ess_total, ess_treated, ess_control) = compute_ess(&full_weights, treatment);
    let (ess_target, ess_reweight, n_target, n_reweight) = match config.estimand {
        SBWEstimand::ATT => (ess_treated, ess_control, nt, nc),
        SBWEstimand::ATC => (ess_control, ess_treated, nc, nt),
        SBWEstimand::ATE => (ess_treated, ess_control, n, n),
    };
    let ess_reweighted_units = match config.estimand {
        SBWEstimand::ATE => ess_treated + ess_control,
        _ => ess_reweight,
    };

    // Weight statistics over the reweighted units
    let all_rw = group_weights.iter().flat_map(|(_, w)| w.iter().cloned());
    let max_weight = all_rw.clone().fold(f64::NEG_INFINITY, f64::max);
    let min_weight = all_rw.filter(|&w| w > 0.0).fold(f64::INFINITY, f64::min);

    // Check for warnings
    if max_weight / min_weight.max(1e-10) > 100.0 {
        warnings.push(format!(
            "High weight variability (max/min = {:.1}). Consider approximate balance.",
            max_weight / min_weight.max(1e-10)
        ));
    }

    if ess_reweighted_units < n_reweight as f64 * 0.3 {
        warnings.push(format!(
            "Low effective sample size ({:.1} of {}, {:.0}%). Weights are highly variable.",
            ess_reweighted_units,
            n_reweight,
            100.0 * ess_reweighted_units / n_reweight as f64
        ));
    }

    if max_violation > 0.01 {
        warnings.push(format!(
            "Balance constraints not fully satisfied (max violation = {:.4}).",
            max_violation
        ));
    }

    let convergence = Convergence::new(
        converged,
        iterations,
        config.max_iter,
        if exact {
            "dual Newton steps (exact QP)"
        } else {
            "proximal Newton steps (exact QP)"
        },
        final_residual,
        config.tolerance,
    );
    if let Some(note) = convergence.note(if exact {
        "Exact SBW"
    } else {
        "Approximate SBW"
    }) {
        warnings.push(note);
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
        ess_treated,
        ess_control,
        objective_treated,
        objective_control,
        lambda_treated,
        convergence: Some(convergence),
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
    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract treatment variable
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    // Build covariate matrix (without intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
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

/// Largest absolute entry, or infinity when any entry is not finite (folding with `f64::max`
/// alone would drop a NaN and let a NaN residual read as zero, i.e. converged).
fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, x| {
        if x.is_finite() {
            a.max(x.abs())
        } else {
            f64::INFINITY
        }
    })
}

/// One group's SBW solve: weights summing to 1, `lambda`, converged, steps, objective,
/// maximum balance violation (covariate units) and final relative residual.
type SbwSolve = (Vec<f64>, Vec<f64>, bool, usize, f64, f64, f64);

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
/// exactly, and r is accumulated with compensated sums.
///
/// If no ascent direction exists, the line search fails or `max_iter` steps (mu_0 counts
/// as one) are used up while ||r||_inf <= tolerance (1 + ||b||_inf), the solve stops there
/// as converged. Only outside that tolerance do these cases mean an error: no ascent
/// direction or a failed line search, like diverging multipliers, means no feasible weights
/// exist and is `EconError::InvalidSpecification` suggesting approximate balance; running
/// out of `max_iter` is an `EconError::ConvergenceFailure` with `iterations = max_iter` and
/// `last_change = ||r||_inf`. A non-finite b, Gram matrix or multiplier is
/// `EconError::Computation`. An inversion that fails outright is `EconError::SingularMatrix`
/// at the first step and the infeasibility error at later steps.
///
/// Returns weights summing to 1, `lambda = -mu` (the sign convention of the KKT system
/// `[I A'; A 0][w; lambda] = [0; b]`), convergence, the number of steps (mu_0 counts as
/// one), the objective, the maximum balance violation and the final relative residual
/// ||r||_inf / (1 + ||b||_inf), the quantity `tolerance` bounds.
fn solve_sbw_exact(
    x: &ArrayView2<f64>,
    target_means: &ArrayView1<f64>,
    config: &SBWConfig,
) -> EconResult<SbwSolve> {
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
    // Every convergence threshold is relative to b_scale and every test reads inf_norm, so
    // a non-finite b, Gram matrix or multiplier must stop the solve rather than pass a test.
    let overflow = || {
        EconError::Computation(
            "Exact balance failed numerically: the balance system is not finite (covariate or target-mean magnitudes too large for f64). Rescale the covariates.".to_string(),
        )
    };
    if !b_scale.is_finite() {
        return Err(overflow());
    }

    // First step: the Schur-complement closed form of the unbounded problem.
    let gram_all = dual.gram(None);
    if gram_all.iter().any(|v| !v.is_finite()) {
        return Err(overflow());
    }
    // Redundant balance constraints make the Gram matrix rank-deficient.
    // inverse-ok: the weights use mu only through X'mu, equal for every generalized-inverse solution
    let (m_all_inv, _) = safe_inverse(&gram_all.view()).map_err(|e| EconError::SingularMatrix {
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
    let mut residual = f64::NAN;

    loop {
        resource::checkpoint()?;
        // mu_0 and every update pass through here.
        if !mu.iter().all(|v| v.is_finite()) {
            return Err(overflow());
        }

        let (r, free) = dual.residual(&mu);
        let r_inf = inf_norm(&r);
        residual = r_inf / b_scale;
        let close = r_inf <= config.tolerance * b_scale;
        if r_inf <= SBW_EXACT_RESIDUAL * b_scale || (close && full_step_kept_set) {
            break;
        }
        if inf_norm(&mu) > SBW_DUAL_DIVERGENCE * mu0_scale {
            return Err(infeasible());
        }
        if iterations >= config.max_iter {
            // Out of steps, like the two exits below: a stop within tolerance is converged,
            // otherwise the solve is unfinished, which says nothing about feasibility.
            if close {
                break;
            }
            return Err(EconError::ConvergenceFailure {
                iterations: config.max_iter,
                last_change: r_inf,
                suggestion: format!(
                    "Exact balance residual {r_inf:.3e} is above the threshold tolerance (1 + ||b||_inf) = {:.3e} (tolerance = {:e}) after max_iter = {} steps; increase max_iter, or use approximate balance (balance_tol > 0) if exact balance may be infeasible.",
                    config.tolerance * b_scale,
                    config.tolerance,
                    config.max_iter
                ),
            });
        }

        // Newton direction on the current piece (pseudo-inverse when |S| < k + 1 or the
        // free units' covariates are collinear).
        // inverse-ok: a Newton direction of the dual ascent; the slope check below rejects a non-ascent one
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

    Ok((
        weights,
        lambda,
        true,
        iterations,
        objective,
        max_violation,
        residual,
    ))
}

/// Solve SBW with approximate balance exactly: the band QP of the module docs through its
/// (k+1)-dimensional dual, by proximal Newton with an exact feature-sign subproblem.
///
/// `delta` holds the band half-widths in the covariates' units. Returns weights summing to
/// 1, `lambda` (length k + 1, the sign and scale of [`solve_sbw_exact`]), convergence, the
/// number of steps, the objective `sum (w - 1/m)^2` and the largest band violation
/// `max_j max(|mean_j - t_j| - delta_j, 0)` in the covariates' units, and the final relative
/// KKT residual (`res / bscale`, the quantity `tolerance` bounds).
fn solve_sbw_approximate(
    x: &ArrayView2<f64>,
    target: &ArrayView1<f64>,
    delta: &[f64],
    names: &[String],
    config: &SBWConfig,
) -> EconResult<SbwSolve> {
    let (m, k) = x.dim();
    let l = config.min_weight;
    let mf = m as f64;
    if l > 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "min_weight must be at most 1 (the mean weight in the n scale); got {l}"
            ),
        });
    }

    // 1. Range pre-check; constant columns are fixed by the normalisation and dropped.
    let mut kept: Vec<usize> = Vec::with_capacity(k);
    for j in 0..k {
        let col = x.column(j);
        let (lo, hi) = (target[j] - delta[j], target[j] + delta[j]);
        let mn = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let constant = mn == mx;
        let slack = if constant {
            1e-12 * mn.abs().max(1.0)
        } else {
            0.0
        };
        if (l >= 0.0 || constant) && (hi < mn - slack || lo > mx + slack) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Approximate balance is infeasible: the band [{lo:.6}, {hi:.6}] of covariate {} does not meet the reweighted group's range [{mn:.6}, {mx:.6}]; increase balance_tol",
                    names[j]
                ),
            });
        }
        if !constant {
            kept.push(j);
        }
    }

    // 2. Standardise: centre on the band, scale by the group SD.
    let kk = kept.len();
    let kd = kk + 1;
    let mut cen = Vec::with_capacity(kk);
    let mut sd = Vec::with_capacity(kk);
    let mut r = vec![0.0; kd];
    for (c, &j) in kept.iter().enumerate() {
        let col = x.column(j);
        let mean = col.sum() / mf;
        let s_j =
            (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (mf - 1.0).max(1.0)).sqrt();
        cen.push(target[j]);
        sd.push(s_j);
        r[c] = delta[j] / s_j;
    }
    let mut a = resource::try_zeros2(m, kd, "SBW constraint rows")?;
    for i in 0..m {
        for (c, &j) in kept.iter().enumerate() {
            a[[i, c]] = (x[[i, j]] - cen[c]) / sd[c] / mf;
        }
        a[[i, kk]] = 1.0 / mf;
    }

    // Phi = -F, the minimised dual objective, from mu and z = A mu.
    let phi = |mu: &Array1<f64>, z: &Array1<f64>| -> f64 {
        let smooth: f64 = z
            .iter()
            .map(|&zi| {
                if zi >= l {
                    0.5 * zi * zi
                } else {
                    l * zi - 0.5 * l * l
                }
            })
            .sum();
        smooth + (0..kk).map(|c| r[c] * mu[c].abs()).sum::<f64>() - mu[kk]
    };
    let rho = |v: &Array1<f64>| -> f64 { (0..kk).map(|c| r[c] * v[c].abs()).sum::<f64>() - v[kk] };

    // 3. Start at uniform weights.
    let mut mu = Array1::<f64>::zeros(kd);
    mu[kk] = mf;
    let ceiling = ((mf - (mf - 1.0) * l).powi(2) + (mf - 1.0) * l * l) / 2.0;
    let bscale = 1.0 + r.iter().cloned().fold(0.0, f64::max);
    let mut it = 0usize;
    let mut converged = false;
    let mut residual = f64::NAN;
    let mut last_clean_step = false;
    let mut prev_free: Option<Vec<bool>> = None;

    // 4. Proximal Newton.
    loop {
        resource::checkpoint()?;
        let z = a.dot(&mu);
        let w = z.mapv(|zi| zi.max(l));
        let g = a.t().dot(&w);
        let free: Vec<bool> = z.iter().map(|&zi| zi > l).collect();
        let phi_mu = phi(&mu, &z);
        if !phi_mu.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err(EconError::Computation(
                "approximate SBW: the dual is not finite (covariate magnitudes too large for f64); rescale the covariates".to_string(),
            ));
        }
        if -phi_mu > ceiling * (1.0 + 1e-12) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Approximate balance is infeasible: no weights >= {l} on the {m} reweighted units keep every covariate mean within balance_tol of its target (dual objective {:.3e} > largest possible primal value {ceiling:.3e}); increase balance_tol",
                    -phi_mu
                ),
            });
        }
        // KKT residual (pseudo-gradient of F).
        let mut res = (1.0 - g[kk]).abs();
        for c in 0..kk {
            let gc = g[c];
            let pg = if mu[c] > 0.0 {
                -r[c] - gc
            } else if mu[c] < 0.0 {
                r[c] - gc
            } else if gc < -r[c] {
                -r[c] - gc
            } else if gc > r[c] {
                r[c] - gc
            } else {
                0.0
            };
            res = res.max(pg.abs());
        }
        residual = res / bscale;
        let set_kept = prev_free.as_ref().is_some_and(|p| *p == free);
        if res <= 1e-14 * bscale
            || (res <= config.tolerance * bscale && last_clean_step && set_kept)
        {
            converged = true;
            break;
        }
        if it == config.max_iter {
            // Running out of steps within tolerance is a stop within tolerance: converged, as
            // the exact solver rules.
            converged = res <= config.tolerance * bscale;
            break;
        }

        // Hessian of the smooth part on the free units, regularised only when singular.
        let mut hm = Array2::<f64>::zeros((kd, kd));
        for i in (0..m).filter(|&i| free[i]) {
            for p in 0..kd {
                let ap = a[[i, p]];
                for q in 0..=p {
                    hm[[p, q]] += ap * a[[i, q]];
                }
            }
        }
        for p in 0..kd {
            for q in 0..p {
                hm[[q, p]] = hm[[p, q]];
            }
        }
        let (ev, _) = eig_symmetric(&hm.view()).map_err(|e| {
            EconError::Computation(format!("approximate SBW: eigendecomposition failed: {e}"))
        })?;
        let lmax = ev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lmin = ev.iter().cloned().fold(f64::INFINITY, f64::min);
        let regular = lmax > 0.0 && lmin > 1e-12 * lmax;
        if !regular {
            let base = if lmax > 0.0 { lmax } else { 1.0 / mf };
            let ridge = base * res.min(1e-8);
            for p in 0..kd {
                hm[[p, p]] += ridge;
            }
        }

        let mut b = hm.dot(&mu) - &g;
        b[kk] += 1.0;
        let zstar = feature_sign(&hm, &b, &r, &mu)?;
        let d = &zstar - &mu;
        let pred = g.dot(&d) + rho(&zstar) - rho(&mu);
        let mut t = 1.0;
        if -pred > 1e-12 * (1.0 + phi_mu.abs()) {
            loop {
                let cand = &mu + &(&d * t);
                let zc = a.dot(&cand);
                if phi(&cand, &zc) <= phi_mu + 1e-4 * t * pred || t <= 1e-12 {
                    break;
                }
                t *= 0.5;
            }
            if t <= 1e-12 {
                if res <= config.tolerance * bscale {
                    converged = true;
                    break;
                }
                return Err(EconError::Computation(format!(
                    "approximate SBW: the dual line search made no progress (residual {res:.3e})"
                )));
            }
        }
        mu = if t == 1.0 { zstar } else { &mu + &(&d * t) };
        last_clean_step = regular && t == 1.0;
        prev_free = Some(free);
        it += 1;
    }

    // 6. Weights, multipliers and diagnostics.
    let z = a.dot(&mu);
    let mut weights: Vec<f64> = z.iter().map(|&zi| zi.max(l)).collect();
    let w_sum: f64 = weights.iter().sum();
    if !(w_sum.is_finite() && w_sum > 0.0) {
        return Err(EconError::Computation(
            "approximate SBW produced weights that do not sum to a positive number".to_string(),
        ));
    }
    for w in &mut weights {
        *w /= w_sum;
    }
    let mut lambda = vec![0.0; k + 1];
    let mut shift = 0.0;
    for (c, &j) in kept.iter().enumerate() {
        lambda[j] = -mu[c] / sd[c];
        shift += cen[c] * mu[c] / sd[c];
    }
    lambda[k] = -(mu[kk] - shift);
    let objective: f64 = weights.iter().map(|w| (w - 1.0 / mf).powi(2)).sum();
    let mut max_violation = 0.0_f64;
    for j in 0..k {
        let mean: f64 = (0..m).map(|i| weights[i] * x[[i, j]]).sum();
        max_violation = max_violation.max((mean - target[j]).abs() - delta[j]);
    }
    Ok((
        weights,
        lambda,
        converged,
        it,
        objective,
        max_violation,
        residual,
    ))
}

/// Sign of `v` with sign(0) = 0.
fn sign0(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Exact minimiser of `1/2 z'Hz - b'z + sum_j r_j |z_j|` (H positive definite) by the
/// feature-sign method of Lee, Battle, Raina & Ng (2007), started from `z0`. Coordinates with
/// `r_j = 0` are always active.
fn feature_sign(
    h: &Array2<f64>,
    b: &Array1<f64>,
    r: &[f64],
    z0: &Array1<f64>,
) -> EconResult<Array1<f64>> {
    let kd = b.len();
    let kink: Vec<bool> = r.iter().map(|&rj| rj > 0.0).collect();
    let mut z = z0.clone();
    let mut active: Vec<bool> = (0..kd).map(|j| !kink[j] || z[j] != 0.0).collect();
    let mut theta: Vec<f64> = z.iter().map(|&v| sign0(v)).collect();
    let q = |v: &Array1<f64>| -> f64 {
        0.5 * v.dot(&h.dot(v)) - b.dot(v) + (0..kd).map(|j| r[j] * v[j].abs()).sum::<f64>()
    };
    let b_scale = 1.0 + b.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for _ in 0..(1000 + 10 * kd) {
        let idx: Vec<usize> = (0..kd).filter(|&j| active[j]).collect();
        let h_aa = h.select(Axis(0), &idx).select(Axis(1), &idx);
        let rhs = Array1::from_shape_fn(idx.len(), |p| {
            let j = idx[p];
            b[j] - r[j] * theta[j]
        });
        let inv = sym_inverse(h_aa.view(), SingularPolicy::Error, "SBW Newton system")?.inverse;
        let z_a = inv.dot(&rhs);
        let mut z_new = Array1::<f64>::zeros(kd);
        for (p, &j) in idx.iter().enumerate() {
            z_new[j] = z_a[p];
        }
        // Discrete line search between z and z_new at the sign changes.
        let mut best = z_new.clone();
        let mut best_q = q(&z_new);
        let mut chose_new = true;
        for &j in &idx {
            if kink[j] && z[j] != 0.0 && sign0(z_new[j]) != sign0(z[j]) {
                let s = z[j] / (z[j] - z_new[j]);
                let mut c = &z + &((&z_new - &z) * s);
                c[j] = 0.0;
                let qc = q(&c);
                if qc < best_q {
                    best = c;
                    best_q = qc;
                    chose_new = false;
                }
            }
        }
        z = best;
        for j in 0..kd {
            if kink[j] {
                active[j] = z[j] != 0.0;
                theta[j] = sign0(z[j]);
            }
        }
        if !chose_new {
            continue;
        }
        // Optimality of the inactive kink coordinates.
        let g = h.dot(&z) - b;
        let mut worst: Option<(usize, f64)> = None;
        for j in 0..kd {
            if kink[j] && !active[j] {
                let viol = g[j].abs() - r[j];
                if worst.is_none_or(|(_, v)| viol > v) {
                    worst = Some((j, viol));
                }
            }
        }
        match worst {
            Some((j, viol)) if viol > 1e-13 * b_scale => {
                active[j] = true;
                theta[j] = -sign0(g[j]);
            }
            _ => return Ok(z),
        }
    }
    Err(EconError::Computation(
        "approximate SBW: the feature-sign subproblem did not terminate".to_string(),
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

/// Per-covariate SMD denominators from the unweighted sample (treated SD for ATT, control SD
/// for ATC, pooled for ATE; binary covariates `p (1 - p)`, cobalt's s.d.denom) and which
/// covariates are binary (their variance ratio is NaN).
fn balance_scale(
    covariates: &ArrayView2<f64>,
    treatment: &ArrayView1<f64>,
    est: SBWEstimand,
) -> BalanceScale {
    let treated: Vec<bool> = treatment.iter().map(|&v| v >= 0.5).collect();
    let denom = match est {
        SBWEstimand::ATT => SdDenominator::Treated,
        SBWEstimand::ATC => SdDenominator::Control,
        SBWEstimand::ATE => SdDenominator::Pooled,
    };
    BalanceScale::new(covariates.view(), &treated, denom)
}

/// Compute balance statistics of a reweighted group against an unweighted target group.
fn compute_balance_stats(
    x_target: &ArrayView2<f64>,
    x_reweight: &ArrayView2<f64>,
    cov_names: &[String],
    weights: Option<&[f64]>,
    scale: &BalanceScale,
) -> BalanceStats {
    compute_balance_stats_two(x_target, None, x_reweight, weights, cov_names, scale)
}

/// Weighted mean of a column (unit weights when none are given) and cobalt's weighted variance.
fn column_moments(col: ArrayView1<f64>, w: Option<&[f64]>) -> (f64, f64) {
    group_mean_var(col, w, |_| true)
}

/// Balance statistics of group a (reported as "target") against group b, each optionally
/// weighted: the SMD over the fixed denominators of `scale`, the ratio of cobalt's weighted
/// variances (NaN for binary covariates), mean and max over the finite |SMD|.
fn compute_balance_stats_two(
    x_a: &ArrayView2<f64>,
    w_a: Option<&[f64]>,
    x_b: &ArrayView2<f64>,
    w_b: Option<&[f64]>,
    cov_names: &[String],
    scale: &BalanceScale,
) -> BalanceStats {
    let k = x_a.ncols();
    let mut std_diff = Vec::with_capacity(k);
    let mut var_ratio = Vec::with_capacity(k);
    let mut mean_target = Vec::with_capacity(k);
    let mut mean_reweighted = Vec::with_capacity(k);

    for j in 0..k {
        let (target_mean, target_var) = column_moments(x_a.column(j), w_a);
        let (reweight_mean, reweight_var) = column_moments(x_b.column(j), w_b);

        std_diff.push((target_mean - reweight_mean) / scale.sd[j]);
        var_ratio.push(if scale.binary[j] {
            f64::NAN
        } else {
            target_var / reweight_var
        });
        mean_target.push(target_mean);
        mean_reweighted.push(reweight_mean);
    }

    let (mean_std_diff, max_std_diff) = finite_abs_mean_max(std_diff.iter().copied());

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

    fn spec_message(err: EconError) -> String {
        match err {
            EconError::InvalidSpecification { message } => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_sbw_estimand_parses_aliases_and_rejects_unknown() {
        SBWEstimand::choices().validate().unwrap();
        for (s, e) in [
            ("ATE", SBWEstimand::ATE),
            ("att", SBWEstimand::ATT),
            ("Atc", SBWEstimand::ATC),
        ] {
            assert_eq!(s.parse::<SBWEstimand>().unwrap(), e, "{s}");
        }
        let m = spec_message("late".parse::<SBWEstimand>().unwrap_err());
        assert!(m.contains("estimand") && m.contains("att, ate, atc"), "{m}");
    }

    #[test]
    fn test_balance_tol_scale_parses_aliases_and_rejects_unknown() {
        BalanceTolScale::choices().validate().unwrap();
        for (s, e) in [
            ("raw", BalanceTolScale::Raw),
            ("absolute", BalanceTolScale::Raw),
            ("sd", BalanceTolScale::StdDev),
            ("Standardized", BalanceTolScale::StdDev),
            ("std-dev", BalanceTolScale::StdDev),
        ] {
            assert_eq!(s.parse::<BalanceTolScale>().unwrap(), e, "{s}");
        }
        let m = spec_message("percent".parse::<BalanceTolScale>().unwrap_err());
        assert!(
            m.contains("balance_tol_scale") && m.contains("raw, sd"),
            "{m}"
        );
        assert_eq!(BalanceTolScale::StdDev.to_string(), "sd");
        assert_eq!(SBWConfig::default().balance_tol_scale, BalanceTolScale::Raw);
    }

    use p2a_validation::{Check, Tol};

    const COVS: [&str; 3] = ["x1", "x2", "x3"];

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        DesignMatrix::extract_column(ds.df(), name)
            .unwrap()
            .to_vec()
    }

    /// The entries of `v` whose unit is in the treated (true) or control (false) group.
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

    /// `weighting_sim.csv` for the property tests (they pin no R value).
    fn weighting_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/weighting_sim.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// The case is loaded in each test body.
    fn sbw_fit(ds: &Dataset, est: SBWEstimand, tol: f64) -> SBWResult {
        let mut c = SBWConfig::default();
        c.estimand = est;
        c.balance_tol = tol;
        c.balance_tol_scale = BalanceTolScale::StdDev;
        c.normalize_to_n = false;
        sbw(ds, "treat", &COVS, c).unwrap()
    }

    #[test]
    fn test_validate_sbw_exact() {
        // R: quadprog::solve.QP on the SBW QP (case weights_sbw). CAUS-79's clipping was
        // removed by the dual Newton solver; this pins it to R.
        let (case, ds) = crate::test_support::load_case("weights_sbw");
        let r = sbw_fit(&ds, SBWEstimand::ATT, 0.0);
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let wc = group(&r.weights, &d, false);
        let n = d.len();
        Check::new(&case)
            .vector("att_exact_weights_head", &wc[..12], Tol::new(1e-8, 1e-12))
            .vector(
                "att_exact_weighted_means",
                &r.balance_after.mean_reweighted,
                Tol::coef(n),
            )
            .scalar("att_exact_objective", r.objective_value, Tol::coef(n))
            .scalar("att_exact_n_zero", n_zero(&wc), Tol::exact())
            .scalar("att_exact_effect", effect(&y, &d, &r.weights), Tol::coef(n))
            .finish();
        assert!(r.converged && r.max_constraint_violation < 1e-10);
    }

    #[test]
    fn test_validate_sbw_approx() {
        // R: quadprog::solve.QP on the band QP, delta = 0.1 sd (case weights_sbw). Fails
        // before (CAUS-79): projected gradient with clipping.
        let (case, ds) = crate::test_support::load_case("weights_sbw");
        let r = sbw_fit(&ds, SBWEstimand::ATT, 0.1);
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let wc = group(&r.weights, &d, false);
        let n = d.len();
        Check::new(&case)
            .vector("att_approx_weights_head", &wc[..12], Tol::new(1e-8, 1e-12))
            .vector(
                "att_approx_weighted_means",
                &r.balance_after.mean_reweighted,
                Tol::coef(n),
            )
            .scalar("att_approx_objective", r.objective_value, Tol::coef(n))
            .scalar("att_approx_n_zero", n_zero(&wc), Tol::exact())
            .scalar(
                "att_approx_effect",
                effect(&y, &d, &r.weights),
                Tol::coef(n),
            )
            .finish();
        assert!(r.converged);
        let sd = case.vector("sd_all");
        for (j, c) in COVS.iter().enumerate() {
            let treated_mean = group(&col(&ds, c), &d, true).iter().sum::<f64>() / 151.0;
            let gap = (r.balance_after.mean_reweighted[j] - treated_mean).abs();
            assert!(gap <= 0.1 * sd[j] + 1e-12, "{c}: {gap}");
        }
    }

    #[test]
    fn test_validate_sbw_atc() {
        // R: quadprog::solve.QP, the treated reweighted to the control means (case weights_sbw)
        let (case, ds) = crate::test_support::load_case("weights_sbw");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let n = d.len();
        for (tag, tol) in [("exact", 0.0), ("approx", 0.1)] {
            let r = sbw_fit(&ds, SBWEstimand::ATC, tol);
            let wt = group(&r.weights, &d, true);
            assert!(r.converged, "{tag}");
            Check::new(&case)
                .vector(
                    &format!("atc_{tag}_weights_head"),
                    &wt[..12],
                    Tol::new(1e-8, 1e-12),
                )
                .scalar(
                    &format!("atc_{tag}_objective"),
                    r.objective_value,
                    Tol::coef(n),
                )
                .scalar(&format!("atc_{tag}_n_zero"), n_zero(&wt), Tol::exact())
                .scalar(
                    &format!("atc_{tag}_effect"),
                    effect(&y, &d, &r.weights),
                    Tol::coef(n),
                )
                .finish();
        }
    }

    #[test]
    fn test_validate_sbw_balance_fixed_sd() {
        // R: cobalt s.d.denom on weighting_sim (case weights_trim_balance): treated SD for ATT,
        // control SD for ATC, pooled for ATE, binary covariates p (1 - p). CAUS-108: the SD was
        // pooled from the weighted groups, so weighting changed the yardstick.
        let (case, ds) = crate::test_support::load_case("weights_trim_balance");
        for (est, key) in [
            (SBWEstimand::ATT, "att_sd_denom"),
            (SBWEstimand::ATC, "atc_sd_denom"),
            (SBWEstimand::ATE, "ate_sd_denom"),
        ] {
            let r = sbw_fit(&ds, est, 0.0);
            let n = r.weights.len();
            let sd = case.vector(key);
            let b = &r.balance_before;
            for j in 0..3 {
                let (p, d) = (
                    b.std_diff[j] * sd[j],
                    b.mean_target[j] - b.mean_reweighted[j],
                );
                assert!(Tol::coef(n).allows(p, d), "{est:?} before {j}: {p} vs {d}");
            }
            let a = &r.balance_after;
            for j in 0..3 {
                let (p, d) = (
                    a.std_diff[j] * sd[j],
                    a.mean_target[j] - a.mean_reweighted[j],
                );
                assert!((p - d).abs() <= 1e-12, "{est:?} after {j}: {p} vs {d}");
                assert!(
                    a.std_diff[j].abs() < 1e-10,
                    "{est:?} after {j}: exact balance"
                );
            }
            assert!(
                b.var_ratio[2].is_nan() && a.var_ratio[2].is_nan(),
                "binary x3"
            );
        }
    }

    #[test]
    fn test_sbw_ate_convergence_record_is_per_solve() {
        // ATE solves the two groups separately: the record reports the larger step count
        // (it summed them, exceeding max_iter) and the solver's relative residual, the
        // quantity `tolerance` bounds (it reported the balance violation in covariate units).
        let ds = weighting_sim();
        for max_iter in [1, 2, 3] {
            let mut c = SBWConfig::default();
            c.estimand = SBWEstimand::ATE;
            c.balance_tol = 0.1;
            c.balance_tol_scale = BalanceTolScale::StdDev;
            c.max_iter = max_iter;
            let r = sbw(&ds, "treat", &COVS, c).unwrap();
            let conv = r.convergence.as_ref().unwrap();
            assert!(
                r.iterations <= max_iter && conv.iterations <= conv.max_iterations,
                "max_iter {max_iter}: iterations {} (record {})",
                r.iterations,
                conv.iterations
            );
            // A stop within tolerance is converged, so an unconverged record lies above it.
            assert_eq!(r.converged, conv.converged, "max_iter {max_iter}");
            assert!(
                r.converged || conv.final_change > conv.tolerance,
                "max_iter {max_iter}: unconverged residual {} within tolerance {}",
                conv.final_change,
                conv.tolerance
            );
            assert!(conv.final_change.is_finite() && conv.final_change >= 0.0);
        }
        for tol in [0.0, 0.1] {
            let r = sbw_fit(&ds, SBWEstimand::ATE, tol);
            let conv = r.convergence.as_ref().unwrap();
            assert!(r.converged && r.iterations <= conv.max_iterations);
            assert!(
                conv.final_change.is_finite()
                    && conv.final_change >= 0.0
                    && conv.final_change <= conv.tolerance,
                "tol {tol}: final relative residual {} vs tolerance {}",
                conv.final_change,
                conv.tolerance
            );
        }
    }

    #[test]
    fn test_validate_sbw_ate() {
        // R: quadprog::solve.QP once per group with the pooled means as target (case
        // weights_sbw). Fails before (CAUS-49): ATE ran as ATT.
        let (case, ds) = crate::test_support::load_case("weights_sbw");
        let (d, y) = (col(&ds, "treat"), col(&ds, "y"));
        let n = d.len();
        for (tag, tol) in [("exact", 0.0), ("approx", 0.1)] {
            let r = sbw_fit(&ds, SBWEstimand::ATE, tol);
            let (wt, wc) = (group(&r.weights, &d, true), group(&r.weights, &d, false));
            assert!(r.converged, "{tag}");
            Check::new(&case)
                .vector(
                    &format!("ate_{tag}_weights_t_head"),
                    &wt[..8],
                    Tol::new(1e-8, 1e-12),
                )
                .vector(
                    &format!("ate_{tag}_weights_c_head"),
                    &wc[..8],
                    Tol::new(1e-8, 1e-12),
                )
                .scalar(
                    &format!("ate_{tag}_objective_t"),
                    r.objective_treated.unwrap(),
                    Tol::coef(n),
                )
                .scalar(
                    &format!("ate_{tag}_objective_c"),
                    r.objective_control.unwrap(),
                    Tol::coef(n),
                )
                .scalar(
                    &format!("ate_{tag}_effect"),
                    effect(&y, &d, &r.weights),
                    Tol::coef(n),
                )
                .finish();
            assert_eq!(
                r.objective_value,
                r.objective_treated.unwrap() + r.objective_control.unwrap()
            );
            assert!(r.n_target == 300 && r.n_reweighted == 300);
        }
    }

    #[test]
    fn test_sbw_estimands_differ() {
        // Tier 3 convention 2; fails before: ATE returned the ATT weights
        let ds = weighting_sim();
        let att = sbw_fit(&ds, SBWEstimand::ATT, 0.0);
        let ate = sbw_fit(&ds, SBWEstimand::ATE, 0.0);
        assert!(
            att.weights
                .iter()
                .zip(&ate.weights)
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    #[test]
    fn test_sbw_approx_matches_quadprog_on_test_data() {
        // R: quadprog::solve.QP of the band QP on create_test_data(), ATT, balance_tol = 0.05
        // raw, w >= 0; the x2 lower bound is active.
        const W: [f64; 20] = [
            0.8992068443097758,
            0.9571339452811691,
            0.9281703947954725,
            1.1888423491667408,
            0.8992068443097762,
            1.130915248195348,
            0.8702432938240796,
            1.2467694501381341,
            0.9281703947954727,
            0.9571339452811691,
            0.8992068443097762,
            1.015061046252562,
            0.9455485250868902,
            1.1598787986810446,
            0.8876214241154976,
            1.1019516977096515,
            0.8528651635326617,
            1.2178058996524375,
            0.9281703947954728,
            0.9860974957668656,
        ];
        let (t, x) = create_test_data();
        let mut c = SBWConfig::default();
        c.balance_tol = 0.05;
        let r = run_sbw(&t.view(), &x.view(), c).unwrap();
        assert!(r.converged && r.max_constraint_violation <= 1e-12);
        for (j, w) in W.iter().enumerate() {
            assert!((r.weights[10 + j] - w).abs() < 1e-10, "control {j}")
        }
        assert!((r.objective_value - 0.0007530523126281094).abs() < 1e-14);
        assert!((r.balance_after.mean_reweighted[1] - 0.5).abs() < 1e-12);
        let (sw, swy) = (10..30).fold((0.0, 0.0), |(a, b), i| {
            (a + r.weights[i], b + r.weights[i] * SBW_Y[i])
        });
        let att = SBW_Y[..10].iter().sum::<f64>() / 10.0 - swy / sw;
        assert!((att - 0.5453423937260495).abs() < 1e-12);
    }

    #[test]
    fn test_sbw_approx_infeasible_is_error() {
        // (a) the band [t - delta, t + delta] misses the controls' range of x: the range
        //     pre-check
        let t = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let x = array![[3.0], [3.2], [2.9], [0.0], [0.3], [0.6], [1.0]];
        let mut c = SBWConfig::default();
        c.balance_tol = 0.1;
        expect_invalid_spec(
            run_sbw(&t.view(), &x.view(), c.clone()).unwrap_err(),
            "balance_tol",
        );
        // (b) each band meets its column's range, but not the controls' convex hull
        //     (x1 + x2 <= 1): the dual certificate (R: solve.QP "constraints are
        //     inconsistent")
        let t2 = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let x2 = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.2],
            [0.3, 0.1],
            [0.1, 0.3],
            [0.9, 0.9],
            [0.8, 1.0],
            [1.0, 0.8]
        ];
        let mut c2 = SBWConfig::default();
        c2.balance_tol = 0.05;
        expect_invalid_spec(
            run_sbw(&t2.view(), &x2.view(), c2).unwrap_err(),
            "balance_tol",
        );
    }

    #[test]
    fn test_sbw_approx_scale_invariant() {
        // convention 5: data x 1e-6 and x 1e6 with balance_tol in SD units give the same
        // weights in similar steps
        let ds = weighting_sim();
        let base = sbw_fit(&ds, SBWEstimand::ATT, 0.1);
        for s in [1e-6, 1e6] {
            let df = df! { "treat" => col(&ds, "treat"), "x1" => scaled(&col(&ds, "x1"), s),
            "x2" => scaled(&col(&ds, "x2"), s), "x3" => scaled(&col(&ds, "x3"), s) }
            .unwrap();
            let r = sbw_fit(&Dataset::new(df), SBWEstimand::ATT, 0.1);
            assert!(
                (r.iterations as i64 - base.iterations as i64).abs() <= 1,
                "scale {s}"
            );
            for (a, b) in r.weights.iter().zip(&base.weights) {
                assert!((a - b).abs() <= 1e-10 * b.abs().max(1e-3))
            }
        }
    }

    #[test]
    fn test_sbw_balance_penalty_is_refused() {
        // The exact band QP has no penalty parameter: a value other than the default is an
        // error naming the field. Fails before: the penalty steered the projected gradient.
        let (t, x) = create_test_data();
        let mut c = SBWConfig::default();
        c.balance_tol = 0.05;
        c.balance_penalty = 5.0;
        expect_invalid_spec(
            run_sbw(&t.view(), &x.view(), c).unwrap_err(),
            "balance_penalty",
        );
        let mut d = SBWConfig::default();
        d.balance_tol = 0.05;
        let r0 = run_sbw(&t.view(), &x.view(), d).unwrap();
        assert!(!r0.warnings.iter().any(|w| w.contains("balance_penalty")));
    }

    /// Outcome for create_test_data(): y = 1 + 2 x1 - x2 + 0.5 d + e (e fixed, 2 decimals).
    const SBW_Y: [f64; 30] = [
        2.32, 2.22, 2.30, 2.29, 2.29, 2.37, 2.04, 2.55, 2.22, 2.27, 1.21, 1.64, 1.56, 2.15, 1.48,
        1.98, 1.38, 2.56, 1.38, 1.61, 1.50, 1.68, 1.42, 2.20, 1.45, 2.01, 1.41, 2.27, 1.51, 1.59,
    ];

    #[test]
    fn test_sbw_exact_matches_r_sbw() {
        // R reference (R 4.3.3, quadprog 1.5.8; the sbw call's output as recorded with sbw 1.2):
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
                0.08913741772994399,
                0.36493223533034325,
                0.07896713213995099
            ]
        );
        let (w_ref, lambda_ref) = dense_kkt_reference(&xc.view(), &target.view());
        assert!(
            w_ref.iter().all(|&w| w > 0.0),
            "precondition: no bound binds"
        );
        let (w, lambda, converged, iterations, _, viol, _) =
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
            let (w, lambda, converged, _, _, viol, _) =
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
    fn test_sbw_exact_max_iter_exhaustion_is_not_infeasibility() {
        // create_test_data() is feasible and takes five dual steps (mu_0 counts as one): mu_0,
        // three line-search steps (the bound binds, so S changes) and a full step that keeps
        // S. ||r||_inf at steps 1-4 is 0.20, 0.10, 0.029, 0.0023. Running out of steps is not
        // evidence of infeasibility.
        // It is a convergence failure, and its message shows both the absolute threshold
        // tolerance (1 + ||b||_inf), here 1e-3 (1 + 1), and the configured tolerance.
        let (t, x) = create_test_data();
        let err = run_sbw(
            &t.view(),
            &x.view(),
            SBWConfig {
                max_iter: 2,
                tolerance: 1e-3,
                ..Default::default()
            },
        )
        .unwrap_err();
        match &err {
            EconError::ConvergenceFailure {
                iterations: 2,
                last_change,
                suggestion,
            } => {
                assert!(
                    suggestion.contains("max_iter = 2")
                        && !suggestion.contains("could not be reached"),
                    "max_iter exhaustion reported as `{suggestion}`"
                );
                assert!(
                    suggestion.contains("(1 + ||b||_inf) = 2.000e-3")
                        && suggestion.contains("tolerance = 1e-3"),
                    "threshold and configured tolerance not both shown in `{suggestion}`"
                );
                assert!(
                    last_change.is_finite() && *last_change > 2e-3,
                    "last_change {last_change} is not the residual outside tolerance"
                );
                assert!(
                    suggestion.contains(&format!("residual {last_change:.3e}")),
                    "last_change {last_change} is not the residual reported in `{suggestion}`"
                );
            }
            other => panic!("expected ConvergenceFailure after 2 iterations, got {other:?}"),
        }
        // With tolerance 1e-2 the fourth step is within tolerance (0.0023 <= 1e-2 (1 + 1)),
        // so running out of steps there stops as converged, as the other two exits do.
        let r = run_sbw(
            &t.view(),
            &x.view(),
            SBWConfig {
                max_iter: 4,
                tolerance: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.converged && r.iterations == 4,
            "converged {}, iterations {}",
            r.converged,
            r.iterations
        );
        assert!(
            r.max_constraint_violation < 1e-2,
            "violation {}",
            r.max_constraint_violation
        );
        // With the default max_iter the same tolerance stops after the full step (five
        // steps), so the max_iter = 4 run above stopped at the max_iter exit.
        let r = run_sbw(
            &t.view(),
            &x.view(),
            SBWConfig {
                tolerance: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.converged && r.iterations == 5,
            "converged {}, iterations {}",
            r.converged,
            r.iterations
        );
    }

    fn expect_computation(err: EconError, needle: &str) {
        match err {
            EconError::Computation(message) => assert!(
                message.contains(needle),
                "message `{message}` does not mention `{needle}`"
            ),
            other => panic!("expected Computation mentioning `{needle}`, got {other:?}"),
        }
    }

    #[test]
    fn test_inf_norm_is_infinite_on_non_finite_entries() {
        // folding with f64::max drops NaN, so a NaN residual used to read as 0 (converged)
        assert_eq!(inf_norm(&array![1.0, f64::NAN, -2.0]), f64::INFINITY);
        assert_eq!(inf_norm(&array![1.0, f64::NEG_INFINITY]), f64::INFINITY);
        assert_eq!(inf_norm(&array![1.0, -3.0, 2.0]), 3.0);
    }

    #[test]
    fn test_sbw_exact_overflow_is_an_error_not_a_converged_fallback() {
        // min_weight < 0 skips the range pre-check, so these reach the dual solver.
        let (t, x) = create_test_data();
        let negative = || SBWConfig {
            min_weight: -0.5,
            ..Default::default()
        };
        // Two finite treated values whose sum overflows: the target mean of X1 is +inf. With
        // min_weight < 0 the exact path returned Ok with converged = true, uniform weights
        // and an infinite violation, and the approximate path Ok with an infinite violation.
        let mut x_sum = x.clone();
        x_sum[[0, 0]] = 1.5e308;
        x_sum[[1, 0]] = 1.6e308;
        for cfg in [
            negative(),
            SBWConfig::default(),
            SBWConfig {
                balance_tol: 0.1,
                ..negative()
            },
        ] {
            expect_invalid_spec(
                run_sbw(&t.view(), &x_sum.view(), cfg).unwrap_err(),
                "covariate X1",
            );
        }
        // The solver itself refuses a non-finite target; it used to return converged = true
        // after one step with an infinite balance violation.
        let x_c = x.slice(ndarray::s![10.., ..]).to_owned();
        let target = array![f64::INFINITY, 0.55];
        expect_computation(
            solve_sbw_exact(&x_c.view(), &target.view(), &negative()).unwrap_err(),
            "magnitude",
        );
        // A finite target mean of 1.7e307 overflows the multipliers of the first step (the
        // old code reached its infeasibility error only because the next, NaN, Gram matrix
        // failed to invert).
        let mut x_big = x.clone();
        x_big[[0, 0]] = 1.7e308;
        expect_computation(
            run_sbw(&t.view(), &x_big.view(), negative()).unwrap_err(),
            "magnitude",
        );
        // Covariates of order 1e200 overflow the (k+1) x (k+1) Gram matrix.
        let x_scaled = x.mapv(|v| v * 1e200);
        expect_computation(
            run_sbw(&t.view(), &x_scaled.view(), SBWConfig::default()).unwrap_err(),
            "magnitude",
        );
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
            (
                SBWConfig {
                    balance_tol: 0.1,
                    min_weight: 1.5,
                    ..d()
                },
                "min_weight",
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

        let x_all = ndarray::concatenate![Axis(0), x_target, x_reweight];
        let d = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let scale = balance_scale(&x_all.view(), &d.view(), SBWEstimand::ATT);
        let stats =
            compute_balance_stats(&x_target.view(), &x_reweight.view(), &names, None, &scale);

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

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_sbw_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| sbw(ds, d, x, SBWConfig::default());
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.weights.clone());
    }
}
