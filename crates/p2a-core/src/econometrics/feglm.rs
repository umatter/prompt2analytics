//! Fixed Effects Generalized Linear Models (FEGLM).
//!
//! Implements GLM estimation with high-dimensional fixed effects absorption using
//! IRLS (Iteratively Reweighted Least Squares) combined with the Method of Alternating
//! Projections (MAP) for efficient demeaning.
//!
//! This is a pure Rust implementation equivalent to R's `alpaca::feglm()`.
//!
//! # Supported GLM Families
//!
//! - **Logit** (binomial with logit link): Binary outcomes
//! - **Probit** (binomial with probit link): Binary outcomes
//! - **Poisson** (log link): Count data
//! - **Gaussian** (identity link): Continuous outcomes (reduces to linear HDFE)
//!
//! # Algorithm
//!
//! The algorithm combines IRLS with weighted demeaning:
//!
//! 1. Initialize linear predictor η
//! 2. For each IRLS iteration:
//!    a. Compute working weights w and working response z
//!    b. Weighted demean z and X using MAP with weights w
//!    c. Solve weighted least squares on demeaned data
//!    d. Update coefficients and check convergence
//!
//! # References
//!
//! - Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models
//!   with High-Dimensional k-way Fixed Effects". ArXiv e-prints.
//!   <https://arxiv.org/abs/1707.01815>
//! - McCullagh, P. & Nelder, J.A. (1989). Generalized Linear Models. 2nd ed.
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::{Dataset, run_feglm, GlmFamily, FeglmConfig};
//!
//! let result = run_feglm(
//!     &dataset,
//!     "outcome",              // binary outcome (0/1)
//!     &["treatment", "age"],  // regressors
//!     &["firm_id", "year"],   // fixed effects to absorb
//!     GlmFamily::Logit,       // logistic regression
//!     None,                   // use default config
//! )?;
//!
//! println!("Coefficient: {}", result.coefficients[0]);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::xtx;
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::regression::ols::normal_equations_inverse;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, deviance_change, notes_text, sci};
use crate::traits::estimator::{
    SignificanceLevel, logistic_cdf, normal_cdf, normal_p_value_two_sided, normal_pdf,
    normal_quantile,
};

use super::hdfe::{
    FactorInfo, MAP_SCALE_FLOOR, demean_matrix_map, factor_info, fe_label, screen_regressors,
};

// ═══════════════════════════════════════════════════════════════════════════════
// GLM Family
// ═══════════════════════════════════════════════════════════════════════════════

/// GLM family specification for FEGLM.
///
/// Each family defines a link function g(μ) relating the linear predictor η to
/// the conditional mean μ = E[Y|X], and a variance function V(μ) characterizing
/// the relationship between variance and mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// Binomial with logit link: P(Y=1) = 1/(1+exp(-η))
    ///
    /// - Link: g(μ) = log(μ/(1-μ))
    /// - Inverse link: μ = 1/(1+exp(-η))
    /// - Variance: V(μ) = μ(1-μ)
    Logit,

    /// Binomial with probit link: P(Y=1) = Φ(η)
    ///
    /// - Link: g(μ) = Φ⁻¹(μ)
    /// - Inverse link: μ = Φ(η)
    /// - Variance: V(μ) = μ(1-μ)
    Probit,

    /// Poisson with log link: E[Y] = exp(η)
    ///
    /// - Link: g(μ) = log(μ)
    /// - Inverse link: μ = exp(η)
    /// - Variance: V(μ) = μ
    Poisson,

    /// Gaussian with identity link: E[Y] = η
    ///
    /// - Link: g(μ) = μ
    /// - Inverse link: μ = η
    /// - Variance: V(μ) = σ² (constant)
    ///
    /// Note: This reduces to standard linear HDFE.
    Gaussian,
}

impl fmt::Display for GlmFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlmFamily::Logit => write!(f, "Binomial (logit)"),
            GlmFamily::Probit => write!(f, "Binomial (probit)"),
            GlmFamily::Poisson => write!(f, "Poisson (log)"),
            GlmFamily::Gaussian => write!(f, "Gaussian (identity)"),
        }
    }
}

impl GlmFamily {
    /// Link function: g(μ) → η
    #[inline]
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            GlmFamily::Logit => {
                // log(μ/(1-μ))
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                (mu_safe / (1.0 - mu_safe)).ln()
            }
            GlmFamily::Probit => {
                // Φ⁻¹(μ) - inverse normal CDF (probit function)
                // mu_safe is inside (0, 1), so the quantile always exists
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                normal_quantile(mu_safe).unwrap_or(f64::NAN)
            }
            GlmFamily::Poisson => {
                // log(μ)
                mu.max(1e-10).ln()
            }
            GlmFamily::Gaussian => {
                // identity
                mu
            }
        }
    }

    /// Inverse link function: g⁻¹(η) → μ
    #[inline]
    pub fn inv_link(&self, eta: f64) -> f64 {
        match self {
            GlmFamily::Logit => logistic_cdf(eta),
            GlmFamily::Probit => normal_cdf(eta),
            GlmFamily::Poisson => {
                // exp(η), clamped to avoid overflow
                eta.clamp(-30.0, 30.0).exp()
            }
            GlmFamily::Gaussian => eta,
        }
    }

    /// Derivative of inverse link: ∂μ/∂η (for working weights)
    #[inline]
    pub fn mu_eta(&self, eta: f64) -> f64 {
        match self {
            GlmFamily::Logit => {
                let mu = logistic_cdf(eta);
                mu * (1.0 - mu)
            }
            GlmFamily::Probit => normal_pdf(eta),
            GlmFamily::Poisson => {
                // For log link: ∂μ/∂η = μ = exp(η)
                eta.clamp(-30.0, 30.0).exp()
            }
            GlmFamily::Gaussian => 1.0,
        }
    }

    /// Variance function: V(μ)
    #[inline]
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            GlmFamily::Logit | GlmFamily::Probit => {
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                mu_safe * (1.0 - mu_safe)
            }
            GlmFamily::Poisson => mu.max(1e-10),
            GlmFamily::Gaussian => 1.0,
        }
    }

    /// Working weight: w = (∂μ/∂η)² / V(μ)
    ///
    /// Used in IRLS to construct the weighted least squares problem.
    #[inline]
    pub fn working_weight(&self, eta: f64, mu: f64) -> f64 {
        match self {
            GlmFamily::Logit => {
                // For logit: w = μ(1-μ)
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                mu_safe * (1.0 - mu_safe)
            }
            GlmFamily::Probit => {
                // For probit: w = φ(η)² / [Φ(η)(1-Φ(η))]
                let phi = normal_pdf(eta);
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                let var = mu_safe * (1.0 - mu_safe);
                if var > 1e-10 { phi * phi / var } else { 1e-10 }
            }
            GlmFamily::Poisson => {
                // For Poisson with log link: w = μ
                mu.max(1e-10)
            }
            GlmFamily::Gaussian => 1.0,
        }
    }

    /// Working response: z = η + (y - μ) × (∂η/∂μ)
    ///
    /// This is the "adjusted dependent variable" in IRLS.
    #[inline]
    pub fn working_response(&self, y: f64, eta: f64, mu: f64) -> f64 {
        let residual = y - mu;
        match self {
            GlmFamily::Logit => {
                // ∂η/∂μ = 1/[μ(1-μ)]
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                let deriv = 1.0 / (mu_safe * (1.0 - mu_safe));
                eta + residual * deriv
            }
            GlmFamily::Probit => {
                // ∂η/∂μ = 1/φ(η)
                let phi = normal_pdf(eta);
                if phi > 1e-10 {
                    eta + residual / phi
                } else {
                    eta
                }
            }
            GlmFamily::Poisson => {
                // ∂η/∂μ = 1/μ
                let mu_safe = mu.max(1e-10);
                eta + residual / mu_safe
            }
            GlmFamily::Gaussian => y,
        }
    }

    /// Log-likelihood contribution for a single observation.
    ///
    /// For Poisson this is `y ln(mu) - mu - ln Gamma(y + 1)`, which takes constant
    /// time, is continuous in `y` (PPML outcomes need not be integers), equals R's
    /// `dpois(y, mu, log = TRUE)` for integer counts and follows fixest's
    /// convention for non-integer ones.
    #[inline]
    pub fn log_lik(&self, y: f64, mu: f64) -> f64 {
        match self {
            GlmFamily::Logit | GlmFamily::Probit => {
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                if y >= 0.5 {
                    mu_safe.ln()
                } else {
                    (1.0 - mu_safe).ln()
                }
            }
            GlmFamily::Poisson => {
                // y*log(μ) - μ - log Γ(y + 1)
                let mu_safe = mu.max(1e-10);
                y * mu_safe.ln() - mu_safe - statrs::function::gamma::ln_gamma(y + 1.0)
            }
            GlmFamily::Gaussian => {
                // -0.5 * (y - μ)² (up to a constant)
                -0.5 * (y - mu).powi(2)
            }
        }
    }

    /// Deviance contribution for a single observation.
    #[inline]
    pub fn deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            GlmFamily::Logit | GlmFamily::Probit => {
                let mu_safe = mu.clamp(1e-10, 1.0 - 1e-10);
                -2.0 * if y >= 0.5 {
                    mu_safe.ln()
                } else {
                    (1.0 - mu_safe).ln()
                }
            }
            GlmFamily::Poisson => {
                let mu_safe = mu.max(1e-10);
                if y > 0.0 {
                    2.0 * (y * (y / mu_safe).ln() - (y - mu_safe))
                } else {
                    2.0 * mu_safe
                }
            }
            GlmFamily::Gaussian => (y - mu).powi(2),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for FEGLM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeglmConfig {
    /// Maximum IRLS iterations.
    /// Must be at least 1.
    /// Default: 25
    pub max_iter: usize,

    /// IRLS stops when |dev - dev_old| / (|dev| + 0.1) < tolerance (fixest's
    /// glm.tol rule).
    /// Must be finite and greater than 0.
    /// Default: 1e-8
    pub tolerance: f64,

    /// Relative MAP tolerance for demeaning within each IRLS step: a sweep stops when
    /// the weighted norm of the change is at most map_tolerance times
    /// max(||demeaned column||, 1e-6 ||column||).
    /// Must be finite and greater than 0.
    /// Default: 1e-8
    pub map_tolerance: f64,

    /// Maximum MAP iterations per IRLS step.
    /// Must be at least 1.
    /// Default: 10000
    pub map_max_iter: usize,

    /// Minimum weight threshold (to avoid division by zero).
    /// Must be finite and at least 0.
    /// Default: 1e-10
    pub weight_min: f64,

    /// Whether to use Gearhart-Koshy acceleration for MAP.
    /// Default: true
    pub accelerate: bool,
}

impl Default for FeglmConfig {
    fn default() -> Self {
        Self {
            max_iter: 25,
            tolerance: 1e-8,
            map_tolerance: 1e-8,
            map_max_iter: 10000,
            weight_min: 1e-10,
            accelerate: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from FEGLM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeglmResult {
    // ═══════════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════════
    /// GLM family used
    pub family: GlmFamily,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (regressors only, no intercept)
    pub variables: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fixed Effects Info
    // ═══════════════════════════════════════════════════════════════════════════
    /// Names of absorbed fixed effect dimensions
    pub fe_dimensions: Vec<String>,
    /// Number of unique levels per FE dimension
    pub fe_counts: Vec<usize>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Results
    // ═══════════════════════════════════════════════════════════════════════════
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics (for MLE)
    pub z_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fit Statistics
    // ═══════════════════════════════════════════════════════════════════════════
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Null log-likelihood (intercept-only model)
    pub log_likelihood_null: f64,
    /// Deviance at convergence
    pub deviance: f64,
    /// Null deviance
    pub null_deviance: f64,
    /// McFadden's Pseudo R-squared: 1 - LL/LL_null
    pub pseudo_r_squared: f64,
    /// AIC: 2k - 2*LL
    pub aic: f64,
    /// BIC: k*ln(n) - 2*LL
    pub bic: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Dispersion (for Poisson)
    // ═══════════════════════════════════════════════════════════════════════════
    /// Dispersion parameter (1.0 for binomial, estimated for Poisson if overdispersed)
    pub dispersion: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Convergence
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of IRLS iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final relative deviance change |dev - dev_old| / (|dev| + 0.1)
    pub final_change: f64,
    /// The IRLS iteration limit
    #[serde(default)]
    pub max_iter: usize,
    /// The IRLS tolerance
    #[serde(default)]
    pub tolerance: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Dimensions
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Number of positive outcomes (for binomial)
    pub n_positive: usize,
    /// Residual degrees of freedom
    pub df_resid: usize,
    /// Degrees of freedom absorbed by fixed effects
    pub df_absorbed: usize,

    // ═══════════════════════════════════════════════════════════════════════════
    // Internal caches (not serialized)
    // ═══════════════════════════════════════════════════════════════════════════
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) beta: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) vcov: Array2<f64>,

    /// Rows dropped because of a missing value in y, a regressor or a fixed-effect id
    /// (fixest drops them too).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,

    /// Regressors not estimated: absorbed by the fixed effects or collinear with the
    /// others after removing them (fixest drops them too)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<String>,
    /// Warnings: dropped regressors, non-convergence (IRLS or the demeaning)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for FeglmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(Convergence::new(
            self.converged,
            self.iterations,
            self.max_iter,
            "relative change in deviance",
            self.final_change,
            self.tolerance,
        ))
    }
}

impl fmt::Display for FeglmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FEGLM: {} Regression with Fixed Effects", self.family)?;
        writeln!(f, "=======================================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(
            f,
            "No. Observations: {} (Positive: {})",
            self.n_obs, self.n_positive
        )?;
        writeln!(f, "Df Residuals: {}", self.df_resid)?;
        writeln!(f, "Df Absorbed: {}", self.df_absorbed)?;
        writeln!(f)?;

        writeln!(f, "Fixed Effects:")?;
        for (name, count) in self.fe_dimensions.iter().zip(self.fe_counts.iter()) {
            writeln!(f, "  {}: {} levels", name, count)?;
        }
        writeln!(f)?;

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Null Log-Likelihood: {:.4}", self.log_likelihood_null)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        if self.dispersion != 1.0 {
            writeln!(f, "Dispersion: {:.4}", self.dispersion)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "z", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // Notes before the missing-value note, which stays the last line
        write!(f, "{}", notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weighted Demeaning Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// The weighted alternating projections onto one set of weights: each factor's group weight
/// sums, which no sweep and no column changes, computed once, and scratch for the group
/// means, so a sweep demeans in place without allocating.
struct WeightedProjector<'a> {
    weights: &'a Array1<f64>,
    factors: &'a [FactorInfo],
    /// Per factor, the weight sum of every level.
    weight_sums: Vec<Vec<f64>>,
    /// Per factor, scratch for the weighted sums and then the means of the levels.
    means: Vec<Vec<f64>>,
}

impl<'a> WeightedProjector<'a> {
    fn new(weights: &'a Array1<f64>, factors: &'a [FactorInfo]) -> Self {
        let weight_sums = factors
            .iter()
            .map(|factor| {
                let mut sums = vec![0.0; factor.n_levels];
                for (&g, &w) in factor.ids.iter().zip(weights.iter()) {
                    sums[g] += w;
                }
                sums
            })
            .collect();
        let means = factors.iter().map(|f| vec![0.0; f.n_levels]).collect();
        WeightedProjector {
            weights,
            factors,
            weight_sums,
            means,
        }
    }

    /// Subtract the weighted means of factor `f`'s levels from `data`, in place.
    fn demean_by(&mut self, f: usize, data: &mut [f64]) {
        let (ids, means) = (&self.factors[f].ids, &mut self.means[f]);
        means.fill(0.0);
        for ((&g, &v), &w) in ids.iter().zip(data.iter()).zip(self.weights.iter()) {
            means[g] += v * w;
        }
        for (m, &w_sum) in means.iter_mut().zip(&self.weight_sums[f]) {
            *m = if w_sum > 1e-10 { *m / w_sum } else { 0.0 };
        }
        for (v, &g) in data.iter_mut().zip(ids) {
            *v -= means[g];
        }
    }

    /// One round of weighted alternating projections (every factor in turn), in place.
    fn sweep(&mut self, data: &mut [f64]) {
        for f in 0..self.factors.len() {
            self.demean_by(f, data);
        }
    }
}

/// Weighted demean a vector by a single factor.
///
/// Subtracts the weighted group mean from each observation.
#[cfg(test)]
fn weighted_demean_by_factor(
    data: &Array1<f64>,
    weights: &Array1<f64>,
    factor: &FactorInfo,
) -> Array1<f64> {
    let mut out = data.to_vec();
    WeightedProjector::new(weights, std::slice::from_ref(factor)).demean_by(0, &mut out);
    Array1::from(out)
}

/// Weighted demean using the Method of Alternating Projections, onto the projector's
/// weights and factors.
///
/// Stops when the weighted norm of a sweep's change is at most `tolerance` times
/// `max(||current||_w, MAP_SCALE_FLOOR * ||data||_w)`; the returned change is that
/// ratio. Returns `EconError::Cancelled` when the current budget runs out between sweeps.
fn weighted_demean_map(
    data: &Array1<f64>,
    proj: &mut WeightedProjector,
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> EconResult<(Array1<f64>, usize, f64, bool)> {
    let weights = proj.weights;
    if proj.factors.is_empty() {
        return Ok((data.clone(), 0, 0.0, true));
    }

    // Single factor: one pass is sufficient
    if proj.factors.len() == 1 {
        let mut demeaned = data.clone();
        proj.demean_by(0, demeaned.as_slice_mut().ok_or_else(not_contiguous)?);
        return Ok((demeaned, 1, 0.0, true));
    }

    let mut current = data.to_vec();
    let mut prev = current.clone();
    let mut prev_prev = current.clone();
    let mut converged = false;
    let mut final_change = f64::MAX;
    let w_norm = |v: &Array1<f64>| -> f64 {
        v.iter()
            .zip(weights.iter())
            .map(|(&c, &w)| w * c * c)
            .sum::<f64>()
            .sqrt()
    };
    let floor = MAP_SCALE_FLOOR * w_norm(data);

    for iter in 1..=max_iter {
        resource::checkpoint()?;
        if iter > 1 {
            prev_prev.copy_from_slice(&prev);
        }
        prev.copy_from_slice(&current);

        // One round of weighted alternating projections
        proj.sweep(&mut current);

        // Change and current norm (weighted L2) in one pass; each sum accumulates in
        // row order, as two separate passes would
        let mut delta = 0.0f64;
        let mut norm_w = 0.0f64;
        for ((&c, &p), &w) in current.iter().zip(prev.iter()).zip(weights.iter()) {
            delta += w * (c - p).powi(2);
            norm_w += w * c * c;
        }
        let delta = delta.sqrt();
        let scale = norm_w.sqrt().max(floor).max(f64::MIN_POSITIVE);
        final_change = delta / scale;

        // Relative stopping rule: invariant to rescaling the column
        if delta <= tolerance * scale {
            converged = true;
            return Ok((Array1::from(current), iter, final_change, converged));
        }

        // Gearhart-Koshy acceleration
        if accelerate && iter > 2 {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            let mut norm_sq = 0.0f64;
            for ((&c, &p), &pp) in current.iter().zip(prev.iter()).zip(prev_prev.iter()) {
                let d = c - p;
                let dp = p - pp;
                numerator += d * dp;
                denominator += dp * dp;
                norm_sq += c * c;
            }
            let current_norm = norm_sq.sqrt();

            // Scale-free guard: skip only a step that is rounding noise
            if denominator > (f64::EPSILON * current_norm).powi(2) {
                let alpha = (numerator / denominator).clamp(0.0, 1.0);
                if alpha > 0.0 && alpha < 1.0 {
                    for (c, &p) in current.iter_mut().zip(prev.iter()) {
                        let d = *c - p;
                        *c = p + d * (1.0 + alpha);
                    }
                }
            }
        }
    }

    Ok((Array1::from(current), max_iter, final_change, converged))
}

fn not_contiguous() -> EconError {
    EconError::Internal("FEGLM: a demeaning buffer is not contiguous".to_string())
}

/// A weighted demeaning of the scaled design and the weights it was computed with.
struct DemeanedX {
    weights: Array1<f64>,
    x: Array2<f64>,
    change: f64,
    converged: bool,
}

/// A weighted demeaning of the scaled working response and the response it was computed
/// from (with the weights of the cached [`DemeanedX`]).
struct DemeanedZ {
    z: Array1<f64>,
    demeaned: Array1<f64>,
    change: f64,
    converged: bool,
}

/// `x` with row `i` multiplied by `s[i]`.
fn scale_rows(x: &Array2<f64>, s: &Array1<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::zeros((n, k));
    for j in 0..k {
        for i in 0..n {
            out[[i, j]] = x[[i, j]] * s[i];
        }
    }
    out
}

/// Whether two vectors hold the same values, bit for bit.
fn same_values(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(p, q)| p.to_bits() == q.to_bits())
}

/// Weighted demean a matrix column-wise, onto the projector's weights and factors.
fn weighted_demean_matrix_map(
    x: &Array2<f64>,
    proj: &mut WeightedProjector,
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> EconResult<(Array2<f64>, usize, f64, bool)> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));
    let mut max_iterations: usize = 0;
    let mut max_change: f64 = 0.0;
    let mut all_converged = true;

    for j in 0..k {
        let col = x.column(j).to_owned();
        let (col_demeaned, iters, change, converged) =
            weighted_demean_map(&col, proj, tolerance, max_iter, accelerate)?;
        x_demeaned.column_mut(j).assign(&col_demeaned);

        max_iterations = max_iterations.max(iters);
        max_change = max_change.max(change);
        all_converged = all_converged && converged;
    }

    Ok((x_demeaned, max_iterations, max_change, all_converged))
}

/// Check the IRLS and MAP controls: a tolerance of zero or less can never be
/// met, zero iterations cannot fit anything, and the weight floor must be a
/// usable number.
fn validate_config(config: &FeglmConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    for (name, value) in [
        ("tolerance", config.tolerance),
        ("map_tolerance", config.map_tolerance),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return invalid(format!(
                "{name} must be a finite number greater than 0 (got {value})"
            ));
        }
    }
    if config.max_iter == 0 {
        return invalid("max_iter must be at least 1 (got 0)".to_string());
    }
    if config.map_max_iter == 0 {
        return invalid("map_max_iter must be at least 1 (got 0)".to_string());
    }
    if !(config.weight_min.is_finite() && config.weight_min >= 0.0) {
        return invalid(format!(
            "weight_min must be a finite number of at least 0 (got {})",
            config.weight_min
        ));
    }
    Ok(())
}

/// Compute degrees of freedom absorbed by fixed effects.
fn compute_absorbed_df(factors: &[FactorInfo]) -> usize {
    if factors.is_empty() {
        return 0;
    }

    let total_levels: usize = factors.iter().map(|f| f.n_levels).sum();

    // For multi-way FE, there's redundancy
    let redundant = if factors.len() > 1 {
        factors.len() - 1
    } else {
        0
    };

    total_levels.saturating_sub(redundant)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Generalized Linear Model with High-Dimensional Fixed Effects.
///
/// Uses IRLS (Iteratively Reweighted Least Squares) combined with the Method of
/// Alternating Projections for efficient absorption of multiple fixed effects.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns (regressors)
/// * `fe_cols` - Names of the fixed effect columns to absorb
/// * `family` - GLM family (Logit, Probit, Poisson, Gaussian)
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
///
/// A `FeglmResult` containing coefficient estimates, standard errors,
/// and diagnostic information.
///
/// # Missing values
///
/// Rows with a missing value (null or NaN) in `y`, a regressor or a fixed-effect
/// column are dropped and reported in `missing` (fixest drops them too); an infinite
/// value is refused (`NonFiniteValues`) before the family's checks.
///
/// # Example
///
/// ```ignore
/// let result = run_feglm(
///     &dataset,
///     "outcome",
///     &["treatment", "control"],
///     &["firm_id", "year"],
///     GlmFamily::Logit,
///     None,
/// )?;
/// ```
///
/// # References
///
/// - Stammann, A. (2018). "Fast and Feasible Estimation of GLMs with HDFE".
pub fn run_feglm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    fe_cols: &[&str],
    family: GlmFamily,
    config: Option<FeglmConfig>,
) -> EconResult<FeglmResult> {
    let config = config.unwrap_or_default();

    // Validate inputs
    if fe_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one fixed effect column must be specified".to_string(),
        });
    }

    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one regressor (X) column must be specified".to_string(),
        });
    }

    validate_config(&config)?;

    // y, X and every fixed-effect column on the rows complete in all of them
    // (fixest drops incomplete rows too); an infinite value is refused here,
    // before the family's checks
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .factors(fe_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric_array(y_col);

    // Validate y for binomial families
    let n_positive = y.iter().filter(|&&v| v >= 0.5).count();
    match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            if n_positive == 0 || n_positive == y.len() {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Dependent variable '{}' must be binary with both 0 and 1 values for {}. \
                         Found {} ones out of {} observations.",
                        y_col,
                        family,
                        n_positive,
                        y.len()
                    ),
                });
            }
        }
        GlmFamily::Poisson => {
            // Check for non-negative values
            if y.iter().any(|&v| v < 0.0) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Dependent variable '{}' must be non-negative for Poisson regression.",
                        y_col
                    ),
                });
            }
        }
        GlmFamily::Gaussian => {}
    }

    // Design matrix WITHOUT intercept (FE absorbs the constant)
    let design = cols.matrix(x_cols, false)?;
    let x_all = design.data;
    let all_names = design.column_names;
    let n = y.len();

    // Factor information for each FE dimension (levels in R factor order)
    let factors: Vec<FactorInfo> = fe_cols
        .iter()
        .map(|col| factor_info(col, cols.factor(col)))
        .collect::<EconResult<Vec<_>>>()?;

    // Demeaning steps that stopped at map_max_iter: (step, relative change)
    let mut map_stalls: Vec<(String, f64)> = Vec::new();

    // Regressors absorbed by the effects or collinear after them are not estimated.
    // Rank against the fixed effects does not depend on positive weights
    // (rank(M_D^W X) = rank([X D]) - rank(D)), so one unit-weight screen covers every
    // IRLS step.
    let (x_screen_dm, _, screen_change, screen_conv) = demean_matrix_map(
        &x_all,
        &factors,
        config.map_tolerance,
        config.map_max_iter,
        config.accelerate,
    )?;
    if !screen_conv {
        map_stalls.push(("the regressor screen".to_string(), screen_change));
    }
    let screen = screen_regressors(x_all.view(), x_screen_dm, &all_names, &fe_label(fe_cols))?;
    let mut warnings = screen.warnings.clone();
    let x = if screen.kept.len() == x_all.ncols() {
        x_all
    } else {
        x_all.select(Axis(1), &screen.kept)
    };
    let var_names: Vec<String> = screen.kept.iter().map(|&j| all_names[j].clone()).collect();
    let k = screen.kept.len();

    // Compute absorbed degrees of freedom
    let df_absorbed = compute_absorbed_df(&factors);

    // Check we have enough observations
    let df_resid = n
        .checked_sub(k + df_absorbed)
        .filter(|&d| d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: k + df_absorbed + 1,
            provided: n,
            context: "Not enough observations for FEGLM estimation".to_string(),
        })?;

    // ═══════════════════════════════════════════════════════════════════════════
    // Initialize
    // ═══════════════════════════════════════════════════════════════════════════
    let mut beta = Array1::zeros(k);

    // Initial guess for eta based on family
    let mut eta: Array1<f64> = match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            let p_bar = (n_positive as f64 / n as f64).clamp(0.01, 0.99);
            Array1::from_elem(n, family.link(p_bar))
        }
        GlmFamily::Poisson => y.mapv(|yi| (yi.max(0.1)).ln()),
        GlmFamily::Gaussian => y.clone(),
    };

    // μ is kept in step with η: computed once per update of η and used by the deviance,
    // the next iteration and the final statistics.
    let deviance_of = |mu: &Array1<f64>| -> f64 {
        y.iter()
            .zip(mu.iter())
            .map(|(&yi, &m)| family.deviance(yi, m))
            .sum()
    };
    let mut mu: Array1<f64> = eta.mapv(|e| family.inv_link(e));
    // The weighted demeaning of X from the last IRLS iteration and its weights: when the
    // weights repeat (Gaussian: all 1), the same inputs give the same demeaned X.
    let mut x_demeaned_cache: Option<DemeanedX> = None;
    let mut z_cache: Option<DemeanedZ> = None;

    // ═══════════════════════════════════════════════════════════════════════════
    // IRLS Loop
    // ═══════════════════════════════════════════════════════════════════════════
    let mut iterations = 0;
    let mut converged = false;
    let mut final_change = f64::NAN;
    let mut devold = deviance_of(&mu);

    for iter in 0..config.max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // 1. μ from η (kept in step with η)

        // 2. Compute working weights
        let weights: Array1<f64> = eta
            .iter()
            .zip(mu.iter())
            .map(|(&e, &m)| family.working_weight(e, m).max(config.weight_min))
            .collect();

        // 3. Compute working response
        let z: Array1<f64> = y
            .iter()
            .zip(eta.iter())
            .zip(mu.iter())
            .map(|((&yi, &ei), &mi)| family.working_response(yi, ei, mi))
            .collect();

        // 4. Scale by sqrt(weights) for weighted least squares
        let sqrt_w: Array1<f64> = weights.mapv(|w| w.sqrt());

        // Scale z and X by sqrt(weights)
        let z_scaled: Array1<f64> = &z * &sqrt_w;

        // 5. Weighted demean z and X by fixed effects
        // Note: We demean the pre-scaled data, which effectively applies weighted demeaning.
        // Both are memoized: the same weights demean the same scaled X to the same result,
        // and the same weights and z the same z, so the solve repeats too (Gaussian: the
        // weights are all 1 and z is y in every iteration).
        let same_weights = x_demeaned_cache
            .as_ref()
            .is_some_and(|c| same_values(&c.weights, &weights));
        let z_repeats = same_weights && z_cache.as_ref().is_some_and(|c| same_values(&c.z, &z));
        let mut proj =
            (!z_repeats || !same_weights).then(|| WeightedProjector::new(&weights, &factors));
        if !z_repeats {
            let proj = proj.as_mut().ok_or_else(|| {
                EconError::Internal("FEGLM: weighted projector missing".to_string())
            })?;
            let (z_demeaned, _, change, converged) = weighted_demean_map(
                &z_scaled,
                proj,
                config.map_tolerance,
                config.map_max_iter,
                config.accelerate,
            )?;
            z_cache = Some(DemeanedZ {
                z: z.clone(),
                demeaned: z_demeaned,
                change,
                converged,
            });
        }

        if !same_weights {
            let proj = proj.as_mut().ok_or_else(|| {
                EconError::Internal("FEGLM: weighted projector missing".to_string())
            })?;
            let x_scaled = scale_rows(&x, &sqrt_w);
            let (x_dm, _, change, conv) = weighted_demean_matrix_map(
                &x_scaled,
                proj,
                config.map_tolerance,
                config.map_max_iter,
                config.accelerate,
            )?;
            x_demeaned_cache = Some(DemeanedX {
                weights: weights.clone(),
                x: x_dm,
                change,
                converged: conv,
            });
        }
        let (
            Some(DemeanedX {
                x: x_demeaned,
                change: x_change,
                converged: x_conv,
                ..
            }),
            Some(DemeanedZ {
                demeaned: z_demeaned,
                change: z_change,
                converged: z_conv,
                ..
            }),
        ) = (x_demeaned_cache.as_ref(), z_cache.as_ref())
        else {
            return Err(EconError::Internal(
                "FEGLM: demeaned design missing".to_string(),
            ));
        };
        let (x_change, x_conv, z_change, z_conv) = (*x_change, *x_conv, *z_change, *z_conv);

        if !z_conv || !x_conv {
            map_stalls.push((
                format!("IRLS iteration {iterations}"),
                z_change.max(x_change),
            ));
        }

        // 6. Solve weighted least squares on demeaned data (OLS's solve decision: the
        // normal equations when well conditioned, the pivoted QR otherwise); the same
        // demeaned X and z give the last iteration's beta again
        if !z_repeats {
            beta = match normal_equations_inverse(x_demeaned.view())? {
                Some(normal) => normal.coefficients(x_demeaned.view(), &z_demeaned),
                None => {
                    let ls = least_squares(x_demeaned.view(), z_demeaned.view())?;
                    if ls.rank < k {
                        return Err(EconError::SingularMatrix {
                            context: format!(
                                "FEGLM IRLS iteration {iterations}: weighted design rank {} of {k}",
                                ls.rank
                            ),
                            suggestion: "Check for perfect multicollinearity or separation"
                                .to_string(),
                        });
                    }
                    ls.coefficients
                }
            };
        }

        // 7. Update linear predictor
        // For FEGLM, we need to recover the FE contribution
        // η_new = Xβ + FE effects
        // The FE effects are implicitly in the demeaned residuals

        // Simple approach: use the fitted values from demeaned regression
        // and add back the structure
        let x_beta: Array1<f64> = x.dot(&beta);

        // The FE contribution can be approximated by:
        // z - X_demeaned β / sqrt_w ≈ FE contribution
        // But for simplicity, we use the iterative update:
        // η_new = η_old + (β_new - β_old) influence
        // Actually, we need to properly update eta including FE effects

        // For IRLS to work correctly, we update eta as follows:
        // 1. The working response z contains information about FE
        // 2. After demeaning, we get the coefficient update
        // 3. The new eta = Xβ + (z - Xβ) projected onto FE space

        // Simplified approach: use the relationship that
        // z_demeaned = (X_demeaned)β + residual
        // The residual contains the deviation from the model

        // For numerical stability, we compute:
        // eta_new = X*beta + FE_effects
        // where FE_effects are recovered from the working model

        // One approach: eta_new = eta + step * (z - eta)
        // weighted by the relative change
        eta = x_beta.clone();

        // Add back FE effects by computing group means from residuals
        // This is a simplified approximation
        let z_residual = &z - &x_beta;
        for factor in &factors {
            let mut group_sums = vec![0.0; factor.n_levels];
            let mut group_counts = vec![0.0; factor.n_levels];

            for i in 0..n {
                let g = factor.ids[i];
                group_sums[g] += z_residual[i] * weights[i];
                group_counts[g] += weights[i];
            }

            let group_means: Vec<f64> = group_sums
                .iter()
                .zip(group_counts.iter())
                .map(|(&s, &c)| if c > 1e-10 { s / c } else { 0.0 })
                .collect();

            for i in 0..n {
                eta[i] += group_means[factor.ids[i]];
            }
        }

        // 8. Check convergence: fixest's (glm.fit's) relative deviance change
        mu = eta.mapv(|e| family.inv_link(e));
        let dev = deviance_of(&mu);
        final_change = deviance_change(dev, devold);
        devold = dev;
        if final_change < config.tolerance {
            converged = true;
            break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute Final Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    // Final μ (in step with η) and log-likelihood
    let mu_final = mu;

    let log_likelihood: f64 = y
        .iter()
        .zip(mu_final.iter())
        .map(|(&yi, &mi)| family.log_lik(yi, mi))
        .sum();

    // Null log-likelihood (intercept-only, approximated)
    let log_likelihood_null = match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            let p_bar = n_positive as f64 / n as f64;
            n_positive as f64 * p_bar.ln() + (n - n_positive) as f64 * (1.0 - p_bar).ln()
        }
        GlmFamily::Poisson => {
            let y_mean = y.mean().unwrap_or(1.0);
            y.iter().map(|&yi| family.log_lik(yi, y_mean)).sum()
        }
        GlmFamily::Gaussian => {
            let y_mean = y.mean().unwrap_or(0.0);
            y.iter().map(|&yi| -0.5 * (yi - y_mean).powi(2)).sum()
        }
    };

    // Deviance (of the final μ, computed with the convergence check)
    let deviance: f64 = devold;

    // Null deviance (approximated)
    let null_deviance = match family {
        GlmFamily::Logit | GlmFamily::Probit => -2.0 * log_likelihood_null,
        GlmFamily::Poisson => {
            let y_mean = y.mean().unwrap_or(1.0);
            y.iter().map(|&yi| family.deviance(yi, y_mean)).sum()
        }
        GlmFamily::Gaussian => {
            let y_mean = y.mean().unwrap_or(0.0);
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum()
        }
    };

    // Pseudo R-squared
    let pseudo_r_squared = if log_likelihood_null.abs() > 1e-10 {
        1.0 - log_likelihood / log_likelihood_null
    } else {
        0.0
    };

    // AIC and BIC
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let bic = (k as f64) * (n as f64).ln() - 2.0 * log_likelihood;

    // Dispersion parameter
    let dispersion = match family {
        GlmFamily::Gaussian => deviance / df_resid as f64,
        GlmFamily::Poisson => {
            // Pearson chi-squared based dispersion estimate
            let pearson_chi2: f64 = y
                .iter()
                .zip(mu_final.iter())
                .map(|(&yi, &mi)| {
                    let var = family.variance(mi);
                    if var > 1e-10 {
                        (yi - mi).powi(2) / var
                    } else {
                        0.0
                    }
                })
                .sum();
            pearson_chi2 / df_resid as f64
        }
        _ => 1.0,
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute Variance-Covariance Matrix
    // ═══════════════════════════════════════════════════════════════════════════

    // Information matrix from final iteration
    let weights_final: Array1<f64> = eta
        .iter()
        .zip(mu_final.iter())
        .map(|(&e, &m)| family.working_weight(e, m).max(config.weight_min))
        .collect();

    let sqrt_w_final: Array1<f64> = weights_final.mapv(|w| w.sqrt());

    let (x_demeaned_final, final_map_change, final_map_conv) =
        match x_demeaned_cache.filter(|c| same_values(&c.weights, &weights_final)) {
            Some(c) => (c.x, c.change, c.converged),
            None => {
                let x_scaled_final = scale_rows(&x, &sqrt_w_final);
                let (x_dm, _, change, conv) = weighted_demean_matrix_map(
                    &x_scaled_final,
                    &mut WeightedProjector::new(&weights_final, &factors),
                    config.map_tolerance,
                    config.map_max_iter,
                    config.accelerate,
                )?;
                (x_dm, change, conv)
            }
        };
    if !final_map_conv {
        map_stalls.push(("the final information matrix".to_string(), final_map_change));
    }

    let xtx_final = xtx(&x_demeaned_final.view());
    let vcov = sym_inverse(
        xtx_final.view(),
        SingularPolicy::Error,
        "FEGLM information matrix",
    )?
    .inverse;

    // Apply dispersion scaling for Gaussian/Poisson
    let vcov = match family {
        GlmFamily::Gaussian | GlmFamily::Poisson => &vcov * dispersion,
        _ => vcov,
    };

    // Standard errors
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // z-statistics and p-values
    let coefficients = beta.to_vec();
    let z_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    if !converged {
        let record = Convergence::new(
            false,
            iterations,
            config.max_iter,
            "relative change in deviance",
            final_change,
            config.tolerance,
        );
        warnings.extend(record.note("FEGLM IRLS"));
    }
    if let Some((step, change)) = map_stalls.first() {
        let more = match map_stalls.len() - 1 {
            0 => String::new(),
            m => format!("; also in {m} later step(s)"),
        };
        warnings.push(format!(
            "FEGLM: the weighted alternating projections reached map_max_iter = {} in {step} \
             (relative change {}, map_tolerance {}){more}; that step used the last iterate",
            config.map_max_iter,
            sci(*change),
            sci(config.map_tolerance)
        ));
    }

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    Ok(FeglmResult {
        family,
        dep_var: y_col.to_string(),
        variables: var_names,
        fe_dimensions: factors.iter().map(|f| f.name.clone()).collect(),
        fe_counts: factors.iter().map(|f| f.n_levels).collect(),
        coefficients,
        std_errors,
        z_stats,
        p_values,
        significance,
        log_likelihood,
        log_likelihood_null,
        deviance,
        null_deviance,
        pseudo_r_squared,
        aic,
        bic,
        dispersion,
        iterations,
        converged,
        final_change,
        max_iter: config.max_iter,
        tolerance: config.tolerance,
        n_obs: n,
        n_positive,
        df_resid,
        df_absorbed,
        beta,
        vcov,
        missing: cols.missing().clone(),
        dropped: screen.dropped,
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use polars::df;

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

    fn expect_invalid<T>(res: EconResult<T>, needle: &str) {
        match res {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains(needle),
                    "message {message:?} should name `{needle}`"
                )
            }
            Err(e) => panic!("expected InvalidSpecification naming `{needle}`, got: {e}"),
            Ok(_) => panic!("expected InvalidSpecification naming `{needle}`, got Ok"),
        }
    }

    // Test data: binary panel with known structure
    fn create_binary_panel() -> Dataset {
        // 3 firms, 4 years each = 12 observations
        // True model: P(y=1) = logistic(1.5*x + firm_effect + time_effect)
        // Firm effects: A=0, B=1, C=-0.5
        // Time effects: 2020=0, 2021=0.5, 2022=0.3, 2023=-0.2
        let df = df! {
            "firm" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2023, 2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023],
            "x" => [0.1, 0.5, -0.2, 0.8, 0.3, 0.7, 0.1, -0.1, -0.3, 0.2, 0.4, 0.6],
            // y based on logistic probability with FE
            "y" => [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn create_count_panel() -> Dataset {
        // Poisson panel data
        let df = df! {
            "firm" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2023, 2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023],
            "x" => [1.0, 1.5, 1.2, 1.8, 2.0, 2.3, 2.1, 2.5, 0.5, 0.8, 0.6, 1.0],
            "count" => [2.0, 4.0, 3.0, 5.0, 8.0, 10.0, 9.0, 12.0, 1.0, 2.0, 1.0, 3.0],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_glm_family_logit_functions() {
        let family = GlmFamily::Logit;

        // Link function: logit(0.5) = 0
        assert!((family.link(0.5)).abs() < 1e-10);

        // Inverse link: logistic(0) = 0.5
        assert!((family.inv_link(0.0) - 0.5).abs() < 1e-10);

        // mu_eta at eta=0
        let mu_eta = family.mu_eta(0.0);
        assert!((mu_eta - 0.25).abs() < 1e-10); // 0.5 * 0.5 = 0.25

        // Variance at mu=0.5
        assert!((family.variance(0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_glm_family_poisson_functions() {
        let family = GlmFamily::Poisson;

        // Link: log(1) = 0
        assert!((family.link(1.0)).abs() < 1e-10);

        // Inverse link: exp(0) = 1
        assert!((family.inv_link(0.0) - 1.0).abs() < 1e-10);

        // Variance equals mean for Poisson
        assert!((family.variance(3.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_glm_family_gaussian_functions() {
        let family = GlmFamily::Gaussian;

        // Identity link
        assert!((family.link(5.0) - 5.0).abs() < 1e-10);
        assert!((family.inv_link(5.0) - 5.0).abs() < 1e-10);

        // Constant variance
        assert!((family.variance(100.0) - 1.0).abs() < 1e-10);

        // mu_eta is 1
        assert!((family.mu_eta(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_demean_single_factor() {
        // Simple test with uniform weights
        let data = Array1::from(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };

        let demeaned = weighted_demean_by_factor(&data, &weights, &factor);

        // Group 0 mean: 2, Group 1 mean: 11
        assert!((demeaned[0] - (-1.0)).abs() < 1e-10);
        assert!((demeaned[1] - 0.0).abs() < 1e-10);
        assert!((demeaned[2] - 1.0).abs() < 1e-10);
        assert!((demeaned[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_demean_with_varying_weights() {
        // Test that weights affect the group mean
        let data = Array1::from(vec![1.0, 2.0, 10.0, 20.0]);
        // High weight on first observation in each group
        let weights = Array1::from(vec![10.0, 1.0, 10.0, 1.0]);
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 1, 1],
            counts: vec![2, 2],
        };

        let demeaned = weighted_demean_by_factor(&data, &weights, &factor);

        // Weighted mean of group 0: (10*1 + 1*2)/(10+1) = 12/11 ≈ 1.09
        // Weighted mean of group 1: (10*10 + 1*20)/(10+1) = 120/11 ≈ 10.91
        let expected_mean_0 = (10.0 * 1.0 + 1.0 * 2.0) / 11.0;
        let expected_mean_1 = (10.0 * 10.0 + 1.0 * 20.0) / 11.0;

        assert!((demeaned[0] - (1.0 - expected_mean_0)).abs() < 1e-10);
        assert!((demeaned[1] - (2.0 - expected_mean_0)).abs() < 1e-10);
        assert!((demeaned[2] - (10.0 - expected_mean_1)).abs() < 1e-10);
        assert!((demeaned[3] - (20.0 - expected_mean_1)).abs() < 1e-10);
    }

    #[test]
    fn test_feglm_logit_basic() {
        let dataset = create_binary_panel();

        let result = run_feglm(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            GlmFamily::Logit,
            None,
        )
        .unwrap();

        // Basic structure checks
        assert!(result.converged || result.iterations > 0);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.fe_dimensions.len(), 2);
        assert_eq!(result.fe_counts[0], 3); // 3 firms
        assert_eq!(result.fe_counts[1], 4); // 4 years

        // Coefficient should be returned
        assert!(!result.coefficients.is_empty());

        // Log-likelihood should be negative (since it's log of probabilities < 1)
        assert!(result.log_likelihood < 0.0 || result.log_likelihood.is_finite());

        println!("FEGLM Logit result:\n{}", result);
    }

    #[test]
    fn test_feglm_probit_basic() {
        let dataset = create_binary_panel();

        let result = run_feglm(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            GlmFamily::Probit,
            None,
        )
        .unwrap();

        // Basic structure checks
        assert!(result.converged || result.iterations > 0);
        assert_eq!(result.family, GlmFamily::Probit);
        assert!(!result.coefficients.is_empty());
    }

    #[test]
    fn test_feglm_poisson_basic() {
        let dataset = create_count_panel();

        let result = run_feglm(
            &dataset,
            "count",
            &["x"],
            &["firm", "year"],
            GlmFamily::Poisson,
            None,
        )
        .unwrap();

        // Basic structure checks
        assert!(result.converged || result.iterations > 0);
        assert_eq!(result.family, GlmFamily::Poisson);
        assert!(!result.coefficients.is_empty());

        // For Poisson, dispersion might be > 1 if overdispersed
        assert!(result.dispersion > 0.0);

        println!("FEGLM Poisson result:\n{}", result);
    }

    #[test]
    fn test_feglm_logit_single_fe() {
        let dataset = create_binary_panel();

        // Single FE should converge quickly
        let result = run_feglm(&dataset, "y", &["x"], &["firm"], GlmFamily::Logit, None).unwrap();

        assert_eq!(result.fe_dimensions.len(), 1);
        assert!(!result.coefficients.is_empty());
    }

    #[test]
    fn test_feglm_missing_column() {
        let dataset = create_binary_panel();

        let result = run_feglm(
            &dataset,
            "y",
            &["nonexistent"],
            &["firm"],
            GlmFamily::Logit,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_feglm_no_fe_columns() {
        let dataset = create_binary_panel();

        let result = run_feglm(&dataset, "y", &["x"], &[], GlmFamily::Logit, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_feglm_display() {
        let dataset = create_binary_panel();
        let result = run_feglm(&dataset, "y", &["x"], &["firm"], GlmFamily::Logit, None).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("FEGLM"));
        assert!(display.contains("Binomial (logit)"));
        assert!(display.contains("firm"));
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Validation Tests: Comparison with R's alpaca::feglm()
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Deterministic data; not an R reference. The R code below shows the kind of DGP
    /// it imitates; no test compares against its output:
    /// ```r
    /// library(alpaca)
    /// set.seed(42)
    /// n <- 100
    ///
    /// # Create panel data with known DGP
    /// data <- data.frame(
    ///   id = factor(rep(1:10, each = 10)),
    ///   time = factor(rep(1:10, 10))
    /// )
    ///
    /// # X variable with variation
    /// data$x <- rnorm(n)
    ///
    /// # Fixed effects
    /// id_eff <- rnorm(10, sd = 0.5)[data$id]
    /// time_eff <- rnorm(10, sd = 0.3)[data$time]
    ///
    /// # True coefficient: beta = 1.0
    /// eta <- 1.0 * data$x + id_eff + time_eff
    /// data$y <- rbinom(n, 1, plogis(eta))
    ///
    /// # Estimate
    /// fit <- feglm(y ~ x | id + time, data = data, family = binomial("logit"))
    /// summary(fit)
    /// ```
    fn create_validation_dataset() -> Dataset {
        // Deterministic data; not an R reference (the R code above is illustrative)
        let n = 50;
        let n_id = 5;
        let n_time = 10;

        let mut id = Vec::with_capacity(n);
        let mut time = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        // Generate deterministic but varied data
        for i in 0..n_id {
            for t in 0..n_time {
                id.push((i + 1) as i64);
                time.push((t + 1) as i64);

                // Pseudo-random x based on position
                let x_val = ((i * 17 + t * 31) % 100) as f64 / 50.0 - 1.0;
                x.push(x_val);

                // y based on logistic model with FE
                let id_eff = (i as f64 - 2.0) * 0.5;
                let time_eff = (t as f64 - 4.5) * 0.2;
                let eta = 1.0 * x_val + id_eff + time_eff;
                let p = 1.0 / (1.0 + (-eta).exp());

                // Deterministic threshold based on position
                let threshold = ((i * 13 + t * 7) % 100) as f64 / 100.0;
                y.push(if p > threshold { 1.0 } else { 0.0 });
            }
        }

        let df = df! {
            "id" => id,
            "time" => time,
            "x" => x,
            "y" => y,
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    /// A plausibility band, not an R comparison: logit/probit coefficient ratio near 1.6.
    fn test_feglm_logit_probit_ratio_plausible() {
        let dataset = create_validation_dataset();

        let logit_result = run_feglm(
            &dataset,
            "y",
            &["x"],
            &["id", "time"],
            GlmFamily::Logit,
            None,
        )
        .unwrap();

        let probit_result = run_feglm(
            &dataset,
            "y",
            &["x"],
            &["id", "time"],
            GlmFamily::Probit,
            None,
        )
        .unwrap();

        // Probit coefficients should be approximately logit/1.6
        // This is because logistic ≈ Φ(z/1.6)
        let ratio = logit_result.coefficients[0] / probit_result.coefficients[0];
        assert!(
            ratio.abs() > 1.2 && ratio.abs() < 2.2,
            "Logit/Probit ratio should be around 1.6, got {}",
            ratio
        );

        // Both should have same sign
        assert!(
            logit_result.coefficients[0].signum() == probit_result.coefficients[0].signum(),
            "Logit and Probit should have same coefficient sign"
        );
    }

    // R 4.3.3, fixest 0.14.2 reference values:
    //   dpois(0, 2.5, log = TRUE)            # -2.5
    //   dpois(3, 2.5, log = TRUE)            # -1.5428872736055899
    //   dpois(17, 12.3, log = TRUE)          # -3.1418859897045666
    //   dpois(1e6, 999000, log = TRUE)       # -8.327027479053644
    //   2.5 * log(3) - 3 - lgamma(3.5)       # -1.4544428806767997 (fixest convention; dpois gives -Inf)
    //   dpois(1e12, 1e12 + 1e6, log = TRUE)  # -15.234448757835947
    //   library(fixest)
    //   i <- 0:39
    //   d <- data.frame(id = i %/% 8, x = ((i * 7) %% 11) / 10)
    //   d$y1 <- ((i * 37) %% 23) + 1 + 0.25 * (i %% 3)
    //   d$y2 <- 1e12 + ((i * 7919) %% 1000003) * 1e6
    //   fepois(y1 ~ x | id, d)$ll_null                              # -165.74049709539213
    //   sum(d$y2 * log(mean(d$y2)) - mean(d$y2) - lgamma(d$y2 + 1))  # -145041993560.08203
    //   fepois(y2 ~ x | id, d)$ll_null                              # -145041993560 (same, printed)

    /// Non-integer y uses fixest's y*log(mu) - mu - lgamma(y + 1); integer y matches dpois.
    #[test]
    fn test_validate_poisson_log_lik_against_r() {
        use statrs::function::gamma::ln_gamma;
        let cases: [(f64, f64, f64); 5] = [
            (2.5, 3.0, -1.4544428806767997),
            (0.0, 2.5, -2.5),
            (3.0, 2.5, -1.5428872736055899),
            (17.0, 12.3, -3.1418859897045666),
            (1e6, 999000.0, -8.327027479053644),
        ];
        for (y, mu, want) in cases {
            let got = GlmFamily::Poisson.log_lik(y, mu);
            let scale = (y * mu.ln()).abs() + mu + ln_gamma(y + 1.0);
            assert!(
                (got - want).abs() <= 1e-12 * scale.max(1.0),
                "log_lik({y}, {mu}) = {got}, R {want}"
            );
        }
    }

    fn ppml_panel(y_of: impl Fn(usize) -> f64) -> Dataset {
        let id: Vec<i64> = (0..40).map(|i| (i / 8) as i64).collect();
        let x: Vec<f64> = (0..40).map(|i| ((i * 7) % 11) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..40).map(y_of).collect();
        Dataset::new(df! { "id" => id, "x" => x, "y" => y }.unwrap())
    }

    #[test]
    fn test_validate_feglm_poisson_null_loglik_fixest_non_integer() {
        // fepois(y1 ~ x | id, d)$ll_null = -165.74049709539213
        let ds = ppml_panel(|i| ((i * 37) % 23) as f64 + 1.0 + 0.25 * (i % 3) as f64);
        let r = run_feglm(&ds, "y", &["x"], &["id"], GlmFamily::Poisson, None).unwrap();
        let want = -165.74049709539213;
        assert!(
            (r.log_likelihood_null - want).abs() <= 1e-10 * want.abs(),
            "got {}",
            r.log_likelihood_null
        );
    }

    /// The column and count of a `NonFiniteValues` error.
    fn non_finite(res: EconResult<FeglmResult>) -> (String, usize) {
        match res {
            Err(EconError::NonFiniteValues { column, count }) => (column, count),
            Err(e) => panic!("expected NonFiniteValues, got: {e}"),
            Ok(_) => panic!("expected NonFiniteValues, got Ok"),
        }
    }

    /// Coefficients, SEs and log-likelihoods of two fits agree to 1e-12.
    fn assert_same_fit(a: &FeglmResult, b: &FeglmResult) {
        let tol = p2a_validation::Tol::new(1e-12, 1e-12);
        assert_eq!(a.variables, b.variables);
        for (x, y) in a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .chain([&a.log_likelihood])
            .zip(
                b.coefficients
                    .iter()
                    .chain(&b.std_errors)
                    .chain([&b.log_likelihood]),
            )
        {
            assert!(tol.allows(*x, *y), "{x} vs {y}");
        }
    }

    /// The 40-row id / x / y frame of `ppml_panel`, without row `drop` when given.
    fn frame(x: &[f64], y: &[f64], drop: Option<usize>) -> Dataset {
        let keep = |i: &usize| Some(*i) != drop;
        let id: Vec<i64> = (0..40).filter(keep).map(|i| (i / 8) as i64).collect();
        let x: Vec<f64> = (0..40).filter(keep).map(|i| x[i]).collect();
        let y: Vec<f64> = (0..40).filter(keep).map(|i| y[i]).collect();
        Dataset::new(df! { "id" => id, "x" => x, "y" => y }.unwrap())
    }

    /// A NaN in y or a regressor is a missing cell (fixest drops it): the
    /// fit equals the fit without that row and reports it. An infinite value is
    /// refused by name, before the family's checks.
    #[test]
    fn test_feglm_non_finite_outcome() {
        let x: Vec<f64> = (0..40).map(|i| ((i * 7) % 11) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..40).map(|i| 3.0 + (i % 4) as f64).collect();
        let fit = |ds: &Dataset, family| run_feglm(ds, "y", &["x"], &["id"], family, None);

        let mut y_nan = y.clone();
        y_nan[5] = f64::NAN;
        let r = fit(&frame(&x, &y_nan, None), GlmFamily::Poisson).unwrap();
        assert_same_fit(
            &r,
            &fit(&frame(&x, &y, Some(5)), GlmFamily::Poisson).unwrap(),
        );
        assert_eq!(r.missing.n_dropped, 1);
        assert_eq!(r.missing.by_column, [("y".to_string(), 1)]);

        let mut y_inf = y.clone();
        y_inf[5] = f64::INFINITY;
        assert_eq!(
            non_finite(fit(&frame(&x, &y_inf, None), GlmFamily::Logit)),
            ("y".to_string(), 1)
        );

        let mut x_nan = x.clone();
        x_nan[7] = f64::NAN;
        let r = fit(&frame(&x_nan, &y, None), GlmFamily::Poisson).unwrap();
        assert_same_fit(
            &r,
            &fit(&frame(&x, &y, Some(7)), GlmFamily::Poisson).unwrap(),
        );
        assert_eq!(r.missing.by_column, [("x".to_string(), 1)]);

        let mut x_inf = x.clone();
        x_inf[7] = f64::INFINITY;
        assert_eq!(
            non_finite(fit(&frame(&x_inf, &y, None), GlmFamily::Poisson)),
            ("x".to_string(), 1)
        );
    }

    /// FEGLM drops rows with a missing outcome, regressor or fixed-effect id, as fixest
    /// does: the fit on Grunfeld with missing cells equals the fit on its 193 complete
    /// rows. ECON-37: missing ids used to become level 0 (or "") and merge into a real
    /// level. Not pinned to fixest: FEGLM's numbers carry ECON-20/66/67/85 (Tier 4).
    #[test]
    fn test_validate_na_feglm_omit_and_missing_fe_ids() {
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        assert_eq!(
            case.dataset,
            "validation/datasets/generated/grunfeld_na.csv"
        );
        let cc = Dataset::new(ds.df().drop_nulls::<String>(None).unwrap());
        assert_eq!(cc.df().height(), 193);
        let subset = [
            "inv".to_string(),
            "value".to_string(),
            "capital".to_string(),
        ];
        let xy = Dataset::new(ds.df().drop_nulls(Some(&subset)).unwrap());
        let fit = |d: &Dataset, fe: &[&str]| {
            run_feglm(
                d,
                "inv",
                &["value", "capital"],
                fe,
                GlmFamily::Gaussian,
                None,
            )
        };
        let reference = fit(&cc, &["firm_id", "year"]).unwrap();
        assert!(reference.missing.is_empty());
        let r = fit(&ds, &["firm_id", "year"]).unwrap();
        assert_same_fit(&r, &reference);
        assert_eq!((r.n_obs, r.missing.n_dropped), (193, 7));
        let by: Vec<(String, usize)> = [("inv", 2), ("value", 1), ("firm_id", 2), ("year", 2)]
            .iter()
            .map(|(c, k)| (c.to_string(), *k))
            .collect();
        assert_eq!(r.missing.by_column, by);
        assert_same_fit(&fit(&xy, &["firm_id", "year"]).unwrap(), &reference);
        assert_same_fit(&fit(&ds, &["firm_name", "year"]).unwrap(), &reference);
        match run_feglm(
            &ds,
            "firm_name",
            &["value"],
            &["firm_id"],
            GlmFamily::Gaussian,
            None,
        ) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "firm_name"),
            Err(e) => panic!("expected NonNumericColumn, got {e}"),
            Ok(_) => panic!("expected NonNumericColumn, got Ok"),
        }
        match fit(&ds, &["nope"]) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            Err(e) => panic!("expected ColumnNotFound, got {e}"),
            Ok(_) => panic!("expected ColumnNotFound, got Ok"),
        }
    }

    /// The closed form is O(1): y = 1e12 used to loop 10^12 times. The 1e-14 x scale
    /// tolerance (about 0.55 here) is the cancellation floor of the closed form, which
    /// fixest shares; R's dpois uses Loader's saddle point.
    #[test]
    fn test_poisson_log_lik_large_counts() {
        use statrs::function::gamma::ln_gamma;
        let (y, mu) = (1e12, 1e12 + 1e6);
        let got = GlmFamily::Poisson.log_lik(y, mu);
        let scale = (y * mu.ln()).abs() + mu + ln_gamma(y + 1.0);
        assert!(
            (got - (-15.234448757835947)).abs() <= 1e-14 * scale,
            "got {got}"
        );
        for v in [1e15, 1e18] {
            assert!(GlmFamily::Poisson.log_lik(v, v).is_finite());
        }
    }

    #[test]
    fn test_validate_feglm_poisson_null_loglik_fixest_large_counts() {
        // sum(d$y2 * log(mean(d$y2)) - mean(d$y2) - lgamma(d$y2 + 1)) = -145041993560.08203
        let ds = ppml_panel(|i| 1e12 + ((i * 7919) % 1_000_003) as f64 * 1e6);
        let r = run_feglm(&ds, "y", &["x"], &["id"], GlmFamily::Poisson, None).unwrap();
        let want = -145041993560.08203;
        assert!(
            (r.log_likelihood_null - want).abs() <= 1e-10 * want.abs(),
            "got {}",
            r.log_likelihood_null
        );
        assert!(r.log_likelihood.is_finite());
    }

    #[test]
    fn test_feglm_rejects_invalid_iteration_controls() {
        let ds = create_count_panel();
        let run = |c: FeglmConfig| {
            run_feglm(&ds, "count", &["x"], &["firm"], GlmFamily::Poisson, Some(c))
        };
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            expect_invalid(
                run(FeglmConfig {
                    tolerance,
                    ..Default::default()
                }),
                "tolerance",
            );
        }
        expect_invalid(
            run(FeglmConfig {
                max_iter: 0,
                ..Default::default()
            }),
            "max_iter",
        );
        expect_invalid(
            run(FeglmConfig {
                map_tolerance: 0.0,
                ..Default::default()
            }),
            "map_tolerance",
        );
        expect_invalid(
            run(FeglmConfig {
                map_max_iter: 0,
                ..Default::default()
            }),
            "map_max_iter",
        );
        expect_invalid(
            run(FeglmConfig {
                weight_min: f64::NAN,
                ..Default::default()
            }),
            "weight_min",
        );
    }

    #[test]
    fn test_feglm_stops_when_cancelled() {
        let ds = create_count_panel();
        expect_cancelled(with_budget(cancelled(), || {
            run_feglm(&ds, "count", &["x"], &["firm"], GlmFamily::Poisson, None)
        }));
    }

    /// The weighted MAP as it was before T3-PERF5 (a fresh vector per factor pass, the group
    /// weight sums recomputed in every pass), the reference of the in-place projector.
    fn weighted_demean_map_reference(
        data: &Array1<f64>,
        weights: &Array1<f64>,
        factors: &[FactorInfo],
        tolerance: f64,
        max_iter: usize,
        accelerate: bool,
    ) -> (Array1<f64>, usize, f64, bool) {
        let by_factor = |data: &Array1<f64>, factor: &FactorInfo| -> Array1<f64> {
            let mut sums = vec![0.0; factor.n_levels];
            let mut w_sums = vec![0.0; factor.n_levels];
            for i in 0..data.len() {
                let g = factor.ids[i];
                sums[g] += data[i] * weights[i];
                w_sums[g] += weights[i];
            }
            let means: Vec<f64> = sums
                .iter()
                .zip(&w_sums)
                .map(|(&s, &w)| if w > 1e-10 { s / w } else { 0.0 })
                .collect();
            Array1::from_shape_fn(data.len(), |i| data[i] - means[factor.ids[i]])
        };
        if factors.len() == 1 {
            return (by_factor(data, &factors[0]), 1, 0.0, true);
        }
        let w_norm = |v: &Array1<f64>| -> f64 {
            v.iter()
                .zip(weights.iter())
                .map(|(&c, &w)| w * c * c)
                .sum::<f64>()
                .sqrt()
        };
        let floor = MAP_SCALE_FLOOR * w_norm(data);
        let (mut current, mut prev, mut prev_prev) = (data.clone(), data.clone(), data.clone());
        let mut final_change = f64::MAX;
        for iter in 1..=max_iter {
            if iter > 1 {
                prev_prev.assign(&prev);
            }
            prev.assign(&current);
            for factor in factors {
                current = by_factor(&current, factor);
            }
            let (mut delta, mut norm_w) = (0.0f64, 0.0f64);
            for ((&c, &p), &w) in current.iter().zip(prev.iter()).zip(weights.iter()) {
                delta += w * (c - p).powi(2);
                norm_w += w * c * c;
            }
            let delta = delta.sqrt();
            let scale = norm_w.sqrt().max(floor).max(f64::MIN_POSITIVE);
            final_change = delta / scale;
            if delta <= tolerance * scale {
                return (current, iter, final_change, true);
            }
            if accelerate && iter > 2 {
                let (mut num, mut den, mut norm_sq) = (0.0f64, 0.0f64, 0.0f64);
                for ((&c, &p), &pp) in current.iter().zip(prev.iter()).zip(prev_prev.iter()) {
                    num += (c - p) * (p - pp);
                    den += (p - pp) * (p - pp);
                    norm_sq += c * c;
                }
                if den > (f64::EPSILON * norm_sq.sqrt()).powi(2) {
                    let alpha = (num / den).clamp(0.0, 1.0);
                    if alpha > 0.0 && alpha < 1.0 {
                        for (c, &p) in current.iter_mut().zip(prev.iter()) {
                            let d = *c - p;
                            *c = p + d * (1.0 + alpha);
                        }
                    }
                }
            }
        }
        (current, max_iter, final_change, false)
    }

    /// T3-PERF5: the weighted MAP through `WeightedProjector` (in place, group weight sums
    /// once per set of weights) is bit-identical to the per-pass reference: sweeps, change,
    /// convergence and every value, one and two (unbalanced) factors, with and without
    /// acceleration, including a sweep limit that stops it early.
    #[test]
    fn test_weighted_projector_matches_reference_exactly() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let n = 600;
        let ids1: Vec<usize> = (0..n).map(|_| rng.gen_range(0..37)).collect();
        let ids2: Vec<usize> = (0..n).map(|i| (i * 7 + rng.gen_range(0..3)) % 23).collect();
        let info = |name: &str, ids: Vec<usize>, levels: usize| {
            let mut counts = vec![0; levels];
            ids.iter().for_each(|&g| counts[g] += 1);
            FactorInfo {
                name: name.to_string(),
                n_levels: levels,
                ids,
                counts,
            }
        };
        let factors = [info("a", ids1, 37), info("b", ids2, 23)];
        let weights = Array1::from_shape_fn(n, |i| {
            if i % 50 == 0 {
                0.0
            } else {
                0.05 + rng.r#gen::<f64>() * 0.2
            }
        });
        let data = Array1::from_shape_fn(n, |i| {
            (i as f64 * 0.01).sin() * 3.0 + factors[0].ids[i] as f64 * 0.1 + rng.r#gen::<f64>()
        });
        for (f, max_iter) in [(&factors[..1], 100), (&factors[..], 100), (&factors[..], 3)] {
            for accelerate in [false, true] {
                let mut proj = WeightedProjector::new(&weights, f);
                let got =
                    weighted_demean_map(&data, &mut proj, 1e-10, max_iter, accelerate).unwrap();
                let want =
                    weighted_demean_map_reference(&data, &weights, f, 1e-10, max_iter, accelerate);
                assert_eq!((got.1, got.3), (want.1, want.3), "{} factors", f.len());
                assert_eq!(got.2.to_bits(), want.2.to_bits());
                for (a, b) in got.0.iter().zip(want.0.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{} factors {accelerate}", f.len());
                }
                // the projector serves another column with the same weights unchanged
                let again =
                    weighted_demean_map(&data, &mut proj, 1e-10, max_iter, accelerate).unwrap();
                assert!(
                    again
                        .0
                        .iter()
                        .zip(got.0.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    #[test]
    fn test_weighted_demean_map_stops_when_cancelled() {
        // the two FactorInfo literals of hdfe's test_demean_map_two_factors, data 1..=6, unit weights
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weights = Array1::from_elem(6, 1.0);
        let factor1 = FactorInfo {
            name: "entity".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };
        let factor2 = FactorInfo {
            name: "time".to_string(),
            n_levels: 3,
            ids: vec![0, 1, 2, 0, 1, 2],
            counts: vec![2, 2, 2],
        };
        expect_cancelled(with_budget(cancelled(), || {
            let factors = [factor1, factor2];
            let mut proj = WeightedProjector::new(&weights, &factors);
            weighted_demean_map(&data, &mut proj, 1e-8, 1000, true)
        }));
    }

    // =========================================================================
    // T3-WP17: relative MAP rule and deviance stopping rule (ECON-69),
    // absorbed regressors (ECON-72)
    // =========================================================================

    #[test]
    fn test_validate_feglm_gaussian_converges_at_scale_1e9() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hdfe_map_scale");
        let r = run_feglm(
            &ds,
            "y_big",
            &["x"],
            &["firm", "year"],
            GlmFamily::Gaussian,
            None,
        )
        .unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        // Gaussian coefficients are exact after one IRLS step (MAP to relative 1e-8; R
        // replica 6.7e-16 from fixest); the SE carries ECON-20 (one-pass FE recovery).
        Check::new(&case)
            .named(
                "coef_feglm_big",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-8),
            )
            .finish();
    }

    #[test]
    fn test_validate_feglm_absorbed_regressors() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_absorbed_regressors");
        let r = run_feglm(
            &ds,
            "y",
            &["x1", "x2", "z", "w"],
            &["id", "t"],
            GlmFamily::Gaussian,
            None,
        )
        .unwrap();
        assert_eq!(r.dropped, vec!["z".to_string(), "w".to_string()]);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("'z', 'w' dropped: no variation left"))
        );
        // MAP exact after 2 sweeps on this balanced design: README tolerances
        Check::new(&case)
            .named("feglm2_coef", &r.variables, &r.coefficients, Tol::coef(180))
            .finish();
        let e = run_feglm(&ds, "y", &["z"], &["id", "t"], GlmFamily::Gaussian, None).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.starts_with("no regressor left to estimate: 'z' dropped")),
            "{e:?}"
        );
    }

    #[test]
    fn test_feglm_irls_limit_is_reported() {
        let config = FeglmConfig {
            max_iter: 1,
            ..FeglmConfig::default()
        };
        let r = run_feglm(
            &create_binary_panel(),
            "y",
            &["x"],
            &["firm"],
            GlmFamily::Logit,
            Some(config),
        )
        .unwrap();
        assert!(!r.converged);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("FEGLM IRLS did not converge after 1 iteration")),
            "{:?}",
            r.warnings
        );
        assert!(format!("{r}").contains("Converged: no (1 iteration"));
    }

    /// fixest::feglm(gaussian) on feglm_families: the coefficients, the residual df and n.
    #[test]
    fn test_validate_feglm_gaussian_families() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("feglm_families");
        let r = run_feglm(
            &ds,
            "y_gauss",
            &["x", "x2"],
            &["id", "time"],
            GlmFamily::Gaussian,
            None,
        )
        .unwrap();
        // deviation ECON-20: gauss_se_lm, gauss_deviance, gauss_dispersion (eta's one-pass FE recovery)
        // deviation ECON-66: gauss_loglik_glm
        Check::new(&case)
            .named("gauss_coef", &r.variables, &r.coefficients, Tol::coef(480))
            .scalar("gauss_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
    }

    /// Logit, probit and Poisson on feglm_families either converge or say that they did
    /// not; the coefficients are finite.
    #[test]
    fn test_validate_feglm_families_report_convergence() {
        let (_case, ds) = crate::test_support::load_case("feglm_families");
        // deviation ECON-20 (coefficients; fixest drops 16 rows, the entities whose outcome is
        // all 0 or all 1, for logit/probit: logit_nobs 464 of 480), ECON-67 (poisson SE)
        for (y, family) in [
            ("y_bin", GlmFamily::Logit),
            ("y_bin", GlmFamily::Probit),
            ("y_cnt", GlmFamily::Poisson),
        ] {
            let r = run_feglm(&ds, y, &["x", "x2"], &["id", "time"], family, None)
                .unwrap_or_else(|e| panic!("{family:?}: {e}"));
            assert!(r.iterations <= r.max_iter, "{family:?}");
            if !r.converged {
                assert!(
                    r.warnings
                        .iter()
                        .any(|w| w.starts_with("FEGLM IRLS did not converge after")),
                    "{family:?} {:?}",
                    r.warnings
                );
                assert!(r.to_string().contains("Converged: no"), "{family:?}");
            }
            assert!(
                r.coefficients.iter().all(|b| b.is_finite()),
                "{family:?} {:?}",
                r.coefficients
            );
        }
    }
}
