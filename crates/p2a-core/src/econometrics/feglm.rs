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
//! IRLS with weighted alternating projections, as `alpaca::feglm` (Stammann 2018):
//!
//! 1. Drop every fixed-effect group whose outcome is all 0 (binomial and Poisson) or
//!    all 1 (binomial), repeating over all dimensions until nothing changes; singletons
//!    with a positive outcome stay (alpaca). Logit and probit outcomes must lie in [0, 1].
//! 2. Start from the constant linear predictor `link(mean((y + 0.5) / 2))` (binomial),
//!    `link(mean(y + 0.1))` (Poisson) or `mean(y)` (Gaussian), with every slope 0.
//! 3. In each IRLS iteration:
//!    a. compute the working weights w and the working residual nu = z - eta;
//!    b. demean the unscaled nu and X with weights w by alternating projections (the
//!       last iteration's demeaned X, and in the first iteration the unit-weight
//!       regressor screen's, is the warm start: `M_W (M_V X) = M_W X`);
//!    c. solve the weighted least squares of the demeaned nu on the demeaned X for the
//!       slope step, and take the step of eta, `nu - (M nu - M X step)`, which lies in the
//!       span of X and the effects;
//!    d. halve the step until the deviance is finite, the mean valid and the deviance
//!       not higher than the last one by more than the tolerance (alpaca's step halving,
//!       which asks for a decrease, would refuse the last Newton step near the estimate and
//!       stop short of glm's); at most 50 halvings, and when none is accepted the iterate
//!       stays and the fit has converged, with a warning when the step it could not
//!       take (or took at below 2^-20) was not negligible.
//! 4. Standard errors are the inverse Hessian at the final weights (dispersion 1 for
//!    binomial and Poisson, deviance / df for Gaussian).
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
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::regression::ols::{NormalInverse, normal_equations_inverse};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, deviance_change, notes_text, sci};
use crate::traits::estimator::{
    SignificanceLevel, logistic_cdf, normal_cdf, normal_p_value_two_sided, normal_pdf,
    normal_quantile,
};

use super::hdfe::{
    FactorInfo, WeightedProjector, compute_absorbed_df, demean_matrix_map, factor_info, fe_label,
    screen_regressors, weighted_demean_map, weighted_demean_matrix_map,
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
                // The kernel -0.5 (y - μ)²; the fit's log-likelihood uses sigma^2 = SSR / n
                // (glm's logLik) instead of summing this
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
    /// The floor uses the column's uncentred norm: for a column whose level is far above
    /// its variation within the groups (values near 1e9 that vary by about 1), the sweeps
    /// stop when the demeaned column is accurate to about `tolerance * 1e-6 * ||column||`.
    /// At the default tolerance 1e-8 that is within about 50 times the rounding error the
    /// stored values already carry (`f64::EPSILON * ||column||`).
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
    /// Log-likelihood at convergence (glm's `logLik`; Gaussian at sigma^2 = SSR / n)
    pub log_likelihood: f64,
    /// Log-likelihood of the intercept-only model on the rows used
    pub log_likelihood_null: f64,
    /// Deviance at convergence
    pub deviance: f64,
    /// Deviance of the intercept-only model on the rows used
    pub null_deviance: f64,
    /// McFadden's Pseudo R-squared: 1 - LL/LL_null
    pub pseudo_r_squared: f64,
    /// AIC: 2p - 2 LL, p counting the slopes, the absorbed effects (`df_absorbed`) and, for
    /// Gaussian, sigma (glm with factor dummies)
    pub aic: f64,
    /// BIC: p ln(n) - 2 LL, with the AIC's p
    pub bic: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Dispersion
    // ═══════════════════════════════════════════════════════════════════════════
    /// Dispersion parameter: 1 for binomial and Poisson, as glm, alpaca and fixest;
    /// deviance / df_resid for Gaussian
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

    /// Rows dropped because their fixed-effect group's outcome is all 0 (or, for binomial,
    /// all 1): such a group's effect has no finite estimate (alpaca and fixest drop them too)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub n_separated: usize,

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

/// glm's Gaussian log-likelihood at the ML variance `ssr / n`: `-n/2 (ln(2 pi ssr / n) + 1)`.
fn gaussian_log_lik(ssr: f64, n: f64) -> f64 {
    -0.5 * n * ((2.0 * std::f64::consts::PI * ssr / n).ln() + 1.0)
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weighted Demeaning Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// The factorization of an IRLS step's weighted demeaned design.
enum IrlsSolve {
    Normal(NormalInverse),
    Qr(Qr),
}

/// Step-halving tries per IRLS iteration (alpaca's inner loop).
const MAX_STEP_TRIES: usize = 50;

/// A fit that stops on an iteration whose step was halved below this fraction of the
/// IRLS step (2^-20), or on one where no halving improved the deviance, counts as
/// converged (alpaca) but warns when that step was not negligible: the deviance stopped
/// changing because the step vanished, for example after a demeaning that did not
/// converge, not because the score is zero.
const SMALL_STEP: f64 = 1.0 / 1_048_576.0;

/// An IRLS step of the linear predictor at most this fraction of `1 + max |eta|` is
/// negligible: at the estimate, rounding in the deviance can refuse such a step under a
/// tight tolerance, and stopping there is convergence.
const NEGLIGIBLE_STEP: f64 = 1e-6;

/// Whether `mu` is a valid mean of the family (glm's `validmu`).
fn valid_mu(family: GlmFamily, mu: f64) -> bool {
    match family {
        GlmFamily::Logit | GlmFamily::Probit => mu.is_finite() && mu > 0.0 && mu < 1.0,
        GlmFamily::Poisson => mu.is_finite() && mu > 0.0,
        GlmFamily::Gaussian => mu.is_finite(),
    }
}

/// The rows kept after dropping every fixed-effect group whose outcome is all 0 (or, for
/// binomial, all 1), repeated over all factors until none is left: such a group's effect
/// has no finite estimate (alpaca's and fixest's rule).
///
/// A worklist: each level's row, zero and one counts are computed once; a dropped row
/// decrements them in every factor, and a level whose kept rows become all 0 (all 1) is
/// queued. A group stays constant once it is, so the result is the closure the round-based
/// rule reaches, in time linear in the rows times the factors.
fn separated_rows_kept(
    y: &Array1<f64>,
    factors: &[FactorInfo],
    binomial: bool,
) -> EconResult<Vec<bool>> {
    struct Levels {
        rows: Vec<usize>,
        zeros: Vec<usize>,
        ones: Vec<usize>,
        queued: Vec<bool>,
        // the level's rows, ascending: members[start[g]..start[g + 1]]
        start: Vec<usize>,
        members: Vec<usize>,
    }
    let n = y.len();
    let constant = |lv: &Levels, g: usize| {
        lv.rows[g] > 0 && (lv.zeros[g] == lv.rows[g] || (binomial && lv.ones[g] == lv.rows[g]))
    };
    let mut keep = resource::try_vec(n, true, "FEGLM row mask")?;
    let mut levels = Vec::with_capacity(factors.len());
    let mut queue: Vec<(usize, usize)> = Vec::new();
    for (f, factor) in factors.iter().enumerate() {
        let m = factor.n_levels;
        let mut lv = Levels {
            rows: resource::try_vec(m, 0usize, "FEGLM group counts")?,
            zeros: resource::try_vec(m, 0usize, "FEGLM group counts")?,
            ones: resource::try_vec(m, 0usize, "FEGLM group counts")?,
            queued: resource::try_vec(m, false, "FEGLM group counts")?,
            start: resource::try_vec(m + 1, 0usize, "FEGLM group rows")?,
            members: resource::try_vec(n, 0usize, "FEGLM group rows")?,
        };
        for (i, &g) in factor.ids.iter().enumerate() {
            lv.rows[g] += 1;
            lv.zeros[g] += usize::from(y[i] == 0.0);
            lv.ones[g] += usize::from(y[i] == 1.0);
        }
        for g in 0..m {
            lv.start[g + 1] = lv.start[g] + lv.rows[g];
        }
        let mut next = resource::try_vec(m + 1, 0usize, "FEGLM group rows")?;
        next.copy_from_slice(&lv.start);
        for (i, &g) in factor.ids.iter().enumerate() {
            lv.members[next[g]] = i;
            next[g] += 1;
        }
        for g in 0..m {
            if constant(&lv, g) {
                lv.queued[g] = true;
                queue.push((f, g));
            }
        }
        levels.push(lv);
    }
    let mut popped = 0usize;
    while let Some((f, g)) = queue.pop() {
        if popped.is_multiple_of(1024) {
            resource::checkpoint()?;
        }
        popped += 1;
        for idx in levels[f].start[g]..levels[f].start[g + 1] {
            let i = levels[f].members[idx];
            if !keep[i] {
                continue;
            }
            keep[i] = false;
            for (h, factor) in factors.iter().enumerate() {
                let lv = &mut levels[h];
                let l = factor.ids[i];
                lv.rows[l] -= 1;
                lv.zeros[l] -= usize::from(y[i] == 0.0);
                lv.ones[l] -= usize::from(y[i] == 1.0);
                if !lv.queued[l] && constant(lv, l) {
                    lv.queued[l] = true;
                    queue.push((h, l));
                }
            }
        }
    }
    Ok(keep)
}

/// `factor` on the rows `rows` (ascending): the levels still used, renumbered in their
/// old order, and their counts. A dimension may keep a single level; it then absorbs
/// only the constant.
fn reduce_factor(factor: &FactorInfo, rows: &[usize]) -> EconResult<FactorInfo> {
    let mut counts = resource::try_vec(factor.n_levels, 0usize, "FEGLM group counts")?;
    for &i in rows {
        counts[factor.ids[i]] += 1;
    }
    let mut new_id = resource::try_vec(factor.n_levels, usize::MAX, "FEGLM group counts")?;
    let mut kept_counts = Vec::new();
    for (g, &c) in counts.iter().enumerate() {
        if c > 0 {
            new_id[g] = kept_counts.len();
            kept_counts.push(c);
        }
    }
    let mut ids = resource::try_vec(rows.len(), 0usize, "FEGLM group ids")?;
    for (dst, &i) in ids.iter_mut().zip(rows) {
        *dst = new_id[factor.ids[i]];
    }
    Ok(FactorInfo {
        name: factor.name.clone(),
        n_levels: kept_counts.len(),
        ids,
        counts: kept_counts,
    })
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

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Generalized Linear Model with High-Dimensional Fixed Effects.
///
/// Uses IRLS (Iteratively Reweighted Least Squares) combined with the Method of
/// Alternating Projections for efficient absorption of multiple fixed effects: each step
/// demeans the working residual and X with the IRLS weights and takes alpaca's update with
/// step halving (see the module docs). Groups whose outcome is all 0 (or all 1 for
/// binomial) are dropped first, until none is left, and reported in `n_separated`.
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
    let n_positive_all = y.iter().filter(|&&v| v >= 0.5).count();
    match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            // The separated-group rule and the valid-mean check assume a binomial mean
            if let Some(&bad) = y.iter().find(|&&v| !(0.0..=1.0).contains(&v)) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Dependent variable '{y_col}' must lie in [0, 1] for {family} \
                         (found {bad})"
                    ),
                });
            }
            if n_positive_all == 0 || n_positive_all == y.len() {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Dependent variable '{}' must be binary with both 0 and 1 values for {}. \
                         Found {} ones out of {} observations.",
                        y_col,
                        family,
                        n_positive_all,
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
    let all_names = design.column_names;
    let n_complete = y.len();

    // Factor information for each FE dimension (levels in R factor order)
    let factors: Vec<FactorInfo> = fe_cols
        .iter()
        .map(|col| factor_info(col, cols.factor(col)))
        .collect::<EconResult<Vec<_>>>()?;

    // Groups whose outcome is all 0 (or all 1 for binomial) have no finite MLE for their
    // effect; they are dropped until no such group is left (alpaca, fixest).
    let mut warnings: Vec<String> = Vec::new();
    let (y, x_all, factors, n_separated) = match family {
        GlmFamily::Logit | GlmFamily::Probit | GlmFamily::Poisson => {
            let binomial = family != GlmFamily::Poisson;
            let keep = separated_rows_kept(&y, &factors, binomial)?;
            let n_kept = keep.iter().filter(|&&k| k).count();
            if n_kept == 0 {
                let outcome = if binomial { "all 0 or all 1" } else { "all 0" };
                return Err(EconError::InsufficientData {
                    required: 1,
                    provided: 0,
                    context: format!(
                        "every row is in a fixed-effect group whose outcome is {outcome}"
                    ),
                });
            }
            if n_kept == n_complete {
                (y, design.data, factors, 0)
            } else {
                let dropped = n_complete - n_kept;
                let outcome = if binomial { "all 0 or all 1" } else { "all 0" };
                warnings.push(format!(
                    "{dropped} of {n_complete} rows dropped: their fixed-effect group's outcome \
                     is {outcome} (alpaca and fixest drop them too)"
                ));
                let rows: Vec<usize> = (0..n_complete).filter(|&i| keep[i]).collect();
                let mut y_kept = resource::try_vec(n_kept, 0.0, "FEGLM outcome")?;
                for (dst, &i) in y_kept.iter_mut().zip(&rows) {
                    *dst = y[i];
                }
                let kx = design.data.ncols();
                let mut x_kept = resource::try_zeros2(n_kept, kx, "FEGLM design")?;
                for (r, &i) in rows.iter().enumerate() {
                    x_kept.row_mut(r).assign(&design.data.row(i));
                }
                let reduced = factors
                    .iter()
                    .map(|f| reduce_factor(f, &rows))
                    .collect::<EconResult<Vec<_>>>()?;
                (Array1::from(y_kept), x_kept, reduced, dropped)
            }
        }
        GlmFamily::Gaussian => (y, design.data, factors, 0),
    };
    let n = y.len();
    let n_positive = match family {
        GlmFamily::Poisson => y.iter().filter(|&&v| v > 0.0).count(),
        _ => y.iter().filter(|&&v| v >= 0.5).count(),
    };

    // Demeaning steps that stopped at map_max_iter: (step, relative change)
    let mut map_stalls: Vec<(String, f64)> = Vec::new();

    // Regressors absorbed by the effects or collinear after them are not estimated.
    // Rank against the fixed effects does not depend on positive weights
    // (rank(M_D^W X) = rank([X D]) - rank(D)), so one unit-weight screen covers every
    // IRLS step; its demeaned columns seed the first weighted demeaning
    // (M_W (M_1 X) = M_W X).
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
    warnings.extend(screen.warnings.iter().cloned());
    let x = if screen.kept.len() == x_all.ncols() {
        x_all
    } else {
        x_all.select(Axis(1), &screen.kept)
    };
    let var_names: Vec<String> = screen.kept.iter().map(|&j| all_names[j].clone()).collect();
    let k = screen.kept.len();

    // Absorbed degrees of freedom: the connected components of the effects, as the linear
    // HDFE estimator counts them
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
    // Initialize: alpaca's constant start, which lies in the span of the effects
    // ═══════════════════════════════════════════════════════════════════════════
    let y_mean = y.mean().unwrap_or(0.0);
    let eta_start = match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            family.link(y.iter().map(|&v| (v + 0.5) / 2.0).sum::<f64>() / n as f64)
        }
        GlmFamily::Poisson => family.link(y_mean + 0.1),
        GlmFamily::Gaussian => y_mean,
    };
    let mut eta: Array1<f64> = Array1::from_elem(n, eta_start);
    let mut beta: Array1<f64> = Array1::zeros(k);

    let deviance_of = |mu: &Array1<f64>| -> f64 {
        y.iter()
            .zip(mu.iter())
            .map(|(&yi, &m)| family.deviance(yi, m))
            .sum()
    };
    // μ is kept in step with η: computed once per update of η and used by the deviance,
    // the next iteration and the final statistics.
    let mut mu: Array1<f64> = eta.mapv(|e| family.inv_link(e));
    let mut dev_old = deviance_of(&mu);

    // The demeaned design of the last iteration (warm start of the next) and the row-scaled
    // buffer of the weighted solve, allocated once
    let mut mx = screen.x;
    let mut xs = resource::try_zeros2(n, k, "FEGLM weighted design")?;
    let mut weights = resource::try_vec(n, 0.0, "FEGLM working weights")?;
    // The weights `mx` was demeaned with. When the next weights are equal to them (value
    // equality), `mx` is already the result. The screen demeaned with unit weights, which are
    // Gaussian's IRLS weights in every iteration, so a Gaussian fit starts from them and
    // demeans X once; the other families start with none.
    let mut mx_weights: Option<Vec<f64>> = if family == GlmFamily::Gaussian {
        Some(resource::try_vec(n, 1.0, "FEGLM working weights")?)
    } else {
        None
    };
    // The factorization of the weighted demeaned design `xs`, kept while `mx` and the weights
    // repeat (Gaussian: every iteration and the final information matrix)
    let mut solve: Option<IrlsSolve> = None;
    let mut nu = resource::try_vec(n, 0.0, "FEGLM working residual")?;
    let mut nu_work = resource::try_vec(n, 0.0, "FEGLM working residual")?;
    let mut ys = Array1::from(resource::try_vec(n, 0.0, "FEGLM weighted residual")?);

    // ═══════════════════════════════════════════════════════════════════════════
    // IRLS Loop (alpaca's update with step halving)
    // ═══════════════════════════════════════════════════════════════════════════
    let mut iterations = 0;
    let mut converged = false;
    let mut final_change = f64::NAN;

    for iter in 1..=config.max_iter {
        resource::checkpoint()?;
        iterations = iter;

        // 1. Working weights and the working residual nu = z - eta
        for i in 0..n {
            weights[i] = family.working_weight(eta[i], mu[i]).max(config.weight_min);
            nu[i] = family.working_response(y[i], eta[i], mu[i]) - eta[i];
        }

        // 2. Weighted demeaning of the unscaled X (warm-started from the last iteration's)
        //    and of nu, with the IRLS weights
        let mut proj = WeightedProjector::new(&weights, &factors);
        let (mut x_change, mut x_conv) = (0.0, true);
        if mx_weights.as_deref() != Some(&weights[..]) {
            solve = None;
            let (mx_new, _, change, conv) = weighted_demean_matrix_map(
                mx.view(),
                &mut proj,
                config.map_tolerance,
                config.map_max_iter,
                config.accelerate,
            )?;
            (mx, x_change, x_conv) = (mx_new, change, conv);
            if mx_weights.is_none() {
                mx_weights = Some(resource::try_vec(n, 0.0, "FEGLM working weights")?);
            }
            if let Some(w) = mx_weights.as_mut() {
                w.copy_from_slice(&weights);
            }
        }
        // nu's demeaning works in the buffer the last iteration's Mnu came back in
        nu_work.copy_from_slice(&nu);
        let (mnu, _, nu_change, nu_conv) = weighted_demean_map(
            std::mem::take(&mut nu_work),
            &mut proj,
            config.map_tolerance,
            config.map_max_iter,
            config.accelerate,
        )?;
        if !nu_conv || !x_conv {
            map_stalls.push((format!("IRLS iteration {iter}"), nu_change.max(x_change)));
        }

        // 3. The weighted least-squares step of sqrt(w) Mnu on sqrt(w) MX (OLS's solve
        //    decision: the normal equations when well conditioned, the pivoted QR otherwise);
        //    the design and its factorization repeat while MX and the weights do
        for i in 0..n {
            ys[i] = weights[i].sqrt() * mnu[i];
        }
        if solve.is_none() {
            for i in 0..n {
                let s = weights[i].sqrt();
                for j in 0..k {
                    xs[[i, j]] = s * mx[[i, j]];
                }
            }
            solve = Some(match normal_equations_inverse(xs.view())? {
                Some(normal) => IrlsSolve::Normal(normal),
                None => {
                    let qr = Qr::decompose(xs.view(), LM_QR_TOL)?;
                    if qr.rank() < k {
                        return Err(EconError::SingularMatrix {
                            context: format!(
                                "FEGLM IRLS iteration {iter}: weighted design rank {} of {k}",
                                qr.rank()
                            ),
                            suggestion: "Check for perfect multicollinearity or separation"
                                .to_string(),
                        });
                    }
                    IrlsSolve::Qr(qr)
                }
            });
        }
        let beta_upd = match &solve {
            Some(IrlsSolve::Normal(normal)) => normal.coefficients(xs.view(), &ys),
            Some(IrlsSolve::Qr(qr)) => qr.coefficients(ys.view())?,
            None => {
                return Err(EconError::Internal(
                    "FEGLM: the weighted design was not factored".to_string(),
                ));
            }
        };

        // 4. The step of the linear predictor, in the span of X and the effects:
        //    eta_upd = nu - (Mnu - MX beta_upd)
        let mx_beta = mx.dot(&beta_upd);
        let eta_upd: Array1<f64> = Array1::from_shape_fn(n, |i| nu[i] - (mnu[i] - mx_beta[i]));
        nu_work = mnu;

        // 5. Step halving: accept the first step that gives a finite deviance, a valid mean
        //    and a deviance no higher than the last by more than the tolerance (relative).
        //    alpaca asks for a decrease of at least the tolerance, which refuses the last
        //    Newton step near the estimate and stops short of glm's.
        let mut accepted = None;
        let mut any_valid = false;
        let mut rho = 1.0f64;
        for _ in 0..MAX_STEP_TRIES {
            let eta_try: Array1<f64> = Array1::from_shape_fn(n, |i| eta[i] + rho * eta_upd[i]);
            let mu_try: Array1<f64> = eta_try.mapv(|e| family.inv_link(e));
            let dev = deviance_of(&mu_try);
            let valid = dev.is_finite()
                && eta_try.iter().all(|e| e.is_finite())
                && mu_try.iter().all(|&m| valid_mu(family, m));
            if valid {
                any_valid = true;
                if (dev - dev_old) / (0.1 + dev.abs()) <= config.tolerance {
                    accepted = Some((eta_try, mu_try, dev, rho));
                    break;
                }
            }
            rho /= 2.0;
        }
        let max_abs = |v: &Array1<f64>| v.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let step_matters = max_abs(&eta_upd) > NEGLIGIBLE_STEP * (1.0 + max_abs(&eta));
        let step_vanished = match accepted {
            Some((eta_new, mu_new, dev, rho)) => {
                beta.scaled_add(rho, &beta_upd);
                eta = eta_new;
                mu = mu_new;
                final_change = deviance_change(dev, dev_old);
                dev_old = dev;
                rho < SMALL_STEP
            }
            None if any_valid => {
                // No step improves the deviance: keep the last iterate (alpaca), which
                // the deviance rule then counts as converged
                final_change = 0.0;
                true
            }
            None => {
                return Err(EconError::ConvergenceFailure {
                    iterations: iter,
                    last_change: final_change,
                    suggestion: "the FEGLM step could not be corrected by step halving; check \
                                 for separation or extreme counts"
                        .to_string(),
                });
            }
        };

        if final_change < config.tolerance {
            converged = true;
            if step_vanished && step_matters {
                warnings.push(format!(
                    "FEGLM IRLS stopped: step halving could not improve the deviance in \
                     iteration {iter}; the estimate may not be the maximum likelihood"
                ));
            }
            break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute Final Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    // Final μ (in step with η) and log-likelihood
    let mu_final = mu;

    // Deviance (of the final μ, computed with the convergence check)
    let deviance: f64 = dev_old;
    let nf = n as f64;

    // Log-likelihood; Gaussian: glm's normal log-likelihood at sigma^2 = SSR / n
    let log_likelihood: f64 = match family {
        GlmFamily::Gaussian => gaussian_log_lik(deviance, nf),
        _ => y
            .iter()
            .zip(mu_final.iter())
            .map(|(&yi, &mi)| family.log_lik(yi, mi))
            .sum(),
    };

    // The intercept-only model on the rows used, whose mean is the outcome mean
    let (log_likelihood_null, null_deviance) = match family {
        GlmFamily::Gaussian => {
            let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            (gaussian_log_lik(tss, nf), tss)
        }
        _ => y.iter().fold((0.0, 0.0), |(ll, dev), &yi| {
            (
                ll + family.log_lik(yi, y_mean),
                dev + family.deviance(yi, y_mean),
            )
        }),
    };

    // Pseudo R-squared (McFadden's)
    let pseudo_r_squared = if log_likelihood_null.abs() > 1e-10 {
        1.0 - log_likelihood / log_likelihood_null
    } else {
        0.0
    };

    // AIC and BIC count every estimated parameter, as glm with factor dummies: the slopes,
    // the absorbed effects and, for Gaussian, sigma
    let n_params = (k + df_absorbed + usize::from(family == GlmFamily::Gaussian)) as f64;
    let aic = 2.0 * n_params - 2.0 * log_likelihood;
    let bic = n_params * nf.ln() - 2.0 * log_likelihood;

    // Dispersion: 1 for binomial and Poisson (glm, alpaca, fixest); deviance / df for Gaussian
    let dispersion = match family {
        GlmFamily::Gaussian => deviance / df_resid as f64,
        _ => 1.0,
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute Variance-Covariance Matrix
    // ═══════════════════════════════════════════════════════════════════════════

    // The inverse Hessian at the final weights; the last demeaned design is the warm start
    for i in 0..n {
        weights[i] = family
            .working_weight(eta[i], mu_final[i])
            .max(config.weight_min);
    }
    let vcov = match solve {
        // The same weights and design as the last solve: X'WX and its inverse are the ones
        // the normal equations computed
        Some(IrlsSolve::Normal(normal)) if mx_weights.as_deref() == Some(&weights[..]) => {
            normal.inverse
        }
        _ => {
            let mx_final = if mx_weights.as_deref() == Some(&weights[..]) {
                mx
            } else {
                let (mx_final, _, final_map_change, final_map_conv) = weighted_demean_matrix_map(
                    mx.view(),
                    &mut WeightedProjector::new(&weights, &factors),
                    config.map_tolerance,
                    config.map_max_iter,
                    config.accelerate,
                )?;
                if !final_map_conv {
                    map_stalls.push(("the final information matrix".to_string(), final_map_change));
                }
                mx_final
            };
            for i in 0..n {
                let s = weights[i].sqrt();
                for j in 0..k {
                    xs[[i, j]] = s * mx_final[[i, j]];
                }
            }
            let information = xtx(&xs.view());
            sym_inverse(
                information.view(),
                SingularPolicy::Error,
                "FEGLM information matrix",
            )?
            .inverse
        }
    };

    // Scale by the dispersion (Gaussian only; 1 otherwise)
    let vcov = match family {
        GlmFamily::Gaussian => &vcov * dispersion,
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

    let tiny = 10.0 * f64::EPSILON;
    match family {
        GlmFamily::Logit | GlmFamily::Probit
            if mu_final.iter().any(|&m| m < tiny || m > 1.0 - tiny) =>
        {
            warnings.push("fitted probabilities numerically 0 or 1 occurred".to_string());
        }
        GlmFamily::Poisson if mu_final.iter().any(|&m| m < tiny) => {
            warnings.push("fitted rates numerically 0 occurred".to_string());
        }
        _ => {}
    }
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
        n_separated,
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
    /// level. The numbers are pinned in feglm_mle.
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
        // replica 6.7e-16 from fixest); the SE, deviance and dispersion carry the MAP's
        // relative 1e-8 in the squared residuals.
        Check::new(&case)
            .named(
                "coef_feglm_big",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-8),
            )
            .named(
                "se_feglm_big_lm",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-6),
            )
            .scalar("deviance_big", r.deviance, Tol::iterative(1e-6))
            .scalar("dispersion_big", r.dispersion, Tol::iterative(1e-6))
            .scalar("df_resid", r.df_resid as f64, Tol::exact())
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

    /// fixest::feglm(gaussian) and lm on feglm_families: the coefficients, lm's SEs, the
    /// deviance and dispersion, glm's log-likelihood, the residual df and n.
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
        Check::new(&case)
            .named("gauss_coef", &r.variables, &r.coefficients, Tol::coef(480))
            .named("gauss_se_lm", &r.variables, &r.std_errors, Tol::se(480))
            .scalar("gauss_deviance", r.deviance, Tol::stat(480))
            .scalar("gauss_dispersion", r.dispersion, Tol::stat(480))
            .scalar("gauss_loglik_glm", r.log_likelihood, Tol::stat(480))
            .scalar("gauss_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
    }

    /// Logit, probit and Poisson on feglm_families converge, say so in neither their
    /// warnings nor their Display, and give finite coefficients.
    #[test]
    fn test_validate_feglm_families_report_convergence() {
        let (_case, ds) = crate::test_support::load_case("feglm_families");
        for (y, family) in [
            ("y_bin", GlmFamily::Logit),
            ("y_bin", GlmFamily::Probit),
            ("y_cnt", GlmFamily::Poisson),
        ] {
            let r = run_feglm(&ds, y, &["x", "x2"], &["id", "time"], family, None)
                .unwrap_or_else(|e| panic!("{family:?}: {e}"));
            assert!(r.converged, "{family:?} {:?}", r.warnings);
            assert!(r.iterations <= r.max_iter, "{family:?}");
            assert!(
                !r.warnings
                    .iter()
                    .any(|w| w.starts_with("FEGLM IRLS did not converge after")),
                "{family:?} {:?}",
                r.warnings
            );
            assert!(!r.to_string().contains("Converged: no"), "{family:?}");
            assert!(
                r.coefficients.iter().all(|b| b.is_finite()),
                "{family:?} {:?}",
                r.coefficients
            );
        }
    }

    // =========================================================================
    // T4: the FE-GLM MLE (glm with factor dummies on the rows alpaca keeps)
    // =========================================================================

    /// The controls of the R reference: glm's epsilon 1e-16 and 100 iterations, MAP
    /// tolerance 1e-12. Probit's IRLS is Fisher scoring, which converges linearly: glm at
    /// epsilon 1e-14 still stops about 1e-8 (relative) short of its own estimate.
    fn tight() -> FeglmConfig {
        let mut config = FeglmConfig::default();
        config.tolerance = 1e-16;
        config.map_tolerance = 1e-12;
        config.max_iter = 100;
        config
    }

    const SEP_WARNING_BIN: &str = "rows dropped: their fixed-effect group's outcome is all 0 or \
                                   all 1 (alpaca and fixest drop them too)";
    const SEP_WARNING_POIS: &str = "rows dropped: their fixed-effect group's outcome is all 0 (alpaca and fixest drop them too)";

    /// Fit `family` on `feglm_mle` with id + time (`<pre>2`) and with id (`<pre>1`) and pin
    /// every reported number the task has made right so far.
    fn check_mle(case_id: &str, family: GlmFamily, y: &str, pre: &str) {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case(case_id);
        for (fe, suffix) in [(&["id", "time"][..], "2"), (&["id"][..], "1")] {
            let p = format!("{pre}{suffix}");
            let r = run_feglm(&ds, y, &["x", "x2"], fe, family, Some(tight()))
                .unwrap_or_else(|e| panic!("{p}: {e}"));
            assert!(r.converged, "{p}: {:?}", r.warnings);
            let n = r.n_obs;
            Check::new(&case)
                .named(
                    &format!("{p}_coef"),
                    &r.variables,
                    &r.coefficients,
                    Tol::coef(n),
                )
                .scalar(&format!("{p}_nobs"), n as f64, Tol::exact())
                .scalar(
                    &format!("{p}_n_positive"),
                    r.n_positive as f64,
                    Tol::exact(),
                )
                .named(&format!("{p}_se"), &r.variables, &r.std_errors, Tol::se(n))
                .scalar(&format!("{p}_loglik"), r.log_likelihood, Tol::stat(n))
                .scalar(
                    &format!("{p}_loglik_null"),
                    r.log_likelihood_null,
                    Tol::stat(n),
                )
                .scalar(&format!("{p}_aic"), r.aic, Tol::stat(n))
                .scalar(&format!("{p}_bic"), r.bic, Tol::stat(n))
                .scalar(&format!("{p}_deviance"), r.deviance, Tol::stat(n))
                .scalar(&format!("{p}_null_deviance"), r.null_deviance, Tol::stat(n))
                .scalar(&format!("{p}_df_resid"), r.df_resid as f64, Tol::exact())
                .scalar(
                    &format!("{p}_df_absorbed"),
                    r.df_absorbed as f64,
                    Tol::exact(),
                )
                .finish();
            // dispersion 1 (glm, alpaca, fixest): no "Dispersion" line
            assert_eq!(r.dispersion, 1.0, "{p}");
            assert!(!r.to_string().contains("Dispersion"), "{p}");
            if family == GlmFamily::Poisson {
                assert_eq!(r.n_separated, 0, "{p}");
                assert!(r.warnings.is_empty(), "{p}: {:?}", r.warnings);
            } else {
                assert_eq!(r.n_separated, 16, "{p}");
                assert_eq!(
                    r.warnings,
                    vec![format!("16 of 480 {SEP_WARNING_BIN}")],
                    "{p}"
                );
            }
        }
    }

    #[test]
    fn test_validate_feglm_mle_logit() {
        check_mle("feglm_mle", GlmFamily::Logit, "y_bin", "logit");
    }

    #[test]
    fn test_validate_feglm_mle_probit() {
        check_mle("feglm_mle", GlmFamily::Probit, "y_bin", "probit");
    }

    #[test]
    fn test_validate_feglm_mle_poisson() {
        check_mle("feglm_mle", GlmFamily::Poisson, "y_cnt", "poisson");
    }

    /// The default controls reach the MLE to the accuracy their deviance rule (1e-8)
    /// allows: logit and Poisson coefficients to 1e-6 relative, probit's to 1e-5 (Fisher
    /// scoring converges linearly: probit1 and probit2 stop 2.8e-6 to 6.2e-6 relative, or
    /// 0.9e-6 to 1.8e-6 absolute, short of R's), every fit converged, and no warning other
    /// than the drop warning.
    #[test]
    fn test_validate_feglm_mle_default_controls() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("feglm_mle");
        for (family, y, pre) in [
            (GlmFamily::Logit, "y_bin", "logit"),
            (GlmFamily::Probit, "y_bin", "probit"),
            (GlmFamily::Poisson, "y_cnt", "poisson"),
        ] {
            for (fe, suffix) in [(&["id", "time"][..], "2"), (&["id"][..], "1")] {
                let p = format!("{pre}{suffix}");
                let r = run_feglm(&ds, y, &["x", "x2"], fe, family, None)
                    .unwrap_or_else(|e| panic!("{p}: {e}"));
                assert!(r.converged, "{p}: {:?}", r.warnings);
                let others: Vec<&String> = r
                    .warnings
                    .iter()
                    .filter(|w| !w.contains("rows dropped: their fixed-effect group"))
                    .collect();
                assert!(others.is_empty(), "{p}: {others:?}");
                assert_eq!(r.warnings.len(), usize::from(family != GlmFamily::Poisson));
                Check::new(&case)
                    .named(
                        &format!("{p}_coef"),
                        &r.variables,
                        &r.coefficients,
                        Tol::iterative(if family == GlmFamily::Probit {
                            1e-5
                        } else {
                            1e-6
                        }),
                    )
                    .finish();
            }
        }
    }

    /// alpaca's drop rule on a panel whose drops cascade: the all-1 and all-0 entities,
    /// then the period left all 0, then the entities that leaves constant, then the
    /// singleton. Poisson drops only the all-0 entity and keeps the singleton; Gaussian
    /// drops nothing.
    #[test]
    fn test_validate_feglm_separation_drop_rule() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("feglm_separation");
        let fit = |y: &str, family| {
            run_feglm(&ds, y, &["x", "x2"], &["id", "time"], family, Some(tight()))
                .unwrap_or_else(|e| panic!("{family:?}: {e}"))
        };

        let r = fit("y_bin", GlmFamily::Logit);
        assert_eq!((r.n_obs, r.n_separated, r.n_positive), (120, 61, 53));
        assert_eq!(r.fe_counts[1], 5, "period 6 has no row left");
        assert_eq!(r.warnings, vec![format!("61 of 181 {SEP_WARNING_BIN}")]);
        Check::new(&case)
            .named("logit_coef", &r.variables, &r.coefficients, Tol::coef(120))
            .scalar("logit_nobs", r.n_obs as f64, Tol::exact())
            .scalar("logit_n_dropped", r.n_separated as f64, Tol::exact())
            .scalar("logit_n_positive", r.n_positive as f64, Tol::exact())
            .named("logit_se", &r.variables, &r.std_errors, Tol::se(120))
            .scalar("logit_loglik", r.log_likelihood, Tol::stat(120))
            .scalar("logit_loglik_null", r.log_likelihood_null, Tol::stat(120))
            .scalar("logit_aic", r.aic, Tol::stat(120))
            .scalar("logit_bic", r.bic, Tol::stat(120))
            .scalar("logit_deviance", r.deviance, Tol::stat(120))
            .scalar("logit_null_deviance", r.null_deviance, Tol::stat(120))
            .scalar("logit_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("logit_df_absorbed", r.df_absorbed as f64, Tol::exact())
            .finish();

        let r = fit("y_cnt", GlmFamily::Poisson);
        assert_eq!((r.n_obs, r.n_separated), (175, 6));
        assert_eq!(r.warnings, vec![format!("6 of 181 {SEP_WARNING_POIS}")]);
        Check::new(&case)
            .named(
                "poisson_coef",
                &r.variables,
                &r.coefficients,
                Tol::coef(175),
            )
            .scalar("poisson_nobs", r.n_obs as f64, Tol::exact())
            .scalar("poisson_n_dropped", r.n_separated as f64, Tol::exact())
            .scalar("poisson_n_positive", r.n_positive as f64, Tol::exact())
            .named("poisson_se", &r.variables, &r.std_errors, Tol::se(175))
            .scalar("poisson_loglik", r.log_likelihood, Tol::stat(175))
            .scalar("poisson_loglik_null", r.log_likelihood_null, Tol::stat(175))
            .scalar("poisson_aic", r.aic, Tol::stat(175))
            .scalar("poisson_bic", r.bic, Tol::stat(175))
            .scalar("poisson_deviance", r.deviance, Tol::stat(175))
            .scalar("poisson_null_deviance", r.null_deviance, Tol::stat(175))
            .scalar("poisson_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("poisson_df_absorbed", r.df_absorbed as f64, Tol::exact())
            .finish();

        let r = fit("y_gauss", GlmFamily::Gaussian);
        assert_eq!((r.n_obs, r.n_separated), (181, 0));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        Check::new(&case)
            .named("gauss_coef", &r.variables, &r.coefficients, Tol::coef(181))
            .scalar("gauss_nobs", r.n_obs as f64, Tol::exact())
            .named("gauss_se", &r.variables, &r.std_errors, Tol::se(181))
            .scalar("gauss_loglik", r.log_likelihood, Tol::stat(181))
            .scalar("gauss_loglik_null", r.log_likelihood_null, Tol::stat(181))
            .scalar("gauss_aic", r.aic, Tol::stat(181))
            .scalar("gauss_bic", r.bic, Tol::stat(181))
            .scalar("gauss_deviance", r.deviance, Tol::stat(181))
            .scalar("gauss_null_deviance", r.null_deviance, Tol::stat(181))
            .scalar("gauss_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("gauss_df_absorbed", r.df_absorbed as f64, Tol::exact())
            .finish();
    }

    /// The 12-row binary panel is quasi-separated: glm with dummies warns "fitted
    /// probabilities numerically 0 or 1 occurred" (R 4.3.3), and so does FEGLM.
    #[test]
    fn test_feglm_fitted_probabilities_warning() {
        let r = run_feglm(
            &create_binary_panel(),
            "y",
            &["x"],
            &["firm", "year"],
            GlmFamily::Logit,
            None,
        )
        .unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "fitted probabilities numerically 0 or 1 occurred"),
            "{:?}",
            r.warnings
        );
    }

    /// Every entity's outcome is constant: no row is left to fit.
    #[test]
    fn test_feglm_all_rows_separated_is_refused() {
        let df = df! {
            "id" => [1i64, 1, 2, 2, 3, 3],
            "x" => [0.1, 0.7, -0.3, 0.4, 0.9, -0.2],
            "y" => [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        }
        .unwrap();
        let ds = Dataset::new(df);
        match run_feglm(&ds, "y", &["x"], &["id"], GlmFamily::Logit, None) {
            Err(EconError::InsufficientData { context, .. }) => assert!(
                context.contains("fixed-effect group whose outcome is all 0 or all 1"),
                "{context}"
            ),
            Err(e) => panic!("expected InsufficientData, got: {e}"),
            Ok(_) => panic!("expected InsufficientData, got Ok"),
        }
    }

    /// lm with factor dummies on a two-way design of two disconnected components: the
    /// absorbed df count the components (24, not the naive 25), so the residual df and the
    /// SEs are lm's.
    #[test]
    fn test_validate_feglm_disconnected_gaussian() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hdfe_disconnected");
        let r = run_feglm(
            &ds,
            "y",
            &["x", "z"],
            &["firm", "year"],
            GlmFamily::Gaussian,
            None,
        )
        .unwrap();
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(80))
            .named("se", &r.variables, &r.std_errors, Tol::se(80))
            .scalar("df_resid", r.df_resid as f64, Tol::exact())
            .scalar("df_absorbed", r.df_absorbed as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
    }

    /// 50 disconnected 2 x 2 blocks (200 rows): 150 absorbed df, 49 residual df, and the
    /// Gaussian fit equals the linear HDFE estimator's.
    #[test]
    fn test_feglm_many_components_matches_hdfe() {
        let n = 200;
        let firm: Vec<i64> = (0..n).map(|i| ((i / 4) * 2 + i % 2) as i64).collect();
        let year: Vec<i64> = (0..n).map(|i| ((i / 4) * 2 + (i / 2) % 2) as i64).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37) % 17) as f64 / 7.0 + 0.1 * (i % 5) as f64)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.8 * x[i] + 0.05 * firm[i] as f64 + ((i * 53) % 29) as f64 / 13.0)
            .collect();
        let ds = Dataset::new(df! { "firm" => firm, "year" => year, "x" => x, "y" => y }.unwrap());
        let r = run_feglm(
            &ds,
            "y",
            &["x"],
            &["firm", "year"],
            GlmFamily::Gaussian,
            None,
        )
        .unwrap();
        assert_eq!((r.df_absorbed, r.df_resid), (150, 49));
        let h = crate::econometrics::hdfe::run_hdfe(
            &ds,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            crate::regression::CovarianceType::Standard,
        )
        .unwrap();
        assert_eq!(h.df_resid, r.df_resid);
        let tol = p2a_validation::Tol::new(1e-10, 1e-10);
        assert!(tol.allows(r.coefficients[0], h.coefficients[0]));
        assert!(
            tol.allows(r.std_errors[0], h.std_errors[0]),
            "{} vs {}",
            r.std_errors[0],
            h.std_errors[0]
        );
    }

    /// Gaussian with id and time effects on feglm_mle: every reported number is glm's (lm's
    /// for the SEs and the dispersion), and the pseudo R-squared is McFadden's on the normal
    /// log-likelihood.
    #[test]
    fn test_validate_feglm_mle_gaussian() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("feglm_mle");
        let r = run_feglm(
            &ds,
            "y_gauss",
            &["x", "x2"],
            &["id", "time"],
            GlmFamily::Gaussian,
            Some(tight()),
        )
        .unwrap();
        assert!(r.converged && r.warnings.is_empty(), "{:?}", r.warnings);
        Check::new(&case)
            .named("gauss2_coef", &r.variables, &r.coefficients, Tol::coef(480))
            .named("gauss2_se", &r.variables, &r.std_errors, Tol::se(480))
            .scalar("gauss2_loglik", r.log_likelihood, Tol::stat(480))
            .scalar("gauss2_loglik_null", r.log_likelihood_null, Tol::stat(480))
            .scalar("gauss2_aic", r.aic, Tol::stat(480))
            .scalar("gauss2_bic", r.bic, Tol::stat(480))
            .scalar("gauss2_deviance", r.deviance, Tol::stat(480))
            .scalar("gauss2_null_deviance", r.null_deviance, Tol::stat(480))
            .scalar("gauss2_dispersion", r.dispersion, Tol::stat(480))
            .scalar("gauss2_df_resid", r.df_resid as f64, Tol::exact())
            .scalar("gauss2_df_absorbed", r.df_absorbed as f64, Tol::exact())
            .scalar("gauss2_nobs", r.n_obs as f64, Tol::exact())
            .finish();
        let mcfadden = 1.0 - r.log_likelihood / r.log_likelihood_null;
        assert!((r.pseudo_r_squared - mcfadden).abs() <= 1e-12);
    }

    /// The round-based drop rule the worklist replaced: every round rescans every factor.
    fn separated_rows_kept_rounds(
        y: &Array1<f64>,
        factors: &[FactorInfo],
        binomial: bool,
    ) -> Vec<bool> {
        let n = y.len();
        let mut keep = vec![true; n];
        loop {
            let mut changed = false;
            for factor in factors {
                let mut sums = vec![0.0; factor.n_levels];
                let mut counts = vec![0usize; factor.n_levels];
                for i in (0..n).filter(|&i| keep[i]) {
                    sums[factor.ids[i]] += y[i];
                    counts[factor.ids[i]] += 1;
                }
                for i in 0..n {
                    let g = factor.ids[i];
                    if keep[i] && (sums[g] == 0.0 || (binomial && sums[g] == counts[g] as f64)) {
                        keep[i] = false;
                        changed = true;
                    }
                }
            }
            if !changed {
                return keep;
            }
        }
    }

    /// A dataset R generated for the FEGLM cases, read as `p2a data load` reads it; these
    /// tests use only its data, not the case's reference values.
    fn generated_csv(file: &str) -> Dataset {
        let path = format!(
            "{}/../../validation/datasets/generated/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    fn info_from_ids(name: &str, ids: Vec<usize>) -> FactorInfo {
        let n_levels = ids.iter().max().map_or(0, |m| m + 1);
        let mut counts = vec![0; n_levels];
        for &g in &ids {
            counts[g] += 1;
        }
        FactorInfo {
            name: name.to_string(),
            n_levels,
            ids,
            counts,
        }
    }

    /// The worklist drop rule keeps the rows the round-based rule keeps: on the separation
    /// case's data, on a chain whose drops cascade one level per round, and on noisy
    /// random panels with many all-0 and all-1 groups.
    #[test]
    fn test_separated_rows_worklist_matches_rounds() {
        use rand::{Rng, SeedableRng};
        let check = |y: &Array1<f64>, factors: &[FactorInfo], what: &str| {
            for binomial in [true, false] {
                if binomial && y.iter().any(|&v| v > 1.0) {
                    continue;
                }
                let got = separated_rows_kept(y, factors, binomial).unwrap();
                let want = separated_rows_kept_rounds(y, factors, binomial);
                assert_eq!(got, want, "{what}, binomial {binomial}");
            }
        };

        // the separation case's data, both outcomes, one and two factors
        let ds = generated_csv("feglm_sep.csv");
        for y_col in ["y_bin", "y_cnt"] {
            let cols = ColumnRequest::new(ds.df())
                .numeric(y_col)
                .factors(&["id", "time"])
                .na(NaPolicy::Omit)
                .extract()
                .unwrap();
            let y = cols.numeric_array(y_col);
            let factors: Vec<FactorInfo> = ["id", "time"]
                .iter()
                .map(|c| factor_info(c, cols.factor(c)).unwrap())
                .collect();
            let n_kept = |k: &[bool]| k.iter().filter(|&&b| b).count();
            let kept = separated_rows_kept(&y, &factors, y_col == "y_bin").unwrap();
            assert!(n_kept(&kept) < y.len(), "{y_col}: the case drops rows");
            check(&y, &factors, y_col);
            check(&y, &factors[..1], y_col);
        }

        // a chain: period 0 holds one 0, entity j has (j, j) = 0 and (j, j + 1) = 1, so each
        // drop makes the next level constant; the last period also holds a noisy tail
        let len = 60;
        let mut ent = Vec::new();
        let mut per = Vec::new();
        let mut y = Vec::new();
        for j in 0..len {
            ent.extend([j, j]);
            per.extend([j, j + 1]);
            y.extend([0.0, 1.0]);
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        for r in 0..20 {
            ent.push(len + r % 4);
            per.push(len - 10 + r % 7);
            y.push(f64::from(rng.gen_range(0..2u8)));
        }
        let factors = [info_from_ids("e", ent), info_from_ids("p", per)];
        let y = Array1::from(y);
        check(&y, &factors, "chain");
        let kept = separated_rows_kept(&y, &factors, true).unwrap();
        assert!(!kept[2 * len - 1] || !kept[0], "the chain cascades");

        // noisy random panels, two and three factors, outcomes 0/1 and counts with many zeros
        for seed in 0..30u64 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let n = rng.gen_range(20..300);
            let m = [
                rng.gen_range(2..40),
                rng.gen_range(2..15),
                rng.gen_range(2..6),
            ];
            let ids: Vec<Vec<usize>> = m
                .iter()
                .map(|&m| (0..n).map(|_| rng.gen_range(0..m)).collect())
                .collect();
            let p_one = rng.gen_range(0.02..0.5);
            let y_bin: Array1<f64> = (0..n)
                .map(|_| f64::from(u8::from(rng.r#gen::<f64>() < p_one)))
                .collect();
            let y_cnt: Array1<f64> = (0..n)
                .map(|_| {
                    if rng.r#gen::<f64>() < 0.7 {
                        0.0
                    } else {
                        rng.gen_range(0.3..4.0)
                    }
                })
                .collect();
            for k in [2, 3] {
                let factors: Vec<FactorInfo> = ids[..k]
                    .iter()
                    .enumerate()
                    .map(|(f, v)| info_from_ids(&format!("f{f}"), v.clone()))
                    .collect();
                check(&y_bin, &factors, &format!("seed {seed}, {k} factors, 0/1"));
                check(
                    &y_cnt,
                    &factors,
                    &format!("seed {seed}, {k} factors, counts"),
                );
            }
        }
    }

    /// Logit and probit refuse an outcome outside [0, 1], naming the column and the value,
    /// before the drop rule or step halving can misread it.
    #[test]
    fn test_feglm_binomial_outcome_outside_unit_interval() {
        let x: Vec<f64> = (0..40)
            .map(|i| ((i * 7) % 11) as f64 / 10.0 + 0.01 * i as f64)
            .collect();
        let base: Vec<f64> = (0..40)
            .map(|i| f64::from(u8::from((i * 5) % 3 == 0)))
            .collect();
        for (family, bad) in [(GlmFamily::Logit, 2.0), (GlmFamily::Probit, -1.0)] {
            let mut y = base.clone();
            y[3] = bad;
            let res = run_feglm(&frame(&x, &y, None), "y", &["x"], &["id"], family, None);
            expect_invalid(
                res,
                &format!("'y' must lie in [0, 1] for {family} (found {bad})"),
            );
        }
        // a fractional outcome inside [0, 1] is still a binomial mean
        let mut y = base.clone();
        y[3] = 0.25;
        assert!(
            run_feglm(
                &frame(&x, &y, None),
                "y",
                &["x"],
                &["id"],
                GlmFamily::Logit,
                None
            )
            .is_ok()
        );
    }

    /// A fit that stops because step halving left no step counts as converged, as in
    /// alpaca, but says so in its warnings. A weight floor far above the IRLS weights
    /// (with one demeaning sweep per step, or converged demeaning) turns the step into a
    /// poor direction on staggered two-way panels; the same panels at the default controls
    /// never warn.
    #[test]
    fn test_feglm_vanished_step_warns() {
        use rand::{Rng, SeedableRng};
        let warned = |r: &FeglmResult| {
            r.warnings
                .iter()
                .any(|w| w.contains("step halving could not improve the deviance"))
        };
        let mut n_warned = 0;
        for seed in 0..6u64 {
            // 40 entities, each observed in 4 consecutive periods starting at its own index
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let (mut e, mut t, mut x, mut y_bin, mut y_cnt) =
                (vec![], vec![], vec![], vec![], vec![]);
            for i in 0..40i64 {
                for k in 0..4 {
                    let period = i + k;
                    let xv = period as f64 * 0.3 + rng.r#gen::<f64>();
                    let eta = 0.5 * xv - 0.1 * period as f64 + 0.5 * (i % 3) as f64;
                    let p = 1.0 / (1.0 + (-eta).exp());
                    e.push(i);
                    t.push(period);
                    x.push(xv);
                    y_bin.push(f64::from(u8::from(rng.r#gen::<f64>() < p)));
                    y_cnt.push((eta.exp() * rng.r#gen::<f64>() * 2.0).floor());
                }
            }
            let ds = Dataset::new(
                df! { "e" => e, "t" => t, "x" => x, "y_bin" => y_bin, "y_cnt" => y_cnt }.unwrap(),
            );
            for (family, y) in [(GlmFamily::Logit, "y_bin"), (GlmFamily::Poisson, "y_cnt")] {
                let fit = |config| run_feglm(&ds, y, &["x"], &["e", "t"], family, config);
                let clean = fit(None).unwrap();
                assert!(
                    !warned(&clean),
                    "seed {seed} {family:?}: {:?}",
                    clean.warnings
                );
                for map_max_iter in [1, 1000] {
                    let mut config = FeglmConfig::default();
                    config.weight_min = 20.0;
                    config.map_max_iter = map_max_iter;
                    let r = fit(Some(config)).unwrap();
                    if warned(&r) {
                        assert!(r.converged, "a vanished step stops the fit as converged");
                        n_warned += 1;
                    }
                }
            }
        }
        assert!(
            n_warned > 0,
            "a weight floor of 20 should leave a vanished step"
        );
    }
}
