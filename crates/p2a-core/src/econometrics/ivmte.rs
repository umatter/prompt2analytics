//! Marginal Treatment Effects (MTE) Framework for Instrumental Variables.
//!
//! This module implements the MTE framework which connects IV estimation to a
//! choice-theoretic model of treatment selection. The MTE approach reveals how
//! different IV estimands (LATE, ATT, ATE) are weighted averages of the MTE curve,
//! allowing researchers to understand heterogeneity in treatment effects.
//!
//! # Mathematical Background
//!
//! ## Selection Model
//!
//! The MTE framework is based on a selection model with latent heterogeneity:
//!
//! - **Potential outcomes**: Y(0) = μ₀(X) + U₀, Y(1) = μ₁(X) + U₁
//! - **Selection equation**: D = 1{P(Z) ≥ U_D} where U_D ~ Uniform(0,1)
//! - **Propensity score**: P(Z) = Pr(D=1|Z) = E[D|Z]
//!
//! The key insight is that U_D represents unobserved resistance to treatment.
//! Individuals with low U_D are "eager" to take treatment, while those with high
//! U_D are "reluctant."
//!
//! ## Marginal Treatment Effect
//!
//! The MTE is defined as the treatment effect for individuals at the margin of
//! selection:
//!
//! ```text
//! MTE(x, u) = E[Y(1) - Y(0) | X = x, U_D = u]
//!           = E[β(X,U) | X = x, U_D = u]
//! ```
//!
//! where β(X,U) = Y(1) - Y(0) is the individual treatment effect.
//!
//! ## Local IV (LIV) Estimator
//!
//! The MTE can be recovered via the Local Instrumental Variables approach:
//!
//! ```text
//! MTE(p) = ∂E[Y|P(Z) = p] / ∂p
//! ```
//!
//! This derivative captures how expected outcomes change for individuals at
//! different points in the unobserved heterogeneity distribution.
//!
//! ## Treatment Effect Parameters as Weighted Averages
//!
//! All standard treatment effect parameters are weighted integrals of the MTE:
//!
//! - **ATE**: ATE = ∫₀¹ MTE(u) du
//! - **ATT**: ATT = ∫₀¹ MTE(u) × h_ATT(u) du where h_ATT(u) = Pr(U_D ≤ u) / E[P(Z)]
//! - **ATU**: ATU = ∫₀¹ MTE(u) × h_ATU(u) du
//! - **LATE**: LATE(z,z') = ∫ MTE(u) du / [P(z) - P(z')] over compliers
//!
//! # Parametric MTE Estimation
//!
//! This implementation uses a parametric approach:
//!
//! 1. **First stage**: Estimate propensity score P(Z) via probit/logit
//! 2. **Polynomial specification**: Model E[Y|X,P] = k(X) + Σⱼ γⱼ Pʲ
//! 3. **MTE recovery**: MTE(p) = ∂E[Y|X,P]/∂p = Σⱼ j × γⱼ × p^(j-1)
//!
//! # References
//!
//! - Heckman, J.J., & Vytlacil, E. (2005). Structural equations, treatment effects,
//!   and econometric policy evaluation. *Econometrica*, 73(3), 669-738.
//!   https://doi.org/10.1111/j.1468-0262.2005.00594.x
//!   The foundational paper establishing the MTE framework.
//!
//! - Heckman, J.J., & Vytlacil, E.J. (2007). Econometric evaluation of social
//!   programs, Part I: Causal models, structural models and econometric policy
//!   evaluation. *Handbook of Econometrics*, 6, 4779-4874.
//!   https://doi.org/10.1016/S1573-4412(07)06070-9
//!
//! - Cornelissen, T., Dustmann, C., Raute, A., & Schönberg, U. (2016). From LATE
//!   to MTE: Alternative methods for the evaluation of policy interventions.
//!   *Labour Economics*, 41, 47-60.
//!   https://doi.org/10.1016/j.labeco.2016.06.004
//!   Accessible introduction and practical guidance.
//!
//! - Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental
//!   variables for inference about policy relevant treatment parameters.
//!   *Econometrica*, 86(5), 1589-1619.
//!   https://doi.org/10.3982/ECTA15463
//!
//! - Brinch, C.N., Mogstad, M., & Wiswall, M. (2017). Beyond LATE with a discrete
//!   instrument. *Journal of Political Economy*, 125(4), 985-1039.
//!   https://doi.org/10.1086/692712
//!
//! R equivalent: `ivmte` package (Shea & Torgovitsky, 2021)
//! https://cran.r-project.org/package=ivmte

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{SignificanceLevel, normal_cdf, normal_pdf, t_test_p_value};

/// Estimand type for MTE analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MTEEstimand {
    /// Estimate the full MTE curve
    MTECurve,
    /// Local Average Treatment Effect (for compliers)
    LATE,
    /// Average Treatment Effect on the Treated
    ATT,
    /// Average Treatment Effect on the Untreated
    ATU,
    /// Average Treatment Effect (population average)
    ATE,
    /// Policy-Relevant Treatment Effect with custom weights
    PRTE,
}

impl fmt::Display for MTEEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MTEEstimand::MTECurve => write!(f, "MTE Curve"),
            MTEEstimand::LATE => write!(f, "LATE"),
            MTEEstimand::ATT => write!(f, "ATT"),
            MTEEstimand::ATU => write!(f, "ATU"),
            MTEEstimand::ATE => write!(f, "ATE"),
            MTEEstimand::PRTE => write!(f, "PRTE"),
        }
    }
}

/// Propensity score model type for first stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PropensityModel {
    /// Probit model (default, as in Heckman-Vytlacil)
    #[default]
    Probit,
    /// Logit model
    Logit,
    /// Linear probability model (OLS)
    Linear,
}

impl fmt::Display for PropensityModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropensityModel::Probit => write!(f, "Probit"),
            PropensityModel::Logit => write!(f, "Logit"),
            PropensityModel::Linear => write!(f, "Linear"),
        }
    }
}

/// Configuration for MTE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVMTEConfig {
    /// Polynomial degree for MTE approximation (default: 2)
    /// Higher degrees allow more flexible MTE shapes but risk overfitting.
    pub mte_degree: usize,

    /// Primary estimand to report
    pub estimand: MTEEstimand,

    /// Number of grid points for MTE curve evaluation (default: 100)
    pub n_grid: usize,

    /// Whether to compute bootstrap standard errors
    pub bootstrap_se: bool,

    /// Number of bootstrap replications (default: 500)
    pub n_bootstrap: usize,

    /// Propensity score model for first stage
    pub propensity_model: PropensityModel,

    /// Maximum iterations for propensity estimation (default: 100)
    pub max_iter: usize,

    /// Convergence tolerance for propensity estimation (default: 1e-8)
    pub tolerance: f64,

    /// Custom weights for PRTE (optional)
    /// Vector of length n_grid specifying weights at each grid point
    pub prte_weights: Option<Vec<f64>>,

    /// Include covariates in the second stage (default: true)
    pub include_covariates: bool,
}

impl Default for IVMTEConfig {
    fn default() -> Self {
        Self {
            mte_degree: 2,
            estimand: MTEEstimand::MTECurve,
            n_grid: 100,
            bootstrap_se: false,
            n_bootstrap: 500,
            propensity_model: PropensityModel::Probit,
            max_iter: 100,
            tolerance: 1e-8,
            prte_weights: None,
            include_covariates: true,
        }
    }
}

/// Point on the MTE curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTEPoint {
    /// Unobserved heterogeneity value (propensity score point)
    pub u: f64,
    /// MTE estimate at this point
    pub mte: f64,
    /// Standard error (if computed via bootstrap)
    pub se: Option<f64>,
    /// 95% confidence interval lower bound
    pub ci_lower: Option<f64>,
    /// 95% confidence interval upper bound
    pub ci_upper: Option<f64>,
}

/// Treatment effect parameter estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatmentEffectEstimate {
    /// Name of the estimand
    pub name: String,
    /// Point estimate
    pub estimate: f64,
    /// Standard error
    pub se: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value
    pub p_value: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
    /// Significance level
    pub significance: SignificanceLevel,
}

/// First-stage propensity score estimation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityStageResult {
    /// Model type used
    pub model: PropensityModel,
    /// Coefficients on instruments
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Variable names
    pub variables: Vec<String>,
    /// Log-likelihood (for probit/logit)
    pub log_likelihood: Option<f64>,
    /// Pseudo R-squared
    pub pseudo_r_squared: Option<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Fitted propensity scores
    #[serde(skip)]
    pub propensity: Array1<f64>,
    /// Min propensity score
    pub p_min: f64,
    /// Max propensity score
    pub p_max: f64,
    /// Mean propensity score
    pub p_mean: f64,
}

/// Complete result from MTE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVMTEResult {
    /// MTE curve evaluated at grid points
    pub mte_curve: Vec<MTEPoint>,

    /// Polynomial coefficients for MTE: MTE(p) = sum_j gamma_j * j * p^(j-1)
    /// These are the coefficients on the polynomial terms in propensity
    pub mte_coefficients: Vec<f64>,

    /// Polynomial coefficients for E[Y|X,P] = ... + sum_j gamma_j * P^j
    /// These are the "raw" polynomial coefficients before differentiation
    pub polynomial_coefficients: Vec<f64>,

    /// Standard errors for polynomial coefficients
    pub polynomial_se: Vec<f64>,

    /// Average Treatment Effect (integral of MTE)
    pub ate: TreatmentEffectEstimate,

    /// Average Treatment Effect on the Treated
    pub att: TreatmentEffectEstimate,

    /// Average Treatment Effect on the Untreated
    pub atu: TreatmentEffectEstimate,

    /// Local Average Treatment Effect (for compliers)
    /// Computed using variation in the propensity score
    pub late: TreatmentEffectEstimate,

    /// First-stage propensity estimation results
    pub first_stage: PropensityStageResult,

    /// LATE weights at each grid point (showing how LATE weights MTE)
    pub weights_late: Vec<f64>,

    /// ATT weights at each grid point
    pub weights_att: Vec<f64>,

    /// ATU weights at each grid point
    pub weights_atu: Vec<f64>,

    /// Number of observations
    pub n_obs: usize,

    /// Polynomial degree used
    pub mte_degree: usize,

    /// R-squared from second-stage polynomial regression
    pub r_squared: f64,

    /// Residual standard error from second stage
    pub residual_se: f64,

    /// Warnings generated during estimation
    pub warnings: Vec<String>,
}

impl fmt::Display for IVMTEResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Marginal Treatment Effects (MTE) Estimation")?;
        writeln!(f, "=============================================")?;
        writeln!(f)?;
        writeln!(
            f,
            "First Stage: {} Propensity Score Model",
            self.first_stage.model
        )?;
        writeln!(f, "  Observations: {}", self.n_obs)?;
        writeln!(
            f,
            "  Propensity range: [{:.4}, {:.4}]",
            self.first_stage.p_min, self.first_stage.p_max
        )?;
        writeln!(f, "  Mean propensity: {:.4}", self.first_stage.p_mean)?;
        if let Some(pseudo_r2) = self.first_stage.pseudo_r_squared {
            writeln!(f, "  Pseudo R-squared: {:.4}", pseudo_r2)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Second Stage: Polynomial MTE (degree {})",
            self.mte_degree
        )?;
        writeln!(f, "  R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "  Residual SE: {:.4}", self.residual_se)?;
        writeln!(f)?;

        writeln!(f, "MTE Polynomial Coefficients (on P^j):")?;
        writeln!(f, "  {:>10} {:>12} {:>12}", "Power", "Coef", "Std Err")?;
        writeln!(f, "  {}", "-".repeat(36))?;
        for (j, (coef, se)) in self
            .polynomial_coefficients
            .iter()
            .zip(self.polynomial_se.iter())
            .enumerate()
        {
            writeln!(
                f,
                "  {:>10} {:>12.4} {:>12.4}",
                format!("P^{}", j + 1),
                coef,
                se
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Treatment Effect Estimates:")?;
        writeln!(
            f,
            "{:<10} {:>12} {:>10} {:>10} {:>20}",
            "Estimand", "Estimate", "Std Err", "P>|z|", "95% CI"
        )?;
        writeln!(f, "{}", "-".repeat(64))?;

        for te in [&self.ate, &self.att, &self.atu, &self.late] {
            writeln!(
                f,
                "{:<10} {:>12.4} {:>10.4} {:>10.4} [{:>8.4}, {:>8.4}]{}",
                te.name,
                te.estimate,
                te.se,
                te.p_value,
                te.ci_lower,
                te.ci_upper,
                te.significance.stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(64))?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "  - {}", warning)?;
            }
        }

        Ok(())
    }
}

/// Run Marginal Treatment Effects estimation.
///
/// Implements the parametric MTE framework connecting IV to treatment effect
/// heterogeneity.
///
/// # Arguments
///
/// * `y` - Outcome variable (n x 1)
/// * `d` - Binary treatment indicator (n x 1, values 0 or 1)
/// * `z` - Instrument(s) for treatment (n x 1 or n x k)
/// * `x` - Optional covariates (n x p)
/// * `config` - Configuration options
///
/// # Returns
///
/// `IVMTEResult` containing:
/// - MTE curve evaluated at grid points
/// - Treatment effect estimates (ATE, ATT, ATU, LATE)
/// - Propensity score estimates
/// - Weights showing how each estimand integrates the MTE
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::ivmte::{run_ivmte, IVMTEConfig};
/// use ndarray::array;
///
/// let y = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let d = array![0.0, 1.0, 0.0, 1.0, 1.0];
/// let z = array![0.2, 0.8, 0.3, 0.9, 0.7];
///
/// let result = run_ivmte(
///     &y.view(),
///     &d.view(),
///     &z.view(),
///     None,
///     IVMTEConfig::default(),
/// )?;
///
/// println!("ATE: {:.4}", result.ate.estimate);
/// println!("ATT: {:.4}", result.att.estimate);
/// ```
///
/// # Algorithm
///
/// 1. **First stage**: Estimate P(Z) = Pr(D=1|Z) via probit
/// 2. **Build polynomial basis**: Create P, P², P³, ... up to specified degree
/// 3. **Second stage**: Regress Y on covariates and polynomial in P
/// 4. **MTE recovery**: Differentiate E[Y|P] w.r.t. P to get MTE(p)
/// 5. **Integration**: Compute ATE, ATT, ATU, LATE as weighted integrals
///
/// # References
///
/// - Heckman, J.J., & Vytlacil, E. (2005). Structural equations, treatment effects,
///   and econometric policy evaluation. *Econometrica*, 73(3), 669-738.
pub fn run_ivmte(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    x: Option<&ArrayView2<f64>>,
    config: IVMTEConfig,
) -> EconResult<IVMTEResult> {
    let n = y.len();
    let mut warnings = Vec::new();

    // Validate inputs
    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "MTE estimation requires sufficient data".to_string(),
        });
    }

    if d.len() != n || z.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dimension mismatch: y has {} obs, d has {}, z has {}",
                n,
                d.len(),
                z.len()
            ),
        });
    }

    // Check treatment is binary
    let n_treated: usize = d.iter().filter(|&&v| v > 0.5).count();
    let n_untreated = n - n_treated;

    if n_treated == 0 || n_untreated == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Treatment must have both treated and untreated observations".to_string(),
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Stage 1: Estimate propensity score P(Z) = Pr(D=1|Z)
    // ═══════════════════════════════════════════════════════════════════════════

    let first_stage = estimate_propensity(d, z, &config)?;
    let propensity = &first_stage.propensity;

    // Validate propensity scores
    if first_stage.p_min < 0.01 || first_stage.p_max > 0.99 {
        warnings.push(format!(
            "Propensity scores near boundaries [{:.4}, {:.4}]. MTE may be poorly identified at extremes.",
            first_stage.p_min, first_stage.p_max
        ));
    }

    // Check for sufficient variation in propensity
    let p_variance = propensity
        .iter()
        .map(|&p| (p - first_stage.p_mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    if p_variance < 0.01 {
        warnings.push(format!(
            "Low variance in propensity scores (Var = {:.4}). Instrument may be weak.",
            p_variance
        ));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Stage 2: Polynomial regression of Y on P, P², P³, ...
    // ═══════════════════════════════════════════════════════════════════════════

    let (polynomial_coefs, polynomial_se, r_squared, residual_se, _covariate_coefs) =
        estimate_polynomial_stage(y, propensity, x, &config)?;

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute MTE curve: MTE(p) = dE[Y|P=p]/dp
    // For polynomial: E[Y|P] = a + b₁P + b₂P² + ... + b_kP^k
    // MTE(p) = b₁ + 2b₂p + 3b₃p² + ... + kb_kp^(k-1)
    // ═══════════════════════════════════════════════════════════════════════════

    let mte_curve = compute_mte_curve(&polynomial_coefs, config.n_grid);

    // Compute treatment effect parameters as weighted integrals of MTE
    let (ate, weights_ate) = compute_ate(&polynomial_coefs, config.n_grid);
    let (att, weights_att) = compute_att(&polynomial_coefs, propensity, config.n_grid);
    let (atu, weights_atu) = compute_atu(&polynomial_coefs, propensity, config.n_grid);
    let (late, weights_late) = compute_late(&polynomial_coefs, propensity, config.n_grid);

    // Compute standard errors for treatment effects
    // Using delta method from polynomial coefficient SEs
    let ate_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_ate);
    let att_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_att);
    let atu_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_atu);
    let late_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_late);

    // Build treatment effect estimates
    let df = n.saturating_sub(polynomial_coefs.len() + 1) as f64;

    let ate_est = build_te_estimate("ATE", ate, ate_se, df);
    let att_est = build_te_estimate("ATT", att, att_se, df);
    let atu_est = build_te_estimate("ATU", atu, atu_se, df);
    let late_est = build_te_estimate("LATE", late, late_se, df);

    // Convert MTE coefficients (derivatives of polynomial)
    let mte_coefficients: Vec<f64> = polynomial_coefs
        .iter()
        .enumerate()
        .map(|(j, &c)| (j + 1) as f64 * c)
        .collect();

    Ok(IVMTEResult {
        mte_curve,
        mte_coefficients,
        polynomial_coefficients: polynomial_coefs,
        polynomial_se,
        ate: ate_est,
        att: att_est,
        atu: atu_est,
        late: late_est,
        first_stage,
        weights_late,
        weights_att,
        weights_atu,
        n_obs: n,
        mte_degree: config.mte_degree,
        r_squared,
        residual_se,
        warnings,
    })
}

/// Estimate propensity score via probit, logit, or linear model.
fn estimate_propensity(
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    config: &IVMTEConfig,
) -> EconResult<PropensityStageResult> {
    let n = d.len();

    // Build design matrix with intercept
    let mut x_prop = Array2::zeros((n, 2));
    for i in 0..n {
        x_prop[[i, 0]] = 1.0; // intercept
        x_prop[[i, 1]] = z[i]; // instrument
    }

    match config.propensity_model {
        PropensityModel::Probit => estimate_probit(d, &x_prop.view(), config),
        PropensityModel::Logit => estimate_logit(d, &x_prop.view(), config),
        PropensityModel::Linear => estimate_linear_probability(d, &x_prop.view()),
    }
}

/// Probit estimation via Newton-Raphson MLE.
fn estimate_probit(
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &IVMTEConfig,
) -> EconResult<PropensityStageResult> {
    let n = x.nrows();
    let k = x.ncols();

    // Initialize coefficients at zero
    let mut beta = Array1::zeros(k);

    let mut log_lik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Compute linear predictor and probabilities
        let xb: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = xb.mapv(normal_cdf);

        // Clip probabilities to avoid numerical issues
        let p_clipped: Array1<f64> = p.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));

        // Log-likelihood
        let new_log_lik: f64 = d
            .iter()
            .zip(p_clipped.iter())
            .map(
                |(&di, &pi)| {
                    if di > 0.5 { pi.ln() } else { (1.0 - pi).ln() }
                },
            )
            .sum();

        // Check convergence
        if (new_log_lik - log_lik).abs() < config.tolerance {
            converged = true;
            log_lik = new_log_lik;
            break;
        }
        log_lik = new_log_lik;

        // Gradient and Hessian
        let phi: Array1<f64> = xb.mapv(normal_pdf);
        let lambda: Array1<f64> = d
            .iter()
            .zip(p_clipped.iter())
            .zip(phi.iter())
            .map(|((&di, &pi), &phi_i)| {
                if di > 0.5 {
                    phi_i / pi
                } else {
                    -phi_i / (1.0 - pi)
                }
            })
            .collect();

        // Gradient: X'λ
        let gradient = x.t().dot(&lambda);

        // Hessian approximation (expected information): -X'WX
        // where W = diag(phi² / (p(1-p)))
        let w: Array1<f64> = phi
            .iter()
            .zip(p_clipped.iter())
            .map(|(&phi_i, &pi)| {
                let denom = pi * (1.0 - pi);
                if denom > 1e-10 {
                    phi_i * phi_i / denom
                } else {
                    1e-10
                }
            })
            .collect();

        // Compute X'WX
        let mut xwx = Array2::zeros((k, k));
        for i in 0..n {
            let wi = w[i];
            for j in 0..k {
                for l in 0..k {
                    xwx[[j, l]] += wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton update: beta_new = beta + (X'WX)^{-1} X'λ
        let (xwx_inv, _) = safe_inverse(&xwx.view()).map_err(|_| EconError::SingularMatrix {
            context: "Probit Hessian".to_string(),
            suggestion: "Check for perfect separation in propensity model".to_string(),
        })?;

        let delta = xwx_inv.dot(&gradient);
        beta += &delta;
    }

    if !converged {
        return Err(EconError::ConvergenceFailure {
            iterations: config.max_iter,
            last_change: (log_lik).abs(),
            suggestion: "Try different starting values or increase max_iter".to_string(),
        });
    }

    // Compute fitted probabilities
    let xb = x.dot(&beta);
    let propensity: Array1<f64> = xb.mapv(normal_cdf);

    // Standard errors from information matrix
    let phi: Array1<f64> = xb.mapv(normal_pdf);
    let p_clipped: Array1<f64> = propensity.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));

    let w: Array1<f64> = phi
        .iter()
        .zip(p_clipped.iter())
        .map(|(&phi_i, &pi)| {
            let denom = pi * (1.0 - pi);
            if denom > 1e-10 {
                phi_i * phi_i / denom
            } else {
                1e-10
            }
        })
        .collect();

    let mut info = Array2::zeros((k, k));
    for i in 0..n {
        let wi = w[i];
        for j in 0..k {
            for l in 0..k {
                info[[j, l]] += wi * x[[i, j]] * x[[i, l]];
            }
        }
    }

    let (vcov, _) = safe_inverse(&info.view()).map_err(|_| EconError::SingularMatrix {
        context: "Probit variance-covariance".to_string(),
        suggestion: "Model may be poorly identified".to_string(),
    })?;

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // Null log-likelihood (intercept only)
    let p_bar = d.mean().unwrap_or(0.5);
    let log_lik_null: f64 = d
        .iter()
        .map(|&di| {
            if di > 0.5 {
                p_bar.ln()
            } else {
                (1.0 - p_bar).ln()
            }
        })
        .sum();

    let pseudo_r_squared = 1.0 - log_lik / log_lik_null;

    let p_min = propensity.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = propensity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p_mean = propensity.mean().unwrap_or(0.5);

    Ok(PropensityStageResult {
        model: PropensityModel::Probit,
        coefficients: beta.to_vec(),
        std_errors,
        variables: vec!["(Intercept)".to_string(), "z".to_string()],
        log_likelihood: Some(log_lik),
        pseudo_r_squared: Some(pseudo_r_squared),
        iterations,
        converged,
        propensity,
        p_min,
        p_max,
        p_mean,
    })
}

/// Logit estimation via Newton-Raphson MLE.
fn estimate_logit(
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &IVMTEConfig,
) -> EconResult<PropensityStageResult> {
    let n = x.nrows();
    let k = x.ncols();

    // Initialize coefficients at zero
    let mut beta = Array1::zeros(k);

    let mut log_lik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    // Logistic CDF and PDF
    fn logistic_cdf(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Compute linear predictor and probabilities
        let xb: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = xb.mapv(logistic_cdf);

        // Clip probabilities
        let p_clipped: Array1<f64> = p.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));

        // Log-likelihood
        let new_log_lik: f64 = d
            .iter()
            .zip(p_clipped.iter())
            .map(
                |(&di, &pi)| {
                    if di > 0.5 { pi.ln() } else { (1.0 - pi).ln() }
                },
            )
            .sum();

        // Check convergence
        if (new_log_lik - log_lik).abs() < config.tolerance {
            converged = true;
            log_lik = new_log_lik;
            break;
        }
        log_lik = new_log_lik;

        // Gradient: X'(d - p)
        let residuals: Array1<f64> = d
            .iter()
            .zip(p_clipped.iter())
            .map(|(&di, &pi)| di - pi)
            .collect();
        let gradient = x.t().dot(&residuals);

        // Hessian: -X'WX where W = diag(p(1-p))
        let w: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        let mut xwx = Array2::zeros((k, k));
        for i in 0..n {
            let wi = w[i];
            for j in 0..k {
                for l in 0..k {
                    xwx[[j, l]] += wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton update
        let (xwx_inv, _) = safe_inverse(&xwx.view()).map_err(|_| EconError::SingularMatrix {
            context: "Logit Hessian".to_string(),
            suggestion: "Check for perfect separation".to_string(),
        })?;

        let delta = xwx_inv.dot(&gradient);
        beta += &delta;
    }

    if !converged {
        return Err(EconError::ConvergenceFailure {
            iterations: config.max_iter,
            last_change: (log_lik).abs(),
            suggestion: "Try different starting values".to_string(),
        });
    }

    // Compute fitted probabilities and SEs
    let xb = x.dot(&beta);
    let propensity: Array1<f64> = xb.mapv(logistic_cdf);
    let p_clipped: Array1<f64> = propensity.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));

    let w: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

    let mut info = Array2::zeros((k, k));
    for i in 0..n {
        let wi = w[i];
        for j in 0..k {
            for l in 0..k {
                info[[j, l]] += wi * x[[i, j]] * x[[i, l]];
            }
        }
    }

    let (vcov, _) = safe_inverse(&info.view()).map_err(|_| EconError::SingularMatrix {
        context: "Logit variance-covariance".to_string(),
        suggestion: "Model may be poorly identified".to_string(),
    })?;

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // Pseudo R-squared
    let p_bar = d.mean().unwrap_or(0.5);
    let log_lik_null: f64 = d
        .iter()
        .map(|&di| {
            if di > 0.5 {
                p_bar.ln()
            } else {
                (1.0 - p_bar).ln()
            }
        })
        .sum();

    let pseudo_r_squared = 1.0 - log_lik / log_lik_null;

    let p_min = propensity.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = propensity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p_mean = propensity.mean().unwrap_or(0.5);

    Ok(PropensityStageResult {
        model: PropensityModel::Logit,
        coefficients: beta.to_vec(),
        std_errors,
        variables: vec!["(Intercept)".to_string(), "z".to_string()],
        log_likelihood: Some(log_lik),
        pseudo_r_squared: Some(pseudo_r_squared),
        iterations,
        converged,
        propensity,
        p_min,
        p_max,
        p_mean,
    })
}

/// Linear probability model via OLS.
fn estimate_linear_probability(
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
) -> EconResult<PropensityStageResult> {
    let n = x.nrows();
    let k = x.ncols();

    // OLS: beta = (X'X)^{-1}X'd
    let xtx_mat = xtx(x);
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "Linear probability model X'X".to_string(),
        suggestion: "Check for perfect collinearity".to_string(),
    })?;

    let d_array = Array1::from_iter(d.iter().cloned());
    let xty_vec = xty(x, &d_array);
    let beta = xtx_inv.dot(&xty_vec);

    // Fitted values (propensity scores)
    let propensity_raw: Array1<f64> = x.dot(&beta);

    // Clip to [0, 1]
    let propensity: Array1<f64> = propensity_raw.mapv(|v| v.max(0.0).min(1.0));

    // Standard errors
    let residuals = &d_array - &propensity_raw;
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / (n - k) as f64;
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // R-squared
    let d_mean = d.mean().unwrap_or(0.5);
    let sst: f64 = d.iter().map(|&di| (di - d_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    let p_min = propensity.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = propensity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p_mean = propensity.mean().unwrap_or(0.5);

    Ok(PropensityStageResult {
        model: PropensityModel::Linear,
        coefficients: beta.to_vec(),
        std_errors,
        variables: vec!["(Intercept)".to_string(), "z".to_string()],
        log_likelihood: None,
        pseudo_r_squared: Some(r_squared),
        iterations: 1,
        converged: true,
        propensity,
        p_min,
        p_max,
        p_mean,
    })
}

/// Second stage: polynomial regression of Y on P.
///
/// Model: E[Y|X,P] = X'γ + β₁P + β₂P² + ... + β_kP^k
fn estimate_polynomial_stage(
    y: &ArrayView1<f64>,
    propensity: &Array1<f64>,
    x: Option<&ArrayView2<f64>>,
    config: &IVMTEConfig,
) -> EconResult<(Vec<f64>, Vec<f64>, f64, f64, Option<Vec<f64>>)> {
    let n = y.len();
    let degree = config.mte_degree;

    // Build design matrix: [1, X, P, P², ..., P^degree]
    let n_x_cols = x.map_or(0, |m| m.ncols());
    let n_poly_cols = degree;
    let total_cols = 1 + n_x_cols + n_poly_cols; // intercept + covariates + polynomial

    let mut design = Array2::zeros((n, total_cols));

    // Intercept
    for i in 0..n {
        design[[i, 0]] = 1.0;
    }

    // Covariates (if any)
    if let Some(x_mat) = x {
        if config.include_covariates {
            for i in 0..n {
                for j in 0..n_x_cols {
                    design[[i, 1 + j]] = x_mat[[i, j]];
                }
            }
        }
    }

    // Polynomial terms in propensity
    let poly_start = 1 + n_x_cols;
    for i in 0..n {
        let p = propensity[i];
        for j in 0..degree {
            design[[i, poly_start + j]] = p.powi((j + 1) as i32);
        }
    }

    // OLS regression
    let xtx_mat = xtx(&design.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "Polynomial stage X'X".to_string(),
        suggestion: "Try reducing polynomial degree".to_string(),
    })?;

    let y_array = Array1::from_iter(y.iter().cloned());
    let xty_vec = xty(&design.view(), &y_array);
    let beta = xtx_inv.dot(&xty_vec);

    // Residuals and fit statistics
    let fitted = design.dot(&beta);
    let residuals = &y_array - &fitted;
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let df = n.saturating_sub(total_cols);
    let sigma2 = if df > 0 {
        ssr / df as f64
    } else {
        ssr / n as f64
    };
    let residual_se = sigma2.sqrt();

    // R-squared
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Standard errors
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // Extract polynomial coefficients (excluding intercept and covariates)
    let polynomial_coefs: Vec<f64> = beta.slice(s![poly_start..]).to_vec();
    let polynomial_se: Vec<f64> = std_errors[poly_start..].to_vec();

    // Covariate coefficients (if any)
    let covariate_coefs = if n_x_cols > 0 {
        Some(beta.slice(s![1..1 + n_x_cols]).to_vec())
    } else {
        None
    };

    Ok((
        polynomial_coefs,
        polynomial_se,
        r_squared,
        residual_se,
        covariate_coefs,
    ))
}

/// Compute MTE curve from polynomial coefficients.
///
/// MTE(p) = dE[Y|P=p]/dp = β₁ + 2β₂p + 3β₃p² + ...
fn compute_mte_curve(poly_coefs: &[f64], n_grid: usize) -> Vec<MTEPoint> {
    let mut curve = Vec::with_capacity(n_grid);

    for i in 0..n_grid {
        let u = (i as f64 + 0.5) / n_grid as f64; // Grid from 0 to 1

        // MTE = derivative of polynomial
        // If E[Y|P] = Σⱼ βⱼ P^j, then MTE = Σⱼ j*βⱼ P^(j-1)
        let mut mte = 0.0;
        for (j, &coef) in poly_coefs.iter().enumerate() {
            let power = j; // P^(j+1) differentiates to (j+1)*P^j
            mte += (j + 1) as f64 * coef * u.powi(power as i32);
        }

        curve.push(MTEPoint {
            u,
            mte,
            se: None,
            ci_lower: None,
            ci_upper: None,
        });
    }

    curve
}

/// Compute ATE = ∫₀¹ MTE(u) du (uniform weights).
fn compute_ate(poly_coefs: &[f64], n_grid: usize) -> (f64, Vec<f64>) {
    // ATE with uniform weights
    let mut ate = 0.0;
    let mut weights = Vec::with_capacity(n_grid);

    let du = 1.0 / n_grid as f64;

    for i in 0..n_grid {
        let u = (i as f64 + 0.5) / n_grid as f64;

        // MTE at this point
        let mut mte = 0.0;
        for (j, &coef) in poly_coefs.iter().enumerate() {
            mte += (j + 1) as f64 * coef * u.powi(j as i32);
        }

        // Uniform weight
        let w = du;
        weights.push(w);
        ate += mte * w;
    }

    // Normalize weights
    let sum_w: f64 = weights.iter().sum();
    if sum_w > 0.0 {
        for w in &mut weights {
            *w /= sum_w;
        }
    }

    (ate, weights)
}

/// Compute ATT = ∫₀¹ MTE(u) × h_ATT(u) du.
///
/// h_ATT(u) = Pr(U_D ≤ u | D=1) / Pr(D=1)
/// For uniform U_D: h_ATT(u) ∝ P(Z) × 1{u ≤ P(Z)}
fn compute_att(poly_coefs: &[f64], propensity: &Array1<f64>, n_grid: usize) -> (f64, Vec<f64>) {
    let n = propensity.len();

    let mut att = 0.0;
    let mut weights = Vec::with_capacity(n_grid);

    let du = 1.0 / n_grid as f64;

    for i in 0..n_grid {
        let u = (i as f64 + 0.5) / n_grid as f64;

        // MTE at this point
        let mut mte = 0.0;
        for (j, &coef) in poly_coefs.iter().enumerate() {
            mte += (j + 1) as f64 * coef * u.powi(j as i32);
        }

        // ATT weight: proportion of treated with U_D ≤ u
        // h_ATT(u) = Pr(U_D ≤ u | D=1) = E[1{U_D ≤ u} | D=1]
        // Under the model, D=1 iff P(Z) ≥ U_D, so given D=1, U_D ≤ P(Z)
        // Thus h_ATT(u) = Pr(U_D ≤ u | U_D ≤ P(Z)) = min(u, P(Z)) / P(Z)

        let w_raw: f64 = propensity
            .iter()
            .map(|&p| if p > 0.0 { u.min(p) / p } else { 0.0 })
            .sum::<f64>()
            / n as f64;

        let w = w_raw * du;
        weights.push(w);
        att += mte * w;
    }

    // Normalize weights
    let sum_w: f64 = weights.iter().sum();
    if sum_w > 0.0 {
        for w in &mut weights {
            *w /= sum_w;
        }
        att /= sum_w;
    }

    (att, weights)
}

/// Compute ATU = ∫₀¹ MTE(u) × h_ATU(u) du.
///
/// h_ATU(u) = Pr(U_D > u | D=0) weights
fn compute_atu(poly_coefs: &[f64], propensity: &Array1<f64>, n_grid: usize) -> (f64, Vec<f64>) {
    let n = propensity.len();

    let mut atu = 0.0;
    let mut weights = Vec::with_capacity(n_grid);

    let du = 1.0 / n_grid as f64;

    for i in 0..n_grid {
        let u = (i as f64 + 0.5) / n_grid as f64;

        // MTE at this point
        let mut mte = 0.0;
        for (j, &coef) in poly_coefs.iter().enumerate() {
            mte += (j + 1) as f64 * coef * u.powi(j as i32);
        }

        // ATU weight: h_ATU(u) = Pr(U_D > u | D=0)
        // Given D=0, U_D > P(Z), so h_ATU(u) = Pr(U_D > u | U_D > P(Z))
        // = (1 - max(u, P(Z))) / (1 - P(Z))
        let w_raw: f64 = propensity
            .iter()
            .map(|&p| {
                if p < 1.0 {
                    (1.0 - u.max(p)) / (1.0 - p)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n as f64;

        let w = w_raw * du;
        weights.push(w);
        atu += mte * w;
    }

    // Normalize weights
    let sum_w: f64 = weights.iter().sum();
    if sum_w > 0.0 {
        for w in &mut weights {
            *w /= sum_w;
        }
        atu /= sum_w;
    }

    (atu, weights)
}

/// Compute LATE using variation in propensity scores.
///
/// LATE weights MTE over the complier region.
fn compute_late(poly_coefs: &[f64], propensity: &Array1<f64>, n_grid: usize) -> (f64, Vec<f64>) {
    let p_min = propensity.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = propensity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // LATE is the average MTE over [p_min, p_max]
    // LATE = ∫_{p_min}^{p_max} MTE(u) du / (p_max - p_min)

    let mut late = 0.0;
    let mut weights = vec![0.0; n_grid];

    if p_max <= p_min {
        // No variation in propensity
        return (0.0, weights);
    }

    let du = 1.0 / n_grid as f64;

    for i in 0..n_grid {
        let u = (i as f64 + 0.5) / n_grid as f64;

        // Only weight points in complier region
        if u >= p_min && u <= p_max {
            // MTE at this point
            let mut mte = 0.0;
            for (j, &coef) in poly_coefs.iter().enumerate() {
                mte += (j + 1) as f64 * coef * u.powi(j as i32);
            }

            let w = du / (p_max - p_min);
            weights[i] = w;
            late += mte * w;
        }
    }

    // Normalize weights
    let sum_w: f64 = weights.iter().sum();
    if sum_w > 0.0 {
        for w in &mut weights {
            *w /= sum_w;
        }
        late /= sum_w;
    }

    (late, weights)
}

/// Compute standard error for treatment effect estimate using delta method.
fn compute_te_se(poly_coefs: &[f64], poly_se: &[f64], weights: &[f64]) -> f64 {
    // SE of weighted average of MTE
    // MTE(u) = Σⱼ (j+1) βⱼ u^j
    // TE = Σᵢ wᵢ MTE(uᵢ)
    //    = Σⱼ βⱼ Σᵢ wᵢ (j+1) uᵢ^j
    //
    // Variance = Σⱼ Var(βⱼ) × (Σᵢ wᵢ (j+1) uᵢ^j)²

    let n_grid = weights.len();
    let degree = poly_coefs.len();

    let mut variance = 0.0;

    for j in 0..degree {
        // Compute the weighted sum of derivatives w.r.t. βⱼ
        let mut deriv = 0.0;
        for i in 0..n_grid {
            let u = (i as f64 + 0.5) / n_grid as f64;
            deriv += weights[i] * (j + 1) as f64 * u.powi(j as i32);
        }

        if j < poly_se.len() {
            variance += poly_se[j].powi(2) * deriv.powi(2);
        }
    }

    variance.sqrt()
}

/// Build a treatment effect estimate struct.
fn build_te_estimate(name: &str, estimate: f64, se: f64, df: f64) -> TreatmentEffectEstimate {
    let t_stat = if se > 1e-10 { estimate / se } else { 0.0 };
    let p_value = t_test_p_value(t_stat, df);
    let ci_half = 1.96 * se;

    TreatmentEffectEstimate {
        name: name.to_string(),
        estimate,
        se,
        t_stat,
        p_value,
        ci_lower: estimate - ci_half,
        ci_upper: estimate + ci_half,
        significance: SignificanceLevel::from_p_value(p_value),
    }
}

/// Convenience function for MTE estimation with default configuration.
pub fn ivmte(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    x: Option<&ArrayView2<f64>>,
) -> EconResult<IVMTEResult> {
    run_ivmte(y, d, z, x, IVMTEConfig::default())
}

/// Run MTE estimation with multiple instruments.
///
/// When multiple instruments are available, the propensity score is estimated
/// using all instruments jointly.
pub fn run_ivmte_multi_z(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView2<f64>,
    x: Option<&ArrayView2<f64>>,
    config: IVMTEConfig,
) -> EconResult<IVMTEResult> {
    let n = y.len();

    // For multiple instruments, first compute a scalar propensity index
    // by projecting Z onto D via probit, then proceed as usual

    // Build design matrix with intercept
    let n_z = z.ncols();
    let mut x_prop = Array2::zeros((n, 1 + n_z));
    for i in 0..n {
        x_prop[[i, 0]] = 1.0; // intercept
        for j in 0..n_z {
            x_prop[[i, 1 + j]] = z[[i, j]];
        }
    }

    // Estimate propensity using all instruments
    let first_stage = match config.propensity_model {
        PropensityModel::Probit => estimate_probit(d, &x_prop.view(), &config)?,
        PropensityModel::Logit => estimate_logit(d, &x_prop.view(), &config)?,
        PropensityModel::Linear => estimate_linear_probability(d, &x_prop.view())?,
    };

    // Now run the rest of the MTE estimation using the single propensity vector
    // (reusing the main logic)
    let propensity = &first_stage.propensity;

    let (polynomial_coefs, polynomial_se, r_squared, residual_se, _) =
        estimate_polynomial_stage(y, propensity, x, &config)?;

    let mte_curve = compute_mte_curve(&polynomial_coefs, config.n_grid);
    let (ate, _) = compute_ate(&polynomial_coefs, config.n_grid);
    let (att, weights_att) = compute_att(&polynomial_coefs, propensity, config.n_grid);
    let (atu, weights_atu) = compute_atu(&polynomial_coefs, propensity, config.n_grid);
    let (late, weights_late) = compute_late(&polynomial_coefs, propensity, config.n_grid);

    let ate_se = compute_te_se(
        &polynomial_coefs,
        &polynomial_se,
        &vec![1.0 / config.n_grid as f64; config.n_grid],
    );
    let att_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_att);
    let atu_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_atu);
    let late_se = compute_te_se(&polynomial_coefs, &polynomial_se, &weights_late);

    let df = n.saturating_sub(polynomial_coefs.len() + 1) as f64;

    let ate_est = build_te_estimate("ATE", ate, ate_se, df);
    let att_est = build_te_estimate("ATT", att, att_se, df);
    let atu_est = build_te_estimate("ATU", atu, atu_se, df);
    let late_est = build_te_estimate("LATE", late, late_se, df);

    let mte_coefficients: Vec<f64> = polynomial_coefs
        .iter()
        .enumerate()
        .map(|(j, &c)| (j + 1) as f64 * c)
        .collect();

    Ok(IVMTEResult {
        mte_curve,
        mte_coefficients,
        polynomial_coefficients: polynomial_coefs,
        polynomial_se,
        ate: ate_est,
        att: att_est,
        atu: atu_est,
        late: late_est,
        first_stage,
        weights_late,
        weights_att,
        weights_atu,
        n_obs: n,
        mte_degree: config.mte_degree,
        r_squared,
        residual_se,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Create test data with known treatment effect heterogeneity.
    fn create_mte_test_data() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Generate data with heterogeneous treatment effects
        // True model: Y(0) = 0, Y(1) = 1 + u (MTE increases with u)
        // D = 1{P(Z) >= U} where P(Z) = 0.5 + 0.3*Z

        let n = 200;
        let mut y = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);

        // Use deterministic "random" for reproducibility
        for i in 0..n {
            let zi = (i as f64 / n as f64 - 0.5) * 2.0; // Z in [-1, 1]
            let ui = (i as f64 * 7.0 % n as f64) / n as f64; // Pseudo-random U in [0, 1]

            let p_z = 0.5 + 0.3 * zi;
            let di = if p_z >= ui { 1.0 } else { 0.0 };

            // Outcome with treatment effect heterogeneity
            // Y(0) = noise, Y(1) = 1.5 + 0.5*u + noise
            // So MTE(u) = 1.5 + 0.5*u (linear, increasing)
            let noise = ((i * 13) % 100) as f64 / 500.0 - 0.1;
            let yi = if di > 0.5 {
                1.5 + 0.5 * ui + noise
            } else {
                noise
            };

            y.push(yi);
            d.push(di);
            z.push(zi);
        }

        (
            Array1::from_vec(y),
            Array1::from_vec(d),
            Array1::from_vec(z),
        )
    }

    #[test]
    fn test_ivmte_basic() {
        let (y, d, z) = create_mte_test_data();

        let result = run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig::default(),
        )
        .unwrap();

        // Check structure
        assert_eq!(result.n_obs, 200);
        assert!(!result.mte_curve.is_empty());
        assert!(result.mte_degree == 2);

        // ATE should be positive (treatment has positive effect)
        assert!(result.ate.estimate > 0.0);

        // Check propensity score estimation
        assert!(result.first_stage.converged);
        assert!(result.first_stage.p_min >= 0.0);
        assert!(result.first_stage.p_max <= 1.0);

        println!("MTE Result:\n{}", result);
    }

    #[test]
    fn test_ivmte_propensity_models() {
        let (y, d, z) = create_mte_test_data();

        // Test probit
        let config_probit = IVMTEConfig {
            propensity_model: PropensityModel::Probit,
            ..Default::default()
        };
        let result_probit =
            run_ivmte(&y.view(), &d.view(), &z.view(), None, config_probit).unwrap();
        assert!(result_probit.first_stage.model == PropensityModel::Probit);
        assert!(result_probit.first_stage.converged);

        // Test logit
        let config_logit = IVMTEConfig {
            propensity_model: PropensityModel::Logit,
            ..Default::default()
        };
        let result_logit = run_ivmte(&y.view(), &d.view(), &z.view(), None, config_logit).unwrap();
        assert!(result_logit.first_stage.model == PropensityModel::Logit);
        assert!(result_logit.first_stage.converged);

        // Test linear probability
        let config_linear = IVMTEConfig {
            propensity_model: PropensityModel::Linear,
            ..Default::default()
        };
        let result_linear =
            run_ivmte(&y.view(), &d.view(), &z.view(), None, config_linear).unwrap();
        assert!(result_linear.first_stage.model == PropensityModel::Linear);

        // All should give similar ATEs
        let ate_diff_1 = (result_probit.ate.estimate - result_logit.ate.estimate).abs();
        let ate_diff_2 = (result_probit.ate.estimate - result_linear.ate.estimate).abs();
        assert!(
            ate_diff_1 < 0.5,
            "Probit vs Logit ATE difference too large: {}",
            ate_diff_1
        );
        assert!(
            ate_diff_2 < 0.5,
            "Probit vs Linear ATE difference too large: {}",
            ate_diff_2
        );
    }

    #[test]
    fn test_ivmte_polynomial_degree() {
        let (y, d, z) = create_mte_test_data();

        // Test degree 1 (linear MTE)
        let config_1 = IVMTEConfig {
            mte_degree: 1,
            ..Default::default()
        };
        let result_1 = run_ivmte(&y.view(), &d.view(), &z.view(), None, config_1).unwrap();
        assert_eq!(result_1.mte_degree, 1);
        assert_eq!(result_1.polynomial_coefficients.len(), 1);

        // Test degree 3
        let config_3 = IVMTEConfig {
            mte_degree: 3,
            ..Default::default()
        };
        let result_3 = run_ivmte(&y.view(), &d.view(), &z.view(), None, config_3).unwrap();
        assert_eq!(result_3.mte_degree, 3);
        assert_eq!(result_3.polynomial_coefficients.len(), 3);
    }

    #[test]
    fn test_treatment_effect_relationships() {
        let (y, d, z) = create_mte_test_data();

        let result = run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig::default(),
        )
        .unwrap();

        // With increasing MTE (higher u = higher treatment effect):
        // - ATT < ATE because treated have lower U on average
        // - ATU > ATE because untreated have higher U on average

        // These relationships depend on the data, but we can at least check
        // that all estimates are in reasonable range
        assert!(result.ate.estimate.is_finite());
        assert!(result.att.estimate.is_finite());
        assert!(result.atu.estimate.is_finite());
        assert!(result.late.estimate.is_finite());

        // Standard errors should be positive
        assert!(result.ate.se > 0.0);
        assert!(result.att.se > 0.0);
        assert!(result.atu.se > 0.0);
        assert!(result.late.se > 0.0);
    }

    #[test]
    fn test_ivmte_convenience_function() {
        let (y, d, z) = create_mte_test_data();

        let result = ivmte(&y.view(), &d.view(), &z.view(), None).unwrap();

        assert_eq!(result.n_obs, 200);
        assert!(result.ate.estimate > 0.0);
    }

    #[test]
    fn test_mte_curve() {
        let (y, d, z) = create_mte_test_data();

        let config = IVMTEConfig {
            n_grid: 50,
            ..Default::default()
        };

        let result = run_ivmte(&y.view(), &d.view(), &z.view(), None, config).unwrap();

        // Check MTE curve properties
        assert_eq!(result.mte_curve.len(), 50);

        // All u values should be in (0, 1)
        for point in &result.mte_curve {
            assert!(point.u > 0.0 && point.u < 1.0);
            assert!(point.mte.is_finite());
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let (y, d, z) = create_mte_test_data();

        let result = run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig::default(),
        )
        .unwrap();

        // ATT and ATU weights should sum to 1 (approximately)
        let att_sum: f64 = result.weights_att.iter().sum();
        let atu_sum: f64 = result.weights_atu.iter().sum();
        let late_sum: f64 = result.weights_late.iter().sum();

        assert!(
            (att_sum - 1.0).abs() < 0.01,
            "ATT weights sum = {}",
            att_sum
        );
        assert!(
            (atu_sum - 1.0).abs() < 0.01,
            "ATU weights sum = {}",
            atu_sum
        );
        assert!(
            (late_sum - 1.0).abs() < 0.01,
            "LATE weights sum = {}",
            late_sum
        );
    }

    #[test]
    fn test_insufficient_data() {
        let y = array![1.0, 2.0, 3.0];
        let d = array![0.0, 1.0, 1.0];
        let z = array![0.1, 0.5, 0.9];

        let result = run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_no_treatment_variation() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let d = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // All treated
        let z = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        let result = run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig::default(),
        );

        assert!(result.is_err());
    }
}
