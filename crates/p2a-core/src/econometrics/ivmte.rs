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
//! ```text
//! MTE(x, u) = E[Y(1) - Y(0) | X = x, U_D = u]
//! ```
//!
//! ## Local IV (LIV) Estimator
//!
//! ```text
//! MTE(p) = ∂E[Y|P(Z) = p] / ∂p
//! ```
//!
//! ## Treatment Effect Parameters as Weighted Averages
//!
//! All standard treatment effect parameters are weighted integrals of the MTE
//! (Heckman & Vytlacil 2005, Table IB), evaluated on the midpoint grid
//! `u_i = (i + 0.5) / n_grid` with weights normalised to sum to one:
//!
//! - **ATE**: `h_ATE(u) = 1`
//! - **ATT**: `h_ATT(u) = Pr(P > u) / E[P]`
//! - **ATU**: `h_ATU(u) = Pr(P <= u) / E[1 - P]`
//! - **LATE(p_min, p_max)**: `h(u) = 1{p_min <= u <= p_max}`, the average MTE over the
//!   propensity support
//! - **PRTE**: user weights on the grid; for a policy that moves P to P',
//!   `h(u) = Pr(P' > u) - Pr(P > u)`
//!
//! ATE, ATT and ATU integrate the parametric MTE over all of [0, 1]: outside the
//! propensity support [p_min, p_max] the polynomial is extrapolated. The result reports
//! each estimand's share of weight outside the support and warns above 1%.
//!
//! # Parametric MTE Estimation
//!
//! 1. **First stage**: P(Z) by probit or logit (the shared GLM, `glm.fit`'s rule) or a
//!    linear probability model (fitted values clipped to [0, 1]).
//! 2. **Second stage**: least squares of Y on `[1, P, .., P^d]` (plus X when
//!    `include_covariates`).
//! 3. **MTE**: `MTE(u) = Σⱼ j γⱼ u^(j-1)`.
//! 4. **Inference**: delta method `sqrt(g' V g)` with V the HC1 covariance of the
//!    polynomial coefficients (Var(Y | P) varies with P by construction); the first-stage
//!    estimation error is not included. With `bootstrap_se`, units are resampled (seeded,
//!    `seed`, default [`DEFAULT_BOOTSTRAP_SEED`]), both stages are refitted, and SEs and
//!    percentile intervals come from the replicates.
//!
//! # References
//!
//! - Heckman, J.J., & Vytlacil, E. (2005). Structural equations, treatment effects,
//!   and econometric policy evaluation. *Econometrica*, 73(3), 669-738.
//!   https://doi.org/10.1111/j.1468-0262.2005.00594.x
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
//! R equivalent: no single function. The point-identified local-IV MTE of Brinch,
//! Mogstad & Wiswall (2017), as Stata `mtefe` with a polynomial; R `ivmte` (Shea &
//! Torgovitsky) also point-identifies with a parametric MTR but solves an LP. Validated
//! against the formulas written out in R (`validation/r/cases/econ_mte.R`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::econometrics::general_gmm::wald_z;
use crate::econometrics::glm::{GlmSpec, PROPENSITY_COEF_TOL, fit_glm};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{aliasing_message, least_squares};
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::stats::robust::percentile_interval;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

/// Default seed of the bootstrap resampling when [`IVMTEConfig::seed`] is `None`.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 42;

/// Largest share of an estimand's weight outside the propensity support that passes
/// without a warning.
const EXTRAPOLATION_WARN_SHARE: f64 = 0.01;

/// Estimand type for MTE analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MTEEstimand {
    /// Estimate the full MTE curve
    #[default]
    MTECurve,
    /// Average MTE over the propensity support, LATE(p_min, p_max)
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

crate::option_enum!(
    MTEEstimand,
    crate::options::Choices {
        option: "estimand",
        values: &[
            (&["mte_curve", "curve", "mte"], MTEEstimand::MTECurve),
            (&["late"], MTEEstimand::LATE),
            (&["att"], MTEEstimand::ATT),
            (&["atu"], MTEEstimand::ATU),
            (&["ate"], MTEEstimand::ATE),
            (&["prte"], MTEEstimand::PRTE),
        ],
    },
    no_display
);

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

crate::option_enum!(
    PropensityModel,
    crate::options::Choices {
        option: "propensity_model",
        values: &[
            (&["probit"], PropensityModel::Probit),
            (&["logit", "logistic"], PropensityModel::Logit),
            (&["linear", "lpm"], PropensityModel::Linear),
        ],
    },
    no_display
);

/// Configuration for MTE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVMTEConfig {
    /// Polynomial degree for MTE approximation (default: 2).
    /// Must be at least 1, and the second stage's 1 + degree (+ covariates) columns
    /// must be fewer than the observations.
    pub mte_degree: usize,

    /// The estimand reported as the target (default: the MTE curve, no target)
    pub estimand: MTEEstimand,

    /// Number of grid points `u_i = (i + 0.5) / n_grid` (default: 100; at least 2).
    pub n_grid: usize,

    /// Bootstrap SEs and percentile intervals (both stages refitted per replicate)
    pub bootstrap_se: bool,

    /// Number of bootstrap replications (default: 500; at least 2)
    pub n_bootstrap: usize,

    /// Propensity score model for first stage
    pub propensity_model: PropensityModel,

    /// Maximum IRLS iterations of the propensity model (default: 100; at least 1).
    pub max_iter: usize,

    /// Relative deviance tolerance of the propensity model (`glm.fit`'s rule; default
    /// 1e-8 = `glm.control()`). Must be finite and greater than 0.
    pub tolerance: f64,

    /// PRTE weights, one per grid point (finite, non-zero sum; normalised)
    pub prte_weights: Option<Vec<f64>>,

    /// Include covariates in the second stage (default: true)
    pub include_covariates: bool,

    /// Seed of the bootstrap resampling (None: [`DEFAULT_BOOTSTRAP_SEED`])
    #[serde(default)]
    pub seed: Option<u64>,

    /// Include the covariates X in the propensity model too (default: false)
    #[serde(default)]
    pub propensity_covariates: bool,
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
            seed: None,
            propensity_covariates: false,
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
    /// Standard error (delta method, or bootstrap SD)
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
    /// z statistic (estimate / se)
    pub t_stat: f64,
    /// Two-sided normal p-value
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
    /// Coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Variable names ("(Intercept)", instruments, covariates)
    pub variables: Vec<String>,
    /// Log-likelihood (for probit/logit)
    pub log_likelihood: Option<f64>,
    /// Pseudo R-squared, 1 - deviance / null deviance
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
    /// Convergence record of the GLM fit (probit and logit)
    #[serde(default)]
    pub convergence: Option<Convergence>,
}

/// Share of each estimand's weight on grid points outside the propensity support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtrapolationShares {
    pub ate: f64,
    pub att: f64,
    pub atu: f64,
    pub prte: Option<f64>,
}

/// What a bootstrap run did.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MteBootstrapInfo {
    /// Replications requested
    pub replications: usize,
    /// Replicates that failed and were dropped
    pub failed: usize,
    /// Seed of the resampling
    pub seed: u64,
}

/// Complete result from MTE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVMTEResult {
    /// MTE curve evaluated at grid points
    pub mte_curve: Vec<MTEPoint>,

    /// MTE coefficients: MTE(u) = sum_j mte_coefficients[j] u^j
    pub mte_coefficients: Vec<f64>,

    /// Coefficients on P, P^2, .., P^d in E[Y | X, P]
    pub polynomial_coefficients: Vec<f64>,

    /// Standard errors of the polynomial coefficients (HC1)
    pub polynomial_se: Vec<f64>,

    /// Average Treatment Effect
    pub ate: TreatmentEffectEstimate,

    /// Average Treatment Effect on the Treated
    pub att: TreatmentEffectEstimate,

    /// Average Treatment Effect on the Untreated
    pub atu: TreatmentEffectEstimate,

    /// Average MTE over the propensity support, LATE(p_min,p_max)
    pub late: TreatmentEffectEstimate,

    /// First-stage propensity estimation results
    pub first_stage: PropensityStageResult,

    /// LATE weights at each grid point (normalised)
    pub weights_late: Vec<f64>,

    /// ATT weights at each grid point (normalised)
    pub weights_att: Vec<f64>,

    /// ATU weights at each grid point (normalised)
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

    /// The configured estimand
    #[serde(default)]
    pub estimand: MTEEstimand,
    /// The estimate of the configured estimand (None for the MTE curve)
    #[serde(default)]
    pub target: Option<TreatmentEffectEstimate>,
    /// Policy-relevant treatment effect (when `prte_weights` is set)
    #[serde(default)]
    pub prte: Option<TreatmentEffectEstimate>,
    /// ATE weights at each grid point (uniform)
    #[serde(default)]
    pub weights_ate: Vec<f64>,
    /// PRTE weights at each grid point (normalised)
    #[serde(default)]
    pub weights_prte: Option<Vec<f64>>,
    /// Second-stage coefficient names: "(Intercept)", "P", "I(P^2)", .., covariates
    #[serde(default)]
    pub second_stage_names: Vec<String>,
    /// Second-stage coefficients, in the order of `second_stage_names`
    #[serde(default)]
    pub second_stage_coefficients: Vec<f64>,
    /// d x d HC1 covariance of the coefficients on [P, .., P^d]
    #[serde(default)]
    pub polynomial_vcov: Vec<Vec<f64>>,
    /// Grid points inside [p_min, p_max]
    #[serde(default)]
    pub n_grid_in_support: usize,
    /// Each estimand's share of weight outside the propensity support
    #[serde(default)]
    pub extrapolation: ExtrapolationShares,
    /// How the standard errors were computed
    #[serde(default)]
    pub se_method: String,
    /// The bootstrap, when `bootstrap_se`
    #[serde(default)]
    pub bootstrap: Option<MteBootstrapInfo>,
}

impl IVMTEResult {
    /// MTE(u) and its delta-method SE `sqrt(g' V g)`, `g = [(j + 1) u^j]`, V the HC1
    /// covariance of the polynomial coefficients.
    pub fn mte_delta_at(&self, u: f64) -> (f64, f64) {
        let g = mte_gradient(u, self.polynomial_coefficients.len());
        let est = g
            .iter()
            .zip(&self.polynomial_coefficients)
            .map(|(a, b)| a * b)
            .sum();
        let mut var = 0.0;
        for (i, gi) in g.iter().enumerate() {
            for (j, gj) in g.iter().enumerate() {
                var += gi * gj * self.polynomial_vcov[i][j];
            }
        }
        (est, var.max(0.0).sqrt())
    }
}

impl ResultNotes for IVMTEResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.first_stage.convergence.clone()
    }
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
            "  Propensity support: [{:.4}, {:.4}] ({} of {} grid points)",
            self.first_stage.p_min,
            self.first_stage.p_max,
            self.n_grid_in_support,
            self.mte_curve.len()
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

        if let Some(target) = &self.target {
            writeln!(f, "Target estimand: {}", target.name)?;
        }
        writeln!(f, "Standard errors: {}", self.se_method)?;
        writeln!(f, "Treatment Effect Estimates:")?;
        writeln!(
            f,
            "{:<18} {:>12} {:>10} {:>10} {:>20}",
            "Estimand", "Estimate", "Std Err", "P>|z|", "95% CI"
        )?;
        writeln!(f, "{}", "-".repeat(72))?;

        for te in [&self.ate, &self.att, &self.atu, &self.late]
            .into_iter()
            .chain(self.prte.as_ref())
        {
            writeln!(
                f,
                "{:<18} {:>12.4} {:>10.4} {:>10.4} [{:>8.4}, {:>8.4}]{}",
                te.name,
                te.estimate,
                te.se,
                te.p_value,
                te.ci_lower,
                te.ci_upper,
                te.significance.stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(72))?;
        let e = &self.extrapolation;
        write!(
            f,
            "Extrapolation outside the support: ATE {:.1}%, ATT {:.1}%, ATU {:.1}%",
            100.0 * e.ate,
            100.0 * e.att,
            100.0 * e.atu
        )?;
        if let Some(p) = e.prte {
            write!(f, ", PRTE {:.1}%", 100.0 * p)?;
        }
        writeln!(f)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }

        Ok(())
    }
}

/// Run Marginal Treatment Effects estimation with one instrument.
///
/// # Arguments
///
/// * `y` - Outcome variable (n x 1)
/// * `d` - Binary treatment indicator (n x 1, values 0 or 1)
/// * `z` - Instrument for treatment (n x 1)
/// * `x` - Optional covariates (n x p)
/// * `config` - Configuration options
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::ivmte::{run_ivmte, IVMTEConfig};
///
/// let result = run_ivmte(&y.view(), &d.view(), &z.view(), None, IVMTEConfig::default())?;
/// println!("ATE: {:.4}", result.ate.estimate);
/// ```
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
    let z2 = z.view().insert_axis(Axis(1));
    estimate_mte(*y, *d, z2, x.copied(), &config, &["z".to_string()])
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

/// Run MTE estimation with several instruments: the propensity model uses all of them.
pub fn run_ivmte_multi_z(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView2<f64>,
    x: Option<&ArrayView2<f64>>,
    config: IVMTEConfig,
) -> EconResult<IVMTEResult> {
    let names: Vec<String> = (1..=z.ncols()).map(|j| format!("z{j}")).collect();
    estimate_mte(*y, *d, *z, x.copied(), &config, &names)
}

/// The fit, then (with `bootstrap_se`) the bootstrap SEs.
fn estimate_mte(
    y: ArrayView1<f64>,
    d: ArrayView1<f64>,
    z: ArrayView2<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
    z_names: &[String],
) -> EconResult<IVMTEResult> {
    let fit = fit_mte(y, d, z, x, config, z_names)?;
    let mut result = fit.into_result(config);
    if config.bootstrap_se {
        let seed = config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED);
        let n = y.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // one resample at a time from the single stream: replicate b reads draws
        // b*n .. (b+1)*n, so no index matrix is ever materialised
        let boot = bootstrap_from_indices(y, d, z, x, config, config.n_bootstrap, |idx| {
            idx.extend((0..n).map(|_| rng.gen_range(0..n)));
        })?;
        apply_bootstrap(&mut result, &boot, config.n_bootstrap, seed)?;
    }
    Ok(result)
}

/// Replace the delta-method inference by the bootstrap's.
fn apply_bootstrap(
    result: &mut IVMTEResult,
    boot: &MteBootstrap,
    replications: usize,
    seed: u64,
) -> EconResult<()> {
    let ok = boot.estimands.nrows();
    if ok < 2.max(replications / 2) {
        return Err(EconError::Computation(format!(
            "only {ok} of {replications} bootstrap replicates succeeded ({})",
            boot.first_error.as_deref().unwrap_or("no error recorded")
        )));
    }
    if boot.failed > 0 {
        result.warnings.push(format!(
            "{} of {} bootstrap replicates failed ({}) and were dropped",
            boot.failed,
            replications,
            boot.first_error.as_deref().unwrap_or("unknown error")
        ));
    }
    let set = |te: &mut TreatmentEffectEstimate, col: ArrayView1<f64>| -> EconResult<()> {
        let s = boot_summary(col, 0.95)?;
        te.se = s.sd;
        te.t_stat = wald_z(te.estimate, s.sd);
        te.p_value = normal_p_value_two_sided(te.t_stat);
        te.ci_lower = s.ci_lower;
        te.ci_upper = s.ci_upper;
        te.significance = SignificanceLevel::from_p_value(te.p_value);
        Ok(())
    };
    set(&mut result.ate, boot.estimands.column(0))?;
    set(&mut result.att, boot.estimands.column(1))?;
    set(&mut result.atu, boot.estimands.column(2))?;
    set(&mut result.late, boot.estimands.column(3))?;
    if let Some(prte) = result.prte.as_mut() {
        set(prte, boot.estimands.column(4))?;
    }
    if let Some(t) = result.target.as_mut() {
        let col = match result.estimand {
            MTEEstimand::ATE => 0,
            MTEEstimand::ATT => 1,
            MTEEstimand::ATU => 2,
            MTEEstimand::LATE => 3,
            MTEEstimand::PRTE => 4,
            MTEEstimand::MTECurve => unreachable!("no target for the MTE curve"),
        };
        set(t, boot.estimands.column(col))?;
    }
    for (i, p) in result.mte_curve.iter_mut().enumerate() {
        resource::checkpoint()?;
        let s = boot_summary(boot.curve.column(i), 0.95)?;
        p.se = Some(s.sd);
        p.ci_lower = Some(s.ci_lower);
        p.ci_upper = Some(s.ci_upper);
    }
    result.se_method = format!(
        "bootstrap ({replications} replications, seed {seed}; both stages refitted; percentile \
         intervals)"
    );
    result.bootstrap = Some(MteBootstrapInfo {
        replications,
        failed: boot.failed,
        seed,
    });
    Ok(())
}

/// The bootstrap replicates that succeeded.
pub(crate) struct MteBootstrap {
    /// One row per replicate: ATE, ATT, ATU, LATE(p_min,p_max), then PRTE when set.
    pub(crate) estimands: Array2<f64>,
    /// One row per replicate: the MTE at each grid point.
    pub(crate) curve: Array2<f64>,
    pub(crate) failed: usize,
    pub(crate) first_error: Option<String>,
}

/// Mean, SD (n - 1) and the type-7 percentile interval of a column of draws.
pub(crate) struct BootSummary {
    pub(crate) mean: f64,
    pub(crate) sd: f64,
    pub(crate) ci_lower: f64,
    pub(crate) ci_upper: f64,
}

fn is_resource_stop(e: &EconError) -> bool {
    matches!(e, EconError::Cancelled(_) | EconError::ResourceLimit(_))
}

/// Refit both stages on each of `reps` resamples and keep the estimands and the curve.
/// `next` fills an empty, reused buffer with one replicate's row indices just before that
/// replicate is fitted, so the resamples are never held all at once.
/// A resource stop propagates; any other error counts the replicate as failed.
/// The estimand and curve stores (`reps x (4|5)` and `reps x n_grid`) are still allocated
/// up front, through `resource::try_zeros2`; the request limits bound both.
pub(crate) fn bootstrap_from_indices(
    y: ArrayView1<f64>,
    d: ArrayView1<f64>,
    z: ArrayView2<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
    reps: usize,
    mut next: impl FnMut(&mut Vec<usize>),
) -> EconResult<MteBootstrap> {
    let n_est = 4 + usize::from(config.prte_weights.is_some());
    let mut estimands = resource::try_zeros2(reps, n_est, "bootstrap estimands")?;
    let mut curve = resource::try_zeros2(reps, config.n_grid, "bootstrap MTE curves")?;
    let mut ok = 0;
    let mut failed = 0;
    let mut first_error = None;
    let names: Vec<String> = (1..=z.ncols()).map(|j| format!("z{j}")).collect();
    let mut idx = resource::try_with_capacity(y.len(), "bootstrap indices")?;
    for _ in 0..reps {
        resource::checkpoint()?;
        idx.clear();
        next(&mut idx);
        let yb = y.select(Axis(0), &idx);
        let db = d.select(Axis(0), &idx);
        let zb = z.select(Axis(0), &idx);
        let xb = x.map(|m| m.select(Axis(0), &idx));
        match fit_mte(
            yb.view(),
            db.view(),
            zb.view(),
            xb.as_ref().map(|m| m.view()),
            config,
            &names,
        ) {
            Ok(fit) => {
                let mut row = estimands.row_mut(ok);
                row[0] = fit.ate.estimate;
                row[1] = fit.att.estimate;
                row[2] = fit.atu.estimate;
                row[3] = fit.late.estimate;
                if let Some(p) = &fit.prte {
                    row[4] = p.estimate;
                }
                for (c, p) in curve.row_mut(ok).iter_mut().zip(&fit.curve) {
                    *c = p.mte;
                }
                ok += 1;
            }
            Err(e) if is_resource_stop(&e) => return Err(e),
            Err(e) => {
                failed += 1;
                first_error.get_or_insert_with(|| e.to_string());
            }
        }
    }
    Ok(MteBootstrap {
        estimands: estimands.slice(s![..ok, ..]).to_owned(),
        curve: curve.slice(s![..ok, ..]).to_owned(),
        failed,
        first_error,
    })
}

/// Mean, SD and the percentile interval at `level` of bootstrap draws (NaN refused).
pub(crate) fn boot_summary(draws: ArrayView1<f64>, level: f64) -> EconResult<BootSummary> {
    let v = draws.to_vec();
    let (ci_lower, ci_upper) = percentile_interval(&v, level)?;
    let n = v.len() as f64;
    // identical draws have mean exactly the draw and SD exactly 0 (the rounded mean
    // would differ in the last digit and leave an SD of ~1e-17)
    let (mean, sd) = match v.first() {
        Some(&v0) if v.iter().all(|&x| x == v0) => (v0, 0.0),
        _ => {
            let mean = v.iter().sum::<f64>() / n;
            (
                mean,
                (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt(),
            )
        }
    };
    Ok(BootSummary {
        mean,
        sd,
        ci_lower,
        ci_upper,
    })
}

/// `g(u) = [(j + 1) u^j]_{j < d}`: MTE(u) = g(u)' gamma.
fn mte_gradient(u: f64, d: usize) -> Vec<f64> {
    (0..d).map(|j| (j + 1) as f64 * u.powi(j as i32)).collect()
}

/// One MTE fit (shared by both entry points and every bootstrap replicate).
struct MteFit {
    n: usize,
    first_stage: PropensityStageResult,
    poly: Vec<f64>,
    poly_vcov: Array2<f64>,
    second_stage_names: Vec<String>,
    second_stage_coefficients: Vec<f64>,
    r_squared: f64,
    residual_se: f64,
    curve: Vec<MTEPoint>,
    weights: MteWeights,
    ate: TreatmentEffectEstimate,
    att: TreatmentEffectEstimate,
    atu: TreatmentEffectEstimate,
    late: TreatmentEffectEstimate,
    prte: Option<TreatmentEffectEstimate>,
    n_grid_in_support: usize,
    extrapolation: ExtrapolationShares,
    warnings: Vec<String>,
}

impl MteFit {
    fn into_result(self, config: &IVMTEConfig) -> IVMTEResult {
        let target = match config.estimand {
            MTEEstimand::MTECurve => None,
            MTEEstimand::ATE => Some(self.ate.clone()),
            MTEEstimand::ATT => Some(self.att.clone()),
            MTEEstimand::ATU => Some(self.atu.clone()),
            MTEEstimand::LATE => Some(self.late.clone()),
            MTEEstimand::PRTE => self.prte.clone(),
        };
        let d = self.poly.len();
        IVMTEResult {
            mte_curve: self.curve,
            mte_coefficients: self
                .poly
                .iter()
                .enumerate()
                .map(|(j, &c)| (j + 1) as f64 * c)
                .collect(),
            polynomial_se: (0..d)
                .map(|j| self.poly_vcov[[j, j]].max(0.0).sqrt())
                .collect(),
            polynomial_coefficients: self.poly,
            ate: self.ate,
            att: self.att,
            atu: self.atu,
            late: self.late,
            first_stage: self.first_stage,
            weights_late: self.weights.late,
            weights_att: self.weights.att,
            weights_atu: self.weights.atu,
            n_obs: self.n,
            mte_degree: d,
            r_squared: self.r_squared,
            residual_se: self.residual_se,
            warnings: self.warnings,
            estimand: config.estimand,
            target,
            prte: self.prte,
            weights_ate: self.weights.ate,
            weights_prte: self.weights.prte,
            second_stage_names: self.second_stage_names,
            second_stage_coefficients: self.second_stage_coefficients,
            polynomial_vcov: self.poly_vcov.outer_iter().map(|r| r.to_vec()).collect(),
            n_grid_in_support: self.n_grid_in_support,
            extrapolation: self.extrapolation,
            se_method: "delta method (HC1 covariance of the MTE polynomial; first-stage \
                        estimation error not included)"
                .to_string(),
            bootstrap: None,
        }
    }
}

/// Check the data and the controls of an MTE estimation.
fn validate(
    y: ArrayView1<f64>,
    d: ArrayView1<f64>,
    z: ArrayView2<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
) -> EconResult<()> {
    let n = y.len();
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "MTE estimation requires sufficient data".to_string(),
        });
    }
    if d.len() != n || z.nrows() != n || x.is_some_and(|m| m.nrows() != n) {
        return invalid(format!(
            "Dimension mismatch: y has {} obs, d has {}, z has {}{}",
            n,
            d.len(),
            z.nrows(),
            x.map_or(String::new(), |m| format!(", x has {}", m.nrows()))
        ));
    }
    if z.ncols() == 0 {
        return invalid("z must have at least one instrument column".to_string());
    }
    if d.iter().any(|&v| v != 0.0 && v != 1.0) {
        return invalid("d must be 0 or 1 (a binary treatment indicator)".to_string());
    }
    let n_treated = d.iter().filter(|&&v| v == 1.0).count();
    if n_treated == 0 || n_treated == n {
        return invalid("Treatment must have both treated and untreated observations".to_string());
    }
    for (what, finite) in [
        ("y", y.iter().all(|v| v.is_finite())),
        ("z", z.iter().all(|v| v.is_finite())),
        ("x", x.is_none_or(|m| m.iter().all(|v| v.is_finite()))),
    ] {
        if !finite {
            return invalid(format!("{what} contains non-finite values"));
        }
    }
    if config.n_grid < 2 {
        return invalid(format!("n_grid must be at least 2 (got {})", config.n_grid));
    }
    if config.mte_degree == 0 {
        return invalid("mte_degree must be at least 1 (got 0)".to_string());
    }
    let n_x_cols = if config.include_covariates {
        x.map_or(0, |m| m.ncols())
    } else {
        0
    };
    let total_cols = 1usize
        .checked_add(n_x_cols)
        .and_then(|c| c.checked_add(config.mte_degree));
    if total_cols.is_none_or(|c| c >= n) {
        let cols = total_cols.map_or_else(|| "more than usize::MAX".to_string(), |c| c.to_string());
        return invalid(format!(
            "mte_degree = {} is too large for {} observations: the second stage would have {} \
             columns and needs fewer columns than observations",
            config.mte_degree, n, cols
        ));
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return invalid(format!(
            "tolerance must be a finite number greater than 0 (got {})",
            config.tolerance
        ));
    }
    if config.max_iter == 0 {
        return invalid("max_iter must be at least 1 (got 0)".to_string());
    }
    resource::check_alloc::<MTEPoint>(config.n_grid, "MTE grid")?;
    if let Some(w) = &config.prte_weights {
        if w.len() != config.n_grid {
            return invalid(format!(
                "prte_weights must have one weight per grid point ({}), got {}",
                config.n_grid,
                w.len()
            ));
        }
        if w.iter().any(|v| !v.is_finite()) {
            return invalid("prte_weights contains non-finite values".to_string());
        }
        if w.iter().sum::<f64>() == 0.0 {
            return invalid("prte_weights sum to 0 and cannot be normalised".to_string());
        }
    } else if config.estimand == MTEEstimand::PRTE {
        return invalid(
            "estimand = prte needs prte_weights (one weight per grid point)".to_string(),
        );
    }
    if config.bootstrap_se {
        if config.n_bootstrap < 2 {
            return invalid(format!(
                "n_bootstrap must be at least 2 with bootstrap_se (got {})",
                config.n_bootstrap
            ));
        }
        let width = config
            .n_grid
            .checked_add(5)
            .ok_or_else(|| EconError::ResourceLimit("bootstrap grid too large".to_string()))?;
        resource::check_alloc::<f64>(
            resource::checked_len(&[config.n_bootstrap, width], "MTE bootstrap draws")?,
            "MTE bootstrap draws",
        )?;
    }
    Ok(())
}

/// The first stage: P(Z) by probit/logit (shared GLM) or a clipped linear probability model.
fn first_stage(
    d: ArrayView1<f64>,
    z: ArrayView2<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
    z_names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<PropensityStageResult> {
    let n = d.len();
    let x_cols = if config.propensity_covariates {
        x.map_or(0, |m| m.ncols())
    } else {
        0
    };
    let k = 1 + z.ncols() + x_cols;
    if k >= n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the propensity model has {k} columns and needs fewer than the {n} observations"
            ),
        });
    }
    let mut design = resource::try_zeros2(n, k, "MTE first-stage design")?;
    design.column_mut(0).fill(1.0);
    design.slice_mut(s![.., 1..1 + z.ncols()]).assign(&z);
    let mut variables = vec!["(Intercept)".to_string()];
    variables.extend(z_names.iter().cloned());
    if x_cols > 0
        && let Some(xm) = x
    {
        design.slice_mut(s![.., 1 + z.ncols()..]).assign(&xm);
        variables.extend((1..=x_cols).map(|j| format!("x{j}")));
    }
    let (
        coefficients,
        std_errors,
        propensity,
        log_likelihood,
        pseudo_r_squared,
        iterations,
        converged,
        convergence,
    ) = match config.propensity_model {
        PropensityModel::Probit | PropensityModel::Logit => {
            let probit = config.propensity_model == PropensityModel::Probit;
            let mut spec = if probit {
                GlmSpec::probit()
            } else {
                GlmSpec::logit()
            };
            spec.epsilon = config.tolerance;
            spec.max_iter = config.max_iter;
            if probit {
                // Fisher scoring converges linearly for the probit: iterate on the
                // coefficients too, to the tighter of the two tolerances
                spec.coef_tol = Some(PROPENSITY_COEF_TOL.min(config.tolerance));
            }
            let fit = fit_glm(design.view(), d, &spec)?;
            for w in &fit.warnings {
                warnings.push(format!("first stage: {w}"));
            }
            let criterion = if probit {
                "relative change in deviance, then in coefficients"
            } else {
                "relative change in deviance"
            };
            let conv = Convergence::new(
                fit.converged,
                fit.iterations,
                config.max_iter,
                criterion,
                f64::NAN,
                config.tolerance,
            );
            if let Some(note) = conv.note("first-stage GLM") {
                warnings.push(note);
            }
            let pseudo = if fit.null_deviance > 0.0 {
                Some(1.0 - fit.deviance / fit.null_deviance)
            } else {
                None
            };
            (
                fit.coefficients.to_vec(),
                fit.std_errors.to_vec(),
                fit.fitted,
                Some(fit.log_likelihood),
                pseudo,
                fit.iterations,
                fit.converged,
                Some(conv),
            )
        }
        PropensityModel::Linear => {
            let ls = least_squares(design.view(), d)?;
            if let Some(w) = ls.aliasing_warning(&variables) {
                warnings.push(format!("first stage: {w}"));
            }
            let ssr: f64 = ls.residuals.iter().map(|r| r * r).sum();
            let sigma2 = ssr / ls.df_residual.max(1) as f64;
            let se = ls
                .unscaled_cov
                .diag()
                .iter()
                .map(|v| (v * sigma2).max(0.0).sqrt())
                .collect();
            let mut clipped = 0;
            let p = ls.fitted.mapv(|v| {
                if !(0.0..=1.0).contains(&v) {
                    clipped += 1;
                }
                v.clamp(0.0, 1.0)
            });
            if clipped > 0 {
                warnings.push(format!(
                    "first stage: {clipped} linear-probability fitted values outside [0, 1] \
                         were clipped"
                ));
            }
            (ls.coefficients.to_vec(), se, p, None, None, 1, true, None)
        }
    };
    if propensity.iter().any(|p| !p.is_finite()) {
        return Err(EconError::Computation(
            "the propensity model produced non-finite fitted values".to_string(),
        ));
    }
    let p_min = propensity.iter().copied().fold(f64::INFINITY, f64::min);
    let p_max = propensity.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p_mean = propensity.sum() / n as f64;
    Ok(PropensityStageResult {
        model: config.propensity_model,
        coefficients,
        std_errors,
        variables,
        log_likelihood,
        pseudo_r_squared,
        iterations,
        converged,
        propensity,
        p_min,
        p_max,
        p_mean,
        convergence,
    })
}

/// The second stage on `[1, P, .., P^d, X]`: coefficients, the HC1 covariance of the
/// polynomial block, R^2 and the residual SE.
struct SecondStage {
    names: Vec<String>,
    coefficients: Vec<f64>,
    poly: Vec<f64>,
    poly_vcov: Array2<f64>,
    r_squared: f64,
    residual_se: f64,
}

fn second_stage(
    y: ArrayView1<f64>,
    p: &Array1<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
    warnings: &mut Vec<String>,
) -> EconResult<SecondStage> {
    let n = y.len();
    let deg = config.mte_degree;
    let xm = x.filter(|_| config.include_covariates);
    let n_x = xm.map_or(0, |m| m.ncols());
    let k = 1 + deg + n_x;
    let mut design = resource::try_zeros2(n, k, "MTE second-stage design")?;
    for (i, &pi) in p.iter().enumerate() {
        let mut row = design.row_mut(i);
        row[0] = 1.0;
        let mut pw = 1.0;
        for j in 0..deg {
            pw *= pi;
            row[1 + j] = pw;
        }
    }
    let mut names = vec!["(Intercept)".to_string(), "P".to_string()];
    names.extend((2..=deg).map(|j| format!("I(P^{j})")));
    if let Some(xm) = xm {
        design.slice_mut(s![.., 1 + deg..]).assign(&xm);
        names.extend((1..=n_x).map(|j| format!("x{j}")));
    }
    let sol = solve_ols(design.view(), y)?;
    if let Some(&j) = sol.aliased.iter().find(|&&j| (1..=deg).contains(&j)) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the degree-{deg} MTE polynomial is not identified: P^{j} is collinear with the \
                 other columns (the propensity takes too few distinct values; lower mte_degree)"
            ),
        });
    }
    if let Some(w) = aliasing_message(&sol.aliased, &names) {
        warnings.push(w);
    }
    // HC1 on the identified columns: n / (n - rank) U (sum e_i^2 w_i w_i') U
    let ids: Vec<usize> = (0..k).filter(|j| !sol.aliased.contains(j)).collect();
    let w = design.select(Axis(1), &ids);
    let u = sol.unscaled.select(Axis(0), &ids).select(Axis(1), &ids);
    let we = &w * &sol.residuals.mapv(|e| e.abs()).insert_axis(Axis(1));
    let meat = we.t().dot(&we);
    let rank = ids.len();
    let df = n.saturating_sub(rank);
    let scale = if df > 0 { n as f64 / df as f64 } else { 1.0 };
    let v = u.dot(&meat).dot(&u) * scale;
    // the polynomial columns 1..=deg are identified, so their positions in `ids` are 1..=deg
    // unless the intercept was aliased
    let pos: Vec<usize> = (1..=deg)
        .map(|j| ids.iter().position(|&c| c == j).expect("identified"))
        .collect();
    let poly_vcov = Array2::from_shape_fn((deg, deg), |(a, b)| {
        0.5 * (v[[pos[a], pos[b]]] + v[[pos[b], pos[a]]])
    });
    let ssr: f64 = sol.residuals.iter().map(|r| r * r).sum();
    let y_mean = y.sum() / n as f64;
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    Ok(SecondStage {
        names,
        coefficients: sol.beta.to_vec(),
        poly: sol.beta.slice(s![1..=deg]).to_vec(),
        poly_vcov,
        r_squared: if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 },
        residual_se: (ssr / df.max(1) as f64).sqrt(),
    })
}

/// The normalised weights on the grid.
struct MteWeights {
    u: Vec<f64>,
    ate: Vec<f64>,
    att: Vec<f64>,
    atu: Vec<f64>,
    late: Vec<f64>,
    prte: Option<Vec<f64>>,
    in_support: Vec<bool>,
}

fn normalise(mut w: Vec<f64>) -> Vec<f64> {
    let s: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= s);
    w
}

/// Heckman-Vytlacil weights on the grid `u_i = (i + 0.5) / n_grid` from the propensity
/// scores (`O(n log n + n_grid log n)`).
fn mte_weights(p: &Array1<f64>, n_grid: usize, prte: Option<&[f64]>) -> EconResult<MteWeights> {
    let n = p.len();
    let mut sorted = p.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (p_min, p_max) = (sorted[0], sorted[n - 1]);
    let p_mean = sorted.iter().sum::<f64>() / n as f64;
    let nf = n as f64;
    let mut u = resource::try_with_capacity(n_grid, "MTE grid")?;
    let mut ate = resource::try_with_capacity(n_grid, "ATE weights")?;
    let mut att = resource::try_with_capacity(n_grid, "ATT weights")?;
    let mut atu = resource::try_with_capacity(n_grid, "ATU weights")?;
    let mut late = resource::try_with_capacity(n_grid, "LATE weights")?;
    let mut in_support = resource::try_with_capacity(n_grid, "MTE support")?;
    for i in 0..n_grid {
        if i % 1024 == 0 {
            resource::checkpoint()?;
        }
        let ui = (i as f64 + 0.5) / n_grid as f64;
        let below = sorted.partition_point(|&v| v <= ui) as f64;
        u.push(ui);
        ate.push(1.0);
        att.push((nf - below) / nf / p_mean);
        atu.push(below / nf / (1.0 - p_mean));
        let inside = p_min <= ui && ui <= p_max;
        late.push(if inside { 1.0 } else { 0.0 });
        in_support.push(inside);
    }
    if !in_support.iter().any(|&b| b) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the propensity support [{p_min:.4}, {p_max:.4}] contains none of the {n_grid} \
                 grid points; LATE(p_min,p_max) is not defined on the grid (increase n_grid)"
            ),
        });
    }
    Ok(MteWeights {
        u,
        ate: normalise(ate),
        att: normalise(att),
        atu: normalise(atu),
        late: normalise(late),
        prte: prte.map(|w| normalise(w.to_vec())),
        in_support,
    })
}

/// `sum_i w_i g(u_i)`, the estimand's gradient with respect to the polynomial.
fn weighted_gradient(u: &[f64], w: &[f64], d: usize) -> EconResult<Vec<f64>> {
    let mut g = vec![0.0; d];
    for (i, (&ui, &wi)) in u.iter().zip(w).enumerate() {
        if i % 1024 == 0 {
            resource::checkpoint()?;
        }
        let mut pw = wi;
        for (j, gj) in g.iter_mut().enumerate() {
            *gj += (j + 1) as f64 * pw;
            pw *= ui;
        }
    }
    Ok(g)
}

fn quad(g: &[f64], v: &Array2<f64>) -> f64 {
    let mut s = 0.0;
    for (i, gi) in g.iter().enumerate() {
        for (j, gj) in g.iter().enumerate() {
            s += gi * gj * v[[i, j]];
        }
    }
    s
}

/// An estimate with its delta-method inference (normal reference).
fn te_estimate(name: &str, estimate: f64, se: f64, z: f64) -> TreatmentEffectEstimate {
    let t_stat = wald_z(estimate, se);
    let p_value = normal_p_value_two_sided(t_stat);
    TreatmentEffectEstimate {
        name: name.to_string(),
        estimate,
        se,
        t_stat,
        p_value,
        ci_lower: estimate - z * se,
        ci_upper: estimate + z * se,
        significance: SignificanceLevel::from_p_value(p_value),
    }
}

/// The estimand of weights `w`: `g'gamma` with SE `sqrt(g'Vg)`.
fn te_from_weights(
    name: &str,
    u: &[f64],
    w: &[f64],
    poly: &[f64],
    v: &Array2<f64>,
    z: f64,
) -> EconResult<TreatmentEffectEstimate> {
    let g = weighted_gradient(u, w, poly.len())?;
    let est = g.iter().zip(poly).map(|(a, b)| a * b).sum();
    Ok(te_estimate(name, est, quad(&g, v).max(0.0).sqrt(), z))
}

/// The MTE curve with pointwise delta-method SEs and CIs.
fn mte_curve_points(u: &[f64], poly: &[f64], v: &Array2<f64>, z: f64) -> EconResult<Vec<MTEPoint>> {
    let mut curve = resource::try_with_capacity(u.len(), "MTE curve")?;
    for (i, &ui) in u.iter().enumerate() {
        if i % 1024 == 0 {
            resource::checkpoint()?;
        }
        let g = mte_gradient(ui, poly.len());
        let mte: f64 = g.iter().zip(poly).map(|(a, b)| a * b).sum();
        let se = quad(&g, v).max(0.0).sqrt();
        curve.push(MTEPoint {
            u: ui,
            mte,
            se: Some(se),
            ci_lower: Some(mte - z * se),
            ci_upper: Some(mte + z * se),
        });
    }
    Ok(curve)
}

/// Share of the weight outside the support.
fn outside_share(w: &[f64], in_support: &[bool]) -> f64 {
    w.iter()
        .zip(in_support)
        .filter(|&(_, &inside)| !inside)
        .map(|(v, _)| v)
        .sum()
}

/// Validate, fit both stages and compute every estimand with its delta-method SE.
fn fit_mte(
    y: ArrayView1<f64>,
    d: ArrayView1<f64>,
    z: ArrayView2<f64>,
    x: Option<ArrayView2<f64>>,
    config: &IVMTEConfig,
    z_names: &[String],
) -> EconResult<MteFit> {
    validate(y, d, z, x, config)?;
    let n = y.len();
    let mut warnings = Vec::new();
    let first = first_stage(d, z, x, config, z_names, &mut warnings)?;
    let p = &first.propensity;
    let p_var = p.iter().map(|&v| (v - first.p_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    if p_var < 0.01 {
        warnings.push(format!(
            "Low variance in propensity scores (Var = {p_var:.4}). Instrument may be weak."
        ));
    }
    let second = second_stage(y, p, x, config, &mut warnings)?;
    let weights = mte_weights(p, config.n_grid, config.prte_weights.as_deref())?;
    let zc = z_critical(0.95)?;
    let (poly, v, u) = (&second.poly, &second.poly_vcov, &weights.u);
    let ate = te_from_weights("ATE", u, &weights.ate, poly, v, zc)?;
    let att = te_from_weights("ATT", u, &weights.att, poly, v, zc)?;
    let atu = te_from_weights("ATU", u, &weights.atu, poly, v, zc)?;
    let late = te_from_weights("LATE(p_min,p_max)", u, &weights.late, poly, v, zc)?;
    let prte = match &weights.prte {
        Some(w) => Some(te_from_weights("PRTE", u, w, poly, v, zc)?),
        None => None,
    };
    let curve = mte_curve_points(u, poly, v, zc)?;
    let n_grid_in_support = weights.in_support.iter().filter(|&&b| b).count();
    let extrapolation = ExtrapolationShares {
        ate: outside_share(&weights.ate, &weights.in_support),
        att: outside_share(&weights.att, &weights.in_support),
        atu: outside_share(&weights.atu, &weights.in_support),
        prte: weights
            .prte
            .as_ref()
            .map(|w| outside_share(w, &weights.in_support)),
    };
    let largest = [extrapolation.ate, extrapolation.att, extrapolation.atu]
        .into_iter()
        .chain(extrapolation.prte)
        .fold(0.0_f64, f64::max);
    if largest > EXTRAPOLATION_WARN_SHARE {
        let prte_part = extrapolation
            .prte
            .map_or(String::new(), |s| format!(", PRTE {:.1}%", 100.0 * s));
        warnings.push(format!(
            "The propensity support [{:.4}, {:.4}] covers {} of the {} grid points; ATE, ATT and \
             ATU extrapolate the degree-{} MTE polynomial outside it (share of weight outside the \
             support: ATE {:.1}%, ATT {:.1}%, ATU {:.1}%{})",
            first.p_min,
            first.p_max,
            n_grid_in_support,
            config.n_grid,
            config.mte_degree,
            100.0 * extrapolation.ate,
            100.0 * extrapolation.att,
            100.0 * extrapolation.atu,
            prte_part
        ));
    }
    Ok(MteFit {
        n,
        first_stage: first,
        poly: second.poly,
        poly_vcov: second.poly_vcov,
        second_stage_names: second.names,
        second_stage_coefficients: second.coefficients,
        r_squared: second.r_squared,
        residual_se: second.residual_se,
        curve,
        weights,
        ate,
        att,
        atu,
        late,
        prte,
        n_grid_in_support,
        extrapolation,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::regression::ols::na_fixtures::{nan_vec, repo_dataset};
    use crate::resource::{Budget, CancelToken, with_budget};

    use ndarray::array;
    use p2a_validation::{Check, Tol};

    /// Identical bootstrap draws have mean exactly the draw and SD exactly 0.
    #[test]
    fn test_boot_summary_identical_draws_exact() {
        let draws = Array1::from_elem(7, 0.1);
        let s = boot_summary(draws.view(), 0.95).unwrap();
        assert_eq!(s.mean, 0.1);
        assert_eq!(s.sd, 0.0);
    }

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
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

    fn expect_resource_limit<T>(res: EconResult<T>) {
        match res {
            Err(EconError::ResourceLimit(_)) => {}
            Err(e) => panic!("expected ResourceLimit, got: {e}"),
            Ok(_) => panic!("expected ResourceLimit, got Ok"),
        }
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

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

        // MTE increasing in u (Y(1) - Y(0) = 1.5 + 0.5 u): the treated have low U_D
        assert!(
            result.att.estimate < result.ate.estimate && result.ate.estimate < result.atu.estimate,
            "ATT {} ATE {} ATU {}",
            result.att.estimate,
            result.ate.estimate,
            result.atu.estimate
        );

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

    #[test]
    fn test_ivmte_rejects_degenerate_controls() {
        let (y, d, z) = create_mte_test_data(); // n = 200
        let run = |config: IVMTEConfig| run_ivmte(&y.view(), &d.view(), &z.view(), None, config);
        expect_invalid(
            run(IVMTEConfig {
                n_grid: 0,
                ..Default::default()
            }),
            "n_grid",
        );
        expect_invalid(
            run(IVMTEConfig {
                mte_degree: 0,
                ..Default::default()
            }),
            "mte_degree",
        );
        expect_invalid(
            run(IVMTEConfig {
                mte_degree: 199,
                ..Default::default()
            }),
            "mte_degree",
        );
        expect_invalid(
            run(IVMTEConfig {
                tolerance: 0.0,
                ..Default::default()
            }),
            "tolerance",
        );
        expect_invalid(
            run(IVMTEConfig {
                tolerance: f64::NAN,
                ..Default::default()
            }),
            "tolerance",
        );
        expect_invalid(
            run(IVMTEConfig {
                max_iter: 0,
                ..Default::default()
            }),
            "max_iter",
        );
        let z2 = z.clone().insert_axis(ndarray::Axis(1));
        expect_invalid(
            run_ivmte_multi_z(
                &y.view(),
                &d.view(),
                &z2.view(),
                None,
                IVMTEConfig {
                    n_grid: 0,
                    ..Default::default()
                },
            ),
            "n_grid",
        );
        expect_invalid(
            run(IVMTEConfig {
                mte_degree: 3,
                n_grid: 1,
                ..Default::default()
            }),
            "n_grid",
        );
        // the support [0.2852, 0.7239] of this data holds u = 0.5 of the 3-point grid (the
        // 2-point grid's 0.25 and 0.75 lie outside it)
        let three = run(IVMTEConfig {
            mte_degree: 3,
            n_grid: 3,
            ..Default::default()
        });
        assert!(three.is_ok(), "{:?}", three.err());
    }

    #[test]
    fn test_ivmte_rejects_unallocatable_grid() {
        let (y, d, z) = create_mte_test_data();
        expect_resource_limit(run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig {
                n_grid: usize::MAX / 4,
                ..Default::default()
            },
        ));
    }

    #[test]
    fn test_ivmte_stops_when_cancelled() {
        let (y, d, z) = create_mte_test_data();
        expect_cancelled(with_budget(cancelled(), || {
            run_ivmte(
                &y.view(),
                &d.view(),
                &z.view(),
                None,
                IVMTEConfig::default(),
            )
        }));
    }

    /// The linear first stage has no iteration loop, so a cancelled budget is
    /// first seen in the grid loops after the second stage.
    #[test]
    fn test_ivmte_linear_first_stage_stops_when_cancelled() {
        let (y, d, z) = create_mte_test_data();
        let config = IVMTEConfig {
            propensity_model: PropensityModel::Linear,
            ..Default::default()
        };
        expect_cancelled(with_budget(cancelled(), || {
            run_ivmte(&y.view(), &d.view(), &z.view(), None, config)
        }));
    }

    /// Each grid loop checks the budget (every 1024 grid points, from the first), so each
    /// one is called directly: an end-to-end run stops at the first of them.
    #[test]
    fn test_ivmte_grid_loops_stop_when_cancelled() {
        let poly = [1.5, 0.5];
        let v = Array2::from_diag(&array![0.01, 0.0025]);
        let propensity = Array1::from_vec((0..20).map(|i| 0.2 + 0.03 * i as f64).collect());
        let n_grid = 10;
        let w = mte_weights(&propensity, n_grid, None).unwrap();
        assert!(te_from_weights("ATE", &w.u, &w.ate, &poly, &v, 1.96).is_ok());
        assert!(mte_curve_points(&w.u, &poly, &v, 1.96).is_ok());

        let stops = |r: EconResult<()>| matches!(r, Err(EconError::Cancelled(_)));
        let not_stopped: Vec<&str> = with_budget(cancelled(), || {
            [
                (
                    "mte_weights",
                    stops(mte_weights(&propensity, n_grid, None).map(drop)),
                ),
                (
                    "te_from_weights",
                    stops(te_from_weights("ATE", &w.u, &w.ate, &poly, &v, 1.96).map(drop)),
                ),
                (
                    "mte_curve_points",
                    stops(mte_curve_points(&w.u, &poly, &v, 1.96).map(drop)),
                ),
            ]
            .into_iter()
            .filter_map(|(name, stopped)| (!stopped).then_some(name))
            .collect()
        });
        assert!(
            not_stopped.is_empty(),
            "these grid loops ignored a cancelled budget: {not_stopped:?}"
        );
    }

    /// A cancelled budget inside the bootstrap propagates as Cancelled instead of counting
    /// the replicates as failed.
    #[test]
    fn test_mte_stops_when_cancelled_in_bootstrap() {
        let (y, d, z) = create_mte_test_data();
        let config = IVMTEConfig {
            bootstrap_se: true,
            n_bootstrap: 20,
            ..Default::default()
        };
        let z2 = z.clone().insert_axis(ndarray::Axis(1));
        let fit = fit_mte(
            y.view(),
            d.view(),
            z2.view(),
            None,
            &config,
            &["z".to_string()],
        );
        assert!(fit.is_ok());
        let idx: Vec<Vec<usize>> = (0..20)
            .map(|b| (0..200).map(|i| (i * 7 + b) % 200).collect())
            .collect();
        expect_cancelled(with_budget(cancelled(), || {
            bootstrap_from_indices(y.view(), d.view(), z2.view(), None, &config, idx.len(), {
                let mut it = idx.iter();
                move |buf| buf.extend_from_slice(it.next().unwrap())
            })
        }));
        expect_cancelled(with_budget(cancelled(), || {
            run_ivmte(&y.view(), &d.view(), &z.view(), None, config.clone())
        }));
    }

    const EST_SE: [&str; 2] = ["est", "se"];

    /// rows of mte_sim (y, d, [z1, z2]) for which keep(row, z1) holds
    fn mte_data(
        ds: &Dataset,
        keep: impl Fn(usize, f64) -> bool,
    ) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let (y, d, z1, z2) = (
            nan_vec(ds, "y"),
            nan_vec(ds, "d"),
            nan_vec(ds, "z1"),
            nan_vec(ds, "z2"),
        );
        let rows: Vec<usize> = (0..y.len()).filter(|&i| keep(i, z1[i])).collect();
        let z = Array2::from_shape_fn((rows.len(), 2), |(r, j)| {
            if j == 0 { z1[rows[r]] } else { z2[rows[r]] }
        });
        (
            rows.iter().map(|&i| y[i]).collect(),
            rows.iter().map(|&i| d[i]).collect(),
            z,
        )
    }

    /// The data of case `mte_local_iv`, for property tests that compare no R value.
    fn mte_sim() -> Dataset {
        repo_dataset("validation/datasets/generated/mte_sim.csv")
    }

    fn case_config() -> IVMTEConfig {
        IVMTEConfig {
            mte_degree: 3,
            n_grid: 1001,
            tolerance: 1e-14, // glm.control(epsilon = 1e-14), as the case
            ..Default::default()
        }
    }

    fn pin_te(check: &mut Check<'_>, key: &str, t: &TreatmentEffectEstimate, n: usize) {
        let v = [t.estimate, t.se];
        check
            .named_subset(key, &["est"], &EST_SE, &v, Tol::coef(n))
            .named_subset(key, &["se"], &EST_SE, &v, Tol::se(n));
    }

    /// Case `mte_local_iv` (no R package; Heckman & Vytlacil 2005 weights written out in R on
    /// the 1001-point grid, probit first stage, cubic in P, delta method with the full HC1
    /// covariance). Fails before: ATT/ATU used the CDF/survival weights (ATT
    /// 1.0965359926919187, ATU 1.6333100706761692 against 1.83006971114244 /
    /// 0.8577795562869492), and the SEs dropped the covariances and assumed homoskedasticity
    /// (ATE 3.0712603129759262 against 0.16241927186853242).
    #[test]
    fn test_validate_mte_local_iv_estimands() {
        let (case, ds) = crate::test_support::load_case("mte_local_iv");
        let n = case.scalar("n") as usize;
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let r = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, case_config()).unwrap();
        let mut check = Check::new(&case);
        // the probit MLE on both sides (glm IRLS at epsilon 1e-14, then Newton in R; IRLS with
        // the coefficient rule at 1e-14 in p2a)
        check
            .named(
                "fs_coef",
                &r.first_stage.variables,
                &r.first_stage.coefficients,
                Tol::iterative(1e-10),
            )
            // the cubic's coefficients amplify the first stage's convergence error (a probit
            // coefficient 1.6e-11 relative from the MLE moves I(P^3) by 4.7e-9 relative), so
            // the second-stage polynomial and MTE(u) use Tol::iterative with the README value
            // for n <= 1000; R's probit is polished to the MLE by Newton (econ_mte.R)
            .named(
                "mte_coef",
                &r.second_stage_names,
                &r.second_stage_coefficients,
                Tol::iterative(1e-8),
            )
            .scalar("p_min", r.first_stage.p_min, Tol::stat(n))
            .scalar("p_max", r.first_stage.p_max, Tol::stat(n));
        for (key, t) in [
            ("ate", &r.ate),
            ("att", &r.att),
            ("atu", &r.atu),
            ("late_support", &r.late),
        ] {
            pin_te(&mut check, key, t, n);
        }
        let at: Vec<(f64, f64)> = [0.1, 0.5, 0.9].iter().map(|&u| r.mte_delta_at(u)).collect();
        check
            .vector(
                "mte_at",
                &at.iter().map(|p| p.0).collect::<Vec<_>>(),
                Tol::iterative(1e-8),
            )
            .vector(
                "mte_se_at",
                &at.iter().map(|p| p.1).collect::<Vec<_>>(),
                Tol::se(n),
            );
        check.finish();
        assert_eq!(r.late.name, "LATE(p_min,p_max)");
        // MTE(u) = 2.5 - 2u: selection on gains
        assert!(r.att.estimate > r.ate.estimate && r.ate.estimate > r.atu.estimate);
        assert!(
            r.mte_curve
                .iter()
                .all(|p| p.se.is_some() && p.ci_lower.is_some() && p.ci_upper.is_some())
        );
        assert!(
            r.first_stage
                .convergence
                .as_ref()
                .is_some_and(|c| c.converged)
        );
        assert_eq!(r.first_stage.variables, ["(Intercept)", "z1", "z2"]);
        assert!(r.se_method.starts_with("delta method (HC1"));
    }

    /// Case `mte_local_iv`, subsample |z1| < 0.5 (719 rows): support [0.2914, 0.8321] holds
    /// 541 of 1001 grid points; ATE/ATT/ATU put 46.0% / 51.1% / 39.1% of their weight outside
    /// it. Fails before: the only support warning fired at full support (p_min < 0.01) and
    /// nothing was said here (ECON-87).
    #[test]
    fn test_validate_mte_narrow_support() {
        let (case, ds) = crate::test_support::load_case("mte_local_iv");
        let (y, d, z) = mte_data(&ds, |_, z1| z1.abs() < 0.5);
        let n = y.len();
        assert_eq!(n as f64, case.scalar("narrow_n"));
        let r = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, case_config()).unwrap();
        let mut check = Check::new(&case);
        check
            .scalar("narrow_p_min", r.first_stage.p_min, Tol::stat(n))
            .scalar("narrow_p_max", r.first_stage.p_max, Tol::stat(n))
            .scalar(
                "narrow_n_grid_in_support",
                r.n_grid_in_support as f64,
                Tol::exact(),
            )
            .scalar(
                "narrow_share_ate_weight_outside_support",
                r.extrapolation.ate,
                Tol::stat(n),
            )
            .scalar(
                "narrow_share_att_weight_outside_support",
                r.extrapolation.att,
                Tol::stat(n),
            )
            .scalar(
                "narrow_share_atu_weight_outside_support",
                r.extrapolation.atu,
                Tol::stat(n),
            );
        for (key, t) in [
            ("narrow_ate", &r.ate),
            ("narrow_att", &r.att),
            ("narrow_atu", &r.atu),
            ("narrow_late_support", &r.late),
        ] {
            pin_te(&mut check, key, t, n);
        }
        let (yf, df, zf) = mte_data(&ds, |_, _| true);
        let full =
            run_ivmte_multi_z(&yf.view(), &df.view(), &zf.view(), None, case_config()).unwrap();
        check
            .scalar(
                "n_grid_in_support",
                full.n_grid_in_support as f64,
                Tol::exact(),
            )
            .scalar(
                "share_ate_weight_outside_support",
                full.extrapolation.ate,
                Tol::stat(2000),
            )
            .scalar(
                "share_att_weight_outside_support",
                full.extrapolation.att,
                Tol::stat(2000),
            )
            .scalar(
                "share_atu_weight_outside_support",
                full.extrapolation.atu,
                Tol::stat(2000),
            );
        check.finish();
        let warned: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("extrapolate"))
            .collect();
        assert_eq!(warned.len(), 1, "{:?}", r.warnings);
        for part in [
            "[0.2914, 0.8321]",
            "541 of the 1001",
            "ATE 46.0%",
            "ATT 51.1%",
            "ATU 39.1%",
        ] {
            assert!(warned[0].contains(part), "{part}: {}", warned[0]);
        }
        assert!(
            !full
                .warnings
                .iter()
                .any(|w| w.contains("extrapolate") || w.contains("near boundaries")),
            "{:?}",
            full.warnings
        );
    }

    #[test]
    fn test_mte_rejects_degenerate_grid_and_support() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let run = |c: IVMTEConfig| run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c);
        for n_grid in [0, 1] {
            let c = IVMTEConfig {
                n_grid,
                ..case_config()
            };
            expect_invalid(run(c), "n_grid");
        }
        let mut half = d.clone();
        half[0] = 0.5;
        expect_invalid(
            run_ivmte_multi_z(&y.view(), &half.view(), &z.view(), None, case_config()),
            "0 or 1",
        );
        // |z1| < 0.05 (77 rows), z1 only: R's probit support [0.5635, 0.7173] holds neither
        // 0.25 nor 0.75
        let (ys, dsub, zs) = mte_data(&ds, |_, z1| z1.abs() < 0.05);
        let z1 = zs.column(0).to_owned();
        let c = IVMTEConfig {
            n_grid: 2,
            mte_degree: 1,
            ..Default::default()
        };
        expect_invalid(
            run_ivmte(&ys.view(), &dsub.view(), &z1.view(), None, c),
            "grid points",
        );
    }

    /// Case `mte_local_iv`: PRTE of the policy P' = min(P + 0.1, 1), weights
    /// w(u) = Pr(P' > u) - Pr(P > u) (Heckman & Vytlacil 2005, Table IB) passed as
    /// prte_weights; estimand = PRTE makes it the target. With the ATT weights the PRTE is
    /// the ATT. Fails before: prte_weights and estimand were never read.
    #[test]
    fn test_validate_mte_prte_shift() {
        let (case, ds) = crate::test_support::load_case("mte_local_iv");
        let n = case.scalar("n") as usize;
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let base = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, case_config()).unwrap();
        let p = &base.first_stage.propensity;
        let w: Vec<f64> = base
            .mte_curve
            .iter()
            .map(|pt| {
                p.iter().filter(|&&pi| (pi + 0.1).min(1.0) > pt.u).count() as f64
                    - p.iter().filter(|&&pi| pi > pt.u).count() as f64
            })
            .collect();
        let c = IVMTEConfig {
            prte_weights: Some(w),
            estimand: MTEEstimand::PRTE,
            ..case_config()
        };
        let r = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c).unwrap();
        let mut check = Check::new(&case);
        pin_te(&mut check, "prte_shift", r.prte.as_ref().expect("PRTE"), n);
        check.finish();
        assert_eq!(r.target.as_ref().map(|t| t.name.as_str()), Some("PRTE"));
        assert_eq!(r.extrapolation.prte, Some(0.0));
        let c = IVMTEConfig {
            prte_weights: Some(base.weights_att.clone()),
            ..case_config()
        };
        let a = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c).unwrap();
        assert!((a.prte.unwrap().estimate / base.att.estimate - 1.0).abs() < 1e-12);
        assert!(base.target.is_none() && base.prte.is_none());
    }

    #[test]
    fn test_mte_estimand_and_prte_validation() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let run = |c: IVMTEConfig| run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c);
        let c = IVMTEConfig {
            estimand: MTEEstimand::PRTE,
            ..Default::default()
        };
        expect_invalid(run(c), "prte_weights");
        let mut with_nan = vec![1.0; 100];
        with_nan[3] = f64::NAN;
        for bad in [vec![1.0; 99], vec![0.0; 100], with_nan] {
            let c = IVMTEConfig {
                prte_weights: Some(bad),
                ..Default::default()
            };
            expect_invalid(run(c), "prte_weights");
        }
        let c = IVMTEConfig {
            estimand: MTEEstimand::ATT,
            ..Default::default()
        };
        let r = run(c).unwrap();
        assert_eq!(
            r.target.as_ref().map(|t| (t.name.as_str(), t.estimate)),
            Some(("ATT", r.att.estimate))
        );
        assert!(format!("{r}").contains("Target estimand: ATT"));
    }

    #[test]
    fn test_mte_estimand_parses_aliases_and_rejects_unknown() {
        MTEEstimand::choices().validate().unwrap();
        for (s, v) in [
            ("ATT", MTEEstimand::ATT),
            ("late", MTEEstimand::LATE),
            ("mte-curve", MTEEstimand::MTECurve),
            ("PRTE", MTEEstimand::PRTE),
        ] {
            assert_eq!(s.parse::<MTEEstimand>().unwrap(), v, "{s}");
        }
        expect_invalid("tot".parse::<MTEEstimand>(), "estimand");
    }

    #[test]
    fn test_propensity_model_parses_aliases_and_rejects_unknown() {
        PropensityModel::choices().validate().unwrap();
        for (s, v) in [
            ("probit", PropensityModel::Probit),
            ("Logit", PropensityModel::Logit),
            ("logistic", PropensityModel::Logit),
            ("LPM", PropensityModel::Linear),
        ] {
            assert_eq!(s.parse::<PropensityModel>().unwrap(), v, "{s}");
        }
        expect_invalid("cloglog".parse::<PropensityModel>(), "propensity_model");
    }

    /// Case `mte_bootstrap`: rows 1-400 of mte_sim resampled with the committed 400 x 25 index
    /// matrix; every replicate refits the probit and the cubic. sd() and quantile(type = 7) as
    /// R. Fails before: bootstrap_se was never read (no bootstrap existed).
    #[test]
    fn test_validate_mte_bootstrap_indices() {
        let (case, idx) = crate::test_support::load_case("mte_bootstrap");
        let ds = mte_sim();
        let n = case.scalar("n") as usize;
        let reps = case.scalar("replications") as usize;
        let (y, d, z) = mte_data(&ds, |i, _| i < n);
        let cols: Vec<Vec<usize>> = (1..=reps)
            .map(|b| {
                nan_vec(&idx, &format!("b{b:02}"))
                    .iter()
                    .map(|&v| v as usize - 1)
                    .collect()
            })
            .collect();
        let cfg = case_config();
        let full = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, cfg.clone()).unwrap();
        let mut it = cols.iter();
        let boot = bootstrap_from_indices(y.view(), d.view(), z.view(), None, &cfg, reps, |buf| {
            buf.extend_from_slice(it.next().unwrap())
        })
        .unwrap();
        assert_eq!((boot.failed, boot.estimands.nrows()), (0, reps));
        let names = ["ate", "att", "atu", "late_support", "mte_mid"];
        let est = [
            full.ate.estimate,
            full.att.estimate,
            full.atu.estimate,
            full.late.estimate,
            full.mte_curve[500].mte,
        ];
        // grid point 500 is u = 0.5
        let columns = [
            boot.estimands.column(0),
            boot.estimands.column(1),
            boot.estimands.column(2),
            boot.estimands.column(3),
            boot.curve.column(500),
        ];
        let s: Vec<BootSummary> = columns
            .iter()
            .map(|c| boot_summary(c.view(), 0.95).unwrap())
            .collect();
        let mut check = Check::new(&case);
        check
            .named("estimate", &names, &est, Tol::coef(n))
            .named(
                "boot_se",
                &names,
                &s.iter().map(|b| b.sd).collect::<Vec<_>>(),
                Tol::se(n),
            )
            .named(
                "boot_ci_lower",
                &names,
                &s.iter().map(|b| b.ci_lower).collect::<Vec<_>>(),
                Tol::stat(n),
            )
            .named(
                "boot_ci_upper",
                &names,
                &s.iter().map(|b| b.ci_upper).collect::<Vec<_>>(),
                Tol::stat(n),
            )
            .named(
                "boot_mean",
                &names,
                &s.iter().map(|b| b.mean).collect::<Vec<_>>(),
                Tol::coef(n),
            );
        check.finish();
    }

    /// Public path: ChaCha8 resampling seeded by IVMTEConfig.seed (None ->
    /// DEFAULT_BOOTSTRAP_SEED). Same seed gives identical output, another seed different SEs;
    /// point estimates do not depend on the bootstrap; the bootstrap SE of the ATE is of the
    /// order of the delta SE. Fails before: bootstrap_se was ignored (ate.se equal to the
    /// analytic SE; curve SEs None).
    #[test]
    fn test_mte_bootstrap_seeded() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let base = IVMTEConfig {
            n_grid: 201,
            ..case_config()
        };
        let run = |seed: Option<u64>| {
            let c = IVMTEConfig {
                bootstrap_se: true,
                n_bootstrap: 50,
                seed,
                ..base.clone()
            };
            run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c).unwrap()
        };
        let delta = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, base.clone()).unwrap();
        let (a, b, c, none, dflt) = (
            run(Some(11)),
            run(Some(11)),
            run(Some(12)),
            run(None),
            run(Some(DEFAULT_BOOTSTRAP_SEED)),
        );
        assert_eq!(
            (a.ate.se.to_bits(), a.att.ci_lower.to_bits()),
            (b.ate.se.to_bits(), b.att.ci_lower.to_bits())
        );
        assert_ne!(a.ate.se, c.ate.se);
        assert_eq!(none.ate.se.to_bits(), dflt.ate.se.to_bits());
        assert_eq!(
            a.bootstrap
                .as_ref()
                .map(|i| (i.replications, i.failed, i.seed)),
            Some((50, 0, 11))
        );
        assert_eq!(a.ate.estimate.to_bits(), delta.ate.estimate.to_bits());
        assert_ne!(a.ate.se, delta.ate.se);
        let ratio = a.ate.se / delta.ate.se;
        assert!(ratio > 0.5 && ratio < 1.5, "bootstrap / delta SE = {ratio}");
        assert!(
            a.mte_curve.iter().all(
                |p| p.se.is_some_and(|s| s > 0.0) && p.ci_lower.unwrap() <= p.ci_upper.unwrap()
            )
        );
        assert!(
            a.se_method
                .starts_with("bootstrap (50 replications, seed 11")
        );
    }

    /// A zero (or non-finite) SE gives NaN z and p, not z = inf and p = 0.
    /// Fails before: `estimate / se` reported infinite significance.
    #[test]
    fn test_te_estimate_zero_se_is_not_significant() {
        for se in [0.0, f64::NAN, f64::INFINITY] {
            let te = te_estimate("ATE", 1.5, se, 1.96);
            assert!(
                te.t_stat.is_nan() && te.p_value.is_nan(),
                "se = {se}: {te:?}"
            );
            assert_eq!(te.significance, SignificanceLevel::NotSignificant);
        }
        let te = te_estimate("ATE", 1.5, 0.5, 1.96);
        assert_eq!(te.t_stat, 3.0);
    }

    /// The public path draws each replicate's resample from the ChaCha8 stream as it is
    /// fitted; the result equals the one from a fully materialised index matrix drawn from the
    /// same stream in the same order.
    #[test]
    fn test_mte_bootstrap_lazy_draws_match_materialised() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let (n, reps, seed) = (y.len(), 8, 11);
        let cfg = IVMTEConfig {
            bootstrap_se: true,
            n_bootstrap: reps,
            seed: Some(seed),
            n_grid: 21,
            ..case_config()
        };
        let public = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, cfg.clone()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let all: Vec<Vec<usize>> = (0..reps)
            .map(|_| (0..n).map(|_| rng.gen_range(0..n)).collect())
            .collect();
        let mut it = all.iter();
        let boot = bootstrap_from_indices(y.view(), d.view(), z.view(), None, &cfg, reps, |buf| {
            buf.extend_from_slice(it.next().unwrap())
        })
        .unwrap();
        let mut manual = run_ivmte_multi_z(
            &y.view(),
            &d.view(),
            &z.view(),
            None,
            IVMTEConfig {
                bootstrap_se: false,
                ..cfg.clone()
            },
        )
        .unwrap();
        apply_bootstrap(&mut manual, &boot, reps, seed).unwrap();
        for (a, b) in [
            (public.ate.se, manual.ate.se),
            (public.att.ci_lower, manual.att.ci_lower),
            (public.late.ci_upper, manual.late.ci_upper),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// Replicates draw their indices one at a time, interleaved with the fits: a budget
    /// cancelled while drawing the third resample stops the run after three draws, however
    /// many replications were asked for. Fails before: estimate_mte drew every replicate's
    /// n indices (n_bootstrap x n usizes) before fitting the first.
    #[test]
    fn test_mte_bootstrap_draws_one_replicate_at_a_time() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let n = y.len();
        // a small grid and a moderate replicate count: the estimand and curve stores are
        // still allocated up front (reps x 4 and reps x n_grid), only the indices are lazy
        let cfg = IVMTEConfig {
            n_grid: 11,
            ..case_config()
        };
        let token = CancelToken::new();
        let mut calls = 0usize;
        let res = with_budget(Budget::new(None, token.clone()), || {
            bootstrap_from_indices(y.view(), d.view(), z.view(), None, &cfg, 10_000, |buf| {
                calls += 1;
                assert!(
                    buf.is_empty() && buf.capacity() >= n,
                    "one reused buffer of length n"
                );
                buf.extend((0..n).map(|i| (i * 7 + calls) % n));
                if calls == 3 {
                    token.cancel();
                }
            })
        });
        expect_cancelled(res);
        assert_eq!(calls, 3);
    }

    /// A degenerate bootstrap (every replicate gives the same draw) has SD exactly 0 and
    /// reports a NaN z and p and no significance, as the delta path does for a zero SE.
    /// Fails before: the rounded mean left an SD of ~1e-17 (0.1 x 10 draws), and
    /// t = estimate / sd was huge or +-inf with p = 0, highly significant.
    #[test]
    fn test_mte_bootstrap_zero_sd_is_not_significant() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let cfg = IVMTEConfig {
            n_grid: 11,
            ..case_config()
        };
        let mut result = run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, cfg).unwrap();
        assert!(result.ate.estimate != 0.0);
        let reps = 10;
        let boot = MteBootstrap {
            estimands: Array2::from_shape_fn((reps, 4), |(_, j)| 0.1 * (j + 1) as f64),
            curve: Array2::from_elem((reps, result.mte_curve.len()), 0.25),
            failed: 0,
            first_error: None,
        };
        apply_bootstrap(&mut result, &boot, reps, 7).unwrap();
        for te in [&result.ate, &result.att, &result.atu, &result.late] {
            assert_eq!(te.se, 0.0, "{}", te.name);
            assert!(te.t_stat.is_nan(), "{}: t = {}", te.name, te.t_stat);
            assert!(te.p_value.is_nan(), "{}: p = {}", te.name, te.p_value);
            assert_eq!(
                te.significance,
                SignificanceLevel::NotSignificant,
                "{}",
                te.name
            );
        }
    }

    #[test]
    fn test_mte_bootstrap_rejects_one_replication() {
        let ds = mte_sim();
        let (y, d, z) = mte_data(&ds, |_, _| true);
        let c = IVMTEConfig {
            bootstrap_se: true,
            n_bootstrap: 1,
            ..Default::default()
        };
        expect_invalid(
            run_ivmte_multi_z(&y.view(), &d.view(), &z.view(), None, c),
            "n_bootstrap",
        );
    }
}
