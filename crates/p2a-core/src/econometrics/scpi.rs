//! Synthetic Control with Prediction Intervals (SCPI).
//!
//! Pure Rust implementation of the SCPI method developed by Cattaneo, Feng, and Titiunik (2021).
//! SCPI extends synthetic control methods with proper uncertainty quantification through
//! prediction intervals that account for both in-sample and out-of-sample uncertainty.
//!
//! # Mathematical Framework
//!
//! The synthetic control problem minimizes:
//! ```text
//! ||Y₁ - W'Y₀||² subject to constraints on W
//! ```
//!
//! Supported constraint types:
//! - **Simplex**: Σwⱼ = 1, wⱼ ≥ 0 (classic Abadie, Diamond, Hainmueller approach)
//! - **Lasso**: Add λ||w||₁ penalty
//! - **Ridge**: Add λ||w||₂² penalty
//! - **Lasso-Simplex**: Lasso penalty with simplex constraints
//!
//! The simplex weights are the exact solution of the constrained least-squares problem (the
//! simplex QP shared with the synthetic control, `synth::simplex_qp`); when several weight vectors
//! fit equally well the minimum-norm one is reported with a warning.
//!
//! Prediction intervals follow R's `scpi::scpi()` (scpi 3.0.1) at its defaults (see [`run_scpi`]):
//! the in-sample uncertainty of the weights is simulated from the conditional (HC1) distribution of
//! the constrained estimator, each draw solving a quadratically constrained program per post period
//! over the localised constraint set, and the out-of-sample error is bounded by sub-Gaussian,
//! location-scale and quantile-regression models of the pre-treatment residuals. The widths
//! follow each post period's donor path.
//!
//! # References
//!
//! - Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic
//!   Control Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880.
//!   DOI: 10.1080/01621459.2021.1979561
//!
//! - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for
//!   Comparative Case Studies." *Journal of the American Statistical Association*, 105(490), 493-505.
//!
//! - Cattaneo, M. D., Feng, Y., Palomba, F., & Titiunik, R. (2025). "scpi: Uncertainty
//!   Quantification for Synthetic Control Methods." *Journal of Statistical Software*; R package
//!   `scpi` (<https://nppackages.github.io/scpi/>), the reference of the simplex intervals.

use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky_inverse, eig_symmetric, xtx, xty};
use crate::linalg::solve::least_squares;
use crate::regression::quantreg::{QuantRegConfig, QuantRegMethod, RqFit, rq_fit};
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::validate_level;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Constraint type for SCPI weight estimation.
///
/// Different constraint types offer trade-offs between interpretability,
/// sparsity, and bias-variance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SCPIConstraint {
    /// Simplex constraint: weights sum to 1 and are non-negative.
    /// This is the classic synthetic control constraint (Abadie et al., 2010).
    /// Most interpretable but may have high variance with many donors.
    #[default]
    Simplex,

    /// Lasso (L1) penalty: λ||w||₁.
    /// Encourages sparse solutions with few non-zero weights.
    /// Good when only a few donors are relevant.
    Lasso {
        /// Regularization parameter (larger = more sparsity)
        lambda: f64,
    },

    /// Ridge (L2) penalty: λ||w||₂².
    /// Shrinks all weights toward zero but doesn't produce sparsity.
    /// Good for reducing variance with many correlated donors.
    Ridge {
        /// Regularization parameter (larger = more shrinkage)
        lambda: f64,
    },

    /// Lasso penalty combined with simplex constraints. On the simplex the L1
    /// norm of the weights is 1, so the penalty is a constant and the weights are
    /// the simplex weights whatever `lambda` (a warning says so).
    LassoSimplex {
        /// L1 regularization parameter (no effect on the simplex)
        lambda: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintName {
    Simplex,
    Lasso,
    Ridge,
    LassoSimplex,
}

crate::option_enum!(
    ConstraintName,
    crate::options::Choices {
        option: "SCPI constraint",
        values: &[
            (&["simplex"], ConstraintName::Simplex),
            (&["lasso", "l1"], ConstraintName::Lasso),
            (&["ridge", "l2"], ConstraintName::Ridge),
            (
                &["lasso_simplex", "lassosimplex"],
                ConstraintName::LassoSimplex
            ),
        ],
    },
    no_display
);

impl SCPIConstraint {
    /// "simplex" (default when `None`), "lasso", "ridge", "lasso_simplex"; `lambda` defaults to
    /// 0.1 for the penalised ones (the MCP default) and is validated by [`run_scpi`]. Unknown
    /// names are `InvalidSpecification` listing the accepted names.
    pub fn from_options(name: Option<&str>, lambda: Option<f64>) -> EconResult<SCPIConstraint> {
        let lambda = lambda.unwrap_or(0.1);
        Ok(
            match crate::options::parse_or(name, ConstraintName::Simplex)? {
                ConstraintName::Simplex => SCPIConstraint::Simplex,
                ConstraintName::Lasso => SCPIConstraint::Lasso { lambda },
                ConstraintName::Ridge => SCPIConstraint::Ridge { lambda },
                ConstraintName::LassoSimplex => SCPIConstraint::LassoSimplex { lambda },
            },
        )
    }

    /// Penalties: ridge `lambda > 0`, lasso and lasso-simplex `lambda >= 0`, all finite.
    fn validate(&self) -> EconResult<()> {
        let (name, lambda, positive) = match self {
            SCPIConstraint::Simplex => return Ok(()),
            SCPIConstraint::Lasso { lambda } => ("lasso", *lambda, false),
            SCPIConstraint::LassoSimplex { lambda } => ("lasso-simplex", *lambda, false),
            SCPIConstraint::Ridge { lambda } => ("ridge", *lambda, true),
        };
        let ok = lambda.is_finite()
            && if positive {
                lambda > 0.0
            } else {
                lambda >= 0.0
            };
        if ok {
            Ok(())
        } else {
            Err(EconError::InvalidSpecification {
                message: format!(
                    "the {name} penalty lambda must be finite and {} (got {lambda})",
                    if positive { "positive" } else { "non-negative" }
                ),
            })
        }
    }
}

/// Seed of the in-sample simulations when [`SCPIConfig::seed`] is `None`.
pub const SCPI_DEFAULT_SEED: u64 = 42;

/// Configuration for SCPI estimation. The defaults are `scpi::scpi()`'s: 200 simulations,
/// `u.missp = TRUE`, `e.method = "all"`, `u.alpha = e.alpha = 0.05`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCPIConfig {
    /// Constraint type for weight estimation
    pub constraint: SCPIConstraint,

    /// Former single level, kept as an alias: `Some(a)` sets `u_alpha` and `e_alpha` to `a / 2`,
    /// so the interval has coverage at least `1 - a` (scpi's union bound). `None` (default)
    /// uses `u_alpha` and `e_alpha`.
    #[serde(default)]
    pub alpha: Option<f64>,

    /// Level of the in-sample (weight-estimation) bounds, scpi's `u.alpha`: the bounds are the
    /// `u_alpha / 2` and `1 - u_alpha / 2` quantiles over the simulations (default 0.05)
    #[serde(default = "default_level")]
    pub u_alpha: f64,

    /// Level of the out-of-sample bounds, scpi's `e.alpha`: each side at `e_alpha / 2`
    /// (default 0.05)
    #[serde(default = "default_level")]
    pub e_alpha: f64,

    /// Number of simulations of the in-sample uncertainty (scpi's `sims`, default 200, at least 10)
    #[serde(default = "default_sims")]
    pub sims: usize,

    /// Seed of the simulations (`None`: [`SCPI_DEFAULT_SEED`])
    #[serde(default)]
    pub seed: Option<u64>,

    /// Out-of-sample bounds: sub-Gaussian, location-scale, quantile regression or all three
    /// (scpi's `e.method`, default all). A value other than `All` wins over
    /// [`SCPIConfig::variance_method`] (with a warning); `All`, the default, cannot be told
    /// apart from no choice, so `variance_method` is then honoured
    #[serde(default)]
    pub e_method: EMethod,

    /// Model the conditional mean of the pseudo-residuals (scpi's `u.missp`, default true)
    #[serde(default = "default_true")]
    pub u_missp: bool,

    /// Deprecated alias of `e_method` (one release): `gaussian` -> gaussian, `subgaussian` ->
    /// all; `loo_cv` and `kfold_cv` -> all with a warning. `None` (default) uses `e_method`.
    /// Honoured only while `e_method` is `All` (its default); another `e_method` wins.
    #[serde(default)]
    pub variance_method: Option<VarianceMethod>,

    /// Unused since the intervals follow scpi (kept so that existing configurations parse)
    pub cv_folds: usize,

    /// Maximum iterations for optimization (lasso)
    pub max_iter: usize,

    /// Convergence tolerance (lasso)
    pub tolerance: f64,

    /// Minimum weight to report (for sparsity in output)
    pub weight_threshold: f64,
}

fn default_level() -> f64 {
    0.05
}

fn default_sims() -> usize {
    200
}

fn default_true() -> bool {
    true
}

impl Default for SCPIConfig {
    fn default() -> Self {
        SCPIConfig {
            constraint: SCPIConstraint::Simplex,
            alpha: None,
            u_alpha: 0.05,
            e_alpha: 0.05,
            sims: 200,
            seed: None,
            e_method: EMethod::All,
            u_missp: true,
            variance_method: None,
            cv_folds: 5,
            max_iter: 10000,
            tolerance: 1e-8,
            weight_threshold: 0.001,
        }
    }
}

/// Out-of-sample bounds of scpi (`e.method`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum EMethod {
    /// Sub-Gaussian bounds: conditional mean plus `sqrt(-2 log(alpha)) sigma`
    Gaussian,
    /// Location-scale model: conditional mean plus sigma times quantiles of the standardised
    /// residuals
    Ls,
    /// Quantile regression of the residuals
    Qreg,
    /// All three (scpi's default); the headline interval is the sub-Gaussian one
    #[default]
    All,
}

crate::option_enum!(
    EMethod,
    crate::options::Choices {
        option: "SCPI e_method",
        values: &[
            (&["all"], EMethod::All),
            (
                &["gaussian", "subgaussian", "sub_gaussian"],
                EMethod::Gaussian
            ),
            (&["ls", "location_scale"], EMethod::Ls),
            (&["qreg", "quantile", "quantile_regression"], EMethod::Qreg),
        ],
    }
);

/// Former out-of-sample variance method, a deprecated alias of [`EMethod`] for one release
/// (see [`SCPIConfig::variance_method`]).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum VarianceMethod {
    /// Maps to `EMethod::Gaussian`
    Gaussian,

    /// The former default; maps to `EMethod::All`
    #[default]
    Subgaussian,

    /// Maps to `EMethod::All` with a warning
    LooCv,

    /// Maps to `EMethod::All` with a warning
    KFoldCv,
}

crate::option_enum!(
    VarianceMethod,
    crate::options::Choices {
        option: "SCPI variance method",
        values: &[
            (
                &["subgaussian", "sub_gaussian"],
                VarianceMethod::Subgaussian
            ),
            (&["gaussian", "normal"], VarianceMethod::Gaussian),
            (&["loo_cv", "loo", "leave_one_out"], VarianceMethod::LooCv),
            (
                &["kfold_cv", "kfold", "k_fold", "k_fold_cv"],
                VarianceMethod::KFoldCv,
            ),
        ],
    }
);

impl VarianceMethod {
    /// The `e_method` this deprecated value stands for, and the warning it carries.
    fn e_method(self) -> (EMethod, Option<String>) {
        match self {
            VarianceMethod::Gaussian => (EMethod::Gaussian, None),
            VarianceMethod::Subgaussian => (EMethod::All, None),
            VarianceMethod::LooCv | VarianceMethod::KFoldCv => (
                EMethod::All,
                Some(format!(
                    "variance_method '{self}' is no longer computed: the intervals follow scpi's \
                     construction, and the out-of-sample bounds are chosen with e_method (all three \
                     are reported)"
                )),
            ),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Prediction interval of the treatment effect for one post-treatment period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    /// Time period index
    pub period: usize,

    /// Point prediction (synthetic control value)
    pub prediction: f64,

    /// Lower bound of the effect's prediction interval
    pub lower: f64,

    /// Upper bound of the effect's prediction interval
    pub upper: f64,
}

/// The intervals of one out-of-sample method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCPIMethodIntervals {
    /// Out-of-sample method (never `All`)
    pub method: EMethod,
    /// Out-of-sample bounds of the error e_t, one per post period (scpi's `bounds`)
    pub e_lower: Vec<f64>,
    /// See `e_lower`
    pub e_upper: Vec<f64>,
    /// Interval of the synthetic (counterfactual) outcome: prediction + in-sample bound +
    /// out-of-sample bound (scpi's `CI.all.<method>`)
    pub synthetic_lower: Vec<f64>,
    /// See `synthetic_lower`
    pub synthetic_upper: Vec<f64>,
    /// Interval of the effect: actual minus the synthetic interval
    pub lower: Vec<f64>,
    /// See `lower`
    pub upper: Vec<f64>,
}

/// Complete results from SCPI estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCPIResult {
    // ─── Weights ───────────────────────────────────────────────────────────────
    /// Estimated donor weights (J × 1)
    #[serde(skip)]
    pub weights: Array1<f64>,

    /// Non-zero weights with donor indices (donor_idx, weight)
    pub nonzero_weights: Vec<(usize, f64)>,

    /// Number of donors with non-negligible weight
    pub n_effective_donors: usize,

    // ─── Time Series ───────────────────────────────────────────────────────────
    /// Actual outcomes for treated unit (T × 1)
    #[serde(skip)]
    pub treated_actual: Array1<f64>,

    /// Synthetic control predictions (T × 1)
    #[serde(skip)]
    pub synthetic: Array1<f64>,

    /// Treatment effect estimates: actual - synthetic (T_post × 1)
    #[serde(skip)]
    pub effect: Array1<f64>,

    // ─── Intervals ─────────────────────────────────────────────────────────────
    /// Lower bounds of the headline effect intervals (sub-Gaussian unless `e_method` names
    /// another method)
    #[serde(skip)]
    pub ci_lower: Array1<f64>,

    /// Upper bounds of the headline effect intervals
    #[serde(skip)]
    pub ci_upper: Array1<f64>,

    /// Headline prediction intervals, one per post-treatment period
    pub prediction_intervals: Vec<PredictionInterval>,

    /// Out-of-sample method of the headline intervals
    pub headline_method: EMethod,

    /// In-sample bounds of p_t'(w_hat - w), one per post period (scpi's `bounds$insample`)
    pub insample_lower: Vec<f64>,

    /// See `insample_lower`
    pub insample_upper: Vec<f64>,

    /// Intervals of every out-of-sample method computed
    pub method_intervals: Vec<SCPIMethodIntervals>,

    /// Regularisation threshold rho (scpi's type-2 rule, at most 0.2)
    pub rho: f64,

    /// Simulations whose optimisation failed at some horizon (excluded there)
    pub n_failed_sims: usize,

    /// Number of simulations
    pub sims: usize,

    /// Seed of the simulations
    pub seed: u64,

    /// In-sample level
    pub u_alpha: f64,

    /// Out-of-sample level
    pub e_alpha: f64,

    // ─── Fit Statistics ────────────────────────────────────────────────────────
    /// Pre-treatment RMSPE (Root Mean Squared Prediction Error)
    pub pre_treatment_rmspe: f64,

    /// Pre-treatment MSPE
    pub pre_treatment_mspe: f64,

    /// R-squared in pre-treatment period
    pub pre_treatment_r2: f64,

    // ─── Configuration ─────────────────────────────────────────────────────────
    /// Number of pre-treatment periods
    pub n_pre_periods: usize,

    /// Number of post-treatment periods
    pub n_post_periods: usize,

    /// Number of donor units
    pub n_donors: usize,

    /// Treatment period index (first post-treatment period)
    pub treatment_period: usize,

    /// Constraint type used
    pub constraint_type: String,

    /// Warnings (a non-unique simplex solution, failed simulations, a deprecated option)
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for SCPIResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        None
    }
}

fn method_label(m: EMethod) -> &'static str {
    match m {
        EMethod::Gaussian => "sub-Gaussian",
        EMethod::Ls => "location-scale",
        EMethod::Qreg => "quantile regression",
        EMethod::All => "all",
    }
}

impl fmt::Display for SCPIResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Synthetic Control with Prediction Intervals (SCPI)")?;
        writeln!(f, "===================================================")?;
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Constraint type: {}", self.constraint_type)?;
        writeln!(f, "  Pre-treatment periods: {}", self.n_pre_periods)?;
        writeln!(f, "  Post-treatment periods: {}", self.n_post_periods)?;
        writeln!(f, "  Donor pool size: {}", self.n_donors)?;
        writeln!(f, "  Effective donors: {}", self.n_effective_donors)?;
        writeln!(
            f,
            "  Levels: in-sample u_alpha = {}, out-of-sample e_alpha = {} (coverage at least {}%)",
            self.u_alpha,
            self.e_alpha,
            fmt_pct(1.0 - self.u_alpha - self.e_alpha)
        )?;
        writeln!(
            f,
            "  Simulations: {} (seed {}), rho = {:.4}",
            self.sims, self.seed, self.rho
        )?;
        writeln!(f)?;

        writeln!(f, "Pre-Treatment Fit:")?;
        writeln!(f, "  RMSPE: {:.6}", self.pre_treatment_rmspe)?;
        writeln!(f, "  R-squared: {:.4}", self.pre_treatment_r2)?;
        writeln!(f)?;

        writeln!(f, "Donor Weights (non-zero):")?;
        for (idx, weight) in &self.nonzero_weights {
            writeln!(f, "  Donor {}: {:.4}", idx, weight)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Treatment Effects with Prediction Intervals (scpi; {} out-of-sample bounds):",
            method_label(self.headline_method)
        )?;
        writeln!(
            f,
            "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Period", "Effect", "Lower", "Upper", "In-sample lo", "In-sample hi"
        )?;
        writeln!(f, "{}", "-".repeat(74))?;
        for (i, pi) in self.prediction_intervals.iter().enumerate() {
            let star = if pi.lower > 0.0 || pi.upper < 0.0 {
                "*"
            } else {
                ""
            };
            writeln!(
                f,
                "{:>8} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}{}",
                pi.period,
                self.effect[i],
                pi.lower,
                pi.upper,
                self.insample_lower[i],
                self.insample_upper[i],
                star
            )?;
        }
        writeln!(
            f,
            "* the interval excludes 0. In-sample bounds: of p_t'(w_hat - w), the weight-estimation \
             error."
        )?;
        let others: Vec<&SCPIMethodIntervals> = self
            .method_intervals
            .iter()
            .filter(|m| m.method != self.headline_method)
            .collect();
        if !others.is_empty() {
            writeln!(f)?;
            writeln!(f, "Effect intervals by out-of-sample method:")?;
            write!(f, "{:>8}", "Period")?;
            for m in &others {
                write!(f, " {:>27}", method_label(m.method))?;
            }
            writeln!(f)?;
            for (i, pi) in self.prediction_intervals.iter().enumerate() {
                write!(f, "{:>8}", pi.period)?;
                for m in &others {
                    write!(f, "  [{:>11.4}, {:>11.4}]", m.lower[i], m.upper[i])?;
                }
                writeln!(f)?;
            }
        }

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }

        Ok(())
    }
}

fn fmt_pct(x: f64) -> String {
    let p = 100.0 * x;
    if (p - p.round()).abs() < 1e-9 {
        format!("{}", p.round())
    } else {
        format!("{p:.1}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Synthetic Control with Prediction Intervals (SCPI).
///
/// Fits the donor weights on the pre-treatment outcomes and computes the prediction intervals of
/// Cattaneo, Feng and Titiunik (2021) as R's `scpi::scpi()` does at its defaults (scpi 3.0.1, on
/// the outcome alone, no constant, no covariates):
///
/// * **In-sample** (weight-estimation) uncertainty: with `Q = B'B / T0`, `Sigma = B' Omega B / T0^2`
///   (HC1: `Omega = diag((u - E[u|H])^2) T0 / (T0 - df)`, `E[u|H]` the fit of the residuals on the
///   active donors and a constant when `T0 - 10` exceeds its columns, otherwise their mean) and
///   `G = Sigma^(1/2) z`, `z ~ N(0, I)`, each simulation solves, for every post period,
///   `min` and `max` of `p_t' delta` subject to `delta' Q delta - 2 G' delta <= 0` and the
///   localised constraint set (simplex: `sum(w + delta) = 1`, `w_j + delta_j >= w_j` for weights
///   below rho, `>= 0` otherwise). The bounds are the `u_alpha / 2` and `1 - u_alpha / 2`
///   quantiles over the simulations of `-max` and `-min`.
/// * **Out-of-sample** uncertainty of `e_t`, at `e_alpha / 2` on each side, from the
///   pre-treatment residuals regressed on the active donors and a constant (only the constant
///   when `T0 - 10` does not exceed the columns): sub-Gaussian, location-scale and quantile
///   regression bounds.
///
/// The interval of the synthetic outcome is `prediction + insample + e`, and the effect interval
/// is the actual outcome minus it. By the union bound its coverage is at least
/// `1 - u_alpha - e_alpha` (90% at the defaults; `alpha = Some(a)` sets both levels to `a / 2`).
///
/// Lasso and ridge (penalised here, constrained in scpi) use the norm ball of the fitted weights
/// widened by rho as the localised set; they are not R-pinned.
///
/// # References
///
/// - Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). JASA 116(536), 1865-1880.
/// - Cattaneo, M. D., Feng, Y., Palomba, F., & Titiunik, R. (2025). scpi: Uncertainty
///   Quantification for Synthetic Control Methods. *Journal of Statistical Software*.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::scpi::{run_scpi, SCPIConfig};
/// use ndarray::{array, Array2};
///
/// let treated = array![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0];
/// let donors = Array2::from_shape_fn((10, 5), |(t, j)| 10.0 + t as f64 + 0.5 * j as f64);
/// let result = run_scpi(&treated.view(), &donors.view(), 7, SCPIConfig::default())?;
/// println!("{result}");
/// ```
pub fn run_scpi(
    treated: &ArrayView1<f64>,
    donors: &ArrayView2<f64>,
    treatment_period: usize,
    config: SCPIConfig,
) -> EconResult<SCPIResult> {
    let seed = config.seed.unwrap_or(SCPI_DEFAULT_SEED);
    run_scpi_with(
        treated,
        donors,
        treatment_period,
        config,
        Draws::seeded(seed),
        seed,
    )
}

/// [`run_scpi`] with the standard normal draws of the simulations given (tests compare against
/// R's own draws).
fn run_scpi_with(
    treated: &ArrayView1<f64>,
    donors: &ArrayView2<f64>,
    treatment_period: usize,
    config: SCPIConfig,
    draws: Draws<'_>,
    seed: u64,
) -> EconResult<SCPIResult> {
    // Validate inputs
    let t_total = treated.len();
    let (t_donors, n_donors) = donors.dim();

    if t_donors != t_total {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treated series length ({}) must match donor matrix rows ({})",
                t_total, t_donors
            ),
        });
    }

    if treatment_period == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Treatment period must be > 0 (need at least one pre-treatment period)"
                .to_string(),
        });
    }

    if treatment_period >= t_total {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment period ({}) must be < total periods ({})",
                treatment_period, t_total
            ),
        });
    }

    let n_pre = treatment_period;
    let n_post = t_total - treatment_period;

    if n_donors == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "At least one donor unit is required".to_string(),
        });
    }

    if n_pre < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_pre,
            context: "At least 2 pre-treatment periods are required".to_string(),
        });
    }

    let tol = config.tolerance;
    if !(tol.is_finite() && tol > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be a finite number greater than 0 (got {tol})"),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1 (got 0)".to_string(),
        });
    }
    let (u_alpha, e_alpha) = match config.alpha {
        Some(a) => {
            validate_level(a, "alpha")?;
            (a / 2.0, a / 2.0)
        }
        None => {
            validate_level(config.u_alpha, "u_alpha")?;
            validate_level(config.e_alpha, "e_alpha")?;
            (config.u_alpha, config.e_alpha)
        }
    };
    if config.sims < 10 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "sims must be at least 10 (scpi's minimum), got {}",
                config.sims
            ),
        });
    }
    config.constraint.validate()?;
    if let Some((t, v)) = treated.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("treated outcome at period {t} is {v}; SCPI needs finite outcomes"),
        });
    }
    if let Some(((t, j), v)) = donors.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "donors outcome at period {t}, donor {j} is {v}; SCPI needs finite outcomes"
            ),
        });
    }

    let mut warnings = Vec::new();
    // an explicit e_method wins over the deprecated alias
    let e_method = match config.variance_method {
        Some(vm) if config.e_method != EMethod::All => {
            warnings.push(format!(
                "SCPI: variance_method '{vm}' is ignored: e_method '{}' was given (variance_method \
                 is a deprecated alias of e_method)",
                config.e_method
            ));
            config.e_method
        }
        Some(vm) => {
            let (m, w) = vm.e_method();
            warnings.extend(w);
            m
        }
        None => config.e_method,
    };

    // Split data into pre- and post-treatment
    let y_pre = treated.slice(s![..treatment_period]).to_owned();
    let y_post = treated.slice(s![treatment_period..]).to_owned();
    let x_pre = donors.slice(s![..treatment_period, ..]);
    let x_post = donors.slice(s![treatment_period.., ..]);

    // Estimate weights
    let weights = estimate_weights(
        &y_pre.view(),
        &x_pre,
        &config.constraint,
        config.max_iter,
        config.tolerance,
        &mut warnings,
    )?;

    let synthetic_pre = x_pre.dot(&weights);
    let synthetic_post = x_post.dot(&weights);
    let mut synthetic = Array1::zeros(t_total);
    synthetic
        .slice_mut(s![..treatment_period])
        .assign(&synthetic_pre);
    synthetic
        .slice_mut(s![treatment_period..])
        .assign(&synthetic_post);
    let effect = &y_post - &synthetic_post;
    let residuals_pre = &y_pre - &synthetic_pre;

    let inf = scpi_inference(
        &InferenceInput {
            b: x_pre,
            p: x_post,
            res: residuals_pre.view(),
            w: weights.view(),
            constraint: &config.constraint,
        },
        &InferenceLevels {
            u_alpha,
            e_alpha,
            sims: config.sims,
            u_missp: config.u_missp,
            e_method,
        },
        draws,
    )?;
    warnings.extend(inf.warnings.iter().cloned());

    let method_intervals: Vec<SCPIMethodIntervals> = inf
        .e_bounds
        .iter()
        .map(|(method, e_lo, e_hi)| {
            let syn_lo: Vec<f64> = (0..n_post)
                .map(|t| synthetic_post[t] + inf.insample_lower[t] + e_lo[t])
                .collect();
            let syn_hi: Vec<f64> = (0..n_post)
                .map(|t| synthetic_post[t] + inf.insample_upper[t] + e_hi[t])
                .collect();
            SCPIMethodIntervals {
                method: *method,
                lower: (0..n_post).map(|t| y_post[t] - syn_hi[t]).collect(),
                upper: (0..n_post).map(|t| y_post[t] - syn_lo[t]).collect(),
                e_lower: e_lo.clone(),
                e_upper: e_hi.clone(),
                synthetic_lower: syn_lo,
                synthetic_upper: syn_hi,
            }
        })
        .collect();
    let headline_method = match e_method {
        EMethod::All => EMethod::Gaussian,
        m => m,
    };
    let headline = method_intervals
        .iter()
        .find(|m| m.method == headline_method)
        .ok_or_else(|| EconError::Computation("SCPI: no out-of-sample bounds".to_string()))?;
    let ci_lower = Array1::from(headline.lower.clone());
    let ci_upper = Array1::from(headline.upper.clone());

    let prediction_intervals: Vec<PredictionInterval> = (0..n_post)
        .map(|i| PredictionInterval {
            period: treatment_period + i,
            prediction: synthetic_post[i],
            lower: ci_lower[i],
            upper: ci_upper[i],
        })
        .collect();

    // Pre-treatment fit statistics
    let rss: f64 = residuals_pre.iter().map(|r| r * r).sum();
    let pre_mspe = rss / n_pre as f64;
    let pre_rmspe = pre_mspe.sqrt();
    let y_pre_mean = y_pre.mean().unwrap_or(0.0);
    let tss: f64 = y_pre.iter().map(|y| (y - y_pre_mean).powi(2)).sum();
    let pre_r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    let nonzero_weights: Vec<(usize, f64)> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| w.abs() >= config.weight_threshold)
        .map(|(i, w)| (i, *w))
        .collect();
    let n_effective_donors = nonzero_weights.len();

    let constraint_type = match &config.constraint {
        SCPIConstraint::Simplex => "Simplex".to_string(),
        SCPIConstraint::Lasso { lambda } => format!("Lasso (lambda={})", lambda),
        SCPIConstraint::Ridge { lambda } => format!("Ridge (lambda={})", lambda),
        SCPIConstraint::LassoSimplex { lambda } => format!("Lasso-Simplex (lambda={})", lambda),
    };

    Ok(SCPIResult {
        weights,
        nonzero_weights,
        n_effective_donors,
        treated_actual: treated.to_owned(),
        synthetic,
        effect,
        ci_lower,
        ci_upper,
        prediction_intervals,
        headline_method,
        insample_lower: inf.insample_lower,
        insample_upper: inf.insample_upper,
        method_intervals,
        rho: inf.rho,
        n_failed_sims: inf.n_failed,
        sims: config.sims,
        seed,
        u_alpha,
        e_alpha,
        pre_treatment_rmspe: pre_rmspe,
        pre_treatment_mspe: pre_mspe,
        pre_treatment_r2: pre_r2,
        n_pre_periods: n_pre,
        n_post_periods: n_post,
        n_donors,
        treatment_period,
        constraint_type,
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate synthetic control weights under various constraints.
///
/// # Arguments
/// * `y` - Pre-treatment outcomes for treated unit (T0 × 1)
/// * `x` - Pre-treatment outcomes for donors (T0 × J)
/// * `constraint` - Constraint type
/// * `max_iter` - Maximum iterations
/// * `tolerance` - Convergence tolerance
fn estimate_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    constraint: &SCPIConstraint,
    max_iter: usize,
    tolerance: f64,
    warnings: &mut Vec<String>,
) -> EconResult<Array1<f64>> {
    match constraint {
        SCPIConstraint::Simplex | SCPIConstraint::LassoSimplex { .. } => {
            if matches!(constraint, SCPIConstraint::LassoSimplex { .. }) {
                warnings.push(
                    "lasso_simplex: on the simplex the L1 penalty is constant, so lambda has no \
                     effect; the weights are the simplex weights"
                        .to_string(),
                );
            }
            let qp = solve_simplex_weights(y, x)?;
            if !qp.unique {
                warnings.push(format!(
                    "the simplex weights are not unique ({} donors fit the pre-treatment \
                     outcomes equally well); the minimum-norm weights are reported",
                    qp.free.len()
                ));
            }
            Ok(qp.weights)
        }

        SCPIConstraint::Lasso { lambda } => solve_lasso_weights(y, x, *lambda, max_iter, tolerance),

        SCPIConstraint::Ridge { lambda } => solve_ridge_weights(y, x, *lambda),
    }
}

/// Simplex weights: `min ||y - X w||^2` subject to `w >= 0`, `sum(w) = 1`, solved exactly by
/// the simplex QP shared with the synthetic control.
fn solve_simplex_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
) -> EconResult<crate::econometrics::synth::SimplexQp> {
    crate::econometrics::synth::simplex_qp(x.view(), y.view())
}

/// Solve for weights with Lasso (L1) penalty using coordinate descent.
///
/// Minimizes ||y - Xw||² + λ||w||₁
fn solve_lasso_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    lambda: f64,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array1<f64>> {
    let (t, n) = x.dim();

    // Coordinate descent (LASSO via soft-thresholding)
    let mut w = Array1::zeros(n);

    // Precompute X'X diagonal and X'y
    let x_sq_sum: Vec<f64> = (0..n)
        .map(|j| x.column(j).iter().map(|v| v * v).sum())
        .collect();

    let _xy: Vec<f64> = (0..n)
        .map(|j| x.column(j).iter().zip(y.iter()).map(|(&x, &y)| x * y).sum())
        .collect();

    // Each sweep recomputes the partial residuals from scratch, O(t n^2); the
    // budget check per sweep lets a deadline stop a long run.
    for _ in 0..max_iter {
        crate::resource::checkpoint()?;

        let w_old = w.clone();

        for j in 0..n {
            // Compute partial residual
            let mut r_partial = 0.0;
            for i in 0..t {
                let mut pred = 0.0;
                for k in 0..n {
                    if k != j {
                        pred += x[[i, k]] * w[k];
                    }
                }
                r_partial += x[[i, j]] * (y[i] - pred);
            }

            // Soft-thresholding
            let denom = x_sq_sum[j];
            if denom > 1e-10 {
                w[j] = soft_threshold(r_partial / denom, lambda / denom);
            }
        }

        // Check convergence
        let change: f64 = w
            .iter()
            .zip(w_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        if change < tolerance {
            break;
        }
    }

    Ok(w)
}

/// Soft-thresholding operator for Lasso.
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Solve for weights with Ridge (L2) penalty.
///
/// Minimizes ||y - Xw||² + λ||w||₂²
///
/// Closed-form solution: w = (X'X + λI)⁻¹ X'y
fn solve_ridge_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    let n = x.ncols();

    // X'X + λI
    let mut xtx_reg = xtx(x);
    for i in 0..n {
        xtx_reg[[i, i]] += lambda;
    }

    // (X'X + λI)⁻¹
    let xtx_inv = cholesky_inverse(&xtx_reg.view()).map_err(|e| {
        EconError::Computation(format!(
            "Ridge weight estimation failed: Matrix inversion error: {:?}",
            e
        ))
    })?;

    // X'y
    let xy = xty(x, &y.to_owned());

    // w = (X'X + λI)⁻¹ X'y
    let w = xtx_inv.dot(&xy);

    Ok(w)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Uncertainty quantification (scpi 3.0.1's scpi())
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard normal draws of the simulations: from a generator seeded by `(seed, simulation)`, or a
/// given `J x sims` matrix (column `s` is simulation `s`).
enum Draws<'a> {
    Seeded(u64),
    #[cfg_attr(not(test), allow(dead_code))]
    Given(ArrayView2<'a, f64>),
}

impl Draws<'_> {
    fn seeded(seed: u64) -> Self {
        Draws::Seeded(seed)
    }

    fn next(&mut self, sim: usize, j: usize) -> EconResult<Array1<f64>> {
        match self {
            Draws::Seeded(seed) => {
                let mut rng = StdRng::seed_from_u64(
                    *seed ^ (sim as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                Ok(Array1::from_iter(
                    (0..j).map(|_| StandardNormal.sample(&mut rng)),
                ))
            }
            Draws::Given(m) => {
                if m.nrows() != j || sim >= m.ncols() {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "SCPI draws: {} x {} given, simulation {sim} of {j} donors asked",
                            m.nrows(),
                            m.ncols()
                        ),
                    });
                }
                Ok(m.column(sim).to_owned())
            }
        }
    }
}

struct InferenceInput<'a> {
    /// Pre-treatment donor outcomes (T0 x J), scpi's B
    b: ArrayView2<'a, f64>,
    /// Post-treatment donor outcomes (T1 x J), scpi's P
    p: ArrayView2<'a, f64>,
    /// Pre-treatment residuals A - B w
    res: ArrayView1<'a, f64>,
    /// Fitted weights
    w: ArrayView1<'a, f64>,
    constraint: &'a SCPIConstraint,
}

struct InferenceLevels {
    u_alpha: f64,
    e_alpha: f64,
    sims: usize,
    u_missp: bool,
    e_method: EMethod,
}

struct Inference {
    rho: f64,
    insample_lower: Vec<f64>,
    insample_upper: Vec<f64>,
    /// (method, lower, upper) of the out-of-sample error, one entry per method computed
    e_bounds: Vec<(EMethod, Vec<f64>, Vec<f64>)>,
    n_failed: usize,
    warnings: Vec<String>,
    #[cfg_attr(not(test), allow(dead_code))]
    sigma: Array2<f64>,
    #[cfg_attr(not(test), allow(dead_code))]
    u_mean: Array1<f64>,
}

/// scpi's `rho.max`.
const RHO_MAX: f64 = 0.2;

/// scpi's `regularize.w`: type 1 `sigma_u / min_j sd(B_j)`, type 2
/// `max_j sd(B_j) sigma_u / min_j var(B_j)`, times `sqrt(log(J) d0 log(T0) / T0)`, capped at
/// `rho_max`.
fn regularize_w(
    type1: bool,
    rho_max: f64,
    res: ArrayView1<f64>,
    b: ArrayView2<f64>,
    d0: usize,
) -> EconResult<f64> {
    let (t0, j) = b.dim();
    let mean = res.sum() / t0 as f64;
    let sigma_u = (res.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / t0 as f64).sqrt();
    let mut min_var = f64::INFINITY;
    let mut max_var = 0.0f64;
    for (jj, col) in b.axis_iter(Axis(1)).enumerate() {
        let m = col.sum() / t0 as f64;
        let v = col.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (t0 - 1) as f64;
        if v == 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "donor column {jj} (0-based, in the order of the donor matrix) has no \
                     variation in the pre-treatment period (scpi needs it to choose the \
                     regularisation threshold rho)"
                ),
            });
        }
        min_var = min_var.min(v);
        max_var = max_var.max(v);
    }
    let cc = if type1 {
        sigma_u / min_var.sqrt()
    } else {
        max_var.sqrt() * sigma_u / min_var
    };
    let t0f = t0 as f64;
    let rho = cc * ((j as f64).ln() * d0 as f64 * t0f.ln()).sqrt() / t0f.sqrt();
    Ok(rho.min(rho_max))
}

/// scpi's default rho (`"type-2"`) with `regularize.check.lb`, and the rule's own rho when that
/// check raised it (scpi warns then).
fn scpi_rho(res: ArrayView1<f64>, b: ArrayView2<f64>, d0: usize) -> EconResult<(f64, Option<f64>)> {
    let rho = regularize_w(false, RHO_MAX, res, b, d0)?;
    if rho < 0.001 {
        let r = regularize_w(true, 0.2, res, b, d0)?.max(regularize_w(false, 0.2, res, b, d0)?);
        return Ok((if r < 0.05 { RHO_MAX } else { r }, Some(rho)));
    }
    Ok((rho, None))
}

/// Lower bounds of the localised simplex (scpi's `local.geom.2step`): a weight below rho cannot
/// fall, any other weight can fall to 0.
fn simplex_lower_bounds(w: ArrayView1<f64>, rho: f64) -> Array1<f64> {
    w.mapv(|wj| if wj < rho { wj } else { 0.0 })
}

/// `[x, 1]`.
fn with_constant(x: ArrayView2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::ones((n, k + 1));
    out.slice_mut(s![.., ..k]).assign(&x);
    out
}

fn ones(n: usize) -> Array2<f64> {
    Array2::ones((n, 1))
}

/// Least-squares fit of `y` on `x` (pivoted QR, aliased coefficients set to 0), evaluated at
/// `eval` and at `x`.
fn lm_predict(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    eval: ArrayView2<f64>,
) -> EconResult<(Array1<f64>, Array1<f64>)> {
    let fit = least_squares(x, y)?;
    let coef = fit.coefficients.mapv(|c| if c.is_nan() { 0.0 } else { c });
    Ok((eval.dot(&coef), x.dot(&coef)))
}

fn rq(x: ArrayView2<f64>, y: ArrayView1<f64>, tau: f64) -> EconResult<RqFit> {
    let config = QuantRegConfig {
        algorithm: QuantRegMethod::FrischNewton,
        ..Default::default()
    };
    rq_fit(x, y, tau, &config)
}

/// `Qtools::rrq(y ~ x - 1, tau)` (He 1997's restricted regression quantiles, as scpi uses them),
/// evaluated at `eval`: the median fit, a median fit of its absolute residuals as the scale, and
/// for each tau a quantile of the residuals along that scale.
fn rrq_predict(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    eval: ArrayView2<f64>,
    taus: &[f64],
) -> EconResult<Vec<Array1<f64>>> {
    let lad = rq(x, y, 0.5)?;
    let r_abs = lad.residuals.mapv(f64::abs);
    let scale = rq(x, r_abs.view(), 0.5)?;
    let s_lad = x.dot(&scale.coefficients).insert_axis(Axis(1));
    taus.iter()
        .map(|&tau| {
            let zeta = rq(s_lad.view(), lad.residuals.view(), tau)?.coefficients[0];
            let coef = &lad.coefficients + &(&scale.coefficients * zeta);
            Ok(eval.dot(&coef))
        })
        .collect()
}

fn quantile7(v: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    quantile_sorted(&sorted, p, QuantileType::T7)
}

fn is_cancelled(e: &EconError) -> bool {
    matches!(e, EconError::Cancelled(_))
}

/// The out-of-sample bounds of scpi's `scpi.out()` (gaussian, ls) and qreg, at `alpha` per side.
fn out_of_sample_bounds(
    res: ArrayView1<f64>,
    x0: ArrayView2<f64>,
    x1: ArrayView2<f64>,
    order0: bool,
    alpha: f64,
    method: EMethod,
    warnings: &mut Vec<String>,
) -> EconResult<Vec<(EMethod, Vec<f64>, Vec<f64>)>> {
    let t1 = x1.nrows();
    let mut out = Vec::new();
    let want = |m: EMethod| method == EMethod::All || method == m;
    if want(EMethod::Gaussian) || want(EMethod::Ls) {
        let (e_mean, res_fit) = lm_predict(res, x0, x1)?;
        let resid = &res - &res_fit;
        let log_sq = resid.mapv(|r| (r * r).ln());
        if log_sq.iter().any(|v| !v.is_finite()) {
            return Err(EconError::Computation(
                "SCPI: a pre-treatment residual of the out-of-sample mean model is exactly 0, so \
                 its log-variance model (e_method gaussian and ls) is undefined; use e_method = \
                 \"qreg\""
                    .to_string(),
            ));
        }
        let (var_pred, var_in) = lm_predict(log_sq.view(), x0, x1)?;
        let iq = match rrq_predict(resid.view(), x0, x1, &[0.25, 0.75]) {
            Ok(q) => Some(
                Zip::from(&q[1])
                    .and(&q[0])
                    .map_collect(|a, b| (a - b).abs()),
            ),
            Err(e) if is_cancelled(&e) => return Err(e),
            Err(e) => {
                warnings.push(format!(
                    "SCPI: the quantile regression of the out-of-sample residuals failed ({e}); \
                     their scale is the log-variance model's alone"
                ));
                None
            }
        };
        let sigma: Vec<f64> = (0..t1)
            .map(|t| {
                let s = var_pred[t].exp().sqrt();
                match &iq {
                    Some(iq) => s.min(iq[t] / 1.34),
                    None => s,
                }
            })
            .collect();
        if want(EMethod::Gaussian) {
            let k = (-2.0 * alpha.ln()).sqrt();
            out.push((
                EMethod::Gaussian,
                (0..t1).map(|t| e_mean[t] - k * sigma[t]).collect(),
                (0..t1).map(|t| e_mean[t] + k * sigma[t]).collect(),
            ));
        }
        if want(EMethod::Ls) {
            let st: Vec<f64> = (0..resid.len())
                .map(|i| resid[i] / var_in[i].exp().sqrt())
                .collect();
            let (q_lo, q_hi) = (quantile7(&st, alpha), quantile7(&st, 1.0 - alpha));
            out.push((
                EMethod::Ls,
                (0..t1).map(|t| e_mean[t] + sigma[t] * q_lo).collect(),
                (0..t1).map(|t| e_mean[t] + sigma[t] * q_hi).collect(),
            ));
        }
    }
    if want(EMethod::Qreg) {
        let raw: Vec<f64> = res.to_vec();
        let unconditional = || {
            (
                vec![quantile7(&raw, alpha); t1],
                vec![quantile7(&raw, 1.0 - alpha); t1],
            )
        };
        let (lo, hi) = if order0 {
            unconditional()
        } else {
            match rrq_predict(res, x0, x1, &[alpha, 1.0 - alpha]) {
                Ok(q) => (q[0].to_vec(), q[1].to_vec()),
                Err(e) if is_cancelled(&e) => return Err(e),
                Err(e) => {
                    warnings.push(format!(
                        "SCPI: the quantile regression bounds failed ({e}); the qreg bounds are \
                         the residuals' unconditional quantiles"
                    ));
                    unconditional()
                }
            }
        };
        out.push((EMethod::Qreg, lo, hi));
    }
    Ok(out)
}

/// In-sample and out-of-sample uncertainty of the synthetic prediction, as scpi's `scpi()`.
fn scpi_inference(
    input: &InferenceInput<'_>,
    levels: &InferenceLevels,
    mut draws: Draws<'_>,
) -> EconResult<Inference> {
    let (b, p, res, w) = (input.b, input.p, input.res, input.w);
    let (t0, j) = b.dim();
    let t1 = p.nrows();
    let t0f = t0 as f64;
    let mut warnings = Vec::new();

    // Regularisation: rho, the active donors and the degrees of freedom (w before regularisation)
    let d0 = w.iter().filter(|v| v.abs() >= 1e-6).count();
    let (rho, raised_from) = scpi_rho(res, b, d0)?;
    if let Some(old) = raised_from {
        warnings.push(format!(
            "SCPI: the regularisation threshold rho was {old:.5}, below 0.001; raised to \
             {rho:.3} (check the donors for collinearity)"
        ));
    }
    let is_ridge = matches!(input.constraint, SCPIConstraint::Ridge { .. });
    let mut active: Vec<usize> = if is_ridge {
        (0..j).collect()
    } else {
        (0..j).filter(|&k| w[k].abs() > rho).collect()
    };
    if active.is_empty() {
        let top = (0..j)
            .max_by(|&a, &c| w[a].total_cmp(&w[c]).then(c.cmp(&a)))
            .unwrap_or(0);
        active.push(top);
        warnings.push(format!(
            "SCPI: no weight exceeds the regularisation threshold rho = {rho:.4}; the largest \
             weight is kept as the active donor"
        ));
    }
    let mut df = match input.constraint {
        SCPIConstraint::Simplex | SCPIConstraint::LassoSimplex { .. } => d0 as f64 - 1.0,
        SCPIConstraint::Lasso { .. } => d0 as f64,
        SCPIConstraint::Ridge { lambda } => {
            let (ev, _) = eig_symmetric(&xtx(&b).view())
                .map_err(|e| EconError::Computation(format!("SCPI ridge df: {e:?}")))?;
            ev.iter()
                .map(|&d2| {
                    let d2 = d2.max(0.0);
                    d2 / (d2 + lambda)
                })
                .sum()
        }
    };
    if df >= t0f {
        df = t0f - 1.0;
        warnings.push(
            "SCPI: the specification uses more degrees of freedom than pre-treatment periods; \
             consider a sparser fit or a smaller donor pool"
                .to_string(),
        );
    }

    // E[u | H] and the HC1 conditional variance
    let bb_active = b.select(Axis(1), &active);
    let u_mean = if levels.u_missp {
        let mut design = with_constant(bb_active.view());
        if (t0 as i64 - 10) <= design.ncols() as i64 {
            design = ones(t0);
        }
        lm_predict(res, design.view(), design.view())?.1
    } else {
        Array1::zeros(t0)
    };
    let vc = t0f / (t0f - df);
    let omega: Array1<f64> = Zip::from(&res)
        .and(&u_mean)
        .map_collect(|r, m| (r - m).powi(2) * vc);
    let mut bo = b.to_owned();
    for (mut row, o) in bo.axis_iter_mut(Axis(0)).zip(omega.iter()) {
        row *= *o;
    }
    let sigma = b.t().dot(&bo) / (t0f * t0f);
    let (ev, evec) = eig_symmetric(&sigma.view())
        .map_err(|e| EconError::Computation(format!("SCPI Sigma square root: {e:?}")))?;
    let mut scaled = evec.clone();
    for (mut col, l) in scaled.axis_iter_mut(Axis(1)).zip(ev.iter()) {
        col *= l.max(0.0).sqrt();
    }
    let sigma_root = scaled.dot(&evec.t());
    let q = xtx(&b) / t0f;

    // The localised constraint set, in delta = x - w
    let mut problem = match input.constraint {
        SCPIConstraint::Simplex | SCPIConstraint::LassoSimplex { .. } => {
            let lb = simplex_lower_bounds(w, rho);
            Qcqp {
                n: j,
                a_eq: Array2::ones((1, j)),
                b_eq: Array1::from_elem(1, 1.0 - w.sum()),
                g: -Array2::eye(j),
                h: &w - &lb,
                quads: vec![(q.clone(), Array1::zeros(j), 0.0)],
            }
        }
        SCPIConstraint::Lasso { .. } => {
            // variables (delta, v), v_k >= |w_k + delta_k|, sum v <= ||w||_1 + rho
            let mut g = Array2::zeros((2 * j + 1, 2 * j));
            let mut h = Array1::zeros(2 * j + 1);
            for k in 0..j {
                g[[k, k]] = 1.0;
                g[[k, j + k]] = -1.0;
                h[k] = -w[k];
                g[[j + k, k]] = -1.0;
                g[[j + k, j + k]] = -1.0;
                h[j + k] = w[k];
                g[[2 * j, j + k]] = 1.0;
            }
            h[2 * j] = w.mapv(f64::abs).sum() + rho;
            let mut qq = Array2::zeros((2 * j, 2 * j));
            qq.slice_mut(s![..j, ..j]).assign(&q);
            Qcqp {
                n: 2 * j,
                a_eq: Array2::zeros((0, 2 * j)),
                b_eq: Array1::zeros(0),
                g,
                h,
                quads: vec![(qq, Array1::zeros(2 * j), 0.0)],
            }
        }
        SCPIConstraint::Ridge { .. } => {
            // ||w + delta||^2 <= (||w|| + rho)^2
            let wn = w.dot(&w);
            let radius = wn.sqrt() + rho;
            Qcqp {
                n: j,
                a_eq: Array2::zeros((0, j)),
                b_eq: Array1::zeros(0),
                g: Array2::zeros((0, j)),
                h: Array1::zeros(0),
                quads: vec![
                    (q.clone(), Array1::zeros(j), 0.0),
                    (Array2::eye(j), &w * 2.0, wn - radius * radius),
                ],
            }
        }
    };
    let n = problem.n;
    // the localised set is the single point delta = 0 when no weight can move
    let pinned = matches!(
        input.constraint,
        SCPIConstraint::Simplex | SCPIConstraint::LassoSimplex { .. }
    ) && problem.h.iter().all(|&v| v <= 0.0);

    let sims = levels.sims;
    let n_cells = crate::resource::checked_len(&[sims, t1], "SCPI simulations x post periods")?;
    let mut lb_f = crate::resource::try_vec(n_cells, f64::NAN, "SCPI simulation bounds")?;
    let mut ub_f = crate::resource::try_vec(n_cells, f64::NAN, "SCPI simulation bounds")?;
    let mut c = Array1::zeros(n);
    // the linear rows are the same in every simulation: scale them once and the simulated
    // quadratic per simulation
    let mut sc = problem.scaled();
    for sim in 0..sims {
        crate::resource::checkpoint()?;
        let z = draws.next(sim, j)?;
        let g_vec = sigma_root.dot(&z);
        // delta' Q delta - 2 G' delta <= 0
        problem.quads[0]
            .1
            .slice_mut(s![..j])
            .assign(&(&g_vec * -2.0));
        if pinned {
            for t in 0..t1 {
                lb_f[sim * t1 + t] = 0.0;
                ub_f[sim * t1 + t] = 0.0;
            }
            continue;
        }
        // one strictly feasible point serves every objective of this simulation
        sc.quads[0] = scale_quad(&problem.quads[0]);
        let Some(start) = sc.start()? else {
            continue;
        };
        for t in 0..t1 {
            crate::resource::checkpoint()?;
            let pt = p.row(t);
            // lower: -max p'delta = min (-p)'delta; upper: -min p'delta
            c.slice_mut(s![..j]).assign(&pt.mapv(|v| -v));
            if let Some(v) = sc.minimize_from(c.view(), &start)? {
                lb_f[sim * t1 + t] = v;
            }
            c.slice_mut(s![..j]).assign(&pt);
            if let Some(v) = sc.minimize_from(c.view(), &start)? {
                ub_f[sim * t1 + t] = -v;
            }
        }
    }
    let n_failed = (0..sims)
        .filter(|&s| (0..t1).any(|t| lb_f[s * t1 + t].is_nan() || ub_f[s * t1 + t].is_nan()))
        .count();
    if n_failed > 0 {
        warnings.push(format!(
            "SCPI: the in-sample optimisation failed in {n_failed} of {sims} simulations at some \
             post period; the bounds there use the other simulations"
        ));
    }
    let column = |v: &[f64], t: usize| -> Vec<f64> { (0..sims).map(|s| v[s * t1 + t]).collect() };
    let mut insample_lower = Vec::with_capacity(t1);
    let mut insample_upper = Vec::with_capacity(t1);
    for t in 0..t1 {
        let lo = quantile7(&column(&lb_f, t), levels.u_alpha / 2.0);
        let hi = quantile7(&column(&ub_f, t), 1.0 - levels.u_alpha / 2.0);
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(EconError::Computation(format!(
                "SCPI: the in-sample optimisation failed in every simulation at post period {t}"
            )));
        }
        insample_lower.push(lo);
        insample_upper.push(hi);
    }

    // Out-of-sample design: the active donors and a constant, or only the constant when the
    // pre-treatment periods do not exceed its columns by more than 10
    let p_active = p.select(Axis(1), &active);
    let (mut x0, mut x1) = (
        with_constant(bb_active.view()),
        with_constant(p_active.view()),
    );
    let order0 = (t0 as i64 - 10) <= x0.ncols() as i64;
    if order0 {
        x0 = ones(t0);
        x1 = ones(t1);
    } else {
        // scpi's detectConstant: drop all-ones and zero-sum columns, append the constant
        let keep: Vec<usize> = (0..x0.ncols() - 1)
            .filter(|&k| {
                let col = x0.column(k);
                !col.iter().all(|&v| v == 1.0) && col.sum() != 0.0
            })
            .collect();
        x0 = with_constant(x0.select(Axis(1), &keep).view());
        x1 = with_constant(x1.select(Axis(1), &keep).view());
        // scpi's avoidCollin
        if x0.ncols() + 1 >= t0 {
            x0 = ones(t0);
            x1 = ones(t1);
        }
    }
    let e_bounds = out_of_sample_bounds(
        res,
        x0.view(),
        x1.view(),
        order0,
        levels.e_alpha / 2.0,
        levels.e_method,
        &mut warnings,
    )?;

    Ok(Inference {
        rho,
        insample_lower,
        insample_upper,
        e_bounds,
        n_failed,
        warnings,
        sigma,
        u_mean,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convex QCQP (logarithmic barrier)
// ═══════════════════════════════════════════════════════════════════════════════

/// `min c'x` subject to `A x = b`, `G x <= h` and `x' P_k x + a_k' x + d_k <= 0` (`P_k` positive
/// semi-definite): the in-sample problems of scpi, which R solves as second-order cone programs
/// with ECOS. Solved here by the logarithmic barrier method (Boyd and Vandenberghe, Convex
/// Optimization, sections 11.3-11.4): a phase I finds a strictly feasible point, which every
/// objective over the same set then starts from.
struct Qcqp {
    n: usize,
    a_eq: Array2<f64>,
    b_eq: Array1<f64>,
    g: Array2<f64>,
    h: Array1<f64>,
    quads: Vec<(Array2<f64>, Array1<f64>, f64)>,
}

/// A point of a [`Qcqp`]'s feasible set: strictly inside it, or (`interior == false`) the set has
/// no interior and the point is where the phase I ended (the set is then, up to rounding, that
/// point: a set of zero width, such as the single weight vector the in-sample problem allows
/// when the simulated G vanishes).
struct QcqpStart {
    x: Array1<f64>,
    interior: bool,
}

/// Duality gap of the barrier method relative to the problem scale (the objective and every
/// constraint are scaled to unit size).
const QCQP_GAP: f64 = 1e-10;
const QCQP_NEWTON_MAX: usize = 60;
/// Largest Newton decrement `lambda^2` of the last centring that still certifies the gap: for
/// the logarithmic barrier of `m` constraints, `lambda <= 1/2` gives `c'x - p* <= (2m + 1) / t`
/// (Nesterov, Introductory Lectures on Convex Optimization, section 4.2), within a factor of
/// about 2 of the centred bound `m / t`.
const QCQP_CENTRED: f64 = 0.25;
/// Growth factor of t between centrings, and the smallest one the retries of
/// [`Qcqp::barrier`] may shrink it to (each retry takes the square root of the factor, so the
/// factor never falls below the floor).
const QCQP_MU: f64 = 50.0;
const QCQP_MU_MIN: f64 = 2.0;

/// `hess += sum_i w_i d_i d_i'` over the rows `d_i` of `rows`.
fn add_weighted_gram(hess: &mut Array2<f64>, rows: &Array2<f64>, w: &Array1<f64>) {
    let n = hess.ncols();
    for (row, &wi) in rows.axis_iter(Axis(0)).zip(w.iter()) {
        for a in 0..n {
            if row[a] == 0.0 {
                continue;
            }
            let ra = wi * row[a];
            for b in 0..n {
                hess[[a, b]] += ra * row[b];
            }
        }
    }
}

/// Solves `[H A'; A 0] (dx, dnu) = (rx, ry)` by LU, with `H` shifted by `1e-14` of its largest
/// diagonal element (a singular `H` with no active bound otherwise has no solution); `None` when
/// the solution is not finite.
fn solve_kkt(
    hess: &Array2<f64>,
    a_eq: &Array2<f64>,
    rx: &Array1<f64>,
    ry: &Array1<f64>,
) -> Option<(Array1<f64>, Array1<f64>)> {
    let (n, m) = (hess.nrows(), a_eq.nrows());
    let reg = 1e-14 * (1.0 + hess.diag().iter().fold(0.0f64, |a, v| a.max(v.abs())));
    let kkt = Mat::<f64>::from_fn(n + m, n + m, |i, j| {
        if i < n && j < n {
            hess[[i, j]] + if i == j { reg } else { 0.0 }
        } else if i < n {
            a_eq[[j - n, i]]
        } else if j < n {
            a_eq[[i - n, j]]
        } else {
            0.0
        }
    });
    let rhs = Col::<f64>::from_fn(n + m, |i| if i < n { rx[i] } else { ry[i - n] });
    let sol = kkt.partial_piv_lu().solve(&rhs);
    let dx = Array1::from_iter((0..n).map(|i| sol[i]));
    let dnu = Array1::from_iter((0..m).map(|i| sol[n + i]));
    dx.iter()
        .chain(dnu.iter())
        .all(|v| v.is_finite())
        .then_some((dx, dnu))
}

/// A quadratic constraint `x' P x + a' x + d <= 0` divided by its largest coefficient.
fn scale_quad((pk, ak, dk): &(Array2<f64>, Array1<f64>, f64)) -> (Array2<f64>, Array1<f64>, f64) {
    let s = pk
        .iter()
        .chain(ak.iter())
        .fold(dk.abs(), |a, v| a.max(v.abs()));
    let s = if s > 0.0 { s } else { 1.0 };
    (pk / s, ak / s, dk / s)
}

/// Newton steps per centring ([`QCQP_NEWTON_MAX`]; tests can lower it to reach the stall path).
#[cfg(not(test))]
fn newton_max() -> usize {
    QCQP_NEWTON_MAX
}

#[cfg(test)]
thread_local! {
    static NEWTON_MAX: std::cell::Cell<usize> = const { std::cell::Cell::new(QCQP_NEWTON_MAX) };
}

#[cfg(test)]
fn newton_max() -> usize {
    NEWTON_MAX.with(|c| c.get())
}

impl Qcqp {
    /// Each quadratic and each linear inequality scaled to unit size. [`Qcqp::start`] and
    /// [`Qcqp::minimize_from`] work on the scaled problem.
    fn scaled(&self) -> Qcqp {
        let quads = self.quads.iter().map(scale_quad).collect();
        let mut g = self.g.clone();
        let mut h = self.h.clone();
        for (mut row, hv) in g.axis_iter_mut(Axis(0)).zip(h.iter_mut()) {
            let s = row.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            if s > 0.0 {
                row /= s;
                *hv /= s;
            }
        }
        Qcqp {
            n: self.n,
            a_eq: self.a_eq.clone(),
            b_eq: self.b_eq.clone(),
            g,
            h,
            quads,
        }
    }

    /// Slacks `h - G x` and `-f_k(x)`, or `None` when one is not positive.
    fn slacks(&self, x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        let r = &self.h - &self.g.dot(x);
        let f: Array1<f64> = self
            .quads
            .iter()
            .map(|(pk, ak, dk)| -(x.dot(&pk.dot(x)) + ak.dot(x) + dk))
            .collect();
        (r.iter().chain(f.iter()).all(|v| *v > 0.0 && v.is_finite())).then_some((r, f))
    }

    /// Constraint values `f_i(x)` (linear rows first, then the quadratics) and their gradients
    /// as the rows of a matrix.
    fn constraints(&self, x: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let (l, kq) = (self.g.nrows(), self.quads.len());
        let mut f = Array1::zeros(l + kq);
        let mut df = Array2::zeros((l + kq, self.n));
        f.slice_mut(s![..l]).assign(&(&self.g.dot(x) - &self.h));
        df.slice_mut(s![..l, ..]).assign(&self.g);
        for (k, (pk, ak, dk)) in self.quads.iter().enumerate() {
            let px = pk.dot(x);
            f[l + k] = x.dot(&px) + ak.dot(x) + dk;
            df.row_mut(l + k).assign(&(&px * 2.0 + ak));
        }
        (f, df)
    }

    /// The Newton step of the barrier problem `min t c'x - sum log(-f_i(x))` subject to
    /// `A x = b` at the strictly feasible `x`, and its Newton decrement `lambda^2`; `None` when
    /// the KKT system fails.
    fn newton_step(&self, c: &Array1<f64>, x: &Array1<f64>, t: f64) -> Option<(Array1<f64>, f64)> {
        let (n, m) = (self.n, self.a_eq.nrows());
        let l = self.g.nrows();
        let (f, df) = self.constraints(x);
        // gradient t c + sum df_i / -f_i; Hessian sum (df_i df_i' / f_i^2 + 2 P_i / -f_i)
        let inv = f.mapv(|v| -1.0 / v);
        let grad = c * t + &df.t().dot(&inv);
        let mut hess = Array2::<f64>::zeros((n, n));
        for (k, (pk, _, _)) in self.quads.iter().enumerate() {
            hess.scaled_add(2.0 * inv[l + k], pk);
        }
        add_weighted_gram(&mut hess, &df, &inv.mapv(|v| v * v));
        let (dx, _) = solve_kkt(&hess, &self.a_eq, &(-&grad), &Array1::zeros(m))?;
        // the Newton decrement: half of it bounds the centring error
        let decrement = -grad.dot(&dx);
        Some((dx, decrement))
    }

    /// Newton's method on the barrier problem `min t c'x - sum log(-f_i(x))` subject to `A x = b`
    /// (Boyd and Vandenberghe, sections 10.2 and 11.3) from the strictly feasible `x`: the point
    /// it ends at and the Newton decrement `lambda^2` at that point (0 when the steps stopped at
    /// `stop_below`). Newton stops once half the decrement is at most 1e-9, when the line search
    /// cannot move, or after [`QCQP_NEWTON_MAX`] steps; in the last case the decrement is
    /// computed once more at the point returned. With `stop_below`, it returns as soon as `c'x`
    /// is below that value.
    fn center(
        &self,
        c: &Array1<f64>,
        mut x: Array1<f64>,
        t: f64,
        stop_below: Option<f64>,
    ) -> Option<(Array1<f64>, f64)> {
        let phi = |x: &Array1<f64>| -> Option<f64> {
            let (r, f) = self.slacks(x)?;
            Some(t * c.dot(x) - r.mapv(f64::ln).sum() - f.mapv(f64::ln).sum())
        };
        for _ in 0..newton_max() {
            if stop_below.is_some_and(|v| c.dot(&x) < v) {
                return Some((x, 0.0));
            }
            let (dx, decrement) = self.newton_step(c, &x, t)?;
            if decrement / 2.0 <= 1e-9 {
                return Some((x, decrement));
            }
            // backtracking line search inside the feasible set; the Armijo test allows for the
            // rounding of phi, whose terms grow with t
            let f0 = phi(&x)?;
            let slack = 1e-13 * (1.0 + f0.abs());
            let mut step = 1.0;
            let mut moved = false;
            while step >= 1e-10 {
                let xn = &x + &(&dx * step);
                if let Some(fv) = phi(&xn)
                    && fv <= f0 - 0.25 * step * decrement + slack
                {
                    x = xn;
                    moved = true;
                    break;
                }
                step *= 0.5;
            }
            if !moved {
                return Some((x, decrement));
            }
        }
        // out of steps: the decrement of the point returned, not of the one before the last step
        if stop_below.is_some_and(|v| c.dot(&x) < v) {
            return Some((x, 0.0));
        }
        let (_, decrement) = self.newton_step(c, &x, t)?;
        Some((x, decrement))
    }

    /// The barrier method (Boyd and Vandenberghe, section 11.3) from the strictly feasible `x`:
    /// centring steps at t = 1, 50, 2500, ... until the duality gap `m / t` of the scaled problem
    /// is below [`QCQP_GAP`]. The gap bound holds only near the central path, so a centring that
    /// ends with a decrement above [`QCQP_CENTRED`] is retried from the last centred point with t
    /// grown by the square root of the previous factor, as long as that root is at least
    /// [`QCQP_MU_MIN`] (50, 7.07, 2.66); after that the path goes on from the uncentred point,
    /// but the last centring must end centred. `Ok(None)`
    /// when a Newton system fails or the last centring ends uncentred (as an ECOS failure in
    /// scpi). The budget is checked once per centring.
    fn barrier(&self, c: &Array1<f64>, x0: &Array1<f64>) -> EconResult<Option<Array1<f64>>> {
        let n_ineq = (self.g.nrows() + self.quads.len()) as f64;
        let mut x = x0.clone();
        let mut t = 1.0;
        let mut mu = QCQP_MU;
        // the last centred point and its t
        let mut prev: Option<(Array1<f64>, f64)> = None;
        loop {
            crate::resource::checkpoint()?;
            let Some((xn, decrement)) = self.center(c, x, t, None) else {
                return Ok(None);
            };
            let last = n_ineq / t < QCQP_GAP;
            if decrement <= QCQP_CENTRED {
                if last {
                    return Ok(Some(xn));
                }
                prev = Some((xn.clone(), t));
            } else {
                if mu.sqrt() >= QCQP_MU_MIN
                    && let Some((xp, tp)) = &prev
                {
                    mu = mu.sqrt();
                    x = xp.clone();
                    t = tp * mu;
                    continue;
                }
                if last {
                    return Ok(None);
                }
            }
            x = xn;
            t *= mu;
        }
    }

    /// Phase I: minimise `s` subject to `G x - s <= h`, `f_k(x) - s <= 0` and `A x = b`, from the
    /// least-norm solution of `A x = b`, on a problem already [`Qcqp::scaled`]. The budget is
    /// checked once per centring.
    fn start(&self) -> EconResult<Option<QcqpStart>> {
        let sc = self;
        let (n, m) = (sc.n, sc.a_eq.nrows());
        let x0 = if m == 0 {
            Array1::zeros(n)
        } else {
            // x0 = A' (A A')^{-1} b
            let aat = sc.a_eq.dot(&sc.a_eq.t());
            let lhs = Mat::<f64>::from_fn(m, m, |i, j| aat[[i, j]]);
            let rhs = Col::<f64>::from_fn(m, |i| sc.b_eq[i]);
            let y = lhs.partial_piv_lu().solve(&rhs);
            sc.a_eq.t().dot(&Array1::from_iter((0..m).map(|i| y[i])))
        };
        let viol = (&sc.g.dot(&x0) - &sc.h)
            .iter()
            .copied()
            .chain(
                sc.quads
                    .iter()
                    .map(|(pk, ak, dk)| x0.dot(&pk.dot(&x0)) + ak.dot(&x0) + dk),
            )
            .fold(f64::NEG_INFINITY, f64::max);
        if !viol.is_finite() {
            return Ok((viol == f64::NEG_INFINITY).then_some(QcqpStart {
                x: x0,
                interior: true,
            }));
        }
        let mut g1 = Array2::zeros((sc.g.nrows(), n + 1));
        g1.slice_mut(s![.., ..n]).assign(&sc.g);
        g1.column_mut(n).fill(-1.0);
        let quads1 = sc
            .quads
            .iter()
            .map(|(pk, ak, dk)| {
                let mut p1 = Array2::zeros((n + 1, n + 1));
                p1.slice_mut(s![..n, ..n]).assign(pk);
                let mut a1 = Array1::from_elem(n + 1, -1.0);
                a1.slice_mut(s![..n]).assign(ak);
                (p1, a1, *dk)
            })
            .collect();
        let mut a1 = Array2::zeros((m, n + 1));
        a1.slice_mut(s![.., ..n]).assign(&sc.a_eq);
        let phase1 = Qcqp {
            n: n + 1,
            a_eq: a1,
            b_eq: sc.b_eq.clone(),
            g: g1,
            h: sc.h.clone(),
            quads: quads1,
        };
        let mut z0 = Array1::zeros(n + 1);
        z0.slice_mut(s![..n]).assign(&x0);
        z0[n] = viol.max(0.0) + 1.0;
        let mut c1 = Array1::zeros(n + 1);
        c1[n] = 1.0;
        let n_ineq_1 = (phase1.g.nrows() + phase1.quads.len()) as f64;
        // a centred point well inside: stop once s is below -1e-3 (the scale is one)
        let mut z = z0;
        let mut t = 1.0;
        while z[n] >= -1e-3 && n_ineq_1 / t >= QCQP_GAP {
            crate::resource::checkpoint()?;
            let Some((zn, _)) = phase1.center(&c1, z, t, Some(-1e-3)) else {
                return Ok(None);
            };
            z = zn;
            t *= 20.0;
        }
        let x = z.slice(s![..n]).to_owned();
        let interior = z[n] < -1e-12 && sc.slacks(&x).is_some();
        Ok(Some(QcqpStart { x, interior }))
    }

    /// The optimal value from `start` on a problem already [`Qcqp::scaled`], or `Ok(None)` when
    /// the barrier iterations fail or end uncentred (as an ECOS failure in scpi).
    fn minimize_from(&self, c: ArrayView1<f64>, start: &QcqpStart) -> EconResult<Option<f64>> {
        let c_scale = c.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if c_scale == 0.0 || !start.interior {
            return Ok(Some(c.dot(&start.x)));
        }
        let cs = &c / c_scale;
        Ok(self.barrier(&cs, &start.x)?.map(|x| cs.dot(&x) * c_scale))
    }

    #[cfg(test)]
    fn minimize(&self, c: ArrayView1<f64>) -> EconResult<Option<f64>> {
        let sc = self.scaled();
        match sc.start()? {
            Some(start) => sc.minimize_from(c, &start),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};

    fn generate_test_data() -> (Array1<f64>, Array2<f64>) {
        // Generate simple test data
        // Treated unit: trend + treatment effect after period 7
        let treated = array![
            10.0, 11.5, 12.8, 14.1, 15.5, 16.9, 18.2, // Pre-treatment
            22.0, 23.5, 25.0 // Post-treatment (with +3 effect)
        ];

        // Donors: similar trends
        let donors = Array2::from_shape_fn((10, 5), |(t, j)| {
            10.0 + 1.4 * t as f64 + 0.3 * j as f64 + 0.1 * ((t * j) as f64).sin()
        });

        (treated, donors)
    }

    #[test]
    fn test_scpi_basic() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig::default();
        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_pre_periods, 7);
        assert_eq!(result.n_post_periods, 3);
        assert_eq!(result.n_donors, 5);
        assert_eq!(result.treatment_period, 7);

        // Weights should sum to 1 (simplex constraint)
        let weight_sum: f64 = result.weights.sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-6);

        // All weights should be non-negative
        assert!(result.weights.iter().all(|&w| w >= -1e-10));

        // Should have prediction intervals
        assert_eq!(result.prediction_intervals.len(), 3);

        // Intervals should be ordered: lower < effect < upper
        for (i, pi) in result.prediction_intervals.iter().enumerate() {
            // Note: effect is actual - synthetic, PI bounds are for the effect
            let effect = result.effect[i];
            assert!(pi.lower < effect || (pi.lower - effect).abs() < 1e-10);
            assert!(effect < pi.upper || (effect - pi.upper).abs() < 1e-10);
        }

        println!("{}", result);
    }

    #[test]
    fn test_scpi_lasso() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::Lasso { lambda: 0.1 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Lasso should produce sparse weights
        assert!(result.n_effective_donors <= result.n_donors);

        println!("Lasso result:\n{}", result);
    }

    #[test]
    fn test_scpi_ridge() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::Ridge { lambda: 0.1 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Ridge weights typically non-zero
        assert!(result.weights.len() == result.n_donors);

        println!("Ridge result:\n{}", result);
    }

    /// CAUS-81: on the simplex the L1 norm is 1, so lasso_simplex is the simplex
    /// least squares fit whatever lambda (before, ADMM stopped at weights that
    /// depended on lambda and the iteration limit).
    #[test]
    fn test_lasso_simplex_equals_simplex() {
        let (treated, donors) = generate_test_data();
        let simplex = run_scpi(&treated.view(), &donors.view(), 7, SCPIConfig::default()).unwrap();
        for lambda in [0.05, 0.2, 1.0] {
            let config = SCPIConfig {
                constraint: SCPIConstraint::LassoSimplex { lambda },
                ..Default::default()
            };
            let r = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();
            for (a, b) in r.weights.iter().zip(simplex.weights.iter()) {
                assert!((a - b).abs() <= 1e-10, "lambda {lambda}: {a} vs {b}");
            }
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.starts_with("lasso_simplex: on the simplex the L1 penalty")),
                "{:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn test_scpi_lasso_simplex() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::LassoSimplex { lambda: 0.05 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Should satisfy simplex constraint
        let weight_sum: f64 = result.weights.sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-4);
        assert!(result.weights.iter().all(|&w| w >= -1e-6));

        println!("Lasso-Simplex result:\n{}", result);
    }

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(1.5, 0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(-1.5, 0.5), -1.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(0.3, 0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(-0.3, 0.5), 0.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Parameter bounds and cancellation (T0-WP11)
    // =========================================================================

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        with_budget(Budget::new(None, token), f)
    }

    fn spec_msg(r: EconResult<SCPIResult>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok={}", other.is_ok()),
        }
    }

    #[test]
    fn test_scpi_rejects_bad_tolerance_and_max_iter() {
        let (t, d) = generate_test_data();
        for tol in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
            let m = spec_msg(run_scpi(
                &t.view(),
                &d.view(),
                7,
                SCPIConfig {
                    tolerance: tol,
                    ..Default::default()
                },
            ));
            assert!(m.contains("tolerance"), "{m}");
        }
        let m = spec_msg(run_scpi(
            &t.view(),
            &d.view(),
            7,
            SCPIConfig {
                max_iter: 0,
                ..Default::default()
            },
        ));
        assert!(m.contains("max_iter"), "{m}");
    }

    #[test]
    fn test_scpi_stops_when_cancelled() {
        let (t, d) = generate_test_data();
        for constraint in [
            SCPIConstraint::Simplex,
            SCPIConstraint::Lasso { lambda: 0.1 },
            SCPIConstraint::LassoSimplex { lambda: 0.05 },
            SCPIConstraint::Ridge { lambda: 0.1 },
        ] {
            let cfg = SCPIConfig {
                constraint,
                ..Default::default()
            };
            let r = cancelled(|| run_scpi(&t.view(), &d.view(), 7, cfg));
            assert!(matches!(r, Err(EconError::Cancelled(_))));
        }
    }

    #[test]
    fn test_scpi_rejects_too_few_sims() {
        let (t, d) = generate_test_data();
        for sims in [0usize, 9] {
            let cfg = SCPIConfig {
                sims,
                ..Default::default()
            };
            let m = spec_msg(run_scpi(&t.view(), &d.view(), 7, cfg));
            assert!(m.contains("sims must be at least 10"), "{m}");
        }
    }

    #[test]
    fn test_scpi_insufficient_data() {
        let treated = array![1.0, 2.0, 3.0];
        let donors = Array2::from_shape_fn((3, 2), |(t, j)| t as f64 + j as f64);

        // Treatment at period 1 (only 1 pre-period)
        let result = run_scpi(&treated.view(), &donors.view(), 1, SCPIConfig::default());
        assert!(result.is_err());
    }

    // =========================================================================
    // Property tests on a known DGP
    // =========================================================================

    /// Create a dataset with a known treatment effect for validation.
    ///
    /// Treated unit follows donors closely in pre-treatment,
    /// then diverges by +5 in post-treatment.
    fn generate_validate_data() -> (Array1<f64>, Array2<f64>) {
        let n_pre = 10;
        let n_post = 4;
        let true_effect = 5.0;

        // Donors: smooth trends with slight differences
        let donors = Array2::from_shape_fn((n_pre + n_post, 6), |(t, j)| {
            10.0 + 1.2 * t as f64 + 0.5 * j as f64 + 0.05 * ((t * (j + 1)) as f64).sin()
        });

        // Treated unit: weighted combination of donors 0 and 1 in pre-treatment
        // with added treatment effect in post-treatment
        let treated = Array1::from_shape_fn(n_pre + n_post, |t| {
            let synth_val = 0.5 * donors[[t, 0]] + 0.3 * donors[[t, 1]] + 0.2 * donors[[t, 2]];
            if t >= n_pre {
                synth_val + true_effect
            } else {
                synth_val
            }
        });

        (treated, donors)
    }

    /// Wider intervals at a higher level (property test): in-sample quantiles further out and
    /// larger out-of-sample bounds for every method.
    #[test]
    fn test_scpi_interval_width_increases_with_level() {
        let (treated, donors) = generate_validate_data();
        let run = |alpha: f64| {
            let config = SCPIConfig {
                alpha: Some(alpha),
                ..Default::default()
            };
            run_scpi(&treated.view(), &donors.view(), 10, config).unwrap()
        };
        let (r95, r90) = (run(0.05), run(0.10));
        assert_eq!((r95.u_alpha, r95.e_alpha), (0.025, 0.025));
        for (m95, m90) in r95.method_intervals.iter().zip(&r90.method_intervals) {
            assert_eq!(m95.method, m90.method);
            for i in 0..m95.lower.len() {
                let (w95, w90) = (m95.upper[i] - m95.lower[i], m90.upper[i] - m90.lower[i]);
                assert!(w95 > w90, "{:?} period {i}: {w95} vs {w90}", m95.method);
            }
        }
        for i in 0..r95.insample_lower.len() {
            assert!(r95.insample_lower[i] <= r90.insample_lower[i]);
            assert!(r95.insample_upper[i] >= r90.insample_upper[i]);
        }
    }

    /// Coverage of the default interval (sub-Gaussian, nominal at least 90% by the union bound)
    /// over 200 seeded DGPs with a known effect of 2: at least 90% minus three Monte Carlo
    /// standard errors of the period-level hit rate.
    #[test]
    fn test_scpi_coverage_on_known_effect() {
        if cfg!(debug_assertions) {
            return; // 200 fits with 200 simulations each: release only
        }
        use rand::Rng;
        let (reps, n_pre, n_post, j) = (200usize, 30usize, 4usize, 6usize);
        let mut hits = 0usize;
        for rep in 0..reps {
            let mut rng = StdRng::seed_from_u64(9000 + rep as u64);
            let mut donors = Array2::zeros((n_pre + n_post, j));
            for k in 0..j {
                let (level, slope) = (rng.gen_range(5.0..15.0), rng.gen_range(0.1..0.5));
                for t in 0..n_pre + n_post {
                    let e: f64 = StandardNormal.sample(&mut rng);
                    donors[[t, k]] = level + slope * t as f64 + 0.5 * e;
                }
            }
            let w0 = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0];
            let treated = Array1::from_shape_fn(n_pre + n_post, |t| {
                let e: f64 = StandardNormal.sample(&mut rng);
                let base: f64 = (0..j).map(|k| w0[k] * donors[[t, k]]).sum();
                base + 0.3 * e + if t >= n_pre { 2.0 } else { 0.0 }
            });
            let cfg = SCPIConfig {
                seed: Some(rep as u64),
                ..Default::default()
            };
            let r = run_scpi(&treated.view(), &donors.view(), n_pre, cfg).unwrap();
            hits += r
                .prediction_intervals
                .iter()
                .filter(|p| p.lower <= 2.0 && 2.0 <= p.upper)
                .count();
        }
        let total = (reps * n_post) as f64;
        let rate = hits as f64 / total;
        let se = (0.9 * 0.1 / total).sqrt();
        assert!(rate >= 0.9 - 3.0 * se, "coverage {rate}");
    }

    // =========================================================================
    // R pins (scpi_quantities) and validation (T3-WP31)
    // =========================================================================

    use p2a_validation::{Check, Tol};

    /// Treated (TR) and donor (D01..D08) outcome series of the planner case, times 1..20.
    fn scpi_case(id: &str) -> (p2a_validation::Case, Array1<f64>, Array2<f64>) {
        let (case, ds) = crate::test_support::load_case(id);
        let df = ds.df();
        let unit = df.column("unit").unwrap().str().unwrap().clone();
        let time = df
            .column("time")
            .unwrap()
            .cast(&polars::prelude::DataType::Int64)
            .unwrap();
        let time = time.i64().unwrap().clone();
        let y = df
            .column("y")
            .unwrap()
            .cast(&polars::prelude::DataType::Float64)
            .unwrap();
        let y = y.f64().unwrap().clone();
        let donors: Vec<String> = (1..=8).map(|j| format!("D{j:02}")).collect();
        let mut treated = Array1::<f64>::zeros(20);
        let mut x = Array2::<f64>::zeros((20, 8));
        for i in 0..df.height() {
            let (u, t, v) = (
                unit.get(i).unwrap(),
                time.get(i).unwrap() as usize - 1,
                y.get(i).unwrap(),
            );
            if u == "TR" {
                treated[t] = v
            } else {
                x[[t, donors.iter().position(|d| d == u).unwrap()]] = v
            }
        }
        (case, treated, x)
    }

    const DONORS: [&str; 8] = ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"];

    #[test]
    fn test_validate_scpi_simplex_weights() {
        let (case, t, x) = scpi_case("scpi_quantities");
        let r = run_scpi(&t.view(), &x.view(), 14, SCPIConfig::default()).unwrap();
        Check::new(&case)
            .named("w", &DONORS, r.weights.as_slice().unwrap(), Tol::coef(14))
            .vector("synthetic", r.synthetic.as_slice().unwrap(), Tol::coef(14))
            .vector("effect", r.effect.as_slice().unwrap(), Tol::coef(14))
            .scalar("pre_rmspe", r.pre_treatment_rmspe, Tol::stat(14))
            .scalar("pre_r2", r.pre_treatment_r2, Tol::stat(14))
            .finish();
        assert_eq!(r.n_effective_donors, 4);
    }

    /// R's draws (`draws`, 8 x 200, column-major) as a J x sims matrix.
    fn r_draws(case: &p2a_validation::Case) -> Array2<f64> {
        Array2::from_shape_vec((200, 8), case.vector("draws"))
            .unwrap()
            .reversed_axes()
    }

    fn method<'a>(r: &'a SCPIResult, m: EMethod) -> &'a SCPIMethodIntervals {
        r.method_intervals.iter().find(|i| i.method == m).unwrap()
    }

    /// The inference stage on R's weights and R's draws, as scpi computes it after scest.
    fn r_inference(
        case: &p2a_validation::Case,
        t: &Array1<f64>,
        x: &Array2<f64>,
        u_alpha: f64,
        e_alpha: f64,
        u_missp: bool,
    ) -> (Inference, Array1<f64>) {
        let w = Array1::from(case.by_names("w", &DONORS));
        let (xp, xq) = (x.slice(s![..14, ..]), x.slice(s![14.., ..]));
        let res = &t.slice(s![..14]) - &xp.dot(&w);
        let z = r_draws(case);
        let inf = scpi_inference(
            &InferenceInput {
                b: xp,
                p: xq,
                res: res.view(),
                w: w.view(),
                constraint: &SCPIConstraint::Simplex,
            },
            &InferenceLevels {
                u_alpha,
                e_alpha,
                sims: 200,
                u_missp,
                e_method: EMethod::All,
            },
            Draws::Given(z.view()),
        )
        .unwrap();
        (inf, xq.dot(&w))
    }

    /// The out-of-sample bounds of method `m` and the interval `CI.all.<m>`: the prediction plus
    /// the in-sample and the out-of-sample bounds.
    fn r_bounds(inf: &Inference, m: EMethod, syn: &Array1<f64>) -> [Vec<f64>; 4] {
        let (_, lo, hi) = inf.e_bounds.iter().find(|(k, _, _)| *k == m).unwrap();
        let clo = (0..6)
            .map(|i| syn[i] + inf.insample_lower[i] + lo[i])
            .collect();
        let chi = (0..6)
            .map(|i| syn[i] + inf.insample_upper[i] + hi[i])
            .collect();
        [lo.clone(), hi.clone(), clo, chi]
    }

    /// CAUS-32: scpi 3.0.1's scpi() at its defaults, on R's weights and R's own draws.
    ///
    /// Tolerances. rho, Sigma and the pseudo-residual mean: 1e-8 (closed forms on R's weights).
    /// The in-sample bounds are quantiles of 200 optima of the in-sample programs; scpi solves them
    /// with ECOS at its default tolerances, about 1e-5 from the optimum, so they are compared with
    /// the same programs re-solved in R at ECOS tolerance 1e-12 (`insample_tight_*`) at 1e-7, and
    /// with scpi's own bounds at 1e-4. The gaussian and ls out-of-sample bounds: 1e-6 (log-variance
    /// regression and sample quantiles); the qreg bounds 1e-5 (quantile regressions solved by
    /// Frisch-Newton interior points stopped at a duality gap of 1e-6, in R and here). The
    /// intervals inherit scpi's in-sample error: 1e-4.
    #[test]
    fn test_validate_scpi_default_intervals() {
        let (case, t, x) = scpi_case("scpi_default_intervals");
        let (inf, syn) = r_inference(&case, &t, &x, 0.05, 0.05, true);
        let closed = Tol::iterative(1e-8);
        let tight = Tol::iterative(1e-7);
        let ecos = Tol::iterative(1e-4);
        let out = Tol::iterative(1e-6);
        let fn_tol = Tol::iterative(1e-5);
        let [glo, ghi, cglo, cghi] = r_bounds(&inf, EMethod::Gaussian, &syn);
        let [llo, lhi, cllo, clhi] = r_bounds(&inf, EMethod::Ls, &syn);
        let [qlo, qhi, cqlo, cqhi] = r_bounds(&inf, EMethod::Qreg, &syn);
        let sigma: Vec<f64> = inf.sigma.t().iter().copied().collect();
        Check::new(&case)
            .scalar("rho", inf.rho, Tol::new(1e-12, 1e-12))
            .vector("sigma", &sigma, closed)
            .vector("u_mean", inf.u_mean.as_slice().unwrap(), closed)
            .vector(
                "y_post_fit",
                syn.as_slice().unwrap(),
                Tol::new(1e-12, 1e-12),
            )
            .vector("insample_tight_lb", &inf.insample_lower, tight)
            .vector("insample_tight_ub", &inf.insample_upper, tight)
            .vector("insample_lb", &inf.insample_lower, ecos)
            .vector("insample_ub", &inf.insample_upper, ecos)
            .vector("e_gaussian_lb", &glo, out)
            .vector("e_gaussian_ub", &ghi, out)
            .vector("e_ls_lb", &llo, out)
            .vector("e_ls_ub", &lhi, out)
            .vector("e_qreg_lb", &qlo, fn_tol)
            .vector("e_qreg_ub", &qhi, fn_tol)
            .vector("ci_gaussian_lb", &cglo, ecos)
            .vector("ci_gaussian_ub", &cghi, ecos)
            .vector("ci_ls_lb", &cllo, ecos)
            .vector("ci_ls_ub", &clhi, ecos)
            .vector("ci_qreg_lb", &cqlo, ecos)
            .vector("ci_qreg_ub", &cqhi, ecos)
            .finish();
        assert_eq!(inf.n_failed, 0);

        // run_scpi end to end on R's draws, with its exact simplex weights (scest's ECOS weights are
        // about 1e-5 from them): the intervals within 1e-3
        let z = r_draws(&case);
        let r = run_scpi_with(
            &t.view(),
            &x.view(),
            14,
            SCPIConfig::default(),
            Draws::Given(z.view()),
            0,
        )
        .unwrap();
        let g = method(&r, EMethod::Gaussian);
        Check::new(&case)
            .named(
                "w",
                &DONORS,
                r.weights.as_slice().unwrap(),
                Tol::new(1e-4, 2e-5),
            )
            .vector("ci_gaussian_lb", &g.synthetic_lower, Tol::iterative(1e-3))
            .vector("ci_gaussian_ub", &g.synthetic_upper, Tol::iterative(1e-3))
            .finish();
        assert_eq!(r.headline_method, EMethod::Gaussian);
        for i in 0..6 {
            assert_eq!(r.ci_lower[i], g.lower[i]);
            assert_eq!(r.ci_upper[i], g.upper[i]);
            // effect interval = actual minus the synthetic interval
            assert!((g.lower[i] - (t[14 + i] - g.synthetic_upper[i])).abs() < 1e-12);
            assert!((g.upper[i] - (t[14 + i] - g.synthetic_lower[i])).abs() < 1e-12);
        }
    }

    /// U6: `alpha = Some(0.05)` sets u_alpha = e_alpha = 0.025 (scpi run at 0.025 each), and
    /// `u_missp = false` drops the pseudo-residual mean. Tolerances as in
    /// test_validate_scpi_default_intervals.
    #[test]
    fn test_validate_scpi_alpha_alias_and_no_misspecification() {
        let (case, t, x) = scpi_case("scpi_default_intervals");
        let tight = Tol::iterative(1e-7);
        let ecos = Tol::iterative(1e-4);
        let out = Tol::iterative(1e-6);
        let fn_tol = Tol::iterative(1e-5);
        let (ia, syn) = r_inference(&case, &t, &x, 0.025, 0.025, true);
        let (im, _) = r_inference(&case, &t, &x, 0.05, 0.05, false);
        let [glo, ghi, cglo, cghi] = r_bounds(&ia, EMethod::Gaussian, &syn);
        let [llo, lhi, _, _] = r_bounds(&ia, EMethod::Ls, &syn);
        let [qlo, qhi, _, _] = r_bounds(&ia, EMethod::Qreg, &syn);
        let sigma: Vec<f64> = im.sigma.t().iter().copied().collect();
        Check::new(&case)
            .vector("a025_insample_tight_lb", &ia.insample_lower, tight)
            .vector("a025_insample_tight_ub", &ia.insample_upper, tight)
            .vector("a025_e_gaussian_lb", &glo, out)
            .vector("a025_e_gaussian_ub", &ghi, out)
            .vector("a025_e_ls_lb", &llo, out)
            .vector("a025_e_ls_ub", &lhi, out)
            .vector("a025_e_qreg_lb", &qlo, fn_tol)
            .vector("a025_e_qreg_ub", &qhi, fn_tol)
            .vector("a025_ci_gaussian_lb", &cglo, ecos)
            .vector("a025_ci_gaussian_ub", &cghi, ecos)
            .vector("nomissp_sigma", &sigma, Tol::iterative(1e-8))
            .vector("nomissp_insample_tight_lb", &im.insample_lower, tight)
            .vector("nomissp_insample_tight_ub", &im.insample_upper, tight)
            .vector("nomissp_insample_lb", &im.insample_lower, ecos)
            .vector("nomissp_insample_ub", &im.insample_upper, ecos)
            .finish();
        // the alias reaches run_scpi
        let (_, t2, x2) = scpi_case("scpi_quantities");
        let cfg = SCPIConfig {
            alpha: Some(0.05),
            ..Default::default()
        };
        let r = run_scpi(&t2.view(), &x2.view(), 14, cfg).unwrap();
        assert_eq!((r.u_alpha, r.e_alpha), (0.025, 0.025));
    }

    /// CAUS-32: the interval width follows each post period's donor path (before: the same width,
    /// 1.5704, in every period).
    #[test]
    fn test_scpi_interval_width_varies_over_post_periods() {
        let (_, t, x) = scpi_case("scpi_quantities");
        let r = run_scpi(&t.view(), &x.view(), 14, SCPIConfig::default()).unwrap();
        let widths: Vec<f64> = r
            .prediction_intervals
            .iter()
            .map(|p| p.upper - p.lower)
            .collect();
        let (lo, hi) = widths
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(a, b), w| (a.min(*w), b.max(*w)));
        assert!(hi - lo > 0.05 * hi, "{widths:?}");
        for m in &r.method_intervals {
            for i in 0..widths.len() {
                assert!(m.lower[i] < m.upper[i], "{:?} {i}", m.method);
            }
        }
    }

    /// The seeded path is reproducible and a single e_method reports only that method.
    #[test]
    fn test_scpi_seed_and_single_e_method() {
        let (_, t, x) = scpi_case("scpi_quantities");
        let run = |cfg: SCPIConfig| run_scpi(&t.view(), &x.view(), 14, cfg).unwrap();
        let a = run(SCPIConfig::default());
        let b = run(SCPIConfig {
            seed: Some(SCPI_DEFAULT_SEED),
            ..Default::default()
        });
        assert_eq!(a.insample_lower, b.insample_lower);
        assert_eq!(a.seed, SCPI_DEFAULT_SEED);
        let c = run(SCPIConfig {
            seed: Some(7),
            ..Default::default()
        });
        assert_ne!(a.insample_lower, c.insample_lower);
        let q = run(SCPIConfig {
            e_method: EMethod::Qreg,
            ..Default::default()
        });
        assert_eq!(q.headline_method, EMethod::Qreg);
        assert_eq!(q.method_intervals.len(), 1);
        assert_eq!(q.method_intervals[0].lower, method(&a, EMethod::Qreg).lower);
        assert_eq!(a.method_intervals.len(), 3);
    }

    /// The deprecated variance_method maps onto e_method (gaussian -> gaussian, subgaussian ->
    /// all, the CV methods -> all with a warning naming e_method).
    #[test]
    fn test_scpi_variance_method_alias() {
        let (_, t, x) = scpi_case("scpi_quantities");
        let run = |vm: VarianceMethod| {
            let cfg = SCPIConfig {
                variance_method: Some(vm),
                ..Default::default()
            };
            run_scpi(&t.view(), &x.view(), 14, cfg).unwrap()
        };
        assert_eq!(run(VarianceMethod::Gaussian).method_intervals.len(), 1);
        assert_eq!(run(VarianceMethod::Subgaussian).method_intervals.len(), 3);
        for vm in [VarianceMethod::LooCv, VarianceMethod::KFoldCv] {
            let r = run(vm);
            assert_eq!(r.method_intervals.len(), 3);
            assert!(
                r.warnings.iter().any(|w| w.contains("e_method")),
                "{:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn test_scpi_rejects_invalid_alpha() {
        let (_, t, x) = scpi_case("scpi_quantities");
        for alpha in [0.0, 1.0, 1.5, -0.3, f64::NAN] {
            let cfg = SCPIConfig {
                alpha: Some(alpha),
                ..Default::default()
            };
            let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, cfg));
            assert!(m.contains("alpha"), "{alpha}: {m}");
            let cfg = SCPIConfig {
                u_alpha: alpha,
                ..Default::default()
            };
            let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, cfg));
            assert!(m.contains("u_alpha"), "{alpha}: {m}");
            let cfg = SCPIConfig {
                e_alpha: alpha,
                ..Default::default()
            };
            let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, cfg));
            assert!(m.contains("e_alpha"), "{alpha}: {m}");
        }
    }

    /// Lasso and ridge intervals (not R-pinned): finite, ordered, around the prediction.
    #[test]
    fn test_scpi_penalised_intervals_are_ordered() {
        let (_, t, x) = scpi_case("scpi_quantities");
        for constraint in [
            SCPIConstraint::Lasso { lambda: 0.1 },
            SCPIConstraint::Ridge { lambda: 0.5 },
        ] {
            let cfg = SCPIConfig {
                constraint: constraint.clone(),
                ..Default::default()
            };
            let r = run_scpi(&t.view(), &x.view(), 14, cfg).unwrap();
            assert_eq!(r.n_failed_sims, 0, "{constraint:?}");
            for i in 0..6 {
                assert!(r.insample_lower[i] <= 1e-9 && r.insample_upper[i] >= -1e-9);
                assert!(r.ci_lower[i].is_finite() && r.ci_lower[i] < r.ci_upper[i]);
            }
        }
    }

    /// The interior-point QCQP on problems with closed-form optima.
    #[test]
    fn test_qcqp_closed_forms() {
        // min p'x s.t. x'x - 2 g'x <= 0 (the ball around g of radius |g|): p'g - |p| |g|
        let g = array![0.3, -0.2, 0.5];
        let p = array![1.0, 2.0, -0.5];
        let prob = Qcqp {
            n: 3,
            a_eq: Array2::zeros((0, 3)),
            b_eq: Array1::zeros(0),
            g: Array2::zeros((0, 3)),
            h: Array1::zeros(0),
            quads: vec![(Array2::eye(3), &g * -2.0, 0.0)],
        };
        let want = p.dot(&g) - p.dot(&p).sqrt() * g.dot(&g).sqrt();
        let got = prob.minimize(p.view()).unwrap().unwrap();
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        // the same ball with sum(x) = 0 and x >= -0.1: the LP on the polytope when it is inside
        let prob2 = Qcqp {
            n: 3,
            a_eq: Array2::ones((1, 3)),
            b_eq: Array1::zeros(1),
            g: -Array2::eye(3),
            h: Array1::from_elem(3, 0.1),
            quads: vec![(Array2::eye(3) * 1e-3, &g * -2.0e-3, 0.0)],
        };
        // without the quadratic: put 0.2 on the cheapest coordinate, -0.1 on the others
        let got2 = prob2.minimize(p.view()).unwrap().unwrap();
        let lp = -0.5 * 0.2 + (1.0 + 2.0) * -0.1;
        let x = array![-0.1, -0.1, 0.2];
        assert!(x.dot(&x) - 2.0 * g.dot(&x) <= 0.0);
        assert!((got2 - lp).abs() < 1e-9, "{got2} vs {lp}");
    }

    fn ball(g: &Array1<f64>) -> Qcqp {
        let n = g.len();
        Qcqp {
            n,
            a_eq: Array2::zeros((0, n)),
            b_eq: Array1::zeros(0),
            g: Array2::zeros((0, n)),
            h: Array1::zeros(0),
            quads: vec![(Array2::eye(n), g * -2.0, 0.0)],
        }
    }

    /// A last centring that ends uncentred is a failed solve, not an optimum: with no Newton
    /// step the barrier method reports no value; with one step per centring, a value it reports
    /// is within the certified gap of the optimum.
    #[test]
    fn test_qcqp_uncentred_last_step_is_a_failure() {
        let g = array![0.3, -0.2, 0.5];
        let p = array![1.0, 2.0, -0.5];
        let sc = ball(&g).scaled();
        let start = sc.start().unwrap().unwrap();
        assert!(start.interior);
        let want = p.dot(&g) - p.dot(&p).sqrt() * g.dot(&g).sqrt();
        let got = sc.minimize_from(p.view(), &start).unwrap().unwrap();
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        // NEWTON_MAX is restored even when a call below panics
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                NEWTON_MAX.with(|c| c.set(QCQP_NEWTON_MAX));
            }
        }
        let with_steps = |k: usize| {
            let _restore = Restore;
            NEWTON_MAX.with(|c| c.set(k));
            sc.minimize_from(p.view(), &start).unwrap()
        };
        assert_eq!(with_steps(0), None);
        // One step per centring on this ball does not centre the last centring: a failure,
        // not an uncertified value.
        assert_eq!(with_steps(1), None);
        // A centring that runs out of steps reports the decrement of the point it returns.
        let c = p.clone();
        let _restore = Restore;
        NEWTON_MAX.with(|cell| cell.set(1));
        let (x1, d1) = sc.center(&c, start.x.clone(), 1.0, None).unwrap();
        let (_, at_x1) = sc.newton_step(&c, &x1, 1.0).unwrap();
        assert_eq!(d1, at_x1);
        let (_, at_x0) = sc.newton_step(&c, &start.x, 1.0).unwrap();
        assert_ne!(d1, at_x0);
    }

    /// The phase I and every barrier centring check the budget.
    #[test]
    fn test_qcqp_stops_when_cancelled() {
        let g = array![0.3, -0.2, 0.5];
        let p = array![1.0, 2.0, -0.5];
        let mut prob = ball(&g);
        // a start outside the set, so that the phase I iterates
        prob.quads[0].2 = 0.01;
        let sc = prob.scaled();
        assert!(matches!(
            cancelled(|| sc.start()),
            Err(EconError::Cancelled(_))
        ));
        let start = sc.start().unwrap().unwrap();
        assert!(matches!(
            cancelled(|| sc.minimize_from(p.view(), &start)),
            Err(EconError::Cancelled(_))
        ));
    }

    /// An explicit e_method wins over the deprecated variance_method, with a warning naming it.
    #[test]
    fn test_scpi_e_method_wins_over_variance_method() {
        let (t, d) = generate_test_data();
        let cfg = SCPIConfig {
            e_method: EMethod::Qreg,
            variance_method: Some(VarianceMethod::Gaussian),
            sims: 10,
            ..Default::default()
        };
        let r = run_scpi(&t.view(), &d.view(), 7, cfg).unwrap();
        assert_eq!(r.headline_method, EMethod::Qreg);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("variance_method 'gaussian' is ignored")),
            "{:?}",
            r.warnings
        );
    }

    /// scpi warns when its regularize.check.lb raises a rho below 0.001; so does this.
    #[test]
    fn test_scpi_warns_when_rho_is_raised() {
        let n = 20;
        let d = Array2::from_shape_fn((n, 3), |(t, j)| {
            let tf = t as f64;
            10.0 + tf * (1.0 + 0.3 * j as f64) + ((tf * (j as f64 + 1.3)).sin())
        });
        // an almost exact convex combination of the donors: tiny residuals, tiny rho
        let treated = Array1::from_shape_fn(n, |t| {
            0.6 * d[[t, 0]] + 0.4 * d[[t, 1]] + 1e-7 * ((t as f64) * 2.7).cos()
        });
        let cfg = SCPIConfig {
            sims: 10,
            ..Default::default()
        };
        let r = run_scpi(&treated.view(), &d.view(), 15, cfg).unwrap();
        assert!(
            r.warnings.iter().any(|w| w.contains("rho was")
                && w.contains("below 0.001")
                && w.contains("collinearity")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_e_method_parses_aliases_and_rejects_unknown() {
        for (s, v) in [
            ("all", EMethod::All),
            ("gaussian", EMethod::Gaussian),
            ("Sub-Gaussian", EMethod::Gaussian),
            ("ls", EMethod::Ls),
            ("location_scale", EMethod::Ls),
            ("qreg", EMethod::Qreg),
            ("quantile", EMethod::Qreg),
        ] {
            assert_eq!(s.parse::<EMethod>().unwrap(), v, "{s}");
        }
        EMethod::choices().validate().unwrap();
        let m = "bootstrap".parse::<EMethod>().unwrap_err().to_string();
        for name in ["all", "gaussian", "ls", "qreg"] {
            assert!(m.contains(name), "{m}");
        }
        assert_eq!(EMethod::default(), EMethod::All);
    }

    #[test]
    fn test_scpi_rejects_invalid_lambda() {
        let (_, t, x) = scpi_case("scpi_quantities");
        for c in [
            SCPIConstraint::Ridge { lambda: 0.0 },
            SCPIConstraint::Ridge { lambda: -1.0 },
            SCPIConstraint::Lasso { lambda: -0.1 },
            SCPIConstraint::LassoSimplex { lambda: f64::NAN },
        ] {
            let cfg = SCPIConfig {
                constraint: c.clone(),
                ..Default::default()
            };
            let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, cfg));
            assert!(m.contains("lambda"), "{c:?}: {m}");
        }
    }

    #[test]
    fn test_scpi_rejects_non_finite_outcomes() {
        let (_, mut t, mut x) = scpi_case("scpi_quantities");
        x[[3, 2]] = f64::NAN;
        let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, SCPIConfig::default()));
        assert!(
            m.contains("donors") && m.contains("period 3, donor 2"),
            "{m}"
        );
        x[[3, 2]] = 1.0;
        t[5] = f64::INFINITY;
        let m = spec_msg(run_scpi(&t.view(), &x.view(), 14, SCPIConfig::default()));
        assert!(m.contains("treated") && m.contains("period 5"), "{m}");
    }

    #[test]
    fn test_variance_method_parses_aliases_and_rejects_unknown() {
        for (s, v) in [
            ("subgaussian", VarianceMethod::Subgaussian),
            ("Sub-Gaussian", VarianceMethod::Subgaussian),
            ("gaussian", VarianceMethod::Gaussian),
            ("loo", VarianceMethod::LooCv),
            ("LOO_CV", VarianceMethod::LooCv),
            ("k-fold", VarianceMethod::KFoldCv),
            ("kfold_cv", VarianceMethod::KFoldCv),
        ] {
            assert_eq!(s.parse::<VarianceMethod>().unwrap(), v, "{s}");
        }
        VarianceMethod::choices().validate().unwrap();
        let m = "bootstrap"
            .parse::<VarianceMethod>()
            .unwrap_err()
            .to_string();
        for name in ["subgaussian", "gaussian", "loo_cv", "kfold_cv"] {
            assert!(m.contains(name), "{m}");
        }
        assert_eq!(VarianceMethod::KFoldCv.to_string(), "kfold_cv");
    }

    #[test]
    fn test_scpi_constraint_from_options() {
        assert!(matches!(
            SCPIConstraint::from_options(None, None),
            Ok(SCPIConstraint::Simplex)
        ));
        assert!(matches!(
            SCPIConstraint::from_options(Some("ridge"), Some(0.5)),
            Ok(SCPIConstraint::Ridge { lambda }) if lambda == 0.5
        ));
        assert!(matches!(
            SCPIConstraint::from_options(Some("lasso"), None),
            Ok(SCPIConstraint::Lasso { lambda }) if lambda == 0.1
        ));
        assert!(matches!(
            SCPIConstraint::from_options(Some("Lasso-Simplex"), Some(0.2)),
            Ok(SCPIConstraint::LassoSimplex { lambda }) if lambda == 0.2
        ));
        assert!(matches!(
            SCPIConstraint::from_options(Some("elastic"), None),
            Err(EconError::InvalidSpecification { .. })
        ));
        ConstraintName::choices().validate().unwrap();
    }
}
