//! Double/Debiased Machine Learning (DML) for causal effect estimation.
//!
//! DML combines Neyman-orthogonal scores with cross-fitting: each nuisance function is
//! predicted for the observations of one fold by a learner trained on the other folds, and
//! the target parameter solves the empirical score built from those out-of-fold
//! predictions (Chernozhukov et al. 2018). The models and scores are DoubleML's
//! (Bach et al. 2022, 2024).
//!
//! # Models and scores
//!
//! `psi = psi_a theta + psi_b`; theta solves `mean(psi) = 0` and `J = mean(psi_a)`.
//!
//! | Model | Nuisances | Score | Estimand |
//! |---|---|---|---|
//! | PLR ([`run_double_ml()`]) | `l = E[Y\|X]`, `m = E[D\|X]` | `(Y - l - theta (D - m)) (D - m)` | theta (partialling out) |
//! | IRM ([`run_double_ml()`]) | `g0 = E[Y\|D=0,X]`, `g1 = E[Y\|D=1,X]`, `m = P(D=1\|X)` | `g1 - g0 + D (Y - g1)/m - (1 - D)(Y - g0)/(1 - m) - theta` | ATE |
//! | PLIV ([`run_double_ml_iv()`]) | `l = E[Y\|X]`, `r = E[D\|X]`, `m = E[Z\|X]` | `(Y - l - theta (D - r)) (Z - m)` | theta (IV, partialling out) |
//! | IIVM ([`run_double_ml_iv()`]) | `g0`, `g1` = `E[Y\|Z=z,X]`, `r0`, `r1` = `P(D=1\|Z=z,X)`, `m = P(Z=1\|X)` | `psi_b = g1 - g0 + Z (Y - g1)/m - (1 - Z)(Y - g0)/(1 - m)`, `psi_a = -(r1 - r0 + Z (D - r1)/m - (1 - Z)(D - r0)/(1 - m))` | LATE |
//!
//! IRM's and IIVM's propensity `m` is bounded to `[trim, 1 - trim]` (DoubleML's
//! `trimming_threshold`, default 0.01). IRM and IIVM need a treatment (and IIVM an
//! instrument) coded exactly 0/1.
//!
//! # Learners
//!
//! Regression nuisances use `MLMethod`: OLS (OLS's solve decision: the normal equations on a
//! well-conditioned design, R's pivoted QR otherwise; aliased covariates are reported and
//! predict as 0, as `predict.lm`), lasso (glmnet coordinate descent at `lambda`,
//! standardized, `thresh` [`DML_GLMNET_THRESH`]) or ridge (the closed form of glmnet's ridge
//! objective at `lambda`, standardized; no iteration). At `lambda = 0` both are OLS. Binary
//! nuisances (IRM's `m`; IIVM's `r0`, `r1`, `m`) are logistic regressions; a binary nuisance
//! whose training outcome has no variation (IIVM's `r0` under one-sided noncompliance) is
//! predicted as that constant, with a note.
//!
//! # Folds, inference, diagnostics
//!
//! Folds are the user's labels (`DoubleMLConfig::folds`) or a ChaCha8 shuffle seeded by
//! `seed` ([`DML_DEFAULT_SEED`] when `None`). Scores are pooled over folds (`dml2`):
//! `se = sqrt(mean(psi^2) / J^2 / n)`, the p-value is `2 Phi(-|theta / se|)` and the 95% CI
//! `theta -/+ qnorm(0.975) se`. Every nuisance reports its RMSE and R² over its out-of-fold
//! predictions ([`NuisanceFit`]; g0, g1, r0, r1 over their own arm). `jacobian` is `-J`:
//! `E[(D - m)^2]` for PLR, 1 for IRM, `E[(D - r)(Z - m)]` for PLIV and the first stage
//! `-E[psi_a]` for IIVM.
//!
//! # Differences from DoubleML
//!
//! One repetition (`n_rep = 1`) and `dml2` only; the ATE and LATE scores only (no ATTE);
//! a fixed `lambda` with no learner tuning; PLR's `m` (and PLIV's `r`, `m`) always by the
//! regression learner, also for a 0/1 treatment; the constant-target rule above instead of
//! DoubleML's `subgroups` option; glmnet always standardized.
//!
//! # References
//!
//! - Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
//!   Robins, J. (2018). "Double/debiased machine learning for treatment and structural
//!   parameters." *The Econometrics Journal*, 21(1), C1-C68.
//!   https://doi.org/10.1111/ectj.12097
//! - Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). "DoubleML - An
//!   Object-Oriented Implementation of Double Machine Learning in Python." *Journal of
//!   Machine Learning Research*, 23(53), 1-6.
//! - Bach, P., Chernozhukov, V., Kurz, M. S., Spindler, M., & Klaassen, S. (2024).
//!   "DoubleML: An Object-Oriented Implementation of Double Machine Learning in R."
//!   *Journal of Statistical Software*, 108(3).
//! - Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization paths for
//!   generalized linear models via coordinate descent." *Journal of Statistical Software*,
//!   33(1), 1-22.
//!
//! # Example
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use p2a_core::econometrics::{DoubleMLConfig, MLMethod, run_double_ml};
//! let n = 40;
//! let x = Array2::from_shape_fn((n, 2), |(i, j)| (((i * (7 + 2 * j) + 3 * j) % 11) as f64) / 11.0);
//! let d = Array1::from_shape_fn(n, |i| x[[i, 0]] + (((i * 5) % 7) as f64) / 7.0);
//! let y = Array1::from_shape_fn(n, |i| 0.5 * d[i] + x[[i, 1]] + (((i * 3) % 5) as f64) / 5.0);
//! let mut config = DoubleMLConfig::default();
//! config.ml_method = MLMethod::Lasso;
//! let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();
//! assert!(result.theta.is_finite() && result.se > 0.0);
//! assert_eq!(result.nuisance.len(), 2);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::econometrics::glm::{GlmLink, bound, fit_propensity, plogis};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr, aliasing_message};
use crate::options::Choices;
use crate::regression::glmnet::{GlmnetConfig, glmnet, glmnet_predict};
use crate::regression::ols::normal_equations_inverse;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

// =============================================================================
// Configuration Types
// =============================================================================

/// DML model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DMLModelType {
    /// Partially Linear Regression Model (PLR)
    /// Y = theta * D + g(X) + epsilon
    #[default]
    PLR,
    /// Interactive Regression Model (IRM) for binary treatment
    /// Used for heterogeneous treatment effects with binary D
    IRM,
    /// Partially Linear IV Regression (PLIV): Y = theta * D + g(X) + zeta, E[zeta|Z,X] = 0,
    /// with one instrument Z ([`run_double_ml_iv()`])
    PLIV,
    /// Interactive IV Model (IIVM): the LATE of a binary D with a binary instrument Z
    /// ([`run_double_ml_iv()`])
    IIVM,
}

crate::option_enum!(
    DMLModelType,
    Choices {
        option: "model",
        values: &[
            (
                &[
                    "plr",
                    "partially_linear",
                    "partially_linear_regression",
                    "doublemlplr",
                ],
                DMLModelType::PLR,
            ),
            (
                &[
                    "irm",
                    "interactive",
                    "interactive_regression",
                    "doublemlirm",
                ],
                DMLModelType::IRM,
            ),
            (
                &["pliv", "partially_linear_iv", "doublemlpliv"],
                DMLModelType::PLIV,
            ),
            (
                &["iivm", "interactive_iv", "doublemliivm"],
                DMLModelType::IIVM,
            ),
        ],
    },
    no_display
);

impl fmt::Display for DMLModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DMLModelType::PLR => write!(f, "Partially Linear Regression (PLR)"),
            DMLModelType::IRM => write!(f, "Interactive Regression Model (IRM)"),
            DMLModelType::PLIV => write!(f, "Partially Linear IV Regression (PLIV)"),
            DMLModelType::IIVM => write!(f, "Interactive IV Model (IIVM)"),
        }
    }
}

/// Treatment variable type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TreatmentType {
    /// Continuous treatment variable
    #[default]
    Continuous,
    /// Binary treatment (0/1)
    Binary,
}

impl fmt::Display for TreatmentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreatmentType::Continuous => write!(f, "Continuous"),
            TreatmentType::Binary => write!(f, "Binary"),
        }
    }
}

/// Learner of the nuisance functions that are regressions (binary nuisances are fitted by
/// logistic regression whatever the method).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MLMethod {
    /// Ordinary Least Squares (default)
    #[default]
    OLS,
    /// Ridge regression at `DoubleMLConfig::lambda` (glmnet's objective, standardized)
    Ridge,
    /// Lasso regression at `DoubleMLConfig::lambda` (glmnet, standardized)
    Lasso,
}

crate::option_enum!(
    MLMethod,
    Choices {
        option: "ml_method",
        values: &[
            (
                &["ols", "lm", "linear", "regr.lm", "least_squares"],
                MLMethod::OLS,
            ),
            (&["lasso", "l1"], MLMethod::Lasso),
            (&["ridge", "l2"], MLMethod::Ridge),
        ],
    },
    no_display
);

impl fmt::Display for MLMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLMethod::OLS => write!(f, "OLS"),
            MLMethod::Ridge => write!(f, "Ridge"),
            MLMethod::Lasso => write!(f, "Lasso"),
        }
    }
}

impl MLMethod {
    /// The learner as the result prints it: "OLS", or "Lasso (glmnet, lambda = 0.05,
    /// standardized)" and the same for ridge.
    pub fn description(self, lambda: f64) -> String {
        match self {
            MLMethod::OLS => "OLS".to_string(),
            MLMethod::Lasso => format!("Lasso (glmnet, lambda = {lambda}, standardized)"),
            MLMethod::Ridge => format!("Ridge (glmnet, lambda = {lambda}, standardized)"),
        }
    }
}

impl DMLModelType {
    /// The parameter the model estimates.
    pub fn estimand(self) -> &'static str {
        match self {
            DMLModelType::PLR => "theta (partialling out)",
            DMLModelType::IRM => "ATE",
            DMLModelType::PLIV => "theta (IV, partialling out)",
            DMLModelType::IIVM => "LATE",
        }
    }

    /// Whether the model takes an instrument (PLIV, IIVM: [`run_double_ml_iv()`]).
    pub fn needs_instrument(self) -> bool {
        matches!(self, DMLModelType::PLIV | DMLModelType::IIVM)
    }
}

/// Seed of the fold assignment when `DoubleMLConfig::seed` is `None`.
pub const DML_DEFAULT_SEED: u64 = 42;
/// Default penalty of the lasso and ridge learners (glmnet's lambda).
pub const DML_DEFAULT_LAMBDA: f64 = 0.05;
/// glmnet's `thresh` for the lasso nuisance fits (glmnet's own default is 1e-7, which stops
/// coordinate descent about 3e-4 from the minimiser).
pub const DML_GLMNET_THRESH: f64 = 1e-14;

/// Configuration for Double Machine Learning estimation.
#[derive(Debug, Clone)]
pub struct DoubleMLConfig {
    /// Number of cross-fitting folds (default: 5), between 2 and n; ignored when `folds` is
    /// given
    pub n_folds: usize,
    /// DML model type
    pub model_type: DMLModelType,
    /// Declared treatment type, checked against d (`Binary` with a d that is not coded 0/1 is
    /// refused); the result reports the type detected from d
    pub treatment_type: TreatmentType,
    /// Learner of every nuisance function that is a regression
    pub ml_method: MLMethod,
    /// Whether to include an intercept in nuisance models
    pub intercept: bool,
    /// Seed of the random fold assignment (`None`: [`DML_DEFAULT_SEED`]); ignored with `folds`
    pub seed: Option<u64>,
    /// Propensity trimming: IRM and IIVM propensities are bounded to `[trim, 1 - trim]`
    /// (default 0.01, DoubleML's `trimming_threshold`); `0 < trim < 0.5`
    pub trim: f64,
    /// Penalty of the lasso and ridge learners (glmnet's lambda, default
    /// [`DML_DEFAULT_LAMBDA`]); finite and >= 0
    pub lambda: f64,
    /// User fold labels, one per observation (any values, at least two distinct); the folds
    /// are taken in label order. `None`: random folds from `seed`
    pub folds: Option<Vec<usize>>,
}

impl Default for DoubleMLConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            model_type: DMLModelType::PLR,
            treatment_type: TreatmentType::Continuous,
            ml_method: MLMethod::OLS,
            intercept: true,
            seed: None,
            trim: 0.01,
            lambda: DML_DEFAULT_LAMBDA,
            folds: None,
        }
    }
}

// =============================================================================
// Diagnostics Types
// =============================================================================

/// Diagnostics for nuisance model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuisanceDiagnostics {
    /// R-squared for outcome model l(X) = E[Y|X]
    pub outcome_r2: f64,
    /// R-squared for treatment model m(X) = E[D|X]
    pub treatment_r2: f64,
    /// Root mean squared error for outcome model
    pub outcome_rmse: f64,
    /// Root mean squared error for treatment model
    pub treatment_rmse: f64,
    /// Mean of residualized treatment (D - m(X)), should be near 0
    pub mean_residual_treatment: f64,
    /// Variance of residualized treatment (should be > 0)
    pub var_residual_treatment: f64,
}

// =============================================================================
// Result Types
// =============================================================================

/// One cross-fitted nuisance function and its out-of-fold accuracy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NuisanceFit {
    /// DoubleML's learner name without "ml_": "l", "m", "r", "g0", "g1", "r0", "r1"
    pub name: String,
    /// The function estimated: "E[Y|X]", "E[D|X]", "P(D=1|X)", "E[Y|D=0,X]", ...
    pub target: String,
    /// "OLS", "lasso (glmnet, lambda = 0.05)", "ridge (glmnet, lambda = 0.05)" or
    /// "logistic regression"
    pub learner: String,
    /// Observations evaluated: all n, or the own arm (g0, g1, r0, r1)
    pub n_eval: usize,
    /// sqrt(mean((target - prediction)^2)) over the out-of-fold predictions evaluated
    pub rmse: f64,
    /// 1 - SSR / SST over the same observations (NaN when SST = 0)
    pub r2: f64,
}

/// Result from Double Machine Learning estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleMLResult {
    /// Estimated treatment effect (theta)
    pub theta: f64,
    /// Standard error of theta (from influence function)
    pub se: f64,
    /// t-statistic (theta / se)
    pub t_stat: f64,
    /// Two-sided p-value for H0: theta = 0
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,

    /// Orthogonal scores psi for each observation
    /// psi_i = (Y_i - l_hat(X_i) - theta*(D_i - m_hat(X_i))) * (D_i - m_hat(X_i))
    #[serde(skip)]
    pub scores: Vec<f64>,

    /// Nuisance model diagnostics
    pub nuisance_diagnostics: NuisanceDiagnostics,

    /// Number of observations
    pub n_obs: usize,
    /// Number of folds used in cross-fitting
    pub n_folds: usize,
    /// Model type used
    pub model_type: DMLModelType,
    /// Treatment type
    pub treatment_type: TreatmentType,
    /// ML method used for nuisance estimation
    pub ml_method: MLMethod,

    /// Jacobian J = E[(D - m(X))^2] used for variance calculation
    pub jacobian: f64,

    /// Warnings generated during estimation
    pub warnings: Vec<String>,

    /// The cross-fitted nuisance functions, in model order (PLR: l, m; IRM: g0, g1, m;
    /// PLIV: l, r, m; IIVM: g0, g1, r0, r1, m)
    #[serde(default)]
    pub nuisance: Vec<NuisanceFit>,
    /// The penalty of the lasso and ridge learners (`None` for OLS)
    #[serde(default)]
    pub lambda: Option<f64>,
    /// The propensity trimming threshold (IRM, IIVM)
    #[serde(default)]
    pub trim: Option<f64>,
    /// Range of the trimmed propensity scores (IRM: P(D=1|X); IIVM: P(Z=1|X))
    #[serde(default)]
    pub propensity_range: Option<[f64; 2]>,
    /// Observations per fold, in label order
    #[serde(default)]
    pub fold_sizes: Vec<usize>,
    /// Whether the folds were supplied by the user
    #[serde(default)]
    pub user_folds: bool,
    /// The seed of the fold assignment (`None` with user folds)
    #[serde(default)]
    pub seed: Option<u64>,
}

impl ResultNotes for DoubleMLResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for DoubleMLResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Double/Debiased Machine Learning Results")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Model Type: {}", self.model_type)?;
        writeln!(f, "Estimand:   {}", self.model_type.estimand())?;
        writeln!(
            f,
            "ML Method:  {}",
            self.ml_method
                .description(self.lambda.unwrap_or(DML_DEFAULT_LAMBDA))
        )?;
        writeln!(f, "Treatment:  {}", self.treatment_type)?;
        match self.seed {
            Some(seed) if !self.user_folds => {
                writeln!(f, "Folds:      {} (random, seed {seed})", self.n_folds)?
            }
            _ => writeln!(f, "Folds:      {} (user-supplied)", self.n_folds)?,
        }
        if let Some(trim) = self.trim {
            writeln!(
                f,
                "Trimming:   propensities bounded to [{trim}, {}]",
                1.0 - trim
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Treatment Effect:")?;
        writeln!(f, "  theta:      {:>12.4}", self.theta)?;
        writeln!(f, "  Std. Error: {:>12.4}", self.se)?;
        writeln!(f, "  t-stat:     {:>12.2}", self.t_stat)?;
        writeln!(
            f,
            "  p-value:    {:>12.4}{}",
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI:     [{:.4}, {:.4}]",
            self.ci_lower, self.ci_upper
        )?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(f, "Nuisance models (out-of-fold predictions):")?;
        for nf in &self.nuisance {
            writeln!(
                f,
                "  {:<3} {:<14} {:<32} n = {:<5} RMSE {:.4}  R² {:.4}",
                nf.name, nf.target, nf.learner, nf.n_eval, nf.rmse, nf.r2
            )?;
        }
        if let Some([lo, hi]) = self.propensity_range {
            writeln!(f, "  Propensity range: [{lo:.4}, {hi:.4}]")?;
        }
        match self.model_type {
            DMLModelType::PLR => {
                writeln!(f, "  Score Jacobian E[(D - m)^2]: {:.4}", self.jacobian)?
            }
            DMLModelType::IRM => writeln!(f, "  Score Jacobian: 1 (ATE score, psi_a = -1)")?,
            DMLModelType::PLIV => writeln!(
                f,
                "  Score Jacobian E[(D - r)(Z - m)]: {:.4}",
                self.jacobian
            )?,
            DMLModelType::IIVM => writeln!(
                f,
                "  Score Jacobian -E[psi_a] (first stage): {:.4}",
                self.jacobian
            )?,
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        write!(f, "{}", notes_text(self))
    }
}

// =============================================================================
// Main DML Functions
// =============================================================================

/// Run Double Machine Learning estimation: PLR or IRM.
///
/// Cross-fits the model's nuisance functions on K folds (each predicted for the
/// observations of one fold by a learner trained on the others), then solves the
/// Neyman-orthogonal score for theta and estimates its standard error from the scores
/// (`dml2`). See the module documentation for the models, scores and learners.
///
/// # Arguments
/// * `y` - Outcome variable (n x 1)
/// * `d` - Treatment variable (n x 1), continuous or 0/1 (IRM: 0/1)
/// * `x` - Covariates (n x p)
/// * `config` - DML configuration options
///
/// # Errors
/// `InvalidSpecification` for inconsistent lengths, non-finite data, a constant
/// treatment, out-of-range `trim`, `lambda` or folds; `InsufficientData` when a training
/// set is too small for its learner; `SingularMatrix` when the score's Jacobian is
/// (numerically) zero.
pub fn run_double_ml(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: DoubleMLConfig,
) -> EconResult<DoubleMLResult> {
    if config.model_type.needs_instrument() {
        return Err(spec(format!(
            "model {:?} needs an instrument: call run_double_ml_iv",
            config.model_type
        )));
    }
    let treatment_type = validate_inputs(y, d, None, x, &config)?;
    let folds = make_folds(y.len(), &config)?;
    let mut warnings = Vec::new();
    match config.model_type {
        DMLModelType::IRM => run_irm(y, d, x, &config, &folds, &mut warnings),
        _ => run_plr(y, d, x, &config, &folds, &mut warnings),
    }
    .map(|r| with_folds_info(r, treatment_type, &folds))
}

/// Run Double Machine Learning estimation with an instrument: PLIV or IIVM.
///
/// PLIV (partially linear IV, partialling-out score with one instrument) estimates theta in
/// `Y = theta D + g(X) + zeta`, `E[zeta | Z, X] = 0`; IIVM (interactive IV, binary D and Z)
/// estimates the local average treatment effect with DoubleML's LATE score. Same learners,
/// folds and inference as [`run_double_ml`]; see the module documentation.
///
/// # Errors
/// As [`run_double_ml`]; in addition PLR and IRM are refused (they take no instrument), a
/// constant PLIV instrument, and IIVM data whose treatment or instrument is not coded 0/1.
pub fn run_double_ml_iv(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: DoubleMLConfig,
) -> EconResult<DoubleMLResult> {
    if !config.model_type.needs_instrument() {
        return Err(spec(format!(
            "model {:?} does not use an instrument: call run_double_ml",
            config.model_type
        )));
    }
    let treatment_type = validate_inputs(y, d, Some(z), x, &config)?;
    let folds = make_folds(y.len(), &config)?;
    let mut warnings = Vec::new();
    match config.model_type {
        DMLModelType::PLIV => run_pliv(y, d, z, x, &config, &folds, &mut warnings),
        _ => run_iivm(y, d, z, x, &config, &folds, &mut warnings),
    }
    .map(|r| with_folds_info(r, treatment_type, &folds))
}

/// The fold fields and the detected treatment type, which the entry points fill in.
fn with_folds_info(
    mut r: DoubleMLResult,
    treatment_type: TreatmentType,
    folds: &Folds,
) -> DoubleMLResult {
    r.treatment_type = treatment_type;
    r.fold_sizes = folds.sizes.clone();
    r.user_folds = folds.seed.is_none();
    r.seed = folds.seed;
    r.n_folds = folds.labels.len();
    r
}

/// Partially linear IV, partialling-out score with one instrument:
/// `psi = (Y - l - theta (D - r)) (Z - m)`, `l = E[Y|X]`, `r = E[D|X]`, `m = E[Z|X]`.
fn run_pliv(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let n = y.len() as f64;
    let (zlo, zhi) = z
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if zlo == zhi {
        return Err(spec("instrument z is constant".to_string()));
    }
    let ml = Learner::Regression(config.ml_method);
    let preds = cross_fit(
        x,
        folds,
        None,
        ml,
        &[
            Nuisance::new("l", "E[Y|X]", y.view()),
            Nuisance::new("r", "E[D|X]", d.view()),
            Nuisance::new("m", "E[Z|X]", z.view()),
        ],
        config,
        warnings,
    )?;
    let (l, r, m) = (&preds[0], &preds[1], &preds[2]);
    let ry = y - l;
    let rd = d - r;
    let rz = z - m;
    let jacobian = rd.dot(&rz) / n;
    if jacobian.is_nan() || jacobian.abs() <= 1e-12 * (var_pop(d) * var_pop(z)).sqrt() {
        return Err(EconError::SingularMatrix {
            context: format!("DoubleML PLIV: E[(D - r)(Z - m)] = {jacobian:e} out of fold"),
            suggestion: "the residualized instrument is uncorrelated with the residualized \
                         treatment; theta is not identified"
                .to_string(),
        });
    }
    let theta = rz.dot(&ry) / rz.dot(&rd);
    let scores: Array1<f64> = (&ry - &(&rd * theta)) * &rz;
    let se = (scores.dot(&scores) / n / (jacobian * jacobian) / n).sqrt();

    let label = ml.label(config.lambda);
    let fit_l = nuisance_fit("l", "E[Y|X]", &label, y, l, None);
    let fit_r = nuisance_fit("r", "E[D|X]", &label, d, r, None);
    let fit_m = nuisance_fit("m", "E[Z|X]", &label, z, m, None);
    let mean_rd = rd.sum() / n;
    let var_rd = rd.dot(&rd) / n;
    let diagnostics = NuisanceDiagnostics {
        outcome_r2: fit_l.r2,
        treatment_r2: fit_r.r2,
        outcome_rmse: fit_l.rmse,
        treatment_rmse: fit_r.rmse,
        mean_residual_treatment: mean_rd,
        var_residual_treatment: var_rd,
    };
    residual_warnings(warnings, "r", var_rd, d, mean_rd);
    assemble(
        DMLModelType::PLIV,
        config,
        theta,
        se,
        scores.to_vec(),
        diagnostics,
        jacobian,
        vec![fit_l, fit_r, fit_m],
        None,
        warnings,
    )
}

/// Interactive IV model (binary D and Z), DoubleML's LATE score:
/// `psi_b = g1 - g0 + Z (Y - g1) / m - (1 - Z)(Y - g0) / (1 - m)`,
/// `psi_a = -(r1 - r0 + Z (D - r1) / m - (1 - Z)(D - r0) / (1 - m))`,
/// `theta = -mean(psi_b) / mean(psi_a)`, `psi = psi_a theta + psi_b`; `g_z = E[Y|Z=z,X]` by
/// `ml_method`, `r_z = P(D=1|Z=z,X)` and `m = P(Z=1|X)` by logistic regression, `m`
/// bounded to `[trim, 1 - trim]`.
fn run_iivm(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let nobs = y.len();
    let n = nobs as f64;
    if let Some(at) = first_non_binary(d) {
        return Err(spec(format!(
            "IIVM needs a binary treatment coded 0/1 ({at})"
        )));
    }
    if let Some(at) = first_non_binary(z) {
        return Err(spec(format!(
            "IIVM needs a binary instrument coded 0/1 ({at})"
        )));
    }
    let z1: Vec<bool> = z.iter().map(|&v| v == 1.0).collect();
    let z0: Vec<bool> = z1.iter().map(|t| !t).collect();
    let n1 = z1.iter().filter(|&&t| t).count();
    if n1.min(nobs - n1) < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n1.min(nobs - n1),
            context: "IIVM requires sufficient observations with each instrument value".to_string(),
        });
    }
    let ml = Learner::Regression(config.ml_method);
    let mut one = |subset: Option<&[bool]>, learner, nu: Nuisance<'_>| {
        cross_fit(x, folds, subset, learner, &[nu], config, warnings).map(|mut v| v.remove(0))
    };
    let g0 = one(Some(&z0), ml, Nuisance::new("g0", "E[Y|Z=0,X]", y.view()))?;
    let g1 = one(Some(&z1), ml, Nuisance::new("g1", "E[Y|Z=1,X]", y.view()))?;
    let lg = Learner::Logistic;
    let r0 = one(Some(&z0), lg, Nuisance::new("r0", "P(D=1|Z=0,X)", d.view()))?;
    let r1 = one(Some(&z1), lg, Nuisance::new("r1", "P(D=1|Z=1,X)", d.view()))?;
    let m = one(None, lg, Nuisance::new("m", "P(Z=1|X)", z.view()))?
        .mapv(|v| bound(v, config.trim, 1.0 - config.trim));

    let psi_b = Array1::from_shape_fn(nobs, |i| {
        g1[i] - g0[i] + z[i] * (y[i] - g1[i]) / m[i] - (1.0 - z[i]) * (y[i] - g0[i]) / (1.0 - m[i])
    });
    let psi_a = Array1::from_shape_fn(nobs, |i| {
        -(r1[i] - r0[i] + z[i] * (d[i] - r1[i]) / m[i]
            - (1.0 - z[i]) * (d[i] - r0[i]) / (1.0 - m[i]))
    });
    let j = psi_a.sum() / n;
    if j.is_nan() || j.abs() <= 1e-12 {
        return Err(EconError::SingularMatrix {
            context: format!("DoubleML IIVM: E[psi_a] = {j:e}"),
            suggestion: "the instrument does not move the treatment out of fold; the LATE is \
                         not identified"
                .to_string(),
        });
    }
    let theta = -(psi_b.sum() / n) / j;
    let scores = &psi_a * theta + &psi_b;
    let se = (scores.dot(&scores) / n / (j * j) / n).sqrt();

    let label = ml.label(config.lambda);
    let logistic = lg.label(config.lambda);
    let fit_g0 = nuisance_fit("g0", "E[Y|Z=0,X]", &label, y, &g0, Some(&z0));
    let fit_g1 = nuisance_fit("g1", "E[Y|Z=1,X]", &label, y, &g1, Some(&z1));
    let fit_r0 = nuisance_fit("r0", "P(D=1|Z=0,X)", &logistic, d, &r0, Some(&z0));
    let fit_r1 = nuisance_fit("r1", "P(D=1|Z=1,X)", &logistic, d, &r1, Some(&z1));
    let fit_m = nuisance_fit("m", "P(Z=1|X)", &logistic, z, &m, None);
    // the models of the observed instrument arm
    let g_obs = Array1::from_shape_fn(nobs, |i| if z1[i] { g1[i] } else { g0[i] });
    let r_obs = Array1::from_shape_fn(nobs, |i| if z1[i] { r1[i] } else { r0[i] });
    let fit_out = nuisance_fit("g", "E[Y|Z,X]", &label, y, &g_obs, None);
    let fit_tr = nuisance_fit("r", "P(D=1|Z,X)", &logistic, d, &r_obs, None);
    let rd = d - &r_obs;
    let diagnostics = NuisanceDiagnostics {
        outcome_r2: fit_out.r2,
        treatment_r2: fit_tr.r2,
        outcome_rmse: fit_out.rmse,
        treatment_rmse: fit_tr.rmse,
        mean_residual_treatment: rd.sum() / n,
        var_residual_treatment: rd.dot(&rd) / n,
    };
    propensity_warning(warnings, &m, config.trim);
    let range = min_max(&m);
    assemble(
        DMLModelType::IIVM,
        config,
        theta,
        se,
        scores.to_vec(),
        diagnostics,
        -j,
        vec![fit_g0, fit_g1, fit_r0, fit_r1, fit_m],
        Some(range),
        warnings,
    )
}

/// Partially linear regression, partialling-out score:
/// `psi = (Y - l(X) - theta (D - m(X))) (D - m(X))`, `l = E[Y|X]`, `m = E[D|X]`.
fn run_plr(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let n = y.len() as f64;
    let ml = Learner::Regression(config.ml_method);
    let preds = cross_fit(
        x,
        folds,
        None,
        ml,
        &[
            Nuisance::new("l", "E[Y|X]", y.view()),
            Nuisance::new("m", "E[D|X]", d.view()),
        ],
        config,
        warnings,
    )?;
    let (l, m) = (&preds[0], &preds[1]);
    let ry = y - l;
    let rd = d - m;
    let ssd = rd.dot(&rd);
    let jacobian = ssd / n;
    let var_d = var_pop(d);
    if jacobian.is_nan() || jacobian <= 1e-12 * var_d {
        return Err(EconError::SingularMatrix {
            context: format!("DoubleML PLR: E[(D - m)^2] = {jacobian:e} out of fold"),
            suggestion: "D is (almost) a linear function of the covariates; theta is not \
                         identified"
                .to_string(),
        });
    }
    let theta = rd.dot(&ry) / ssd;
    let scores: Array1<f64> = (&ry - &(&rd * theta)) * &rd;
    let se = (scores.dot(&scores) / n / (jacobian * jacobian) / n).sqrt();

    let label = ml.label(config.lambda);
    let fit_l = nuisance_fit("l", "E[Y|X]", &label, y, l, None);
    let fit_m = nuisance_fit("m", "E[D|X]", &label, d, m, None);
    let mean_rd = rd.sum() / n;
    let diagnostics = NuisanceDiagnostics {
        outcome_r2: fit_l.r2,
        treatment_r2: fit_m.r2,
        outcome_rmse: fit_l.rmse,
        treatment_rmse: fit_m.rmse,
        mean_residual_treatment: mean_rd,
        var_residual_treatment: jacobian,
    };
    residual_warnings(warnings, "m", jacobian, d, mean_rd);
    assemble(
        DMLModelType::PLR,
        config,
        theta,
        se,
        scores.to_vec(),
        diagnostics,
        jacobian,
        vec![fit_l, fit_m],
        None,
        warnings,
    )
}

/// Interactive regression model (binary treatment), ATE score:
/// `psi = g1 - g0 + D (Y - g1) / m - (1 - D)(Y - g0) / (1 - m) - theta`,
/// `g_d = E[Y|D=d,X]` by `ml_method` on arm d of the training folds, `m = P(D=1|X)` by
/// logistic regression, bounded to `[trim, 1 - trim]`. The score's Jacobian is 1.
fn run_irm(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let nobs = y.len();
    let n = nobs as f64;
    if let Some(at) = first_non_binary(d) {
        return Err(spec(format!(
            "IRM needs a binary treatment coded 0/1 ({at})"
        )));
    }
    let treated: Vec<bool> = d.iter().map(|&v| v == 1.0).collect();
    let control: Vec<bool> = treated.iter().map(|t| !t).collect();
    let n_treated = treated.iter().filter(|&&t| t).count();
    if n_treated.min(nobs - n_treated) < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(nobs - n_treated),
            context: "IRM requires sufficient observations in both treatment groups".to_string(),
        });
    }
    let ml = Learner::Regression(config.ml_method);
    let g0 = cross_fit(
        x,
        folds,
        Some(&control),
        ml,
        &[Nuisance::new("g0", "E[Y|D=0,X]", y.view())],
        config,
        warnings,
    )?
    .remove(0);
    let g1 = cross_fit(
        x,
        folds,
        Some(&treated),
        ml,
        &[Nuisance::new("g1", "E[Y|D=1,X]", y.view())],
        config,
        warnings,
    )?
    .remove(0);
    let m = cross_fit(
        x,
        folds,
        None,
        Learner::Logistic,
        &[Nuisance::new("m", "P(D=1|X)", d.view())],
        config,
        warnings,
    )?
    .remove(0)
    .mapv(|v| bound(v, config.trim, 1.0 - config.trim));

    let psi_b = Array1::from_shape_fn(nobs, |i| {
        g1[i] - g0[i] + d[i] * (y[i] - g1[i]) / m[i] - (1.0 - d[i]) * (y[i] - g0[i]) / (1.0 - m[i])
    });
    let theta = psi_b.sum() / n;
    let scores = psi_b.mapv(|v| v - theta);
    let se = (scores.dot(&scores) / n / n).sqrt();

    let label = ml.label(config.lambda);
    let fit_g0 = nuisance_fit("g0", "E[Y|D=0,X]", &label, y, &g0, Some(&control));
    let fit_g1 = nuisance_fit("g1", "E[Y|D=1,X]", &label, y, &g1, Some(&treated));
    let logistic = Learner::Logistic.label(config.lambda);
    let fit_m = nuisance_fit("m", "P(D=1|X)", &logistic, d, &m, None);
    // the outcome model of the observed arm
    let g_obs = Array1::from_shape_fn(nobs, |i| if treated[i] { g1[i] } else { g0[i] });
    let fit_out = nuisance_fit("g", "E[Y|D,X]", &label, y, &g_obs, None);
    let rd = d - &m;
    let diagnostics = NuisanceDiagnostics {
        outcome_r2: fit_out.r2,
        treatment_r2: fit_m.r2,
        outcome_rmse: fit_out.rmse,
        treatment_rmse: fit_m.rmse,
        mean_residual_treatment: rd.sum() / n,
        var_residual_treatment: rd.dot(&rd) / n,
    };
    propensity_warning(warnings, &m, config.trim);
    let range = min_max(&m);
    assemble(
        DMLModelType::IRM,
        config,
        theta,
        se,
        scores.to_vec(),
        diagnostics,
        1.0,
        vec![fit_g0, fit_g1, fit_m],
        Some(range),
        warnings,
    )
}

/// The result of one model: inference from theta and its SE, the nuisance fits and the
/// fields every model shares (folds are filled in by the entry point).
#[allow(clippy::too_many_arguments)]
fn assemble(
    model_type: DMLModelType,
    config: &DoubleMLConfig,
    theta: f64,
    se: f64,
    scores: Vec<f64>,
    nuisance_diagnostics: NuisanceDiagnostics,
    jacobian: f64,
    nuisance: Vec<NuisanceFit>,
    propensity_range: Option<[f64; 2]>,
    warnings: &[String],
) -> EconResult<DoubleMLResult> {
    let t_stat = if se > 0.0 && se.is_finite() {
        theta / se
    } else {
        f64::NAN
    };
    let p_value = normal_p_value_two_sided(t_stat);
    let z = z_critical(0.95)?;
    let n_obs = scores.len();
    Ok(DoubleMLResult {
        theta,
        se,
        t_stat,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        ci_lower: theta - z * se,
        ci_upper: theta + z * se,
        scores,
        nuisance_diagnostics,
        n_obs,
        n_folds: 0,
        model_type,
        treatment_type: TreatmentType::Continuous,
        ml_method: config.ml_method,
        jacobian,
        warnings: warnings.to_vec(),
        nuisance,
        lambda: matches!(config.ml_method, MLMethod::Lasso | MLMethod::Ridge)
            .then_some(config.lambda),
        trim: propensity_range.map(|_| config.trim),
        propensity_range,
        fold_sizes: Vec::new(),
        user_folds: false,
        seed: None,
    })
}

/// PLR and PLIV: the residualized treatment's out-of-fold variance and mean, relative to D.
fn residual_warnings(
    warnings: &mut Vec<String>,
    name: &str,
    w: f64,
    d: &ArrayView1<f64>,
    mean_rd: f64,
) {
    let var_d = var_pop(d);
    let ratio = w / var_d;
    if ratio < 0.01 {
        warnings.push(format!(
            "E[(D - {name})^2] / Var(D) = {ratio:.4} out of fold: the covariates explain more \
             than 99% of the variance of D, so theta is weakly identified and its standard \
             error is large"
        ));
    }
    let sd_d = var_d.sqrt();
    if mean_rd.abs() > 0.1 * sd_d {
        warnings.push(format!(
            "the mean of the residualized treatment is {mean_rd:.4} ({:.2} standard deviations \
             of D); the treatment learner is biased out of fold",
            mean_rd.abs() / sd_d
        ));
    }
}

/// IRM and IIVM: many trimmed propensities near the bounds.
fn propensity_warning(warnings: &mut Vec<String>, m: &Array1<f64>, trim: f64) {
    let n = m.len();
    let n_extreme = m
        .iter()
        .filter(|&&v| v < 2.0 * trim || v > 1.0 - 2.0 * trim)
        .count();
    if n_extreme > n / 10 {
        warnings.push(format!(
            "Many observations ({n_extreme}/{n}) have extreme propensity scores. Consider \
             stronger trimming."
        ));
    }
}

fn min_max(v: &Array1<f64>) -> [f64; 2] {
    v.iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], &x| {
            [lo.min(x), hi.max(x)]
        })
}

/// Population variance.
fn var_pop(v: &ArrayView1<f64>) -> f64 {
    let n = v.len() as f64;
    let mean = v.sum() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// RMSE and R² of the out-of-fold predictions `pred` of `target`, over `subset` (or all).
fn nuisance_fit(
    name: &str,
    target_label: &str,
    learner: &str,
    target: &ArrayView1<f64>,
    pred: &Array1<f64>,
    subset: Option<&[bool]>,
) -> NuisanceFit {
    let rows: Vec<usize> = (0..target.len())
        .filter(|&i| subset.is_none_or(|s| s[i]))
        .collect();
    let k = rows.len() as f64;
    let mean = rows.iter().map(|&i| target[i]).sum::<f64>() / k;
    let ssr: f64 = rows.iter().map(|&i| (target[i] - pred[i]).powi(2)).sum();
    let sst: f64 = rows.iter().map(|&i| (target[i] - mean).powi(2)).sum();
    NuisanceFit {
        name: name.to_string(),
        target: target_label.to_string(),
        learner: learner.to_string(),
        n_eval: rows.len(),
        rmse: (ssr / k).sqrt(),
        r2: if sst > 0.0 { 1.0 - ssr / sst } else { f64::NAN },
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn is_binary(v: &ArrayView1<f64>) -> bool {
    v.iter().all(|&x| x == 0.0 || x == 1.0)
}

/// The first value of `v` that is not exactly 0 or 1, as "row i has v" (1-based).
fn first_non_binary(v: &ArrayView1<f64>) -> Option<String> {
    v.iter()
        .position(|&x| x != 0.0 && x != 1.0)
        .map(|i| format!("row {} has {}", i + 1, v[i]))
}

/// Shared input checks of every model; returns the treatment type detected from `d`.
fn validate_inputs(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    z: Option<&ArrayView1<f64>>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
) -> EconResult<TreatmentType> {
    let n = y.len();
    if d.len() != n {
        return Err(spec(format!(
            "Treatment vector length ({}) must match outcome vector length ({n})",
            d.len()
        )));
    }
    if let Some(z) = z
        && z.len() != n
    {
        return Err(spec(format!(
            "Instrument vector length ({}) must match outcome vector length ({n})",
            z.len()
        )));
    }
    if x.nrows() != n {
        return Err(spec(format!(
            "Covariate matrix rows ({}) must match outcome vector length ({n})",
            x.nrows()
        )));
    }
    if !(config.trim.is_finite() && config.trim > 0.0 && config.trim < 0.5) {
        return Err(spec(format!(
            "trim must be in (0, 0.5), got {}",
            config.trim
        )));
    }
    if !(config.lambda.is_finite() && config.lambda >= 0.0) {
        return Err(spec(format!(
            "lambda must be finite and >= 0, got {}",
            config.lambda
        )));
    }
    if x.ncols() == 0 {
        match config.ml_method {
            MLMethod::Lasso => {
                return Err(spec(
                    "the lasso learner needs at least one covariate".to_string(),
                ));
            }
            MLMethod::Ridge => {
                return Err(spec(
                    "the ridge learner needs at least one covariate".to_string(),
                ));
            }
            MLMethod::OLS if !config.intercept => {
                return Err(spec(
                    "no covariates and no intercept: the nuisance functions are not defined"
                        .to_string(),
                ));
            }
            MLMethod::OLS => {}
        }
    }
    let first_bad = |v: &ArrayView1<f64>| v.iter().position(|x| !x.is_finite());
    let named = [
        ("y", first_bad(y)),
        ("d", first_bad(d)),
        ("z", z.and_then(first_bad)),
    ];
    for (name, bad) in named {
        if let Some(i) = bad {
            return Err(spec(format!(
                "{name} contains a non-finite value (row {})",
                i + 1
            )));
        }
    }
    for (j, column) in x.columns().into_iter().enumerate() {
        if let Some(i) = column.iter().position(|v| !v.is_finite()) {
            return Err(spec(format!(
                "x column {} contains a non-finite value (row {})",
                j + 1,
                i + 1
            )));
        }
    }
    let (lo, hi) = d
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if lo == hi {
        return Err(spec("treatment d is constant".to_string()));
    }
    if config.treatment_type == TreatmentType::Binary
        && let Some(at) = first_non_binary(d)
    {
        return Err(spec(format!(
            "treatment_type = Binary but d is not coded 0/1 ({at})"
        )));
    }
    Ok(if is_binary(d) {
        TreatmentType::Binary
    } else {
        TreatmentType::Continuous
    })
}

/// The cross-fitting folds: `index[i]` is observation i's fold (its label's position in
/// `labels`) and `sizes[k]` the size of fold k; `seed` is `None` for user folds.
struct Folds {
    index: Vec<usize>,
    labels: Vec<usize>,
    sizes: Vec<usize>,
    seed: Option<u64>,
}

/// User folds (labels sorted, each observation mapped to its label's position) or random
/// folds from the seed.
fn make_folds(n: usize, config: &DoubleMLConfig) -> EconResult<Folds> {
    let (index, labels, seed) = match &config.folds {
        Some(v) => {
            if v.len() != n {
                return Err(spec(format!(
                    "folds has {} labels for {n} observations",
                    v.len()
                )));
            }
            let mut labels = v.clone();
            labels.sort_unstable();
            labels.dedup();
            if labels.len() < 2 {
                return Err(spec(format!(
                    "folds must contain at least 2 distinct labels, got {}",
                    labels.len()
                )));
            }
            let index = v
                .iter()
                .map(|l| labels.binary_search(l))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    spec(format!(
                        "folds must contain at least 2 distinct labels, got {}",
                        labels.len()
                    ))
                })?;
            (index, labels, None)
        }
        None => {
            let k = config.n_folds;
            if k < 2 || k > n {
                return Err(spec(format!(
                    "n_folds must be between 2 and the number of observations ({n}), got {k}"
                )));
            }
            let seed = config.seed.unwrap_or(DML_DEFAULT_SEED);
            (
                create_fold_assignments(n, k, seed),
                (1..=k).collect(),
                Some(seed),
            )
        }
    };
    let mut sizes = vec![0; labels.len()];
    for &k in &index {
        sizes[k] += 1;
    }
    Ok(Folds {
        index,
        labels,
        sizes,
        seed,
    })
}

/// Random fold assignment: `0..n` shuffled with ChaCha8 from `seed`; the observation at
/// shuffled position j gets fold `j % k_folds`.
fn create_fold_assignments(n: usize, k_folds: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Create and shuffle indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    // Assign folds
    let mut fold_ids = vec![0; n];
    for (i, &idx) in indices.iter().enumerate() {
        fold_ids[idx] = i % k_folds;
    }

    fold_ids
}

/// Add intercept column (column of ones) to design matrix.
fn add_intercept(x: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();

    let mut x_with_intercept = Array2::zeros((n, p + 1));

    // First column is intercept
    for i in 0..n {
        x_with_intercept[[i, 0]] = 1.0;
    }

    // Copy original columns
    for i in 0..n {
        for j in 0..p {
            x_with_intercept[[i, j + 1]] = x[[i, j]];
        }
    }

    x_with_intercept
}

// =============================================================================
// Cross-fitting
// =============================================================================

/// The learner of one nuisance function.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Learner {
    /// `config.ml_method` (a regression nuisance)
    Regression(MLMethod),
    /// Logistic regression (a binary nuisance)
    Logistic,
}

impl Learner {
    fn label(self, lambda: f64) -> String {
        match self {
            Learner::Regression(MLMethod::OLS) => "OLS".to_string(),
            Learner::Regression(MLMethod::Lasso) => format!("lasso (glmnet, lambda = {lambda})"),
            Learner::Regression(MLMethod::Ridge) => format!("ridge (glmnet, lambda = {lambda})"),
            Learner::Logistic => "logistic regression".to_string(),
        }
    }
}

/// One nuisance function `name = target_label`: the regression of `target` on X.
struct Nuisance<'a> {
    name: &'static str,
    target_label: &'static str,
    target: ArrayView1<'a, f64>,
}

impl<'a> Nuisance<'a> {
    fn new(name: &'static str, target_label: &'static str, target: ArrayView1<'a, f64>) -> Self {
        Self {
            name,
            target_label,
            target,
        }
    }
}

/// The notes of one nuisance, merged over folds in first-occurrence order.
#[derive(Default)]
struct FoldNotes(Vec<(String, Vec<usize>)>);

impl FoldNotes {
    fn add(&mut self, text: String, label: usize) {
        match self.0.iter_mut().find(|(t, _)| *t == text) {
            Some((_, labels)) => labels.push(label),
            None => self.0.push((text, vec![label])),
        }
    }

    /// "`prefix`, folds 1, 2, 3: `note`" ("fold 3" for one fold) per distinct note.
    fn emit(self, prefix: &str, warnings: &mut Vec<String>) {
        for (text, labels) in self.0 {
            let list: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
            let folds = if list.len() == 1 {
                format!("fold {}", list[0])
            } else {
                format!("folds {}", list.join(", "))
            };
            warnings.push(format!("{prefix}, {folds}: {text}"));
        }
    }
}

/// Cross-fit the nuisances that share a training subset and a learner: for each fold k,
/// train on the observations outside fold k (and in `subset`, when given) and predict
/// every observation of fold k. Returns one length-n prediction vector per nuisance; the
/// learners' notes go to `warnings`, merged over folds.
fn cross_fit(
    x: &ArrayView2<f64>,
    folds: &Folds,
    subset: Option<&[bool]>,
    learner: Learner,
    nuisances: &[Nuisance<'_>],
    config: &DoubleMLConfig,
    warnings: &mut Vec<String>,
) -> EconResult<Vec<Array1<f64>>> {
    let (n, p) = x.dim();
    let q = p + usize::from(config.intercept);
    // At lambda = 0 the lasso and ridge objectives are least squares.
    let fitted_by = match learner {
        Learner::Regression(MLMethod::Lasso | MLMethod::Ridge) if config.lambda == 0.0 => {
            Learner::Regression(MLMethod::OLS)
        }
        other => other,
    };
    let min_train = match fitted_by {
        Learner::Regression(MLMethod::OLS) | Learner::Logistic => q + 1,
        Learner::Regression(_) => 2,
    };
    let mut preds = nuisances
        .iter()
        .map(|_| resource::try_zeros1(n, "DoubleML cross-fitted predictions"))
        .collect::<EconResult<Vec<_>>>()?;
    let mut notes: Vec<FoldNotes> = nuisances.iter().map(|_| FoldNotes::default()).collect();
    let names = design_names(p, config.intercept);
    for (k, &label) in folds.labels.iter().enumerate() {
        resource::checkpoint()?;
        let train: Vec<usize> = (0..n)
            .filter(|&i| folds.index[i] != k && subset.is_none_or(|s| s[i]))
            .collect();
        let test: Vec<usize> = (0..n).filter(|&i| folds.index[i] == k).collect();
        if train.len() < min_train {
            let first = &nuisances[0];
            return Err(EconError::InsufficientData {
                required: min_train,
                provided: train.len(),
                context: format!(
                    "training rows of fold {label} for {} = {}",
                    first.name, first.target_label
                ),
            });
        }
        let fold = fit_predict(
            fitted_by, x, &train, &test, nuisances, config, &names, &mut notes, label,
        )?;
        for (pred, f) in preds.iter_mut().zip(fold) {
            for (t, &i) in test.iter().enumerate() {
                pred[i] = f[t];
            }
        }
    }
    let learner_label = learner.label(config.lambda);
    for (nu, nn) in nuisances.iter().zip(notes) {
        nn.emit(
            &format!("{} = {} by {learner_label}", nu.name, nu.target_label),
            warnings,
        );
    }
    Ok(preds)
}

/// One fold: fit every nuisance on the `train` rows and predict the `test` rows.
#[allow(clippy::too_many_arguments)]
fn fit_predict(
    learner: Learner,
    x: &ArrayView2<f64>,
    train: &[usize],
    test: &[usize],
    nuisances: &[Nuisance<'_>],
    config: &DoubleMLConfig,
    names: &[String],
    notes: &mut [FoldNotes],
    label: usize,
) -> EconResult<Vec<Array1<f64>>> {
    let targets: Vec<Array1<f64>> = nuisances
        .iter()
        .map(|nu| train.iter().map(|&i| nu.target[i]).collect())
        .collect();
    match learner {
        Learner::Regression(MLMethod::OLS) => {
            let a_train = design_rows(x, train, config.intercept)?;
            let a_test = design_rows(x, test, config.intercept)?;
            // OLS's solve decision: the normal equations on a well-conditioned design (one
            // inverse for every target), lm.fit's pivoted QR otherwise.
            if let Some(inv) = normal_equations_inverse(a_train.view())? {
                let mut t = resource::try_zeros2(train.len(), targets.len(), "DoubleML targets")?;
                for (j, tj) in targets.iter().enumerate() {
                    t.column_mut(j).assign(tj);
                }
                let beta = inv.coefficients_multi(a_train.view(), t.view());
                let fitted = a_test.dot(&beta);
                return Ok(fitted.columns().into_iter().map(|c| c.to_owned()).collect());
            }
            let qr = Qr::decompose(a_train.view(), LM_QR_TOL)?;
            let note = aliasing_message(&qr.aliased(), names);
            let mut out = Vec::with_capacity(targets.len());
            for (j, tj) in targets.iter().enumerate() {
                // aliased coefficients predict as 0 (predict.lm on a rank-deficient fit)
                let beta = qr
                    .coefficients(tj.view())?
                    .mapv(|v| if v.is_nan() { 0.0 } else { v });
                if let Some(text) = &note {
                    notes[j].add(text.clone(), label);
                }
                out.push(a_test.dot(&beta));
            }
            Ok(out)
        }
        Learner::Regression(MLMethod::Lasso) => {
            let x_train = design_rows(x, train, false)?;
            let x_test = design_rows(x, test, false)?;
            let mut c = GlmnetConfig::default();
            c.alpha = 1.0;
            c.lambda = Some(vec![config.lambda]);
            c.standardize = true;
            c.intercept = config.intercept;
            c.thresh = DML_GLMNET_THRESH;
            let mut out = Vec::with_capacity(targets.len());
            for (j, tj) in targets.iter().enumerate() {
                let fit = glmnet(x_train.view(), tj.view(), &c)?;
                for w in &fit.warnings {
                    notes[j].add(w.clone(), label);
                }
                out.push(glmnet_predict(&fit, x_test.view(), config.lambda)?);
            }
            Ok(out)
        }
        Learner::Regression(MLMethod::Ridge) => {
            let x_train = design_rows(x, train, false)?;
            let x_test = design_rows(x, test, false)?;
            let fit = RidgeFit::new(x_train.view(), config.intercept)?;
            targets
                .iter()
                .map(|tj| {
                    let (b0, beta) = fit.coefficients(tj.view(), config.lambda)?;
                    Ok(x_test.dot(&beta) + b0)
                })
                .collect()
        }
        Learner::Logistic => {
            let a_train = design_rows(x, train, config.intercept)?;
            let a_test = design_rows(x, test, config.intercept)?;
            let mut out = Vec::with_capacity(targets.len());
            for (j, tj) in targets.iter().enumerate() {
                let c = tj[0];
                if tj.iter().all(|&v| v == c) {
                    notes[j].add(
                        format!(
                            "the training outcome has no variation (all {c}); predicted as the \
                             constant {c}"
                        ),
                        label,
                    );
                    out.push(Array1::from_elem(test.len(), c));
                    continue;
                }
                let pf = fit_propensity(a_train.view(), tj.view(), GlmLink::Logit)?;
                if let Some(text) = aliasing_message(&pf.fit.aliased, names) {
                    notes[j].add(text, label);
                }
                for w in &pf.fit.warnings {
                    notes[j].add(w.clone(), label);
                }
                let beta = pf
                    .fit
                    .coefficients
                    .mapv(|v| if v.is_nan() { 0.0 } else { v });
                out.push(a_test.dot(&beta).mapv(plogis));
            }
            Ok(out)
        }
    }
}

/// The rows `rows` of `x`, with a leading column of ones when `intercept`.
fn design_rows(x: &ArrayView2<f64>, rows: &[usize], intercept: bool) -> EconResult<Array2<f64>> {
    let off = usize::from(intercept);
    let mut a = resource::try_zeros2(rows.len(), x.ncols() + off, "DoubleML design")?;
    for (r, &i) in rows.iter().enumerate() {
        if intercept {
            a[[r, 0]] = 1.0;
        }
        for (j, &v) in x.row(i).iter().enumerate() {
            a[[r, j + off]] = v;
        }
    }
    Ok(a)
}

/// Column names of the design for the aliasing note: "(Intercept)", "x1", .., "xp".
fn design_names(p: usize, intercept: bool) -> Vec<String> {
    intercept
        .then(|| "(Intercept)".to_string())
        .into_iter()
        .chain((1..=p).map(|j| format!("x{j}")))
        .collect()
}

/// The ridge learner: the exact minimiser of glmnet's gaussian ridge objective
/// (alpha = 0, standardize = TRUE) at one lambda, by its closed form instead of coordinate
/// descent.
///
/// glmnet centres x by its column means when it fits an intercept, divides every
/// non-constant column by its population sd (a constant column keeps coefficient 0),
/// and solves on y / ys (ys the population sd of y with an intercept, sqrt(mean(y²))
/// without), so its ridge penalty in y units is `lambda / ys`:
/// `(Z'Z / n + (lambda / ys) I) b = Z'yc / n`, `coef = b / scale`. With more columns than
/// rows the dual `b = Z' (Z Z' / n + (lambda / ys) I)^-1 yc / n` is solved instead. All
/// targets on one training set share Z and its Gram (or kernel) matrix.
struct RidgeFit {
    /// Standardised usable columns (n x u).
    z: Array2<f64>,
    /// Indices of the usable (non-constant) columns of x.
    usable: Vec<usize>,
    center: Vec<f64>,
    scale: Vec<f64>,
    p: usize,
    intercept: bool,
    /// Z'Z / n (u <= n) or Z Z' / n (u > n).
    gram: Array2<f64>,
}

impl RidgeFit {
    fn new(x: ArrayView2<f64>, intercept: bool) -> EconResult<Self> {
        let (n, p) = x.dim();
        let nf = n as f64;
        let mut usable = Vec::new();
        let mut center = vec![0.0; p];
        let mut scale = vec![1.0; p];
        for (j, col) in x.columns().into_iter().enumerate() {
            let mean = col.sum() / nf;
            let (lo, hi) = col
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            if intercept {
                center[j] = mean;
            }
            if hi > lo {
                scale[j] = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nf).sqrt();
                usable.push(j);
            }
        }
        if usable.is_empty() {
            return Err(spec(
                "ridge learner: every covariate is constant on a training set".to_string(),
            ));
        }
        let mut z = resource::try_zeros2(n, usable.len(), "ridge standardised design")?;
        for (u, &j) in usable.iter().enumerate() {
            for i in 0..n {
                z[[i, u]] = (x[[i, j]] - center[j]) / scale[j];
            }
        }
        let gram = if usable.len() <= n {
            z.t().dot(&z) / nf
        } else {
            resource::check_alloc::<f64>(
                resource::checked_len(&[n, n], "ridge kernel matrix")?,
                "ridge kernel matrix",
            )?;
            z.dot(&z.t()) / nf
        };
        Ok(Self {
            z,
            usable,
            center,
            scale,
            p,
            intercept,
            gram,
        })
    }

    /// The response as glmnet standardises it: `(y - ym, ym, ys)`.
    fn response(&self, y: ArrayView1<f64>) -> EconResult<(Array1<f64>, f64, f64)> {
        let nf = y.len() as f64;
        let ym = if self.intercept { y.sum() / nf } else { 0.0 };
        let yc = y.mapv(|v| v - ym);
        let ys = (yc.dot(&yc) / nf).sqrt();
        if ys.is_nan() || ys <= 0.0 {
            return Err(spec(
                "ridge learner: the training outcome is constant (glmnet's gaussian fit is not \
                 defined)"
                    .to_string(),
            ));
        }
        Ok((yc, ym, ys))
    }

    fn primal(&self) -> bool {
        self.usable.len() <= self.z.nrows()
    }

    /// `(Z'Z / n + l2 I)^-1 Z'yc / n`.
    fn solve_primal(&self, yc: &Array1<f64>, l2: f64) -> EconResult<Array1<f64>> {
        let nf = yc.len() as f64;
        let mut a = if self.primal() {
            self.gram.clone()
        } else {
            self.z.t().dot(&self.z) / nf
        };
        a.diag_mut().mapv_inplace(|v| v + l2);
        spd_solve(a, self.z.t().dot(yc) / nf)
    }

    /// `Z' (Z Z' / n + l2 I)^-1 yc / n`.
    fn solve_dual(&self, yc: &Array1<f64>, l2: f64) -> EconResult<Array1<f64>> {
        let nf = yc.len() as f64;
        let mut a = if self.primal() {
            self.z.dot(&self.z.t()) / nf
        } else {
            self.gram.clone()
        };
        a.diag_mut().mapv_inplace(|v| v + l2);
        let w = spd_solve(a, yc / nf)?;
        Ok(self.z.t().dot(&w))
    }

    /// Intercept and coefficients (original scale, one per column of x) at `lambda` > 0.
    fn coefficients(&self, y: ArrayView1<f64>, lambda: f64) -> EconResult<(f64, Array1<f64>)> {
        let (yc, ym, ys) = self.response(y)?;
        let l2 = lambda / ys;
        let b = if self.primal() {
            self.solve_primal(&yc, l2)?
        } else {
            self.solve_dual(&yc, l2)?
        };
        let mut coef = Array1::zeros(self.p);
        for (u, &j) in self.usable.iter().enumerate() {
            coef[j] = b[u] / self.scale[j];
        }
        let b0 = if self.intercept {
            ym - coef
                .iter()
                .zip(&self.center)
                .map(|(c, m)| c * m)
                .sum::<f64>()
        } else {
            0.0
        };
        Ok((b0, coef))
    }
}

/// Solve `a x = b` for a symmetric positive definite `a` by Cholesky.
fn spd_solve(mut a: Array2<f64>, mut b: Array1<f64>) -> EconResult<Array1<f64>> {
    let m = a.nrows();
    for j in 0..m {
        let mut d = a[[j, j]];
        for k in 0..j {
            d -= a[[j, k]] * a[[j, k]];
        }
        if d.is_nan() || d <= 0.0 {
            return Err(EconError::SingularMatrix {
                context: "ridge learner: the penalised Gram matrix is not positive definite"
                    .to_string(),
                suggestion: "use a larger lambda or rescale the covariates".to_string(),
            });
        }
        let d = d.sqrt();
        a[[j, j]] = d;
        for i in j + 1..m {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= a[[i, k]] * a[[j, k]];
            }
            a[[i, j]] = s / d;
        }
    }
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= a[[i, k]] * b[k];
        }
        b[i] = s / a[[i, i]];
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= a[[k, i]] * b[k];
        }
        b[i] = s / a[[i, i]];
    }
    Ok(b)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Create a test dataset for PLR model.
    ///
    /// DGP:
    ///   X ~ Uniform(0, 1)
    ///   D = 0.5 * X + 0.3 + noise
    ///   Y = 0.5 * D + 0.3 * X + noise
    ///
    /// True theta = 0.5
    fn create_plr_test_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut y = Array1::zeros(n);
        let mut d = Array1::zeros(n);
        let mut x = Array2::zeros((n, 2)); // Two covariates

        for i in 0..n {
            let x1: f64 = rng.r#gen();
            let x2: f64 = rng.r#gen();
            let noise_d: f64 = rng.r#gen::<f64>() * 0.2 - 0.1;
            let noise_y: f64 = rng.r#gen::<f64>() * 0.2 - 0.1;

            x[[i, 0]] = x1;
            x[[i, 1]] = x2;

            // Treatment: depends on X with noise
            d[i] = 0.5 * x1 + 0.3 * x2 + noise_d;

            // Outcome: theta=0.5 on D, plus X effect, plus noise
            y[i] = 0.5 * d[i] + 0.3 * x1 + 0.2 * x2 + noise_y;
        }

        (y, d, x)
    }

    /// Create a test dataset for IRM model (binary treatment).
    ///
    /// DGP:
    ///   X ~ Uniform(0, 1)
    ///   D ~ Bernoulli(expit(0.5 + X))
    ///   Y(0) = 0.3 * X + noise
    ///   Y(1) = 0.5 + 0.3 * X + noise
    ///
    /// True ATE = 0.5
    fn create_irm_test_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut y = Array1::zeros(n);
        let mut d = Array1::zeros(n);
        let mut x = Array2::zeros((n, 2));

        for i in 0..n {
            let x1: f64 = rng.r#gen();
            let x2: f64 = rng.r#gen();
            let noise: f64 = rng.r#gen::<f64>() * 0.2 - 0.1;

            x[[i, 0]] = x1;
            x[[i, 1]] = x2;

            // Propensity score
            let ps = 1.0 / (1.0 + (-0.5 - x1 + 0.3 * x2).exp());
            let u: f64 = rng.r#gen();
            d[i] = if u < ps { 1.0 } else { 0.0 };

            // Potential outcomes
            let y0 = 0.3 * x1 + 0.2 * x2 + noise;
            let y1 = 0.5 + 0.3 * x1 + 0.2 * x2 + noise; // ATE = 0.5

            y[i] = if d[i] >= 0.5 { y1 } else { y0 };
        }

        (y, d, x)
    }

    #[test]
    fn test_plr_basic() {
        let (y, d, x) = create_plr_test_data(500, 42);
        let config = DoubleMLConfig {
            n_folds: 5,
            model_type: DMLModelType::PLR,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 500);
        assert_eq!(result.n_folds, 5);
        assert_eq!(result.model_type, DMLModelType::PLR);

        // Check theta is approximately 0.5 (true value)
        // With n=500 and noise, we expect theta to be within a reasonable range
        assert!(result.theta > 0.2, "theta too low: {}", result.theta);
        assert!(result.theta < 0.8, "theta too high: {}", result.theta);

        // Check standard error is positive and reasonable
        assert!(result.se > 0.0, "SE should be positive: {}", result.se);
        assert!(result.se < 0.5, "SE seems too large: {}", result.se);

        // Check confidence interval
        assert!(result.ci_lower < result.theta);
        assert!(result.ci_upper > result.theta);

        // Check nuisance diagnostics
        assert!(result.nuisance_diagnostics.outcome_r2 >= 0.0);
        assert!(result.nuisance_diagnostics.treatment_r2 >= 0.0);
    }

    #[test]
    fn test_irm_basic() {
        let (y, d, x) = create_irm_test_data(500, 42);
        let config = DoubleMLConfig {
            n_folds: 5,
            model_type: DMLModelType::IRM,
            treatment_type: TreatmentType::Binary,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 500);
        assert_eq!(result.model_type, DMLModelType::IRM);

        // Check theta is approximately 0.5 (true ATE)
        assert!(result.theta > 0.2, "theta too low: {}", result.theta);
        assert!(result.theta < 0.8, "theta too high: {}", result.theta);

        // Standard error should be positive
        assert!(result.se > 0.0);
    }

    #[test]
    fn test_fold_assignments() {
        let fold_ids = create_fold_assignments(100, 5, 42);

        // Check all folds are represented
        let mut counts = vec![0; 5];
        for &f in &fold_ids {
            assert!(f < 5);
            counts[f] += 1;
        }

        // Each fold should have approximately 20 observations
        for c in counts {
            assert!((15..=25).contains(&c), "Fold count {} is unbalanced", c);
        }
    }

    #[test]
    fn test_add_intercept() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x_int = add_intercept(&x.view());

        assert_eq!(x_int.dim(), (3, 3));
        // First column should be ones
        assert!((x_int[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((x_int[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((x_int[[2, 0]] - 1.0).abs() < 1e-10);
        // Original values preserved
        assert!((x_int[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((x_int[[0, 2]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data_error() {
        let y = Array1::from(vec![1.0, 2.0, 3.0]);
        let d = Array1::from(vec![0.0, 1.0, 0.5]);
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let config = DoubleMLConfig {
            n_folds: 5, // Can't have 5 folds with 3 observations
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config);
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let (y, d, x) = create_plr_test_data(100, 42);
        let config = DoubleMLConfig {
            n_folds: 3,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();
        let output = format!("{}", result);

        assert!(output.contains("Double/Debiased Machine Learning"));
        assert!(output.contains("theta"));
        assert!(output.contains("Std. Error"));
        assert!(output.contains("95% CI"));
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let (y, d, x) = create_plr_test_data(200, 42);

        let config1 = DoubleMLConfig {
            n_folds: 5,
            seed: Some(123),
            ..Default::default()
        };
        let config2 = DoubleMLConfig {
            n_folds: 5,
            seed: Some(123),
            ..Default::default()
        };

        let result1 = run_double_ml(&y.view(), &d.view(), &x.view(), config1).unwrap();
        let result2 = run_double_ml(&y.view(), &d.view(), &x.view(), config2).unwrap();

        // With same seed, results should be identical
        assert!((result1.theta - result2.theta).abs() < 1e-10);
        assert!((result1.se - result2.se).abs() < 1e-10);
    }

    // =========================================================================
    // DGP recovery (property tests, not R pins)
    // =========================================================================

    /// DGP recovery (property test, not an R pin): PLR recovers a known treatment effect
    /// from a linear DGP.
    ///
    /// DGP (linear, so OLS nuisance models are correctly specified):
    ///   X1, X2 ~ Uniform(0,1)
    ///   D = 0.5*X1 + 0.3*X2 + V,  V ~ Uniform(-0.1, 0.1)
    ///   Y = theta*D + 0.3*X1 + 0.2*X2 + U,  U ~ Uniform(-0.1, 0.1)
    ///
    /// True theta = 0.5. With n=1000 and linear nuisance, cross-fitted OLS
    /// should recover theta to within ~0.1 of the true value.
    #[test]
    fn test_doubleml_plr_recovers_dgp() {
        let (y, d, x) = create_plr_test_data(1000, 99);
        let true_theta = 0.5;

        let config = DoubleMLConfig {
            n_folds: 5,
            model_type: DMLModelType::PLR,
            seed: Some(99),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

        // theta_hat should be close to true theta (within 0.15 for n=1000)
        assert!(
            (result.theta - true_theta).abs() < 0.15,
            "PLR theta_hat = {:.4} should be within 0.15 of true theta = {}",
            result.theta,
            true_theta
        );

        // Standard error should be positive and reasonable (< 0.1 for n=1000)
        assert!(result.se > 0.0, "SE must be positive");
        assert!(
            result.se < 0.1,
            "SE = {:.4} too large for n=1000",
            result.se
        );

        // 95% CI should contain the true theta
        assert!(
            result.ci_lower < true_theta && true_theta < result.ci_upper,
            "95% CI [{:.4}, {:.4}] should contain true theta = {}",
            result.ci_lower,
            result.ci_upper,
            true_theta
        );

        // t-stat should be consistent with theta/se
        assert!(
            (result.t_stat - result.theta / result.se).abs() < 1e-8,
            "t_stat = {:.4} should equal theta/se = {:.4}",
            result.t_stat,
            result.theta / result.se
        );

        // p-value should be small (theta != 0)
        assert!(
            result.p_value < 0.05,
            "p-value = {:.4} should reject H0: theta=0",
            result.p_value
        );
    }

    /// DGP recovery (property test, not an R pin): cross-fitting uses K folds and computes
    /// one score per observation.
    #[test]
    fn test_doubleml_crossfitting_structure() {
        let (y, d, x) = create_plr_test_data(500, 77);

        for n_folds in [2, 3, 5] {
            let config = DoubleMLConfig {
                n_folds,
                seed: Some(77),
                ..Default::default()
            };

            let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

            // Result should report the correct number of folds
            assert_eq!(result.n_folds, n_folds);
            assert_eq!(result.n_obs, 500);

            // Scores (psi) should have one entry per observation
            assert_eq!(
                result.scores.len(),
                500,
                "Should have n={} orthogonal scores",
                500
            );

            // All scores should be finite
            assert!(
                result.scores.iter().all(|s| s.is_finite()),
                "All scores must be finite"
            );

            // Mean of scores should be close to zero at the estimated theta
            // (this is the moment condition)
            let mean_score: f64 = result.scores.iter().sum::<f64>() / 500.0;
            assert!(
                mean_score.abs() < 0.1,
                "Mean orthogonal score = {:.6} should be near zero",
                mean_score
            );

            // Jacobian should be positive
            assert!(
                result.jacobian > 0.0,
                "Jacobian J = {:.6} should be positive",
                result.jacobian
            );
        }
    }

    /// DGP recovery (property test, not an R pin): IRM recovers a known ATE from a binary
    /// treatment DGP.
    ///
    /// True ATE = 0.5. With n=1000 and linear nuisance models, should be
    /// recovered within ~0.3 (wider tolerance for binary treatment).
    #[test]
    fn test_doubleml_irm_recovers_dgp() {
        let (y, d, x) = create_irm_test_data(1000, 55);
        let true_ate = 0.5;

        let config = DoubleMLConfig {
            n_folds: 5,
            model_type: DMLModelType::IRM,
            treatment_type: TreatmentType::Binary,
            seed: Some(55),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

        // theta_hat should be in reasonable range of true ATE
        assert!(
            (result.theta - true_ate).abs() < 0.3,
            "IRM theta_hat = {:.4} should be within 0.3 of true ATE = {}",
            result.theta,
            true_ate
        );

        // SE should be positive
        assert!(result.se > 0.0, "SE must be positive: {}", result.se);

        // CI should be properly ordered
        assert!(
            result.ci_lower < result.ci_upper,
            "CI lower ({:.4}) must be < CI upper ({:.4})",
            result.ci_lower,
            result.ci_upper
        );

        // Nuisance diagnostics should be populated
        assert!(
            result.nuisance_diagnostics.outcome_r2 >= 0.0,
            "Outcome R2 should be non-negative"
        );
        assert!(
            result.nuisance_diagnostics.var_residual_treatment > 0.0,
            "Residual treatment variance should be positive"
        );
    }

    /// DGP recovery (property test, not an R pin): with no treatment effect theta is near 0.
    #[test]
    fn test_doubleml_null_effect_recovers_dgp() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let n = 500;
        let mut y = Array1::zeros(n);
        let mut d = Array1::zeros(n);
        let mut x = Array2::zeros((n, 2));

        for i in 0..n {
            let x1: f64 = rng.r#gen();
            let x2: f64 = rng.r#gen();
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;

            // Treatment depends on X
            d[i] = 0.5 * x1 + 0.3 * x2 + (rng.r#gen::<f64>() * 0.2 - 0.1);

            // Outcome: NO treatment effect (theta = 0)
            y[i] = 0.3 * x1 + 0.2 * x2 + (rng.r#gen::<f64>() * 0.2 - 0.1);
        }

        let config = DoubleMLConfig {
            n_folds: 5,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_double_ml(&y.view(), &d.view(), &x.view(), config).unwrap();

        // theta should be close to zero
        assert!(
            result.theta.abs() < 0.15,
            "Under null, theta_hat = {:.4} should be near zero",
            result.theta
        );

        // p-value should typically NOT reject (but allow for randomness)
        // Just check it is properly computed (between 0 and 1)
        assert!(
            (0.0..=1.0).contains(&result.p_value),
            "p-value = {} should be in [0, 1]",
            result.p_value
        );
    }

    // =========================================================================
    // The reference case doubleml_models (validation/r/cases/causal_doubleml.R)
    // =========================================================================

    use crate::data::Dataset;
    use crate::data::columns::{NaPolicy, numeric_column};
    use crate::linalg::DesignMatrix;
    use p2a_validation::{Check, Tol};

    const X6: [&str; 6] = ["X1", "X2", "X3", "X4", "X5", "X6"];
    /// relative: Tol::p_value's abs 1e-12 would accept 0 or 1 - cdf for these tails
    const P_REL: Tol = Tol::new(1e-6, 0.0);

    /// The case's dataset for property tests: read without `load_case`, so the registry does
    /// not count them as R pins (only `test_validate_*` tests load the case).
    fn sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/doubleml_sim.csv");
        match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        }
    }
    fn col(ds: &Dataset, name: &str) -> Array1<f64> {
        Array1::from_vec(numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0)
    }
    fn covariates(ds: &Dataset) -> Array2<f64> {
        DesignMatrix::from_dataframe(ds.df(), &X6, false)
            .unwrap()
            .data
    }
    fn case_folds(ds: &Dataset) -> Vec<usize> {
        col(ds, "fold").iter().map(|&v| v as usize).collect()
    }
    fn with_folds(model: DMLModelType, ml: MLMethod, folds: Vec<usize>) -> DoubleMLConfig {
        let mut c = DoubleMLConfig::default();
        c.model_type = model;
        c.ml_method = ml;
        c.folds = Some(folds);
        c
    }
    /// Uses only fields that existed before the learner rewrite.
    fn seeded(model: DMLModelType, ml: MLMethod, k: usize, seed: u64) -> DoubleMLConfig {
        let mut c = DoubleMLConfig::default();
        c.model_type = model;
        c.ml_method = ml;
        c.n_folds = k;
        c.seed = Some(seed);
        c
    }
    fn spec_err(r: EconResult<DoubleMLResult>, needle: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{needle:?} not in {message:?}")
            }
            other => panic!("expected InvalidSpecification containing {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn test_ml_method_parses_aliases_and_rejects_unknown() {
        MLMethod::choices().validate().unwrap();
        for s in ["ols", "OLS", "lm", "regr.lm", "linear"] {
            assert_eq!(s.parse::<MLMethod>().unwrap(), MLMethod::OLS, "{s}");
        }
        for s in ["Lasso", "L1"] {
            assert_eq!(s.parse::<MLMethod>().unwrap(), MLMethod::Lasso, "{s}");
        }
        for s in ["ridge", "l2"] {
            assert_eq!(s.parse::<MLMethod>().unwrap(), MLMethod::Ridge, "{s}");
        }
        match "random forest".parse::<MLMethod>() {
            Err(EconError::InvalidSpecification { message }) => {
                for needle in ["ml_method", "ols", "lasso", "ridge"] {
                    assert!(message.contains(needle), "{needle} not in {message}");
                }
            }
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[test]
    fn test_dml_model_type_parses_aliases_and_rejects_unknown() {
        DMLModelType::choices().validate().unwrap();
        for s in ["plr", "PLR", "partially-linear", "DoubleMLPLR"] {
            assert_eq!(s.parse::<DMLModelType>().unwrap(), DMLModelType::PLR, "{s}");
        }
        for s in ["irm", "IRM", "interactive"] {
            assert_eq!(s.parse::<DMLModelType>().unwrap(), DMLModelType::IRM, "{s}");
        }
        for s in ["pliv", "partially linear iv", "DoubleMLPLIV"] {
            assert_eq!(
                s.parse::<DMLModelType>().unwrap(),
                DMLModelType::PLIV,
                "{s}"
            );
        }
        for s in ["iivm", "interactive_iv", "DoubleMLIIVM"] {
            assert_eq!(
                s.parse::<DMLModelType>().unwrap(),
                DMLModelType::IIVM,
                "{s}"
            );
        }
        match "ols".parse::<DMLModelType>() {
            Err(EconError::InvalidSpecification { message }) => {
                for needle in ["model", "plr", "irm", "iivm"] {
                    assert!(message.contains(needle), "{needle} not in {message}");
                }
            }
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_doubleml_user_folds() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let n = y.len();
        let folds = case_folds(&ds);
        let r = run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            with_folds(DMLModelType::PLR, MLMethod::OLS, folds.clone()),
        )
        .unwrap();
        let sizes: Vec<f64> = r.fold_sizes.iter().map(|&s| s as f64).collect();
        Check::new(&case)
            .vector("fold_sizes", &sizes, Tol::exact())
            .scalar("plr_ols_theta", r.theta, Tol::coef(n))
            .scalar("plr_ols_se", r.se, Tol::se(n))
            .finish();
        assert!(r.user_folds);
        assert_eq!(r.seed, None);
        assert_eq!(r.n_folds, 5);
        // the labels are names, not positions
        let shifted: Vec<usize> = folds.iter().map(|f| f + 10).collect();
        let s = run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            with_folds(DMLModelType::PLR, MLMethod::OLS, shifted),
        )
        .unwrap();
        assert_eq!(s.theta.to_bits(), r.theta.to_bits());
    }

    #[test]
    fn test_doubleml_default_seed_is_reproducible() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let run = |seed: Option<u64>| {
            let mut c = DoubleMLConfig::default();
            c.seed = seed;
            run_double_ml(&y.view(), &d.view(), &x.view(), c).unwrap()
        };
        let (a, b, c) = (run(None), run(None), run(Some(DML_DEFAULT_SEED)));
        for r in [&b, &c] {
            assert_eq!(r.theta.to_bits(), a.theta.to_bits());
            assert_eq!(r.se.to_bits(), a.se.to_bits());
        }
        assert_eq!(a.seed, Some(42));
        assert!(!a.user_folds);
    }

    #[test]
    fn test_doubleml_rejects_invalid_inputs() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let run = |f: &dyn Fn(&mut DoubleMLConfig)| {
            let mut c = DoubleMLConfig::default();
            f(&mut c);
            run_double_ml(&y.view(), &d.view(), &x.view(), c)
        };
        spec_err(run(&|c| c.folds = Some(vec![1; 399])), "folds");
        spec_err(
            run(&|c| c.folds = Some(vec![1; 400])),
            "at least 2 distinct labels",
        );
        spec_err(run(&|c| c.n_folds = 1), "n_folds");
        spec_err(run(&|c| c.n_folds = 401), "n_folds");
        for t in [0.0, 0.5, -0.1, f64::NAN] {
            spec_err(run(&|c| c.trim = t), "trim");
        }
        for l in [-1.0, f64::NAN, f64::INFINITY] {
            spec_err(run(&|c| c.lambda = l), "lambda");
        }
        let mut y_nan = y.clone();
        y_nan[3] = f64::NAN;
        spec_err(
            run_double_ml(
                &y_nan.view(),
                &d.view(),
                &x.view(),
                DoubleMLConfig::default(),
            ),
            "y contains a non-finite value (row 4)",
        );
        let d_const = Array1::from_elem(400, 2.0);
        spec_err(
            run_double_ml(
                &y.view(),
                &d_const.view(),
                &x.view(),
                DoubleMLConfig::default(),
            ),
            "constant",
        );
        let x0 = Array2::<f64>::zeros((400, 0));
        let mut lasso = DoubleMLConfig::default();
        lasso.ml_method = MLMethod::Lasso;
        spec_err(
            run_double_ml(&y.view(), &d.view(), &x0.view(), lasso),
            "covariate",
        );
        let mut labels = vec![1usize; 400];
        labels[..5].fill(2);
        assert!(
            matches!(
                run(&|c| c.folds = Some(labels.clone())),
                Err(EconError::InsufficientData {
                    required: 8,
                    provided: 5,
                    ..
                })
            ),
            "fold 1 trains on 5 rows for 7 coefficients"
        );
    }

    #[test]
    fn test_doubleml_rejects_contradicting_treatment_type() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let mut c = DoubleMLConfig::default();
        c.treatment_type = TreatmentType::Binary;
        spec_err(
            run_double_ml(&y.view(), &d.view(), &x.view(), c),
            "treatment_type = Binary but d is not coded 0/1 (row 1 has",
        );
        // a 0/1 treatment declared continuous is accepted and reported as binary
        let db = col(&ds, "dbin");
        let r = run_double_ml(&y.view(), &db.view(), &x.view(), DoubleMLConfig::default()).unwrap();
        assert_eq!(r.treatment_type, TreatmentType::Binary);
    }

    /// coordinate descent at DML_GLMNET_THRESH: within 3e-11 (relative) of the exact
    /// minimiser's estimates
    const TIGHT: Tol = Tol::new(1e-9, 1e-12);
    /// glmnet 5.1's own output at thresh 1e-14 (cross-check)
    const GLMNET: Tol = Tol::new(1e-8, 1e-12);

    fn nuis<'a>(r: &'a DoubleMLResult, name: &str) -> &'a NuisanceFit {
        r.nuisance
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no nuisance {name}"))
    }

    /// PLR on the case data with the case folds.
    fn plr_case(ds: &Dataset, ml: MLMethod) -> DoubleMLResult {
        let (y, d, x) = (col(ds, "y"), col(ds, "dcont"), covariates(ds));
        run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            with_folds(DMLModelType::PLR, ml, case_folds(ds)),
        )
        .unwrap()
    }

    /// The PLR keys every learner reports, at the given tolerances.
    fn check_plr(case: &p2a_validation::Case, r: &DoubleMLResult, prefix: &str, est: [Tol; 3]) {
        let [t_theta, t_se, t_stat] = est;
        let nd = &r.nuisance_diagnostics;
        Check::new(case)
            .scalar(&format!("{prefix}_theta"), r.theta, t_theta)
            .scalar(&format!("{prefix}_se"), r.se, t_se)
            .scalar(&format!("{prefix}_rmse_l"), nd.outcome_rmse, t_stat)
            .scalar(&format!("{prefix}_rmse_m"), nd.treatment_rmse, t_stat)
            .scalar(&format!("{prefix}_r2_l"), nd.outcome_r2, t_stat)
            .scalar(&format!("{prefix}_r2_m"), nd.treatment_r2, t_stat)
            .scalar(&format!("{prefix}_jacobian"), r.jacobian, t_stat)
            .scalar(
                &format!("{prefix}_mean_resid_m"),
                nd.mean_residual_treatment,
                t_stat,
            )
            .vector(&format!("{prefix}_ci95"), &[r.ci_lower, r.ci_upper], t_stat)
            .scalar(&format!("{prefix}_p"), r.p_value, P_REL)
            .finish();
    }

    #[test]
    fn test_validate_doubleml_plr_ols() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let r = plr_case(&ds, MLMethod::OLS);
        let n = r.n_obs;
        check_plr(
            &case,
            &r,
            "plr_ols",
            [Tol::coef(n), Tol::se(n), Tol::stat(n)],
        );
        let nd = &r.nuisance_diagnostics;
        assert_eq!(nuis(&r, "l").rmse, nd.outcome_rmse);
        assert_eq!(nuis(&r, "m").n_eval, 400);
        assert!((nd.var_residual_treatment - r.jacobian).abs() <= 1e-15 * r.jacobian);
        assert_eq!(r.lambda, None);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_doubleml_plr_lasso() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let r = plr_case(&ds, MLMethod::Lasso);
        let nd = &r.nuisance_diagnostics;
        Check::new(&case)
            .scalar("plr_lasso_theta_tight", r.theta, TIGHT)
            .scalar("plr_lasso_se_tight", r.se, TIGHT)
            .scalar("plr_lasso_rmse_l_tight", nd.outcome_rmse, TIGHT)
            .scalar("plr_lasso_rmse_m_tight", nd.treatment_rmse, TIGHT)
            .finish();
        check_plr(&case, &r, "plr_lasso", [GLMNET, GLMNET, GLMNET]);
        assert!((r.theta - case.scalar("plr_ols_theta")).abs() > 1e-3);
        assert_eq!(r.lambda, Some(0.05));
        assert_eq!(nuis(&r, "m").learner, "lasso (glmnet, lambda = 0.05)");
    }

    #[test]
    fn test_validate_doubleml_plr_ridge() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let r = plr_case(&ds, MLMethod::Ridge);
        let n = r.n_obs;
        let nd = &r.nuisance_diagnostics;
        // the closed form is the exact minimiser: README tolerances against *_tight
        Check::new(&case)
            .scalar("plr_ridge_theta_tight", r.theta, Tol::coef(n))
            .scalar("plr_ridge_se_tight", r.se, Tol::se(n))
            .scalar("plr_ridge_rmse_l_tight", nd.outcome_rmse, Tol::stat(n))
            .scalar("plr_ridge_rmse_m_tight", nd.treatment_rmse, Tol::stat(n))
            .finish();
        check_plr(&case, &r, "plr_ridge", [GLMNET, GLMNET, GLMNET]);
        assert!((r.theta - case.scalar("plr_ols_theta")).abs() > 1e-4);
        assert_eq!(nuis(&r, "l").learner, "ridge (glmnet, lambda = 0.05)");
    }

    #[test]
    fn test_doubleml_learners_are_not_ols() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let fit = |ml| {
            run_double_ml(
                &y.view(),
                &d.view(),
                &x.view(),
                seeded(DMLModelType::PLR, ml, 5, 11),
            )
            .unwrap()
        };
        let (ols, lasso, ridge) = (
            fit(MLMethod::OLS),
            fit(MLMethod::Lasso),
            fit(MLMethod::Ridge),
        );
        for (a, b) in [(&ols, &lasso), (&ols, &ridge), (&lasso, &ridge)] {
            assert!((a.theta - b.theta).abs() > 1e-8, "{} {}", a.theta, b.theta);
        }
        assert!(
            (lasso.nuisance_diagnostics.outcome_rmse - ols.nuisance_diagnostics.outcome_rmse).abs()
                > 1e-8
        );
    }

    #[test]
    fn test_doubleml_nuisance_rmse_is_out_of_fold() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let n = y.len() as f64;
        let ym = y.sum() / n;
        let var_y = y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / n;
        for k in [5, 10] {
            let r = run_double_ml(
                &y.view(),
                &d.view(),
                &x.view(),
                seeded(DMLModelType::PLR, MLMethod::OLS, k, 11),
            )
            .unwrap();
            let nd = &r.nuisance_diagnostics;
            assert!(
                (1.20..1.32).contains(&nd.outcome_rmse),
                "K = {k}: {}",
                nd.outcome_rmse
            );
            assert!(
                (0.96..1.04).contains(&nd.treatment_rmse),
                "K = {k}: {}",
                nd.treatment_rmse
            );
            assert!((nd.outcome_r2 - (1.0 - nd.outcome_rmse.powi(2) / var_y)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_doubleml_aliased_covariate_is_reported() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let x7 = ndarray::concatenate(
            ndarray::Axis(1),
            &[x.view(), x.slice(ndarray::s![.., 0..1])],
        )
        .unwrap();
        let config = || seeded(DMLModelType::PLR, MLMethod::OLS, 5, 11);
        let a = run_double_ml(&y.view(), &d.view(), &x.view(), config()).unwrap();
        let b = run_double_ml(&y.view(), &d.view(), &x7.view(), config()).unwrap();
        assert!(
            Tol::coef(400).allows(b.theta, a.theta),
            "{} {}",
            b.theta,
            a.theta
        );
        assert!(Tol::se(400).allows(b.se, a.se));
        let aliased: Vec<&String> = b
            .warnings
            .iter()
            .filter(|w| w.contains("Coefficients not defined because of singularities: x7"))
            .collect();
        assert_eq!(aliased.len(), 2, "{:?}", b.warnings);
        assert!(aliased[0].starts_with("l = E[Y|X] by OLS, folds 1, 2, 3, 4, 5: "));
        assert!(aliased[1].starts_with("m = E[D|X] by OLS, folds 1, 2, 3, 4, 5: "));
        assert!(a.warnings.is_empty(), "{:?}", a.warnings);
    }

    #[test]
    fn test_doubleml_is_scale_equivariant() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let config = || seeded(DMLModelType::PLR, MLMethod::OLS, 5, 11);
        let a = run_double_ml(&y.view(), &d.view(), &x.view(), config()).unwrap();
        let (ys, dsc) = (&y * 1e6, &d * 1e-6);
        let b = run_double_ml(&ys.view(), &dsc.view(), &x.view(), config()).unwrap();
        let tol = Tol::new(1e-9, 0.0);
        let (na, nb) = (&a.nuisance_diagnostics, &b.nuisance_diagnostics);
        assert!(tol.allows(b.theta, a.theta * 1e12));
        assert!(tol.allows(b.se, a.se * 1e12));
        assert!(tol.allows(nb.outcome_rmse, na.outcome_rmse * 1e6));
        assert!(tol.allows(nb.treatment_rmse, na.treatment_rmse * 1e-6));
        assert!(tol.allows(b.jacobian, a.jacobian * 1e-12));
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }

    #[test]
    fn test_doubleml_display_names_learners_and_jacobian() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "y"), col(&ds, "dcont"), covariates(&ds));
        let r = run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            seeded(DMLModelType::PLR, MLMethod::Lasso, 5, 11),
        )
        .unwrap();
        let text = r.to_string();
        for needle in [
            "ML Method:  Lasso (glmnet, lambda = 0.05, standardized)",
            "Folds:      5 (random, seed 11)",
            "Nuisance models (out-of-fold predictions):",
            "lasso (glmnet, lambda = 0.05)",
            "Score Jacobian E[(D - m)^2]: ",
        ] {
            assert!(text.contains(needle), "{needle:?} not in\n{text}");
        }
        assert!(!text.contains("Var(D-m)"));
    }

    #[test]
    fn test_doubleml_ridge_closed_form_matches_glmnet() {
        use crate::regression::glmnet::{GlmnetConfig, glmnet, glmnet_predict};
        let ds = sim();
        let (y, x) = (col(&ds, "y"), covariates(&ds));
        // one fold of the case: train on folds 2..5, predict fold 1
        let folds = case_folds(&ds);
        let train: Vec<usize> = (0..400).filter(|&i| folds[i] != 1).collect();
        let test: Vec<usize> = (0..400).filter(|&i| folds[i] == 1).collect();
        let xtr = x.select(ndarray::Axis(0), &train);
        let xte = x.select(ndarray::Axis(0), &test);
        let ytr = y.select(ndarray::Axis(0), &train);
        for intercept in [true, false] {
            let fit = RidgeFit::new(xtr.view(), intercept).unwrap();
            let (b0, b) = fit.coefficients(ytr.view(), 0.05).unwrap();
            let ours = xte.dot(&b) + b0;
            let mut c = GlmnetConfig::default();
            c.alpha = 0.0;
            c.lambda = Some(vec![0.05]);
            c.intercept = intercept;
            c.thresh = 1e-14;
            let g = glmnet(xtr.view(), ytr.view(), &c).unwrap();
            let theirs = glmnet_predict(&g, xte.view(), 0.05).unwrap();
            let tol = Tol::new(1e-8, 1e-12);
            for (a, e) in ours.iter().zip(theirs.iter()) {
                assert!(tol.allows(*a, *e), "intercept = {intercept}: {a} vs {e}");
            }
        }
        // the dual (n x n) equals the primal (p x p) where p > n
        let sub = x.slice(ndarray::s![..30, ..]).to_owned();
        let wide = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                sub.view(),
                sub.mapv(|v| v * v).view(),
                sub.mapv(f64::sin).view(),
            ],
        )
        .unwrap();
        let wide = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                wide.view(),
                sub.mapv(f64::cos).view(),
                sub.mapv(|v| v.powi(3)).view(),
            ],
        )
        .unwrap();
        let wide = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                wide.view(),
                sub.mapv(|v| (0.5 * v).exp()).view(),
                sub.mapv(f64::tanh).view(),
            ],
        )
        .unwrap();
        assert!(wide.ncols() > 30);
        let yw = y.slice(ndarray::s![..30]).to_owned();
        let fit = RidgeFit::new(wide.view(), true).unwrap();
        let (yc, _, ys) = fit.response(yw.view()).unwrap();
        let primal = fit.solve_primal(&yc, 0.05 / ys).unwrap();
        let dual = fit.solve_dual(&yc, 0.05 / ys).unwrap();
        for (a, e) in primal.iter().zip(dual.iter()) {
            assert!(Tol::new(1e-10, 1e-14).allows(*a, *e), "{a} vs {e}");
        }
    }

    /// IRM on the case data with the case folds.
    fn irm_case(ds: &Dataset, ml: MLMethod) -> DoubleMLResult {
        let (y, d, x) = (col(ds, "yb"), col(ds, "dbin"), covariates(ds));
        run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            with_folds(DMLModelType::IRM, ml, case_folds(ds)),
        )
        .unwrap()
    }

    #[test]
    fn test_validate_doubleml_irm() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let r = irm_case(&ds, MLMethod::OLS);
        let n = r.n_obs;
        let (g0, g1, m) = (nuis(&r, "g0"), nuis(&r, "g1"), nuis(&r, "m"));
        let nd = &r.nuisance_diagnostics;
        let range = r.propensity_range.unwrap();
        let mean_psi_sq = r.scores.iter().map(|s| s * s).sum::<f64>() / n as f64;
        Check::new(&case)
            .scalar("irm_theta", r.theta, Tol::coef(n))
            .scalar("irm_se", r.se, Tol::se(n))
            .vector("irm_ps_range", &range, Tol::stat(n))
            .scalar("irm_rmse_g0", g0.rmse, Tol::stat(n))
            .scalar("irm_rmse_g1", g1.rmse, Tol::stat(n))
            .scalar("irm_r2_g0", g0.r2, Tol::stat(n))
            .scalar("irm_r2_g1", g1.r2, Tol::stat(n))
            .scalar("irm_rmse_m", m.rmse, Tol::stat(n))
            .scalar("irm_r2_m", m.r2, Tol::stat(n))
            .scalar("irm_rmse_m", nd.treatment_rmse, Tol::stat(n))
            .scalar("irm_r2_m", nd.treatment_r2, Tol::stat(n))
            .scalar("irm_rmse_outcome", nd.outcome_rmse, Tol::stat(n))
            .scalar("irm_r2_outcome", nd.outcome_r2, Tol::stat(n))
            .scalar("irm_mean_resid_m", nd.mean_residual_treatment, Tol::stat(n))
            .vector("irm_ci95", &[r.ci_lower, r.ci_upper], Tol::stat(n))
            .scalar("irm_p", r.p_value, P_REL)
            // what the old code printed as the Jacobian is the score variance
            .scalar("irm_mean_psi_sq", mean_psi_sq, Tol::se(n))
            .finish();
        assert_eq!(r.jacobian, 1.0);
        let rmse_m = case.scalar("irm_rmse_m");
        assert!(Tol::stat(n).allows(nd.var_residual_treatment, rmse_m * rmse_m));
        assert_eq!(g0.n_eval, 194);
        assert_eq!(m.learner, "logistic regression");
        assert_eq!(r.trim, Some(0.01));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_doubleml_irm_lasso() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let r = irm_case(&ds, MLMethod::Lasso);
        let n = r.n_obs;
        let g0 = nuis(&r, "g0");
        Check::new(&case)
            .scalar("irm_lasso_theta_tight", r.theta, TIGHT)
            .scalar("irm_lasso_se_tight", r.se, TIGHT)
            .scalar("irm_lasso_rmse_g0_tight", g0.rmse, TIGHT)
            .scalar("irm_lasso_theta", r.theta, GLMNET)
            .scalar("irm_lasso_se", r.se, GLMNET)
            .scalar("irm_lasso_rmse_g0", g0.rmse, GLMNET)
            // m stays logistic, identical to the OLS run
            .vector("irm_ps_range", &r.propensity_range.unwrap(), Tol::stat(n))
            .finish();
        assert_eq!(nuis(&r, "g1").learner, "lasso (glmnet, lambda = 0.05)");
        assert_eq!(nuis(&r, "m").learner, "logistic regression");
    }

    #[test]
    fn test_doubleml_irm_jacobian_is_not_the_score_variance() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "yb"), col(&ds, "dbin"), covariates(&ds));
        let r = run_double_ml(
            &y.view(),
            &d.view(),
            &x.view(),
            seeded(DMLModelType::IRM, MLMethod::OLS, 5, 11),
        )
        .unwrap();
        let nd = &r.nuisance_diagnostics;
        assert_eq!(r.jacobian, 1.0);
        assert!(
            (0.2..0.3).contains(&nd.var_residual_treatment),
            "{}",
            nd.var_residual_treatment
        );
        assert!((nd.var_residual_treatment - nd.treatment_rmse.powi(2)).abs() < 1e-12);
        let text = r.to_string();
        for needle in [
            "Score Jacobian: 1 (ATE score, psi_a = -1)",
            "P(D=1|X)",
            "logistic regression",
            "Propensity range: [",
            "Trimming:   propensities bounded to [0.01, 0.99]",
        ] {
            assert!(text.contains(needle), "{needle:?} not in\n{text}");
        }
        assert!(!text.contains("Var(D-m)"));
    }

    #[test]
    fn test_doubleml_irm_rejects_non_binary_treatment() {
        let ds = sim();
        let (y, d, x) = (col(&ds, "yb"), col(&ds, "dcont"), covariates(&ds));
        let mut c = DoubleMLConfig::default();
        c.model_type = DMLModelType::IRM;
        for needle in ["binary", "0/1", "row 1 has 1.3474565949"] {
            spec_err(
                run_double_ml(&y.view(), &d.view(), &x.view(), c.clone()),
                needle,
            );
        }
    }

    fn iv_data(ds: &Dataset, y: &str, d: &str, z: &str) -> [Array1<f64>; 3] {
        [col(ds, y), col(ds, d), col(ds, z)]
    }

    #[test]
    fn test_validate_doubleml_pliv() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let [y, d, z] = iv_data(&ds, "yz", "dz", "z");
        let x = covariates(&ds);
        let r = run_double_ml_iv(
            &y.view(),
            &d.view(),
            &z.view(),
            &x.view(),
            with_folds(DMLModelType::PLIV, MLMethod::OLS, case_folds(&ds)),
        )
        .unwrap();
        let n = r.n_obs;
        let s = Tol::stat(n);
        let nd = &r.nuisance_diagnostics;
        let (l, rr, m) = (nuis(&r, "l"), nuis(&r, "r"), nuis(&r, "m"));
        Check::new(&case)
            .scalar("pliv_theta", r.theta, Tol::coef(n))
            .scalar("pliv_se", r.se, Tol::se(n))
            .scalar("pliv_rmse_l", l.rmse, s)
            .scalar("pliv_rmse_r", rr.rmse, s)
            .scalar("pliv_rmse_m", m.rmse, s)
            .scalar("pliv_r2_l", l.r2, s)
            .scalar("pliv_r2_r", rr.r2, s)
            .scalar("pliv_r2_m", m.r2, s)
            .scalar("pliv_rmse_l", nd.outcome_rmse, s)
            .scalar("pliv_rmse_r", nd.treatment_rmse, s)
            .scalar("pliv_mean_resid_r", nd.mean_residual_treatment, s)
            .scalar("pliv_jacobian", r.jacobian, s)
            .vector("pliv_ci95", &[r.ci_lower, r.ci_upper], s)
            .scalar("pliv_p", r.p_value, P_REL)
            .finish();
        let text = r.to_string();
        assert!(
            text.contains("Score Jacobian E[(D - r)(Z - m)]: 0.8263"),
            "{text}"
        );
        assert!(text.contains("Model Type: Partially Linear IV Regression (PLIV)"));
        assert_eq!(r.propensity_range, None);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_doubleml_iivm() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let [y, d, z] = iv_data(&ds, "yzb", "db", "zb");
        let x = covariates(&ds);
        let r = run_double_ml_iv(
            &y.view(),
            &d.view(),
            &z.view(),
            &x.view(),
            with_folds(DMLModelType::IIVM, MLMethod::OLS, case_folds(&ds)),
        )
        .unwrap();
        let n = r.n_obs;
        let s = Tol::stat(n);
        let nd = &r.nuisance_diagnostics;
        let mut check = Check::new(&case);
        check
            .scalar("iivm_theta", r.theta, Tol::coef(n))
            .scalar("iivm_se", r.se, Tol::se(n))
            .scalar("iivm_jacobian", r.jacobian, s)
            .vector("iivm_ps_range", &r.propensity_range.unwrap(), s);
        for name in ["g0", "g1", "r0", "r1", "m"] {
            let f = nuis(&r, name);
            check
                .scalar(&format!("iivm_rmse_{name}"), f.rmse, s)
                .scalar(&format!("iivm_r2_{name}"), f.r2, s);
        }
        check
            .scalar("iivm_rmse_outcome", nd.outcome_rmse, s)
            .scalar("iivm_r2_outcome", nd.outcome_r2, s)
            .scalar("iivm_rmse_treatment", nd.treatment_rmse, s)
            .scalar("iivm_r2_treatment", nd.treatment_r2, s)
            .scalar("iivm_mean_resid_treatment", nd.mean_residual_treatment, s)
            .vector("iivm_ci95", &[r.ci_lower, r.ci_upper], s)
            .scalar("iivm_p", r.p_value, P_REL)
            .finish();
        assert_eq!(nuis(&r, "r0").n_eval, 199);
        assert_eq!(nuis(&r, "r1").learner, "logistic regression");
        let text = r.to_string();
        assert!(
            text.contains("Score Jacobian -E[psi_a] (first stage): 0.5485"),
            "{text}"
        );
        assert!(text.contains("Estimand:   LATE"));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_doubleml_iivm_one_sided() {
        let (case, ds) = crate::test_support::load_case("doubleml_models");
        let [y, d, z] = iv_data(&ds, "yzb", "db", "zb");
        let d1 = &d * &z;
        let x = covariates(&ds);
        let r = run_double_ml_iv(
            &y.view(),
            &d1.view(),
            &z.view(),
            &x.view(),
            with_folds(DMLModelType::IIVM, MLMethod::OLS, case_folds(&ds)),
        )
        .unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("iivm_onesided_theta", r.theta, Tol::coef(n))
            .scalar("iivm_onesided_se", r.se, Tol::se(n))
            .scalar("iivm_onesided_jacobian", r.jacobian, Tol::stat(n))
            .finish();
        let constant: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("no variation"))
            .collect();
        assert_eq!(constant.len(), 1, "{:?}", r.warnings);
        assert!(constant[0].starts_with(
            "r0 = P(D=1|Z=0,X) by logistic regression, folds 1, 2, 3, 4, 5: the training \
             outcome has no variation (all 0); predicted as the constant 0"
        ));
        assert_eq!(nuis(&r, "r0").rmse, 0.0);
    }

    #[test]
    fn test_doubleml_iv_models_need_an_instrument() {
        let ds = sim();
        let x = covariates(&ds);
        let [yz, dz, z] = iv_data(&ds, "yz", "dz", "z");
        let [_, db, zb] = iv_data(&ds, "yzb", "db", "zb");
        let cfg = |m| {
            let mut c = DoubleMLConfig::default();
            c.model_type = m;
            c
        };
        spec_err(
            run_double_ml(&yz.view(), &dz.view(), &x.view(), cfg(DMLModelType::PLIV)),
            "run_double_ml_iv",
        );
        spec_err(
            run_double_ml_iv(
                &yz.view(),
                &dz.view(),
                &z.view(),
                &x.view(),
                cfg(DMLModelType::PLR),
            ),
            "run_double_ml",
        );
        spec_err(
            run_double_ml_iv(
                &yz.view(),
                &db.view(),
                &z.view(),
                &x.view(),
                cfg(DMLModelType::IIVM),
            ),
            "binary instrument",
        );
        spec_err(
            run_double_ml_iv(
                &yz.view(),
                &dz.view(),
                &zb.view(),
                &x.view(),
                cfg(DMLModelType::IIVM),
            ),
            "binary treatment",
        );
        let ones = Array1::from_elem(400, 1.0);
        spec_err(
            run_double_ml_iv(
                &yz.view(),
                &dz.view(),
                &ones.view(),
                &x.view(),
                cfg(DMLModelType::PLIV),
            ),
            "instrument z is constant",
        );
        let short = z.slice(ndarray::s![..399]);
        spec_err(
            run_double_ml_iv(
                &yz.view(),
                &dz.view(),
                &short,
                &x.view(),
                cfg(DMLModelType::PLIV),
            ),
            "Instrument vector length (399)",
        );
    }
}
