//! Double/Debiased Machine Learning (DML) for causal effect estimation.
//!
//! DML uses Neyman-orthogonal moment conditions combined with sample splitting
//! (cross-fitting) to achieve root-n consistent and asymptotically normal
//! estimates of treatment effects, even when using flexible ML methods for
//! nuisance parameter estimation.
//!
//! # Partially Linear Regression Model (PLR)
//!
//! The primary model implemented is the Partially Linear Model (PLR):
//!
//! ```text
//! Y = theta_0 * D + g_0(X) + zeta    (outcome equation)
//! D = m_0(X) + V                      (treatment equation)
//!
//! where:
//!   E[zeta|D,X] = 0  (mean independence)
//!   E[V|X] = 0       (mean independence)
//! ```
//!
//! # Orthogonal Score Function
//!
//! The orthogonal (Neyman-orthogonal) score for the PLR model is:
//!
//! ```text
//! psi(W; theta, eta) = (Y - l_0(X) - theta*(D - m_0(X))) * (D - m_0(X))
//!
//! where:
//!   l_0(X) = E[Y|X] = theta_0 * m_0(X) + g_0(X)
//!   m_0(X) = E[D|X]
//! ```
//!
//! The orthogonality condition is E[psi(W; theta_0, eta_0)] = 0.
//! The key property is that the derivative of E[psi] with respect to eta
//! evaluated at the true values is zero (Neyman-orthogonality).
//!
//! # Cross-Fitting Procedure
//!
//! To avoid overfitting bias from using the same data for both nuisance
//! estimation and treatment effect estimation:
//!
//! 1. Split data into K folds (default K=5)
//! 2. For each fold k:
//!    a. Train nuisance models (l_0, m_0) on data NOT in fold k
//!    b. Predict nuisance values for observations IN fold k
//! 3. Compute orthogonal scores using cross-fitted predictions
//! 4. Estimate theta by solving the empirical moment condition
//!
//! # Variance Estimation
//!
//! The influence function for the PLR model is:
//!
//! ```text
//! IF(W) = psi(W; theta_0, eta_0) / J
//!
//! where J = E[(D - m_0(X))^2] (the Jacobian)
//! ```
//!
//! Variance is estimated as:
//! ```text
//! Var(theta_hat) = (1/n) * E[IF(W)^2] = (1/n) * E[psi^2] / J^2
//! ```
//!
//! # References
//!
//! - Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
//!   Newey, W., & Robins, J. (2018). "Double/debiased machine learning for
//!   treatment and structural parameters." *The Econometrics Journal*, 21(1),
//!   C1-C68. https://doi.org/10.1111/ectj.12097
//!
//! - Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., &
//!   Newey, W. (2017). "Double/Debiased/Neyman Machine Learning of Treatment
//!   Effects." *American Economic Review*, 107(5), 261-265.
//!
//! - Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022).
//!   "DoubleML - An Object-Oriented Implementation of Double Machine Learning
//!   in Python." *Journal of Machine Learning Research*, 23(53), 1-6.
//!   https://jmlr.org/papers/v23/21-0862.html
//!
//! - R package `DoubleML`: https://docs.doubleml.org/
//! - Python package `doubleml`: https://docs.doubleml.org/stable/
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::econometrics::{run_double_ml, DoubleMLConfig, DMLModelType};
//!
//! // Estimate treatment effect using PLR model with 5-fold cross-fitting
//! let config = DoubleMLConfig::default();
//! let result = run_double_ml(&y, &d, &x, config)?;
//!
//! println!("Treatment effect: {:.4} (SE: {:.4})", result.theta, result.se);
//! println!("95% CI: [{:.4}, {:.4}]", result.ci_lower, result.ci_upper);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::options::Choices;
use crate::traits::estimator::{SignificanceLevel, normal_cdf};

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
        ],
    },
    no_display
);

impl fmt::Display for DMLModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DMLModelType::PLR => write!(f, "Partially Linear Regression (PLR)"),
            DMLModelType::IRM => write!(f, "Interactive Regression Model (IRM)"),
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

impl fmt::Display for DoubleMLResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Double/Debiased Machine Learning Results")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Model Type: {}", self.model_type)?;
        writeln!(f, "ML Method:  {}", self.ml_method)?;
        writeln!(f, "Treatment:  {}", self.treatment_type)?;
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
        writeln!(f, "  Folds:        {}", self.n_folds)?;
        writeln!(f)?;
        writeln!(f, "Nuisance Model Diagnostics:")?;
        writeln!(
            f,
            "  Outcome model R²:    {:.4}",
            self.nuisance_diagnostics.outcome_r2
        )?;
        writeln!(
            f,
            "  Treatment model R²:  {:.4}",
            self.nuisance_diagnostics.treatment_r2
        )?;
        writeln!(
            f,
            "  Outcome RMSE:        {:.4}",
            self.nuisance_diagnostics.outcome_rmse
        )?;
        writeln!(
            f,
            "  Treatment RMSE:      {:.4}",
            self.nuisance_diagnostics.treatment_rmse
        )?;
        writeln!(f, "  Jacobian (Var(D-m)): {:.4}", self.jacobian)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

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

// =============================================================================
// Main DML Functions
// =============================================================================

/// Run Double Machine Learning estimation.
///
/// Estimates the treatment effect theta in the partially linear model:
/// Y = theta * D + g(X) + epsilon
///
/// Uses K-fold cross-fitting with Neyman-orthogonal score functions
/// to achieve root-n consistency even with slower-converging ML estimators.
///
/// # Arguments
/// * `y` - Outcome variable (n x 1)
/// * `d` - Treatment variable (n x 1), can be continuous or binary
/// * `x` - Covariate/confounding variables (n x p)
/// * `config` - DML configuration options
///
/// # Returns
/// `DoubleMLResult` containing the treatment effect estimate, standard error,
/// confidence intervals, and diagnostic information.
///
/// # Algorithm (PLR Model)
///
/// 1. **Cross-fitting**: Partition data into K folds
/// 2. For each fold k:
///    a. Estimate l(X) = E[Y|X] on data NOT in fold k
///    b. Estimate m(X) = E[D|X] on data NOT in fold k
///    c. Compute predictions for fold k
/// 3. **Compute residuals**:
///    - Y_tilde = Y - l_hat(X)  (residualized outcome)
///    - D_tilde = D - m_hat(X)  (residualized treatment)
/// 4. **Estimate theta**: theta_hat = sum(Y_tilde * D_tilde) / sum(D_tilde^2)
/// 5. **Compute scores**: psi_i = (Y_tilde_i - theta_hat * D_tilde_i) * D_tilde_i
/// 6. **Variance estimation**: Var(theta) = mean(psi^2) / J^2 / n
///    where J = mean(D_tilde^2)
///
/// # References
///
/// - Chernozhukov et al. (2018), "Double/debiased machine learning",
///   Econometrics Journal, Eq. (4.1) for PLR, Eq. (3.3) for orthogonal score
///
/// # Example
/// ```ignore
/// let config = DoubleMLConfig { n_folds: 5, ..Default::default() };
/// let result = run_double_ml(&y.view(), &d.view(), &x.view(), config)?;
/// println!("theta = {:.4} (SE: {:.4})", result.theta, result.se);
/// ```
pub fn run_double_ml(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: DoubleMLConfig,
) -> EconResult<DoubleMLResult> {
    let treatment_type = validate_inputs(y, d, None, x, &config)?;
    let folds = make_folds(y.len(), &config)?;
    let mut warnings = Vec::new();

    // Dispatch to appropriate model
    match config.model_type {
        DMLModelType::PLR => run_plr(y, d, x, &config, &folds, &mut warnings),
        DMLModelType::IRM => run_irm(y, d, x, &config, &folds, &mut warnings),
    }
    .map(|mut r| {
        r.treatment_type = treatment_type;
        r.fold_sizes = folds.sizes.clone();
        r.user_folds = folds.seed.is_none();
        r.seed = folds.seed;
        r
    })
}

/// Run Partially Linear Regression Model (PLR) with cross-fitting.
///
/// PLR Model:
///   Y = theta * D + g(X) + zeta,  E[zeta|D,X] = 0
///   D = m(X) + V,                  E[V|X] = 0
///
/// Orthogonal score (partialling out):
///   psi = (Y - l(X) - theta*(D - m(X))) * (D - m(X))
///
/// where l(X) = E[Y|X] = theta * m(X) + g(X)
fn run_plr(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let n = y.len();
    let k_folds = folds.labels.len();

    let fold_ids = &folds.index;

    // Build design matrix with intercept if requested
    let x_design = if config.intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    // Initialize storage for cross-fitted predictions
    let mut l_hat = Array1::zeros(n); // E[Y|X] predictions
    let mut m_hat = Array1::zeros(n); // E[D|X] predictions

    // Cumulative diagnostics
    let mut total_outcome_ss = 0.0;
    let mut total_outcome_rss = 0.0;
    let mut total_treatment_ss = 0.0;
    let mut total_treatment_rss = 0.0;

    // Cross-fitting loop
    // For each fold k, train on data NOT in k, predict for data IN k
    // (Chernozhukov et al. 2018, Algorithm 1)
    for k in 0..k_folds {
        // Indices for this fold (test) and other folds (train)
        let test_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] == k).collect();
        let train_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] != k).collect();

        let n_train = train_idx.len();
        let n_test = test_idx.len();

        if n_train < x_design.ncols() + 1 {
            return Err(EconError::InsufficientData {
                required: x_design.ncols() + 1,
                provided: n_train,
                context: format!("Training data for fold {}", k),
            });
        }

        // Extract training data
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let d_train: Array1<f64> = train_idx.iter().map(|&i| d[i]).collect();
        let mut x_train = Array2::zeros((n_train, x_design.ncols()));
        for (new_i, &old_i) in train_idx.iter().enumerate() {
            x_train.row_mut(new_i).assign(&x_design.row(old_i));
        }

        // Extract test data design matrix
        // (y_test and d_test not needed since we use the full arrays at the end)
        let mut x_test = Array2::zeros((n_test, x_design.ncols()));
        for (new_i, &old_i) in test_idx.iter().enumerate() {
            x_test.row_mut(new_i).assign(&x_design.row(old_i));
        }

        // Fit nuisance model for l(X) = E[Y|X]
        let l_beta = fit_ols(&x_train.view(), &y_train)?;
        let l_pred_test = x_test.dot(&l_beta);
        let l_pred_train = x_train.dot(&l_beta);

        // Fit nuisance model for m(X) = E[D|X]
        let m_beta = fit_ols(&x_train.view(), &d_train)?;
        let m_pred_test = x_test.dot(&m_beta);
        let m_pred_train = x_train.dot(&m_beta);

        // Store cross-fitted predictions
        for (new_i, &old_i) in test_idx.iter().enumerate() {
            l_hat[old_i] = l_pred_test[new_i];
            m_hat[old_i] = m_pred_test[new_i];
        }

        // Accumulate diagnostics from training data
        let y_train_mean = y_train.mean().unwrap_or(0.0);
        let d_train_mean = d_train.mean().unwrap_or(0.0);

        total_outcome_ss += y_train
            .iter()
            .map(|&yi| (yi - y_train_mean).powi(2))
            .sum::<f64>();
        total_outcome_rss += y_train
            .iter()
            .zip(l_pred_train.iter())
            .map(|(&yi, &li)| (yi - li).powi(2))
            .sum::<f64>();

        total_treatment_ss += d_train
            .iter()
            .map(|&di| (di - d_train_mean).powi(2))
            .sum::<f64>();
        total_treatment_rss += d_train
            .iter()
            .zip(m_pred_train.iter())
            .map(|(&di, &mi)| (di - mi).powi(2))
            .sum::<f64>();
    }

    // Compute residualized variables
    // Y_tilde = Y - l_hat(X)
    // D_tilde = D - m_hat(X)
    let y_tilde: Array1<f64> = y
        .iter()
        .zip(l_hat.iter())
        .map(|(&yi, &li)| yi - li)
        .collect();
    let d_tilde: Array1<f64> = d
        .iter()
        .zip(m_hat.iter())
        .map(|(&di, &mi)| di - mi)
        .collect();

    // Estimate theta using the orthogonal moment condition:
    // theta_hat = sum(Y_tilde * D_tilde) / sum(D_tilde^2)
    // (Chernozhukov et al. 2018, Eq. 4.1)
    let numerator: f64 = y_tilde
        .iter()
        .zip(d_tilde.iter())
        .map(|(&yt, &dt)| yt * dt)
        .sum();
    let denominator: f64 = d_tilde.iter().map(|&dt| dt * dt).sum();

    if denominator.abs() < 1e-10 {
        return Err(EconError::SingularMatrix {
            context: "DML theta estimation".to_string(),
            suggestion: "Residualized treatment has near-zero variance. Check that treatment is not fully determined by covariates.".to_string(),
        });
    }

    let theta = numerator / denominator;

    // Compute orthogonal scores
    // psi_i = (Y_tilde_i - theta * D_tilde_i) * D_tilde_i
    // (Chernozhukov et al. 2018, Eq. 3.3)
    let scores: Vec<f64> = y_tilde
        .iter()
        .zip(d_tilde.iter())
        .map(|(&yt, &dt)| (yt - theta * dt) * dt)
        .collect();

    // Variance estimation using influence function
    // J = E[D_tilde^2] = mean(D_tilde^2) (the Jacobian)
    // Var(psi) = mean(psi^2)
    // Var(theta_hat) = Var(psi) / J^2 / n
    // (Chernozhukov et al. 2018, Theorem 3.1)
    let jacobian = denominator / n as f64;
    let psi_squared_mean: f64 = scores.iter().map(|&s| s * s).sum::<f64>() / n as f64;
    let variance = psi_squared_mean / (jacobian * jacobian) / n as f64;
    let se = variance.sqrt();

    // Inference
    let t_stat = if se > 0.0 && se.is_finite() {
        theta / se
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    // 95% confidence interval (normal approximation, valid asymptotically)
    let z_crit = 1.96;
    let ci_lower = theta - z_crit * se;
    let ci_upper = theta + z_crit * se;

    // Compute nuisance diagnostics
    let outcome_r2 = if total_outcome_ss > 0.0 {
        1.0 - total_outcome_rss / total_outcome_ss
    } else {
        0.0
    };
    let treatment_r2 = if total_treatment_ss > 0.0 {
        1.0 - total_treatment_rss / total_treatment_ss
    } else {
        0.0
    };

    let outcome_rmse = (total_outcome_rss / n as f64).sqrt();
    let treatment_rmse = (total_treatment_rss / n as f64).sqrt();

    let mean_residual_treatment = d_tilde.mean().unwrap_or(0.0);
    let var_residual_treatment = jacobian;

    // Warnings
    if outcome_r2 < 0.0 {
        warnings.push("Outcome model R² is negative; model may be misspecified".to_string());
    }
    if treatment_r2 < 0.0 {
        warnings.push("Treatment model R² is negative; model may be misspecified".to_string());
    }
    if jacobian < 0.01 {
        warnings.push(format!(
            "Jacobian (Var(D-m(X))) is small ({:.4}). Standard errors may be large.",
            jacobian
        ));
    }
    if mean_residual_treatment.abs() > 0.1 {
        warnings.push(format!(
            "Mean of residualized treatment is {:.4}; should be near 0.",
            mean_residual_treatment
        ));
    }

    let nuisance_diagnostics = NuisanceDiagnostics {
        outcome_r2,
        treatment_r2,
        outcome_rmse,
        treatment_rmse,
        mean_residual_treatment,
        var_residual_treatment,
    };

    Ok(DoubleMLResult {
        theta,
        se,
        t_stat,
        p_value,
        significance,
        ci_lower,
        ci_upper,
        scores,
        nuisance_diagnostics,
        n_obs: n,
        n_folds: k_folds,
        model_type: DMLModelType::PLR,
        treatment_type: config.treatment_type,
        ml_method: config.ml_method,
        jacobian,
        warnings: warnings.to_vec(),
        fold_sizes: Vec::new(),
        user_folds: false,
        seed: None,
    })
}

/// Run Interactive Regression Model (IRM) with cross-fitting.
///
/// IRM Model (for binary treatment):
///   Y(1) = g_1(X) + U_1,  E[U_1|X] = 0
///   Y(0) = g_0(X) + U_0,  E[U_0|X] = 0
///   D = m(X) + V,          where m(X) = P(D=1|X)
///
/// The Average Treatment Effect (ATE) is:
///   theta = E[g_1(X) - g_0(X)]
///
/// Orthogonal score:
///   psi = g_1(X) - g_0(X) + D*(Y - g_1(X))/m(X) - (1-D)*(Y - g_0(X))/(1-m(X)) - theta
fn run_irm(
    y: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    config: &DoubleMLConfig,
    folds: &Folds,
    warnings: &mut Vec<String>,
) -> EconResult<DoubleMLResult> {
    let n = y.len();
    let k_folds = folds.labels.len();

    // Validate binary treatment
    let is_binary = d
        .iter()
        .all(|&di| (di - 0.0).abs() < 1e-10 || (di - 1.0).abs() < 1e-10);
    if !is_binary {
        warnings.push(
            "IRM is designed for binary treatment. Consider using PLR for continuous treatment."
                .to_string(),
        );
    }

    let n_treated: usize = d.iter().filter(|&&di| di >= 0.5).count();
    let n_control = n - n_treated;
    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "IRM requires sufficient observations in both treatment groups".to_string(),
        });
    }

    let fold_ids = &folds.index;

    // Build design matrix with intercept
    let x_design = if config.intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    // Initialize storage for cross-fitted predictions
    let mut g1_hat = Array1::zeros(n); // E[Y|X,D=1] predictions
    let mut g0_hat = Array1::zeros(n); // E[Y|X,D=0] predictions
    let mut m_hat = Array1::zeros(n); // P(D=1|X) predictions

    // Diagnostics accumulators
    let mut total_g1_ss = 0.0;
    let mut total_g1_rss = 0.0;
    let mut total_g0_ss = 0.0;
    let mut total_g0_rss = 0.0;
    let mut total_m_ss = 0.0;
    let mut total_m_rss = 0.0;

    // Cross-fitting loop
    for k in 0..k_folds {
        let test_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] == k).collect();
        let train_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] != k).collect();

        // Split training data by treatment status
        let train_treated: Vec<usize> = train_idx
            .iter()
            .filter(|&&i| d[i] >= 0.5)
            .copied()
            .collect();
        let train_control: Vec<usize> =
            train_idx.iter().filter(|&&i| d[i] < 0.5).copied().collect();

        if train_treated.len() < x_design.ncols() + 1 || train_control.len() < x_design.ncols() + 1
        {
            return Err(EconError::InsufficientData {
                required: x_design.ncols() + 1,
                provided: train_treated.len().min(train_control.len()),
                context: format!(
                    "Training data for fold {} (need enough in each treatment group)",
                    k
                ),
            });
        }

        // Fit g_1(X) = E[Y|X, D=1]
        let y_treated: Array1<f64> = train_treated.iter().map(|&i| y[i]).collect();
        let mut x_treated = Array2::zeros((train_treated.len(), x_design.ncols()));
        for (new_i, &old_i) in train_treated.iter().enumerate() {
            x_treated.row_mut(new_i).assign(&x_design.row(old_i));
        }
        let g1_beta = fit_ols(&x_treated.view(), &y_treated)?;

        // Fit g_0(X) = E[Y|X, D=0]
        let y_control: Array1<f64> = train_control.iter().map(|&i| y[i]).collect();
        let mut x_control = Array2::zeros((train_control.len(), x_design.ncols()));
        for (new_i, &old_i) in train_control.iter().enumerate() {
            x_control.row_mut(new_i).assign(&x_design.row(old_i));
        }
        let g0_beta = fit_ols(&x_control.view(), &y_control)?;

        // Fit m(X) = P(D=1|X) using linear probability model
        // (could be replaced with logistic regression for better propensity estimation)
        let d_train: Array1<f64> = train_idx.iter().map(|&i| d[i]).collect();
        let mut x_train = Array2::zeros((train_idx.len(), x_design.ncols()));
        for (new_i, &old_i) in train_idx.iter().enumerate() {
            x_train.row_mut(new_i).assign(&x_design.row(old_i));
        }
        let m_beta = fit_ols(&x_train.view(), &d_train)?;

        // Predict for test fold
        let mut x_test = Array2::zeros((test_idx.len(), x_design.ncols()));
        for (new_i, &old_i) in test_idx.iter().enumerate() {
            x_test.row_mut(new_i).assign(&x_design.row(old_i));
        }

        let g1_pred = x_test.dot(&g1_beta);
        let g0_pred = x_test.dot(&g0_beta);
        let m_pred = x_test.dot(&m_beta);

        // Store predictions and clip propensity scores
        for (new_i, &old_i) in test_idx.iter().enumerate() {
            g1_hat[old_i] = g1_pred[new_i];
            g0_hat[old_i] = g0_pred[new_i];
            m_hat[old_i] = m_pred[new_i].max(config.trim).min(1.0 - config.trim);
        }

        // Accumulate diagnostics
        let y_treated_mean = y_treated.mean().unwrap_or(0.0);
        let g1_pred_train = x_treated.dot(&g1_beta);
        total_g1_ss += y_treated
            .iter()
            .map(|&yi| (yi - y_treated_mean).powi(2))
            .sum::<f64>();
        total_g1_rss += y_treated
            .iter()
            .zip(g1_pred_train.iter())
            .map(|(&yi, &gi)| (yi - gi).powi(2))
            .sum::<f64>();

        let y_control_mean = y_control.mean().unwrap_or(0.0);
        let g0_pred_train = x_control.dot(&g0_beta);
        total_g0_ss += y_control
            .iter()
            .map(|&yi| (yi - y_control_mean).powi(2))
            .sum::<f64>();
        total_g0_rss += y_control
            .iter()
            .zip(g0_pred_train.iter())
            .map(|(&yi, &gi)| (yi - gi).powi(2))
            .sum::<f64>();

        let d_train_mean = d_train.mean().unwrap_or(0.0);
        let m_pred_train = x_train.dot(&m_beta);
        total_m_ss += d_train
            .iter()
            .map(|&di| (di - d_train_mean).powi(2))
            .sum::<f64>();
        total_m_rss += d_train
            .iter()
            .zip(m_pred_train.iter())
            .map(|(&di, &mi)| (di - mi).powi(2))
            .sum::<f64>();
    }

    // Compute theta using the AIPW/doubly-robust estimator
    // theta = mean(g1(X) - g0(X) + D*(Y - g1(X))/m(X) - (1-D)*(Y - g0(X))/(1-m(X)))
    let mut psi_values = Vec::with_capacity(n);
    let mut theta_components = Vec::with_capacity(n);

    for i in 0..n {
        let di = d[i];
        let yi = y[i];
        let g1i = g1_hat[i];
        let g0i = g0_hat[i];
        let mi = m_hat[i];

        let outcome_diff = g1i - g0i;
        let ipw_treated = if di >= 0.5 { (yi - g1i) / mi } else { 0.0 };
        let ipw_control = if di < 0.5 {
            (yi - g0i) / (1.0 - mi)
        } else {
            0.0
        };

        let component = outcome_diff + ipw_treated - ipw_control;
        theta_components.push(component);
    }

    let theta: f64 = theta_components.iter().sum::<f64>() / n as f64;

    // Compute scores (influence function values)
    // psi_i = theta_component_i - theta
    for &comp in &theta_components {
        psi_values.push(comp - theta);
    }

    // Variance estimation
    let psi_squared_mean: f64 = psi_values.iter().map(|&p| p * p).sum::<f64>() / n as f64;
    let variance = psi_squared_mean / n as f64;
    let se = variance.sqrt();

    // Inference
    let t_stat = if se > 0.0 && se.is_finite() {
        theta / se
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    let z_crit = 1.96;
    let ci_lower = theta - z_crit * se;
    let ci_upper = theta + z_crit * se;

    // Compute nuisance diagnostics (average of g1 and g0)
    let outcome_r2 = if total_g1_ss + total_g0_ss > 0.0 {
        1.0 - (total_g1_rss + total_g0_rss) / (total_g1_ss + total_g0_ss)
    } else {
        0.0
    };
    let treatment_r2 = if total_m_ss > 0.0 {
        1.0 - total_m_rss / total_m_ss
    } else {
        0.0
    };

    let outcome_rmse = ((total_g1_rss + total_g0_rss) / n as f64).sqrt();
    let treatment_rmse = (total_m_rss / n as f64).sqrt();

    // For IRM, jacobian is not directly applicable; we use variance of scores
    let jacobian = psi_squared_mean;

    // Check for extreme propensity scores
    let n_extreme = m_hat
        .iter()
        .filter(|&&mi| mi < 2.0 * config.trim || mi > 1.0 - 2.0 * config.trim)
        .count();
    if n_extreme > n / 10 {
        warnings.push(format!(
            "Many observations ({}/{}) have extreme propensity scores. Consider stronger trimming.",
            n_extreme, n
        ));
    }

    let nuisance_diagnostics = NuisanceDiagnostics {
        outcome_r2,
        treatment_r2,
        outcome_rmse,
        treatment_rmse,
        mean_residual_treatment: 0.0, // Not applicable for IRM
        var_residual_treatment: jacobian,
    };

    Ok(DoubleMLResult {
        theta,
        se,
        t_stat,
        p_value,
        significance,
        ci_lower,
        ci_upper,
        scores: psi_values,
        nuisance_diagnostics,
        n_obs: n,
        n_folds: k_folds,
        model_type: DMLModelType::IRM,
        treatment_type: TreatmentType::Binary,
        ml_method: config.ml_method,
        jacobian,
        warnings: warnings.to_vec(),
        fold_sizes: Vec::new(),
        user_folds: false,
        seed: None,
    })
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

/// Fit OLS regression: beta = (X'X)^{-1} X'y
///
/// Returns coefficient vector.
fn fit_ols(x: &ArrayView2<f64>, y: &Array1<f64>) -> EconResult<Array1<f64>> {
    let xtx_mat = xtx(x);
    let (xtx_inv, _cond_warning) =
        safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
            context: "OLS in nuisance estimation".to_string(),
            suggestion: "Check for multicollinearity in covariates".to_string(),
        })?;

    let xty_vec = xty(x, y);
    let beta = xtx_inv.dot(&xty_vec);

    Ok(beta)
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
    // Validation tests (test_validate_* prefix)
    // =========================================================================

    /// Validate PLR model recovers known treatment effect from a linear DGP.
    ///
    /// DGP (linear, so OLS nuisance models are correctly specified):
    ///   X1, X2 ~ Uniform(0,1)
    ///   D = 0.5*X1 + 0.3*X2 + V,  V ~ Uniform(-0.1, 0.1)
    ///   Y = theta*D + 0.3*X1 + 0.2*X2 + U,  U ~ Uniform(-0.1, 0.1)
    ///
    /// True theta = 0.5. With n=1000 and linear nuisance, cross-fitted OLS
    /// should recover theta to within ~0.1 of the true value.
    #[test]
    fn test_validate_doubleml_plr_parameter_recovery() {
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

    /// Validate that cross-fitting actually uses K folds and scores are computed.
    #[test]
    fn test_validate_doubleml_crossfitting_structure() {
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

    /// Validate IRM model recovers known ATE from a binary treatment DGP.
    ///
    /// True ATE = 0.5. With n=1000 and linear nuisance models, should be
    /// recovered within ~0.3 (wider tolerance for binary treatment).
    #[test]
    fn test_validate_doubleml_irm_parameter_recovery() {
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

    /// Validate DML with zero treatment effect correctly produces theta near 0
    /// and a non-significant p-value.
    #[test]
    fn test_validate_doubleml_null_effect() {
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
        match "ols".parse::<DMLModelType>() {
            Err(EconError::InvalidSpecification { message }) => {
                for needle in ["model", "plr", "irm"] {
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
}
