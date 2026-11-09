//! Average Marginal Effects (AME) computation for regression models.
//!
//! Provides computation of marginal effects for linear and nonlinear regression models.
//! For linear models (OLS), marginal effects equal the coefficients. For nonlinear
//! models (Logit, Probit, Poisson), marginal effects depend on covariate values and
//! are typically averaged across observations (Average Marginal Effects, AME).
//!
//! # Mathematical Background
//!
//! ## Linear Models (OLS)
//!
//! For E[y|X] = X'beta, the marginal effect is simply:
//!
//! dE[y|X]/dx_j = beta_j (constant for all observations)
//!
//! ## Logit Model
//!
//! For P(y=1|X) = Lambda(X'beta) where Lambda is the logistic CDF:
//!
//! dP(y=1|X)/dx_j = beta_j * Lambda(X'beta) * (1 - Lambda(X'beta)) = beta_j * lambda(X'beta)
//!
//! where lambda(z) = Lambda(z) * (1 - Lambda(z)) is the logistic PDF.
//!
//! ## Probit Model
//!
//! For P(y=1|X) = Phi(X'beta) where Phi is the standard normal CDF:
//!
//! dP(y=1|X)/dx_j = beta_j * phi(X'beta)
//!
//! where phi is the standard normal PDF.
//!
//! ## Average Marginal Effects (AME)
//!
//! AME_j = (1/n) * sum_{i=1}^n dE[y|X_i]/dx_j
//!
//! For Logit/Probit: AME_j = (1/n) * sum_{i=1}^n beta_j * f(X_i'beta)
//!
//! where f is the PDF of the link function.
//!
//! ## Standard Errors via Delta Method
//!
//! SE(AME_j) = sqrt(G_j' * Var(beta) * G_j)
//!
//! where G_j is the gradient of AME_j with respect to beta.
//!
//! For Logit: G_j\[k\] = (1/n) * sum_i (d/dbeta_k)\[beta_j * lambda(X_i'beta)\]
//!           = (1/n) * sum_i [I(j=k) * lambda(z_i) + beta_j * lambda'(z_i) * x_ik]
//!
//! where lambda'(z) = lambda(z) * (1 - 2*Lambda(z))
//!
//! # References
//!
//! - Bartus, T. (2005). Estimation of marginal effects using margeff.
//!   *The Stata Journal*, 5(3), 309-329.
//!
//! - Cameron, A.C., & Trivedi, P.K. (2005). *Microeconometrics: Methods and Applications*.
//!   Cambridge University Press. Chapter 15.
//!
//! - Greene, W.H. (2018). *Econometric Analysis* (8th ed.). Pearson. Chapter 14.
//!
//! - Leeper, T.J. (2021). Margins: Marginal effects for model objects. R package.
//!   <https://cran.r-project.org/package=margins>
//!
//! - Arel-Bundock, V. (2023). marginaleffects: Predictions, Comparisons, Slopes,
//!   Marginal Means, and Hypothesis Tests. R package.
//!   <https://vincentarelbundock.github.io/marginaleffects/>
//!
//! R equivalent: `marginaleffects::avg_slopes()`, `margins::margins()`

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::econometrics::{DiscreteModelType, DiscreteResult, run_logit, run_probit};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::regression::ols::{CovarianceType, OlsResult, run_ols};
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{
    SignificanceLevel, logistic_cdf, logistic_pdf, normal_cdf, normal_p_value_two_sided,
    normal_pdf, z_critical,
};

/// Type of model for marginal effects computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Ordinary Least Squares
    Ols,
    /// Logistic regression
    Logit,
    /// Probit regression
    Probit,
    /// Poisson regression (count data)
    Poisson,
    /// Negative binomial regression
    NegBin,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::Ols => write!(f, "OLS"),
            ModelType::Logit => write!(f, "Logit"),
            ModelType::Probit => write!(f, "Probit"),
            ModelType::Poisson => write!(f, "Poisson"),
            ModelType::NegBin => write!(f, "Negative Binomial"),
        }
    }
}

/// A single marginal effect with its statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginalEffect {
    /// Variable name
    pub variable: String,
    /// Point estimate of marginal effect (same as dy_dx)
    pub estimate: f64,
    /// Standard error of marginal effect
    pub std_error: f64,
    /// Z-value (estimate / std_error)
    pub z_value: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Lower bound of 95% confidence interval
    pub ci_lower: f64,
    /// Upper bound of 95% confidence interval
    pub ci_upper: f64,
    /// The marginal effect value (dE[y]/dx)
    pub dy_dx: f64,
    /// Significance level indicator
    pub significance: SignificanceLevel,
}

impl fmt::Display for MarginalEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<20} {:>10.4} {:>10.4} {:>8.3} {:>8.4} [{:>8.4}, {:>8.4}]{}",
            self.variable,
            self.estimate,
            self.std_error,
            self.z_value,
            self.p_value,
            self.ci_lower,
            self.ci_upper,
            self.significance.stars()
        )
    }
}

impl MarginalEffect {
    /// The row of a regressor whose effect is not defined (aliased): every number NaN.
    fn undefined(variable: &str) -> Self {
        MarginalEffect {
            variable: variable.to_string(),
            estimate: f64::NAN,
            std_error: f64::NAN,
            z_value: f64::NAN,
            p_value: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            dy_dx: f64::NAN,
            significance: SignificanceLevel::from_p_value(f64::NAN),
        }
    }
}

/// "Marginal effects not defined for aliased regressors: x3", or `None` for none.
fn aliased_warning(names: &[String]) -> Option<String> {
    (!names.is_empty()).then(|| {
        format!(
            "Marginal effects not defined for aliased regressors: {}",
            names.join(", ")
        )
    })
}

/// Result from marginal effects computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginalEffectsResult {
    /// Average marginal effects (AME) for each variable
    pub average_marginal: Vec<MarginalEffect>,
    /// Marginal effects at the mean (MEM) for each variable
    pub at_means: Option<Vec<MarginalEffect>>,
    /// All individual marginal effects (for each observation)
    /// Organized as: effects[var_idx][obs_idx]
    #[serde(skip)]
    pub effects_by_observation: Option<Vec<Array1<f64>>>,
    /// Model type used
    pub model_type: ModelType,
    /// Number of observations
    pub n_obs: usize,
    /// Variable names
    pub variables: Vec<String>,
    /// Whether intercept was included
    pub has_intercept: bool,
    /// Notes on the computation (aliased regressors whose effects are not defined)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for MarginalEffectsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for MarginalEffectsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Average Marginal Effects ({})", self.model_type)?;
        writeln!(f, "{}", "=".repeat(80))?;
        writeln!(f, "N = {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>8} {:>8}   {:>19}",
            "Variable", "dy/dx", "Std.Err.", "z", "P>|z|", "95% CI"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        for me in &self.average_marginal {
            writeln!(f, "{}", me)?;
        }

        writeln!(f, "{}", "-".repeat(80))?;

        if let Some(ref mem) = self.at_means {
            writeln!(f)?;
            writeln!(f, "Marginal Effects at the Mean:")?;
            writeln!(f, "{}", "-".repeat(80))?;
            for me in mem {
                writeln!(f, "{}", me)?;
            }
            writeln!(f, "{}", "-".repeat(80))?;
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 + 0.1")?;
        write!(f, "{}", notes_text(self))?;

        Ok(())
    }
}

/// Result from contrasts computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastsResult {
    /// Variable being contrasted
    pub variable: String,
    /// Contrast type (e.g., "mean" for average)
    pub contrast_type: String,
    /// Contrast values (value2 - value1)
    pub contrasts: Vec<ContrastEffect>,
    /// Number of observations
    pub n_obs: usize,
}

/// A single contrast effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastEffect {
    /// First value (reference)
    pub value1: f64,
    /// Second value
    pub value2: f64,
    /// Contrast estimate (effect at value2 - effect at value1)
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// Z-value
    pub z_value: f64,
    /// P-value
    pub p_value: f64,
}

// =============================================================================
// OLS Marginal Effects
// =============================================================================

/// Compute marginal effects from OLS regression.
///
/// For OLS, marginal effects are simply the coefficients since
/// dE[y|X]/dx_j = beta_j (constant).
///
/// # Arguments
/// * `ols_result` - Result from `run_ols`
///
/// # Returns
/// `MarginalEffectsResult` where AME equals the coefficients.
///
/// # Example
/// ```ignore
/// let ols = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::HC1)?;
/// let me = marginal_effects_ols(&ols)?;
/// println!("{}", me);
/// ```
pub fn marginal_effects_ols(ols_result: &OlsResult) -> EconResult<MarginalEffectsResult> {
    let n_obs = ols_result.n_obs;
    let variables = ols_result.variable_names.clone();
    let has_intercept = variables
        .first()
        .map(|s| s == "(Intercept)")
        .unwrap_or(false);

    let mut average_marginal = Vec::new();

    for coef in &ols_result.coefficients {
        let me = MarginalEffect {
            variable: coef.name.clone(),
            estimate: coef.estimate,
            std_error: coef.std_error,
            z_value: coef.t_value,
            p_value: coef.p_value,
            ci_lower: coef.ci_lower_95,
            ci_upper: coef.ci_upper_95,
            dy_dx: coef.estimate,
            significance: coef.significance,
        };
        average_marginal.push(me);
    }

    // For OLS, MEM = AME (constant marginal effects)
    let at_means = Some(average_marginal.clone());
    let aliased: Vec<String> = ols_result
        .coefficients
        .iter()
        .filter(|c| !c.estimate.is_finite())
        .map(|c| c.name.clone())
        .collect();

    Ok(MarginalEffectsResult {
        average_marginal,
        at_means,
        effects_by_observation: None, // All observations have same ME
        model_type: ModelType::Ols,
        n_obs,
        variables,
        has_intercept,
        warnings: aliased_warning(&aliased).into_iter().collect(),
    })
}

// =============================================================================
// Discrete Model (Logit/Probit) Marginal Effects
// =============================================================================

/// Compute marginal effects from discrete choice model (Logit/Probit).
///
/// For nonlinear models, marginal effects vary by observation:
/// - Logit: dP/dx_j = beta_j * Lambda(X'beta) * (1 - Lambda(X'beta))
/// - Probit: dP/dx_j = beta_j * phi(X'beta)
///
/// # Arguments
/// * `discrete_result` - Result from `run_logit` or `run_probit`
/// * `dataset` - The original dataset (needed to compute observation-specific effects)
/// * `x_cols` - Names of the independent variables (same as used in estimation)
///
/// # Returns
/// `MarginalEffectsResult` with both AME and MEM.
///
/// # Example
/// ```ignore
/// let logit = run_logit(&dataset, "y", &["x1", "x2"])?;
/// let me = marginal_effects_discrete(&logit, &dataset, &["x1", "x2"])?;
/// println!("{}", me);
/// ```
pub fn marginal_effects_discrete(
    discrete_result: &DiscreteResult,
    dataset: &Dataset,
    x_cols: &[&str],
) -> EconResult<MarginalEffectsResult> {
    let n_obs = discrete_result.n_obs;
    let model_type = match discrete_result.model_type {
        DiscreteModelType::Logit => ModelType::Logit,
        DiscreteModelType::Probit => ModelType::Probit,
    };

    // The design matrix on the rows the model used: those with no missing value in the
    // outcome or a regressor, as run_logit/run_probit drop them.
    let dep_var = discrete_result.dep_var.as_str();
    let cols = ColumnRequest::new(dataset.df())
        .numeric(dep_var)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    if cols.n() != n_obs {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "marginal effects: the model was estimated on {n_obs} rows but '{dep_var}' and \
                 the regressors have {} complete rows; pass the dataset the model was fitted on",
                cols.n()
            ),
        });
    }
    let dm = cols.matrix(x_cols, true)?;
    let k_all = dm.data.ncols();
    if discrete_result.coefficients.len() != k_all {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "marginal effects: the model has {} coefficients but the design has {k_all} \
                 columns; pass the regressors the model was fitted with",
                discrete_result.coefficients.len()
            ),
        });
    }
    // The identified columns: an aliased regressor (NaN coefficient, R's NA) has no effect,
    // as marginaleffects drops NA coefficients; its rows are NaN and a warning names it.
    let id: Vec<usize> = (0..k_all)
        .filter(|&j| discrete_result.coefficients[j].is_finite())
        .collect();
    let x_id = dm.data.select(Axis(1), &id);
    let x = &x_id;
    let k = id.len();

    let beta: Array1<f64> = id
        .iter()
        .map(|&j| discrete_result.coefficients[j])
        .collect();
    let variables = discrete_result.variables.clone();
    let has_intercept = variables
        .first()
        .map(|s| s == "(Intercept)")
        .unwrap_or(false);

    // Select PDF function based on model type
    let pdf_fn: fn(f64) -> f64 = match discrete_result.model_type {
        DiscreteModelType::Logit => logistic_pdf,
        DiscreteModelType::Probit => normal_pdf,
    };

    // Compute linear predictor for each observation: z_i = X_i' * beta
    let z: Array1<f64> = x.dot(&beta);

    // Compute PDF at each observation
    let f_z: Array1<f64> = z.mapv(pdf_fn);

    // =========================================================================
    // Average Marginal Effects (AME)
    // =========================================================================
    // AME_j = (1/n) * sum_i beta_j * f(z_i)
    //       = beta_j * (1/n) * sum_i f(z_i)
    //       = beta_j * mean(f(z))

    let mean_f_z = f_z.mean().unwrap_or(0.0);
    let ame: Array1<f64> = &beta * mean_f_z;

    // =========================================================================
    // Standard Errors via Delta Method
    // =========================================================================
    // For AME_j, we need the gradient G_j with respect to beta:
    //
    // For Logit:  lambda(z) = Lambda(z) * (1 - Lambda(z))
    //             lambda'(z) = lambda(z) * (1 - 2*Lambda(z))
    //
    // For Probit: phi'(z) = -z * phi(z)
    //
    // G_j[l] = d(AME_j)/d(beta_l)
    //        = (1/n) * sum_i [ I(j=l) * f(z_i) + beta_j * f'(z_i) * x_il ]

    let vcov = compute_discrete_vcov(discrete_result.model_type, x, &beta)?;

    // Compute gradient matrix G (k x k): G[j, l] = d(AME_j)/d(beta_l)
    let mut g = Array2::zeros((k, k));

    for j in 0..k {
        for l in 0..k {
            let mut grad_jl = 0.0;

            for i in 0..n_obs {
                let z_i = z[i];
                let f_i = f_z[i];

                // Compute f'(z) based on model type
                let f_prime = match discrete_result.model_type {
                    DiscreteModelType::Logit => {
                        // lambda'(z) = lambda(z) * (1 - 2*Lambda(z))
                        let lambda_z = logistic_cdf(z_i);
                        f_i * (1.0 - 2.0 * lambda_z)
                    }
                    DiscreteModelType::Probit => {
                        // phi'(z) = -z * phi(z)
                        -z_i * f_i
                    }
                };

                // G_j[l] = (1/n) * [ I(j=l)*f(z_i) + beta_j * f'(z_i) * x_il ]
                let indicator = if j == l { 1.0 } else { 0.0 };
                grad_jl += indicator * f_i + beta[j] * f_prime * x[[i, l]];
            }

            g[[j, l]] = grad_jl / n_obs as f64;
        }
    }

    // SE(AME_j) = sqrt(G_j' * Var(beta) * G_j)
    let mut ame_se = Array1::zeros(k);
    for j in 0..k {
        let g_j = g.row(j);
        let var_ame_j = g_j.dot(&vcov.dot(&g_j.t()));
        ame_se[j] = var_ame_j.max(0.0).sqrt();
    }

    // =========================================================================
    // Marginal Effects at the Mean (MEM)
    // =========================================================================
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let z_mean: f64 = x_mean.dot(&beta);
    let f_z_mean = pdf_fn(z_mean);

    let mem: Array1<f64> = &beta * f_z_mean;

    // MEM standard errors (simpler since evaluated at fixed point)
    // SE(MEM_j) = |f(z_bar)| * SE(beta_j) approximately
    // More precisely, use delta method with gradient at mean
    let mut mem_se = Array1::zeros(k);
    for j in 0..k {
        // Simpler approximation: SE(MEM_j) = sqrt(Var(beta_j)) * f(z_mean)
        // This ignores covariance but is a reasonable approximation
        mem_se[j] = vcov[[j, j]].max(0.0).sqrt() * f_z_mean.abs();
    }

    // =========================================================================
    // Build result structures
    // =========================================================================

    // Z critical value for 95% CI (standard normal)
    let z_crit = z_critical(0.95)?;

    let mut average_marginal = Vec::new();
    let mut at_means = Vec::new();

    for j in 0..k {
        // AME
        let z_val = if ame_se[j] > 1e-12 {
            ame[j] / ame_se[j]
        } else {
            0.0
        };
        let p_val = normal_p_value_two_sided(z_val);

        average_marginal.push(MarginalEffect {
            variable: variables[id[j]].clone(),
            estimate: ame[j],
            std_error: ame_se[j],
            z_value: z_val,
            p_value: p_val,
            ci_lower: ame[j] - z_crit * ame_se[j],
            ci_upper: ame[j] + z_crit * ame_se[j],
            dy_dx: ame[j],
            significance: SignificanceLevel::from_p_value(p_val),
        });

        // MEM
        let z_val_mem = if mem_se[j] > 1e-12 {
            mem[j] / mem_se[j]
        } else {
            0.0
        };
        let p_val_mem = normal_p_value_two_sided(z_val_mem);

        at_means.push(MarginalEffect {
            variable: variables[id[j]].clone(),
            estimate: mem[j],
            std_error: mem_se[j],
            z_value: z_val_mem,
            p_value: p_val_mem,
            ci_lower: mem[j] - z_crit * mem_se[j],
            ci_upper: mem[j] + z_crit * mem_se[j],
            dy_dx: mem[j],
            significance: SignificanceLevel::from_p_value(p_val_mem),
        });
    }

    // Store observation-level marginal effects (NaN for an aliased regressor)
    let mut effects_by_observation = Vec::new();
    for j in 0..k {
        let me_j: Array1<f64> = f_z.mapv(|f| beta[j] * f);
        effects_by_observation.push(me_j);
    }

    // Every regressor in model order, the aliased ones NaN
    let aliased: Vec<usize> = (0..k_all).filter(|j| !id.contains(j)).collect();
    let mut average_full = Vec::with_capacity(k_all);
    let mut at_means_full = Vec::with_capacity(k_all);
    let mut effects_full = Vec::with_capacity(k_all);
    let (mut ame_rows, mut mem_rows, mut eff_rows) = (
        average_marginal.into_iter(),
        at_means.into_iter(),
        effects_by_observation.into_iter(),
    );
    for j in 0..k_all {
        if aliased.contains(&j) {
            average_full.push(MarginalEffect::undefined(&variables[j]));
            at_means_full.push(MarginalEffect::undefined(&variables[j]));
            effects_full.push(Array1::from_elem(n_obs, f64::NAN));
        } else {
            average_full.extend(ame_rows.next());
            at_means_full.extend(mem_rows.next());
            effects_full.extend(eff_rows.next());
        }
    }
    let aliased_names: Vec<String> = aliased.iter().map(|&j| variables[j].clone()).collect();

    Ok(MarginalEffectsResult {
        average_marginal: average_full,
        at_means: Some(at_means_full),
        effects_by_observation: Some(effects_full),
        model_type,
        n_obs,
        variables,
        has_intercept,
        warnings: aliased_warning(&aliased_names).into_iter().collect(),
    })
}

/// The inverse information matrix of a logit/probit fit on the identified columns `x` with
/// coefficients `beta`: the variance-covariance of `beta`.
fn compute_discrete_vcov(
    model_type: DiscreteModelType,
    x: &Array2<f64>,
    beta: &Array1<f64>,
) -> EconResult<Array2<f64>> {
    let (n, k) = x.dim();

    // Compute linear predictor
    let z: Array1<f64> = x.dot(beta);

    // Compute probabilities
    let (cdf_fn, pdf_fn): (fn(f64) -> f64, fn(f64) -> f64) = match model_type {
        DiscreteModelType::Logit => (logistic_cdf, logistic_pdf),
        DiscreteModelType::Probit => (normal_cdf, normal_pdf),
    };

    let p: Array1<f64> = z.mapv(cdf_fn);
    let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

    // Compute weights for information matrix
    let weights: Array1<f64> = match model_type {
        DiscreteModelType::Logit => p_clipped.mapv(|pi| pi * (1.0 - pi)),
        DiscreteModelType::Probit => z
            .iter()
            .zip(p_clipped.iter())
            .map(|(&zi, &pi)| {
                let pdf = pdf_fn(zi);
                let denom = pi * (1.0 - pi);
                if denom > 1e-10 {
                    pdf * pdf / denom
                } else {
                    1e-10
                }
            })
            .collect(),
    };

    // Information matrix: I = X' W X
    let mut info_matrix = Array2::zeros((k, k));
    for i in 0..n {
        let wi = weights[i];
        for j in 0..k {
            for l in 0..k {
                info_matrix[[j, l]] += wi * x[[i, j]] * x[[i, l]];
            }
        }
    }

    // Invert information matrix to get variance-covariance
    let vcov = sym_inverse(
        info_matrix.view(),
        SingularPolicy::Error,
        "information matrix for marginal effects",
    )
    .map_err(|e| EconError::SingularMatrix {
        context: "Information matrix for marginal effects".to_string(),
        suggestion: format!("Model may have separation or multicollinearity: {:?}", e),
    })?
    .inverse;

    Ok(vcov)
}

// =============================================================================
// General Marginal Effects Function
// =============================================================================

/// Compute marginal effects for a specified model type.
///
/// This is a convenience function that estimates the model and computes
/// marginal effects in one step.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable
/// * `x_cols` - Names of the independent variables
/// * `model_type` - Type of model (OLS, Logit, Probit, etc.)
///
/// # Returns
/// `MarginalEffectsResult` with average marginal effects and (for nonlinear models)
/// marginal effects at the mean.
///
/// # Example
/// ```ignore
/// let me = marginal_effects(&dataset, "y", &["x1", "x2"], ModelType::Logit)?;
/// println!("{}", me);
/// ```
///
/// # References
///
/// - Arel-Bundock, V. (2023). marginaleffects R package.
///   <https://vincentarelbundock.github.io/marginaleffects/>
pub fn marginal_effects(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    model_type: ModelType,
) -> EconResult<MarginalEffectsResult> {
    match model_type {
        ModelType::Ols => {
            let ols = run_ols(dataset, y_col, x_cols, true, CovarianceType::HC1)?;
            marginal_effects_ols(&ols)
        }
        ModelType::Logit => {
            let logit = run_logit(dataset, y_col, x_cols)?;
            marginal_effects_discrete(&logit, dataset, x_cols)
        }
        ModelType::Probit => {
            let probit = run_probit(dataset, y_col, x_cols)?;
            marginal_effects_discrete(&probit, dataset, x_cols)
        }
        ModelType::Poisson | ModelType::NegBin => Err(EconError::InvalidSpecification {
            message: format!(
                "{} marginal effects not yet implemented. Use Logit, Probit, or OLS.",
                model_type
            ),
        }),
    }
}

// =============================================================================
// Contrasts
// =============================================================================

/// Compute contrasts between specified values of a variable.
///
/// Contrasts show how the predicted outcome changes when a variable changes
/// from one value to another, holding other variables constant.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `result` - Existing marginal effects result
/// * `variable` - Name of the variable to contrast
/// * `values` - Values to contrast (e.g., &[0.0, 1.0] for binary)
///
/// # Returns
/// `ContrastsResult` with pairwise contrasts between all specified values.
pub fn contrasts(
    _dataset: &Dataset,
    result: &MarginalEffectsResult,
    variable: &str,
    values: &[f64],
) -> EconResult<ContrastsResult> {
    // Find the variable in the result
    let var_idx = result
        .variables
        .iter()
        .position(|v| v == variable)
        .ok_or_else(|| EconError::ColumnNotFound {
            column: variable.to_string(),
            available: result.variables.clone(),
        })?;

    // For simple contrasts, we use the marginal effect to compute
    // the change: contrast = ME * (value2 - value1)
    let me = &result.average_marginal[var_idx];

    let mut contrasts_vec = Vec::new();

    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            let v1 = values[i];
            let v2 = values[j];
            let diff = v2 - v1;

            let estimate = me.dy_dx * diff;
            let std_error = me.std_error * diff.abs();
            let z_value = if std_error > 1e-12 {
                estimate / std_error
            } else {
                0.0
            };
            let p_value = normal_p_value_two_sided(z_value);

            contrasts_vec.push(ContrastEffect {
                value1: v1,
                value2: v2,
                estimate,
                std_error,
                z_value,
                p_value,
            });
        }
    }

    Ok(ContrastsResult {
        variable: variable.to_string(),
        contrast_type: "mean".to_string(),
        contrasts: contrasts_vec,
        n_obs: result.n_obs,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::Tol;
    use polars::prelude::*;

    #[test]
    fn test_validate_marginal_effects_logit_tail() {
        use p2a_validation::Check;
        let (case, ds) = crate::test_support::load_case("tail_marginal_logit");
        let fit = run_logit(&ds, "y", &["x1", "x2"]).unwrap();
        let me = marginal_effects_discrete(&fit, &ds, &["x1", "x2"]).unwrap();
        let names: Vec<String> = me
            .average_marginal
            .iter()
            .map(|m| m.variable.clone())
            .collect();
        let col =
            |f: fn(&MarginalEffect) -> f64| me.average_marginal.iter().map(f).collect::<Vec<_>>();
        // glm at epsilon 1e-14; p2a's GLM stops at its own tolerance: the iterative row.
        // d log p / d z = -z, so the relative p error is about z^2 times the relative z
        // error: 5e-5 covers z = 27.85.
        Check::new(&case)
            .named(
                "coef",
                &fit.variables,
                &fit.coefficients,
                Tol::iterative(1e-8),
            )
            .named("ame", &names, &col(|m| m.estimate), Tol::iterative(1e-8))
            .named(
                "ame_se",
                &names,
                &col(|m| m.std_error),
                Tol::iterative(1e-8),
            )
            .named("ame_z", &names, &col(|m| m.z_value), Tol::iterative(1e-8))
            .named("ame_p", &names, &col(|m| m.p_value), Tol::new(5e-5, 0.0))
            .finish();
        let z = crate::traits::estimator::z_critical(0.95).unwrap();
        let m = &me.average_marginal[1];
        assert!((m.ci_upper - (m.estimate + z * m.std_error)).abs() < 1e-15);
    }

    #[test]
    fn test_validate_marginal_effects_aliased_regressor_is_nan() {
        // x3 = 2 x1: glm reports NA for x3. Its effect is undefined (NaN in every column, with
        // a warning); the other effects are those of the fit without x3.
        use p2a_validation::Check;
        let (case, ds) = crate::test_support::load_case("tail_marginal_logit");
        let fit = run_logit(&ds, "y", &["x1", "x2", "x3"]).unwrap();
        let me = marginal_effects_discrete(&fit, &ds, &["x1", "x2", "x3"]).unwrap();
        let names: Vec<String> = me
            .average_marginal
            .iter()
            .map(|m| m.variable.clone())
            .collect();
        let col =
            |f: fn(&MarginalEffect) -> f64| me.average_marginal.iter().map(f).collect::<Vec<_>>();
        let identified: Vec<String> = names.iter().filter(|n| *n != "x3").cloned().collect();
        let pick = |v: Vec<f64>| -> Vec<f64> {
            names
                .iter()
                .zip(v)
                .filter(|(n, _)| *n != "x3")
                .map(|(_, x)| x)
                .collect()
        };
        Check::new(&case)
            .named(
                "coef_aliased",
                &fit.variables,
                &fit.coefficients,
                Tol::iterative(1e-8),
            )
            .named(
                "ame",
                &identified,
                &pick(col(|m| m.estimate)),
                Tol::iterative(1e-8),
            )
            .named(
                "ame_se",
                &identified,
                &pick(col(|m| m.std_error)),
                Tol::iterative(1e-8),
            )
            .finish();
        let x3 = me
            .average_marginal
            .iter()
            .find(|m| m.variable == "x3")
            .unwrap();
        assert!(x3.estimate.is_nan() && x3.std_error.is_nan() && x3.z_value.is_nan());
        assert!(x3.p_value.is_nan() && x3.ci_lower.is_nan() && x3.ci_upper.is_nan());
        assert_eq!(
            me.warnings,
            vec!["Marginal effects not defined for aliased regressors: x3".to_string()]
        );
        assert!(me.to_string().contains("aliased regressors: x3"));
    }

    /// Case data for tests that compare no R value.
    fn case_data(id: &str) -> (p2a_validation::Case, Dataset) {
        crate::test_support::load_case(id)
    }

    /// The marginal effects of a logit that dropped rows with a missing outcome are those
    /// of the same logit on the complete rows: X is built from the rows the model used.
    #[test]
    fn test_na_marginal_effects_logit_uses_estimation_rows() {
        let (_, ds) = case_data("disc_choice_na");
        let full = marginal_effects(&ds, "choice_na", &["x", "w"], ModelType::Logit).unwrap();
        let complete = Dataset::new(
            ds.df()
                .drop_nulls::<String>(Some(&["choice_na".into()]))
                .unwrap(),
        );
        let cc = marginal_effects(&complete, "choice_na", &["x", "w"], ModelType::Logit).unwrap();
        assert_eq!(full.n_obs, 298);
        let values = |m: &MarginalEffectsResult| -> Vec<f64> {
            let mut v: Vec<f64> = m.average_marginal.iter().map(|e| e.estimate).collect();
            v.extend(m.average_marginal.iter().map(|e| e.std_error));
            v.extend(m.at_means.iter().flatten().map(|e| e.estimate));
            v
        };
        let (a, b) = (values(&full), values(&cc));
        assert_eq!(a.len(), b.len());
        let tol = Tol::new(1e-12, 1e-14);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(tol.allows(*x, *y), "[{i}] {x} vs complete rows {y}");
        }
    }

    fn create_test_dataset() -> Dataset {
        // Create test data with known properties
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // not x1 - 0.5 exactly: that design aliases x2 with the intercept
        let x2 = vec![0.5, 1.7, 2.2, 3.9, 4.1, 5.8, 6.3, 7.4, 8.9, 9.2];
        // y = 1 + 2*x1 + 0.5*x2 + noise
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .enumerate()
            .map(|(i, (&a, &b))| 1.0 + 2.0 * a + 0.5 * b + (i as f64 * 0.1 - 0.5))
            .collect();

        let df = df! {
            "y" => y,
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();

        Dataset::new(df).with_name("test")
    }

    fn create_binary_dataset() -> Dataset {
        // Create binary outcome data
        let x1: Vec<f64> = (0..100)
            .map(|i| (i as f64) / 10.0 - 5.0 + 0.1 * ((i % 7) as f64))
            .collect();
        let x2: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();

        // P(y=1) = logistic(0.5 + 0.3*x1 + 0.1*x2)
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .enumerate()
            .map(|(i, (&a, &b))| {
                let z = 0.5 + 0.3 * a + 0.1 * b;
                let p = 1.0 / (1.0 + (-z).exp());
                // Deterministic assignment based on probability + noise
                if p > 0.5 + 0.1 * ((i % 5) as f64 / 5.0 - 0.1) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let df = df! {
            "y" => y,
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();

        Dataset::new(df).with_name("binary_test")
    }

    #[test]
    fn test_marginal_effects_ols() {
        let dataset = create_test_dataset();
        let ols = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard).unwrap();
        let me = marginal_effects_ols(&ols).unwrap();

        assert_eq!(me.model_type, ModelType::Ols);
        assert_eq!(me.n_obs, 10);
        assert_eq!(me.average_marginal.len(), 3); // Intercept + x1 + x2

        // For OLS, marginal effects equal coefficients
        for (i, coef) in ols.coefficients.iter().enumerate() {
            assert!(
                (me.average_marginal[i].estimate - coef.estimate).abs() < 1e-10,
                "OLS marginal effect should equal coefficient"
            );
        }
    }

    #[test]
    fn test_marginal_effects_logit() {
        let dataset = create_binary_dataset();
        let logit = run_logit(&dataset, "y", &["x1", "x2"]).unwrap();
        let me = marginal_effects_discrete(&logit, &dataset, &["x1", "x2"]).unwrap();

        assert_eq!(me.model_type, ModelType::Logit);
        assert_eq!(me.n_obs, 100);
        assert_eq!(me.average_marginal.len(), 3);

        // AME should be smaller than coefficients (scaled by PDF)
        for (i, me_i) in me.average_marginal.iter().enumerate() {
            // Skip intercept check, just verify we get reasonable values
            if i > 0 {
                assert!(
                    me_i.estimate.abs() < logit.coefficients[i].abs(),
                    "Logit AME should be smaller than coefficient in magnitude"
                );
            }
        }

        // Check that MEM exists
        assert!(me.at_means.is_some());
    }

    #[test]
    fn test_marginal_effects_probit() {
        let dataset = create_binary_dataset();
        let probit = run_probit(&dataset, "y", &["x1", "x2"]).unwrap();
        let me = marginal_effects_discrete(&probit, &dataset, &["x1", "x2"]).unwrap();

        assert_eq!(me.model_type, ModelType::Probit);
        assert_eq!(me.n_obs, 100);

        // Verify standard errors are positive
        for me_i in &me.average_marginal {
            assert!(
                me_i.std_error >= 0.0,
                "Standard errors should be non-negative"
            );
        }
    }

    #[test]
    fn test_marginal_effects_dispatcher() {
        let dataset = create_test_dataset();
        let me_ols = marginal_effects(&dataset, "y", &["x1", "x2"], ModelType::Ols).unwrap();
        assert_eq!(me_ols.model_type, ModelType::Ols);

        let binary_dataset = create_binary_dataset();
        let me_logit =
            marginal_effects(&binary_dataset, "y", &["x1", "x2"], ModelType::Logit).unwrap();
        assert_eq!(me_logit.model_type, ModelType::Logit);
    }

    #[test]
    fn test_contrasts() {
        let dataset = create_binary_dataset();
        let me = marginal_effects(&dataset, "y", &["x1", "x2"], ModelType::Logit).unwrap();

        let contrast_result = contrasts(&dataset, &me, "x1", &[0.0, 1.0, 2.0]).unwrap();

        assert_eq!(contrast_result.variable, "x1");
        assert_eq!(contrast_result.contrasts.len(), 3); // (0,1), (0,2), (1,2)

        // Check that contrast from 0 to 1 equals ME (since diff = 1)
        let c01 = &contrast_result.contrasts[0];
        assert!((c01.value1 - 0.0).abs() < 1e-10);
        assert!((c01.value2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_formatting() {
        let dataset = create_binary_dataset();
        let me = marginal_effects(&dataset, "y", &["x1", "x2"], ModelType::Logit).unwrap();

        // Check that Display works without panicking
        let display_str = format!("{}", me);
        assert!(display_str.contains("Average Marginal Effects"));
        assert!(display_str.contains("Logit"));
    }
}
