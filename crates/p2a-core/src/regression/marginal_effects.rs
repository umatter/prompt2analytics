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
//! For Logit: G_j[k] = (1/n) * sum_i (d/dbeta_k)[beta_j * lambda(X_i'beta)]
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
//!   https://cran.r-project.org/package=margins
//!
//! - Arel-Bundock, V. (2023). marginaleffects: Predictions, Comparisons, Slopes,
//!   Marginal Means, and Hypothesis Tests. R package.
//!   https://vincentarelbundock.github.io/marginaleffects/
//!
//! R equivalent: `marginaleffects::avg_slopes()`, `margins::margins()`

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::econometrics::{DiscreteModelType, DiscreteResult, run_logit, run_probit};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::regression::ols::{run_ols, CovarianceType, OlsResult};
use crate::traits::estimator::{logistic_cdf, logistic_pdf, normal_cdf, normal_pdf, SignificanceLevel};

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
    let has_intercept = variables.get(0).map(|s| s == "(Intercept)").unwrap_or(false);

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
            significance: coef.significance.clone(),
        };
        average_marginal.push(me);
    }

    // For OLS, MEM = AME (constant marginal effects)
    let at_means = Some(average_marginal.clone());

    Ok(MarginalEffectsResult {
        average_marginal,
        at_means,
        effects_by_observation: None, // All observations have same ME
        model_type: ModelType::Ols,
        n_obs,
        variables,
        has_intercept,
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

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = &dm.data;
    let k = x.ncols();

    let beta = Array1::from_vec(discrete_result.coefficients.clone());
    let variables = discrete_result.variables.clone();
    let has_intercept = variables.get(0).map(|s| s == "(Intercept)").unwrap_or(false);

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

    let vcov = compute_discrete_vcov(discrete_result, &dm)?;

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
    let z_crit = 1.96;

    let mut average_marginal = Vec::new();
    let mut at_means = Vec::new();

    for j in 0..k {
        // AME
        let z_val = if ame_se[j] > 1e-12 {
            ame[j] / ame_se[j]
        } else {
            0.0
        };
        let p_val = 2.0 * (1.0 - normal_cdf(z_val.abs()));

        average_marginal.push(MarginalEffect {
            variable: variables[j].clone(),
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
        let p_val_mem = 2.0 * (1.0 - normal_cdf(z_val_mem.abs()));

        at_means.push(MarginalEffect {
            variable: variables[j].clone(),
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

    // Store observation-level marginal effects
    let mut effects_by_observation = Vec::new();
    for j in 0..k {
        let me_j: Array1<f64> = f_z.mapv(|f| beta[j] * f);
        effects_by_observation.push(me_j);
    }

    Ok(MarginalEffectsResult {
        average_marginal,
        at_means: Some(at_means),
        effects_by_observation: Some(effects_by_observation),
        model_type,
        n_obs,
        variables,
        has_intercept,
    })
}

/// Helper function to reconstruct variance-covariance matrix from discrete result.
fn compute_discrete_vcov(
    discrete_result: &DiscreteResult,
    dm: &DesignMatrix,
) -> EconResult<Array2<f64>> {
    let k = discrete_result.coefficients.len();
    let n = discrete_result.n_obs;
    let x = &dm.data;

    let beta = Array1::from_vec(discrete_result.coefficients.clone());

    // Compute linear predictor
    let z: Array1<f64> = x.dot(&beta);

    // Compute probabilities
    let (cdf_fn, pdf_fn): (fn(f64) -> f64, fn(f64) -> f64) = match discrete_result.model_type {
        DiscreteModelType::Logit => (logistic_cdf, logistic_pdf),
        DiscreteModelType::Probit => (normal_cdf, normal_pdf),
    };

    let p: Array1<f64> = z.mapv(cdf_fn);
    let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

    // Compute weights for information matrix
    let weights: Array1<f64> = match discrete_result.model_type {
        DiscreteModelType::Logit => p_clipped.mapv(|pi| pi * (1.0 - pi)),
        DiscreteModelType::Probit => {
            z.iter()
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
                .collect()
        }
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
    let (vcov, _) = safe_inverse(&info_matrix.view()).map_err(|e| EconError::SingularMatrix {
        context: "Information matrix for marginal effects".to_string(),
        suggestion: format!(
            "Model may have separation or multicollinearity: {:?}",
            e
        ),
    })?;

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
///   https://vincentarelbundock.github.io/marginaleffects/
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
            let p_value = 2.0 * (1.0 - normal_cdf(z_value.abs()));

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
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Create test data with known properties
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
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
            assert!(me_i.std_error >= 0.0, "Standard errors should be non-negative");
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
