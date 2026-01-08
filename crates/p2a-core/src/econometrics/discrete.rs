//! Discrete choice models: Logit and Probit.
//!
//! Pure Rust implementation using Newton-Raphson MLE.
//! Uses column-based API for simplicity.

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::safe_inverse;
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{SignificanceLevel, normal_cdf, normal_pdf, logistic_cdf, logistic_pdf};

/// Discrete choice model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscreteModelType {
    Logit,
    Probit,
}

impl fmt::Display for DiscreteModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscreteModelType::Logit => write!(f, "Logit"),
            DiscreteModelType::Probit => write!(f, "Probit"),
        }
    }
}

/// Result from a discrete choice model (Logit/Probit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteResult {
    /// Model type
    pub model_type: DiscreteModelType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics
    pub z_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Null log-likelihood (model with only intercept)
    pub log_likelihood_null: f64,
    /// McFadden's Pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC: Akaike Information Criterion
    pub aic: f64,
    /// BIC: Bayesian Information Criterion
    pub bic: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Number of positive outcomes (y=1)
    pub n_positive: usize,
    /// Marginal effects at the mean
    pub marginal_effects: Vec<f64>,
}

impl fmt::Display for DiscreteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Regression Results (MLE)", self.model_type)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {} (Positive: {})", self.n_obs, self.n_positive)?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Null Log-Likelihood: {:.4}", self.log_likelihood_null)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f, "Iterations: {} (Converged: {})", self.iterations, self.converged)?;
        writeln!(f)?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.z_stats[i],
                     self.p_values[i],
                     self.significance[i].stars())?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f)?;
        writeln!(f, "Marginal Effects at Mean:")?;
        for i in 0..self.variables.len() {
            writeln!(f, "  {}: {:.4}", self.variables[i], self.marginal_effects[i])?;
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Newton-Raphson MLE settings.
#[derive(Debug, Clone)]
pub struct MleSettings {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance (for gradient norm)
    pub tolerance: f64,
    /// Step size dampening factor (0 < α ≤ 1)
    pub step_size: f64,
}

impl Default for MleSettings {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-8,
            step_size: 1.0,
        }
    }
}

/// Run Logit (logistic) regression.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable (binary 0/1)
/// * `x_cols` - Names of the independent variables
///
/// # Note
/// The dependent variable should be binary (0/1).
pub fn run_logit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Logit, MleSettings::default())
}

/// Run Probit regression.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable (binary 0/1)
/// * `x_cols` - Names of the independent variables
///
/// # Note
/// The dependent variable should be binary (0/1).
pub fn run_probit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Probit, MleSettings::default())
}

/// Run a discrete choice model (Logit or Probit) with custom settings.
pub fn run_discrete_model(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    model_type: DiscreteModelType,
    settings: MleSettings,
) -> EconResult<DiscreteResult> {
    // Extract y (binary outcome)
    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    // Validate y is binary
    let n_positive = y.iter().filter(|&&v| v >= 0.5).count();
    if n_positive == 0 || n_positive == y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dependent variable '{}' must be binary with both 0 and 1 values. Found {} ones out of {} observations.",
                y_col, n_positive, y.len()
            ),
        });
    }

    // Build design matrix with intercept
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = design.data;
    let var_names = design.column_names;
    let n = y.len();
    let k = x.ncols();

    // Link function and its derivative based on model type
    let (link_fn, link_pdf): (fn(f64) -> f64, fn(f64) -> f64) = match model_type {
        DiscreteModelType::Logit => (logistic_cdf, logistic_pdf),
        DiscreteModelType::Probit => (normal_cdf, normal_pdf),
    };

    // Initialize coefficients (zeros or simple OLS-based starting values)
    let mut beta = Array1::zeros(k);

    // Newton-Raphson iteration
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..settings.max_iter {
        iterations = iter + 1;

        // Compute linear predictor z = Xβ
        let z: Array1<f64> = x.dot(&beta);

        // Compute probabilities p = F(z)
        let p: Array1<f64> = z.mapv(link_fn);

        // Clip probabilities to avoid log(0)
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Compute gradient: g = X'(y - p)
        let residuals = &y - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < settings.tolerance {
            converged = true;
            break;
        }

        // Compute weights w = p(1-p) for Logit, or f(z)²/(F(z)(1-F(z))) for Probit
        let weights: Array1<f64> = match model_type {
            DiscreteModelType::Logit => {
                p_clipped.mapv(|pi| pi * (1.0 - pi))
            }
            DiscreteModelType::Probit => {
                z.iter().zip(p_clipped.iter())
                    .map(|(&zi, &pi)| {
                        let pdf = link_pdf(zi);
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

        // Compute Hessian: H = -X' W X
        let mut hessian: Array2<f64> = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Invert negative Hessian: (-H)^{-1}
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) = safe_inverse(&neg_hessian.view())
            .map_err(|e| EconError::SingularMatrix {
                context: format!("Hessian in {} iteration {}", model_type, iterations),
                suggestion: format!("Try different starting values or check for separation: {:?}", e),
            })?;

        // Newton-Raphson update: β ← β + α * H^{-1} g
        let delta = hess_inv.dot(&gradient);
        beta = &beta + &(&delta * settings.step_size);
    }

    // Final linear predictor and probabilities
    let z_final: Array1<f64> = x.dot(&beta);
    let p_final: Array1<f64> = z_final.mapv(link_fn);
    let p_clipped: Array1<f64> = p_final.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

    // Log-likelihood
    let log_likelihood: f64 = y.iter()
        .zip(p_clipped.iter())
        .map(|(&yi, &pi)| {
            if yi >= 0.5 {
                pi.ln()
            } else {
                (1.0 - pi).ln()
            }
        })
        .sum();

    // Null log-likelihood (intercept only)
    let p_bar = n_positive as f64 / n as f64;
    let log_likelihood_null = n_positive as f64 * p_bar.ln() + (n - n_positive) as f64 * (1.0 - p_bar).ln();

    // McFadden's Pseudo R²
    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;

    // AIC and BIC
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let bic = (k as f64) * (n as f64).ln() - 2.0 * log_likelihood;

    // Compute variance-covariance matrix (from inverse of information matrix)
    let weights: Array1<f64> = match model_type {
        DiscreteModelType::Logit => {
            p_clipped.mapv(|pi| pi * (1.0 - pi))
        }
        DiscreteModelType::Probit => {
            z_final.iter().zip(p_clipped.iter())
                .map(|(&zi, &pi)| {
                    let pdf = link_pdf(zi);
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
    let mut info_matrix: Array2<f64> = Array2::zeros((k, k));
    for i in 0..n {
        let wi = weights[i];
        for j in 0..k {
            for l in 0..k {
                info_matrix[[j, l]] += wi * x[[i, j]] * x[[i, l]];
            }
        }
    }

    let (vcov, _) = safe_inverse(&info_matrix.view())
        .map_err(|e| EconError::SingularMatrix {
            context: format!("Information matrix in {}", model_type),
            suggestion: format!("Model may have separation or multicollinearity: {:?}", e),
        })?;

    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();
    let coefficients = beta.to_vec();

    // z-statistics and p-values
    let z_stats: Vec<f64> = coefficients.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats.iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Marginal effects at the mean
    let x_mean: Array1<f64> = x.mean_axis(ndarray::Axis(0)).unwrap();
    let z_mean: f64 = x_mean.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum();
    let pdf_at_mean = link_pdf(z_mean);

    let marginal_effects: Vec<f64> = coefficients.iter()
        .map(|&b| b * pdf_at_mean)
        .collect();

    Ok(DiscreteResult {
        model_type,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients,
        std_errors,
        z_stats,
        p_values,
        significance,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        iterations,
        converged,
        n_obs: n,
        n_positive,
        marginal_effects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_binary_dataset() -> Dataset {
        // Binary outcome: y = 1 if x > threshold + noise
        // True model: P(y=1) increases with x
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_logistic_cdf() {
        assert!((logistic_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(logistic_cdf(10.0) > 0.99);
        assert!(logistic_cdf(-10.0) < 0.01);
    }

    #[test]
    fn test_logistic_pdf() {
        // Maximum at x=0
        let pdf_0 = logistic_pdf(0.0);
        let pdf_1 = logistic_pdf(1.0);
        assert!(pdf_0 > pdf_1);
    }

    #[test]
    fn test_logit_basic() {
        let dataset = create_binary_dataset();
        let result = run_logit(&dataset, "y", &["x"]).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 10);
        assert!(result.variables.len() >= 1);

        // Coefficient on x should be positive (higher x -> higher P(y=1))
        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(result.coefficients[x_idx] > 0.0,
            "Logit coefficient on x should be positive, got {}", result.coefficients[x_idx]);

        // Pseudo R-squared should be reasonable for this clear separation
        assert!(result.pseudo_r_squared > 0.3,
            "Pseudo R² should be > 0.3, got {}", result.pseudo_r_squared);
    }

    #[test]
    fn test_probit_basic() {
        let dataset = create_binary_dataset();
        let result = run_probit(&dataset, "y", &["x"]).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 10);

        // Coefficient on x should be positive
        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(result.coefficients[x_idx] > 0.0,
            "Probit coefficient on x should be positive, got {}", result.coefficients[x_idx]);

        // Pseudo R-squared should be reasonable
        assert!(result.pseudo_r_squared > 0.3,
            "Pseudo R² should be > 0.3, got {}", result.pseudo_r_squared);
    }

    #[test]
    fn test_logit_probit_sign_consistency() {
        let dataset = create_binary_dataset();
        let logit_result = run_logit(&dataset, "y", &["x"]).unwrap();
        let probit_result = run_probit(&dataset, "y", &["x"]).unwrap();

        // Both should have same sign for coefficient
        let logit_x = logit_result.coefficients.iter()
            .zip(&logit_result.variables)
            .find(|(_, v)| *v == "x")
            .map(|(c, _)| *c)
            .unwrap();
        let probit_x = probit_result.coefficients.iter()
            .zip(&probit_result.variables)
            .find(|(_, v)| *v == "x")
            .map(|(c, _)| *c)
            .unwrap();

        assert!(logit_x.signum() == probit_x.signum(),
            "Logit and Probit should have same sign: {} vs {}", logit_x, probit_x);

        // Logit coefficient is typically larger than probit coefficient
        // The ratio varies based on the data but logit should be larger
        assert!(logit_x.abs() > probit_x.abs(),
            "Logit coefficient should be larger in absolute value: {} vs {}", logit_x, probit_x);
    }

    #[test]
    fn test_logit_missing_column() {
        let dataset = create_binary_dataset();
        let result = run_logit(&dataset, "y", &["nonexistent"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_probit_missing_column() {
        let dataset = create_binary_dataset();
        let result = run_probit(&dataset, "nonexistent", &["x"]);
        assert!(result.is_err());
    }
}
