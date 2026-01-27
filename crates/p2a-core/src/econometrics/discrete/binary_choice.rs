//! Binary choice models: Logit and Probit.
//!
//! Pure Rust implementation using Newton-Raphson MLE.
//!
//! # Mathematical Background
//!
//! For binary outcomes y in {0, 1}, the latent variable model is:
//!
//! y*_i = X_i'beta + epsilon_i,  y_i = 1[y*_i > 0]
//!
//! ## Logit Model
//!
//! Assumes epsilon follows a logistic distribution:
//!
//! P(y_i = 1 | X_i) = Lambda(X_i'beta) = exp(X_i'beta) / (1 + exp(X_i'beta))
//!
//! ## Probit Model
//!
//! Assumes epsilon follows a standard normal distribution:
//!
//! P(y_i = 1 | X_i) = Phi(X_i'beta)
//!
//! where Phi is the standard normal CDF.
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).

use ndarray::{Array1, Array2};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::{logistic_cdf, normal_cdf, normal_pdf, SignificanceLevel};

use super::types::{DiscreteModelType, DiscreteResult, MleSettings};

/// Run logit regression.
///
/// R equivalent: `stats::glm(y ~ x1 + x2, family = binomial(link = "logit"), data = df)`
pub fn run_logit(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Logit, None)
}

/// Run probit regression.
///
/// R equivalent: `stats::glm(y ~ x1 + x2, family = binomial(link = "probit"), data = df)`
pub fn run_probit(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Probit, None)
}

/// Detect perfect or quasi-complete separation in binary response data.
///
/// Perfect separation occurs when a linear combination of predictors perfectly
/// predicts the binary outcome. Quasi-complete separation occurs when the
/// predictions are nearly perfect.
///
/// Returns (has_perfect_separation, has_quasi_separation, problematic_variables)
fn detect_separation(y: &[f64], x: &Array2<f64>, var_names: &[String]) -> (bool, bool, Vec<String>) {
    let n = y.len();
    let k = x.ncols();
    let mut problematic_vars = Vec::new();

    // Check each variable individually for separation
    for j in 0..k {
        // Get min and max of x[j] for y=0 and y=1 cases
        let mut x_when_y0: Vec<f64> = Vec::new();
        let mut x_when_y1: Vec<f64> = Vec::new();

        for i in 0..n {
            if y[i] < 0.5 {
                x_when_y0.push(x[[i, j]]);
            } else {
                x_when_y1.push(x[[i, j]]);
            }
        }

        if x_when_y0.is_empty() || x_when_y1.is_empty() {
            // All outcomes are same value - not really separation but problematic
            continue;
        }

        let min_y0 = x_when_y0.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y0 = x_when_y0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y1 = x_when_y1.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y1 = x_when_y1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Perfect separation: ranges don't overlap
        if max_y0 < min_y1 || max_y1 < min_y0 {
            if j < var_names.len() {
                problematic_vars.push(var_names[j].clone());
            }
        }
    }

    let has_perfect = !problematic_vars.is_empty();

    // Check for quasi-complete separation (ranges barely touch)
    let mut quasi_vars = Vec::new();
    for j in 0..k {
        if problematic_vars.iter().any(|v| var_names.get(j) == Some(v)) {
            continue;
        }

        let mut x_when_y0: Vec<f64> = Vec::new();
        let mut x_when_y1: Vec<f64> = Vec::new();

        for i in 0..n {
            if y[i] < 0.5 {
                x_when_y0.push(x[[i, j]]);
            } else {
                x_when_y1.push(x[[i, j]]);
            }
        }

        if x_when_y0.is_empty() || x_when_y1.is_empty() {
            continue;
        }

        let min_y0 = x_when_y0.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y0 = x_when_y0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y1 = x_when_y1.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y1 = x_when_y1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Count overlap points
        let overlap_count = x_when_y0
            .iter()
            .filter(|&&x| x >= min_y1 && x <= max_y1)
            .count()
            + x_when_y1
                .iter()
                .filter(|&&x| x >= min_y0 && x <= max_y0)
                .count();

        // Quasi-separation if overlap is very small (< 5% of data)
        if overlap_count > 0 && overlap_count < n / 20 {
            if j < var_names.len() {
                quasi_vars.push(var_names[j].clone());
            }
        }
    }

    let has_quasi = !quasi_vars.is_empty();
    problematic_vars.extend(quasi_vars);

    (has_perfect, has_quasi, problematic_vars)
}

/// Run a discrete choice model (Logit or Probit).
pub fn run_discrete_model(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    model_type: DiscreteModelType,
    settings: Option<MleSettings>,
) -> EconResult<DiscreteResult> {
    let settings = settings.unwrap_or_default();
    let df = dataset.df();
    let n = df.height();

    // Extract y
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Build design matrix with intercept
    let dm = DesignMatrix::from_dataframe(df, x_cols, true)?;
    let x = dm.view().to_owned();
    let k = x.ncols();

    // Build variable names
    let mut var_names = vec!["(Intercept)".to_string()];
    var_names.extend(x_cols.iter().map(|s| s.to_string()));

    // Check for separation
    let (has_perfect, has_quasi, problem_vars) = detect_separation(&y, &x, &var_names);

    if has_perfect {
        return Err(EconError::PerfectSeparation {
            variables: problem_vars,
        });
    }

    let mut warnings = Vec::new();
    if has_quasi {
        warnings.push(format!(
            "Quasi-complete separation detected for: {}. Coefficients may be unstable.",
            problem_vars.join(", ")
        ));
    }

    // Initialize coefficients
    let mut beta: Array1<f64> = Array1::zeros(k);

    // Null log-likelihood (intercept only)
    let p_bar = y.iter().sum::<f64>() / n as f64;
    let log_likelihood_null = n as f64 * (p_bar * p_bar.ln() + (1.0 - p_bar) * (1.0 - p_bar).ln());

    let mut converged = false;
    let mut iterations = 0;
    let mut prev_ll = f64::NEG_INFINITY;

    // Track coefficient explosion for detecting multivariate separation
    let mut max_coef_magnitude = 0.0f64;
    let explosion_threshold = 50.0;

    for iter in 0..settings.max_iter {
        iterations = iter + 1;

        // Compute linear predictor: eta = X * beta
        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..k {
                    sum += x[[i, j]] * beta[j];
                }
                sum
            })
            .collect();

        // Compute probabilities and log-likelihood
        let (probs, ll): (Vec<f64>, f64) = match model_type {
            DiscreteModelType::Logit => {
                let p: Vec<f64> = eta.iter().map(|&e| logistic_cdf(e)).collect();
                let ll: f64 = y
                    .iter()
                    .zip(p.iter())
                    .map(|(&yi, &pi)| {
                        yi * pi.max(1e-15).ln() + (1.0 - yi) * (1.0 - pi).max(1e-15).ln()
                    })
                    .sum();
                (p, ll)
            }
            DiscreteModelType::Probit => {
                let p: Vec<f64> = eta.iter().map(|&e| normal_cdf(e)).collect();
                let ll: f64 = y
                    .iter()
                    .zip(p.iter())
                    .map(|(&yi, &pi)| {
                        yi * pi.max(1e-15).ln() + (1.0 - yi) * (1.0 - pi).max(1e-15).ln()
                    })
                    .sum();
                (p, ll)
            }
        };

        // Check coefficient explosion (sign of multivariate separation)
        let current_max = beta.iter().map(|b| b.abs()).fold(0.0f64, f64::max);
        if current_max > explosion_threshold && current_max > max_coef_magnitude * 2.0 {
            return Err(EconError::PerfectSeparation {
                variables: var_names
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| beta[*i].abs() > explosion_threshold)
                    .map(|(_, v)| v.clone())
                    .collect(),
            });
        }
        max_coef_magnitude = max_coef_magnitude.max(current_max);

        // Check convergence on log-likelihood change
        if (ll - prev_ll).abs() < settings.tolerance && iter > 0 {
            converged = true;
            break;
        }
        prev_ll = ll;

        // Compute gradient and Hessian
        let (gradient, hessian) = match model_type {
            DiscreteModelType::Logit => {
                // Gradient: X'(y - p)
                let mut grad = Array1::zeros(k);
                for i in 0..n {
                    let residual = y[i] - probs[i];
                    for j in 0..k {
                        grad[j] += x[[i, j]] * residual;
                    }
                }

                // Hessian: -X' diag(p(1-p)) X
                let mut hess = Array2::zeros((k, k));
                for i in 0..n {
                    let w = probs[i] * (1.0 - probs[i]);
                    for j in 0..k {
                        for l in 0..k {
                            hess[[j, l]] -= x[[i, j]] * w * x[[i, l]];
                        }
                    }
                }

                (grad, hess)
            }
            DiscreteModelType::Probit => {
                // For probit, gradient involves inverse Mills ratio
                let mut grad = Array1::zeros(k);
                for i in 0..n {
                    let phi = normal_pdf(eta[i]);
                    let big_phi = probs[i];
                    let lambda = if y[i] > 0.5 {
                        phi / big_phi.max(1e-15)
                    } else {
                        -phi / (1.0 - big_phi).max(1e-15)
                    };
                    for j in 0..k {
                        grad[j] += x[[i, j]] * lambda;
                    }
                }

                // Hessian approximation using outer product of gradient
                let mut hess = Array2::zeros((k, k));
                for i in 0..n {
                    let phi = normal_pdf(eta[i]);
                    let big_phi = probs[i].clamp(1e-15, 1.0 - 1e-15);
                    let w = phi * phi / (big_phi * (1.0 - big_phi));
                    for j in 0..k {
                        for l in 0..k {
                            hess[[j, l]] -= x[[i, j]] * w * x[[i, l]];
                        }
                    }
                }

                (grad, hess)
            }
        };

        // Newton-Raphson update with line search
        let (h_inv, _) = match safe_inverse(&hessian.view()) {
            Ok(inv) => inv,
            Err(_) => {
                // If Hessian is singular, try gradient descent step
                let step = 0.01;
                for j in 0..k {
                    beta[j] += step * gradient[j];
                }
                continue;
            }
        };

        let delta = h_inv.dot(&gradient);

        if settings.use_line_search {
            // Backtracking line search (Armijo condition)
            let mut step = settings.step_size;
            let grad_dot_delta: f64 = gradient.iter().zip(delta.iter()).map(|(g, d)| g * d).sum();

            for _ in 0..settings.max_line_search {
                let beta_new: Array1<f64> =
                    Array1::from_iter(beta.iter().zip(delta.iter()).map(|(b, d)| b - step * d));

                let new_ll = compute_log_likelihood(&y, &x, &beta_new, model_type);

                // Armijo condition: sufficient decrease
                if new_ll >= ll + settings.armijo_c * step * grad_dot_delta {
                    beta = beta_new;
                    break;
                }
                step *= settings.step_reduction;
            }
        } else {
            // Simple Newton step
            beta = &beta - &(settings.step_size * &delta);
        }
    }

    // Final evaluation
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..k {
                sum += x[[i, j]] * beta[j];
            }
            sum
        })
        .collect();

    let probs: Vec<f64> = match model_type {
        DiscreteModelType::Logit => eta.iter().map(|&e| logistic_cdf(e)).collect(),
        DiscreteModelType::Probit => eta.iter().map(|&e| normal_cdf(e)).collect(),
    };

    let log_likelihood: f64 = y
        .iter()
        .zip(probs.iter())
        .map(|(&yi, &pi)| yi * pi.max(1e-15).ln() + (1.0 - yi) * (1.0 - pi).max(1e-15).ln())
        .sum();

    // Compute variance-covariance matrix from Hessian
    let hessian = match model_type {
        DiscreteModelType::Logit => {
            let mut hess = Array2::zeros((k, k));
            for i in 0..n {
                let w = probs[i] * (1.0 - probs[i]);
                for j in 0..k {
                    for l in 0..k {
                        hess[[j, l]] -= x[[i, j]] * w * x[[i, l]];
                    }
                }
            }
            hess
        }
        DiscreteModelType::Probit => {
            let mut hess = Array2::zeros((k, k));
            for i in 0..n {
                let phi = normal_pdf(eta[i]);
                let big_phi = probs[i].clamp(1e-15, 1.0 - 1e-15);
                let w = phi * phi / (big_phi * (1.0 - big_phi));
                for j in 0..k {
                    for l in 0..k {
                        hess[[j, l]] -= x[[i, j]] * w * x[[i, l]];
                    }
                }
            }
            hess
        }
    };

    let neg_hessian = -hessian;
    let vcov = match safe_inverse(&neg_hessian.view()) {
        Ok((inv, _)) => inv,
        Err(_) => Array2::eye(k) * 1e-6,
    };

    let std_errors: Vec<f64> = (0..k).map(|i| vcov[[i, i]].max(0.0).sqrt()).collect();

    let z_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    // Marginal effects at the mean
    let x_mean: Vec<f64> = (0..k).map(|j| (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64).collect();

    let eta_mean: f64 = x_mean.iter().zip(beta.iter()).map(|(x, b)| x * b).sum();

    let marginal_effects: Vec<f64> = match model_type {
        DiscreteModelType::Logit => {
            let p = logistic_cdf(eta_mean);
            beta.iter().map(|&b| b * p * (1.0 - p)).collect()
        }
        DiscreteModelType::Probit => {
            let phi = normal_pdf(eta_mean);
            beta.iter().map(|&b| b * phi).collect()
        }
    };

    // Model fit statistics
    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
    let aic = -2.0 * log_likelihood + 2.0 * k as f64;
    let bic = -2.0 * log_likelihood + (k as f64) * (n as f64).ln();

    // Calculate significance levels
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Count positive outcomes
    let n_positive = y.iter().filter(|&&yi| yi > 0.5).count();

    Ok(DiscreteResult {
        model_type,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta.to_vec(),
        std_errors,
        z_stats,
        p_values,
        significance,
        marginal_effects,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        n_obs: n,
        n_positive,
        iterations,
        converged,
        warnings,
    })
}

/// Compute log-likelihood for a given beta.
fn compute_log_likelihood(
    y: &[f64],
    x: &Array2<f64>,
    beta: &Array1<f64>,
    model_type: DiscreteModelType,
) -> f64 {
    let n = y.len();
    let k = beta.len();

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..k {
                sum += x[[i, j]] * beta[j];
            }
            sum
        })
        .collect();

    let probs: Vec<f64> = match model_type {
        DiscreteModelType::Logit => eta.iter().map(|&e| logistic_cdf(e)).collect(),
        DiscreteModelType::Probit => eta.iter().map(|&e| normal_cdf(e)).collect(),
    };

    y.iter()
        .zip(probs.iter())
        .map(|(&yi, &pi)| yi * pi.max(1e-15).ln() + (1.0 - yi) * (1.0 - pi).max(1e-15).ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_binary_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_logit_basic() {
        let dataset = create_binary_dataset();
        let result = run_logit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert!(result.variables.len() >= 1);

        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(
            result.coefficients[x_idx] > 0.0,
            "Logit coefficient on x should be positive"
        );
    }

    #[test]
    fn test_probit_basic() {
        let dataset = create_binary_dataset();
        let result = run_probit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);

        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(
            result.coefficients[x_idx] > 0.0,
            "Probit coefficient on x should be positive"
        );
    }

    #[test]
    fn test_perfect_separation_detection() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_logit(&dataset, "y", &["x"]);
        assert!(result.is_err(), "Should detect perfect separation");
    }
}
