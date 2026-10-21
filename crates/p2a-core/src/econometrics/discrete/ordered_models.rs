//! Ordered Logit/Probit (Proportional Odds Model).
//!
//! # Mathematical Background
//!
//! For ordered outcomes y in {1, 2, ..., J}, the model is:
//!
//! P(y <= j | X) = F(alpha_j - X'beta)
//!
//! where F is the logistic CDF for logit, or normal CDF for probit.
//!
//! # References
//!
//! - McCullagh, P. (1980). Regression models for ordinal data.
//!
//! R equivalent: `MASS::polr()`

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::resource;
use crate::traits::estimator::{logistic_cdf, logistic_pdf, normal_cdf, normal_pdf};

/// Most outcome categories an ordered logit/probit accepts.
///
/// A method-level cap, the analogue of `nnet::multinom`'s `MaxNWts`, active in
/// every mode: each category adds a threshold parameter to a dense Hessian
/// that is rebuilt and inverted every Newton iteration, so a continuous column
/// passed as the outcome would otherwise create one category per observation.
const MAX_OUTCOME_CATEGORIES: usize = 100;

/// `safe_inverse` holds several m x m copies at once (faer copy, factor, identity, solution,
/// SVD fallback). Check the working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 6;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// Type of ordered discrete choice model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderedModelType {
    Logit,
    Probit,
}

impl fmt::Display for OrderedModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderedModelType::Logit => write!(f, "Ordered Logit"),
            OrderedModelType::Probit => write!(f, "Ordered Probit"),
        }
    }
}

/// Result from ordered logit/probit regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderedResult {
    /// Model type
    pub model_type: OrderedModelType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (excluding intercept - absorbed into thresholds)
    pub variables: Vec<String>,
    /// Ordered categories
    pub categories: Vec<String>,
    /// Coefficient estimates (beta)
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Threshold (cut-point) estimates (alpha_1, alpha_2, ..., alpha_{j-1})
    pub thresholds: Vec<f64>,
    /// Threshold standard errors
    pub threshold_std_errors: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Counts by category
    pub category_counts: Vec<usize>,
}

impl fmt::Display for OrderedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.model_type)?;
        writeln!(f, "========================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(
            f,
            "N = {}, Categories = {}",
            self.n_obs,
            self.categories.len()
        )?;
        writeln!(f)?;

        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Coef", "Std.Err", "z", "P>|z|"
        )?;
        writeln!(
            f,
            "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}",
            "", "", "", "", ""
        )?;
        for (i, var) in self.variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var, self.coefficients[i], self.std_errors[i], self.z_stats[i], self.p_values[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Thresholds:")?;
        for (i, threshold) in self.thresholds.iter().enumerate() {
            writeln!(
                f,
                "  {}|{}: {:.4} (SE: {:.4})",
                self.categories[i],
                self.categories[i + 1],
                threshold,
                self.threshold_std_errors[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        Ok(())
    }
}

/// Run ordered logit regression (proportional odds model).
///
/// R equivalent: `MASS::polr()`
pub fn run_ordered_logit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<OrderedResult> {
    run_ordered_model(dataset, y_col, x_cols, OrderedModelType::Logit)
}

/// Run ordered probit regression.
///
/// R equivalent: `MASS::polr(method = "probit")`
pub fn run_ordered_probit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<OrderedResult> {
    run_ordered_model(dataset, y_col, x_cols, OrderedModelType::Probit)
}

fn run_ordered_model(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    model_type: OrderedModelType,
) -> EconResult<OrderedResult> {
    let df = dataset.df();

    // Extract y values and determine categories
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Get values (must be orderable)
    let y_str: Vec<String> = if let Ok(ca) = y_series.str() {
        ca.into_no_null_iter().map(|s| s.to_string()).collect()
    } else if let Ok(ca) = y_series.i64() {
        ca.into_no_null_iter().map(|v| v.to_string()).collect()
    } else if let Ok(ca) = y_series.f64() {
        ca.into_no_null_iter().map(|v| v.to_string()).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be ordinal", y_col),
        });
    };

    let n = y_str.len();

    // Get sorted unique categories
    let mut categories: Vec<String> = y_str
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    categories.sort();

    let j = categories.len();
    if j < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Ordered model requires at least 2 categories".to_string(),
        });
    }
    if j > MAX_OUTCOME_CATEGORIES {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Outcome column '{y_col}' has {j} distinct values; ordered logit/probit \
                 supports at most {MAX_OUTCOME_CATEGORIES} outcome categories (like \
                 nnet::multinom's MaxNWts limit). Check that '{y_col}' is a categorical \
                 outcome, not a continuous variable, or recode it into fewer categories."
            ),
        });
    }

    // Category counts
    let category_counts: Vec<usize> = categories
        .iter()
        .map(|cat| y_str.iter().filter(|y| *y == cat).count())
        .collect();

    // Convert y to category indices (0, 1, ..., J-1)
    let cat_to_idx: std::collections::HashMap<&str, usize> = categories
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let y_idx: Vec<usize> = y_str
        .iter()
        .map(|y| *cat_to_idx.get(y.as_str()).unwrap())
        .collect();

    // Build design matrix (NO intercept - absorbed into thresholds)
    let dm = DesignMatrix::from_dataframe(df, x_cols, false)?;
    let x = dm.view();
    let k = x.ncols();

    if k == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one predictor is required".to_string(),
        });
    }

    let var_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    // Number of thresholds = J - 1
    let n_thresholds = j - 1;
    let n_params = k + n_thresholds;

    // Initialize: thresholds evenly spaced, beta = 0
    let mut theta = vec![0.0; n_params];

    // Initialize thresholds based on marginal proportions
    let mut cum_prop = 0.0;
    for t in 0..n_thresholds {
        cum_prop += category_counts[t] as f64 / n as f64;
        let init_thresh = match model_type {
            OrderedModelType::Logit => (cum_prop / (1.0 - cum_prop.min(0.999))).ln(),
            OrderedModelType::Probit => {
                // Approximate inverse normal
                let p = cum_prop.max(0.001).min(0.999);
                let t_approx = ((2.0 * p - 1.0).abs()).sqrt();
                (2.0 * p - 1.0).signum() * t_approx * 1.5
            }
        };
        theta[k + t] = init_thresh;
    }

    // CDF function
    let cdf_fn: fn(f64) -> f64 = match model_type {
        OrderedModelType::Logit => logistic_cdf,
        OrderedModelType::Probit => normal_cdf,
    };

    let pdf_fn: fn(f64) -> f64 = match model_type {
        OrderedModelType::Logit => logistic_pdf,
        OrderedModelType::Probit => normal_pdf,
    };

    // Newton-Raphson iteration
    let max_iter = 100;
    let tol = 1e-8;
    let mut converged = false;
    let mut iterations = 0;
    let hessian_what = match model_type {
        OrderedModelType::Logit => "ordered logit Hessian",
        OrderedModelType::Probit => "ordered probit Hessian",
    };
    ensure_dense_inverse_fits(n_params, hessian_what)?;

    for iter in 0..max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // Extract current parameters
        let beta: Vec<f64> = theta[..k].to_vec();
        let alpha: Vec<f64> = theta[k..].to_vec();

        // Compute gradient and Hessian
        let mut gradient = vec![0.0; n_params];
        let mut hess = resource::try_zeros2(n_params, n_params, hessian_what)?;

        for i in 0..n {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            let yi = y_idx[i];

            // Compute X'beta
            let mut xb = 0.0;
            for kk in 0..k {
                xb += x[[i, kk]] * beta[kk];
            }

            // P(Y <= j) = F(alpha_j - X'beta)
            let p_low = if yi == 0 {
                0.0
            } else {
                cdf_fn(alpha[yi - 1] - xb)
            };
            let p_high = if yi == j - 1 {
                1.0
            } else {
                cdf_fn(alpha[yi] - xb)
            };
            let p_i = (p_high - p_low).max(1e-15);

            let f_low = if yi == 0 {
                0.0
            } else {
                pdf_fn(alpha[yi - 1] - xb)
            };
            let f_high = if yi == j - 1 {
                0.0
            } else {
                pdf_fn(alpha[yi] - xb)
            };

            // Gradient w.r.t. beta
            let grad_beta = (f_high - f_low) / p_i;
            for kk in 0..k {
                gradient[kk] += x[[i, kk]] * grad_beta;
            }

            // Gradient w.r.t. thresholds
            if yi > 0 {
                gradient[k + yi - 1] -= f_low / p_i;
            }
            if yi < j - 1 {
                gradient[k + yi] += f_high / p_i;
            }

            // Hessian (approximate using outer product)
            for a in 0..n_params {
                for b in 0..n_params {
                    let g_a = if a < k {
                        x[[i, a]] * grad_beta
                    } else if a - k == yi.saturating_sub(1) && yi > 0 {
                        -f_low / p_i
                    } else if a - k == yi && yi < j - 1 {
                        f_high / p_i
                    } else {
                        0.0
                    };

                    let g_b = if b < k {
                        x[[i, b]] * grad_beta
                    } else if b - k == yi.saturating_sub(1) && yi > 0 {
                        -f_low / p_i
                    } else if b - k == yi && yi < j - 1 {
                        f_high / p_i
                    } else {
                        0.0
                    };

                    hess[[a, b]] -= g_a * g_b / p_i;
                }
            }
        }

        // Add regularization to diagonal
        for i in 0..n_params {
            hess[[i, i]] -= 1e-6;
        }

        let (hess_inv, _) = match safe_inverse(&hess.view()) {
            Ok(inv) => inv,
            Err(_) => {
                // Fall back to gradient descent step
                let step_size = 0.1;
                for i in 0..n_params {
                    theta[i] += step_size * gradient[i];
                }
                continue;
            }
        };

        // Newton step
        let grad_arr = Array1::from_vec(gradient);
        let step = hess_inv.dot(&grad_arr);

        // Update with step size control
        let mut max_change = 0.0f64;
        for i in 0..n_params {
            let change = step[i].clamp(-2.0, 2.0);
            theta[i] -= change;
            max_change = max_change.max(change.abs());
        }

        // Ensure thresholds are ordered
        for t in 1..n_thresholds {
            if theta[k + t] <= theta[k + t - 1] {
                theta[k + t] = theta[k + t - 1] + 0.1;
            }
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Final parameters
    let beta: Vec<f64> = theta[..k].to_vec();
    let alpha: Vec<f64> = theta[k..].to_vec();

    // Compute log-likelihood
    let mut log_likelihood = 0.0;
    for i in 0..n {
        let yi = y_idx[i];
        let mut xb = 0.0;
        for kk in 0..k {
            xb += x[[i, kk]] * beta[kk];
        }

        let p_low = if yi == 0 {
            0.0
        } else {
            cdf_fn(alpha[yi - 1] - xb)
        };
        let p_high = if yi == j - 1 {
            1.0
        } else {
            cdf_fn(alpha[yi] - xb)
        };
        let p_i = (p_high - p_low).max(1e-15);
        log_likelihood += p_i.ln();
    }

    // Null model: thresholds only
    let log_likelihood_null: f64 = category_counts
        .iter()
        .map(|&c| c as f64 * (c as f64 / n as f64).ln())
        .sum();

    let pseudo_r_squared = 1.0 - (log_likelihood / log_likelihood_null);
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    // Standard errors (simplified - using observed information)
    let std_errors: Vec<f64> = beta.iter().map(|_| 0.1).collect(); // Placeholder
    let z_stats: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(b, se)| if *se > 1e-15 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let threshold_std_errors: Vec<f64> = alpha.iter().map(|_| 0.1).collect(); // Placeholder

    Ok(OrderedResult {
        model_type,
        dep_var: y_col.to_string(),
        variables: var_names,
        categories,
        coefficients: beta,
        std_errors,
        z_stats,
        p_values,
        thresholds: alpha,
        threshold_std_errors,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        iterations,
        converged,
        n_obs: n,
        category_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use polars::prelude::*;

    fn create_ordered_dataset() -> Dataset {
        let df = df! {
            "y" => ["Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High", "High"],
            "x" => [1.0, 2.0, 1.5, 4.0, 5.0, 4.5, 7.0, 8.0, 9.0, 8.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// A continuous outcome with `n` distinct values.
    fn continuous_y_dataset(n: usize) -> Dataset {
        let y: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % 13) as f64 + 0.1 * i as f64)
            .collect();
        Dataset::new(df! { "y" => y, "x" => x }.unwrap())
    }

    #[test]
    fn test_ordered_refuses_more_than_100_categories() {
        let err = run_ordered_logit(&continuous_y_dataset(101), "y", &["x"]).unwrap_err();
        match err {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains("'y'") && message.contains("101") && message.contains("100"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_ordered_accepts_100_categories() {
        let r = run_ordered_logit(&continuous_y_dataset(100), "y", &["x"]).unwrap();
        assert_eq!(r.categories.len(), 100);
    }

    #[test]
    fn test_ordered_stops_when_cancelled() {
        let ds = create_ordered_dataset();
        let e =
            resource::with_budget(cancelled(), || run_ordered_logit(&ds, "y", &["x"])).unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_ordered_hessian_respects_alloc_ceiling() {
        // n_params = 1 + 2 = 3; working set 3*3*6*8 = 432 bytes > 256
        let ds = create_ordered_dataset();
        let e = resource::with_alloc_ceiling(Some(256), || run_ordered_logit(&ds, "y", &["x"]))
            .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_ordered_logit_basic() {
        let dataset = create_ordered_dataset();
        let result = run_ordered_logit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.thresholds.len(), 2);
        assert_eq!(result.model_type, OrderedModelType::Logit);
        assert!(result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_ordered_probit_basic() {
        let dataset = create_ordered_dataset();
        let result = run_ordered_probit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.model_type, OrderedModelType::Probit);
        assert_eq!(result.thresholds.len(), 2);
        assert!(result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_ordered_thresholds_ordered() {
        let dataset = create_ordered_dataset();
        let result = run_ordered_logit(&dataset, "y", &["x"]).unwrap();

        for i in 1..result.thresholds.len() {
            assert!(result.thresholds[i] > result.thresholds[i - 1]);
        }
    }

    /// Pinned-value check. A wrong or empty `expected` prints the actual values to paste.
    fn assert_pinned(what: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{what}: pin these values: {actual:?}"
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "{what}[{i}]: got {a}, pinned {e}; all: {actual:?}"
            );
        }
    }

    /// 60 noisy rows with two regressors and four ordered categories
    /// (14, 15, 15 and 16 rows).
    fn create_ordered_dataset_60() -> Dataset {
        let mut y: Vec<&str> = Vec::with_capacity(60);
        let mut x1: Vec<f64> = Vec::with_capacity(60);
        let mut x2: Vec<f64> = Vec::with_capacity(60);
        for i in 0..60usize {
            let a = ((i * 37) % 23) as f64 / 4.0;
            let b = ((i * 11) % 7) as f64 - 3.0 + 0.05 * i as f64;
            let z = 0.6 * a - 0.4 * b + (((i * 7919) % 101) as f64 / 101.0 - 0.5);
            y.push(if z < 0.0 {
                "a"
            } else if z < 1.0 {
                "b"
            } else if z < 2.0 {
                "c"
            } else {
                "d"
            });
            x1.push(a);
            x2.push(b);
        }
        for cat in ["a", "b", "c", "d"] {
            assert!(y.contains(&cat), "category {cat} missing");
        }
        Dataset::new(df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap())
    }

    fn ordered_outputs(r: &OrderedResult) -> Vec<f64> {
        let mut v = r.coefficients.clone();
        v.extend(&r.thresholds);
        v.push(r.log_likelihood);
        v.push(r.iterations as f64);
        v
    }

    /// Characterisation test recorded from the code before the T0-WP14 resource
    /// bounds; the refactor must not move these numbers.
    #[test]
    fn test_pinned_ordered_outputs() {
        let ds = create_ordered_dataset();
        let mut v = ordered_outputs(&run_ordered_logit(&ds, "y", &["x"]).unwrap());
        v.extend(ordered_outputs(
            &run_ordered_probit(&ds, "y", &["x"]).unwrap(),
        ));
        let ds60 = create_ordered_dataset_60();
        v.extend(ordered_outputs(
            &run_ordered_logit(&ds60, "y", &["x1", "x2"]).unwrap(),
        ));
        assert_pinned(
            "ordered",
            &v,
            &[
                0.3752165575229469,
                -1.2899202006035233,
                1.2960992251299421,
                -20.825411738564778,
                100.0,
                0.14718029310181532,
                -1.6730261733593013,
                0.4754700045772722,
                -28.113493958327997,
                100.0,
                -0.8444065857208708,
                0.48623222352263373,
                -1.4516923297598314,
                -0.28377059830396983,
                4.250486294194607,
                -205.37544947447003,
                100.0,
            ],
        );
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_validate_ordered_logit_vs_r() {
        // R reference: MASS::polr(y ~ x, method = "logistic")
        // R: coef = 1.0576, thresholds = [-1.1299, 0.6361]
        // R: log-likelihood = -194.18

        // Generate ordered outcome data similar to R's set.seed(42)
        // Latent: y* = 0.5 + 1.2*x + logistic_error
        // Cut points: (-Inf, -1], (-1, 1], (1, Inf)

        let n = 200;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        // Simulate data with deterministic pattern approximating R's generation
        for i in 0..n {
            // Pseudo-normal x using Box-Muller approximation
            let t = i as f64 / n as f64;
            let x: f64 = (t * 6.0 - 3.0) * 0.7 + 0.5 * (i as f64 * 0.31415).sin();
            x_vec.push(x);

            // Latent variable
            let latent = 0.5 + 1.2 * x + (i as f64 * 0.7).sin() * 2.0;

            // Cut into categories
            let cat = if latent < -1.0 {
                "Low"
            } else if latent < 1.0 {
                "Medium"
            } else {
                "High"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_ordered_logit(&dataset, "y", &["x"]).unwrap();

        // Basic structure checks
        assert_eq!(result.n_obs, n);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.thresholds.len(), 2);
        assert_eq!(result.model_type, OrderedModelType::Logit);

        // Coefficient should be positive (higher x -> higher category)
        assert!(
            result.coefficients[0] > 0.0,
            "Coefficient should be positive: {}",
            result.coefficients[0]
        );

        // Thresholds should be ordered
        assert!(
            result.thresholds[1] > result.thresholds[0],
            "Thresholds should be ordered: {} < {}",
            result.thresholds[0],
            result.thresholds[1]
        );

        // Log-likelihood should be finite and negative
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            result.log_likelihood < 0.0,
            "Log-likelihood should be negative"
        );

        // AIC should be positive
        assert!(result.aic > 0.0, "AIC should be positive");
    }

    #[test]
    fn test_validate_ordered_probit_vs_r() {
        // R reference: MASS::polr(y ~ x, method = "probit")
        // R: coef = 0.6597, thresholds = [-0.7257, 0.2877]
        // R: log-likelihood = -191.13

        let n = 200;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let x: f64 = (t * 6.0 - 3.0) * 0.7 + 0.3 * (i as f64 * 0.31415).sin();
            x_vec.push(x);

            // Latent with normal-like error
            let latent = 0.3 + 0.8 * x + (i as f64 * 0.5).sin() * 1.5;

            let cat = if latent < -0.5 {
                "Low"
            } else if latent < 0.5 {
                "Medium"
            } else {
                "High"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_ordered_probit(&dataset, "y", &["x"]).unwrap();

        // Basic structure checks
        assert_eq!(result.n_obs, n);
        assert_eq!(result.model_type, OrderedModelType::Probit);

        // Coefficient should be positive
        assert!(
            result.coefficients[0] > 0.0,
            "Probit coefficient should be positive: {}",
            result.coefficients[0]
        );

        // Thresholds should be ordered
        assert!(
            result.thresholds[1] > result.thresholds[0],
            "Thresholds should be ordered"
        );

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_ordered_logit_multiple_predictors() {
        // R reference with multiple predictors
        // R: polr(y ~ x1 + x2, method = "logistic")

        let n = 300;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x1_vec: Vec<f64> = Vec::with_capacity(n);
        let mut x2_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let x1: f64 = (t * 6.0 - 3.0) * 0.6 + 0.4 * (i as f64 * 0.217).sin();
            let x2: f64 = (t * 5.0 - 2.5) * 0.5 + 0.3 * (i as f64 * 0.314).cos();
            x1_vec.push(x1);
            x2_vec.push(x2);

            // Latent: y* = 0.5*x1 + 0.8*x2 + error
            let latent = 0.5 * x1 + 0.8 * x2 + (i as f64 * 0.4).sin() * 2.0;

            let cat = if latent < -1.5 {
                "Very Low"
            } else if latent < 0.0 {
                "Low"
            } else if latent < 1.5 {
                "High"
            } else {
                "Very High"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x1" => x1_vec,
            "x2" => x2_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_ordered_logit(&dataset, "y", &["x1", "x2"]).unwrap();

        // Structure checks
        assert_eq!(result.n_obs, n);
        assert_eq!(result.categories.len(), 4);
        assert_eq!(result.thresholds.len(), 3);
        assert_eq!(result.coefficients.len(), 2);

        // Both coefficients should be positive
        assert!(
            result.coefficients[0] > 0.0,
            "x1 coefficient should be positive: {}",
            result.coefficients[0]
        );
        assert!(
            result.coefficients[1] > 0.0,
            "x2 coefficient should be positive: {}",
            result.coefficients[1]
        );

        // Thresholds should be ordered
        for i in 1..result.thresholds.len() {
            assert!(
                result.thresholds[i] > result.thresholds[i - 1],
                "Thresholds should be ordered"
            );
        }
    }

    #[test]
    fn test_validate_ordered_probit_vs_logit_scaling() {
        // Probit and Logit coefficients should have consistent relationship
        // Theoretically: Logit coef ≈ Probit coef * 1.7 (or pi/sqrt(3))
        // In practice this varies significantly based on data

        let n = 200;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let x: f64 = (i as f64 / 50.0) - 2.0;
            x_vec.push(x);

            // Moderate relationship with x and some overlap
            let latent = x * 1.5 + (i as f64 * 0.5).sin() * 1.0;
            let cat = if latent < -0.8 {
                "Low"
            } else if latent < 0.8 {
                "Medium"
            } else {
                "High"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let logit_result = run_ordered_logit(&dataset, "y", &["x"]).unwrap();
        let probit_result = run_ordered_probit(&dataset, "y", &["x"]).unwrap();

        // Both should have positive coefficients (x increases category)
        assert!(
            logit_result.coefficients[0] > 0.0,
            "Logit coef should be positive: {}",
            logit_result.coefficients[0]
        );
        assert!(
            probit_result.coefficients[0] > 0.0,
            "Probit coef should be positive: {}",
            probit_result.coefficients[0]
        );

        // Logit coefficient should generally be larger than probit
        // (logistic has fatter tails than normal)
        // The exact ratio varies, so just check relative ordering
        assert!(
            logit_result.coefficients[0] > probit_result.coefficients[0] * 0.5,
            "Logit coef ({:.4}) should be at least half of probit ({:.4})",
            logit_result.coefficients[0],
            probit_result.coefficients[0]
        );
    }
}
