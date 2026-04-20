//! Regularized linear regression with elastic net penalty (glmnet).
//!
//! Implements the elastic net estimator combining L1 (lasso) and L2 (ridge) penalties:
//!
//! min_β (1/2n) ||y - Xβ||² + λ[(1-α)||β||₂²/2 + α||β||₁]
//!
//! where:
//! - α = 1 gives the lasso
//! - α = 0 gives ridge regression
//! - 0 < α < 1 gives elastic net
//!
//! # Algorithm
//!
//! Uses cyclical coordinate descent with warm starts and active set convergence.
//! The coordinate update for coefficient j is:
//!
//! β̃_j = S(∂L/∂β_j, λα) / (1 + λ(1-α))
//!
//! where S is the soft-thresholding operator: S(z, γ) = sign(z)(|z| - γ)₊
//!
//! # Features
//!
//! - Pathwise solution along sequence of λ values
//! - Cross-validation for λ selection
//! - Supports Gaussian and binomial families
//! - Standardization of predictors (optional)
//!
//! # References
//!
//! - Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
//!   generalized linear models via coordinate descent. *Journal of Statistical Software*,
//!   33(1), 1-22. https://doi.org/10.18637/jss.v033.i01
//!
//! - Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic
//!   net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
//!   https://doi.org/10.1111/j.1467-9868.2005.00503.x
//!
//! R equivalent: `glmnet::glmnet()`, `glmnet::cv.glmnet()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::DesignMatrix;
use polars::prelude::*;

/// Family type for GLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GlmnetFamily {
    /// Gaussian (linear regression)
    #[default]
    Gaussian,
    /// Binomial (logistic regression)
    Binomial,
}

/// Configuration for glmnet fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetConfig {
    /// Elastic net mixing parameter (0 = ridge, 1 = lasso)
    pub alpha: f64,
    /// Sequence of lambda values (if None, automatically generated)
    pub lambda: Option<Vec<f64>>,
    /// Number of lambda values in path (default: 100)
    pub nlambda: usize,
    /// Minimum lambda ratio (lambda_min / lambda_max)
    pub lambda_min_ratio: f64,
    /// Standardize predictors before fitting (default: true)
    pub standardize: bool,
    /// Include intercept (default: true)
    pub intercept: bool,
    /// Maximum iterations for coordinate descent
    pub max_iter: usize,
    /// Convergence threshold
    pub thresh: f64,
    /// Model family
    pub family: GlmnetFamily,
    /// Maximum number of non-zero coefficients
    pub dfmax: Option<usize>,
    /// Minimum deviance ratio for early stopping
    pub deviance_thresh: f64,
}

impl Default for GlmnetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0, // lasso by default
            lambda: None,
            nlambda: 100,
            lambda_min_ratio: 0.0001,
            standardize: true,
            intercept: true,
            max_iter: 100_000,
            thresh: 1e-7,
            family: GlmnetFamily::Gaussian,
            dfmax: None,
            deviance_thresh: 0.999,
        }
    }
}

/// Result for a single lambda value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetCoefficients {
    /// Lambda value
    pub lambda: f64,
    /// Intercept term
    pub intercept: f64,
    /// Coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Number of non-zero coefficients
    pub df: usize,
    /// Percent deviance explained
    pub dev_ratio: f64,
}

/// Result of glmnet fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetResult {
    /// Variable names (excluding intercept)
    pub variable_names: Vec<String>,
    /// Dependent variable name
    pub dependent_var: String,
    /// Alpha parameter used
    pub alpha: f64,
    /// Lambda sequence
    pub lambda: Vec<f64>,
    /// Results for each lambda
    pub fits: Vec<GlmnetCoefficients>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Null deviance
    pub null_deviance: f64,
    /// Family used
    pub family: GlmnetFamily,
    /// Standardization means (if standardized)
    pub x_mean: Option<Vec<f64>>,
    /// Standardization scales (if standardized)
    pub x_scale: Option<Vec<f64>>,
    /// Response mean (for standardization)
    pub y_mean: f64,
}

/// Cross-validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetCvResult {
    /// Full glmnet result
    pub fit: GlmnetResult,
    /// Lambda values tested
    pub lambda: Vec<f64>,
    /// Mean cross-validated error for each lambda
    pub cvm: Vec<f64>,
    /// Standard error of cross-validated error
    pub cvsd: Vec<f64>,
    /// Upper confidence bound (cvm + cvsd)
    pub cvup: Vec<f64>,
    /// Lower confidence bound (cvm - cvsd)
    pub cvlo: Vec<f64>,
    /// Lambda that gives minimum CV error
    pub lambda_min: f64,
    /// Index of lambda_min in lambda sequence
    pub lambda_min_index: usize,
    /// Largest lambda within 1 SE of minimum
    pub lambda_1se: f64,
    /// Index of lambda_1se
    pub lambda_1se_index: usize,
    /// Number of folds used
    pub nfolds: usize,
}

/// Soft-thresholding operator: S(z, γ) = sign(z)(|z| - γ)₊
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Compute the maximum lambda (where all coefficients are zero)
fn compute_lambda_max(x: &ArrayView2<f64>, y: &ArrayView1<f64>, alpha: f64) -> f64 {
    let n = x.nrows() as f64;
    let mut max_grad = 0.0f64;

    for j in 0..x.ncols() {
        let grad = x.column(j).dot(y).abs() / n;
        max_grad = max_grad.max(grad);
    }

    // For elastic net, lambda_max is scaled by alpha
    if alpha > 0.0 {
        max_grad / alpha.max(1e-3)
    } else {
        max_grad * 100.0 // Large value for ridge
    }
}

/// Coordinate descent for Gaussian family
/// OPTIMIZED: Uses incremental residual updates instead of recomputing X*beta each iteration
fn coordinate_descent_gaussian(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    lambda: f64,
    alpha: f64,
    beta: &mut Array1<f64>,
    max_iter: usize,
    thresh: f64,
) -> usize {
    let n = x.nrows() as f64;
    let p = x.ncols();

    // Precompute X'X diagonal (column norms squared) and X'y
    let col_norms_sq: Vec<f64> = (0..p).map(|j| x.column(j).dot(&x.column(j))).collect();
    let xty: Vec<f64> = (0..p).map(|j| x.column(j).dot(y)).collect();

    // Precompute X'X for active set (we'll compute these lazily)
    // For now, maintain residual = y - X*beta
    let mut residual: Array1<f64> = y - &x.dot(beta);

    // Active set - start with all variables
    let mut active: Vec<usize> = (0..p).collect();
    let mut ever_active = vec![false; p];

    for iter in 0..max_iter {
        let mut max_change = 0.0f64;

        // First pass: cycle through active set
        for &j in &active {
            let old_beta = beta[j];

            // Gradient using residual: (1/n) * X_j' * (residual + X_j * beta_j)
            // = (1/n) * (X_j' * residual + col_norm_sq[j] * beta_j)
            let xj_residual = x.column(j).dot(&residual);
            let rho = (xj_residual + col_norms_sq[j] * old_beta) / n;

            // Coordinate update with elastic net penalty
            let z = soft_threshold(rho, lambda * alpha);
            let new_beta = z / (col_norms_sq[j] / n + lambda * (1.0 - alpha));

            // Update residual incrementally: r += X_j * (old_beta - new_beta)
            if (new_beta - old_beta).abs() > 1e-15 {
                let delta = old_beta - new_beta;
                for i in 0..x.nrows() {
                    residual[i] += x[[i, j]] * delta;
                }
                beta[j] = new_beta;
                max_change = max_change.max((new_beta - old_beta).abs());

                if new_beta.abs() > 1e-10 {
                    ever_active[j] = true;
                }
            }
        }

        // Convergence check
        if max_change < thresh {
            return iter + 1;
        }

        // Update active set every 10 iterations or when converging
        if iter % 10 == 9 || max_change < thresh * 10.0 {
            // Include variables that are currently non-zero or have been active
            active = (0..p)
                .filter(|&j| beta[j].abs() > 1e-10 || ever_active[j])
                .collect();

            // If active set is empty or very small, scan all variables
            if active.len() < 3 {
                active = (0..p).collect();
            }
        }
    }

    max_iter
}

/// Coordinate descent for Binomial family (logistic regression)
fn coordinate_descent_binomial(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    lambda: f64,
    alpha: f64,
    beta: &mut Array1<f64>,
    intercept: &mut f64,
    max_iter: usize,
    thresh: f64,
) -> usize {
    let n = x.nrows() as f64;
    let p = x.ncols();

    for iter in 0..max_iter {
        let mut max_change = 0.0f64;

        // Compute probabilities: p = 1 / (1 + exp(-eta))
        let eta: Array1<f64> = x.dot(beta) + *intercept;
        let prob: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));

        // Working weights: w = p * (1 - p)
        let w: Array1<f64> = prob.mapv(|p| (p * (1.0 - p)).max(1e-5));

        // Working response: z = eta + (y - p) / w
        let z: Array1<f64> = &eta + &((&y.to_owned() - &prob) / &w);

        // Update intercept (unpenalized)
        let old_intercept = *intercept;
        let w_sum: f64 = w.sum();
        let w_z_sum: f64 = (&w * &z).sum();
        *intercept = w_z_sum / w_sum;
        max_change = max_change.max((*intercept - old_intercept).abs());

        // Update each coefficient
        for j in 0..p {
            let old_beta = beta[j];

            // Weighted partial residual
            let r: Array1<f64> = &z - &x.dot(beta) - *intercept + &(&x.column(j) * old_beta);

            // Weighted gradient
            let xj = x.column(j);
            let w_xj_r: f64 = (&w * &xj.to_owned() * &r).sum();
            let w_xj_sq: f64 = (&w * &xj.to_owned() * &xj.to_owned()).sum();

            // Coordinate update
            let z_val = soft_threshold(w_xj_r / n, lambda * alpha);
            let new_beta = z_val / (w_xj_sq / n + lambda * (1.0 - alpha));

            beta[j] = new_beta;
            max_change = max_change.max((new_beta - old_beta).abs());
        }

        if max_change < thresh {
            return iter + 1;
        }
    }

    max_iter
}

/// Compute deviance for Gaussian family
fn deviance_gaussian(y: &ArrayView1<f64>, fitted: &Array1<f64>) -> f64 {
    let residuals: Array1<f64> = y - fitted;
    residuals.mapv(|r| r * r).sum()
}

/// Compute deviance for Binomial family
fn deviance_binomial(y: &ArrayView1<f64>, prob: &Array1<f64>) -> f64 {
    let mut dev = 0.0;
    for i in 0..y.len() {
        let p = prob[i].max(1e-10).min(1.0 - 1e-10);
        dev -= 2.0 * (y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln());
    }
    dev
}

/// Fit glmnet model
pub fn glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
) -> EconResult<GlmnetResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::Computation(format!(
            "X has {} rows but y has {} elements",
            n,
            y.len()
        )));
    }

    if config.alpha < 0.0 || config.alpha > 1.0 {
        return Err(EconError::Computation(
            "alpha must be between 0 and 1".to_string(),
        ));
    }

    // Standardize X and center y
    let y_mean = y.mean().unwrap_or(0.0);
    let y_centered: Array1<f64> = if config.intercept {
        y.to_owned() - y_mean
    } else {
        y.to_owned()
    };

    let (x_scaled, x_mean, x_scale) = if config.standardize {
        let x_mean: Vec<f64> = (0..p).map(|j| x.column(j).mean().unwrap_or(0.0)).collect();
        let x_scale: Vec<f64> = (0..p)
            .map(|j| {
                let col = x.column(j);
                let mean = x_mean[j];
                let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64);
                var.sqrt().max(1e-10)
            })
            .collect();

        let mut x_scaled = Array2::zeros((n, p));
        for j in 0..p {
            for i in 0..n {
                x_scaled[[i, j]] = (x[[i, j]] - x_mean[j]) / x_scale[j];
            }
        }
        (x_scaled, Some(x_mean), Some(x_scale))
    } else {
        (x.to_owned(), None, None)
    };

    // Compute null deviance
    let null_deviance = match config.family {
        GlmnetFamily::Gaussian => y_centered.mapv(|v| v * v).sum(),
        GlmnetFamily::Binomial => {
            let p_null = y.mean().unwrap_or(0.5);
            let prob_null = Array1::from_elem(n, p_null);
            deviance_binomial(&y.view(), &prob_null)
        }
    };

    // Generate lambda sequence
    let lambda_max = compute_lambda_max(&x_scaled.view(), &y_centered.view(), config.alpha);
    let lambda_min = lambda_max * config.lambda_min_ratio;

    let lambda_seq: Vec<f64> = if let Some(ref user_lambda) = config.lambda {
        user_lambda.clone()
    } else {
        let log_max = lambda_max.ln();
        let log_min = lambda_min.ln();
        (0..config.nlambda)
            .map(|i| {
                let t = i as f64 / (config.nlambda - 1) as f64;
                (log_max + t * (log_min - log_max)).exp()
            })
            .collect()
    };

    // Fit along the path with warm starts
    let mut fits = Vec::with_capacity(lambda_seq.len());
    let mut beta = Array1::zeros(p);
    let mut intercept_val = 0.0;

    for &lambda in &lambda_seq {
        // Coordinate descent
        match config.family {
            GlmnetFamily::Gaussian => {
                coordinate_descent_gaussian(
                    &x_scaled.view(),
                    &y_centered.view(),
                    lambda,
                    config.alpha,
                    &mut beta,
                    config.max_iter,
                    config.thresh,
                );
            }
            GlmnetFamily::Binomial => {
                coordinate_descent_binomial(
                    &x_scaled.view(),
                    &y.view(),
                    lambda,
                    config.alpha,
                    &mut beta,
                    &mut intercept_val,
                    config.max_iter,
                    config.thresh,
                );
            }
        }

        // Compute deviance
        let fitted = match config.family {
            GlmnetFamily::Gaussian => x_scaled.dot(&beta),
            GlmnetFamily::Binomial => {
                let eta = x_scaled.dot(&beta) + intercept_val;
                eta.mapv(|e| 1.0 / (1.0 + (-e).exp()))
            }
        };

        let deviance = match config.family {
            GlmnetFamily::Gaussian => deviance_gaussian(&y_centered.view(), &fitted),
            GlmnetFamily::Binomial => deviance_binomial(&y.view(), &fitted),
        };

        let dev_ratio = 1.0 - deviance / null_deviance;

        // Transform coefficients back to original scale
        let (final_coefs, final_intercept) = if config.standardize {
            let x_scale = x_scale.as_ref().unwrap();
            let x_mean = x_mean.as_ref().unwrap();

            let coefs: Vec<f64> = (0..p).map(|j| beta[j] / x_scale[j]).collect();
            let intercept = if config.intercept {
                match config.family {
                    GlmnetFamily::Gaussian => {
                        y_mean - (0..p).map(|j| coefs[j] * x_mean[j]).sum::<f64>()
                    }
                    GlmnetFamily::Binomial => {
                        intercept_val
                            - (0..p)
                                .map(|j| beta[j] * x_mean[j] / x_scale[j])
                                .sum::<f64>()
                    }
                }
            } else {
                0.0
            };
            (coefs, intercept)
        } else {
            let coefs: Vec<f64> = beta.iter().copied().collect();
            let intercept = if config.intercept {
                match config.family {
                    GlmnetFamily::Gaussian => y_mean,
                    GlmnetFamily::Binomial => intercept_val,
                }
            } else {
                0.0
            };
            (coefs, intercept)
        };

        let df = final_coefs.iter().filter(|&&c| c.abs() > 1e-10).count();

        fits.push(GlmnetCoefficients {
            lambda,
            intercept: final_intercept,
            coefficients: final_coefs,
            df,
            dev_ratio,
        });

        // Early stopping
        if dev_ratio > config.deviance_thresh {
            break;
        }
        if let Some(dfmax) = config.dfmax {
            if df >= dfmax {
                break;
            }
        }
    }

    Ok(GlmnetResult {
        variable_names: (0..p).map(|i| format!("V{}", i + 1)).collect(),
        dependent_var: "y".to_string(),
        alpha: config.alpha,
        lambda: fits.iter().map(|f| f.lambda).collect(),
        fits,
        n_obs: n,
        n_features: p,
        null_deviance,
        family: config.family,
        x_mean,
        x_scale,
        y_mean,
    })
}

/// Extract a numeric column from a DataFrame as Array1<f64>
fn extract_column(df: &DataFrame, col_name: &str) -> EconResult<Array1<f64>> {
    let series = df
        .column(col_name)
        .map_err(|_| EconError::Computation(format!("Column '{}' not found", col_name)))?;

    let values: Vec<f64> = series
        .cast(&DataType::Float64)
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", col_name)))?
        .f64()
        .map_err(|_| EconError::Computation(format!("Could not convert '{}' to f64", col_name)))?
        .into_no_null_iter()
        .collect();

    Ok(Array1::from_vec(values))
}

/// Fit glmnet model from Dataset
pub fn run_glmnet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &GlmnetConfig,
) -> EconResult<GlmnetResult> {
    // Build design matrix (without intercept - handled internally)
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = dm.data.view();
    let y = extract_column(dataset.df(), y_col)?;

    let mut result = glmnet(x, y.view(), config)?;

    // Update variable names
    result.variable_names = x_cols.iter().map(|&s| s.to_string()).collect();
    result.dependent_var = y_col.to_string();

    Ok(result)
}

/// Cross-validation for glmnet
pub fn cv_glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
    nfolds: usize,
    seed: Option<u64>,
) -> EconResult<GlmnetCvResult> {
    let n = x.nrows();

    if nfolds < 2 {
        return Err(EconError::Computation(
            "nfolds must be at least 2".to_string(),
        ));
    }
    if nfolds > n {
        return Err(EconError::Computation(format!(
            "nfolds ({}) cannot exceed number of observations ({})",
            nfolds, n
        )));
    }

    // First fit on full data to get lambda sequence
    let full_fit = glmnet(x, y, config)?;
    let lambda_seq = full_fit.lambda.clone();
    let nlambda = lambda_seq.len();

    // Create fold assignments
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut fold_ids: Vec<usize> = (0..n).map(|i| i % nfolds).collect();
    fold_ids.shuffle(&mut rng);

    // Cross-validation errors
    let mut cv_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(nlambda); nfolds];

    for fold in 0..nfolds {
        // Create train/test split
        let train_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] == fold).collect();

        let n_train = train_idx.len();
        let n_test = test_idx.len();

        let mut x_train = Array2::zeros((n_train, x.ncols()));
        let mut y_train = Array1::zeros(n_train);
        let mut x_test = Array2::zeros((n_test, x.ncols()));
        let mut y_test = Array1::zeros(n_test);

        for (new_i, &orig_i) in train_idx.iter().enumerate() {
            x_train.row_mut(new_i).assign(&x.row(orig_i));
            y_train[new_i] = y[orig_i];
        }
        for (new_i, &orig_i) in test_idx.iter().enumerate() {
            x_test.row_mut(new_i).assign(&x.row(orig_i));
            y_test[new_i] = y[orig_i];
        }

        // Fit on training data with fixed lambda sequence
        let mut fold_config = config.clone();
        fold_config.lambda = Some(lambda_seq.clone());
        let fold_fit = glmnet(x_train.view(), y_train.view(), &fold_config)?;

        // Compute test error for each lambda
        for fit in &fold_fit.fits {
            let coefs = Array1::from_vec(fit.coefficients.clone());
            let fitted = x_test.dot(&coefs) + fit.intercept;

            let error = match config.family {
                GlmnetFamily::Gaussian => {
                    let mse: f64 = (&y_test - &fitted).mapv(|r| r * r).sum() / n_test as f64;
                    mse
                }
                GlmnetFamily::Binomial => {
                    let prob = fitted.mapv(|e| 1.0 / (1.0 + (-e).exp()));
                    deviance_binomial(&y_test.view(), &prob) / n_test as f64
                }
            };
            cv_errors[fold].push(error);
        }
    }

    // Compute mean and standard error across folds
    let cvm: Vec<f64> = (0..nlambda)
        .map(|l| {
            let errors: Vec<f64> = cv_errors.iter().filter_map(|f| f.get(l).copied()).collect();
            errors.iter().sum::<f64>() / errors.len() as f64
        })
        .collect();

    let cvsd: Vec<f64> = (0..nlambda)
        .map(|l| {
            let errors: Vec<f64> = cv_errors.iter().filter_map(|f| f.get(l).copied()).collect();
            let mean = cvm[l];
            let var: f64 =
                errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (errors.len() - 1) as f64;
            (var / errors.len() as f64).sqrt()
        })
        .collect();

    let cvup: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m + s).collect();
    let cvlo: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m - s).collect();

    // Find lambda.min and lambda.1se
    let (lambda_min_index, &lambda_min_cvm) = cvm
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    let lambda_min = lambda_seq[lambda_min_index];

    // lambda.1se: largest lambda within 1 SE of minimum
    let threshold = lambda_min_cvm + cvsd[lambda_min_index];
    let lambda_1se_index = (0..=lambda_min_index)
        .find(|&i| cvm[i] <= threshold)
        .unwrap_or(lambda_min_index);
    let lambda_1se = lambda_seq[lambda_1se_index];

    Ok(GlmnetCvResult {
        fit: full_fit,
        lambda: lambda_seq,
        cvm,
        cvsd,
        cvup,
        cvlo,
        lambda_min,
        lambda_min_index,
        lambda_1se,
        lambda_1se_index,
        nfolds,
    })
}

/// Run cross-validated glmnet from Dataset
pub fn run_cv_glmnet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &GlmnetConfig,
    nfolds: usize,
    seed: Option<u64>,
) -> EconResult<GlmnetCvResult> {
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = dm.data.view();
    let y = extract_column(dataset.df(), y_col)?;

    let mut result = cv_glmnet(x, y.view(), config, nfolds, seed)?;

    // Update variable names
    result.fit.variable_names = x_cols.iter().map(|&s| s.to_string()).collect();
    result.fit.dependent_var = y_col.to_string();

    Ok(result)
}

/// Convenience function for ridge regression (alpha = 0)
pub fn ridge(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 0.0,
        lambda,
        ..Default::default()
    };
    glmnet(x, y, &config)
}

/// Convenience function for lasso regression (alpha = 1)
pub fn lasso(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 1.0,
        lambda,
        ..Default::default()
    };
    glmnet(x, y, &config)
}

/// Run ridge regression from Dataset
pub fn run_ridge(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 0.0,
        lambda,
        ..Default::default()
    };
    run_glmnet(dataset, y_col, x_cols, &config)
}

/// Run lasso regression from Dataset
pub fn run_lasso(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 1.0,
        lambda,
        ..Default::default()
    };
    run_glmnet(dataset, y_col, x_cols, &config)
}

/// Predict from glmnet result at a specific lambda
pub fn glmnet_predict(
    result: &GlmnetResult,
    x: ArrayView2<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    // Find the closest lambda
    let idx = result
        .lambda
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - lambda).abs())
                .partial_cmp(&((*b - lambda).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .ok_or_else(|| EconError::Computation("Empty lambda sequence".to_string()))?;

    let fit = &result.fits[idx];
    let coefs = Array1::from_vec(fit.coefficients.clone());

    let eta = x.dot(&coefs) + fit.intercept;

    match result.family {
        GlmnetFamily::Gaussian => Ok(eta),
        GlmnetFamily::Binomial => Ok(eta.mapv(|e| 1.0 / (1.0 + (-e).exp()))),
    }
}

/// Get coefficients at a specific lambda
pub fn glmnet_coef(result: &GlmnetResult, lambda: f64) -> EconResult<(f64, Vec<(String, f64)>)> {
    let idx = result
        .lambda
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - lambda).abs())
                .partial_cmp(&((*b - lambda).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .ok_or_else(|| EconError::Computation("Empty lambda sequence".to_string()))?;

    let fit = &result.fits[idx];
    let coefs: Vec<(String, f64)> = result
        .variable_names
        .iter()
        .zip(&fit.coefficients)
        .filter(|(_, c)| c.abs() > 1e-10)
        .map(|(name, &c)| (name.clone(), c))
        .collect();

    Ok((fit.intercept, coefs))
}

impl std::fmt::Display for GlmnetResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Glmnet Result: {} ~ {} variables",
            self.dependent_var, self.n_features
        )?;
        writeln!(f, "  Family:     {:?}", self.family)?;
        writeln!(
            f,
            "  Alpha:      {:.2} ({})",
            self.alpha,
            if self.alpha == 1.0 {
                "lasso"
            } else if self.alpha == 0.0 {
                "ridge"
            } else {
                "elastic net"
            }
        )?;
        writeln!(f, "  N:          {}", self.n_obs)?;
        writeln!(f, "  Lambda path: {} values", self.lambda.len())?;
        writeln!(f)?;
        writeln!(f, "  Lambda      Df    %Dev")?;
        for fit in self.fits.iter().take(10) {
            writeln!(
                f,
                "  {:.4e}    {:3}   {:.3}",
                fit.lambda, fit.df, fit.dev_ratio
            )?;
        }
        if self.fits.len() > 10 {
            writeln!(f, "  ... ({} more)", self.fits.len() - 10)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for GlmnetCvResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cross-Validated Glmnet ({}-fold)", self.nfolds)?;
        writeln!(f)?;
        writeln!(
            f,
            "  lambda.min:  {:.4e} (index {})",
            self.lambda_min, self.lambda_min_index
        )?;
        writeln!(
            f,
            "  lambda.1se:  {:.4e} (index {})",
            self.lambda_1se, self.lambda_1se_index
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  At lambda.min: {} non-zero coefs, {:.1}% dev explained",
            self.fit.fits[self.lambda_min_index].df,
            self.fit.fits[self.lambda_min_index].dev_ratio * 100.0
        )?;
        writeln!(
            f,
            "  At lambda.1se: {} non-zero coefs, {:.1}% dev explained",
            self.fit.fits[self.lambda_1se_index].df,
            self.fit.fits[self.lambda_1se_index].dev_ratio * 100.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_relative_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_relative_eq!(soft_threshold(1.0, 2.0), 0.0);
        assert_relative_eq!(soft_threshold(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_ridge_closed_form() {
        // Ridge has closed form: β = (X'X + λI)⁻¹ X'y
        // For simple case, verify it produces non-zero coefficients for all variables
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = GlmnetConfig {
            alpha: 0.0,
            lambda: Some(vec![0.1]),
            nlambda: 1,
            standardize: false,
            intercept: false,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // Both coefficients should be non-zero (ridge doesn't zero out coefficients)
        assert_eq!(result.fits.len(), 1);
        assert!(result.fits[0].coefficients[0].abs() > 0.01);
        assert!(result.fits[0].coefficients[1].abs() > 0.01);
    }

    #[test]
    fn test_lasso_sparsity() {
        // Lasso should produce sparse solutions with high lambda
        let x = array![
            [1.0, 0.1, 0.2],
            [2.0, 0.2, 0.1],
            [3.0, 0.1, 0.3],
            [4.0, 0.3, 0.1],
            [5.0, 0.2, 0.2]
        ];
        // y is strongly correlated with first column, weakly with others
        let y = array![1.1, 2.0, 3.1, 4.0, 5.1];

        let config = GlmnetConfig {
            alpha: 1.0, // lasso
            nlambda: 50,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // At high lambda, should have fewer non-zero coefficients
        let high_lambda_fit = &result.fits[0];
        let low_lambda_fit = result.fits.last().unwrap();

        assert!(high_lambda_fit.df <= low_lambda_fit.df);
    }

    #[test]
    fn test_lambda_path() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![2.0, 4.0, 5.0, 7.0, 9.0];

        let config = GlmnetConfig {
            alpha: 0.5,
            nlambda: 20,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // Lambda should be decreasing
        for i in 1..result.lambda.len() {
            assert!(result.lambda[i] < result.lambda[i - 1]);
        }

        // Deviance ratio should generally increase
        assert!(result.fits.last().unwrap().dev_ratio >= result.fits[0].dev_ratio);
    }

    #[test]
    fn test_cv_glmnet() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0]
        ];
        let y = array![2.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        let config = GlmnetConfig {
            alpha: 1.0,
            nlambda: 20,
            ..Default::default()
        };

        let cv_result = cv_glmnet(x.view(), y.view(), &config, 5, Some(42)).unwrap();

        // lambda.1se should be >= lambda.min
        assert!(cv_result.lambda_1se >= cv_result.lambda_min);

        // CV errors should be computed
        assert_eq!(cv_result.cvm.len(), cv_result.lambda.len());
        assert_eq!(cv_result.cvsd.len(), cv_result.lambda.len());
    }
}
