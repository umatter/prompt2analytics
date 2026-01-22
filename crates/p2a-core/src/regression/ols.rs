//! Ordinary Least Squares (OLS) regression with robust standard errors.
//!
//! Provides pure Rust implementation of OLS regression with:
//! - Standard OLS estimation: β = (X'X)^{-1} X'y
//! - Heteroskedasticity-robust standard errors (HC0, HC1, HC2, HC3)
//! - Clustered standard errors (one-way and two-way)
//! - Full diagnostics and fit statistics

use ndarray::{Array1, Array2};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult, EstimationWarning};
use crate::linalg::{
    DesignMatrix, DesignError,
    xtx, xty, safe_inverse, matmul, CONDITION_THRESHOLD,
};
use crate::traits::{LinearEstimator, SignificanceLevel, t_test_p_value, f_test_p_value};

/// Type of covariance/standard error estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CovarianceType {
    /// Standard OLS (homoskedastic) standard errors
    #[default]
    Standard,
    /// White's heteroskedasticity-consistent (HC0): no small sample correction
    HC0,
    /// HC1: multiply by n/(n-k) for small sample correction (default robust)
    HC1,
    /// HC2: divide by (1 - h_ii) where h_ii is leverage
    HC2,
    /// HC3: divide by (1 - h_ii)^2 - most conservative
    HC3,
}

impl std::fmt::Display for CovarianceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard (homoskedastic)"),
            Self::HC0 => write!(f, "HC0 (White)"),
            Self::HC1 => write!(f, "HC1 (White with small sample correction)"),
            Self::HC2 => write!(f, "HC2 (leverage-adjusted)"),
            Self::HC3 => write!(f, "HC3 (conservative leverage-adjusted)"),
        }
    }
}

/// A single coefficient with its statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsCoefficient {
    pub name: String,
    pub estimate: f64,
    pub std_error: f64,
    pub t_value: f64,
    pub p_value: f64,
    pub significance: SignificanceLevel,
    pub ci_lower_95: f64,
    pub ci_upper_95: f64,
}

/// Result of an OLS regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Dependent variable name
    pub dependent_var: String,
    /// Independent variable names (including intercept if present)
    pub variable_names: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════
    // Core results
    // ═══════════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters (including intercept)
    pub n_params: usize,
    /// Degrees of freedom for residuals (n - k)
    pub df_resid: usize,
    /// Degrees of freedom for model (k - 1)
    pub df_model: usize,

    /// Coefficients with full statistics
    pub coefficients: Vec<OlsCoefficient>,

    // ═══════════════════════════════════════════════════════════════════════
    // Fit statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Residual standard error (sigma)
    pub residual_std_error: f64,
    /// F-statistic for overall significance
    pub f_statistic: f64,
    /// P-value for F-statistic
    pub f_p_value: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Standard error info
    // ═══════════════════════════════════════════════════════════════════════
    /// Type of standard errors used
    pub cov_type: CovarianceType,

    // ═══════════════════════════════════════════════════════════════════════
    // Warnings
    // ═══════════════════════════════════════════════════════════════════════
    /// Any warnings generated during estimation
    pub warnings: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════
    // Internal caches (not serialized)
    // ═══════════════════════════════════════════════════════════════════════
    /// Coefficient vector (for LinearEstimator trait)
    #[serde(skip)]
    pub(crate) beta: Array1<f64>,
    /// Standard errors vector
    #[serde(skip)]
    pub(crate) se: Array1<f64>,
    /// Residuals
    #[serde(skip)]
    pub(crate) resid: Array1<f64>,
    /// Variance-covariance matrix
    #[serde(skip)]
    pub(crate) vcov: Array2<f64>,
    /// (X'X)^{-1} for later computations
    #[serde(skip)]
    pub(crate) xtx_inv: Array2<f64>,
    /// Sum of squared residuals (cached for diagnostics)
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) ssr: f64,
    /// Total sum of squares (cached for diagnostics)
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) sst: f64,
}

impl LinearEstimator for OlsResult {
    fn coefficients(&self) -> &Array1<f64> {
        &self.beta
    }

    fn std_errors(&self) -> &Array1<f64> {
        &self.se
    }

    fn residuals(&self) -> &Array1<f64> {
        &self.resid
    }

    fn vcov_matrix(&self) -> &Array2<f64> {
        &self.vcov
    }

    fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    fn degrees_of_freedom(&self) -> usize {
        self.df_resid
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

impl std::fmt::Display for OlsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OLS Regression Results")?;
        writeln!(f, "======================")?;
        writeln!(f, "Dependent Variable: {}", self.dependent_var)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f, "F-statistic: {:.4} (p = {:.4})", self.f_statistic, self.f_p_value)?;
        writeln!(f, "Standard Errors: {}", self.cov_type)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Error", "t-value", "Pr(>|t|)"
        )?;
        writeln!(f, "{:-<65}", "")?;
        for coef in &self.coefficients {
            writeln!(
                f,
                "{:>15} {:>12.4} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&coef.name, 15),
                coef.estimate,
                coef.std_error,
                coef.t_value,
                coef.p_value,
                coef.significance.stars()
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

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

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// Run OLS regression on pre-extracted ndarray data.
///
/// This is the core computation function that operates directly on arrays,
/// avoiding DataFrame extraction overhead. Use this for benchmarking or when
/// you already have data in ndarray format.
///
/// # Arguments
/// * `x` - Design matrix (n × k), should include intercept column if desired
/// * `y` - Response vector (n × 1)
/// * `variable_names` - Names for each column in X (including intercept if present)
/// * `y_name` - Name of the dependent variable
/// * `cov_type` - Type of standard errors to compute
///
/// # Returns
/// An `OlsResult` containing the regression results.
pub fn run_ols_raw(
    x: &Array2<f64>,
    y: &Array1<f64>,
    variable_names: &[String],
    y_name: &str,
    cov_type: CovarianceType,
) -> EconResult<OlsResult> {
    let n = x.nrows();
    let k = x.ncols();

    if n != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "X has {} rows but y has {} elements",
                n, y.len()
            ),
        });
    }

    if variable_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "X has {} columns but {} variable names provided",
                k, variable_names.len()
            ),
        });
    }

    // Check we have enough observations
    if n <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n,
            context: format!("OLS regression with {} parameters", k),
        });
    }

    let mut warnings = Vec::new();

    // Check if first column is intercept (all 1s)
    let has_intercept = x.column(0).iter().all(|&v| (v - 1.0).abs() < 1e-10);

    // Compute (X'X)^{-1}
    let (xtx_inv, cond_warning) = safe_inverse(&xtx(&x.view()).view())
        .map_err(|_e| EconError::SingularMatrix {
            context: "X'X matrix in OLS".to_string(),
            suggestion: "Check for perfect multicollinearity between independent variables".to_string(),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(EstimationWarning::HighConditionNumber {
            value: cond,
            threshold: CONDITION_THRESHOLD,
        }.message());
    }

    // Compute β = (X'X)^{-1} X'y
    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // Compute fitted values and residuals
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = y - &y_hat;

    // Compute statistics
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ssr: f64 = residuals.iter().map(|&e| e.powi(2)).sum();
    let sse = sst - ssr; // Explained sum of squares

    let df_resid = n - k;
    let df_model = k - if has_intercept { 1 } else { 0 };

    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
    let adj_r_squared = if sst > 0.0 && df_resid > 0 {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df_resid as f64)
    } else {
        0.0
    };

    let sigma_squared = ssr / (df_resid as f64);
    let residual_std_error = sigma_squared.sqrt();

    // F-statistic
    let f_statistic = if df_model > 0 && ssr > 0.0 {
        (sse / df_model as f64) / sigma_squared
    } else {
        0.0
    };
    let f_p_value = if df_model > 0 && df_resid > 0 {
        f_test_p_value(f_statistic, df_model as f64, df_resid as f64)
    } else {
        f64::NAN
    };

    // Log-likelihood
    let log_likelihood = -0.5 * (n as f64) * (1.0 + (2.0 * std::f64::consts::PI * sigma_squared).ln());
    let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
    let bic = (k as f64) * (n as f64).ln() - 2.0 * log_likelihood;

    // Compute variance-covariance matrix based on cov_type
    let vcov = compute_vcov(&x.view(), &residuals, &xtx_inv, cov_type, n, k)?;

    // Extract standard errors from diagonal
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // Compute t-statistics and p-values
    let t_stats: Array1<f64> = &beta / &se;
    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();

    // Compute 95% confidence intervals
    let t_crit = crate::traits::t_critical(0.05, df_resid as f64);

    // Build coefficient results
    let coefficients: Vec<OlsCoefficient> = variable_names.iter()
        .enumerate()
        .map(|(i, name)| {
            let p = p_values[i];
            OlsCoefficient {
                name: name.clone(),
                estimate: beta[i],
                std_error: se[i],
                t_value: t_stats[i],
                p_value: p,
                significance: SignificanceLevel::from_p_value(p),
                ci_lower_95: beta[i] - t_crit * se[i],
                ci_upper_95: beta[i] + t_crit * se[i],
            }
        })
        .collect();

    Ok(OlsResult {
        dependent_var: y_name.to_string(),
        variable_names: variable_names.to_vec(),
        n_obs: n,
        n_params: k,
        df_resid,
        df_model,
        coefficients,
        r_squared,
        adj_r_squared,
        residual_std_error,
        f_statistic,
        f_p_value,
        log_likelihood,
        aic,
        bic,
        cov_type,
        warnings,
        beta,
        se,
        resid: residuals,
        vcov,
        xtx_inv,
        ssr,
        sst,
    })
}

/// Run OLS regression on a dataset.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `intercept` - Whether to include an intercept term
/// * `cov_type` - Type of standard errors to compute
///
/// # Returns
/// An `OlsResult` containing the regression results.
///
/// # Example
/// ```ignore
/// let result = run_ols(&dataset, "wage", &["education", "experience"], true, CovarianceType::HC1)?;
/// println!("{}", result);
/// ```
pub fn run_ols(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    intercept: bool,
    cov_type: CovarianceType,
) -> EconResult<OlsResult> {
    let df = dataset.df();

    // Build design matrix X
    let design = DesignMatrix::from_dataframe(df, x_cols, intercept)
        .map_err(|e| match e {
            DesignError::ColumnNotFound(c) => EconError::ColumnNotFound {
                column: c,
                available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            },
            DesignError::NonNumericColumn(c) => EconError::NonNumericColumn { column: c },
            DesignError::NullValues(c, indices) => EconError::NullValues {
                column: c,
                count: indices.len()
            },
            DesignError::EmptyDataset => EconError::EmptyDataset,
            DesignError::NoColumns => EconError::InvalidSpecification {
                message: "No independent variables specified".to_string(),
            },
            DesignError::PolarsError(e) => EconError::Internal(e.to_string()),
        })?;

    // Extract y vector
    let y = DesignMatrix::extract_column(df, y_col)
        .map_err(|e| match e {
            DesignError::ColumnNotFound(c) => EconError::ColumnNotFound {
                column: c,
                available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            },
            DesignError::NonNumericColumn(c) => EconError::NonNumericColumn { column: c },
            DesignError::NullValues(c, indices) => EconError::NullValues {
                column: c,
                count: indices.len(),
            },
            _ => EconError::Internal(e.to_string()),
        })?;

    let x = &design.data;
    let n = design.n_obs;
    let k = design.n_features;

    // Check we have enough observations
    if n <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n,
            context: format!("OLS regression with {} parameters", k),
        });
    }

    let mut warnings = Vec::new();

    // Compute (X'X)^{-1}
    let (xtx_inv, cond_warning) = safe_inverse(&xtx(&x.view()).view())
        .map_err(|_e| EconError::SingularMatrix {
            context: "X'X matrix in OLS".to_string(),
            suggestion: "Check for perfect multicollinearity between independent variables".to_string(),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(EstimationWarning::HighConditionNumber {
            value: cond,
            threshold: CONDITION_THRESHOLD,
        }.message());
    }

    // Compute β = (X'X)^{-1} X'y
    let xty_vec = xty(&x.view(), &y);
    let beta = xtx_inv.dot(&xty_vec);

    // Compute fitted values and residuals
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = &y - &y_hat;

    // Compute statistics
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ssr: f64 = residuals.iter().map(|&e| e.powi(2)).sum();
    let sse = sst - ssr; // Explained sum of squares

    let df_resid = n - k;
    let df_model = k - if intercept { 1 } else { 0 };

    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
    let adj_r_squared = if sst > 0.0 && df_resid > 0 {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df_resid as f64)
    } else {
        0.0
    };

    let sigma_squared = ssr / (df_resid as f64);
    let residual_std_error = sigma_squared.sqrt();

    // F-statistic
    let f_statistic = if df_model > 0 && ssr > 0.0 {
        (sse / df_model as f64) / sigma_squared
    } else {
        0.0
    };
    let f_p_value = if df_model > 0 && df_resid > 0 {
        f_test_p_value(f_statistic, df_model as f64, df_resid as f64)
    } else {
        f64::NAN
    };

    // Log-likelihood
    let log_likelihood = -0.5 * (n as f64) * (1.0 + (2.0 * std::f64::consts::PI * sigma_squared).ln());
    let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
    let bic = (k as f64) * (n as f64).ln() - 2.0 * log_likelihood;

    // Compute variance-covariance matrix based on cov_type
    let vcov = compute_vcov(&x.view(), &residuals, &xtx_inv, cov_type, n, k)?;

    // Extract standard errors from diagonal
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // Compute t-statistics and p-values
    let t_stats: Array1<f64> = &beta / &se;
    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();

    // Compute 95% confidence intervals
    let t_crit = crate::traits::t_critical(0.05, df_resid as f64);

    // Build coefficient results
    let coefficients: Vec<OlsCoefficient> = design.column_names.iter()
        .enumerate()
        .map(|(i, name)| {
            let p = p_values[i];
            OlsCoefficient {
                name: name.clone(),
                estimate: beta[i],
                std_error: se[i],
                t_value: t_stats[i],
                p_value: p,
                significance: SignificanceLevel::from_p_value(p),
                ci_lower_95: beta[i] - t_crit * se[i],
                ci_upper_95: beta[i] + t_crit * se[i],
            }
        })
        .collect();

    Ok(OlsResult {
        dependent_var: y_col.to_string(),
        variable_names: design.column_names.clone(),
        n_obs: n,
        n_params: k,
        df_resid,
        df_model,
        coefficients,
        r_squared,
        adj_r_squared,
        residual_std_error,
        f_statistic,
        f_p_value,
        log_likelihood,
        aic,
        bic,
        cov_type,
        warnings,
        beta,
        se,
        resid: residuals,
        vcov,
        xtx_inv,
        ssr,
        sst,
    })
}

/// Compute variance-covariance matrix based on the covariance type.
fn compute_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    cov_type: CovarianceType,
    n: usize,
    k: usize,
) -> EconResult<Array2<f64>> {
    match cov_type {
        CovarianceType::Standard => {
            // Standard OLS: σ² (X'X)^{-1}
            let sigma2 = residuals.iter().map(|&e| e.powi(2)).sum::<f64>() / ((n - k) as f64);
            Ok(xtx_inv * sigma2)
        }
        CovarianceType::HC0 | CovarianceType::HC1 | CovarianceType::HC2 | CovarianceType::HC3 => {
            // Robust standard errors using sandwich estimator
            compute_hc_vcov(x, residuals, xtx_inv, cov_type, n, k)
        }
    }
}

/// Compute heteroskedasticity-consistent variance-covariance matrix.
///
/// The sandwich estimator: (X'X)^{-1} X' Ω X (X'X)^{-1}
/// where Ω is a diagonal matrix of squared residuals (possibly adjusted).
fn compute_hc_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    cov_type: CovarianceType,
    n: usize,
    k: usize,
) -> EconResult<Array2<f64>> {
    // Compute leverage values for HC2/HC3
    let leverage = if matches!(cov_type, CovarianceType::HC2 | CovarianceType::HC3) {
        Some(compute_leverage(x, xtx_inv))
    } else {
        None
    };

    // Compute weights for each residual based on HC type
    let weights: Array1<f64> = (0..n)
        .map(|i| {
            let e = residuals[i];
            let e2 = e * e;
            match cov_type {
                CovarianceType::HC0 => e2,
                CovarianceType::HC1 => {
                    let correction = (n as f64) / ((n - k) as f64);
                    e2 * correction
                }
                CovarianceType::HC2 => {
                    let h = leverage.as_ref().unwrap()[i];
                    e2 / (1.0 - h)
                }
                CovarianceType::HC3 => {
                    let h = leverage.as_ref().unwrap()[i];
                    let denom = (1.0 - h).powi(2);
                    e2 / denom
                }
                _ => e2,
            }
        })
        .collect();

    // Compute meat of the sandwich: X' diag(weights) X
    // This can be computed as sum of w_i * x_i * x_i'
    let meat = compute_sandwich_meat(x, &weights);

    // Sandwich: (X'X)^{-1} * meat * (X'X)^{-1}
    let temp = matmul(&xtx_inv.view(), &meat.view())
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let vcov = matmul(&temp.view(), &xtx_inv.view())
        .map_err(|e| EconError::Internal(e.to_string()))?;

    Ok(vcov)
}

/// Compute leverage values: h_ii = x_i' (X'X)^{-1} x_i
fn compute_leverage(x: &ndarray::ArrayView2<f64>, xtx_inv: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    let k = x.ncols();

    // Parallel computation of leverage for each observation
    let leverage: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = x.row(i);
            let mut h = 0.0;
            for j in 0..k {
                for l in 0..k {
                    h += xi[j] * xtx_inv[[j, l]] * xi[l];
                }
            }
            h.clamp(0.0, 1.0 - 1e-10) // Prevent division by zero in HC2/HC3
        })
        .collect();

    Array1::from_vec(leverage)
}

/// Compute the meat of the sandwich estimator: X' diag(weights) X
fn compute_sandwich_meat(x: &ndarray::ArrayView2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();

    // Parallel accumulation
    let meat: Vec<Vec<f64>> = (0..k)
        .into_par_iter()
        .map(|j| {
            let mut row = vec![0.0; k];
            for i in 0..n {
                let w = weights[i];
                for l in 0..k {
                    row[l] += x[[i, j]] * w * x[[i, l]];
                }
            }
            row
        })
        .collect();

    let mut result = Array2::zeros((k, k));
    for j in 0..k {
        for l in 0..k {
            result[[j, l]] = meat[j][l];
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Clustered Standard Errors
// ═══════════════════════════════════════════════════════════════════════════

/// Result of OLS with clustered standard errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsClusteredResult {
    /// Base OLS result
    #[serde(flatten)]
    pub ols: OlsResult,
    /// Type of standard errors used
    pub cluster_type: String,
    /// Number of clusters (dimension 1)
    pub n_clusters_1: usize,
    /// Number of clusters (dimension 2, for two-way)
    pub n_clusters_2: Option<usize>,
    /// First cluster variable name
    pub cluster_var_1: String,
    /// Second cluster variable name (for two-way)
    pub cluster_var_2: Option<String>,
}

impl std::fmt::Display for OlsClusteredResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OLS Regression Results (Clustered)")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Dependent Variable: {}", self.ols.dependent_var)?;
        writeln!(f, "Observations: {}", self.ols.n_obs)?;
        writeln!(f, "Standard Errors: {}", self.cluster_type)?;
        write!(f, "Clusters: {}", self.n_clusters_1)?;
        if let Some(n2) = self.n_clusters_2 {
            writeln!(f, " x {}", n2)?;
        } else {
            writeln!(f)?;
        }
        writeln!(f, "R-squared: {:.4}", self.ols.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.ols.adj_r_squared)?;
        writeln!(f, "F-statistic: {:.4} (p = {:.4})", self.ols.f_statistic, self.ols.f_p_value)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Error", "t-value", "Pr(>|t|)"
        )?;
        writeln!(f, "{:-<65}", "")?;
        for coef in &self.ols.coefficients {
            writeln!(
                f,
                "{:>15} {:>12.4} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&coef.name, 15),
                coef.estimate,
                coef.std_error,
                coef.t_value,
                coef.p_value,
                coef.significance.stars()
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        if !self.ols.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.ols.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        Ok(())
    }
}

/// Run OLS regression with clustered standard errors.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `cluster1` - Column name for first clustering dimension (e.g., "firm_id")
/// * `cluster2` - Optional column name for second clustering dimension (e.g., "year")
///
/// # Returns
/// An `OlsClusteredResult` containing the regression results with clustered SEs.
pub fn run_ols_clustered(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    cluster1: &str,
    cluster2: Option<&str>,
) -> EconResult<OlsClusteredResult> {
    let df = dataset.df();

    // First run standard OLS to get coefficients and residuals
    let mut base_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    // Build design matrix again (we need it for clustered SEs)
    let design = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let x = &design.data;

    // Extract cluster IDs
    let clusters1 = extract_cluster_ids(df, cluster1)?;
    let n_clusters_1 = clusters1.values().count();

    if n_clusters_1 < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_clusters_1,
            context: "Need at least 2 clusters for clustered standard errors".to_string(),
        });
    }

    // Add warning for few clusters
    if n_clusters_1 < 10 {
        base_result.warnings.push(EstimationWarning::FewClusters {
            n_clusters: n_clusters_1,
            recommended: 10,
        }.message());
    }

    let (vcov, n_clusters_2, cluster_type) = if let Some(c2) = cluster2 {
        // Two-way clustering
        let clusters2 = extract_cluster_ids(df, c2)?;
        let n_c2 = clusters2.values().count();

        if n_c2 < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: n_c2,
                context: format!("Need at least 2 clusters for '{}'", c2),
            });
        }

        let vcov = compute_twoway_clustered_vcov(
            &x.view(),
            &base_result.resid,
            &base_result.xtx_inv,
            &clusters1,
            &clusters2,
        )?;

        (vcov, Some(n_c2), format!("Two-way clustered ({}, {})", cluster1, c2))
    } else {
        // One-way clustering
        let vcov = compute_clustered_vcov(
            &x.view(),
            &base_result.resid,
            &base_result.xtx_inv,
            &clusters1,
        )?;

        (vcov, None, format!("Clustered ({})", cluster1))
    };

    // Update standard errors and statistics
    let k = design.n_features;
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();
    let t_stats: Array1<f64> = &base_result.beta / &se;
    let df = base_result.df_resid;

    let t_crit = crate::traits::t_critical(0.05, df as f64);

    let coefficients: Vec<OlsCoefficient> = design.column_names.iter()
        .enumerate()
        .map(|(i, name)| {
            let p = t_test_p_value(t_stats[i], df as f64);
            OlsCoefficient {
                name: name.clone(),
                estimate: base_result.beta[i],
                std_error: se[i],
                t_value: t_stats[i],
                p_value: p,
                significance: SignificanceLevel::from_p_value(p),
                ci_lower_95: base_result.beta[i] - t_crit * se[i],
                ci_upper_95: base_result.beta[i] + t_crit * se[i],
            }
        })
        .collect();

    // Update the result
    base_result.coefficients = coefficients;
    base_result.se = se;
    base_result.vcov = vcov;

    Ok(OlsClusteredResult {
        ols: base_result,
        cluster_type,
        n_clusters_1,
        n_clusters_2,
        cluster_var_1: cluster1.to_string(),
        cluster_var_2: cluster2.map(|s| s.to_string()),
    })
}

/// Extract cluster IDs from a column.
/// Returns a mapping from cluster identifier to vector of row indices.
fn extract_cluster_ids(
    df: &DataFrame,
    col: &str,
) -> EconResult<HashMap<String, Vec<usize>>> {
    let column = df.column(col)
        .map_err(|_| EconError::ColumnNotFound {
            column: col.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    let mut clusters: HashMap<String, Vec<usize>> = HashMap::new();

    for i in 0..column.len() {
        let key = format!("{:?}", column.get(i).unwrap());
        clusters.entry(key).or_default().push(i);
    }

    Ok(clusters)
}

/// Compute one-way clustered variance-covariance matrix.
///
/// Uses the cluster-robust sandwich estimator with small-sample correction:
/// V = (G / (G-1)) * ((n-1)/(n-k)) * (X'X)^{-1} * B * (X'X)^{-1}
///
/// where B = sum_g (X_g' e_g) (X_g' e_g)'
fn compute_clustered_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    clusters: &HashMap<String, Vec<usize>>,
) -> EconResult<Array2<f64>> {
    let n = x.nrows();
    let k = x.ncols();
    let g = clusters.len();

    // Small sample correction
    let correction = (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / (n - k) as f64);

    // Compute meat: sum over clusters of (X_g' e_g)(X_g' e_g)'
    let meat: Array2<f64> = clusters
        .par_iter()
        .map(|(_, indices)| {
            // Compute X_g' e_g for this cluster
            let mut xe = vec![0.0; k];
            for &i in indices {
                let e = residuals[i];
                for j in 0..k {
                    xe[j] += x[[i, j]] * e;
                }
            }

            // Outer product
            let mut outer = Array2::zeros((k, k));
            for j in 0..k {
                for l in 0..k {
                    outer[[j, l]] = xe[j] * xe[l];
                }
            }
            outer
        })
        .reduce(|| Array2::zeros((k, k)), |a, b| a + b);

    // Sandwich: (X'X)^{-1} * meat * (X'X)^{-1}
    let temp = matmul(&xtx_inv.view(), &meat.view())
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let vcov = matmul(&temp.view(), &xtx_inv.view())
        .map_err(|e| EconError::Internal(e.to_string()))?;

    Ok(vcov * correction)
}

/// Compute two-way clustered variance-covariance matrix.
///
/// Uses Cameron, Gelbach, Miller (2011) formula:
/// V = V_1 + V_2 - V_12
///
/// where V_1 is clustered on dimension 1, V_2 on dimension 2,
/// and V_12 on the intersection.
fn compute_twoway_clustered_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    clusters1: &HashMap<String, Vec<usize>>,
    clusters2: &HashMap<String, Vec<usize>>,
) -> EconResult<Array2<f64>> {
    // Compute V_1 (clustered on dimension 1)
    let v1 = compute_clustered_vcov(x, residuals, xtx_inv, clusters1)?;

    // Compute V_2 (clustered on dimension 2)
    let v2 = compute_clustered_vcov(x, residuals, xtx_inv, clusters2)?;

    // Compute intersection clusters
    let intersection = compute_intersection_clusters(clusters1, clusters2);
    let v12 = compute_clustered_vcov(x, residuals, xtx_inv, &intersection)?;

    // V = V_1 + V_2 - V_12
    Ok(&v1 + &v2 - &v12)
}

/// Compute intersection of two clustering dimensions.
fn compute_intersection_clusters(
    clusters1: &HashMap<String, Vec<usize>>,
    clusters2: &HashMap<String, Vec<usize>>,
) -> HashMap<String, Vec<usize>> {
    let n = clusters1.values().map(|v| v.len()).sum::<usize>();

    // Create reverse lookup for cluster2
    let mut idx_to_c2: Vec<String> = vec![String::new(); n];
    for (key, indices) in clusters2 {
        for &i in indices {
            idx_to_c2[i] = key.clone();
        }
    }

    // Create intersection clusters
    let mut intersection: HashMap<String, Vec<usize>> = HashMap::new();
    for (c1_key, indices) in clusters1 {
        for &i in indices {
            let key = format!("{}-{}", c1_key, idx_to_c2[i]);
            intersection.entry(key).or_default().push(i);
        }
    }

    intersection
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // y = x1 + noise (not perfect linear relationship)
        let df = df! {
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "cluster" => ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_basic_ols() {
        let dataset = create_test_dataset();
        let result = run_ols(&dataset, "y", &["x1"], true, CovarianceType::Standard).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_params, 2); // intercept + x1
        assert!(result.r_squared > 0.98); // Very strong linear relationship
    }

    #[test]
    fn test_robust_se() {
        let dataset = create_test_dataset();
        let result = run_ols(&dataset, "y", &["x1"], true, CovarianceType::HC1).unwrap();

        assert_eq!(result.cov_type, CovarianceType::HC1);
        assert!(result.coefficients[1].std_error > 0.0);
    }

    #[test]
    fn test_multiple_regressors() {
        let dataset = create_test_dataset();
        let result = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard).unwrap();

        assert_eq!(result.n_params, 3); // intercept + x1 + x2
        assert_eq!(result.coefficients.len(), 3);
    }

    #[test]
    fn test_column_not_found() {
        let dataset = create_test_dataset();
        let result = run_ols(&dataset, "nonexistent", &["x1"], true, CovarianceType::Standard);

        assert!(matches!(result, Err(EconError::ColumnNotFound { .. })));
    }

    #[test]
    fn test_clustered_se() {
        let dataset = create_test_dataset();
        let result = run_ols_clustered(&dataset, "y", &["x1"], "cluster", None).unwrap();

        assert_eq!(result.n_clusters_1, 3);
        assert!(result.ols.coefficients[1].std_error > 0.0);
    }
}
