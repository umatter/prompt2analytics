//! Ordinary Least Squares (OLS) regression with robust standard errors.
//!
//! Provides pure Rust implementation of OLS regression with:
//! - Standard OLS estimation: β = (X'X)⁻¹ X'y
//! - Heteroskedasticity-robust standard errors (HC0, HC1, HC2, HC3)
//! - Clustered standard errors (one-way and two-way)
//! - Full diagnostics and fit statistics
//!
//! # Mathematical Background
//!
//! The OLS estimator minimizes the sum of squared residuals:
//!
//! β̂ = argmin_β ||y - Xβ||² = (X'X)⁻¹ X'y
//!
//! Under the Gauss-Markov assumptions, OLS is the Best Linear Unbiased Estimator (BLUE).
//!
//! ## Robust Standard Errors
//!
//! The heteroskedasticity-consistent (HC) covariance estimators are:
//! - **HC0** (White): V = (X'X)⁻¹ X' diag(e²) X (X'X)⁻¹
//! - **HC1**: HC0 × n/(n-k) small-sample correction
//! - **HC2**: Uses e²ᵢ/(1-hᵢᵢ) where hᵢᵢ is leverage
//! - **HC3**: Uses e²ᵢ/(1-hᵢᵢ)² (most conservative)
//!
//! ## Clustered Standard Errors
//!
//! For clustered data with G groups:
//! V = (X'X)⁻¹ (Σᵍ Xᵍ'eᵍeᵍ'Xᵍ) (X'X)⁻¹ × G/(G-1) × (n-1)/(n-k)
//!
//! # References
//!
//! - Gauss, C.F. (1821). *Theoria combinationis observationum erroribus minimis obnoxiae*.
//!   The original derivation of least squares estimation.
//!
//! - White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and
//!   a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
//!   https://doi.org/10.2307/1912934
//!
//! - MacKinnon, J.G., & White, H. (1985). Some heteroskedasticity-consistent covariance
//!   matrix estimators with improved finite sample properties. *Journal of Econometrics*,
//!   29(3), 305-325. https://doi.org/10.1016/0304-4076(85)90158-7
//!
//! - Liang, K.Y., & Zeger, S.L. (1986). Longitudinal data analysis using generalized
//!   linear models. *Biometrika*, 73(1), 13-22. https://doi.org/10.1093/biomet/73.1.13
//!
//! - Cameron, A.C., & Miller, D.L. (2015). A practitioner's guide to cluster-robust
//!   inference. *Journal of Human Resources*, 50(2), 317-372.
//!   https://doi.org/10.3368/jhr.50.2.317
//!
//! - Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
//!   (2nd ed.). MIT Press. ISBN: 978-0262232586.
//!
//! R equivalent: `stats::lm()`, `sandwich::vcovHC()`, `sandwich::vcovCL()`

use ndarray::{Array1, Array2};
use polars::prelude::*;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
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

// ============================================================================
// HAC (Newey-West) Standard Errors
// ============================================================================

/// Result of HAC covariance estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HacResult {
    /// HAC-adjusted variance-covariance matrix
    pub vcov: Vec<Vec<f64>>,
    /// HAC-adjusted standard errors
    pub std_errors: Vec<f64>,
    /// Bandwidth (number of lags) used
    pub bandwidth: usize,
    /// Kernel type used
    pub kernel: HacKernel,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Original OLS coefficients (for reference)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub names: Vec<String>,
    /// Whether the prewhitening option was used
    pub prewhiten: bool,
}

impl std::fmt::Display for HacResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HAC (Newey-West) Standard Errors")?;
        writeln!(f, "================================")?;
        writeln!(f, "Kernel: {}", self.kernel)?;
        writeln!(f, "Bandwidth: {}", self.bandwidth)?;
        writeln!(f, "Prewhitening: {}", self.prewhiten)?;
        writeln!(f)?;
        writeln!(f, "{:<15} {:>12} {:>12}", "Variable", "Coef", "HAC SE")?;
        writeln!(f, "{:-<15} {:-<12} {:-<12}", "", "", "")?;
        for (i, name) in self.names.iter().enumerate() {
            writeln!(f, "{:<15} {:>12.6} {:>12.6}",
                name,
                self.coefficients[i],
                self.std_errors[i]
            )?;
        }
        Ok(())
    }
}

/// Kernel type for HAC estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HacKernel {
    /// Bartlett (Newey-West) kernel: w(j) = 1 - j/(m+1) for j <= m
    #[default]
    Bartlett,
    /// Parzen kernel (Andrews, 1991)
    Parzen,
    /// Quadratic Spectral kernel (Andrews, 1991)
    QuadraticSpectral,
    /// Truncated (rectangular) kernel
    Truncated,
    /// Tukey-Hanning kernel
    TukeyHanning,
}

impl std::fmt::Display for HacKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bartlett => write!(f, "Bartlett (Newey-West)"),
            Self::Parzen => write!(f, "Parzen"),
            Self::QuadraticSpectral => write!(f, "Quadratic Spectral"),
            Self::Truncated => write!(f, "Truncated"),
            Self::TukeyHanning => write!(f, "Tukey-Hanning"),
        }
    }
}

impl HacKernel {
    /// Parse kernel type from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bartlett" | "newey-west" | "nw" => Some(Self::Bartlett),
            "parzen" => Some(Self::Parzen),
            "quadratic-spectral" | "qs" | "quadraticspectral" => Some(Self::QuadraticSpectral),
            "truncated" | "rectangular" => Some(Self::Truncated),
            "tukey-hanning" | "tukeyhanning" | "th" => Some(Self::TukeyHanning),
            _ => None,
        }
    }

    /// Compute the kernel weight for a given lag j and bandwidth m.
    pub fn weight(&self, j: usize, m: usize) -> f64 {
        if j == 0 {
            return 1.0;
        }
        let x = j as f64 / (m as f64 + 1.0);
        match self {
            Self::Bartlett => {
                if j <= m { 1.0 - x } else { 0.0 }
            }
            Self::Parzen => {
                let z = j as f64 / m as f64;
                if z <= 0.5 {
                    1.0 - 6.0 * z.powi(2) + 6.0 * z.powi(3)
                } else if z <= 1.0 {
                    2.0 * (1.0 - z).powi(3)
                } else {
                    0.0
                }
            }
            Self::QuadraticSpectral => {
                let z = 6.0 * std::f64::consts::PI * j as f64 / (5.0 * m as f64);
                let t1 = 3.0 / z.powi(2);
                let t2 = z.sin() / z - z.cos();
                t1 * t2
            }
            Self::Truncated => {
                if j <= m { 1.0 } else { 0.0 }
            }
            Self::TukeyHanning => {
                if j <= m {
                    0.5 * (1.0 + (std::f64::consts::PI * j as f64 / m as f64).cos())
                } else {
                    0.0
                }
            }
        }
    }
}

/// Compute HAC (Heteroskedasticity and Autocorrelation Consistent) covariance matrix.
///
/// Implements the Newey-West (1987) estimator with various kernel options.
/// This is the R equivalent of `sandwich::vcovHAC()`.
///
/// # Mathematical Background
///
/// The HAC covariance estimator is:
/// V_HAC = (X'X)⁻¹ Ω̂ (X'X)⁻¹
///
/// where Ω̂ = Σⱼ₌₋ₘᵐ w(j/m) Γ̂ⱼ
///
/// and Γ̂ⱼ = (1/n) Σᵢ uᵢ uᵢ₊ⱼ xᵢ xᵢ₊ⱼ'
///
/// # Arguments
///
/// * `ols_result` - OLS regression result
/// * `x` - Design matrix (n × k)
/// * `bandwidth` - Optional bandwidth (number of lags). If None, uses Newey-West automatic selection.
/// * `kernel` - Kernel type (default: Bartlett/Newey-West)
/// * `prewhiten` - Whether to use VAR(1) prewhitening (default: false)
///
/// # Returns
///
/// `HacResult` containing HAC-adjusted standard errors and variance-covariance matrix.
///
/// # References
///
/// - Newey, W.K., & West, K.D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity
///   and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708.
///   https://doi.org/10.2307/1913610
///
/// - Andrews, D.W.K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance
///   Matrix Estimation. *Econometrica*, 59(3), 817-858.
///   https://doi.org/10.2307/2938229
///
/// R equivalent: `sandwich::vcovHAC()`, `sandwich::NeweyWest()`
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::{run_ols, vcov_hac, HacKernel, CovarianceType};
///
/// let ols = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard)?;
/// let hac = vcov_hac(&ols, &x_matrix, None, HacKernel::Bartlett, false)?;
/// println!("HAC standard errors: {:?}", hac.std_errors);
/// ```
pub fn vcov_hac(
    ols_result: &OlsResult,
    x: &Array2<f64>,
    bandwidth: Option<usize>,
    kernel: HacKernel,
    prewhiten: bool,
) -> EconResult<HacResult> {
    let n = ols_result.n_obs;
    let k = ols_result.n_params;
    let residuals = &ols_result.resid;

    // Validate inputs
    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!("Design matrix rows ({}) does not match number of observations ({})", x.nrows(), n),
        });
    }
    if x.ncols() != k {
        return Err(EconError::InvalidSpecification {
            message: format!("Design matrix columns ({}) does not match number of parameters ({})", x.ncols(), k),
        });
    }

    // Automatic bandwidth selection (Newey-West rule of thumb)
    let bw = bandwidth.unwrap_or_else(|| {
        // NW rule: floor(4 * (n/100)^(2/9))
        let nw_bw = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
        nw_bw.max(1).min(n - 1)
    });

    // Compute X'X and its inverse
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _cond) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "HAC covariance computation".to_string(),
        suggestion: format!("Original error: {}", e),
    })?;

    // Optionally prewhiten residuals with VAR(1)
    let (u, rho) = if prewhiten {
        prewhiten_residuals(residuals)
    } else {
        (residuals.clone(), 0.0)
    };

    // Compute score vectors: s_i = u_i * x_i (n × k matrix)
    let mut scores = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            scores[[i, j]] = u[i] * x[[i, j]];
        }
    }

    // Compute the meat of the sandwich: Ω̂ = Σⱼ w(j) (Γ̂ⱼ + Γ̂ⱼ')
    let mut omega = Array2::<f64>::zeros((k, k));

    // Lag 0: Γ̂₀ = Σᵢ sᵢ sᵢ'
    for i in 0..n {
        for j in 0..k {
            for l in 0..k {
                omega[[j, l]] += scores[[i, j]] * scores[[i, l]];
            }
        }
    }

    // Lags 1 to bw: add cross-products with kernel weights
    for lag in 1..=bw {
        let w = kernel.weight(lag, bw);
        if w.abs() < 1e-15 {
            continue;
        }

        let mut gamma_lag = Array2::<f64>::zeros((k, k));
        for i in lag..n {
            for j in 0..k {
                for l in 0..k {
                    gamma_lag[[j, l]] += scores[[i, j]] * scores[[i - lag, l]];
                }
            }
        }

        // Add symmetrically (Γ̂ⱼ + Γ̂ⱼ')
        for j in 0..k {
            for l in 0..k {
                omega[[j, l]] += w * (gamma_lag[[j, l]] + gamma_lag[[l, j]]);
            }
        }
    }

    // Scale by 1/n
    omega /= n as f64;

    // If prewhitening was used, recolor the covariance matrix
    if prewhiten && rho.abs() > 1e-10 {
        // Recolor: V = (1 - ρ)^(-2) V_prewhitened
        let recolor_factor: f64 = 1.0 / (1.0 - rho).powi(2);
        omega *= recolor_factor;
    }

    // Compute the sandwich: V = (X'X)⁻¹ Ω̂ (X'X)⁻¹
    let meat_bread = matmul(&omega.view(), &xtx_inv.view()).map_err(|e| EconError::SingularMatrix {
        context: "HAC sandwich computation".to_string(),
        suggestion: format!("Matrix multiplication error: {}", e),
    })?;
    let vcov_hac = matmul(&xtx_inv.view(), &meat_bread.view()).map_err(|e| EconError::SingularMatrix {
        context: "HAC sandwich computation".to_string(),
        suggestion: format!("Matrix multiplication error: {}", e),
    })?;

    // Scale by n (to get variance, not sum of squares)
    let vcov_scaled = &vcov_hac * (n as f64);

    // Extract standard errors (square root of diagonal)
    let std_errors: Vec<f64> = (0..k)
        .map(|i| {
            let var = vcov_scaled[[i, i]];
            if var >= 0.0 { var.sqrt() } else { 0.0 }
        })
        .collect();

    // Convert vcov to Vec<Vec<f64>> for output
    let vcov_vec: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_scaled[[i, j]]).collect())
        .collect();

    // Extract coefficient values and names
    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result.coefficients.iter().map(|c| c.name.clone()).collect();

    Ok(HacResult {
        vcov: vcov_vec,
        std_errors,
        bandwidth: bw,
        kernel,
        n_obs: n,
        n_params: k,
        coefficients,
        names,
        prewhiten,
    })
}

/// Prewhiten residuals using AR(1) model.
fn prewhiten_residuals(residuals: &Array1<f64>) -> (Array1<f64>, f64) {
    let n = residuals.len();
    if n < 3 {
        return (residuals.clone(), 0.0);
    }

    // Estimate AR(1) coefficient: ρ = Σᵢ u_{i} u_{i-1} / Σᵢ u_{i-1}²
    let mut sum_lag_prod = 0.0;
    let mut sum_lag_sq = 0.0;
    for i in 1..n {
        sum_lag_prod += residuals[i] * residuals[i - 1];
        sum_lag_sq += residuals[i - 1] * residuals[i - 1];
    }

    let rho = if sum_lag_sq > 1e-15 {
        (sum_lag_prod / sum_lag_sq).clamp(-0.99, 0.99)
    } else {
        0.0
    };

    // Prewhitened residuals: v_i = u_i - ρ u_{i-1}
    let mut prewhitened = Array1::<f64>::zeros(n - 1);
    for i in 1..n {
        prewhitened[i - 1] = residuals[i] - rho * residuals[i - 1];
    }

    // Pad to original length for consistency
    let mut result = Array1::<f64>::zeros(n);
    result[0] = residuals[0];
    for i in 1..n {
        result[i] = prewhitened[i - 1];
    }

    (result, rho)
}

/// Convenience function to compute HAC standard errors for a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `bandwidth` - Optional bandwidth (None for automatic)
/// * `kernel` - Kernel type string ("bartlett", "parzen", "qs", "truncated", "tukey-hanning")
/// * `prewhiten` - Whether to use prewhitening
///
/// # Returns
///
/// `HacResult` with HAC-adjusted standard errors.
///
/// R equivalent: `sandwich::NeweyWest(lm(y ~ x), lag = bandwidth, prewhite = prewhiten)`
pub fn run_vcov_hac(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    bandwidth: Option<usize>,
    kernel: Option<&str>,
    prewhiten: bool,
) -> EconResult<HacResult> {
    // First run OLS
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = dm.view().to_owned();

    // Parse kernel
    let kernel_type = kernel
        .and_then(HacKernel::from_str)
        .unwrap_or(HacKernel::Bartlett);

    vcov_hac(&ols_result, &x, bandwidth, kernel_type, prewhiten)
}

// ============================================================================
// Bootstrap Covariance Estimation (vcovBS)
// ============================================================================

/// Bootstrap type for covariance estimation.
///
/// R equivalent: `sandwich::vcovBS(..., type = "xy")` or `type = "residual"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BootstrapType {
    /// Pairs bootstrap (xy): Resample (y, X) pairs together.
    /// More robust to misspecification, widely applicable.
    #[default]
    Pairs,
    /// Residual bootstrap: Resample residuals, keeping X fixed.
    /// More efficient under correct specification but assumes homoskedasticity.
    Residual,
    /// Wild bootstrap: Multiply residuals by random weights (Rademacher).
    /// Robust to heteroskedasticity, preserves X structure.
    Wild,
}

impl BootstrapType {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pairs" | "xy" | "case" => Some(BootstrapType::Pairs),
            "residual" | "resid" => Some(BootstrapType::Residual),
            "wild" => Some(BootstrapType::Wild),
            _ => None,
        }
    }
}

impl std::fmt::Display for BootstrapType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootstrapType::Pairs => write!(f, "Pairs (xy)"),
            BootstrapType::Residual => write!(f, "Residual"),
            BootstrapType::Wild => write!(f, "Wild"),
        }
    }
}

/// Result from bootstrap covariance estimation.
///
/// R equivalent: Output from `sandwich::vcovBS()`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Bootstrap covariance matrix (k × k).
    pub vcov: Vec<Vec<f64>>,
    /// Bootstrap standard errors.
    pub std_errors: Vec<f64>,
    /// Number of bootstrap replications.
    pub n_boot: usize,
    /// Bootstrap type used.
    pub bootstrap_type: BootstrapType,
    /// Original coefficient estimates.
    pub coefficients: Vec<f64>,
    /// Coefficient names.
    pub names: Vec<String>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of parameters.
    pub n_params: usize,
    /// Bootstrap coefficient samples (optional, for diagnostics).
    #[serde(skip)]
    pub boot_samples: Option<Vec<Vec<f64>>>,
    /// Convergence rate (fraction of successful replications).
    pub convergence_rate: f64,
}

impl std::fmt::Display for BootstrapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bootstrap Covariance Estimation")?;
        writeln!(f, "================================")?;
        writeln!(f, "Method: {} bootstrap", self.bootstrap_type)?;
        writeln!(f, "Replications: {}", self.n_boot)?;
        writeln!(f, "Convergence rate: {:.1}%", self.convergence_rate * 100.0)?;
        writeln!(f)?;
        writeln!(f, "Bootstrap Standard Errors:")?;
        for (i, name) in self.names.iter().enumerate() {
            writeln!(f, "  {:15} {:12.6} (SE: {:8.6})",
                     name, self.coefficients[i], self.std_errors[i])?;
        }
        Ok(())
    }
}

/// Compute bootstrap covariance matrix for OLS regression.
///
/// # Arguments
///
/// * `ols_result` - Result from OLS estimation
/// * `x` - Design matrix (n × k)
/// * `y` - Response vector (n)
/// * `n_boot` - Number of bootstrap replications (default: 999)
/// * `bootstrap_type` - Type of bootstrap (Pairs, Residual, Wild)
/// * `seed` - Optional RNG seed for reproducibility
///
/// # Returns
///
/// `BootstrapResult` with bootstrap covariance matrix and standard errors.
///
/// # References
///
/// - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."
///   Annals of Statistics, 7(1), 1-26.
/// - Wu, C. F. J. (1986). "Jackknife, Bootstrap and Other Resampling Methods
///   in Regression Analysis." Annals of Statistics, 14(4), 1261-1295.
/// - MacKinnon, J. G. (2006). "Bootstrap Methods in Econometrics."
///   Economic Record, 82, S2-S18.
///
/// # R Equivalent
///
/// ```r
/// library(sandwich)
/// vcovBS(lm(y ~ x), R = 999, type = "xy")
/// ```
pub fn vcov_bootstrap(
    ols_result: &OlsResult,
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_boot: Option<usize>,
    bootstrap_type: BootstrapType,
    seed: Option<u64>,
) -> EconResult<BootstrapResult> {
    let n = x.nrows();
    let k = x.ncols();
    let replications = n_boot.unwrap_or(999);

    if n < k {
        return Err(EconError::InsufficientData {
            required: k,
            provided: n,
            context: "Bootstrap covariance requires n >= k".to_string(),
        });
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Original fitted values and residuals
    let y_fitted = x.dot(&ols_result.coefficients().clone());
    let residuals = ols_result.residuals();

    // Storage for bootstrap coefficient estimates
    let mut boot_coefs: Vec<Vec<f64>> = Vec::with_capacity(replications);
    let indices: Vec<usize> = (0..n).collect();

    for _ in 0..replications {
        let (y_boot, x_boot) = match bootstrap_type {
            BootstrapType::Pairs => {
                // Pairs bootstrap: resample (y_i, x_i) pairs together
                let sample: Vec<usize> = (0..n)
                    .map(|_| *indices.choose(&mut rng).unwrap())
                    .collect();

                let y_boot: Array1<f64> = sample.iter().map(|&i| y[i]).collect();
                let x_boot: Array2<f64> = Array2::from_shape_fn((n, k), |(i, j)| {
                    x[[sample[i], j]]
                });
                (y_boot, x_boot)
            }
            BootstrapType::Residual => {
                // Residual bootstrap: y* = X*β̂ + ε*, where ε* resampled from residuals
                let resid_sample: Vec<f64> = (0..n)
                    .map(|_| residuals[*indices.choose(&mut rng).unwrap()])
                    .collect();

                let y_boot: Array1<f64> = (0..n)
                    .map(|i| y_fitted[i] + resid_sample[i])
                    .collect();

                (y_boot, x.clone())
            }
            BootstrapType::Wild => {
                // Wild bootstrap with Rademacher weights
                let weights: Vec<f64> = (0..n)
                    .map(|_| if rng.r#gen::<bool>() { 1.0 } else { -1.0 })
                    .collect();

                let y_boot: Array1<f64> = (0..n)
                    .map(|i| y_fitted[i] + weights[i] * residuals[i])
                    .collect();

                (y_boot, x.clone())
            }
        };

        // Estimate OLS on bootstrap sample
        let xtx_boot = xtx(&x_boot.view());
        match safe_inverse(&xtx_boot.view()) {
            Ok((xtx_inv, _)) => {
                let xty_boot = xty(&x_boot.view(), &y_boot);
                let beta_boot = xtx_inv.dot(&xty_boot);
                boot_coefs.push(beta_boot.to_vec());
            }
            Err(_) => {
                // Skip singular samples
                continue;
            }
        }
    }

    let successful_reps = boot_coefs.len();
    if successful_reps < 10 {
        return Err(EconError::ConvergenceFailure {
            iterations: replications,
            last_change: successful_reps as f64 / replications as f64,
            suggestion: format!(
                "Bootstrap failed: only {} of {} replications converged. Try pairs bootstrap or increase sample size.",
                successful_reps, replications
            ),
        });
    }

    // Compute bootstrap covariance matrix
    // Cov(β) = (1/(B-1)) * Σᵢ (β*ᵢ - β̄*)(β*ᵢ - β̄*)'
    let boot_mean: Vec<f64> = (0..k)
        .map(|j| boot_coefs.iter().map(|b| b[j]).sum::<f64>() / successful_reps as f64)
        .collect();

    let mut vcov = vec![vec![0.0; k]; k];
    for boot in &boot_coefs {
        for i in 0..k {
            for j in 0..k {
                vcov[i][j] += (boot[i] - boot_mean[i]) * (boot[j] - boot_mean[j]);
            }
        }
    }

    // Normalize by (B-1)
    let divisor = (successful_reps - 1) as f64;
    for i in 0..k {
        for j in 0..k {
            vcov[i][j] /= divisor;
        }
    }

    // Standard errors from diagonal
    let std_errors: Vec<f64> = (0..k)
        .map(|i| vcov[i][i].max(0.0).sqrt())
        .collect();

    // Extract original coefficients and names
    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result.coefficients.iter().map(|c| c.name.clone()).collect();

    Ok(BootstrapResult {
        vcov,
        std_errors,
        n_boot: successful_reps,
        bootstrap_type,
        coefficients,
        names,
        n_obs: n,
        n_params: k,
        boot_samples: Some(boot_coefs),
        convergence_rate: successful_reps as f64 / replications as f64,
    })
}

/// Convenience function to compute bootstrap covariance for a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `n_boot` - Number of bootstrap replications (default: 999)
/// * `bootstrap_type` - Type string ("pairs", "residual", "wild")
/// * `seed` - Optional RNG seed
///
/// # Returns
///
/// `BootstrapResult` with bootstrap covariance matrix.
///
/// # R Equivalent
///
/// ```r
/// library(sandwich)
/// vcovBS(lm(y ~ x, data), R = 999, type = "xy")
/// ```
pub fn run_vcov_bootstrap(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    n_boot: Option<usize>,
    bootstrap_type: Option<&str>,
    seed: Option<u64>,
) -> EconResult<BootstrapResult> {
    // First run OLS
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = dm.view().to_owned();

    // Extract y
    let y_series = dataset.df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound { column: y_col.to_string(), available: vec![] })?;
    let y: Array1<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn { column: y_col.to_string() })?
        .into_no_null_iter()
        .collect();

    // Parse bootstrap type
    let boot_type = bootstrap_type
        .and_then(BootstrapType::from_str)
        .unwrap_or(BootstrapType::Pairs);

    vcov_bootstrap(&ols_result, &x, &y, n_boot, boot_type, seed)
}

// ============================================================================
// Driscoll-Kraay Panel-Robust Standard Errors (vcovPL)
// ============================================================================

/// Result from Driscoll-Kraay panel covariance estimation.
///
/// R equivalent: Output from `sandwich::vcovPL()`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriscollKraayResult {
    /// Panel-robust covariance matrix (k × k).
    pub vcov: Vec<Vec<f64>>,
    /// Panel-robust standard errors.
    pub std_errors: Vec<f64>,
    /// Coefficient estimates.
    pub coefficients: Vec<f64>,
    /// t-statistics.
    pub t_stats: Vec<f64>,
    /// p-values.
    pub p_values: Vec<f64>,
    /// Coefficient names.
    pub names: Vec<String>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of parameters.
    pub n_params: usize,
    /// Number of time periods.
    pub n_periods: usize,
    /// Number of cross-sectional units.
    pub n_units: usize,
    /// Bandwidth used.
    pub bandwidth: usize,
    /// Kernel type used.
    pub kernel: HacKernel,
}

impl std::fmt::Display for DriscollKraayResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Driscoll-Kraay Panel-Robust Standard Errors")?;
        writeln!(f, "============================================")?;
        writeln!(f, "N = {} observations, T = {} periods, N_units = {}",
                 self.n_obs, self.n_periods, self.n_units)?;
        writeln!(f, "Bandwidth: {}, Kernel: {}", self.bandwidth, self.kernel)?;
        writeln!(f)?;
        writeln!(f, "{:>15} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std.Err", "t", "P>|t|")?;
        writeln!(f, "{}", "-".repeat(65))?;
        for (i, name) in self.names.iter().enumerate() {
            let sig = crate::traits::estimator::SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(f, "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                     name, self.coefficients[i], self.std_errors[i],
                     self.t_stats[i], self.p_values[i], sig.stars())?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        writeln!(f)?;
        writeln!(f, "Standard errors robust to cross-sectional and serial correlation.")?;
        Ok(())
    }
}

/// Compute Driscoll-Kraay panel-robust covariance matrix.
///
/// Implements the Driscoll and Kraay (1998) estimator for panel data that is
/// robust to arbitrary cross-sectional correlation and serial correlation up
/// to a specified lag.
///
/// # Algorithm
///
/// 1. Aggregate moment conditions (score vectors) by time period:
///    h̄_t = (1/N_t) Σᵢ uᵢₜ xᵢₜ
///
/// 2. Apply Newey-West HAC estimation to the time series of aggregated moments:
///    Ω̂ = Σⱼ w(j) (Γ̂ⱼ + Γ̂ⱼ')
///    where Γ̂ⱼ = (1/T) Σₜ h̄_t h̄_{t-j}'
///
/// 3. Scale and compute sandwich:
///    V = (X'X)⁻¹ (T × Ω̂) (X'X)⁻¹
///
/// # Arguments
///
/// * `ols_result` - Result from OLS estimation
/// * `x` - Design matrix (n × k)
/// * `time_ids` - Time period identifier for each observation
/// * `bandwidth` - Optional bandwidth (None for automatic)
/// * `kernel` - Kernel type for HAC
///
/// # Returns
///
/// `DriscollKraayResult` with panel-robust covariance matrix.
///
/// # References
///
/// - Driscoll, J.C. & Kraay, A.C. (1998). "Consistent Covariance Matrix
///   Estimation with Spatially Dependent Panel Data." Review of Economics
///   and Statistics, 80(4), 549-560.
///
/// - Hoechle, D. (2007). "Robust Standard Errors for Panel Regressions with
///   Cross-Sectional Dependence." Stata Journal, 7(3), 281-312.
///
/// R equivalent: `sandwich::vcovPL()`
pub fn vcov_driscoll_kraay(
    ols_result: &OlsResult,
    x: &Array2<f64>,
    time_ids: &[i64],
    bandwidth: Option<usize>,
    kernel: HacKernel,
) -> EconResult<DriscollKraayResult> {
    let n = ols_result.n_obs;
    let k = ols_result.n_params;
    let residuals = &ols_result.resid;

    // Validate inputs
    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!("Design matrix rows ({}) != observations ({})", x.nrows(), n),
        });
    }
    if time_ids.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!("Time IDs length ({}) != observations ({})", time_ids.len(), n),
        });
    }

    // Identify unique time periods and their indices
    let mut time_map: std::collections::BTreeMap<i64, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &t) in time_ids.iter().enumerate() {
        time_map.entry(t).or_default().push(i);
    }

    let time_periods: Vec<i64> = time_map.keys().copied().collect();
    let n_periods = time_periods.len();
    let n_units = n / n_periods.max(1); // Average units per period

    if n_periods < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n_periods,
            context: "Driscoll-Kraay requires at least 3 time periods".to_string(),
        });
    }

    // Compute (X'X)^{-1}
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Driscoll-Kraay covariance".to_string(),
        suggestion: format!("Error: {}", e),
    })?;

    // Step 1: Aggregate score vectors by time period
    // h̄_t = (1/N_t) Σᵢ∈t uᵢ xᵢ
    let mut h_bar: Vec<Array1<f64>> = Vec::with_capacity(n_periods);

    for t in &time_periods {
        let indices = &time_map[t];
        let n_t = indices.len();

        let mut h_t = Array1::<f64>::zeros(k);
        for &i in indices {
            for j in 0..k {
                h_t[j] += residuals[i] * x[[i, j]];
            }
        }
        // Average within time period
        h_t /= n_t as f64;
        h_bar.push(h_t);
    }

    // Step 2: Automatic bandwidth selection
    let bw = bandwidth.unwrap_or_else(|| {
        // Newey-West 1987 rule: floor(T^(1/4))
        let nw_bw = (n_periods as f64).powf(0.25).floor() as usize;
        nw_bw.max(1).min(n_periods - 1)
    });

    // Step 3: Compute HAC-adjusted covariance of aggregated moments
    // Ω̂ = Γ̂₀ + Σⱼ₌₁ᵐ w(j)(Γ̂ⱼ + Γ̂ⱼ')
    let mut omega = Array2::<f64>::zeros((k, k));

    // Lag 0: Γ̂₀ = (1/T) Σₜ h̄_t h̄_t'
    for t in 0..n_periods {
        for j in 0..k {
            for l in 0..k {
                omega[[j, l]] += h_bar[t][j] * h_bar[t][l];
            }
        }
    }

    // Lags 1 to bw
    for lag in 1..=bw {
        let w = kernel.weight(lag, bw);
        if w.abs() < 1e-15 {
            continue;
        }

        let mut gamma_lag = Array2::<f64>::zeros((k, k));
        for t in lag..n_periods {
            for j in 0..k {
                for l in 0..k {
                    gamma_lag[[j, l]] += h_bar[t][j] * h_bar[t - lag][l];
                }
            }
        }

        // Add symmetrically
        for j in 0..k {
            for l in 0..k {
                omega[[j, l]] += w * (gamma_lag[[j, l]] + gamma_lag[[l, j]]);
            }
        }
    }

    // Scale by T (number of periods) to get consistent estimator
    // The aggregation by N already accounted for cross-sectional dimension
    // Final formula: V = (X'X)⁻¹ × (T × Ω̂) × (X'X)⁻¹
    let scaled_omega = &omega * (n_periods as f64);

    // Sandwich: V = (X'X)⁻¹ Ω (X'X)⁻¹
    let meat_bread = matmul(&scaled_omega.view(), &xtx_inv.view()).map_err(|e| EconError::SingularMatrix {
        context: "Driscoll-Kraay sandwich".to_string(),
        suggestion: format!("Error: {}", e),
    })?;
    let vcov = matmul(&xtx_inv.view(), &meat_bread.view()).map_err(|e| EconError::SingularMatrix {
        context: "Driscoll-Kraay sandwich".to_string(),
        suggestion: format!("Error: {}", e),
    })?;

    // Extract standard errors and compute statistics
    let std_errors: Vec<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();

    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result.coefficients.iter().map(|c| c.name.clone()).collect();

    let df = (n_periods - k) as f64;
    let t_stats: Vec<f64> = coefficients.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| crate::traits::estimator::t_test_p_value(t, df))
        .collect();

    // Convert vcov to Vec<Vec<f64>>
    let vcov_vec: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov[[i, j]]).collect())
        .collect();

    Ok(DriscollKraayResult {
        vcov: vcov_vec,
        std_errors,
        coefficients,
        t_stats,
        p_values,
        names,
        n_obs: n,
        n_params: k,
        n_periods,
        n_units,
        bandwidth: bw,
        kernel,
    })
}

/// Convenience function to compute Driscoll-Kraay standard errors for a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `time_col` - Name of the time period identifier column
/// * `bandwidth` - Optional bandwidth (None for automatic)
/// * `kernel` - Kernel type string ("bartlett", "parzen", "qs", etc.)
///
/// # Returns
///
/// `DriscollKraayResult` with panel-robust standard errors.
///
/// # R Equivalent
///
/// ```r
/// library(plm)
/// library(sandwich)
/// model <- plm(y ~ x, data = panel_data, model = "pooling")
/// vcovPL(model, cluster = ~ firm + year)
/// ```
pub fn run_vcov_driscoll_kraay(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    time_col: &str,
    bandwidth: Option<usize>,
    kernel: Option<&str>,
) -> EconResult<DriscollKraayResult> {
    // Run OLS
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = dm.view().to_owned();

    // Extract time IDs
    let time_series = dataset.df()
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![]
        })?;

    let time_ids: Vec<i64> = if time_series.dtype().is_integer() {
        time_series.cast(&DataType::Int64)
            .map_err(|_| EconError::NonNumericColumn { column: time_col.to_string() })?
            .i64()
            .map_err(|_| EconError::NonNumericColumn { column: time_col.to_string() })?
            .into_no_null_iter()
            .collect()
    } else if time_series.dtype().is_float() {
        time_series.f64()
            .map_err(|_| EconError::NonNumericColumn { column: time_col.to_string() })?
            .into_no_null_iter()
            .map(|v| v as i64)
            .collect()
    } else {
        return Err(EconError::NonNumericColumn { column: time_col.to_string() });
    };

    // Parse kernel
    let kernel_type = kernel
        .and_then(HacKernel::from_str)
        .unwrap_or(HacKernel::Bartlett);

    vcov_driscoll_kraay(&ols_result, &x, &time_ids, bandwidth, kernel_type)
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

    // ========================================
    // HAC (Newey-West) Tests
    // ========================================

    fn create_timeseries_dataset() -> Dataset {
        // Create time series data with autocorrelated errors
        // y = 1 + 0.5*x + AR(1) errors
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        // Simulate AR(1) errors with ρ ≈ 0.5
        let y: Vec<f64> = vec![
            1.5, 2.1, 2.9, 3.3, 4.2, 4.8, 5.5, 5.9, 6.6, 7.1,
            7.8, 8.3, 8.9, 9.4, 10.0, 10.5, 11.1, 11.6, 12.2, 12.7,
            13.3, 13.8, 14.4, 14.9, 15.5, 16.0, 16.6, 17.1, 17.7, 18.2,
        ];

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_hac_basic() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], None, None, false).unwrap();

        assert_eq!(result.n_obs, 30);
        assert_eq!(result.n_params, 2);
        assert!(result.std_errors[0] > 0.0);
        assert!(result.std_errors[1] > 0.0);
        assert!(matches!(result.kernel, HacKernel::Bartlett));
    }

    #[test]
    fn test_hac_with_bandwidth() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], Some(5), None, false).unwrap();

        assert_eq!(result.bandwidth, 5);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
    }

    #[test]
    fn test_hac_parzen_kernel() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], None, Some("parzen"), false).unwrap();

        assert!(matches!(result.kernel, HacKernel::Parzen));
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
    }

    #[test]
    fn test_hac_with_prewhitening() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], None, None, true).unwrap();

        assert!(result.prewhiten);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
    }

    #[test]
    fn test_hac_kernel_parsing() {
        assert!(matches!(HacKernel::from_str("bartlett"), Some(HacKernel::Bartlett)));
        assert!(matches!(HacKernel::from_str("newey-west"), Some(HacKernel::Bartlett)));
        assert!(matches!(HacKernel::from_str("parzen"), Some(HacKernel::Parzen)));
        assert!(matches!(HacKernel::from_str("qs"), Some(HacKernel::QuadraticSpectral)));
        assert!(matches!(HacKernel::from_str("truncated"), Some(HacKernel::Truncated)));
        assert!(matches!(HacKernel::from_str("tukey-hanning"), Some(HacKernel::TukeyHanning)));
        assert!(HacKernel::from_str("invalid").is_none());
    }

    #[test]
    fn test_hac_vcov_positive_semidefinite() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], Some(3), None, false).unwrap();

        // All diagonal elements should be positive (variances)
        for i in 0..result.n_params {
            assert!(result.vcov[i][i] >= 0.0, "Diagonal element {} should be non-negative", i);
        }

        // Standard errors should be consistent with vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!((result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal", i);
        }
    }

    #[test]
    fn test_hac_displays_correctly() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], None, None, false).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("HAC (Newey-West)"));
        assert!(display.contains("Bartlett"));
        assert!(display.contains("Bandwidth"));
    }

    // ========================================
    // Bootstrap Covariance Tests
    // ========================================

    #[test]
    fn test_bootstrap_pairs() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("pairs"), Some(42)).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_params, 2);
        assert!(result.std_errors[0] > 0.0, "Intercept SE should be positive");
        assert!(result.std_errors[1] > 0.0, "Slope SE should be positive");
        assert!(result.convergence_rate > 0.9, "Most replications should converge");
        assert!(matches!(result.bootstrap_type, BootstrapType::Pairs));
    }

    #[test]
    fn test_bootstrap_residual() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("residual"), Some(42)).unwrap();

        assert_eq!(result.n_params, 2);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
        assert!(matches!(result.bootstrap_type, BootstrapType::Residual));
    }

    #[test]
    fn test_bootstrap_wild() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("wild"), Some(42)).unwrap();

        assert_eq!(result.n_params, 2);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
        assert!(matches!(result.bootstrap_type, BootstrapType::Wild));
    }

    #[test]
    fn test_bootstrap_vcov_positive_semidefinite() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("pairs"), Some(42)).unwrap();

        // All diagonal elements should be positive (variances)
        for i in 0..result.n_params {
            assert!(result.vcov[i][i] >= 0.0, "Diagonal element {} should be non-negative", i);
        }

        // Standard errors should be consistent with vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!((result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal", i);
        }
    }

    #[test]
    fn test_bootstrap_type_parsing() {
        assert!(matches!(BootstrapType::from_str("pairs"), Some(BootstrapType::Pairs)));
        assert!(matches!(BootstrapType::from_str("xy"), Some(BootstrapType::Pairs)));
        assert!(matches!(BootstrapType::from_str("residual"), Some(BootstrapType::Residual)));
        assert!(matches!(BootstrapType::from_str("wild"), Some(BootstrapType::Wild)));
        assert!(BootstrapType::from_str("invalid").is_none());
    }

    #[test]
    fn test_bootstrap_displays_correctly() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(&dataset, "y", &["x1"], Some(100), Some("pairs"), Some(42)).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Bootstrap Covariance"));
        assert!(display.contains("Pairs"));
        assert!(display.contains("Replications"));
    }

    // ========================================
    // Driscoll-Kraay Panel-Robust Tests
    // ========================================

    fn create_panel_dataset() -> Dataset {
        // Simple panel: 5 units, 10 time periods
        // y = 1 + 0.5*x + unit_fe + time_fe + error
        let mut y_vec = Vec::new();
        let mut x_vec = Vec::new();
        let mut unit_vec = Vec::new();
        let mut time_vec = Vec::new();

        for unit in 1..=5 {
            for time in 1..=10 {
                let x = (unit + time) as f64;
                let y = 1.0 + 0.5 * x + 0.1 * unit as f64 + 0.05 * time as f64
                    + (((unit * time) % 7) as f64 - 3.0) * 0.1;
                y_vec.push(y);
                x_vec.push(x);
                unit_vec.push(unit as i64);
                time_vec.push(time as i64);
            }
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec,
            "unit" => unit_vec,
            "time" => time_vec
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_driscoll_kraay_basic() {
        let dataset = create_panel_dataset();
        let result = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, None).unwrap();

        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_params, 2);
        assert_eq!(result.n_periods, 10);
        assert!(result.std_errors[0] > 0.0, "Intercept SE should be positive");
        assert!(result.std_errors[1] > 0.0, "Slope SE should be positive");
    }

    #[test]
    fn test_driscoll_kraay_with_bandwidth() {
        let dataset = create_panel_dataset();
        let result = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", Some(3), None).unwrap();

        assert_eq!(result.bandwidth, 3);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
    }

    #[test]
    fn test_driscoll_kraay_different_kernels() {
        let dataset = create_panel_dataset();

        // Test different kernels
        let bartlett = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, Some("bartlett")).unwrap();
        let parzen = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, Some("parzen")).unwrap();

        // Both should produce valid results
        assert!(bartlett.std_errors.iter().all(|&se| se > 0.0));
        assert!(parzen.std_errors.iter().all(|&se| se > 0.0));

        // SEs may differ due to different kernel shapes
        assert!(matches!(bartlett.kernel, HacKernel::Bartlett));
        assert!(matches!(parzen.kernel, HacKernel::Parzen));
    }

    #[test]
    fn test_driscoll_kraay_vcov_positive_semidefinite() {
        let dataset = create_panel_dataset();
        let result = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, None).unwrap();

        // All diagonal elements should be positive (variances)
        for i in 0..result.n_params {
            assert!(result.vcov[i][i] >= 0.0, "Variance {} should be non-negative", i);
        }

        // Standard errors should match sqrt of vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!((result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal", i);
        }
    }

    #[test]
    fn test_driscoll_kraay_displays_correctly() {
        let dataset = create_panel_dataset();
        let result = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, None).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Driscoll-Kraay"));
        assert!(display.contains("Panel-Robust"));
        assert!(display.contains("T = 10"));
    }
}
