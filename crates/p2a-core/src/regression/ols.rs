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
    CONDITION_THRESHOLD, DesignError, DesignMatrix, matmul, safe_inverse, xtx, xty,
};
use crate::resource;
use crate::traits::{LinearEstimator, SignificanceLevel, f_test_p_value, t_test_p_value};

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

impl CovarianceType {
    /// Short label for output headers: `"classical"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`.
    ///
    /// Every label parses back to its variant (see the [`FromStr`](std::str::FromStr) impl).
    ///
    /// ```
    /// use p2a_core::regression::CovarianceType;
    /// let t: CovarianceType = "HC1".parse().unwrap();
    /// assert_eq!(t.short_label(), "HC1");
    /// assert_eq!(CovarianceType::Standard.short_label(), "classical");
    /// ```
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::Standard => "classical",
            Self::HC0 => "HC0",
            Self::HC1 => "HC1",
            Self::HC2 => "HC2",
            Self::HC3 => "HC3",
        }
    }
}

impl std::str::FromStr for CovarianceType {
    type Err = EconError;

    /// Parse a covariance type, ignoring case and surrounding whitespace:
    /// `standard` or `classical` give [`CovarianceType::Standard`]; `hc0`, `hc1`, `hc2`
    /// and `hc3` the HC variants. Anything else is an [`EconError::InvalidSpecification`]
    /// whose message quotes the input (its first 64 characters) and lists the accepted
    /// values.
    fn from_str(s: &str) -> Result<CovarianceType, EconError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "standard" | "classical" => Ok(Self::Standard),
            "hc0" => Ok(Self::HC0),
            "hc1" => Ok(Self::HC1),
            "hc2" => Ok(Self::HC2),
            "hc3" => Ok(Self::HC3),
            _ => {
                const MAX_CHARS: usize = 64;
                let mut quoted: String = s.chars().take(MAX_CHARS).collect();
                if s.chars().nth(MAX_CHARS).is_some() {
                    quoted.push_str("...");
                }
                Err(EconError::InvalidSpecification {
                    message: format!(
                        "unknown covariance type '{quoted}'; use 'standard' (or 'classical'), \
                         'hc0', 'hc1', 'hc2' or 'hc3'"
                    ),
                })
            }
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
    /// Gaussian log-likelihood at σ̂² = SSR/n (R `logLik`)
    pub log_likelihood: f64,
    /// Akaike Information Criterion, 2(k + 1) − 2 logLik (R `AIC`)
    pub aic: f64,
    /// Bayesian Information Criterion, (k + 1) ln n − 2 logLik (R `BIC`)
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
    /// Sum of squared residuals (cached; used by R², F-test, diagnostics).
    #[serde(skip)]
    pub ssr: f64,
    /// Total sum of squares (cached; used by R², F-test, diagnostics).
    #[serde(skip)]
    pub sst: f64,
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

    /// The per-coefficient p-values computed at fit time.
    fn p_values(&self) -> Array1<f64> {
        self.coefficients.iter().map(|c| c.p_value).collect()
    }

    fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }

    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn aic(&self) -> f64 {
        self.aic
    }

    fn bic(&self) -> f64 {
        self.bic
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
        writeln!(
            f,
            "F-statistic: {:.4} (p = {:.4})",
            self.f_statistic, self.f_p_value
        )?;
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

impl OlsResult {
    /// Export coefficient table to a CSV string.
    ///
    /// Produces a table with columns: variable, coefficient, std_error, t_stat, p_value,
    /// significance, ci_lower_95, ci_upper_95
    pub fn to_csv_string(&self) -> String {
        let mut csv = String::new();
        csv.push_str(
            "variable,coefficient,std_error,t_stat,p_value,significance,ci_lower_95,ci_upper_95\n",
        );

        for coef in &self.coefficients {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.4},{:.6},{},{:.6},{:.6}\n",
                coef.name,
                coef.estimate,
                coef.std_error,
                coef.t_value,
                coef.p_value,
                coef.significance.stars().replace(" ", ""),
                coef.ci_lower_95,
                coef.ci_upper_95
            ));
        }

        csv
    }

    /// Export coefficient table to a CSV file.
    pub fn to_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_csv_string())
    }

    /// Export results to a JSON string.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
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

/// Gaussian log-likelihood, AIC and BIC of a least-squares fit, as R's
/// `logLik.lm`, `AIC` and `BIC` report them for `lm`.
///
/// The log-likelihood is evaluated at the ML estimate of the error variance,
/// σ̂² = SSR/n: `ll = -n/2 (1 + ln(2π SSR/n))`. The error variance counts as a
/// parameter, so with `k` coefficients `AIC = 2(k + 1) - 2 ll` and
/// `BIC = (k + 1) ln n - 2 ll`. A perfect fit (SSR = 0) gives `ll = +∞`, as in R.
///
/// Returns `(log_likelihood, aic, bic)`.
fn gaussian_fit_criteria(ssr: f64, n: usize, k: usize) -> (f64, f64, f64) {
    let n = n as f64;
    let p = k as f64 + 1.0;
    let ll = -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI * ssr / n).ln());
    (ll, 2.0 * p - 2.0 * ll, p * n.ln() - 2.0 * ll)
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
            message: format!("X has {} rows but y has {} elements", n, y.len()),
        });
    }

    if variable_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "X has {} columns but {} variable names provided",
                k,
                variable_names.len()
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
    let (xtx_inv, cond_warning) =
        safe_inverse(&xtx(&x.view()).view()).map_err(|_e| EconError::SingularMatrix {
            context: "X'X matrix in OLS".to_string(),
            suggestion: "Check for perfect multicollinearity between independent variables"
                .to_string(),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(
            EstimationWarning::HighConditionNumber {
                value: cond,
                threshold: CONDITION_THRESHOLD,
            }
            .message(),
        );
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

    let (log_likelihood, aic, bic) = gaussian_fit_criteria(ssr, n, k);

    // Compute variance-covariance matrix based on cov_type
    let vcov = compute_vcov(&x.view(), &residuals, &xtx_inv, cov_type, n, k)?;

    // Extract standard errors from diagonal
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // Compute t-statistics and p-values
    let t_stats: Array1<f64> = &beta / &se;
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();

    // Compute 95% confidence intervals
    let t_crit = crate::traits::t_critical(0.05, df_resid as f64);

    // Build coefficient results
    let coefficients: Vec<OlsCoefficient> = variable_names
        .iter()
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
    let design = DesignMatrix::from_dataframe(df, x_cols, intercept).map_err(|e| match e {
        DesignError::ColumnNotFound(c) => EconError::ColumnNotFound {
            column: c,
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        },
        DesignError::NonNumericColumn(c) => EconError::NonNumericColumn { column: c },
        DesignError::NullValues(c, indices) => EconError::NullValues {
            column: c,
            count: indices.len(),
        },
        DesignError::EmptyDataset => EconError::EmptyDataset,
        DesignError::NoColumns => EconError::InvalidSpecification {
            message: "No independent variables specified".to_string(),
        },
        DesignError::PolarsError(e) => EconError::Internal(e.to_string()),
    })?;

    // Extract y vector
    let y = DesignMatrix::extract_column(df, y_col).map_err(|e| match e {
        DesignError::ColumnNotFound(c) => EconError::ColumnNotFound {
            column: c,
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
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
    let (xtx_inv, cond_warning) =
        safe_inverse(&xtx(&x.view()).view()).map_err(|_e| EconError::SingularMatrix {
            context: "X'X matrix in OLS".to_string(),
            suggestion: "Check for perfect multicollinearity between independent variables"
                .to_string(),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(
            EstimationWarning::HighConditionNumber {
                value: cond,
                threshold: CONDITION_THRESHOLD,
            }
            .message(),
        );
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

    let (log_likelihood, aic, bic) = gaussian_fit_criteria(ssr, n, k);

    // Compute variance-covariance matrix based on cov_type
    let vcov = compute_vcov(&x.view(), &residuals, &xtx_inv, cov_type, n, k)?;

    // Extract standard errors from diagonal
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // Compute t-statistics and p-values
    let t_stats: Array1<f64> = &beta / &se;
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();

    // Compute 95% confidence intervals
    let t_crit = crate::traits::t_critical(0.05, df_resid as f64);

    // Build coefficient results
    let coefficients: Vec<OlsCoefficient> = design
        .column_names
        .iter()
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
    let temp =
        matmul(&xtx_inv.view(), &meat.view()).map_err(|e| EconError::Internal(e.to_string()))?;
    let vcov =
        matmul(&temp.view(), &xtx_inv.view()).map_err(|e| EconError::Internal(e.to_string()))?;

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
///
/// When the `cuda` feature is enabled and a GPU is available, dispatches to
/// GPU for large matrices (n >= xtx threshold).
fn compute_sandwich_meat(x: &ndarray::ArrayView2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();

    // GPU dispatch: sandwich meat = (W^{1/2} X)' (W^{1/2} X) via cuBLAS DGEMM
    #[cfg(feature = "cuda")]
    if let Some(ctx) = crate::linalg::gpu::GpuContext::get() {
        if k >= ctx.thresholds.xtx_min_k && n * k * k >= ctx.thresholds.xtx_min_nkk {
            match crate::linalg::gpu::sandwich_meat_gpu(ctx, x, weights) {
                Ok(result) => return result,
                Err(e) => {
                    tracing::warn!("GPU sandwich_meat failed, falling back to CPU: {:?}", e);
                }
            }
        }
    }

    // CPU path: parallel accumulation
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
        writeln!(
            f,
            "F-statistic: {:.4} (p = {:.4})",
            self.ols.f_statistic, self.ols.f_p_value
        )?;
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

    // Add graduated warning for few clusters (Cameron-Miller 2015 guidance)
    // - G < 20: Severe finite-sample bias concerns
    // - 20 <= G < 50: Moderate concerns, consider wild bootstrap
    // - G >= 50: Generally adequate for asymptotic inference
    if n_clusters_1 < 50 {
        let severity = if n_clusters_1 < 20 {
            "severe"
        } else {
            "moderate"
        };
        base_result.warnings.push(
            EstimationWarning::FewClusters {
                n_clusters: n_clusters_1,
                severity: severity.to_string(),
            }
            .message(),
        );
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

        (
            vcov,
            Some(n_c2),
            format!("Two-way clustered ({}, {})", cluster1, c2),
        )
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

    let coefficients: Vec<OlsCoefficient> = design
        .column_names
        .iter()
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
fn extract_cluster_ids(df: &DataFrame, col: &str) -> EconResult<HashMap<String, Vec<usize>>> {
    let column = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
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
    let temp =
        matmul(&xtx_inv.view(), &meat.view()).map_err(|e| EconError::Internal(e.to_string()))?;
    let vcov =
        matmul(&temp.view(), &xtx_inv.view()).map_err(|e| EconError::Internal(e.to_string()))?;

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
    /// Warnings about the estimate
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
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
            writeln!(
                f,
                "{:<15} {:>12.6} {:>12.6}",
                name, self.coefficients[i], self.std_errors[i]
            )?;
        }
        write_warnings(f, &self.warnings)
    }
}

/// Append a `Warnings:` block to a result's `Display` output (nothing when empty).
fn write_warnings(f: &mut std::fmt::Formatter<'_>, warnings: &[String]) -> std::fmt::Result {
    if warnings.is_empty() {
        return Ok(());
    }
    writeln!(f)?;
    writeln!(f, "Warnings:")?;
    for w in warnings {
        writeln!(f, "  - {w}")?;
    }
    Ok(())
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
                if j <= m {
                    1.0 - x
                } else {
                    0.0
                }
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
                if j <= m {
                    1.0
                } else {
                    0.0
                }
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
///   A user bandwidth must be smaller than n (otherwise `InvalidSpecification`); one above n/4
///   adds an entry to `HacResult::warnings`.
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
            message: format!(
                "Design matrix rows ({}) does not match number of observations ({})",
                x.nrows(),
                n
            ),
        });
    }
    if x.ncols() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Design matrix columns ({}) does not match number of parameters ({})",
                x.ncols(),
                k
            ),
        });
    }

    // A user bandwidth must leave data at every lag. sandwich::meatHAC warns and
    // truncates when lag >= n; p2a refuses that case and additionally warns when
    // a user bandwidth exceeds n/4 (REG-1).
    let mut warnings = Vec::new();
    if let Some(bw) = bandwidth {
        if bw >= n {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "HAC bandwidth {bw} must be smaller than the number of observations ({n}): \
                     lags of n or more have no data, and bandwidths close to n shrink the \
                     standard errors towards zero"
                ),
            });
        }
        if bw.saturating_mul(4) > n {
            warnings.push(format!(
                "Bandwidth {bw} exceeds a quarter of the {n} observations; HAC standard errors \
                 are biased towards zero when the bandwidth is a large fraction of the sample"
            ));
        }
    }

    // Automatic bandwidth selection (Newey-West rule of thumb)
    let bw = bandwidth.unwrap_or_else(|| {
        // NW rule: floor(4 * (n/100)^(2/9))
        let nw_bw = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
        nw_bw.max(1).min(n - 1)
    });

    // Compute X'X and its inverse
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _cond) =
        safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
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
    for lag in 1..=bw.min(n.saturating_sub(1)) {
        resource::checkpoint()?;
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
    let meat_bread =
        matmul(&omega.view(), &xtx_inv.view()).map_err(|e| EconError::SingularMatrix {
            context: "HAC sandwich computation".to_string(),
            suggestion: format!("Matrix multiplication error: {}", e),
        })?;
    let vcov_hac =
        matmul(&xtx_inv.view(), &meat_bread.view()).map_err(|e| EconError::SingularMatrix {
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
    let names: Vec<String> = ols_result
        .coefficients
        .iter()
        .map(|c| c.name.clone())
        .collect();

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
        warnings,
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
/// * `bandwidth` - Optional bandwidth (None for automatic). Must be smaller than n; one above n/4
///   adds a warning (see [`vcov_hac`]).
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
            writeln!(
                f,
                "  {:15} {:12.6} (SE: {:8.6})",
                name, self.coefficients[i], self.std_errors[i]
            )?;
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
/// * `n_boot` - Number of bootstrap replications (default: 999). At least 10, otherwise
///   `InvalidSpecification`; replicate storage above the allocation ceiling is `ResourceLimit`.
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

    if replications < 10 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n_boot must be at least 10 (got {replications}): the bootstrap covariance \
                 needs at least 10 successful replications"
            ),
        });
    }

    if n < k {
        return Err(EconError::InsufficientData {
            required: k,
            provided: n,
            context: "Bootstrap covariance requires n >= k".to_string(),
        });
    }

    // Replicate storage: k coefficients plus a `Vec` header (3 words) each
    let what = "bootstrap replicate storage";
    resource::check_alloc::<f64>(
        resource::checked_len(&[replications, k.saturating_add(3)], what)?,
        what,
    )?;

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Original fitted values and residuals
    let y_fitted = x.dot(&ols_result.coefficients().clone());
    let residuals = ols_result.residuals();

    // Storage for bootstrap coefficient estimates
    let mut boot_coefs: Vec<Vec<f64>> = resource::try_with_capacity(replications, what)?;
    let indices: Vec<usize> = (0..n).collect();

    for _ in 0..replications {
        resource::checkpoint()?;
        let (y_boot, x_boot) = match bootstrap_type {
            BootstrapType::Pairs => {
                // Pairs bootstrap: resample (y_i, x_i) pairs together
                let sample: Vec<usize> =
                    (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();

                let y_boot: Array1<f64> = sample.iter().map(|&i| y[i]).collect();
                let x_boot: Array2<f64> = Array2::from_shape_fn((n, k), |(i, j)| x[[sample[i], j]]);
                (y_boot, x_boot)
            }
            BootstrapType::Residual => {
                // Residual bootstrap: y* = X*β̂ + ε*, where ε* resampled from residuals
                let resid_sample: Vec<f64> = (0..n)
                    .map(|_| residuals[*indices.choose(&mut rng).unwrap()])
                    .collect();

                let y_boot: Array1<f64> = (0..n).map(|i| y_fitted[i] + resid_sample[i]).collect();

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
    let std_errors: Vec<f64> = (0..k).map(|i| vcov[i][i].max(0.0).sqrt()).collect();

    // Extract original coefficients and names
    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result
        .coefficients
        .iter()
        .map(|c| c.name.clone())
        .collect();

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
/// * `n_boot` - Number of bootstrap replications (default: 999). At least 10 (see
///   [`vcov_bootstrap`]).
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
    let y_series = dataset
        .df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![],
        })?;
    let y: Array1<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
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
    /// Warnings about the estimate
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl std::fmt::Display for DriscollKraayResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Driscoll-Kraay Panel-Robust Standard Errors")?;
        writeln!(f, "============================================")?;
        writeln!(
            f,
            "N = {} observations, T = {} periods, N_units = {}",
            self.n_obs, self.n_periods, self.n_units
        )?;
        writeln!(f, "Bandwidth: {}, Kernel: {}", self.bandwidth, self.kernel)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std.Err", "t", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;
        for (i, name) in self.names.iter().enumerate() {
            let sig = crate::traits::estimator::SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                name,
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        writeln!(f)?;
        writeln!(
            f,
            "Standard errors robust to cross-sectional and serial correlation."
        )?;
        write_warnings(f, &self.warnings)
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
/// * `bandwidth` - Optional bandwidth (None for automatic). A user bandwidth must be smaller than
///   the number of time periods T (otherwise `InvalidSpecification`); one above T/4 adds an
///   entry to `DriscollKraayResult::warnings`.
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
            message: format!(
                "Time IDs length ({}) != observations ({})",
                time_ids.len(),
                n
            ),
        });
    }

    // Identify unique time periods and their indices
    let mut time_map: std::collections::BTreeMap<i64, Vec<usize>> =
        std::collections::BTreeMap::new();
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

    // A user bandwidth must leave data at every lag of the period series.
    // sandwich::meatHAC warns and truncates when lag >= n (sandwich::meatPL, its
    // Driscoll-Kraay meat, does neither); p2a refuses lag >= T here and
    // additionally warns when a user bandwidth exceeds T/4 (REG-1).
    let mut warnings = Vec::new();
    if let Some(bw) = bandwidth {
        if bw >= n_periods {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Driscoll-Kraay bandwidth {bw} must be smaller than the number of time \
                     periods ({n_periods}): lags of T or more have no data, and bandwidths \
                     close to T shrink the standard errors towards zero"
                ),
            });
        }
        if bw.saturating_mul(4) > n_periods {
            warnings.push(format!(
                "Bandwidth {bw} exceeds a quarter of the {n_periods} time periods; \
                 Driscoll-Kraay standard errors are biased towards zero when the bandwidth \
                 is a large fraction of the sample"
            ));
        }
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
    for lag in 1..=bw.min(n_periods - 1) {
        resource::checkpoint()?;
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
    let meat_bread =
        matmul(&scaled_omega.view(), &xtx_inv.view()).map_err(|e| EconError::SingularMatrix {
            context: "Driscoll-Kraay sandwich".to_string(),
            suggestion: format!("Error: {}", e),
        })?;
    let vcov =
        matmul(&xtx_inv.view(), &meat_bread.view()).map_err(|e| EconError::SingularMatrix {
            context: "Driscoll-Kraay sandwich".to_string(),
            suggestion: format!("Error: {}", e),
        })?;

    // Extract standard errors and compute statistics
    let std_errors: Vec<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();

    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result
        .coefficients
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let df = (n_periods - k) as f64;
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
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
        warnings,
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
/// * `bandwidth` - Optional bandwidth (None for automatic). Must be smaller than the number of
///   time periods T; one above T/4 adds a warning (see [`vcov_driscoll_kraay`]).
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
    let time_series = dataset
        .df()
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![],
        })?;

    let time_ids: Vec<i64> = if time_series.dtype().is_integer() {
        time_series
            .cast(&DataType::Int64)
            .map_err(|_| EconError::NonNumericColumn {
                column: time_col.to_string(),
            })?
            .i64()
            .map_err(|_| EconError::NonNumericColumn {
                column: time_col.to_string(),
            })?
            .into_no_null_iter()
            .collect()
    } else if time_series.dtype().is_float() {
        time_series
            .f64()
            .map_err(|_| EconError::NonNumericColumn {
                column: time_col.to_string(),
            })?
            .into_no_null_iter()
            .map(|v| v as i64)
            .collect()
    } else {
        return Err(EconError::NonNumericColumn {
            column: time_col.to_string(),
        });
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
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};

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
        let result = run_ols(
            &dataset,
            "nonexistent",
            &["x1"],
            true,
            CovarianceType::Standard,
        );

        assert!(matches!(result, Err(EconError::ColumnNotFound { .. })));
    }

    #[test]
    fn test_covariance_type_from_str_and_short_label() {
        use CovarianceType::*;
        for (s, t) in [
            ("standard", Standard),
            ("classical", Standard),
            ("Classical", Standard),
            (" HC1 ", HC1),
            ("hc0", HC0),
            ("Hc2", HC2),
            ("HC3", HC3),
        ] {
            assert_eq!(s.parse::<CovarianceType>().unwrap(), t, "{s:?}");
        }
        let all = [Standard, HC0, HC1, HC2, HC3];
        let labels: Vec<&str> = all.iter().map(|t| t.short_label()).collect();
        assert_eq!(labels, ["classical", "HC0", "HC1", "HC2", "HC3"]);
        for t in all {
            assert_eq!(t.short_label().parse::<CovarianceType>().unwrap(), t);
        }
    }

    #[test]
    fn test_covariance_type_from_str_rejects_unknown() {
        for s in ["", "hc4", "robust", "HC1 robust", "hç1"] {
            let e = s.parse::<CovarianceType>();
            assert!(
                matches!(&e, Err(EconError::InvalidSpecification { message })
                    if message.contains("hc0")
                        && message.contains("classical")
                        && message.contains(&format!("'{s}'"))),
                "{s:?}: {e:?}"
            );
        }
        // 200 characters whose byte 64 is not a char boundary: quoted char-based, capped.
        let long = format!("a{}", "ç".repeat(199));
        match long.parse::<CovarianceType>() {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.len() < 300, "{} bytes: {message}", message.len());
                assert!(
                    message.contains(&format!("'a{}", "ç".repeat(63))),
                    "{message}"
                );
                assert!(!message.contains(&"ç".repeat(64)), "{message}");
            }
            other => panic!("{other:?}"),
        }
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
            1.5, 2.1, 2.9, 3.3, 4.2, 4.8, 5.5, 5.9, 6.6, 7.1, 7.8, 8.3, 8.9, 9.4, 10.0, 10.5, 11.1,
            11.6, 12.2, 12.7, 13.3, 13.8, 14.4, 14.9, 15.5, 16.0, 16.6, 17.1, 17.7, 18.2,
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
        assert!(matches!(
            HacKernel::from_str("bartlett"),
            Some(HacKernel::Bartlett)
        ));
        assert!(matches!(
            HacKernel::from_str("newey-west"),
            Some(HacKernel::Bartlett)
        ));
        assert!(matches!(
            HacKernel::from_str("parzen"),
            Some(HacKernel::Parzen)
        ));
        assert!(matches!(
            HacKernel::from_str("qs"),
            Some(HacKernel::QuadraticSpectral)
        ));
        assert!(matches!(
            HacKernel::from_str("truncated"),
            Some(HacKernel::Truncated)
        ));
        assert!(matches!(
            HacKernel::from_str("tukey-hanning"),
            Some(HacKernel::TukeyHanning)
        ));
        assert!(HacKernel::from_str("invalid").is_none());
    }

    #[test]
    fn test_hac_vcov_positive_semidefinite() {
        let dataset = create_timeseries_dataset();
        let result = run_vcov_hac(&dataset, "y", &["x"], Some(3), None, false).unwrap();

        // All diagonal elements should be positive (variances)
        for i in 0..result.n_params {
            assert!(
                result.vcov[i][i] >= 0.0,
                "Diagonal element {} should be non-negative",
                i
            );
        }

        // Standard errors should be consistent with vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!(
                (result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal",
                i
            );
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
        let result =
            run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("pairs"), Some(42)).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_params, 2);
        assert!(
            result.std_errors[0] > 0.0,
            "Intercept SE should be positive"
        );
        assert!(result.std_errors[1] > 0.0, "Slope SE should be positive");
        assert!(
            result.convergence_rate > 0.9,
            "Most replications should converge"
        );
        assert!(matches!(result.bootstrap_type, BootstrapType::Pairs));
    }

    #[test]
    fn test_bootstrap_residual() {
        let dataset = create_test_dataset();
        let result = run_vcov_bootstrap(
            &dataset,
            "y",
            &["x1"],
            Some(200),
            Some("residual"),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.n_params, 2);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
        assert!(matches!(result.bootstrap_type, BootstrapType::Residual));
    }

    #[test]
    fn test_bootstrap_wild() {
        let dataset = create_test_dataset();
        let result =
            run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("wild"), Some(42)).unwrap();

        assert_eq!(result.n_params, 2);
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
        assert!(matches!(result.bootstrap_type, BootstrapType::Wild));
    }

    #[test]
    fn test_bootstrap_vcov_positive_semidefinite() {
        let dataset = create_test_dataset();
        let result =
            run_vcov_bootstrap(&dataset, "y", &["x1"], Some(200), Some("pairs"), Some(42)).unwrap();

        // All diagonal elements should be positive (variances)
        for i in 0..result.n_params {
            assert!(
                result.vcov[i][i] >= 0.0,
                "Diagonal element {} should be non-negative",
                i
            );
        }

        // Standard errors should be consistent with vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!(
                (result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal",
                i
            );
        }
    }

    #[test]
    fn test_bootstrap_type_parsing() {
        assert!(matches!(
            BootstrapType::from_str("pairs"),
            Some(BootstrapType::Pairs)
        ));
        assert!(matches!(
            BootstrapType::from_str("xy"),
            Some(BootstrapType::Pairs)
        ));
        assert!(matches!(
            BootstrapType::from_str("residual"),
            Some(BootstrapType::Residual)
        ));
        assert!(matches!(
            BootstrapType::from_str("wild"),
            Some(BootstrapType::Wild)
        ));
        assert!(BootstrapType::from_str("invalid").is_none());
    }

    #[test]
    fn test_bootstrap_displays_correctly() {
        let dataset = create_test_dataset();
        let result =
            run_vcov_bootstrap(&dataset, "y", &["x1"], Some(100), Some("pairs"), Some(42)).unwrap();

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
                let y = 1.0
                    + 0.5 * x
                    + 0.1 * unit as f64
                    + 0.05 * time as f64
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
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_driscoll_kraay_basic() {
        let dataset = create_panel_dataset();
        let result = run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, None).unwrap();

        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_params, 2);
        assert_eq!(result.n_periods, 10);
        assert!(
            result.std_errors[0] > 0.0,
            "Intercept SE should be positive"
        );
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
        let bartlett =
            run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, Some("bartlett")).unwrap();
        let parzen =
            run_vcov_driscoll_kraay(&dataset, "y", &["x"], "time", None, Some("parzen")).unwrap();

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
            assert!(
                result.vcov[i][i] >= 0.0,
                "Variance {} should be non-negative",
                i
            );
        }

        // Standard errors should match sqrt of vcov diagonal
        for i in 0..result.n_params {
            let se_from_vcov = result.vcov[i][i].sqrt();
            assert!(
                (result.std_errors[i] - se_from_vcov).abs() < 1e-10,
                "SE {} should match sqrt of vcov diagonal",
                i
            );
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

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    // R-pinned tests (validation/r/cases/regression.R; registry validation/methods/regression.json).

    fn col(r: &OlsResult, f: fn(&OlsCoefficient) -> f64) -> Vec<f64> {
        r.coefficients.iter().map(f).collect()
    }

    /// Coefficients and the inference R reports for SE flavour `tag` ("classical",
    /// "hc0".."hc3"): keys coef, se_<tag>, t_<tag>, p_<tag>, ci95_lower_<tag>, ci95_upper_<tag>.
    fn pin_inference(check: &mut Check<'_>, r: &OlsResult, tag: &str) {
        let n = r.n_obs;
        let names = r.variable_names.as_slice();
        check
            .named("coef", names, &col(r, |c| c.estimate), Tol::coef(n))
            .named(
                &format!("se_{tag}"),
                names,
                &col(r, |c| c.std_error),
                Tol::se(n),
            )
            .named(
                &format!("t_{tag}"),
                names,
                &col(r, |c| c.t_value),
                Tol::stat(n),
            )
            .named(
                &format!("p_{tag}"),
                names,
                &col(r, |c| c.p_value),
                Tol::p_value(n),
            )
            .named(
                &format!("ci95_lower_{tag}"),
                names,
                &col(r, |c| c.ci_lower_95),
                Tol::stat(n),
            )
            .named(
                &format!("ci95_upper_{tag}"),
                names,
                &col(r, |c| c.ci_upper_95),
                Tol::stat(n),
            );
    }

    /// summary.lm / logLik / AIC / BIC, from the struct fields.
    fn pin_fit(check: &mut Check<'_>, r: &OlsResult) {
        let n = r.n_obs;
        check
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .scalar("sigma", r.residual_std_error, Tol::stat(n))
            .scalar("f_stat", r.f_statistic, Tol::stat(n))
            .scalar("f_df1", r.df_model as f64, Tol::exact())
            .scalar("f_df2", r.df_resid as f64, Tol::exact())
            .scalar("f_p", r.f_p_value, Tol::p_value(n))
            .scalar("log_lik", r.log_likelihood, Tol::stat(n))
            .scalar("aic", r.aic, Tol::stat(n))
            .scalar("bic", r.bic, Tol::stat(n))
            .scalar("n_obs", n as f64, Tol::exact())
            .scalar("df_resid", r.df_resid as f64, Tol::exact());
    }

    /// The same quantities through `LinearEstimator`, which is what `p2a reg ols` prints (REG-3).
    fn pin_trait(check: &mut Check<'_>, r: &OlsResult, tag: &str) {
        let e: &dyn LinearEstimator = r;
        let n = e.n_obs();
        let (lo, hi): (Vec<f64>, Vec<f64>) = e.confidence_intervals_95().into_iter().unzip();
        check
            .named(
                &format!("p_{tag}"),
                e.variable_names(),
                &e.p_values().to_vec(),
                Tol::p_value(n),
            )
            .named(
                &format!("ci95_lower_{tag}"),
                e.variable_names(),
                &lo,
                Tol::stat(n),
            )
            .named(
                &format!("ci95_upper_{tag}"),
                e.variable_names(),
                &hi,
                Tol::stat(n),
            )
            .scalar("adj_r2", e.adj_r_squared(), Tol::stat(n))
            .scalar("sigma", e.residual_std_error(), Tol::stat(n))
            .scalar("log_lik", e.log_likelihood(), Tol::stat(n))
            .scalar("aic", e.aic(), Tol::stat(n))
            .scalar("bic", e.bic(), Tol::stat(n));
    }

    /// `lm(inv ~ value + capital)` on Grunfeld (case `ols_grunfeld`), classical SEs.
    #[test]
    fn test_validate_ols_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_ols(
            &ds,
            "inv",
            &["value", "capital"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let mut check = Check::new(&case);
        pin_inference(&mut check, &r, "classical");
        pin_fit(&mut check, &r);
        pin_trait(&mut check, &r, "classical");
        check.finish();
    }

    /// `run_ols_raw` on the same design with HC1 (case `ols_grunfeld`).
    #[test]
    fn test_validate_ols_raw_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let d = DesignMatrix::from_dataframe(ds.df(), &["value", "capital"], true).unwrap();
        let y = DesignMatrix::extract_column(ds.df(), "inv").unwrap();
        let r = run_ols_raw(&d.data, &y, &d.column_names, "inv", CovarianceType::HC1).unwrap();
        let mut check = Check::new(&case);
        pin_inference(&mut check, &r, "hc1");
        pin_fit(&mut check, &r);
        pin_trait(&mut check, &r, "hc1");
        check.finish();
    }

    /// `lm(mpg ~ wt + hp + disp)` on all 32 rows of mtcars (case `ols_mtcars`).
    #[test]
    fn test_validate_ols_mtcars() {
        let (case, ds) = crate::test_support::load_case("ols_mtcars");
        let r = run_ols(
            &ds,
            "mpg",
            &["wt", "hp", "disp"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let mut check = Check::new(&case);
        pin_inference(&mut check, &r, "classical");
        pin_fit(&mut check, &r);
        pin_trait(&mut check, &r, "classical");
        check.finish();
    }

    /// OLS on the ill-conditioned Longley (1967) data (case `ols_longley`), classical SEs.
    ///
    /// Tol::coef(16) = 1e-6 is the README n < 100 row; the normal equations reach about
    /// 3.6e-8 relative here (QR OLS is REG-18, Tier 3). No HC pin on Longley before REG-18.
    #[test]
    fn test_validate_ols_longley() {
        let (case, ds) = crate::test_support::load_case("ols_longley");
        let r = run_ols(
            &ds,
            "Employed",
            &[
                "GNP.deflator",
                "GNP",
                "Unemployed",
                "Armed.Forces",
                "Population",
                "Year",
            ],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let mut check = Check::new(&case);
        pin_inference(&mut check, &r, "classical");
        pin_fit(&mut check, &r);
        check.finish();
    }

    /// Validation test: Simple linear regression with known DGP
    /// y = 2.0 + 1.5*x + noise
    #[test]
    fn test_validate_ols_simple_regression() {
        // Create data with known DGP: y = 2.0 + 1.5*x + N(0, 0.5)
        // Using fixed seed for reproducibility
        let x: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                // Deterministic noise pattern for reproducibility
                let noise = (i as f64 * 1.234).sin() * 0.5;
                2.0 + 1.5 * xi + noise
            })
            .collect();

        let df = df! {
            "y" => y.clone(),
            "x" => x.clone()
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).unwrap();

        // Intercept should be close to 2.0
        assert!(
            (result.coefficients[0].estimate - 2.0).abs() < 0.2,
            "Intercept should be close to 2.0, got {}",
            result.coefficients[0].estimate
        );

        // Slope should be close to 1.5
        assert!(
            (result.coefficients[1].estimate - 1.5).abs() < 0.1,
            "Slope should be close to 1.5, got {}",
            result.coefficients[1].estimate
        );

        // High R² expected with low noise
        assert!(
            result.r_squared > 0.95,
            "R² should be > 0.95, got {}",
            result.r_squared
        );
    }

    /// HC0-HC3 on Grunfeld against `sandwich::vcovHC(type = )` (case `ols_grunfeld`).
    #[test]
    fn test_validate_ols_robust_se_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let mut check = Check::new(&case);
        for (cov, tag) in [
            (CovarianceType::HC0, "hc0"),
            (CovarianceType::HC1, "hc1"),
            (CovarianceType::HC2, "hc2"),
            (CovarianceType::HC3, "hc3"),
        ] {
            let r = run_ols(&ds, "inv", &["value", "capital"], true, cov).unwrap();
            assert_eq!(r.cov_type, cov);
            pin_inference(&mut check, &r, tag);
        }
        check.finish();
    }

    /// HC0-HC3 on mtcars, `mpg ~ wt + hp + disp` (case `ols_mtcars`).
    #[test]
    fn test_validate_ols_robust_se_mtcars() {
        let (case, ds) = crate::test_support::load_case("ols_mtcars");
        let mut check = Check::new(&case);
        for (cov, tag) in [
            (CovarianceType::HC0, "hc0"),
            (CovarianceType::HC1, "hc1"),
            (CovarianceType::HC2, "hc2"),
            (CovarianceType::HC3, "hc3"),
        ] {
            let r = run_ols(&ds, "mpg", &["wt", "hp", "disp"], true, cov).unwrap();
            assert_eq!(r.cov_type, cov);
            pin_inference(&mut check, &r, tag);
        }
        check.finish();
    }

    // Cluster-robust SEs against `sandwich::vcovCL(type = "HC1")` (CR1: G/(G-1) and
    // (n-1)/(n-k)). The case's se_cluster_firm_hc0 is deliberately not pinned: p2a offers
    // CR1 only, and vcovCL(type = "HC0") applies G/(G-1) without (n-1)/(n-k).

    /// One-way clustering by firm (case `ols_grunfeld`).
    #[test]
    fn test_validate_ols_clustered_grunfeld_firm() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_ols_clustered(&ds, "inv", &["value", "capital"], "firm", None).unwrap();
        assert_eq!((r.n_clusters_1, r.n_clusters_2), (10, None));
        let n = r.ols.n_obs;
        let names = r.ols.variable_names.as_slice();
        Check::new(&case)
            .named("coef", names, &col(&r.ols, |c| c.estimate), Tol::coef(n))
            .named(
                "se_cluster_firm_hc1",
                names,
                &col(&r.ols, |c| c.std_error),
                Tol::se(n),
            )
            .named(
                "p_cluster_firm_hc1",
                names,
                &col(&r.ols, |c| c.p_value),
                Tol::p_value(n),
            )
            // What `p2a reg clustered` prints.
            .scalar(
                "adj_r2",
                LinearEstimator::adj_r_squared(&r.ols),
                Tol::stat(n),
            )
            .finish();
    }

    /// One-way clustering by year (case `ols_grunfeld`).
    #[test]
    fn test_validate_ols_clustered_grunfeld_year() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_ols_clustered(&ds, "inv", &["value", "capital"], "year", None).unwrap();
        assert_eq!((r.n_clusters_1, r.n_clusters_2), (20, None));
        let n = r.ols.n_obs;
        let names = r.ols.variable_names.as_slice();
        Check::new(&case)
            .named("coef", names, &col(&r.ols, |c| c.estimate), Tol::coef(n))
            .named(
                "se_cluster_year_hc1",
                names,
                &col(&r.ols, |c| c.std_error),
                Tol::se(n),
            )
            .finish();
    }

    /// Two-way clustering by firm and year, V_firm + V_year - V_firm∩year, each with its own
    /// G/(G-1) (vcovCL's default multi0 = FALSE; case `ols_grunfeld`).
    #[test]
    fn test_validate_ols_clustered_grunfeld_firm_year() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_ols_clustered(&ds, "inv", &["value", "capital"], "firm", Some("year")).unwrap();
        assert_eq!((r.n_clusters_1, r.n_clusters_2), (10, Some(20)));
        let n = r.ols.n_obs;
        let names = r.ols.variable_names.as_slice();
        Check::new(&case)
            .named("coef", names, &col(&r.ols, |c| c.estimate), Tol::coef(n))
            .named(
                "se_cluster_firm_year_hc1",
                names,
                &col(&r.ols, |c| c.std_error),
                Tol::se(n),
            )
            .finish();
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - HAC and Driscoll-Kraay (R cases); bootstrap property test
    // ════════════════════════════════════════════════════════════════════════════

    /// Pinned to case `hac_ar1` (sandwich::NeweyWest / kernHAC on the legacy Test 8 DGP; validation/r/cases/regression_legacy.R).
    #[test]
    fn test_validate_hac_ar1() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hac_ar1");
        let n = case.scalar("n_obs") as usize;
        let bw = case.scalar("bw_rule") as usize; // 4 = floor(4 (n/100)^(2/9))
        let hac = |bw: Option<usize>, k: &str, pw: bool| {
            run_vcov_hac(&ds, "y", &["x"], bw, Some(k), pw).unwrap()
        };
        let auto = hac(None, "bartlett", false);
        let bartlett = hac(Some(bw), "bartlett", false);
        let parzen = hac(Some(bw), "parzen", false);
        let truncated = hac(Some(bw), "truncated", false);
        let tukey = hac(Some(bw), "tukey-hanning", false);
        Check::new(&case)
            .named("coef", &auto.names, &auto.coefficients, Tol::coef(n))
            .scalar("bw_rule", auto.bandwidth as f64, Tol::exact())
            // p2a's Bartlett bandwidth m weights lag j by 1 - j/(m + 1):
            // NeweyWest(lag = m) = kernHAC(bw = m + 1)
            .named(
                "neweywest_lag4",
                &bartlett.names,
                &bartlett.std_errors,
                Tol::se(n),
            )
            .vector("neweywest_lag4_vcov", &bartlett.vcov.concat(), Tol::se(n))
            .named(
                "kernhac_bartlett_bw5",
                &bartlett.names,
                &bartlett.std_errors,
                Tol::se(n),
            )
            .named(
                "kernhac_parzen_bw4",
                &parzen.names,
                &parzen.std_errors,
                Tol::se(n),
            )
            .vector("kernhac_parzen_bw4_vcov", &parzen.vcov.concat(), Tol::se(n))
            .named(
                "kernhac_truncated_bw4",
                &truncated.names,
                &truncated.std_errors,
                Tol::se(n),
            )
            .vector(
                "kernhac_truncated_bw4_vcov",
                &truncated.vcov.concat(),
                Tol::se(n),
            )
            .named(
                "kernhac_tukey_hanning_bw4",
                &tukey.names,
                &tukey.std_errors,
                Tol::se(n),
            )
            .vector(
                "kernhac_tukey_hanning_bw4_vcov",
                &tukey.vcov.concat(),
                Tol::se(n),
            )
            .finish();
        assert_eq!(auto.std_errors, bartlett.std_errors); // default bandwidth = bw_rule
        // Tier 0 (REG-1, 428d7cf): a user bandwidth above n/4 warns; 4 <= 100/4, and the
        // default never warns
        assert!(
            [&auto, &bartlett, &parzen, &truncated, &tukey]
                .iter()
                .all(|r| r.warnings.is_empty())
        );
        // deviation REG-50: "qs" truncates the QS kernel at the bandwidth (kernhac_qs_bw4),
        //   and p2a's Bartlett bandwidth m is kernHAC(bw = m + 1), not kernhac_bartlett_bw4.
        // deviation REG-22: prewhiten = true is a scalar AR(1) on the residuals
        //   (neweywest_lag4_prewhite).
        // deviation REG-61: default bandwidth is the fixed rule, not bwNeweyWest
        //   (bw_neweywest); no adjust = TRUE (kernhac_*_adjust).
        let qs = hac(Some(bw), "qs", false);
        let prewhitened = hac(Some(bw), "bartlett", true);
        assert!(
            qs.std_errors
                .iter()
                .chain(&prewhitened.std_errors)
                .all(|s| s.is_finite() && *s > 0.0)
        );
        assert!(prewhitened.prewhiten && matches!(qs.kernel, HacKernel::QuadraticSpectral));
        assert!(
            matches!(auto.kernel, HacKernel::Bartlett)
                && matches!(parzen.kernel, HacKernel::Parzen)
        );
        assert_eq!((auto.n_obs, auto.n_params), (n, 2));
    }

    /// Property test, not R-pinned: sandwich::vcovBS resamples with R's RNG, which cannot be
    /// reproduced; see validation/regression/vcovBS.md.
    #[test]
    fn test_bootstrap_pairs_and_wild_on_heteroskedastic_data() {
        // Create heteroskedastic data
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 2.345).sin() * 2.0).collect();

        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let noise_scale = 1.0 + xi.abs();
                let noise = ((i as f64 * 1.234).cos()) * noise_scale * 0.5;
                2.0 + 3.0 * xi + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Test pairs bootstrap
        let result_pairs =
            run_vcov_bootstrap(&dataset, "y", &["x"], Some(200), Some("pairs"), Some(42)).unwrap();

        assert_eq!(result_pairs.n_obs, 50);
        assert!(matches!(result_pairs.bootstrap_type, BootstrapType::Pairs));
        assert!(
            result_pairs.std_errors[0] > 0.0,
            "Pairs bootstrap intercept SE should be positive"
        );
        assert!(
            result_pairs.std_errors[1] > 0.0,
            "Pairs bootstrap slope SE should be positive"
        );

        // Test wild bootstrap
        let result_wild =
            run_vcov_bootstrap(&dataset, "y", &["x"], Some(200), Some("wild"), Some(42)).unwrap();

        assert!(matches!(result_wild.bootstrap_type, BootstrapType::Wild));
        assert!(
            result_wild.std_errors[0] > 0.0,
            "Wild bootstrap intercept SE should be positive"
        );
        assert!(
            result_wild.std_errors[1] > 0.0,
            "Wild bootstrap slope SE should be positive"
        );

        // Bootstrap SEs from different methods should be in similar ballpark
        // (with same data, pairs and wild should give roughly similar results)
        let ratio_intercept = result_pairs.std_errors[0] / result_wild.std_errors[0];
        assert!(
            ratio_intercept > 0.5 && ratio_intercept < 2.0,
            "Pairs and wild bootstrap SEs should be within 2x: ratio={:.3}",
            ratio_intercept
        );

        // Convergence rate should be high
        assert!(
            result_pairs.convergence_rate > 0.9,
            "Bootstrap convergence rate should be > 90%: got {:.2}%",
            result_pairs.convergence_rate * 100.0
        );
    }

    /// Case `dk_panel` (sandwich::vcovPL / plm::vcovSCC on the legacy Test 10 DGP;
    /// validation/r/cases/regression_legacy.R). Only the parts p2a reproduces are pinned.
    #[test]
    fn test_validate_driscoll_kraay_panel() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("dk_panel");
        let n = case.scalar("n_obs") as usize;
        let dk = run_vcov_driscoll_kraay(&ds, "y", &["x"], "time", None, None).unwrap();
        let ols = run_ols(&ds, "y", &["x"], true, CovarianceType::Standard).unwrap();
        let ols_se: Vec<f64> = ols.coefficients.iter().map(|c| c.std_error).collect();
        Check::new(&case)
            .named("coef", &dk.names, &dk.coefficients, Tol::coef(n))
            .named("se_ols", &dk.names, &ols_se, Tol::se(n))
            // default bandwidth floor(T^(1/4)) = vcovPL's lag = "NW1987"; Bartlett
            // 1 - j/(m + 1) = vcovPL's bw = lag + 1
            .scalar("lag", dk.bandwidth as f64, Tol::exact())
            .scalar("n_periods", dk.n_periods as f64, Tol::exact())
            .scalar("n_obs", dk.n_obs as f64, Tol::exact())
            .finish();
        // deviation REG-24: std_errors, vcov, t_stats and p_values average the scores within
        //   period and scale by T; here they are sqrt(T)/N = 0.4472 times vcovpl_lag2
        //   (= vcovscc_lag2).
        assert_eq!(dk.n_units, 10);
        assert!(matches!(dk.kernel, HacKernel::Bartlett));
        // Tier 0 (REG-1, 428d7cf) warns only for a user bandwidth above T/4
        assert!(dk.warnings.is_empty());
        let ratio = dk.std_errors[0] / ols_se[0];
        assert!(
            (ratio - 1.0).abs() > 0.01,
            "DK and OLS SEs should differ: ratio={ratio:.3}"
        );
    }

    // ========================================
    // Resource bounds (T0-WP14)
    // ========================================

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    #[test]
    fn test_hac_rejects_bandwidth_at_or_above_n() {
        let ds = create_timeseries_dataset();
        for bw in [30usize, 31, 300] {
            match run_vcov_hac(&ds, "y", &["x"], Some(bw), None, false).unwrap_err() {
                EconError::InvalidSpecification { message } => {
                    assert!(
                        message.contains("bandwidth") && message.contains("30"),
                        "{message}"
                    )
                }
                e => panic!("bw {bw}: {e:?}"),
            }
        }
        assert!(run_vcov_hac(&ds, "y", &["x"], Some(29), None, false).is_ok());
    }

    #[test]
    fn test_hac_warns_when_bandwidth_exceeds_quarter_of_n() {
        let ds = create_timeseries_dataset();
        let r = run_vcov_hac(&ds, "y", &["x"], Some(8), None, false).unwrap(); // 32 > 30
        assert_eq!(r.warnings.len(), 1);
        assert!(r.to_string().contains("Warnings:"));
        // 28 <= 30
        let r = run_vcov_hac(&ds, "y", &["x"], Some(7), None, false).unwrap();
        assert!(r.warnings.is_empty());
        assert!(!r.to_string().contains("Warnings:"));
        // the default rule never warns
        let r = run_vcov_hac(&ds, "y", &["x"], None, None, false).unwrap();
        assert!(r.warnings.is_empty());
        // JSON is unchanged without warnings
        assert!(!serde_json::to_string(&r).unwrap().contains("warnings"));
    }

    #[test]
    fn test_driscoll_kraay_rejects_bandwidth_at_or_above_t() {
        let ds = create_panel_dataset();
        let e = run_vcov_driscoll_kraay(&ds, "y", &["x"], "time", Some(10), None).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("bandwidth")),
            "{e:?}"
        );
        let r = run_vcov_driscoll_kraay(&ds, "y", &["x"], "time", Some(9), None).unwrap();
        assert_eq!(r.warnings.len(), 1);
        assert!(r.to_string().contains("Warnings:"));
        let r = run_vcov_driscoll_kraay(&ds, "y", &["x"], "time", Some(2), None).unwrap();
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn test_hac_and_dk_stop_when_cancelled() {
        let ts = create_timeseries_dataset();
        let e = resource::with_budget(cancelled(), || {
            run_vcov_hac(&ts, "y", &["x"], Some(5), None, false)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
        let p = create_panel_dataset();
        let e = resource::with_budget(cancelled(), || {
            run_vcov_driscoll_kraay(&p, "y", &["x"], "time", Some(3), None)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_bootstrap_rejects_too_few_replications() {
        let ds = create_test_dataset();
        for b in [0usize, 9] {
            let e = run_vcov_bootstrap(&ds, "y", &["x1"], Some(b), Some("pairs"), Some(42))
                .unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("n_boot")),
                "{e:?}"
            );
        }
    }

    #[test]
    fn test_bootstrap_impossible_replication_count_is_an_error() {
        let ds = create_test_dataset();
        let e = run_vcov_bootstrap(
            &ds,
            "y",
            &["x1"],
            Some(1usize << 61),
            Some("pairs"),
            Some(42),
        )
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_bootstrap_replicate_storage_respects_alloc_ceiling() {
        // 100_000 replicates * (k + 3) * 8 bytes = 4 MB > 1 MiB
        let ds = create_test_dataset();
        let e = resource::with_alloc_ceiling(Some(1 << 20), || {
            run_vcov_bootstrap(&ds, "y", &["x1"], Some(100_000), Some("pairs"), Some(42))
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_bootstrap_stops_when_cancelled() {
        let ds = create_test_dataset();
        let e = resource::with_budget(cancelled(), || {
            run_vcov_bootstrap(&ds, "y", &["x1"], Some(200), Some("wild"), Some(42))
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
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

    /// The truncated kernel at bandwidth n - 1 gives every lag weight 1, so the
    /// meat is (X'u)(X'u)'/n, which the OLS normal equations make zero up to
    /// rounding. The variances are then noise of about 1e-17 whose sign depends
    /// on the machine's floating-point path, so they are bounded here, not
    /// pinned.
    #[test]
    fn test_hac_truncated_kernel_at_n_minus_1_collapses() {
        let ts = create_timeseries_dataset();
        let r = run_vcov_hac(&ts, "y", &["x"], Some(29), Some("truncated"), false).unwrap();
        assert!(r.std_errors.iter().all(|&s| s < 1e-6), "{:?}", r.std_errors);
        // the bandwidth-above-n/4 warning
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    }

    /// Characterisation test: the code before the T0-WP14 resource bounds gives
    /// the same numbers, and later refactors must not move them.
    ///
    /// The truncated kernel is pinned at bandwidth 5, where its SEs are 0.148
    /// and 0.0072. At bandwidth n - 1 = 29 they collapse to rounding noise
    /// (see `test_hac_truncated_kernel_at_n_minus_1_collapses`). A numpy
    /// re-implementation of the lag loop reproduces the three HAC pins to
    /// 2e-14, and moving beta by up to 10 ulp changes their variances by at
    /// most 3e-13 relative.
    #[test]
    fn test_pinned_hac_dk_bootstrap_outputs() {
        let ts = create_timeseries_dataset();
        let mut v = Vec::new();
        for (bw, kernel) in [(5usize, "bartlett"), (29, "bartlett"), (5, "truncated")] {
            let r = run_vcov_hac(&ts, "y", &["x"], Some(bw), Some(kernel), false).unwrap();
            v.extend(&r.std_errors);
        }
        let panel = create_panel_dataset();
        let dk = run_vcov_driscoll_kraay(&panel, "y", &["x"], "time", Some(9), None).unwrap();
        v.extend(&dk.std_errors);
        let ds = create_test_dataset();
        for kind in ["pairs", "residual", "wild"] {
            let b = run_vcov_bootstrap(&ds, "y", &["x1"], Some(200), Some(kind), Some(42)).unwrap();
            v.extend(&b.std_errors);
            v.push(b.convergence_rate);
        }
        assert_pinned(
            "HAC/DK/bootstrap",
            &v,
            &[
                0.1380678477098943,
                0.0067448421041296615,
                0.10055839075154051,
                0.005817042539300054,
                0.14772400250120787,
                0.007223766353723901,
                0.01250309685000187,
                0.0011882147319589296,
                0.09631970723095434,
                0.01566998133541119,
                1.0,
                0.10357515044968392,
                0.017372550023008526,
                1.0,
                0.08621209537976078,
                0.01294828222674302,
                1.0,
            ],
        );
    }
}
