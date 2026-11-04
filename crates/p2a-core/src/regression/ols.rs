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
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::data::Dataset;
use crate::data::columns::{
    ColumnRequest, Columns, Factor, MissingSummary, NaPolicy, econ_error_from_design,
};
use crate::errors::{EconError, EconResult, EstimationWarning};
use crate::linalg::solve::{
    LM_QR_TOL, Qr, SingularPolicy, aliasing_message, least_squares, sym_inverse,
};
use crate::linalg::{DesignMatrix, matmul, xtx, xty};
use crate::options::Choices;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
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
    /// Degrees of freedom for model (k - 1 with an intercept, k without)
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
    /// Rows dropped because an identifier (a cluster id) was missing. Empty for
    /// `run_ols` and `run_ols_raw`, which refuse missing values; absent from JSON when
    /// nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Names of the aliased (linearly dependent) design columns, in column order; their
    /// coefficient, SE, t, p and CI are NaN, as R's `lm` reports NA. Absent from JSON when
    /// nothing is aliased.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliased: Vec<String>,

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
    /// Total sum of squares (cached; used by R², F-test, diagnostics): about the mean
    /// with an intercept, about zero without (summary.lm).
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
                p2a_text::ellipsize(&coef.name, 15),
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
    /// Number of identified coefficients (R's `fit$rank`): `n_params - aliased.len()`.
    pub fn rank(&self) -> usize {
        self.n_params - self.aliased.len()
    }

    /// The coefficient table and model statistics as CSV: the same as [`CsvExport`]
    /// (`crate::export::CsvExport`), kept so method syntax works without importing the
    /// trait. Names are escaped; the columns are `variable, estimate, std_error, t_value,
    /// p_value, significance, ci_lower_95, ci_upper_95`, then a `# Model Statistics` block.
    ///
    /// [`CsvExport`]: crate::export::CsvExport
    pub fn to_csv_string(&self) -> String {
        <Self as crate::export::CsvExport>::to_csv_string(self)
    }

    /// Write [`OlsResult::to_csv_string`] to a file (the same as `CsvExport::to_csv`).
    pub fn to_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::export::CsvExport::to_csv(self, path)
    }

    /// Export results to a JSON string.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl ResultNotes for OlsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl ResultNotes for OlsClusteredResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.ols.warnings.clone()
    }
}

impl ResultNotes for HacResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl ResultNotes for DriscollKraayResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl ResultNotes for BootstrapResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
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

/// True when some column of `x` is constant and non-zero, i.e. the model contains an
/// intercept in any position.
///
/// R takes the intercept from the model's terms (`attr(terms, "intercept")`); a raw design
/// matrix has no terms, so a constant non-zero column marks it. A column counts as constant
/// when its first entry `v0` is finite and non-zero and every entry lies within
/// `1e-10 * max(|v0|, 1)` of it. The scan stops at the first entry that differs.
pub(crate) fn has_constant_column(x: ndarray::ArrayView2<'_, f64>) -> bool {
    if x.nrows() == 0 {
        return false;
    }
    x.columns().into_iter().any(|col| {
        let v0 = col[0];
        if !v0.is_finite() || v0 == 0.0 {
            return false;
        }
        let tol = 1e-10 * v0.abs().max(1.0);
        col.iter().all(|&v| (v - v0).abs() <= tol)
    })
}

/// The fit statistics that depend on whether the model has an intercept.
struct FitSummary {
    /// Total sum of squares: about the mean with an intercept, about zero without.
    sst: f64,
    r_squared: f64,
    adj_r_squared: f64,
    /// Numerator df of the overall F test: `k - 1` with an intercept, `k` without.
    df_model: usize,
    f_statistic: f64,
    f_p_value: f64,
}

/// R², adjusted R² and the overall F test as R's `summary.lm` computes them (REG-19).
///
/// With an intercept the total sum of squares is centered, adjusted R² uses `n - 1` and F
/// tests the `k - 1` slopes. Without one (regression through the origin) the total sum of
/// squares is `Σ y²`, adjusted R² uses `n` and F tests all `k` coefficients, so
/// `R² = Σ ŷ² / (Σ ŷ² + SSR)`. `ssr` is the residual sum of squares, `sigma_squared` is
/// `ssr / df_resid`.
fn fit_summary(
    y: &Array1<f64>,
    ssr: f64,
    k: usize,
    df_resid: usize,
    sigma_squared: f64,
    intercept: bool,
) -> FitSummary {
    let n = y.len();
    let (sst, df_int) = if intercept {
        let y_mean = y.mean().unwrap_or(0.0);
        (y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>(), 1)
    } else {
        // summary.lm without an intercept: about zero
        (y.iter().map(|&yi| yi.powi(2)).sum::<f64>(), 0)
    };
    // Explained sum of squares; Σ ŷ² without an intercept (the residuals are orthogonal to ŷ)
    let sse = sst - ssr;
    let df_model = k.saturating_sub(df_int);

    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
    let adj_r_squared = if sst > 0.0 && df_resid > 0 {
        1.0 - (1.0 - r_squared) * ((n - df_int) as f64) / (df_resid as f64)
    } else {
        0.0
    };
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
    FitSummary {
        sst,
        r_squared,
        adj_r_squared,
        df_model,
        f_statistic,
        f_p_value,
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

    // The model has an intercept when some column is constant and non-zero, in any
    // position (REG-19; R reads it from the terms, which a raw matrix does not have).
    let has_intercept = has_constant_column(x.view());

    fit_ols(x.view(), y, variable_names, y_name, has_intercept, cov_type)
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
        .map_err(|e| econ_error_from_design(e, df))?;

    // Extract y vector
    let y = DesignMatrix::extract_column(df, y_col).map_err(|e| econ_error_from_design(e, df))?;

    fit_ols(
        design.data.view(),
        &y,
        &design.column_names,
        y_col,
        intercept,
        cov_type,
    )
}

/// Which solve an OLS fit took (decision D1, variant B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OlsPath {
    /// Cholesky inverse of X'X: full rank and well conditioned.
    NormalEquations,
    /// `lm.fit`'s pivoted QR (`linalg::solve::least_squares`): rank-deficient or
    /// ill-conditioned designs.
    PivotedQr,
}

/// The normal equations are used only when the diagonally equilibrated X'X has a
/// reciprocal condition of at least this value (`sym_inverse`'s estimate, which is a lower
/// bound, so the check errs toward the QR). Their relative error grows like
/// `eps / rcond`; below the threshold the pivoted QR (error `eps * sqrt(1 / rcond)`)
/// decides rank and coefficients.
const OLS_NORMAL_EQUATIONS_MIN_RCOND: f64 = 1e-8;

/// `(X'X)^{-1}` when the design is of full rank and well conditioned (see
/// [`OLS_NORMAL_EQUATIONS_MIN_RCOND`]); `None` when the pivoted QR must decide.
/// Cancellation and the allocation ceiling propagate; any other failure of the inverse
/// (a singular or indefinite X'X) means "use the QR".
fn normal_equations_inverse(x: ndarray::ArrayView2<'_, f64>) -> EconResult<Option<Array2<f64>>> {
    let (n, k) = x.dim();
    if k == 0 || n <= k {
        return Ok(None);
    }
    let gram = xtx(&x);
    match sym_inverse(gram.view(), SingularPolicy::Error, "X'X") {
        Ok(inv)
            if inv.reciprocal_condition >= OLS_NORMAL_EQUATIONS_MIN_RCOND
                && inv.inverse.diag().iter().all(|v| v.is_finite() && *v > 0.0) =>
        {
            Ok(Some(inv.inverse))
        }
        Ok(_) => Ok(None),
        Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => Err(e),
        Err(_) => Ok(None),
    }
}

/// The least-squares solution of one OLS fit.
struct OlsSolve {
    /// Coefficients in column order, `NaN` for aliased columns.
    beta: Array1<f64>,
    residuals: Array1<f64>,
    /// `(X'X)^{-1}` (k x k), `NaN` rows and columns for aliased columns.
    unscaled: Array2<f64>,
    rank: usize,
    /// Indices of the aliased columns, ascending.
    aliased: Vec<usize>,
    path: OlsPath,
}

/// Solve `min ||y - X b||`: the normal equations on a full-rank, well-conditioned design
/// (bit for bit the solve OLS always used there), `lm.fit`'s pivoted QR otherwise.
fn solve_ols(
    x: ndarray::ArrayView2<'_, f64>,
    y: ndarray::ArrayView1<'_, f64>,
) -> EconResult<OlsSolve> {
    if let Some(unscaled) = normal_equations_inverse(x)? {
        let y = y.to_owned();
        let beta = unscaled.dot(&xty(&x, &y));
        let residuals = &y - &x.dot(&beta);
        return Ok(OlsSolve {
            beta,
            residuals,
            unscaled,
            rank: x.ncols(),
            aliased: Vec::new(),
            path: OlsPath::NormalEquations,
        });
    }
    let ls = least_squares(x, y)?;
    Ok(OlsSolve {
        beta: ls.coefficients,
        residuals: ls.residuals,
        unscaled: ls.unscaled_cov,
        rank: ls.rank,
        aliased: ls.aliased,
        path: OlsPath::PivotedQr,
    })
}

/// The columns of `x` listed in `cols`, through the allocation ceiling.
fn select_columns(x: ndarray::ArrayView2<'_, f64>, cols: &[usize]) -> EconResult<Array2<f64>> {
    let mut out = resource::try_zeros2(x.nrows(), cols.len(), "OLS identified design")?;
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&x.column(c));
    }
    Ok(out)
}

/// The `(cols, cols)` block of a square matrix.
fn select_block(m: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((cols.len(), cols.len()), |(i, j)| m[[cols[i], cols[j]]])
}

/// A k x k matrix holding `m` at `(cols, cols)` and `NaN` elsewhere.
fn embed_block(m: &Array2<f64>, cols: &[usize], k: usize) -> Array2<f64> {
    let mut out = Array2::from_elem((k, k), f64::NAN);
    for (i, &ci) in cols.iter().enumerate() {
        for (j, &cj) in cols.iter().enumerate() {
            out[[ci, cj]] = m[[i, j]];
        }
    }
    out
}

/// The identified (non-aliased) columns of a fitted OLS model's design and the matching
/// block of `(X'X)^{-1}`, taken from the fit itself (no second decomposition). `sandwich`
/// drops aliased columns from `estfun` and `bread`; every sandwich and resampling estimator
/// here works on these columns and embeds its result back with `NaN` for aliased ones.
struct IdentifiedDesign<'a> {
    /// Original indices of the identified columns, ascending.
    cols: Vec<usize>,
    /// Number of design columns (identified and aliased).
    k: usize,
    /// The identified columns (a view of the design when nothing is aliased).
    x: ndarray::CowArray<'a, f64, ndarray::Ix2>,
    /// `(X1'X1)^{-1}` of the identified columns.
    unscaled: Array2<f64>,
}

impl<'a> IdentifiedDesign<'a> {
    /// The identified part of `x`, the design `ols` was fitted on. Refuses an `x` whose shape
    /// is not the fit's.
    fn of(x: ndarray::ArrayView2<'a, f64>, ols: &OlsResult) -> EconResult<Self> {
        let k = ols.n_params;
        if x.nrows() != ols.n_obs || x.ncols() != k {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "design matrix is {} x {} but the OLS fit has {} observations and {} \
                     parameters",
                    x.nrows(),
                    x.ncols(),
                    ols.n_obs,
                    k
                ),
            });
        }
        let cols: Vec<usize> = (0..k).filter(|&j| !ols.beta[j].is_nan()).collect();
        if cols.len() == k {
            return Ok(IdentifiedDesign {
                cols,
                k,
                x: ndarray::CowArray::from(x),
                unscaled: ols.xtx_inv.clone(),
            });
        }
        let x1 = select_columns(x, &cols)?;
        let unscaled = select_block(&ols.xtx_inv, &cols);
        Ok(IdentifiedDesign {
            cols,
            k,
            x: ndarray::CowArray::from(x1),
            unscaled,
        })
    }

    /// Number of identified columns (the rank of the fit).
    fn rank(&self) -> usize {
        self.cols.len()
    }

    /// `m` (rank x rank) as a k x k matrix with `NaN` rows and columns for aliased columns.
    fn embed(&self, m: &Array2<f64>) -> Array2<f64> {
        if self.cols.len() == self.k {
            m.clone()
        } else {
            embed_block(m, &self.cols, self.k)
        }
    }

    /// `v` (length rank) as a length-k vector with `NaN` for aliased columns.
    fn embed_vec(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![f64::NAN; self.k];
        for (&c, &x) in self.cols.iter().zip(v) {
            out[c] = x;
        }
        out
    }
}

/// The aliasing warnings of a fit ("Coefficients not defined because of singularities: ..."),
/// copied verbatim, for the estimators built on it.
fn alias_notes(ols: &OlsResult) -> Vec<String> {
    ols.warnings
        .iter()
        .filter(|w| w.starts_with("Coefficients not defined because of singularities"))
        .cloned()
        .collect()
}

/// A fixed design's least-squares solver for many responses (the residual and wild
/// bootstraps): the inverse of X'X on a full-rank, well-conditioned design (the normal
/// equations, as [`solve_ols`]), the pivoted QR otherwise. One factorisation, O(n k) per
/// response.
enum FixedDesignSolver {
    Normal(Array2<f64>),
    Qr(Qr),
}

impl FixedDesignSolver {
    fn new(x: ndarray::ArrayView2<'_, f64>) -> EconResult<Self> {
        Ok(match normal_equations_inverse(x)? {
            Some(u) => FixedDesignSolver::Normal(u),
            None => FixedDesignSolver::Qr(Qr::decompose(x, LM_QR_TOL)?),
        })
    }

    /// Coefficients of `y` on `x` (the design this solver was built from).
    fn coefficients(
        &self,
        x: ndarray::ArrayView2<'_, f64>,
        y: &Array1<f64>,
    ) -> EconResult<Array1<f64>> {
        match self {
            FixedDesignSolver::Normal(u) => Ok(u.dot(&xty(&x, y))),
            FixedDesignSolver::Qr(q) => q.coefficients(y.view()),
        }
    }
}

/// One OLS fit on arrays, shared by [`run_ols`] and [`run_ols_raw`]: the least-squares
/// solution ([`solve_ols`]), fit statistics on the identified columns as `summary.lm`, and
/// the requested covariance on the identified columns (as `sandwich`). Aliased columns get
/// `NaN` coefficients, SEs, t, p and CIs and R's warning.
fn fit_ols(
    x: ndarray::ArrayView2<'_, f64>,
    y: &Array1<f64>,
    names: &[String],
    y_name: &str,
    intercept: bool,
    cov_type: CovarianceType,
) -> EconResult<OlsResult> {
    let (n, k) = x.dim();
    // lm.fit refuses NA/NaN/Inf; run_ols input is already NA-checked by the extractor.
    if x.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "OLS input contains non-finite values (NA/NaN/Inf) in x".to_string(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "OLS input contains non-finite values (NA/NaN/Inf) in y".to_string(),
        });
    }
    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "OLS regression".to_string(),
        });
    }

    let solve = solve_ols(x, y.view())?;
    let rank = solve.rank;
    if rank == 0 {
        return Err(EconError::SingularMatrix {
            context: "OLS design matrix has rank 0".to_string(),
            suggestion: "every column is zero; include an intercept or a non-zero regressor"
                .to_string(),
        });
    }
    let df_resid =
        n.checked_sub(rank)
            .filter(|d| *d > 0)
            .ok_or_else(|| EconError::InsufficientData {
                required: rank + 1,
                provided: n,
                context: format!("OLS regression with {rank} identified parameters"),
            })?;

    let mut warnings = Vec::new();
    if let Some(w) = aliasing_message(&solve.aliased, names) {
        warnings.push(w);
    }
    let aliased: Vec<String> = solve
        .aliased
        .iter()
        .map(|&j| {
            names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("x{}", j + 1))
        })
        .collect();

    let OlsSolve {
        beta,
        residuals,
        unscaled: xtx_inv,
        aliased: aliased_idx,
        ..
    } = solve;

    // Compute statistics
    let ssr: f64 = residuals.iter().map(|&e| e.powi(2)).sum();
    let sigma_squared = ssr / (df_resid as f64);
    let residual_std_error = sigma_squared.sqrt();
    let FitSummary {
        sst,
        r_squared,
        adj_r_squared,
        df_model,
        f_statistic,
        f_p_value,
    } = fit_summary(y, ssr, rank, df_resid, sigma_squared, intercept);

    let (log_likelihood, aic, bic) = gaussian_fit_criteria(ssr, n, rank);

    // Covariance on the identified columns; aliased rows and columns are NaN.
    let vcov = if aliased.is_empty() {
        compute_vcov(&x, &residuals, &xtx_inv, cov_type, n, df_resid)?
    } else {
        let cols: Vec<usize> = (0..k).filter(|j| !aliased_idx.contains(j)).collect();
        let x1 = select_columns(x, &cols)?;
        let u1 = select_block(&xtx_inv, &cols);
        let v1 = compute_vcov(&x1.view(), &residuals, &u1, cov_type, n, df_resid)?;
        embed_block(&v1, &cols, k)
    };

    // Extract standard errors from diagonal
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // Compute t-statistics and p-values (NaN for aliased columns)
    let t_stats: Array1<f64> = &beta / &se;
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();

    // Compute 95% confidence intervals
    let t_crit = crate::traits::t_critical(0.05, df_resid as f64);

    // Build coefficient results
    let coefficients: Vec<OlsCoefficient> = names
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
        variable_names: names.to_vec(),
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
        missing: MissingSummary::none(n),
        aliased,
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
    df_resid: usize,
) -> EconResult<Array2<f64>> {
    match cov_type {
        CovarianceType::Standard => {
            // Standard OLS: σ² (X'X)^{-1}
            let sigma2 = residuals.iter().map(|&e| e.powi(2)).sum::<f64>() / (df_resid as f64);
            Ok(xtx_inv * sigma2)
        }
        CovarianceType::HC0 | CovarianceType::HC1 | CovarianceType::HC2 | CovarianceType::HC3 => {
            // Robust standard errors using sandwich estimator
            compute_hc_vcov(x, residuals, xtx_inv, cov_type, n, df_resid)
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
    df_resid: usize,
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
                    let correction = (n as f64) / (df_resid as f64);
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
                p2a_text::ellipsize(&coef.name, 15),
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
/// Cluster ids are read as factors (strings, integers, floats, booleans or dates). A row
/// whose id is missing, in either dimension of a two-way clustering, is dropped and the
/// model is re-estimated on the remaining rows, as `fixest::feols(cluster = )` does
/// (`sandwich::vcovCL` refuses missing clusters). The drop is reported in
/// `OlsResult::missing` and as the first warning; when no row has an id the error is
/// `InsufficientData` naming the id column(s). A missing value of `y_col` or of an
/// `x_cols` column in a kept row is an error, as in [`run_ols`].
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
///
/// This is [`OlsClusteredResult::fit`] with an intercept.
pub fn run_ols_clustered(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    cluster1: &str,
    cluster2: Option<&str>,
) -> EconResult<OlsClusteredResult> {
    OlsClusteredResult::fit(dataset, y_col, x_cols, cluster1, cluster2, true)
}

impl OlsClusteredResult {
    /// OLS with one-way (`cluster2 = None`) or two-way cluster-robust CR1 standard errors, as
    /// `sandwich::vcovCL(cluster = ~c1 [+ c2], type = "HC1")`. `intercept = false` fits the
    /// model through the origin (k counts the slope columns only; R², adjusted R² and F
    /// uncentered, as `summary.lm`). Missing cluster ids are handled as in
    /// [`run_ols_clustered`], which is `fit(.., true)`.
    pub fn fit(
        dataset: &Dataset,
        y_col: &str,
        x_cols: &[&str],
        cluster1: &str,
        cluster2: Option<&str>,
        intercept: bool,
    ) -> EconResult<OlsClusteredResult> {
        // Cluster ids, with the rows whose id is missing left out.
        let mut request = ColumnRequest::new(dataset.df()).factor(cluster1);
        let context = match cluster2 {
            Some(c2) => {
                request = request.factor(c2);
                format!("clustered OLS (rows with ids in '{cluster1}' and '{c2}')")
            }
            None => format!("clustered OLS (rows with an id in '{cluster1}')"),
        };
        let ids = request.na(NaPolicy::Omit).min_rows(1, &context).extract()?;

        let mut names: Vec<&str> = Vec::with_capacity(x_cols.len() + 3);
        names.push(y_col);
        names.extend_from_slice(x_cols);
        names.push(cluster1);
        names.extend(cluster2);
        let sample = estimation_sample(dataset, &names, &ids)?;
        let df = sample.df();

        // First run standard OLS to get coefficients and residuals
        let mut base_result = run_ols(&sample, y_col, x_cols, intercept, CovarianceType::Standard)?;

        // Build design matrix again (we need it for clustered SEs); the sandwich uses the
        // identified columns and the fit's (X'X)^{-1} block, as sandwich::vcovCL does.
        let design = DesignMatrix::from_dataframe(df, x_cols, intercept)
            .map_err(|e| econ_error_from_design(e, df))?;
        let id = IdentifiedDesign::of(design.data.view(), &base_result)?;

        // Row indices (into the sample) of each cluster
        let factor1 = ids.factor(cluster1);
        let clusters1 = cluster_groups(factor1)?;
        let n_clusters_1 = clusters1.len();

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
            let factor2 = ids.factor(c2);
            let clusters2 = cluster_groups(factor2)?;
            let n_c2 = clusters2.len();

            if n_c2 < 2 {
                return Err(EconError::InsufficientData {
                    required: 2,
                    provided: n_c2,
                    context: format!("Need at least 2 clusters for '{}'", c2),
                });
            }

            let intersection = intersection_groups(factor1, factor2);
            let vcov = compute_twoway_clustered_vcov(
                &id.x.view(),
                &base_result.resid,
                &id.unscaled,
                &clusters1,
                &clusters2,
                &intersection,
            )?;

            (
                id.embed(&vcov),
                Some(n_c2),
                format!("Two-way clustered ({}, {})", cluster1, c2),
            )
        } else {
            // One-way clustering
            let vcov =
                compute_clustered_vcov(&id.x.view(), &base_result.resid, &id.unscaled, &clusters1)?;

            (id.embed(&vcov), None, format!("Clustered ({})", cluster1))
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
        if let Some(note) = ids.missing().note() {
            base_result.missing = ids.missing().clone();
            base_result.warnings.insert(0, note);
        }

        Ok(OlsClusteredResult {
            ols: base_result,
            cluster_type,
            n_clusters_1,
            n_clusters_2,
            cluster_var_1: cluster1.to_string(),
            cluster_var_2: cluster2.map(|s| s.to_string()),
        })
    }
}

/// The rows of `dataset` that `ids` kept, restricted to the columns `names` (each must
/// exist; duplicates are taken once). Without a dropped row this is `dataset` itself, so
/// the estimates on complete ids are computed exactly as before.
fn estimation_sample<'a>(
    dataset: &'a Dataset,
    names: &[&str],
    ids: &Columns,
) -> EconResult<Cow<'a, Dataset>> {
    if ids.missing().is_empty() {
        return Ok(Cow::Borrowed(dataset));
    }
    let df = dataset.df();
    let mut seen: HashSet<&str> = HashSet::with_capacity(names.len());
    let mut columns: Vec<&str> = Vec::with_capacity(names.len());
    for &name in names {
        crate::data::columns::column(df, name)?;
        if seen.insert(name) {
            columns.push(name);
        }
    }
    let mut idx: Vec<IdxSize> = resource::try_with_capacity(ids.n(), "estimation sample")?;
    for &row in ids.rows() {
        // A polars frame's height fits its index type.
        idx.push(IdxSize::try_from(row).map_err(|_| {
            EconError::Internal(format!("row {row} exceeds the polars row index type"))
        })?);
    }
    let sample = df
        .select(columns)
        .and_then(|selected| selected.take(&IdxCa::new("rows".into(), idx)))
        .map_err(|e| EconError::Internal(e.to_string()))?;
    Ok(Cow::Owned(Dataset::new(sample)))
}

/// Row indices of each level of `f` (indices into the rows `f` was extracted from).
fn cluster_groups(f: &Factor) -> EconResult<Vec<Vec<usize>>> {
    let mut groups: Vec<Vec<usize>> =
        resource::try_vec(f.n_levels(), Vec::new(), "cluster groups")?;
    for (i, &code) in f.codes.iter().enumerate() {
        groups[code as usize].push(i);
    }
    Ok(groups)
}

/// Row indices of each observed (level of `a`, level of `b`) pair: the intersection
/// clusters of a two-way clustering, keyed by factor codes.
fn intersection_groups(a: &Factor, b: &Factor) -> Vec<Vec<usize>> {
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, (&ca, &cb)) in a.codes.iter().zip(&b.codes).enumerate() {
        let g = *index.entry((ca, cb)).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    groups
}

/// Compute one-way clustered variance-covariance matrix.
///
/// Uses the cluster-robust sandwich estimator with small-sample correction:
/// V = (G / (G-1)) * ((n-1)/(n-k)) * (X'X)^{-1} * B * (X'X)^{-1}
///
/// `x` holds the identified columns only, so k is the rank of the fit (sandwich's CR1).
///
/// where B = sum_g (X_g' e_g) (X_g' e_g)'
fn compute_clustered_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    clusters: &[Vec<usize>],
) -> EconResult<Array2<f64>> {
    let n = x.nrows();
    let k = x.ncols();
    let g = clusters.len();

    // Small sample correction
    let correction = (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / (n - k) as f64);

    // Compute meat: sum over clusters of (X_g' e_g)(X_g' e_g)'
    let meat: Array2<f64> = clusters
        .par_iter()
        .map(|indices| {
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
    clusters1: &[Vec<usize>],
    clusters2: &[Vec<usize>],
    intersection: &[Vec<usize>],
) -> EconResult<Array2<f64>> {
    // Compute V_1 (clustered on dimension 1)
    let v1 = compute_clustered_vcov(x, residuals, xtx_inv, clusters1)?;

    // Compute V_2 (clustered on dimension 2)
    let v2 = compute_clustered_vcov(x, residuals, xtx_inv, clusters2)?;

    // Compute V_12 (clustered on the intersection)
    let v12 = compute_clustered_vcov(x, residuals, xtx_inv, intersection)?;

    // V = V_1 + V_2 - V_12
    Ok(&v1 + &v2 - &v12)
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
    /// Residual degrees of freedom of the fit, n - rank (for t tests on the HAC SEs).
    #[serde(default)]
    pub df_resid: usize,
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

/// Accepted spellings of the HAC kernel (matched after `options::normalize`, so "Quadratic
/// Spectral", "quadratic-spectral" and "QS" all parse): sandwich's names and common
/// abbreviations. The first spelling of each entry is canonical.
const HAC_KERNEL_CHOICES: Choices<HacKernel> = Choices {
    option: "kernel",
    values: &[
        (
            &["bartlett", "newey_west", "neweywest", "nw"],
            HacKernel::Bartlett,
        ),
        (&["parzen"], HacKernel::Parzen),
        (
            &["quadratic_spectral", "qs", "quadraticspectral"],
            HacKernel::QuadraticSpectral,
        ),
        (&["truncated", "rectangular"], HacKernel::Truncated),
        (
            &["tukey_hanning", "tukeyhanning", "tukey", "th"],
            HacKernel::TukeyHanning,
        ),
    ],
};

// FromStr (unknown names are an error listing the accepted ones) and choices(); the
// Display above ("Bartlett (Newey-West)") is printed by MCP and the CLI and stays.
crate::option_enum!(HacKernel, HAC_KERNEL_CHOICES, no_display);

impl HacKernel {
    /// Parse kernel type from string; `None` for an unknown name.
    #[deprecated(note = "use str::parse, which reports unknown names")]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
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

    // The identified columns and the fit's (X'X)^{-1} block (sandwich drops aliased
    // columns from estfun and bread).
    let id = IdentifiedDesign::of(x.view(), ols_result)?;
    let full_k = k;
    let k = id.rank();
    let x = &id.x;
    let xtx_inv = &id.unscaled;

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
    let vcov_identified = &vcov_hac * (n as f64);

    // Extract standard errors (square root of diagonal); NaN for aliased columns
    let identified_se: Vec<f64> = (0..k)
        .map(|i| {
            let var = vcov_identified[[i, i]];
            if var >= 0.0 { var.sqrt() } else { 0.0 }
        })
        .collect();
    let std_errors = id.embed_vec(&identified_se);
    let vcov_scaled = id.embed(&vcov_identified);
    let k = full_k;

    // Convert vcov to Vec<Vec<f64>> for output
    let vcov_vec: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_scaled[[i, j]]).collect())
        .collect();
    warnings.extend(alias_notes(ols_result));

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
        df_resid: ols_result.df_resid,
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
/// * `kernel` - Kernel name: `bartlett` (also `Newey-West`, `nw`), `parzen`,
///   `quadratic_spectral` (`Quadratic Spectral`, `qs`), `truncated` (`rectangular`),
///   `tukey_hanning` (`Tukey-Hanning`, `tukey`); case, spaces and dashes are ignored. `None`
///   is Bartlett; an unknown name is `InvalidSpecification` listing the accepted ones.
/// * `prewhiten` - Whether to use prewhitening
///
/// # Returns
///
/// `HacResult` with HAC-adjusted standard errors.
///
/// R equivalent: `sandwich::NeweyWest(lm(y ~ x), lag = bandwidth, prewhite = prewhiten)`
///
/// This is [`HacResult::fit`] with an intercept.
pub fn run_vcov_hac(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    bandwidth: Option<usize>,
    kernel: Option<&str>,
    prewhiten: bool,
) -> EconResult<HacResult> {
    HacResult::fit(dataset, y_col, x_cols, bandwidth, kernel, prewhiten, true)
}

impl HacResult {
    /// OLS with HAC (Newey-West family) standard errors, as [`run_vcov_hac`], with or without an
    /// intercept (`intercept = false`: `NeweyWest(lm(y ~ 0 + x), ..)`). [`run_vcov_hac`] is
    /// `fit(.., true)`.
    pub fn fit(
        dataset: &Dataset,
        y_col: &str,
        x_cols: &[&str],
        bandwidth: Option<usize>,
        kernel: Option<&str>,
        prewhiten: bool,
        intercept: bool,
    ) -> EconResult<HacResult> {
        // First run OLS
        let ols_result = run_ols(dataset, y_col, x_cols, intercept, CovarianceType::Standard)?;

        // Build design matrix
        let dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, intercept)?;
        let x = dm.view().to_owned();

        // Parse kernel (unknown names are an error)
        let kernel_type = crate::options::parse_or(kernel, HacKernel::Bartlett)?;

        vcov_hac(&ols_result, &x, bandwidth, kernel_type, prewhiten)
    }
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
    /// Wild bootstrap: Multiply residuals by random weights (Rademacher, ±1 with
    /// probability 1/2; sandwich's `type = "wild"` default).
    /// Robust to heteroskedasticity, preserves X structure.
    Wild,
    /// Wild bootstrap with Mammen's two-point weights (sandwich `type = "mammen"`):
    /// -(sqrt5 - 1)/2 with probability (sqrt5 + 1)/(2 sqrt5), else (sqrt5 + 1)/2.
    WildMammen,
    /// Wild bootstrap with Webb's six-point weights (sandwich `type = "webb"`):
    /// ±sqrt(1/2), ±1, ±sqrt(3/2), each with probability 1/6.
    WildWebb,
}

/// Accepted spellings of the bootstrap type (sandwich's `vcovBS` names; `norm`,
/// `jackknife` and `fractional` are not implemented and are refused).
const BOOTSTRAP_TYPE_CHOICES: Choices<BootstrapType> = Choices {
    option: "bootstrap_type",
    values: &[
        (&["pairs", "xy", "case", "cases"], BootstrapType::Pairs),
        (&["residual", "resid", "residuals"], BootstrapType::Residual),
        (
            &["wild", "rademacher", "wild_rademacher"],
            BootstrapType::Wild,
        ),
        (&["mammen", "wild_mammen"], BootstrapType::WildMammen),
        (&["webb", "wild_webb"], BootstrapType::WildWebb),
    ],
};

crate::option_enum!(BootstrapType, BOOTSTRAP_TYPE_CHOICES, no_display);

impl BootstrapType {
    /// Parse from string; `None` for an unknown name.
    #[deprecated(note = "use str::parse, which reports unknown names")]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// One wild-bootstrap weight (sandwich::vcovBS laws). Rademacher keeps the original
/// `rng.r#gen::<bool>()` draw, so seeded `Wild` results are unchanged; Mammen draws one
/// uniform against (sqrt5 + 1)/(2 sqrt5); Webb one of six points uniformly.
fn wild_weight(kind: BootstrapType, rng: &mut ChaCha8Rng) -> f64 {
    match kind {
        BootstrapType::WildMammen => {
            let s5 = 5f64.sqrt();
            if rng.r#gen::<f64>() < (s5 + 1.0) / (2.0 * s5) {
                -(s5 - 1.0) / 2.0
            } else {
                (s5 + 1.0) / 2.0
            }
        }
        BootstrapType::WildWebb => {
            const WEBB: [f64; 6] = [
                -1.224_744_871_391_589,
                -1.0,
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                1.0,
                1.224_744_871_391_589,
            ];
            WEBB[rng.gen_range(0..6)]
        }
        _ => {
            if rng.r#gen::<bool>() {
                1.0
            } else {
                -1.0
            }
        }
    }
}

impl std::fmt::Display for BootstrapType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootstrapType::Pairs => write!(f, "Pairs (xy)"),
            BootstrapType::Residual => write!(f, "Residual"),
            BootstrapType::Wild => write!(f, "Wild"),
            BootstrapType::WildMammen => write!(f, "Wild (Mammen)"),
            BootstrapType::WildWebb => write!(f, "Wild (Webb)"),
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
    /// Warnings: R's aliasing note of the fit, and how many pairs replicates were skipped
    /// because the resampled design lost rank. Absent from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
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
        f.write_str(&notes_text(self))
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

    // The identified columns (aliased ones are left out of every replicate and get NaN)
    let id = IdentifiedDesign::of(x.view(), ols_result)?;
    let r = id.rank();
    let x1 = id.x.view();

    // Original fitted values (never X b: NaN for aliased columns) and residuals
    let residuals = ols_result.residuals();
    let y_fitted: Array1<f64> = y - residuals;

    // Residual and wild bootstraps keep X fixed: one factorisation, then O(n r) per
    // replicate. The pairs bootstrap refits each resampled design.
    let fixed = match bootstrap_type {
        BootstrapType::Pairs => None,
        _ => Some(FixedDesignSolver::new(x1)?),
    };

    // Storage for bootstrap coefficient estimates
    let mut boot_coefs: Vec<Vec<f64>> = resource::try_with_capacity(replications, what)?;
    let indices: Vec<usize> = (0..n).collect();
    let mut rank_deficient = 0usize;

    for _ in 0..replications {
        resource::checkpoint()?;
        let beta_boot = match (bootstrap_type, &fixed) {
            (BootstrapType::Pairs, _) | (_, None) => {
                // Pairs bootstrap: resample (y_i, x_i) pairs together
                let sample: Vec<usize> =
                    (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();

                let y_boot: Array1<f64> = sample.iter().map(|&i| y[i]).collect();
                let x_boot: Array2<f64> =
                    Array2::from_shape_fn((n, r), |(i, j)| x1[[sample[i], j]]);
                let fit = solve_ols(x_boot.view(), y_boot.view())?;
                if fit.rank < r {
                    // A resample that loses a column identifies no full coefficient vector
                    rank_deficient += 1;
                    continue;
                }
                fit.beta
            }
            (BootstrapType::Residual, Some(solver)) => {
                // Residual bootstrap: y* = ŷ + ε*, ε* resampled from the residuals
                let y_boot: Array1<f64> = (0..n)
                    .map(|i| y_fitted[i] + residuals[*indices.choose(&mut rng).unwrap()])
                    .collect();
                solver.coefficients(x1, &y_boot)?
            }
            (
                BootstrapType::Wild | BootstrapType::WildMammen | BootstrapType::WildWebb,
                Some(solver),
            ) => {
                // Wild bootstrap: residuals times Rademacher, Mammen or Webb weights
                let weights: Vec<f64> = (0..n)
                    .map(|_| wild_weight(bootstrap_type, &mut rng))
                    .collect();

                let y_boot: Array1<f64> = (0..n)
                    .map(|i| y_fitted[i] + weights[i] * residuals[i])
                    .collect();
                solver.coefficients(x1, &y_boot)?
            }
        };
        boot_coefs.push(beta_boot.to_vec());
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

    // Compute bootstrap covariance matrix on the identified columns
    // Cov(β) = (1/(B-1)) * Σᵢ (β*ᵢ - β̄*)(β*ᵢ - β̄*)'
    let boot_mean: Vec<f64> = (0..r)
        .map(|j| boot_coefs.iter().map(|b| b[j]).sum::<f64>() / successful_reps as f64)
        .collect();

    let mut vcov_r = Array2::<f64>::zeros((r, r));
    for boot in &boot_coefs {
        for i in 0..r {
            for j in 0..r {
                vcov_r[[i, j]] += (boot[i] - boot_mean[i]) * (boot[j] - boot_mean[j]);
            }
        }
    }

    // Normalize by (B-1)
    let divisor = (successful_reps - 1) as f64;
    vcov_r /= divisor;

    // Standard errors from diagonal; NaN rows, columns and SEs for aliased columns
    let se_r: Vec<f64> = (0..r).map(|i| vcov_r[[i, i]].max(0.0).sqrt()).collect();
    let std_errors = id.embed_vec(&se_r);
    let vcov_k = id.embed(&vcov_r);
    let vcov: Vec<Vec<f64>> = vcov_k.rows().into_iter().map(|row| row.to_vec()).collect();
    let boot_samples: Vec<Vec<f64>> = boot_coefs.iter().map(|b| id.embed_vec(b)).collect();

    // Extract original coefficients and names
    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result
        .coefficients
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let mut warnings = alias_notes(ols_result);
    if rank_deficient > 0 {
        warnings.push(format!(
            "{rank_deficient} of {replications} bootstrap replicates had a rank-deficient \
             design and were skipped"
        ));
    }

    Ok(BootstrapResult {
        vcov,
        std_errors,
        n_boot: successful_reps,
        bootstrap_type,
        coefficients,
        names,
        n_obs: n,
        n_params: k,
        boot_samples: Some(boot_samples),
        convergence_rate: successful_reps as f64 / replications as f64,
        warnings,
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
/// * `bootstrap_type` - `pairs` (also `xy`, `case`), `residual`, `wild` (Rademacher), `mammen`,
///   `webb` (sandwich's names). `None` is pairs; an unknown name (also sandwich's `norm`,
///   `jackknife`, `fractional`, which are not implemented) is `InvalidSpecification`.
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

    // X and y as run_ols reads them (integer and boolean columns are numbers; run_ols
    // has already refused missing values)
    let df = dataset.df();
    let dm = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| econ_error_from_design(e, df))?;
    let x = dm.data;
    let y = DesignMatrix::extract_column(df, y_col).map_err(|e| econ_error_from_design(e, df))?;

    // Parse bootstrap type (unknown names are an error)
    let boot_type = crate::options::parse_or(bootstrap_type, BootstrapType::Pairs)?;

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
    /// Rows dropped because their time id was missing ([`run_vcov_driscoll_kraay`]);
    /// absent from JSON when nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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

    // The period-aggregated scores have rank at most T: the t tests use T - rank df and
    // need T > rank (REG-51; the subtraction underflowed before).
    let rank = ols_result.rank();
    let df_periods = n_periods
        .checked_sub(rank)
        .filter(|d| *d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: rank + 1,
            provided: n_periods,
            context: format!(
                "Driscoll-Kraay standard errors need more time periods than identified \
                 coefficients (k = {rank}): the period-aggregated scores have rank at most T"
            ),
        })?;

    // The identified columns and the fit's (X'X)^{-1} block (aliased columns: NaN).
    let id = IdentifiedDesign::of(x.view(), ols_result)?;
    let full_k = k;
    let k = id.rank();
    let x = &id.x;
    let xtx_inv = &id.unscaled;

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

    // Extract standard errors and compute statistics; NaN for aliased columns
    let identified_se: Vec<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
    let std_errors = id.embed_vec(&identified_se);
    let vcov = id.embed(&vcov);
    let k = full_k;

    let coefficients: Vec<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let names: Vec<String> = ols_result
        .coefficients
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let df = df_periods as f64;
    // Aliased columns: NaN (the se > 0 rule would turn them into t = 0, p = 1)
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se.is_nan() {
                f64::NAN
            } else if se > 0.0 {
                b / se
            } else {
                0.0
            }
        })
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
        warnings: {
            let mut w = warnings;
            w.extend(alias_notes(ols_result));
            w
        },
        missing: MissingSummary::none(n),
    })
}

/// Convenience function to compute Driscoll-Kraay standard errors for a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `time_col` - Name of the time period identifier column: whole numbers (integers or
///   integral floats) or dates. A fractional or text id is an `InvalidSpecification`; a row
///   whose id is missing is dropped, the model is re-estimated on the other rows, and the
///   drop is reported in `DriscollKraayResult::missing` and as the first warning (no row
///   with an id is `InsufficientData` naming the column).
/// * `bandwidth` - Optional bandwidth (None for automatic). Must be smaller than the number of
///   time periods T; one above T/4 adds a warning (see [`vcov_driscoll_kraay`]).
/// * `kernel` - Kernel name, as for [`run_vcov_hac`] (`None` is Bartlett; an unknown name is
///   `InvalidSpecification`)
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
    // Time ids: whole numbers (integers or integral floats) or dates. A fractional or
    // text id is an error; a row whose id is missing is dropped.
    let ids = ColumnRequest::new(dataset.df())
        .time_index(time_col)
        .na(NaPolicy::Omit)
        .min_rows(
            1,
            &format!("Driscoll-Kraay (rows with a time id in '{time_col}')"),
        )
        .extract()?;
    let mut names: Vec<&str> = Vec::with_capacity(x_cols.len() + 2);
    names.push(y_col);
    names.extend_from_slice(x_cols);
    names.push(time_col);
    let sample = estimation_sample(dataset, &names, &ids)?;
    let df = sample.df();

    // Run OLS on the estimation sample
    let ols_result = run_ols(&sample, y_col, x_cols, true, CovarianceType::Standard)?;

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| econ_error_from_design(e, df))?;

    // Parse kernel (unknown names are an error)
    let kernel_type = crate::options::parse_or(kernel, HacKernel::Bartlett)?;

    let mut result = vcov_driscoll_kraay(
        &ols_result,
        &dm.data,
        ids.time_index(time_col),
        bandwidth,
        kernel_type,
    )?;
    if let Some(note) = ids.missing().note() {
        result.missing = ids.missing().clone();
        result.warnings.insert(0, note);
    }
    Ok(result)
}

/// Fixtures for the missing-value and column-type tests of the regression helpers
/// (T2-WP15), built from the R case datasets. `complete_rows` filters with polars
/// directly, independently of `data::columns`, so a test compares the extractor's
/// estimation sample against a sample built another way.
#[cfg(test)]
pub(crate) mod na_fixtures {
    use crate::data::Dataset;
    use crate::data::columns::{NaPolicy, numeric_column, numeric_options};
    use polars::prelude::*;

    /// The CSV at `rel` under the repository root (`validation/datasets/...`), loaded as
    /// `p2a data load` loads it, for the property tests that compare no R value and so
    /// load no R case.
    pub(crate) fn repo_dataset(rel: &str) -> Dataset {
        let path = p2a_validation::repo_path(rel);
        match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        }
    }

    /// `validation/datasets/generated/mtcars.csv` (`hp` is Int64, `model` is text).
    pub(crate) fn mtcars() -> Dataset {
        repo_dataset("validation/datasets/generated/mtcars.csv")
    }

    /// `validation/datasets/generated/smoothers_na.csv`: 60 rows, 2 missing x, 2 missing y.
    pub(crate) fn smoothers_na() -> Dataset {
        repo_dataset("validation/datasets/generated/smoothers_na.csv")
    }

    /// `ds` with column `name` cast to Float64 (the same values).
    pub(crate) fn as_float(ds: &Dataset, name: &str) -> Dataset {
        let mut df = ds.df().clone();
        let cast = df.column(name).unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(cast).unwrap();
        Dataset::new(df)
    }

    /// The rows of `ds` where every column in `names` is not null.
    pub(crate) fn complete_rows(ds: &Dataset, names: &[&str]) -> Dataset {
        let df = ds.df();
        let mut mask = df.column(names[0]).unwrap().is_not_null();
        for name in &names[1..] {
            mask = &mask & &df.column(name).unwrap().is_not_null();
        }
        Dataset::new(df.filter(&mask).unwrap())
    }

    /// Column `name` as f64 with missing cells as NaN (the array APIs' input).
    pub(crate) fn nan_vec(ds: &Dataset, name: &str) -> Vec<f64> {
        numeric_options(ds.df(), name)
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect()
    }

    /// `ds` plus a column `name` of type `dtype` whose every cell is missing.
    pub(crate) fn with_all_null(ds: &Dataset, name: &str, dtype: DataType) -> Dataset {
        let mut df = ds.df().clone();
        let n = df.height();
        df.with_column(Series::full_null(name.into(), n, &dtype))
            .unwrap();
        Dataset::new(df)
    }

    /// The context of an `InsufficientData` error (panics on any other result).
    pub(crate) fn insufficient_context<T: std::fmt::Debug>(
        r: crate::errors::EconResult<T>,
    ) -> String {
        match r {
            Err(crate::errors::EconError::InsufficientData {
                required: 1,
                provided: 0,
                context,
            }) => context,
            other => panic!("expected InsufficientData (1 required, 0 provided), got {other:?}"),
        }
    }

    /// `ds` plus a Float64 column `name` holding `f(source)` row by row.
    pub(crate) fn with_derived(
        ds: &Dataset,
        name: &str,
        source: &str,
        f: impl Fn(f64) -> f64,
    ) -> Dataset {
        let (values, _) = numeric_column(ds.df(), source, NaPolicy::Fail).unwrap();
        let derived: Vec<f64> = values.into_iter().map(f).collect();
        let mut df = ds.df().clone();
        df.with_column(Series::new(name.into(), derived)).unwrap();
        Dataset::new(df)
    }
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
    #[allow(deprecated)]
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
    #[allow(deprecated)]
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

    /// Case ols_no_intercept_grunfeld: lm(inv ~ 0 + value + capital) (validation/r/cases/cli_flags.R). REG-19.
    #[test]
    fn test_validate_ols_no_intercept_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_no_intercept_grunfeld");
        let n = case.scalar("n_obs") as usize;
        let x = ["value", "capital"];
        let r = run_ols(&ds, "inv", &x, false, CovarianceType::Standard).unwrap();
        let names: Vec<&str> = r.coefficients.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, x);
        let col = |r: &OlsResult, f: fn(&OlsCoefficient) -> f64| {
            r.coefficients.iter().map(f).collect::<Vec<f64>>()
        };
        let hc1 = run_ols(&ds, "inv", &x, false, CovarianceType::HC1).unwrap();
        Check::new(&case)
            .named("coef", &names, &col(&r, |c| c.estimate), Tol::coef(n))
            .named(
                "se_classical",
                &names,
                &col(&r, |c| c.std_error),
                Tol::se(n),
            )
            .named("t_classical", &names, &col(&r, |c| c.t_value), Tol::stat(n))
            .named(
                "p_classical",
                &names,
                &col(&r, |c| c.p_value),
                Tol::p_value(n),
            )
            .named("se_hc1", &names, &col(&hc1, |c| c.std_error), Tol::se(n))
            .named("p_hc1", &names, &col(&hc1, |c| c.p_value), Tol::p_value(n))
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .scalar("sigma", r.residual_std_error, Tol::stat(n))
            .scalar("f_stat", r.f_statistic, Tol::stat(n))
            .scalar("f_df1", r.df_model as f64, Tol::exact())
            .scalar("f_df2", r.df_resid as f64, Tol::exact())
            .scalar("f_p", r.f_p_value, Tol::p_value(n))
            .scalar("log_lik", r.log_likelihood, Tol::stat(n))
            .scalar("aic", r.aic, Tol::stat(n))
            .finish();
    }

    /// REG-19 (b): the intercept is found in any column of a raw design (case ols_grunfeld).
    #[test]
    fn test_validate_ols_raw_intercept_in_last_column() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let n = case.scalar("n_obs") as usize;
        let d = DesignMatrix::from_dataframe(ds.df(), &["value", "capital"], false).unwrap();
        let y = DesignMatrix::extract_column(ds.df(), "inv").unwrap();
        let mut x = Array2::<f64>::ones((n, 3));
        x.slice_mut(ndarray::s![.., 0..2]).assign(&d.data);
        let names = ["value", "capital", "(Intercept)"]
            .map(String::from)
            .to_vec();
        let r = run_ols_raw(&x, &y, &names, "inv", CovarianceType::Standard).unwrap();
        Check::new(&case)
            .scalar("f_stat", r.f_statistic, Tol::stat(n))
            .scalar("f_df1", r.df_model as f64, Tol::exact())
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .finish();
    }

    /// CR1 one- and two-way without intercept: sandwich::vcovCL(lm(inv ~ 0 + value + capital), ~firm [+ year], HC1).
    #[test]
    fn test_validate_ols_clustered_no_intercept_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_no_intercept_grunfeld");
        let n = case.scalar("n_obs") as usize;
        for (c2, key) in [
            (None, "cluster_firm_hc1"),
            (Some("year"), "cluster_firm_year_hc1"),
        ] {
            let r = OlsClusteredResult::fit(&ds, "inv", &["value", "capital"], "firm", c2, false)
                .unwrap();
            let names: Vec<&str> = r.ols.coefficients.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(names, ["value", "capital"]);
            assert_eq!((r.n_clusters_1, r.n_clusters_2), (10, c2.map(|_| 20)));
            let se: Vec<f64> = r.ols.coefficients.iter().map(|c| c.std_error).collect();
            let p: Vec<f64> = r.ols.coefficients.iter().map(|c| c.p_value).collect();
            let b: Vec<f64> = r.ols.coefficients.iter().map(|c| c.estimate).collect();
            Check::new(&case)
                .named("coef", &names, &b, Tol::coef(n))
                .named(&format!("se_{key}"), &names, &se, Tol::se(n))
                .named(&format!("p_{key}"), &names, &p, Tol::p_value(n))
                .scalar("r2", r.ols.r_squared, Tol::stat(n))
                .finish();
        }
    }

    /// NeweyWest(lm(inv ~ 0 + value + capital), lag = 4, prewhite = FALSE) = p2a Bartlett bandwidth 4 (T1 hac_ar1).
    #[test]
    fn test_validate_hac_no_intercept_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_no_intercept_grunfeld");
        let n = case.scalar("n_obs") as usize;
        let h = HacResult::fit(
            &ds,
            "inv",
            &["value", "capital"],
            Some(4),
            Some("bartlett"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(h.names, ["value", "capital"]);
        assert_eq!((h.bandwidth, h.n_params), (4, 2));
        Check::new(&case)
            .named("coef", &h.names, &h.coefficients, Tol::coef(n))
            .named("se_nw_lag4", &h.names, &h.std_errors, Tol::se(n))
            .finish();
    }

    /// run_ols_clustered and run_vcov_hac are fit(.., intercept = true) (case ols_grunfeld).
    #[test]
    fn test_validate_clustered_and_hac_wrappers_keep_the_intercept() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let x = ["value", "capital"];
        let a = run_ols_clustered(&ds, "inv", &x, "firm", Some("year")).unwrap();
        let b = OlsClusteredResult::fit(&ds, "inv", &x, "firm", Some("year"), true).unwrap();
        let se = |r: &OlsClusteredResult| {
            r.ols
                .coefficients
                .iter()
                .map(|c| c.std_error)
                .collect::<Vec<f64>>()
        };
        // Equal up to rounding: the cluster meat is a parallel (rayon) sum, whose order of
        // additions, and so its last bit, can differ between two calls.
        for (sa, sb) in se(&a).iter().zip(se(&b)) {
            assert!(Tol::new(1e-12, 0.0).allows(*sa, sb), "{sa} vs {sb}");
        }
        let names: Vec<&str> = a.ols.coefficients.iter().map(|c| c.name.as_str()).collect();
        Check::new(&case)
            .named("se_cluster_firm_year_hc1", &names, &se(&a), Tol::se(200))
            .finish();
        let h = run_vcov_hac(&ds, "inv", &x, Some(4), Some("bartlett"), false).unwrap();
        let g = HacResult::fit(&ds, "inv", &x, Some(4), Some("bartlett"), false, true).unwrap();
        assert_eq!((h.std_errors, h.names), (g.std_errors, g.names));
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

    /// y and x1 of the REG-32 regressions: noisy, with `x1[2]` replaced by `x1_2`.
    fn reg32_dataset(x1_2: f64) -> Dataset {
        Dataset::new(
            df! {
                "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 6.3],
                "x1" => [1.0, 2.0, x1_2, 4.0, 5.0, 6.0],
            }
            .unwrap(),
        )
    }

    #[test]
    fn test_reg32_ols_nan_regressor_reported_as_missing() {
        match run_ols(
            &reg32_dataset(f64::NAN),
            "y",
            &["x1"],
            true,
            CovarianceType::HC1,
        ) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "x1");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues for x1, got {other:?}"),
        }
    }

    #[test]
    fn test_reg32_ols_infinite_regressor_reported() {
        let err = run_ols(
            &reg32_dataset(f64::INFINITY),
            "y",
            &["x1"],
            true,
            CovarianceType::HC1,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Column 'x1' contains 1 infinite value(s); remove or recode them before estimating."
        );
    }

    #[test]
    fn test_reg32_ols_missing_y_and_text_regressor() {
        let ds = Dataset::new(
            df! {
                "y" => [f64::NAN, 1.9, 3.2, 3.8, 5.1, 6.3],
                "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x2" => ["a", "b", "a", "c", "b", "a"],
            }
            .unwrap(),
        );
        match run_ols(&ds, "y", &["x1"], true, CovarianceType::HC1) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "y");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues for y, got {other:?}"),
        }
        match run_ols(&ds, "x1", &["x2"], true, CovarianceType::HC1) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "x2"),
            other => panic!("expected NonNumericColumn for x2, got {other:?}"),
        }
    }

    /// The reg_cluster_na case as polars reads it: the dtypes and missing cells the
    /// missing-id tests rely on (3 empty firm labels, 2 empty years), and R's row counts.
    #[test]
    fn test_validate_reg_cluster_na_fixture() {
        let (case, ds) = crate::test_support::load_case("reg_cluster_na");
        let df = ds.df();
        assert_eq!(df.height(), 200);
        for (c, t) in [
            ("firm", DataType::Int64),
            ("year", DataType::Int64),
            ("year_na", DataType::Int64),
            ("firm_label", DataType::String),
            ("inv", DataType::Float64),
        ] {
            assert_eq!(df.column(c).unwrap().dtype(), &t, "{c}");
        }
        let fl = df.column("firm_label").unwrap().is_not_null();
        let yn = df.column("year_na").unwrap().is_not_null();
        let (n_fl, n_yn) = (fl.sum().unwrap() as usize, yn.sum().unwrap() as usize);
        assert_eq!((200 - n_fl, 200 - n_yn), (3, 2));
        let both = (&fl & &yn).sum().unwrap() as usize;
        Check::new(&case)
            .scalar("n_used", n_fl as f64, Tol::exact())
            .scalar("n_dropped", (200 - n_fl) as f64, Tol::exact())
            .scalar("n_used_twoway", both as f64, Tol::exact())
            .finish();
    }

    /// REG-21: rows with a missing cluster id are dropped and the model is re-estimated
    /// on the rest, one- and two-way; the CR1 SEs then equal lm + vcovCL(type = "HC1")
    /// on those rows (and fixest's one-way SEs).
    #[test]
    fn test_validate_ols_cluster_na() {
        use crate::data::columns::MissingSummary;
        let (case, ds) = crate::test_support::load_case("reg_cluster_na");
        let x = ["value", "capital"];
        let one = run_ols_clustered(&ds, "inv", &x, "firm_label", None).unwrap();
        let two = run_ols_clustered(&ds, "inv", &x, "firm_label", Some("year_na")).unwrap();
        assert_eq!((one.n_clusters_1, one.n_clusters_2), (10, None));
        assert_eq!((two.n_clusters_1, two.n_clusters_2), (10, Some(20)));
        assert_eq!(
            one.ols.missing,
            MissingSummary {
                n_total: 200,
                n_used: 197,
                n_dropped: 3,
                by_column: vec![("firm_label".into(), 3)],
            }
        );
        assert_eq!(
            two.ols.missing,
            MissingSummary {
                n_total: 200,
                n_used: 195,
                n_dropped: 5,
                by_column: vec![("firm_label".into(), 3), ("year_na".into(), 2)],
            }
        );
        assert_eq!(
            one.ols.warnings[0],
            "3 of 200 rows dropped because of missing values (firm_label: 3)"
        );
        assert_eq!(
            two.ols.warnings[0],
            "5 of 200 rows dropped because of missing values (firm_label: 3, year_na: 2)"
        );
        let cols = |r: &OlsClusteredResult| -> (Vec<String>, Vec<f64>, Vec<f64>) {
            (
                r.ols.coefficients.iter().map(|c| c.name.clone()).collect(),
                r.ols.coefficients.iter().map(|c| c.estimate).collect(),
                r.ols.coefficients.iter().map(|c| c.std_error).collect(),
            )
        };
        let (n1, b1, s1) = cols(&one);
        let (n2, b2, s2) = cols(&two);
        Check::new(&case)
            .named("coef", &n1, &b1, Tol::coef(197))
            .named("se_cluster_hc1", &n1, &s1, Tol::se(197))
            .named("se_cluster_fixest", &n1, &s1, Tol::se(197))
            .scalar("n_used", one.ols.n_obs as f64, Tol::exact())
            .scalar("n_dropped", one.ols.missing.n_dropped as f64, Tol::exact())
            .named("coef_twoway", &n2, &b2, Tol::coef(195))
            .named("se_cluster_twoway_hc1", &n2, &s2, Tol::se(195))
            .scalar("n_used_twoway", two.ols.n_obs as f64, Tol::exact())
            .finish();
        // The note is printed once, in the Warnings block.
        let text = one.to_string();
        assert_eq!(
            text.matches("rows dropped because of missing values")
                .count(),
            1
        );
        assert!(text.contains("Observations: 197\n"), "{text}");
        // No missing id: every row is used and nothing is reported (the Tier 1 path).
        let full = run_ols_clustered(&ds, "inv", &x, "firm", None).unwrap();
        assert_eq!((full.ols.n_obs, full.n_clusters_1), (200, 10));
        assert!(full.ols.missing.is_empty() && !full.to_string().contains("dropped"));
        assert!(
            serde_json::to_value(&full)
                .unwrap()
                .get("missing")
                .is_none()
        );
        // ... and the JSON of a fit that dropped rows has the object.
        assert_eq!(
            serde_json::to_value(&one).unwrap()["missing"],
            serde_json::json!({"n_total": 200, "n_used": 197, "n_dropped": 3,
                               "by_column": {"firm_label": 3}})
        );
        // An absent id column.
        match run_ols_clustered(&ds, "inv", &x, "nope", None).unwrap_err() {
            EconError::ColumnNotFound { column, available } => {
                assert_eq!(column, "nope");
                assert_eq!(available.len(), 7);
            }
            e => panic!("{e:?}"),
        }
        // No row has an id: the error names the method and the id column(s).
        use super::na_fixtures::{insufficient_context, with_all_null};
        let none = with_all_null(&ds, "no_id", DataType::String);
        assert_eq!(
            insufficient_context(run_ols_clustered(&none, "inv", &x, "no_id", None)),
            "clustered OLS (rows with an id in 'no_id'): 0 of 200 rows have no missing value"
        );
        assert_eq!(
            insufficient_context(run_ols_clustered(&none, "inv", &x, "firm", Some("no_id"))),
            "clustered OLS (rows with ids in 'firm' and 'no_id'): \
             0 of 200 rows have no missing value"
        );
    }

    /// REG-21 (Driscoll-Kraay): a missing time id drops its row and the model is
    /// re-estimated; integral float ids equal the Int64 ids; fractional and text ids are
    /// refused instead of truncated. A property test: the DK SEs are not R-pinned
    /// (REG-24), so it compares no R value and loads the dataset without a case; it pins
    /// the estimation sample against the same estimator on the complete rows.
    #[test]
    fn test_na_driscoll_kraay_time_ids() {
        use super::na_fixtures::{complete_rows, repo_dataset, with_derived};
        use crate::data::columns::MissingSummary;
        let ds = repo_dataset("validation/datasets/generated/grunfeld_cluster_na.csv");
        let x = ["value", "capital"];
        let r = run_vcov_driscoll_kraay(&ds, "inv", &x, "year_na", None, None).unwrap();
        assert_eq!((r.n_obs, r.n_periods), (198, 20));
        assert_eq!(
            r.missing,
            MissingSummary {
                n_total: 200,
                n_used: 198,
                n_dropped: 2,
                by_column: vec![("year_na".into(), 2)],
            }
        );
        assert_eq!(
            r.warnings[0],
            "2 of 200 rows dropped because of missing values (year_na: 2)"
        );
        assert_eq!(
            r.to_string()
                .matches("rows dropped because of missing values")
                .count(),
            1
        );
        let complete = complete_rows(&ds, &["year_na"]);
        let c = run_vcov_driscoll_kraay(&complete, "inv", &x, "year_na", None, None).unwrap();
        assert_eq!(
            (&r.coefficients, &r.std_errors, r.bandwidth),
            (&c.coefficients, &c.std_errors, c.bandwidth)
        );
        assert!(c.missing.is_empty() && c.warnings.iter().all(|w| !w.contains("dropped")));
        assert!(serde_json::to_value(&c).unwrap().get("missing").is_none());
        // Integral floats are accepted and equal the Int64 ids.
        let f = with_derived(&ds, "t_f", "year", |v| v);
        assert_eq!(
            run_vcov_driscoll_kraay(&f, "inv", &x, "t_f", None, None)
                .unwrap()
                .std_errors,
            run_vcov_driscoll_kraay(&ds, "inv", &x, "year", None, None)
                .unwrap()
                .std_errors
        );
        // Fractional ids are refused, not truncated (1935.25 used to become period 1935).
        let q = with_derived(&ds, "t_q", "year", |v| v + 0.25);
        match run_vcov_driscoll_kraay(&q, "inv", &x, "t_q", None, None).unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("Time column 't_q' must hold whole numbers")
                    && message.contains("1935.25"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
        // Text ids.
        match run_vcov_driscoll_kraay(&ds, "inv", &x, "firm_label", None, None).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("Time column 'firm_label'"), "{message}")
            }
            e => panic!("{e:?}"),
        }
        // An absent time column.
        assert!(matches!(
            run_vcov_driscoll_kraay(&ds, "inv", &x, "nope", None, None),
            Err(EconError::ColumnNotFound { .. })
        ));
        // No row has a time id: the error names the method and the time column.
        use super::na_fixtures::{insufficient_context, with_all_null};
        let none = with_all_null(&ds, "no_t", DataType::Int64);
        assert_eq!(
            insufficient_context(run_vcov_driscoll_kraay(
                &none, "inv", &x, "no_t", None, None
            )),
            "Driscoll-Kraay (rows with a time id in 'no_t'): 0 of 200 rows have no missing value"
        );
    }

    /// REG-23: the bootstrap reads y as `run_ols` does, so an Int64 outcome (mtcars hp)
    /// gives exactly the bootstrap of the same values as Float64 (same seed). A property
    /// test: no value is compared with R (the RNGs differ), so it loads the dataset
    /// without a case and is not registered as R-pinned.
    #[test]
    fn test_na_vcov_bootstrap_integer_outcome() {
        use super::na_fixtures::{as_float, mtcars};
        let ds = mtcars();
        assert_eq!(ds.df().column("hp").unwrap().dtype(), &DataType::Int64);
        let a = run_vcov_bootstrap(&ds, "hp", &["wt"], Some(200), Some("pairs"), Some(42)).unwrap();
        let b = run_vcov_bootstrap(
            &as_float(&ds, "hp"),
            "hp",
            &["wt"],
            Some(200),
            Some("pairs"),
            Some(42),
        )
        .unwrap();
        assert_eq!(
            (a.n_obs, &a.std_errors, &a.coefficients),
            (32, &b.std_errors, &b.coefficients)
        );
        let ols = run_ols(&ds, "hp", &["wt"], true, CovarianceType::Standard).unwrap();
        assert_eq!(
            a.coefficients,
            ols.coefficients
                .iter()
                .map(|c| c.estimate)
                .collect::<Vec<_>>()
        );
        match run_vcov_bootstrap(&ds, "model", &["wt"], Some(200), None, Some(1)).unwrap_err() {
            EconError::NonNumericColumn { column } => assert_eq!(column, "model"),
            e => panic!("{e:?}"),
        }
        assert!(matches!(
            run_vcov_bootstrap(&ds, "nope", &["wt"], Some(200), None, Some(1)),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    // ════════════════════════════════════════════════════════════════════════════
    // T3-WP09: rank-deficient and ill-conditioned designs (validation/r/cases/reg_rank_qr.R;
    // registry validation/methods/reg_rank_qr.json)
    // ════════════════════════════════════════════════════════════════════════════

    /// (names, values) of the identified coefficients: R prints only these in
    /// coef(fit)[!is.na(coef(fit))].
    fn identified(r: &OlsResult, f: impl Fn(&OlsCoefficient) -> f64) -> (Vec<String>, Vec<f64>) {
        r.coefficients
            .iter()
            .filter(|c| c.estimate.is_finite())
            .map(|c| (c.name.clone(), f(c)))
            .unzip()
    }

    fn all_named(r: &OlsResult, f: impl Fn(&OlsCoefficient) -> f64) -> (Vec<String>, Vec<f64>) {
        r.coefficients
            .iter()
            .map(|c| (c.name.clone(), f(c)))
            .unzip()
    }

    fn alias_flags(r: &OlsResult) -> Vec<f64> {
        r.coefficients
            .iter()
            .map(|c| if c.estimate.is_nan() { 1.0 } else { 0.0 })
            .collect()
    }

    const ALIAS_MSG: &str = "Coefficients not defined because of singularities";

    /// R: lm(y ~ x1 + da + db + dc) on ols_aliased (reg_rank_qr.R). dc is aliased; the
    /// identified columns, df = n - rank, summary.lm, logLik (rank + 1 parameters) and
    /// vcovHC(HC1) as R.
    #[test]
    fn test_validate_ols_aliased_dummy_trap() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let x = ["x1", "da", "db", "dc"];
        let r = run_ols(&ds, "y", &x, true, CovarianceType::Standard).unwrap();
        let hc1 = run_ols(&ds, "y", &x, true, CovarianceType::HC1).unwrap();
        let n = r.n_obs;
        let (cn, cv) = identified(&r, |c| c.estimate);
        let (sn, sv) = all_named(&r, |c| c.std_error); // dc: NaN, R: NA (null) -> equal
        let (hn, hv) = identified(&hc1, |c| c.std_error);
        Check::new(&case)
            .named("trap_coef", &cn, &cv, Tol::coef(n))
            .vector("trap_aliased", &alias_flags(&r), Tol::exact())
            .named("trap_se", &sn, &sv, Tol::se(n))
            .scalar("trap_rank", r.rank() as f64, Tol::exact())
            .scalar("trap_df", r.df_resid as f64, Tol::exact())
            .scalar("trap_r2", r.r_squared, Tol::stat(n))
            .scalar("trap_adj_r2", r.adj_r_squared, Tol::stat(n))
            .scalar("trap_sigma", r.residual_std_error, Tol::stat(n))
            .scalar("trap_fstat", r.f_statistic, Tol::stat(n))
            .vector(
                "trap_fstat_df",
                &[r.df_model as f64, r.df_resid as f64],
                Tol::exact(),
            )
            .scalar("trap_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("trap_aic", r.aic, Tol::stat(n))
            .scalar("trap_bic", r.bic, Tol::stat(n))
            .named("trap_hc1_se", &hn, &hv, Tol::se(n))
            .finish();
        assert_eq!(r.aliased, vec!["dc".to_string()]);
        assert_eq!((r.n_params, r.rank()), (5, 4));
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains(ALIAS_MSG) && w.contains("dc")),
            "{:?}",
            r.warnings
        );
        let dc = &r.coefficients[4];
        assert!(
            dc.estimate.is_nan()
                && dc.std_error.is_nan()
                && dc.t_value.is_nan()
                && dc.p_value.is_nan()
        );
        assert!(dc.ci_lower_95.is_nan() && r.vcov_matrix().row(4).iter().all(|v| v.is_nan()));
        assert!(r.vcov_matrix()[[1, 1]].is_finite() && r.coefficients()[4].is_nan());
        assert!(r.to_string().contains(ALIAS_MSG));
    }

    /// R: lm(y ~ x1 + x2 + x3), x2 = 2 x1 up to 1e-10: x2 (the later column) is aliased,
    /// x3 is kept. Also the HC0/HC1/HC3 sandwiches on the identified columns, and
    /// run_ols_raw on the same arrays.
    #[test]
    fn test_validate_ols_aliased_duplicate() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let x = ["x1", "x2", "x3"];
        let fit = |cov| run_ols(&ds, "y", &x, true, cov).unwrap();
        let (r, h0, h1, h3) = (
            fit(CovarianceType::Standard),
            fit(CovarianceType::HC0),
            fit(CovarianceType::HC1),
            fit(CovarianceType::HC3),
        );
        let n = r.n_obs;
        let (cn, cv) = identified(&r, |c| c.estimate);
        let (sn, sv) = all_named(&r, |c| c.std_error);
        let se = |m: &OlsResult| identified(m, |c| c.std_error);
        let ((n0, v0), (n1, v1), (n3, v3)) = (se(&h0), se(&h1), se(&h3));
        Check::new(&case)
            .named("dup_coef", &cn, &cv, Tol::coef(n))
            .vector("dup_aliased", &alias_flags(&r), Tol::exact())
            .named("dup_se", &sn, &sv, Tol::se(n))
            .scalar("dup_rank", r.rank() as f64, Tol::exact())
            .scalar("dup_df", r.df_resid as f64, Tol::exact())
            .scalar("dup_r2", r.r_squared, Tol::stat(n))
            .scalar("dup_sigma", r.residual_std_error, Tol::stat(n))
            .scalar("dup_fstat", r.f_statistic, Tol::stat(n))
            .named("dup_hc0_se", &n0, &v0, Tol::se(n))
            .named("dup_hc1_se", &n1, &v1, Tol::se(n))
            .named("dup_hc3_se", &n3, &v3, Tol::se(n))
            .finish();
        assert_eq!(r.aliased, vec!["x2".to_string()]);
        // run_ols_raw: same design, same numbers (one solve path)
        let dm = DesignMatrix::from_dataframe(ds.df(), &x, true).unwrap();
        let y = DesignMatrix::extract_column(ds.df(), "y").unwrap();
        let raw = run_ols_raw(&dm.data, &y, &dm.column_names, "y", CovarianceType::HC1).unwrap();
        assert_eq!(raw.aliased, h1.aliased);
        for (a, b) in raw.coefficients.iter().zip(&h1.coefficients) {
            assert!(
                (a.estimate.is_nan() && b.estimate.is_nan())
                    || (a.estimate == b.estimate && a.std_error == b.std_error)
            );
        }
    }

    /// R: lm(y ~ x1 + k5) with k5 = 5: the constant is aliased with the intercept.
    #[test]
    fn test_validate_ols_aliased_constant() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let r = run_ols(&ds, "y", &["x1", "k5"], true, CovarianceType::Standard).unwrap();
        let (cn, cv) = identified(&r, |c| c.estimate);
        Check::new(&case)
            .named("const_coef", &cn, &cv, Tol::coef(r.n_obs))
            .vector("const_aliased", &alias_flags(&r), Tol::exact())
            .finish();
        assert_eq!(r.aliased, vec!["k5".to_string()]);
    }

    /// R: lm on kappa(X) = 1.68e8, at the README tolerance (Tol::coef(80) = 1e-6). The
    /// normal equations miss by 2.8e-5 here, so the design must take the QR.
    #[test]
    fn test_validate_ols_illcond_qr() {
        let (case, ds) = crate::test_support::load_case("ols_illcond");
        let r = run_ols(
            &ds,
            "y",
            &["x1", "x2", "x3"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let n = r.n_obs;
        let (cn, cv) = all_named(&r, |c| c.estimate);
        let (sn, sv) = all_named(&r, |c| c.std_error);
        Check::new(&case)
            .named("coef", &cn, &cv, Tol::coef(n))
            .named("se", &sn, &sv, Tol::se(n))
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .scalar("sigma", r.residual_std_error, Tol::stat(n))
            .scalar("fstat", r.f_statistic, Tol::stat(n))
            .scalar("loglik", r.log_likelihood, Tol::stat(n))
            .scalar("rank", r.rank() as f64, Tol::exact())
            .finish();
        // lm warns about nothing
        assert!(
            r.aliased.is_empty() && r.warnings.is_empty(),
            "{:?}",
            r.warnings
        );
    }

    /// REG-18 on T1's Longley case: an accuracy check at 1e-10 (lm is 3.5e-14 from NIST's
    /// certified values, and any Householder QR meets 1e-10 with a wide margin), not a path
    /// pin. Before: 3.56e-8 (the normal equations).
    #[test]
    fn test_validate_ols_longley_qr() {
        let (case, ds) = crate::test_support::load_case("ols_longley");
        let x = [
            "GNP.deflator",
            "GNP",
            "Unemployed",
            "Armed.Forces",
            "Population",
            "Year",
        ];
        let r = run_ols(&ds, "Employed", &x, true, CovarianceType::Standard).unwrap();
        let (cn, cv) = all_named(&r, |c| c.estimate);
        let (sn, sv) = all_named(&r, |c| c.std_error);
        Check::new(&case)
            .named("coef", &cn, &cv, Tol::new(1e-10, 1e-12))
            .named("se_classical", &sn, &sv, Tol::new(1e-10, 1e-14))
            .finish();
    }

    /// Variant B of D1: a well-conditioned design takes the normal equations (today's
    /// numbers, bit for bit), every design whose equilibrated X'X has a reciprocal condition
    /// below 1e-8 or that is rank-deficient takes the pivoted QR. Each pinned design is
    /// checked against its R case on the path it takes.
    #[test]
    fn test_validate_ols_solve_paths() {
        let path_of = |ds: &Dataset, y: &str, x: &[&str]| {
            let d = DesignMatrix::from_dataframe(ds.df(), x, true).unwrap();
            let yv = DesignMatrix::extract_column(ds.df(), y).unwrap();
            solve_ols(d.data.view(), yv.view()).unwrap().path
        };
        let (grunfeld, g) = crate::test_support::load_case("ols_grunfeld");
        assert_eq!(
            path_of(&g, "inv", &["value", "capital"]),
            OlsPath::NormalEquations
        );
        let r = run_ols(
            &g,
            "inv",
            &["value", "capital"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let (cn, cv) = all_named(&r, |c| c.estimate);
        Check::new(&grunfeld)
            .named("coef", &cn, &cv, Tol::coef(r.n_obs))
            .finish();
        let (_, ill) = crate::test_support::load_case("ols_illcond");
        assert_eq!(path_of(&ill, "y", &["x1", "x2", "x3"]), OlsPath::PivotedQr);
        let (_, lo) = crate::test_support::load_case("ols_longley");
        let lx = [
            "GNP.deflator",
            "GNP",
            "Unemployed",
            "Armed.Forces",
            "Population",
            "Year",
        ];
        assert_eq!(path_of(&lo, "Employed", &lx), OlsPath::PivotedQr);
        let (_, al) = crate::test_support::load_case("ols_aliased");
        assert_eq!(
            path_of(&al, "y", &["x1", "da", "db", "dc"]),
            OlsPath::PivotedQr
        );
        assert_eq!(path_of(&al, "y", &["x1", "x3"]), OlsPath::NormalEquations);
    }

    /// Two columns, no intercept, cosine c between them: the equilibrated X'X is
    /// [[1, c], [c, 1]] with reciprocal condition (1 - c) / (1 + c). Just inside the
    /// threshold (about 1e-7) the fit takes the normal equations; just past it (about
    /// 3e-9) the QR. Both agree with each other's solve to the accuracy the conditioning
    /// allows.
    #[test]
    fn test_ols_solve_path_threshold() {
        let n = 50;
        let base: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.7).sin()).collect();
        let wiggle: Vec<f64> = (0..n).map(|i| (i as f64 * 1.9).cos()).collect();
        let design = |eps: f64| {
            let mut x = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                x[[i, 0]] = base[i];
                x[[i, 1]] = base[i] + eps * wiggle[i];
            }
            x
        };
        let rcond = |x: &Array2<f64>| {
            let a = x.t().dot(x);
            let c = a[[0, 1]] / (a[[0, 0]] * a[[1, 1]]).sqrt();
            (1.0 - c) / (1.0 + c)
        };
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * base[i] + 0.1 * ((i * 7) % 5) as f64)
            .collect();
        let y = Array1::from(y);
        let (inside, past) = (design(1.1e-3), design(1.9e-4));
        let (ri, rp) = (rcond(&inside), rcond(&past));
        assert!(ri > 3e-8 && ri < 3e-7, "{ri}");
        assert!(rp > 1e-9 && rp < 5e-9, "{rp}");
        let fi = solve_ols(inside.view(), y.view()).unwrap();
        let fp = solve_ols(past.view(), y.view()).unwrap();
        assert_eq!(fi.path, OlsPath::NormalEquations);
        assert_eq!(fp.path, OlsPath::PivotedQr);
        assert_eq!((fi.rank, fp.rank), (2, 2));
        let qi = crate::linalg::solve::least_squares(inside.view(), y.view()).unwrap();
        for (a, b) in fi.beta.iter().zip(&qi.coefficients) {
            assert!((a - b).abs() <= 1e-6 * b.abs(), "{a} vs {b}");
        }
    }

    #[test]
    fn test_ols_rank_zero_is_an_error() {
        let df = df! {
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 5.9],
            "z" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        .unwrap();
        match run_ols(
            &Dataset::new(df),
            "y",
            &["z"],
            false,
            CovarianceType::Standard,
        )
        .unwrap_err()
        {
            EconError::SingularMatrix { context, .. } => {
                assert!(context.contains("rank 0"), "{context}")
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn test_ols_raw_rejects_non_finite() {
        let x = ndarray::array![[1.0, 0.3], [1.0, 1.1], [1.0, 2.2], [1.0, 2.9], [1.0, 4.2]];
        let y = ndarray::array![1.0, f64::NAN, 3.1, 3.9, 5.2];
        let names = vec!["(Intercept)".to_string(), "x".to_string()];
        match run_ols_raw(&x, &y, &names, "y", CovarianceType::Standard).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("non-finite"), "{message}")
            }
            e => panic!("{e:?}"),
        }
    }

    /// sandwich::vcovCL(cluster = ~cl, type = "HC1") and NeweyWest(lag = 4, prewhite =
    /// FALSE) on lm(y ~ x1 + x2 + x3): estfun and bread drop the aliased x2.
    #[test]
    fn test_validate_ols_aliased_sandwich() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let x = ["x1", "x2", "x3"];
        let cl = run_ols_clustered(&ds, "y", &x, "cl", None).unwrap();
        let hac = run_vcov_hac(&ds, "y", &x, Some(4), Some("bartlett"), false).unwrap();
        let n = cl.ols.n_obs;
        let (cn, cv) = identified(&cl.ols, |c| c.std_error);
        let (hn, hv): (Vec<String>, Vec<f64>) = hac
            .names
            .iter()
            .zip(&hac.std_errors)
            .zip(&hac.coefficients)
            .filter(|(_, b)| b.is_finite())
            .map(|((nm, s), _)| (nm.clone(), *s))
            .unzip();
        // compute_clustered_vcov sums with a rayon reduce: compared with a tolerance
        Check::new(&case)
            .named("dup_cl_hc1_se", &cn, &cv, Tol::se(n))
            .named("dup_nw_lag4_se", &hn, &hv, Tol::se(n))
            .finish();
        assert!(cl.ols.coefficients[2].std_error.is_nan() && hac.std_errors[2].is_nan());
        assert_eq!((cl.n_clusters_1, hac.df_resid), (12, 117));
        assert!(
            cl.ols.warnings.iter().any(|w| w.contains(ALIAS_MSG))
                && hac.warnings.iter().any(|w| w.contains(ALIAS_MSG))
        );
    }

    /// Property test (not R-pinned: sandwich::vcovBS uses R's RNG). On the aliased design the
    /// bootstrap resamples the identified columns: x2's SE is NaN (sandwich: NA), and the
    /// Rademacher wild SEs are within 10% of R's HC0 (E[vcov_wild] = HC0 exactly; at B = 2000
    /// the Monte Carlo sd is 1.6%, so 10% is 6 sd; seeded).
    #[test]
    fn test_validate_bootstrap_aliased_design_uses_identified_columns() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let b = run_vcov_bootstrap(
            &ds,
            "y",
            &["x1", "x2", "x3"],
            Some(2000),
            Some("wild"),
            Some(20261118),
        )
        .unwrap();
        let hc0 = case.vector("dup_hc0_se");
        assert!(b.std_errors[2].is_nan() && b.coefficients[2].is_nan());
        for (s, h) in [b.std_errors[0], b.std_errors[1], b.std_errors[3]]
            .iter()
            .zip(&hc0)
        {
            assert!((s / h - 1.0).abs() < 0.10, "se {s} vs HC0 {h}");
        }
        assert_eq!(b.convergence_rate, 1.0);
        assert!(b.warnings.iter().any(|w| w.contains(ALIAS_MSG)));
        assert!(b.to_string().contains(ALIAS_MSG));
    }

    /// The residual and wild bootstraps solve every replicate on one factorisation of the
    /// identified design; each solve equals a fresh least-squares refit (dup design of
    /// ols_aliased, five seeded wild responses).
    #[test]
    fn test_validate_bootstrap_fixed_design_solve_matches_refit() {
        let (_, ds) = crate::test_support::load_case("ols_aliased");
        let ols = run_ols(
            &ds,
            "y",
            &["x1", "x2", "x3"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let dm = DesignMatrix::from_dataframe(ds.df(), &["x1", "x2", "x3"], true).unwrap();
        let id = IdentifiedDesign::of(dm.data.view(), &ols).unwrap();
        assert_eq!(id.cols, vec![0, 1, 3]);
        let solver = FixedDesignSolver::new(id.x.view()).unwrap();
        let fitted = DesignMatrix::extract_column(ds.df(), "y").unwrap() - &ols.resid;
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for _ in 0..5 {
            let y_star: Array1<f64> = fitted
                .iter()
                .zip(&ols.resid)
                .map(|(f, e)| f + if rng.r#gen::<bool>() { *e } else { -*e })
                .collect();
            let b = solver.coefficients(id.x.view(), &y_star).unwrap();
            let refit = least_squares(id.x.view(), y_star.view()).unwrap();
            for (a, c) in b.iter().zip(&refit.coefficients) {
                assert!((a - c).abs() <= 1e-12 * c.abs().max(1e-3), "{a} vs {c}");
            }
        }
    }

    /// Panel helper (Tasks 3 and 4 of T3-WP09): 20 units, noisy and non-collinear unless
    /// `alias` (x4 = 2 x1).
    fn short_panel(t_periods: usize, alias: bool) -> Dataset {
        let (mut unit, mut time, mut x1, mut x2, mut x3, mut y) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for i in 0..20usize {
            for t in 0..t_periods {
                let s = (i * t_periods + t) as f64;
                let (a, b, c) = (
                    (s * 0.37).sin(),
                    (s * 1.13).cos(),
                    ((i * 7 + t * 3) % 11) as f64 / 11.0,
                );
                unit.push(i as i64);
                time.push(2000 + t as i64);
                x1.push(a);
                x2.push(b);
                x3.push(c);
                y.push(1.0 + 0.5 * a - 0.3 * b + 0.8 * c + 0.1 * (s * 2.71).sin());
            }
        }
        let x4: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        let mut df =
            df! { "unit" => unit, "time" => time, "x1" => x1, "x2" => x2, "x3" => x3, "y" => y }
                .unwrap();
        if alias {
            df.with_column(Series::new("x4".into(), x4)).unwrap();
        }
        Dataset::new(df)
    }

    /// Driscoll-Kraay on an aliased panel: NaN for the aliased column, finite elsewhere (DK
    /// SEs are not R-pinned: REG-24, Tier 4).
    #[test]
    fn test_driscoll_kraay_aliased_column_is_nan() {
        let ds = short_panel(12, true);
        let dk = run_vcov_driscoll_kraay(&ds, "y", &["x1", "x2", "x3", "x4"], "time", None, None)
            .unwrap();
        assert!(dk.std_errors[4].is_nan() && dk.p_values[4].is_nan());
        assert!(
            dk.std_errors[..4]
                .iter()
                .chain(&dk.p_values[..4])
                .all(|v| v.is_finite())
        );
        assert!(dk.warnings.iter().any(|w| w.contains(ALIAS_MSG)));
    }

    /// k = 4 (intercept + x1, x2, x3): T = 3 underflowed n_periods - k (debug panic, release
    /// p = 1 for every coefficient); T = 4 gave df 0 and NaN p-values. T must exceed the rank.
    #[test]
    fn test_driscoll_kraay_rejects_t_le_k() {
        for t in [3usize, 4] {
            match run_vcov_driscoll_kraay(
                &short_panel(t, false),
                "y",
                &["x1", "x2", "x3"],
                "time",
                None,
                None,
            )
            .unwrap_err()
            {
                EconError::InsufficientData {
                    required,
                    provided,
                    context,
                } => {
                    assert_eq!((required, provided), (5, t));
                    assert!(context.contains("time periods"), "{context}");
                }
                e => panic!("T = {t}: {e:?}"),
            }
        }
        let ok = run_vcov_driscoll_kraay(
            &short_panel(5, false),
            "y",
            &["x1", "x2", "x3"],
            "time",
            None,
            None,
        )
        .unwrap();
        assert!(
            ok.p_values
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        );
    }

    #[test]
    fn test_hac_kernel_parses_sandwich_names() {
        use HacKernel::*;
        HacKernel::choices().validate().unwrap();
        for (s, k) in [
            ("Quadratic Spectral", QuadraticSpectral),
            ("quadratic-spectral", QuadraticSpectral),
            ("QS", QuadraticSpectral),
            ("Tukey-Hanning", TukeyHanning),
            ("tukey_hanning", TukeyHanning),
            ("tukey", TukeyHanning),
            ("Bartlett", Bartlett),
            ("Newey-West", Bartlett),
            ("nw", Bartlett),
            ("Parzen", Parzen),
            ("Truncated", Truncated),
            ("rectangular", Truncated),
        ] {
            assert_eq!(s.parse::<HacKernel>().unwrap(), k, "{s}");
        }
        match "qs2".parse::<HacKernel>().unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("kernel")
                    && message.contains("'qs2'")
                    && message.contains("quadratic_spectral"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
        let ts = create_timeseries_dataset();
        let qs =
            run_vcov_hac(&ts, "y", &["x"], Some(4), Some("Quadratic Spectral"), false).unwrap();
        assert_eq!(qs.kernel, QuadraticSpectral); // before: Bartlett
        assert!(run_vcov_hac(&ts, "y", &["x"], Some(4), Some("qs2"), false).is_err()); // before: Ok (Bartlett)
        let p = create_panel_dataset();
        assert_eq!(
            run_vcov_driscoll_kraay(&p, "y", &["x"], "time", None, Some("Tukey-Hanning"))
                .unwrap()
                .kernel,
            TukeyHanning
        );
        assert!(run_vcov_driscoll_kraay(&p, "y", &["x"], "time", None, Some("bogus")).is_err());
    }

    #[test]
    fn test_bootstrap_type_parses_aliases_and_rejects_unknown() {
        use BootstrapType::*;
        BootstrapType::choices().validate().unwrap();
        for (s, t) in [
            ("pairs", Pairs),
            ("xy", Pairs),
            ("case", Pairs),
            ("Residual", Residual),
            ("wild", Wild),
            ("rademacher", Wild),
            ("wild-rademacher", Wild),
            ("Mammen", WildMammen),
            ("wild-mammen", WildMammen),
            ("webb", WildWebb),
            ("wild_webb", WildWebb),
        ] {
            assert_eq!(s.parse::<BootstrapType>().unwrap(), t, "{s}");
        }
        for bad in ["norm", "jackknife", "mamen"] {
            match bad.parse::<BootstrapType>().unwrap_err() {
                EconError::InvalidSpecification { message } => assert!(
                    message.contains("bootstrap_type") && message.contains("mammen"),
                    "{message}"
                ),
                e => panic!("{e:?}"),
            }
        }
        let ds = create_test_dataset();
        let m = run_vcov_bootstrap(&ds, "y", &["x1"], Some(200), Some("mammen"), Some(42)).unwrap();
        assert_eq!(m.bootstrap_type, WildMammen); // before: Pairs
        assert!(run_vcov_bootstrap(&ds, "y", &["x1"], Some(200), Some("norm"), Some(42)).is_err());
    }

    /// The weight laws of sandwich::vcovBS (sandwich 3.1.3 source): Mammen -(sqrt5 - 1)/2
    /// w.p. (sqrt5 + 1)/(2 sqrt5), else (sqrt5 + 1)/2; Webb ±sqrt(1/2), ±1, ±sqrt(3/2) w.p.
    /// 1/6; Rademacher ±1. Seeded; the bounds are >= 4.5 Monte Carlo sd.
    #[test]
    fn test_wild_weights_match_sandwich() {
        let n = 200_000;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let draw = |kind, rng: &mut ChaCha8Rng| -> Vec<f64> {
            (0..n).map(|_| wild_weight(kind, rng)).collect()
        };
        let moment = |w: &[f64], p: i32| w.iter().map(|v| v.powi(p)).sum::<f64>() / n as f64;
        let m = draw(BootstrapType::WildMammen, &mut rng);
        let (lo, hi) = (-(5f64.sqrt() - 1.0) / 2.0, (5f64.sqrt() + 1.0) / 2.0);
        assert!(m.iter().all(|&w| w == lo || w == hi));
        let p_lo = m.iter().filter(|&&w| w == lo).count() as f64 / n as f64;
        assert!((p_lo - 0.723_606_797_749_978_9).abs() < 0.005, "{p_lo}");
        assert!(
            moment(&m, 1).abs() < 0.01
                && (moment(&m, 2) - 1.0).abs() < 0.01
                && (moment(&m, 3) - 1.0).abs() < 0.03
        );
        let w = draw(BootstrapType::WildWebb, &mut rng);
        let support = [
            -(1.5f64).sqrt(),
            -1.0,
            -(0.5f64).sqrt(),
            (0.5f64).sqrt(),
            1.0,
            (1.5f64).sqrt(),
        ];
        for s in support {
            let f = w.iter().filter(|&&v| v == s).count() as f64 / n as f64;
            assert!((f - 1.0 / 6.0).abs() < 0.005, "{s}: {f}");
        }
        assert!((moment(&w, 2) - 1.0).abs() < 0.01);
        let r = draw(BootstrapType::Wild, &mut rng);
        assert!(r.iter().all(|&v| v == 1.0 || v == -1.0) && moment(&r, 3).abs() < 0.01);
    }

    /// Property test as test_validate_bootstrap_aliased_design_uses_identified_columns, for
    /// the three wild laws; Mammen and Rademacher are different methods in R and must give
    /// different p2a output.
    #[test]
    fn test_validate_bootstrap_wild_variants_near_hc0() {
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let hc0 = case.vector("dup_hc0_se");
        let run = |t| {
            run_vcov_bootstrap(
                &ds,
                "y",
                &["x1", "x2", "x3"],
                Some(2000),
                Some(t),
                Some(20261118),
            )
            .unwrap()
        };
        let (rad, mam, web) = (run("wild"), run("mammen"), run("webb"));
        for b in [&rad, &mam, &web] {
            assert!(b.std_errors[2].is_nan());
            for (s, h) in [b.std_errors[0], b.std_errors[1], b.std_errors[3]]
                .iter()
                .zip(&hc0)
            {
                assert!(
                    (s / h - 1.0).abs() < 0.10,
                    "{}: se {s} vs HC0 {h}",
                    b.bootstrap_type
                );
            }
        }
        assert_ne!(rad.std_errors[1], mam.std_errors[1]);
        assert_ne!(mam.std_errors[1], web.std_errors[1]);
    }

    /// 'ö' spans bytes 11..13 of "Haushaltsgröße_netto": the byte slice &s[..12] panicked in
    /// Display, which MCP regression_ols/regression_clustered return.
    #[test]
    fn test_ols_display_non_ascii_names() {
        let name = "Haushaltsgröße_netto";
        assert!(!name.is_char_boundary(12));
        let n = 30;
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 + 0.3 * ((i * 7) % 5) as f64)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 1.0 + 0.5 * v + 0.2 * ((i * 13) % 7) as f64)
            .collect();
        let cl: Vec<i64> = (0..n as i64).map(|i| i % 5).collect();
        let long: Vec<f64> = x.iter().map(|v| v.sqrt() + 0.01 * v).collect();
        let ds = Dataset::new(
            df! { "y" => y, name => x, "a_very_long_variable_name" => long, "cl" => cl }.unwrap(),
        );
        let r = run_ols(
            &ds,
            "y",
            &[name, "a_very_long_variable_name"],
            true,
            CovarianceType::HC1,
        )
        .unwrap();
        let text = r.to_string();
        // ASCII names are cut as before
        assert!(
            text.contains("Haushaltsgrö...") && text.contains("a_very_long_..."),
            "{text}"
        );
        let c = run_ols_clustered(&ds, "y", &[name], "cl", None).unwrap();
        assert!(c.to_string().contains("Haushaltsgrö..."));
    }
}
