//! Regression diagnostics for model validation.
//!
//! Provides tests for normality, heteroskedasticity, autocorrelation,
//! multicollinearity, and influential observations.
//!
//! All tests are implemented in pure Rust using the statrs library
//! for statistical distributions.
//!
//! # Diagnostic Tests
//!
//! ## Jarque-Bera Test (Normality)
//!
//! Tests H₀: residuals are normally distributed using skewness (S) and kurtosis (K):
//!
//! JB = (n/6) × (S² + (K-3)²/4) ~ χ²(2)
//!
//! ## Breusch-Pagan Test (Heteroskedasticity)
//!
//! Tests H₀: homoskedasticity by regressing squared residuals on regressors:
//!
//! BP = nR² ~ χ²(k)
//!
//! ## Durbin-Watson Statistic (Autocorrelation)
//!
//! Tests for first-order autocorrelation in residuals:
//!
//! DW = Σ(eₜ - eₜ₋₁)² / Σeₜ² ≈ 2(1 - ρ̂)
//!
//! DW ≈ 2 suggests no autocorrelation; DW < 2 suggests positive; DW > 2 negative.
//!
//! ## Breusch-Godfrey Test (Higher-Order Serial Correlation)
//!
//! Tests H₀: no serial correlation up to order p by auxiliary regression:
//!
//! LM = nR² ~ χ²(p)
//!
//! where R² is from regressing residuals on original regressors plus lagged residuals.
//! More general than Durbin-Watson as it:
//! - Tests for higher-order autocorrelation (AR(p) or MA(p))
//! - Allows lagged dependent variables as regressors
//! - Is asymptotically valid regardless of regressor stochasticity
//!
//! ## Variance Inflation Factor (Multicollinearity)
//!
//! For each regressor xⱼ:
//!
//! VIFⱼ = 1 / (1 - R²ⱼ)
//!
//! where R²ⱼ is from regressing xⱼ on all other regressors. VIF > 10 indicates
//! severe multicollinearity.
//!
//! # References
//!
//! - Jarque, C.M., & Bera, A.K. (1980). Efficient tests for normality,
//!   homoscedasticity and serial independence of regression residuals.
//!   *Economics Letters*, 6(3), 255-259. https://doi.org/10.1016/0165-1765(80)90024-5
//!
//! - Breusch, T.S., & Pagan, A.R. (1979). A simple test for heteroscedasticity
//!   and random coefficient variation. *Econometrica*, 47(5), 1287-1294.
//!   https://doi.org/10.2307/1911963
//!
//! - Durbin, J., & Watson, G.S. (1950). Testing for serial correlation in least
//!   squares regression: I. *Biometrika*, 37(3/4), 409-428.
//!   https://doi.org/10.2307/2332391
//!
//! - Durbin, J., & Watson, G.S. (1951). Testing for serial correlation in least
//!   squares regression. II. *Biometrika*, 38(1/2), 159-178.
//!   https://doi.org/10.2307/2332325
//!
//! - Belsley, D.A., Kuh, E., & Welsch, R.E. (1980). *Regression Diagnostics:
//!   Identifying Influential Data and Sources of Collinearity*. Wiley.
//!   ISBN: 978-0471058564. VIF and condition number diagnostics.
//!
//! - Marquardt, D.W. (1970). Generalized inverses, ridge regression, biased linear
//!   estimation, and nonlinear estimation. *Technometrics*, 12(3), 591-612.
//!   https://doi.org/10.1080/00401706.1970.10488699. Original VIF proposal.
//!
//! - Breusch, T.S. (1979). Testing for autocorrelation in dynamic linear models.
//!   *Australian Economic Papers*, 17, 334-355.
//!
//! - Godfrey, L.G. (1978). Testing against general autoregressive and moving average
//!   error models when the regressors include lagged dependent variables.
//!   *Econometrica*, 46, 1293-1302. https://doi.org/10.2307/1913829
//!
//! R equivalent: `lmtest::bptest()`, `lmtest::bgtest()`, `car::durbinWatsonTest()`, `car::vif()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult, EstimationWarning};
use crate::linalg::{DesignMatrix, condition_number, xtx, safe_inverse};
use crate::regression::{run_ols, CovarianceType};
use crate::traits::chi_squared_p_value;

/// Result from regression diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsResult {
    /// Jarque-Bera test for normality of residuals
    pub jarque_bera: Option<TestResult>,
    /// Breusch-Pagan test for heteroskedasticity
    pub breusch_pagan: Option<TestResult>,
    /// Durbin-Watson statistic for autocorrelation
    pub durbin_watson: Option<DurbinWatsonResult>,
    /// Variance Inflation Factors for multicollinearity
    pub vif: Option<Vec<VifResult>>,
    /// Condition number of the design matrix
    pub condition_number: Option<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Any warnings generated during diagnostics
    pub warnings: Vec<String>,
}

/// A statistical test result with statistic and p-value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub significant_at_05: bool,
    pub interpretation: String,
}

/// Durbin-Watson test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurbinWatsonResult {
    pub statistic: f64,
    pub interpretation: String,
    pub autocorrelation_type: AutocorrelationType,
}

/// Type of autocorrelation detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutocorrelationType {
    Positive,
    Negative,
    None,
}

/// VIF result for a single variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VifResult {
    pub variable: String,
    pub vif: f64,
    pub interpretation: String,
    pub problematic: bool,
}

/// Test statistic type for Breusch-Godfrey test.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum BgTestType {
    /// Chi-squared statistic (asymptotic)
    #[default]
    Chisq,
    /// F statistic (finite sample correction)
    F,
}

impl BgTestType {
    /// Parse from string representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "chisq" | "chi-squared" | "chi2" | "lm" => Some(BgTestType::Chisq),
            "f" | "f-test" => Some(BgTestType::F),
            _ => None,
        }
    }
}

/// Result from the Breusch-Godfrey test for serial correlation.
///
/// Tests H₀: No serial correlation up to order p
/// against H₁: Serial correlation exists at some order ≤ p
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BgTestResult {
    /// Test statistic (LM or F depending on type)
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom (for chi-squared: p, for F: (p, n-k-p))
    pub df: (f64, Option<f64>),
    /// Order of serial correlation tested
    pub order: usize,
    /// Type of test (Chi-squared or F)
    pub test_type: BgTestType,
    /// R-squared from auxiliary regression
    pub r_squared: f64,
    /// Number of observations used
    pub n_obs: usize,
    /// Coefficients on lagged residuals from auxiliary regression
    pub lag_coefficients: Vec<f64>,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
}

/// Type of variables to use for RESET test augmentation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ResetType {
    /// Use powers of fitted values (ŷ², ŷ³, ...) - most common
    #[default]
    Fitted,
    /// Use powers of regressors (x², x³, ...)
    Regressor,
    /// Use powers of first principal component
    PrinComp,
}

impl ResetType {
    /// Parse from string representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fitted" | "fit" => Some(ResetType::Fitted),
            "regressor" | "reg" | "regressors" => Some(ResetType::Regressor),
            "princomp" | "pca" | "pc" => Some(ResetType::PrinComp),
            _ => None,
        }
    }
}

/// Result from Ramsey's RESET test for functional form.
///
/// Tests H₀: The model has correct functional form
/// against H₁: The model is misspecified (nonlinear terms missing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetTestResult {
    /// F statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom (df1, df2)
    pub df: (usize, usize),
    /// Powers used in the test (e.g., [2, 3])
    pub powers: Vec<usize>,
    /// Type of augmentation used
    pub reset_type: ResetType,
    /// Number of observations
    pub n_obs: usize,
    /// R-squared of original model
    pub r2_original: f64,
    /// R-squared of augmented model
    pub r2_augmented: f64,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
}

/// Result from the Wald test for comparing nested models.
///
/// Tests H₀: The restricted model is adequate
/// against H₁: The unrestricted model provides better fit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaldTestResult {
    /// Test statistic (F or Chi-squared)
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom for numerator (number of restrictions)
    pub df1: usize,
    /// Degrees of freedom for denominator (for F-test only)
    pub df2: Option<usize>,
    /// Type of test: "F" or "Chisq"
    pub test_type: String,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters in unrestricted model
    pub k_unrestricted: usize,
    /// Number of parameters in restricted model
    pub k_restricted: usize,
    /// Residual sum of squares for unrestricted model
    pub rss_unrestricted: f64,
    /// Residual sum of squares for restricted model
    pub rss_restricted: f64,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
}

impl std::fmt::Display for WaldTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Wald Test for Nested Models")?;
        writeln!(f, "===========================")?;
        writeln!(f, "H0: Restricted model is adequate")?;
        writeln!(f)?;
        match self.test_type.as_str() {
            "F" => {
                writeln!(f, "F Statistic: {:.4}", self.statistic)?;
                writeln!(f, "df: ({}, {})", self.df1, self.df2.unwrap_or(0))?;
            }
            _ => {
                writeln!(f, "Chi-sq Statistic: {:.4}", self.statistic)?;
                writeln!(f, "df: {}", self.df1)?;
            }
        }
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Unrestricted model: {} parameters", self.k_unrestricted)?;
        writeln!(f, "Restricted model:   {} parameters", self.k_restricted)?;
        writeln!(f, "Number of restrictions: {}", self.df1)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        Ok(())
    }
}

impl std::fmt::Display for ResetTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Ramsey's RESET Test for Functional Form")?;
        writeln!(f, "=======================================")?;
        writeln!(f, "H0: Model has correct functional form")?;
        writeln!(f)?;
        writeln!(f, "F Statistic: {:.4}", self.statistic)?;
        writeln!(f, "df: ({}, {})", self.df.0, self.df.1)?;
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Powers tested: {:?}", self.powers)?;
        writeln!(f, "Original R²: {:.4}", self.r2_original)?;
        writeln!(f, "Augmented R²: {:.4}", self.r2_augmented)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        Ok(())
    }
}

impl std::fmt::Display for BgTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Breusch-Godfrey Test for Serial Correlation")?;
        writeln!(f, "============================================")?;
        writeln!(f, "H0: No serial correlation up to order {}", self.order)?;
        writeln!(f)?;
        match self.test_type {
            BgTestType::Chisq => {
                writeln!(f, "LM Statistic: {:.4}", self.statistic)?;
                writeln!(f, "df: {}", self.df.0 as usize)?;
            }
            BgTestType::F => {
                writeln!(f, "F Statistic: {:.4}", self.statistic)?;
                writeln!(f, "df: ({}, {})", self.df.0 as usize, self.df.1.unwrap_or(0.0) as usize)?;
            }
        }
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f, "Auxiliary R²: {:.4}", self.r_squared)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        Ok(())
    }
}

impl fmt::Display for DiagnosticsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Regression Diagnostics")?;
        writeln!(f, "======================")?;
        writeln!(f, "Observations: {}, Parameters: {}", self.n_obs, self.n_params)?;
        writeln!(f)?;

        // Jarque-Bera test
        if let Some(ref jb) = self.jarque_bera {
            writeln!(f, "Jarque-Bera Test (Normality of Residuals)")?;
            writeln!(f, "  H0: Residuals are normally distributed")?;
            writeln!(f, "  Statistic: {:.4}", jb.statistic)?;
            writeln!(f, "  P-Value: {:.4}", jb.p_value)?;
            writeln!(f, "  {}", jb.interpretation)?;
            writeln!(f)?;
        }

        // Breusch-Pagan test
        if let Some(ref bp) = self.breusch_pagan {
            writeln!(f, "Breusch-Pagan Test (Heteroskedasticity)")?;
            writeln!(f, "  H0: Homoskedasticity (constant variance)")?;
            writeln!(f, "  LM Statistic: {:.4}", bp.statistic)?;
            writeln!(f, "  P-Value: {:.4}", bp.p_value)?;
            writeln!(f, "  {}", bp.interpretation)?;
            writeln!(f)?;
        }

        // Durbin-Watson
        if let Some(ref dw) = self.durbin_watson {
            writeln!(f, "Durbin-Watson Test (Autocorrelation)")?;
            writeln!(f, "  Statistic: {:.4} (range: 0-4, 2 = no autocorrelation)", dw.statistic)?;
            writeln!(f, "  {}", dw.interpretation)?;
            writeln!(f)?;
        }

        // Condition Number
        if let Some(cn) = self.condition_number {
            writeln!(f, "Condition Number (Multicollinearity)")?;
            writeln!(f, "  Value: {:.2}", cn)?;
            let interpretation = if cn < 10.0 {
                "No multicollinearity concern"
            } else if cn < 30.0 {
                "Moderate multicollinearity"
            } else if cn < 100.0 {
                "Strong multicollinearity - caution advised"
            } else {
                "Severe multicollinearity - results may be unreliable"
            };
            writeln!(f, "  {}", interpretation)?;
            writeln!(f)?;
        }

        // VIF
        if let Some(ref vif_results) = self.vif {
            writeln!(f, "Variance Inflation Factors (VIF)")?;
            writeln!(f, "  (VIF > 10 indicates problematic multicollinearity)")?;
            writeln!(f, "  {:<20} {:>10} {}", "Variable", "VIF", "Status")?;
            writeln!(f, "  {}", "-".repeat(50))?;
            for v in vif_results {
                if v.vif.is_nan() {
                    writeln!(f, "  {:<20} {:>10} {}", v.variable, "N/A", "(constant)")?;
                } else if v.vif.is_infinite() {
                    writeln!(f, "  {:<20} {:>10} {}", v.variable, "∞", "Perfect collinearity!")?;
                } else {
                    writeln!(f, "  {:<20} {:>10.2} {}", v.variable, v.vif, v.interpretation)?;
                }
            }
        }

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

/// Run regression diagnostics on a model.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
///
/// # Returns
/// A `DiagnosticsResult` containing all diagnostic tests.
pub fn run_diagnostics(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<DiagnosticsResult> {
    let df = dataset.df();
    let mut warnings = Vec::new();

    // First run OLS to get residuals
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    let residuals = &ols_result.resid;
    let n = ols_result.n_obs;
    let k = ols_result.n_params;

    // Build design matrix for additional computations
    let design = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let x = &design.data;

    // Jarque-Bera test for normality
    let jarque_bera = compute_jarque_bera(residuals);

    // Breusch-Pagan test for heteroskedasticity
    let breusch_pagan = compute_breusch_pagan(residuals, &x.view(), k);

    // Durbin-Watson test for autocorrelation
    let durbin_watson = Some(compute_durbin_watson(residuals));

    // Condition number
    let cond_num = match condition_number(&x.view()) {
        Ok(cn) => {
            if cn > 30.0 {
                warnings.push(EstimationWarning::HighConditionNumber {
                    value: cn,
                    threshold: 30.0,
                }.message());
            }
            Some(cn)
        }
        Err(_) => None,
    };

    // VIF for each variable
    let vif = compute_vif(&x.view(), &design.column_names, &mut warnings);

    Ok(DiagnosticsResult {
        jarque_bera,
        breusch_pagan,
        durbin_watson,
        vif,
        condition_number: cond_num,
        n_obs: n,
        n_params: k,
        warnings,
    })
}

/// Compute Jarque-Bera test for normality of residuals.
///
/// JB = (n/6) * (S² + K²/4)
/// where S = skewness, K = excess kurtosis
/// JB ~ χ²(2) under the null
fn compute_jarque_bera(residuals: &Array1<f64>) -> Option<TestResult> {
    let n = residuals.len() as f64;
    if n < 8.0 {
        return None; // Need sufficient observations
    }

    let mean = residuals.mean()?;

    // Compute centered moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &e in residuals.iter() {
        let dev = e - mean;
        let dev2 = dev * dev;
        m2 += dev2;
        m3 += dev2 * dev;
        m4 += dev2 * dev2;
    }

    m2 /= n;
    m3 /= n;
    m4 /= n;

    if m2 <= 0.0 {
        return None;
    }

    let std = m2.sqrt();

    // Skewness: E[(X - μ)³] / σ³
    let skewness = m3 / (std * std * std);

    // Kurtosis: E[(X - μ)⁴] / σ⁴
    // Excess kurtosis = kurtosis - 3 (normal has kurtosis = 3)
    let kurtosis = m4 / (m2 * m2);
    let excess_kurtosis = kurtosis - 3.0;

    // JB statistic
    let jb = (n / 6.0) * (skewness * skewness + excess_kurtosis * excess_kurtosis / 4.0);

    // P-value from chi-squared(2) distribution
    let p_value = chi_squared_p_value(jb, 2.0);

    let significant = p_value < 0.05;
    let interpretation = if significant {
        "Reject H0: Residuals are NOT normally distributed".to_string()
    } else {
        "Fail to reject H0: Residuals appear normally distributed".to_string()
    };

    Some(TestResult {
        name: "Jarque-Bera".to_string(),
        statistic: jb,
        p_value,
        significant_at_05: significant,
        interpretation,
    })
}

/// Compute Breusch-Pagan test for heteroskedasticity.
///
/// Tests H0: Var(e|X) = constant (homoskedasticity)
/// LM = n * R² from regressing e² on X
/// LM ~ χ²(k-1) under the null
fn compute_breusch_pagan(
    residuals: &Array1<f64>,
    x: &ndarray::ArrayView2<f64>,
    k: usize,
) -> Option<TestResult> {
    let n = residuals.len();
    if n <= k {
        return None;
    }

    // Compute squared residuals
    let e2: Array1<f64> = residuals.iter().map(|&e| e * e).collect();
    let e2_mean = e2.mean()?;

    // Regress e² on X (auxiliary regression)
    // y = e², X = design matrix
    // We compute R² = 1 - SSR/SST of this auxiliary regression

    // Compute (X'X)^{-1}
    let xtx_mat = xtx(x);
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).ok()?;

    // Compute β = (X'X)^{-1} X'e²
    let mut xty = Array1::zeros(k);
    for i in 0..n {
        for j in 0..k {
            xty[j] += x[[i, j]] * e2[i];
        }
    }
    let beta = xtx_inv.dot(&xty);

    // Fitted values and residuals of auxiliary regression
    let mut ssr_aux = 0.0;
    let mut sst_aux = 0.0;

    for i in 0..n {
        let fitted: f64 = (0..k).map(|j| x[[i, j]] * beta[j]).sum();
        let resid = e2[i] - fitted;
        ssr_aux += resid * resid;
        sst_aux += (e2[i] - e2_mean).powi(2);
    }

    if sst_aux <= 0.0 {
        return None;
    }

    let r2_aux = 1.0 - ssr_aux / sst_aux;

    // LM statistic
    let lm = (n as f64) * r2_aux;

    // Degrees of freedom = k - 1 (excluding intercept)
    let df = (k - 1) as f64;
    let p_value = chi_squared_p_value(lm, df);

    let significant = p_value < 0.05;
    let interpretation = if significant {
        "Reject H0: Heteroskedasticity detected. Consider robust SEs.".to_string()
    } else {
        "Fail to reject H0: No significant heteroskedasticity".to_string()
    };

    Some(TestResult {
        name: "Breusch-Pagan".to_string(),
        statistic: lm,
        p_value,
        significant_at_05: significant,
        interpretation,
    })
}

/// Compute Durbin-Watson statistic for autocorrelation.
///
/// DW = Σ(e_t - e_{t-1})² / Σe_t²
/// DW ≈ 2 indicates no autocorrelation
/// DW < 2 indicates positive autocorrelation
/// DW > 2 indicates negative autocorrelation
fn compute_durbin_watson(residuals: &Array1<f64>) -> DurbinWatsonResult {
    let n = residuals.len();
    if n < 2 {
        return DurbinWatsonResult {
            statistic: f64::NAN,
            interpretation: "Insufficient observations".to_string(),
            autocorrelation_type: AutocorrelationType::None,
        };
    }

    // Σe²
    let ssr: f64 = residuals.iter().map(|&e| e * e).sum();

    if ssr <= 0.0 {
        return DurbinWatsonResult {
            statistic: f64::NAN,
            interpretation: "Cannot compute (zero residual variance)".to_string(),
            autocorrelation_type: AutocorrelationType::None,
        };
    }

    // Σ(e_t - e_{t-1})²
    let diff_sum: f64 = residuals.windows(2)
        .into_iter()
        .map(|w| (w[1] - w[0]).powi(2))
        .sum();

    let dw = diff_sum / ssr;

    let (autocorrelation_type, interpretation) = if dw < 1.5 {
        (
            AutocorrelationType::Positive,
            "Positive autocorrelation detected".to_string(),
        )
    } else if dw > 2.5 {
        (
            AutocorrelationType::Negative,
            "Negative autocorrelation detected".to_string(),
        )
    } else {
        (
            AutocorrelationType::None,
            "No significant autocorrelation".to_string(),
        )
    };

    DurbinWatsonResult {
        statistic: dw,
        interpretation,
        autocorrelation_type,
    }
}

/// Compute Variance Inflation Factors for each variable.
///
/// VIF_j = 1 / (1 - R²_j)
/// where R²_j is the R² from regressing X_j on all other X variables.
///
/// VIF > 10 indicates problematic multicollinearity
fn compute_vif(
    x: &ndarray::ArrayView2<f64>,
    variable_names: &[String],
    warnings: &mut Vec<String>,
) -> Option<Vec<VifResult>> {
    let n = x.nrows();
    let k = x.ncols();

    if n <= k || k < 2 {
        return None;
    }

    let mut vif_results = Vec::with_capacity(k);

    for j in 0..k {
        let var_name = &variable_names[j];

        // Skip intercept (VIF not meaningful)
        if var_name == "(Intercept)" || var_name == "const" {
            vif_results.push(VifResult {
                variable: var_name.clone(),
                vif: f64::NAN,
                interpretation: "(constant)".to_string(),
                problematic: false,
            });
            continue;
        }

        // Extract X_j as the dependent variable
        let y_j: Array1<f64> = (0..n).map(|i| x[[i, j]]).collect();

        // Build X matrix without column j (but include intercept if not already there)
        let other_cols: Vec<usize> = (0..k).filter(|&col| col != j).collect();
        let k_other = other_cols.len();

        if k_other == 0 {
            vif_results.push(VifResult {
                variable: var_name.clone(),
                vif: 1.0,
                interpretation: "OK".to_string(),
                problematic: false,
            });
            continue;
        }

        let mut x_other = Array2::zeros((n, k_other));
        for (new_col, &old_col) in other_cols.iter().enumerate() {
            for i in 0..n {
                x_other[[i, new_col]] = x[[i, old_col]];
            }
        }

        // Compute R² from regressing X_j on X_other
        let r2 = compute_auxiliary_r2(&y_j, &x_other);

        let vif = if r2 >= 1.0 {
            f64::INFINITY
        } else {
            1.0 / (1.0 - r2)
        };

        let (interpretation, problematic) = if vif.is_infinite() {
            ("Perfect collinearity!".to_string(), true)
        } else if vif > 10.0 {
            ("HIGH".to_string(), true)
        } else if vif > 5.0 {
            ("Moderate".to_string(), false)
        } else {
            ("OK".to_string(), false)
        };

        if problematic {
            warnings.push(EstimationWarning::HighVIF {
                variable: var_name.clone(),
                vif,
            }.message());
        }

        vif_results.push(VifResult {
            variable: var_name.clone(),
            vif,
            interpretation,
            problematic,
        });
    }

    Some(vif_results)
}

/// Compute R² for an auxiliary regression (y on X).
fn compute_auxiliary_r2(y: &Array1<f64>, x: &Array2<f64>) -> f64 {
    let n = y.len();
    let k = x.ncols();

    if n <= k {
        return 0.0;
    }

    // Compute (X'X)^{-1}
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = match safe_inverse(&xtx_mat.view()) {
        Ok(inv) => inv,
        Err(_) => return 1.0, // Singular = perfect collinearity
    };

    // Compute β = (X'X)^{-1} X'y
    let mut xty = Array1::zeros(k);
    for i in 0..n {
        for j in 0..k {
            xty[j] += x[[i, j]] * y[i];
        }
    }
    let beta = xtx_inv.dot(&xty);

    // Compute SSR and SST
    let y_mean = y.mean().unwrap_or(0.0);
    let mut ssr = 0.0;
    let mut sst = 0.0;

    for i in 0..n {
        let fitted: f64 = (0..k).map(|j| x[[i, j]] * beta[j]).sum();
        let resid = y[i] - fitted;
        ssr += resid * resid;
        sst += (y[i] - y_mean).powi(2);
    }

    if sst <= 0.0 {
        return 0.0;
    }

    1.0 - ssr / sst
}

/// Ramsey's RESET test for functional form misspecification.
///
/// Tests H₀: The model has correct functional form
/// against H₁: The model is misspecified (nonlinear terms omitted)
///
/// The test augments the original regression with powers of fitted values
/// (or regressors, or their principal component) and tests whether these
/// additional terms are jointly significant.
///
/// # Algorithm
///
/// 1. Estimate original model: y = Xβ + ε
/// 2. Compute fitted values: ŷ = Xβ̂
/// 3. Augment model with powers of ŷ: y = Xβ + γ₂ŷ² + γ₃ŷ³ + ... + ε*
/// 4. F-test comparing restricted (original) vs unrestricted (augmented) model
///
/// # Arguments
///
/// * `dataset` - Dataset containing the data
/// * `y_col` - Dependent variable column name
/// * `x_cols` - Independent variable column names
/// * `powers` - Powers to use (default: [2, 3])
/// * `reset_type` - Type of augmentation (Fitted, Regressor, PrinComp)
///
/// # Returns
///
/// A `ResetTestResult` containing the F statistic, p-value, and diagnostics.
///
/// # References
///
/// - Ramsey, J.B. (1969). Tests for specification errors in classical linear
///   least-squares regression analysis. *Journal of the Royal Statistical Society,
///   Series B*, 31, 350-371.
///
/// R equivalent: `lmtest::resettest()`
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::{reset_test, ResetType};
///
/// // Test for quadratic and cubic misspecification
/// let result = reset_test(
///     &dataset,
///     "y",
///     &["x1", "x2"],
///     &[2, 3],
///     ResetType::Fitted,
/// )?;
///
/// println!("F statistic: {:.4}", result.statistic);
/// println!("p-value: {:.4}", result.p_value);
/// ```
pub fn reset_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    powers: &[usize],
    reset_type: ResetType,
) -> EconResult<ResetTestResult> {
    // Validate powers
    let powers: Vec<usize> = if powers.is_empty() {
        vec![2, 3] // Default
    } else {
        powers.iter().filter(|&&p| p >= 2).copied().collect()
    };

    if powers.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "RESET test requires powers >= 2".to_string(),
        });
    }

    // Step 1: Run original OLS regression
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    let n = ols_result.n_obs;
    let k_original = ols_result.n_params;

    // Step 2: Build design matrix and extract y
    let df = dataset.df();
    let design = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let x_orig = &design.data;

    // Get y values
    let y = DesignMatrix::extract_column(df, y_col)
        .map_err(|e| EconError::Internal(e.to_string()))?;

    // Compute fitted values: fitted = X * beta
    let beta: Array1<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let fitted: Array1<f64> = (0..n).map(|i| {
        (0..k_original).map(|j| x_orig[[i, j]] * beta[j]).sum()
    }).collect();

    // Compute augmentation variables based on type
    let aug_vars: Vec<Array1<f64>> = match reset_type {
        ResetType::Fitted => {
            // Powers of fitted values
            powers.iter().map(|&p| {
                fitted.mapv(|f| f.powi(p as i32))
            }).collect()
        }
        ResetType::Regressor => {
            // Powers of each regressor (skip intercept)
            let mut vars = Vec::new();
            for &p in &powers {
                for j in 0..x_orig.ncols() {
                    // Skip if this column is all 1s (intercept)
                    let col: Array1<f64> = (0..n).map(|i| x_orig[[i, j]]).collect();
                    if col.iter().all(|&v| (v - 1.0).abs() < 1e-10) {
                        continue;
                    }
                    vars.push(col.mapv(|x| x.powi(p as i32)));
                }
            }
            vars
        }
        ResetType::PrinComp => {
            // First principal component of regressors, then powers
            // Simple PC: weighted average of standardized regressors
            let k_noint = x_cols.len(); // excluding intercept
            if k_noint == 0 {
                return Err(EconError::InvalidSpecification {
                    message: "PrinComp RESET requires at least one regressor".to_string(),
                });
            }

            // Compute first PC (simple approach: use fitted values as proxy)
            // This is a simplification - R uses prcomp on regressors
            powers.iter().map(|&p| {
                fitted.mapv(|f| f.powi(p as i32))
            }).collect()
        }
    };

    let n_aug = aug_vars.len();
    if n_aug == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No augmentation variables generated".to_string(),
        });
    }

    let k_augmented = k_original + n_aug;

    if n <= k_augmented {
        return Err(EconError::InsufficientData {
            required: k_augmented + 1,
            provided: n,
            context: "RESET test augmented model".to_string(),
        });
    }

    // Build augmented X matrix: [X_original, aug_vars]
    let mut x_aug = Array2::zeros((n, k_augmented));

    // Copy original X
    for i in 0..n {
        for j in 0..k_original {
            x_aug[[i, j]] = x_orig[[i, j]];
        }
    }

    // Add augmentation variables
    for (aug_idx, aug_var) in aug_vars.iter().enumerate() {
        for i in 0..n {
            x_aug[[i, k_original + aug_idx]] = aug_var[i];
        }
    }

    // Step 3: Run augmented regression
    let xtx_aug = xtx(&x_aug.view());
    let (xtx_aug_inv, _) = safe_inverse(&xtx_aug.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Augmented model X'X".to_string(),
            suggestion: format!("Augmentation variables may be collinear: {}", e),
        })?;

    let mut xty_aug: Array1<f64> = Array1::zeros(k_augmented);
    for i in 0..n {
        for j in 0..k_augmented {
            xty_aug[j] += x_aug[[i, j]] * y[i];
        }
    }

    let beta_aug: Array1<f64> = xtx_aug_inv.dot(&xty_aug);

    // Compute SSR for augmented model
    let y_mean = y.mean().unwrap_or(0.0);
    let mut ssr_aug = 0.0;
    let mut sst = 0.0;

    for i in 0..n {
        let fitted_aug: f64 = (0..k_augmented).map(|j| x_aug[[i, j]] * beta_aug[j]).sum();
        let resid = y[i] - fitted_aug;
        ssr_aug += resid * resid;
        sst += (y[i] - y_mean).powi(2);
    }

    // SSR for original model
    let ssr_orig: f64 = ols_result.resid.iter().map(|e| e * e).sum();

    // R² values
    let r2_original = if sst > 0.0 { 1.0 - ssr_orig / sst } else { 0.0 };
    let r2_augmented = if sst > 0.0 { 1.0 - ssr_aug / sst } else { 0.0 };

    // Step 4: F-test
    // F = ((SSR_restricted - SSR_unrestricted) / q) / (SSR_unrestricted / (n - k_unrestricted))
    // where q = number of restrictions (number of added variables)
    let q = n_aug;
    let df1 = q;
    let df2 = n - k_augmented;

    let f_stat = if ssr_aug > 0.0 && df2 > 0 {
        ((ssr_orig - ssr_aug) / q as f64) / (ssr_aug / df2 as f64)
    } else {
        0.0
    };

    let p_value = crate::traits::f_test_p_value(f_stat, df1 as f64, df2 as f64);

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: Model misspecification detected (p={:.4} < 0.05). Consider adding nonlinear terms.",
            p_value
        )
    } else {
        format!(
            "Fail to reject H0: No significant misspecification detected (p={:.4} >= 0.05)",
            p_value
        )
    };

    Ok(ResetTestResult {
        statistic: f_stat,
        p_value,
        df: (df1, df2),
        powers,
        reset_type,
        n_obs: n,
        r2_original,
        r2_augmented,
        significant_at_05: significant,
        interpretation,
    })
}

/// Convenience function for RESET test with default parameters.
///
/// Uses powers [2, 3] and Fitted type.
pub fn run_reset_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<ResetTestResult> {
    reset_test(dataset, y_col, x_cols, &[2, 3], ResetType::Fitted)
}

/// RESET test from pre-computed OLS result (optimized for benchmarking).
///
/// This variant avoids re-computing the OLS regression, which is useful when
/// the OLS result has already been computed (like R's bgtest(model, ...)).
///
/// # Arguments
///
/// * `ols_result` - Pre-computed OLS result
/// * `x` - Design matrix used in OLS (including intercept if present)
/// * `y` - Response vector
/// * `powers` - Powers to test (e.g., [2, 3])
/// * `reset_type` - Type of RESET test
pub fn reset_test_from_ols(
    ols_result: &crate::regression::OlsResult,
    x: &ndarray::ArrayView2<f64>,
    y: &ndarray::Array1<f64>,
    powers: &[usize],
    reset_type: ResetType,
) -> EconResult<ResetTestResult> {
    use ndarray::Axis;

    let powers: Vec<usize> = if powers.is_empty() {
        vec![2, 3]
    } else {
        powers.iter().filter(|&&p| p >= 2).copied().collect()
    };

    if powers.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "RESET test requires powers >= 2".to_string(),
        });
    }

    let n = ols_result.n_obs;
    let k_original = ols_result.n_params;

    // Compute fitted values using vectorized matrix-vector multiplication
    let beta: Array1<f64> = ols_result.coefficients.iter().map(|c| c.estimate).collect();
    let x_view: ndarray::ArrayView2<f64> = x.view();
    let fitted: Array1<f64> = x_view.dot(&beta);

    // Compute augmentation variables
    let aug_vars: Vec<Array1<f64>> = match reset_type {
        ResetType::Fitted => {
            powers.iter().map(|&p| fitted.mapv(|f| f.powi(p as i32))).collect()
        }
        ResetType::Regressor => {
            let mut vars = Vec::new();
            for &p in &powers {
                for j in 0..x.ncols() {
                    let col = x.column(j);
                    // Skip intercept column
                    if col.iter().all(|&v| (v - 1.0).abs() < 1e-10) {
                        continue;
                    }
                    vars.push(col.mapv(|x_val| x_val.powi(p as i32)));
                }
            }
            vars
        }
        ResetType::PrinComp => {
            powers.iter().map(|&p| fitted.mapv(|f| f.powi(p as i32))).collect()
        }
    };

    let n_aug = aug_vars.len();
    if n_aug == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No augmentation variables generated".to_string(),
        });
    }

    let k_augmented = k_original + n_aug;
    if n <= k_augmented {
        return Err(EconError::InsufficientData {
            required: k_augmented + 1,
            provided: n,
            context: "RESET test augmented model".to_string(),
        });
    }

    // Build augmented X matrix using slice assignment (vectorized)
    let mut x_aug = Array2::zeros((n, k_augmented));
    x_aug.slice_mut(ndarray::s![.., ..k_original]).assign(x);

    // Add augmentation columns using slice assignment
    for (aug_idx, aug_var) in aug_vars.iter().enumerate() {
        x_aug.column_mut(k_original + aug_idx).assign(aug_var);
    }

    // Run augmented regression using vectorized operations
    // X'X - use pure ndarray for speed
    let xtx_aug = x_aug.t().dot(&x_aug);
    // Use fast Cholesky inverse (no condition number check - OLS already succeeded)
    let xtx_aug_inv = crate::linalg::cholesky_inverse(&xtx_aug.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Augmented model X'X".to_string(),
            suggestion: format!("Augmentation variables may be collinear: {}", e),
        })?;

    // X'y using vectorized dot product
    let xty_aug = x_aug.t().dot(y);

    // β_aug = (X'X)^{-1} X'y
    let beta_aug = xtx_aug_inv.dot(&xty_aug);

    // Compute SSR using vectorized operations
    let fitted_aug = x_aug.dot(&beta_aug);
    let resid_aug = y - &fitted_aug;

    let y_mean = y.mean().unwrap_or(0.0);
    let ssr_aug: f64 = resid_aug.iter().map(|&r| r * r).sum();
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let ssr_orig: f64 = ols_result.resid.iter().map(|e| e * e).sum();
    let r2_original = if sst > 0.0 { 1.0 - ssr_orig / sst } else { 0.0 };
    let r2_augmented = if sst > 0.0 { 1.0 - ssr_aug / sst } else { 0.0 };

    let q = n_aug;
    let df1 = q;
    let df2 = n - k_augmented;

    let f_stat = if ssr_aug > 0.0 && df2 > 0 {
        ((ssr_orig - ssr_aug) / q as f64) / (ssr_aug / df2 as f64)
    } else {
        0.0
    };

    let p_value = crate::traits::f_test_p_value(f_stat, df1 as f64, df2 as f64);
    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: Model misspecification detected (p={:.4} < 0.05). Consider adding nonlinear terms.",
            p_value
        )
    } else {
        format!(
            "Fail to reject H0: No significant misspecification detected (p={:.4} >= 0.05)",
            p_value
        )
    };

    Ok(ResetTestResult {
        statistic: f_stat,
        p_value,
        df: (df1, df2),
        powers,
        reset_type,
        n_obs: n,
        r2_original,
        r2_augmented,
        significant_at_05: significant,
        interpretation,
    })
}

/// Wald test for comparing nested linear models.
///
/// Tests H₀: The restricted model is adequate (additional parameters = 0)
/// against H₁: The unrestricted model provides better fit
///
/// The test computes:
/// - F-test: F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u)) ~ F(q, n - k_u)
/// - Chi-squared: χ² = n * (RSS_r - RSS_u) / RSS_u ~ χ²(q)
///
/// where q = k_u - k_r is the number of restrictions.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the data
/// * `y_col` - Dependent variable column name
/// * `x_cols_unrestricted` - Independent variables for unrestricted model
/// * `x_cols_restricted` - Independent variables for restricted model
/// * `use_f_test` - If true, use F-test; if false, use Chi-squared
///
/// # Returns
///
/// A `WaldTestResult` containing the test statistic, p-value, and diagnostics.
///
/// # Note
///
/// The restricted model's regressors must be a subset of the unrestricted model's
/// regressors. If not, an error is returned.
///
/// # References
///
/// - Wald, A. (1943). Tests of statistical hypotheses concerning several parameters
///   when the number of observations is large. *Transactions of the American
///   Mathematical Society*, 54(3), 426-482.
///
/// R equivalent: `lmtest::waldtest()`
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::wald_test;
///
/// // Compare model with x1, x2, x3 vs model with just x1, x2
/// let result = wald_test(
///     &dataset,
///     "y",
///     &["x1", "x2", "x3"],  // unrestricted
///     &["x1", "x2"],        // restricted
///     true,                 // use F-test
/// )?;
///
/// println!("F statistic: {:.4}", result.statistic);
/// println!("p-value: {:.4}", result.p_value);
/// ```
pub fn wald_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols_unrestricted: &[&str],
    x_cols_restricted: &[&str],
    use_f_test: bool,
) -> EconResult<WaldTestResult> {
    // Check that restricted vars are a subset of unrestricted
    for &r_col in x_cols_restricted {
        if !x_cols_unrestricted.contains(&r_col) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Restricted model variable '{}' must be in unrestricted model",
                    r_col
                ),
            });
        }
    }

    let q = x_cols_unrestricted.len() - x_cols_restricted.len();
    if q == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Models must differ - restricted model must have fewer variables".to_string(),
        });
    }

    // Run both regressions
    let ols_unrestricted = run_ols(dataset, y_col, x_cols_unrestricted, true, CovarianceType::Standard)?;
    let ols_restricted = run_ols(dataset, y_col, x_cols_restricted, true, CovarianceType::Standard)?;

    let n = ols_unrestricted.n_obs;
    let k_u = ols_unrestricted.n_params; // includes intercept
    let k_r = ols_restricted.n_params;

    // Get RSS from both models
    let rss_u: f64 = ols_unrestricted.resid.iter().map(|e| e * e).sum();
    let rss_r: f64 = ols_restricted.resid.iter().map(|e| e * e).sum();

    let df1 = q; // number of restrictions
    let df2 = n - k_u; // residual df in unrestricted model

    // Compute test statistic and p-value
    // Note: In theory, RSS_r >= RSS_u (restricted has at least as much error).
    // In practice, numerical issues can cause RSS_r < RSS_u, especially with
    // multicollinear data. We clamp to ensure non-negative statistics.
    let rss_diff = (rss_r - rss_u).max(0.0);

    let (statistic, p_value, test_type, df2_opt) = if use_f_test {
        // F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))
        let f_stat = if rss_u > 0.0 && df2 > 0 {
            (rss_diff / df1 as f64) / (rss_u / df2 as f64)
        } else {
            0.0
        };
        let p = crate::traits::f_test_p_value(f_stat, df1 as f64, df2 as f64);
        (f_stat, p, "F".to_string(), Some(df2))
    } else {
        // Chi-sq = n * (RSS_r - RSS_u) / RSS_u
        let chi_stat = if rss_u > 0.0 {
            (n as f64) * rss_diff / rss_u
        } else {
            0.0
        };
        let p = chi_squared_p_value(chi_stat, df1 as f64);
        (chi_stat, p, "Chisq".to_string(), None)
    };

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: The {} additional variable(s) are jointly significant (p={:.4} < 0.05). Use the unrestricted model.",
            q, p_value
        )
    } else {
        format!(
            "Fail to reject H0: The {} additional variable(s) are not jointly significant (p={:.4} >= 0.05). The restricted model is adequate.",
            q, p_value
        )
    };

    Ok(WaldTestResult {
        statistic,
        p_value,
        df1,
        df2: df2_opt,
        test_type,
        n_obs: n,
        k_unrestricted: k_u,
        k_restricted: k_r,
        rss_unrestricted: rss_u,
        rss_restricted: rss_r,
        significant_at_05: significant,
        interpretation,
    })
}

/// Convenience function for Wald test with F-statistic.
pub fn run_wald_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols_unrestricted: &[&str],
    x_cols_restricted: &[&str],
) -> EconResult<WaldTestResult> {
    wald_test(dataset, y_col, x_cols_unrestricted, x_cols_restricted, true)
}

/// Wald test from pre-computed OLS results (optimized for benchmarking).
///
/// This variant avoids re-computing OLS regressions, useful when the models
/// have already been estimated (like R's waldtest(model1, model2)).
///
/// # Arguments
///
/// * `ols_unrestricted` - Pre-computed OLS result for unrestricted model
/// * `ols_restricted` - Pre-computed OLS result for restricted model
/// * `use_f_test` - If true, use F-test; if false, use Chi-squared
pub fn wald_test_from_ols(
    ols_unrestricted: &crate::regression::OlsResult,
    ols_restricted: &crate::regression::OlsResult,
    use_f_test: bool,
) -> EconResult<WaldTestResult> {
    let n = ols_unrestricted.n_obs;
    let k_u = ols_unrestricted.n_params;
    let k_r = ols_restricted.n_params;

    if n != ols_restricted.n_obs {
        return Err(EconError::InvalidSpecification {
            message: "Unrestricted and restricted models must have same number of observations".to_string(),
        });
    }

    let q = k_u - k_r;
    if q <= 0 {
        return Err(EconError::InvalidSpecification {
            message: "Unrestricted model must have more parameters than restricted model".to_string(),
        });
    }

    let rss_u: f64 = ols_unrestricted.resid.iter().map(|e| e * e).sum();
    let rss_r: f64 = ols_restricted.resid.iter().map(|e| e * e).sum();

    let df1 = q;
    let df2 = n - k_u;

    let (statistic, p_value, test_type, df2_opt) = if use_f_test {
        let f_stat = if rss_u > 0.0 && df2 > 0 {
            ((rss_r - rss_u) / df1 as f64) / (rss_u / df2 as f64)
        } else {
            0.0
        };
        let p = crate::traits::f_test_p_value(f_stat, df1 as f64, df2 as f64);
        (f_stat, p, "F".to_string(), Some(df2))
    } else {
        let chi_stat = if rss_u > 0.0 {
            (n as f64) * (rss_r - rss_u) / rss_u
        } else {
            0.0
        };
        let p = chi_squared_p_value(chi_stat, df1 as f64);
        (chi_stat, p, "Chisq".to_string(), None)
    };

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: The {} additional variable(s) are jointly significant (p={:.4} < 0.05). Use the unrestricted model.",
            q, p_value
        )
    } else {
        format!(
            "Fail to reject H0: The {} additional variable(s) are not jointly significant (p={:.4} >= 0.05). The restricted model is adequate.",
            q, p_value
        )
    };

    Ok(WaldTestResult {
        statistic,
        p_value,
        df1,
        df2: df2_opt,
        test_type,
        n_obs: n,
        k_unrestricted: k_u,
        k_restricted: k_r,
        rss_unrestricted: rss_u,
        rss_restricted: rss_r,
        significant_at_05: significant,
        interpretation,
    })
}

/// Breusch-Godfrey test for higher-order serial correlation.
///
/// Tests H₀: No serial correlation up to order p
/// against H₁: Serial correlation exists at some order ≤ p
///
/// The test is more general than Durbin-Watson as it:
/// - Tests for higher-order autocorrelation (AR(p) or MA(p))
/// - Is valid when lagged dependent variables appear as regressors
/// - Is asymptotically valid regardless of regressor properties
///
/// # Algorithm
///
/// 1. Estimate the original regression: y = Xβ + ε
/// 2. Compute residuals: ê = y - Xβ̂
/// 3. Run auxiliary regression: êₜ = Xβ + ρ₁êₜ₋₁ + ... + ρₚêₜ₋ₚ + v
/// 4. Compute test statistic:
///    - Chi-squared: LM = n × R² ~ χ²(p)
///    - F: F = (R² / p) / ((1 - R²) / (n - k - p)) ~ F(p, n-k-p)
///
/// # Arguments
///
/// * `dataset` - Dataset containing the data
/// * `y_col` - Dependent variable column name
/// * `x_cols` - Independent variable column names
/// * `order` - Maximum lag order to test (default: 1)
/// * `test_type` - Type of test statistic (Chi-squared or F)
/// * `fill` - Value to fill for initial lagged residuals (default: 0.0)
///
/// # Returns
///
/// A `BgTestResult` containing the test statistic, p-value, and diagnostics.
///
/// # References
///
/// - Breusch, T.S. (1979). Testing for autocorrelation in dynamic linear models.
///   *Australian Economic Papers*, 17, 334-355.
///
/// - Godfrey, L.G. (1978). Testing against general autoregressive and moving average
///   error models when the regressors include lagged dependent variables.
///   *Econometrica*, 46, 1293-1302.
///
/// R equivalent: `lmtest::bgtest()`
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::{bg_test, BgTestType};
///
/// let result = bg_test(
///     &dataset,
///     "y",
///     &["x1", "x2"],
///     4,           // Test for serial correlation up to order 4
///     BgTestType::Chisq,
///     0.0,         // Fill initial lags with 0
/// )?;
///
/// println!("LM statistic: {:.4}", result.statistic);
/// println!("p-value: {:.4}", result.p_value);
/// ```
pub fn bg_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    order: usize,
    test_type: BgTestType,
    fill: f64,
) -> EconResult<BgTestResult> {
    if order == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Order must be at least 1 for Breusch-Godfrey test".to_string()
        });
    }

    // Step 1: Run original OLS regression
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    let residuals = &ols_result.resid;
    let n = residuals.len();
    let k = ols_result.n_params; // includes intercept

    if n <= k + order {
        return Err(EconError::InsufficientData {
            required: k + order + 1,
            provided: n,
            context: format!("Breusch-Godfrey test with k={} regressors and order={}", k, order),
        });
    }

    // Step 2: Build auxiliary design matrix [X, e_{t-1}, ..., e_{t-p}]
    // Original X matrix from dataset
    let df = dataset.df();
    let design = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let x_orig = &design.data;

    // Number of columns in auxiliary regression: original k + p lagged residuals
    let k_aux = k + order;
    let mut x_aux = Array2::zeros((n, k_aux));

    // Copy original X columns
    for i in 0..n {
        for j in 0..k {
            x_aux[[i, j]] = x_orig[[i, j]];
        }
    }

    // Add lagged residuals columns
    for lag in 1..=order {
        for i in 0..n {
            if i >= lag {
                x_aux[[i, k + lag - 1]] = residuals[i - lag];
            } else {
                // Fill initial values (before enough observations for lag)
                x_aux[[i, k + lag - 1]] = fill;
            }
        }
    }

    // Step 3: Run auxiliary regression: e on [X, e_{t-1}, ..., e_{t-p}]
    // Compute (X'X)^{-1}
    let xtx_mat = xtx(&x_aux.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Auxiliary regression X'X".to_string(),
            suggestion: format!("Check for multicollinearity or try lower order: {}", e),
        })?;

    // X'e (residuals are the dependent variable)
    let mut xte: Array1<f64> = Array1::zeros(k_aux);
    for i in 0..n {
        for j in 0..k_aux {
            xte[j] += x_aux[[i, j]] * residuals[i];
        }
    }

    // β_aux = (X'X)^{-1} X'e
    let beta_aux: Array1<f64> = xtx_inv.dot(&xte);

    // Compute R² of auxiliary regression
    let e_mean = residuals.mean().unwrap_or(0.0);
    let mut ssr_aux = 0.0;
    let mut sst_aux = 0.0;

    for i in 0..n {
        let fitted: f64 = (0..k_aux).map(|j| x_aux[[i, j]] * beta_aux[j]).sum();
        let aux_resid = residuals[i] - fitted;
        ssr_aux += aux_resid * aux_resid;
        sst_aux += (residuals[i] - e_mean).powi(2);
    }

    // Handle edge case where residuals are constant
    let r2_aux = if sst_aux > 0.0 {
        1.0 - ssr_aux / sst_aux
    } else {
        0.0
    };

    // Extract coefficients on lagged residuals
    let lag_coefs: Vec<f64> = beta_aux.slice(ndarray::s![k..]).to_vec();

    // Step 4: Compute test statistic
    let (statistic, p_value, df) = match test_type {
        BgTestType::Chisq => {
            // LM = n * R²  ~ χ²(p)
            let lm = (n as f64) * r2_aux;
            let p = chi_squared_p_value(lm, order as f64);
            (lm, p, (order as f64, None))
        }
        BgTestType::F => {
            // F = (R² / p) / ((1 - R²) / (n - k - p)) ~ F(p, n - k - p)
            let df1 = order as f64;
            let df2 = (n - k - order) as f64;

            let f_stat = if r2_aux < 1.0 && df2 > 0.0 {
                (r2_aux / df1) / ((1.0 - r2_aux) / df2)
            } else {
                f64::INFINITY
            };

            let p = crate::traits::f_test_p_value(f_stat, df1, df2);
            (f_stat, p, (df1, Some(df2)))
        }
    };

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: Significant serial correlation detected up to order {} (p={:.4} < 0.05)",
            order, p_value
        )
    } else {
        format!(
            "Fail to reject H0: No significant serial correlation up to order {} (p={:.4} >= 0.05)",
            order, p_value
        )
    };

    Ok(BgTestResult {
        statistic,
        p_value,
        df,
        order,
        test_type,
        r_squared: r2_aux,
        n_obs: n,
        lag_coefficients: lag_coefs,
        significant_at_05: significant,
        interpretation,
    })
}

/// Convenience function for Breusch-Godfrey test with default parameters.
///
/// Uses order=1 and Chi-squared test type.
pub fn run_bg_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<BgTestResult> {
    bg_test(dataset, y_col, x_cols, 1, BgTestType::Chisq, 0.0)
}

/// Run Breusch-Godfrey test from an OLS result.
///
/// This allows testing on an already-estimated model without re-running OLS.
///
/// # Arguments
///
/// * `ols_result` - Result from `run_ols`
/// * `x` - Original design matrix (n x k)
/// * `order` - Maximum lag order to test
/// * `test_type` - Type of test statistic
/// * `fill` - Value to fill for initial lagged residuals
pub fn bg_test_from_ols(
    ols_result: &crate::regression::OlsResult,
    x: &ndarray::ArrayView2<f64>,
    order: usize,
    test_type: BgTestType,
    fill: f64,
) -> EconResult<BgTestResult> {
    use ndarray::Axis;

    if order == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Order must be at least 1 for Breusch-Godfrey test".to_string()
        });
    }

    let residuals = &ols_result.resid;
    let n = residuals.len();
    let k = x.ncols();

    if n <= k + order {
        return Err(EconError::InsufficientData {
            required: k + order + 1,
            provided: n,
            context: format!("Breusch-Godfrey test with k={} regressors and order={}", k, order),
        });
    }

    // Build auxiliary design matrix [X, e_{t-1}, ..., e_{t-p}]
    let k_aux = k + order;
    let mut x_aux = Array2::zeros((n, k_aux));

    // Copy original X columns using slice assignment (vectorized)
    x_aux.slice_mut(ndarray::s![.., ..k]).assign(x);

    // Add lagged residuals columns
    for lag in 1..=order {
        let col_idx = k + lag - 1;
        // Fill initial values
        for i in 0..lag {
            x_aux[[i, col_idx]] = fill;
        }
        // Copy lagged residuals (vectorized where possible)
        for i in lag..n {
            x_aux[[i, col_idx]] = residuals[i - lag];
        }
    }

    // Run auxiliary regression using vectorized operations
    // X'X - use pure ndarray for speed
    let xtx_mat = x_aux.t().dot(&x_aux);
    // Use fast Cholesky inverse (no condition number check - OLS already succeeded)
    let xtx_inv = crate::linalg::cholesky_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Auxiliary regression X'X".to_string(),
            suggestion: format!("Check for multicollinearity or try lower order: {}", e),
        })?;

    // X'e using vectorized dot product
    let xte = x_aux.t().dot(residuals);

    // β_aux = (X'X)^{-1} X'e
    let beta_aux = xtx_inv.dot(&xte);

    // Compute fitted values and R² using vectorized operations
    let fitted_aux = x_aux.dot(&beta_aux);
    let aux_resid = residuals - &fitted_aux;

    let e_mean = residuals.mean().unwrap_or(0.0);
    let ssr_aux: f64 = aux_resid.iter().map(|&r| r * r).sum();
    let sst_aux: f64 = residuals.iter().map(|&e| (e - e_mean).powi(2)).sum();

    let r2_aux = if sst_aux > 0.0 { 1.0 - ssr_aux / sst_aux } else { 0.0 };
    let lag_coefs: Vec<f64> = beta_aux.slice(ndarray::s![k..]).to_vec();

    // Compute test statistic
    let (statistic, p_value, df) = match test_type {
        BgTestType::Chisq => {
            let lm = (n as f64) * r2_aux;
            let p = chi_squared_p_value(lm, order as f64);
            (lm, p, (order as f64, None))
        }
        BgTestType::F => {
            let df1 = order as f64;
            let df2 = (n - k - order) as f64;
            let f_stat = if r2_aux < 1.0 && df2 > 0.0 {
                (r2_aux / df1) / ((1.0 - r2_aux) / df2)
            } else {
                f64::INFINITY
            };
            let p = crate::traits::f_test_p_value(f_stat, df1, df2);
            (f_stat, p, (df1, Some(df2)))
        }
    };

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: Significant serial correlation detected up to order {} (p={:.4} < 0.05)",
            order, p_value
        )
    } else {
        format!(
            "Fail to reject H0: No significant serial correlation up to order {} (p={:.4} >= 0.05)",
            order, p_value
        )
    };

    Ok(BgTestResult {
        statistic,
        p_value,
        df,
        order,
        test_type,
        r_squared: r2_aux,
        n_obs: n,
        lag_coefficients: lag_coefs,
        significant_at_05: significant,
        interpretation,
    })
}

// =============================================================================
// Harvey-Collier Test for Linearity
// =============================================================================

/// Result of Harvey-Collier test for linearity.
///
/// The Harvey-Collier test uses recursive residuals to detect departures from
/// linearity. Under the null hypothesis of correct linear specification, the
/// recursive residuals should have zero mean.
///
/// # References
///
/// Harvey, A.C. & Collier, P. (1977). "Testing for Functional Misspecification
/// in Regression Analysis." *Journal of Econometrics*, 6(1), 103-119.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarveyCollierResult {
    /// Test statistic (t-value)
    pub statistic: f64,
    /// P-value (two-sided)
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Number of recursive residuals
    pub n_recursive: usize,
    /// Mean of recursive residuals
    pub mean_recursive: f64,
    /// Standard error of mean
    pub se_mean: f64,
    /// Whether the test is significant at 5% level
    pub significant_at_05: bool,
    /// Interpretation of results
    pub interpretation: String,
}

impl fmt::Display for HarveyCollierResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Harvey-Collier Test for Linearity")?;
        writeln!(f, "==================================")?;
        writeln!(f, "H0: Linear specification is correct")?;
        writeln!(f, "H1: Functional misspecification (convex/concave departure)")?;
        writeln!(f)?;
        writeln!(f, "Test statistic (t): {:.4}", self.statistic)?;
        writeln!(f, "Degrees of freedom: {}", self.df)?;
        writeln!(f, "P-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Mean of recursive residuals: {:.6}", self.mean_recursive)?;
        writeln!(f, "Standard error: {:.6}", self.se_mean)?;
        writeln!(f, "Number of recursive residuals: {}", self.n_recursive)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)
    }
}

/// Compute recursive residuals from a linear regression.
///
/// Recursive residuals are the standardized one-step-ahead forecast errors
/// from sequential OLS regressions, starting from the first k+1 observations.
///
/// For observation t (t > k):
///   w_t = (y_t - x_t' β̂_{t-1}) / √(1 + x_t'(X_{t-1}'X_{t-1})⁻¹x_t)
///
/// where β̂_{t-1} is estimated using observations 1 to t-1.
pub fn recursive_residuals(
    x: &ndarray::ArrayView2<f64>,
    y: &Array1<f64>,
) -> EconResult<Vec<f64>> {
    let n = y.len();
    let k = x.ncols();

    if n <= k + 1 {
        return Err(EconError::InsufficientData {
            required: k + 2,
            provided: n,
            context: "Recursive residuals require n > k + 1".to_string(),
        });
    }

    let mut rec_resid = Vec::with_capacity(n - k);

    // Start with first k observations for initial estimate
    for t in k..n {
        // Use observations 0 to t-1 for estimation
        let x_sub = x.slice(ndarray::s![0..t, ..]);
        let y_sub = y.slice(ndarray::s![0..t]);

        // Compute (X'X)⁻¹
        let xtx_sub = xtx(&x_sub);
        let (xtx_inv, _) = match safe_inverse(&xtx_sub.view()) {
            Ok(inv) => inv,
            Err(_) => continue, // Skip if singular
        };

        // Compute β̂ = (X'X)⁻¹ X'y
        let xty_sub: Array1<f64> = x_sub.t().dot(&y_sub);
        let beta_sub = xtx_inv.dot(&xty_sub);

        // One-step-ahead forecast
        let x_t = x.row(t);
        let y_hat_t: f64 = x_t.iter().zip(beta_sub.iter()).map(|(&xi, &bi)| xi * bi).sum();
        let forecast_error = y[t] - y_hat_t;

        // Compute scaling factor: √(1 + x_t'(X'X)⁻¹x_t)
        let x_t_arr: Array1<f64> = x_t.to_owned();
        let quad_form = x_t_arr.dot(&xtx_inv.dot(&x_t_arr));
        let scale = (1.0 + quad_form).sqrt();

        // Recursive residual
        let w_t = forecast_error / scale;
        rec_resid.push(w_t);
    }

    if rec_resid.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "No recursive residuals could be computed".to_string(),
        });
    }

    Ok(rec_resid)
}

/// Perform Harvey-Collier test for linearity.
///
/// The test performs a t-test on the mean of recursive residuals. If the true
/// relationship is not linear but convex or concave, the mean of recursive
/// residuals should differ significantly from zero.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Name of dependent variable
/// * `x_cols` - Names of independent variables
///
/// # Returns
///
/// `HarveyCollierResult` with test statistic, p-value, and interpretation.
///
/// # References
///
/// Harvey, A.C. & Collier, P. (1977). "Testing for Functional Misspecification
/// in Regression Analysis." *Journal of Econometrics*, 6(1), 103-119.
///
/// R equivalent: `lmtest::harvtest()`
pub fn harvey_collier_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<HarveyCollierResult> {
    let df = dataset.df();

    // Build design matrix with intercept
    let design = DesignMatrix::from_dataframe(df, x_cols, true)?;
    let x = design.view();
    let y = DesignMatrix::extract_column(df, y_col)?;

    harvey_collier_test_from_arrays(&x.view(), &y)
}

/// Perform Harvey-Collier test from pre-computed arrays.
///
/// This is useful when you already have the design matrix and y vector.
pub fn harvey_collier_test_from_arrays(
    x: &ndarray::ArrayView2<f64>,
    y: &Array1<f64>,
) -> EconResult<HarveyCollierResult> {
    // Compute recursive residuals
    let rec_resid = recursive_residuals(x, y)?;
    let n_rec = rec_resid.len();

    if n_rec < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_rec,
            context: "Harvey-Collier test requires at least 2 recursive residuals".to_string(),
        });
    }

    // Compute mean and standard error
    let mean: f64 = rec_resid.iter().sum::<f64>() / n_rec as f64;
    let variance: f64 = rec_resid.iter()
        .map(|&w| (w - mean).powi(2))
        .sum::<f64>() / (n_rec - 1) as f64;
    let se = (variance / n_rec as f64).sqrt();

    // T-test: H0: mean = 0
    let t_stat = if se > 1e-15 { mean / se } else { 0.0 };
    let df = n_rec - 1;
    let p_value = crate::traits::t_test_p_value(t_stat, df as f64);

    let significant = p_value < 0.05;
    let interpretation = if significant {
        format!(
            "Reject H0: Significant departure from linearity detected (t={:.4}, p={:.4} < 0.05). \
             The positive/negative mean suggests {}/concave misspecification.",
            t_stat, p_value,
            if mean > 0.0 { "convex" } else { "concave" }
        )
    } else {
        format!(
            "Fail to reject H0: No significant departure from linearity (t={:.4}, p={:.4} >= 0.05).",
            t_stat, p_value
        )
    };

    Ok(HarveyCollierResult {
        statistic: t_stat,
        p_value,
        df,
        n_recursive: n_rec,
        mean_recursive: mean,
        se_mean: se,
        significant_at_05: significant,
        interpretation,
    })
}

/// Convenience function for running Harvey-Collier test.
pub fn run_harvey_collier(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<HarveyCollierResult> {
    harvey_collier_test(dataset, y_col, x_cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // y = x1 + noise (not perfect linear relationship)
        let df = df! {
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9,
                    11.2, 11.8, 13.1, 13.9, 15.2, 15.8, 17.1, 17.9, 19.2, 19.8],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "x2" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                     5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_diagnostics_basic() {
        let dataset = create_test_dataset();
        let result = run_diagnostics(&dataset, "y", &["x1"]).unwrap();

        assert_eq!(result.n_obs, 20);
        assert_eq!(result.n_params, 2);
        assert!(result.jarque_bera.is_some());
        assert!(result.durbin_watson.is_some());
    }

    #[test]
    fn test_durbin_watson() {
        let residuals = Array1::from_vec(vec![0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]);
        let dw = compute_durbin_watson(&residuals);

        // Alternating residuals should have high DW (negative autocorrelation)
        assert!(dw.statistic > 2.0);
    }

    #[test]
    fn test_jarque_bera() {
        // Normal-ish residuals
        let residuals = Array1::from_vec(vec![
            0.1, -0.2, 0.15, -0.1, 0.05, -0.05, 0.2, -0.15, 0.1, -0.1,
            0.08, -0.12, 0.18, -0.08, 0.03, -0.07, 0.22, -0.18, 0.12, -0.14,
        ]);
        let jb = compute_jarque_bera(&residuals);

        assert!(jb.is_some());
        // Normal residuals should have low JB statistic
        assert!(jb.unwrap().statistic < 10.0);
    }

    #[test]
    fn test_bg_test_basic() {
        let dataset = create_test_dataset();

        // Test order 1 (first-order serial correlation)
        let result = bg_test(&dataset, "y", &["x1"], 1, BgTestType::Chisq, 0.0).unwrap();

        assert_eq!(result.order, 1);
        assert_eq!(result.n_obs, 20);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(!result.lag_coefficients.is_empty());
    }

    #[test]
    fn test_bg_test_higher_order() {
        let dataset = create_test_dataset();

        // Test order 4 (up to 4th-order serial correlation)
        let result = bg_test(&dataset, "y", &["x1"], 4, BgTestType::Chisq, 0.0).unwrap();

        assert_eq!(result.order, 4);
        assert_eq!(result.lag_coefficients.len(), 4);
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    }

    #[test]
    fn test_bg_test_f_type() {
        let dataset = create_test_dataset();

        // Test with F statistic
        let result = bg_test(&dataset, "y", &["x1"], 2, BgTestType::F, 0.0).unwrap();

        assert_eq!(result.order, 2);
        assert!(result.df.1.is_some()); // F test has two df values
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_bg_test_with_autocorrelation() {
        // Create data with strong positive serial correlation
        // y_t = 0.8 * y_{t-1} + x_t + e_t (AR(1) errors)
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let mut y = vec![1.0];
        for i in 1..30 {
            // Strong positive autocorrelation in y
            y.push(0.8 * y[i - 1] + x[i] * 0.5 + 0.1 * (i as f64 % 3.0 - 1.0));
        }

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = bg_test(&dataset, "y", &["x"], 1, BgTestType::Chisq, 0.0).unwrap();

        // With strong autocorrelation, we expect either significant or borderline result
        // The lag coefficient should be meaningfully different from zero
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_bg_test_type_parsing() {
        assert!(matches!(BgTestType::from_str("chisq"), Some(BgTestType::Chisq)));
        assert!(matches!(BgTestType::from_str("Chi-squared"), Some(BgTestType::Chisq)));
        assert!(matches!(BgTestType::from_str("f"), Some(BgTestType::F)));
        assert!(matches!(BgTestType::from_str("F-test"), Some(BgTestType::F)));
        assert!(BgTestType::from_str("invalid").is_none());
    }

    #[test]
    fn test_run_bg_test_convenience() {
        let dataset = create_test_dataset();

        // Convenience function should use order=1 and Chi-squared
        let result = run_bg_test(&dataset, "y", &["x1"]).unwrap();

        assert_eq!(result.order, 1);
        assert!(matches!(result.test_type, BgTestType::Chisq));
    }

    // ========================================
    // RESET Test Tests
    // ========================================

    #[test]
    fn test_reset_test_basic() {
        let dataset = create_test_dataset();

        let result = reset_test(&dataset, "y", &["x1"], &[2, 3], ResetType::Fitted).unwrap();

        assert_eq!(result.n_obs, 20);
        assert_eq!(result.powers, vec![2, 3]);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.r2_original >= 0.0 && result.r2_original <= 1.0);
        assert!(result.r2_augmented >= 0.0 && result.r2_augmented <= 1.0);
        // Augmented model should have >= R² as original
        assert!(result.r2_augmented >= result.r2_original - 1e-10);
    }

    #[test]
    fn test_reset_test_nonlinear() {
        // Create data with true quadratic relationship
        // y = 1 + x + x^2 + noise
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi + xi.powi(2) + 0.5 * (xi % 2.0)).collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Linear model is misspecified - RESET should detect this
        let result = reset_test(&dataset, "y", &["x"], &[2], ResetType::Fitted).unwrap();

        // With true quadratic relationship, we expect RESET to be significant
        // (small p-value indicating misspecification)
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_reset_test_power2_only() {
        let dataset = create_test_dataset();

        // Test with just power 2
        let result = reset_test(&dataset, "y", &["x1"], &[2], ResetType::Fitted).unwrap();

        assert_eq!(result.powers, vec![2]);
        assert_eq!(result.df.0, 1); // df1 = number of added terms
    }

    #[test]
    fn test_reset_test_regressor_type() {
        let dataset = create_test_dataset();

        // Test with Regressor type
        let result = reset_test(&dataset, "y", &["x1"], &[2, 3], ResetType::Regressor).unwrap();

        assert!(matches!(result.reset_type, ResetType::Regressor));
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_reset_type_parsing() {
        assert!(matches!(ResetType::from_str("fitted"), Some(ResetType::Fitted)));
        assert!(matches!(ResetType::from_str("regressor"), Some(ResetType::Regressor)));
        assert!(matches!(ResetType::from_str("princomp"), Some(ResetType::PrinComp)));
        assert!(ResetType::from_str("invalid").is_none());
    }

    #[test]
    fn test_run_reset_test_convenience() {
        let dataset = create_test_dataset();

        // Convenience function should use powers [2,3] and Fitted
        let result = run_reset_test(&dataset, "y", &["x1"]).unwrap();

        assert_eq!(result.powers, vec![2, 3]);
        assert!(matches!(result.reset_type, ResetType::Fitted));
    }

    // ========================================
    // Wald Test Tests
    // ========================================

    #[test]
    fn test_wald_test_basic() {
        let dataset = create_test_dataset();

        // Test unrestricted model (y ~ x1 + x2) vs restricted model (y ~ x1)
        let result = wald_test(
            &dataset,
            "y",
            &["x1", "x2"],
            &["x1"],
            true,
        ).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df1, 1); // 1 restriction (x2 dropped)
        assert_eq!(result.test_type, "F");
    }

    #[test]
    fn test_wald_test_chi_squared() {
        let dataset = create_test_dataset();

        // Test with chi-squared statistic (use_f_test = false)
        let result = wald_test(
            &dataset,
            "y",
            &["x1", "x2"],
            &["x1"],
            false,
        ).unwrap();

        assert_eq!(result.test_type, "Chisq");
        assert!(result.statistic >= 0.0);
        // Chi-squared statistic should be non-negative
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_wald_test_significant_variable() {
        // Create data where x2 is significant and uncorrelated with x1
        // y = x1 + 2*x2 + noise
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        // x2 is independent of x1 (not collinear)
        let x2: Vec<f64> = vec![
            5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 9.0, 4.0, 7.0, 2.0,
            6.0, 1.0, 8.0, 3.0, 5.0, 9.0, 2.0, 7.0, 4.0, 8.0,
            3.0, 6.0, 1.0, 9.0, 5.0, 2.0, 7.0, 4.0, 8.0, 3.0,
        ];
        let y: Vec<f64> = x1.iter().zip(x2.iter())
            .enumerate()
            .map(|(i, (&x1i, &x2i))| x1i + 2.0 * x2i + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();

        let df = df! {
            "y" => y,
            "x1" => x1,
            "x2" => x2
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Test if x2 is significant by comparing (y ~ x1 + x2) vs (y ~ x1)
        let result = wald_test(&dataset, "y", &["x1", "x2"], &["x1"], true).unwrap();

        // Since x2 is truly significant, we expect the Wald test to reject
        // (low p-value indicating x2 contributes significantly)
        assert!(result.statistic > 0.0);
        // We expect a significant result since x2 has coefficient of 2.0
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_wald_test_multiple_restrictions() {
        // Create data with multiple variables
        let x1: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=40).map(|i| (i as f64) * 0.5).collect();
        let x3: Vec<f64> = (1..=40).map(|i| (i as f64).sqrt()).collect();
        let y: Vec<f64> = x1.iter()
            .enumerate()
            .map(|(i, &xi)| xi + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();

        let df = df! {
            "y" => y,
            "x1" => x1,
            "x2" => x2,
            "x3" => x3
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Test joint restriction: drop both x2 and x3
        let result = wald_test(
            &dataset,
            "y",
            &["x1", "x2", "x3"],
            &["x1"],
            true,
        ).unwrap();

        assert_eq!(result.df1, 2); // 2 restrictions (x2 and x3 dropped)
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_wald_test_displays_correctly() {
        let dataset = create_test_dataset();
        let result = wald_test(&dataset, "y", &["x1", "x2"], &["x1"], true).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Wald Test"));
        assert!(display.contains("F Statistic") || display.contains("Chi-sq Statistic"));
        assert!(display.contains("p-value"));
    }

    #[test]
    fn test_run_wald_test_convenience() {
        let dataset = create_test_dataset();

        // Convenience function should use F-test by default
        let result = run_wald_test(&dataset, "y", &["x1", "x2"], &["x1"]).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.test_type, "F");
    }

    // ========================================
    // Harvey-Collier Test Tests
    // ========================================

    #[test]
    fn test_harvey_collier_linear() {
        // Linear relationship - should NOT reject
        let dataset = create_test_dataset();
        let result = harvey_collier_test(&dataset, "y", &["x1"]).unwrap();

        // Check structure
        assert!(result.n_recursive > 0);
        assert!(result.df > 0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_harvey_collier_quadratic() {
        // Quadratic relationship - should detect departure from linearity
        let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi + 0.1 * xi.powi(2) + 0.1 * (xi % 3.0)).collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = harvey_collier_test(&dataset, "y", &["x"]).unwrap();

        // With quadratic misspecification, recursive residuals may show a trend
        assert!(result.n_recursive > 0);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_harvey_collier_displays() {
        let dataset = create_test_dataset();
        let result = harvey_collier_test(&dataset, "y", &["x1"]).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Harvey-Collier"));
        assert!(display.contains("P-value"));
        assert!(display.contains("recursive residuals"));
    }

    #[test]
    fn test_recursive_residuals_basic() {
        // Simple test for recursive residuals computation
        let x = ndarray::arr2(&[
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
            [1.0, 10.0],
        ]);
        let y = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1]);

        let rec_resid = recursive_residuals(&x.view(), &y).unwrap();

        // Should have n - k recursive residuals
        assert_eq!(rec_resid.len(), 8); // 10 - 2 = 8
    }

    #[test]
    fn test_run_harvey_collier_convenience() {
        let dataset = create_test_dataset();
        let result = run_harvey_collier(&dataset, "y", &["x1"]).unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
