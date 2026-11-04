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
use crate::linalg::{DesignMatrix, condition_number, safe_inverse, xtx};
use crate::options::Choices;
use crate::regression::{CovarianceType, run_ols};
use crate::resource;
use crate::traits::chi_squared_p_value;
use crate::traits::convergence::{ResultNotes, notes_text};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BgTestType {
    /// Chi-squared statistic (asymptotic)
    #[default]
    Chisq,
    /// F statistic (finite sample correction)
    F,
}

/// Accepted spellings of the Breusch-Godfrey statistic (lmtest's `type = "Chisq" | "F"`).
const BG_TEST_TYPE_CHOICES: Choices<BgTestType> = Choices {
    option: "test_type",
    values: &[
        (&["chisq", "chi_squared", "chi2", "lm"], BgTestType::Chisq),
        (&["f", "f_test"], BgTestType::F),
    ],
};

// FromStr (unknown names are an error listing the accepted ones), choices(), Display.
crate::option_enum!(BgTestType, BG_TEST_TYPE_CHOICES);

impl BgTestType {
    /// Parse from string representation; `None` for an unknown name. Prefer `str::parse`,
    /// which reports unknown names.
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResetType {
    /// Use powers of fitted values (ŷ², ŷ³, ...) - most common
    #[default]
    Fitted,
    /// Use powers of regressors (x², x³, ...)
    Regressor,
    /// Use powers of the first principal component of the regressors, as
    /// `lmtest::resettest(type = "princomp")`: `Z` = the non-constant design columns,
    /// `v1` = the first eigenvector of `cov(Z)`, `pc1 = Z v1` (not centred).
    PrinComp,
}

/// Accepted spellings of the RESET augmentation (lmtest's `type = "fitted" | "regressor" |
/// "princomp"`).
const RESET_TYPE_CHOICES: Choices<ResetType> = Choices {
    option: "reset_type",
    values: &[
        (&["fitted", "fit", "fitted_values"], ResetType::Fitted),
        (&["regressor", "reg", "regressors"], ResetType::Regressor),
        (
            &["princomp", "pca", "pc", "principal_component"],
            ResetType::PrinComp,
        ),
    ],
};

// FromStr (unknown names are an error listing the accepted ones), choices(), Display.
crate::option_enum!(ResetType, RESET_TYPE_CHOICES);

impl ResetType {
    /// Parse from string representation; `None` for an unknown name. Prefer `str::parse`,
    /// which reports unknown names.
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
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
    /// Warnings: the model's aliasing note, and augmentation terms dropped as collinear.
    /// Absent from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
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
    /// Identified parameters (rank) of the unrestricted model
    pub k_unrestricted: usize,
    /// Identified parameters (rank) of the restricted model
    pub k_restricted: usize,
    /// Residual sum of squares for unrestricted model
    pub rss_unrestricted: f64,
    /// Residual sum of squares for restricted model
    pub rss_restricted: f64,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
    /// Warnings: duplicate names ignored, the models' aliasing notes. Absent from JSON when
    /// empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for WaldTestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
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
        f.write_str(&notes_text(self))
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
        let what = match self.reset_type {
            ResetType::Fitted => "the fitted values",
            ResetType::Regressor => "each regressor",
            ResetType::PrinComp => "the first principal component of the regressors",
        };
        writeln!(
            f,
            "Augmentation: powers of {what} (type = \"{}\")",
            self.reset_type
        )?;
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Powers tested: {:?}", self.powers)?;
        writeln!(f, "Original R²: {:.4}", self.r2_original)?;
        writeln!(f, "Augmented R²: {:.4}", self.r2_augmented)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        f.write_str(&notes_text(self))
    }
}

impl ResultNotes for ResetTestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
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
                writeln!(
                    f,
                    "df: ({}, {})",
                    self.df.0 as usize,
                    self.df.1.unwrap_or(0.0) as usize
                )?;
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
        writeln!(
            f,
            "Observations: {}, Parameters: {}",
            self.n_obs, self.n_params
        )?;
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
            writeln!(
                f,
                "  Statistic: {:.4} (range: 0-4, 2 = no autocorrelation)",
                dw.statistic
            )?;
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
            writeln!(f, "  {:<20} {:>10} Status", "Variable", "VIF")?;
            writeln!(f, "  {}", "-".repeat(50))?;
            for v in vif_results {
                if v.vif.is_nan() {
                    writeln!(f, "  {:<20} {:>10} (constant)", v.variable, "N/A")?;
                } else if v.vif.is_infinite() {
                    writeln!(f, "  {:<20} {:>10} Perfect collinearity!", v.variable, "∞")?;
                } else {
                    writeln!(
                        f,
                        "  {:<20} {:>10.2} {}",
                        v.variable, v.vif, v.interpretation
                    )?;
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
                warnings.push(
                    EstimationWarning::HighConditionNumber {
                        value: cn,
                        threshold: 30.0,
                    }
                    .message(),
                );
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
///
/// # Implementation notes
///
/// Uses Welford's online algorithm for numerically stable single-pass
/// computation of mean, variance, skewness, and kurtosis simultaneously.
/// The chi-squared(2) p-value is computed analytically as exp(-x/2)
/// rather than through the general chi-squared distribution.
///
/// # References
///
/// - Jarque, C. M., & Bera, A. K. (1980). "Efficient tests for normality,
///   homoscedasticity and serial independence of regression residuals".
///   *Economics Letters*, 6(3), 255-259.
/// - Welford, B. P. (1962). "Note on a method for calculating corrected sums
///   of squares and products". *Technometrics*, 4(3), 419-420.
fn compute_jarque_bera(residuals: &Array1<f64>) -> Option<TestResult> {
    let n_usize = residuals.len();
    if n_usize < 8 {
        return None; // Need sufficient observations
    }
    let n = n_usize as f64;

    // Single-pass computation using Welford's online algorithm for moments.
    // We accumulate M1 (mean), M2 (sum of squared deviations), M3, and M4
    // in a single pass with O(1) updates per element.
    // Reference: Pebay, P. (2008). "Formulas for robust, one-pass parallel
    // computation of covariances and arbitrary-order statistical moments".
    // Sandia National Laboratories, SAND2008-6212.
    let mut count: f64 = 0.0;
    let mut m1: f64 = 0.0; // mean
    let mut m2: f64 = 0.0; // sum of (x - mean)^2
    let mut m3: f64 = 0.0; // sum of (x - mean)^3 (scaled)
    let mut m4: f64 = 0.0; // sum of (x - mean)^4 (scaled)

    for &x in residuals.iter() {
        let n1 = count;
        count += 1.0;
        let delta = x - m1;
        let delta_n = delta / count;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        // Update M4 before M3 and M2 since it depends on their current values
        // Pebay (2008), Eq. 1.6
        m4 += term1 * delta_n2 * (count * count - 3.0 * count + 3.0) + 6.0 * delta_n2 * m2
            - 4.0 * delta_n * m3;
        // Pebay (2008), Eq. 1.4
        m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * m2;
        // Welford (1962)
        m2 += term1;
        m1 += delta_n;
    }

    // m2 is the sum of squared deviations; variance = m2 / n
    if m2 <= 0.0 {
        return None;
    }

    // Compute skewness^2 and excess_kurtosis^2 directly to avoid sqrt.
    // skewness = (m3 / n) / (m2 / n)^(3/2) = m3 * sqrt(n) / m2^(3/2)
    // skewness^2 = m3^2 * n / m2^3
    let m2_sq = m2 * m2;
    let skewness_sq = (m3 * m3 * n) / (m2_sq * m2);

    // kurtosis = (m4 / n) / (m2 / n)^2 = m4 * n / m2^2
    // excess_kurtosis = kurtosis - 3
    let kurtosis = (m4 * n) / m2_sq;
    let excess_kurtosis = kurtosis - 3.0;

    // JB = (n/6) * (S^2 + (K-3)^2 / 4)  -- Jarque & Bera (1980), Eq. 1
    let jb = (n / 6.0) * (skewness_sq + excess_kurtosis * excess_kurtosis * 0.25);

    // P-value from chi-squared(2): the survival function of chi-squared with
    // df=2 has the closed-form expression P(X > x) = exp(-x/2).
    // This avoids constructing a distribution object and evaluating the
    // general incomplete gamma function.
    let p_value = if jb.is_nan() || jb < 0.0 {
        f64::NAN
    } else if jb.is_infinite() || jb > 1e10 {
        0.0
    } else {
        (-0.5 * jb).exp()
    };

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

/// Jarque-Bera test for normality of a data vector.
///
/// A standalone version of the Jarque-Bera test that accepts any `&[f64]`
/// slice (e.g. residuals from OLS). Uses Welford's single-pass online
/// algorithm for numerically stable moment computation.
///
/// Returns `None` if fewer than 8 observations are provided.
pub fn jarque_bera_test(data: &[f64]) -> Option<TestResult> {
    let arr = ndarray::Array1::from_vec(data.to_vec());
    compute_jarque_bera(&arr)
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
    let diff_sum: f64 = residuals
        .windows(2)
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
            warnings.push(
                EstimationWarning::HighVIF {
                    variable: var_name.clone(),
                    vif,
                }
                .message(),
            );
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
/// * `powers` - Powers to use (default: [2, 3]). Duplicates are removed (first occurrence
///   kept) and powers below 2 are ignored; a power above `i32::MAX` is `InvalidSpecification`.
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
    // Validate powers before fitting anything
    normalize_reset_powers(powers)?;

    // Step 1: Run original OLS regression
    let ols_result = run_ols(dataset, y_col, x_cols, true, CovarianceType::Standard)?;

    // Step 2: The design and y as run_ols read them
    let df = dataset.df();
    let design = DesignMatrix::from_dataframe(df, x_cols, true)
        .map_err(|e| EconError::Internal(e.to_string()))?;
    let y =
        DesignMatrix::extract_column(df, y_col).map_err(|e| EconError::Internal(e.to_string()))?;

    // Steps 3-4: augmentation, augmented fit and F test (one implementation)
    reset_test_from_ols(&ols_result, &design.data.view(), &y, powers, reset_type)
}

/// lmtest's first principal component of the regressors `z` (n x p): `v1` is the
/// eigenvector of `cov(z)` (n - 1 denominator) with the largest eigenvalue (chosen by its
/// Rayleigh quotient, so the eigen solver's order is not relied on), signed so that its
/// largest |entry| is positive (first index on ties); the scores are `z v1`, NOT centred
/// (`resettest(type = "princomp")` uses `Z %*% v1`). Returns `(scores, loadings)`.
fn first_principal_component(
    z: ndarray::ArrayView2<'_, f64>,
) -> EconResult<(Array1<f64>, Array1<f64>)> {
    let (n, p) = z.dim();
    if n < 2 || p == 0 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "first principal component of the regressors".to_string(),
        });
    }
    let means: Vec<f64> = (0..p).map(|j| z.column(j).sum() / n as f64).collect();
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for a in 0..p {
            let da = z[[i, a]] - means[a];
            for b in a..p {
                cov[[a, b]] += da * (z[[i, b]] - means[b]);
            }
        }
    }
    for a in 0..p {
        for b in a..p {
            cov[[a, b]] /= (n - 1) as f64;
            cov[[b, a]] = cov[[a, b]];
        }
    }
    let (_, vecs) = crate::linalg::eig_symmetric(&cov.view())
        .map_err(|e| EconError::Computation(format!("eigen decomposition of cov(Z): {e}")))?;
    let rayleigh = |j: usize| {
        let v = vecs.column(j);
        v.dot(&cov.dot(&v))
    };
    let best = (0..vecs.ncols())
        .max_by(|&a, &b| rayleigh(a).total_cmp(&rayleigh(b)))
        .ok_or_else(|| EconError::Computation("cov(Z) has no eigenvectors".to_string()))?;
    let mut v = vecs.column(best).to_owned();
    let mut lead = 0;
    for j in 1..p {
        if v[j].abs() > v[lead].abs() {
            lead = j;
        }
    }
    if v[lead] < 0.0 {
        v.mapv_inplace(|x| -x);
    }
    let scores = z.dot(&v);
    Ok((scores, v))
}

/// RESET powers as both RESET variants use them: an empty list means `[2, 3]`,
/// powers below 2 are dropped, duplicates are removed (the first occurrence is
/// kept; a repeated power would only add a collinear column), and powers that do
/// not fit `powi`'s `i32` exponent are refused.
fn normalize_reset_powers(powers: &[usize]) -> EconResult<Vec<usize>> {
    if powers.is_empty() {
        return Ok(vec![2, 3]);
    }
    let mut seen = std::collections::HashSet::new();
    let mut normalized = Vec::new();
    for &p in powers {
        if p > i32::MAX as usize {
            return Err(EconError::InvalidSpecification {
                message: format!("RESET power {p} is too large (at most {})", i32::MAX),
            });
        }
        if p >= 2 && seen.insert(p) {
            normalized.push(p);
        }
    }
    if normalized.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "RESET test requires powers >= 2".to_string(),
        });
    }
    Ok(normalized)
}

/// The columns of `x` whose powers the `Regressor` RESET adds: every column that
/// is not all ones (the intercept).
fn reset_regressor_columns(x: &ndarray::ArrayView2<f64>) -> Vec<usize> {
    (0..x.ncols())
        .filter(|&j| !x.column(j).iter().all(|&v| (v - 1.0).abs() < 1e-10))
        .collect()
}

/// Width of the augmented RESET design (original plus augmentation columns),
/// checked against the sample size before anything is allocated.
fn reset_augmented_width(
    n: usize,
    k_original: usize,
    n_powers: usize,
    reset_type: ResetType,
    n_regressor_cols: usize,
) -> EconResult<usize> {
    let n_aug = match reset_type {
        ResetType::Fitted | ResetType::PrinComp => n_powers,
        ResetType::Regressor => {
            resource::checked_len(&[n_powers, n_regressor_cols], "RESET augmented design")?
        }
    };
    if n_aug == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No augmentation variables generated".to_string(),
        });
    }
    let k_augmented = k_original.saturating_add(n_aug);
    if n <= k_augmented {
        return Err(EconError::InsufficientData {
            required: k_augmented.saturating_add(1),
            provided: n,
            context: "RESET test augmented model".to_string(),
        });
    }
    Ok(k_augmented)
}

/// Fill the augmentation columns of `x_aug`, from column `k_original` on: for
/// each power in order, the power of `base` (the fitted values for `Fitted`, the first
/// principal component of the regressors for `PrinComp`) or of each regressor column in
/// `regressor_cols` (`Regressor`).
fn fill_reset_augmentation(
    x_aug: &mut Array2<f64>,
    k_original: usize,
    powers: &[usize],
    reset_type: ResetType,
    base: &Array1<f64>,
    x: &ndarray::ArrayView2<f64>,
    regressor_cols: &[usize],
) -> EconResult<()> {
    let n = x_aug.nrows();
    let mut col = k_original;
    for &p in powers {
        resource::checkpoint()?;
        // normalize_reset_powers keeps every power within i32
        let p = p as i32;
        match reset_type {
            ResetType::Fitted | ResetType::PrinComp => {
                for i in 0..n {
                    x_aug[[i, col]] = base[i].powi(p);
                }
                col += 1;
            }
            ResetType::Regressor => {
                for &j in regressor_cols {
                    for i in 0..n {
                        x_aug[[i, col]] = x[[i, j]].powi(p);
                    }
                    col += 1;
                }
            }
        }
    }
    Ok(())
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
/// * `powers` - Powers to test (e.g., [2, 3]). Deduplicated and checked as in [`reset_test`]:
///   first occurrence kept, powers below 2 ignored, each at most `i32::MAX`.
/// * `reset_type` - Type of RESET test
pub fn reset_test_from_ols(
    ols_result: &crate::regression::OlsResult,
    x: &ndarray::ArrayView2<f64>,
    y: &ndarray::Array1<f64>,
    powers: &[usize],
    reset_type: ResetType,
) -> EconResult<ResetTestResult> {
    let powers = normalize_reset_powers(powers)?;

    let n = ols_result.n_obs;
    let k_original = ols_result.n_params;
    if x.nrows() != n || x.ncols() != k_original || y.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "RESET: the design is {} x {} and y has {} values, but the OLS fit has {n} \
                 observations and {k_original} parameters",
                x.nrows(),
                x.ncols(),
                y.len()
            ),
        });
    }

    // Fitted values of the fit: y - residuals (X b would be NaN for an aliased column)
    let regressor_cols = reset_regressor_columns(x);
    let base: Array1<f64> = match reset_type {
        ResetType::PrinComp => {
            if regressor_cols.is_empty() {
                return Err(EconError::InvalidSpecification {
                    message: "PrinComp RESET requires at least one regressor".to_string(),
                });
            }
            let mut z = resource::try_zeros2(n, regressor_cols.len(), "RESET regressors")?;
            for (c, &j) in regressor_cols.iter().enumerate() {
                z.column_mut(c).assign(&x.column(j));
            }
            first_principal_component(z.view())?.0
        }
        _ => y - &ols_result.resid,
    };

    // Size the augmented model before building it
    let k_augmented = reset_augmented_width(
        n,
        k_original,
        powers.len(),
        reset_type,
        regressor_cols.len(),
    )?;
    let q = k_augmented - k_original;

    // Build augmented X matrix: [X, augmentation columns]
    let mut x_aug = resource::try_zeros2(n, k_augmented, "RESET augmented design")?;
    x_aug.slice_mut(ndarray::s![.., ..k_original]).assign(x);
    fill_reset_augmentation(
        &mut x_aug,
        k_original,
        &powers,
        reset_type,
        &base,
        x,
        &regressor_cols,
    )?;

    // Augmented fit (normal equations when well conditioned, pivoted QR otherwise)
    let aug = crate::regression::ols::solve_ols(x_aug.view(), y.view())?;

    let y_mean = y.mean().unwrap_or(0.0);
    let ssr_aug: f64 = aug.residuals.iter().map(|&r| r * r).sum();
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ssr_orig: f64 = ols_result.resid.iter().map(|e| e * e).sum();
    let r2_original = if sst > 0.0 { 1.0 - ssr_orig / sst } else { 0.0 };
    let r2_augmented = if sst > 0.0 { 1.0 - ssr_aug / sst } else { 0.0 };

    // Degrees of freedom by rank (anova.lm's rule): df1 = rank([X, Z]) - rank(X),
    // df2 = n - rank([X, Z]). lmtest counts ncol(X) and every augmentation column instead,
    // which differs only when a column aliases.
    let rank_x = ols_result.rank();
    let df1 = aug
        .rank
        .checked_sub(rank_x)
        .filter(|d| *d > 0)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: "every RESET augmentation term is collinear with the model; nothing to test"
                .to_string(),
        })?;
    let df2 =
        n.checked_sub(aug.rank)
            .filter(|d| *d > 0)
            .ok_or_else(|| EconError::InsufficientData {
                required: aug.rank + 1,
                provided: n,
                context: "RESET test augmented model".to_string(),
            })?;

    let mut warnings = alias_warnings(ols_result);
    if df1 < q {
        warnings.push(format!(
            "RESET: {} of {q} augmentation terms are collinear with the model and were \
             dropped; df1 = {df1}",
            q - df1
        ));
    }

    let f_stat = if ssr_aug > 0.0 {
        ((ssr_orig - ssr_aug) / df1 as f64) / (ssr_aug / df2 as f64)
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
        warnings,
    })
}

/// The aliasing warnings of a fit (R's "Coefficients not defined because of
/// singularities: ..."), copied for the tests built on it.
fn alias_warnings(ols: &crate::regression::OlsResult) -> Vec<String> {
    ols.warnings
        .iter()
        .filter(|w| w.starts_with("Coefficients not defined because of singularities"))
        .cloned()
        .collect()
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
/// where q = rank_u - rank_r is the number of restrictions (anova.lm: an aliased or
/// collinear extra regressor adds none) and n - k_u is n - rank_u.
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
/// Repeated names are ignored (first occurrence kept, with a warning), as R's formula
/// interface does. The restricted model's regressors must be a strict subset of the
/// unrestricted model's, and the unrestricted model must identify more coefficients;
/// otherwise the result is `InvalidSpecification`.
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
    // Unique names (first occurrence kept), as R's formula interface treats a repeated
    // term; each duplicate is reported.
    let mut warnings = Vec::new();
    let mut dedupe = |cols: &[&str], what: &str| -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(cols.len());
        for &c in cols {
            if out.iter().any(|o| o == c) {
                let note =
                    format!("'{c}' is listed more than once in {what}; duplicates are ignored");
                if !warnings.contains(&note) {
                    warnings.push(note);
                }
            } else {
                out.push(c.to_string());
            }
        }
        out
    };
    let unrestricted = dedupe(x_cols_unrestricted, "x_unrestricted");
    let restricted = dedupe(x_cols_restricted, "x_restricted");

    // The restricted model must be nested in the unrestricted one, strictly
    if restricted.iter().any(|r| !unrestricted.contains(r))
        || restricted.len() >= unrestricted.len()
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the restricted model's variables must be a strict subset of the unrestricted \
                 model's (restricted: [{}], unrestricted: [{}])",
                restricted.join(", "),
                unrestricted.join(", ")
            ),
        });
    }

    // Run both regressions
    let u_cols: Vec<&str> = unrestricted.iter().map(String::as_str).collect();
    let r_cols: Vec<&str> = restricted.iter().map(String::as_str).collect();
    let ols_unrestricted = run_ols(dataset, y_col, &u_cols, true, CovarianceType::Standard)?;
    let ols_restricted = run_ols(dataset, y_col, &r_cols, true, CovarianceType::Standard)?;

    let mut result = wald_test_from_ols(&ols_unrestricted, &ols_restricted, use_f_test)?;
    warnings.append(&mut result.warnings);
    result.warnings = warnings;
    Ok(result)
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
    if n != ols_restricted.n_obs {
        return Err(EconError::InvalidSpecification {
            message: "Unrestricted and restricted models must have same number of observations"
                .to_string(),
        });
    }

    // Restrictions counted by rank (anova.lm): an aliased or collinear extra regressor
    // adds no restriction.
    let (rank_u, rank_r) = (ols_unrestricted.rank(), ols_restricted.rank());
    let q = rank_u
        .checked_sub(rank_r)
        .filter(|q| *q > 0)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "the unrestricted model must have more identified coefficients than the \
                 restricted model (ranks {rank_u} and {rank_r}); collinear extra regressors \
                 cannot be tested"
            ),
        })?;

    let rss_u: f64 = ols_unrestricted.resid.iter().map(|e| e * e).sum();
    let rss_r: f64 = ols_restricted.resid.iter().map(|e| e * e).sum();
    // RSS_r >= RSS_u in exact arithmetic; rounding can invert a tiny difference
    let rss_diff = (rss_r - rss_u).max(0.0);

    let df1 = q;
    let df2 = ols_unrestricted.df_resid; // n - rank_u

    let (statistic, p_value, test_type, df2_opt) = if use_f_test {
        let f_stat = if rss_u > 0.0 && df2 > 0 {
            (rss_diff / df1 as f64) / (rss_u / df2 as f64)
        } else {
            0.0
        };
        let p = crate::traits::f_test_p_value(f_stat, df1 as f64, df2 as f64);
        (f_stat, p, "F".to_string(), Some(df2))
    } else {
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

    let mut warnings = alias_warnings(ols_unrestricted);
    for w in alias_warnings(ols_restricted) {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }

    Ok(WaldTestResult {
        statistic,
        p_value,
        df1,
        df2: df2_opt,
        test_type,
        n_obs: n,
        k_unrestricted: rank_u,
        k_restricted: rank_r,
        rss_unrestricted: rss_u,
        rss_restricted: rss_r,
        significant_at_05: significant,
        interpretation,
        warnings,
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
            message: "Order must be at least 1 for Breusch-Godfrey test".to_string(),
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
            context: format!(
                "Breusch-Godfrey test with k={} regressors and order={}",
                k, order
            ),
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
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
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
pub fn run_bg_test(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<BgTestResult> {
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
    if order == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Order must be at least 1 for Breusch-Godfrey test".to_string(),
        });
    }

    let residuals = &ols_result.resid;
    let n = residuals.len();
    let k = x.ncols();

    if n <= k + order {
        return Err(EconError::InsufficientData {
            required: k + order + 1,
            provided: n,
            context: format!(
                "Breusch-Godfrey test with k={} regressors and order={}",
                k, order
            ),
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
    let xtx_inv = crate::linalg::cholesky_inverse(&xtx_mat.view()).map_err(|e| {
        EconError::SingularMatrix {
            context: "Auxiliary regression X'X".to_string(),
            suggestion: format!("Check for multicollinearity or try lower order: {}", e),
        }
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

    let r2_aux = if sst_aux > 0.0 {
        1.0 - ssr_aux / sst_aux
    } else {
        0.0
    };
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
        writeln!(
            f,
            "H1: Functional misspecification (convex/concave departure)"
        )?;
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
pub fn recursive_residuals(x: &ndarray::ArrayView2<f64>, y: &Array1<f64>) -> EconResult<Vec<f64>> {
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
        let y_hat_t: f64 = x_t
            .iter()
            .zip(beta_sub.iter())
            .map(|(&xi, &bi)| xi * bi)
            .sum();
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
    let variance: f64 =
        rec_resid.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / (n_rec - 1) as f64;
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
            t_stat,
            p_value,
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
    fn test_reset_deduplicates_powers() {
        let ds = create_test_dataset();
        let base = reset_test(&ds, "y", &["x1"], &[2, 3], ResetType::Fitted).unwrap();
        let dup = reset_test(&ds, "y", &["x1"], &[2, 3, 2, 3, 3], ResetType::Fitted).unwrap();
        assert_eq!(dup.powers, vec![2, 3]);
        assert_eq!(dup.df, base.df);
        assert!((dup.statistic - base.statistic).abs() <= 1e-12 * base.statistic.abs().max(1.0));
    }

    #[test]
    fn test_reset_rejects_power_beyond_i32() {
        let ds = create_test_dataset();
        let e = reset_test(
            &ds,
            "y",
            &["x1"],
            &[2, i32::MAX as usize + 1],
            ResetType::Fitted,
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("power")),
            "{e:?}"
        );
    }

    #[test]
    fn test_reset_checks_sample_size_before_allocating() {
        // Guard: passes before and after. After the fix, no augmented buffer is
        // built for an impossible model.
        let ds = create_test_dataset();
        let powers: Vec<usize> = (2..302).collect();
        let e = crate::resource::with_alloc_ceiling(Some(1024), || {
            reset_test(&ds, "y", &["x1"], &powers, ResetType::Fitted)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::InsufficientData { .. }), "{e:?}");
    }

    #[test]
    fn test_reset_stops_when_cancelled() {
        let cancelled = || {
            let t = resource::CancelToken::new();
            t.cancel();
            resource::Budget::new(None, t)
        };
        let ds = create_test_dataset();
        for reset_type in [ResetType::Fitted, ResetType::Regressor] {
            let e = resource::with_budget(cancelled(), || {
                reset_test(&ds, "y", &["x1", "x2"], &[2, 3], reset_type)
            })
            .unwrap_err();
            assert!(
                matches!(e, EconError::Cancelled(_)),
                "{reset_type:?}: {e:?}"
            );
        }
        // The OLS fit is done outside the budget, so this reaches RESET's own checkpoint
        let ols = run_ols(&ds, "y", &["x1"], true, CovarianceType::Standard).unwrap();
        let x = DesignMatrix::from_dataframe(ds.df(), &["x1"], true).unwrap();
        let y = DesignMatrix::extract_column(ds.df(), "y").unwrap();
        let e = resource::with_budget(cancelled(), || {
            reset_test_from_ols(&ols, &x.view(), &y, &[2, 3], ResetType::Fitted)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_reset_from_ols_deduplicates_and_bounds_powers() {
        let ds = create_test_dataset();
        let ols = run_ols(&ds, "y", &["x1"], true, CovarianceType::Standard).unwrap();
        let x = DesignMatrix::from_dataframe(ds.df(), &["x1"], true).unwrap();
        let y = DesignMatrix::extract_column(ds.df(), "y").unwrap();
        let base = reset_test_from_ols(&ols, &x.view(), &y, &[3, 2], ResetType::Fitted).unwrap();
        let dup = reset_test_from_ols(&ols, &x.view(), &y, &[3, 2, 3], ResetType::Fitted).unwrap();
        assert_eq!(dup.powers, vec![3, 2]);
        assert_eq!(dup.df, base.df);
        assert!((dup.statistic - base.statistic).abs() <= 1e-12 * base.statistic.abs().max(1.0));
        let e = reset_test_from_ols(
            &ols,
            &x.view(),
            &y,
            &[i32::MAX as usize + 1],
            ResetType::Regressor,
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("power")),
            "{e:?}"
        );
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
            0.1, -0.2, 0.15, -0.1, 0.05, -0.05, 0.2, -0.15, 0.1, -0.1, 0.08, -0.12, 0.18, -0.08,
            0.03, -0.07, 0.22, -0.18, 0.12, -0.14,
        ]);
        let jb = compute_jarque_bera(&residuals);

        assert!(jb.is_some());
        // Normal residuals should have low JB statistic
        assert!(jb.unwrap().statistic < 10.0);
    }

    /// The `p2a reg diagnostics` exhibit on Grunfeld against tseries::jarque.bera.test,
    /// lmtest::bptest (studentized), lmtest::dwtest, car::vif and
    /// kappa(model.matrix, exact = TRUE) (case `ols_diagnostics_grunfeld`).
    #[test]
    fn test_validate_diagnostics_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        let d = run_diagnostics(&ds, "inv", &["value", "capital"]).unwrap();
        let n = d.n_obs;
        let jb = d
            .jarque_bera
            .as_ref()
            .expect("Jarque-Bera is computed for n = 200");
        let bp = d.breusch_pagan.as_ref().expect("Breusch-Pagan is computed");
        let dw = d.durbin_watson.as_ref().expect("Durbin-Watson is computed");
        // car::vif reports no intercept row (p2a's is NaN; T1-WP05 drops it from the text output).
        let (names, vif): (Vec<&str>, Vec<f64>) = d
            .vif
            .as_ref()
            .expect("VIF is computed")
            .iter()
            .filter(|v| v.variable != "(Intercept)")
            .map(|v| (v.variable.as_str(), v.vif))
            .unzip();
        assert_eq!((d.n_obs, d.n_params), (200, 3));
        Check::new(&case)
            .scalar("jarque_bera", jb.statistic, Tol::stat(n))
            .scalar("jarque_bera_p", jb.p_value, Tol::p_value(n))
            // Studentized (Koenker), as bptest's default.
            .scalar("breusch_pagan", bp.statistic, Tol::stat(n))
            .scalar("breusch_pagan_p", bp.p_value, Tol::p_value(n))
            .scalar("durbin_watson", dw.statistic, Tol::stat(n))
            .named("vif", &names, &vif, Tol::stat(n))
            .scalar(
                "condition_number",
                d.condition_number.expect("condition number is computed"),
                Tol::stat(n),
            )
            .finish();
        // Not pinned: breusch_pagan_df (TestResult has no df field; chi2(k - 1) = 2 is
        // implied), durbin_watson_p_greater (p2a computes no DW p-value),
        // breusch_pagan_nonstudentized (not offered), bgtest*/reset*/harvey_collier*/wald*
        // (T1-WP04 pins them).
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
        assert!(matches!(
            BgTestType::from_str("chisq"),
            Some(BgTestType::Chisq)
        ));
        assert!(matches!(
            BgTestType::from_str("Chi-squared"),
            Some(BgTestType::Chisq)
        ));
        assert!(matches!(BgTestType::from_str("f"), Some(BgTestType::F)));
        assert!(matches!(
            BgTestType::from_str("F-test"),
            Some(BgTestType::F)
        ));
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
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 1.0 + xi + xi.powi(2) + 0.5 * (xi % 2.0))
            .collect();

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
        assert!(matches!(
            ResetType::from_str("fitted"),
            Some(ResetType::Fitted)
        ));
        assert!(matches!(
            ResetType::from_str("regressor"),
            Some(ResetType::Regressor)
        ));
        assert!(matches!(
            ResetType::from_str("princomp"),
            Some(ResetType::PrinComp)
        ));
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

    /// Noisy, x2 not collinear with x1 (create_test_dataset has x2 = 0.5 x1 exactly).
    fn create_wald_dataset() -> Dataset {
        let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=20)
            .map(|i| ((i * 7) % 11) as f64 * 0.5 + 0.01 * i as f64)
            .collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .enumerate()
            .map(|(i, (a, b))| 0.9 * a + 0.2 * b + 0.3 * ((i * 13) % 5) as f64)
            .collect();
        Dataset::new(df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap())
    }

    #[test]
    fn test_wald_test_basic() {
        let dataset = create_wald_dataset();

        // Test unrestricted model (y ~ x1 + x2) vs restricted model (y ~ x1)
        let result = wald_test(&dataset, "y", &["x1", "x2"], &["x1"], true).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df1, 1); // 1 restriction (x2 dropped)
        assert_eq!(result.test_type, "F");
    }

    #[test]
    fn test_wald_test_chi_squared() {
        let dataset = create_wald_dataset();

        // Test with chi-squared statistic (use_f_test = false)
        let result = wald_test(&dataset, "y", &["x1", "x2"], &["x1"], false).unwrap();

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
            5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 9.0, 4.0, 7.0, 2.0, 6.0, 1.0, 8.0, 3.0, 5.0, 9.0, 2.0,
            7.0, 4.0, 8.0, 3.0, 6.0, 1.0, 9.0, 5.0, 2.0, 7.0, 4.0, 8.0, 3.0,
        ];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
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
        let x2: Vec<f64> = (1..=40)
            .map(|i| ((i * 7) % 11) as f64 * 0.5 + 0.01 * i as f64)
            .collect();
        let x3: Vec<f64> = (1..=40).map(|i| (i as f64).sqrt()).collect();
        let y: Vec<f64> = x1
            .iter()
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
        let result = wald_test(&dataset, "y", &["x1", "x2", "x3"], &["x1"], true).unwrap();

        assert_eq!(result.df1, 2); // 2 restrictions (x2 and x3 dropped)
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_wald_test_displays_correctly() {
        let dataset = create_wald_dataset();
        let result = wald_test(&dataset, "y", &["x1", "x2"], &["x1"], true).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Wald Test"));
        assert!(display.contains("F Statistic") || display.contains("Chi-sq Statistic"));
        assert!(display.contains("p-value"));
    }

    #[test]
    fn test_run_wald_test_convenience() {
        let dataset = create_wald_dataset();

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
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 1.0 + xi + 0.1 * xi.powi(2) + 0.1 * (xi % 3.0))
            .collect();

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

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Pinned to case `diag_bgtest_ar1` (lmtest::bgtest on the legacy Test 4 DGP; validation/r/cases/regression_legacy.R).
    #[test]
    fn test_validate_bgtest_ar1() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("diag_bgtest_ar1");
        let n = case.scalar("n_obs") as usize;
        let lm1 = bg_test(&ds, "y", &["x"], 1, BgTestType::Chisq, 0.0).unwrap();
        let lm4 = bg_test(&ds, "y", &["x"], 4, BgTestType::Chisq, 0.0).unwrap();
        let f1 = bg_test(&ds, "y", &["x"], 1, BgTestType::F, 0.0).unwrap();
        let f4 = bg_test(&ds, "y", &["x"], 4, BgTestType::F, 0.0).unwrap();
        let lags = |k: usize| {
            (1..=k)
                .map(|j| format!("lag(resid)_{j}"))
                .collect::<Vec<_>>()
        };
        Check::new(&case)
            .scalar("bgtest1_chisq", lm1.statistic, Tol::stat(n))
            .scalar("bgtest1_chisq_df", lm1.df.0, Tol::exact())
            .scalar("bgtest1_chisq_p", lm1.p_value, Tol::p_value(n))
            .scalar("bgtest4_chisq", lm4.statistic, Tol::stat(n))
            .scalar("bgtest4_chisq_df", lm4.df.0, Tol::exact())
            .scalar("bgtest4_chisq_p", lm4.p_value, Tol::p_value(n))
            .scalar("bgtest1_f", f1.statistic, Tol::stat(n))
            .scalar("bgtest1_f_df1", f1.df.0, Tol::exact())
            .scalar("bgtest1_f_df2", f1.df.1.unwrap(), Tol::exact())
            .scalar("bgtest1_f_p", f1.p_value, Tol::p_value(n))
            .scalar("bgtest4_f", f4.statistic, Tol::stat(n))
            .scalar("bgtest4_f_df1", f4.df.0, Tol::exact())
            .scalar("bgtest4_f_df2", f4.df.1.unwrap(), Tol::exact())
            .scalar("bgtest4_f_p", f4.p_value, Tol::p_value(n))
            .named(
                "bgtest1_lag_coef",
                &lags(1),
                &lm1.lag_coefficients,
                Tol::coef(n),
            )
            .named(
                "bgtest4_lag_coef",
                &lags(4),
                &lm4.lag_coefficients,
                Tol::coef(n),
            )
            .scalar("bgtest1_aux_r2", lm1.r_squared, Tol::stat(n))
            .scalar("bgtest4_aux_r2", lm4.r_squared, Tol::stat(n))
            .scalar("n_obs", lm1.n_obs as f64, Tol::exact())
            .finish();
        // AR(1) errors are detected
        assert!(lm1.df.1.is_none() && lm1.significant_at_05 && f1.significant_at_05);
    }

    /// Pinned to case `ols_diagnostics_grunfeld` (lmtest::bgtest on Grunfeld; validation/r/cases/regression.R).
    #[test]
    fn test_validate_bgtest_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        let x = ["value", "capital"];
        let lm1 = bg_test(&ds, "inv", &x, 1, BgTestType::Chisq, 0.0).unwrap();
        let lm4 = bg_test(&ds, "inv", &x, 4, BgTestType::Chisq, 0.0).unwrap();
        let f1 = bg_test(&ds, "inv", &x, 1, BgTestType::F, 0.0).unwrap();
        let n = lm1.n_obs;
        Check::new(&case)
            .scalar("bgtest1_chisq", lm1.statistic, Tol::stat(n))
            .scalar("bgtest1_chisq_p", lm1.p_value, Tol::p_value(n))
            .scalar("bgtest1_f", f1.statistic, Tol::stat(n))
            .scalar("bgtest1_f_p", f1.p_value, Tol::p_value(n))
            .scalar("bgtest4_chisq", lm4.statistic, Tol::stat(n))
            .scalar("bgtest4_chisq_p", lm4.p_value, Tol::p_value(n))
            .finish();
        assert_eq!(n, 200);
        // R's bgtest(type = "F") parameter is (1, 196); the case does not store df.
        assert_eq!(f1.df, (1.0, Some(196.0)));
    }

    /// Pinned to case `diag_reset_quad` (lmtest::resettest on the legacy Test 5 DGP; validation/r/cases/regression_legacy.R).
    #[test]
    fn test_validate_reset_quad() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("diag_reset_quad");
        let n = case.scalar("n_obs") as usize;
        let rf = reset_test(&ds, "y", &["x"], &[2, 3], ResetType::Fitted).unwrap();
        let rr = reset_test(&ds, "y", &["x"], &[2, 3], ResetType::Regressor).unwrap();
        Check::new(&case)
            .scalar("reset_fitted", rf.statistic, Tol::stat(n))
            .scalar("reset_fitted_df1", rf.df.0 as f64, Tol::exact())
            .scalar("reset_fitted_df2", rf.df.1 as f64, Tol::exact())
            .scalar("reset_fitted_p", rf.p_value, Tol::p_value(n))
            .scalar("reset_regressor", rr.statistic, Tol::stat(n))
            .scalar("reset_regressor_df1", rr.df.0 as f64, Tol::exact())
            .scalar("reset_regressor_df2", rr.df.1 as f64, Tol::exact())
            .scalar("reset_regressor_p", rr.p_value, Tol::p_value(n))
            .scalar("r2_original", rf.r2_original, Tol::stat(n))
            .scalar("r2_augmented_fitted", rf.r2_augmented, Tol::stat(n))
            .scalar("r2_augmented_regressor", rr.r2_augmented, Tol::stat(n))
            .scalar("n_obs", rf.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!(rf.powers, vec![2, 3]);
        assert!(rf.significant_at_05);
    }

    /// Pinned to case `ols_diagnostics_grunfeld` (lmtest::resettest on Grunfeld; validation/r/cases/regression.R).
    #[test]
    fn test_validate_reset_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        let x = ["value", "capital"];
        let rf = reset_test(&ds, "inv", &x, &[2, 3], ResetType::Fitted).unwrap();
        let rr = reset_test(&ds, "inv", &x, &[2, 3], ResetType::Regressor).unwrap();
        let n = rf.n_obs;
        Check::new(&case)
            .scalar("reset_fitted", rf.statistic, Tol::stat(n))
            .scalar("reset_fitted_p", rf.p_value, Tol::p_value(n))
            .scalar("reset_regressor", rr.statistic, Tol::stat(n))
            .scalar("reset_regressor_p", rr.p_value, Tol::p_value(n))
            .finish();
        // R's parameters: resettest prints df1 = 2, df2 = 195 for the fitted type, and
        // powers 2:3 of two regressors add four terms.
        assert_eq!(rf.df, (2, 195));
        assert_eq!(rr.df, (4, 193));
    }

    /// Pinned to case `diag_harvtest_nonlin` (lmtest::harvtest on the legacy Test 6 DGP; validation/r/cases/regression_legacy.R).
    #[test]
    fn test_validate_harvey_collier_nonlin() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("diag_harvtest_nonlin");
        let n = case.scalar("n_obs") as usize;
        let r = harvey_collier_test(&ds, "y", &["x"]).unwrap();
        Check::new(&case)
            // lmtest::harvtest reports |t|; p2a keeps the sign of the mean recursive residual
            .scalar("harvey_collier", r.statistic.abs(), Tol::stat(n))
            .scalar("harvey_collier_df", r.df as f64, Tol::exact())
            .scalar("harvey_collier_p", r.p_value, Tol::p_value(n))
            .scalar("recresid_mean", r.mean_recursive, Tol::stat(n))
            .scalar("recresid_se_mean", r.se_mean, Tol::stat(n))
            .scalar("n_recresid", r.n_recursive as f64, Tol::exact())
            .finish();
        assert_eq!(r.statistic.signum(), case.scalar("recresid_mean").signum());
    }

    /// Pinned to case `ols_diagnostics_grunfeld` (lmtest::harvtest on Grunfeld; validation/r/cases/regression.R).
    #[test]
    fn test_validate_harvey_collier_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        let r = harvey_collier_test(&ds, "inv", &["value", "capital"]).unwrap();
        let n = ds.nrows();
        Check::new(&case)
            // lmtest::harvtest reports |t|; p2a keeps the sign of the mean recursive residual
            .scalar("harvey_collier", r.statistic.abs(), Tol::stat(n))
            .scalar("harvey_collier_p", r.p_value, Tol::p_value(n))
            .finish();
        // R's df is n - k - 1 = 196.
        assert_eq!((r.df, r.n_recursive), (196, 197));
    }

    /// Pinned to case `diag_wald_sim` (lmtest::waldtest on the legacy Test 7 DGP; validation/r/cases/regression_legacy.R).
    #[test]
    fn test_validate_wald_sim() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("diag_wald_sim");
        let n = case.scalar("n_obs") as usize;
        let full = ["x1", "x2", "x3"];
        let f1 = wald_test(&ds, "y", &full, &["x1", "x3"], true).unwrap();
        let f2 = wald_test(&ds, "y", &full, &["x1"], true).unwrap();
        let c1 = wald_test(&ds, "y", &full, &["x1", "x3"], false).unwrap();
        let c2 = wald_test(&ds, "y", &full, &["x1"], false).unwrap();
        Check::new(&case)
            .scalar("wald_drop_x2_f", f1.statistic, Tol::stat(n))
            .scalar("wald_drop_x2_f_df1", f1.df1 as f64, Tol::exact())
            .scalar("wald_drop_x2_f_df2", f1.df2.unwrap() as f64, Tol::exact())
            .scalar("wald_drop_x2_f_p", f1.p_value, Tol::p_value(n))
            .scalar("wald_drop_x2_x3_f", f2.statistic, Tol::stat(n))
            .scalar("wald_drop_x2_x3_f_df1", f2.df1 as f64, Tol::exact())
            .scalar(
                "wald_drop_x2_x3_f_df2",
                f2.df2.unwrap() as f64,
                Tol::exact(),
            )
            .scalar("wald_drop_x2_x3_f_p", f2.p_value, Tol::p_value(n))
            .scalar("rss_unrestricted", f1.rss_unrestricted, Tol::stat(n))
            .scalar("rss_drop_x2", f1.rss_restricted, Tol::stat(n))
            .scalar("rss_drop_x2_x3", f2.rss_restricted, Tol::stat(n))
            .scalar("k_unrestricted", f1.k_unrestricted as f64, Tol::exact())
            .scalar("k_drop_x2", f1.k_restricted as f64, Tol::exact())
            .scalar("k_drop_x2_x3", f2.k_restricted as f64, Tol::exact())
            .scalar("n_obs", f1.n_obs as f64, Tol::exact())
            .finish();
        // deviation REG-37: n (RSS_r - RSS_u) / RSS_u, not lmtest's q F (wald_drop_*_chisq, aod_chi2_*)
        assert_eq!(c1.test_type, "Chisq");
        assert_eq!((c1.df1, c1.df2), (1, None));
        assert!(c2.statistic > c1.statistic);
        assert!(!f1.significant_at_05 && f2.significant_at_05);
    }

    /// Pinned to case `ols_diagnostics_grunfeld` (lmtest::waldtest on Grunfeld; validation/r/cases/regression.R).
    #[test]
    fn test_validate_wald_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        let f = wald_test(&ds, "inv", &["value", "capital"], &["value"], true).unwrap();
        let n = f.n_obs;
        Check::new(&case)
            .scalar("wald_drop_capital_f", f.statistic, Tol::stat(n))
            .scalar("wald_drop_capital_f_p", f.p_value, Tol::p_value(n))
            .finish();
        // deviation REG-37: n (RSS_r - RSS_u) / RSS_u, not lmtest's q F (wald_drop_capital_chisq)
        assert_eq!((f.df1, f.df2), (1, Some(197)));
    }

    #[test]
    fn test_reset_type_parses_aliases_and_rejects_unknown() {
        ResetType::choices().validate().unwrap();
        for (s, t) in [
            ("fitted", ResetType::Fitted),
            ("FIT", ResetType::Fitted),
            ("regressor", ResetType::Regressor),
            ("regressors", ResetType::Regressor),
            ("princomp", ResetType::PrinComp),
            ("PCA", ResetType::PrinComp),
            ("principal-component", ResetType::PrinComp),
        ] {
            assert_eq!(s.parse::<ResetType>().unwrap(), t, "{s}");
        }
        match "pca2".parse::<ResetType>().unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("reset_type") && message.contains("princomp"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn test_bg_test_type_parses_aliases_and_rejects_unknown() {
        BgTestType::choices().validate().unwrap();
        for (s, t) in [
            ("Chisq", BgTestType::Chisq),
            ("chi-squared", BgTestType::Chisq),
            ("LM", BgTestType::Chisq),
            ("F", BgTestType::F),
            ("f-test", BgTestType::F),
        ] {
            assert_eq!(s.parse::<BgTestType>().unwrap(), t, "{s}");
        }
        assert!(matches!(
            "wald".parse::<BgTestType>(),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    /// lmtest::resettest(type = "princomp"): pc1 = Z v1, v1 the first eigenvector of
    /// cov(Z), NOT centred; power = 3 alone separates it from centred scores
    /// (22.918632127631003 vs 6.540979982413889). Before: princomp returned the fitted
    /// statistic 91.53815584456794.
    #[test]
    fn test_validate_reset_princomp() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("reset_princomp");
        let x = ["x1", "x2"];
        let pc = reset_test(&ds, "y", &x, &[2, 3], ResetType::PrinComp).unwrap();
        let fit = reset_test(&ds, "y", &x, &[2, 3], ResetType::Fitted).unwrap();
        let reg = reset_test(&ds, "y", &x, &[2, 3], ResetType::Regressor).unwrap();
        let p3 = reset_test(&ds, "y", &x, &[3], ResetType::PrinComp).unwrap();
        let n = pc.n_obs;
        let df = |r: &ResetTestResult| [r.df.0 as f64, r.df.1 as f64];
        Check::new(&case)
            .scalar("princomp_stat", pc.statistic, Tol::stat(n))
            .vector("princomp_df", &df(&pc), Tol::exact())
            .scalar("princomp_p", pc.p_value, Tol::p_value(n))
            .scalar("fitted_stat", fit.statistic, Tol::stat(n))
            .scalar("fitted_p", fit.p_value, Tol::p_value(n))
            .scalar("regressor_stat", reg.statistic, Tol::stat(n))
            .vector("regressor_df", &df(&reg), Tol::exact())
            .scalar("princomp_p3_stat", p3.statistic, Tol::stat(n))
            .vector("princomp_p3_df", &df(&p3), Tol::exact())
            .scalar("princomp_p3_p", p3.p_value, Tol::p_value(n))
            .finish();
        assert!((pc.statistic - fit.statistic).abs() > 1.0); // R distinguishes them
        assert!((p3.statistic - case.scalar("princomp_p3_centred_stat")).abs() > 0.1);
        assert!(pc.to_string().contains("princomp") && pc.warnings.is_empty());

        // The component itself: a unit eigenvector of cov(Z) for the largest eigenvalue,
        // scores = Z v (uncentred); the statistic does not depend on its sign.
        let z = DesignMatrix::from_dataframe(ds.df(), &x, false)
            .unwrap()
            .data;
        let (scores, v) = first_principal_component(z.view()).unwrap();
        assert!((v.dot(&v) - 1.0).abs() < 1e-12);
        let zc = &z - &z.mean_axis(ndarray::Axis(0)).unwrap();
        let cov = zc.t().dot(&zc) / (n as f64 - 1.0);
        let cv = cov.dot(&v);
        let lambda = v.dot(&cv);
        let tr = cov[[0, 0]] + cov[[1, 1]];
        assert!(lambda >= tr / 2.0); // the larger of the two eigenvalues
        for (a, b) in cv.iter().zip(v.iter()) {
            assert!(
                (a - lambda * b).abs() <= 1e-10 * lambda.abs(),
                "{a} vs {}",
                lambda * b
            );
        }
        let zv = z.dot(&v);
        assert!(
            scores
                .iter()
                .zip(&zv)
                .all(|(a, b)| (a - b).abs() <= 1e-12 * b.abs().max(1.0))
        );
        let xi = DesignMatrix::from_dataframe(ds.df(), &x, true)
            .unwrap()
            .data;
        let y = DesignMatrix::extract_column(ds.df(), "y").unwrap();
        let ssr = |m: &Array2<f64>| {
            let f = crate::linalg::solve::least_squares(m.view(), y.view()).unwrap();
            f.residuals.iter().map(|e| e * e).sum::<f64>()
        };
        let mut aug = Array2::<f64>::zeros((n, 5));
        aug.slice_mut(ndarray::s![.., ..3]).assign(&xi);
        for i in 0..n {
            let s = -scores[i];
            aug[[i, 3]] = s * s;
            aug[[i, 4]] = s * s * s;
        }
        let (s0, s1) = (ssr(&xi), ssr(&aug));
        let f_neg = ((s0 - s1) / 2.0) / (s1 / (n as f64 - 5.0));
        assert!(
            (f_neg - pc.statistic).abs() <= 1e-10 * pc.statistic,
            "{f_neg}"
        );

        // the pre-computed-OLS variant agrees
        let ols = run_ols(&ds, "y", &x, true, CovarianceType::Standard).unwrap();
        let pc2 = reset_test_from_ols(&ols, &xi.view(), &y, &[2, 3], ResetType::PrinComp).unwrap();
        Check::new(&case)
            .scalar("princomp_stat", pc2.statistic, Tol::stat(n))
            .finish();
    }

    /// RESET on the dummy-trap model: the fitted values of the QR fit, and the degrees of
    /// freedom by rank, pinned to anova.lm of the augmented fit: df2 = n - rank = 114.
    #[test]
    fn test_validate_reset_aliased_model() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let r = reset_test(
            &ds,
            "y",
            &["x1", "da", "db", "dc"],
            &[2, 3],
            ResetType::Fitted,
        )
        .unwrap();
        // deviation REG-4: lmtest::resettest reports df2 = n - (ncol(X) + q) = 113 here
        // (trap_reset_fitted*), counting the aliased column; p2a uses n - rank, as anova.lm.
        Check::new(&case)
            .scalar("trap_reset_fitted_anova_f", r.statistic, Tol::stat(r.n_obs))
            .vector(
                "trap_reset_fitted_anova_df",
                &[r.df.0 as f64, r.df.1 as f64],
                Tol::exact(),
            )
            .scalar(
                "trap_reset_fitted_anova_p",
                r.p_value,
                Tol::p_value(r.n_obs),
            )
            .finish();
        assert!(
            r.warnings.iter().any(|w| w.contains("dc")),
            "{:?}",
            r.warnings
        );
    }

    /// type = regressor with a 0/1 dummy: d^2 = d^3 = d alias the model, so they add
    /// nothing; df1 counts the identified terms (2 of 4).
    #[test]
    fn test_reset_collinear_augmentation_df_by_rank() {
        let n = 40;
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let d: Vec<f64> = (0..n).map(|i| ((i * 7) % 3 == 0) as i32 as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + 0.5 * x[i] + 0.8 * d[i] + 0.1 * x[i] * x[i] + 0.2 * ((i * 13) % 5) as f64
            })
            .collect();
        let ds = Dataset::new(df! { "y" => y, "x" => x, "d" => d }.unwrap());
        let r = reset_test(&ds, "y", &["x", "d"], &[2, 3], ResetType::Regressor).unwrap();
        assert_eq!(r.df, (2, n - 5));
        assert!(
            r.warnings.iter().any(|w| w.contains("2 of 4")),
            "{:?}",
            r.warnings
        );
        assert!(r.statistic.is_finite() && r.statistic > 0.0);
    }

    /// Grunfeld (T1 case ols_diagnostics_grunfeld). (a) a restricted list longer than the
    /// unrestricted one underflowed len_u - len_r; (b) equal sets; (c) wald_test_from_ols
    /// with swapped models underflowed k_u - k_r; (d) a duplicate in the unrestricted list
    /// gave F = 0, p = 1, df (2, 196): duplicates are ignored as R's formula interface does,
    /// F = 81.98954629331581 = waldtest(m, m_value).
    #[test]
    fn test_validate_wald_rejects_duplicates() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_diagnostics_grunfeld");
        match wald_test(&ds, "inv", &["value"], &["value", "value"], true).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("strict subset"), "{message}")
            }
            e => panic!("{e:?}"),
        }
        assert!(matches!(
            wald_test(
                &ds,
                "inv",
                &["value", "capital"],
                &["capital", "value"],
                true
            ),
            Err(EconError::InvalidSpecification { .. })
        ));
        let u = run_ols(
            &ds,
            "inv",
            &["value", "capital"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let r = run_ols(&ds, "inv", &["value"], true, CovarianceType::Standard).unwrap();
        assert!(matches!(
            wald_test_from_ols(&r, &u, true),
            Err(EconError::InvalidSpecification { .. })
        ));
        let w = wald_test(
            &ds,
            "inv",
            &["value", "capital", "capital"],
            &["value"],
            true,
        )
        .unwrap();
        Check::new(&case)
            .scalar("wald_drop_capital_f", w.statistic, Tol::stat(w.n_obs))
            .scalar("wald_drop_capital_f_p", w.p_value, Tol::p_value(w.n_obs))
            .finish();
        assert_eq!((w.df1, w.df2), (1, Some(197)));
        assert!(
            w.warnings
                .iter()
                .any(|m| m.contains("capital") && m.contains("more than once")),
            "{:?}",
            w.warnings
        );
        assert!(w.to_string().contains("more than once"));
    }

    /// anova(lm(y ~ x1), lm(y ~ x1 + x2 + x3)) with x2 aliased: q = rank 3 - rank 2 = 1,
    /// not 3 - 1 = 2 names.
    #[test]
    fn test_validate_wald_aliased_rank_q() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_aliased");
        let w = wald_test(&ds, "y", &["x1", "x2", "x3"], &["x1"], true).unwrap();
        Check::new(&case)
            .scalar("wald_alias_f", w.statistic, Tol::stat(w.n_obs))
            .scalar("wald_alias_df1", w.df1 as f64, Tol::exact())
            .scalar("wald_alias_df2", w.df2.unwrap() as f64, Tol::exact())
            .scalar("wald_alias_p", w.p_value, Tol::p_value(w.n_obs))
            .finish();
    }

    /// x2 = 0.5 x1 exactly: the unrestricted model has the restricted model's rank, so
    /// there is nothing to test (before: a statistic with df1 = 1).
    #[test]
    fn test_wald_collinear_restriction_is_an_error() {
        match wald_test(&create_test_dataset(), "y", &["x1", "x2"], &["x1"], true).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("rank"), "{message}")
            }
            e => panic!("{e:?}"),
        }
    }
}
