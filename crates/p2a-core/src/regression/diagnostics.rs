//! Regression diagnostics for model validation.
//!
//! Provides tests for normality, heteroskedasticity, autocorrelation,
//! multicollinearity, and influential observations.
//!
//! All tests are implemented in pure Rust using the statrs library
//! for statistical distributions.

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
}
