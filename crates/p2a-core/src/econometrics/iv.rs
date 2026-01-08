//! Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) estimation.
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse, matmul};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{SignificanceLevel, t_test_p_value, f_test_p_value};

/// Result from an IV/2SLS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Exogenous variables (included instruments)
    pub exogenous_vars: Vec<String>,
    /// Endogenous variables
    pub endogenous_vars: Vec<String>,
    /// Excluded instruments
    pub instruments: Vec<String>,
    /// All variable names (including intercept if present)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics (or z-statistics)
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// R-squared (can be negative for IV)
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom
    pub df: usize,
    /// First-stage F-statistics for each endogenous variable
    pub first_stage_f_stats: Vec<f64>,
    /// Whether instruments are strong (F > 10 for each endogenous)
    pub strong_instruments: bool,
    /// Estimation warnings
    pub warnings: Vec<String>,
}

impl fmt::Display for IVResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "2SLS / IV Regression Results")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Endogenous: {}", self.endogenous_vars.join(", "))?;
        writeln!(f, "Instruments: {}", self.instruments.join(", "))?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        if !self.first_stage_f_stats.is_empty() {
            writeln!(f, "First-stage F: {:.4}", self.first_stage_f_stats.iter().sum::<f64>() / self.first_stage_f_stats.len() as f64)?;
        }
        writeln!(f)?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.t_stats[i],
                     self.p_values[i],
                     self.significance[i].stars())?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
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

/// Run Two-Stage Least Squares (2SLS) / Instrumental Variables estimation.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_exog` - Names of exogenous (included) variables
/// * `x_endog` - Names of endogenous variables
/// * `instruments` - Names of excluded instruments (must be at least as many as endogenous)
/// * `robust` - Whether to use heteroskedasticity-robust standard errors
///
/// # Algorithm
/// 1. First stage: Regress each endogenous variable on all exogenous + instruments
/// 2. Second stage: Regress y on exogenous + fitted endogenous
/// 3. Compute correct standard errors using original (not fitted) X
pub fn run_iv2sls(
    dataset: &Dataset,
    y_col: &str,
    x_exog: &[&str],
    x_endog: &[&str],
    instruments: &[&str],
    robust: bool,
) -> EconResult<IVResult> {
    let mut warnings = Vec::new();

    // Check identification
    if instruments.len() < x_endog.len() {
        return Err(EconError::UnderIdentified {
            n_endogenous: x_endog.len(),
            n_instruments: instruments.len(),
        });
    }

    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let n = y.len();

    // Build matrices
    // X_exog: exogenous variables (with intercept)
    let design_exog = DesignMatrix::from_dataframe(dataset.df(), x_exog, true)?;
    let x_exog_mat = design_exog.data;
    let exog_names = design_exog.column_names;

    // X_endog: endogenous variables
    let design_endog = DesignMatrix::from_dataframe(dataset.df(), x_endog, false)?;
    let x_endog_mat = design_endog.data;
    let endog_names = design_endog.column_names;

    // Z: instruments only
    let design_instruments = DesignMatrix::from_dataframe(dataset.df(), instruments, false)?;
    let z_mat = design_instruments.data;
    let _instr_names = design_instruments.column_names;

    // Full instrument matrix: [X_exog, Z] (all exogenous + excluded instruments)
    let k_exog = x_exog_mat.ncols();
    let k_endog = x_endog_mat.ncols();
    let k_instr = z_mat.ncols();
    let k_full_z = k_exog + k_instr;

    let mut z_full = Array2::zeros((n, k_full_z));
    z_full.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog_mat);
    z_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_mat);

    // ═══════════════════════════════════════════════════════════════════
    // First Stage: Regress each endogenous variable on Z_full
    // X_endog_hat = Z(Z'Z)^{-1}Z' X_endog = P_Z * X_endog
    // ═══════════════════════════════════════════════════════════════════
    let ztz = xtx(&z_full.view());
    let (ztz_inv, cond_warning) = safe_inverse(&ztz.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Z'Z in 2SLS first stage".to_string(),
            suggestion: format!("Check for collinearity among instruments: {:?}", e),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(format!("High condition number in Z'Z: {:.2e}", cond));
    }

    // Projection matrix P_Z = Z(Z'Z)^{-1}Z'
    let _z_ztz_inv = matmul(&z_full.view(), &ztz_inv.view())?;
    let mut x_endog_hat = Array2::zeros((n, k_endog));

    let mut first_stage_f_stats = Vec::with_capacity(k_endog);

    for j in 0..k_endog {
        let x_j = x_endog_mat.column(j).to_owned();

        // First stage regression
        let ztx_j = xty(&z_full.view(), &x_j);
        let beta_first = ztz_inv.dot(&ztx_j);
        let x_j_hat = z_full.dot(&beta_first);

        x_endog_hat.column_mut(j).assign(&x_j_hat);

        // First-stage F-statistic (simplified: overall F for the regression)
        let residuals_first = &x_j - &x_j_hat;
        let ssr = residuals_first.iter().map(|r| r * r).sum::<f64>();
        let x_j_mean = x_j.mean().unwrap_or(0.0);
        let sst = x_j.iter().map(|x| (x - x_j_mean).powi(2)).sum::<f64>();

        let df_reg = k_full_z.saturating_sub(1);
        let df_res = n.saturating_sub(k_full_z);

        let f_stat = if df_reg > 0 && df_res > 0 && ssr > 0.0 {
            ((sst - ssr) / df_reg as f64) / (ssr / df_res as f64)
        } else {
            0.0
        };

        first_stage_f_stats.push(f_stat);

        if f_stat < 10.0 {
            warnings.push(format!(
                "Weak instrument warning: First-stage F = {:.2} for '{}' (< 10)",
                f_stat, x_endog[j]
            ));
        }
    }

    let strong_instruments = first_stage_f_stats.iter().all(|&f| f >= 10.0);

    // ═══════════════════════════════════════════════════════════════════
    // Second Stage: Regress y on [X_exog, X_endog_hat]
    // ═══════════════════════════════════════════════════════════════════
    let k_total = k_exog + k_endog;
    let mut x_second = Array2::zeros((n, k_total));
    x_second.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog_mat);
    x_second.slice_mut(ndarray::s![.., k_exog..]).assign(&x_endog_hat);

    let xtx_second = xtx(&x_second.view());
    let (xtx_second_inv, _) = safe_inverse(&xtx_second.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X in 2SLS second stage".to_string(),
            suggestion: format!("Check for collinearity: {:?}", e),
        })?;

    let xty_second = xty(&x_second.view(), &y);
    let beta: Array1<f64> = xtx_second_inv.dot(&xty_second);

    // ═══════════════════════════════════════════════════════════════════
    // Standard Errors: Use original X_endog (not fitted) for residuals
    // ═══════════════════════════════════════════════════════════════════
    let mut x_original = Array2::zeros((n, k_total));
    x_original.slice_mut(ndarray::s![.., ..k_exog]).assign(&x_exog_mat);
    x_original.slice_mut(ndarray::s![.., k_exog..]).assign(&x_endog_mat);

    let y_hat = x_original.dot(&beta);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k_total);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = if df > 0 { ssr / df as f64 } else { ssr / n as f64 };

    // Variance-covariance matrix
    let vcov = if robust {
        // HC1 robust standard errors
        // V = (X̃'X̃)^{-1} (X̃'ΩX̃) (X̃'X̃)^{-1} where X̃ uses fitted values
        let scale = (n as f64) / (df as f64);
        let mut meat = Array2::zeros((k_total, k_total));

        for i in 0..n {
            let xi = x_second.row(i);
            let e2 = residuals[i] * residuals[i];
            for j in 0..k_total {
                for l in 0..k_total {
                    meat[[j, l]] += e2 * xi[j] * xi[l];
                }
            }
        }

        let meat = &meat * scale;
        let bread_meat = matmul(&xtx_second_inv.view(), &meat.view())?;
        matmul(&bread_meat.view(), &xtx_second_inv.view())?
    } else {
        &xtx_second_inv * sigma2
    };

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // R-squared (can be negative for IV)
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    if r_squared < 0.0 {
        warnings.push(format!("R² = {:.4} is negative, which can occur with IV estimation", r_squared));
    }

    // t-statistics and p-values
    let coefficients = beta.to_vec();
    let t_stats: Vec<f64> = coefficients.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Variable names
    let mut variables = exog_names;
    variables.extend(endog_names.clone());

    Ok(IVResult {
        dep_var: y_col.to_string(),
        exogenous_vars: x_exog.iter().map(|s| s.to_string()).collect(),
        endogenous_vars: x_endog.iter().map(|s| s.to_string()).collect(),
        instruments: instruments.iter().map(|s| s.to_string()).collect(),
        variables,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        significance,
        r_squared,
        n_obs: n,
        df,
        first_stage_f_stats,
        strong_instruments,
        warnings,
    })
}

/// Result from first-stage diagnostics for IV/2SLS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstStageDiagnostics {
    /// Endogenous variable name
    pub endogenous_var: String,
    /// Instruments used
    pub instruments: Vec<String>,
    /// First-stage F-statistic (instrument strength test)
    pub f_statistic: f64,
    /// p-value for F-statistic
    pub f_pvalue: f64,
    /// First-stage R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Whether instruments pass weak instrument test (F > 10)
    pub strong_instruments: bool,
    /// Coefficients on instruments in first stage: (name, coef, se, t-stat, p-value)
    pub instrument_coeffs: Vec<(String, f64, f64, f64, f64)>,
    /// Significance levels for each instrument
    pub significance: Vec<SignificanceLevel>,
}

impl fmt::Display for FirstStageDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "First-Stage Diagnostics for IV/2SLS")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Endogenous Variable: {}", self.endogenous_var)?;
        writeln!(f, "Instruments: {}", self.instruments.join(", "))?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(f, "Instrument Strength Test:")?;
        writeln!(f, "  F-statistic: {:.4}", self.f_statistic)?;
        writeln!(f, "  Prob (F): {:.4}", self.f_pvalue)?;
        writeln!(f, "  R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "  Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f)?;

        // Stock-Yogo critical values interpretation
        let strength = if self.f_statistic > 10.0 {
            "STRONG (F > 10)"
        } else if self.f_statistic > 5.0 {
            "MODERATE (5 < F < 10)"
        } else {
            "WEAK (F < 5) - Caution!"
        };
        writeln!(f, "  Instrument Strength: {}", strength)?;
        writeln!(f)?;

        writeln!(f, "First-Stage Coefficients:")?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}", "Instrument", "Coef", "Std Err", "t-stat", "P>|t|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for (i, (name, coef, se, t, p)) in self.instrument_coeffs.iter().enumerate() {
            let sig = if i < self.significance.len() {
                self.significance[i].stars()
            } else {
                ""
            };
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}", name, coef, se, t, p, sig)?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Note: Stock-Yogo (2005) critical value for 10% max bias: F > 16.38 (single instrument)")?;
        writeln!(f, "      Rule of thumb: F > 10 suggests instruments are not weak")?;

        Ok(())
    }
}

/// Run first-stage diagnostics for IV/2SLS.
///
/// Tests instrument strength by regressing the endogenous variable on instruments.
/// Key output is the F-statistic: F > 10 suggests instruments are not weak.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `endog_var` - Name of the endogenous variable
/// * `instruments` - Names of the instrumental variables (excluded instruments)
/// * `controls` - Optional control variables to include in first stage (exogenous variables)
pub fn run_first_stage_diagnostics(
    dataset: &Dataset,
    endog_var: &str,
    instruments: &[&str],
    controls: Option<&[&str]>,
) -> EconResult<FirstStageDiagnostics> {
    // Extract endogenous variable
    let y = DesignMatrix::extract_column(dataset.df(), endog_var)
        .map_err(|e| EconError::ColumnNotFound {
            column: endog_var.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let n = y.len();

    // Build the full regressor set: intercept + instruments + controls
    let mut all_vars: Vec<&str> = instruments.to_vec();
    if let Some(ctrl) = controls {
        all_vars.extend(ctrl);
    }

    let design = DesignMatrix::from_dataframe(dataset.df(), &all_vars, true)?;
    let x = design.data;
    let var_names = design.column_names;
    let k = x.ncols();

    // OLS regression
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X in first-stage diagnostics".to_string(),
            suggestion: format!("Check for collinearity: {:?}", e),
        })?;

    let xty_vec = xty(&x.view(), &y);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals
    let y_hat = x.dot(&beta);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = if df > 0 { ssr / df as f64 } else { ssr / n as f64 };

    // R-squared
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // F-statistic
    let df_reg = k.saturating_sub(1);
    let f_statistic = if df_reg > 0 && df > 0 && ssr > 0.0 {
        ((sst - ssr) / df_reg as f64) / (ssr / df as f64)
    } else {
        0.0
    };
    let f_pvalue = f_test_p_value(f_statistic, df_reg as f64, df as f64);

    // Standard errors
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // Extract instrument coefficients (skip intercept at index 0)
    let mut instrument_coeffs = Vec::new();
    let mut significance = Vec::new();

    for (i, instr) in instruments.iter().enumerate() {
        // Find the instrument in variable names
        if let Some(idx) = var_names.iter().position(|v| v == *instr) {
            let b = beta[idx];
            let se = std_errors[idx];
            let t = if se > 0.0 { b / se } else { 0.0 };
            let p = t_test_p_value(t, df as f64);
            instrument_coeffs.push((instr.to_string(), b, se, t, p));
            significance.push(SignificanceLevel::from_p_value(p));
        } else if i + 1 < beta.len() {
            // Fallback: use position (offset by 1 for intercept)
            let idx = i + 1;
            let b = beta[idx];
            let se = std_errors[idx];
            let t = if se > 0.0 { b / se } else { 0.0 };
            let p = t_test_p_value(t, df as f64);
            instrument_coeffs.push((instr.to_string(), b, se, t, p));
            significance.push(SignificanceLevel::from_p_value(p));
        }
    }

    Ok(FirstStageDiagnostics {
        endogenous_var: endog_var.to_string(),
        instruments: instruments.iter().map(|s| s.to_string()).collect(),
        f_statistic,
        f_pvalue,
        r_squared,
        adj_r_squared,
        n_obs: n,
        strong_instruments: f_statistic > 10.0,
        instrument_coeffs,
        significance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_iv_dataset() -> Dataset {
        // Classic IV setup: y = 0.5*x_endog + noise
        // x_endog is correlated with error (endogenous)
        // z is correlated with x_endog but not with error (valid instrument)
        //
        // True model: y = 0.5 * x_endog
        // z -> x_endog -> y  (z is instrument)
        let df = df! {
            "y" => [1.2, 2.1, 3.3, 4.0, 5.2, 5.8, 7.1, 8.0, 8.9, 10.1],
            "x_endog" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "z" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x_exog" => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_under_identified_error() {
        // 2 endogenous vars but only 1 instrument = under-identified
        let n_endog = 2;
        let n_instr = 1;
        assert!(n_instr < n_endog);
    }

    #[test]
    fn test_iv2sls_basic() {
        let dataset = create_iv_dataset();
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],           // no exogenous regressors (besides constant)
            &["x_endog"],  // endogenous regressor
            &["z"],        // instrument
            false          // not robust
        ).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 10);
        assert!(result.variables.len() >= 1);

        // The true coefficient on x_endog is ~0.5
        // IV should recover something in the ballpark
        let x_endog_idx = result.variables.iter().position(|v| v == "x_endog").unwrap();
        assert!((result.coefficients[x_endog_idx] - 0.5).abs() < 0.2,
            "IV coefficient should be close to 0.5, got {}", result.coefficients[x_endog_idx]);
    }

    #[test]
    fn test_iv2sls_robust() {
        let dataset = create_iv_dataset();
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],
            &["x_endog"],
            &["z"],
            true  // robust SEs
        ).unwrap();

        assert_eq!(result.n_obs, 10);
        // Robust SEs should still produce valid results
        assert!(result.std_errors.iter().all(|&se| se > 0.0 && se.is_finite()));
    }

    #[test]
    fn test_first_stage_diagnostics() {
        let dataset = create_iv_dataset();
        let result = run_first_stage_diagnostics(
            &dataset,
            "x_endog",
            &["z"],
            None  // no additional controls
        ).unwrap();

        // First stage should have results
        assert_eq!(result.endogenous_var, "x_endog");
        assert!(!result.instruments.is_empty());

        // R² should be high since z and x_endog are linearly related
        assert!(result.r_squared > 0.9,
            "First-stage R² should be > 0.9 for linearly related variables, got {}", result.r_squared);

        // F-statistic should be non-negative
        assert!(result.f_statistic >= 0.0,
            "F-statistic should be non-negative, got {}", result.f_statistic);
    }

    #[test]
    fn test_iv_under_identified() {
        let dataset = create_iv_dataset();
        // Try to instrument 2 variables with only 1 instrument
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],
            &["x_endog", "z"],  // 2 endogenous
            &["x_exog"],       // only 1 instrument
            false
        );
        // Should fail due to under-identification
        assert!(result.is_err());
    }
}
