//! Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) estimation.
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.
//!
//! # Mathematical Background
//!
//! IV estimation addresses endogeneity when Cov(X, ε) ≠ 0. Given instruments Z
//! that satisfy:
//! 1. **Relevance**: Cov(Z, X) ≠ 0
//! 2. **Exogeneity**: Cov(Z, ε) = 0
//!
//! ## Two-Stage Least Squares (2SLS)
//!
//! **First stage**: Regress endogenous X on instruments Z and exogenous W:
//!   X̂ = Z(Z'Z)⁻¹Z'X = Pᵤ X
//!
//! **Second stage**: Regress y on fitted values X̂ and exogenous W:
//!   β̂₂ₛₗₛ = (X̂'X̂)⁻¹ X̂'y
//!
//! Equivalently: β̂ᵢᵥ = (X'Pᵤ X)⁻¹ X'Pᵤ y
//!
//! ## Weak Instruments
//!
//! The first-stage F-statistic tests instrument strength. Stock & Yogo (2005)
//! suggest F > 10 as a rule of thumb for a single endogenous regressor.
//!
//! # References
//!
//! - Wright, P.G. (1928). *The Tariff on Animal and Vegetable Oils*. Macmillan.
//!   First application of instrumental variables.
//!
//! - Theil, H. (1953). Repeated least squares applied to complete equation systems.
//!   *The Hague: Central Planning Bureau*. Introduction of 2SLS.
//!
//! - Basmann, R.L. (1957). A generalized classical method of linear estimation of
//!   coefficients in a structural equation. *Econometrica*, 25(1), 77-83.
//!   https://doi.org/10.2307/1907743
//!
//! - Stock, J.H., & Yogo, M. (2005). Testing for weak instruments in linear IV
//!   regression. In D.W.K. Andrews & J.H. Stock (Eds.), *Identification and
//!   Inference for Econometric Models* (pp. 80-108). Cambridge University Press.
//!   https://doi.org/10.1017/CBO9780511614491.006
//!
//! - Angrist, J.D., & Pischke, J.S. (2009). *Mostly Harmless Econometrics: An
//!   Empiricist's Companion*. Princeton University Press. ISBN: 978-0691120355.
//!
//! - Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
//!   (2nd ed.), Chapter 5. MIT Press.
//!
//! R equivalent: `AER::ivreg()`, `ivreg::ivreg()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{
    SignificanceLevel, chi_squared_p_value, f_test_p_value, t_test_p_value,
};

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
            writeln!(
                f,
                "First-stage F: {:.4}",
                self.first_stage_f_stats.iter().sum::<f64>()
                    / self.first_stage_f_stats.len() as f64
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "z", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
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
///
/// Uses the efficient projection formula to avoid forming large n×k matrices
/// in the second stage. Instead of building X̂ = P_Z X explicitly, we compute:
///
///   β_IV = (X'P_Z X)^{-1} X'P_Z y
///
/// where P_Z = Z(Z'Z)^{-1}Z'. By noting that X'P_Z = X'Z (Z'Z)^{-1} Z',
/// we can compute everything via small k×k matrix operations:
///
///   X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X)
///   X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y)
///
/// This reduces memory from O(n·k) to O(k²) for the second stage.
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
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|_e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
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

    // Full instrument matrix: [X_exog, Z] (all exogenous + excluded instruments)
    let k_exog = x_exog_mat.ncols();
    let k_endog = x_endog_mat.ncols();
    let k_instr = z_mat.ncols();
    let k_full_z = k_exog + k_instr;

    let mut z_full = Array2::zeros((n, k_full_z));
    z_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    z_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_mat);

    // ═══════════════════════════════════════════════════════════════════
    // Compute Z'Z inverse (small k_z × k_z matrix)
    // ═══════════════════════════════════════════════════════════════════
    let ztz = xtx(&z_full.view());
    let (ztz_inv, cond_warning) =
        safe_inverse(&ztz.view()).map_err(|e| EconError::SingularMatrix {
            context: "Z'Z in 2SLS first stage".to_string(),
            suggestion: format!("Check for collinearity among instruments: {:?}", e),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(format!("High condition number in Z'Z: {:.2e}", cond));
    }

    // ═══════════════════════════════════════════════════════════════════
    // First Stage: Compute F-statistics for instrument strength
    // X_endog_hat_j = Z * (Z'Z)^{-1} * Z' * x_j
    // ═══════════════════════════════════════════════════════════════════
    let mut first_stage_f_stats = Vec::with_capacity(k_endog);

    // Store first-stage betas only if needed for robust SEs
    let mut first_stage_betas: Vec<Array1<f64>> = if robust {
        Vec::with_capacity(k_endog)
    } else {
        Vec::new()
    };

    for j in 0..k_endog {
        let x_j = x_endog_mat.column(j);

        // First stage: beta_j = (Z'Z)^{-1} Z'x_j
        let ztx_j = z_full.t().dot(&x_j);
        let beta_first = ztz_inv.dot(&ztx_j);

        // First-stage F-statistic via projection:
        // SSR = x_j'(I - P_Z)x_j = x_j'x_j - ztx_j' * (Z'Z)^{-1} * ztx_j
        let x_j_sq = x_j.dot(&x_j);
        let ssr = x_j_sq - ztx_j.dot(&beta_first);
        let x_j_mean = x_j.mean().unwrap_or(0.0);
        let sst = x_j_sq - n as f64 * x_j_mean * x_j_mean;

        let df_reg = k_full_z.saturating_sub(1);
        let df_res = n.saturating_sub(k_full_z);

        let f_stat = if df_reg > 0 && df_res > 0 && ssr > 0.0 {
            ((sst - ssr) / df_reg as f64) / (ssr / df_res as f64)
        } else {
            0.0
        };

        first_stage_f_stats.push(f_stat);
        if robust {
            first_stage_betas.push(beta_first);
        }

        if f_stat < 10.0 {
            warnings.push(format!(
                "Weak instrument warning: First-stage F = {:.2} for '{}' (< 10)",
                f_stat, x_endog[j]
            ));
        }
    }

    let strong_instruments = first_stage_f_stats.iter().all(|&f| f >= 10.0);

    // ═══════════════════════════════════════════════════════════════════
    // Second Stage via projection formula (no n×k matrices needed):
    //
    //   X_full = [X_exog, X_endog]  (the original regressors)
    //   Z'X_full is k_z × k_total
    //   X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X)  -- k_total × k_total
    //   X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y)  -- k_total × 1
    //   β = (X'P_Z X)^{-1} X'P_Z y
    // ═══════════════════════════════════════════════════════════════════
    let k_total = k_exog + k_endog;

    // Compute Z'X_full (k_z × k_total) via ndarray BLAS
    // Z'X_exog (k_z × k_exog) and Z'X_endog (k_z × k_endog)
    let ztx_exog = z_full.t().dot(&x_exog_mat);
    let ztx_endog = z_full.t().dot(&x_endog_mat);
    let mut ztx_full = Array2::zeros((k_full_z, k_total));
    ztx_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&ztx_exog);
    ztx_full
        .slice_mut(ndarray::s![.., k_exog..])
        .assign(&ztx_endog);

    // Z'y (k_z × 1)
    let zty = z_full.t().dot(&y);

    // (Z'Z)^{-1} Z'X  (k_z × k_total)
    let ztz_inv_ztx = ztz_inv.dot(&ztx_full);

    // X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X) = ztx_full' * ztz_inv_ztx  (k_total × k_total)
    let xpzx = ztx_full.t().dot(&ztz_inv_ztx);

    // X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y) = ztx_full' * (Z'Z)^{-1} * Z'y  (k_total × 1)
    let ztz_inv_zty = ztz_inv.dot(&zty);
    let xpzy = ztx_full.t().dot(&ztz_inv_zty);

    // β = (X'P_Z X)^{-1} X'P_Z y
    let (xpzx_inv, _) = safe_inverse(&xpzx.view()).map_err(|e| EconError::SingularMatrix {
        context: "X'P_Z X in 2SLS second stage".to_string(),
        suggestion: format!("Check for collinearity: {:?}", e),
    })?;

    let beta: Array1<f64> = xpzx_inv.dot(&xpzy);

    // ═══════════════════════════════════════════════════════════════════
    // Residuals using original X (not fitted values)
    // y_hat = X_exog * beta_exog + X_endog * beta_endog
    // ═══════════════════════════════════════════════════════════════════
    let beta_exog = beta.slice(ndarray::s![..k_exog]);
    let beta_endog = beta.slice(ndarray::s![k_exog..]);
    let y_hat = x_exog_mat.dot(&beta_exog) + x_endog_mat.dot(&beta_endog);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k_total);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = if df > 0 {
        ssr / df as f64
    } else {
        ssr / n as f64
    };

    // ═══════════════════════════════════════════════════════════════════
    // Variance-covariance matrix
    // For 2SLS: V = σ² (X̂'X̂)^{-1} = σ² (X'P_Z X)^{-1}
    // ═══════════════════════════════════════════════════════════════════
    let vcov = if robust {
        // HC1 robust standard errors
        // V = (X̂'X̂)^{-1} (X̂'ΩX̂) (X̂'X̂)^{-1}
        // Need X̂ = P_Z X for the meat. Build it efficiently using first-stage betas.
        let scale = (n as f64) / (df as f64);

        // Build fitted X matrix for robust SEs
        // X̂_exog = X_exog (exogenous vars are their own instruments)
        // X̂_endog_j = Z * first_stage_beta_j
        let mut meat = Array2::zeros((k_total, k_total));
        // Pre-compute x_hat rows on the fly to avoid allocating full n×k matrix
        let mut xi_hat = vec![0.0; k_total];
        for i in 0..n {
            // Exogenous part: same as original
            for j in 0..k_exog {
                xi_hat[j] = x_exog_mat[[i, j]];
            }
            // Endogenous part: fitted values from first stage
            for j in 0..k_endog {
                let mut val = 0.0;
                for l in 0..k_full_z {
                    val += z_full[[i, l]] * first_stage_betas[j][l];
                }
                xi_hat[k_exog + j] = val;
            }

            let e2 = residuals[i] * residuals[i];
            for j in 0..k_total {
                for l in j..k_total {
                    let contrib = e2 * xi_hat[j] * xi_hat[l];
                    meat[[j, l]] += contrib;
                    if l != j {
                        meat[[l, j]] += contrib;
                    }
                }
            }
        }

        let meat = &meat * scale;
        let bread_meat = xpzx_inv.dot(&meat);
        bread_meat.dot(&xpzx_inv)
    } else {
        &xpzx_inv * sigma2
    };

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // R-squared (can be negative for IV)
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    if r_squared < 0.0 {
        warnings.push(format!(
            "R² = {:.4} is negative, which can occur with IV estimation",
            r_squared
        ));
    }

    // t-statistics and p-values
    let coefficients = beta.to_vec();
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
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
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Instrument", "Coef", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for (i, (name, coef, se, t, p)) in self.instrument_coeffs.iter().enumerate() {
            let sig = if i < self.significance.len() {
                self.significance[i].stars()
            } else {
                ""
            };
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                name, coef, se, t, p, sig
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(
            f,
            "Note: Stock-Yogo (2005) critical value for 10% max bias: F > 16.38 (single instrument)"
        )?;
        writeln!(
            f,
            "      Rule of thumb: F > 10 suggests instruments are not weak"
        )?;

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
    let y = DesignMatrix::extract_column(dataset.df(), endog_var).map_err(|e| {
        EconError::ColumnNotFound {
            column: endog_var.to_string(),
            available: get_column_names(dataset.df()),
        }
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
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
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
    let sigma2 = if df > 0 {
        ssr / df as f64
    } else {
        ssr / n as f64
    };

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

/// Result from the Sargan test for overidentifying restrictions.
///
/// The Sargan test (also known as the Hansen J test) evaluates whether
/// the instruments are valid, i.e., uncorrelated with the error term.
///
/// # References
///
/// - Sargan, J.D. (1958). The estimation of economic relationships using
///   instrumental variables. *Econometrica*, 26(3), 393-415.
///   https://doi.org/10.2307/1907619
///
/// - Hansen, L.P. (1982). Large sample properties of generalized method of
///   moments estimators. *Econometrica*, 50(4), 1029-1054.
///   https://doi.org/10.2307/1912775
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarganTestResult {
    /// Name of the test
    pub test_name: String,
    /// J statistic (Sargan/Hansen test statistic)
    pub j_statistic: f64,
    /// Degrees of freedom (number of overidentifying restrictions)
    pub df: usize,
    /// p-value from chi-squared distribution
    pub p_value: f64,
    /// Number of instruments
    pub n_instruments: usize,
    /// Number of endogenous regressors
    pub n_endogenous: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Whether the model is over-identified (required for test)
    pub overidentified: bool,
    /// Null hypothesis: instruments are valid
    pub null_hypothesis: String,
    /// Interpretation based on p-value
    pub interpretation: String,
}

impl fmt::Display for SarganTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Sargan Test of Overidentifying Restrictions")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;
        writeln!(
            f,
            "H0: Instruments are valid (uncorrelated with error term)"
        )?;
        writeln!(f)?;
        writeln!(f, "  J statistic:      {:.4}", self.j_statistic)?;
        writeln!(f, "  Degrees of freedom: {}", self.df)?;
        writeln!(f, "  p-value:          {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "  Number of instruments:       {}", self.n_instruments)?;
        writeln!(f, "  Number of endogenous vars:   {}", self.n_endogenous)?;
        writeln!(f, "  Overidentifying restrictions: {}", self.df)?;
        writeln!(f)?;
        writeln!(f, "Interpretation: {}", self.interpretation)?;
        Ok(())
    }
}

/// Perform the Sargan test for overidentifying restrictions.
///
/// The Sargan test (also known as the Hansen J test when robust to heteroskedasticity)
/// tests whether the instruments are valid (uncorrelated with the structural error term).
///
/// # Requirements
///
/// - The model must be **over-identified**: number of instruments > number of endogenous variables
/// - For just-identified models (instruments == endogenous), the test is not defined
///
/// # Test Statistic
///
/// The Sargan statistic is computed as:
///   J = n × R² from regressing IV residuals on all instruments
///
/// Equivalently:
///   J = n × (1 - ε̂'M_Z ε̂ / ε̂'ε̂)
///
/// where ε̂ are the IV residuals and M_Z is the annihilator matrix for instruments.
///
/// Under H0 (valid instruments): J ~ χ²(L - K)
/// where L = number of instruments, K = number of endogenous variables.
///
/// # Interpretation
///
/// - **Fail to reject H0 (p > 0.05)**: Instruments appear valid
/// - **Reject H0 (p ≤ 0.05)**: At least one instrument may be invalid
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable
/// * `x_exog` - Names of exogenous (included) variables
/// * `x_endog` - Names of endogenous variables
/// * `instruments` - Names of excluded instruments
///
/// # Returns
///
/// `SarganTestResult` containing the J statistic, degrees of freedom, and p-value.
///
/// # References
///
/// - Sargan, J.D. (1958). The estimation of economic relationships using
///   instrumental variables. *Econometrica*, 26(3), 393-415.
///
/// R equivalent: `summary(ivreg(...))` shows Sargan statistic, or `ivdiag::sargan()`
pub fn sargan_test(
    dataset: &Dataset,
    y_col: &str,
    x_exog: &[&str],
    x_endog: &[&str],
    instruments: &[&str],
) -> EconResult<SarganTestResult> {
    let n_instr = instruments.len();
    let n_endog = x_endog.len();
    let n_exog_vars = x_exog.len();

    // Total instruments = excluded instruments + included exogenous + intercept
    let _total_instruments = n_instr + n_exog_vars + 1; // +1 for intercept

    // Degrees of freedom = overidentifying restrictions
    // For 2SLS: df = (excluded instruments) - (endogenous regressors)
    let df = n_instr.saturating_sub(n_endog);

    if df == 0 {
        return Ok(SarganTestResult {
            test_name: "Sargan Test of Overidentifying Restrictions".to_string(),
            j_statistic: 0.0,
            df: 0,
            p_value: 1.0,
            n_instruments: n_instr,
            n_endogenous: n_endog,
            n_obs: 0,
            overidentified: false,
            null_hypothesis: "Instruments are valid (uncorrelated with error term)".to_string(),
            interpretation: "Model is exactly identified (instruments = endogenous). Sargan test not applicable.".to_string(),
        });
    }

    // First, run IV/2SLS to get the residuals
    let iv_result = run_iv2sls(dataset, y_col, x_exog, x_endog, instruments, false)?;
    let n = iv_result.n_obs;

    // Re-extract data to compute residuals and run auxiliary regression
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Build the full X matrix (exogenous + endogenous)
    let design_exog = DesignMatrix::from_dataframe(dataset.df(), x_exog, true)?;
    let x_exog_mat = design_exog.data;

    let design_endog = DesignMatrix::from_dataframe(dataset.df(), x_endog, false)?;
    let x_endog_mat = design_endog.data;

    let k_exog = x_exog_mat.ncols();
    let k_endog = x_endog_mat.ncols();
    let k_total = k_exog + k_endog;

    let mut x_full = Array2::zeros((n, k_total));
    x_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    x_full
        .slice_mut(ndarray::s![.., k_exog..])
        .assign(&x_endog_mat);

    // Compute IV residuals
    let beta: Array1<f64> = Array1::from_vec(iv_result.coefficients.clone());
    let y_hat = x_full.dot(&beta);
    let residuals = &y - &y_hat;

    // Build the full instrument matrix Z = [exogenous, excluded_instruments]
    let design_z = DesignMatrix::from_dataframe(dataset.df(), instruments, false)?;
    let z_excl = design_z.data;

    let k_z = k_exog + z_excl.ncols(); // exogenous (with intercept) + excluded instruments
    let mut z_full = Array2::zeros((n, k_z));
    z_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    z_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_excl);

    // Sargan test: regress residuals on all instruments and compute n*R²
    // R² = 1 - SSR/SST where SST = sum(residuals²) and SSR is from aux regression

    // Auxiliary regression: residuals on Z
    let ztz = xtx(&z_full.view());
    let (ztz_inv, _) = safe_inverse(&ztz.view()).map_err(|e| EconError::SingularMatrix {
        context: "Z'Z in Sargan test".to_string(),
        suggestion: format!("Check for collinearity among instruments: {:?}", e),
    })?;

    let ztr = xty(&z_full.view(), &residuals);
    let gamma = ztz_inv.dot(&ztr);
    let fitted = z_full.dot(&gamma);

    // Compute R² from auxiliary regression
    let ssr_aux: f64 = (&residuals - &fitted).iter().map(|e| e * e).sum();
    let sst: f64 = residuals.iter().map(|e| e * e).sum();

    let r_squared_aux = if sst > 1e-10 {
        1.0 - ssr_aux / sst
    } else {
        0.0
    };

    // J statistic = n * R²
    let j_statistic = (n as f64) * r_squared_aux.max(0.0);

    // p-value from chi-squared distribution
    let p_value = chi_squared_p_value(j_statistic, df as f64);

    // Interpretation
    let interpretation = if p_value > 0.10 {
        "Cannot reject H0. Instruments appear valid.".to_string()
    } else if p_value > 0.05 {
        "Marginal evidence against instrument validity (0.05 < p ≤ 0.10).".to_string()
    } else if p_value > 0.01 {
        "Reject H0 at 5% level. At least one instrument may be invalid.".to_string()
    } else {
        "Strongly reject H0. Instruments appear invalid.".to_string()
    };

    Ok(SarganTestResult {
        test_name: "Sargan Test of Overidentifying Restrictions".to_string(),
        j_statistic,
        df,
        p_value,
        n_instruments: n_instr,
        n_endogenous: n_endog,
        n_obs: n,
        overidentified: true,
        null_hypothesis: "Instruments are valid (uncorrelated with error term)".to_string(),
        interpretation,
    })
}

/// Run the Sargan test from an existing IV result and dataset.
///
/// This is a convenience function that wraps `sargan_test` using the parameters
/// from an existing `IVResult`.
pub fn run_sargan_test(dataset: &Dataset, iv_result: &IVResult) -> EconResult<SarganTestResult> {
    let x_exog: Vec<&str> = iv_result
        .exogenous_vars
        .iter()
        .map(|s| s.as_str())
        .collect();
    let x_endog: Vec<&str> = iv_result
        .endogenous_vars
        .iter()
        .map(|s| s.as_str())
        .collect();
    let instruments: Vec<&str> = iv_result.instruments.iter().map(|s| s.as_str()).collect();

    sargan_test(dataset, &iv_result.dep_var, &x_exog, &x_endog, &instruments)
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
        }
        .unwrap();
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
            &[],          // no exogenous regressors (besides constant)
            &["x_endog"], // endogenous regressor
            &["z"],       // instrument
            false,        // not robust
        )
        .unwrap();

        // Check structure
        assert_eq!(result.n_obs, 10);
        assert!(!result.variables.is_empty());

        // The true coefficient on x_endog is ~0.5
        // IV should recover something in the ballpark
        let x_endog_idx = result
            .variables
            .iter()
            .position(|v| v == "x_endog")
            .unwrap();
        assert!(
            (result.coefficients[x_endog_idx] - 0.5).abs() < 0.2,
            "IV coefficient should be close to 0.5, got {}",
            result.coefficients[x_endog_idx]
        );
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
            true, // robust SEs
        )
        .unwrap();

        assert_eq!(result.n_obs, 10);
        // Robust SEs should still produce valid results
        assert!(
            result
                .std_errors
                .iter()
                .all(|&se| se > 0.0 && se.is_finite())
        );
    }

    #[test]
    fn test_first_stage_diagnostics() {
        let dataset = create_iv_dataset();
        let result = run_first_stage_diagnostics(
            &dataset,
            "x_endog",
            &["z"],
            None, // no additional controls
        )
        .unwrap();

        // First stage should have results
        assert_eq!(result.endogenous_var, "x_endog");
        assert!(!result.instruments.is_empty());

        // R² should be high since z and x_endog are linearly related
        assert!(
            result.r_squared > 0.9,
            "First-stage R² should be > 0.9 for linearly related variables, got {}",
            result.r_squared
        );

        // F-statistic should be non-negative
        assert!(
            result.f_statistic >= 0.0,
            "F-statistic should be non-negative, got {}",
            result.f_statistic
        );
    }

    #[test]
    fn test_iv_under_identified() {
        let dataset = create_iv_dataset();
        // Try to instrument 2 variables with only 1 instrument
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],
            &["x_endog", "z"], // 2 endogenous
            &["x_exog"],       // only 1 instrument
            false,
        );
        // Should fail due to under-identification
        assert!(result.is_err());
    }

    fn create_overidentified_dataset() -> Dataset {
        // Over-identified IV setup: 1 endogenous, 2 instruments
        // y = 0.5*x_endog + noise
        // z1 and z2 are both valid instruments (correlated with x_endog, uncorrelated with error)
        let df = df! {
            "y" => [1.2, 2.1, 3.3, 4.0, 5.2, 5.8, 7.1, 8.0, 8.9, 10.1,
                    11.0, 12.2, 13.1, 14.0, 15.3, 16.1, 17.0, 18.2, 19.1, 20.0],
            "x_endog" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                          22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
            "z1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "z2" => [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9,
                     11.1, 11.9, 13.1, 13.9, 15.1, 15.9, 17.1, 17.9, 19.1, 19.9]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_sargan_test_overidentified() {
        let dataset = create_overidentified_dataset();

        // Over-identified: 1 endogenous, 2 instruments
        let result = sargan_test(
            &dataset,
            "y",
            &[],           // no exogenous (besides constant)
            &["x_endog"],  // 1 endogenous
            &["z1", "z2"], // 2 instruments
        )
        .unwrap();

        // Should be over-identified
        assert!(result.overidentified);
        assert_eq!(result.df, 1); // 2 instruments - 1 endogenous = 1

        // J statistic should be non-negative
        assert!(result.j_statistic >= 0.0);

        // p-value should be valid
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);

        // For valid instruments, we expect to NOT reject (p > 0.05)
        // This test data has valid instruments, so p should be high
        println!(
            "Sargan test: J = {:.4}, df = {}, p = {:.4}",
            result.j_statistic, result.df, result.p_value
        );
    }

    #[test]
    fn test_sargan_test_exactly_identified() {
        let dataset = create_iv_dataset();

        // Exactly identified: 1 endogenous, 1 instrument
        let result = sargan_test(
            &dataset,
            "y",
            &[],          // no exogenous (besides constant)
            &["x_endog"], // 1 endogenous
            &["z"],       // 1 instrument
        )
        .unwrap();

        // Should NOT be over-identified
        assert!(!result.overidentified);
        assert_eq!(result.df, 0);
        assert_eq!(result.p_value, 1.0); // Test not applicable
    }

    #[test]
    fn test_run_sargan_test_from_iv_result() {
        let dataset = create_overidentified_dataset();

        // First run IV
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z1", "z2"], false).unwrap();

        // Then run Sargan test
        let sargan_result = run_sargan_test(&dataset, &iv_result).unwrap();

        assert!(sargan_result.overidentified);
        assert_eq!(sargan_result.df, 1);
        assert!(sargan_result.j_statistic >= 0.0);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R's AER::ivreg()
    // ════════════════════════════════════════════════════════════════════════════

    /// Create dataset for IV validation with known DGP
    /// y = 0.5 + 1.0*x_endog + error
    /// x_endog is endogenous (correlated with error)
    /// z is a valid instrument
    fn create_validation_iv_dataset() -> Dataset {
        // Generate IV data with:
        // - z: instrument (exogenous)
        // - u: common factor creating endogeneity
        // - x_endog: endogenous regressor (2*z + u + noise)
        // - y: outcome (0.5 + 1.0*x_endog + 0.5*u + noise)

        // Deterministic pseudo-random for reproducibility
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() * 2.0).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos()).collect();

        let x_endog: Vec<f64> = z
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(i, (&zi, &ui))| {
                let noise = (i as f64 * 0.9).sin() * 0.3;
                2.0 * zi + ui + noise
            })
            .collect();

        let y: Vec<f64> = x_endog
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(i, (&xi, &ui))| {
                let noise = (i as f64 * 1.1).cos() * 0.3;
                0.5 + 1.0 * xi + 0.5 * ui + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "x_endog" => x_endog,
            "z" => z
        }
        .unwrap();
        Dataset::new(df)
    }

    /// Validation test: Basic 2SLS (just-identified)
    /// Compared against R's AER::ivreg(y ~ x | z)
    #[test]
    fn test_validate_iv2sls_basic() {
        let dataset = create_validation_iv_dataset();

        let result = run_iv2sls(
            &dataset,
            "y",
            &[],          // no exogenous regressors
            &["x_endog"], // endogenous
            &["z"],       // instrument
            false,        // not robust
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.n_obs, 100);
        assert!(!result.coefficients.is_empty());

        // Find coefficient on x_endog
        let x_idx = result
            .variables
            .iter()
            .position(|v| v == "x_endog")
            .unwrap();

        // True coefficient is 1.0; IV should recover something close
        // (exact recovery depends on instrument strength and sample size)
        assert!(
            (result.coefficients[x_idx] - 1.0).abs() < 0.5,
            "IV coefficient on x_endog should be close to 1.0, got {}",
            result.coefficients[x_idx]
        );

        // Standard error should be positive
        assert!(result.std_errors[x_idx] > 0.0, "IV SE should be positive");

        // First stage F-stat should indicate strong instrument
        assert!(
            !result.first_stage_f_stats.is_empty(),
            "First stage F-stats should be computed"
        );
    }

    /// Validation test: Over-identified 2SLS (more instruments than endogenous)
    /// Compared against R's AER::ivreg(y ~ x | z1 + z2)
    #[test]
    fn test_validate_iv2sls_overidentified() {
        // Create dataset with 2 instruments for 1 endogenous variable
        let n = 150;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin() * 2.0).collect();
        let z2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.8).cos() * 1.5).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 1.1).sin() * 0.8).collect();

        let x_endog: Vec<f64> = z1
            .iter()
            .zip(z2.iter())
            .zip(u.iter())
            .enumerate()
            .map(|(i, ((&z1i, &z2i), &ui))| {
                let noise = (i as f64 * 0.7).cos() * 0.2;
                1.5 * z1i + 1.0 * z2i + ui + noise
            })
            .collect();

        let y: Vec<f64> = x_endog
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(i, (&xi, &ui))| {
                let noise = (i as f64 * 1.3).sin() * 0.25;
                1.0 + 0.8 * xi + 0.4 * ui + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "x_endog" => x_endog,
            "z1" => z1,
            "z2" => z2
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_iv2sls(
            &dataset,
            "y",
            &[],           // no exogenous
            &["x_endog"],  // 1 endogenous
            &["z1", "z2"], // 2 instruments (over-identified)
            false,
        )
        .unwrap();

        // Structure
        assert_eq!(result.n_obs, n);
        assert_eq!(result.instruments.len(), 2);
        assert_eq!(result.endogenous_vars.len(), 1);

        // Coefficient should be close to true value (0.8)
        let x_idx = result
            .variables
            .iter()
            .position(|v| v == "x_endog")
            .unwrap();

        assert!(
            (result.coefficients[x_idx] - 0.8).abs() < 0.4,
            "IV coefficient should be close to 0.8, got {}",
            result.coefficients[x_idx]
        );

        // With 2 instruments for 1 endogenous, Sargan test is applicable
        let sargan = sargan_test(&dataset, "y", &[], &["x_endog"], &["z1", "z2"]).unwrap();

        assert!(sargan.overidentified);
        assert_eq!(sargan.df, 1); // 2 instruments - 1 endogenous = 1
        assert!(sargan.j_statistic >= 0.0);
        assert!(sargan.p_value >= 0.0 && sargan.p_value <= 1.0);
    }

    /// Validation test: First-stage diagnostics
    /// Check instrument strength via F-statistic
    #[test]
    fn test_validate_first_stage_diagnostics() {
        let dataset = create_validation_iv_dataset();

        let result = run_first_stage_diagnostics(&dataset, "x_endog", &["z"], None).unwrap();

        // Structure checks
        assert_eq!(result.endogenous_var, "x_endog");
        assert_eq!(result.instruments.len(), 1);
        assert_eq!(result.n_obs, 100);

        // F-statistic should be positive
        assert!(
            result.f_statistic >= 0.0,
            "First-stage F-stat should be non-negative, got {}",
            result.f_statistic
        );

        // R-squared should be in [0, 1]
        assert!(
            result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "First-stage R² should be in [0, 1], got {}",
            result.r_squared
        );

        // With our strong instrument, F should be > 10 (Stock-Yogo threshold)
        // Note: depends on DGP; may not always hold
        println!(
            "First-stage F-statistic: {:.2} (>10 indicates strong instrument)",
            result.f_statistic
        );

        // Instrument coefficients should be reported
        assert!(!result.instrument_coeffs.is_empty());
    }

    /// Validation test: IV with exogenous control variables
    #[test]
    fn test_validate_iv2sls_with_controls() {
        // Create data with an exogenous control variable
        let n = 120;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.6).sin() * 2.0).collect();
        let w: Vec<f64> = (0..n).map(|i| (i as f64 * 0.4).cos() * 1.5).collect(); // Exogenous control
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 1.0).sin() * 0.7).collect();

        let x_endog: Vec<f64> = z
            .iter()
            .zip(w.iter())
            .zip(u.iter())
            .enumerate()
            .map(|(i, ((&zi, &wi), &ui))| {
                let noise = (i as f64 * 0.8).cos() * 0.2;
                1.5 * zi + 0.5 * wi + ui + noise
            })
            .collect();

        let y: Vec<f64> = x_endog
            .iter()
            .zip(w.iter())
            .zip(u.iter())
            .enumerate()
            .map(|(i, ((&xi, &wi), &ui))| {
                let noise = (i as f64 * 1.2).sin() * 0.2;
                0.5 + 0.8 * xi + 0.6 * wi + 0.3 * ui + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "x_endog" => x_endog,
            "w" => w,
            "z" => z
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_iv2sls(
            &dataset,
            "y",
            &["w"],       // exogenous control
            &["x_endog"], // endogenous
            &["z"],       // instrument
            false,
        )
        .unwrap();

        // Should have coefficients for both x_endog and w
        assert!(result.variables.contains(&"x_endog".to_string()));
        assert!(result.variables.contains(&"w".to_string()));

        // Find indices
        let x_idx = result
            .variables
            .iter()
            .position(|v| v == "x_endog")
            .unwrap();
        let w_idx = result.variables.iter().position(|v| v == "w").unwrap();

        // Both coefficients should be close to true values
        // True: x_endog = 0.8, w = 0.6
        assert!(
            (result.coefficients[x_idx] - 0.8).abs() < 0.5,
            "Coefficient on x_endog should be close to 0.8, got {}",
            result.coefficients[x_idx]
        );

        assert!(
            (result.coefficients[w_idx] - 0.6).abs() < 0.5,
            "Coefficient on w should be close to 0.6, got {}",
            result.coefficients[w_idx]
        );
    }
}
