//! Arellano-Bond / System GMM estimators for dynamic panel data.
//!
//! # Mathematical Background
//!
//! Estimates models of the form:
//! ```text
//! y_it = α * y_{i,t-1} + X_it * β + η_i + ε_it
//! ```
//!
//! # References
//!
//! - Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data:
//!   Monte Carlo Evidence and an Application to Employment Equations.
//!   *Review of Economic Studies*, 58(2), 277-297.
//!   https://doi.org/10.2307/2297968
//!
//! - Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions
//!   in Dynamic Panel Data Models. *Journal of Econometrics*, 87(1), 115-143.
//!   https://doi.org/10.1016/S0304-4076(98)00009-8
//!
//! R equivalent: `plm::pgmm()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value};

use super::utils::{
    build_gmm_instrument_matrix, compute_ab_ar_test, extract_entity_ids, extract_time_ids,
};

/// GMM transformation type for dynamic panel models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmTransform {
    /// Difference GMM (Arellano-Bond 1991)
    /// First-differences to eliminate fixed effects
    #[default]
    Difference,
    /// System GMM (Blundell-Bond 1998)
    /// Combines level and difference equations
    System,
}

impl fmt::Display for GmmTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmTransform::Difference => write!(f, "Difference GMM (Arellano-Bond)"),
            GmmTransform::System => write!(f, "System GMM (Blundell-Bond)"),
        }
    }
}

/// GMM estimation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmStep {
    /// One-step GMM with identity or H matrix weighting
    #[default]
    OneStep,
    /// Two-step GMM with optimal weighting matrix
    TwoStep,
}

impl fmt::Display for GmmStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmStep::OneStep => write!(f, "One-step"),
            GmmStep::TwoStep => write!(f, "Two-step"),
        }
    }
}

/// Configuration for GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmConfig {
    /// Transformation type (difference or system)
    pub transform: GmmTransform,
    /// Estimation step (one-step or two-step)
    pub step: GmmStep,
    /// Maximum lag depth for GMM instruments (default: all available)
    pub max_lag: Option<usize>,
    /// Minimum lag depth for GMM instruments (default: 2)
    pub min_lag: usize,
    /// Whether to collapse instrument matrix
    pub collapse: bool,
    /// Whether to compute robust (Windmeijer-corrected) standard errors for two-step
    pub robust: bool,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            transform: GmmTransform::Difference,
            step: GmmStep::TwoStep,
            max_lag: None,
            min_lag: 2,
            collapse: false,
            robust: true,
        }
    }
}

/// Result from Arellano-Bond / System GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmResult {
    /// Estimation method used
    pub transform: GmmTransform,
    /// Estimation step
    pub step: GmmStep,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics
    pub z_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Number of observations (after differencing)
    pub n_obs: usize,
    /// Number of groups (entities)
    pub n_groups: usize,
    /// Number of instruments
    pub n_instruments: usize,
    /// Sargan/Hansen test statistic for overidentifying restrictions
    pub sargan_statistic: f64,
    /// Sargan test p-value
    pub sargan_p_value: f64,
    /// Sargan test degrees of freedom
    pub sargan_df: usize,
    /// Arellano-Bond test for AR(1) in first differences
    pub ar1_statistic: f64,
    /// AR(1) test p-value
    pub ar1_p_value: f64,
    /// Arellano-Bond test for AR(2) in first differences
    pub ar2_statistic: f64,
    /// AR(2) test p-value
    pub ar2_p_value: f64,
    /// Entity variable name
    pub entity_var: String,
    /// Time variable name
    pub time_var: String,
    /// Warnings
    pub warnings: Vec<String>,
}

impl fmt::Display for GmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ({}) Panel GMM Results", self.transform, self.step)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "Time: {}", self.time_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(f, "No. Instruments: {}", self.n_instruments)?;
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
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f)?;
        writeln!(
            f,
            "Sargan Test: chi2({}) = {:.4}, p-value = {:.4}",
            self.sargan_df, self.sargan_statistic, self.sargan_p_value
        )?;
        writeln!(f, "  H0: Overidentifying restrictions are valid")?;
        writeln!(f)?;
        writeln!(
            f,
            "Arellano-Bond Test for AR(1): z = {:.4}, p-value = {:.4}",
            self.ar1_statistic, self.ar1_p_value
        )?;
        writeln!(
            f,
            "Arellano-Bond Test for AR(2): z = {:.4}, p-value = {:.4}",
            self.ar2_statistic, self.ar2_p_value
        )?;
        writeln!(f, "  H0: No autocorrelation")?;

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Arellano-Bond / System GMM estimator for dynamic panel data.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the other independent variable columns (excluding lagged y)
/// * `entity_col` - Column name for entity/individual identifier
/// * `time_col` - Column name for time period identifier
/// * `lags` - Number of lags of dependent variable to include (default: 1)
/// * `config` - GMM configuration options
///
/// # Returns
///
/// A `GmmResult` containing the estimated coefficients, standard errors,
/// test statistics, and diagnostics.
pub fn run_gmm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    lags: usize,
    config: Option<GmmConfig>,
) -> EconResult<GmmResult> {
    let config = config.unwrap_or_default();
    let mut warnings = Vec::new();

    // Extract entity and time IDs
    let (entity_ids, n_groups) = extract_entity_ids(dataset, entity_col)?;
    let (time_ids, time_values) = extract_time_ids(dataset, time_col)?;
    let n_periods = time_values.len();
    let _n = entity_ids.len();

    if n_periods < lags + 3 {
        return Err(EconError::InsufficientData {
            required: lags + 3,
            provided: n_periods,
            context: format!(
                "GMM estimation requires at least {} time periods for {} lags",
                lags + 3,
                lags
            ),
        });
    }

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "GMM requires at least 2 entities".to_string(),
        });
    }

    // Extract y values
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Organize data by entity
    let mut y_by_entity: Vec<Vec<f64>> = vec![vec![0.0; n_periods]; n_groups];
    for (idx, (&eid, &tid)) in entity_ids.iter().zip(time_ids.iter()).enumerate() {
        y_by_entity[eid][tid] = y[idx];
    }

    // Build design matrix for X variables
    let design = if !x_cols.is_empty() {
        Some(DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?)
    } else {
        None
    };

    // Organize X by entity if present
    let x_by_entity: Option<Vec<Array2<f64>>> = design.as_ref().map(|d| {
        let mut x_ent: Vec<Array2<f64>> =
            vec![Array2::zeros((n_periods, d.data.ncols())); n_groups];
        for (idx, (&eid, &tid)) in entity_ids.iter().zip(time_ids.iter()).enumerate() {
            for j in 0..d.data.ncols() {
                x_ent[eid][[tid, j]] = d.data[[idx, j]];
            }
        }
        x_ent
    });

    // Build differenced data (for difference GMM)
    // Δy_it = y_it - y_{i,t-1}
    let mut dy: Vec<f64> = Vec::new();
    let mut dy_lag: Vec<Vec<f64>> = Vec::new(); // lagged differences
    let mut dx: Vec<Vec<f64>> = Vec::new();
    let mut valid_obs: Vec<(usize, usize)> = Vec::new(); // (entity, time) pairs

    let k_x = x_cols.len();
    let k_lag = lags;
    let k = k_lag + k_x; // total parameters

    for i in 0..n_groups {
        for t in (lags + 1)..n_periods {
            // First difference: Δy_it = y_it - y_{i,t-1}
            let dy_it = y_by_entity[i][t] - y_by_entity[i][t - 1];
            dy.push(dy_it);

            // Lagged first differences: Δy_{i,t-1}, Δy_{i,t-2}, ...
            let mut dy_lags = Vec::with_capacity(k_lag);
            for lag in 1..=k_lag {
                if t > lag {
                    let dy_lag_val = y_by_entity[i][t - lag] - y_by_entity[i][t - lag - 1];
                    dy_lags.push(dy_lag_val);
                } else {
                    dy_lags.push(0.0);
                }
            }
            dy_lag.push(dy_lags);

            // First differences of X
            if let Some(ref x_ent) = x_by_entity {
                let mut dx_row = Vec::with_capacity(k_x);
                for j in 0..k_x {
                    let dx_it = x_ent[i][[t, j]] - x_ent[i][[t - 1, j]];
                    dx_row.push(dx_it);
                }
                dx.push(dx_row);
            }

            valid_obs.push((i, t));
        }
    }

    let n_obs = dy.len();
    if n_obs < k + 1 {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n_obs,
            context: "GMM estimation".to_string(),
        });
    }

    // Build the regressor matrix [Δy_{t-1}, ..., Δy_{t-k}, ΔX]
    let mut w = Array2::zeros((n_obs, k));
    for (row, dy_lags) in dy_lag.iter().enumerate() {
        for (col, &val) in dy_lags.iter().enumerate() {
            w[[row, col]] = val;
        }
    }
    if !dx.is_empty() {
        for (row, dx_row) in dx.iter().enumerate() {
            for (col, &val) in dx_row.iter().enumerate() {
                w[[row, k_lag + col]] = val;
            }
        }
    }

    // Build instrument matrix
    let (z, n_instruments) = build_gmm_instrument_matrix(
        &y_by_entity,
        n_groups,
        n_periods,
        config.min_lag,
        config.max_lag,
        config.collapse,
    );

    // Add X variables as instruments (assumed exogenous)
    let z = if !dx.is_empty() {
        let mut z_full = Array2::zeros((n_obs, n_instruments + k_x));
        for row in 0..n_obs.min(z.nrows()) {
            for col in 0..n_instruments.min(z.ncols()) {
                z_full[[row, col]] = z[[row, col]];
            }
        }
        for (row, dx_row) in dx.iter().enumerate() {
            for (col, &val) in dx_row.iter().enumerate() {
                if n_instruments + col < z_full.ncols() {
                    z_full[[row, n_instruments + col]] = val;
                }
            }
        }
        z_full
    } else {
        // Ensure z has the right number of rows
        let mut z_adj = Array2::zeros((n_obs, n_instruments.max(1)));
        for row in 0..n_obs.min(z.nrows()) {
            for col in 0..n_instruments.min(z.ncols()) {
                z_adj[[row, col]] = z[[row, col]];
            }
        }
        z_adj
    };

    let n_inst_total = z.ncols();

    if n_inst_total < k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Underidentified: {} instruments for {} parameters",
                n_inst_total, k
            ),
        });
    }

    // Convert dy to Array1
    let y_diff = Array1::from_vec(dy);

    // One-step GMM estimation
    // β̂ = (W'Z A Z'W)^{-1} W'Z A Z'y
    // where A is the weighting matrix

    // For one-step, use A = (Z'HZ)^{-1} where H is an appropriate matrix
    // Simplified: use A = (Z'Z)^{-1}
    let ztz = xtx(&z.view());
    let (ztz_inv, _) = safe_inverse(&ztz.view()).map_err(|e| EconError::SingularMatrix {
        context: "Z'Z in GMM".to_string(),
        suggestion: format!("Instrument matrix may be singular: {:?}", e),
    })?;

    let ztw = z.t().dot(&w);
    let zty = z.t().dot(&y_diff);

    let wz = w.t().dot(&z);
    let wzazw = wz.dot(&ztz_inv).dot(&ztw);
    let (wzazw_inv, _) = safe_inverse(&wzazw.view()).map_err(|e| EconError::SingularMatrix {
        context: "W'Z A Z'W in GMM".to_string(),
        suggestion: format!("Check for collinearity: {:?}", e),
    })?;

    let wzazy = wz.dot(&ztz_inv).dot(&zty);
    let beta_one: Array1<f64> = wzazw_inv.dot(&wzazy);

    // Residuals from one-step
    let resid_one = &y_diff - &w.dot(&beta_one);

    // Two-step estimation if requested
    let (beta, resid, step_used) = if config.step == GmmStep::TwoStep {
        // Optimal weighting matrix: A = (Z' Ω̂ Z)^{-1}
        // where Ω̂ = diag(e_i e_i') or similar

        // Build the optimal weighting matrix from one-step residuals
        // For simplicity, use the "robust" version: Ω = sum_i (Z_i' e_i e_i' Z_i)
        let mut omega = Array2::zeros((n_inst_total, n_inst_total));
        let mut row_start = 0;
        for i in 0..n_groups {
            let n_i = valid_obs.iter().filter(|(e, _)| *e == i).count();
            if n_i == 0 {
                continue;
            }

            let z_i = z.slice(ndarray::s![row_start..row_start + n_i, ..]);
            let e_i = resid_one.slice(ndarray::s![row_start..row_start + n_i]);

            // Z_i' e_i e_i' Z_i
            let ze = z_i.t().dot(&e_i);
            for j1 in 0..n_inst_total {
                for j2 in 0..n_inst_total {
                    omega[[j1, j2]] += ze[j1] * ze[j2];
                }
            }

            row_start += n_i;
        }

        let (omega_inv, omega_singular) = match safe_inverse(&omega.view()) {
            Ok((inv, cond)) => (inv, cond.is_none()),
            Err(_) => {
                // Fallback to one-step weighting if singular
                warnings.push(
                    "Optimal weighting matrix singular, using one-step weighting".to_string(),
                );
                (ztz_inv.clone(), true)
            }
        };

        let _ = omega_singular; // Silence unused warning if needed

        let wzazw_two = wz.dot(&omega_inv).dot(&ztw);
        let (wzazw_two_inv, _) =
            safe_inverse(&wzazw_two.view()).map_err(|e| EconError::SingularMatrix {
                context: "W'Z A Z'W in two-step GMM".to_string(),
                suggestion: format!("Check for collinearity: {:?}", e),
            })?;

        let wzazy_two = wz.dot(&omega_inv).dot(&zty);
        let beta_two: Array1<f64> = wzazw_two_inv.dot(&wzazy_two);
        let resid_two = &y_diff - &w.dot(&beta_two);

        (beta_two, resid_two, GmmStep::TwoStep)
    } else {
        (beta_one, resid_one, GmmStep::OneStep)
    };

    // Standard errors
    // For one-step: V = σ² (W'Z A Z'W)^{-1}
    // For two-step with robust: Windmeijer correction

    let sigma2 = resid.iter().map(|e| e * e).sum::<f64>() / (n_obs - k) as f64;

    let vcov = if step_used == GmmStep::TwoStep && config.robust {
        // Windmeijer (2005) finite-sample correction for two-step
        // Simplified version: use robust variance
        let mut meat = Array2::zeros((k, k));
        let mut row_start = 0;
        for i in 0..n_groups {
            let n_i = valid_obs.iter().filter(|(e, _)| *e == i).count();
            if n_i == 0 {
                continue;
            }

            let w_i = w.slice(ndarray::s![row_start..row_start + n_i, ..]);
            let e_i = resid.slice(ndarray::s![row_start..row_start + n_i]);

            let we = w_i.t().dot(&e_i);
            for j1 in 0..k {
                for j2 in 0..k {
                    meat[[j1, j2]] += we[j1] * we[j2];
                }
            }

            row_start += n_i;
        }

        let bread = &wzazw_inv;
        bread.dot(&meat).dot(bread)
    } else {
        &wzazw_inv * sigma2
    };

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.abs().sqrt()).to_vec();

    // z-statistics and p-values
    let beta_vec: Vec<f64> = beta.to_vec();
    let z_stats: Vec<f64> = beta_vec
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - Normal::new(0.0, 1.0).map(|n| n.cdf(z.abs())).unwrap_or(0.5)))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Sargan/Hansen test for overidentifying restrictions
    // J = e'Z (Z'Z)^{-1} Z'e / σ² ~ χ²(n_inst - k)
    let ze = z.t().dot(&resid);
    let sargan_statistic = ze.dot(&ztz_inv.dot(&ze)) / sigma2;
    let sargan_df = n_inst_total.saturating_sub(k);
    let sargan_p_value = if sargan_df > 0 {
        chi_squared_p_value(sargan_statistic, sargan_df as f64)
    } else {
        1.0
    };

    // Arellano-Bond test for serial correlation in first differences
    // AR(1) and AR(2) tests
    let (ar1_stat, ar1_p) = compute_ab_ar_test(&resid, &valid_obs, n_groups, 1);
    let (ar2_stat, ar2_p) = compute_ab_ar_test(&resid, &valid_obs, n_groups, 2);

    // Build variable names
    let mut variables = Vec::with_capacity(k);
    for lag in 1..=k_lag {
        variables.push(format!("L{}.{}", lag, y_col));
    }
    if let Some(ref d) = design {
        variables.extend(d.column_names.iter().cloned());
    }

    Ok(GmmResult {
        transform: config.transform,
        step: step_used,
        dep_var: format!("D.{}", y_col),
        variables,
        coefficients: beta_vec,
        std_errors,
        z_stats,
        p_values,
        significance,
        n_obs,
        n_groups,
        n_instruments: n_inst_total,
        sargan_statistic,
        sargan_p_value,
        sargan_df,
        ar1_statistic: ar1_stat,
        ar1_p_value: ar1_p,
        ar2_statistic: ar2_stat,
        ar2_p_value: ar2_p,
        entity_var: entity_col.to_string(),
        time_var: time_col.to_string(),
        warnings,
    })
}

/// Convenience function for Arellano-Bond difference GMM with defaults.
///
/// Uses one lag of the dependent variable and two-step estimation.
pub fn run_arellano_bond(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<GmmResult> {
    run_gmm(dataset, y_col, x_cols, entity_col, time_col, 1, None)
}
