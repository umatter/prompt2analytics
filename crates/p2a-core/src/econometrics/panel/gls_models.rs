//! Panel GLS (FGLS) estimators.
//!
//! Feasible Generalized Least Squares for panel data with cross-sectional
//! correlation and heteroskedasticity.
//!
//! # Reference
//!
//! Parks, R.W. (1967). "Efficient Estimation of a System of Regression Equations
//! When Disturbances Are Both Serially and Contemporaneously Correlated."
//! *Journal of the American Statistical Association*, 62, 500-509.
//!
//! R equivalent: `plm::pggls()`

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

use super::utils::demean_by_entity;

/// Panel GLS model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelGlsModel {
    /// Fixed effects GLS (within transformation before FGLS)
    #[default]
    FixedEffects,
    /// Pooled GLS (no effects, just FGLS)
    Pooling,
    /// First-difference GLS
    FirstDifference,
}

impl fmt::Display for PanelGlsModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelGlsModel::FixedEffects => write!(f, "Fixed Effects GLS (FEGLS)"),
            PanelGlsModel::Pooling => write!(f, "Pooled GLS"),
            PanelGlsModel::FirstDifference => write!(f, "First-Difference GLS (FDGLS)"),
        }
    }
}

/// Result from Panel GLS estimation.
///
/// Panel GLS (FGLS) allows for cross-sectional correlation and heteroskedasticity
/// in the error structure while being robust to any form of intragroup correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelGlsResult {
    /// Model type used
    pub model: PanelGlsModel,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values (two-sided)
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// R-squared (based on transformed model)
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities (cross-sectional units)
    pub n_groups: usize,
    /// Number of time periods
    pub n_periods: usize,
    /// Degrees of freedom
    pub df_residual: usize,
    /// Residual sum of squares
    pub rss: f64,
    /// Residual standard error
    pub sigma: f64,
    /// Estimated error covariance matrix dimension (T x T)
    #[serde(skip)]
    pub error_cov_dim: usize,
    /// Any warnings generated during estimation
    #[serde(skip)]
    pub warnings: Vec<String>,
}

impl fmt::Display for PanelGlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(70))?;
        writeln!(f, "Panel GLS (FGLS) Estimation Results")?;
        writeln!(f, "{}", "=".repeat(70))?;
        writeln!(f, "Model: {}", self.model)?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Observations: {}  Entities: {}  Time periods: {}",
            self.n_obs, self.n_groups, self.n_periods)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coefficient", "Std.Error", "t-stat", "P>|t|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            let sig = match &self.significance[i] {
                SignificanceLevel::NotSignificant => "",
                SignificanceLevel::TenPercent => ".",
                SignificanceLevel::FivePercent => "*",
                SignificanceLevel::OnePercent => "**",
                SignificanceLevel::TenthPercent => "***",
            };
            writeln!(f, "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig)?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        writeln!(f, "Sigma: {:.6}  RSS: {:.4}", self.sigma, self.rss)?;

        if !self.warnings.is_empty() {
            writeln!(f, "\nWarnings:")?;
            for warning in &self.warnings {
                writeln!(f, "  - {}", warning)?;
            }
        }

        Ok(())
    }
}

/// Run Panel GLS (Feasible Generalized Least Squares) estimation.
///
/// Panel GLS estimates regression coefficients while accounting for cross-sectional
/// correlation and heteroskedasticity in the error terms. The method:
/// 1. Estimates initial model (OLS, FE, or FD depending on model type)
/// 2. Uses residuals to estimate the error covariance matrix
/// 3. Re-estimates using GLS with the estimated covariance
///
/// # Arguments
///
/// * `dataset` - The dataset containing panel data
/// * `y_col` - Name of dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `entity_col` - Name of entity/group identifier column
/// * `time_col` - Name of time period identifier column
/// * `model` - Model type: FixedEffects, Pooling, or FirstDifference
///
/// # Returns
///
/// `PanelGlsResult` containing estimates, standard errors, and diagnostics.
///
/// # Reference
///
/// Implementation adapted from R's plm package pggls() function.
/// Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package."
/// *Journal of Statistical Software*, 27(2).
pub fn run_panel_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    model: Option<PanelGlsModel>,
) -> EconResult<PanelGlsResult> {
    let model = model.unwrap_or_default();
    let mut warnings = Vec::new();

    // Extract data from dataset
    let df = dataset.df();
    let y_series = df.column(y_col)
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Dependent variable '{}' not found", y_col),
        })?;
    let y: Vec<f64> = y_series.f64()
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Dependent variable '{}' must be numeric", y_col),
        })?
        .into_no_null_iter()
        .collect();

    // Extract entity and time identifiers
    let entity_series = df.column(entity_col)
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Entity column '{}' not found", entity_col),
        })?;

    let time_series = df.column(time_col)
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Time column '{}' not found", time_col),
        })?;

    // Get unique entities
    let entity_strings: Vec<String> = if let Ok(utf8) = entity_series.str() {
        utf8.into_iter().map(|s| s.unwrap_or("").to_string()).collect()
    } else if let Ok(i64_col) = entity_series.i64() {
        i64_col.into_iter().map(|v| v.unwrap_or(0).to_string()).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Entity column must be string or integer".to_string(),
        });
    };

    let time_values: Vec<i64> = if let Ok(i64_col) = time_series.i64() {
        i64_col.into_iter().map(|v| v.unwrap_or(0)).collect()
    } else if let Ok(f64_col) = time_series.f64() {
        f64_col.into_iter().map(|v| v.unwrap_or(0.0) as i64).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Time column must be numeric".to_string(),
        });
    };

    // Build entity mapping
    let mut entity_map: HashMap<String, usize> = HashMap::new();
    let mut entity_ids: Vec<usize> = Vec::with_capacity(y.len());
    let mut entity_counter = 0;

    for entity in &entity_strings {
        let id = *entity_map.entry(entity.clone()).or_insert_with(|| {
            let id = entity_counter;
            entity_counter += 1;
            id
        });
        entity_ids.push(id);
    }

    let n_groups = entity_map.len();

    // Get unique time periods
    let mut unique_times: Vec<i64> = time_values.iter().copied().collect();
    unique_times.sort_unstable();
    unique_times.dedup();
    let n_periods = unique_times.len();

    let n = y.len();

    // Check data requirements: need N >> T for consistent estimation
    if n_groups < n_periods {
        warnings.push(format!(
            "Panel GLS requires N >> T. Found N={}, T={}. Estimates may be inconsistent.",
            n_groups, n_periods
        ));
    }

    // Build design matrix (includes intercept for non-FE models)
    let has_intercept = model != PanelGlsModel::FixedEffects;
    let dm = DesignMatrix::from_dataframe(df, x_cols, has_intercept)?;
    let x = &dm.data;
    let k = x.ncols();

    // Step 1: Initial estimation based on model type
    let (initial_resid, _) = match model {
        PanelGlsModel::FixedEffects => {
            // Within transformation (demean by entity)
            let y_arr = Array1::from(y.clone());
            let y_demeaned = demean_by_entity(&y_arr, &entity_ids, n_groups);

            let mut x_demeaned = Array2::<f64>::zeros((n, k));
            for j in 0..k {
                let col = x.column(j).to_owned();
                let demeaned = demean_by_entity(&col, &entity_ids, n_groups);
                for i in 0..n {
                    x_demeaned[[i, j]] = demeaned[i];
                }
            }

            // OLS on demeaned data
            let xtx_mat = xtx(&x_demeaned.view());
            let (xtx_inv, _) = safe_inverse(&xtx_mat.view())?;
            let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
            let beta: Array1<f64> = xtx_inv.dot(&xty_vec);
            let fitted = x_demeaned.dot(&beta);
            let resid = &y_demeaned - &fitted;

            (resid.to_vec(), beta.to_vec())
        },
        PanelGlsModel::Pooling => {
            // Simple OLS
            let y_arr = Array1::from(y.clone());
            let xtx_mat = xtx(&x.view());
            let (xtx_inv, _) = safe_inverse(&xtx_mat.view())?;
            let xty_vec = xty(&x.view(), &y_arr);
            let beta: Array1<f64> = xtx_inv.dot(&xty_vec);
            let fitted = x.dot(&beta);
            let resid = &y_arr - &fitted;

            (resid.to_vec(), beta.to_vec())
        },
        PanelGlsModel::FirstDifference => {
            // First-difference transformation
            warnings.push("First-difference GLS drops first observation per entity".to_string());

            // Sort data by entity and time for differencing
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by_key(|&i| (&entity_strings[i], time_values[i]));

            // Compute first differences
            let mut y_diff = Vec::new();
            let mut x_diff = Vec::new();

            let mut prev_entity = String::new();
            let mut prev_y = 0.0;
            let mut prev_x = vec![0.0; k];

            for &idx in &sorted_indices {
                let curr_entity = &entity_strings[idx];

                if *curr_entity == prev_entity {
                    // Same entity, compute difference
                    y_diff.push(y[idx] - prev_y);
                    let mut row = Vec::with_capacity(k);
                    for j in 0..k {
                        row.push(x[[idx, j]] - prev_x[j]);
                    }
                    x_diff.push(row);
                }

                prev_entity = curr_entity.clone();
                prev_y = y[idx];
                prev_x = (0..k).map(|j| x[[idx, j]]).collect();
            }

            if y_diff.is_empty() {
                return Err(EconError::InsufficientData {
                    required: 2,
                    provided: 1,
                    context: "First-difference requires at least 2 time periods per entity".to_string(),
                });
            }

            let n_diff = y_diff.len();
            let y_diff_arr = Array1::from(y_diff);
            let mut x_diff_arr = Array2::<f64>::zeros((n_diff, k));
            for (i, row) in x_diff.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    x_diff_arr[[i, j]] = val;
                }
            }

            // OLS on differenced data
            let xtx_mat = xtx(&x_diff_arr.view());
            let (xtx_inv, _) = safe_inverse(&xtx_mat.view())?;
            let xty_vec = xty(&x_diff_arr.view(), &y_diff_arr);
            let beta: Array1<f64> = xtx_inv.dot(&xty_vec);
            let fitted = x_diff_arr.dot(&beta);
            let resid = &y_diff_arr - &fitted;

            (resid.to_vec(), beta.to_vec())
        },
    };

    // Step 2: Estimate error covariance matrix
    // We estimate a T x T covariance matrix from cross-sectional residuals
    // Omega[s,t] = (1/N) * sum_i(e_is * e_it)

    // Group residuals by entity
    let mut entity_resids: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
    let mut entity_times: Vec<Vec<i64>> = vec![Vec::new(); n_groups];

    // Create time index mapping
    let mut time_idx_map: HashMap<i64, usize> = HashMap::new();
    for (idx, &t) in unique_times.iter().enumerate() {
        time_idx_map.insert(t, idx);
    }

    for (i, (&entity_id, &resid)) in entity_ids.iter().zip(initial_resid.iter()).enumerate() {
        entity_resids[entity_id].push(resid);
        entity_times[entity_id].push(time_values[i]);
    }

    // Build T x T omega matrix
    let t_dim = n_periods;
    let mut omega = Array2::<f64>::zeros((t_dim, t_dim));
    let mut omega_counts = Array2::<usize>::zeros((t_dim, t_dim));

    for entity_id in 0..n_groups {
        let resids = &entity_resids[entity_id];
        let times = &entity_times[entity_id];

        for (s, &ts) in times.iter().enumerate() {
            for (t_idx, &tt) in times.iter().enumerate() {
                if let (Some(&s_idx), Some(&t_idx_val)) = (time_idx_map.get(&ts), time_idx_map.get(&tt)) {
                    omega[[s_idx, t_idx_val]] += resids[s] * resids[t_idx];
                    omega_counts[[s_idx, t_idx_val]] += 1;
                }
            }
        }
    }

    // Average the covariance estimates
    for s in 0..t_dim {
        for t in 0..t_dim {
            if omega_counts[[s, t]] > 0 {
                omega[[s, t]] /= omega_counts[[s, t]] as f64;
            }
        }
    }

    // Ensure positive definiteness
    let omega_inv = match safe_inverse(&omega.view()) {
        Ok((inv, _)) => inv,
        Err(_) => {
            warnings.push("Omega matrix near-singular, using regularization".to_string());
            // Add small ridge to diagonal
            for i in 0..t_dim {
                omega[[i, i]] += 1e-6;
            }
            safe_inverse(&omega.view())
                .map(|(inv, _)| inv)
                .map_err(|_| EconError::SingularMatrix {
                    context: "Error covariance matrix".to_string(),
                    suggestion: "Check for multicollinearity or insufficient time variation".to_string(),
                })?
        }
    };

    // Step 3: GLS estimation
    let y_arr = Array1::from(y.clone());

    // Compute GLS estimates using the formula:
    // beta_GLS = (X' * Omega_block^{-1} * X)^{-1} * (X' * Omega_block^{-1} * y)

    let mut xtox = Array2::<f64>::zeros((k, k));
    let mut xtoy = Array1::<f64>::zeros(k);

    for entity_id in 0..n_groups {
        // Get indices for this entity
        let mut entity_indices: Vec<usize> = Vec::new();
        let mut entity_time_indices: Vec<usize> = Vec::new();

        for (i, &eid) in entity_ids.iter().enumerate() {
            if eid == entity_id {
                entity_indices.push(i);
                if let Some(&t_idx) = time_idx_map.get(&time_values[i]) {
                    entity_time_indices.push(t_idx);
                }
            }
        }

        let ti = entity_indices.len();
        if ti == 0 {
            continue;
        }

        // Extract submatrices for this entity
        let mut x_i = Array2::<f64>::zeros((ti, k));
        let mut y_i = Array1::<f64>::zeros(ti);

        for (local_idx, &global_idx) in entity_indices.iter().enumerate() {
            y_i[local_idx] = y_arr[global_idx];
            for j in 0..k {
                x_i[[local_idx, j]] = x[[global_idx, j]];
            }
        }

        // Extract relevant part of Omega^{-1} for this entity's time periods
        let mut omega_inv_i = Array2::<f64>::zeros((ti, ti));
        for (s, &s_idx) in entity_time_indices.iter().enumerate() {
            for (t, &t_idx) in entity_time_indices.iter().enumerate() {
                omega_inv_i[[s, t]] = omega_inv[[s_idx, t_idx]];
            }
        }

        // Accumulate X'Omega^{-1}X and X'Omega^{-1}y
        let xto = x_i.t().dot(&omega_inv_i);
        xtox = &xtox + &xto.dot(&x_i);
        xtoy = &xtoy + &xto.dot(&y_i);
    }

    // Solve for GLS coefficients
    let (xtox_inv, cond) = safe_inverse(&xtox.view()).map_err(|_| EconError::SingularMatrix {
        context: "X'Omega^{-1}X in Panel GLS".to_string(),
        suggestion: "Check for perfect collinearity".to_string(),
    })?;

    if cond.map_or(false, |c| c > 1e10) {
        warnings.push("High condition number detected, results may be numerically unstable".to_string());
    }

    let beta: Array1<f64> = xtox_inv.dot(&xtoy);

    // Compute residuals and diagnostics
    let fitted = x.dot(&beta);
    let residuals = &y_arr - &fitted;

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let tss: f64 = y_arr.iter().map(|yi| {
        let y_mean = y_arr.mean().unwrap_or(0.0);
        (yi - y_mean).powi(2)
    }).sum();

    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    let df_residual = n.saturating_sub(k);
    let sigma = if df_residual > 0 {
        (rss / df_residual as f64).sqrt()
    } else {
        0.0
    };

    // Standard errors from (X'Omega^{-1}X)^{-1}
    let se_scale = rss / df_residual as f64;
    let std_errors: Vec<f64> = (0..k)
        .map(|j| (xtox_inv[[j, j]] * se_scale).abs().sqrt())
        .collect();

    // T-statistics and p-values
    let t_stats: Vec<f64> = beta.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df_residual as f64))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Build variable names
    let variables: Vec<String> = if model != PanelGlsModel::FixedEffects {
        std::iter::once("(Intercept)".to_string())
            .chain(x_cols.iter().map(|s| s.to_string()))
            .collect()
    } else {
        x_cols.iter().map(|s| s.to_string()).collect()
    };

    Ok(PanelGlsResult {
        model,
        dep_var: y_col.to_string(),
        variables,
        coefficients: beta.to_vec(),
        std_errors,
        t_stats,
        p_values,
        significance,
        r_squared,
        n_obs: n,
        n_groups,
        n_periods,
        df_residual,
        rss,
        sigma,
        error_cov_dim: t_dim,
        warnings,
    })
}

/// Convenience function for Panel GLS with fixed effects (FEGLS).
pub fn run_fegls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<PanelGlsResult> {
    run_panel_gls(dataset, y_col, x_cols, entity_col, time_col, Some(PanelGlsModel::FixedEffects))
}

/// Convenience function for Pooled GLS.
pub fn run_pooled_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<PanelGlsResult> {
    run_panel_gls(dataset, y_col, x_cols, entity_col, time_col, Some(PanelGlsModel::Pooling))
}
