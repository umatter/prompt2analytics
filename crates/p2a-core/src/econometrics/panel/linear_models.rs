//! Linear panel data models: Fixed Effects and Random Effects.
//!
//! # Mathematical Background
//!
//! For panel data with observations yᵢₜ for entity i at time t:
//!
//! yᵢₜ = αᵢ + Xᵢₜ'β + εᵢₜ
//!
//! ## Fixed Effects (Within) Estimator
//!
//! The FE estimator demeans the data within each entity:
//!
//! (yᵢₜ - ȳᵢ) = (Xᵢₜ - X̄ᵢ)'β + (εᵢₜ - ε̄ᵢ)
//!
//! This eliminates time-invariant unobserved heterogeneity αᵢ.
//!
//! ## Random Effects (GLS) Estimator
//!
//! The RE estimator assumes αᵢ is uncorrelated with Xᵢₜ and uses quasi-demeaning:
//!
//! (yᵢₜ - θȳᵢ) = (1-θ)α + (Xᵢₜ - θX̄ᵢ)'β + (εᵢₜ - θε̄ᵢ)
//!
//! where θ = 1 - √(σ²ₑ / (σ²ₑ + Tσ²ᵤ))
//!
//! # References
//!
//! - Mundlak, Y. (1978). On the pooling of time series and cross section data.
//!   *Econometrica*, 46(1), 69-85. https://doi.org/10.2307/1913646
//!
//! - Baltagi, B.H. (2013). *Econometric Analysis of Panel Data* (5th ed.).
//!   Wiley. ISBN: 978-1118672327.
//!
//! R equivalent: `plm::plm()` with `model = "within"` or `model = "random"`

use ndarray::Array1;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

use super::types::{PanelResult, PanelMethod};
use super::utils::{extract_entity_ids, compute_entity_means, demean_by_entity, demean_matrix_by_entity};

/// Run Fixed Effects (within) panel estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `entity_col` - Column name for entity/individual identifier
pub fn run_fixed_effects(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PanelResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    // Build design matrix without intercept for FE
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;

    let x = design.data;
    let var_names = design.column_names;
    let n = y.len();
    let k = x.ncols();

    // Extract entity IDs
    let (entity_ids, n_groups) = extract_entity_ids(dataset, entity_col)?;

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }

    // Demean y and X by entity
    let y_demeaned = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x, &entity_ids, n_groups);

    // Degrees of freedom: n - N - k (lost N entity dummies)
    let df = n.saturating_sub(n_groups).saturating_sub(k);
    if df == 0 {
        return Err(EconError::InsufficientData {
            required: n_groups + k + 1,
            provided: n,
            context: "Not enough observations for degrees of freedom".to_string(),
        });
    }

    // OLS on demeaned data: β = (X̃'X̃)^{-1} X̃'ỹ
    let xtx_mat = xtx(&x_demeaned.view());
    let (xtx_inv, _cond_warning) = safe_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X in Fixed Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity among variables: {:?}", e),
        })?;

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df as f64;

    // Variance-covariance matrix
    let vcov = &xtx_inv * sigma2;

    // Standard errors
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.sqrt()).to_vec();

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned.iter().map(|y| (y - y_mean_demeaned).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - n_groups - 1) as f64) / (df as f64);

    // F-statistic
    let f_stat = if k > 0 && ssr > 0.0 {
        (sst - ssr) / (k as f64) / (ssr / df as f64)
    } else {
        0.0
    };
    let f_p_value = crate::traits::estimator::f_test_p_value(f_stat, k as f64, df as f64);

    // t-statistics and p-values
    let beta_vec: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = beta_vec.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    Ok(PanelResult {
        method: PanelMethod::FixedEffects,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta_vec,
        std_errors,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        f_stat,
        f_p_value,
        n_obs: n,
        n_groups,
        df,
        entity_var: entity_col.to_string(),
        significance,
        sigma_u: None,
        sigma_e: Some(sigma2.sqrt()),
        theta: None,
    })
}

/// Run Random Effects (GLS) panel estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `entity_col` - Column name for entity/individual identifier
pub fn run_random_effects(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PanelResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    // Build design matrix with intercept for RE
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;

    let x = design.data;
    let var_names = design.column_names;
    let n = y.len();
    let k = x.ncols();

    // Extract entity IDs
    let (entity_ids, n_groups) = extract_entity_ids(dataset, entity_col)?;

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }

    // Step 1: Pooled OLS to get initial residuals
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X in Random Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity: {:?}", e),
        })?;
    let xty_vec = xty(&x.view(), &y);
    let beta_pooled: Array1<f64> = xtx_inv.dot(&xty_vec);
    let residuals_pooled = &y - &x.dot(&beta_pooled);

    // Step 2: Estimate variance components using Fixed Effects residuals
    let y_demeaned = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x, &entity_ids, n_groups);

    let xtx_demeaned = xtx(&x_demeaned.view());
    let (xtx_demeaned_inv, _) = safe_inverse(&xtx_demeaned.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X (demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity: {:?}", e),
        })?;
    let xty_demeaned = xty(&x_demeaned.view(), &y_demeaned);
    let beta_fe: Array1<f64> = xtx_demeaned_inv.dot(&xty_demeaned);
    let residuals_fe = &y_demeaned - &x_demeaned.dot(&beta_fe);

    let df_fe = n.saturating_sub(n_groups).saturating_sub(k);
    let sigma2_e = if df_fe > 0 {
        residuals_fe.iter().map(|r| r * r).sum::<f64>() / df_fe as f64
    } else {
        residuals_pooled.iter().map(|r| r * r).sum::<f64>() / n as f64
    };

    // Count observations per group
    let mut group_counts = vec![0usize; n_groups];
    for &g in &entity_ids {
        group_counts[g] += 1;
    }
    let t_bar: f64 = group_counts.iter().map(|&c| c as f64).sum::<f64>() / n_groups as f64;

    // Compute between-groups variance
    let y_means = compute_entity_means(&y, &entity_ids, n_groups);

    // Extract unique group means
    let y_between: Array1<f64> = (0..n_groups)
        .filter_map(|g| {
            let first_idx = entity_ids.iter().position(|&id| id == g)?;
            Some(y_means[first_idx])
        })
        .collect();

    let y_overall_mean = y.mean().unwrap_or(0.0);
    let sigma2_between = if n_groups > 1 {
        y_between.iter().map(|&ym| (ym - y_overall_mean).powi(2)).sum::<f64>() / (n_groups - 1) as f64
    } else {
        0.0
    };

    // σ²_u = (σ²_between - σ²_e / T̄)
    let sigma2_u = (sigma2_between - sigma2_e / t_bar).max(0.0);

    // Step 3: Compute theta for quasi-demeaning
    let theta = if sigma2_u > 0.0 {
        1.0 - (sigma2_e / (t_bar * sigma2_u + sigma2_e)).sqrt()
    } else {
        0.0
    };

    // Step 4: Quasi-demean the data
    let y_quasi = &y - &(&y_means * theta);
    let x_means = {
        let mut means = ndarray::Array2::zeros((n, k));
        for j in 0..k {
            let col = x.column(j).to_owned();
            let col_means = compute_entity_means(&col, &entity_ids, n_groups);
            means.column_mut(j).assign(&col_means);
        }
        means
    };
    let x_quasi = {
        let mut xq = ndarray::Array2::zeros((n, k));
        for j in 0..k {
            let col = x.column(j).to_owned();
            let col_means = x_means.column(j).to_owned();
            let col_quasi = &col - &(&col_means * theta);
            xq.column_mut(j).assign(&col_quasi);
        }
        xq
    };

    // Step 5: OLS on quasi-demeaned data
    let xtx_quasi = xtx(&x_quasi.view());
    let (xtx_quasi_inv, _) = safe_inverse(&xtx_quasi.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X (quasi-demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
    let xty_quasi = xty(&x_quasi.view(), &y_quasi);
    let beta: Array1<f64> = xtx_quasi_inv.dot(&xty_quasi);

    // Residuals (from original data)
    let y_hat: Array1<f64> = x.dot(&beta);
    let residuals = &y - &y_hat;

    // Degrees of freedom
    let df = n.saturating_sub(k);

    // Variance estimation
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df as f64;
    let vcov = &xtx_quasi_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.sqrt()).to_vec();

    // R-squared (overall)
    let sst: f64 = y.iter().map(|yi| (yi - y_overall_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // F-statistic
    let f_stat = if k > 1 && ssr > 0.0 {
        (sst - ssr) / ((k - 1) as f64) / (ssr / df as f64)
    } else {
        0.0
    };
    let f_p_value = crate::traits::estimator::f_test_p_value(f_stat, (k.saturating_sub(1)) as f64, df as f64);

    // t-statistics and p-values
    let beta_vec: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = beta_vec.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    Ok(PanelResult {
        method: PanelMethod::RandomEffects,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta_vec,
        std_errors,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        f_stat,
        f_p_value,
        n_obs: n,
        n_groups,
        df,
        entity_var: entity_col.to_string(),
        significance,
        sigma_u: Some(sigma2_u.sqrt()),
        sigma_e: Some(sigma2_e.sqrt()),
        theta: Some(theta),
    })
}
