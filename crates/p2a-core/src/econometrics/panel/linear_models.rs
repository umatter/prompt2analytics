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
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

use super::types::{PanelMethod, PanelResult};
use super::utils::{
    compute_entity_means, demean_by_entity, demean_matrix_by_entity, extract_entity_ids,
};

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
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
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
    let (xtx_inv, _cond_warning) =
        safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X in Fixed Effects".to_string(),
            suggestion: format!(
                "Check for perfect multicollinearity among variables: {:?}",
                e
            ),
        })?;

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df as f64;

    // Classical (homoskedastic) FE variance, matching R's
    // `plm(..., model = "within")` default: Var(β̂) = sigma^2 * (X̃'X̃)^{-1}.
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned
        .iter()
        .map(|y| (y - y_mean_demeaned).powi(2))
        .sum();
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
    let t_stats: Vec<f64> = beta_vec
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

    let coef_arr = ndarray::Array1::from_vec(beta_vec.clone());
    let se_arr = ndarray::Array1::from_vec(std_errors.clone());

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
        coef_arr,
        se_arr,
        residuals,
        vcov,
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
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
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

    // Swamy-Arora variance component estimation (matches R's plm default)
    //
    // Step 1: Within (FE) estimation to get σ²_e
    // Use only non-intercept columns for the within estimator
    let x_no_intercept = {
        let design_no_int = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
        design_no_int.data
    };
    let k_no_int = x_no_intercept.ncols(); // number of slope coefficients

    let y_demeaned = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x_no_intercept, &entity_ids, n_groups);

    let xtx_demeaned = xtx(&x_demeaned.view());
    let (xtx_demeaned_inv, _) =
        safe_inverse(&xtx_demeaned.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity: {:?}", e),
        })?;
    let xty_demeaned = xty(&x_demeaned.view(), &y_demeaned);
    let beta_fe: Array1<f64> = xtx_demeaned_inv.dot(&xty_demeaned);
    let residuals_fe = &y_demeaned - &x_demeaned.dot(&beta_fe);

    // σ²_e = SSR_within / (n - n_groups - k_no_int)
    let df_fe = n.saturating_sub(n_groups).saturating_sub(k_no_int);
    let sigma2_e = if df_fe > 0 {
        residuals_fe.iter().map(|r| r * r).sum::<f64>() / df_fe as f64
    } else {
        // Fallback: use pooled OLS residual variance
        let xtx_mat = xtx(&x.view());
        let (xtx_inv, _) =
            safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
                context: "X'X in Random Effects".to_string(),
                suggestion: format!("Check for perfect multicollinearity: {:?}", e),
            })?;
        let xty_vec = xty(&x.view(), &y);
        let beta_pooled: Array1<f64> = xtx_inv.dot(&xty_vec);
        let residuals_pooled = &y - &x.dot(&beta_pooled);
        residuals_pooled.iter().map(|r| r * r).sum::<f64>() / n as f64
    };

    // Count observations per group
    let mut group_counts = vec![0usize; n_groups];
    for &g in &entity_ids {
        group_counts[g] += 1;
    }

    // Step 2: Between regression to get σ²_between
    // Compute group means of y and X (with intercept)
    let y_means = compute_entity_means(&y, &entity_ids, n_groups);

    // Extract unique group means for y and x (no intercept)
    let y_between: Array1<f64> = (0..n_groups)
        .map(|g| {
            let first_idx = entity_ids.iter().position(|&id| id == g).unwrap();
            y_means[first_idx]
        })
        .collect();

    // Build between-regression X matrix: group means of x_cols + intercept
    let x_between = {
        let mut xb = ndarray::Array2::zeros((n_groups, k_no_int + 1));
        // Intercept column
        for i in 0..n_groups {
            xb[[i, 0]] = 1.0;
        }
        // Group means of each x column
        for j in 0..k_no_int {
            let col = x_no_intercept.column(j).to_owned();
            let col_means = compute_entity_means(&col, &entity_ids, n_groups);
            for g in 0..n_groups {
                let first_idx = entity_ids.iter().position(|&id| id == g).unwrap();
                xb[[g, j + 1]] = col_means[first_idx];
            }
        }
        xb
    };

    // Between OLS: regress group-mean y on group-mean X
    let xtx_between = xtx(&x_between.view());
    let (xtx_between_inv, _) =
        safe_inverse(&xtx_between.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (between) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
    let xty_between = xty(&x_between.view(), &y_between);
    let beta_between: Array1<f64> = xtx_between_inv.dot(&xty_between);
    let residuals_between = &y_between - &x_between.dot(&beta_between);

    // σ²_between = SSR_between / (N - K - 1) where K = number of slope coefficients
    let df_between = n_groups.saturating_sub(k_no_int + 1);
    let sigma2_between = if df_between > 0 {
        residuals_between.iter().map(|r| r * r).sum::<f64>() / df_between as f64
    } else {
        0.0
    };

    // Step 3: Swamy-Arora σ²_u estimation
    // Use harmonic mean of T_i for unbalanced panels (matches plm)
    let t_harmonic: f64 = {
        let sum_inv = group_counts.iter().map(|&c| 1.0 / c as f64).sum::<f64>();
        n_groups as f64 / sum_inv
    };

    // σ²_u = σ²_between - σ²_e / T_harmonic
    let sigma2_u = (sigma2_between - sigma2_e / t_harmonic).max(0.0);

    // Step 4: Compute theta for quasi-demeaning (per-group for unbalanced panels)
    // For balanced panels all thetas are identical; for unbalanced, each group has its own
    // θ_i = 1 - sqrt(σ²_e / (T_i * σ²_u + σ²_e))
    let theta_per_group: Vec<f64> = group_counts
        .iter()
        .map(|&ti| {
            if sigma2_u > 0.0 {
                1.0 - (sigma2_e / (ti as f64 * sigma2_u + sigma2_e)).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Report a representative theta (harmonic-mean based, matches plm summary output)
    let theta = if sigma2_u > 0.0 {
        1.0 - (sigma2_e / (t_harmonic * sigma2_u + sigma2_e)).sqrt()
    } else {
        0.0
    };

    // Step 5: Quasi-demean the data using per-group theta
    // y*_it = y_it - θ_i * ȳ_i
    let y_quasi = {
        let mut yq = y.clone();
        for i in 0..n {
            let g = entity_ids[i];
            yq[i] -= theta_per_group[g] * y_means[i];
        }
        yq
    };
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
            for i in 0..n {
                let g = entity_ids[i];
                xq[[i, j]] = x[[i, j]] - theta_per_group[g] * x_means[[i, j]];
            }
        }
        xq
    };

    // Step 6: GLS on quasi-demeaned data
    let xtx_quasi = xtx(&x_quasi.view());
    let (xtx_quasi_inv, _) =
        safe_inverse(&xtx_quasi.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (quasi-demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
    let xty_quasi = xty(&x_quasi.view(), &y_quasi);
    let beta: Array1<f64> = xtx_quasi_inv.dot(&xty_quasi);

    // Residuals (from original data, for R² and diagnostics)
    let y_hat: Array1<f64> = x.dot(&beta);
    let residuals = &y - &y_hat;

    // Degrees of freedom: n - k (k includes intercept)
    let df = n.saturating_sub(k);

    // Entity-clustered standard errors (CR1) — standard for panel RE.
    // Residuals from original (non-quasi-demeaned) data capture the full error.
    let g = n_groups;
    let correction_re = if g > 1 {
        (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / df as f64)
    } else {
        1.0
    };

    let mut meat_re = ndarray::Array2::<f64>::zeros((k, k));
    let mut entity_indices_re: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for i in 0..n {
        entity_indices_re[entity_ids[i]].push(i);
    }
    for indices in &entity_indices_re {
        let mut xe = vec![0.0; k];
        for &i in indices {
            let e = residuals[i];
            for j in 0..k {
                xe[j] += x_quasi[[i, j]] * e;
            }
        }
        for j in 0..k {
            for l in 0..k {
                meat_re[[j, l]] += xe[j] * xe[l];
            }
        }
    }

    let temp_re = xtx_quasi_inv.dot(&meat_re);
    let vcov = temp_re.dot(&xtx_quasi_inv) * correction_re;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();

    // R-squared: plm uses squared correlation between y_hat and y
    // This is equivalent to 1 - var(residuals)/var(y) when computed properly
    let y_overall_mean = y.mean().unwrap_or(0.0);
    let y_hat_mean = y_hat.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_overall_mean).powi(2)).sum();
    let cov_y_yhat: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yhi)| (yi - y_overall_mean) * (yhi - y_hat_mean))
        .sum::<f64>();
    let ss_yhat: f64 = y_hat.iter().map(|yhi| (yhi - y_hat_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 && ss_yhat > 0.0 {
        (cov_y_yhat * cov_y_yhat) / (sst * ss_yhat)
    } else {
        0.0
    };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // F-statistic (Wald test: all slope coefficients = 0)
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let f_stat = if k > 1 && ssr > 0.0 {
        (sst - ssr) / ((k - 1) as f64) / (ssr / df as f64)
    } else {
        0.0
    };
    let f_p_value =
        crate::traits::estimator::f_test_p_value(f_stat, (k.saturating_sub(1)) as f64, df as f64);

    // t-statistics and p-values
    let beta_vec: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = beta_vec
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

    let coef_arr = ndarray::Array1::from_vec(beta_vec.clone());
    let se_arr = ndarray::Array1::from_vec(std_errors.clone());

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
        coef_arr,
        se_arr,
        residuals,
        vcov,
        theta: Some(theta),
    })
}
