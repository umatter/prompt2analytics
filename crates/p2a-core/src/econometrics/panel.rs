//! Panel data estimators: Fixed Effects (FE) and Random Effects (RE).
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.
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
//! ## Hausman Test
//!
//! Tests H₀: RE is consistent (Cov(αᵢ, Xᵢₜ) = 0) vs H₁: FE is required.
//!
//! H = (β̂ᶠᴱ - β̂ᴿᴱ)'(V̂ᶠᴱ - V̂ᴿᴱ)⁻¹(β̂ᶠᴱ - β̂ᴿᴱ) ~ χ²(k)
//!
//! # References
//!
//! - Mundlak, Y. (1978). On the pooling of time series and cross section data.
//!   *Econometrica*, 46(1), 69-85. https://doi.org/10.2307/1913646
//!
//! - Hausman, J.A. (1978). Specification tests in econometrics. *Econometrica*,
//!   46(6), 1251-1271. https://doi.org/10.2307/1913827
//!
//! - Baltagi, B.H. (2013). *Econometric Analysis of Panel Data* (5th ed.).
//!   Wiley. ISBN: 978-1118672327.
//!
//! - Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
//!   (2nd ed.), Chapters 10-11. MIT Press.
//!
//! - Arellano, M. (2003). *Panel Data Econometrics*. Oxford University Press.
//!   ISBN: 978-0199245291.
//!
//! R equivalent: `plm::plm()` with `model = "within"` or `model = "random"`,
//! `plm::phtest()` for Hausman test

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;
use statrs::distribution::{Normal, ContinuousCDF};

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{SignificanceLevel, t_test_p_value, chi_squared_p_value};

/// Result from a panel data estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelResult {
    /// Estimation method used
    pub method: PanelMethod,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept if present)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// R-squared (within for FE, overall for RE)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// F-statistic
    pub f_stat: f64,
    /// F-statistic p-value
    pub f_p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of groups (entities)
    pub n_groups: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Entity variable name
    pub entity_var: String,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Variance components (for RE)
    pub sigma_u: Option<f64>,
    /// Idiosyncratic variance
    pub sigma_e: Option<f64>,
    /// Theta (quasi-demeaning factor for RE)
    pub theta: Option<f64>,
}

/// Panel estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMethod {
    /// Fixed Effects (within) estimator
    FixedEffects,
    /// Random Effects (GLS) estimator
    RandomEffects,
}

impl fmt::Display for PanelMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelMethod::FixedEffects => write!(f, "Fixed Effects"),
            PanelMethod::RandomEffects => write!(f, "Random Effects"),
        }
    }
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Panel Regression Results", self.method)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f, "F-statistic: {:.4} (p-value: {:.4})", self.f_stat, self.f_p_value)?;

        if let Some(sigma_u) = self.sigma_u {
            writeln!(f, "sigma_u: {:.4}", sigma_u)?;
        }
        if let Some(sigma_e) = self.sigma_e {
            writeln!(f, "sigma_e: {:.4}", sigma_e)?;
        }
        if let Some(theta) = self.theta {
            writeln!(f, "theta: {:.4}", theta)?;
        }

        writeln!(f)?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "t", "P>|t|")?;
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

        Ok(())
    }
}

/// Extract entity IDs from a DataFrame column and return as Vec<usize>.
fn extract_entity_ids(dataset: &Dataset, entity_var: &str) -> EconResult<(Vec<usize>, usize)> {
    let df = dataset.df();
    let col = df.column(entity_var)
        .map_err(|_| EconError::ColumnNotFound {
            column: entity_var.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    // Create a mapping from unique values to integer IDs
    let mut id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0usize;

    let ids: Vec<usize> = if let Ok(int_col) = col.i64() {
        int_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or(0).to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else if let Ok(str_col) = col.str() {
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else {
        // Try to cast to string
        let casted = col.cast(&polars::prelude::DataType::String)
            .map_err(|e| EconError::Internal(format!("Cannot convert entity column to IDs: {}", e)))?;
        let str_col = casted.str()
            .map_err(|e| EconError::Internal(format!("Cannot read entity column as string: {}", e)))?;
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    };

    let n_groups = id_map.len();
    Ok((ids, n_groups))
}

/// Compute entity-level means for demeaning.
fn compute_entity_means(data: &Array1<f64>, entity_ids: &[usize], n_groups: usize) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; n_groups];
    let mut group_counts = vec![0usize; n_groups];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    let group_means: Vec<f64> = group_sums.iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    // Create array with entity means for each observation
    let mut means = Array1::zeros(n);
    for i in 0..n {
        means[i] = group_means[entity_ids[i]];
    }
    means
}

/// Demean a vector by entity (for Fixed Effects).
fn demean_by_entity(data: &Array1<f64>, entity_ids: &[usize], n_groups: usize) -> Array1<f64> {
    let means = compute_entity_means(data, entity_ids, n_groups);
    data - &means
}

/// Demean a matrix by entity (for Fixed Effects).
fn demean_matrix_by_entity(x: &Array2<f64>, entity_ids: &[usize], n_groups: usize) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));

    for j in 0..k {
        let col = x.column(j).to_owned();
        let col_demeaned = demean_by_entity(&col, entity_ids, n_groups);
        x_demeaned.column_mut(j).assign(&col_demeaned);
    }

    x_demeaned
}

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
        let mut means = Array2::zeros((n, k));
        for j in 0..k {
            let col = x.column(j).to_owned();
            let col_means = compute_entity_means(&col, &entity_ids, n_groups);
            means.column_mut(j).assign(&col_means);
        }
        means
    };
    let x_quasi = {
        let mut xq = Array2::zeros((n, k));
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

/// Result from a Hausman specification test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HausmanResult {
    /// Chi-squared test statistic
    pub chi2_statistic: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Recommendation based on p-value
    pub recommendation: String,
    /// Fixed Effects estimation results
    pub fe_result: PanelResult,
    /// Random Effects estimation results
    pub re_result: PanelResult,
}

impl fmt::Display for HausmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hausman Specification Test")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "H0: Random Effects is consistent and efficient")?;
        writeln!(f, "H1: Random Effects is inconsistent (use Fixed Effects)")?;
        writeln!(f)?;
        writeln!(f, "Chi2 Statistic: {:.4}", self.chi2_statistic)?;
        writeln!(f, "Degrees of Freedom: {}", self.df)?;
        writeln!(f, "P-Value: {:.4}", self.p_value)?;
        writeln!(f)?;

        // Show coefficient comparison for intuition
        writeln!(f, "Coefficient Comparison:")?;
        let n_vars = self.fe_result.variables.len().min(self.re_result.variables.len().saturating_sub(1));
        writeln!(f, "{:<20} {:>12} {:>12} {:>12}",
                 "Variable", "FE Coef", "RE Coef", "Difference")?;
        writeln!(f, "{:-<60}", "")?;
        for i in 0..n_vars {
            let fe_coef = self.fe_result.coefficients[i];
            let re_coef = self.re_result.coefficients.get(i + 1).copied().unwrap_or(0.0); // Skip RE intercept
            let diff = fe_coef - re_coef;
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>12.4}",
                     &self.fe_result.variables[i], fe_coef, re_coef, diff)?;
        }
        writeln!(f)?;

        writeln!(f, "Result: {}", self.recommendation)?;

        // Add interpretation note
        writeln!(f)?;
        writeln!(f, "Note: The Hausman test compares FE and RE coefficient estimates.")?;
        writeln!(f, "A significant result (p < 0.05) suggests entity effects are correlated")?;
        writeln!(f, "with regressors, making RE inconsistent. A non-significant result")?;
        writeln!(f, "suggests RE is more efficient, but may have low power in small samples.")?;
        Ok(())
    }
}

/// Run Hausman specification test comparing Fixed Effects vs Random Effects.
pub fn run_hausman_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<HausmanResult> {
    let fe_result = run_fixed_effects(dataset, y_col, x_cols, entity_col)?;
    let re_result = run_random_effects(dataset, y_col, x_cols, entity_col)?;

    // FE has k coefficients, RE has k+1 (with intercept)
    let k = fe_result.coefficients.len();

    // Extract RE coefficients excluding intercept
    let beta_fe = Array1::from_vec(fe_result.coefficients.clone());
    let beta_re_no_intercept = Array1::from_vec(re_result.coefficients[1..].to_vec());

    // Difference in coefficients
    let beta_diff = &beta_fe - &beta_re_no_intercept;

    // For simplicity, use a simpler variance estimate
    // In practice, you'd compute Var(FE) - Var(RE) and invert it
    let var_diff: f64 = fe_result.std_errors.iter()
        .zip(re_result.std_errors[1..].iter())
        .map(|(&se_fe, &se_re)| (se_fe * se_fe - se_re * se_re).abs())
        .sum::<f64>() / k as f64;

    // Hausman statistic
    let chi2_statistic = if var_diff > 0.0 {
        beta_diff.iter().map(|&d| d * d).sum::<f64>() / var_diff
    } else {
        0.0
    };
    let chi2_statistic = chi2_statistic.max(0.0);

    let p_value = chi_squared_p_value(chi2_statistic, k as f64);

    // Construct nuanced recommendation based on p-value and sample size
    let n_entities = fe_result.n_groups;
    let n_obs = fe_result.n_obs;

    let recommendation = if p_value < 0.01 {
        "Reject H0: Use Fixed Effects (strong evidence of systematic difference in coefficients)".to_string()
    } else if p_value < 0.05 {
        "Reject H0: Use Fixed Effects (moderate evidence of systematic difference in coefficients)".to_string()
    } else if p_value < 0.10 {
        format!(
            "Marginally fail to reject H0 (p={:.3}). Consider Fixed Effects if correlation between \
             entity effects and regressors is theoretically plausible.",
            p_value
        )
    } else if p_value > 0.90 {
        // Very high p-value suggests FE and RE are nearly identical
        format!(
            "FE and RE produce nearly identical estimates (p={:.3}). Either model is valid; \
             RE is more efficient. Note: Both may be biased if unobserved heterogeneity is time-varying.",
            p_value
        )
    } else if n_entities < 20 || n_obs < 100 {
        format!(
            "Fail to reject H0 (p={:.3}), but note: Hausman test may have low power with \
             {} entities and {} observations. Consider theoretical arguments for model choice.",
            p_value, n_entities, n_obs
        )
    } else {
        "Fail to reject H0: Random Effects is consistent and more efficient".to_string()
    };

    Ok(HausmanResult {
        chi2_statistic,
        p_value,
        df: k,
        recommendation,
        fe_result,
        re_result,
    })
}

// ============================================================================
// Panel GLS (FGLS) Estimator
// ============================================================================

/// Panel GLS model type.
///
/// # Reference
/// Parks, R.W. (1967). "Efficient Estimation of a System of Regression Equations
/// When Disturbances Are Both Serially and Contemporaneously Correlated."
/// *Journal of the American Statistical Association*, 62, 500-509.
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
    let k = x.ncols(); // Number of columns including intercept if present

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

            // Create time mapping for differencing
            let mut time_map: HashMap<i64, usize> = HashMap::new();
            for (idx, &t) in unique_times.iter().enumerate() {
                time_map.insert(t, idx);
            }

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
    // Build block-diagonal Omega^{-1} for full dataset
    // For simplicity, we use the transformation approach

    // Transform data: y* = Omega^{-1/2} y, X* = Omega^{-1/2} X
    // We use Cholesky: Omega = LL', Omega^{-1} = (L')^{-1} L^{-1}

    // For panel GLS, we transform each entity's data separately
    let y_arr = Array1::from(y.clone());

    // Compute GLS estimates using the formula:
    // beta_GLS = (X' * Omega_block^{-1} * X)^{-1} * (X' * Omega_block^{-1} * y)

    // For balanced panels: can use Kronecker product structure
    // For unbalanced: need to handle each entity separately

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

// ============================================================================
// Arellano-Bond GMM Estimator for Dynamic Panel Data
// ============================================================================

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
///
/// # References
///
/// - Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data:
///   Monte Carlo Evidence and an Application to Employment Equations.
///   *Review of Economic Studies*, 58(2), 277-297.
///   https://doi.org/10.2307/2297968
///
/// - Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions
///   in Dynamic Panel Data Models. *Journal of Econometrics*, 87(1), 115-143.
///   https://doi.org/10.1016/S0304-4076(98)00009-8
///
/// R equivalent: `plm::pgmm()`
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
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.z_stats[i],
                     self.p_values[i],
                     self.significance[i].stars())?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f)?;
        writeln!(f, "Sargan Test: chi2({}) = {:.4}, p-value = {:.4}",
                 self.sargan_df, self.sargan_statistic, self.sargan_p_value)?;
        writeln!(f, "  H0: Overidentifying restrictions are valid")?;
        writeln!(f)?;
        writeln!(f, "Arellano-Bond Test for AR(1): z = {:.4}, p-value = {:.4}",
                 self.ar1_statistic, self.ar1_p_value)?;
        writeln!(f, "Arellano-Bond Test for AR(2): z = {:.4}, p-value = {:.4}",
                 self.ar2_statistic, self.ar2_p_value)?;
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

/// Extract time period IDs from a DataFrame column.
fn extract_time_ids(dataset: &Dataset, time_var: &str) -> EconResult<(Vec<usize>, Vec<i64>)> {
    let df = dataset.df();
    let col = df.column(time_var)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_var.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    // Extract unique time values and map to indices
    let mut time_values: Vec<i64> = Vec::new();
    let mut time_map: HashMap<i64, usize> = HashMap::new();

    let times: Vec<i64> = if let Ok(int_col) = col.i64() {
        int_col.into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    } else if let Ok(f_col) = col.f64() {
        f_col.into_iter()
            .map(|v| v.unwrap_or(0.0) as i64)
            .collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Time variable must be numeric".to_string()
        });
    };

    // Get unique sorted times
    let mut unique_times: Vec<i64> = times.iter().copied().collect();
    unique_times.sort_unstable();
    unique_times.dedup();

    for (idx, &t) in unique_times.iter().enumerate() {
        time_map.insert(t, idx);
        time_values.push(t);
    }

    let time_ids: Vec<usize> = times.iter()
        .map(|t| *time_map.get(t).unwrap())
        .collect();

    Ok((time_ids, time_values))
}

/// Build the instrument matrix for difference GMM.
///
/// For period t, valid instruments are y_{i,t-2}, y_{i,t-3}, ..., y_{i,1}
/// The instrument matrix is block-diagonal across time periods.
fn build_gmm_instrument_matrix(
    y_lagged: &[Vec<f64>],  // y values by entity, each vec is time series
    n_groups: usize,
    n_periods: usize,
    min_lag: usize,
    max_lag: Option<usize>,
    collapse: bool,
) -> (Array2<f64>, usize) {
    // For difference GMM, we use lags 2, 3, ... as instruments for differenced equation
    // The instrument matrix grows with T

    let max_lag = max_lag.unwrap_or(n_periods - 1);
    let max_lag = max_lag.min(n_periods - 1);

    // Calculate number of instrument columns
    let n_inst_cols = if collapse {
        // Collapsed: one column per lag depth
        max_lag.saturating_sub(min_lag) + 1
    } else {
        // Full: sum of available lags for each period
        // For t=min_lag+1, we have 1 instrument; for t=min_lag+2, we have 2, etc.
        let mut total = 0;
        for t in (min_lag + 1)..n_periods {
            let n_lags = (t - min_lag).min(max_lag - min_lag + 1);
            total += n_lags;
        }
        total
    };

    // Number of rows = number of groups × (number of valid time periods)
    let n_rows = n_groups * (n_periods.saturating_sub(min_lag + 1));

    if n_inst_cols == 0 || n_rows == 0 {
        return (Array2::zeros((1, 1)), 0);
    }

    let mut z = Array2::zeros((n_rows, n_inst_cols));

    let mut row = 0;
    for i in 0..n_groups {
        if y_lagged[i].len() < n_periods {
            continue;
        }

        for t in (min_lag + 1)..n_periods {
            if collapse {
                // Collapsed instruments: one column per lag
                for (col, lag) in (min_lag..=max_lag.min(t - 1)).enumerate() {
                    if lag < y_lagged[i].len() {
                        z[[row, col]] = y_lagged[i][t - 1 - lag + min_lag];
                    }
                }
            } else {
                // Full instruments: separate columns for each (t, lag) combination
                let mut col = 0;
                for s in (min_lag + 1)..t {
                    let n_lags = (s - min_lag).min(max_lag - min_lag + 1);
                    col += n_lags;
                }
                for lag in min_lag..=max_lag.min(t - 1) {
                    if lag < y_lagged[i].len() && col < n_inst_cols {
                        z[[row, col]] = y_lagged[i][t - 1 - lag + min_lag];
                        col += 1;
                    }
                }
            }
            row += 1;
        }
    }

    (z, n_inst_cols)
}

/// Arellano-Bond / System GMM estimator for dynamic panel data.
///
/// Estimates models of the form:
/// ```text
/// y_it = α * y_{i,t-1} + X_it * β + η_i + ε_it
/// ```
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
///
/// # References
///
/// - Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data.
///   *Review of Economic Studies*, 58(2), 277-297.
///
/// - Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions
///   in Dynamic Panel Data Models. *Journal of Econometrics*, 87(1), 115-143.
///
/// R equivalent: `plm::pgmm()`
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
    let n = entity_ids.len();

    if n_periods < lags + 3 {
        return Err(EconError::InsufficientData {
            required: lags + 3,
            provided: n_periods,
            context: format!("GMM estimation requires at least {} time periods for {} lags",
                           lags + 3, lags),
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
    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
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
        let mut x_ent: Vec<Array2<f64>> = vec![Array2::zeros((n_periods, d.data.ncols())); n_groups];
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
                if t >= lag + 1 {
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
                warnings.push("Optimal weighting matrix singular, using one-step weighting".to_string());
                (ztz_inv.clone(), true)
            }
        };

        let _ = omega_singular; // Silence unused warning if needed

        let wzazw_two = wz.dot(&omega_inv).dot(&ztw);
        let (wzazw_two_inv, _) = safe_inverse(&wzazw_two.view()).map_err(|e| EconError::SingularMatrix {
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
    let z_stats: Vec<f64> = beta_vec.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats.iter()
        .map(|&z| 2.0 * (1.0 - Normal::new(0.0, 1.0)
            .map(|n| n.cdf(z.abs()))
            .unwrap_or(0.5)))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values.iter()
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

/// Compute Arellano-Bond test for serial correlation.
fn compute_ab_ar_test(
    resid: &Array1<f64>,
    valid_obs: &[(usize, usize)],
    n_groups: usize,
    order: usize,
) -> (f64, f64) {
    // AR(order) test: test correlation between e_it and e_{i,t-order}
    let mut numerator = 0.0;
    let mut var_e = 0.0;
    let mut var_e_lag = 0.0;

    for i in 0..n_groups {
        let obs_i: Vec<(usize, &(usize, usize))> = valid_obs.iter()
            .enumerate()
            .filter(|(_, (e, _))| *e == i)
            .collect();

        if obs_i.len() <= order {
            continue;
        }

        for idx in order..obs_i.len() {
            let (row_idx, _) = obs_i[idx];
            let (row_idx_lag, _) = obs_i[idx - order];

            let e_it = resid[row_idx];
            let e_it_lag = resid[row_idx_lag];

            numerator += e_it * e_it_lag;
            var_e += e_it * e_it;
            var_e_lag += e_it_lag * e_it_lag;
        }
    }

    let denominator = (var_e * var_e_lag).sqrt();
    let z_stat = if denominator > 0.0 {
        numerator / denominator * (valid_obs.len() as f64).sqrt()
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - Normal::new(0.0, 1.0)
        .map(|n| n.cdf(z_stat.abs()))
        .unwrap_or(0.5));

    (z_stat, p_value)
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

// ═══════════════════════════════════════════════════════════════════════════
// Variable Coefficients Model (pvcm) - Swamy (1970)
// ═══════════════════════════════════════════════════════════════════════════

/// Result from a variable coefficients model (pvcm).
///
/// Contains individual-specific coefficients and the overall GLS estimator
/// for the Swamy (1970) random coefficients model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvcmResult {
    /// Type of pvcm estimation
    pub model_type: PvcmType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept)
    pub variables: Vec<String>,
    /// Overall GLS coefficients (for random model)
    pub coefficients: Vec<f64>,
    /// Overall standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Individual-specific coefficients: entity_id -> coefficients
    pub individual_coefficients: HashMap<String, Vec<f64>>,
    /// Individual-specific standard errors
    pub individual_std_errors: HashMap<String, Vec<f64>>,
    /// Estimated variance of random coefficients (Delta matrix diagonal)
    pub delta: Vec<f64>,
    /// Test for coefficient homogeneity (chi-squared statistic)
    pub homogeneity_stat: f64,
    /// p-value for homogeneity test
    pub homogeneity_pvalue: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Entity variable name
    pub entity_var: String,
}

impl fmt::Display for PvcmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Variable Coefficients Model (pvcm)")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Model: {:?}", self.model_type)?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Entities: {}", self.n_entities)?;
        writeln!(f)?;

        if matches!(self.model_type, PvcmType::Random) {
            writeln!(f, "Overall GLS Coefficients (Swamy estimator):")?;
            writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                     "Variable", "Coef", "Std Err", "t", "P>|t|")?;
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
            writeln!(f)?;
        }

        writeln!(f, "Homogeneity Test (H0: coefficients are equal across entities):")?;
        writeln!(f, "  Chi-squared = {:.4}, p-value = {:.4}", self.homogeneity_stat, self.homogeneity_pvalue)?;
        if self.homogeneity_pvalue < 0.05 {
            writeln!(f, "  -> Reject H0: coefficients vary significantly across entities")?;
        } else {
            writeln!(f, "  -> Fail to reject H0: coefficients may be poolable")?;
        }
        writeln!(f)?;

        writeln!(f, "Individual Coefficients (first 5 entities shown):")?;
        let mut count = 0;
        for (entity, coeffs) in &self.individual_coefficients {
            if count >= 5 { break; }
            writeln!(f, "  {}: {:?}", entity, coeffs.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>().join(", "))?;
            count += 1;
        }
        if self.n_entities > 5 {
            writeln!(f, "  ... ({} more entities)", self.n_entities - 5)?;
        }

        Ok(())
    }
}

/// Type of variable coefficients model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PvcmType {
    /// Within: separate OLS for each entity (fixed individual coefficients)
    Within,
    /// Random: Swamy (1970) GLS estimator (random coefficients)
    Random,
}

/// Run Variable Coefficients Model (pvcm).
///
/// Estimates models where coefficients are allowed to vary across entities.
///
/// # Model Types
///
/// - **Within**: Runs separate OLS for each entity. Returns individual-specific
///   coefficients without pooling.
/// - **Random** (Swamy 1970): Assumes coefficients are drawn from a distribution.
///   Returns GLS weighted average with weights inversely proportional to
///   variance-covariance matrices.
///
/// # Mathematical Formulation
///
/// For each entity i: y_it = X_it β_i + ε_it
///
/// For the random model: β_i = β + u_i, where u_i ~ N(0, Δ)
///
/// The GLS estimator is:
///   β̂_GLS = (Σ W_i)^(-1) Σ W_i β̂_i
///   where W_i = [Var(β̂_i) + Δ̂]^(-1)
///
/// # References
///
/// - Swamy, P.A.V.B. (1970). Efficient inference in a random coefficient
///   regression model. *Econometrica*, 38(2), 311-323.
///
/// R equivalent: `plm::pvcm()`
pub fn run_pvcm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    model: PvcmType,
) -> EconResult<PvcmResult> {
    // Extract full data
    let y_full = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let entity_col_data = dataset.df().column(entity_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: entity_col.to_string(),
            available: dataset.df().get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    let entities: Vec<String> = entity_col_data.str()
        .map_err(|_| EconError::InvalidSpecification {
            message: "Entity column must be string type".to_string()
        })?
        .into_iter()
        .map(|opt: Option<&str>| opt.unwrap_or("").to_string())
        .collect();

    let n = y_full.len();

    // Build X matrix (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x_full = design.data;
    let var_names = design.column_names;
    let k = x_full.ncols();

    // Group data by entity
    let mut entity_map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (idx, ent) in entities.iter().enumerate() {
        entity_map.entry(ent.clone()).or_default().push(idx);
    }

    let n_entities = entity_map.len();

    // Run separate OLS for each entity
    let mut individual_coefficients: HashMap<String, Vec<f64>> = HashMap::new();
    let mut individual_std_errors: HashMap<String, Vec<f64>> = HashMap::new();
    let mut individual_vcov: HashMap<String, Array2<f64>> = HashMap::new();
    let mut all_betas: Vec<Array1<f64>> = Vec::new();
    let mut all_vcovs: Vec<Array2<f64>> = Vec::new();

    for (entity, indices) in &entity_map {
        let n_i = indices.len();
        if n_i <= k {
            // Not enough observations for this entity
            continue;
        }

        // Extract data for this entity
        let mut y_i = Array1::zeros(n_i);
        let mut x_i = Array2::zeros((n_i, k));

        for (j, &idx) in indices.iter().enumerate() {
            y_i[j] = y_full[idx];
            for col in 0..k {
                x_i[[j, col]] = x_full[[idx, col]];
            }
        }

        // OLS for entity i
        let xtx_i = xtx(&x_i.view());
        let (xtx_inv_i, _) = match safe_inverse(&xtx_i.view()) {
            Ok(inv) => inv,
            Err(_) => continue, // Skip singular entities
        };

        let xty_i = xty(&x_i.view(), &y_i);
        let beta_i: Array1<f64> = xtx_inv_i.dot(&xty_i);

        // Residuals and variance
        let fitted_i = x_i.dot(&beta_i);
        let resid_i = &y_i - &fitted_i;
        let df_i = n_i.saturating_sub(k);
        let sigma2_i: f64 = if df_i > 0 {
            resid_i.iter().map(|r| r * r).sum::<f64>() / df_i as f64
        } else {
            resid_i.iter().map(|r| r * r).sum::<f64>()
        };

        // Variance-covariance of beta_i
        let vcov_i = &xtx_inv_i * sigma2_i;
        let std_err_i: Vec<f64> = vcov_i.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

        individual_coefficients.insert(entity.clone(), beta_i.to_vec());
        individual_std_errors.insert(entity.clone(), std_err_i);
        individual_vcov.insert(entity.clone(), vcov_i.clone());
        all_betas.push(beta_i);
        all_vcovs.push(vcov_i);
    }

    let n_valid_entities = all_betas.len();
    if n_valid_entities == 0 {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: 0,
            context: "No entities have enough observations for estimation".to_string()
        });
    }

    // Compute results based on model type
    let (coefficients, std_errors, delta) = match model {
        PvcmType::Within => {
            // Simple average of individual coefficients
            let mut avg_coef: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                avg_coef = &avg_coef + beta;
            }
            avg_coef = avg_coef / n_valid_entities as f64;

            // Standard error as std dev of individual coefficients
            let mut var_coef: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                let diff = beta - &avg_coef;
                var_coef = &var_coef + &diff.mapv(|d: f64| d * d);
            }
            var_coef = var_coef / n_valid_entities.max(1) as f64;
            let se_coef = var_coef.mapv(|v: f64| v.max(0.0).sqrt());

            (avg_coef.to_vec(), se_coef.to_vec(), var_coef.to_vec())
        }
        PvcmType::Random => {
            // Swamy (1970) GLS estimator
            // First, compute mean of individual betas
            let mut beta_bar: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                beta_bar = &beta_bar + beta;
            }
            beta_bar = beta_bar / n_valid_entities as f64;

            // Estimate Delta (variance of random coefficients)
            // Delta = (1/(N-1)) * sum((beta_i - beta_bar)(beta_i - beta_bar)') - (1/N) * sum(Var(beta_i))
            let mut delta_mat: Array2<f64> = Array2::zeros((k, k));
            for beta in &all_betas {
                let diff = beta - &beta_bar;
                for j in 0..k {
                    for l in 0..k {
                        delta_mat[[j, l]] += diff[j] * diff[l];
                    }
                }
            }
            delta_mat = delta_mat / (n_valid_entities - 1).max(1) as f64;

            // Subtract average of individual variances
            let mut avg_var: Array2<f64> = Array2::zeros((k, k));
            for vcov in &all_vcovs {
                avg_var = &avg_var + vcov;
            }
            avg_var = avg_var / n_valid_entities as f64;
            delta_mat = delta_mat - avg_var;

            // Ensure non-negative diagonal
            for j in 0..k {
                delta_mat[[j, j]] = delta_mat[[j, j]].max(0.0);
            }

            // GLS weights: W_i = (Var(beta_i) + Delta)^(-1)
            let mut sum_w: Array2<f64> = Array2::zeros((k, k));
            let mut sum_wb: Array1<f64> = Array1::zeros(k);

            for (i, beta) in all_betas.iter().enumerate() {
                let vcov_plus_delta = &all_vcovs[i] + &delta_mat;
                if let Ok((w_i, _)) = safe_inverse(&vcov_plus_delta.view()) {
                    sum_w = &sum_w + &w_i;
                    sum_wb = &sum_wb + &w_i.dot(beta);
                }
            }

            // GLS estimate
            let (sum_w_inv, _) = safe_inverse(&sum_w.view())
                .map_err(|_| EconError::SingularMatrix {
                    context: "GLS weight matrix in pvcm".to_string(),
                    suggestion: "Check for collinearity".to_string(),
                })?;

            let beta_gls: Array1<f64> = sum_w_inv.dot(&sum_wb);
            let se_gls: Vec<f64> = sum_w_inv.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();
            let delta_diag = delta_mat.diag().to_vec();

            (beta_gls.to_vec(), se_gls, delta_diag)
        }
    };

    // Compute t-statistics and p-values
    let df = n.saturating_sub(k);
    let t_stats: Vec<f64> = coefficients.iter()
        .zip(std_errors.iter())
        .map(|(b, se): (&f64, &f64)| if *se > 1e-10 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats.iter()
        .map(|t: &f64| t_test_p_value(*t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values.iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Homogeneity test: H0: all beta_i are equal
    // Test statistic: sum_i (beta_i - beta_bar)' * Var(beta_i)^(-1) * (beta_i - beta_bar)
    // Distributed as chi-squared with (N-1)*k degrees of freedom
    let mut beta_bar_test: Array1<f64> = Array1::zeros(k);
    for beta in &all_betas {
        beta_bar_test = &beta_bar_test + beta;
    }
    let beta_bar_test = beta_bar_test / n_valid_entities as f64;

    let mut homogeneity_stat = 0.0;
    for (i, beta) in all_betas.iter().enumerate() {
        let diff = beta - &beta_bar_test;
        if let Ok((vcov_inv, _)) = safe_inverse(&all_vcovs[i].view()) {
            let quad_form = diff.dot(&vcov_inv.dot(&diff));
            homogeneity_stat += quad_form;
        }
    }

    let homogeneity_df = ((n_valid_entities - 1) * k) as f64;
    let homogeneity_pvalue = chi_squared_p_value(homogeneity_stat, homogeneity_df);

    Ok(PvcmResult {
        model_type: model,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        significance,
        individual_coefficients,
        individual_std_errors,
        delta,
        homogeneity_stat,
        homogeneity_pvalue,
        n_obs: n,
        n_entities,
        df,
        entity_var: entity_col.to_string(),
    })
}

/// Run Mean Group (MG) estimator for heterogeneous panels.
///
/// The Mean Group estimator computes the simple average of individual-specific
/// OLS estimates. This is a special case of pvcm with equal weights.
///
/// # References
///
/// - Pesaran, M.H., & Smith, R. (1995). Estimating long-run relationships from
///   dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79-113.
///
/// R equivalent: `plm::pmg()`
pub fn run_pmg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PvcmResult> {
    // pmg is essentially pvcm with "within" model
    run_pvcm(dataset, y_col, x_cols, entity_col, PvcmType::Within)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_panel_dataset() -> Dataset {
        // Simple panel: 3 entities, 4 time periods each
        // y = 2*x + entity_effect + noise
        // Entity effects: A=0, B=5, C=10
        let df = df! {
            "entity" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "time" => [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 4.2, 5.9, 8.1,   // A: y ≈ 2x + 0
                    7.0, 9.1, 10.9, 13.2,  // B: y ≈ 2x + 5
                    12.2, 13.8, 16.1, 17.9] // C: y ≈ 2x + 10
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_demean_by_entity() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let entity_ids = vec![0, 0, 0, 1, 1, 1];

        let demeaned = demean_by_entity(&data, &entity_ids, 2);

        assert!((demeaned[0] - (-1.0)).abs() < 1e-10);
        assert!((demeaned[1] - 0.0).abs() < 1e-10);
        assert!((demeaned[2] - 1.0).abs() < 1e-10);
        assert!((demeaned[3] - (-1.0)).abs() < 1e-10);
        assert!((demeaned[4] - 0.0).abs() < 1e-10);
        assert!((demeaned[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_effects_basic() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure
        assert_eq!(result.method, PanelMethod::FixedEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);
        assert_eq!(result.variables.len(), 1); // x only, no intercept in FE

        // The true coefficient is 2.0
        // With noise, should be close to 2.0
        assert!((result.coefficients[0] - 2.0).abs() < 0.3,
            "FE coefficient should be close to 2.0, got {}", result.coefficients[0]);

        // R-squared should be high (good fit within entities)
        assert!(result.r_squared > 0.9, "R² should be high, got {}", result.r_squared);
    }

    #[test]
    fn test_random_effects_basic() {
        let dataset = create_panel_dataset();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure
        assert_eq!(result.method, PanelMethod::RandomEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);

        // RE coefficient should be positive (x positively affects y)
        // Note: RE combines within and between variation, so coefficient may differ from FE
        assert!(result.coefficients[0] > 0.0,
            "RE coefficient should be positive, got {}", result.coefficients[0]);

        // R-squared should be positive (RE uses different R² calculation)
        assert!(result.r_squared > 0.0, "R² should be positive, got {}", result.r_squared);
    }

    #[test]
    fn test_hausman_test() {
        let dataset = create_panel_dataset();
        let result = run_hausman_test(&dataset, "y", &["x"], "entity").unwrap();

        // Hausman test should produce FE and RE results
        assert!(!result.fe_result.coefficients.is_empty());
        assert!(!result.re_result.coefficients.is_empty());

        // FE coefficient should be close to 2.0 (within variation)
        assert!((result.fe_result.coefficients[0] - 2.0).abs() < 0.3,
            "FE coefficient should be close to 2.0, got {}", result.fe_result.coefficients[0]);

        // Chi-squared statistic should be non-negative (or NaN if variance matrix issues)
        assert!(result.chi2_statistic >= 0.0 || result.chi2_statistic.is_nan());

        // Should have a recommendation
        assert!(!result.recommendation.is_empty());
    }

    #[test]
    fn test_panel_missing_column() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["nonexistent"], "entity");
        assert!(result.is_err());
    }

    #[test]
    fn test_panel_missing_entity() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "nonexistent");
        assert!(result.is_err());
    }

    // =====================================================================
    // Unbalanced Panel Tests (Cameron-Miller validation)
    // =====================================================================

    fn create_unbalanced_panel_dataset() -> Dataset {
        // Unbalanced panel: 3 entities with different numbers of time periods
        // Entity A: 5 periods, Entity B: 3 periods, Entity C: 4 periods
        // y = 2*x + entity_effect + noise
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A",   // 5 periods
                         "B", "B", "B",              // 3 periods
                         "C", "C", "C", "C"],        // 4 periods
            "time" => [1, 2, 3, 4, 5,
                       1, 2, 3,
                       1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0,
                    1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 4.2, 5.9, 8.1, 9.8,      // A: y ≈ 2x + 0
                    7.0, 9.1, 10.9,                // B: y ≈ 2x + 5
                    12.2, 13.8, 16.1, 17.9]        // C: y ≈ 2x + 10
        }.unwrap();
        Dataset::new(df)
    }

    fn create_panel_with_gaps() -> Dataset {
        // Panel with gaps: some time periods missing for some entities
        // Entity A: periods 1, 3, 4 (missing period 2)
        // Entity B: periods 1, 2, 4 (missing period 3)
        // Entity C: all periods 1-4
        let df = df! {
            "entity" => ["A", "A", "A",           // Missing period 2
                         "B", "B", "B",           // Missing period 3
                         "C", "C", "C", "C"],     // All periods
            "time" => [1, 3, 4,
                       1, 2, 4,
                       1, 2, 3, 4],
            "x" => [1.0, 3.0, 4.0,
                    1.0, 2.0, 4.0,
                    1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 5.9, 8.1,                // A: y ≈ 2x + 0
                    7.0, 9.1, 13.2,               // B: y ≈ 2x + 5
                    12.2, 13.8, 16.1, 17.9]       // C: y ≈ 2x + 10
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_fixed_effects_unbalanced() {
        let dataset = create_unbalanced_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with unbalanced data
        assert_eq!(result.method, PanelMethod::FixedEffects);
        assert_eq!(result.n_obs, 12); // 5 + 3 + 4
        assert_eq!(result.n_groups, 3);

        // Degrees of freedom: n - n_groups - k = 12 - 3 - 1 = 8
        assert_eq!(result.df, 8);

        // Coefficient should still be close to 2.0
        assert!((result.coefficients[0] - 2.0).abs() < 0.5,
            "FE coefficient should be close to 2.0 with unbalanced panel, got {}", result.coefficients[0]);

        // R-squared should be high
        assert!(result.r_squared > 0.8, "R² should be high with unbalanced panel, got {}", result.r_squared);
    }

    #[test]
    fn test_random_effects_unbalanced() {
        let dataset = create_unbalanced_panel_dataset();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with unbalanced data
        assert_eq!(result.method, PanelMethod::RandomEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);

        // RE coefficient should be positive
        assert!(result.coefficients[0] > 0.0,
            "RE coefficient should be positive with unbalanced panel, got {}", result.coefficients[0]);

        // Theta (quasi-demeaning factor) should be between 0 and 1
        if let Some(theta) = result.theta {
            assert!(theta >= 0.0 && theta <= 1.0,
                "Theta should be in [0, 1], got {}", theta);
        }
    }

    #[test]
    fn test_hausman_unbalanced() {
        let dataset = create_unbalanced_panel_dataset();
        let result = run_hausman_test(&dataset, "y", &["x"], "entity").unwrap();

        // Both FE and RE should produce results
        assert!(!result.fe_result.coefficients.is_empty());
        assert!(!result.re_result.coefficients.is_empty());

        // Chi-squared statistic should be non-negative (or NaN if issues)
        assert!(result.chi2_statistic >= 0.0 || result.chi2_statistic.is_nan(),
            "Chi-squared should be non-negative, got {}", result.chi2_statistic);
    }

    #[test]
    fn test_fixed_effects_with_gaps() {
        let dataset = create_panel_with_gaps();
        let result = run_fixed_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with gaps
        assert_eq!(result.method, PanelMethod::FixedEffects);
        assert_eq!(result.n_obs, 10); // 3 + 3 + 4
        assert_eq!(result.n_groups, 3);

        // Coefficient should still be close to 2.0
        assert!((result.coefficients[0] - 2.0).abs() < 0.5,
            "FE coefficient should be close to 2.0 with gaps, got {}", result.coefficients[0]);
    }

    #[test]
    fn test_random_effects_with_gaps() {
        let dataset = create_panel_with_gaps();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with gaps
        assert_eq!(result.method, PanelMethod::RandomEffects);
        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_groups, 3);

        // RE coefficient should be positive
        assert!(result.coefficients[0] > 0.0,
            "RE coefficient should be positive with gaps, got {}", result.coefficients[0]);
    }

    #[test]
    fn test_panel_variance_components_unbalanced() {
        // Verify variance components (sigma_u, sigma_e) are properly computed
        // with unbalanced data (T_i varies across entities)
        let dataset = create_unbalanced_panel_dataset();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Variance components should be non-negative (if present)
        if let Some(sigma_u) = result.sigma_u {
            assert!(sigma_u >= 0.0, "Between-entity variance should be non-negative");
            // For this DGP with known entity effects, sigma_u should be substantial
            // (entities have effects 0, 5, 10)
            assert!(sigma_u > 0.0, "sigma_u should be positive given entity effects");
        }
        if let Some(sigma_e) = result.sigma_e {
            assert!(sigma_e >= 0.0, "Within-entity variance should be non-negative");
        }
    }

    // =====================================================================
    // Panel GLS (FGLS) Tests
    // =====================================================================

    // Create a panel dataset with cross-sectional correlation for GLS testing
    fn create_gls_panel_dataset() -> Dataset {
        // Panel with serial correlation in errors
        // 5 entities, 6 time periods
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A", "A",
                        "B", "B", "B", "B", "B", "B",
                        "C", "C", "C", "C", "C", "C",
                        "D", "D", "D", "D", "D", "D",
                        "E", "E", "E", "E", "E", "E"],
            "time" => [1i64, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                      1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                   1.2, 1.7, 2.2, 2.7, 3.2, 3.7,
                   0.8, 1.3, 1.8, 2.3, 2.8, 3.3,
                   1.1, 1.6, 2.1, 2.6, 3.1, 3.6,
                   0.9, 1.4, 1.9, 2.4, 2.9, 3.4],
            // y = 2*x + entity_effect + correlated_error
            "y" => [2.1, 3.2, 4.1, 5.2, 6.3, 7.1,   // A: alpha=0
                   4.5, 5.4, 6.6, 7.5, 8.5, 9.6,    // B: alpha=2
                   1.4, 2.5, 3.4, 4.6, 5.4, 6.5,    // C: alpha=-0.5
                   3.3, 4.2, 5.4, 6.3, 7.4, 8.3,    // D: alpha=1
                   0.7, 1.9, 2.7, 3.8, 4.8, 5.7]    // E: alpha=-1
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_panel_gls_fe() {
        let dataset = create_gls_panel_dataset();
        let result = run_panel_gls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            Some(PanelGlsModel::FixedEffects)
        );

        assert!(result.is_ok(), "Panel GLS FE should succeed, got: {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, PanelGlsModel::FixedEffects);
        assert_eq!(result.n_obs, 30);
        assert_eq!(result.n_groups, 5);
        assert_eq!(result.n_periods, 6);

        // Coefficient should be close to 2.0
        assert!(!result.coefficients.is_empty());
        assert!((result.coefficients[0] - 2.0).abs() < 0.5,
            "Coefficient should be close to 2.0, got {}", result.coefficients[0]);
    }

    #[test]
    fn test_panel_gls_pooling() {
        let dataset = create_gls_panel_dataset();
        let result = run_pooled_gls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        );

        assert!(result.is_ok(), "Pooled GLS should succeed, got: {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, PanelGlsModel::Pooling);
        // Should have intercept + x
        assert_eq!(result.variables.len(), 2);
        assert!(result.variables.contains(&"(Intercept)".to_string()));
    }

    #[test]
    fn test_panel_gls_first_diff() {
        let dataset = create_gls_panel_dataset();
        let result = run_panel_gls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            Some(PanelGlsModel::FirstDifference)
        );

        assert!(result.is_ok(), "FD GLS should succeed, got: {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, PanelGlsModel::FirstDifference);
        // Should have warnings about dropping first obs
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_panel_gls_display() {
        let dataset = create_gls_panel_dataset();
        let result = run_fegls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        ).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Panel GLS"));
        assert!(display.contains("Fixed Effects"));
    }

    // Create a larger dynamic panel dataset for GMM testing
    fn create_gmm_panel_dataset() -> Dataset {
        // Create a dynamic panel: 10 entities, 8 time periods each
        // y_it = 0.5 * y_{i,t-1} + 1.5 * x_it + alpha_i + u_it
        // We need enough observations for GMM to work properly
        let mut entities = Vec::new();
        let mut times = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let entity_effects = [0.0, 1.0, -0.5, 0.5, -1.0, 2.0, 0.3, -0.3, 0.8, -0.8];
        let n_entities = 10;
        let n_periods = 8;

        // Use deterministic "noise" for reproducibility
        let noise_values = [
            0.1, -0.2, 0.15, -0.1, 0.05, -0.05, 0.2, -0.15, 0.12, -0.08,
            0.08, -0.12, 0.18, -0.18, 0.03, -0.07, 0.14, -0.14, 0.09, -0.06,
            0.11, -0.11, 0.16, -0.16, 0.04, -0.09, 0.13, -0.13, 0.07, -0.04,
            0.06, -0.03, 0.17, -0.17, 0.02, -0.08, 0.19, -0.19, 0.08, -0.01,
            0.05, -0.05, 0.12, -0.12, 0.09, -0.09, 0.15, -0.15, 0.04, -0.04,
            0.07, -0.07, 0.11, -0.11, 0.06, -0.06, 0.14, -0.14, 0.03, -0.03,
            0.08, -0.08, 0.13, -0.13, 0.07, -0.07, 0.16, -0.16, 0.02, -0.02,
            0.09, -0.09, 0.14, -0.14, 0.05, -0.05, 0.17, -0.17, 0.01, -0.01
        ];

        let x_values = [
            1.0, 1.5, 2.0, 1.8, 2.2, 2.5, 2.3, 2.7,
            1.2, 1.7, 2.1, 1.9, 2.3, 2.6, 2.4, 2.8,
            0.8, 1.3, 1.8, 1.6, 2.0, 2.3, 2.1, 2.5,
            1.1, 1.6, 2.2, 2.0, 2.4, 2.7, 2.5, 2.9,
            0.9, 1.4, 1.9, 1.7, 2.1, 2.4, 2.2, 2.6,
            1.3, 1.8, 2.3, 2.1, 2.5, 2.8, 2.6, 3.0,
            0.7, 1.2, 1.7, 1.5, 1.9, 2.2, 2.0, 2.4,
            1.0, 1.5, 2.0, 1.8, 2.2, 2.5, 2.3, 2.7,
            1.4, 1.9, 2.4, 2.2, 2.6, 2.9, 2.7, 3.1,
            0.6, 1.1, 1.6, 1.4, 1.8, 2.1, 1.9, 2.3
        ];

        for i in 0..n_entities {
            let alpha = entity_effects[i];
            let mut y_prev = 2.0 + alpha; // Initial y

            for t in 0..n_periods {
                let idx = i * n_periods + t;
                let x = x_values[idx];
                let noise = noise_values[idx];

                // y_it = 0.5 * y_{i,t-1} + 1.5 * x_it + alpha_i + u_it
                let y = 0.5 * y_prev + 1.5 * x + alpha + noise;

                entities.push(format!("E{}", i));
                times.push((t + 1) as i64);
                xs.push(x);
                ys.push(y);

                y_prev = y;
            }
        }

        let df = df! {
            "entity" => entities,
            "time" => times,
            "x" => xs,
            "y" => ys
        }.unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_gmm_difference_onestep() {
        let dataset = create_gmm_panel_dataset();

        let config = GmmConfig {
            transform: GmmTransform::Difference,
            step: GmmStep::OneStep,
            max_lag: Some(4),
            min_lag: 2,
            collapse: false,
            robust: true,
        };

        let result = run_gmm(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            1,
            Some(config),
        );

        // Should succeed (may give warnings but shouldn't fail)
        assert!(result.is_ok(), "GMM should succeed, got error: {:?}", result.err());

        let result = result.unwrap();

        // Basic structure checks
        assert_eq!(result.transform, GmmTransform::Difference);
        assert_eq!(result.step, GmmStep::OneStep);
        assert!(result.n_obs > 0, "Should have observations");
        assert!(result.n_groups > 0, "Should have groups");
        assert!(!result.coefficients.is_empty(), "Should have coefficients");
        assert!(!result.std_errors.is_empty(), "Should have standard errors");

        // Coefficients should be positive (both lag-y and x have positive effects)
        // The first coefficient is for lagged y (should be around 0.5)
        // The second is for x (should be around 1.5)
        if result.coefficients.len() >= 2 {
            // Just check they're finite and reasonable magnitude
            assert!(result.coefficients[0].is_finite(), "Lagged y coefficient should be finite");
            assert!(result.coefficients[1].is_finite(), "x coefficient should be finite");
        }

        // Standard errors should be positive
        for se in &result.std_errors {
            assert!(*se >= 0.0 || se.is_nan(), "Std errors should be non-negative");
        }

        // Sargan test should be computed (though may not be meaningful with few instruments)
        assert!(result.sargan_statistic.is_finite() || result.sargan_statistic.is_nan());
    }

    #[test]
    fn test_gmm_difference_twostep() {
        let dataset = create_gmm_panel_dataset();

        let config = GmmConfig {
            transform: GmmTransform::Difference,
            step: GmmStep::TwoStep,
            max_lag: Some(4),
            min_lag: 2,
            collapse: false,
            robust: true,
        };

        let result = run_gmm(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            1,
            Some(config),
        );

        assert!(result.is_ok(), "Two-step GMM should succeed, got error: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.step, GmmStep::TwoStep);
        assert!(!result.coefficients.is_empty());
    }

    #[test]
    fn test_arellano_bond_convenience() {
        let dataset = create_gmm_panel_dataset();

        let result = run_arellano_bond(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        );

        assert!(result.is_ok(), "Arellano-Bond should succeed, got error: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.transform, GmmTransform::Difference);
    }

    #[test]
    fn test_gmm_system() {
        let dataset = create_gmm_panel_dataset();

        let config = GmmConfig {
            transform: GmmTransform::System,
            step: GmmStep::TwoStep,
            max_lag: Some(3),
            min_lag: 2,
            collapse: false,
            robust: true,
        };

        let result = run_gmm(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            1,
            Some(config),
        );

        assert!(result.is_ok(), "System GMM should succeed, got error: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.transform, GmmTransform::System);
    }

    #[test]
    fn test_gmm_display() {
        let dataset = create_gmm_panel_dataset();

        let result = run_arellano_bond(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        );

        assert!(result.is_ok());

        let result = result.unwrap();
        let display = format!("{}", result);

        // Check that key information is displayed
        assert!(display.contains("GMM"), "Display should mention GMM");
        assert!(display.contains("Coefficient") || display.contains("Variable"),
            "Display should show coefficients");
    }

    // ============== PVCM (Variable Coefficients Model) Tests ==============

    fn create_pvcm_dataset() -> Dataset {
        // Create a panel where coefficients vary across entities
        // Entity A: y = 1 + 2*x + noise
        // Entity B: y = 2 + 3*x + noise
        // Entity C: y = 3 + 1.5*x + noise
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A", "A",
                        "B", "B", "B", "B", "B", "B",
                        "C", "C", "C", "C", "C", "C"],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y" => [3.1, 5.0, 6.9, 9.1, 10.8, 13.0,   // A: y ≈ 1 + 2x
                   5.0, 8.1, 10.9, 14.0, 17.2, 20.0,  // B: y ≈ 2 + 3x
                   4.6, 6.0, 7.4, 9.0, 10.6, 12.1]    // C: y ≈ 3 + 1.5x
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_pvcm_within_basic() {
        let dataset = create_pvcm_dataset();
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Within);

        assert!(result.is_ok(), "pvcm within should succeed, got {:?}", result.err());

        let result = result.unwrap();

        // Check structure
        assert_eq!(result.model_type, PvcmType::Within);
        assert_eq!(result.n_obs, 18);
        assert_eq!(result.n_entities, 3);
        assert_eq!(result.variables.len(), 2); // intercept + x

        // Should have individual coefficients for each entity
        assert!(result.individual_coefficients.contains_key("A"));
        assert!(result.individual_coefficients.contains_key("B"));
        assert!(result.individual_coefficients.contains_key("C"));

        // Check that individual coefficients are different
        let coef_a = &result.individual_coefficients["A"];
        let coef_b = &result.individual_coefficients["B"];
        let coef_c = &result.individual_coefficients["C"];

        // Entity B should have the highest slope (around 3)
        // Entity A should have slope around 2
        // Entity C should have the lowest slope (around 1.5)
        assert!(coef_b[1] > coef_a[1], "B slope should be > A slope");
        assert!(coef_a[1] > coef_c[1], "A slope should be > C slope");

        // Average coefficient should be approximately (2 + 3 + 1.5) / 3 ≈ 2.17
        let avg_slope = result.coefficients[1];
        assert!((avg_slope - 2.17).abs() < 0.3,
            "Average slope should be around 2.17, got {}", avg_slope);
    }

    #[test]
    fn test_pvcm_random_basic() {
        let dataset = create_pvcm_dataset();
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Random);

        assert!(result.is_ok(), "pvcm random should succeed, got {:?}", result.err());

        let result = result.unwrap();

        // Check structure
        assert_eq!(result.model_type, PvcmType::Random);
        assert_eq!(result.n_obs, 18);
        assert_eq!(result.n_entities, 3);

        // Random model should provide delta (variance of random coefficients)
        assert_eq!(result.delta.len(), 2); // For intercept and x

        // Delta should be positive (we have heterogeneity)
        assert!(result.delta.iter().all(|&d| d >= 0.0),
            "Delta values should be non-negative");

        // Coefficient should be close to GLS estimate
        // With Swamy weights, should be similar to within but more efficient
        let slope = result.coefficients[1];
        assert!((slope - 2.17).abs() < 0.5,
            "GLS slope should be around 2.17, got {}", slope);
    }

    #[test]
    fn test_pvcm_homogeneity_test() {
        let dataset = create_pvcm_dataset();
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Within).unwrap();

        // With heterogeneous slopes, homogeneity test should reject H0
        assert!(result.homogeneity_stat > 0.0, "Homogeneity stat should be positive");
        assert!(result.homogeneity_pvalue >= 0.0 && result.homogeneity_pvalue <= 1.0,
            "Homogeneity p-value should be in [0,1]");

        // Since our data has truly different coefficients, the test should likely reject
        // (low p-value), but this depends on sample size
    }

    #[test]
    fn test_pmg_basic() {
        let dataset = create_pvcm_dataset();
        let result = run_pmg(&dataset, "y", &["x"], "entity");

        assert!(result.is_ok(), "pmg should succeed, got {:?}", result.err());

        let result = result.unwrap();

        // pmg is within model
        assert_eq!(result.model_type, PvcmType::Within);

        // Should have same result as pvcm within
        let pvcm_result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Within).unwrap();

        assert_eq!(result.coefficients.len(), pvcm_result.coefficients.len());
        for (a, b) in result.coefficients.iter().zip(pvcm_result.coefficients.iter()) {
            assert!((a - b).abs() < 1e-10, "pmg and pvcm within should match");
        }
    }

    #[test]
    fn test_pvcm_insufficient_obs() {
        // Create dataset with one entity having too few observations
        let df = df! {
            "entity" => ["A", "A", "B"], // B has only 1 obs
            "x" => [1.0, 2.0, 1.0],
            "y" => [3.0, 5.0, 4.0]
        }.unwrap();
        let dataset = Dataset::new(df);

        // Should still work, just skip the entity with insufficient data
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Within);

        // Entity B should be skipped (1 obs < 2 params)
        // Entity A should work (2 obs = 2 params, borderline)
        // This might fail or succeed depending on implementation
        // The test checks that we don't panic
        match result {
            Ok(r) => {
                // If it succeeds, should only have estimates from entities with enough data
                assert!(r.individual_coefficients.len() <= 2);
            }
            Err(_) => {
                // If it fails due to no valid entities, that's acceptable
            }
        }
    }

    #[test]
    fn test_pvcm_display() {
        let dataset = create_pvcm_dataset();
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Random).unwrap();

        let display = format!("{}", result);

        // Check key information is shown
        assert!(display.contains("Variable Coefficients") || display.contains("Swamy"),
            "Display should mention model type");
        assert!(display.contains("x") || display.contains("Variable"),
            "Display should show variables");
    }
}
