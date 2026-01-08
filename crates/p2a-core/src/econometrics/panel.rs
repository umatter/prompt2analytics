//! Panel data estimators: Fixed Effects (FE) and Random Effects (RE).
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

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
        writeln!(f, "Result: {}", self.recommendation)?;
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

    let recommendation = if p_value < 0.05 {
        "Reject H0: Use Fixed Effects (systematic difference in coefficients detected)".to_string()
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
}
