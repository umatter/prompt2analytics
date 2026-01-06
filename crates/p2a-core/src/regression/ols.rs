//! Ordinary Least Squares (OLS) regression implementation using greeners.

use greeners::{CovarianceType, Formula, OLS};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::Dataset;
use crate::econometrics::polars_to_greeners;

/// Errors that can occur during OLS regression.
#[derive(Error, Debug)]
pub enum OlsError {
    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Column is not numeric: {0}")]
    NotNumeric(String),

    #[error("Insufficient observations: need at least {0}, got {1}")]
    InsufficientObs(usize, usize),

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("Regression error: {0}")]
    RegressionError(String),

    #[error("No features specified")]
    NoFeatures,
}

/// Result of an OLS regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsResult {
    /// Dependent variable name
    pub dependent_var: String,
    /// Independent variable names
    pub independent_vars: Vec<String>,
    /// Number of observations
    pub n_obs: usize,
    /// Intercept coefficient
    pub intercept: f64,
    /// Coefficients for each independent variable
    pub coefficients: Vec<OlsCoefficient>,
    /// R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Residual standard error
    pub residual_std_error: f64,
    /// F-statistic
    pub f_statistic: f64,
}

/// A single coefficient with its statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsCoefficient {
    pub name: String,
    pub estimate: f64,
    pub std_error: f64,
    pub t_value: f64,
    pub p_value: f64,
}

impl std::fmt::Display for OlsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OLS Regression Results")?;
        writeln!(f, "======================")?;
        writeln!(f, "Dependent Variable: {}", self.dependent_var)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f, "F-statistic: {:.4}", self.f_statistic)?;
        writeln!(f, "Residual Std. Error: {:.4}", self.residual_std_error)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(f, "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Error", "t-value", "Pr(>|t|)")?;
        writeln!(f, "{:-<61}", "")?;
        writeln!(f, "{:>15} {:>12.4} {:>12} {:>10} {:>10}",
            "(Intercept)", self.intercept, "-", "-", "-")?;
        for coef in &self.coefficients {
            let sig = significance_code(coef.p_value);
            writeln!(f, "{:>15} {:>12.4} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&coef.name, 15), coef.estimate, coef.std_error,
                coef.t_value, coef.p_value, sig)?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")?;
        Ok(())
    }
}

/// Get significance code for a p-value.
fn significance_code(p: f64) -> &'static str {
    if p < 0.001 { "***" }
    else if p < 0.01 { "**" }
    else if p < 0.05 { "*" }
    else if p < 0.1 { "." }
    else { "" }
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

/// Check if a DataType is numeric.
fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Run OLS regression on a dataset.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
///
/// # Returns
/// An `OlsResult` containing the regression results.
pub fn run_ols(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> Result<OlsResult, OlsError> {
    let df = dataset.df();

    if x_cols.is_empty() {
        return Err(OlsError::NoFeatures);
    }

    // Verify all columns exist and are numeric
    verify_column(df, y_col)?;
    for col in x_cols {
        verify_column(df, col)?;
    }

    // Build R-style formula: y ~ x1 + x2 + ...
    let formula_str = format!("{} ~ {}", y_col, x_cols.join(" + "));

    // Parse the formula
    let formula = Formula::parse(&formula_str)
        .map_err(|e| OlsError::RegressionError(format!("Failed to parse formula '{}': {}", formula_str, e)))?;

    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(df)
        .map_err(|e: anyhow::Error| OlsError::RegressionError(e.to_string()))?;

    // Fit the model with robust standard errors (HC1)
    let result = OLS::from_formula(&formula, &gdf, CovarianceType::HC1)
        .map_err(|e| OlsError::RegressionError(format!("OLS fitting failed: {}", e)))?;

    // Extract results
    let params = result.params.to_vec();
    let std_errors = result.std_errors.to_vec();
    let t_values = result.t_values.to_vec();
    let p_values = result.p_values.to_vec();

    // Extract variable names from result (includes intercept as first)
    let var_names = result.variable_names.unwrap_or_else(|| {
        let mut names = vec!["const".to_string()];
        names.extend(x_cols.iter().map(|s| s.to_string()));
        names
    });

    // Separate intercept from other coefficients
    let intercept = if var_names.first().map(|s| s == "const").unwrap_or(false) {
        params.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    // Build coefficient list (skip intercept)
    let coefficients: Vec<OlsCoefficient> = var_names.iter()
        .enumerate()
        .filter(|(_, name)| *name != "const")
        .map(|(i, name)| {
            OlsCoefficient {
                name: name.clone(),
                estimate: params.get(i).copied().unwrap_or(0.0),
                std_error: std_errors.get(i).copied().unwrap_or(f64::NAN),
                t_value: t_values.get(i).copied().unwrap_or(f64::NAN),
                p_value: p_values.get(i).copied().unwrap_or(f64::NAN),
            }
        })
        .collect();

    Ok(OlsResult {
        dependent_var: y_col.to_string(),
        independent_vars: x_cols.iter().map(|s| s.to_string()).collect(),
        n_obs: result.n_obs,
        intercept,
        coefficients,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.sigma,
        f_statistic: result.f_statistic,
    })
}

/// Verify that a column exists and is numeric.
fn verify_column(df: &DataFrame, col: &str) -> Result<(), OlsError> {
    let column = df.column(col).map_err(|_| OlsError::ColumnNotFound(col.to_string()))?;
    if !is_numeric_dtype(column.dtype()) {
        return Err(OlsError::NotNumeric(col.to_string()));
    }
    Ok(())
}

/// Result of OLS with clustered standard errors.
#[derive(Debug, Clone)]
pub struct OlsClusteredResult {
    /// Base OLS result
    pub ols: OlsResult,
    /// Type of standard errors used
    pub se_type: String,
    /// Number of clusters (dimension 1)
    pub n_clusters_1: Option<usize>,
    /// Number of clusters (dimension 2, for two-way)
    pub n_clusters_2: Option<usize>,
}

impl std::fmt::Display for OlsClusteredResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OLS Regression Results")?;
        writeln!(f, "======================")?;
        writeln!(f, "Dependent Variable: {}", self.ols.dependent_var)?;
        writeln!(f, "Observations: {}", self.ols.n_obs)?;
        writeln!(f, "Standard Errors: {}", self.se_type)?;
        if let Some(n1) = self.n_clusters_1 {
            write!(f, "Clusters: {}", n1)?;
            if let Some(n2) = self.n_clusters_2 {
                writeln!(f, " x {}", n2)?;
            } else {
                writeln!(f)?;
            }
        }
        writeln!(f, "R-squared: {:.4}", self.ols.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.ols.adj_r_squared)?;
        writeln!(f, "F-statistic: {:.4}", self.ols.f_statistic)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(f, "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Error", "t-value", "Pr(>|t|)")?;
        writeln!(f, "{:-<61}", "")?;
        writeln!(f, "{:>15} {:>12.4} {:>12} {:>10} {:>10}",
            "(Intercept)", self.ols.intercept, "-", "-", "-")?;
        for coef in &self.ols.coefficients {
            let sig = significance_code(coef.p_value);
            writeln!(f, "{:>15} {:>12.4} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&coef.name, 15), coef.estimate, coef.std_error,
                coef.t_value, coef.p_value, sig)?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")?;
        Ok(())
    }
}

/// Run OLS regression with clustered standard errors.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
/// * `cluster1` - Column name for first clustering dimension (e.g., "firm_id")
/// * `cluster2` - Optional column name for second clustering dimension (e.g., "year")
///
/// # Returns
/// An `OlsClusteredResult` containing the regression results with clustered SEs.
pub fn run_ols_clustered(
    dataset: &Dataset,
    formula: &str,
    cluster1: &str,
    cluster2: Option<&str>,
) -> Result<OlsClusteredResult, OlsError> {
    let df = dataset.df();

    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| OlsError::RegressionError(format!("Failed to parse formula '{}': {}", formula, e)))?;

    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(df)
        .map_err(|e: anyhow::Error| OlsError::RegressionError(e.to_string()))?;

    // Extract cluster IDs for dimension 1
    let cluster_ids_1 = extract_cluster_ids(df, cluster1)?;
    let n_clusters_1 = cluster_ids_1.iter().max().map(|m| m + 1).unwrap_or(0);

    // Determine covariance type
    let (cov_type, se_type, n_clusters_2) = if let Some(c2) = cluster2 {
        // Two-way clustering
        let cluster_ids_2 = extract_cluster_ids(df, c2)?;
        let n_c2 = cluster_ids_2.iter().max().map(|m| m + 1).unwrap_or(0);
        (
            CovarianceType::ClusteredTwoWay(cluster_ids_1, cluster_ids_2),
            format!("Two-way clustered ({}, {})", cluster1, c2),
            Some(n_c2),
        )
    } else {
        // One-way clustering
        (
            CovarianceType::Clustered(cluster_ids_1),
            format!("Clustered ({})", cluster1),
            None,
        )
    };

    // Fit the model
    let result = OLS::from_formula(&parsed_formula, &gdf, cov_type)
        .map_err(|e| OlsError::RegressionError(format!("OLS fitting failed: {}", e)))?;

    // Extract results
    let params = result.params.to_vec();
    let std_errors = result.std_errors.to_vec();
    let t_values = result.t_values.to_vec();
    let p_values = result.p_values.to_vec();

    // Extract variable names
    let var_names = result.variable_names.unwrap_or_else(|| {
        let mut names = vec![];
        if parsed_formula.intercept {
            names.push("const".to_string());
        }
        names.extend(parsed_formula.independents.iter().cloned());
        names
    });

    // Separate intercept from coefficients
    let intercept = if var_names.first().map(|s| s == "const").unwrap_or(false) {
        params.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    // Build coefficient list (skip intercept)
    let coefficients: Vec<OlsCoefficient> = var_names.iter()
        .enumerate()
        .filter(|(_, name)| *name != "const")
        .map(|(i, name)| {
            OlsCoefficient {
                name: name.clone(),
                estimate: params.get(i).copied().unwrap_or(0.0),
                std_error: std_errors.get(i).copied().unwrap_or(f64::NAN),
                t_value: t_values.get(i).copied().unwrap_or(f64::NAN),
                p_value: p_values.get(i).copied().unwrap_or(f64::NAN),
            }
        })
        .collect();

    // Extract dependent variable from formula
    let dep_var = formula.split('~').next().unwrap_or("y").trim().to_string();

    Ok(OlsClusteredResult {
        ols: OlsResult {
            dependent_var: dep_var,
            independent_vars: parsed_formula.independents.clone(),
            n_obs: result.n_obs,
            intercept,
            coefficients,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            residual_std_error: result.sigma,
            f_statistic: result.f_statistic,
        },
        se_type,
        n_clusters_1: Some(n_clusters_1),
        n_clusters_2,
    })
}

/// Extract cluster IDs from a column and return as Vec<usize>.
fn extract_cluster_ids(df: &DataFrame, col: &str) -> Result<Vec<usize>, OlsError> {
    use std::collections::HashMap;

    let column = df.column(col)
        .map_err(|_| OlsError::ColumnNotFound(col.to_string()))?;

    let mut id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0usize;

    let ids: Vec<usize> = if let Ok(int_col) = column.i64() {
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
    } else if let Ok(str_col) = column.str() {
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
        let casted = column.cast(&DataType::String)
            .map_err(|e| OlsError::RegressionError(format!("Cannot convert cluster column: {}", e)))?;
        let str_col = casted.str()
            .map_err(|e| OlsError::RegressionError(format!("Cannot read cluster column: {}", e)))?;
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

    Ok(ids)
}
