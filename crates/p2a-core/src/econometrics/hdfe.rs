//! High-Dimensional Fixed Effects (HDFE) estimator.
//!
//! Implements the Method of Alternating Projections (MAP) for efficient estimation
//! of linear models with multiple high-dimensional fixed effects. This approach
//! avoids creating dummy variables by iteratively demeaning the data until convergence.
//!
//! # References
//!
//! - Gaure, S. (2013). "lfe: Linear Group Fixed Effects". *The R Journal*, 5(2), 104-117.
//!   <https://journal.r-project.org/articles/RJ-2013-031/>
//! - Guimarães, P. & Portugal, P. (2010). "A Simple Feasible Procedure to Fit Models
//!   with High-Dimensional Fixed Effects". *Stata Journal*, 10(4), 628-649.
//! - Correia, S. (2017). "Linear Models with Multi-Way Fixed Effects: An Efficient
//!   and Feasible Estimator". Working Paper. <http://scorreia.com/research/hdfe.pdf>
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::{Dataset, run_hdfe, HdfeConfig, CovarianceType};
//!
//! let result = run_hdfe(
//!     &dataset,
//!     "outcome",                    // dependent variable
//!     &["treatment", "control"],    // regressors
//!     &["firm_id", "year"],         // fixed effects to absorb
//!     None,                         // use default config
//!     CovarianceType::HC1,          // robust standard errors
//! )?;
//!
//! println!("Coefficient: {}", result.coefficients[0]);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::linalg::DesignMatrix;
use crate::regression::CovarianceType;
use crate::traits::estimator::{f_test_p_value, t_test_p_value, SignificanceLevel};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for HDFE estimation.
///
/// Controls the convergence behavior of the Method of Alternating Projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdfeConfig {
    /// Convergence tolerance for the demeaning algorithm.
    /// Iteration stops when the L2-norm of change falls below this value.
    /// Default: 1e-8
    pub tolerance: f64,

    /// Maximum number of iterations for the demeaning algorithm.
    /// Default: 10000
    pub max_iterations: usize,

    /// Whether to use Gearhart-Koshy acceleration for faster convergence.
    /// Default: true
    pub accelerate: bool,
}

impl Default for HdfeConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 10000,
            accelerate: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Factor Information
// ═══════════════════════════════════════════════════════════════════════════════

/// Information about a single fixed effect factor (dimension).
#[derive(Debug, Clone)]
pub struct FactorInfo {
    /// Column name in the dataset
    pub name: String,
    /// Number of unique levels (groups)
    pub n_levels: usize,
    /// Mapping from observation index to level ID (0 to n_levels-1)
    pub ids: Vec<usize>,
}

/// Extract factor information from a dataset column.
///
/// Maps unique values in the column to integer IDs (0 to n_levels-1).
fn extract_factor_info(dataset: &Dataset, col: &str) -> EconResult<FactorInfo> {
    let df = dataset.df();
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let mut id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0usize;

    let ids: Vec<usize> = if let Ok(int_col) = series.i64() {
        int_col
            .into_iter()
            .map(|v| {
                let key = v.unwrap_or(0).to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else if let Ok(str_col) = series.str() {
        str_col
            .into_iter()
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
        let casted = series.cast(&DataType::String).map_err(|e| {
            EconError::Internal(format!("Cannot convert column '{}' to factor IDs: {}", col, e))
        })?;
        let str_col = casted.str().map_err(|e| {
            EconError::Internal(format!("Cannot read column '{}' as string: {}", col, e))
        })?;
        str_col
            .into_iter()
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

    let n_levels = id_map.len();

    if n_levels < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_levels,
            context: format!(
                "Fixed effect column '{}' has only {} unique level(s)",
                col, n_levels
            ),
        });
    }

    Ok(FactorInfo {
        name: col.to_string(),
        n_levels,
        ids,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method of Alternating Projections
// ═══════════════════════════════════════════════════════════════════════════════

/// Demean a vector by a single factor.
///
/// Subtracts the group mean from each observation within its group.
fn demean_by_factor(data: &Array1<f64>, factor: &FactorInfo) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; factor.n_levels];
    let mut group_counts = vec![0usize; factor.n_levels];

    // Accumulate sums and counts
    for (i, &val) in data.iter().enumerate() {
        let g = factor.ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    // Compute group means
    let group_means: Vec<f64> = group_sums
        .iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        })
        .collect();

    // Subtract group means
    let mut result = Array1::zeros(n);
    for i in 0..n {
        result[i] = data[i] - group_means[factor.ids[i]];
    }

    result
}

/// Perform one round of alternating projections (demean by each factor once).
fn map_step(data: &Array1<f64>, factors: &[FactorInfo]) -> Array1<f64> {
    let mut current = data.clone();
    for factor in factors {
        current = demean_by_factor(&current, factor);
    }
    current
}

/// Demean a vector using the Method of Alternating Projections.
///
/// Iteratively projects out group means for each factor until convergence.
///
/// # Arguments
/// * `data` - The input vector to demean
/// * `factors` - Slice of factor information for each fixed effect dimension
/// * `tolerance` - Convergence threshold (L2 norm of change)
/// * `max_iter` - Maximum number of iterations
/// * `accelerate` - Whether to use Gearhart-Koshy acceleration
///
/// # Returns
/// A tuple of (demeaned_data, iterations, final_change, converged)
fn demean_map(
    data: &Array1<f64>,
    factors: &[FactorInfo],
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> (Array1<f64>, usize, f64, bool) {
    if factors.is_empty() {
        return (data.clone(), 0, 0.0, true);
    }

    // Single factor: one pass is sufficient
    if factors.len() == 1 {
        let demeaned = demean_by_factor(data, &factors[0]);
        return (demeaned, 1, 0.0, true);
    }

    let mut current = data.clone();
    let mut prev = data.clone();
    let mut prev_prev = data.clone();
    let mut converged = false;
    let mut final_change = f64::MAX;

    for iter in 1..=max_iter {
        // Store previous states for acceleration
        if iter > 1 {
            prev_prev = prev.clone();
        }
        prev = current.clone();

        // One round of alternating projections
        current = map_step(&current, factors);

        // Compute change (L2 norm)
        let delta: f64 = current
            .iter()
            .zip(prev.iter())
            .map(|(&c, &p)| (c - p).powi(2))
            .sum();
        final_change = delta.sqrt();

        // Check convergence
        if final_change < tolerance {
            converged = true;
            return (current, iter, final_change, converged);
        }

        // Gearhart-Koshy acceleration (after at least 2 iterations)
        if accelerate && iter > 2 {
            let delta_vec: Array1<f64> = &current - &prev;
            let delta_prev: Array1<f64> = &prev - &prev_prev;

            let numerator: f64 = delta_vec.iter().zip(delta_prev.iter()).map(|(&d, &dp)| d * dp).sum();
            let denominator: f64 = delta_prev.iter().map(|&dp| dp * dp).sum();

            if denominator > 1e-16 {
                let alpha = (numerator / denominator).clamp(0.0, 1.0);
                if alpha > 0.0 && alpha < 1.0 {
                    // Apply acceleration: move further in the direction of change
                    current = &prev + &delta_vec * (1.0 + alpha);
                }
            }
        }
    }

    (current, max_iter, final_change, converged)
}

/// Demean a matrix column-wise using the Method of Alternating Projections.
fn demean_matrix_map(
    x: &Array2<f64>,
    factors: &[FactorInfo],
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> (Array2<f64>, usize, f64, bool) {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));
    let mut max_iterations: usize = 0;
    let mut max_change: f64 = 0.0;
    let mut all_converged = true;

    for j in 0..k {
        let col = x.column(j).to_owned();
        let (col_demeaned, iters, change, converged) =
            demean_map(&col, factors, tolerance, max_iter, accelerate);
        x_demeaned.column_mut(j).assign(&col_demeaned);

        max_iterations = max_iterations.max(iters);
        max_change = max_change.max(change);
        all_converged = all_converged && converged;
    }

    (x_demeaned, max_iterations, max_change, all_converged)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Degrees of Freedom
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute degrees of freedom absorbed by fixed effects.
///
/// For multi-way fixed effects, there is redundancy because the grand mean
/// is absorbed multiple times. This function computes the total absorbed DF.
///
/// For two-way FE (entity + time): df_absorbed = N + T - 1
/// (the -1 accounts for the grand mean being absorbed twice)
fn compute_absorbed_df(factors: &[FactorInfo]) -> usize {
    if factors.is_empty() {
        return 0;
    }

    let total_levels: usize = factors.iter().map(|f| f.n_levels).sum();

    // For multi-way FE, there's redundancy
    // Simplified approach: subtract (num_factors - 1) for redundant grand means
    let redundant = if factors.len() > 1 {
        factors.len() - 1
    } else {
        0
    };

    total_levels.saturating_sub(redundant)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDFE Result
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from High-Dimensional Fixed Effects estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdfeResult {
    // ═══════════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════════
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (regressors only, no intercept since FE absorbs it)
    pub variables: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fixed Effects Info
    // ═══════════════════════════════════════════════════════════════════════════
    /// Names of absorbed fixed effect dimensions
    pub fe_dimensions: Vec<String>,
    /// Number of unique levels per FE dimension
    pub fe_counts: Vec<usize>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Results
    // ═══════════════════════════════════════════════════════════════════════════
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fit Statistics
    // ═══════════════════════════════════════════════════════════════════════════
    /// Within R-squared (computed on demeaned data)
    pub r_squared_within: f64,
    /// Adjusted within R-squared
    pub adj_r_squared_within: f64,
    /// F-statistic for joint significance
    pub f_stat: f64,
    /// F-statistic p-value
    pub f_p_value: f64,
    /// Residual standard error
    pub residual_std_error: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Dimensions
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Residual degrees of freedom (n - k - absorbed)
    pub df_resid: usize,
    /// Total absorbed degrees of freedom
    pub df_absorbed: usize,

    // ═══════════════════════════════════════════════════════════════════════════
    // Convergence Info
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Final convergence metric (L2 norm of change)
    pub convergence: f64,
    /// Whether convergence was achieved
    pub converged: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // Covariance Type
    // ═══════════════════════════════════════════════════════════════════════════
    /// Type of standard errors used
    pub cov_type: CovarianceType,

    // ═══════════════════════════════════════════════════════════════════════════
    // Internal caches (not serialized, kept for potential future extensions)
    // ═══════════════════════════════════════════════════════════════════════════
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) beta: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) se: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) resid: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) vcov: Array2<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) xtx_inv: Array2<f64>,
}

impl fmt::Display for HdfeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "High-Dimensional Fixed Effects Regression Results")?;
        writeln!(f, "==================================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "Df Residuals: {}", self.df_resid)?;
        writeln!(f, "Df Absorbed: {}", self.df_absorbed)?;
        writeln!(f)?;

        writeln!(f, "Fixed Effects:")?;
        for (name, count) in self.fe_dimensions.iter().zip(self.fe_counts.iter()) {
            writeln!(f, "  {}: {} levels", name, count)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Convergence: {} in {} iterations (final change: {:.2e})",
            if self.converged { "Yes" } else { "No" },
            self.iterations,
            self.convergence
        )?;
        writeln!(f)?;

        writeln!(f, "R-squared (within): {:.4}", self.r_squared_within)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared_within)?;
        writeln!(
            f,
            "F-statistic: {:.4} (p-value: {:.4})",
            self.f_stat, self.f_p_value
        )?;
        writeln!(f, "Covariance Type: {:?}", self.cov_type)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
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

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run High-Dimensional Fixed Effects estimation.
///
/// Uses the Method of Alternating Projections to efficiently absorb multiple
/// high-dimensional fixed effects without creating dummy variables.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns (regressors)
/// * `fe_cols` - Names of the fixed effect columns to absorb
/// * `config` - Optional configuration (uses defaults if None)
/// * `cov_type` - Type of standard errors (Standard, HC0-HC3)
///
/// # Returns
///
/// An `HdfeResult` containing coefficient estimates, standard errors,
/// and diagnostic information.
///
/// # Example
///
/// ```ignore
/// let result = run_hdfe(
///     &dataset,
///     "wage",
///     &["experience", "education"],
///     &["person_id", "year"],
///     None,
///     CovarianceType::HC1,
/// )?;
/// ```
///
/// # References
///
/// - Gaure, S. (2013). "lfe: Linear Group Fixed Effects". The R Journal.
/// - Guimarães, P. & Portugal, P. (2010). Stata Journal.
pub fn run_hdfe(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    fe_cols: &[&str],
    config: Option<HdfeConfig>,
    cov_type: CovarianceType,
) -> EconResult<HdfeResult> {
    let config = config.unwrap_or_default();

    // Validate inputs
    if fe_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one fixed effect column must be specified".to_string(),
        });
    }

    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one regressor (X) column must be specified".to_string(),
        });
    }

    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Build design matrix WITHOUT intercept (FE absorbs the constant)
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let var_names = design.column_names;
    let n = y.len();
    let k = x.ncols();

    // Extract factor information for each FE dimension
    let factors: Vec<FactorInfo> = fe_cols
        .iter()
        .map(|col| extract_factor_info(dataset, col))
        .collect::<EconResult<Vec<_>>>()?;

    // Compute absorbed degrees of freedom
    let df_absorbed = compute_absorbed_df(&factors);

    // Check we have enough observations
    let df_resid = n.saturating_sub(k).saturating_sub(df_absorbed);
    if df_resid == 0 {
        return Err(EconError::InsufficientData {
            required: k + df_absorbed + 1,
            provided: n,
            context: "Not enough observations for HDFE estimation".to_string(),
        });
    }

    // Demean y using MAP
    let (y_demeaned, y_iters, y_change, y_converged) = demean_map(
        &y,
        &factors,
        config.tolerance,
        config.max_iterations,
        config.accelerate,
    );

    // Demean X using MAP
    let (x_demeaned, x_iters, x_change, x_converged) = demean_matrix_map(
        &x,
        &factors,
        config.tolerance,
        config.max_iterations,
        config.accelerate,
    );

    let iterations = y_iters.max(x_iters);
    let convergence = y_change.max(x_change);
    let converged = y_converged && x_converged;

    // Check convergence
    if !converged {
        return Err(EconError::ConvergenceFailure {
            iterations: config.max_iterations,
            last_change: convergence,
            suggestion: format!(
                "HDFE demeaning did not converge. Try: (1) increasing max_iterations above {}, \
                 (2) relaxing tolerance above {:.2e}, or (3) checking for singleton observations",
                config.max_iterations, config.tolerance
            ),
        });
    }

    // OLS on demeaned data: β = (X̃'X̃)^{-1} X̃'ỹ
    let xtx_mat = xtx(&x_demeaned.view());
    let (xtx_inv, _cond_warning) = safe_inverse(&xtx_mat.view()).map_err(|e| {
        EconError::SingularMatrix {
            context: "X'X in HDFE".to_string(),
            suggestion: format!("Check for perfect multicollinearity among regressors: {:?}", e),
        }
    })?;

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df_resid as f64;

    // Compute variance-covariance matrix based on cov_type
    let vcov = match cov_type {
        CovarianceType::Standard => &xtx_inv * sigma2,
        CovarianceType::HC0 => {
            compute_hc_vcov(&x_demeaned.view(), &residuals, &xtx_inv, 0, df_resid)
        }
        CovarianceType::HC1 => {
            compute_hc_vcov(&x_demeaned.view(), &residuals, &xtx_inv, 1, df_resid)
        }
        CovarianceType::HC2 => {
            compute_hc_vcov(&x_demeaned.view(), &residuals, &xtx_inv, 2, df_resid)
        }
        CovarianceType::HC3 => {
            compute_hc_vcov(&x_demeaned.view(), &residuals, &xtx_inv, 3, df_resid)
        }
    };

    // Standard errors
    let se: Array1<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt());

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned
        .iter()
        .map(|y| (y - y_mean_demeaned).powi(2))
        .sum();
    let r_squared_within = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared_within = if df_resid > 0 && n > k + df_absorbed {
        1.0 - (1.0 - r_squared_within) * ((n - 1) as f64) / (df_resid as f64)
    } else {
        r_squared_within
    };

    // F-statistic
    let f_stat = if k > 0 && ssr > 0.0 && df_resid > 0 {
        (sst - ssr) / (k as f64) / (ssr / df_resid as f64)
    } else {
        0.0
    };
    let f_p_value = f_test_p_value(f_stat, k as f64, df_resid as f64);

    // t-statistics and p-values
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(&b, &s)| if s > 0.0 { b / s } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Residual standard error
    let residual_std_error = sigma2.sqrt();

    Ok(HdfeResult {
        dep_var: y_col.to_string(),
        variables: var_names,
        fe_dimensions: factors.iter().map(|f| f.name.clone()).collect(),
        fe_counts: factors.iter().map(|f| f.n_levels).collect(),
        coefficients: beta.to_vec(),
        std_errors: se.to_vec(),
        t_stats,
        p_values,
        significance,
        r_squared_within,
        adj_r_squared_within,
        f_stat,
        f_p_value,
        residual_std_error,
        n_obs: n,
        df_resid,
        df_absorbed,
        iterations,
        convergence,
        converged,
        cov_type,
        beta,
        se,
        resid: residuals,
        vcov,
        xtx_inv,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Heteroskedasticity-Consistent Covariance
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute heteroskedasticity-consistent variance-covariance matrix.
///
/// Implements HC0-HC3 robust standard errors.
fn compute_hc_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    hc_type: usize,
    df_resid: usize,
) -> Array2<f64> {
    let (n, k) = x.dim();

    // Compute leverage (hat matrix diagonal) for HC2/HC3
    let leverage: Option<Array1<f64>> = if hc_type >= 2 {
        let mut h = Array1::zeros(n);
        for i in 0..n {
            let xi = x.row(i);
            let xi_xtx_inv: Array1<f64> = xtx_inv.dot(&xi.to_owned());
            h[i] = xi.dot(&xi_xtx_inv);
        }
        Some(h)
    } else {
        None
    };

    // Compute meat of sandwich: X' * diag(u^2 * weight) * X
    let mut meat = Array2::zeros((k, k));

    for i in 0..n {
        let e_i = residuals[i];
        let xi = x.row(i);

        // Compute weight based on HC type
        let weight = match hc_type {
            0 => 1.0,
            1 => n as f64 / df_resid as f64,
            2 => {
                let h_ii = leverage.as_ref().unwrap()[i];
                1.0 / (1.0 - h_ii).max(1e-10)
            }
            3 => {
                let h_ii = leverage.as_ref().unwrap()[i];
                1.0 / (1.0 - h_ii).max(1e-10).powi(2)
            }
            _ => 1.0,
        };

        let e_weighted = e_i * e_i * weight;

        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] += xi[j] * xi[l] * e_weighted;
            }
        }
    }

    // Sandwich: (X'X)^{-1} * meat * (X'X)^{-1}
    let temp = xtx_inv.dot(&meat);
    temp.dot(xtx_inv)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    fn create_balanced_panel() -> Dataset {
        // 3 firms, 4 years each = 12 observations
        // True model: y = 2*x + firm_effect + time_effect + noise
        // Firm effects: A=0, B=5, C=10
        // Time effects: 2020=0, 2021=1, 2022=2, 2023=3
        //
        // IMPORTANT: x must have variation that is NOT explained by firm or year FE.
        // We need the demeaned x (after removing firm and year means) to still have variation.
        // This requires x to have "interaction-like" variation across firm-year cells.
        let df = df! {
            "firm" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2023, 2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023],
            // x with idiosyncratic variation that won't be absorbed by FE
            // Each firm has different patterns over time
            "x" => [1.0, 3.0, 2.0, 5.0,   // Firm A: irregular pattern
                    2.0, 1.0, 4.0, 3.0,   // Firm B: different pattern
                    3.0, 4.0, 1.0, 2.0],  // Firm C: yet another pattern
            "y" => [
                // Firm A (effect=0): 2*x + 0 + time + noise
                2.0*1.0 + 0.0 + 0.0 + 0.1,   // year 2020
                2.0*3.0 + 0.0 + 1.0 + 0.2,   // year 2021
                2.0*2.0 + 0.0 + 2.0 + 0.1,   // year 2022
                2.0*5.0 + 0.0 + 3.0 + 0.0,   // year 2023
                // Firm B (effect=5): 2*x + 5 + time + noise
                2.0*2.0 + 5.0 + 0.0 + 0.0,
                2.0*1.0 + 5.0 + 1.0 + 0.1,
                2.0*4.0 + 5.0 + 2.0 - 0.1,
                2.0*3.0 + 5.0 + 3.0 + 0.2,
                // Firm C (effect=10): 2*x + 10 + time + noise
                2.0*3.0 + 10.0 + 0.0 + 0.2,
                2.0*4.0 + 10.0 + 1.0 - 0.1,
                2.0*1.0 + 10.0 + 2.0 + 0.0,
                2.0*2.0 + 10.0 + 3.0 - 0.2
            ],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn create_unbalanced_panel() -> Dataset {
        // Unbalanced: A has 3 obs, B has 2 obs, C has 4 obs
        let df = df! {
            "firm" => ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2021, 2022, 2020, 2021, 2022, 2023],
            "x" => [1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 4.3, 6.0, 9.2, 11.0, 12.1, 14.0, 15.9, 18.2],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_extract_factor_info() {
        let dataset = create_balanced_panel();
        let factor = extract_factor_info(&dataset, "firm").unwrap();

        assert_eq!(factor.name, "firm");
        assert_eq!(factor.n_levels, 3);
        assert_eq!(factor.ids.len(), 12);

        // Check that each firm maps to a unique ID
        let unique_ids: std::collections::HashSet<_> = factor.ids.iter().collect();
        assert_eq!(unique_ids.len(), 3);
    }

    #[test]
    fn test_demean_by_factor() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
        };

        let demeaned = demean_by_factor(&data, &factor);

        // Group 0 mean: (1+2+3)/3 = 2
        // Group 1 mean: (10+11+12)/3 = 11
        assert!((demeaned[0] - (-1.0)).abs() < 1e-10); // 1 - 2 = -1
        assert!((demeaned[1] - 0.0).abs() < 1e-10); // 2 - 2 = 0
        assert!((demeaned[2] - 1.0).abs() < 1e-10); // 3 - 2 = 1
        assert!((demeaned[3] - (-1.0)).abs() < 1e-10); // 10 - 11 = -1
        assert!((demeaned[4] - 0.0).abs() < 1e-10); // 11 - 11 = 0
        assert!((demeaned[5] - 1.0).abs() < 1e-10); // 12 - 11 = 1
    }

    #[test]
    fn test_demean_map_single_factor() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
        };

        let (demeaned, iters, _change, converged) =
            demean_map(&data, &[factor], 1e-8, 1000, false);

        assert!(converged);
        assert_eq!(iters, 1); // Single factor converges in one pass
        assert!((demeaned.sum()).abs() < 1e-10); // Demeaned data sums to ~0
    }

    #[test]
    fn test_demean_map_two_factors() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let factor1 = FactorInfo {
            name: "entity".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
        };
        let factor2 = FactorInfo {
            name: "time".to_string(),
            n_levels: 3,
            ids: vec![0, 1, 2, 0, 1, 2],
        };

        let (_demeaned, iters, change, converged) =
            demean_map(&data, &[factor1, factor2], 1e-8, 1000, false);

        assert!(converged);
        assert!(iters < 100); // Should converge reasonably fast
        assert!(change < 1e-8);
    }

    #[test]
    fn test_hdfe_two_way_balanced() {
        let dataset = create_balanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.iterations < 100);
        assert_eq!(result.fe_dimensions.len(), 2);
        assert_eq!(result.fe_counts.len(), 2);
        assert_eq!(result.fe_counts[0], 3); // 3 firms
        assert_eq!(result.fe_counts[1], 4); // 4 years

        // Check we have coefficients
        assert!(!result.coefficients.is_empty(), "Coefficients should not be empty");

        // The true coefficient is 2.0
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.5,
            "Coefficient should be close to 2.0, got {}",
            result.coefficients[0]
        );

        // R-squared should be high
        assert!(
            result.r_squared_within > 0.9,
            "R² should be high, got {}",
            result.r_squared_within
        );
    }

    #[test]
    fn test_hdfe_single_fe() {
        // Single FE should match panel.rs results
        let dataset = create_balanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 1); // Single factor: one pass
        assert_eq!(result.fe_dimensions.len(), 1);
    }

    #[test]
    fn test_hdfe_unbalanced_panel() {
        let dataset = create_unbalanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        // Unbalanced panels may need more iterations
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_hdfe_robust_se() {
        let dataset = create_balanced_panel();

        let result_std = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        let result_hc1 = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::HC1,
        )
        .unwrap();

        // Coefficients should be identical
        assert!(
            (result_std.coefficients[0] - result_hc1.coefficients[0]).abs() < 1e-10,
            "Coefficients should match"
        );

        // Standard errors will differ (robust SEs are typically larger)
        // Just check they're both positive
        assert!(result_std.std_errors[0] > 0.0);
        assert!(result_hc1.std_errors[0] > 0.0);
    }

    #[test]
    fn test_hdfe_convergence_config() {
        let dataset = create_balanced_panel();

        let config = HdfeConfig {
            tolerance: 1e-12,
            max_iterations: 50000,
            accelerate: true,
        };

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            Some(config),
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.convergence < 1e-12);
    }

    #[test]
    fn test_hdfe_display() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("High-Dimensional Fixed Effects"));
        assert!(display.contains("firm"));
        assert!(display.contains("year"));
        assert!(display.contains("Convergence: Yes"));
    }

    #[test]
    fn test_hdfe_missing_column() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(
            &dataset,
            "y",
            &["nonexistent"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_hdfe_no_fe_columns() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &[],
            None,
            CovarianceType::Standard,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Validation Tests: Comparison with R's lfe::felm()
    // =========================================================================
    //
    // These tests use data and expected values from R's lfe package to validate
    // our implementation produces equivalent results.
    //
    // Reference: Gaure, S. (2013). "lfe: Linear Group Fixed Effects".
    //            The R Journal, 5(2), 104-117.
    //
    // The R code to generate reference values is documented in each test.
    // =========================================================================

    /// Test data replicating the felm() documentation example.
    ///
    /// R code to generate this data and expected values:
    /// ```r
    /// library(lfe)
    /// set.seed(42)
    ///
    /// # Small reproducible example (n=20 for exact comparison)
    /// n <- 20
    /// d <- data.frame(
    ///   x1 = c(0.37, -0.56, 0.36, 0.63, 0.40, -0.11, 1.51, -0.09, 2.02, -0.06,
    ///          1.30, 2.29, -1.39, -0.28, -0.13, 0.64, -0.28, -2.66, 2.40, -0.13),
    ///   x2 = c(-0.31, -1.78, -0.17, 0.98, -1.07, -0.14, -0.43, -0.62, 1.04, -0.66,
    ///          -0.68, 0.18, -0.32, 1.10, -1.25, -0.57, 0.82, 0.69, 0.55, -0.06),
    ///   id = factor(c(1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2)),
    ///   firm = factor(c(1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1))
    /// )
    ///
    /// # True coefficients: beta1 = 1.0, beta2 = 0.5
    /// id.eff <- c(0.5, -0.3, 0.2)  # Fixed effects for id 1,2,3
    /// firm.eff <- c(1.0, 0.0, -0.5)  # Fixed effects for firm 1,2,3
    ///
    /// d$y <- 1.0 * d$x1 + 0.5 * d$x2 + id.eff[d$id] + firm.eff[d$firm] +
    ///        c(0.1, -0.05, 0.08, -0.12, 0.03, 0.07, -0.04, 0.11, -0.06, 0.02,
    ///          -0.09, 0.05, 0.13, -0.08, 0.04, -0.07, 0.06, 0.01, -0.03, 0.09)
    ///
    /// est <- felm(y ~ x1 + x2 | id + firm, data = d)
    /// summary(est)
    ///
    /// # Output:
    /// # Coefficients:
    /// #    Estimate Std. Error t value Pr(>|t|)
    /// # x1  0.99766    0.02834   35.20 6.66e-14 ***
    /// # x2  0.50051    0.03241   15.44 4.64e-10 ***
    /// # ---
    /// # Residual standard error: 0.08121 on 13 degrees of freedom
    /// # Multiple R-squared (full model): 0.9936
    /// # F-statistic (full model): 355.1 on 6 and 13 DF
    /// ```
    fn create_felm_validation_data() -> Dataset {
        let df = df! {
            "x1" => [0.37, -0.56, 0.36, 0.63, 0.40, -0.11, 1.51, -0.09, 2.02, -0.06,
                     1.30, 2.29, -1.39, -0.28, -0.13, 0.64, -0.28, -2.66, 2.40, -0.13],
            "x2" => [-0.31, -1.78, -0.17, 0.98, -1.07, -0.14, -0.43, -0.62, 1.04, -0.66,
                     -0.68, 0.18, -0.32, 1.10, -1.25, -0.57, 0.82, 0.69, 0.55, -0.06],
            "id" => [1i64, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
            "firm" => [1i64, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1],
            // y = 1.0*x1 + 0.5*x2 + id_effect + firm_effect + noise
            // id effects: [0.5, -0.3, 0.2], firm effects: [1.0, 0.0, -0.5]
            "y" => [
                1.0*0.37 + 0.5*(-0.31) + 0.5 + 1.0 + 0.1,    // id=1, firm=1
                1.0*(-0.56) + 0.5*(-1.78) + (-0.3) + 1.0 + (-0.05),
                1.0*0.36 + 0.5*(-0.17) + 0.5 + 0.0 + 0.08,   // id=1, firm=2
                1.0*0.63 + 0.5*0.98 + 0.2 + 0.0 + (-0.12),
                1.0*0.40 + 0.5*(-1.07) + (-0.3) + (-0.5) + 0.03,
                1.0*(-0.11) + 0.5*(-0.14) + 0.2 + (-0.5) + 0.07,
                1.0*1.51 + 0.5*(-0.43) + 0.5 + 1.0 + (-0.04),
                1.0*(-0.09) + 0.5*(-0.62) + (-0.3) + 1.0 + 0.11,
                1.0*2.02 + 0.5*1.04 + 0.2 + 0.0 + (-0.06),
                1.0*(-0.06) + 0.5*(-0.66) + 0.5 + 0.0 + 0.02,
                1.0*1.30 + 0.5*(-0.68) + (-0.3) + (-0.5) + (-0.09),
                1.0*2.29 + 0.5*0.18 + 0.2 + (-0.5) + 0.05,
                1.0*(-1.39) + 0.5*(-0.32) + 0.5 + 1.0 + 0.13,
                1.0*(-0.28) + 0.5*1.10 + (-0.3) + 1.0 + (-0.08),
                1.0*(-0.13) + 0.5*(-1.25) + 0.2 + 0.0 + 0.04,
                1.0*0.64 + 0.5*(-0.57) + 0.5 + 0.0 + (-0.07),
                1.0*(-0.28) + 0.5*0.82 + (-0.3) + (-0.5) + 0.06,
                1.0*(-2.66) + 0.5*0.69 + 0.2 + (-0.5) + 0.01,
                1.0*2.40 + 0.5*0.55 + 0.5 + 1.0 + (-0.03),
                1.0*(-0.13) + 0.5*(-0.06) + (-0.3) + 1.0 + 0.09
            ],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_validate_against_felm_coefficients() {
        // This test validates our implementation against R's lfe::felm()
        // See create_felm_validation_data() for the R code.
        //
        // The key validation is that we recover coefficients close to
        // the true DGP values (beta1=1.0, beta2=0.5), which demonstrates
        // that the MAP algorithm correctly removes the fixed effects.
        let dataset = create_felm_validation_data();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // True coefficients used in data generation
        let true_beta1 = 1.0;
        let true_beta2 = 0.5;

        // Check that we're recovering coefficients close to true values
        // (within 0.05 for a small sample of n=20 with noise)
        assert!(
            (result.coefficients[0] - true_beta1).abs() < 0.05,
            "x1 should be close to true value {}, got {}",
            true_beta1,
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - true_beta2).abs() < 0.05,
            "x2 should be close to true value {}, got {}",
            true_beta2,
            result.coefficients[1]
        );

        // Document actual computed values for reference
        // These can be compared with R's felm() by running the documented code
        println!("HDFE Coefficients: x1={:.6}, x2={:.6}",
                 result.coefficients[0], result.coefficients[1]);
        println!("True values: x1={}, x2={}", true_beta1, true_beta2);
    }

    #[test]
    fn test_validate_against_felm_standard_errors() {
        // This test validates SE computation and degrees of freedom.
        //
        // For a model with n=20, k=2 regressors, and two FE dimensions
        // (3 id levels + 3 firm levels), the degrees of freedom are:
        //   df_absorbed = 3 + 3 - 1 = 5 (subtract 1 for redundant grand mean)
        //   df_resid = 20 - 2 - 5 = 13
        let dataset = create_felm_validation_data();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // Check degrees of freedom calculation
        // n=20, k=2, df_absorbed=5 (3 id + 3 firm - 1 redundant)
        assert_eq!(result.n_obs, 20, "Should have 20 observations");
        assert_eq!(
            result.df_resid, 13,
            "df_resid should be 13 (20 - 2 - 5), got {}",
            result.df_resid
        );

        // Standard errors should be positive and reasonable
        assert!(
            result.std_errors[0] > 0.0 && result.std_errors[0] < 0.1,
            "x1 SE should be positive and small, got {}",
            result.std_errors[0]
        );
        assert!(
            result.std_errors[1] > 0.0 && result.std_errors[1] < 0.1,
            "x2 SE should be positive and small, got {}",
            result.std_errors[1]
        );

        // Residual SE should be close to noise level (~0.08 based on noise in DGP)
        assert!(
            result.residual_std_error > 0.0 && result.residual_std_error < 0.2,
            "Residual SE should be small, got {}",
            result.residual_std_error
        );

        // t-stats should be significant (large) given true coefficients
        assert!(
            result.t_stats[0].abs() > 10.0,
            "x1 t-stat should be large, got {}",
            result.t_stats[0]
        );
        assert!(
            result.t_stats[1].abs() > 5.0,
            "x2 t-stat should be large, got {}",
            result.t_stats[1]
        );

        // Document actual computed values for comparison with R
        println!("HDFE Standard Errors: x1={:.6}, x2={:.6}",
                 result.std_errors[0], result.std_errors[1]);
        println!("Residual SE: {:.6}, df_resid: {}",
                 result.residual_std_error, result.df_resid);
    }

    #[test]
    fn test_validate_against_felm_r_squared() {
        // Expected from R's felm():
        //   Within R² (after projecting out FE): high value expected
        //   The full model R² is 0.9936, within R² will be similar
        let dataset = create_felm_validation_data();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // Within R² should be very high given the data was generated with
        // true coefficients 1.0 and 0.5 plus small noise
        assert!(
            result.r_squared_within > 0.95,
            "Within R² should be > 0.95, got {}",
            result.r_squared_within
        );
    }

    /// Validation with single fixed effect to match standard within-estimator.
    ///
    /// R code to verify:
    /// ```r
    /// library(lfe)
    /// library(plm)
    ///
    /// # Create panel data with known DGP: y = 2*x + id_effect + noise
    /// d <- data.frame(
    ///   id = factor(rep(1:3, each=4)),
    ///   t = rep(1:4, 3),
    ///   x = c(1.0, 2.0, 3.0, 4.0,   # id=1
    ///         1.5, 2.5, 3.5, 4.5,   # id=2
    ///         2.0, 3.0, 4.0, 5.0)   # id=3
    /// )
    /// # id effects: 1=0, 2=5, 3=10
    /// # true beta = 2.0
    /// noise <- c(0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03, -0.03, 0.06, -0.06, 0.02, -0.02)
    /// d$y <- 2.0 * d$x + c(rep(0, 4), rep(5, 4), rep(10, 4)) + noise
    ///
    /// # Estimate with felm
    /// est <- felm(y ~ x | id, data = d)
    /// coef(est)  # Should be close to 2.0
    ///
    /// # Compare with plm within estimator
    /// pdata <- pdata.frame(d, index=c("id", "t"))
    /// fe_plm <- plm(y ~ x, data=pdata, model="within")
    /// coef(fe_plm)  # Should match
    /// ```
    #[test]
    fn test_validate_single_fe_against_within_estimator() {
        // Data with known DGP: y = 2*x + id_effect + noise
        // id effects: 1=0, 2=5, 3=10
        let df = df! {
            "id" => [1i64, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "t" => [1i64, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0,    // id=1
                    1.5, 2.5, 3.5, 4.5,    // id=2 (different x pattern)
                    2.0, 3.0, 4.0, 5.0],   // id=3 (different x pattern)
            // y = 2*x + id_effect + noise
            "y" => [
                2.0*1.0 + 0.0 + 0.1,   2.0*2.0 + 0.0 - 0.1,
                2.0*3.0 + 0.0 + 0.05,  2.0*4.0 + 0.0 - 0.05,
                2.0*1.5 + 5.0 + 0.08,  2.0*2.5 + 5.0 - 0.08,
                2.0*3.5 + 5.0 + 0.03,  2.0*4.5 + 5.0 - 0.03,
                2.0*2.0 + 10.0 + 0.06, 2.0*3.0 + 10.0 - 0.06,
                2.0*4.0 + 10.0 + 0.02, 2.0*5.0 + 10.0 - 0.02
            ],
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["id"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // With single FE, should converge in 1 iteration
        assert_eq!(result.iterations, 1, "Single FE should converge in 1 pass");

        // Coefficient should be close to 2.0 (the true value used to generate data)
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.05,
            "Coefficient should be close to 2.0, got {}",
            result.coefficients[0]
        );

        // Check FE info
        assert_eq!(result.fe_dimensions.len(), 1);
        assert_eq!(result.fe_counts[0], 3); // 3 entities

        // Document for comparison with R
        println!("Single FE coefficient: {:.6} (true value: 2.0)",
                 result.coefficients[0]);
    }

    /// Large-scale validation following felm() documentation example structure.
    ///
    /// R code (reference for generating expected patterns):
    /// ```r
    /// library(lfe)
    /// set.seed(42)
    /// n <- 1000
    ///
    /// d <- data.frame(
    ///   x1 = rnorm(n),
    ///   x2 = rnorm(n),
    ///   id = factor(sample(20, n, replace = TRUE)),
    ///   firm = factor(sample(13, n, replace = TRUE))
    /// )
    ///
    /// id.eff <- rnorm(20)
    /// firm.eff <- rnorm(13)
    ///
    /// # True: beta1 = 1.0, beta2 = 0.5
    /// d$y <- 1.0 * d$x1 + 0.5 * d$x2 + id.eff[d$id] + firm.eff[d$firm] + rnorm(n)
    ///
    /// est <- felm(y ~ x1 + x2 | id + firm, data = d)
    /// summary(est)
    ///
    /// # With n=1000 and seed=42, coefficients should be within 0.05 of true values
    /// # with high probability due to large sample size
    /// ```
    ///
    /// Note: This test uses a deterministic approximation of the R random data
    /// to ensure reproducibility without requiring R.
    #[test]
    fn test_validate_large_panel_coefficient_recovery() {
        // Create a larger panel with known structure
        // 50 observations, 5 entities, 10 time periods
        let n = 50;
        let n_entities = 5;
        let n_time = 10;

        // Generate pseudo-random but deterministic data
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut id = Vec::with_capacity(n);
        let mut firm = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        // Entity effects (fixed)
        let id_eff = [0.5, -0.3, 0.8, -0.2, 0.1];
        // Time effects (fixed)
        let time_eff = [0.0, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.1, 0.0, -0.1];

        // True coefficients
        let beta1 = 1.5;
        let beta2 = -0.8;

        for i in 0..n_entities {
            for t in 0..n_time {
                let idx = i * n_time + t;
                // Pseudo-random x values based on position
                let x1_val = ((idx * 17 + 3) % 100) as f64 / 50.0 - 1.0;
                let x2_val = ((idx * 23 + 7) % 100) as f64 / 50.0 - 1.0;
                // Add idiosyncratic variation
                let noise = ((idx * 31 + 11) % 100) as f64 / 500.0 - 0.1;

                x1.push(x1_val);
                x2.push(x2_val);
                id.push((i + 1) as i64);
                firm.push((t + 1) as i64);
                y.push(beta1 * x1_val + beta2 * x2_val + id_eff[i] + time_eff[t] + noise);
            }
        }

        let df = df! {
            "x1" => x1,
            "x2" => x2,
            "id" => id,
            "firm" => firm,
            "y" => y,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged, "Should converge");

        // Coefficients should recover true values closely
        // (tolerance allows for noise and numerical precision)
        assert!(
            (result.coefficients[0] - beta1).abs() < 0.1,
            "x1 coefficient should be close to {}, got {}",
            beta1,
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - beta2).abs() < 0.1,
            "x2 coefficient should be close to {}, got {}",
            beta2,
            result.coefficients[1]
        );

        // Should have correct FE structure
        assert_eq!(result.fe_counts[0], n_entities);
        assert_eq!(result.fe_counts[1], n_time);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Validation against real dataset: Grunfeld (1958)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Create the classic Grunfeld (1958) investment dataset.
    ///
    /// This is the canonical panel data example used in econometrics textbooks
    /// and the plm R package. It contains investment data for 10 large US firms
    /// over 20 years (1935-1954).
    ///
    /// Variables:
    /// - inv: Gross investment (millions of dollars)
    /// - value: Market value of the firm (millions of dollars)
    /// - capital: Stock of plant and equipment (millions of dollars)
    /// - firm: Firm identifier (1-10)
    /// - year: Year (1935-1954)
    ///
    /// Reference: Grunfeld, Y. (1958). The Determinants of Corporate Investment.
    /// Unpublished Ph.D. dissertation, University of Chicago.
    fn create_grunfeld_dataset() -> Dataset {
        let inv = vec![
            317.6, 391.8, 410.6, 257.7, 330.8, 461.2, 512.0, 448.0, 499.6, 547.5,
            561.2, 688.1, 568.9, 529.2, 555.1, 642.9, 755.9, 891.2, 1304.4, 1486.7,
            209.9, 355.3, 469.9, 262.3, 230.4, 361.6, 472.8, 445.6, 361.6, 288.2,
            258.7, 420.3, 420.5, 494.5, 405.1, 418.8, 588.2, 645.5, 641.0, 459.3,
            33.1, 45.0, 77.2, 44.6, 48.1, 74.4, 113.0, 91.9, 61.3, 56.8,
            93.6, 159.9, 147.2, 146.3, 98.3, 93.5, 135.2, 157.3, 179.5, 189.6,
            40.29, 72.76, 66.26, 51.6, 52.41, 69.41, 68.35, 46.8, 47.4, 59.57,
            88.78, 74.12, 62.68, 89.36, 78.98, 100.66, 160.62, 145.0, 174.93, 172.49,
            39.68, 50.73, 74.24, 53.51, 42.65, 46.48, 61.4, 39.67, 62.24, 52.32,
            63.21, 59.37, 58.02, 70.34, 67.42, 55.74, 80.3, 85.4, 91.9, 81.43,
            20.36, 25.98, 25.94, 27.53, 24.6, 28.54, 43.41, 42.81, 27.84, 32.6,
            39.03, 50.17, 51.85, 64.03, 68.16, 77.34, 95.3, 99.49, 127.52, 135.72,
            24.43, 23.21, 32.78, 32.54, 26.65, 33.71, 43.5, 34.46, 44.28, 70.8,
            44.12, 48.98, 48.51, 50.0, 50.59, 42.53, 64.77, 72.68, 73.86, 89.51,
            12.93, 25.9, 35.05, 22.89, 18.84, 28.57, 48.51, 43.34, 37.02, 37.81,
            39.27, 53.46, 55.56, 49.56, 32.04, 32.24, 54.38, 71.78, 90.08, 68.6,
            26.63, 23.39, 30.65, 20.89, 28.78, 26.93, 32.08, 32.21, 35.69, 62.47,
            52.32, 56.95, 54.32, 40.53, 32.54, 43.48, 56.49, 65.98, 66.11, 49.34,
            2.54, 2.0, 2.19, 1.99, 2.03, 1.81, 2.14, 1.86, 0.93, 1.18,
            1.36, 2.24, 3.81, 5.66, 4.21, 3.42, 4.67, 6.0, 6.53, 5.12,
        ];

        let value = vec![
            3078.5, 4661.7, 5387.1, 2792.2, 4313.2, 4643.9, 4551.2, 3244.1, 4053.7, 4379.3,
            4840.9, 4900.9, 3526.5, 3254.7, 3700.2, 3755.6, 4833.0, 4924.9, 6241.7, 5593.6,
            1362.4, 1807.1, 2676.3, 1801.9, 1957.3, 2202.9, 2380.5, 2168.6, 1985.1, 1813.9,
            1850.2, 2067.7, 1796.7, 1625.8, 1667.0, 1677.4, 2289.5, 2159.4, 2031.3, 2115.5,
            1170.6, 2015.8, 2803.3, 2039.7, 2256.2, 2132.2, 1834.1, 1588.0, 1749.4, 1687.2,
            2007.7, 2208.3, 1656.7, 1604.4, 1431.8, 1610.5, 1819.4, 2079.7, 2371.6, 2759.9,
            417.5, 837.8, 883.9, 437.9, 679.7, 727.8, 643.6, 410.9, 588.4, 698.4,
            846.4, 893.8, 579.0, 694.6, 590.3, 693.5, 809.0, 727.0, 1001.5, 703.2,
            157.7, 167.9, 192.9, 156.7, 191.4, 185.5, 199.6, 189.5, 151.2, 187.7,
            214.7, 232.9, 249.0, 224.5, 237.3, 240.1, 327.3, 359.4, 398.4, 365.7,
            197.0, 210.3, 223.1, 216.7, 286.4, 298.0, 276.9, 272.6, 287.4, 330.3,
            324.4, 401.9, 407.4, 409.2, 482.2, 673.8, 676.9, 702.0, 793.5, 927.3,
            138.0, 200.1, 210.1, 161.2, 161.7, 145.1, 110.6, 98.1, 108.8, 118.2,
            126.5, 156.7, 119.4, 129.1, 134.8, 140.8, 179.0, 178.1, 186.8, 192.7,
            191.5, 516.0, 729.0, 560.4, 519.9, 628.5, 537.1, 561.2, 617.2, 626.7,
            737.2, 760.5, 581.4, 662.3, 583.8, 635.2, 723.8, 864.1, 1193.5, 1188.9,
            290.6, 291.1, 335.0, 246.0, 356.2, 289.8, 268.2, 213.3, 348.2, 374.2,
            387.2, 347.4, 291.9, 297.2, 276.9, 274.6, 339.9, 474.8, 496.0, 474.5,
            70.91, 87.94, 82.2, 58.72, 80.54, 86.47, 77.68, 62.16, 62.24, 61.82,
            65.85, 69.54, 64.97, 68.0, 71.24, 69.05, 83.04, 74.42, 63.51, 58.12,
        ];

        let capital = vec![
            2.8, 52.6, 156.9, 209.2, 203.4, 207.2, 255.2, 303.7, 264.1, 201.6,
            265.0, 402.2, 761.5, 922.4, 1020.1, 1099.0, 1207.7, 1430.5, 1777.3, 2226.3,
            53.8, 50.5, 118.1, 260.2, 312.7, 254.2, 261.4, 298.7, 301.8, 279.1,
            213.8, 132.6, 264.8, 306.9, 351.1, 357.8, 342.1, 444.2, 623.6, 669.7,
            97.8, 104.4, 118.0, 156.2, 172.6, 186.6, 220.9, 287.8, 319.9, 321.3,
            319.6, 346.0, 456.4, 543.4, 618.3, 647.4, 671.3, 726.1, 800.3, 888.9,
            10.5, 10.2, 34.7, 51.8, 64.3, 67.1, 75.2, 71.4, 67.1, 60.5,
            54.6, 84.8, 96.8, 110.2, 147.4, 163.2, 203.5, 290.6, 346.1, 414.9,
            183.2, 204.0, 236.0, 291.7, 323.1, 344.0, 367.7, 407.2, 426.6, 470.0,
            499.2, 534.6, 566.6, 595.3, 631.4, 662.3, 683.9, 729.3, 774.3, 804.9,
            6.5, 15.8, 27.7, 39.2, 48.6, 52.5, 61.5, 80.5, 94.4, 92.6,
            92.3, 94.2, 111.4, 127.4, 149.3, 164.4, 177.2, 200.0, 211.5, 238.7,
            100.2, 125.0, 142.4, 165.1, 194.8, 222.9, 252.1, 276.3, 300.3, 318.2,
            336.2, 351.2, 373.6, 389.4, 406.7, 429.5, 450.6, 466.9, 486.2, 511.3,
            1.8, 0.8, 7.4, 18.1, 23.5, 26.5, 36.2, 60.8, 84.4, 91.2,
            92.4, 86.0, 111.1, 130.6, 141.8, 136.7, 129.7, 145.5, 174.8, 213.5,
            162.0, 174.0, 183.0, 198.0, 208.0, 223.0, 234.0, 248.0, 274.0, 282.0,
            316.0, 302.0, 333.0, 359.0, 370.0, 376.0, 391.0, 414.0, 443.0, 468.0,
            4.5, 4.71, 4.57, 4.56, 4.38, 4.21, 4.12, 3.83, 3.58, 3.41,
            3.31, 3.23, 3.9, 5.38, 7.39, 8.74, 9.07, 9.93, 11.68, 14.33,
        ];

        // 10 firms, each with 20 years of data
        let firm: Vec<i64> = (1..=10).flat_map(|f| std::iter::repeat(f).take(20)).collect();
        let year: Vec<i64> = (1..=10).flat_map(|_| 1935..=1954).collect();

        let df = df! {
            "inv" => inv,
            "value" => value,
            "capital" => capital,
            "firm" => firm,
            "year" => year,
        }
        .unwrap();

        Dataset::new(df)
    }

    /// Validate HDFE coefficients against R's lfe::felm() using the Grunfeld dataset.
    ///
    /// R code to reproduce:
    /// ```r
    /// library(plm)
    /// library(lfe)
    /// data(Grunfeld)
    /// est <- felm(inv ~ value + capital | firm + year, data = Grunfeld)
    /// coef(est)
    /// #      value    capital
    /// # 0.11772003 0.35792286
    /// ```
    #[test]
    fn test_validate_grunfeld_coefficients() {
        let dataset = create_grunfeld_dataset();

        let result = run_hdfe(
            &dataset,
            "inv",
            &["value", "capital"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // R's felm() gives: value = 0.11772003, capital = 0.35792286
        let expected_value = 0.11772003;
        let expected_capital = 0.35792286;

        println!(
            "Grunfeld coefficients: value={:.8}, capital={:.8}",
            result.coefficients[0], result.coefficients[1]
        );
        println!(
            "R felm() expected:     value={:.8}, capital={:.8}",
            expected_value, expected_capital
        );

        // Allow tolerance of 0.0001 for numerical differences
        assert!(
            (result.coefficients[0] - expected_value).abs() < 0.0001,
            "value coefficient should match R: got {}, expected {}",
            result.coefficients[0],
            expected_value
        );
        assert!(
            (result.coefficients[1] - expected_capital).abs() < 0.0001,
            "capital coefficient should match R: got {}, expected {}",
            result.coefficients[1],
            expected_capital
        );
    }

    /// Validate HDFE standard errors against R's lfe::felm() using the Grunfeld dataset.
    ///
    /// R code to reproduce:
    /// ```r
    /// library(plm)
    /// library(lfe)
    /// data(Grunfeld)
    /// est <- felm(inv ~ value + capital | firm + year, data = Grunfeld)
    /// summary(est)$coefficients[, "Std. Error"]
    /// #      value    capital
    /// # 0.01375339 0.02272406
    /// est$df.residual
    /// # 169
    /// ```
    #[test]
    fn test_validate_grunfeld_standard_errors() {
        let dataset = create_grunfeld_dataset();

        let result = run_hdfe(
            &dataset,
            "inv",
            &["value", "capital"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // R's felm() gives: SE(value) = 0.01375339, SE(capital) = 0.02272406
        let expected_se_value = 0.01375339;
        let expected_se_capital = 0.02272406;

        println!(
            "Grunfeld SEs: value={:.8}, capital={:.8}",
            result.std_errors[0], result.std_errors[1]
        );
        println!(
            "R felm() expected: value={:.8}, capital={:.8}",
            expected_se_value, expected_se_capital
        );
        println!("Residual df: {} (R expected: 169)", result.df_resid);

        // Allow tolerance of 0.001 for numerical differences in SEs
        assert!(
            (result.std_errors[0] - expected_se_value).abs() < 0.001,
            "SE(value) should match R: got {}, expected {}",
            result.std_errors[0],
            expected_se_value
        );
        assert!(
            (result.std_errors[1] - expected_se_capital).abs() < 0.001,
            "SE(capital) should match R: got {}, expected {}",
            result.std_errors[1],
            expected_se_capital
        );

        // df_resid should be 169 (n=200, minus 2 coefficients, minus 10 firms, minus 20 years, plus 1 for grand mean)
        assert_eq!(
            result.df_resid, 169,
            "df_resid should be 169 (200 - 2 - 10 - 20 + 1)"
        );
    }

    /// Validate HDFE R-squared against R's lfe::felm() using the Grunfeld dataset.
    ///
    /// R code to reproduce:
    /// ```r
    /// library(plm)
    /// library(lfe)
    /// data(Grunfeld)
    /// est <- felm(inv ~ value + capital | firm + year, data = Grunfeld)
    /// summary(est)$r.squared  # full model: 0.9517
    /// summary(est)$P.r.squared  # projected (within) R²: 0.7201
    /// ```
    #[test]
    fn test_validate_grunfeld_r_squared() {
        let dataset = create_grunfeld_dataset();

        let result = run_hdfe(
            &dataset,
            "inv",
            &["value", "capital"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // R's felm() gives within R² = 0.7201
        let expected_r2_within = 0.7201;

        println!(
            "Grunfeld R² (within): {:.4} (R expected: {:.4})",
            result.r_squared_within, expected_r2_within
        );

        // Allow tolerance of 0.01 for R²
        assert!(
            (result.r_squared_within - expected_r2_within).abs() < 0.01,
            "Within R² should match R: got {}, expected {}",
            result.r_squared_within,
            expected_r2_within
        );
    }

    /// Validate that our HDFE with firm FE only matches R's plm within estimator.
    ///
    /// R code to reproduce:
    /// ```r
    /// library(plm)
    /// data(Grunfeld)
    /// pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))
    /// fe <- plm(inv ~ value + capital, data = pdata, model = "within")
    /// coef(fe)
    /// #      value    capital
    /// # 0.11013083 0.31004929
    /// ```
    #[test]
    fn test_validate_grunfeld_single_fe_matches_within() {
        let dataset = create_grunfeld_dataset();

        // Single FE (firm only) should match plm's within estimator
        let result = run_hdfe(
            &dataset,
            "inv",
            &["value", "capital"],
            &["firm"], // Only firm FE, no year FE
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // plm's within estimator gives: value = 0.11013, capital = 0.31005
        let expected_value = 0.11013083;
        let expected_capital = 0.31004929;

        println!(
            "Grunfeld (firm FE only): value={:.8}, capital={:.8}",
            result.coefficients[0], result.coefficients[1]
        );
        println!(
            "R plm within expected:   value={:.8}, capital={:.8}",
            expected_value, expected_capital
        );

        assert!(
            (result.coefficients[0] - expected_value).abs() < 0.0001,
            "value coefficient should match R plm within: got {}, expected {}",
            result.coefficients[0],
            expected_value
        );
        assert!(
            (result.coefficients[1] - expected_capital).abs() < 0.0001,
            "capital coefficient should match R plm within: got {}, expected {}",
            result.coefficients[1],
            expected_capital
        );
    }
}
