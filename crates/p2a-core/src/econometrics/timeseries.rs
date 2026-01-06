//! Time series models: VAR, VARMA, VECM.
//!
//! Multivariate time series models for econometric analysis.

use anyhow::{anyhow, Result};
use greeners::{VAR, VARMA, VECM};
use ndarray::Array2;
use polars::prelude::*;
use std::fmt;

use crate::data::Dataset;

/// Result from a VAR model estimation.
#[derive(Debug, Clone)]
pub struct VarResult {
    /// Number of lags
    pub lags: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of observations used
    pub n_obs: usize,
    /// Variable names
    pub var_names: Vec<String>,
    /// AIC information criterion
    pub aic: f64,
    /// BIC information criterion
    pub bic: f64,
    /// Coefficient matrix (flattened: intercept + lag coefficients)
    pub params: Vec<Vec<f64>>,
    /// Residual covariance matrix
    pub sigma_u: Vec<Vec<f64>>,
}

impl fmt::Display for VarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Autoregression VAR({}) Results", self.lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Residual Covariance Matrix (Sigma_u):")?;
        for (i, row) in self.sigma_u.iter().enumerate() {
            write!(f, "  {}: [", self.var_names.get(i).unwrap_or(&format!("Var{}", i)))?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:.4}", val)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

/// Result from a VARMA model estimation.
#[derive(Debug, Clone)]
pub struct VarmaResult {
    /// AR lags (p)
    pub p_lags: usize,
    /// MA lags (q)
    pub q_lags: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of observations used
    pub n_obs: usize,
    /// AIC information criterion
    pub aic: f64,
    /// BIC information criterion
    pub bic: f64,
    /// AR coefficient matrix
    pub ar_params: Vec<Vec<f64>>,
    /// MA coefficient matrix
    pub ma_params: Vec<Vec<f64>>,
    /// Residual covariance matrix
    pub sigma_u: Vec<Vec<f64>>,
}

impl fmt::Display for VarmaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VARMA({}, {}) Results (Hannan-Rissanen)", self.p_lags, self.q_lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "AR Parameters shape: {} x {}", self.ar_params.len(),
                 self.ar_params.first().map(|r| r.len()).unwrap_or(0))?;
        writeln!(f, "MA Parameters shape: {} x {}",self.ma_params.len(),
                 self.ma_params.first().map(|r| r.len()).unwrap_or(0))?;
        Ok(())
    }
}

/// Result from a VECM model estimation.
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Cointegration rank
    pub rank: usize,
    /// Number of lags
    pub lags: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of observations used
    pub n_obs: usize,
    /// Johansen eigenvalues
    pub eigenvalues: Vec<f64>,
    /// Cointegration vectors (beta) - long-run equilibrium relationships
    pub beta: Vec<Vec<f64>>,
    /// Adjustment coefficients (alpha) - speed of correction
    pub alpha: Vec<Vec<f64>>,
    /// Short-run dynamics (gamma)
    pub gamma: Vec<Vec<f64>>,
}

impl fmt::Display for VecmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Error Correction Model (VECM) Results")?;
        writeln!(f, "==============================================")?;
        writeln!(f, "Cointegration Rank: {}", self.rank)?;
        writeln!(f, "Lags: {}", self.lags)?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(f, "Johansen Eigenvalues:")?;
        write!(f, "  [")?;
        for (i, ev) in self.eigenvalues.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", ev)?;
        }
        writeln!(f, "]")?;
        writeln!(f)?;

        writeln!(f, "Cointegration Vectors (Beta) - Long-run relationships:")?;
        for (i, row) in self.beta.iter().enumerate() {
            write!(f, "  Var{}: [", i)?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:.4}", val)?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f)?;

        writeln!(f, "Adjustment Coefficients (Alpha) - Speed of correction:")?;
        for (i, row) in self.alpha.iter().enumerate() {
            write!(f, "  Var{}: [", i)?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:.4}", val)?;
            }
            writeln!(f, "]")?;
        }

        Ok(())
    }
}

/// Convert selected columns from a Polars DataFrame to an ndarray Array2.
fn df_to_array2(df: &DataFrame, columns: &[&str]) -> Result<(Array2<f64>, Vec<String>)> {
    let n_rows = df.height();
    let n_cols = columns.len();

    if n_cols == 0 {
        return Err(anyhow!("No columns specified for time series analysis"));
    }

    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    let mut var_names = Vec::with_capacity(n_cols);

    for (col_idx, col_name) in columns.iter().enumerate() {
        let series = df.column(col_name)
            .map_err(|e| anyhow!("Column '{}' not found: {}", col_name, e))?;

        let values = series.cast(&DataType::Float64)
            .map_err(|e| anyhow!("Cannot convert column '{}' to float: {}", col_name, e))?;

        let ca = values.f64()
            .map_err(|e| anyhow!("Failed to get float values for '{}': {}", col_name, e))?;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            data[[row_idx, col_idx]] = opt_val.ok_or_else(||
                anyhow!("NULL value in column '{}' at row {}", col_name, row_idx))?;
        }

        var_names.push(col_name.to_string());
    }

    Ok((data, var_names))
}

/// Convert ndarray Array2 to Vec<Vec<f64>>.
fn array2_to_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

/// Run a Vector Autoregression (VAR) model.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VAR
/// * `lags` - Number of lags to include
pub fn run_var(dataset: &Dataset, columns: &[&str], lags: usize) -> Result<VarResult> {
    if lags == 0 {
        return Err(anyhow!("VAR requires at least 1 lag"));
    }

    let (data, var_names) = df_to_array2(dataset.df(), columns)?;

    let result = VAR::fit(&data, lags, Some(var_names.clone()))
        .map_err(|e| anyhow!("VAR estimation failed: {:?}", e))?;

    Ok(VarResult {
        lags: result.lags,
        n_vars: result.n_vars,
        n_obs: result.n_obs,
        var_names: result.var_names,
        aic: result.aic,
        bic: result.bic,
        params: array2_to_vec(&result.params),
        sigma_u: array2_to_vec(&result.sigma_u),
    })
}

/// Run a VARMA model (Vector Autoregressive Moving Average).
///
/// Uses the Hannan-Rissanen two-step estimation method.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VARMA
/// * `p` - Number of AR lags
/// * `q` - Number of MA lags
pub fn run_varma(dataset: &Dataset, columns: &[&str], p: usize, q: usize) -> Result<VarmaResult> {
    if p == 0 && q == 0 {
        return Err(anyhow!("VARMA requires at least p=1 or q=1"));
    }

    let (data, _var_names) = df_to_array2(dataset.df(), columns)?;

    let result = VARMA::fit(&data, p, q)
        .map_err(|e| anyhow!("VARMA estimation failed: {:?}", e))?;

    Ok(VarmaResult {
        p_lags: result.p_lags,
        q_lags: result.q_lags,
        n_vars: result.n_vars,
        n_obs: result.n_obs,
        aic: result.aic,
        bic: result.bic,
        ar_params: array2_to_vec(&result.ar_params),
        ma_params: array2_to_vec(&result.ma_params),
        sigma_u: array2_to_vec(&result.sigma_u),
    })
}

/// Run a Vector Error Correction Model (VECM).
///
/// Uses Johansen's Maximum Likelihood estimation method.
/// Appropriate for cointegrated time series.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VECM
/// * `lags` - Number of lags
/// * `rank` - Cointegration rank (must be between 1 and k-1 where k is number of variables)
pub fn run_vecm(dataset: &Dataset, columns: &[&str], lags: usize, rank: usize) -> Result<VecmResult> {
    let (data, _var_names) = df_to_array2(dataset.df(), columns)?;

    let n_vars = columns.len();
    if rank == 0 || rank >= n_vars {
        return Err(anyhow!("Cointegration rank must be between 1 and {} (k-1)", n_vars - 1));
    }

    if lags < 2 {
        return Err(anyhow!("VECM requires at least 2 lags"));
    }

    let result = VECM::fit(&data, lags, rank)
        .map_err(|e| anyhow!("VECM estimation failed: {:?}", e))?;

    Ok(VecmResult {
        rank: result.rank,
        lags,
        n_vars: result.n_vars,
        n_obs: result.n_obs,
        eigenvalues: result.eigenvalues.to_vec(),
        beta: array2_to_vec(&result.beta),
        alpha: array2_to_vec(&result.alpha),
        gamma: array2_to_vec(&result.gamma),
    })
}

/// Compute Impulse Response Functions for a VAR model.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VAR
/// * `lags` - Number of lags
/// * `steps` - Number of periods for IRF
pub fn run_var_irf(
    dataset: &Dataset,
    columns: &[&str],
    lags: usize,
    steps: usize
) -> Result<VarIrfResult> {
    if steps == 0 {
        return Err(anyhow!("IRF requires at least 1 step"));
    }

    let (data, var_names) = df_to_array2(dataset.df(), columns)?;

    let var_result = VAR::fit(&data, lags, Some(var_names.clone()))
        .map_err(|e| anyhow!("VAR estimation failed: {:?}", e))?;

    let irf_tensor = var_result.irf(steps)
        .map_err(|e| anyhow!("IRF computation failed: {:?}", e))?;

    // Convert 3D tensor to nested vectors: [step][response_var][shock_var]
    let mut irf_data = Vec::with_capacity(steps);
    for h in 0..steps {
        let mut step_data = Vec::with_capacity(var_names.len());
        for i in 0..var_names.len() {
            let mut response_data = Vec::with_capacity(var_names.len());
            for j in 0..var_names.len() {
                response_data.push(irf_tensor[[h, i, j]]);
            }
            step_data.push(response_data);
        }
        irf_data.push(step_data);
    }

    Ok(VarIrfResult {
        var_names,
        steps,
        lags,
        irf: irf_data,
    })
}

/// Result from VAR Impulse Response Function computation.
#[derive(Debug, Clone)]
pub struct VarIrfResult {
    /// Variable names
    pub var_names: Vec<String>,
    /// Number of steps
    pub steps: usize,
    /// Number of lags in the VAR
    pub lags: usize,
    /// IRF tensor: [step][response_var][shock_var]
    pub irf: Vec<Vec<Vec<f64>>>,
}

impl fmt::Display for VarIrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VAR({}) Impulse Response Functions", self.lags)?;
        writeln!(f, "=======================================")?;
        writeln!(f, "Steps: {}", self.steps)?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f)?;

        // Show IRF for first few steps
        let show_steps = self.steps.min(5);
        for h in 0..show_steps {
            writeln!(f, "Step {}:", h)?;
            for (i, response_var) in self.var_names.iter().enumerate() {
                write!(f, "  Response of {}: [", response_var)?;
                for (j, _shock_var) in self.var_names.iter().enumerate() {
                    if j > 0 { write!(f, ", ")?; }
                    write!(f, "{:.4}", self.irf[h][i][j])?;
                }
                writeln!(f, "]")?;
            }
        }

        if self.steps > show_steps {
            writeln!(f, "  ... ({} more steps)", self.steps - show_steps)?;
        }

        Ok(())
    }
}
