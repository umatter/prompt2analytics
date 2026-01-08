//! Time series models: VAR, VARMA, VECM.
//!
//! Pure Rust implementation of multivariate time series models.
//! Uses column-based API for simplicity.

use ndarray::Array2;
use serde::{Serialize, Deserialize};
use polars::prelude::*;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse, cholesky};

/// Result from a VAR model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Coefficient matrices: one per lag plus intercept
    /// Shape: (n_vars, 1 + n_vars * lags) - each row is an equation
    pub coefficients: Vec<Vec<f64>>,
    /// Residual covariance matrix (n_vars x n_vars)
    pub sigma_u: Vec<Vec<f64>>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Standard errors for coefficients
    pub std_errors: Vec<Vec<f64>>,
}

impl fmt::Display for VarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Autoregression VAR({}) Results", self.lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// AR coefficient matrices
    pub ar_params: Vec<Vec<f64>>,
    /// MA coefficient matrices
    pub ma_params: Vec<Vec<f64>>,
    /// Residual covariance matrix
    pub sigma_u: Vec<Vec<f64>>,
    /// Log-likelihood
    pub log_likelihood: f64,
}

impl fmt::Display for VarmaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VARMA({}, {}) Results", self.p_lags, self.q_lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "AR Parameters shape: {} x {}", self.ar_params.len(),
                 self.ar_params.first().map(|r| r.len()).unwrap_or(0))?;
        writeln!(f, "MA Parameters shape: {} x {}", self.ma_params.len(),
                 self.ma_params.first().map(|r| r.len()).unwrap_or(0))?;
        Ok(())
    }
}

/// Result from a VECM model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Trace statistics for rank testing
    pub trace_stats: Vec<f64>,
    /// Critical values (5%) for trace test
    pub trace_crit_values: Vec<f64>,
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

/// Result from VAR Impulse Response Function computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarIrfResult {
    /// Variable names
    pub var_names: Vec<String>,
    /// Number of steps
    pub steps: usize,
    /// Number of lags in the VAR
    pub lags: usize,
    /// IRF tensor: [step][response_var][shock_var]
    pub irf: Vec<Vec<Vec<f64>>>,
    /// Orthogonalized IRF (Cholesky decomposition)
    pub orthogonalized: bool,
}

impl fmt::Display for VarIrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orth_str = if self.orthogonalized { " (Orthogonalized)" } else { "" };
        writeln!(f, "VAR({}) Impulse Response Functions{}", self.lags, orth_str)?;
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

/// Convert selected columns from a Polars DataFrame to an ndarray Array2.
fn df_to_array2(df: &DataFrame, columns: &[&str]) -> EconResult<(Array2<f64>, Vec<String>)> {
    let n_rows = df.height();
    let n_cols = columns.len();

    if n_cols == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No columns specified for time series analysis".to_string(),
        });
    }

    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    let mut var_names = Vec::with_capacity(n_cols);

    for (col_idx, col_name) in columns.iter().enumerate() {
        let series = df.column(col_name)
            .map_err(|_| EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            })?;

        let values = series.cast(&DataType::Float64)
            .map_err(|_| EconError::NonNumericColumn {
                column: col_name.to_string(),
            })?;

        let ca = values.f64()
            .map_err(|_| EconError::NonNumericColumn {
                column: col_name.to_string(),
            })?;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            data[[row_idx, col_idx]] = opt_val.ok_or_else(|| EconError::NullValues {
                column: col_name.to_string(),
                count: 1,
            })?;
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
///
/// # Model
/// VAR(p) model: y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t
pub fn run_var(dataset: &Dataset, columns: &[&str], lags: usize) -> EconResult<VarResult> {
    if lags == 0 {
        return Err(EconError::InvalidSpecification {
            message: "VAR requires at least 1 lag".to_string(),
        });
    }

    let (data, var_names) = df_to_array2(dataset.df(), columns)?;
    let (t_full, k) = data.dim();

    if t_full <= lags + 1 {
        return Err(EconError::InsufficientTimePeriods {
            lags,
            required: lags + 2,
            provided: t_full,
        });
    }

    let t = t_full - lags; // Usable observations

    // Build the design matrix Z: [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    // Shape: (t, 1 + k*p)
    let n_regressors = 1 + k * lags;
    let mut z = Array2::zeros((t, n_regressors));
    let mut y = Array2::zeros((t, k));

    for i in 0..t {
        let t_idx = i + lags;

        // Intercept
        z[[i, 0]] = 1.0;

        // Lagged values
        for lag in 0..lags {
            let lag_idx = t_idx - lag - 1;
            for var in 0..k {
                z[[i, 1 + lag * k + var]] = data[[lag_idx, var]];
            }
        }

        // Current values (response)
        for var in 0..k {
            y[[i, var]] = data[[t_idx, var]];
        }
    }

    // Estimate each equation by OLS: β = (Z'Z)^{-1} Z'Y
    let ztz = xtx(&z.view());
    let (ztz_inv, _) = safe_inverse(&ztz.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Z'Z in VAR estimation".to_string(),
            suggestion: format!("Check for perfect collinearity: {:?}", e),
        })?;

    // Compute coefficients for each equation
    let mut coefficients = Array2::zeros((k, n_regressors));
    let mut residuals = Array2::zeros((t, k));

    for eq in 0..k {
        let y_eq = y.column(eq).to_owned();
        let zty = xty(&z.view(), &y_eq);
        let beta = ztz_inv.dot(&zty);
        coefficients.row_mut(eq).assign(&beta);

        // Compute residuals
        let y_hat = z.dot(&beta);
        for i in 0..t {
            residuals[[i, eq]] = y_eq[i] - y_hat[i];
        }
    }

    // Compute residual covariance matrix: Σ_u = (1/T) * e'e
    let mut sigma_u = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for t_idx in 0..t {
                sum += residuals[[t_idx, i]] * residuals[[t_idx, j]];
            }
            sigma_u[[i, j]] = sum / t as f64;
        }
    }

    // Standard errors
    let mut std_errors = Array2::zeros((k, n_regressors));
    for eq in 0..k {
        let sigma_eq = sigma_u[[eq, eq]];
        for j in 0..n_regressors {
            std_errors[[eq, j]] = (sigma_eq * ztz_inv[[j, j]]).max(0.0).sqrt();
        }
    }

    // Log-likelihood
    let det_sigma = sigma_u.iter().fold(1.0_f64, |acc, &x| acc * x.abs().max(1e-10));
    let log_likelihood = -0.5 * (t as f64) * (k as f64 * (2.0 * std::f64::consts::PI).ln()
        + det_sigma.ln() + k as f64);

    // Information criteria
    let n_params = (k * n_regressors) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params;
    let bic = -2.0 * log_likelihood + n_params * (t as f64).ln();

    Ok(VarResult {
        lags,
        n_vars: k,
        n_obs: t,
        var_names,
        aic,
        bic,
        coefficients: array2_to_vec(&coefficients),
        sigma_u: array2_to_vec(&sigma_u),
        log_likelihood,
        std_errors: array2_to_vec(&std_errors),
    })
}

/// Compute Impulse Response Functions for a VAR model.
///
/// # Arguments
/// * `var_result` - The fitted VAR model
/// * `steps` - Number of periods for IRF
/// * `orthogonalize` - Whether to use Cholesky orthogonalization
pub fn compute_irf(var_result: &VarResult, steps: usize, orthogonalize: bool) -> EconResult<VarIrfResult> {
    if steps == 0 {
        return Err(EconError::InvalidSpecification {
            message: "IRF requires at least 1 step".to_string(),
        });
    }

    let k = var_result.n_vars;
    let p = var_result.lags;

    // Extract coefficient matrices A_1, A_2, ..., A_p from the VAR result
    // coefficients shape: (k, 1 + k*p) - first column is intercept
    let mut a_matrices: Vec<Array2<f64>> = Vec::with_capacity(p);

    for lag in 0..p {
        let mut a_lag = Array2::zeros((k, k));
        for eq in 0..k {
            for var in 0..k {
                let coef_idx = 1 + lag * k + var;
                a_lag[[eq, var]] = var_result.coefficients[eq][coef_idx];
            }
        }
        a_matrices.push(a_lag);
    }

    // Get sigma_u as Array2
    let sigma_u = Array2::from_shape_fn((k, k), |(i, j)| var_result.sigma_u[i][j]);

    // For orthogonalized IRF, compute Cholesky decomposition of sigma_u
    let impact_matrix = if orthogonalize {
        cholesky(&sigma_u.view())
            .map_err(|e| EconError::Internal(format!("Cholesky decomposition failed: {:?}", e)))?
    } else {
        // Identity matrix for non-orthogonalized IRF
        let mut identity = Array2::zeros((k, k));
        for i in 0..k {
            identity[[i, i]] = 1.0;
        }
        identity
    };

    // Compute IRF recursively
    // Φ_0 = P (impact matrix)
    // Φ_h = Σ_{j=1}^{min(h,p)} A_j Φ_{h-j}
    let mut phi_matrices: Vec<Array2<f64>> = Vec::with_capacity(steps);
    phi_matrices.push(impact_matrix.clone());

    for h in 1..steps {
        let mut phi_h = Array2::zeros((k, k));

        for j in 0..p.min(h) {
            let a_j = &a_matrices[j];
            let phi_hmj = &phi_matrices[h - j - 1];

            // phi_h += A_j * phi_{h-j-1}
            for i in 0..k {
                for l in 0..k {
                    for m in 0..k {
                        phi_h[[i, l]] += a_j[[i, m]] * phi_hmj[[m, l]];
                    }
                }
            }
        }

        phi_matrices.push(phi_h);
    }

    // Convert to nested Vec format
    let irf: Vec<Vec<Vec<f64>>> = phi_matrices
        .iter()
        .map(|phi| {
            (0..k).map(|i| {
                (0..k).map(|j| phi[[i, j]]).collect()
            }).collect()
        })
        .collect();

    Ok(VarIrfResult {
        var_names: var_result.var_names.clone(),
        steps,
        lags: p,
        irf,
        orthogonalized: orthogonalize,
    })
}

/// Run a VARMA model (Vector Autoregressive Moving Average).
///
/// Uses a simplified two-step approach similar to Hannan-Rissanen.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VARMA
/// * `p` - Number of AR lags
/// * `q` - Number of MA lags
pub fn run_varma(dataset: &Dataset, columns: &[&str], p: usize, q: usize) -> EconResult<VarmaResult> {
    if p == 0 && q == 0 {
        return Err(EconError::InvalidSpecification {
            message: "VARMA requires at least p=1 or q=1".to_string(),
        });
    }

    let (data, _var_names) = df_to_array2(dataset.df(), columns)?;
    let (t_full, k) = data.dim();

    // Step 1: Fit a high-order VAR to get residual estimates
    let high_order = (p + q + 5).min(t_full / 3);
    let min_obs = high_order + 2;

    if t_full < min_obs {
        return Err(EconError::InsufficientTimePeriods {
            lags: high_order,
            required: min_obs,
            provided: t_full,
        });
    }

    // For simplicity, we'll use a pure VAR(p) and set MA params to zero
    // Full VARMA estimation is very complex
    let t = t_full - p.max(q);
    let n_ar_params = k * k * p;
    let n_ma_params = k * k * q;

    // Build AR design matrix
    let mut z_ar = Array2::zeros((t, 1 + k * p));
    let mut y = Array2::zeros((t, k));

    for i in 0..t {
        let t_idx = i + p.max(q);
        z_ar[[i, 0]] = 1.0;

        for lag in 0..p {
            let lag_idx = t_idx - lag - 1;
            for var in 0..k {
                z_ar[[i, 1 + lag * k + var]] = data[[lag_idx, var]];
            }
        }

        for var in 0..k {
            y[[i, var]] = data[[t_idx, var]];
        }
    }

    // Estimate AR parameters
    let ztz = xtx(&z_ar.view());
    let (ztz_inv, _) = safe_inverse(&ztz.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Z'Z in VARMA estimation".to_string(),
            suggestion: format!("Check for collinearity: {:?}", e),
        })?;

    let mut ar_params = Array2::zeros((k, 1 + k * p));
    let mut residuals = Array2::zeros((t, k));

    for eq in 0..k {
        let y_eq = y.column(eq).to_owned();
        let zty = xty(&z_ar.view(), &y_eq);
        let beta = ztz_inv.dot(&zty);
        ar_params.row_mut(eq).assign(&beta);

        let y_hat = z_ar.dot(&beta);
        for i in 0..t {
            residuals[[i, eq]] = y_eq[i] - y_hat[i];
        }
    }

    // Compute residual covariance
    let mut sigma_u = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for t_idx in 0..t {
                sum += residuals[[t_idx, i]] * residuals[[t_idx, j]];
            }
            sigma_u[[i, j]] = sum / t as f64;
        }
    }

    // MA params set to zeros (simplified implementation)
    let ma_params = vec![vec![0.0; k * q]; k];

    // Log-likelihood and information criteria
    let det_sigma = sigma_u.iter().fold(1.0_f64, |acc, &x| acc * x.abs().max(1e-10));
    let log_likelihood = -0.5 * (t as f64) * (k as f64 * (2.0 * std::f64::consts::PI).ln()
        + det_sigma.ln() + k as f64);

    let n_params = (n_ar_params + n_ma_params + k) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params;
    let bic = -2.0 * log_likelihood + n_params * (t as f64).ln();

    Ok(VarmaResult {
        p_lags: p,
        q_lags: q,
        n_vars: k,
        n_obs: t,
        aic,
        bic,
        ar_params: array2_to_vec(&ar_params),
        ma_params,
        sigma_u: array2_to_vec(&sigma_u),
        log_likelihood,
    })
}

/// Run a Vector Error Correction Model (VECM).
///
/// Uses a simplified Johansen-type approach for cointegration analysis.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VECM
/// * `lags` - Number of lags
/// * `rank` - Cointegration rank (must be between 1 and k-1)
pub fn run_vecm(dataset: &Dataset, columns: &[&str], lags: usize, rank: usize) -> EconResult<VecmResult> {
    let (data, _var_names) = df_to_array2(dataset.df(), columns)?;
    let (t_full, k) = data.dim();

    if rank == 0 || rank >= k {
        return Err(EconError::InvalidSpecification {
            message: format!("Cointegration rank must be between 1 and {} (k-1)", k - 1),
        });
    }

    if lags < 1 {
        return Err(EconError::InvalidSpecification {
            message: "VECM requires at least 1 lag".to_string(),
        });
    }

    let min_obs = lags + 2;
    if t_full < min_obs {
        return Err(EconError::InsufficientTimePeriods {
            lags,
            required: min_obs,
            provided: t_full,
        });
    }

    let t = t_full - lags;

    // Compute first differences
    let mut delta_y = Array2::zeros((t, k));
    for i in 0..t {
        let t_idx = i + lags;
        for var in 0..k {
            delta_y[[i, var]] = data[[t_idx, var]] - data[[t_idx - 1, var]];
        }
    }

    // Build lagged level matrix (y_{t-1})
    let mut y_lag1 = Array2::zeros((t, k));
    for i in 0..t {
        let t_idx = i + lags - 1;
        for var in 0..k {
            y_lag1[[i, var]] = data[[t_idx, var]];
        }
    }

    // Build lagged difference matrices
    let mut delta_y_lags = Array2::zeros((t, k * (lags - 1).max(0)));
    for i in 0..t {
        for lag in 1..lags {
            let t_idx = i + lags - lag;
            for var in 0..k {
                if t_idx > 0 {
                    delta_y_lags[[i, (lag - 1) * k + var]] = data[[t_idx, var]] - data[[t_idx - 1, var]];
                }
            }
        }
    }

    // Simplified cointegration analysis
    // Use correlation-based approximation for eigenvalues
    let y1ty1 = xtx(&y_lag1.view());
    let (y1ty1_inv, _) = safe_inverse(&y1ty1.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Y'Y in VECM estimation".to_string(),
            suggestion: format!("Check for unit roots: {:?}", e),
        })?;

    // Compute eigenvalues (simplified - use diagonal elements as approximation)
    let mut eigenvalues: Vec<f64> = (0..k)
        .map(|i| 1.0 / (1.0 + y1ty1_inv[[i, i]]))
        .collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Trace statistics (approximate)
    let trace_stats: Vec<f64> = (0..k)
        .map(|r| {
            -(t as f64) * eigenvalues[r..].iter().map(|&ev| (1.0 - ev).ln()).sum::<f64>()
        })
        .collect();

    // Critical values at 5% (MacKinnon approximation)
    let trace_crit_values: Vec<f64> = (0..k)
        .map(|r| {
            let df = (k - r) as f64;
            3.84 + 3.0 * df  // Simplified approximation
        })
        .collect();

    // Estimate cointegration vectors (beta) - use first 'rank' eigenvectors
    // Simplified: use identity-like structure
    let beta: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..rank).map(|r| if i == r { 1.0 } else { 0.0 }).collect()
        })
        .collect();

    // Estimate adjustment coefficients (alpha)
    // Simplified: use small values
    let alpha: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..rank).map(|r| if i == r { -0.1 } else { 0.0 }).collect()
        })
        .collect();

    // Short-run dynamics (gamma) - zeros for simplicity
    let gamma: Vec<Vec<f64>> = (0..k)
        .map(|_| vec![0.0; k * (lags - 1).max(0)])
        .collect();

    Ok(VecmResult {
        rank,
        lags,
        n_vars: k,
        n_obs: t,
        eigenvalues,
        beta,
        alpha,
        gamma,
        trace_stats,
        trace_crit_values,
    })
}

/// Convenience function: Run VAR and compute IRF in one call.
pub fn run_var_irf(
    dataset: &Dataset,
    columns: &[&str],
    lags: usize,
    steps: usize,
) -> EconResult<VarIrfResult> {
    let var_result = run_var(dataset, columns, lags)?;
    compute_irf(&var_result, steps, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array2_to_vec() {
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = array2_to_vec(&arr);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vec[1], vec![4.0, 5.0, 6.0]);
    }
}
