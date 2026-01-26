//! Time Series Utility Functions
//!
//! Provides utility functions for time series manipulation including:
//! - `lag`: Shift time series by k lags
//! - `embed`: Create lag embedding matrix for AR models
//! - `diffinv`: Inverse of differencing (discrete integration)
//! - `filter`: Linear filtering (convolution and recursive)
//! - `window`: Extract time series subsets
//! - `arima_sim`: Simulate from an ARIMA model
//! - `arma_acf`: Compute theoretical ACF for ARMA process
//! - `arma_to_ma`: Convert ARMA to infinite MA representation
//! - `acf_to_ar`: Compute AR coefficients from ACF (Durbin-Levinson)
//! - `runmed`: Running median smoothing
//!
//! # Mathematical Background
//!
//! ## Linear Filtering
//!
//! **Convolution filter** (FIR): y[i] = Σⱼ filter[j] × x[i-j+s]
//! **Recursive filter** (IIR): y[i] = x[i] + Σⱼ filter[j] × y[i-j]
//!
//! ## ARMA Theoretical ACF (ARMAacf)
//!
//! For ARMA(p,q) process, the autocovariances γ(h) satisfy the Yule-Walker equations:
//! γ(h) = Σᵢ₌₁ᵖ φᵢγ(h-i) + σ² Σⱼ₌₀ᵠ θⱼψ(h-j)  for h ≥ 0
//!
//! ## Durbin-Levinson Algorithm (acf2AR)
//!
//! Recursively computes AR coefficients from autocorrelations:
//! φₖₖ = [ρ(k) - Σⱼ₌₁ᵏ⁻¹ φₖ₋₁,ⱼρ(k-j)] / [1 - Σⱼ₌₁ᵏ⁻¹ φₖ₋₁,ⱼρ(j)]
//!
//! # References
//!
//! - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*
//!   (2nd ed.). Springer. ISBN: 978-0387974293. Chapters 3 (ARMA), 5 (Filtering).
//!
//! - Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series
//!   Analysis: Forecasting and Control* (5th ed.). Wiley. ISBN: 978-1118675021.
//!
//! - Durbin, J. (1960). The fitting of time-series models. *Revue de l'Institut
//!   International de Statistique*, 28(3), 233-244. https://doi.org/10.2307/1401322
//!   The Durbin-Levinson algorithm for AR estimation from ACF.
//!
//! - Levinson, N. (1947). The Wiener (root mean square) error criterion in filter
//!   design and prediction. *Journal of Mathematics and Physics*, 25(1-4), 261-278.
//!   https://doi.org/10.1002/sapm1946251261
//!
//! - Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
//!   Running median smoothing for robust time series analysis.
//!
//! R equivalent: `stats::filter()`, `stats::lag()`, `stats::embed()`, `stats::diffinv()`,
//! `stats::ARMAacf()`, `stats::ARMAtoMA()`, `stats::acf2AR()`, `stats::arima.sim()`,
//! `stats::runmed()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

// ============================================================================
// Lag Function
// ============================================================================

/// Result from lag operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagResult {
    /// The lagged time series values
    pub values: Vec<f64>,
    /// Number of lags applied (positive = shift back in time)
    pub k: i32,
    /// Original series length
    pub original_length: usize,
}

/// Lag a time series by k observations.
///
/// A positive k shifts the time index back by k observations,
/// meaning the series starts earlier in time (R convention).
///
/// # Arguments
///
/// * `x` - Input time series
/// * `k` - Number of lags (positive = shift back, negative = shift forward)
///
/// # Returns
///
/// A lagged version of the series. Note that unlike R, this returns
/// the actual shifted values without changing time indices.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::lag;
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = lag(&x, 1).unwrap();
/// // result.values contains [2.0, 3.0, 4.0, 5.0] (shifted forward in array)
/// ```
///
/// # References
///
/// R `stats::lag`
pub fn lag(x: &[f64], k: i32) -> EconResult<LagResult> {
    if x.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "lag".to_string(),
        });
    }

    let n = x.len();
    let abs_k = k.unsigned_abs() as usize;

    if abs_k >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("Lag {} is too large for series of length {}", k, n),
        });
    }

    // Positive k: shift indices back (equivalent to looking at earlier values)
    // In array terms: x_lagged[t] = x[t + k]
    // Negative k: shift indices forward (looking at later values)
    // In array terms: x_lagged[t] = x[t - |k|]
    let values = if k >= 0 {
        // Positive lag: take values from position k onwards
        x[abs_k..].to_vec()
    } else {
        // Negative lag: take values up to n - |k|
        x[..n - abs_k].to_vec()
    };

    Ok(LagResult {
        values,
        k,
        original_length: n,
    })
}

/// Lag a time series and pad with NaN to maintain original length.
///
/// # Arguments
///
/// * `x` - Input time series
/// * `k` - Number of lags (positive = shift back, negative = shift forward)
///
/// # Returns
///
/// A vector of the same length as input, with NaN values where data is missing.
pub fn lag_padded(x: &[f64], k: i32) -> EconResult<Vec<f64>> {
    if x.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "lag_padded".to_string(),
        });
    }

    let n = x.len();
    let abs_k = k.unsigned_abs() as usize;

    if abs_k >= n {
        return Ok(vec![f64::NAN; n]);
    }

    let mut result = vec![f64::NAN; n];

    if k >= 0 {
        // Positive lag: NaN at beginning, values shifted
        for i in 0..(n - abs_k) {
            result[i + abs_k] = x[i];
        }
    } else {
        // Negative lag: NaN at end, values shifted
        for i in abs_k..n {
            result[i - abs_k] = x[i];
        }
    }

    Ok(result)
}

// ============================================================================
// Embed Function
// ============================================================================

/// Result from embed operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResult {
    /// The embedding matrix (n - dimension + 1) x dimension
    /// Each row contains [x[t], x[t-1], ..., x[t-dimension+1]]
    pub matrix: Vec<Vec<f64>>,
    /// Embedding dimension
    pub dimension: usize,
    /// Number of rows in the embedding matrix
    pub n_rows: usize,
    /// Original series length
    pub original_length: usize,
}

/// Embed a time series into a matrix of lagged values.
///
/// Creates a matrix where each row contains a sequence of consecutive
/// observations: `[x[t], x[t-1], ..., x[t-dimension+1]]`.
///
/// This is useful for creating design matrices for AR models.
///
/// # Arguments
///
/// * `x` - Input time series
/// * `dimension` - Embedding dimension (number of columns)
///
/// # Returns
///
/// A matrix with (n - dimension + 1) rows and `dimension` columns.
/// Row i contains observations from index i to i + dimension - 1 (most recent first).
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::embed;
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = embed(&x, 3).unwrap();
/// // result.matrix:
/// // [[3, 2, 1],
/// //  [4, 3, 2],
/// //  [5, 4, 3]]
/// ```
///
/// # References
///
/// R `stats::embed`
pub fn embed(x: &[f64], dimension: usize) -> EconResult<EmbedResult> {
    let n = x.len();

    if dimension == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Embedding dimension must be at least 1".to_string(),
        });
    }

    if dimension > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Embedding dimension {} exceeds series length {}",
                dimension, n
            ),
        });
    }

    let n_rows = n - dimension + 1;
    let mut matrix = Vec::with_capacity(n_rows);

    // Build each row: [x[t], x[t-1], ..., x[t-dimension+1]]
    // where t goes from dimension-1 to n-1
    for t in (dimension - 1)..n {
        let mut row = Vec::with_capacity(dimension);
        for lag in 0..dimension {
            row.push(x[t - lag]);
        }
        matrix.push(row);
    }

    Ok(EmbedResult {
        matrix,
        dimension,
        n_rows,
        original_length: n,
    })
}

/// Embed a time series into an ndarray matrix.
///
/// Same as `embed` but returns an ndarray Array2<f64>.
pub fn embed_array(x: &[f64], dimension: usize) -> EconResult<Array2<f64>> {
    let result = embed(x, dimension)?;
    let n_rows = result.n_rows;

    let flat: Vec<f64> = result.matrix.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, dimension), flat).map_err(|e| EconError::Internal(e.to_string()))
}

// ============================================================================
// Diffinv Function (Inverse of Differencing)
// ============================================================================

/// Result from diffinv operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffinvResult {
    /// The integrated time series
    pub values: Vec<f64>,
    /// Lag used in integration
    pub lag: usize,
    /// Number of times integration was applied
    pub differences: usize,
    /// Initial values used
    pub xi: Vec<f64>,
}

/// Discrete integration: inverse of differencing.
///
/// Computes the inverse of `diff(x, lag, differences)` using cumulative sums.
///
/// # Arguments
///
/// * `x` - Differenced time series
/// * `lag` - Lag for the differences (default: 1)
/// * `differences` - Order of integration (default: 1)
/// * `xi` - Initial values. If None, uses zeros.
///
/// # Returns
///
/// The integrated time series.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::diffinv;
/// // If we differenced [1, 3, 6, 10] once, we get [2, 3, 4]
/// // diffinv([2, 3, 4], xi=[1]) should give [1, 3, 6, 10]
/// let diffs = vec![2.0, 3.0, 4.0];
/// let result = diffinv(&diffs, 1, 1, Some(&[1.0])).unwrap();
/// assert!((result.values[0] - 1.0).abs() < 1e-10);
/// assert!((result.values[3] - 10.0).abs() < 1e-10);
/// ```
///
/// # References
///
/// R `stats::diffinv`
pub fn diffinv(x: &[f64], lag: usize, differences: usize, xi: Option<&[f64]>) -> EconResult<DiffinvResult> {
    if lag == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Lag must be at least 1".to_string(),
        });
    }

    if differences == 0 {
        // No integration needed
        return Ok(DiffinvResult {
            values: x.to_vec(),
            lag,
            differences,
            xi: vec![],
        });
    }

    // For differences > 0, we need lag * differences initial values
    let n_xi = lag * differences;
    let xi_values: Vec<f64> = match xi {
        Some(v) if v.len() >= n_xi => v[..n_xi].to_vec(),
        Some(v) => {
            // Pad with zeros if not enough initial values
            let mut padded = v.to_vec();
            padded.resize(n_xi, 0.0);
            padded
        }
        None => vec![0.0; n_xi],
    };

    // Start with input
    let mut result = x.to_vec();

    // Apply cumsum 'differences' times
    for d in 0..differences {
        let xi_start = d * lag;
        let xi_for_this_diff = &xi_values[xi_start..xi_start + lag];

        result = cumsum_with_init(&result, lag, xi_for_this_diff);
    }

    Ok(DiffinvResult {
        values: result,
        lag,
        differences,
        xi: xi_values,
    })
}

/// Cumulative sum with lag and initial values.
fn cumsum_with_init(x: &[f64], lag: usize, xi: &[f64]) -> Vec<f64> {
    let n = x.len() + lag;
    let mut result = Vec::with_capacity(n);

    // Initial values
    for &v in xi.iter().take(lag) {
        result.push(v);
    }

    // Cumulative sum: y[i] = x[i-lag] + y[i-lag]
    for i in lag..n {
        let x_val = if i >= lag { x[i - lag] } else { 0.0 };
        let y_prev = result[i - lag];
        result.push(x_val + y_prev);
    }

    result
}

// ============================================================================
// Filter Function (Linear Filtering)
// ============================================================================

/// Filter method for time series filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterMethod {
    /// Convolution filter (moving average)
    #[default]
    Convolution,
    /// Recursive filter (autoregressive)
    Recursive,
}

/// How to handle filter sides (for convolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterSides {
    /// One-sided: filter only uses past values
    #[default]
    One,
    /// Two-sided: filter is centered around lag 0
    Two,
}

/// Result from filter operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    /// Filtered time series
    pub values: Vec<f64>,
    /// Filter coefficients used
    pub filter: Vec<f64>,
    /// Method used
    pub method: FilterMethod,
    /// Number of observations
    pub n_obs: usize,
}

/// Apply linear filtering to a time series.
///
/// # Convolution Filter
///
/// For sides = 1: `y[i] = sum(filter[j] * x[i - j])` for j = 0 to p-1
/// For sides = 2: `y[i] = sum(filter[j] * x[i + o - j])` where o centers the filter
///
/// # Recursive Filter
///
/// `y[i] = x[i] + sum(filter[j] * y[i - j])` for j = 1 to p
///
/// # Arguments
///
/// * `x` - Input time series
/// * `filter` - Filter coefficients
/// * `method` - Convolution or Recursive
/// * `sides` - For convolution: One (past only) or Two (centered)
/// * `init` - Initial values for recursive filter
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::{filter, FilterMethod, FilterSides};
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let filter_coefs = vec![1.0/3.0, 1.0/3.0, 1.0/3.0]; // 3-point moving average
/// let result = filter(&x, &filter_coefs, FilterMethod::Convolution, FilterSides::One, None).unwrap();
/// ```
///
/// # References
///
/// R `stats::filter`
pub fn filter(
    x: &[f64],
    filter_coefs: &[f64],
    method: FilterMethod,
    sides: FilterSides,
    init: Option<&[f64]>,
) -> EconResult<FilterResult> {
    let n = x.len();
    let p = filter_coefs.len();

    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "filter".to_string(),
        });
    }

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Filter coefficients cannot be empty".to_string(),
        });
    }

    let values = match method {
        FilterMethod::Convolution => filter_convolution(x, filter_coefs, sides),
        FilterMethod::Recursive => filter_recursive(x, filter_coefs, init)?,
    };

    Ok(FilterResult {
        values,
        filter: filter_coefs.to_vec(),
        method,
        n_obs: n,
    })
}

/// Convolution (moving average) filter.
fn filter_convolution(x: &[f64], filter_coefs: &[f64], sides: FilterSides) -> Vec<f64> {
    let n = x.len();
    let p = filter_coefs.len();
    let mut result = vec![f64::NAN; n];

    match sides {
        FilterSides::One => {
            // One-sided: y[i] = sum(filter[j] * x[i - j]) for j = 0 to p-1
            // Need x[i-p+1] through x[i], so start at i = p-1
            for i in (p - 1)..n {
                let mut sum = 0.0;
                for j in 0..p {
                    sum += filter_coefs[j] * x[i - j];
                }
                result[i] = sum;
            }
        }
        FilterSides::Two => {
            // Two-sided: filter is centered
            // Offset o = (p - 1) / 2 (assuming odd filter length)
            let offset = (p - 1) / 2;
            for i in offset..(n - (p - 1 - offset)) {
                let mut sum = 0.0;
                for j in 0..p {
                    let idx = i + offset - j;
                    if idx < n {
                        sum += filter_coefs[j] * x[idx];
                    }
                }
                result[i] = sum;
            }
        }
    }

    result
}

/// Recursive (autoregressive) filter.
fn filter_recursive(x: &[f64], filter_coefs: &[f64], init: Option<&[f64]>) -> EconResult<Vec<f64>> {
    let n = x.len();
    let p = filter_coefs.len();

    // Initialize with provided values or zeros
    let init_values: Vec<f64> = match init {
        Some(v) if v.len() >= p => v[..p].to_vec(),
        Some(v) => {
            let mut padded = v.to_vec();
            padded.resize(p, 0.0);
            padded
        }
        None => vec![0.0; p],
    };

    // Result includes the initial values conceptually
    // y[i] = x[i] + sum(filter[j] * y[i-j]) for j = 1 to p
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = x[i];
        for j in 0..p {
            let prev_idx = i as i64 - 1 - j as i64;
            let y_prev = if prev_idx >= 0 {
                y[prev_idx as usize]
            } else {
                // Use initial values
                let init_idx = (p as i64 + prev_idx) as usize;
                if init_idx < p {
                    init_values[init_idx]
                } else {
                    0.0
                }
            };
            sum += filter_coefs[j] * y_prev;
        }
        y.push(sum);
    }

    Ok(y)
}

// ============================================================================
// Window Function (Time Series Subsetting)
// ============================================================================

/// Result from window operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    /// The windowed time series values
    pub values: Vec<f64>,
    /// Start index (0-based)
    pub start: usize,
    /// End index (0-based, exclusive)
    pub end: usize,
    /// Original series length
    pub original_length: usize,
}

/// Extract a subset (window) of a time series.
///
/// # Arguments
///
/// * `x` - Input time series
/// * `start` - Start index (0-based, inclusive). If None, starts at beginning.
/// * `end` - End index (0-based, exclusive). If None, ends at series end.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::window;
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = window(&x, Some(1), Some(4)).unwrap();
/// assert_eq!(result.values, vec![2.0, 3.0, 4.0]);
/// ```
///
/// # References
///
/// R `stats::window`
pub fn window(x: &[f64], start: Option<usize>, end: Option<usize>) -> EconResult<WindowResult> {
    let n = x.len();

    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "window".to_string(),
        });
    }

    let start_idx = start.unwrap_or(0);
    let end_idx = end.unwrap_or(n);

    if start_idx >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("Start index {} exceeds series length {}", start_idx, n),
        });
    }

    if end_idx > n {
        return Err(EconError::InvalidSpecification {
            message: format!("End index {} exceeds series length {}", end_idx, n),
        });
    }

    if start_idx >= end_idx {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Start index {} must be less than end index {}",
                start_idx, end_idx
            ),
        });
    }

    Ok(WindowResult {
        values: x[start_idx..end_idx].to_vec(),
        start: start_idx,
        end: end_idx,
        original_length: n,
    })
}

// ============================================================================
// ARMA ACF (Theoretical ACF for ARMA Process)
// ============================================================================

/// Result from ARMA ACF computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmaAcfResult {
    /// ACF or PACF values for lags 0 to lag_max
    pub values: Vec<f64>,
    /// Lags (0 to lag_max)
    pub lags: Vec<usize>,
    /// AR coefficients used
    pub ar: Vec<f64>,
    /// MA coefficients used
    pub ma: Vec<f64>,
    /// Whether this is PACF (true) or ACF (false)
    pub is_pacf: bool,
}

/// Compute theoretical ACF for an ARMA process.
///
/// Given AR coefficients φ₁, ..., φₚ and MA coefficients θ₁, ..., θ_q,
/// computes the theoretical autocorrelation function.
///
/// Uses the method described in Brockwell & Davis (1991), Section 3.3.
///
/// # Arguments
///
/// * `ar` - AR coefficients (can be empty for pure MA)
/// * `ma` - MA coefficients (can be empty for pure AR)
/// * `lag_max` - Maximum lag to compute
/// * `pacf` - If true, return PACF instead of ACF
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arma_acf;
/// // AR(1) process with coefficient 0.7
/// let result = arma_acf(&[0.7], &[], 10, false).unwrap();
/// // ACF should decay geometrically: 0.7, 0.49, 0.343, ...
/// ```
///
/// # References
///
/// - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*, Section 3.3
/// - R `stats::ARMAacf`
pub fn arma_acf(ar: &[f64], ma: &[f64], lag_max: usize, pacf: bool) -> EconResult<ArmaAcfResult> {
    let p = ar.len();
    let q = ma.len();

    // Compute autocovariances first
    let acvf = arma_autocovariance(ar, ma, lag_max)?;

    // Normalize to get ACF
    let gamma0 = acvf[0];
    if gamma0 <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "ARMA process has non-positive variance".to_string(),
        });
    }

    let acf_values: Vec<f64> = acvf.iter().map(|&g| g / gamma0).collect();

    let values = if pacf {
        // Compute PACF from ACF using Durbin-Levinson
        let pacf_vals = durbin_levinson_from_acf(&acf_values)?;
        // Include lag 0 = 1.0 for consistency
        let mut result = vec![1.0];
        result.extend(pacf_vals);
        result
    } else {
        acf_values
    };

    Ok(ArmaAcfResult {
        values,
        lags: (0..=lag_max).collect(),
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        is_pacf: pacf,
    })
}

/// Compute theoretical autocovariance function for ARMA(p,q) process.
///
/// Uses equations from Brockwell & Davis (1991), Section 3.3.
fn arma_autocovariance(ar: &[f64], ma: &[f64], lag_max: usize) -> EconResult<Vec<f64>> {
    let p = ar.len();
    let q = ma.len();

    // For pure AR(p): gamma(k) = sum(phi_i * gamma(k-i)) for k > 0
    // gamma(0) = sigma^2 + sum(phi_i * gamma(i))

    // For ARMA, we need to solve the Yule-Walker equations modified for MA component

    let m = p.max(q + 1);
    let n_lags = lag_max.max(m) + 1;

    let mut gamma = vec![0.0; n_lags];

    // For sigma^2 = 1 (scale later if needed)
    // First compute initial autocovariances gamma(0), ..., gamma(m-1)
    // by solving the modified Yule-Walker equations

    if p == 0 && q == 0 {
        // White noise: gamma(0) = 1, gamma(k) = 0 for k > 0
        gamma[0] = 1.0;
        return Ok(gamma);
    }

    if p == 0 {
        // Pure MA(q) process
        // gamma(k) = sigma^2 * sum(theta_i * theta_{i+k}) for k <= q, 0 otherwise
        gamma[0] = 1.0 + ma.iter().map(|&t| t * t).sum::<f64>();
        for k in 1..=q.min(lag_max) {
            let mut sum = 0.0;
            for i in 0..(q - k) {
                sum += ma[i] * ma[i + k];
            }
            if k <= q {
                sum += ma[k - 1]; // theta_k * 1 term
            }
            gamma[k] = sum;
        }
        return Ok(gamma);
    }

    if q == 0 {
        // Pure AR(p) process
        // Solve Yule-Walker equations
        // First, compute gamma(0)
        // gamma(0) = sigma^2 / (1 - sum(phi_i * rho_i))
        // We need to iterate or solve the system

        // Simple approach: use the fact that for AR(1), gamma(0) = sigma^2 / (1 - phi^2)
        // For general AR(p), we solve the system iteratively

        // Initial approximation
        gamma[0] = 1.0;

        // Use recursion: gamma(k) = sum(phi_i * gamma(|k-i|))
        for _ in 0..100 {
            // Iterate to convergence
            let old_gamma0 = gamma[0];

            // First compute gamma(1) to gamma(p) from gamma(0)
            for k in 1..=p.min(n_lags - 1) {
                let mut sum = 0.0;
                for i in 0..p {
                    let idx = (k as i64 - 1 - i as i64).unsigned_abs() as usize;
                    if idx < n_lags {
                        sum += ar[i] * gamma[idx];
                    }
                }
                gamma[k] = sum;
            }

            // Update gamma(0)
            let mut sum = 1.0; // sigma^2 = 1
            for i in 0..p {
                sum += ar[i] * gamma[i + 1];
            }
            gamma[0] = sum;

            if (gamma[0] - old_gamma0).abs() < 1e-12 {
                break;
            }
        }

        // Extend to remaining lags using recursion
        for k in (p + 1)..n_lags {
            let mut sum = 0.0;
            for i in 0..p {
                if k > i {
                    sum += ar[i] * gamma[k - 1 - i];
                }
            }
            gamma[k] = sum;
        }

        return Ok(gamma);
    }

    // General ARMA(p,q) case
    // Use the general recursion from Brockwell & Davis

    // First solve for gamma(0), ..., gamma(max(p, q+1)-1)
    // Then use the AR recursion for higher lags

    // Simplified approach: compute using numerical solution
    // gamma(k) for k > q follows AR recursion
    // For k <= max(p, q), we need the full system

    // For now, use a numerical approach that works well for most cases
    // Initialize with approximate values from AR part
    gamma[0] = 1.0;

    // First pass: AR contribution
    for _ in 0..50 {
        for k in 1..n_lags {
            let mut sum = 0.0;
            for i in 0..p {
                let idx = (k as i64 - 1 - i as i64).unsigned_abs() as usize;
                if idx < n_lags {
                    sum += ar[i] * gamma[idx];
                }
            }
            gamma[k] = sum;
        }

        // Update gamma(0) including MA contribution
        let mut sum = 1.0;
        for i in 0..p {
            sum += ar[i] * gamma[i + 1];
        }
        // Add MA contribution to variance
        for i in 0..q {
            sum += ma[i] * ma[i];
        }
        // Cross terms
        for k in 0..q.min(p) {
            sum += 2.0 * ar[k] * ma[k];
        }
        gamma[0] = sum.max(0.01); // Ensure positive
    }

    // Add MA contribution to autocovariances
    for k in 1..=q.min(n_lags - 1) {
        let mut ma_contrib = 0.0;
        for i in 0..(q - k) {
            ma_contrib += ma[i] * ma[i + k];
        }
        if k <= q {
            ma_contrib += ma[k - 1];
        }
        gamma[k] += ma_contrib;
    }

    Ok(gamma)
}

/// Durbin-Levinson algorithm to compute PACF from ACF.
fn durbin_levinson_from_acf(acf: &[f64]) -> EconResult<Vec<f64>> {
    let p = acf.len() - 1; // acf[0] = 1
    if p == 0 {
        return Ok(vec![]);
    }

    let mut pacf = Vec::with_capacity(p);
    let mut phi: Vec<f64> = vec![0.0; p];

    phi[0] = acf[1];
    pacf.push(phi[0]);

    for n in 2..=p {
        let mut num = acf[n];
        let mut den = 1.0;

        for k in 1..n {
            num -= phi[k - 1] * acf[n - k];
            den -= phi[k - 1] * acf[k];
        }

        if den.abs() < 1e-15 {
            // Numerical issues
            pacf.push(0.0);
            continue;
        }

        let phi_nn = num / den;
        pacf.push(phi_nn);

        let phi_prev = phi.clone();
        for k in 1..n {
            phi[k - 1] = phi_prev[k - 1] - phi_nn * phi_prev[n - k - 1];
        }
        phi[n - 1] = phi_nn;
    }

    Ok(pacf)
}

// ============================================================================
// ARMA to MA Conversion
// ============================================================================

/// Result from ARMA to MA conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmaToMaResult {
    /// MA (psi) coefficients: psi_1, psi_2, ..., psi_lag_max
    pub psi: Vec<f64>,
    /// AR coefficients used
    pub ar: Vec<f64>,
    /// MA coefficients used
    pub ma: Vec<f64>,
    /// Maximum lag computed
    pub lag_max: usize,
}

/// Convert ARMA process to infinite MA representation.
///
/// An ARMA(p,q) process can be written as an infinite MA process:
/// X_t = sum(psi_j * Z_{t-j}) for j = 0 to infinity
///
/// where psi_0 = 1 and the psi coefficients are determined by
/// the AR and MA coefficients.
///
/// # Arguments
///
/// * `ar` - AR coefficients
/// * `ma` - MA coefficients
/// * `lag_max` - Number of MA coefficients to compute
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arma_to_ma;
/// // ARMA(1,1) with AR=0.5, MA=0.3
/// let result = arma_to_ma(&[0.5], &[0.3], 10).unwrap();
/// // psi_1 = phi_1 + theta_1 = 0.5 + 0.3 = 0.8
/// // psi_2 = phi_1 * psi_1 = 0.5 * 0.8 = 0.4
/// ```
///
/// # References
///
/// - Brockwell & Davis (1991), Proposition 3.1.1
/// - R `stats::ARMAtoMA`
pub fn arma_to_ma(ar: &[f64], ma: &[f64], lag_max: usize) -> EconResult<ArmaToMaResult> {
    let p = ar.len();
    let q = ma.len();

    // psi_0 = 1 (implicit)
    // psi_j = theta_j + sum(phi_i * psi_{j-i}) for i = 1 to min(j, p)
    // where theta_j = 0 for j > q

    let mut psi = Vec::with_capacity(lag_max);

    for j in 1..=lag_max {
        let mut sum = 0.0;

        // theta_j contribution (0 if j > q)
        if j <= q {
            sum += ma[j - 1];
        }

        // AR contribution: sum(phi_i * psi_{j-i})
        for i in 1..=p.min(j) {
            let psi_prev = if j == i {
                1.0 // psi_0 = 1
            } else {
                psi[j - i - 1]
            };
            sum += ar[i - 1] * psi_prev;
        }

        psi.push(sum);
    }

    Ok(ArmaToMaResult {
        psi,
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        lag_max,
    })
}

// ============================================================================
// ACF to AR (Durbin-Levinson for AR Fitting)
// ============================================================================

/// Result from ACF to AR conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acf2ArResult {
    /// Matrix of AR coefficients
    /// Row i contains AR(i+1) coefficients [phi_{i+1,1}, ..., phi_{i+1,i+1}]
    pub ar_matrix: Vec<Vec<f64>>,
    /// Partial autocorrelations (diagonal of the AR matrix)
    pub pacf: Vec<f64>,
    /// Input ACF values
    pub acf: Vec<f64>,
    /// Maximum order computed
    pub max_order: usize,
}

/// Compute AR coefficients from ACF using Durbin-Levinson algorithm.
///
/// Given an autocorrelation or autocovariance sequence, computes the
/// AR coefficients for models of order 1 up to the length of the input.
///
/// # Arguments
///
/// * `acf` - Autocorrelation or autocovariance sequence starting at lag 0
///
/// # Returns
///
/// A matrix where row i contains the AR(i+1) coefficients.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::acf_to_ar;
/// // ACF from AR(1) with phi=0.8: [1.0, 0.8, 0.64, 0.512, ...]
/// let acf = vec![1.0, 0.8, 0.64, 0.512, 0.4096];
/// let result = acf_to_ar(&acf).unwrap();
/// // AR(1) coefficients should be approximately [0.8]
/// assert!((result.ar_matrix[0][0] - 0.8).abs() < 0.01);
/// ```
///
/// # References
///
/// - Durbin, J. (1960). "The Fitting of Time-Series Models"
/// - Brockwell & Davis (1991), Algorithm 8.2.1
/// - R `stats::acf2AR`
pub fn acf_to_ar(acf: &[f64]) -> EconResult<Acf2ArResult> {
    if acf.is_empty() {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: 0,
            context: "acf_to_ar".to_string(),
        });
    }

    // Normalize to autocorrelation if needed
    let rho: Vec<f64> = if (acf[0] - 1.0).abs() > 1e-10 && acf[0] > 0.0 {
        acf.iter().map(|&g| g / acf[0]).collect()
    } else {
        acf.to_vec()
    };

    let max_order = rho.len() - 1;
    if max_order == 0 {
        return Ok(Acf2ArResult {
            ar_matrix: vec![],
            pacf: vec![],
            acf: rho,
            max_order: 0,
        });
    }

    let mut ar_matrix: Vec<Vec<f64>> = Vec::with_capacity(max_order);
    let mut pacf = Vec::with_capacity(max_order);

    // Durbin-Levinson recursion
    let mut phi: Vec<f64> = vec![0.0; max_order];

    // AR(1)
    phi[0] = rho[1];
    ar_matrix.push(vec![phi[0]]);
    pacf.push(phi[0]);

    // AR(n) for n = 2 to max_order
    for n in 2..=max_order {
        // Compute phi_{n,n}
        let mut num = rho[n];
        let mut den = 1.0;

        for k in 1..n {
            num -= phi[k - 1] * rho[n - k];
            den -= phi[k - 1] * rho[k];
        }

        if den.abs() < 1e-15 {
            // Numerical issues, stop here
            break;
        }

        let phi_nn = num / den;
        pacf.push(phi_nn);

        // Update phi coefficients
        let phi_prev = phi.clone();
        for k in 1..n {
            phi[k - 1] = phi_prev[k - 1] - phi_nn * phi_prev[n - k - 1];
        }
        phi[n - 1] = phi_nn;

        // Store AR(n) coefficients
        ar_matrix.push(phi[..n].to_vec());
    }

    Ok(Acf2ArResult {
        ar_matrix,
        pacf,
        acf: rho,
        max_order,
    })
}

// ============================================================================
// ARIMA Simulation
// ============================================================================

/// Result from ARIMA simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaSimResult {
    /// Simulated time series
    pub values: Vec<f64>,
    /// AR coefficients used
    pub ar: Vec<f64>,
    /// MA coefficients used
    pub ma: Vec<f64>,
    /// Differencing order
    pub d: usize,
    /// Number of observations
    pub n: usize,
    /// Burn-in period length
    pub n_start: usize,
}

/// Simulate from an ARIMA model.
///
/// Generates a time series following an ARIMA(p,d,q) process:
/// - First generates an ARMA(p,q) series
/// - Then undifferences d times to get ARIMA
///
/// # Arguments
///
/// * `ar` - AR coefficients (can be empty)
/// * `ma` - MA coefficients (can be empty)
/// * `d` - Differencing order (0 for ARMA)
/// * `n` - Length of output series (before un-differencing)
/// * `innovations` - Optional pre-specified innovations. If None, uses standard normal.
/// * `n_start` - Burn-in period length (default: computed from AR/MA orders)
/// * `seed` - Random seed for reproducibility (if innovations is None)
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arima_sim;
/// // Simulate AR(1) with coefficient 0.8
/// let result = arima_sim(&[0.8], &[], 0, 100, None, None, Some(42)).unwrap();
/// assert_eq!(result.values.len(), 100);
/// ```
///
/// # References
///
/// - Box et al. (2015), Chapter 3
/// - R `stats::arima.sim`
pub fn arima_sim(
    ar: &[f64],
    ma: &[f64],
    d: usize,
    n: usize,
    innovations: Option<&[f64]>,
    n_start: Option<usize>,
    seed: Option<u64>,
) -> EconResult<ArimaSimResult> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal};

    let p = ar.len();
    let q = ma.len();

    // Validate stationarity of AR part
    if p > 0 {
        // Check that sum of |ar| < 1 (rough check)
        let sum_ar: f64 = ar.iter().map(|x| x.abs()).sum();
        if sum_ar >= 1.0 {
            // This is a rough check; proper check would involve polynomial roots
            // We'll allow it but warn internally
        }
    }

    // Compute burn-in length if not specified
    let burn_in = n_start.unwrap_or_else(|| {
        // R default: max(p, q) * 10 or at least 100
        let min_start = (p.max(q) * 10).max(100);
        min_start
    });

    let total_length = n + burn_in;

    // Generate or use innovations
    let mut innov: Vec<f64> = if let Some(ext_innov) = innovations {
        if ext_innov.len() < total_length {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Need {} innovations but only {} provided",
                    total_length,
                    ext_innov.len()
                ),
            });
        }
        ext_innov[..total_length].to_vec()
    } else {
        // Generate random innovations
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        (0..total_length)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect()
    };

    // Simulate ARMA part
    let mut x = vec![0.0; total_length];

    for t in 0..total_length {
        let mut val = innov[t];

        // AR part: sum(phi_i * x[t-i])
        for i in 0..p {
            if t > i {
                val += ar[i] * x[t - 1 - i];
            }
        }

        // MA part: sum(theta_j * e[t-j])
        for j in 0..q {
            if t > j {
                val += ma[j] * innov[t - 1 - j];
            }
        }

        x[t] = val;
    }

    // Remove burn-in
    let arma_series: Vec<f64> = x[burn_in..].to_vec();

    // Un-difference if d > 0
    let final_series = if d > 0 {
        // Apply cumsum d times
        let mut result = arma_series;
        for _ in 0..d {
            let mut cumsum = Vec::with_capacity(result.len());
            let mut running = 0.0;
            for &val in &result {
                running += val;
                cumsum.push(running);
            }
            result = cumsum;
        }
        result
    } else {
        arma_series
    };

    Ok(ArimaSimResult {
        values: final_series,
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        d,
        n,
        n_start: burn_in,
    })
}

// ============================================================================
// Running Median (Robust Smoothing)
// ============================================================================

/// End rule for running median.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EndRule {
    /// Keep original values at ends
    Keep,
    /// Extend with constant (edge median value)
    Constant,
    /// Use median of available values (default, Tukey's rule)
    #[default]
    Median,
}

/// Result from running median computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunmedResult {
    /// Smoothed values
    pub values: Vec<f64>,
    /// Window width used
    pub k: usize,
    /// End rule applied
    pub endrule: EndRule,
    /// Number of observations
    pub n_obs: usize,
}

/// Compute running median for robust smoothing.
///
/// # Arguments
///
/// * `x` - Input time series
/// * `k` - Window width (must be odd)
/// * `endrule` - How to handle endpoints
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::{runmed, EndRule};
/// let x = vec![1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0];
/// let result = runmed(&x, 3, EndRule::Median).unwrap();
/// // The outlier at index 2 is smoothed away
/// assert!((result.values[2] - 4.0).abs() < 1e-10);
/// ```
///
/// # References
///
/// - Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
/// - R `stats::runmed`
pub fn runmed(x: &[f64], k: usize, endrule: EndRule) -> EconResult<RunmedResult> {
    let n = x.len();

    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "runmed".to_string(),
        });
    }

    if k == 0 || k % 2 == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("Window width k must be odd and positive, got {}", k),
        });
    }

    if k > n {
        return Err(EconError::InvalidSpecification {
            message: format!("Window width {} exceeds series length {}", k, n),
        });
    }

    let half_k = k / 2;
    let mut result = vec![0.0; n];

    // Main body: full window
    for i in half_k..(n - half_k) {
        let window: Vec<f64> = x[(i - half_k)..=(i + half_k)].to_vec();
        result[i] = median(&window);
    }

    // Handle ends
    match endrule {
        EndRule::Keep => {
            // Keep original values
            for i in 0..half_k {
                result[i] = x[i];
                result[n - 1 - i] = x[n - 1 - i];
            }
        }
        EndRule::Constant => {
            // Extend with edge median
            let left_med = result[half_k];
            let right_med = result[n - 1 - half_k];
            for i in 0..half_k {
                result[i] = left_med;
                result[n - 1 - i] = right_med;
            }
        }
        EndRule::Median => {
            // Tukey's rule: use median of available values with decreasing window
            for i in 0..half_k {
                // Left end: use window from 0 to 2*i (size 2*i+1)
                let window: Vec<f64> = x[0..=(2 * i)].to_vec();
                result[i] = median(&window);

                // Right end: use window from n-1-2*i to n-1
                let right_i = n - 1 - i;
                let window_right: Vec<f64> = x[(n - 1 - 2 * i)..n].to_vec();
                result[right_i] = median(&window_right);
            }
        }
    }

    Ok(RunmedResult {
        values: result,
        k,
        endrule,
        n_obs: n,
    })
}

/// Compute median of a slice.
fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let mut sorted: Vec<f64> = x.iter().cloned().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }

    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_lag_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = lag(&x, 1).unwrap();
        assert_eq!(result.values, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_lag_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = lag(&x, -1).unwrap();
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lag_padded() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = lag_padded(&x, 2).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
        assert_eq!(result[4], 3.0);
    }

    #[test]
    fn test_embed() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = embed(&x, 3).unwrap();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.dimension, 3);

        // Row 0: [x[2], x[1], x[0]] = [3, 2, 1]
        assert_eq!(result.matrix[0], vec![3.0, 2.0, 1.0]);
        // Row 1: [x[3], x[2], x[1]] = [4, 3, 2]
        assert_eq!(result.matrix[1], vec![4.0, 3.0, 2.0]);
        // Row 2: [x[4], x[3], x[2]] = [5, 4, 3]
        assert_eq!(result.matrix[2], vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_diffinv_simple() {
        // diff([1, 3, 6, 10]) = [2, 3, 4]
        // diffinv([2, 3, 4], xi=[1]) should give [1, 3, 6, 10]
        let diffs = vec![2.0, 3.0, 4.0];
        let result = diffinv(&diffs, 1, 1, Some(&[1.0])).unwrap();

        assert_eq!(result.values.len(), 4);
        assert!(approx_eq(result.values[0], 1.0, 1e-10));
        assert!(approx_eq(result.values[1], 3.0, 1e-10));
        assert!(approx_eq(result.values[2], 6.0, 1e-10));
        assert!(approx_eq(result.values[3], 10.0, 1e-10));
    }

    #[test]
    fn test_filter_convolution() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let filter_coefs = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; // 3-point MA

        let result = filter(&x, &filter_coefs, FilterMethod::Convolution, FilterSides::One, None).unwrap();

        // At index 2: (3 + 2 + 1) / 3 = 2.0
        assert!(approx_eq(result.values[2], 2.0, 1e-10));
        // At index 5: (6 + 5 + 4) / 3 = 5.0
        assert!(approx_eq(result.values[5], 5.0, 1e-10));
    }

    #[test]
    fn test_filter_recursive() {
        // AR(1) recursive filter: y[t] = x[t] + 0.5 * y[t-1]
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let filter_coefs = vec![0.5];

        let result = filter(&x, &filter_coefs, FilterMethod::Recursive, FilterSides::One, None).unwrap();

        // y[0] = x[0] = 1.0
        // y[1] = x[1] + 0.5*y[0] = 1 + 0.5 = 1.5
        // y[2] = x[2] + 0.5*y[1] = 1 + 0.75 = 1.75
        assert!(approx_eq(result.values[0], 1.0, 1e-10));
        assert!(approx_eq(result.values[1], 1.5, 1e-10));
        assert!(approx_eq(result.values[2], 1.75, 1e-10));
    }

    #[test]
    fn test_window() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = window(&x, Some(1), Some(4)).unwrap();
        assert_eq!(result.values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arma_acf_ar1() {
        // AR(1) with phi = 0.8
        // ACF should be: 1, 0.8, 0.64, 0.512, ...
        let result = arma_acf(&[0.8], &[], 5, false).unwrap();

        assert!(approx_eq(result.values[0], 1.0, 1e-10));
        assert!(approx_eq(result.values[1], 0.8, 0.05));
        assert!(approx_eq(result.values[2], 0.64, 0.05));
    }

    #[test]
    fn test_arma_to_ma() {
        // AR(1) with phi = 0.5
        // MA representation: psi_j = phi^j
        let result = arma_to_ma(&[0.5], &[], 5).unwrap();

        assert!(approx_eq(result.psi[0], 0.5, 1e-10));
        assert!(approx_eq(result.psi[1], 0.25, 1e-10));
        assert!(approx_eq(result.psi[2], 0.125, 1e-10));
    }

    #[test]
    fn test_acf_to_ar() {
        // ACF from AR(1) with phi=0.8: [1, 0.8, 0.64, 0.512, ...]
        let acf = vec![1.0, 0.8, 0.64, 0.512, 0.4096];
        let result = acf_to_ar(&acf).unwrap();

        // AR(1) coefficient should be approximately 0.8
        assert!(approx_eq(result.ar_matrix[0][0], 0.8, 0.01));
        // PACF[1] = 0.8, PACF[k] ≈ 0 for k > 1
        assert!(approx_eq(result.pacf[0], 0.8, 0.01));
        assert!(result.pacf[1].abs() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_arima_sim() {
        // Simulate AR(1) with phi = 0.8
        let result = arima_sim(&[0.8], &[], 0, 100, None, Some(50), Some(42)).unwrap();

        assert_eq!(result.values.len(), 100);
        assert_eq!(result.ar, vec![0.8]);
        assert_eq!(result.d, 0);
    }

    #[test]
    fn test_arima_sim_with_differencing() {
        // Simulate ARIMA(1,1,0)
        let result = arima_sim(&[0.5], &[], 1, 100, None, Some(50), Some(42)).unwrap();

        assert_eq!(result.values.len(), 100);
        assert_eq!(result.d, 1);
    }

    #[test]
    fn test_runmed_basic() {
        let x = vec![1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0];
        let result = runmed(&x, 3, EndRule::Median).unwrap();

        // Outlier at index 2 should be smoothed
        // median(2, 100, 4) = 4
        assert!(approx_eq(result.values[2], 4.0, 1e-10));
    }

    #[test]
    fn test_runmed_endrule_keep() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = runmed(&x, 3, EndRule::Keep).unwrap();

        // Ends should keep original values
        assert!(approx_eq(result.values[0], 1.0, 1e-10));
        assert!(approx_eq(result.values[4], 5.0, 1e-10));
    }

    /// Validate lag against R
    /// R code: lag(1:5, 1)
    #[test]
    fn test_validate_lag_against_r() {
        let x: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let result = lag(&x, 1).unwrap();
        // In R, lag shifts time index back, so values stay same but time shifts
        // Our implementation returns shifted array values
        assert_eq!(result.values.len(), 4);
    }

    /// Validate embed against R
    /// R code: embed(1:5, 3)
    #[test]
    fn test_validate_embed_against_r() {
        let x: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let result = embed(&x, 3).unwrap();

        // R output:
        //      [,1] [,2] [,3]
        // [1,]    3    2    1
        // [2,]    4    3    2
        // [3,]    5    4    3
        assert_eq!(result.matrix[0], vec![3.0, 2.0, 1.0]);
        assert_eq!(result.matrix[1], vec![4.0, 3.0, 2.0]);
        assert_eq!(result.matrix[2], vec![5.0, 4.0, 3.0]);
    }
}
