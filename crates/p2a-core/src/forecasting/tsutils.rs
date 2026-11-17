//! Time Series Utility Functions
//!
//! Provides utility functions for time series manipulation including:
//! - `lag`: R's `stats::lag` values (x[t + k]); `lag_padded`: the conventional lag x[t - k]
//! - `embed`: Create lag embedding matrix for AR models
//! - `diffinv`: Inverse of differencing (discrete integration)
//! - `filter`: Linear filtering (convolution and recursive)
//! - `window`: Extract time series subsets
//! - `arima_sim`: Simulate from an ARIMA model
//! - `arma_acf`: Compute theoretical ACF for ARMA process
//! - `arma_to_ma`: Convert ARMA to infinite MA representation
//! - `acf_to_ar`: Compute AR coefficients from ACF (Durbin-Levinson)
//! - `runmed`: Running median smoothing (R's `smoothEnds` for the median end rule)
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
//! For ARMA(p,q) process, the autocovariances γ(h) satisfy
//! γ(h) − Σᵢ₌₁ᵖ φᵢγ(|h−i|) = σ² Σⱼ₌ₕ^q θⱼψⱼ₋ₕ  for h ≥ 0 (θ₀ = ψ₀ = 1),
//! which R's `ARMAacf` solves for h = 0..max(p, q + 1) and extends by the AR recursion.
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

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource::{
    check_alloc, checked_len, checkpoint, checkpoint_stride, try_vec, try_with_capacity,
};

fn size_overflow(what: &str) -> EconError {
    EconError::ResourceLimit(format!("{what} is too large: the requested size overflows"))
}

// ============================================================================
// Lag Function
// ============================================================================

/// Result from lag operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagResult {
    /// The lagged time series values
    pub values: Vec<f64>,
    /// The k of R's `stats::lag(x, k)`: element t of `values` is x[t + k] (k > 0 drops the first k
    /// values, k < 0 the last |k|)
    pub k: i32,
    /// Original series length
    pub original_length: usize,
}

/// The values of R's `stats::lag(x, k)` on the shifted time base.
///
/// R's `lag` shifts the time index back by k and keeps the values; read against the original
/// index, element t of the result is x[t + k] (a lead in array terms), and the result is |k|
/// shorter: k > 0 drops the first k values, k < 0 the last |k|. For the conventional lag
/// x[t - k] at full length with NaN padding (`dplyr::lag`), use [`lag_padded`].
///
/// # Arguments
///
/// * `x` - Input time series
/// * `k` - R's k; |k| must be below the series length
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::lag;
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = lag(&x, 1).unwrap();
/// assert_eq!(result.values, vec![2.0, 3.0, 4.0, 5.0]); // x[t + 1]
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

/// The conventional lag x[t - k] at full length, NaN where it falls outside the series
/// (`dplyr::lag(x, k)`; a negative k is the lead x[t + |k|], `dplyr::lead`).
///
/// # Arguments
///
/// * `x` - Input time series
/// * `k` - Positive k lags (NaN at the start), negative k leads (NaN at the end); |k| >= n gives
///   all NaN
///
/// # Returns
///
/// A vector of the same length as input, with NaN values where data is missing.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::lag_padded;
/// let lagged = lag_padded(&[1.0, 2.0, 3.0], 1).unwrap();
/// assert!(lagged[0].is_nan());
/// assert_eq!(&lagged[1..], &[1.0, 2.0]);
/// ```
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
    let (n_rows, cells) = embed_shape(n, dimension)?;
    check_alloc::<f64>(cells, EMBED_WHAT)?;
    let mut matrix = try_with_capacity(n_rows, EMBED_WHAT)?;

    // Each row costs O(dimension).
    let stride = checkpoint_stride(dimension);

    // Build each row: [x[t], x[t-1], ..., x[t-dimension+1]]
    // where t goes from dimension-1 to n-1
    for (i, t) in ((dimension - 1)..n).enumerate() {
        if i % stride == 0 {
            checkpoint()?;
        }
        let mut row = try_with_capacity(dimension, EMBED_WHAT)?;
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

const EMBED_WHAT: &str = "the embedding matrix (rows × dimension)";

/// Validate an embedding of a length-`n` series; returns its row count
/// `n - dimension + 1` and its cell count `rows × dimension`. The caller
/// admits the cells against the allocation ceiling.
fn embed_shape(n: usize, dimension: usize) -> EconResult<(usize, usize)> {
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
    let cells = checked_len(&[n_rows, dimension], EMBED_WHAT)?;
    Ok((n_rows, cells))
}

/// Embed a time series into an ndarray matrix.
///
/// Same as `embed` but returns an ndarray Array2<f64>.
pub fn embed_array(x: &[f64], dimension: usize) -> EconResult<Array2<f64>> {
    let n = x.len();
    let (n_rows, cells) = embed_shape(n, dimension)?;

    // Row-major, row i = [x[t], x[t-1], ..., x[t-dimension+1]] with
    // t = i + dimension - 1, exactly as `embed` builds it.
    let mut flat = try_with_capacity(cells, EMBED_WHAT)?;
    // Each row costs O(dimension).
    let stride = checkpoint_stride(dimension);
    for (i, t) in ((dimension - 1)..n).enumerate() {
        if i % stride == 0 {
            checkpoint()?;
        }
        flat.extend((0..dimension).map(|lag| x[t - lag]));
    }
    Array2::from_shape_vec((n_rows, dimension), flat)
        .map_err(|e| EconError::Internal(e.to_string()))
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

/// Discrete integration: inverse of differencing (R's `diffinv`).
///
/// Returns the series y of length `x.len() + lag * differences` whose `differences`-fold lag-`lag`
/// difference is `x` and whose first `lag * differences` values are `xi`.
///
/// # Algorithm
///
/// R's recursion `diffinv(x, lag, d, xi) = diffinv(diffinv(x, lag, d - 1, diff(xi, lag)), lag, 1,
/// xi[1:lag])`, unrolled: with D^k the k-th lag-`lag` difference of `xi`, the series is integrated
/// for k = d - 1 down to 0 with the first `lag` values of D^k as start values.
///
/// # Arguments
///
/// * `x` - Differenced time series
/// * `lag` - Lag for the differences (at least 1)
/// * `differences` - Order of integration (0 returns `x`)
/// * `xi` - The first `lag * differences` values of the undifferenced series. `None` means zeros.
///
/// # Errors
///
/// `InvalidSpecification` for `lag = 0` or an `xi` whose length is not `lag * differences`
/// (R: "'xi' does not have the right length"); `ResourceLimit` for an output too large to
/// allocate; `Cancelled` when the budget stops (checked once per level).
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
/// // Twice: the second differences of [1, 3, 7, 14, 25] are [2, 3, 4]
/// let twice = diffinv(&diffs, 1, 2, Some(&[1.0, 3.0])).unwrap();
/// assert_eq!(twice.values, vec![1.0, 3.0, 7.0, 14.0, 25.0]);
/// ```
///
/// # References
///
/// R `stats::diffinv`
pub fn diffinv(
    x: &[f64],
    lag: usize,
    differences: usize,
    xi: Option<&[f64]>,
) -> EconResult<DiffinvResult> {
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

    const WHAT: &str = "the integrated series (length + lag × differences)";

    // lag * differences start values
    let n_xi = lag
        .checked_mul(differences)
        .ok_or_else(|| size_overflow(WHAT))?;
    let final_len = x
        .len()
        .checked_add(n_xi)
        .ok_or_else(|| size_overflow(WHAT))?;
    // Admit the output before any work, so an oversized request fails at once.
    check_alloc::<f64>(final_len, WHAT)?;

    let xi_values = match xi {
        None => try_vec(n_xi, 0.0, WHAT)?,
        Some(v) if v.len() == n_xi => {
            let mut values = try_with_capacity(n_xi, WHAT)?;
            values.extend_from_slice(v);
            values
        }
        Some(v) => {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "xi must hold lag * differences = {n_xi} values (the first values of the \
                     undifferenced series), got {}",
                    v.len()
                ),
            });
        }
    };

    // heads[k * lag..(k + 1) * lag]: the first lag values of D^k, the k-th lag difference of xi.
    let mut heads = try_vec(n_xi, 0.0, WHAT)?;
    let mut level = try_with_capacity(n_xi, WHAT)?;
    level.extend_from_slice(&xi_values);
    for k in 0..differences {
        // Each level costs O(lag (d - k)).
        checkpoint()?;
        heads[k * lag..(k + 1) * lag].copy_from_slice(&level[..lag]);
        let len = level.len();
        for i in 0..len - lag {
            level[i] = level[i + lag] - level[i];
        }
        level.truncate(len - lag);
    }

    // Integrate from the innermost level out.
    let mut result = x.to_vec();
    for k in (0..differences).rev() {
        checkpoint()?;
        result = cumsum_with_init(&result, lag, &heads[k * lag..(k + 1) * lag])?;
    }

    Ok(DiffinvResult {
        values: result,
        lag,
        differences,
        xi: xi_values,
    })
}

/// Cumulative sum with lag and initial values.
fn cumsum_with_init(x: &[f64], lag: usize, xi: &[f64]) -> EconResult<Vec<f64>> {
    const WHAT: &str = "the integrated series (length + lag × differences)";
    let n = x
        .len()
        .checked_add(lag)
        .ok_or_else(|| size_overflow(WHAT))?;
    let mut result = try_with_capacity(n, WHAT)?;

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

    Ok(result)
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
/// For sides = 2: `y[i] = sum(filter[j] * x[i + p/2 - j])`, so the window is
/// `x[i + p/2 - p + 1 ..= i + p/2]` (R's `nshift = length(filter) %/% 2`). Positions whose window
/// leaves the series are NaN.
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
/// * `init` - Initial values for the recursive filter: the p values before `x[0]`, oldest first
///   (R's `init` is newest first)
///
/// # Errors
///
/// `InvalidSpecification` for empty or missing (NaN) coefficients, as R's "missing values in
/// 'filter'", and for a convolution filter longer than the series, as R's "'filter' is longer
/// than time series"; `Cancelled` when the budget stops a convolution (checked per block of
/// outputs).
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

    if filter_coefs.iter().any(|v| v.is_nan()) {
        return Err(EconError::InvalidSpecification {
            message:
                "filter coefficients contain missing values (NaN); R: missing values in 'filter'"
                    .to_string(),
        });
    }

    if method == FilterMethod::Convolution && p > n {
        return Err(EconError::InvalidSpecification {
            message: format!("'filter' (length {p}) is longer than the time series (length {n})"),
        });
    }

    let values = match method {
        FilterMethod::Convolution => filter_convolution(x, filter_coefs, sides)?,
        FilterMethod::Recursive => filter_recursive(x, filter_coefs, init)?,
    };

    Ok(FilterResult {
        values,
        filter: filter_coefs.to_vec(),
        method,
        n_obs: n,
    })
}

/// Convolution (moving average) filter, as R's C `cfilter` without `circular`.
///
/// `result[i]` multiplies `f[0]` with `x[i + nshift]`, and is NaN where the window leaves the
/// series. The caller has refused a filter longer than the series.
fn filter_convolution(x: &[f64], filter_coefs: &[f64], sides: FilterSides) -> EconResult<Vec<f64>> {
    let n = x.len();
    let p = filter_coefs.len();
    let mut result = vec![f64::NAN; n];
    let nshift = match sides {
        FilterSides::Two => p / 2,
        FilterSides::One => 0,
    };

    // p <= n (filter refuses a longer filter) and nshift <= p - 1 for p >= 2, so the window fits
    // from first to last; outside it the value stays NA.
    let first = p.saturating_sub(1 + nshift);
    let last = n.saturating_sub(nshift);
    // Each output costs O(p), with p up to n. Blocks, because a remainder per output is a measurable
    // share of a short filter's window.
    let stride = checkpoint_stride(p);
    for block in (first..last).step_by(stride) {
        checkpoint()?;
        for i in block..block.saturating_add(stride).min(last) {
            let top = i + nshift;
            let mut sum = 0.0;
            for (&fj, &xv) in filter_coefs.iter().zip(x[..=top].iter().rev()) {
                sum += fj * xv;
            }
            result[i] = sum; // a NaN in the window propagates, where R gives NA
        }
    }

    Ok(result)
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

/// Compute the theoretical ACF or PACF of an ARMA process (R's `ARMAacf`).
///
/// Given AR coefficients φ₁, ..., φₚ and MA coefficients θ₁, ..., θ_q of
/// X_t = Σ φᵢ X_{t-i} + Z_t + Σ θⱼ Z_{t-j}, returns the autocorrelations ρ(0..lag_max).
///
/// # Algorithm
///
/// As R's `ARMAacf`: with r = max(p, q + 1), the AR part padded with zeros to r terms and σ² = 1, the
/// autocovariances γ(0..r) solve the r + 1 equations
/// γ(k) − Σᵢ₌₁ʳ φᵢ γ(|k − i|) = Σⱼ₌ₖ^q θⱼ ψⱼ₋ₖ (k = 0..r; θ₀ = ψ₀ = 1, ψ the MA(∞) weights),
/// solved by Gaussian elimination with partial pivoting; ρ(k) = γ(k) / γ(0), extended by
/// ρ(k) = Σ φᵢ ρ(k − i) beyond r. AR(1) gives φᵏ and a pure MA(q) the closed form
/// Σⱼ θⱼ θⱼ₊ₖ / Σⱼ θⱼ². The PACF is the Durbin-Levinson recursion on that ACF.
///
/// # Arguments
///
/// * `ar` - AR coefficients (can be empty for pure MA)
/// * `ma` - MA coefficients (can be empty for pure AR)
/// * `lag_max` - Maximum lag; the result always has `lag_max + 1` values (lags 0..lag_max; for
///   the PACF lag 0 is 1.0). R returns `q + 1` values for a pure MA with `lag.max < q`, and two
///   values for an AR model with `lag.max = 0`.
/// * `pacf` - If true, return PACF instead of ACF
///
/// # Errors
///
/// `InvalidSpecification` when an AR or MA coefficient is not finite, or when the AR part is not
/// stationary (a root of 1 − φ₁z − ... − φₚzᵖ on or inside the unit circle): the theoretical ACF
/// does not exist there. R's `ARMAacf` does not check and returns meaningless numbers or a
/// singular-solve error.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arma_acf;
/// // AR(1) process with coefficient 0.7
/// let result = arma_acf(&[0.7], &[], 10, false).unwrap();
/// // ACF decays geometrically: 0.7, 0.49, 0.343, ...
/// assert!((result.values[2] - 0.49).abs() < 1e-12);
/// ```
///
/// # References
///
/// - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*, Section 3.3
/// - R `stats::ARMAacf`
pub fn arma_acf(ar: &[f64], ma: &[f64], lag_max: usize, pacf: bool) -> EconResult<ArmaAcfResult> {
    if ar.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the AR coefficients must be finite".to_string(),
        });
    }
    if ma.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the MA coefficients must be finite".to_string(),
        });
    }
    if !ar_is_stationary(ar)? {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the AR part is not stationary: a root of 1 - ar[1] z - ... - ar[p] z^p lies on or \
                 inside the unit circle (ar = {ar:?}), so the theoretical ACF does not exist"
            ),
        });
    }

    // Autocorrelations ρ(0..lag_max)
    let acf_values = arma_autocorrelation(ar, ma, lag_max)?;

    let values = if pacf {
        // Compute PACF from ACF using Durbin-Levinson
        let pacf_vals = durbin_levinson_from_acf(&acf_values)?;
        // Include lag 0 = 1.0 for consistency
        let mut result = try_with_capacity(
            pacf_vals.len() + 1,
            "the partial autocorrelations up to lag_max",
        )?;
        result.push(1.0);
        result.extend(pacf_vals);
        result
    } else {
        acf_values
    };

    // lag_max + 1 cannot overflow: arma_autocorrelation allocated that many values.
    let mut lags = try_with_capacity::<usize>(lag_max + 1, "the lags up to lag_max")?;
    lags.extend(0..=lag_max);

    Ok(ArmaAcfResult {
        values,
        lags,
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        is_pacf: pacf,
    })
}

/// Whether the AR polynomial 1 − φ₁z − ... − φₚzᵖ has every root outside the unit circle.
///
/// Step-down (Schur-Cohn) recursion: the partial autocorrelations a_kk of the AR part are recovered
/// from k = p down to 1 by a_j ← (a_j + a_k a_{k−j}) / (1 − a_k²); the part is stationary iff every
/// |a_kk| < 1. Trailing zero coefficients are allowed; a non-finite coefficient is not stationary.
/// O(p²) time, one buffer of p values.
pub(crate) fn ar_is_stationary(ar: &[f64]) -> EconResult<bool> {
    if ar.iter().any(|v| !v.is_finite()) {
        return Ok(false);
    }
    // Low orders (all in practice) work in stack buffers.
    const STACK: usize = 8;
    let (mut a_stack, mut prev_stack) = ([0.0; STACK], [0.0; STACK]);
    let (mut a_heap, mut prev_heap);
    let (a, prev): (&mut [f64], &mut [f64]) = if ar.len() <= STACK {
        (&mut a_stack[..ar.len()], &mut prev_stack[..ar.len()])
    } else {
        a_heap = try_vec(ar.len(), 0.0, "the AR step-down recursion")?;
        prev_heap = try_vec(ar.len(), 0.0, "the AR step-down recursion")?;
        (&mut a_heap[..], &mut prev_heap[..])
    };
    a.copy_from_slice(ar);
    // Each order costs O(k).
    let stride = checkpoint_stride(ar.len().max(1));
    for k in (1..=ar.len()).rev() {
        if k % stride == 0 {
            checkpoint()?;
        }
        let ak = a[k - 1];
        if ak.abs() >= 1.0 || !ak.is_finite() {
            return Ok(false);
        }
        let den = 1.0 - ak * ak;
        prev[..k - 1].copy_from_slice(&a[..k - 1]);
        for j in 1..k {
            a[j - 1] = (prev[j - 1] + ak * prev[k - j - 1]) / den;
        }
    }
    Ok(true)
}

/// The smallest modulus of the roots of 1 − φ₁z − ... − φₚzᵖ (R's `min(Mod(polyroot(c(1, -ar))))`).
///
/// Trailing zero coefficients are dropped first, as `polyroot` drops them. Closed form for p ≤ 2;
/// above, 1 / max |λ| over the eigenvalues λ of the companion matrix. `f64::INFINITY` when every
/// coefficient is 0 (no root).
pub(crate) fn ar_min_root_modulus(ar: &[f64]) -> EconResult<f64> {
    let p = ar.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1);
    let ar = &ar[..p];
    if ar.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the AR coefficients must be finite".to_string(),
        });
    }
    // max |λ| over the roots λ of λᵖ − φ₁λᵖ⁻¹ − ... − φₚ (the reciprocals of the polynomial's roots).
    let max_lambda = match p {
        0 => return Ok(f64::INFINITY),
        1 => ar[0].abs(),
        2 => {
            let (a1, a2) = (ar[0], ar[1]);
            let disc = a1 * a1 + 4.0 * a2;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                ((a1 + sq) / 2.0).abs().max(((a1 - sq) / 2.0).abs())
            } else {
                (-a2).sqrt()
            }
        }
        _ => {
            checkpoint()?;
            let n_cells = checked_len(&[p, p], "the AR companion matrix")?;
            let mut cells = try_vec(n_cells, 0.0, "the AR companion matrix")?;
            cells[..p].copy_from_slice(ar);
            for i in 1..p {
                cells[i * p + i - 1] = 1.0;
            }
            let companion = Array2::from_shape_vec((p, p), cells)
                .map_err(|e| EconError::Computation(format!("AR companion matrix: {e}")))?;
            let evs = crate::linalg::matrix_ops::ndarray_to_faer(&companion.view())
                .eigenvalues()
                .map_err(|_| {
                    EconError::Computation(
                        "the eigenvalues of the AR companion matrix did not converge".to_string(),
                    )
                })?;
            evs.iter().map(|ev| ev.re.hypot(ev.im)).fold(0.0, f64::max)
        }
    };
    Ok(1.0 / max_lambda)
}

/// Theoretical autocorrelations ρ(0..lag_max) of an ARMA(p, q) process with a stationary AR part
/// (R's `ARMAacf`; see [`arma_acf`] for the equations).
fn arma_autocorrelation(ar: &[f64], ma: &[f64], lag_max: usize) -> EconResult<Vec<f64>> {
    const WHAT: &str = "the ARMA autocorrelations up to lag_max";

    let p = ar.len();
    let q = ma.len();
    let n_lags = lag_max.checked_add(1).ok_or_else(|| size_overflow(WHAT))?;
    let mut rho = try_vec(n_lags, 0.0, WHAT)?;
    rho[0] = 1.0;

    if p == 0 {
        // Pure MA(q) (white noise for q = 0): γ(k) = Σⱼ θⱼ θⱼ₊ₖ with θ₀ = 1, zero beyond q.
        let gamma0 = 1.0 + ma.iter().map(|&t| t * t).sum::<f64>();
        let stride = checkpoint_stride(q.max(1));
        for k in 1..=q.min(lag_max) {
            if k % stride == 0 {
                checkpoint()?;
            }
            let mut sum = ma[k - 1];
            for i in 0..(q - k) {
                sum += ma[i] * ma[i + k];
            }
            rho[k] = sum / gamma0;
        }
        return Ok(rho);
    }

    let r = p.max(q + 1);
    // ρ(0..=r_known) from the equations; the AR recursion extends beyond.
    let r_known = if r == 1 {
        // AR(1): ρ(k) = φᵏ, which the recursion below produces from ρ(0) = 1.
        0
    } else {
        // Every caller-sized solve checks the budget before it starts.
        checkpoint()?;
        let gamma = arma_autocovariance_system(ar, ma, r)?;
        let gamma0 = gamma[0];
        if !(gamma0 > 0.0 && gamma0.is_finite()) {
            return Err(EconError::Computation(
                "the ARMA autocovariance equations gave a non-positive variance".to_string(),
            ));
        }
        let known = r.min(lag_max);
        for k in 1..=known {
            rho[k] = gamma[k] / gamma0;
        }
        known
    };

    // ρ(k) = Σᵢ φᵢ ρ(k − i) for k > r (each lag costs O(p)).
    extend_ar_recursion(ar, &mut rho, r_known + 1)?;
    Ok(rho)
}

/// 0 for a sum below the normal range, otherwise the sum. Subnormals are flushed because in that
/// range the AR recursion can cycle through tiny non-zero values instead of reaching 0, and
/// subnormal arithmetic is slow. Every ±0 becomes +0, so whether a sum started from +0 or from its
/// first term does not change the result. The flush is rare: a branch keeps it off the recursion's
/// chain of dependent operations.
#[inline(always)]
fn flush_subnormal(sum: f64) -> f64 {
    #[cold]
    #[inline(never)]
    fn zero() -> f64 {
        0.0
    }
    if sum.abs() < f64::MIN_POSITIVE {
        zero()
    } else {
        sum
    }
}

/// Fills `rho[from..]` by ρ(k) = Σᵢ φᵢ ρ(k − i) (terms before lag 0 omitted), with each sum taken
/// in the order of the coefficients and flushed by [`flush_subnormal`]. `rho[from..]` must be 0 on
/// entry: once p successive values are 0 every later one is, and the fill stops there. Orders 1
/// and 2 carry the previous values in registers.
fn extend_ar_recursion(ar: &[f64], rho: &mut [f64], from: usize) -> EconResult<()> {
    let p = ar.len();
    let n = rho.len();
    if p == 0 {
        return Ok(());
    }
    // Lags below p read fewer than p earlier values.
    let mut k = from;
    while k < n.min(p) {
        let mut sum = 0.0;
        for (i, &phi) in ar.iter().enumerate().take(k) {
            sum += phi * rho[k - 1 - i];
        }
        rho[k] = flush_subnormal(sum);
        k += 1;
    }
    // Each lag costs O(p); the budget is checked once per stride of lags.
    let stride = checkpoint_stride(p);
    while k < n {
        checkpoint()?;
        let end = k.saturating_add(stride).min(n);
        match *ar {
            [a1] => {
                let mut r1 = rho[k - 1];
                for slot in &mut rho[k..end] {
                    r1 = flush_subnormal(a1 * r1);
                    *slot = r1;
                }
            }
            [a1, a2] => {
                let (mut r1, mut r2) = (rho[k - 1], rho[k - 2]);
                for slot in &mut rho[k..end] {
                    let v = flush_subnormal(a1 * r1 + a2 * r2);
                    *slot = v;
                    (r1, r2) = (v, r1);
                }
            }
            _ => {
                for j in k..end {
                    let (head, tail) = rho.split_at_mut(j);
                    let mut terms = ar.iter().zip(head[j - p..].iter().rev());
                    let mut sum = terms.next().map_or(0.0, |(&phi, &r)| phi * r);
                    for (&phi, &r) in terms {
                        sum += phi * r;
                    }
                    tail[0] = flush_subnormal(sum);
                }
            }
        }
        if rho[end - p..end].iter().all(|&v| v == 0.0) {
            break;
        }
        k = end;
    }
    Ok(())
}

/// γ(0..=r) of an ARMA(p, q) with σ² = 1 and r = max(p, q + 1) ≥ 2, from the r + 1 linear equations
/// of R's `ARMAacf`, solved by Gaussian elimination with partial pivoting.
fn arma_autocovariance_system(ar: &[f64], ma: &[f64], r: usize) -> EconResult<Vec<f64>> {
    const WHAT: &str = "the ARMA autocovariance equations";
    let q = ma.len();
    let m = r + 1;
    // AR part padded with zeros to r terms.
    let mut phi = try_vec(r, 0.0, WHAT)?;
    phi[..ar.len()].copy_from_slice(ar);
    let psi = arma_to_ma(&phi, ma, q)?.psi;

    let n_cells = checked_len(&[m, m], WHAT)?;
    let mut a = try_vec(n_cells, 0.0, WHAT)?;
    let mut b = try_vec(m, 0.0, WHAT)?;
    let stride = checkpoint_stride(m);
    for k in 0..m {
        if k % stride == 0 {
            checkpoint()?;
        }
        a[k * m + k] += 1.0;
        for (i, &phi_i) in phi.iter().enumerate() {
            a[k * m + (k as isize - (i as isize + 1)).unsigned_abs()] -= phi_i;
        }
        // Σⱼ₌ₖ^q θⱼ ψⱼ₋ₖ with θ₀ = ψ₀ = 1.
        let theta = |j: usize| if j == 0 { 1.0 } else { ma[j - 1] };
        let psi_at = |j: usize| if j == 0 { 1.0 } else { psi[j - 1] };
        b[k] = (k..=q).map(|j| theta(j) * psi_at(j - k)).sum();
    }

    // Forward elimination with partial pivoting (each column costs O(m²)).
    let stride = checkpoint_stride(m.saturating_mul(m));
    for col in 0..m {
        if col % stride == 0 {
            checkpoint()?;
        }
        let pivot = (col..m)
            .max_by(|&i, &j| a[i * m + col].abs().total_cmp(&a[j * m + col].abs()))
            .unwrap_or(col);
        let pv = a[pivot * m + col];
        if pv == 0.0 || !pv.is_finite() {
            return Err(EconError::SingularMatrix {
                context: "the ARMA autocovariance equations".to_string(),
                suggestion: "check that the AR part is stationary".to_string(),
            });
        }
        if pivot != col {
            for j in 0..m {
                a.swap(pivot * m + j, col * m + j);
            }
            b.swap(pivot, col);
        }
        for row in (col + 1)..m {
            let f = a[row * m + col] / pv;
            if f != 0.0 {
                for j in col..m {
                    a[row * m + j] -= f * a[col * m + j];
                }
                b[row] -= f * b[col];
            }
        }
    }
    // Back substitution.
    let mut x = try_vec(m, 0.0, WHAT)?;
    for row in (0..m).rev() {
        let mut sum = b[row];
        for j in (row + 1)..m {
            sum -= a[row * m + j] * x[j];
        }
        x[row] = sum / a[row * m + row];
    }
    Ok(x)
}

/// Durbin-Levinson algorithm to compute PACF from ACF.
///
/// Autocorrelations, partial autocorrelations and AR coefficients below 1e-150 in magnitude are
/// treated as 0, so no product of two of them leaves the normal range: subnormal arithmetic is
/// slow, and an exact ACF of a long lag range decays into it. A partial autocorrelation below
/// 1e-150 is reported as 0.
fn durbin_levinson_from_acf(acf: &[f64]) -> EconResult<Vec<f64>> {
    const WHAT: &str = "the partial autocorrelations up to lag_max";
    const TINY: f64 = 1e-150;
    let flush = |v: f64| if v.abs() < TINY { 0.0 } else { v };

    let p = acf.len() - 1; // acf[0] = 1
    if p == 0 {
        return Ok(vec![]);
    }
    let mut flushed = try_with_capacity(acf.len(), WHAT)?;
    flushed.extend(acf.iter().map(|&v| flush(v)));
    let acf = &flushed[..];

    let mut pacf = try_with_capacity(p, WHAT)?;
    let mut phi: Vec<f64> = try_vec(p, 0.0, WHAT)?;
    // phi of the previous order, reused across orders instead of a clone each time
    let mut prev: Vec<f64> = try_vec(p, 0.0, WHAT)?;

    phi[0] = acf[1];
    pacf.push(phi[0]);

    for n in 2..=p {
        // Each order costs O(n).
        checkpoint()?;
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

        let phi_nn = flush(num / den);
        pacf.push(phi_nn);

        // A zero partial autocorrelation leaves the lower-order coefficients as they are.
        if phi_nn != 0.0 {
            prev[..n - 1].copy_from_slice(&phi[..n - 1]);
            for k in 1..n {
                phi[k - 1] = flush(prev[k - 1] - phi_nn * prev[n - k - 1]);
            }
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

    let mut psi = try_with_capacity(lag_max, "the psi weights up to lag_max")?;

    // Each weight costs O(min(j, p)).
    let stride = checkpoint_stride(p.saturating_add(1));
    for j in 1..=lag_max {
        if j % stride == 0 {
            checkpoint()?;
        }
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

    // The triangular matrix holds max_order * (max_order + 1) / 2 coefficients;
    // admit them before any work.
    const WHAT: &str = "the AR coefficient matrix (order × order / 2)";
    let order_plus_one = max_order
        .checked_add(1)
        .ok_or_else(|| size_overflow(WHAT))?;
    let cells = checked_len(&[max_order, order_plus_one], WHAT)? / 2;
    check_alloc::<f64>(cells, WHAT)?;

    let mut ar_matrix: Vec<Vec<f64>> = try_with_capacity(max_order, WHAT)?;
    let mut pacf = try_with_capacity(max_order, WHAT)?;

    // Durbin-Levinson recursion
    let mut phi: Vec<f64> = try_vec(max_order, 0.0, WHAT)?;
    // phi of the previous order, reused across orders instead of a clone each time
    let mut prev: Vec<f64> = try_vec(max_order, 0.0, WHAT)?;

    // AR(1)
    phi[0] = rho[1];
    ar_matrix.push(vec![phi[0]]);
    pacf.push(phi[0]);

    // AR(n) for n = 2 to max_order
    for n in 2..=max_order {
        // Each order costs O(n).
        checkpoint()?;
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
        prev[..n - 1].copy_from_slice(&phi[..n - 1]);
        for k in 1..n {
            phi[k - 1] = prev[k - 1] - phi_nn * prev[n - k - 1];
        }
        phi[n - 1] = phi_nn;

        // Store AR(n) coefficients
        let mut row = try_with_capacity(n, WHAT)?;
        row.extend_from_slice(&phi[..n]);
        ar_matrix.push(row);
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

/// Simulate from an ARIMA model with unit-variance innovations (R's `arima.sim` with `sd = 1`).
///
/// See [`arima_sim_with_sd`], which this calls with `sd = 1.0`.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arima_sim;
/// // Simulate AR(1) with coefficient 0.8
/// let result = arima_sim(&[0.8], &[], 0, 100, None, None, Some(42)).unwrap();
/// assert_eq!(result.values.len(), 100);
/// assert_eq!(result.n_start, 28); // R's default burn-in: 1 + ceiling(6 / log(1 / 0.8))
/// ```
pub fn arima_sim(
    ar: &[f64],
    ma: &[f64],
    d: usize,
    n: usize,
    innovations: Option<&[f64]>,
    n_start: Option<usize>,
    seed: Option<u64>,
) -> EconResult<ArimaSimResult> {
    arima_sim_with_sd(ar, ma, d, n, 1.0, innovations, n_start, seed)
}

/// Simulate from an ARIMA model (R's `arima.sim`).
///
/// Generates a time series following an ARIMA(p,d,q) process, as R does:
/// - the innovations are `n_start` burn-in values followed by `n` values;
/// - the MA filter e[t] + Σ θⱼ e[t − j] is applied, with its first q values set to 0
///   (R's `x[seq_along(ma)] <- 0`);
/// - the AR recursion y[t] = x[t] + Σ φᵢ y[t − i] starts from zeros;
/// - the burn-in is dropped and the series is integrated d times.
///
/// # Arguments
///
/// * `ar` - AR coefficients (can be empty); the AR part must be stationary
/// * `ma` - MA coefficients (can be empty)
/// * `d` - Differencing order (0 for ARMA)
/// * `n` - Number of observations returned; must be at least 1, as in R's
///   `arima.sim`. The integrated series keeps length `n`: R returns `n + d`
///   values for `d > 0` because `diffinv` prepends `d` zeros; this returns the
///   `d`-fold cumulative sum of the ARMA part, which is that output without
///   the leading zeros.
/// * `sd` - Standard deviation of the generated N(0, sd²) innovations (R's `sd` passed to
///   `rnorm`); finite and positive. Supplied `innovations` are used as given.
/// * `innovations` - Optional innovations: the `n_start` burn-in values (R's `start.innov`)
///   followed by the `n` values (R's `innov`). If None, N(0, sd²) draws.
/// * `n_start` - Burn-in length. Default R's `p + q + ceiling(6 / log(minroots))` (`p + q` for a
///   pure MA), where minroots is the smallest modulus of the roots of 1 − φ₁z − ... − φₚzᵖ; the
///   burn-in grows like 6 / log(minroots) as the AR part nears a unit root, and the allocation
///   ceiling and the budget bound it. At least `p + q`.
/// * `seed` - Random seed for reproducibility (if innovations is None)
///
/// # Errors
///
/// - [`EconError::InvalidSpecification`] for `n = 0`; for a non-finite AR or MA coefficient; for
///   an AR part that is not stationary (R: "'ar' part of model is not stationary"); for
///   `n_start < p + q` (R: "burn-in 'n.start' must be as long as 'ar + ma'"); for an `sd` that is
///   not finite and positive; or for `innovations` shorter than `n` plus the burn-in.
/// - [`EconError::ResourceLimit`] when `n` plus the burn-in overflows or the
///   series is too large to allocate (the per-allocation ceiling included).
/// - [`EconError::Cancelled`] when the budget checked through [`checkpoint`]
///   is cancelled or past its deadline, during the ARMA recursion or before
///   each integration pass.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::tsutils::arima_sim_with_sd;
/// let r = arima_sim_with_sd(&[0.5], &[0.3], 0, 50, 2.0, None, None, Some(7)).unwrap();
/// assert_eq!(r.values.len(), 50);
/// assert_eq!(r.n_start, 11); // 1 + 1 + ceiling(6 / log(2))
/// ```
///
/// # References
///
/// - Box et al. (2015), Chapter 3
/// - R `stats::arima.sim`
pub fn arima_sim_with_sd(
    ar: &[f64],
    ma: &[f64],
    d: usize,
    n: usize,
    sd: f64,
    innovations: Option<&[f64]>,
    n_start: Option<usize>,
    seed: Option<u64>,
) -> EconResult<ArimaSimResult> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    const WHAT: &str = "the simulated series (n plus burn-in)";

    let p = ar.len();
    let q = ma.len();

    if n == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n must be at least 1 (R: 'n' must be strictly positive)".to_string(),
        });
    }
    if !(sd.is_finite() && sd > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("sd must be finite and positive, got {sd}"),
        });
    }
    if ma.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the MA coefficients must be finite".to_string(),
        });
    }

    // R: minroots <- min(Mod(polyroot(c(1, -ar)))); if (minroots <= 1) stop(...)
    let not_stationary = || EconError::InvalidSpecification {
        message: "'ar' part of model is not stationary: a root of 1 - ar[1] z - ... - ar[p] z^p \
                  lies on or inside the unit circle"
            .to_string(),
    };
    if ar.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the AR coefficients must be finite".to_string(),
        });
    }
    let min_root = if p > 0 {
        if !ar_is_stationary(ar)? {
            return Err(not_stationary());
        }
        let m = ar_min_root_modulus(ar)?;
        if m <= 1.0 {
            return Err(not_stationary());
        }
        m
    } else {
        f64::INFINITY
    };

    let p_plus_q = p.checked_add(q).ok_or_else(|| size_overflow(WHAT))?;
    let burn_in = match n_start {
        Some(m) if m < p_plus_q => {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "burn-in 'n_start' must be as long as 'ar + ma' ({p_plus_q}), got {m}"
                ),
            });
        }
        Some(m) => m,
        None => {
            // R: n.start <- p + q + ifelse(p > 0, ceiling(6 / log(minroots)), 0)
            let extra = if p > 0 {
                (6.0 / min_root.ln()).ceil()
            } else {
                0.0
            };
            // usize::MAX as f64 rounds up to 2^64, so `<` keeps the cast exact.
            if extra.is_nan() || extra >= usize::MAX as f64 {
                return Err(size_overflow(WHAT));
            }
            p_plus_q
                .checked_add(extra as usize)
                .ok_or_else(|| size_overflow(WHAT))?
        }
    };

    let total_length = n.checked_add(burn_in).ok_or_else(|| size_overflow(WHAT))?;

    // Generate or use innovations
    let innov: Vec<f64> = if let Some(ext_innov) = innovations {
        if ext_innov.len() < total_length {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Need {} innovations but only {} provided",
                    total_length,
                    ext_innov.len()
                ),
            });
        }
        let mut innov = try_with_capacity(total_length, WHAT)?;
        innov.extend_from_slice(&ext_innov[..total_length]);
        innov
    } else {
        // Generate random innovations
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut innov: Vec<f64> = try_with_capacity(total_length, WHAT)?;
        innov.extend((0..total_length).map(|_| -> f64 {
            let z: f64 = StandardNormal.sample(&mut rng);
            sd * z
        }));
        innov
    };

    // Simulate ARMA part
    let mut x = try_vec(total_length, 0.0, WHAT)?;
    arma_filter(ar, ma, &innov, &mut x)?;

    // Remove burn-in
    x.drain(..burn_in);

    // Un-difference: apply cumsum d times, in place
    integrate_in_place(&mut x, d)?;

    Ok(ArimaSimResult {
        values: x,
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        d,
        n,
        n_start: burn_in,
    })
}

/// The ARMA part of [`arima_sim_with_sd`]: `x[t]` is the MA filter e[t] + Σⱼ θⱼ e[t − j] (0 for
/// the first q steps, as R's `x[seq_along(ma)] <- 0`) plus Σᵢ φᵢ x[t − i] (zeros before the start),
/// each sum taken in the order of the coefficients. `x` must be as long as `innov`.
fn arma_filter(ar: &[f64], ma: &[f64], innov: &[f64], x: &mut [f64]) -> EconResult<()> {
    let (p, q) = (ar.len(), ma.len());
    let total = innov.len().min(x.len());
    // The first max(p, q) steps reach before the start.
    let warm_up = p.max(q).min(total);
    for t in 0..warm_up {
        let mut val = if t < q {
            0.0
        } else {
            let mut v = innov[t];
            for (j, &theta) in ma.iter().enumerate() {
                v += theta * innov[t - 1 - j];
            }
            v
        };
        for (i, &phi) in ar.iter().enumerate().take(t) {
            val += phi * x[t - 1 - i];
        }
        x[t] = val;
    }
    // Each step costs O(p + q); the budget is checked once per stride of steps.
    let stride = checkpoint_stride(p + q + 1);
    let mut t = warm_up;
    while t < total {
        checkpoint()?;
        let end = t.saturating_add(stride).min(total);
        match p {
            0 => arma_filter_steps::<0>(ar, ma, innov, x, t, end),
            1 => arma_filter_steps::<1>(ar, ma, innov, x, t, end),
            2 => arma_filter_steps::<2>(ar, ma, innov, x, t, end),
            _ => {
                for s in t..end {
                    let mut val = innov[s];
                    for (&theta, &e) in ma.iter().zip(innov[s - q..s].iter().rev()) {
                        val += theta * e;
                    }
                    for (&phi, &v) in ar.iter().zip(x[s - p..s].iter().rev()) {
                        val += phi * v;
                    }
                    x[s] = val;
                }
            }
        }
        t = end;
    }
    Ok(())
}

/// Steps `from..to` (`from >= max(p, q)`) of [`arma_filter`] for an AR order of `P` (`ar.len()`),
/// the last `P` values carried in registers.
#[inline(always)]
fn arma_filter_steps<const P: usize>(
    ar: &[f64],
    ma: &[f64],
    innov: &[f64],
    x: &mut [f64],
    from: usize,
    to: usize,
) {
    let q = ma.len();
    let mut phi = [0.0; P];
    phi.copy_from_slice(&ar[..P]);
    // last[i] = x[t - 1 - i]
    let mut last = [0.0; P];
    for (i, v) in last.iter_mut().enumerate() {
        *v = x[from - 1 - i];
    }
    for t in from..to {
        let mut val = innov[t];
        for (&theta, &e) in ma.iter().zip(innov[t - q..t].iter().rev()) {
            val += theta * e;
        }
        for i in 0..P {
            val += phi[i] * last[i];
        }
        x[t] = val;
        for i in (1..P).rev() {
            last[i] = last[i - 1];
        }
        if P > 0 {
            last[0] = val;
        }
    }
}

/// Integrate `x` `d` times in place: each pass replaces `x` by its
/// cumulative sum. The budget is checked before every pass, so a large `d`
/// stays cancellable.
fn integrate_in_place(x: &mut [f64], d: usize) -> EconResult<()> {
    for _ in 0..d {
        // Each pass costs O(n).
        checkpoint()?;
        let mut running = 0.0;
        for v in x.iter_mut() {
            running += *v;
            *v = running;
        }
    }
    Ok(())
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
    /// R's `smoothEnds`: medians of decreasing odd spans of the end-kept series, and Tukey's
    /// end-point rule for the first and last value (default, as R's `runmed`)
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
            // R: smoothEnds(runmed(x, k, endrule = "keep"), k). The windows read the end-kept
            // running median y (raw ends, smoothed interior), never the end-smoothed values being
            // written.
            for i in 0..half_k {
                result[i] = x[i];
                result[n - 1 - i] = x[n - 1 - i];
            }
            smooth_ends(&mut result, half_k)?;
        }
    }

    Ok(RunmedResult {
        values: result,
        k,
        endrule,
        n_obs: n,
    })
}

/// R's `smoothEnds(y, 2 h + 1)` in place: for i = 2..=h (while 2 i <= n) the i-th value from each
/// end becomes the median of the 2 i - 1 values of `y` at that end, then the first and last value
/// follow Tukey's end-point rule med3(y[1], sm[2], 3 sm[2] - 2 sm[3]).
fn smooth_ends(y: &mut [f64], h: usize) -> EconResult<()> {
    let n = y.len();
    if h < 1 || n < 3 {
        return Ok(());
    }
    let med3 = |a: f64, b: f64, c: f64| -> f64 {
        // R's med3: the middle value
        if a < b {
            if c < b { if a >= c { a } else { c } } else { b }
        } else if c > b {
            if a <= c { a } else { c }
        } else {
            b
        }
    };
    let mut src = try_with_capacity(n, "the running-median end windows")?;
    src.extend_from_slice(y);
    if h >= 2 {
        y[1] = med3(src[0], src[1], src[2]);
        y[n - 2] = med3(src[n - 1], src[n - 2], src[n - 3]);
        let mut window = try_with_capacity(2 * h - 1, "the running-median end windows")?;
        for i in 3..=h {
            if 2 * i > n {
                break;
            }
            // Each pair of end windows costs O(i log i).
            checkpoint()?;
            window.clear();
            window.extend_from_slice(&src[..2 * i - 1]);
            y[i - 1] = median_odd(&mut window);
            window.clear();
            window.extend_from_slice(&src[n + 1 - 2 * i..]);
            y[n - i] = median_odd(&mut window);
        }
    }
    y[0] = med3(src[0], y[1], 3.0 * y[1] - 2.0 * y[2]);
    y[n - 1] = med3(src[n - 1], y[n - 2], 3.0 * y[n - 2] - 2.0 * y[n - 3]);
    Ok(())
}

/// The middle value of an odd-length window (R's `med.odd`), NaN last.
fn median_odd(w: &mut [f64]) -> f64 {
    let half = w.len() / 2;
    let (_, m, _) = w.select_nth_unstable_by(half, f64::total_cmp);
    *m
}

/// Compute median of a slice.
fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let mut sorted: Vec<f64> = x.iter().cloned().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));

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
    use p2a_validation::{Check, Tol};

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), name, crate::data::columns::NaPolicy::Omit)
            .unwrap()
            .0
    }

    /// Case ts_filter_ar_cpt: stats::filter(x, f, method = "convolution", sides = s) on the
    /// 60-value random walk x.
    fn check_filters(case_id: &str, lengths: &[usize], sides: FilterSides, suffix: &str) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let x = col(&ds, "x");
        let filters: [(&str, Vec<f64>); 4] = [
            ("filter_p2_05", vec![0.5, 0.5]),
            ("filter_p4_025", vec![0.25; 4]),
            ("filter_p3_02", vec![0.2, 0.3, 0.5]),
            (
                "filter_p5_01111111",
                [1.0, 2.0, 3.0, 2.0, 1.0].iter().map(|v| v / 9.0).collect(),
            ),
        ];
        let mut check = Check::new(&case);
        for (key, f) in filters.iter().filter(|(_, f)| lengths.contains(&f.len())) {
            let r = filter(&x, f, FilterMethod::Convolution, sides, None).unwrap();
            check.vector(&format!("{key}{suffix}"), &r.values, Tol::coef(x.len())); // NA == NaN
        }
        check.finish();
    }

    #[test]
    fn test_validate_filter_even_sides2() {
        check_filters("ts_filter_ar_cpt", &[2, 4], FilterSides::Two, "");
    }

    #[test]
    fn test_validate_filter_odd_sides2() {
        check_filters("ts_filter_ar_cpt", &[3, 5], FilterSides::Two, "");
    }

    #[test]
    fn test_validate_filter_even_sides1() {
        check_filters("ts_filter_ar_cpt", &[2, 4], FilterSides::One, "_sides1");
    }

    #[test]
    fn test_validate_filter_odd_sides1() {
        check_filters("ts_filter_ar_cpt", &[3, 5], FilterSides::One, "_sides1");
    }

    /// stats::filter(x, c(0.5, -0.2), method = "recursive"), init = 0 (guard).
    #[test]
    fn test_validate_filter_recursive() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let x = col(&ds, "x");
        let r = filter(
            &x,
            &[0.5, -0.2],
            FilterMethod::Recursive,
            FilterSides::One,
            None,
        )
        .unwrap();
        Check::new(&case)
            .vector("filter_recursive", &r.values, Tol::coef(x.len()))
            .finish();
    }

    /// The finding's example. R 4.3.3: filter(x, rep(.25, 4), sides = 2) = NA 3.75 4.75 5.5
    /// 5.75 6 6.25 7 NA NA; filter(x, 1:4, sides = 2) = NA 28 43 49 64 56 61 61 NA NA.
    #[test]
    fn test_filter_even_two_sided_matches_r_cfilter() {
        let x = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 10.0];
        let run = |f: &[f64]| {
            filter(&x, f, FilterMethod::Convolution, FilterSides::Two, None)
                .unwrap()
                .values
        };
        let nan = f64::NAN;
        let a = run(&[0.25; 4]);
        let want_a = [nan, 3.75, 4.75, 5.5, 5.75, 6.0, 6.25, 7.0, nan, nan];
        let b = run(&[1.0, 2.0, 3.0, 4.0]);
        let want_b = [nan, 28.0, 43.0, 49.0, 64.0, 56.0, 61.0, 61.0, nan, nan];
        for (got, want) in [(a, want_a), (b, want_b)] {
            for (g, w) in got.iter().zip(want) {
                assert!(*g == w || (g.is_nan() && w.is_nan()), "{got:?}");
            }
        }
    }

    /// R: "'filter' is longer than time series" for sides 1 and 2.
    #[test]
    fn test_filter_longer_than_series_is_an_error() {
        let x = [1.0, 4.0, 2.0];
        for (f, sides) in [
            (vec![0.1; 10], FilterSides::Two),
            (vec![0.1; 5], FilterSides::One),
            (vec![0.1; 4], FilterSides::Two),
        ] {
            let err = filter(&x, &f, FilterMethod::Convolution, sides, None).unwrap_err();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. })
                    && err.to_string().contains("longer than the time series"),
                "{err}"
            );
        }
        // R's recursive filter has no such limit
        assert!(
            filter(
                &x,
                &[0.1; 5],
                FilterMethod::Recursive,
                FilterSides::One,
                None
            )
            .is_ok()
        );
    }

    /// R: "missing values in 'filter'".
    #[test]
    fn test_filter_rejects_missing_coefficients() {
        let err = filter(
            &[1.0, 2.0, 3.0],
            &[0.5, f64::NAN],
            FilterMethod::Convolution,
            FilterSides::Two,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing values"), "{err}");
    }

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

        let result = filter(
            &x,
            &filter_coefs,
            FilterMethod::Convolution,
            FilterSides::One,
            None,
        )
        .unwrap();

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

        let result = filter(
            &x,
            &filter_coefs,
            FilterMethod::Recursive,
            FilterSides::One,
            None,
        )
        .unwrap();

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

    /// diffinv(x[1:12], xi = 1.5); diffinv(x[1:12], lag = 2, xi = c(1, -2)); diffinv(x[1:12], differences = 2);
    /// diffinv(x[1:12], differences = 2, xi = c(1, 3)).
    #[test]
    fn test_validate_diffinv_ts_filter_ar_cpt() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let x12 = &col(&ds, "x")[..12];
        Check::new(&case)
            .vector(
                "diffinv_lag1",
                &diffinv(x12, 1, 1, Some(&[1.5])).unwrap().values,
                Tol::coef(12),
            )
            .vector(
                "diffinv_lag2",
                &diffinv(x12, 2, 1, Some(&[1.0, -2.0])).unwrap().values,
                Tol::coef(12),
            )
            .vector(
                "diffinv_d2_zero",
                &diffinv(x12, 1, 2, None).unwrap().values,
                Tol::coef(12),
            )
            .vector(
                "diffinv_d2_xi",
                &diffinv(x12, 1, 2, Some(&[1.0, 3.0])).unwrap().values,
                Tol::coef(12),
            )
            .finish();
    }

    /// ARMAacf(ar = 0.8), (0.5, -0.3) with its PACF (lags 1..10; R omits lag 0), ma = (0.4, 0.25),
    /// ARMA(1, 1) (0.5; 0.3) and the persistent AR(2) (1.5, -0.56). Closed-form reference: rel 1e-10, abs 1e-12.
    #[test]
    fn test_validate_arma_acf_ts_filter_ar_cpt() {
        let (case, _) = crate::test_support::load_case("ts_filter_ar_cpt");
        let tol = Tol::new(1e-10, 1e-12);
        let pacf = arma_acf(&[0.5, -0.3], &[], 10, true).unwrap().values;
        Check::new(&case)
            .vector(
                "armaacf_arma11",
                &arma_acf(&[0.5], &[0.3], 6, false).unwrap().values,
                tol,
            )
            .vector(
                "armaacf_ar2_persistent",
                &arma_acf(&[1.5, -0.56], &[], 10, false).unwrap().values,
                tol,
            )
            .vector(
                "armaacf_ar1",
                &arma_acf(&[0.8], &[], 10, false).unwrap().values,
                tol,
            )
            .vector(
                "armaacf_ar2",
                &arma_acf(&[0.5, -0.3], &[], 10, false).unwrap().values,
                tol,
            )
            .vector("armaacf_ar2_pacf", &pacf[1..], tol)
            .vector(
                "armaacf_ma2",
                &arma_acf(&[], &[0.4, 0.25], 6, false).unwrap().values,
                tol,
            )
            .finish();
    }

    /// Case ts_utils_t4: ARMAacf of mixed ARMA(p, q) models, a persistent AR(2), AR(0.99), and lag.max below
    /// p and below q + 1 (R truncates to lag.max + 1 values there). PACF compared from lag 1, as R returns it.
    /// Closed-form reference (exact autocovariance equations): rel 1e-10, abs 1e-12.
    #[test]
    fn test_validate_arma_acf_mixed_ts_utils_t4() {
        let (case, _) = crate::test_support::load_case("ts_utils_t4");
        let tol = Tol::new(1e-10, 1e-12);
        let acf = |ar: &[f64], ma: &[f64], lag_max: usize| {
            arma_acf(ar, ma, lag_max, false).unwrap().values
        };
        let pacf = |ar: &[f64], ma: &[f64], lag_max: usize| {
            arma_acf(ar, ma, lag_max, true).unwrap().values[1..].to_vec()
        };
        Check::new(&case)
            .vector("acf_arma11", &acf(&[0.5], &[0.3], 6), tol)
            .vector("acf_arma12", &acf(&[0.9], &[-0.5, 0.4], 6), tol)
            .vector("acf_arma21", &acf(&[0.5, 0.3], &[0.4], 6), tol)
            .vector("pacf_arma21", &pacf(&[0.5, 0.3], &[0.4], 6), tol)
            .vector("acf_ar2_persistent", &acf(&[1.5, -0.56], &[], 10), tol)
            .vector("acf_ar1_099", &acf(&[0.99], &[], 5), tol)
            .vector("acf_arma32", &acf(&[0.4, -0.2, 0.3], &[0.5, 0.25], 8), tol)
            .vector(
                "pacf_arma32",
                &pacf(&[0.4, -0.2, 0.3], &[0.5, 0.25], 8),
                tol,
            )
            .vector("acf_arma13_lag2", &acf(&[0.6], &[0.3, -0.2, 0.1], 2), tol)
            .vector("acf_ar3_lag1", &acf(&[0.3, 0.2, 0.1], &[], 1), tol)
            .finish(); // acf_ma3_lag5: R returns q + 1 values for lag.max < q; p2a returns lag_max + 1 (method doc)
    }

    #[test]
    fn test_arma_acf_refuses_non_stationary_ar() {
        for ar in [vec![1.2], vec![1.0], vec![1.5, -0.5], vec![0.5, 0.5]] {
            for pacf in [false, true] {
                let err = arma_acf(&ar, &[0.3], 5, pacf).unwrap_err();
                assert!(
                    matches!(&err, EconError::InvalidSpecification { message } if message.contains("not stationary")),
                    "{ar:?}: {err}"
                );
            }
        }
        assert!(arma_acf(&[0.99], &[], 5, false).is_ok());
        assert!(arma_acf(&[1.5, -0.56], &[], 5, false).is_ok());
    }

    #[test]
    fn test_arma_acf_refuses_non_finite_coefficients() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for pacf in [false, true] {
                // pure MA, mixed, and the AR side
                for (ar, ma, what) in [
                    (vec![], vec![0.4, bad], "MA"),
                    (vec![0.5], vec![bad], "MA"),
                    (vec![bad], vec![0.3], "AR"),
                    (vec![0.5, bad], vec![], "AR"),
                ] {
                    let err = arma_acf(&ar, &ma, 5, pacf).unwrap_err();
                    let want = format!("the {what} coefficients must be finite");
                    assert!(
                        matches!(&err, EconError::InvalidSpecification { message } if *message == want),
                        "{ar:?} {ma:?}: {err}"
                    );
                }
            }
        }
    }

    /// A long lag range decays below the normal range: the ACF holds no subnormal value and its
    /// tail is exactly 0, and the PACF's AR coefficients do not leave the normal range either.
    #[test]
    fn test_arma_acf_tail_has_no_subnormals() {
        let normal_or_zero = |v: &f64| *v == 0.0 || v.abs() >= f64::MIN_POSITIVE;
        let acf = arma_acf(&[0.5, 0.3], &[0.4], 6_000, false).unwrap().values;
        assert!(acf.iter().all(normal_or_zero));
        assert_eq!(*acf.last().unwrap(), 0.0);
        let pacf = arma_acf(&[0.5], &[0.3], 1_500, true).unwrap().values;
        assert!(pacf.iter().all(|v| v.is_finite() && normal_or_zero(v)));
        // The PACF of ARMA(1, 1) decays like (-0.3)^k: its first lags are unaffected.
        let short = arma_acf(&[0.5], &[0.3], 20, true).unwrap().values;
        assert_eq!(&pacf[..21], &short[..]);
    }

    /// A seeded uniform draw in (-0.5, 0.5) for the equivalence tests.
    fn noise_source(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// The AR recursion of `arma_autocorrelation` as it was: a sum from 0 for every lag.
    fn ar_recursion_reference(ar: &[f64], rho: &mut [f64], from: usize) {
        for k in from..rho.len() {
            let mut sum = 0.0;
            for (i, &phi) in ar.iter().enumerate().take(k) {
                sum += phi * rho[k - 1 - i];
            }
            rho[k] = if sum.abs() < f64::MIN_POSITIVE {
                0.0
            } else {
                sum
            };
        }
    }

    #[test]
    fn test_ar_recursion_matches_the_sum_from_zero() {
        let mut noise = noise_source(11);
        let orders: [&[f64]; 8] = [
            &[0.5],
            &[-0.97],
            &[0.5, -0.3],
            &[1.2, -0.5],
            &[-0.4, -0.2],
            &[0.3, 0.2, 0.1],
            &[0.2, -0.1, 0.3, 0.05, 0.1],
            &[0.0, 0.0, 0.6],
        ];
        for ar in orders {
            for &len in &[1usize, 2, 3, 6, 200, 5000] {
                for scale in [1.0, 1e-300] {
                    let mut new = vec![0.0; len];
                    for v in new.iter_mut().take(ar.len().min(len)) {
                        *v = scale * noise();
                    }
                    let mut old = new.clone();
                    let from = ar.len().min(len);
                    extend_ar_recursion(ar, &mut new, from).unwrap();
                    ar_recursion_reference(ar, &mut old, from);
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&new),
                        bits(&old),
                        "ar = {ar:?}, len = {len}, scale = {scale}"
                    );
                }
            }
        }
        // Started below the order too (lags 1.. of an AR(3) from rho[0] alone).
        let mut new = vec![0.0; 50];
        new[0] = 1.0;
        let mut old = new.clone();
        extend_ar_recursion(&[0.3, 0.2, 0.1], &mut new, 1).unwrap();
        ar_recursion_reference(&[0.3, 0.2, 0.1], &mut old, 1);
        assert_eq!(new, old);
    }

    /// The ARMA filter of `arima_sim_with_sd` as it was: one loop over every step.
    fn arma_filter_reference(ar: &[f64], ma: &[f64], innov: &[f64]) -> Vec<f64> {
        let q = ma.len();
        let mut x = vec![0.0; innov.len()];
        for t in 0..innov.len() {
            let mut val = if t < q {
                0.0
            } else {
                let mut v = innov[t];
                for (j, &theta) in ma.iter().enumerate() {
                    v += theta * innov[t - 1 - j];
                }
                v
            };
            for (i, &phi) in ar.iter().enumerate().take(t) {
                val += phi * x[t - 1 - i];
            }
            x[t] = val;
        }
        x
    }

    #[test]
    fn test_arma_filter_matches_the_single_loop() {
        let mut noise = noise_source(5);
        let ars: [&[f64]; 5] = [
            &[],
            &[0.95],
            &[0.5, -0.3],
            &[0.3, 0.2, 0.1],
            &[0.1, 0.0, 0.2, -0.1],
        ];
        let mas: [&[f64]; 4] = [&[], &[0.4], &[0.4, 0.2], &[0.3, -0.2, 0.1, 0.05, 0.2]];
        for ar in ars {
            for ma in mas {
                for &len in &[1usize, 2, 4, 5, 7, 300] {
                    let innov: Vec<f64> = (0..len).map(|_| 3.0 * noise()).collect();
                    let mut x = vec![0.0; len];
                    arma_filter(ar, ma, &innov, &mut x).unwrap();
                    let old = arma_filter_reference(ar, ma, &innov);
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&x),
                        bits(&old),
                        "ar = {ar:?}, ma = {ma:?}, len = {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_arma_acf_length_is_lag_max_plus_one() {
        let cases: [(&[f64], &[f64], usize); 5] = [
            (&[0.3, 0.2, 0.1], &[], 1),
            (&[], &[0.6, -0.3, 0.2], 1),
            (&[0.6], &[0.3, -0.2, 0.1], 2),
            (&[0.5], &[0.3], 0),
            (&[0.4, -0.2, 0.3], &[0.5, 0.25], 0),
        ];
        for (ar, ma, lag_max) in cases {
            for pacf in [false, true] {
                let r = arma_acf(ar, ma, lag_max, pacf).unwrap();
                assert_eq!(
                    r.values.len(),
                    lag_max + 1,
                    "{ar:?} {ma:?} {lag_max} {pacf}"
                );
                assert_eq!(r.lags.len(), lag_max + 1, "{ar:?} {ma:?} {lag_max} {pacf}");
                assert_eq!(r.values[0], 1.0);
            }
        }
    }

    #[test]
    fn test_ar_min_root_modulus() {
        let close = |a: f64, b: f64| ((a - b) / b).abs() < 1e-10;
        assert!(close(ar_min_root_modulus(&[0.5]).unwrap(), 2.0));
        // 1 - 1.5 z + 0.56 z^2 has roots 1 / 0.8 and 1 / 0.7 (R polyroot: 1.2500000000000018).
        assert!(close(ar_min_root_modulus(&[1.5, -0.56]).unwrap(), 1.25));
        // R: min(Mod(polyroot(c(1, -0.4, 0.2, -0.3)))) = 1.4054665058271332
        assert!(close(
            ar_min_root_modulus(&[0.4, -0.2, 0.3]).unwrap(),
            1.4054665058271332
        ));
        // Complex pair: 1 - 0.5 z + 0.3 z^2 has |root|^2 = 1 / 0.3.
        assert!(close(
            ar_min_root_modulus(&[0.5, -0.3]).unwrap(),
            (1.0f64 / 0.3).sqrt()
        ));
        assert_eq!(ar_min_root_modulus(&[0.0, 0.0]).unwrap(), f64::INFINITY);
        assert_eq!(ar_min_root_modulus(&[]).unwrap(), f64::INFINITY);
        assert!(close(ar_min_root_modulus(&[0.5, 0.0]).unwrap(), 2.0)); // trailing zero dropped
        assert!(ar_is_stationary(&[0.5, 0.0]).unwrap());
        assert!(!ar_is_stationary(&[0.5, 0.5]).unwrap());
        assert!(!ar_is_stationary(&[f64::NAN]).unwrap());
    }

    /// Case ts_utils_t4: diffinv with differences >= 2 and xi, lag 1, 2 and 3. Dyadic inputs, so the sums are
    /// exact.
    #[test]
    fn test_validate_diffinv_xi_ts_utils_t4() {
        let (case, _) = crate::test_support::load_case("ts_utils_t4");
        let tol = Tol::new(0.0, 1e-12);
        let dinv = |x: &[f64], lag: usize, d: usize, xi: &[f64]| {
            diffinv(x, lag, d, Some(xi)).unwrap().values
        };
        Check::new(&case)
            .vector(
                "dinv_d2_xi",
                &dinv(&[2.0, 3.0, 4.0], 1, 2, &[1.0, 3.0]),
                tol,
            )
            .vector(
                "dinv_l2_d2",
                &dinv(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0, 2.0, 3.0, 4.0]),
                tol,
            )
            .vector(
                "dinv_d3",
                &dinv(&[0.5, -1.0, 2.0, 0.25, 3.0], 1, 3, &[1.0, -2.0, 0.5]),
                tol,
            )
            .vector(
                "dinv_l3_d2",
                &dinv(
                    &[0.5, -1.0, 2.0, 0.25, 3.0, 1.5, -0.75],
                    3,
                    2,
                    &[1.0, -2.0, 0.5, 4.0, 0.0, 2.5],
                ),
                tol,
            )
            .finish();
    }

    #[test]
    fn test_diffinv_refuses_wrong_xi_length() {
        for xi in [vec![1.0], vec![1.0, 2.0, 3.0]] {
            let err = diffinv(&[2.0, 3.0, 4.0], 1, 2, Some(&xi)).unwrap_err();
            assert!(
                matches!(&err, EconError::InvalidSpecification { message }
                    if message.contains('2') && message.contains(&xi.len().to_string())),
                "{err}"
            );
        }
        // Round trip: diffinv(diff(x, differences = 2), xi = x[1:2]) == x
        let x = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0];
        let d1: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let d2: Vec<f64> = d1.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(
            diffinv(&d2, 1, 2, Some(&x[..2])).unwrap().values,
            x.to_vec()
        );
    }

    /// Case ts_utils_t4: runmed(x, k, endrule = "median") = smoothEnds(runmed(x, k, "keep"), k), with Tukey's
    /// end-point rule. Medians of given values, so exact up to the 3 y[1] - 2 y[2] extrapolation.
    #[test]
    fn test_validate_runmed_median_ts_utils_t4() {
        let (case, _) = crate::test_support::load_case("ts_utils_t4");
        let tol = Tol::new(0.0, 1e-12);
        let rm = |x: &[f64], k: usize| runmed(x, k, EndRule::Median).unwrap().values;
        let z = [
            3.1, -0.4, 2.2, 5.0, 1.7, 0.3, 4.4, 2.9, -1.2, 0.8, 3.3, 2.0, 6.1,
        ];
        Check::new(&case)
            .vector(
                "rm_k3_a",
                &rm(&[5.0, 1.0, 2.0, 3.0, 10.0, 4.0, 6.0], 3),
                tol,
            )
            .vector(
                "rm_k5_b",
                &rm(&[5.0, 1.0, 2.0, 3.0, 10.0, 4.0, 6.0, 8.0, 0.0], 5),
                tol,
            )
            .vector(
                "rm_k3_doc",
                &rm(&[1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0], 3),
                tol,
            )
            .vector("rm_z_k3", &rm(&z, 3), tol)
            .vector("rm_z_k7", &rm(&z, 7), tol)
            .vector("rm_z_k9", &rm(&z, 9), tol)
            .vector("rm_z_k13", &rm(&z, 13), tol)
            .vector("rm_n5_k5", &rm(&[3.0, 1.0, 2.0, 9.0, 4.0], 5), tol)
            .finish();
    }

    /// Case ts_utils_t4: arima.sim with given innovations (start.innov = e[1:n.start], innov = e[...]) and R's
    /// default burn-in n.start = p + q + ceiling(6 / log(minroots)).
    #[test]
    fn test_validate_arima_sim_given_innovations_ts_utils_t4() {
        let (case, ds) = crate::test_support::load_case("ts_utils_t4");
        let e = col(&ds, "e");
        assert_eq!(e.len(), 40);
        let arma = arima_sim(&[0.5, -0.3], &[0.4], 0, 20, Some(&e[..33]), Some(13), None).unwrap();
        let arima = arima_sim(&[0.5], &[0.4], 1, 20, Some(&e[..31]), Some(11), None).unwrap();
        // R returns n + d values for d > 0, the first d of them 0; p2a returns the rest.
        let mut arima_r = vec![0.0];
        arima_r.extend(&arima.values);
        let models: [(&[f64], &[f64]); 7] = [
            (&[0.5], &[]),
            (&[0.99], &[]),
            (&[0.95], &[]),
            (&[0.5, -0.3], &[0.4]),
            (&[], &[0.4, 0.2]),
            (&[1.5, -0.56], &[]),
            (&[0.5, 0.0], &[]),
        ];
        let nstart: Vec<f64> = models
            .iter()
            .map(|(ar, ma)| {
                arima_sim(ar, ma, 0, 1, None, None, Some(1))
                    .unwrap()
                    .n_start as f64
            })
            .collect();
        Check::new(&case)
            .vector("sim_arma21", &arma.values, Tol::coef(33))
            .vector("sim_arima111", &arima_r, Tol::coef(33))
            .vector("nstart", &nstart, Tol::new(0.0, 0.0))
            .finish();
    }

    #[test]
    fn test_arima_sim_refuses_non_stationary_ar() {
        for ar in [vec![1.2], vec![1.0], vec![0.5, 0.5]] {
            let err = arima_sim(&ar, &[], 0, 10, None, None, Some(1)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("'ar' part of model is not stationary"),
                "{ar:?}: {err}"
            );
        }
        // A non-finite AR coefficient is refused for what it is, not as a non-stationary part.
        for ar in [vec![f64::NAN], vec![0.5, f64::INFINITY]] {
            let err = arima_sim(&ar, &[], 0, 10, None, None, Some(1)).unwrap_err();
            assert!(
                matches!(&err, EconError::InvalidSpecification { message }
                    if message == "the AR coefficients must be finite"),
                "{ar:?}: {err}"
            );
        }
        let err = arima_sim(&[0.5], &[0.3], 0, 10, None, Some(1), Some(1)).unwrap_err();
        assert!(
            err.to_string()
                .contains("burn-in 'n_start' must be as long as 'ar + ma'"),
            "{err}"
        );
        assert!(arima_sim(&[0.5], &[0.3], 0, 10, None, Some(2), Some(1)).is_ok());
    }

    #[test]
    fn test_arima_sim_sd_scales_the_series() {
        let one = arima_sim_with_sd(&[0.5, -0.3], &[0.4], 1, 30, 1.0, None, None, Some(9)).unwrap();
        let big = arima_sim_with_sd(&[0.5, -0.3], &[0.4], 1, 30, 2.5, None, None, Some(9)).unwrap();
        // The bound scales with the series: sd * z through the filter and the integration is not
        // bit-for-bit 2.5 times the sd = 1 path, so a value near 0 has no relative accuracy.
        let max_abs_b = big.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in one.values.iter().zip(&big.values) {
            assert!((b - 2.5 * a).abs() <= 1e-12 * max_abs_b, "{a} {b}");
        }
        assert_eq!(
            one.values,
            arima_sim(&[0.5, -0.3], &[0.4], 1, 30, None, None, Some(9))
                .unwrap()
                .values
        );
        for sd in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = arima_sim_with_sd(&[0.5], &[], 0, 10, sd, None, None, Some(1)).unwrap_err();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. }),
                "{sd}: {err}"
            );
        }
        // Supplied innovations are used as given.
        let innov: Vec<f64> = (0..40)
            .map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.5)
            .collect();
        let given1 =
            arima_sim_with_sd(&[0.5], &[], 0, 20, 1.0, Some(&innov), Some(10), None).unwrap();
        let given2 =
            arima_sim_with_sd(&[0.5], &[], 0, 20, 3.0, Some(&innov), Some(10), None).unwrap();
        assert_eq!(given1.values, given2.values);
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

    /// lag gives R's stats::lag values (x[t + k], |k| shorter); lag_padded gives the conventional
    /// lag x[t - k] with NaN padding.
    #[test]
    fn test_lag_and_lag_padded_directions() {
        let y = [1.5, -0.25, 3.0, 2.25, 0.5];
        // lag: R's stats::lag values on the shifted time base, so element t is y[t + k] (k shorter).
        assert_eq!(lag(&y, 1).unwrap().values, y[1..].to_vec());
        assert_eq!(lag(&y, -2).unwrap().values, y[..3].to_vec());
        // lag_padded: the conventional lag y[t - k] (dplyr::lag), NaN padding, full length.
        let lp = lag_padded(&y, 1).unwrap();
        assert!(lp[0].is_nan());
        assert_eq!(lp[1..].to_vec(), y[..4].to_vec());
        let ld = lag_padded(&y, -2).unwrap();
        assert_eq!(ld[..3].to_vec(), y[2..].to_vec());
        assert!(ld[3].is_nan() && ld[4].is_nan());
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

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn arima_sim_absurd_n_is_an_error() {
        // Before the fix: "memory allocation of 9007199254741792 bytes failed", SIGABRT.
        let err = arima_sim(&[0.5], &[], 0, 1usize << 50, None, None, Some(1)).unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        // n + burn-in overflows (a debug-build panic before the fix)
        let err = arima_sim(&[0.5], &[], 0, usize::MAX, None, Some(1), Some(1)).unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
    }

    /// R 4.3.3: arima.sim(list(ar = 0.5), n = 0) -> Error: 'n' must be strictly positive
    #[test]
    fn arima_sim_rejects_zero_n() {
        let err = arima_sim(&[0.5], &[], 0, 0, None, None, Some(1)).unwrap_err();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. }),
            "{err}"
        );
    }

    /// Stops at the ARMA recursion's first check (t = 0), before integrating;
    /// `integrate_in_place_honours_cancellation` covers the integration passes.
    #[test]
    fn arima_sim_honours_cancellation() {
        let r = cancelled(|| arima_sim(&[0.5], &[], 3, 100, None, None, Some(1)));
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn integrate_in_place_honours_cancellation() {
        assert!(matches!(
            cancelled(|| integrate_in_place(&mut [1.0, 2.0], 3)),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn integrate_in_place_is_repeated_cumulative_sum() {
        let mut x = [1.5, -0.25, 2.0, 0.75];
        integrate_in_place(&mut x, 2).unwrap();
        // cumsum: 1.5, 1.25, 3.25, 4.0; cumsum again: 1.5, 2.75, 6.0, 10.0
        assert_eq!(x, [1.5, 2.75, 6.0, 10.0]);
        let mut y = [0.3, 0.4];
        integrate_in_place(&mut y, 0).unwrap();
        assert_eq!(y, [0.3, 0.4]);
    }

    /// Guard for the innovation buffer: with no burn-in and no AR/MA terms the
    /// series is the innovations, which must be the seeded RNG's draws in order.
    #[test]
    fn arima_sim_seeded_innovations_are_the_rng_draws_in_order() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = StdRng::seed_from_u64(11);
        let expected: Vec<f64> = (0..50).map(|_| StandardNormal.sample(&mut rng)).collect();
        let r = arima_sim(&[], &[], 0, 50, None, Some(0), Some(11)).unwrap();
        assert_eq!(r.values, expected);
    }

    #[test]
    fn arma_acf_and_psi_weights_refuse_absurd_lag_max() {
        // Before the fix: "memory allocation of 9007199254741000 bytes failed", SIGABRT.
        let err = arma_acf(&[0.5], &[], 1usize << 50, false).unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        let err = arma_acf(&[0.5], &[], usize::MAX, false).unwrap_err(); // lag_max + 1 overflows
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        let err = arma_to_ma(&[0.5], &[0.3], 1usize << 50).unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        let err = crate::resource::with_alloc_ceiling(Some(1 << 16), || {
            arma_to_ma(&[0.5], &[0.3], 100_000)
        })
        .unwrap_err();
        assert!(err.to_string().contains("lag_max"), "{err}");
    }

    #[test]
    fn arma_pacf_honours_cancellation() {
        // arma_autocorrelation calls checkpoint() once before the (r + 1)-equation solve, so a
        // cancelled budget stops it before Durbin-Levinson runs.
        // durbin_levinson_honours_cancellation covers the order loop.
        assert!(matches!(
            cancelled(|| arma_acf(&[0.5, 0.2], &[], 2_000, true)),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn durbin_levinson_honours_cancellation() {
        // p = 3, so the order loop runs and checks the budget at n = 2.
        assert!(matches!(
            cancelled(|| durbin_levinson_from_acf(&[1.0, 0.5, 0.25, 0.125])),
            Err(EconError::Cancelled(_))
        ));
    }

    /// The pre-change recursion, verbatim (clones phi every order).
    fn durbin_levinson_reference(acf: &[f64]) -> Vec<f64> {
        let p = acf.len() - 1; // acf[0] = 1
        if p == 0 {
            return vec![];
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

        pacf
    }

    #[test]
    fn durbin_levinson_buffer_reuse_is_bit_identical() {
        let acf = arma_acf(&[0.6, -0.3, 0.1], &[0.4], 300, false)
            .unwrap()
            .values;
        assert_eq!(
            durbin_levinson_from_acf(&acf).unwrap(),
            durbin_levinson_reference(&acf)
        );
    }

    /// R 4.3.3: diffinv(c(2,3,4), differences = 2)            -> 0 0 2 7 16
    ///          diffinv(c(2,3,4,5), lag = 2, differences = 2) -> 0 0 0 0 2 3 8 11
    /// (zero xi; non-zero xi is pinned in test_validate_diffinv_xi_ts_utils_t4)
    #[test]
    fn test_validate_diffinv_zero_xi_against_r() {
        assert_eq!(
            diffinv(&[2.0, 3.0, 4.0], 1, 2, None).unwrap().values,
            vec![0.0, 0.0, 2.0, 7.0, 16.0]
        );
        assert_eq!(
            diffinv(&[2.0, 3.0, 4.0, 5.0], 2, 2, None).unwrap().values,
            vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 8.0, 11.0]
        );
    }

    #[test]
    fn diffinv_refuses_impossible_sizes() {
        let x = [1.0, 2.0];
        let err = diffinv(&x, 1, 1usize << 50, None).unwrap_err(); // aborted before the fix
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        let err = diffinv(&x, usize::MAX, 2, None).unwrap_err(); // lag * differences overflows
        assert!(matches!(err, EconError::ResourceLimit(_)), "{err}");
        let err =
            crate::resource::with_alloc_ceiling(Some(1 << 16), || diffinv(&x, 100, 100, None))
                .unwrap_err();
        assert!(err.to_string().contains("integrated series"), "{err}"); // 10,002 values > 64 KiB
    }

    #[test]
    fn diffinv_honours_cancellation() {
        assert!(matches!(
            cancelled(|| diffinv(&[1.0, 2.0, 3.0], 1, 3, None)),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn embed_and_acf_to_ar_check_their_output_size() {
        let x: Vec<f64> = (0..5_000)
            .map(|t| (t as f64 * 0.37).sin() + 0.01 * ((t * 13 % 7) as f64))
            .collect();
        // 4,001 x 1,000 f64 = 32 MB against a 1 MiB ceiling
        let err =
            crate::resource::with_alloc_ceiling(Some(1 << 20), || embed(&x, 1_000)).unwrap_err();
        assert!(err.to_string().contains("embedding matrix"), "{err}");
        let err = crate::resource::with_alloc_ceiling(Some(1 << 20), || embed_array(&x, 1_000))
            .unwrap_err();
        assert!(err.to_string().contains("embedding matrix"), "{err}");
        // 200 orders -> 20,100 coefficients = 161 KB against a 64 KiB ceiling
        let acf: Vec<f64> = (0..201).map(|k| 0.9f64.powi(k)).collect();
        let err =
            crate::resource::with_alloc_ceiling(Some(1 << 16), || acf_to_ar(&acf)).unwrap_err();
        assert!(err.to_string().contains("AR coefficient matrix"), "{err}");
    }

    #[test]
    fn convolution_filter_honours_cancellation() {
        let x: Vec<f64> = (0..300)
            .map(|t| (t as f64 * 0.41).cos() + 0.003 * t as f64)
            .collect();
        let f = [0.25, 0.5, 0.25];
        for sides in [FilterSides::One, FilterSides::Two] {
            assert!(filter(&x, &f, FilterMethod::Convolution, sides, None).is_ok());
            assert!(matches!(
                cancelled(|| filter(&x, &f, FilterMethod::Convolution, sides, None)),
                Err(EconError::Cancelled(_))
            ));
        }
    }

    // Guards only that the row loops check the budget at all. The stride rule
    // (checkpoint_stride(dimension)) is covered by
    // resource::tests::checkpoint_stride_targets_65536_operations; a
    // pre-cancelled budget cannot tell strides apart, since every stride
    // checks row 0.
    #[test]
    fn embed_and_embed_array_honour_cancellation() {
        let x: Vec<f64> = (0..300)
            .map(|t| (t as f64 * 0.41).cos() + 0.003 * t as f64)
            .collect();
        assert!(matches!(
            cancelled(|| embed(&x, 4)),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            cancelled(|| embed_array(&x, 4)),
            Err(EconError::Cancelled(_))
        ));
    }

    #[test]
    fn acf_to_ar_honours_cancellation() {
        let acf: Vec<f64> = (0..50).map(|k| 0.9f64.powi(k)).collect();
        assert!(matches!(
            cancelled(|| acf_to_ar(&acf)),
            Err(EconError::Cancelled(_))
        ));
    }

    /// Guard for the in-place integration: d = 2 is exactly two cumulative sums of the d = 0 series.
    #[test]
    fn arima_sim_differencing_is_cumulative_sum_of_arma_part() {
        let base = arima_sim(&[0.5], &[0.3], 0, 200, None, Some(60), Some(7))
            .unwrap()
            .values;
        let d2 = arima_sim(&[0.5], &[0.3], 2, 200, None, Some(60), Some(7))
            .unwrap()
            .values;
        let mut expected = base;
        for _ in 0..2 {
            let mut run = 0.0;
            for v in expected.iter_mut() {
                run += *v;
                *v = run;
            }
        }
        assert_eq!(d2, expected);
    }
}
