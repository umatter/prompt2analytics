//! Holt-Winters Exponential Smoothing.
//!
//! Implements the Holt-Winters method for time series forecasting with level,
//! trend, and seasonal components. Supports both additive and multiplicative
//! seasonal models, as well as non-seasonal (Holt's linear) and simple
//! exponential smoothing variants.
//!
//! # References
//!
//! - Holt, C. C. (1957, reprinted 2004). "Forecasting Seasonals and Trends by
//!   Exponentially Weighted Moving Averages." *International Journal of Forecasting*,
//!   20(1), 5-10.
//! - Winters, P. R. (1960). "Forecasting Sales by Exponentially Weighted Moving
//!   Averages." *Management Science*, 6(3), 324-342.
//! - Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
//!   Practice* (3rd ed.). OTexts.
//! - R Documentation: `stats::HoltWinters()`

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};

/// Type of seasonal component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SeasonalType {
    /// Additive seasonality: Y = Level + Trend + Seasonal + Error
    #[default]
    Additive,
    /// Multiplicative seasonality: Y = (Level + Trend) * Seasonal * Error
    Multiplicative,
}

/// Configuration for Holt-Winters fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWintersConfig {
    /// Level smoothing parameter (0 < α < 1). None = optimize automatically.
    pub alpha: Option<f64>,
    /// Trend smoothing parameter (0 < β < 1). None = optimize, Some(false) = no trend.
    /// Use beta = None for automatic optimization, or explicitly set a value.
    /// Set to Some(0.0) or use_trend = false to disable trend component.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter (0 < γ < 1). None = optimize automatically.
    /// Set to Some(0.0) or use_seasonal = false to disable seasonal component.
    pub gamma: Option<f64>,
    /// Type of seasonal component.
    pub seasonal: SeasonalType,
    /// Seasonal period (e.g., 12 for monthly, 4 for quarterly, 7 for daily with weekly pattern).
    pub period: usize,
    /// Number of starting periods for initialization (minimum 2).
    pub start_periods: usize,
    /// Whether to include trend component.
    pub use_trend: bool,
    /// Whether to include seasonal component.
    pub use_seasonal: bool,
    /// Maximum iterations for parameter optimization.
    pub max_iter: usize,
    /// Convergence tolerance for optimization.
    pub tolerance: f64,
}

impl Default for HoltWintersConfig {
    fn default() -> Self {
        Self {
            alpha: None,
            beta: None,
            gamma: None,
            seasonal: SeasonalType::Additive,
            period: 12,
            start_periods: 2,
            use_trend: true,
            use_seasonal: true,
            max_iter: 1000,
            tolerance: 1e-8,
        }
    }
}

/// Coefficients from Holt-Winters model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWintersCoefficients {
    /// Final level value (a).
    pub level: f64,
    /// Final trend value (b). None if trend disabled.
    pub trend: Option<f64>,
    /// Final seasonal indices (s1, s2, ..., sp). None if seasonal disabled.
    pub seasonal: Option<Vec<f64>>,
}

/// Result from Holt-Winters fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWintersResult {
    /// Level smoothing parameter used.
    pub alpha: f64,
    /// Trend smoothing parameter used. None if trend disabled.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter used. None if seasonal disabled.
    pub gamma: Option<f64>,
    /// Type of seasonal model.
    pub seasonal_type: SeasonalType,
    /// Seasonal period.
    pub period: usize,
    /// Level component at each time point.
    pub level: Vec<f64>,
    /// Trend component at each time point. None if trend disabled.
    pub trend: Option<Vec<f64>>,
    /// Seasonal component at each time point. None if seasonal disabled.
    pub seasonal_component: Option<Vec<f64>>,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Residuals (y - fitted).
    pub residuals: Vec<f64>,
    /// Sum of squared errors.
    pub sse: f64,
    /// Model coefficients (final values).
    pub coefficients: HoltWintersCoefficients,
    /// Number of observations.
    pub n_obs: usize,
    /// Column name (if run from dataset).
    pub column: Option<String>,
}

impl HoltWintersResult {
    /// Generate forecasts for h periods ahead.
    pub fn forecast(&self, h: usize) -> EconResult<Vec<f64>> {
        holt_winters_forecast(self, h)
    }
}

/// Fit Holt-Winters exponential smoothing model.
///
/// # Arguments
/// * `y` - Time series data
/// * `config` - Configuration for the model
///
/// # Returns
/// `HoltWintersResult` containing fitted values, components, and parameters.
///
/// # Example
/// ```ignore
/// use p2a_core::forecasting::{holt_winters, HoltWintersConfig, SeasonalType};
///
/// let y = vec![112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0,
///              115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0];
/// let config = HoltWintersConfig {
///     period: 12,
///     seasonal: SeasonalType::Additive,
///     ..Default::default()
/// };
/// let result = holt_winters(&y, config)?;
/// let forecast = result.forecast(12)?;
/// ```
pub fn holt_winters(y: &[f64], config: HoltWintersConfig) -> EconResult<HoltWintersResult> {
    let n = y.len();
    let period = config.period;

    // Validation
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "Holt-Winters requires at least 3 observations".to_string(),
        });
    }

    if config.use_seasonal && n < period * 2 {
        return Err(EconError::InsufficientData {
            required: period * 2,
            provided: n,
            context: "Seasonal Holt-Winters requires at least 2 full seasonal cycles".to_string(),
        });
    }

    if config.use_seasonal && period < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Seasonal period must be at least 2".to_string(),
        });
    }

    // Check for non-positive values in multiplicative model
    if config.use_seasonal
        && config.seasonal == SeasonalType::Multiplicative
        && y.iter().any(|&v| v <= 0.0)
    {
        return Err(EconError::InvalidSpecification {
            message: "Multiplicative seasonal model requires all positive values".to_string(),
        });
    }

    // Determine which parameters to optimize
    let optimize_alpha = config.alpha.is_none();
    let optimize_beta = config.use_trend && config.beta.is_none();
    let optimize_gamma = config.use_seasonal && config.gamma.is_none();

    // Initial parameters for optimization
    let mut alpha = config.alpha.unwrap_or(0.3);
    let mut beta = if config.use_trend {
        config.beta.unwrap_or(0.1)
    } else {
        0.0
    };
    let mut gamma = if config.use_seasonal {
        config.gamma.unwrap_or(0.1)
    } else {
        0.0
    };

    // Initialize level, trend, and seasonal components
    let (l_start, b_start, s_start) = initialize_components(
        y,
        period,
        config.use_trend,
        config.use_seasonal,
        config.seasonal,
        config.start_periods,
    )?;

    // Optimize parameters if needed
    if optimize_alpha || optimize_beta || optimize_gamma {
        (alpha, beta, gamma) = optimize_parameters(
            y,
            period,
            &l_start,
            &b_start,
            &s_start,
            alpha,
            beta,
            gamma,
            config.use_trend,
            config.use_seasonal,
            config.seasonal,
            optimize_alpha,
            optimize_beta,
            optimize_gamma,
            config.max_iter,
            config.tolerance,
        )?;
    }

    // Run the filter with optimized parameters
    let (level, trend, seasonal, fitted, residuals, sse) = run_filter(
        y,
        period,
        &l_start,
        &b_start,
        &s_start,
        alpha,
        beta,
        gamma,
        config.use_trend,
        config.use_seasonal,
        config.seasonal,
    )?;

    // For seasonal models, run_filter returns n-period values for level/trend
    // and we need to prepend NaN values to match the original series length
    let (full_level, full_fitted, full_residuals) = if config.use_seasonal {
        let mut full_level = vec![f64::NAN; period];
        full_level.extend(level.iter().copied());

        let mut full_fitted = vec![f64::NAN; period];
        full_fitted.extend(fitted.iter().copied());

        let mut full_residuals = vec![f64::NAN; period];
        full_residuals.extend(residuals.iter().copied());

        (full_level, full_fitted, full_residuals)
    } else {
        (level.clone(), fitted.clone(), residuals.clone())
    };

    let full_trend = if config.use_trend {
        if config.use_seasonal {
            let mut full_t = vec![f64::NAN; period];
            if let Some(ref t) = trend {
                full_t.extend(t.iter().copied());
            }
            Some(full_t)
        } else {
            trend.clone()
        }
    } else {
        None
    };

    // Extract final coefficients
    let final_level = *level.last().unwrap_or(&l_start);
    let final_trend = if config.use_trend {
        Some(*trend.as_ref().and_then(|t| t.last()).unwrap_or(&b_start))
    } else {
        None
    };
    let final_seasonal = if config.use_seasonal {
        // Get the last period's worth of seasonal values
        let s = seasonal.as_ref().unwrap();
        let start_idx = s.len().saturating_sub(period);
        Some(s[start_idx..].to_vec())
    } else {
        None
    };

    Ok(HoltWintersResult {
        alpha,
        beta: if config.use_trend { Some(beta) } else { None },
        gamma: if config.use_seasonal {
            Some(gamma)
        } else {
            None
        },
        seasonal_type: config.seasonal,
        period,
        level: full_level,
        trend: full_trend,
        seasonal_component: seasonal,
        fitted: full_fitted,
        residuals: full_residuals,
        sse,
        coefficients: HoltWintersCoefficients {
            level: final_level,
            trend: final_trend,
            seasonal: final_seasonal,
        },
        n_obs: n,
        column: None,
    })
}

/// Generate forecasts from a fitted Holt-Winters model.
///
/// # Arguments
/// * `result` - Fitted Holt-Winters result
/// * `h` - Forecast horizon (number of periods ahead)
///
/// # Returns
/// Vector of h forecast values.
pub fn holt_winters_forecast(result: &HoltWintersResult, h: usize) -> EconResult<Vec<f64>> {
    if h == 0 {
        return Ok(vec![]);
    }

    let a = result.coefficients.level;
    let b = result.coefficients.trend.unwrap_or(0.0);
    let period = result.period;

    let mut forecasts = Vec::with_capacity(h);

    for i in 1..=h {
        let base = a + (i as f64) * b;

        let forecast = if let Some(ref seasonal) = result.coefficients.seasonal {
            // Get the appropriate seasonal index
            // Seasonal indices cycle: s[t-p+1+(h-1) mod p]
            let idx = (i - 1) % period;
            match result.seasonal_type {
                SeasonalType::Additive => base + seasonal[idx],
                SeasonalType::Multiplicative => base * seasonal[idx],
            }
        } else {
            base
        };

        forecasts.push(forecast);
    }

    Ok(forecasts)
}

/// Run Holt-Winters on a dataset column.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series
/// * `column` - Column name with time series values
/// * `period` - Seasonal period
/// * `seasonal` - Type of seasonality (additive or multiplicative)
/// * `alpha` - Level smoothing parameter (None = optimize)
/// * `beta` - Trend smoothing parameter (None = optimize, Some(0) = no trend)
/// * `gamma` - Seasonal smoothing parameter (None = optimize, Some(0) = no seasonal)
pub fn run_holt_winters(
    dataset: &Dataset,
    column: &str,
    period: usize,
    seasonal: SeasonalType,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
) -> EconResult<HoltWintersResult> {
    // Extract data
    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let col = df.column(column).map_err(|_| EconError::ColumnNotFound {
        column: column.to_string(),
        available: available_cols.clone(),
    })?;

    let y: Vec<f64> = col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: column.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Determine if trend/seasonal should be used
    let use_trend = beta.map(|b| b > 0.0).unwrap_or(true);
    let use_seasonal = gamma.map(|g| g > 0.0).unwrap_or(period >= 2);

    let config = HoltWintersConfig {
        alpha,
        beta: if use_trend { beta } else { Some(0.0) },
        gamma: if use_seasonal { gamma } else { Some(0.0) },
        seasonal,
        period,
        use_trend,
        use_seasonal,
        ..Default::default()
    };

    let mut result = holt_winters(&y, config)?;
    result.column = Some(column.to_string());

    Ok(result)
}

// ============================================================================
// Internal Functions
// ============================================================================

/// Initialize level, trend, and seasonal components using R-compatible method.
///
/// R's HoltWinters uses the following initialization for seasonal models:
/// 1. Decompose the first `start_periods * period` observations using centered moving average
/// 2. Extract trend component (has NAs at edges due to centering)
/// 3. Fit linear regression on non-NA trend values: trend ~ 1:n
/// 4. l_start = intercept, b_start = slope
/// 5. s_start = seasonal component from decompose
///
/// This matches R's HoltWinters() initialization exactly.
fn initialize_components(
    y: &[f64],
    period: usize,
    use_trend: bool,
    use_seasonal: bool,
    seasonal_type: SeasonalType,
    start_periods: usize,
) -> EconResult<(f64, f64, Vec<f64>)> {
    let n = y.len();

    if !use_seasonal {
        // Non-seasonal initialization: R uses x[1] for level, x[2]-x[1] for trend
        let l0 = y[0];
        let b0 = if use_trend && n >= 2 {
            y[1] - y[0]
        } else {
            0.0
        };
        return Ok((l0, b0, vec![]));
    }

    // Need at least 2 periods for seasonal initialization
    let min_obs = start_periods * period;
    if n < min_obs {
        return Err(EconError::InsufficientData {
            required: min_obs,
            provided: n,
            context: format!(
                "Seasonal Holt-Winters requires at least {} observations ({} periods)",
                min_obs, start_periods
            ),
        });
    }

    // Use first start_periods * period observations for initialization
    let init_data = &y[..min_obs];

    // R's decompose: compute centered moving average for trend
    // For even period, R uses a 2xperiod MA (weighted)
    let trend = centered_moving_average(init_data, period);

    // Extract non-NA trend values (edges have NAs due to centering)
    let trend_valid: Vec<(usize, f64)> = trend
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_nan())
        .map(|(i, t)| (i, *t))
        .collect();

    if trend_valid.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Could not compute trend for initialization".to_string(),
        });
    }

    // Fit linear regression: trend ~ 1:n (1-indexed time)
    // This is R's approach: .lm.fit(cbind(1, seq_along(dat)), dat)
    let (l_start, b_start) = if use_trend {
        fit_trend_regression(&trend_valid)
    } else {
        // No trend: just use mean of trend values
        let mean_trend = trend_valid.iter().map(|(_, t)| t).sum::<f64>() / trend_valid.len() as f64;
        (mean_trend, 0.0)
    };

    // Compute seasonal indices: y - trend for additive, y / trend for multiplicative
    let mut s_init = vec![0.0; period];
    let mut s_count = vec![0usize; period];

    for (i, &yi) in init_data.iter().enumerate() {
        let ti = trend[i];
        if ti.is_nan() {
            continue;
        }
        let j = i % period;
        let seasonal_val = match seasonal_type {
            SeasonalType::Additive => yi - ti,
            SeasonalType::Multiplicative => {
                if ti > 0.0 {
                    yi / ti
                } else {
                    1.0
                }
            }
        };
        s_init[j] += seasonal_val;
        s_count[j] += 1;
    }

    // Average the seasonal values
    for j in 0..period {
        if s_count[j] > 0 {
            s_init[j] /= s_count[j] as f64;
        }
    }

    // Normalize seasonal: sum to 0 for additive, average to 1 for multiplicative
    match seasonal_type {
        SeasonalType::Additive => {
            let s_mean: f64 = s_init.iter().sum::<f64>() / period as f64;
            for s in &mut s_init {
                *s -= s_mean;
            }
        }
        SeasonalType::Multiplicative => {
            let s_mean: f64 = s_init.iter().sum::<f64>() / period as f64;
            if s_mean > 0.0 {
                for s in &mut s_init {
                    *s /= s_mean;
                }
            }
        }
    }

    Ok((l_start, b_start, s_init))
}

/// Compute centered moving average (R's decompose approach).
///
/// For even period p, R uses a 2×p weighted MA where the first and last
/// weights are 0.5 and the rest are 1. This is equivalent to a (1/2p) weighted
/// sum: 0.5*y[t-p/2] + y[t-p/2+1] + ... + y[t+p/2-1] + 0.5*y[t+p/2].
fn centered_moving_average(y: &[f64], period: usize) -> Vec<f64> {
    let n = y.len();
    let mut trend = vec![f64::NAN; n];

    if period == 0 || n < period {
        return trend;
    }

    let half = period / 2;

    // For even period, the centered MA at position t requires:
    // - (period/2) observations before t
    // - (period/2) observations after t
    // This means valid range is [period/2, n - period/2 - 1]

    for t in half..(n - half) {
        let mut sum = 0.0;

        if period % 2 == 0 {
            // Even period: use 2×period weighted MA
            // Weights: 0.5 at edges, 1.0 in middle
            sum += 0.5 * y[t - half];
            for i in (t - half + 1)..(t + half) {
                sum += y[i];
            }
            sum += 0.5 * y[t + half];
            trend[t] = sum / period as f64;
        } else {
            // Odd period: simple centered MA
            for i in (t - half)..=(t + half) {
                sum += y[i];
            }
            trend[t] = sum / period as f64;
        }
    }

    trend
}

/// Fit simple linear regression: y = a + b*t
///
/// Takes (index, value) pairs where index is 0-based.
/// Returns (intercept, slope) fitted as: y ~ 1:n (1-indexed time).
fn fit_trend_regression(data: &[(usize, f64)]) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 {
        return (data.first().map(|(_, v)| *v).unwrap_or(0.0), 0.0);
    }

    // R uses 1-indexed time: seq_along(dat) = 1, 2, 3, ...
    // So we use (i + 1) as the time value
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_ty = 0.0;

    for (idx, (_, y)) in data.iter().enumerate() {
        let t = (idx + 1) as f64; // 1-indexed time
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }

    // OLS: b = (n*sum_ty - sum_t*sum_y) / (n*sum_tt - sum_t*sum_t)
    //      a = (sum_y - b*sum_t) / n
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-10 {
        return (sum_y / n, 0.0);
    }

    let b = (n * sum_ty - sum_t * sum_y) / denom;
    let a = (sum_y - b * sum_t) / n;

    (a, b)
}

/// Run the Holt-Winters filter with given parameters.
///
/// Implements R's exact filtering algorithm:
/// - For seasonal models, filtering starts at observation `period`
/// - Uses circular buffer for seasonal indices (like R)
/// - One-step-ahead forecast: a[t-1] + b[t-1] + s[j] where j = t mod period
fn run_filter(
    y: &[f64],
    period: usize,
    l_start: &f64,
    b_start: &f64,
    s_start: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    use_trend: bool,
    use_seasonal: bool,
    seasonal_type: SeasonalType,
) -> EconResult<(
    Vec<f64>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    f64,
)> {
    let n = y.len();

    // For seasonal models, R's HoltWinters starts filtering at observation `period`
    let start_t = if use_seasonal { period } else { 0 };

    let mut level = Vec::with_capacity(n - start_t);
    let mut trend = if use_trend {
        Some(Vec::with_capacity(n - start_t))
    } else {
        None
    };

    // Use circular buffer for seasonal (like R)
    let mut s_buffer: Vec<f64> = if use_seasonal {
        s_start.to_vec()
    } else {
        vec![]
    };

    // Store all seasonal values for output (including initial + updated)
    let mut seasonal_history = if use_seasonal {
        Some(s_start.to_vec())
    } else {
        None
    };

    let mut fitted = Vec::with_capacity(n - start_t);
    let mut residuals = Vec::with_capacity(n - start_t);
    let mut sse = 0.0;

    // Previous state
    let mut l_prev = *l_start;
    let mut b_prev = *b_start;

    for t in start_t..n {
        // Position within seasonal cycle
        let j = t % period;

        // Get seasonal index from circular buffer
        let s_prev = if use_seasonal {
            s_buffer[j]
        } else {
            match seasonal_type {
                SeasonalType::Additive => 0.0,
                SeasonalType::Multiplicative => 1.0,
            }
        };

        // One-step-ahead forecast
        let forecast = match seasonal_type {
            SeasonalType::Additive => l_prev + (if use_trend { b_prev } else { 0.0 }) + s_prev,
            SeasonalType::Multiplicative => {
                (l_prev + (if use_trend { b_prev } else { 0.0 })) * s_prev
            }
        };

        fitted.push(forecast);
        let resid = y[t] - forecast;
        residuals.push(resid);
        sse += resid * resid;

        // Update level
        let l_new = match seasonal_type {
            SeasonalType::Additive => {
                alpha * (y[t] - s_prev)
                    + (1.0 - alpha) * (l_prev + if use_trend { b_prev } else { 0.0 })
            }
            SeasonalType::Multiplicative => {
                let y_deseason = if s_prev > 0.0 { y[t] / s_prev } else { y[t] };
                alpha * y_deseason + (1.0 - alpha) * (l_prev + if use_trend { b_prev } else { 0.0 })
            }
        };

        // Update trend
        let b_new = if use_trend {
            beta * (l_new - l_prev) + (1.0 - beta) * b_prev
        } else {
            0.0
        };

        // Update seasonal (in-place in circular buffer)
        if use_seasonal {
            let s_new = match seasonal_type {
                SeasonalType::Additive => gamma * (y[t] - l_new) + (1.0 - gamma) * s_prev,
                SeasonalType::Multiplicative => {
                    let ratio = if l_new > 0.0 { y[t] / l_new } else { 1.0 };
                    gamma * ratio + (1.0 - gamma) * s_prev
                }
            };
            s_buffer[j] = s_new;
            seasonal_history.as_mut().unwrap().push(s_new);
        }

        level.push(l_new);
        if let Some(ref mut t_vec) = trend {
            t_vec.push(b_new);
        }

        l_prev = l_new;
        b_prev = b_new;
    }

    Ok((level, trend, seasonal_history, fitted, residuals, sse))
}

/// Optimize smoothing parameters by minimizing SSE using Nelder-Mead.
///
/// Nelder-Mead is a derivative-free simplex method that typically converges
/// in 50-150 function evaluations for 3 parameters, much faster than
/// coordinate descent for smooth objective functions.
#[allow(clippy::too_many_arguments)]
fn optimize_parameters(
    y: &[f64],
    period: usize,
    l_start: &f64,
    b_start: &f64,
    s_start: &[f64],
    init_alpha: f64,
    init_beta: f64,
    init_gamma: f64,
    use_trend: bool,
    use_seasonal: bool,
    seasonal_type: SeasonalType,
    optimize_alpha: bool,
    optimize_beta: bool,
    optimize_gamma: bool,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(f64, f64, f64)> {
    let alpha = init_alpha.clamp(0.01, 0.99);
    let beta = init_beta.clamp(0.01, 0.99);
    let gamma = init_gamma.clamp(0.01, 0.99);

    // Build the objective function that evaluates SSE
    let objective = |params: &[f64]| -> f64 {
        let a = params[0].clamp(0.01, 0.99);
        let b = params[1].clamp(0.01, 0.99);
        let g = params[2].clamp(0.01, 0.99);

        match run_filter(
            y,
            period,
            l_start,
            b_start,
            s_start,
            a,
            b,
            g,
            use_trend,
            use_seasonal,
            seasonal_type,
        ) {
            Ok((_, _, _, _, _, sse)) => sse,
            Err(_) => f64::MAX,
        }
    };

    // Determine which parameters to optimize
    let n_params = optimize_alpha as usize + optimize_beta as usize + optimize_gamma as usize;

    if n_params == 0 {
        // Nothing to optimize
        return Ok((alpha, beta, gamma));
    }

    // Build initial point and bounds for parameters being optimized
    let mut initial = Vec::with_capacity(3);
    let mut bounds = Vec::with_capacity(3);
    let mut param_mask = [false, false, false]; // which params are being optimized

    if optimize_alpha {
        initial.push(alpha);
        bounds.push((0.01, 0.99));
        param_mask[0] = true;
    }
    if optimize_beta && use_trend {
        initial.push(beta);
        bounds.push((0.01, 0.99));
        param_mask[1] = true;
    }
    if optimize_gamma && use_seasonal {
        initial.push(gamma);
        bounds.push((0.01, 0.99));
        param_mask[2] = true;
    }

    // Wrapper to map reduced parameter vector to full [alpha, beta, gamma]
    let full_objective = |reduced_params: &[f64]| -> f64 {
        let mut full = [alpha, beta, gamma];
        let mut idx = 0;
        for (i, &optimizing) in param_mask.iter().enumerate() {
            if optimizing {
                full[i] = reduced_params[idx];
                idx += 1;
            }
        }
        objective(&full)
    };

    // Run Nelder-Mead optimization
    let (optimized, _best_sse) =
        nelder_mead(&initial, &bounds, full_objective, max_iter, tolerance);

    // Extract optimized parameters
    let mut result = [alpha, beta, gamma];
    let mut idx = 0;
    for (i, &optimizing) in param_mask.iter().enumerate() {
        if optimizing {
            result[i] = optimized[idx].clamp(0.01, 0.99);
            idx += 1;
        }
    }

    Ok((result[0], result[1], result[2]))
}

/// Nelder-Mead simplex optimization for box-constrained problems.
///
/// This is a derivative-free optimization method well-suited for smooth
/// functions with few parameters (like our 3-parameter Holt-Winters).
///
/// # Arguments
/// * `initial` - Starting point [alpha, beta, gamma]
/// * `bounds` - Box constraints [(min, max), ...] for each parameter
/// * `f` - Objective function to minimize
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// Optimized parameters and final function value
fn nelder_mead<F>(
    initial: &[f64],
    bounds: &[(f64, f64)],
    f: F,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = initial.len();

    // Nelder-Mead parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    // Helper to clamp point to bounds
    let clamp = |point: &[f64]| -> Vec<f64> {
        point
            .iter()
            .zip(bounds.iter())
            .map(|(&x, &(lo, hi))| x.clamp(lo, hi))
            .collect()
    };

    // Initialize simplex with n+1 vertices
    // Vertex 0 is the initial point, others are perturbed
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(clamp(initial));

    for i in 0..n {
        let mut vertex = initial.to_vec();
        // Perturb in dimension i by 5% of range or 0.05 if initial is 0
        let range = bounds[i].1 - bounds[i].0;
        let delta = if initial[i].abs() < 1e-10 {
            0.05 * range
        } else {
            0.05 * range.min(initial[i].abs())
        };
        vertex[i] += delta;
        simplex.push(clamp(&vertex));
    }

    // Evaluate function at all vertices
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort vertices by function value (best first)
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        let f_best = values[best_idx];
        let f_worst = values[worst_idx];
        let f_second_worst = values[second_worst_idx];

        // Check convergence: simplex is small enough
        let spread = f_worst - f_best;
        if spread < tol && spread.abs() < tol * f_best.abs().max(1.0) {
            return (simplex[best_idx].clone(), f_best);
        }

        // Compute centroid of all vertices except worst
        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection: x_r = centroid + alpha * (centroid - worst)
        let x_r: Vec<f64> = clamp(
            &centroid
                .iter()
                .zip(simplex[worst_idx].iter())
                .map(|(&c, &w)| c + alpha * (c - w))
                .collect::<Vec<_>>(),
        );
        let f_r = f(&x_r);

        if f_r < f_second_worst && f_r >= f_best {
            // Accept reflection
            simplex[worst_idx] = x_r;
            values[worst_idx] = f_r;
            continue;
        }

        if f_r < f_best {
            // Try expansion: x_e = centroid + gamma * (x_r - centroid)
            let x_e: Vec<f64> = clamp(
                &centroid
                    .iter()
                    .zip(x_r.iter())
                    .map(|(&c, &r)| c + gamma * (r - c))
                    .collect::<Vec<_>>(),
            );
            let f_e = f(&x_e);

            if f_e < f_r {
                // Accept expansion
                simplex[worst_idx] = x_e;
                values[worst_idx] = f_e;
            } else {
                // Accept reflection
                simplex[worst_idx] = x_r;
                values[worst_idx] = f_r;
            }
            continue;
        }

        // f_r >= f_second_worst, try contraction
        if f_r < f_worst {
            // Outside contraction: x_c = centroid + rho * (x_r - centroid)
            let x_c: Vec<f64> = clamp(
                &centroid
                    .iter()
                    .zip(x_r.iter())
                    .map(|(&c, &r)| c + rho * (r - c))
                    .collect::<Vec<_>>(),
            );
            let f_c = f(&x_c);

            if f_c < f_r {
                simplex[worst_idx] = x_c;
                values[worst_idx] = f_c;
                continue;
            }
        } else {
            // Inside contraction: x_c = centroid - rho * (centroid - worst)
            let x_c: Vec<f64> = clamp(
                &centroid
                    .iter()
                    .zip(simplex[worst_idx].iter())
                    .map(|(&c, &w)| c - rho * (c - w))
                    .collect::<Vec<_>>(),
            );
            let f_c = f(&x_c);

            if f_c < f_worst {
                simplex[worst_idx] = x_c;
                values[worst_idx] = f_c;
                continue;
            }
        }

        // Shrink: move all vertices except best towards best
        for &idx in &indices[1..] {
            for j in 0..n {
                simplex[idx][j] =
                    simplex[best_idx][j] + sigma * (simplex[idx][j] - simplex[best_idx][j]);
            }
            simplex[idx] = clamp(&simplex[idx]);
            values[idx] = f(&simplex[idx]);
        }
    }

    // Return best found
    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    (simplex[best_idx].clone(), values[best_idx])
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
    fn test_simple_exponential_smoothing() {
        // Test without trend or seasonality
        let y: Vec<f64> = vec![10.0, 12.0, 14.0, 13.0, 15.0, 17.0, 16.0, 18.0, 20.0, 19.0];

        let config = HoltWintersConfig {
            alpha: Some(0.3),
            beta: Some(0.0),
            gamma: Some(0.0),
            use_trend: false,
            use_seasonal: false,
            period: 1,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        assert_eq!(result.n_obs, 10);
        assert!(result.beta.is_none());
        assert!(result.gamma.is_none());
        assert!(result.trend.is_none());
        assert!(result.seasonal_component.is_none());
        assert_eq!(result.fitted.len(), 10);
        assert_eq!(result.residuals.len(), 10);
    }

    #[test]
    fn test_holt_linear() {
        // Test with trend but no seasonality
        let y: Vec<f64> = (0..20)
            .map(|i| 10.0 + 0.5 * i as f64 + 0.1 * (i % 3) as f64)
            .collect();

        let config = HoltWintersConfig {
            alpha: Some(0.4),
            beta: Some(0.2),
            use_trend: true,
            use_seasonal: false,
            period: 1,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.beta.is_some());
        assert!(result.trend.is_some());
        assert!(result.gamma.is_none());
        assert!(result.seasonal_component.is_none());

        // Forecast should continue trend
        let forecast = result.forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);
        // Forecasts should be increasing (positive trend)
        for i in 1..forecast.len() {
            assert!(forecast[i] > forecast[i - 1] - 1.0); // Allow some tolerance
        }
    }

    #[test]
    fn test_additive_seasonal() {
        // Create data with clear additive seasonal pattern
        let period = 4;
        let seasonal_pattern = [10.0, -5.0, 0.0, -5.0];
        let mut y = Vec::with_capacity(24);
        for i in 0..24 {
            let trend = 100.0 + 0.5 * i as f64;
            let seasonal = seasonal_pattern[i % period];
            y.push(trend + seasonal);
        }

        let config = HoltWintersConfig {
            alpha: Some(0.3),
            beta: Some(0.1),
            gamma: Some(0.2),
            seasonal: SeasonalType::Additive,
            period,
            use_trend: true,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        assert_eq!(result.n_obs, 24);
        assert!(result.gamma.is_some());
        assert!(result.seasonal_component.is_some());
        assert_eq!(result.period, 4);

        // Forecast
        let forecast = result.forecast(4).unwrap();
        assert_eq!(forecast.len(), 4);
    }

    #[test]
    fn test_multiplicative_seasonal() {
        // Create data with multiplicative seasonal pattern
        let period = 4;
        let seasonal_pattern = [1.2, 0.8, 1.0, 1.0];
        let mut y = Vec::with_capacity(24);
        for i in 0..24 {
            let trend = 100.0 + 2.0 * i as f64;
            let seasonal = seasonal_pattern[i % period];
            y.push(trend * seasonal);
        }

        let config = HoltWintersConfig {
            alpha: Some(0.3),
            beta: Some(0.1),
            gamma: Some(0.2),
            seasonal: SeasonalType::Multiplicative,
            period,
            use_trend: true,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        assert_eq!(result.n_obs, 24);
        assert_eq!(result.seasonal_type, SeasonalType::Multiplicative);
    }

    #[test]
    fn test_parameter_optimization() {
        // Test automatic parameter optimization
        let period = 4;
        let seasonal_pattern = [5.0, -3.0, 2.0, -4.0];
        let mut y = Vec::with_capacity(40);
        for i in 0..40 {
            let trend = 50.0 + 0.3 * i as f64;
            let seasonal = seasonal_pattern[i % period];
            y.push(trend + seasonal + 0.5 * ((i * 7) % 11) as f64 - 2.5); // Add some noise
        }

        let config = HoltWintersConfig {
            alpha: None, // Optimize
            beta: None,  // Optimize
            gamma: None, // Optimize
            seasonal: SeasonalType::Additive,
            period,
            use_trend: true,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        // Parameters should be reasonable
        assert!(result.alpha > 0.0 && result.alpha < 1.0);
        assert!(result.beta.unwrap() > 0.0 && result.beta.unwrap() < 1.0);
        assert!(result.gamma.unwrap() > 0.0 && result.gamma.unwrap() < 1.0);
    }

    #[test]
    fn test_insufficient_data() {
        let y = vec![1.0, 2.0];

        let config = HoltWintersConfig {
            period: 12,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiplicative_non_positive() {
        let y = vec![10.0, -5.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];

        let config = HoltWintersConfig {
            seasonal: SeasonalType::Multiplicative,
            period: 4,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_holt_winters_against_r() {
        // Test case: AirPassengers-like data (first 24 months)
        // R code:
        // > ap <- AirPassengers[1:24]
        // > m <- HoltWinters(ts(ap, frequency=12), alpha=0.2, beta=0.1, gamma=0.3)
        // > m$SSE  # 369.5158
        // > m$coefficients["a"]  # 147.9523
        // > m$coefficients["b"]  # 1.66295
        let y = vec![
            112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0,
            115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0,
        ];

        let config = HoltWintersConfig {
            alpha: Some(0.2),
            beta: Some(0.1),
            gamma: Some(0.3),
            seasonal: SeasonalType::Additive,
            period: 12,
            use_trend: true,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 24);
        assert_eq!(result.period, 12);
        assert!(approx_eq(result.alpha, 0.2, 0.001));

        // Check SSE matches R exactly
        // R's SSE: 369.5158
        assert!(
            approx_eq(result.sse, 369.5158, 0.01),
            "SSE mismatch: Rust={}, R=369.5158",
            result.sse
        );

        // Check final coefficients match R
        // R: a = 147.9523, b = 1.66295
        assert!(
            approx_eq(result.coefficients.level, 147.9523, 0.01),
            "Level mismatch: Rust={}, R=147.9523",
            result.coefficients.level
        );
        assert!(
            approx_eq(result.coefficients.trend.unwrap(), 1.66295, 0.01),
            "Trend mismatch: Rust={:?}, R=1.66295",
            result.coefficients.trend
        );

        // Fitted values should be close to original (filter out NaN values)
        let valid_residuals: Vec<f64> = result
            .residuals
            .iter()
            .filter(|r| !r.is_nan())
            .copied()
            .collect();
        let mean_abs_error: f64 =
            valid_residuals.iter().map(|r| r.abs()).sum::<f64>() / valid_residuals.len() as f64;
        assert!(
            mean_abs_error < 30.0,
            "Mean absolute error too high: {}",
            mean_abs_error
        );
    }

    #[test]
    fn test_forecast() {
        let y: Vec<f64> = (0..40)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = vec![10.0, -10.0, 5.0, -5.0][i % 4];
                trend + seasonal
            })
            .collect();

        let config = HoltWintersConfig {
            alpha: Some(0.3),
            beta: Some(0.1),
            gamma: Some(0.2),
            seasonal: SeasonalType::Additive,
            period: 4,
            use_trend: true,
            use_seasonal: true,
            ..Default::default()
        };

        let result = holt_winters(&y, config).unwrap();
        let forecast = result.forecast(8).unwrap();

        assert_eq!(forecast.len(), 8);

        // Forecasts should show the seasonal pattern
        // (repeating with period 4)
        for i in 0..4 {
            // Each pair of forecasts separated by period should be similar
            // relative to trend
            let diff1 = forecast[i + 4] - forecast[i];
            // The difference should be approximately the trend component * 4
            assert!(diff1 > 0.0); // Should be increasing due to trend
        }
    }
}
