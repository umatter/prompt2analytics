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
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_with_capacity};
use crate::traits::estimator::{validate_level, z_critical};

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
    /// Level smoothing parameter (0 < α <= 1). None = optimize automatically.
    pub alpha: Option<f64>,
    /// Trend smoothing parameter (0 <= β <= 1), used with `use_trend`. None = optimize.
    /// Some(0.0) keeps a trend that is not updated (R's beta = 0); `use_trend = false` has none.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter (0 <= γ <= 1), used with `use_seasonal`. None = optimize.
    /// Some(0.0) keeps a seasonal component that is not updated (R's gamma = 0);
    /// `use_seasonal = false` has none.
    pub gamma: Option<f64>,
    /// Type of seasonal component.
    pub seasonal: SeasonalType,
    /// Seasonal period (e.g., 12 for monthly, 4 for quarterly, 7 for daily with weekly pattern):
    /// at least 1, and at least 2 for a seasonal model.
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
    /// Level component at each time point; NaN before the first filtered observation, as in
    /// `fitted`, `residuals` and `trend`. The filter starts as R's `HoltWinters`: at t = `period`
    /// (0-based) for a seasonal model, t = 2 for Holt's linear trend, t = 1 for simple exponential
    /// smoothing.
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

    /// Forecasts with prediction intervals; see [`holt_winters_predict`].
    pub fn predict(&self, horizon: usize, level: f64) -> EconResult<HoltWintersForecast> {
        holt_winters_predict(self, horizon, level)
    }

    /// R's `var(residuals(fit))`: the sample variance (divisor m - 1) of the m one-step
    /// errors, for every seasonal type; `None` with fewer than two errors. The prediction
    /// intervals of [`holt_winters_predict`] scale it.
    pub fn residual_variance(&self) -> Option<f64> {
        let (mut count, mut mean) = (0usize, 0.0);
        for r in self.residuals.iter().filter(|r| !r.is_nan()) {
            count += 1;
            mean += r;
        }
        if count < 2 {
            return None;
        }
        mean /= count as f64;
        let ss = self
            .residuals
            .iter()
            .filter(|r| !r.is_nan())
            .map(|r| (r - mean).powi(2))
            .sum::<f64>();
        Some(ss / (count - 1) as f64)
    }
}

/// Forecasts with prediction intervals from [`holt_winters_predict`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWintersForecast {
    /// Number of periods ahead.
    pub horizon: usize,
    /// Confidence level of the intervals.
    pub level: f64,
    /// R `predict()[, "fit"]`.
    pub forecast: Vec<f64>,
    /// Standard error of each forecast; the interval is
    /// forecast -/+ qnorm((1 + level) / 2) * std_errors.
    pub std_errors: Vec<f64>,
    /// Lower limits.
    pub lower: Vec<f64>,
    /// Upper limits.
    pub upper: Vec<f64>,
    /// var(residuals(fit)): sample variance (divisor m - 1) of the m one-step errors.
    pub residual_variance: f64,
}

/// Fit Holt-Winters exponential smoothing model.
///
/// # Arguments
/// * `y` - Time series data
/// * `config` - Configuration for the model
///
/// # Returns
/// `HoltWintersResult` containing fitted values, components, and parameters,
/// or [`EconError::Cancelled`] when the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline (checked
/// once per optimiser iteration; when no smoothing parameter is left to
/// estimate, no optimiser runs and the budget is not checked).
///
/// # Errors
/// [`EconError::InvalidSpecification`], naming the parameter, for `period` (0; below 2 when
/// seasonal), `alpha` (not in (0, 1]), `beta` and `gamma` (not in [0, 1], when used),
/// `start_periods` (below 2 when seasonal), `max_iter` (0), `tolerance` (not finite and positive)
/// and a seasonal size that overflows; [`EconError::InsufficientData`] for a series shorter than
/// 3 values or two seasonal cycles.
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

    validate_config(&config, n)?;

    if config.use_seasonal && n < period * 2 {
        return Err(EconError::InsufficientData {
            required: period * 2,
            provided: n,
            context: "Seasonal Holt-Winters requires at least 2 full seasonal cycles".to_string(),
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
    let (l_start, b_start, s_start, start_t) = initialize_components(
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
            start_t,
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
        start_t,
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

    // run_filter returns the values from start_t on; the series' first start_t positions are NaN.
    let with_prefix = |v: &[f64]| {
        let mut full = vec![f64::NAN; start_t];
        full.extend_from_slice(v);
        full
    };
    let full_level = with_prefix(&level);
    let full_fitted = with_prefix(&fitted);
    let full_residuals = with_prefix(&residuals);
    let full_trend = trend.as_deref().map(with_prefix);

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

/// Check a configuration against R's rules, before any work.
fn validate_config(config: &HoltWintersConfig, n: usize) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    let period = config.period;

    if period == 0 {
        return invalid("period must be at least 1, got 0".to_string());
    }
    if config.use_seasonal {
        if period < 2 {
            return invalid(format!(
                "period must be at least 2 for a seasonal model, got {period}"
            ));
        }
        if config.start_periods < 2 {
            return invalid(format!(
                "start_periods must be at least 2 (need at least 2 periods to compute seasonal start values), got {}",
                config.start_periods
            ));
        }
        if period.checked_mul(2).is_none() || period.checked_mul(config.start_periods).is_none() {
            return invalid(format!(
                "period {period} with {} start periods is too large for {n} observations",
                config.start_periods
            ));
        }
    }
    if let Some(a) = config.alpha
        && !(a.is_finite() && a > 0.0 && a <= 1.0)
    {
        return invalid(format!("alpha must be in (0, 1], got {a}"));
    }
    if config.use_trend
        && let Some(b) = config.beta
        && !(0.0..=1.0).contains(&b)
    {
        return invalid(format!("beta must be in [0, 1], got {b}"));
    }
    if config.use_seasonal
        && let Some(g) = config.gamma
        && !(0.0..=1.0).contains(&g)
    {
        return invalid(format!("gamma must be in [0, 1], got {g}"));
    }
    if config.max_iter == 0 {
        return invalid("max_iter must be at least 1, got 0".to_string());
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return invalid(format!(
            "tolerance must be finite and positive, got {}",
            config.tolerance
        ));
    }
    Ok(())
}

/// Generate forecasts from a fitted Holt-Winters model.
///
/// # Arguments
/// * `result` - Fitted Holt-Winters result
/// * `h` - Forecast horizon (number of periods ahead)
///
/// # Returns
/// Vector of h forecast values, or [`EconError::InvalidSpecification`] for a result whose seasonal
/// coefficients do not fit its period.
pub fn holt_winters_forecast(result: &HoltWintersResult, h: usize) -> EconResult<Vec<f64>> {
    if let Some(s) = &result.coefficients.seasonal
        && (result.period == 0 || s.len() != result.period)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "HoltWintersResult is inconsistent: {} seasonal coefficients for period {}",
                s.len(),
                result.period
            ),
        });
    }

    if h == 0 {
        return Ok(vec![]);
    }

    let a = result.coefficients.level;
    let b = result.coefficients.trend.unwrap_or(0.0);
    let period = result.period;

    let mut forecasts = try_with_capacity(h, "the Holt-Winters forecast horizon")?;

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

/// Forecasts with prediction intervals, as `predict.HoltWinters(prediction.interval = TRUE)` for
/// additive and non-seasonal fits: the variance at horizon h is
/// var(residuals) (1 + sum_{j=1}^{h-1} psi_j^2), with
/// psi_j = alpha (1 + j beta) + 1{j mod f = 0} gamma (1 - alpha) (Yar and Chatfield).
///
/// Multiplicative seasonal intervals are refused: `predict.HoltWinters`' formula for them reads the
/// trend where a seasonal index belongs, puts alpha + gamma (1 - alpha) where the one-step term
/// is 1, and shifts the seasonal indices without a trend. A fit without a seasonal component uses
/// the additive formula whatever its label.
///
/// # Errors
/// [`EconError::InvalidSpecification`] for a horizon of 0, a level outside (0, 1), and a
/// multiplicative seasonal fit; [`EconError::InsufficientData`] for fewer than two one-step errors;
/// [`EconError::ResourceLimit`] for a horizon over the allocation ceiling; [`EconError::Cancelled`]
/// when the budget stops (checked once per horizon).
pub fn holt_winters_predict(
    result: &HoltWintersResult,
    horizon: usize,
    level: f64,
) -> EconResult<HoltWintersForecast> {
    validate_level(level, "level")?;
    if horizon == 0 {
        return Err(EconError::InvalidSpecification {
            message: "horizon must be at least 1".to_string(),
        });
    }
    if result.coefficients.seasonal.is_some()
        && result.seasonal_type == SeasonalType::Multiplicative
    {
        return Err(EconError::InvalidSpecification {
            message: "prediction intervals are implemented for additive and non-seasonal Holt-Winters fits; use holt_winters_forecast for multiplicative point forecasts".to_string(),
        });
    }
    let forecast = holt_winters_forecast(result, horizon)?;

    let residual_variance = match result.residual_variance() {
        Some(v) => v,
        None => {
            let provided = result.residuals.iter().filter(|r| !r.is_nan()).count();
            return Err(EconError::InsufficientData {
                required: 2,
                provided,
                context: "prediction intervals need the variance of at least two one-step errors"
                    .to_string(),
            });
        }
    };

    let alpha = result.alpha;
    let beta = result.beta.unwrap_or(0.0);
    let gamma = result.gamma.unwrap_or(0.0);
    let f = result.period.max(1);
    let z = z_critical(level)?;

    let what = "the Holt-Winters prediction horizon";
    let mut std_errors = try_with_capacity(horizon, what)?;
    let mut lower = try_with_capacity(horizon, what)?;
    let mut upper = try_with_capacity(horizon, what)?;
    let mut factor = 1.0;
    for (h, &fit) in forecast.iter().enumerate() {
        checkpoint()?;
        // h is 0-based: this step adds psi_h^2 to the variance of horizon h + 1.
        if h > 0 {
            let seasonal_term = if h % f == 0 {
                gamma * (1.0 - alpha)
            } else {
                0.0
            };
            let psi = alpha * (1.0 + h as f64 * beta) + seasonal_term;
            factor += psi * psi;
        }
        let se = (residual_variance * factor).sqrt();
        std_errors.push(se);
        lower.push(fit - z * se);
        upper.push(fit + z * se);
    }

    Ok(HoltWintersForecast {
        horizon,
        level,
        forecast,
        std_errors,
        lower,
        upper,
        residual_variance,
    })
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
///
/// # Returns
/// As [`holt_winters`], including [`EconError::Cancelled`] on a stopped budget.
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
pub fn run_holt_winters(
    dataset: &Dataset,
    column: &str,
    period: usize,
    seasonal: SeasonalType,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
) -> EconResult<HoltWintersResult> {
    let (y, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    // Determine if trend/seasonal should be used
    // (only an exact 0 turns a component off; an out-of-range value reaches the validation)
    let use_trend = beta.map(|b| b != 0.0).unwrap_or(true);
    let use_seasonal = gamma.map(|g| g != 0.0).unwrap_or(period >= 2);

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

/// Initialize level, trend, and seasonal components as R's `HoltWinters`, and return the first
/// filtered position (0-based) with them.
///
/// Holt's linear trend starts at t = 2 with l = x[1], b = x[1] - x[0]; simple exponential
/// smoothing at t = 1 with l = x[0]. A seasonal model starts at t = period, as follows:
/// 1. Decompose the first `start_periods * period` observations using centered moving average
/// 2. Extract trend component (has NAs at edges due to centering)
/// 3. Fit linear regression on non-NA trend values: trend ~ 1:n
/// 4. l_start = intercept, b_start = slope
/// 5. s_start = seasonal component from decompose
///
/// Without a trend, b_start is 0 and l_start stays the intercept.
fn initialize_components(
    y: &[f64],
    period: usize,
    use_trend: bool,
    use_seasonal: bool,
    seasonal_type: SeasonalType,
    start_periods: usize,
) -> EconResult<(f64, f64, Vec<f64>, usize)> {
    let n = y.len();

    if !use_seasonal {
        return Ok(if use_trend {
            (y[1], y[1] - y[0], vec![], 2)
        } else {
            (y[0], 0.0, vec![], 1)
        });
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
    let (l_start, slope) = fit_trend_regression(&trend_valid);
    let b_start = if use_trend { slope } else { 0.0 };

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

    Ok((l_start, b_start, s_init, period))
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
/// R's filtering algorithm, from observation `start_t` (see `initialize_components`):
/// - Uses circular buffer for seasonal indices (like R)
/// - One-step-ahead forecast: a[t-1] + b[t-1] + s[j] where j = t mod period
fn run_filter(
    y: &[f64],
    period: usize,
    start_t: usize,
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
        let j = if use_seasonal { t % period } else { 0 };

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
    start_t: usize,
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
    // Only what is optimised stays in the optimiser's box: a fixed value is used as given.
    let start = |init: f64, optimize: bool| {
        if optimize {
            init.clamp(0.01, 0.99)
        } else {
            init
        }
    };
    let alpha = start(init_alpha, optimize_alpha);
    let beta = start(init_beta, optimize_beta);
    let gamma = start(init_gamma, optimize_gamma);

    // Build the objective function that evaluates SSE
    let objective = |params: &[f64]| -> f64 {
        match run_filter(
            y,
            period,
            start_t,
            l_start,
            b_start,
            s_start,
            params[0],
            params[1],
            params[2],
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
                full[i] = reduced_params[idx].clamp(0.01, 0.99);
                idx += 1;
            }
        }
        objective(&full)
    };

    // Run Nelder-Mead optimization
    let (optimized, _best_sse) =
        nelder_mead(&initial, &bounds, full_objective, max_iter, tolerance)?;

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
/// Optimized parameters and final function value, or [`EconError::Cancelled`]
/// when the budget is cancelled or past its deadline (checked once per
/// iteration; each iteration runs `f`, one filter pass over the series, at
/// most `initial.len() + 2` times).
fn nelder_mead<F>(
    initial: &[f64],
    bounds: &[(f64, f64)],
    f: F,
    max_iter: usize,
    tol: f64,
) -> EconResult<(Vec<f64>, f64)>
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
        checkpoint()?;
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
            return Ok((simplex[best_idx].clone(), f_best));
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

    Ok((simplex[best_idx].clone(), values[best_idx]))
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

    /// The case's series, for tests that compare no R number.
    fn series_dataset() -> Dataset {
        let path =
            p2a_validation::repo_path("validation/datasets/generated/holtwinters_series.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    fn hw(
        y: &[f64],
        period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        seasonal: SeasonalType,
    ) -> HoltWintersResult {
        let mut c = HoltWintersConfig::default();
        c.period = period;
        c.seasonal = seasonal;
        c.alpha = Some(alpha);
        c.use_trend = beta.is_some();
        c.beta = beta;
        c.use_seasonal = gamma.is_some(); // Some(0.0) is R's gamma = 0 (seasonal, not updated)
        c.gamma = gamma;
        holt_winters(y, c).unwrap()
    }

    fn coef_names(r: &HoltWintersResult) -> Vec<String> {
        let mut v = vec!["a".to_string()];
        if r.coefficients.trend.is_some() {
            v.push("b".into());
        }
        if let Some(s) = &r.coefficients.seasonal {
            v.extend((1..=s.len()).map(|i| format!("s{i}")));
        }
        v
    }

    fn coef_values(r: &HoltWintersResult) -> Vec<f64> {
        let mut v = vec![r.coefficients.level];
        v.extend(r.coefficients.trend);
        if let Some(s) = &r.coefficients.seasonal {
            v.extend(s.iter().copied());
        }
        v
    }

    fn sample_variance(res: &[f64]) -> f64 {
        let m = res.iter().sum::<f64>() / res.len() as f64;
        res.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (res.len() - 1) as f64
    }

    /// SSE, coefficients, one-step error count and variance, 12-step forecasts with 95 % intervals.
    fn check_fit(case: &p2a_validation::Case, tag: &str, r: &HoltWintersResult, n: usize) {
        let p = holt_winters_predict(r, 12, 0.95).unwrap();
        let nres = r.residuals.iter().filter(|v| !v.is_nan()).count();
        Check::new(case)
            .scalar(&format!("{tag}_sse"), r.sse, Tol::stat(n))
            .named(
                &format!("{tag}_coef"),
                &coef_names(r),
                &coef_values(r),
                Tol::coef(n),
            )
            .scalar(&format!("{tag}_nres"), nres as f64, Tol::exact())
            .scalar(&format!("{tag}_resvar"), p.residual_variance, Tol::stat(n))
            .vector(&format!("{tag}_fit"), &p.forecast, Tol::coef(n))
            .vector(&format!("{tag}_upr"), &p.upper, Tol::stat(n))
            .vector(&format!("{tag}_lwr"), &p.lower, Tol::stat(n))
            .finish();
    }

    /// A fixed value stays as given while another parameter is optimised: at 1.0 and 0.0 it left
    /// the optimiser's box and came back as 0.99 and 0.01.
    #[test]
    fn test_fixed_parameters_are_not_clamped_while_optimising() {
        let ds = series_dataset();
        let y = col(&ds, "airpass");
        let fit = |alpha: Option<f64>, beta: Option<f64>, gamma: Option<f64>| {
            let mut c = HoltWintersConfig::default();
            c.period = 12;
            c.alpha = alpha;
            c.beta = beta;
            c.gamma = gamma;
            holt_winters(&y, c).unwrap()
        };

        let r = fit(Some(1.0), None, Some(0.2));
        assert_eq!(r.alpha, 1.0);
        assert_eq!(r.gamma, Some(0.2));
        for fixed in [0.0, 1.0] {
            let r = fit(None, Some(fixed), None);
            assert_eq!(r.beta, Some(fixed));
            let r = fit(None, None, Some(fixed));
            assert_eq!(r.gamma, Some(fixed));
        }

        // The reported fit uses the fixed values: refitting at the optimum reproduces the SSE.
        let opt = fit(Some(1.0), None, Some(0.0));
        let again = fit(Some(1.0), opt.beta, Some(0.0));
        assert_eq!(opt.sse.to_bits(), again.sse.to_bits());
    }

    /// The objective uses a fixed value as given, not the optimiser's edge. The first step of a
    /// noisy level series starts Holt's trend at 3. With a fixed beta of 0 that trend stays, and
    /// alpha must be large to keep the errors down (0.81 on a grid of this series). With a beta of
    /// 0.01 the trend decays, and an objective on that value would choose a small alpha (0.23), with
    /// several times the SSE at beta = 0.
    #[test]
    fn test_fixed_beta_of_zero_reaches_the_objective() {
        let mut seed = 20_261_107u64;
        let mut y: Vec<f64> = (0..5000)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) % (1 << 31);
                100.0 + 17.0 * (seed as f64 / (1u64 << 31) as f64 - 0.5)
            })
            .collect();
        y[0] = y[1] - 3.0;
        let fit = |alpha: Option<f64>| {
            let mut c = HoltWintersConfig::default();
            c.use_seasonal = false;
            c.alpha = alpha;
            c.beta = Some(0.0);
            holt_winters(&y, c).unwrap()
        };
        let opt = fit(None);
        assert_eq!(opt.beta, Some(0.0));
        let grid_best = (1..100)
            .map(|i| fit(Some(i as f64 / 100.0)).sse)
            .fold(f64::INFINITY, f64::min);
        assert!(
            opt.sse <= grid_best * 1.01,
            "alpha {}: {} vs {grid_best}",
            opt.alpha,
            opt.sse
        );
    }

    /// X-5: `run_holt_winters` sends period 0 to the non-seasonal path, and the filter took
    /// the seasonal position of every observation.
    #[test]
    fn test_holtwinters_rejects_period_0() {
        let ds = series_dataset();
        let err = run_holt_winters(&ds, "airpass", 0, SeasonalType::Additive, None, None, None)
            .unwrap_err();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. })
                && err.to_string().contains("period"),
            "{err}"
        );
        let y = col(&ds, "airpass");
        let mut c = HoltWintersConfig::default();
        c.period = 0;
        c.use_seasonal = false;
        c.alpha = Some(0.5);
        c.beta = Some(0.1);
        assert!(
            holt_winters(&y, c)
                .unwrap_err()
                .to_string()
                .contains("period")
        );
        // a deserialized result with seasonal coefficients and period 0 must not stop the
        // process in the forecasts
        let mut fit = hw(&y, 12, 0.3, Some(0.05), Some(0.4), SeasonalType::Additive);
        fit.period = 0;
        assert!(
            holt_winters_forecast(&fit, 3).is_err() && holt_winters_predict(&fit, 3, 0.95).is_err()
        );
    }

    /// R: "cannot fit models without level ('alpha' must not be 0 or FALSE)", "'alpha', 'beta'
    /// and 'gamma' must be within the unit interval", "need at least 2 periods to compute
    /// seasonal start values".
    #[test]
    fn test_holtwinters_rejects_out_of_range_parameters() {
        let ds = series_dataset();
        let y = col(&ds, "airpass");
        let base = || {
            let mut c = HoltWintersConfig::default();
            c.alpha = Some(0.3);
            c.beta = Some(0.1);
            c.gamma = Some(0.2);
            c
        };
        type Set = fn(&mut HoltWintersConfig);
        let cases: [(&str, Set); 9] = [
            ("alpha", |c: &mut HoltWintersConfig| c.alpha = Some(0.0)),
            ("alpha", |c: &mut HoltWintersConfig| c.alpha = Some(1.5)),
            ("beta", |c: &mut HoltWintersConfig| c.beta = Some(-0.1)),
            ("gamma", |c: &mut HoltWintersConfig| c.gamma = Some(2.0)),
            ("start_periods", |c: &mut HoltWintersConfig| {
                c.start_periods = 1
            }),
            ("max_iter", |c: &mut HoltWintersConfig| c.max_iter = 0), // checked always, not only when optimising
            ("tolerance", |c: &mut HoltWintersConfig| {
                c.tolerance = f64::NAN
            }),
            ("period", |c: &mut HoltWintersConfig| c.period = 1),
            ("too large", |c: &mut HoltWintersConfig| {
                c.period = usize::MAX
            }),
        ];
        for (name, set) in cases.iter() {
            let mut c = base();
            set(&mut c);
            let err = holt_winters(&y, c).unwrap_err();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. })
                    && err.to_string().contains(name),
                "{name}: {err}"
            );
        }
        let mut c = base();
        c.start_periods = usize::MAX;
        let err = holt_winters(&y, c).unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
        let fit = hw(&y, 12, 0.3, Some(0.05), Some(0.4), SeasonalType::Additive);
        for (h, level) in [(0usize, 0.95), (3, 1.0), (3, 0.0), (3, f64::NAN)] {
            assert!(holt_winters_predict(&fit, h, level).is_err(), "{h} {level}");
        }
    }

    /// co2 to 1969 (n = 132), alpha 0.5, beta 0.01, gamma 0.3, additive; plus level 0.8 to
    /// h = 24 (key co2_add_upr80_h24).
    #[test]
    fn test_validate_holtwinters_forecast_additive() {
        let (case, ds) = crate::test_support::load_case("ts_holtwinters_forecast");
        let y = col(&ds, "co2");
        let r = hw(&y, 12, 0.5, Some(0.01), Some(0.3), SeasonalType::Additive);
        check_fit(&case, "co2_add", &r, y.len());
        let p80 = holt_winters_predict(&r, 24, 0.8).unwrap();
        Check::new(&case)
            .vector("co2_add_upr80_h24", &p80.upper, Tol::stat(y.len()))
            .finish();
    }

    /// AirPassengers, alpha 0.3, beta 0.05, gamma 0.4, multiplicative: the fit and point
    /// forecasts are pinned; intervals are refused.
    #[test]
    fn test_validate_holtwinters_forecast_multiplicative() {
        let (case, ds) = crate::test_support::load_case("ts_holtwinters_forecast");
        let y = col(&ds, "airpass");
        let n = y.len();
        let r = hw(
            &y,
            12,
            0.3,
            Some(0.05),
            Some(0.4),
            SeasonalType::Multiplicative,
        );
        let res: Vec<f64> = r
            .residuals
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        // deviation TS-22: airpass_mult_upr, airpass_mult_lwr
        Check::new(&case)
            .scalar("airpass_mult_sse", r.sse, Tol::stat(n))
            .named(
                "airpass_mult_coef",
                &coef_names(&r),
                &coef_values(&r),
                Tol::coef(n),
            )
            .scalar("airpass_mult_nres", res.len() as f64, Tol::exact())
            .scalar("airpass_mult_resvar", sample_variance(&res), Tol::stat(n))
            .vector(
                "airpass_mult_fit",
                &holt_winters_forecast(&r, 12).unwrap(),
                Tol::coef(n),
            )
            .finish();
        let err = holt_winters_predict(&r, 12, 0.95).unwrap_err();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. })
                && err.to_string().contains("multiplicative"),
            "{err}"
        );
    }

    /// Holt (gamma = FALSE), alpha 0.6, beta 0.1: R starts at t = 3 with l = x[2],
    /// b = x[2] - x[1]; 142 one-step errors.
    #[test]
    fn test_validate_holtwinters_forecast_holt() {
        let (case, ds) = crate::test_support::load_case("ts_holtwinters_forecast");
        let y = col(&ds, "airpass");
        let r = hw(&y, 12, 0.6, Some(0.1), None, SeasonalType::Additive);
        check_fit(&case, "airpass_holt", &r, y.len());
        // The label of a fit without a seasonal component changes nothing.
        let m = hw(&y, 12, 0.6, Some(0.1), None, SeasonalType::Multiplicative);
        assert_eq!(m.sse.to_bits(), r.sse.to_bits());
        assert_eq!(
            holt_winters_predict(&m, 12, 0.95).unwrap().std_errors,
            holt_winters_predict(&r, 12, 0.95).unwrap().std_errors
        );
    }

    /// Additive without trend (beta = FALSE, gamma 0.2): l.start = intercept of the trend
    /// regression. SES (alpha 0.4): starts at t = 2, 143 errors.
    #[test]
    fn test_validate_holtwinters_notrend_and_ses() {
        let (case, ds) = crate::test_support::load_case("ts_holtwinters_forecast");
        let y = col(&ds, "airpass");
        check_fit(
            &case,
            "airpass_notrend",
            &hw(&y, 12, 0.3, None, Some(0.2), SeasonalType::Additive),
            y.len(),
        );
        check_fit(
            &case,
            "airpass_ses",
            &hw(&y, 12, 0.4, None, None, SeasonalType::Additive),
            y.len(),
        );
    }

    /// The filter at R's optimum (alpha, beta, gamma read from the case; they reproduce the
    /// legacy validation/expected/holtwinters_*.csv to their 15 digits): AirPassengers
    /// multiplicative, additive (gamma = 1), the legacy quarterly series (period 4, gamma = 0)
    /// and the legacy non-seasonal Holt series. The optimiser itself is not pinned (deviation
    /// TS-24).
    #[test]
    fn test_validate_holtwinters_at_r_optimum() {
        let (case, ds) = crate::test_support::load_case("ts_holtwinters_forecast");
        let par = |tag: &str, p: &str| case.named(&format!("{tag}_par"), p);
        let runs = [
            (
                "opt_mult",
                "airpass",
                12,
                SeasonalType::Multiplicative,
                true,
                true,
            ),
            ("opt_add", "airpass", 12, SeasonalType::Additive, true, true),
            (
                "opt_quarterly",
                "q",
                4,
                SeasonalType::Multiplicative,
                true,
                true,
            ),
            (
                "opt_nonseasonal",
                "ns",
                1,
                SeasonalType::Additive,
                true,
                false,
            ),
        ];
        for (tag, column, period, kind, trend, season) in runs {
            let y = col(&ds, column);
            let r = hw(
                &y,
                period,
                par(tag, "alpha"),
                trend.then(|| par(tag, "beta")),
                season.then(|| par(tag, "gamma")),
                kind,
            );
            let mut check = Check::new(&case);
            check
                .scalar(&format!("{tag}_sse"), r.sse, Tol::stat(y.len()))
                .named(
                    &format!("{tag}_coef"),
                    &coef_names(&r),
                    &coef_values(&r),
                    Tol::coef(y.len()),
                );
            if tag == "opt_mult" {
                check.vector(
                    "opt_mult_fit",
                    &holt_winters_forecast(&r, 12).unwrap(),
                    Tol::coef(y.len()),
                );
            }
            check.finish();
        }
    }

    /// X-2: an Int64 column fits as its Float64 copy; missing values, text and absent
    /// columns are refused by name (they were read as 0 or refused as non-numeric).
    #[test]
    fn test_na_holt_winters_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let run =
            |col: &str| run_holt_winters(&ds, col, 12, SeasonalType::Additive, None, None, None);
        let int = run("ai").unwrap();
        let flt = run("ai_f64").unwrap();
        assert_same(&int.fitted, &flt.fitted);
        assert_eq!(int.sse.to_bits(), flt.sse.to_bits());
        assert_null_values(&run("a_na").unwrap_err(), "a_na", 2);
        assert_null_values(&run("ai_na").unwrap_err(), "ai_na", 2);
        assert_non_numeric(&run("lab").unwrap_err(), "lab");
        assert_not_found(&run("nope").unwrap_err(), "nope");
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
    fn test_holt_winters_residual_variance_accessor() {
        // var(residuals(fit)) for every seasonal type: predict's interval scale for an
        // additive fit, and the same sample variance for a multiplicative one.
        let period = 4;
        let y: Vec<f64> = (0..32)
            .map(|i| {
                (100.0 + 2.0 * i as f64) * [1.2, 0.8, 1.0, 1.0][i % period]
                    + ((i * 7) % 11) as f64 * 0.3
            })
            .collect();
        for seasonal in [SeasonalType::Additive, SeasonalType::Multiplicative] {
            let config = HoltWintersConfig {
                alpha: Some(0.3),
                beta: Some(0.1),
                gamma: Some(0.2),
                seasonal,
                period,
                use_trend: true,
                use_seasonal: true,
                ..Default::default()
            };
            let fit = holt_winters(&y, config).unwrap();
            let e: Vec<f64> = fit
                .residuals
                .iter()
                .copied()
                .filter(|r| !r.is_nan())
                .collect();
            let m = e.iter().sum::<f64>() / e.len() as f64;
            let v = e.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (e.len() - 1) as f64;
            let got = fit.residual_variance().unwrap();
            assert!((got - v).abs() <= 1e-12 * v, "{seasonal:?}: {got} vs {v}");
            if seasonal == SeasonalType::Additive {
                assert_eq!(fit.predict(3, 0.95).unwrap().residual_variance, got);
            }
        }
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

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn holt_winters_optimiser_stops_when_the_budget_is_cancelled() {
        // Before the fix the Nelder-Mead loop (up to max_iter iterations of
        // O(n) filter passes) never looked at the budget and returned Ok.
        let period = 4;
        let pattern = [5.0, -3.0, 2.0, -4.0];
        let y: Vec<f64> = (0..48)
            .map(|i| 50.0 + 0.3 * i as f64 + pattern[i % period] + 0.5 * ((i * 7) % 11) as f64)
            .collect();
        let config = HoltWintersConfig {
            period,
            ..Default::default()
        };
        assert!(matches!(
            cancelled(|| holt_winters(&y, config.clone())),
            Err(EconError::Cancelled(_))
        ));
        // Without a budget nothing changes.
        assert!(holt_winters(&y, config).is_ok());
    }

    #[test]
    fn nelder_mead_checkpoints_each_iteration() {
        // optimize_parameters' objective never looks at the budget, so only
        // the loop's own checkpoint can stop the search. The test above uses a
        // budget cancelled before the call, which a single check before the
        // loop would also stop; this one cancels mid-search.
        let bounds = [(0.01, 0.99); 2];
        let init = [0.9, 0.1];
        let quadratic = |x: &[f64]| (x[0] - 0.3).powi(2) + (x[1] - 0.6).powi(2);

        // With a check on every iteration, at most one iteration's evaluations
        // follow the cancel (a reflection, a contraction, and a shrink of n
        // vertices).
        const CANCEL_AT: usize = 20;
        let bound = CANCEL_AT + init.len() + 2;
        let free_evals = std::cell::Cell::new(0usize);
        let free = |x: &[f64]| {
            free_evals.set(free_evals.get() + 1);
            quadratic(x)
        };
        let (x, f) = nelder_mead(&init, &bounds, free, 500, 1e-12).unwrap();
        assert!(
            f < 1e-6 && (x[0] - 0.3).abs() < 1e-3 && (x[1] - 0.6).abs() < 1e-3,
            "{x:?} {f}"
        );
        // The uncancelled search needs more evaluations than the bound allows,
        // so the bound below can only hold if the search was stopped.
        assert!(free_evals.get() > bound, "{} evaluations", free_evals.get());

        let token = crate::resource::CancelToken::new();
        let evals = std::cell::Cell::new(0usize);
        let cancelling = |x: &[f64]| {
            evals.set(evals.get() + 1);
            if evals.get() == CANCEL_AT {
                token.cancel();
            }
            quadratic(x)
        };
        let budget = crate::resource::Budget::new(None, token.clone());
        let stopped = crate::resource::with_budget(budget, || {
            nelder_mead(&init, &bounds, cancelling, 500, 1e-12)
        });
        assert!(
            matches!(stopped, Err(EconError::Cancelled(_))),
            "{stopped:?}"
        );
        assert!(evals.get() <= bound, "{} evaluations", evals.get());
    }

    #[test]
    fn holt_winters_forecast_refuses_absurd_horizon() {
        let y: Vec<f64> = (0..20)
            .map(|i| 10.0 + 0.5 * i as f64 + 0.3 * (((i * 7) % 5) as f64 - 2.0))
            .collect();
        let cfg = HoltWintersConfig {
            use_trend: true,
            use_seasonal: false,
            period: 1,
            ..Default::default()
        };
        let fit = holt_winters(&y, cfg).unwrap();
        // Before the fix: "memory allocation of 9007199254740992 bytes failed", SIGABRT.
        assert!(matches!(
            holt_winters_forecast(&fit, 1usize << 50),
            Err(EconError::ResourceLimit(_))
        ));
    }
}
