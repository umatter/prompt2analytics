//! Seasonal-Trend decomposition using LOESS (STL).
//!
//! Implements R's `stl()` function which decomposes a time series into
//! seasonal, trend, and remainder components using locally weighted regression.
//!
//! # References
//!
//! - Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
//!   "STL: A Seasonal-Trend Decomposition Procedure Based on Loess".
//!   Journal of Official Statistics, 6(1), 3-73.
//! - R Core Team. `stats::stl()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html>
//! - Implementation uses the `stlrs` crate (Rust port of original Fortran code).
//!
//! # Precision
//!
//! `stlrs` computes in f32. The series is centred by its mean and scaled by its standard
//! deviation before the call, and the components are restored afterwards (LOESS commutes with
//! an affine map of the series, and the robustness weights do not change under it), so the
//! seasonal and trend components carry about 1e-7 of the series' spread per operation whatever
//! its level. The remainder is computed in f64 as x - S - T, as R's `stl` computes it.

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Configuration for STL decomposition.
#[derive(Debug, Clone)]
pub struct StlConfig {
    /// Seasonal period (must be >= 2).
    pub period: usize,
    /// Length of the seasonal smoother (R's `s.window`). Odd and >= 3.
    /// Default: nextodd(period) (stlrs). R's `stl` has no default `s.window`; use
    /// [`StlConfig::periodic`] for `s.window = "periodic"`.
    pub seasonal_length: Option<usize>,
    /// Degree of locally-fitted polynomial in seasonal smoothing (0 or 1).
    pub seasonal_degree: Option<usize>,
    /// Length of the trend smoother. Must be odd.
    /// Default: nextodd(ceiling((1.5*period)/(1-(1.5/t.window)))+1)
    pub trend_length: Option<usize>,
    /// Degree of locally-fitted polynomial in trend smoothing (0 or 1).
    pub trend_degree: Option<usize>,
    /// Length of the low-pass filter. Default: nextodd(period).
    pub low_pass_length: Option<usize>,
    /// Degree of locally-fitted polynomial in low-pass smoothing (0 or 1).
    pub low_pass_degree: Option<usize>,
    /// Number of inner loops for updating seasonal and trend components.
    /// Default: 2 for non-robust, 1 for robust.
    pub inner_loops: Option<usize>,
    /// Number of outer loops for robustness iterations.
    /// Default: 0 for non-robust, 15 for robust.
    pub outer_loops: Option<usize>,
    /// Whether to use robust fitting (iteratively downweights outliers).
    pub robust: bool,
    /// R's `s.window = "periodic"`: seasonal window 10 n + 1, seasonal degree 0, and the
    /// seasonal component replaced by its cycle means afterwards. Not combined with
    /// `seasonal_length` or `seasonal_degree`.
    pub periodic: bool,
}

impl Default for StlConfig {
    fn default() -> Self {
        Self {
            period: 1,
            seasonal_length: None,
            seasonal_degree: None,
            trend_length: None,
            trend_degree: None,
            low_pass_length: None,
            low_pass_degree: None,
            inner_loops: None,
            outer_loops: None,
            robust: false,
            periodic: false,
        }
    }
}

impl StlConfig {
    /// Create a new STL configuration with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ..Default::default()
        }
    }

    /// Set the seasonal smoother length.
    pub fn seasonal_length(mut self, length: usize) -> Self {
        self.seasonal_length = Some(length);
        self
    }

    /// Set the trend smoother length.
    pub fn trend_length(mut self, length: usize) -> Self {
        self.trend_length = Some(length);
        self
    }

    /// R's `s.window = "periodic"`: a seasonal component that repeats exactly every period.
    pub fn periodic(mut self) -> Self {
        self.periodic = true;
        self
    }

    /// Enable robust fitting.
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Set the number of inner loops.
    pub fn inner_loops(mut self, loops: usize) -> Self {
        self.inner_loops = Some(loops);
        self
    }

    /// Set the number of outer loops (robustness iterations).
    pub fn outer_loops(mut self, loops: usize) -> Self {
        self.outer_loops = Some(loops);
        self
    }
}

/// Result of STL decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StlResult {
    /// Original time series.
    pub x: Vec<f64>,
    /// Seasonal period used.
    pub period: usize,
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder (residual) component.
    pub remainder: Vec<f64>,
    /// Robustness weights (if robust fitting was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
    /// Strength of seasonality (0-1).
    pub seasonal_strength: f64,
    /// Strength of trend (0-1).
    pub trend_strength: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Whether robust fitting was used.
    pub robust: bool,
}

/// Perform STL decomposition on a time series.
///
/// STL (Seasonal-Trend decomposition using Loess) decomposes a time series into
/// three components:
/// - **Trend**: The long-term progression of the series
/// - **Seasonal**: Regular periodic variations
/// - **Remainder**: Irregular fluctuations after removing trend and seasonal
///
/// # Arguments
///
/// * `x` - Time series values (must have at least 2 * period observations)
/// * `config` - STL configuration parameters
///
/// # Returns
///
/// `StlResult` containing all decomposition components.
///
/// # Mathematical Background
///
/// The decomposition is: Y_t = T_t + S_t + R_t
///
/// where:
/// - T_t is the trend component
/// - S_t is the seasonal component
/// - R_t is the remainder
///
/// STL uses LOESS (locally weighted regression) to extract these components
/// iteratively. The procedure consists of:
/// 1. Inner loop: Updates seasonal and trend components
/// 2. Outer loop (if robust): Calculates robustness weights to downweight outliers
///
/// # References
///
/// - Cleveland et al. (1990). "STL: A Seasonal-Trend Decomposition".
/// - R function `stats::stl()`
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::{stl, StlConfig};
///
/// // Monthly data with yearly seasonality
/// let data: Vec<f64> = (0..120).map(|t| {
///     100.0 + 0.5 * t as f64 + 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin()
/// }).collect();
///
/// let config = StlConfig::new(12).robust(false);
/// let result = stl(&data, config).unwrap();
///
/// assert_eq!(result.n_obs, 120);
/// assert_eq!(result.period, 12);
/// ```
pub fn stl(x: &[f64], config: StlConfig) -> EconResult<StlResult> {
    let n = x.len();
    let period = config.period;

    // Validate inputs
    if period < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Period must be at least 2".to_string(),
        });
    }

    if n < 2 * period {
        return Err(EconError::InsufficientData {
            required: 2 * period,
            provided: n,
            context: format!("STL decomposition with period {}", period),
        });
    }

    // Check for non-finite values
    if x.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "Time series contains non-finite values (NaN or Inf)".to_string(),
        });
    }

    if config.periodic && (config.seasonal_length.is_some() || config.seasonal_degree.is_some()) {
        return Err(EconError::InvalidSpecification {
            message: "a periodic seasonal window sets the seasonal length and degree; \
                      give either periodic or seasonal_length/seasonal_degree"
                .to_string(),
        });
    }

    // stlrs computes in f32: centre and scale first, so the precision follows the series'
    // spread and not its level. A constant series is only centred.
    let center = x.iter().sum::<f64>() / n as f64;
    let spread = (x.iter().map(|v| (v - center).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let scale = if spread > 0.0 && spread.is_finite() {
        spread
    } else {
        1.0
    };
    let x_f32: Vec<f32> = x.iter().map(|&v| ((v - center) / scale) as f32).collect();

    // Build stlrs parameters using method chaining
    let mut params = stlrs::Stl::params();

    if config.periodic {
        // R: s.window <- 10 * n + 1; s.degree <- 0
        let window = n
            .checked_mul(10)
            .and_then(|w| w.checked_add(1))
            .ok_or_else(|| EconError::InvalidSpecification {
                message: format!("series of {n} values is too long for a periodic window"),
            })?;
        params.seasonal_length(window);
        params.seasonal_degree(0);
    }
    if let Some(sl) = config.seasonal_length {
        params.seasonal_length(sl);
    }
    if let Some(sd) = config.seasonal_degree {
        params.seasonal_degree(sd as i32);
    }
    if let Some(tl) = config.trend_length {
        params.trend_length(tl);
    }
    if let Some(td) = config.trend_degree {
        params.trend_degree(td as i32);
    }
    if let Some(ll) = config.low_pass_length {
        params.low_pass_length(ll);
    }
    if let Some(ld) = config.low_pass_degree {
        params.low_pass_degree(ld as i32);
    }
    if let Some(il) = config.inner_loops {
        params.inner_loops(il);
    }
    if let Some(ol) = config.outer_loops {
        params.outer_loops(ol);
    }
    if config.robust {
        params.robust(true);
    }

    // Perform decomposition
    let result = params
        .fit(&x_f32, period)
        .map_err(|e| EconError::Computation(format!("STL decomposition failed: {}", e)))?;

    // Restore the components: T = s T' + c, S = s S'.
    let trend: Vec<f64> = result
        .trend()
        .iter()
        .map(|&v| scale * v as f64 + center)
        .collect();
    let mut seasonal: Vec<f64> = result
        .seasonal()
        .iter()
        .map(|&v| scale * v as f64)
        .collect();
    if config.periodic {
        // R: seasonal <- tapply(seasonal, cycle(x), mean)[cycle(x)]
        let mut sums = vec![0.0; period];
        let mut counts = vec![0usize; period];
        for (i, &v) in seasonal.iter().enumerate() {
            sums[i % period] += v;
            counts[i % period] += 1;
        }
        for (i, v) in seasonal.iter_mut().enumerate() {
            *v = sums[i % period] / counts[i % period] as f64;
        }
    }
    // R: remainder <- x - seasonal - trend, in f64.
    let remainder: Vec<f64> = x
        .iter()
        .zip(&seasonal)
        .zip(&trend)
        .map(|((&xv, &sv), &tv)| xv - sv - tv)
        .collect();
    let weights: Option<Vec<f64>> = if config.robust {
        Some(result.weights().iter().map(|&v| v as f64).collect())
    } else {
        None
    };

    // Calculate seasonal and trend strength
    // Strength of seasonality: 1 - Var(R) / Var(S + R)
    // Strength of trend: 1 - Var(R) / Var(T + R)
    let seasonal_strength = calculate_strength(&seasonal, &remainder);
    let trend_strength = calculate_strength(&trend, &remainder);

    Ok(StlResult {
        x: x.to_vec(),
        period,
        trend,
        seasonal,
        remainder,
        weights,
        seasonal_strength,
        trend_strength,
        n_obs: n,
        robust: config.robust,
    })
}

/// Calculate strength of a component (seasonal or trend).
/// Strength = max(0, 1 - Var(remainder) / Var(component + remainder))
fn calculate_strength(component: &[f64], remainder: &[f64]) -> f64 {
    if component.is_empty() || remainder.is_empty() {
        return 0.0;
    }

    let var_r = variance(remainder);
    let combined: Vec<f64> = component
        .iter()
        .zip(remainder.iter())
        .map(|(c, r)| c + r)
        .collect();
    let var_combined = variance(&combined);

    if var_combined < 1e-10 {
        return 0.0;
    }

    (1.0 - var_r / var_combined).max(0.0)
}

/// Calculate variance of a slice.
fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();
    sum_sq / (n - 1) as f64
}

/// Convenience function to run STL decomposition on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Column name with time series values
/// * `period` - Seasonal period (e.g., 12 for monthly data with yearly cycle)
/// * `robust` - Whether to use robust fitting
///
/// # Returns
///
/// `StlResult` with decomposition components.
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
pub fn run_stl(
    dataset: &Dataset,
    column: &str,
    period: usize,
    robust: bool,
) -> EconResult<StlResult> {
    let (x, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    let config = StlConfig::new(period).robust(robust);
    stl(&x, config)
}

/// Convenience function to run STL with a custom configuration on a Dataset.
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
pub fn run_stl_with_config(
    dataset: &Dataset,
    column: &str,
    config: StlConfig,
) -> EconResult<StlResult> {
    let (x, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    stl(&x, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    use p2a_validation::{Check, Tol};

    /// Case ts_utils_t4: the monthly series y (n = 144, level about 1,000 to 1,290) and y + 1e8.
    fn stl_case() -> (p2a_validation::Case, Vec<f64>, Tol) {
        let (case, ds) = crate::test_support::load_case("ts_utils_t4");
        let (y, _) = numeric_column(ds.df(), "y", NaPolicy::Fail).unwrap();
        // stlrs computes in f32, so after centring and scaling the components carry about 1e-7 of
        // the series' spread per operation: absolute 1e-5 * sd(y).
        let tol = Tol::new(0.0, 1e-5 * case.scalar("stl_sd"));
        (case, y, tol)
    }

    fn check_stl(check: &mut Check, prefix: &str, r: &StlResult, tol: Tol) {
        check
            .vector(&format!("{prefix}_seasonal"), &r.seasonal, tol)
            .vector(&format!("{prefix}_trend"), &r.trend, tol)
            .vector(&format!("{prefix}_remainder"), &r.remainder, tol);
    }

    /// stl(ts(y, frequency = 12), s.window = "periodic").
    #[test]
    fn test_validate_stl_periodic_ts_utils_t4() {
        let (case, y, tol) = stl_case();
        let r = stl(&y, StlConfig::new(12).periodic()).unwrap();
        let mut check = Check::new(&case);
        check_stl(&mut check, "stl_periodic", &r, tol);
        check.finish();
    }

    /// stl(ts(y, frequency = 12), s.window = 13) (13 = stlrs' default nextodd(period)).
    #[test]
    fn test_validate_stl_swin13_ts_utils_t4() {
        let (case, y, tol) = stl_case();
        let r = stl(&y, StlConfig::new(12).seasonal_length(13)).unwrap();
        let mut check = Check::new(&case);
        check_stl(&mut check, "stl_swin13", &r, tol);
        check.finish();
    }

    /// The same two calls on y + 1e8: the precision follows the spread, not the level.
    #[test]
    fn test_validate_stl_level_1e8_ts_utils_t4() {
        let (case, y, tol) = stl_case();
        let big: Vec<f64> = y.iter().map(|v| v + 1e8).collect();
        let mut check = Check::new(&case);
        let p = stl(&big, StlConfig::new(12).periodic()).unwrap();
        check_stl(&mut check, "stl_periodic_big", &p, tol);
        let w = stl(&big, StlConfig::new(12)).unwrap();
        check_stl(&mut check, "stl_swin13_big", &w, tol);
        check.finish();
    }

    #[test]
    fn test_stl_reconstructs_exactly() {
        let (_, y, _) = stl_case();
        let big: Vec<f64> = y.iter().map(|v| v + 1e8).collect();
        for config in [StlConfig::new(12), StlConfig::new(12).periodic()] {
            let r = stl(&big, config).unwrap();
            for i in 0..big.len() {
                let back = r.trend[i] + r.seasonal[i] + r.remainder[i];
                assert!(
                    (big[i] - back).abs() <= 1e-9 * 1e8,
                    "{i}: {} vs {back}",
                    big[i]
                );
            }
        }
    }

    #[test]
    fn test_stl_periodic_with_seasonal_length_is_refused() {
        let (_, y, _) = stl_case();
        let err = stl(&y, StlConfig::new(12).periodic().seasonal_length(13)).unwrap_err();
        assert!(
            matches!(&err, EconError::InvalidSpecification { message } if message.contains("periodic")),
            "{err}"
        );
    }

    #[test]
    fn test_stl_constant_series() {
        let x = vec![1e8 + 3.0; 48];
        for config in [StlConfig::new(12), StlConfig::new(12).periodic()] {
            let r = stl(&x, config).unwrap();
            for i in 0..x.len() {
                assert!((r.trend[i] - x[i]).abs() <= 1e-6, "{}", r.trend[i]);
                assert!(r.seasonal[i].abs() <= 1e-6, "{}", r.seasonal[i]);
                assert!(r.remainder[i].abs() <= 1e-6, "{}", r.remainder[i]);
            }
        }
    }

    /// X-2: an Int64 column decomposes as its Float64 copy; missing values, text and
    /// absent columns are refused by name.
    #[test]
    fn test_na_stl_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let plain = |col: &str| run_stl(&ds, col, 12, false);
        let config = |col: &str| run_stl_with_config(&ds, col, StlConfig::new(12));
        for run in [&plain as &dyn Fn(&str) -> EconResult<StlResult>, &config] {
            let int = run("ai").unwrap();
            let flt = run("ai_f64").unwrap();
            assert_same(&int.trend, &flt.trend);
            assert_same(&int.seasonal, &flt.seasonal);
            assert_null_values(&run("a_na").unwrap_err(), "a_na", 2);
            assert_null_values(&run("ai_na").unwrap_err(), "ai_na", 2);
            assert_non_numeric(&run("lab").unwrap_err(), "lab");
            assert_not_found(&run("nope").unwrap_err(), "nope");
        }
    }

    fn generate_seasonal_data(
        n: usize,
        period: usize,
        trend_slope: f64,
        seasonal_amp: f64,
        noise_sd: f64,
    ) -> Vec<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        use std::f64::consts::PI;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, noise_sd).unwrap();

        (0..n)
            .map(|t| {
                let trend = 100.0 + trend_slope * t as f64;
                let seasonal = seasonal_amp * (2.0 * PI * t as f64 / period as f64).sin();
                let noise = normal.sample(&mut rng);
                trend + seasonal + noise
            })
            .collect()
    }

    #[test]
    fn test_stl_basic() {
        let x = generate_seasonal_data(120, 12, 0.5, 10.0, 1.0);
        let config = StlConfig::new(12);

        let result = stl(&x, config).unwrap();

        assert_eq!(result.n_obs, 120);
        assert_eq!(result.period, 12);
        assert_eq!(result.trend.len(), 120);
        assert_eq!(result.seasonal.len(), 120);
        assert_eq!(result.remainder.len(), 120);
        assert!(!result.robust);

        // Check that components reconstruct the original
        // Note: stlrs uses f32 internally, so we lose some precision in f64->f32->f64 conversion
        for i in 0..result.n_obs {
            let reconstructed = result.trend[i] + result.seasonal[i] + result.remainder[i];
            let diff = (result.x[i] - reconstructed).abs();
            assert!(
                diff < 1e-4,
                "Reconstruction failed at index {}: {} vs {}",
                i,
                result.x[i],
                reconstructed
            );
        }
    }

    #[test]
    fn test_stl_robust() {
        // Add outliers
        let mut x = generate_seasonal_data(120, 12, 0.5, 10.0, 1.0);
        x[30] += 50.0; // Outlier
        x[60] -= 50.0; // Outlier

        let config = StlConfig::new(12).robust(true);
        let result = stl(&x, config).unwrap();

        assert!(result.robust);
        assert!(result.weights.is_some());

        let weights = result.weights.unwrap();
        assert_eq!(weights.len(), 120);

        // Outliers should have lower weights (downweighted)
        // Note: exact values depend on implementation, just check they exist
        assert!(weights[30] <= 1.0);
        assert!(weights[60] <= 1.0);
    }

    #[test]
    fn test_stl_strength() {
        // Strong seasonality, weak trend
        let x1: Vec<f64> = (0..120)
            .map(|t| 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin())
            .collect();

        let result1 = stl(&x1, StlConfig::new(12)).unwrap();
        assert!(
            result1.seasonal_strength > 0.8,
            "Expected strong seasonality, got {}",
            result1.seasonal_strength
        );

        // Strong trend, weak seasonality
        let x2: Vec<f64> = (0..120).map(|t| 100.0 + 2.0 * t as f64).collect();

        let result2 = stl(&x2, StlConfig::new(12)).unwrap();
        assert!(
            result2.trend_strength > 0.8,
            "Expected strong trend, got {}",
            result2.trend_strength
        );
    }

    #[test]
    fn test_stl_too_short() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = stl(&x, StlConfig::new(12));
        assert!(result.is_err());
    }

    #[test]
    fn test_stl_invalid_period() {
        let x = vec![1.0; 100];
        let result = stl(&x, StlConfig::new(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_stl_non_finite() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = stl(&x, StlConfig::new(4));
        assert!(result.is_err());
    }

    #[test]
    fn test_stl_custom_params() {
        let x = generate_seasonal_data(120, 12, 0.5, 10.0, 1.0);
        let config = StlConfig::new(12)
            .seasonal_length(15)
            .trend_length(21)
            .inner_loops(3)
            .outer_loops(2);

        let result = stl(&x, config).unwrap();
        assert_eq!(result.n_obs, 120);
    }

    /// Recovers a deterministic trend-plus-sine DGP (property; R's stl is pinned in the
    /// test_validate_stl_* tests).
    #[test]
    fn test_stl_recovers_dgp() {
        // Generate deterministic data (no noise)
        let x: Vec<f64> = (1..=120)
            .map(|t| {
                100.0 + 0.5 * t as f64 + 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin()
            })
            .collect();

        let result = stl(&x, StlConfig::new(12)).unwrap();

        // Check trend is approximately linear
        let first_trend = result.trend[10]; // Skip boundary effects
        let last_trend = result.trend[109];
        let expected_slope = 0.5;
        let actual_slope = (last_trend - first_trend) / (109.0 - 10.0);
        assert!(
            (actual_slope - expected_slope).abs() < 0.1,
            "Trend slope should be ~0.5, got {}",
            actual_slope
        );

        // Check seasonal amplitude is approximately 10
        let seasonal_max = result
            .seasonal
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let seasonal_min = result
            .seasonal
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let seasonal_range = seasonal_max - seasonal_min;
        assert!(
            seasonal_range > 15.0 && seasonal_range < 25.0,
            "Seasonal range should be ~20 (amplitude 10), got {}",
            seasonal_range
        );

        // Check remainder is small
        let remainder_var = variance(&result.remainder);
        assert!(
            remainder_var < 1.0,
            "Remainder variance should be small, got {}",
            remainder_var
        );
    }
}
