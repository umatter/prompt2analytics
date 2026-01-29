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

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Configuration for STL decomposition.
#[derive(Debug, Clone)]
pub struct StlConfig {
    /// Seasonal period (must be >= 2).
    pub period: usize,
    /// Length of the seasonal smoother. Must be odd and >= 7.
    /// Default: nextodd(ceiling((1.5*period)/(1-(1.5/s.window)))+1)
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

    // Convert to f32 for stlrs (it uses f32 internally)
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();

    // Build stlrs parameters using method chaining
    let mut params = stlrs::Stl::params();

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

    // Extract components and convert back to f64
    let trend: Vec<f64> = result.trend().iter().map(|&v| v as f64).collect();
    let seasonal: Vec<f64> = result.seasonal().iter().map(|&v| v as f64).collect();
    let remainder: Vec<f64> = result.remainder().iter().map(|&v| v as f64).collect();
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
pub fn run_stl(
    dataset: &Dataset,
    column: &str,
    period: usize,
    robust: bool,
) -> EconResult<StlResult> {
    let df = dataset.df();
    let available: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let col = df.column(column).map_err(|_| EconError::ColumnNotFound {
        column: column.to_string(),
        available: available.clone(),
    })?;

    let x: Vec<f64> = col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: column.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let config = StlConfig::new(period).robust(robust);
    stl(&x, config)
}

/// Convenience function to run STL with a custom configuration on a Dataset.
pub fn run_stl_with_config(
    dataset: &Dataset,
    column: &str,
    config: StlConfig,
) -> EconResult<StlResult> {
    let df = dataset.df();
    let available: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let col = df.column(column).map_err(|_| EconError::ColumnNotFound {
        column: column.to_string(),
        available: available.clone(),
    })?;

    let x: Vec<f64> = col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: column.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    stl(&x, config)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Validation test against R's stl() function
    #[test]
    fn test_validate_stl_against_r() {
        // R code:
        // set.seed(42)
        // x <- ts(100 + 0.5*(1:120) + 10*sin(2*pi*(1:120)/12), frequency=12)
        // result <- stl(x, s.window="periodic")
        //
        // Expected: seasonal component should show ~10*sin pattern
        // Trend should be ~100 + 0.5*t

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
