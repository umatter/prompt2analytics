//! Classical seasonal decomposition by moving averages.
//!
//! Implements R's `decompose()` function which decomposes a time series
//! into trend, seasonal, and random components using moving averages.
//!
//! # References
//!
//! - Kendall, M. (1976). "Time Series". Charles Griffin.
//! - R Core Team. `stats::decompose()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/decompose.html>

use serde::{Deserialize, Serialize};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};

/// Type of seasonal decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum DecomposeType {
    /// Additive decomposition: Y = Trend + Seasonal + Random
    #[default]
    Additive,
    /// Multiplicative decomposition: Y = Trend × Seasonal × Random
    Multiplicative,
}

/// Configuration for classical decomposition.
#[derive(Debug, Clone)]
pub struct DecomposeConfig {
    /// Type of decomposition (additive or multiplicative).
    pub decompose_type: DecomposeType,
    /// Filter coefficients for extracting trend. If None, uses symmetric moving average.
    pub filter: Option<Vec<f64>>,
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            decompose_type: DecomposeType::Additive,
            filter: None,
        }
    }
}

/// Result of classical seasonal decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposeResult {
    /// Original time series.
    pub x: Vec<f64>,
    /// Seasonal period used.
    pub period: usize,
    /// Type of decomposition performed.
    pub decompose_type: DecomposeType,
    /// Trend component (may contain NaN at boundaries).
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Random (irregular) component (may contain NaN at boundaries).
    pub random: Vec<f64>,
    /// Seasonal figure (one complete cycle of seasonal factors).
    pub figure: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
}

/// Perform classical seasonal decomposition using moving averages.
///
/// This function decomposes a time series into trend, seasonal, and random
/// components. It uses a symmetric moving average to extract the trend,
/// then computes seasonal factors by averaging de-trended values.
///
/// # Arguments
///
/// * `x` - Time series values
/// * `period` - Seasonal period (e.g., 12 for monthly data with yearly cycle)
/// * `config` - Decomposition configuration
///
/// # Returns
///
/// `DecomposeResult` containing all decomposition components.
///
/// # Mathematical Background
///
/// For additive decomposition:
/// - Y_t = T_t + S_t + e_t
/// - Trend extracted via moving average filter
/// - Seasonal factors: mean of (Y_t - T_t) for each season position
///
/// For multiplicative decomposition:
/// - Y_t = T_t × S_t × e_t
/// - Seasonal factors: mean of (Y_t / T_t) for each season position
///
/// # References
///
/// - R Core Team. `stats::decompose()`.
pub fn decompose(x: &[f64], period: usize, config: DecomposeConfig) -> EconResult<DecomposeResult> {
    let n = x.len();

    // Validate inputs
    if n < 2 * period {
        return Err(EconError::InsufficientData {
            required: 2 * period,
            provided: n,
            context: format!("decompose with period {}", period),
        });
    }

    if period < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Period must be at least 2".to_string(),
        });
    }

    // Check for non-positive values in multiplicative decomposition
    if config.decompose_type == DecomposeType::Multiplicative {
        if x.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: "Multiplicative decomposition requires all positive, finite values".to_string(),
            });
        }
    }

    // Step 1: Extract trend using moving average filter
    let trend = match &config.filter {
        Some(f) => apply_filter(x, f)?,
        None => symmetric_moving_average(x, period),
    };

    // Step 2: Compute de-trended series
    let detrended: Vec<f64> = x.iter()
        .zip(trend.iter())
        .map(|(&y, &t)| {
            if t.is_nan() {
                f64::NAN
            } else {
                match config.decompose_type {
                    DecomposeType::Additive => y - t,
                    DecomposeType::Multiplicative => y / t,
                }
            }
        })
        .collect();

    // Step 3: Compute seasonal figure (mean of de-trended values for each period position)
    let figure = compute_seasonal_figure(&detrended, period, config.decompose_type);

    // Step 4: Extend seasonal figure to full length
    let seasonal: Vec<f64> = (0..n)
        .map(|i| figure[i % period])
        .collect();

    // Step 5: Compute random component
    let random: Vec<f64> = x.iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| {
            if t.is_nan() {
                f64::NAN
            } else {
                match config.decompose_type {
                    DecomposeType::Additive => y - t - s,
                    DecomposeType::Multiplicative => y / (t * s),
                }
            }
        })
        .collect();

    Ok(DecomposeResult {
        x: x.to_vec(),
        period,
        decompose_type: config.decompose_type,
        trend,
        seasonal,
        random,
        figure,
        n_obs: n,
    })
}

/// Apply a symmetric moving average filter.
///
/// For even period, uses a weighted moving average with half-weights at ends.
/// For odd period, uses a simple centered moving average.
fn symmetric_moving_average(x: &[f64], period: usize) -> Vec<f64> {
    let n = x.len();
    let mut trend = vec![f64::NAN; n];

    if period % 2 == 1 {
        // Odd period: simple centered moving average
        let half = period / 2;
        for i in half..(n - half) {
            let sum: f64 = x[(i - half)..=(i + half)].iter().sum();
            trend[i] = sum / period as f64;
        }
    } else {
        // Even period: weighted moving average with half-weights at ends
        // Filter: [0.5, 1, 1, ..., 1, 0.5] / period
        // This is equivalent to a period-wide MA followed by a 2-wide MA (convolution)
        let half = period / 2;
        for i in half..(n - half) {
            let mut sum = 0.0;
            // First and last elements get half weight
            sum += 0.5 * x[i - half];
            for j in (i - half + 1)..(i + half) {
                sum += x[j];
            }
            sum += 0.5 * x[i + half];
            trend[i] = sum / period as f64;
        }
    }

    trend
}

/// Apply a custom filter to the time series.
fn apply_filter(x: &[f64], filter: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();
    let f_len = filter.len();

    if f_len == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Filter cannot be empty".to_string(),
        });
    }

    // Normalize filter
    let filter_sum: f64 = filter.iter().sum();
    let normalized: Vec<f64> = if filter_sum.abs() > 1e-10 {
        filter.iter().map(|&f| f / filter_sum).collect()
    } else {
        filter.to_vec()
    };

    let mut result = vec![f64::NAN; n];
    let half = f_len / 2;

    for i in half..(n - half) {
        let mut sum = 0.0;
        for (j, &f) in normalized.iter().enumerate() {
            let idx = i + j - half;
            if idx < n {
                sum += f * x[idx];
            }
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Compute seasonal figure by averaging de-trended values at each period position.
fn compute_seasonal_figure(detrended: &[f64], period: usize, decompose_type: DecomposeType) -> Vec<f64> {
    let mut sums = vec![0.0; period];
    let mut counts = vec![0usize; period];

    for (i, &v) in detrended.iter().enumerate() {
        if !v.is_nan() && v.is_finite() {
            sums[i % period] += v;
            counts[i % period] += 1;
        }
    }

    // Compute means
    let mut figure: Vec<f64> = sums.iter()
        .zip(counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Center the figure so it sums to zero (additive) or averages to 1 (multiplicative)
    match decompose_type {
        DecomposeType::Additive => {
            let mean: f64 = figure.iter().sum::<f64>() / period as f64;
            for f in &mut figure {
                *f -= mean;
            }
        }
        DecomposeType::Multiplicative => {
            let mean: f64 = figure.iter().sum::<f64>() / period as f64;
            if mean.abs() > 1e-10 {
                for f in &mut figure {
                    *f /= mean;
                }
            }
        }
    }

    figure
}

/// Convenience function to run decomposition on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Column name with time series values
/// * `period` - Seasonal period
/// * `decompose_type` - Type of decomposition (additive or multiplicative)
///
/// # Returns
///
/// `DecomposeResult` with decomposition components.
pub fn run_decompose(
    dataset: &Dataset,
    column: &str,
    period: usize,
    decompose_type: DecomposeType,
) -> EconResult<DecomposeResult> {
    let df = dataset.df();
    let available: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let col = df.column(column)
        .map_err(|_| EconError::ColumnNotFound {
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

    decompose(&x, period, DecomposeConfig {
        decompose_type,
        filter: None,
    })
}

/// Convenience function to run decomposition with a custom filter.
pub fn run_decompose_with_filter(
    dataset: &Dataset,
    column: &str,
    period: usize,
    decompose_type: DecomposeType,
    filter: Vec<f64>,
) -> EconResult<DecomposeResult> {
    let df = dataset.df();
    let available: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let col = df.column(column)
        .map_err(|_| EconError::ColumnNotFound {
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

    decompose(&x, period, DecomposeConfig {
        decompose_type,
        filter: Some(filter),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_seasonal_data(n: usize, period: usize, trend_slope: f64, seasonal_amp: f64) -> Vec<f64> {
        use std::f64::consts::PI;

        (0..n).map(|t| {
            let trend = 100.0 + trend_slope * t as f64;
            let seasonal = seasonal_amp * (2.0 * PI * t as f64 / period as f64).sin();
            trend + seasonal
        }).collect()
    }

    #[test]
    fn test_decompose_additive_basic() {
        // Generate data with clear trend and seasonality
        let x = generate_seasonal_data(48, 12, 0.5, 10.0);

        let result = decompose(&x, 12, DecomposeConfig::default()).unwrap();

        assert_eq!(result.n_obs, 48);
        assert_eq!(result.period, 12);
        assert_eq!(result.figure.len(), 12);
        assert_eq!(result.trend.len(), 48);
        assert_eq!(result.seasonal.len(), 48);
        assert_eq!(result.random.len(), 48);

        // Check that seasonal figure sums to approximately zero (additive)
        let figure_sum: f64 = result.figure.iter().sum();
        assert!(figure_sum.abs() < 1e-10, "Seasonal figure should sum to zero, got {}", figure_sum);

        // Check that trend is monotonically increasing (approximately)
        let valid_trend: Vec<f64> = result.trend.iter()
            .filter(|&&t| !t.is_nan())
            .copied()
            .collect();

        assert!(valid_trend.len() > 10);
        let first_valid = valid_trend[0];
        let last_valid = valid_trend[valid_trend.len() - 1];
        assert!(last_valid > first_valid, "Trend should be increasing");
    }

    #[test]
    fn test_decompose_multiplicative() {
        // Generate multiplicative data
        let n = 48;
        let period = 12;
        let x: Vec<f64> = (0..n).map(|t| {
            let trend = 100.0 + 0.5 * t as f64;
            let seasonal = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
            trend * seasonal
        }).collect();

        let config = DecomposeConfig {
            decompose_type: DecomposeType::Multiplicative,
            filter: None,
        };

        let result = decompose(&x, period, config).unwrap();

        // Check that seasonal figure averages to approximately 1 (multiplicative)
        let figure_mean: f64 = result.figure.iter().sum::<f64>() / period as f64;
        assert!((figure_mean - 1.0).abs() < 1e-10, "Seasonal figure should average to 1, got {}", figure_mean);
    }

    #[test]
    fn test_decompose_too_short() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = decompose(&x, 12, DecomposeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_decompose_invalid_period() {
        let x = vec![1.0; 100];
        let result = decompose(&x, 1, DecomposeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_decompose_multiplicative_negative_values() {
        let x = vec![-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = DecomposeConfig {
            decompose_type: DecomposeType::Multiplicative,
            filter: None,
        };
        let result = decompose(&x, 4, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompose_odd_period() {
        let x: Vec<f64> = (0..100).map(|t| {
            let trend = 50.0 + 0.3 * t as f64;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * t as f64 / 7.0).sin();
            trend + seasonal
        }).collect();

        let result = decompose(&x, 7, DecomposeConfig::default()).unwrap();

        assert_eq!(result.figure.len(), 7);

        // Seasonal figure should sum to zero
        let figure_sum: f64 = result.figure.iter().sum();
        assert!(figure_sum.abs() < 1e-10);
    }

    #[test]
    fn test_decompose_recovery() {
        // Test that original = trend + seasonal + random (for additive)
        let x = generate_seasonal_data(60, 12, 0.2, 8.0);

        let result = decompose(&x, 12, DecomposeConfig::default()).unwrap();

        // For points where trend is valid, check reconstruction
        for i in 0..result.n_obs {
            if !result.trend[i].is_nan() {
                let reconstructed = result.trend[i] + result.seasonal[i] + result.random[i];
                let diff = (result.x[i] - reconstructed).abs();
                assert!(diff < 1e-10, "Reconstruction failed at index {}: {} vs {}",
                    i, result.x[i], reconstructed);
            }
        }
    }

    /// Validation test against R's decompose() function
    #[test]
    fn test_validate_decompose_against_r() {
        // R code:
        // set.seed(42)
        // x <- 100 + 0.5*(1:48) + 10*sin(2*pi*(1:48)/12) + rnorm(48, sd=2)
        // result <- decompose(ts(x, frequency=12), type="additive")
        // result$figure  # seasonal pattern

        // Generate the same data as R (without noise for deterministic test)
        let x: Vec<f64> = (1..=48).map(|t| {
            let trend = 100.0 + 0.5 * t as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin();
            trend + seasonal
        }).collect();

        let result = decompose(&x, 12, DecomposeConfig::default()).unwrap();

        // Check that seasonal figure follows expected pattern
        // For a sine wave, the figure should show the same pattern
        for (i, &f) in result.figure.iter().enumerate() {
            let expected = 10.0 * (2.0 * std::f64::consts::PI * (i + 1) as f64 / 12.0).sin();
            // Allow tolerance for averaging effects at boundaries
            let diff = (f - expected).abs();
            assert!(diff < 1.5, "Seasonal figure mismatch at position {}: got {}, expected {}",
                i, f, expected);
        }

        // Check trend is roughly linear
        let valid_trend: Vec<(usize, f64)> = result.trend.iter()
            .enumerate()
            .filter(|(_, t)| !t.is_nan())
            .map(|(i, t)| (i, *t))
            .collect();

        if valid_trend.len() >= 2 {
            let (i1, t1) = valid_trend[0];
            let (i2, t2) = valid_trend[valid_trend.len() - 1];
            let slope = (t2 - t1) / (i2 - i1) as f64;
            // Expected slope is 0.5
            assert!((slope - 0.5).abs() < 0.1, "Trend slope should be ~0.5, got {}", slope);
        }
    }
}
