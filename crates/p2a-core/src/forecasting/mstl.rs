//! MSTL (Multiple Seasonal-Trend decomposition using LOESS).

use anyhow::{Result, anyhow};
use augurs_core::Fit;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;

/// Result from MSTL decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstlResult {
    /// Column that was decomposed
    pub column: String,
    /// Seasonal periods used
    pub periods: Vec<usize>,
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal components (one per period)
    pub seasonal: Vec<Vec<f64>>,
    /// Residual component
    pub residuals: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
}

/// Perform MSTL decomposition on a time series.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series
/// * `column` - The column name with time series values
/// * `periods` - Seasonal periods to extract (e.g., [7, 365] for daily data with weekly and yearly seasonality)
///
/// # Returns
/// `MstlResult` containing the decomposition components.
pub fn run_mstl(dataset: &Dataset, column: &str, periods: &[usize]) -> Result<MstlResult> {
    use augurs_mstl::{MSTLModel, NaiveTrend};

    // Extract time series data
    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

    let values: Vec<f64> = col
        .f64()
        .map_err(|e| anyhow!("Column must be numeric: {}", e))?
        .into_no_null_iter()
        .collect();

    if values.is_empty() {
        return Err(anyhow!("Empty time series"));
    }

    // Check that we have enough observations for the longest period
    let max_period = periods.iter().max().copied().unwrap_or(1);
    if values.len() < max_period * 2 {
        return Err(anyhow!(
            "Not enough observations ({}) for MSTL with period {}. Need at least {}",
            values.len(),
            max_period,
            max_period * 2
        ));
    }

    // Build MSTL model - periods should be Vec<usize>
    let periods_vec: Vec<usize> = periods.to_vec();
    let mstl = MSTLModel::new(periods_vec, NaiveTrend::new());

    // Fit the decomposition using the Fit trait
    let fitted = mstl
        .fit(&values)
        .map_err(|e| anyhow!("MSTL fitting failed: {}", e))?;

    // Extract the MSTL result containing the decomposition
    let decomposition = fitted.fit();
    let n = values.len();

    // Get trend component (augurs uses f32 internally, convert to f64)
    // Pre-allocate for efficiency
    let trend_f32 = decomposition.trend();
    let mut trend = Vec::with_capacity(n);
    trend.extend(trend_f32.iter().map(|&x| x as f64));

    // Get seasonal components - pre-allocate outer vec
    let seasonal_f32 = decomposition.seasonal();
    let mut seasonal = Vec::with_capacity(seasonal_f32.len());
    for s in seasonal_f32 {
        let mut season = Vec::with_capacity(n);
        season.extend(s.iter().map(|&x| x as f64));
        seasonal.push(season);
    }

    // Get residuals
    let remainder_f32 = decomposition.remainder();
    let mut residuals = Vec::with_capacity(n);
    residuals.extend(remainder_f32.iter().map(|&x| x as f64));

    Ok(MstlResult {
        column: column.to_string(),
        periods: periods.to_vec(),
        trend,
        seasonal,
        residuals,
        n_obs: values.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_validate_mstl_two_seasonal_components() {
        // Create data with two seasonal periods: period=7 and period=28
        // y_t = trend_t + S1_t(period=7) + S2_t(period=28) + noise
        let n = 7 * 28; // = 196, enough for both periods
        let mut y = Vec::with_capacity(n);
        for t in 0..n {
            let trend = 100.0 + 0.01 * t as f64;
            let season1 = 5.0 * (2.0 * std::f64::consts::PI * t as f64 / 7.0).sin();
            let season2 = 3.0 * (2.0 * std::f64::consts::PI * t as f64 / 28.0).sin();
            let noise = 0.1 * ((t * 17 + 3) % 11) as f64 / 11.0 - 0.05;
            y.push(trend + season1 + season2 + noise);
        }

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_mstl(&dataset, "y", &[7, 28]).unwrap();

        // Should extract two seasonal components
        assert_eq!(result.seasonal.len(), 2, "Should have 2 seasonal components");
        assert_eq!(result.seasonal[0].len(), n);
        assert_eq!(result.seasonal[1].len(), n);
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.residuals.len(), n);
        assert_eq!(result.n_obs, n);
    }

    #[test]
    fn test_validate_mstl_reconstruction() {
        // Verify that trend + sum(seasonal) + residuals == original series
        let n = 100;
        let period = 10;
        let mut y = Vec::with_capacity(n);
        for t in 0..n {
            let trend = 50.0 + 0.2 * t as f64;
            let season = 8.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
            let noise = 0.3 * ((t * 7 + 5) % 9) as f64 / 9.0 - 0.15;
            y.push(trend + season + noise);
        }

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_mstl(&dataset, "y", &[period]).unwrap();

        // Reconstruct: trend + seasonal[0] + residuals should approximate original
        let max_reconstruction_error: f64 = (0..n)
            .map(|t| {
                let reconstructed = result.trend[t]
                    + result.seasonal.iter().map(|s| s[t]).sum::<f64>()
                    + result.residuals[t];
                (reconstructed - y[t]).abs()
            })
            .fold(0.0f64, f64::max);

        assert!(
            max_reconstruction_error < 1e-4,
            "Reconstruction error should be near zero, got {:.6}",
            max_reconstruction_error
        );
    }

    #[test]
    fn test_validate_mstl_seasonal_has_correct_period() {
        // For data with strong period-12 seasonality, the extracted seasonal
        // component should exhibit period-12 repetition.
        let n = 120; // 10 cycles of period 12
        let period = 12;
        let mut y = Vec::with_capacity(n);
        for t in 0..n {
            let trend = 100.0;
            let season = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
            y.push(trend + season);
        }

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_mstl(&dataset, "y", &[period]).unwrap();

        // The seasonal component at time t and t+period should be similar
        // (after a few initial cycles to stabilize)
        let seasonal = &result.seasonal[0];
        let start = period * 2; // skip first 2 cycles
        let mut max_diff = 0.0f64;
        for t in start..(n - period) {
            let diff = (seasonal[t] - seasonal[t + period]).abs();
            max_diff = max_diff.max(diff);
        }

        assert!(
            max_diff < 2.0,
            "Seasonal component should repeat with period {}, max period-to-period diff = {:.4}",
            period,
            max_diff
        );
    }
}
