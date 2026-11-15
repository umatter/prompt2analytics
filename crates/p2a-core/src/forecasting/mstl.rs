//! MSTL (Multiple Seasonal-Trend decomposition using LOESS).
//!
//! The decomposition runs in `augurs-mstl`, which computes in f32. The series is centred by its
//! mean and scaled by its standard deviation before the fit and the components are restored
//! afterwards, so their precision follows the series' spread and not its level; the residuals
//! are computed in f64 as x - T - sum S.

use anyhow::{Result, anyhow};
use augurs_core::Fit;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};

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
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
pub fn run_mstl(dataset: &Dataset, column: &str, periods: &[usize]) -> Result<MstlResult> {
    use augurs_mstl::{MSTLModel, NaiveTrend};

    // Extract time series data (an EconError inside the anyhow error)
    let (values, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

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

    // augurs computes in f32: centre and scale first (a constant series is only centred).
    let n = values.len();
    let center = values.iter().sum::<f64>() / n as f64;
    let spread = if n > 1 {
        (values.iter().map(|v| (v - center).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let scale = if spread > 0.0 && spread.is_finite() {
        spread
    } else {
        1.0
    };
    let standardised: Vec<f64> = values.iter().map(|v| (v - center) / scale).collect();

    // Fit the decomposition using the Fit trait
    let fitted = mstl
        .fit(&standardised)
        .map_err(|e| anyhow!("MSTL fitting failed: {}", e))?;

    // Extract the MSTL result containing the decomposition
    let decomposition = fitted.fit();

    // Restore the components (augurs returns f32): T = s T' + c, S_i = s S_i'.
    let trend_f32 = decomposition.trend();
    let mut trend = Vec::with_capacity(n);
    trend.extend(trend_f32.iter().map(|&x| scale * x as f64 + center));

    let seasonal_f32 = decomposition.seasonal();
    let mut seasonal = Vec::with_capacity(seasonal_f32.len());
    for s in seasonal_f32 {
        let mut season = Vec::with_capacity(n);
        season.extend(s.iter().map(|&x| scale * x as f64));
        seasonal.push(season);
    }

    // Residuals in f64: x - T - sum S.
    let mut residuals = Vec::with_capacity(n);
    for (i, (&x, &t)) in values.iter().zip(&trend).enumerate() {
        residuals.push(x - t - seasonal.iter().map(|s| s[i]).sum::<f64>());
    }

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

    /// X-2: an Int64 column decomposes as its Float64 copy; missing values, text and
    /// absent columns are refused by name (EconError inside the anyhow error).
    #[test]
    fn test_na_mstl_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let run = |col: &str| run_mstl(&ds, col, &[12]);
        let int = run("ai").unwrap();
        let flt = run("ai_f64").unwrap();
        assert_same(&int.trend, &flt.trend);
        assert_null_values(econ(&run("a_na").unwrap_err()), "a_na", 2);
        assert_null_values(econ(&run("ai_na").unwrap_err()), "ai_na", 2);
        assert_non_numeric(econ(&run("lab").unwrap_err()), "lab");
        assert_not_found(econ(&run("nope").unwrap_err()), "nope");
    }

    /// The decomposition does not depend on the series' level: y and y + 1e8 give the same
    /// seasonal components and residuals, and trends 1e8 apart, to 1e-5 of the spread.
    #[test]
    fn test_mstl_level_invariance() {
        let y: Vec<f64> = (0..144)
            .map(|t| {
                let t = t as f64;
                1000.0
                    + 2.0 * t
                    + 50.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin()
                    + 8.0 * ((t * 7.3).sin() * 43.1).sin()
            })
            .collect();
        let big: Vec<f64> = y.iter().map(|v| v + 1e8).collect();
        let m = y.iter().sum::<f64>() / y.len() as f64;
        let sd = (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (y.len() - 1) as f64).sqrt();
        let ds = |v: &[f64]| Dataset::new(df! { "y" => v }.unwrap());
        let a = run_mstl(&ds(&y), "y", &[12]).unwrap();
        let b = run_mstl(&ds(&big), "y", &[12]).unwrap();
        let tol = 1e-5 * sd;
        for i in 0..y.len() {
            assert!((b.trend[i] - 1e8 - a.trend[i]).abs() <= tol, "trend {i}");
            assert!(
                (b.seasonal[0][i] - a.seasonal[0][i]).abs() <= tol,
                "seasonal {i}"
            );
            assert!(
                (b.residuals[i] - a.residuals[i]).abs() <= tol,
                "residual {i}"
            );
            let back = b.trend[i] + b.seasonal[0][i] + b.residuals[i];
            assert!((big[i] - back).abs() <= 1e-9 * 1e8, "reconstruction {i}");
        }
    }

    #[test]
    fn test_mstl_two_seasonal_components_properties() {
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
        assert_eq!(
            result.seasonal.len(),
            2,
            "Should have 2 seasonal components"
        );
        assert_eq!(result.seasonal[0].len(), n);
        assert_eq!(result.seasonal[1].len(), n);
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.residuals.len(), n);
        assert_eq!(result.n_obs, n);
    }

    #[test]
    fn test_mstl_reconstruction_properties() {
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
    fn test_mstl_seasonal_has_correct_period_properties() {
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
