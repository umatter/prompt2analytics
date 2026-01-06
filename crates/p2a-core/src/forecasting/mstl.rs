//! MSTL (Multiple Seasonal-Trend decomposition using LOESS).

use anyhow::{anyhow, Result};
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
pub fn run_mstl(
    dataset: &Dataset,
    column: &str,
    periods: &[usize],
) -> Result<MstlResult> {
    use augurs_mstl::{MSTLModel, NaiveTrend};

    // Extract time series data
    let df = dataset.df();
    let col = df.column(column).map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

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
    let fitted = mstl.fit(&values)
        .map_err(|e| anyhow!("MSTL fitting failed: {}", e))?;

    // Extract the MSTL result containing the decomposition
    let decomposition = fitted.fit();

    // Get trend component (convert from f32 to f64)
    let trend: Vec<f64> = decomposition.trend().iter().map(|&x| x as f64).collect();

    // Get seasonal components (convert from Vec<f32> to Vec<f64>)
    let seasonal: Vec<Vec<f64>> = decomposition.seasonal()
        .iter()
        .map(|s| s.iter().map(|&x| x as f64).collect())
        .collect();

    // Get residuals (convert from f32 to f64)
    let residuals: Vec<f64> = decomposition.remainder().iter().map(|&x| x as f64).collect();

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
    // MSTL tests would require generating synthetic seasonal data
}
