//! Changepoint detection for time series.
//!
//! Implements the PELT (Pruned Exact Linear Time) algorithm for detecting
//! multiple changepoints in time series data.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;

/// Result of changepoint detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangepointResult {
    /// Indices of detected changepoints (0-indexed positions where changes occur)
    pub changepoints: Vec<usize>,
    /// Number of changepoints detected
    pub n_changepoints: usize,
    /// Segment statistics (mean and variance for each segment)
    pub segments: Vec<SegmentStats>,
    /// Total cost (negative log-likelihood) of the segmentation
    pub total_cost: f64,
    /// Penalty value used
    pub penalty: f64,
    /// Method used for detection
    pub method: String,
}

/// Statistics for a segment between changepoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentStats {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Number of points in segment
    pub n_points: usize,
    /// Mean of the segment
    pub mean: f64,
    /// Variance of the segment
    pub variance: f64,
}

/// Cost function type for changepoint detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostFunction {
    /// Detect changes in mean (assumes constant variance)
    MeanChange,
    /// Detect changes in variance (assumes constant mean)
    VarianceChange,
    /// Detect changes in both mean and variance
    MeanAndVariance,
}

impl Default for CostFunction {
    fn default() -> Self {
        CostFunction::MeanChange
    }
}

/// Detect changepoints in a time series using the PELT algorithm.
///
/// # Arguments
/// * `data` - Time series data
/// * `penalty` - Penalty for adding a changepoint (use None for automatic BIC penalty)
/// * `min_segment_length` - Minimum length of a segment (default: 2)
/// * `cost_function` - Type of change to detect
///
/// # Returns
/// * `ChangepointResult` with detected changepoints and segment statistics
pub fn detect_changepoints(
    data: &[f64],
    penalty: Option<f64>,
    min_segment_length: Option<usize>,
    cost_function: CostFunction,
) -> Result<ChangepointResult, String> {
    let n = data.len();

    if n < 2 {
        return Err("Data must have at least 2 points".to_string());
    }

    let min_seg = min_segment_length.unwrap_or(2).max(2);

    if n < min_seg * 2 {
        return Err(format!(
            "Data length ({}) must be at least 2x minimum segment length ({})",
            n, min_seg
        ));
    }

    // Default penalty: BIC-style penalty
    let pen = penalty.unwrap_or_else(|| {
        let log_n = (n as f64).ln();
        match cost_function {
            CostFunction::MeanChange => log_n,
            CostFunction::VarianceChange => log_n,
            CostFunction::MeanAndVariance => 2.0 * log_n,
        }
    });

    // Precompute cumulative sums for efficient cost calculation
    let (cum_sum, cum_sum_sq) = compute_cumulative_sums(data);

    // PELT algorithm
    let changepoints = pelt(
        data,
        &cum_sum,
        &cum_sum_sq,
        pen,
        min_seg,
        cost_function,
    );

    // Compute segment statistics
    let segments = compute_segment_stats(data, &changepoints);

    // Compute total cost
    let total_cost = compute_total_cost(data, &changepoints, cost_function);

    let method = match cost_function {
        CostFunction::MeanChange => "PELT (mean change)",
        CostFunction::VarianceChange => "PELT (variance change)",
        CostFunction::MeanAndVariance => "PELT (mean and variance change)",
    };

    Ok(ChangepointResult {
        n_changepoints: changepoints.len(),
        changepoints,
        segments,
        total_cost,
        penalty: pen,
        method: method.to_string(),
    })
}

/// Compute cumulative sums for efficient segment cost calculation
fn compute_cumulative_sums(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut cum_sum = vec![0.0; n + 1];
    let mut cum_sum_sq = vec![0.0; n + 1];

    for i in 0..n {
        cum_sum[i + 1] = cum_sum[i] + data[i];
        cum_sum_sq[i + 1] = cum_sum_sq[i] + data[i] * data[i];
    }

    (cum_sum, cum_sum_sq)
}

/// Compute the cost of a segment [start, end) using cumulative sums
fn segment_cost(
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    start: usize,
    end: usize,
    cost_function: CostFunction,
) -> f64 {
    let n = (end - start) as f64;
    if n <= 0.0 {
        return f64::INFINITY;
    }

    let sum = cum_sum[end] - cum_sum[start];
    let sum_sq = cum_sum_sq[end] - cum_sum_sq[start];
    let mean = sum / n;
    let variance = (sum_sq / n) - mean * mean;

    match cost_function {
        CostFunction::MeanChange => {
            // Negative log-likelihood assuming known variance
            // Cost = n * log(variance) where variance is computed from residuals
            let ss = sum_sq - sum * sum / n;
            if ss <= 0.0 {
                0.0
            } else {
                ss // Sum of squared residuals from mean
            }
        }
        CostFunction::VarianceChange => {
            // Cost for variance change detection
            let var = variance.max(1e-10);
            n * var.ln() + n
        }
        CostFunction::MeanAndVariance => {
            // Cost for both mean and variance change
            let var = variance.max(1e-10);
            n * (var.ln() + 1.0)
        }
    }
}

/// PELT (Pruned Exact Linear Time) algorithm for multiple changepoint detection
fn pelt(
    data: &[f64],
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    penalty: f64,
    min_seg: usize,
    cost_function: CostFunction,
) -> Vec<usize> {
    let n = data.len();

    // F[t] = minimum cost of segmenting data[0..t]
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = -penalty; // So that the first segment doesn't get penalized twice

    // cp[t] = last changepoint before t in the optimal segmentation
    let mut cp: Vec<Option<usize>> = vec![None; n + 1];

    // R = set of candidate changepoint positions (pruning set)
    let mut r: Vec<usize> = vec![0];

    for t in min_seg..=n {
        // Find the best last changepoint for position t
        let mut best_cost = f64::INFINITY;
        let mut best_cp = None;

        for &s in &r {
            if t - s >= min_seg {
                let cost = f[s] + segment_cost(cum_sum, cum_sum_sq, s, t, cost_function) + penalty;
                if cost < best_cost {
                    best_cost = cost;
                    best_cp = Some(s);
                }
            }
        }

        f[t] = best_cost;
        cp[t] = best_cp;

        // Pruning step: remove candidates that can never be optimal
        // Keep s if F[s] + C(y_{s+1:t}) <= F[t]
        let mut new_r = Vec::new();
        for &s in &r {
            if s + min_seg <= t {
                let cost_to_t = f[s] + segment_cost(cum_sum, cum_sum_sq, s, t, cost_function);
                if cost_to_t <= f[t] {
                    new_r.push(s);
                }
            } else {
                new_r.push(s);
            }
        }
        new_r.push(t);
        r = new_r;
    }

    // Backtrack to find changepoints
    let mut changepoints = Vec::new();
    let mut t = n;

    while let Some(s) = cp[t] {
        if s > 0 {
            changepoints.push(s);
        }
        t = s;
    }

    changepoints.reverse();
    changepoints
}

/// Compute statistics for each segment
fn compute_segment_stats(data: &[f64], changepoints: &[usize]) -> Vec<SegmentStats> {
    let n = data.len();
    let mut segments = Vec::new();

    // Create segment boundaries
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(changepoints.iter().copied());
    boundaries.push(n);

    for i in 0..boundaries.len() - 1 {
        let start = boundaries[i];
        let end = boundaries[i + 1];
        let segment_data = &data[start..end];

        let n_points = segment_data.len();
        let mean = segment_data.iter().sum::<f64>() / n_points as f64;
        let variance = segment_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n_points as f64;

        segments.push(SegmentStats {
            start,
            end,
            n_points,
            mean,
            variance,
        });
    }

    segments
}

/// Compute total cost of the segmentation
fn compute_total_cost(data: &[f64], changepoints: &[usize], cost_function: CostFunction) -> f64 {
    let (cum_sum, cum_sum_sq) = compute_cumulative_sums(data);
    let n = data.len();

    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(changepoints.iter().copied());
    boundaries.push(n);

    let mut total = 0.0;
    for i in 0..boundaries.len() - 1 {
        total += segment_cost(&cum_sum, &cum_sum_sq, boundaries[i], boundaries[i + 1], cost_function);
    }

    total
}

/// Binary segmentation algorithm (simpler alternative to PELT)
/// Recursively finds the single best changepoint in each segment
pub fn binary_segmentation(
    data: &[f64],
    max_changepoints: Option<usize>,
    min_segment_length: Option<usize>,
    significance_threshold: Option<f64>,
) -> Result<ChangepointResult, String> {
    let n = data.len();

    if n < 2 {
        return Err("Data must have at least 2 points".to_string());
    }

    let min_seg = min_segment_length.unwrap_or(2).max(2);
    let max_cp = max_changepoints.unwrap_or(10);
    let threshold = significance_threshold.unwrap_or(0.0);

    let (cum_sum, cum_sum_sq) = compute_cumulative_sums(data);

    let mut changepoints = Vec::new();
    let mut segments_to_check: Vec<(usize, usize)> = vec![(0, n)];

    while !segments_to_check.is_empty() && changepoints.len() < max_cp {
        // Find segment with best potential changepoint
        let mut best_gain = threshold;
        let mut best_cp = None;
        let mut best_segment_idx = 0;

        for (idx, &(start, end)) in segments_to_check.iter().enumerate() {
            if end - start < 2 * min_seg {
                continue;
            }

            let base_cost = segment_cost(&cum_sum, &cum_sum_sq, start, end, CostFunction::MeanChange);

            for cp in (start + min_seg)..(end - min_seg + 1) {
                let left_cost = segment_cost(&cum_sum, &cum_sum_sq, start, cp, CostFunction::MeanChange);
                let right_cost = segment_cost(&cum_sum, &cum_sum_sq, cp, end, CostFunction::MeanChange);
                let gain = base_cost - left_cost - right_cost;

                if gain > best_gain {
                    best_gain = gain;
                    best_cp = Some(cp);
                    best_segment_idx = idx;
                }
            }
        }

        if let Some(cp) = best_cp {
            let (start, end) = segments_to_check.remove(best_segment_idx);
            changepoints.push(cp);
            segments_to_check.push((start, cp));
            segments_to_check.push((cp, end));
        } else {
            break;
        }
    }

    changepoints.sort();

    let segments = compute_segment_stats(data, &changepoints);
    let total_cost = compute_total_cost(data, &changepoints, CostFunction::MeanChange);

    // Use BIC as penalty estimate
    let penalty = (n as f64).ln();

    Ok(ChangepointResult {
        n_changepoints: changepoints.len(),
        changepoints,
        segments,
        total_cost,
        penalty,
        method: "Binary Segmentation".to_string(),
    })
}

/// Run changepoint detection on a dataset column.
///
/// This is a convenience wrapper that extracts the column data from a Dataset
/// and runs the changepoint detection algorithm.
pub fn run_changepoint(
    dataset: &Dataset,
    column: &str,
    penalty: Option<f64>,
    min_segment_length: Option<usize>,
    cost_function: CostFunction,
) -> Result<ChangepointResult> {
    // Extract time series data from the dataset
    let df = dataset.df();
    let col = df.column(column).map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

    // Try to get as f64 first, then i64
    let values: Vec<f64> = match col.f64() {
        Ok(ca) => ca.into_no_null_iter().collect(),
        Err(_) => {
            // Try i64
            match col.i64() {
                Ok(ca) => ca.into_no_null_iter().map(|v| v as f64).collect(),
                Err(_) => {
                    return Err(anyhow!("Column '{}' must be numeric (f64 or i64)", column));
                }
            }
        }
    };

    if values.len() < 4 {
        return Err(anyhow!(
            "Time series must have at least 4 observations, found {}",
            values.len()
        ));
    }

    detect_changepoints(&values, penalty, min_segment_length, cost_function)
        .map_err(|e| anyhow!(e))
}

/// Run binary segmentation changepoint detection on a dataset column.
pub fn run_binary_segmentation(
    dataset: &Dataset,
    column: &str,
    max_changepoints: Option<usize>,
    min_segment_length: Option<usize>,
    significance_threshold: Option<f64>,
) -> Result<ChangepointResult> {
    // Extract time series data from the dataset
    let df = dataset.df();
    let col = df.column(column).map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

    // Try to get as f64 first, then i64
    let values: Vec<f64> = match col.f64() {
        Ok(ca) => ca.into_no_null_iter().collect(),
        Err(_) => {
            // Try i64
            match col.i64() {
                Ok(ca) => ca.into_no_null_iter().map(|v| v as f64).collect(),
                Err(_) => {
                    return Err(anyhow!("Column '{}' must be numeric (f64 or i64)", column));
                }
            }
        }
    };

    if values.len() < 4 {
        return Err(anyhow!(
            "Time series must have at least 4 observations, found {}",
            values.len()
        ));
    }

    binary_segmentation(&values, max_changepoints, min_segment_length, significance_threshold)
        .map_err(|e| anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_changepoint() {
        // Constant data - no changepoints expected
        let data: Vec<f64> = vec![1.0; 50];
        let result = detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();
        assert_eq!(result.n_changepoints, 0);
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn test_single_changepoint() {
        // Clear mean shift at position 25
        let mut data: Vec<f64> = vec![0.0; 25];
        data.extend(vec![10.0; 25]);

        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Should detect changepoint around position 25
        assert!(result.n_changepoints >= 1, "Should detect at least one changepoint");
        if result.n_changepoints > 0 {
            let cp = result.changepoints[0];
            assert!((cp as i32 - 25).abs() <= 2, "Changepoint should be near position 25, got {}", cp);
        }
    }

    #[test]
    fn test_multiple_changepoints() {
        // Three segments with different means
        let mut data: Vec<f64> = vec![0.0; 30];
        data.extend(vec![5.0; 30]);
        data.extend(vec![0.0; 30]);

        let result = detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();

        // Should detect 2 changepoints
        assert_eq!(result.n_changepoints, 2, "Should detect 2 changepoints");
        assert_eq!(result.segments.len(), 3, "Should have 3 segments");
    }

    #[test]
    fn test_binary_segmentation() {
        let mut data: Vec<f64> = vec![0.0; 25];
        data.extend(vec![10.0; 25]);

        let result = binary_segmentation(&data, Some(5), None, Some(10.0)).unwrap();

        assert!(result.n_changepoints >= 1);
    }

    #[test]
    fn test_segment_stats() {
        let mut data: Vec<f64> = vec![1.0; 20];
        data.extend(vec![5.0; 20]);

        let changepoints = vec![20];
        let segments = compute_segment_stats(&data, &changepoints);

        assert_eq!(segments.len(), 2);
        assert!((segments[0].mean - 1.0).abs() < 0.01);
        assert!((segments[1].mean - 5.0).abs() < 0.01);
    }
}
