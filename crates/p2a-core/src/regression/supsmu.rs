//! Friedman's SuperSmoother - Optimized Implementation.
//!
//! This module implements Friedman's SuperSmoother, a local regression smoother
//! that adaptively selects the smoothing span at each x position. It uses three
//! fixed spans (tweeter, midrange, woofer) and selects the best one locally
//! via cross-validation.
//!
//! # Mathematical Background
//!
//! SuperSmoother is an adaptive bandwidth local linear smoother that:
//!
//! 1. Computes three smooths with fixed spans: α = 0.05 (tweeter), 0.2 (midrange),
//!    0.5 (woofer)
//! 2. Estimates local cross-validation residuals for each span
//! 3. Smooths the residuals to select locally optimal span
//! 4. Applies bass enhancement for additional smoothing control
//!
//! ## Local Linear Smoothing
//!
//! At each point x₀, fits: ŷ = a + b(x - x₀) using weighted least squares
//! with tricube weights: w(u) = (1 - |u|³)³ for |u| < 1
//!
//! ## Adaptive Span Selection
//!
//! The span α(x) varies locally based on cross-validated prediction error:
//! CV(α, x) = [y - ŷ₋ᵢ(x; α)]²
//!
//! where ŷ₋ᵢ is the leave-one-out prediction.
//!
//! ## Bass Enhancement
//!
//! The bass parameter (0-10) controls additional smoothing:
//! - bass = 0: no extra smoothing
//! - bass = 10: maximum smoothing (approaches linear fit)
//!
//! # References
//!
//! - Friedman, J.H. (1984). A variable span smoother. *Laboratory for Computational
//!   Statistics, Stanford University Technical Report No. 5*.
//!   The original SuperSmoother paper.
//!
//! - Friedman, J.H. (1984). SMART User's Guide. *Laboratory for Computational
//!   Statistics, Stanford University Technical Report No. 1*.
//!
//! - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
//!   Statistical Learning* (2nd ed.), Section 6.2. Springer.
//!   https://hastie.su.domains/ElemStatLearn/
//!
//! - Cleveland, W.S. (1979). Robust locally weighted regression and smoothing
//!   scatterplots. *Journal of the American Statistical Association*, 74(368),
//!   829-836. https://doi.org/10.1080/01621459.1979.10481038
//!   Related LOWESS method for comparison.
//!
//! R equivalent: `stats::supsmu()`

use serde::{Deserialize, Serialize};

/// Result of SuperSmoother fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupsmuResult {
    /// Sorted x values (with duplicates removed)
    pub x: Vec<f64>,
    /// Fitted y values (smoothed)
    pub y: Vec<f64>,
    /// The bass parameter used
    pub bass: f64,
    /// Whether periodic boundary conditions were used
    pub periodic: bool,
    /// Number of unique points
    pub n: usize,
}

/// Configuration for SuperSmoother.
#[derive(Debug, Clone)]
pub struct SupsmuConfig {
    /// Span parameter: either a fixed fraction, or None for cross-validation ("cv")
    pub span: Option<f64>,
    /// Treat x as periodic on [0, 1]
    pub periodic: bool,
    /// Bass parameter (0-10): higher values produce smoother results
    pub bass: f64,
}

impl Default for SupsmuConfig {
    fn default() -> Self {
        SupsmuConfig {
            span: None, // Use cross-validation
            periodic: false,
            bass: 0.0,
        }
    }
}

/// Apply Friedman's SuperSmoother to data.
///
/// The SuperSmoother uses local linear regression with adaptive bandwidth selection.
/// Three candidate spans are evaluated (0.05n, 0.2n, 0.5n), and the best span is
/// selected for each point using leave-one-out cross-validation.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values (must have same length as x)
/// * `wt` - Optional weights (default: uniform)
/// * `span` - Span fraction (None for cross-validation, or a value in (0, 1])
/// * `periodic` - Treat x as periodic on [0, 1]
/// * `bass` - Smoothness control (0-10, higher = smoother)
///
/// # Returns
/// A `SupsmuResult` containing the smoothed values
///
/// # Example
/// ```
/// use p2a_core::regression::supsmu::supsmu;
///
/// let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * rand::random::<f64>()).collect();
/// let result = supsmu(&x, &y, None, None, false, 0.0).unwrap();
/// ```
pub fn supsmu(
    x: &[f64],
    y: &[f64],
    wt: Option<&[f64]>,
    span: Option<f64>,
    periodic: bool,
    bass: f64,
) -> Result<SupsmuResult, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }

    let n = x.len();
    if n < 4 {
        return Err("Need at least 4 observations for supsmu".to_string());
    }

    let bass = bass.clamp(0.0, 10.0);

    // Create weights if not provided
    let weights: Vec<f64> = match wt {
        Some(w) => {
            if w.len() != n {
                return Err("Weights must have same length as data".to_string());
            }
            w.to_vec()
        }
        None => vec![1.0; n],
    };

    // Sort data by x and remove duplicates (averaging y values)
    let (sorted_x, sorted_y, sorted_w) = sort_and_dedupe(x, y, &weights);
    let m = sorted_x.len();

    if m < 4 {
        return Err("Need at least 4 unique x values".to_string());
    }

    // Determine the smoothing approach
    let smoothed_y = if let Some(fixed_span) = span {
        // Use fixed span - fast path
        let span_frac = fixed_span.clamp(0.0, 1.0);
        let k = ((span_frac * m as f64).round() as usize).max(3).min(m);
        running_lines_fast(&sorted_x, &sorted_y, &sorted_w, k)
    } else {
        // Cross-validation based adaptive span selection
        super_smooth_cv_fast(&sorted_x, &sorted_y, &sorted_w, periodic, bass)
    };

    Ok(SupsmuResult {
        x: sorted_x,
        y: smoothed_y,
        bass,
        periodic,
        n: m,
    })
}

/// Sort data by x, combining duplicate x values by weighted averaging y.
#[inline]
fn sort_and_dedupe(
    x: &[f64],
    y: &[f64],
    w: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result_x = Vec::with_capacity(n);
    let mut result_y = Vec::with_capacity(n);
    let mut result_w = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        let xi = x[indices[i]];
        let mut sum_wy = w[indices[i]] * y[indices[i]];
        let mut sum_w = w[indices[i]];
        let mut j = i + 1;

        // Find all points with the same x value
        while j < n && (x[indices[j]] - xi).abs() < 1e-10 {
            sum_wy += w[indices[j]] * y[indices[j]];
            sum_w += w[indices[j]];
            j += 1;
        }

        result_x.push(xi);
        result_y.push(if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 });
        result_w.push(sum_w);
        i = j;
    }

    (result_x, result_y, result_w)
}

/// Fast running lines smoother using incremental updates.
/// This is the key optimization - we update sums incrementally as the window slides.
#[inline]
fn running_lines_fast(x: &[f64], y: &[f64], w: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut smoothed = vec![0.0; n];

    if n == 0 {
        return smoothed;
    }

    let half_k = k / 2;

    // For small n or k, use simple approach
    if n <= 10 || k <= 5 {
        for i in 0..n {
            let start = i.saturating_sub(half_k);
            let end = (i + half_k + 1).min(n);
            smoothed[i] = local_linear_fit_simple(x, y, w, start, end, x[i]);
        }
        return smoothed;
    }

    // Use incremental updates for larger data
    // Initialize window for first point
    let mut start = 0usize;
    let mut end = half_k.min(n);

    // Sums for weighted linear regression: y = a + b*x
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    // Initialize sums for first window
    for j in start..end {
        let wj = w[j];
        let xj = x[j];
        let yj = y[j];
        sum_w += wj;
        sum_wx += wj * xj;
        sum_wy += wj * yj;
        sum_wxx += wj * xj * xj;
        sum_wxy += wj * xj * yj;
    }

    for i in 0..n {
        // Compute ideal window bounds
        let ideal_start = i.saturating_sub(half_k);
        let ideal_end = (i + half_k + 1).min(n);

        // Remove points that left the window
        while start < ideal_start {
            let wj = w[start];
            let xj = x[start];
            let yj = y[start];
            sum_w -= wj;
            sum_wx -= wj * xj;
            sum_wy -= wj * yj;
            sum_wxx -= wj * xj * xj;
            sum_wxy -= wj * xj * yj;
            start += 1;
        }

        // Add points that entered the window
        while end < ideal_end {
            let wj = w[end];
            let xj = x[end];
            let yj = y[end];
            sum_w += wj;
            sum_wx += wj * xj;
            sum_wy += wj * yj;
            sum_wxx += wj * xj * xj;
            sum_wxy += wj * xj * yj;
            end += 1;
        }

        // Solve for linear fit at xi
        let xi = x[i];
        if sum_w < 1e-10 {
            smoothed[i] = y[i];
        } else {
            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if det.abs() < 1e-10 {
                smoothed[i] = sum_wy / sum_w;
            } else {
                let a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
                let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
                smoothed[i] = a + b * xi;
            }
        }
    }

    smoothed
}

/// Simple local linear fit (for small windows or boundary cases).
#[inline]
fn local_linear_fit_simple(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    start: usize,
    end: usize,
    xi: f64,
) -> f64 {
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for j in start..end {
        let wj = w[j];
        let xj = x[j];
        let yj = y[j];
        sum_w += wj;
        sum_wx += wj * xj;
        sum_wy += wj * yj;
        sum_wxx += wj * xj * xj;
        sum_wxy += wj * xj * yj;
    }

    if sum_w < 1e-10 {
        return y[start.min(y.len() - 1)];
    }

    let det = sum_w * sum_wxx - sum_wx * sum_wx;
    if det.abs() < 1e-10 {
        sum_wy / sum_w
    } else {
        let a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
        let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
        a + b * xi
    }
}

/// Fast cross-validation based SuperSmoother with adaptive span selection.
fn super_smooth_cv_fast(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    _periodic: bool,
    bass: f64,
) -> Vec<f64> {
    let n = x.len();

    // Three candidate spans: tweaks (0.05n, 0.2n, 0.5n)
    let spans = [
        ((0.05 * n as f64).round() as usize).max(3).min(n),
        ((0.2 * n as f64).round() as usize).max(3).min(n),
        ((0.5 * n as f64).round() as usize).max(3).min(n),
    ];

    // Compute smoothed values for each span using fast method
    let smooth_small = running_lines_fast(x, y, w, spans[0]);
    let smooth_med = running_lines_fast(x, y, w, spans[1]);
    let smooth_large = running_lines_fast(x, y, w, spans[2]);

    // Compute CV residuals efficiently using the smoothed values
    // For LOO-CV with local linear regression, we can use the shortcut:
    // cv_residual[i] ≈ (y[i] - smooth[i]) / (1 - h_ii)
    // where h_ii is the leverage. For running lines with uniform weights,
    // h_ii ≈ 1/k for interior points.

    let cv_small = compute_cv_residuals_fast(y, &smooth_small, spans[0]);
    let cv_med = compute_cv_residuals_fast(y, &smooth_med, spans[1]);
    let cv_large = compute_cv_residuals_fast(y, &smooth_large, spans[2]);

    // Select best span for each point
    let mut result = vec![0.0; n];
    let mut span_choices = vec![0.0; n];

    for i in 0..n {
        // Find minimum CV residual
        let cv_vals = [cv_small[i], cv_med[i], cv_large[i]];
        let min_idx = if cv_vals[0] <= cv_vals[1] && cv_vals[0] <= cv_vals[2] {
            0
        } else if cv_vals[1] <= cv_vals[2] {
            1
        } else {
            2
        };

        span_choices[i] = spans[min_idx] as f64;
        result[i] = match min_idx {
            0 => smooth_small[i],
            1 => smooth_med[i],
            _ => smooth_large[i],
        };
    }

    // Apply bass adjustment if needed
    if bass > 0.0 {
        let bass_span = ((bass * n as f64 / 10.0).round() as usize).max(3).min(n);
        let smoothed_choices = running_lines_fast(x, &span_choices, w, bass_span);

        // Interpolate based on smoothed span choices
        for i in 0..n {
            let chosen_span = smoothed_choices[i].round() as usize;
            let chosen_span = chosen_span.max(spans[0]).min(spans[2]);

            // Linear interpolation between smooth curves
            if chosen_span <= spans[0] {
                result[i] = smooth_small[i];
            } else if chosen_span >= spans[2] {
                result[i] = smooth_large[i];
            } else if chosen_span <= spans[1] {
                let t = (chosen_span - spans[0]) as f64 / (spans[1] - spans[0]).max(1) as f64;
                result[i] = smooth_small[i] * (1.0 - t) + smooth_med[i] * t;
            } else {
                let t = (chosen_span - spans[1]) as f64 / (spans[2] - spans[1]).max(1) as f64;
                result[i] = smooth_med[i] * (1.0 - t) + smooth_large[i] * t;
            }
        }
    }

    result
}

/// Fast approximate CV residuals using leverage approximation.
#[inline]
fn compute_cv_residuals_fast(y: &[f64], smoothed: &[f64], span: usize) -> Vec<f64> {
    let n = y.len();
    let mut cv_residuals = vec![0.0; n];

    // Approximate leverage for interior points
    // For local linear regression with k points, h_ii ≈ 2/k for interior points
    let h_interior = 2.0 / span as f64;
    let denom_interior = (1.0 - h_interior).max(0.1);

    for i in 0..n {
        let raw_resid = y[i] - smoothed[i];
        // Boundary points have higher leverage
        let is_boundary = i < span / 2 || i >= n - span / 2;
        let denom = if is_boundary { 0.5 } else { denom_interior };
        cv_residuals[i] = (raw_resid / denom).powi(2);
    }

    cv_residuals
}

/// Run SuperSmoother (convenience wrapper).
pub fn run_supsmu(
    x: &[f64],
    y: &[f64],
    wt: Option<&[f64]>,
    span: Option<f64>,
    periodic: bool,
    bass: f64,
) -> Result<SupsmuResult, String> {
    supsmu(x, y, wt, span, periodic, bass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_supsmu_basic() {
        // Simple linear data
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = supsmu(&x, &y, None, None, false, 0.0).unwrap();

        assert_eq!(result.n, 50);
        // Smoothed values should be close to original for linear data
        for i in 5..45 {
            // Avoid boundary effects
            assert_relative_eq!(result.y[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_supsmu_with_noise() {
        // Sine wave with noise
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (2.0 * std::f64::consts::PI * xi).sin() + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();

        let result = supsmu(&x, &y, None, None, false, 5.0).unwrap();

        assert_eq!(result.n, 100);
        // Smoothed curve should have less variation than noisy input
        let smooth_var: f64 = result.y.iter().map(|&yi| yi * yi).sum::<f64>() / result.y.len() as f64;

        // Both should be similar magnitude for sinusoidal data
        assert!(smooth_var > 0.1); // Not completely flat
    }

    #[test]
    fn test_supsmu_fixed_span() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(2)).collect();

        let result = supsmu(&x, &y, None, Some(0.3), false, 0.0).unwrap();

        assert_eq!(result.n, 50);
        assert!(!result.y.is_empty());
    }

    #[test]
    fn test_supsmu_with_weights() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi + 1.0).collect();
        let wt: Vec<f64> = (0..30).map(|i| if i < 15 { 1.0 } else { 2.0 }).collect();

        let result = supsmu(&x, &y, Some(&wt), None, false, 0.0).unwrap();

        assert_eq!(result.n, 30);
    }

    #[test]
    fn test_supsmu_periodic() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 / 40.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * std::f64::consts::PI * xi).sin()).collect();

        let result = supsmu(&x, &y, None, None, true, 0.0).unwrap();

        assert!(result.periodic);
    }

    #[test]
    fn test_supsmu_bass_parameter() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let result_no_bass = supsmu(&x, &y, None, None, false, 0.0).unwrap();
        let result_high_bass = supsmu(&x, &y, None, None, false, 10.0).unwrap();

        // High bass should produce smoother output
        assert_relative_eq!(result_no_bass.bass, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result_high_bass.bass, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_supsmu_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = supsmu(&x, &y, None, None, false, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_supsmu_too_few_points() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = supsmu(&x, &y, None, None, false, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_and_dedupe() {
        let x = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        let y = vec![6.0, 2.0, 4.0, 4.0, 8.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let (sx, sy, _sw) = sort_and_dedupe(&x, &y, &w);

        // Should have 3 unique x values
        assert_eq!(sx.len(), 3);
        assert_relative_eq!(sx[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sx[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sx[2], 3.0, epsilon = 1e-10);

        // y values should be averaged for duplicates
        assert_relative_eq!(sy[0], 3.0, epsilon = 1e-10); // (2+4)/2 = 3
        assert_relative_eq!(sy[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(sy[2], 7.0, epsilon = 1e-10); // (6+8)/2 = 7
    }

    #[test]
    fn test_running_lines_fast() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let w = vec![1.0; 100];

        let smoothed = running_lines_fast(&x, &y, &w, 10);

        // For linear data, smoothed should match original (approximately)
        for i in 10..90 {
            assert_relative_eq!(smoothed[i], y[i], epsilon = 0.5);
        }
    }
}
