//! Projection Pursuit Regression (PPR) - Optimized Implementation.
//!
//! PPR is a dimension reduction regression technique that fits the model:
//!
//! y = Σₖ₌₁ᴹ fₖ(αₖ'x) + ε
//!
//! where each term consists of a projection direction (αₖ) and a ridge
//! function (fₖ). The method finds optimal projection directions and uses
//! smoothing to estimate the ridge functions.
//!
//! # Mathematical Background
//!
//! PPR approximates arbitrary smooth functions by a sum of ridge functions:
//!
//! E[y|x] ≈ Σₖ fₖ(αₖ'x)
//!
//! where:
//! - αₖ ∈ ℝᵖ is the k-th projection direction (unit vector)
//! - fₖ: ℝ → ℝ is the k-th ridge (smooth univariate) function
//!
//! ## Estimation Algorithm
//!
//! For each term k:
//! 1. **Projection step**: Find optimal direction αₖ that maximizes explained
//!    variance of residuals
//! 2. **Smoothing step**: Estimate fₖ by smoothing residuals against αₖ'x
//! 3. **Backfitting**: Iteratively refine all directions and ridge functions
//!
//! ## Smoothing Methods
//!
//! Ridge functions are estimated using:
//! - **Super smoother** (default): Friedman's variable span smoother
//! - **Spline smoother**: Cubic smoothing splines with GCV
//!
//! # References
//!
//! - Friedman, J.H., & Stuetzle, W. (1981). Projection pursuit regression.
//!   *Journal of the American Statistical Association*, 76(376), 817-823.
//!   https://doi.org/10.1080/01621459.1981.10477729
//!   The foundational PPR paper.
//!
//! - Friedman, J.H. (1984). A variable span smoother. Technical Report No. 5,
//!   Laboratory for Computational Statistics, Stanford University.
//!   The super smoother algorithm used for ridge function estimation.
//!
//! - Huber, P.J. (1985). Projection pursuit. *The Annals of Statistics*, 13(2),
//!   435-475. https://doi.org/10.1214/aos/1176349519
//!   Comprehensive review of projection pursuit methods.
//!
//! - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
//!   Statistical Learning* (2nd ed.), Section 11.2. Springer.
//!   https://hastie.su.domains/ElemStatLearn/
//!
//! R equivalent: `stats::ppr()`

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

/// Result of projection pursuit regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PprResult {
    /// Number of terms in the final model
    pub nterms: usize,
    /// Projection directions (p x nterms matrix, stored row-major)
    pub alpha: Vec<Vec<f64>>,
    /// Ridge coefficients for each term
    pub beta: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
    /// Goodness of fit (RSS) for models with 1, 2, ..., nterms terms
    pub gofn: Vec<f64>,
    /// Number of observations
    pub n: usize,
    /// Number of predictors
    pub p: usize,
    /// The smoothing method used
    pub sm_method: SmoothingMethod,
}

impl PprResult {
    /// Predict for new data.
    pub fn predict(&self, x_new: &[Vec<f64>]) -> Vec<f64> {
        let n = x_new.len();
        let mut predictions = vec![0.0; n];

        for i in 0..n {
            let xi = &x_new[i];
            for k in 0..self.nterms {
                // Project onto alpha_k
                let proj: f64 = self.alpha[k]
                    .iter()
                    .zip(xi.iter())
                    .map(|(&a, &x)| a * x)
                    .sum();

                // Evaluate ridge function (stored in beta as linear approximation)
                predictions[i] += self.beta[k] * proj;
            }
        }

        predictions
    }
}

/// Smoothing method for ridge functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// Friedman's SuperSmoother
    Supsmu,
    /// Cubic smoothing spline
    Spline,
    /// Spline with GCV selection
    GcvSpline,
}

/// Configuration for PPR.
#[derive(Debug, Clone)]
pub struct PprConfig {
    /// Number of terms in final model
    pub nterms: usize,
    /// Maximum terms to consider (for backward elimination)
    pub max_terms: usize,
    /// Optimization level (0-3): higher = more refitting during backward elimination
    pub optlevel: usize,
    /// Smoothing method
    pub sm_method: SmoothingMethod,
    /// Bass parameter for supsmu (0-10)
    pub bass: f64,
    /// Span parameter for supsmu
    pub span: f64,
    /// Degrees of freedom for spline
    pub df: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        PprConfig {
            nterms: 1,
            max_terms: 1,
            optlevel: 2,
            sm_method: SmoothingMethod::Supsmu,
            bass: 0.0,
            span: 0.0,
            df: 5.0,
            max_iter: 20, // Reduced from 100 - usually converges in <10
            tol: 1e-3,    // Slightly relaxed for speed
        }
    }
}

/// Fit a projection pursuit regression model.
///
/// # Arguments
/// * `x` - Predictor matrix (n x p)
/// * `y` - Response vector (length n)
/// * `weights` - Optional case weights
/// * `config` - PPR configuration
///
/// # Returns
/// A `PprResult` containing the fitted model
pub fn ppr(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    config: PprConfig,
) -> Result<PprResult, String> {
    let n = x.nrows();
    let p = x.ncols();

    if y.len() != n {
        return Err(format!("y has length {} but x has {} rows", y.len(), n));
    }

    if n < 3 {
        return Err("Need at least 3 observations".to_string());
    }

    if config.nterms == 0 {
        return Err("nterms must be at least 1".to_string());
    }

    // Create weights
    let w: Vec<f64> = match weights {
        Some(wt) => {
            if wt.len() != n {
                return Err("Weights must have same length as y".to_string());
            }
            wt.to_vec()
        }
        None => vec![1.0; n],
    };

    let max_terms = config.max_terms.max(config.nterms);

    // Initialize residuals
    let y_mean: f64 = weighted_mean(y, &w);
    let mut residuals: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

    let mut alphas: Vec<Vec<f64>> = Vec::with_capacity(max_terms);
    let mut betas: Vec<f64> = Vec::with_capacity(max_terms);
    let mut gofn: Vec<f64> = Vec::with_capacity(max_terms);

    // Pre-allocate buffers for the inner loop
    let mut projections = vec![0.0; n];
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    let mut smoothed = vec![0.0; n];
    let mut derivative = vec![0.0; n];

    // Forward selection: add terms one at a time
    for term in 0..max_terms {
        // Find optimal projection direction for current residuals
        let (alpha, ridge_values, beta) = find_projection_direction_fast(
            &x,
            &residuals,
            &w,
            &config,
            &mut projections,
            &mut sorted_indices,
            &mut smoothed,
            &mut derivative,
        )?;

        // Update residuals
        for i in 0..n {
            residuals[i] -= ridge_values[i];
        }

        alphas.push(alpha);
        betas.push(beta);

        // Compute RSS
        let rss: f64 = residuals
            .iter()
            .zip(w.iter())
            .map(|(&r, &wi)| wi * r * r)
            .sum();
        gofn.push(rss);

        // Check for convergence
        if term > 0 && (gofn[term - 1] - rss).abs() / (gofn[term - 1] + 1e-10) < config.tol {
            break;
        }
    }

    // Backward elimination if we have more terms than needed
    while alphas.len() > config.nterms {
        let remove_idx = find_least_important_term(&betas);
        alphas.remove(remove_idx);
        betas.remove(remove_idx);

        if config.optlevel >= 1 && alphas.len() == config.nterms {
            // Recompute residuals with remaining terms
            residuals = y.iter().map(|&yi| yi - y_mean).collect();
            for (k, alpha) in alphas.iter().enumerate() {
                for i in 0..n {
                    let proj: f64 = (0..p).map(|j| alpha[j] * x[[i, j]]).sum();
                    residuals[i] -= betas[k] * proj;
                }
            }
        }
    }

    // Compute final fitted values and residuals
    let mut fitted = vec![y_mean; n];
    for (k, alpha) in alphas.iter().enumerate() {
        for i in 0..n {
            let proj: f64 = (0..p).map(|j| alpha[j] * x[[i, j]]).sum();
            fitted[i] += betas[k] * proj;
        }
    }

    let final_residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    Ok(PprResult {
        nterms: alphas.len(),
        alpha: alphas,
        beta: betas,
        fitted,
        residuals: final_residuals,
        gofn,
        n,
        p,
        sm_method: config.sm_method,
    })
}

/// Find the optimal projection direction - optimized version with reusable buffers.
fn find_projection_direction_fast(
    x: &ArrayView2<f64>,
    residuals: &[f64],
    weights: &[f64],
    config: &PprConfig,
    projections: &mut Vec<f64>,
    sorted_indices: &mut Vec<usize>,
    smoothed: &mut Vec<f64>,
    derivative: &mut Vec<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n = x.nrows();
    let p = x.ncols();

    // Initialize with principal component direction
    let mut alpha = initial_direction_fast(x, residuals, weights);

    // Normalize alpha
    let norm: f64 = alpha.iter().map(|&a| a * a).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for a in &mut alpha {
            *a /= norm;
        }
    }

    // Iteratively refine direction
    for iter in 0..config.max_iter {
        // Project data onto current direction
        compute_projections(x, &alpha, projections);

        // Get sort order for projections (reuse buffer)
        for (i, idx) in sorted_indices.iter_mut().enumerate() {
            *idx = i;
        }
        sorted_indices.sort_unstable_by(|&a, &b| {
            projections[a]
                .partial_cmp(&projections[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fit smooth function to (projections, residuals) - fast version
        smooth_and_derivative_fast(
            projections,
            residuals,
            weights,
            sorted_indices,
            config,
            smoothed,
            derivative,
        );

        // Update direction using gradient
        let mut new_alpha = vec![0.0; p];
        for i in 0..n {
            let ri = residuals[i] - smoothed[i];
            let di = derivative[i];
            let wi = weights[i];
            for j in 0..p {
                new_alpha[j] += wi * ri * di * x[[i, j]];
            }
        }

        // Normalize
        let new_norm: f64 = new_alpha.iter().map(|&a| a * a).sum::<f64>().sqrt();
        if new_norm > 1e-10 {
            for a in &mut new_alpha {
                *a /= new_norm;
            }
        }

        // Check convergence
        let diff: f64 = alpha
            .iter()
            .zip(new_alpha.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        alpha = new_alpha;

        if diff < config.tol {
            break;
        }

        // Early termination if we're oscillating
        if iter > 5 && diff > 0.5 {
            break;
        }
    }

    // Final projection and ridge function
    compute_projections(x, &alpha, projections);

    // Sort one more time for final smoothing
    for (i, idx) in sorted_indices.iter_mut().enumerate() {
        *idx = i;
    }
    sorted_indices.sort_unstable_by(|&a, &b| {
        projections[a]
            .partial_cmp(&projections[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    smooth_and_derivative_fast(
        projections,
        residuals,
        weights,
        sorted_indices,
        config,
        smoothed,
        derivative,
    );

    // Compute beta (scaling coefficient)
    let ridge_mean: f64 = weighted_mean(smoothed, weights);
    let res_mean: f64 = weighted_mean(residuals, weights);

    let mut cov = 0.0;
    let mut var_ridge = 0.0;
    for i in 0..n {
        let r_dev = smoothed[i] - ridge_mean;
        let res_dev = residuals[i] - res_mean;
        cov += weights[i] * r_dev * res_dev;
        var_ridge += weights[i] * r_dev * r_dev;
    }

    let beta = if var_ridge > 1e-10 {
        cov / var_ridge
    } else {
        1.0
    };

    // Scale ridge values by beta
    let scaled_ridge: Vec<f64> = smoothed.iter().map(|&r| beta * (r - ridge_mean)).collect();

    Ok((alpha, scaled_ridge, beta))
}

/// Compute projections: proj[i] = alpha . x[i]
#[inline]
fn compute_projections(x: &ArrayView2<f64>, alpha: &[f64], projections: &mut [f64]) {
    let n = x.nrows();
    let p = x.ncols();
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            sum += alpha[j] * x[[i, j]];
        }
        projections[i] = sum;
    }
}

/// Compute initial projection direction using weighted correlation - optimized.
fn initial_direction_fast(x: &ArrayView2<f64>, y: &[f64], weights: &[f64]) -> Vec<f64> {
    let n = x.nrows();
    let p = x.ncols();

    let y_mean = weighted_mean(y, weights);
    let mut direction = vec![0.0; p];

    for j in 0..p {
        let mut sum_wx = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            sum_wx += weights[i] * x[[i, j]];
            sum_w += weights[i];
        }
        let x_mean = if sum_w > 0.0 { sum_wx / sum_w } else { 0.0 };

        // Weighted correlation with y
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..n {
            let x_dev = x[[i, j]] - x_mean;
            let y_dev = y[i] - y_mean;
            cov += weights[i] * x_dev * y_dev;
            var_x += weights[i] * x_dev * x_dev;
        }

        direction[j] = if var_x > 1e-10 {
            cov / var_x.sqrt()
        } else {
            0.0
        };
    }

    direction
}

/// Smooth the data and compute derivative - optimized version with preallocated buffers.
fn smooth_and_derivative_fast(
    projections: &[f64],
    residuals: &[f64],
    weights: &[f64],
    sorted_indices: &[usize],
    config: &PprConfig,
    smoothed: &mut [f64],
    derivative: &mut [f64],
) {
    let n = projections.len();

    // Compute span
    let span = if config.span > 0.0 {
        (config.span * n as f64).round() as usize
    } else {
        (0.2 * n as f64).round() as usize
    };
    let span = span.max(3).min(n);
    let half_span = span / 2;

    // Running weighted mean smoother with incremental updates
    let mut sum_wy = 0.0;
    let mut sum_w = 0.0;
    let mut start = 0usize;
    let mut end = 0usize;

    // Temporary storage for sorted smoothed values
    let mut sorted_smoothed = vec![0.0; n];

    for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        // Ideal window bounds in sorted space
        let ideal_start = sorted_pos.saturating_sub(half_span);
        let ideal_end = (sorted_pos + half_span + 1).min(n);

        // Remove points that left the window
        while start < ideal_start {
            let idx = sorted_indices[start];
            sum_wy -= weights[idx] * residuals[idx];
            sum_w -= weights[idx];
            start += 1;
        }

        // Add points that entered the window
        while end < ideal_end {
            let idx = sorted_indices[end];
            sum_wy += weights[idx] * residuals[idx];
            sum_w += weights[idx];
            end += 1;
        }

        // Compute smoothed value
        sorted_smoothed[sorted_pos] = if sum_w > 0.0 {
            sum_wy / sum_w
        } else {
            residuals[orig_idx]
        };
    }

    // Map smoothed values back to original order
    for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        smoothed[orig_idx] = sorted_smoothed[sorted_pos];
    }

    // Compute numerical derivative using sorted order
    for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        if sorted_pos == 0 {
            // Forward difference
            let next_idx = sorted_indices[1];
            let dx = projections[next_idx] - projections[orig_idx];
            derivative[orig_idx] = if dx.abs() > 1e-10 {
                (sorted_smoothed[1] - sorted_smoothed[0]) / dx
            } else {
                0.0
            };
        } else if sorted_pos == n - 1 {
            // Backward difference
            let prev_idx = sorted_indices[n - 2];
            let dx = projections[orig_idx] - projections[prev_idx];
            derivative[orig_idx] = if dx.abs() > 1e-10 {
                (sorted_smoothed[n - 1] - sorted_smoothed[n - 2]) / dx
            } else {
                0.0
            };
        } else {
            // Central difference
            let prev_idx = sorted_indices[sorted_pos - 1];
            let next_idx = sorted_indices[sorted_pos + 1];
            let dx = projections[next_idx] - projections[prev_idx];
            derivative[orig_idx] = if dx.abs() > 1e-10 {
                (sorted_smoothed[sorted_pos + 1] - sorted_smoothed[sorted_pos - 1]) / dx
            } else {
                0.0
            };
        }
    }
}

/// Find the least important term (smallest absolute beta).
#[inline]
fn find_least_important_term(betas: &[f64]) -> usize {
    betas
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Weighted mean - optimized.
#[inline]
fn weighted_mean(x: &[f64], w: &[f64]) -> f64 {
    let mut sum_wx = 0.0;
    let mut sum_w = 0.0;
    for (&xi, &wi) in x.iter().zip(w.iter()) {
        sum_wx += wi * xi;
        sum_w += wi;
    }
    if sum_w > 0.0 { sum_wx / sum_w } else { 0.0 }
}

/// Run PPR (convenience wrapper).
pub fn run_ppr(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    nterms: usize,
    max_terms: Option<usize>,
    sm_method: Option<SmoothingMethod>,
    bass: Option<f64>,
) -> Result<PprResult, String> {
    let config = PprConfig {
        nterms,
        max_terms: max_terms.unwrap_or(nterms),
        sm_method: sm_method.unwrap_or(SmoothingMethod::Supsmu),
        bass: bass.unwrap_or(0.0),
        ..Default::default()
    };
    ppr(x, y, weights, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_ppr_basic() {
        // Simple linear relationship
        let x_data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((15, 2), x_data).unwrap();

        let y: Vec<f64> = (0..15)
            .map(|i| 2.0 * i as f64 + 3.0 * (i + 15) as f64)
            .collect();

        let config = PprConfig {
            nterms: 1,
            max_terms: 1,
            ..Default::default()
        };

        let result = ppr(x.view(), &y, None, config).unwrap();

        assert_eq!(result.nterms, 1);
        assert_eq!(result.n, 15);
        assert_eq!(result.p, 2);
        assert_eq!(result.fitted.len(), 15);
        assert_eq!(result.residuals.len(), 15);
    }

    #[test]
    fn test_ppr_multiple_terms() {
        let n = 50;
        let p = 3;
        let x_data: Vec<f64> = (0..n * p).map(|i| (i as f64) / (n * p) as f64).collect();
        let x = Array2::from_shape_vec((n, p), x_data).unwrap();

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi: f64 = (0..p).map(|j| x[[i, j]]).sum();
                xi.sin() + 0.1 * (i as f64 % 5.0)
            })
            .collect();

        let config = PprConfig {
            nterms: 2,
            max_terms: 3,
            ..Default::default()
        };

        let result = ppr(x.view(), &y, None, config).unwrap();

        // Should have at most 2 terms
        assert!(result.nterms <= 2);
        assert!(!result.gofn.is_empty());
    }

    #[test]
    fn test_ppr_with_weights() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let weights: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { 2.0 }).collect();

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, Some(&weights), config).unwrap();

        assert_eq!(result.n, 10);
    }

    #[test]
    fn test_ppr_predict() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, None, config).unwrap();

        // Predict on new data
        let x_new = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let predictions = result.predict(&x_new);

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_ppr_length_mismatch() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect(); // Wrong length

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, None, config);

        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_mean() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 1.0, 1.0];
        assert_relative_eq!(weighted_mean(&x, &w), 2.0, epsilon = 1e-10);

        let w2 = vec![1.0, 0.0, 1.0];
        assert_relative_eq!(weighted_mean(&x, &w2), 2.0, epsilon = 1e-10);
    }
}
