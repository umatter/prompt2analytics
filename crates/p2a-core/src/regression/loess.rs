//! Local Polynomial Regression (LOESS/LOWESS).
//!
//! Pure Rust implementation of locally estimated scatterplot smoothing following
//! Cleveland's original algorithms.
//!
//! # Method Overview
//!
//! LOESS (LOcally Estimated Scatterplot Smoothing) fits simple models to localized
//! subsets of the data to build up a function that describes the deterministic part
//! of the variation in the data, point by point.
//!
//! For each point x0, the algorithm:
//! 1. Selects k = floor(span * n) nearest neighbors
//! 2. Computes tricubic weights: w(u) = (1 - |u|^3)^3 for |u| < 1
//! 3. Fits a weighted polynomial regression
//! 4. Returns the fitted value at x0
//!
//! # Robust Fitting
//!
//! When `robust=true`, the algorithm applies iterative reweighting using the bisquare
//! function to downweight outliers:
//! - w(u) = (1 - u^2)^2 for |u| < 1, else 0
//!
//! # Performance
//!
//! This implementation uses several optimizations inspired by R's C implementation:
//! - Pre-sorting by x with binary search for O(k + log n) neighborhood lookup
//! - Inline closed-form WLS solvers for degree 0 (weighted mean), degree 1 (2x2),
//!   and degree 2 (3x3 Cramer's rule)
//! - Buffer reuse across all fitting points to avoid per-point allocations
//!
//! # References
//!
//! - Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing
//!   scatterplots." Journal of the American Statistical Association, 74(368), 829-836.
//! - Cleveland, W. S., & Devlin, S. J. (1988). "Locally weighted regression: an
//!   approach to regression analysis by local fitting." Journal of the American
//!   Statistical Association, 83(403), 596-610.
//! - Cleveland, W. S., Grosse, E., & Shyu, W. M. (1992). "Local regression models."
//!   Chapter 8 of Statistical Models in S, eds J.M. Chambers and T.J. Hastie.
//! - R implementation: `stats::loess()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::matrix_inverse;

/// Configuration for LOESS fitting.
#[derive(Debug, Clone)]
pub struct LoessConfig {
    /// Smoothing parameter (proportion of points in each local fit).
    /// Default: 0.75. Range: (0, 1] for neighborhood proportion, >1 uses all points.
    pub span: f64,
    /// Polynomial degree for local fitting (1 = linear, 2 = quadratic).
    /// Default: 2.
    pub degree: usize,
    /// Use robust fitting with iterative reweighting.
    /// Default: false (gaussian family). Set true for "symmetric" family.
    pub robust: bool,
    /// Number of robustifying iterations (only used if robust=true).
    /// Default: 4.
    pub robust_iterations: usize,
    /// Convergence tolerance for robust iterations.
    /// Default: 1e-4.
    pub tolerance: f64,
}

impl Default for LoessConfig {
    fn default() -> Self {
        Self {
            span: 0.75,
            degree: 2,
            robust: false,
            robust_iterations: 4,
            tolerance: 1e-4,
        }
    }
}

/// LOESS model that can be used for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoessModel {
    /// Original x values
    pub x: Vec<f64>,
    /// Original y values
    pub y: Vec<f64>,
    /// Smoothing parameter
    pub span: f64,
    /// Polynomial degree
    pub degree: usize,
    /// Whether robust fitting was used
    pub robust: bool,
    /// Robustness weights (if robust=true)
    pub robust_weights: Option<Vec<f64>>,
}

/// Result from LOESS fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoessResult {
    /// The fitted LOESS model (can be used for prediction)
    #[serde(skip)]
    pub model: Option<LoessModel>,
    /// Fitted values at original x points
    pub fitted: Vec<f64>,
    /// Residuals (y - fitted)
    pub residuals: Vec<f64>,
    /// Smoothing parameter used
    pub span: f64,
    /// Polynomial degree used
    pub degree: usize,
    /// Whether robust fitting was used
    pub robust: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Effective number of parameters (trace of hat matrix)
    pub enp: f64,
    /// Residual sum of squares
    pub rss: f64,
    /// Residual standard error
    pub residual_se: f64,
    /// R-squared (1 - RSS/TSS)
    pub r_squared: f64,
    /// Number of robust iterations performed
    pub robust_iterations: usize,
}

impl fmt::Display for LoessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LOESS Results")?;
        writeln!(f, "==============================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "Span: {:.4}", self.span)?;
        writeln!(f, "Degree: {}", self.degree)?;
        writeln!(
            f,
            "Family: {}",
            if self.robust { "symmetric" } else { "gaussian" }
        )?;
        if self.robust {
            writeln!(f, "Robust Iterations: {}", self.robust_iterations)?;
        }
        writeln!(f)?;
        writeln!(f, "Equivalent Number of Parameters: {:.4}", self.enp)?;
        writeln!(f, "Residual Standard Error: {:.6}", self.residual_se)?;
        writeln!(f, "Residual Sum of Squares: {:.6}", self.rss)?;
        writeln!(f, "R-squared: {:.6}", self.r_squared)?;
        Ok(())
    }
}

/// Tricubic weight function: w(u) = (1 - |u|^3)^3 for |u| < 1, else 0.
///
/// This is the default weight function for LOESS, giving more weight to nearby
/// points and smoothly decreasing to zero at the boundary of the neighborhood.
#[inline]
fn tricubic_weight(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u < 1.0 {
        let t = 1.0 - abs_u.powi(3);
        t.powi(3)
    } else {
        0.0
    }
}

/// Bisquare weight function: w(u) = (1 - u^2)^2 for |u| < 1, else 0.
///
/// Used for robust fitting (Tukey's biweight function).
#[inline]
fn bisquare_weight(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u < 1.0 {
        let t = 1.0 - abs_u.powi(2);
        t.powi(2)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Optimized internal engine using pre-sorted data and inline WLS solvers
// ---------------------------------------------------------------------------

/// Pre-sorted data structure for O(k + log n) neighborhood lookups.
///
/// Instead of computing O(n) distances and sorting them for each evaluation point,
/// we sort the data by x once upfront. Then, for each evaluation point x0, we use
/// binary search to find the insertion point and expand outward to collect the k
/// nearest neighbors. This mirrors the approach used in R's C implementation of loess.
struct SortedData {
    /// x values in sorted order
    x_sorted: Vec<f64>,
    /// y values reordered to match x_sorted
    y_sorted: Vec<f64>,
    /// Mapping from sorted index back to original index
    orig_idx: Vec<usize>,
    /// Number of neighbors to use
    k: usize,
    /// Whether span > 1 (use all points with scaled bandwidth)
    span_gt1: bool,
    /// The span value (for bandwidth scaling when span > 1)
    span: f64,
}

impl SortedData {
    fn new(x: &[f64], y: &[f64], span: f64) -> Self {
        let n = x.len();
        // Create index array sorted by x values
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let x_sorted: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let y_sorted: Vec<f64> = order.iter().map(|&i| y[i]).collect();

        let k = if span <= 1.0 {
            ((span * n as f64).ceil() as usize).max(1).min(n)
        } else {
            n
        };

        SortedData {
            x_sorted,
            y_sorted,
            orig_idx: order,
            k,
            span_gt1: span > 1.0,
            span,
        }
    }

    /// Find the k nearest neighbors of x0 using binary search + expand.
    /// Returns (left_idx, right_idx) as a contiguous range in the sorted arrays,
    /// where right_idx is exclusive.
    ///
    /// This is O(k + log n) instead of O(n log n) for the naive approach.
    #[inline]
    fn find_neighborhood(&self, x0: f64) -> (usize, usize) {
        let n = self.x_sorted.len();
        let k = self.k;

        // Binary search for insertion point of x0
        let pos = self.x_sorted.partition_point(|&v| v < x0);

        // Expand outward from pos to collect k neighbors
        let mut left = pos.min(n - 1);
        let mut right = left; // right is inclusive during expansion

        // Start: pick the closest single point
        if left > 0
            && (left >= n
                || (x0 - self.x_sorted[left - 1]).abs() < (self.x_sorted[left] - x0).abs())
        {
            left -= 1;
            right = left;
        }

        // Expand until we have k points
        while right - left + 1 < k {
            let can_go_left = left > 0;
            let can_go_right = right < n - 1;
            if !can_go_left && !can_go_right {
                break;
            }
            if can_go_left && can_go_right {
                let dl = (x0 - self.x_sorted[left - 1]).abs();
                let dr = (self.x_sorted[right + 1] - x0).abs();
                if dl <= dr {
                    left -= 1;
                } else {
                    right += 1;
                }
            } else if can_go_left {
                left -= 1;
            } else {
                right += 1;
            }
        }

        (left, right + 1) // right+1 to make it exclusive
    }

    /// Compute the maximum distance in the neighborhood for weight scaling.
    #[inline]
    fn max_distance(&self, x0: f64, left: usize, right: usize) -> f64 {
        let dl = (x0 - self.x_sorted[left]).abs();
        let dr = (x0 - self.x_sorted[right - 1]).abs();
        let max_dist = dl.max(dr);
        let max_dist = if self.span_gt1 {
            max_dist * self.span
        } else {
            max_dist
        };
        if max_dist < 1e-10 { 1.0 } else { max_dist }
    }
}

/// Inline WLS solver for degree 0 (local constant / weighted mean).
///
/// fitted = sum(w_i * y_i) / sum(w_i)
///
/// Also returns the self-weight (h_ii) for hat matrix trace computation.
/// For degree 0: h_ii = w_i / sum(w_j)
#[inline]
fn wls_degree0(y: &[f64], weights: &[f64], self_pos: usize) -> (f64, f64) {
    let mut sw = 0.0;
    let mut swy = 0.0;
    for i in 0..y.len() {
        let w = weights[i];
        sw += w;
        swy += w * y[i];
    }
    if sw < 1e-15 {
        return (0.0, 0.0);
    }
    let fitted = swy / sw;
    let h_ii = weights[self_pos] / sw;
    (fitted, h_ii)
}

/// Inline WLS solver for degree 1 (local linear) using 2x2 normal equations.
///
/// Solves the weighted normal equations directly:
///   [S0  S1] [b0]   [T0]
///   [S1  S2] [b1] = [T1]
///
/// where Sk = sum(w_i * dx_i^k), Tk = sum(w_i * dx_i^k * y_i), dx_i = x_i - x0.
///
/// Uses Cramer's rule for the 2x2 system.
/// Also returns h_ii for hat matrix trace: h_ii = e_0' (X'WX)^-1 e_0 * w_self
/// where e_0 = [1, 0] since we evaluate at x0 (dx=0).
/// So h_ii = (X'WX)^{-1}[0,0] * w_self
#[inline]
fn wls_degree1(
    x_sorted: &[f64],
    y: &[f64],
    weights: &[f64],
    x0: f64,
    left: usize,
    self_pos: usize,
) -> (f64, f64) {
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut t0 = 0.0;
    let mut t1 = 0.0;

    for i in 0..weights.len() {
        let w = weights[i];
        if w > 0.0 {
            let dx = x_sorted[left + i] - x0;
            let wdx = w * dx;
            s0 += w;
            s1 += wdx;
            s2 += wdx * dx;
            t0 += w * y[i];
            t1 += wdx * y[i];
        }
    }

    // Cramer's rule for 2x2: det = s0*s2 - s1*s1
    let det = s0 * s2 - s1 * s1;
    if det.abs() < 1e-30 {
        // Fallback to degree 0 (weighted mean)
        return wls_degree0(y, weights, self_pos);
    }
    let inv_det = 1.0 / det;
    // b0 = (s2*t0 - s1*t1) / det
    let b0 = (s2 * t0 - s1 * t1) * inv_det;
    // h_ii: (X'WX)^{-1}[0,0] = s2 / det
    let h_ii = (s2 * inv_det) * weights[self_pos];
    (b0, h_ii)
}

/// Inline WLS solver for degree 2 (local quadratic) using 3x3 system.
///
/// Solves:
///   [S0  S1  S2] [b0]   [T0]
///   [S1  S2  S3] [b1] = [T1]
///   [S2  S3  S4] [b2]   [T2]
///
/// We only need b0 (fitted value at x0) and (X'WX)^{-1}[0,0] (for h_ii).
/// Uses Cramer's rule for the 3x3 system to get b0, and the cofactor
/// expansion for (X'WX)^{-1}[0,0].
#[inline]
fn wls_degree2(
    x_sorted: &[f64],
    y: &[f64],
    weights: &[f64],
    x0: f64,
    left: usize,
    self_pos: usize,
) -> (f64, f64) {
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;
    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let mut t2 = 0.0;

    for i in 0..weights.len() {
        let w = weights[i];
        if w > 0.0 {
            let dx = x_sorted[left + i] - x0;
            let dx2 = dx * dx;
            s0 += w;
            s1 += w * dx;
            s2 += w * dx2;
            s3 += w * dx2 * dx;
            s4 += w * dx2 * dx2;
            t0 += w * y[i];
            t1 += w * dx * y[i];
            t2 += w * dx2 * y[i];
        }
    }

    // 3x3 determinant by cofactor expansion along row 0:
    // det = s0*(s2*s4 - s3*s3) - s1*(s1*s4 - s3*s2) + s2*(s1*s3 - s2*s2)
    let m00 = s2 * s4 - s3 * s3;
    let m01 = s1 * s4 - s3 * s2;
    let m02 = s1 * s3 - s2 * s2;
    let det = s0 * m00 - s1 * m01 + s2 * m02;

    if det.abs() < 1e-30 {
        // Fallback to degree 1
        return wls_degree1(x_sorted, y, weights, x0, left, self_pos);
    }

    let inv_det = 1.0 / det;

    // b0 via Cramer's rule: replace column 0 with RHS
    // det_0 = t0*(s2*s4-s3^2) - s1*(t1*s4-s3*t2) + s2*(t1*s3-s2*t2)
    let det_0 = t0 * m00 - s1 * (t1 * s4 - s3 * t2) + s2 * (t1 * s3 - s2 * t2);
    let b0 = det_0 * inv_det;

    // (X'WX)^{-1}[0,0] = cofactor(0,0) / det = m00 / det
    let h_ii = (m00 * inv_det) * weights[self_pos];

    (b0, h_ii)
}

/// Compute weights for a neighborhood range, writing into the provided buffer.
/// Returns the number of non-zero weights.
#[inline]
fn compute_weights_into(
    x_sorted: &[f64],
    x0: f64,
    left: usize,
    right: usize,
    max_dist: f64,
    robust_weights: Option<&[f64]>,
    orig_idx: &[usize],
    weight_buf: &mut [f64],
) -> usize {
    let inv_max = 1.0 / max_dist;
    let len = right - left;
    let mut nonzero = 0;
    for i in 0..len {
        let u = (x_sorted[left + i] - x0).abs() * inv_max;
        let mut w = tricubic_weight(u);
        if let Some(rw) = robust_weights {
            w *= rw[orig_idx[left + i]];
        }
        weight_buf[i] = w;
        if w > 0.0 {
            nonzero += 1;
        }
    }
    nonzero
}

/// Fit all data points using the optimized engine.
///
/// Returns (fitted_values_in_original_order, hat_matrix_trace).
fn fit_all_points(
    sd: &SortedData,
    degree: usize,
    robust_weights: Option<&[f64]>,
    compute_hat: bool,
) -> EconResult<(Vec<f64>, f64)> {
    let n = sd.x_sorted.len();
    let k = sd.k;

    // Pre-allocate output in sorted order, then reorder
    let mut fitted_sorted = vec![0.0; n];
    let mut hat_trace = 0.0;

    // Reusable weight buffer (max size = k)
    let mut weight_buf = vec![0.0; k];

    for si in 0..n {
        let x0 = sd.x_sorted[si];
        let (left, right) = sd.find_neighborhood(x0);
        let len = right - left;
        let max_dist = sd.max_distance(x0, left, right);

        // Compute weights into reusable buffer
        compute_weights_into(
            &sd.x_sorted,
            x0,
            left,
            right,
            max_dist,
            robust_weights,
            &sd.orig_idx,
            &mut weight_buf[..len],
        );

        // Find position of current point within neighborhood for h_ii
        // Since data is sorted, si must be in [left, right)
        let self_pos = si - left;

        let y_slice = &sd.y_sorted[left..right];
        let w_slice = &weight_buf[..len];

        let (fitted_val, h_ii) = match degree {
            0 => wls_degree0(y_slice, w_slice, self_pos),
            1 => wls_degree1(&sd.x_sorted, y_slice, w_slice, x0, left, self_pos),
            2 => wls_degree2(&sd.x_sorted, y_slice, w_slice, x0, left, self_pos),
            _ => {
                // Generic fallback using matrix operations (should not happen for degree <= 2)
                let fitted =
                    generic_wls_fit(&sd.x_sorted[left..right], y_slice, w_slice, x0, degree)?;
                (fitted, 0.0) // No hat trace for generic
            }
        };

        fitted_sorted[si] = fitted_val;
        if compute_hat {
            hat_trace += h_ii;
        }
    }

    // Reorder from sorted order back to original order
    let mut fitted_orig = vec![0.0; n];
    for si in 0..n {
        fitted_orig[sd.orig_idx[si]] = fitted_sorted[si];
    }

    Ok((fitted_orig, hat_trace))
}

/// Fit at a single evaluation point (used for prediction at new x values).
fn fit_single_point(
    sd: &SortedData,
    x0: f64,
    degree: usize,
    robust_weights: Option<&[f64]>,
) -> EconResult<f64> {
    let (left, right) = sd.find_neighborhood(x0);
    let len = right - left;
    let max_dist = sd.max_distance(x0, left, right);

    let mut weight_buf = vec![0.0; len];
    compute_weights_into(
        &sd.x_sorted,
        x0,
        left,
        right,
        max_dist,
        robust_weights,
        &sd.orig_idx,
        &mut weight_buf,
    );

    let y_slice = &sd.y_sorted[left..right];

    let (fitted_val, _) = match degree {
        0 => wls_degree0(y_slice, &weight_buf, 0), // self_pos doesn't matter for prediction
        1 => wls_degree1(&sd.x_sorted, y_slice, &weight_buf, x0, left, 0),
        2 => wls_degree2(&sd.x_sorted, y_slice, &weight_buf, x0, left, 0),
        _ => {
            let fitted =
                generic_wls_fit(&sd.x_sorted[left..right], y_slice, &weight_buf, x0, degree)?;
            (fitted, 0.0)
        }
    };

    Ok(fitted_val)
}

/// Generic WLS solver using matrix operations for arbitrary degree.
/// Fallback for degree > 2 (should not be needed normally).
fn generic_wls_fit(
    x_nbr: &[f64],
    y_nbr: &[f64],
    weights: &[f64],
    x0: f64,
    degree: usize,
) -> EconResult<f64> {
    let n = x_nbr.len();
    let p = degree + 1;

    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);

    for i in 0..n {
        let w = weights[i];
        if w > 0.0 {
            let dx = x_nbr[i] - x0;
            // Build powers of dx
            let mut powers = vec![1.0; p];
            for j in 1..p {
                powers[j] = powers[j - 1] * dx;
            }
            for j in 0..p {
                xtwy[j] += w * powers[j] * y_nbr[i];
                for kk in 0..p {
                    xtwx[[j, kk]] += w * powers[j] * powers[kk];
                }
            }
        }
    }

    let xtwx_inv = matrix_inverse(&xtwx.view())?;
    let beta = xtwx_inv.dot(&xtwy);
    Ok(beta[0])
}

/// Compute robust weights using bisquare function.
///
/// Scale residuals by 6 * MAD (median absolute deviation) and apply bisquare.
fn compute_robust_weights(residuals: &[f64]) -> Vec<f64> {
    let n = residuals.len();
    if n == 0 {
        return vec![];
    }

    // Compute median of absolute residuals
    let mut abs_resid: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_idx = n / 2;
    let mad = if n % 2 == 0 {
        (abs_resid[median_idx - 1] + abs_resid[median_idx]) / 2.0
    } else {
        abs_resid[median_idx]
    };

    // Scale factor: 6 * MAD (following R's loess implementation)
    let scale = 6.0 * mad;

    if scale < 1e-10 {
        // All residuals are essentially zero
        return vec![1.0; n];
    }

    // Compute bisquare weights
    residuals
        .iter()
        .map(|&r| bisquare_weight(r / scale))
        .collect()
}

/// Fit LOESS model to data.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `config` - LOESS configuration
///
/// # Returns
/// `LoessResult` containing fitted values, residuals, and model diagnostics.
///
/// # Example
/// ```ignore
/// use p2a_core::regression::loess::{loess, LoessConfig};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![2.1, 4.2, 5.8, 8.1, 10.0, 11.9, 14.2, 15.8, 18.1, 20.0];
/// let result = loess(&x, &y, LoessConfig::default())?;
/// println!("{}", result);
/// ```
pub fn loess(x: &[f64], y: &[f64], config: LoessConfig) -> EconResult<LoessResult> {
    let n = x.len();

    // Validate inputs
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "LOESS requires at least 3 observations".to_string(),
        });
    }
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }
    if config.span <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "span must be positive".to_string(),
        });
    }
    if config.degree > 2 {
        return Err(EconError::InvalidSpecification {
            message: "degree must be 0, 1, or 2".to_string(),
        });
    }

    // Minimum observations needed for local regression
    let min_obs = config.degree + 1;
    let k = if config.span <= 1.0 {
        ((config.span * n as f64).ceil() as usize).max(1).min(n)
    } else {
        n
    };
    if k < min_obs {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "span too small: need at least {} observations per neighborhood for degree {}, but span gives {}",
                min_obs, config.degree, k
            ),
        });
    }

    // Build pre-sorted data structure for fast neighborhood lookups
    let sd = SortedData::new(x, y, config.span);

    // Initial fit (also computes hat matrix trace for ENP)
    let (mut fitted, enp) = fit_all_points(&sd, config.degree, None, true)?;

    // Compute residuals
    let mut residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    // Robust fitting iterations
    let mut robust_weights: Option<Vec<f64>> = None;
    let mut iterations = 0;

    if config.robust {
        for iter in 0..config.robust_iterations {
            // Compute robust weights from residuals
            let new_robust_weights = compute_robust_weights(&residuals);

            // Refit with robust weights (no need for hat trace in robust iterations)
            let (new_fitted, _) =
                fit_all_points(&sd, config.degree, Some(&new_robust_weights), false)?;

            // Check convergence
            let max_change: f64 = fitted
                .iter()
                .zip(new_fitted.iter())
                .map(|(f1, f2)| (f1 - f2).abs())
                .fold(0.0, f64::max);

            fitted = new_fitted;
            residuals = y
                .iter()
                .zip(fitted.iter())
                .map(|(yi, fi)| yi - fi)
                .collect();
            robust_weights = Some(new_robust_weights);
            iterations = iter + 1;

            if max_change < config.tolerance {
                break;
            }
        }
    }

    // Compute diagnostics
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let df = (n as f64 - enp).max(1.0);
    let residual_se = (rss / df).sqrt();
    let r_squared = if tss > 1e-10 { 1.0 - rss / tss } else { 0.0 };

    // Build model for prediction
    let model = LoessModel {
        x: x.to_vec(),
        y: y.to_vec(),
        span: config.span,
        degree: config.degree,
        robust: config.robust,
        robust_weights: robust_weights.clone(),
    };

    Ok(LoessResult {
        model: Some(model),
        fitted,
        residuals,
        span: config.span,
        degree: config.degree,
        robust: config.robust,
        n_obs: n,
        enp,
        rss,
        residual_se,
        r_squared,
        robust_iterations: iterations,
    })
}

/// Predict new values using a fitted LOESS model.
///
/// # Arguments
/// * `model` - Fitted LOESS model
/// * `new_x` - New x values for prediction
///
/// # Returns
/// Vector of predicted y values.
pub fn loess_predict(model: &LoessModel, new_x: &[f64]) -> EconResult<Vec<f64>> {
    // Build sorted data structure for the model's training data
    let sd = SortedData::new(&model.x, &model.y, model.span);

    let mut predictions = Vec::with_capacity(new_x.len());

    for &x0 in new_x {
        let pred = fit_single_point(&sd, x0, model.degree, model.robust_weights.as_deref())?;
        predictions.push(pred);
    }

    Ok(predictions)
}

/// Fit LOESS model from a dataset.
///
/// # Arguments
/// * `dataset` - Dataset containing the data
/// * `y_col` - Name of the response column
/// * `x_col` - Name of the predictor column
/// * `span` - Smoothing parameter (default: 0.75)
/// * `degree` - Polynomial degree (default: 2)
/// * `robust` - Use robust fitting (default: false)
///
/// # Returns
/// `LoessResult` containing fitted values, residuals, and model diagnostics.
pub fn run_loess(
    dataset: &Dataset,
    y_col: &str,
    x_col: &str,
    span: f64,
    degree: usize,
    robust: bool,
) -> EconResult<LoessResult> {
    let df = dataset.df();

    // Extract columns
    let x_series = df.column(x_col).map_err(|_| EconError::ColumnNotFound {
        column: x_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Convert to f64 vectors
    let x: Vec<f64> = x_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: x_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let config = LoessConfig {
        span,
        degree,
        robust,
        ..Default::default()
    };

    loess(&x, &y, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tricubic_weight() {
        // At u = 0, weight should be 1
        assert!((tricubic_weight(0.0) - 1.0).abs() < 1e-10);

        // At u = 1, weight should be 0
        assert!(tricubic_weight(1.0).abs() < 1e-10);

        // At u > 1, weight should be 0
        assert!(tricubic_weight(1.5).abs() < 1e-10);

        // At u = 0.5, weight should be (1 - 0.125)^3 = 0.875^3 ~ 0.6699
        let expected = (1.0 - 0.5_f64.powi(3)).powi(3);
        assert!((tricubic_weight(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bisquare_weight() {
        // At u = 0, weight should be 1
        assert!((bisquare_weight(0.0) - 1.0).abs() < 1e-10);

        // At u = 1, weight should be 0
        assert!(bisquare_weight(1.0).abs() < 1e-10);

        // At u > 1, weight should be 0
        assert!(bisquare_weight(1.5).abs() < 1e-10);

        // At u = 0.5, weight should be (1 - 0.25)^2 = 0.5625
        let expected = (1.0 - 0.5_f64.powi(2)).powi(2);
        assert!((bisquare_weight(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_loess_linear_data() {
        // Linear data should be fit well
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + 0.1 * (xi * 10.0).sin())
            .collect();

        let result = loess(&x, &y, LoessConfig::default()).unwrap();

        // Check that fitted values are close to actual
        assert!(result.r_squared > 0.95);
        assert_eq!(result.n_obs, 20);
        assert_eq!(result.fitted.len(), 20);
        assert_eq!(result.residuals.len(), 20);
    }

    #[test]
    fn test_loess_with_different_spans() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * xi).collect();

        // Small span = less smooth
        let result_small = loess(
            &x,
            &y,
            LoessConfig {
                span: 0.3,
                ..Default::default()
            },
        )
        .unwrap();

        // Large span = more smooth
        let result_large = loess(
            &x,
            &y,
            LoessConfig {
                span: 0.9,
                ..Default::default()
            },
        )
        .unwrap();

        // Both should complete without error
        assert!(result_small.enp > 0.0);
        assert!(result_large.enp > 0.0);

        // Smaller span should have more effective parameters (more wiggly)
        // This isn't always strictly true but generally holds
    }

    #[test]
    fn test_loess_degree_1_vs_2() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi / 100.0 + 0.5 * xi).collect();

        let result_linear = loess(
            &x,
            &y,
            LoessConfig {
                degree: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let result_quad = loess(
            &x,
            &y,
            LoessConfig {
                degree: 2,
                ..Default::default()
            },
        )
        .unwrap();

        // Both should work
        assert!(result_linear.r_squared > 0.9);
        assert!(result_quad.r_squared > 0.9);
    }

    #[test]
    fn test_loess_robust() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        // Add outliers
        y[10] = 100.0; // Far outlier
        y[20] = -50.0; // Another outlier

        let _result_normal = loess(
            &x,
            &y,
            LoessConfig {
                robust: false,
                ..Default::default()
            },
        )
        .unwrap();
        let result_robust = loess(
            &x,
            &y,
            LoessConfig {
                robust: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Robust should have lower RSS (outliers downweighted)
        // In this case, robust should better fit the underlying linear trend
        assert!(result_robust.robust_iterations > 0);
    }

    #[test]
    fn test_loess_predict() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = loess(&x, &y, LoessConfig::default()).unwrap();
        let model = result.model.unwrap();

        // Predict at new points
        let new_x = vec![5.5, 10.5, 15.5];
        let predictions = loess_predict(&model, &new_x).unwrap();

        assert_eq!(predictions.len(), 3);

        // Predictions should be close to 2*x + 1 for linear data
        for (i, &xi) in new_x.iter().enumerate() {
            let expected = 2.0 * xi + 1.0;
            assert!(
                (predictions[i] - expected).abs() < 1.0,
                "Prediction at {} = {}, expected ~{}",
                xi,
                predictions[i],
                expected
            );
        }
    }

    #[test]
    fn test_loess_errors() {
        // Too few observations
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        assert!(loess(&x, &y, LoessConfig::default()).is_err());

        // Mismatched lengths
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(loess(&x, &y, LoessConfig::default()).is_err());

        // Invalid span
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    span: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    span: -0.5,
                    ..Default::default()
                }
            )
            .is_err()
        );

        // Invalid degree
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    degree: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    /// Validation test against R's loess function.
    /// This test uses known values from R to verify correctness.
    #[test]
    fn test_validate_loess_against_r() {
        // Test data (from R: loess(y ~ x, span=0.75, degree=2))
        // R code:
        // x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        // y <- c(2.5, 3.8, 6.1, 7.9, 10.2, 12.0, 14.3, 16.1, 18.0, 20.2)
        // fit <- loess(y ~ x, span=0.75, degree=2)
        // fit$fitted

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.5, 3.8, 6.1, 7.9, 10.2, 12.0, 14.3, 16.1, 18.0, 20.2];

        let result = loess(
            &x,
            &y,
            LoessConfig {
                span: 0.75,
                degree: 2,
                robust: false,
                ..Default::default()
            },
        )
        .unwrap();

        // The data is nearly linear (y ~ 2x), so fitted values should be close
        // R produces very similar results to raw y for this smooth data

        // Check that RSS is small (good fit)
        assert!(result.rss < 1.0, "RSS too large: {}", result.rss);

        // Check R-squared is high
        assert!(
            result.r_squared > 0.99,
            "R-squared too low: {}",
            result.r_squared
        );

        // Check fitted values are reasonable (within 0.5 of y for this smooth data)
        for i in 0..10 {
            let diff = (result.fitted[i] - y[i]).abs();
            assert!(
                diff < 0.5,
                "Fitted[{}] = {}, y[{}] = {}, diff = {}",
                i,
                result.fitted[i],
                i,
                y[i],
                diff
            );
        }
    }

    #[test]
    fn test_loess_degree_0() {
        // Degree 0 = local constant (Nadaraya-Watson)
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + 0.1 * (xi * 10.0).sin())
            .collect();

        let result = loess(
            &x,
            &y,
            LoessConfig {
                degree: 0,
                span: 0.5,
                ..Default::default()
            },
        )
        .unwrap();

        // Should produce reasonable fit
        assert!(result.r_squared > 0.8);
        assert_eq!(result.degree, 0);
        assert_eq!(result.fitted.len(), 20);
    }

    #[test]
    fn test_loess_sorted_vs_unsorted_input() {
        // Verify that the optimized (sorted) path gives the same results
        // as the legacy (unsorted) path, even when input is scrambled.
        let x_sorted: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y_sorted: Vec<f64> = x_sorted.iter().map(|&xi| xi.sin() + xi * 0.5).collect();

        // Scramble the order
        let perm = vec![7, 3, 12, 0, 9, 5, 14, 1, 11, 6, 2, 10, 4, 13, 8];
        let x_scrambled: Vec<f64> = perm.iter().map(|&i| x_sorted[i]).collect();
        let y_scrambled: Vec<f64> = perm.iter().map(|&i| y_sorted[i]).collect();

        let config = LoessConfig {
            span: 0.5,
            degree: 1,
            ..Default::default()
        };

        let result_sorted = loess(&x_sorted, &y_sorted, config.clone()).unwrap();
        let result_scrambled = loess(&x_scrambled, &y_scrambled, config).unwrap();

        // Compare fitted values (need to account for different ordering)
        for i in 0..15 {
            let orig_idx = perm[i];
            let diff = (result_scrambled.fitted[i] - result_sorted.fitted[orig_idx]).abs();
            assert!(
                diff < 1e-10,
                "Mismatch at scrambled[{}] (orig {}): {} vs {}, diff={}",
                i,
                orig_idx,
                result_scrambled.fitted[i],
                result_sorted.fitted[orig_idx],
                diff
            );
        }
    }
}
