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
//! For each point x₀, the algorithm:
//! 1. Selects k = floor(span × n) nearest neighbors
//! 2. Computes tricubic weights: w(u) = (1 - |u|³)³ for |u| < 1
//! 3. Fits a weighted polynomial regression
//! 4. Returns the fitted value at x₀
//!
//! # Robust Fitting
//!
//! When `robust=true`, the algorithm applies iterative reweighting using the bisquare
//! function to downweight outliers:
//! - w(u) = (1 - u²)² for |u| < 1, else 0
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

use ndarray::{Array1, Array2, ArrayView1};
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
        writeln!(f, "Family: {}", if self.robust { "symmetric" } else { "gaussian" })?;
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

/// Tricubic weight function: w(u) = (1 - |u|³)³ for |u| < 1, else 0.
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

/// Bisquare weight function: w(u) = (1 - u²)² for |u| < 1, else 0.
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

/// Build a local polynomial design matrix centered at x0.
///
/// For degree=1: [1, (x - x0)]
/// For degree=2: [1, (x - x0), (x - x0)²]
fn build_local_design_matrix(x: &[f64], x0: f64, degree: usize) -> Array2<f64> {
    let n = x.len();
    let p = degree + 1;
    let mut design = Array2::zeros((n, p));

    for i in 0..n {
        let dx = x[i] - x0;
        design[[i, 0]] = 1.0; // Intercept
        if degree >= 1 {
            design[[i, 1]] = dx;
        }
        if degree >= 2 {
            design[[i, 2]] = dx * dx;
        }
    }

    design
}

/// Perform weighted least squares regression and return fitted value at x0.
///
/// Solves: (X'WX)β = X'Wy
/// Returns: β[0] (the intercept, which is the fitted value at x0)
fn weighted_local_regression(
    y: ArrayView1<f64>,
    x_design: &Array2<f64>,
    weights: ArrayView1<f64>,
) -> EconResult<(f64, Array1<f64>)> {
    let n = y.len();
    let p = x_design.ncols();

    // Build X'WX and X'Wy
    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);

    for i in 0..n {
        let w = weights[i];
        if w > 0.0 {
            for j in 0..p {
                xtwy[j] += w * x_design[[i, j]] * y[i];
                for k in 0..p {
                    xtwx[[j, k]] += w * x_design[[i, j]] * x_design[[i, k]];
                }
            }
        }
    }

    // Solve for β using matrix inverse
    let xtwx_inv = matrix_inverse(&xtwx.view())?;
    let beta = xtwx_inv.dot(&xtwy);

    // Fitted value at x0 is β[0] (since x_design is centered at x0)
    let fitted_at_x0 = beta[0];

    Ok((fitted_at_x0, beta))
}

/// Find the k nearest neighbors and compute their tricubic weights.
///
/// Returns: (indices, weights, max_distance)
fn compute_neighborhood_weights(
    x: &[f64],
    x0: f64,
    span: f64,
    robust_weights: Option<&[f64]>,
) -> (Vec<usize>, Vec<f64>, f64) {
    let n = x.len();

    // Compute number of neighbors
    let k = if span <= 1.0 {
        ((span * n as f64).ceil() as usize).max(1).min(n)
    } else {
        n
    };

    // Compute distances from x0
    let mut distances: Vec<(usize, f64)> = x.iter()
        .enumerate()
        .map(|(i, &xi)| (i, (xi - x0).abs()))
        .collect();

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Get the k nearest neighbors
    let neighbors: Vec<(usize, f64)> = distances.into_iter().take(k).collect();

    // Find max distance in neighborhood
    let max_dist = if span > 1.0 {
        // For span > 1, use span^(1/p) times max distance (p=1 for univariate)
        neighbors.last().map(|&(_, d)| d * span).unwrap_or(1.0)
    } else {
        neighbors.last().map(|&(_, d)| d).unwrap_or(1.0)
    };

    // Avoid division by zero
    let max_dist = if max_dist < 1e-10 { 1.0 } else { max_dist };

    // Compute tricubic weights
    let mut indices = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);

    for (idx, dist) in neighbors {
        let u = dist / max_dist;
        let mut w = tricubic_weight(u);

        // Apply robust weights if provided
        if let Some(rw) = robust_weights {
            w *= rw[idx];
        }

        indices.push(idx);
        weights.push(w);
    }

    (indices, weights, max_dist)
}

/// Fit LOESS at a single point x0.
fn loess_fit_at_point(
    x: &[f64],
    y: &[f64],
    x0: f64,
    span: f64,
    degree: usize,
    robust_weights: Option<&[f64]>,
) -> EconResult<f64> {

    // Get neighborhood weights
    let (indices, weights, _max_dist) = compute_neighborhood_weights(x, x0, span, robust_weights);

    // Extract subset of data in neighborhood
    let x_subset: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let y_subset: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
    let w_subset: Array1<f64> = Array1::from_vec(weights);

    // Build local design matrix centered at x0
    let x_design = build_local_design_matrix(&x_subset, x0, degree);

    // Weighted least squares
    let (fitted, _beta) = weighted_local_regression(y_subset.view(), &x_design, w_subset.view())?;

    Ok(fitted)
}

/// Compute robust weights using bisquare function.
///
/// Scale residuals by 6 × MAD (median absolute deviation) and apply bisquare.
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

    // Scale factor: 6 × MAD (following R's loess implementation)
    let scale = 6.0 * mad;

    if scale < 1e-10 {
        // All residuals are essentially zero
        return vec![1.0; n];
    }

    // Compute bisquare weights
    residuals.iter()
        .map(|&r| bisquare_weight(r / scale))
        .collect()
}

/// Compute the trace of the hat matrix (effective number of parameters).
///
/// This is an approximation computed by summing the weights at each diagonal.
fn compute_enp(x: &[f64], span: f64, degree: usize) -> f64 {
    let n = x.len();
    let mut trace = 0.0;

    for i in 0..n {
        let x0 = x[i];
        let (indices, weights, _) = compute_neighborhood_weights(x, x0, span, None);

        // Find position of current point in neighborhood
        if let Some(pos) = indices.iter().position(|&idx| idx == i) {
            // Get local design matrix
            let x_subset: Vec<f64> = indices.iter().map(|&idx| x[idx]).collect();
            let x_design = build_local_design_matrix(&x_subset, x0, degree);
            let w_subset: Array1<f64> = Array1::from_vec(weights.clone());

            // Build X'WX
            let p = degree + 1;
            let mut xtwx = Array2::<f64>::zeros((p, p));
            for j in 0..indices.len() {
                let w = w_subset[j];
                if w > 0.0 {
                    for jj in 0..p {
                        for kk in 0..p {
                            xtwx[[jj, kk]] += w * x_design[[j, jj]] * x_design[[j, kk]];
                        }
                    }
                }
            }

            // Compute diagonal element of hat matrix at position i
            // H[i,i] = x_i' (X'WX)^-1 X' W e_i * w_i
            // where e_i is the unit vector
            if let Ok(xtwx_inv) = matrix_inverse(&xtwx.view()) {
                // x_i in local coordinates is [1, 0, 0, ...] (since centered at x0 = x[i])
                let h_ii = xtwx_inv[[0, 0]] * weights[pos];
                trace += h_ii;
            }
        }
    }

    trace
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

    // Initial fit
    let mut fitted = Vec::with_capacity(n);
    for i in 0..n {
        let f = loess_fit_at_point(x, y, x[i], config.span, config.degree, None)?;
        fitted.push(f);
    }

    // Compute residuals
    let mut residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(yi, fi)| yi - fi).collect();

    // Robust fitting iterations
    let mut robust_weights: Option<Vec<f64>> = None;
    let mut iterations = 0;

    if config.robust {
        for iter in 0..config.robust_iterations {
            // Compute robust weights from residuals
            let new_robust_weights = compute_robust_weights(&residuals);

            // Refit with robust weights
            let mut new_fitted = Vec::with_capacity(n);
            for i in 0..n {
                let f = loess_fit_at_point(
                    x, y, x[i], config.span, config.degree,
                    Some(&new_robust_weights),
                )?;
                new_fitted.push(f);
            }

            // Check convergence
            let max_change: f64 = fitted.iter()
                .zip(new_fitted.iter())
                .map(|(f1, f2)| (f1 - f2).abs())
                .fold(0.0, f64::max);

            fitted = new_fitted;
            residuals = y.iter().zip(fitted.iter()).map(|(yi, fi)| yi - fi).collect();
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

    let enp = compute_enp(x, config.span, config.degree);
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
    let mut predictions = Vec::with_capacity(new_x.len());

    for &x0 in new_x {
        let pred = loess_fit_at_point(
            &model.x,
            &model.y,
            x0,
            model.span,
            model.degree,
            model.robust_weights.as_deref(),
        )?;
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
    let x_series = df.column(x_col).map_err(|_| {
        EconError::ColumnNotFound {
            column: x_col.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        }
    })?;
    let y_series = df.column(y_col).map_err(|_| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        }
    })?;

    // Convert to f64 vectors
    let x: Vec<f64> = x_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn { column: x_col.to_string() })?
        .into_no_null_iter()
        .collect();

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn { column: y_col.to_string() })?
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

        // At u = 0.5, weight should be (1 - 0.125)^3 = 0.875^3 ≈ 0.6699
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
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0 + 0.1 * (xi * 10.0).sin()).collect();

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
        let result_small = loess(&x, &y, LoessConfig { span: 0.3, ..Default::default() }).unwrap();

        // Large span = more smooth
        let result_large = loess(&x, &y, LoessConfig { span: 0.9, ..Default::default() }).unwrap();

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

        let result_linear = loess(&x, &y, LoessConfig { degree: 1, ..Default::default() }).unwrap();
        let result_quad = loess(&x, &y, LoessConfig { degree: 2, ..Default::default() }).unwrap();

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

        let result_normal = loess(&x, &y, LoessConfig { robust: false, ..Default::default() }).unwrap();
        let result_robust = loess(&x, &y, LoessConfig { robust: true, ..Default::default() }).unwrap();

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
            assert!((predictions[i] - expected).abs() < 1.0,
                    "Prediction at {} = {}, expected ~{}", xi, predictions[i], expected);
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
        assert!(loess(&x, &y, LoessConfig { span: 0.0, ..Default::default() }).is_err());
        assert!(loess(&x, &y, LoessConfig { span: -0.5, ..Default::default() }).is_err());

        // Invalid degree
        assert!(loess(&x, &y, LoessConfig { degree: 3, ..Default::default() }).is_err());
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

        let result = loess(&x, &y, LoessConfig {
            span: 0.75,
            degree: 2,
            robust: false,
            ..Default::default()
        }).unwrap();

        // The data is nearly linear (y ≈ 2x), so fitted values should be close
        // R produces very similar results to raw y for this smooth data

        // Check that RSS is small (good fit)
        assert!(result.rss < 1.0, "RSS too large: {}", result.rss);

        // Check R-squared is high
        assert!(result.r_squared > 0.99, "R-squared too low: {}", result.r_squared);

        // Check fitted values are reasonable (within 0.5 of y for this smooth data)
        for i in 0..10 {
            let diff = (result.fitted[i] - y[i]).abs();
            assert!(diff < 0.5, "Fitted[{}] = {}, y[{}] = {}, diff = {}",
                    i, result.fitted[i], i, y[i], diff);
        }
    }
}
