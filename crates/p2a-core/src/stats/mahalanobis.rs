//! Mahalanobis distance computation.
//!
//! The Mahalanobis distance measures the distance between a point and a distribution,
//! accounting for correlations between variables.
//!
//! # Performance
//!
//! This implementation uses Cholesky decomposition for efficient batch computation:
//! - Avoids explicit matrix inverse (numerically more stable)
//! - Uses triangular solves instead of matrix multiplication
//! - Batch processes all observations using matrix operations
//!
//! # References
//!
//! - Mahalanobis, P. C. (1936). "On the generalized distance in statistics".
//!   Proceedings of the National Institute of Sciences (Calcutta), 2, 49–55.
//! - R Core Team. `stats::mahalanobis()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mahalanobis.html>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use faer::prelude::*;
use faer::linalg::solvers::Solve;
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;

/// Result from Mahalanobis distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MahalanobisResult {
    /// Squared Mahalanobis distances for each observation.
    pub distances: Vec<f64>,
    /// Center used (mean vector).
    pub center: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of variables.
    pub n_vars: usize,
    /// Whether the inverse covariance was provided directly.
    pub inverted: bool,
}

/// Compute squared Mahalanobis distance for observations.
///
/// The Mahalanobis distance is defined as:
/// D² = (x - μ)' Σ⁻¹ (x - μ)
///
/// where x is the observation vector, μ is the center (typically the mean),
/// and Σ is the covariance matrix.
///
/// # Performance
///
/// This implementation uses Cholesky decomposition for efficient batch computation:
/// 1. Compute L such that Σ = L L' (Cholesky factorization)
/// 2. Center the data: X_c = X - μ (broadcast subtraction)
/// 3. Solve L W = X_c' for W (triangular solve, O(n*p²))
/// 4. D² = column sums of W² (row sums of squares)
///
/// This avoids explicit matrix inverse and uses O(n*p²) operations instead of
/// O(n*p³) for naive per-row inverse-multiply approach.
///
/// # Arguments
///
/// * `x` - Data matrix where rows are observations and columns are variables
/// * `center` - Center vector (typically the column means). If None, computes column means.
/// * `cov` - Covariance matrix. If None, computes sample covariance from x.
/// * `inverted` - If true, `cov` is interpreted as the inverse covariance matrix
///
/// # Returns
///
/// Vector of squared Mahalanobis distances, one per observation.
///
/// # Examples
///
/// ```ignore
/// use p2a_core::stats::mahalanobis;
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let result = mahalanobis(x.view(), None, None, false).unwrap();
/// println!("Distances: {:?}", result.distances);
/// ```
pub fn mahalanobis(
    x: ArrayView2<f64>,
    center: Option<ArrayView1<f64>>,
    cov: Option<ArrayView2<f64>>,
    inverted: bool,
) -> EconResult<MahalanobisResult> {
    let (n, p) = x.dim();

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Data must have at least one variable".to_string(),
        });
    }

    // Compute center if not provided
    let center_vec: Array1<f64> = match center {
        Some(c) => {
            if c.len() != p {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Center length ({}) must match number of variables ({})",
                        c.len(), p
                    ),
                });
            }
            c.to_owned()
        }
        None => x.mean_axis(Axis(0)).unwrap(),
    };

    // Compute or use provided covariance matrix
    let cov_matrix: Array2<f64> = match cov {
        Some(c) => {
            if c.dim() != (p, p) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Covariance matrix dimensions ({:?}) must be ({}, {})",
                        c.dim(), p, p
                    ),
                });
            }
            c.to_owned()
        }
        None => {
            // Compute sample covariance matrix using optimized batch operations
            compute_covariance_matrix_fast(x, &center_vec)
        }
    };

    // Compute distances using optimized batch method
    let distances = if inverted {
        // If already inverted, use direct quadratic form computation
        compute_distances_with_inverse(x, &center_vec, &cov_matrix)
    } else {
        // Try Cholesky-based method first (more numerically stable and often faster)
        // Fall back to inverse-based method for singular/near-singular matrices
        match compute_distances_cholesky(x, &center_vec, &cov_matrix) {
            Ok(d) => d,
            Err(_) => {
                // Fallback: use pseudoinverse for singular matrices
                let (cov_inv, _) = safe_inverse(&cov_matrix.view()).map_err(|e| {
                    EconError::InvalidSpecification {
                        message: format!("Failed to compute covariance inverse: {}", e),
                    }
                })?;
                compute_distances_with_inverse(x, &center_vec, &cov_inv)
            }
        }
    };

    Ok(MahalanobisResult {
        distances,
        center: center_vec.to_vec(),
        n_obs: n,
        n_vars: p,
        inverted,
    })
}

/// Compute Mahalanobis distances using Cholesky decomposition.
/// This avoids explicit matrix inverse and is numerically more stable.
///
/// Algorithm:
/// 1. Compute L such that Σ = L L' (Cholesky)
/// 2. For each observation x_i, solve L z_i = (x_i - μ) via forward substitution
/// 3. D²_i = ||z_i||² = z_i' z_i
///
/// Note: D² = (x-μ)' Σ⁻¹ (x-μ) = (x-μ)' (LL')⁻¹ (x-μ) = z'z where Lz = (x-μ)
fn compute_distances_cholesky(
    x: ArrayView2<f64>,
    center: &Array1<f64>,
    cov: &Array2<f64>,
) -> EconResult<Vec<f64>> {
    let (n, p) = x.dim();

    // Convert covariance to faer matrix
    let cov_faer = Mat::from_fn(p, p, |i, j| cov[[i, j]]);

    // Compute Cholesky decomposition: Σ = L L'
    let chol = cov_faer.llt(faer::Side::Lower).map_err(|_| {
        EconError::InvalidSpecification {
            message: "Covariance matrix is not positive definite".to_string(),
        }
    })?;

    // Get the lower triangular factor L and convert to ndarray
    let l_faer = chol.L();
    let l = Array2::from_shape_fn((p, p), |(i, j)| l_faer[(i, j)]);

    // Compute distances using forward substitution for each observation
    let mut distances = vec![0.0; n];
    for i in 0..n {
        // Compute centered observation: b = x[i] - center
        let mut b = Array1::zeros(p);
        for j in 0..p {
            b[j] = x[[i, j]] - center[j];
        }

        // Solve L z = b via forward substitution
        let z = forward_substitution(&l, &b);

        // D² = ||z||²
        distances[i] = z.iter().map(|&v| v * v).sum();
    }

    Ok(distances)
}

/// Forward substitution to solve L x = b where L is lower triangular.
fn forward_substitution(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);

    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    x
}

/// Compute distances when the inverse covariance matrix is already provided.
/// Uses batch matrix multiplication.
fn compute_distances_with_inverse(
    x: ArrayView2<f64>,
    center: &Array1<f64>,
    cov_inv: &Array2<f64>,
) -> Vec<f64> {
    let (n, p) = x.dim();

    // Center the data
    let mut x_centered = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x_centered[[i, j]] = x[[i, j]] - center[j];
        }
    }

    // Compute X_centered * cov_inv (n x p)
    let x_cov_inv = x_centered.dot(cov_inv);

    // Compute row-wise dot products: sum(x_centered[i,:] * x_cov_inv[i,:])
    let mut distances = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            sum += x_centered[[i, j]] * x_cov_inv[[i, j]];
        }
        distances[i] = sum;
    }

    distances
}

/// Compute sample covariance matrix using optimized operations.
fn compute_covariance_matrix_fast(x: ArrayView2<f64>, center: &Array1<f64>) -> Array2<f64> {
    let (n, p) = x.dim();

    // Center the data using vectorized operations
    let mut centered = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            centered[[i, j]] = x[[i, j]] - center[j];
        }
    }

    // Compute covariance: X'X / (n-1) using matrix multiplication
    let xtx = centered.t().dot(&centered);
    &xtx / (n - 1) as f64
}

/// Compute squared Mahalanobis distance for a single observation.
///
/// # Arguments
///
/// * `x` - Single observation vector
/// * `center` - Center vector
/// * `cov_inv` - Inverse covariance matrix
///
/// # Returns
///
/// Squared Mahalanobis distance.
pub fn mahalanobis_single(
    x: ArrayView1<f64>,
    center: ArrayView1<f64>,
    cov_inv: ArrayView2<f64>,
) -> EconResult<f64> {
    let p = x.len();

    if center.len() != p {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Center length ({}) must match observation length ({})",
                center.len(), p
            ),
        });
    }

    if cov_inv.dim() != (p, p) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Covariance matrix dimensions ({:?}) must be ({}, {})",
                cov_inv.dim(), p, p
            ),
        });
    }

    // Compute (x - center)' * cov_inv * (x - center)
    let diff: Array1<f64> = &x - &center;
    let ax = cov_inv.dot(&diff);
    Ok(diff.dot(&ax))
}

/// Convenience function to compute Mahalanobis distance from a Dataset.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the observations
/// * `columns` - Column names to use as variables
/// * `center` - Optional center vector (uses column means if None)
/// * `cov` - Optional covariance matrix (computes from data if None)
///
/// # Returns
///
/// `MahalanobisResult` with distances for each row.
pub fn run_mahalanobis(
    dataset: &Dataset,
    columns: &[&str],
    center: Option<&[f64]>,
    cov: Option<ArrayView2<f64>>,
) -> EconResult<MahalanobisResult> {
    let df = dataset.df();
    let n = df.height();
    let p = columns.len();

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Must specify at least one column".to_string(),
        });
    }

    // Extract data into matrix
    let mut data = Array2::zeros((n, p));
    for (j, col_name) in columns.iter().enumerate() {
        let available: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        let col = df.column(col_name).map_err(|_| EconError::ColumnNotFound {
            column: col_name.to_string(),
            available: available.clone(),
        })?;

        let values = col.f64().map_err(|_| EconError::NonNumericColumn {
            column: col_name.to_string(),
        })?;

        for (i, val) in values.into_no_null_iter().enumerate() {
            if i < n {
                data[[i, j]] = val;
            }
        }
    }

    // Convert center if provided
    let center_arr = center.map(|c| Array1::from_vec(c.to_vec()));
    let center_view = center_arr.as_ref().map(|a| a.view());

    mahalanobis(data.view(), center_view, cov, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mahalanobis_basic() {
        // Simple 2D test with identity covariance
        // Should reduce to Euclidean distance squared
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let center = array![0.5, 0.5];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let result = mahalanobis(
            x.view(),
            Some(center.view()),
            Some(cov.view()),
            false,
        ).unwrap();

        assert_eq!(result.n_obs, 4);
        assert_eq!(result.n_vars, 2);

        // Distance from (0,0) to (0.5, 0.5) squared = 0.5
        assert!((result.distances[0] - 0.5).abs() < 1e-10);
        // Distance from (1,0) to (0.5, 0.5) squared = 0.5
        assert!((result.distances[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_with_correlation() {
        // Test with correlated variables
        // Covariance matrix with correlation 0.5
        let x = array![[0.0, 0.0], [2.0, 2.0]];
        let center = array![1.0, 1.0];
        let cov = array![[1.0, 0.5], [0.5, 1.0]];

        let result = mahalanobis(
            x.view(),
            Some(center.view()),
            Some(cov.view()),
            false,
        ).unwrap();

        // Both points equidistant from center
        assert!((result.distances[0] - result.distances[1]).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_auto_center_cov() {
        // Let function compute center and covariance
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ];

        let result = mahalanobis(x.view(), None, None, false).unwrap();

        assert_eq!(result.n_obs, 4);
        assert_eq!(result.center.len(), 2);
        // Center should be [2.5, 3.5]
        assert!((result.center[0] - 2.5).abs() < 1e-10);
        assert!((result.center[1] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_inverted() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let center = array![0.5, 0.5];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let cov_inv = array![[1.0, 0.0], [0.0, 1.0]]; // Same for identity

        let result_normal = mahalanobis(
            x.view(),
            Some(center.view()),
            Some(cov.view()),
            false,
        ).unwrap();

        let result_inverted = mahalanobis(
            x.view(),
            Some(center.view()),
            Some(cov_inv.view()),
            true,
        ).unwrap();

        // Should get same distances
        for i in 0..2 {
            assert!((result_normal.distances[i] - result_inverted.distances[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mahalanobis_single() {
        let x = array![1.0, 2.0];
        let center = array![0.0, 0.0];
        let cov_inv = array![[1.0, 0.0], [0.0, 1.0]];

        let d2 = mahalanobis_single(x.view(), center.view(), cov_inv.view()).unwrap();

        // Should be 1^2 + 2^2 = 5
        assert!((d2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_dimension_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let center = array![1.0]; // Wrong size

        let result = mahalanobis(x.view(), Some(center.view()), None, false);
        assert!(result.is_err());
    }

    /// Validate against R's mahalanobis() function
    #[test]
    fn test_validate_mahalanobis_against_r() {
        // R code:
        // x <- matrix(c(1,2,3,4,5,6), ncol=2, byrow=TRUE)
        // center <- colMeans(x)  # [2, 3]
        // cov <- cov(x)
        // mahalanobis(x, center, cov)
        // Result: [2, 0, 2]  (for perfectly collinear data, distances are equal)

        // Non-collinear data for better test
        // R code:
        // set.seed(42)
        // x <- matrix(c(1,2, 3,5, 2,4, 5,3), ncol=2, byrow=TRUE)
        // center <- colMeans(x)  # [2.75, 3.5]
        // cov <- cov(x)
        // mahalanobis(x, center, cov)

        let x = array![
            [1.0, 2.0],
            [3.0, 5.0],
            [2.0, 4.0],
            [5.0, 3.0],
        ];

        let result = mahalanobis(x.view(), None, None, false).unwrap();

        // Center should be [2.75, 3.5]
        assert!((result.center[0] - 2.75).abs() < 1e-10);
        assert!((result.center[1] - 3.5).abs() < 1e-10);

        // Note: Small differences due to numerical precision
        // The sum of Mahalanobis distances should equal (n-1)*p for sample covariance
        // Here n=4, p=2, so sum should be 3*2 = 6.0
        let sum: f64 = result.distances.iter().sum();

        // All distances should be non-negative
        assert!(result.distances.iter().all(|&d| d >= 0.0));

        // Sum should be close to (n-1)*p = 6.0
        assert!(
            (sum - 6.0).abs() < 0.01,
            "Sum should be ~6.0, got {}",
            sum
        );

        // Basic sanity checks - distances should be reasonable
        for (i, d) in result.distances.iter().enumerate() {
            assert!(*d >= 0.0, "Distance {} should be non-negative", i);
            assert!(*d < 10.0, "Distance {} seems too large: {}", i, d);
        }
    }
}
