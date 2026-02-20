//! Linear algebra utilities for econometric computations.
//!
//! Provides numerically stable matrix operations using the faer library.

use faer::linalg::solvers::Solve;
use faer::prelude::*;
use ndarray::{Array1, Array2, ArrayView2};
use thiserror::Error;

/// Error type for linear algebra operations.
#[derive(Debug, Error)]
pub enum LinalgError {
    #[error("Matrix is singular and cannot be inverted")]
    SingularMatrix,

    #[error("Matrix has high condition number ({0:.2e}), results may be numerically unstable")]
    IllConditioned(f64),

    #[error("Matrix dimensions are incompatible: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("SVD decomposition failed")]
    SvdFailed,

    #[error("Matrix must be square, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("Cholesky decomposition failed (matrix may not be positive definite)")]
    CholeskyFailed,

    #[error("Eigendecomposition failed")]
    EigenFailed,
}

/// Condition number threshold for ill-conditioning warnings
pub const CONDITION_THRESHOLD: f64 = 1e10;

/// Convert ndarray Array2 to faer Mat
pub fn ndarray_to_faer(arr: &ArrayView2<f64>) -> Mat<f64> {
    let (rows, cols) = arr.dim();
    Mat::from_fn(rows, cols, |i, j| arr[[i, j]])
}

/// Convert faer Mat to ndarray Array2
pub fn faer_to_ndarray(mat: &Mat<f64>) -> Array2<f64> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    Array2::from_shape_fn((rows, cols), |(i, j)| mat[(i, j)])
}

/// Convert faer MatRef to ndarray Array2
pub fn faer_ref_to_ndarray(mat: MatRef<'_, f64>) -> Array2<f64> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    Array2::from_shape_fn((rows, cols), |(i, j)| mat[(i, j)])
}

/// Convert faer Col to ndarray Array1
pub fn faer_col_to_ndarray(col: &Col<f64>) -> Array1<f64> {
    Array1::from_iter((0..col.nrows()).map(|i| col[i]))
}

/// Compute matrix inverse using LU decomposition.
/// Returns error if matrix is singular.
pub fn matrix_inverse(m: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }

    let mat = ndarray_to_faer(m);
    let lu = mat.full_piv_lu();

    // Create identity matrix to solve for inverse
    let n = rows;
    let mut identity = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        identity[(i, i)] = 1.0;
    }

    let inv = lu.solve(&identity);

    // Verify the inverse is valid by checking condition number
    let cond = condition_number(m)?;
    if cond > CONDITION_THRESHOLD {
        return Err(LinalgError::IllConditioned(cond));
    }

    Ok(faer_to_ndarray(&inv))
}

/// Compute Moore-Penrose pseudoinverse using SVD.
/// This is more numerically stable than regular inverse for ill-conditioned matrices.
pub fn pseudoinverse(m: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let mat = ndarray_to_faer(m);
    let (rows, cols) = m.dim();
    let min_dim = rows.min(cols);

    // Use singular_values to get Vec<f64> - unwrap the Result
    let s_vals = mat.singular_values().map_err(|_| LinalgError::SvdFailed)?;

    // Tolerance for considering singular values as zero
    let max_s = s_vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let tol = f64::EPSILON * (rows.max(cols) as f64) * max_s;

    // Now do thin_svd for the actual decomposition
    let svd = mat.thin_svd().map_err(|_| LinalgError::SvdFailed)?;
    let u = svd.U();
    let v = svd.V();

    // Compute S^+: reciprocal of non-zero singular values
    let mut s_inv = Mat::<f64>::zeros(cols, rows);
    for i in 0..min_dim {
        let val = s_vals[i];
        if val.abs() > tol {
            s_inv[(i, i)] = 1.0 / val;
        }
    }

    // Pseudoinverse = V * S^+ * U^T
    let result = v * &s_inv * u.transpose();

    Ok(faer_to_ndarray(&result))
}

/// Compute condition number of a matrix using SVD.
/// Condition number = max(singular values) / min(singular values)
pub fn condition_number(m: &ArrayView2<f64>) -> Result<f64, LinalgError> {
    let mat = ndarray_to_faer(m);

    let (rows, cols) = m.dim();
    let n = rows.min(cols);
    if n == 0 {
        return Err(LinalgError::SingularMatrix);
    }

    // Use singular_values() which returns Result<Vec<f64>, SvdError>
    let s_vals = mat.singular_values().map_err(|_| LinalgError::SvdFailed)?;

    let max_s = s_vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let min_s = s_vals.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);

    if min_s == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(max_s / min_s)
}

/// Safe matrix inverse with pseudoinverse fallback.
/// Returns the regular inverse if well-conditioned, otherwise uses pseudoinverse.
///
/// Optimized to avoid redundant condition number computation.
pub fn safe_inverse(m: &ArrayView2<f64>) -> Result<(Array2<f64>, Option<f64>), LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }

    let mat = ndarray_to_faer(m);

    // Try Cholesky first (faster for symmetric positive definite matrices like X'X)
    if let Ok(chol) = mat.llt(faer::Side::Lower) {
        // Solve for inverse: A * A^{-1} = I
        let n = rows;
        let mut identity = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            identity[(i, i)] = 1.0;
        }
        let inv = chol.solve(&identity);
        return Ok((faer_to_ndarray(&inv), None));
    }

    // Cholesky failed - matrix might not be positive definite
    // Fall back to LU decomposition with condition number check
    let lu = mat.full_piv_lu();

    // Create identity matrix to solve for inverse
    let n = rows;
    let mut identity = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        identity[(i, i)] = 1.0;
    }

    let inv = lu.solve(&identity);

    // Check condition number only if Cholesky failed (non-positive-definite case)
    let cond = condition_number(m)?;
    if cond > CONDITION_THRESHOLD {
        // Ill-conditioned: use pseudoinverse and return warning
        let pinv = pseudoinverse(m)?;
        return Ok((pinv, Some(cond)));
    }

    Ok((faer_to_ndarray(&inv), None))
}

/// Cholesky decomposition: M = L * L^T
/// Returns the lower triangular matrix L.
/// Only works for symmetric positive definite matrices.
pub fn cholesky(m: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }

    let mat = ndarray_to_faer(m);

    // Use faer's llt() for Cholesky decomposition
    let chol = mat
        .llt(faer::Side::Lower)
        .map_err(|_| LinalgError::CholeskyFailed)?;
    let l = chol.L();

    Ok(faer_ref_to_ndarray(l))
}

/// Solve linear system Ax = b using LU decomposition.
pub fn solve(a: &ArrayView2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let (rows, cols) = a.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }
    if rows != b.len() {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("{} rows", rows),
            actual: format!("{} elements in b", b.len()),
        });
    }

    let mat_a = ndarray_to_faer(a);
    let lu = mat_a.full_piv_lu();

    // Convert b to faer Col
    let col_b = Col::<f64>::from_fn(b.len(), |i| b[i]);
    let x = lu.solve(&col_b);

    Ok(faer_col_to_ndarray(&x))
}

/// Eigenvalue decomposition for symmetric matrices.
/// Returns (eigenvalues, eigenvectors).
pub fn eig_symmetric(m: &ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }

    let mat = ndarray_to_faer(m);

    // Use self_adjoint_eigen for symmetric matrices
    let eig = mat
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| LinalgError::EigenFailed)?;

    // Extract eigenvalues using self_adjoint_eigenvalues which returns Result<Vec<f64>, EvdError>
    let eigenvalues_vec = mat
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|_| LinalgError::EigenFailed)?;
    let eigenvectors = eig.U();

    let vals = Array1::from_vec(eigenvalues_vec);
    let vecs = faer_ref_to_ndarray(eigenvectors);

    Ok((vals, vecs))
}

/// Matrix multiplication: A * B
///
/// When the `cuda` feature is enabled and a GPU is available, dispatches to
/// cuBLAS DGEMM for large matrices (m*n*k >= threshold).
pub fn matmul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();

    if a_cols != b_rows {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("{} columns in A", a_cols),
            actual: format!("{} rows in B", b_rows),
        });
    }

    // GPU dispatch for large matrices. Also check shape ratio to avoid
    // tall-skinny cases where GPU is much slower (e.g., 100Kx50x50 → 6x slower).
    #[cfg(feature = "cuda")]
    if let Some(ctx) = super::gpu::GpuContext::get() {
        let mnk = a_rows * b_cols * a_cols;
        let dims = [a_rows, a_cols, b_cols];
        let min_d = *dims.iter().min().unwrap() as f64;
        let max_d = *dims.iter().max().unwrap() as f64;
        let shape_ratio = if max_d > 0.0 { min_d / max_d } else { 0.0 };
        if mnk >= ctx.thresholds.matmul_min_mnk
            && shape_ratio >= ctx.thresholds.matmul_min_shape_ratio
        {
            match super::gpu::matmul_gpu(ctx, a, b) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!("GPU matmul failed, falling back to CPU: {:?}", e);
                }
            }
        }
    }

    let mat_a = ndarray_to_faer(a);
    let mat_b = ndarray_to_faer(b);
    let result = &mat_a * &mat_b;

    Ok(faer_to_ndarray(&result))
}

/// Compute X'X (X transpose times X)
/// Uses ndarray's native dot product for better performance on larger matrices.
///
/// When the `cuda` feature is enabled and a GPU is available, dispatches to
/// cuBLAS DGEMM for large matrices (n >= threshold).
pub fn xtx(x: &ArrayView2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();

    // GPU dispatch: xtx compute is O(n*k²), so use n*k*k as dispatch metric.
    // GPU wins when k is large (k >= ~30); for small k the CPU path is faster
    // even at very large n (benchmarked up to 1M x 10 → CPU still wins).
    #[cfg(feature = "cuda")]
    if let Some(ctx) = super::gpu::GpuContext::get() {
        if k >= ctx.thresholds.xtx_min_k && n * k * k >= ctx.thresholds.xtx_min_nkk {
            match super::gpu::xtx_gpu(ctx, x) {
                Ok(result) => return result,
                Err(e) => {
                    tracing::warn!("GPU xtx failed, falling back to CPU: {:?}", e);
                }
            }
        }
    }

    // CPU path (unchanged)
    if k <= 20 {
        x.t().dot(x)
    } else {
        // Use faer for larger matrices where BLAS benefits outweigh conversion cost
        let mat = ndarray_to_faer(x);
        let result = mat.transpose() * &mat;
        faer_to_ndarray(&result)
    }
}

/// Fast inverse using Cholesky decomposition.
/// Use this for positive definite matrices (like X'X) where we don't need
/// condition number checks. Much faster than safe_inverse for small matrices.
///
/// When the `cuda` feature is enabled and a GPU is available, dispatches to
/// GPU for large matrices (k >= threshold).
pub fn cholesky_inverse(m: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }

    // GPU dispatch for large matrices
    #[cfg(feature = "cuda")]
    if let Some(ctx) = super::gpu::GpuContext::get() {
        if rows >= ctx.thresholds.inverse_min_k {
            match super::gpu::cholesky_inverse_gpu(ctx, m) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!("GPU cholesky_inverse failed, falling back to CPU: {:?}", e);
                }
            }
        }
    }

    let mat = ndarray_to_faer(m);

    // Use Cholesky decomposition (L * L^T = M)
    let chol = mat
        .llt(faer::Side::Lower)
        .map_err(|_| LinalgError::CholeskyFailed)?;

    // Solve for inverse: M^{-1} = (L * L^T)^{-1} = L^{-T} * L^{-1}
    // We solve M * X = I for X
    let n = rows;
    let mut identity = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        identity[(i, i)] = 1.0;
    }

    let inv = chol.solve(&identity);
    Ok(faer_to_ndarray(&inv))
}

/// Compute X'y (X transpose times y)
/// Uses ndarray's native dot product for better performance.
///
/// When the `cuda` feature is enabled and a GPU is available, dispatches to
/// cuBLAS DGEMV for large matrices (n*k >= threshold).
pub fn xty(x: &ArrayView2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let (n, k) = x.dim();

    // GPU dispatch: xty is bandwidth-bound O(n*k) DGEMV; GPU only helps at
    // large n. For large k the CPU is already well-parallelized via BLAS.
    #[cfg(feature = "cuda")]
    if let Some(ctx) = super::gpu::GpuContext::get() {
        if n >= ctx.thresholds.xty_min_n {
            match super::gpu::xty_gpu(ctx, x, y) {
                Ok(result) => return result,
                Err(e) => {
                    tracing::warn!("GPU xty failed, falling back to CPU: {:?}", e);
                }
            }
        }
    }

    let _ = (n, k); // suppress unused warnings when cuda feature is off
    x.t().dot(y)
}

/// Compute (X'X)^{-1} with optional condition number warning
pub fn xtx_inv(x: &ArrayView2<f64>) -> Result<(Array2<f64>, Option<f64>), LinalgError> {
    let xtx_mat = xtx(x);
    safe_inverse(&xtx_mat.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matrix_inverse() {
        let m = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = matrix_inverse(&m.view()).unwrap();

        // Check M * M^{-1} ≈ I
        let identity = matmul(&m.view(), &inv.view()).unwrap();
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(identity[[0, 1]].abs() < 1e-10);
        assert!(identity[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_condition_number() {
        // Well-conditioned matrix
        let m = array![[1.0, 0.0], [0.0, 1.0]];
        let cond = condition_number(&m.view()).unwrap();
        assert!((cond - 1.0).abs() < 1e-10);

        // Ill-conditioned matrix
        let m2 = array![[1.0, 1.0], [1.0, 1.0 + 1e-15]];
        let cond2 = condition_number(&m2.view()).unwrap();
        assert!(cond2 > 1e10);
    }

    #[test]
    fn test_solve() {
        let a = array![[3.0, 1.0], [1.0, 2.0]];
        let b = array![9.0, 8.0];
        let x = solve(&a.view(), &b).unwrap();

        // Check Ax ≈ b
        assert!((3.0 * x[0] + 1.0 * x[1] - 9.0).abs() < 1e-10);
        assert!((1.0 * x[0] + 2.0 * x[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        // Symmetric positive definite matrix
        let m = array![[4.0, 2.0], [2.0, 5.0]];
        let l = cholesky(&m.view()).unwrap();

        // Check L * L^T ≈ M
        let reconstructed = matmul(&l.view(), &l.t().to_owned().view()).unwrap();
        assert!((reconstructed[[0, 0]] - m[[0, 0]]).abs() < 1e-10);
        assert!((reconstructed[[1, 1]] - m[[1, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_xtx() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = xtx(&x.view());

        // X'X should be 2x2
        assert_eq!(result.dim(), (2, 2));
        // [1,3,5]'[1,3,5] = 1+9+25 = 35
        assert!((result[[0, 0]] - 35.0).abs() < 1e-10);
    }
}
