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

    #[error("Matrix contains non-finite values (NaN or Inf)")]
    NonFinite,
}

/// Condition number threshold for ill-conditioning warnings
pub const CONDITION_THRESHOLD: f64 = 1e10;

/// Residual allowance of a computed inverse or solution: in units of
/// `n * eps * cond(A)` for an inverse (`max_ij |(A X - I)_ij|`), and of
/// `n * eps * (||A|| ||x|| + ||b||)` for a solution of `A x = b` (infinity norms,
/// that is, a normwise backward error).
///
/// A correct inverse leaves a residual of a small multiple of `n * eps * cond(A)`;
/// LAPACK's own test of its inverses (`dget03`) accepts ratios up to 30. The
/// planning sweep for RF-31 (435 matrices with `cond <= 1e10`: random dense,
/// negative definite, symmetric indefinite, sparse structured, graded, and
/// multinomial logit Hessians up to 495 x 495) measured at most 0.37 for
/// partial-pivot LU and 0.75 for the SVD route on correct inverses, and at least
/// 5.4e12 (or NaN) on every wrong inverse of faer 0.22.6's full-pivot LU, so 100
/// separates the two with a wide margin on both sides.
pub const INVERSE_RESIDUAL_FACTOR: f64 = 100.0;

/// Whether every entry of `m` is finite.
fn is_all_finite(m: &ArrayView2<f64>) -> bool {
    m.iter().all(|v| v.is_finite())
}

/// `max_ij |(A X - I)_ij|`, or `+inf` if `X` has a non-finite entry (checked before
/// the product) or the product overflows.
fn inverse_residual(a: &Mat<f64>, x: &Mat<f64>) -> f64 {
    let n = a.nrows();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return f64::INFINITY;
            }
        }
    }
    let p = a * x;
    let mut r = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let d = (p[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs();
            if !d.is_finite() {
                return f64::INFINITY;
            }
            r = r.max(d);
        }
    }
    r
}

/// `mat^{-1}` by LU with partial (row) pivoting, solved against the identity.
fn partial_piv_lu_inverse(mat: &Mat<f64>) -> Mat<f64> {
    let n = mat.nrows();
    mat.partial_piv_lu().solve(&Mat::<f64>::identity(n, n))
}

/// Inverse of a finite square matrix with condition number `cond <= CONDITION_THRESHOLD`.
///
/// `lu_inverse(m)` is tried first (production: [`partial_piv_lu_inverse`]), then
/// [`pseudoinverse`]. A candidate is returned only if it is finite and
/// `max |A X - I| <= INVERSE_RESIDUAL_FACTOR * n * f64::EPSILON * cond`; if neither
/// passes, the result is [`LinalgError::SingularMatrix`]. The bound uses `cond`
/// from the SVD, not the candidate's norm, so a wrong candidate cannot widen it.
///
/// Working set: the LU route holds at most five n x n matrices at once, counting
/// the caller's (its faer copy, the LU factor, the identity and the solution). The
/// faer copy and the LU candidate are dropped before the SVD route, so that route
/// peaks at [`pseudoinverse`]'s own working set: besides the caller's matrix, its
/// own copy of it, U, V, S^+ and the product, plus faer's SVD workspace.
fn verified_inverse(
    m: &ArrayView2<f64>,
    cond: f64,
    lu_inverse: impl FnOnce(&Mat<f64>) -> Mat<f64>,
) -> Result<Array2<f64>, LinalgError> {
    let n = m.nrows();
    let bound = INVERSE_RESIDUAL_FACTOR * n as f64 * f64::EPSILON * cond;
    let mat = ndarray_to_faer(m);
    let candidate = lu_inverse(&mat);
    if inverse_residual(&mat, &candidate) <= bound {
        return Ok(faer_to_ndarray(&candidate));
    }
    drop(candidate);
    drop(mat);
    let pinv = pseudoinverse(m)?;
    if inverse_residual(&ndarray_to_faer(m), &ndarray_to_faer(&pinv.view())) <= bound {
        return Ok(pinv);
    }
    Err(LinalgError::SingularMatrix)
}

/// Whether `x` solves `A x = b` to working precision: `x` is finite and
/// `||A x - b||_inf <= INVERSE_RESIDUAL_FACTOR * n * eps * (||A||_inf ||x||_inf + ||b||_inf)`.
fn solve_residual_ok(a: &ArrayView2<f64>, x: &Array1<f64>, b: &Array1<f64>) -> bool {
    if !x.iter().all(|v| v.is_finite()) {
        return false;
    }
    let inf_norm = |v: &Array1<f64>| v.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    let a_norm = a
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|e| e.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let r = a.dot(x) - b;
    if !r.iter().all(|v| v.is_finite()) {
        return false;
    }
    let n = a.nrows();
    let bound =
        INVERSE_RESIDUAL_FACTOR * n as f64 * f64::EPSILON * (a_norm * inf_norm(x) + inf_norm(b));
    inf_norm(&r) <= bound
}

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

/// Inverse of a square, finite, well-conditioned matrix, by verified LU.
///
/// - Non-square input: [`LinalgError::NotSquare`]. A non-finite entry:
///   [`LinalgError::NonFinite`]. An empty matrix: [`LinalgError::SingularMatrix`].
/// - `cond = condition_number(m)`; above [`CONDITION_THRESHOLD`] the result is
///   [`LinalgError::IllConditioned`]`(cond)`.
/// - Otherwise the inverse by LU with partial pivoting is returned if it is finite
///   and `max |A X - I| <= INVERSE_RESIDUAL_FACTOR * n * eps * cond`; if it is not,
///   the SVD pseudoinverse under the same check; if neither passes,
///   [`LinalgError::SingularMatrix`].
///
/// Unlike [`safe_inverse`], this never tries Cholesky. Until RF-31 it used faer
/// 0.22.6's full-pivot LU without any check, which returned all-NaN inverses for
/// well-conditioned structured matrices of size 16 or more (for example
/// `I - rho W` on a rook grid).
pub fn matrix_inverse(m: &ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }
    if !is_all_finite(m) {
        return Err(LinalgError::NonFinite);
    }

    let cond = condition_number(m)?;
    if cond > CONDITION_THRESHOLD {
        return Err(LinalgError::IllConditioned(cond));
    }

    verified_inverse(m, cond, partial_piv_lu_inverse)
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

    // Compute S^+: reciprocal of non-zero singular values. With the thin SVD,
    // U is (rows x min_dim) and V is (cols x min_dim), so S^+ must be
    // (min_dim x min_dim) for `V * S^+ * U^T` to be conformable and yield the
    // (cols x rows) pseudoinverse. Allocating (cols x rows) here only happens
    // to be conformable when the input is square.
    let mut s_inv = Mat::<f64>::zeros(min_dim, min_dim);
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

/// Inverse of a square matrix, with a pseudoinverse fallback for ill-conditioned
/// input that is not positive definite.
///
/// Returns `(inverse, None)`, or `(pseudoinverse, Some(cond))` with the condition
/// number as a warning.
///
/// 1. Non-square input: [`LinalgError::NotSquare`].
/// 2. If the Cholesky factorisation succeeds (symmetric positive definite input,
///    such as X'X or an information matrix), the Cholesky inverse is returned with
///    `None`. Cholesky reads only the lower triangle, so the upper triangle of the
///    input is not checked on this path (REG-59, T3-WP01).
/// 3. Otherwise:
///    - a non-finite entry gives [`LinalgError::NonFinite`];
///    - `cond = condition_number(m)`; if `cond > CONDITION_THRESHOLD`, the result
///      is `(pseudoinverse(m), Some(cond))`;
///    - else the inverse by LU with partial pivoting is returned with `None` if it
///      is finite and `max |A X - I| <= INVERSE_RESIDUAL_FACTOR * n * eps * cond`;
///      if it is not, the SVD pseudoinverse under the same check; if neither
///      passes, [`LinalgError::SingularMatrix`].
///
/// Why partial pivoting, and why every LU result is checked: faer 0.22.6's
/// `FullPivLu`, which this fallback used until RF-31, returned wrong or all-NaN
/// inverses for well-conditioned structured matrices of size 16 or more
/// (diagonal, tridiagonal, multinomial and ordered logit Hessians), and nothing
/// checked the result.
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
    drop(mat);

    // Cholesky failed: not positive definite (or not finite).
    if !is_all_finite(m) {
        return Err(LinalgError::NonFinite);
    }
    let cond = condition_number(m)?;
    if cond > CONDITION_THRESHOLD {
        // Ill-conditioned: use pseudoinverse and return warning
        let pinv = pseudoinverse(m)?;
        return Ok((pinv, Some(cond)));
    }

    Ok((verified_inverse(m, cond, partial_piv_lu_inverse)?, None))
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

/// Solve the linear system `A x = b` by verified LU.
///
/// - Non-square `a`: [`LinalgError::NotSquare`]; `b` of another length:
///   [`LinalgError::DimensionMismatch`]; a non-finite entry in `a` or `b`:
///   [`LinalgError::NonFinite`].
/// - The solution by LU with partial pivoting is returned if it is finite and
///   `||A x - b||_inf <= INVERSE_RESIDUAL_FACTOR * n * eps * (||A||_inf ||x||_inf + ||b||_inf)`
///   (a normwise backward error of at most 100 n eps).
/// - Otherwise, if `condition_number(a) > CONDITION_THRESHOLD`, the matrix is treated
///   as singular ([`LinalgError::SingularMatrix`]); else `pseudoinverse(a) b` is
///   returned under the same check, or [`LinalgError::SingularMatrix`].
///
/// Until RF-31 this used faer 0.22.6's full-pivot LU without any check: it returned
/// NaN for well-conditioned structured systems of size 16 or more, and
/// `Ok([-inf, inf])` for a singular `A`.
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
    if !is_all_finite(a) || !b.iter().all(|v| v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }

    let mat_a = ndarray_to_faer(a);
    let col_b = Col::<f64>::from_fn(b.len(), |i| b[i]);
    let x = faer_col_to_ndarray(&mat_a.partial_piv_lu().solve(&col_b));
    drop(mat_a);
    if solve_residual_ok(a, &x, b) {
        return Ok(x);
    }

    let cond = condition_number(a)?;
    if cond > CONDITION_THRESHOLD {
        return Err(LinalgError::SingularMatrix);
    }
    let x = pseudoinverse(a)?.dot(b);
    if solve_residual_ok(a, &x, b) {
        Ok(x)
    } else {
        Err(LinalgError::SingularMatrix)
    }
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

    /// A * A^+ * A == A (Moore-Penrose property) for a wide matrix.
    #[test]
    fn test_pseudoinverse_wide() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let pinv = pseudoinverse(&a.view()).unwrap();
        assert_eq!(pinv.dim(), (3, 2));
        let reconstructed = a.dot(&pinv).dot(&a);
        for (orig, recon) in a.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-9,
                "A*A+*A != A: {orig} vs {recon}"
            );
        }
    }

    /// A * A^+ * A == A (Moore-Penrose property) for a tall matrix.
    #[test]
    fn test_pseudoinverse_tall() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2
        let pinv = pseudoinverse(&a.view()).unwrap();
        assert_eq!(pinv.dim(), (2, 3));
        let reconstructed = a.dot(&pinv).dot(&a);
        for (orig, recon) in a.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-9,
                "A*A+*A != A: {orig} vs {recon}"
            );
        }
    }

    /// For a square invertible matrix the pseudoinverse equals the inverse.
    #[test]
    fn test_pseudoinverse_square_matches_inverse() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let pinv = pseudoinverse(&a.view()).unwrap();
        let inv = matrix_inverse(&a.view()).unwrap();
        for (p, i) in pinv.iter().zip(inv.iter()) {
            assert!((p - i).abs() < 1e-9, "pinv != inv: {p} vs {i}");
        }
    }

    // ==========================================================================
    // RF-31: the LU fallback of safe_inverse, matrix_inverse and solve
    // ==========================================================================

    /// max_ij |(A X - I)_ij|; +inf if any entry of A X is NaN (so a NaN inverse
    /// can never pass a residual bound through `f64::max` dropping NaN).
    fn max_abs_residual(a: &Array2<f64>, x: &Array2<f64>) -> f64 {
        let n = a.nrows();
        let p = a.dot(x);
        let mut r = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let d = (p[[i, j]] - if i == j { 1.0 } else { 0.0 }).abs();
                if d.is_nan() {
                    return f64::INFINITY;
                }
                r = r.max(d);
            }
        }
        r
    }

    fn diag_from(n: usize, sign: f64) -> Array2<f64> {
        Array2::from_shape_fn(
            (n, n),
            |(i, j)| {
                if i == j { sign * (i as f64 + 1.0) } else { 0.0 }
            },
        )
    }

    /// -tridiag(-1, 3, -1): negative definite, so Cholesky fails and the LU fallback runs.
    fn neg_tridiag(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                -3.0
            } else if i.abs_diff(j) == 1 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// I - 0.5 W for row-standardised rook contiguity on a g x g grid (cell i = r * g + c).
    fn rook_i_minus_half_w(g: usize) -> Array2<f64> {
        let n = g * g;
        let mut a = Array2::<f64>::eye(n);
        for r in 0..g {
            for c in 0..g {
                let i = r * g + c;
                let mut nb = Vec::new();
                if r > 0 {
                    nb.push(i - g);
                }
                if r + 1 < g {
                    nb.push(i + g);
                }
                if c > 0 {
                    nb.push(i - 1);
                }
                if c + 1 < g {
                    nb.push(i + 1);
                }
                for &j in &nb {
                    a[[i, j]] -= 0.5 / nb.len() as f64;
                }
            }
        }
        a
    }

    /// The sweep's generator: a 64-bit LCG (Knuth's MMIX constants), u in [0, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next_u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_safe_inverse_lu_fallback_negative_diagonal() {
        for n in [15, 16, 32] {
            let (inv, w) = safe_inverse(&diag_from(n, -1.0).view()).unwrap();
            assert_eq!(w, None, "n = {n}");
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { -1.0 / (i as f64 + 1.0) } else { 0.0 };
                    assert_eq!(inv[[i, j]], want, "n = {n}, [{i},{j}]");
                }
            }
        }
    }

    #[test]
    fn test_safe_inverse_lu_fallback_tridiagonal() {
        for n in [15, 16, 30, 100] {
            let a = neg_tridiag(n);
            let (inv, w) = safe_inverse(&a.view()).unwrap();
            assert_eq!(w, None, "n = {n}");
            let r = max_abs_residual(&a, &inv);
            assert!(r <= 1e-14, "n = {n}: max |A X - I| = {r}");
            // R 4.3.3: solve(-tridiag(-1, 3, -1)) (1-based indices in R)
            let pins: &[(usize, usize, f64)] = match n {
                16 => &[
                    (0, 0, -0.3819660112500914),
                    (0, 15, -1.7534978336411016e-07),
                    (7, 7, -0.44721349029009344),
                ],
                30 => &[
                    (0, 0, -0.38196601125010515),
                    (0, 29, -2.467466736149682e-13),
                    (14, 14, -0.44721359549980993),
                ],
                _ => &[],
            };
            for &(i, j, want) in pins {
                assert!(
                    (inv[[i, j]] - want).abs() <= 1e-15,
                    "n = {n}, [{i},{j}]: {} vs R {want}",
                    inv[[i, j]]
                );
            }
        }
    }

    /// The shape class of RF-31: sparse symmetric matrices with a negative diagonal,
    /// sizes on both sides of 16.
    #[test]
    fn test_safe_inverse_lu_fallback_sparse_sweep() {
        let mut rng = Lcg(42);
        for n in [16, 17, 20, 31, 32, 33, 50] {
            for rep in 0..4 {
                let mut x = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        if (7 * i + 3 * j) % 5 == 0 {
                            x[[i, j]] = 2.0 * rng.next_u() - 1.0;
                        }
                    }
                }
                let mut m = x.t().dot(&x);
                for i in 0..n {
                    m[[i, i]] = -(m[[i, i]] + 1.0);
                }
                let cond = condition_number(&m.view()).unwrap();
                let (inv, w) = safe_inverse(&m.view()).unwrap();
                assert_eq!(w, None, "n = {n}, rep {rep}: cond {cond}");
                assert!(
                    inv.iter().all(|v| v.is_finite()),
                    "n = {n}, rep {rep}: non-finite inverse"
                );
                let r = max_abs_residual(&m, &inv);
                let bound = 100.0 * n as f64 * f64::EPSILON * cond;
                assert!(
                    r <= bound,
                    "n = {n}, rep {rep}: max |A X - I| = {r} > {bound} (cond {cond})"
                );
            }
        }
    }

    #[test]
    fn test_matrix_inverse_structured() {
        // diag(1..16) is positive definite, but matrix_inverse never tried Cholesky.
        let inv = matrix_inverse(&diag_from(16, 1.0).view()).unwrap();
        for i in 0..16 {
            for j in 0..16 {
                let want = if i == j { 1.0 / (i as f64 + 1.0) } else { 0.0 };
                assert_eq!(inv[[i, j]], want, "[{i},{j}]");
            }
        }
        // R 4.3.3: solve(I - 0.5 W) for the rook grid on 5 x 5.
        let inv = matrix_inverse(&rook_i_minus_half_w(5).view()).unwrap();
        assert!(
            (inv[[0, 0]] - 1.098953394019214).abs() <= 1e-15,
            "{}",
            inv[[0, 0]]
        );
        assert!(
            (inv[[12, 12]] - 1.0739348370927317).abs() <= 1e-15,
            "{}",
            inv[[12, 12]]
        );
        for i in 0..25 {
            let s: f64 = inv.row(i).sum();
            assert!((s - 2.0).abs() <= 1e-14, "row {i} sums to {s}");
        }
    }

    #[test]
    fn test_solve_structured_and_singular() {
        // R 4.3.3: solve(A, rep(1, 25)) is 2 in every entry (W is row-standardised).
        let x = solve(&rook_i_minus_half_w(5).view(), &Array1::ones(25)).unwrap();
        for (i, v) in x.iter().enumerate() {
            assert!((v - 2.0).abs() <= 1e-14, "x[{i}] = {v}");
        }
        let x = solve(&diag_from(16, 1.0).view(), &Array1::ones(16)).unwrap();
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 1.0 / (i as f64 + 1.0), "x[{i}]");
        }
        let singular = array![[1.0, 2.0], [2.0, 4.0]];
        let r = solve(&singular.view(), &array![1.0, 3.0]);
        assert!(matches!(r, Err(LinalgError::SingularMatrix)), "{r:?}");
    }

    #[test]
    fn test_linalg_rejects_non_finite() {
        let nan_off = array![[1.0, f64::NAN], [f64::NAN, 1.0]];
        let r = safe_inverse(&nan_off.view());
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
        let r = matrix_inverse(&nan_off.view());
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
        let inf_diag = array![[f64::INFINITY, 0.0], [0.0, 1.0]];
        let r = safe_inverse(&inf_diag.view());
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
        let nan_neg = array![[-1.0, f64::NAN], [f64::NAN, -1.0]];
        let r = safe_inverse(&nan_neg.view());
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
        let a = array![[3.0, 1.0], [1.0, 2.0]];
        let r = solve(&a.view(), &array![f64::NAN, 1.0]);
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
        let a = array![[f64::NAN, 1.0], [1.0, 2.0]];
        let r = solve(&a.view(), &array![1.0, 1.0]);
        assert!(matches!(r, Err(LinalgError::NonFinite)), "{r:?}");
    }

    /// Guard: symmetric positive definite input keeps the Cholesky bits.
    #[test]
    fn test_safe_inverse_spd_keeps_cholesky_path() {
        let m = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let mut rng = Lcg(42);
        let x = Array2::from_shape_fn((20, 20), |_| 2.0 * rng.next_u() - 1.0);
        let gram = x.t().dot(&x) + Array2::<f64>::eye(20);
        for m in [m, gram] {
            let (inv, w) = safe_inverse(&m.view()).unwrap();
            assert_eq!(w, None);
            assert_eq!(inv, cholesky_inverse(&m.view()).unwrap());
        }
    }

    /// Guard: a non-positive-definite matrix with cond > 1e10 keeps the pseudoinverse.
    #[test]
    fn test_safe_inverse_ill_conditioned_keeps_pseudoinverse() {
        let m = array![[-1.0, -1.0], [-1.0, -1.0 - 1e-12]];
        let cond = condition_number(&m.view()).unwrap();
        assert!(cond > CONDITION_THRESHOLD, "{cond}");
        let (inv, w) = safe_inverse(&m.view()).unwrap();
        assert_eq!(w, Some(cond));
        assert_eq!(inv, pseudoinverse(&m.view()).unwrap());
    }

    #[test]
    fn test_verified_inverse_rejects_bad_candidates() {
        let m = diag_from(16, -1.0);
        let cond = condition_number(&m.view()).unwrap();
        let svd = pseudoinverse(&m.view()).unwrap();
        // (1) faer 0.22.6's full-pivot candidate (all NaN for this matrix) is rejected.
        let inv = verified_inverse(&m.view(), cond, |a| {
            a.full_piv_lu().solve(&Mat::<f64>::identity(16, 16))
        })
        .unwrap();
        assert!(max_abs_residual(&m, &inv) <= 1e-15);
        assert_eq!(inv, svd, "expected the SVD route");
        // (2) A finite candidate that is wrong by 1e-9 relative in one entry is rejected.
        let inv = verified_inverse(&m.view(), cond, |a| {
            let mut x = partial_piv_lu_inverse(a);
            x[(0, 0)] *= 1.0 + 1e-9;
            x
        })
        .unwrap();
        assert!((inv[[0, 0]] + 1.0).abs() <= 1e-15, "{}", inv[[0, 0]]);
        assert_eq!(inv, svd, "expected the SVD route");
        // (3) The correct partial-pivot candidate is returned bit for bit.
        let lu = faer_to_ndarray(&partial_piv_lu_inverse(&ndarray_to_faer(&m.view())));
        let inv = verified_inverse(&m.view(), cond, partial_piv_lu_inverse).unwrap();
        assert_eq!(inv, lu);
    }
}
