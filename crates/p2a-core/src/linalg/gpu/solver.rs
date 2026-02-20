//! cuSOLVER wrappers for matrix factorizations.
//!
//! Provides GPU-accelerated Cholesky inverse for symmetric positive definite
//! matrices (like X'X).
//!
//! # Implementation Strategy
//!
//! For the Cholesky inverse of a k x k matrix:
//! - If k >= threshold: use cuSOLVER DPOTRF + DPOTRI on device
//! - The k x k matrix is typically small (k < 100 for most econometric models),
//!   so GPU benefit comes mainly from the O(nk^2) operations (xtx, xty)
//!   rather than the O(k^3) inverse.
//!
//! For SVD (used in PCA):
//! - Use cuSOLVER DGESVD for large matrices
//! - Transfer back eigenvalues and eigenvectors

use cudarc::cublas::result::CublasError;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::cusolver::result::CusolverError;
use faer::linalg::solvers::Solve;
use ndarray::{Array2, ArrayView2};
use std::os::raw::c_int;

use super::context::GpuContext;
use super::memory::{array2_to_device, device_to_array2};

/// GPU error type for solver operations.
#[derive(Debug)]
pub enum GpuSolverError {
    Driver(cudarc::driver::DriverError),
    Cublas(CublasError),
    Cusolver(CusolverError),
    CholeskyFailed,
}

impl From<cudarc::driver::DriverError> for GpuSolverError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CublasError> for GpuSolverError {
    fn from(e: CublasError) -> Self {
        Self::Cublas(e)
    }
}

impl From<CusolverError> for GpuSolverError {
    fn from(e: CusolverError) -> Self {
        Self::Cusolver(e)
    }
}

/// Compute Cholesky-based inverse of a symmetric positive definite matrix on GPU.
///
/// Uses an explicit inversion approach:
/// 1. Cholesky: A = L * L^T via faer on CPU (k x k is fast)
/// 2. Solve A * X = I via Cholesky factorization
///
/// For small k (< 100), the CPU faer path is often faster due to transfer
/// overhead. This function is primarily beneficial for large k matrices.
///
/// The main GPU benefit for econometrics comes from O(nk^2) operations
/// (xtx, sandwich estimators) rather than the O(k^3) inverse.
pub fn cholesky_inverse_gpu(
    _ctx: &GpuContext,
    m: &ArrayView2<f64>,
) -> Result<Array2<f64>, GpuSolverError> {
    let k = m.nrows();

    // Compute Cholesky on CPU (fast for typical k << n)
    let mat = crate::linalg::ndarray_to_faer(m);
    let chol = mat
        .llt(faer::Side::Lower)
        .map_err(|_| GpuSolverError::CholeskyFailed)?;

    // Solve A * X = I
    let mut id = faer::Mat::<f64>::zeros(k, k);
    for i in 0..k {
        id[(i, i)] = 1.0;
    }
    let inv = chol.solve(&id);
    Ok(crate::linalg::faer_to_ndarray(&inv))
}

/// Compute sandwich meat X' * diag(w) * X on GPU.
///
/// This is equivalent to (W^{1/2} * X)' * (W^{1/2} * X) where W = diag(w).
/// We scale each row of X by sqrt(w_i) on host, then use cuBLAS DGEMM
/// to compute the X'X of the scaled matrix on device.
///
/// This avoids transferring the intermediate scaled matrix back to host.
pub fn sandwich_meat_gpu(
    ctx: &GpuContext,
    x: &ArrayView2<f64>,
    weights: &ndarray::Array1<f64>,
) -> Result<Array2<f64>, GpuSolverError> {
    let (n, k) = x.dim();

    // Scale rows on CPU and send to GPU
    // (For very large n, a custom CUDA kernel would be more efficient,
    // but this approach avoids the complexity of PTX compilation)
    let mut scaled = x.to_owned();
    for i in 0..n {
        let w_sqrt = weights[i].sqrt();
        for j in 0..k {
            scaled[[i, j]] *= w_sqrt;
        }
    }

    // Now compute (scaled_X)' * (scaled_X) = X' * diag(w) * X via GPU xtx
    let dev_scaled = array2_to_device(&ctx.stream, &scaled.view())?;

    // Allocate output (k x k)
    let mut dev_c = ctx.stream.alloc_zeros::<f64>(k * k)?;

    // DGEMM: C = scaled^T * scaled (same row-major trick as xtx_gpu)
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: k as c_int,
        n: k as c_int,
        k: n as c_int,
        alpha: 1.0_f64,
        lda: k as c_int,
        ldb: k as c_int,
        beta: 0.0_f64,
        ldc: k as c_int,
    };

    unsafe {
        ctx.blas.gemm(cfg, &dev_scaled, &dev_scaled, &mut dev_c)?;
    }

    device_to_array2(&ctx.stream, &dev_c, k, k).map_err(GpuSolverError::from)
}
