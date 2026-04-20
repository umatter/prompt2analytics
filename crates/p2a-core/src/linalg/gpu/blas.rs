//! cuBLAS wrappers for core matrix operations.
//!
//! Implements GPU-accelerated versions of `xtx`, `xty`, and `matmul`
//! using cuBLAS DGEMM and DGEMV.
//!
//! # Row-Major Convention
//!
//! ndarray uses row-major storage. When passed to cuBLAS (column-major),
//! a row-major (n x k) matrix appears as its transpose (k x n) in
//! column-major. We exploit this to avoid data reordering:
//!
//! - **xtx (X'X)**: cuBLAS sees A = X^T (k x n). We compute A * A^T via
//!   DGEMM with transa=N, transb=T, yielding X^T * X = X'X (k x k).
//!
//! - **xty (X'y)**: cuBLAS sees A = X^T (k x n). We compute A * y via
//!   DGEMV with trans=N, yielding X^T * y = X'y (k x 1).
//!
//! - **matmul (A*B)**: We compute C^T = B^T * A^T via DGEMM, which gives
//!   C = A*B in row-major storage.

use std::os::raw::c_int;

use cudarc::cublas::result::CublasError;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig, Gemv, GemvConfig};
use ndarray::{Array1, Array2, ArrayView2};

use super::context::GpuContext;
use super::memory::{array1_to_device, array2_to_device, device_to_array1, device_to_array2};

/// GPU error type for BLAS operations.
#[derive(Debug)]
pub enum GpuBlasError {
    Driver(cudarc::driver::DriverError),
    Cublas(CublasError),
}

impl From<cudarc::driver::DriverError> for GpuBlasError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CublasError> for GpuBlasError {
    fn from(e: CublasError) -> Self {
        Self::Cublas(e)
    }
}

/// Compute X'X on GPU via cuBLAS DGEMM.
///
/// Input: X is (n x k) in row-major.
/// cuBLAS sees: A = X^T (k x n) in column-major with lda = k.
/// We compute: C = A * A^T = X^T * X (k x k).
/// DGEMM params: transa=N, transb=T, m=k, n=k, k_inner=n.
///
/// Result is (k x k) symmetric, stored row-major (same as column-major for symmetric).
pub fn xtx_gpu(ctx: &GpuContext, x: &ArrayView2<f64>) -> Result<Array2<f64>, GpuBlasError> {
    let (n, k) = x.dim();

    // Copy X to device
    let dev_x = array2_to_device(&ctx.stream, x)?;

    // Allocate output (k x k)
    let mut dev_c = ctx.stream.alloc_zeros::<f64>(k * k)?;

    // cuBLAS DGEMM: C = alpha * op(A) * op(B) + beta * C
    // A = X^T (k x n), B = X^T (k x n), we want A * B^T = X^T * X
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

    // SAFETY: `dev_x` and `dev_c` are owned device buffers on `ctx.stream`,
    // their sizes match the cfg's leading dimensions (lda = ldb = ldc = k,
    // k_inner = n), and alpha/beta are finite. The cuBLAS handle is held
    // exclusively for the duration of the call via the Mutex guard.
    unsafe {
        ctx.blas().gemm(cfg, &dev_x, &dev_x, &mut dev_c)?;
    }

    // Copy result back
    device_to_array2(&ctx.stream, &dev_c, k, k).map_err(GpuBlasError::from)
}

/// Compute X'y on GPU via cuBLAS DGEMV.
///
/// Input: X is (n x k) in row-major, y is (n,).
/// cuBLAS sees: A = X^T (k x n) in column-major with lda = k.
/// We compute: result = A * y = X^T * y (k,).
/// DGEMV params: trans=N, m=k, n=n.
pub fn xty_gpu(
    ctx: &GpuContext,
    x: &ArrayView2<f64>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, GpuBlasError> {
    let (n, k) = x.dim();

    // Copy to device
    let dev_x = array2_to_device(&ctx.stream, x)?;
    let dev_y = array1_to_device(&ctx.stream, &y.view())?;

    // Allocate output (k,)
    let mut dev_result = ctx.stream.alloc_zeros::<f64>(k)?;

    // cuBLAS DGEMV: y_out = alpha * op(A) * x_in + beta * y_out
    // A = X^T (k x n), x_in = y (n,), result is (k,)
    let cfg = GemvConfig {
        trans: cublasOperation_t::CUBLAS_OP_N,
        m: k as c_int,
        n: n as c_int,
        alpha: 1.0_f64,
        lda: k as c_int,
        incx: 1,
        beta: 0.0_f64,
        incy: 1,
    };

    // SAFETY: `dev_x` is a (k * n) device buffer with leading dimension k,
    // `dev_y` has length n, `dev_result` has length k, and increments are 1.
    // The cuBLAS handle is locked exclusively via `ctx.blas()`.
    unsafe {
        ctx.blas().gemv(cfg, &dev_x, &dev_y, &mut dev_result)?;
    }

    device_to_array1(&ctx.stream, &dev_result, k).map_err(GpuBlasError::from)
}

/// General matrix multiplication C = A * B on GPU via cuBLAS DGEMM.
///
/// Input: A is (m x p) row-major, B is (p x n_out) row-major.
/// cuBLAS sees: A_cublas = A^T (p x m), B_cublas = B^T (n_out x p).
/// We compute: C^T = B^T * A^T (n_out x m), which is C = A*B in row-major.
/// DGEMM params: transa=N, transb=N, m=n_out, n=m_rows, k=p.
pub fn matmul_gpu(
    ctx: &GpuContext,
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
) -> Result<Array2<f64>, GpuBlasError> {
    let (m_rows, p) = a.dim();
    let (_p2, n_out) = b.dim();

    // Copy to device
    let dev_a = array2_to_device(&ctx.stream, a)?;
    let dev_b = array2_to_device(&ctx.stream, b)?;

    // Allocate output (m_rows x n_out)
    let mut dev_c = ctx.stream.alloc_zeros::<f64>(m_rows * n_out)?;

    // cuBLAS DGEMM: C = alpha * op(A) * op(B) + beta * C
    // We compute C^T = B^T * A^T (in cuBLAS's column-major world)
    // "A" for cuBLAS = B_raw (cuBLAS sees B^T, n_out x p), no trans
    // "B" for cuBLAS = A_raw (cuBLAS sees A^T, p x m_rows), no trans
    // Result: (n_out x m_rows) col-major = (m_rows x n_out) row-major
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n_out as c_int,
        n: m_rows as c_int,
        k: p as c_int,
        alpha: 1.0_f64,
        lda: n_out as c_int, // leading dim of B^T (n_out x p)
        ldb: p as c_int,     // leading dim of A^T (p x m_rows)
        beta: 0.0_f64,
        ldc: n_out as c_int, // leading dim of result
    };

    // SAFETY: The row-major -> column-major transpose trick (see module
    // doc) means cuBLAS sees `dev_b` as (n_out x p) with lda = n_out and
    // `dev_a` as (p x m_rows) with ldb = p; `dev_c` has ldc = n_out and
    // is a `m_rows * n_out` buffer. All three buffers live on the same
    // stream and the cuBLAS handle is locked via `ctx.blas()`.
    unsafe {
        ctx.blas().gemm(cfg, &dev_b, &dev_a, &mut dev_c)?;
    }

    device_to_array2(&ctx.stream, &dev_c, m_rows, n_out).map_err(GpuBlasError::from)
}
