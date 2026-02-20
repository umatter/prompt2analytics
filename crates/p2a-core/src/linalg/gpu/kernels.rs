//! Custom GPU-accelerated operations.
//!
//! Operations that don't map directly to a single cuBLAS/cuSOLVER call
//! but benefit from GPU acceleration through reformulation.

use std::os::raw::c_int;

use cudarc::cublas::result::CublasError;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use ndarray::{Array2, ArrayView2};

use super::context::GpuContext;
use super::memory::{array2_to_device, device_to_array2};

/// GPU error type for kernel operations.
#[derive(Debug)]
pub enum GpuKernelError {
    Driver(cudarc::driver::DriverError),
    Cublas(CublasError),
}

impl From<cudarc::driver::DriverError> for GpuKernelError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CublasError> for GpuKernelError {
    fn from(e: CublasError) -> Self {
        Self::Cublas(e)
    }
}

/// Compute pairwise squared Euclidean distances using GEMM reformulation.
///
/// For K-means clustering, we need ||x_i - c_j||^2 for all points x_i
/// and centroids c_j. This can be reformulated as:
///
///   D_ij = ||x_i||^2 + ||c_j||^2 - 2 * x_i . c_j
///
/// The cross term -2 * X * C^T is a DGEMM, and the squared norms are
/// cheap to compute. This is much faster than point-by-point distance
/// computation for large n.
///
/// # Arguments
/// * `data` - (n x d) data matrix
/// * `centroids` - (k x d) centroid matrix
///
/// # Returns
/// (n x k) distance matrix where D[i,j] = ||data[i] - centroids[j]||^2
pub fn pairwise_distances_gpu(
    ctx: &GpuContext,
    data: &ArrayView2<f64>,
    centroids: &ArrayView2<f64>,
) -> Result<Array2<f64>, GpuKernelError> {
    let (n, d) = data.dim();
    let (k, _d2) = centroids.dim();

    // Compute squared norms on CPU (cheap: O(n*d) and O(k*d))
    let data_sq_norms: Vec<f64> = (0..n).map(|i| data.row(i).dot(&data.row(i))).collect();
    let cent_sq_norms: Vec<f64> = (0..k)
        .map(|j| centroids.row(j).dot(&centroids.row(j)))
        .collect();

    // Compute cross term: -2 * data * centroids^T via GPU DGEMM
    // data is (n x d) row-major → cuBLAS sees data^T (d x n), lda=d
    // centroids is (k x d) row-major → cuBLAS sees centroids^T (d x k), lda=d
    //
    // We want: result = data * centroids^T (n x k)
    // In cuBLAS: result^T = centroids * data^T (k x n)
    // DGEMM: transa=N (centroids^T as-is: d x k), transb=N (data^T as-is: d x n)
    // Wait, that gives (d x n)... Let me reconsider.
    //
    // cuBLAS sees:
    //   data_raw as col-major: data^T (d x n), lda = d
    //   centroids_raw as col-major: centroids^T (d x k), lda = d
    //
    // We want: C = data * centroids^T (n x k)
    // C^T = centroids * data^T (k x n)
    // In cuBLAS: "A" = centroids_raw (cuBLAS sees cent^T, d x k)
    //            "B" = data_raw (cuBLAS sees data^T, d x n)
    //   op(A) = (cent^T)^T = cent (k x d) → transa=T
    //   op(B) = data^T (d x n) → transb=N
    //   C_cublas = cent * data^T (k x n) in col-major = (n x k) in row-major ✓
    let dev_data = array2_to_device(&ctx.stream, data)?;
    let dev_cent = array2_to_device(&ctx.stream, centroids)?;
    let mut dev_cross = ctx.stream.alloc_zeros::<f64>(n * k)?;

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: k as c_int,   // rows of op(A) = cent (k x d)
        n: n as c_int,   // cols of op(B) = data^T (d x n)
        k: d as c_int,   // inner dimension
        alpha: -2.0_f64, // we want -2 * X * C^T
        lda: d as c_int, // leading dim of cent_raw in col-major
        ldb: d as c_int, // leading dim of data_raw in col-major
        beta: 0.0_f64,
        ldc: k as c_int, // leading dim of result
    };

    unsafe {
        ctx.blas.gemm(cfg, &dev_cent, &dev_data, &mut dev_cross)?;
    }

    // Copy cross term back and add squared norms
    let mut distances = device_to_array2(&ctx.stream, &dev_cross, n, k)?;

    // D_ij = ||x_i||^2 + ||c_j||^2 - 2 * x_i . c_j
    for i in 0..n {
        for j in 0..k {
            distances[[i, j]] += data_sq_norms[i] + cent_sq_norms[j];
            // Clamp to avoid negative distances from floating point errors
            if distances[[i, j]] < 0.0 {
                distances[[i, j]] = 0.0;
            }
        }
    }

    Ok(distances)
}
