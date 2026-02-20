//! GPU dispatch thresholds.
//!
//! Controls when operations are dispatched to GPU vs CPU based on problem size.
//! Thresholds are configurable via environment variables.

use std::sync::OnceLock;

/// Minimum thresholds for GPU dispatch.
///
/// Below these thresholds, the CPU path is faster due to transfer overhead.
/// Thresholds are calibrated on DGX Spark (Grace Blackwell, unified memory)
/// from systematic benchmarking at matrix sizes up to 1M x 200.
///
/// Key findings from benchmarks:
/// - xtx (X'X): GPU wins when k >= ~30 (2-5x speedup for k=50-200). For
///   k=10 the CPU path is faster even at n=1M. Dispatch on n*k² with a
///   minimum k guard.
/// - xty (X'y): bandwidth-bound O(n*k) DGEMV; GPU is consistently slower
///   or break-even across all tested sizes. Disabled by default.
/// - matmul: GPU helps for square matrices (2x at 2000³) but hurts badly
///   for tall-skinny shapes (100Kx50x50 is 6x slower). Needs shape guard.
/// - PCA: biggest GPU win (3.5-10x speedup) via covariance eigendecomposition
///   instead of full SVD.
/// - K-means: GPU helps for d >= 20 (1.4-2.2x) but hurts for d=10.
#[derive(Debug, Clone)]
pub struct GpuThresholds {
    /// Minimum n*k² (compute intensity) for xtx (X'X) via cuBLAS DGEMM.
    /// Also requires k >= `xtx_min_k`. Default: 20_000_000.
    pub xtx_min_nkk: usize,
    /// Minimum k (columns) for xtx GPU dispatch.
    /// GPU hurts for k <= ~20 even at very large n. Default: 30.
    pub xtx_min_k: usize,
    /// Minimum n (rows) for xty (X'y) via cuBLAS DGEMV.
    /// Disabled by default: set very high because GPU DGEMV is consistently
    /// slower than CPU OpenBLAS on Grace Blackwell. Default: usize::MAX.
    pub xty_min_n: usize,
    /// Minimum k for safe_inverse via cuSOLVER DPOTRF+DPOTRI.
    /// Default: 100.
    pub inverse_min_k: usize,
    /// Minimum m*n*k for matmul via cuBLAS DGEMM.
    /// Default: 1_000_000.
    pub matmul_min_mnk: usize,
    /// Minimum dimension ratio (min_dim / max_dim) for matmul GPU dispatch.
    /// Prevents GPU dispatch for tall-skinny shapes where CPU is much faster.
    /// 0.0 = any shape, 1.0 = only square. Default: 0.1.
    pub matmul_min_shape_ratio: f64,
    /// Minimum n for K-means distance computation via DGEMM.
    /// Default: 10_000.
    pub kmeans_min_n: usize,
    /// Minimum d (dimensions) for K-means GPU dispatch.
    /// GPU hurts for d <= ~15. Default: 20.
    pub kmeans_min_d: usize,
    /// Minimum bootstrap replications to use batched GPU path.
    pub bootstrap_min_b: usize,
}

impl Default for GpuThresholds {
    fn default() -> Self {
        Self {
            xtx_min_nkk: 20_000_000,
            xtx_min_k: 30,
            xty_min_n: usize::MAX, // disabled: GPU DGEMV slower than CPU on Grace Blackwell
            inverse_min_k: 100,
            matmul_min_mnk: 1_000_000,
            matmul_min_shape_ratio: 0.1,
            kmeans_min_n: 10_000,
            kmeans_min_d: 20,
            bootstrap_min_b: 100,
        }
    }
}

impl GpuThresholds {
    /// Load thresholds from environment variables, falling back to defaults.
    ///
    /// Supported env vars:
    /// - `P2A_GPU_XTX_MIN_NKK` (default: 20000000)
    /// - `P2A_GPU_XTX_MIN_K` (default: 30)
    /// - `P2A_GPU_XTY_MIN_N` (default: MAX, i.e. disabled)
    /// - `P2A_GPU_INVERSE_MIN_K` (default: 100)
    /// - `P2A_GPU_MATMUL_MIN_MNK` (default: 1000000)
    /// - `P2A_GPU_MATMUL_MIN_SHAPE_RATIO` (default: 0.1)
    /// - `P2A_GPU_KMEANS_MIN_N` (default: 10000)
    /// - `P2A_GPU_KMEANS_MIN_D` (default: 20)
    /// - `P2A_GPU_BOOTSTRAP_MIN_B` (default: 100)
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            xtx_min_nkk: parse_env("P2A_GPU_XTX_MIN_NKK", defaults.xtx_min_nkk),
            xtx_min_k: parse_env("P2A_GPU_XTX_MIN_K", defaults.xtx_min_k),
            xty_min_n: parse_env("P2A_GPU_XTY_MIN_N", defaults.xty_min_n),
            inverse_min_k: parse_env("P2A_GPU_INVERSE_MIN_K", defaults.inverse_min_k),
            matmul_min_mnk: parse_env("P2A_GPU_MATMUL_MIN_MNK", defaults.matmul_min_mnk),
            matmul_min_shape_ratio: parse_env_f64(
                "P2A_GPU_MATMUL_MIN_SHAPE_RATIO",
                defaults.matmul_min_shape_ratio,
            ),
            kmeans_min_n: parse_env("P2A_GPU_KMEANS_MIN_N", defaults.kmeans_min_n),
            kmeans_min_d: parse_env("P2A_GPU_KMEANS_MIN_D", defaults.kmeans_min_d),
            bootstrap_min_b: parse_env("P2A_GPU_BOOTSTRAP_MIN_B", defaults.bootstrap_min_b),
        }
    }
}

fn parse_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_env_f64(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

static THRESHOLDS: OnceLock<GpuThresholds> = OnceLock::new();

/// Get the global GPU dispatch thresholds (initialized once from env vars).
pub fn thresholds() -> &'static GpuThresholds {
    THRESHOLDS.get_or_init(GpuThresholds::from_env)
}
