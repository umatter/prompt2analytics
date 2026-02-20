//! Optional GPU acceleration via CUDA.
//!
//! This module provides GPU-accelerated versions of core linear algebra
//! operations using cuBLAS (GEMM, GEMV) and cuSOLVER (Cholesky, SVD).
//!
//! # Architecture
//!
//! GPU dispatch is transparent: the public API of `xtx()`, `safe_inverse()`,
//! etc. remains unchanged. GPU paths are selected automatically when:
//! 1. The `cuda` feature is enabled at compile time
//! 2. A CUDA device is detected at runtime
//! 3. The problem size exceeds configurable thresholds
//!
//! # Singleton Context
//!
//! A single `GpuContext` is initialized lazily on first use via `OnceLock`.
//! If no GPU is available, `GpuContext::get()` returns `None` and all
//! operations transparently fall back to the CPU path.
//!
//! # Thresholds
//!
//! GPU dispatch thresholds can be configured via environment variables:
//! - `P2A_GPU_XTX_MIN_NKK` (default: 20000000) — min n*k² for xtx
//! - `P2A_GPU_XTX_MIN_K` (default: 30) — min k for xtx GPU dispatch
//! - `P2A_GPU_XTY_MIN_N` (default: MAX, disabled) — min n for xty
//! - `P2A_GPU_INVERSE_MIN_K` (default: 100)
//! - `P2A_GPU_MATMUL_MIN_MNK` (default: 1000000)
//! - `P2A_GPU_MATMUL_MIN_SHAPE_RATIO` (default: 0.1)
//! - `P2A_GPU_KMEANS_MIN_N` (default: 10000)
//! - `P2A_GPU_KMEANS_MIN_D` (default: 20)
//! - `P2A_GPU_BOOTSTRAP_MIN_B` (default: 100)

pub mod blas;
pub mod context;
pub mod dispatch;
pub mod kernels;
pub mod memory;
pub mod solver;

use std::sync::OnceLock;

pub use blas::{matmul_gpu, xtx_gpu, xty_gpu};
pub use context::GpuContext;
pub use dispatch::thresholds;
pub use kernels::pairwise_distances_gpu;
pub use solver::{cholesky_inverse_gpu, sandwich_meat_gpu};

/// Global GPU context singleton.
///
/// Initialized once on first access. Returns `None` if no CUDA device
/// is available or initialization fails.
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get a reference to the global GPU context, if available.
    ///
    /// This function is safe to call from any thread. On first call,
    /// it attempts to initialize CUDA. If initialization fails, all
    /// subsequent calls return `None` immediately.
    pub fn get() -> Option<&'static Self> {
        GPU_CONTEXT.get_or_init(GpuContext::try_init).as_ref()
    }
}
