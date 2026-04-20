//! CUDA context initialization and device management.
//!
//! Provides a `GpuContext` struct wrapping CUDA device, stream, cuBLAS handle,
//! and cuSOLVER handle. Uses lazy initialization via `OnceLock` singleton.
//!
//! # Thread safety
//!
//! The cuBLAS and cuSOLVER C APIs do **not** support concurrent calls on the
//! same handle. Stacking them behind a single `unsafe impl Sync` was
//! unsound: two threads reaching for `ctx.blas.gemm(...)` at once could
//! corrupt internal handle state. To make `Sync` truthful, the handles
//! are wrapped in [`std::sync::Mutex`], and every call site locks the
//! mutex before issuing a cuBLAS/cuSOLVER call. The CUDA driver context
//! and stream are reference-counted (`Arc`) and are designed to be
//! cloned across threads, so they remain bare fields.

use std::sync::{Arc, Mutex};

use cudarc::cublas::CudaBlas;
use cudarc::cusolver::safe::DnHandle;
use cudarc::driver::{CudaContext as CudaCtx, CudaStream};

use super::dispatch::GpuThresholds;

/// Holds all CUDA resources needed for GPU-accelerated linear algebra.
///
/// Created once via `GpuContext::try_init()` and accessed globally via
/// `GpuContext::get()`. If CUDA is not available, initialization returns
/// `None` and all operations fall back to CPU.
///
/// The cuBLAS and cuSOLVER handles are guarded by mutexes; lock them with
/// [`GpuContext::blas`] and [`GpuContext::solver`] before issuing calls.
pub struct GpuContext {
    /// CUDA context (device handle).
    pub ctx: Arc<CudaCtx>,
    /// Default CUDA stream for operations.
    pub stream: Arc<CudaStream>,
    /// cuBLAS handle for BLAS operations. cuBLAS handles are not thread-safe
    /// for concurrent calls, so external synchronization is required.
    blas: Mutex<CudaBlas>,
    /// cuSOLVER dense handle for factorizations. Same thread-safety
    /// constraint as `blas`.
    solver: Mutex<DnHandle>,
    /// Dispatch thresholds (from env vars).
    pub thresholds: GpuThresholds,
}

// SAFETY: CudaBlas and DnHandle are moved between threads with the CUDA
// driver's activation semantics (stream-bound operations). They are *not*
// safe for concurrent access on the same handle, which is why the fields
// are wrapped in `Mutex`. Send is honest because the handles can be moved;
// Sync is honest because the mutexes serialize all access.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Acquire a lock on the cuBLAS handle. The returned guard must be
    /// held for the duration of any cuBLAS call.
    pub fn blas(&self) -> std::sync::MutexGuard<'_, CudaBlas> {
        self.blas
            .lock()
            .expect("GPU cuBLAS mutex poisoned (another thread panicked while holding it)")
    }

    /// Acquire a lock on the cuSOLVER dense handle.
    pub fn solver(&self) -> std::sync::MutexGuard<'_, DnHandle> {
        self.solver
            .lock()
            .expect("GPU cuSOLVER mutex poisoned (another thread panicked while holding it)")
    }

    /// Attempt to initialize CUDA. Returns `None` if no GPU is available
    /// or if any initialization step fails.
    pub fn try_init() -> Option<Self> {
        // Check if any CUDA device is available
        let device_count = CudaCtx::device_count().ok()?;
        if device_count == 0 {
            tracing::info!("No CUDA devices found, GPU acceleration disabled");
            return None;
        }

        // Create context on device 0
        let ctx = CudaCtx::new(0).ok()?;
        let stream = ctx.default_stream();

        // Log device info
        if let Ok(name) = ctx.name() {
            if let Ok((major, minor)) = ctx.compute_capability() {
                tracing::info!(
                    "GPU acceleration enabled: {} (compute {}.{})",
                    name,
                    major,
                    minor
                );
            }
        }

        // Create cuBLAS handle
        let blas = CudaBlas::new(stream.clone()).ok()?;

        // Create cuSOLVER handle
        let solver = DnHandle::new(stream.clone()).ok()?;

        // Load thresholds from environment
        let thresholds = GpuThresholds::from_env();

        Some(Self {
            ctx,
            stream,
            blas: Mutex::new(blas),
            solver: Mutex::new(solver),
            thresholds,
        })
    }
}
