//! CUDA context initialization and device management.
//!
//! Provides a `GpuContext` struct wrapping CUDA device, stream, cuBLAS handle,
//! and cuSOLVER handle. Uses lazy initialization via `OnceLock` singleton.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::cusolver::safe::DnHandle;
use cudarc::driver::{CudaContext as CudaCtx, CudaStream};

use super::dispatch::GpuThresholds;

/// Holds all CUDA resources needed for GPU-accelerated linear algebra.
///
/// Created once via `GpuContext::try_init()` and accessed globally via
/// `GpuContext::get()`. If CUDA is not available, initialization returns
/// `None` and all operations fall back to CPU.
pub struct GpuContext {
    /// CUDA context (device handle)
    pub ctx: Arc<CudaCtx>,
    /// Default CUDA stream for operations
    pub stream: Arc<CudaStream>,
    /// cuBLAS handle for BLAS operations
    pub blas: CudaBlas,
    /// cuSOLVER dense handle for factorizations
    pub solver: DnHandle,
    /// Dispatch thresholds (from env vars)
    pub thresholds: GpuThresholds,
}

// Safety: CUDA handles are thread-safe when used with proper stream synchronization.
// cuBLAS and cuSOLVER handles are internally synchronized.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
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
            blas,
            solver,
            thresholds,
        })
    }
}
