//! Host-device memory transfer utilities.
//!
//! Converts between ndarray arrays and CUDA device memory (`CudaSlice`).
//!
//! # Row-Major Handling
//!
//! ndarray stores data in row-major (C) order by default. cuBLAS expects
//! column-major (Fortran) order. Key insight: a row-major (n x k) matrix
//! stored as contiguous bytes is identical to a column-major (k x n) matrix.
//! This means cuBLAS sees the transpose of what ndarray stores.
//!
//! We exploit this property to avoid any reordering:
//! - `xtx = X'X`: cuBLAS sees X^T, so we compute X^T * (X^T)^T = X^T * X
//! - `xty = X'y`: cuBLAS sees X^T, so we compute X^T * y directly
//! - `matmul = A*B`: compute C^T = B^T * A^T in cuBLAS, yielding C in row-major

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Copy a 2D ndarray to device memory as a contiguous f64 slice.
///
/// If the array is already contiguous in memory (standard row-major),
/// the raw bytes are copied directly. Otherwise, a contiguous copy is
/// made first.
pub fn array2_to_device(
    stream: &Arc<CudaStream>,
    arr: &ArrayView2<f64>,
) -> Result<CudaSlice<f64>, cudarc::driver::DriverError> {
    let contiguous = if let Some(slice) = arr.as_slice() {
        // Already contiguous in memory
        slice.to_vec()
    } else {
        // Need to make contiguous (e.g., transposed view)
        let owned = arr.to_owned();
        owned.into_raw_vec()
    };
    stream.clone_htod(&contiguous)
}

/// Copy a 1D ndarray to device memory.
pub fn array1_to_device(
    stream: &Arc<CudaStream>,
    arr: &ArrayView1<f64>,
) -> Result<CudaSlice<f64>, cudarc::driver::DriverError> {
    let data = if let Some(slice) = arr.as_slice() {
        slice.to_vec()
    } else {
        arr.to_vec()
    };
    stream.clone_htod(&data)
}

/// Copy device memory back to a 2D ndarray (row-major).
pub fn device_to_array2(
    stream: &Arc<CudaStream>,
    dev: &CudaSlice<f64>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>, cudarc::driver::DriverError> {
    let host: Vec<f64> = stream.clone_dtoh(dev)?;
    Ok(Array2::from_shape_vec((rows, cols), host).expect("shape mismatch in device_to_array2"))
}

/// Copy device memory back to a 1D ndarray.
pub fn device_to_array1(
    stream: &Arc<CudaStream>,
    dev: &CudaSlice<f64>,
    len: usize,
) -> Result<Array1<f64>, cudarc::driver::DriverError> {
    let host: Vec<f64> = stream.clone_dtoh(dev)?;
    debug_assert_eq!(host.len(), len, "length mismatch in device_to_array1");
    Ok(Array1::from_vec(host))
}
