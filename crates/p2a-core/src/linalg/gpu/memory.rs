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

/// Copy a 2D ndarray to device memory as a contiguous f64 slice in logical row-major order.
///
/// A standard-layout view is copied as it is stored. Any other view (transposed, Fortran
/// order, strided) is copied element by element in row order: `to_owned` would keep a Fortran
/// layout, whose raw vector is column-major. An allocation the host cannot make is reported
/// as `CUDA_ERROR_OUT_OF_MEMORY`.
pub fn array2_to_device(
    stream: &Arc<CudaStream>,
    arr: &ArrayView2<f64>,
) -> Result<CudaSlice<f64>, cudarc::driver::DriverError> {
    let contiguous = crate::linalg::layout::row_major_copy(arr).map_err(|_| {
        cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
    })?;
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
