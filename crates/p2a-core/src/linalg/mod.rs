//! Linear algebra utilities for econometric computations.
//!
//! This module provides:
//! - Safe matrix operations (inverse, pseudoinverse, condition number)
//! - Design matrix construction from DataFrames
//! - Group-based data transformations (demeaning for panel data)

pub mod matrix_ops;
pub mod design;

pub use matrix_ops::{
    LinalgError, CONDITION_THRESHOLD,
    matrix_inverse, pseudoinverse, safe_inverse, condition_number,
    cholesky, solve, eig_symmetric,
    matmul, xtx, xty, xtx_inv,
    ndarray_to_faer, faer_to_ndarray, faer_col_to_ndarray,
};

pub use design::{
    DesignError, DesignMatrix,
    extract_groups, demean_within_groups, quasi_demean_within_groups,
};
