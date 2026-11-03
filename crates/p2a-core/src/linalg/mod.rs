//! Linear algebra utilities for econometric computations.
//!
//! This module provides:
//! - Safe matrix operations (inverse, pseudoinverse, condition number)
//! - Design matrix construction from DataFrames
//! - Group-based data transformations (demeaning for panel data)
//! - Toeplitz matrix construction
//! - Rank-revealing least squares (pivoted QR as R's `lm.fit`) and symmetric inverses that
//!   report their rank ([`solve`](self::solve))

pub mod design;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod matrix_ops;
pub mod solve;
pub mod toeplitz;

pub use matrix_ops::{
    CONDITION_THRESHOLD, LinalgError, cholesky, cholesky_inverse, condition_number, eig_symmetric,
    faer_col_to_ndarray, faer_ref_to_ndarray, faer_to_ndarray, matmul, matrix_inverse,
    ndarray_to_faer, pseudoinverse, safe_inverse, solve, xtx, xtx_inv, xty,
};

pub use design::{
    DesignError, DesignMatrix, demean_within_groups, extract_groups, get_column_names,
    quasi_demean_within_groups,
};

pub use toeplitz::{toeplitz, toeplitz_acf, toeplitz_asymmetric, toeplitz_to_vec, toeplitz2};
