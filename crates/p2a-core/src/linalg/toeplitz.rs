//! Toeplitz matrix construction.
//!
//! A Toeplitz matrix has constant values along each diagonal. This module provides
//! functions to construct symmetric and asymmetric Toeplitz matrices, matching
//! R's `toeplitz()` function.

use ndarray::{Array1, Array2};

/// Result type for Toeplitz matrix operations.
pub type ToeplitzResult<T> = Result<T, String>;

/// Create a symmetric Toeplitz matrix from the first column.
///
/// A symmetric Toeplitz matrix has T[i,j] = x[|i-j|], meaning the matrix is
/// symmetric and determined entirely by its first column.
///
/// # Arguments
/// * `x` - The first column (and first row) of the symmetric Toeplitz matrix
///
/// # Returns
/// A square n×n symmetric Toeplitz matrix where n = x.len()
///
/// # Example
/// ```
/// use p2a_core::linalg::toeplitz::toeplitz;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let mat = toeplitz(&x).unwrap();
/// // Result:
/// // [1, 2, 3]
/// // [2, 1, 2]
/// // [3, 2, 1]
/// ```
pub fn toeplitz(x: &[f64]) -> ToeplitzResult<Array2<f64>> {
    if x.is_empty() {
        return Err("Input vector cannot be empty".to_string());
    }

    let n = x.len();
    let mut mat = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let idx = if i > j { i - j } else { j - i };
            mat[[i, j]] = x[idx];
        }
    }

    Ok(mat)
}

/// Create an asymmetric Toeplitz matrix from the first column and first row.
///
/// # Arguments
/// * `col` - The first column of the Toeplitz matrix
/// * `row` - The first row of the Toeplitz matrix
///
/// # Returns
/// A matrix of dimensions n×m where n = col.len() and m = row.len()
///
/// # Note
/// If col[0] != row[0], col[0] is used for the [0,0] element (matching R's behavior).
///
/// # Example
/// ```
/// use p2a_core::linalg::toeplitz::toeplitz_asymmetric;
///
/// let col = vec![1.0, 2.0, 3.0];
/// let row = vec![1.0, 4.0, 5.0, 6.0];
/// let mat = toeplitz_asymmetric(&col, &row).unwrap();
/// // Result:
/// // [1, 4, 5, 6]
/// // [2, 1, 4, 5]
/// // [3, 2, 1, 4]
/// ```
pub fn toeplitz_asymmetric(col: &[f64], row: &[f64]) -> ToeplitzResult<Array2<f64>> {
    if col.is_empty() || row.is_empty() {
        return Err("Input vectors cannot be empty".to_string());
    }

    let n = col.len();
    let m = row.len();
    let mut mat = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            if i > j {
                // Below main diagonal: use column values
                mat[[i, j]] = col[i - j];
            } else if j > i {
                // Above main diagonal: use row values
                mat[[i, j]] = row[j - i];
            } else {
                // On main diagonal: use col[0] (matches R behavior)
                mat[[i, j]] = col[0];
            }
        }
    }

    Ok(mat)
}

/// Create a Toeplitz matrix from the "upper-and-left border" specification.
///
/// This matches R's `toeplitz2()` function, where x represents the border
/// from top-right to bottom-left, and T[i,j] = x[i - j + ncol].
///
/// # Arguments
/// * `x` - The upper-and-left border values
/// * `nrow` - Number of rows (if None, computed from x and ncol)
/// * `ncol` - Number of columns (if None, computed from x and nrow)
///
/// # Returns
/// A matrix of the specified dimensions
pub fn toeplitz2(
    x: &[f64],
    nrow: Option<usize>,
    ncol: Option<usize>,
) -> ToeplitzResult<Array2<f64>> {
    if x.is_empty() {
        return Err("Input vector cannot be empty".to_string());
    }

    let len = x.len();

    // Compute dimensions (following R's logic)
    let (n, m) = match (nrow, ncol) {
        (Some(nr), Some(nc)) => {
            if nr + nc > len + 1 {
                return Err(format!(
                    "nrow ({}) + ncol ({}) must be <= length(x) + 1 ({})",
                    nr,
                    nc,
                    len + 1
                ));
            }
            (nr, nc)
        }
        (Some(nr), None) => {
            if nr > len {
                return Err(format!("nrow ({}) cannot exceed length(x) ({})", nr, len));
            }
            let nc = len + 1 - nr;
            (nr, nc)
        }
        (None, Some(nc)) => {
            if nc > len {
                return Err(format!("ncol ({}) cannot exceed length(x) ({})", nc, len));
            }
            let nr = len + 1 - nc;
            (nr, nc)
        }
        (None, None) => {
            // Default to square matrix if possible
            let side = (len + 1) / 2;
            (side, len + 1 - side)
        }
    };

    let mut mat = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            // T[i,j] = x[i - j + ncol - 1] (0-indexed adjustment)
            let idx = (i as isize) - (j as isize) + (m as isize) - 1;
            if idx >= 0 && (idx as usize) < len {
                mat[[i, j]] = x[idx as usize];
            }
        }
    }

    Ok(mat)
}

/// Create an AR(p) covariance matrix (symmetric Toeplitz from autocorrelations).
///
/// This is a common use case: given autocorrelations ρ(0), ρ(1), ..., ρ(p),
/// construct the covariance matrix for an AR process.
///
/// # Arguments
/// * `acf` - Autocorrelation values starting with ρ(0) = 1
///
/// # Returns
/// The autocorrelation Toeplitz matrix
pub fn toeplitz_acf(acf: &[f64]) -> ToeplitzResult<Array2<f64>> {
    toeplitz(acf)
}

/// Convert the result to a Vec<Vec<f64>> for easier JSON serialization.
pub fn toeplitz_to_vec(mat: &Array2<f64>) -> Vec<Vec<f64>> {
    let (nrow, ncol) = mat.dim();
    (0..nrow)
        .map(|i| (0..ncol).map(|j| mat[[i, j]]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_toeplitz_symmetric() {
        let x = vec![1.0, 2.0, 3.0];
        let mat = toeplitz(&x).unwrap();

        assert_eq!(mat.dim(), (3, 3));
        // Expected:
        // [1, 2, 3]
        // [2, 1, 2]
        // [3, 2, 1]
        assert_relative_eq!(mat[[0, 0]], 1.0);
        assert_relative_eq!(mat[[0, 1]], 2.0);
        assert_relative_eq!(mat[[0, 2]], 3.0);
        assert_relative_eq!(mat[[1, 0]], 2.0);
        assert_relative_eq!(mat[[1, 1]], 1.0);
        assert_relative_eq!(mat[[1, 2]], 2.0);
        assert_relative_eq!(mat[[2, 0]], 3.0);
        assert_relative_eq!(mat[[2, 1]], 2.0);
        assert_relative_eq!(mat[[2, 2]], 1.0);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(mat[[i, j]], mat[[j, i]]);
            }
        }
    }

    #[test]
    fn test_toeplitz_asymmetric() {
        let col = vec![1.0, 2.0, 3.0];
        let row = vec![1.0, 4.0, 5.0, 6.0];
        let mat = toeplitz_asymmetric(&col, &row).unwrap();

        assert_eq!(mat.dim(), (3, 4));
        // Expected:
        // [1, 4, 5, 6]
        // [2, 1, 4, 5]
        // [3, 2, 1, 4]
        assert_relative_eq!(mat[[0, 0]], 1.0);
        assert_relative_eq!(mat[[0, 1]], 4.0);
        assert_relative_eq!(mat[[0, 2]], 5.0);
        assert_relative_eq!(mat[[0, 3]], 6.0);
        assert_relative_eq!(mat[[1, 0]], 2.0);
        assert_relative_eq!(mat[[1, 1]], 1.0);
        assert_relative_eq!(mat[[2, 2]], 1.0);
    }

    #[test]
    fn test_toeplitz2() {
        // Border from top-right to bottom-left
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mat = toeplitz2(&x, Some(3), Some(3)).unwrap();

        assert_eq!(mat.dim(), (3, 3));
        // T[i,j] = x[i - j + ncol - 1] = x[i - j + 2]
        // T[0,0] = x[2] = 3
        // T[0,1] = x[1] = 2
        // T[0,2] = x[0] = 1
        // T[1,0] = x[3] = 4
        // T[1,1] = x[2] = 3
        // T[2,2] = x[2] = 3
        assert_relative_eq!(mat[[0, 2]], 1.0);
        assert_relative_eq!(mat[[0, 1]], 2.0);
        assert_relative_eq!(mat[[0, 0]], 3.0);
        assert_relative_eq!(mat[[1, 0]], 4.0);
    }

    #[test]
    fn test_toeplitz_single_element() {
        let x = vec![5.0];
        let mat = toeplitz(&x).unwrap();
        assert_eq!(mat.dim(), (1, 1));
        assert_relative_eq!(mat[[0, 0]], 5.0);
    }

    #[test]
    fn test_toeplitz_acf() {
        // AR(1) autocorrelation structure: 1, ρ, ρ², ρ³, ...
        let rho: f64 = 0.5;
        let acf = vec![1.0, rho, rho.powi(2), rho.powi(3)];
        let mat = toeplitz_acf(&acf).unwrap();

        assert_eq!(mat.dim(), (4, 4));
        // Main diagonal should all be 1
        for i in 0..4 {
            assert_relative_eq!(mat[[i, i]], 1.0);
        }
        // First off-diagonal should be ρ
        assert_relative_eq!(mat[[0, 1]], rho);
        assert_relative_eq!(mat[[1, 0]], rho);
    }

    #[test]
    fn test_toeplitz_empty_error() {
        let result = toeplitz(&[]);
        assert!(result.is_err());
    }
}
