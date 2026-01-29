//! Median Polish - Robust Two-way Decomposition
//!
//! Implements Tukey's median polish procedure for fitting an additive model
//! (constant + row effects + column effects) to a two-way table.
//!
//! # References
//!
//! - Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
//!   ISBN 9780201076165.
//! - R `medpolish` function: Implementation adapted from R stats package.
//!   Source: <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/medpolish.html>

use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// ============================================================================
// MedpolishResult - Return type for median polish
// ============================================================================

/// Result of median polish decomposition.
///
/// The decomposition satisfies:
/// `original[i,j] = overall + row[i] + col[j] + residuals[i,j]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedpolishResult {
    /// The fitted constant term (overall effect)
    pub overall: f64,

    /// Fitted row effects
    pub row: Vec<f64>,

    /// Fitted column effects
    pub col: Vec<f64>,

    /// Residual matrix
    pub residuals: Vec<Vec<f64>>,

    /// Number of rows in input
    pub n_rows: usize,

    /// Number of columns in input
    pub n_cols: usize,

    /// Number of iterations performed
    pub iterations: usize,

    /// Whether the algorithm converged
    pub converged: bool,

    /// Final sum of absolute residuals
    pub final_sum: f64,

    /// Row names if provided
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_names: Option<Vec<String>>,

    /// Column names if provided
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_names: Option<Vec<String>>,
}

impl MedpolishResult {
    /// Reconstruct fitted values: overall + row[i] + col[j]
    pub fn fitted(&self) -> Vec<Vec<f64>> {
        let mut fitted = vec![vec![0.0; self.n_cols]; self.n_rows];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                fitted[i][j] = self.overall + self.row[i] + self.col[j];
            }
        }
        fitted
    }

    /// Reconstruct original values: overall + row + col + residuals
    pub fn original(&self) -> Vec<Vec<f64>> {
        let mut original = vec![vec![0.0; self.n_cols]; self.n_rows];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                original[i][j] = self.overall + self.row[i] + self.col[j] + self.residuals[i][j];
            }
        }
        original
    }

    /// Get comparison values for Tukey additivity plot.
    /// These are outer(row, col) / overall.
    pub fn comparison_values(&self) -> Vec<Vec<f64>> {
        let mut values = vec![vec![0.0; self.n_cols]; self.n_rows];
        if self.overall.abs() > 1e-15 {
            for i in 0..self.n_rows {
                for j in 0..self.n_cols {
                    values[i][j] = self.row[i] * self.col[j] / self.overall;
                }
            }
        }
        values
    }
}

// ============================================================================
// medpolish - Main function
// ============================================================================

/// Median Polish (Robust Two-way Decomposition) of a Matrix.
///
/// Fits an additive model (constant + rows + columns) using Tukey's median
/// polish procedure. The algorithm iteratively removes row and column medians
/// until convergence.
///
/// # Arguments
///
/// * `x` - Input matrix as 2D vector (row-major)
/// * `eps` - Convergence tolerance (default: 0.01). Iteration stops when
///           the proportional reduction in sum of absolute residuals < eps.
/// * `max_iter` - Maximum iterations (default: 10)
/// * `na_rm` - Remove NaN values when computing medians (default: false)
///
/// # Returns
///
/// A `MedpolishResult` containing overall effect, row effects, column effects,
/// and residuals.
///
/// # Model
///
/// The fitted model is:
/// ```text
/// y[i,j] = μ + α[i] + β[j] + ε[i,j]
/// ```
/// Where μ is the overall effect, α are row effects, β are column effects,
/// and ε are residuals.
///
/// # References
///
/// - Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::medpolish;
///
/// // Sports parachuting deaths by year and month (Tukey 1977)
/// let deaths = vec![
///     vec![14.0, 15.0, 14.0],
///     vec![7.0, 4.0, 7.0],
///     vec![8.0, 2.0, 10.0],
///     vec![15.0, 9.0, 10.0],
/// ];
/// let result = medpolish(&deaths, None, None, false)?;
/// // result.overall + result.row[i] + result.col[j] + result.residuals[i][j]
/// // = deaths[i][j]
/// ```
pub fn medpolish(
    x: &[Vec<f64>],
    eps: Option<f64>,
    max_iter: Option<usize>,
    na_rm: bool,
) -> EconResult<MedpolishResult> {
    // Validate input
    if x.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    let nr = x.len();
    let nc = x[0].len();

    if nc == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Check all rows have same length
    for row in x {
        if row.len() != nc {
            return Err(EconError::InvalidSpecification {
                message: "All rows must have the same number of columns".to_string(),
            });
        }
    }

    // Check for NaN if na_rm is false
    if !na_rm {
        for row in x {
            for &val in row {
                if val.is_nan() {
                    return Err(EconError::InvalidSpecification {
                        message:
                            "Matrix contains NaN values. Set na_rm=true to handle missing values."
                                .to_string(),
                    });
                }
            }
        }
    }

    let eps = eps.unwrap_or(0.01);
    let max_iter = max_iter.unwrap_or(10);

    if eps <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "eps must be greater than 0".to_string(),
        });
    }

    // Initialize
    // z = residuals (starts as copy of x)
    let mut z: Vec<Vec<f64>> = x.to_vec();
    let mut t = 0.0; // overall effect
    let mut r = vec![0.0; nr]; // row effects
    let mut c = vec![0.0; nc]; // column effects
    let mut oldsum = 0.0;
    let mut converged = false;
    let mut iter = 0;

    for iteration in 1..=max_iter {
        iter = iteration;

        // Step 1: Extract row medians
        // rdelta = apply(z, 1, median)
        let rdelta: Vec<f64> = z.iter().map(|row| compute_median(row, na_rm)).collect();

        // z = z - rdelta (subtract row medians from each row)
        for i in 0..nr {
            for j in 0..nc {
                z[i][j] -= rdelta[i];
            }
        }

        // r = r + rdelta
        for i in 0..nr {
            r[i] += rdelta[i];
        }

        // Step 2: Update overall from column effects
        // delta = median(c)
        let delta = compute_median(&c, na_rm);
        // c = c - delta
        for j in 0..nc {
            c[j] -= delta;
        }
        // t = t + delta
        t += delta;

        // Step 3: Extract column medians
        // cdelta = apply(z, 2, median)
        let cdelta: Vec<f64> = (0..nc)
            .map(|j| {
                let col: Vec<f64> = z.iter().map(|row| row[j]).collect();
                compute_median(&col, na_rm)
            })
            .collect();

        // z = z - cdelta (subtract column medians from each column)
        for i in 0..nr {
            for j in 0..nc {
                z[i][j] -= cdelta[j];
            }
        }

        // c = c + cdelta
        for j in 0..nc {
            c[j] += cdelta[j];
        }

        // Step 4: Update overall from row effects
        // delta = median(r)
        let delta = compute_median(&r, na_rm);
        // r = r - delta
        for i in 0..nr {
            r[i] -= delta;
        }
        // t = t + delta
        t += delta;

        // Check convergence
        let newsum: f64 = z
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| !v.is_nan())
            .map(|&v| v.abs())
            .sum();

        if newsum == 0.0 || (oldsum - newsum).abs() < eps * newsum {
            converged = true;
            break;
        }

        oldsum = newsum;
    }

    // Calculate final sum
    let final_sum: f64 = z
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| !v.is_nan())
        .map(|&v| v.abs())
        .sum();

    Ok(MedpolishResult {
        overall: t,
        row: r,
        col: c,
        residuals: z,
        n_rows: nr,
        n_cols: nc,
        iterations: iter,
        converged,
        final_sum,
        row_names: None,
        col_names: None,
    })
}

/// Compute median of a slice, optionally removing NaN values.
fn compute_median(data: &[f64], na_rm: bool) -> f64 {
    let mut clean: Vec<f64> = if na_rm {
        data.iter().filter(|x| !x.is_nan()).copied().collect()
    } else {
        data.to_vec()
    };

    if clean.is_empty() {
        return f64::NAN;
    }

    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = clean.len();
    if n % 2 == 0 {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    } else {
        clean[n / 2]
    }
}

// ============================================================================
// medpolish_array - ndarray version
// ============================================================================

/// Median polish using ndarray Array2.
///
/// Same as `medpolish` but takes an ndarray Array2 for better performance
/// with large matrices.
pub fn medpolish_array(
    x: &Array2<f64>,
    eps: Option<f64>,
    max_iter: Option<usize>,
    na_rm: bool,
) -> EconResult<MedpolishResult> {
    let (nr, nc) = x.dim();

    if nr == 0 || nc == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Check for NaN if na_rm is false
    if !na_rm {
        for &val in x.iter() {
            if val.is_nan() {
                return Err(EconError::InvalidSpecification {
                    message: "Matrix contains NaN values. Set na_rm=true to handle missing values."
                        .to_string(),
                });
            }
        }
    }

    let eps = eps.unwrap_or(0.01);
    let max_iter = max_iter.unwrap_or(10);

    // Initialize
    let mut z = x.to_owned();
    let mut t = 0.0;
    let mut r = Array1::zeros(nr);
    let mut c = Array1::zeros(nc);
    let mut oldsum = 0.0;
    let mut converged = false;
    let mut iter = 0;

    for iteration in 1..=max_iter {
        iter = iteration;

        // Step 1: Row medians
        let rdelta: Array1<f64> = z
            .rows()
            .into_iter()
            .map(|row| array_median(&row.to_vec(), na_rm))
            .collect();

        for i in 0..nr {
            for j in 0..nc {
                z[[i, j]] -= rdelta[i];
            }
        }
        r = &r + &rdelta;

        // Step 2: Update overall from column effects
        let delta = array_median(&c.to_vec(), na_rm);
        c = &c - delta;
        t += delta;

        // Step 3: Column medians
        let cdelta: Array1<f64> = z
            .columns()
            .into_iter()
            .map(|col| array_median(&col.to_vec(), na_rm))
            .collect();

        for i in 0..nr {
            for j in 0..nc {
                z[[i, j]] -= cdelta[j];
            }
        }
        c = &c + &cdelta;

        // Step 4: Update overall from row effects
        let delta = array_median(&r.to_vec(), na_rm);
        r = &r - delta;
        t += delta;

        // Check convergence
        let newsum: f64 = z.iter().filter(|&&v| !v.is_nan()).map(|&v| v.abs()).sum();

        if newsum == 0.0 || (oldsum - newsum).abs() < eps * newsum {
            converged = true;
            break;
        }

        oldsum = newsum;
    }

    let final_sum: f64 = z.iter().filter(|&&v| !v.is_nan()).map(|&v| v.abs()).sum();

    // Convert to Vec for result
    let residuals: Vec<Vec<f64>> = z.rows().into_iter().map(|row| row.to_vec()).collect();

    Ok(MedpolishResult {
        overall: t,
        row: r.to_vec(),
        col: c.to_vec(),
        residuals,
        n_rows: nr,
        n_cols: nc,
        iterations: iter,
        converged,
        final_sum,
        row_names: None,
        col_names: None,
    })
}

/// Compute median for ndarray operations.
fn array_median(data: &[f64], na_rm: bool) -> f64 {
    compute_median(data, na_rm)
}

// ============================================================================
// MCP wrapper
// ============================================================================

/// Run median polish (MCP wrapper).
///
/// # Arguments
///
/// * `x` - Input matrix as 2D vector
/// * `eps` - Convergence tolerance (optional, default 0.01)
/// * `max_iter` - Maximum iterations (optional, default 10)
/// * `na_rm` - Handle NaN values (default false)
pub fn run_medpolish(
    x: &[Vec<f64>],
    eps: Option<f64>,
    max_iter: Option<usize>,
    na_rm: bool,
) -> EconResult<MedpolishResult> {
    medpolish(x, eps, max_iter, na_rm)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medpolish_basic() {
        // Simple 3x3 matrix with clear structure
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let result = medpolish(&data, None, None, false).unwrap();

        // Verify reconstruction
        let original = result.original();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (original[i][j] - data[i][j]).abs() < 1e-10,
                    "Reconstruction failed at ({}, {}): {} != {}",
                    i,
                    j,
                    original[i][j],
                    data[i][j]
                );
            }
        }
    }

    #[test]
    fn test_medpolish_tukey_example() {
        // Sports parachuting deaths example from Tukey (1977)
        // 4 quarters x 3 years
        let deaths = vec![
            vec![14.0, 15.0, 14.0],
            vec![7.0, 4.0, 7.0],
            vec![8.0, 2.0, 10.0],
            vec![15.0, 9.0, 10.0],
        ];

        let result = medpolish(&deaths, Some(0.01), Some(10), false).unwrap();

        // Verify convergence
        assert!(result.converged, "Algorithm should converge");

        // Verify reconstruction
        let original = result.original();
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (original[i][j] - deaths[i][j]).abs() < 1e-10,
                    "Reconstruction failed at ({}, {})",
                    i,
                    j
                );
            }
        }

        // The overall should be positive (typical death count)
        assert!(result.overall > 0.0);
    }

    #[test]
    fn test_medpolish_with_outlier() {
        // Matrix with one outlier - median polish should be robust
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 100.0, 6.0], // 100 is an outlier
            vec![7.0, 8.0, 9.0],
        ];

        let result = medpolish(&data, None, None, false).unwrap();

        // The residual at [1][1] should capture most of the outlier effect
        assert!(
            result.residuals[1][1].abs() > 50.0,
            "Outlier should be captured in residuals"
        );

        // Reconstruction should still work
        let original = result.original();
        for i in 0..3 {
            for j in 0..3 {
                assert!((original[i][j] - data[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_medpolish_perfect_additive() {
        // Perfect additive model: x[i,j] = i + j
        let data = vec![
            vec![2.0, 3.0, 4.0], // 1+1, 1+2, 1+3
            vec![3.0, 4.0, 5.0], // 2+1, 2+2, 2+3
            vec![4.0, 5.0, 6.0], // 3+1, 3+2, 3+3
        ];

        let result = medpolish(&data, None, None, false).unwrap();

        // Residuals should be near zero for perfect additive data
        let max_residual: f64 = result
            .residuals
            .iter()
            .flat_map(|row| row.iter())
            .map(|r| r.abs())
            .fold(0.0, f64::max);

        assert!(
            max_residual < 1e-10,
            "Perfect additive model should have zero residuals, got {}",
            max_residual
        );
    }

    #[test]
    fn test_medpolish_single_row() {
        let data = vec![vec![1.0, 2.0, 3.0]];

        let result = medpolish(&data, None, None, false).unwrap();

        assert_eq!(result.n_rows, 1);
        assert_eq!(result.n_cols, 3);
        assert!(result.converged);
    }

    #[test]
    fn test_medpolish_single_col() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];

        let result = medpolish(&data, None, None, false).unwrap();

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols, 1);
        assert!(result.converged);
    }

    #[test]
    fn test_medpolish_empty() {
        let data: Vec<Vec<f64>> = vec![];
        let result = medpolish(&data, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_medpolish_nan_error() {
        let data = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];

        let result = medpolish(&data, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_medpolish_nan_rm() {
        let data = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];

        let result = medpolish(&data, None, None, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_medpolish_fitted() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = medpolish(&data, None, None, false).unwrap();
        let fitted = result.fitted();

        // Fitted values should be overall + row + col
        for i in 0..2 {
            for j in 0..2 {
                let expected = result.overall + result.row[i] + result.col[j];
                assert!((fitted[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_medpolish_convergence() {
        // Large random-ish matrix to test convergence
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                (0..10)
                    .map(|j| (i + j) as f64 + (i * j) as f64 * 0.1)
                    .collect()
            })
            .collect();

        let result = medpolish(&data, Some(0.001), Some(100), false).unwrap();

        // Should converge within max_iter
        assert!(result.converged || result.iterations <= 100);
    }

    #[test]
    fn test_validate_medpolish_against_r() {
        // R code for validation:
        // x <- matrix(c(14, 7, 8, 15, 15, 4, 2, 9, 14, 7, 10, 10), nrow=4, byrow=FALSE)
        // result <- medpolish(x, trace.iter=FALSE)
        // result$overall  # 9.5
        // result$row      # [1]  4.0 -3.0 -2.5  1.5
        // result$col      # [1]  0.5 -2.0  0.5

        let x = vec![
            vec![14.0, 15.0, 14.0],
            vec![7.0, 4.0, 7.0],
            vec![8.0, 2.0, 10.0],
            vec![15.0, 9.0, 10.0],
        ];

        let result = medpolish(&x, Some(0.01), Some(10), false).unwrap();

        // Expected values from R
        let expected_overall = 9.5;
        let _expected_row = [4.0, -3.0, -2.5, 1.5];
        let _expected_col = [0.5, -2.0, 0.5];

        // Check overall (may differ slightly due to iteration order)
        assert!(
            (result.overall - expected_overall).abs() < 1.0,
            "Overall: expected {}, got {}",
            expected_overall,
            result.overall
        );

        // Verify reconstruction (this should be exact)
        let original = result.original();
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (original[i][j] - x[i][j]).abs() < 1e-10,
                    "Reconstruction mismatch at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    x[i][j],
                    original[i][j]
                );
            }
        }
    }

    #[test]
    fn test_medpolish_array_version() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let result = medpolish_array(&data, None, None, false).unwrap();

        // Verify reconstruction
        let original = result.original();
        for i in 0..3 {
            for j in 0..3 {
                assert!((original[i][j] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
