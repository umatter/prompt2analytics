//! Mahalanobis distance computation.
//!
//! The Mahalanobis distance measures the distance between a point and a distribution,
//! accounting for correlations between variables.
//!
//! # Performance
//!
//! This implementation uses Cholesky decomposition for efficient batch computation:
//! - Avoids explicit matrix inverse (numerically more stable)
//! - Uses triangular solves instead of matrix multiplication
//! - Batch processes all observations using matrix operations
//!
//! # Singular covariance
//!
//! A rank-deficient covariance is an error, as R's `mahalanobis` stops on it ("system is exactly
//! singular"). [`SingularCovariance::Generalized`] gives the distances on the rank-r subspace
//! instead, with a warning. The chi-square df of the distances is the rank
//! ([`MahalanobisResult::rank`], [`MahalanobisResult::outlier_threshold`]).
//!
//! # References
//!
//! - Mahalanobis, P. C. (1936). "On the generalized distance in statistics".
//!   Proceedings of the National Institute of Sciences (Calcutta), 2, 49–55.
//! - R Core Team. `stats::mahalanobis()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mahalanobis.html>

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::traits::convergence::ResultNotes;
use crate::traits::estimator::{chi_squared_quantile, validate_level};
use faer::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// What to do with a rank-deficient covariance matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SingularCovariance {
    /// Refuse it, as R's `mahalanobis` does.
    #[default]
    Error,
    /// Use a generalized inverse and report the rank and a warning.
    Generalized,
}

crate::option_enum!(
    SingularCovariance,
    crate::options::Choices {
        option: "singular covariance policy",
        values: &[
            (&["error", "refuse"], SingularCovariance::Error),
            (
                &["generalized", "generalised", "ginv", "pseudoinverse"],
                SingularCovariance::Generalized
            ),
        ],
    }
);

/// Name of the matrix in messages.
const COVARIANCE: &str = "Mahalanobis covariance matrix";

/// Result from Mahalanobis distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MahalanobisResult {
    /// Squared Mahalanobis distances for each observation.
    pub distances: Vec<f64>,
    /// Center used (mean vector).
    pub center: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of variables.
    pub n_vars: usize,
    /// Whether the inverse covariance was provided directly.
    pub inverted: bool,
    /// Original (0-based) row index of each distance; 0..n for the array-level function.
    #[serde(default)]
    pub rows: Vec<usize>,
    /// Rows dropped because of missing values (dataset wrapper; nothing for a matrix).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Numerical rank of the covariance used, the chi-square df of the distances; `n_vars` when
    /// `inverted`.
    #[serde(default)]
    pub rank: usize,
    /// Warnings (a generalized inverse was used).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl MahalanobisResult {
    /// The chi-square outlier threshold `qchisq(level, rank)`. The level must lie in (0, 1); a
    /// covariance of rank 0 has no threshold.
    pub fn outlier_threshold(&self, level: f64) -> EconResult<f64> {
        validate_level(level, "level")?;
        if self.rank == 0 {
            return Err(EconError::InvalidSpecification {
                message: "rank 0 covariance: no chi-square threshold".to_string(),
            });
        }
        chi_squared_quantile(level, self.rank as f64)
    }
}

impl ResultNotes for MahalanobisResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// Compute squared Mahalanobis distance for observations.
///
/// The Mahalanobis distance is defined as:
/// D² = (x - μ)' Σ⁻¹ (x - μ)
///
/// where x is the observation vector, μ is the center (typically the mean),
/// and Σ is the covariance matrix.
///
/// # Performance
///
/// This implementation uses Cholesky decomposition for efficient batch computation:
/// 1. Compute L such that Σ = L L' (Cholesky factorization)
/// 2. Center the data: X_c = X - μ (broadcast subtraction)
/// 3. Solve L W = X_c' for W (triangular solve, O(n*p²))
/// 4. D² = column sums of W² (row sums of squares)
///
/// This avoids explicit matrix inverse and uses O(n*p²) operations instead of
/// O(n*p³) for naive per-row inverse-multiply approach.
///
/// # Arguments
///
/// * `x` - Data matrix where rows are observations and columns are variables
/// * `center` - Center vector (typically the column means). If None, computes column means.
/// * `cov` - Covariance matrix. If None, computes sample covariance from x.
/// * `inverted` - If true, `cov` is interpreted as the inverse covariance matrix
///
/// # Returns
///
/// Vector of squared Mahalanobis distances, one per observation. A rank-deficient covariance is
/// [`EconError::SingularMatrix`], as in R; see [`mahalanobis_with`] for the alternative. An
/// estimated covariance needs at least two rows.
///
/// # Examples
///
/// ```ignore
/// use p2a_core::stats::mahalanobis;
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let result = mahalanobis(x.view(), None, None, false).unwrap();
/// println!("Distances: {:?}", result.distances);
/// ```
pub fn mahalanobis(
    x: ArrayView2<f64>,
    center: Option<ArrayView1<f64>>,
    cov: Option<ArrayView2<f64>>,
    inverted: bool,
) -> EconResult<MahalanobisResult> {
    mahalanobis_with(x, center, cov, inverted, SingularCovariance::Error)
}

/// [`mahalanobis`] with a policy for a rank-deficient covariance.
///
/// With [`SingularCovariance::Generalized`] the distances use a generalized inverse. The centred
/// rows of a sample covariance lie in its range, where every reflexive generalized inverse gives
/// the same quadratic form, so these are also `MASS::ginv`'s distances. `rank` is the chi-square
/// df, and `warnings` says the inverse was generalized. A given `cov` must be symmetric.
pub fn mahalanobis_with(
    x: ArrayView2<f64>,
    center: Option<ArrayView1<f64>>,
    cov: Option<ArrayView2<f64>>,
    inverted: bool,
    singular: SingularCovariance,
) -> EconResult<MahalanobisResult> {
    let (n, p) = x.dim();

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Data must have at least one variable".to_string(),
        });
    }

    if cov.is_none() && n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "Mahalanobis: the sample covariance needs at least 2 rows".to_string(),
        });
    }

    // Compute center if not provided
    let center_vec: Array1<f64> = match center {
        Some(c) => {
            if c.len() != p {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Center length ({}) must match number of variables ({})",
                        c.len(),
                        p
                    ),
                });
            }
            c.to_owned()
        }
        None => x.mean_axis(Axis(0)).ok_or(EconError::EmptyDataset)?,
    };

    // Compute or use provided covariance matrix
    let cov_matrix: Array2<f64> = match cov {
        Some(c) => {
            if c.dim() != (p, p) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Covariance matrix dimensions ({:?}) must be ({}, {})",
                        c.dim(),
                        p,
                        p
                    ),
                });
            }
            c.to_owned()
        }
        None => {
            // Compute sample covariance matrix using optimized batch operations
            compute_covariance_matrix_fast(x, &center_vec)
        }
    };

    // Compute distances using optimized batch method
    let mut warnings = Vec::new();
    let (distances, rank) = if inverted {
        // If already inverted, use direct quadratic form computation (R checks no inverse either)
        (
            compute_distances_with_inverse(x, &center_vec, &cov_matrix),
            p,
        )
    } else {
        let policy = match singular {
            SingularCovariance::Error => SingularPolicy::Error,
            SingularCovariance::Generalized => SingularPolicy::Generalized,
        };
        let inv = sym_inverse(cov_matrix.view(), policy, COVARIANCE).map_err(|e| match e {
            EconError::SingularMatrix { context, .. } => EconError::SingularMatrix {
                context,
                suggestion: "drop the redundant column(s) (R's mahalanobis stops: 'system is \
                             exactly singular'), or pass SingularCovariance::Generalized for \
                             distances on the rank-r subspace (chi-square df = rank)"
                    .to_string(),
            },
            other => other,
        })?;
        let distances = if inv.is_full_rank() {
            // The Cholesky route is the stable one. It fails only on an indefinite given matrix,
            // for which the inverse is what R's solve gives.
            match compute_distances_cholesky(x, &center_vec, &cov_matrix) {
                Ok(d) => d,
                Err(_) => compute_distances_with_inverse(x, &center_vec, &inv.inverse),
            }
        } else {
            warnings.extend(inv.warning(COVARIANCE));
            compute_distances_with_inverse(x, &center_vec, &inv.inverse)
        };
        (distances, inv.rank)
    };

    let mut rows = crate::resource::try_with_capacity(n, "row indices")?;
    rows.extend(0..n);

    Ok(MahalanobisResult {
        distances,
        center: center_vec.to_vec(),
        n_obs: n,
        n_vars: p,
        inverted,
        rows,
        missing: MissingSummary::none(n),
        rank,
        warnings,
    })
}

/// Compute Mahalanobis distances using Cholesky decomposition.
/// This avoids explicit matrix inverse and is numerically more stable.
///
/// Algorithm:
/// 1. Compute L such that Σ = L L' (Cholesky)
/// 2. For each observation x_i, solve L z_i = (x_i - μ) via forward substitution
/// 3. D²_i = ||z_i||² = z_i' z_i
///
/// Note: D² = (x-μ)' Σ⁻¹ (x-μ) = (x-μ)' (LL')⁻¹ (x-μ) = z'z where Lz = (x-μ)
fn compute_distances_cholesky(
    x: ArrayView2<f64>,
    center: &Array1<f64>,
    cov: &Array2<f64>,
) -> EconResult<Vec<f64>> {
    let (n, p) = x.dim();

    // Convert covariance to faer matrix
    let cov_faer = Mat::from_fn(p, p, |i, j| cov[[i, j]]);

    // Compute Cholesky decomposition: Σ = L L'
    let chol = cov_faer
        .llt(faer::Side::Lower)
        .map_err(|_| EconError::InvalidSpecification {
            message: "Covariance matrix is not positive definite".to_string(),
        })?;

    // Get the lower triangular factor L and convert to ndarray
    let l_faer = chol.L();
    let l = Array2::from_shape_fn((p, p), |(i, j)| l_faer[(i, j)]);

    // Solve L z = x[i] - center by forward substitution in one reused buffer; D² = ||z||²
    let mut z = vec![0.0; p];
    let mut distances = vec![0.0; n];
    for (i, d2) in distances.iter_mut().enumerate() {
        let mut sum_sq = 0.0;
        for j in 0..p {
            let mut sum = x[[i, j]] - center[j];
            for k in 0..j {
                sum -= l[[j, k]] * z[k];
            }
            z[j] = sum / l[[j, j]];
            sum_sq += z[j] * z[j];
        }
        *d2 = sum_sq;
    }

    Ok(distances)
}

/// Compute distances when the inverse covariance matrix is already provided.
/// Uses batch matrix multiplication.
fn compute_distances_with_inverse(
    x: ArrayView2<f64>,
    center: &Array1<f64>,
    cov_inv: &Array2<f64>,
) -> Vec<f64> {
    let (n, p) = x.dim();

    // Center the data
    let mut x_centered = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x_centered[[i, j]] = x[[i, j]] - center[j];
        }
    }

    // Compute X_centered * cov_inv (n x p)
    let x_cov_inv = x_centered.dot(cov_inv);

    // Compute row-wise dot products: sum(x_centered[i,:] * x_cov_inv[i,:])
    let mut distances = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            sum += x_centered[[i, j]] * x_cov_inv[[i, j]];
        }
        distances[i] = sum;
    }

    distances
}

/// Compute sample covariance matrix using optimized operations.
fn compute_covariance_matrix_fast(x: ArrayView2<f64>, center: &Array1<f64>) -> Array2<f64> {
    let (n, p) = x.dim();

    // Center the data using vectorized operations
    let mut centered = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            centered[[i, j]] = x[[i, j]] - center[j];
        }
    }

    // Compute covariance: X'X / (n-1) using matrix multiplication
    let xtx = centered.t().dot(&centered);
    &xtx / (n - 1) as f64
}

/// Compute squared Mahalanobis distance for a single observation.
///
/// # Arguments
///
/// * `x` - Single observation vector
/// * `center` - Center vector
/// * `cov_inv` - Inverse covariance matrix
///
/// # Returns
///
/// Squared Mahalanobis distance.
pub fn mahalanobis_single(
    x: ArrayView1<f64>,
    center: ArrayView1<f64>,
    cov_inv: ArrayView2<f64>,
) -> EconResult<f64> {
    let p = x.len();

    if center.len() != p {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Center length ({}) must match observation length ({})",
                center.len(),
                p
            ),
        });
    }

    if cov_inv.dim() != (p, p) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Covariance matrix dimensions ({:?}) must be ({}, {})",
                cov_inv.dim(),
                p,
                p
            ),
        });
    }

    // Compute (x - center)' * cov_inv * (x - center)
    let diff: Array1<f64> = &x - &center;
    let ax = cov_inv.dot(&diff);
    Ok(diff.dot(&ax))
}

/// Convenience function to compute Mahalanobis distance from a Dataset.
///
/// Rows with a missing value (null or NaN) in any of the columns are dropped; the centre and
/// the covariance are computed from the kept rows (as R's
/// `mahalanobis(X, colMeans(X), cov(X))` on `X[complete.cases(X), ]`), and the distances are
/// those of the kept rows. `rows` maps each distance back to its (0-based) dataset row and
/// `missing` reports the drop. Integer, decimal and boolean columns are accepted; a text
/// column is refused (`EconError::NonNumericColumn`), as are infinite values
/// (`EconError::NonFiniteValues`).
///
/// # Arguments
///
/// * `dataset` - Dataset containing the observations
/// * `columns` - Column names to use as variables
/// * `center` - Optional center vector (uses column means if None)
/// * `cov` - Optional covariance matrix (computes from data if None)
///
/// # Returns
///
/// `MahalanobisResult` with distances for each complete row. A rank-deficient covariance is an
/// error; see [`run_mahalanobis_with`].
pub fn run_mahalanobis(
    dataset: &Dataset,
    columns: &[&str],
    center: Option<&[f64]>,
    cov: Option<ArrayView2<f64>>,
) -> EconResult<MahalanobisResult> {
    run_mahalanobis_with(dataset, columns, center, cov, SingularCovariance::Error)
}

/// [`run_mahalanobis`] with a policy for a rank-deficient covariance (see [`mahalanobis_with`]).
pub fn run_mahalanobis_with(
    dataset: &Dataset,
    columns: &[&str],
    center: Option<&[f64]>,
    cov: Option<ArrayView2<f64>>,
    singular: SingularCovariance,
) -> EconResult<MahalanobisResult> {
    if columns.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Must specify at least one column".to_string(),
        });
    }

    let cols = ColumnRequest::new(dataset.df())
        .numerics(columns)
        .na(NaPolicy::Omit)
        .extract()?;
    let data = cols.matrix(columns, false)?.data;

    // Convert center if provided
    let center_arr = center.map(|c| Array1::from_vec(c.to_vec()));
    let center_view = center_arr.as_ref().map(|a| a.view());

    let mut result = mahalanobis_with(data.view(), center_view, cov, false, singular)?;
    result.rows.clear();
    result.rows.extend_from_slice(cols.rows());
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_mahalanobis_basic() {
        // Simple 2D test with identity covariance
        // Should reduce to Euclidean distance squared
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let center = array![0.5, 0.5];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let result = mahalanobis(x.view(), Some(center.view()), Some(cov.view()), false).unwrap();

        assert_eq!(result.n_obs, 4);
        assert_eq!(result.n_vars, 2);

        // Distance from (0,0) to (0.5, 0.5) squared = 0.5
        assert!((result.distances[0] - 0.5).abs() < 1e-10);
        // Distance from (1,0) to (0.5, 0.5) squared = 0.5
        assert!((result.distances[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_with_correlation() {
        // Test with correlated variables
        // Covariance matrix with correlation 0.5
        let x = array![[0.0, 0.0], [2.0, 2.0]];
        let center = array![1.0, 1.0];
        let cov = array![[1.0, 0.5], [0.5, 1.0]];

        let result = mahalanobis(x.view(), Some(center.view()), Some(cov.view()), false).unwrap();

        // Both points equidistant from center
        assert!((result.distances[0] - result.distances[1]).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_auto_center_cov() {
        // Let function compute center and covariance
        let x = array![[1.0, 2.0], [2.0, 3.5], [3.0, 3.0], [4.0, 5.5],];

        let result = mahalanobis(x.view(), None, None, false).unwrap();

        assert_eq!(result.n_obs, 4);
        assert_eq!(result.center.len(), 2);
        // Center should be [2.5, 3.5]
        assert!((result.center[0] - 2.5).abs() < 1e-10);
        assert!((result.center[1] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_inverted() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let center = array![0.5, 0.5];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let cov_inv = array![[1.0, 0.0], [0.0, 1.0]]; // Same for identity

        let result_normal =
            mahalanobis(x.view(), Some(center.view()), Some(cov.view()), false).unwrap();

        let result_inverted =
            mahalanobis(x.view(), Some(center.view()), Some(cov_inv.view()), true).unwrap();

        // Should get same distances
        for i in 0..2 {
            assert!((result_normal.distances[i] - result_inverted.distances[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mahalanobis_single() {
        let x = array![1.0, 2.0];
        let center = array![0.0, 0.0];
        let cov_inv = array![[1.0, 0.0], [0.0, 1.0]];

        let d2 = mahalanobis_single(x.view(), center.view(), cov_inv.view()).unwrap();

        // Should be 1^2 + 2^2 = 5
        assert!((d2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_dimension_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let center = array![1.0]; // Wrong size

        let result = mahalanobis(x.view(), Some(center.view()), None, false);
        assert!(result.is_err());
    }

    /// Validate against R's mahalanobis() function
    #[test]
    fn test_mahalanobis_distance_sum_identity() {
        // With the sample covariance the distances sum to (n - 1) p.
        let x = array![[1.0, 2.0], [3.0, 5.0], [2.0, 4.0], [5.0, 3.0],];

        let result = mahalanobis(x.view(), None, None, false).unwrap();

        // Center should be [2.75, 3.5]
        assert!((result.center[0] - 2.75).abs() < 1e-10);
        assert!((result.center[1] - 3.5).abs() < 1e-10);

        // Here n=4, p=2, so sum should be 3*2 = 6.0
        let sum: f64 = result.distances.iter().sum();

        // All distances should be non-negative
        assert!(result.distances.iter().all(|&d| d >= 0.0));

        // Sum should be close to (n-1)*p = 6.0
        assert!((sum - 6.0).abs() < 0.01, "Sum should be ~6.0, got {}", sum);

        // Basic sanity checks - distances should be reasonable
        for (i, d) in result.distances.iter().enumerate() {
            assert!(*d >= 0.0, "Distance {} should be non-negative", i);
            assert!(*d < 10.0, "Distance {} seems too large: {}", i, d);
        }
    }

    // ========================================================================
    // Missing values and integer columns (case stat_pairwise_mahalanobis_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8: distances on the complete rows (a is missing in 0-based row 6), with centre
    /// and covariance from those rows.
    #[test]
    fn test_validate_mahalanobis_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_pairwise_mahalanobis_na");
        let r = run_mahalanobis(&ds, &["a", "b"], None, None).unwrap();
        let rows: Vec<f64> = r.rows.iter().map(|&i| i as f64).collect();
        Check::new(&case)
            .vector("mahalanobis_complete", &r.distances, Tol::stat(29))
            .vector("mahalanobis_rows", &rows, Tol::exact())
            .named("mahalanobis_center", &["a", "b"], &r.center, Tol::coef(29))
            .scalar("n_complete", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!(r.missing.by_column, vec![("a".to_string(), 1)]);
    }

    #[test]
    fn test_validate_na_mahalanobis_columns() {
        use polars::prelude::*;
        let (_case, ds) = crate::test_support::load_case("stat_pairwise_mahalanobis_na");
        // Int64 accepted: id (Int64, complete) gives the same distances as its Float64 copy
        let mut df = ds.df().clone();
        let id_f = df
            .column("id")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name("id_f".into());
        df.with_column(id_f).unwrap();
        let ds2 = Dataset::new(df);
        let a = run_mahalanobis(&ds2, &["id", "b"], None, None).unwrap();
        let b = run_mahalanobis(&ds2, &["id_f", "b"], None, None).unwrap();
        assert_eq!(a.rows, (0..30).collect::<Vec<_>>());
        assert!(a.missing.is_empty());
        assert_eq!(a.distances.len(), 30);
        for (x, y) in a.distances.iter().zip(&b.distances) {
            assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "{x} vs {y}");
        }
        assert!(matches!(
            run_mahalanobis(&ds, &["a", "g"], None, None),
            Err(EconError::NonNumericColumn { column }) if column == "g"
        ));
        assert!(matches!(
            run_mahalanobis(&ds, &["a", "nope"], None, None),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    /// STAT-54: R's mahalanobis stops on a singular covariance ("system is exactly singular"); p2a silently used a
    /// pseudoinverse and dropped the warning.
    #[test]
    fn test_validate_mahalanobis_t3_singular_is_error() {
        let (case, ds) = crate::test_support::load_case("stat_mahalanobis_t3");
        assert_eq!(case.scalar("ab_r_error"), 1.0);
        assert_eq!(case.scalar("total_r_error"), 1.0);
        for (cols, rank) in [
            (&["a", "b"][..], "rank 1 of 2"),
            (&["p1", "p2", "total"][..], "rank 2 of 3"),
        ] {
            let e = run_mahalanobis(&ds, cols, None, None).unwrap_err();
            match &e {
                EconError::SingularMatrix {
                    context,
                    suggestion,
                } => {
                    assert!(context.contains(rank), "{context}");
                    assert!(suggestion.contains("Generalized"), "{suggestion}");
                }
                other => panic!("expected SingularMatrix, got {other}"),
            }
        }
        // the array entry point too
        let x = ndarray::array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.5, 11.0]];
        assert!(matches!(
            mahalanobis(x.view(), None, None, false),
            Err(EconError::SingularMatrix { .. })
        ));
    }

    /// R4: the covariance needs n >= 2 rows (n - 1 = 0 divided by zero).
    #[test]
    fn test_mahalanobis_needs_two_rows_for_covariance() {
        let x = ndarray::array![[1.0, 2.0]];
        assert!(matches!(
            mahalanobis(x.view(), None, None, false),
            Err(EconError::InsufficientData {
                required: 2,
                provided: 1,
                ..
            })
        ));
        let cov = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        assert!(mahalanobis(x.view(), None, Some(cov.view()), false).is_ok()); // given covariance: one row is fine
    }

    /// STAT-54: the generalised distances live on the rank-r subspace; rank and threshold are reported.
    #[test]
    fn test_validate_mahalanobis_t3_generalized() {
        let (case, ds) = crate::test_support::load_case("stat_mahalanobis_t3");
        let ab = run_mahalanobis_with(
            &ds,
            &["a", "b"],
            None,
            None,
            SingularCovariance::Generalized,
        )
        .unwrap();
        let tot = run_mahalanobis_with(
            &ds,
            &["p1", "p2", "total"],
            None,
            None,
            SingularCovariance::Generalized,
        )
        .unwrap();
        Check::new(&case)
            .vector("ab_ginv_d2", &ab.distances, Tol::stat(5))
            .scalar("ab_rank", ab.rank as f64, Tol::exact())
            .scalar(
                "ab_threshold_rank",
                ab.outlier_threshold(0.975).unwrap(),
                Tol::new(1e-10, 0.0),
            )
            .vector("total_ginv_d2", &tot.distances, Tol::stat(8))
            .scalar("total_rank", tot.rank as f64, Tol::exact())
            .scalar(
                "total_threshold_rank",
                tot.outlier_threshold(0.975).unwrap(),
                Tol::new(1e-10, 0.0),
            )
            .finish();
        assert_eq!(ab.warnings, vec!["Mahalanobis covariance matrix is singular (rank 1 of 2); a generalized inverse was used".to_string()]);
        assert!(
            crate::traits::convergence::notes_text(&ab)
                .contains("Warning: Mahalanobis covariance matrix is singular")
        );
        assert!(ab.outlier_threshold(95.0).is_err());
    }

    /// STAT-44 (mahalanobis.rs part): full-rank distances and centre pinned to R; rank = p, no warning.
    #[test]
    fn test_validate_mahalanobis_t3_full_rank() {
        let (case, ds) = crate::test_support::load_case("stat_mahalanobis_t3");
        let r = run_mahalanobis(&ds, &["f1", "f2", "f3"], None, None).unwrap();
        Check::new(&case)
            .vector("full_d2", &r.distances, Tol::stat(30))
            .vector("full_center", &r.center, Tol::coef(30))
            .scalar(
                "full_threshold",
                r.outlier_threshold(0.975).unwrap(),
                Tol::new(1e-10, 0.0),
            )
            .finish();
        assert_eq!(r.rank, 3);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn test_singular_covariance_parses_aliases_and_rejects_unknown() {
        assert_eq!(SingularCovariance::choices().validate(), Ok(()));
        assert_eq!(
            "ginv".parse::<SingularCovariance>().unwrap(),
            SingularCovariance::Generalized
        );
        assert_eq!(
            "Generalised".parse::<SingularCovariance>().unwrap(),
            SingularCovariance::Generalized
        );
        assert_eq!(
            "refuse".parse::<SingularCovariance>().unwrap(),
            SingularCovariance::Error
        );
        assert!("ridge".parse::<SingularCovariance>().is_err());
        assert_eq!(SingularCovariance::default(), SingularCovariance::Error);
    }
}
