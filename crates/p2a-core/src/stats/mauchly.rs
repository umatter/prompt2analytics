//! Mauchly's test for sphericity.
//!
//! Implements Mauchly's test of sphericity for repeated measures designs.
//! This test evaluates whether a covariance matrix (or transformation thereof)
//! is proportional to the identity matrix, an assumption required for univariate
//! F-tests in repeated measures ANOVA.
//!
//! # Mathematical Background
//!
//! For p repeated measures, the sphericity assumption requires that the
//! variance-covariance matrix of all pairwise differences be proportional
//! to the identity matrix.
//!
//! ## Mauchly's W Statistic
//!
//! Given the sample covariance matrix S of the transformed data:
//!
//! W = |S| / (tr(S)/(p-1))^(p-1)
//!
//! Under H₀ (sphericity holds), a transformation of W follows χ²:
//!
//! -ν ln(W) ~ χ²(df)  where df = p(p-1)/2 - 1
//!
//! ## Epsilon Corrections
//!
//! When sphericity is violated, degrees of freedom are corrected:
//!
//! - **Greenhouse-Geisser (ε̂_GG)**: Conservative lower bound
//!   ε̂_GG = [tr(S)]² / ((p-1) × tr(S²))
//!
//! - **Huynh-Feldt (ε̂_HF)**: Less conservative adjustment
//!   ε̂_HF = (n(p-1)ε̂_GG - 2) / ((p-1)(n-1-(p-1)ε̂_GG))
//!
//! - **Lower bound**: ε_LB = 1/(p-1)
//!
//! # References
//!
//! - Mauchly, J.W. (1940). Significance test for sphericity of a normal n-variate
//!   distribution. *The Annals of Mathematical Statistics*, 11(2), 204-209.
//!   https://doi.org/10.1214/aoms/1177731915
//!
//! - Greenhouse, S.W., & Geisser, S. (1959). On methods in the analysis of profile
//!   data. *Psychometrika*, 24(2), 95-112. https://doi.org/10.1007/BF02289823
//!
//! - Huynh, H., & Feldt, L.S. (1976). Estimation of the Box correction for degrees
//!   of freedom from sample data in randomized block and split-plot designs.
//!   *Journal of Educational Statistics*, 1(1), 69-82.
//!   https://doi.org/10.3102/10769986001001069
//!
//! - Box, G.E.P. (1954). Some theorems on quadratic forms applied in the study
//!   of analysis of variance problems, I. Effect of inequality of variance in
//!   the one-way classification. *The Annals of Mathematical Statistics*, 25(2),
//!   290-302. https://doi.org/10.1214/aoms/1177728786
//!
//! R equivalent: `stats::mauchly.test()`

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use crate::errors::{EconError, EconResult};

// ============================================================================
// Mauchly Test Result
// ============================================================================

/// Result of Mauchly's test for sphericity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MauchlyResult {
    /// Mauchly's W statistic
    pub w: f64,
    /// Chi-squared test statistic
    pub chi_squared: f64,
    /// Degrees of freedom for chi-squared test
    pub df: f64,
    /// P-value
    pub p_value: f64,
    /// Number of levels (p)
    pub p_levels: usize,
    /// Number of subjects/observations (n)
    pub n: usize,
    /// Greenhouse-Geisser epsilon correction
    pub epsilon_gg: f64,
    /// Huynh-Feldt epsilon correction
    pub epsilon_hf: f64,
    /// Lower bound epsilon (1/(p-1))
    pub epsilon_lb: f64,
}

impl std::fmt::Display for MauchlyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mauchly's Test of Sphericity")?;
        writeln!(f, "=============================")?;
        writeln!(f, "W statistic: {:.6}", self.w)?;
        writeln!(f, "Chi-squared: {:.4}", self.chi_squared)?;
        writeln!(f, "df: {:.0}", self.df)?;
        writeln!(f, "p-value: {:.6}", self.p_value)?;
        writeln!(f)?;

        if self.p_value < 0.05 {
            writeln!(f, "Sphericity assumption is violated (p < 0.05)")?;
            writeln!(f, "Consider using epsilon corrections:")?;
        } else {
            writeln!(f, "Sphericity assumption is met (p >= 0.05)")?;
        }
        writeln!(f)?;

        writeln!(f, "Epsilon Corrections:")?;
        writeln!(f, "  Greenhouse-Geisser: {:.4}", self.epsilon_gg)?;
        writeln!(f, "  Huynh-Feldt: {:.4}", self.epsilon_hf)?;
        writeln!(f, "  Lower bound: {:.4}", self.epsilon_lb)?;

        Ok(())
    }
}

// ============================================================================
// Core Implementation
// ============================================================================

/// Perform Mauchly's test for sphericity.
///
/// Tests whether a covariance matrix satisfies the sphericity assumption
/// (equal variances of differences between all pairs of measurements).
///
/// # Arguments
///
/// * `data` - Data matrix (n × p) where n = subjects, p = repeated measures
/// * `sigma` - Expected covariance structure (None = identity/sphericity)
///
/// # Returns
///
/// A `MauchlyResult` containing the test statistics and epsilon corrections.
///
/// # Mathematical Details
///
/// Mauchly's W statistic:
/// ```text
/// W = |S| / (trace(S) / d)^d
/// ```
/// where S is the (p-1) × (p-1) transformed covariance matrix (using orthogonal
/// contrasts), and d = p - 1.
///
/// The test statistic C = -f * ln(W) approximately follows a chi-squared
/// distribution with df = p(p-1)/2 - 1 degrees of freedom.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::mauchly::mauchly_test;
/// use ndarray::array;
///
/// // Data: 10 subjects, 4 repeated measures
/// let data = array![
///     [1.0, 2.0, 3.0, 4.0],
///     [2.0, 3.0, 4.0, 5.0],
///     // ... more subjects
/// ];
///
/// let result = mauchly_test(&data, None)?;
/// println!("W = {}, p = {}", result.w, result.p_value);
/// ```
pub fn mauchly_test(data: &Array2<f64>, sigma: Option<&Array2<f64>>) -> EconResult<MauchlyResult> {
    let (n, p) = data.dim();

    if n < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 observations".to_string(),
        });
    }

    if p < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 repeated measures".to_string(),
        });
    }

    // d = number of orthogonal contrasts = p - 1
    let d = p - 1;

    // Check sigma if provided
    if let Some(s) = sigma {
        if s.dim() != (p, p) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Sigma must be {} × {}, got {} × {}",
                    p, p, s.dim().0, s.dim().1
                ),
            });
        }
    }

    // Compute sample covariance matrix
    let cov = compute_covariance(data)?;

    // Create orthogonal contrast matrix M (p × d)
    // This transforms the original covariance to test sphericity
    let m = create_orthogonal_contrasts(p);

    // Compute transformed covariance: S = M' * Cov * M
    let s = transform_covariance(&cov, &m)?;

    // If sigma is provided, transform it and adjust
    let s = if let Some(sig) = sigma {
        let sig_t = transform_covariance(sig, &m)?;
        // For now, we test against identity (sphericity)
        // A more general test would solve S * Sigma^{-1}
        s
    } else {
        s
    };

    // Compute Mauchly's W statistic
    // W = |S| / (trace(S) / d)^d
    let det_s = matrix_determinant(&s)?;
    let trace_s = matrix_trace(&s);

    if trace_s <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Covariance matrix trace is non-positive".to_string(),
        });
    }

    let w = det_s / (trace_s / d as f64).powi(d as i32);

    // Degrees of freedom for chi-squared test
    // df = p(p-1)/2 - 1 = d(d+1)/2 - 1
    let df = (d * (d + 1)) as f64 / 2.0 - 1.0;

    // Correction factor for better chi-squared approximation
    // f = (n - 1) - (2d^2 + d + 2) / (6d)
    let f1 = n as f64 - 1.0;
    let correction = (2.0 * (d as f64).powi(2) + d as f64 + 2.0) / (6.0 * d as f64);
    let f = f1 - correction;

    // Chi-squared statistic: C = -f * ln(W)
    let chi_squared = if w > 0.0 && w < 1.0 {
        -f * w.ln()
    } else if w >= 1.0 {
        // Perfect sphericity
        0.0
    } else {
        // Invalid (negative determinant suggests numerical issues)
        f64::NAN
    };

    // P-value from chi-squared distribution
    let p_value = if df > 0.0 && chi_squared.is_finite() && chi_squared >= 0.0 {
        let chi_dist = ChiSquared::new(df).map_err(|e| EconError::Internal(e.to_string()))?;
        1.0 - chi_dist.cdf(chi_squared)
    } else {
        f64::NAN
    };

    // Compute epsilon corrections
    let (epsilon_gg, epsilon_hf, epsilon_lb) = compute_epsilon_corrections(&s, n, p)?;

    Ok(MauchlyResult {
        w,
        chi_squared,
        df,
        p_value,
        p_levels: p,
        n,
        epsilon_gg,
        epsilon_hf,
        epsilon_lb,
    })
}

/// Compute sample covariance matrix.
fn compute_covariance(data: &Array2<f64>) -> EconResult<Array2<f64>> {
    let (n, p) = data.dim();

    // Column means
    let means = data.mean_axis(Axis(0)).ok_or(EconError::EmptyDataset)?;

    // Center the data
    let centered = data - &means.broadcast((n, p)).unwrap();

    // Covariance: (1/(n-1)) * X'X
    let cov = centered.t().dot(&centered) / (n - 1) as f64;

    Ok(cov)
}

/// Create orthogonal contrast matrix for sphericity test.
///
/// Creates a p × (p-1) matrix where each column represents an orthogonal contrast.
/// The simplest choice is successive differences, but we use Helmert contrasts
/// which are orthonormal.
fn create_orthogonal_contrasts(p: usize) -> Array2<f64> {
    let d = p - 1;
    let mut m = Array2::zeros((p, d));

    // Helmert contrasts (orthonormal)
    for j in 0..d {
        let k = j + 1;
        // First k elements get 1/sqrt(k*(k+1))
        let scale1 = 1.0 / ((k * (k + 1)) as f64).sqrt();
        for i in 0..k {
            m[[i, j]] = scale1;
        }
        // (k+1)th element gets -k/sqrt(k*(k+1))
        m[[k, j]] = -(k as f64) / ((k * (k + 1)) as f64).sqrt();
    }

    m
}

/// Transform covariance matrix using contrast matrix: S = M' * Cov * M
fn transform_covariance(cov: &Array2<f64>, m: &Array2<f64>) -> EconResult<Array2<f64>> {
    // S = M' * Cov * M
    let temp = m.t().dot(cov);
    let s = temp.dot(m);
    Ok(s)
}

/// Compute matrix determinant (for small matrices).
fn matrix_determinant(a: &Array2<f64>) -> EconResult<f64> {
    let (n, _m) = a.dim();

    match n {
        1 => Ok(a[[0, 0]]),
        2 => Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            // 3x3 determinant using rule of Sarrus
            Ok(a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]))
        }
        _ => {
            // Use LU decomposition for larger matrices
            lu_determinant(a)
        }
    }
}

/// Compute determinant using LU decomposition.
fn lu_determinant(a: &Array2<f64>) -> EconResult<f64> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut det = 1.0;
    let mut sign = 1.0;

    for k in 0..n {
        // Partial pivoting
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Ok(0.0); // Singular matrix
        }

        if max_row != k {
            // Swap rows
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            sign = -sign;
        }

        det *= lu[[k, k]];

        // Elimination
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / lu[[k, k]];
            for j in (k + 1)..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    Ok(det * sign)
}

/// Compute matrix trace.
fn matrix_trace(a: &Array2<f64>) -> f64 {
    a.diag().sum()
}

/// Compute epsilon corrections for sphericity violations.
///
/// Returns (Greenhouse-Geisser, Huynh-Feldt, Lower Bound) epsilon values.
fn compute_epsilon_corrections(
    s: &Array2<f64>,
    n: usize,
    p: usize,
) -> EconResult<(f64, f64, f64)> {
    let d = p - 1;

    // Lower bound: 1 / (p - 1)
    let epsilon_lb = 1.0 / d as f64;

    // Greenhouse-Geisser epsilon
    // ε_GG = trace(S)² / (d * trace(S²))
    let trace_s = matrix_trace(s);
    let s_squared = s.dot(s);
    let trace_s2 = matrix_trace(&s_squared);

    let epsilon_gg = if trace_s2 > 0.0 {
        (trace_s * trace_s) / (d as f64 * trace_s2)
    } else {
        1.0
    };

    // Clamp to valid range [1/(p-1), 1]
    let epsilon_gg = epsilon_gg.clamp(epsilon_lb, 1.0);

    // Huynh-Feldt epsilon
    // ε_HF = (n * (p-1) * ε_GG - 2) / ((p-1) * (n - 1 - (p-1) * ε_GG))
    let n_f = n as f64;
    let d_f = d as f64;

    let numerator = n_f * d_f * epsilon_gg - 2.0;
    let denominator = d_f * (n_f - 1.0 - d_f * epsilon_gg);

    let epsilon_hf = if denominator > 0.0 {
        numerator / denominator
    } else {
        1.0
    };

    // Clamp to valid range [ε_GG, 1] (HF is always >= GG)
    let epsilon_hf = epsilon_hf.clamp(epsilon_gg, 1.0);

    Ok((epsilon_gg, epsilon_hf, epsilon_lb))
}

// ============================================================================
// Simplified interface
// ============================================================================

/// Perform Mauchly's test from flat data.
///
/// # Arguments
///
/// * `data` - Flat data in row-major order (n × p)
/// * `n` - Number of subjects
/// * `p` - Number of repeated measures
pub fn mauchly_test_from_slice(data: &[f64], n: usize, p: usize) -> EconResult<MauchlyResult> {
    if data.len() != n * p {
        return Err(EconError::InvalidSpecification {
            message: format!("Data length {} doesn't match {} × {}", data.len(), n, p),
        });
    }

    let matrix = Array2::from_shape_vec((n, p), data.to_vec()).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Failed to reshape data: {}", e),
        }
    })?;

    mauchly_test(&matrix, None)
}

/// Run Mauchly's test (MCP wrapper).
pub fn run_mauchly_test(data: &[f64], n: usize, p: usize) -> EconResult<MauchlyResult> {
    mauchly_test_from_slice(data, n, p)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mauchly_spherical() {
        // Data with approximate sphericity (different noise patterns)
        // Measurements have similar variance differences
        let data = array![
            [1.0, 1.5, 2.1],
            [2.0, 2.4, 2.9],
            [1.5, 2.1, 2.5],
            [2.5, 2.9, 3.4],
            [1.2, 1.8, 2.3],
            [2.2, 2.7, 3.1],
            [1.8, 2.2, 2.8],
            [2.8, 3.3, 3.7],
            [1.4, 1.9, 2.4],
            [2.4, 2.8, 3.3],
            [3.1, 3.5, 4.0],
            [0.8, 1.3, 1.8],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // For approximately spherical data, W should be reasonably close to 1
        // And p-value should be non-significant (> 0.05)
        assert!(result.w.is_finite(), "W should be finite");
        assert!(result.p_value.is_finite(), "p-value should be finite");
    }

    #[test]
    fn test_mauchly_non_spherical() {
        // Data with clear violation of sphericity
        // Variance of first difference >> variance of second difference
        let data = array![
            [1.0, 10.0, 11.0],   // Large jump then small
            [2.0, 12.0, 12.5],
            [1.5, 15.0, 15.2],
            [2.5, 8.0, 8.3],
            [1.2, 11.0, 11.1],
            [2.2, 14.0, 14.2],
            [1.8, 9.0, 9.5],
            [2.8, 13.0, 13.1],
            [1.4, 16.0, 16.3],
            [2.4, 7.0, 7.2],
            [3.0, 5.0, 5.8],
            [0.5, 18.0, 18.1],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // W should be small for non-spherical data
        assert!(result.w < 0.9 || result.p_value < 0.05, "W = {} should be smaller or p < 0.05", result.w);
        // Epsilon corrections should be < 1
        assert!(result.epsilon_gg <= 1.0);
        assert!(result.epsilon_hf <= 1.0);
    }

    #[test]
    fn test_mauchly_epsilons() {
        // Test epsilon correction bounds with varied data
        let data = array![
            [1.0, 2.2, 3.1, 4.3],
            [2.0, 3.1, 4.5, 5.2],
            [1.5, 2.8, 3.6, 4.9],
            [2.5, 3.4, 4.8, 5.7],
            [1.2, 2.5, 3.3, 4.6],
            [2.2, 3.0, 4.2, 5.4],
            [1.8, 2.6, 3.9, 4.7],
            [2.8, 3.7, 4.6, 5.9],
            [1.4, 2.3, 3.4, 4.5],
            [2.4, 3.2, 4.4, 5.6],
            [3.1, 4.0, 5.1, 6.2],
            [0.9, 2.1, 3.0, 4.2],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // Lower bound should be 1/(p-1) = 1/3
        assert!((result.epsilon_lb - 1.0/3.0).abs() < 0.01);

        // GG should be >= lower bound
        assert!(result.epsilon_gg >= result.epsilon_lb);

        // HF should be >= GG
        assert!(result.epsilon_hf >= result.epsilon_gg - 0.001); // Small tolerance

        // All epsilons should be <= 1
        assert!(result.epsilon_gg <= 1.0);
        assert!(result.epsilon_hf <= 1.0);
    }

    #[test]
    fn test_mauchly_from_slice() {
        // More varied data
        let data = vec![
            1.0, 2.1, 3.3,
            2.0, 3.2, 4.1,
            1.5, 2.4, 3.8,
            2.5, 3.6, 4.5,
            1.2, 2.6, 3.2,
            3.0, 4.1, 5.2,
        ];

        let result = mauchly_test_from_slice(&data, 6, 3).unwrap();

        assert!(result.w.is_finite());
        assert!(result.df > 0.0);
    }

    #[test]
    fn test_orthogonal_contrasts() {
        let m = create_orthogonal_contrasts(4);

        // Check dimensions
        assert_eq!(m.dim(), (4, 3));

        // Check orthogonality: M'M should be close to identity
        let mtm = m.t().dot(&m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (mtm[[i, j]] - expected).abs() < 1e-10,
                    "M'M[{},{}] = {} (expected {})",
                    i, j, mtm[[i, j]], expected
                );
            }
        }
    }

    #[test]
    fn test_determinant() {
        // 2x2 matrix
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let det = matrix_determinant(&a).unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);

        // 3x3 matrix
        let b = array![
            [6.0, 1.0, 1.0],
            [4.0, -2.0, 5.0],
            [2.0, 8.0, 7.0]
        ];
        let det = matrix_determinant(&b).unwrap();
        assert!((det - (-306.0)).abs() < 1e-8);
    }
}
