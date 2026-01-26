//! Weighted statistical functions.
//!
//! Implements weighted.mean and cov.wt from R stats package.
//!
//! # Mathematical Background
//!
//! ## Weighted Mean
//!
//! The weighted mean of values x with weights w is:
//!
//! x̄_w = Σᵢ wᵢxᵢ / Σᵢ wᵢ
//!
//! ## Weighted Covariance Matrix
//!
//! For data matrix X (n × p) and weights w:
//!
//! **Frequency weights (default)**: Each wᵢ represents how many times
//! observation i was observed.
//!
//! μ̂ⱼ = Σᵢ wᵢxᵢⱼ / Σᵢ wᵢ
//! Σ̂ⱼₖ = Σᵢ wᵢ(xᵢⱼ - μ̂ⱼ)(xᵢₖ - μ̂ₖ) / (Σᵢ wᵢ - 1)
//!
//! **Probability weights**: Each wᵢ represents the probability of
//! sampling observation i (inverse of sampling probability).
//!
//! Σ̂ⱼₖ = Σᵢ wᵢ(xᵢⱼ - μ̂ⱼ)(xᵢₖ - μ̂ₖ) / (Σᵢ wᵢ - Σᵢ wᵢ²/Σᵢ wᵢ)
//!
//! # References
//!
//! - Cochran, W.G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
//!   ISBN: 978-0471162407. Chapter 5 on weighted estimation.
//!
//! - Kish, L. (1965). *Survey Sampling*. Wiley. ISBN: 978-0471489009.
//!   Foundational text on weighted survey statistics.
//!
//! - Särndal, C.E., Swensson, B., & Wretman, J. (1992). *Model Assisted Survey
//!   Sampling*. Springer. ISBN: 978-0387975283.
//!
//! - Gelman, A. (2007). Struggles with survey weighting and regression modeling.
//!   *Statistical Science*, 22(2), 153-164. https://doi.org/10.1214/088342306000000691
//!
//! R equivalent: `stats::weighted.mean()`, `stats::cov.wt()`

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::errors::{EconError, EconResult};

// ============================================================================
// weighted.mean - Weighted Arithmetic Mean
// ============================================================================

/// Compute the weighted arithmetic mean.
///
/// The weighted mean is computed as: sum(w * x) / sum(w)
///
/// Follows R's weighted.mean behavior:
/// - Weights are normalized to sum to one
/// - Zero weights are handled (corresponding x values excluded)
/// - NA values in x are removed if na_rm is true
/// - Missing/NA weights give NaN result
///
/// # Arguments
///
/// * `x` - Values for which to compute the weighted mean
/// * `w` - Weights (must have same length as x)
/// * `na_rm` - Whether to remove NA values before computation
///
/// # Returns
///
/// The weighted mean as f64.
///
/// # Example
///
/// ```
/// use p2a_core::stats::weighted::weighted_mean;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let w = vec![1.0, 2.0, 3.0, 2.0, 1.0];  // More weight on middle values
/// let result = weighted_mean(&x, &w, true).unwrap();
/// // Result will be closer to 3.0 than simple mean
/// ```
pub fn weighted_mean(x: &[f64], w: &[f64], na_rm: bool) -> EconResult<f64> {
    if x.len() != w.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x and weights must have same length, got {} and {}",
                x.len(),
                w.len()
            ),
        });
    }

    if x.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    // Check for any NaN weights - if present, return NaN
    if w.iter().any(|wi| wi.is_nan()) {
        return Ok(f64::NAN);
    }

    // Check for negative weights
    if w.iter().any(|wi| *wi < 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "Weights must be non-negative".to_string(),
        });
    }

    // Filter out NaN values in x if na_rm is true, along with corresponding weights
    let (clean_x, clean_w): (Vec<f64>, Vec<f64>) = if na_rm {
        x.iter()
            .zip(w.iter())
            .filter(|(xi, _)| !xi.is_nan())
            .map(|(&xi, &wi)| (xi, wi))
            .unzip()
    } else {
        // If not removing NAs and there are any, return NaN
        if x.iter().any(|xi| xi.is_nan()) {
            return Ok(f64::NAN);
        }
        (x.to_vec(), w.to_vec())
    };

    if clean_x.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    // Skip zero weights
    let sum_w: f64 = clean_w.iter().filter(|&&wi| wi != 0.0).sum();

    if sum_w == 0.0 || sum_w.is_infinite() {
        return Ok(f64::NAN);
    }

    // Compute weighted sum (zero weights excluded)
    let weighted_sum: f64 = clean_x
        .iter()
        .zip(clean_w.iter())
        .filter(|&(_, wi)| *wi != 0.0)
        .map(|(&xi, &wi)| xi * wi)
        .sum();

    Ok(weighted_sum / sum_w)
}

/// Run weighted_mean (MCP wrapper).
pub fn run_weighted_mean(x: &[f64], w: &[f64], na_rm: Option<bool>) -> EconResult<f64> {
    weighted_mean(x, w, na_rm.unwrap_or(false))
}

// ============================================================================
// cov.wt - Weighted Covariance Matrices
// ============================================================================

/// Method for computing weighted covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CovWtMethod {
    /// Unbiased estimate (divisor: 1 - sum(w^2)), R's default
    #[default]
    Unbiased,
    /// Maximum likelihood estimate (no correction)
    ML,
}

/// Result of weighted covariance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovWtResult {
    /// Weighted covariance matrix
    pub cov: Vec<Vec<f64>>,
    /// Data center (weighted mean of each variable)
    pub center: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Weights used (normalized)
    pub wt: Vec<f64>,
    /// Correlation matrix (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cor: Option<Vec<Vec<f64>>>,
    /// Variable names (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub names: Option<Vec<String>>,
}

/// Compute weighted covariance matrices.
///
/// Follows R's cov.wt function behavior.
///
/// # Arguments
///
/// * `x` - Data matrix (n × p) with observations as rows and variables as columns
/// * `wt` - Weights vector of length n (default: equal weights 1/n)
/// * `cor` - If true, also compute correlation matrix
/// * `center` - Centering: true = weighted mean, false = zero, or custom values
/// * `method` - Scaling method: Unbiased (default) or ML
///
/// # Returns
///
/// A `CovWtResult` with the weighted covariance matrix and center.
///
/// # Mathematical Details
///
/// For weights w_i normalized to sum to 1:
/// - Center: μ_j = Σ w_i x_ij
/// - Covariance: Σ_jk = Σ w_i (x_ij - μ_j)(x_ik - μ_k) / (1 - Σ w_i²)  [unbiased]
/// - Covariance: Σ_jk = Σ w_i (x_ij - μ_j)(x_ik - μ_k)  [ML]
///
/// With default equal weights (1/n), the unbiased method gives the standard
/// sample covariance with divisor (n-1).
pub fn cov_wt(
    x: &Array2<f64>,
    wt: Option<&[f64]>,
    cor: bool,
    center: CovWtCenter,
    method: CovWtMethod,
) -> EconResult<CovWtResult> {
    let (n, p) = x.dim();

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Data must have at least one variable".to_string(),
        });
    }

    // Handle weights
    let weights: Vec<f64> = match wt {
        Some(w) => {
            if w.len() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!("Weights length {} doesn't match row count {}", w.len(), n),
                });
            }
            if w.iter().any(|wi| *wi < 0.0) {
                return Err(EconError::InvalidSpecification {
                    message: "Weights must be non-negative".to_string(),
                });
            }
            if w.iter().any(|wi| wi.is_nan()) {
                return Err(EconError::InvalidSpecification {
                    message: "Weights cannot contain NaN".to_string(),
                });
            }
            let sum: f64 = w.iter().sum();
            if sum <= 0.0 || sum.is_infinite() {
                return Err(EconError::InvalidSpecification {
                    message: "Sum of weights must be positive and finite".to_string(),
                });
            }
            // Normalize weights to sum to 1
            w.iter().map(|wi| wi / sum).collect()
        }
        None => {
            // Default: equal weights
            vec![1.0 / n as f64; n]
        }
    };

    // Compute center
    let center_vec: Vec<f64> = match center {
        CovWtCenter::WeightedMean => {
            // Weighted mean of each column
            (0..p)
                .map(|j| {
                    x.column(j)
                        .iter()
                        .zip(weights.iter())
                        .map(|(&xij, &wi)| wi * xij)
                        .sum()
                })
                .collect()
        }
        CovWtCenter::Zero => vec![0.0; p],
        CovWtCenter::Custom(ref vals) => {
            if vals.len() != p {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Custom center length {} doesn't match variable count {}",
                        vals.len(),
                        p
                    ),
                });
            }
            vals.clone()
        }
    };

    // Compute weighted covariance matrix
    // Cov_jk = Σ w_i (x_ij - μ_j)(x_ik - μ_k) / divisor
    let mut cov_matrix = vec![vec![0.0; p]; p];

    for j in 0..p {
        for k in j..p {
            let cov_jk: f64 = (0..n)
                .map(|i| {
                    let dev_j = x[[i, j]] - center_vec[j];
                    let dev_k = x[[i, k]] - center_vec[k];
                    weights[i] * dev_j * dev_k
                })
                .sum();

            cov_matrix[j][k] = cov_jk;
            cov_matrix[k][j] = cov_jk; // Symmetric
        }
    }

    // Apply divisor based on method
    let divisor = match method {
        CovWtMethod::Unbiased => {
            // 1 - sum(w^2)
            let sum_w_sq: f64 = weights.iter().map(|wi| wi * wi).sum();
            1.0 - sum_w_sq
        }
        CovWtMethod::ML => 1.0,
    };

    if divisor <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Cannot compute unbiased covariance with these weights (divisor <= 0)"
                .to_string(),
        });
    }

    // Scale covariance matrix
    for j in 0..p {
        for k in 0..p {
            cov_matrix[j][k] /= divisor;
        }
    }

    // Compute correlation matrix if requested
    let cor_matrix = if cor {
        let mut cor_mat = vec![vec![0.0; p]; p];
        let std_devs: Vec<f64> = (0..p).map(|j| cov_matrix[j][j].sqrt()).collect();

        for j in 0..p {
            for k in 0..p {
                if std_devs[j] > 0.0 && std_devs[k] > 0.0 {
                    cor_mat[j][k] = cov_matrix[j][k] / (std_devs[j] * std_devs[k]);
                } else {
                    cor_mat[j][k] = if j == k { 1.0 } else { f64::NAN };
                }
            }
        }
        Some(cor_mat)
    } else {
        None
    };

    Ok(CovWtResult {
        cov: cov_matrix,
        center: center_vec,
        n_obs: n,
        wt: weights,
        cor: cor_matrix,
        names: None,
    })
}

/// Options for centering in cov.wt.
#[derive(Debug, Clone)]
pub enum CovWtCenter {
    /// Use weighted mean (default)
    WeightedMean,
    /// Use zero (no centering)
    Zero,
    /// Use custom center values
    Custom(Vec<f64>),
}

impl Default for CovWtCenter {
    fn default() -> Self {
        CovWtCenter::WeightedMean
    }
}

/// Simplified interface for cov_wt using slices.
///
/// # Arguments
///
/// * `data` - Data as a flat slice in row-major order
/// * `n_rows` - Number of observations
/// * `n_cols` - Number of variables
/// * `weights` - Optional weights
/// * `compute_cor` - Whether to compute correlation matrix
/// * `method` - "unbiased" or "ml"
pub fn cov_wt_from_slice(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    weights: Option<&[f64]>,
    compute_cor: bool,
    method: &str,
) -> EconResult<CovWtResult> {
    if data.len() != n_rows * n_cols {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Data length {} doesn't match {} × {}",
                data.len(),
                n_rows,
                n_cols
            ),
        });
    }

    let x = Array2::from_shape_vec((n_rows, n_cols), data.to_vec()).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Failed to reshape data: {}", e),
        }
    })?;

    let method = match method.to_lowercase().as_str() {
        "ml" => CovWtMethod::ML,
        _ => CovWtMethod::Unbiased,
    };

    cov_wt(&x, weights, compute_cor, CovWtCenter::WeightedMean, method)
}

/// Run cov_wt (MCP wrapper).
pub fn run_cov_wt(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    weights: Option<&[f64]>,
    compute_cor: bool,
    method: Option<&str>,
) -> EconResult<CovWtResult> {
    cov_wt_from_slice(
        data,
        n_rows,
        n_cols,
        weights,
        compute_cor,
        method.unwrap_or("unbiased"),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_weighted_mean_equal_weights() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = weighted_mean(&x, &w, true).unwrap();

        // Should equal simple mean
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_unequal_weights() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // More weight in middle
        let result = weighted_mean(&x, &w, true).unwrap();

        // Expected: (1*1 + 2*2 + 3*3 + 4*2 + 5*1) / (1+2+3+2+1) = 27/9 = 3.0
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_with_zeros() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![0.0, 0.0, 1.0, 0.0, 0.0]; // Only middle value
        let result = weighted_mean(&x, &w, true).unwrap();

        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_with_nan_na_rm_true() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = weighted_mean(&x, &w, true).unwrap();

        // Expected: (1+3+4+5) / 4 = 3.25
        assert!((result - 3.25).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_with_nan_na_rm_false() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = weighted_mean(&x, &w, false).unwrap();

        assert!(result.is_nan());
    }

    #[test]
    fn test_weighted_mean_nan_weight() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, f64::NAN, 1.0];
        let result = weighted_mean(&x, &w, true).unwrap();

        assert!(result.is_nan());
    }

    #[test]
    fn test_weighted_mean_negative_weight_error() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, -1.0, 1.0];
        let result = weighted_mean(&x, &w, true);

        assert!(result.is_err());
    }

    #[test]
    fn test_cov_wt_equal_weights_unbiased() {
        // Simple 2-variable case
        let x = array![
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
        ];

        let result = cov_wt(&x, None, false, CovWtCenter::WeightedMean, CovWtMethod::Unbiased).unwrap();

        // With equal weights (1/5 each), should match regular cov with divisor (n-1)
        // Var(X1) = 2.5, Var(X2) = 2.5, Cov(X1,X2) = 2.5
        assert!((result.cov[0][0] - 2.5).abs() < 1e-10);
        assert!((result.cov[1][1] - 2.5).abs() < 1e-10);
        assert!((result.cov[0][1] - 2.5).abs() < 1e-10);

        // Check center (means)
        assert!((result.center[0] - 3.0).abs() < 1e-10);
        assert!((result.center[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_wt_ml_method() {
        let x = array![
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
        ];

        let result = cov_wt(&x, None, false, CovWtCenter::WeightedMean, CovWtMethod::ML).unwrap();

        // ML method: no small-sample correction, divisor = 1
        // Var_ML = Var_unbiased * (n-1)/n
        // With equal weights: sum(w^2) = 5 * (1/5)^2 = 1/5
        // Unbiased divisor: 1 - 1/5 = 4/5
        // So ML = Unbiased * (4/5) = 2.5 * 0.8 = 2.0
        assert!((result.cov[0][0] - 2.0).abs() < 1e-10);
        assert!((result.cov[1][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_wt_with_correlation() {
        let x = array![
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
        ];

        let result = cov_wt(&x, None, true, CovWtCenter::WeightedMean, CovWtMethod::Unbiased).unwrap();

        // Perfect correlation for this data
        let cor = result.cor.unwrap();
        assert!((cor[0][0] - 1.0).abs() < 1e-10);
        assert!((cor[1][1] - 1.0).abs() < 1e-10);
        assert!((cor[0][1] - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_cov_wt_unequal_weights() {
        let x = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ];
        let weights = vec![1.0, 2.0, 1.0]; // More weight on middle observation

        let result = cov_wt(&x, Some(&weights), false, CovWtCenter::WeightedMean, CovWtMethod::Unbiased).unwrap();

        // Weighted center: (1*1 + 2*2 + 1*3)/4 = 2.0 for X1
        //                  (1*10 + 2*20 + 1*30)/4 = 20.0 for X2
        assert!((result.center[0] - 2.0).abs() < 1e-10);
        assert!((result.center[1] - 20.0).abs() < 1e-10);

        // Check weights are normalized
        let sum_wt: f64 = result.wt.iter().sum();
        assert!((sum_wt - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_wt_zero_center() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];

        let result = cov_wt(&x, None, false, CovWtCenter::Zero, CovWtMethod::Unbiased).unwrap();

        // Center should be zero
        assert_eq!(result.center[0], 0.0);
        assert_eq!(result.center[1], 0.0);
    }

    #[test]
    fn test_cov_wt_from_slice() {
        // Row-major data: [[1, 2], [3, 4], [5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = cov_wt_from_slice(&data, 3, 2, None, true, "unbiased").unwrap();

        assert_eq!(result.n_obs, 3);
        assert_eq!(result.cov.len(), 2);
        assert!(result.cor.is_some());
    }
}
