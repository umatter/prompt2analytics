//! Factor Analysis by Maximum Likelihood.
//!
//! Performs Maximum Likelihood Factor Analysis (MLFA) to identify latent
//! factors underlying observed variables.
//!
//! # References
//!
//! - Jöreskog, K. G. (1967). "Some Contributions to Maximum Likelihood Factor Analysis".
//!   *Psychometrika*, 32, 443-482.
//! - Jöreskog, K. G. (1969). "A General Approach to Confirmatory Maximum Likelihood
//!   Factor Analysis". *Psychometrika*, 34, 183-202.
//! - Kaiser, H. F. (1958). "The varimax criterion for analytic rotation in factor analysis".
//!   *Psychometrika*, 23, 187-200.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/factanal.html

use faer::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{
    eig_symmetric, faer_to_ndarray, matmul, matrix_inverse, ndarray_to_faer,
};

/// Rotation method for factor loadings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RotationMethod {
    /// No rotation applied
    None,
    /// Varimax (orthogonal) rotation - maximizes sum of variances of squared loadings
    #[default]
    Varimax,
    /// Promax (oblique) rotation - allows correlated factors
    Promax,
}

/// Method for computing factor scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScoresMethod {
    /// Do not compute scores
    #[default]
    None,
    /// Thomson's regression method (1951)
    Regression,
    /// Bartlett's weighted least squares (1937)
    Bartlett,
}

crate::option_enum!(
    RotationMethod,
    crate::options::Choices {
        option: "rotation",
        values: &[
            (&["varimax"], RotationMethod::Varimax),
            (&["promax"], RotationMethod::Promax),
            (&["none"], RotationMethod::None),
        ],
    }
);

crate::option_enum!(
    ScoresMethod,
    crate::options::Choices {
        option: "scores",
        values: &[
            (&["none"], ScoresMethod::None),
            (&["regression", "thomson"], ScoresMethod::Regression),
            (&["bartlett"], ScoresMethod::Bartlett),
        ],
    }
);

/// Configuration for factor analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisConfig {
    /// Number of factors to extract
    pub n_factors: usize,
    /// Rotation method
    pub rotation: RotationMethod,
    /// Method for computing factor scores
    pub scores: ScoresMethod,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Lower bound for uniquenesses (prevents Heywood cases)
    pub lower_bound: f64,
    /// Number of random starting values to try
    pub n_start: usize,
}

impl Default for FactorAnalysisConfig {
    fn default() -> Self {
        Self {
            n_factors: 1,
            rotation: RotationMethod::Varimax,
            scores: ScoresMethod::None,
            max_iter: 1000,
            tolerance: 1e-6,
            lower_bound: 0.005,
            n_start: 1,
        }
    }
}

/// Result of factor analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisResult {
    /// Factor loadings matrix (p x k)
    #[serde(skip)]
    pub loadings: Array2<f64>,
    /// Uniquenesses (specific variances)
    #[serde(skip)]
    pub uniquenesses: Array1<f64>,
    /// Communalities `1 - uniquenesses` up to the fit: `rowSums(L0^2)` of the unrotated loadings,
    /// which is `diag(Lambda Phi Lambda')` for every rotation
    #[serde(skip)]
    pub communalities: Array1<f64>,
    /// Proportion of variance explained by each factor
    #[serde(skip)]
    pub variance_proportions: Array1<f64>,
    /// Cumulative proportion of variance
    #[serde(skip)]
    pub cumulative_variance: Array1<f64>,
    /// Factor scores (n x k) if requested
    #[serde(skip)]
    pub scores: Option<Array2<f64>>,
    /// Factor correlation matrix (promax), in the order and signs of `loadings`
    #[serde(skip)]
    pub factor_correlation: Option<Array2<f64>>,
    /// Chi-squared test statistic for goodness of fit
    pub chi_squared: f64,
    /// Degrees of freedom for chi-squared test
    pub df: usize,
    /// P-value for chi-squared test
    pub p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of factors
    pub n_factors: usize,
    /// Whether the solution converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
    /// Final objective function value (negative log-likelihood)
    pub objective: f64,
    /// Variable names if provided
    pub var_names: Option<Vec<String>>,
}

/// Perform maximum likelihood factor analysis.
///
/// # Arguments
///
/// * `data` - Data matrix (n x p) with observations in rows
/// * `n_factors` - Number of factors to extract
/// * `rotation` - Rotation method
/// * `scores` - Method for computing factor scores
///
/// # Returns
///
/// Factor analysis result containing loadings, uniquenesses, and test statistics.
pub fn factanal(
    data: &ArrayView2<f64>,
    n_factors: usize,
    rotation: RotationMethod,
    scores: ScoresMethod,
) -> EconResult<FactorAnalysisResult> {
    let config = FactorAnalysisConfig {
        n_factors,
        rotation,
        scores,
        ..Default::default()
    };
    factanal_with_config(data, &config)
}

/// Perform factor analysis with full configuration.
pub fn factanal_with_config(
    data: &ArrayView2<f64>,
    config: &FactorAnalysisConfig,
) -> EconResult<FactorAnalysisResult> {
    let (n, p) = data.dim();

    if n < p {
        return Err(EconError::InsufficientData {
            required: p,
            provided: n,
            context: "Factor analysis requires at least as many observations as variables"
                .to_string(),
        });
    }

    if config.n_factors == 0 || config.n_factors > p {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of factors must be between 1 and {} (number of variables)",
                p
            ),
        });
    }

    // Compute correlation matrix
    let corr = correlation_matrix(data)?;

    // Run factor analysis on correlation matrix
    factanal_from_corr(&corr.view(), n, config, Some(data))
}

/// Perform factor analysis from a correlation matrix.
///
/// # Arguments
///
/// * `corr` - Correlation matrix (p x p)
/// * `n_obs` - Number of observations (for chi-squared test)
/// * `config` - Analysis configuration
/// * `data` - Original data (optional, for computing scores)
pub fn factanal_from_corr(
    corr: &ArrayView2<f64>,
    n_obs: usize,
    config: &FactorAnalysisConfig,
    data: Option<&ArrayView2<f64>>,
) -> EconResult<FactorAnalysisResult> {
    let p = corr.nrows();
    let k = config.n_factors;

    // Validate inputs
    if corr.nrows() != corr.ncols() {
        return Err(EconError::InvalidSpecification {
            message: "Correlation matrix must be square".to_string(),
        });
    }

    // Check degrees of freedom
    // df = ((p - k)^2 - p - k) / 2
    let df_num = (p as i64 - k as i64).pow(2) - p as i64 - k as i64;
    if df_num < 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Too many factors ({}) for {} variables. Maximum is {}",
                k,
                p,
                (p - 1).min((2 * p + 1) / 3) // Approximate formula
            ),
        });
    }
    let df = (df_num / 2) as usize;

    // Initialize uniquenesses using squared multiple correlations
    // Following Jöreskog (1963) initialization
    let mut psi = initialize_uniquenesses(corr, k)?;

    // Optimize uniquenesses using EM-like algorithm
    let (loadings, final_psi, converged, iterations, objective) =
        optimize_uniquenesses(corr, k, &mut psi, config)?;

    // Communalities from the unrotated loadings: diag(Lambda Phi Lambda') for every rotation
    let communalities = compute_communalities(&loadings.view());

    // Rotate, then sort as factanal's sortLoadings
    let (rotated, phi) = match config.rotation {
        RotationMethod::None => (loadings, None),
        RotationMethod::Varimax => (varimax(&loadings)?.0, None),
        RotationMethod::Promax => {
            let (l, t) = promax(&loadings, 4.0)?;
            let phi = factor_correlation(&t)?;
            (l, Some(phi))
        }
    };
    let (rotated_loadings, factor_corr) = sort_loadings(rotated, phi);

    // Compute variance proportions
    let (var_props, cum_var) = compute_variance_proportions(&rotated_loadings.view());

    // Compute chi-squared test statistic
    // χ² = (n - 1 - (2p + 5)/6 - 2k/3) * f
    // where f is the minimum value of the objective function
    let correction = (2.0 * p as f64 + 5.0) / 6.0 + 2.0 * k as f64 / 3.0;
    let chi_squared = (n_obs as f64 - 1.0 - correction) * objective;
    let chi_squared = chi_squared.max(0.0); // Ensure non-negative

    let p_value = if df > 0 {
        crate::traits::estimator::chi_squared_p_value(chi_squared, df as f64)
    } else {
        1.0
    };

    // Compute factor scores if requested
    let scores = match config.scores {
        ScoresMethod::None => None,
        ScoresMethod::Regression | ScoresMethod::Bartlett => {
            if let Some(data) = data {
                Some(compute_factor_scores(
                    data,
                    &rotated_loadings.view(),
                    &final_psi.view(),
                    corr,
                    config.scores,
                )?)
            } else {
                None
            }
        }
    };

    Ok(FactorAnalysisResult {
        loadings: rotated_loadings,
        uniquenesses: final_psi,
        communalities,
        variance_proportions: var_props,
        cumulative_variance: cum_var,
        scores,
        factor_correlation: factor_corr,
        chi_squared,
        df,
        p_value,
        n_obs,
        n_vars: p,
        n_factors: k,
        converged,
        iterations,
        objective,
        var_names: None,
    })
}

/// Initialize uniquenesses based on squared multiple correlations.
fn initialize_uniquenesses(corr: &ArrayView2<f64>, _n_factors: usize) -> EconResult<Array1<f64>> {
    let p = corr.nrows();
    let mut psi = Array1::zeros(p);

    // Use inverse of correlation matrix diagonal (1 - R²)
    // where R² is the squared multiple correlation of each variable with all others
    // Approximation: 1 - 1/diag(R^{-1})

    // For numerical stability, use a simpler initialization:
    // Start with psi_i = 1 - max eigenvalue contribution
    let (eigenvalues, _) = eig_symmetric(corr)
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

    // Sort eigenvalues in descending order
    let mut sorted_eig: Vec<f64> = eigenvalues.iter().cloned().collect();
    sorted_eig.sort_by(|a, b| b.total_cmp(a));

    // Initial uniquenesses: use communality estimate from first k eigenvalues
    let total_var: f64 = sorted_eig.iter().sum();
    let comm_estimate = sorted_eig.iter().take(_n_factors).sum::<f64>() / total_var;

    // Initial uniqueness = 1 - average communality (bounded)
    let init_uniq = (1.0 - comm_estimate).max(0.1).min(0.9);
    psi.fill(init_uniq);

    Ok(psi)
}

/// Cached information about the correlation matrix for optimization.
struct CorrCache {
    /// Log determinant of correlation matrix (doesn't change during optimization)
    log_det_r: f64,
}

impl CorrCache {
    fn new(corr: &ArrayView2<f64>) -> EconResult<Self> {
        // Compute log|R| once - this never changes during optimization
        let (corr_eigenvalues, _) = eig_symmetric(corr)
            .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;
        let log_det_r: f64 = corr_eigenvalues
            .iter()
            .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 })
            .sum();
        Ok(Self { log_det_r })
    }
}

/// Extract top-k eigenpairs using iterative deflation with power iteration.
///
/// For k << p, this is much faster than a full eigendecomposition.
/// Each eigenpair is found via power iteration, then deflated from the matrix.
///
/// # References
/// - Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins. Ch. 8.
fn top_k_eigenpairs(
    mat: &Array2<f64>,
    k: usize,
    max_power_iter: usize,
    tol: f64,
) -> EconResult<(Vec<f64>, Array2<f64>, Vec<usize>)> {
    let p = mat.nrows();

    // For very small matrices or when k is close to p, full eigendecomposition is faster
    if k * 3 >= p || p <= 8 {
        let (eigenvalues, eigenvectors) = eig_symmetric(&mat.view())
            .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

        let mut idx_val: Vec<(usize, f64)> = eigenvalues.iter().cloned().enumerate().collect();
        idx_val.sort_by(|a, b| b.1.total_cmp(&a.1));

        let sorted_vals: Vec<f64> = idx_val.iter().take(k).map(|&(_, v)| v).collect();
        let sorted_indices: Vec<usize> = idx_val.iter().take(k).map(|&(i, _)| i).collect();

        return Ok((sorted_vals, eigenvectors, sorted_indices));
    }

    // Power iteration with deflation for top-k eigenpairs
    let mut deflated = mat.clone();
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors_cols: Vec<Array1<f64>> = Vec::with_capacity(k);

    for _factor in 0..k {
        // Initialize with a random-ish vector (use alternating signs for diversity)
        let mut v = Array1::from_elem(p, 1.0 / (p as f64).sqrt());
        for i in 0..p {
            if i % 2 == 1 {
                v[i] = -v[i];
            }
            // Add a small perturbation based on position
            v[i] += (i as f64 * 0.01).sin() * 0.1;
        }
        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        let mut eigenvalue = 0.0;

        for _iter in 0..max_power_iter {
            // w = A * v  (matrix-vector multiply)
            let mut w = Array1::zeros(p);
            for i in 0..p {
                let mut sum = 0.0;
                for j in 0..p {
                    sum += deflated[[i, j]] * v[j];
                }
                w[i] = sum;
            }

            // Rayleigh quotient: eigenvalue = v' * w
            let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();

            // Normalize w
            let w_norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if w_norm < 1e-15 {
                break;
            }
            w.mapv_inplace(|x| x / w_norm);

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tol * (1.0 + eigenvalue.abs()) {
                v = w;
                eigenvalue = new_eigenvalue;
                break;
            }

            eigenvalue = new_eigenvalue;
            v = w;
        }

        eigenvalues.push(eigenvalue);
        eigenvectors_cols.push(v.clone());

        // Deflate: A = A - eigenvalue * v * v'
        for i in 0..p {
            for j in 0..p {
                deflated[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    // Build eigenvector matrix (p x p) with only the k columns we found
    // We store them in a p x p matrix for compatibility, with the k columns
    // placed at indices 0..k
    let mut eigvec_matrix = Array2::zeros((p, p));
    for (j, col) in eigenvectors_cols.iter().enumerate() {
        for i in 0..p {
            eigvec_matrix[[i, j]] = col[i];
        }
    }

    // Indices are simply 0..k since we stored them in order
    let sorted_indices: Vec<usize> = (0..k).collect();

    Ok((eigenvalues, eigvec_matrix, sorted_indices))
}

/// Compute log-determinant of a small k x k positive definite matrix using Cholesky.
/// Falls back to eigendecomposition if Cholesky fails.
fn log_det_small(m: &Array2<f64>) -> EconResult<f64> {
    let k = m.nrows();

    // Try Cholesky first (O(k^3/3) vs O(k^3) for eigendecomposition)
    let mat_faer = ndarray_to_faer(&m.view());
    if let Ok(chol) = mat_faer.llt(faer::Side::Lower) {
        // log|M| = 2 * sum(log(diag(L)))
        let l = chol.L();
        let mut log_det = 0.0;
        for i in 0..k {
            let diag = l[(i, i)];
            if diag > 0.0 {
                log_det += diag.ln();
            } else {
                // Cholesky diagonal is non-positive; fall through to eigendecomposition
                return log_det_small_via_eigen(m);
            }
        }
        return Ok(2.0 * log_det);
    }

    log_det_small_via_eigen(m)
}

fn log_det_small_via_eigen(m: &Array2<f64>) -> EconResult<f64> {
    let (eig, _) = eig_symmetric(&m.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;
    Ok(eig
        .iter()
        .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 })
        .sum())
}

/// Optimize uniquenesses using the EM algorithm.
///
/// Returns (loadings, uniquenesses, converged, iterations, objective).
///
/// # Performance optimizations
///
/// 1. **Top-k eigendecomposition**: Uses power iteration with deflation to extract
///    only the top `n_factors` eigenvalues instead of a full O(p^3) decomposition
///    when k << p and p > 8.
/// 2. **Reduced objective frequency**: Full objective function is only computed
///    every 5 iterations; intermediate convergence is checked via psi change only.
/// 3. **Pre-allocated scaled matrix**: The p x p scaled correlation matrix is
///    allocated once and updated in-place each iteration.
/// 4. **Cholesky log-determinant**: Uses Cholesky decomposition for the k x k
///    matrix log-determinant instead of eigendecomposition.
fn optimize_uniquenesses(
    corr: &ArrayView2<f64>,
    n_factors: usize,
    psi: &mut Array1<f64>,
    config: &FactorAnalysisConfig,
) -> EconResult<(Array2<f64>, Array1<f64>, bool, usize, f64)> {
    let p = corr.nrows();
    let mut converged = false;
    let mut iteration = 0;
    let mut prev_obj = f64::INFINITY;
    let mut loadings = Array2::zeros((p, n_factors));

    // Cache correlation matrix info that doesn't change
    let cache = CorrCache::new(corr)?;

    // Optimization 3: Pre-allocate scaled correlation matrix (reused each iteration)
    let mut scaled_corr = Array2::zeros((p, p));

    // How often to compute the full objective (Optimization 2)
    let obj_check_interval = 5;

    for iter in 0..config.max_iter {
        iteration = iter + 1;

        // E-step: Compute reduced correlation matrix R* = Psi^{-1/2} R Psi^{-1/2}
        // Then extract loadings from eigendecomposition
        let (new_loadings, obj) = compute_loadings_optimized(
            corr,
            psi,
            n_factors,
            &cache,
            &mut scaled_corr,
            iter % obj_check_interval == 0, // Only compute objective periodically
        )?;

        // M-step: Update uniquenesses
        // psi_i = 1 - sum(lambda_ij^2)
        let mut new_psi = Array1::zeros(p);
        for i in 0..p {
            let comm: f64 = (0..n_factors).map(|j| new_loadings[[i, j]].powi(2)).sum();
            new_psi[i] = (1.0 - comm)
                .max(config.lower_bound)
                .min(1.0 - config.lower_bound);
        }

        // Check convergence
        let psi_change: f64 = psi
            .iter()
            .zip(new_psi.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        *psi = new_psi;
        loadings = new_loadings;

        // Update objective only when we computed it
        if let Some(obj_val) = obj {
            let obj_change = (prev_obj - obj_val).abs();
            prev_obj = obj_val;

            if psi_change < config.tolerance && obj_change < config.tolerance {
                converged = true;
                break;
            }
        } else {
            // On non-objective iterations, check psi convergence only as a fast check.
            // If psi has converged, compute objective on next iteration to confirm.
            if psi_change < config.tolerance {
                // Compute final objective to confirm convergence
                let final_obj = compute_objective_cached(corr, &loadings.view(), psi, &cache)?;
                let obj_change = (prev_obj - final_obj).abs();
                prev_obj = final_obj;
                if obj_change < config.tolerance {
                    converged = true;
                    break;
                }
            }
        }
    }

    Ok((loadings, psi.clone(), converged, iteration, prev_obj))
}

/// Optimized loadings computation with in-place matrix update and top-k eigendecomposition.
///
/// Given uniquenesses Psi, compute loadings Lambda by:
/// 1. Form R* = Psi^(-1/2) R Psi^(-1/2) (in-place into pre-allocated buffer)
/// 2. Top-k eigendecompose R* (power iteration when k << p, full otherwise)
/// 3. Lambda = Psi^(1/2) V_k (D_k - I)^(1/2) for top k eigenpairs
///
/// Returns (loadings, Some(objective)) or (loadings, None) if compute_obj is false.
fn compute_loadings_optimized(
    corr: &ArrayView2<f64>,
    psi: &Array1<f64>,
    n_factors: usize,
    cache: &CorrCache,
    scaled_corr: &mut Array2<f64>,
    compute_obj: bool,
) -> EconResult<(Array2<f64>, Option<f64>)> {
    let p = corr.nrows();

    // Compute Psi^(-1/2)
    let psi_inv_sqrt: Array1<f64> = psi.iter().map(|&x| 1.0 / x.sqrt()).collect();

    // Update scaled correlation matrix in-place (no allocation)
    for i in 0..p {
        let pis_i = psi_inv_sqrt[i];
        for j in i..p {
            let val = corr[[i, j]] * pis_i * psi_inv_sqrt[j];
            scaled_corr[[i, j]] = val;
            scaled_corr[[j, i]] = val;
        }
    }

    // Top-k eigendecomposition (uses power iteration for large p, full for small p)
    let (top_eigenvalues, eigenvectors, sorted_indices) =
        top_k_eigenpairs(scaled_corr, n_factors, 200, 1e-12)?;

    // Compute loadings: Lambda = Psi^(1/2) V_k (D_k - I)^(1/2)
    // where D_k are the top k eigenvalues
    let psi_sqrt: Array1<f64> = psi.iter().map(|&x| x.sqrt()).collect();
    let mut loadings = Array2::zeros((p, n_factors));

    for (j, &orig_idx) in sorted_indices.iter().take(n_factors).enumerate() {
        let eigval = top_eigenvalues[j];
        // eigenvalue must be > 1 for positive loading contribution
        let sqrt_eigval = (eigval - 1.0).max(0.0).sqrt();
        for i in 0..p {
            loadings[[i, j]] = psi_sqrt[i] * eigenvectors[[i, orig_idx]] * sqrt_eigval;
        }
    }

    // Optimization 3: Only compute objective when requested
    let obj = if compute_obj {
        Some(compute_objective_cached(
            corr,
            &loadings.view(),
            psi,
            cache,
        )?)
    } else {
        None
    };

    Ok((loadings, obj))
}

/// Compute loadings from uniquenesses using eigendecomposition (with caching).
///
/// Given uniquenesses Psi, compute Lambda by:
/// 1. Form R* = Psi^(-1/2) R Psi^(-1/2) - I
/// 2. Eigendecompose R*
/// 3. Lambda = Psi^(1/2) V D^(1/2) for top k eigenpairs
#[allow(dead_code)]
fn compute_loadings_from_psi_cached(
    corr: &ArrayView2<f64>,
    psi: &Array1<f64>,
    n_factors: usize,
    cache: &CorrCache,
) -> EconResult<(Array2<f64>, f64)> {
    let p = corr.nrows();

    // Compute Psi^(-1/2)
    let psi_inv_sqrt: Array1<f64> = psi.iter().map(|&x| 1.0 / x.sqrt()).collect();

    // Form scaled correlation matrix: Psi^(-1/2) R Psi^(-1/2)
    let mut scaled_corr = Array2::zeros((p, p));
    for i in 0..p {
        let pis_i = psi_inv_sqrt[i];
        for j in i..p {
            let val = corr[[i, j]] * pis_i * psi_inv_sqrt[j];
            scaled_corr[[i, j]] = val;
            scaled_corr[[j, i]] = val;
        }
    }

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = eig_symmetric(&scaled_corr.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

    // Sort eigenvalues in descending order and get indices
    let mut idx_val: Vec<(usize, f64)> = eigenvalues.iter().cloned().enumerate().collect();
    idx_val.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Compute loadings: Lambda = Psi^(1/2) V_k (D_k - I)^(1/2)
    // where D_k are the top k eigenvalues
    let psi_sqrt: Array1<f64> = psi.iter().map(|&x| x.sqrt()).collect();
    let mut loadings = Array2::zeros((p, n_factors));

    for (j, &(orig_idx, eigval)) in idx_val.iter().take(n_factors).enumerate() {
        // eigenvalue must be > 1 for positive loading contribution
        let sqrt_eigval = (eigval - 1.0).max(0.0).sqrt();
        for i in 0..p {
            loadings[[i, j]] = psi_sqrt[i] * eigenvectors[[i, orig_idx]] * sqrt_eigval;
        }
    }

    // Compute objective function using cached log|R|
    let obj = compute_objective_cached(corr, &loadings.view(), psi, cache)?;

    Ok((loadings, obj))
}

/// Compute loadings from uniquenesses using eigendecomposition.
/// (Non-cached version for backward compatibility)
#[allow(dead_code)]
fn compute_loadings_from_psi(
    corr: &ArrayView2<f64>,
    psi: &Array1<f64>,
    n_factors: usize,
) -> EconResult<(Array2<f64>, f64)> {
    let cache = CorrCache::new(corr)?;
    compute_loadings_from_psi_cached(corr, psi, n_factors, &cache)
}

/// Compute the objective function (discrepancy function) using cached correlation log-determinant.
fn compute_objective_cached(
    corr: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    psi: &Array1<f64>,
    cache: &CorrCache,
) -> EconResult<f64> {
    let p = corr.nrows();
    let k = loadings.ncols();

    // For factor analysis, Σ = ΛΛ' + Ψ has a special structure we can exploit.
    // Using the Woodbury matrix identity for efficient computation when k << p.
    //
    // If k is small relative to p, we can use:
    // Σ^{-1} = Ψ^{-1} - Ψ^{-1}Λ(I + Λ'Ψ^{-1}Λ)^{-1}Λ'Ψ^{-1}
    // log|Σ| = log|Ψ| + log|I + Λ'Ψ^{-1}Λ|

    if k <= p / 2 {
        // Use Woodbury identity - more efficient for few factors
        compute_objective_woodbury(corr, loadings, psi, cache)
    } else {
        // Fall back to direct computation for many factors
        compute_objective_direct(corr, loadings, psi, cache)
    }
}

/// Optimized Woodbury objective computation.
///
/// Key optimizations vs the original:
/// 1. Uses Cholesky for log|M| computation (O(k^3/3) vs O(k^3) eigendecomp)
/// 2. Pre-computes Psi^{-1} Lambda to avoid redundant diagonal scaling
/// 3. Computes tr(M^{-1} B) via D = M^{-1} C then dot with psi_inv_lambda,
///    avoiding explicit formation of the full B matrix product
fn compute_objective_woodbury(
    corr: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    psi: &Array1<f64>,
    cache: &CorrCache,
) -> EconResult<f64> {
    let p = corr.nrows();
    let k = loadings.ncols();

    // Compute Psi^{-1}
    let psi_inv: Array1<f64> = psi.iter().map(|&x| 1.0 / x.max(1e-10)).collect();

    // Pre-compute Psi^{-1} Lambda (p x k) -- reused multiple times
    let mut psi_inv_lambda = Array2::zeros((p, k));
    for i in 0..p {
        let pi = psi_inv[i];
        for j in 0..k {
            psi_inv_lambda[[i, j]] = pi * loadings[[i, j]];
        }
    }

    // Compute M = I + Lambda' Psi^{-1} Lambda (k x k matrix, symmetric)
    let mut m_matrix = Array2::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut sum = 0.0;
            for m in 0..p {
                sum += psi_inv_lambda[[m, i]] * loadings[[m, j]];
            }
            m_matrix[[i, j]] = sum;
            m_matrix[[j, i]] = sum;
        }
        m_matrix[[i, i]] += 1.0;
    }

    // Optimization: Use Cholesky for log|M| (k x k is small, typically 2-5)
    let log_det_m = log_det_small(&m_matrix)?;

    // Compute M^{-1}
    let m_inv = matrix_inverse(&m_matrix.view())
        .map_err(|e| EconError::Internal(format!("Matrix inversion failed: {}", e)))?;

    // log|Sigma| = log|Psi| + log|M|
    let log_det_psi: f64 = psi.iter().map(|&x| x.max(1e-10).ln()).sum();
    let log_det_sigma = log_det_psi + log_det_m;

    // Compute tr(Sigma^{-1} R) using Woodbury:
    // Sigma^{-1} = Psi^{-1} - Psi^{-1} Lambda M^{-1} Lambda' Psi^{-1}
    // tr(Sigma^{-1} R) = tr(Psi^{-1} R) - tr(M^{-1} Lambda' Psi^{-1} R Psi^{-1} Lambda)

    // First term: tr(Psi^{-1} R) = sum_i psi_inv[i] * R[i,i]
    let trace_psi_inv_r: f64 = (0..p).map(|i| psi_inv[i] * corr[[i, i]]).sum();

    // Fused computation of trace correction:
    // B = Lambda' Psi^{-1} R Psi^{-1} Lambda  (k x k)
    // tr(M^{-1} B) = sum_{i,j} M_inv[i,j] * B[j,i]
    //
    // Step 1: C = psi_inv_lambda' * R  (k x p)
    let mut c_matrix = Array2::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            let mut sum = 0.0;
            for m in 0..p {
                sum += psi_inv_lambda[[m, i]] * corr[[m, j]];
            }
            c_matrix[[i, j]] = sum;
        }
    }

    // Step 2: D = M_inv * C  (k x p)
    let mut d_matrix = Array2::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            let mut sum = 0.0;
            for l in 0..k {
                sum += m_inv[[i, l]] * c_matrix[[l, j]];
            }
            d_matrix[[i, j]] = sum;
        }
    }

    // Step 3: trace_correction = sum_m sum_i psi_inv_lambda[m,i] * D[i,m]
    // This equals tr(M^{-1} B) without forming B explicitly
    let mut trace_correction = 0.0;
    for m in 0..p {
        for i in 0..k {
            trace_correction += psi_inv_lambda[[m, i]] * d_matrix[[i, m]];
        }
    }

    let trace = trace_psi_inv_r - trace_correction;

    // f = log|Sigma| + tr(Sigma^{-1} R) - log|R| - p
    let obj = log_det_sigma + trace - cache.log_det_r - p as f64;

    Ok(obj.max(0.0))
}

/// Compute objective using direct matrix computation (for many factors).
fn compute_objective_direct(
    corr: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    psi: &Array1<f64>,
    cache: &CorrCache,
) -> EconResult<f64> {
    let p = corr.nrows();

    // Σ = ΛΛ' + Ψ
    // Use faer for efficient matrix multiplication
    let loadings_faer = ndarray_to_faer(loadings);
    let llt = &loadings_faer * loadings_faer.transpose();
    let mut sigma = faer_to_ndarray(&llt);
    for i in 0..p {
        sigma[[i, i]] += psi[i];
    }

    // Compute log determinant and inverse of sigma using eigendecomposition
    let (sigma_eigenvalues, sigma_eigenvectors) = eig_symmetric(&sigma.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

    // Log determinant
    let log_det_sigma: f64 = sigma_eigenvalues
        .iter()
        .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 })
        .sum();

    // Compute Σ^(-1) using faer (more efficient than manual loops)
    let sigma_inv = match matrix_inverse(&sigma.view()) {
        Ok(inv) => inv,
        Err(_) => {
            // Fall back to eigendecomposition-based inverse for singular matrices
            let mut inv = Array2::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    let mut sum = 0.0;
                    for m in 0..p {
                        if sigma_eigenvalues[m] > 1e-10 {
                            sum += sigma_eigenvectors[[i, m]] * sigma_eigenvectors[[j, m]]
                                / sigma_eigenvalues[m];
                        }
                    }
                    inv[[i, j]] = sum;
                }
            }
            inv
        }
    };

    // tr(Σ^(-1) R) using faer
    let sigma_inv_faer = ndarray_to_faer(&sigma_inv.view());
    let corr_faer = ndarray_to_faer(corr);
    let sigma_inv_r = &sigma_inv_faer * &corr_faer;
    let trace: f64 = (0..p).map(|i| sigma_inv_r[(i, i)]).sum();

    // f = log|Σ| + tr(Σ^(-1) R) - log|R| - p
    // Use cached log|R|
    let obj = log_det_sigma + trace - cache.log_det_r - p as f64;

    Ok(obj.max(0.0))
}

/// Compute the objective function (discrepancy function).
/// (Non-cached version for backward compatibility)
#[allow(dead_code)]
fn compute_objective(
    corr: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    psi: &Array1<f64>,
) -> EconResult<f64> {
    let cache = CorrCache::new(corr)?;
    compute_objective_cached(corr, loadings, psi, &cache)
}

/// Maximum number of varimax iterations.
const VARIMAX_MAX_ITER: usize = 1000;
/// Relative stop of the varimax iteration: the sum of singular values (the criterion's
/// linearisation) grows by less than this fraction. R's `varimax` default is `eps = 1e-5`;
/// p2a iterates to the optimum (R's `varimax(eps = 1e-14)`).
const VARIMAX_REL_TOL: f64 = 1e-14;

/// `x * diag(d)`: scale the columns of `x`.
fn scale_columns(x: &Array2<f64>, d: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for (mut col, &s) in out.columns_mut().into_iter().zip(d) {
        col *= s;
    }
    out
}

/// Varimax rotation with Kaiser normalisation (R's `stats::varimax(x, normalize = TRUE)`):
/// the SVD iteration `B = x' (z^3 - z diag(colSums(z^2)) / p)`, `T = U V'`, run to the
/// optimum. Returns the rotated loadings `x T` and the rotation matrix `T`.
///
/// A zero row of `x` keeps scale 1 (R would give NaN).
///
/// Reference: Kaiser (1958); R's `stats::varimax`.
pub fn varimax(x: &Array2<f64>) -> EconResult<(Array2<f64>, Array2<f64>)> {
    let (p, k) = x.dim();
    if k < 2 {
        return Ok((x.clone(), Array2::eye(k)));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "varimax: the loadings contain non-finite values".to_string(),
        });
    }
    let sc: Vec<f64> = x
        .rows()
        .into_iter()
        .map(|r| {
            let s = r.dot(&r).sqrt();
            if s > 0.0 { s } else { 1.0 }
        })
        .collect();
    let mut xn = x.clone();
    for (mut row, &s) in xn.rows_mut().into_iter().zip(&sc) {
        row /= s;
    }
    let mut tt: Array2<f64> = Array2::eye(k);
    let mut d = 0.0;
    for _ in 0..VARIMAX_MAX_ITER {
        crate::resource::checkpoint()?;
        let z = xn.dot(&tt);
        let col_ss: Vec<f64> = z
            .columns()
            .into_iter()
            .map(|c| c.dot(&c) / p as f64)
            .collect();
        let z3 = z.mapv(|v| v * v * v) - scale_columns(&z, &col_ss);
        let b = xn.t().dot(&z3);
        let svd = ndarray_to_faer(&b.view())
            .thin_svd()
            .map_err(|_| EconError::Computation("varimax: SVD failed".to_string()))?;
        let u = faer_to_ndarray(&svd.U().to_owned());
        let v = faer_to_ndarray(&svd.V().to_owned());
        tt = u.dot(&v.t());
        let dpast = d;
        d = (0..k).map(|i| svd.S()[i]).sum::<f64>();
        if !d.is_finite() {
            return Err(EconError::Computation(
                "varimax: the criterion is not finite".to_string(),
            ));
        }
        if d < dpast * (1.0 + VARIMAX_REL_TOL) {
            break;
        }
    }
    let mut z = xn.dot(&tt);
    for (mut row, &s) in z.rows_mut().into_iter().zip(&sc) {
        row *= s;
    }
    Ok((z, tt))
}

/// Promax rotation (Hendrickson & White 1964; R's `stats::promax(x, m)`): varimax, the target
/// `Q = xv |xv|^(m - 1)`, `U = lm.fit(xv, Q)`, columns scaled by `sqrt(diag(solve(U'U)))`.
/// Returns the loadings `xv U` and the rotation matrix `T_v U`.
pub fn promax(x: &Array2<f64>, m: f64) -> EconResult<(Array2<f64>, Array2<f64>)> {
    let k = x.ncols();
    if k < 2 {
        return Ok((x.clone(), Array2::eye(k)));
    }
    if !m.is_finite() || m < 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("promax: the power m must be finite and at least 1, got {m}"),
        });
    }
    let (xv, tv) = varimax(x)?;
    let q = xv.mapv(|v| v * v.abs().powf(m - 1.0));
    let mut u = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        let fit = crate::linalg::solve::least_squares(xv.view(), q.column(j))?;
        if fit.coefficients.iter().any(|c| c.is_nan()) {
            return Err(EconError::SingularMatrix {
                context: "promax: the varimax loadings are rank deficient".to_string(),
                suggestion: "extract fewer factors".to_string(),
            });
        }
        u.column_mut(j).assign(&fit.coefficients);
    }
    let utu = u.t().dot(&u);
    let inv = crate::linalg::solve::sym_inverse(
        utu.view(),
        crate::linalg::solve::SingularPolicy::Error,
        "promax U'U",
    )?
    .inverse;
    let d: Vec<f64> = (0..k).map(|i| inv[[i, i]].sqrt()).collect();
    let u = scale_columns(&u, &d);
    Ok((xv.dot(&u), tv.dot(&u)))
}

/// Factor correlations of an oblique rotation with rotation matrix `T`:
/// `solve(T) solve(T)' = (T'T)^-1` (R's `factanal`).
pub fn factor_correlation(rotmat: &Array2<f64>) -> EconResult<Array2<f64>> {
    let ttt = rotmat.t().dot(rotmat);
    let inv = crate::linalg::solve::sym_inverse(
        ttt.view(),
        crate::linalg::solve::SingularPolicy::Error,
        "promax rotation matrix",
    )?
    .inverse;
    // symmetric by construction; average away rounding
    Ok((&inv + &inv.t()) / 2.0)
}

/// The column order and signs of R's `sortLoadings` (`factanal`): columns by descending sum of
/// squared loadings (stable), then each column with a negative sum negated. Returns the
/// permutation `o` and the signs `s`: new column `j` is `s[j] * l[, o[j]]`.
pub fn sort_order(l: &Array2<f64>) -> (Vec<usize>, Vec<f64>) {
    let ss: Vec<f64> = l.columns().into_iter().map(|c| c.dot(&c)).collect();
    let mut o: Vec<usize> = (0..l.ncols()).collect();
    o.sort_by(|&a, &b| ss[b].total_cmp(&ss[a]));
    let signs = o
        .iter()
        .map(|&j| if l.column(j).sum() < 0.0 { -1.0 } else { 1.0 })
        .collect();
    (o, signs)
}

/// Apply a column order and signs (from [`sort_order`]) to the columns of `m`.
pub fn permute_columns(m: &Array2<f64>, o: &[usize], signs: &[f64]) -> Array2<f64> {
    let mut out = Array2::zeros((m.nrows(), o.len()));
    for (j, (&src, &s)) in o.iter().zip(signs).enumerate() {
        out.column_mut(j).assign(&(&m.column(src) * s));
    }
    out
}

/// R's `sortLoadings`: loadings sorted by SS loadings with positive column sums; the factor
/// correlations get the same permutation P and signs D (`D P' phi P D`).
pub fn sort_loadings(
    l: Array2<f64>,
    phi: Option<Array2<f64>>,
) -> (Array2<f64>, Option<Array2<f64>>) {
    let (o, signs) = sort_order(&l);
    let sorted = permute_columns(&l, &o, &signs);
    let phi = phi.map(|phi| {
        let k = o.len();
        Array2::from_shape_fn((k, k), |(a, b)| signs[a] * signs[b] * phi[[o[a], o[b]]])
    });
    (sorted, phi)
}

/// Compute communalities (h² = sum of squared loadings for each variable).
fn compute_communalities(loadings: &ArrayView2<f64>) -> Array1<f64> {
    let (p, k) = loadings.dim();
    Array1::from_iter((0..p).map(|i| (0..k).map(|j| loadings[[i, j]].powi(2)).sum()))
}

/// Compute variance proportions and cumulative variance.
fn compute_variance_proportions(loadings: &ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (p, k) = loadings.dim();

    let ss_loadings: Vec<f64> = (0..k)
        .map(|j| (0..p).map(|i| loadings[[i, j]].powi(2)).sum())
        .collect();

    let total_var = p as f64; // Total variance in standardized data

    let proportions: Array1<f64> = Array1::from_iter(ss_loadings.iter().map(|&ss| ss / total_var));

    let cumulative: Array1<f64> = {
        let mut cum = Vec::with_capacity(k);
        let mut sum = 0.0;
        for prop in proportions.iter() {
            sum += prop;
            cum.push(sum);
        }
        Array1::from_vec(cum)
    };

    (proportions, cumulative)
}

/// Compute factor scores.
fn compute_factor_scores(
    data: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    uniquenesses: &ArrayView1<f64>,
    _corr: &ArrayView2<f64>,
    method: ScoresMethod,
) -> EconResult<Array2<f64>> {
    let (n, p) = data.dim();
    let k = loadings.ncols();

    // Standardize data
    let means: Vec<f64> = (0..p)
        .map(|j| data.column(j).mean().unwrap_or(0.0))
        .collect();
    let stds: Vec<f64> = (0..p)
        .map(|j| {
            let col = data.column(j);
            let mean = means[j];
            let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            var.sqrt().max(1e-10)
        })
        .collect();

    let mut z = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = (data[[i, j]] - means[j]) / stds[j];
        }
    }

    // Compute score coefficient matrix
    let score_coef = match method {
        ScoresMethod::Regression => {
            // Thomson's method: Λ' Σ^(-1)
            // Σ = ΛΛ' + Ψ
            let sigma = compute_sigma(loadings, uniquenesses)?;
            let sigma_inv = invert_matrix(&sigma.view())?;
            matmul(&loadings.t().to_owned().view(), &sigma_inv.view())
                .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?
        }
        ScoresMethod::Bartlett => {
            // Bartlett's method: (Λ' Ψ^(-1) Λ)^(-1) Λ' Ψ^(-1)
            let psi_inv: Array1<f64> = uniquenesses.iter().map(|&x| 1.0 / x.max(1e-10)).collect();

            // Λ' Ψ^(-1)
            let mut lt_psi_inv = Array2::zeros((k, p));
            for i in 0..k {
                for j in 0..p {
                    lt_psi_inv[[i, j]] = loadings[[j, i]] * psi_inv[j];
                }
            }

            // Λ' Ψ^(-1) Λ
            let lt_psi_inv_l = matmul(&lt_psi_inv.view(), loadings)
                .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

            // Invert
            let lt_psi_inv_l_inv = invert_matrix(&lt_psi_inv_l.view())?;

            // (Λ' Ψ^(-1) Λ)^(-1) Λ' Ψ^(-1)
            matmul(&lt_psi_inv_l_inv.view(), &lt_psi_inv.view())
                .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?
        }
        ScoresMethod::None => {
            return Err(EconError::InvalidSpecification {
                message: "Score method is None".to_string(),
            });
        }
    };

    // Compute scores: F = Z * B'
    // where B is the score coefficient matrix (k x p)
    let scores = matmul(&z.view(), &score_coef.t().to_owned().view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    Ok(scores)
}

/// Compute Σ = ΛΛ' + Ψ
fn compute_sigma(
    loadings: &ArrayView2<f64>,
    uniquenesses: &ArrayView1<f64>,
) -> EconResult<Array2<f64>> {
    let p = loadings.nrows();

    let lambda_lambda_t = matmul(loadings, &loadings.t().to_owned().view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    let mut sigma = lambda_lambda_t;
    for i in 0..p {
        sigma[[i, i]] += uniquenesses[i];
    }

    Ok(sigma)
}

/// Invert a symmetric matrix using faer's optimized routines.
/// Falls back to eigendecomposition for near-singular matrices.
fn invert_matrix(m: &ArrayView2<f64>) -> EconResult<Array2<f64>> {
    // Try faer's optimized matrix inverse first
    match matrix_inverse(m) {
        Ok(inv) => Ok(inv),
        Err(_) => {
            // Fall back to eigendecomposition-based inverse for singular/near-singular matrices
            let n = m.nrows();
            let (eigenvalues, eigenvectors) = eig_symmetric(m)
                .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

            let mut inv = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        if eigenvalues[k] > 1e-10 {
                            sum += eigenvectors[[i, k]] * eigenvectors[[j, k]] / eigenvalues[k];
                        }
                    }
                    inv[[i, j]] = sum;
                }
            }
            Ok(inv)
        }
    }
}

/// Compute correlation matrix from data.
/// Uses parallel computation for large datasets (n > 1000).
fn correlation_matrix(data: &ArrayView2<f64>) -> EconResult<Array2<f64>> {
    let (n, p) = data.dim();

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "Need at least 2 observations to compute correlation".to_string(),
        });
    }

    // Compute means (parallel for large n)
    let means: Vec<f64> = if n > 1000 {
        (0..p)
            .into_par_iter()
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect()
    } else {
        (0..p)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect()
    };

    // Compute standard deviations
    let stds: Vec<f64> = if n > 1000 {
        (0..p)
            .into_par_iter()
            .map(|j| {
                let col = data.column(j);
                let mean = means[j];
                let var: f64 =
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt().max(1e-10)
            })
            .collect()
    } else {
        (0..p)
            .map(|j| {
                let col = data.column(j);
                let mean = means[j];
                let var: f64 =
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt().max(1e-10)
            })
            .collect()
    };

    // Compute correlation matrix
    // For large datasets and many variables, use parallel computation
    if n > 1000 && p > 10 {
        // Parallel: compute upper triangular entries
        let _num_pairs = p * (p - 1) / 2;
        let pair_indices: Vec<(usize, usize)> = (0..p)
            .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
            .collect();

        let correlations: Vec<((usize, usize), f64)> = pair_indices
            .par_iter()
            .map(|&(i, j)| {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
                }
                let r = sum / ((n - 1) as f64 * stds[i] * stds[j]);
                ((i, j), r)
            })
            .collect();

        let mut corr = Array2::zeros((p, p));
        for i in 0..p {
            corr[[i, i]] = 1.0;
        }
        for ((i, j), r) in correlations {
            corr[[i, j]] = r;
            corr[[j, i]] = r;
        }
        Ok(corr)
    } else {
        // Sequential for small datasets
        let mut corr = Array2::zeros((p, p));
        for i in 0..p {
            corr[[i, i]] = 1.0;
            for j in (i + 1)..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
                }
                let r = sum / ((n - 1) as f64 * stds[i] * stds[j]);
                corr[[i, j]] = r;
                corr[[j, i]] = r;
            }
        }
        Ok(corr)
    }
}

/// Run factor analysis from Dataset.
///
/// Missing values (null or NaN) are refused with `EconError::NullValues` naming the column,
/// as R's `factanal` refuses NA; integer, decimal and boolean columns are accepted, a text
/// column is `NonNumericColumn` and an infinite value `NonFiniteValues`.
pub fn run_factanal(
    dataset: &crate::data::Dataset,
    columns: &[&str],
    n_factors: usize,
    rotation: RotationMethod,
    scores: ScoresMethod,
) -> EconResult<FactorAnalysisResult> {
    let df = dataset.df();
    if df.height() == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Missing values are refused, as R's factanal() refuses NA.
    let data = ColumnRequest::new(df)
        .numerics(columns)
        .na(NaPolicy::Fail)
        .extract()?
        .matrix(columns, false)?
        .data;

    let mut result = factanal(&data.view(), n_factors, rotation, scores)?;
    result.var_names = Some(columns.iter().map(|s| s.to_string()).collect());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use p2a_validation::{Case, Check, Tol};

    fn create_test_data() -> Array2<f64> {
        // Create data with known factor structure
        // 2 underlying factors, 6 observed variables
        // Factor 1 loads on vars 1-3, Factor 2 loads on vars 4-6
        let n = 200;
        let mut data = Array2::zeros((n, 6));

        // Use fixed seed for reproducibility
        let mut rng_state = 42u64;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((*state >> 16) & 0x7fff) as f64 / 32768.0 - 0.5
        };
        let next_normal = |state: &mut u64| -> f64 {
            // Box-Muller transform
            let u1 = (next_rand(state) + 0.5).max(1e-10);
            let u2 = next_rand(state) + 0.5;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        for i in 0..n {
            let f1 = next_normal(&mut rng_state);
            let f2 = next_normal(&mut rng_state);

            // Variables 1-3 load on factor 1
            data[[i, 0]] = 0.8 * f1 + 0.3 * next_normal(&mut rng_state);
            data[[i, 1]] = 0.7 * f1 + 0.4 * next_normal(&mut rng_state);
            data[[i, 2]] = 0.75 * f1 + 0.35 * next_normal(&mut rng_state);

            // Variables 4-6 load on factor 2
            data[[i, 3]] = 0.8 * f2 + 0.3 * next_normal(&mut rng_state);
            data[[i, 4]] = 0.7 * f2 + 0.4 * next_normal(&mut rng_state);
            data[[i, 5]] = 0.75 * f2 + 0.35 * next_normal(&mut rng_state);
        }

        data
    }

    #[test]
    fn test_factanal_basic() {
        let data = create_test_data();
        let result =
            factanal(&data.view(), 2, RotationMethod::Varimax, ScoresMethod::None).unwrap();

        assert_eq!(result.n_factors, 2);
        assert_eq!(result.n_vars, 6);
        assert_eq!(result.loadings.dim(), (6, 2));
        assert_eq!(result.uniquenesses.len(), 6);
        assert!(result.converged);
    }

    #[test]
    fn test_factanal_communalities() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::None, ScoresMethod::None).unwrap();

        // Communalities should be between 0 and 1
        for &h2 in result.communalities.iter() {
            assert!(
                (0.0..=1.0).contains(&h2),
                "Communality out of range: {}",
                h2
            );
        }

        // Communalities + uniquenesses should sum to 1
        for i in 0..6 {
            let sum = result.communalities[i] + result.uniquenesses[i];
            assert!(
                (sum - 1.0).abs() < 0.1,
                "Communality + uniqueness != 1: {}",
                sum
            );
        }
    }

    #[test]
    fn test_factanal_with_scores() {
        let data = create_test_data();
        let result = factanal(
            &data.view(),
            2,
            RotationMethod::Varimax,
            ScoresMethod::Regression,
        )
        .unwrap();

        assert!(result.scores.is_some());
        let scores = result.scores.unwrap();
        assert_eq!(scores.dim(), (200, 2));
    }

    #[test]
    fn test_varimax_rotation() {
        let loadings = array![
            [0.7, 0.3],
            [0.6, 0.4],
            [0.65, 0.35],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.35, 0.65]
        ];

        let (rotated, t) = varimax(&loadings).unwrap();

        // Rotated loadings have the same communalities; T is orthogonal
        for i in 0..6 {
            let orig_h2: f64 = (0..2).map(|j| loadings[[i, j]].powi(2)).sum();
            let rot_h2: f64 = (0..2).map(|j| rotated[[i, j]].powi(2)).sum();
            assert!(
                (orig_h2 - rot_h2).abs() < 1e-12,
                "Communality changed after rotation"
            );
        }
        let tt = t.t().dot(&t);
        for a in 0..2 {
            for b in 0..2 {
                let e = if a == b { 1.0 } else { 0.0 };
                assert!((tt[[a, b]] - e).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_factanal_promax() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::Promax, ScoresMethod::None).unwrap();

        let phi = result.factor_correlation.unwrap();
        assert_eq!(phi.dim(), (2, 2));
        // a correlation matrix: unit diagonal, symmetric
        for a in 0..2 {
            assert!((phi[[a, a]] - 1.0).abs() < 1e-12, "{phi:?}");
            for b in 0..2 {
                assert!((phi[[a, b]] - phi[[b, a]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_factanal_chi_squared() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::None, ScoresMethod::None).unwrap();

        // Chi-squared should be non-negative
        assert!(result.chi_squared >= 0.0);

        // P-value should be between 0 and 1
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);

        // Degrees of freedom for 6 vars, 2 factors: ((6-2)^2 - 6 - 2)/2 = (16-8)/2 = 4
        assert_eq!(result.df, 4);
    }

    #[test]
    fn test_factanal_too_many_factors() {
        let data = create_test_data();
        let result = factanal(&data.view(), 10, RotationMethod::None, ScoresMethod::None);
        assert!(result.is_err());
    }

    /// Validation test against R's factanal function.
    ///
    /// R code to generate expected values:
    /// ```r
    /// set.seed(42)
    /// n <- 100
    /// f1 <- rnorm(n)
    /// f2 <- rnorm(n)
    /// data <- data.frame(
    ///   x1 = 0.9*f1 + 0.1*rnorm(n),
    ///   x2 = 0.8*f1 + 0.2*rnorm(n),
    ///   x3 = 0.85*f1 + 0.15*rnorm(n),
    ///   x4 = 0.9*f2 + 0.1*rnorm(n),
    ///   x5 = 0.8*f2 + 0.2*rnorm(n),
    ///   x6 = 0.85*f2 + 0.15*rnorm(n)
    /// )
    /// result <- factanal(data, factors = 2, rotation = "none")
    /// print(result$uniquenesses)
    /// print(result$loadings)
    /// ```
    fn rows(case: &Case, key: &str, k: usize) -> Array2<f64> {
        let v = case.vector(key);
        Array2::from_shape_vec((v.len() / k, k), v).unwrap()
    }

    fn flat(a: &Array2<f64>) -> Vec<f64> {
        a.as_standard_layout().iter().copied().collect()
    }

    fn factanal_case(id: &str) -> (Case, Array2<f64>) {
        let (case, ds) = crate::test_support::load_case(id);
        let cols: Vec<Vec<f64>> = (1..=12)
            .map(|j| {
                crate::data::columns::numeric_column(
                    ds.df(),
                    &format!("v{j}"),
                    crate::data::columns::NaPolicy::Fail,
                )
                .unwrap()
                .0
            })
            .collect();
        let n = cols[0].len();
        (case, Array2::from_shape_fn((n, 12), |(i, j)| cols[j][i]))
    }

    /// The normalised varimax criterion (the case's `vcrit`).
    fn vcrit(l: &Array2<f64>) -> f64 {
        let p = l.nrows() as f64;
        let mut z = l.clone();
        for mut row in z.rows_mut() {
            let s = row.dot(&row).sqrt();
            row.mapv_inplace(|v| (v / s).powi(2));
        }
        z.columns()
            .into_iter()
            .map(|c| c.dot(&c) - c.sum().powi(2) / p)
            .sum()
    }

    #[test]
    fn test_validate_factanal_rotation_promax() {
        // Rotation alone, on R's own unrotated loadings: R's varimax at its optimum
        // (eps = 1e-14) and promax's four lines (m = 4) on it, after sortLoadings' order and
        // signs. R's default varimax iterate (eps 1e-5, keys rot_varimax etc.) is not read.
        let case = Case::load("stat_factanal_promax");
        let l0 = rows(&case, "rot_input", 3);
        let (vm, vt) = varimax(&l0).unwrap();
        let (pm, pt) = promax(&l0, 4.0).unwrap();
        let (vo, vs) = sort_order(&vm);
        let (po, ps) = sort_order(&pm);
        let pt_sorted = permute_columns(&pt, &po, &ps);
        let phi = factor_correlation(&pt_sorted).unwrap();
        // the sorted phi equals sort_loadings' D P' phi P D of the unsorted one
        let (_, phi_sorted) = sort_loadings(pm.clone(), Some(factor_correlation(&pt).unwrap()));
        let phi_sorted = phi_sorted.unwrap();
        for (a, b) in phi.iter().zip(phi_sorted.iter()) {
            assert!((a - b).abs() < 1e-12, "{phi:?} vs {phi_sorted:?}");
        }
        let rot = Tol::iterative(1e-8);
        Check::new(&case)
            .vector(
                "rot_varimax_tight",
                &flat(&permute_columns(&vm, &vo, &vs)),
                rot,
            )
            .vector(
                "rot_varimax_rotmat_tight",
                &flat(&permute_columns(&vt, &vo, &vs)),
                rot,
            )
            .vector(
                "rot_promax_tight",
                &flat(&permute_columns(&pm, &po, &ps)),
                rot,
            )
            .vector("rot_promax_rotmat_tight", &flat(&pt_sorted), rot)
            .vector("rot_promax_phi_tight", &flat(&phi), rot)
            .finish();
        // not a different local optimum
        let tight = case.scalar("rot_varimax_criterion_tight");
        assert!(vcrit(&vm) >= tight - 1e-12, "{} < {tight}", vcrit(&vm));
    }

    #[test]
    fn test_validate_factanal_against_r() {
        // factanal(X, 3, rotation = ..., control = list(opt = list(factr = 1, pgtol = 0)));
        // p2a's fixed point with tolerance 1e-12. Iterative on both sides: 1e-4 relative
        // (STAT-62's rule; R's default factr moves its own uniquenesses by 7.2e-6, and R's
        // varimax inside factanal stops at eps 1e-5).
        let (case, x) = factanal_case("stat_factanal_promax");
        let fa = Tol::iterative(1e-4);
        let mut c = Check::new(&case);
        for (rot, key) in [
            (RotationMethod::None, "none"),
            (RotationMethod::Varimax, "varimax"),
            (RotationMethod::Promax, "promax"),
        ] {
            let mut cfg = FactorAnalysisConfig::default();
            cfg.n_factors = 3;
            cfg.rotation = rot;
            cfg.tolerance = 1e-12;
            cfg.max_iter = 100_000;
            let r = factanal_with_config(&x.view(), &cfg).unwrap();
            assert!(r.converged, "{key}");
            // deviation STAT-23: R's factanal stops varimax at eps 1e-5, up to 7.1e-4 from the
            // varimax optimum (rot_varimax vs rot_varimax_tight), so loadings_varimax,
            // loadings_promax, prop_var_varimax, prop_var_promax and promax_phi (R's default
            // iterate) are not compared; the rotated fits are compared with R's rotations at the
            // optimum (rot_*_tight, rotations of R's tight unrotated loadings).
            let reference = match rot {
                RotationMethod::None => "loadings_none".to_string(),
                _ => format!("rot_{key}_tight"),
            };
            c.vector(&reference, &flat(&r.loadings), fa)
                .vector("uniquenesses", &r.uniquenesses.to_vec(), fa)
                // diag(Lambda Phi Lambda') = rowSums(L0^2), not rowSums(pattern^2)
                .vector("communalities", &r.communalities.to_vec(), fa)
                .scalar("statistic", r.chi_squared, fa)
                .scalar("pval", r.p_value, fa)
                .scalar("objective", r.objective, fa)
                .scalar("dof", r.df as f64, Tol::exact());
            if rot == RotationMethod::None {
                c.vector("prop_var_none", &r.variance_proportions.to_vec(), fa);
            } else {
                // SS loadings / p of R's rotation at the optimum
                let lr = rows(&case, &reference, 3);
                for (j, &v) in r.variance_proportions.iter().enumerate() {
                    let col = lr.column(j);
                    let expected = col.dot(&col) / 12.0;
                    assert!(
                        fa.allows(v, expected),
                        "{key} prop_var[{j}] {v} vs {expected}"
                    );
                }
            }
            match rot {
                RotationMethod::Promax => {
                    c.vector(
                        "rot_promax_phi_tight",
                        &flat(r.factor_correlation.as_ref().unwrap()),
                        fa,
                    );
                }
                _ => assert!(r.factor_correlation.is_none()),
            }
        }
        c.finish();
    }

    fn generated(name: &str) -> crate::data::Dataset {
        let p = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        crate::data::Dataset::new(crate::data::DataLoader::load_csv(p).unwrap())
    }

    #[test]
    fn test_rotation_and_scores_methods_parse_aliases_and_reject_unknown() {
        RotationMethod::choices().validate().unwrap();
        ScoresMethod::choices().validate().unwrap();
        assert_eq!(
            "Promax".parse::<RotationMethod>().unwrap(),
            RotationMethod::Promax
        );
        assert_eq!(
            "thomson".parse::<ScoresMethod>().unwrap(),
            ScoresMethod::Regression
        );
        assert!(
            "oblimin"
                .parse::<RotationMethod>()
                .unwrap_err()
                .to_string()
                .contains("varimax")
        );
        assert_eq!(RotationMethod::Promax.to_string(), "promax");
    }

    #[test]
    fn test_na_factanal_columns() {
        use crate::data::Dataset;
        use polars::prelude::*;
        // year is Int64
        let path = p2a_validation::repo_path("validation/datasets/grunfeld.csv");
        let ds = Dataset::new(crate::data::DataLoader::load_csv(path).unwrap());
        let mut df = ds.df().clone();
        let year_f = df
            .column("year")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name("year_f".into());
        df.with_column(year_f).unwrap();
        let ds2 = Dataset::new(df);
        let run = |c: &str| {
            run_factanal(
                &ds2,
                &["inv", "value", "capital", c],
                1,
                RotationMethod::None,
                ScoresMethod::None,
            )
        };
        let (i, f) = (run("year"), run("year_f"));
        assert!(
            !matches!(i, Err(EconError::NonNumericColumn { .. })),
            "{i:?}"
        );
        match (i, f) {
            (Ok(a), Ok(b)) => {
                assert!(
                    (a.chi_squared - b.chi_squared).abs() <= 1e-9 * b.chi_squared.abs().max(1.0)
                );
                for (x, y) in a.uniquenesses.iter().zip(b.uniquenesses.iter()) {
                    assert!((x - y).abs() <= 1e-9);
                }
            }
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            (a, b) => panic!("Int64 and Float64 columns disagree: {a:?} vs {b:?}"),
        }
        let pm = generated("pairwise_mahalanobis_na");
        assert!(matches!(
            run_factanal(&pm, &["a", "b", "id"], 1, RotationMethod::None, ScoresMethod::None),
            Err(EconError::NullValues { column, count: 1 }) if column == "a"
        ));
        // a NaN cell is missing too (inline fixture; the harness cannot write NaN)
        let nan = Dataset::new(
            df!(
                "u" => [1.0, 2.1, f64::NAN, 3.9, 5.2, 6.1],
                "v" => [2.0, 2.9, 4.2, 5.1, 5.8, 7.2],
                "w" => [0.5, 1.7, 2.2, 2.8, 4.1, 4.4]
            )
            .unwrap(),
        );
        assert!(matches!(
            run_factanal(&nan, &["u", "v", "w"], 1, RotationMethod::None, ScoresMethod::None),
            Err(EconError::NullValues { column, count: 1 }) if column == "u"
        ));
    }
}
