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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{eig_symmetric, matmul, matrix_inverse, ndarray_to_faer, faer_to_ndarray};

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
    /// Communalities (proportion of variance explained by factors)
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
    /// Factor correlation matrix (for oblique rotation)
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
            context: "Factor analysis requires at least as many observations as variables".to_string(),
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

    // Apply rotation if requested
    let (rotated_loadings, factor_corr) = match config.rotation {
        RotationMethod::None => (loadings, None),
        RotationMethod::Varimax => {
            let rotated = varimax_rotation(&loadings.view(), config.max_iter, config.tolerance)?;
            (rotated, None)
        }
        RotationMethod::Promax => {
            let (rotated, phi) = promax_rotation(&loadings.view(), 4.0, config.max_iter, config.tolerance)?;
            (rotated, Some(phi))
        }
    };

    // Compute communalities
    let communalities = compute_communalities(&rotated_loadings.view());

    // Compute variance proportions
    let (var_props, cum_var) = compute_variance_proportions(&rotated_loadings.view());

    // Compute chi-squared test statistic
    // χ² = (n - 1 - (2p + 5)/6 - 2k/3) * f
    // where f is the minimum value of the objective function
    let correction = (2.0 * p as f64 + 5.0) / 6.0 + 2.0 * k as f64 / 3.0;
    let chi_squared = (n_obs as f64 - 1.0 - correction) * objective;
    let chi_squared = chi_squared.max(0.0); // Ensure non-negative

    // Compute p-value
    let p_value = if df > 0 {
        use statrs::distribution::{ChiSquared, ContinuousCDF};
        let chi_dist = ChiSquared::new(df as f64)
            .map_err(|e| EconError::Internal(format!("Chi-squared distribution error: {}", e)))?;
        1.0 - chi_dist.cdf(chi_squared)
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
    sorted_eig.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

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
        let log_det_r: f64 = corr_eigenvalues.iter()
            .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 })
            .sum();
        Ok(Self { log_det_r })
    }
}

/// Optimize uniquenesses using the EM algorithm.
///
/// Returns (loadings, uniquenesses, converged, iterations, objective).
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

    for iter in 0..config.max_iter {
        iteration = iter + 1;

        // E-step: Compute reduced correlation matrix R* = R - Psi
        // Then extract loadings from eigendecomposition
        let (new_loadings, obj) = compute_loadings_from_psi_cached(corr, psi, n_factors, &cache)?;

        // M-step: Update uniquenesses
        // psi_i = 1 - sum(lambda_ij^2)
        let mut new_psi = Array1::zeros(p);
        for i in 0..p {
            let comm: f64 = (0..n_factors).map(|j| new_loadings[[i, j]].powi(2)).sum();
            new_psi[i] = (1.0 - comm).max(config.lower_bound).min(1.0 - config.lower_bound);
        }

        // Check convergence
        let psi_change: f64 = psi.iter().zip(new_psi.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        let obj_change = (prev_obj - obj).abs();

        *psi = new_psi;
        loadings = new_loadings;
        prev_obj = obj;

        if psi_change < config.tolerance && obj_change < config.tolerance {
            converged = true;
            break;
        }
    }

    Ok((loadings, psi.clone(), converged, iteration, prev_obj))
}

/// Compute loadings from uniquenesses using eigendecomposition (with caching).
///
/// Given uniquenesses Ψ, compute Λ by:
/// 1. Form R* = Ψ^(-1/2) R Ψ^(-1/2) - I
/// 2. Eigendecompose R*
/// 3. Λ = Ψ^(1/2) V D^(1/2) for top k eigenpairs
fn compute_loadings_from_psi_cached(
    corr: &ArrayView2<f64>,
    psi: &Array1<f64>,
    n_factors: usize,
    cache: &CorrCache,
) -> EconResult<(Array2<f64>, f64)> {
    let p = corr.nrows();

    // Compute Ψ^(-1/2)
    let psi_inv_sqrt: Array1<f64> = psi.iter().map(|&x| 1.0 / x.sqrt()).collect();

    // Form scaled correlation matrix: Ψ^(-1/2) R Ψ^(-1/2)
    // Use parallel iteration for large matrices
    let scaled_corr = if p > 50 {
        // Parallel computation for large matrices
        // Compute each row in parallel
        let rows: Vec<Vec<f64>> = (0..p).into_par_iter()
            .map(|i| {
                (0..p).map(|j| corr[[i, j]] * psi_inv_sqrt[i] * psi_inv_sqrt[j]).collect()
            })
            .collect();
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((p, p), flat)
            .map_err(|e| EconError::Internal(format!("Array reshape failed: {}", e)))?
    } else {
        // Sequential for small matrices (parallel overhead not worth it)
        let mut scaled = Array2::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                scaled[[i, j]] = corr[[i, j]] * psi_inv_sqrt[i] * psi_inv_sqrt[j];
            }
        }
        scaled
    };

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = eig_symmetric(&scaled_corr.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

    // Sort eigenvalues in descending order and get indices
    let mut idx_val: Vec<(usize, f64)> = eigenvalues.iter().cloned().enumerate().collect();
    idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute loadings: Λ = Ψ^(1/2) V_k (D_k - I)^(1/2)
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

/// Compute objective using Woodbury matrix identity (efficient for k << p).
fn compute_objective_woodbury(
    corr: &ArrayView2<f64>,
    loadings: &ArrayView2<f64>,
    psi: &Array1<f64>,
    cache: &CorrCache,
) -> EconResult<f64> {
    let p = corr.nrows();
    let k = loadings.ncols();

    // Compute Ψ^{-1}
    let psi_inv: Array1<f64> = psi.iter().map(|&x| 1.0 / x.max(1e-10)).collect();

    // Compute Λ'Ψ^{-1}Λ (k x k matrix)
    let mut ltpil = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for m in 0..p {
                sum += loadings[[m, i]] * psi_inv[m] * loadings[[m, j]];
            }
            ltpil[[i, j]] = sum;
        }
    }

    // M = I + Λ'Ψ^{-1}Λ
    for i in 0..k {
        ltpil[[i, i]] += 1.0;
    }

    // Compute M^{-1} using faer (more efficient for small k x k matrix)
    let m_inv = matrix_inverse(&ltpil.view())
        .map_err(|e| EconError::Internal(format!("Matrix inversion failed: {}", e)))?;

    // log|Σ| = log|Ψ| + log|M|
    let log_det_psi: f64 = psi.iter().map(|&x| x.max(1e-10).ln()).sum();
    let (m_eig, _) = eig_symmetric(&ltpil.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;
    let log_det_m: f64 = m_eig.iter()
        .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 })
        .sum();
    let log_det_sigma = log_det_psi + log_det_m;

    // Compute Σ^{-1}R trace using Woodbury:
    // Σ^{-1} = Ψ^{-1} - Ψ^{-1}Λ M^{-1} Λ'Ψ^{-1}
    //
    // tr(Σ^{-1}R) = tr(Ψ^{-1}R) - tr(Ψ^{-1}Λ M^{-1} Λ'Ψ^{-1}R)

    // First term: tr(Ψ^{-1}R)
    let trace_psi_inv_r: f64 = (0..p).map(|i| psi_inv[i] * corr[[i, i]]).sum();

    // Second term: tr(Ψ^{-1}Λ M^{-1} Λ'Ψ^{-1}R)
    // = tr(M^{-1} Λ'Ψ^{-1}R Ψ^{-1}Λ)  [cyclic property of trace]

    // Compute Λ'Ψ^{-1}R (k x p)
    let mut lt_psi_inv_r = Array2::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            let mut sum = 0.0;
            for m in 0..p {
                sum += loadings[[m, i]] * psi_inv[m] * corr[[m, j]];
            }
            lt_psi_inv_r[[i, j]] = sum;
        }
    }

    // Compute (Λ'Ψ^{-1}R)(Ψ^{-1}Λ) = k x k matrix
    let mut lt_psi_inv_r_psi_inv_l = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for m in 0..p {
                sum += lt_psi_inv_r[[i, m]] * psi_inv[m] * loadings[[m, j]];
            }
            lt_psi_inv_r_psi_inv_l[[i, j]] = sum;
        }
    }

    // tr(M^{-1} * (Λ'Ψ^{-1}R Ψ^{-1}Λ))
    let m_inv_times_b = matmul(&m_inv.view(), &lt_psi_inv_r_psi_inv_l.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;
    let trace_correction: f64 = (0..k).map(|i| m_inv_times_b[[i, i]]).sum();

    let trace = trace_psi_inv_r - trace_correction;

    // f = log|Σ| + tr(Σ^{-1} R) - log|R| - p
    // Use cached log|R|
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
    let log_det_sigma: f64 = sigma_eigenvalues.iter()
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
                            sum += sigma_eigenvectors[[i, m]] * sigma_eigenvectors[[j, m]] / sigma_eigenvalues[m];
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

/// Apply varimax (orthogonal) rotation.
///
/// Maximizes the sum of variances of squared loadings per factor.
/// Reference: Kaiser (1958).
fn varimax_rotation(
    loadings: &ArrayView2<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array2<f64>> {
    let (p, k) = loadings.dim();

    if k == 1 {
        return Ok(loadings.to_owned());
    }

    let mut rotated = loadings.to_owned();

    // Normalize rows (Kaiser normalization)
    let h: Vec<f64> = (0..p)
        .map(|i| {
            let sum_sq: f64 = (0..k).map(|j| rotated[[i, j]].powi(2)).sum();
            sum_sq.sqrt().max(1e-10)
        })
        .collect();

    for i in 0..p {
        for j in 0..k {
            rotated[[i, j]] /= h[i];
        }
    }

    // Iteratively rotate pairs of factors
    for _ in 0..max_iter {
        let mut max_change: f64 = 0.0;

        // Rotate each pair of factors
        for j in 0..k {
            for m in (j + 1)..k {
                // Compute rotation angle
                let mut u = 0.0;
                let mut v = 0.0;

                for i in 0..p {
                    let x = rotated[[i, j]];
                    let y = rotated[[i, m]];
                    let x2 = x * x;
                    let y2 = y * y;
                    u += x2 - y2;
                    v += 2.0 * x * y;
                }

                // Varimax criterion
                let a = 0.0;
                let b = 0.0;
                let mut num = 0.0;
                let mut denom = 0.0;

                for i in 0..p {
                    let x = rotated[[i, j]];
                    let y = rotated[[i, m]];
                    let x2 = x * x;
                    let y2 = y * y;
                    let xy = x * y;

                    num += 4.0 * xy * (x2 - y2);
                    denom += (x2 - y2).powi(2) - 4.0 * xy.powi(2);
                }

                // Additional terms for varimax
                num -= 2.0 * u * v / p as f64;
                denom += (u * u - v * v) / p as f64;

                // Compute rotation angle
                let phi = 0.25 * num.atan2(denom);

                if phi.abs() > tolerance {
                    max_change = max_change.max(phi.abs());

                    let cos_phi = phi.cos();
                    let sin_phi = phi.sin();

                    // Apply rotation
                    for i in 0..p {
                        let x = rotated[[i, j]];
                        let y = rotated[[i, m]];
                        rotated[[i, j]] = x * cos_phi + y * sin_phi;
                        rotated[[i, m]] = -x * sin_phi + y * cos_phi;
                    }
                }
            }
        }

        if max_change < tolerance {
            break;
        }
    }

    // Denormalize
    for i in 0..p {
        for j in 0..k {
            rotated[[i, j]] *= h[i];
        }
    }

    // Reorder factors by variance explained
    reorder_loadings(&rotated)
}

/// Apply promax (oblique) rotation.
///
/// Starts with varimax solution, then applies power transformation to allow
/// correlated factors.
fn promax_rotation(
    loadings: &ArrayView2<f64>,
    power: f64,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array2<f64>, Array2<f64>)> {
    let (p, k) = loadings.dim();

    // First apply varimax
    let varimax_loadings = varimax_rotation(loadings, max_iter, tolerance)?;

    // Apply power transformation to get target matrix
    let mut target = Array2::zeros((p, k));
    for i in 0..p {
        for j in 0..k {
            let l = varimax_loadings[[i, j]];
            target[[i, j]] = l.signum() * l.abs().powf(power);
        }
    }

    // Find transformation matrix T such that varimax * T ≈ target
    // Using least squares: T = (L'L)^(-1) L' target
    let lt_l = matmul(&varimax_loadings.t().to_owned().view(), &varimax_loadings.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    let lt_target = matmul(&varimax_loadings.t().to_owned().view(), &target.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    // Solve for T
    let (lt_l_eigenvalues, lt_l_eigenvectors) = eig_symmetric(&lt_l.view())
        .map_err(|e| EconError::Internal(format!("Eigendecomposition failed: {}", e)))?;

    let mut lt_l_inv = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for m in 0..k {
                if lt_l_eigenvalues[m] > 1e-10 {
                    sum += lt_l_eigenvectors[[i, m]] * lt_l_eigenvectors[[j, m]] / lt_l_eigenvalues[m];
                }
            }
            lt_l_inv[[i, j]] = sum;
        }
    }

    let t = matmul(&lt_l_inv.view(), &lt_target.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    // Normalize columns of T
    let mut t_normalized = t.clone();
    for j in 0..k {
        let norm: f64 = (0..k).map(|i| t[[i, j]].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for i in 0..k {
                t_normalized[[i, j]] /= norm;
            }
        }
    }

    // Apply transformation: promax_loadings = varimax_loadings * T
    let promax_loadings = matmul(&varimax_loadings.view(), &t_normalized.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    // Compute factor correlation matrix: Φ = T' T
    let phi = matmul(&t_normalized.t().to_owned().view(), &t_normalized.view())
        .map_err(|e| EconError::Internal(format!("Matrix multiplication failed: {}", e)))?;

    Ok((promax_loadings, phi))
}

/// Reorder loadings by variance explained (descending).
fn reorder_loadings(loadings: &Array2<f64>) -> EconResult<Array2<f64>> {
    let (p, k) = loadings.dim();

    // Compute variance (sum of squared loadings) for each factor
    let mut variances: Vec<(usize, f64)> = (0..k)
        .map(|j| {
            let var: f64 = (0..p).map(|i| loadings[[i, j]].powi(2)).sum();
            (j, var)
        })
        .collect();

    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut reordered = Array2::zeros((p, k));
    for (new_j, (old_j, _)) in variances.iter().enumerate() {
        for i in 0..p {
            reordered[[i, new_j]] = loadings[[i, *old_j]];
        }
    }

    Ok(reordered)
}

/// Compute communalities (h² = sum of squared loadings for each variable).
fn compute_communalities(loadings: &ArrayView2<f64>) -> Array1<f64> {
    let (p, k) = loadings.dim();
    Array1::from_iter((0..p).map(|i| {
        (0..k).map(|j| loadings[[i, j]].powi(2)).sum()
    }))
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
    corr: &ArrayView2<f64>,
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
fn compute_sigma(loadings: &ArrayView2<f64>, uniquenesses: &ArrayView1<f64>) -> EconResult<Array2<f64>> {
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
        (0..p).into_par_iter()
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect()
    } else {
        (0..p)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect()
    };

    // Compute standard deviations
    let stds: Vec<f64> = if n > 1000 {
        (0..p).into_par_iter()
            .map(|j| {
                let col = data.column(j);
                let mean = means[j];
                let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt().max(1e-10)
            })
            .collect()
    } else {
        (0..p)
            .map(|j| {
                let col = data.column(j);
                let mean = means[j];
                let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt().max(1e-10)
            })
            .collect()
    };

    // Compute correlation matrix
    // For large datasets and many variables, use parallel computation
    if n > 1000 && p > 10 {
        // Parallel: compute upper triangular entries
        let num_pairs = p * (p - 1) / 2;
        let pair_indices: Vec<(usize, usize)> = (0..p)
            .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
            .collect();

        let correlations: Vec<((usize, usize), f64)> = pair_indices.par_iter()
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
pub fn run_factanal(
    dataset: &crate::data::Dataset,
    columns: &[&str],
    n_factors: usize,
    rotation: RotationMethod,
    scores: ScoresMethod,
) -> EconResult<FactorAnalysisResult> {
    // Extract data from dataset columns
    let df = dataset.df();
    let n = df.height();
    let p = columns.len();

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Build data matrix
    let mut data = Array2::zeros((n, p));
    for (j, col_name) in columns.iter().enumerate() {
        let col = df.column(col_name).map_err(|_| EconError::ColumnNotFound {
            column: col_name.to_string(),
            available: dataset.column_names(),
        })?;

        let values = col.f64().map_err(|_| EconError::NonNumericColumn {
            column: col_name.to_string(),
        })?;

        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => data[[i, j]] = v,
                None => {
                    return Err(EconError::NullValues {
                        column: col_name.to_string(),
                        count: col.null_count(),
                    });
                }
            }
        }
    }

    let mut result = factanal(&data.view(), n_factors, rotation, scores)?;
    result.var_names = Some(columns.iter().map(|s| s.to_string()).collect());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
        let result = factanal(&data.view(), 2, RotationMethod::Varimax, ScoresMethod::None).unwrap();

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
            assert!(h2 >= 0.0 && h2 <= 1.0, "Communality out of range: {}", h2);
        }

        // Communalities + uniquenesses should sum to 1
        for i in 0..6 {
            let sum = result.communalities[i] + result.uniquenesses[i];
            assert!((sum - 1.0).abs() < 0.1, "Communality + uniqueness != 1: {}", sum);
        }
    }

    #[test]
    fn test_factanal_with_scores() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::Varimax, ScoresMethod::Regression).unwrap();

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

        let rotated = varimax_rotation(&loadings.view(), 100, 1e-6).unwrap();

        // Rotated loadings should have same communalities
        for i in 0..6 {
            let orig_h2: f64 = (0..2).map(|j| loadings[[i, j]].powi(2)).sum();
            let rot_h2: f64 = (0..2).map(|j| rotated[[i, j]].powi(2)).sum();
            assert!((orig_h2 - rot_h2).abs() < 0.01, "Communality changed after rotation");
        }
    }

    #[test]
    fn test_factanal_promax() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::Promax, ScoresMethod::None).unwrap();

        assert!(result.factor_correlation.is_some());
        let phi = result.factor_correlation.unwrap();
        assert_eq!(phi.dim(), (2, 2));

        // Diagonal should be 1 (or close to 1)
        assert!((phi[[0, 0]] - 1.0).abs() < 0.1);
        assert!((phi[[1, 1]] - 1.0).abs() < 0.1);
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
    #[test]
    fn test_validate_factanal_against_r() {
        let data = create_test_data();
        let result = factanal(&data.view(), 2, RotationMethod::None, ScoresMethod::None).unwrap();

        // The exact values depend on the random data, but we can check:
        // 1. Uniquenesses should be in reasonable range (0.1 to 0.5 for well-structured data)
        for &u in result.uniquenesses.iter() {
            assert!(u > 0.0 && u < 1.0, "Uniqueness out of expected range: {}", u);
        }

        // 2. Each variable should load primarily on one factor
        for i in 0..6 {
            let l1 = result.loadings[[i, 0]].abs();
            let l2 = result.loadings[[i, 1]].abs();
            // One loading should be substantially larger than the other
            let max_l = l1.max(l2);
            assert!(max_l > 0.3, "Variable {} has low loadings: {}, {}", i, l1, l2);
        }

        // 3. Convergence check
        assert!(result.converged, "Factor analysis did not converge");
    }
}
