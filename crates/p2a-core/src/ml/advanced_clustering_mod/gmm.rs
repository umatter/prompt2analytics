//! Gaussian mixture models fitted by EM ([`gaussian_mixture`]); R analogue `mclust`.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// =============================================================================
// Gaussian Mixture Model
// =============================================================================

/// Result of Gaussian Mixture Model clustering.
///
/// # References
///
/// - McLachlan, G.J. and Peel, D. (2000). "Finite Mixture Models". Wiley.
/// - R mclust package documentation
///   Source: https://www.rdocumentation.org/packages/mclust/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixtureResult {
    /// Hard cluster assignments (most likely cluster for each point)
    pub labels: Vec<usize>,
    /// Soft assignments (n x k matrix of responsibilities)
    pub responsibilities: Vec<Vec<f64>>,
    /// Mixture weights (k weights summing to 1)
    pub weights: Vec<f64>,
    /// Component means (k x d matrix)
    pub means: Vec<Vec<f64>>,
    /// Component covariances (k matrices, each d x d)
    pub covariances: Vec<Vec<Vec<f64>>>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// BIC (Bayesian Information Criterion) - lower is better
    pub bic: f64,
    /// AIC (Akaike Information Criterion) - lower is better
    pub aic: f64,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Number of components
    pub n_components: usize,
    /// Number of observations
    pub n: usize,
    /// Number of features
    pub n_features: usize,
}

impl std::fmt::Display for GaussianMixtureResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gaussian Mixture Model Results")?;
        writeln!(f, "==============================")?;
        writeln!(f, "Number of components: {}", self.n_components)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Number of features: {}", self.n_features)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f)?;
        writeln!(f, "Log-likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f)?;
        writeln!(f, "Component weights:")?;
        for (i, w) in self.weights.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i, w)?;
        }

        let mut cluster_counts = vec![0usize; self.n_components];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes (hard assignment):")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }
        Ok(())
    }
}

/// Covariance type for Gaussian mixture models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrix for each component
    Full,
    /// Diagonal covariance (independent features)
    Diagonal,
    /// Spherical (single variance per component)
    Spherical,
    /// Tied (same covariance for all components)
    Tied,
}

impl std::str::FromStr for CovarianceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" => Ok(CovarianceType::Full),
            "diagonal" | "diag" => Ok(CovarianceType::Diagonal),
            "spherical" => Ok(CovarianceType::Spherical),
            "tied" => Ok(CovarianceType::Tied),
            _ => Err(format!("Unknown covariance type: {}", s)),
        }
    }
}

/// Run Gaussian Mixture Model clustering using EM algorithm.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_components` - Number of mixture components
/// * `covariance_type` - Type of covariance parameters (default: Full)
/// * `max_iterations` - Maximum EM iterations (default: 100)
/// * `tolerance` - Convergence tolerance for log-likelihood (default: 1e-4)
/// * `seed` - Optional random seed for initialization
///
/// # Algorithm
///
/// 1. Initialize parameters (k-means initialization)
/// 2. E-step: Compute responsibilities (posterior probabilities)
/// 3. M-step: Update means, covariances, and weights
/// 4. Repeat until convergence
///
/// # Returns
/// * `GaussianMixtureResult` containing cluster assignments and model parameters
///
/// # References
///
/// - McLachlan, G.J. and Peel, D. (2000). "Finite Mixture Models".
pub fn gaussian_mixture(
    data: ArrayView2<f64>,
    n_components: usize,
    covariance_type: Option<CovarianceType>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
) -> Result<GaussianMixtureResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_components == 0 || n_components > n {
        return Err(format!("n_components must be between 1 and {}", n));
    }

    let cov_type = covariance_type.unwrap_or(CovarianceType::Full);
    let max_iter = max_iterations.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-4);

    // Initialize with k-means
    let kmeans_result = crate::ml::kmeans(
        data.view(),
        n_components,
        Some(20),
        Some(1e-4),
        Some(5),
        seed,
    )?;

    // Initialize parameters
    let mut weights = vec![1.0 / n_components as f64; n_components];
    let mut means: Vec<Vec<f64>> = (0..n_components)
        .map(|k| kmeans_result.centroids.row(k).to_vec())
        .collect();

    // Initialize covariances
    let mut covariances: Vec<Array2<f64>> = (0..n_components)
        .map(|k| {
            // Initialize to identity scaled by data variance
            let cluster_points: Vec<usize> = kmeans_result
                .labels
                .iter()
                .enumerate()
                .filter(|&(_, l)| *l == k)
                .map(|(i, _)| i)
                .collect();

            if cluster_points.is_empty() {
                Array2::eye(d)
            } else {
                compute_covariance(&data, &cluster_points, &means[k], cov_type)
            }
        })
        .collect();

    // EM algorithm
    let mut responsibilities = Array2::zeros((n, n_components));
    let mut prev_ll = f64::NEG_INFINITY;
    let mut n_iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // E-step: Compute responsibilities
        let mut log_likelihood = 0.0;
        for i in 0..n {
            let point: Vec<f64> = data.row(i).to_vec();
            let mut log_probs = Vec::with_capacity(n_components);

            for k in 0..n_components {
                let log_prob =
                    log_gaussian_pdf(&point, &means[k], &covariances[k]) + weights[k].ln();
                log_probs.push(log_prob);
            }

            // Log-sum-exp for numerical stability
            let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
            let log_sum = max_log + sum_exp.ln();

            log_likelihood += log_sum;

            for k in 0..n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }

        // Check convergence
        if (log_likelihood - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = log_likelihood;

        // M-step: Update parameters
        let nk: Vec<f64> = (0..n_components)
            .map(|k| responsibilities.column(k).sum())
            .collect();

        // Update weights
        weights = nk.iter().map(|&n_k| n_k / n as f64).collect();

        // Update means
        for k in 0..n_components {
            if nk[k] > 1e-10 {
                means[k] = (0..d)
                    .map(|j| {
                        let sum: f64 = (0..n)
                            .map(|i| responsibilities[[i, k]] * data[[i, j]])
                            .sum();
                        sum / nk[k]
                    })
                    .collect();
            }
        }

        // Update covariances
        for k in 0..n_components {
            if nk[k] > 1e-10 {
                let cluster_points: Vec<(usize, f64)> =
                    (0..n).map(|i| (i, responsibilities[[i, k]])).collect();
                covariances[k] =
                    compute_weighted_covariance(&data, &cluster_points, &means[k], cov_type, nk[k]);
            }
        }
    }

    // Compute final log-likelihood
    let mut log_likelihood = 0.0;
    for i in 0..n {
        let point: Vec<f64> = data.row(i).to_vec();
        let mut log_probs = Vec::with_capacity(n_components);

        for k in 0..n_components {
            let log_prob = log_gaussian_pdf(&point, &means[k], &covariances[k]) + weights[k].ln();
            log_probs.push(log_prob);
        }

        let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
        log_likelihood += max_log + sum_exp.ln();
    }

    // Compute BIC and AIC
    let n_params = match cov_type {
        CovarianceType::Full => n_components * (d + d * (d + 1) / 2) + n_components - 1,
        CovarianceType::Diagonal => n_components * (d + d) + n_components - 1,
        CovarianceType::Spherical => n_components * (d + 1) + n_components - 1,
        CovarianceType::Tied => n_components * d + d * (d + 1) / 2 + n_components - 1,
    };

    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;

    // Hard assignments
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            let row = responsibilities.row(i);
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0)
        })
        .collect();

    // Convert responsibilities to Vec<Vec<f64>>
    let resp_vecs: Vec<Vec<f64>> = (0..n).map(|i| responsibilities.row(i).to_vec()).collect();

    // Convert covariances to Vec<Vec<Vec<f64>>>
    let cov_vecs: Vec<Vec<Vec<f64>>> = covariances
        .iter()
        .map(|cov| (0..d).map(|i| cov.row(i).to_vec()).collect())
        .collect();

    Ok(GaussianMixtureResult {
        labels,
        responsibilities: resp_vecs,
        weights,
        means,
        covariances: cov_vecs,
        log_likelihood,
        bic,
        aic,
        n_iterations,
        converged,
        n_components,
        n,
        n_features: d,
    })
}

/// Compute log probability density of multivariate Gaussian.
fn log_gaussian_pdf(x: &[f64], mean: &[f64], cov: &Array2<f64>) -> f64 {
    let d = x.len();

    // Compute (x - mean)
    let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();

    // Compute determinant and inverse (simplified for diagonal/spherical)
    let det = compute_determinant(cov);

    if det <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Compute (x - mean)^T * Sigma^(-1) * (x - mean)
    let inv = compute_inverse(cov);
    let mut mahal = 0.0;
    for i in 0..d {
        for j in 0..d {
            mahal += diff[i] * inv[[i, j]] * diff[j];
        }
    }

    -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + mahal)
}

/// Compute sample covariance matrix.
fn compute_covariance(
    data: &ArrayView2<f64>,
    indices: &[usize],
    mean: &[f64],
    cov_type: CovarianceType,
) -> Array2<f64> {
    let d = data.ncols();
    let n = indices.len();

    if n < 2 {
        return Array2::eye(d) * 0.01; // Regularization
    }

    match cov_type {
        CovarianceType::Full => {
            let mut cov = Array2::zeros((d, d));
            for &i in indices {
                for j in 0..d {
                    for k in 0..d {
                        cov[[j, k]] += (data[[i, j]] - mean[j]) * (data[[i, k]] - mean[k]);
                    }
                }
            }
            cov /= (n - 1) as f64;
            // Add regularization
            for i in 0..d {
                cov[[i, i]] += 1e-6;
            }
            cov
        }
        CovarianceType::Diagonal => {
            let mut cov = Array2::zeros((d, d));
            for j in 0..d {
                let var: f64 = indices
                    .iter()
                    .map(|&i| (data[[i, j]] - mean[j]).powi(2))
                    .sum::<f64>()
                    / (n - 1) as f64;
                cov[[j, j]] = var.max(1e-6);
            }
            cov
        }
        CovarianceType::Spherical => {
            let var: f64 = indices
                .iter()
                .flat_map(|&i| (0..d).map(move |j| (data[[i, j]] - mean[j]).powi(2)))
                .sum::<f64>()
                / ((n - 1) * d) as f64;
            Array2::eye(d) * var.max(1e-6)
        }
        CovarianceType::Tied => compute_covariance(data, indices, mean, CovarianceType::Full),
    }
}

/// Compute weighted sample covariance matrix.
fn compute_weighted_covariance(
    data: &ArrayView2<f64>,
    weights: &[(usize, f64)], // (index, weight)
    mean: &[f64],
    cov_type: CovarianceType,
    total_weight: f64,
) -> Array2<f64> {
    let d = data.ncols();

    if total_weight < 1e-10 {
        return Array2::eye(d) * 0.01;
    }

    match cov_type {
        CovarianceType::Full => {
            let mut cov = Array2::zeros((d, d));
            for &(i, w) in weights {
                for j in 0..d {
                    for k in 0..d {
                        cov[[j, k]] += w * (data[[i, j]] - mean[j]) * (data[[i, k]] - mean[k]);
                    }
                }
            }
            cov /= total_weight;
            for i in 0..d {
                cov[[i, i]] += 1e-6;
            }
            cov
        }
        CovarianceType::Diagonal => {
            let mut cov = Array2::zeros((d, d));
            for j in 0..d {
                let var: f64 = weights
                    .iter()
                    .map(|&(i, w)| w * (data[[i, j]] - mean[j]).powi(2))
                    .sum::<f64>()
                    / total_weight;
                cov[[j, j]] = var.max(1e-6);
            }
            cov
        }
        CovarianceType::Spherical => {
            let var: f64 = weights
                .iter()
                .flat_map(|&(i, w)| (0..d).map(move |j| w * (data[[i, j]] - mean[j]).powi(2)))
                .sum::<f64>()
                / (total_weight * d as f64);
            Array2::eye(d) * var.max(1e-6)
        }
        CovarianceType::Tied => {
            compute_weighted_covariance(data, weights, mean, CovarianceType::Full, total_weight)
        }
    }
}

/// Simple determinant computation (for small matrices or diagonal).
fn compute_determinant(m: &Array2<f64>) -> f64 {
    let d = m.nrows();

    // Check if diagonal
    let is_diagonal = (0..d).all(|i| (0..d).all(|j| i == j || m[[i, j]].abs() < 1e-10));

    if is_diagonal {
        return (0..d).map(|i| m[[i, i]]).product();
    }

    // LU decomposition for general case
    // Simplified: use product of diagonal for now (assumes well-conditioned)
    if d == 1 {
        return m[[0, 0]];
    }
    if d == 2 {
        return m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    }

    // For larger matrices, use eigenvalues approximation
    // This is a simplification - in production, use proper linear algebra
    let trace: f64 = (0..d).map(|i| m[[i, i]]).sum();
    let avg_diag = trace / d as f64;
    avg_diag.powi(d as i32)
}

/// Simple matrix inverse (for small matrices or diagonal).
fn compute_inverse(m: &Array2<f64>) -> Array2<f64> {
    let d = m.nrows();

    // Check if diagonal
    let is_diagonal = (0..d).all(|i| (0..d).all(|j| i == j || m[[i, j]].abs() < 1e-10));

    if is_diagonal {
        let mut inv = Array2::zeros((d, d));
        for i in 0..d {
            inv[[i, i]] = 1.0 / m[[i, i]].max(1e-10);
        }
        return inv;
    }

    // For 2x2
    if d == 2 {
        let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
        if det.abs() < 1e-10 {
            return Array2::eye(d);
        }
        let mut inv = Array2::zeros((2, 2));
        inv[[0, 0]] = m[[1, 1]] / det;
        inv[[1, 1]] = m[[0, 0]] / det;
        inv[[0, 1]] = -m[[0, 1]] / det;
        inv[[1, 0]] = -m[[1, 0]] / det;
        return inv;
    }

    // For larger matrices, use regularized pseudo-inverse approximation
    // This is a simplification
    let mut inv = Array2::zeros((d, d));
    for i in 0..d {
        inv[[i, i]] = 1.0 / m[[i, i]].max(1e-6);
    }
    inv
}

/// Convenience wrapper for gaussian_mixture.
pub fn run_gaussian_mixture(
    data: ArrayView2<f64>,
    n_components: usize,
    seed: Option<u64>,
) -> Result<GaussianMixtureResult, String> {
    gaussian_mixture(data, n_components, None, None, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_gaussian_mixture_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = gaussian_mixture(data.view(), 2, None, Some(50), None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.weights.len(), 2);
        assert!(result.weights.iter().sum::<f64>() - 1.0 < 1e-6);
    }
}
