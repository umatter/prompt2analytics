//! Spectral clustering on an RBF or k-nearest-neighbour affinity graph ([`spectral_clustering`]).

use crate::options::Choices;
use crate::resource::{checkpoint, try_zeros2};
use ndarray::{Array2, ArrayView2, Axis};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::common::try_distance_matrix;

/// Seed of the k-means step when `seed` is `None`.
pub const SPECTRAL_DEFAULT_SEED: u64 = 42;

// =============================================================================
// Spectral Clustering
// =============================================================================

/// Result of spectral clustering.
///
/// # References
///
/// - Ng, A.Y., Jordan, M.I., and Weiss, Y. (2002). "On Spectral Clustering:
///   Analysis and an algorithm". NIPS 2001.
/// - von Luxburg, U. (2007). "A Tutorial on Spectral Clustering".
///   Statistics and Computing, 17(4), 395-416.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralClusteringResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Eigenvalues of the Laplacian (smallest k)
    pub eigenvalues: Vec<f64>,
    /// Affinity matrix used
    pub affinity_type: String,
    /// Gamma parameter for RBF kernel (if used)
    pub gamma: Option<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for SpectralClusteringResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spectral Clustering Results")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Affinity type: {}", self.affinity_type)?;
        if let Some(gamma) = self.gamma {
            writeln!(f, "Gamma (RBF): {:.4}", gamma)?;
        }
        writeln!(f)?;
        writeln!(f, "Smallest eigenvalues of Laplacian:")?;
        for (i, &ev) in self.eigenvalues.iter().enumerate() {
            writeln!(f, "  λ_{}: {:.6}", i, ev)?;
        }

        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }
        Ok(())
    }
}

/// Affinity matrix type for spectral clustering.
#[derive(Debug, Clone, Copy)]
pub enum AffinityType {
    /// RBF (Gaussian) kernel: exp(-gamma * ||x - y||^2)
    Rbf { gamma: f64 },
    /// k-nearest neighbors graph
    Knn { k: usize },
    /// Precomputed affinity matrix
    Precomputed,
}

/// The affinity names `AffinityType` parses (its payloads get their defaults).
#[derive(Clone, Copy, PartialEq)]
enum AffinityKind {
    Rbf,
    Knn,
    Precomputed,
}

static AFFINITY_CHOICES: Choices<AffinityKind> = Choices {
    option: "affinity type",
    values: &[
        (&["rbf", "gaussian"], AffinityKind::Rbf),
        (&["knn", "nearest_neighbors"], AffinityKind::Knn),
        (&["precomputed"], AffinityKind::Precomputed),
    ],
};

impl std::str::FromStr for AffinityType {
    type Err = crate::errors::EconError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match AFFINITY_CHOICES.parse(s)? {
            AffinityKind::Rbf => AffinityType::Rbf { gamma: 1.0 },
            AffinityKind::Knn => AffinityType::Knn { k: 10 },
            AffinityKind::Precomputed => AffinityType::Precomputed,
        })
    }
}

/// Run spectral clustering.
///
/// Spectral clustering uses the eigenvalues of the graph Laplacian to perform
/// dimensionality reduction before clustering in fewer dimensions.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `affinity` - Affinity matrix type (default: RBF with auto gamma)
/// * `seed` - Seed of the k-means step (default: [`SPECTRAL_DEFAULT_SEED`])
///
/// # Algorithm
///
/// 1. Construct affinity matrix A from data
/// 2. Compute normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
/// 3. Find k smallest eigenvectors of I - L (or largest of L)
/// 4. Form matrix U from eigenvectors
/// 5. Normalize rows of U
/// 6. Cluster rows with k-means
///
/// # Returns
/// * `SpectralClusteringResult` containing cluster assignments
///
/// # References
///
/// - Ng, A.Y., Jordan, M.I., and Weiss, Y. (2002). "On Spectral Clustering".
pub fn spectral_clustering(
    data: ArrayView2<f64>,
    n_clusters: usize,
    affinity: Option<AffinityType>,
    seed: Option<u64>,
) -> Result<SpectralClusteringResult, String> {
    let n = data.nrows();

    if n_clusters == 0 || n_clusters > n {
        return Err(format!(
            "n_clusters must be between 1 and {} (n_samples)",
            n
        ));
    }

    crate::ml::extract::require_finite(&data, "spectral clustering")?;
    match affinity {
        Some(AffinityType::Rbf { gamma }) if !gamma.is_finite() || gamma <= 0.0 => {
            return Err(format!("gamma must be finite and positive, got {gamma}"));
        }
        Some(AffinityType::Knn { k: 0 }) => {
            return Err("knn affinity: k must be at least 1, got 0".to_string());
        }
        _ => {}
    }

    // Default affinity: RBF with auto gamma
    let affinity_type = affinity.unwrap_or_else(|| {
        // Auto-compute gamma as 1 / (n_features * variance)
        let variance = compute_data_variance(&data);
        let gamma = 1.0 / (data.ncols() as f64 * variance.max(1e-10));
        AffinityType::Rbf { gamma }
    });

    let (gamma_val, affinity_name) = match affinity_type {
        AffinityType::Rbf { gamma } => (Some(gamma), "rbf".to_string()),
        AffinityType::Knn { k } => (None, format!("knn(k={})", k)),
        AffinityType::Precomputed => (None, "precomputed".to_string()),
    };

    // Build affinity matrix
    let affinity_matrix = match affinity_type {
        AffinityType::Rbf { gamma } => build_rbf_affinity(&data, gamma)?,
        AffinityType::Knn { k } => build_knn_affinity(&data, k)?,
        AffinityType::Precomputed => {
            // Assume data IS the affinity matrix
            if data.nrows() != data.ncols() {
                return Err("Precomputed affinity must be square".to_string());
            }
            data.to_owned()
        }
    };

    // Compute normalized Laplacian and its eigenvectors
    let (eigenvalues, eigenvectors) = compute_laplacian_eigenvectors(&affinity_matrix, n_clusters)?;

    // Cluster the eigenvector embeddings with k-means
    let labels = cluster_eigenvectors(
        &eigenvectors,
        n_clusters,
        Some(seed.unwrap_or(SPECTRAL_DEFAULT_SEED)),
    )?;

    Ok(SpectralClusteringResult {
        labels,
        eigenvalues,
        affinity_type: affinity_name,
        gamma: gamma_val,
        n_clusters,
        n,
    })
}

/// Build RBF affinity matrix.
fn build_rbf_affinity(data: &ArrayView2<f64>, gamma: f64) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let mut affinity =
        try_zeros2(n, n, "spectral clustering affinity").map_err(|e| e.to_string())?;

    for i in 0..n {
        for j in i..n {
            if i == j {
                affinity[[i, j]] = 0.0; // No self-loops in standard formulation
            } else {
                let mut sq_dist = 0.0;
                for k in 0..data.ncols() {
                    sq_dist += (data[[i, k]] - data[[j, k]]).powi(2);
                }
                let a = (-gamma * sq_dist).exp();
                affinity[[i, j]] = a;
                affinity[[j, i]] = a;
            }
        }
    }

    Ok(affinity)
}

/// Build k-nearest neighbors affinity matrix.
fn build_knn_affinity(data: &ArrayView2<f64>, k: usize) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let k = k.min(n - 1);

    // Compute all pairwise distances
    let distances = try_distance_matrix(data, "spectral clustering distances")?;

    let mut affinity =
        try_zeros2(n, n, "spectral clustering affinity").map_err(|e| e.to_string())?;

    // For each point, find its k nearest neighbors
    for i in 0..n {
        let mut dist_idx: Vec<(f64, usize)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (distances[[i, j]], j))
            .collect();
        dist_idx.sort_by(|a, b| a.0.total_cmp(&b.0));

        for (dist, j) in dist_idx.into_iter().take(k) {
            // Use Gaussian kernel for edge weights
            let sigma = dist.max(1e-10);
            let w = (-dist.powi(2) / (2.0 * sigma.powi(2))).exp();
            affinity[[i, j]] = w;
            affinity[[j, i]] = w; // Make symmetric
        }
    }

    Ok(affinity)
}

/// Compute normalized Laplacian eigenvectors using power iteration.
fn compute_laplacian_eigenvectors(
    affinity: &Array2<f64>,
    k: usize,
) -> Result<(Vec<f64>, Array2<f64>), String> {
    let n = affinity.nrows();

    // Compute degree matrix D
    let degrees: Vec<f64> = (0..n).map(|i| affinity.row(i).sum()).collect();

    // Compute D^(-1/2)
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Compute normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
    // Actually, we'll compute the random walk Laplacian and use power iteration
    // to find the largest eigenvectors of D^(-1) * A

    // For simplicity, use the symmetric normalized affinity
    let mut normalized =
        try_zeros2(n, n, "spectral clustering Laplacian").map_err(|e| e.to_string())?;
    for i in 0..n {
        for j in 0..n {
            normalized[[i, j]] = d_inv_sqrt[i] * affinity[[i, j]] * d_inv_sqrt[j];
        }
    }

    // Power iteration for top k eigenvectors
    let mut eigenvectors = Array2::zeros((n, k));
    let mut eigenvalues = vec![0.0; k];
    let mut rng = StdRng::seed_from_u64(42);

    for eig_idx in 0..k {
        checkpoint().map_err(|e| e.to_string())?;
        // Initialize random vector
        let mut v: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() - 0.5).collect();

        // Orthogonalize against previous eigenvectors
        for prev in 0..eig_idx {
            let dot: f64 = (0..n).map(|i| v[i] * eigenvectors[[i, prev]]).sum();
            for i in 0..n {
                v[i] -= dot * eigenvectors[[i, prev]];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm.max(1e-10);
        }

        // Power iteration
        for _ in 0..100 {
            checkpoint().map_err(|e| e.to_string())?;
            // Matrix-vector multiplication
            let mut v_new = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += normalized[[i, j]] * v[j];
                }
            }

            // Orthogonalize against previous eigenvectors
            for prev in 0..eig_idx {
                let dot: f64 = (0..n).map(|i| v_new[i] * eigenvectors[[i, prev]]).sum();
                for i in 0..n {
                    v_new[i] -= dot * eigenvectors[[i, prev]];
                }
            }

            // Compute eigenvalue and normalize
            let lambda: f64 = v_new.iter().zip(&v).map(|(a, b)| a * b).sum();
            let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut v_new {
                *x /= norm.max(1e-10);
            }

            v = v_new;
            eigenvalues[eig_idx] = lambda;
        }

        // Store eigenvector
        for i in 0..n {
            eigenvectors[[i, eig_idx]] = v[i];
        }
    }

    // Sort by eigenvalue (we want largest for normalized affinity)
    // Actually the k largest eigenvectors of D^(-1/2) A D^(-1/2) correspond
    // to the k smallest eigenvectors of the normalized Laplacian

    Ok((eigenvalues, eigenvectors))
}

/// Cluster eigenvector embeddings using k-means.
fn cluster_eigenvectors(
    embeddings: &Array2<f64>,
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<usize>, String> {
    let n = embeddings.nrows();

    // Normalize rows
    let mut normalized = embeddings.clone();
    for i in 0..n {
        let norm: f64 = normalized.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for j in 0..embeddings.ncols() {
                normalized[[i, j]] /= norm;
            }
        }
    }

    // Run k-means
    let result = crate::ml::kmeans(normalized.view(), k, Some(100), Some(1e-4), Some(10), seed)?;

    Ok(result.labels)
}

/// Convenience wrapper for spectral_clustering.
pub fn run_spectral_clustering(
    data: ArrayView2<f64>,
    n_clusters: usize,
    gamma: Option<f64>,
    seed: Option<u64>,
) -> Result<SpectralClusteringResult, String> {
    let affinity = gamma.map(|g| AffinityType::Rbf { gamma: g });
    spectral_clustering(data, n_clusters, affinity, seed)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute overall data variance.
fn compute_data_variance(data: &ArrayView2<f64>) -> f64 {
    let mean = match data.mean_axis(Axis(0)) {
        Some(m) => m,
        None => return 0.0,
    };
    let n = data.nrows();

    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..data.ncols() {
            variance += (data[[i, j]] - mean[j]).powi(2);
        }
    }
    variance / (n * data.ncols()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_spectral_rejects_bad_parameters_and_default_seed_is_fixed() {
        let x = Array2::from_shape_fn((12, 2), |(i, j)| {
            (i % 2) as f64 * 5.0 + 0.1 * ((i * 3 + j) as f64).sin()
        });
        let e = |a| spectral_clustering(x.view(), 2, Some(a), None).unwrap_err();
        assert!(e(AffinityType::Rbf { gamma: 0.0 }).contains("gamma"));
        assert!(e(AffinityType::Knn { k: 0 }).contains("k must"));
        let mut nan = x.clone();
        nan[[1, 1]] = f64::NAN;
        assert!(
            spectral_clustering(nan.view(), 2, None, None)
                .unwrap_err()
                .contains("finite")
        );
        let a = spectral_clustering(x.view(), 2, None, None).unwrap();
        let b = spectral_clustering(x.view(), 2, None, Some(SPECTRAL_DEFAULT_SEED)).unwrap();
        assert_eq!(a.labels, b.labels);
        assert!("RBF".parse::<AffinityType>().is_ok());
        assert!(
            "cosine"
                .parse::<AffinityType>()
                .unwrap_err()
                .to_string()
                .contains("expected one of: rbf, knn, precomputed")
        );
        AFFINITY_CHOICES.validate().unwrap();
    }

    #[test]
    fn test_spectral_clustering_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        let result = spectral_clustering(data.view(), 2, None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.n_clusters, 2);
    }
}
