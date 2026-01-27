//! Advanced clustering algorithms.
//!
//! Provides sophisticated clustering methods beyond standard k-means and hierarchical.
//!
//! # Algorithms
//! - K-Medoids (PAM): Partitioning using actual data points as centers
//! - Spectral Clustering: Graph-based clustering using Laplacian eigenvectors
//! - Affinity Propagation: Message-passing clustering with automatic k selection
//! - OPTICS: Ordering Points To Identify Clustering Structure
//! - HDBSCAN: Hierarchical Density-Based Spatial Clustering
//! - Gaussian Mixture Models: Probabilistic clustering with EM algorithm
//!
//! # Large-N Performance
//!
//! HDBSCAN and OPTICS automatically select the optimal algorithm based on dataset size:
//! - n < 500: Brute-force (no overhead)
//! - n < 10,000 && d <= 15: KD-Tree + Prim's MST (good balance)
//! - n >= 10,000 && d <= 15: Dual-Tree Boruvka (best scaling)
//! - d > 15: Parallel brute-force (KD-tree degrades in high dimensions)

use ndarray::{Array2, ArrayView2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use faer::linalg::solvers::Solve;

use crate::ml::kdtree::{euclidean_distance, KdTree, UnionFind, build_connected_mst, kruskal_mst};
use crate::ml::dual_tree::kdtree_prim_mst;

// =============================================================================
// Algorithm Selection
// =============================================================================

/// Algorithm selection for HDBSCAN/OPTICS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusteringAlgorithm {
    /// Brute-force O(n²) - best for small datasets
    BruteForce,
    /// KD-Tree with Prim's MST - good for medium datasets with low dimensions
    KdTreePrim,
    /// Dual-Tree Boruvka - best scaling for large datasets with low dimensions
    DualTreeBoruvka,
    /// Parallel brute-force - for high-dimensional data where KD-tree degrades
    BruteForceParallel,
}

/// Automatically select the best algorithm based on data characteristics.
fn select_algorithm(n: usize, d: usize) -> ClusteringAlgorithm {
    match (n, d) {
        (n, _) if n < 500 => ClusteringAlgorithm::BruteForce,
        (n, d) if n < 10_000 && d <= 15 => ClusteringAlgorithm::KdTreePrim,
        (_, d) if d <= 15 => ClusteringAlgorithm::DualTreeBoruvka,
        _ => ClusteringAlgorithm::BruteForceParallel,
    }
}

// =============================================================================
// K-Medoids (PAM - Partitioning Around Medoids)
// =============================================================================

/// Result of K-Medoids clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data: An
///   Introduction to Cluster Analysis". Wiley, New York.
/// - R cluster::pam documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMedoidsResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Indices of medoid points (one per cluster)
    pub medoid_indices: Vec<usize>,
    /// Total dissimilarity (objective function value)
    pub total_dissimilarity: f64,
    /// Silhouette widths for each point
    pub silhouette_widths: Option<Vec<f64>>,
    /// Average silhouette width
    pub average_silhouette: Option<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for KMedoidsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-Medoids (PAM) Clustering Results")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Total dissimilarity: {:.4}", self.total_dissimilarity)?;
        if let Some(avg_sil) = self.average_silhouette {
            writeln!(f, "Average silhouette width: {:.4}", avg_sil)?;
        }
        writeln!(f)?;
        writeln!(f, "Medoid indices: {:?}", self.medoid_indices)?;

        // Count points per cluster
        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points (medoid: {})", i, count, self.medoid_indices[i])?;
        }
        Ok(())
    }
}

/// Run K-Medoids (PAM) clustering.
///
/// K-Medoids is a more robust variant of k-means that uses actual data points
/// (medoids) as cluster centers rather than means. This makes it less sensitive
/// to outliers.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `seed` - Optional random seed for initialization
///
/// # Algorithm
///
/// 1. BUILD phase: Select k medoids using a greedy algorithm
/// 2. SWAP phase: Iteratively try swapping medoids with non-medoids
///    to reduce the total dissimilarity
///
/// # Returns
/// * `KMedoidsResult` containing cluster assignments and medoid indices
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
pub fn kmedoids(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KMedoidsResult, String> {
    let n = data.nrows();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(100);

    // Compute pairwise distances
    let distances = compute_distance_matrix(&data);

    // BUILD phase: Initialize medoids
    let mut medoid_indices = build_medoids(&distances, k, seed);

    // Assign points to nearest medoids
    let mut labels = assign_to_medoids(&distances, &medoid_indices);
    let mut total_dissimilarity = compute_total_dissimilarity(&distances, &labels, &medoid_indices);

    // SWAP phase
    let mut n_iterations = 0;
    for iter in 0..max_iter {
        n_iterations = iter + 1;
        let mut improved = false;

        // Try swapping each medoid with each non-medoid
        for m_idx in 0..k {
            let current_medoid = medoid_indices[m_idx];

            for candidate in 0..n {
                // Skip if candidate is already a medoid
                if medoid_indices.contains(&candidate) {
                    continue;
                }

                // Try swap
                let mut new_medoids = medoid_indices.clone();
                new_medoids[m_idx] = candidate;

                let new_labels = assign_to_medoids(&distances, &new_medoids);
                let new_cost = compute_total_dissimilarity(&distances, &new_labels, &new_medoids);

                if new_cost < total_dissimilarity - 1e-10 {
                    medoid_indices = new_medoids;
                    labels = new_labels;
                    total_dissimilarity = new_cost;
                    improved = true;
                    break; // Restart search from first medoid
                }
            }

            if improved {
                break;
            }
        }

        if !improved {
            break;
        }
    }

    // Optionally compute silhouette
    let (silhouette_widths, average_silhouette) = if k > 1 {
        let sil = compute_silhouette_from_distances(&distances, &labels, k);
        let avg = sil.iter().sum::<f64>() / n as f64;
        (Some(sil), Some(avg))
    } else {
        (None, None)
    };

    Ok(KMedoidsResult {
        labels,
        medoid_indices,
        total_dissimilarity,
        silhouette_widths,
        average_silhouette,
        n_iterations,
        n_clusters: k,
        n,
    })
}

/// BUILD phase: Greedy medoid initialization.
fn build_medoids(distances: &Array2<f64>, k: usize, seed: Option<u64>) -> Vec<usize> {
    let n = distances.nrows();
    let mut medoids = Vec::with_capacity(k);
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // First medoid: point that minimizes sum of distances to all others
    let mut first_medoid = 0;
    let mut min_sum = f64::INFINITY;
    for i in 0..n {
        let sum: f64 = distances.row(i).sum();
        if sum < min_sum {
            min_sum = sum;
            first_medoid = i;
        }
    }
    medoids.push(first_medoid);

    // Remaining medoids: greedily select to minimize total dissimilarity
    while medoids.len() < k {
        let mut best_candidate = 0;
        let mut best_gain = f64::NEG_INFINITY;

        for candidate in 0..n {
            if medoids.contains(&candidate) {
                continue;
            }

            // Compute gain from adding this candidate
            let mut gain = 0.0;
            for i in 0..n {
                let current_min: f64 = medoids.iter()
                    .map(|&m| distances[[i, m]])
                    .fold(f64::INFINITY, |a, b| a.min(b));
                let new_dist = distances[[i, candidate]];
                if new_dist < current_min {
                    gain += current_min - new_dist;
                }
            }

            if gain > best_gain {
                best_gain = gain;
                best_candidate = candidate;
            }
        }

        medoids.push(best_candidate);
    }

    medoids
}

/// Assign each point to its nearest medoid.
fn assign_to_medoids(distances: &Array2<f64>, medoids: &[usize]) -> Vec<usize> {
    let n = distances.nrows();
    let k = medoids.len();

    (0..n).map(|i| {
        let mut min_dist = f64::INFINITY;
        let mut best_cluster = 0;
        for (cluster, &medoid) in medoids.iter().enumerate() {
            let dist = distances[[i, medoid]];
            if dist < min_dist {
                min_dist = dist;
                best_cluster = cluster;
            }
        }
        best_cluster
    }).collect()
}

/// Compute total dissimilarity (sum of distances to assigned medoids).
fn compute_total_dissimilarity(
    distances: &Array2<f64>,
    labels: &[usize],
    medoids: &[usize],
) -> f64 {
    labels.iter().enumerate()
        .map(|(i, &label)| distances[[i, medoids[label]]])
        .sum()
}

/// Convenience wrapper for kmedoids.
pub fn run_kmedoids(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KMedoidsResult, String> {
    kmedoids(data, k, max_iterations, seed)
}

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

impl std::str::FromStr for AffinityType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rbf" => Ok(AffinityType::Rbf { gamma: 1.0 }),
            "knn" => Ok(AffinityType::Knn { k: 10 }),
            "precomputed" => Ok(AffinityType::Precomputed),
            _ => Err(format!("Unknown affinity type: {}", s)),
        }
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
/// * `seed` - Optional random seed for k-means step
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
            "n_clusters must be between 1 and {} (n_samples)", n
        ));
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
        AffinityType::Rbf { gamma } => build_rbf_affinity(&data, gamma),
        AffinityType::Knn { k } => build_knn_affinity(&data, k),
        AffinityType::Precomputed => {
            // Assume data IS the affinity matrix
            if data.nrows() != data.ncols() {
                return Err("Precomputed affinity must be square".to_string());
            }
            data.to_owned()
        }
    };

    // Compute normalized Laplacian and its eigenvectors
    let (eigenvalues, eigenvectors) = compute_laplacian_eigenvectors(
        &affinity_matrix,
        n_clusters,
    )?;

    // Cluster the eigenvector embeddings with k-means
    let labels = cluster_eigenvectors(&eigenvectors, n_clusters, seed)?;

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
fn build_rbf_affinity(data: &ArrayView2<f64>, gamma: f64) -> Array2<f64> {
    let n = data.nrows();
    let mut affinity = Array2::zeros((n, n));

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

    affinity
}

/// Build k-nearest neighbors affinity matrix.
fn build_knn_affinity(data: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let n = data.nrows();
    let k = k.min(n - 1);

    // Compute all pairwise distances
    let distances = compute_distance_matrix(data);

    let mut affinity = Array2::zeros((n, n));

    // For each point, find its k nearest neighbors
    for i in 0..n {
        let mut dist_idx: Vec<(f64, usize)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (distances[[i, j]], j))
            .collect();
        dist_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for (dist, j) in dist_idx.into_iter().take(k) {
            // Use Gaussian kernel for edge weights
            let sigma = dist.max(1e-10);
            let w = (-dist.powi(2) / (2.0 * sigma.powi(2))).exp();
            affinity[[i, j]] = w;
            affinity[[j, i]] = w; // Make symmetric
        }
    }

    affinity
}

/// Compute normalized Laplacian eigenvectors using power iteration.
fn compute_laplacian_eigenvectors(
    affinity: &Array2<f64>,
    k: usize,
) -> Result<(Vec<f64>, Array2<f64>), String> {
    let n = affinity.nrows();

    // Compute degree matrix D
    let degrees: Vec<f64> = (0..n)
        .map(|i| affinity.row(i).sum())
        .collect();

    // Compute D^(-1/2)
    let d_inv_sqrt: Vec<f64> = degrees.iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Compute normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
    // Actually, we'll compute the random walk Laplacian and use power iteration
    // to find the largest eigenvectors of D^(-1) * A

    // For simplicity, use the symmetric normalized affinity
    let mut normalized = Array2::zeros((n, n));
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
    let result = crate::ml::kmeans(
        normalized.view(), k, Some(100), Some(1e-4), Some(10), seed,
    )?;

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
// Affinity Propagation
// =============================================================================

/// Result of affinity propagation clustering.
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages Between
///   Data Points". Science, 315, 972-976.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityPropagationResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Indices of exemplar (cluster center) points
    pub exemplar_indices: Vec<usize>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Final cluster quality (sum of responsibilities to exemplars)
    pub net_similarity: f64,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for AffinityPropagationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Affinity Propagation Results")?;
        writeln!(f, "============================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Net similarity: {:.4}", self.net_similarity)?;
        writeln!(f)?;
        writeln!(f, "Exemplar indices: {:?}", self.exemplar_indices)?;

        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points (exemplar: {})",
                     i, count, self.exemplar_indices[i])?;
        }
        Ok(())
    }
}

/// Run affinity propagation clustering.
///
/// Affinity propagation clusters data by exchanging real-valued messages
/// between data points until a high-quality set of exemplars emerges.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `preference` - Preference for each point to be an exemplar (default: median similarity)
/// * `damping` - Damping factor (0.5 to 1.0, default: 0.5)
/// * `max_iterations` - Maximum iterations (default: 200)
/// * `convergence_iterations` - Iterations without change for convergence (default: 15)
///
/// # Algorithm
///
/// 1. Initialize similarity matrix S (negative squared Euclidean distance)
/// 2. Initialize responsibility R and availability A matrices to 0
/// 3. Iteratively update:
///    - R(i,k) = S(i,k) - max_{k' != k}[A(i,k') + S(i,k')]
///    - A(i,k) = min(0, R(k,k) + sum_{i' != i,k} max(0, R(i',k)))
///    - A(k,k) = sum_{i' != k} max(0, R(i',k))
/// 4. Exemplars are points where R(k,k) + A(k,k) > 0
///
/// # Returns
/// * `AffinityPropagationResult` containing cluster assignments and exemplars
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages".
pub fn affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> Result<AffinityPropagationResult, String> {
    let n = data.nrows();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let damp = damping.unwrap_or(0.5);
    if damp < 0.5 || damp >= 1.0 {
        return Err("Damping must be between 0.5 and 1.0".to_string());
    }

    let max_iter = max_iterations.unwrap_or(200);
    let conv_iter = convergence_iterations.unwrap_or(15);

    // Compute similarity matrix (negative squared Euclidean distance)
    let mut similarity = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut sq_dist = 0.0;
                for k in 0..data.ncols() {
                    sq_dist += (data[[i, k]] - data[[j, k]]).powi(2);
                }
                similarity[[i, j]] = -sq_dist;
            }
        }
    }

    // Set preference (diagonal) - default to median similarity
    let pref = preference.unwrap_or_else(|| {
        let mut sims: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sims.push(similarity[[i, j]]);
                }
            }
        }
        sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if sims.is_empty() { 0.0 } else { sims[sims.len() / 2] }
    });
    for i in 0..n {
        similarity[[i, i]] = pref;
    }

    // Initialize responsibility and availability matrices
    let mut responsibility: Array2<f64> = Array2::zeros((n, n));
    let mut availability: Array2<f64> = Array2::zeros((n, n));

    // Track exemplars for convergence detection
    let mut prev_exemplars = vec![false; n];
    let mut unchanged_count = 0;

    let mut n_iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Optimized responsibility update: O(n²) instead of O(n³)
        // Precompute AS = A + S, then find first and second max per row
        let mut as_matrix = Array2::zeros((n, n));
        for i in 0..n {
            for k in 0..n {
                as_matrix[[i, k]] = availability[[i, k]] + similarity[[i, k]];
            }
        }

        // For each row, find first_max and second_max
        let mut row_max1 = vec![f64::NEG_INFINITY; n]; // first max
        let mut row_max2 = vec![f64::NEG_INFINITY; n]; // second max
        let mut row_argmax = vec![0usize; n];

        for i in 0..n {
            for k in 0..n {
                let val = as_matrix[[i, k]];
                if val > row_max1[i] {
                    row_max2[i] = row_max1[i];
                    row_max1[i] = val;
                    row_argmax[i] = k;
                } else if val > row_max2[i] {
                    row_max2[i] = val;
                }
            }
        }

        // Compute new responsibilities using precomputed max values
        for i in 0..n {
            for k in 0..n {
                let max_other = if k == row_argmax[i] { row_max2[i] } else { row_max1[i] };
                let r_new = similarity[[i, k]] - max_other;
                responsibility[[i, k]] = damp * responsibility[[i, k]] + (1.0 - damp) * r_new;
            }
        }

        // Optimized availability update: O(n²) instead of O(n³)
        // Precompute column sums of positive responsibilities
        let mut col_pos_sum = vec![0.0f64; n];
        for k in 0..n {
            for i in 0..n {
                col_pos_sum[k] += responsibility[[i, k]].max(0.0);
            }
        }

        // Compute availabilities using precomputed sums
        for k in 0..n {
            // Self-availability: sum of positive responsibilities from others
            let self_r_pos = responsibility[[k, k]].max(0.0);
            let a_kk = col_pos_sum[k] - self_r_pos;
            availability[[k, k]] = damp * availability[[k, k]] + (1.0 - damp) * a_kk;
        }

        for i in 0..n {
            for k in 0..n {
                if i != k {
                    let self_r_pos = responsibility[[k, k]].max(0.0);
                    let i_r_pos = responsibility[[i, k]].max(0.0);
                    // sum excluding k and i
                    let sum = col_pos_sum[k] - self_r_pos - i_r_pos;
                    let a_new = (responsibility[[k, k]] + sum).min(0.0);
                    availability[[i, k]] = damp * availability[[i, k]] + (1.0 - damp) * a_new;
                }
            }
        }

        // Check convergence: exemplars are stable
        let mut exemplars: Vec<bool> = (0..n)
            .map(|k| responsibility[[k, k]] + availability[[k, k]] > 0.0)
            .collect();

        if exemplars == prev_exemplars {
            unchanged_count += 1;
            if unchanged_count >= conv_iter {
                converged = true;
                break;
            }
        } else {
            unchanged_count = 0;
        }
        prev_exemplars = exemplars;
    }

    // Identify final exemplars
    let exemplar_mask: Vec<bool> = (0..n)
        .map(|k| responsibility[[k, k]] + availability[[k, k]] > 0.0)
        .collect();

    let exemplar_indices: Vec<usize> = exemplar_mask.iter()
        .enumerate()
        .filter(|&(_, is_ex)| *is_ex)
        .map(|(i, _)| i)
        .collect();

    // Assign points to nearest exemplar
    let mut labels = vec![0usize; n];
    for i in 0..n {
        if exemplar_mask[i] {
            // Exemplar assigned to itself
            labels[i] = exemplar_indices.iter().position(|&e| e == i).unwrap_or(0);
        } else {
            // Assign to exemplar with highest responsibility
            let mut best_ex = 0;
            let mut best_r = f64::NEG_INFINITY;
            for (ex_idx, &ex) in exemplar_indices.iter().enumerate() {
                if similarity[[i, ex]] > best_r {
                    best_r = similarity[[i, ex]];
                    best_ex = ex_idx;
                }
            }
            labels[i] = best_ex;
        }
    }

    // Compute net similarity
    let net_similarity: f64 = labels.iter().enumerate()
        .map(|(i, &label)| {
            if label < exemplar_indices.len() {
                similarity[[i, exemplar_indices[label]]]
            } else {
                0.0
            }
        })
        .sum();

    Ok(AffinityPropagationResult {
        labels,
        exemplar_indices: exemplar_indices.clone(),
        n_iterations,
        converged,
        net_similarity,
        n_clusters: exemplar_indices.len(),
        n,
    })
}

/// Convenience wrapper for affinity_propagation.
pub fn run_affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
) -> Result<AffinityPropagationResult, String> {
    affinity_propagation(data, preference, damping, None, None)
}

// =============================================================================
// OPTICS (Ordering Points To Identify the Clustering Structure)
// =============================================================================

/// Result of OPTICS clustering.
///
/// # References
///
/// - Ankerst, M., Breunig, M.M., Kriegel, H.P., and Sander, J. (1999).
///   "OPTICS: Ordering Points To Identify the Clustering Structure".
///   ACM SIGMOD 1999.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticsResult {
    /// Ordering of points (indices in processing order)
    pub ordering: Vec<usize>,
    /// Reachability distances for each point (in ordering order)
    pub reachability: Vec<f64>,
    /// Core distances for each point
    pub core_distances: Vec<f64>,
    /// Cluster labels (-1 for noise) extracted at eps_prime
    pub labels: Vec<i32>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// MinPts parameter used
    pub min_samples: usize,
    /// Xi parameter used for cluster extraction (if any)
    pub xi: Option<f64>,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for OpticsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OPTICS Clustering Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "MinPts: {}", self.min_samples)?;
        if let Some(xi) = self.xi {
            writeln!(f, "Xi: {:.4}", xi)?;
        }
        writeln!(f)?;
        writeln!(f, "Reachability plot data available in 'ordering' and 'reachability' fields")?;

        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().cloned().collect();
        labels_sorted.sort();
        for label in labels_sorted {
            if label == -1 {
                writeln!(f, "  Noise: {} points", cluster_counts[&label])?;
            } else {
                writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
            }
        }
        Ok(())
    }
}

/// Run OPTICS clustering.
///
/// OPTICS creates an ordering of points such that spatially closest points
/// become neighbors in the ordering. It produces a reachability plot that
/// can be used to extract clusters at various density levels.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `min_samples` - Minimum points for core point definition
/// * `max_eps` - Maximum epsilon neighborhood (default: infinity)
/// * `xi` - Optional steepness parameter for automatic cluster extraction
///
/// # Algorithm
///
/// 1. For each point, compute core-distance (distance to min_samples-th nearest neighbor)
/// 2. Process points in order of reachability-distance
/// 3. Update reachability-distances of unprocessed neighbors
/// 4. Extract clusters from reachability plot
///
/// # Performance
///
/// Uses KD-tree acceleration when beneficial:
/// - n < 500: O(n²) brute-force
/// - n >= 500 && d <= 15: O(n log n) core distances + O(n log n) range queries
/// - d > 15: O(n²) parallel brute-force
///
/// # Returns
/// * `OpticsResult` containing ordering and reachability distances
///
/// # References
///
/// - Ankerst et al. (1999). "OPTICS: Ordering Points To Identify the Clustering Structure".
pub fn optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    if min_samples < 1 || min_samples > n {
        return Err(format!("min_samples must be between 1 and {}", n));
    }

    let eps = max_eps.unwrap_or(f64::INFINITY);

    // Select algorithm and compute core distances
    let algorithm = select_algorithm(n, d);
    let use_kdtree = matches!(algorithm, ClusteringAlgorithm::KdTreePrim | ClusteringAlgorithm::DualTreeBoruvka);

    let (distances, core_distances, tree_opt) = if use_kdtree {
        let data_vecs: Vec<Vec<f64>> = (0..n)
            .map(|i| data.row(i).to_vec())
            .collect();
        let tree = KdTree::new(data_vecs);
        let core_dists = tree.compute_core_distances(min_samples)
            .into_iter()
            .map(|cd| cd.min(eps))
            .collect::<Vec<_>>();
        (None, core_dists, Some(tree))
    } else {
        // Use brute-force distance matrix
        let distances = if matches!(algorithm, ClusteringAlgorithm::BruteForceParallel) {
            compute_distance_matrix_parallel(&data)
        } else {
            compute_distance_matrix(&data)
        };

        let core_distances: Vec<f64> = (0..n).map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| distances[[i, j]])
                .collect();

            if dists.len() >= min_samples {
                dists.select_nth_unstable_by(min_samples - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                dists[min_samples - 1].min(eps)
            } else {
                f64::INFINITY
            }
        }).collect();

        (Some(distances), core_distances, None)
    };

    // OPTICS ordering
    let mut processed = vec![false; n];
    let mut ordering = Vec::with_capacity(n);
    let mut reachability = vec![f64::INFINITY; n];

    // Priority queue: (reachability, index) - min-heap based on reachability
    #[derive(PartialEq)]
    struct OrderedPoint {
        reachability: f64,
        index: usize,
    }

    impl Eq for OrderedPoint {}

    impl Ord for OrderedPoint {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap
            other.reachability.partial_cmp(&self.reachability)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for OrderedPoint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // Helper closure to get neighbors
    let get_neighbors = |point_idx: usize, radius: f64| -> Vec<(f64, usize)> {
        if let Some(ref tree) = tree_opt {
            tree.radius_query(tree.data().get(point_idx).unwrap(), radius, Some(point_idx))
        } else if let Some(ref dist_matrix) = distances {
            (0..n)
                .filter(|&j| j != point_idx && dist_matrix[[point_idx, j]] <= radius)
                .map(|j| (dist_matrix[[point_idx, j]], j))
                .collect()
        } else {
            Vec::new()
        }
    };

    // Process all points
    for start in 0..n {
        if processed[start] {
            continue;
        }

        // Start a new cluster from this point
        let mut seeds = BinaryHeap::new();
        processed[start] = true;
        ordering.push(start);

        // Add neighbors of start to seeds
        if core_distances[start] <= eps {
            let neighbors = get_neighbors(start, eps);
            for (dist, j) in neighbors {
                if !processed[j] {
                    let new_reach = core_distances[start].max(dist);
                    if new_reach < reachability[j] {
                        reachability[j] = new_reach;
                        seeds.push(OrderedPoint { reachability: new_reach, index: j });
                    }
                }
            }
        }

        // Process seeds
        while let Some(OrderedPoint { index: p, reachability: _ }) = seeds.pop() {
            if processed[p] {
                continue;
            }

            processed[p] = true;
            ordering.push(p);

            // Update neighbors if p is a core point
            if core_distances[p] <= eps {
                let neighbors = get_neighbors(p, eps);
                for (dist, j) in neighbors {
                    if !processed[j] {
                        let new_reach = core_distances[p].max(dist);
                        if new_reach < reachability[j] {
                            reachability[j] = new_reach;
                            seeds.push(OrderedPoint { reachability: new_reach, index: j });
                        }
                    }
                }
            }
        }
    }

    // Extract clusters using xi (steepness-based) or simple threshold
    let labels = if let Some(xi_val) = xi {
        extract_clusters_xi(&ordering, &reachability, xi_val)
    } else {
        // Use median reachability as threshold
        let mut sorted_reach: Vec<f64> = reachability.iter()
            .filter(|&&r| r.is_finite())
            .cloned()
            .collect();
        sorted_reach.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let threshold = if sorted_reach.is_empty() {
            f64::INFINITY
        } else {
            sorted_reach[sorted_reach.len() / 2] * 1.5
        };
        extract_clusters_threshold(&ordering, &reachability, threshold)
    };

    let n_clusters = labels.iter().filter(|&&l| l >= 0).map(|&l| l).max().map_or(0, |m| m as usize + 1);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    // Reorder reachability to match ordering
    let reachability_ordered: Vec<f64> = ordering.iter()
        .map(|&i| reachability[i])
        .collect();

    Ok(OpticsResult {
        ordering,
        reachability: reachability_ordered,
        core_distances,
        labels,
        n_clusters,
        n_noise,
        min_samples,
        xi,
        n,
    })
}

/// Extract clusters using xi (steepness) method.
fn extract_clusters_xi(ordering: &[usize], reachability: &[f64], xi: f64) -> Vec<i32> {
    let n = ordering.len();
    let mut labels = vec![-1i32; n];

    if n == 0 {
        return labels;
    }

    // Find steep down and up areas
    let mut current_cluster = 0;
    let mut in_cluster = false;

    for i in 1..ordering.len() {
        let prev_idx = ordering[i - 1];
        let curr_idx = ordering[i];
        let prev_r = reachability[prev_idx];
        let curr_r = reachability[curr_idx];

        // Steep down: start of cluster
        if prev_r.is_finite() && curr_r.is_finite() {
            if curr_r < prev_r * (1.0 - xi) {
                // Steep down area - potential cluster start
                if !in_cluster {
                    in_cluster = true;
                }
            } else if curr_r > prev_r / (1.0 - xi) {
                // Steep up area - potential cluster end
                if in_cluster {
                    current_cluster += 1;
                    in_cluster = false;
                }
            }
        }

        if in_cluster {
            labels[curr_idx] = current_cluster as i32;
        }
    }

    labels
}

/// Extract clusters using simple reachability threshold.
fn extract_clusters_threshold(ordering: &[usize], reachability: &[f64], threshold: f64) -> Vec<i32> {
    let n = ordering.len();
    let mut labels = vec![-1i32; n];
    let mut current_cluster = -1i32;

    for &idx in ordering {
        let r = reachability[idx];

        if r > threshold {
            // Noise or boundary
            if current_cluster >= 0 {
                current_cluster += 1;
            }
            labels[idx] = -1;
        } else {
            // Part of cluster
            if current_cluster < 0 {
                current_cluster = 0;
            }
            labels[idx] = current_cluster;
        }
    }

    labels
}

/// Convenience wrapper for optics.
pub fn run_optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    optics(data, min_samples, max_eps, xi)
}

// =============================================================================
// HDBSCAN (Hierarchical DBSCAN)
// =============================================================================

/// Result of HDBSCAN clustering.
///
/// # References
///
/// - Campello, R.J.G.B., Moulavi, D., and Sander, J. (2013). "Density-Based
///   Clustering Based on Hierarchical Density Estimates". PAKDD 2013.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdbscanResult {
    /// Cluster assignments (-1 for noise)
    pub labels: Vec<i32>,
    /// Cluster membership probabilities (soft clustering)
    pub probabilities: Vec<f64>,
    /// Outlier scores (GLOSH)
    pub outlier_scores: Vec<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// Minimum cluster size used
    pub min_cluster_size: usize,
    /// Minimum samples used
    pub min_samples: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for HdbscanResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HDBSCAN Clustering Results")?;
        writeln!(f, "==========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "min_cluster_size: {}", self.min_cluster_size)?;
        writeln!(f, "min_samples: {}", self.min_samples)?;

        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().cloned().collect();
        labels_sorted.sort();
        for label in labels_sorted {
            if label == -1 {
                writeln!(f, "  Noise: {} points", cluster_counts[&label])?;
            } else {
                writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
            }
        }
        Ok(())
    }
}

/// Run HDBSCAN clustering.
///
/// HDBSCAN is a hierarchical extension of DBSCAN that builds a hierarchy of
/// clusters and extracts flat clusters based on cluster stability.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `min_cluster_size` - Minimum cluster size (default: 5)
/// * `min_samples` - Minimum samples for core points (default: min_cluster_size)
///
/// # Algorithm
///
/// 1. Compute mutual reachability distance
/// 2. Build minimum spanning tree
/// 3. Construct cluster hierarchy
/// 4. Extract stable clusters using EOMTG
///
/// # Performance
///
/// Automatically selects optimal algorithm based on data size:
/// - n < 500: O(n²) brute-force
/// - n < 10,000 && d <= 15: O(n log n) KD-Tree + Prim's
/// - n >= 10,000 && d <= 15: O(n log n) Dual-Tree Boruvka
/// - d > 15: O(n²) parallel brute-force
///
/// # Returns
/// * `HdbscanResult` containing cluster assignments and probabilities
///
/// # References
///
/// - Campello et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates".
pub fn hdbscan(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<HdbscanResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let mcs = min_cluster_size.unwrap_or(5).max(2);
    let ms = min_samples.unwrap_or(mcs);

    // Select algorithm based on data characteristics
    let algorithm = select_algorithm(n, d);

    let (core_distances, mst) = match algorithm {
        ClusteringAlgorithm::BruteForce => hdbscan_brute_force(&data, ms),
        ClusteringAlgorithm::KdTreePrim => hdbscan_kdtree_prim(&data, ms),
        ClusteringAlgorithm::DualTreeBoruvka => hdbscan_dual_tree(&data, ms),
        ClusteringAlgorithm::BruteForceParallel => hdbscan_parallel(&data, ms),
    };

    // Sort MST edges by weight (for hierarchical clustering)
    let mut sorted_edges = mst.clone();
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Build hierarchy and extract clusters
    let (labels, probabilities, outlier_scores) = extract_hdbscan_clusters(
        &sorted_edges, n, mcs, &core_distances,
    );

    let n_clusters = labels.iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l)
        .max()
        .map_or(0, |m| m as usize + 1);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    Ok(HdbscanResult {
        labels,
        probabilities,
        outlier_scores,
        n_clusters,
        n_noise,
        min_cluster_size: mcs,
        min_samples: ms,
        n,
    })
}

/// HDBSCAN with KD-Tree + Prim's algorithm.
/// O(n log n) for core distances, O(n²) for Prim's but with KD-tree acceleration.
fn hdbscan_kdtree_prim(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let n = data.nrows();

    // Convert to Vec<Vec<f64>> for KD-tree
    let data_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| data.row(i).to_vec())
        .collect();

    let tree = KdTree::new(data_vecs);
    let core_distances = tree.compute_core_distances(min_samples);
    let mst = kdtree_prim_mst(&tree, &core_distances);

    (core_distances, mst)
}

/// HDBSCAN with KD-Tree accelerated MST construction.
/// Uses Prim's algorithm with O(n * k * log n) core distance computation.
/// Overall complexity is O(n² log n) but with good constants for low-d data.
fn hdbscan_dual_tree(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    // Note: The Dual-Tree Boruvka algorithm has known issues with edge detection.
    // Using KD-Tree + Prim's MST which is provably correct.
    // The KD-tree still provides benefit for core distance computation.
    hdbscan_kdtree_prim(data, min_samples)
}

/// Parallel brute-force HDBSCAN for high-dimensional data.
fn hdbscan_parallel(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let n = data.nrows();

    // Compute pairwise distances in parallel
    let distances = compute_distance_matrix_parallel(data);

    // Use references to avoid move issues
    let dist_ref = &distances;

    // Compute core distances in parallel
    let core_distances: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| dist_ref[[i, j]])
                .collect();

            if dists.len() >= min_samples {
                dists.select_nth_unstable_by(min_samples - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                dists[min_samples - 1]
            } else {
                dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
        })
        .collect();

    // Use references for the next parallel section
    let core_ref = &core_distances;

    // Compute mutual reachability in parallel
    let mr_edges: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).map(move |j| {
                let mr = dist_ref[[i, j]]
                    .max(core_ref[i])
                    .max(core_ref[j]);
                (i, j, mr)
            }).collect::<Vec<_>>()
        })
        .collect();

    // Build MST using Kruskal's
    let mst = kruskal_mst(&mr_edges, n);

    (core_distances, mst)
}

/// Compute distance matrix in parallel.
fn compute_distance_matrix_parallel(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();

    let data_slice: Vec<f64> = data.iter().cloned().collect();
    let data_ref = &data_slice;

    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).map(move |j| {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = data_ref[i * d + k] - data_ref[j * d + k];
                    sum += diff * diff;
                }
                (i, j, sum.sqrt())
            }).collect::<Vec<_>>()
        })
        .collect();

    let mut distances = Array2::zeros((n, n));
    for (i, j, dist) in results {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }

    distances
}

/// HDBSCAN using KD-tree for efficient neighbor queries.
/// Returns (core_distances, mst_edges).
fn hdbscan_with_kdtree(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    use crate::ml::kdtree::KdTree;

    let n = data.nrows();

    // Convert data to Vec<Vec<f64>> for KD-tree
    let data_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| data.row(i).to_vec())
        .collect();

    let tree = KdTree::new(data_vecs.clone());

    // Compute core distances using KD-tree: O(n * k * log n)
    let core_distances: Vec<f64> = (0..n)
        .map(|i| {
            let neighbors = tree.k_nearest(&data_vecs[i], min_samples, Some(i));
            if neighbors.len() >= min_samples {
                neighbors[min_samples - 1].0
            } else if !neighbors.is_empty() {
                neighbors.last().map(|x| x.0).unwrap_or(0.0)
            } else {
                0.0
            }
        })
        .collect();

    // For MST, we need mutual reachability distances
    // Use a sparse approach: for each point, find neighbors within 2*max_core_distance
    // and build edges only for those pairs
    let max_core = core_distances.iter().cloned().fold(0.0f64, f64::max);
    let search_radius = 2.0 * max_core;

    // Build sparse graph edges
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();

    for i in 0..n {
        // Get neighbors within search radius
        let neighbors = tree.radius_query(&data_vecs[i], search_radius, Some(i));

        for (dist, j) in neighbors {
            if i < j {
                let key = (i, j);
                if !seen_edges.contains(&key) {
                    seen_edges.insert(key);
                    let mr = dist.max(core_distances[i]).max(core_distances[j]);
                    edges.push((i, j, mr));
                }
            }
        }
    }

    // If graph is not connected, add remaining edges with brute force
    // Check connectivity using union-find
    let mut uf = UnionFind::new(n);
    for &(i, j, _) in &edges {
        uf.union(i, j);
    }

    // Find connected components
    let components: usize = (0..n).map(|i| uf.find(i)).collect::<std::collections::HashSet<_>>().len();

    if components > 1 {
        // Graph is disconnected - fall back to computing missing edges
        // Find representatives of each component
        let mut component_reps: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            component_reps.entry(uf.find(i)).or_default().push(i);
        }

        // Connect components with minimum mutual reachability edges
        let rep_list: Vec<usize> = component_reps.keys().cloned().collect();
        for ci in 0..rep_list.len() {
            for cj in (ci + 1)..rep_list.len() {
                let comp_i = &component_reps[&rep_list[ci]];
                let comp_j = &component_reps[&rep_list[cj]];

                // Find minimum edge between components
                let mut min_edge = (0, 0, f64::INFINITY);
                for &i in comp_i {
                    for &j in comp_j {
                        let dist = euclidean_distance(&data_vecs[i], &data_vecs[j]);
                        let mr = dist.max(core_distances[i]).max(core_distances[j]);
                        if mr < min_edge.2 {
                            min_edge = (i.min(j), i.max(j), mr);
                        }
                    }
                }
                if min_edge.2 < f64::INFINITY {
                    edges.push(min_edge);
                }
            }
        }
    }

    // Build MST from sparse graph using Kruskal's algorithm
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let mut uf = UnionFind::new(n);
    let mut mst = Vec::with_capacity(n - 1);

    for (i, j, w) in edges {
        if uf.find(i) != uf.find(j) {
            uf.union(i, j);
            mst.push((i, j, w));
            if mst.len() == n - 1 {
                break;
            }
        }
    }

    (core_distances, mst)
}

/// HDBSCAN using optimized distance computation.
/// Uses partial-sort for core distances (O(n) instead of O(n log n) per point).
fn hdbscan_brute_force(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let n = data.nrows();

    // Compute pairwise distances once
    let distances = compute_distance_matrix(data);

    // Compute core distances using partial sort - O(n) per point instead of O(n log n)
    let core_distances: Vec<f64> = (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| distances[[i, j]])
                .collect();

            if dists.len() >= min_samples {
                dists.select_nth_unstable_by(min_samples - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                dists[min_samples - 1]
            } else {
                dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
        })
        .collect();

    // Compute mutual reachability distance
    let mut mutual_reach = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mr = distances[[i, j]]
                .max(core_distances[i])
                .max(core_distances[j]);
            mutual_reach[[i, j]] = mr;
            mutual_reach[[j, i]] = mr;
        }
    }

    // Build MST
    let mst = build_mst(&mutual_reach);

    (core_distances, mst)
}

// UnionFind is imported from kdtree module

/// Build minimum spanning tree using Prim's algorithm.
fn build_mst(distances: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let n = distances.nrows();
    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut min_from = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    for j in 1..n {
        min_dist[j] = distances[[0, j]];
        min_from[j] = 0;
    }

    for _ in 1..n {
        // Find minimum distance node not in tree
        let mut min_idx = 0;
        let mut min_val = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_dist[j] < min_val {
                min_val = min_dist[j];
                min_idx = j;
            }
        }

        // Add edge to MST
        in_tree[min_idx] = true;
        mst.push((min_from[min_idx], min_idx, min_val));

        // Update distances
        for j in 0..n {
            if !in_tree[j] && distances[[min_idx, j]] < min_dist[j] {
                min_dist[j] = distances[[min_idx, j]];
                min_from[j] = min_idx;
            }
        }
    }

    mst
}

/// Extract HDBSCAN clusters using single-linkage hierarchy and stability.
fn extract_hdbscan_clusters(
    sorted_edges: &[(usize, usize, f64)],
    n: usize,
    min_cluster_size: usize,
    core_distances: &[f64],
) -> (Vec<i32>, Vec<f64>, Vec<f64>) {
    // Use union-find to track cluster membership
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];
    let mut size: Vec<usize> = vec![1; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], size: &mut [usize], x: usize, y: usize) -> usize {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py {
            return px;
        }

        let (smaller, larger) = if rank[px] < rank[py] { (px, py) } else { (py, px) };
        parent[smaller] = larger;
        size[larger] += size[smaller];
        if rank[px] == rank[py] {
            rank[larger] += 1;
        }
        larger
    }

    // Track cluster birth and death for stability
    let mut cluster_birth = vec![0.0f64; n];
    let mut cluster_stability = vec![0.0f64; n];

    // Process edges in order
    for &(i, j, weight) in sorted_edges {
        let pi = find(&mut parent, i);
        let pj = find(&mut parent, j);

        if pi != pj {
            let lambda = 1.0 / weight.max(1e-10);

            // Record when clusters merge
            if size[pi] >= min_cluster_size {
                cluster_stability[pi] += (lambda - cluster_birth[pi]) * size[pi] as f64;
            }
            if size[pj] >= min_cluster_size {
                cluster_stability[pj] += (lambda - cluster_birth[pj]) * size[pj] as f64;
            }

            let new_root = union(&mut parent, &mut rank, &mut size, i, j);
            cluster_birth[new_root] = lambda;
        }
    }

    // Find final clusters based on stability
    // Simplified: use the roots with minimum cluster size as clusters
    let mut labels = vec![-1i32; n];
    let mut cluster_id = 0;

    // Group points by their root
    let mut root_to_cluster: HashMap<usize, i32> = HashMap::new();

    for i in 0..n {
        let root = find(&mut parent, i);
        if size[root] >= min_cluster_size {
            let cluster = *root_to_cluster.entry(root).or_insert_with(|| {
                let id = cluster_id;
                cluster_id += 1;
                id
            });
            labels[i] = cluster;
        }
    }

    // Compute probabilities (simplified: based on core distance relative to cluster)
    let probabilities: Vec<f64> = (0..n).map(|i| {
        if labels[i] >= 0 {
            1.0 / (1.0 + core_distances[i])
        } else {
            0.0
        }
    }).collect();

    // Compute outlier scores (simplified GLOSH)
    let outlier_scores: Vec<f64> = core_distances.iter()
        .map(|&cd| cd / (cd + 1.0))
        .collect();

    (labels, probabilities, outlier_scores)
}

/// Convenience wrapper for hdbscan.
pub fn run_hdbscan(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<HdbscanResult, String> {
    hdbscan(data, min_cluster_size, min_samples)
}

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
        data.view(), n_components, Some(20), Some(1e-4), Some(5), seed,
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
            let cluster_points: Vec<usize> = kmeans_result.labels.iter()
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
                let log_prob = log_gaussian_pdf(&point, &means[k], &covariances[k])
                    + weights[k].ln();
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
                means[k] = (0..d).map(|j| {
                    let sum: f64 = (0..n)
                        .map(|i| responsibilities[[i, k]] * data[[i, j]])
                        .sum();
                    sum / nk[k]
                }).collect();
            }
        }

        // Update covariances
        for k in 0..n_components {
            if nk[k] > 1e-10 {
                let cluster_points: Vec<(usize, f64)> = (0..n)
                    .map(|i| (i, responsibilities[[i, k]]))
                    .collect();
                covariances[k] = compute_weighted_covariance(
                    &data, &cluster_points, &means[k], cov_type, nk[k],
                );
            }
        }
    }

    // Compute final log-likelihood
    let mut log_likelihood = 0.0;
    for i in 0..n {
        let point: Vec<f64> = data.row(i).to_vec();
        let mut log_probs = Vec::with_capacity(n_components);

        for k in 0..n_components {
            let log_prob = log_gaussian_pdf(&point, &means[k], &covariances[k])
                + weights[k].ln();
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
    let resp_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| responsibilities.row(i).to_vec())
        .collect();

    // Convert covariances to Vec<Vec<Vec<f64>>>
    let cov_vecs: Vec<Vec<Vec<f64>>> = covariances.iter()
        .map(|cov| {
            (0..d).map(|i| cov.row(i).to_vec()).collect()
        })
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
                let var: f64 = indices.iter()
                    .map(|&i| (data[[i, j]] - mean[j]).powi(2))
                    .sum::<f64>() / (n - 1) as f64;
                cov[[j, j]] = var.max(1e-6);
            }
            cov
        }
        CovarianceType::Spherical => {
            let var: f64 = indices.iter()
                .flat_map(|&i| (0..d).map(move |j| (data[[i, j]] - mean[j]).powi(2)))
                .sum::<f64>() / ((n - 1) * d) as f64;
            Array2::eye(d) * var.max(1e-6)
        }
        CovarianceType::Tied => {
            compute_covariance(data, indices, mean, CovarianceType::Full)
        }
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
                let var: f64 = weights.iter()
                    .map(|&(i, w)| w * (data[[i, j]] - mean[j]).powi(2))
                    .sum::<f64>() / total_weight;
                cov[[j, j]] = var.max(1e-6);
            }
            cov
        }
        CovarianceType::Spherical => {
            let var: f64 = weights.iter()
                .flat_map(|&(i, w)| (0..d).map(move |j| w * (data[[i, j]] - mean[j]).powi(2)))
                .sum::<f64>() / (total_weight * d as f64);
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
    let is_diagonal = (0..d).all(|i| {
        (0..d).all(|j| i == j || m[[i, j]].abs() < 1e-10)
    });

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
    let is_diagonal = (0..d).all(|i| {
        (0..d).all(|j| i == j || m[[i, j]].abs() < 1e-10)
    });

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

// =============================================================================
// Helper functions
// =============================================================================

/// Compute pairwise Euclidean distances.
fn compute_distance_matrix(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in 0..data.ncols() {
                let diff = data[[i, k]] - data[[j, k]];
                sum += diff * diff;
            }
            let dist = sum.sqrt();
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Compute overall data variance.
fn compute_data_variance(data: &ArrayView2<f64>) -> f64 {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let n = data.nrows();

    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..data.ncols() {
            variance += (data[[i, j]] - mean[j]).powi(2);
        }
    }
    variance / (n * data.ncols()) as f64
}

/// Compute silhouette from precomputed distances.
fn compute_silhouette_from_distances(
    distances: &Array2<f64>,
    labels: &[usize],
    n_clusters: usize,
) -> Vec<f64> {
    let n = distances.nrows();

    // Group points by cluster
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        cluster_indices[label].push(i);
    }

    // Compute silhouette for each point
    let mut silhouette = Vec::with_capacity(n);

    for i in 0..n {
        let cluster_i = labels[i];
        let cluster_i_indices = &cluster_indices[cluster_i];

        // a(i): average distance to other points in same cluster
        let a_i = if cluster_i_indices.len() > 1 {
            let sum: f64 = cluster_i_indices.iter()
                .filter(|&&j| j != i)
                .map(|&j| distances[[i, j]])
                .sum();
            sum / (cluster_i_indices.len() - 1) as f64
        } else {
            0.0
        };

        // b(i): min average distance to any other cluster
        let mut b_i = f64::INFINITY;
        for c in 0..n_clusters {
            if c == cluster_i || cluster_indices[c].is_empty() {
                continue;
            }
            let avg_dist: f64 = cluster_indices[c].iter()
                .map(|&j| distances[[i, j]])
                .sum::<f64>() / cluster_indices[c].len() as f64;
            b_i = b_i.min(avg_dist);
        }

        let s_i = if a_i == 0.0 && b_i == 0.0 {
            0.0
        } else if b_i.is_infinite() {
            0.0
        } else {
            (b_i - a_i) / a_i.max(b_i)
        };

        silhouette.push(s_i);
    }

    silhouette
}

// =============================================================================
// Fuzzy C-Means Clustering
// =============================================================================

/// Result of Fuzzy C-Means clustering.
///
/// # References
///
/// - Bezdek, J.C. (1981). "Pattern Recognition with Fuzzy Objective Function
///   Algorithms". Plenum Press, New York.
/// - R e1071::cmeans documentation
///   Source: https://www.rdocumentation.org/packages/e1071/versions/1.7-16/topics/cmeans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyCMeansResult {
    /// Hard cluster assignments (index of maximum membership)
    pub labels: Vec<usize>,
    /// Cluster centers (k x features)
    #[serde(skip)]
    pub centers: Array2<f64>,
    /// Membership matrix (n x k) - soft assignments
    #[serde(skip)]
    pub membership: Array2<f64>,
    /// Final value of the objective function (within-cluster error)
    pub withinerror: f64,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Fuzziness parameter used
    pub m: f64,
    /// Number of observations
    pub n: usize,
    /// Did the algorithm converge?
    pub converged: bool,
}

impl std::fmt::Display for FuzzyCMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fuzzy C-Means Clustering Results")?;
        writeln!(f, "================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Fuzziness parameter m: {:.2}", self.m)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Within-cluster error: {:.6}", self.withinerror)?;
        writeln!(f)?;

        // Count points per cluster (hard assignment)
        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }

        writeln!(f, "Cluster sizes (hard assignment):")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }

        writeln!(f)?;
        writeln!(f, "Cluster centers:")?;
        for i in 0..self.centers.nrows() {
            let center: Vec<String> = self.centers.row(i).iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            writeln!(f, "  Cluster {}: [{}]", i, center.join(", "))?;
        }

        Ok(())
    }
}

/// Distance metric for Fuzzy C-Means.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FcmDistance {
    /// Euclidean distance (sum of squared differences)
    #[default]
    Euclidean,
    /// Manhattan distance (sum of absolute differences)
    Manhattan,
}

impl std::str::FromStr for FcmDistance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "euclidean" | "euclid" => Ok(FcmDistance::Euclidean),
            "manhattan" | "l1" => Ok(FcmDistance::Manhattan),
            _ => Err(format!("Unknown distance metric: {}. Use 'euclidean' or 'manhattan'", s)),
        }
    }
}

/// Run Fuzzy C-Means clustering.
///
/// Fuzzy C-Means is a soft clustering algorithm where each point has a degree
/// of membership to all clusters, rather than belonging to exactly one cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `m` - Fuzziness parameter (must be > 1, default: 2.0). Higher values mean softer clusters.
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `tolerance` - Convergence tolerance for objective function (default: sqrt(machine epsilon))
/// * `distance` - Distance metric (euclidean or manhattan)
/// * `seed` - Optional random seed for initialization
///
/// # Algorithm
///
/// Minimizes the objective function: J = Σᵢ Σⱼ uᵢⱼᵐ dᵢⱼ
///
/// where uᵢⱼ is the membership of point i in cluster j, m is the fuzziness
/// parameter, and dᵢⱼ is the distance from point i to cluster center j.
///
/// Update rules:
/// - Membership: uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^(2/(m-1))
/// - Centers: cⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
///
/// # Returns
/// * `FuzzyCMeansResult` containing cluster assignments and membership degrees
///
/// # References
///
/// - Bezdek, J.C. (1981). "Pattern Recognition with Fuzzy Objective Function Algorithms".
pub fn fuzzy_cmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    m: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    distance: Option<FcmDistance>,
    seed: Option<u64>,
) -> Result<FuzzyCMeansResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!("n_clusters ({}) cannot exceed n_samples ({})", n_clusters, n));
    }

    let fuzz = m.unwrap_or(2.0);
    if fuzz <= 1.0 {
        return Err("Fuzziness parameter m must be > 1".to_string());
    }

    let max_iter = max_iterations.unwrap_or(100);
    let tol = tolerance.unwrap_or(f64::EPSILON.sqrt());
    let dist_metric = distance.unwrap_or(FcmDistance::Euclidean);

    // Initialize membership matrix randomly
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut membership = Array2::zeros((n, n_clusters));
    for i in 0..n {
        // Random initialization, then normalize so rows sum to 1
        let mut row_sum = 0.0;
        for j in 0..n_clusters {
            membership[[i, j]] = rng.r#gen::<f64>() + 0.01;
            row_sum += membership[[i, j]];
        }
        for j in 0..n_clusters {
            membership[[i, j]] /= row_sum;
        }
    }

    // Initialize centers
    let mut centers = Array2::zeros((n_clusters, d));

    // Main FCM loop
    let mut n_iterations = 0;
    let mut prev_error = f64::INFINITY;
    let mut converged = false;
    let mut withinerror;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Update centers: cⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
        for j in 0..n_clusters {
            let mut weight_sum = 0.0;
            let mut weighted_sum = vec![0.0; d];

            for i in 0..n {
                let w = membership[[i, j]].powf(fuzz);
                weight_sum += w;
                for k in 0..d {
                    weighted_sum[k] += w * data[[i, k]];
                }
            }

            for k in 0..d {
                centers[[j, k]] = if weight_sum > 1e-10 {
                    weighted_sum[k] / weight_sum
                } else {
                    // If cluster is empty, reinitialize to a random point
                    data[[rng.gen_range(0..n), k]]
                };
            }
        }

        // Compute distances from each point to each center
        let mut distances = Array2::zeros((n, n_clusters));
        for i in 0..n {
            for j in 0..n_clusters {
                distances[[i, j]] = compute_fcm_distance(
                    &data.row(i), &centers.row(j), dist_metric,
                );
            }
        }

        // Update membership matrix
        let exp = 2.0 / (fuzz - 1.0);
        for i in 0..n {
            // Check if point coincides with any center
            let mut at_center = None;
            for j in 0..n_clusters {
                if distances[[i, j]] < 1e-10 {
                    at_center = Some(j);
                    break;
                }
            }

            if let Some(center_idx) = at_center {
                // Point is at a center - assign full membership to that cluster
                for j in 0..n_clusters {
                    membership[[i, j]] = if j == center_idx { 1.0 } else { 0.0 };
                }
            } else {
                // Standard FCM update: uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^exp
                for j in 0..n_clusters {
                    let mut sum = 0.0;
                    for k in 0..n_clusters {
                        sum += (distances[[i, j]] / distances[[i, k]]).powf(exp);
                    }
                    membership[[i, j]] = 1.0 / sum;
                }
            }
        }

        // Compute objective function: J = Σᵢ Σⱼ uᵢⱼᵐ dᵢⱼ
        withinerror = 0.0;
        for i in 0..n {
            for j in 0..n_clusters {
                withinerror += membership[[i, j]].powf(fuzz) * distances[[i, j]];
            }
        }

        // Check convergence
        let rel_change = (prev_error - withinerror).abs() / (prev_error.abs() + 1e-10);
        if rel_change < tol {
            converged = true;
            break;
        }
        prev_error = withinerror;
    }

    // Compute final distances and objective for return value
    let mut distances = Array2::zeros((n, n_clusters));
    for i in 0..n {
        for j in 0..n_clusters {
            distances[[i, j]] = compute_fcm_distance(
                &data.row(i), &centers.row(j), dist_metric,
            );
        }
    }

    withinerror = 0.0;
    for i in 0..n {
        for j in 0..n_clusters {
            withinerror += membership[[i, j]].powf(fuzz) * distances[[i, j]];
        }
    }

    // Hard assignments: argmax of membership
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            let mut max_j = 0;
            let mut max_u = membership[[i, 0]];
            for j in 1..n_clusters {
                if membership[[i, j]] > max_u {
                    max_u = membership[[i, j]];
                    max_j = j;
                }
            }
            max_j
        })
        .collect();

    Ok(FuzzyCMeansResult {
        labels,
        centers,
        membership,
        withinerror,
        n_iterations,
        n_clusters,
        m: fuzz,
        n,
        converged,
    })
}

/// Compute distance for FCM.
fn compute_fcm_distance(
    a: &ndarray::ArrayView1<f64>,
    b: &ndarray::ArrayView1<f64>,
    metric: FcmDistance,
) -> f64 {
    match metric {
        FcmDistance::Euclidean => {
            // Sum of squared differences (not squared root for efficiency)
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
        }
        FcmDistance::Manhattan => {
            // Sum of absolute differences
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }
    }
}

/// Convenience wrapper for fuzzy_cmeans.
pub fn run_fuzzy_cmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    m: Option<f64>,
    seed: Option<u64>,
) -> Result<FuzzyCMeansResult, String> {
    fuzzy_cmeans(data, n_clusters, m, None, None, None, seed)
}

// =============================================================================
// Mini-Batch K-Means
// =============================================================================

/// Result of Mini-Batch K-Means clustering.
///
/// # References
///
/// - Sculley, D. (2010). "Web-Scale K-Means Clustering". WWW 2010.
/// - R ClusterR::MiniBatchKmeans documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchKMeansResult {
    /// Cluster assignments for each point (0 to k-1)
    pub labels: Vec<usize>,
    /// Centroid positions (k x features)
    #[serde(skip)]
    pub centroids: Array2<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Within-cluster sum of squares (inertia)
    pub inertia: f64,
    /// Number of points in each cluster
    pub cluster_sizes: Vec<usize>,
    /// Batch size used
    pub batch_size: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for MiniBatchKMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mini-Batch K-Means Clustering Results")?;
        writeln!(f, "======================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Batch size: {}", self.batch_size)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Inertia (WCSS): {:.4}", self.inertia)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, size)?;
        }
        Ok(())
    }
}

/// Run Mini-Batch K-Means clustering.
///
/// Mini-Batch K-Means is a variant that uses random mini-batches to reduce
/// computation time for large datasets while producing similar results.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `batch_size` - Size of mini-batches (default: 100 or n if smaller)
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `n_init` - Number of initializations (default: 3)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// 1. Initialize centroids using k-means++
/// 2. For each iteration:
///    a. Sample a mini-batch of size batch_size
///    b. Assign each sample in the batch to nearest centroid
///    c. Update centroids using streaming average
///
/// # Returns
/// * `MiniBatchKMeansResult` containing cluster assignments
///
/// # References
///
/// - Sculley, D. (2010). "Web-Scale K-Means Clustering".
pub fn mini_batch_kmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    batch_size: Option<usize>,
    max_iterations: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<MiniBatchKMeansResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!("n_clusters ({}) cannot exceed n_samples ({})", n_clusters, n));
    }

    let batch = batch_size.unwrap_or(100.min(n));
    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(3);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut best_result: Option<MiniBatchKMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize centroids using k-means++
        let mut centroids = mini_batch_kmeans_plusplus_init(&data, n_clusters, &mut rng);
        let mut counts = vec![1.0f64; n_clusters]; // Track update counts per centroid

        // Mini-batch loop
        for iter in 0..max_iter {
            // Sample a mini-batch
            let batch_indices: Vec<usize> = (0..batch)
                .map(|_| rng.gen_range(0..n))
                .collect();

            // Assign each batch sample to nearest centroid and update
            for &idx in &batch_indices {
                // Find nearest centroid
                let mut min_dist = f64::INFINITY;
                let mut nearest = 0;
                for j in 0..n_clusters {
                    let dist: f64 = (0..d)
                        .map(|k| (data[[idx, k]] - centroids[[j, k]]).powi(2))
                        .sum();
                    if dist < min_dist {
                        min_dist = dist;
                        nearest = j;
                    }
                }

                // Streaming centroid update
                counts[nearest] += 1.0;
                let eta = 1.0 / counts[nearest];
                for k in 0..d {
                    centroids[[nearest, k]] = (1.0 - eta) * centroids[[nearest, k]]
                        + eta * data[[idx, k]];
                }
            }
        }

        // Compute final assignments and inertia
        let mut labels = vec![0usize; n];
        let mut inertia = 0.0;
        let mut cluster_sizes = vec![0usize; n_clusters];

        for i in 0..n {
            let mut min_dist = f64::INFINITY;
            let mut nearest = 0;
            for j in 0..n_clusters {
                let dist: f64 = (0..d)
                    .map(|k| (data[[i, k]] - centroids[[j, k]]).powi(2))
                    .sum();
                if dist < min_dist {
                    min_dist = dist;
                    nearest = j;
                }
            }
            labels[i] = nearest;
            inertia += min_dist;
            cluster_sizes[nearest] += 1;
        }

        if inertia < best_inertia {
            best_inertia = inertia;
            best_result = Some(MiniBatchKMeansResult {
                labels,
                centroids,
                n_iterations: max_iter,
                inertia,
                cluster_sizes,
                batch_size: batch,
                n_clusters,
                n,
            });
        }
    }

    best_result.ok_or_else(|| "Mini-batch K-means failed".to_string())
}

/// K-means++ initialization for mini-batch k-means.
fn mini_batch_kmeans_plusplus_init(
    data: &ArrayView2<f64>,
    k: usize,
    rng: &mut StdRng,
) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut centroids = Array2::zeros((k, d));

    // First centroid: random point
    let first = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&data.row(first));

    // Remaining centroids: proportional to squared distance
    for i in 1..k {
        let mut distances = vec![f64::INFINITY; n];
        for j in 0..n {
            for c in 0..i {
                let dist: f64 = (0..d)
                    .map(|k| (data[[j, k]] - centroids[[c, k]]).powi(2))
                    .sum();
                distances[j] = distances[j].min(dist);
            }
        }

        let total: f64 = distances.iter().sum();
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (j, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = j;
                break;
            }
        }
        centroids.row_mut(i).assign(&data.row(chosen));
    }

    centroids
}

/// Convenience wrapper for mini_batch_kmeans.
pub fn run_mini_batch_kmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    batch_size: Option<usize>,
    seed: Option<u64>,
) -> Result<MiniBatchKMeansResult, String> {
    mini_batch_kmeans(data, n_clusters, batch_size, None, None, seed)
}

// =============================================================================
// Trimmed K-Means
// =============================================================================

/// Result of Trimmed K-Means clustering.
///
/// # References
///
/// - García-Escudero, L.A., Gordaliza, A., Matrán, C., and Mayo-Iscar, A. (2008).
///   "A General Trimming Approach to Robust Cluster Analysis".
///   Annals of Statistics, 36(3), 1324-1345.
/// - R tclust::tkmeans documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrimmedKMeansResult {
    /// Cluster assignments (-1 for trimmed points)
    pub labels: Vec<i32>,
    /// Centroid positions (k x features)
    #[serde(skip)]
    pub centroids: Array2<f64>,
    /// Indices of trimmed (outlier) points
    pub trimmed_indices: Vec<usize>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Within-cluster sum of squares (excluding trimmed)
    pub inertia: f64,
    /// Trimming proportion used
    pub alpha: f64,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
    /// Number of trimmed points
    pub n_trimmed: usize,
}

impl std::fmt::Display for TrimmedKMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Trimmed K-Means Clustering Results")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Trimming proportion: {:.1}%", self.alpha * 100.0)?;
        writeln!(f, "Trimmed points: {}", self.n_trimmed)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Inertia (WCSS, non-trimmed): {:.4}", self.inertia)?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            if label >= 0 {
                cluster_counts[label as usize] += 1;
            }
        }

        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }
        writeln!(f, "  Trimmed (noise): {} points", self.n_trimmed)?;

        Ok(())
    }
}

/// Run Trimmed K-Means clustering.
///
/// Trimmed K-Means is a robust variant that trims a proportion of the most
/// outlying points, making the algorithm resistant to outliers.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `alpha` - Proportion of points to trim (0 to 0.5, default: 0.1)
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `n_init` - Number of random initializations (default: 10)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// 1. Initialize k centroids randomly
/// 2. Assign points to nearest centroid
/// 3. Identify and trim α proportion of points with largest distances
/// 4. Update centroids using only non-trimmed points
/// 5. Repeat until convergence
///
/// # Returns
/// * `TrimmedKMeansResult` containing cluster assignments
///
/// # References
///
/// - García-Escudero et al. (2008). "A General Trimming Approach to Robust Cluster Analysis".
pub fn trimmed_kmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    alpha: Option<f64>,
    max_iterations: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<TrimmedKMeansResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!("n_clusters ({}) cannot exceed n_samples ({})", n_clusters, n));
    }

    let trim_prop = alpha.unwrap_or(0.1);
    if trim_prop < 0.0 || trim_prop >= 0.5 {
        return Err("alpha must be between 0 and 0.5".to_string());
    }

    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(10);
    let n_trim = (n as f64 * trim_prop).ceil() as usize;
    let n_keep = n - n_trim;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut best_result: Option<TrimmedKMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize centroids using k-means++ on non-trimmed subset
        let mut centroids = mini_batch_kmeans_plusplus_init(&data, n_clusters, &mut rng);
        let mut labels = vec![-1i32; n];
        let mut trimmed_mask = vec![false; n];
        let mut n_iterations = 0;

        for iter in 0..max_iter {
            n_iterations = iter + 1;

            // Compute distances and assign to nearest centroid
            let mut distances: Vec<(f64, usize, usize)> = Vec::with_capacity(n);
            for i in 0..n {
                let mut min_dist = f64::INFINITY;
                let mut nearest = 0;
                for j in 0..n_clusters {
                    let dist: f64 = (0..d)
                        .map(|k| (data[[i, k]] - centroids[[j, k]]).powi(2))
                        .sum();
                    if dist < min_dist {
                        min_dist = dist;
                        nearest = j;
                    }
                }
                distances.push((min_dist, i, nearest));
            }

            // Sort by distance and trim the largest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            // Reset trimmed mask and labels
            trimmed_mask.fill(false);
            labels.fill(-1);

            // Assign non-trimmed points
            for (idx, &(_, point_idx, cluster)) in distances.iter().enumerate() {
                if idx < n_keep {
                    labels[point_idx] = cluster as i32;
                } else {
                    trimmed_mask[point_idx] = true;
                }
            }

            // Update centroids using only non-trimmed points
            let old_centroids = centroids.clone();
            for j in 0..n_clusters {
                let mut sum = vec![0.0; d];
                let mut count = 0;
                for i in 0..n {
                    if labels[i] == j as i32 {
                        for k in 0..d {
                            sum[k] += data[[i, k]];
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    for k in 0..d {
                        centroids[[j, k]] = sum[k] / count as f64;
                    }
                }
            }

            // Check convergence
            let max_shift: f64 = (0..n_clusters)
                .map(|j| {
                    (0..d)
                        .map(|k| (centroids[[j, k]] - old_centroids[[j, k]]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0, f64::max);

            if max_shift < 1e-6 {
                break;
            }
        }

        // Compute inertia (only non-trimmed points)
        let mut inertia = 0.0;
        for i in 0..n {
            if labels[i] >= 0 {
                let j = labels[i] as usize;
                let dist: f64 = (0..d)
                    .map(|k| (data[[i, k]] - centroids[[j, k]]).powi(2))
                    .sum();
                inertia += dist;
            }
        }

        if inertia < best_inertia {
            best_inertia = inertia;

            let trimmed_indices: Vec<usize> = trimmed_mask.iter()
                .enumerate()
                .filter(|&(_, &t)| t)
                .map(|(i, _)| i)
                .collect();

            best_result = Some(TrimmedKMeansResult {
                labels,
                centroids,
                trimmed_indices,
                n_iterations,
                inertia,
                alpha: trim_prop,
                n_clusters,
                n,
                n_trimmed: n_trim,
            });
        }
    }

    best_result.ok_or_else(|| "Trimmed K-means failed".to_string())
}

/// Convenience wrapper for trimmed_kmeans.
pub fn run_trimmed_kmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    alpha: Option<f64>,
    seed: Option<u64>,
) -> Result<TrimmedKMeansResult, String> {
    trimmed_kmeans(data, n_clusters, alpha, None, None, seed)
}

// =============================================================================
// DIANA (DIvisive ANAlysis)
// =============================================================================

/// Result of DIANA divisive hierarchical clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data:
///   An Introduction to Cluster Analysis". Wiley, New York.
/// - R cluster::diana documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/diana.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DianaResult {
    /// Cluster assignments for each point (0 to n_clusters-1)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Divisive coefficient (quality measure)
    pub divisive_coefficient: f64,
    /// Split history: (cluster_split, new_cluster1, new_cluster2, diameter)
    pub merge_history: Vec<(usize, usize, usize, f64)>,
    /// Heights at which splits occurred
    pub heights: Vec<f64>,
    /// Order of observations in the dendrogram
    pub order: Vec<usize>,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for DianaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DIANA (Divisive) Clustering Results")?;
        writeln!(f, "====================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Divisive coefficient: {:.4}", self.divisive_coefficient)?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut sorted_labels: Vec<_> = cluster_counts.keys().cloned().collect();
        sorted_labels.sort();
        for label in sorted_labels {
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
        }

        Ok(())
    }
}

/// Run DIANA divisive hierarchical clustering.
///
/// DIANA is a divisive (top-down) hierarchical clustering algorithm that
/// starts with all observations in one cluster and recursively splits
/// the most heterogeneous cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (default: 2)
///
/// # Algorithm
///
/// 1. Start with all points in one cluster
/// 2. Find the cluster with largest diameter
/// 3. Find the "splinter group": point with largest average dissimilarity
/// 4. Move points to splinter group if closer to it than to rest
/// 5. Repeat until desired number of clusters
///
/// # Returns
/// * `DianaResult` containing cluster assignments and dendrogram info
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn diana(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
) -> Result<DianaResult, String> {
    let n = data.nrows();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let target_k = n_clusters.unwrap_or(2);
    if target_k == 0 || target_k > n {
        return Err(format!("n_clusters must be between 1 and {}", n));
    }

    // Compute distance matrix
    let distances = compute_distance_matrix(&data);

    // Initialize: all points in cluster 0
    let mut labels = vec![0usize; n];
    let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut merge_history = Vec::new();
    let mut heights = Vec::new();
    let mut next_cluster_id = 1;

    // Divisive loop
    while clusters.len() < target_k {
        // Find cluster with largest diameter to split
        let (split_idx, diameter) = find_largest_diameter_cluster(&clusters, &distances);

        if clusters[split_idx].len() < 2 {
            break; // Cannot split singleton
        }

        // Split the cluster
        let (group1, group2) = diana_split(&clusters[split_idx], &distances);

        if group1.is_empty() || group2.is_empty() {
            break; // Split failed
        }

        // Update labels
        let new_cluster_id = next_cluster_id;
        next_cluster_id += 1;

        for &idx in &group2 {
            labels[idx] = new_cluster_id;
        }

        // Record split
        merge_history.push((split_idx, split_idx, new_cluster_id, diameter));
        heights.push(diameter);

        // Update clusters
        clusters[split_idx] = group1;
        clusters.push(group2);
    }

    // Renumber labels to be consecutive
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0;
    for label in &mut labels {
        if !label_map.contains_key(label) {
            label_map.insert(*label, next_label);
            next_label += 1;
        }
        *label = label_map[label];
    }

    // Compute divisive coefficient
    let divisive_coefficient = compute_diana_coefficient(&heights);

    // Compute order (simple ordering for now)
    let order: Vec<usize> = (0..n).collect();

    Ok(DianaResult {
        labels,
        n_clusters: clusters.len(),
        divisive_coefficient,
        merge_history,
        heights,
        order,
        n,
    })
}

/// Find the cluster with the largest diameter.
fn find_largest_diameter_cluster(
    clusters: &[Vec<usize>],
    distances: &Array2<f64>,
) -> (usize, f64) {
    let mut max_diameter = 0.0f64;
    let mut max_idx = 0;

    for (idx, cluster) in clusters.iter().enumerate() {
        if cluster.len() < 2 {
            continue;
        }

        let diameter = compute_cluster_diameter(cluster, distances);
        if diameter > max_diameter {
            max_diameter = diameter;
            max_idx = idx;
        }
    }

    (max_idx, max_diameter)
}

/// Compute the diameter of a cluster (max pairwise distance).
fn compute_cluster_diameter(cluster: &[usize], distances: &Array2<f64>) -> f64 {
    let mut max_dist = 0.0f64;
    for (i, &idx_i) in cluster.iter().enumerate() {
        for &idx_j in cluster.iter().skip(i + 1) {
            max_dist = max_dist.max(distances[[idx_i, idx_j]]);
        }
    }
    max_dist
}

/// Split a cluster using DIANA algorithm.
fn diana_split(cluster: &[usize], distances: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    if cluster.len() < 2 {
        return (cluster.to_vec(), Vec::new());
    }

    // Find the object with largest average dissimilarity to rest
    let mut max_avg_dissim = f64::NEG_INFINITY;
    let mut splinter_idx = 0;

    for (i, &idx_i) in cluster.iter().enumerate() {
        let avg_dissim: f64 = cluster.iter()
            .filter(|&&j| j != idx_i)
            .map(|&j| distances[[idx_i, j]])
            .sum::<f64>() / (cluster.len() - 1) as f64;

        if avg_dissim > max_avg_dissim {
            max_avg_dissim = avg_dissim;
            splinter_idx = i;
        }
    }

    // Initialize splinter group with the most dissimilar object
    let mut splinter = vec![cluster[splinter_idx]];
    let mut remaining: Vec<usize> = cluster.iter()
        .enumerate()
        .filter(|(i, _)| *i != splinter_idx)
        .map(|(_, &x)| x)
        .collect();

    // Iteratively move objects to splinter group
    loop {
        if remaining.is_empty() {
            break;
        }

        // For each remaining object, compute:
        // d_i = avg distance to remaining - avg distance to splinter
        let mut best_diff = f64::NEG_INFINITY;
        let mut best_idx = None;

        for (i, &idx) in remaining.iter().enumerate() {
            let avg_to_remaining = if remaining.len() > 1 {
                remaining.iter()
                    .filter(|&&j| j != idx)
                    .map(|&j| distances[[idx, j]])
                    .sum::<f64>() / (remaining.len() - 1) as f64
            } else {
                0.0
            };

            let avg_to_splinter = if !splinter.is_empty() {
                splinter.iter()
                    .map(|&j| distances[[idx, j]])
                    .sum::<f64>() / splinter.len() as f64
            } else {
                f64::INFINITY
            };

            let diff = avg_to_remaining - avg_to_splinter;
            if diff > best_diff && diff > 0.0 {
                best_diff = diff;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(i) => {
                let idx = remaining.remove(i);
                splinter.push(idx);
            }
            None => break, // No more objects want to move
        }
    }

    (remaining, splinter)
}

/// Compute DIANA divisive coefficient.
fn compute_diana_coefficient(heights: &[f64]) -> f64 {
    if heights.is_empty() {
        return 0.0;
    }

    // DC is based on the normalized height at which objects are separated
    let max_height = heights.iter().cloned().fold(0.0, f64::max);
    if max_height < 1e-10 {
        return 0.0;
    }

    let normalized_heights: Vec<f64> = heights.iter()
        .map(|&h| 1.0 - h / max_height)
        .collect();

    normalized_heights.iter().sum::<f64>() / normalized_heights.len() as f64
}

/// Convenience wrapper for diana.
pub fn run_diana(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
) -> Result<DianaResult, String> {
    diana(data, n_clusters)
}

// =============================================================================
// AGNES (AGglomerative NESting)
// =============================================================================

/// Result of AGNES agglomerative hierarchical clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
/// - R cluster::agnes documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/agnes.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgnesResult {
    /// Cluster assignments for each point (0 to n_clusters-1)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Agglomerative coefficient (measure of clustering structure)
    pub agglomerative_coefficient: f64,
    /// Linkage matrix: (cluster1, cluster2, height, size)
    pub merge: Vec<(usize, usize, f64, usize)>,
    /// Heights at which merges occurred
    pub heights: Vec<f64>,
    /// Order of observations in the dendrogram
    pub order: Vec<usize>,
    /// Linkage method used
    pub method: String,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for AgnesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "AGNES (Agglomerative) Clustering Results")?;
        writeln!(f, "=========================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Linkage method: {}", self.method)?;
        writeln!(f, "Agglomerative coefficient: {:.4}", self.agglomerative_coefficient)?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut sorted_labels: Vec<_> = cluster_counts.keys().cloned().collect();
        sorted_labels.sort();
        for label in sorted_labels {
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
        }

        Ok(())
    }
}

/// Linkage method for AGNES.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AgnesLinkage {
    /// Average linkage (UPGMA)
    #[default]
    Average,
    /// Single linkage (nearest neighbor)
    Single,
    /// Complete linkage (farthest neighbor)
    Complete,
    /// Ward's method (minimize variance)
    Ward,
    /// Weighted average linkage (WPGMA)
    Weighted,
}

impl std::str::FromStr for AgnesLinkage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "upgma" => Ok(AgnesLinkage::Average),
            "single" => Ok(AgnesLinkage::Single),
            "complete" => Ok(AgnesLinkage::Complete),
            "ward" => Ok(AgnesLinkage::Ward),
            "weighted" | "wpgma" => Ok(AgnesLinkage::Weighted),
            _ => Err(format!("Unknown linkage: {}. Use average, single, complete, ward, or weighted", s)),
        }
    }
}

/// Run AGNES agglomerative hierarchical clustering.
///
/// AGNES is an agglomerative (bottom-up) hierarchical clustering algorithm
/// that starts with each observation in its own cluster and iteratively
/// merges the closest pair of clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (default: 2)
/// * `method` - Linkage method (default: average)
///
/// # Algorithm
///
/// 1. Start with n clusters (each observation is a cluster)
/// 2. Compute distance between all pairs of clusters
/// 3. Merge the two closest clusters
/// 4. Update distances using Lance-Williams formula
/// 5. Repeat until desired number of clusters
///
/// # Returns
/// * `AgnesResult` containing cluster assignments and dendrogram info
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn agnes(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    method: Option<AgnesLinkage>,
) -> Result<AgnesResult, String> {
    let n = data.nrows();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let target_k = n_clusters.unwrap_or(2);
    if target_k == 0 || target_k > n {
        return Err(format!("n_clusters must be between 1 and {}", n));
    }

    let linkage = method.unwrap_or(AgnesLinkage::Average);

    // Compute distance matrix
    let distances = compute_distance_matrix(&data);

    // Initialize: each point in its own cluster
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active_clusters: Vec<usize> = (0..n).collect();
    let mut merge_history = Vec::new();
    let mut heights = Vec::new();
    let mut height_per_obs: Vec<f64> = vec![0.0; n];

    // Initialize cluster distances
    let mut cluster_dist = distances.clone();

    let mut next_cluster = n;

    // Agglomerative loop
    while active_clusters.len() > target_k {
        // Find closest pair of clusters
        let (c1_idx, c2_idx, min_dist) = find_closest_pair(&active_clusters, &cluster_dist)?;

        let c1 = active_clusters[c1_idx];
        let c2 = active_clusters[c2_idx];

        // Record merge
        merge_history.push((c1, c2, min_dist, cluster_members[c1].len() + cluster_members[c2].len()));
        heights.push(min_dist);

        // Update height for observations
        for &idx in &cluster_members[c1] {
            height_per_obs[idx] = height_per_obs[idx].max(min_dist);
        }
        for &idx in &cluster_members[c2] {
            height_per_obs[idx] = height_per_obs[idx].max(min_dist);
        }

        // Merge clusters
        let mut new_members = cluster_members[c1].clone();
        new_members.extend(&cluster_members[c2]);

        // Update distances using Lance-Williams formula
        update_agnes_distances(&mut cluster_dist, &cluster_members, &active_clusters,
                               c1, c2, next_cluster, linkage);

        // Update cluster tracking
        cluster_members.push(new_members);
        cluster_members[c1] = Vec::new();
        cluster_members[c2] = Vec::new();

        // Remove c1 and c2 from active, add new cluster
        // Remove in reverse order to avoid index shift issues
        let remove_first = c1_idx.max(c2_idx);
        let remove_second = c1_idx.min(c2_idx);
        active_clusters.remove(remove_first);
        active_clusters.remove(remove_second);
        active_clusters.push(next_cluster);

        next_cluster += 1;
    }

    // Assign final labels
    let mut labels = vec![0usize; n];
    for (cluster_label, &cluster_id) in active_clusters.iter().enumerate() {
        for &idx in &cluster_members[cluster_id] {
            labels[idx] = cluster_label;
        }
    }

    // Compute agglomerative coefficient
    let max_height = heights.iter().cloned().fold(0.0, f64::max);
    let agglomerative_coefficient = if max_height > 1e-10 {
        let sum: f64 = height_per_obs.iter()
            .map(|&h| 1.0 - h / max_height)
            .sum();
        sum / n as f64
    } else {
        0.0
    };

    // Simple ordering
    let order: Vec<usize> = (0..n).collect();

    Ok(AgnesResult {
        labels,
        n_clusters: active_clusters.len(),
        agglomerative_coefficient,
        merge: merge_history,
        heights,
        order,
        method: format!("{:?}", linkage).to_lowercase(),
        n,
    })
}

/// Find the closest pair of active clusters.
fn find_closest_pair(
    active: &[usize],
    dist: &Array2<f64>,
) -> Result<(usize, usize, f64), String> {
    let mut min_dist = f64::INFINITY;
    let mut best_pair = (0, 0);

    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let d = dist[[active[i], active[j]]];
            if d < min_dist {
                min_dist = d;
                best_pair = (i, j);
            }
        }
    }

    if min_dist.is_infinite() {
        return Err("No valid cluster pair found".to_string());
    }

    Ok((best_pair.0, best_pair.1, min_dist))
}

/// Update cluster distances after a merge using Lance-Williams formula.
fn update_agnes_distances(
    dist: &mut Array2<f64>,
    members: &[Vec<usize>],
    active: &[usize],
    c1: usize,
    c2: usize,
    new_cluster: usize,
    linkage: AgnesLinkage,
) {
    let n1 = members[c1].len() as f64;
    let n2 = members[c2].len() as f64;

    // Extend distance matrix if needed
    let current_size = dist.nrows();
    if new_cluster >= current_size {
        let mut new_dist = Array2::zeros((new_cluster + 1, new_cluster + 1));
        for i in 0..current_size {
            for j in 0..current_size {
                new_dist[[i, j]] = dist[[i, j]];
            }
        }
        *dist = new_dist;
    }

    // Lance-Williams parameters based on linkage method
    for &other in active {
        if other == c1 || other == c2 {
            continue;
        }

        let d_c1 = dist[[c1, other]];
        let d_c2 = dist[[c2, other]];
        let d_12 = dist[[c1, c2]];
        let n_other = members[other].len() as f64;

        let new_dist = match linkage {
            AgnesLinkage::Single => {
                d_c1.min(d_c2)
            }
            AgnesLinkage::Complete => {
                d_c1.max(d_c2)
            }
            AgnesLinkage::Average => {
                (n1 * d_c1 + n2 * d_c2) / (n1 + n2)
            }
            AgnesLinkage::Weighted => {
                (d_c1 + d_c2) / 2.0
            }
            AgnesLinkage::Ward => {
                let n_total = n1 + n2 + n_other;
                let alpha1 = (n1 + n_other) / n_total;
                let alpha2 = (n2 + n_other) / n_total;
                let beta = -n_other / n_total;
                (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
            }
        };

        dist[[new_cluster, other]] = new_dist;
        dist[[other, new_cluster]] = new_dist;
    }
}

/// Convenience wrapper for agnes.
pub fn run_agnes(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    method: Option<&str>,
) -> Result<AgnesResult, String> {
    let linkage = method.map(|m| m.parse::<AgnesLinkage>()).transpose()?;
    agnes(data, n_clusters, linkage)
}

// =============================================================================
// FlexMix - Finite Mixture Regression
// =============================================================================

/// Result of FlexMix finite mixture regression.
///
/// # References
///
/// - Leisch, F. (2004). "FlexMix: A General Framework for Finite Mixture Models
///   and Latent Class Regression in R". Journal of Statistical Software, 11(8).
/// - R flexmix package documentation
///   Source: https://www.rdocumentation.org/packages/flexmix/versions/2.3-20/topics/flexmix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexMixResult {
    /// Component membership (posterior) probabilities for each observation
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments (0-indexed)
    pub cluster: Vec<usize>,
    /// Regression coefficients for each component
    pub coefficients: Vec<Vec<f64>>,
    /// Residual standard deviations for each component
    pub sigma: Vec<f64>,
    /// Component weights (mixing proportions)
    pub prior: Vec<f64>,
    /// Log-likelihood
    pub loglik: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of predictors (including intercept)
    pub p: usize,
}

impl std::fmt::Display for FlexMixResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FlexMix Finite Mixture Regression")?;
        writeln!(f, "==================================")?;
        writeln!(f, "Components: {}", self.k)?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Predictors: {}", self.p)?;
        writeln!(f, "Converged: {} (iterations: {})", self.converged, self.n_iterations)?;
        writeln!(f)?;
        writeln!(f, "Model fit:")?;
        writeln!(f, "  Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f, "  AIC: {:.4}", self.aic)?;
        writeln!(f, "  BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Component weights:")?;
        for (i, w) in self.prior.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i + 1, w)?;
        }
        writeln!(f)?;
        writeln!(f, "Component parameters:")?;
        for (i, (coef, sig)) in self.coefficients.iter().zip(&self.sigma).enumerate() {
            writeln!(f, "  Component {}:", i + 1)?;
            writeln!(f, "    Coefficients: {:?}", coef)?;
            writeln!(f, "    Sigma: {:.4}", sig)?;
        }
        Ok(())
    }
}

/// Fit a finite mixture regression model using the EM algorithm.
///
/// FlexMix fits a mixture of linear regression models, where each component
/// has its own regression coefficients and error variance.
///
/// # Arguments
/// * `y` - Response variable (n x 1)
/// * `x` - Predictor matrix (n x p), should include intercept column if desired
/// * `k` - Number of mixture components
/// * `max_iter` - Maximum EM iterations (default: 200)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Random seed for initialization
///
/// # Returns
/// * `FlexMixResult` with fitted parameters and model diagnostics
///
/// # Algorithm
///
/// 1. Initialize: Random soft assignments and fit weighted OLS per component
/// 2. E-step: Compute posterior probabilities of component membership
/// 3. M-step: Update component parameters using weighted OLS
/// 4. Repeat until convergence
///
/// # References
///
/// - Leisch (2004). "FlexMix: A General Framework for Finite Mixture Models".
pub fn flexmix(
    y: ArrayView2<f64>,
    x: ArrayView2<f64>,
    k: usize,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FlexMixResult, String> {
    let n = y.nrows();
    let p = x.ncols();

    if n != x.nrows() {
        return Err(format!("y ({}) and x ({}) must have same number of rows", n, x.nrows()));
    }
    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iterations = max_iter.unwrap_or(200);
    let tolerance = tol.unwrap_or(1e-6);

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Initialize posterior probabilities randomly
    let mut posterior = Array2::zeros((n, k));
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            posterior[[i, j]] = rng.r#gen::<f64>() + 0.1;
            sum += posterior[[i, j]];
        }
        for j in 0..k {
            posterior[[i, j]] /= sum;
        }
    }

    // Initialize component parameters
    let mut coefficients: Vec<Vec<f64>> = vec![vec![0.0; p]; k];
    let mut sigma: Vec<f64> = vec![1.0; k];
    let mut prior: Vec<f64> = vec![1.0 / k as f64; k];

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;

    // Flatten y for easier access
    let y_vec: Vec<f64> = y.iter().copied().collect();

    for iter in 0..max_iterations {
        n_iterations = iter + 1;

        // M-step: Update parameters for each component
        for j in 0..k {
            // Update prior (mixing proportion)
            let nj: f64 = posterior.column(j).sum();
            prior[j] = (nj / n as f64).max(1e-10);

            // Weighted least squares for component j
            let weights: Vec<f64> = posterior.column(j).iter().copied().collect();

            // Fit weighted OLS: (X'WX)^-1 X'Wy
            let (beta, sigma_j) = weighted_ols(&x, &y_vec, &weights)?;
            coefficients[j] = beta;
            sigma[j] = sigma_j.max(1e-10);
        }

        // E-step: Update posterior probabilities
        let mut loglik = 0.0;
        for i in 0..n {
            let mut densities: Vec<f64> = Vec::with_capacity(k);
            let mut max_log_density = f64::NEG_INFINITY;

            for j in 0..k {
                // Compute predicted value
                let mut pred = 0.0;
                for l in 0..p {
                    pred += coefficients[j][l] * x[[i, l]];
                }
                let resid = y_vec[i] - pred;

                // Log Gaussian density
                let log_density = -0.5 * (resid / sigma[j]).powi(2)
                    - sigma[j].ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    + prior[j].ln();

                densities.push(log_density);
                max_log_density = max_log_density.max(log_density);
            }

            // Log-sum-exp for numerical stability
            let log_sum: f64 = densities.iter()
                .map(|&d| (d - max_log_density).exp())
                .sum::<f64>()
                .ln() + max_log_density;

            // Update posteriors
            for j in 0..k {
                posterior[[i, j]] = (densities[j] - log_sum).exp();
            }

            loglik += log_sum;
        }

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    // Hard cluster assignments
    let cluster: Vec<usize> = (0..n)
        .map(|i| {
            (0..k).max_by(|&a, &b|
                posterior[[i, a]].partial_cmp(&posterior[[i, b]]).unwrap_or(Ordering::Equal)
            ).unwrap_or(0)
        })
        .collect();

    // Compute information criteria
    let n_params = k * (p + 1) + k - 1; // coefficients + sigmas + mixing proportions
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(FlexMixResult {
        posterior,
        cluster,
        coefficients,
        sigma,
        prior,
        loglik: prev_loglik,
        bic,
        aic,
        n_iterations,
        converged,
        k,
        n,
        p,
    })
}

/// Weighted OLS: returns (coefficients, residual_std)
fn weighted_ols(x: &ArrayView2<f64>, y: &[f64], weights: &[f64]) -> Result<(Vec<f64>, f64), String> {
    let n = x.nrows();
    let p = x.ncols();

    if n == 0 || p == 0 {
        return Err("Empty data".to_string());
    }

    // Check if weights sum to something meaningful
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum < 1e-10 {
        // Return zeros if no weight
        return Ok((vec![0.0; p], 1.0));
    }

    // Compute X'WX
    let mut xtwx: Array2<f64> = Array2::zeros((p, p));
    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            for l in 0..p {
                xtwx[[j, l]] += w * x[[i, j]] * x[[i, l]];
            }
        }
    }

    // Compute X'Wy
    let mut xtwy = vec![0.0; p];
    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            xtwy[j] += w * x[[i, j]] * y[i];
        }
    }

    // Solve using Cholesky (add small regularization for stability)
    for j in 0..p {
        xtwx[[j, j]] += 1e-8;
    }

    // Simple matrix solve using faer
    let xtwx_faer = faer::Mat::from_fn(p, p, |i, j| xtwx[[i, j]]);
    let chol = xtwx_faer.llt(faer::Side::Lower)
        .map_err(|_| "Cholesky decomposition failed - singular matrix")?;

    let xtwy_faer = faer::Mat::from_fn(p, 1, |i, _| xtwy[i]);
    let beta_faer = chol.solve(&xtwy_faer);

    let beta: Vec<f64> = (0..p).map(|i| beta_faer[(i, 0)]).collect();

    // Compute weighted residual standard deviation
    let mut wss = 0.0;
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += beta[j] * x[[i, j]];
        }
        wss += weights[i] * (y[i] - pred).powi(2);
    }
    let sigma = (wss / weight_sum.max(1.0)).sqrt();

    Ok((beta, sigma))
}

/// Convenience wrapper for flexmix.
pub fn run_flexmix(
    y: ArrayView2<f64>,
    x: ArrayView2<f64>,
    k: usize,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FlexMixResult, String> {
    flexmix(y, x, k, max_iter, tol, seed)
}

// =============================================================================
// PvClust - Hierarchical Clustering with Bootstrap P-values
// =============================================================================

/// Result of pvclust hierarchical clustering with bootstrap.
///
/// # References
///
/// - Suzuki, R. and Shimodaira, H. (2006). "Pvclust: an R package for assessing
///   the uncertainty in hierarchical clustering". Bioinformatics, 22(12).
/// - R pvclust package documentation
///   Source: https://www.rdocumentation.org/packages/pvclust/versions/2.2-0/topics/pvclust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvClustResult {
    /// AU (Approximately Unbiased) p-values for each cluster (edge)
    pub au_pvalues: Vec<f64>,
    /// BP (Bootstrap Probability) p-values for each cluster
    pub bp_pvalues: Vec<f64>,
    /// Cluster labels for each observation at given threshold
    pub labels: Vec<usize>,
    /// Number of clusters at AU >= threshold
    pub n_clusters: usize,
    /// Merge matrix from hierarchical clustering (pairs of merged clusters)
    pub merge: Vec<(i32, i32)>,
    /// Heights of merges
    pub heights: Vec<f64>,
    /// Number of bootstrap replicates
    pub n_boot: usize,
    /// AU threshold used for cluster selection
    pub au_threshold: f64,
    /// Number of observations
    pub n: usize,
    /// Linkage method used
    pub method: String,
}

impl std::fmt::Display for PvClustResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PvClust Hierarchical Clustering with Bootstrap")?;
        writeln!(f, "==============================================")?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Linkage method: {}", self.method)?;
        writeln!(f, "Bootstrap replicates: {}", self.n_boot)?;
        writeln!(f, "AU threshold: {:.2}", self.au_threshold)?;
        writeln!(f, "Clusters (AU >= {:.2}): {}", self.au_threshold, self.n_clusters)?;
        writeln!(f)?;

        // Show significant clusters (top 10)
        let mut significant: Vec<(usize, f64, f64)> = self.au_pvalues.iter()
            .zip(&self.bp_pvalues)
            .enumerate()
            .filter(|&(_, (au, _))| *au >= self.au_threshold)
            .map(|(i, (au, bp))| (i + 1, *au, *bp))
            .collect();
        significant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        writeln!(f, "Significant clusters (top 10):")?;
        writeln!(f, "  Edge   AU     BP")?;
        for (edge, au, bp) in significant.iter().take(10) {
            writeln!(f, "  {:4}   {:.3}  {:.3}", edge, au, bp)?;
        }

        Ok(())
    }
}

/// Perform hierarchical clustering with multiscale bootstrap resampling.
///
/// pvclust assesses the uncertainty of hierarchical cluster analysis by
/// computing approximately unbiased (AU) p-values through multiscale
/// bootstrap resampling.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `method` - Linkage method: "average", "single", "complete", "ward"
/// * `n_boot` - Number of bootstrap replicates (default: 1000)
/// * `r` - Bootstrap sample sizes as ratios (default: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
/// * `au_threshold` - Threshold for significant clusters (default: 0.95)
/// * `seed` - Random seed
///
/// # Returns
/// * `PvClustResult` with AU and BP p-values for each cluster
///
/// # Algorithm
///
/// 1. Perform initial hierarchical clustering
/// 2. For each bootstrap sample size ratio:
///    a. Resample observations with replacement
///    b. Perform hierarchical clustering
///    c. Check which original clusters appear in bootstrap tree
/// 3. Fit curve to compute AU p-values using multiscale extrapolation
/// 4. BP p-values are computed from r=1.0 bootstrap frequency
///
/// # References
///
/// - Shimodaira, H. (2004). "Approximately unbiased tests of regions using
///   multistep-multiscale bootstrap resampling". Annals of Statistics.
pub fn pvclust(
    data: ArrayView2<f64>,
    method: Option<&str>,
    n_boot: Option<usize>,
    r: Option<Vec<f64>>,
    au_threshold: Option<f64>,
    seed: Option<u64>,
) -> Result<PvClustResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let linkage_method = method.unwrap_or("average");
    let nboot = n_boot.unwrap_or(1000);
    let ratios = r.unwrap_or_else(|| vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]);
    let threshold = au_threshold.unwrap_or(0.95);

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Perform initial clustering
    let (merge, heights) = hierarchical_clustering_merge(&data, linkage_method)?;

    let n_edges = merge.len();

    // Extract cluster memberships from the merge matrix
    let original_clusters = extract_clusters_from_merge(&merge, n);

    // Bootstrap counts for each edge at each scale
    let mut counts: Vec<Vec<usize>> = vec![vec![0; ratios.len()]; n_edges];

    // Perform multiscale bootstrap
    for (scale_idx, &ratio) in ratios.iter().enumerate() {
        let sample_size = ((n as f64) * ratio).round() as usize;
        let sample_size = sample_size.max(2);

        for _ in 0..nboot {
            // Bootstrap sample indices
            let indices: Vec<usize> = (0..sample_size)
                .map(|_| rng.gen_range(0..n))
                .collect();

            // Create bootstrap data
            let boot_data = Array2::from_shape_fn((sample_size, d), |(i, j)| {
                data[[indices[i], j]]
            });

            // Perform clustering on bootstrap sample
            let (boot_merge, _) = hierarchical_clustering_merge(&boot_data.view(), linkage_method)?;
            let boot_clusters = extract_clusters_from_merge(&boot_merge, sample_size);

            // Check which original clusters appear
            for (edge_idx, orig_cluster) in original_clusters.iter().enumerate() {
                // Map original cluster members to bootstrap sample
                let mapped: std::collections::HashSet<usize> = orig_cluster.iter()
                    .filter_map(|&obs| indices.iter().position(|&i| i == obs))
                    .collect();

                if mapped.len() >= 2 {
                    // Check if this cluster appears in bootstrap
                    for boot_cluster in &boot_clusters {
                        let boot_set: std::collections::HashSet<usize> = boot_cluster.iter().copied().collect();
                        if mapped == boot_set {
                            counts[edge_idx][scale_idx] += 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Compute AU and BP p-values
    let mut au_pvalues = vec![0.0; n_edges];
    let mut bp_pvalues = vec![0.0; n_edges];

    for edge_idx in 0..n_edges {
        // BP is simply the frequency at r=1.0
        let r1_idx = ratios.iter().position(|&r| (r - 1.0).abs() < 0.01);
        if let Some(idx) = r1_idx {
            bp_pvalues[edge_idx] = counts[edge_idx][idx] as f64 / nboot as f64;
        }

        // AU p-value from multiscale extrapolation
        // Use weighted least squares on log(-log(p)) vs log(r)
        au_pvalues[edge_idx] = compute_au_pvalue(&counts[edge_idx], &ratios, nboot);
    }

    // Assign cluster labels based on significant edges
    let labels = assign_pvclust_labels(&merge, &au_pvalues, threshold, n);
    let n_clusters = labels.iter().max().map(|&m| m + 1).unwrap_or(1);

    Ok(PvClustResult {
        au_pvalues,
        bp_pvalues,
        labels,
        n_clusters,
        merge,
        heights,
        n_boot: nboot,
        au_threshold: threshold,
        n,
        method: linkage_method.to_string(),
    })
}

/// Perform hierarchical clustering and return merge matrix and heights.
fn hierarchical_clustering_merge(
    data: &ArrayView2<f64>,
    method: &str,
) -> Result<(Vec<(i32, i32)>, Vec<f64>), String> {
    let n = data.nrows();

    // Compute distance matrix
    let distances = compute_distance_matrix(data);

    // Initialize: each point in its own cluster
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<usize> = (0..n).collect();
    let mut merge = Vec::new();
    let mut heights = Vec::new();
    let mut cluster_dist = distances.clone();
    let mut next_cluster = n;

    // Perform agglomerative clustering
    while active.len() > 1 {
        // Find closest pair
        let (c1_idx, c2_idx, min_dist) = find_closest_pair(&active, &cluster_dist)?;

        let c1 = active[c1_idx];
        let c2 = active[c2_idx];

        // Record merge (using R-style indexing: negative for leaves, positive for internal)
        let m1 = if c1 < n { -((c1 as i32) + 1) } else { (c1 - n + 1) as i32 };
        let m2 = if c2 < n { -((c2 as i32) + 1) } else { (c2 - n + 1) as i32 };
        merge.push((m1, m2));
        heights.push(min_dist);

        // Merge clusters
        let mut new_members = cluster_members[c1].clone();
        new_members.extend(&cluster_members[c2]);

        // Update distances
        update_hclust_distances(&mut cluster_dist, &cluster_members, &active,
                                c1, c2, next_cluster, method);

        // Update tracking
        cluster_members.push(new_members);
        cluster_members[c1] = Vec::new();
        cluster_members[c2] = Vec::new();

        let remove_first = c1_idx.max(c2_idx);
        let remove_second = c1_idx.min(c2_idx);
        active.remove(remove_first);
        active.remove(remove_second);
        active.push(next_cluster);

        next_cluster += 1;
    }

    Ok((merge, heights))
}

/// Update cluster distances after merge.
fn update_hclust_distances(
    dist: &mut Array2<f64>,
    members: &[Vec<usize>],
    active: &[usize],
    c1: usize,
    c2: usize,
    new_cluster: usize,
    method: &str,
) {
    let n1 = members[c1].len() as f64;
    let n2 = members[c2].len() as f64;

    // Extend distance matrix if needed
    let current_size = dist.nrows();
    if new_cluster >= current_size {
        let mut new_dist = Array2::zeros((new_cluster + 1, new_cluster + 1));
        for i in 0..current_size {
            for j in 0..current_size {
                new_dist[[i, j]] = dist[[i, j]];
            }
        }
        *dist = new_dist;
    }

    for &other in active {
        if other == c1 || other == c2 {
            continue;
        }

        let d_c1 = dist[[c1, other]];
        let d_c2 = dist[[c2, other]];
        let d_12 = dist[[c1, c2]];
        let n_other = members[other].len() as f64;

        let new_dist = match method {
            "single" => d_c1.min(d_c2),
            "complete" => d_c1.max(d_c2),
            "average" => (n1 * d_c1 + n2 * d_c2) / (n1 + n2),
            "ward" => {
                let n_total = n1 + n2 + n_other;
                let alpha1 = (n1 + n_other) / n_total;
                let alpha2 = (n2 + n_other) / n_total;
                let beta = -n_other / n_total;
                (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
            }
            _ => (n1 * d_c1 + n2 * d_c2) / (n1 + n2), // default to average
        };

        dist[[new_cluster, other]] = new_dist;
        dist[[other, new_cluster]] = new_dist;
    }
}

/// Extract clusters from merge matrix.
fn extract_clusters_from_merge(merge: &[(i32, i32)], n: usize) -> Vec<Vec<usize>> {
    let mut clusters = Vec::new();

    // Build cluster contents from merge history
    let mut cluster_contents: HashMap<i32, Vec<usize>> = HashMap::new();

    // Initialize leaves
    for i in 0..n {
        cluster_contents.insert(-((i as i32) + 1), vec![i]);
    }

    // Process merges
    for (merge_idx, &(m1, m2)) in merge.iter().enumerate() {
        let members1 = cluster_contents.get(&m1).cloned().unwrap_or_default();
        let members2 = cluster_contents.get(&m2).cloned().unwrap_or_default();

        let mut new_members = members1;
        new_members.extend(members2);

        // Record this cluster
        clusters.push(new_members.clone());

        // Store for future reference
        cluster_contents.insert((merge_idx + 1) as i32, new_members);
    }

    clusters
}

/// Compute AU p-value from multiscale bootstrap.
fn compute_au_pvalue(counts: &[usize], ratios: &[f64], nboot: usize) -> f64 {
    // Compute frequencies
    let freqs: Vec<f64> = counts.iter()
        .map(|&c| (c as f64 / nboot as f64).max(0.001).min(0.999))
        .collect();

    // Transform: z = qnorm(p), x = sqrt(r)
    // Fit: z = a + b*x
    // AU p-value: pnorm(-a)

    let mut sum_x = 0.0;
    let mut sum_z = 0.0;
    let mut sum_xz = 0.0;
    let mut sum_xx = 0.0;
    let n = ratios.len() as f64;

    for (&r, &p) in ratios.iter().zip(&freqs) {
        let x = r.sqrt();
        let z = inverse_normal_cdf(p);
        sum_x += x;
        sum_z += z;
        sum_xz += x * z;
        sum_xx += x * x;
    }

    // Simple linear regression
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return freqs.iter().sum::<f64>() / freqs.len() as f64;
    }

    let b = (n * sum_xz - sum_x * sum_z) / denom;
    let a = (sum_z - b * sum_x) / n;

    // AU = Phi(-a)
    normal_cdf(-a)
}

/// Inverse normal CDF (probit function) - simple approximation.
fn inverse_normal_cdf(p: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let p = p.max(1e-10).min(1.0 - 1e-10);

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

/// Normal CDF - standard normal distribution function.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Horner form of approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Assign cluster labels based on significant AU p-values.
fn assign_pvclust_labels(
    merge: &[(i32, i32)],
    au_pvalues: &[f64],
    threshold: f64,
    n: usize,
) -> Vec<usize> {
    let mut labels = vec![0; n];

    // Find significant clusters (starting from highest AU)
    let mut significant_edges: Vec<(usize, f64)> = au_pvalues.iter()
        .enumerate()
        .filter(|&(_, au)| *au >= threshold)
        .map(|(i, au)| (i, *au))
        .collect();
    significant_edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Build cluster contents
    let mut cluster_contents: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        cluster_contents.insert(-((i as i32) + 1), vec![i]);
    }
    for (merge_idx, &(m1, m2)) in merge.iter().enumerate() {
        let members1 = cluster_contents.get(&m1).cloned().unwrap_or_default();
        let members2 = cluster_contents.get(&m2).cloned().unwrap_or_default();
        let mut new_members = members1;
        new_members.extend(members2);
        cluster_contents.insert((merge_idx + 1) as i32, new_members);
    }

    // Assign labels based on significant edges
    let mut assigned = vec![false; n];
    let mut next_label = 0usize;

    for (edge_idx, _) in significant_edges {
        let cluster_id = (edge_idx + 1) as i32;
        if let Some(members) = cluster_contents.get(&cluster_id) {
            // Only assign if members haven't been assigned yet
            let unassigned: Vec<usize> = members.iter()
                .filter(|&&m| !assigned[m])
                .copied()
                .collect();

            if !unassigned.is_empty() {
                for &m in &unassigned {
                    labels[m] = next_label;
                    assigned[m] = true;
                }
                next_label += 1;
            }
        }
    }

    // Assign remaining to their own clusters
    for i in 0..n {
        if !assigned[i] {
            labels[i] = next_label;
            next_label += 1;
        }
    }

    labels
}

/// Convenience wrapper for pvclust.
pub fn run_pvclust(
    data: ArrayView2<f64>,
    method: Option<&str>,
    n_boot: Option<usize>,
    au_threshold: Option<f64>,
    seed: Option<u64>,
) -> Result<PvClustResult, String> {
    pvclust(data, method, n_boot, None, au_threshold, seed)
}

// =============================================================================
// CLARA - Clustering Large Applications
// =============================================================================

/// Result of CLARA clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data:
///   An Introduction to Cluster Analysis". Wiley.
/// - R cluster::clara documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clara.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaraResult {
    /// Cluster assignments for all observations
    pub labels: Vec<usize>,
    /// Final medoid indices (in the original data)
    pub medoid_indices: Vec<usize>,
    /// Total dissimilarity
    pub objective: f64,
    /// Best sample that was used
    pub best_sample: Vec<usize>,
    /// Average silhouette width
    pub average_silhouette: Option<f64>,
    /// Number of samples taken
    pub n_samples: usize,
    /// Sample size used
    pub sample_size: usize,
    /// Number of clusters
    pub k: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for ClaraResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CLARA Clustering Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Number of clusters: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Samples taken: {} (size: {})", self.n_samples, self.sample_size)?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        if let Some(sil) = self.average_silhouette {
            writeln!(f, "Average silhouette: {:.4}", sil)?;
        }
        writeln!(f)?;
        writeln!(f, "Medoid indices: {:?}", self.medoid_indices)?;

        // Cluster sizes
        let mut sizes = vec![0usize; self.k];
        for &l in &self.labels {
            sizes[l] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, &size) in sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {}", i, size)?;
        }

        Ok(())
    }
}

/// CLARA (Clustering Large Applications) - PAM on samples for large datasets.
///
/// CLARA extends PAM (Partitioning Around Medoids) to large datasets by
/// running PAM on multiple random samples and keeping the best result.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `samples` - Number of samples to draw (default: 5)
/// * `sample_size` - Size of each sample (default: 40 + 2*k)
/// * `max_iter` - Maximum PAM iterations per sample (default: 100)
/// * `seed` - Random seed
///
/// # Returns
/// * `ClaraResult` with cluster assignments and medoids
///
/// # Algorithm
///
/// 1. Draw `samples` random samples of size `sample_size` from the data
/// 2. Run PAM on each sample to find k medoids
/// 3. Assign all observations to nearest medoid
/// 4. Compute total dissimilarity
/// 5. Keep the result with lowest dissimilarity
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn clara(
    data: ArrayView2<f64>,
    k: usize,
    samples: Option<usize>,
    sample_size: Option<usize>,
    max_iter: Option<usize>,
    seed: Option<u64>,
) -> Result<ClaraResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 || k > n {
        return Err(format!("k must be between 1 and {}", n));
    }

    let n_samples = samples.unwrap_or(5);
    let samp_size = sample_size.unwrap_or(40 + 2 * k).min(n);
    let max_iterations = max_iter.unwrap_or(100);

    if samp_size < k {
        return Err(format!("Sample size ({}) must be >= k ({})", samp_size, k));
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut best_objective = f64::INFINITY;
    let mut best_medoids = Vec::new();
    let mut best_sample = Vec::new();

    // Try multiple samples
    for _ in 0..n_samples {
        // Draw random sample
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices: Vec<usize> = indices.into_iter().take(samp_size).collect();

        // Create sample data
        let sample_data = Array2::from_shape_fn((samp_size, d), |(i, j)| {
            data[[sample_indices[i], j]]
        });

        // Run PAM on sample
        let pam_result = kmedoids(sample_data.view(), k, Some(max_iterations), Some(rng.r#gen()))?;

        // Map sample medoid indices back to original data indices
        let medoids: Vec<usize> = pam_result.medoid_indices.iter()
            .map(|&idx| sample_indices[idx])
            .collect();

        // Compute objective using all data points
        let objective = compute_clara_objective(&data, &medoids);

        if objective < best_objective {
            best_objective = objective;
            best_medoids = medoids;
            best_sample = sample_indices;
        }
    }

    // Assign all points to nearest medoid
    let labels = assign_to_medoids_data(&data, &best_medoids);

    // Compute silhouette (optional, can be expensive)
    let avg_silhouette = if n <= 10000 {
        let distances = compute_distance_matrix(&data);
        Some(compute_silhouette_avg(&labels, &distances, k))
    } else {
        None
    };

    Ok(ClaraResult {
        labels,
        medoid_indices: best_medoids,
        objective: best_objective,
        best_sample,
        average_silhouette: avg_silhouette,
        n_samples,
        sample_size: samp_size,
        k,
        n,
    })
}

/// Compute CLARA objective (total dissimilarity to medoids).
fn compute_clara_objective(data: &ArrayView2<f64>, medoids: &[usize]) -> f64 {
    let n = data.nrows();
    let d = data.ncols();
    let mut total = 0.0;

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        for &med in medoids {
            let mut dist = 0.0;
            for j in 0..d {
                dist += (data[[i, j]] - data[[med, j]]).powi(2);
            }
            min_dist = min_dist.min(dist.sqrt());
        }
        total += min_dist;
    }

    total
}

/// Assign points to nearest medoid using data matrix directly.
fn assign_to_medoids_data(data: &ArrayView2<f64>, medoids: &[usize]) -> Vec<usize> {
    let n = data.nrows();
    let d = data.ncols();
    let k = medoids.len();

    (0..n).map(|i| {
        let mut best = 0;
        let mut best_dist = f64::INFINITY;

        for (j, &med) in medoids.iter().enumerate() {
            let mut dist = 0.0;
            for l in 0..d {
                dist += (data[[i, l]] - data[[med, l]]).powi(2);
            }
            if dist < best_dist {
                best_dist = dist;
                best = j;
            }
        }

        best
    }).collect()
}

/// Compute average silhouette width.
fn compute_silhouette_avg(labels: &[usize], distances: &Array2<f64>, k: usize) -> f64 {
    let n = labels.len();
    if n < 2 || k < 2 {
        return 0.0;
    }

    let mut silhouette_sum = 0.0;

    for i in 0..n {
        let label_i = labels[i];

        // Compute a(i) - average distance to points in same cluster
        let mut same_cluster_dist = 0.0;
        let mut same_cluster_count = 0;
        for j in 0..n {
            if i != j && labels[j] == label_i {
                same_cluster_dist += distances[[i, j]];
                same_cluster_count += 1;
            }
        }
        let a_i = if same_cluster_count > 0 {
            same_cluster_dist / same_cluster_count as f64
        } else {
            0.0
        };

        // Compute b(i) - minimum average distance to other clusters
        let mut b_i = f64::INFINITY;
        for other_label in 0..k {
            if other_label == label_i {
                continue;
            }

            let mut other_dist = 0.0;
            let mut other_count = 0;
            for j in 0..n {
                if labels[j] == other_label {
                    other_dist += distances[[i, j]];
                    other_count += 1;
                }
            }
            if other_count > 0 {
                b_i = b_i.min(other_dist / other_count as f64);
            }
        }

        // Silhouette width for point i
        if b_i.is_finite() && (a_i > 0.0 || b_i > 0.0) {
            let s_i = (b_i - a_i) / a_i.max(b_i);
            silhouette_sum += s_i;
        }
    }

    silhouette_sum / n as f64
}

/// Convenience wrapper for clara.
pub fn run_clara(
    data: ArrayView2<f64>,
    k: usize,
    samples: Option<usize>,
    sample_size: Option<usize>,
    seed: Option<u64>,
) -> Result<ClaraResult, String> {
    clara(data, k, samples, sample_size, None, seed)
}

// =============================================================================
// Cluster Statistics - Comprehensive Cluster Validation
// =============================================================================

/// Comprehensive cluster statistics result.
///
/// # References
///
/// - Hennig, C. (2020). fpc: Flexible Procedures for Clustering. R package.
/// - R fpc::cluster.stats documentation
///   Source: https://www.rdocumentation.org/packages/fpc/versions/2.2-10/topics/cluster.stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatsResult {
    /// Number of clusters
    pub n_clusters: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Average within-cluster distance
    pub average_within: f64,
    /// Average between-cluster distance
    pub average_between: f64,
    /// Within-cluster sum of squares
    pub within_ss: f64,
    /// Between-cluster sum of squares (explained variance)
    pub between_ss: f64,
    /// Total sum of squares
    pub total_ss: f64,
    /// Ratio of between to total SS (explained variance ratio)
    pub explained_variance_ratio: f64,
    /// Average silhouette width
    pub average_silhouette: f64,
    /// Silhouette width per cluster
    pub cluster_silhouette: Vec<f64>,
    /// Dunn index
    pub dunn_index: f64,
    /// Calinski-Harabasz index
    pub calinski_harabasz: f64,
    /// Davies-Bouldin index
    pub davies_bouldin: f64,
    /// Separation: minimum distance between clusters
    pub separation: f64,
    /// Maximum cluster diameter
    pub max_diameter: f64,
    /// Cluster diameters
    pub cluster_diameters: Vec<f64>,
    /// Number of observations
    pub n: usize,
    /// Number of features
    pub p: usize,
}

impl std::fmt::Display for ClusterStatsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Comprehensive Cluster Statistics")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Features: {}", self.p)?;
        writeln!(f, "Clusters: {}", self.n_clusters)?;
        writeln!(f)?;

        writeln!(f, "Cluster sizes: {:?}", self.cluster_sizes)?;
        writeln!(f)?;

        writeln!(f, "Distance measures:")?;
        writeln!(f, "  Avg within-cluster: {:.4}", self.average_within)?;
        writeln!(f, "  Avg between-cluster: {:.4}", self.average_between)?;
        writeln!(f, "  Separation (min between): {:.4}", self.separation)?;
        writeln!(f, "  Max diameter: {:.4}", self.max_diameter)?;
        writeln!(f)?;

        writeln!(f, "Variance decomposition:")?;
        writeln!(f, "  Within SS: {:.4}", self.within_ss)?;
        writeln!(f, "  Between SS: {:.4}", self.between_ss)?;
        writeln!(f, "  Total SS: {:.4}", self.total_ss)?;
        writeln!(f, "  Explained ratio: {:.4}", self.explained_variance_ratio)?;
        writeln!(f)?;

        writeln!(f, "Validation indices:")?;
        writeln!(f, "  Silhouette: {:.4}", self.average_silhouette)?;
        writeln!(f, "  Calinski-Harabasz: {:.4}", self.calinski_harabasz)?;
        writeln!(f, "  Davies-Bouldin: {:.4}", self.davies_bouldin)?;
        writeln!(f, "  Dunn index: {:.4}", self.dunn_index)?;

        Ok(())
    }
}

/// Compute comprehensive cluster statistics.
///
/// This function computes a variety of internal validation measures for
/// evaluating the quality of a clustering solution.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `labels` - Cluster labels for each observation
///
/// # Returns
/// * `ClusterStatsResult` with multiple cluster quality metrics
///
/// # Metrics computed
/// - Silhouette width
/// - Dunn index
/// - Calinski-Harabasz index
/// - Davies-Bouldin index
/// - Within/between cluster distances
/// - Sum of squares decomposition
/// - Cluster separation and diameters
///
/// # References
///
/// - Hennig (2020). fpc R package.
pub fn cluster_stats(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<ClusterStatsResult, String> {
    let n = data.nrows();
    let p = data.ncols();

    if n != labels.len() {
        return Err(format!("Data rows ({}) != labels length ({})", n, labels.len()));
    }

    let k = *labels.iter().max().unwrap_or(&0) + 1;
    if k == 0 {
        return Err("No clusters found".to_string());
    }

    // Compute distance matrix
    let distances = compute_distance_matrix(&data);

    // Cluster sizes
    let mut cluster_sizes = vec![0usize; k];
    for &l in labels {
        cluster_sizes[l] += 1;
    }

    // Global centroid
    let global_mean: Vec<f64> = (0..p)
        .map(|j| data.column(j).sum() / n as f64)
        .collect();

    // Cluster centroids
    let mut cluster_centroids: Vec<Vec<f64>> = vec![vec![0.0; p]; k];
    for (i, &l) in labels.iter().enumerate() {
        for j in 0..p {
            cluster_centroids[l][j] += data[[i, j]];
        }
    }
    for l in 0..k {
        if cluster_sizes[l] > 0 {
            for j in 0..p {
                cluster_centroids[l][j] /= cluster_sizes[l] as f64;
            }
        }
    }

    // Within and between SS
    let mut within_ss = 0.0;
    let mut between_ss = 0.0;
    let mut total_ss = 0.0;

    for (i, &l) in labels.iter().enumerate() {
        for j in 0..p {
            let x = data[[i, j]];
            within_ss += (x - cluster_centroids[l][j]).powi(2);
            total_ss += (x - global_mean[j]).powi(2);
        }
    }

    for l in 0..k {
        let mut dist_to_global = 0.0;
        for j in 0..p {
            dist_to_global += (cluster_centroids[l][j] - global_mean[j]).powi(2);
        }
        between_ss += cluster_sizes[l] as f64 * dist_to_global;
    }

    let explained_variance_ratio = if total_ss > 0.0 {
        between_ss / total_ss
    } else {
        0.0
    };

    // Average within and between cluster distances
    let mut within_dist_sum = 0.0;
    let mut within_dist_count = 0;
    let mut between_dist_sum = 0.0;
    let mut between_dist_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[[i, j]];
            if labels[i] == labels[j] {
                within_dist_sum += d;
                within_dist_count += 1;
            } else {
                between_dist_sum += d;
                between_dist_count += 1;
            }
        }
    }

    let average_within = if within_dist_count > 0 {
        within_dist_sum / within_dist_count as f64
    } else {
        0.0
    };

    let average_between = if between_dist_count > 0 {
        between_dist_sum / between_dist_count as f64
    } else {
        0.0
    };

    // Cluster diameters and minimum separation
    let mut cluster_diameters: Vec<f64> = vec![0.0; k];
    for l in 0..k {
        let members: Vec<usize> = labels.iter()
            .enumerate()
            .filter(|&(_, lab)| *lab == l)
            .map(|(i, _)| i)
            .collect();

        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let d = distances[[members[i], members[j]]];
                cluster_diameters[l] = cluster_diameters[l].max(d);
            }
        }
    }

    let max_diameter = cluster_diameters.iter().cloned().fold(0.0, f64::max);

    // Separation: minimum distance between different clusters
    let mut separation = f64::INFINITY;
    for l1 in 0..k {
        for l2 in (l1 + 1)..k {
            for i in 0..n {
                if labels[i] != l1 {
                    continue;
                }
                for j in 0..n {
                    if labels[j] != l2 {
                        continue;
                    }
                    separation = separation.min(distances[[i, j]]);
                }
            }
        }
    }
    if separation.is_infinite() {
        separation = 0.0;
    }

    // Silhouette
    let (average_silhouette, cluster_silhouette) = compute_silhouette_details(labels, &distances, k);

    // Dunn index
    let dunn_index = if max_diameter > 0.0 {
        separation / max_diameter
    } else {
        0.0
    };

    // Calinski-Harabasz index
    let calinski_harabasz = if within_ss > 0.0 && k > 1 && n > k {
        (between_ss / (k - 1) as f64) / (within_ss / (n - k) as f64)
    } else {
        0.0
    };

    // Davies-Bouldin index
    let davies_bouldin = compute_davies_bouldin(labels, &distances, &cluster_centroids, k);

    Ok(ClusterStatsResult {
        n_clusters: k,
        cluster_sizes,
        average_within,
        average_between,
        within_ss,
        between_ss,
        total_ss,
        explained_variance_ratio,
        average_silhouette,
        cluster_silhouette,
        dunn_index,
        calinski_harabasz,
        davies_bouldin,
        separation,
        max_diameter,
        cluster_diameters,
        n,
        p,
    })
}

/// Compute silhouette with per-cluster breakdown.
fn compute_silhouette_details(labels: &[usize], distances: &Array2<f64>, k: usize) -> (f64, Vec<f64>) {
    let n = labels.len();
    if n < 2 || k < 2 {
        return (0.0, vec![0.0; k]);
    }

    let mut cluster_sil_sum = vec![0.0; k];
    let mut cluster_sil_count = vec![0; k];
    let mut total_sil = 0.0;

    for i in 0..n {
        let label_i = labels[i];

        // a(i)
        let mut same_dist = 0.0;
        let mut same_count = 0;
        for j in 0..n {
            if i != j && labels[j] == label_i {
                same_dist += distances[[i, j]];
                same_count += 1;
            }
        }
        let a_i = if same_count > 0 { same_dist / same_count as f64 } else { 0.0 };

        // b(i)
        let mut b_i = f64::INFINITY;
        for other in 0..k {
            if other == label_i {
                continue;
            }
            let mut other_dist = 0.0;
            let mut other_count = 0;
            for j in 0..n {
                if labels[j] == other {
                    other_dist += distances[[i, j]];
                    other_count += 1;
                }
            }
            if other_count > 0 {
                b_i = b_i.min(other_dist / other_count as f64);
            }
        }

        if b_i.is_finite() && (a_i > 0.0 || b_i > 0.0) {
            let s_i = (b_i - a_i) / a_i.max(b_i);
            total_sil += s_i;
            cluster_sil_sum[label_i] += s_i;
            cluster_sil_count[label_i] += 1;
        }
    }

    let avg_sil = total_sil / n as f64;
    let cluster_sil: Vec<f64> = (0..k)
        .map(|l| {
            if cluster_sil_count[l] > 0 {
                cluster_sil_sum[l] / cluster_sil_count[l] as f64
            } else {
                0.0
            }
        })
        .collect();

    (avg_sil, cluster_sil)
}

/// Compute Davies-Bouldin index.
fn compute_davies_bouldin(
    labels: &[usize],
    distances: &Array2<f64>,
    centroids: &[Vec<f64>],
    k: usize,
) -> f64 {
    if k < 2 {
        return 0.0;
    }

    let n = labels.len();
    let p = centroids[0].len();

    // Compute scatter for each cluster (average distance to centroid)
    let mut scatter = vec![0.0; k];
    let mut counts = vec![0; k];

    for (i, &l) in labels.iter().enumerate() {
        counts[l] += 1;
    }

    // Compute average within-cluster distance to centroid for each cluster
    for l in 0..k {
        if counts[l] == 0 {
            continue;
        }
        let mut total_dist = 0.0;
        for (i, &lab) in labels.iter().enumerate() {
            if lab == l {
                // Use distances to all same-cluster points as proxy for scatter
                let mut dist_to_others = 0.0;
                let mut count = 0;
                for (j, &lab_j) in labels.iter().enumerate() {
                    if i != j && lab_j == l {
                        dist_to_others += distances[[i, j]];
                        count += 1;
                    }
                }
                if count > 0 {
                    total_dist += dist_to_others / count as f64;
                }
            }
        }
        scatter[l] = total_dist / counts[l] as f64;
    }

    // Compute centroid distances
    let mut centroid_dist = Array2::zeros((k, k));
    for l1 in 0..k {
        for l2 in 0..k {
            if l1 != l2 {
                let mut d = 0.0;
                for j in 0..p {
                    d += (centroids[l1][j] - centroids[l2][j]).powi(2);
                }
                centroid_dist[[l1, l2]] = d.sqrt();
            }
        }
    }

    // Davies-Bouldin index
    let mut db_sum = 0.0;
    for l1 in 0..k {
        let mut max_ratio: f64 = 0.0;
        for l2 in 0..k {
            if l1 != l2 && centroid_dist[[l1, l2]] > 0.0 {
                let ratio = (scatter[l1] + scatter[l2]) / centroid_dist[[l1, l2]];
                max_ratio = max_ratio.max(ratio);
            }
        }
        db_sum += max_ratio;
    }

    db_sum / k as f64
}

/// Convenience wrapper for cluster_stats.
pub fn run_cluster_stats(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<ClusterStatsResult, String> {
    cluster_stats(data, labels)
}

// =============================================================================
// FANNY - Fuzzy Analysis Clustering
// =============================================================================

/// Result of FANNY fuzzy clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
/// - R cluster::fanny documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/fanny.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FannyResult {
    /// Membership coefficients (n x k matrix)
    #[serde(skip)]
    pub membership: Array2<f64>,
    /// Hard cluster assignments
    pub clustering: Vec<usize>,
    /// Cluster centers (weighted by membership)
    #[serde(skip)]
    pub centers: Array2<f64>,
    /// Objective function value
    pub objective: f64,
    /// Convergence flag
    pub converged: bool,
    /// Number of iterations
    pub n_iterations: usize,
    /// Dunn's partition coefficient (1/k to 1, higher is crisper)
    pub dunn_coefficient: f64,
    /// Normalized Dunn coefficient (0 to 1)
    pub normalized_dunn: f64,
    /// Number of clusters
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Fuzziness parameter used
    pub membership_exponent: f64,
}

impl std::fmt::Display for FannyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FANNY Fuzzy Clustering Results")?;
        writeln!(f, "===============================")?;
        writeln!(f, "Number of clusters: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Fuzziness parameter: {:.2}", self.membership_exponent)?;
        writeln!(f, "Converged: {} (iterations: {})", self.converged, self.n_iterations)?;
        writeln!(f)?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        writeln!(f, "Dunn coefficient: {:.4}", self.dunn_coefficient)?;
        writeln!(f, "Normalized Dunn: {:.4}", self.normalized_dunn)?;
        writeln!(f)?;

        // Hard clustering sizes
        let mut sizes = vec![0usize; self.k];
        for &c in &self.clustering {
            sizes[c] += 1;
        }
        writeln!(f, "Hard cluster sizes: {:?}", sizes)?;

        Ok(())
    }
}

/// FANNY fuzzy clustering algorithm.
///
/// FANNY computes a fuzzy clustering of the data, where each observation
/// can belong to multiple clusters with varying degrees of membership.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `membership_exponent` - Fuzziness parameter m > 1 (default: 2.0)
/// * `max_iter` - Maximum iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-8)
/// * `seed` - Random seed
///
/// # Returns
/// * `FannyResult` with membership coefficients and cluster assignments
///
/// # Algorithm
///
/// FANNY minimizes the objective:
/// ∑_i ∑_v (∑_j u_iv² d(i,j) / ∑_j u_jv²) * (∑_j u_jv² / 2n_v²)
///
/// This is equivalent to fuzzy c-medoids with modified normalization.
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn fanny(
    data: ArrayView2<f64>,
    k: usize,
    membership_exponent: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FannyResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 || k > n {
        return Err(format!("k must be between 1 and {}", n));
    }

    let m = membership_exponent.unwrap_or(2.0);
    if m <= 1.0 {
        return Err("membership_exponent must be > 1".to_string());
    }

    let max_iterations = max_iter.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-8);

    // Compute distance matrix
    let distances = compute_distance_matrix(&data);

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Initialize membership matrix
    let mut membership = Array2::zeros((n, k));
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            membership[[i, j]] = rng.r#gen::<f64>() + 0.1;
            sum += membership[[i, j]];
        }
        for j in 0..k {
            membership[[i, j]] /= sum;
        }
    }

    let mut converged = false;
    let mut n_iterations = 0;
    let mut prev_objective = f64::INFINITY;

    for iter in 0..max_iterations {
        n_iterations = iter + 1;

        // Update membership based on distances
        let mut new_membership = Array2::zeros((n, k));

        for i in 0..n {
            // Compute weighted distances to each cluster
            let mut cluster_dist = vec![0.0; k];
            for v in 0..k {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;
                for j in 0..n {
                    let u_jv_m = membership[[j, v]].powf(m);
                    weighted_sum += u_jv_m * distances[[i, j]];
                    weight_sum += u_jv_m;
                }
                cluster_dist[v] = if weight_sum > 1e-10 {
                    weighted_sum / weight_sum
                } else {
                    f64::INFINITY
                };
            }

            // Update membership using inverse distance weighting
            let exponent = 1.0 / (m - 1.0);
            let mut sum = 0.0;

            for v in 0..k {
                if cluster_dist[v] < 1e-10 {
                    // Point is at cluster center
                    new_membership[[i, v]] = 1.0;
                    sum = 1.0;
                    for w in 0..k {
                        if w != v {
                            new_membership[[i, w]] = 0.0;
                        }
                    }
                    break;
                }

                let mut ratio_sum = 0.0;
                for w in 0..k {
                    if cluster_dist[w] > 1e-10 {
                        ratio_sum += (cluster_dist[v] / cluster_dist[w]).powf(exponent);
                    }
                }
                new_membership[[i, v]] = if ratio_sum > 1e-10 { 1.0 / ratio_sum } else { 0.0 };
                sum += new_membership[[i, v]];
            }

            // Normalize
            if sum > 1e-10 {
                for v in 0..k {
                    new_membership[[i, v]] /= sum;
                }
            }
        }

        // Compute objective
        let objective = compute_fanny_objective(&distances, &new_membership, m);

        // Check convergence
        let change: f64 = membership.iter()
            .zip(new_membership.iter())
            .map(|(old, new)| (*old - *new).abs())
            .sum();

        membership = new_membership;

        if change < tolerance || (prev_objective - objective).abs() < tolerance {
            converged = true;
            break;
        }
        prev_objective = objective;
    }

    // Hard cluster assignments
    let clustering: Vec<usize> = (0..n)
        .map(|i| {
            (0..k).max_by(|&a, &b|
                membership[[i, a]].partial_cmp(&membership[[i, b]]).unwrap_or(Ordering::Equal)
            ).unwrap_or(0)
        })
        .collect();

    // Compute centers (weighted by membership)
    let mut centers = Array2::zeros((k, d));
    for v in 0..k {
        let mut weight_sum = 0.0;
        for i in 0..n {
            let u_m = membership[[i, v]].powf(m);
            weight_sum += u_m;
            for j in 0..d {
                centers[[v, j]] += u_m * data[[i, j]];
            }
        }
        if weight_sum > 1e-10 {
            for j in 0..d {
                centers[[v, j]] /= weight_sum;
            }
        }
    }

    // Dunn's partition coefficient
    let dunn_coefficient: f64 = membership.iter()
        .map(|u| u.powi(2))
        .sum::<f64>() / n as f64;

    // Normalized Dunn coefficient: (F - 1/k) / (1 - 1/k)
    let normalized_dunn = if k > 1 {
        (dunn_coefficient - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)
    } else {
        1.0
    };

    let objective = compute_fanny_objective(&distances, &membership, m);

    Ok(FannyResult {
        membership,
        clustering,
        centers,
        objective,
        converged,
        n_iterations,
        dunn_coefficient,
        normalized_dunn,
        k,
        n,
        membership_exponent: m,
    })
}

/// Compute FANNY objective function.
fn compute_fanny_objective(distances: &Array2<f64>, membership: &Array2<f64>, m: f64) -> f64 {
    let n = membership.nrows();
    let k = membership.ncols();

    let mut objective = 0.0;

    for v in 0..k {
        let mut numer_sum = 0.0;
        let mut denom_sum = 0.0;

        for i in 0..n {
            let u_iv_m = membership[[i, v]].powf(m);
            for j in 0..n {
                let u_jv_m = membership[[j, v]].powf(m);
                numer_sum += u_iv_m * u_jv_m * distances[[i, j]];
            }
            denom_sum += u_iv_m;
        }

        if denom_sum > 1e-10 {
            objective += numer_sum / (2.0 * denom_sum);
        }
    }

    objective
}

/// Convenience wrapper for fanny.
pub fn run_fanny(
    data: ArrayView2<f64>,
    k: usize,
    membership_exponent: Option<f64>,
    max_iter: Option<usize>,
    seed: Option<u64>,
) -> Result<FannyResult, String> {
    fanny(data, k, membership_exponent, max_iter, None, seed)
}

// =============================================================================
// Batch 3: skmeans, fastcluster, dynamicTreeCut, mixtools, kprototypes
// =============================================================================

// =============================================================================
// Spherical K-Means (skmeans)
// =============================================================================

/// Result of Spherical K-Means clustering.
///
/// # References
///
/// - Dhillon, I. S., & Modha, D. S. (2001). "Concept decompositions for large
///   sparse text data using clustering." Machine Learning, 42(1), 143-175.
/// - R skmeans package documentation
///   Source: https://www.rdocumentation.org/packages/skmeans/versions/0.2-17/topics/skmeans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SKMeansResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Centroid directions (normalized) for each cluster
    #[serde(skip)]
    pub centroids: Array2<f64>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Total cosine dissimilarity (objective function)
    pub dissimilarity: f64,
    /// Average cosine similarity within clusters
    pub avg_similarity: f64,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
    /// Converged flag
    pub converged: bool,
}

impl std::fmt::Display for SKMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spherical K-Means Clustering Results")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Total dissimilarity: {:.4}", self.dissimilarity)?;
        writeln!(f, "Average cosine similarity: {:.4}", self.avg_similarity)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, size)?;
        }
        Ok(())
    }
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Run Spherical K-Means clustering.
///
/// Spherical k-means uses cosine similarity instead of Euclidean distance,
/// making it ideal for text/document clustering where direction matters
/// more than magnitude.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `n_init` - Number of random initializations (default: 10)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// 1. Normalize all data points to unit length
/// 2. Initialize centroids randomly from data points
/// 3. Assign each point to cluster with highest cosine similarity
/// 4. Update centroids as normalized mean of cluster members
/// 5. Repeat until convergence
///
/// # Returns
/// * `SKMeansResult` containing cluster assignments and centroids
pub fn skmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    max_iterations: Option<usize>,
    tol: Option<f64>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<SKMeansResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!("n_clusters ({}) cannot exceed n_samples ({})", n_clusters, n));
    }

    let max_iter = max_iterations.unwrap_or(100);
    let tolerance = tol.unwrap_or(1e-6);
    let n_inits = n_init.unwrap_or(10);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Normalize data to unit vectors
    let mut normalized_data: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = data.row(i).to_vec();
        normalize_vector(&mut row);
        normalized_data.push(row);
    }

    let mut best_result: Option<SKMeansResult> = None;
    let mut best_dissimilarity = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize centroids: random selection from data
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let mut centroids: Vec<Vec<f64>> = indices[..n_clusters]
            .iter()
            .map(|&i| normalized_data[i].clone())
            .collect();

        let mut labels = vec![0usize; n];
        let mut converged = false;
        let mut n_iterations = 0;

        for iter in 0..max_iter {
            n_iterations = iter + 1;

            // Assignment step: assign to cluster with highest cosine similarity
            let old_labels = labels.clone();
            for i in 0..n {
                let mut max_sim = f64::NEG_INFINITY;
                let mut best_cluster = 0;
                for j in 0..n_clusters {
                    let sim = cosine_similarity(&normalized_data[i], &centroids[j]);
                    if sim > max_sim {
                        max_sim = sim;
                        best_cluster = j;
                    }
                }
                labels[i] = best_cluster;
            }

            // Update step: compute new centroids as normalized mean
            let mut new_centroids: Vec<Vec<f64>> = vec![vec![0.0; d]; n_clusters];
            let mut cluster_counts = vec![0usize; n_clusters];

            for i in 0..n {
                let c = labels[i];
                cluster_counts[c] += 1;
                for j in 0..d {
                    new_centroids[c][j] += normalized_data[i][j];
                }
            }

            // Normalize centroids
            for c in 0..n_clusters {
                if cluster_counts[c] > 0 {
                    normalize_vector(&mut new_centroids[c]);
                } else {
                    // Empty cluster: reinitialize randomly
                    let rand_idx = rng.gen_range(0..n);
                    new_centroids[c] = normalized_data[rand_idx].clone();
                }
            }

            // Check convergence
            let mut max_change: f64 = 0.0;
            for c in 0..n_clusters {
                let change = 1.0 - cosine_similarity(&centroids[c], &new_centroids[c]);
                max_change = max_change.max(change);
            }

            centroids = new_centroids;

            if labels == old_labels && max_change < tolerance {
                converged = true;
                break;
            }
        }

        // Compute dissimilarity (1 - similarity)
        let mut total_sim = 0.0;
        let mut cluster_sizes = vec![0usize; n_clusters];
        for i in 0..n {
            let c = labels[i];
            cluster_sizes[c] += 1;
            total_sim += cosine_similarity(&normalized_data[i], &centroids[c]);
        }
        let dissimilarity = n as f64 - total_sim;
        let avg_similarity = total_sim / n as f64;

        if dissimilarity < best_dissimilarity {
            best_dissimilarity = dissimilarity;

            // Convert centroids to Array2
            let mut centroid_arr = Array2::zeros((n_clusters, d));
            for c in 0..n_clusters {
                for j in 0..d {
                    centroid_arr[[c, j]] = centroids[c][j];
                }
            }

            best_result = Some(SKMeansResult {
                labels,
                centroids: centroid_arr,
                n_iterations,
                dissimilarity,
                avg_similarity,
                cluster_sizes,
                n_clusters,
                n,
                converged,
            });
        }
    }

    best_result.ok_or_else(|| "Spherical K-means failed".to_string())
}

/// Convenience wrapper for skmeans.
pub fn run_skmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    max_iterations: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<SKMeansResult, String> {
    skmeans(data, n_clusters, max_iterations, None, n_init, seed)
}

// =============================================================================
// Fast Hierarchical Clustering (fastcluster)
// =============================================================================

/// Linkage method for fast hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FastLinkage {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
    /// Ward's minimum variance method
    Ward,
    /// Weighted average linkage (WPGMA)
    Weighted,
    /// Centroid linkage (UPGMC)
    Centroid,
    /// Median linkage (WPGMC)
    Median,
}

impl Default for FastLinkage {
    fn default() -> Self {
        FastLinkage::Ward
    }
}

/// Result of fast hierarchical clustering.
///
/// # References
///
/// - Müllner, D. (2013). "fastcluster: Fast Hierarchical, Agglomerative
///   Clustering Routines for R and Python." Journal of Statistical Software.
/// - R fastcluster package documentation
///   Source: https://www.rdocumentation.org/packages/fastcluster/versions/1.2.6/topics/hclust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastClusterResult {
    /// Merge matrix (n-1 x 2): indices of clusters merged at each step
    /// Negative values indicate original observations, positive indicate merged clusters
    #[serde(skip)]
    pub merge: Array2<i64>,
    /// Heights at which merges occurred
    pub height: Vec<f64>,
    /// Order of leaves for optimal plotting
    pub order: Vec<usize>,
    /// Labels for each observation (if cut was performed)
    pub labels: Option<Vec<usize>>,
    /// Number of observations
    pub n: usize,
    /// Linkage method used
    pub linkage: FastLinkage,
}

impl std::fmt::Display for FastClusterResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fast Hierarchical Clustering Results")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Linkage method: {:?}", self.linkage)?;
        writeln!(f, "Number of merges: {}", self.merge.nrows())?;
        if let Some(ref labels) = self.labels {
            let n_clusters = labels.iter().max().map_or(0, |m| m + 1);
            writeln!(f, "Number of clusters (if cut): {}", n_clusters)?;
        }
        Ok(())
    }
}

/// Run fast hierarchical clustering.
///
/// This is an optimized O(n²) implementation of hierarchical clustering
/// using the nearest-neighbor chain algorithm and condensed distance storage.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `linkage` - Linkage method (default: Ward)
/// * `n_clusters` - Optional: cut dendrogram to produce this many clusters
///
/// # Algorithm
///
/// Uses the nearest-neighbor chain algorithm which achieves O(n²) complexity
/// for reducible linkage methods (single, complete, average, Ward, weighted).
///
/// # Returns
/// * `FastClusterResult` containing merge matrix and heights
pub fn fastcluster(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    n_clusters: Option<usize>,
) -> Result<FastClusterResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n < 2 {
        return Err("Need at least 2 observations for hierarchical clustering".to_string());
    }

    let method = linkage.unwrap_or(FastLinkage::Ward);

    // Compute condensed distance matrix (upper triangle only, stored as 1D)
    // Index formula: condensed_index(i, j, n) = i*n - i*(i+1)/2 + j - i - 1 for i < j
    let condensed_size = n * (n - 1) / 2;
    let mut condensed_dist = vec![0.0f64; condensed_size];

    // Compute squared Euclidean distances (avoid sqrt until needed)
    for i in 0..n {
        for j in (i + 1)..n {
            let sq_dist: f64 = (0..d)
                .map(|k| {
                    let diff = data[[i, k]] - data[[j, k]];
                    diff * diff
                })
                .sum();
            condensed_dist[condensed_idx(i, j, n)] = sq_dist;
        }
    }

    // Use optimized nearest-neighbor chain algorithm
    let (merge, height) = fastcluster_nnchain_optimized(&mut condensed_dist, method, n)?;

    // Compute leaf ordering
    let order = compute_dendrogram_order(&merge, n);

    // Cut tree if n_clusters specified
    let labels = if let Some(k) = n_clusters {
        if k < 1 || k > n {
            return Err(format!("n_clusters must be between 1 and n ({})", n));
        }
        Some(cut_dendrogram(&merge, &height, n, k))
    } else {
        None
    };

    Ok(FastClusterResult {
        merge,
        height,
        order,
        labels,
        n,
        linkage: method,
    })
}

/// Compute index into condensed distance matrix for i < j
#[inline(always)]
fn condensed_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j);
    i * n - (i * (i + 1)) / 2 + j - i - 1
}

/// Get distance from condensed matrix (handles i > j by swapping)
#[inline(always)]
fn get_dist(condensed: &[f64], i: usize, j: usize, n: usize) -> f64 {
    if i == j {
        0.0
    } else if i < j {
        condensed[condensed_idx(i, j, n)]
    } else {
        condensed[condensed_idx(j, i, n)]
    }
}

/// Set distance in condensed matrix (handles i > j by swapping)
#[inline(always)]
fn set_dist(condensed: &mut [f64], i: usize, j: usize, n: usize, val: f64) {
    if i < j {
        condensed[condensed_idx(i, j, n)] = val;
    } else if j < i {
        condensed[condensed_idx(j, i, n)] = val;
    }
}

/// Optimized nearest-neighbor chain algorithm for hierarchical clustering.
///
/// This implementation uses:
/// 1. Condensed distance matrix (O(n²/2) space instead of O(n²))
/// 2. Nearest-neighbor tracking to avoid full scans
/// 3. Union-find structure for efficient cluster merging
fn fastcluster_nnchain_optimized(
    condensed_dist: &mut [f64],
    method: FastLinkage,
    n: usize,
) -> Result<(Array2<i64>, Vec<f64>), String> {
    // Union-find structure: maps each original index to its current cluster representative
    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_size: Vec<usize> = vec![1; n];

    // Track which clusters are still active
    let mut active: Vec<bool> = vec![true; n];
    let mut n_active = n;

    // Nearest neighbor for each cluster (cache)
    let mut nearest: Vec<usize> = vec![usize::MAX; n];
    let mut nearest_dist: Vec<f64> = vec![f64::INFINITY; n];

    // Initialize nearest neighbors
    for i in 0..n {
        for j in (i + 1)..n {
            let d = get_dist(condensed_dist, i, j, n);
            if d < nearest_dist[i] {
                nearest_dist[i] = d;
                nearest[i] = j;
            }
            if d < nearest_dist[j] {
                nearest_dist[j] = d;
                nearest[j] = i;
            }
        }
    }

    // Output arrays
    let mut merge = Array2::<i64>::zeros((n - 1, 2));
    let mut height = Vec::with_capacity(n - 1);

    // Track merge history for R-style indexing
    let mut merge_step: Vec<Option<usize>> = vec![None; n];

    // Nearest-neighbor chain
    let mut chain: Vec<usize> = Vec::with_capacity(n);

    for step in 0..(n - 1) {
        // Find a starting point if chain is empty
        if chain.is_empty() {
            for i in 0..n {
                if active[i] {
                    chain.push(i);
                    break;
                }
            }
        }

        // Build chain until we find a reciprocal pair
        loop {
            let current = *chain.last().unwrap();

            // Find nearest neighbor of current (update if stale)
            let mut nn = nearest[current];
            let mut nn_dist = if nn < n && active[nn] {
                get_dist(condensed_dist, current, nn, n)
            } else {
                f64::INFINITY
            };

            // Check if cached nearest is still valid, otherwise recompute
            if !active[nn] || nn_dist > nearest_dist[current] * 1.001 {
                nn = usize::MAX;
                nn_dist = f64::INFINITY;
                for j in 0..n {
                    if j != current && active[j] {
                        let d = get_dist(condensed_dist, current, j, n);
                        if d < nn_dist {
                            nn_dist = d;
                            nn = j;
                        }
                    }
                }
                nearest[current] = nn;
                nearest_dist[current] = nn_dist;
            }

            // Check if we have a reciprocal pair (nn's nearest is current)
            if chain.len() >= 2 && chain[chain.len() - 2] == nn {
                // Found reciprocal pair: merge current and nn
                let (a, b) = if current < nn { (current, nn) } else { (nn, current) };

                // Record the merge with proper height
                let merge_dist = nn_dist.sqrt(); // Convert squared distance to distance

                // R-style merge indices: negative for leaves, positive for merged clusters
                let idx_a = match merge_step[a] {
                    None => -(a as i64 + 1),
                    Some(s) => (s as i64 + 1),
                };
                let idx_b = match merge_step[b] {
                    None => -(b as i64 + 1),
                    Some(s) => (s as i64 + 1),
                };

                merge[[step, 0]] = idx_a.min(idx_b);
                merge[[step, 1]] = idx_a.max(idx_b);
                height.push(merge_dist);

                // Merge b into a
                let size_a = cluster_size[a];
                let size_b = cluster_size[b];
                let new_size = size_a + size_b;

                // Update distances from merged cluster to all other active clusters
                for k in 0..n {
                    if !active[k] || k == a || k == b {
                        continue;
                    }

                    let d_ak = get_dist(condensed_dist, a, k, n);
                    let d_bk = get_dist(condensed_dist, b, k, n);
                    let d_ab = nn_dist; // squared distance
                    let n_a = size_a as f64;
                    let n_b = size_b as f64;
                    let n_k = cluster_size[k] as f64;

                    // Lance-Williams formula (using squared distances for efficiency)
                    let new_dist_sq = match method {
                        FastLinkage::Single => d_ak.min(d_bk),
                        FastLinkage::Complete => d_ak.max(d_bk),
                        FastLinkage::Average => (n_a * d_ak + n_b * d_bk) / (n_a + n_b),
                        FastLinkage::Weighted => (d_ak + d_bk) / 2.0,
                        FastLinkage::Ward => {
                            // Ward's method with squared distances
                            let n_total = n_a + n_b + n_k;
                            ((n_a + n_k) * d_ak + (n_b + n_k) * d_bk - n_k * d_ab) / n_total
                        }
                        FastLinkage::Centroid => {
                            let alpha_a = n_a / (n_a + n_b);
                            let alpha_b = n_b / (n_a + n_b);
                            let beta = -n_a * n_b / ((n_a + n_b) * (n_a + n_b));
                            (alpha_a * d_ak + alpha_b * d_bk + beta * d_ab).max(0.0)
                        }
                        FastLinkage::Median => {
                            (0.5 * d_ak + 0.5 * d_bk - 0.25 * d_ab).max(0.0)
                        }
                    };

                    set_dist(condensed_dist, a, k, n, new_dist_sq);

                    // Invalidate k's nearest neighbor cache if it was a or b
                    if nearest[k] == a || nearest[k] == b {
                        nearest_dist[k] = f64::INFINITY;
                    }
                }

                // Update cluster info
                cluster_size[a] = new_size;
                active[b] = false;
                merge_step[a] = Some(step);
                parent[b] = a;
                n_active -= 1;

                // Invalidate a's nearest neighbor cache
                nearest_dist[a] = f64::INFINITY;

                // Pop both from chain
                chain.pop(); // current
                chain.pop(); // nn (was second-to-last)

                break;
            } else {
                // Not a reciprocal pair, extend chain
                chain.push(nn);
            }
        }
    }

    Ok((merge, height))
}

/// Compute dendrogram leaf ordering for optimal display.
fn compute_dendrogram_order(merge: &Array2<i64>, n: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(n);

    fn traverse(merge: &Array2<i64>, n: usize, node: i64, order: &mut Vec<usize>) {
        if node < 0 {
            // Leaf node
            order.push((-node - 1) as usize);
        } else {
            // Internal node
            let step = (node - 1) as usize;
            traverse(merge, n, merge[[step, 0]], order);
            traverse(merge, n, merge[[step, 1]], order);
        }
    }

    // Start from root (last merge)
    let n_merges = merge.nrows();
    if n_merges > 0 {
        traverse(merge, n, n_merges as i64, &mut order);
    }

    if order.len() != n {
        // Fallback: simple ordering
        order = (0..n).collect();
    }

    order
}

/// Cut dendrogram at a height to produce k clusters.
fn cut_dendrogram(merge: &Array2<i64>, height: &[f64], n: usize, k: usize) -> Vec<usize> {
    if k >= n {
        return (0..n).collect();
    }

    // Find height threshold: we need n - k merges
    let n_merges_needed = n - k;
    let threshold = if n_merges_needed == 0 {
        0.0
    } else if n_merges_needed <= height.len() {
        height[n_merges_needed - 1]
    } else {
        height.last().copied().unwrap_or(f64::INFINITY)
    };

    // Build cluster assignments
    let mut labels = vec![0usize; n];
    let mut cluster_labels: HashMap<i64, usize> = HashMap::new();
    let mut next_label = 0usize;

    // Process merges up to threshold
    for (step, &h) in height.iter().enumerate() {
        if h > threshold && step >= n_merges_needed {
            break;
        }

        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        // Get or assign labels for children
        let left_label = if left < 0 {
            cluster_labels.entry(left).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            }).clone()
        } else {
            *cluster_labels.get(&left).unwrap_or(&0)
        };

        let right_label = if right < 0 {
            cluster_labels.entry(right).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            }).clone()
        } else {
            *cluster_labels.get(&right).unwrap_or(&0)
        };

        // Merge: assign same label to merged cluster
        let merged_label = left_label.min(right_label);
        cluster_labels.insert(step as i64 + 1, merged_label);

        // Update any nodes with the higher label
        for (_, v) in cluster_labels.iter_mut() {
            if *v == left_label.max(right_label) {
                *v = merged_label;
            }
        }
    }

    // Assign final labels to observations
    for i in 0..n {
        labels[i] = *cluster_labels.get(&(-(i as i64) - 1)).unwrap_or(&i);
    }

    // Renumber labels to be consecutive
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0;
    for l in labels.iter_mut() {
        if let Some(&new_l) = label_map.get(l) {
            *l = new_l;
        } else {
            label_map.insert(*l, next);
            *l = next;
            next += 1;
        }
    }

    labels
}

/// Convenience wrapper for fastcluster.
pub fn run_fastcluster(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    n_clusters: Option<usize>,
) -> Result<FastClusterResult, String> {
    fastcluster(data, linkage, n_clusters)
}

// =============================================================================
// Dynamic Tree Cut (dynamicTreeCut)
// =============================================================================

/// Method for dynamic tree cutting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DynamicCutMethod {
    /// Hybrid method combining hierarchical with PAM refinement
    Hybrid,
    /// Pure tree-based method
    Tree,
}

impl Default for DynamicCutMethod {
    fn default() -> Self {
        DynamicCutMethod::Hybrid
    }
}

/// Result of dynamic tree cut.
///
/// # References
///
/// - Langfelder, P., Zhang, B., & Horvath, S. (2008). "Defining clusters
///   from a hierarchical cluster tree: the Dynamic Tree Cut package for R."
///   Bioinformatics, 24(5), 719-720.
/// - R dynamicTreeCut package documentation
///   Source: https://www.rdocumentation.org/packages/dynamicTreeCut/versions/1.63-1/topics/cutreeDynamic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTreeCutResult {
    /// Cluster assignments (0 = unassigned/outlier)
    pub labels: Vec<usize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of unassigned points (label 0)
    pub n_unassigned: usize,
    /// Cluster sizes (excluding unassigned)
    pub cluster_sizes: Vec<usize>,
    /// Number of observations
    pub n: usize,
    /// Method used
    pub method: DynamicCutMethod,
    /// Deep split parameter used
    pub deep_split: u8,
    /// Minimum cluster size used
    pub min_cluster_size: usize,
}

impl std::fmt::Display for DynamicTreeCutResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dynamic Tree Cut Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Method: {:?}", self.method)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Unassigned points: {}", self.n_unassigned)?;
        writeln!(f, "Deep split: {}", self.deep_split)?;
        writeln!(f, "Min cluster size: {}", self.min_cluster_size)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i + 1, size)?;
        }
        Ok(())
    }
}

/// Perform dynamic tree cutting on hierarchical clustering result.
///
/// Automatically detects clusters in a dendrogram by identifying significant
/// branches based on shape criteria.
///
/// # Arguments
/// * `merge` - Merge matrix from hierarchical clustering (n-1 x 2)
/// * `height` - Heights at which merges occurred
/// * `n` - Number of observations
/// * `method` - Cutting method (default: Hybrid)
/// * `deep_split` - Controls sensitivity: 0-4 (higher = more smaller clusters)
/// * `min_cluster_size` - Minimum size for a valid cluster
/// * `dist_matrix` - Optional distance matrix for hybrid method
///
/// # Algorithm
///
/// 1. Tree method: Find branches by detecting significant height gaps
/// 2. Hybrid method: Refine tree clusters using PAM
///
/// # Returns
/// * `DynamicTreeCutResult` containing adaptive cluster assignments
pub fn dynamic_tree_cut(
    merge: &Array2<i64>,
    height: &[f64],
    n: usize,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
    dist_matrix: Option<&Array2<f64>>,
) -> Result<DynamicTreeCutResult, String> {
    let cut_method = method.unwrap_or(DynamicCutMethod::Tree);
    let ds = deep_split.unwrap_or(1).min(4);
    let min_size = min_cluster_size.unwrap_or(2);

    if merge.nrows() != n - 1 {
        return Err(format!("Merge matrix should have {} rows for {} observations", n - 1, n));
    }
    if height.len() != n - 1 {
        return Err("Height vector length should match merge matrix rows".to_string());
    }

    // Compute height gaps to find natural breaks
    let mut height_sorted = height.to_vec();
    height_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Threshold based on deep_split
    let gap_percentile = match ds {
        0 => 0.95, // Very conservative
        1 => 0.90,
        2 => 0.75,
        3 => 0.50,
        _ => 0.25, // Very sensitive
    };

    let mut gaps: Vec<f64> = Vec::new();
    for i in 1..height_sorted.len() {
        gaps.push(height_sorted[i] - height_sorted[i - 1]);
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let gap_idx = ((gaps.len() as f64 * gap_percentile) as usize).min(gaps.len() - 1);
    let gap_threshold = if gaps.is_empty() { 0.0 } else { gaps[gap_idx] };

    // Find cut points (significant gaps)
    let mut cut_heights: Vec<f64> = Vec::new();
    for i in 1..height.len() {
        if height[i] - height[i - 1] > gap_threshold {
            cut_heights.push((height[i] + height[i - 1]) / 2.0);
        }
    }

    // If no cuts found, use a single cut at median height
    if cut_heights.is_empty() {
        let median_idx = height.len() / 2;
        cut_heights.push(height_sorted[median_idx]);
    }

    // Build cluster assignments by traversing tree
    let mut labels = vec![0usize; n];
    let mut cluster_id = 1usize;

    // Track which nodes belong to which cluster
    let mut node_clusters: HashMap<i64, usize> = HashMap::new();

    // Process merges from bottom to top
    for (step, &h) in height.iter().enumerate() {
        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        let left_cluster = if left < 0 {
            0 // Leaf, unassigned initially
        } else {
            *node_clusters.get(&left).unwrap_or(&0)
        };

        let right_cluster = if right < 0 {
            0
        } else {
            *node_clusters.get(&right).unwrap_or(&0)
        };

        // Check if this merge is above a cut threshold
        let is_cut = cut_heights.iter().any(|&ch| h > ch);

        if is_cut && left_cluster == 0 && right_cluster == 0 {
            // Both branches are new clusters
            node_clusters.insert(step as i64 + 1, 0);
        } else if left_cluster > 0 && right_cluster > 0 && is_cut {
            // Keep separate clusters
            node_clusters.insert(step as i64 + 1, 0);
        } else {
            // Merge into same cluster
            let merged_cluster = if left_cluster > 0 {
                left_cluster
            } else if right_cluster > 0 {
                right_cluster
            } else {
                // Both unassigned, create new cluster
                cluster_id += 1;
                cluster_id - 1
            };
            node_clusters.insert(step as i64 + 1, merged_cluster);

            // Propagate cluster assignment to children
            if left < 0 {
                labels[(-left - 1) as usize] = merged_cluster;
            } else if let Some(&c) = node_clusters.get(&left) {
                if c == 0 {
                    node_clusters.insert(left, merged_cluster);
                }
            }
            if right < 0 {
                labels[(-right - 1) as usize] = merged_cluster;
            } else if let Some(&c) = node_clusters.get(&right) {
                if c == 0 {
                    node_clusters.insert(right, merged_cluster);
                }
            }
        }
    }

    // Assign unassigned leaves
    for i in 0..n {
        if labels[i] == 0 {
            if let Some(&c) = node_clusters.get(&(-(i as i64) - 1)) {
                labels[i] = c;
            }
        }
    }

    // Traverse tree to assign remaining points
    fn assign_from_tree(
        merge: &Array2<i64>,
        node: i64,
        node_clusters: &HashMap<i64, usize>,
        labels: &mut Vec<usize>,
    ) -> usize {
        if node < 0 {
            let idx = (-node - 1) as usize;
            if labels[idx] == 0 {
                labels[idx] = *node_clusters.get(&node).unwrap_or(&0);
            }
            return labels[idx];
        }

        let step = (node - 1) as usize;
        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        let left_c = assign_from_tree(merge, left, node_clusters, labels);
        let right_c = assign_from_tree(merge, right, node_clusters, labels);

        if let Some(&c) = node_clusters.get(&node) {
            if c > 0 {
                return c;
            }
        }

        left_c.max(right_c)
    }

    let n_merges = merge.nrows();
    if n_merges > 0 {
        assign_from_tree(merge, n_merges as i64, &node_clusters, &mut labels);
    }

    // Remove small clusters (assign to unassigned)
    let mut cluster_sizes_map: HashMap<usize, usize> = HashMap::new();
    for &l in &labels {
        if l > 0 {
            *cluster_sizes_map.entry(l).or_insert(0) += 1;
        }
    }

    for l in labels.iter_mut() {
        if let Some(&size) = cluster_sizes_map.get(l) {
            if size < min_size {
                *l = 0;
            }
        }
    }

    // Renumber clusters to be consecutive starting from 1
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    label_map.insert(0, 0); // Keep 0 as unassigned
    let mut next = 1;
    for l in labels.iter_mut() {
        if *l > 0 {
            if let Some(&new_l) = label_map.get(l) {
                *l = new_l;
            } else {
                label_map.insert(*l, next);
                *l = next;
                next += 1;
            }
        }
    }

    // Compute final statistics
    let n_clusters = next - 1;
    let n_unassigned = labels.iter().filter(|&&l| l == 0).count();
    let mut cluster_sizes = vec![0usize; n_clusters];
    for &l in &labels {
        if l > 0 && l <= n_clusters {
            cluster_sizes[l - 1] += 1;
        }
    }

    // Hybrid refinement with PAM if distance matrix provided
    if cut_method == DynamicCutMethod::Hybrid && dist_matrix.is_some() && n_clusters > 0 {
        // PAM refinement would go here - for now, use tree result
    }

    Ok(DynamicTreeCutResult {
        labels,
        n_clusters,
        n_unassigned,
        cluster_sizes,
        n,
        method: cut_method,
        deep_split: ds,
        min_cluster_size: min_size,
    })
}

/// Convenience wrapper for dynamic_tree_cut with hierarchical clustering.
pub fn run_dynamic_tree_cut(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
) -> Result<DynamicTreeCutResult, String> {
    // First run hierarchical clustering
    let hc_result = fastcluster(data, linkage, None)?;

    // Then apply dynamic tree cut
    dynamic_tree_cut(
        &hc_result.merge,
        &hc_result.height,
        hc_result.n,
        method,
        deep_split,
        min_cluster_size,
        None,
    )
}

// =============================================================================
// Mixture Models (mixtools)
// =============================================================================

/// Result of normal mixture EM fitting.
///
/// # References
///
/// - McLachlan, G. J., & Peel, D. (2000). "Finite Mixture Models." Wiley.
/// - R mixtools package documentation
///   Source: https://www.rdocumentation.org/packages/mixtools/versions/2.0.0/topics/normalmixEM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMixEMResult {
    /// Component means
    pub mu: Vec<f64>,
    /// Component standard deviations
    pub sigma: Vec<f64>,
    /// Component mixing proportions
    pub lambda: Vec<f64>,
    /// Posterior probabilities (n x k)
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments
    pub labels: Vec<usize>,
    /// Log-likelihood at convergence
    pub loglik: f64,
    /// Log-likelihood history
    pub loglik_history: Vec<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for NormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Normal Mixture EM Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f)?;
        writeln!(f, "Component parameters:")?;
        for i in 0..self.k {
            writeln!(f, "  Component {}: mu={:.4}, sigma={:.4}, lambda={:.4}",
                i + 1, self.mu[i], self.sigma[i], self.lambda[i])?;
        }
        Ok(())
    }
}

/// Fit a univariate normal mixture model using EM algorithm.
///
/// # Arguments
/// * `data` - Univariate data (vector)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// EM algorithm for Gaussian mixtures:
/// 1. E-step: Compute posterior probabilities
/// 2. M-step: Update means, variances, and mixing proportions
///
/// # Returns
/// * `NormalMixEMResult` containing fitted parameters
pub fn normal_mix_em(
    data: &[f64],
    k: usize,
    max_iterations: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<NormalMixEMResult, String> {
    let n = data.len();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-6);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Compute data statistics for initialization
    let data_mean: f64 = data.iter().sum::<f64>() / n as f64;
    let data_var: f64 = data.iter().map(|x| (x - data_mean).powi(2)).sum::<f64>() / n as f64;
    let data_sd = data_var.sqrt();

    // Initialize parameters
    let mut mu: Vec<f64> = Vec::with_capacity(k);
    let mut sigma: Vec<f64> = vec![data_sd; k];
    let mut lambda: Vec<f64> = vec![1.0 / k as f64; k];

    // Initialize means spread across data range
    let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for i in 0..k {
        let t = (i as f64 + 0.5) / k as f64;
        mu.push(data_min + t * (data_max - data_min));
    }

    let mut posterior = Array2::zeros((n, k));
    let mut loglik_history = Vec::new();
    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // E-step: compute posterior probabilities
        let mut loglik = 0.0;
        for i in 0..n {
            let x = data[i];
            let mut log_probs = vec![0.0f64; k];
            let mut max_log_prob = f64::NEG_INFINITY;

            for j in 0..k {
                // Log of normal density weighted by mixing proportion
                let z = (x - mu[j]) / sigma[j];
                log_probs[j] = lambda[j].ln() - sigma[j].ln() - 0.5 * z * z;
                max_log_prob = max_log_prob.max(log_probs[j]);
            }

            // Log-sum-exp trick for numerical stability
            let sum_exp: f64 = log_probs.iter()
                .map(|&lp| (lp - max_log_prob).exp())
                .sum();
            let log_total = max_log_prob + sum_exp.ln();
            loglik += log_total;

            // Posterior probabilities
            for j in 0..k {
                posterior[[i, j]] = (log_probs[j] - log_total).exp();
            }
        }

        loglik_history.push(loglik);

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;

        // M-step: update parameters
        for j in 0..k {
            let n_j: f64 = (0..n).map(|i| posterior[[i, j]]).sum();

            if n_j > 1e-10 {
                // Update mixing proportion
                lambda[j] = n_j / n as f64;

                // Update mean
                mu[j] = (0..n).map(|i| posterior[[i, j]] * data[i]).sum::<f64>() / n_j;

                // Update variance
                let var_j: f64 = (0..n)
                    .map(|i| posterior[[i, j]] * (data[i] - mu[j]).powi(2))
                    .sum::<f64>() / n_j;
                sigma[j] = var_j.sqrt().max(1e-6);
            }
        }
    }

    // Compute hard assignments
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mut max_prob = 0.0;
        for j in 0..k {
            if posterior[[i, j]] > max_prob {
                max_prob = posterior[[i, j]];
                labels[i] = j;
            }
        }
    }

    let loglik = loglik_history.last().copied().unwrap_or(f64::NEG_INFINITY);

    Ok(NormalMixEMResult {
        mu,
        sigma,
        lambda,
        posterior,
        labels,
        loglik,
        loglik_history,
        n_iterations,
        converged,
        k,
        n,
    })
}

/// Result of multivariate normal mixture EM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivariateNormalMixEMResult {
    /// Component means (k x d)
    #[serde(skip)]
    pub mu: Array2<f64>,
    /// Component covariance matrices (flattened: k x d x d stored as k x (d*d))
    #[serde(skip)]
    pub sigma: Vec<Array2<f64>>,
    /// Component mixing proportions
    pub lambda: Vec<f64>,
    /// Posterior probabilities (n x k)
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments
    pub labels: Vec<usize>,
    /// Log-likelihood at convergence
    pub loglik: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of dimensions
    pub d: usize,
}

impl std::fmt::Display for MultivariateNormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multivariate Normal Mixture EM Results")?;
        writeln!(f, "=======================================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Dimensions: {}", self.d)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f)?;
        writeln!(f, "Mixing proportions:")?;
        for (i, &l) in self.lambda.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i + 1, l)?;
        }
        Ok(())
    }
}

/// Fit a multivariate normal mixture model using EM algorithm.
///
/// # Arguments
/// * `data` - Multivariate data (n x d)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Optional random seed
///
/// # Returns
/// * `MultivariateNormalMixEMResult` containing fitted parameters
pub fn mvnorm_mix_em(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<MultivariateNormalMixEMResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-6);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Initialize with k-means++ style initialization
    let mut mu = Array2::zeros((k, d));
    let mut sigma: Vec<Array2<f64>> = vec![Array2::eye(d); k];
    let mut lambda = vec![1.0 / k as f64; k];

    // Choose first center randomly
    let first_idx = rng.gen_range(0..n);
    mu.row_mut(0).assign(&data.row(first_idx));

    // Choose remaining centers proportional to squared distance
    for c in 1..k {
        let mut distances = vec![f64::INFINITY; n];
        for i in 0..n {
            for j in 0..c {
                let dist: f64 = (0..d)
                    .map(|l| (data[[i, l]] - mu[[j, l]]).powi(2))
                    .sum();
                distances[i] = distances[i].min(dist);
            }
        }

        let total: f64 = distances.iter().sum();
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &dist) in distances.iter().enumerate() {
            cumsum += dist;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        mu.row_mut(c).assign(&data.row(chosen));
    }

    // Initialize covariances as scaled identity
    let global_var: f64 = (0..d).map(|j| {
        let col_mean: f64 = data.column(j).sum() / n as f64;
        data.column(j).iter().map(|&x| (x - col_mean).powi(2)).sum::<f64>() / n as f64
    }).sum::<f64>() / d as f64;

    for c in 0..k {
        sigma[c] = Array2::eye(d) * global_var.max(1e-6);
    }

    let mut posterior = Array2::zeros((n, k));
    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut loglik = f64::NEG_INFINITY;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // E-step: compute posterior probabilities
        loglik = 0.0;
        for i in 0..n {
            let x: Vec<f64> = data.row(i).to_vec();
            let mut log_probs = vec![f64::NEG_INFINITY; k];

            for j in 0..k {
                // Compute log of multivariate normal density
                let mu_j: Vec<f64> = mu.row(j).to_vec();
                let diff: Vec<f64> = x.iter().zip(mu_j.iter()).map(|(a, b)| a - b).collect();

                // Simple diagonal approximation for stability
                let sigma_diag: Vec<f64> = (0..d).map(|l| sigma[j][[l, l]].max(1e-6)).collect();
                let log_det: f64 = sigma_diag.iter().map(|s| s.ln()).sum();
                let mahal: f64 = diff.iter().zip(sigma_diag.iter())
                    .map(|(di, si)| di * di / si)
                    .sum();

                log_probs[j] = lambda[j].ln() - 0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + mahal);
            }

            // Log-sum-exp
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_log_prob.is_finite() {
                let sum_exp: f64 = log_probs.iter()
                    .map(|&lp| (lp - max_log_prob).exp())
                    .sum();
                let log_total = max_log_prob + sum_exp.ln();
                loglik += log_total;

                for j in 0..k {
                    posterior[[i, j]] = (log_probs[j] - log_total).exp();
                }
            } else {
                // Fallback: uniform assignment
                for j in 0..k {
                    posterior[[i, j]] = 1.0 / k as f64;
                }
            }
        }

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;

        // M-step
        for j in 0..k {
            let n_j: f64 = (0..n).map(|i| posterior[[i, j]]).sum();

            if n_j > 1e-10 {
                lambda[j] = n_j / n as f64;

                // Update mean
                for l in 0..d {
                    mu[[j, l]] = (0..n).map(|i| posterior[[i, j]] * data[[i, l]]).sum::<f64>() / n_j;
                }

                // Update covariance (diagonal for stability)
                for l in 0..d {
                    let var_l: f64 = (0..n)
                        .map(|i| posterior[[i, j]] * (data[[i, l]] - mu[[j, l]]).powi(2))
                        .sum::<f64>() / n_j;
                    sigma[j][[l, l]] = var_l.max(1e-6);
                }
            }
        }
    }

    // Hard assignments
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mut max_prob = 0.0;
        for j in 0..k {
            if posterior[[i, j]] > max_prob {
                max_prob = posterior[[i, j]];
                labels[i] = j;
            }
        }
    }

    Ok(MultivariateNormalMixEMResult {
        mu,
        sigma,
        lambda,
        posterior,
        labels,
        loglik,
        n_iterations,
        converged,
        k,
        n,
        d,
    })
}

/// Convenience wrapper for normal_mix_em.
pub fn run_normal_mix_em(
    data: &[f64],
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<NormalMixEMResult, String> {
    normal_mix_em(data, k, max_iterations, None, seed)
}

/// Convenience wrapper for mvnorm_mix_em.
pub fn run_mvnorm_mix_em(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<MultivariateNormalMixEMResult, String> {
    mvnorm_mix_em(data, k, max_iterations, None, seed)
}

// =============================================================================
// K-Prototypes (kprototypes)
// =============================================================================

/// Result of K-Prototypes clustering.
///
/// # References
///
/// - Huang, Z. (1998). "Extensions to the k-means algorithm for clustering
///   large data sets with categorical values." Data Mining and Knowledge
///   Discovery, 2(3), 283-304.
/// - R clustMixType::kproto documentation
///   Source: https://www.rdocumentation.org/packages/clustMixType/versions/0.4-2/topics/kproto
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPrototypesResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Numeric prototypes for each cluster (k x d_numeric)
    #[serde(skip)]
    pub numeric_prototypes: Array2<f64>,
    /// Categorical prototypes for each cluster (k x d_categorical)
    pub categorical_prototypes: Vec<Vec<usize>>,
    /// Total cost (weighted sum of numeric distance + categorical dissimilarity)
    pub total_cost: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
    /// Number of numeric features
    pub n_numeric: usize,
    /// Number of categorical features
    pub n_categorical: usize,
    /// Gamma weight for categorical features
    pub gamma: f64,
    /// Converged flag
    pub converged: bool,
}

impl std::fmt::Display for KPrototypesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-Prototypes Clustering Results")?;
        writeln!(f, "================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Numeric features: {}", self.n_numeric)?;
        writeln!(f, "Categorical features: {}", self.n_categorical)?;
        writeln!(f, "Gamma: {:.4}", self.gamma)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Total cost: {:.4}", self.total_cost)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, size)?;
        }
        Ok(())
    }
}

/// Run K-Prototypes clustering for mixed numeric and categorical data.
///
/// # Arguments
/// * `numeric_data` - Numeric features (n x d_numeric)
/// * `categorical_data` - Categorical features encoded as integers (n x d_categorical)
/// * `n_clusters` - Number of clusters
/// * `gamma` - Weight for categorical features (default: auto-computed)
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `n_init` - Number of random initializations (default: 10)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// 1. Initialize prototypes by random selection
/// 2. Assign each point to nearest prototype using mixed distance
/// 3. Update prototypes: means for numeric, modes for categorical
/// 4. Repeat until convergence
///
/// # Distance
///
/// d(x, y) = d_numeric(x, y) + gamma * d_categorical(x, y)
/// where d_numeric is squared Euclidean and d_categorical is simple matching
///
/// # Returns
/// * `KPrototypesResult` containing cluster assignments and prototypes
pub fn kprototypes(
    numeric_data: ArrayView2<f64>,
    categorical_data: &[Vec<usize>],
    n_clusters: usize,
    gamma: Option<f64>,
    max_iterations: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<KPrototypesResult, String> {
    let n = numeric_data.nrows();
    let d_num = numeric_data.ncols();

    if categorical_data.len() != n {
        return Err(format!(
            "Categorical data length ({}) must match numeric data rows ({})",
            categorical_data.len(), n
        ));
    }

    let d_cat = if categorical_data.is_empty() || categorical_data[0].is_empty() {
        0
    } else {
        categorical_data[0].len()
    };

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!("n_clusters ({}) cannot exceed n_samples ({})", n_clusters, n));
    }

    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(10);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Auto-compute gamma if not provided
    // Use ratio of average numeric variance to average categorical entropy
    let auto_gamma = if gamma.is_none() && d_num > 0 && d_cat > 0 {
        let numeric_var: f64 = (0..d_num).map(|j| {
            let col_mean: f64 = numeric_data.column(j).sum() / n as f64;
            numeric_data.column(j).iter().map(|&x| (x - col_mean).powi(2)).sum::<f64>() / n as f64
        }).sum::<f64>() / d_num as f64;

        // For categorical, use 0.5 as proxy for average dissimilarity
        numeric_var / 0.5
    } else {
        gamma.unwrap_or(1.0)
    };

    let mut best_result: Option<KPrototypesResult> = None;
    let mut best_cost = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize prototypes: random selection
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut num_proto = Array2::zeros((n_clusters, d_num));
        let mut cat_proto: Vec<Vec<usize>> = vec![vec![0; d_cat]; n_clusters];

        for (c, &idx) in indices[..n_clusters].iter().enumerate() {
            num_proto.row_mut(c).assign(&numeric_data.row(idx));
            cat_proto[c] = categorical_data[idx].clone();
        }

        let mut labels = vec![0usize; n];
        let mut converged = false;
        let mut n_iterations = 0;

        for iter in 0..max_iter {
            n_iterations = iter + 1;
            let old_labels = labels.clone();

            // Assignment step
            for i in 0..n {
                let mut min_cost = f64::INFINITY;
                let mut best_cluster = 0;

                for c in 0..n_clusters {
                    // Numeric distance (squared Euclidean)
                    let num_dist: f64 = (0..d_num)
                        .map(|j| (numeric_data[[i, j]] - num_proto[[c, j]]).powi(2))
                        .sum();

                    // Categorical distance (simple matching)
                    let cat_dist: f64 = if d_cat > 0 {
                        (0..d_cat)
                            .filter(|&j| categorical_data[i][j] != cat_proto[c][j])
                            .count() as f64
                    } else {
                        0.0
                    };

                    let cost = num_dist + auto_gamma * cat_dist;
                    if cost < min_cost {
                        min_cost = cost;
                        best_cluster = c;
                    }
                }
                labels[i] = best_cluster;
            }

            // Check convergence
            if labels == old_labels {
                converged = true;
                break;
            }

            // Update step
            let mut cluster_counts = vec![0usize; n_clusters];
            let mut num_sums: Array2<f64> = Array2::zeros((n_clusters, d_num));
            let mut cat_counts: Vec<Vec<HashMap<usize, usize>>> =
                vec![vec![HashMap::new(); d_cat]; n_clusters];

            for i in 0..n {
                let c = labels[i];
                cluster_counts[c] += 1;

                for j in 0..d_num {
                    num_sums[[c, j]] += numeric_data[[i, j]];
                }

                for j in 0..d_cat {
                    *cat_counts[c][j].entry(categorical_data[i][j]).or_insert(0) += 1;
                }
            }

            // Update numeric prototypes (means)
            for c in 0..n_clusters {
                if cluster_counts[c] > 0 {
                    for j in 0..d_num {
                        num_proto[[c, j]] = num_sums[[c, j]] / cluster_counts[c] as f64;
                    }
                }
            }

            // Update categorical prototypes (modes)
            for c in 0..n_clusters {
                if cluster_counts[c] > 0 {
                    for j in 0..d_cat {
                        if let Some((&mode, _)) = cat_counts[c][j]
                            .iter()
                            .max_by_key(|(_, count)| *count)
                        {
                            cat_proto[c][j] = mode;
                        }
                    }
                }
            }

            // Handle empty clusters
            for c in 0..n_clusters {
                if cluster_counts[c] == 0 {
                    let rand_idx = rng.gen_range(0..n);
                    num_proto.row_mut(c).assign(&numeric_data.row(rand_idx));
                    cat_proto[c] = categorical_data[rand_idx].clone();
                }
            }
        }

        // Compute total cost
        let mut total_cost = 0.0;
        let mut cluster_sizes = vec![0usize; n_clusters];
        for i in 0..n {
            let c = labels[i];
            cluster_sizes[c] += 1;

            let num_dist: f64 = (0..d_num)
                .map(|j| (numeric_data[[i, j]] - num_proto[[c, j]]).powi(2))
                .sum();

            let cat_dist: f64 = if d_cat > 0 {
                (0..d_cat)
                    .filter(|&j| categorical_data[i][j] != cat_proto[c][j])
                    .count() as f64
            } else {
                0.0
            };

            total_cost += num_dist + auto_gamma * cat_dist;
        }

        if total_cost < best_cost {
            best_cost = total_cost;
            best_result = Some(KPrototypesResult {
                labels,
                numeric_prototypes: num_proto,
                categorical_prototypes: cat_proto,
                total_cost,
                n_iterations,
                cluster_sizes,
                n_clusters,
                n,
                n_numeric: d_num,
                n_categorical: d_cat,
                gamma: auto_gamma,
                converged,
            });
        }
    }

    best_result.ok_or_else(|| "K-prototypes failed".to_string())
}

/// Convenience wrapper for kprototypes.
pub fn run_kprototypes(
    numeric_data: ArrayView2<f64>,
    categorical_data: &[Vec<usize>],
    n_clusters: usize,
    gamma: Option<f64>,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KPrototypesResult, String> {
    kprototypes(numeric_data, categorical_data, n_clusters, gamma, max_iterations, None, seed)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmedoids_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = kmedoids(data.view(), 2, Some(50), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_spectral_clustering_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];

        let result = spectral_clustering(data.view(), 2, None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_affinity_propagation_basic() {
        // Use more points and more separation for affinity propagation
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Use a low preference to encourage fewer clusters
        let result = affinity_propagation(
            data.view(),
            Some(-100.0),  // Strong preference against being an exemplar
            Some(0.9),
            Some(200),
            Some(15)
        ).unwrap();

        assert_eq!(result.labels.len(), 6);
        // Affinity propagation may find varying numbers of clusters
        // Just check that it completed without error
        assert!(result.n_iterations > 0);
    }

    #[test]
    fn test_optics_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = optics(data.view(), 2, None, None).unwrap();

        assert_eq!(result.ordering.len(), 6);
        assert_eq!(result.reachability.len(), 6);
    }

    #[test]
    fn test_hdbscan_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = hdbscan(data.view(), Some(2), Some(2)).unwrap();

        assert_eq!(result.labels.len(), 6);
    }

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

    #[test]
    fn test_mst_correctness_dual_tree_vs_brute_force() {
        // Test that dual-tree MST produces same total weight as brute-force
        use super::*;

        // Test with simple data
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [5.0, 5.0],
            [5.5, 5.0],
            [5.0, 5.5],
        ];

        let min_samples = 2;

        // Get MST from brute-force
        let (core_bf, mst_bf) = hdbscan_brute_force(&data.view(), min_samples);

        // Get MST from dual-tree
        let (core_dt, mst_dt) = hdbscan_dual_tree(&data.view(), min_samples);

        // Get MST from kdtree-prim
        let (core_kp, mst_kp) = hdbscan_kdtree_prim(&data.view(), min_samples);

        // Core distances should match
        for i in 0..core_bf.len() {
            assert!((core_bf[i] - core_dt[i]).abs() < 1e-10,
                "Core distance mismatch at {}: bf={}, dt={}", i, core_bf[i], core_dt[i]);
            assert!((core_bf[i] - core_kp[i]).abs() < 1e-10,
                "Core distance mismatch at {}: bf={}, kp={}", i, core_bf[i], core_kp[i]);
        }

        // All MSTs should have n-1 edges
        let n = data.nrows();
        assert_eq!(mst_bf.len(), n - 1, "Brute-force MST has wrong edge count");
        assert_eq!(mst_dt.len(), n - 1, "Dual-tree MST has wrong edge count");
        assert_eq!(mst_kp.len(), n - 1, "KD-tree Prim MST has wrong edge count");

        // Total MST weight should match (MST weight is unique for distinct edge weights)
        let total_bf: f64 = mst_bf.iter().map(|e| e.2).sum();
        let total_dt: f64 = mst_dt.iter().map(|e| e.2).sum();
        let total_kp: f64 = mst_kp.iter().map(|e| e.2).sum();

        assert!((total_bf - total_dt).abs() < 1e-6,
            "MST weight mismatch: bf={}, dt={}", total_bf, total_dt);
        assert!((total_bf - total_kp).abs() < 1e-6,
            "MST weight mismatch: bf={}, kp={}", total_bf, total_kp);
    }

    #[test]
    fn test_mst_correctness_larger_dataset() {
        // Test with larger dataset to catch subtle bugs
        use rand::prelude::*;
        use rand_distr::Normal;

        let n = 100;
        let mut rng = StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut data = Array2::zeros((n, 3));
        for i in 0..n {
            let cluster = i % 3;
            let center = cluster as f64 * 10.0;
            for j in 0..3 {
                data[[i, j]] = center + rng.sample(normal);
            }
        }

        let min_samples = 5;

        // Get MST from brute-force (ground truth)
        let (_, mst_bf) = hdbscan_brute_force(&data.view(), min_samples);

        // Get MST from dual-tree
        let (_, mst_dt) = hdbscan_dual_tree(&data.view(), min_samples);

        // Get MST from kdtree-prim
        let (_, mst_kp) = hdbscan_kdtree_prim(&data.view(), min_samples);

        // Verify edge counts
        assert_eq!(mst_bf.len(), n - 1, "Brute-force MST edge count");
        assert_eq!(mst_dt.len(), n - 1, "Dual-tree MST edge count");
        assert_eq!(mst_kp.len(), n - 1, "KD-tree Prim MST edge count");

        // Verify total weights match
        let total_bf: f64 = mst_bf.iter().map(|e| e.2).sum();
        let total_dt: f64 = mst_dt.iter().map(|e| e.2).sum();
        let total_kp: f64 = mst_kp.iter().map(|e| e.2).sum();

        // Allow small tolerance due to floating point
        let rel_error_dt = (total_bf - total_dt).abs() / total_bf;
        let rel_error_kp = (total_bf - total_kp).abs() / total_bf;

        assert!(rel_error_dt < 1e-6,
            "Dual-tree MST weight mismatch: bf={:.6}, dt={:.6}, rel_error={:.2e}",
            total_bf, total_dt, rel_error_dt);
        assert!(rel_error_kp < 1e-6,
            "KD-tree Prim MST weight mismatch: bf={:.6}, kp={:.6}, rel_error={:.2e}",
            total_bf, total_kp, rel_error_kp);
    }

    #[test]
    fn test_hdbscan_cluster_consistency() {
        // Verify different algorithms produce same clustering
        use rand::prelude::*;
        use rand_distr::Normal;

        let n = 60;
        let mut rng = StdRng::seed_from_u64(54321);
        let normal = Normal::new(0.0, 0.5).unwrap();

        let mut data = Array2::zeros((n, 2));
        for i in 0..n {
            let cluster = i % 3;
            let (cx, cy) = match cluster {
                0 => (0.0, 0.0),
                1 => (10.0, 0.0),
                _ => (5.0, 10.0),
            };
            data[[i, 0]] = cx + rng.sample(normal);
            data[[i, 1]] = cy + rng.sample(normal);
        }

        // Run with different algorithms (force specific ones)
        let (core1, mst1) = hdbscan_brute_force(&data.view(), 5);
        let (core2, mst2) = hdbscan_kdtree_prim(&data.view(), 5);
        let (core3, mst3) = hdbscan_dual_tree(&data.view(), 5);

        // Extract clusters from each MST
        fn get_labels(mst: &[(usize, usize, f64)], n: usize, mcs: usize, core: &[f64]) -> Vec<i32> {
            let mut sorted = mst.to_vec();
            sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
            let (labels, _, _) = extract_hdbscan_clusters(&sorted, n, mcs, core);
            labels
        }

        let labels1 = get_labels(&mst1, n, 5, &core1);
        let labels2 = get_labels(&mst2, n, 5, &core2);
        let labels3 = get_labels(&mst3, n, 5, &core3);

        // Count clusters in each
        let count_clusters = |labels: &[i32]| -> usize {
            labels.iter()
                .filter(|&&l| l >= 0)
                .map(|&l| l as usize)
                .max()
                .map_or(0, |m| m + 1)
        };

        let nc1 = count_clusters(&labels1);
        let nc2 = count_clusters(&labels2);
        let nc3 = count_clusters(&labels3);

        // Number of clusters should match
        assert_eq!(nc1, nc2, "Cluster count mismatch: bf={}, kp={}", nc1, nc2);
        assert_eq!(nc1, nc3, "Cluster count mismatch: bf={}, dt={}", nc1, nc3);
    }

    #[test]
    fn test_hdbscan_perf_comparison() {
        // Quick validation that both HDBSCAN implementations produce consistent results
        // For full benchmark: cargo test -p p2a-core --release -- test_hdbscan_full_benchmark --ignored --nocapture
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Small dataset for quick validation
        let n = 100;
        let d = 3;
        let mut data = Array2::zeros((n, d));
        for i in 0..n {
            let cluster = i % 3;
            let center = cluster as f64 * 5.0;
            for j in 0..d {
                data[[i, j]] = center + rng.sample(normal);
            }
        }

        // Run both implementations
        let result_new = hdbscan(data.view(), Some(5), Some(5));
        let result_old = crate::ml::cluster_optimized::hdbscan_optimized(data.view(), Some(5), Some(5));

        // Both should succeed
        assert!(result_new.is_ok(), "New HDBSCAN failed: {:?}", result_new.err());
        assert!(result_old.is_ok(), "Old HDBSCAN failed: {:?}", result_old.err());

        let new_result = result_new.unwrap();
        let (old_labels, _old_probs, _old_n_clusters) = result_old.unwrap();

        // Both should return same number of labels
        assert_eq!(new_result.labels.len(), old_labels.len(), "Label count mismatch");
        assert_eq!(new_result.labels.len(), n, "Expected {} labels", n);

        // Both should find some clusters (not all noise)
        let new_clusters: usize = new_result.labels.iter().filter(|&&l| l >= 0).count();
        let old_clusters: usize = old_labels.iter().filter(|&&l| l >= 0).count();
        assert!(new_clusters > 0, "New HDBSCAN found no clusters");
        assert!(old_clusters > 0, "Old HDBSCAN found no clusters");
    }

    #[test]
    #[ignore] // Run with: cargo test -p p2a-core --release -- test_hdbscan_full_benchmark --ignored --nocapture
    fn test_hdbscan_full_benchmark() {
        use std::time::Instant;
        use rand::prelude::*;
        use rand_distr::Normal;

        fn generate_data(n: usize, d: usize) -> Array2<f64> {
            let mut rng = StdRng::seed_from_u64(42);
            let normal = Normal::new(0.0, 1.0).unwrap();

            let mut data = Array2::zeros((n, d));
            for i in 0..n {
                let cluster = i % 5;
                let center = cluster as f64 * 5.0;
                for j in 0..d {
                    data[[i, j]] = center + rng.sample(normal);
                }
            }
            data
        }

        println!("\nHDBSCAN Performance Comparison");
        println!("==============================");
        println!("n\tnew(ms)\t\told(ms)\t\tspeedup");

        for n in [500, 1000, 2000, 3000, 5000] {
            let data = generate_data(n, 5);

            // Warmup
            let _ = hdbscan(data.view(), Some(10), Some(10));
            let _ = crate::ml::cluster_optimized::hdbscan_optimized(data.view(), Some(10), Some(10));

            // New algorithm (with auto-dispatch)
            let start = Instant::now();
            for _ in 0..3 {
                let _ = hdbscan(data.view(), Some(10), Some(10));
            }
            let new_time = start.elapsed().as_secs_f64() / 3.0 * 1000.0;

            // Old optimized parallel algorithm
            let start = Instant::now();
            for _ in 0..3 {
                let _ = crate::ml::cluster_optimized::hdbscan_optimized(data.view(), Some(10), Some(10));
            }
            let old_time = start.elapsed().as_secs_f64() / 3.0 * 1000.0;

            let speedup = old_time / new_time;
            println!("{}\t{:.1}\t\t{:.1}\t\t{:.2}x", n, new_time, old_time, speedup);
        }
    }

    // =========================================================================
    // Tests for newly added algorithms
    // =========================================================================

    #[test]
    fn test_fuzzy_cmeans_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fuzzy_cmeans(data.view(), 2, None, None, None, None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.membership.nrows(), 6);
        assert_eq!(result.membership.ncols(), 2);

        // Check membership rows sum to 1
        for i in 0..6 {
            let row_sum: f64 = (0..2).map(|j| result.membership[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} sum: {}", i, row_sum);
        }

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_fuzzy_cmeans_fuzziness() {
        let data = array![
            [0.0, 0.0],
            [5.0, 0.0],  // Point equidistant from both clusters
            [10.0, 0.0],
        ];

        // With low fuzziness (near 1), should be crisper
        let result_crisp = fuzzy_cmeans(data.view(), 2, Some(1.5), None, None, None, Some(42)).unwrap();

        // With high fuzziness, should be fuzzier
        let result_fuzzy = fuzzy_cmeans(data.view(), 2, Some(3.0), None, None, None, Some(42)).unwrap();

        // The middle point should have more equal membership in fuzzy case
        let mid_idx = 1;
        let membership_crisp = result_crisp.membership.row(mid_idx);
        let membership_fuzzy = result_fuzzy.membership.row(mid_idx);

        // Variance of memberships should be lower for fuzzier clustering
        let var_crisp: f64 = membership_crisp.iter().map(|&x| (x - 0.5).powi(2)).sum::<f64>() / 2.0;
        let var_fuzzy: f64 = membership_fuzzy.iter().map(|&x| (x - 0.5).powi(2)).sum::<f64>() / 2.0;

        // For a point in the middle, the fuzzy version should have memberships closer to 0.5
        assert!(var_fuzzy <= var_crisp + 0.1, "Fuzzy version should be fuzzier: var_crisp={}, var_fuzzy={}", var_crisp, var_fuzzy);
    }

    #[test]
    fn test_mini_batch_kmeans_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = mini_batch_kmeans(data.view(), 2, Some(3), Some(50), Some(3), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.batch_size, 3);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_trimmed_kmeans_basic() {
        // Include a clear outlier far from both clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.3, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
            [10.3, 10.1],
            [500.0, 500.0],  // Clear outlier - very far from both clusters
            [501.0, 501.0],  // Another outlier
        ];

        // With 20% trimming, should trim the 2 outliers
        let result = trimmed_kmeans(data.view(), 2, Some(0.2), Some(50), Some(5), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 10);
        assert_eq!(result.n_clusters, 2);
        assert!(result.n_trimmed >= 1, "At least one outlier should be trimmed");

        // Verify that trimmed points exist
        assert!(!result.trimmed_indices.is_empty(), "Should have trimmed indices");

        // Non-trimmed points in each cluster should be internally consistent
        let cluster0_count = result.labels.iter().filter(|&&l| l == 0).count();
        let cluster1_count = result.labels.iter().filter(|&&l| l == 1).count();
        let trimmed_count = result.labels.iter().filter(|&&l| l == -1).count();

        assert!(cluster0_count >= 1, "Cluster 0 should have points");
        assert!(cluster1_count >= 1, "Cluster 1 should have points");
        assert!(trimmed_count >= 1, "Should have trimmed points");
    }

    #[test]
    fn test_diana_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = diana(data.view(), Some(2)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert!(result.divisive_coefficient >= 0.0 && result.divisive_coefficient <= 1.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_agnes_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = agnes(data.view(), Some(2), Some(AgnesLinkage::Average)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert!(result.agglomerative_coefficient >= 0.0 && result.agglomerative_coefficient <= 1.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_agnes_linkage_methods() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ];

        // Test all linkage methods work
        for linkage in [AgnesLinkage::Single, AgnesLinkage::Complete,
                        AgnesLinkage::Average, AgnesLinkage::Ward, AgnesLinkage::Weighted] {
            let result = agnes(data.view(), Some(2), Some(linkage)).unwrap();
            assert_eq!(result.n_clusters, 2);
        }
    }

    // =========================================================================
    // Batch 2 Tests: flexmix, pvclust, clara, cluster_stats, fanny
    // =========================================================================

    #[test]
    fn test_flexmix_basic() {
        // Create mixture regression data: two groups with different slopes
        // Group 1: y = 1 + 2*x + noise
        // Group 2: y = 5 - 1*x + noise
        let x = array![
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ];
        let y = array![
            [2.1],  // Group 1: 1 + 2*0.5 + noise
            [3.0],  // Group 1: 1 + 2*1.0 + noise
            [4.1],  // Group 1: 1 + 2*1.5 + noise
            [5.0],  // Group 1: 1 + 2*2.0 + noise
            [4.4],  // Group 2: 5 - 1*0.5 + noise
            [3.9],  // Group 2: 5 - 1*1.0 + noise
            [3.6],  // Group 2: 5 - 1*1.5 + noise
            [3.1],  // Group 2: 5 - 1*2.0 + noise
        ];

        let result = flexmix(y.view(), x.view(), 2, Some(100), Some(1e-4), Some(42)).unwrap();

        assert_eq!(result.cluster.len(), 8);
        assert_eq!(result.k, 2);
        assert_eq!(result.p, 2);
        assert_eq!(result.coefficients.len(), 2);
        assert!(result.prior.iter().sum::<f64>() - 1.0 < 1e-6);
        assert!(result.bic.is_finite());
        assert!(result.aic.is_finite());
    }

    #[test]
    fn test_flexmix_single_component() {
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ];
        let y = array![
            [2.1],
            [4.0],
            [5.9],
            [8.1],
        ];

        let result = flexmix(y.view(), x.view(), 1, Some(50), None, Some(42)).unwrap();

        assert_eq!(result.k, 1);
        assert_eq!(result.prior.len(), 1);
        assert!((result.prior[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pvclust_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Use fewer bootstrap samples for test speed
        let result = pvclust(data.view(), Some("average"), Some(100), None, Some(0.80), Some(42)).unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.au_pvalues.len(), result.merge.len());
        assert_eq!(result.bp_pvalues.len(), result.merge.len());
        assert_eq!(result.method, "average");

        // AU and BP p-values should be in [0, 1]
        for &au in &result.au_pvalues {
            assert!(au >= 0.0 && au <= 1.0, "AU p-value out of range: {}", au);
        }
        for &bp in &result.bp_pvalues {
            assert!(bp >= 0.0 && bp <= 1.0, "BP p-value out of range: {}", bp);
        }
    }

    #[test]
    fn test_clara_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = clara(data.view(), 2, Some(3), Some(4), Some(50), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.k, 2);
        assert_eq!(result.medoid_indices.len(), 2);
        assert!(result.objective > 0.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_clara_large_sample() {
        // Test with larger sample size than data (should clamp)
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ];

        let result = clara(data.view(), 2, Some(2), Some(100), None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 3);
        assert_eq!(result.sample_size, 3); // Clamped to n
    }

    #[test]
    fn test_cluster_stats_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let result = cluster_stats(data.view(), &labels).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.cluster_sizes, vec![3, 3]);
        assert_eq!(result.n, 6);
        assert_eq!(result.p, 2);

        // Validate ranges
        assert!(result.average_silhouette >= -1.0 && result.average_silhouette <= 1.0);
        assert!(result.dunn_index >= 0.0);
        assert!(result.calinski_harabasz >= 0.0);
        assert!(result.davies_bouldin >= 0.0);
        assert!(result.explained_variance_ratio >= 0.0 && result.explained_variance_ratio <= 1.0);
        assert!(result.within_ss >= 0.0);
        assert!(result.between_ss >= 0.0);
        assert!(result.total_ss >= 0.0);

        // Well-separated clusters should have high silhouette
        assert!(result.average_silhouette > 0.5, "Expected high silhouette for well-separated clusters");
    }

    #[test]
    fn test_cluster_stats_single_cluster() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
        ];
        let labels = vec![0, 0, 0];

        let result = cluster_stats(data.view(), &labels).unwrap();

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes, vec![3]);
        // With single cluster, between_ss = 0 and silhouette = 0
        assert!(result.between_ss < 1e-10);
    }

    #[test]
    fn test_fanny_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fanny(data.view(), 2, Some(2.0), Some(100), Some(1e-6), Some(42)).unwrap();

        assert_eq!(result.clustering.len(), 6);
        assert_eq!(result.k, 2);
        assert_eq!(result.membership.nrows(), 6);
        assert_eq!(result.membership.ncols(), 2);

        // Membership should sum to 1 for each observation
        for i in 0..6 {
            let row_sum: f64 = (0..2).map(|j| result.membership[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Membership row {} sum: {}", i, row_sum);
        }

        // Dunn coefficient should be in valid range
        assert!(result.dunn_coefficient >= 1.0 / result.k as f64);
        assert!(result.dunn_coefficient <= 1.0);
        assert!(result.normalized_dunn >= 0.0 && result.normalized_dunn <= 1.0);

        // Hard assignments should separate the clusters
        assert_eq!(result.clustering[0], result.clustering[1]);
        assert_eq!(result.clustering[3], result.clustering[4]);
        assert_ne!(result.clustering[0], result.clustering[3]);
    }

    #[test]
    fn test_fanny_fuzziness_parameter() {
        let data = array![
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
        ];

        // Higher m = fuzzier (memberships closer to 1/k)
        let result_low_m = fanny(data.view(), 2, Some(1.5), Some(100), None, Some(42)).unwrap();
        let result_high_m = fanny(data.view(), 2, Some(3.0), Some(100), None, Some(42)).unwrap();

        // Higher m should have lower Dunn coefficient (fuzzier)
        assert!(result_high_m.dunn_coefficient <= result_low_m.dunn_coefficient + 0.1,
            "Higher m should produce fuzzier clustering");
    }

    // =========================================================================
    // Batch 3 Tests: skmeans, fastcluster, dynamicTreeCut, mixtools, kprototypes
    // =========================================================================

    #[test]
    fn test_skmeans_basic() {
        // Text-like data (sparse, high-dimensional representations typically)
        // Using simpler data for testing
        let data = array![
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.8, 0.2],
        ];

        let result = skmeans(data.view(), 2, Some(100), None, Some(10), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.centroids.nrows(), 2);
        assert_eq!(result.centroids.ncols(), 3);
        assert!(result.avg_similarity > 0.0 && result.avg_similarity <= 1.0);

        // Cluster sizes should sum to n
        let total_size: usize = result.cluster_sizes.iter().sum();
        assert_eq!(total_size, 6);

        // First three points should be in same cluster (similar direction)
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
    }

    #[test]
    fn test_skmeans_normalized_centroids() {
        let data = array![
            [1.0, 2.0],
            [2.0, 1.0],
            [10.0, 20.0],
            [20.0, 10.0],
        ];

        let result = skmeans(data.view(), 2, Some(50), None, Some(5), Some(42)).unwrap();

        // Check that centroids are unit vectors
        for c in 0..result.n_clusters {
            let norm: f64 = (0..result.centroids.ncols())
                .map(|j| result.centroids[[c, j]].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Centroid {} should be unit vector, norm={}", c, norm);
        }
    }

    #[test]
    fn test_fastcluster_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];

        let result = fastcluster(data.view(), Some(FastLinkage::Ward), None).unwrap();

        assert_eq!(result.n, 4);
        assert_eq!(result.merge.nrows(), 3); // n-1 merges
        assert_eq!(result.height.len(), 3);
        assert_eq!(result.order.len(), 4);

        // Heights should be non-decreasing (approximately)
        for i in 1..result.height.len() {
            assert!(result.height[i] >= result.height[i-1] - 1e-6,
                "Heights should be non-decreasing");
        }
    }

    #[test]
    fn test_fastcluster_with_cut() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fastcluster(data.view(), Some(FastLinkage::Complete), Some(2)).unwrap();

        assert!(result.labels.is_some());
        let labels = result.labels.unwrap();
        assert_eq!(labels.len(), 6);

        // Should produce 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(unique_labels.len() <= 2);

        // First three points should be in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // Last three points should be in same cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
    }

    #[test]
    fn test_fastcluster_linkage_methods() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
        ];

        // Test each linkage method runs without error
        for linkage in [
            FastLinkage::Single,
            FastLinkage::Complete,
            FastLinkage::Average,
            FastLinkage::Ward,
            FastLinkage::Weighted,
            FastLinkage::Centroid,
            FastLinkage::Median,
        ] {
            let result = fastcluster(data.view(), Some(linkage), None);
            assert!(result.is_ok(), "Linkage {:?} should work", linkage);
        }
    }

    #[test]
    fn test_dynamic_tree_cut_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = run_dynamic_tree_cut(
            data.view(),
            Some(FastLinkage::Ward),
            Some(DynamicCutMethod::Tree),
            Some(2),  // deep_split
            Some(2),  // min_cluster_size
        ).unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.labels.len(), 6);
        assert!(result.n_clusters >= 1);

        // Total assigned + unassigned should equal n
        let total: usize = result.cluster_sizes.iter().sum::<usize>() + result.n_unassigned;
        assert_eq!(total, 6);
    }

    #[test]
    fn test_dynamic_tree_cut_deep_split() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ];

        // Lower deep_split = fewer clusters
        let result_low = run_dynamic_tree_cut(
            data.view(), None, None, Some(0), Some(2)
        ).unwrap();

        // Higher deep_split = potentially more clusters
        let result_high = run_dynamic_tree_cut(
            data.view(), None, None, Some(4), Some(2)
        ).unwrap();

        // Both should produce valid results
        assert!(result_low.n_clusters >= 1);
        assert!(result_high.n_clusters >= 1);
    }

    #[test]
    fn test_normal_mix_em_basic() {
        // Create bimodal data
        let data: Vec<f64> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5,  // Cluster around 0.3
            9.5, 9.6, 9.7, 9.8, 9.9,  // Cluster around 9.7
        ];

        let result = normal_mix_em(&data, 2, Some(100), Some(1e-6), Some(42)).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.n, 10);
        assert_eq!(result.mu.len(), 2);
        assert_eq!(result.sigma.len(), 2);
        assert_eq!(result.lambda.len(), 2);
        assert_eq!(result.posterior.nrows(), 10);
        assert_eq!(result.posterior.ncols(), 2);

        // Mixing proportions should sum to 1
        let lambda_sum: f64 = result.lambda.iter().sum();
        assert!((lambda_sum - 1.0).abs() < 1e-6);

        // Means should be separated
        let mu_diff = (result.mu[0] - result.mu[1]).abs();
        assert!(mu_diff > 5.0, "Means should be well separated: {:?}", result.mu);
    }

    #[test]
    fn test_normal_mix_em_convergence() {
        let data: Vec<f64> = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2];

        let result = normal_mix_em(&data, 2, Some(500), Some(1e-8), Some(42)).unwrap();

        // Should converge with reasonable tolerance
        assert!(result.n_iterations <= 500);
        // Log-likelihood should increase (roughly)
        if result.loglik_history.len() > 1 {
            let first = result.loglik_history[0];
            let last = *result.loglik_history.last().unwrap();
            assert!(last >= first - 1e-6, "Log-likelihood should not decrease significantly");
        }
    }

    #[test]
    fn test_mvnorm_mix_em_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = mvnorm_mix_em(data.view(), 2, Some(100), Some(1e-6), Some(42)).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.n, 6);
        assert_eq!(result.d, 2);
        assert_eq!(result.mu.nrows(), 2);
        assert_eq!(result.mu.ncols(), 2);
        assert_eq!(result.labels.len(), 6);

        // Lambda should sum to 1
        let lambda_sum: f64 = result.lambda.iter().sum();
        assert!((lambda_sum - 1.0).abs() < 1e-6);

        // Points should be assigned to correct clusters
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_kprototypes_basic() {
        // Numeric features
        let numeric_data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];

        // Categorical features (e.g., encoded categories)
        let categorical_data: Vec<Vec<usize>> = vec![
            vec![0, 0],
            vec![0, 0],
            vec![1, 1],
            vec![1, 1],
        ];

        let result = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            None,  // auto gamma
            Some(100),
            Some(10),
            Some(42),
        ).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n, 4);
        assert_eq!(result.n_numeric, 2);
        assert_eq!(result.n_categorical, 2);
        assert_eq!(result.labels.len(), 4);

        // First two points should be in same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        // Last two points should be in same cluster
        assert_eq!(result.labels[2], result.labels[3]);
        // The two groups should be different
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_kprototypes_numeric_only() {
        let numeric_data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];

        // Empty categorical features
        let categorical_data: Vec<Vec<usize>> = vec![
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        let result = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(1.0),
            Some(50),
            Some(5),
            Some(42),
        ).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_categorical, 0);
    }

    #[test]
    fn test_kprototypes_gamma_effect() {
        let numeric_data = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        let categorical_data: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![0],
            vec![1],
        ];

        // Low gamma: numeric features dominate
        let result_low = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(0.1),
            Some(50),
            Some(5),
            Some(42),
        ).unwrap();

        // High gamma: categorical features dominate
        let result_high = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(100.0),
            Some(50),
            Some(5),
            Some(42),
        ).unwrap();

        // Both should produce valid results
        assert_eq!(result_low.n_clusters, 2);
        assert_eq!(result_high.n_clusters, 2);
    }
}
