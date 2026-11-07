//! K-means variants: mini-batch ([`mini_batch_kmeans`]), trimmed ([`trimmed_kmeans`]) and spherical ([`skmeans`]).

use crate::resource::checkpoint;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Seed of the random starts when `seed` is `None`.
pub const KMEANS_VARIANTS_DEFAULT_SEED: u64 = 42;

/// Shared parameter checks of the three variants.
fn check_runs(max_iterations: usize, n_init: usize) -> Result<(), String> {
    if max_iterations == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    if n_init == 0 {
        return Err("n_init must be at least 1, got 0".to_string());
    }
    Ok(())
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
/// * `batch_size` - Size of mini-batches, between 1 and n (default: 100 or n if smaller)
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
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }

    let batch = batch_size.unwrap_or(100.min(n));
    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(3);
    if batch == 0 || batch > n {
        return Err(format!(
            "batch_size must be between 1 and {n} (the number of rows), got {batch}"
        ));
    }
    check_runs(max_iter, n_inits)?;
    crate::ml::extract::require_finite(&data, "mini-batch k-means")?;

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(KMEANS_VARIANTS_DEFAULT_SEED));

    let mut best_result: Option<MiniBatchKMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize centroids using k-means++
        let mut centroids = mini_batch_kmeans_plusplus_init(&data, n_clusters, &mut rng);
        let mut counts = vec![1.0f64; n_clusters]; // Track update counts per centroid

        // Mini-batch loop
        for _iter in 0..max_iter {
            checkpoint().map_err(|e| e.to_string())?;
            // Sample a mini-batch
            let batch_indices: Vec<usize> = (0..batch).map(|_| rng.gen_range(0..n)).collect();

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
                    centroids[[nearest, k]] =
                        (1.0 - eta) * centroids[[nearest, k]] + eta * data[[idx, k]];
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
    /// Whether the concentration step reached its fixed point (the assignment did not change)
    #[serde(default)]
    pub converged: bool,
    /// What the fit could not do
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for TrimmedKMeansResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for TrimmedKMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Trimmed K-Means Clustering Results")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Trimming proportion: {:.1}%", self.alpha * 100.0)?;
        writeln!(f, "Trimmed points: {}", self.n_trimmed)?;
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

        write!(f, "{}", notes_text(self))
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
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }

    let trim_prop = alpha.unwrap_or(0.1);
    if !(0.0..0.5).contains(&trim_prop) {
        return Err("alpha must be between 0 and 0.5".to_string());
    }

    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(10);
    let n_trim = (n as f64 * trim_prop).ceil() as usize;
    let n_keep = n - n_trim;
    check_runs(max_iter, n_inits)?;
    if n_keep < n_clusters {
        return Err(format!(
            "alpha = {trim_prop} trims {n_trim} of {n} rows, which leaves fewer rows than the \
             {n_clusters} clusters"
        ));
    }
    crate::ml::extract::require_finite(&data, "trimmed k-means")?;

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(KMEANS_VARIANTS_DEFAULT_SEED));

    let mut best_result: Option<TrimmedKMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize centroids using k-means++ on non-trimmed subset
        let mut centroids = mini_batch_kmeans_plusplus_init(&data, n_clusters, &mut rng);
        let mut labels = vec![-1i32; n];
        let mut previous = vec![-2i32; n];
        let mut trimmed_mask = vec![false; n];
        let mut n_iterations = 0;
        let mut converged = false;
        let mut changed = n;

        for iter in 0..max_iter {
            checkpoint().map_err(|e| e.to_string())?;
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
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));

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

            // The concentration step's fixed point: the same assignment gives the same centroids.
            changed = labels.iter().zip(&previous).filter(|(a, b)| a != b).count();
            if changed == 0 {
                converged = true;
                break;
            }
            previous.copy_from_slice(&labels);

            // Update centroids using only non-trimmed points
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

            let convergence = Convergence::new(
                converged,
                n_iterations,
                max_iter,
                "labels changed in the last iteration",
                changed as f64,
                0.0,
            );
            let warnings: Vec<String> = convergence.note("Trimmed k-means").into_iter().collect();
            let trimmed_indices: Vec<usize> = trimmed_mask
                .iter()
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
                converged,
                warnings,
                convergence,
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
// Spherical K-Means (skmeans)
// =============================================================================

/// Result of Spherical K-Means clustering.
///
/// # References
///
/// - Dhillon, I. S., & Modha, D. S. (2001). "Concept decompositions for large
///   sparse text data using clustering." Machine Learning, 42(1), 143-175.
/// - R skmeans package documentation
///   Source: <https://www.rdocumentation.org/packages/skmeans/versions/0.2-17/topics/skmeans>
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
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }

    let max_iter = max_iterations.unwrap_or(100);
    let tolerance = tol.unwrap_or(1e-6);
    let n_inits = n_init.unwrap_or(10);
    check_runs(max_iter, n_inits)?;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(format!("tol must be finite and positive, got {tolerance}"));
    }
    crate::ml::extract::require_finite(&data, "spherical k-means")?;

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(KMEANS_VARIANTS_DEFAULT_SEED));

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
            checkpoint().map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::array;

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn noisy_blobs() -> Array2<f64> {
        Array2::from_shape_fn((30, 2), |(i, j)| {
            [[1.0, 0.2], [0.2, 1.0], [-1.0, 0.4]][i % 3][j] + 0.1 * ((i * 7 + j * 3) as f64).sin()
        })
    }

    #[test]
    fn test_kmeans_variants_stop_on_a_cancelled_budget() {
        let x = noisy_blobs();
        let e = with_budget(cancelled(), || {
            skmeans(x.view(), 3, None, None, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "skmeans: {e}");
        let e = with_budget(cancelled(), || {
            mini_batch_kmeans(x.view(), 3, None, None, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "mini-batch: {e}");
        let e = with_budget(cancelled(), || {
            trimmed_kmeans(x.view(), 3, None, None, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "trimmed: {e}");
    }

    #[test]
    fn test_mini_batch_kmeans_bounds_the_batch_by_the_rows() {
        let x = noisy_blobs();
        for bad in [0, 31, usize::MAX] {
            let e = mini_batch_kmeans(x.view(), 3, Some(bad), None, None, Some(1)).unwrap_err();
            assert!(e.contains("batch_size must be between 1 and 30"), "{e}");
        }
        let full = mini_batch_kmeans(x.view(), 3, Some(30), Some(5), Some(1), Some(1)).unwrap();
        assert_eq!(full.batch_size, 30);
    }

    #[test]
    fn test_trimmed_kmeans_is_scale_equivariant() {
        // The old stop |shift| < 1e-6 (absolute) ended after one pass on data in small units.
        let x = Array2::from_shape_fn((60, 2), |(i, j)| {
            [[0.0, 0.0], [2.0, 0.5], [1.0, 2.0]][i % 3][j] + 0.6 * ((i * 13 + j * 5) as f64).sin()
        });
        let base = trimmed_kmeans(x.view(), 3, Some(0.1), Some(100), Some(1), Some(4)).unwrap();
        let small = trimmed_kmeans(
            (&x * 2f64.powi(-24)).view(),
            3,
            Some(0.1),
            Some(100),
            Some(1),
            Some(4),
        )
        .unwrap();
        assert!(base.converged && small.converged);
        assert_eq!(
            (small.labels.clone(), small.n_iterations),
            (base.labels.clone(), base.n_iterations)
        );
        assert!(base.n_iterations > 1);
    }

    #[test]
    fn test_trimmed_kmeans_reports_the_iteration_limit() {
        let x = Array2::from_shape_fn((60, 2), |(i, j)| {
            [[0.0, 0.0], [2.0, 0.5], [1.0, 2.0]][i % 3][j] + 0.6 * ((i * 13 + j * 5) as f64).sin()
        });
        let r = trimmed_kmeans(x.view(), 3, Some(0.1), Some(1), Some(1), Some(4)).unwrap();
        assert!(!r.converged);
        assert!(
            r.warnings.iter().any(|w| w.contains("did not converge")),
            "{:?}",
            r.warnings
        );
        assert!(format!("{r}").contains("Converged: no"));
    }

    #[test]
    fn test_kmeans_variants_reject_bad_parameters_and_default_seeds_are_fixed() {
        let x = Array2::from_shape_fn((12, 2), |(i, j)| {
            (i % 3) as f64 * 4.0 + 0.1 * ((i + j) as f64).cos()
        });
        assert!(
            mini_batch_kmeans(x.view(), 2, Some(0), None, None, None)
                .unwrap_err()
                .contains("batch_size")
        );
        assert!(
            mini_batch_kmeans(x.view(), 2, None, Some(0), None, None)
                .unwrap_err()
                .contains("max_iterations")
        );
        assert!(
            trimmed_kmeans(x.view(), 2, Some(0.1), None, Some(0), None)
                .unwrap_err()
                .contains("n_init")
        );
        assert!(
            trimmed_kmeans(x.view(), 11, Some(0.45), None, None, None)
                .unwrap_err()
                .contains("alpha")
        );
        assert!(
            skmeans(x.view(), 2, None, Some(-1.0), None, None)
                .unwrap_err()
                .contains("tol")
        );
        let mut nan = x.clone();
        nan[[2, 1]] = f64::NAN;
        assert!(
            mini_batch_kmeans(nan.view(), 2, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
        assert!(
            trimmed_kmeans(nan.view(), 2, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
        assert!(
            skmeans(nan.view(), 2, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
        let a = mini_batch_kmeans(x.view(), 2, None, None, None, None).unwrap();
        let b = mini_batch_kmeans(
            x.view(),
            2,
            None,
            None,
            None,
            Some(KMEANS_VARIANTS_DEFAULT_SEED),
        )
        .unwrap();
        assert_eq!(a.inertia.to_bits(), b.inertia.to_bits());
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

        let result =
            mini_batch_kmeans(data.view(), 2, Some(3), Some(50), Some(3), Some(42)).unwrap();

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
            [500.0, 500.0], // Clear outlier - very far from both clusters
            [501.0, 501.0], // Another outlier
        ];

        // With 20% trimming, should trim the 2 outliers
        let result =
            trimmed_kmeans(data.view(), 2, Some(0.2), Some(50), Some(5), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 10);
        assert_eq!(result.n_clusters, 2);
        assert!(
            result.n_trimmed >= 1,
            "At least one outlier should be trimmed"
        );

        // Verify that trimmed points exist
        assert!(
            !result.trimmed_indices.is_empty(),
            "Should have trimmed indices"
        );

        // Non-trimmed points in each cluster should be internally consistent
        let cluster0_count = result.labels.iter().filter(|&&l| l == 0).count();
        let cluster1_count = result.labels.iter().filter(|&&l| l == 1).count();
        let trimmed_count = result.labels.iter().filter(|&&l| l == -1).count();

        assert!(cluster0_count >= 1, "Cluster 0 should have points");
        assert!(cluster1_count >= 1, "Cluster 1 should have points");
        assert!(trimmed_count >= 1, "Should have trimmed points");
    }

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
        let data = array![[1.0, 2.0], [2.0, 1.0], [10.0, 20.0], [20.0, 10.0],];

        let result = skmeans(data.view(), 2, Some(50), None, Some(5), Some(42)).unwrap();

        // Check that centroids are unit vectors
        for c in 0..result.n_clusters {
            let norm: f64 = (0..result.centroids.ncols())
                .map(|j| result.centroids[[c, j]].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Centroid {} should be unit vector, norm={}",
                c,
                norm
            );
        }
    }
}
