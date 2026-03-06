//! Clustering algorithms: K-means, DBSCAN, and Hierarchical Clustering.
//!
//! Pure Rust implementations using ndarray.
//!
//! DBSCAN uses KD-tree acceleration for O(n log n) average complexity
//! when dimensionality ≤ 20, with parallel neighborhood queries via rayon.

use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// K-means clustering result.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster assignments for each point (0 to k-1)
    pub labels: Vec<usize>,
    /// Centroid positions (k x features)
    pub centroids: Array2<f64>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Within-cluster sum of squares (inertia)
    pub inertia: f64,
    /// Number of points in each cluster
    pub cluster_sizes: Vec<usize>,
}

impl std::fmt::Display for KMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-Means Clustering Results")?;
        writeln!(f, "==========================")?;
        writeln!(f, "Number of clusters: {}", self.centroids.nrows())?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Inertia (WCSS): {:.4}", self.inertia)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, size)?;
        }
        writeln!(f)?;
        writeln!(f, "Centroids:")?;
        for i in 0..self.centroids.nrows() {
            let centroid: Vec<String> = self
                .centroids
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            writeln!(f, "  Cluster {}: [{}]", i, centroid.join(", "))?;
        }
        Ok(())
    }
}

/// DBSCAN clustering result.
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster assignments (-1 for noise, 0+ for clusters)
    pub labels: Vec<i32>,
    /// Number of clusters found (excluding noise)
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// Core sample indices
    pub core_sample_indices: Vec<usize>,
}

impl std::fmt::Display for DBSCANResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DBSCAN Clustering Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(
            f,
            "Number of core samples: {}",
            self.core_sample_indices.len()
        )?;

        // Count points per cluster
        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().collect();
        labels_sorted.sort();
        for &label in &labels_sorted {
            let count = cluster_counts[label];
            if *label == -1 {
                writeln!(f, "  Noise: {} points", count)?;
            } else {
                writeln!(f, "  Cluster {}: {} points", label, count)?;
            }
        }
        Ok(())
    }
}

/// Run K-means clustering.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations (default 300)
/// * `tolerance` - Convergence tolerance (default 1e-4)
/// * `n_init` - Number of initializations to try (default 10)
/// * `seed` - Optional random seed for reproducibility
pub fn kmeans(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<KMeansResult, String> {
    let n_samples = data.nrows();
    let _n_features = data.ncols();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n_samples {
        return Err(format!(
            "k ({}) cannot be greater than n_samples ({})",
            k, n_samples
        ));
    }

    let max_iter = max_iterations.unwrap_or(300);
    let tol = tolerance.unwrap_or(1e-4);
    let n_initializations = n_init.unwrap_or(10);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut best_result: Option<KMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_initializations {
        // Initialize centroids using k-means++
        let centroids = kmeans_plusplus_init(&data, k, &mut rng);

        // Run k-means
        let result = kmeans_single(&data, centroids, max_iter, tol);

        if result.inertia < best_inertia {
            best_inertia = result.inertia;
            best_result = Some(result);
        }
    }

    best_result.ok_or_else(|| "K-means failed to converge".to_string())
}

/// K-means++ initialization.
fn kmeans_plusplus_init(data: &ArrayView2<f64>, k: usize, rng: &mut StdRng) -> Array2<f64> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    let mut centroids = Array2::zeros((k, n_features));

    // Choose first centroid randomly
    let first_idx = rng.gen_range(0..n_samples);
    centroids.row_mut(0).assign(&data.row(first_idx));

    // Choose remaining centroids
    for i in 1..k {
        // Compute distances to nearest centroid
        let mut distances = Vec::with_capacity(n_samples);
        for j in 0..n_samples {
            let point = data.row(j);
            let mut min_dist = f64::INFINITY;
            for c in 0..i {
                let dist = euclidean_distance_squared(&point, &centroids.row(c));
                min_dist = min_dist.min(dist);
            }
            distances.push(min_dist);
        }

        // Sample proportional to squared distance
        let total: f64 = distances.iter().sum();
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen_idx = 0;
        for (idx, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen_idx = idx;
                break;
            }
        }

        centroids.row_mut(i).assign(&data.row(chosen_idx));
    }

    centroids
}

/// Single run of k-means.
///
/// When the `cuda` feature is enabled and a GPU is available, uses DGEMM-based
/// pairwise distance computation for the assignment step (n >= threshold).
fn kmeans_single(
    data: &ArrayView2<f64>,
    mut centroids: Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> KMeansResult {
    let n_samples = data.nrows();
    let k = centroids.nrows();

    let mut labels = vec![0usize; n_samples];
    let mut n_iterations = 0;

    // Check if GPU is available for distance computation.
    // GPU helps for d >= 20 (DGEMM-based distances) but hurts for small d.
    #[cfg(feature = "cuda")]
    let gpu_ctx = crate::linalg::gpu::GpuContext::get().filter(|ctx| {
        let d = data.ncols();
        n_samples >= ctx.thresholds.kmeans_min_n && d >= ctx.thresholds.kmeans_min_d
    });
    #[cfg(not(feature = "cuda"))]
    let gpu_ctx: Option<&()> = None;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        if gpu_ctx.is_some() {
            // GPU path: compute all pairwise distances via DGEMM
            #[cfg(feature = "cuda")]
            {
                let ctx = gpu_ctx.unwrap();
                match crate::linalg::gpu::pairwise_distances_gpu(ctx, data, &centroids.view()) {
                    Ok(distances) => {
                        // Assign each point to nearest centroid
                        for i in 0..n_samples {
                            let mut min_dist = f64::INFINITY;
                            let mut min_idx = 0;
                            for j in 0..k {
                                if distances[[i, j]] < min_dist {
                                    min_dist = distances[[i, j]];
                                    min_idx = j;
                                }
                            }
                            labels[i] = min_idx;
                        }
                    }
                    Err(_) => {
                        // Fall back to CPU assignment
                        assign_labels_cpu(data, &centroids, &mut labels);
                    }
                }
            }
        } else {
            // CPU path: point-by-point distance
            assign_labels_cpu(data, &centroids, &mut labels);
        }

        // Update centroids
        let old_centroids = centroids.clone();
        for j in 0..k {
            let mut sum = Array1::zeros(centroids.ncols());
            let mut count = 0;
            for i in 0..n_samples {
                if labels[i] == j {
                    sum += &data.row(i);
                    count += 1;
                }
            }
            if count > 0 {
                centroids.row_mut(j).assign(&(sum / count as f64));
            }
        }

        // Check convergence
        let mut max_shift: f64 = 0.0;
        for j in 0..k {
            let shift = euclidean_distance_squared(&old_centroids.row(j), &centroids.row(j)).sqrt();
            max_shift = max_shift.max(shift);
        }

        if max_shift < tol {
            break;
        }
    }

    // Compute inertia and cluster sizes
    let mut inertia = 0.0;
    let mut cluster_sizes = vec![0usize; k];
    for i in 0..n_samples {
        let dist = euclidean_distance_squared(&data.row(i), &centroids.row(labels[i]));
        inertia += dist;
        cluster_sizes[labels[i]] += 1;
    }

    KMeansResult {
        labels,
        centroids,
        n_iterations,
        inertia,
        cluster_sizes,
    }
}

/// CPU path for K-means label assignment.
fn assign_labels_cpu(data: &ArrayView2<f64>, centroids: &Array2<f64>, labels: &mut [usize]) {
    let n_samples = data.nrows();
    let k = centroids.nrows();
    for i in 0..n_samples {
        let point = data.row(i);
        let mut min_dist = f64::INFINITY;
        let mut min_idx = 0;
        for j in 0..k {
            let dist = euclidean_distance_squared(&point, &centroids.row(j));
            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }
        labels[i] = min_idx;
    }
}

/// Maximum dimensionality for KDTree-based DBSCAN.
/// Above this threshold, the curse of dimensionality makes KDTree inefficient.
const KDTREE_MAX_DIMS: usize = 20;

/// Threshold below which we use the small-n optimized path (condensed distance matrix,
/// no KD-tree, no rayon) to avoid overhead that dominates for small datasets.
const DBSCAN_SMALL_N_THRESHOLD: usize = 500;

/// Run DBSCAN clustering.
///
/// Uses three strategies depending on data size and dimensionality:
/// - **n < 500**: O(n^2) with pre-computed condensed distance matrix (no KD-tree, no rayon).
///   For small datasets the overhead of KD-tree construction and thread pool startup
///   exceeds the cost of a brute-force pairwise distance scan.
/// - **n >= 500, d <= 20**: KD-tree acceleration with parallel neighborhood queries via rayon.
/// - **d > 20**: O(n^2) naive with parallel pairwise distance computation.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `eps` - Maximum distance between two samples for neighborhood
/// * `min_samples` - Minimum samples in neighborhood for core point
pub fn dbscan(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Small datasets: use condensed distance matrix, no KD-tree, no rayon
    if n_samples < DBSCAN_SMALL_N_THRESHOLD {
        return dbscan_small(data, eps, min_samples);
    }

    // Large datasets: use KD-tree for low-dimensional data, naive for high-dimensional
    if n_features <= KDTREE_MAX_DIMS {
        dbscan_kdtree(data, eps, min_samples)
    } else {
        dbscan_naive(data, eps, min_samples)
    }
}

/// Run DBSCAN using a pre-computed condensed distance matrix.
///
/// Optimized for small datasets (n < 500) where KD-tree construction overhead
/// and rayon thread pool startup dominate the actual computation. Pre-computes
/// all pairwise distances into a flat condensed matrix (same layout as the
/// hierarchical clustering code), then scans the array for each neighborhood query.
///
/// Uses the condensed index formula: `idx = i * (2*n - i - 1) / 2 + j - i - 1`
/// for i < j, which is equivalent to the `condensed_index` helper used by
/// hierarchical clustering.
fn dbscan_small(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    let n = data.nrows();

    if eps <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "eps must be positive".to_string(),
        });
    }
    if min_samples == 0 {
        return Err(EconError::InvalidSpecification {
            message: "min_samples must be at least 1".to_string(),
        });
    }

    if n == 0 {
        return Ok(DBSCANResult {
            labels: Vec::new(),
            n_clusters: 0,
            n_noise: 0,
            core_sample_indices: Vec::new(),
        });
    }

    // Single point: it is a core point only if min_samples <= 1
    if n == 1 {
        let is_core = min_samples <= 1;
        return Ok(DBSCANResult {
            labels: vec![if is_core { 0 } else { -1 }],
            n_clusters: if is_core { 1 } else { 0 },
            n_noise: if is_core { 0 } else { 1 },
            core_sample_indices: if is_core { vec![0] } else { Vec::new() },
        });
    }

    let eps_sq = eps * eps;

    // Pre-compute condensed distance matrix (squared Euclidean distances).
    // Layout: for i < j, index = i * (2*n - i - 1) / 2 + j - i - 1
    let condensed_len = n * (n - 1) / 2;
    let mut dist_sq = vec![0.0f64; condensed_len];
    for i in 0..n {
        let row_i = data.row(i);
        // Use the condensed_index formula inline for i < j
        let base = i * (2 * n - i - 1) / 2;
        for j in (i + 1)..n {
            let idx = base + j - i - 1;
            dist_sq[idx] = euclidean_distance_squared(&row_i, &data.row(j));
        }
    }

    // Count neighbors for each point to identify core points.
    // neighbor_count[i] = number of points within eps of point i (including itself).
    let mut neighbor_count = vec![1usize; n]; // each point is its own neighbor
    for i in 0..n {
        let base = i * (2 * n - i - 1) / 2;
        for j in (i + 1)..n {
            let idx = base + j - i - 1;
            if dist_sq[idx] <= eps_sq {
                neighbor_count[i] += 1;
                neighbor_count[j] += 1;
            }
        }
    }

    // Identify core points using a flat bool array (no HashSet overhead)
    let mut is_core = vec![false; n];
    let mut core_sample_indices = Vec::new();
    for i in 0..n {
        if neighbor_count[i] >= min_samples {
            is_core[i] = true;
            core_sample_indices.push(i);
        }
    }

    // Initialize labels (-1 = unvisited/noise)
    let mut labels = vec![-1i32; n];
    let mut current_cluster = 0i32;

    // DFS expansion from core points.
    // We re-scan the condensed matrix for neighbors during expansion rather than
    // storing full neighbor lists, trading a small amount of redundant scanning
    // for lower memory usage.
    let mut stack: Vec<usize> = Vec::with_capacity(n);

    for &core_idx in &core_sample_indices {
        if labels[core_idx] != -1 {
            continue;
        }

        // Start new cluster
        labels[core_idx] = current_cluster;
        stack.clear();
        stack.push(core_idx);

        while let Some(idx) = stack.pop() {
            // Find neighbors of idx by scanning the condensed matrix.
            // For each j != idx, check if dist_sq(idx, j) <= eps_sq.
            //
            // We split into two ranges to use the condensed index formula:
            //   - j < idx: condensed index = j * (2*n - j - 1) / 2 + idx - j - 1
            //   - j > idx: condensed index = idx * (2*n - idx - 1) / 2 + j - idx - 1

            // Range j < idx
            for j in 0..idx {
                let cidx = j * (2 * n - j - 1) / 2 + idx - j - 1;
                if dist_sq[cidx] <= eps_sq && labels[j] == -1 {
                    labels[j] = current_cluster;
                    if is_core[j] {
                        stack.push(j);
                    }
                }
            }

            // Range j > idx
            let base = idx * (2 * n - idx - 1) / 2;
            for j in (idx + 1)..n {
                let cidx = base + j - idx - 1;
                if dist_sq[cidx] <= eps_sq && labels[j] == -1 {
                    labels[j] = current_cluster;
                    if is_core[j] {
                        stack.push(j);
                    }
                }
            }
        }

        current_cluster += 1;
    }

    let n_clusters = current_cluster as usize;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    Ok(DBSCANResult {
        labels,
        n_clusters,
        n_noise,
        core_sample_indices,
    })
}

/// Run DBSCAN using KD-tree for efficient neighborhood queries.
///
/// Achieves O(n log n) average case for low-dimensional data.
fn dbscan_kdtree(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    use super::kdtree::KdTree;

    let n_samples = data.nrows();

    if eps <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "eps must be positive".to_string(),
        });
    }
    if min_samples == 0 {
        return Err(EconError::InvalidSpecification {
            message: "min_samples must be at least 1".to_string(),
        });
    }

    if n_samples == 0 {
        return Ok(DBSCANResult {
            labels: Vec::new(),
            n_clusters: 0,
            n_noise: 0,
            core_sample_indices: Vec::new(),
        });
    }

    // Build KD-tree from data
    let data_vec: Vec<Vec<f64>> = data.rows().into_iter().map(|row| row.to_vec()).collect();
    let tree = KdTree::new(data_vec);

    // Find neighbors for each point using KD-tree radius queries (parallel, unsorted)
    let neighborhoods: Vec<Vec<usize>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            tree.radius_query_unsorted(tree.data().get(i).unwrap(), eps, None)
                .into_iter()
                .map(|(_, idx)| idx)
                .collect()
        })
        .collect();

    // Identify core points
    let core_sample_indices: Vec<usize> = (0..n_samples)
        .filter(|&i| neighborhoods[i].len() >= min_samples)
        .collect();
    let core_set: HashSet<usize> = core_sample_indices.iter().cloned().collect();

    // Initialize labels (-1 = unvisited/noise)
    let mut labels = vec![-1i32; n_samples];
    let mut current_cluster = 0i32;

    // Expand clusters from core points
    for &core_idx in &core_sample_indices {
        if labels[core_idx] != -1 {
            continue;
        }

        // Start new cluster
        labels[core_idx] = current_cluster;
        let mut stack = vec![core_idx];

        while let Some(idx) = stack.pop() {
            for &neighbor in &neighborhoods[idx] {
                if labels[neighbor] == -1 {
                    labels[neighbor] = current_cluster;

                    // If neighbor is a core point, expand from it
                    if core_set.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        current_cluster += 1;
    }

    let n_clusters = current_cluster as usize;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    Ok(DBSCANResult {
        labels,
        n_clusters,
        n_noise,
        core_sample_indices,
    })
}

/// Run DBSCAN using naive O(n²) pairwise distance computation.
///
/// Used for high-dimensional data where KD-tree is inefficient.
fn dbscan_naive(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    let n_samples = data.nrows();

    if eps <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "eps must be positive".to_string(),
        });
    }
    if min_samples == 0 {
        return Err(EconError::InvalidSpecification {
            message: "min_samples must be at least 1".to_string(),
        });
    }

    let eps_squared = eps * eps;

    // Find neighbors for each point (parallel)
    let neighborhoods: Vec<Vec<usize>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row_i = data.row(i);
            (0..n_samples)
                .filter(|&j| euclidean_distance_squared(&row_i, &data.row(j)) <= eps_squared)
                .collect()
        })
        .collect();

    // Identify core points
    let core_sample_indices: Vec<usize> = (0..n_samples)
        .filter(|&i| neighborhoods[i].len() >= min_samples)
        .collect();
    let core_set: HashSet<usize> = core_sample_indices.iter().cloned().collect();

    // Initialize labels (-1 = unvisited/noise)
    let mut labels = vec![-1i32; n_samples];
    let mut current_cluster = 0i32;

    // Expand clusters from core points
    for &core_idx in &core_sample_indices {
        if labels[core_idx] != -1 {
            continue;
        }

        // Start new cluster
        labels[core_idx] = current_cluster;
        let mut stack = vec![core_idx];

        while let Some(idx) = stack.pop() {
            for &neighbor in &neighborhoods[idx] {
                if labels[neighbor] == -1 {
                    labels[neighbor] = current_cluster;

                    // If neighbor is a core point, expand from it
                    if core_set.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        current_cluster += 1;
    }

    let n_clusters = current_cluster as usize;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    Ok(DBSCANResult {
        labels,
        n_clusters,
        n_noise,
        core_sample_indices,
    })
}

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    /// Minimum distance between clusters (nearest neighbor)
    Single,
    /// Maximum distance between clusters (furthest neighbor)
    Complete,
    /// Average distance between all pairs
    Average,
    /// Ward's minimum variance method
    Ward,
}

impl std::str::FromStr for Linkage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "single" => Ok(Linkage::Single),
            "complete" => Ok(Linkage::Complete),
            "average" => Ok(Linkage::Average),
            "ward" => Ok(Linkage::Ward),
            _ => Err(format!(
                "Unknown linkage method: {}. Use single, complete, average, or ward",
                s
            )),
        }
    }
}

/// Hierarchical clustering result.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Cluster assignments for each point (0 to n_clusters-1)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Linkage matrix: (cluster1, cluster2, distance, size)
    /// Each row represents a merge step
    pub linkage_matrix: Vec<(usize, usize, f64, usize)>,
    /// Merge distances in order
    pub merge_distances: Vec<f64>,
    /// Linkage method used
    pub linkage: String,
}

impl std::fmt::Display for HierarchicalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Hierarchical Clustering Results")?;
        writeln!(f, "================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Linkage method: {}", self.linkage)?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().collect();
        labels_sorted.sort();
        for &label in &labels_sorted {
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[label])?;
        }

        writeln!(f)?;
        writeln!(f, "Dendrogram (merge history):")?;
        writeln!(f, "  Step  Cluster1  Cluster2  Distance    Size")?;
        for (i, &(c1, c2, dist, size)) in self.linkage_matrix.iter().enumerate() {
            writeln!(
                f,
                "  {:4}  {:8}  {:8}  {:10.4}  {:4}",
                i + 1,
                c1,
                c2,
                dist,
                size
            )?;
        }

        Ok(())
    }
}

/// Run hierarchical agglomerative clustering.
///
/// Uses an optimized nearest-neighbor algorithm with condensed distance matrix
/// and Lance-Williams recurrence for O(n^2) time and O(n^2) memory.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (if None, returns full dendrogram)
/// * `linkage` - Linkage method (single, complete, average, ward)
/// * `distance_threshold` - If set, cut tree at this distance instead of n_clusters
///
/// # References
///
/// - Murtagh, F. (1983). "A Survey of Recent Advances in Hierarchical Clustering
///   Algorithms". The Computer Journal, 26(4), 354-359.
/// - Lance, G. N. & Williams, W. T. (1967). "A General Theory of Classificatory
///   Sorting Strategies: 1. Hierarchical Systems". The Computer Journal, 9(4), 373-380.
/// - Mullner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms".
///   arXiv:1109.2378.
pub fn hierarchical(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    linkage: Linkage,
    distance_threshold: Option<f64>,
) -> Result<HierarchicalResult, String> {
    let n = data.nrows();

    if n == 0 {
        return Err("Cannot cluster empty data".to_string());
    }
    if n == 1 {
        return Ok(HierarchicalResult {
            labels: vec![0],
            n_clusters: 1,
            linkage_matrix: vec![],
            merge_distances: vec![],
            linkage: format!("{:?}", linkage).to_lowercase(),
        });
    }

    let target_clusters = match (n_clusters, distance_threshold) {
        (Some(nc), _) => {
            if nc == 0 || nc > n {
                return Err(format!(
                    "n_clusters must be between 1 and {} (n_samples)",
                    n
                ));
            }
            Some(nc)
        }
        (None, None) => Some(1),
        (None, Some(_)) => None,
    };

    // --- Condensed distance matrix ---
    // Store pairwise distances in a flat Vec using the condensed index formula.
    // For Ward's method we store squared Euclidean distances; for others, Euclidean distances.
    let use_squared = linkage == Linkage::Ward;
    let condensed_len = n * (n - 1) / 2;
    let mut dist = vec![0.0f64; condensed_len];
    for i in 0..n {
        let row_i = data.row(i);
        for j in (i + 1)..n {
            let d = euclidean_distance_squared(&row_i, &data.row(j));
            let idx = condensed_index(n, i, j);
            dist[idx] = if use_squared { d } else { d.sqrt() };
        }
    }

    // --- Cluster sizes ---
    // size[i] = number of original observations in cluster i.
    // Indices 0..n are original observations (size 1).
    // We use a Vec indexed by cluster label; new clusters get appended.
    let mut size = vec![1usize; n];

    // --- Nearest neighbor tracking ---
    // nn[i] = index of nearest active neighbor of cluster i (among active clusters).
    // nn_dist[i] = distance to that neighbor.
    // We also track which clusters are active.
    let mut active = vec![true; n];
    let mut nn = vec![0usize; n];
    let mut nn_dist = vec![f64::INFINITY; n];

    // Initialize nearest neighbors for each cluster.
    for i in 0..n {
        for j in (i + 1)..n {
            let d = ward_or_raw_dist(&dist, n, i, j, &size, linkage);
            if d < nn_dist[i] {
                nn_dist[i] = d;
                nn[i] = j;
            }
            if d < nn_dist[j] {
                nn_dist[j] = d;
                nn[j] = i;
            }
        }
    }

    // --- Linkage matrix storage ---
    let mut linkage_matrix: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n - 1);
    let mut merge_distances: Vec<f64> = Vec::with_capacity(n - 1);

    // Mapping: original cluster labels (0..n, n, n+1, ...) -> internal labels.
    // Internal labels grow as new clusters form; we keep track via `label_map`.
    // label_map[internal] = external label used in the linkage_matrix output.
    let mut label_map: Vec<usize> = (0..n).collect();
    let mut next_external_id = n;

    // --- Union-find for final label assignment ---
    let mut uf_parent: Vec<usize> = (0..n).collect();
    let mut uf_rank: Vec<usize> = vec![0; n];

    let mut n_active = n;

    // --- Main agglomerative loop ---
    for _step in 0..(n - 1) {
        if let Some(target) = target_clusters {
            if n_active <= target {
                break;
            }
        }

        // Find the global minimum distance pair using the nearest-neighbor cache.
        // Scan all active clusters to find the one with smallest nn_dist.
        let mut min_d = f64::INFINITY;
        let mut c_i = 0;
        for i in 0..active.len() {
            if active[i] && nn_dist[i] < min_d {
                min_d = nn_dist[i];
                c_i = i;
            }
        }
        let mut c_j = nn[c_i];

        // Validate that nn[c_i] is still active; if not, rescan.
        if !active[c_j] {
            // Recompute nn for c_i
            nn_dist[c_i] = f64::INFINITY;
            for k in 0..active.len() {
                if k != c_i && active[k] {
                    let d = ward_or_raw_dist(&dist, n, c_i, k, &size, linkage);
                    if d < nn_dist[c_i] {
                        nn_dist[c_i] = d;
                        nn[c_i] = k;
                    }
                }
            }
            // Restart search for global minimum -- this invalidated our choice.
            // Re-scan for global minimum.
            min_d = f64::INFINITY;
            for i in 0..active.len() {
                if active[i] && nn_dist[i] < min_d {
                    min_d = nn_dist[i];
                    c_i = i;
                }
            }
            c_j = nn[c_i];
            // If still invalid, do a full nn rebuild (rare edge case)
            if !active[c_j] {
                rebuild_nn_full(&dist, n, &active, &size, linkage, &mut nn, &mut nn_dist);
                min_d = f64::INFINITY;
                for i in 0..active.len() {
                    if active[i] && nn_dist[i] < min_d {
                        min_d = nn_dist[i];
                        c_i = i;
                    }
                }
                c_j = nn[c_i];
            }
        }

        // Ensure c_i < c_j for canonical ordering in output.
        if c_i > c_j {
            std::mem::swap(&mut c_i, &mut c_j);
        }

        // Compute the reported merge distance (for display / cutree).
        // For Ward, convert squared distance to the Ward distance metric:
        //   ward_dist = sqrt(2 * n_i * n_j / (n_i + n_j) * d_sq_euclidean(centroid_i, centroid_j))
        // The condensed matrix stores squared Euclidean centroid distances already embedded in
        // the Lance-Williams updates; the stored value equals (n_i + n_j) / (n_i * n_j) * ward_dist^2 / 2.
        // Actually for Ward we store the quantity that the Lance-Williams formula preserves, which is
        // the squared Euclidean distance between centroids scaled appropriately.
        // The conventional Ward merge height is sqrt(2*n_i*n_j/(n_i+n_j)) * ||c_i - c_j||.
        let merge_dist = if use_squared {
            let raw = get_condensed(&dist, n, c_i, c_j);
            let ni = size[c_i] as f64;
            let nj = size[c_j] as f64;
            // raw = squared Euclidean distance between centroids
            // Ward merge distance = sqrt(2 * ni * nj / (ni + nj)) * sqrt(raw)
            ((2.0 * ni * nj / (ni + nj)) * raw).sqrt()
        } else {
            get_condensed(&dist, n, c_i, c_j)
        };

        // Check distance threshold
        if let Some(thresh) = distance_threshold {
            if merge_dist > thresh {
                break;
            }
        }

        let new_size = size[c_i] + size[c_j];

        // Record merge in linkage matrix using external labels
        linkage_matrix.push((label_map[c_i], label_map[c_j], merge_dist, new_size));
        merge_distances.push(merge_dist);

        // --- Lance-Williams distance update ---
        // Update dist[c_i, k] for all active k != c_i, c_j using the recurrence.
        // After update, c_i becomes the merged cluster; c_j is deactivated.
        let ni = size[c_i] as f64;
        let nj = size[c_j] as f64;
        let d_ij = get_condensed(&dist, n, c_i, c_j);

        for k in 0..active.len() {
            if !active[k] || k == c_i || k == c_j {
                continue;
            }
            let nk = size[k] as f64;
            let d_ki = get_condensed(&dist, n, k, c_i);
            let d_kj = get_condensed(&dist, n, k, c_j);

            // Lance-Williams recurrence (Lance & Williams, 1967)
            let new_d = match linkage {
                Linkage::Single => {
                    // d(k, i+j) = min(d(k,i), d(k,j))
                    d_ki.min(d_kj)
                }
                Linkage::Complete => {
                    // d(k, i+j) = max(d(k,i), d(k,j))
                    d_ki.max(d_kj)
                }
                Linkage::Average => {
                    // d(k, i+j) = (n_i * d(k,i) + n_j * d(k,j)) / (n_i + n_j)
                    (ni * d_ki + nj * d_kj) / (ni + nj)
                }
                Linkage::Ward => {
                    // Lance-Williams for Ward (on squared Euclidean distances):
                    // d(k, i+j) = ((n_i+n_k)*d(k,i) + (n_j+n_k)*d(k,j) - n_k*d(i,j)) / (n_i+n_j+n_k)
                    ((ni + nk) * d_ki + (nj + nk) * d_kj - nk * d_ij) / (ni + nj + nk)
                }
            };

            set_condensed(&mut dist, n, c_i, k, new_d);
        }

        // Update cluster size and label
        size[c_i] = new_size;
        label_map[c_i] = next_external_id;
        next_external_id += 1;

        // Deactivate c_j
        active[c_j] = false;
        nn_dist[c_j] = f64::INFINITY;
        n_active -= 1;

        // Union-find: merge original observations.
        // We track which original observations belong to c_i and c_j via the union-find.
        // For the union-find, we just need a representative from each.
        // c_i and c_j are internal indices (0..n), so we use them directly.
        uf_union(&mut uf_parent, &mut uf_rank, c_i, c_j);

        // --- Update nearest neighbors ---
        // For the merged cluster c_i, recompute its nearest neighbor from scratch.
        nn_dist[c_i] = f64::INFINITY;
        for k in 0..active.len() {
            if k != c_i && active[k] {
                let d = ward_or_raw_dist(&dist, n, c_i, k, &size, linkage);
                if d < nn_dist[c_i] {
                    nn_dist[c_i] = d;
                    nn[c_i] = k;
                }
                // Update k's nearest neighbor:
                // - If k's NN was c_j (now deactivated) or c_i (distances changed),
                //   we must do a full rescan for k.
                // - Otherwise, just check if the merged cluster c_i is now closer.
                if nn[k] == c_j || nn[k] == c_i {
                    // NN invalidated — full rescan required
                    nn_dist[k] = f64::INFINITY;
                    for m in 0..active.len() {
                        if m != k && active[m] {
                            let dm = ward_or_raw_dist(&dist, n, k, m, &size, linkage);
                            if dm < nn_dist[k] {
                                nn_dist[k] = dm;
                                nn[k] = m;
                            }
                        }
                    }
                } else if d < nn_dist[k] {
                    // Merged cluster is closer than current NN — just update
                    nn_dist[k] = d;
                    nn[k] = c_i;
                }
            }
        }
    }

    // --- Assign final labels using union-find ---
    let mut labels = vec![0usize; n];
    // Find roots
    for i in 0..n {
        labels[i] = uf_find(&mut uf_parent, i);
    }
    // Map roots to consecutive 0-based labels
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0;
    for i in 0..n {
        let root = labels[i];
        if !root_to_label.contains_key(&root) {
            root_to_label.insert(root, next_label);
            next_label += 1;
        }
        labels[i] = root_to_label[&root];
    }

    Ok(HierarchicalResult {
        labels,
        n_clusters: root_to_label.len(),
        linkage_matrix,
        merge_distances,
        linkage: format!("{:?}", linkage).to_lowercase(),
    })
}

// =============================================================================
// Condensed distance matrix helpers
// =============================================================================

/// Compute condensed index for pair (i, j) where i < j in an n-element matrix.
/// Formula: idx = n*(n-1)/2 - (n-i)*(n-i-1)/2 + j - i - 1
#[inline(always)]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    let ni = n - i;
    n * (n - 1) / 2 - ni * (ni - 1) / 2 + j - i - 1
}

/// Get distance from condensed matrix, handling i == j and i > j.
#[inline(always)]
fn get_condensed(dist: &[f64], n: usize, i: usize, j: usize) -> f64 {
    if i == j {
        0.0
    } else if i < j {
        dist[condensed_index(n, i, j)]
    } else {
        dist[condensed_index(n, j, i)]
    }
}

/// Set distance in condensed matrix, handling i > j.
#[inline(always)]
fn set_condensed(dist: &mut [f64], n: usize, i: usize, j: usize, val: f64) {
    if i < j {
        dist[condensed_index(n, i, j)] = val;
    } else if j < i {
        dist[condensed_index(n, j, i)] = val;
    }
}

/// Get the "effective" distance for nearest-neighbor comparison.
/// For Ward's method (squared distances), we convert to the Ward merge metric
/// for comparison: sqrt(2*ni*nj/(ni+nj) * d_sq).
/// For other linkages, the raw stored value is the distance.
#[inline]
fn ward_or_raw_dist(
    dist: &[f64],
    n: usize,
    i: usize,
    j: usize,
    size: &[usize],
    linkage: Linkage,
) -> f64 {
    let raw = get_condensed(dist, n, i, j);
    if linkage == Linkage::Ward {
        let ni = size[i] as f64;
        let nj = size[j] as f64;
        // Ward merge distance = sqrt(2 * ni * nj / (ni + nj) * raw)
        // where raw = squared Euclidean distance between centroids
        ((2.0 * ni * nj / (ni + nj)) * raw).sqrt()
    } else {
        raw
    }
}

/// Full rebuild of nearest-neighbor arrays (fallback for edge cases).
fn rebuild_nn_full(
    dist: &[f64],
    n: usize,
    active: &[bool],
    size: &[usize],
    linkage: Linkage,
    nn: &mut [usize],
    nn_dist: &mut [f64],
) {
    for i in 0..active.len() {
        if !active[i] {
            nn_dist[i] = f64::INFINITY;
            continue;
        }
        nn_dist[i] = f64::INFINITY;
        for j in 0..active.len() {
            if j != i && active[j] {
                let d = ward_or_raw_dist(dist, n, i, j, size, linkage);
                if d < nn_dist[i] {
                    nn_dist[i] = d;
                    nn[i] = j;
                }
            }
        }
    }
}

// =============================================================================
// Union-find helpers
// =============================================================================

/// Find with path compression.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path splitting
        x = parent[x];
    }
    x
}

/// Union by rank.
fn uf_union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
    let px = uf_find(parent, x);
    let py = uf_find(parent, y);
    if px == py {
        return;
    }
    if rank[px] < rank[py] {
        parent[px] = py;
    } else if rank[px] > rank[py] {
        parent[py] = px;
    } else {
        parent[py] = px;
        rank[px] += 1;
    }
}

/// Squared Euclidean distance between two vectors.
#[inline]
fn euclidean_distance_squared(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

// =============================================================================
// cutree - Cut hierarchical clustering tree
// =============================================================================

use serde::{Deserialize, Serialize};

/// Result of cutting a hierarchical clustering tree.
///
/// # References
///
/// - R stats::cutree documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cutree.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutreeResult {
    /// Cluster assignments for each observation (1-indexed, like R).
    /// If multiple k or h values were provided, this is for the first k/h.
    pub labels: Vec<usize>,
    /// Number of clusters formed
    pub k: usize,
    /// The height at which the tree was cut (if h was used)
    pub cut_height: Option<f64>,
    /// Number of observations
    pub n: usize,
    /// If multiple k values were given, this contains all assignments (one column per k)
    pub labels_matrix: Option<Vec<Vec<usize>>>,
    /// The k values used (if multiple)
    pub k_values: Option<Vec<usize>>,
}

impl std::fmt::Display for CutreeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "cutree Results")?;
        writeln!(f, "==============")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Number of clusters: {}", self.k)?;
        if let Some(h) = self.cut_height {
            writeln!(f, "Cut height: {:.6}", h)?;
        }
        writeln!(f)?;

        // Count cluster sizes
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().cloned().collect();
        labels_sorted.sort();
        for label in labels_sorted {
            writeln!(
                f,
                "  Cluster {}: {} observations",
                label, cluster_counts[&label]
            )?;
        }

        if let Some(ref k_vals) = self.k_values {
            writeln!(f)?;
            writeln!(f, "Multiple k values requested: {:?}", k_vals)?;
        }

        Ok(())
    }
}

/// Cut a hierarchical clustering tree into groups.
///
/// Given a `HierarchicalResult` from `hierarchical()`, this function divides
/// the tree by specifying the desired number of groups (`k`) or the cut height (`h`).
///
/// # Arguments
/// * `hclust` - A hierarchical clustering result from `hierarchical()`
/// * `k` - Number of groups to form (mutually exclusive with `h` unless both given, then `k` takes priority)
/// * `h` - Height at which to cut the tree
///
/// # Returns
/// * `CutreeResult` containing cluster assignments
///
/// # Algorithm
///
/// The function traverses the linkage matrix (merge history) to reconstruct
/// which observations belong to which cluster at the desired cut level.
///
/// # References
///
/// - R stats::cutree documentation
pub fn cutree(
    hclust: &HierarchicalResult,
    k: Option<usize>,
    h: Option<f64>,
) -> Result<CutreeResult, String> {
    if k.is_none() && h.is_none() {
        return Err("Must specify either k (number of clusters) or h (cut height)".to_string());
    }

    // Get number of original observations
    let n = hclust.labels.len();

    if n == 0 {
        return Err("Empty clustering result".to_string());
    }

    // If k is specified, use it; otherwise use h to determine k
    let target_k = if let Some(k_val) = k {
        if k_val == 0 || k_val > n {
            return Err(format!(
                "k must be between 1 and {} (number of observations)",
                n
            ));
        }
        k_val
    } else if let Some(h_val) = h {
        // Determine k from height: count how many merges happen below height h
        // n - (number of merges at or below h) = k
        let merges_below_h = hclust
            .merge_distances
            .iter()
            .filter(|&&d| d <= h_val)
            .count();
        let computed_k = n - merges_below_h;
        if computed_k == 0 {
            n // All merges above h means n clusters
        } else {
            computed_k
        }
    } else {
        unreachable!()
    };

    // Cut the tree to get target_k clusters
    let labels = cut_at_k(hclust, target_k, n)?;

    // Determine cut height if cutting by k
    let cut_height = if k.is_some() && !hclust.merge_distances.is_empty() {
        // The cut height is the distance at which we would have target_k clusters
        // This is the (n - target_k)th merge distance (0-indexed)
        let merge_idx = n.saturating_sub(target_k);
        if merge_idx > 0 && merge_idx <= hclust.merge_distances.len() {
            Some(hclust.merge_distances[merge_idx - 1])
        } else {
            None
        }
    } else {
        h
    };

    Ok(CutreeResult {
        labels,
        k: target_k,
        cut_height,
        n,
        labels_matrix: None,
        k_values: None,
    })
}

/// Cut a hierarchical clustering tree at multiple k values.
///
/// # Arguments
/// * `hclust` - A hierarchical clustering result
/// * `k_values` - Vector of k values to cut at
///
/// # Returns
/// * `CutreeResult` with labels_matrix containing assignments for each k
pub fn cutree_multiple_k(
    hclust: &HierarchicalResult,
    k_values: &[usize],
) -> Result<CutreeResult, String> {
    if k_values.is_empty() {
        return Err("k_values cannot be empty".to_string());
    }

    let n = hclust.labels.len();
    let mut labels_matrix: Vec<Vec<usize>> = Vec::with_capacity(k_values.len());

    for &k in k_values {
        if k == 0 || k > n {
            return Err(format!(
                "k must be between 1 and {} (number of observations)",
                n
            ));
        }
        let labels = cut_at_k(hclust, k, n)?;
        labels_matrix.push(labels);
    }

    // Use first k's labels as the primary result
    let first_labels = labels_matrix[0].clone();
    let first_k = k_values[0];

    Ok(CutreeResult {
        labels: first_labels,
        k: first_k,
        cut_height: None,
        n,
        labels_matrix: Some(labels_matrix),
        k_values: Some(k_values.to_vec()),
    })
}

/// Internal function to cut tree at a specific k.
fn cut_at_k(hclust: &HierarchicalResult, target_k: usize, n: usize) -> Result<Vec<usize>, String> {
    if target_k == n {
        // Each point is its own cluster
        return Ok((1..=n).collect());
    }

    if target_k == 1 {
        // All points in one cluster
        return Ok(vec![1; n]);
    }

    // Build union-find structure by replaying merges up to the point
    // where we have target_k clusters
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py {
            return;
        }
        if rank[px] < rank[py] {
            parent[px] = py;
        } else if rank[px] > rank[py] {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px] += 1;
        }
    }

    // We need to map cluster IDs in linkage_matrix back to original observations
    // The linkage matrix uses IDs: 0..n are original observations,
    // n, n+1, ... are newly formed clusters from merges

    // Map: cluster_id -> set of original observation indices
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        cluster_members.insert(i, vec![i]);
    }

    // Perform merges until we have target_k clusters
    let num_merges_needed = n - target_k;

    for (merge_idx, &(c1, c2, _dist, _size)) in hclust.linkage_matrix.iter().enumerate() {
        if merge_idx >= num_merges_needed {
            break;
        }

        // Get members of c1 and c2
        let members1 = cluster_members.remove(&c1).unwrap_or_default();
        let members2 = cluster_members.remove(&c2).unwrap_or_default();

        // Union in union-find
        if !members1.is_empty() && !members2.is_empty() {
            union(&mut parent, &mut rank, members1[0], members2[0]);
            // Union all members
            for &m in &members1[1..] {
                union(&mut parent, &mut rank, members1[0], m);
            }
            for &m in &members2 {
                union(&mut parent, &mut rank, members1[0], m);
            }
        }

        // New cluster ID
        let new_id = n + merge_idx;
        let mut new_members = members1;
        new_members.extend(members2);
        cluster_members.insert(new_id, new_members);
    }

    // Extract final cluster assignments
    // Find root for each observation
    let mut labels = vec![0usize; n];
    for i in 0..n {
        labels[i] = find(&mut parent, i);
    }

    // Renumber clusters to be 1-indexed consecutive integers
    let unique_roots: HashSet<usize> = labels.iter().cloned().collect();
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    for (idx, &root) in unique_roots.iter().enumerate() {
        root_to_label.insert(root, idx + 1); // 1-indexed
    }

    for label in &mut labels {
        *label = root_to_label[label];
    }

    Ok(labels)
}

/// Convenience function to run cutree.
pub fn run_cutree(
    hclust: &HierarchicalResult,
    k: Option<usize>,
    h: Option<f64>,
) -> Result<CutreeResult, String> {
    cutree(hclust, k, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmeans_basic() {
        // Simple 2D data with 2 clear clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = kmeans(data.view(), 2, Some(100), Some(1e-4), Some(5), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.centroids.nrows(), 2);

        // Points should be in different clusters
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_dbscan_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [100.0, 100.0], // Noise point
        ];

        let result = dbscan(data.view(), 0.5, 2).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_noise, 1);
        assert_eq!(result.labels[5], -1); // Noise
    }

    #[test]
    fn test_dbscan_kdtree_larger() {
        // Create a larger dataset to test KDTree-based DBSCAN
        // Two clusters of 50 points each, plus 5 noise points
        let mut data_vec = Vec::new();

        // Cluster 1 around (0, 0)
        for i in 0..50 {
            let x = (i as f64) * 0.01;
            let y = (i as f64) * 0.01;
            data_vec.push([x, y]);
        }

        // Cluster 2 around (10, 10)
        for i in 0..50 {
            let x = 10.0 + (i as f64) * 0.01;
            let y = 10.0 + (i as f64) * 0.01;
            data_vec.push([x, y]);
        }

        // Noise points
        for i in 0..5 {
            data_vec.push([50.0 + i as f64 * 10.0, 50.0 + i as f64 * 10.0]);
        }

        let data =
            Array2::from_shape_vec((105, 2), data_vec.into_iter().flatten().collect()).unwrap();

        let result = dbscan(data.view(), 0.5, 3).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_noise, 5);

        // Verify cluster assignments
        // First 50 points should be in one cluster
        let first_cluster = result.labels[0];
        for i in 0..50 {
            assert_eq!(result.labels[i], first_cluster);
        }

        // Next 50 points should be in another cluster
        let second_cluster = result.labels[50];
        assert_ne!(first_cluster, second_cluster);
        for i in 50..100 {
            assert_eq!(result.labels[i], second_cluster);
        }

        // Last 5 points should be noise
        for i in 100..105 {
            assert_eq!(result.labels[i], -1);
        }
    }

    #[test]
    fn test_dbscan_empty() {
        let data: Array2<f64> = Array2::zeros((0, 2));
        let result = dbscan(data.view(), 0.5, 2).unwrap();
        assert_eq!(result.n_clusters, 0);
        assert_eq!(result.n_noise, 0);
    }

    #[test]
    fn test_hierarchical_basic() {
        // Simple 2D data with 2 clear clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = hierarchical(data.view(), Some(2), Linkage::Ward, None).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);

        // Points 0-2 should be in one cluster, points 3-5 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_hierarchical_linkage_methods() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0],];

        // Test all linkage methods work
        for linkage in [
            Linkage::Single,
            Linkage::Complete,
            Linkage::Average,
            Linkage::Ward,
        ] {
            let result = hierarchical(data.view(), Some(2), linkage, None).unwrap();
            assert_eq!(result.n_clusters, 2);
        }
    }

    #[test]
    fn test_hierarchical_distance_threshold() {
        let data = array![[0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [10.5, 0.0],];

        // With threshold of 1.0, should get 2 clusters
        let result = hierarchical(data.view(), None, Linkage::Single, Some(1.0)).unwrap();
        assert_eq!(result.n_clusters, 2);
    }

    // =========================================================================
    // cutree tests
    // =========================================================================

    #[test]
    fn test_cutree_basic() {
        // Create hierarchical clustering result first
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Get full dendrogram (cluster down to 1 cluster)
        let hclust = hierarchical(data.view(), Some(1), Linkage::Ward, None).unwrap();

        // Cut into 2 clusters
        let cut_result = cutree(&hclust, Some(2), None).unwrap();

        assert_eq!(cut_result.n, 6);
        assert_eq!(cut_result.k, 2);
        assert_eq!(cut_result.labels.len(), 6);

        // Points 0-2 should be in same cluster, points 3-5 in another
        assert_eq!(cut_result.labels[0], cut_result.labels[1]);
        assert_eq!(cut_result.labels[1], cut_result.labels[2]);
        assert_eq!(cut_result.labels[3], cut_result.labels[4]);
        assert_eq!(cut_result.labels[4], cut_result.labels[5]);
        assert_ne!(cut_result.labels[0], cut_result.labels[3]);
    }

    #[test]
    fn test_cutree_k_equals_n() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        // Cut into n clusters (each point is its own cluster)
        let cut_result = cutree(&hclust, Some(3), None).unwrap();

        assert_eq!(cut_result.k, 3);
        // All labels should be different
        assert_ne!(cut_result.labels[0], cut_result.labels[1]);
        assert_ne!(cut_result.labels[1], cut_result.labels[2]);
        assert_ne!(cut_result.labels[0], cut_result.labels[2]);
    }

    #[test]
    fn test_cutree_k_equals_1() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        // Cut into 1 cluster (all points together)
        let cut_result = cutree(&hclust, Some(1), None).unwrap();

        assert_eq!(cut_result.k, 1);
        // All labels should be the same
        assert_eq!(cut_result.labels[0], cut_result.labels[1]);
        assert_eq!(cut_result.labels[1], cut_result.labels[2]);
    }

    #[test]
    fn test_cutree_by_height() {
        let data = array![[0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [10.5, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        // Cut at height 1.0 should give 2 clusters (close pairs merge but far pairs don't)
        let cut_result = cutree(&hclust, None, Some(1.0)).unwrap();

        assert_eq!(cut_result.k, 2);
        assert!(cut_result.cut_height.is_some());
    }

    #[test]
    fn test_cutree_multiple_k() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        let cut_result = cutree_multiple_k(&hclust, &[1, 2, 3, 4]).unwrap();

        assert!(cut_result.labels_matrix.is_some());
        let matrix = cut_result.labels_matrix.unwrap();
        assert_eq!(matrix.len(), 4);

        // k=1: all same label
        assert!(matrix[0].iter().all(|&x| x == matrix[0][0]));

        // k=4: all different labels
        let k4_unique: HashSet<_> = matrix[3].iter().collect();
        assert_eq!(k4_unique.len(), 4);
    }

    #[test]
    fn test_cutree_validation() {
        let data = array![[0.0, 0.0], [1.0, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        // k=0 should fail
        let result = cutree(&hclust, Some(0), None);
        assert!(result.is_err());

        // k > n should fail
        let result = cutree(&hclust, Some(10), None);
        assert!(result.is_err());

        // Neither k nor h should fail
        let result = cutree(&hclust, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cutree_display() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0],];

        let hclust = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();
        let cut_result = cutree(&hclust, Some(2), None).unwrap();

        let display = format!("{}", cut_result);
        assert!(display.contains("cutree Results"));
        assert!(display.contains("Number of observations: 3"));
        assert!(display.contains("Number of clusters: 2"));
    }

    // ========================================================================
    // R-vs-Rust Validation Tests (Phase 8)
    // ========================================================================

    fn create_validation_cluster_data() -> Array2<f64> {
        // Create 3 well-separated clusters in 2D
        // Cluster 1: around (0, 0)
        // Cluster 2: around (10, 0)
        // Cluster 3: around (5, 10)
        array![
            // Cluster 1 (around origin)
            [0.0, 0.0],
            [0.5, 0.3],
            [0.2, -0.4],
            [0.8, 0.1],
            [-0.3, 0.5],
            // Cluster 2 (around (10, 0))
            [10.0, 0.0],
            [10.3, 0.2],
            [9.8, -0.1],
            [10.5, 0.4],
            [9.7, 0.3],
            // Cluster 3 (around (5, 10))
            [5.0, 10.0],
            [5.2, 10.3],
            [4.8, 9.8],
            [5.5, 10.1],
            [4.9, 9.6],
        ]
    }

    #[test]
    fn test_validate_kmeans_vs_r() {
        // R reference:
        // set.seed(42)
        // kmeans(data, centers=3)
        let data = create_validation_cluster_data();
        let result = kmeans(data.view(), 3, Some(100), Some(1e-4), Some(10), Some(42)).unwrap();

        // Should find 3 clusters
        assert_eq!(result.centroids.nrows(), 3);
        assert_eq!(result.labels.len(), 15);

        // Each cluster should have 5 points
        for size in &result.cluster_sizes {
            assert_eq!(*size, 5, "Each cluster should have 5 points");
        }

        // Inertia should be small for well-separated clusters
        assert!(
            result.inertia < 10.0,
            "Inertia {} should be small for well-separated clusters",
            result.inertia
        );
    }

    #[test]
    fn test_validate_kmeans_centroids() {
        let data = create_validation_cluster_data();
        let result = kmeans(data.view(), 3, Some(100), Some(1e-4), Some(10), Some(42)).unwrap();

        // Centroids should be near the true centers
        // True centers: (0, 0), (10, 0), (5, 10)
        let mut found_near_origin = false;
        let mut found_near_10_0 = false;
        let mut found_near_5_10 = false;

        for i in 0..3 {
            let cx = result.centroids[[i, 0]];
            let cy = result.centroids[[i, 1]];

            if (cx - 0.0).abs() < 1.5 && (cy - 0.0).abs() < 1.5 {
                found_near_origin = true;
            }
            if (cx - 10.0).abs() < 1.5 && (cy - 0.0).abs() < 1.5 {
                found_near_10_0 = true;
            }
            if (cx - 5.0).abs() < 1.5 && (cy - 10.0).abs() < 1.5 {
                found_near_5_10 = true;
            }
        }

        assert!(found_near_origin, "Should find centroid near (0, 0)");
        assert!(found_near_10_0, "Should find centroid near (10, 0)");
        assert!(found_near_5_10, "Should find centroid near (5, 10)");
    }

    #[test]
    fn test_validate_kmeans_cluster_assignment() {
        let data = create_validation_cluster_data();
        let result = kmeans(data.view(), 3, Some(100), Some(1e-4), Some(10), Some(42)).unwrap();

        // Points 0-4 should be in the same cluster
        let cluster1 = result.labels[0];
        for i in 1..5 {
            assert_eq!(
                result.labels[i], cluster1,
                "Points 0-4 should be in same cluster"
            );
        }

        // Points 5-9 should be in the same cluster
        let cluster2 = result.labels[5];
        for i in 6..10 {
            assert_eq!(
                result.labels[i], cluster2,
                "Points 5-9 should be in same cluster"
            );
        }

        // Points 10-14 should be in the same cluster
        let cluster3 = result.labels[10];
        for i in 11..15 {
            assert_eq!(
                result.labels[i], cluster3,
                "Points 10-14 should be in same cluster"
            );
        }

        // Clusters should be different
        assert_ne!(cluster1, cluster2);
        assert_ne!(cluster2, cluster3);
        assert_ne!(cluster1, cluster3);
    }

    #[test]
    fn test_validate_dbscan_vs_r() {
        // R reference:
        // library(dbscan)
        // dbscan(data, eps=2, minPts=3)
        let data = create_validation_cluster_data();
        let result = dbscan(data.view(), 2.0, 3).unwrap();

        // Should find 3 clusters (no noise with these params)
        assert_eq!(result.n_clusters, 3, "DBSCAN should find 3 clusters");

        // No noise points expected
        assert_eq!(result.n_noise, 0, "Should have no noise points");
    }

    #[test]
    fn test_validate_dbscan_noise_detection() {
        // Add an outlier to test noise detection
        let data = array![
            [0.0, 0.0],
            [0.5, 0.3],
            [0.2, -0.4],
            [10.0, 0.0],
            [10.3, 0.2],
            [9.8, -0.1],
            [50.0, 50.0], // Outlier
        ];

        let result = dbscan(data.view(), 2.0, 3).unwrap();

        // Outlier should be noise
        assert_eq!(result.labels[6], -1, "Outlier should be labeled as noise");
        assert!(
            result.n_noise >= 1,
            "Should detect at least one noise point"
        );
    }

    #[test]
    fn test_validate_hierarchical_single_linkage_vs_r() {
        // R reference:
        // hclust(dist(data), method="single")
        let data = create_validation_cluster_data();
        let result = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();

        // n_clusters is the target, linkage_matrix has n-1 merges
        assert_eq!(
            result.linkage_matrix.len(),
            14,
            "Should have n-1 = 14 merges"
        );
        assert_eq!(result.labels.len(), 15);
        assert_eq!(result.merge_distances.len(), 14);

        // Heights should be increasing for single linkage
        for i in 1..result.merge_distances.len() {
            assert!(
                result.merge_distances[i] >= result.merge_distances[i - 1] - 1e-10,
                "Heights should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_validate_hierarchical_complete_linkage_vs_r() {
        // R reference:
        // hclust(dist(data), method="complete")
        let data = create_validation_cluster_data();
        let result = hierarchical(data.view(), Some(1), Linkage::Complete, None).unwrap();

        // linkage_matrix should have n-1 merges
        assert_eq!(result.linkage_matrix.len(), 14);

        // Heights should be increasing
        for i in 1..result.merge_distances.len() {
            assert!(
                result.merge_distances[i] >= result.merge_distances[i - 1] - 1e-10,
                "Heights should increase in complete linkage"
            );
        }
    }

    #[test]
    fn test_validate_hierarchical_ward_vs_r() {
        // R reference:
        // hclust(dist(data), method="ward.D2")
        let data = create_validation_cluster_data();
        let result = hierarchical(data.view(), Some(1), Linkage::Ward, None).unwrap();

        assert_eq!(result.linkage_matrix.len(), 14);
        assert_eq!(result.linkage, "ward");

        // Heights should increase
        for i in 1..result.merge_distances.len() {
            assert!(
                result.merge_distances[i] >= result.merge_distances[i - 1] - 1e-10,
                "Ward heights should increase"
            );
        }
    }

    #[test]
    fn test_validate_cutree_vs_r() {
        // R reference:
        // hc <- hclust(dist(data))
        // cutree(hc, k=3)
        let data = create_validation_cluster_data();
        let hclust = hierarchical(data.view(), Some(1), Linkage::Ward, None).unwrap();
        let result = cutree(&hclust, Some(3), None).unwrap();

        assert_eq!(result.k, 3);
        assert_eq!(result.n, 15);
        assert_eq!(result.labels.len(), 15);

        // Should produce 3 distinct cluster labels
        let unique_labels: HashSet<_> = result.labels.iter().collect();
        assert_eq!(unique_labels.len(), 3, "Should have 3 distinct clusters");
    }

    #[test]
    fn test_validate_kmeans_inertia_decreases() {
        // Adding more clusters should decrease inertia
        let data = create_validation_cluster_data();

        let result2 = kmeans(data.view(), 2, Some(100), None, Some(5), Some(42)).unwrap();
        let result3 = kmeans(data.view(), 3, Some(100), None, Some(5), Some(42)).unwrap();
        let result4 = kmeans(data.view(), 4, Some(100), None, Some(5), Some(42)).unwrap();

        assert!(
            result3.inertia <= result2.inertia,
            "k=3 inertia {} should be <= k=2 inertia {}",
            result3.inertia,
            result2.inertia
        );
        assert!(
            result4.inertia <= result3.inertia,
            "k=4 inertia {} should be <= k=3 inertia {}",
            result4.inertia,
            result3.inertia
        );
    }

    #[test]
    fn test_validate_dbscan_eps_sensitivity() {
        // Larger eps should result in fewer clusters
        let data = create_validation_cluster_data();

        let result_small = dbscan(data.view(), 1.0, 3).unwrap();
        let result_large = dbscan(data.view(), 5.0, 3).unwrap();

        // With larger eps, should merge more into same cluster
        assert!(
            result_large.n_clusters <= result_small.n_clusters,
            "Larger eps should produce <= clusters"
        );
    }

    #[test]
    fn test_validate_hierarchical_linkage_heights() {
        // Different linkages should produce different merge heights
        let data = create_validation_cluster_data();

        let single = hierarchical(data.view(), Some(1), Linkage::Single, None).unwrap();
        let complete = hierarchical(data.view(), Some(1), Linkage::Complete, None).unwrap();

        // Complete linkage should have larger max height than single linkage
        let single_max = single
            .merge_distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let complete_max = complete
            .merge_distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            complete_max >= single_max,
            "Complete linkage max height {} should >= single linkage {}",
            complete_max,
            single_max
        );
    }

    #[test]
    fn test_validate_kmeans_reproducibility() {
        // Same seed should produce same results
        let data = create_validation_cluster_data();

        let result1 = kmeans(data.view(), 3, Some(100), None, Some(5), Some(42)).unwrap();
        let result2 = kmeans(data.view(), 3, Some(100), None, Some(5), Some(42)).unwrap();

        // Labels should be identical
        assert_eq!(
            result1.labels, result2.labels,
            "Same seed should give same labels"
        );

        // Inertia should be identical
        assert!(
            (result1.inertia - result2.inertia).abs() < 1e-10,
            "Same seed should give same inertia"
        );
    }
}
