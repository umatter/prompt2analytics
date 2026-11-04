//! Clustering algorithms: K-means, DBSCAN, and Hierarchical Clustering.
//!
//! Pure Rust implementations using ndarray.
//!
//! DBSCAN uses KD-tree acceleration for O(n log n) average complexity
//! when dimensionality ≤ 20. On its large-n paths, core points are found
//! from neighbour counts computed in parallel (rayon); the cluster expansion
//! then re-queries neighbourhoods serially, so memory is O(n) plus one query
//! result.

use super::extract::require_finite;
use super::hclust::{HclustTree, hclust_euclidean};
use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, current_budget, try_vec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

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
///
/// Refuses NaN and ±Inf, naming the first non-finite cell (R `kmeans` stops too).
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

    if n_initializations == 0 {
        return Err("n_init must be at least 1 (got 0)".to_string());
    }
    if max_iter == 0 {
        return Err("max_iterations must be at least 1 (got 0)".to_string());
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!(
            "tolerance must be finite and non-negative (got {tol})"
        ));
    }
    require_finite(&data, "k-means")?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut best_result: Option<KMeansResult> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_initializations {
        checkpoint().map_err(|e| e.to_string())?;

        // Initialize centroids using k-means++
        let centroids = kmeans_plusplus_init(&data, k, &mut rng)?;

        // Run k-means
        let result = kmeans_single(&data, centroids, max_iter, tol)?;

        if result.inertia < best_inertia {
            best_inertia = result.inertia;
            best_result = Some(result);
        }
    }

    best_result.ok_or_else(|| "K-means failed to converge".to_string())
}

/// K-means++ initialization.
///
/// Keeps each point's squared distance to its nearest chosen centroid and
/// updates it with the newest centroid only (O(n k d) overall). `f64::min` is
/// exact and order-independent, so the minima, and with them the sampled
/// centroids, equal those of a full rescan over all chosen centroids.
fn kmeans_plusplus_init(
    data: &ArrayView2<f64>,
    k: usize,
    rng: &mut StdRng,
) -> Result<Array2<f64>, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    let mut centroids = Array2::zeros((k, n_features));

    // Choose first centroid randomly
    let first_idx = rng.gen_range(0..n_samples);
    centroids.row_mut(0).assign(&data.row(first_idx));

    // Squared distance of each point to its nearest chosen centroid
    let mut distances = vec![f64::INFINITY; n_samples];

    // Choose remaining centroids
    for i in 1..k {
        checkpoint().map_err(|e| e.to_string())?;

        // Fold the newest centroid into the nearest-centroid distances
        let newest = centroids.row(i - 1);
        for (j, min_dist) in distances.iter_mut().enumerate() {
            let dist = euclidean_distance_squared(&data.row(j), &newest);
            *min_dist = min_dist.min(dist);
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

    Ok(centroids)
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
) -> Result<KMeansResult, String> {
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
        checkpoint().map_err(|e| e.to_string())?;
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
        update_centroids(data, &labels, &mut centroids);

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

    Ok(KMeansResult {
        labels,
        centroids,
        n_iterations,
        inertia,
        cluster_sizes,
    })
}

/// Moves each centroid with at least one point to the mean of its points; a centroid
/// with none keeps its position.
///
/// One pass over the rows: row `i` is added to the sum of cluster `labels[i]`, so each
/// cluster's sum takes its rows in row order, the same additions in the same order as a
/// separate pass per cluster (O(n k d)), and the centroids are bit-identical to that.
fn update_centroids(data: &ArrayView2<f64>, labels: &[usize], centroids: &mut Array2<f64>) {
    let (k, d) = centroids.dim();
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];
    for (row, &label) in data.rows().into_iter().zip(labels) {
        let sum = &mut sums[label * d..(label + 1) * d];
        for (s, &x) in sum.iter_mut().zip(row.iter()) {
            *s += x;
        }
        counts[label] += 1;
    }
    for (j, (sum, &count)) in sums.chunks_exact(d.max(1)).zip(&counts).enumerate() {
        if count > 0 {
            let c = count as f64;
            for (dst, &s) in centroids.row_mut(j).iter_mut().zip(sum) {
                *dst = s / c;
            }
        }
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
/// - **n >= 500, d <= 20**: KD-tree radius queries.
/// - **n >= 500, d > 20**: naive O(n^2) distance scans.
///
/// On the two large-n paths no neighbourhood is stored: core points are found
/// from neighbour counts computed in parallel (rayon), and the cluster
/// expansion then re-queries each core point's neighbourhood serially, so
/// memory is O(n) plus one query result.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `eps` - Maximum distance between two samples for neighborhood
/// * `min_samples` - Minimum samples in neighborhood for core point
pub fn dbscan(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if !(eps.is_finite() && eps > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("eps must be finite and positive (got {eps})"),
        });
    }
    if min_samples == 0 {
        return Err(EconError::InvalidSpecification {
            message: "min_samples must be at least 1 (got 0)".to_string(),
        });
    }
    if let Some(((i, j), v)) = data.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("data must contain only finite values (row {i}, column {j} is {v})"),
        });
    }

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
    let what = "DBSCAN distance matrix";
    let condensed_len = checked_len(&[n, n - 1], what)? / 2;
    let mut dist_sq = try_vec(condensed_len, 0.0f64, what)?;
    for i in 0..n {
        checkpoint()?;
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
            checkpoint()?;

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
/// Achieves O(n log n) average case for low-dimensional data. Neighbourhoods
/// are not stored: core points are found from parallel neighbour counts, and
/// the expansion re-queries the tree for every core point it pops (each point
/// is pushed at most once, so there are at most n queries). The queries return
/// neighbours in the same order every time, so the labels equal those of an
/// expansion over stored neighbourhoods, in O(n) memory plus one query result.
fn dbscan_kdtree(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    use super::kdtree::KdTree;

    let n_samples = data.nrows();

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
    let query = |i: usize| tree.radius_query_unsorted(&tree.data()[i], eps, None);

    // Count neighbors for each point (parallel); nothing else is kept
    let budget = current_budget();
    let neighbor_counts: Vec<usize> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            if let Some(budget) = &budget {
                budget.check()?;
            }
            Ok(query(i).len())
        })
        .collect::<EconResult<Vec<usize>>>()?;

    // Identify core points
    let is_core: Vec<bool> = neighbor_counts.iter().map(|&c| c >= min_samples).collect();
    let core_sample_indices: Vec<usize> = (0..n_samples).filter(|&i| is_core[i]).collect();

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
            checkpoint()?;
            for (_, neighbor) in query(idx) {
                if labels[neighbor] == -1 {
                    labels[neighbor] = current_cluster;

                    // If neighbor is a core point, expand from it
                    if is_core[neighbor] {
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
/// Used for high-dimensional data where KD-tree is inefficient. Like the
/// KD-tree path it keeps only neighbour counts and recomputes a core point's
/// neighbours (ascending index order) when the expansion pops it.
fn dbscan_naive(data: ArrayView2<f64>, eps: f64, min_samples: usize) -> EconResult<DBSCANResult> {
    let n_samples = data.nrows();

    let eps_squared = eps * eps;
    let neighbors = |i: usize| {
        let row_i = data.row(i);
        (0..n_samples)
            .filter(move |&j| euclidean_distance_squared(&row_i, &data.row(j)) <= eps_squared)
    };

    // Count neighbors for each point (parallel)
    let budget = current_budget();
    let neighbor_counts: Vec<usize> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            if let Some(budget) = &budget {
                budget.check()?;
            }
            Ok(neighbors(i).count())
        })
        .collect::<EconResult<Vec<usize>>>()?;

    // Identify core points
    let is_core: Vec<bool> = neighbor_counts.iter().map(|&c| c >= min_samples).collect();
    let core_sample_indices: Vec<usize> = (0..n_samples).filter(|&i| is_core[i]).collect();

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
            checkpoint()?;
            for neighbor in neighbors(idx) {
                if labels[neighbor] == -1 {
                    labels[neighbor] = current_cluster;

                    // If neighbor is a core point, expand from it
                    if is_core[neighbor] {
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

pub use super::hclust::Linkage;

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

/// Run hierarchical agglomerative clustering: R's `hclust(dist(data), linkage.r_name())`.
///
/// The tree is built by the shared core [`crate::ml::hclust`] (R's `hclust.f`): all eight R
/// methods, with [`Linkage::Ward`] = `ward.D2`; centroid and median work on squared Euclidean
/// distances, as `?hclust` documents, and their heights are those of `hclust(dist(x)^2, m)`.
/// The full tree is always returned (`linkage_matrix` and `merge_distances` hold all `n - 1`
/// merges); the labels cut it, numbered by first appearance in observation order (R's
/// `cutree`, minus 1):
///
/// * `n_clusters = Some(k)`: `cutree(k = k)`; as in R, `k` overrides `distance_threshold`;
/// * `distance_threshold = Some(h)`: `cutree(h = h)`; refused for a tree whose heights decrease
///   (centroid, median), as R's `cutree` refuses it;
/// * neither: one cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to cut the tree into
/// * `linkage` - Linkage method (any of the eight R methods)
/// * `distance_threshold` - If set (and `n_clusters` is not), cut the tree at this height
///
/// Refuses NaN and ±Inf, naming the first non-finite cell. Distances that overflow f64 (also
/// after squaring, for Ward, centroid and median) are refused in both modes, with or without
/// `distance_threshold`; rescale the columns. A row with one missing coordinate is refused: R's
/// `dist()` would use the coordinates both rows have, which is not implemented here.
///
/// # References
///
/// - Murtagh, F. (1983). "A Survey of Recent Advances in Hierarchical Clustering
///   Algorithms". The Computer Journal, 26(4), 354-359.
/// - Lance, G. N. & Williams, W. T. (1967). "A General Theory of Classificatory
///   Sorting Strategies: 1. Hierarchical Systems". The Computer Journal, 9(4), 373-380.
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
    require_finite(&data, "hierarchical clustering")?;
    if let Some(nc) = n_clusters {
        if nc == 0 || nc > n {
            return Err(format!(
                "n_clusters must be between 1 and {} (n_samples)",
                n
            ));
        }
    }
    if let Some(h) = distance_threshold {
        if !h.is_finite() {
            return Err(format!("distance_threshold must be finite (got {h})"));
        }
    }
    if n == 1 {
        return Ok(HierarchicalResult {
            labels: vec![0],
            n_clusters: 1,
            linkage_matrix: vec![],
            merge_distances: vec![],
            linkage: linkage.r_name().to_string(),
        });
    }

    let tree = hclust_euclidean(data, linkage, "hierarchical clustering distance matrix")
        .map_err(|e| e.to_string())?;
    let cut = match (n_clusters, distance_threshold) {
        (Some(k), _) => tree.cut_k(k),
        (None, Some(h)) => tree.cut_height(h),
        (None, None) => tree.cut_k(1),
    }
    .map_err(|e| e.to_string())?;
    let n_found = cut.iter().copied().max().unwrap_or(0);
    let labels = cut.into_iter().map(|l| l - 1).collect();

    Ok(HierarchicalResult {
        labels,
        n_clusters: n_found,
        linkage_matrix: tree.scipy_linkage(),
        merge_distances: tree.height().to_vec(),
        linkage: linkage.r_name().to_string(),
    })
}

/// Squared Euclidean distance between two vectors.
///
/// Contiguous rows (any standard-layout matrix) take a plain slice loop, which does not
/// depend on how ndarray's iterator gets inlined; strided rows take the ndarray iterator.
/// Both add the terms in index order, so the result is the same either way.
#[inline]
fn euclidean_distance_squared(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    if let (Some(a), Some(b)) = (a.as_slice(), b.as_slice()) {
        return a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    }
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

/// Cut a hierarchical clustering tree into groups: R's `cutree`.
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
/// * `CutreeResult` containing cluster assignments (1-based, numbered by first appearance in
///   observation order, as R)
///
/// The linkage matrix must hold the complete tree (`n - 1` merges); a cut by `h` is refused
/// for a tree whose heights decrease (centroid, median), as R's `cutree` refuses it.
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
    let n = hclust.labels.len();
    if n == 0 {
        return Err("Empty clustering result".to_string());
    }
    if let Some(k_val) = k {
        check_cut_k(k_val, n)?;
    }
    let (labels, target_k) = if n == 1 {
        if let Some(h_val) = h.filter(|_| k.is_none()) {
            if !h_val.is_finite() {
                return Err(format!("the cut height must be finite (got {h_val})"));
            }
        }
        (vec![1], 1)
    } else {
        let tree = complete_tree(hclust)?;
        let target_k = match k {
            Some(k_val) => k_val,
            None => tree
                .k_for_height(h.unwrap_or(f64::NAN))
                .map_err(|e| e.to_string())?,
        };
        (tree.cut_k(target_k).map_err(|e| e.to_string())?, target_k)
    };

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

fn check_cut_k(k: usize, n: usize) -> Result<(), String> {
    if k == 0 || k > n {
        return Err(format!(
            "k must be between 1 and {} (number of observations)",
            n
        ));
    }
    Ok(())
}

/// The tree of a `HierarchicalResult`; refuses a linkage matrix without all `n - 1` merges.
fn complete_tree(hclust: &HierarchicalResult) -> Result<HclustTree, String> {
    let n = hclust.labels.len();
    let need = n.saturating_sub(1);
    if hclust.linkage_matrix.len() != need {
        return Err(format!(
            "cutree needs the complete tree: the linkage matrix has {} merges, {need} merges are \
             needed for {n} observations",
            hclust.linkage_matrix.len()
        ));
    }
    HclustTree::from_scipy_linkage(n, &hclust.linkage_matrix).map_err(|e| e.to_string())
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
    if n == 0 {
        return Err("Empty clustering result".to_string());
    }
    for &k in k_values {
        check_cut_k(k, n)?;
    }
    let labels_matrix: Vec<Vec<usize>> = if n == 1 {
        vec![vec![1]; k_values.len()]
    } else {
        complete_tree(hclust)?
            .cut_ks(k_values)
            .map_err(|e| e.to_string())?
    };

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
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;
    use std::collections::HashSet;

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// Three loose clusters along the diagonal, uniform noise in each coordinate.
    fn three_clusters(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, d), |(i, _)| (i % 3) as f64 * 4.0 + rng.r#gen::<f64>())
    }

    /// The review's ML-24 rows; R: kmeans stops ("NA/NaN/Inf in foreign function call (arg 1)").
    fn ml24_rows(nan_row: usize) -> Array2<f64> {
        // nan_row 4 gives the review's (NA, 1) row; nan_row 0 puts the NaN row first (the
        // self-merge case)
        let mut m = array![[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0], [9.0, 1.0]];
        m[[nan_row, 0]] = f64::NAN;
        m
    }

    #[test]
    fn test_na_hierarchical_refuses_non_finite_input() {
        // ML-24. Before the fix: Ok with a corrupt tree (Ward, n_clusters = 1 reports 2
        // clusters and merges a cluster twice; with the NaN row first, a self-merge
        // (0, 0, 0.0, 2)).
        for linkage in [
            Linkage::Ward,
            Linkage::Single,
            Linkage::Complete,
            Linkage::Average,
        ] {
            for (nan_row, cell) in [(4, "row 4, column 0 is NaN"), (0, "row 0, column 0 is NaN")] {
                let err =
                    hierarchical(ml24_rows(nan_row).view(), Some(1), linkage, None).unwrap_err();
                assert!(
                    err.starts_with(
                        "hierarchical clustering: data must contain only finite values"
                    ),
                    "{linkage:?}: {err}"
                );
                assert!(err.contains(cell), "{linkage:?}: {err}");
            }
        }
        let mut inf = ml24_rows(4);
        inf[[4, 0]] = 9.0;
        inf[[2, 1]] = f64::NEG_INFINITY;
        let err = hierarchical(inf.view(), Some(2), Linkage::Ward, None).unwrap_err();
        assert!(err.contains("row 2, column 1 is -inf"), "{err}");
        // a single NaN row is refused too (it used to return the one-row tree)
        let one = array![[f64::NAN, 1.0]];
        let err = hierarchical(one.view(), None, Linkage::Ward, None).unwrap_err();
        assert!(err.contains("row 0, column 0 is NaN"), "{err}");
    }

    #[test]
    fn test_hierarchical_errors_when_no_finite_distance_remains() {
        // ML-24's loop: finite data whose squared distances overflow to Inf. R: dist() is
        // Inf Inf Inf and hclust() stops with "NA/NaN/Inf in foreign function call
        // (arg 10)". Before the fix (Ward): Ok with n_clusters 3 although 1 was requested
        // and the self-merges (0, 0, 0.0, 2), (3, 3, 0.0, 4).
        let big = array![[0.0, 0.0], [1e200, 0.0], [2e200, 0.0]];
        for linkage in [
            Linkage::Ward,
            Linkage::Single,
            Linkage::Complete,
            Linkage::Average,
        ] {
            let err = hierarchical(big.view(), Some(1), linkage, None).unwrap_err();
            assert!(
                err.contains("no finite distance remains"),
                "{linkage:?}: {err}"
            );
        }
    }

    #[test]
    fn test_na_kmeans_refuses_non_finite_input() {
        // Before the fix: "K-means failed to converge" (every inertia is NaN).
        let err = kmeans(ml24_rows(4).view(), 2, Some(100), None, Some(2), Some(1)).unwrap_err();
        assert_eq!(
            err,
            "k-means: data must contain only finite values (row 4, column 0 is NaN); remove or impute missing values first"
        );
    }

    #[test]
    fn kmeans_validates_counts_and_tolerance() {
        let data = three_clusters(60, 2, 1);
        assert!(
            kmeans(data.view(), 3, Some(100), None, Some(0), Some(1))
                .unwrap_err()
                .starts_with("n_init")
        );
        assert!(
            kmeans(data.view(), 3, Some(0), None, Some(2), Some(1))
                .unwrap_err()
                .starts_with("max_iterations")
        );
        assert!(
            kmeans(data.view(), 3, Some(100), Some(f64::NAN), Some(2), Some(1))
                .unwrap_err()
                .starts_with("tolerance")
        );
        assert!(
            kmeans(data.view(), 3, Some(100), Some(-1.0), Some(2), Some(1))
                .unwrap_err()
                .starts_with("tolerance")
        );
        let e = with_budget(cancelled(), || {
            kmeans(data.view(), 3, None, None, Some(2), Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn dbscan_validates_eps_and_data() {
        let mut data = three_clusters(50, 2, 2);
        for eps in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert!(
                matches!(
                    dbscan(data.view(), eps, 3),
                    Err(EconError::InvalidSpecification { .. })
                ),
                "eps={eps}"
            );
        }
        data[[7, 0]] = f64::NAN;
        assert!(
            dbscan(data.view(), 0.5, 3)
                .unwrap_err()
                .to_string()
                .contains("finite")
        );
    }

    #[test]
    fn dbscan_and_hierarchical_buffers_are_fallible_and_cancellable() {
        let data = three_clusters(100, 2, 3);
        let r = with_alloc_ceiling(Some(4096), || dbscan(data.view(), 0.5, 3)); // small-n path, 39,600 B condensed
        assert!(matches!(r, Err(EconError::ResourceLimit(_))), "{r:?}");
        let e = with_alloc_ceiling(Some(4096), || {
            hierarchical(data.view(), Some(3), Linkage::Average, None)
        })
        .unwrap_err();
        assert!(e.contains("limit"), "{e}");
        let e = with_budget(cancelled(), || {
            hierarchical(data.view(), Some(3), Linkage::Ward, None)
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        let big = three_clusters(600, 2, 4); // kd-tree path
        let r = with_budget(cancelled(), || dbscan(big.view(), 0.3, 4));
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
        assert!(
            dbscan(data.view(), 0.5, 3).is_ok()
                && hierarchical(data.view(), Some(3), Linkage::Ward, None).is_ok()
        );
    }

    /// Verbatim copy of `dbscan_kdtree` before it stopped storing every
    /// neighbourhood; the reference for
    /// `dbscan_without_stored_neighbourhoods_matches_reference`.
    fn reference_dbscan_kdtree(
        data: ArrayView2<f64>,
        eps: f64,
        min_samples: usize,
    ) -> EconResult<DBSCANResult> {
        use super::super::kdtree::KdTree;

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

    /// Verbatim copy of `dbscan_naive` before it stopped storing every
    /// neighbourhood; the reference for
    /// `dbscan_without_stored_neighbourhoods_matches_reference`.
    fn reference_dbscan_naive(
        data: ArrayView2<f64>,
        eps: f64,
        min_samples: usize,
    ) -> EconResult<DBSCANResult> {
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

    /// Reference guard: DBSCAN without stored neighbourhoods must label
    /// exactly as the versions that stored them (kd-tree and naive paths).
    #[test]
    fn dbscan_without_stored_neighbourhoods_matches_reference() {
        for (n, d, eps, min_samples, seed) in [
            (700, 2, 0.1, 5, 10),
            (700, 3, 0.2, 8, 11),
            (520, 21, 1.4, 5, 12),
        ] {
            let data = three_clusters(n, d, seed);
            let got = dbscan(data.view(), eps, min_samples).unwrap();
            let want = if d <= KDTREE_MAX_DIMS {
                reference_dbscan_kdtree(data.view(), eps, min_samples)
            } else {
                reference_dbscan_naive(data.view(), eps, min_samples)
            }
            .unwrap();
            assert!(
                want.n_clusters >= 2 && want.n_noise > 0,
                "tune eps: {} clusters, {} noise",
                want.n_clusters,
                want.n_noise
            );
            assert_eq!(got.labels, want.labels, "n={n} d={d}");
            assert_eq!(got.core_sample_indices, want.core_sample_indices);
            assert_eq!(
                (got.n_clusters, got.n_noise),
                (want.n_clusters, want.n_noise)
            );
        }
    }

    /// The one-pass centroid update equals a pass per cluster bit for bit, including a
    /// cluster with no points (which keeps its centroid).
    #[test]
    fn update_centroids_matches_per_cluster_passes() {
        let data = three_clusters(301, 4, 7);
        let k = 5;
        let labels: Vec<usize> = (0..data.nrows()).map(|i| (i * 7 + i / 3) % 4).collect();
        let start = Array2::from_shape_fn((k, 4), |(j, c)| j as f64 - 0.25 * c as f64);
        let mut reference = start.clone();
        for j in 0..k {
            let mut sum = Array1::<f64>::zeros(4);
            let mut count = 0;
            for i in 0..data.nrows() {
                if labels[i] == j {
                    sum += &data.row(i);
                    count += 1;
                }
            }
            if count > 0 {
                reference.row_mut(j).assign(&(sum / count as f64));
            }
        }
        let mut got = start.clone();
        update_centroids(&data.view(), &labels, &mut got);
        let bits = |a: &Array2<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&reference));
        assert_eq!(got.row(4), start.row(4));
    }

    /// The slice path of the distance equals the strided (iterator) path bit for bit.
    #[test]
    fn euclidean_distance_squared_slice_and_strided_paths_agree() {
        let data = three_clusters(40, 6, 3);
        // Standard-layout transpose: its columns are the rows of data, strided
        let t = Array2::from_shape_fn((data.ncols(), data.nrows()), |(c, r)| data[[r, c]]);
        for i in 0..data.nrows() {
            for j in 0..data.nrows() {
                let (a, b) = (data.row(i), data.row(j));
                let (sa, sb) = (t.column(i), t.column(j));
                assert!(a.as_slice().is_some() && sa.as_slice().is_none());
                assert_eq!(
                    euclidean_distance_squared(&a, &b).to_bits(),
                    euclidean_distance_squared(&sa, &sb).to_bits()
                );
            }
        }
    }

    /// Golden (characterization) test: captured on the code before the
    /// running-minimum k-means++; it passes before and after.
    #[test]
    fn kmeans_plusplus_running_min_is_unchanged() {
        const GOLDEN_INERTIA: u64 = 4633615604438900480;
        const GOLDEN_ITERS: usize = 8;
        const GOLDEN_LABELS: [usize; 12] = [3, 2, 0, 1, 4, 0, 1, 2, 0, 1, 2, 0];
        let data = three_clusters(300, 3, 4);
        let r = kmeans(data.view(), 5, Some(100), Some(1e-6), Some(3), Some(11)).unwrap();
        assert_eq!(
            (r.inertia.to_bits(), r.n_iterations, &r.labels[..12]),
            (GOLDEN_INERTIA, GOLDEN_ITERS, &GOLDEN_LABELS[..]),
            "got inertia {:?} ({}), iterations {}, labels {:?}",
            r.inertia,
            r.inertia.to_bits(),
            r.n_iterations,
            &r.labels[..12]
        );
    }

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

    use crate::ml::hclust::{HclustTree, test_util::*};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_hierarchical_linkages() {
        // ML-25: ml::hierarchical with Linkage::Ward is hclust(dist(x), "ward.D2"); before: heights inflated
        // and the merge order differs. hclust_methods, validation/r/cases/ml_hierarchical.R.
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let g = matrix(&case_frame(&case, "hclust_ties.csv"), &["u", "v"]);
        let mut c = Check::new(&case);
        for (l, tag) in METHODS {
            for (data, pre, n) in [(&x, "", 60usize), (&g, "ties_", 40usize)] {
                let r = hierarchical(data.view(), None, l, None).unwrap();
                let tree = HclustTree::from_scipy_linkage(n, &r.linkage_matrix).unwrap();
                c.vector(
                    &format!("{pre}{tag}_merge"),
                    &merge_colmajor(tree.merge()),
                    Tol::exact(),
                )
                .vector(
                    &format!("{pre}{tag}_height"),
                    &r.merge_distances,
                    Tol::coef(n),
                );
                assert_eq!(r.linkage, l.r_name());
            }
        }
        c.finish();
    }

    #[test]
    fn test_validate_cutree3() {
        // ML-83 / ML-14 (ml::cutree): R numbers groups by first appearance; before: HashSet order, and Ward wrong.
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let mut c = Check::new(&case);
        for (l, tag) in METHODS {
            let full = hierarchical(x.view(), Some(1), l, None).unwrap();
            c.vector(
                &format!("{tag}_cutree3"),
                &as_f64(&cutree(&full, Some(3), None).unwrap().labels),
                Tol::exact(),
            );
            c.vector(
                &format!("{tag}_cutree3"),
                &plus_one(&hierarchical(x.view(), Some(3), l, None).unwrap().labels),
                Tol::exact(),
            );
        }
        let avg = hierarchical(x.view(), None, Linkage::Average, None).unwrap();
        let multi = cutree_multiple_k(&avg, &[2, 4, 6]).unwrap();
        c.vector(
            "average_cutree_k246",
            &as_f64(&multi.labels_matrix.unwrap().concat()),
            Tol::exact(),
        );
        let h = cutree(&avg, None, Some(1.5)).unwrap();
        c.vector("average_cutree_h15", &as_f64(&h.labels), Tol::exact())
            .scalar("average_k_h15", h.k as f64, Tol::exact());
        c.vector(
            "average_cutree_h15",
            &plus_one(
                &hierarchical(x.view(), None, Linkage::Average, Some(1.5))
                    .unwrap()
                    .labels,
            ),
            Tol::exact(),
        );
        c.finish();
    }

    #[test]
    fn test_validate_cutree_labels_deterministic_first_appearance() {
        // ML-83: three calls gave [3,3,..], [2,2,..], [2,2,..] before
        let (_, ds) = crate::test_support::load_case("hclust_methods");
        let h = hierarchical(
            matrix(ds.df(), &["a", "b", "c"]).view(),
            Some(1),
            Linkage::Ward,
            None,
        )
        .unwrap();
        let first = cutree(&h, Some(3), None).unwrap().labels;
        assert_eq!(first[0], 1);
        for _ in 0..8 {
            assert_eq!(cutree(&h, Some(3), None).unwrap().labels, first);
        }
    }

    #[test]
    fn test_cutree_refuses_a_truncated_tree() {
        // ML-82: a HierarchicalResult whose linkage matrix lacks merges (built by hand, or by old code) is refused
        let x = array![[0.0], [1.0], [5.0], [6.0], [20.0]];
        let mut h = hierarchical(x.view(), Some(1), Linkage::Average, None).unwrap();
        h.linkage_matrix.truncate(2);
        let e = cutree(&h, Some(2), None).unwrap_err();
        assert!(e.contains("4 merges"), "{e}");
        // always the full tree
        assert_eq!(
            hierarchical(x.view(), Some(5), Linkage::Average, None)
                .unwrap()
                .linkage_matrix
                .len(),
            4
        );
    }

    #[test]
    fn test_hierarchical_height_cut_refuses_inversions() {
        let x = array![[0.0, 0.0], [2.0, 0.0], [1.0, 1.8]];
        assert!(
            hierarchical(x.view(), None, Linkage::Centroid, Some(1.0))
                .unwrap_err()
                .contains("not sorted")
        );
        assert!(
            hierarchical(x.view(), None, Linkage::Single, Some(f64::NAN))
                .unwrap_err()
                .starts_with("distance_threshold")
        );
    }

    #[test]
    fn test_hierarchical_threshold_mode_refuses_overflow_like_k_mode() {
        // T2-WP17 leftover: with distance_threshold, overflowing distances gave an error or a
        // partition depending on where the overflow happened (condensed distance vs Ward height). One
        // rule now: the tree needs every dissimilarity finite, in both modes and for every linkage.
        let big = array![[0.0, 0.0], [1e200, 0.0], [2e200, 0.0]];
        for l in [Linkage::Single, Linkage::Ward] {
            let e = hierarchical(big.view(), None, l, Some(10.0)).unwrap_err();
            assert!(e.contains("no finite distance remains"), "{l:?}: {e}");
            assert_eq!(e, hierarchical(big.view(), Some(2), l, None).unwrap_err());
        }
        // finite distances whose squares overflow: Ward (internally squared) is refused as well
        let wide = array![[0.0], [1e155], [3e155]];
        assert!(
            hierarchical(wide.view(), None, Linkage::Ward, Some(10.0))
                .unwrap_err()
                .contains("no finite distance remains")
        );
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
    fn test_hierarchical_complete_max_height_at_least_single() {
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
