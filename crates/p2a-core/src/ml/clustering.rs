//! Clustering algorithms: K-means, DBSCAN, and Hierarchical Clustering.
//!
//! Pure Rust implementations using ndarray.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
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
            let centroid: Vec<String> = self.centroids.row(i).iter()
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
        writeln!(f, "Number of core samples: {}", self.core_sample_indices.len())?;

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
            let count = cluster_counts[&label];
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
        return Err(format!("k ({}) cannot be greater than n_samples ({})", k, n_samples));
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

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Assign points to nearest centroid
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

/// Run DBSCAN clustering.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `eps` - Maximum distance between two samples for neighborhood
/// * `min_samples` - Minimum samples in neighborhood for core point
pub fn dbscan(
    data: ArrayView2<f64>,
    eps: f64,
    min_samples: usize,
) -> Result<DBSCANResult, String> {
    let n_samples = data.nrows();

    if eps <= 0.0 {
        return Err("eps must be positive".to_string());
    }
    if min_samples == 0 {
        return Err("min_samples must be at least 1".to_string());
    }

    let eps_squared = eps * eps;

    // Find neighbors for each point
    let mut neighborhoods: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut neighbors = Vec::new();
        for j in 0..n_samples {
            if euclidean_distance_squared(&data.row(i), &data.row(j)) <= eps_squared {
                neighbors.push(j);
            }
        }
        neighborhoods.push(neighbors);
    }

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
            _ => Err(format!("Unknown linkage method: {}. Use single, complete, average, or ward", s)),
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
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
        }

        writeln!(f)?;
        writeln!(f, "Dendrogram (merge history):")?;
        writeln!(f, "  Step  Cluster1  Cluster2  Distance    Size")?;
        for (i, &(c1, c2, dist, size)) in self.linkage_matrix.iter().enumerate() {
            writeln!(f, "  {:4}  {:8}  {:8}  {:10.4}  {:4}", i + 1, c1, c2, dist, size)?;
        }

        Ok(())
    }
}

/// Run hierarchical agglomerative clustering.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (if None, returns full dendrogram)
/// * `linkage` - Linkage method (single, complete, average, ward)
/// * `distance_threshold` - If set, cut tree at this distance instead of n_clusters
pub fn hierarchical(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    linkage: Linkage,
    distance_threshold: Option<f64>,
) -> Result<HierarchicalResult, String> {
    let n_samples = data.nrows();

    if n_samples == 0 {
        return Err("Cannot cluster empty data".to_string());
    }
    if n_samples == 1 {
        return Ok(HierarchicalResult {
            labels: vec![0],
            n_clusters: 1,
            linkage_matrix: vec![],
            merge_distances: vec![],
            linkage: format!("{:?}", linkage).to_lowercase(),
        });
    }

    let target_clusters = match (n_clusters, distance_threshold) {
        (Some(n), _) => {
            if n == 0 || n > n_samples {
                return Err(format!(
                    "n_clusters must be between 1 and {} (n_samples)",
                    n_samples
                ));
            }
            Some(n)
        }
        (None, None) => Some(1), // Default: cluster all into one
        (None, Some(_)) => None, // Will use distance threshold
    };

    // Compute initial pairwise distance matrix
    let mut distances = compute_distance_matrix(&data);

    // Track cluster membership: cluster_id -> indices in that cluster
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_samples {
        clusters.insert(i, vec![i]);
    }

    // Active cluster IDs
    let mut active: HashSet<usize> = (0..n_samples).collect();

    // Linkage matrix storage
    let mut linkage_matrix: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n_samples - 1);
    let mut merge_distances: Vec<f64> = Vec::with_capacity(n_samples - 1);

    let mut next_cluster_id = n_samples;

    // Agglomerative clustering loop
    while active.len() > 1 {
        // Check if we've reached target number of clusters
        if let Some(target) = target_clusters {
            if active.len() <= target {
                break;
            }
        }

        // Find closest pair of clusters
        let (c1, c2, min_dist) = find_closest_clusters(&active, &distances, &clusters, &data, linkage)?;

        // Check distance threshold
        if let Some(thresh) = distance_threshold {
            if min_dist > thresh {
                break;
            }
        }

        // Merge clusters
        let merged_indices: Vec<usize> = {
            let mut merged = clusters.remove(&c1).unwrap();
            merged.extend(clusters.remove(&c2).unwrap());
            merged
        };
        let merged_size = merged_indices.len();

        // Record merge
        linkage_matrix.push((c1, c2, min_dist, merged_size));
        merge_distances.push(min_dist);

        // Update distance matrix for new cluster
        update_distances_for_merge(
            &mut distances,
            &active,
            c1,
            c2,
            next_cluster_id,
            &merged_indices,
            &clusters,
            &data,
            linkage,
        );

        // Update cluster tracking
        active.remove(&c1);
        active.remove(&c2);
        active.insert(next_cluster_id);
        clusters.insert(next_cluster_id, merged_indices);

        next_cluster_id += 1;
    }

    // Assign final labels
    let mut labels = vec![0usize; n_samples];
    for (cluster_label, &cluster_id) in active.iter().enumerate() {
        if let Some(indices) = clusters.get(&cluster_id) {
            for &idx in indices {
                labels[idx] = cluster_label;
            }
        }
    }

    Ok(HierarchicalResult {
        labels,
        n_clusters: active.len(),
        linkage_matrix,
        merge_distances,
        linkage: format!("{:?}", linkage).to_lowercase(),
    })
}

/// Compute initial pairwise distance matrix.
fn compute_distance_matrix(data: &ArrayView2<f64>) -> HashMap<(usize, usize), f64> {
    let n = data.nrows();
    let mut distances = HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
            distances.insert((i, j), dist);
            distances.insert((j, i), dist);
        }
    }

    distances
}

/// Find the closest pair of clusters.
fn find_closest_clusters(
    active: &HashSet<usize>,
    distances: &HashMap<(usize, usize), f64>,
    clusters: &HashMap<usize, Vec<usize>>,
    data: &ArrayView2<f64>,
    linkage: Linkage,
) -> Result<(usize, usize, f64), String> {
    let mut min_dist = f64::INFINITY;
    let mut best_pair = (0, 0);

    let active_vec: Vec<usize> = active.iter().cloned().collect();

    for i in 0..active_vec.len() {
        for j in (i + 1)..active_vec.len() {
            let c1 = active_vec[i];
            let c2 = active_vec[j];

            let dist = cluster_distance(c1, c2, distances, clusters, data, linkage);

            if dist < min_dist {
                min_dist = dist;
                best_pair = (c1, c2);
            }
        }
    }

    if min_dist.is_infinite() {
        return Err("Could not find valid cluster pair".to_string());
    }

    Ok((best_pair.0, best_pair.1, min_dist))
}

/// Calculate distance between two clusters based on linkage method.
fn cluster_distance(
    c1: usize,
    c2: usize,
    distances: &HashMap<(usize, usize), f64>,
    clusters: &HashMap<usize, Vec<usize>>,
    data: &ArrayView2<f64>,
    linkage: Linkage,
) -> f64 {
    let indices1 = clusters.get(&c1).unwrap();
    let indices2 = clusters.get(&c2).unwrap();

    match linkage {
        Linkage::Single => {
            // Minimum distance between any pair
            let mut min_dist = f64::INFINITY;
            for &i in indices1 {
                for &j in indices2 {
                    if let Some(&d) = distances.get(&(i, j)) {
                        min_dist = min_dist.min(d);
                    } else {
                        // Compute if not in cache
                        let d = euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                        min_dist = min_dist.min(d);
                    }
                }
            }
            min_dist
        }
        Linkage::Complete => {
            // Maximum distance between any pair
            let mut max_dist = 0.0f64;
            for &i in indices1 {
                for &j in indices2 {
                    if let Some(&d) = distances.get(&(i, j)) {
                        max_dist = max_dist.max(d);
                    } else {
                        let d = euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                        max_dist = max_dist.max(d);
                    }
                }
            }
            max_dist
        }
        Linkage::Average => {
            // Average distance between all pairs
            let mut total = 0.0;
            let mut count = 0;
            for &i in indices1 {
                for &j in indices2 {
                    if let Some(&d) = distances.get(&(i, j)) {
                        total += d;
                    } else {
                        total += euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                    }
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f64
            } else {
                f64::INFINITY
            }
        }
        Linkage::Ward => {
            // Ward's minimum variance: increase in total within-cluster variance
            let n1 = indices1.len() as f64;
            let n2 = indices2.len() as f64;

            // Compute centroids
            let centroid1 = compute_centroid(data, indices1);
            let centroid2 = compute_centroid(data, indices2);

            // Ward distance: sqrt(2 * n1 * n2 / (n1 + n2)) * ||c1 - c2||
            let centroid_dist = centroid1
                .iter()
                .zip(centroid2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            ((2.0 * n1 * n2) / (n1 + n2)).sqrt() * centroid_dist
        }
    }
}

/// Compute centroid of a cluster.
fn compute_centroid(data: &ArrayView2<f64>, indices: &[usize]) -> Vec<f64> {
    let n_features = data.ncols();
    let mut centroid = vec![0.0; n_features];
    let n = indices.len() as f64;

    for &idx in indices {
        for (j, val) in data.row(idx).iter().enumerate() {
            centroid[j] += val;
        }
    }

    for val in &mut centroid {
        *val /= n;
    }

    centroid
}

/// Update distance matrix after merging two clusters.
#[allow(clippy::too_many_arguments)]
fn update_distances_for_merge(
    distances: &mut HashMap<(usize, usize), f64>,
    active: &HashSet<usize>,
    c1: usize,
    c2: usize,
    new_id: usize,
    merged_indices: &[usize],
    clusters: &HashMap<usize, Vec<usize>>,
    data: &ArrayView2<f64>,
    linkage: Linkage,
) {
    // For each other active cluster, compute distance to new merged cluster
    for &other in active {
        if other == c1 || other == c2 {
            continue;
        }

        let other_indices = clusters.get(&other).unwrap();

        let dist = match linkage {
            Linkage::Single => {
                let mut min_dist = f64::INFINITY;
                for &i in merged_indices {
                    for &j in other_indices {
                        if let Some(&d) = distances.get(&(i, j)) {
                            min_dist = min_dist.min(d);
                        } else {
                            let d = euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                            min_dist = min_dist.min(d);
                        }
                    }
                }
                min_dist
            }
            Linkage::Complete => {
                let mut max_dist = 0.0f64;
                for &i in merged_indices {
                    for &j in other_indices {
                        if let Some(&d) = distances.get(&(i, j)) {
                            max_dist = max_dist.max(d);
                        } else {
                            let d = euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                            max_dist = max_dist.max(d);
                        }
                    }
                }
                max_dist
            }
            Linkage::Average => {
                let mut total = 0.0;
                let mut count = 0;
                for &i in merged_indices {
                    for &j in other_indices {
                        if let Some(&d) = distances.get(&(i, j)) {
                            total += d;
                        } else {
                            total += euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                        }
                        count += 1;
                    }
                }
                total / count as f64
            }
            Linkage::Ward => {
                let n1 = merged_indices.len() as f64;
                let n2 = other_indices.len() as f64;
                let centroid1 = compute_centroid(data, merged_indices);
                let centroid2 = compute_centroid(data, other_indices);
                let centroid_dist = centroid1
                    .iter()
                    .zip(centroid2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                ((2.0 * n1 * n2) / (n1 + n2)).sqrt() * centroid_dist
            }
        };

        distances.insert((new_id, other), dist);
        distances.insert((other, new_id), dist);
    }
}

/// Squared Euclidean distance between two vectors.
#[inline]
fn euclidean_distance_squared(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
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
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ];

        // Test all linkage methods work
        for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward] {
            let result = hierarchical(data.view(), Some(2), linkage, None).unwrap();
            assert_eq!(result.n_clusters, 2);
        }
    }

    #[test]
    fn test_hierarchical_distance_threshold() {
        let data = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [10.0, 0.0],
            [10.5, 0.0],
        ];

        // With threshold of 1.0, should get 2 clusters
        let result = hierarchical(data.view(), None, Linkage::Single, Some(1.0)).unwrap();
        assert_eq!(result.n_clusters, 2);
    }
}
