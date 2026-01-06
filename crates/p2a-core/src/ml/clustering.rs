//! Clustering algorithms: K-means and DBSCAN.
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
}
