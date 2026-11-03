//! Pieces shared by the advanced clustering methods: the HDBSCAN/OPTICS algorithm selection ([`ClusteringAlgorithm`]), distance matrices, silhouette helpers and cluster statistics ([`cluster_stats`]).

use crate::resource::{checkpoint, try_zeros2};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
pub(crate) fn select_algorithm(n: usize, d: usize) -> ClusteringAlgorithm {
    match (n, d) {
        (n, _) if n < 500 => ClusteringAlgorithm::BruteForce,
        (n, d) if n < 10_000 && d <= 15 => ClusteringAlgorithm::KdTreePrim,
        (_, d) if d <= 15 => ClusteringAlgorithm::DualTreeBoruvka,
        _ => ClusteringAlgorithm::BruteForceParallel,
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Compute distance matrix in parallel.
pub(crate) fn compute_distance_matrix_parallel(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();

    let data_slice: Vec<f64> = data.iter().cloned().collect();
    let data_ref = &data_slice;

    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n)
                .map(move |j| {
                    let mut sum = 0.0;
                    for k in 0..d {
                        let diff = data_ref[i * d + k] - data_ref[j * d + k];
                        sum += diff * diff;
                    }
                    (i, j, sum.sqrt())
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = Array2::zeros((n, n));
    for (i, j, dist) in results {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }

    distances
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute pairwise Euclidean distances.
pub(crate) fn compute_distance_matrix(data: &ArrayView2<f64>) -> Array2<f64> {
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

/// Pairwise Euclidean distances (as [`compute_distance_matrix`]), with the
/// `n x n` matrix allocated through the resource guards and a cancellation
/// checkpoint per row.
pub(crate) fn try_distance_matrix(
    data: &ArrayView2<f64>,
    what: &str,
) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let mut distances = try_zeros2(n, n, what).map_err(|e| e.to_string())?;

    for i in 0..n {
        checkpoint().map_err(|e| e.to_string())?;
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

    Ok(distances)
}

/// Compute silhouette from precomputed distances.
pub(crate) fn compute_silhouette_from_distances(
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
            let sum: f64 = cluster_i_indices
                .iter()
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
            let avg_dist: f64 = cluster_indices[c]
                .iter()
                .map(|&j| distances[[i, j]])
                .sum::<f64>()
                / cluster_indices[c].len() as f64;
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
// Shared helpers
// =============================================================================

/// Compute average silhouette width.
pub(crate) fn compute_silhouette_avg(labels: &[usize], distances: &Array2<f64>, k: usize) -> f64 {
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

// =============================================================================
// Cluster Statistics - Comprehensive Cluster Validation
// =============================================================================

/// Comprehensive cluster statistics result.
///
/// # References
///
/// - Hennig, C. (2020). fpc: Flexible Procedures for Clustering. R package.
/// - R fpc::cluster.stats documentation
///   Source: <https://www.rdocumentation.org/packages/fpc/versions/2.2-10/topics/cluster.stats>
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
        return Err(format!(
            "Data rows ({}) != labels length ({})",
            n,
            labels.len()
        ));
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
    let global_mean: Vec<f64> = (0..p).map(|j| data.column(j).sum() / n as f64).collect();

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
        let members: Vec<usize> = labels
            .iter()
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
    let (average_silhouette, cluster_silhouette) =
        compute_silhouette_details(labels, &distances, k);

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
fn compute_silhouette_details(
    labels: &[usize],
    distances: &Array2<f64>,
    k: usize,
) -> (f64, Vec<f64>) {
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
        let a_i = if same_count > 0 {
            same_dist / same_count as f64
        } else {
            0.0
        };

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

    let _n = labels.len();
    let p = centroids[0].len();

    // Compute scatter for each cluster (average distance to centroid)
    let mut scatter = vec![0.0; k];
    let mut counts = vec![0; k];

    for &l in labels.iter() {
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

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    /// R 4.3.3: library(cluster); x <- <the 12 x 2 matrix of test_validate_agnes_heights_match_r>
    /// lab <- c(1,1,2,2,1,3,3,1,2,3,1,2); mean(silhouette(lab, dist(x))[, "sil_width"])  # 0.617882355558917
    #[test]
    fn test_validate_clara_silhouette_helper_matches_r() {
        let x = array![
            [0.12, 1.05],
            [0.48, 0.31],
            [2.91, 3.17],
            [3.36, 2.62],
            [1.07, 0.74],
            [5.21, 0.43],
            [4.83, 1.12],
            [0.71, 2.38],
            [2.24, 2.85],
            [5.66, 0.97],
            [1.53, 1.36],
            [3.95, 3.48]
        ];
        let labels = vec![0, 0, 1, 1, 0, 2, 2, 0, 1, 2, 0, 1];
        let d = compute_distance_matrix(&x.view());
        assert!((compute_silhouette_avg(&labels, &d, 3) - 0.617882355558917).abs() < 1e-12);
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
        assert!(
            result.average_silhouette > 0.5,
            "Expected high silhouette for well-separated clusters"
        );
    }

    #[test]
    fn test_cluster_stats_single_cluster() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],];
        let labels = vec![0, 0, 0];

        let result = cluster_stats(data.view(), &labels).unwrap();

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes, vec![3]);
        // With single cluster, between_ss = 0 and silhouette = 0
        assert!(result.between_ss < 1e-10);
    }
}
