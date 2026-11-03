//! Pieces shared by the advanced clustering methods: the HDBSCAN/OPTICS algorithm selection ([`ClusteringAlgorithm`]), distance matrices, silhouette helpers and cluster statistics ([`cluster_stats`]).

use crate::errors::{EconError, EconResult};
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

/// PAM's silhouette widths (`cluster::pam()$silinfo`): `cluster::silhouette` through the one core
/// ([`crate::ml::cluster_validation`]); singletons score 0.
pub(crate) fn compute_silhouette_from_distances(
    distances: &Array2<f64>,
    labels: &[usize],
    _n_clusters: usize,
) -> Vec<f64> {
    crate::ml::cluster_validation::silhouette_core(distances.view(), labels).widths
}

/// Average silhouette width (CLARA's `silinfo$avg.width`); 0 when n < 2 or k < 2.
pub(crate) fn compute_silhouette_avg(labels: &[usize], distances: &Array2<f64>, k: usize) -> f64 {
    let n = labels.len();
    if n < 2 || k < 2 {
        return 0.0;
    }
    crate::ml::cluster_validation::silhouette_core(distances.view(), labels)
        .widths
        .iter()
        .sum::<f64>()
        / n as f64
}

/// Refuse data the distance-based clusterings cannot use: zero columns, non-finite values, and
/// coordinates whose squared distances overflow. Data with no rows pass (the callers check `n`).
pub(crate) fn check_points(data: &ArrayView2<f64>, method: &str) -> EconResult<()> {
    let spec = |message: String| EconError::InvalidSpecification { message };
    if data.nrows() == 0 {
        return Ok(());
    }
    if data.ncols() == 0 {
        return Err(spec(format!("{method}: the data have no columns")));
    }
    let mut bad = 0usize;
    let mut first = None;
    for ((i, j), &v) in data.indexed_iter() {
        if !v.is_finite() {
            bad += 1;
            first.get_or_insert((i, j));
        }
    }
    if let Some((i, j)) = first {
        return Err(spec(format!(
            "{method}: the data contain {bad} non-finite value(s) (first at row {i}, column {j}); remove or impute them"
        )));
    }
    // every squared distance is <= the sum of squared column ranges (monotone rounding), so this
    // bounds them all
    let mut s = 0.0f64;
    for col in data.columns() {
        let (lo, hi) = col
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        s += (hi - lo) * (hi - lo);
    }
    if !s.is_finite() {
        return Err(spec(format!(
            "{method}: squared distances overflow (the coordinate ranges are too large); rescale the data"
        )));
    }
    Ok(())
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

/// Silhouette with per-cluster breakdown (the one core): the mean width over all n, and per label
/// `< k` the mean over all its members (0 for an empty label).
fn compute_silhouette_details(
    labels: &[usize],
    distances: &Array2<f64>,
    k: usize,
) -> (f64, Vec<f64>) {
    let n = labels.len();
    if n < 2 || k < 2 {
        return (0.0, vec![0.0; k]);
    }
    let widths = crate::ml::cluster_validation::silhouette_core(distances.view(), labels).widths;
    let mut sum = vec![0.0; k];
    let mut count = vec![0usize; k];
    for (&l, &w) in labels.iter().zip(&widths) {
        if l < k {
            sum[l] += w;
            count[l] += 1;
        }
    }
    let avg = widths.iter().sum::<f64>() / n as f64;
    let per = sum
        .iter()
        .zip(&count)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();
    (avg, per)
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

/// Shared test data of the HDBSCAN, OPTICS and silhouette tests.
#[cfg(test)]
pub(crate) mod test_data {
    use ndarray::Array2;
    use rand::prelude::*;

    /// blobs_noise.csv (x, y) as an n x 2 matrix, read through the column extractor.
    pub(crate) fn blobs(ds: &crate::data::Dataset) -> Array2<f64> {
        use crate::data::columns::{NaPolicy, numeric_column};
        let x = numeric_column(ds.df(), "x", NaPolicy::Fail).unwrap().0;
        let y = numeric_column(ds.df(), "y", NaPolicy::Fail).unwrap().0;
        Array2::from_shape_fn((x.len(), 2), |(i, j)| if j == 0 { x[i] } else { y[i] })
    }

    /// p2a labels (-1 noise, clusters from 0) as R's codes (0 noise, clusters from 1).
    pub(crate) fn r_codes(labels: &[i32]) -> Vec<f64> {
        labels.iter().map(|&l| f64::from(l + 1)).collect()
    }

    /// Undefined (infinite) distances as NaN, which is how the case JSON's null (R NA) is read.
    pub(crate) fn na(v: &[f64]) -> Vec<f64> {
        v.iter()
            .map(|&x| if x.is_finite() { x } else { f64::NAN })
            .collect()
    }

    /// Seeded uniform points with noise in every coordinate (large-n and degenerate-input tests).
    pub(crate) fn uniform(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, d), |_| rng.r#gen::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    /// R: cluster::silhouette(cutree(hclust(dist(X), "average"), 6), dist(X)) on blobs_noise, through the
    /// PAM (silinfo widths), CLARA (average) and cluster-statistics helpers. Case silhouette_singletons.
    #[test]
    fn test_validate_silhouette_helpers_singletons() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("silhouette_singletons");
        let x = test_data::blobs(&ds);
        let n = x.nrows();
        let labels: Vec<usize> = case
            .vector("labels")
            .iter()
            .map(|&v| v as usize - 1)
            .collect();
        let d = compute_distance_matrix(&x.view());
        let (avg_details, cluster_details) = compute_silhouette_details(&labels, &d, 6);
        let stats = cluster_stats(x.view(), &labels).unwrap();
        Check::new(&case)
            .vector(
                "sil_width",
                &compute_silhouette_from_distances(&d, &labels, 6),
                Tol::coef(n),
            )
            .scalar(
                "avg_width",
                compute_silhouette_avg(&labels, &d, 6),
                Tol::stat(n),
            )
            .vector("cluster_avg", &cluster_details, Tol::stat(n))
            .finish();
        assert!((avg_details - stats.average_silhouette).abs() < 1e-15);
        assert_eq!(stats.cluster_silhouette, cluster_details);
    }

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
