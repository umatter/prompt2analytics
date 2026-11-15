//! Pieces shared by the advanced clustering methods: the HDBSCAN/OPTICS algorithm selection ([`ClusteringAlgorithm`]), distance matrices, silhouette helpers and cluster statistics ([`cluster_stats`]).

use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_zeros2};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Algorithm Selection
// =============================================================================

/// Neighbour-search choice of HDBSCAN and OPTICS. The variants choose only how neighbours are found;
/// HDBSCAN's spanning tree is always Prim's algorithm on the dense mutual-reachability graph, so every
/// variant gives the same result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusteringAlgorithm {
    /// Brute-force neighbour search (n < 500)
    BruteForce,
    /// k-d tree neighbour search (500 <= n < 10,000, d <= 15)
    KdTreePrim,
    /// k-d tree neighbour search (n >= 10,000, d <= 15). A historical name: no Boruvka tree is built.
    DualTreeBoruvka,
    /// Brute-force neighbour search for high-dimensional data, where a k-d tree degrades (d > 15)
    BruteForceParallel,
}

impl ClusteringAlgorithm {
    /// The neighbour search `hdbscan` and `optics` use for `n` rows of `d` columns.
    /// Informational: no function takes it.
    pub fn for_data(n: usize, d: usize) -> ClusteringAlgorithm {
        select_algorithm(n, d)
    }
}

/// Select the neighbour search from the data size.
pub(crate) fn select_algorithm(n: usize, d: usize) -> ClusteringAlgorithm {
    match (n, d) {
        (n, _) if n < 500 => ClusteringAlgorithm::BruteForce,
        (n, d) if n < 10_000 && d <= 15 => ClusteringAlgorithm::KdTreePrim,
        (_, d) if d <= 15 => ClusteringAlgorithm::DualTreeBoruvka,
        _ => ClusteringAlgorithm::BruteForceParallel,
    }
}

/// The k-d tree serves the neighbour searches of this algorithm choice (HDBSCAN's core distances,
/// OPTICS's eps-neighbourhoods).
pub(crate) fn uses_tree(alg: ClusteringAlgorithm) -> bool {
    matches!(
        alg,
        ClusteringAlgorithm::KdTreePrim | ClusteringAlgorithm::DualTreeBoruvka
    )
}

/// Data points copied row-major into one buffer, with the one distance kernel
/// ([`crate::ml::kdtree::sq_euclidean`]).
pub(crate) struct Points {
    pub(crate) n: usize,
    pub(crate) d: usize,
    buf: Vec<f64>,
}

impl Points {
    /// Copy `data` (allocated through the resource guards).
    pub(crate) fn new(data: &ArrayView2<f64>, what: &str) -> EconResult<Points> {
        let (n, d) = data.dim();
        let mut buf =
            crate::resource::try_with_capacity(crate::resource::checked_len(&[n, d], what)?, what)?;
        for row in data.rows() {
            buf.extend(row.iter().copied());
        }
        Ok(Points { n, d, buf })
    }

    /// Row `i`.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[f64] {
        &self.buf[i * self.d..(i + 1) * self.d]
    }

    /// Squared Euclidean distance of rows `i` and `j`.
    #[inline]
    pub(crate) fn sq(&self, i: usize, j: usize) -> f64 {
        crate::ml::kdtree::sq_euclidean(self.row(i), self.row(j))
    }

    /// The rows as vectors (for [`crate::ml::kdtree::KdTree::new`]).
    pub(crate) fn rows_vec(&self) -> Vec<Vec<f64>> {
        (0..self.n).map(|i| self.row(i).to_vec()).collect()
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute pairwise Euclidean distances (tests; product code uses [`try_distance_matrix`]).
#[cfg(test)]
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

/// Pairwise Euclidean distances, with the
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
    // one pass in memory order for the ranges and finiteness; the count and the first position are
    // only gathered on failure
    let d = data.ncols();
    let mut lo = vec![f64::INFINITY; d];
    let mut hi = vec![f64::NEG_INFINITY; d];
    let mut finite = true;
    for row in data.rows() {
        for ((&v, l), h) in row.iter().zip(lo.iter_mut()).zip(hi.iter_mut()) {
            finite &= v.is_finite();
            *l = l.min(v);
            *h = h.max(v);
        }
    }
    if !finite {
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        let (i, j) = data
            .indexed_iter()
            .find(|(_, v)| !v.is_finite())
            .map_or((0, 0), |(ij, _)| ij);
        return Err(spec(format!(
            "{method}: the data contain {bad} non-finite value(s) (first at row {i}, column {j}); remove or impute them"
        )));
    }
    // every squared distance is <= the sum of squared column ranges (monotone rounding), so this
    // bounds them all
    let mut s = 0.0f64;
    for (&l, &h) in lo.iter().zip(&hi) {
        s += (h - l) * (h - l);
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
    /// Number of clusters: the distinct labels
    pub n_clusters: usize,
    /// Cluster sizes, one per distinct label in increasing order of the label (as
    /// `cluster_silhouette` and `cluster_diameters`)
    pub cluster_sizes: Vec<usize>,
    /// Average within-cluster distance: the size-weighted mean of each cluster's mean pairwise
    /// distance, singletons left out (fpc's `average.within`)
    pub average_within: f64,
    /// Average between-cluster distance (mean over pairs in different clusters)
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
    /// Davies-Bouldin index ([`crate::ml::cluster_validation::davies_bouldin`]; 0 for one
    /// cluster)
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
/// evaluating the quality of a clustering solution, as `fpc::cluster.stats(dist(data), labels)`
/// and `clusterCrit::intCriteria` do. Clusters are the distinct labels, whatever their values;
/// per-cluster vectors are in increasing order of the label.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `labels` - Cluster labels for each observation
///
/// # Returns
/// * `ClusterStatsResult` with multiple cluster quality metrics
///
/// # Metrics computed
/// - Silhouette width (`cluster::silhouette`; fpc's `avg.silwidth`, `clus.avg.silwidths`)
/// - Dunn index (minimum separation / maximum diameter)
/// - Calinski-Harabasz index
/// - Davies-Bouldin index (mean distance to the centroid as scatter)
/// - Within/between cluster distances (fpc's `average.within`, `average.between`)
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
    if n == 0 {
        return Err("No clusters found".to_string());
    }

    let compact = crate::ml::cluster_validation::compact_labels(labels, "cluster_stats labels")?;
    let labels = compact.ids.as_slice();
    let k = compact.k;

    let distances = try_distance_matrix(&data, "cluster_stats distance matrix")?;

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
        for j in 0..p {
            cluster_centroids[l][j] /= cluster_sizes[l] as f64;
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

    // One pass over the pairs: per-cluster distance sums and diameters, the between-cluster
    // sum and the separation.
    let mut within_sum = vec![0.0f64; k];
    let mut cluster_diameters: Vec<f64> = vec![0.0; k];
    let mut between_dist_sum = 0.0;
    let mut between_dist_count = 0usize;
    let mut separation = f64::INFINITY;
    for i in 0..n {
        checkpoint().map_err(|e| e.to_string())?;
        let li = labels[i];
        for j in (i + 1)..n {
            let d = distances[[i, j]];
            if li == labels[j] {
                within_sum[li] += d;
                cluster_diameters[li] = cluster_diameters[li].max(d);
            } else {
                between_dist_sum += d;
                between_dist_count += 1;
                separation = separation.min(d);
            }
        }
    }

    // fpc: weighted.mean(average.distance, cluster.size, na.rm = TRUE)
    let (mut weighted, mut weight) = (0.0, 0.0);
    for l in 0..k {
        let m = cluster_sizes[l];
        if m >= 2 {
            let pairs = (m * (m - 1) / 2) as f64;
            weighted += m as f64 * (within_sum[l] / pairs);
            weight += m as f64;
        }
    }
    let average_within = if weight > 0.0 { weighted / weight } else { 0.0 };

    let average_between = if between_dist_count > 0 {
        between_dist_sum / between_dist_count as f64
    } else {
        0.0
    };

    let max_diameter = cluster_diameters.iter().cloned().fold(0.0, f64::max);
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

    // Davies-Bouldin index: the one definition in cluster_validation
    let davies_bouldin = if k >= 2 {
        crate::ml::cluster_validation::davies_bouldin(data, labels)
            .map_err(|e| e.to_string())?
            .db_index
    } else {
        0.0
    };

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

    /// A budget whose token is already cancelled (cancellation tests).
    pub(crate) fn cancelled() -> crate::resource::Budget {
        let t = crate::resource::CancelToken::new();
        t.cancel();
        crate::resource::Budget::new(None, t)
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

    #[test]
    fn test_validate_cluster_stats_blobs() {
        // fpc::cluster.stats(dist(ap_blobs), pam(ap_blobs, 3)$clustering) and clusterCrit.
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("cluster_stats_blobs");
        let x = test_data::blobs(&ds);
        let labels: Vec<usize> = case
            .vector("labels")
            .iter()
            .map(|&v| v as usize - 1)
            .collect();
        let r = cluster_stats(x.view(), &labels).unwrap();
        Check::new(&case)
            .scalar("db", r.davies_bouldin, Tol::stat(115))
            .scalar("ch", r.calinski_harabasz, Tol::stat(115))
            .scalar("dunn", r.dunn_index, Tol::stat(115))
            .scalar("within_ss", r.within_ss, Tol::stat(115))
            .scalar("avg_silwidth", r.average_silhouette, Tol::stat(115))
            .vector(
                "fpc_cluster_avg_silwidths",
                &r.cluster_silhouette,
                Tol::stat(115),
            )
            .scalar("fpc_average_within", r.average_within, Tol::stat(115))
            .scalar("fpc_average_between", r.average_between, Tol::stat(115))
            .scalar("fpc_separation", r.separation, Tol::stat(115))
            .scalar("fpc_max_diameter", r.max_diameter, Tol::stat(115))
            .finish();
        let db = crate::ml::cluster_validation::davies_bouldin(x.view(), &labels).unwrap();
        assert_eq!(r.davies_bouldin.to_bits(), db.db_index.to_bits());
    }

    #[test]
    fn test_for_data_is_select_algorithm() {
        for n in [0, 1, 499, 500, 9_999, 10_000, 50_000] {
            for d in [1, 2, 15, 16, 40] {
                assert_eq!(ClusteringAlgorithm::for_data(n, d), select_algorithm(n, d));
            }
        }
    }

    #[test]
    fn test_cluster_stats_gapped_labels() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.3],
            [0.2, 0.1],
            [10.0, 10.0],
            [10.4, 10.1],
            [5.2, 0.0],
            [5.0, 0.3],
        ];
        let a = cluster_stats(data.view(), &[0, 0, 0, 1, 1, 2, 2]).unwrap();
        let b = cluster_stats(data.view(), &[0, 0, 0, 2, 2, 3, 3]).unwrap();
        assert_eq!(b.n_clusters, 3);
        assert_eq!(a.cluster_sizes, b.cluster_sizes);
        assert_eq!(a.davies_bouldin.to_bits(), b.davies_bouldin.to_bits());
        assert_eq!(a.calinski_harabasz.to_bits(), b.calinski_harabasz.to_bits());
        assert_eq!(a.average_within.to_bits(), b.average_within.to_bits());
        assert_eq!(a.cluster_silhouette, b.cluster_silhouette);
        assert_eq!(a.cluster_diameters, b.cluster_diameters);
    }
}
