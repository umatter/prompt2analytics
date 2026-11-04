//! Cluster validation metrics.
//!
//! Provides internal and external validation metrics for clustering quality assessment.
//!
//! # Internal Validation Metrics
//! - Silhouette coefficient: Measures cluster cohesion vs separation
//! - Calinski-Harabasz index: Variance ratio criterion
//! - Davies-Bouldin index: Average cluster similarity
//! - Dunn index: Ratio of min inter-cluster to max intra-cluster distance
//! - Gap statistic: Compares within-cluster dispersion to null reference
//!
//! # External Validation Metrics
//! - Rand Index: Measures agreement between two clusterings
//! - Adjusted Rand Index: Rand Index corrected for chance
//! - Normalized Mutual Information: Information-theoretic measure

use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Silhouette Coefficient
// =============================================================================

/// Result of silhouette coefficient computation.
///
/// # References
///
/// - Rousseeuw, P.J. (1987). "Silhouettes: A graphical aid to the interpretation
///   and validation of cluster analysis". Journal of Computational and Applied
///   Mathematics, 20, 53-65.
/// - R cluster::silhouette documentation
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/silhouette.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilhouetteResult {
    /// Silhouette width for each point (ranging from -1 to 1)
    pub silhouette_widths: Vec<f64>,
    /// Average silhouette width (overall cluster quality)
    pub average_silhouette: f64,
    /// Cluster assignments (0-indexed)
    pub labels: Vec<usize>,
    /// Average silhouette width per cluster
    pub cluster_silhouettes: Vec<f64>,
    /// For each point: (cluster, neighbor_cluster, a_i, b_i, s_i)
    /// where a_i = avg distance to own cluster, b_i = avg distance to nearest other cluster
    pub silhouette_info: Vec<SilhouetteInfo>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

/// Detailed silhouette information for a single observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilhouetteInfo {
    /// Observation index
    pub index: usize,
    /// Assigned cluster (0-indexed)
    pub cluster: usize,
    /// Nearest neighboring cluster
    pub neighbor_cluster: usize,
    /// Average distance to points in own cluster (a(i))
    pub a_i: f64,
    /// Average distance to points in nearest other cluster (b(i))
    pub b_i: f64,
    /// Silhouette width: (b(i) - a(i)) / max(a(i), b(i))
    pub s_i: f64,
}

impl std::fmt::Display for SilhouetteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Silhouette Analysis")?;
        writeln!(f, "===================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f)?;
        writeln!(
            f,
            "Average Silhouette Width: {:.4}",
            self.average_silhouette
        )?;
        writeln!(f)?;
        writeln!(f, "Per-cluster average silhouette widths:")?;
        for (i, &avg) in self.cluster_silhouettes.iter().enumerate() {
            let count = self.labels.iter().filter(|&&l| l == i).count();
            writeln!(f, "  Cluster {}: {:.4} ({} observations)", i, avg, count)?;
        }
        writeln!(f)?;
        writeln!(f, "Interpretation:")?;
        if self.average_silhouette > 0.7 {
            writeln!(f, "  Strong structure has been found")?;
        } else if self.average_silhouette > 0.5 {
            writeln!(f, "  A reasonable structure has been found")?;
        } else if self.average_silhouette > 0.25 {
            writeln!(f, "  The structure is weak and could be artificial")?;
        } else {
            writeln!(f, "  No substantial structure has been found")?;
        }
        Ok(())
    }
}

/// Compute silhouette coefficient for cluster validation (R `cluster::silhouette`).
///
/// The silhouette value measures how similar an object is to its own cluster
/// compared to other clusters. Ranges from -1 (poor clustering) to +1 (good clustering).
///
/// For each observation i:
/// - a(i) = average distance to other points in same cluster
/// - b(i) = smallest average distance to points in any other cluster (the neighbour cluster is the
///   first minimiser)
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i)), and s(i) = 0 when i is alone in its cluster
///   (Rousseeuw 1987) or a(i) = b(i)
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features), finite values
/// * `labels` - Cluster assignments (0-indexed, each below n)
///
/// # Errors
/// `InvalidSpecification` when the labels do not have between 2 and n - 1 distinct values (where
/// `cluster::silhouette` returns `NA`), for non-finite data, for a length mismatch, or when a label is
/// n or above (labels are 0-indexed cluster indices below n).
///
/// # Returns
/// * `SilhouetteResult` containing individual and average silhouette widths
///
/// # References
///
/// - Rousseeuw, P.J. (1987). "Silhouettes: A graphical aid to the interpretation
///   and validation of cluster analysis". Journal of Computational and Applied
///   Mathematics, 20, 53-65.
pub fn silhouette(data: ArrayView2<f64>, labels: &[usize]) -> EconResult<SilhouetteResult> {
    let n = data.nrows();

    if labels.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Labels length ({}) does not match data rows ({})",
                labels.len(),
                n
            ),
        });
    }

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    crate::ml::advanced_clustering_mod::common::check_points(&data, "silhouette")?;
    check_silhouette_clusters(labels, n)
        .map_err(|message| EconError::InvalidSpecification { message })?;
    let d = compute_pairwise_distances(&data)?;
    Ok(silhouette_result(d.view(), labels))
}

/// Compute silhouette coefficient from a precomputed distance matrix (R `cluster::silhouette(x,
/// dist)`); the same rules as [`silhouette()`].
///
/// # Arguments
/// * `dist_matrix` - Precomputed symmetric distance matrix (n x n), finite; row i gives i's sums
/// * `labels` - Cluster assignments (0-indexed, each below n)
pub fn silhouette_from_dist(
    dist_matrix: ArrayView2<f64>,
    labels: &[usize],
) -> Result<SilhouetteResult, String> {
    let n = dist_matrix.nrows();

    if dist_matrix.ncols() != n {
        return Err("Distance matrix must be square".to_string());
    }

    if labels.len() != n {
        return Err(format!(
            "Labels length ({}) does not match matrix size ({})",
            labels.len(),
            n
        ));
    }

    if n == 0 {
        return Err("Empty data".to_string());
    }

    if dist_matrix.iter().any(|v| !v.is_finite()) {
        return Err("the dissimilarities contain non-finite values".to_string());
    }
    check_silhouette_clusters(labels, n)?;
    Ok(silhouette_result(dist_matrix, labels))
}

/// The silhouette of every observation, as `cluster::silhouette`'s C routine `sildist`.
pub(crate) struct SilhouetteCore {
    /// s(i); 0 for singletons and when a(i) = b(i).
    pub(crate) widths: Vec<f64>,
    /// a(i); 0 for singletons.
    pub(crate) a: Vec<f64>,
    /// b(i); `INFINITY` when no other non-empty cluster exists.
    pub(crate) b: Vec<f64>,
    /// The first minimising cluster of b(i); the own label when none exists.
    pub(crate) neighbor: Vec<usize>,
    /// Members per label `0..=max label`.
    pub(crate) sizes: Vec<usize>,
}

/// The one silhouette (R `cluster::silhouette`, C routine `sildist`): sums over the other members of
/// every cluster in index order; `s(i) = 0` for a singleton (Rousseeuw 1987) and when `a(i) = b(i)`;
/// the neighbour is the first minimising cluster. Row i of `dist` gives i's sums.
pub(crate) fn silhouette_core(dist: ArrayView2<f64>, labels: &[usize]) -> SilhouetteCore {
    let n = labels.len();
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l] += 1;
    }
    let mut widths = vec![0.0f64; n];
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let mut neighbor = vec![0usize; n];
    let put =
        |sums: &mut Vec<f64>,
         (i, (((w, a), b), nb)): (usize, (((&mut f64, &mut f64), &mut f64), &mut usize))| {
            (*w, *a, *b, *nb) = silhouette_row(dist, labels, &sizes, i, sums);
        };
    if n < SILHOUETTE_PAR_MIN_N {
        // one thread pays off only once a row's O(n) sums dwarf the fork-join cost
        let mut sums = vec![0.0f64; k];
        widths
            .iter_mut()
            .zip(a.iter_mut())
            .zip(b.iter_mut())
            .zip(neighbor.iter_mut())
            .enumerate()
            .for_each(|row| put(&mut sums, row));
    } else {
        // written in place: no per-row tuple buffer beside the four outputs
        widths
            .par_iter_mut()
            .zip(a.par_iter_mut())
            .zip(b.par_iter_mut())
            .zip(neighbor.par_iter_mut())
            .enumerate()
            .for_each_init(|| vec![0.0f64; k], put);
    }
    SilhouetteCore {
        widths,
        a,
        b,
        neighbor,
        sizes,
    }
}

/// Below this many observations [`silhouette_core`] runs on the calling thread.
const SILHOUETTE_PAR_MIN_N: usize = 2048;

/// `(s(i), a(i), b(i), neighbour)` of row i; `sums` is a scratch buffer of one slot per label. The
/// sums run over j in index order, skipping j = i.
#[inline]
fn silhouette_row(
    dist: ArrayView2<f64>,
    labels: &[usize],
    sizes: &[usize],
    i: usize,
    sums: &mut [f64],
) -> (f64, f64, f64, usize) {
    sums.iter_mut().for_each(|s| *s = 0.0);
    let row = dist.row(i);
    if let Some(r) = row.as_slice() {
        for (&dij, &lj) in r[..i].iter().zip(&labels[..i]) {
            sums[lj] += dij;
        }
        for (&dij, &lj) in r[i + 1..].iter().zip(&labels[i + 1..]) {
            sums[lj] += dij;
        }
    } else {
        for (j, (&dij, &lj)) in row.iter().zip(labels).enumerate() {
            if j != i {
                sums[lj] += dij;
            }
        }
    }
    let ci = labels[i];
    let a = if sizes[ci] > 1 {
        sums[ci] / (sizes[ci] - 1) as f64
    } else {
        0.0
    };
    let (mut b, mut nb) = (f64::INFINITY, ci);
    for (c, (&sc, &mc)) in sums.iter().zip(sizes).enumerate() {
        if c != ci && mc > 0 {
            let m = sc / mc as f64;
            if m < b {
                b = m;
                nb = c;
            }
        }
    }
    // Rousseeuw (1987): s(i) = 0 for a singleton; sildist also gives 0 when a(i) = b(i)
    let s = if sizes[ci] == 1 || b.is_infinite() || a == b {
        0.0
    } else {
        (b - a) / a.max(b)
    };
    (s, a, b, nb)
}

/// `2 <= distinct labels <= n - 1` (cluster::silhouette returns NA otherwise).
pub(crate) fn check_silhouette_clusters(labels: &[usize], n: usize) -> Result<(), String> {
    let mut seen: Vec<bool> = Vec::new();
    let mut k = 0usize;
    for &l in labels {
        if l >= n {
            return Err(format!(
                "silhouette: labels must be below n = {n} (got {l})"
            ));
        }
        if l >= seen.len() {
            seen.resize(l + 1, false);
        }
        if !seen[l] {
            seen[l] = true;
            k += 1;
        }
    }
    if k < 2 || k + 1 > n {
        return Err(format!(
            "silhouette needs between 2 and n - 1 = {} non-empty clusters (got {k}); cluster::silhouette returns NA otherwise",
            n.saturating_sub(1)
        ));
    }
    Ok(())
}

/// The [`SilhouetteResult`] of a dissimilarity matrix: per-cluster means over all members (0 for an
/// absent label).
fn silhouette_result(dist: ArrayView2<f64>, labels: &[usize]) -> SilhouetteResult {
    let n = labels.len();
    let core = silhouette_core(dist, labels);
    let n_clusters = core.sizes.len();
    let mut sums = vec![0.0f64; n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        sums[l] += core.widths[i];
    }
    let cluster_silhouettes = sums
        .iter()
        .zip(&core.sizes)
        .map(|(&s, &m)| if m > 0 { s / m as f64 } else { 0.0 })
        .collect();
    let silhouette_info = (0..n)
        .map(|i| SilhouetteInfo {
            index: i,
            cluster: labels[i],
            neighbor_cluster: core.neighbor[i],
            a_i: core.a[i],
            b_i: core.b[i],
            s_i: core.widths[i],
        })
        .collect();
    let average_silhouette = core.widths.iter().sum::<f64>() / n as f64;
    SilhouetteResult {
        silhouette_widths: core.widths,
        average_silhouette,
        labels: labels.to_vec(),
        cluster_silhouettes,
        silhouette_info,
        n_clusters,
        n,
    }
}

/// Convenience wrapper for silhouette.
pub fn run_silhouette(data: ArrayView2<f64>, labels: &[usize]) -> EconResult<SilhouetteResult> {
    silhouette(data, labels)
}

// =============================================================================
// Calinski-Harabasz Index (Variance Ratio Criterion)
// =============================================================================

/// Result of Calinski-Harabasz index computation.
///
/// # References
///
/// - Calinski, T. and Harabasz, J. (1974). "A dendrite method for cluster analysis".
///   Communications in Statistics, 3, 1-27.
/// - R fpc::calinhara documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalinskiHarabaszResult {
    /// Calinski-Harabasz index (higher is better)
    pub ch_index: f64,
    /// Between-group sum of squares
    pub between_ss: f64,
    /// Within-group sum of squares
    pub within_ss: f64,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for CalinskiHarabaszResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Calinski-Harabasz Index")?;
        writeln!(f, "=======================")?;
        writeln!(f, "CH Index: {:.4}", self.ch_index)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(f, "Between-group SS: {:.4}", self.between_ss)?;
        writeln!(f, "Within-group SS: {:.4}", self.within_ss)?;
        Ok(())
    }
}

/// Compute the Calinski-Harabasz index (Variance Ratio Criterion).
///
/// The index is defined as:
/// CH = (between_SS / (k-1)) / (within_SS / (n-k))
///
/// Higher values indicate better-defined clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (0-indexed)
///
/// # Returns
/// * `CalinskiHarabaszResult` containing the CH index and components
///
/// # References
///
/// - Calinski, T. and Harabasz, J. (1974). "A dendrite method for cluster analysis".
pub fn calinski_harabasz(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<CalinskiHarabaszResult, String> {
    let n = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n {
        return Err(format!(
            "Labels length ({}) does not match data rows ({})",
            labels.len(),
            n
        ));
    }

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let n_clusters = *labels.iter().max().unwrap_or(&0) + 1;

    if n_clusters < 2 {
        return Err("Need at least 2 clusters for Calinski-Harabasz index".to_string());
    }

    if n <= n_clusters {
        return Err("Number of observations must be greater than number of clusters".to_string());
    }

    // Compute overall centroid
    let overall_centroid: Array1<f64> = data.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;

    // Group points by cluster and compute cluster centroids
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        cluster_indices[label].push(i);
    }

    let mut cluster_centroids: Vec<Array1<f64>> = Vec::with_capacity(n_clusters);
    for c in 0..n_clusters {
        if cluster_indices[c].is_empty() {
            cluster_centroids.push(Array1::zeros(n_features));
        } else {
            let mut centroid = Array1::zeros(n_features);
            for &i in &cluster_indices[c] {
                centroid += &data.row(i);
            }
            centroid /= cluster_indices[c].len() as f64;
            cluster_centroids.push(centroid);
        }
    }

    // Compute between-group sum of squares
    let mut between_ss = 0.0;
    for (c, centroid) in cluster_centroids.iter().enumerate() {
        let n_c = cluster_indices[c].len() as f64;
        if n_c > 0.0 {
            let diff = centroid - &overall_centroid;
            between_ss += n_c * diff.dot(&diff);
        }
    }

    // Compute within-group sum of squares
    let mut within_ss = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        let centroid = &cluster_centroids[label];
        let point = data.row(i);
        let diff = &point - centroid;
        within_ss += diff.dot(&diff);
    }

    // Compute CH index
    let k = n_clusters as f64;
    let n_f = n as f64;

    let ch_index = if within_ss > 0.0 {
        (between_ss / (k - 1.0)) / (within_ss / (n_f - k))
    } else {
        f64::INFINITY // Perfect clustering
    };

    Ok(CalinskiHarabaszResult {
        ch_index,
        between_ss,
        within_ss,
        n_clusters,
        n,
    })
}

/// Convenience wrapper for calinski_harabasz.
pub fn run_calinski_harabasz(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<CalinskiHarabaszResult, String> {
    calinski_harabasz(data, labels)
}

// =============================================================================
// Davies-Bouldin Index
// =============================================================================

/// Result of Davies-Bouldin index computation.
///
/// # References
///
/// - Davies, D.L. and Bouldin, D.W. (1979). "A Cluster Separation Measure".
///   IEEE Transactions on Pattern Analysis and Machine Intelligence, 1(2), 224-227.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaviesBouldinResult {
    /// Davies-Bouldin index (lower is better)
    pub db_index: f64,
    /// Cluster dispersions (average distance to centroid)
    pub dispersions: Vec<f64>,
    /// Cluster centroids
    pub centroids: Vec<Vec<f64>>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for DaviesBouldinResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Davies-Bouldin Index")?;
        writeln!(f, "====================")?;
        writeln!(f, "DB Index: {:.4}", self.db_index)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(f, "Interpretation: Lower values indicate better clustering")?;
        writeln!(f)?;
        writeln!(f, "Cluster dispersions:")?;
        for (i, &d) in self.dispersions.iter().enumerate() {
            writeln!(f, "  Cluster {}: {:.4}", i, d)?;
        }
        Ok(())
    }
}

/// Compute the Davies-Bouldin index.
///
/// The index is the average similarity between each cluster and its most similar cluster,
/// where similarity = (dispersion_i + dispersion_j) / distance(centroid_i, centroid_j).
///
/// Lower values indicate better-separated clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (0-indexed)
///
/// # Returns
/// * `DaviesBouldinResult` containing the DB index
///
/// # References
///
/// - Davies, D.L. and Bouldin, D.W. (1979). "A Cluster Separation Measure".
pub fn davies_bouldin(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<DaviesBouldinResult, String> {
    let n = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n {
        return Err(format!(
            "Labels length ({}) does not match data rows ({})",
            labels.len(),
            n
        ));
    }

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let n_clusters = *labels.iter().max().unwrap_or(&0) + 1;

    if n_clusters < 2 {
        return Err("Need at least 2 clusters for Davies-Bouldin index".to_string());
    }

    // Group points by cluster
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        cluster_indices[label].push(i);
    }

    // Compute cluster centroids
    let mut centroids: Vec<Array1<f64>> = Vec::with_capacity(n_clusters);
    for c in 0..n_clusters {
        if cluster_indices[c].is_empty() {
            centroids.push(Array1::zeros(n_features));
        } else {
            let mut centroid = Array1::zeros(n_features);
            for &i in &cluster_indices[c] {
                centroid += &data.row(i);
            }
            centroid /= cluster_indices[c].len() as f64;
            centroids.push(centroid);
        }
    }

    // Compute cluster dispersions (average distance to centroid)
    let mut dispersions = vec![0.0; n_clusters];
    for c in 0..n_clusters {
        if !cluster_indices[c].is_empty() {
            let mut sum = 0.0;
            for &i in &cluster_indices[c] {
                let diff = &data.row(i) - &centroids[c];
                sum += diff.dot(&diff).sqrt();
            }
            dispersions[c] = sum / cluster_indices[c].len() as f64;
        }
    }

    // Compute DB index
    let mut db_sum = 0.0;
    for i in 0..n_clusters {
        if cluster_indices[i].is_empty() {
            continue;
        }

        let mut max_r = 0.0;
        for j in 0..n_clusters {
            if i == j || cluster_indices[j].is_empty() {
                continue;
            }

            // Distance between centroids
            let diff = &centroids[i] - &centroids[j];
            let centroid_dist = diff.dot(&diff).sqrt();

            if centroid_dist > 0.0 {
                let r_ij = (dispersions[i] + dispersions[j]) / centroid_dist;
                max_r = f64::max(max_r, r_ij);
            }
        }

        db_sum += max_r;
    }

    let non_empty = cluster_indices.iter().filter(|v| !v.is_empty()).count();
    let db_index = if non_empty > 0 {
        db_sum / non_empty as f64
    } else {
        0.0
    };

    Ok(DaviesBouldinResult {
        db_index,
        dispersions,
        centroids: centroids.iter().map(|c| c.to_vec()).collect(),
        n_clusters,
        n,
    })
}

/// Convenience wrapper for davies_bouldin.
pub fn run_davies_bouldin(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<DaviesBouldinResult, String> {
    davies_bouldin(data, labels)
}

// =============================================================================
// Dunn Index
// =============================================================================

/// Result of Dunn index computation.
///
/// # References
///
/// - Dunn, J.C. (1973). "A Fuzzy Relative of the ISODATA Process and Its Use in
///   Detecting Compact Well-Separated Clusters". Journal of Cybernetics, 3(3), 32-57.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DunnIndexResult {
    /// Dunn index (higher is better)
    pub dunn_index: f64,
    /// Minimum inter-cluster distance
    pub min_inter_cluster_dist: f64,
    /// Maximum intra-cluster diameter
    pub max_intra_cluster_diameter: f64,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for DunnIndexResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dunn Index")?;
        writeln!(f, "==========")?;
        writeln!(f, "Dunn Index: {:.4}", self.dunn_index)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(
            f,
            "Min inter-cluster distance: {:.4}",
            self.min_inter_cluster_dist
        )?;
        writeln!(
            f,
            "Max intra-cluster diameter: {:.4}",
            self.max_intra_cluster_diameter
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Interpretation: Higher values indicate better clustering"
        )?;
        Ok(())
    }
}

/// Compute the Dunn index.
///
/// The Dunn index is defined as:
/// D = min(inter-cluster distance) / max(intra-cluster diameter)
///
/// Higher values indicate compact and well-separated clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (0-indexed)
///
/// # Returns
/// * `DunnIndexResult` containing the Dunn index
///
/// # References
///
/// - Dunn, J.C. (1973). "A Fuzzy Relative of the ISODATA Process".
pub fn dunn_index(data: ArrayView2<f64>, labels: &[usize]) -> EconResult<DunnIndexResult> {
    let n = data.nrows();

    if labels.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Labels length ({}) does not match data rows ({})",
                labels.len(),
                n
            ),
        });
    }

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    let n_clusters = *labels.iter().max().unwrap_or(&0) + 1;

    if n_clusters < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 clusters for Dunn index".to_string(),
        });
    }

    // Group points by cluster
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        cluster_indices[label].push(i);
    }

    // Compute pairwise distances
    let distances = compute_pairwise_distances(&data)?;

    // Compute minimum inter-cluster distance
    let mut min_inter_dist = f64::INFINITY;
    for i in 0..n_clusters {
        if cluster_indices[i].is_empty() {
            continue;
        }
        for j in (i + 1)..n_clusters {
            if cluster_indices[j].is_empty() {
                continue;
            }

            // Minimum distance between any two points in different clusters
            for &pi in &cluster_indices[i] {
                for &pj in &cluster_indices[j] {
                    min_inter_dist = min_inter_dist.min(distances[[pi, pj]]);
                }
            }
        }
    }

    // Compute maximum intra-cluster diameter
    let mut max_intra_diameter = 0.0;
    for c in 0..n_clusters {
        if cluster_indices[c].len() < 2 {
            continue;
        }

        // Maximum distance between any two points in the same cluster
        for i in 0..cluster_indices[c].len() {
            for j in (i + 1)..cluster_indices[c].len() {
                let pi = cluster_indices[c][i];
                let pj = cluster_indices[c][j];
                max_intra_diameter = f64::max(max_intra_diameter, distances[[pi, pj]]);
            }
        }
    }

    let dunn = if max_intra_diameter > 0.0 {
        min_inter_dist / max_intra_diameter
    } else {
        f64::INFINITY // Perfect clustering with singleton clusters
    };

    Ok(DunnIndexResult {
        dunn_index: dunn,
        min_inter_cluster_dist: min_inter_dist,
        max_intra_cluster_diameter: max_intra_diameter,
        n_clusters,
        n,
    })
}

/// Convenience wrapper for dunn_index.
pub fn run_dunn_index(data: ArrayView2<f64>, labels: &[usize]) -> EconResult<DunnIndexResult> {
    dunn_index(data, labels)
}

// =============================================================================
// Rand Index and Adjusted Rand Index
// =============================================================================

/// Result of Rand index computation.
///
/// # References
///
/// - Rand, W.M. (1971). "Objective criteria for the evaluation of clustering methods".
///   Journal of the American Statistical Association, 66, 846-850.
/// - Hubert, L. and Arabie, P. (1985). "Comparing partitions". Journal of
///   Classification, 2, 193-218.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandIndexResult {
    /// Rand Index (0 to 1, higher is better)
    pub rand_index: f64,
    /// Adjusted Rand Index (-1 to 1, accounting for chance)
    pub adjusted_rand_index: f64,
    /// Number of pairs in agreement (both same or both different)
    pub pairs_agreement: usize,
    /// Total number of pairs
    pub total_pairs: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for RandIndexResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rand Index")?;
        writeln!(f, "==========")?;
        writeln!(f, "Rand Index: {:.4}", self.rand_index)?;
        writeln!(f, "Adjusted Rand Index: {:.4}", self.adjusted_rand_index)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(
            f,
            "Pairs in agreement: {} / {}",
            self.pairs_agreement, self.total_pairs
        )?;
        writeln!(f)?;
        writeln!(f, "Interpretation:")?;
        writeln!(f, "  RI = 1: Perfect agreement")?;
        writeln!(f, "  ARI = 1: Perfect agreement")?;
        writeln!(f, "  ARI = 0: Random clustering")?;
        writeln!(f, "  ARI < 0: Worse than random")?;
        Ok(())
    }
}

/// Compute Rand Index and Adjusted Rand Index between two clusterings.
///
/// The Rand Index measures the similarity between two clusterings by considering
/// all pairs of samples and counting pairs that are assigned in the same or
/// different clusters in both clusterings.
///
/// The Adjusted Rand Index corrects for chance.
///
/// # Arguments
/// * `labels_true` - Ground truth cluster assignments
/// * `labels_pred` - Predicted cluster assignments
///
/// # Returns
/// * `RandIndexResult` containing RI and ARI
///
/// # References
///
/// - Rand, W.M. (1971). "Objective criteria for the evaluation of clustering methods".
/// - Hubert, L. and Arabie, P. (1985). "Comparing partitions".
pub fn rand_index(labels_true: &[usize], labels_pred: &[usize]) -> EconResult<RandIndexResult> {
    let n = labels_true.len();

    if labels_pred.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Labels length mismatch: {} vs {}",
                labels_true.len(),
                labels_pred.len()
            ),
        });
    }

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "Rand index".to_string(),
        });
    }

    // Build contingency table
    let n_clusters_true = *labels_true.iter().max().unwrap_or(&0) + 1;
    let n_clusters_pred = *labels_pred.iter().max().unwrap_or(&0) + 1;

    let mut contingency = vec![vec![0usize; n_clusters_pred]; n_clusters_true];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    // Compute sums
    let row_sums: Vec<usize> = contingency.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..n_clusters_pred)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Compute combinations
    fn comb2(x: usize) -> f64 {
        if x < 2 {
            0.0
        } else {
            (x * (x - 1)) as f64 / 2.0
        }
    }

    let n_comb = comb2(n);

    // Sum of n_ij choose 2
    let sum_nij: f64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| comb2(x))
        .sum();

    // Sum of a_i choose 2 and b_j choose 2
    let sum_a: f64 = row_sums.iter().map(|&x| comb2(x)).sum();
    let sum_b: f64 = col_sums.iter().map(|&x| comb2(x)).sum();

    // Rand Index
    // RI = (same in both + different in both) / total pairs
    // same in both = sum of n_ij choose 2
    // Using the formula: RI = 1 - (sum_a + sum_b - 2*sum_nij) / n_comb
    let rand_idx = 1.0 - (sum_a + sum_b - 2.0 * sum_nij) / n_comb;

    // Adjusted Rand Index
    // ARI = (sum_nij - expected) / (max - expected)
    // expected = sum_a * sum_b / n_comb
    let expected = sum_a * sum_b / n_comb;
    let max_index = (sum_a + sum_b) / 2.0;

    let ari = if (max_index - expected).abs() < 1e-12 {
        if (sum_nij - expected).abs() < 1e-12 {
            1.0
        } else {
            0.0
        }
    } else {
        (sum_nij - expected) / (max_index - expected)
    };

    // Count pairs in agreement (for RI)
    let pairs_agreement = (rand_idx * n_comb).round() as usize;

    Ok(RandIndexResult {
        rand_index: rand_idx,
        adjusted_rand_index: ari,
        pairs_agreement,
        total_pairs: n_comb as usize,
        n,
    })
}

/// Convenience wrapper for rand_index.
pub fn run_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> EconResult<RandIndexResult> {
    rand_index(labels_true, labels_pred)
}

// =============================================================================
// Normalized Mutual Information
// =============================================================================

/// Result of Normalized Mutual Information computation.
///
/// # References
///
/// - Strehl, A. and Ghosh, J. (2002). "Cluster ensembles - a knowledge reuse
///   framework for combining multiple partitions". Journal of Machine Learning
///   Research, 3, 583-617.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NmiResult {
    /// Normalized Mutual Information (0 to 1)
    pub nmi: f64,
    /// Mutual Information (unnormalized)
    pub mi: f64,
    /// Entropy of first clustering
    pub entropy_true: f64,
    /// Entropy of second clustering
    pub entropy_pred: f64,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for NmiResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Normalized Mutual Information")?;
        writeln!(f, "==============================")?;
        writeln!(f, "NMI: {:.4}", self.nmi)?;
        writeln!(f, "MI: {:.4}", self.mi)?;
        writeln!(f, "H(true): {:.4}", self.entropy_true)?;
        writeln!(f, "H(pred): {:.4}", self.entropy_pred)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(f, "Interpretation:")?;
        writeln!(f, "  NMI = 1: Perfect agreement")?;
        writeln!(f, "  NMI = 0: Independent clusterings")?;
        Ok(())
    }
}

/// Compute Normalized Mutual Information between two clusterings.
///
/// NMI measures the mutual information between two clusterings, normalized
/// by the geometric mean of their entropies.
///
/// # Arguments
/// * `labels_true` - Ground truth cluster assignments
/// * `labels_pred` - Predicted cluster assignments
/// * `average_method` - Normalization method: "geometric" (default), "arithmetic", "min", "max"
///
/// # Returns
/// * `NmiResult` containing NMI and components
///
/// # References
///
/// - Strehl, A. and Ghosh, J. (2002). "Cluster ensembles".
pub fn nmi(
    labels_true: &[usize],
    labels_pred: &[usize],
    average_method: Option<&str>,
) -> EconResult<NmiResult> {
    let n = labels_true.len();

    if labels_pred.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Labels length mismatch: {} vs {}",
                labels_true.len(),
                labels_pred.len()
            ),
        });
    }

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    let method = average_method.unwrap_or("geometric");

    // Build contingency table
    let n_clusters_true = *labels_true.iter().max().unwrap_or(&0) + 1;
    let n_clusters_pred = *labels_pred.iter().max().unwrap_or(&0) + 1;

    let mut contingency = vec![vec![0usize; n_clusters_pred]; n_clusters_true];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    // Compute marginal probabilities
    let row_sums: Vec<usize> = contingency.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..n_clusters_pred)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    let n_f = n as f64;

    // Compute entropies
    let entropy_true: f64 = row_sums
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| {
            let p = x as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    let entropy_pred: f64 = col_sums
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| {
            let p = x as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    // Compute mutual information
    let mut mi = 0.0;
    for i in 0..n_clusters_true {
        for j in 0..n_clusters_pred {
            let n_ij = contingency[i][j];
            if n_ij > 0 && row_sums[i] > 0 && col_sums[j] > 0 {
                let p_ij = n_ij as f64 / n_f;
                let p_i = row_sums[i] as f64 / n_f;
                let p_j = col_sums[j] as f64 / n_f;
                mi += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }

    // Normalize
    let normalizer = match method {
        "geometric" => (entropy_true * entropy_pred).sqrt(),
        "arithmetic" => (entropy_true + entropy_pred) / 2.0,
        "min" => entropy_true.min(entropy_pred),
        "max" => entropy_true.max(entropy_pred),
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!("Unknown average method: {}", method),
            });
        }
    };

    let nmi_value = if normalizer > 0.0 {
        mi / normalizer
    } else {
        1.0 // Both clusterings are singletons
    };

    Ok(NmiResult {
        nmi: nmi_value,
        mi,
        entropy_true,
        entropy_pred,
        n,
    })
}

/// Convenience wrapper for nmi.
pub fn run_nmi(labels_true: &[usize], labels_pred: &[usize]) -> EconResult<NmiResult> {
    nmi(labels_true, labels_pred, None)
}

// =============================================================================
// Gap Statistic
// =============================================================================

use rand::prelude::*;

/// Result of gap statistic computation.
///
/// # References
///
/// - Tibshirani, R., Walther, G., and Hastie, T. (2001). "Estimating the number
///   of clusters in a data set via the gap statistic". Journal of the Royal
///   Statistical Society B, 63(2), 411-423.
/// - R cluster::clusGap documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapStatisticResult {
    /// Gap statistic for each k value tested
    pub gap: Vec<f64>,
    /// Standard error of gap for each k
    pub se: Vec<f64>,
    /// log(W_k) - pooled within-cluster dispersion
    pub log_w: Vec<f64>,
    /// E[log(W_k)] - expected log dispersion under null
    pub e_log_w: Vec<f64>,
    /// k values tested
    pub k_values: Vec<usize>,
    /// Optimal k determined by gap criterion
    pub optimal_k: usize,
    /// Number of bootstrap samples used
    pub n_boot: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for GapStatisticResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gap Statistic Analysis")?;
        writeln!(f, "======================")?;
        writeln!(f, "Optimal number of clusters: {}", self.optimal_k)?;
        writeln!(f, "Bootstrap samples: {}", self.n_boot)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f)?;
        writeln!(f, "  k     Gap(k)     SE(k)    log(W_k)  E[log(W_k)]")?;
        for i in 0..self.k_values.len() {
            writeln!(
                f,
                "  {:2}    {:7.4}    {:6.4}    {:7.4}    {:7.4}",
                self.k_values[i], self.gap[i], self.se[i], self.log_w[i], self.e_log_w[i]
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Selection criterion: Gap(k) >= Gap(k+1) - SE(k+1)")?;
        Ok(())
    }
}

/// Compute the gap statistic for determining optimal number of clusters.
///
/// The gap statistic compares the total within-cluster variation for different
/// values of k with their expected values under a null reference distribution
/// of the data.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `k_max` - Maximum number of clusters to test
/// * `n_boot` - Number of bootstrap samples (default: 50)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `GapStatisticResult` containing gap values and optimal k
///
/// # Algorithm
///
/// For each k = 1, ..., k_max:
/// 1. Cluster the data and compute W_k (pooled within-cluster dispersion)
/// 2. Generate B reference datasets (uniform over bounding box)
/// 3. Cluster each reference dataset and compute W_kb*
/// 4. Gap(k) = E[log(W_kb*)] - log(W_k)
/// 5. Choose smallest k such that Gap(k) >= Gap(k+1) - SE(k+1)
///
/// # References
///
/// - Tibshirani, R., Walther, G., and Hastie, T. (2001). "Estimating the number
///   of clusters in a data set via the gap statistic".
pub fn gap_statistic(
    data: ArrayView2<f64>,
    k_max: usize,
    n_boot: Option<usize>,
    seed: Option<u64>,
) -> Result<GapStatisticResult, String> {
    let n = data.nrows();
    let n_features = data.ncols();
    let b = n_boot.unwrap_or(50);

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    if k_max < 1 {
        return Err("k_max must be at least 1".to_string());
    }

    if k_max > n {
        return Err(format!("k_max ({}) cannot exceed n ({})", k_max, n));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Compute data range for generating reference distributions
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];
    for i in 0..n {
        for j in 0..n_features {
            mins[j] = mins[j].min(data[[i, j]]);
            maxs[j] = maxs[j].max(data[[i, j]]);
        }
    }

    let k_values: Vec<usize> = (1..=k_max).collect();
    let mut log_w = Vec::with_capacity(k_max);
    let mut e_log_w = Vec::with_capacity(k_max);
    let mut se = Vec::with_capacity(k_max);
    let mut gap = Vec::with_capacity(k_max);

    for &k in &k_values {
        // Cluster the actual data and compute log(W_k)
        let w_k = compute_pooled_within_cluster_dispersion(&data, k, Some(rng.r#gen()))?;
        let log_w_k = w_k.ln();
        log_w.push(log_w_k);

        // Generate B reference datasets and compute their log(W_kb*)
        let mut log_w_b = Vec::with_capacity(b);
        for _ in 0..b {
            // Generate uniform reference data
            let mut ref_data = Array2::zeros((n, n_features));
            for i in 0..n {
                for j in 0..n_features {
                    ref_data[[i, j]] = rng.gen_range(mins[j]..=maxs[j]);
                }
            }

            let w_b =
                compute_pooled_within_cluster_dispersion(&ref_data.view(), k, Some(rng.r#gen()))?;
            log_w_b.push(w_b.ln());
        }

        // Compute E[log(W_kb*)] and standard error
        let mean_log_w_b: f64 = log_w_b.iter().sum::<f64>() / b as f64;
        e_log_w.push(mean_log_w_b);

        let variance: f64 = log_w_b
            .iter()
            .map(|&x| (x - mean_log_w_b).powi(2))
            .sum::<f64>()
            / b as f64;
        let sd = variance.sqrt();
        let se_k = sd * (1.0 + 1.0 / b as f64).sqrt();
        se.push(se_k);

        // Gap(k) = E[log(W_kb*)] - log(W_k)
        let gap_k = mean_log_w_b - log_w_k;
        gap.push(gap_k);
    }

    // Determine optimal k using the gap criterion
    // Choose smallest k such that Gap(k) >= Gap(k+1) - SE(k+1)
    let mut optimal_k = k_max;
    for i in 0..(k_max - 1) {
        if gap[i] >= gap[i + 1] - se[i + 1] {
            optimal_k = k_values[i];
            break;
        }
    }

    Ok(GapStatisticResult {
        gap,
        se,
        log_w,
        e_log_w,
        k_values,
        optimal_k,
        n_boot: b,
        n,
    })
}

/// Compute pooled within-cluster sum of squares (dispersion).
fn compute_pooled_within_cluster_dispersion(
    data: &ArrayView2<f64>,
    k: usize,
    seed: Option<u64>,
) -> Result<f64, String> {
    // Use kmeans to cluster data
    let result = super::super::ml::kmeans(data.view(), k, Some(100), Some(1e-4), Some(3), seed)?;

    // Inertia is the within-cluster sum of squares
    Ok(result.inertia)
}

/// Convenience wrapper for gap_statistic.
pub fn run_gap_statistic(
    data: ArrayView2<f64>,
    k_max: usize,
    n_boot: Option<usize>,
    seed: Option<u64>,
) -> Result<GapStatisticResult, String> {
    gap_statistic(data, k_max, n_boot, seed)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Pairwise Euclidean distances (the one kernel [`crate::ml::kdtree::sq_euclidean`]), with the
/// n x n matrix allocated through the resource guards.
fn compute_pairwise_distances(data: &ArrayView2<f64>) -> EconResult<Array2<f64>> {
    let n = data.nrows();
    let d = data.ncols();
    let mut out = crate::resource::try_vec(
        crate::resource::checked_len(&[n, n], "silhouette dissimilarity matrix")?,
        0.0f64,
        "silhouette dissimilarity matrix",
    )?;
    // one contiguous row-major copy only when the view is not already one
    let owned: Vec<f64>;
    let flat: &[f64] = match data.as_slice() {
        Some(s) => s,
        None => {
            owned = data.iter().copied().collect();
            &owned
        }
    };
    // a matrix this small stays in cache, so the mirrored write goes in the same loop; above it the
    // upper triangle is written row by row and mirrored in cache-sized tiles (the column writes of
    // the plain loop stride by n)
    const MIRROR_TILED_MIN_N: usize = 256;
    const TILE: usize = 64;
    if d > 0 && n < MIRROR_TILED_MIN_N {
        for (i, ri) in flat.chunks_exact(d).enumerate() {
            crate::resource::checkpoint()?;
            for (j, rj) in flat.chunks_exact(d).enumerate().skip(i + 1) {
                let dist = crate::ml::kdtree::sq_euclidean(ri, rj).sqrt();
                out[i * n + j] = dist;
                out[j * n + i] = dist;
            }
        }
    } else if d > 0 {
        for (i, ri) in flat.chunks_exact(d).enumerate() {
            crate::resource::checkpoint()?;
            let row = &mut out[i * n..(i + 1) * n];
            for (o, rj) in row[i + 1..]
                .iter_mut()
                .zip(flat.chunks_exact(d).skip(i + 1))
            {
                *o = crate::ml::kdtree::sq_euclidean(ri, rj).sqrt();
            }
        }
        for bi in (0..n).step_by(TILE) {
            crate::resource::checkpoint()?;
            for bj in (bi..n).step_by(TILE) {
                for i in bi..(bi + TILE).min(n) {
                    for j in bj.max(i + 1)..(bj + TILE).min(n) {
                        out[j * n + i] = out[i * n + j];
                    }
                }
            }
        }
    }
    Array2::from_shape_vec((n, n), out).map_err(|e| EconError::Computation(e.to_string()))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::advanced_clustering_mod::common::test_data::blobs;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    fn case_labels(case: &p2a_validation::Case) -> Vec<usize> {
        case.vector("labels")
            .iter()
            .map(|&v| v as usize - 1)
            .collect()
    }

    /// R: cluster::silhouette(cutree(hclust(dist(X), "average"), 6), dist(X)) on blobs_noise (n = 115, one
    /// singleton, row 107). validation/r/cases/ml_density_clustering.R, case silhouette_singletons.
    #[test]
    fn test_validate_silhouette_singletons() {
        let (case, ds) = crate::test_support::load_case("silhouette_singletons");
        let x = blobs(&ds);
        let n = x.nrows();
        let labels = case_labels(&case);
        let r = silhouette(x.view(), &labels).unwrap();
        let neighbor: Vec<f64> = r
            .silhouette_info
            .iter()
            .map(|s| (s.neighbor_cluster + 1) as f64)
            .collect();
        let singletons = (0..r.n_clusters)
            .filter(|&c| labels.iter().filter(|&&l| l == c).count() == 1)
            .count();
        Check::new(&case)
            .vector("sil_width", &r.silhouette_widths, Tol::coef(n))
            .scalar("avg_width", r.average_silhouette, Tol::stat(n))
            .vector("cluster_avg", &r.cluster_silhouettes, Tol::stat(n))
            .vector("neighbor", &neighbor, Tol::exact())
            .scalar("n_singletons", singletons as f64, Tol::exact())
            .finish();
    }

    /// The same case through a precomputed dissimilarity matrix.
    #[test]
    fn test_validate_silhouette_from_dist_singletons() {
        let (case, ds) = crate::test_support::load_case("silhouette_singletons");
        let x = blobs(&ds);
        let n = x.nrows();
        let d = compute_pairwise_distances(&x.view()).unwrap();
        let r = silhouette_from_dist(d.view(), &case_labels(&case)).unwrap();
        Check::new(&case)
            .vector("sil_width", &r.silhouette_widths, Tol::coef(n))
            .scalar("avg_width", r.average_silhouette, Tol::stat(n))
            .vector("cluster_avg", &r.cluster_silhouettes, Tol::stat(n))
            .finish();
    }

    /// cluster::silhouette returns NA unless 2 <= k <= n - 1 non-empty clusters; p2a refuses.
    #[test]
    fn test_silhouette_needs_two_to_n_minus_1_clusters() {
        let x = array![[0.0, 0.1], [1.0, 0.0], [5.0, 5.2], [5.1, 4.9]];
        let spec = |r: EconResult<SilhouetteResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        // one non-empty cluster: was Ok(0)
        assert!(spec(silhouette(x.view(), &[1, 1, 1, 1])).contains("between 2 and n - 1 = 3"));
        // all singletons: was Ok(average 1)
        assert!(spec(silhouette(x.view(), &[0, 1, 2, 3])).contains("got 4"));
        // a label at or above n is refused before any buffer is sized by it
        assert!(
            spec(silhouette(x.view(), &[0, 0, 1, usize::MAX]))
                .contains(&format!("labels must be below n = 4 (got {})", usize::MAX))
        );
        let d4 = compute_pairwise_distances(&x.view()).unwrap();
        assert!(
            silhouette_from_dist(d4.view(), &[0, 0, 1, 4])
                .unwrap_err()
                .contains("labels must be below n = 4 (got 4)")
        );
        let bad = array![[0.0, 0.1], [f64::NAN, 0.0], [5.0, 5.2], [5.1, 4.9]];
        assert!(spec(silhouette(bad.view(), &[0, 0, 1, 1])).contains("non-finite"));
        let d = compute_pairwise_distances(&x.view()).unwrap();
        assert!(
            silhouette_from_dist(d.view(), &[0, 1, 2, 3])
                .unwrap_err()
                .contains("between 2 and n - 1")
        );
        // the finding's six points (R: s_6 = 0, average 0.81312650186401625)
        let p = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [20.0, 0.0]
        ];
        let r = silhouette(p.view(), &[0, 0, 0, 1, 1, 2]).unwrap();
        assert_eq!(r.silhouette_widths[5], 0.0);
        assert!((r.average_silhouette - 0.8131265018640162).abs() < 1e-12);
    }

    /// The sequential (n below the threshold) and parallel branches, and the strided-row fallback,
    /// give the bits of a naive sildist loop (sums over j != i in index order).
    #[test]
    fn test_silhouette_core_branches_match_naive_sums() {
        use rand::prelude::*;
        let naive = |d: ArrayView2<f64>, labels: &[usize]| -> Vec<u64> {
            let n = labels.len();
            let k = labels.iter().max().unwrap() + 1;
            let mut sizes = vec![0usize; k];
            labels.iter().for_each(|&l| sizes[l] += 1);
            (0..n)
                .map(|i| {
                    let mut sums = vec![0.0f64; k];
                    for j in 0..n {
                        if j != i {
                            sums[labels[j]] += d[[i, j]];
                        }
                    }
                    let ci = labels[i];
                    let a = if sizes[ci] > 1 {
                        sums[ci] / (sizes[ci] - 1) as f64
                    } else {
                        0.0
                    };
                    let b = (0..k)
                        .filter(|&c| c != ci && sizes[c] > 0)
                        .map(|c| sums[c] / sizes[c] as f64)
                        .fold(f64::INFINITY, f64::min);
                    let s = if sizes[ci] == 1 || b.is_infinite() || a == b {
                        0.0
                    } else {
                        (b - a) / a.max(b)
                    };
                    s.to_bits()
                })
                .collect()
        };
        let bits = |v: &[f64]| v.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(7);
        for n in [37, 301, SILHOUETTE_PAR_MIN_N + 13] {
            let x =
                Array2::from_shape_fn((n, 3), |(i, _)| (i % 4) as f64 * 1.5 + rng.r#gen::<f64>());
            let mut labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
            labels[5] = 4; // a singleton
            let d = compute_pairwise_distances(&x.view()).unwrap();
            // every entry, both triangles across the mirror's tiles, is the row pair's distance
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j {
                        0.0
                    } else {
                        crate::ml::kdtree::sq_euclidean(
                            x.row(i).as_slice().unwrap(),
                            x.row(j).as_slice().unwrap(),
                        )
                        .sqrt()
                    };
                    assert_eq!(d[[i, j]].to_bits(), want.to_bits(), "({i}, {j})");
                }
            }
            let want = naive(d.view(), &labels);
            assert_eq!(bits(&silhouette_core(d.view(), &labels).widths), want);
            if n < 100 {
                // a transposed (strided) view of the symmetric matrix takes the fallback loop
                assert_eq!(bits(&silhouette_core(d.t(), &labels).widths), want);
            }
        }
    }

    #[test]
    fn test_silhouette_basic() {
        // Two well-separated clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let result = silhouette(data.view(), &labels).unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.n_clusters, 2);
        assert!(result.average_silhouette > 0.9); // Well-separated clusters
    }

    #[test]
    fn test_silhouette_poor_clustering() {
        // Poorly separated clusters
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],];
        // Assign alternating clusters (bad clustering)
        let labels = vec![0, 1, 0, 1];

        let result = silhouette(data.view(), &labels).unwrap();

        // Silhouette should be low or negative for poor clustering
        assert!(result.average_silhouette < 0.5);
    }

    #[test]
    fn test_calinski_harabasz_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];
        let labels = vec![0, 0, 1, 1];

        let result = calinski_harabasz(data.view(), &labels).unwrap();

        assert!(result.ch_index > 100.0); // Well-separated clusters
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_davies_bouldin_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];
        let labels = vec![0, 0, 1, 1];

        let result = davies_bouldin(data.view(), &labels).unwrap();

        assert!(result.db_index < 0.1); // Well-separated clusters have low DB
    }

    #[test]
    fn test_dunn_index_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0],];
        let labels = vec![0, 0, 1, 1];

        let result = dunn_index(data.view(), &labels).unwrap();

        assert!(result.dunn_index > 10.0); // Well-separated clusters have high Dunn
    }

    #[test]
    fn test_rand_index_perfect() {
        let labels_true = vec![0, 0, 1, 1, 2, 2];
        let labels_pred = vec![0, 0, 1, 1, 2, 2];

        let result = rand_index(&labels_true, &labels_pred).unwrap();

        assert!((result.rand_index - 1.0).abs() < 1e-10);
        assert!((result.adjusted_rand_index - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rand_index_relabeled() {
        // Same clustering, different labels
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![1, 1, 0, 0]; // Swapped labels

        let result = rand_index(&labels_true, &labels_pred).unwrap();

        assert!((result.rand_index - 1.0).abs() < 1e-10);
        assert!((result.adjusted_rand_index - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nmi_perfect() {
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];

        let result = nmi(&labels_true, &labels_pred, None).unwrap();

        assert!((result.nmi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nmi_relabeled() {
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![1, 1, 0, 0];

        let result = nmi(&labels_true, &labels_pred, None).unwrap();

        assert!((result.nmi - 1.0).abs() < 1e-10);
    }
}
