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
// Label compaction
// =============================================================================

/// Labels mapped to `0..k` in increasing order of the original label.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactLabels {
    /// The compacted label of each observation
    pub(crate) ids: Vec<usize>,
    /// Number of distinct labels
    pub(crate) k: usize,
    /// The original label of each compacted id (increasing)
    pub(crate) levels: Vec<usize>,
}

/// Distinct labels in increasing order mapped to `0..k`. A dense remap table of `max + 1` entries
/// when `max < 2n`, otherwise a sorted, de-duplicated copy; no buffer is sized by a label value
/// beyond `2n`.
pub(crate) fn compact_labels(labels: &[usize], what: &str) -> Result<CompactLabels, String> {
    let n = labels.len();
    let to_string = |e: EconError| e.to_string();
    let max = labels.iter().copied().max().unwrap_or(0);
    let mut ids = crate::resource::try_vec(n, 0usize, what).map_err(to_string)?;
    if max / 2 < n.max(1) {
        let mut remap = crate::resource::try_vec(max + 1, usize::MAX, what).map_err(to_string)?;
        for &l in labels {
            remap[l] = 0;
        }
        let mut levels = Vec::new();
        for (l, r) in remap.iter_mut().enumerate() {
            if *r == 0 {
                *r = levels.len();
                levels.push(l);
            }
        }
        for (id, &l) in ids.iter_mut().zip(labels) {
            *id = remap[l];
        }
        let k = levels.len();
        Ok(CompactLabels { ids, k, levels })
    } else {
        let mut levels = crate::resource::try_vec(n, 0usize, what).map_err(to_string)?;
        levels.copy_from_slice(labels);
        levels.sort_unstable();
        levels.dedup();
        for (id, l) in ids.iter_mut().zip(labels) {
            *id = levels.binary_search(l).unwrap_or(0);
        }
        let k = levels.len();
        Ok(CompactLabels { ids, k, levels })
    }
}

/// The contingency table of two labellings: its nonzero cells `(t, p, count)` in lexicographic
/// order of the labels, the row and column sums (indexed as the cells; zero sums are allowed) and
/// the numbers of distinct labels.
struct Contingency {
    cells: Vec<(usize, usize, usize)>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    k_true: usize,
    k_pred: usize,
}

/// The contingency table of two labellings of equal length. Indexed by the raw labels when the
/// table of `(max_t + 1) (max_p + 1)` entries has at most `4n` of them (one pass, as for
/// contiguous labels), otherwise by the compacted labels, densely or from the sorted pairs. Every
/// path gives the same cells in the same order, so the sums over them are identical.
fn contingency(
    labels_true: &[usize],
    labels_pred: &[usize],
    what: &str,
) -> Result<Contingency, String> {
    let n = labels_true.len();
    let to_string = |e: EconError| e.to_string();
    let max_t = labels_true.iter().copied().max().unwrap_or(0);
    let max_p = labels_pred.iter().copied().max().unwrap_or(0);
    let raw = max_t
        .checked_add(1)
        .zip(max_p.checked_add(1))
        .and_then(|(a, b)| a.checked_mul(b).map(|size| (a, b, size)))
        .filter(|&(_, _, size)| size <= 4 * n.max(1));
    if let Some((kt, kp, size)) = raw {
        let mut table = crate::resource::try_vec(size, 0usize, what).map_err(to_string)?;
        for (&a, &b) in labels_true.iter().zip(labels_pred) {
            table[a * kp + b] += 1;
        }
        let mut rows = crate::resource::try_vec(kt, 0usize, what).map_err(to_string)?;
        let mut cols = crate::resource::try_vec(kp, 0usize, what).map_err(to_string)?;
        let mut cells = Vec::new();
        for (idx, &c) in table.iter().enumerate() {
            if c > 0 {
                let (i, j) = (idx / kp, idx % kp);
                rows[i] += c;
                cols[j] += c;
                cells.push((i, j, c));
            }
        }
        let k_true = rows.iter().filter(|&&r| r > 0).count();
        let k_pred = cols.iter().filter(|&&c| c > 0).count();
        return Ok(Contingency {
            cells,
            rows,
            cols,
            k_true,
            k_pred,
        });
    }
    let t = compact_labels(labels_true, what)?;
    let p = compact_labels(labels_pred, what)?;
    let (cells, rows, cols) = contingency_cells(&t, &p)?;
    Ok(Contingency {
        cells,
        rows,
        cols,
        k_true: t.k,
        k_pred: p.k,
    })
}

/// The nonzero cells `(t, p, count)` of the contingency table of two compacted labellings, in
/// lexicographic order of `(t, p)`, with the row and column sums. Dense when `k_t * k_p <= 4n`,
/// otherwise from the sorted pairs; both give the same cells in the same order.
#[allow(clippy::type_complexity)]
fn contingency_cells(
    t: &CompactLabels,
    p: &CompactLabels,
) -> Result<(Vec<(usize, usize, usize)>, Vec<usize>, Vec<usize>), String> {
    let n = t.ids.len();
    let what = "contingency table";
    let to_string = |e: EconError| e.to_string();
    let mut rows = crate::resource::try_vec(t.k, 0usize, what).map_err(to_string)?;
    let mut cols = crate::resource::try_vec(p.k, 0usize, what).map_err(to_string)?;
    for (&a, &b) in t.ids.iter().zip(&p.ids) {
        rows[a] += 1;
        cols[b] += 1;
    }
    let mut cells = Vec::new();
    match t.k.checked_mul(p.k) {
        Some(size) if size <= 4 * n => {
            let mut table = crate::resource::try_vec(size, 0usize, what).map_err(to_string)?;
            for (&a, &b) in t.ids.iter().zip(&p.ids) {
                table[a * p.k + b] += 1;
            }
            for (idx, &c) in table.iter().enumerate() {
                if c > 0 {
                    cells.push((idx / p.k, idx % p.k, c));
                }
            }
        }
        _ => {
            let mut pairs =
                crate::resource::try_vec(n, (0usize, 0usize), what).map_err(to_string)?;
            for (pair, (&a, &b)) in pairs.iter_mut().zip(t.ids.iter().zip(&p.ids)) {
                *pair = (a, b);
            }
            pairs.sort_unstable();
            for run in pairs.chunk_by(|x, y| x == y) {
                cells.push((run[0].0, run[0].1, run.len()));
            }
        }
    }
    Ok((cells, rows, cols))
}

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
    /// Average silhouette width per cluster, indexed by label id (0 for an id no point has)
    pub cluster_silhouettes: Vec<f64>,
    /// For each point: (cluster, neighbor_cluster, a_i, b_i, s_i)
    /// where a_i = avg distance to own cluster, b_i = avg distance to nearest other cluster
    pub silhouette_info: Vec<SilhouetteInfo>,
    /// Number of clusters: the distinct labels
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
    let n_ids = core.sizes.len();
    let n_clusters = core.sizes.iter().filter(|&&m| m > 0).count();
    let mut sums = vec![0.0f64; n_ids];
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
/// Higher values indicate better-defined clusters. k is the number of distinct labels, so any
/// label values (gaps, 1-based) give the same index (`clusterCrit::intCriteria`,
/// "Calinski_Harabasz").
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (any ids; at least 2 distinct)
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

    let CompactLabels {
        ids: labels,
        k: n_clusters,
        ..
    } = compact_labels(labels, "Calinski-Harabasz labels")?;
    let labels = labels.as_slice();

    if n_clusters < 2 {
        return Err(format!(
            "Calinski-Harabasz index needs at least 2 distinct clusters (got {n_clusters})"
        ));
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
    /// Cluster dispersions (average distance to centroid), one per distinct label in increasing
    /// order of the label
    pub dispersions: Vec<f64>,
    /// Cluster centroids, in the same order
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
/// Lower values indicate better-separated clusters. The dispersion is the mean Euclidean distance
/// to the centroid and the centroid distance Euclidean (`clusterCrit::intCriteria`,
/// "Davies_Bouldin"); clusters are the distinct labels, whatever their values.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (any ids; at least 2 distinct)
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

    let CompactLabels {
        ids: labels,
        k: n_clusters,
        ..
    } = compact_labels(labels, "Davies-Bouldin labels")?;
    let labels = labels.as_slice();

    if n_clusters < 2 {
        return Err(format!(
            "Davies-Bouldin index needs at least 2 distinct clusters (got {n_clusters})"
        ));
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
/// Higher values indicate compact and well-separated clusters. Clusters are the distinct labels,
/// whatever their values (`clusterCrit::intCriteria`, "Dunn").
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments (any ids; at least 2 distinct)
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

    let CompactLabels {
        ids: labels,
        k: n_clusters,
        ..
    } = compact_labels(labels, "Dunn index labels").map_err(EconError::Computation)?;
    let labels = labels.as_slice();

    if n_clusters < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("Dunn index needs at least 2 distinct clusters (got {n_clusters})"),
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

    // Contingency table of the compacted labels (any label values)
    let Contingency {
        cells,
        rows: row_sums,
        cols: col_sums,
        ..
    } = contingency(labels_true, labels_pred, "Rand index labels")
        .map_err(EconError::Computation)?;

    // Compute combinations
    fn comb2(x: usize) -> f64 {
        if x < 2 {
            0.0
        } else {
            (x * (x - 1)) as f64 / 2.0
        }
    }

    let n_comb = comb2(n);

    // Sum of n_ij choose 2 (nonzero cells)
    let sum_nij: f64 = cells.iter().map(|&(_, _, x)| comb2(x)).sum();

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
/// by the geometric mean of their entropies (or another average of them). Labels may take any
/// values. When both clusterings have one cluster NMI is 1; otherwise, when the mutual
/// information is 0 (in particular when exactly one clustering has one cluster), NMI is 0, as
/// scikit-learn's `normalized_mutual_info_score` (aricode's variants disagree with each other
/// there).
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

    // Contingency table of the compacted labels (any label values)
    let Contingency {
        cells,
        rows: row_sums,
        cols: col_sums,
        k_true,
        k_pred,
    } = contingency(labels_true, labels_pred, "NMI labels").map_err(EconError::Computation)?;

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
    for &(i, j, n_ij) in &cells {
        let p_ij = n_ij as f64 / n_f;
        let p_i = row_sums[i] as f64 / n_f;
        let p_j = col_sums[j] as f64 / n_f;
        mi += p_ij * (p_ij / (p_i * p_j)).ln();
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

    let nmi_value = if k_true == 1 && k_pred == 1 {
        1.0 // one cluster on both sides: the same (trivial) partition
    } else if mi == 0.0 {
        0.0 // independent; includes one trivial partition against a non-trivial one
    } else {
        mi / normalizer
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
    /// Standard error of gap for each k: sd(log W*_kb) sqrt(1 + 1/B), sd with divisor B - 1
    pub se: Vec<f64>,
    /// log(W_k), W_k the within-cluster sum of squares (the paper's W_k); clusGap's `logW` is
    /// `log_w - log 2`
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
/// This is Tibshirani et al. (2001) as `cluster::clusGap(x, kmeans, K.max = k_max, B = n_boot,
/// d.power = 2, spaceH0 = "original")` with `maxSE(method = "Tibs2001SEmax")`. `log_w` is the log
/// of the paper's W_k (the within-cluster sum of squares); clusGap's `logW` is `log_w - log 2`
/// (its W_k sums each pair once), so gap and SE are equal. clusGap's own defaults differ:
/// `d.power = 1`, `spaceH0 = "scaledPCA"`, `B = 100`, `method = "firstSEmax"`; B defaults to 50
/// here. The k-means fits (3 starts) run Lloyd to convergence.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `k_max` - Maximum number of clusters to test
/// * `n_boot` - Number of reference data sets B (default: 50; at least 2)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `GapStatisticResult` containing gap values and optimal k
///
/// # Algorithm
///
/// For each k = 1, ..., k_max:
/// 1. Cluster the data and compute W_k (within-cluster sum of squares)
/// 2. Generate B reference datasets (uniform over the bounding box)
/// 3. Cluster each reference dataset and compute W_kb*
/// 4. Gap(k) = mean(log W_kb*) - log W_k, SE(k) = sd(log W_kb*) sqrt(1 + 1/B)
/// 5. Choose the smallest k such that Gap(k) >= Gap(k+1) - SE(k+1) (k_max if none)
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

    if b < 2 {
        return Err(format!(
            "n_boot must be at least 2 reference data sets for a standard error (got {b})"
        ));
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
            crate::resource::checkpoint().map_err(|e| e.to_string())?;
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

        let se_k = sample_sd(&log_w_b) * (1.0 + 1.0 / b as f64).sqrt();
        se.push(se_k);

        // Gap(k) = E[log(W_kb*)] - log(W_k)
        let gap_k = mean_log_w_b - log_w_k;
        gap.push(gap_k);
    }

    let optimal_k = tibs2001_se_max(&gap, &se);

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

/// `cluster::maxSE(f, SE, method = "Tibs2001SEmax")`: the smallest k (1-based) with
/// `f(k) >= f(k+1) - SE(k+1)`, the last k if none.
pub(crate) fn tibs2001_se_max(gap: &[f64], se: &[f64]) -> usize {
    (0..gap.len().saturating_sub(1))
        .find(|&i| gap[i] >= gap[i + 1] - se[i + 1])
        .map_or(gap.len(), |i| i + 1)
}

/// R's `sd`: divisor `len - 1`.
fn sample_sd(x: &[f64]) -> f64 {
    let m = x.len() as f64;
    let mean = x.iter().sum::<f64>() / m;
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt()
}

/// Within-cluster sum of squares of the best of 3 k-means fits (Lloyd to convergence).
fn compute_pooled_within_cluster_dispersion(
    data: &ArrayView2<f64>,
    k: usize,
    seed: Option<u64>,
) -> Result<f64, String> {
    let result = super::super::ml::kmeans(data.view(), k, Some(100), None, Some(3), seed)?;

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

    fn validation_blobs(id: &str) -> (p2a_validation::Case, Array2<f64>) {
        let (case, ds) = crate::test_support::load_case(id);
        let x = blobs(&ds);
        (case, x)
    }

    #[test]
    fn test_validate_cluster_indices_blobs() {
        // clusterCrit::intCriteria on ap_blobs with pam(ap_blobs, 3)$clustering.
        let (case, x) = validation_blobs("cluster_stats_blobs");
        let base = case_labels(&case);
        let ch = calinski_harabasz(x.view(), &base).unwrap();
        let db = davies_bouldin(x.view(), &base).unwrap();
        let dunn = dunn_index(x.view(), &base).unwrap();
        Check::new(&case)
            .scalar("ch", ch.ch_index, Tol::stat(115))
            .scalar("db", db.db_index, Tol::stat(115))
            .scalar("dunn", dunn.dunn_index, Tol::stat(115))
            .finish();
        // The same partition with gapped ids and with 1-based ids gives identical values.
        for map in [[0usize, 5, 9], [1, 2, 3]] {
            let l: Vec<usize> = base.iter().map(|&c| map[c]).collect();
            let ch2 = calinski_harabasz(x.view(), &l).unwrap();
            let db2 = davies_bouldin(x.view(), &l).unwrap();
            let dunn2 = dunn_index(x.view(), &l).unwrap();
            assert_eq!(ch2.ch_index.to_bits(), ch.ch_index.to_bits());
            assert_eq!(db2.db_index.to_bits(), db.db_index.to_bits());
            assert_eq!(dunn2.dunn_index.to_bits(), dunn.dunn_index.to_bits());
            assert_eq!(
                (ch2.n_clusters, db2.n_clusters, dunn2.n_clusters),
                (3, 3, 3)
            );
        }
    }

    #[test]
    fn test_validate_partition_agreement() {
        // aricode::ARI and aricode::NMI(variant = max, min, sqrt, sum) on fixed partitions.
        let (case, _) = crate::test_support::load_case("partition_agreement");
        let u: Vec<usize> = case.vector("u").iter().map(|&v| v as usize).collect();
        let w: Vec<usize> = case.vector("w").iter().map(|&v| v as usize).collect();
        let big = [0usize, 3, 1000, 1 << 40];
        let shifted: Vec<usize> = u.iter().map(|&v| v + 7).collect();
        let mapped: Vec<usize> = w.iter().map(|&v| big[v - 1]).collect();
        let mut check = Check::new(&case);
        for (a, b) in [(&u, &w), (&shifted, &w), (&u, &mapped)] {
            let ri = rand_index(a, b).unwrap();
            check.scalar("ari", ri.adjusted_rand_index, Tol::stat(15));
            for (variant, key) in [
                ("max", "nmi_max"),
                ("min", "nmi_min"),
                ("geometric", "nmi_sqrt"),
                ("arithmetic", "nmi_sum"),
            ] {
                check.scalar(key, nmi(a, b, Some(variant)).unwrap().nmi, Tol::stat(15));
            }
        }
        check.finish();
    }

    #[test]
    fn test_nmi_degenerate_partitions() {
        // scikit-learn's normalized_mutual_info_score: 1 when both partitions are one cluster,
        // 0 when the mutual information is 0 (one trivial partition against a non-trivial one).
        let one = [4usize; 6];
        let two = [0usize, 0, 0, 1, 1, 1];
        for m in ["geometric", "arithmetic", "min", "max"] {
            assert_eq!(nmi(&one, &one, Some(m)).unwrap().nmi, 1.0);
            assert_eq!(nmi(&one, &two, Some(m)).unwrap().nmi, 0.0, "{m}");
            assert_eq!(nmi(&two, &one, Some(m)).unwrap().nmi, 0.0, "{m}");
            // independent partitions: MI exactly 0
            let a = [0usize, 0, 1, 1];
            let b = [0usize, 1, 0, 1];
            assert_eq!(nmi(&a, &b, Some(m)).unwrap().nmi, 0.0, "{m}");
        }
    }

    #[test]
    fn test_indices_refuse_one_distinct_label() {
        let x = Array2::from_shape_fn((6, 2), |(i, j)| i as f64 + 0.1 * j as f64);
        let l = [1usize; 6];
        assert!(
            calinski_harabasz(x.view(), &l)
                .unwrap_err()
                .contains("at least 2 distinct clusters")
        );
        assert!(
            davies_bouldin(x.view(), &l)
                .unwrap_err()
                .contains("at least 2 distinct clusters")
        );
        assert!(
            dunn_index(x.view(), &l)
                .unwrap_err()
                .to_string()
                .contains("at least 2 distinct clusters")
        );
    }

    #[test]
    fn test_huge_labels_do_not_allocate_by_value() {
        let x = Array2::from_shape_fn((6, 2), |(i, j)| (i / 3) as f64 * 5.0 + 0.1 * (i + j) as f64);
        let l = [0usize, 0, 0, usize::MAX / 2, usize::MAX / 2, usize::MAX / 2];
        let m = [0usize, 0, 1, usize::MAX / 2, 1, usize::MAX / 2];
        crate::resource::with_alloc_ceiling(Some(1 << 20), || {
            assert_eq!(rand_index(&l, &l).unwrap().adjusted_rand_index, 1.0);
            assert!(nmi(&l, &m, None).unwrap().nmi.is_finite());
            assert_eq!(calinski_harabasz(x.view(), &l).unwrap().n_clusters, 2);
        });
    }

    #[test]
    fn test_silhouette_reports_distinct_cluster_count() {
        let x = Array2::from_shape_fn((6, 2), |(i, j)| {
            (i / 2) as f64 * 5.0 + 0.1 * (i * 3 + j) as f64
        });
        let r = silhouette(x.view(), &[0, 0, 2, 2, 4, 4]).unwrap();
        assert_eq!(r.n_clusters, 3);
        assert_eq!(r.cluster_silhouettes.len(), 5);
        assert_eq!(r.cluster_silhouettes[1], 0.0);
        assert_eq!(r.cluster_silhouettes[3], 0.0);
    }

    #[test]
    fn test_contingency_paths_agree() {
        let t = [0usize, 2, 2, 5, 0, 5, 2];
        let p = [1usize, 1, 0, 3, 3, 1, 0];
        let raw = contingency(&t, &p, "t").unwrap();
        let ct = compact_labels(&t, "t").unwrap();
        let cp = compact_labels(&p, "t").unwrap();
        let (cells, rows, cols) = contingency_cells(&ct, &cp).unwrap();
        let remap = |v: &[(usize, usize, usize)]| -> Vec<(usize, usize, usize)> {
            v.iter()
                .map(|&(i, j, c)| (ct.levels[i], cp.levels[j], c))
                .collect()
        };
        assert_eq!(raw.cells, remap(&cells));
        assert_eq!((raw.k_true, raw.k_pred), (3, 3));
        let nz = |v: &[usize]| v.iter().copied().filter(|&x| x > 0).collect::<Vec<_>>();
        assert_eq!(nz(&raw.rows), rows);
        assert_eq!(nz(&raw.cols), cols);
        let far: Vec<usize> = t.iter().map(|&l| l << 40).collect();
        let r1 = nmi(&t, &p, None).unwrap().nmi;
        let r2 = nmi(&far, &p, None).unwrap().nmi;
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_compact_labels_dense_and_sparse_agree() {
        let dense = compact_labels(&[3, 1, 3, 0, 1], "t").unwrap();
        assert_eq!(dense.ids, vec![2, 1, 2, 0, 1]);
        assert_eq!((dense.k, dense.levels.clone()), (3, vec![0, 1, 3]));
        let sparse = compact_labels(&[3 << 40, 1, 3 << 40, 0, 1], "t").unwrap();
        assert_eq!(sparse.ids, dense.ids);
        assert_eq!(sparse.levels, vec![0, 1, 3 << 40]);
    }

    #[test]
    fn test_validate_gap_logw_and_rule() {
        // clusGap(ap_blobs, kmeans, 5, d.power = 2, spaceH0 = "original")$Tab[, "logW"] =
        // log(WSS) - log 2 (clusGap sums each pair once). k = 4, 5 depend on k-means finding the
        // global optimum: checked as a property only.
        let (case, x) = validation_blobs("gap_logw_blobs");
        let g = gap_statistic(x.view(), 5, Some(5), Some(1)).unwrap();
        let logw: Vec<f64> = g.log_w[..3]
            .iter()
            .map(|v| v - std::f64::consts::LN_2)
            .collect();
        let r_logw = case.vector("logW");
        for (a, b) in logw.iter().zip(&r_logw[..3]) {
            assert!(Tol::coef(115).allows(*a, *b), "{a} vs {b}");
        }
        assert!(g.log_w.windows(2).all(|w| w[1] <= w[0]));
        let f1 = case.vector("f1");
        let f2 = case.vector("f2");
        Check::new(&case)
            .scalar(
                "tibs_k1",
                tibs2001_se_max(&f1, &case.vector("se1")) as f64,
                Tol::exact(),
            )
            .scalar(
                "tibs_k2",
                tibs2001_se_max(&f2, &case.vector("se2")) as f64,
                Tol::exact(),
            )
            .scalar("sd_example", sample_sd(&[1.0, 1.5, 0.7, 1.2]), Tol::stat(4))
            .finish();
    }

    #[test]
    fn test_gap_refuses_fewer_than_two_references() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| {
            (i % 2) as f64 * 4.0 + 0.1 * (i + 3 * j) as f64
        });
        for b in [0, 1] {
            let e = gap_statistic(x.view(), 3, Some(b), Some(1)).unwrap_err();
            assert!(e.contains("at least 2"), "{e}");
        }
        let g = gap_statistic(x.view(), 3, Some(2), Some(1)).unwrap();
        assert!(g.se.iter().all(|s| s.is_finite()));
    }
}
