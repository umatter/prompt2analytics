//! Hierarchical clustering: DIANA ([`diana`]), AGNES ([`agnes`]), pvclust bootstrap p-values ([`pvclust`]) and fastcluster ([`fastcluster`]).

use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_with_capacity};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

use super::common::{compute_distance_matrix, try_distance_matrix};

// =============================================================================
// Helper functions
// =============================================================================

/// Inputs of one Lance-Williams update: the distances from the merged
/// clusters `c1`, `c2` to another cluster, their distance to each other, and
/// the three cluster sizes.
#[derive(Clone, Copy)]
struct LanceWilliams {
    d_c1: f64,
    d_c2: f64,
    d_12: f64,
    n1: f64,
    n2: f64,
    n_other: f64,
}

/// Outcome of [`agglomerate`]. Cluster ids: observations are `0..n` and the
/// merge at step `s` creates id `n + s`.
struct Agglomeration {
    /// Per merge: `(c1, c2, height, size of the merged cluster)`.
    merges: Vec<(usize, usize, f64, usize)>,
    /// Remaining cluster ids, in the order the clusters were kept.
    active: Vec<usize>,
    /// Members of every id (ids that were merged away are empty).
    members: Vec<Vec<usize>>,
}

/// Agglomerative clustering on the `n x n` distance matrix `dist`, merging
/// the closest pair until `target_k` clusters remain.
///
/// The matrix is updated in place: a merged cluster takes over the row and
/// column of `c1`, so no matrix grows with the cluster ids. For every active
/// id `r` the minimum distance to the ids after it in the active order (and
/// the first id attaining it) is cached; the closest pair is the first row
/// whose cached minimum is smallest. That is exactly the lexicographically
/// first minimal pair a full scan over the active order finds, so merges,
/// heights and tie order equal those of the quadratic search. Ids keep the
/// old order: after a merge `c1` and `c2` leave the active list and the new
/// id is appended. `update` is the linkage's Lance-Williams formula and
/// `on_merge` sees the members of `c1` and `c2` and the merge height.
fn agglomerate(
    mut dist: Array2<f64>,
    target_k: usize,
    update: impl Fn(&LanceWilliams) -> f64,
    mut on_merge: impl FnMut(&[usize], &[usize], f64),
) -> Result<Agglomeration, String> {
    let n = dist.nrows();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<usize> = (0..n).collect();
    // Row and column of `dist` that holds each id
    let mut slot: Vec<usize> = (0..n).collect();
    // Cached minimum over the later active ids, and the first id attaining it
    let mut row_min = vec![f64::INFINITY; n];
    let mut row_arg = vec![usize::MAX; n];

    fn rescan(
        pos: usize,
        active: &[usize],
        slot: &[usize],
        dist: &Array2<f64>,
        row_min: &mut [f64],
        row_arg: &mut [usize],
    ) {
        let r = active[pos];
        let (mut best, mut arg) = (f64::INFINITY, usize::MAX);
        for &s in &active[pos + 1..] {
            let d = dist[[slot[r], slot[s]]];
            if d < best {
                best = d;
                arg = s;
            }
        }
        row_min[r] = best;
        row_arg[r] = arg;
    }

    for pos in 0..n {
        rescan(pos, &active, &slot, &dist, &mut row_min, &mut row_arg);
    }

    let mut merges = Vec::with_capacity(n.saturating_sub(target_k));

    while active.len() > target_k {
        checkpoint().map_err(|e| e.to_string())?;

        // Closest pair: the first row with the smallest cached minimum
        let mut min_dist = f64::INFINITY;
        let mut p1 = 0;
        for (pos, &r) in active.iter().enumerate() {
            if row_min[r] < min_dist {
                min_dist = row_min[r];
                p1 = pos;
            }
        }
        if min_dist.is_infinite() {
            return Err("No valid cluster pair found".to_string());
        }
        let c1 = active[p1];
        let c2 = row_arg[c1];
        let p2 = active[p1 + 1..]
            .iter()
            .position(|&id| id == c2)
            .map(|q| p1 + 1 + q)
            .ok_or_else(|| "No valid cluster pair found".to_string())?;

        on_merge(&members[c1], &members[c2], min_dist);
        merges.push((c1, c2, min_dist, members[c1].len() + members[c2].len()));

        // Lance-Williams update, written into c1's row and column
        let (s1, s2) = (slot[c1], slot[c2]);
        let n1 = members[c1].len() as f64;
        let n2 = members[c2].len() as f64;
        let d_12 = dist[[s1, s2]];
        for &other in &active {
            if other == c1 || other == c2 {
                continue;
            }
            let so = slot[other];
            let new_dist = update(&LanceWilliams {
                d_c1: dist[[s1, so]],
                d_c2: dist[[s2, so]],
                d_12,
                n1,
                n2,
                n_other: members[other].len() as f64,
            });
            dist[[s1, so]] = new_dist;
            dist[[so, s1]] = new_dist;
        }

        // Members and ids
        let mut merged = std::mem::take(&mut members[c1]);
        merged.extend_from_slice(&members[c2]);
        members[c2] = Vec::new();
        members.push(merged);
        let new_id = members.len() - 1;
        active.remove(p2);
        active.remove(p1);
        active.push(new_id);
        slot.push(s1);
        row_min.push(f64::INFINITY);
        row_arg.push(usize::MAX);

        // Cached minima: rows that pointed at c1 or c2 are rescanned; the
        // others only compare against the new cluster, which comes last
        for pos in 0..active.len() - 1 {
            let r = active[pos];
            if row_arg[r] == c1 || row_arg[r] == c2 {
                rescan(pos, &active, &slot, &dist, &mut row_min, &mut row_arg);
            } else {
                let d = dist[[slot[r], s1]];
                if d < row_min[r] {
                    row_min[r] = d;
                    row_arg[r] = new_id;
                }
            }
        }
    }

    Ok(Agglomeration {
        merges,
        active,
        members,
    })
}

// =============================================================================
// DIANA (DIvisive ANAlysis)
// =============================================================================

/// Result of DIANA divisive hierarchical clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data:
///   An Introduction to Cluster Analysis". Wiley, New York.
/// - R cluster::diana documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/diana.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DianaResult {
    /// Cluster assignments for each point (0 to n_clusters-1)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Divisive coefficient (quality measure)
    pub divisive_coefficient: f64,
    /// Split history: (cluster_split, new_cluster1, new_cluster2, diameter)
    pub merge_history: Vec<(usize, usize, usize, f64)>,
    /// Heights at which splits occurred
    pub heights: Vec<f64>,
    /// Order of observations in the dendrogram
    pub order: Vec<usize>,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for DianaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DIANA (Divisive) Clustering Results")?;
        writeln!(f, "====================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Divisive coefficient: {:.4}", self.divisive_coefficient)?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut sorted_labels: Vec<_> = cluster_counts.keys().cloned().collect();
        sorted_labels.sort();
        for label in sorted_labels {
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
        }

        Ok(())
    }
}

/// Run DIANA divisive hierarchical clustering.
///
/// DIANA is a divisive (top-down) hierarchical clustering algorithm that
/// starts with all observations in one cluster and recursively splits
/// the most heterogeneous cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (default: 2)
///
/// # Algorithm
///
/// 1. Start with all points in one cluster
/// 2. Find the cluster with largest diameter
/// 3. Find the "splinter group": point with largest average dissimilarity
/// 4. Move points to splinter group if closer to it than to rest
/// 5. Repeat until desired number of clusters
///
/// # Returns
/// * `DianaResult` containing cluster assignments and dendrogram info
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn diana(data: ArrayView2<f64>, n_clusters: Option<usize>) -> EconResult<DianaResult> {
    let n = data.nrows();

    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    let target_k = n_clusters.unwrap_or(2);
    if target_k == 0 || target_k > n {
        return Err(EconError::InvalidSpecification {
            message: format!("n_clusters must be between 1 and {}", n),
        });
    }

    // Compute distance matrix
    let distances = compute_distance_matrix(&data);

    // Initialize: all points in cluster 0
    let mut labels = vec![0usize; n];
    let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut merge_history = Vec::new();
    let mut heights = Vec::new();
    let mut next_cluster_id = 1;

    // Divisive loop
    while clusters.len() < target_k {
        // Find cluster with largest diameter to split
        let (split_idx, diameter) = find_largest_diameter_cluster(&clusters, &distances);

        if clusters[split_idx].len() < 2 {
            break; // Cannot split singleton
        }

        // Split the cluster
        let (group1, group2) = diana_split(&clusters[split_idx], &distances);

        if group1.is_empty() || group2.is_empty() {
            break; // Split failed
        }

        // Update labels
        let new_cluster_id = next_cluster_id;
        next_cluster_id += 1;

        for &idx in &group2 {
            labels[idx] = new_cluster_id;
        }

        // Record split
        merge_history.push((split_idx, split_idx, new_cluster_id, diameter));
        heights.push(diameter);

        // Update clusters
        clusters[split_idx] = group1;
        clusters.push(group2);
    }

    // Renumber labels to be consecutive
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0;
    for label in &mut labels {
        if !label_map.contains_key(label) {
            label_map.insert(*label, next_label);
            next_label += 1;
        }
        *label = label_map[label];
    }

    // Compute divisive coefficient
    let divisive_coefficient = compute_diana_coefficient(&heights);

    // Compute order (simple ordering for now)
    let order: Vec<usize> = (0..n).collect();

    Ok(DianaResult {
        labels,
        n_clusters: clusters.len(),
        divisive_coefficient,
        merge_history,
        heights,
        order,
        n,
    })
}

/// Find the cluster with the largest diameter.
fn find_largest_diameter_cluster(clusters: &[Vec<usize>], distances: &Array2<f64>) -> (usize, f64) {
    let mut max_diameter = 0.0f64;
    let mut max_idx = 0;

    for (idx, cluster) in clusters.iter().enumerate() {
        if cluster.len() < 2 {
            continue;
        }

        let diameter = compute_cluster_diameter(cluster, distances);
        if diameter > max_diameter {
            max_diameter = diameter;
            max_idx = idx;
        }
    }

    (max_idx, max_diameter)
}

/// Compute the diameter of a cluster (max pairwise distance).
fn compute_cluster_diameter(cluster: &[usize], distances: &Array2<f64>) -> f64 {
    let mut max_dist = 0.0f64;
    for (i, &idx_i) in cluster.iter().enumerate() {
        for &idx_j in cluster.iter().skip(i + 1) {
            max_dist = max_dist.max(distances[[idx_i, idx_j]]);
        }
    }
    max_dist
}

/// Split a cluster using DIANA algorithm.
fn diana_split(cluster: &[usize], distances: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    if cluster.len() < 2 {
        return (cluster.to_vec(), Vec::new());
    }

    // Find the object with largest average dissimilarity to rest
    let mut max_avg_dissim = f64::NEG_INFINITY;
    let mut splinter_idx = 0;

    for (i, &idx_i) in cluster.iter().enumerate() {
        let avg_dissim: f64 = cluster
            .iter()
            .filter(|&&j| j != idx_i)
            .map(|&j| distances[[idx_i, j]])
            .sum::<f64>()
            / (cluster.len() - 1) as f64;

        if avg_dissim > max_avg_dissim {
            max_avg_dissim = avg_dissim;
            splinter_idx = i;
        }
    }

    // Initialize splinter group with the most dissimilar object
    let mut splinter = vec![cluster[splinter_idx]];
    let mut remaining: Vec<usize> = cluster
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != splinter_idx)
        .map(|(_, &x)| x)
        .collect();

    // Iteratively move objects to splinter group
    loop {
        if remaining.is_empty() {
            break;
        }

        // For each remaining object, compute:
        // d_i = avg distance to remaining - avg distance to splinter
        let mut best_diff = f64::NEG_INFINITY;
        let mut best_idx = None;

        for (i, &idx) in remaining.iter().enumerate() {
            let avg_to_remaining = if remaining.len() > 1 {
                remaining
                    .iter()
                    .filter(|&&j| j != idx)
                    .map(|&j| distances[[idx, j]])
                    .sum::<f64>()
                    / (remaining.len() - 1) as f64
            } else {
                0.0
            };

            let avg_to_splinter = if !splinter.is_empty() {
                splinter.iter().map(|&j| distances[[idx, j]]).sum::<f64>() / splinter.len() as f64
            } else {
                f64::INFINITY
            };

            let diff = avg_to_remaining - avg_to_splinter;
            if diff > best_diff && diff > 0.0 {
                best_diff = diff;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(i) => {
                let idx = remaining.remove(i);
                splinter.push(idx);
            }
            None => break, // No more objects want to move
        }
    }

    (remaining, splinter)
}

/// Compute DIANA divisive coefficient.
fn compute_diana_coefficient(heights: &[f64]) -> f64 {
    if heights.is_empty() {
        return 0.0;
    }

    // DC is based on the normalized height at which objects are separated
    let max_height = heights.iter().cloned().fold(0.0, f64::max);
    if max_height < 1e-10 {
        return 0.0;
    }

    let normalized_heights: Vec<f64> = heights.iter().map(|&h| 1.0 - h / max_height).collect();

    normalized_heights.iter().sum::<f64>() / normalized_heights.len() as f64
}

/// Convenience wrapper for diana.
pub fn run_diana(data: ArrayView2<f64>, n_clusters: Option<usize>) -> EconResult<DianaResult> {
    diana(data, n_clusters)
}

// =============================================================================
// AGNES (AGglomerative NESting)
// =============================================================================

/// Result of AGNES agglomerative hierarchical clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
/// - R cluster::agnes documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/agnes.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgnesResult {
    /// Cluster assignments for each point (0 to n_clusters-1)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Agglomerative coefficient (measure of clustering structure)
    pub agglomerative_coefficient: f64,
    /// Linkage matrix: (cluster1, cluster2, height, size)
    pub merge: Vec<(usize, usize, f64, usize)>,
    /// Heights at which merges occurred
    pub heights: Vec<f64>,
    /// Order of observations in the dendrogram
    pub order: Vec<usize>,
    /// Linkage method used
    pub method: String,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for AgnesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "AGNES (Agglomerative) Clustering Results")?;
        writeln!(f, "=========================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Linkage method: {}", self.method)?;
        writeln!(
            f,
            "Agglomerative coefficient: {:.4}",
            self.agglomerative_coefficient
        )?;
        writeln!(f)?;

        // Count points per cluster
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f, "Cluster sizes:")?;
        let mut sorted_labels: Vec<_> = cluster_counts.keys().cloned().collect();
        sorted_labels.sort();
        for label in sorted_labels {
            writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
        }

        Ok(())
    }
}

/// Linkage method for AGNES.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AgnesLinkage {
    /// Average linkage (UPGMA)
    #[default]
    Average,
    /// Single linkage (nearest neighbor)
    Single,
    /// Complete linkage (farthest neighbor)
    Complete,
    /// Ward's method (minimize variance)
    Ward,
    /// Weighted average linkage (WPGMA)
    Weighted,
}

impl std::str::FromStr for AgnesLinkage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "upgma" => Ok(AgnesLinkage::Average),
            "single" => Ok(AgnesLinkage::Single),
            "complete" => Ok(AgnesLinkage::Complete),
            "ward" => Ok(AgnesLinkage::Ward),
            "weighted" | "wpgma" => Ok(AgnesLinkage::Weighted),
            _ => Err(format!(
                "Unknown linkage: {}. Use average, single, complete, ward, or weighted",
                s
            )),
        }
    }
}

/// Run AGNES agglomerative hierarchical clustering.
///
/// AGNES is an agglomerative (bottom-up) hierarchical clustering algorithm
/// that starts with each observation in its own cluster and iteratively
/// merges the closest pair of clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to form (default: 2)
/// * `method` - Linkage method (default: average)
///
/// # Algorithm
///
/// 1. Start with n clusters (each observation is a cluster)
/// 2. Compute distance between all pairs of clusters
/// 3. Merge the two closest clusters
/// 4. Update distances using Lance-Williams formula
/// 5. Repeat until desired number of clusters
///
/// # Returns
/// * `AgnesResult` containing cluster assignments and dendrogram info
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn agnes(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    method: Option<AgnesLinkage>,
) -> Result<AgnesResult, String> {
    let n = data.nrows();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let target_k = n_clusters.unwrap_or(2);
    if target_k == 0 || target_k > n {
        return Err(format!("n_clusters must be between 1 and {}", n));
    }

    let linkage = method.unwrap_or(AgnesLinkage::Average);

    // Compute distance matrix
    let distances = try_distance_matrix(&data, "AGNES distance matrix")?;

    let mut height_per_obs: Vec<f64> = vec![0.0; n];

    // Agglomerative loop
    let agglomeration = agglomerate(
        distances,
        target_k,
        |lw| agnes_lance_williams(linkage, lw),
        |members1, members2, height| {
            // Update height for observations
            for &idx in members1.iter().chain(members2) {
                height_per_obs[idx] = height_per_obs[idx].max(height);
            }
        },
    )?;
    let active_clusters = agglomeration.active;
    let cluster_members = agglomeration.members;
    let merge_history = agglomeration.merges;
    let heights: Vec<f64> = merge_history.iter().map(|m| m.2).collect();

    // Assign final labels
    let mut labels = vec![0usize; n];
    for (cluster_label, &cluster_id) in active_clusters.iter().enumerate() {
        for &idx in &cluster_members[cluster_id] {
            labels[idx] = cluster_label;
        }
    }

    // Compute agglomerative coefficient
    let max_height = heights.iter().cloned().fold(0.0, f64::max);
    let agglomerative_coefficient = if max_height > 1e-10 {
        let sum: f64 = height_per_obs.iter().map(|&h| 1.0 - h / max_height).sum();
        sum / n as f64
    } else {
        0.0
    };

    // Simple ordering
    let order: Vec<usize> = (0..n).collect();

    Ok(AgnesResult {
        labels,
        n_clusters: active_clusters.len(),
        agglomerative_coefficient,
        merge: merge_history,
        heights,
        order,
        method: format!("{:?}", linkage).to_lowercase(),
        n,
    })
}

/// AGNES's Lance-Williams update for `linkage`.
fn agnes_lance_williams(linkage: AgnesLinkage, lw: &LanceWilliams) -> f64 {
    let LanceWilliams {
        d_c1,
        d_c2,
        d_12,
        n1,
        n2,
        n_other,
    } = *lw;
    match linkage {
        AgnesLinkage::Single => d_c1.min(d_c2),
        AgnesLinkage::Complete => d_c1.max(d_c2),
        AgnesLinkage::Average => (n1 * d_c1 + n2 * d_c2) / (n1 + n2),
        AgnesLinkage::Weighted => (d_c1 + d_c2) / 2.0,
        AgnesLinkage::Ward => {
            let n_total = n1 + n2 + n_other;
            let alpha1 = (n1 + n_other) / n_total;
            let alpha2 = (n2 + n_other) / n_total;
            let beta = -n_other / n_total;
            (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
        }
    }
}

/// Convenience wrapper for agnes.
pub fn run_agnes(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    method: Option<&str>,
) -> Result<AgnesResult, String> {
    let linkage = method.map(|m| m.parse::<AgnesLinkage>()).transpose()?;
    agnes(data, n_clusters, linkage)
}

// =============================================================================
// PvClust - Hierarchical Clustering with Bootstrap P-values
// =============================================================================

/// Result of pvclust hierarchical clustering with bootstrap.
///
/// # References
///
/// - Suzuki, R. and Shimodaira, H. (2006). "Pvclust: an R package for assessing
///   the uncertainty in hierarchical clustering". Bioinformatics, 22(12).
/// - R pvclust package documentation
///   Source: <https://www.rdocumentation.org/packages/pvclust/versions/2.2-0/topics/pvclust>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvClustResult {
    /// AU (Approximately Unbiased) p-values for each cluster (edge)
    pub au_pvalues: Vec<f64>,
    /// BP (Bootstrap Probability) p-values for each cluster
    pub bp_pvalues: Vec<f64>,
    /// Cluster labels for each observation at given threshold
    pub labels: Vec<usize>,
    /// Number of clusters at AU >= threshold
    pub n_clusters: usize,
    /// Merge matrix from hierarchical clustering (pairs of merged clusters)
    pub merge: Vec<(i32, i32)>,
    /// Heights of merges
    pub heights: Vec<f64>,
    /// Number of bootstrap replicates
    pub n_boot: usize,
    /// AU threshold used for cluster selection
    pub au_threshold: f64,
    /// Number of observations
    pub n: usize,
    /// Linkage method used
    pub method: String,
}

impl std::fmt::Display for PvClustResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PvClust Hierarchical Clustering with Bootstrap")?;
        writeln!(f, "==============================================")?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Linkage method: {}", self.method)?;
        writeln!(f, "Bootstrap replicates: {}", self.n_boot)?;
        writeln!(f, "AU threshold: {:.2}", self.au_threshold)?;
        writeln!(
            f,
            "Clusters (AU >= {:.2}): {}",
            self.au_threshold, self.n_clusters
        )?;
        writeln!(f)?;

        // Show significant clusters (top 10)
        let mut significant: Vec<(usize, f64, f64)> = self
            .au_pvalues
            .iter()
            .zip(&self.bp_pvalues)
            .enumerate()
            .filter(|&(_, (au, _))| *au >= self.au_threshold)
            .map(|(i, (au, bp))| (i + 1, *au, *bp))
            .collect();
        significant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        writeln!(f, "Significant clusters (top 10):")?;
        writeln!(f, "  Edge   AU     BP")?;
        for (edge, au, bp) in significant.iter().take(10) {
            writeln!(f, "  {:4}   {:.3}  {:.3}", edge, au, bp)?;
        }

        Ok(())
    }
}

/// Perform hierarchical clustering with multiscale bootstrap resampling.
///
/// pvclust assesses the uncertainty of hierarchical cluster analysis by
/// computing approximately unbiased (AU) p-values through multiscale
/// bootstrap resampling.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `method` - Linkage method: "average", "single", "complete", "ward"
/// * `n_boot` - Number of bootstrap replicates (default: 1000)
/// * `r` - Bootstrap sample sizes as ratios (default: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]);
///   at least one, each finite and positive
/// * `au_threshold` - Threshold for significant clusters (default: 0.95)
/// * `seed` - Random seed
///
/// # Returns
/// * `PvClustResult` with AU and BP p-values for each cluster
///
/// # Algorithm
///
/// 1. Perform initial hierarchical clustering
/// 2. For each bootstrap sample size ratio:
///    a. Resample observations with replacement
///    b. Perform hierarchical clustering
///    c. Check which original clusters appear in bootstrap tree
/// 3. Fit curve to compute AU p-values using multiscale extrapolation
/// 4. BP p-values are computed from r=1.0 bootstrap frequency
///
/// # References
///
/// - Shimodaira, H. (2004). "Approximately unbiased tests of regions using
///   multistep-multiscale bootstrap resampling". Annals of Statistics.
pub fn pvclust(
    data: ArrayView2<f64>,
    method: Option<&str>,
    n_boot: Option<usize>,
    r: Option<Vec<f64>>,
    au_threshold: Option<f64>,
    seed: Option<u64>,
) -> Result<PvClustResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let linkage_method = method.unwrap_or("average");
    let nboot = n_boot.unwrap_or(1000);
    let ratios = r.unwrap_or_else(|| vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]);
    let threshold = au_threshold.unwrap_or(0.95);

    if nboot == 0 {
        return Err("n_boot must be at least 1 (got 0)".to_string());
    }
    if ratios.is_empty() {
        return Err("r must contain at least one ratio (got none)".to_string());
    }
    if let Some(&bad) = ratios.iter().find(|&&r| !(r.is_finite() && r > 0.0)) {
        return Err(format!(
            "r must contain only finite positive ratios (got {bad})"
        ));
    }
    let to_string = |e: EconError| e.to_string();

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Perform initial clustering
    let (merge, heights) = hierarchical_clustering_merge(&data, linkage_method)?;

    let n_edges = merge.len();

    // Extract cluster memberships from the merge matrix
    let original_clusters = extract_clusters_from_merge(&merge, n);

    // Bootstrap counts for each edge at each scale
    let mut counts: Vec<Vec<usize>> = vec![vec![0; ratios.len()]; n_edges];

    // First position of each observation in the current bootstrap sample
    let mut first_pos = vec![usize::MAX; n];

    // Perform multiscale bootstrap
    for (scale_idx, &ratio) in ratios.iter().enumerate() {
        let sample_size = ((n as f64) * ratio).round() as usize;
        let sample_size = sample_size.max(2);

        // Bootstrap buffers, sized once per scale and refilled per replicate
        let what = "pvclust bootstrap sample";
        let mut indices = try_with_capacity::<usize>(sample_size, what).map_err(to_string)?;
        let boot_values = try_vec(
            checked_len(&[sample_size, d], what).map_err(to_string)?,
            0.0f64,
            what,
        )
        .map_err(to_string)?;
        let mut boot_data = Array2::from_shape_vec((sample_size, d), boot_values)
            .map_err(|e| format!("{what}: {e}"))?;

        for _ in 0..nboot {
            checkpoint().map_err(to_string)?;

            // Bootstrap sample indices
            indices.clear();
            indices.extend((0..sample_size).map(|_| rng.gen_range(0..n)));

            // Create bootstrap data
            for (i, &idx) in indices.iter().enumerate() {
                boot_data.row_mut(i).assign(&data.row(idx));
            }

            // Perform clustering on bootstrap sample
            let (boot_merge, _) = hierarchical_clustering_merge(&boot_data.view(), linkage_method)?;
            let boot_clusters = extract_clusters_from_merge(&boot_merge, sample_size);

            // Map each original observation to its first position in the sample
            first_pos.fill(usize::MAX);
            for (pos, &obs) in indices.iter().enumerate() {
                if first_pos[obs] == usize::MAX {
                    first_pos[obs] = pos;
                }
            }

            // Bootstrap clusters as sorted position lists (positions within a
            // cluster are distinct, so sorted equality is set equality)
            let boot_sets: std::collections::HashSet<Vec<usize>> = boot_clusters
                .into_iter()
                .map(|mut cluster| {
                    cluster.sort_unstable();
                    cluster
                })
                .collect();

            // Check which original clusters appear
            for (edge_idx, orig_cluster) in original_clusters.iter().enumerate() {
                // Map original cluster members to bootstrap sample
                let mut mapped: Vec<usize> = orig_cluster
                    .iter()
                    .map(|&obs| first_pos[obs])
                    .filter(|&pos| pos != usize::MAX)
                    .collect();

                if mapped.len() >= 2 {
                    mapped.sort_unstable();
                    // Check if this cluster appears in bootstrap
                    if boot_sets.contains(&mapped) {
                        counts[edge_idx][scale_idx] += 1;
                    }
                }
            }
        }
    }

    // Compute AU and BP p-values
    let mut au_pvalues = vec![0.0; n_edges];
    let mut bp_pvalues = vec![0.0; n_edges];

    for edge_idx in 0..n_edges {
        // BP is simply the frequency at r=1.0
        let r1_idx = ratios.iter().position(|&r| (r - 1.0).abs() < 0.01);
        if let Some(idx) = r1_idx {
            bp_pvalues[edge_idx] = counts[edge_idx][idx] as f64 / nboot as f64;
        }

        // AU p-value from multiscale extrapolation
        // Use weighted least squares on log(-log(p)) vs log(r)
        au_pvalues[edge_idx] = compute_au_pvalue(&counts[edge_idx], &ratios, nboot);
    }

    // Assign cluster labels based on significant edges
    let labels = assign_pvclust_labels(&merge, &au_pvalues, threshold, n);
    let n_clusters = labels.iter().max().map(|&m| m + 1).unwrap_or(1);

    Ok(PvClustResult {
        au_pvalues,
        bp_pvalues,
        labels,
        n_clusters,
        merge,
        heights,
        n_boot: nboot,
        au_threshold: threshold,
        n,
        method: linkage_method.to_string(),
    })
}

/// Perform hierarchical clustering and return merge matrix and heights.
fn hierarchical_clustering_merge(
    data: &ArrayView2<f64>,
    method: &str,
) -> Result<(Vec<(i32, i32)>, Vec<f64>), String> {
    let n = data.nrows();

    // Compute distance matrix
    let distances = try_distance_matrix(data, "pvclust distance matrix")?;

    // Perform agglomerative clustering
    let agglomeration = agglomerate(
        distances,
        1,
        |lw| hclust_lance_williams(method, lw),
        |_, _, _| {},
    )?;

    // R-style indexing: negative for leaves, positive for internal nodes
    let to_r = |c: usize| {
        if c < n {
            -((c as i32) + 1)
        } else {
            (c - n + 1) as i32
        }
    };
    let merge = agglomeration
        .merges
        .iter()
        .map(|&(c1, c2, _, _)| (to_r(c1), to_r(c2)))
        .collect();
    let heights = agglomeration.merges.iter().map(|m| m.2).collect();

    Ok((merge, heights))
}

/// pvclust's Lance-Williams update for `method` (unknown methods use average).
fn hclust_lance_williams(method: &str, lw: &LanceWilliams) -> f64 {
    let LanceWilliams {
        d_c1,
        d_c2,
        d_12,
        n1,
        n2,
        n_other,
    } = *lw;
    match method {
        "single" => d_c1.min(d_c2),
        "complete" => d_c1.max(d_c2),
        "average" => (n1 * d_c1 + n2 * d_c2) / (n1 + n2),
        "ward" => {
            let n_total = n1 + n2 + n_other;
            let alpha1 = (n1 + n_other) / n_total;
            let alpha2 = (n2 + n_other) / n_total;
            let beta = -n_other / n_total;
            (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
        }
        _ => (n1 * d_c1 + n2 * d_c2) / (n1 + n2), // default to average
    }
}

/// Extract clusters from merge matrix.
fn extract_clusters_from_merge(merge: &[(i32, i32)], n: usize) -> Vec<Vec<usize>> {
    let mut clusters = Vec::new();

    // Build cluster contents from merge history
    let mut cluster_contents: HashMap<i32, Vec<usize>> = HashMap::new();

    // Initialize leaves
    for i in 0..n {
        cluster_contents.insert(-((i as i32) + 1), vec![i]);
    }

    // Process merges
    for (merge_idx, &(m1, m2)) in merge.iter().enumerate() {
        let members1 = cluster_contents.get(&m1).cloned().unwrap_or_default();
        let members2 = cluster_contents.get(&m2).cloned().unwrap_or_default();

        let mut new_members = members1;
        new_members.extend(members2);

        // Record this cluster
        clusters.push(new_members.clone());

        // Store for future reference
        cluster_contents.insert((merge_idx + 1) as i32, new_members);
    }

    clusters
}

/// Compute AU p-value from multiscale bootstrap.
fn compute_au_pvalue(counts: &[usize], ratios: &[f64], nboot: usize) -> f64 {
    // Compute frequencies
    let freqs: Vec<f64> = counts
        .iter()
        .map(|&c| (c as f64 / nboot as f64).max(0.001).min(0.999))
        .collect();

    // Transform: z = qnorm(p), x = sqrt(r)
    // Fit: z = a + b*x
    // AU p-value: pnorm(-a)

    let mut sum_x = 0.0;
    let mut sum_z = 0.0;
    let mut sum_xz = 0.0;
    let mut sum_xx = 0.0;
    let n = ratios.len() as f64;

    for (&r, &p) in ratios.iter().zip(&freqs) {
        let x = r.sqrt();
        let z = inverse_normal_cdf(p);
        sum_x += x;
        sum_z += z;
        sum_xz += x * z;
        sum_xx += x * x;
    }

    // Simple linear regression
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return freqs.iter().sum::<f64>() / freqs.len() as f64;
    }

    let b = (n * sum_xz - sum_x * sum_z) / denom;
    let a = (sum_z - b * sum_x) / n;

    // AU = Phi(-a)
    normal_cdf(-a)
}

/// Inverse normal CDF (probit function) - simple approximation.
fn inverse_normal_cdf(p: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let p = p.max(1e-10).min(1.0 - 1e-10);

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

/// Normal CDF - standard normal distribution function.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Horner form of approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Assign cluster labels based on significant AU p-values.
fn assign_pvclust_labels(
    merge: &[(i32, i32)],
    au_pvalues: &[f64],
    threshold: f64,
    n: usize,
) -> Vec<usize> {
    let mut labels = vec![0; n];

    // Find significant clusters (starting from highest AU)
    let mut significant_edges: Vec<(usize, f64)> = au_pvalues
        .iter()
        .enumerate()
        .filter(|&(_, au)| *au >= threshold)
        .map(|(i, au)| (i, *au))
        .collect();
    significant_edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Build cluster contents
    let mut cluster_contents: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        cluster_contents.insert(-((i as i32) + 1), vec![i]);
    }
    for (merge_idx, &(m1, m2)) in merge.iter().enumerate() {
        let members1 = cluster_contents.get(&m1).cloned().unwrap_or_default();
        let members2 = cluster_contents.get(&m2).cloned().unwrap_or_default();
        let mut new_members = members1;
        new_members.extend(members2);
        cluster_contents.insert((merge_idx + 1) as i32, new_members);
    }

    // Assign labels based on significant edges
    let mut assigned = vec![false; n];
    let mut next_label = 0usize;

    for (edge_idx, _) in significant_edges {
        let cluster_id = (edge_idx + 1) as i32;
        if let Some(members) = cluster_contents.get(&cluster_id) {
            // Only assign if members haven't been assigned yet
            let unassigned: Vec<usize> =
                members.iter().filter(|&&m| !assigned[m]).copied().collect();

            if !unassigned.is_empty() {
                for &m in &unassigned {
                    labels[m] = next_label;
                    assigned[m] = true;
                }
                next_label += 1;
            }
        }
    }

    // Assign remaining to their own clusters
    for i in 0..n {
        if !assigned[i] {
            labels[i] = next_label;
            next_label += 1;
        }
    }

    labels
}

/// Convenience wrapper for pvclust.
pub fn run_pvclust(
    data: ArrayView2<f64>,
    method: Option<&str>,
    n_boot: Option<usize>,
    au_threshold: Option<f64>,
    seed: Option<u64>,
) -> Result<PvClustResult, String> {
    pvclust(data, method, n_boot, None, au_threshold, seed)
}

// =============================================================================
// Fast Hierarchical Clustering (fastcluster)
// =============================================================================

/// Linkage method for fast hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum FastLinkage {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
    /// Ward's minimum variance method
    #[default]
    Ward,
    /// Weighted average linkage (WPGMA)
    Weighted,
    /// Centroid linkage (UPGMC)
    Centroid,
    /// Median linkage (WPGMC)
    Median,
}

/// Result of fast hierarchical clustering.
///
/// # References
///
/// - Müllner, D. (2013). "fastcluster: Fast Hierarchical, Agglomerative
///   Clustering Routines for R and Python." Journal of Statistical Software.
/// - R fastcluster package documentation
///   Source: <https://www.rdocumentation.org/packages/fastcluster/versions/1.2.6/topics/hclust>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastClusterResult {
    /// Merge matrix (n-1 x 2): indices of clusters merged at each step
    /// Negative values indicate original observations, positive indicate merged clusters
    #[serde(skip)]
    pub merge: Array2<i64>,
    /// Heights at which merges occurred
    pub height: Vec<f64>,
    /// Order of leaves for optimal plotting
    pub order: Vec<usize>,
    /// Labels for each observation (if cut was performed)
    pub labels: Option<Vec<usize>>,
    /// Number of observations
    pub n: usize,
    /// Linkage method used
    pub linkage: FastLinkage,
}

impl std::fmt::Display for FastClusterResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fast Hierarchical Clustering Results")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Linkage method: {:?}", self.linkage)?;
        writeln!(f, "Number of merges: {}", self.merge.nrows())?;
        if let Some(ref labels) = self.labels {
            let n_clusters = labels.iter().max().map_or(0, |m| m + 1);
            writeln!(f, "Number of clusters (if cut): {}", n_clusters)?;
        }
        Ok(())
    }
}

/// Run fast hierarchical clustering.
///
/// This is an optimized O(n²) implementation of hierarchical clustering
/// using the nearest-neighbor chain algorithm and condensed distance storage.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `linkage` - Linkage method (default: Ward)
/// * `n_clusters` - Optional: cut dendrogram to produce this many clusters
///
/// # Algorithm
///
/// Uses the nearest-neighbor chain algorithm which achieves O(n²) complexity
/// for reducible linkage methods (single, complete, average, Ward, weighted).
///
/// # Returns
/// * `FastClusterResult` containing merge matrix and heights
pub fn fastcluster(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    n_clusters: Option<usize>,
) -> Result<FastClusterResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n < 2 {
        return Err("Need at least 2 observations for hierarchical clustering".to_string());
    }

    let method = linkage.unwrap_or(FastLinkage::Ward);

    // Compute condensed distance matrix (upper triangle only, stored as 1D)
    // Index formula: condensed_index(i, j, n) = i*n - i*(i+1)/2 + j - i - 1 for i < j
    let condensed_size = n * (n - 1) / 2;
    let mut condensed_dist = vec![0.0f64; condensed_size];

    // Compute squared Euclidean distances (avoid sqrt until needed)
    for i in 0..n {
        for j in (i + 1)..n {
            let sq_dist: f64 = (0..d)
                .map(|k| {
                    let diff = data[[i, k]] - data[[j, k]];
                    diff * diff
                })
                .sum();
            condensed_dist[condensed_idx(i, j, n)] = sq_dist;
        }
    }

    // Use optimized nearest-neighbor chain algorithm
    let (merge, height) = fastcluster_nnchain_optimized(&mut condensed_dist, method, n)?;

    // Compute leaf ordering
    let order = compute_dendrogram_order(&merge, n);

    // Cut tree if n_clusters specified
    let labels = if let Some(k) = n_clusters {
        if k < 1 || k > n {
            return Err(format!("n_clusters must be between 1 and n ({})", n));
        }
        Some(cut_dendrogram(&merge, &height, n, k))
    } else {
        None
    };

    Ok(FastClusterResult {
        merge,
        height,
        order,
        labels,
        n,
        linkage: method,
    })
}

/// Compute index into condensed distance matrix for i < j
#[inline(always)]
fn condensed_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j);
    i * n - (i * (i + 1)) / 2 + j - i - 1
}

/// Get distance from condensed matrix (handles i > j by swapping)
#[inline(always)]
fn get_dist(condensed: &[f64], i: usize, j: usize, n: usize) -> f64 {
    if i == j {
        0.0
    } else if i < j {
        condensed[condensed_idx(i, j, n)]
    } else {
        condensed[condensed_idx(j, i, n)]
    }
}

/// Set distance in condensed matrix (handles i > j by swapping)
#[inline(always)]
fn set_dist(condensed: &mut [f64], i: usize, j: usize, n: usize, val: f64) {
    if i < j {
        condensed[condensed_idx(i, j, n)] = val;
    } else if j < i {
        condensed[condensed_idx(j, i, n)] = val;
    }
}

/// Optimized nearest-neighbor chain algorithm for hierarchical clustering.
///
/// This implementation uses:
/// 1. Condensed distance matrix (O(n²/2) space instead of O(n²))
/// 2. Nearest-neighbor tracking to avoid full scans
/// 3. Union-find structure for efficient cluster merging
fn fastcluster_nnchain_optimized(
    condensed_dist: &mut [f64],
    method: FastLinkage,
    n: usize,
) -> Result<(Array2<i64>, Vec<f64>), String> {
    // Union-find structure: maps each original index to its current cluster representative
    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_size: Vec<usize> = vec![1; n];

    // Track which clusters are still active
    let mut active: Vec<bool> = vec![true; n];
    let mut n_active = n;

    // Nearest neighbor for each cluster (cache)
    let mut nearest: Vec<usize> = vec![usize::MAX; n];
    let mut nearest_dist: Vec<f64> = vec![f64::INFINITY; n];

    // Initialize nearest neighbors
    for i in 0..n {
        for j in (i + 1)..n {
            let d = get_dist(condensed_dist, i, j, n);
            if d < nearest_dist[i] {
                nearest_dist[i] = d;
                nearest[i] = j;
            }
            if d < nearest_dist[j] {
                nearest_dist[j] = d;
                nearest[j] = i;
            }
        }
    }

    // Output arrays
    let mut merge = Array2::<i64>::zeros((n - 1, 2));
    let mut height = Vec::with_capacity(n - 1);

    // Track merge history for R-style indexing
    let mut merge_step: Vec<Option<usize>> = vec![None; n];

    // Nearest-neighbor chain
    let mut chain: Vec<usize> = Vec::with_capacity(n);

    for step in 0..(n - 1) {
        // Find a starting point if chain is empty
        if chain.is_empty() {
            for i in 0..n {
                if active[i] {
                    chain.push(i);
                    break;
                }
            }
        }

        // Build chain until we find a reciprocal pair
        loop {
            let current = *chain
                .last()
                .ok_or_else(|| "Internal error: nearest-neighbor chain is empty".to_string())?;

            // Find nearest neighbor of current (update if stale)
            let mut nn = nearest[current];
            let mut nn_dist = if nn < n && active[nn] {
                get_dist(condensed_dist, current, nn, n)
            } else {
                f64::INFINITY
            };

            // Check if cached nearest is still valid, otherwise recompute
            if !active[nn] || nn_dist > nearest_dist[current] * 1.001 {
                nn = usize::MAX;
                nn_dist = f64::INFINITY;
                for j in 0..n {
                    if j != current && active[j] {
                        let d = get_dist(condensed_dist, current, j, n);
                        if d < nn_dist {
                            nn_dist = d;
                            nn = j;
                        }
                    }
                }
                nearest[current] = nn;
                nearest_dist[current] = nn_dist;
            }

            // Check if we have a reciprocal pair (nn's nearest is current)
            if chain.len() >= 2 && chain[chain.len() - 2] == nn {
                // Found reciprocal pair: merge current and nn
                let (a, b) = if current < nn {
                    (current, nn)
                } else {
                    (nn, current)
                };

                // Record the merge with proper height
                let merge_dist = nn_dist.sqrt(); // Convert squared distance to distance

                // R-style merge indices: negative for leaves, positive for merged clusters
                let idx_a = match merge_step[a] {
                    None => -(a as i64 + 1),
                    Some(s) => s as i64 + 1,
                };
                let idx_b = match merge_step[b] {
                    None => -(b as i64 + 1),
                    Some(s) => s as i64 + 1,
                };

                merge[[step, 0]] = idx_a.min(idx_b);
                merge[[step, 1]] = idx_a.max(idx_b);
                height.push(merge_dist);

                // Merge b into a
                let size_a = cluster_size[a];
                let size_b = cluster_size[b];
                let new_size = size_a + size_b;

                // Update distances from merged cluster to all other active clusters
                for k in 0..n {
                    if !active[k] || k == a || k == b {
                        continue;
                    }

                    let d_ak = get_dist(condensed_dist, a, k, n);
                    let d_bk = get_dist(condensed_dist, b, k, n);
                    let d_ab = nn_dist; // squared distance
                    let n_a = size_a as f64;
                    let n_b = size_b as f64;
                    let n_k = cluster_size[k] as f64;

                    // Lance-Williams formula (using squared distances for efficiency)
                    let new_dist_sq = match method {
                        FastLinkage::Single => d_ak.min(d_bk),
                        FastLinkage::Complete => d_ak.max(d_bk),
                        FastLinkage::Average => (n_a * d_ak + n_b * d_bk) / (n_a + n_b),
                        FastLinkage::Weighted => (d_ak + d_bk) / 2.0,
                        FastLinkage::Ward => {
                            // Ward's method with squared distances
                            let n_total = n_a + n_b + n_k;
                            ((n_a + n_k) * d_ak + (n_b + n_k) * d_bk - n_k * d_ab) / n_total
                        }
                        FastLinkage::Centroid => {
                            let alpha_a = n_a / (n_a + n_b);
                            let alpha_b = n_b / (n_a + n_b);
                            let beta = -n_a * n_b / ((n_a + n_b) * (n_a + n_b));
                            (alpha_a * d_ak + alpha_b * d_bk + beta * d_ab).max(0.0)
                        }
                        FastLinkage::Median => (0.5 * d_ak + 0.5 * d_bk - 0.25 * d_ab).max(0.0),
                    };

                    set_dist(condensed_dist, a, k, n, new_dist_sq);

                    // Invalidate k's nearest neighbor cache if it was a or b
                    if nearest[k] == a || nearest[k] == b {
                        nearest_dist[k] = f64::INFINITY;
                    }
                }

                // Update cluster info
                cluster_size[a] = new_size;
                active[b] = false;
                merge_step[a] = Some(step);
                parent[b] = a;
                n_active -= 1;

                // Invalidate a's nearest neighbor cache
                nearest_dist[a] = f64::INFINITY;

                // Pop both from chain
                chain.pop(); // current
                chain.pop(); // nn (was second-to-last)

                break;
            } else {
                // Not a reciprocal pair, extend chain
                chain.push(nn);
            }
        }
    }

    Ok((merge, height))
}

/// Compute dendrogram leaf ordering for optimal display.
fn compute_dendrogram_order(merge: &Array2<i64>, n: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(n);

    fn traverse(merge: &Array2<i64>, n: usize, node: i64, order: &mut Vec<usize>) {
        if node < 0 {
            // Leaf node
            order.push((-node - 1) as usize);
        } else {
            // Internal node
            let step = (node - 1) as usize;
            traverse(merge, n, merge[[step, 0]], order);
            traverse(merge, n, merge[[step, 1]], order);
        }
    }

    // Start from root (last merge)
    let n_merges = merge.nrows();
    if n_merges > 0 {
        traverse(merge, n, n_merges as i64, &mut order);
    }

    if order.len() != n {
        // Fallback: simple ordering
        order = (0..n).collect();
    }

    order
}

/// Cut dendrogram at a height to produce k clusters.
fn cut_dendrogram(merge: &Array2<i64>, height: &[f64], n: usize, k: usize) -> Vec<usize> {
    if k >= n {
        return (0..n).collect();
    }

    // Find height threshold: we need n - k merges
    let n_merges_needed = n - k;
    let threshold = if n_merges_needed == 0 {
        0.0
    } else if n_merges_needed <= height.len() {
        height[n_merges_needed - 1]
    } else {
        height.last().copied().unwrap_or(f64::INFINITY)
    };

    // Build cluster assignments
    let mut labels = vec![0usize; n];
    let mut cluster_labels: HashMap<i64, usize> = HashMap::new();
    let mut next_label = 0usize;

    // Process merges up to threshold
    for (step, &h) in height.iter().enumerate() {
        if h > threshold && step >= n_merges_needed {
            break;
        }

        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        // Get or assign labels for children
        let left_label = if left < 0 {
            *cluster_labels.entry(left).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            })
        } else {
            *cluster_labels.get(&left).unwrap_or(&0)
        };

        let right_label = if right < 0 {
            *cluster_labels.entry(right).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            })
        } else {
            *cluster_labels.get(&right).unwrap_or(&0)
        };

        // Merge: assign same label to merged cluster
        let merged_label = left_label.min(right_label);
        cluster_labels.insert(step as i64 + 1, merged_label);

        // Update any nodes with the higher label
        for (_, v) in cluster_labels.iter_mut() {
            if *v == left_label.max(right_label) {
                *v = merged_label;
            }
        }
    }

    // Assign final labels to observations
    for i in 0..n {
        labels[i] = *cluster_labels.get(&(-(i as i64) - 1)).unwrap_or(&i);
    }

    // Renumber labels to be consecutive
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0;
    for l in labels.iter_mut() {
        if let Some(&new_l) = label_map.get(l) {
            *l = new_l;
        } else {
            label_map.insert(*l, next);
            *l = next;
            next += 1;
        }
    }

    labels
}

/// Convenience wrapper for fastcluster.
pub fn run_fastcluster(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    n_clusters: Option<usize>,
) -> Result<FastClusterResult, String> {
    fastcluster(data, linkage, n_clusters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// R 4.3.3 (cluster 2.x): x <- matrix(c(0.12,1.05, 0.48,0.31, 2.91,3.17, 3.36,2.62, 1.07,0.74,
    ///   5.21,0.43, 4.83,1.12, 0.71,2.38, 2.24,2.85, 5.66,0.97, 1.53,1.36, 3.95,3.48), ncol=2, byrow=TRUE)
    /// h <- as.hclust(agnes(x, method=m, stand=FALSE)); h$merge; h$height
    #[test]
    fn test_validate_agnes_heights_match_r() {
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
        let cases: [(AgnesLinkage, [(i64, i64); 11], [f64; 11]); 5] = [
            (
                AgnesLinkage::Average,
                [
                    (-6, -10),
                    (-3, -4),
                    (-2, -5),
                    (1, -7),
                    (-1, 3),
                    (2, -9),
                    (5, -11),
                    (6, -12),
                    (7, -8),
                    (9, 8),
                    (10, 4),
                ],
                [
                    0.702922470831599,
                    0.710633520177595,
                    0.730068489937759,
                    0.815581769545351,
                    0.911110689642995,
                    0.942933998299164,
                    1.233536822337978,
                    1.316836182244754,
                    1.631379297886202,
                    3.072068278418086,
                    3.900204945891715,
                ],
            ),
            (
                AgnesLinkage::Single,
                [
                    (-6, -10),
                    (-3, -4),
                    (-2, -5),
                    (2, -9),
                    (3, -11),
                    (1, -7),
                    (-1, 5),
                    (4, -12),
                    (7, -8),
                    (9, 8),
                    (10, 6),
                ],
                [
                    0.702922470831599,
                    0.710633520177595,
                    0.730068489937759,
                    0.742495791233863,
                    0.772010362624751,
                    0.787718223732319,
                    0.822921624457639,
                    1.042928568982555,
                    1.308739851918631,
                    1.600562401157793,
                    2.100214274782457,
                ],
            ),
            (
                AgnesLinkage::Complete,
                [
                    (-6, -10),
                    (-3, -4),
                    (-2, -5),
                    (1, -7),
                    (-1, 3),
                    (2, -12),
                    (-8, -11),
                    (7, -9),
                    (5, 8),
                    (6, 4),
                    (9, 10),
                ],
                [
                    0.702922470831599,
                    0.710633520177595,
                    0.730068489937759,
                    0.843445315358382,
                    0.999299754828350,
                    1.085218871933215,
                    1.308739851918631,
                    1.650515071121739,
                    3.090177988401316,
                    3.577373338079211,
                    5.540577587219585,
                ],
            ),
            (
                AgnesLinkage::Weighted,
                [
                    (-6, -10),
                    (-3, -4),
                    (-2, -5),
                    (1, -7),
                    (-1, 3),
                    (2, -9),
                    (5, -11),
                    (6, -12),
                    (7, -8),
                    (9, 8),
                    (10, 4),
                ],
                [
                    0.702922470831599,
                    0.710633520177595,
                    0.730068489937759,
                    0.815581769545351,
                    0.911110689642995,
                    0.942933998299164,
                    1.286071582727842,
                    1.443217413138189,
                    1.488341019526256,
                    2.957362070554049,
                    3.710583460924809,
                ],
            ),
            (
                AgnesLinkage::Ward,
                [
                    (-6, -10),
                    (-3, -4),
                    (-2, -5),
                    (1, -7),
                    (-1, 3),
                    (2, -9),
                    (-8, -11),
                    (6, -12),
                    (5, 7),
                    (8, 4),
                    (9, 10),
                ],
                [
                    0.702922470831599,
                    0.710633520177595,
                    0.730068489937759,
                    0.850431263144373,
                    0.969295276648624,
                    1.034762452610904,
                    1.308739851918631,
                    1.548956638084704,
                    2.011712371753642,
                    5.641712843589596,
                    8.13895608726796,
                ],
            ),
        ];
        let n = 12usize;
        let to_r = |id: usize| {
            if id < n {
                -(id as i64 + 1)
            } else {
                (id - n + 1) as i64
            }
        };
        let sorted = |(a, b): (i64, i64)| if a <= b { (a, b) } else { (b, a) };
        for (linkage, merges, heights) in cases {
            let r = agnes(x.view(), Some(1), Some(linkage)).unwrap();
            assert_eq!(r.merge.len(), 11, "{linkage:?}");
            for (step, (&(c1, c2, h, _), (&want_pair, &want_h))) in r
                .merge
                .iter()
                .zip(merges.iter().zip(heights.iter()))
                .enumerate()
            {
                assert_eq!(
                    sorted((to_r(c1), to_r(c2))),
                    sorted(want_pair),
                    "{linkage:?} step {step}"
                );
                assert!(
                    (h - want_h).abs() < 1e-12,
                    "{linkage:?} step {step}: {h} vs {want_h}"
                );
            }
        }
    }

    #[test]
    fn agnes_distance_matrix_is_fallible_and_cancellable() {
        let mut rng = StdRng::seed_from_u64(3);
        let data = Array2::from_shape_fn((100, 2), |_| rng.r#gen::<f64>());
        let e = with_alloc_ceiling(Some(40_000), || agnes(data.view(), Some(1), None)).unwrap_err(); // 80,000 B needed
        assert!(e.contains("limit"), "{e}");
        let e = with_budget(cancelled(), || agnes(data.view(), Some(2), None)).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn pvclust_validates_and_bounds_bootstrap_sizes() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0]
        ];
        assert!(
            pvclust(data.view(), None, Some(0), None, None, Some(1))
                .unwrap_err()
                .starts_with("n_boot")
        );
        assert!(
            pvclust(
                data.view(),
                None,
                Some(5),
                Some(vec![0.0, 1.0]),
                None,
                Some(1)
            )
            .unwrap_err()
            .starts_with("r ")
        );
        assert!(
            pvclust(
                data.view(),
                None,
                Some(5),
                Some(vec![f64::NAN]),
                None,
                Some(1)
            )
            .unwrap_err()
            .starts_with("r ")
        );
        // No scales: every AU p-value would be 0 / 0
        assert!(
            pvclust(data.view(), None, Some(5), Some(vec![]), None, Some(1))
                .unwrap_err()
                .starts_with("r ")
        );
        // 6e14 resampled rows: above the x86_64 address space, so the pre-fix run aborts rather than thrashing
        let e = with_alloc_ceiling(Some(1 << 30), || {
            pvclust(data.view(), None, Some(2), Some(vec![1e14]), None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("limit") || e.contains("too large"), "{e}");
        let e = with_budget(cancelled(), || {
            pvclust(data.view(), None, Some(5), None, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    /// Verbatim copy of `agnes` before the in-place engine (rewired to the
    /// reference helpers below); the reference for
    /// `in_place_engine_matches_reference`.
    fn reference_agnes(
        data: ArrayView2<f64>,
        n_clusters: Option<usize>,
        method: Option<AgnesLinkage>,
    ) -> Result<AgnesResult, String> {
        let n = data.nrows();

        if n == 0 {
            return Err("Empty data".to_string());
        }

        let target_k = n_clusters.unwrap_or(2);
        if target_k == 0 || target_k > n {
            return Err(format!("n_clusters must be between 1 and {}", n));
        }

        let linkage = method.unwrap_or(AgnesLinkage::Average);

        // Compute distance matrix
        let distances = compute_distance_matrix(&data);

        // Initialize: each point in its own cluster
        let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut active_clusters: Vec<usize> = (0..n).collect();
        let mut merge_history = Vec::new();
        let mut heights = Vec::new();
        let mut height_per_obs: Vec<f64> = vec![0.0; n];

        // Initialize cluster distances
        let mut cluster_dist = distances.clone();

        let mut next_cluster = n;

        // Agglomerative loop
        while active_clusters.len() > target_k {
            // Find closest pair of clusters
            let (c1_idx, c2_idx, min_dist) =
                reference_find_closest_pair(&active_clusters, &cluster_dist)?;

            let c1 = active_clusters[c1_idx];
            let c2 = active_clusters[c2_idx];

            // Record merge
            merge_history.push((
                c1,
                c2,
                min_dist,
                cluster_members[c1].len() + cluster_members[c2].len(),
            ));
            heights.push(min_dist);

            // Update height for observations
            for &idx in &cluster_members[c1] {
                height_per_obs[idx] = height_per_obs[idx].max(min_dist);
            }
            for &idx in &cluster_members[c2] {
                height_per_obs[idx] = height_per_obs[idx].max(min_dist);
            }

            // Merge clusters
            let mut new_members = cluster_members[c1].clone();
            new_members.extend(&cluster_members[c2]);

            // Update distances using Lance-Williams formula
            reference_update_agnes_distances(
                &mut cluster_dist,
                &cluster_members,
                &active_clusters,
                c1,
                c2,
                next_cluster,
                linkage,
            );

            // Update cluster tracking
            cluster_members.push(new_members);
            cluster_members[c1] = Vec::new();
            cluster_members[c2] = Vec::new();

            // Remove c1 and c2 from active, add new cluster
            // Remove in reverse order to avoid index shift issues
            let remove_first = c1_idx.max(c2_idx);
            let remove_second = c1_idx.min(c2_idx);
            active_clusters.remove(remove_first);
            active_clusters.remove(remove_second);
            active_clusters.push(next_cluster);

            next_cluster += 1;
        }

        // Assign final labels
        let mut labels = vec![0usize; n];
        for (cluster_label, &cluster_id) in active_clusters.iter().enumerate() {
            for &idx in &cluster_members[cluster_id] {
                labels[idx] = cluster_label;
            }
        }

        // Compute agglomerative coefficient
        let max_height = heights.iter().cloned().fold(0.0, f64::max);
        let agglomerative_coefficient = if max_height > 1e-10 {
            let sum: f64 = height_per_obs.iter().map(|&h| 1.0 - h / max_height).sum();
            sum / n as f64
        } else {
            0.0
        };

        // Simple ordering
        let order: Vec<usize> = (0..n).collect();

        Ok(AgnesResult {
            labels,
            n_clusters: active_clusters.len(),
            agglomerative_coefficient,
            merge: merge_history,
            heights,
            order,
            method: format!("{:?}", linkage).to_lowercase(),
            n,
        })
    }

    /// Verbatim copy of `find_closest_pair` before the in-place engine.
    fn reference_find_closest_pair(
        active: &[usize],
        dist: &Array2<f64>,
    ) -> Result<(usize, usize, f64), String> {
        let mut min_dist = f64::INFINITY;
        let mut best_pair = (0, 0);

        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let d = dist[[active[i], active[j]]];
                if d < min_dist {
                    min_dist = d;
                    best_pair = (i, j);
                }
            }
        }

        if min_dist.is_infinite() {
            return Err("No valid cluster pair found".to_string());
        }

        Ok((best_pair.0, best_pair.1, min_dist))
    }

    /// Verbatim copy of `update_agnes_distances` before the in-place engine.
    fn reference_update_agnes_distances(
        dist: &mut Array2<f64>,
        members: &[Vec<usize>],
        active: &[usize],
        c1: usize,
        c2: usize,
        new_cluster: usize,
        linkage: AgnesLinkage,
    ) {
        let n1 = members[c1].len() as f64;
        let n2 = members[c2].len() as f64;

        // Extend distance matrix if needed
        let current_size = dist.nrows();
        if new_cluster >= current_size {
            let mut new_dist = Array2::zeros((new_cluster + 1, new_cluster + 1));
            for i in 0..current_size {
                for j in 0..current_size {
                    new_dist[[i, j]] = dist[[i, j]];
                }
            }
            *dist = new_dist;
        }

        // Lance-Williams parameters based on linkage method
        for &other in active {
            if other == c1 || other == c2 {
                continue;
            }

            let d_c1 = dist[[c1, other]];
            let d_c2 = dist[[c2, other]];
            let d_12 = dist[[c1, c2]];
            let n_other = members[other].len() as f64;

            let new_dist = match linkage {
                AgnesLinkage::Single => d_c1.min(d_c2),
                AgnesLinkage::Complete => d_c1.max(d_c2),
                AgnesLinkage::Average => (n1 * d_c1 + n2 * d_c2) / (n1 + n2),
                AgnesLinkage::Weighted => (d_c1 + d_c2) / 2.0,
                AgnesLinkage::Ward => {
                    let n_total = n1 + n2 + n_other;
                    let alpha1 = (n1 + n_other) / n_total;
                    let alpha2 = (n2 + n_other) / n_total;
                    let beta = -n_other / n_total;
                    (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
                }
            };

            dist[[new_cluster, other]] = new_dist;
            dist[[other, new_cluster]] = new_dist;
        }
    }

    /// Verbatim copy of `hierarchical_clustering_merge` before the in-place
    /// engine (rewired to the reference helpers).
    fn reference_hierarchical_clustering_merge(
        data: &ArrayView2<f64>,
        method: &str,
    ) -> Result<(Vec<(i32, i32)>, Vec<f64>), String> {
        let n = data.nrows();

        // Compute distance matrix
        let distances = compute_distance_matrix(data);

        // Initialize: each point in its own cluster
        let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut active: Vec<usize> = (0..n).collect();
        let mut merge = Vec::new();
        let mut heights = Vec::new();
        let mut cluster_dist = distances.clone();
        let mut next_cluster = n;

        // Perform agglomerative clustering
        while active.len() > 1 {
            // Find closest pair
            let (c1_idx, c2_idx, min_dist) = reference_find_closest_pair(&active, &cluster_dist)?;

            let c1 = active[c1_idx];
            let c2 = active[c2_idx];

            // Record merge (using R-style indexing: negative for leaves, positive for internal)
            let m1 = if c1 < n {
                -((c1 as i32) + 1)
            } else {
                (c1 - n + 1) as i32
            };
            let m2 = if c2 < n {
                -((c2 as i32) + 1)
            } else {
                (c2 - n + 1) as i32
            };
            merge.push((m1, m2));
            heights.push(min_dist);

            // Merge clusters
            let mut new_members = cluster_members[c1].clone();
            new_members.extend(&cluster_members[c2]);

            // Update distances
            reference_update_hclust_distances(
                &mut cluster_dist,
                &cluster_members,
                &active,
                c1,
                c2,
                next_cluster,
                method,
            );

            // Update tracking
            cluster_members.push(new_members);
            cluster_members[c1] = Vec::new();
            cluster_members[c2] = Vec::new();

            let remove_first = c1_idx.max(c2_idx);
            let remove_second = c1_idx.min(c2_idx);
            active.remove(remove_first);
            active.remove(remove_second);
            active.push(next_cluster);

            next_cluster += 1;
        }

        Ok((merge, heights))
    }

    /// Verbatim copy of `update_hclust_distances` before the in-place engine.
    fn reference_update_hclust_distances(
        dist: &mut Array2<f64>,
        members: &[Vec<usize>],
        active: &[usize],
        c1: usize,
        c2: usize,
        new_cluster: usize,
        method: &str,
    ) {
        let n1 = members[c1].len() as f64;
        let n2 = members[c2].len() as f64;

        // Extend distance matrix if needed
        let current_size = dist.nrows();
        if new_cluster >= current_size {
            let mut new_dist = Array2::zeros((new_cluster + 1, new_cluster + 1));
            for i in 0..current_size {
                for j in 0..current_size {
                    new_dist[[i, j]] = dist[[i, j]];
                }
            }
            *dist = new_dist;
        }

        for &other in active {
            if other == c1 || other == c2 {
                continue;
            }

            let d_c1 = dist[[c1, other]];
            let d_c2 = dist[[c2, other]];
            let d_12 = dist[[c1, c2]];
            let n_other = members[other].len() as f64;

            let new_dist = match method {
                "single" => d_c1.min(d_c2),
                "complete" => d_c1.max(d_c2),
                "average" => (n1 * d_c1 + n2 * d_c2) / (n1 + n2),
                "ward" => {
                    let n_total = n1 + n2 + n_other;
                    let alpha1 = (n1 + n_other) / n_total;
                    let alpha2 = (n2 + n_other) / n_total;
                    let beta = -n_other / n_total;
                    (alpha1 * d_c1.powi(2) + alpha2 * d_c2.powi(2) + beta * d_12.powi(2)).sqrt()
                }
                _ => (n1 * d_c1 + n2 * d_c2) / (n1 + n2), // default to average
            };

            dist[[new_cluster, other]] = new_dist;
            dist[[other, new_cluster]] = new_dist;
        }
    }

    /// Reference guard: the in-place engine with cached row minima must
    /// reproduce the reallocating AGNES and pvclust hierarchies bit for bit,
    /// including their tie order.
    #[test]
    fn in_place_engine_matches_reference() {
        for seed in 0..4u64 {
            // coordinates on a 0.5 grid: duplicate points and many exactly tied distances exercise tie-breaking
            let mut rng = StdRng::seed_from_u64(seed);
            let data = Array2::from_shape_fn((60, 2), |_| rng.gen_range(0..10) as f64 * 0.5);
            for linkage in [
                AgnesLinkage::Average,
                AgnesLinkage::Single,
                AgnesLinkage::Complete,
                AgnesLinkage::Ward,
                AgnesLinkage::Weighted,
            ] {
                for target in [1usize, 2, 5] {
                    let got = agnes(data.view(), Some(target), Some(linkage)).unwrap();
                    let want = reference_agnes(data.view(), Some(target), Some(linkage)).unwrap();
                    assert_eq!(got.merge, want.merge, "seed {seed} {linkage:?} k={target}");
                    assert_eq!(got.heights, want.heights);
                    assert_eq!(got.labels, want.labels);
                    assert_eq!(
                        got.agglomerative_coefficient.to_bits(),
                        want.agglomerative_coefficient.to_bits()
                    );
                }
            }
            for method in ["average", "single", "complete", "ward", "other"] {
                assert_eq!(
                    hierarchical_clustering_merge(&data.view(), method).unwrap(),
                    reference_hierarchical_clustering_merge(&data.view(), method).unwrap(),
                    "seed {seed} {method}"
                );
            }
        }
    }

    /// Golden (characterization) test: captured on the code before the
    /// in-place engine and the per-replicate matching maps; it passes before
    /// and after.
    #[test]
    fn pvclust_outputs_unchanged_by_the_engine() {
        const GOLDEN_SUMS: [u64; 2] = [4625097201419323349, 4615739258092021352];
        const GOLDEN_LABELS: [usize; 30] = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2,
            2,
        ];
        const GOLDEN_AU: [u64; 5] = [
            4605713939569136160,
            4607136844457579383,
            4607182408038478870,
            4586554553958278736,
            4606467524958177223,
        ];
        let mut rng = StdRng::seed_from_u64(21);
        let centers = [[0.0, 0.0], [4.0, 4.0], [0.0, 5.0]];
        let data = Array2::from_shape_fn((30, 2), |(i, j)| centers[i % 3][j] + rng.r#gen::<f64>());
        let r = pvclust(
            data.view(),
            Some("average"),
            Some(15),
            None,
            Some(0.9),
            Some(7),
        )
        .unwrap();
        let sums = [
            r.au_pvalues.iter().sum::<f64>().to_bits(),
            r.bp_pvalues.iter().sum::<f64>().to_bits(),
        ];
        let au: Vec<u64> = r.au_pvalues[..5].iter().map(|v| v.to_bits()).collect();
        let msg = format!("got sums {sums:?}, labels {:?}, au {au:?}", r.labels);
        // Exact bits only where the constants were captured (x86_64 glibc
        // Linux with AVX2+FMA, crate::ml::golden_bits_platform): the AU
        // p-values go through libm's ln and exp, which can differ in the last
        // ulp elsewhere. One- and two-ulp changes of the scale ratios moved
        // them by at most 1.2e-13 relative here (and by 2.5e-11 over 2,000
        // random count vectors), while a changed bootstrap count moves them
        // by far more than the 1e-9 tolerance used off the reference
        // platform. The labels and the BP sum (counts / n_boot) are exact
        // everywhere.
        assert_eq!(&r.labels[..], &GOLDEN_LABELS[..], "{msg}");
        assert_eq!(sums[1], GOLDEN_SUMS[1], "{msg}");
        if crate::ml::golden_bits_platform() {
            assert_eq!(
                (sums[0], &au[..]),
                (GOLDEN_SUMS[0], &GOLDEN_AU[..]),
                "{msg}"
            );
        } else {
            let want = std::iter::once(GOLDEN_SUMS[0])
                .chain(GOLDEN_AU)
                .map(f64::from_bits);
            let got = std::iter::once(sums[0])
                .chain(au.iter().copied())
                .map(f64::from_bits);
            for (g, w) in got.zip(want) {
                assert!((g - w).abs() <= 1e-9 * w.abs(), "{msg}");
            }
        }
    }

    #[test]
    fn test_diana_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = diana(data.view(), Some(2)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert!(result.divisive_coefficient >= 0.0 && result.divisive_coefficient <= 1.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_agnes_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = agnes(data.view(), Some(2), Some(AgnesLinkage::Average)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert!(result.agglomerative_coefficient >= 0.0 && result.agglomerative_coefficient <= 1.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_agnes_linkage_methods() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0],];

        // Test all linkage methods work
        for linkage in [
            AgnesLinkage::Single,
            AgnesLinkage::Complete,
            AgnesLinkage::Average,
            AgnesLinkage::Ward,
            AgnesLinkage::Weighted,
        ] {
            let result = agnes(data.view(), Some(2), Some(linkage)).unwrap();
            assert_eq!(result.n_clusters, 2);
        }
    }

    #[test]
    fn test_pvclust_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Use fewer bootstrap samples for test speed
        let result = pvclust(
            data.view(),
            Some("average"),
            Some(100),
            None,
            Some(0.80),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.au_pvalues.len(), result.merge.len());
        assert_eq!(result.bp_pvalues.len(), result.merge.len());
        assert_eq!(result.method, "average");

        // AU and BP p-values should be in [0, 1]
        for &au in &result.au_pvalues {
            assert!((0.0..=1.0).contains(&au), "AU p-value out of range: {}", au);
        }
        for &bp in &result.bp_pvalues {
            assert!((0.0..=1.0).contains(&bp), "BP p-value out of range: {}", bp);
        }
    }

    #[test]
    fn test_fastcluster_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        let result = fastcluster(data.view(), Some(FastLinkage::Ward), None).unwrap();

        assert_eq!(result.n, 4);
        assert_eq!(result.merge.nrows(), 3); // n-1 merges
        assert_eq!(result.height.len(), 3);
        assert_eq!(result.order.len(), 4);

        // Heights should be non-decreasing (approximately)
        for i in 1..result.height.len() {
            assert!(
                result.height[i] >= result.height[i - 1] - 1e-6,
                "Heights should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_fastcluster_with_cut() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fastcluster(data.view(), Some(FastLinkage::Complete), Some(2)).unwrap();

        assert!(result.labels.is_some());
        let labels = result.labels.unwrap();
        assert_eq!(labels.len(), 6);

        // Should produce 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(unique_labels.len() <= 2);

        // First three points should be in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // Last three points should be in same cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
    }

    #[test]
    fn test_fastcluster_linkage_methods() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0],];

        // Test each linkage method runs without error
        for linkage in [
            FastLinkage::Single,
            FastLinkage::Complete,
            FastLinkage::Average,
            FastLinkage::Ward,
            FastLinkage::Weighted,
            FastLinkage::Centroid,
            FastLinkage::Median,
        ] {
            let result = fastcluster(data.view(), Some(linkage), None);
            assert!(result.is_ok(), "Linkage {:?} should work", linkage);
        }
    }
}
