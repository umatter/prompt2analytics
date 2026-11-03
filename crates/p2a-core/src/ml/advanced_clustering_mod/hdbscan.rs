//! HDBSCAN as R's `dbscan::hdbscan` (dbscan 1.2.6), ported step by step from its sources (`R/hdbscan.R`,
//! `src/mst.cpp`, `src/hdbscan.cpp`):
//!
//! - core distances that count the point itself (`minPts`: the distance to the `(min_samples - 1)`-th
//!   nearest other point, `kNNdist(x, minPts - 1)`);
//! - the mutual-reachability minimum spanning tree by Prim's algorithm on the dense graph, with dbscan's
//!   tie rules (start at point 0, strict `<`, lowest index among equal weights);
//! - the single-linkage merge order (`hclustMergeOrder`);
//! - the cluster tree with eps-birth, eps-death and stabilities (`computeStability`);
//! - excess-of-mass selection (`extractUnsupervised`, FOSC);
//! - noise, dbscan's membership probabilities and GLOSH outlier scores.
//!
//! `min_cluster_size >= 2` and `1 <= min_samples <= n` are validated, and non-finite data are refused.
//! The spanning tree takes O(n^2 d) time and O(n) memory for every n, as in dbscan; a k-d tree
//! (n >= 500, d <= 15) only speeds up the core distances.
//!
//! # References
//!
//! - Campello, R.J.G.B., Moulavi, D., and Sander, J. (2013). "Density-Based Clustering Based on
//!   Hierarchical Density Estimates". PAKDD 2013, LNCS 7819, 160-172.
//! - Campello, R.J.G.B., Moulavi, D., Zimek, A., and Sander, J. (2015). "Hierarchical Density Estimates
//!   for Data Clustering, Visualization, and Outlier Detection". ACM TKDD 10(1), 5 (GLOSH).
//! - Hahsler, M., Piekenbrock, M., and Doran, D. (2019). "dbscan: Fast Density-Based Clustering with R".
//!   Journal of Statistical Software 91(1).

use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::common::{Points, check_points, select_algorithm, uses_tree};
use crate::errors::{EconError, EconResult};
use crate::ml::kdtree::{KdTree, UnionFind};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};

/// `n * d` from which Prim's relaxation runs in parallel chunks. A tuning constant only: the chunked
/// minimum with the (weight, then lower index) rule picks the vertex the sequential scan picks.
const PRIM_PARALLEL_MIN_WORK: usize = 1 << 16;
/// Candidates per parallel chunk of Prim's relaxation.
const PRIM_CHUNK: usize = 1024;

// =============================================================================
// Configuration and result
// =============================================================================

/// Parameters of [`hdbscan_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdbscanConfig {
    /// Smallest cluster of the cluster tree (dbscan's `minPts` in `computeStability`); at least 2.
    /// Default 5.
    pub min_cluster_size: usize,
    /// Core-distance neighbourhood, counting the point itself (dbscan's `minPts`); `1..=n`.
    /// `None` = `min_cluster_size`. `1` gives core distances 0 (single linkage).
    pub min_samples: Option<usize>,
}

impl Default for HdbscanConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
        }
    }
}

/// One node of dbscan's cluster tree (`computeStability`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CondensedCluster {
    /// 0 = root; a split of node c creates the next two ids (top-down), so children have larger ids.
    pub id: usize,
    /// Parent node (`None` for the root).
    pub parent: Option<usize>,
    /// The two children created when this node splits.
    pub children: Option<(usize, usize)>,
    /// Mutual-reachability distance at which the cluster appears (lambda_birth = 1 / eps_birth).
    pub eps_birth: f64,
    /// Distance at which it splits, or at which its last points separate.
    pub eps_death: f64,
    /// `sum over leaving points (1/eps_p - 1/eps_birth) + n_at_split (1/eps_death - 1/eps_birth)`.
    pub stability: f64,
    /// Points that leave this node as noise (dbscan's `contains`).
    pub n_leaving: usize,
    /// Points handed to the children (dbscan's `n_children`; 0 for a leaf).
    pub n_at_split: usize,
    /// Chosen by excess of mass.
    pub selected: bool,
}

/// Result of HDBSCAN clustering ([`hdbscan_with`]; R `dbscan::hdbscan`).
///
/// # References
///
/// - Campello, R.J.G.B., Moulavi, D., and Sander, J. (2013). "Density-Based
///   Clustering Based on Hierarchical Density Estimates". PAKDD 2013.
/// - Hahsler, Piekenbrock and Doran (2019), JSS 91(1) (R package dbscan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdbscanResult {
    /// Cluster assignments: -1 for noise; label j is the j-th selected cluster in dbscan's order
    /// (R's label j + 1; R's 0 is noise).
    pub labels: Vec<i32>,
    /// dbscan's membership probability: `(max core distance in the cluster - core_i) / that maximum`;
    /// 0 for noise, 1 when every member of the cluster has core distance 0. Scale-free.
    pub probabilities: Vec<f64>,
    /// GLOSH outlier scores (Campello et al. 2015), as dbscan: `1 - eps_max / eps_p`, with `eps_p` the
    /// distance at which the point leaves its cluster and `eps_max` the smallest death distance among
    /// that cluster's leaf descendants. Scale-free; 0 = not an outlier.
    pub outlier_scores: Vec<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// Minimum cluster size used
    pub min_cluster_size: usize,
    /// Minimum samples used (effective value, counting the point itself)
    pub min_samples: usize,
    /// Number of observations
    pub n: usize,
    /// Core distance of each point (R `coredist`).
    #[serde(default)]
    pub core_distances: Vec<f64>,
    /// Stability of each selected cluster, by label (R `cluster_scores`).
    #[serde(default)]
    pub cluster_stability: Vec<f64>,
    /// Cluster-tree id of each label.
    #[serde(default)]
    pub cluster_ids: Vec<usize>,
    /// Single-linkage merge heights of the mutual-reachability tree (R `hc$height`), n - 1 values.
    #[serde(default)]
    pub merge_heights: Vec<f64>,
    /// The cluster tree, by id.
    #[serde(default)]
    pub condensed_tree: Vec<CondensedCluster>,
    /// Warnings about the fit.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for HdbscanResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for HdbscanResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HDBSCAN Clustering Results")?;
        writeln!(f, "==========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "min_cluster_size: {}", self.min_cluster_size)?;
        writeln!(f, "min_samples: {}", self.min_samples)?;

        let mut cluster_counts: BTreeMap<i32, usize> = BTreeMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (&label, &count) in &cluster_counts {
            if label == -1 {
                writeln!(f, "  Noise: {} points", count)?;
            } else {
                writeln!(f, "  Cluster {}: {} points", label, count)?;
            }
        }
        if !self.cluster_stability.is_empty() {
            writeln!(f)?;
            writeln!(f, "Cluster stability (excess of mass):")?;
            for (j, s) in self.cluster_stability.iter().enumerate() {
                writeln!(f, "  Cluster {}: {:.4}", j, s)?;
            }
        }
        write!(f, "{}", notes_text(self))
    }
}

// =============================================================================
// Entry points
// =============================================================================

/// Run HDBSCAN clustering (R `dbscan::hdbscan`).
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features), finite values
/// * `config` - [`HdbscanConfig`]: `min_cluster_size` (>= 2) and `min_samples` (`1..=n`, counting the
///   point itself; default `min_cluster_size`)
///
/// # Algorithm
///
/// 1. Core distances: the distance to the `(min_samples - 1)`-th nearest other point.
/// 2. The minimum spanning tree of the mutual-reachability distances
///    `max(d(i, j), core_i, core_j)` by Prim's algorithm from point 0 (dbscan's `mst.cpp`).
/// 3. Single linkage of the tree (`hclustMergeOrder`).
/// 4. The cluster tree: a merge whose two sides both hold at least `min_cluster_size` points is a split
///    (`computeStability`).
/// 5. Excess-of-mass selection of the most stable clusters (FOSC).
/// 6. Labels, membership probabilities and GLOSH outlier scores.
///
/// # Errors
/// `InsufficientData` for fewer than 2 rows; `InvalidSpecification` for `min_cluster_size < 2`,
/// `min_samples` outside `1..=n`, zero columns, non-finite values, or coordinates whose squared
/// distances overflow.
pub fn hdbscan_with(data: ArrayView2<f64>, config: &HdbscanConfig) -> EconResult<HdbscanResult> {
    let n = data.nrows();
    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "hdbscan".to_string(),
        });
    }
    let mcs = config.min_cluster_size;
    if mcs < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("hdbscan: min_cluster_size must be at least 2 (got {mcs})"),
        });
    }
    let ms = config.min_samples.unwrap_or(mcs);
    if ms < 1 || ms > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "hdbscan: min_samples must be between 1 and n = {n} (got {ms}); it counts the point itself, as dbscan's minPts"
            ),
        });
    }
    check_points(&data, "hdbscan")?;
    let pts = Points::new(&data, "hdbscan data")?;

    let core = core_distances(&pts, ms, uses_tree(select_algorithm(pts.n, pts.d)))?;
    let mst = mutual_reachability_mst(&pts, &core)?;
    let sl = single_linkage(&mst, n);
    let mut nodes = build_tree(&sl, mcs);
    let glosh = glosh_scores(&nodes, n);
    let selected = select_eom(&mut nodes);
    let labels = assign_labels(&nodes, &selected, n);

    // membership probabilities (dbscan 1.2.6 membership_prob)
    let k = selected.len();
    let mut max_core = vec![0.0f64; k];
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            let m = &mut max_core[l as usize];
            *m = m.max(core[i]);
        }
    }
    let probabilities: Vec<f64> = labels
        .iter()
        .zip(&core)
        .map(|(&l, &c)| {
            if l < 0 {
                0.0
            } else {
                let m = max_core[l as usize];
                // every member at core distance 0: dbscan's 0/0 = NaN; p2a reports 1
                if m > 0.0 { (m - c) / m } else { 1.0 }
            }
        })
        .collect();

    let cluster_stability: Vec<f64> = selected.iter().map(|&id| nodes[id].stability).collect();
    let mut warnings = Vec::new();
    if selected.is_empty() {
        warnings.push(format!(
            "no split leaves two clusters of at least min_cluster_size = {mcs} points, so every point is noise (as dbscan::hdbscan)"
        ));
    }
    let n_inf = cluster_stability.iter().filter(|s| s.is_infinite()).count();
    if n_inf > 0 {
        warnings.push(format!(
            "{n_inf} selected cluster(s) have infinite stability: at least min_samples points coincide, so some mutual-reachability distances are 0 (lambda = 1/0); excess of mass then keeps the parent, as dbscan does"
        ));
    }

    let condensed_tree = nodes
        .iter()
        .enumerate()
        .map(|(id, c)| CondensedCluster {
            id,
            parent: c.parent,
            children: c.children,
            eps_birth: c.eps_birth,
            eps_death: c.eps_death,
            stability: c.stability,
            n_leaving: c.leaving.len(),
            n_at_split: c.n_at_split,
            selected: c.selected,
        })
        .collect();
    let n_noise = labels.iter().filter(|&&l| l < 0).count();

    Ok(HdbscanResult {
        labels,
        probabilities,
        outlier_scores: glosh,
        n_clusters: k,
        n_noise,
        min_cluster_size: mcs,
        min_samples: ms,
        n,
        core_distances: core,
        cluster_stability,
        cluster_ids: selected,
        merge_heights: sl.height,
        condensed_tree,
        warnings,
    })
}

/// Run HDBSCAN clustering (R `dbscan::hdbscan`); see [`hdbscan_with`].
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `min_cluster_size` - Minimum cluster size, at least 2 (default: 5)
/// * `min_samples` - Core-distance neighbourhood counting the point itself, as dbscan's `minPts`
///   (default: `min_cluster_size`)
///
/// # Returns
/// * `HdbscanResult` with labels, membership probabilities, GLOSH outlier scores and the cluster tree
pub fn hdbscan(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<HdbscanResult, String> {
    hdbscan_with(
        data,
        &HdbscanConfig {
            min_cluster_size: min_cluster_size.unwrap_or(5),
            min_samples,
        },
    )
    .map_err(|e| e.to_string())
}

/// Convenience wrapper for hdbscan.
pub fn run_hdbscan(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<HdbscanResult, String> {
    hdbscan(data, min_cluster_size, min_samples)
}

// =============================================================================
// Pipeline steps
// =============================================================================

/// Core distances counting the point itself (dbscan `coredist = kNNdist(x, minPts - 1)`): 0 for
/// `min_samples = 1`, otherwise the distance to the `(min_samples - 1)`-th nearest other point. The k-d
/// tree and the brute-force paths agree bit for bit (one distance kernel).
pub(crate) fn core_distances(
    pts: &Points,
    min_samples: usize,
    use_tree: bool,
) -> EconResult<Vec<f64>> {
    let n = pts.n;
    if min_samples <= 1 {
        return resource::try_vec(n, 0.0, "hdbscan core distances");
    }
    let k = min_samples - 1;
    let budget = resource::current_budget();
    let check = || budget.as_ref().map_or(Ok(()), |b| b.check());
    if use_tree {
        let tree = KdTree::new(pts.rows_vec());
        (0..n)
            .into_par_iter()
            .map(|i| {
                check()?;
                let nb = tree.k_nearest(pts.row(i), k, Some(i));
                Ok(nb.get(k - 1).map_or(f64::INFINITY, |x| x.0))
            })
            .collect()
    } else {
        (0..n)
            .into_par_iter()
            .map_init(Vec::new, |buf: &mut Vec<f64>, i| {
                check()?;
                buf.clear();
                buf.extend((0..n).filter(|&j| j != i).map(|j| pts.sq(i, j)));
                if buf.len() < k {
                    return Ok(f64::INFINITY);
                }
                let (_, kth, _) = buf.select_nth_unstable_by(k - 1, f64::total_cmp);
                Ok(kth.sqrt())
            })
            .collect()
    }
}

/// An unvisited vertex of Prim's algorithm: its current weight and the tree vertex it would attach to.
#[derive(Clone, Copy)]
struct PrimCand {
    id: usize,
    weight: f64,
    parent: usize,
}

/// `(weight, index)` order of Prim's next-vertex rule: the smaller weight, then the lower index.
#[inline]
fn prim_better(w: f64, id: usize, best_w: f64, best_id: usize) -> bool {
    w < best_w || (w == best_w && id < best_id)
}

/// Relax every candidate against the new tree vertex `node` and return the position (in `cands`) of the
/// best next vertex: weights below `INFINITY` only, ties to the lower vertex index.
fn prim_relax(
    cands: &mut [PrimCand],
    base: usize,
    pts: &Points,
    core: &[f64],
    node: usize,
) -> Option<(f64, usize, usize)> {
    let row = pts.row(node);
    let cn = core[node];
    let mut best: Option<(f64, usize, usize)> = None;
    for (off, c) in cands.iter_mut().enumerate() {
        let w = crate::ml::kdtree::sq_euclidean(row, pts.row(c.id))
            .sqrt()
            .max(cn.max(core[c.id]));
        if w < c.weight {
            c.weight = w;
            c.parent = node;
        }
        if c.weight < f64::INFINITY {
            match best {
                Some((bw, bid, _)) if !prim_better(c.weight, c.id, bw, bid) => {}
                _ => best = Some((c.weight, c.id, base + off)),
            }
        }
    }
    best
}

/// Minimum spanning tree of the mutual-reachability graph `max(d(i, j), core_i, core_j)` by Prim's
/// algorithm on the dense graph (dbscan `mst.cpp`): start at vertex 0, strict `<` for weight updates,
/// the next vertex is the smallest weight with the lowest index among equal weights. Edge `node - 1` is
/// `(node, parent, weight)`. O(n^2 d) time, O(n) memory, a checkpoint per vertex.
pub(crate) fn mutual_reachability_mst(
    pts: &Points,
    core: &[f64],
) -> EconResult<Vec<(usize, usize, f64)>> {
    let parallel = pts.n.saturating_mul(pts.d.max(1)) >= PRIM_PARALLEL_MIN_WORK;
    mutual_reachability_mst_with(pts, core, parallel)
}

/// [`mutual_reachability_mst`] with the relaxation run sequentially or in parallel chunks (same tree).
fn mutual_reachability_mst_with(
    pts: &Points,
    core: &[f64],
    parallel: bool,
) -> EconResult<Vec<(usize, usize, f64)>> {
    let n = pts.n;
    if n <= 1 {
        return Ok(Vec::new());
    }
    let mut edges: Vec<(usize, usize, f64)> =
        resource::try_vec(n - 1, (0, 0, f64::INFINITY), "hdbscan spanning tree")?;
    let mut cands: Vec<PrimCand> = resource::try_with_capacity(n - 1, "hdbscan spanning tree")?;
    cands.extend((1..n).map(|id| PrimCand {
        id,
        weight: f64::INFINITY,
        parent: 0,
    }));
    let mut node = 0usize;
    loop {
        resource::checkpoint()?;
        let best = if parallel && cands.len() > PRIM_CHUNK {
            cands
                .par_chunks_mut(PRIM_CHUNK)
                .enumerate()
                .map(|(ci, chunk)| prim_relax(chunk, ci * PRIM_CHUNK, pts, core, node))
                .reduce(
                    || None,
                    |a, b| match (a, b) {
                        (None, x) | (x, None) => x,
                        (Some(x), Some(y)) => {
                            if prim_better(y.0, y.1, x.0, x.1) {
                                Some(y)
                            } else {
                                Some(x)
                            }
                        }
                    },
                )
        } else {
            prim_relax(&mut cands, 0, pts, core, node)
        };
        let Some((_, _, pos)) = best else { break };
        let c = cands.swap_remove(pos);
        edges[c.id - 1] = (c.id, c.parent, c.weight);
        node = c.id;
    }
    if !cands.is_empty() {
        return Err(EconError::Computation(
            "hdbscan: the mutual-reachability graph is not connected (non-finite distances)"
                .to_string(),
        ));
    }
    Ok(edges)
}

/// Single linkage of a spanning tree in R's `hclust` encoding (`hclustMergeOrder`): merge k joins
/// `merge[k].0` and `merge[k].1`, where `-(i + 1)` is point i and `s > 0` is merge step `s` (1-based).
pub(crate) struct SingleLinkage {
    pub(crate) merge: Vec<(i64, i64)>,
    pub(crate) height: Vec<f64>,
}

/// `hclustMergeOrder(mst, order(mst[, 3]))`: the edges in stable weight order (equal weights keep the
/// vertex order, as R's `order()`), labelled through a union-find.
pub(crate) fn single_linkage(mst: &[(usize, usize, f64)], n: usize) -> SingleLinkage {
    let mut order: Vec<usize> = (0..mst.len()).collect();
    order.sort_by(|&a, &b| mst[a].2.total_cmp(&mst[b].2)); // stable
    let mut uf = UnionFind::new(n);
    let mut label: Vec<i64> = (0..n).map(|i| -(i as i64) - 1).collect();
    let mut merge = Vec::with_capacity(mst.len());
    let mut height = Vec::with_capacity(mst.len());
    for (k, &e) in order.iter().enumerate() {
        let (from, to, w) = mst[e];
        let (rf, rt) = (uf.find(from), uf.find(to));
        merge.push((label[rf], label[rt]));
        height.push(w);
        uf.union(from, to);
        let r = uf.find(from);
        label[r] = k as i64 + 1;
    }
    SingleLinkage { merge, height }
}

/// One node of the cluster tree while it is built.
#[derive(Debug, Clone, Default)]
struct TreeNode {
    leaving: Vec<usize>,
    leaving_eps: Vec<f64>,
    eps_birth: f64,
    eps_death: f64,
    processed: bool,
    n_at_split: usize,
    parent: Option<usize>,
    children: Option<(usize, usize)>,
    stability: f64,
    score: f64,
    selected: bool,
}

/// dbscan's `computeStability`: walk the merges top-down; a merge whose two sides both hold at least
/// `mcs` points splits the current node into two new ones (the next two ids). A point leaves its node at
/// the height where it becomes a singleton, or at the node's death if the node has already split.
fn build_tree(sl: &SingleLinkage, mcs: usize) -> Vec<TreeNode> {
    let m = sl.merge.len(); // n - 1 >= 1
    let mut size = vec![0usize; m];
    for k in 0..m {
        let (l, r) = sl.merge[k];
        size[k] = match (l < 0, r < 0) {
            (true, true) => 2,
            (true, false) => size[r as usize - 1] + 1,
            (false, true) => size[l as usize - 1] + 1,
            (false, false) => size[l as usize - 1] + size[r as usize - 1],
        };
    }
    // root: eps_birth = the last height. Its eps_death is set by its split; a root that never splits
    // dies at its smallest merge height (dbscan leaves its unordered_map default 0 there, so every GLOSH
    // score becomes 1; p2a computes GLOSH by its definition).
    let mut nodes = vec![TreeNode {
        eps_birth: sl.height[m - 1],
        eps_death: f64::INFINITY,
        ..Default::default()
    }];
    let mut tracker = vec![0usize; m];
    for k in (0..m).rev() {
        let (l, r) = sl.merge[k];
        let (cid, h) = (tracker[k], sl.height[k]);
        match (l < 0, r < 0) {
            (true, true) => {
                let c = &mut nodes[cid];
                let e = if c.processed { c.eps_death } else { h };
                c.leaving.extend([(-l - 1) as usize, (-r - 1) as usize]);
                c.leaving_eps.extend([e, e]);
                if !c.processed {
                    c.eps_death = h.min(c.eps_death);
                }
            }
            (true, false) | (false, true) => {
                let (pt, sub) = if l < 0 {
                    ((-l - 1) as usize, r as usize)
                } else {
                    ((-r - 1) as usize, l as usize)
                };
                let c = &mut nodes[cid];
                let e = if c.processed { c.eps_death } else { h };
                c.leaving.push(pt);
                c.leaving_eps.push(e);
                tracker[sub - 1] = cid;
            }
            (false, false) => {
                let (lu, ru) = (l as usize, r as usize);
                let (s1, s2) = (size[lu - 1], size[ru - 1]);
                if s1 >= mcs && s2 >= mcs {
                    let (a, b) = (nodes.len(), nodes.len() + 1);
                    let c = &mut nodes[cid];
                    c.eps_death = h;
                    c.processed = true;
                    c.children = Some((a, b));
                    c.n_at_split = s1 + s2;
                    nodes.push(TreeNode {
                        eps_birth: h,
                        eps_death: sl.height[lu - 1],
                        parent: Some(cid),
                        ..Default::default()
                    });
                    nodes.push(TreeNode {
                        eps_birth: h,
                        eps_death: sl.height[ru - 1],
                        parent: Some(cid),
                        ..Default::default()
                    });
                    tracker[lu - 1] = a;
                    tracker[ru - 1] = b;
                } else {
                    tracker[lu - 1] = cid;
                    tracker[ru - 1] = cid;
                }
            }
        }
    }
    for c in &mut nodes {
        let s: f64 = c
            .leaving_eps
            .iter()
            .map(|&e| 1.0 / e - 1.0 / c.eps_birth)
            .sum();
        let nc = c.n_at_split as f64;
        c.stability = s + (nc / c.eps_death - nc / c.eps_birth);
    }
    nodes
}

/// GLOSH (Campello et al. 2015) as dbscan's `computeStability`: a point leaving node c at distance
/// `e` scores `1 - eps_max / e`, where `eps_max` is c's death distance for a leaf and the smallest death
/// distance among c's leaf descendants otherwise; 0 where that is 0/0 ("never an outlier").
fn glosh_scores(nodes: &[TreeNode], n: usize) -> Vec<f64> {
    // children have larger ids than their parent, so one pass in decreasing id order
    let mut leaf_min = vec![0.0f64; nodes.len()];
    for id in (0..nodes.len()).rev() {
        leaf_min[id] = match nodes[id].children {
            None => nodes[id].eps_death,
            Some((a, b)) => leaf_min[a].min(leaf_min[b]),
        };
    }
    let mut out = vec![0.0f64; n];
    for (id, c) in nodes.iter().enumerate() {
        let eps_max = leaf_min[id];
        for (&p, &e) in c.leaving.iter().zip(&c.leaving_eps) {
            let g = 1.0 - eps_max / e;
            out[p] = if g.is_nan() { 0.0 } else { g };
        }
    }
    out
}

/// Excess-of-mass selection (dbscan's `fosc` with `prune_unstable = FALSE` and
/// `cluster_selection_epsilon = 0`), without recursion. Returns the selected ids in dbscan's salient
/// order (post-order: first child, second child, node).
fn select_eom(nodes: &mut [TreeNode]) -> Vec<usize> {
    let m = nodes.len();
    let mut parent_wins = vec![false; m];
    for id in (0..m).rev() {
        match nodes[id].children {
            None => nodes[id].score = nodes[id].stability,
            Some((a, b)) => {
                let new = nodes[a].score + nodes[b].score;
                let old = nodes[id].stability;
                if old < new {
                    nodes[id].score = new;
                } else {
                    nodes[id].score = old; // ties and NaN keep the parent, as fosc
                    parent_wins[id] = id != 0; // the root is never a cluster
                }
            }
        }
    }
    let mut blocked = vec![false; m];
    for id in 0..m {
        if let Some((a, b)) = nodes[id].children {
            let bl = blocked[id] || parent_wins[id];
            blocked[a] = bl;
            blocked[b] = bl;
        }
    }
    let mut post = Vec::with_capacity(m);
    let mut stack = vec![(0usize, false)];
    while let Some((id, expanded)) = stack.pop() {
        match nodes[id].children {
            Some((a, b)) if !expanded => {
                stack.push((id, true));
                stack.push((b, false));
                stack.push((a, false));
            }
            _ => post.push(id),
        }
    }
    let selected: Vec<usize> = post
        .into_iter()
        .filter(|&id| id != 0 && !blocked[id] && (nodes[id].children.is_none() || parent_wins[id]))
        .collect();
    for &id in &selected {
        nodes[id].selected = true;
    }
    selected
}

/// dbscan's `getSalientAssignments`: every point leaving a selected cluster or one of its descendants
/// gets that cluster's label (its position in `selected`); the others are noise (-1).
fn assign_labels(nodes: &[TreeNode], selected: &[usize], n: usize) -> Vec<i32> {
    let mut node_label = vec![-1i32; nodes.len()];
    for (j, &id) in selected.iter().enumerate() {
        node_label[id] = j as i32;
    }
    for id in 0..nodes.len() {
        if node_label[id] < 0
            && let Some(p) = nodes[id].parent
        {
            node_label[id] = node_label[p];
        }
    }
    let mut labels = vec![-1i32; n];
    for (id, c) in nodes.iter().enumerate() {
        for &p in &c.leaving {
            labels[p] = node_label[id];
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::super::common::Points;
    use super::super::common::test_data::{blobs, r_codes, uniform};
    use super::*;
    use ndarray::{Array2, Axis, array, s};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_hdbscan_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = hdbscan(data.view(), Some(2), Some(2)).unwrap();

        assert_eq!(result.labels.len(), 6);
    }

    /// R: dbscan::hdbscan(blobs_noise, minPts = 5). validation/r/cases/ml_density_clustering.R, case hdbscan_blobs.
    #[test]
    fn test_validate_hdbscan_blobs_mp5() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let r = hdbscan(x.view(), Some(5), Some(5)).unwrap();
        Check::new(&case)
            .vector("mp5_cluster", &r_codes(&r.labels), Tol::exact())
            .vector("mp5_membership_prob", &r.probabilities, Tol::stat(n))
            .vector("mp5_outlier_scores", &r.outlier_scores, Tol::stat(n))
            .finish();
        assert_eq!((r.n_clusters, r.n_noise), (3, 6));
    }

    /// R: dbscan::hdbscan(blobs_noise, minPts = 10) (case hdbscan_blobs).
    #[test]
    fn test_validate_hdbscan_blobs_mp10() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let r = hdbscan(x.view(), Some(10), None).unwrap();
        assert_eq!(r.min_samples, 10);
        Check::new(&case)
            .vector("mp10_cluster", &r_codes(&r.labels), Tol::exact())
            .vector("mp10_membership_prob", &r.probabilities, Tol::stat(n))
            .vector("mp10_outlier_scores", &r.outlier_scores, Tol::stat(n))
            .finish();
    }

    /// The computeStability cluster tree, excess-of-mass selection, core distances (kNNdist(x, minPts - 1)) and
    /// merge heights (hc$height) of dbscan::hdbscan at minPts 5 and 10 (case hdbscan_blobs).
    #[test]
    fn test_validate_hdbscan_blobs_tree() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let mut chk = Check::new(&case);
        for mp in [5usize, 10] {
            let r = hdbscan_with(
                x.view(),
                &HdbscanConfig {
                    min_cluster_size: mp,
                    min_samples: None,
                },
            )
            .unwrap();
            let col = |f: fn(&CondensedCluster) -> f64| -> Vec<f64> {
                r.condensed_tree.iter().map(f).collect()
            };
            let ids: Vec<f64> = r.cluster_ids.iter().map(|&c| c as f64).collect();
            let p = format!("mp{mp}_");
            chk.vector(&format!("{p}coredist"), &r.core_distances, Tol::coef(n))
                .vector(&format!("{p}hc_height"), &r.merge_heights, Tol::coef(n))
                .vector(
                    &format!("{p}cluster_scores"),
                    &r.cluster_stability,
                    Tol::stat(n),
                )
                .vector(&format!("{p}salient"), &ids, Tol::exact())
                .vector(&format!("{p}tree_id"), &col(|c| c.id as f64), Tol::exact())
                .vector(
                    &format!("{p}tree_parent"),
                    &col(|c| c.parent.map_or(f64::NAN, |q| q as f64)),
                    Tol::exact(),
                )
                .vector(
                    &format!("{p}tree_eps_birth"),
                    &col(|c| c.eps_birth),
                    Tol::coef(n),
                )
                .vector(
                    &format!("{p}tree_eps_death"),
                    &col(|c| c.eps_death),
                    Tol::coef(n),
                )
                .vector(
                    &format!("{p}tree_stability"),
                    &col(|c| c.stability),
                    Tol::stat(n),
                )
                .vector(
                    &format!("{p}tree_n_children"),
                    &col(|c| c.n_at_split as f64),
                    Tol::exact(),
                )
                .vector(
                    &format!("{p}tree_n_contains"),
                    &col(|c| c.n_leaving as f64),
                    Tol::exact(),
                );
            if mp == 5 {
                chk.vector("knndist_k4", &r.core_distances, Tol::coef(n));
            }
            let selected: Vec<usize> = r
                .condensed_tree
                .iter()
                .filter(|c| c.selected)
                .map(|c| c.id)
                .collect();
            let mut want = r.cluster_ids.clone();
            want.sort_unstable();
            assert_eq!(selected, want);
        }
        chk.finish();
    }

    /// ML-49: probabilities and GLOSH do not depend on the units (R at 1000x scale; case hdbscan_blobs).
    #[test]
    fn test_validate_hdbscan_scale_free() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let unit = hdbscan(x.view(), Some(5), Some(5)).unwrap();
        let big = hdbscan((&x * 1000.0).view(), Some(5), Some(5)).unwrap();
        Check::new(&case)
            .vector("mp5_x1000_cluster", &r_codes(&big.labels), Tol::exact())
            .vector(
                "mp5_x1000_membership_prob",
                &big.probabilities,
                Tol::stat(n),
            )
            .vector(
                "mp5_x1000_outlier_scores",
                &big.outlier_scores,
                Tol::stat(n),
            )
            .vector(
                "mp5_x1000_cluster_scores",
                &big.cluster_stability,
                Tol::stat(n),
            )
            .finish();
        for (a, b) in unit
            .probabilities
            .iter()
            .zip(&big.probabilities)
            .chain(unit.outlier_scores.iter().zip(&big.outlier_scores))
        {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// No split leaves two clusters of >= 5 points (rows 1-9): every point is noise (dbscan; its
    /// root-based membership values are deviation P3a: p2a reports 0), and GLOSH follows its definition
    /// (dbscan gives every point 1). Rows 1-5 and 41-45 split into two children of exactly
    /// min_cluster_size points (case hdbscan_blobs).
    #[test]
    fn test_validate_hdbscan_no_split_and_boundary() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let sub = x.slice(s![0..9, ..]);
        let nosplit = hdbscan(sub, Some(5), Some(5)).unwrap();
        let rows: Vec<usize> = (0..5).chain(40..45).collect();
        let boundary = hdbscan(x.select(Axis(0), &rows).view(), Some(5), Some(5)).unwrap();
        Check::new(&case)
            .vector("nosplit_cluster", &r_codes(&nosplit.labels), Tol::exact())
            // deviation ML-49 (GLOSH without a split): nosplit_outlier_scores is not asserted
            // deviation ML-49 (P3a): nosplit_membership_prob_dbscan is not asserted
            .vector("boundary_cluster", &r_codes(&boundary.labels), Tol::exact())
            .vector(
                "boundary_cluster_scores",
                &boundary.cluster_stability,
                Tol::stat(10),
            )
            .finish();
        assert_eq!((nosplit.n_clusters, nosplit.n_noise), (0, 9));
        assert!(nosplit.probabilities.iter().all(|&p| p == 0.0));
        assert!(
            nosplit
                .warnings
                .iter()
                .any(|w| w.contains("every point is noise")),
            "{:?}",
            nosplit.warnings
        );
        assert!(nosplit.to_string().contains("Warning: "));
        // GLOSH by definition: e_p = the height at which p first merges, the minimum over q != p of
        // the mutual-reachability distance; every score is 1 - h_min / e_p
        let core = &nosplit.core_distances;
        let e: Vec<f64> = (0..9)
            .map(|p| {
                (0..9)
                    .filter(|&q| q != p)
                    .map(|q| {
                        let d = crate::ml::kdtree::euclidean_distance(
                            sub.row(p).as_slice().unwrap(),
                            sub.row(q).as_slice().unwrap(),
                        );
                        d.max(core[p]).max(core[q])
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let h_min = e.iter().copied().fold(f64::INFINITY, f64::min);
        for p in 0..9 {
            let want = 1.0 - h_min / e[p];
            let want = if want.is_nan() { 0.0 } else { want };
            assert!(
                (nosplit.outlier_scores[p] - want).abs() < 1e-12,
                "row {p}: {} vs {want}",
                nosplit.outlier_scores[p]
            );
            assert!(nosplit.outlier_scores[p] < 1.0);
        }
        let min = nosplit
            .outlier_scores
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert_eq!(min, 0.0);
        assert!(nosplit.condensed_tree[0].stability.is_finite());
    }

    /// min_samples = 1: core distances 0, so the merge heights are single linkage on dist(X)
    /// (R: hclust(dist(X), "single")$height; dbscan itself needs minPts >= 2).
    #[test]
    fn test_validate_hdbscan_min_samples_one_is_single_linkage() {
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let r = hdbscan_with(
            x.view(),
            &HdbscanConfig {
                min_cluster_size: 5,
                min_samples: Some(1),
            },
        )
        .unwrap();
        assert!(r.core_distances.iter().all(|&c| c == 0.0));
        assert!(
            r.probabilities
                .iter()
                .zip(&r.labels)
                .all(|(&p, &l)| l < 0 || p == 1.0)
        ); // 0/0 rule, P3b
        Check::new(&case)
            .vector("single_linkage_height", &r.merge_heights, Tol::coef(n))
            .finish();
    }

    #[test]
    fn test_hdbscan_min_samples_0_is_error() {
        // n = 40 took the brute-force path, n = 600 the k-d tree path; both panicked (ML-48, ML-134)
        for n in [40usize, 600] {
            let x = uniform(n, 2, 4141);
            let e = hdbscan(x.view(), Some(5), Some(0)).unwrap_err();
            assert!(e.contains("min_samples") && e.contains("got 0"), "{e}");
            let e = hdbscan(x.view(), Some(5), Some(n + 1)).unwrap_err();
            assert!(
                e.contains("min_samples") && e.contains(&format!("n = {n}")),
                "{e}"
            );
        }
    }

    #[test]
    fn test_hdbscan_rejects_bad_parameters() {
        let x = uniform(30, 2, 7);
        assert!(
            hdbscan(x.view(), Some(1), None)
                .unwrap_err()
                .contains("min_cluster_size must be at least 2 (got 1)")
        ); // was raised to 2
        assert!(
            hdbscan(x.slice(s![0..1, ..]), Some(5), None)
                .unwrap_err()
                .contains("at least 2")
        );
        let mut bad = x.clone();
        bad[[3, 1]] = f64::NAN;
        assert!(
            hdbscan(bad.view(), Some(5), None)
                .unwrap_err()
                .contains("non-finite")
        );
        bad[[3, 1]] = 1e200;
        bad[[4, 1]] = -1e200;
        assert!(
            hdbscan(bad.view(), Some(5), None)
                .unwrap_err()
                .contains("overflow")
        );
        assert!(
            hdbscan(Array2::<f64>::zeros((5, 0)).view(), Some(2), None)
                .unwrap_err()
                .contains("no columns")
        );
    }

    /// The k-d tree core distances (n >= 500, d <= 15) equal the brute-force ones bit for bit, so both paths
    /// give the same tree and labels.
    #[test]
    fn test_hdbscan_kdtree_core_distances_match_brute_force() {
        let x = uniform(600, 2, 99);
        assert!(super::super::common::uses_tree(select_algorithm(600, 2)));
        let pts = Points::new(&x.view(), "test").unwrap();
        for ms in [2usize, 5, 17] {
            let tree = core_distances(&pts, ms, true).unwrap();
            let brute = core_distances(&pts, ms, false).unwrap();
            assert_eq!(
                tree.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                brute.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "min_samples {ms}"
            );
        }
    }

    /// The parallel Prim relaxation (chunked minimum) picks the vertices of the sequential scan.
    #[test]
    fn test_hdbscan_parallel_prim_matches_sequential() {
        let x = uniform(3000, 3, 2718);
        let pts = Points::new(&x.view(), "test").unwrap();
        let core = core_distances(&pts, 5, false).unwrap();
        let par = mutual_reachability_mst_with(&pts, &core, true).unwrap();
        let seq = mutual_reachability_mst_with(&pts, &core, false).unwrap();
        assert_eq!(par.len(), seq.len());
        for (a, b) in par.iter().zip(&seq) {
            assert_eq!((a.0, a.1, a.2.to_bits()), (b.0, b.1, b.2.to_bits()));
        }
    }

    fn mst_test_data_small() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [5.0, 5.0],
            [5.5, 5.0],
            [5.0, 5.5],
        ]
    }

    /// Prim's tree and the k-d tree Prim of dual_tree agree on the total weight of the
    /// mutual-reachability spanning tree; the dual-tree Boruvka builder (unused by HDBSCAN) returns a
    /// spanning tree. Its weight is not compared: on the 100-point data it is not minimal (265.45
    /// against 150.95), which the old test hid by calling the k-d tree Prim under its name.
    fn check_msts(data: &Array2<f64>, ms: usize) {
        use crate::ml::dual_tree::{dual_tree_boruvka_mst, kdtree_prim_mst};
        let pts = Points::new(&data.view(), "test").unwrap();
        let core = core_distances(&pts, ms, false).unwrap();
        let mst = mutual_reachability_mst(&pts, &core).unwrap();
        let tree = KdTree::new(pts.rows_vec());
        let dt = dual_tree_boruvka_mst(&tree, &core);
        let kp = kdtree_prim_mst(&tree, &core);
        let n = data.nrows();
        assert_eq!(mst.len(), n - 1, "Prim MST edge count");
        assert_eq!(dt.edges.len(), n - 1, "Dual-tree MST edge count");
        assert_eq!(kp.len(), n - 1, "KD-tree Prim MST edge count");
        let total: f64 = mst.iter().map(|e| e.2).sum();
        let total_kp: f64 = kp.iter().map(|e| e.2).sum();
        assert!(
            (total - total_kp).abs() <= 1e-12 * total,
            "{total} vs {total_kp}"
        );
        let mut uf = UnionFind::new(n);
        for &(a, b, _) in &dt.edges {
            uf.union(a, b);
        }
        assert_eq!(uf.n_components(), 1, "the dual-tree edges span the points");
    }

    #[test]
    fn test_mst_correctness_dual_tree_vs_brute_force() {
        check_msts(&mst_test_data_small(), 2);
    }

    #[test]
    fn test_mst_correctness_larger_dataset() {
        use rand::prelude::*;
        use rand_distr::Normal;

        let n = 100;
        let mut rng = StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut data = Array2::zeros((n, 3));
        for i in 0..n {
            let cluster = i % 3;
            let center = cluster as f64 * 10.0;
            for j in 0..3 {
                data[[i, j]] = center + rng.sample(normal);
            }
        }
        check_msts(&data, 5);
    }

    /// The labels of the pipeline run from the given core distances.
    fn labels_from(pts: &Points, core: &[f64], mcs: usize) -> Vec<i32> {
        let mst = mutual_reachability_mst(pts, core).unwrap();
        let sl = single_linkage(&mst, pts.n);
        let mut nodes = build_tree(&sl, mcs);
        let selected = select_eom(&mut nodes);
        assign_labels(&nodes, &selected, pts.n)
    }

    #[test]
    fn test_hdbscan_cluster_consistency() {
        // the k-d tree and brute-force core distances give the same labels
        use rand::prelude::*;
        use rand_distr::Normal;

        let n = 60;
        let mut rng = StdRng::seed_from_u64(54321);
        let normal = Normal::new(0.0, 0.5).unwrap();

        let mut data = Array2::zeros((n, 2));
        for i in 0..n {
            let cluster = i % 3;
            let (cx, cy) = match cluster {
                0 => (0.0, 0.0),
                1 => (10.0, 0.0),
                _ => (5.0, 10.0),
            };
            data[[i, 0]] = cx + rng.sample(normal);
            data[[i, 1]] = cy + rng.sample(normal);
        }
        let pts = Points::new(&data.view(), "test").unwrap();
        let brute = core_distances(&pts, 5, false).unwrap();
        let tree = core_distances(&pts, 5, true).unwrap();
        assert_eq!(
            brute.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            tree.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let a = labels_from(&pts, &brute, 5);
        let b = labels_from(&pts, &tree, 5);
        assert_eq!(a, b);
        assert_eq!(a, hdbscan(data.view(), Some(5), Some(5)).unwrap().labels);
    }

    #[test]
    fn test_hdbscan_perf_comparison() {
        // hdbscan_optimized delegates to hdbscan
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let n = 100;
        let d = 3;
        let mut data = Array2::zeros((n, d));
        for i in 0..n {
            let cluster = i % 3;
            let center = cluster as f64 * 5.0;
            for j in 0..d {
                data[[i, j]] = center + rng.sample(normal);
            }
        }
        assert_eq!(
            hdbscan(data.view(), Some(5), Some(5)).unwrap().labels,
            crate::ml::cluster_optimized::hdbscan_optimized(data.view(), Some(5), Some(5))
                .unwrap()
                .0
        );
    }

    #[test]
    #[ignore = "benchmark; run with --release -- --ignored"]
    fn test_hdbscan_full_benchmark() {
        use rand::prelude::*;
        use rand_distr::Normal;
        use std::time::Instant;

        fn generate_data(n: usize, d: usize) -> Array2<f64> {
            let mut rng = StdRng::seed_from_u64(42);
            let normal = Normal::new(0.0, 1.0).unwrap();

            let mut data = Array2::zeros((n, d));
            for i in 0..n {
                let cluster = i % 5;
                let center = cluster as f64 * 5.0;
                for j in 0..d {
                    data[[i, j]] = center + rng.sample(normal);
                }
            }
            data
        }

        println!("\nHDBSCAN Performance Comparison");
        println!("==============================");
        println!("n\tnew(ms)\t\told(ms)\t\tspeedup");

        for n in [500, 1000, 2000, 3000, 5000] {
            let data = generate_data(n, 5);

            // Warmup
            let _ = hdbscan(data.view(), Some(10), Some(10));
            let _ =
                crate::ml::cluster_optimized::hdbscan_optimized(data.view(), Some(10), Some(10));

            // New algorithm (with auto-dispatch)
            let start = Instant::now();
            for _ in 0..3 {
                let _ = hdbscan(data.view(), Some(10), Some(10));
            }
            let new_time = start.elapsed().as_secs_f64() / 3.0 * 1000.0;

            // Old optimized parallel algorithm
            let start = Instant::now();
            for _ in 0..3 {
                let _ = crate::ml::cluster_optimized::hdbscan_optimized(
                    data.view(),
                    Some(10),
                    Some(10),
                );
            }
            let old_time = start.elapsed().as_secs_f64() / 3.0 * 1000.0;

            let speedup = old_time / new_time;
            println!(
                "{}\t{:.1}\t\t{:.1}\t\t{:.2}x",
                n, new_time, old_time, speedup
            );
        }
    }
}
