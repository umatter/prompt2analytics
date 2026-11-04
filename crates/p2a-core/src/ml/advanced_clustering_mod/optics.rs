//! OPTICS as R's `dbscan::optics` (dbscan 1.2.6, `optics_int`), with the flat clusterings
//! `extractDBSCAN` ([`OpticsResult::extract_dbscan`]) and `extractXi` ([`OpticsResult::extract_xi`]).
//!
//! - The core distance counts the point itself (`minPts`): the distance to the `(min_samples - 1)`-th
//!   nearest other point within `eps`, and undefined (`INFINITY`) when fewer than `min_samples` points
//!   (the point included) lie within `eps`.
//! - The ordering, reachabilities and predecessors are dbscan's: reachabilities are compared as squared
//!   distances, the next point is the smallest reachability with the higher id on ties, and a point's
//!   predecessor is the point that last strictly lowered its reachability.
//!
//! # References
//!
//! - Ankerst, M., Breunig, M.M., Kriegel, H.P., and Sander, J. (1999). "OPTICS: Ordering Points To
//!   Identify the Clustering Structure". ACM SIGMOD 1999, 49-60.
//! - Schubert, E. and Gertz, M. (2018). "Improving the Cluster Structure Extracted from OPTICS Plots".
//!   LWDA 2018, 318-329 (the Xi corrections).
//! - Hahsler, M., Piekenbrock, M., and Doran, D. (2019). "dbscan: Fast Density-Based Clustering with R".
//!   Journal of Statistical Software 91(1).

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::common::{Points, check_points, select_algorithm, uses_tree};
use crate::errors::{EconError, EconResult};
use crate::ml::kdtree::KdTree;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::validate_level;

// =============================================================================
// Configuration and result
// =============================================================================

/// The flat clustering [`optics_with`] extracts from the ordering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum OpticsExtraction {
    /// No flat clustering (R: `optics()` alone): every label is -1.
    #[default]
    NoClustering,
    /// `dbscan::extractDBSCAN(object, eps_cl)` (Ankerst et al. 1999, ExtractDBSCAN-Clustering).
    Dbscan {
        /// The DBSCAN radius, `0 < eps_cl <= eps`.
        eps_cl: f64,
    },
    /// `dbscan::extractXi(object, xi, minimum = FALSE, correctPredecessors)`.
    Xi {
        /// Steepness, strictly between 0 and 1.
        xi: f64,
        /// ELKI's predecessor correction (R's default `TRUE`).
        correct_predecessors: bool,
    },
}

/// Parameters of [`optics_with`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpticsConfig {
    /// `minPts`, counting the point itself; `1..=n`. Default 5 (R).
    pub min_samples: usize,
    /// Neighbourhood radius; `None` (or `+inf`) = no limit (R's default). Otherwise positive.
    pub eps: Option<f64>,
    /// The flat clustering to extract.
    pub extraction: OpticsExtraction,
}

impl Default for OpticsConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            eps: None,
            extraction: OpticsExtraction::NoClustering,
        }
    }
}

/// One xi-cluster: 0-based inclusive positions in the ordering (R `clusters_xi$start - 1`, `$end - 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XiCluster {
    /// First position in the ordering.
    pub start: usize,
    /// Last position in the ordering (inclusive).
    pub end: usize,
}

/// Result of OPTICS clustering ([`optics_with`]; R `dbscan::optics`).
///
/// # References
///
/// - Ankerst, M., Breunig, M.M., Kriegel, H.P., and Sander, J. (1999).
///   "OPTICS: Ordering Points To Identify the Clustering Structure".
///   ACM SIGMOD 1999.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticsResult {
    /// Ordering of points (indices in processing order)
    pub ordering: Vec<usize>,
    /// Reachability distance of each point, in ordering order; `INFINITY` = undefined (the first
    /// point of each connected part). [`OpticsResult::reachability_by_point`] gives R's `reachdist`.
    pub reachability: Vec<f64>,
    /// Core distance of each point; `INFINITY` = fewer than `min_samples` points within `eps`.
    pub core_distances: Vec<f64>,
    /// Cluster label of each point (-1 for noise) from the extraction in `extraction`.
    pub labels: Vec<i32>,
    /// Number of distinct clusters in `labels`
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// `minPts` used, counting the point itself
    pub min_samples: usize,
    /// Xi parameter of the extraction, when it is `extractXi`
    pub xi: Option<f64>,
    /// Number of observations
    pub n: usize,
    /// Neighbourhood radius; `None` = infinite.
    #[serde(default)]
    pub eps: Option<f64>,
    /// dbscan's predecessor of each point (`None` for the first point of each connected part and for
    /// points that are never reached).
    #[serde(default)]
    pub predecessor: Vec<Option<usize>>,
    /// The flat clustering in `labels`.
    #[serde(default)]
    pub extraction: OpticsExtraction,
    /// The xi-clusters of an `extractXi` extraction, sorted by (start, end); cluster id = position.
    #[serde(default)]
    pub xi_clusters: Vec<XiCluster>,
    /// Warnings about the fit.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for OpticsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for OpticsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OPTICS Clustering Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "MinPts: {}", self.min_samples)?;
        match self.eps {
            Some(e) => writeln!(f, "eps: {}", e)?,
            None => writeln!(f, "eps: inf")?,
        }
        if let Some(xi) = self.xi {
            writeln!(f, "Xi: {:.4}", xi)?;
        }
        match self.extraction {
            OpticsExtraction::NoClustering => writeln!(f, "Extraction: none")?,
            OpticsExtraction::Dbscan { eps_cl } => {
                writeln!(f, "Extraction: extractDBSCAN (eps_cl = {})", eps_cl)?
            }
            OpticsExtraction::Xi { xi, .. } => writeln!(
                f,
                "Extraction: extractXi (xi = {}; {} xi-clusters)",
                xi,
                self.xi_clusters.len()
            )?,
        }
        writeln!(f)?;
        writeln!(
            f,
            "Reachability plot data available in 'ordering' and 'reachability' fields"
        )?;

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
        write!(f, "{}", notes_text(self))
    }
}

impl OpticsResult {
    /// Reachability distance by point (R `reachdist`): the inverse of the ordering.
    pub fn reachability_by_point(&self) -> Vec<f64> {
        let mut v = vec![f64::INFINITY; self.n];
        for (pos, &p) in self.ordering.iter().enumerate() {
            v[p] = self.reachability[pos];
        }
        v
    }

    /// `n_clusters` = distinct non-negative labels, `n_noise` = labels -1.
    fn recount(&mut self) {
        let mut seen: Vec<bool> = Vec::new();
        let mut k = 0usize;
        let mut noise = 0usize;
        for &l in &self.labels {
            if l < 0 {
                noise += 1;
                continue;
            }
            let l = l as usize;
            if l >= seen.len() {
                seen.resize(l + 1, false);
            }
            if !seen[l] {
                seen[l] = true;
                k += 1;
            }
        }
        self.n_clusters = k;
        self.n_noise = noise;
    }

    /// The DBSCAN clustering at radius `eps_cl` from the ordering (R `dbscan::extractDBSCAN`; Ankerst et
    /// al. 1999, ExtractDBSCAN-Clustering): walking the ordering, a point whose reachability exceeds
    /// `eps_cl` starts a new cluster if its core distance is at most `eps_cl` and is noise otherwise;
    /// every other point joins the current cluster.
    ///
    /// # Errors
    /// `InvalidSpecification` unless `0 < eps_cl <= eps` and `eps_cl` is finite.
    pub fn extract_dbscan(&self, eps_cl: f64) -> EconResult<OpticsResult> {
        let limit = self.eps.unwrap_or(f64::INFINITY);
        if !(eps_cl.is_finite() && eps_cl > 0.0 && eps_cl <= limit) {
            let eps = match self.eps {
                Some(e) => e.to_string(),
                None => "inf".to_string(),
            };
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "extract_dbscan: eps_cl must be positive, finite and at most eps = {eps} (got {eps_cl})"
                ),
            });
        }
        let mut out = self.clone();
        let reach = self.reachability_by_point();
        let mut cid: i32 = -1; // R's clusterid 0 (noise) is -1 here
        for &p in &self.ordering {
            out.labels[p] = if reach[p] > eps_cl {
                if self.core_distances[p] <= eps_cl {
                    cid += 1;
                    cid
                } else {
                    -1
                }
            } else {
                cid
            };
        }
        out.extraction = OpticsExtraction::Dbscan { eps_cl };
        out.xi = None;
        out.xi_clusters = Vec::new();
        out.warnings.retain(|w| !is_extraction_warning(w));
        out.recount();
        Ok(out)
    }

    /// The xi-clusters of the ordering and their flat labelling (R `dbscan::extractXi(object, xi,
    /// minimum = FALSE, correctPredecessors)`): steep down and steep up areas of the reachability plot
    /// (Ankerst et al. 1999), the mib bookkeeping, the `minPts` rule and, with `correct_predecessors`,
    /// ELKI's predecessor correction (Schubert and Gertz 2018). Labels as `extractClusterLabels`: every
    /// point takes the smallest xi-cluster that contains it; points in none are noise.
    ///
    /// # Errors
    /// `InvalidSpecification` unless `0 < xi < 1`.
    pub fn extract_xi(&self, xi: f64, correct_predecessors: bool) -> EconResult<OpticsResult> {
        validate_level(xi, "xi")?;
        let n = self.ordering.len();
        let clusters = xi_clusters(
            &self.reachability,
            &self.ordering,
            &self.predecessor,
            self.min_samples,
            xi,
            correct_predecessors,
        );
        // extractClusterLabels(minimum = FALSE): larger clusters first, smaller ones overwrite them
        let mut by_size: Vec<usize> = (0..clusters.len()).collect();
        by_size.sort_by(|&a, &b| {
            let (sa, sb) = (clusters[a].1 - clusters[a].0, clusters[b].1 - clusters[b].0);
            sb.cmp(&sa) // stable: R's order(-cluster_size)
        });
        let mut lab_pos = vec![-1i32; n];
        for &c in &by_size {
            let (s, e) = clusters[c];
            for slot in &mut lab_pos[s - 1..e] {
                *slot = c as i32;
            }
        }
        let mut out = self.clone();
        for (pos, &p) in self.ordering.iter().enumerate() {
            out.labels[p] = lab_pos[pos];
        }
        out.xi_clusters = clusters
            .iter()
            .map(|&(s, e)| XiCluster {
                start: s - 1,
                end: e - 1,
            })
            .collect();
        out.extraction = OpticsExtraction::Xi {
            xi,
            correct_predecessors,
        };
        out.xi = Some(xi);
        out.warnings.retain(|w| !is_extraction_warning(w));
        if clusters.is_empty() {
            out.warnings.push(format!(
                "extractXi found no cluster with xi = {xi}; every point is noise (R warns 'No clusters were found with threshold: {xi}')"
            ));
        }
        out.recount();
        Ok(out)
    }
}

const NO_EXTRACTION_WARNING: &str = "no flat clustering was extracted";
const NO_XI_CLUSTER_WARNING: &str = "extractXi found no cluster";

fn is_extraction_warning(w: &str) -> bool {
    w.starts_with(NO_EXTRACTION_WARNING) || w.starts_with(NO_XI_CLUSTER_WARNING)
}

// =============================================================================
// Entry points
// =============================================================================

/// Run OPTICS (R `dbscan::optics`) and the flat clustering of `config.extraction`.
///
/// # Errors
/// `EmptyDataset` for no rows; `InvalidSpecification` for zero columns, non-finite values, coordinates
/// whose squared distances overflow, `min_samples` outside `1..=n`, a non-positive `eps`, or invalid
/// extraction parameters.
pub fn optics_with(data: ArrayView2<f64>, config: &OpticsConfig) -> EconResult<OpticsResult> {
    let n = data.nrows();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }
    check_points(&data, "optics")?;
    let ms = config.min_samples;
    if ms < 1 || ms > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "optics: min_samples must be between 1 and n = {n} (got {ms}); it counts the point itself, as dbscan's minPts"
            ),
        });
    }
    let eps = match config.eps {
        None => f64::INFINITY,
        Some(e) if e > 0.0 => e,
        Some(e) => {
            return Err(EconError::InvalidSpecification {
                message: format!("optics: eps must be positive (got {e}); use None for no limit"),
            });
        }
    };
    let pts = Points::new(&data, "optics data")?;
    let ord = optics_ordering(&pts, ms, eps, uses_tree(select_algorithm(pts.n, pts.d)))?;
    let reachability = ord.order.iter().map(|&p| ord.reach2[p].sqrt()).collect();
    let base = OpticsResult {
        reachability,
        core_distances: ord.core2.iter().map(|c| c.sqrt()).collect(),
        ordering: ord.order,
        labels: vec![-1; n],
        n_clusters: 0,
        n_noise: n,
        min_samples: ms,
        xi: None,
        n,
        eps: if eps.is_finite() { Some(eps) } else { None },
        predecessor: ord.predecessor,
        extraction: OpticsExtraction::NoClustering,
        xi_clusters: Vec::new(),
        warnings: Vec::new(),
    };
    match config.extraction {
        OpticsExtraction::NoClustering => Ok(base),
        OpticsExtraction::Dbscan { eps_cl } => base.extract_dbscan(eps_cl),
        OpticsExtraction::Xi {
            xi,
            correct_predecessors,
        } => base.extract_xi(xi, correct_predecessors),
    }
}

/// Run OPTICS clustering (R `dbscan::optics`); see [`optics_with`].
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `min_samples` - `minPts`, counting the point itself (`1..=n`)
/// * `max_eps` - Neighbourhood radius (default: infinity)
/// * `xi` - Steepness for `extractXi`
///
/// The flat clustering: with `xi`, `extractXi(xi)` with the predecessor correction; without `xi` and
/// with a finite `max_eps`, `extractDBSCAN(eps_cl = max_eps)`; otherwise none (every label -1 and a
/// warning; call [`OpticsResult::extract_dbscan`] or [`OpticsResult::extract_xi`] on the result), as
/// R's `optics()` alone.
pub fn optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    let extraction = match (xi, max_eps) {
        (Some(x), _) => OpticsExtraction::Xi {
            xi: x,
            correct_predecessors: true,
        },
        (None, Some(e)) if e.is_finite() => OpticsExtraction::Dbscan { eps_cl: e },
        _ => OpticsExtraction::NoClustering,
    };
    let mut r = optics_with(
        data,
        &OpticsConfig {
            min_samples,
            eps: max_eps,
            extraction,
        },
    )
    .map_err(|e| e.to_string())?;
    if extraction == OpticsExtraction::NoClustering {
        r.warnings.push(format!(
            "{NO_EXTRACTION_WARNING}: max_eps is infinite and no xi was given; call extract_dbscan(eps_cl) or extract_xi(xi) on the result (R: extractDBSCAN, extractXi)"
        ));
    }
    Ok(r)
}

/// Convenience wrapper for optics.
pub fn run_optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    optics(data, min_samples, max_eps, xi)
}

// =============================================================================
// The ordering (dbscan optics_int)
// =============================================================================

/// The OPTICS ordering in squared distances, as dbscan keeps it (ties compare exactly as in R).
pub(crate) struct OpticsOrdering {
    pub(crate) order: Vec<usize>,
    pub(crate) reach2: Vec<f64>,
    pub(crate) core2: Vec<f64>,
    pub(crate) predecessor: Vec<Option<usize>>,
}

/// Seeds of the ordering: an indexed binary min-heap over point ids keyed by `reach2`, the higher id
/// first on equal keys (dbscan/ELKI "higher ID wins"). Decrease-key keeps one entry per point, O(n).
struct SeedHeap {
    heap: Vec<usize>,
    pos: Vec<usize>, // NONE = not in the heap
}

const NONE: usize = usize::MAX;

impl SeedHeap {
    fn new(n: usize) -> EconResult<Self> {
        Ok(Self {
            heap: resource::try_with_capacity(n, "optics seeds")?,
            pos: resource::try_vec(n, NONE, "optics seeds")?,
        })
    }

    /// `a` comes before `b`: smaller reachability, then the higher id.
    #[inline]
    fn before(key: &[f64], a: usize, b: usize) -> bool {
        match key[a].total_cmp(&key[b]) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => a > b,
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.pos[self.heap[i]] = i;
        self.pos[self.heap[j]] = j;
    }

    fn sift_up(&mut self, key: &[f64], mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if Self::before(key, self.heap[i], self.heap[p]) {
                self.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, key: &[f64], mut i: usize) {
        let len = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut best = i;
            if l < len && Self::before(key, self.heap[l], self.heap[best]) {
                best = l;
            }
            if r < len && Self::before(key, self.heap[r], self.heap[best]) {
                best = r;
            }
            if best == i {
                break;
            }
            self.swap(i, best);
            i = best;
        }
    }

    /// Insert `id`, or move it up after its key decreased.
    fn push_or_decrease(&mut self, key: &[f64], id: usize) {
        let i = self.pos[id];
        if i == NONE {
            self.heap.push(id);
            let at = self.heap.len() - 1;
            self.pos[id] = at;
            self.sift_up(key, at);
        } else {
            self.sift_up(key, i);
        }
    }

    fn pop(&mut self, key: &[f64]) -> Option<usize> {
        let top = *self.heap.first()?;
        let last = self.heap.len() - 1;
        self.swap(0, last);
        self.heap.pop();
        self.pos[top] = NONE;
        if !self.heap.is_empty() {
            self.sift_down(key, 0);
        }
        Some(top)
    }
}

/// The eps-neighbourhood of `p` as (index, squared distance), the point itself included (ANN's
/// inclusive `sqDist <= sqRad`). The k-d tree only proposes candidates; every kept distance is
/// `pts.sq(p, j)`, so both paths give the same neighbourhood.
fn region(
    pts: &Points,
    tree: Option<&KdTree>,
    p: usize,
    eps: f64,
    eps2: f64,
    out: &mut Vec<(usize, f64)>,
) {
    out.clear();
    match tree {
        Some(t) if eps.is_finite() => {
            for (_, j) in t.radius_query_unsorted(pts.row(p), eps * (1.0 + 1e-12), None) {
                let d2 = pts.sq(p, j);
                if d2 <= eps2 {
                    out.push((j, d2));
                }
            }
        }
        _ => {
            for j in 0..pts.n {
                let d2 = pts.sq(p, j);
                if d2 <= eps2 {
                    out.push((j, d2));
                }
            }
        }
    }
}

/// The squared core distance: the `min_samples`-th smallest squared distance in the neighbourhood (the
/// point itself is at 0, so this is the `(min_samples - 1)`-th other point), `INFINITY` when the
/// neighbourhood holds fewer than `min_samples` points.
fn set_core(nbhd: &[(usize, f64)], min_samples: usize, scratch: &mut Vec<f64>) -> f64 {
    if nbhd.len() < min_samples {
        return f64::INFINITY;
    }
    scratch.clear();
    scratch.extend(nbhd.iter().map(|&(_, d2)| d2));
    let (_, kth, _) = scratch.select_nth_unstable_by(min_samples - 1, f64::total_cmp);
    *kth
}

/// dbscan's `update()`: lower the reachability of every unvisited neighbour `o` to
/// `max(core(p), d(p, o))`; a strict decrease of a defined reachability records `p` as predecessor.
fn update(
    nbhd: &[(usize, f64)],
    p: usize,
    visited: &[bool],
    core2: &[f64],
    reach2: &mut [f64],
    predecessor: &mut [Option<usize>],
    seeds: &mut SeedHeap,
) {
    for &(o, d2) in nbhd {
        if visited[o] {
            continue;
        }
        let nr = core2[p].max(d2);
        if reach2[o].is_infinite() {
            reach2[o] = nr;
            seeds.push_or_decrease(reach2, o);
        } else if nr < reach2[o] {
            reach2[o] = nr;
            predecessor[o] = Some(p);
            seeds.push_or_decrease(reach2, o);
        }
    }
}

/// dbscan's `optics_int`: the ordering, squared reachabilities and core distances and the predecessors.
/// A checkpoint per point; O(n) memory besides one neighbourhood.
pub(crate) fn optics_ordering(
    pts: &Points,
    min_samples: usize,
    eps: f64,
    use_tree: bool,
) -> EconResult<OpticsOrdering> {
    let n = pts.n;
    let eps2 = eps * eps;
    let tree = if use_tree && eps.is_finite() {
        Some(KdTree::new(pts.rows_vec()))
    } else {
        None
    };
    let mut visited = resource::try_vec(n, false, "optics")?;
    let mut order = resource::try_with_capacity(n, "optics ordering")?;
    let mut predecessor: Vec<Option<usize>> = resource::try_vec(n, None, "optics predecessors")?;
    let mut reach2 = resource::try_vec(n, f64::INFINITY, "optics reachability")?;
    let mut core2 = resource::try_vec(n, f64::INFINITY, "optics core distances")?;
    let mut nbhd: Vec<(usize, f64)> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    let mut seeds = SeedHeap::new(n)?;
    for p in 0..n {
        if visited[p] {
            continue;
        }
        resource::checkpoint()?;
        region(pts, tree.as_ref(), p, eps, eps2, &mut nbhd);
        visited[p] = true;
        core2[p] = set_core(&nbhd, min_samples, &mut scratch);
        let tmp_p = if predecessor[p].is_none() {
            Some(p)
        } else {
            None
        };
        order.push(p);
        if core2[p].is_infinite() {
            continue; // core distance undefined (fewer than min_samples points within eps)
        }
        update(
            &nbhd,
            p,
            &visited,
            &core2,
            &mut reach2,
            &mut predecessor,
            &mut seeds,
        );
        while let Some(q) = seeds.pop(&reach2) {
            resource::checkpoint()?;
            region(pts, tree.as_ref(), q, eps, eps2, &mut nbhd);
            visited[q] = true;
            core2[q] = set_core(&nbhd, min_samples, &mut scratch);
            if predecessor[q].is_none() {
                predecessor[q] = tmp_p;
            }
            order.push(q);
            if nbhd.len() < min_samples {
                continue;
            }
            update(
                &nbhd,
                q,
                &visited,
                &core2,
                &mut reach2,
                &mut predecessor,
                &mut seeds,
            );
        }
    }
    Ok(OpticsOrdering {
        order,
        reach2,
        core2,
        predecessor,
    })
}

// =============================================================================
// extractXi
// =============================================================================

/// R's `extractXi` on the reachabilities in ordering order: the xi-clusters as 1-based inclusive
/// (start, end) positions, sorted.
fn xi_clusters(
    ord_rd: &[f64],
    order: &[usize],
    predecessor: &[Option<usize>],
    min_pts: usize,
    xi: f64,
    correct_predecessors: bool,
) -> Vec<(usize, usize)> {
    #[derive(Clone, Copy)]
    struct Sda {
        s: usize,
        maximum: f64,
        mib: f64,
    }
    let n = ord_rd.len();
    let ixi = 1.0 - xi;
    let valid = |i: usize| (1..=n).contains(&i);
    let r = |i: usize| ord_rd[i - 1];
    let steep_up =
        |i: usize, f: f64| !r(i).is_infinite() && (!valid(i + 1) || r(i) <= r(i + 1) * f);
    let steep_down =
        |i: usize, f: f64| valid(i + 1) && !r(i + 1).is_infinite() && r(i) * f >= r(i + 1);
    let update_filter = |mib: f64, set: &mut Vec<Sda>| {
        set.retain(|s| s.maximum * ixi > mib);
        for s in set.iter_mut() {
            if mib > s.mib {
                s.mib = mib;
            }
        }
    };
    // 1-based position of every point in the ordering, for the predecessor test
    let mut pos1 = vec![0usize; predecessor.len().max(n)];
    for (k, &p) in order.iter().enumerate() {
        pos1[p] = k + 1;
    }
    let pred = |p: usize| predecessor.get(p).copied().flatten();
    let (mut sdaset, mut clusters) = (Vec::<Sda>::new(), Vec::<(usize, usize)>::new());
    let (mut index, mut mib) = (1usize, 0.0f64);
    while index <= n {
        mib = mib.max(r(index));
        if !valid(index + 1) {
            break;
        }
        if steep_down(index, ixi) {
            update_filter(mib, &mut sdaset);
            let (startval, start) = (r(index), index);
            mib = 0.0;
            let mut end = index + 1;
            while valid(index + 1) {
                index += 1;
                if steep_down(index, ixi) {
                    end = index + 1;
                    continue;
                }
                if !steep_down(index, 1.0) || index > end + min_pts {
                    break; // R: index - endsteep > minPts (index >= end here)
                }
            }
            sdaset.push(Sda {
                s: start,
                maximum: startval,
                mib: 0.0,
            });
            continue;
        }
        if steep_up(index, ixi) {
            update_filter(mib, &mut sdaset);
            let mut end = index + 1;
            mib = r(index);
            let mut esuccr = if valid(index + 1) {
                r(index + 1)
            } else {
                f64::INFINITY
            };
            if esuccr.is_finite() {
                while valid(index + 1) {
                    index += 1;
                    if steep_up(index, ixi) {
                        end = index + 1;
                        mib = r(index);
                        esuccr = if valid(index + 1) {
                            r(index + 1)
                        } else {
                            f64::INFINITY
                        };
                        if esuccr.is_infinite() {
                            end -= 1;
                            break;
                        }
                        continue;
                    }
                    if !steep_up(index, 1.0) || index > end + min_pts {
                        break;
                    }
                }
            } else {
                end -= 1;
                index += 1;
            }
            let (sua_end, sua_max) = (end, esuccr);
            for sda in sdaset.iter().rev() {
                if mib * ixi < sda.mib {
                    continue; // condition 3b
                }
                let (mut cstart, mut cend) = (sda.s, sua_end);
                if correct_predecessors {
                    while cend > cstart && r(cend).is_infinite() {
                        cend -= 1;
                    }
                }
                if sda.maximum * ixi >= sua_max {
                    while cstart < cend && r(cstart + 1) > sua_max {
                        cstart += 1; // condition 4b
                    }
                } else if sua_max * ixi >= sda.maximum {
                    while cend > cstart && r(cend - 1) > sda.maximum {
                        cend -= 1; // condition 4c
                    }
                }
                if correct_predecessors {
                    // ELKI: the end point's predecessor must lie in the cluster, at a 1-based
                    // position in cstart..=cend - 1
                    while cend > cstart {
                        if let Some(q) = pred(order[cend - 1])
                            && pos1[q] >= cstart
                            && pos1[q] < cend
                        {
                            break;
                        }
                        cend -= 1;
                    }
                }
                if index >= 2 && steep_up(index - 1, ixi) {
                    cend -= 1; // cend >= cstart >= 1 here
                }
                if cend + 1 < cstart + min_pts {
                    continue; // R: cend - cstart + 1 < minPts, without underflow
                }
                clusters.push((cstart, cend));
            }
        } else {
            index += 1;
        }
    }
    clusters.sort_unstable(); // R: order(start, end); equal tuples are indistinguishable
    clusters
}

#[cfg(test)]
mod tests {
    use super::super::common::Points;
    use super::super::common::test_data::{blobs, na, r_codes, uniform};
    use super::*;
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    fn codes(v: &[usize]) -> Vec<f64> {
        v.iter().map(|&p| (p + 1) as f64).collect()
    }
    fn pred(v: &[Option<usize>]) -> Vec<f64> {
        v.iter()
            .map(|p| p.map_or(f64::NAN, |q| (q + 1) as f64))
            .collect()
    }
    fn by_point(r: &OpticsResult) -> Vec<f64> {
        let mut v = vec![f64::INFINITY; r.n];
        for (pos, &p) in r.ordering.iter().enumerate() {
            v[p] = r.reachability[pos];
        }
        v
    }

    #[test]
    fn test_optics_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = optics(data.view(), 2, None, None).unwrap();

        assert_eq!(result.ordering.len(), 6);
        assert_eq!(result.reachability.len(), 6);
    }

    /// R: dbscan::optics(blobs_noise, minPts = 5) (eps = Inf). Case optics_blobs.
    #[test]
    fn test_validate_optics_inf() {
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let r = optics(x.view(), 5, None, None).unwrap();
        Check::new(&case)
            .vector("inf_order", &codes(&r.ordering), Tol::exact())
            .vector("inf_reachdist", &na(&by_point(&r)), Tol::coef(n))
            .vector("inf_coredist", &r.core_distances, Tol::coef(n))
            .finish();
    }

    /// R: dbscan::optics(blobs_noise, eps = 0.5, minPts = 5): 14 core distances undefined (ML-45).
    #[test]
    fn test_validate_optics_eps05() {
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = blobs(&ds);
        let n = x.nrows();
        let r = optics(x.view(), 5, Some(0.5), None).unwrap();
        Check::new(&case)
            .vector("eps05_order", &codes(&r.ordering), Tol::exact())
            .vector("eps05_reachdist", &na(&by_point(&r)), Tol::coef(n))
            .vector("eps05_coredist", &na(&r.core_distances), Tol::coef(n))
            .finish();
        assert_eq!(
            r.core_distances.iter().filter(|c| c.is_infinite()).count(),
            14
        );
    }

    /// dbscan's predecessor bookkeeping (pre[o] = the point that last strictly improved o's reachability,
    /// else the component's first point; NA for that first point). Case optics_blobs.
    #[test]
    fn test_validate_optics_predecessor() {
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = blobs(&ds);
        let inf = optics_with(
            x.view(),
            &OpticsConfig {
                min_samples: 5,
                eps: None,
                extraction: OpticsExtraction::NoClustering,
            },
        )
        .unwrap();
        let e05 = optics_with(
            x.view(),
            &OpticsConfig {
                min_samples: 5,
                eps: Some(0.5),
                extraction: OpticsExtraction::NoClustering,
            },
        )
        .unwrap();
        Check::new(&case)
            .vector("inf_predecessor", &pred(&inf.predecessor), Tol::exact())
            .vector("eps05_predecessor", &pred(&e05.predecessor), Tol::exact())
            .vector(
                "inf_reachdist",
                &na(&inf.reachability_by_point()),
                Tol::coef(115),
            )
            .finish();
        assert!(inf.warnings.is_empty()); // no extraction was asked for
    }

    /// k-d tree eps-neighbourhoods (n >= 500, d <= 15) give the brute-force ordering bit for bit.
    #[test]
    fn test_optics_kdtree_neighbourhoods_match_brute_force() {
        let x = uniform(600, 2, 314);
        let pts = Points::new(&x.view(), "test").unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (ms, eps) in [(5usize, 0.05), (9, 0.08), (3, f64::INFINITY)] {
            let a = optics_ordering(&pts, ms, eps, true).unwrap();
            let b = optics_ordering(&pts, ms, eps, false).unwrap();
            assert_eq!(a.order, b.order);
            assert_eq!(a.predecessor, b.predecessor);
            assert_eq!(bits(&a.reach2), bits(&b.reach2));
            assert_eq!(bits(&a.core2), bits(&b.core2));
        }
    }

    #[test]
    fn test_optics_rejects_bad_parameters() {
        let x = uniform(20, 2, 5);
        for ms in [0usize, 21] {
            assert!(
                optics(x.view(), ms, None, None)
                    .unwrap_err()
                    .contains("min_samples")
            );
        }
        for eps in [0.0, -1.0, f64::NAN] {
            assert!(
                optics(x.view(), 5, Some(eps), None)
                    .unwrap_err()
                    .contains("eps must be positive"),
                "{eps}"
            );
        }
        let mut bad = x.clone();
        bad[[2, 0]] = f64::INFINITY;
        assert!(
            optics(bad.view(), 5, None, None)
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(optics(x.view(), 1, Some(f64::INFINITY), None).is_ok()); // minPts 1: core distance 0 (R allows it)
    }

    /// R: extractDBSCAN(optics(X, eps, minPts = 5), eps_cl = 0.35) for eps Inf, 0.5 and 0.35 (the legacy
    /// optics(.., Some(0.35), None) call extracts at eps_cl = max_eps). Case optics_blobs.
    #[test]
    fn test_validate_optics_extract_dbscan() {
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = blobs(&ds);
        let legacy = optics(x.view(), 5, Some(0.35), None).unwrap();
        let inf = optics_with(x.view(), &OpticsConfig::default())
            .unwrap()
            .extract_dbscan(0.35)
            .unwrap();
        let e05 = optics_with(
            x.view(),
            &OpticsConfig {
                eps: Some(0.5),
                ..OpticsConfig::default()
            },
        )
        .unwrap();
        let e05 = e05.extract_dbscan(0.35).unwrap();
        Check::new(&case)
            .vector("eps035_order", &codes(&legacy.ordering), Tol::exact())
            .vector(
                "eps035_dbscan_cluster",
                &r_codes(&legacy.labels),
                Tol::exact(),
            )
            .vector("inf_dbscan_cluster", &r_codes(&inf.labels), Tol::exact())
            .vector("eps05_dbscan_cluster", &r_codes(&e05.labels), Tol::exact())
            .finish();
        assert_eq!((inf.n_clusters, inf.n_noise), (3, 20)); // n_clusters counted distinct labels (ML-6)
        assert_eq!(legacy.extraction, OpticsExtraction::Dbscan { eps_cl: 0.35 });
        assert!(inf.to_string().contains("extractDBSCAN"));
    }

    /// R: extractXi(optics(X, minPts = 5), xi = 0.05) with correctPredecessors TRUE and FALSE, and on the
    /// eps = 0.5 ordering (infinite reachabilities inside the ordering). Case optics_blobs.
    #[test]
    fn test_validate_optics_extract_xi() {
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = blobs(&ds);
        let legacy = optics(x.view(), 5, None, Some(0.05)).unwrap();
        let base = optics_with(x.view(), &OpticsConfig::default()).unwrap();
        let nocorr = base.extract_xi(0.05, false).unwrap();
        let e05 = optics_with(
            x.view(),
            &OpticsConfig {
                eps: Some(0.5),
                ..OpticsConfig::default()
            },
        )
        .unwrap();
        let e05 = e05.extract_xi(0.05, true).unwrap();
        let starts = |r: &OpticsResult| {
            r.xi_clusters
                .iter()
                .map(|c| (c.start + 1) as f64)
                .collect::<Vec<_>>()
        };
        let ends = |r: &OpticsResult| {
            r.xi_clusters
                .iter()
                .map(|c| (c.end + 1) as f64)
                .collect::<Vec<_>>()
        };
        Check::new(&case)
            .vector("inf_xi_cluster", &r_codes(&legacy.labels), Tol::exact())
            .vector("inf_xi_clusters_start", &starts(&legacy), Tol::exact())
            .vector("inf_xi_clusters_end", &ends(&legacy), Tol::exact())
            .vector(
                "inf_xi_nocorrect_cluster",
                &r_codes(&nocorr.labels),
                Tol::exact(),
            )
            .vector(
                "inf_xi_nocorrect_clusters_start",
                &starts(&nocorr),
                Tol::exact(),
            )
            .vector(
                "inf_xi_nocorrect_clusters_end",
                &ends(&nocorr),
                Tol::exact(),
            )
            .vector("eps05_xi_cluster", &r_codes(&e05.labels), Tol::exact())
            .vector("eps05_xi_clusters_start", &starts(&e05), Tol::exact())
            .vector("eps05_xi_clusters_end", &ends(&e05), Tol::exact())
            .finish();
        assert_eq!(
            (legacy.n_clusters, legacy.n_noise, legacy.xi),
            (9, 0, Some(0.05))
        );
        assert_eq!((e05.n_clusters, e05.n_noise), (7, 9));
        assert_ne!(legacy.labels, nocorr.labels); // the two correctPredecessors values R distinguishes
    }

    #[test]
    fn test_optics_extraction_rejects_bad_parameters() {
        let x = uniform(40, 2, 11);
        let r = optics_with(
            x.view(),
            &OpticsConfig {
                eps: Some(0.3),
                ..OpticsConfig::default()
            },
        )
        .unwrap();
        for xi in [0.0, 1.0, -0.2, f64::NAN] {
            assert!(
                r.extract_xi(xi, true)
                    .unwrap_err()
                    .to_string()
                    .contains("xi must be strictly between 0 and 1"),
                "{xi}"
            );
        }
        for eps_cl in [0.0, -1.0, f64::NAN, f64::INFINITY, 0.31] {
            assert!(
                r.extract_dbscan(eps_cl)
                    .unwrap_err()
                    .to_string()
                    .contains("eps_cl"),
                "{eps_cl}"
            );
        }
        assert!(
            optics(x.view(), 5, None, Some(1.5))
                .unwrap_err()
                .contains("xi")
        );
        let none = optics(x.view(), 5, None, None).unwrap(); // no extraction without max_eps or xi
        assert_eq!((none.n_clusters, none.n_noise), (0, 40));
        assert!(none.warnings.iter().any(|w| w.contains("extract_dbscan")));
    }
}
