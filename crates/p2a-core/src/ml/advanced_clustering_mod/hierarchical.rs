//! Hierarchical clustering: DIANA ([`diana`]), AGNES ([`agnes`]), pvclust bootstrap p-values ([`pvclust`]) and fastcluster ([`fastcluster`]).

use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_with_capacity};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

use super::common::try_distance_matrix;
use crate::ml::extract::require_finite;
use crate::ml::hclust::{Dissimilarity, HclustTree, Linkage, agnes_banner_order, hclust_euclidean};
use crate::options::Choices;

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

/// Result of DIANA divisive hierarchical clustering (`cluster::diana`).
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data:
///   An Introduction to Cluster Analysis". Wiley, New York.
/// - R cluster::diana documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/diana.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DianaResult {
    /// Cluster assignments for each point (0 to n_clusters-1): the full divisive tree cut into
    /// `n_clusters` groups, as R's `cutree(as.hclust(diana(x)), k)` minus 1
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Divisive coefficient `dc` of the full tree (cluster's `bncoef`)
    pub divisive_coefficient: f64,
    /// One entry per split of the full divisive tree, in split order: (0-based position in
    /// `order` where the split cluster starts, size of its first part, size of its second part,
    /// level of the split = the split cluster's diameter)
    pub merge_history: Vec<(usize, usize, usize, f64)>,
    /// The levels of `merge_history`, in split order
    pub heights: Vec<f64>,
    /// Order of observations in the banner (R's `$order` - 1)
    pub order: Vec<usize>,
    /// Number of observations
    pub n: usize,
    /// The banner (R's `$height`): `banner[j]` is the level at which `order[j]` and
    /// `order[j + 1]` are separated
    #[serde(default)]
    pub banner: Vec<f64>,
    /// R's `$merge`: the agglomerative reading of the divisive tree (`-i` observation `i`,
    /// `+s` the cluster of step `s`, both 1-based)
    #[serde(default)]
    pub merge: Vec<(i64, i64)>,
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

/// Run DIANA divisive hierarchical clustering: `cluster::diana(x)`.
///
/// DIANA starts with all observations in one cluster and splits, at every step, the cluster of
/// largest diameter: the object with the largest average dissimilarity to the others starts a
/// splinter group, and objects move to it while their average dissimilarity to the rest
/// exceeds that to the splinter group. The full divisive tree is always built (as R does), so
/// the divisive coefficient, the banner, the order and the merge matrix equal `cluster::diana`;
/// `n_clusters` only cuts it (`cutree(as.hclust(diana(x)), k)`).
///
/// This follows cluster's `splyt` (`twins.c`, cluster 2.1.6) rule for rule (strict
/// comparisons, the ordered insertion, the switch of the two parts, the order of the next
/// cluster to split). The averages are kept as running sums per object, updated in O(m) when an
/// object moves, instead of being recomputed: a split of `m` objects costs O(m² + m·moves),
/// the full tree O(n²) on balanced data and O(n³) on chain-like data.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features); refused if any value is not finite
/// * `n_clusters` - Number of clusters to cut the tree into (default: 2)
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data", chapter 6.
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
    if n < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("DIANA needs at least 2 observations (got {n})"),
        });
    }
    require_finite(&data, "DIANA")
        .map_err(|message| EconError::InvalidSpecification { message })?;
    let d = Dissimilarity::euclidean(data, "DIANA dissimilarities")?;
    let cs = d.as_slice().iter().cloned().fold(0.0, f64::max);
    // 1-based objects and positions, as in twins.c
    let dd = |a: usize, b: usize| d.get(a - 1, b - 1);
    let what = "DIANA";
    let mut kwan = try_vec(n + 1, 0usize, what)?;
    let mut ban = try_vec(n + 1, 0.0f64, what)?;
    let mut ner: Vec<usize> = (0..=n).collect();
    // running sums per object: dissimilarities to the rest and to the splinter group
    let mut sum_rest = try_vec(n + 1, 0.0f64, what)?;
    let mut sum_spl = try_vec(n + 1, 0.0f64, what)?;
    let mut merge_history = Vec::with_capacity(n - 1);

    kwan[1] = n;
    let mut ja = 1usize;
    let mut nclu = 1usize;
    loop {
        checkpoint()?;
        let jb = ja + kwan[ja] - 1;
        if kwan[ja] == 2 {
            kwan[ja] = 1;
            kwan[jb] = 1;
            ban[jb] = dd(ner[ja], ner[jb]);
            merge_history.push((ja - 1, 1, 1, ban[jb]));
        } else {
            // One pass: each object's sum of dissimilarities to the cluster (in position order)
            // and the cluster's diameter.
            let mut bygsd = -1.0;
            let mut lndsd = ja;
            let mut diameter = 0.0f64;
            for l in ja..=jb {
                let mut sd = 0.0;
                for j in ja..=jb {
                    let v = dd(ner[l], ner[j]);
                    sd += v;
                    if j > l && diameter < v {
                        diameter = v;
                    }
                }
                sum_rest[ner[l]] = sd;
                sum_spl[ner[l]] = 0.0;
                if bygsd < sd {
                    bygsd = sd;
                    lndsd = l;
                }
            }
            let move_object =
                |o: usize, sum_rest: &mut [f64], sum_spl: &mut [f64], ner: &[usize]| {
                    for &j in &ner[ja..=jb] {
                        let v = dd(j, o);
                        sum_rest[j] -= v;
                        sum_spl[j] += v;
                    }
                };
            kwan[ja] -= 1;
            kwan[jb] = 1;
            if jb != lndsd {
                let lchan = ner[lndsd];
                ner.copy_within(lndsd + 1..=jb, lndsd);
                ner[jb] = lchan;
            }
            move_object(ner[jb], &mut sum_rest, &mut sum_spl, &ner);
            let mut splyn = 0usize;
            let mut jma = jb - 1;
            let mut jmb;
            loop {
                splyn += 1;
                let rest = (jma - ja) as f64;
                let mut jaway = ja;
                let mut bdyff = -1.0;
                for l in ja..=jma {
                    let da = sum_rest[ner[l]] / rest;
                    let db = sum_spl[ner[l]] / splyn as f64;
                    if bdyff < da - db {
                        bdyff = da - db;
                        jaway = l;
                    }
                }
                jmb = jma + 1;
                if bdyff <= 0.0 {
                    break;
                }
                if jma != jaway {
                    let lchan = ner[jaway];
                    ner.copy_within(jaway + 1..=jma, jaway);
                    ner[jma] = lchan;
                }
                move_object(ner[jma], &mut sum_rest, &mut sum_spl, &ner);
                for lxx in jmb..=jb {
                    let l1 = lxx - 1;
                    if ner[l1] < ner[lxx] {
                        break;
                    }
                    ner.swap(l1, lxx);
                }
                kwan[ja] -= 1;
                kwan[jma] = kwan[jmb] + 1;
                kwan[jmb] = 0;
                jma -= 1;
                jmb = jma + 1;
                if jma == ja {
                    break;
                }
            }
            // the part holding the smaller object index comes first
            if ner[ja] >= ner[jmb] {
                let mut lxxa = ja;
                for lgrb in jmb..=jb {
                    lxxa += 1;
                    let lchan = ner[lgrb];
                    ner.copy_within(lxxa - 1..lgrb, lxxa);
                    ner[lxxa - 1] = lchan;
                }
                let llq = kwan[jmb];
                kwan[jmb] = 0;
                jma = ja + jb - jma - 1;
                jmb = jma + 1;
                kwan[jmb] = kwan[ja];
                kwan[ja] = llq;
            }
            ban[jmb] = if nclu == 1 { cs } else { diameter };
            merge_history.push((ja - 1, jmb - ja, jb - jmb + 1, ban[jmb]));
        }
        nclu += 1;
        if nclu >= n {
            break;
        }
        // the next cluster to split: the next block of more than one object after this one,
        // else the first such block from the start
        let mut found = false;
        if jb != n {
            let mut ja2 = ja;
            loop {
                ja2 += kwan[ja2].max(1);
                if ja2 > n {
                    break;
                }
                if kwan[ja2] > 1 {
                    ja = ja2;
                    found = true;
                    break;
                }
            }
        }
        if !found {
            ja = 1;
            while kwan[ja] <= 1 {
                ja += kwan[ja].max(1);
                if ja > n {
                    return Err(EconError::Computation(
                        "DIANA: no cluster left to split".into(),
                    ));
                }
            }
        }
    }

    // R's merge from the banner: n - 1 passes, each joining the two neighbours in the banner
    // separated at the lowest level (the last such position on ties), as twins.c.
    let mut parent: Vec<usize> = (0..=n).collect();
    let mut step_of_root = vec![0i64; n + 1];
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut used = vec![false; n + 1];
    let mut merge = Vec::with_capacity(n - 1);
    let mut height = Vec::with_capacity(n - 1);
    for step in 1..n {
        let mut dmin = cs;
        let mut jj = 0usize;
        for j in 2..=n {
            if !used[j] && dmin >= ban[j] {
                dmin = ban[j];
                jj = j;
            }
        }
        if jj == 0 {
            return Err(EconError::Computation(
                "DIANA: the banner levels exceed the data diameter".into(),
            ));
        }
        used[jj] = true;
        let ra = find(&mut parent, jj - 1);
        let rb = find(&mut parent, jj);
        let side = |r: usize, pos: usize| {
            if step_of_root[r] > 0 {
                step_of_root[r]
            } else {
                -(ner[pos] as i64)
            }
        };
        let (l1, l2) = (side(ra, jj - 1), side(rb, jj));
        parent[rb] = ra;
        step_of_root[ra] = step as i64;
        merge.push((l1, l2));
        height.push(ban[jj]);
    }

    // dc: cluster's bncoef on the banner
    let sup = ban[2..=n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut cf = 0.0;
    for k in 0..n {
        let kearl = k.max(1);
        let kafte = (k + 1).min(n - 1);
        cf += 1.0 - ban[kearl + 1].min(ban[kafte + 1]) / sup;
    }
    let divisive_coefficient = cf / n as f64;

    let order: Vec<usize> = ner[1..=n].iter().map(|&o| o - 1).collect();
    let tree = HclustTree::from_parts(n, merge.clone(), height, order.clone())?;
    let labels = tree.cut_k(target_k)?.into_iter().map(|l| l - 1).collect();
    let heights = merge_history.iter().map(|m| m.3).collect();

    Ok(DianaResult {
        labels,
        n_clusters: target_k,
        divisive_coefficient,
        merge_history,
        heights,
        order,
        n,
        banner: ban[2..=n].to_vec(),
        merge,
    })
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

/// Linkage method for AGNES (`cluster::agnes`).
///
/// `Ward` is `agnes(method = "ward")`, which is `hclust`'s ward.D2; `Weighted` is WPGMA
/// (`hclust`'s mcquitty).
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

crate::option_enum!(
    AgnesLinkage,
    Choices {
        option: "linkage",
        values: &[
            (&["average", "upgma"], AgnesLinkage::Average),
            (&["single"], AgnesLinkage::Single),
            (&["complete"], AgnesLinkage::Complete),
            (&["ward"], AgnesLinkage::Ward),
            (&["weighted", "wpgma", "mcquitty"], AgnesLinkage::Weighted),
        ],
    }
);

impl From<AgnesLinkage> for Linkage {
    /// The `stats::hclust` method of each AGNES method (agnes' Ward is ward.D2).
    fn from(l: AgnesLinkage) -> Linkage {
        match l {
            AgnesLinkage::Average => Linkage::Average,
            AgnesLinkage::Single => Linkage::Single,
            AgnesLinkage::Complete => Linkage::Complete,
            AgnesLinkage::Ward => Linkage::Ward,
            AgnesLinkage::Weighted => Linkage::Mcquitty,
        }
    }
}

/// Run AGNES agglomerative hierarchical clustering: `cluster::agnes(x, method)`.
///
/// The full tree is built by the shared core [`crate::ml::hclust`] and cut into `n_clusters`
/// groups. On data without exactly tied dissimilarities the tree equals `cluster::agnes`'s
/// (merges and heights), and `order` is agnes' banner order. On ties the core merges the first
/// tied pair, as `stats::hclust` does, while `cluster::agnes` merges the last, so the order of
/// tied merges can differ from R's `agnes`.
///
/// The agglomerative coefficient is `ac = mean(1 - m_i / max height)`, with `m_i` the height at
/// which observation `i` first joins a cluster (cluster's `bncoef`); it is a property of the
/// full tree and does not depend on `n_clusters`.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features); refused if any value is not finite
/// * `n_clusters` - Number of clusters to cut the tree into (default: 2)
/// * `method` - Linkage method (default: average)
///
/// # Returns
/// * `AgnesResult`: `merge` is the full tree in SciPy linkage form (ids `< n` observations,
///   `n + s` the cluster of merge `s`), `heights` the merge heights in merge order
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data", chapter 5.
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
    if n < 2 {
        return Err(format!("AGNES needs at least 2 observations (got {n})"));
    }
    require_finite(&data, "AGNES")?;

    let linkage = method.unwrap_or(AgnesLinkage::Average);
    let tree = hclust_euclidean(data, linkage.into(), "AGNES dissimilarities")
        .map_err(|e| e.to_string())?;
    let labels = tree
        .cut_k(target_k)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|l| l - 1)
        .collect();

    Ok(AgnesResult {
        labels,
        n_clusters: target_k,
        agglomerative_coefficient: tree.agglomerative_coefficient(),
        merge: tree.scipy_linkage(),
        heights: tree.height().to_vec(),
        order: agnes_banner_order(&tree),
        method: format!("{:?}", linkage).to_lowercase(),
        n,
    })
}

/// Convenience wrapper for agnes.
pub fn run_agnes(
    data: ArrayView2<f64>,
    n_clusters: Option<usize>,
    method: Option<&str>,
) -> Result<AgnesResult, String> {
    let linkage = method
        .map(|m| m.parse::<AgnesLinkage>())
        .transpose()
        .map_err(|e| e.to_string())?;
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
///
/// | Variant | R method (`fastcluster::hclust`, `stats::hclust`) |
/// |---|---|
/// | `Single` | `"single"` |
/// | `Complete` | `"complete"` |
/// | `Average` | `"average"` |
/// | `Ward` (default) | `"ward.D2"` (`hclust.vector(method = "ward")`) |
/// | `Weighted` | `"mcquitty"` |
/// | `Centroid` | `"centroid"` (on squared distances; heights square-rooted) |
/// | `Median` | `"median"` (on squared distances; heights square-rooted) |
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

impl From<FastLinkage> for Linkage {
    /// The `stats::hclust` method of each variant (`Weighted` is mcquitty, `Ward` is ward.D2).
    fn from(l: FastLinkage) -> Linkage {
        match l {
            FastLinkage::Single => Linkage::Single,
            FastLinkage::Complete => Linkage::Complete,
            FastLinkage::Average => Linkage::Average,
            FastLinkage::Ward => Linkage::Ward,
            FastLinkage::Weighted => Linkage::Mcquitty,
            FastLinkage::Centroid => Linkage::Centroid,
            FastLinkage::Median => Linkage::Median,
        }
    }
}

/// Run hierarchical clustering with the results of `fastcluster::hclust` /
/// `fastcluster::hclust.vector`.
///
/// The tree is built by the shared core [`crate::ml::hclust`] (R's `hclust.f`), so merges,
/// heights and order equal `stats::hclust(dist(data), method)` and `fastcluster::hclust`, which
/// agree. Centroid and median are computed on squared Euclidean distances and their heights
/// reported as square roots, as `fastcluster::hclust.vector` does. Memory is one condensed
/// triangle (O(n²)); time is typically O(n²) (nearest-neighbour list), O(n³) in the worst case.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features); refused if any value is not finite
/// * `linkage` - Linkage method (default: Ward = ward.D2)
/// * `n_clusters` - Optional: cut the tree into this many clusters (R's `cutree`; labels
///   0-based and numbered by first appearance)
///
/// # Returns
/// * `FastClusterResult` containing merge matrix and heights
pub fn fastcluster(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    n_clusters: Option<usize>,
) -> Result<FastClusterResult, String> {
    let n = data.nrows();
    if n < 2 {
        return Err("Need at least 2 observations for hierarchical clustering".to_string());
    }
    require_finite(&data, "fastcluster")?;
    let method = linkage.unwrap_or_default();
    if let Some(k) = n_clusters {
        if k < 1 || k > n {
            return Err(format!("n_clusters must be between 1 and n ({})", n));
        }
    }
    let lk: Linkage = method.into();
    let tree =
        hclust_euclidean(data, lk, "fastcluster distance matrix").map_err(|e| e.to_string())?;
    let height = if lk.uses_squared_euclidean() {
        tree.height().iter().map(|h| h.max(0.0).sqrt()).collect()
    } else {
        tree.height().to_vec()
    };
    let labels = n_clusters
        .map(|k| {
            tree.cut_k(k)
                .map(|v| v.into_iter().map(|l| l - 1).collect())
        })
        .transpose()
        .map_err(|e| e.to_string())?;

    Ok(FastClusterResult {
        merge: tree.merge_array(),
        height,
        order: tree.order().to_vec(),
        labels,
        n,
        linkage: method,
    })
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
    use crate::ml::hclust::Dissimilarity;
    use crate::ml::hclust::test_util::*;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;
    use p2a_validation::{Check, Tol};

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
        // condensed triangle: 39,600 B needed
        let e = with_alloc_ceiling(Some(20_000), || agnes(data.view(), Some(1), None)).unwrap_err();
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
        // R: cluster::diana(six)$dc (before: 0)
        assert!(
            (result.divisive_coefficient - 0.98873252070075102).abs() < 1e-12,
            "{}",
            result.divisive_coefficient
        );

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
        // R: cluster::agnes(six, method = "average")$ac (before: 0.033 at k = 2)
        assert!(
            (result.agglomerative_coefficient - 0.98930988162683642).abs() < 1e-12,
            "{}",
            result.agglomerative_coefficient
        );

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

        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1]);
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

    const FAST: [(FastLinkage, &str); 7] = [
        (FastLinkage::Single, "single"),
        (FastLinkage::Complete, "complete"),
        (FastLinkage::Average, "average"),
        (FastLinkage::Weighted, "mcquitty"),
        (FastLinkage::Ward, "wardD2"),
        (FastLinkage::Centroid, "centroid"),
        (FastLinkage::Median, "median"),
    ];

    #[test]
    fn test_validate_fastcluster_linkages() {
        // ML-13 (average/weighted squared before averaging), ML-12 (NN-chain hang), ML-56
        // (centroid/median through an NN-chain). fastcluster::hclust.vector: centroid/median heights
        // are sqrt of hclust(dist(x)^2, m) heights (checked in R: equal merge and order, heights
        // within 1.1e-15). hclust_methods, validation/r/cases/ml_hierarchical.R.
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let g = matrix(&case_frame(&case, "hclust_ties.csv"), &["u", "v"]);
        let squared = |l: FastLinkage, h: &[f64]| -> Vec<f64> {
            if matches!(l, FastLinkage::Centroid | FastLinkage::Median) {
                h.iter().map(|v| v * v).collect()
            } else {
                h.to_vec()
            }
        };
        let mut c = Check::new(&case);
        for (l, tag) in FAST {
            for (data, pre, n) in [(&x, "", 60usize), (&g, "ties_", 40usize)] {
                let r = fastcluster(data.view(), Some(l), Some(3)).unwrap();
                let merge: Vec<(i64, i64)> = r
                    .merge
                    .rows()
                    .into_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                c.vector(
                    &format!("{pre}{tag}_merge"),
                    &merge_colmajor(&merge),
                    Tol::exact(),
                )
                .vector(
                    &format!("{pre}{tag}_height"),
                    &squared(l, &r.height),
                    Tol::coef(n),
                )
                .vector(
                    &format!("{pre}{tag}_order"),
                    &plus_one(&r.order),
                    Tol::exact(),
                );
            }
            let r = fastcluster(x.view(), Some(l), Some(3)).unwrap();
            c.vector(
                &format!("{tag}_cutree3"),
                &plus_one(&r.labels.unwrap()),
                Tol::exact(),
            );
        }
        c.finish();
    }

    #[test]
    fn test_validate_fastcluster_finding_cases() {
        // ML-13: {0,1,3,7,15}; ML-14: the four cut cases (before: [0,1,0,0,2], [0,0,0,0,0],
        // [0,0,1,1,2], 3 clusters). hclust_methods.
        let (case, _) = crate::test_support::load_case("hclust_methods");
        let col = |v: &[f64]| Array2::from_shape_vec((v.len(), 1), v.to_vec()).unwrap();
        let cut = |v: &[f64], l, k| {
            plus_one(
                &fastcluster(col(v).view(), Some(l), Some(k))
                    .unwrap()
                    .labels
                    .unwrap(),
            )
        };
        let h = |l| {
            fastcluster(col(&[0., 1., 3., 7., 15.]).view(), Some(l), None)
                .unwrap()
                .height
        };
        Check::new(&case)
            .vector(
                "ml13_average_height",
                &h(FastLinkage::Average),
                Tol::coef(5),
            )
            .vector(
                "ml13_mcquitty_height",
                &h(FastLinkage::Weighted),
                Tol::coef(5),
            )
            .vector(
                "ml14_single_cutree4",
                &cut(&[0., 100., 1., 1.1, 300.], FastLinkage::Single, 4),
                Tol::exact(),
            )
            .vector(
                "ml14_complete_cutree2",
                &cut(&[1000., 1., 2., 3., 4.], FastLinkage::Complete, 2),
                Tol::exact(),
            )
            .vector(
                "ml14_ties_cutree4",
                &cut(&[0., 0., 5., 5., 10.], FastLinkage::Complete, 4),
                Tol::exact(),
            )
            .vector(
                "ml14_chain_cutree4",
                &cut(&[0., 1., 10., 10.5, 30.], FastLinkage::Single, 4),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_fastcluster_terminates_and_refuses_nan() {
        // ML-12: 40-60 points in two blobs hung for complete/weighted/ward/median/centroid on
        // most seeds; one NaN cell panicked
        for seed in 0..5u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let x = Array2::from_shape_fn((60, 2), |(i, _)| {
                (if i < 30 { 0.0 } else { 4.0 }) + rng.r#gen::<f64>() * 2.0
            });
            for (l, _) in FAST {
                let r = fastcluster(x.view(), Some(l), Some(2)).unwrap();
                assert_eq!(r.merge.nrows(), 59, "{l:?} seed {seed}");
                if !matches!(l, FastLinkage::Centroid | FastLinkage::Median) {
                    assert!(
                        r.height.windows(2).all(|w| w[0] <= w[1]),
                        "{l:?} seed {seed}"
                    );
                }
            }
        }
        let mut x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]];
        x[[2, 1]] = f64::NAN;
        assert!(
            fastcluster(x.view(), None, None)
                .unwrap_err()
                .contains("finite")
        );
    }

    const AGNES: [(AgnesLinkage, &str); 5] = [
        (AgnesLinkage::Average, "average"),
        (AgnesLinkage::Single, "single"),
        (AgnesLinkage::Complete, "complete"),
        (AgnesLinkage::Ward, "ward"),
        (AgnesLinkage::Weighted, "weighted"),
    ];

    #[test]
    fn test_validate_agnes_diana_coefficients() {
        // ML-51: before, ac = 0 for k = 1 (0.033 for k = 2) and dc = 0 for k = 2; R: ac average
        // 0.8259032898863274, single 0.6885297065427618, complete 0.8984982990829474, ward
        // 0.9590527579487206, weighted 0.844242183861284; dc 0.8885450615416582. The coefficient
        // does not depend on n_clusters. hclust_methods.
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let mut c = Check::new(&case);
        for (l, tag) in AGNES {
            for k in [1usize, 2, 3] {
                c.scalar(
                    &format!("agnes_{tag}_ac"),
                    agnes(x.view(), Some(k), Some(l))
                        .unwrap()
                        .agglomerative_coefficient,
                    Tol::stat(60),
                );
            }
        }
        for k in [1usize, 2, 3] {
            c.scalar(
                "diana_dc",
                diana(x.view(), Some(k)).unwrap().divisive_coefficient,
                Tol::stat(60),
            );
        }
        c.finish();
    }

    #[test]
    fn test_validate_agnes_tree() {
        // R sort(agnes$height) (monotone methods), agnes$order, cutree(as.hclust(agnes), 3)
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let mut c = Check::new(&case);
        for (l, tag) in AGNES {
            let r = agnes(x.view(), Some(3), Some(l)).unwrap();
            assert_eq!(r.merge.len(), 59);
            c.vector(&format!("agnes_{tag}_height"), &r.heights, Tol::coef(60))
                .vector(
                    &format!("agnes_{tag}_order"),
                    &plus_one(&r.order),
                    Tol::exact(),
                )
                .vector(
                    &format!("agnes_{tag}_cutree3"),
                    &plus_one(&r.labels),
                    Tol::exact(),
                );
        }
        c.finish();
    }

    #[test]
    fn test_validate_diana_banner() {
        // cluster::diana(X): $height (banner), $order, $merge, cutree(as.hclust(.), 3). hclust_methods.
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let r = diana(x.view(), Some(3)).unwrap();
        Check::new(&case)
            .vector("diana_banner", &r.banner, Tol::coef(60))
            .vector("diana_order", &plus_one(&r.order), Tol::exact())
            .vector("diana_merge", &merge_colmajor(&r.merge), Tol::exact())
            .vector("diana_cutree3", &plus_one(&r.labels), Tol::exact())
            .finish();
        assert_eq!(r.merge_history.len(), 59);
        assert!(r.merge_history.iter().all(|&(_, a, b, _)| a >= 1 && b >= 1));
        // twins.c: the first split sits at the data diameter cs (R: max(dist(X)) =
        // max(diana(X)$height) = 10.304529754773773)
        let cs = Dissimilarity::euclidean(x.view(), "test")
            .unwrap()
            .as_slice()
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        assert_eq!(r.merge_history[0].3, cs);
    }

    #[test]
    fn test_agnes_linkage_parses_aliases_and_rejects_unknown() {
        AgnesLinkage::choices().validate().unwrap();
        assert_eq!(
            "UPGMA".parse::<AgnesLinkage>().unwrap(),
            AgnesLinkage::Average
        );
        assert_eq!(
            "mcquitty".parse::<AgnesLinkage>().unwrap(),
            AgnesLinkage::Weighted
        );
        assert!(
            run_agnes(array![[0.0], [1.0]].view(), Some(1), Some("ward.D"))
                .unwrap_err()
                .contains("unknown linkage")
        );
    }
}
