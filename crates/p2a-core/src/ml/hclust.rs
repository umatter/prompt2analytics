//! Agglomerative hierarchical clustering core shared by the clustering methods of [`crate::ml`].
//!
//! One implementation of R's `stats::hclust` (Lance-Williams nearest-neighbour-list algorithm,
//! R's `hclust.f` `HCLUST` + `HCASS2`) and `stats::cutree` serves `ml::hierarchical`,
//! `ml::cutree`, AGNES, pvclust, fastcluster and the dynamic tree cut. Results follow R's
//! conventions: the merge matrix (`-i` for observation `i`, `+s` for the cluster made at step `s`,
//! both 1-based), heights in merge order, the leaf order, and `cutree` labels numbered by first
//! appearance.
//!
//! **Algorithm.** For every active cluster `i` the nearest later cluster (`nn[i]`) and its
//! dissimilarity are cached. Each step merges the pair with the smallest cached dissimilarity
//! (ascending scans with strict `<`, so the first minimal pair wins, as in R), updates the
//! dissimilarities of the merged cluster with the Lance-Williams formula, and rescans only the
//! clusters whose cached neighbour was one of the two merged clusters. Memory is one condensed
//! triangle of `n(n-1)/2` doubles; time is typically O(n²), O(n³) in the worst case.
//!
//! **Methods.** The eight methods of `hclust`: single, complete, average (UPGMA), mcquitty
//! (WPGMA), ward.D, ward.D2, centroid (UPGMC) and median (WPGMC). Centroid and median are
//! defined on squared Euclidean distances, as `?hclust` documents: [`hclust_euclidean`] squares
//! `dist(x)` for them, and [`hclust`] expects the caller to pass squared dissimilarities.
//! Their heights can decrease (inversions); [`HclustTree::cut_height`] refuses such trees, as R's
//! `cutree(h =)` does.
//!
//! **"ward" means ward.D2** everywhere in p2a: the strings `"ward"`, `"ward.D2"` and `"ward2"`
//! parse to [`Linkage::Ward`], which is `hclust(method = "ward.D2")`,
//! `cluster::agnes(method = "ward")` and `fastcluster::hclust.vector(method = "ward")`. R's own
//! `hclust(method = "ward")` means ward.D (deprecated since R 3.1); that is [`Linkage::WardD`]
//! (`"ward.D"`, `"ward1"`).
//!
//! # References
//!
//! - Murtagh, F. (1983). "A Survey of Recent Advances in Hierarchical Clustering Algorithms".
//!   The Computer Journal, 26(4), 354-359.
//! - Murtagh, F. (1985). *Multidimensional Clustering Algorithms*. COMPSTAT Lectures 4.
//! - Lance, G. N. & Williams, W. T. (1967). "A General Theory of Classificatory Sorting
//!   Strategies: 1. Hierarchical Systems". The Computer Journal, 9(4), 373-380.
//! - R Core Team, `?hclust` and `src/library/stats/src/hclust.f` (R 4.3.3).

use crate::errors::{EconError, EconResult};
use crate::options::Choices;
use crate::resource::{checked_len, checkpoint, try_vec, try_zeros2};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Agglomeration method of `stats::hclust`.
///
/// | Variant | R `hclust` method | Also |
/// |---|---|---|
/// | `Single` | `"single"` | |
/// | `Complete` | `"complete"` | |
/// | `Average` | `"average"` | UPGMA |
/// | `Ward` | `"ward.D2"` | `"ward"`, `"ward2"`; `agnes(method = "ward")` |
/// | `Mcquitty` | `"mcquitty"` | `"weighted"`, WPGMA |
/// | `WardD` | `"ward.D"` | `"ward1"`; R's deprecated `hclust("ward")` |
/// | `Centroid` | `"centroid"` | UPGMC; on squared distances |
/// | `Median` | `"median"` | WPGMC, `"gower"`; on squared distances |
///
/// Height scales: `ml::hierarchical`, AGNES and pvclust report the heights of
/// `hclust(dist(x)^2, m)` for centroid and median; `fastcluster` reports their square roots, as
/// `fastcluster::hclust.vector` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Linkage {
    /// Minimum distance between clusters (nearest neighbour).
    Single,
    /// Maximum distance between clusters (furthest neighbour).
    Complete,
    /// Average distance between all pairs (UPGMA).
    Average,
    /// Ward's minimum variance method on Euclidean distances: R's `ward.D2`.
    Ward,
    /// McQuitty's weighted average (WPGMA).
    Mcquitty,
    /// R's `ward.D`: the Ward update applied to the dissimilarities as given.
    WardD,
    /// Centroid linkage (UPGMC), on squared Euclidean distances.
    Centroid,
    /// Median linkage (WPGMC, Gower), on squared Euclidean distances.
    Median,
}

crate::option_enum!(
    Linkage,
    Choices {
        option: "linkage",
        values: &[
            (&["single"], Linkage::Single),
            (&["complete"], Linkage::Complete),
            (&["average", "upgma"], Linkage::Average),
            (&["ward.D2", "ward", "ward2"], Linkage::Ward),
            (&["mcquitty", "weighted", "wpgma"], Linkage::Mcquitty),
            (&["ward.D", "ward1"], Linkage::WardD),
            (&["centroid", "upgmc"], Linkage::Centroid),
            (&["median", "wpgmc", "gower"], Linkage::Median),
        ],
    }
);

impl Linkage {
    /// R's method name (the canonical spelling): `"ward.D2"` for [`Linkage::Ward`].
    pub fn r_name(self) -> &'static str {
        Self::choices().canonical(self)
    }

    /// Centroid and median are defined on squared Euclidean distances.
    pub fn uses_squared_euclidean(self) -> bool {
        matches!(self, Linkage::Centroid | Linkage::Median)
    }

    /// Methods whose heights never decrease (all but centroid and median).
    pub fn is_monotone(self) -> bool {
        !self.uses_squared_euclidean()
    }
}

/// `n*i - i*(i+1)/2 + j - i - 1` for `i < j`: R's `dist` storage, 0-based.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    n * i - i * (i + 1) / 2 + j - i - 1
}

fn overflow_error(what: &str) -> EconError {
    EconError::Computation(format!(
        "{what}: no finite distance remains between the clusters still to be merged \
         (the data are too large for f64 arithmetic); rescale the columns"
    ))
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// Condensed dissimilarities in R's `dist` storage: (0,1), (0,2), .., (0,n-1), (1,2), ..
#[derive(Debug, Clone, PartialEq)]
pub struct Dissimilarity {
    n: usize,
    d: Vec<f64>,
}

impl Dissimilarity {
    /// From a condensed vector of length `n(n-1)/2`; `n >= 1` and every value finite.
    pub fn from_condensed(n: usize, d: Vec<f64>) -> EconResult<Self> {
        if n == 0 {
            return Err(invalid("dissimilarities need n >= 1 objects".into()));
        }
        let len = checked_len(&[n, n - 1], "dissimilarities")? / 2;
        if d.len() != len {
            return Err(invalid(format!(
                "dissimilarities for {n} objects need {len} values (got {})",
                d.len()
            )));
        }
        if let Some(p) = d.iter().position(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "dissimilarities must be finite (value {p} is {})",
                d[p]
            )));
        }
        Ok(Dissimilarity { n, d })
    }

    /// From a square matrix: finite, zero diagonal, symmetric to 1e-10 relative. The upper
    /// triangle is kept.
    pub fn from_square(m: ArrayView2<f64>) -> EconResult<Self> {
        let n = m.nrows();
        if n == 0 || m.ncols() != n {
            return Err(invalid(format!(
                "a dissimilarity matrix must be square with at least one row (got {} x {})",
                m.nrows(),
                m.ncols()
            )));
        }
        if let Some(((i, j), v)) = m.indexed_iter().find(|(_, v)| !v.is_finite()) {
            return Err(invalid(format!(
                "dissimilarities must be finite (d[{i},{j}] is {v})"
            )));
        }
        let len = checked_len(&[n, n - 1], "dissimilarities")? / 2;
        let mut d = try_vec(len, 0.0, "dissimilarities")?;
        for i in 0..n {
            checkpoint()?;
            if m[[i, i]] != 0.0 {
                return Err(invalid(format!(
                    "a dissimilarity matrix needs a zero diagonal (d[{i},{i}] is {})",
                    m[[i, i]]
                )));
            }
            for j in (i + 1)..n {
                let (a, b) = (m[[i, j]], m[[j, i]]);
                if (a - b).abs() > 1e-10 * a.abs().max(b.abs()) {
                    return Err(invalid(format!(
                        "a dissimilarity matrix must be symmetric (d[{i},{j}] = {a}, d[{j},{i}] = {b})"
                    )));
                }
                d[condensed_index(n, i, j)] = a;
            }
        }
        Ok(Dissimilarity { n, d })
    }

    /// R's `dist(x)` (Euclidean; columns summed in order). Refuses non-finite data and
    /// distances that overflow f64; buffers are fallible and each row is a checkpoint.
    pub fn euclidean(x: ArrayView2<f64>, what: &str) -> EconResult<Self> {
        let n = x.nrows();
        if n == 0 {
            return Err(invalid(format!("{what}: needs n >= 1 observations")));
        }
        super::extract::require_finite(&x, what).map_err(invalid)?;
        let len = checked_len(&[n, n - 1], what)? / 2;
        let mut d = try_vec(len, 0.0, what)?;
        let p = x.ncols();
        let mut k = 0;
        for i in 0..n {
            checkpoint()?;
            let ri = x.row(i);
            for j in (i + 1)..n {
                let rj = x.row(j);
                let mut s = 0.0;
                for c in 0..p {
                    let dev = ri[c] - rj[c];
                    s += dev * dev;
                }
                let v = s.sqrt();
                if !v.is_finite() {
                    return Err(overflow_error(what));
                }
                d[k] = v;
                k += 1;
            }
        }
        Ok(Dissimilarity { n, d })
    }

    /// The squared dissimilarities (`dist(x)^2`), in place; an overflow is an error.
    pub fn into_squared(mut self) -> EconResult<Self> {
        for v in &mut self.d {
            *v *= *v;
            if !v.is_finite() {
                return Err(overflow_error("squared dissimilarities"));
            }
        }
        Ok(self)
    }

    /// Number of objects.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The dissimilarity of objects `i` and `j` (0 on the diagonal).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match i.cmp(&j) {
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Less => self.d[condensed_index(self.n, i, j)],
            std::cmp::Ordering::Greater => self.d[condensed_index(self.n, j, i)],
        }
    }

    /// The condensed values in R's order.
    pub fn as_slice(&self) -> &[f64] {
        &self.d
    }

    /// The full `n x n` matrix (fallible allocation).
    pub fn to_square(&self, what: &str) -> EconResult<Array2<f64>> {
        let n = self.n;
        let mut m = try_zeros2(n, n, what)?;
        let mut k = 0;
        for i in 0..n {
            checkpoint()?;
            for j in (i + 1)..n {
                m[[i, j]] = self.d[k];
                m[[j, i]] = self.d[k];
                k += 1;
            }
        }
        Ok(m)
    }
}

/// A dendrogram in R's `hclust` convention.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HclustTree {
    n: usize,
    merge: Vec<(i64, i64)>,
    height: Vec<f64>,
    order: Vec<usize>,
}

impl HclustTree {
    fn validate(n: usize, merge: &[(i64, i64)], height: &[f64]) -> EconResult<()> {
        if n < 2 {
            return Err(invalid(format!(
                "a tree needs n >= 2 observations (got {n})"
            )));
        }
        let need = n - 1;
        if merge.len() != need {
            return Err(invalid(format!(
                "a tree of {n} observations needs {need} merges (got {})",
                merge.len()
            )));
        }
        if height.len() != need {
            return Err(invalid(format!(
                "a tree of {n} observations needs {need} heights (got {})",
                height.len()
            )));
        }
        if let Some(p) = height.iter().position(|h| !h.is_finite()) {
            return Err(invalid(format!(
                "tree heights must be finite (height {} is {})",
                p + 1,
                height[p]
            )));
        }
        let mut seen_obs = vec![false; n];
        let mut seen_step = vec![false; need];
        for (s, &(a, b)) in merge.iter().enumerate() {
            for v in [a, b] {
                if v < 0 {
                    let i = v.unsigned_abs() as usize;
                    if i > n || seen_obs[i - 1] {
                        return Err(invalid(format!(
                            "merge {}: observation {v} is out of range or used twice",
                            s + 1
                        )));
                    }
                    seen_obs[i - 1] = true;
                } else if v > 0 {
                    let t = v as usize;
                    if t > s || seen_step[t - 1] {
                        return Err(invalid(format!(
                            "merge {}: step {v} is not an earlier unused merge",
                            s + 1
                        )));
                    }
                    seen_step[t - 1] = true;
                } else {
                    return Err(invalid(format!("merge {}: entry 0 is not allowed", s + 1)));
                }
            }
        }
        Ok(())
    }

    /// Left-first depth-first leaf order from the last merge (equal to `HCASS2`'s order).
    fn traversal_order(n: usize, merge: &[(i64, i64)]) -> Vec<usize> {
        let mut order = Vec::with_capacity(n);
        let mut stack = vec![merge.len() as i64];
        while let Some(v) = stack.pop() {
            if v < 0 {
                order.push(v.unsigned_abs() as usize - 1);
            } else {
                let (a, b) = merge[v as usize - 1];
                stack.push(b);
                stack.push(a);
            }
        }
        order
    }

    /// A validated tree from R's merge matrix and heights (rows are not reordered); the leaf
    /// order is the left-first traversal.
    pub fn from_merge(n: usize, merge: Vec<(i64, i64)>, height: Vec<f64>) -> EconResult<Self> {
        Self::validate(n, &merge, &height)?;
        let order = Self::traversal_order(n, &merge);
        Ok(HclustTree {
            n,
            merge,
            height,
            order,
        })
    }

    /// A validated tree with a given leaf order (a permutation of `0..n`).
    pub fn from_parts(
        n: usize,
        merge: Vec<(i64, i64)>,
        height: Vec<f64>,
        order: Vec<usize>,
    ) -> EconResult<Self> {
        Self::validate(n, &merge, &height)?;
        let mut seen = vec![false; n];
        if order.len() != n
            || order
                .iter()
                .any(|&o| o >= n || std::mem::replace(&mut seen[o], true))
        {
            return Err(invalid(format!(
                "the leaf order must be a permutation of the {n} observations"
            )));
        }
        Ok(HclustTree {
            n,
            merge,
            height,
            order,
        })
    }

    /// From a SciPy-style linkage `(id1, id2, height, size)` with ids `< n` for observations
    /// and `n + s` for the cluster of merge `s` (0-based). Needs all `n - 1` rows.
    pub fn from_scipy_linkage(
        n: usize,
        linkage: &[(usize, usize, f64, usize)],
    ) -> EconResult<Self> {
        let map = |id: usize| -> i64 {
            if id < n {
                -(id as i64 + 1)
            } else {
                (id - n) as i64 + 1
            }
        };
        let merge = linkage
            .iter()
            .map(|&(a, b, _, _)| (map(a), map(b)))
            .collect();
        let height = linkage.iter().map(|r| r.2).collect();
        Self::from_merge(n, merge, height)
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.n
    }

    /// R's merge matrix rows: `-i` observation `i` (1-based), `+s` merge step `s` (1-based).
    pub fn merge(&self) -> &[(i64, i64)] {
        &self.merge
    }

    /// Heights in merge order.
    pub fn height(&self) -> &[f64] {
        &self.height
    }

    /// 0-based leaf order (R's `order` - 1).
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The merge matrix as an `(n-1) x 2` array.
    pub fn merge_array(&self) -> Array2<i64> {
        Array2::from_shape_fn((self.merge.len(), 2), |(i, j)| {
            if j == 0 {
                self.merge[i].0
            } else {
                self.merge[i].1
            }
        })
    }

    /// Sizes of the clusters formed at each merge.
    fn sizes(&self) -> Vec<usize> {
        let mut size = Vec::with_capacity(self.merge.len());
        for &(a, b) in &self.merge {
            let s = |v: i64| if v < 0 { 1 } else { size[v as usize - 1] };
            let t = s(a) + s(b);
            size.push(t);
        }
        size
    }

    /// SciPy-style linkage: ids `< n` observations, `n + s` the cluster of the `(s+1)`-th merge;
    /// the last field is the size of the merged cluster.
    pub fn scipy_linkage(&self) -> Vec<(usize, usize, f64, usize)> {
        let n = self.n;
        let id = |v: i64| -> usize {
            if v < 0 {
                v.unsigned_abs() as usize - 1
            } else {
                n + v as usize - 1
            }
        };
        let sizes = self.sizes();
        self.merge
            .iter()
            .zip(&self.height)
            .zip(sizes)
            .map(|((&(a, b), &h), s)| (id(a), id(b), h, s))
            .collect()
    }

    /// Per merge: `(min, max)` of the smallest observation index of the two clusters
    /// (R's `IA`/`IB` - 1).
    pub fn slots(&self) -> Vec<(usize, usize)> {
        let mut rep: Vec<usize> = Vec::with_capacity(self.merge.len());
        let mut out = Vec::with_capacity(self.merge.len());
        for &(a, b) in &self.merge {
            let r = |v: i64| {
                if v < 0 {
                    v.unsigned_abs() as usize - 1
                } else {
                    rep[v as usize - 1]
                }
            };
            let (ra, rb) = (r(a), r(b));
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            rep.push(lo);
            out.push((lo, hi));
        }
        out
    }

    /// Heights never decrease (R's `!is.unsorted(height)`).
    pub fn is_monotone(&self) -> bool {
        self.height.windows(2).all(|w| w[0] <= w[1])
    }

    /// R's `cutree(tree, k)`: 1-based labels numbered by first appearance in observation order.
    pub fn cut_k(&self, k: usize) -> EconResult<Vec<usize>> {
        let n = self.n;
        if k < 1 || k > n {
            return Err(invalid(format!("k must be between 1 and {n} (got {k})")));
        }
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        let mut rep_step: Vec<usize> = Vec::with_capacity(n - k);
        for &(a, b) in self.merge.iter().take(n - k) {
            let obs = |v: i64, rep_step: &[usize]| {
                if v < 0 {
                    v.unsigned_abs() as usize - 1
                } else {
                    rep_step[v as usize - 1]
                }
            };
            let ra = find(&mut parent, obs(a, &rep_step));
            let rb = find(&mut parent, obs(b, &rep_step));
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            parent[hi] = lo;
            rep_step.push(lo);
        }
        let mut label_of_root = vec![0usize; n];
        let mut next = 0;
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let r = find(&mut parent, i);
            if label_of_root[r] == 0 {
                next += 1;
                label_of_root[r] = next;
            }
            labels.push(label_of_root[r]);
        }
        Ok(labels)
    }

    /// `cut_k` for several `k`.
    pub fn cut_ks(&self, ks: &[usize]) -> EconResult<Vec<Vec<usize>>> {
        ks.iter().map(|&k| self.cut_k(k)).collect()
    }

    /// The number of groups R's `cutree(tree, h = h)` forms: `n + 1 - i`, with `i` the first
    /// 1-based index of `c(height, Inf)` above `h`. Refused for a tree whose heights decrease.
    pub fn k_for_height(&self, h: f64) -> EconResult<usize> {
        if !h.is_finite() {
            return Err(invalid(format!("the cut height must be finite (got {h})")));
        }
        if !self.is_monotone() {
            return Err(invalid(
                "the 'height' component of the tree is not sorted (increasingly) \
                 (R cutree refuses h for it); cut by k instead"
                    .into(),
            ));
        }
        let i = self
            .height
            .iter()
            .position(|&x| x > h)
            .unwrap_or(self.height.len())
            + 1;
        Ok(self.n + 1 - i)
    }

    /// R's `cutree(tree, h = h)`.
    pub fn cut_height(&self, h: f64) -> EconResult<Vec<usize>> {
        self.cut_k(self.k_for_height(h)?)
    }

    /// The height of the merge in which each observation first joins a cluster.
    pub fn first_merge_heights(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        for (&(a, b), &h) in self.merge.iter().zip(&self.height) {
            for v in [a, b] {
                if v < 0 {
                    out[v.unsigned_abs() as usize - 1] = h;
                }
            }
        }
        out
    }

    /// The agglomerative coefficient `mean(1 - m_i / max height)`, with `m_i` the first-merge
    /// height of observation `i` (cluster's `bncoef`, `agnes()$ac`). A zero maximum gives NaN,
    /// as in R.
    pub fn agglomerative_coefficient(&self) -> f64 {
        let max = self
            .height
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let m = self.first_merge_heights();
        m.iter().map(|&mi| 1.0 - mi / max).sum::<f64>() / self.n as f64
    }
}

/// `hclust(d, linkage)`: copies `d` (fallible) and builds the tree. For centroid and median the
/// dissimilarities are used as given; pass squared Euclidean distances, as R expects.
pub fn hclust(d: &Dissimilarity, linkage: Linkage) -> EconResult<HclustTree> {
    let mut diss = try_vec(d.d.len(), 0.0, "hclust dissimilarities")?;
    diss.copy_from_slice(&d.d);
    hclust_owned(d.n, diss, linkage)
}

/// `hclust(dist(x), linkage)`, with `dist(x)^2` for centroid and median (as `?hclust`
/// documents); the dissimilarities are built once and consumed without a copy.
pub fn hclust_euclidean(
    x: ArrayView2<f64>,
    linkage: Linkage,
    what: &str,
) -> EconResult<HclustTree> {
    let mut d = Dissimilarity::euclidean(x, what)?;
    if linkage.uses_squared_euclidean() {
        d = d.into_squared().map_err(|_| overflow_error(what))?;
    }
    hclust_owned(d.n, d.d, linkage)
}

/// The nearest-neighbour-list algorithm of `hclust.f` (`HCLUST`) and the merge/order
/// conventions of `HCASS2`.
fn hclust_owned(n: usize, mut diss: Vec<f64>, linkage: Linkage) -> EconResult<HclustTree> {
    const WHAT: &str = "hclust";
    if n < 2 {
        return Err(invalid(format!(
            "hclust: must have n >= 2 objects to cluster (got {n})"
        )));
    }
    let ward2 = linkage == Linkage::Ward;
    if ward2 {
        for v in &mut diss {
            *v *= *v;
            if !v.is_finite() {
                return Err(overflow_error(WHAT));
            }
        }
    }
    const INF: f64 = f64::INFINITY;
    let idx = |i: usize, j: usize| condensed_index(n, i, j);
    let mut membr = try_vec(n, 1.0f64, WHAT)?;
    let mut flag = try_vec(n, true, WHAT)?;
    let mut nn = try_vec(n, 0usize, WHAT)?;
    let mut disnn = try_vec(n, INF, WHAT)?;
    let mut ia = Vec::with_capacity(n - 1);
    let mut ib = Vec::with_capacity(n - 1);
    let mut crit = Vec::with_capacity(n - 1);

    for i in 0..n - 1 {
        let mut dmin = INF;
        let mut jm = i + 1;
        let base = idx(i, i + 1);
        for (off, &v) in diss[base..base + (n - i - 1)].iter().enumerate() {
            if v < dmin {
                dmin = v;
                jm = i + 1 + off;
            }
        }
        nn[i] = jm;
        disnn[i] = dmin;
    }

    let mut jj = usize::MAX;
    for _ in 0..n - 1 {
        checkpoint()?;
        let mut dmin = INF;
        let (mut im, mut jm) = (usize::MAX, usize::MAX);
        for i in 0..n - 1 {
            if flag[i] && disnn[i] < dmin {
                dmin = disnn[i];
                im = i;
                jm = nn[i];
            }
        }
        if dmin.is_nan() || dmin >= INF {
            return Err(overflow_error(WHAT));
        }
        let (i2, j2) = (im.min(jm), im.max(jm));
        ia.push(i2);
        ib.push(j2);
        crit.push(if ward2 { dmin.sqrt() } else { dmin });
        flag[j2] = false;

        dmin = INF;
        let d12 = diss[idx(i2, j2)];
        let (m1, m2) = (membr[i2], membr[j2]);
        for k in 0..n {
            if !flag[k] || k == i2 {
                continue;
            }
            let ind1 = if i2 < k { idx(i2, k) } else { idx(k, i2) };
            let ind2 = if j2 < k { idx(j2, k) } else { idx(k, j2) };
            let (d1, d2) = (diss[ind1], diss[ind2]);
            let mk = membr[k];
            let v = match linkage {
                Linkage::Ward | Linkage::WardD => {
                    ((m1 + mk) * d1 + (m2 + mk) * d2 - mk * d12) / (m1 + m2 + mk)
                }
                Linkage::Single => d1.min(d2),
                Linkage::Complete => d1.max(d2),
                Linkage::Average => (m1 * d1 + m2 * d2) / (m1 + m2),
                Linkage::Mcquitty => (d1 + d2) / 2.0,
                Linkage::Median => ((d1 + d2) - d12 / 2.0) / 2.0,
                Linkage::Centroid => (m1 * d1 + m2 * d2 - m1 * m2 * d12 / (m1 + m2)) / (m1 + m2),
            };
            if !v.is_finite() {
                return Err(overflow_error(WHAT));
            }
            diss[ind1] = v;
            if i2 < k {
                if v < dmin {
                    dmin = v;
                    jj = k;
                }
            } else if v < disnn[k] {
                // J. B.'s fix for non-monotone methods: a cluster before i2 may now be nearest to it
                disnn[k] = v;
                nn[k] = i2;
            }
        }
        membr[i2] = m1 + m2;
        disnn[i2] = dmin;
        nn[i2] = jj;

        for i in 0..n - 1 {
            if flag[i] && (nn[i] == i2 || nn[i] == j2) {
                let mut dm = INF;
                for j in (i + 1)..n {
                    if flag[j] {
                        let v = diss[idx(i, j)];
                        if v < dm {
                            dm = v;
                            jj = j;
                        }
                    }
                }
                nn[i] = jj;
                disnn[i] = dm;
            }
        }
    }

    // HCASS2: R's merge matrix
    let m = n - 1;
    let mut iia: Vec<i64> = ia.iter().map(|&v| v as i64 + 1).collect();
    let mut iib: Vec<i64> = ib.iter().map(|&v| v as i64 + 1).collect();
    // The first merge, after step i, whose lower slot or upper slot is the slot step i kept.
    // Each slot (= the smallest member) is reused by later merges; resolve forward in O(n).
    let mut last_step_of_slot: Vec<Option<usize>> = vec![None; n];
    for s in 0..m {
        let a = ia[s];
        let b = ib[s];
        if let Some(t) = last_step_of_slot[a] {
            iia[s] = -(t as i64 + 1);
        }
        if let Some(t) = last_step_of_slot[b] {
            iib[s] = -(t as i64 + 1);
        }
        last_step_of_slot[a.min(b)] = Some(s);
    }
    let mut merge = Vec::with_capacity(m);
    for s in 0..m {
        let (mut a, mut b) = (-iia[s], -iib[s]);
        if a > 0 && b < 0 {
            std::mem::swap(&mut a, &mut b);
        }
        if a > 0 && b > 0 && a > b {
            std::mem::swap(&mut a, &mut b);
        }
        merge.push((a, b));
    }
    let order = HclustTree::traversal_order(n, &merge);
    Ok(HclustTree {
        n,
        merge,
        height: crit,
        order,
    })
}

/// `cluster::agnes`'s banner `order` - 1, from the tree's slots: each merge moves the block of
/// the cluster with the larger slot right after the block of the cluster with the smaller one.
pub(crate) fn agnes_banner_order(tree: &HclustTree) -> Vec<usize> {
    let n = tree.n();
    let mut ner: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    for (a, b) in tree.slots() {
        let pb = ner.iter().position(|&v| v == b).unwrap_or(0);
        let block: Vec<usize> = ner.drain(pb..pb + size[b]).collect();
        let pa = ner.iter().position(|&v| v == a).unwrap_or(0);
        let at = pa + size[a];
        ner.splice(at..at, block);
        size[a] += size[b];
    }
    ner
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::data::columns::{NaPolicy, numeric_column};
    use ndarray::Array2;
    use polars::prelude::DataFrame;

    /// Columns of a data frame as an n x p matrix (no missing values in case data).
    pub(crate) fn matrix(df: &DataFrame, cols: &[&str]) -> Array2<f64> {
        let c: Vec<Vec<f64>> = cols
            .iter()
            .map(|n| numeric_column(df, n, NaPolicy::Fail).unwrap().0)
            .collect();
        Array2::from_shape_fn((c[0].len(), c.len()), |(i, j)| c[j][i])
    }

    /// A secondary dataset of a case, found by file-name suffix.
    pub(crate) fn case_frame(case: &p2a_validation::Case, suffix: &str) -> DataFrame {
        let rel = case
            .datasets
            .iter()
            .find(|p| p.ends_with(suffix))
            .expect("dataset in case");
        crate::data::loader::DataLoader::load_csv(p2a_validation::repo_path(rel)).unwrap()
    }

    /// R's `as.numeric(h$merge)`: column-major.
    pub(crate) fn merge_colmajor(m: &[(i64, i64)]) -> Vec<f64> {
        m.iter()
            .map(|p| p.0 as f64)
            .chain(m.iter().map(|p| p.1 as f64))
            .collect()
    }

    pub(crate) fn as_f64(v: &[usize]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }

    pub(crate) fn plus_one(v: &[usize]) -> Vec<f64> {
        v.iter().map(|&x| (x + 1) as f64).collect()
    }

    pub(crate) const METHODS: [(super::Linkage, &str); 8] = [
        (super::Linkage::Single, "single"),
        (super::Linkage::Complete, "complete"),
        (super::Linkage::Average, "average"),
        (super::Linkage::Mcquitty, "mcquitty"),
        (super::Linkage::WardD, "wardD"),
        (super::Linkage::Ward, "wardD2"),
        (super::Linkage::Centroid, "centroid"),
        (super::Linkage::Median, "median"),
    ];
}

#[cfg(test)]
mod tests {
    use super::test_util::*;
    use super::*;
    use p2a_validation::{Check, Tol};

    /// hclust_methods (validation/r/cases/ml_hierarchical.R): merges, heights, order and
    /// cutree(k = 3) on hclust_60, and merges, heights and order on the tied grid hclust_ties.
    fn check_core(case_id: &str, linkage: Linkage, tag: &str) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let t = hclust_euclidean(x.view(), linkage, "test").unwrap();
        let g = matrix(&case_frame(&case, "hclust_ties.csv"), &["u", "v"]);
        let tt = hclust_euclidean(g.view(), linkage, "test").unwrap();
        Check::new(&case)
            .vector(
                &format!("{tag}_merge"),
                &merge_colmajor(t.merge()),
                Tol::exact(),
            )
            .vector(&format!("{tag}_height"), t.height(), Tol::coef(60))
            .vector(&format!("{tag}_order"), &plus_one(t.order()), Tol::exact())
            .vector(
                &format!("{tag}_cutree3"),
                &as_f64(&t.cut_k(3).unwrap()),
                Tol::exact(),
            )
            // R's tie-breaking on exactly tied distances
            .vector(
                &format!("ties_{tag}_merge"),
                &merge_colmajor(tt.merge()),
                Tol::exact(),
            )
            .vector(&format!("ties_{tag}_height"), tt.height(), Tol::coef(40))
            .vector(
                &format!("ties_{tag}_order"),
                &plus_one(tt.order()),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_hclust_single() {
        check_core("hclust_methods", Linkage::Single, "single")
    }
    #[test]
    fn test_validate_hclust_complete() {
        check_core("hclust_methods", Linkage::Complete, "complete")
    }
    #[test]
    fn test_validate_hclust_average() {
        check_core("hclust_methods", Linkage::Average, "average")
    }
    #[test]
    fn test_validate_hclust_mcquitty() {
        check_core("hclust_methods", Linkage::Mcquitty, "mcquitty")
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_validate_hclust_wardD() {
        check_core("hclust_methods", Linkage::WardD, "wardD")
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_validate_hclust_wardD2() {
        check_core("hclust_methods", Linkage::Ward, "wardD2")
    }
    /// Non-monotone: the last two ties heights are 8.483471074380164, 8.317035147392287.
    #[test]
    fn test_validate_hclust_centroid() {
        check_core("hclust_methods", Linkage::Centroid, "centroid")
    }
    #[test]
    fn test_validate_hclust_median() {
        check_core("hclust_methods", Linkage::Median, "median")
    }

    #[test]
    fn test_validate_hclust_cuts() {
        // cutree(hclust(dist(X), "average"), k = c(2, 4, 6)) column-major; cutree(.., h = 1.5) has k = 25
        let (case, ds) = crate::test_support::load_case("hclust_methods");
        let t = hclust_euclidean(
            matrix(ds.df(), &["a", "b", "c"]).view(),
            Linkage::Average,
            "test",
        )
        .unwrap();
        let ks: Vec<f64> = t
            .cut_ks(&[2, 4, 6])
            .unwrap()
            .concat()
            .iter()
            .map(|&l| l as f64)
            .collect();
        Check::new(&case)
            .vector("average_cutree_k246", &ks, Tol::exact())
            .vector(
                "average_cutree_h15",
                &as_f64(&t.cut_height(1.5).unwrap()),
                Tol::exact(),
            )
            .scalar(
                "average_k_h15",
                t.k_for_height(1.5).unwrap() as f64,
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_hclust_hand_cases() {
        // ML-13 (R heights 1, 2.5, 5.666666666666667, 12.25 and 1, 2.5, 5.25, 10.625), ML-14 (cutree), ML-25
        let (case, _) = crate::test_support::load_case("hclust_methods");
        let col = |v: &[f64]| Array2::from_shape_vec((v.len(), 1), v.to_vec()).unwrap();
        let t = |v: &[f64], l| hclust_euclidean(col(v).view(), l, "test").unwrap();
        let cut = |v: &[f64], l, k| as_f64(&t(v, l).cut_k(k).unwrap());
        let three = ndarray::array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]];
        Check::new(&case)
            .vector(
                "ml13_average_height",
                t(&[0., 1., 3., 7., 15.], Linkage::Average).height(),
                Tol::coef(5),
            )
            .vector(
                "ml13_mcquitty_height",
                t(&[0., 1., 3., 7., 15.], Linkage::Mcquitty).height(),
                Tol::coef(5),
            )
            .vector(
                "ml14_single_cutree4",
                &cut(&[0., 100., 1., 1.1, 300.], Linkage::Single, 4),
                Tol::exact(),
            )
            .vector(
                "ml14_complete_cutree2",
                &cut(&[1000., 1., 2., 3., 4.], Linkage::Complete, 2),
                Tol::exact(),
            )
            .vector(
                "ml14_ties_cutree4",
                &cut(&[0., 0., 5., 5., 10.], Linkage::Complete, 4),
                Tol::exact(),
            )
            .vector(
                "ml14_chain_cutree4",
                &cut(&[0., 1., 10., 10.5, 30.], Linkage::Single, 4),
                Tol::exact(),
            )
            // 1, 10.969655114602888
            .vector(
                "ml25_wardD2_height",
                hclust_euclidean(three.view(), Linkage::Ward, "test")
                    .unwrap()
                    .height(),
                Tol::coef(3),
            )
            .finish();
    }

    #[test]
    fn test_linkage_parses_aliases_and_rejects_unknown() {
        Linkage::choices().validate().unwrap();
        for (s, l) in [
            ("ward", Linkage::Ward),
            ("Ward.D2", Linkage::Ward),
            ("ward_d", Linkage::WardD),
            ("WPGMA", Linkage::Mcquitty),
            ("weighted", Linkage::Mcquitty),
            ("UPGMC", Linkage::Centroid),
            ("gower", Linkage::Median),
            ("Average", Linkage::Average),
        ] {
            assert_eq!(s.parse::<Linkage>().unwrap(), l, "{s}");
        }
        assert_eq!(Linkage::Ward.to_string(), "ward.D2");
        assert_eq!(Linkage::WardD.r_name(), "ward.D");
        let e = "compelte".parse::<Linkage>().unwrap_err().to_string();
        assert!(
            e.contains("unknown linkage 'compelte'")
                && e.contains(
                    "single, complete, average, ward.D2, mcquitty, ward.D, centroid, median"
                ),
            "{e}"
        );
    }

    #[test]
    fn test_validate_hclust_tree_conversions_round_trip() {
        let (_, ds) = crate::test_support::load_case("hclust_methods");
        let t = hclust_euclidean(
            matrix(ds.df(), &["a", "b", "c"]).view(),
            Linkage::Average,
            "test",
        )
        .unwrap();
        let back = HclustTree::from_scipy_linkage(60, &t.scipy_linkage()).unwrap();
        assert_eq!(back.merge(), t.merge());
        assert_eq!(back.order(), t.order()); // left-first traversal == HCASS2's order
        let rebuilt = HclustTree::from_merge(60, t.merge().to_vec(), t.height().to_vec()).unwrap();
        assert_eq!(rebuilt.order(), t.order());
        assert_eq!(t.scipy_linkage().last().unwrap().3, 60);
        let d = Dissimilarity::euclidean(matrix(ds.df(), &["a", "b", "c"]).view(), "test").unwrap();
        let sq = d.to_square("test").unwrap();
        assert_eq!(Dissimilarity::from_square(sq.view()).unwrap(), d);
        assert_eq!(hclust(&d, Linkage::Average).unwrap(), t);
    }

    #[test]
    fn test_hclust_rejects_degenerate_input() {
        // R: hclust() "must have n >= 2 objects to cluster"
        let e = hclust(
            &Dissimilarity::from_condensed(1, vec![]).unwrap(),
            Linkage::Average,
        )
        .unwrap_err();
        assert!(e.to_string().contains("n >= 2"), "{e}");
        assert!(
            Dissimilarity::from_condensed(3, vec![1.0, f64::NAN, 2.0])
                .unwrap_err()
                .to_string()
                .contains("finite")
        );
        assert!(Dissimilarity::from_condensed(3, vec![1.0, 2.0]).is_err()); // wrong length
        let big = ndarray::array![[0.0, 0.0], [1e200, 0.0], [2e200, 0.0]];
        let e = hclust_euclidean(big.view(), Linkage::Single, "test")
            .unwrap_err()
            .to_string();
        assert!(e.contains("no finite distance remains"), "{e}");
        let t = hclust_euclidean(
            ndarray::array![[0.0], [1.0], [3.0]].view(),
            Linkage::Average,
            "test",
        )
        .unwrap();
        for k in [0, 4] {
            assert!(
                t.cut_k(k)
                    .unwrap_err()
                    .to_string()
                    .contains("k must be between 1 and 3"),
                "k={k}"
            );
        }
        assert!(t.cut_height(f64::NAN).is_err());
        let e = HclustTree::from_scipy_linkage(4, &[(0, 1, 1.0, 2), (2, 3, 2.0, 2)])
            .unwrap_err()
            .to_string();
        assert!(e.contains("3 merges"), "{e}"); // ML-82: a truncated tree is refused
        // centroid inversions: R cutree(h =) refuses an unsorted tree
        let g = ndarray::array![[0.0, 0.0], [2.0, 0.0], [1.0, 1.8]];
        let c = hclust_euclidean(g.view(), Linkage::Centroid, "test").unwrap();
        assert!(!c.is_monotone());
        assert!(
            c.cut_height(1.0)
                .unwrap_err()
                .to_string()
                .contains("not sorted")
        );
        assert_eq!(c.cut_k(2).unwrap().len(), 3);
    }

    #[test]
    fn test_hclust_buffers_are_fallible_and_cancellable() {
        use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
        let x = Array2::from_shape_fn((100, 2), |(i, j)| {
            ((i * 7 + j * 3) % 11) as f64 + 0.01 * i as f64
        });
        let e = with_alloc_ceiling(Some(4096), || {
            hclust_euclidean(x.view(), Linkage::Average, "hclust dissimilarities")
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
        let t = CancelToken::new();
        t.cancel();
        let e = with_budget(Budget::new(None, t), || {
            hclust_euclidean(x.view(), Linkage::Ward, "x")
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }
}
