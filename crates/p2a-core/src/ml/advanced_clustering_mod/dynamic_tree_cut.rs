//! Dynamic tree cut of a dendrogram as `dynamicTreeCut::cutreeDynamic` ([`cutree_dynamic()`],
//! [`dynamic_tree_cut()`], [`run_dynamic_tree_cut()`]).
//!
//! A port of dynamicTreeCut 1.63.1 (pure R), default paths:
//!
//! - `method = "tree"`: `cutreeDynamicTree` with `.cutTreeStatic`, `.locateCluster`,
//!   `.processIndividualCluster`, `.runlengthSign`, `.shiftSequence` and `.assignModuleNumber`
//!   (`deepSplit` FALSE for `deep_split = 0`, TRUE otherwise);
//! - `method = "hybrid"`: `cutreeHybrid` with `.CoreSize` and `.interpolate`, deepSplit 0-4, the
//!   core-scatter and gap criteria, and the PAM stage (`pamStage`, `maxPamDist`).
//!
//! Fixed at dynamicTreeCut's defaults: `pamRespectsDendro = TRUE`, `useMedoids = FALSE`,
//! `respectSmallClusters = TRUE`, no external split functions; `maxCoreScatter`, `minGap` and
//! `minSplitHeight` follow from `deep_split`. As in `cutreeDynamic`, the default method is
//! hybrid with `minClusterSize = 20`, and hybrid without a dissimilarity matrix runs the tree
//! method with a warning.
//!
//! Two differences from R, both where R returns NA or stops:
//! - the tree method returns all zeros (unassigned) when no module of `min_cluster_size` objects
//!   remains; R returns NA labels;
//! - in the PAM stage, when the nearest "cluster" of a small cluster or an object is the
//!   unassigned group (colour 0, reachable when a basic branch of the same composite branch is not
//!   a cluster), the objects stay unassigned; R stops with "argument is of length zero".
//!
//! Means and column sums are taken in f64 (R accumulates them in long double); the thresholds
//! they are compared with are rarely close enough for this to change a label.
//!
//! # References
//!
//! - Langfelder, P., Zhang, B., and Horvath, S. (2008). "Defining clusters from a hierarchical
//!   cluster tree: the Dynamic Tree Cut package for R." Bioinformatics, 24(5), 719-720.
//! - dynamicTreeCut 1.63.1, `cutreeDynamic`:
//!   <https://www.rdocumentation.org/packages/dynamicTreeCut/versions/1.63-1/topics/cutreeDynamic>

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use super::hierarchical::{FastLinkage, fastcluster};
use crate::ml::extract::require_finite;
use crate::ml::hclust::{Dissimilarity, HclustTree};
use crate::options::Choices;
use crate::resource::checkpoint;
use crate::traits::convergence::{ResultNotes, notes_text};

/// Method of the dynamic tree cut (`cutreeDynamic(method =)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DynamicCutMethod {
    /// `cutreeHybrid`: branch criteria on the dendrogram plus a PAM-like stage on the
    /// dissimilarities (needs them; without, the tree method runs with a warning)
    #[default]
    Hybrid,
    /// `cutreeDynamicTree`: the dendrogram's shape only
    Tree,
}

crate::option_enum!(
    DynamicCutMethod,
    Choices {
        option: "method",
        values: &[
            (&["hybrid"], DynamicCutMethod::Hybrid),
            (&["tree"], DynamicCutMethod::Tree),
        ],
    }
);

/// Options of [`cutree_dynamic()`] (`cutreeDynamic`'s arguments).
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicTreeCutConfig {
    /// Method (default hybrid)
    pub method: DynamicCutMethod,
    /// `deepSplit`. None: 1 for hybrid, 0 (FALSE) for tree, as `cutreeDynamic`; hybrid takes
    /// 0..=4; for tree 0 is FALSE and any value >= 1 is TRUE
    pub deep_split: Option<u8>,
    /// `minClusterSize` (default 20); at least 2
    pub min_cluster_size: usize,
    /// `cutHeight`. None: 99% of the largest height (tree) or of the height range above the 5%
    /// merge quantile (hybrid); a value above the largest height is clipped as R does
    pub cut_height: Option<f64>,
    /// `pamStage` (hybrid; default true)
    pub pam_stage: bool,
    /// `maxPamDist` (hybrid). None: `cut_height` as given, else the computed cut height
    pub max_pam_dist: Option<f64>,
}

impl Default for DynamicTreeCutConfig {
    fn default() -> Self {
        Self {
            method: DynamicCutMethod::Hybrid,
            deep_split: None,
            min_cluster_size: 20,
            cut_height: None,
            pam_stage: true,
            max_pam_dist: None,
        }
    }
}

/// Dissimilarities for the hybrid method, read without a copy: a square matrix (R's `distM`;
/// `get(i, j)` is row `i`, column `j`, the diagonal is read as 0) or a condensed triangle.
#[derive(Debug, Clone, Copy)]
pub enum DynamicCutDistances<'a> {
    /// n x n matrix
    Square(&'a Array2<f64>),
    /// R `dist` storage, n(n-1)/2 values
    Condensed(&'a Dissimilarity),
}

impl DynamicCutDistances<'_> {
    fn n(&self) -> usize {
        match self {
            Self::Square(m) => m.nrows(),
            Self::Condensed(d) => d.n(),
        }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Square(m) => {
                if i == j {
                    0.0
                } else {
                    m[[i, j]]
                }
            }
            Self::Condensed(d) => d.get(i, j),
        }
    }
}

/// Result of dynamic tree cut.
///
/// # References
///
/// - Langfelder, P., Zhang, B., & Horvath, S. (2008). "Defining clusters
///   from a hierarchical cluster tree: the Dynamic Tree Cut package for R."
///   Bioinformatics, 24(5), 719-720.
/// - R dynamicTreeCut package documentation
///   Source: <https://www.rdocumentation.org/packages/dynamicTreeCut/versions/1.63-1/topics/cutreeDynamic>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTreeCutResult {
    /// Cluster assignments (0 = unassigned; clusters 1.. by decreasing size), as `cutreeDynamic`
    pub labels: Vec<usize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of unassigned points (label 0)
    pub n_unassigned: usize,
    /// Cluster sizes (excluding unassigned), for labels 1..=n_clusters
    pub cluster_sizes: Vec<usize>,
    /// Number of observations
    pub n: usize,
    /// Method run (tree when hybrid had no dissimilarities)
    pub method: DynamicCutMethod,
    /// Deep split used (tree: 0 = FALSE, 1 = TRUE; hybrid: 0..=4)
    pub deep_split: u8,
    /// Minimum cluster size used
    pub min_cluster_size: usize,
    /// Cut height used
    #[serde(default)]
    pub cut_height: f64,
    /// Warnings (the fallback from hybrid to tree)
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for DynamicTreeCutResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for DynamicTreeCutResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dynamic Tree Cut Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Unassigned points: {}", self.n_unassigned)?;
        writeln!(f, "Deep split: {}", self.deep_split)?;
        writeln!(f, "Min cluster size: {}", self.min_cluster_size)?;
        writeln!(f, "Cut height: {:.6}", self.cut_height)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i + 1, size)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Dynamic tree cut of a dendrogram: `dynamicTreeCut::cutreeDynamic(tree, ...)`.
///
/// `dist` is `distM` for the hybrid method (n x n, finite); see the module documentation for
/// the port and its two differences from R.
pub fn cutree_dynamic(
    tree: &HclustTree,
    dist: Option<DynamicCutDistances<'_>>,
    config: &DynamicTreeCutConfig,
) -> Result<DynamicTreeCutResult, String> {
    let n = tree.n();
    let mcs = config.min_cluster_size;
    if mcs < 2 {
        return Err(format!(
            "min_cluster_size must be at least 2 (got {mcs}); dynamicTreeCut fails on one-object clusters"
        ));
    }
    if let Some(h) = config.cut_height {
        if !(h.is_finite() && h > 0.0) {
            return Err(format!("cut_height must be finite and positive (got {h})"));
        }
    }
    if let Some(p) = config.max_pam_dist {
        if !(p.is_finite() && p >= 0.0) {
            return Err(format!(
                "max_pam_dist must be finite and non-negative (got {p})"
            ));
        }
    }
    let requested_ds = config.deep_split.unwrap_or(match config.method {
        DynamicCutMethod::Hybrid => 1,
        DynamicCutMethod::Tree => 0,
    });
    if config.method == DynamicCutMethod::Hybrid && requested_ds > 4 {
        return Err(format!(
            "deep_split must be between 0 and 4 for method 'hybrid' (got {requested_ds})"
        ));
    }
    if let Some(d) = dist {
        if let DynamicCutDistances::Square(m) = d {
            if m.nrows() != n || m.ncols() != n {
                return Err(format!(
                    "the dissimilarity matrix must be {n} x {n} (got {} x {})",
                    m.nrows(),
                    m.ncols()
                ));
            }
            if m.iter().any(|v| !v.is_finite()) {
                return Err("the dissimilarity matrix must contain only finite values".into());
            }
        } else if d.n() != n {
            return Err(format!(
                "the dissimilarities are for {} observations, the tree has {n}",
                d.n()
            ));
        }
    }

    let mut warnings = Vec::new();
    let (labels, method, deep_split, cut_height) = match (config.method, dist) {
        (DynamicCutMethod::Hybrid, Some(d)) => {
            let (labels, cut) = cutree_hybrid(tree, d, config, requested_ds)?;
            (labels, DynamicCutMethod::Hybrid, requested_ds, cut)
        }
        (method, _) => {
            if method == DynamicCutMethod::Hybrid {
                warnings.push(
                    "method 'hybrid' requires a dissimilarity matrix (distM); defaulting to method 'tree'"
                        .to_string(),
                );
            }
            let deep = requested_ds >= 1;
            let (labels, cut) = cutree_dynamic_tree(tree, config.cut_height, deep, mcs)?;
            (labels, DynamicCutMethod::Tree, u8::from(deep), cut)
        }
    };

    let n_clusters = labels.iter().copied().max().unwrap_or(0);
    let mut cluster_sizes = vec![0usize; n_clusters];
    for &l in &labels {
        if l > 0 {
            cluster_sizes[l - 1] += 1;
        }
    }
    let n_unassigned = labels.iter().filter(|&&l| l == 0).count();
    Ok(DynamicTreeCutResult {
        labels,
        n_clusters,
        n_unassigned,
        cluster_sizes,
        n,
        method,
        deep_split,
        min_cluster_size: mcs,
        cut_height,
        warnings,
    })
}

/// Perform dynamic tree cutting on a hierarchical clustering result (`cutreeDynamic`).
///
/// # Arguments
/// * `merge` - Merge matrix in R's convention (n-1 x 2; `-i` observation i, `+s` merge step s)
/// * `height` - Heights at which merges occurred
/// * `n` - Number of observations (at least 2)
/// * `method` - Cutting method (default: Hybrid, as `cutreeDynamic`)
/// * `deep_split` - `deepSplit` (default 1 for hybrid, 0 for tree; hybrid 0-4)
/// * `min_cluster_size` - `minClusterSize` (default 20)
/// * `dist_matrix` - `distM` for the hybrid method (n x n); without it, hybrid runs the tree
///   method with a warning
///
/// # Returns
/// * `DynamicTreeCutResult` (labels 0 = unassigned, 1.. by decreasing size)
pub fn dynamic_tree_cut(
    merge: &Array2<i64>,
    height: &[f64],
    n: usize,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
    dist_matrix: Option<&Array2<f64>>,
) -> Result<DynamicTreeCutResult, String> {
    if n < 2 {
        return Err(format!(
            "dynamic tree cut needs at least 2 observations (got {n})"
        ));
    }
    if merge.ncols() != 2 {
        return Err(format!(
            "the merge matrix must have 2 columns (got {})",
            merge.ncols()
        ));
    }
    let rows: Vec<(i64, i64)> = merge.rows().into_iter().map(|r| (r[0], r[1])).collect();
    let tree = HclustTree::from_merge(n, rows, height.to_vec()).map_err(|e| e.to_string())?;
    let config = config_from(method, deep_split, min_cluster_size);
    cutree_dynamic(&tree, dist_matrix.map(DynamicCutDistances::Square), &config)
}

fn config_from(
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
) -> DynamicTreeCutConfig {
    let mut c = DynamicTreeCutConfig::default();
    if let Some(m) = method {
        c.method = m;
    }
    c.deep_split = deep_split;
    if let Some(s) = min_cluster_size {
        c.min_cluster_size = s;
    }
    c
}

/// Hierarchical clustering ([`fastcluster`], Euclidean) followed by [`cutree_dynamic()`]; the
/// hybrid method receives the Euclidean dissimilarities.
pub fn run_dynamic_tree_cut(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
) -> Result<DynamicTreeCutResult, String> {
    let n = data.nrows();
    if n < 2 {
        return Err(format!(
            "dynamic tree cut needs at least 2 observations (got {n})"
        ));
    }
    require_finite(&data, "dynamic tree cut")?;
    let hc = fastcluster(data, linkage, None)?;
    let merge = hc.merge.rows().into_iter().map(|r| (r[0], r[1])).collect();
    let tree = HclustTree::from_parts(n, merge, hc.height, hc.order).map_err(|e| e.to_string())?;
    let config = config_from(method, deep_split, min_cluster_size);
    let dist = if config.method == DynamicCutMethod::Hybrid {
        Some(
            Dissimilarity::euclidean(data, "dynamic tree cut dissimilarities")
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    cutree_dynamic(
        &tree,
        dist.as_ref().map(DynamicCutDistances::Condensed),
        &config,
    )
}

// =============================================================================
// Shared helpers
// =============================================================================

/// R's `mean`: sum / n plus the mean of the residuals.
fn r_mean(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let (mut s, mut k) = (0.0, 0usize);
    for v in x.clone() {
        s += v;
        k += 1;
    }
    if k == 0 {
        return f64::NAN;
    }
    let m = s / k as f64;
    if !m.is_finite() {
        return m;
    }
    let t: f64 = x.map(|v| v - m).sum();
    m + t / k as f64
}

/// Labels from `labels` (> 0 a group, 0 none) renumbered 1.. by decreasing group size, ties by
/// ascending label (R's stable `-sort(-table())` and `rank(ties.method = "first")`).
fn by_decreasing_size(labels: &[usize]) -> Vec<usize> {
    let max = labels.iter().copied().max().unwrap_or(0);
    let mut sizes = vec![0usize; max + 1];
    for &l in labels {
        sizes[l] += 1;
    }
    let mut present: Vec<usize> = (1..=max).filter(|&l| sizes[l] > 0).collect();
    present.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut rank = vec![0usize; max + 1];
    for (r, &l) in present.iter().enumerate() {
        rank[l] = r + 1;
    }
    labels.iter().map(|&l| rank[l]).collect()
}

fn cancelled(e: crate::errors::EconError) -> String {
    e.to_string()
}

// =============================================================================
// cutreeDynamicTree
// =============================================================================

fn cutree_dynamic_tree(
    tree: &HclustTree,
    cut_height: Option<f64>,
    deep_split: bool,
    min_size: usize,
) -> Result<(Vec<usize>, f64), String> {
    let n = tree.n();
    if !tree.is_monotone() {
        return Err(
            "the dynamic tree cut 'tree' method needs non-decreasing heights (R cutree refuses this tree)"
                .into(),
        );
    }
    let max_h = tree
        .height()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut max_tree_height = cut_height.unwrap_or(0.99 * max_h);
    if max_tree_height > max_h {
        max_tree_height = 0.99 * max_h;
    }

    // .cutTreeStatic
    let static_cut = tree
        .cut_height(max_tree_height)
        .map_err(|e| e.to_string())?;
    let ranked = by_decreasing_size(&static_cut);
    let mut sizes = vec![0usize; n + 1];
    for &r in &ranked {
        sizes[r] += 1;
    }
    let static_label: Vec<i64> = ranked
        .iter()
        .map(|&r| if sizes[r] >= min_size { r as i64 } else { -1 })
        .collect();

    let fmh = tree.first_merge_heights();
    let order = tree.order();
    let hei: Vec<f64> = order.iter().map(|&i| fmh[i]).collect();
    let lab: Vec<i64> = order.iter().map(|&i| static_label[i]).collect();

    let mut static_runs = locate_cluster(&lab);
    if static_runs.is_empty() {
        return Ok((vec![0; n], max_tree_height));
    }
    let cmin = min_size;
    let cattach = 2 * min_size;
    let runs = loop {
        checkpoint().map_err(cancelled)?;
        let mut runs = Vec::new();
        for &(s, e) in &static_runs {
            for (a, b) in process_individual_cluster(&hei[s..=e], cmin, cattach, 0) {
                runs.push((a + s, b + s));
            }
        }
        if !deep_split || runs.len() == static_runs.len() {
            break runs;
        }
        static_runs = runs;
    };

    let mut module = vec![0usize; n];
    let mut cnt = 1;
    for (s, e) in runs {
        if e - s + 1 < min_size {
            continue;
        }
        module[s..=e].fill(cnt);
        cnt += 1;
    }
    // .assignModuleNumber (every module here has at least min_size objects)
    let renumbered = by_decreasing_size(&module);
    let mut labels = vec![0usize; n];
    for (p, &obs) in order.iter().enumerate() {
        labels[obs] = renumbered[p];
    }
    Ok((labels, max_tree_height))
}

/// `.locateCluster`: runs (inclusive positions) of positive static labels in leaf order.
fn locate_cluster(l: &[i64]) -> Vec<(usize, usize)> {
    let n = l.len();
    let mut diff: Vec<i64> = (0..n)
        .map(|p| if p == 0 { 0 } else { (l[p] - l[p - 1]).abs() })
        .collect();
    if l[0] > 0 {
        diff[0] = 1;
    }
    if l[n - 1] > 0 {
        diff[n - 1] = 1;
    }
    let flags: Vec<usize> = (0..n).filter(|&p| diff[p] > 0).collect();
    let mut runs = Vec::new();
    if flags.len() == 1 {
        if l[flags[0]] > 0 {
            runs.push((flags[0], flags[0]));
        }
        return runs;
    }
    for i in 0..flags.len().saturating_sub(1) {
        let idx = flags[i];
        if l[idx] > 0 {
            if flags[i + 1] == n - 1 {
                runs.push((idx, n - 1));
                break;
            }
            runs.push((idx, flags[i + 1] - 1));
        }
    }
    runs
}

/// `.runlengthSign(x, leftOrright = -1, mysign = -1)`: entries counted from the end before the
/// first positive one; 0 if none is positive.
fn runlength_sign(x: &[f64]) -> usize {
    x.iter().rev().position(|&v| v > 0.0).unwrap_or(0)
}

/// `.processIndividualCluster`: runs (inclusive positions within `hei`).
fn process_individual_cluster(
    hei: &[f64],
    cmin: usize,
    cattach: usize,
    use_mean: i8,
) -> Vec<(usize, usize)> {
    let m = hei.len();
    if m < 2 {
        // R stops on a one-object run; it is its own run here
        return vec![(0, m.saturating_sub(1))];
    }
    let min_tail = (cmin as f64 / 3.0) as usize + 1;
    let recurse = |next: i8| process_individual_cluster(hei, cmin, cattach, next);
    let cmax = hei.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let cmin_h = hei.iter().copied().fold(f64::INFINITY, f64::min);
    let cmean = r_mean(hei.iter().copied());
    let comphei = match use_mean {
        1 => (cmean + cmax) / 2.0,
        -1 => (cmean + cmin_h) / 2.0,
        _ => cmean,
    };
    let diff: Vec<f64> = hei.iter().map(|h| h - comphei).collect();
    let mut cuts: Vec<bool> = (0..m)
        .map(|i| {
            let shift = if i + 1 < m { diff[i + 1] } else { diff[m - 1] };
            diff[i] < 0.0 && shift > 0.0
        })
        .collect();
    cuts[0] = true;
    cuts[m - 1] = true;
    if cuts.iter().filter(|&&c| c).count() == 2 {
        return match use_mean {
            0 => recurse(1),
            1 => recurse(-1),
            _ => vec![(0, m - 1)],
        };
    }
    let positions = |c: &[bool]| -> Vec<usize> { (0..m).filter(|&i| c[i]).collect() };
    let cut_index = positions(&cuts);
    let mut cuts2 = cuts.clone();
    for i in 1..cut_index.len() - 1 {
        let run = runlength_sign(&diff[cut_index[i - 1] + 1..=cut_index[i]]);
        if run < min_tail {
            cuts2[cut_index[i]] = false;
        }
    }
    if cuts2.iter().filter(|&&c| c).count() > 3 {
        let mut curj = 2usize; // 1-based, as R
        loop {
            let c = positions(&cuts2);
            let no_clus = c.len() - 1;
            if curj > no_clus {
                break;
            }
            let (pre_s, pre_e) = (c[curj - 2] + 1, c[curj - 1]);
            let (cur_s, cur_e) = (c[curj - 1] + 1, c[curj]);
            let pre_mean = r_mean(hei[pre_s..=pre_e].iter().copied());
            let cur_mean = r_mean(hei[cur_s..=cur_e].iter().copied());
            if cur_mean < pre_mean && cur_e - cur_s + 1 < cattach {
                cuts2[c[curj - 1]] = false;
            } else {
                curj += 1;
            }
        }
    }
    let c = positions(&cuts2);
    let k = c.len();
    if k > 2 && c[k - 1] - c[k - 2] + 1 < cmin {
        cuts2[c[k - 2]] = false;
    }
    let c = positions(&cuts2);
    let runs: Vec<(usize, usize)> = if c.len() > 2 {
        (0..c.len() - 1)
            .map(|j| (if j == 0 { 0 } else { c[j] + 1 }, c[j + 1]))
            .collect()
    } else {
        vec![(0, m - 1)]
    };
    if runs.len() == 1 {
        match use_mean {
            0 => return recurse(1),
            1 => return recurse(-1),
            _ => {}
        }
    }
    runs
}

// =============================================================================
// cutreeHybrid
// =============================================================================

#[derive(Debug, Clone)]
struct Branch {
    basic: bool,
    top_basic: bool,
    fail_size: bool,
    size: usize,
    /// observations (0-based) of a basic branch
    singletons: Vec<usize>,
    /// basic branches (1-based ids) of a composite branch
    basic_clusters: Vec<usize>,
    attach_height: Option<f64>,
}

impl Branch {
    fn basic(singletons: Vec<usize>) -> Self {
        Self {
            basic: true,
            top_basic: true,
            fail_size: false,
            size: singletons.len(),
            singletons,
            basic_clusters: Vec::new(),
            attach_height: None,
        }
    }
}

/// `.CoreSize`
fn core_size(branch_size: usize, min_cluster_size: usize) -> usize {
    let base = min_cluster_size as f64 / 2.0 + 1.0;
    if base < branch_size as f64 {
        (base + (branch_size as f64 - base).sqrt()) as usize
    } else {
        branch_size
    }
}

/// `mean(colSums(distM[Core, Core]) / (coresize - 1))` over the first `.CoreSize` singletons.
fn core_scatter(b: &Branch, mcs: usize, d: DynamicCutDistances<'_>) -> f64 {
    let cs = core_size(b.singletons.len(), mcs);
    let core = &b.singletons[..cs];
    r_mean(core.iter().map(|&j| {
        let s: f64 = core.iter().map(|&i| d.get(i, j)).sum();
        s / (cs - 1) as f64
    }))
}

const INTERNAL: &str = "Internal error: a previous merge has no associated cluster (the tree's heights are not non-decreasing below the cut height)";

fn cutree_hybrid(
    tree: &HclustTree,
    d: DynamicCutDistances<'_>,
    config: &DynamicTreeCutConfig,
    deep_split: u8,
) -> Result<(Vec<usize>, f64), String> {
    let n = tree.n();
    let height = tree.height();
    let merges = tree.merge();
    let n_merge = height.len();
    let mcs = config.min_cluster_size;
    let max_h = height.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let ref_merge = ((n_merge as f64 * 0.05).round_ties_even() as usize).max(1);
    let ref_height = height[ref_merge - 1];
    let cut_height = match config.cut_height {
        Some(c) => c.min(max_h),
        None => 0.99 * (max_h - ref_height) + ref_height,
    };
    let max_pam_dist = config
        .max_pam_dist
        .or(config.cut_height)
        .unwrap_or(cut_height);
    let below = height.iter().filter(|&&h| h <= cut_height).count();
    if below < mcs {
        return Ok((vec![0; n], cut_height));
    }

    const DEF_MCS: [f64; 5] = [0.64, 0.73, 0.82, 0.91, 0.95];
    let max_core_scatter = DEF_MCS[deep_split as usize];
    let min_gap = (1.0 - max_core_scatter) * 3.0 / 4.0;
    let max_abs_core_scatter = ref_height + max_core_scatter * (cut_height - ref_height);
    let min_abs_gap = min_gap * (cut_height - ref_height);
    let min_abs_split_height = ref_height + 0.0 * (cut_height - ref_height);

    // branch ids are 1-based (0 = none), as R's
    let mut br: Vec<Branch> = Vec::new();
    let mut merge_to_branch = vec![0usize; n_merge];
    let mut on_branch = vec![0usize; n];

    for m in 0..n_merge {
        let h = height[m];
        if h > cut_height {
            continue;
        }
        if m % 256 == 0 {
            checkpoint().map_err(cancelled)?;
        }
        let (a, b) = merges[m];
        if a < 0 && b < 0 {
            br.push(Branch::basic(vec![(-a - 1) as usize, (-b - 1) as usize]));
            merge_to_branch[m] = br.len();
        } else if (a < 0) != (b < 0) {
            let step = a.max(b) as usize;
            let clust = merge_to_branch[step - 1];
            if clust == 0 {
                return Err(INTERNAL.into());
            }
            let gene = (-a.min(b) - 1) as usize;
            let c = &mut br[clust - 1];
            if c.basic {
                c.singletons.push(gene);
            } else {
                on_branch[gene] = clust;
            }
            c.size += 1;
            merge_to_branch[m] = clust;
        } else {
            let c1 = merge_to_branch[a as usize - 1];
            let c2 = merge_to_branch[b as usize - 1];
            if c1 == 0 || c2 == 0 {
                return Err(INTERNAL.into());
            }
            // rank(sizes, ties.method = "first")
            let (mut small, mut large) = if br[c1 - 1].size <= br[c2 - 1].size {
                (c1, c2)
            } else {
                (c2, c1)
            };
            let scatter = |id: usize| {
                let b = &br[id - 1];
                if b.basic {
                    core_scatter(b, mcs, d)
                } else {
                    0.0
                }
            };
            let scores = |id: usize, ave: f64| {
                let b = &br[id - 1];
                let s = [
                    b.size < mcs,
                    ave > max_abs_core_scatter,
                    h - ave < min_abs_gap,
                    h < min_abs_split_height,
                ];
                (b.basic && s.iter().any(|&x| x), !(s[1] || s[2]))
            };
            let sm_ave = scatter(small);
            let lg_ave = scatter(large);
            let (sm_merge, sm_fail) = scores(small, sm_ave);
            let mut merge_fail = None;
            if sm_merge {
                merge_fail = Some(sm_fail);
            } else {
                let (lg_merge, lg_fail) = scores(large, lg_ave);
                if lg_merge {
                    merge_fail = Some(lg_fail);
                    std::mem::swap(&mut small, &mut large);
                }
            }
            if let Some(fail) = merge_fail {
                {
                    let s = &mut br[small - 1];
                    s.fail_size = fail;
                    s.attach_height = Some(h);
                    s.top_basic = false;
                }
                let small_singletons = br[small - 1].singletons.clone();
                let small_basic = br[small - 1].basic;
                let small_size = br[small - 1].size;
                let l = &mut br[large - 1];
                if l.basic {
                    l.singletons.extend_from_slice(&small_singletons);
                } else {
                    if !small_basic {
                        return Err("Internal error: merging two composite clusters".into());
                    }
                    for &g in &small_singletons {
                        on_branch[g] = large;
                    }
                }
                l.size += small_size;
                merge_to_branch[m] = large;
            } else {
                if br[large - 1].basic && !br[small - 1].basic {
                    std::mem::swap(&mut small, &mut large);
                }
                let add_of = |id: usize, br: &[Branch]| -> Vec<usize> {
                    let b = &br[id - 1];
                    if b.basic {
                        vec![id]
                    } else {
                        b.basic_clusters.clone()
                    }
                };
                if br[large - 1].basic || config.pam_stage {
                    let mut basic_clusters = add_of(small, &br);
                    basic_clusters.extend(add_of(large, &br));
                    let size = br[small - 1].size + br[large - 1].size;
                    br[small - 1].attach_height = Some(h);
                    br[large - 1].attach_height = Some(h);
                    br.push(Branch {
                        basic: false,
                        top_basic: false,
                        fail_size: false,
                        size,
                        singletons: Vec::new(),
                        basic_clusters,
                        attach_height: None,
                    });
                    merge_to_branch[m] = br.len();
                } else {
                    let add = add_of(small, &br);
                    let small_size = br[small - 1].size;
                    br[small - 1].attach_height = Some(h);
                    let l = &mut br[large - 1];
                    l.basic_clusters.extend(add);
                    l.size += small_size;
                    merge_to_branch[m] = large;
                }
            }
        }
    }

    // clusters
    let n_branches = br.len();
    let mut is_cluster = vec![false; n_branches];
    let mut small_labels = vec![0usize; n];
    for c in 0..n_branches {
        if br[c].attach_height.is_none() {
            br[c].attach_height = Some(cut_height);
        }
        let b = &br[c];
        if b.top_basic {
            let cs = core_scatter(b, mcs, d);
            let attach = b.attach_height.unwrap_or(cut_height);
            is_cluster[c] = b.size >= mcs && cs < max_abs_core_scatter && attach - cs > min_abs_gap;
        }
        if b.fail_size {
            for &g in &b.singletons {
                small_labels[g] = c + 1;
            }
        }
    }
    let mut colors = vec![0i64; n];
    let mut branch_labels = vec![0i64; n_branches + 1];
    let mut color = 0i64;
    for c in 0..n_branches {
        if !is_cluster[c] {
            continue;
        }
        color += 1;
        for &g in &br[c].singletons {
            colors[g] = color;
            small_labels[g] = 0;
        }
        branch_labels[c + 1] = color;
    }
    let n_proper = color as usize;

    if config.pam_stage && colors.contains(&0) && n_proper > 0 {
        // ClusterDiam
        let mut diam = vec![0.0f64; n_proper + 1];
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_proper + 1];
        for (i, &c) in colors.iter().enumerate() {
            if c > 0 {
                members[c as usize].push(i);
            }
        }
        for c in 1..=n_proper {
            let mem = &members[c];
            if mem.len() > 1 {
                checkpoint().map_err(cancelled)?;
                diam[c] = mem
                    .iter()
                    .map(|&j| {
                        mem.iter().map(|&i| d.get(i, j)).sum::<f64>() / (mem.len() - 1) as f64
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }
        let colors_x = colors.clone();
        // objects whose pre-PAM colour is among the labels of branch `on_br`'s basic clusters
        let use_objects = |on_br: usize| -> Vec<usize> {
            let mut labels: Vec<i64> = br[on_br - 1]
                .basic_clusters
                .iter()
                .map(|&bc| branch_labels[bc])
                .collect();
            labels.sort_unstable();
            labels.dedup();
            (0..n)
                .filter(|&p| labels.binary_search(&colors_x[p]).is_ok())
                .collect()
        };
        // tapply(values, factor(ColorsX[useObjects]), mean), then the first minimum
        let nearest = |objects: &[usize], value: &dyn Fn(usize) -> f64| -> (i64, f64) {
            let mut levels: Vec<i64> = objects.iter().map(|&p| colors_x[p]).collect();
            levels.sort_unstable();
            levels.dedup();
            let mut best = (0i64, f64::INFINITY);
            let mut first = true;
            for &lv in &levels {
                let mean = r_mean(
                    objects
                        .iter()
                        .filter(|&&p| colors_x[p] == lv)
                        .map(|&p| value(p)),
                );
                if first || mean < best.1 {
                    best = (lv, mean);
                    first = false;
                }
            }
            best
        };
        let accept = |label: i64, dist: f64| {
            label > 0 && (dist < diam[label as usize] || dist < max_pam_dist)
        };

        // small clusters, by ascending label
        let mut small_levels: Vec<usize> =
            small_labels.iter().copied().filter(|&s| s > 0).collect();
        small_levels.sort_unstable();
        small_levels.dedup();
        for s in small_levels {
            checkpoint().map_err(cancelled)?;
            let in_cluster: Vec<usize> = (0..n).filter(|&p| small_labels[p] == s).collect();
            let on_br = on_branch[in_cluster[0]];
            if in_cluster.iter().any(|&p| on_branch[p] != on_br) {
                return Err(
                    "Internal error: objects in a small cluster are marked to belong to several large branches"
                        .into(),
                );
            }
            if on_br == 0 {
                continue;
            }
            let objects = use_objects(on_br);
            let k = in_cluster.len() as f64;
            let (label, dist) = nearest(&objects, &|p| {
                in_cluster.iter().map(|&i| d.get(i, p)).sum::<f64>() / k
            });
            let new = if accept(label, dist) { label } else { -1 };
            for &p in &in_cluster {
                colors[p] = new;
            }
        }
        // remaining unlabelled objects on a composite branch
        for obj in 0..n {
            if colors[obj] != 0 || on_branch[obj] == 0 {
                continue;
            }
            if obj % 256 == 0 {
                checkpoint().map_err(cancelled)?;
            }
            let objects = use_objects(on_branch[obj]);
            let (label, dist) = nearest(&objects, &|p| d.get(p, obj));
            if accept(label, dist) {
                colors[obj] = label;
            }
        }
    }

    // final labels: unassigned 0, clusters 1.. by decreasing size, ties by colour
    let labels: Vec<usize> = colors.iter().map(|&c| c.max(0) as usize).collect();
    Ok((by_decreasing_size(&labels), cut_height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::hclust::{Linkage, hclust, test_util::*};
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    fn avg_tree(
        case_id: &str,
    ) -> (
        p2a_validation::Case,
        HclustTree,
        Array2<f64>,
        Array2<f64>,
        Dissimilarity,
    ) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let d = Dissimilarity::euclidean(x.view(), "test").unwrap();
        let m = d.to_square("test").unwrap();
        (case, hclust(&d, Linkage::Average).unwrap(), m, x, d)
    }
    fn cfg(method: DynamicCutMethod, ds: u8) -> DynamicTreeCutConfig {
        DynamicTreeCutConfig {
            method,
            deep_split: Some(ds),
            min_cluster_size: 5,
            ..Default::default()
        }
    }
    fn sq(m: &Array2<f64>) -> Option<DynamicCutDistances<'_>> {
        Some(DynamicCutDistances::Square(m))
    }
    fn labels(r: DynamicTreeCutResult) -> Vec<f64> {
        as_f64(&r.labels)
    }

    #[test]
    fn test_validate_cutree_dynamic_tree() {
        // ML-57: before, a percentile-of-gaps heuristic (56 clusters on blob data where R finds 3)
        let (case, t, _, _, _) = avg_tree("dynamic_tree_cut");
        let mut cut = cfg(DynamicCutMethod::Tree, 1);
        cut.cut_height = Some(1.5);
        let run = |c: &DynamicTreeCutConfig| labels(cutree_dynamic(&t, None, c).unwrap());
        Check::new(&case)
            .vector("hclust_average_height", t.height(), Tol::coef(60))
            .vector(
                "tree_deep_false",
                &run(&cfg(DynamicCutMethod::Tree, 0)),
                Tol::exact(),
            )
            .vector(
                "tree_deep_true",
                &run(&cfg(DynamicCutMethod::Tree, 1)),
                Tol::exact(),
            )
            .vector("tree_deep_true_cut15", &run(&cut), Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_cutree_dynamic_hybrid() {
        // ML-57: before, Hybrid did nothing (empty PAM placeholder; run_dynamic_tree_cut passed no
        // distances)
        let (case, t, m, x, d) = avg_tree("dynamic_tree_cut");
        let run = |c: &DynamicTreeCutConfig| labels(cutree_dynamic(&t, sq(&m), c).unwrap());
        let mut c = Check::new(&case);
        for ds in 0..=4u8 {
            c.vector(
                &format!("hybrid_ds{ds}"),
                &run(&cfg(DynamicCutMethod::Hybrid, ds)),
                Tol::exact(),
            );
            // the condensed dissimilarities give the same labels
            assert_eq!(
                cutree_dynamic(
                    &t,
                    Some(DynamicCutDistances::Condensed(&d)),
                    &cfg(DynamicCutMethod::Hybrid, ds)
                )
                .unwrap()
                .labels,
                cutree_dynamic(&t, sq(&m), &cfg(DynamicCutMethod::Hybrid, ds))
                    .unwrap()
                    .labels
            );
        }
        let mut nopam = cfg(DynamicCutMethod::Hybrid, 2);
        nopam.pam_stage = false;
        let mut cut = cfg(DynamicCutMethod::Hybrid, 2);
        cut.cut_height = Some(1.5);
        let mut pam0 = cfg(DynamicCutMethod::Hybrid, 2);
        pam0.max_pam_dist = Some(0.0);
        let ward = run_dynamic_tree_cut(
            x.view(),
            Some(FastLinkage::Ward),
            Some(DynamicCutMethod::Hybrid),
            Some(1),
            Some(5),
        )
        .unwrap();
        c.vector("hybrid_ds2_nopam", &run(&nopam), Tol::exact())
            .vector("hybrid_ds2_cut15", &run(&cut), Tol::exact())
            .vector("hybrid_ds2_maxpam0", &run(&pam0), Tol::exact())
            .vector(
                "hybrid_defaults",
                &run(&DynamicTreeCutConfig::default()),
                Tol::exact(),
            )
            .vector("ward_hybrid_ds1", &labels(ward), Tol::exact());
        c.finish();
    }

    #[test]
    fn test_validate_dynamic_tree_cut_options_change_the_result() {
        // phantom-option rule: R distinguishes each pair (see the dynamic_tree_cut case values)
        let (_, t, m, _, _) = avg_tree("dynamic_tree_cut");
        let run = |c: &DynamicTreeCutConfig, d| cutree_dynamic(&t, d, c).unwrap().labels;
        let base = cfg(DynamicCutMethod::Hybrid, 2);
        let mut nopam = base.clone();
        nopam.pam_stage = false;
        let mut pam0 = base.clone();
        pam0.max_pam_dist = Some(0.0);
        assert_ne!(run(&base, sq(&m)), run(&nopam, sq(&m)));
        assert_ne!(run(&base, sq(&m)), run(&pam0, sq(&m)));
        assert_ne!(
            run(&cfg(DynamicCutMethod::Hybrid, 1), sq(&m)),
            run(&cfg(DynamicCutMethod::Hybrid, 4), sq(&m))
        );
        assert_ne!(
            run(&cfg(DynamicCutMethod::Tree, 0), None),
            run(&cfg(DynamicCutMethod::Tree, 1), None)
        );
        assert_ne!(
            run(&cfg(DynamicCutMethod::Tree, 1), None),
            run(&cfg(DynamicCutMethod::Hybrid, 1), sq(&m))
        );
    }

    #[test]
    fn test_validate_dynamic_tree_cut_defaults_and_fallback() {
        // ML-57: the documented default is Hybrid; without distances R warns and runs 'tree'
        // (dynamic_tree_cut data)
        assert_eq!(DynamicCutMethod::default(), DynamicCutMethod::Hybrid);
        let d = DynamicTreeCutConfig::default();
        assert_eq!(
            (d.method, d.deep_split, d.min_cluster_size, d.pam_stage),
            (DynamicCutMethod::Hybrid, None, 20, true)
        );
        let (_, t, _, _, _) = avg_tree("dynamic_tree_cut");
        let r = cutree_dynamic(&t, None, &cfg(DynamicCutMethod::Hybrid, 1)).unwrap();
        assert_eq!(r.method, DynamicCutMethod::Tree);
        assert_eq!(
            r.labels,
            cutree_dynamic(&t, None, &cfg(DynamicCutMethod::Tree, 1))
                .unwrap()
                .labels
        );
        assert!(
            r.warnings[0].contains("requires a dissimilarity matrix"),
            "{:?}",
            r.warnings
        );
        assert!(format!("{r}").contains("Warning:"));
        let mut bad = cfg(DynamicCutMethod::Hybrid, 5);
        assert!(
            cutree_dynamic(&t, None, &bad)
                .unwrap_err()
                .starts_with("deep_split")
        );
        bad = cfg(DynamicCutMethod::Hybrid, 1);
        bad.min_cluster_size = 1;
        assert!(
            cutree_dynamic(&t, None, &bad)
                .unwrap_err()
                .starts_with("min_cluster_size")
        );
        bad = cfg(DynamicCutMethod::Hybrid, 1);
        bad.cut_height = Some(f64::NAN);
        assert!(
            cutree_dynamic(&t, None, &bad)
                .unwrap_err()
                .starts_with("cut_height")
        );
    }

    #[test]
    fn test_dynamic_cut_method_parses_aliases_and_rejects_unknown() {
        DynamicCutMethod::choices().validate().unwrap();
        assert_eq!(
            "Hybrid".parse::<DynamicCutMethod>().unwrap(),
            DynamicCutMethod::Hybrid
        );
        assert!(
            "pam"
                .parse::<DynamicCutMethod>()
                .unwrap_err()
                .to_string()
                .contains("expected one of: hybrid, tree")
        );
    }

    #[test]
    fn test_dynamic_tree_cut_n_below_2_is_error() {
        // ML-115: n = 2 and n = 1 panicked with "attempt to subtract with overflow" in debug
        // builds; n = 0 underflowed n - 1
        let empty = Array2::<i64>::zeros((0, 2));
        for n in [0usize, 1] {
            let e = dynamic_tree_cut(&empty, &[], n, None, None, None, None).unwrap_err();
            assert!(e.contains("at least 2 observations"), "n={n}: {e}");
        }
        assert!(
            run_dynamic_tree_cut(array![[1.0, 2.0]].view(), None, None, None, Some(2)).is_err()
        );
    }

    #[test]
    fn test_dynamic_tree_cut_n2_unassigned_as_r() {
        // R: cutreeDynamic(hclust(dist(rbind(c(0,0), c(1,2)))), ...) gives 0 0 (hybrid default,
        // hybrid and tree with minClusterSize 2)
        let merge = array![[-1i64, -2]];
        let d = array![[0.0, 5f64.sqrt()], [5f64.sqrt(), 0.0]];
        for (method, dist) in [
            (Some(DynamicCutMethod::Hybrid), Some(&d)),
            (Some(DynamicCutMethod::Tree), None),
        ] {
            let r =
                dynamic_tree_cut(&merge, &[5f64.sqrt()], 2, method, None, Some(2), dist).unwrap();
            assert_eq!((r.labels, r.n_clusters, r.n_unassigned), (vec![0, 0], 0, 2));
        }
        let r = dynamic_tree_cut(&merge, &[5f64.sqrt()], 2, None, None, None, Some(&d)).unwrap();
        assert_eq!(r.labels, vec![0, 0]);
    }

    #[test]
    fn test_dynamic_tree_cut_basic() {
        // R (dynamicTreeCut 1.63.1): ward.D2, method "tree", deepSplit TRUE, minClusterSize 2
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = run_dynamic_tree_cut(
            data.view(),
            Some(FastLinkage::Ward),
            Some(DynamicCutMethod::Tree),
            Some(2), // deep_split
            Some(2), // min_cluster_size
        )
        .unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.labels, vec![2, 2, 2, 1, 1, 1]);
        assert_eq!(result.method, DynamicCutMethod::Tree);
    }

    #[test]
    fn test_dynamic_tree_cut_deep_split() {
        // R: ward.D2, method "hybrid", minClusterSize 2: deepSplit 0 and 4 both give 1 1 1 2 2 2
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ];
        for ds in [0u8, 4] {
            let r = run_dynamic_tree_cut(data.view(), None, None, Some(ds), Some(2)).unwrap();
            assert_eq!(r.method, DynamicCutMethod::Hybrid);
            assert_eq!(r.labels, vec![1, 1, 1, 2, 2, 2], "deep_split {ds}");
        }
    }
}
