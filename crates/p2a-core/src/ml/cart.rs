//! CART (Classification and Regression Trees), as R's `rpart` for continuous predictors.
//!
//! The tree follows Breiman et al. (1984) and rpart 4.1.23: unit weights, default priors, 0/1 loss,
//! no surrogate or competitor splits.
//!
//! - **Regression trees** (`anova`): the split that most reduces the sum of squares.
//! - **Classification trees** (`gini`, `information`): the split that most reduces the impurity;
//!   the node risk is the misclassification count, so the tree does not depend on how the classes
//!   are coded.
//! - **Complexity table**: weakest-link complexities of the nodes and the table of the nested
//!   optimal subtrees (rpart's `cptable`), one row per subtree in the sequence.
//! - **Cross-validation**: `xerror` and `xstd` from a real V-fold cross-validation over the
//!   geometric means of the table's complexities (`CartConfig::folds`, rpart's `xval = <vector>`).
//! - **Pruning**: [`cart_prune`], as `rpart::prune`.
//! - **Variable importance**: the summed improvement of the primary splits on each variable.
//!
//! Not implemented: surrogate splits and missing feature values, factor predictors, case weights,
//! priors and loss matrices.
//!
//! ## Example
//!
//! ```rust
//! use p2a_core::ml::{cart, CartConfig, CartMethod};
//! use ndarray::{Array1, Array2};
//!
//! // A step at x = 0.5 and a little noise.
//! let x = Array2::from_shape_fn((40, 1), |(i, _)| i as f64 / 40.0);
//! let y = Array1::from_shape_fn(40, |i| {
//!     2.0 * f64::from(u8::from(i as f64 / 40.0 > 0.5)) + ((7 * i) % 5) as f64 / 10.0
//! });
//!
//! let mut config = CartConfig::default();
//! config.method = CartMethod::Anova;
//! config.folds = Some((0..40).map(|i| i % 5 + 1).collect());
//!
//! let result = cart(x.view(), y.view(), &config).unwrap();
//! assert!((result.cp_table[0].rel_error - 1.0).abs() < 1e-12);
//! assert!(result.cp_table[0].xerror.is_some());
//!
//! // The 1-SE rule: the simplest subtree within one standard error of the best one.
//! let pruned = p2a_core::ml::cart_prune(&result, result.one_se_cp().unwrap()).unwrap();
//! assert!(pruned.n_terminal <= result.n_terminal);
//! ```

use std::collections::BTreeMap;

use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::extract::class_labels_line;
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};

/// Method for splitting nodes (determines loss function).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CartMethod {
    /// ANOVA (regression): the split that most reduces the sum of squares.
    #[default]
    Anova,
    /// Classification by the Gini index (`rpart(method = "class")`, its default).
    Gini,
    /// Classification by the information index (`parms = list(split = "information")`).
    Entropy,
}

crate::option_enum!(
    CartMethod,
    crate::options::Choices {
        option: "method",
        values: &[
            (&["anova", "regression", "mse"], CartMethod::Anova),
            (&["gini", "class", "classification"], CartMethod::Gini),
            (&["information", "entropy", "deviance"], CartMethod::Entropy),
        ],
    }
);

/// Configuration for CART.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartConfig {
    /// Method for splitting (anova for regression, gini/entropy for classification)
    pub method: CartMethod,
    /// Maximum depth of the tree, the root counting as 0: 1 to 30 (rpart's limit); 0 means 30.
    pub max_depth: usize,
    /// Smallest node that is tried for a split (rpart `minsplit`).
    pub min_split: usize,
    /// Smallest terminal node (rpart `minbucket`); 0 acts as 1.
    pub min_bucket: usize,
    /// Complexity parameter: a split that does not reduce the risk by this share of the root risk
    /// is not kept (finite and not negative).
    pub cp: f64,
    /// Number of random cross-validation folds when `folds` is not given; below 2 there is no
    /// cross-validation. Default 10, as rpart.
    pub xval: usize,
    /// Not implemented (surrogate splits, ML-125): has no effect; non-finite features are refused.
    pub max_surrogate: usize,
    /// Not implemented (surrogate splits, ML-125): has no effect; non-finite features are refused.
    pub use_surrogate: bool,
    /// Seed of the random folds; `None` means 42.
    pub seed: Option<u64>,
    /// The fold (1..=V, every label used) of every row, rpart's `xval = <vector>`. Takes precedence
    /// over `xval`.
    #[serde(default)]
    pub folds: Option<Vec<usize>>,
}

impl Default for CartConfig {
    fn default() -> Self {
        CartConfig {
            method: CartMethod::Anova,
            max_depth: 30,
            min_split: 20,
            min_bucket: 7,
            cp: 0.01,
            xval: 10,
            max_surrogate: 5,
            use_surrogate: true,
            seed: None,
            folds: None,
        }
    }
}

/// A split in the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartSplit {
    /// Feature index used for the split.
    pub feature: usize,
    /// Split point.
    pub threshold: f64,
    /// Absolute decrease of the split criterion: the sum of squares (anova), n x impurity (class).
    /// For anova this is rpart's `improve` times the node deviance; for class, rpart's `improve`.
    pub improve: f64,
    /// `true`: rows with `x < threshold` go to the left child (rpart `ncat = -1`). `false`: they go
    /// to the right child. The side with the smaller mean (or mean class index) goes left.
    pub direction: bool,
}

/// A node in the CART tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartNode {
    /// rpart's node number: the root is 1 and the children of k are 2k and 2k + 1.
    pub id: usize,
    /// Number of observations in this node
    pub n: usize,
    /// Weighted number of observations
    pub wt: f64,
    /// Node risk (rpart `frame$dev`): the sum of squares (anova), the misclassification count (class).
    pub dev: f64,
    /// Fitted value: the mean (anova) or the class value, an element of [`CartResult::classes`].
    pub yval: f64,
    /// Class shares in the order of [`CartResult::classes`] (classification only).
    pub class_probs: Option<Vec<f64>>,
    /// Weakest-link complexity (rpart `frame$complexity`), scaled by 1 / root risk.
    pub complexity: f64,
    /// Split information (None for terminal nodes)
    pub split: Option<CartSplit>,
    /// Left child node (if split)
    pub left: Option<Box<CartNode>>,
    /// Right child node (if split)
    pub right: Option<Box<CartNode>>,
}

/// Result from CART fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartResult {
    /// Root node of the tree
    pub root: CartNode,
    /// Total number of nodes
    pub n_nodes: usize,
    /// Number of terminal nodes (leaves)
    pub n_terminal: usize,
    /// Maximum depth reached
    pub depth: usize,
    /// Summed improvement of the splits on each feature; 0 for an unused one. Primary splits only,
    /// as `rpart(maxsurrogate = 0)`; rpart's default also counts surrogate splits.
    pub variable_importance: Vec<f64>,
    /// Complexity table of the nested optimal subtrees (rpart `cptable`).
    pub cp_table: Vec<CpTableRow>,
    /// The `xerror` column when cross-validation ran.
    pub cv_error: Option<Vec<f64>>,
    /// Configuration used
    pub config: CartConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Labels of the class codes in `predictions` when the target was text, categorical or
    /// boolean (gini or entropy method, [`run_cart`] only); code k is `class_labels[k]`.
    pub class_labels: Option<Vec<String>>,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Sorted distinct class values; empty for anova.
    #[serde(default)]
    pub classes: Vec<f64>,
    /// Risk of the root (rpart `frame$dev[1]`), the scale of `cp` and `rel_error`.
    #[serde(default)]
    pub root_risk: f64,
    /// The node id of each training row's leaf (rpart `where`, as node numbers).
    #[serde(default)]
    pub leaf_ids: Vec<usize>,
    /// Folds of the cross-validation that hold rows (0 = no cross-validation).
    #[serde(default)]
    pub n_folds: usize,
    /// Warnings about the fit.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl CartResult {
    /// The nodes in rpart's frame order: node, left subtree, right subtree.
    pub fn nodes_preorder(&self) -> Vec<&CartNode> {
        let mut out = Vec::with_capacity(self.n_nodes);
        let mut stack = vec![&self.root];
        while let Some(node) = stack.pop() {
            out.push(node);
            if let Some(right) = &node.right {
                stack.push(right);
            }
            if let Some(left) = &node.left {
                stack.push(left);
            }
        }
        out
    }

    /// The row of the smallest `xerror` (the first one on a tie).
    fn min_xerror_row(&self) -> Option<(f64, f64, f64)> {
        let mut best: Option<(f64, f64, f64)> = None;
        for row in &self.cp_table {
            let (xerror, xstd) = (row.xerror?, row.xstd?);
            if best.is_none_or(|(_, b, _)| xerror < b) {
                best = Some((row.cp, xerror, xstd));
            }
        }
        best
    }

    /// CP of the first row with the smallest `xerror`; `None` without cross-validation.
    pub fn min_xerror_cp(&self) -> Option<f64> {
        self.min_xerror_row().map(|(cp, _, _)| cp)
    }

    /// The 1-SE rule: CP of the first row whose `xerror` is at most the smallest `xerror` plus its
    /// `xstd`; `None` without cross-validation.
    pub fn one_se_cp(&self) -> Option<f64> {
        let (_, xerror, xstd) = self.min_xerror_row()?;
        let bound = xerror + xstd;
        self.cp_table
            .iter()
            .find(|row| row.xerror.is_some_and(|x| x <= bound))
            .map(|row| row.cp)
    }
}

impl ResultNotes for CartResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// Row in the complexity table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpTableRow {
    /// Complexity parameter: the largest cp at which this subtree is the optimal one.
    pub cp: f64,
    /// Number of splits of the subtree.
    pub nsplit: usize,
    /// Risk of the subtree relative to the root risk.
    pub rel_error: f64,
    /// Cross-validated risk relative to the root risk; `None` without cross-validation.
    pub xerror: Option<f64>,
    /// Standard error of `xerror`; `None` without cross-validation.
    pub xstd: Option<f64>,
}

impl std::fmt::Display for CartResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CART Decision Tree")?;
        writeln!(f, "==================")?;
        writeln!(f, "Method: {}", self.config.method)?;
        if let Some(labels) = &self.class_labels {
            writeln!(f, "{}", class_labels_line(labels))?;
        }
        writeln!(f, "Nodes: {} ({} terminal)", self.n_nodes, self.n_terminal)?;
        writeln!(f, "Depth: {}", self.depth)?;
        writeln!(
            f,
            "Root node error: {:.4}/{} = {:.4}",
            self.root_risk,
            self.root.n,
            self.root_risk / self.root.n as f64
        )?;
        if self.n_folds > 0 {
            writeln!(f, "Cross-validation: {} folds", self.n_folds)?;
        } else {
            writeln!(f, "Cross-validation: none")?;
        }

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Normalize to percentage
        let total: f64 = indexed.iter().map(|(_, v)| v).sum();
        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            let pct = if total > 0.0 {
                importance / total * 100.0
            } else {
                0.0
            };
            writeln!(f, "  {}: {:.1}%", name, pct)?;
        }

        if !self.cp_table.is_empty() {
            writeln!(f)?;
            writeln!(f, "Complexity Parameter Table:")?;
            if self.n_folds > 0 {
                writeln!(f, "  CP      nsplit  rel_error  xerror    xstd")?;
            } else {
                writeln!(f, "  CP      nsplit  rel_error")?;
            }
            for row in &self.cp_table {
                write!(
                    f,
                    "  {:.4}  {:>6}  {:>9.4}",
                    row.cp, row.nsplit, row.rel_error
                )?;
                if let (Some(xerror), Some(xstd)) = (row.xerror, row.xstd) {
                    write!(f, "  {xerror:>7.4}  {xstd:>7.4}")?;
                }
                writeln!(f)?;
            }
        }

        write!(f, "{}", notes_text(self))
    }
}

/// Twice `min_bucket`, the rows two children of `min_bucket` rows hold: a smaller node cannot be
/// split. It saturates at `usize::MAX`, more rows than any node can hold, so a `min_bucket` above
/// `usize::MAX / 2` stops every split instead of overflowing; below that it is exactly
/// `2 * min_bucket`.
fn two_buckets(min_bucket: usize) -> usize {
    min_bucket.saturating_mul(2)
}

/// The split a node keeps while the tree grows.
#[derive(Clone, Copy)]
struct GrowSplit {
    feature: usize,
    threshold: f64,
    /// Absolute decrease of the criterion.
    decrease: f64,
    /// The criterion as the feature comparison sees it: relative to the node risk for anova.
    improve: f64,
    /// Rows with `x < threshold` go left.
    dir_left: bool,
}

struct GrowNode {
    id: usize,
    n: usize,
    risk: f64,
    /// While growing: first the bound the parent sets, then the node's weakest-link complexity.
    complexity: f64,
    /// Node mean (anova).
    mean: f64,
    /// Index of the fitted class (class trees).
    class: usize,
    /// Class counts (class trees).
    counts: Vec<usize>,
    split: Option<GrowSplit>,
    left: Option<Box<GrowNode>>,
    right: Option<Box<GrowNode>>,
}

impl GrowNode {
    fn new(id: usize, complexity: f64) -> Self {
        GrowNode {
            id,
            n: 0,
            risk: 0.0,
            complexity,
            mean: 0.0,
            class: 0,
            counts: Vec::new(),
            split: None,
            left: None,
            right: None,
        }
    }

    fn make_leaf(&mut self) {
        self.split = None;
        self.left = None;
        self.right = None;
    }

    /// The child a row with the feature value `x` goes to.
    fn branch(&self, split: &GrowSplit, x: f64) -> Option<&GrowNode> {
        if (x < split.threshold) == split.dir_left {
            self.left.as_deref()
        } else {
            self.right.as_deref()
        }
    }
}

/// The split point between the adjacent sorted values `lo < hi`: their midpoint, unless the sum
/// overflows or rounding puts it on `lo`, where no row would go to the `x < threshold` side.
fn split_point(lo: f64, hi: f64) -> f64 {
    let mut mid = (lo + hi) / 2.0;
    if !mid.is_finite() {
        mid = lo / 2.0 + hi / 2.0;
    }
    if mid > lo && mid <= hi { mid } else { hi }
}

/// The best split point of one feature in one node.
struct Choice {
    /// Absolute decrease of the criterion; 0 when no position improves it.
    decrease: f64,
    lo: f64,
    hi: f64,
    dir_left: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Criterion {
    Anova,
    Gini,
    Information,
}

/// `-sum_j c_j ln(c_j / t)` over the classes of a side with `t` rows.
fn information(counts: &[u64], t: f64) -> f64 {
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let c = c as f64;
            -c * (c / t).ln()
        })
        .sum()
}

/// One tree at a time over presorted row indices. A node owns the segment `[n1, n2)` of every
/// feature's sorted rows, and splitting it partitions those segments stably, so one sort serves the
/// whole tree and every cross-validation fold.
struct Grower<'a> {
    n: usize,
    p: usize,
    /// Features, column after column.
    x: &'a [f64],
    /// Target (anova).
    y: &'a [f64],
    /// Class index of every row (class trees).
    codes: &'a [usize],
    n_classes: usize,
    criterion: Criterion,
    min_split: usize,
    edge: usize,
    too_small: usize,
    max_node: usize,
    alpha: f64,
    /// The largest improvement of this fit: a split below 1e-10 of it is rounding noise.
    iscale: f64,
    /// Sorted rows of every feature, column after column.
    sorts: Vec<usize>,
    tmp: Vec<usize>,
    /// Scratch: the side of each row at the split being applied.
    goes_left: Vec<bool>,
    /// Scratch: class counts of the two sides.
    left_counts: Vec<u64>,
    right_counts: Vec<u64>,
}

impl Grower<'_> {
    fn column(&self, j: usize) -> &[f64] {
        &self.x[j * self.n..(j + 1) * self.n]
    }

    /// Risk and fitted value of the rows `[n1, n2)`.
    fn eval_node(&mut self, node: &mut GrowNode, n1: usize, n2: usize) {
        let rows = &self.sorts[n1..n2];
        node.n = rows.len();
        if self.criterion == Criterion::Anova {
            let sum: f64 = rows.iter().map(|&r| self.y[r]).sum();
            let mean = sum / rows.len() as f64;
            node.mean = mean;
            node.risk = rows
                .iter()
                .map(|&r| {
                    let d = self.y[r] - mean;
                    d * d
                })
                .sum();
        } else {
            let mut counts = vec![0usize; self.n_classes];
            for &r in rows {
                counts[self.codes[r]] += 1;
            }
            // The first class of minimal risk wins, so ties go to the lowest class.
            let (mut best, mut best_count) = (0, 0);
            for (i, &c) in counts.iter().enumerate() {
                if c > best_count {
                    best = i;
                    best_count = c;
                }
            }
            node.class = best;
            node.risk = (rows.len() - best_count) as f64;
            node.counts = counts;
        }
    }

    /// The best position of one feature, anova.
    fn choose_anova(&self, seg: &[usize], xc: &[f64], grand: f64) -> Choice {
        let k = seg.len();
        let edge = self.edge;
        let mut c = Choice {
            decrease: 0.0,
            lo: 0.0,
            hi: 0.0,
            dir_left: true,
        };
        if k <= edge {
            return c;
        }
        let (mut left_sum, mut right_sum) = (0.0f64, 0.0f64);
        let mut cur = xc[seg[0]];
        for i in 0..k - edge {
            let t = self.y[seg[i]] - grand;
            left_sum += t;
            right_sum -= t;
            let next = xc[seg[i + 1]];
            if next != cur && i + 1 >= edge {
                let (lw, rw) = ((i + 1) as f64, (k - i - 1) as f64);
                let t = left_sum * left_sum / lw + right_sum * right_sum / rw;
                if t > c.decrease {
                    c.decrease = t;
                    c.lo = cur;
                    c.hi = next;
                    c.dir_left = left_sum < right_sum;
                }
            }
            cur = next;
        }
        c
    }

    /// The best position of one feature, class trees. Gini is `t - sum_j c_j^2 / t` for a side with
    /// class counts `c_j` and `t` rows, with the squares kept in integers, so that two positions
    /// with the same counts compare exactly equal whatever the class order: the first one wins.
    fn choose_class(
        &self,
        seg: &[usize],
        xc: &[f64],
        counts: &[usize],
        left: &mut [u64],
        right: &mut [u64],
    ) -> Choice {
        let k = seg.len();
        let edge = self.edge;
        let total = k as f64;
        left.fill(0);
        let (mut right_sq, mut right_index) = (0u64, 0u64);
        for (j, (&c, r)) in counts.iter().zip(right.iter_mut()).enumerate() {
            *r = c as u64;
            right_sq += *r * *r;
            right_index += j as u64 * *r;
        }
        let (mut left_sq, mut left_index) = (0u64, 0u64);
        let gini = self.criterion == Criterion::Gini;
        let total_ss = if gini {
            total - right_sq as f64 / total
        } else {
            information(right, total)
        };
        let mut best = total_ss;
        let (mut lo, mut hi, mut dir_left) = (0.0, 0.0, true);
        if k > edge {
            let mut cur = xc[seg[0]];
            for i in 0..k - edge {
                let j = self.codes[seg[i]];
                left_sq += 2 * left[j] + 1;
                right_sq -= 2 * right[j] - 1;
                left[j] += 1;
                right[j] -= 1;
                left_index += j as u64;
                right_index -= j as u64;
                let next = xc[seg[i + 1]];
                if next != cur && i + 1 >= edge {
                    let (lt, rt) = ((i + 1) as f64, (k - i - 1) as f64);
                    let t = if gini {
                        total - (left_sq as f64 / lt + right_sq as f64 / rt)
                    } else {
                        information(left, lt) + information(right, rt)
                    };
                    if t < best {
                        best = t;
                        lo = cur;
                        hi = next;
                        dir_left = left_index as f64 / lt < right_index as f64 / rt;
                    }
                }
                cur = next;
            }
        }
        Choice {
            decrease: total_ss - best,
            lo,
            hi,
            dir_left,
        }
    }

    /// The primary split of the node over `[n1, n2)`: the first feature of the largest improvement.
    fn best_split(
        &mut self,
        node: &GrowNode,
        n1: usize,
        n2: usize,
    ) -> EconResult<Option<GrowSplit>> {
        let mut left = std::mem::take(&mut self.left_counts);
        let mut right = std::mem::take(&mut self.right_counts);
        let mut primary: Option<GrowSplit> = None;
        for j in 0..self.p {
            resource::checkpoint()?;
            let seg = &self.sorts[j * self.n + n1..j * self.n + n2];
            let xc = self.column(j);
            if xc[seg[0]] == xc[seg[seg.len() - 1]] {
                continue;
            }
            let (choice, improve) = if self.criterion == Criterion::Anova {
                let c = self.choose_anova(seg, xc, node.mean);
                let improve = c.decrease / node.risk;
                (c, improve)
            } else {
                let c = self.choose_class(seg, xc, &node.counts, &mut left, &mut right);
                let improve = c.decrease;
                (c, improve)
            };
            if improve > self.iscale {
                self.iscale = improve;
            }
            if improve > self.iscale * 1e-10 && primary.is_none_or(|b| improve > b.improve) {
                primary = Some(GrowSplit {
                    feature: j,
                    threshold: split_point(choice.lo, choice.hi),
                    decrease: choice.decrease,
                    improve,
                    dir_left: choice.dir_left,
                });
            }
        }
        self.left_counts = left;
        self.right_counts = right;
        Ok(primary)
    }

    /// Partition `[n1, n2)` of every feature's rows into the left rows, then the right rows, each
    /// in their sorted order. Returns the number of left rows.
    fn apply_split(&mut self, split: &GrowSplit, n1: usize, n2: usize) -> usize {
        let n = self.n;
        let mut nleft = 0;
        for i in n1..n2 {
            let r = self.sorts[split.feature * n + i];
            let left = (self.x[split.feature * n + r] < split.threshold) == split.dir_left;
            self.goes_left[r] = left;
            nleft += usize::from(left);
        }
        for j in 0..self.p {
            let seg = &mut self.sorts[j * n + n1..j * n + n2];
            let (mut l, mut t) = (0, 0);
            for i in 0..seg.len() {
                let r = seg[i];
                if self.goes_left[r] {
                    seg[l] = r;
                    l += 1;
                } else {
                    self.tmp[t] = r;
                    t += 1;
                }
            }
            seg[l..].copy_from_slice(&self.tmp[..t]);
        }
        nleft
    }

    /// Grow the node over `[n1, n2)`: returns the splits and the risk of its optimal subtree.
    fn partition(&mut self, node: &mut GrowNode, n1: usize, n2: usize) -> EconResult<(usize, f64)> {
        resource::checkpoint()?;
        let mut tempcp = if node.id > 1 {
            self.eval_node(node, n1, n2);
            node.risk.min(node.complexity)
        } else {
            node.risk
        };
        if node.n < self.min_split
            || node.n < self.too_small
            || tempcp <= self.alpha
            || node.id > self.max_node
        {
            node.complexity = self.alpha;
            return Ok((0, node.risk));
        }
        let Some(split) = self.best_split(node, n1, n2)? else {
            node.complexity = self.alpha;
            return Ok((0, node.risk));
        };
        node.split = Some(split);
        let nleft = self.apply_split(&split, n1, n2);

        let mut left = Box::new(GrowNode::new(2 * node.id, tempcp - self.alpha));
        let (mut left_split, mut left_risk) = self.partition(&mut left, n1, n1 + nleft)?;

        tempcp = ((node.risk - left_risk) / (left_split + 1) as f64).max(node.risk - left.risk);
        if tempcp > node.complexity {
            tempcp = node.complexity;
        }
        let mut right = Box::new(GrowNode::new(2 * node.id + 1, tempcp - self.alpha));
        let (mut right_split, mut right_risk) = self.partition(&mut right, n1 + nleft, n2)?;

        // Which child collapses first as the complexity grows: the node's own complexity is the
        // risk reduction per split of what is left when it is reached.
        let cost =
            |lr: f64, rr: f64, ls: usize, rs: usize| (node.risk - (lr + rr)) / (ls + rs + 1) as f64;
        tempcp = cost(left_risk, right_risk, left_split, right_split);
        if right.complexity > left.complexity {
            if tempcp > left.complexity {
                left_risk = left.risk;
                left_split = 0;
                tempcp = cost(left_risk, right_risk, left_split, right_split);
                if tempcp > right.complexity {
                    right_risk = right.risk;
                    right_split = 0;
                }
            }
        } else if tempcp > right.complexity {
            right_split = 0;
            right_risk = right.risk;
            tempcp = cost(left_risk, right_risk, left_split, right_split);
            if tempcp > left.complexity {
                left_risk = left.risk;
                left_split = 0;
            }
        }
        let complexity = cost(left_risk, right_risk, left_split, right_split);
        node.complexity = complexity;
        if complexity <= self.alpha {
            // The whole subtree does not pay for itself: the node stays a leaf with this complexity.
            node.make_leaf();
            return Ok((0, node.risk));
        }
        node.left = Some(left);
        node.right = Some(right);
        Ok((left_split + right_split + 1, left_risk + right_risk))
    }

    /// Grow a tree over the first `m` rows of every feature's segment.
    fn fit(&mut self, m: usize) -> EconResult<GrowNode> {
        self.iscale = 0.0;
        let mut root = GrowNode::new(1, 0.0);
        self.eval_node(&mut root, 0, m);
        root.complexity = root.risk;
        self.partition(&mut root, 0, m)?;
        Ok(root)
    }

    /// The fitted-value node of `row`'s feature values at the complexity `cp`, from `start`.
    fn descend<'t>(&self, start: &'t GrowNode, row: usize, cp: f64) -> &'t GrowNode {
        let mut node = start;
        while cp < node.complexity {
            let Some(split) = &node.split else { break };
            match node.branch(split, self.column(split.feature)[row]) {
                Some(child) => node = child,
                None => break,
            }
        }
        node
    }

    /// The error of `row` at `node`.
    fn error(&self, node: &GrowNode, row: usize) -> f64 {
        if self.criterion == Criterion::Anova {
            let d = self.y[row] - node.mean;
            d * d
        } else if self.codes[row] != node.class {
            1.0
        } else {
            0.0
        }
    }

    /// Run a left-out row down a fold's tree over the descending grid `cpx`. A row visits at most
    /// one node per depth, and each node is its fitted-value node over a contiguous stretch of the
    /// grid, so the errors go into difference arrays by stretch.
    fn rundown(
        &self,
        tree: &GrowNode,
        row: usize,
        cpx: &[f64],
        xrisk: &mut [f64],
        xsq: &mut [f64],
    ) {
        let mut node = tree;
        let mut i = 0;
        while i < cpx.len() {
            node = self.descend(node, row, cpx[i]);
            let end = if node.split.is_some() {
                i + 1 + cpx[i + 1..].partition_point(|&c| c >= node.complexity)
            } else {
                cpx.len()
            };
            let err = self.error(node, row);
            xrisk[i] += err;
            xrisk[end] -= err;
            let sq = err * err;
            xsq[i] += sq;
            xsq[end] -= sq;
            i = end;
        }
    }
}

/// Complexity of every node at most that of its parent, starting from the root's own value.
fn fix_cp(node: &mut GrowNode, parent: f64) {
    if node.complexity > parent {
        node.complexity = parent;
    }
    let me = node.complexity;
    if let Some(left) = &mut node.left {
        fix_cp(left, me);
    }
    if let Some(right) = &mut node.right {
        fix_cp(right, me);
    }
}

/// The complexities of the subtree sequence: fixes each node's complexity against its parent and
/// collects the distinct values at or above `alpha`.
fn make_cp_list(node: &mut GrowNode, parent: f64, alpha: f64, out: &mut Vec<f64>) {
    if node.complexity > parent {
        node.complexity = parent;
    }
    let me = node.complexity.max(alpha);
    if let Some(left) = &mut node.left {
        make_cp_list(left, me, alpha, out);
    }
    if let Some(right) = &mut node.right {
        make_cp_list(right, me, alpha, out);
    }
    if me < parent {
        out.push(me);
    }
}

/// Add each node's risk and the splits above it to the table rows where it is a leaf of the optimal
/// subtree. Returns the row above them. `cps` is descending and its first row has the root's
/// complexity, so the walk stops at row 0 at the latest.
fn make_cp_table(
    node: &GrowNode,
    parent: f64,
    nsplit: usize,
    cps: &[f64],
    risk: &mut [f64],
    splits: &mut [usize],
) -> usize {
    let mut pos = cps.len() - 1;
    if let (Some(left), Some(right)) = (&node.left, &node.right) {
        make_cp_table(left, node.complexity, 0, cps, risk, splits);
        pos = make_cp_table(right, node.complexity, nsplit + 1, cps, risk, splits);
    }
    while cps[pos] < parent {
        risk[pos] += node.risk;
        splits[pos] += nsplit;
        match pos.checked_sub(1) {
            Some(q) => pos = q,
            None => break,
        }
    }
    pos
}

/// The cumulative sums of a difference array, the last (sentinel) entry dropped.
fn cumulate(diff: &[f64]) -> Vec<f64> {
    diff[..diff.len() - 1]
        .iter()
        .scan(0.0, |sum, &d| {
            *sum += d;
            Some(*sum)
        })
        .collect()
}

fn count_non_finite(values: impl Iterator<Item = f64>) -> usize {
    values.filter(|v| !v.is_finite()).count()
}

/// A random permutation of `1..=v` repeated over `n` rows.
fn random_folds(n: usize, v: usize, seed: u64) -> Vec<usize> {
    let mut folds: Vec<usize> = (0..n).map(|i| i % v + 1).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        let j = super::lcg_random(&mut state) % (i + 1);
        folds.swap(i, j);
    }
    folds
}

/// Check a fold vector; returns V.
fn check_folds(folds: &[usize], n: usize) -> EconResult<usize> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    if folds.len() != n {
        return Err(invalid(format!(
            "CART: folds has {} entries but the data have {n} rows",
            folds.len()
        )));
    }
    if let Some(i) = folds.iter().position(|&f| f == 0) {
        return Err(invalid(format!(
            "CART: fold labels must be 1..=V; label 0 at row {i}"
        )));
    }
    let mut labels = folds.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let v = labels[labels.len() - 1];
    if v < 2 {
        return Err(invalid(format!(
            "CART: cross-validation needs at least 2 folds; folds uses only label {v}"
        )));
    }
    if let Some(missing) = labels
        .iter()
        .enumerate()
        .find(|&(i, &label)| label != i + 1)
        .map(|(i, _)| i + 1)
    {
        return Err(invalid(format!(
            "CART: fold labels must be 1..=V with every label used; label {missing} does not occur"
        )));
    }
    Ok(v)
}

fn to_public(node: &GrowNode, scale: f64, classes: &[f64], importance: &mut [f64]) -> CartNode {
    let is_class = !classes.is_empty();
    let split = node.split.map(|s| {
        importance[s.feature] += s.decrease;
        CartSplit {
            feature: s.feature,
            threshold: s.threshold,
            improve: s.decrease,
            direction: s.dir_left,
        }
    });
    let left = node
        .left
        .as_ref()
        .map(|c| Box::new(to_public(c, scale, classes, importance)));
    let right = node
        .right
        .as_ref()
        .map(|c| Box::new(to_public(c, scale, classes, importance)));
    CartNode {
        id: node.id,
        n: node.n,
        wt: node.n as f64,
        dev: node.risk,
        yval: if is_class {
            classes[node.class]
        } else {
            node.mean
        },
        class_probs: is_class.then(|| {
            node.counts
                .iter()
                .map(|&c| c as f64 / node.n as f64)
                .collect()
        }),
        complexity: node.complexity * scale,
        split,
        left,
        right,
    }
}

/// Count tree statistics recursively.
fn count_tree_stats(node: &CartNode) -> (usize, usize, usize) {
    // Returns (n_nodes, n_terminal, max_depth)
    match (&node.left, &node.right) {
        (Some(left), Some(right)) => {
            let (ln, lt, ld) = count_tree_stats(left);
            let (rn, rt, rd) = count_tree_stats(right);
            (1 + ln + rn, lt + rt, 1 + ld.max(rd))
        }
        _ => (1, 1, 0),
    }
}

/// The leaf a row reaches.
fn find_leaf(root: &CartNode, value: impl Fn(usize) -> f64) -> &CartNode {
    let mut node = root;
    while let (Some(split), Some(left), Some(right)) = (&node.split, &node.left, &node.right) {
        node = if (value(split.feature) < split.threshold) == split.direction {
            left
        } else {
            right
        };
    }
    node
}

/// The result for a constant target: one leaf and no complexity table.
fn single_leaf(
    node: GrowNode,
    classes: &[f64],
    p: usize,
    n: usize,
    config: &CartConfig,
) -> CartResult {
    let mut importance = vec![0.0; p];
    let root = to_public(&node, 1.0, classes, &mut importance);
    CartResult {
        n_nodes: 1,
        n_terminal: 1,
        depth: 0,
        variable_importance: importance,
        cp_table: Vec::new(),
        cv_error: None,
        config: config.clone(),
        feature_names: None,
        class_labels: None,
        predictions: vec![root.yval; n],
        classes: classes.to_vec(),
        root_risk: 0.0,
        leaf_ids: vec![1; n],
        n_folds: 0,
        warnings: vec![
            "the target is constant (root deviance 0): the tree is a single node and has no \
             complexity table"
                .to_string(),
        ],
        root,
    }
}

/// Fit a CART decision tree.
///
/// Without [`CartConfig::folds`], `xval >= 2` draws a seeded random permutation of `1..=V` repeated
/// over the rows (rpart's `sample(rep(1:xval, length.out = n))`, with p2a's generator, so the folds
/// differ from R's: pass `folds` to reproduce rpart).
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples); class values for the gini and entropy methods, found by value
/// * `config` - CART configuration
///
/// # Errors
///
/// `InvalidSpecification` for non-finite features or targets, a `cp` that is not finite or is
/// negative, `max_depth` above 30, malformed `folds` or a matrix without columns;
/// `InsufficientData` below 2 rows.
///
/// # Returns
///
/// CartResult containing the fitted tree and diagnostics.
pub fn cart(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &CartConfig) -> EconResult<CartResult> {
    let n = x.nrows();
    let p = x.ncols();
    let invalid = |message: String| EconError::InvalidSpecification { message };

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "CART".to_string(),
        });
    }
    if n != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }
    if p == 0 {
        return Err(invalid("CART: x has no columns".to_string()));
    }
    for j in 0..p {
        let k = count_non_finite(x.column(j).iter().copied());
        if k > 0 {
            return Err(invalid(format!(
                "CART: feature column {j} (0-based) has {k} non-finite values (NaN or infinite); \
                 remove or impute them first (surrogate splits for missing values are not implemented)"
            )));
        }
    }
    let k = count_non_finite(y.iter().copied());
    if k > 0 {
        return Err(invalid(format!(
            "CART: the target has {k} non-finite values (NaN or infinite); remove them first"
        )));
    }
    if !config.cp.is_finite() || config.cp < 0.0 {
        return Err(invalid(format!(
            "CART: cp must be finite and not negative, got {}",
            config.cp
        )));
    }
    if config.max_depth > 30 {
        return Err(invalid(format!(
            "CART: max_depth must be at most 30 (rpart's limit; 0 means 30), got {}",
            config.max_depth
        )));
    }
    let max_depth = if config.max_depth == 0 {
        30
    } else {
        config.max_depth
    };
    // Random folds beyond the row count would stay empty, so at most n of them are drawn.
    let random: Option<Vec<usize>> = (config.folds.is_none() && config.xval >= 2)
        .then(|| random_folds(n, config.xval.min(n), config.seed.unwrap_or(42)));
    let fold_labels: Option<(&[usize], usize)> = match (&config.folds, &random) {
        (Some(f), _) => {
            let v = check_folds(f, n)?;
            Some((f.as_slice(), v))
        }
        (None, Some(f)) => Some((f.as_slice(), config.xval.min(n))),
        (None, None) => None,
    };

    // Classes by value, never through an integer cast.
    let criterion = match config.method {
        CartMethod::Anova => Criterion::Anova,
        CartMethod::Gini => Criterion::Gini,
        CartMethod::Entropy => Criterion::Information,
    };
    let is_class = criterion != Criterion::Anova;
    let mut classes: Vec<f64> = Vec::new();
    let mut codes: Vec<usize> = Vec::new();
    if is_class {
        classes = y.to_vec();
        classes.sort_unstable_by(f64::total_cmp);
        classes.dedup();
        codes = y
            .iter()
            .map(|&v| classes.partition_point(|&c| c < v))
            .collect();
    }
    let n_classes = classes.len();
    let y_values = y.to_vec();

    // The features column after column, and every column's rows in sorted order (ties in row
    // order). This is the only sort: a fold filters it.
    let total = resource::checked_len(&[p, n], "CART sort indices")?;
    let mut xc = resource::try_vec(total, 0.0f64, "CART features")?;
    let mut sorts = resource::try_vec(total, 0usize, "CART sort indices")?;
    for j in 0..p {
        resource::checkpoint()?;
        let col = &mut xc[j * n..(j + 1) * n];
        for (slot, &v) in col.iter_mut().zip(x.column(j).iter()) {
            *slot = v;
        }
        let order = &mut sorts[j * n..(j + 1) * n];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        let col = &xc[j * n..(j + 1) * n];
        order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
    }

    let mut grower = Grower {
        n,
        p,
        x: &xc,
        y: &y_values,
        codes: &codes,
        n_classes,
        criterion,
        min_split: config.min_split,
        edge: config.min_bucket.max(1),
        too_small: two_buckets(config.min_bucket),
        max_node: (1usize << max_depth) - 1,
        alpha: 0.0,
        iscale: 0.0,
        sorts: Vec::new(),
        tmp: resource::try_vec(n, 0usize, "CART scratch rows")?,
        goes_left: resource::try_vec(n, false, "CART scratch rows")?,
        left_counts: resource::try_vec(n_classes, 0u64, "CART class counts")?,
        right_counts: resource::try_vec(n_classes, 0u64, "CART class counts")?,
    };
    let saved = fold_labels.is_some().then(|| sorts.clone());
    grower.sorts = sorts;

    // Main fit. alpha is the risk a split has to save.
    let mut root_eval = GrowNode::new(1, 0.0);
    grower.eval_node(&mut root_eval, 0, n);
    if is_class && n_classes == 1 || !is_class && y.iter().all(|&v| v == y[0]) {
        // rpart: NaN in every column of the table.
        root_eval.risk = 0.0;
        if !is_class {
            root_eval.mean = y[0];
        }
        return Ok(single_leaf(root_eval, &classes, p, n, config));
    }
    let root_risk = root_eval.risk;
    if !root_risk.is_finite() {
        return Err(invalid(
            "CART: the sum of squares of the target overflows; rescale the target".to_string(),
        ));
    }
    let alpha = config.cp * root_risk;
    grower.alpha = alpha;
    let mut tree = grower.fit(n)?;

    // The subtree sequence.
    let mut cps = vec![tree.complexity];
    if tree.split.is_some() {
        let root_complexity = tree.complexity;
        make_cp_list(&mut tree, root_complexity, alpha, &mut cps);
        cps.sort_unstable_by(|a, b| b.total_cmp(a));
        cps.dedup();
    }
    let rows = cps.len();
    let mut risk = resource::try_vec(rows, 0.0f64, "CART complexity table")?;
    let mut splits = resource::try_vec(rows, 0usize, "CART complexity table")?;
    risk[0] = root_risk;
    if tree.split.is_some() {
        let root_complexity = tree.complexity;
        make_cp_table(&tree, root_complexity, 0, &cps, &mut risk, &mut splits);
    }

    // Cross-validation over the geometric means of the table's complexities.
    let mut cv: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut n_folds = 0;
    if let (Some((folds, v)), Some(saved), true) = (fold_labels, &saved, tree.split.is_some()) {
        let mut cpx = Vec::with_capacity(rows);
        cpx.push(10.0 * cps[0]);
        for pair in cps.windows(2) {
            cpx.push((pair[0] * pair[1]).sqrt());
        }
        let mut xrisk_diff = resource::try_vec(rows + 1, 0.0f64, "CART cross-validation")?;
        let mut xsq_diff = resource::try_vec(rows + 1, 0.0f64, "CART cross-validation")?;
        let mut cpx_fold = vec![0.0; rows];
        for g in 1..=v {
            resource::checkpoint()?;
            let held_out = folds.iter().filter(|&&f| f == g).count();
            if held_out == 0 {
                continue;
            }
            n_folds += 1;
            let train = n - held_out;
            let weight = train as f64 / n as f64;
            grower.alpha = alpha * weight;
            for (c, &base) in cpx_fold.iter_mut().zip(&cpx) {
                *c = base * weight;
            }
            for j in 0..p {
                let from = &saved[j * n..(j + 1) * n];
                let to = &mut grower.sorts[j * n..(j + 1) * n];
                let mut w = 0;
                for &r in from {
                    if folds[r] != g {
                        to[w] = r;
                        w += 1;
                    }
                }
            }
            let mut fold_tree = grower.fit(train)?;
            let root_complexity = fold_tree.complexity;
            fix_cp(&mut fold_tree, root_complexity);
            for (row, _) in folds.iter().enumerate().filter(|&(_, &f)| f == g) {
                grower.rundown(&fold_tree, row, &cpx_fold, &mut xrisk_diff, &mut xsq_diff);
            }
        }
        let xrisk = cumulate(&xrisk_diff);
        let xsq = cumulate(&xsq_diff);
        let xstd: Vec<f64> = xsq
            .iter()
            .zip(&xrisk)
            .map(|(&sq, &r)| (sq - r * r / n as f64).max(0.0).sqrt())
            .collect();
        cv = Some((xrisk, xstd));
    }

    // Assembly.
    let scale = 1.0 / root_risk;
    let cp_table: Vec<CpTableRow> = (0..rows)
        .map(|i| CpTableRow {
            cp: cps[i] * scale,
            nsplit: splits[i],
            rel_error: risk[i] * scale,
            xerror: cv.as_ref().map(|(r, _)| r[i] * scale),
            xstd: cv.as_ref().map(|(_, s)| s[i] * scale),
        })
        .collect();
    let cv_error = cv
        .as_ref()
        .map(|(r, _)| r.iter().map(|v| v * scale).collect::<Vec<f64>>());

    let mut importance = vec![0.0; p];
    let root = to_public(&tree, scale, &classes, &mut importance);
    let (n_nodes, n_terminal, depth) = count_tree_stats(&root);

    let mut predictions = Vec::with_capacity(n);
    let mut leaf_ids = Vec::with_capacity(n);
    for i in 0..n {
        let leaf = find_leaf(&root, |j| x[[i, j]]);
        predictions.push(leaf.yval);
        leaf_ids.push(leaf.id);
    }

    Ok(CartResult {
        root,
        n_nodes,
        n_terminal,
        depth,
        variable_importance: importance,
        cp_table,
        cv_error,
        config: config.clone(),
        feature_names: None,
        class_labels: None,
        predictions,
        classes,
        root_risk,
        leaf_ids,
        n_folds,
        warnings: Vec::new(),
    })
}

fn check_prediction_input(result: &CartResult, x: ArrayView2<f64>) -> EconResult<()> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    let p = result.variable_importance.len();
    if x.ncols() != p {
        return Err(invalid(format!(
            "CART: x has {} columns but the tree was fitted on {p}",
            x.ncols()
        )));
    }
    if let Some(((i, j), _)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(invalid(format!(
            "CART: x row {i}, column {j} is not finite; surrogate splits are not implemented"
        )));
    }
    Ok(())
}

/// Predict using a fitted CART tree.
///
/// # Arguments
///
/// * `result` - Fitted CART result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions for each observation.
///
/// # Errors
///
/// `InvalidSpecification` when `x` has another column count than the training data or a value that
/// is not finite.
pub fn cart_predict(result: &CartResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    check_prediction_input(result, x)?;
    Ok((0..x.nrows())
        .map(|i| find_leaf(&result.root, |j| x[[i, j]]).yval)
        .collect())
}

/// Class probabilities of a fitted classification tree: one row per observation, one column per
/// class in the order of [`CartResult::classes`].
///
/// # Errors
///
/// `InvalidSpecification` for a regression tree, when `x` has another column count than the
/// training data, or for a value that is not finite.
pub fn cart_predict_proba(result: &CartResult, x: ArrayView2<f64>) -> EconResult<Array2<f64>> {
    if result.classes.is_empty() {
        return Err(EconError::InvalidSpecification {
            message:
                "CART: class probabilities need a classification tree (method gini or information)"
                    .to_string(),
        });
    }
    let mut out =
        resource::try_zeros2(x.nrows(), result.classes.len(), "CART class probabilities")?;
    for i in 0..x.nrows() {
        let leaf = find_leaf(&result.root, |j| x[[i, j]]);
        if let Some(probs) = &leaf.class_probs {
            for (k, &v) in probs.iter().enumerate() {
                out[[i, k]] = v;
            }
        }
    }
    Ok(out)
}

/// Prune a CART tree as `rpart::prune`: every split node whose complexity is at most `cp` becomes a
/// leaf, and the training predictions, the leaf ids, the importance of the remaining splits and the
/// complexity table follow. With nothing to collapse the tree comes back unchanged, its table
/// included.
///
/// # Arguments
///
/// * `result` - Fitted CART result
/// * `cp` - Complexity parameter to prune to
///
/// # Errors
///
/// `InvalidSpecification` when `cp` is not finite.
///
/// # Returns
///
/// Pruned tree result.
pub fn cart_prune(result: &CartResult, cp: f64) -> EconResult<CartResult> {
    if !cp.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("CART: cp must be finite, got {cp}"),
        });
    }
    let collapses = result
        .nodes_preorder()
        .iter()
        .any(|node| node.split.is_some() && node.complexity <= cp);
    if !collapses {
        return Ok(result.clone());
    }

    let mut pruned = result.clone();
    collapse_nodes(&mut pruned.root, cp);
    let (n_nodes, n_terminal, depth) = count_tree_stats(&pruned.root);
    pruned.n_nodes = n_nodes;
    pruned.n_terminal = n_terminal;
    pruned.depth = depth;

    let kept: BTreeMap<usize, f64> = pruned
        .nodes_preorder()
        .iter()
        .map(|node| (node.id, node.yval))
        .collect();
    for (id, prediction) in pruned.leaf_ids.iter_mut().zip(&mut pruned.predictions) {
        // A collapsed node's rows go to its deepest kept ancestor; the root is always kept.
        while *id > 1 && !kept.contains_key(id) {
            *id /= 2;
        }
        if let Some(&yval) = kept.get(id) {
            *prediction = yval;
        }
    }

    let mut importance = vec![0.0; pruned.variable_importance.len()];
    for node in pruned.nodes_preorder() {
        if let Some(split) = &node.split {
            importance[split.feature] += split.improve;
        }
    }
    pruned.variable_importance = importance;

    // The rows with CP above `cp` and the first row at or below it, whose CP becomes `cp`.
    let mut table: Vec<CpTableRow> = Vec::with_capacity(pruned.cp_table.len());
    for row in &pruned.cp_table {
        let collapsed = table
            .last()
            .is_some_and(|last| last.cp.max(cp) == row.cp.max(cp));
        if !collapsed {
            table.push(row.clone());
        }
    }
    if let Some(last) = table.last_mut() {
        last.cp = cp;
    }
    pruned.cv_error = pruned
        .cv_error
        .is_some()
        .then(|| table.iter().filter_map(|row| row.xerror).collect());
    pruned.cp_table = table;
    Ok(pruned)
}

/// Turn every split node at or below `cp` into a leaf.
fn collapse_nodes(node: &mut CartNode, cp: f64) {
    if node.split.is_none() {
        return;
    }
    if node.complexity <= cp {
        node.split = None;
        node.left = None;
        node.right = None;
        return;
    }
    if let Some(left) = &mut node.left {
        collapse_nodes(left, cp);
    }
    if let Some(right) = &mut node.right {
        collapse_nodes(right, cp);
    }
}

/// Run CART on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column: numeric (integer, decimal and, for anova, boolean 0/1
///   columns); with the gini or entropy method also a text, categorical or boolean
///   column, whose labels become the class codes 0..K-1 in R `factor()` order (reported
///   in [`CartResult::class_labels`]). A missing value (null or NaN) in the target or a
///   feature is an error naming the column (`EconError::NullValues`)
/// * `x_cols` - Feature column names
/// * `config` - CART configuration
///
/// # Returns
///
/// CartResult with fitted tree and diagnostics.
pub fn run_cart(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &CartConfig,
) -> EconResult<CartResult> {
    let fail = crate::data::columns::NaPolicy::Fail;
    let (x, target) = match config.method {
        CartMethod::Gini | CartMethod::Entropy => {
            let (x, target, _missing) =
                super::extract::extract_x_classes(dataset, y_col, x_cols, fail)?;
            (x, target)
        }
        CartMethod::Anova => {
            let (x, y, _missing) = super::extract::extract_xy(dataset, y_col, x_cols, fail)?;
            (x, super::extract::ClassTarget { y, labels: None })
        }
    };
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    let mut result = cart(x.view(), target.y.view(), config)?;
    result.feature_names = Some(feature_names);
    result.class_labels = target.labels;

    Ok(result)
}

/// Run CART with default configuration.
pub fn run_cart_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<CartResult> {
    run_cart(dataset, y_col, x_cols, &CartConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use crate::ml::extract_xy;
    use ndarray::{Array1, array};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_cart_regression_basic() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.1, 1.9, 2.2, 2.0, 5.0, 5.8, 6.1, 5.9, 9.0, 9.2];

        let config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x.view(), y.view(), &config).unwrap();

        assert!(result.n_nodes >= 1);
        assert!(result.n_terminal >= 1);
        assert!(result.depth <= 5);
        assert_eq!(result.predictions.len(), 10);
    }

    #[test]
    fn test_cart_classification_gini() {
        // Two clear groups
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [8.0, 1.0],
            [8.5, 0.5],
            [9.0, 1.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = CartConfig {
            method: CartMethod::Gini,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x.view(), y.view(), &config).unwrap();

        // Predictions should be close to actual classes
        for i in 0..3 {
            assert!(
                result.predictions[i] < 0.5,
                "Sample {} should be class 0",
                i
            );
        }
        for i in 3..6 {
            assert!(
                result.predictions[i] > 0.5,
                "Sample {} should be class 1",
                i
            );
        }
    }

    #[test]
    fn test_cart_predict() {
        let x_train = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
        let y_train = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];

        let config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![[2.5], [7.5]];
        let predictions = cart_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] < 5.0); // Should predict ~1
        assert!(predictions[1] > 5.0); // Should predict ~9
    }

    #[test]
    fn test_cart_prune() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 10,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x.view(), y.view(), &config).unwrap();
        let pruned = cart_prune(&result, 0.5).unwrap();

        assert!(pruned.n_nodes <= result.n_nodes);
    }

    #[test]
    fn test_cart_variable_importance() {
        // First feature is predictive, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [7.0, 2.0],
            [8.0, 8.0],
            [9.0, 4.0]
        ];
        let y = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];

        let config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x.view(), y.view(), &config).unwrap();

        // First feature should be more important
        assert!(
            result.variable_importance[0] >= result.variable_importance[1],
            "Feature 0 ({}) should be >= Feature 1 ({})",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }

    #[test]
    fn test_cart_depth_limit() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 2,
            min_split: 2,
            min_bucket: 1,
            cp: 0.0,
            ..Default::default()
        };

        let result = cart(x.view(), y.view(), &config).unwrap();

        assert!(result.depth <= 2, "Depth {} exceeds max 2", result.depth);
    }

    /// A `min_bucket` above `usize::MAX / 2` cannot split any node: twice it (the
    /// rows two children of `min_bucket` rows hold) used to overflow, a panic in a
    /// debug build and a product wrapped to a small number in a release build.
    /// The tree is a single leaf, exactly as at a `min_bucket` of all the rows, for
    /// regression and classification and through both entry points.
    #[test]
    fn a_min_bucket_above_half_of_usize_neither_panics_nor_wraps() {
        let x = array![
            [1.0, 0.3],
            [2.0, 0.1],
            [3.0, 0.7],
            [4.0, 0.2],
            [5.0, 0.9],
            [6.0, 0.4],
            [7.0, 0.8],
            [8.0, 0.6],
            [9.0, 0.5],
            [10.0, 0.0]
        ];
        let y_reg = array![1.1, 1.9, 2.2, 2.0, 5.0, 5.8, 6.1, 5.9, 9.0, 9.2];
        // No tie for the majority class, and the class probabilities sorted.
        let y_class = array![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let summary = |r: &CartResult| {
            let mut v = vec![r.root.dev, r.root.yval];
            let mut probs = r.root.class_probs.clone().unwrap_or_default();
            probs.sort_by(f64::total_cmp);
            v.extend(probs);
            v.extend(&r.predictions);
            v.extend(&r.variable_importance);
            v.extend(r.cp_table.iter().flat_map(|row| [row.cp, row.rel_error]));
            bits(&v)
        };
        for (method, y) in [(CartMethod::Anova, &y_reg), (CartMethod::Gini, &y_class)] {
            let fit = |min_bucket| {
                let config = CartConfig {
                    method,
                    max_depth: 5,
                    min_split: 2,
                    min_bucket,
                    cp: 0.0,
                    ..Default::default()
                };
                cart(x.view(), y.view(), &config).unwrap()
            };
            let want = fit(10);
            assert!(fit(1).n_nodes > 1, "{method:?}: the data can be split");
            for min_bucket in [usize::MAX / 2 + 1, usize::MAX - 1, usize::MAX] {
                let got = fit(min_bucket);
                let case = format!("{method:?}, min_bucket {min_bucket}");
                assert!(got.root.split.is_none(), "{case}");
                assert_eq!((got.n_nodes, got.n_terminal), (1, 1), "{case}");
                assert_eq!(summary(&got), summary(&want), "{case}");
            }
        }

        // run_cart, the Dataset entry point
        let ds = Dataset::new(
            polars::prelude::df! {
                "y" => y_reg.to_vec(),
                "x1" => x.column(0).to_vec(),
                "x2" => x.column(1).to_vec(),
            }
            .unwrap(),
        );
        let config = CartConfig {
            min_split: 2,
            min_bucket: usize::MAX / 2 + 1,
            ..Default::default()
        };
        let got = run_cart(&ds, "y", &["x1", "x2"], &config).unwrap();
        assert_eq!((got.n_nodes, got.n_terminal), (1, 1));

        // The rule itself: the product saturates above `usize::MAX / 2` and is
        // exact up to it.
        assert_eq!(two_buckets(usize::MAX / 2 + 1), usize::MAX);
        assert_eq!(two_buckets(usize::MAX), usize::MAX);
        assert_eq!(two_buckets(usize::MAX / 2), usize::MAX - 1);
        assert_eq!((two_buckets(0), two_buckets(7)), (0, 14));
    }

    const X_ANOVA: [&str; 4] = ["x1", "x2", "x3", "x4"];
    const X_CLASS: [&str; 3] = ["x1", "x2", "x3"];

    /// The fold column of a CART case (rpart's 1-based fold labels).
    fn case_folds(ds: &crate::Dataset) -> Vec<usize> {
        let (f, _) = numeric_column(ds.df(), "folds", NaPolicy::Fail).unwrap();
        f.iter().map(|&v| v as usize).collect()
    }

    /// The rpart.control of the cart_cp_* cases.
    fn case_config(ds: &crate::Dataset, method: CartMethod) -> CartConfig {
        let mut config = CartConfig::default();
        config.method = method;
        config.cp = 0.001;
        config.min_split = 20;
        config.min_bucket = 7;
        config.max_surrogate = 0;
        config.use_surrogate = false;
        config.folds = Some(case_folds(ds));
        config
    }

    #[derive(Default)]
    struct Frame {
        id: Vec<f64>,
        var: Vec<f64>,
        n: Vec<f64>,
        dev: Vec<f64>,
        yval: Vec<f64>,
        complexity: Vec<f64>,
        split_index: Vec<f64>,
        split_ncat: Vec<f64>,
    }

    /// rpart's frame and splits matrix of `r` (preorder), as the case files store them.
    fn frame(r: &CartResult) -> Frame {
        let mut f = Frame::default();
        for node in r.nodes_preorder() {
            f.id.push(node.id as f64);
            f.n.push(node.n as f64);
            f.dev.push(node.dev);
            f.yval.push(node.yval);
            f.complexity.push(node.complexity);
            match &node.split {
                Some(s) => {
                    f.var.push(s.feature as f64);
                    f.split_index.push(s.threshold);
                    f.split_ncat.push(if s.direction { -1.0 } else { 1.0 });
                }
                None => f.var.push(-1.0),
            }
        }
        f
    }

    fn check_frame(c: &mut Check<'_>, prefix: &str, r: &CartResult, tol: Tol) {
        let f = frame(r);
        c.vector(&format!("{prefix}_frame_id"), &f.id, Tol::exact())
            .vector(&format!("{prefix}_frame_var"), &f.var, Tol::exact())
            .vector(&format!("{prefix}_frame_n"), &f.n, Tol::exact())
            .vector(&format!("{prefix}_frame_dev"), &f.dev, tol)
            .vector(&format!("{prefix}_frame_yval"), &f.yval, tol)
            .vector(&format!("{prefix}_frame_complexity"), &f.complexity, tol);
        if !f.split_index.is_empty() {
            c.vector(&format!("{prefix}_split_index"), &f.split_index, tol)
                .vector(&format!("{prefix}_split_ncat"), &f.split_ncat, Tol::exact());
        }
    }

    fn check_cp_table(c: &mut Check<'_>, prefix: &str, rows: &[CpTableRow], tol: Tol, cv: bool) {
        let col = |g: fn(&CpTableRow) -> f64| rows.iter().map(g).collect::<Vec<f64>>();
        c.vector(&format!("{prefix}_cp"), &col(|r| r.cp), tol)
            .vector(
                &format!("{prefix}_nsplit"),
                &col(|r| r.nsplit as f64),
                Tol::exact(),
            )
            .vector(&format!("{prefix}_rel_error"), &col(|r| r.rel_error), tol);
        if cv {
            c.vector(
                &format!("{prefix}_xerror"),
                &col(|r| r.xerror.unwrap_or(f64::NAN)),
                tol,
            )
            .vector(
                &format!("{prefix}_xstd"),
                &col(|r| r.xstd.unwrap_or(f64::NAN)),
                tol,
            );
        }
    }

    /// rpart's variable.importance lists only variables with a split; the others must be exactly 0 here.
    fn check_importance(c: &mut Check<'_>, key: &str, r: &CartResult, tol: Tol) {
        let names = r
            .feature_names
            .clone()
            .expect("run_cart sets feature names");
        let used: Vec<usize> = (0..names.len())
            .filter(|&j| r.variable_importance[j] != 0.0)
            .collect();
        let n: Vec<&str> = used.iter().map(|&j| names[j].as_str()).collect();
        let v: Vec<f64> = used.iter().map(|&j| r.variable_importance[j]).collect();
        c.named(key, &n, &v, tol);
    }

    #[test]
    fn test_validate_cart_cp_anova() {
        // ML-71. rpart's cp table with the case's 10 folds. Before: xerror = 1.1 rel_error, xstd = 0.1,
        // 24 rows (R 18), rel_error[0] = 0.7831359233 (R 1).
        let (case, ds) = crate::test_support::load_case("cart_cp_anova");
        let r = run_cart(&ds, "y", &X_ANOVA, &case_config(&ds, CartMethod::Anova)).unwrap();
        let tol = Tol::coef(300);
        let mut c = Check::new(&case);
        check_cp_table(&mut c, "anova", &r.cp_table, tol, true);
        check_frame(&mut c, "anova", &r, tol);
        check_importance(&mut c, "anova_importance", &r, tol);
        c.vector("anova_pred", &r.predictions, tol)
            .scalar("anova_root_risk", r.root_risk, tol)
            .scalar("anova_min_xerror_cp", r.min_xerror_cp().unwrap(), tol)
            .scalar("anova_one_se_cp", r.one_se_cp().unwrap(), tol)
            .finish();
        assert_eq!(
            (r.n_nodes, r.n_terminal, r.depth, r.n_folds),
            (49, 25, 9, 10)
        );
        let xerror: Vec<f64> = r.cp_table.iter().map(|row| row.xerror.unwrap()).collect();
        assert_eq!(r.cv_error.as_deref(), Some(&xerror[..]));
        // without cross-validation: the same three columns, no xerror/xstd (R: fit0 in the case)
        let mut no_cv = case_config(&ds, CartMethod::Anova);
        no_cv.folds = None;
        no_cv.xval = 0;
        let r0 = run_cart(&ds, "y", &X_ANOVA, &no_cv).unwrap();
        assert_eq!(r0.cp_table.len(), 18);
        assert!(
            r0.cp_table
                .iter()
                .all(|row| row.xerror.is_none() && row.xstd.is_none())
        );
        assert!(
            r0.cp_table
                .iter()
                .zip(&r.cp_table)
                .all(|(a, b)| a.cp == b.cp && a.rel_error == b.rel_error)
        );
        assert_eq!((r0.cv_error, r0.n_folds), (None, 0));
    }

    #[test]
    fn test_validate_cart_cp_class() {
        // ML-70: labels 3 and 17. Before: the cp test divided gini x n by the variance of the codes (14^2 x ...),
        // giving a 3-node tree with 3 cp rows (R: 17 nodes, 5 rows).
        let (case, ds) = crate::test_support::load_case("cart_cp_class");
        let r = run_cart(&ds, "cls", &X_CLASS, &case_config(&ds, CartMethod::Gini)).unwrap();
        let (x, _, _) = extract_xy(&ds, "cls", &X_CLASS, NaPolicy::Fail).unwrap();
        let prob = cart_predict_proba(&r, x.view()).unwrap();
        let tol = Tol::coef(300);
        let mut c = Check::new(&case);
        check_cp_table(&mut c, "class", &r.cp_table, tol, true);
        check_frame(&mut c, "class", &r, tol);
        check_importance(&mut c, "class_importance", &r, tol);
        c.vector("class_pred", &r.predictions, Tol::exact())
            .vector("class_prob3", &prob.column(0).to_vec(), tol)
            .vector("class_prob17", &prob.column(1).to_vec(), tol)
            .scalar("class_root_risk", r.root_risk, Tol::exact())
            .scalar("class_min_xerror_cp", r.min_xerror_cp().unwrap(), tol)
            .scalar("class_one_se_cp", r.one_se_cp().unwrap(), tol)
            .finish();
        assert_eq!(r.classes, vec![3.0, 17.0]);
        assert_eq!(r.class_labels, None); // numeric target: T2-WP17 semantics
        assert_eq!((r.n_nodes, r.n_terminal, r.depth), (17, 9, 6));
    }

    #[test]
    fn test_validate_cart_info_split() {
        let (case, ds) = crate::test_support::load_case("cart_cp_class");
        let r = run_cart(&ds, "cls", &X_CLASS, &case_config(&ds, CartMethod::Entropy)).unwrap();
        let tol = Tol::coef(300);
        let n: Vec<f64> = r
            .nodes_preorder()
            .iter()
            .map(|node| node.n as f64)
            .collect();
        let mut c = Check::new(&case);
        check_cp_table(&mut c, "info", &r.cp_table, tol, true);
        check_importance(&mut c, "info_importance", &r, tol);
        c.vector("info_frame_n", &n, Tol::exact()).finish();
    }

    #[test]
    fn test_validate_cart_class_ties() {
        // ML-69. Before: class counts in a std HashMap, so the tied 4/7 leaf predicted 4 or 7 and class_probs came
        // in a random order; the left child was the `x <= t` side (frame n 30, 10, 20; R 30, 20, 10).
        let (case, ds) = crate::test_support::load_case("cart_class_ties");
        let mut config = CartConfig::default();
        config.method = CartMethod::Gini;
        config.cp = 0.001;
        config.min_split = 2;
        config.min_bucket = 1;
        config.max_depth = 1;
        config.xval = 0;
        config.max_surrogate = 0;
        config.use_surrogate = false;
        let r = run_cart(&ds, "cls", &["x", "z"], &config).unwrap();
        let (x, _, _) = extract_xy(&ds, "cls", &["x", "z"], NaPolicy::Fail).unwrap();
        let prob = cart_predict_proba(&r, x.view()).unwrap();
        let mut root_config = config.clone();
        root_config.min_split = 40; // rpart.control(minsplit = 40, xval = 0): cp 0.01, maxdepth 30
        root_config.min_bucket = 13;
        root_config.cp = 0.01;
        root_config.max_depth = 30;
        let root = run_cart(&ds, "cls", &["x", "z"], &root_config).unwrap();
        let tol = Tol::coef(30);
        let mut c = Check::new(&case);
        check_frame(&mut c, "ties", &r, tol);
        c.vector("ties_pred", &r.predictions, Tol::exact())
            .vector("ties_prob_row1", &prob.row(0).to_vec(), tol)
            .vector("ties_prob_row30", &prob.row(29).to_vec(), tol)
            .scalar("ties_root_yval", root.root.yval, Tol::exact())
            .scalar("ties_root_dev", root.root.dev, Tol::exact())
            .vector(
                "ties_root_cp",
                &root.cp_table.iter().map(|row| row.cp).collect::<Vec<_>>(),
                tol,
            )
            .vector(
                "ties_root_rel_error",
                &root
                    .cp_table
                    .iter()
                    .map(|row| row.rel_error)
                    .collect::<Vec<_>>(),
                tol,
            )
            .finish();
        assert_eq!(r.classes, vec![4.0, 7.0, 9.0]);
        assert!(root.cp_table[0].xerror.is_none()); // rpart prints 0 placeholders here: no split, no cross-validation
        let key = |t: &CartResult| -> Vec<(u64, Vec<u64>)> {
            t.nodes_preorder()
                .iter()
                .map(|nd| {
                    (
                        nd.yval.to_bits(),
                        nd.class_probs
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|p| p.to_bits())
                            .collect(),
                    )
                })
                .collect()
        };
        let first = key(&r);
        for _ in 0..25 {
            assert_eq!(
                key(&run_cart(&ds, "cls", &["x", "z"], &config).unwrap()),
                first
            );
        }
    }

    #[test]
    fn test_validate_cart_class_codes_do_not_change_the_tree() {
        // ML-70 (R stopifnot in cart_cp_class). Before: (3, 17) gave 3 nodes and (0, 1) 43 nodes.
        let (case, ds) = crate::test_support::load_case("cart_cp_class");
        let (x, y, _) = extract_xy(&ds, "cls", &X_CLASS, NaPolicy::Fail).unwrap();
        let config = case_config(&ds, CartMethod::Gini);
        let fit = |a: f64, b: f64| {
            cart(
                x.view(),
                y.mapv(|v| if v == 3.0 { a } else { b }).view(),
                &config,
            )
            .unwrap()
        };
        let table = |r: &CartResult| -> Vec<[f64; 4]> {
            r.cp_table
                .iter()
                .map(|row| {
                    [
                        row.cp,
                        row.rel_error,
                        row.xerror.unwrap(),
                        row.xstd.unwrap(),
                    ]
                })
                .collect()
        };
        let sizes =
            |r: &CartResult| -> Vec<usize> { r.nodes_preorder().iter().map(|nd| nd.n).collect() };
        let base = fit(3.0, 17.0);
        for (a, b) in [(0.0, 1.0), (-5.0, 1e6), (0.1, 0.25)] {
            let other = fit(a, b);
            assert_eq!(table(&other), table(&base), "codes {a}/{b}");
            assert_eq!(sizes(&other), sizes(&base), "codes {a}/{b}");
            assert_eq!(other.classes, vec![a, b]);
            let mut c = Check::new(&case);
            check_cp_table(&mut c, "class", &other.cp_table, Tol::coef(300), true);
            c.finish();
        }
        let reversed = fit(20.0, 1.0); // class order reversed: rounding may differ (R: 1e-14)
        let tol = Tol::new(1e-12, 1e-15);
        for (row, base_row) in table(&reversed).iter().zip(table(&base).iter()) {
            for k in 0..4 {
                assert!(tol.allows(row[k], base_row[k]), "{row:?} vs {base_row:?}");
            }
        }
    }

    #[test]
    fn test_cart_exact_tie_takes_the_first_candidate() {
        // (a) A strictly increasing map of a predictor gives the same partitions and equal improvements:
        // the first feature wins at every split, for anova and gini.
        let n = 80;
        let x0: Vec<f64> = (0..n)
            .map(|i| ((i * 37) % 80) as f64 / 8.0 + 0.003 * ((i * 11) % 7) as f64)
            .collect();
        let x = Array2::from_shape_fn(
            (n, 2),
            |(i, j)| if j == 0 { x0[i] } else { 3.0 * x0[i] + 1.0 },
        );
        let y_reg = Array1::from_shape_fn(n, |i| {
            f64::from(u8::from(x0[i] > 4.0)) * 2.0 + 0.1 * ((i * 13) % 5) as f64
        });
        let y_cls = Array1::from_shape_fn(n, |i| {
            if x0[i] + 0.4 * ((i * 17) % 5) as f64 > 5.0 {
                1.0
            } else {
                0.0
            }
        });
        for (method, y) in [(CartMethod::Anova, &y_reg), (CartMethod::Gini, &y_cls)] {
            let mut config = CartConfig::default();
            config.method = method;
            config.cp = 0.0;
            config.min_split = 4;
            config.min_bucket = 2;
            config.xval = 0;
            let r = cart(x.view(), y.view(), &config).unwrap();
            assert!(r.n_nodes > 1, "{method:?}");
            for node in r.nodes_preorder() {
                if let Some(split) = &node.split {
                    assert_eq!(split.feature, 0, "{method:?}, node {}", node.id);
                }
            }
        }
        // (b) Gini on classes 0, 1, 1, 0 in x order: the positions after the first and third row give the
        // same count multiset, so the criterion is exactly equal and the first position wins, whichever
        // way the classes are coded.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        for (a, b) in [(0.0, 1.0), (1.0, 0.0), (5.0, 2.0), (0.1, 0.25)] {
            let y = array![a, b, b, a];
            let mut config = CartConfig::default();
            config.method = CartMethod::Gini;
            config.cp = 0.0;
            config.min_split = 2;
            config.min_bucket = 1;
            config.max_depth = 1;
            config.xval = 0;
            let r = cart(x.view(), y.view(), &config).unwrap();
            assert_eq!(r.root.split.as_ref().unwrap().threshold, 1.5, "{a}/{b}");
        }
    }

    #[test]
    fn test_cart_rundown_stretches_equal_the_direct_loop() {
        // The difference arrays of the complexity table and of the cross-validation rundown give what
        // rpart's direct loops give: row by row and node by node, on a tree with many table rows.
        let n = 200;
        let (x, y) = {
            let mut s = 5u64;
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            };
            let x: Vec<f64> = (0..2 * n).map(|_| 10.0 * next()).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| 2.0 * f64::from(u8::from(x[i] > 5.0)) + 0.3 * x[n + i] + next() - 0.5)
                .collect();
            (x, y)
        };
        let mut sorts = vec![0usize; 2 * n];
        for j in 0..2 {
            let order = &mut sorts[j * n..(j + 1) * n];
            for (i, slot) in order.iter_mut().enumerate() {
                *slot = i;
            }
            let col = &x[j * n..(j + 1) * n];
            order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
        }
        let mut grower = Grower {
            n,
            p: 2,
            x: &x,
            y: &y,
            codes: &[],
            n_classes: 0,
            criterion: Criterion::Anova,
            min_split: 2,
            edge: 1,
            too_small: 2,
            max_node: (1 << 30) - 1,
            alpha: 0.0,
            iscale: 0.0,
            sorts,
            tmp: vec![0; n],
            goes_left: vec![false; n],
            left_counts: Vec::new(),
            right_counts: Vec::new(),
        };
        let mut tree = grower.fit(n).unwrap();
        let root_complexity = tree.complexity;
        let mut cps = vec![root_complexity];
        make_cp_list(&mut tree, root_complexity, 0.0, &mut cps);
        cps.sort_unstable_by(|a, b| b.total_cmp(a));
        cps.dedup();
        let rows = cps.len();
        assert!(rows > 50, "{rows}");

        // the rundown
        let mut cpx = vec![10.0 * cps[0]];
        cpx.extend(cps.windows(2).map(|w| (w[0] * w[1]).sqrt()));
        let (mut d_risk, mut d_sq) = (vec![0.0; rows + 1], vec![0.0; rows + 1]);
        let (mut risk_direct, mut sq_direct) = (vec![0.0; rows], vec![0.0; rows]);
        fix_cp(&mut tree, root_complexity);
        for row in 0..n {
            grower.rundown(&tree, row, &cpx, &mut d_risk, &mut d_sq);
            let mut node = &tree;
            for i in 0..rows {
                node = grower.descend(node, row, cpx[i]);
                let err = grower.error(node, row);
                risk_direct[i] += err;
                sq_direct[i] += err * err;
            }
        }
        let (r, q) = (cumulate(&d_risk), cumulate(&d_sq));
        for i in 0..rows {
            assert!(
                (r[i] - risk_direct[i]).abs() <= 1e-9 * (1.0 + risk_direct[i]),
                "row {i}"
            );
            assert!(
                (q[i] - sq_direct[i]).abs() <= 1e-9 * (1.0 + sq_direct[i]),
                "row {i}"
            );
        }
    }

    /// Noisy regression data from a fixed LCG (no external RNG): y = 2 (x0 > 5) + 0.3 x1 + noise.
    fn noisy(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = 10.0 * next();
            }
            y[i] = 2.0 * f64::from(u8::from(x[[i, 0]] > 5.0)) + 0.3 * x[[i, 1]] + (next() - 0.5);
        }
        (x, y)
    }

    #[test]
    fn test_cart_rejects_bad_arguments() {
        // ML-105: before, NaN went into partial_cmp(..).unwrap_or(Equal) sorts silently, and max_depth = 0 (unlimited)
        // let node ids overflow on deep chains.
        let (x, y) = noisy(60, 3);
        let ok = CartConfig::default();
        assert!(cart(x.view(), y.view(), &ok).is_ok());
        for bad in [f64::NAN, f64::INFINITY] {
            let mut xb = x.clone();
            xb[[4, 1]] = bad;
            let e = cart(xb.view(), y.view(), &ok).unwrap_err().to_string();
            assert!(
                e.contains("feature column 1") && e.contains("non-finite"),
                "{e}"
            );
            let mut yb = y.clone();
            yb[7] = bad;
            assert!(
                cart(x.view(), yb.view(), &ok)
                    .unwrap_err()
                    .to_string()
                    .contains("target")
            );
        }
        let cfg = |f: &dyn Fn(&mut CartConfig)| {
            let mut c = CartConfig::default();
            f(&mut c);
            c
        };
        let err = |c: CartConfig| cart(x.view(), y.view(), &c).unwrap_err().to_string();
        assert!(err(cfg(&|c| c.max_depth = 31)).contains("max_depth must be at most 30"));
        assert!(err(cfg(&|c| c.cp = f64::NAN)).contains("cp"));
        assert!(err(cfg(&|c| c.folds = Some(vec![1; 59]))).contains("60 rows"));
        assert!(err(cfg(&|c| c.folds = Some(vec![1; 60]))).contains("at least 2 folds"));
        assert!(
            err(cfg(
                &|c| c.folds = Some((0..60).map(|i| if i < 30 { 1 } else { 3 }).collect())
            ))
            .contains("label 2")
        );
        assert!(
            err(cfg(&|c| c.folds = Some((0..60).map(|i| i % 2).collect()))).contains("label 0")
        ); // 0 is not a label
        // min_bucket = 0 acts as 1 (rpart's C loop would read past the node otherwise)
        let a = cart(
            x.view(),
            y.view(),
            &cfg(&|c| {
                c.min_bucket = 0;
                c.min_split = 2;
                c.xval = 0
            }),
        )
        .unwrap();
        let b = cart(
            x.view(),
            y.view(),
            &cfg(&|c| {
                c.min_bucket = 1;
                c.min_split = 2;
                c.xval = 0
            }),
        )
        .unwrap();
        assert_eq!(
            a.cp_table
                .iter()
                .map(|r| r.cp.to_bits())
                .collect::<Vec<_>>(),
            b.cp_table
                .iter()
                .map(|r| r.cp.to_bits())
                .collect::<Vec<_>>()
        );
        // prediction input
        assert!(
            cart_predict(&a, x.slice(ndarray::s![.., 0..2]))
                .unwrap_err()
                .to_string()
                .contains("3")
        );
        let mut xn = x.clone();
        xn[[0, 0]] = f64::NAN;
        assert!(cart_predict(&a, xn.view()).is_err());
        assert!(
            cart_predict_proba(&a, x.view())
                .unwrap_err()
                .to_string()
                .contains("classification")
        );
        // max_depth = 0 means 30 (rpart's limit): a 400-row chain stays within node ids and recursion
        let xc = Array2::from_shape_fn((400, 1), |(i, _)| i as f64);
        let yc = Array1::from_shape_fn(400, |i| (i * i) as f64 + 0.1 * ((i * 7) % 3) as f64);
        let deep = cart(
            xc.view(),
            yc.view(),
            &cfg(&|c| {
                c.max_depth = 0;
                c.min_split = 2;
                c.min_bucket = 1;
                c.cp = 0.0;
                c.xval = 0
            }),
        )
        .unwrap();
        assert!(deep.depth <= 30);
    }

    #[test]
    fn test_cart_random_folds_are_seeded_and_balanced() {
        // Before: no cross-validation (cv_error None, xerror = 1.1 rel_error for every seed).
        let (x, y) = noisy(300, 11);
        let run = |seed: Option<u64>| {
            let mut c = CartConfig::default();
            c.seed = seed;
            c.xval = 10;
            cart(x.view(), y.view(), &c).unwrap()
        };
        let bits = |r: &CartResult| {
            r.cp_table
                .iter()
                .map(|row| {
                    (
                        row.cp.to_bits(),
                        row.rel_error.to_bits(),
                        row.xerror.unwrap().to_bits(),
                        row.xstd.unwrap().to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let (a, b, c, d) = (run(Some(7)), run(Some(7)), run(Some(8)), run(None));
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(bits(&d), bits(&run(Some(42)))); // None is the documented seed 42
        assert_eq!(a.n_folds, 10);
        assert!(
            a.cp_table
                .iter()
                .zip(&c.cp_table)
                .all(|(p, q)| p.cp == q.cp && p.rel_error == q.rel_error)
        );
        assert!(
            a.cp_table
                .iter()
                .zip(&c.cp_table)
                .any(|(p, q)| p.xerror != q.xerror)
        );
        let f = random_folds(300, 10, 7);
        for label in 1..=10 {
            assert_eq!(f.iter().filter(|&&v| v == label).count(), 30);
        }
        // xval > n (rpart allows it: the extra folds are empty)
        let (xs, ys) = noisy(8, 2);
        let mut small = CartConfig::default();
        small.min_split = 2;
        small.min_bucket = 1;
        small.xval = 10;
        assert!(cart(xs.view(), ys.view(), &small).is_ok());
    }

    #[test]
    fn test_cart_constant_target_is_a_single_leaf() {
        // Before: a fabricated cp row {cp 1, rel_error 1, xerror 1, xstd 0}; rpart gives NaN here.
        let (x, _) = noisy(40, 5);
        let y = Array1::from_elem(40, 2.5);
        for method in [CartMethod::Anova, CartMethod::Gini] {
            let mut c = CartConfig::default();
            c.method = method;
            let r = cart(x.view(), y.view(), &c).unwrap();
            assert_eq!((r.n_nodes, r.n_terminal, r.root_risk), (1, 1, 0.0));
            assert!(r.cp_table.is_empty() && r.cv_error.is_none());
            assert!(
                r.warnings.iter().any(|w| w.contains("constant")),
                "{:?}",
                r.warnings
            );
            assert!(r.predictions.iter().all(|&p| p == 2.5));
            assert!(r.to_string().contains("Warning: "));
        }
    }

    #[test]
    fn test_cart_unsplittable_root_has_one_cp_row() {
        // rpart: CP = cp (alpha / root risk), nsplit 0, rel error 1; R prints xerror = xstd = 0 placeholders
        // because no cross-validation runs without a split; here they are None.
        let (x, y) = noisy(30, 9);
        let mut c = CartConfig::default();
        c.min_split = 40;
        let r = cart(x.view(), y.view(), &c).unwrap();
        assert_eq!(r.cp_table.len(), 1);
        // risk x (1 / risk) is 1 only up to an ulp (49 x (1/49) = 0.9999999999999999), as in rpart
        assert!(
            (r.cp_table[0].cp - 0.01).abs() < 1e-15
                && (r.cp_table[0].rel_error - 1.0).abs() < 1e-15
        );
        assert!(r.cp_table[0].xerror.is_none() && r.n_folds == 0);
    }

    #[test]
    fn test_cart_method_parses_aliases_and_rejects_unknown() {
        CartMethod::choices().validate().unwrap();
        for (s, m) in [
            ("anova", CartMethod::Anova),
            ("MSE", CartMethod::Anova),
            ("class", CartMethod::Gini),
            ("Gini", CartMethod::Gini),
            ("information", CartMethod::Entropy),
            ("entropy", CartMethod::Entropy),
        ] {
            assert_eq!(s.parse::<CartMethod>().unwrap(), m, "{s}");
        }
        let e = "poisson".parse::<CartMethod>().unwrap_err().to_string();
        assert!(
            e.contains("anova") && e.contains("gini") && e.contains("information"),
            "{e}"
        );
        assert_eq!(CartMethod::Entropy.to_string(), "information");
    }

    #[test]
    fn test_validate_cart_prune_recomputes() {
        // ML-72. Before: prune kept the unpruned predictions (25 distinct values), importance and cp table (24 rows).
        let (case, ds) = crate::test_support::load_case("cart_cp_anova");
        let r = run_cart(&ds, "y", &X_ANOVA, &case_config(&ds, CartMethod::Anova)).unwrap();
        let p = cart_prune(&r, 0.02).unwrap();
        let tol = Tol::coef(300);
        let mut c = Check::new(&case);
        check_cp_table(&mut c, "anova_pruned_002", &p.cp_table, tol, true);
        check_frame(&mut c, "anova_pruned_002", &p, tol);
        check_importance(&mut c, "anova_pruned_002_importance", &p, tol);
        c.vector("anova_pruned_002_pred", &p.predictions, tol)
            .vector("anova_pruned_002_pred_head", &p.predictions[..10], tol)
            .finish();
        assert_eq!(p.variable_importance[3], 0.0); // x4's splits are all below cp
        assert_eq!((p.n_nodes, p.n_terminal, p.depth), (11, 6, 3));
        assert_eq!(p.cv_error.as_ref().map(Vec::len), Some(5));
        let (x, _, _) = extract_xy(&ds, "y", &X_ANOVA, NaPolicy::Fail).unwrap();
        assert_eq!(cart_predict(&p, x.view()).unwrap(), p.predictions);
        // nothing to collapse: rpart returns the tree unchanged, cp table included
        let same = cart_prune(&r, 0.0005).unwrap();
        assert_eq!(same.n_nodes, r.n_nodes);
        assert_eq!(
            same.cp_table
                .iter()
                .map(|row| row.cp.to_bits())
                .collect::<Vec<_>>(),
            r.cp_table
                .iter()
                .map(|row| row.cp.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(cart_prune(&r, f64::NAN).is_err());
    }

    #[test]
    fn test_validate_cart_prune_xor() {
        // ML-72: the root's own improvement is 0.000346 of the root deviance, but its weakest-link complexity
        // (whole subtree) is 0.33. Before: the cp < 0.001 test made the fit a single leaf.
        let (case, ds) = crate::test_support::load_case("cart_prune_xor");
        let mut config = CartConfig::default();
        config.cp = 0.001;
        config.min_split = 20;
        config.min_bucket = 7;
        config.xval = 0;
        config.max_surrogate = 0;
        config.use_surrogate = false;
        let r = run_cart(&ds, "y", &["x0", "x1"], &config).unwrap();
        let p5 = cart_prune(&r, 0.05).unwrap();
        let p50 = cart_prune(&r, 0.5).unwrap();
        let tol = Tol::coef(40);
        let mut c = Check::new(&case);
        check_cp_table(&mut c, "xor", &r.cp_table, tol, false);
        check_frame(&mut c, "xor", &r, tol);
        check_importance(&mut c, "xor_importance", &r, tol);
        check_cp_table(&mut c, "xor_pruned_005", &p5.cp_table, tol, false);
        check_frame(&mut c, "xor_pruned_005", &p5, tol);
        check_importance(&mut c, "xor_pruned_005_importance", &p5, tol);
        check_cp_table(&mut c, "xor_pruned_05", &p50.cp_table, tol, false);
        c.vector("xor_pruned_005_pred", &p5.predictions, tol)
            .scalar("xor_pruned_005_leaves", p5.n_terminal as f64, Tol::exact())
            .scalar("xor_pruned_05_leaves", p50.n_terminal as f64, Tol::exact())
            .vector("xor_pruned_05_pred", &p50.predictions, tol)
            .finish();
        assert!(p50.variable_importance.iter().all(|&v| v == 0.0));
    }
}
