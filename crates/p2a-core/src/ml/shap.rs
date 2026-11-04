//! SHAP (SHapley Additive exPlanations) for model interpretation.
//!
//! SHAP values are the Shapley values of a game `v(S)` defined by a model, a point `x` and a
//! coalition `S` of features: `phi_i = sum_{S not containing i} |S|! (M - |S| - 1)! / M! (v(S + i)
//! - v(S))`. They add up: `v(empty) + sum_i phi_i = f(x)` (the base value plus the SHAP values is
//! the prediction), which every result checks on the rows it explains
//! ([`ShapResult::additivity_check_passed`]).
//!
//! ## Methods and games
//!
//! - **TreeSHAP** ([`ShapMethod::Tree`]; [`shap_values_model`], [`shap_tree_ensemble`]): the
//!   path-dependent game `v(S) = E[f(x) | x_S]` of a tree, which follows `x` at splits on
//!   features in `S` and weights both children by their training cover otherwise (Lundberg et
//!   al. 2020, Algorithm 1; shap's `feature_perturbation = "tree_path_dependent"`). Algorithm 2
//!   computes its Shapley values exactly in O(L D^2) per tree and row (L leaves, depth D); an
//!   ensemble's values are the tree average and its base value is the mean over trees of
//!   `sum_leaf value * cover / root cover`. With `compute_interactions` it also returns the
//!   Shapley interaction index (eqs. 8-9), one symmetric M x M matrix per row whose rows sum to
//!   the SHAP values.
//! - **Kernel SHAP** ([`ShapMethod::Kernel`]; [`kernel_shap`], and for a forest
//!   [`shap_values_model_with`]): the interventional game `v(S) = mean_b f(x_S, b_notS)` over a
//!   background sample, for any model (Lundberg & Lee 2017; Janzing et al. 2020), by
//!   shap.KernelExplainer's constrained weighted regression. `n_samples` is the coalition budget:
//!   every coalition is evaluated, and the values are exact, when `2^M - 2 <= n_samples` (the
//!   default `2 M + 2048` does so for M <= 11); otherwise shap's sampled design with a seeded,
//!   per-row ChaCha8 stream gives a Monte Carlo estimate that still adds up exactly. One feature
//!   gives `f(x) - E f`.
//! - **Perturbations** ([`FeaturePerturbation`]): each method computes its own game, and asking
//!   for the other one is an error that names the route. Observational (true conditional) SHAP
//!   is not implemented: it needs a model of the feature distribution.
//!
//! ## Example
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use p2a_core::ml::{ShapConfig, random_forest_with_trees, shap_values_model};
//!
//! // 30 rows: y = 2 x0 + x1 x2 plus a deterministic wiggle
//! let x = Array2::from_shape_fn((30, 3), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 10.0);
//! let y = Array1::from_shape_fn(30, |i| {
//!     2.0 * x[[i, 0]] + x[[i, 1]] * x[[i, 2]] + 0.05 * ((i * 13) % 5) as f64
//! });
//! let model = random_forest_with_trees(
//!     x.view(), y.view(), Some(10), Some(4), Some(2), Some("all"), Some(42), None,
//! )
//! .unwrap();
//! let shap = shap_values_model(&model, x.view(), &ShapConfig::default()).unwrap();
//! assert!(shap.exact);
//! assert_eq!(shap.additivity_check_passed, Some(true));
//! ```
//!
//! # References
//!
//! - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions".
//!   Advances in Neural Information Processing Systems 30 (NeurIPS 2017).
//!   <https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
//!
//! - Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018). "Consistent Individualized Feature
//!   Attribution for Tree Ensembles". arXiv:1802.03888.
//!   <https://arxiv.org/abs/1802.03888>
//!
//! - Lundberg, S. M., Erion, G., Chen, H., et al. (2020). "From local explanations to global
//!   understanding with explainable AI for trees". Nature Machine Intelligence 2, 56-67.
//!
//! - Janzing, D., Minorics, L., & Blöbaum, P. (2020). "Feature relevance quantification in
//!   explainable AI: A causal problem". AISTATS 2020.
//!
//! - Python `shap` (Lundberg): `TreeExplainer` (`tree_shap_recursive`) and `KernelExplainer`.
//!   <https://github.com/shap/shap>
//!
//! Validation: `validation/ml/shap.md` (exact enumeration written out in R).

use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::convergence::{ResultNotes, notes_text};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default seed of the Kernel SHAP sampling path (R13); `ShapConfig::seed = None` uses it too.
pub const SHAP_DEFAULT_SEED: u64 = 42;
/// Additivity tolerance, relative to `max(1, max_i |f(x_i)|)`.
pub const ADDITIVITY_TOLERANCE: f64 = 1e-8;

/// Which SHAP algorithm (option "method").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapMethod {
    /// TreeSHAP (Lundberg, Erion & Lee 2020, Algorithm 2): exact, tree path-dependent.
    #[default]
    Tree,
    /// Kernel SHAP (Lundberg & Lee 2017; shap.KernelExplainer): interventional, any model.
    Kernel,
}
crate::option_enum!(
    ShapMethod,
    crate::options::Choices {
        option: "method",
        values: &[
            (&["tree", "treeshap", "tree_shap"], ShapMethod::Tree),
            (&["kernel", "kernelshap", "kernel_shap"], ShapMethod::Kernel),
        ],
    }
);

/// The game whose Shapley values are computed (option "feature_perturbation").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePerturbation {
    /// `v(S) = mean_b f(x_S, b_notS)` over a background sample (Kernel SHAP).
    Interventional,
    /// `v(S) = E[f(x) | x_S]` from the trees' training cover (TreeSHAP; shap's
    /// "tree_path_dependent").
    TreePathDependent,
    /// True conditional expectations; not implemented (it needs a model of the feature
    /// distribution), so requesting it is an error.
    Observational,
}
crate::option_enum!(
    FeaturePerturbation,
    crate::options::Choices {
        option: "feature_perturbation",
        values: &[
            (
                &["interventional", "marginal", "independent"],
                FeaturePerturbation::Interventional
            ),
            (
                &["tree_path_dependent", "path_dependent"],
                FeaturePerturbation::TreePathDependent
            ),
            (
                &["observational", "conditional", "correlation_dependent"],
                FeaturePerturbation::Observational
            ),
        ],
    }
);

impl FeaturePerturbation {
    /// The perturbation a method computes: Tree -> TreePathDependent, Kernel -> Interventional.
    pub fn default_for(method: ShapMethod) -> Self {
        match method {
            ShapMethod::Tree => FeaturePerturbation::TreePathDependent,
            ShapMethod::Kernel => FeaturePerturbation::Interventional,
        }
    }
}

/// Configuration for SHAP value computation.
#[derive(Debug, Clone)]
pub struct ShapConfig {
    /// Kernel SHAP coalition budget (shap's `nsamples`; default `2 M + 2048`): every coalition is
    /// evaluated (exact) when `2^M - 2 <= n_samples`, which the default does for M <= 11;
    /// otherwise shap.KernelExplainer's sampled design with `n_samples` coalitions. TreeSHAP does
    /// not sample: a value given to the tree path is reported in `warnings`. Must be >= 1 on
    /// both paths (0 is an error).
    pub n_samples: Option<usize>,

    /// `None`: the method's own perturbation ([`FeaturePerturbation::default_for`]). Anything
    /// else must be the one the method computes, or the call is an error.
    pub feature_perturbation: Option<FeaturePerturbation>,

    /// TreeSHAP interaction values (Lundberg et al. 2020, eqs. 8-9); an error for Kernel SHAP.
    pub compute_interactions: bool,

    /// Seed of the Kernel sampling path (row i uses ChaCha8 stream i); `None` uses
    /// [`SHAP_DEFAULT_SEED`].
    pub seed: Option<u64>,

    /// Check `base + sum(phi) = f(x)` on the explained rows (default true).
    pub check_additivity: bool,
}

impl Default for ShapConfig {
    fn default() -> Self {
        Self {
            n_samples: None,
            feature_perturbation: None,
            compute_interactions: false,
            seed: Some(SHAP_DEFAULT_SEED),
            check_additivity: true,
        }
    }
}

/// Result of SHAP value computation.
#[derive(Debug, Clone, Serialize)]
pub struct ShapResult {
    /// SHAP values matrix (n_obs x n_features): the contribution of each feature to each
    /// explained row's prediction.
    #[serde(skip)]
    pub shap_values: Array2<f64>,

    /// TreeSHAP: the mean over trees of `sum_leaf value * cover / root cover` (`E[f]` under the
    /// trees' training distribution); Kernel SHAP: the mean prediction over the background.
    pub base_value: f64,

    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,

    /// Number of explained rows
    pub n_obs: usize,

    /// Number of features
    pub n_features: usize,

    /// Mean absolute SHAP values per feature (global importance)
    pub feature_importance: Vec<f64>,

    /// Interaction values (TreeSHAP with `compute_interactions`): one M x M matrix per row.
    #[serde(skip)]
    pub interaction_values: Option<Vec<Array2<f64>>>,

    /// Whether `base + sum(phi)` reproduced the predictions; `None` when not checked.
    pub additivity_check_passed: Option<bool>,

    /// `max_i |base + sum_j phi_ij - f(x_i)|`; `None` when not checked.
    pub max_additivity_error: Option<f64>,

    /// The algorithm that ran.
    pub method: ShapMethod,

    /// The game whose Shapley values these are.
    pub feature_perturbation: FeaturePerturbation,

    /// TreeSHAP, or Kernel SHAP with every coalition enumerated (or one feature).
    pub exact: bool,

    /// Kernel SHAP: coalitions per row (0 for one feature); `None` for TreeSHAP.
    pub n_coalitions: Option<usize>,

    /// Kernel SHAP sampling seed actually used; `None` when no coalition was sampled.
    pub seed: Option<u64>,

    /// `f(x_i)` of the explained rows (the additivity target).
    #[serde(skip)]
    pub predictions: Vec<f64>,

    /// Warnings about the computation.
    pub warnings: Vec<String>,
}

impl ResultNotes for ShapResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl ShapResult {
    /// The "Method: ..." line of the Display: which algorithm ran on which game, and whether the
    /// values are exact.
    fn method_line(&self) -> String {
        match self.method {
            ShapMethod::Tree => "Method: TreeSHAP (tree path-dependent; exact)".to_string(),
            ShapMethod::Kernel => {
                let n = self.n_coalitions.unwrap_or(0);
                if self.n_features == 1 {
                    "Method: Kernel SHAP (interventional; one feature: phi = f(x) - E f, exact)"
                        .to_string()
                } else if self.exact {
                    format!("Method: Kernel SHAP (interventional; all {n} coalitions, exact)")
                } else {
                    let total = if self.n_features <= 127 {
                        ((1u128 << self.n_features) - 2).to_string()
                    } else {
                        "2^M - 2".to_string()
                    };
                    format!(
                        "Method: Kernel SHAP (interventional; {n} of {total} coalitions evaluated (complete sizes enumerated, the rest sampled), seed {}; Monte Carlo estimate)",
                        self.seed.unwrap_or(SHAP_DEFAULT_SEED)
                    )
                }
            }
        }
    }
}

impl std::fmt::Display for ShapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Values Summary")?;
        writeln!(f, "===================")?;
        writeln!(f, "{}", self.method_line())?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Features: {}", self.n_features)?;
        writeln!(f, "Base value: {:.6}", self.base_value)?;

        match (self.additivity_check_passed, self.max_additivity_error) {
            (Some(passed), err) => writeln!(
                f,
                "Additivity check: {} (max error: {:.1e})",
                if passed { "PASSED" } else { "FAILED" },
                err.unwrap_or(f64::NAN)
            )?,
            (None, _) => writeln!(f, "Additivity check: not run")?,
        }

        writeln!(f)?;
        writeln!(f, "Feature Importance (mean |SHAP|):")?;

        // Sort features by importance (finite: inputs and predictions are validated)
        let mut indexed: Vec<(usize, f64)> = self
            .feature_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (i, importance) in indexed.iter().take(15) {
            writeln!(f, "  {:20} {:.6}", self.feature_name(*i), importance)?;
        }

        if self.n_features > 15 {
            writeln!(f, "  ... ({} more features)", self.n_features - 15)?;
        }

        // Show SHAP value statistics
        writeln!(f)?;
        writeln!(f, "SHAP Value Statistics per Feature:")?;
        writeln!(
            f,
            "  {:20} {:>10} {:>10} {:>10}",
            "Feature", "Min", "Mean", "Max"
        )?;

        for (i, _) in indexed.iter().take(10) {
            let col = self.shap_values.column(*i);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = col.sum() / col.len() as f64;
            writeln!(
                f,
                "  {:20} {:10.4} {:10.4} {:10.4}",
                self.feature_name(*i),
                min,
                mean,
                max
            )?;
        }

        if self.interaction_values.is_some() {
            writeln!(f)?;
            writeln!(
                f,
                "Interaction values computed: Yes (TreeSHAP, {} x {} x {})",
                self.n_obs, self.n_features, self.n_features
            )?;
        }

        write!(f, "{}", notes_text(self))
    }
}

impl ShapResult {
    fn feature_name(&self, i: usize) -> String {
        match &self.feature_names {
            Some(names) => names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Feature_{}", i)),
            None => format!("Feature_{}", i),
        }
    }
}

/// Summary of SHAP values aggregated across samples.
#[derive(Debug, Clone, Serialize)]
pub struct ShapSummary {
    /// Feature names
    pub feature_names: Vec<String>,

    /// Mean absolute SHAP value per feature
    pub mean_abs_shap: Vec<f64>,

    /// Mean SHAP value per feature
    pub mean_shap: Vec<f64>,

    /// Standard deviation of SHAP values per feature
    pub std_shap: Vec<f64>,

    /// Rank of each feature by importance
    pub importance_rank: Vec<usize>,

    /// Total number of observations
    pub n_obs: usize,
}

impl std::fmt::Display for ShapSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Feature Importance Summary")?;
        writeln!(f, "================================")?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:5} {:20} {:>12} {:>12} {:>12}",
            "Rank", "Feature", "Mean |SHAP|", "Mean SHAP", "Std SHAP"
        )?;
        writeln!(f, "{:-<65}", "")?;

        // Create sorted indices by importance
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();
        indices.sort_by(|&a, &b| self.mean_abs_shap[b].total_cmp(&self.mean_abs_shap[a]));

        for (rank, &i) in indices.iter().enumerate().take(20) {
            writeln!(
                f,
                "{:5} {:20} {:12.6} {:12.6} {:12.6}",
                rank + 1,
                self.feature_names[i],
                self.mean_abs_shap[i],
                self.mean_shap[i],
                self.std_shap[i]
            )?;
        }

        if self.feature_names.len() > 20 {
            writeln!(f, "... {} more features", self.feature_names.len() - 20)?;
        }

        Ok(())
    }
}

// ============================================================================
// Input checks and the additivity check (shared by both methods)
// ============================================================================

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// The rows to explain: at least one row and one column, every value finite, names matching.
fn validate_explain_inputs(x: ArrayView2<f64>, feature_names: Option<&[String]>) -> EconResult<()> {
    if x.nrows() == 0 {
        return Err(spec("SHAP: x has no rows to explain".into()));
    }
    if x.ncols() == 0 {
        return Err(spec("SHAP needs at least one feature column".into()));
    }
    if let Some(((i, j), _)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(spec(format!(
            "SHAP: x has a non-finite value at row {i}, column {j}"
        )));
    }
    if let Some(names) = feature_names
        && names.len() != x.ncols()
    {
        return Err(spec(format!(
            "SHAP: {} feature names for {} features",
            names.len(),
            x.ncols()
        )));
    }
    Ok(())
}

/// The Kernel SHAP background: `m` columns, at least one row, every value finite.
fn validate_background(bg: ArrayView2<f64>, m: usize) -> EconResult<()> {
    if bg.ncols() != m {
        return Err(spec(format!(
            "Kernel SHAP: the background has {} columns but x has {m}",
            bg.ncols()
        )));
    }
    if bg.nrows() == 0 {
        return Err(spec("Kernel SHAP: the background has no rows".into()));
    }
    if let Some(((i, j), _)) = bg.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(spec(format!(
            "SHAP: the background has a non-finite value at row {i}, column {j}"
        )));
    }
    Ok(())
}

/// A `predict_fn` output: one finite value per row.
fn check_predictions(p: &[f64], rows: usize) -> EconResult<()> {
    if p.len() != rows {
        return Err(spec(format!(
            "Kernel SHAP: predict_fn returned {} values for {rows} rows",
            p.len()
        )));
    }
    if p.iter().any(|v| !v.is_finite()) {
        return Err(spec(
            "Kernel SHAP: predict_fn returned a non-finite value".into(),
        ));
    }
    Ok(())
}

/// `max_i |base + sum_j phi_ij - pred_i|` against `ADDITIVITY_TOLERANCE * max(1, max_i |pred_i|)`:
/// (passed, max error, failure warning); `(None, None, None)` when the check is off. NaN fails.
fn additivity(
    base: f64,
    phi: &Array2<f64>,
    pred: &[f64],
    check: bool,
) -> (Option<bool>, Option<f64>, Option<String>) {
    if !check {
        return (None, None, None);
    }
    let mut max_err = 0.0f64;
    let mut any_nan = false;
    let mut scale = 1.0f64;
    for (row, &p) in phi.outer_iter().zip(pred) {
        let e = (base + row.sum() - p).abs();
        if e.is_nan() {
            any_nan = true;
        } else if e > max_err {
            max_err = e;
        }
        scale = scale.max(p.abs());
    }
    if any_nan {
        max_err = f64::NAN;
    }
    let tol = ADDITIVITY_TOLERANCE * scale;
    let passed = !any_nan && max_err <= tol;
    let warning = (!passed).then(|| {
        format!(
            "SHAP values do not add up to the predictions (max error {max_err:.1e} > tolerance {tol:.1e}); please report this"
        )
    });
    (Some(passed), Some(max_err), warning)
}

fn mean_abs_columns(phi: &Array2<f64>) -> Vec<f64> {
    phi.columns()
        .into_iter()
        .map(|c| c.mapv(f64::abs).mean().unwrap_or(0.0))
        .collect()
}

// ============================================================================
// TreeSHAP Implementation
// ============================================================================

/// Internal tree node representation for TreeSHAP.
#[derive(Debug, Clone)]
pub(crate) struct TreeShapNode {
    /// Feature index for split (-1 for leaf)
    pub feature: i32,
    /// Threshold for split
    pub threshold: f64,
    /// Left child index (-1 for leaf)
    pub left: i32,
    /// Right child index (-1 for leaf)
    pub right: i32,
    /// Value at node (for leaves) or mean value for internal nodes
    pub value: f64,
    /// Number of samples that reach this node
    pub n_samples: usize,
}

/// Node table checks before any traversal: preorder ids with children after their parent and
/// every non-root node reached once, leaves without children, split features below
/// `n_features`, leaf covers >= 1 and every internal cover the sum of its children's, finite leaf
/// values and non-NaN thresholds. Returns the depth of the deepest leaf (root depth 0). `tree`
/// numbers the tree in messages.
fn validate_nodes(nodes: &[TreeShapNode], n_features: usize, tree: usize) -> EconResult<usize> {
    if nodes.is_empty() {
        return Err(spec(format!("TreeSHAP: tree {tree} is not fitted")));
    }
    let len = nodes.len();
    let bad = |k: usize, what: String| spec(format!("TreeSHAP: tree {tree}, node {k}: {what}"));
    let mut depth = crate::resource::try_vec(len, 0usize, "TreeSHAP node depths")?;
    let mut parents = crate::resource::try_vec(len, 0u8, "TreeSHAP node parents")?;
    let mut max_depth = 0usize;
    for (k, n) in nodes.iter().enumerate() {
        if k > 0 && parents[k] != 1 {
            return Err(bad(
                k,
                "the node is not reached exactly once from the root".into(),
            ));
        }
        if n.feature < 0 {
            if n.left != -1 || n.right != -1 {
                return Err(bad(k, "a leaf has children".into()));
            }
            if n.n_samples == 0 {
                return Err(bad(k, "leaf cover is 0".into()));
            }
            if !n.value.is_finite() {
                return Err(bad(k, "leaf value is not finite".into()));
            }
            max_depth = max_depth.max(depth[k]);
            continue;
        }
        let f = n.feature as usize;
        if f >= n_features {
            return Err(spec(format!(
                "TreeSHAP: tree {tree} splits on feature {f} but x has {n_features} columns"
            )));
        }
        if n.threshold.is_nan() {
            return Err(bad(k, "split threshold is NaN".into()));
        }
        let child = |c: i32| (c > k as i32 && (c as usize) < len).then_some(c as usize);
        let (Some(l), Some(r)) = (child(n.left), child(n.right)) else {
            return Err(bad(
                k,
                format!(
                    "children {} and {} must follow their parent in preorder",
                    n.left, n.right
                ),
            ));
        };
        if l == r {
            return Err(bad(k, "both children are the same node".into()));
        }
        let (cl, cr) = (nodes[l].n_samples, nodes[r].n_samples);
        if cl.checked_add(cr) != Some(n.n_samples) {
            return Err(bad(
                k,
                format!(
                    "cover {} is not the sum of its children's ({cl} + {cr})",
                    n.n_samples
                ),
            ));
        }
        for c in [l, r] {
            parents[c] = parents[c].saturating_add(1);
            depth[c] = depth[k] + 1;
        }
    }
    Ok(max_depth)
}

/// The flat node table of a fitted tree (preorder; `x[feature] <= threshold` goes left). Each
/// internal `n_samples` is the sum of its children's and its `value` the cover-weighted mean of
/// theirs. An unfitted tree gives an empty table, which [`validate_nodes`] reports.
fn extract_tree_structure(tree: &super::trees::DecisionTree) -> Vec<TreeShapNode> {
    let mut nodes = Vec::new();
    if let Some(root) = tree.root() {
        flatten(root, &mut nodes);
    }
    nodes
}

/// Appends `node`'s subtree in preorder and returns the index of `node`. The recursion depth is
/// the tree's depth, which the fitting recursion has already reached.
fn flatten(node: &super::trees::TreeNode, out: &mut Vec<TreeShapNode>) -> usize {
    use super::trees::TreeNode;
    let k = out.len();
    match node {
        TreeNode::Leaf {
            value, n_samples, ..
        } => out.push(TreeShapNode {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            value: *value,
            n_samples: *n_samples,
        }),
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
            ..
        } => {
            out.push(TreeShapNode {
                feature: *feature_index as i32,
                threshold: *threshold,
                left: -1,
                right: -1,
                value: 0.0,
                n_samples: 0,
            });
            let l = flatten(left, out);
            let r = flatten(right, out);
            let (cl, cr) = (out[l].n_samples, out[r].n_samples);
            let total = cl + cr;
            let value = if total > 0 {
                (out[l].value * cl as f64 + out[r].value * cr as f64) / total as f64
            } else {
                0.0
            };
            let n = &mut out[k];
            n.left = l as i32;
            n.right = r as i32;
            n.n_samples = total;
            n.value = value;
        }
    }
    k
}

/// The leaf a point reaches (`x[feature] <= threshold` goes left, as `DecisionTree::predict_one`).
fn predict_nodes(nodes: &[TreeShapNode], x: &[f64]) -> f64 {
    let mut j = 0usize;
    while nodes[j].feature >= 0 {
        let n = &nodes[j];
        j = if x[n.feature as usize] <= n.threshold {
            n.left
        } else {
            n.right
        } as usize;
    }
    nodes[j].value
}

/// `sum_leaf value * cover / root cover`: `E[f]` under the tree's training distribution, the
/// value of the empty coalition of the path-dependent game.
fn tree_expected_value(nodes: &[TreeShapNode]) -> f64 {
    nodes
        .iter()
        .filter(|n| n.feature < 0)
        .map(|n| n.value * n.n_samples as f64)
        .sum::<f64>()
        / nodes[0].n_samples as f64
}

/// One element of TreeSHAP's unique-feature path (Lundberg et al. 2020, Algorithm 2).
#[derive(Debug, Clone, Copy, Default)]
struct PathElement {
    feature: i32,
    zero: f64,
    one: f64,
    weight: f64,
}

/// Length of the path buffer for a tree whose deepest leaf is at `depth`: the recursion at depth
/// d keeps its path (at most d + 1 elements) after its ancestors', `(D + 1)(D + 2) / 2` in all
/// (shap's `unique_path` layout).
fn path_buffer_len(depth: usize) -> EconResult<usize> {
    Ok(crate::resource::checked_len(
        &[depth.saturating_add(1), depth.saturating_add(2)],
        "TreeSHAP path buffer",
    )? / 2)
}

/// Working memory of the TreeSHAP recursion for trees of depth at most `D`: the path buffer
/// ([`path_buffer_len`]) and `inv[k] = 1 / k` for `k = 1..=D + 1` (`inv[0] = 0`, unused). A path
/// has at most `D + 1` elements, so every count EXTEND and UNWIND divide by is in the table: the
/// recursions then multiply instead of dividing, and the divisions leave the serial chain of
/// UNWIND's weight recurrence.
#[derive(Clone)]
struct TreeShapScratch {
    path: Vec<PathElement>,
    inv: Vec<f64>,
}

impl TreeShapScratch {
    fn new(depth: usize) -> EconResult<Self> {
        use crate::resource;
        let path = resource::try_vec(
            path_buffer_len(depth)?,
            PathElement::default(),
            "TreeSHAP path buffer",
        )?;
        let len = depth.saturating_add(2);
        let mut inv = resource::try_vec(len, 0.0, "TreeSHAP reciprocal table")?;
        for (k, v) in inv.iter_mut().enumerate().skip(1) {
            *v = 1.0 / k as f64;
        }
        Ok(Self { path, inv })
    }
}

/// EXTEND: append (feature, zero fraction, one fraction) to the path of length `len` and update
/// the permutation weights. `inv` is [`TreeShapScratch::inv`] (`len + 1` in range).
fn extend_path(
    path: &mut [PathElement],
    inv: &[f64],
    len: usize,
    zero: f64,
    one: f64,
    feature: i32,
) {
    path[len] = PathElement {
        feature,
        zero,
        one,
        weight: if len == 0 { 1.0 } else { 0.0 },
    };
    let inv_l1 = inv[len + 1];
    let (one_l1, zero_l1) = (one * inv_l1, zero * inv_l1);
    // weight'[i + 1] = weight[i + 1] zero (len - i - 1) / l1 + weight[i] one (i + 1) / l1: the
    // first term is carried in a register rather than stored and reloaded, so iterations do not
    // wait on each other through memory.
    let mut carry = path[len].weight;
    for i in (0..len).rev() {
        let w = path[i].weight;
        path[i + 1].weight = carry + w * (one_l1 * (i + 1) as f64);
        carry = w * (zero_l1 * (len - i) as f64);
    }
    path[0].weight = carry;
}

/// UNWIND: undo the extension of element `k` of the path of length `len` (a feature split on
/// again further down); the path then has length `len - 1`.
fn unwind_path(path: &mut [PathElement], inv: &[f64], len: usize, k: usize) {
    let depth = len - 1;
    let (one, zero) = (path[k].one, path[k].zero);
    let d1 = (depth + 1) as f64;
    let inv_d1 = inv[depth + 1];
    if one != 0.0 {
        let d1_one = d1 / one;
        let zero_d1 = zero * inv_d1;
        let mut next_one = path[depth].weight;
        for i in (0..depth).rev() {
            let tmp = path[i].weight;
            let w = next_one * (d1_one * inv[i + 1]);
            path[i].weight = w;
            next_one = tmp - w * (zero_d1 * (depth - i) as f64);
        }
    } else {
        let d1_zero = d1 / zero;
        for i in (0..depth).rev() {
            path[i].weight *= d1_zero * inv[depth - i];
        }
    }
    for i in k..depth {
        path[i].feature = path[i + 1].feature;
        path[i].zero = path[i + 1].zero;
        path[i].one = path[i + 1].one;
    }
}

/// The total permutation weight of the path of length `len` if element `k` were unwound (the
/// leaf contribution's weight), one element at a time: the reference [`leaf_weights`] is tested
/// against.
#[cfg(test)]
fn unwound_path_sum(path: &[PathElement], inv: &[f64], len: usize, k: usize) -> f64 {
    let depth = len - 1;
    let (one, zero) = (path[k].one, path[k].zero);
    let d1 = (depth + 1) as f64;
    let mut total = 0.0;
    if one != 0.0 {
        let d1_one = d1 / one;
        let zero_d1 = zero * inv[depth + 1];
        let mut next_one = path[depth].weight;
        for i in (0..depth).rev() {
            let tmp = next_one * (d1_one * inv[i + 1]);
            total += tmp;
            next_one = path[i].weight - tmp * (zero_d1 * (depth - i) as f64);
        }
    } else {
        let d1_zero = d1 / zero;
        for i in (0..depth).rev() {
            total += path[i].weight * (d1_zero * inv[depth - i]);
        }
    }
    total
}

/// Adds each element's leaf contribution `unwound_path_sum(path, k) * (one_k - zero_k) * scale`
/// to `phi[feature_k]`, for k = 1..len (Algorithm 2's leaf step), with all elements at once and
/// no data-dependent branch. Every one fraction is 0 or 1 (the root's is 1, a hot child inherits
/// the one fraction of the element it unwinds or 1, a cold child gets 0), so:
///
/// * an element with one fraction 0 unwinds by `w_j * d1 / (zero (depth - j))`; its contribution
///   (times `one - zero = -zero`) is `-d1 * sum_j w_j / (depth - j) * scale`, one O(D) sum shared
///   by every such element;
/// * an element with one fraction 1 runs UNWIND's recurrence `tmp = next * d1 / (j + 1)`,
///   `next = w_j - tmp * zero (depth - j) / d1`, written as `next = w_j - next * g_j zero` so
///   that one multiply-add is the only step on the serial chain; four elements run side by side
///   in fixed-width lanes (the recurrence runs for every lane and the one fraction selects the
///   result).
///
/// The same weights as [`unwound_path_sum`] up to rounding (tested).
fn leaf_weights(path: &[PathElement], inv: &[f64], phi: &mut [f64], scale: f64) {
    const LANES: usize = 4;
    let len = path.len();
    let depth = len - 1;
    let d1 = (depth + 1) as f64;
    let inv_d1 = inv[depth + 1];
    let mut cold_sum = 0.0;
    for j in 0..depth {
        cold_sum += path[j].weight * inv[depth - j];
    }
    let cold = -(cold_sum * d1);
    let w_top = path[depth].weight;
    for chunk in path[1..].chunks(LANES) {
        let mut z = [0.0; LANES];
        for (zl, el) in z.iter_mut().zip(chunk) {
            debug_assert!(el.one == 0.0 || el.one == 1.0);
            *zl = el.zero;
        }
        let mut next = [w_top; LANES];
        let mut total = [0.0; LANES];
        for j in (0..depth).rev() {
            let w = path[j].weight;
            let c = d1 * inv[j + 1];
            let g = c * ((depth - j) as f64 * inv_d1);
            for l in 0..LANES {
                total[l] += next[l] * c;
                next[l] = w - next[l] * (g * z[l]);
            }
        }
        for (l, el) in chunk.iter().enumerate() {
            let hot = el.one;
            let contribution = hot * (total[l] * (1.0 - el.zero)) + (1.0 - hot) * cold;
            phi[el.feature as usize] += contribution * scale;
        }
    }
}

/// TreeSHAP recursion state over one validated node table. `condition` 0: plain SHAP; +1 / -1:
/// feature `condition_feature` held present / absent (shap's `tree_shap_recursive` condition
/// mechanism, used for interaction values). `phi` is added to; `path` is the preallocated
/// buffer of [`path_buffer_len`] elements, reused across calls.
struct TreeShapRun<'n, 'x, 'p> {
    nodes: &'n [TreeShapNode],
    /// [`cover_fractions`] of `nodes`.
    frac: &'n [f64],
    x: &'x [f64],
    phi: &'p mut [f64],
    path: &'p mut [PathElement],
    inv: &'p [f64],
    condition: i8,
    condition_feature: i32,
}

impl TreeShapRun<'_, '_, '_> {
    /// Visit `node`; the parent's path is `path[off..off + parent_len]`, and this node's path is
    /// built right after it.
    #[allow(clippy::too_many_arguments)]
    fn recurse(
        &mut self,
        node: usize,
        off: usize,
        parent_len: usize,
        parent_zero: f64,
        parent_one: f64,
        parent_feature: i32,
        condition_fraction: f64,
    ) {
        if condition_fraction == 0.0 {
            return;
        }
        let start = off + parent_len;
        // a few elements: an inline loop, not a memmove call
        let (parent, child) = self.path.split_at_mut(start);
        for (c, p) in child[..parent_len].iter_mut().zip(&parent[off..]) {
            *c = *p;
        }
        let mut len = parent_len;
        if self.condition == 0 || self.condition_feature != parent_feature {
            extend_path(
                &mut self.path[start..],
                self.inv,
                len,
                parent_zero,
                parent_one,
                parent_feature,
            );
            len += 1;
        }
        let nodes = self.nodes;
        let n = &nodes[node];
        if n.feature < 0 {
            leaf_weights(
                &self.path[start..start + len],
                self.inv,
                self.phi,
                n.value * condition_fraction,
            );
            return;
        }
        let f = n.feature as usize;
        let (hot, cold) = if self.x[f] <= n.threshold {
            (n.left as usize, n.right as usize)
        } else {
            (n.right as usize, n.left as usize)
        };
        let hot_zero = self.frac[hot];
        let cold_zero = self.frac[cold];
        let (mut inc_zero, mut inc_one) = (1.0, 1.0);
        // never the root element's feature -1; a unique path holds the feature at most once, so
        // the scan needs no early exit (and no data-dependent branch)
        let mut found = usize::MAX;
        for (k, e) in self.path[start..start + len].iter().enumerate() {
            if e.feature == n.feature {
                found = k;
            }
        }
        if found != usize::MAX {
            let k = found;
            inc_zero = self.path[start + k].zero;
            inc_one = self.path[start + k].one;
            unwind_path(&mut self.path[start..], self.inv, len, k);
            len -= 1;
        }
        let (mut hot_cf, mut cold_cf) = (condition_fraction, condition_fraction);
        if self.condition > 0 && n.feature == self.condition_feature {
            cold_cf = 0.0;
        } else if self.condition < 0 && n.feature == self.condition_feature {
            hot_cf *= hot_zero;
            cold_cf *= cold_zero;
        }
        self.recurse(
            hot,
            start,
            len,
            hot_zero * inc_zero,
            inc_one,
            n.feature,
            hot_cf,
        );
        self.recurse(
            cold,
            start,
            len,
            cold_zero * inc_zero,
            0.0,
            n.feature,
            cold_cf,
        );
    }
}

/// `frac[c] = cover(c) / cover(parent(c))` (1 for the root): the zero fraction of each edge,
/// computed once per tree instead of once per explained row. `nodes` must have passed
/// [`validate_nodes`] (internal covers are positive).
fn cover_fractions(nodes: &[TreeShapNode]) -> EconResult<Vec<f64>> {
    let mut frac = crate::resource::try_vec(nodes.len(), 1.0, "TreeSHAP cover fractions")?;
    for n in nodes.iter().filter(|n| n.feature >= 0) {
        let cover = n.n_samples as f64;
        for c in [n.left as usize, n.right as usize] {
            frac[c] = nodes[c].n_samples as f64 / cover;
        }
    }
    Ok(frac)
}

/// Adds one tree's (conditioned) TreeSHAP values at `x` to `phi`. `nodes` must have passed
/// [`validate_nodes`], `frac` must be its [`cover_fractions`], and `scratch` must have been
/// built for at least its depth.
fn tree_shap_into(
    nodes: &[TreeShapNode],
    frac: &[f64],
    x: &[f64],
    phi: &mut [f64],
    scratch: &mut TreeShapScratch,
    condition: i8,
    condition_feature: i32,
) {
    let mut run = TreeShapRun {
        nodes,
        frac,
        x,
        phi,
        path: &mut scratch.path,
        inv: &scratch.inv,
        condition,
        condition_feature,
    };
    run.recurse(0, 0, 0, 1.0, 1.0, -1, 1.0);
}

/// Exact TreeSHAP values of one tree (Lundberg, Erion & Lee 2020, Algorithm 2): the Shapley values
/// of v(S) = E[f(x) | x_S] computed from node covers (Algorithm 1). O(L D^2) for L leaves and
/// depth D. `nodes` must be a valid table (it is checked).
#[cfg(test)]
fn tree_shap_single(nodes: &[TreeShapNode], x: ArrayView1<f64>, n_features: usize) -> Array1<f64> {
    let depth = validate_nodes(nodes, n_features, 0).expect("valid node table");
    let frac = cover_fractions(nodes).unwrap();
    let mut path = TreeShapScratch::new(depth).unwrap();
    let mut phi = vec![0.0; n_features];
    let x = x.to_vec();
    tree_shap_into(nodes, &frac, &x, &mut phi, &mut path, 0, -1);
    Array1::from_vec(phi)
}

/// The features a node table splits on, ascending, each once.
fn split_features(nodes: &[TreeShapNode], n_features: usize) -> Vec<usize> {
    let mut used = vec![false; n_features];
    for n in nodes.iter().filter(|n| n.feature >= 0) {
        used[n.feature as usize] = true;
    }
    (0..n_features).filter(|&j| used[j]).collect()
}

/// Adds one tree's SHAP interaction values at `x` to `out` (M x M; Lundberg et al. 2020, eqs.
/// 8-9, computed as shap's `dense_tree_interactions_path_dependent`): for each feature j the tree
/// splits on, `Phi[j][l] = (phi_l | j present - phi_l | j absent) / 2`, and
/// `Phi[l][l] = phi_l - sum_{j != l} Phi[j][l]`. `phi` holds the tree's TreeSHAP values at `x`,
/// `features` its [`split_features`], `scratch` 2 M values. The result is symmetric and its rows
/// sum to `phi`.
#[allow(clippy::too_many_arguments)]
fn tree_interactions_into(
    nodes: &[TreeShapNode],
    frac: &[f64],
    features: &[usize],
    x: &[f64],
    phi: &[f64],
    out: &mut Array2<f64>,
    scratch: &mut [f64],
    path: &mut TreeShapScratch,
) {
    let m = phi.len();
    let (on, off) = scratch.split_at_mut(m);
    for (k, &p) in phi.iter().enumerate() {
        out[[k, k]] += p;
    }
    for &j in features {
        on.fill(0.0);
        off.fill(0.0);
        tree_shap_into(nodes, frac, x, on, path, 1, j as i32);
        tree_shap_into(nodes, frac, x, off, path, -1, j as i32);
        for l in 0..m {
            let val = (on[l] - off[l]) / 2.0;
            out[[j, l]] += val;
            out[[l, l]] -= val;
        }
    }
}

/// SHAP interaction values of one tree at `x` (tests: a single valid table).
#[cfg(test)]
fn tree_shap_interactions_single(
    nodes: &[TreeShapNode],
    x: ArrayView1<f64>,
    n_features: usize,
) -> Array2<f64> {
    let depth = validate_nodes(nodes, n_features, 0).expect("valid node table");
    let frac = cover_fractions(nodes).unwrap();
    let mut path = TreeShapScratch::new(depth).unwrap();
    let mut phi = vec![0.0; n_features];
    let x = x.to_vec();
    tree_shap_into(nodes, &frac, &x, &mut phi, &mut path, 0, -1);
    let mut out = Array2::zeros((n_features, n_features));
    let mut scratch = vec![0.0; 2 * n_features];
    let features = split_features(nodes, n_features);
    tree_interactions_into(
        nodes,
        &frac,
        &features,
        &x,
        &phi,
        &mut out,
        &mut scratch,
        &mut path,
    );
    out
}

/// One validated tree of an ensemble with what every explained row reuses: its
/// [`cover_fractions`] and (for interaction values) its [`split_features`].
struct ShapTree {
    nodes: Vec<TreeShapNode>,
    frac: Vec<f64>,
    features: Vec<usize>,
}

/// The perturbation TreeSHAP computes; any other request is an error that names the route.
fn tree_perturbation(config: &ShapConfig) -> EconResult<FeaturePerturbation> {
    match config
        .feature_perturbation
        .unwrap_or(FeaturePerturbation::TreePathDependent)
    {
        FeaturePerturbation::TreePathDependent => Ok(FeaturePerturbation::TreePathDependent),
        FeaturePerturbation::Interventional => Err(spec(
            "TreeSHAP computes tree_path_dependent values; interventional SHAP values of a forest come from the Kernel path (shap_values_model_with(.., ShapMethod::Kernel, ..) with a background sample)"
                .into(),
        )),
        FeaturePerturbation::Observational => Err(observational_error()),
    }
}

fn observational_error() -> EconError {
    spec("feature_perturbation = observational (conditional expectations) is not implemented; use tree_path_dependent (TreeSHAP) or interventional (Kernel SHAP)".into())
}

/// TreeSHAP values of a tree ensemble whose prediction is the mean of its trees (a random
/// forest): the exact Shapley values of the path-dependent game `v(S) = E[f(x) | x_S]`
/// (Lundberg, Erion & Lee 2020, Algorithms 1-2; shap's `TreeExplainer` with
/// `feature_perturbation = "tree_path_dependent"`), averaged over the trees.
///
/// The base value is the mean over trees of `sum_leaf value * cover / root cover`, and the
/// additivity check compares `base + sum(phi)` with the ensemble's predictions of the rows
/// explained, computed here (tolerance [`ADDITIVITY_TOLERANCE`] relative to `max(1, |f|)`).
/// Cost O(T L D^2) per row for T trees with L leaves and depth D.
///
/// # Errors
///
/// `InvalidSpecification` for no rows or columns, a non-finite value, a feature-name count
/// that differs from the columns, no trees, an unfitted or malformed tree, a tree splitting on a
/// column `x` does not have, a perturbation other than tree_path_dependent, and
/// `n_samples = 0`.
///
/// # References
///
/// - Lundberg, S. M., Erion, G., Chen, H., et al. (2020). "From local explanations to global
///   understanding with explainable AI for trees". Nature Machine Intelligence 2, 56-67.
pub fn shap_tree_ensemble(
    trees: &[super::trees::DecisionTree],
    x: ArrayView2<f64>,
    feature_names: Option<Vec<String>>,
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    use crate::resource;
    let feature_perturbation = tree_perturbation(config)?;
    if config.n_samples == Some(0) {
        return Err(spec("n_samples must be at least 1".into()));
    }
    validate_explain_inputs(x, feature_names.as_deref())?;
    if trees.is_empty() {
        return Err(spec("TreeSHAP needs at least one tree".into()));
    }
    let (n, m) = x.dim();
    let mut flat = resource::try_with_capacity::<ShapTree>(trees.len(), "TreeSHAP trees")?;
    let mut max_depth = 0usize;
    for (t, tree) in trees.iter().enumerate() {
        let nodes = extract_tree_structure(tree);
        max_depth = max_depth.max(validate_nodes(&nodes, m, t)?);
        let frac = cover_fractions(&nodes)?;
        let features = if config.compute_interactions {
            split_features(&nodes, m)
        } else {
            Vec::new()
        };
        flat.push(ShapTree {
            nodes,
            frac,
            features,
        });
    }
    let n_trees = flat.len() as f64;
    // Cloned for each rayon job that runs rows (a few hundred bytes for the deepest tree the
    // forest allows).
    let proto = (
        TreeShapScratch::new(max_depth)?,
        resource::try_vec(4 * m, 0.0, "SHAP row scratch")?,
    );
    let base_value = flat
        .iter()
        .map(|t| tree_expected_value(&t.nodes))
        .sum::<f64>()
        / n_trees;

    // Every output is allocated here, through the allocation checks, and each row writes only
    // its own slots.
    let mut shap_values = resource::try_zeros2(n, m, "SHAP values")?;
    let mut predictions = resource::try_vec(n, 0.0, "SHAP predictions")?;
    let mut interactions = if config.compute_interactions {
        let len = resource::checked_len(&[n, m, m], "SHAP interaction values")?;
        resource::check_alloc::<f64>(len, "SHAP interaction values")?;
        let mut all = resource::try_with_capacity::<Array2<f64>>(n, "SHAP interaction values")?;
        for _ in 0..n {
            all.push(resource::try_zeros2(m, m, "SHAP interaction values")?);
        }
        Some(all)
    } else {
        None
    };
    // Rows are independent: each worker owns its path buffer and scratch, so the values do not
    // depend on the number of threads. The budget is captured here (rayon's threads have none).
    let budget = resource::current_budget();
    let row = |(path, scratch): &mut (TreeShapScratch, Vec<f64>),
               i: usize,
               phi: &mut [f64],
               pred_out: &mut f64,
               acc: Option<&mut Array2<f64>>|
     -> EconResult<()> {
        if let Some(b) = budget.as_ref() {
            b.check()?;
        }
        let (xi, rest) = scratch.split_at_mut(m);
        for (v, &xv) in xi.iter_mut().zip(x.row(i)) {
            *v = xv;
        }
        let mut pred = 0.0;
        if let Some(acc) = acc {
            let (tree_phi, rest) = rest.split_at_mut(m);
            for t in &flat {
                tree_phi.fill(0.0);
                tree_shap_into(&t.nodes, &t.frac, xi, tree_phi, path, 0, -1);
                for (p, v) in phi.iter_mut().zip(tree_phi.iter()) {
                    *p += v;
                }
                tree_interactions_into(
                    &t.nodes,
                    &t.frac,
                    &t.features,
                    xi,
                    tree_phi,
                    acc,
                    rest,
                    path,
                );
                pred += predict_nodes(&t.nodes, xi);
            }
            *acc /= n_trees;
        } else {
            for t in &flat {
                tree_shap_into(&t.nodes, &t.frac, xi, phi, path, 0, -1);
                pred += predict_nodes(&t.nodes, xi);
            }
        }
        for v in phi.iter_mut() {
            *v /= n_trees;
        }
        *pred_out = pred / n_trees;
        Ok(())
    };
    let rows = shap_values
        .as_slice_mut()
        .expect("a freshly allocated matrix is in standard layout")
        .par_chunks_mut(m)
        .zip(predictions.par_iter_mut())
        .enumerate();
    match interactions.as_mut() {
        Some(all) => rows.zip(all.par_iter_mut()).try_for_each_init(
            || proto.clone(),
            |s, ((i, (phi, pred)), acc)| row(s, i, phi, pred, Some(acc)),
        )?,
        None => rows.try_for_each_init(
            || proto.clone(),
            |s, (i, (phi, pred))| row(s, i, phi, pred, None),
        )?,
    }

    let feature_importance = mean_abs_columns(&shap_values);
    let (additivity_check_passed, max_additivity_error, additivity_warning) = additivity(
        base_value,
        &shap_values,
        &predictions,
        config.check_additivity,
    );
    let mut warnings = Vec::new();
    if let Some(ns) = config.n_samples {
        warnings.push(format!(
            "n_samples = {ns} is a Kernel SHAP setting; TreeSHAP is exact and does not sample"
        ));
    }
    warnings.extend(additivity_warning);

    Ok(ShapResult {
        shap_values,
        base_value,
        feature_names,
        n_obs: n,
        n_features: m,
        feature_importance,
        interaction_values: interactions,
        additivity_check_passed,
        max_additivity_error,
        method: ShapMethod::Tree,
        feature_perturbation,
        exact: true,
        n_coalitions: None,
        seed: None,
        predictions,
        warnings,
    })
}

// ============================================================================
// Kernel SHAP Implementation
// ============================================================================

/// Rows of the Kernel SHAP regression: `z` (coalitions x M; 1.0 = the feature takes x's value)
/// and their weights.
struct CoalitionDesign {
    z: Array2<f64>,
    weights: Vec<f64>,
    /// Rows from completely enumerated subset sizes (the rest were sampled).
    n_fixed: usize,
    /// Subset sizes (with their complements) enumerated completely.
    num_full: usize,
    /// Every proper non-empty coalition is in the design.
    exact: bool,
}

/// Shapley kernel `pi(s) = (M - 1) / (C(M, s) s (M - s))`, `1 <= s <= M - 1` (Lundberg & Lee
/// 2017, Theorem 2).
fn shap_kernel_weight(m: usize, s: usize) -> f64 {
    (m - 1) as f64 / (binomial_coefficient(m, s) * s as f64 * (m - s) as f64)
}

/// `C(n, k)`, exact while `C(n, k) < 2^53` (multiplicative; every partial product is an
/// integer).
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64)
}

/// Every proper non-empty coalition (`2^M - 2` rows) with weight `pi(|S|)`. Caller: `2 <= m <= 62`.
fn enumerate_coalitions(m: usize) -> EconResult<CoalitionDesign> {
    use crate::resource;
    let rows = ((1u64 << m) - 2) as usize;
    let len = resource::checked_len(&[rows, m], "Kernel SHAP coalitions")?;
    resource::check_alloc::<f64>(len, "Kernel SHAP coalitions")?;
    let mut z = resource::try_zeros2(rows, m, "Kernel SHAP coalitions")?;
    let mut weights = resource::try_vec(rows, 0.0, "Kernel SHAP weights")?;
    for (r, mut row) in z.outer_iter_mut().enumerate() {
        if r % 4096 == 0 {
            resource::checkpoint()?;
        }
        let mask = r as u64 + 1;
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((mask >> j) & 1) as f64;
        }
        weights[r] = shap_kernel_weight(m, mask.count_ones() as usize);
    }
    Ok(CoalitionDesign {
        z,
        weights,
        n_fixed: rows,
        num_full: m / 2,
        exact: true,
    })
}

/// Whether subset size s, with `n_subsets` coalitions (complements included) and remaining
/// size weight `remaining`, is enumerated completely from a budget of `samples_left`: shap's test
/// `samples_left * remaining / n_subsets >= 1 - 1e-8`, and the coalitions must fit the budget.
/// The tolerance alone admits up to `n_subsets * 1e-8` coalitions too many, which a size with
/// more than 1e8 subsets (M >= 30) can reach.
fn size_fits(samples_left: usize, remaining: f64, n_subsets: f64) -> bool {
    samples_left as f64 * remaining / n_subsets >= 1.0 - 1e-8 && n_subsets <= samples_left as f64
}

/// shap.KernelExplainer's coalition design (`explain()`, shap 0.4x):
///  * subset sizes s = 1..=ceil((M-1)/2), paired with M - s when s <= floor((M-1)/2); size
///    weights `w_s ∝ (M-1)/(s(M-s))`, doubled for paired sizes, normalised;
///  * complete sizes from s = 1 upwards while `n_left * w_s / n_subsets(s) >= 1 - 1e-8`
///    (renormalising the remaining weights after each), each subset (and complement) with weight
///    `w_s / C(M, s)` (halved if paired);
///  * the remaining budget: sizes drawn from the remaining size weights (halved if paired,
///    renormalised), members drawn uniformly without replacement, the complement added for
///    paired sizes; a repeated coalition increments its weight (and its complement's); at most
///    4 x the remaining budget draws; the sampled weights are then scaled to sum to the remaining
///    size weight `sum_{s > num_full} w_s`.
///
/// The budget is capped at `2^M - 2` (as shap does for M <= 30), and `exact` is true when every
/// size was complete.
fn sampled_coalitions(
    m: usize,
    n_samples: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> EconResult<CoalitionDesign> {
    use crate::resource;
    use rand::distributions::{Distribution, WeightedIndex};

    let n_samples = if m < 63 {
        n_samples.min(((1u64 << m) - 2) as usize)
    } else {
        n_samples
    };
    let len = resource::checked_len(&[n_samples, m], "Kernel SHAP coalitions")?;
    resource::check_alloc::<f64>(len, "Kernel SHAP coalitions")?;
    let mut masks = resource::try_with_capacity::<bool>(len, "Kernel SHAP coalitions")?;
    let mut weights = resource::try_with_capacity::<f64>(n_samples, "Kernel SHAP weights")?;
    let add = |mask: &[bool],
               w: f64,
               masks: &mut Vec<bool>,
               weights: &mut Vec<f64>|
     -> EconResult<usize> {
        let row = weights.len();
        if row.is_multiple_of(4096) {
            resource::checkpoint()?;
        }
        masks.extend_from_slice(mask);
        weights.push(w);
        Ok(row)
    };

    let num_subset_sizes = m / 2; // ceil((m - 1) / 2)
    let num_paired = (m - 1) / 2;
    let mut weight_vector: Vec<f64> = (1..=num_subset_sizes)
        .map(|s| {
            let w = (m - 1) as f64 / (s * (m - s)) as f64;
            if s <= num_paired { 2.0 * w } else { w }
        })
        .collect();
    let total: f64 = weight_vector.iter().sum();
    weight_vector.iter_mut().for_each(|w| *w /= total);

    let mut num_full = 0usize;
    let mut samples_left = n_samples;
    let mut remaining = weight_vector.clone();
    let mut mask = vec![false; m];
    for s in 1..=num_subset_sizes {
        let paired = s <= num_paired;
        let n_subsets = binomial_coefficient(m, s) * if paired { 2.0 } else { 1.0 };
        if !size_fits(samples_left, remaining[s - 1], n_subsets) {
            break;
        }
        num_full += 1;
        samples_left -= n_subsets as usize;
        if remaining[s - 1] < 1.0 {
            let r = 1.0 - remaining[s - 1];
            remaining.iter_mut().for_each(|w| *w /= r);
        }
        let w = weight_vector[s - 1] / binomial_coefficient(m, s) / if paired { 2.0 } else { 1.0 };
        // every s-subset in lexicographic order
        let mut idx: Vec<usize> = (0..s).collect();
        loop {
            mask.fill(false);
            idx.iter().for_each(|&j| mask[j] = true);
            add(&mask, w, &mut masks, &mut weights)?;
            if paired {
                mask.iter_mut().for_each(|b| *b = !*b);
                add(&mask, w, &mut masks, &mut weights)?;
            }
            let Some(i) = (0..s).rev().find(|&i| idx[i] != i + m - s) else {
                break;
            };
            idx[i] += 1;
            for k in i + 1..s {
                idx[k] = idx[k - 1] + 1;
            }
        }
    }

    let n_fixed = weights.len();
    let exact = num_full == num_subset_sizes;
    if !exact && samples_left > 0 {
        let mut rem: Vec<f64> = weight_vector
            .iter()
            .enumerate()
            .map(|(k, &w)| if k < num_paired { w / 2.0 } else { w })
            .skip(num_full)
            .collect();
        let rs: f64 = rem.iter().sum();
        rem.iter_mut().for_each(|w| *w /= rs);
        let sizes = WeightedIndex::new(&rem)
            .map_err(|e| EconError::Computation(format!("Kernel SHAP size weights: {e}")))?;
        let draws = samples_left
            .checked_mul(4)
            .ok_or_else(|| spec("n_samples overflows".into()))?;
        let mut used: HashMap<Vec<bool>, (usize, Option<usize>)> = HashMap::new();
        for _ in 0..draws {
            if samples_left == 0 {
                break;
            }
            let size = sizes.sample(rng) + num_full + 1;
            mask.fill(false);
            for j in rand::seq::index::sample(rng, m, size).iter() {
                mask[j] = true;
            }
            let paired = size <= num_paired;
            if let Some(&(row, comp)) = used.get(&mask) {
                weights[row] += 1.0;
                if let Some(c) = comp {
                    weights[c] += 1.0;
                }
                continue;
            }
            let row = add(&mask, 1.0, &mut masks, &mut weights)?;
            samples_left -= 1;
            let key = mask.clone();
            let mut comp = None;
            if samples_left > 0 && paired {
                mask.iter_mut().for_each(|b| *b = !*b);
                comp = Some(add(&mask, 1.0, &mut masks, &mut weights)?);
                samples_left -= 1;
            }
            used.insert(key, (row, comp));
        }
        let weight_left: f64 = weight_vector[num_full..].iter().sum();
        let sampled: f64 = weights[n_fixed..].iter().sum();
        weights[n_fixed..]
            .iter_mut()
            .for_each(|w| *w *= weight_left / sampled);
    }

    let rows = weights.len();
    let mut z = resource::try_zeros2(rows, m, "Kernel SHAP coalitions")?;
    for (v, &b) in z.iter_mut().zip(&masks) {
        *v = if b { 1.0 } else { 0.0 };
    }
    Ok(CoalitionDesign {
        z,
        weights,
        n_fixed,
        num_full,
        exact,
    })
}

/// Kernel SHAP: the interventional Shapley values of any model, `v(S) = mean_b f(x_S, b_notS)`
/// over the rows `b` of `background` (Lundberg & Lee 2017; Janzing, Minorics & Blöbaum 2020), by
/// shap.KernelExplainer's weighted regression on coalitions with the efficiency constraint
/// `sum(phi) = f(x) - E f` imposed exactly (the last feature is eliminated) and the Shapley kernel
/// applied once.
///
/// With `n = config.n_samples` (default `2 M + 2048`), every coalition is evaluated when
/// `2^M - 2 <= n`, and the values are then exact (the default does so for M <= 11); otherwise
/// shap's design with `n` coalitions: complete subset sizes first, then sizes drawn in proportion
/// to the kernel mass and members without replacement, from ChaCha8 seeded with
/// `config.seed` (default [`SHAP_DEFAULT_SEED`]) on stream i for row i, so each row is
/// reproducible on its own. One feature gives `phi = f(x) - E f`. The base value is the mean
/// prediction over the background.
///
/// # Errors
///
/// `InvalidSpecification` for no rows or features, a non-finite value in `x` or `background`, a
/// background with other columns or no rows, a feature-name count that differs from the columns,
/// a `predict_fn` output of the wrong length or with a non-finite value, `n_samples = 0`, a
/// sampled design that does not identify every feature, a perturbation other than
/// interventional, and `compute_interactions`.
///
/// # References
///
/// - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting
///   Model Predictions". NeurIPS 2017.
/// - Janzing, D., Minorics, L., & Blöbaum, P. (2020). "Feature relevance quantification in
///   explainable AI: A causal problem". AISTATS 2020.
pub fn kernel_shap<F>(
    predict_fn: F,
    x: ArrayView2<f64>,
    background: ArrayView2<f64>,
    config: &ShapConfig,
    feature_names: Option<Vec<String>>,
) -> EconResult<ShapResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    use crate::linalg::solve::{LM_QR_TOL, Qr, weighted_least_squares};
    use crate::resource;
    use rand::SeedableRng;

    match config
        .feature_perturbation
        .unwrap_or(FeaturePerturbation::Interventional)
    {
        FeaturePerturbation::Interventional => {}
        FeaturePerturbation::TreePathDependent => {
            return Err(spec(
                "Kernel SHAP computes interventional values; tree_path_dependent needs TreeSHAP"
                    .into(),
            ));
        }
        FeaturePerturbation::Observational => return Err(observational_error()),
    }
    if config.compute_interactions {
        return Err(spec(
            "SHAP interaction values are computed by TreeSHAP only".into(),
        ));
    }
    validate_explain_inputs(x, feature_names.as_deref())?;
    let (n, m) = x.dim();
    validate_background(background, m)?;
    let n_samples = match config.n_samples {
        Some(0) => return Err(spec("n_samples must be at least 1".into())),
        Some(v) => v,
        None => m
            .checked_mul(2)
            .and_then(|v| v.checked_add(2048))
            .ok_or_else(|| spec("n_samples overflows".into()))?,
    };

    let n_bg = background.nrows();
    let background_preds = predict_fn(background);
    check_predictions(&background_preds, n_bg)?;
    let base_value = background_preds.iter().sum::<f64>() / n_bg as f64;
    let predictions = predict_fn(x);
    check_predictions(&predictions, n)?;

    let mut shap_values = resource::try_zeros2(n, m, "SHAP values")?;
    let mut warnings = Vec::new();
    let (exact, n_coalitions, seed) = if m == 1 {
        for (i, &f) in predictions.iter().enumerate() {
            shap_values[[i, 0]] = f - base_value;
        }
        (true, 0, None)
    } else {
        let shared = if m < 63 && (1u64 << m) - 2 <= n_samples as u64 {
            Some(enumerate_coalitions(m)?)
        } else {
            None
        };
        // Every row shares the enumerated design, so its weighted regression matrix
        // `sqrt(w) * E` (E[r, j] = z_rj - z_r,M) is factored once; a row then only builds its
        // response. The rows' arithmetic is weighted_least_squares' (its weights are positive).
        let shared_fit = match &shared {
            Some(d) => {
                let k = d.z.nrows();
                let mut xs = resource::try_zeros2(k, m - 1, "Kernel SHAP regression")?;
                for (r, z) in d.z.outer_iter().enumerate() {
                    let sw = d.weights[r].sqrt();
                    let last = z[m - 1];
                    for j in 0..m - 1 {
                        xs[[r, j]] = (z[j] - last) * sw;
                    }
                }
                let qr = Qr::decompose(xs.view(), LM_QR_TOL)?;
                if qr.rank() < m - 1 {
                    return Err(EconError::Computation(format!(
                        "Kernel SHAP: the enumerated design has rank {} of {}",
                        qr.rank(),
                        m - 1
                    )));
                }
                let sw: Vec<f64> = d.weights.iter().map(|w| w.sqrt()).collect();
                Some((qr, sw))
            }
            None => None,
        };
        let seed = config.seed.unwrap_or(SHAP_DEFAULT_SEED);
        let mut masked = resource::try_zeros2(n_bg, m, "Kernel SHAP masked rows")?;
        let (mut min_rows, mut max_rows) = (usize::MAX, 0usize);
        for i in 0..n {
            resource::checkpoint()?;
            let own;
            let design = match &shared {
                Some(d) => d,
                None => {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                    rng.set_stream(i as u64);
                    own = sampled_coalitions(m, n_samples, &mut rng)?;
                    &own
                }
            };
            let k = design.z.nrows();
            min_rows = min_rows.min(k);
            max_rows = max_rows.max(k);
            let delta = predictions[i] - base_value;
            // the sampled design differs by row and is built per row
            let mut e = match shared_fit {
                Some(_) => Array2::zeros((0, m - 1)),
                None => resource::try_zeros2(k, m - 1, "Kernel SHAP regression")?,
            };
            let mut y2 = resource::try_zeros1(k, "Kernel SHAP regression")?;
            for (r, z) in design.z.outer_iter().enumerate() {
                resource::checkpoint()?;
                masked.assign(&background);
                for (j, &zj) in z.iter().enumerate() {
                    if zj == 1.0 {
                        masked.column_mut(j).fill(x[[i, j]]);
                    }
                }
                let p = predict_fn(masked.view());
                check_predictions(&p, n_bg)?;
                let y = p.iter().sum::<f64>() / n_bg as f64 - base_value;
                let last = z[m - 1];
                if shared_fit.is_none() {
                    for j in 0..m - 1 {
                        e[[r, j]] = z[j] - last;
                    }
                }
                y2[r] = y - last * delta;
            }
            // weighted_least_squares refuses a non-finite response; the factored shared design
            // does not, so both paths check it here with one message
            if let Some(r) = y2.iter().position(|v| !v.is_finite()) {
                return Err(spec(format!(
                    "Kernel SHAP: row {i}: the value of coalition {r} is not finite (the model's predictions overflow when averaged or differenced)"
                )));
            }
            let coefficients = match &shared_fit {
                Some((qr, sw)) => {
                    for (v, s) in y2.iter_mut().zip(sw) {
                        *v *= s;
                    }
                    qr.coefficients(y2.view())?
                }
                None => {
                    let w = ndarray::ArrayView1::from(&design.weights[..]);
                    let ls = weighted_least_squares(e.view(), y2.view(), w)?;
                    if ls.rank < m - 1 {
                        return Err(spec(format!(
                            "Kernel SHAP: the {k} sampled coalitions do not identify all {m} features (rank {} of {}); increase n_samples",
                            ls.rank,
                            m - 1
                        )));
                    }
                    ls.coefficients
                }
            };
            let mut rest = delta;
            for j in 0..m - 1 {
                shap_values[[i, j]] = coefficients[j];
                rest -= coefficients[j];
            }
            shap_values[[i, m - 1]] = rest;
        }
        let exact = shared.is_some();
        if min_rows != max_rows {
            warnings.push(format!(
                "Kernel SHAP: the sampled design reached between {min_rows} and {max_rows} distinct coalitions per row (budget {n_samples})"
            ));
        }
        (exact, max_rows, (!exact).then_some(seed))
    };

    let feature_importance = mean_abs_columns(&shap_values);
    let (additivity_check_passed, max_additivity_error, additivity_warning) = additivity(
        base_value,
        &shap_values,
        &predictions,
        config.check_additivity,
    );
    warnings.extend(additivity_warning);

    Ok(ShapResult {
        shap_values,
        base_value,
        feature_names,
        n_obs: n,
        n_features: m,
        feature_importance,
        interaction_values: None,
        additivity_check_passed,
        max_additivity_error,
        method: ShapMethod::Kernel,
        feature_perturbation: FeaturePerturbation::Interventional,
        exact,
        n_coalitions: Some(n_coalitions),
        seed,
        predictions,
        warnings,
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Compute SHAP values for a Random Forest model.
///
/// Uses TreeSHAP ([`shap_tree_ensemble`]) on `trees`; `result` supplies only the feature names
/// (its training predictions are not read: the rows explained are predicted here).
///
/// # Arguments
/// * `result` - A trained Random Forest result
/// * `x` - Input data matrix (n_samples x n_features)
/// * `config` - SHAP configuration
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{random_forest, random_forest_with_trees, shap_values_random_forest, ShapConfig};
///
/// let rf_result = random_forest(x.view(), y.view(), Some(50), None, None, None, Some(42), names)?;
/// let model = random_forest_with_trees(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
/// let shap = shap_values_random_forest(&rf_result, x.view(), &model.trees, &ShapConfig::default())?;
/// println!("Feature importance: {:?}", shap.feature_importance);
/// ```
///
/// # References
///
/// - Lundberg et al. (2018). "Consistent Individualized Feature Attribution for
///   Tree Ensembles". arXiv:1802.03888.
pub fn shap_values_random_forest(
    result: &super::RandomForestResult,
    x: ArrayView2<f64>,
    trees: &[super::trees::DecisionTree],
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    shap_tree_ensemble(trees, x, result.feature_names.clone(), config)
}

/// Compute SHAP summary statistics.
///
/// Aggregates SHAP values across all samples to produce global feature importance.
pub fn shap_summary(result: &ShapResult) -> ShapSummary {
    let n_features = result.n_features;

    let feature_names: Vec<String> = result
        .feature_names
        .clone()
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    let mean_abs_shap: Vec<f64> = (0..n_features)
        .map(|j| {
            result
                .shap_values
                .column(j)
                .mapv(|v| v.abs())
                .mean()
                .unwrap_or(0.0)
        })
        .collect();

    let mean_shap: Vec<f64> = (0..n_features)
        .map(|j| result.shap_values.column(j).mean().unwrap_or(0.0))
        .collect();

    let std_shap: Vec<f64> = (0..n_features)
        .map(|j| {
            let col = result.shap_values.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
            variance.sqrt()
        })
        .collect();

    // Compute importance ranks
    let mut indexed: Vec<(usize, f64)> = mean_abs_shap
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut importance_rank = vec![0; n_features];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        importance_rank[*idx] = rank + 1;
    }

    ShapSummary {
        feature_names,
        mean_abs_shap,
        mean_shap,
        std_shap,
        importance_rank,
        n_obs: result.n_obs,
    }
}

/// Compute SHAP values using Kernel SHAP (for any model).
///
/// An alias of [`kernel_shap`]: the interventional Shapley values of any model by the
/// constrained weighted regression on coalitions, exact when every coalition fits the
/// `n_samples` budget.
///
/// # Arguments
/// * `predict_fn` - Function that takes a batch of inputs and returns predictions
/// * `x` - Input data to explain (n_samples x n_features)
/// * `background` - Background dataset for marginalization
/// * `config` - SHAP configuration
/// * `feature_names` - Optional feature names
///
/// # Example
///
/// ```
/// use ndarray::{ArrayView2, array};
/// use p2a_core::ml::{ShapConfig, shap_kernel};
///
/// // a linear model: phi_j = beta_j (x_j - mean of the background), exactly
/// let predict = |z: ArrayView2<f64>| {
///     z.outer_iter().map(|r| r[0] + 2.0 * r[1]).collect::<Vec<f64>>()
/// };
/// let x = array![[1.0, 2.0]];
/// let background = array![[0.0, 0.0], [1.0, 1.0]];
/// let shap = shap_kernel(predict, x.view(), background.view(), &ShapConfig::default(), None).unwrap();
/// assert!((shap.shap_values[[0, 0]] - 0.5).abs() < 1e-12);
/// assert!((shap.shap_values[[0, 1]] - 3.0).abs() < 1e-12);
/// assert!(shap.exact);
/// ```
///
/// # References
///
/// - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting
///   Model Predictions". NeurIPS 2017.
pub fn shap_kernel<F>(
    predict_fn: F,
    x: ArrayView2<f64>,
    background: ArrayView2<f64>,
    config: &ShapConfig,
    feature_names: Option<Vec<String>>,
) -> EconResult<ShapResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    kernel_shap(predict_fn, x, background, config, feature_names)
}

/// Compute SHAP values for a RandomForestModel (includes stored trees).
///
/// TreeSHAP ([`shap_values_model_with`] with [`ShapMethod::Tree`]): the exact Shapley values of
/// the trees' path-dependent game `E[f(x) | x_S]`.
///
/// This is the recommended way to compute SHAP values when using
/// `random_forest_with_trees()` which retains the tree structures.
///
/// # Arguments
/// * `model` - A trained Random Forest model with stored trees
/// * `x` - Input data matrix (n_samples x n_features)
/// * `config` - SHAP configuration
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{random_forest_with_trees, shap_values_model, ShapConfig};
///
/// let model = random_forest_with_trees(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
/// let shap = shap_values_model(&model, x.view(), &ShapConfig::default())?;
/// println!("Feature importance: {:?}", shap.feature_importance);
/// ```
///
/// # References
///
/// - Lundberg et al. (2018). "Consistent Individualized Feature Attribution for
///   Tree Ensembles". arXiv:1802.03888.
pub fn shap_values_model(
    model: &super::trees::RandomForestModel,
    x: ArrayView2<f64>,
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    shap_values_model_with(model, x, None, ShapMethod::Tree, config)
}

/// SHAP values of a random forest by the chosen method (the explicit entry point).
///
/// - [`ShapMethod::Tree`]: TreeSHAP, the exact Shapley values of the trees' path-dependent game
///   `E[f(x) | x_S]` ([`shap_tree_ensemble`]); `background` must be `None` (the expectation uses
///   the trees' training cover).
/// - [`ShapMethod::Kernel`]: Kernel SHAP ([`kernel_shap`]) of the forest's mean prediction, the
///   interventional game `v(S) = mean_b f(x_S, b_notS)` over the rows of `background`
///   (required); exact with every coalition when `2^M - 2 <= n_samples` (M <= 11 at the default
///   budget).
///
/// `config.feature_perturbation` must be the method's own game, and `compute_interactions` is
/// TreeSHAP's only; anything else is `InvalidSpecification`.
///
/// # References
///
/// - Lundberg et al. (2020). Nature Machine Intelligence 2, 56-67 (TreeSHAP).
/// - Lundberg & Lee (2017). NeurIPS 2017 (Kernel SHAP).
pub fn shap_values_model_with(
    model: &super::trees::RandomForestModel,
    x: ArrayView2<f64>,
    background: Option<ArrayView2<f64>>,
    method: ShapMethod,
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    match method {
        ShapMethod::Tree => {
            if background.is_some() {
                return Err(spec(
                    "TreeSHAP does not use a background sample (the path-dependent expectation uses the trees' training cover)"
                        .into(),
                ));
            }
            shap_tree_ensemble(&model.trees, x, model.feature_names.clone(), config)
        }
        ShapMethod::Kernel => {
            let Some(bg) = background else {
                return Err(spec("Kernel SHAP needs a background sample".into()));
            };
            validate_explain_inputs(x, model.feature_names.as_deref())?;
            let trees = &model.trees;
            if trees.is_empty() {
                return Err(spec("SHAP: the forest has no trees".into()));
            }
            // the trees are predicted on rows of x's width: check their splits first
            for (t, tree) in trees.iter().enumerate() {
                validate_nodes(&extract_tree_structure(tree), x.ncols(), t)?;
            }
            let n_trees = trees.len() as f64;
            let forest_mean = |z: ArrayView2<f64>| -> Vec<f64> {
                z.outer_iter()
                    .map(|r| trees.iter().map(|t| t.predict_one(&r)).sum::<f64>() / n_trees)
                    .collect()
            };
            kernel_shap(forest_mean, x, bg, config, model.feature_names.clone())
        }
    }
}

/// Compute SHAP values for a Random Forest model using a Dataset.
///
/// Convenience wrapper around [`shap_values_model`] that extracts the feature matrix
/// from a [`Dataset`] using column names.
///
/// # Arguments
/// * `model` - A trained Random Forest model with stored trees
///   (from [`random_forest_with_trees`](super::random_forest_with_trees))
/// * `dataset` - Input dataset containing the feature columns
/// * `x_cols` - Names of the feature columns to explain
/// * `config` - SHAP configuration
pub fn run_shap_values_model(
    model: &super::trees::RandomForestModel,
    dataset: &Dataset,
    x_cols: &[&str],
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    use crate::linalg::design::DesignMatrix;

    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;

    shap_values_model(model, x.view(), config)
}

/// Compute Kernel SHAP values using a Dataset.
///
/// Convenience wrapper around [`kernel_shap`] that extracts the feature matrices
/// from [`Dataset`] objects using column names.
///
/// # Arguments
/// * `predict_fn` - Model prediction function
/// * `dataset` - Input dataset containing the observations to explain
/// * `background_dataset` - Background dataset for computing baseline expectations
/// * `x_cols` - Names of the feature columns
/// * `config` - SHAP configuration
pub fn run_kernel_shap<F>(
    predict_fn: F,
    dataset: &Dataset,
    background_dataset: &Dataset,
    x_cols: &[&str],
    config: &ShapConfig,
) -> EconResult<ShapResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    use crate::linalg::design::DesignMatrix;

    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    let bg_design = DesignMatrix::from_dataframe(background_dataset.df(), x_cols, false)?;
    let background = bg_design.data;

    kernel_shap(
        predict_fn,
        x.view(),
        background.view(),
        config,
        Some(feature_names),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use ndarray::{array, s};
    use p2a_validation::{Case, Check, Tol};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Exact algorithm against exact enumeration: observed <= 1.5e-14 (plan-time R and rustc checks).
    const EXACT_SHAP: Tol = Tol::new(1e-10, 1e-12);
    /// Kernel SHAP sampled at M = 12, default n_samples: 2x the largest of 100 R replications (0.07295).
    const KERNEL_MC_TOL: f64 = 0.15;

    fn case_matrix(ds: &crate::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&Array1::from_vec(v));
        }
        m
    }

    /// The case's node table (0-based preorder ids; x[feature] <= threshold goes left).
    fn case_nodes(case: &Case) -> Vec<TreeShapNode> {
        let (f, t, l, r, v, c) = (
            case.vector("node_feature"),
            case.vector("node_threshold"),
            case.vector("node_left"),
            case.vector("node_right"),
            case.vector("node_value"),
            case.vector("node_cover"),
        );
        (0..f.len())
            .map(|k| TreeShapNode {
                feature: f[k] as i32,
                threshold: t[k],
                left: l[k] as i32,
                right: r[k] as i32,
                value: v[k],
                n_samples: c[k] as usize,
            })
            .collect()
    }

    /// Leaf lookup on the node table (test-side, independent of the library's predictor).
    fn walk(nodes: &[TreeShapNode], x: ArrayView1<f64>) -> f64 {
        let mut j = 0usize;
        while nodes[j].feature >= 0 {
            j = if x[nodes[j].feature as usize] <= nodes[j].threshold {
                nodes[j].left
            } else {
                nodes[j].right
            } as usize;
        }
        nodes[j].value
    }

    /// Lundberg et al. (2020) Algorithm 1 (EXPVALUE): E[f(x) | x_S] for the coalition bitmask `mask`.
    fn expvalue(nodes: &[TreeShapNode], j: usize, x: ArrayView1<f64>, mask: usize) -> f64 {
        let n = &nodes[j];
        if n.feature < 0 {
            return n.value;
        }
        let (l, r, f) = (n.left as usize, n.right as usize, n.feature as usize);
        if (mask >> f) & 1 == 1 {
            return expvalue(nodes, if x[f] <= n.threshold { l } else { r }, x, mask);
        }
        let (cl, cr) = (nodes[l].n_samples as f64, nodes[r].n_samples as f64);
        (cl * expvalue(nodes, l, x, mask) + cr * expvalue(nodes, r, x, mask)) / (cl + cr)
    }

    fn fact(k: usize) -> f64 {
        (1..=k).map(|i| i as f64).product()
    }

    /// Shapley values of the game `v` (coalition bitmask -> value) by enumeration of all 2^m coalitions.
    fn bruteforce_shapley(m: usize, v: &dyn Fn(usize) -> f64) -> Vec<f64> {
        let vals: Vec<f64> = (0..1usize << m).map(v).collect();
        (0..m)
            .map(|i| {
                (0..1usize << m)
                    .filter(|s| (s >> i) & 1 == 0)
                    .map(|s| {
                        let k = s.count_ones() as usize;
                        fact(k) * fact(m - k - 1) / fact(m) * (vals[s | 1 << i] - vals[s])
                    })
                    .sum()
            })
            .collect()
    }

    /// Shapley interaction index (Lundberg et al. 2020, eqs. 8-9) by enumeration.
    fn bruteforce_interactions(m: usize, v: &dyn Fn(usize) -> f64) -> Array2<f64> {
        let vals: Vec<f64> = (0..1usize << m).map(v).collect();
        let phi = bruteforce_shapley(m, v);
        let mut out = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    continue;
                }
                out[[i, j]] = (0..1usize << m)
                    .filter(|s| (s >> i) & 1 == 0 && (s >> j) & 1 == 0)
                    .map(|s| {
                        let k = s.count_ones() as usize;
                        fact(k) * fact(m - k - 2) / (2.0 * fact(m - 1))
                            * (vals[s | 1 << i | 1 << j] - vals[s | 1 << i] - vals[s | 1 << j]
                                + vals[s])
                    })
                    .sum();
            }
        }
        for i in 0..m {
            out[[i, i]] = phi[i] - (0..m).filter(|&j| j != i).map(|j| out[[i, j]]).sum::<f64>();
        }
        out
    }

    /// y = 4 x0 x1 + sin(6 x0) + x2^2 + noise; x ~ U(0, 1) (noisy: CLAUDE.md test-data rule).
    fn random_regression(n: usize, m: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x: Array2<f64> = Array2::from_shape_fn((n, m), |_| rng.gen_range(0.0..1.0));
        let y = Array1::from_shape_fn(n, |i| {
            4.0 * x[[i, 0]] * x[[i, 1]]
                + (6.0 * x[[i, 0]]).sin()
                + x[[i, 2]] * x[[i, 2]]
                + 0.2 * (rng.gen_range(0.0..1.0) - 0.5)
        });
        (x, y)
    }

    fn leaf_expectation(nodes: &[TreeShapNode]) -> f64 {
        nodes
            .iter()
            .filter(|n| n.feature < 0)
            .map(|n| n.value * n.n_samples as f64)
            .sum::<f64>()
            / nodes[0].n_samples as f64
    }

    #[test]
    fn test_shap_method_parses_aliases_and_rejects_unknown() {
        ShapMethod::choices().validate().unwrap();
        for s in ["tree", "TreeSHAP", "tree-shap", "tree_shap"] {
            assert_eq!(s.parse::<ShapMethod>().unwrap(), ShapMethod::Tree);
        }
        for s in ["kernel", "KernelSHAP", "kernel shap"] {
            assert_eq!(s.parse::<ShapMethod>().unwrap(), ShapMethod::Kernel);
        }
        let e = "permutation".parse::<ShapMethod>().unwrap_err().to_string();
        assert!(
            e.contains("unknown method 'permutation'") && e.contains("tree, kernel"),
            "{e}"
        );
        assert_eq!(ShapMethod::Kernel.to_string(), "kernel");
    }

    #[test]
    fn test_feature_perturbation_parses_aliases_and_rejects_unknown() {
        use FeaturePerturbation::*;
        FeaturePerturbation::choices().validate().unwrap();
        for (s, v) in [
            ("interventional", Interventional),
            ("Marginal", Interventional),
            ("independent", Interventional),
            ("tree_path_dependent", TreePathDependent),
            ("tree-path-dependent", TreePathDependent),
            ("path_dependent", TreePathDependent),
            ("observational", Observational),
            ("conditional", Observational),
            ("correlation_dependent", Observational),
        ] {
            assert_eq!(s.parse::<FeaturePerturbation>().unwrap(), v, "{s}");
        }
        let e = "causal"
            .parse::<FeaturePerturbation>()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("unknown feature_perturbation 'causal'")
                && e.contains("interventional, tree_path_dependent, observational"),
            "{e}"
        );
        assert_eq!(
            FeaturePerturbation::default_for(ShapMethod::Tree),
            TreePathDependent
        );
        assert_eq!(
            FeaturePerturbation::default_for(ShapMethod::Kernel),
            Interventional
        );
    }

    #[test]
    fn test_additivity_not_checked_is_none() {
        // Before: check_additivity = false reported Some(true) for a check that never ran.
        let f = |z: ArrayView2<f64>| {
            z.outer_iter()
                .map(|r| r[0] + 2.0 * r[1])
                .collect::<Vec<f64>>()
        };
        let mut config = ShapConfig::default();
        config.check_additivity = false;
        let r = kernel_shap(
            f,
            array![[1.0, 2.0]].view(),
            array![[0.0, 0.0], [1.0, 1.0]].view(),
            &config,
            None,
        )
        .unwrap();
        assert_eq!(r.additivity_check_passed, None);
        assert_eq!(r.max_additivity_error, None);
    }

    #[test]
    fn test_shap_inputs_are_validated() {
        let f = |z: ArrayView2<f64>| z.outer_iter().map(|r| r.sum()).collect::<Vec<f64>>();
        let bg = array![[0.0, 0.0], [1.0, 1.0]];
        let c = ShapConfig::default();
        let err = |r: EconResult<ShapResult>| r.unwrap_err().to_string();
        assert!(
            err(kernel_shap(
                f,
                Array2::<f64>::zeros((0, 2)).view(),
                bg.view(),
                &c,
                None
            ))
            .contains("no rows to explain")
        );
        assert!(
            err(kernel_shap(
                f,
                array![[1.0, f64::NAN]].view(),
                bg.view(),
                &c,
                None
            ))
            .contains("row 0, column 1")
        );
        assert!(
            err(kernel_shap(
                f,
                array![[1.0, 2.0]].view(),
                array![[0.0, 0.0, 0.0]].view(),
                &c,
                None
            ))
            .contains("3 columns but x has 2")
        );
        assert!(
            err(kernel_shap(
                f,
                array![[1.0, 2.0]].view(),
                Array2::<f64>::zeros((0, 2)).view(),
                &c,
                None
            ))
            .contains("no rows")
        );
        assert!(
            err(kernel_shap(
                f,
                array![[1.0, 2.0]].view(),
                bg.view(),
                &c,
                Some(vec!["a".into()])
            ))
            .contains("1 feature names for 2 features")
        );
        let short = |z: ArrayView2<f64>| vec![0.0; z.nrows().saturating_sub(1)];
        assert!(
            err(kernel_shap(
                short,
                array![[1.0, 2.0]].view(),
                bg.view(),
                &c,
                None
            ))
            .contains("returned 1 values for 2 rows")
        );
        let nan = |z: ArrayView2<f64>| vec![f64::NAN; z.nrows()];
        assert!(
            err(kernel_shap(
                nan,
                array![[1.0, 2.0]].view(),
                bg.view(),
                &c,
                None
            ))
            .contains("non-finite value")
        );
        let mut c0 = ShapConfig::default();
        c0.n_samples = Some(0);
        assert!(
            err(kernel_shap(
                f,
                array![[1.0, 2.0]].view(),
                bg.view(),
                &c0,
                None
            ))
            .contains("n_samples must be at least 1")
        );
        // Tree path: a model on 3 features explained on 2 columns (before: index out of bounds).
        let (x, y) = random_regression(40, 3, 5);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(3),
            Some(3),
            Some(2),
            Some("all"),
            Some(1),
            None,
        )
        .unwrap();
        let e = err(shap_values_model(&model, x.slice(s![.., 0..2]), &c));
        assert!(e.contains("splits on feature 2 but x has 2 columns"), "{e}");
    }

    #[test]
    fn test_leaf_weights_match_unwound_path_sum() {
        // The batched leaf step against the per-element UNWOUND sum on random unique paths
        // (one fractions 0 or 1, as TreeSHAP builds them; noisy zero fractions), including
        // paths longer than one lane group and all-hot / all-cold paths.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let inv: Vec<f64> = (0..16)
            .map(|k| if k == 0 { 0.0 } else { 1.0 / k as f64 })
            .collect();
        for case in 0..400 {
            let len = 2 + case % 12;
            let mut path = vec![PathElement::default(); len];
            for k in 0..len {
                let (zero, one) = if k == 0 {
                    (1.0, 1.0)
                } else {
                    let one = match case % 5 {
                        0 => 1.0,
                        1 => 0.0,
                        _ => f64::from(rng.gen_range(0..2u8)),
                    };
                    (rng.gen_range(0.05..1.0), one)
                };
                let feature = if k == 0 { -1 } else { k as i32 - 1 };
                extend_path(&mut path, &inv, k, zero, one, feature);
            }
            let scale = rng.gen_range(-2.0..2.0);
            let mut want = vec![0.0; len];
            for k in 1..len {
                let el = path[k];
                want[k - 1] = unwound_path_sum(&path, &inv, len, k) * (el.one - el.zero) * scale;
            }
            let mut got = vec![0.0; len];
            leaf_weights(&path, &inv, &mut got, scale);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() <= 1e-13 * w.abs().max(1e-3),
                    "case {case}, len {len}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn test_treeshap_matches_bruteforce_shapley() {
        // Random CART trees (depth 6, 60 rows, 5 features: features repeat on paths), 10 training
        // rows and 10 points outside [0, 1]; TreeSHAP against enumeration of E[f(x) | x_S] over all
        // 32 coalitions.
        for seed in [1u64, 2, 3] {
            let (x, y) = random_regression(60, 5, seed);
            let mut tree = super::super::trees::DecisionTree::new(6, 2, None);
            let mut state = seed;
            tree.fit(x.view(), y.view(), None, &mut state);
            let nodes = extract_tree_structure(&tree);
            let mut rng = ChaCha8Rng::seed_from_u64(100 + seed);
            let mut points: Vec<Array1<f64>> = (0..10).map(|i| x.row(i).to_owned()).collect();
            points.extend((0..10).map(|_| Array1::from_shape_fn(5, |_| rng.gen_range(-0.1..1.1))));
            for p in &points {
                let fast = tree_shap_single(&nodes, p.view(), 5);
                let exact = bruteforce_shapley(5, &|mask| expvalue(&nodes, 0, p.view(), mask));
                for j in 0..5 {
                    assert!(
                        EXACT_SHAP.allows(fast[j], exact[j]),
                        "seed {seed}, feature {j}: {} vs {}",
                        fast[j],
                        exact[j]
                    );
                }
                let total: f64 = fast.sum();
                assert!((total + leaf_expectation(&nodes) - walk(&nodes, p.view())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_validate_treeshap_rpart_tree() {
        let (case, ds) = crate::test_support::load_case("shap_tree_rpart");
        let nodes = case_nodes(&case);
        let x = case_matrix(&ds, &["x0", "x1", "x2", "x3"]);
        let phi: Vec<f64> = x
            .rows()
            .into_iter()
            .flat_map(|r| tree_shap_single(&nodes, r, 4).to_vec())
            .collect();
        let pred: Vec<f64> = x.rows().into_iter().map(|r| walk(&nodes, r)).collect();
        let x_new = Array2::from_shape_vec((5, 4), case.vector("x_new")).unwrap();
        let phi_new: Vec<f64> = x_new
            .rows()
            .into_iter()
            .flat_map(|r| tree_shap_single(&nodes, r, 4).to_vec())
            .collect();
        let pred_new: Vec<f64> = x_new.rows().into_iter().map(|r| walk(&nodes, r)).collect();
        Check::new(&case)
            .vector("pred", &pred, Tol::exact()) // the leaf values themselves
            .vector("pred_new", &pred_new, Tol::exact())
            .scalar("expected_value", leaf_expectation(&nodes), EXACT_SHAP)
            .vector("phi", &phi, EXACT_SHAP)
            .vector("phi_new", &phi_new, EXACT_SHAP)
            .finish();
    }

    #[test]
    fn test_extract_tree_matches_predict_one() {
        let (x, y) = random_regression(60, 5, 9);
        let mut tree = super::super::trees::DecisionTree::new(6, 2, None);
        let mut state = 9u64;
        tree.fit(x.view(), y.view(), None, &mut state);
        let nodes = extract_tree_structure(&tree);
        validate_nodes(&nodes, 5, 0).unwrap();
        assert_eq!(nodes[0].n_samples, 60);
        for r in x.rows() {
            assert_eq!(predict_nodes(&nodes, &r.to_vec()), tree.predict_one(&r));
            assert_eq!(walk(&nodes, r), tree.predict_one(&r));
        }
    }

    #[test]
    fn test_treeshap_additivity_new_rows() {
        // 150 training rows, 155 explained (5 new): before, predictions[150] panicked.
        let (x, y) = random_regression(150, 3, 7);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(20),
            Some(5),
            Some(5),
            Some("sqrt"),
            Some(42),
            None,
        )
        .unwrap();
        let (x_new, _) = random_regression(5, 3, 8);
        let x_all = ndarray::concatenate(Axis(0), &[x.view(), x_new.view()]).unwrap();
        let shap = shap_values_model(&model, x_all.view(), &ShapConfig::default()).unwrap();
        let t = model.trees.len() as f64;
        let base: f64 = model
            .trees
            .iter()
            .map(|tr| leaf_expectation(&extract_tree_structure(tr)))
            .sum::<f64>()
            / t;
        assert!(
            (shap.base_value - base).abs() < 1e-12,
            "base {} vs E[f] {}",
            shap.base_value,
            base
        );
        for i in 0..x_all.nrows() {
            let f: f64 = model
                .trees
                .iter()
                .map(|tr| tr.predict_one(&x_all.row(i)))
                .sum::<f64>()
                / t;
            let err = (shap.base_value + shap.shap_values.row(i).sum() - f).abs();
            assert!(err < 1e-10, "row {i}: additivity error {err}");
        }
        assert_eq!(shap.additivity_check_passed, Some(true));
        assert!(shap.max_additivity_error.unwrap() < 1e-10);
        // the training rows alone: the flag was false before (max error 0.48 in the verifier's repro)
        let train = shap_values_model(&model, x.view(), &ShapConfig::default()).unwrap();
        assert_eq!(train.additivity_check_passed, Some(true));
        assert!(
            train
                .to_string()
                .contains("Method: TreeSHAP (tree path-dependent; exact)")
        );
        assert!(
            train
                .to_string()
                .contains("Additivity check: PASSED (max error: ")
        );
    }

    #[test]
    fn test_tree_path_settings_are_honoured() {
        // Before: n_samples (the MCP field reaches this path) and feature_perturbation were ignored
        // silently.
        let (x, y) = random_regression(40, 3, 15);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(3),
            Some(3),
            Some(2),
            Some("all"),
            Some(1),
            None,
        )
        .unwrap();
        let mut c = ShapConfig::default();
        c.n_samples = Some(50);
        let r = shap_values_model(&model, x.view(), &c).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("n_samples = 50 is a Kernel SHAP setting")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Warning: n_samples = 50"));
        // n_samples = 0 is refused here as on the Kernel path (before: only a warning)
        c.n_samples = Some(0);
        let e = shap_values_model(&model, x.view(), &c)
            .unwrap_err()
            .to_string();
        assert!(e.contains("n_samples must be at least 1"), "{e}");
        let mut interv = ShapConfig::default();
        interv.feature_perturbation = Some(FeaturePerturbation::Interventional);
        let e = shap_values_model(&model, x.view(), &interv)
            .unwrap_err()
            .to_string();
        assert!(e.contains("come from the Kernel path"), "{e}");
        let mut obs = ShapConfig::default();
        obs.feature_perturbation = Some(FeaturePerturbation::Observational);
        assert!(
            shap_values_model(&model, x.view(), &obs)
                .unwrap_err()
                .to_string()
                .contains("not implemented")
        );
    }

    #[test]
    fn test_treeshap_rows_do_not_depend_on_the_thread_count() {
        // Rows run on rayon with per-worker scratch: one thread and four give identical bits
        // (values, predictions, interactions), and a cancelled budget stops the parallel rows.
        let (x, y) = random_regression(120, 4, 21);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(8),
            Some(5),
            Some(2),
            Some("all"),
            Some(5),
            None,
        )
        .unwrap();
        let mut config = ShapConfig::default();
        config.compute_interactions = true;
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| shap_values_model(&model, x.view(), &config).unwrap())
        };
        let (one, four) = (run(1), run(4));
        assert_eq!(one.shap_values, four.shap_values);
        assert_eq!(one.predictions, four.predictions);
        assert_eq!(one.base_value, four.base_value);
        assert_eq!(one.interaction_values, four.interaction_values);
        assert_eq!(one.interaction_values.as_ref().map(Vec::len), Some(120));
        let token = crate::resource::CancelToken::new();
        token.cancel();
        let e = crate::resource::with_budget(crate::resource::Budget::new(None, token), || {
            shap_values_model(&model, x.view(), &config)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e}");
    }

    #[test]
    fn test_tree_interactions_sum_to_shap_values() {
        let (x, y) = random_regression(80, 4, 11);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(10),
            Some(4),
            Some(2),
            Some("all"),
            Some(3),
            None,
        )
        .unwrap();
        let mut config = ShapConfig::default();
        config.compute_interactions = true;
        let r = shap_values_model(&model, x.slice(s![0..20, ..]), &config).unwrap();
        let inter = r.interaction_values.as_ref().expect("interaction values"); // before: None
        assert_eq!(inter.len(), 20);
        for (i, m) in inter.iter().enumerate() {
            assert_eq!(m.dim(), (4, 4));
            for a in 0..4 {
                assert!(
                    (m.row(a).sum() - r.shap_values[[i, a]]).abs() < 1e-10,
                    "row {i} feature {a}"
                );
                for b in 0..4 {
                    assert!((m[[a, b]] - m[[b, a]]).abs() < 1e-12);
                }
            }
        }
        assert!(
            r.to_string()
                .contains("Interaction values computed: Yes (TreeSHAP, 20 x 4 x 4)")
        );
    }

    #[test]
    fn test_treeshap_interactions_match_bruteforce() {
        for seed in [4u64, 5] {
            let (x, y) = random_regression(60, 4, seed);
            let mut tree = super::super::trees::DecisionTree::new(5, 2, None);
            let mut state = seed;
            tree.fit(x.view(), y.view(), None, &mut state);
            let nodes = extract_tree_structure(&tree);
            for i in 0..8 {
                let p = x.row(i);
                let fast = tree_shap_interactions_single(&nodes, p, 4);
                let exact = bruteforce_interactions(4, &|mask| expvalue(&nodes, 0, p, mask));
                for a in 0..4 {
                    for b in 0..4 {
                        assert!(
                            EXACT_SHAP.allows(fast[[a, b]], exact[[a, b]]),
                            "({a},{b}): {} vs {}",
                            fast[[a, b]],
                            exact[[a, b]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_validate_treeshap_interactions_rpart_tree() {
        let (case, ds) = crate::test_support::load_case("shap_tree_rpart");
        let nodes = case_nodes(&case);
        let x = case_matrix(&ds, &["x0", "x1", "x2", "x3"]);
        // [point, i, j], row-major as the case
        let inter: Vec<f64> = (0..10)
            .flat_map(|i| {
                tree_shap_interactions_single(&nodes, x.row(i), 4)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        Check::new(&case)
            .vector("interactions", &inter, EXACT_SHAP)
            .finish();
    }

    fn f1(z: ArrayView2<f64>) -> Vec<f64> {
        z.outer_iter()
            .map(|r| (r[0] / 2.0).exp() + r[0] * r[0])
            .collect()
    }

    fn f4(z: ArrayView2<f64>) -> Vec<f64> {
        z.outer_iter()
            .map(|r| r[0] * r[1] * r[2] + 2.0 * r[3] + 0.5 * r[0] * r[0])
            .collect()
    }

    fn f11_row(r: ArrayView1<f64>) -> f64 {
        r[0] * r[1] * r[2]
            + r[3].sin() * r[4]
            + r[5] * r[6] * r[7] * r[8]
            + (5..=10).map(|j| (j as f64 - 4.0) / 4.0 * r[j]).sum::<f64>()
    }

    fn f11(z: ArrayView2<f64>) -> Vec<f64> {
        z.outer_iter().map(f11_row).collect()
    }

    fn f12(z: ArrayView2<f64>) -> Vec<f64> {
        z.outer_iter()
            .map(|r| f11_row(r) + 0.5 * r[11] * r[0])
            .collect()
    }

    /// The case's 20 x 12 matrix (rows 1-4 explained, 5-20 background); the test names the case id.
    fn kernel_case(case_id: &str) -> (Case, Array2<f64>) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let names: Vec<String> = (0..12).map(|j| format!("x{j}")).collect();
        let cols: Vec<&str> = names.iter().map(String::as_str).collect();
        let x = case_matrix(&ds, &cols);
        (case, x)
    }

    /// Row-major (logical order).
    fn flat(a: &Array2<f64>) -> Vec<f64> {
        a.iter().copied().collect()
    }

    #[test]
    fn test_kernel_shap_exact_coalitions_match_bruteforce() {
        // ML-100's example: exact (1/3, 1/3, 1/3, 1); before (seed 42): (0.2787, 0.2664, 0.3106, 1.1442).
        let f = |z: ArrayView2<f64>| {
            z.outer_iter()
                .map(|r| r[0] * r[1] * r[2] + r[3])
                .collect::<Vec<f64>>()
        };
        let r = kernel_shap(
            f,
            array![[1.0, 1.0, 1.0, 1.0]].view(),
            Array2::zeros((1, 4)).view(),
            &ShapConfig::default(),
            None,
        )
        .unwrap();
        for (j, e) in [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0].iter().enumerate() {
            assert!(
                EXACT_SHAP.allows(r.shap_values[[0, j]], *e),
                "feature {j}: {}",
                r.shap_values[[0, j]]
            );
        }
        // A non-additive model on a random background against enumeration of the interventional game.
        let g = |z: ArrayView2<f64>| {
            z.outer_iter()
                .map(|r| r[0] * r[1] + r[2] * r[3] * r[4] + (r[4]).exp())
                .collect::<Vec<f64>>()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let bg: Array2<f64> = Array2::from_shape_fn((7, 5), |_| rng.gen_range(-1.0..1.0));
        let xs: Array2<f64> = Array2::from_shape_fn((3, 5), |_| rng.gen_range(-1.0..1.0));
        let r = kernel_shap(g, xs.view(), bg.view(), &ShapConfig::default(), None).unwrap();
        for i in 0..3 {
            let v = |mask: usize| {
                let mut z = bg.clone();
                for j in 0..5 {
                    if (mask >> j) & 1 == 1 {
                        z.column_mut(j).fill(xs[[i, j]]);
                    }
                }
                g(z.view()).iter().sum::<f64>() / 7.0
            };
            let exact = bruteforce_shapley(5, &v);
            for j in 0..5 {
                assert!(
                    EXACT_SHAP.allows(r.shap_values[[i, j]], exact[j]),
                    "row {i} feature {j}"
                );
            }
        }
    }

    #[test]
    fn test_validate_kernel_shap_sim_m4() {
        let (case, x) = kernel_case("shap_kernel_sim");
        let r = kernel_shap(
            f4,
            x.slice(s![0..4, 0..4]),
            x.slice(s![4..20, 0..4]),
            &ShapConfig::default(),
            None,
        )
        .unwrap();
        Check::new(&case)
            .vector("m4_phi", &flat(&r.shap_values), EXACT_SHAP)
            .scalar("m4_base", r.base_value, EXACT_SHAP)
            .finish();
    }

    #[test]
    fn test_validate_kernel_shap_sim_m11_enumerated() {
        // default n_samples = 2 * 11 + 2048 = 2070 >= 2^11 - 2 = 2046: every coalition, exact
        let (case, x) = kernel_case("shap_kernel_sim");
        let r = kernel_shap(
            f11,
            x.slice(s![0..4, 0..11]),
            x.slice(s![4..20, 0..11]),
            &ShapConfig::default(),
            None,
        )
        .unwrap();
        assert!(r.exact);
        assert_eq!(r.n_coalitions, Some(2046));
        Check::new(&case)
            .vector("m11_phi", &flat(&r.shap_values), EXACT_SHAP)
            .scalar("m11_base", r.base_value, EXACT_SHAP)
            .vector("m11_fx", &r.predictions, EXACT_SHAP)
            .finish();
    }

    #[test]
    fn test_validate_kernel_shap_sim_m12_sampled() {
        // default n_samples = 2072 < 4094: shap's design (sizes 1-3 complete, 4-8 sampled), seeded
        let (case, x) = kernel_case("shap_kernel_sim");
        let (ex, bg) = (x.slice(s![0..4, ..]), x.slice(s![4..20, ..]));
        let r = kernel_shap(f12, ex, bg, &ShapConfig::default(), None).unwrap();
        assert!(!r.exact);
        assert_eq!(r.n_coalitions, Some(2072));
        assert_eq!(r.seed, Some(SHAP_DEFAULT_SEED));
        let exact = case.vector("m12_phi");
        let fx = case.vector("m12_fx");
        let base = case.scalar("m12_base");
        for i in 0..4 {
            let s: f64 = r.shap_values.row(i).sum();
            assert!((s - (fx[i] - base)).abs() < 1e-10, "efficiency row {i}"); // exact by construction
            for j in 0..12 {
                let d = (r.shap_values[[i, j]] - exact[i * 12 + j]).abs();
                assert!(
                    d < KERNEL_MC_TOL,
                    "row {i} feature {j}: |diff| {d} >= {KERNEL_MC_TOL}"
                );
            }
        }
        Check::new(&case)
            .scalar("m12_base", r.base_value, EXACT_SHAP)
            .vector("m12_fx", &r.predictions, EXACT_SHAP)
            .finish();
        // seeds: identical seeds give identical values; another seed gives other samples
        let again = kernel_shap(f12, ex, bg, &ShapConfig::default(), None).unwrap();
        assert_eq!(again.shap_values, r.shap_values);
        let mut c7 = ShapConfig::default();
        c7.seed = Some(7);
        let other = kernel_shap(f12, ex, bg, &c7, None).unwrap();
        assert_ne!(other.shap_values, r.shap_values);
    }

    #[test]
    fn test_kernel_shap_single_feature() {
        // M = 1: phi = f(x) - E f (before: remainder by zero)
        let f = |z: ArrayView2<f64>| {
            z.outer_iter()
                .map(|r| (r[0] / 2.0).exp() + r[0] * r[0])
                .collect::<Vec<f64>>()
        };
        let bg = array![[-1.0], [0.5], [2.0]];
        let x = array![[0.3], [1.7]];
        let r = kernel_shap(f, x.view(), bg.view(), &ShapConfig::default(), None).unwrap();
        let ef = f(bg.view()).iter().sum::<f64>() / 3.0;
        let fx = f(x.view());
        for i in 0..2 {
            assert!((r.shap_values[[i, 0]] - (fx[i] - ef)).abs() < 1e-14);
        }
        assert!(r.exact);
        assert_eq!(r.n_coalitions, Some(0));
    }

    #[test]
    fn test_validate_kernel_shap_sim_m1() {
        let (case, x) = kernel_case("shap_kernel_sim");
        let r = kernel_shap(
            f1,
            x.slice(s![0..4, 0..1]),
            x.slice(s![4..20, 0..1]),
            &ShapConfig::default(),
            None,
        )
        .unwrap();
        Check::new(&case)
            .vector("m1_phi", &flat(&r.shap_values), EXACT_SHAP)
            .scalar("m1_base", r.base_value, EXACT_SHAP)
            .finish();
    }

    #[test]
    fn test_kernel_shap_zero_features_is_error() {
        let f = |z: ArrayView2<f64>| vec![1.0; z.nrows()];
        let e = kernel_shap(
            f,
            Array2::<f64>::zeros((2, 0)).view(),
            Array2::<f64>::zeros((3, 0)).view(),
            &ShapConfig::default(),
            None,
        )
        .unwrap_err(); // before: subtract-with-overflow panic
        assert!(e.to_string().contains("at least one feature column"), "{e}");
    }

    #[test]
    fn test_kernel_shap_non_finite_coalition_value_is_error() {
        // Finite predictions whose differences overflow (f = +-1e308, one background row): the
        // regression response is not finite. Before: the enumerated design (factored once)
        // returned Ok with NaN SHAP values; the sampled design failed with linalg's generic
        // "y contains a non-finite value".
        let f = |z: ArrayView2<f64>| {
            z.outer_iter()
                .map(|r| if r[0] > 0.0 { 1e308 } else { -1e308 })
                .collect::<Vec<f64>>()
        };
        let x = array![[1.0, 0.5]];
        let bg = array![[-1.0, 0.5]];
        for n_samples in [None, Some(1)] {
            let mut c = ShapConfig::default();
            c.n_samples = n_samples;
            let e = kernel_shap(f, x.view(), bg.view(), &c, None)
                .map(|r| r.shap_values)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("Kernel SHAP: row 0: the value of coalition"),
                "{n_samples:?}: {e}"
            );
        }
    }

    #[test]
    fn test_kernel_shap_small_budget_uses_the_sampled_design() {
        // n_samples = 10 below the 14 coalitions of M = 4: shap's sampled design with that budget
        let x = array![[0.3, -1.0, 2.0, 0.5]];
        let bg = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0]];
        let mut c = ShapConfig::default();
        c.n_samples = Some(10);
        let r = kernel_shap(f4, x.view(), bg.view(), &c, None).unwrap();
        assert!(!r.exact);
        assert_eq!(r.n_coalitions, Some(10));
        let delta = f4(x.view())[0] - f4(bg.view()).iter().sum::<f64>() / 2.0;
        assert!((r.shap_values.row(0).sum() - delta).abs() < 1e-10);
        let again = kernel_shap(f4, x.view(), bg.view(), &c, None).unwrap();
        assert_eq!(again.shap_values, r.shap_values);
        c.seed = Some(9);
        let other = kernel_shap(f4, x.view(), bg.view(), &c, None).unwrap();
        assert_ne!(other.shap_values, r.shap_values);
        assert!(
            r.to_string()
                .contains(
                    "10 of 14 coalitions evaluated (complete sizes enumerated, the rest sampled), seed 42; Monte Carlo estimate"
                )
        );
    }

    #[test]
    fn test_shap_display_reports_method_and_notes() {
        let f = |z: ArrayView2<f64>| z.outer_iter().map(|r| r[0] * r[1]).collect::<Vec<f64>>();
        let r = kernel_shap(
            f,
            array![[1.0, 2.0]].view(),
            array![[0.0, 0.0], [1.0, 1.0]].view(),
            &ShapConfig::default(),
            None,
        )
        .unwrap();
        let text = r.to_string();
        assert!(
            text.contains("Method: Kernel SHAP (interventional; all 2 coalitions, exact)"),
            "{text}"
        );
        assert!(
            text.contains("Additivity check: PASSED (max error: "),
            "{text}"
        );
    }

    #[test]
    fn test_kernel_shap_rejects_tree_settings() {
        let f = |z: ArrayView2<f64>| z.outer_iter().map(|r| r.sum()).collect::<Vec<f64>>();
        let (x, bg) = (array![[1.0, 2.0]], array![[0.0, 0.0], [1.0, 1.0]]);
        for (p, frag) in [
            (
                FeaturePerturbation::Observational,
                "observational (conditional expectations) is not implemented",
            ),
            (
                FeaturePerturbation::TreePathDependent,
                "tree_path_dependent needs TreeSHAP",
            ),
        ] {
            let mut c = ShapConfig::default();
            c.feature_perturbation = Some(p);
            let e = kernel_shap(f, x.view(), bg.view(), &c, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains(frag), "{e}");
        }
        let mut c = ShapConfig::default();
        c.compute_interactions = true;
        assert!(
            kernel_shap(f, x.view(), bg.view(), &c, None)
                .unwrap_err()
                .to_string()
                .contains("TreeSHAP only")
        );
    }

    fn neumaier_sum(v: &[f64]) -> f64 {
        let (mut sum, mut c) = (0.0f64, 0.0f64);
        for &x in v {
            let t = sum + x;
            c += if sum.abs() >= x.abs() {
                (sum - t) + x
            } else {
                (x - t) + sum
            };
            sum = t;
        }
        sum + c
    }

    #[test]
    fn test_kernel_design_complete_size_must_fit_the_budget() {
        // shap's tolerance 1 - 1e-8 alone admits a size with more than 1e8 subsets when the
        // budget is a few coalitions short of it (before: `samples_left -= n_subsets` then
        // underflowed: a panic in debug builds, a wrapped budget in release builds).
        let n_subsets = 2.0e8;
        assert!(199_999_999.0 / n_subsets >= 1.0 - 1e-8); // the tolerance test alone passes
        assert!(!size_fits(199_999_999, 1.0, n_subsets));
        assert!(size_fits(200_000_000, 1.0, n_subsets));
        // a budget one below a complete design at M = 4 (2^4 - 2 = 14): size 1 (8 coalitions
        // with complements) is enumerated, size 2 (6) is not, and the design stays in budget
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let d = sampled_coalitions(4, 13, &mut rng).unwrap();
        assert_eq!((d.num_full, d.n_fixed, d.exact), (1, 8, false));
        assert!(d.z.nrows() <= 13, "{}", d.z.nrows());
        let d = sampled_coalitions(4, 14, &mut rng).unwrap();
        assert_eq!((d.num_full, d.n_fixed, d.exact), (2, 14, true));
    }

    #[test]
    fn test_kernel_design_matches_shap_explainer() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (m, n, rows, n_fixed, num_full, fixed_w) in [
            (
                12usize,
                2072usize,
                2072usize,
                596usize,
                3usize,
                0.7070994254040688f64,
            ),
            (20, 2088, 2088, 420, 2, 0.4532990833373862),
            (12, 100, 100, 24, 1, 0.3612428474155129),
        ] {
            let d = sampled_coalitions(m, n, &mut rng).unwrap();
            // rows: seeded, so deterministic; the R port reached the full budget in all 100 runs
            // at (12, 2072)
            assert_eq!(
                (d.z.nrows(), d.n_fixed, d.num_full, d.exact),
                (rows, n_fixed, num_full, false),
                "M = {m}"
            );
            // compensated sums: a plain sum of ~2000 weights carries ~1e-14 of rounding (R's
            // reference sums in long double)
            let wf = neumaier_sum(&d.weights[..d.n_fixed]);
            let wt = neumaier_sum(&d.weights);
            assert!(
                (wf - fixed_w).abs() < 1e-14 && (wt - 1.0).abs() < 1e-14,
                "M = {m}: {wf} {wt}"
            );
            let distinct: std::collections::HashSet<Vec<u8>> =
                d.z.rows()
                    .into_iter()
                    .map(|r| r.iter().map(|&v| v as u8).collect())
                    .collect();
            assert_eq!(distinct.len(), d.z.nrows()); // every row is a distinct coalition
            assert!(d.z.rows().into_iter().all(|r| {
                let s = r.sum();
                s >= 1.0 && s <= (m - 1) as f64
            }));
        }
        let d = sampled_coalitions(11, 2070, &mut rng).unwrap();
        assert!(d.exact);
        assert_eq!((d.z.nrows(), d.n_fixed, d.num_full), (2046, 2046, 5));
        let e = enumerate_coalitions(4).unwrap();
        assert_eq!(e.z.nrows(), 14);
        assert!(
            (shap_kernel_weight(4, 1) - 0.25).abs() < 1e-15
                && (shap_kernel_weight(4, 2) - 0.125).abs() < 1e-15
        );
    }

    #[test]
    fn test_shap_values_model_with_kernel_matches_interventional_enumeration() {
        let (x, y) = random_regression(80, 3, 13);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(8),
            Some(4),
            Some(2),
            Some("all"),
            Some(5),
            None,
        )
        .unwrap();
        let bg = x.slice(s![40..60, ..]);
        let ex = x.slice(s![0..5, ..]);
        let r = shap_values_model_with(
            &model,
            ex,
            Some(bg),
            ShapMethod::Kernel,
            &ShapConfig::default(),
        )
        .unwrap();
        assert_eq!(
            (r.method, r.feature_perturbation, r.exact),
            (
                ShapMethod::Kernel,
                FeaturePerturbation::Interventional,
                true
            )
        );
        let forest = |z: ArrayView2<f64>| -> Vec<f64> {
            z.outer_iter()
                .map(|row| {
                    model.trees.iter().map(|t| t.predict_one(&row)).sum::<f64>()
                        / model.trees.len() as f64
                })
                .collect()
        };
        for i in 0..5 {
            let v = |mask: usize| {
                let mut z = bg.to_owned();
                for j in 0..3 {
                    if (mask >> j) & 1 == 1 {
                        z.column_mut(j).fill(ex[[i, j]]);
                    }
                }
                forest(z.view()).iter().sum::<f64>() / 20.0
            };
            let exact = bruteforce_shapley(3, &v);
            for j in 0..3 {
                assert!(
                    EXACT_SHAP.allows(r.shap_values[[i, j]], exact[j]),
                    "row {i} feature {j}"
                );
            }
        }
        // the tree method through the same entry point is TreeSHAP
        let t = shap_values_model_with(&model, ex, None, ShapMethod::Tree, &ShapConfig::default())
            .unwrap();
        let direct = shap_values_model(&model, ex, &ShapConfig::default()).unwrap();
        assert_eq!(t.shap_values, direct.shap_values);
        assert_eq!(t.method, ShapMethod::Tree);
    }

    #[test]
    fn test_shap_values_model_with_refuses_mismatched_settings() {
        let (x, y) = random_regression(40, 3, 14);
        let model = super::super::trees::random_forest_with_trees(
            x.view(),
            y.view(),
            Some(3),
            Some(3),
            Some(2),
            Some("all"),
            Some(1),
            None,
        )
        .unwrap();
        let bg = x.slice(s![0..10, ..]);
        assert!(
            shap_values_model_with(
                &model,
                x.view(),
                Some(bg),
                ShapMethod::Tree,
                &ShapConfig::default()
            )
            .unwrap_err()
            .to_string()
            .contains("does not use a background sample")
        );
        assert!(
            shap_values_model_with(
                &model,
                x.view(),
                None,
                ShapMethod::Kernel,
                &ShapConfig::default()
            )
            .unwrap_err()
            .to_string()
            .contains("needs a background sample")
        );
        let mut inter = ShapConfig::default();
        inter.compute_interactions = true;
        assert!(
            shap_values_model_with(&model, x.view(), Some(bg), ShapMethod::Kernel, &inter)
                .unwrap_err()
                .to_string()
                .contains("TreeSHAP only")
        );
        let mut tpd = ShapConfig::default();
        tpd.feature_perturbation = Some(FeaturePerturbation::TreePathDependent);
        assert!(
            shap_values_model_with(&model, x.view(), Some(bg), ShapMethod::Kernel, &tpd)
                .unwrap_err()
                .to_string()
                .contains("tree_path_dependent needs TreeSHAP")
        );
    }

    #[test]
    fn test_shap_kernel_weight() {
        // Test that weights are computed correctly
        let m = 4; // 4 features

        // Weight for |z| = 1 should be higher than |z| = 2
        let w1 = shap_kernel_weight(m, 1);
        let w2 = shap_kernel_weight(m, 2);

        assert!(w1 > w2, "Weight for |z|=1 should be > weight for |z|=2");

        // Symmetry: weight for |z| should equal weight for |M-z|
        let w3 = shap_kernel_weight(m, 3);
        assert!((w1 - w3).abs() < 1e-10, "Weights should be symmetric");
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(4, 0) as i64, 1);
        assert_eq!(binomial_coefficient(4, 1) as i64, 4);
        assert_eq!(binomial_coefficient(4, 2) as i64, 6);
        assert_eq!(binomial_coefficient(4, 3) as i64, 4);
        assert_eq!(binomial_coefficient(4, 4) as i64, 1);
        assert_eq!(binomial_coefficient(10, 5) as i64, 252);
        assert_eq!(binomial_coefficient(30, 15), 155117520.0);
    }

    #[test]
    fn test_kernel_shap_simple() {
        // Simple linear model: f(x) = x[0] + 2*x[1]
        let predict_fn =
            |x: ArrayView2<f64>| x.outer_iter().map(|row| row[0] + 2.0 * row[1]).collect();

        let x = array![[1.0, 2.0]];
        let background = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let config = ShapConfig {
            n_samples: Some(1000),
            seed: Some(42),
            check_additivity: true,
            ..Default::default()
        };

        let result = kernel_shap(predict_fn, x.view(), background.view(), &config, None).unwrap();

        // SHAP values should approximate feature contributions
        // x[0] = 1 contributes ~1 (relative to background mean of 0.5)
        // x[1] = 2 contributes ~4 (relative to background mean of 1)

        assert_eq!(result.n_obs, 1);
        assert_eq!(result.n_features, 2);

        // f is linear: phi = (1 - 0.5, 2 * (2 - 0.5)) exactly
        assert!((result.shap_values[[0, 0]] - 0.5).abs() < 1e-12);
        assert!((result.shap_values[[0, 1]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_shap_summary() {
        let shap_values = array![[1.0, 2.0, 0.5], [1.2, 1.8, 0.3], [0.8, 2.2, 0.7],];

        let result = ShapResult {
            shap_values,
            base_value: 0.0,
            feature_names: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
            n_obs: 3,
            n_features: 3,
            feature_importance: vec![1.0, 2.0, 0.5],
            interaction_values: None,
            additivity_check_passed: Some(true),
            max_additivity_error: Some(0.001),
            method: ShapMethod::Tree,
            feature_perturbation: FeaturePerturbation::TreePathDependent,
            exact: true,
            n_coalitions: None,
            seed: None,
            predictions: vec![0.0; 3],
            warnings: Vec::new(),
        };

        let summary = shap_summary(&result);

        assert_eq!(summary.feature_names, vec!["A", "B", "C"]);
        assert_eq!(summary.importance_rank[1], 1); // B is most important
        assert_eq!(summary.importance_rank[0], 2); // A is second
        assert_eq!(summary.importance_rank[2], 3); // C is third
    }

    #[test]
    fn test_shap_config_default() {
        let config = ShapConfig::default();

        assert!(config.n_samples.is_none());
        assert_eq!(config.feature_perturbation, None);
        assert!(!config.compute_interactions);
        assert!(config.check_additivity);
    }

    #[test]
    fn test_tree_shap_random_forest() {
        use super::super::trees::random_forest_with_trees;

        // Create test data with a simple relationship: y = x0 + 2*x1 + noise
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 0.3],
            [3.0, 3.0, 0.7],
            [4.0, 2.0, 0.4],
            [5.0, 4.0, 0.6],
            [6.0, 3.0, 0.5],
            [7.0, 5.0, 0.8],
            [8.0, 4.0, 0.3],
            [9.0, 6.0, 0.9],
            [10.0, 5.0, 0.2],
            [1.5, 2.5, 0.4],
            [2.5, 1.5, 0.6],
            [3.5, 3.5, 0.5],
            [4.5, 2.5, 0.7],
            [5.5, 4.5, 0.3],
        ];

        // y = x0 + 2*x1 + small noise
        let y = array![
            5.1, 4.0, 9.2, 8.1, 13.0, 12.1, 17.0, 16.1, 21.0, 20.1, 6.4, 5.5, 10.3, 9.5, 14.2
        ];

        // Train Random Forest with stored trees
        let model = random_forest_with_trees(
            x.view(),
            y.view(),
            Some(10), // Small forest for testing
            Some(5),  // max_depth
            Some(2),  // min_samples_split
            Some("all"),
            Some(42),
            Some(vec![
                "X1".to_string(),
                "X2".to_string(),
                "Noise".to_string(),
            ]),
        )
        .unwrap();

        assert_eq!(model.n_trees, 10);
        assert!(model.base_value.is_finite());

        // Compute SHAP values
        let config = ShapConfig {
            seed: Some(42),
            check_additivity: true,
            ..Default::default()
        };

        let shap_result = shap_values_model(&model, x.view(), &config).unwrap();

        assert_eq!(shap_result.n_obs, 15);
        assert_eq!(shap_result.n_features, 3);
        assert!(shap_result.base_value.is_finite());

        // Feature importance should rank X2 (coefficient 2) higher than X1 (coefficient 1)
        // and both should be higher than Noise
        println!("Feature importance: {:?}", shap_result.feature_importance);

        // The predictors (X1, X2) should have higher importance than noise
        let noise_importance = shap_result.feature_importance[2];
        let x1_importance = shap_result.feature_importance[0];
        let x2_importance = shap_result.feature_importance[1];

        // At least one of the true predictors should be more important than noise
        assert!(
            x1_importance > noise_importance || x2_importance > noise_importance,
            "True predictors should generally be more important than noise"
        );

        assert_eq!(shap_result.additivity_check_passed, Some(true));
        assert!(shap_result.max_additivity_error.unwrap() < 1e-10);

        // Compute summary
        let summary = shap_summary(&shap_result);
        assert_eq!(summary.feature_names.len(), 3);
        assert_eq!(summary.n_obs, 15);
    }

    #[test]
    fn test_run_kernel_shap_dataset() {
        use polars::prelude::*;

        let df = df! {
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            "x2" => [2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5, 4.5, 5.5, 6.5],
        }
        .unwrap();

        let dataset = Dataset::new(df.clone());
        let background = Dataset::new(df);

        // Simple linear prediction function: y = x1 + x2
        let predict_fn = |x: ArrayView2<f64>| -> Vec<f64> {
            x.rows().into_iter().map(|row| row[0] + row[1]).collect()
        };

        let config = ShapConfig {
            seed: Some(42),
            ..Default::default()
        };

        let result =
            run_kernel_shap(predict_fn, &dataset, &background, &["x1", "x2"], &config).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.nrows(), 10);
        assert_eq!(result.shap_values.ncols(), 2);
        assert!(result.feature_names.is_some());
        assert_eq!(
            result.feature_names.as_ref().unwrap(),
            &["x1".to_string(), "x2".to_string()]
        );
        // f = x1 + x2: phi_ij = x_ij - mean_j(background)
        let means = [3.25, 4.25];
        let xs = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            [2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5, 4.5, 5.5, 6.5],
        ];
        for i in 0..10 {
            for j in 0..2 {
                assert!((result.shap_values[[i, j]] - (xs[j][i] - means[j])).abs() < 1e-12);
            }
        }
    }
}
