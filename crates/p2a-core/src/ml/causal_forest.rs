//! Causal Forests for heterogeneous treatment effects.
//!
//! Pure Rust implementation of causal forests based on the methodology from
//! Wager & Athey (2018). Estimates conditional average treatment effects (CATE)
//! using random forests adapted for causal inference.
//!
//! # Key Features
//!
//! - **Honest splitting**: Uses separate data for determining tree structure vs. estimation
//! - **Local centering**: Removes confounding by centering outcomes within leaves
//! - **Variance estimation**: Bootstrap-based uncertainty quantification
//!
//! # Mathematical Background
//!
//! ## Causal Setup
//!
//! For unit i with covariates Xᵢ, treatment Wᵢ ∈ {0,1}, and potential outcomes Y(0), Y(1):
//!
//! - **CATE (Conditional Average Treatment Effect)**: τ(x) = E[Y(1) - Y(0) | X = x]
//! - **ATE (Average Treatment Effect)**: τ = E[τ(X)]
//!
//! ## Causal Tree Algorithm
//!
//! 1. Split sample into "structure" sample I₁ and "estimation" sample I₂
//! 2. Use I₁ to determine tree structure by maximizing treatment effect heterogeneity
//! 3. Use I₂ to estimate treatment effects within each leaf
//!
//! The splitting criterion maximizes:
//!
//! Δ = Var(τ̂(Xₗₑft)) + Var(τ̂(Xᵣᵢght)) - Var(τ̂(Xparent))
//!
//! where τ̂ is the estimated treatment effect.
//!
//! ## Honest Estimation
//!
//! Within leaf L, the treatment effect is estimated as:
//!
//! τ̂(L) = Ȳ(L, W=1) - Ȳ(L, W=0)
//!
//! using only data from the estimation sample I₂.
//!
//! # References
//!
//! - Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
//!   Treatment Effects using Random Forests. *Journal of the American Statistical
//!   Association*, 113(523), 1228-1242. https://doi.org/10.1080/01621459.2017.1319839
//!
//! - Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
//!   *Annals of Statistics*, 47(2), 1148-1178. https://doi.org/10.1214/18-AOS1709
//!
//! - Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous
//!   causal effects. *Proceedings of the National Academy of Sciences*, 113(27),
//!   7353-7360. https://doi.org/10.1073/pnas.1510489113
//!
//! - R package `grf`: Tibshirani, J., Athey, S., & Wager, S. (2024).
//!   *grf: Generalized Random Forests*. https://grf-labs.github.io/grf/
//!
//! R equivalent: `grf::causal_forest()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::{default_mtry, heap_chunk_bytes, mul, partition_exact, pow2};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::resource::{checkpoint, try_with_capacity};
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

/// Configuration for causal forest estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestConfig {
    /// Number of trees (default: 2000)
    pub n_trees: usize,
    /// Minimum node size for splitting (default: 5)
    pub min_node_size: usize,
    /// Number of variables to consider at each split (default: sqrt(p))
    pub mtry: Option<usize>,
    /// Use honest splitting (default: true)
    pub honesty: bool,
    /// Fraction of data used for estimation in honest splitting (default: 0.5)
    pub honesty_fraction: f64,
    /// Maximum tree depth (default: 10)
    pub max_depth: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Compute variable importance (default: true)
    pub compute_importance: bool,
    /// Sample fraction for each tree (default: 0.5)
    pub sample_fraction: f64,
}

impl Default for CausalForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 2000,
            min_node_size: 5,
            mtry: None,
            honesty: true,
            honesty_fraction: 0.5,
            max_depth: 10,
            seed: None,
            compute_importance: true,
            sample_fraction: 0.5,
        }
    }
}

impl CausalForestConfig {
    /// An upper bound, in bytes, of what [`causal_forest`] holds at once for a
    /// dataset of `n_obs` rows and `n_covariates` covariates under this
    /// configuration: the trees it keeps and the buffers that are alive
    /// together while it runs. Every buffer it counts is allocated at its
    /// exact length, the index vectors of the subsample and of the recursion
    /// included, so no vector holds unused capacity. The allocator's own
    /// overhead (chunk headers and free lists) and the dataset itself come on
    /// top. Saturates at `u64::MAX`.
    ///
    /// The bound holds whatever the data are, because each term counts the
    /// most the code can allocate:
    ///
    /// - **Trees.** Each tree has at most `2 L - 1` nodes for `L` leaves.
    ///   `L` is at most `2^max_depth`, and at most each part of the tree's
    ///   subsample (structure and estimation; with honesty off, both are the
    ///   whole subsample) divided by `max(min_node_size, 4)`: a split is kept
    ///   only when each child has `min_node_size` rows and two treated and two
    ///   control units in both parts, and the leaves partition both parts.
    ///   Every node is counted at its heap size, and each tree with its slot
    ///   in the per-row prediction buffer.
    /// - **Data and results.** The outcome, treatment and covariates
    ///   [`causal_forest`] extracts, and the column it is extracting (cast to
    ///   `f64` and copied); the out-of-bag counts, means and sums of squares;
    ///   the predictions, the variances and the sorted copy the result's
    ///   summary makes.
    /// - **One tree's working copies.** The subsample's flags and indices and
    ///   its rows; with honesty, the shuffled indices, the two parts and their
    ///   rows; and what the recursion holds. Every level on the path to the
    ///   node being split holds its children's index vectors, which together
    ///   are at most the two parts, and the covariates its split drew (`mtry`,
    ///   at most `n_covariates`). The path is at most `max_depth` levels deep
    ///   and at most the structure part divided by `max(min_node_size, 4)`,
    ///   since each split leaves the child on the path at least that many rows
    ///   fewer. The split search's sort order adds one structure part, and the
    ///   list the covariates are drawn from one covariate list.
    pub fn memory_bound(&self, n_obs: usize, n_covariates: usize) -> u64 {
        let n = n_obs as u128;
        let p = n_covariates as u128;
        let (s, t, e) = self.subsample_parts(n_obs);
        let leaf = self.min_node_size.max(4) as u128;
        let levels = (t / leaf).min(self.max_depth as u128) + 1;
        // The covariates one split draws, as `select_features` keeps them.
        let drawn = self
            .mtry
            .unwrap_or_else(|| default_mtry(n_covariates))
            .min(n_covariates) as u128;
        let tree = mul(
            self.max_nodes_per_tree(n_obs),
            heap_chunk_bytes(size_of::<CausalTreeNode>()),
        )
        .saturating_add(size_of::<CausalTree>() as u128 + 8);
        let trees = mul(self.n_trees as u128, tree);
        // Extracted y, w and x, and the column being extracted (its cast and its values) (2);
        // out-of-bag counts, means and sums of squares (3); predictions, variances and the
        // summary's sorted copy (3); the importance sums, and the list the covariates are drawn
        // from or one tree's importances.
        let data = mul(mul(8, n), p.saturating_add(2 + 2 + 3 + 3)).saturating_add(mul(16, p));
        // Subsample flags, indices and rows.
        let mut work = mul(n, 9).saturating_add(mul(mul(8, s), p.saturating_add(2)));
        if self.honesty {
            work = work
                .saturating_add(mul(24, s))
                .saturating_add(mul(mul(8, t + e), p.saturating_add(2)));
        }
        // Recursion path (each level's child index vectors and drawn covariates) and split
        // search.
        work = work
            .saturating_add(mul(mul(8, t + e + drawn), levels))
            .saturating_add(mul(8, t));
        u64::try_from(trees.saturating_add(data).saturating_add(work)).unwrap_or(u64::MAX)
    }

    /// The sizes `(s, t, e)` of a tree's subsample and of its structure and
    /// estimation parts, as the fit draws them for `n_obs` rows; without
    /// honesty both parts are the whole subsample.
    fn subsample_parts(&self, n_obs: usize) -> (u128, u128, u128) {
        let s = ((n_obs as f64) * self.sample_fraction) as usize;
        if self.honesty {
            let e = ((s as f64) * self.honesty_fraction) as usize;
            (s as u128, (s - e.min(s)) as u128, e as u128)
        } else {
            (s as u128, s as u128, s as u128)
        }
    }

    /// The most nodes one tree can have for `n_obs` rows ([`Self::memory_bound`]).
    fn max_nodes_per_tree(&self, n_obs: usize) -> u128 {
        let (_, t, e) = self.subsample_parts(n_obs);
        let leaf = self.min_node_size.max(4) as u128;
        let leaves = (t / leaf).min(e / leaf).min(pow2(self.max_depth)).max(1);
        2 * leaves - 1
    }
}

/// Result from causal forest estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestResult {
    /// CATE estimates for each observation
    pub predictions: Vec<f64>,
    /// Variance estimates for each CATE prediction
    pub variance_estimates: Vec<f64>,
    /// Average treatment effect (ATE)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// T-statistic for ATE
    pub ate_t_stat: f64,
    /// P-value for ATE
    pub ate_p_value: f64,
    /// 95% CI lower bound for ATE
    pub ate_ci_lower: f64,
    /// 95% CI upper bound for ATE
    pub ate_ci_upper: f64,
    /// Significance level for ATE
    pub ate_significance: SignificanceLevel,
    /// Variable importance scores (covariate name, importance)
    pub variable_importance: Vec<(String, f64)>,
    /// Number of trees
    pub n_trees: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Out-of-bag prediction error
    pub oob_error: f64,
    /// Covariate names
    pub covariate_names: Vec<String>,
    /// Configuration used
    pub config: CausalForestConfig,
    /// Internal: tree structures for prediction (not serialized)
    #[serde(skip)]
    trees: Vec<CausalTree>,
}

impl fmt::Display for CausalForestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Causal Forest Results")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Trees: {}", self.n_trees)?;
        writeln!(f, "Out-of-bag Error: {:.4}", self.oob_error)?;
        writeln!(f)?;

        writeln!(f, "AVERAGE TREATMENT EFFECT (ATE):")?;
        writeln!(
            f,
            "  ATE = {:.4} (SE: {:.4}, t = {:.2}, p = {:.3}){}",
            self.ate,
            self.ate_se,
            self.ate_t_stat,
            self.ate_p_value,
            self.ate_significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI: [{:.4}, {:.4}]",
            self.ate_ci_lower, self.ate_ci_upper
        )?;
        writeln!(f)?;

        writeln!(f, "CATE DISTRIBUTION:")?;
        if !self.predictions.is_empty() {
            let mut sorted = self.predictions.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let p10 = sorted[n / 10];
            let p25 = sorted[n / 4];
            let p50 = sorted[n / 2];
            let p75 = sorted[3 * n / 4];
            let p90 = sorted[9 * n / 10];
            writeln!(f, "  Min:    {:.4}", sorted.first().unwrap_or(&0.0))?;
            writeln!(f, "  10%:    {:.4}", p10)?;
            writeln!(f, "  25%:    {:.4}", p25)?;
            writeln!(f, "  Median: {:.4}", p50)?;
            writeln!(f, "  75%:    {:.4}", p75)?;
            writeln!(f, "  90%:    {:.4}", p90)?;
            writeln!(f, "  Max:    {:.4}", sorted.last().unwrap_or(&0.0))?;
        }
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE:")?;
        for (i, (name, importance)) in self.variable_importance.iter().enumerate() {
            if i >= 10 {
                writeln!(
                    f,
                    "  ... ({} more variables)",
                    self.variable_importance.len() - 10
                )?;
                break;
            }
            writeln!(f, "  {:<20} {:.4}", name, importance)?;
        }
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Honest splitting: {}", self.config.honesty)?;
        writeln!(f, "  Min node size: {}", self.config.min_node_size)?;
        writeln!(f, "  Max depth: {}", self.config.max_depth)?;

        Ok(())
    }
}

/// A single causal tree node.
#[derive(Debug, Clone)]
enum CausalTreeNode {
    /// Internal node with a split
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<CausalTreeNode>,
        right: Box<CausalTreeNode>,
    },
    /// Leaf node with treatment effect estimate
    Leaf {
        treatment_effect: f64,
        variance: f64,
        n_treated: usize,
        n_control: usize,
    },
}

/// A single causal tree.
#[derive(Debug, Clone)]
struct CausalTree {
    root: Option<CausalTreeNode>,
    n_features: usize,
    max_depth: usize,
    min_node_size: usize,
    mtry: usize,
}

impl CausalTree {
    /// Create a new causal tree.
    fn new(max_depth: usize, min_node_size: usize, mtry: usize) -> Self {
        Self {
            root: None,
            n_features: 0,
            max_depth,
            min_node_size,
            mtry,
        }
    }

    /// The fewest rows a node needs in each part (structure and estimation) to be split:
    /// two children of `min_node_size` rows. It saturates at `usize::MAX`, more rows than
    /// any node can hold, so a `min_node_size` above `usize::MAX / 2` stops every split
    /// instead of overflowing; below that it is exactly `2 * min_node_size`.
    fn min_rows_to_split(&self) -> usize {
        self.min_node_size.saturating_mul(2)
    }

    /// Fit the tree using honest splitting.
    ///
    /// The splitting sample is used to determine tree structure,
    /// and the estimation sample is used to estimate treatment effects in leaves.
    fn fit_honest(
        &mut self,
        x_split: ArrayView2<f64>,
        y_split: ArrayView1<f64>,
        w_split: ArrayView1<f64>,
        x_est: ArrayView2<f64>,
        y_est: ArrayView1<f64>,
        w_est: ArrayView1<f64>,
        rng_state: &mut u64,
    ) {
        self.n_features = x_split.ncols();

        // Build tree structure using splitting sample
        let split_indices: Vec<usize> = (0..x_split.nrows()).collect();
        let est_indices: Vec<usize> = (0..x_est.nrows()).collect();

        self.root = Some(self.build_tree_honest(
            &x_split,
            &y_split,
            &w_split,
            &x_est,
            &y_est,
            &w_est,
            &split_indices,
            &est_indices,
            0,
            rng_state,
        ));
    }

    /// Build tree recursively with honest splitting.
    fn build_tree_honest(
        &self,
        x_split: &ArrayView2<f64>,
        y_split: &ArrayView1<f64>,
        w_split: &ArrayView1<f64>,
        x_est: &ArrayView2<f64>,
        y_est: &ArrayView1<f64>,
        w_est: &ArrayView1<f64>,
        split_indices: &[usize],
        est_indices: &[usize],
        depth: usize,
        rng_state: &mut u64,
    ) -> CausalTreeNode {
        // Check stopping conditions
        let n_split = split_indices.len();
        let n_est = est_indices.len();

        // Count treated and control in estimation sample
        let (n_treated, n_control) = count_treated_control(w_est, est_indices);

        // Stop if: max depth reached, too few samples, or no variation in treatment
        let min_rows = self.min_rows_to_split();
        if depth >= self.max_depth
            || n_split < min_rows
            || n_est < min_rows
            || n_treated < self.min_node_size
            || n_control < self.min_node_size
        {
            return self.create_leaf(y_est, w_est, est_indices);
        }

        // Select random subset of features
        let features_to_try = self.select_features(rng_state);

        // Find best split using splitting sample
        if let Some((best_feature, best_threshold)) =
            self.find_best_causal_split(x_split, y_split, w_split, split_indices, &features_to_try)
        {
            // Split both samples based on the threshold
            let (split_left, split_right) =
                partition_indices(x_split, split_indices, best_feature, best_threshold);
            let (est_left, est_right) =
                partition_indices(x_est, est_indices, best_feature, best_threshold);

            // Check minimum node sizes
            if split_left.len() < self.min_node_size
                || split_right.len() < self.min_node_size
                || est_left.len() < self.min_node_size
                || est_right.len() < self.min_node_size
            {
                return self.create_leaf(y_est, w_est, est_indices);
            }

            // Check for treatment variation in children
            let (left_treated, left_control) = count_treated_control(w_est, &est_left);
            let (right_treated, right_control) = count_treated_control(w_est, &est_right);

            if left_treated < 2 || left_control < 2 || right_treated < 2 || right_control < 2 {
                return self.create_leaf(y_est, w_est, est_indices);
            }

            let left = self.build_tree_honest(
                x_split,
                y_split,
                w_split,
                x_est,
                y_est,
                w_est,
                &split_left,
                &est_left,
                depth + 1,
                rng_state,
            );
            let right = self.build_tree_honest(
                x_split,
                y_split,
                w_split,
                x_est,
                y_est,
                w_est,
                &split_right,
                &est_right,
                depth + 1,
                rng_state,
            );

            CausalTreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(y_est, w_est, est_indices)
        }
    }

    /// Create a leaf node with treatment effect estimate.
    fn create_leaf(
        &self,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        indices: &[usize],
    ) -> CausalTreeNode {
        let (treatment_effect, variance, n_treated, n_control) =
            estimate_treatment_effect(y, w, indices);

        CausalTreeNode::Leaf {
            treatment_effect,
            variance,
            n_treated,
            n_control,
        }
    }

    /// Select random features for splitting.
    fn select_features(&self, rng_state: &mut u64) -> Vec<usize> {
        let mut selected = Vec::with_capacity(self.mtry.min(self.n_features));
        let mut available: Vec<usize> = (0..self.n_features).collect();

        for _ in 0..self.mtry.min(self.n_features) {
            if available.is_empty() {
                break;
            }
            let idx = lcg_random(rng_state) % available.len();
            selected.push(available.swap_remove(idx));
        }

        selected
    }

    /// Find the best split for causal effect heterogeneity.
    ///
    /// The splitting criterion maximizes the variance of treatment effects
    /// across child nodes, following Athey & Imbens (2016): it maximises
    /// `w_L * tau_L^2 + w_R * tau_R^2`, where `w` is the harmonic mean of the
    /// treated and control counts of a child.
    ///
    /// Each feature's node values are sorted once and the candidate
    /// thresholds (midpoints of consecutive distinct values, ascending) are
    /// swept with running treated/control counts and sums. The left child of a
    /// threshold is every observation with `x <= threshold`, exactly as a
    /// partition of the node would give, so the candidate children are those
    /// of the re-partitioning search this replaces. `x` must be finite.
    ///
    /// The child means now come from sums accumulated in sorted-x order (the
    /// right child's by subtraction from the node totals) rather than in
    /// index order. With integer outcomes whose sums stay below 2^53 (about
    /// 9e15) in magnitude, as 0/1 outcomes, counts and other moderate
    /// integers do, those sums are exact and both searches choose the same
    /// split. Otherwise the chosen split is the same except where two
    /// candidates' criteria are equal, exactly or to within rounding: the two
    /// searches may break those ties either way, so decimal outcomes (say, in
    /// tenths) can take the other, equally good split where two candidates
    /// tie, as they do when two covariate groups mirror each other, and the
    /// tree then grows differently below it.
    fn find_best_causal_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_criterion = f64::NEG_INFINITY;
        let mut best_split = None;

        let n = indices.len();
        if n < 2 {
            return None;
        }

        // Node totals (treated means w > 0.5)
        let (mut tot_nt, mut tot_nc, mut tot_st, mut tot_sc) = (0usize, 0usize, 0.0, 0.0);
        for &i in indices {
            if w[i] > 0.5 {
                tot_nt += 1;
                tot_st += y[i];
            } else {
                tot_nc += 1;
                tot_sc += y[i];
            }
        }

        let harmonic = |nt: usize, nc: usize| {
            if nt > 0 && nc > 0 {
                2.0 * (nt as f64) * (nc as f64) / ((nt + nc) as f64)
            } else {
                0.0
            }
        };

        let mut order: Vec<usize> = indices.to_vec();
        for &feature in features {
            order.sort_by(|&a, &b| x[[a, feature]].total_cmp(&x[[b, feature]]));

            // Left-child accumulators, advanced by a pointer into `order`
            let (mut l_nt, mut l_nc, mut l_st, mut l_sc) = (0usize, 0usize, 0.0, 0.0);
            let mut ptr = 0usize;
            let mut lower = x[[order[0], feature]];

            for pos in 1..n {
                let upper = x[[order[pos], feature]];
                if upper == lower {
                    continue;
                }
                let threshold = (lower + upper) / 2.0;
                lower = upper;

                while ptr < n && x[[order[ptr], feature]] <= threshold {
                    let i = order[ptr];
                    if w[i] > 0.5 {
                        l_nt += 1;
                        l_st += y[i];
                    } else {
                        l_nc += 1;
                        l_sc += y[i];
                    }
                    ptr += 1;
                }

                if ptr < self.min_node_size || n - ptr < self.min_node_size {
                    continue;
                }

                // Check for treatment variation in both children
                let (r_nt, r_nc) = (tot_nt - l_nt, tot_nc - l_nc);
                if l_nt < 2 || l_nc < 2 || r_nt < 2 || r_nc < 2 {
                    continue;
                }

                let tau_left = l_st / l_nt as f64 - l_sc / l_nc as f64;
                let tau_right = (tot_st - l_st) / r_nt as f64 - (tot_sc - l_sc) / r_nc as f64;
                let criterion = harmonic(l_nt, l_nc) * tau_left.powi(2)
                    + harmonic(r_nt, r_nc) * tau_right.powi(2);

                if criterion > best_criterion {
                    best_criterion = criterion;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
    }

    /// Predict treatment effect for a single observation.
    fn predict_one(&self, x: &ArrayView1<f64>) -> (f64, f64) {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => (0.0, f64::INFINITY),
        }
    }

    /// Traverse tree to get prediction.
    fn traverse(&self, node: &CausalTreeNode, x: &ArrayView1<f64>) -> (f64, f64) {
        match node {
            CausalTreeNode::Leaf {
                treatment_effect,
                variance,
                ..
            } => (*treatment_effect, *variance),
            CausalTreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[*feature_index] <= *threshold {
                    self.traverse(left, x)
                } else {
                    self.traverse(right, x)
                }
            }
        }
    }

    /// Get feature importances based on split frequency.
    fn feature_importances(&self) -> Array1<f64> {
        let mut importances = Array1::zeros(self.n_features);
        if let Some(ref root) = self.root {
            self.accumulate_importances(root, &mut importances, 1.0);
        }

        // Normalize
        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances /= sum;
        }

        importances
    }

    fn accumulate_importances(
        &self,
        node: &CausalTreeNode,
        importances: &mut Array1<f64>,
        weight: f64,
    ) {
        if let CausalTreeNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            importances[*feature_index] += weight;
            self.accumulate_importances(left, importances, weight * 0.5);
            self.accumulate_importances(right, importances, weight * 0.5);
        }
    }
}

/// Estimate treatment effect within a set of indices.
///
/// Returns (treatment_effect, variance, n_treated, n_control)
fn estimate_treatment_effect(
    y: &ArrayView1<f64>,
    w: &ArrayView1<f64>,
    indices: &[usize],
) -> (f64, f64, usize, usize) {
    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;
    let mut n_treated = 0;
    let mut n_control = 0;

    for &i in indices {
        if w[i] > 0.5 {
            sum_treated += y[i];
            n_treated += 1;
        } else {
            sum_control += y[i];
            n_control += 1;
        }
    }

    if n_treated == 0 || n_control == 0 {
        return (0.0, f64::INFINITY, n_treated, n_control);
    }

    let mean_treated = sum_treated / n_treated as f64;
    let mean_control = sum_control / n_control as f64;
    let treatment_effect = mean_treated - mean_control;

    // Compute variance using formula: Var(tau) = Var(Y|W=1)/n1 + Var(Y|W=0)/n0
    let mut var_treated = 0.0;
    let mut var_control = 0.0;

    for &i in indices {
        if w[i] > 0.5 {
            var_treated += (y[i] - mean_treated).powi(2);
        } else {
            var_control += (y[i] - mean_control).powi(2);
        }
    }

    let var_treated = if n_treated > 1 {
        var_treated / (n_treated - 1) as f64
    } else {
        0.0
    };
    let var_control = if n_control > 1 {
        var_control / (n_control - 1) as f64
    } else {
        0.0
    };

    let variance = var_treated / n_treated as f64 + var_control / n_control as f64;

    (treatment_effect, variance, n_treated, n_control)
}

/// Count treated and control units in indices.
fn count_treated_control(w: &ArrayView1<f64>, indices: &[usize]) -> (usize, usize) {
    let mut n_treated = 0;
    let mut n_control = 0;

    for &i in indices {
        if w[i] > 0.5 {
            n_treated += 1;
        } else {
            n_control += 1;
        }
    }

    (n_treated, n_control)
}

/// Partition indices based on a split, each child's vector at its exact length (the
/// recursion holds them on every level, and [`CausalForestConfig::memory_bound`] counts them
/// by their lengths).
fn partition_indices(
    x: &ArrayView2<f64>,
    indices: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    partition_exact(indices.iter().copied(), |i| x[[i, feature]] <= threshold)
}

use super::lcg_random;

/// Generate subsample indices without replacement and the out-of-bag rest, both vectors at
/// their exact lengths.
fn subsample(n_samples: usize, fraction: f64, rng_state: &mut u64) -> (Vec<usize>, Vec<usize>) {
    let subsample_size = ((n_samples as f64) * fraction) as usize;
    let mut selected = vec![false; n_samples];
    let mut sample = Vec::with_capacity(subsample_size);

    while sample.len() < subsample_size {
        let idx = lcg_random(rng_state) % n_samples;
        if !selected[idx] {
            selected[idx] = true;
            sample.push(idx);
        }
    }

    // `sample` holds distinct indices, so exactly the others are out of bag.
    let mut oob = Vec::with_capacity(n_samples - sample.len());
    for (i, &in_sample) in selected.iter().enumerate() {
        if !in_sample {
            oob.push(i);
        }
    }
    (sample, oob)
}

/// Split sample into structure and estimation parts for honest estimation.
fn honest_split(
    indices: &[usize],
    honesty_fraction: f64,
    rng_state: &mut u64,
) -> (Vec<usize>, Vec<usize>) {
    let n = indices.len();
    let est_size = ((n as f64) * honesty_fraction) as usize;

    let mut shuffled = indices.to_vec();
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = lcg_random(rng_state) % (i + 1);
        shuffled.swap(i, j);
    }

    let estimation = shuffled[..est_size].to_vec();
    let structure = shuffled[est_size..].to_vec();

    (structure, estimation)
}

/// Run causal forest estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration for the causal forest
///
/// # Returns
/// `CausalForestResult` containing CATE estimates, ATE, and diagnostics
///
/// # Example
/// ```ignore
/// use p2a_core::ml::{causal_forest, CausalForestConfig};
///
/// let config = CausalForestConfig {
///     n_trees: 1000,
///     honesty: true,
///     ..Default::default()
/// };
///
/// let result = causal_forest(
///     &dataset,
///     "outcome",
///     "treatment",
///     &["age", "income", "education"],
///     config,
/// )?;
///
/// println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
/// ```
///
/// # References
///
/// - Wager & Athey (2018), JASA. Algorithm 1.
/// - R `grf::causal_forest()` for reference implementation.
pub fn causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    // Extract outcome
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract treatment
    let w = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Validate treatment is binary
    for &val in w.iter() {
        if val != 0.0 && val != 1.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Treatment variable '{}' must be binary (0/1), found value: {}",
                    treatment_col, val
                ),
            });
        }
    }

    // Extract covariates
    let n = dataset.nrows();
    let p = covariate_cols.len();

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one covariate is required".to_string(),
        });
    }

    let mut x = Array2::zeros((n, p));
    for (j, &col_name) in covariate_cols.iter().enumerate() {
        let col_data = DesignMatrix::extract_column(dataset.df(), col_name).map_err(|e| {
            EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        for i in 0..n {
            x[[i, j]] = col_data[i];
        }
    }

    // Run the core algorithm
    causal_forest_arrays(
        x.view(),
        y.view(),
        w.view(),
        covariate_cols.iter().map(|s| s.to_string()).collect(),
        config,
    )
}

/// Check the counts and fractions of a causal-forest configuration.
///
/// `n_trees >= 1`; `sample_fraction` finite and in (0, 1]; with honesty,
/// `honesty_fraction` in the open interval (0, 1), as `grf` requires.
fn validate_causal_forest_config(config: &CausalForestConfig) -> EconResult<()> {
    if config.n_trees == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_trees must be at least 1 (got 0)".to_string(),
        });
    }
    let sf = config.sample_fraction;
    if !(sf.is_finite() && sf > 0.0 && sf <= 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("sample_fraction must be in (0, 1] (got {sf})"),
        });
    }
    let hf = config.honesty_fraction;
    if config.honesty && !(hf.is_finite() && hf > 0.0 && hf < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("honesty_fraction must be in (0, 1) when honesty is on (got {hf})"),
        });
    }
    Ok(())
}

/// `InvalidSpecification` naming the first non-finite value of `values`.
fn check_finite_1d(values: &ArrayView1<f64>, name: &str) -> EconResult<()> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(EconError::InvalidSpecification {
            message: format!(
                "{name} must contain only finite values (row {i} is {})",
                values[i]
            ),
        }),
        None => Ok(()),
    }
}

/// Core causal forest algorithm working with arrays.
///
/// This is the main implementation, separate from data loading for testability.
pub fn causal_forest_arrays(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    covariate_names: Vec<String>,
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    validate_causal_forest_config(&config)?;

    let n = x.nrows();
    let p = x.ncols();

    // Validate inputs
    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "Causal forest requires at least 10 observations".to_string(),
        });
    }

    if y.len() != n || w.len() != n {
        return Err(EconError::InvalidSpecification {
            message: "Outcome, treatment, and covariates must have same number of observations"
                .to_string(),
        });
    }

    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("x must contain only finite values (row {i}, column {j} is {v})"),
        });
    }
    check_finite_1d(&y, "y")?;
    check_finite_1d(&w, "w")?;

    // Count treated/control
    let (n_treated, n_control) = count_treated_control(&w, &(0..n).collect::<Vec<_>>());
    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "Need at least 5 treated and 5 control units".to_string(),
        });
    }

    // Initialize RNG
    let mut rng_state = config.seed.unwrap_or(42);

    // Determine mtry
    let mtry = config.mtry.unwrap_or_else(|| default_mtry(p));

    // Build forest
    let mut trees = try_with_capacity::<CausalTree>(config.n_trees, "causal forest trees")?;
    // Per-row prediction buffer, reused for every observation below
    let mut preds = try_with_capacity::<f64>(config.n_trees, "causal forest per-row predictions")?;
    // Out-of-bag statistics per observation, accumulated with Welford's algorithm
    let mut oob_count = vec![0usize; n];
    let mut oob_mean = vec![0.0f64; n];
    let mut oob_m2 = vec![0.0f64; n];
    let mut feature_importance_sum = Array1::zeros(p);

    for _tree_idx in 0..config.n_trees {
        checkpoint()?;

        // Subsample
        let (sample_indices, oob_indices) = subsample(n, config.sample_fraction, &mut rng_state);

        // Extract subsample data
        let x_sample = select_rows_view(&x, &sample_indices);
        let y_sample: Array1<f64> = sample_indices.iter().map(|&i| y[i]).collect();
        let w_sample: Array1<f64> = sample_indices.iter().map(|&i| w[i]).collect();

        // Build tree
        let mut tree = CausalTree::new(config.max_depth, config.min_node_size, mtry);

        if config.honesty {
            // Split sample for honest estimation
            let sample_idx_local: Vec<usize> = (0..sample_indices.len()).collect();
            let (structure_idx, estimation_idx) =
                honest_split(&sample_idx_local, config.honesty_fraction, &mut rng_state);

            let x_struct = select_rows_view(&x_sample.view(), &structure_idx);
            let y_struct: Array1<f64> = structure_idx.iter().map(|&i| y_sample[i]).collect();
            let w_struct: Array1<f64> = structure_idx.iter().map(|&i| w_sample[i]).collect();

            let x_est = select_rows_view(&x_sample.view(), &estimation_idx);
            let y_est: Array1<f64> = estimation_idx.iter().map(|&i| y_sample[i]).collect();
            let w_est: Array1<f64> = estimation_idx.iter().map(|&i| w_sample[i]).collect();

            tree.fit_honest(
                x_struct.view(),
                y_struct.view(),
                w_struct.view(),
                x_est.view(),
                y_est.view(),
                w_est.view(),
                &mut rng_state,
            );
        } else {
            // Non-honest: use same data for structure and estimation
            tree.fit_honest(
                x_sample.view(),
                y_sample.view(),
                w_sample.view(),
                x_sample.view(),
                y_sample.view(),
                w_sample.view(),
                &mut rng_state,
            );
        }

        // OOB predictions
        for &oob_idx in &oob_indices {
            let (pred, _) = tree.predict_one(&x.row(oob_idx));
            oob_count[oob_idx] += 1;
            let delta = pred - oob_mean[oob_idx];
            oob_mean[oob_idx] += delta / oob_count[oob_idx] as f64;
            oob_m2[oob_idx] += delta * (pred - oob_mean[oob_idx]);
        }

        // Accumulate feature importance
        if config.compute_importance {
            feature_importance_sum = feature_importance_sum + tree.feature_importances();
        }

        trees.push(tree);
    }

    // Compute final predictions (average across all trees)
    let mut predictions = Vec::with_capacity(n);
    let mut variance_estimates = Vec::with_capacity(n);

    for i in 0..n {
        checkpoint()?;
        preds.clear();
        for tree in &trees {
            let (pred, _) = tree.predict_one(&x.row(i));
            preds.push(pred);
        }

        let mean_pred: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
        predictions.push(mean_pred);

        // Variance from bootstrap (across trees)
        let var: f64 = preds.iter().map(|&p| (p - mean_pred).powi(2)).sum::<f64>()
            / (preds.len() - 1).max(1) as f64;
        variance_estimates.push(var);
    }

    // Compute OOB error
    let oob_error = compute_oob_error(&oob_count, &oob_m2);

    // Compute ATE and its standard error
    let ate: f64 = predictions.iter().sum::<f64>() / n as f64;
    let ate_var: f64 = variance_estimates.iter().sum::<f64>() / n as f64
        + predictions.iter().map(|&p| (p - ate).powi(2)).sum::<f64>() / (n * (n - 1)) as f64;
    let ate_se = ate_var.sqrt();
    let ate_t_stat = ate / ate_se;
    let ate_df = (n - 2) as f64;
    let ate_p_value = t_test_p_value(ate_t_stat, ate_df);

    // 95% CI for ATE
    let z = 1.96;
    let ate_ci_lower = ate - z * ate_se;
    let ate_ci_upper = ate + z * ate_se;

    // Normalize feature importances
    let total_importance: f64 = feature_importance_sum.sum();
    let variable_importance: Vec<(String, f64)> = if total_importance > 0.0 {
        covariate_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), feature_importance_sum[i] / total_importance))
            .collect()
    } else {
        covariate_names
            .iter()
            .map(|name| (name.clone(), 0.0))
            .collect()
    };

    // Sort by importance
    let mut variable_importance = variable_importance;
    variable_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(CausalForestResult {
        predictions,
        variance_estimates,
        ate,
        ate_se,
        ate_t_stat,
        ate_p_value,
        ate_ci_lower,
        ate_ci_upper,
        ate_significance: SignificanceLevel::from_p_value(ate_p_value),
        variable_importance,
        n_trees: config.n_trees,
        n_obs: n,
        oob_error,
        covariate_names,
        config,
        trees,
    })
}

/// Select rows from a 2D array.
fn select_rows_view(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Compute OOB error (MSE of treatment effect predictions).
///
/// `counts[i]` and `m2[i]` are observation `i`'s number of out-of-bag
/// predictions and Welford sum of squared deviations from their mean.
fn compute_oob_error(counts: &[usize], m2: &[f64]) -> f64 {
    // For causal forest, we can't directly compute OOB error because
    // we don't observe the true treatment effect. Instead, we use
    // a proxy based on R-loss (Nie & Wager, 2021).
    //
    // R-loss = (Y - tau(X) * W)^2 adjusted for propensity
    //
    // Here we use a simplified version: variance of OOB predictions

    let mut sum_var = 0.0;
    let mut count = 0;

    for (&c, &ss) in counts.iter().zip(m2) {
        if c > 1 {
            sum_var += ss / (c - 1) as f64;
            count += 1;
        }
    }

    if count > 0 {
        sum_var / count as f64
    } else {
        f64::NAN
    }
}

/// Predict CATE for new observations.
///
/// # Arguments
/// * `forest` - A fitted causal forest result
/// * `new_data` - Dataset with covariate values for prediction
/// * `covariate_cols` - Names of covariate columns (must match training)
///
/// # Returns
/// Vector of CATE predictions
pub fn causal_forest_predict(
    forest: &CausalForestResult,
    new_data: &Dataset,
    covariate_cols: &[&str],
) -> EconResult<Vec<f64>> {
    // Validate covariate names match
    if covariate_cols.len() != forest.covariate_names.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of covariates ({}) doesn't match training ({})",
                covariate_cols.len(),
                forest.covariate_names.len()
            ),
        });
    }

    // Extract covariates
    let n = new_data.nrows();
    let p = covariate_cols.len();
    let mut x = Array2::zeros((n, p));

    for (j, &col_name) in covariate_cols.iter().enumerate() {
        let col_data = DesignMatrix::extract_column(new_data.df(), col_name).map_err(|e| {
            EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        for i in 0..n {
            x[[i, j]] = col_data[i];
        }
    }

    // Predict using forest
    causal_forest_predict_arrays(forest, x.view())
}

/// Predict CATE using array inputs.
pub fn causal_forest_predict_arrays(
    forest: &CausalForestResult,
    x: ArrayView2<f64>,
) -> EconResult<Vec<f64>> {
    if forest.trees.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Forest has no trees (may have been deserialized without tree structures)"
                .to_string(),
        });
    }

    let n = x.nrows();
    let mut predictions = Vec::with_capacity(n);

    for i in 0..n {
        checkpoint()?;
        let mut sum = 0.0;
        for tree in &forest.trees {
            let (pred, _) = tree.predict_one(&x.row(i));
            sum += pred;
        }
        predictions.push(sum / forest.trees.len() as f64);
    }

    Ok(predictions)
}

/// Get average treatment effect for a subset of observations.
///
/// # Arguments
/// * `forest` - A fitted causal forest result
/// * `subset_indices` - Optional indices of observations to include
///
/// # Returns
/// (ATE, CI lower, CI upper)
pub fn average_treatment_effect(
    forest: &CausalForestResult,
    subset_indices: Option<&[usize]>,
) -> EconResult<(f64, f64, f64)> {
    let predictions: Vec<f64> = match subset_indices {
        Some(indices) => indices.iter().map(|&i| forest.predictions[i]).collect(),
        None => forest.predictions.clone(),
    };

    if predictions.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No observations in subset".to_string(),
        });
    }

    let n = predictions.len();
    let ate: f64 = predictions.iter().sum::<f64>() / n as f64;

    // Estimate standard error
    let variance: f64 = predictions.iter().map(|&p| (p - ate).powi(2)).sum::<f64>() / n as f64;
    let se = (variance / n as f64).sqrt();

    // 95% CI
    let z = 1.96;
    let ci_lower = ate - z * se;
    let ci_upper = ate + z * se;

    Ok((ate, ci_lower, ci_upper))
}

/// Run causal forest from dataset with string arguments (for MCP integration).
///
/// This is a convenience wrapper that handles string-to-slice conversion.
pub fn run_causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: Vec<String>,
    n_trees: Option<usize>,
    min_node_size: Option<usize>,
    honesty: Option<bool>,
    max_depth: Option<usize>,
    seed: Option<u64>,
) -> EconResult<CausalForestResult> {
    let cov_refs: Vec<&str> = covariate_cols.iter().map(|s| s.as_str()).collect();
    let config = run_causal_forest_config(n_trees, min_node_size, honesty, max_depth, seed);
    causal_forest(dataset, outcome_col, treatment_col, &cov_refs, config)
}

/// The configuration [`run_causal_forest`] fits with for these arguments
/// (the defaults of [`CausalForestConfig`] where an argument is `None`), for
/// example to check its [`CausalForestConfig::memory_bound`] first.
pub fn run_causal_forest_config(
    n_trees: Option<usize>,
    min_node_size: Option<usize>,
    honesty: Option<bool>,
    max_depth: Option<usize>,
    seed: Option<u64>,
) -> CausalForestConfig {
    let defaults = CausalForestConfig::default();
    CausalForestConfig {
        n_trees: n_trees.unwrap_or(defaults.n_trees),
        min_node_size: min_node_size.unwrap_or(defaults.min_node_size),
        honesty: honesty.unwrap_or(defaults.honesty),
        max_depth: max_depth.unwrap_or(defaults.max_depth),
        seed,
        ..defaults
    }
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

    fn names3() -> Vec<String> {
        vec!["x0".into(), "x1".into(), "x2".into()]
    }

    fn small_config() -> CausalForestConfig {
        CausalForestConfig {
            n_trees: 10,
            max_depth: 4,
            seed: Some(3),
            ..Default::default()
        }
    }

    fn count_nodes(node: &CausalTreeNode) -> u128 {
        match node {
            CausalTreeNode::Leaf { .. } => 1,
            CausalTreeNode::Split { left, right, .. } => 1 + count_nodes(left) + count_nodes(right),
        }
    }

    /// Distinct covariate values, alternating treatment and a noisy outcome: every threshold
    /// the size and treatment-count checks allow is a candidate, so the trees grow as large as
    /// those checks let them.
    fn splittable_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = seed;
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = (lcg_random(&mut rng) % 1_000_003) as f64 + i as f64 * 1e-7;
            }
            w[i] = (i % 2) as f64;
            let noise = (lcg_random(&mut rng) % 1000) as f64 / 1000.0;
            y[i] = x[[i, 0]] * 1e-6 * (1.0 + w[i]) + noise;
        }
        (x, y, w)
    }

    /// The node bound behind `memory_bound` holds for every tree of fitted forests, with and
    /// without honesty, at node sizes 0 to 5 and depths 3 to 100.
    #[test]
    fn memory_bound_counts_at_least_the_nodes_every_tree_grows() {
        for (n, honesty, min_node_size, max_depth) in [
            (400, true, 5, 10),
            (400, true, 0, 100),
            (400, false, 0, 100),
            (400, false, 1, 100),
            (400, false, 5, 100),
            (250, false, 4, 3),
            (61, true, 5, 10),
            (1_000, false, 0, 100),
        ] {
            let cfg = CausalForestConfig {
                n_trees: 6,
                honesty,
                min_node_size,
                max_depth,
                seed: Some(11),
                ..Default::default()
            };
            let bound = cfg.max_nodes_per_tree(n);
            let (x, y, w) = splittable_data(n, 7 + n as u64);
            let fit =
                causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg.clone()).unwrap();
            let most = fit
                .trees
                .iter()
                .map(|t| count_nodes(t.root.as_ref().unwrap()))
                .max()
                .unwrap();
            assert!(
                most <= bound,
                "{cfg:?} on {n} rows: a tree has {most} nodes, bound {bound}"
            );
            // The trees fill a good part of what the bound allows (it is not loose by orders
            // of magnitude), also where the depth stops them first.
            assert!(4 * most >= bound, "{cfg:?} on {n} rows: {most} of {bound}");
        }
    }

    /// The subsample, its out-of-bag rest and the recursion's child index vectors are
    /// allocated at their exact lengths, as `memory_bound` counts them: `Iterator::partition`
    /// and `filter().collect()` grow a vector by pushing and can leave close to twice its
    /// length allocated, and the recursion holds child vectors on every level of its path.
    #[test]
    fn index_vectors_are_allocated_at_their_lengths() {
        let mut rng = 5;
        for (n, fraction) in [(1_000, 0.5), (777, 0.632), (61, 1.0), (10, 0.1)] {
            let (sample, oob) = subsample(n, fraction, &mut rng);
            assert_eq!(sample.len() + oob.len(), n);
            assert_eq!(
                sample.capacity(),
                sample.len(),
                "{n} rows, fraction {fraction}"
            );
            assert_eq!(oob.capacity(), oob.len(), "{n} rows, fraction {fraction}");
        }
        // Thresholds that send one row, or all but a few, to one side, as the splits of very
        // unbalanced data do, and balanced ones.
        let x = Array2::from_shape_fn((1_000, 1), |(i, _)| i as f64 + 0.25 * (i % 3) as f64);
        let indices: Vec<usize> = (0..1_000).rev().collect();
        for threshold in [0.1, 3.0, 500.0, 996.9, 2_000.0] {
            let (left, right) = partition_indices(&x.view(), &indices, 0, threshold);
            assert_eq!(left.len() + right.len(), 1_000);
            assert!(left.iter().all(|&i| x[[i, 0]] <= threshold));
            assert!(right.iter().all(|&i| x[[i, 0]] > threshold));
            assert_eq!(left.capacity(), left.len(), "threshold {threshold}");
            assert_eq!(right.capacity(), right.len(), "threshold {threshold}");
        }
    }

    #[test]
    fn memory_bound_grows_with_its_arguments_and_saturates() {
        let d = CausalForestConfig::default();
        let at = |cfg: &CausalForestConfig, n: usize, p: usize| cfg.memory_bound(n, p);
        let base = at(&d, 10_000, 4);
        // Every level of the recursion's path keeps the covariates its split drew.
        let mtry = |m: usize| CausalForestConfig {
            mtry: Some(m),
            ..d.clone()
        };
        assert!(at(&mtry(16), 10_000, 16) > at(&mtry(4), 10_000, 16));
        assert_eq!(at(&mtry(16), 10_000, 16), at(&mtry(1_000), 10_000, 16));
        let more_trees = CausalForestConfig {
            n_trees: 3_000,
            ..d.clone()
        };
        let deeper = CausalForestConfig {
            max_depth: 20,
            ..d.clone()
        };
        let no_honesty = CausalForestConfig {
            honesty: false,
            ..d.clone()
        };
        assert!(at(&more_trees, 10_000, 4) > base);
        assert!(at(&d, 20_000, 4) > base);
        assert!(at(&d, 10_000, 8) > base);
        assert!(at(&no_honesty, 10_000, 4) > base);
        assert!(at(&deeper, 1_000_000, 4) > at(&d, 1_000_000, 4));
        // At the default depth a tree has at most 2^11 - 1 nodes however many rows there are.
        let node = heap_chunk_bytes(size_of::<CausalTreeNode>());
        assert_eq!(d.max_nodes_per_tree(10_000_000), 2_047);
        assert!(u128::from(at(&d, 1_000_000, 4)) > 2_000 * 2_047 * node);
        let huge = CausalForestConfig {
            n_trees: usize::MAX,
            max_depth: usize::MAX,
            min_node_size: 0,
            ..d.clone()
        };
        assert_eq!(at(&huge, usize::MAX, usize::MAX), u64::MAX);
        assert_eq!(at(&d, usize::MAX, 4), u64::MAX);
        assert!(at(&d, 0, 0) > 0);
    }

    /// A `min_node_size` above `usize::MAX / 2` cannot split any node, with or without
    /// honesty: twice it (the rows a node needs in each part to be split) used to overflow,
    /// a panic in a debug build and a product wrapped to a small number in a release build.
    /// Every tree is a single leaf, exactly as at a `min_node_size` of all the rows.
    #[test]
    fn a_min_node_size_above_half_of_usize_neither_panics_nor_wraps() {
        let (x, y, w) = generate_test_data(60, 5);
        for honesty in [true, false] {
            let fit = |min_node_size| {
                let cfg = CausalForestConfig {
                    honesty,
                    min_node_size,
                    ..small_config()
                };
                causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg).unwrap()
            };
            let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let want = fit(60);
            for min_node_size in [usize::MAX / 2 + 1, usize::MAX - 1, usize::MAX] {
                let got = fit(min_node_size);
                let case = format!("honesty {honesty}, min_node_size {min_node_size}");
                assert!(
                    got.trees
                        .iter()
                        .all(|t| matches!(t.root, Some(CausalTreeNode::Leaf { .. }))),
                    "{case}"
                );
                assert_eq!(bits(&got.predictions), bits(&want.predictions), "{case}");
                assert_eq!(
                    bits(&got.variance_estimates),
                    bits(&want.variance_estimates),
                    "{case}"
                );
                assert_eq!(
                    bits(&[got.ate, got.ate_se, got.oob_error]),
                    bits(&[want.ate, want.ate_se, want.oob_error]),
                    "{case}"
                );
            }
        }
        // The rule itself, which a release build would otherwise wrap without a sign: the
        // product saturates above `usize::MAX / 2` and is exact up to it.
        let rows = |min_node_size| CausalTree::new(10, min_node_size, 1).min_rows_to_split();
        assert_eq!(rows(usize::MAX / 2 + 1), usize::MAX);
        assert_eq!(rows(usize::MAX), usize::MAX);
        assert_eq!(rows(usize::MAX / 2), usize::MAX - 1);
        assert_eq!((rows(0), rows(5)), (0, 10));
    }

    /// `run_causal_forest_config` gives the configuration `run_causal_forest` used before it
    /// existed: the arguments, and the defaults where they are `None`.
    #[test]
    fn run_config_takes_the_arguments_and_the_defaults() {
        let c = run_causal_forest_config(None, None, None, None, None);
        let d = CausalForestConfig::default();
        assert_eq!(
            (c.n_trees, c.min_node_size, c.honesty, c.max_depth, c.seed),
            (2000, 5, true, 10, None)
        );
        assert_eq!(
            (c.sample_fraction, c.honesty_fraction, c.mtry),
            (d.sample_fraction, d.honesty_fraction, d.mtry)
        );
        let c = run_causal_forest_config(Some(7), Some(2), Some(false), Some(3), Some(9));
        assert_eq!(
            (c.n_trees, c.min_node_size, c.honesty, c.max_depth, c.seed),
            (7, 2, false, 3, Some(9))
        );
    }

    #[test]
    fn causal_forest_huge_n_trees_is_an_error_not_an_abort() {
        let (x, y, w) = generate_test_data(60, 5);
        let cfg = CausalForestConfig {
            n_trees: 1usize << 44,
            ..small_config()
        };
        let r = with_alloc_ceiling(Some(1 << 30), || {
            causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg)
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn causal_forest_validates_counts_and_fractions() {
        let ok = small_config();
        assert!(validate_causal_forest_config(&ok).is_ok());
        for (cfg, param) in [
            (
                CausalForestConfig {
                    n_trees: 0,
                    ..ok.clone()
                },
                "n_trees",
            ),
            (
                CausalForestConfig {
                    sample_fraction: 1.5,
                    ..ok.clone()
                },
                "sample_fraction",
            ), // looped forever
            (
                CausalForestConfig {
                    sample_fraction: 0.0,
                    ..ok.clone()
                },
                "sample_fraction",
            ),
            (
                CausalForestConfig {
                    sample_fraction: f64::NAN,
                    ..ok.clone()
                },
                "sample_fraction",
            ),
            (
                CausalForestConfig {
                    honesty_fraction: 1.0,
                    ..ok.clone()
                },
                "honesty_fraction",
            ),
            (
                CausalForestConfig {
                    honesty_fraction: 0.0,
                    ..ok.clone()
                },
                "honesty_fraction",
            ),
        ] {
            let e = validate_causal_forest_config(&cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(param)),
                "{e}"
            );
        }
        // honesty_fraction is irrelevant without honesty
        let no_honesty = CausalForestConfig {
            honesty: false,
            honesty_fraction: 1.0,
            ..ok.clone()
        };
        assert!(validate_causal_forest_config(&no_honesty).is_ok());
        // full path: n_trees = 0 used to underflow `preds.len() - 1`
        let (x, y, w) = generate_test_data(60, 5);
        let zero = CausalForestConfig { n_trees: 0, ..ok };
        assert!(causal_forest_arrays(x.view(), y.view(), w.view(), names3(), zero).is_err());
    }

    #[test]
    fn causal_forest_honours_cancellation_and_rejects_non_finite_inputs() {
        let (mut x, y, w) = generate_test_data(60, 5);
        let r = with_budget(cancelled(), || {
            causal_forest_arrays(x.view(), y.view(), w.view(), names3(), small_config())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{:?}", r.err());
        x[[3, 1]] = f64::NAN;
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names3(), small_config());
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("finite")),
            "{:?}",
            r.err()
        );
    }

    /// Verbatim copy of the split search before the sorted sweep (with
    /// `self.min_node_size` as an argument and the criterion inlined), kept
    /// as the reference for the split sweep's guard tests.
    fn naive_find_best_causal_split(
        min_node_size: usize,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_criterion = f64::NEG_INFINITY;
        let mut best_split = None;

        for &feature in features {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try thresholds between consecutive values
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);

                if left_indices.len() < min_node_size || right_indices.len() < min_node_size {
                    continue;
                }

                // Check for treatment variation in both children
                let (left_treated, left_control) = count_treated_control(w, &left_indices);
                let (right_treated, right_control) = count_treated_control(w, &right_indices);

                if left_treated < 2 || left_control < 2 || right_treated < 2 || right_control < 2 {
                    continue;
                }

                // compute_causal_criterion, inlined
                let criterion = {
                    let (tau_left, _, n_t_left, n_c_left) =
                        estimate_treatment_effect(y, w, &left_indices);
                    let (tau_right, _, n_t_right, n_c_right) =
                        estimate_treatment_effect(y, w, &right_indices);

                    let w_left = if n_t_left > 0 && n_c_left > 0 {
                        2.0 * (n_t_left as f64) * (n_c_left as f64) / ((n_t_left + n_c_left) as f64)
                    } else {
                        0.0
                    };

                    let w_right = if n_t_right > 0 && n_c_right > 0 {
                        2.0 * (n_t_right as f64) * (n_c_right as f64)
                            / ((n_t_right + n_c_right) as f64)
                    } else {
                        0.0
                    };

                    w_left * tau_left.powi(2) + w_right * tau_right.powi(2)
                };

                if criterion > best_criterion {
                    best_criterion = criterion;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
    }

    /// The split criterion of `split` over `idx`, recomputed from the
    /// children with `estimate_treatment_effect`; NaN for no split.
    fn split_criterion(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        idx: &[usize],
        split: Option<(usize, f64)>,
    ) -> f64 {
        let Some((feature, threshold)) = split else {
            return f64::NAN;
        };
        let (left, right) = partition_indices(&x.view(), idx, feature, threshold);
        let part = |side: &[usize]| {
            let (tau, _, nt, nc) = estimate_treatment_effect(&y.view(), &w.view(), side);
            2.0 * nt as f64 * nc as f64 / (nt + nc) as f64 * tau * tau
        };
        part(&left) + part(&right)
    }

    /// Mirror-image data: the groups x = 0 and x = 2 hold the same (y, w)
    /// pairs, at least two treated and two control rows each, and x = 1 a
    /// few other rows, all shuffled; y on a grid of 1 / `denom`. The two
    /// thresholds then have exactly equal criteria.
    fn mirrored_data(s: &mut u64, denom: f64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let k = 4 + lcg_random(s) % 3;
        let mut group: Vec<(f64, f64)> = Vec::with_capacity(k);
        for i in 0..k {
            let w = match i {
                0 | 1 => 1.0,
                2 | 3 => 0.0,
                _ => (lcg_random(s) % 2) as f64,
            };
            group.push(((lcg_random(s) % 21) as f64 / denom, w));
        }
        let mut rows: Vec<(f64, f64, f64)> = group.iter().map(|&(y, w)| (0.0, y, w)).collect();
        for _ in 0..1 + lcg_random(s) % 3 {
            let y = (lcg_random(s) % 21) as f64 / denom;
            rows.push((1.0, y, (lcg_random(s) % 2) as f64));
        }
        rows.extend(group.iter().map(|&(y, w)| (2.0, y, w)));
        for i in (1..rows.len()).rev() {
            let j = lcg_random(s) % (i + 1);
            rows.swap(i, j);
        }
        let n = rows.len();
        (
            Array2::from_shape_fn((n, 1), |(i, _)| rows[i].0),
            Array1::from_shape_fn(n, |i| rows[i].1),
            Array1::from_shape_fn(n, |i| rows[i].2),
        )
    }

    /// Mirror-image groups (x = 0 and x = 2 hold the same (y, w) pairs) give
    /// x <= 0.5 and x <= 1.5 exactly equal criteria, 143/3000, on outcomes in
    /// tenths. Rounding decides which one a search sees as larger, so either
    /// is a correct answer; the sweep and the old search must each return one.
    #[test]
    fn causal_split_tie_takes_one_of_the_tied_splits() {
        let x = Array2::from_shape_vec((9, 1), vec![2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0])
            .unwrap();
        let y = Array1::from(vec![0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.5, 0.4, 0.4]);
        let w = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let idx: Vec<usize> = (0..9).collect();
        let tree = CausalTree {
            n_features: 1,
            ..CausalTree::new(10, 1, 1)
        };
        for (who, split) in [
            (
                "sweep",
                tree.find_best_causal_split(&x.view(), &y.view(), &w.view(), &idx, &[0]),
            ),
            (
                "reference",
                naive_find_best_causal_split(1, &x.view(), &y.view(), &w.view(), &idx, &[0]),
            ),
        ] {
            assert!(
                matches!(split, Some((0, t)) if t == 0.5 || t == 1.5),
                "{who}: {split:?}"
            );
            let c = split_criterion(&x, &y, &w, &idx, split);
            assert!((c - 143.0 / 3000.0).abs() < 1e-12, "{who}: criterion {c}");
        }
    }

    /// Reference guard on tie-prone (mirror-image) data. With small integer
    /// outcomes every sum is exact, so the sweep must choose the old search's
    /// split; with outcomes in tenths it must choose that split or a
    /// different one with the same criterion.
    #[test]
    fn causal_split_sweep_on_tied_data_matches_or_ties_the_reference() {
        let mut s = 77u64;
        for denom in [1.0, 10.0] {
            for rep in 0..1000 {
                let (x, y, w) = mirrored_data(&mut s, denom);
                let idx: Vec<usize> = (0..y.len()).collect();
                let min_node_size = [1usize, 3][rep % 2];
                let tree = CausalTree {
                    n_features: 1,
                    ..CausalTree::new(10, min_node_size, 1)
                };
                let got = tree.find_best_causal_split(&x.view(), &y.view(), &w.view(), &idx, &[0]);
                let want = naive_find_best_causal_split(
                    min_node_size,
                    &x.view(),
                    &y.view(),
                    &w.view(),
                    &idx,
                    &[0],
                );
                let ctx = format!("denom={denom} rep={rep} x={x} y={y} w={w}");
                if denom == 1.0 {
                    assert_eq!(got, want, "{ctx}");
                } else if got != want {
                    let (a, b) = (
                        split_criterion(&x, &y, &w, &idx, got),
                        split_criterion(&x, &y, &w, &idx, want),
                    );
                    assert!(
                        (a - b).abs() <= 1e-12 * b.max(1.0),
                        "{ctx}: {got:?} criterion {a} vs {want:?} criterion {b}"
                    );
                }
            }
        }
    }

    /// Reference guard on continuous outcomes (no exact ties in the
    /// criterion): the sorted sweep must choose the same split as the
    /// re-partitioning search it replaced.
    #[test]
    fn causal_split_sweep_matches_naive_reference() {
        let (x, y, w) = generate_test_data(400, 77); // x on a 0.01 grid: many tied values
        let mut s = 1234u64;
        for min_node_size in [1usize, 5, 20] {
            let tree = CausalTree {
                n_features: 3,
                ..CausalTree::new(10, min_node_size, 3)
            };
            for rep in 0..30 {
                let mut idx: Vec<usize> = (0..400).collect();
                for i in (1..idx.len()).rev() {
                    let j = lcg_random(&mut s) % (i + 1);
                    idx.swap(i, j);
                }
                idx.truncate(20 + lcg_random(&mut s) % 380);
                let got =
                    tree.find_best_causal_split(&x.view(), &y.view(), &w.view(), &idx, &[0, 1, 2]);
                let want = naive_find_best_causal_split(
                    min_node_size,
                    &x.view(),
                    &y.view(),
                    &w.view(),
                    &idx,
                    &[0, 1, 2],
                );
                assert_eq!(got, want, "min_node_size={min_node_size} rep={rep}");
            }
        }
    }

    /// Golden (characterization) test: captured on the code before the
    /// sweep and the Welford OOB accumulation, on continuous outcomes where no
    /// criterion tie breaks differently; it passes before and after.
    #[test]
    fn causal_forest_outputs_unchanged_by_the_sweep() {
        const GOLDEN: [u64; 5] = [
            4612786064148362328,
            4597828295416546379,
            4612736549338654852,
            4612690148895398169,
            4590552795750182621,
        ];
        const GOLDEN_OOB: f64 = 0.058642006360354874;
        let (x, y, w) = generate_test_data(200, 42);
        let cfg = CausalForestConfig {
            n_trees: 25,
            max_depth: 5,
            seed: Some(42),
            ..Default::default()
        };
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg).unwrap();
        let got = [
            r.ate,
            r.ate_se,
            r.predictions[0],
            r.predictions[199],
            r.variance_estimates[0],
        ];
        let bits = got.map(f64::to_bits);
        assert_eq!(bits, GOLDEN, "got {bits:?} = {got:?}");
        assert!(
            (r.oob_error - GOLDEN_OOB).abs() <= 1e-12 * GOLDEN_OOB.abs(),
            "oob_error {:?}",
            r.oob_error
        );
    }

    fn generate_test_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = seed;

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);

        for i in 0..n {
            // Generate covariates
            x[[i, 0]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 1]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 2]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;

            // Treatment depends on x0
            w[i] = if x[[i, 0]] + (lcg_random(&mut rng) % 100) as f64 / 200.0 > 0.5 {
                1.0
            } else {
                0.0
            };

            // True CATE: tau(x) = 1 + 2*x0 (heterogeneous treatment effect)
            let tau = 1.0 + 2.0 * x[[i, 0]];

            // Outcome with treatment effect and noise
            let noise = ((lcg_random(&mut rng) % 100) as f64 - 50.0) / 100.0;
            y[i] = 5.0 + x[[i, 0]] + 0.5 * x[[i, 1]] + tau * w[i] + noise;
        }

        (x, y, w)
    }

    #[test]
    fn test_causal_forest_basic() {
        let (x, y, w) = generate_test_data(200, 42);

        let config = CausalForestConfig {
            n_trees: 50,
            min_node_size: 5,
            max_depth: 5,
            honesty: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 200);
        assert_eq!(result.n_trees, 50);
        assert_eq!(result.predictions.len(), 200);
        assert_eq!(result.variance_estimates.len(), 200);

        // ATE should be positive (true ATE is 1 + 2*E[x0] ~ 2)
        assert!(
            result.ate > 0.5,
            "ATE should be positive, got {}",
            result.ate
        );

        // Variable importance: x0 should be most important
        let _x0_importance = result
            .variable_importance
            .iter()
            .find(|(name, _)| name == "x0")
            .map(|(_, imp)| *imp)
            .unwrap_or(0.0);

        println!("Variable importances: {:?}", result.variable_importance);
        println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
    }

    #[test]
    fn test_causal_forest_no_honesty() {
        let (x, y, w) = generate_test_data(100, 123);

        let config = CausalForestConfig {
            n_trees: 20,
            min_node_size: 5,
            max_depth: 4,
            honesty: false,
            seed: Some(123),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        assert_eq!(result.n_obs, 100);
        assert!(result.ate.is_finite());
    }

    #[test]
    fn test_causal_tree_honest_split() {
        let indices: Vec<usize> = (0..100).collect();
        let mut rng = 42u64;

        let (structure, estimation) = honest_split(&indices, 0.5, &mut rng);

        // Check sizes
        assert!(
            estimation.len() >= 45 && estimation.len() <= 55,
            "Estimation size: {}",
            estimation.len()
        );
        assert_eq!(structure.len() + estimation.len(), 100);

        // Check no overlap
        let est_set: std::collections::HashSet<_> = estimation.iter().collect();
        for &idx in &structure {
            assert!(!est_set.contains(&idx));
        }
    }

    #[test]
    fn test_treatment_effect_estimation() {
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let w = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices: Vec<usize> = (0..6).collect();

        let (tau, variance, n_t, n_c) = estimate_treatment_effect(&y.view(), &w.view(), &indices);

        // Treatment effect should be: mean(10,11,12) - mean(1,2,3) = 11 - 2 = 9
        assert!((tau - 9.0).abs() < 0.001, "Expected tau=9, got {}", tau);
        assert_eq!(n_t, 3);
        assert_eq!(n_c, 3);
        assert!(variance.is_finite());
    }

    #[test]
    fn test_average_treatment_effect_subset() {
        let (x, y, w) = generate_test_data(100, 456);

        let config = CausalForestConfig {
            n_trees: 20,
            min_node_size: 5,
            max_depth: 4,
            honesty: true,
            seed: Some(456),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Test full ATE
        let (ate, ci_lower, ci_upper) = average_treatment_effect(&result, None).unwrap();
        assert!(ate.is_finite());
        assert!(ci_lower <= ate && ate <= ci_upper);

        // Test subset ATE
        let subset: Vec<usize> = (0..50).collect();
        let (subset_ate, _, _) = average_treatment_effect(&result, Some(&subset)).unwrap();
        assert!(subset_ate.is_finite());
    }

    #[test]
    fn test_insufficient_data() {
        let x = Array2::zeros((5, 2));
        let y = Array1::zeros(5);
        let w = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let config = CausalForestConfig::default();

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string()],
            config,
        );

        assert!(result.is_err());
    }
}
