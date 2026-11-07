//! Fast XGBoost implementation with histogram-based splitting and parallelization.
//!
//! Key optimizations over the basic implementation:
//! 1. Histogram-based splitting: O(n) per split instead of O(n log n)
//! 2. Parallel histogram construction with Rayon (the global pool, or n_jobs threads)
//! 3. Histogram subtraction trick for sibling nodes: the larger child's histogram is parent minus
//!    the smaller child's
//! 4. Pre-binned data to avoid repeated sorting
//! 5. Cache-efficient column-major storage for histograms
//!
//! Reference: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// Largest `n_jobs`: a thread pool of this size already exceeds any machine this runs on.
pub const FAST_XGB_MAX_JOBS: usize = 1024;

/// Fast XGBoost configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastXgbConfig {
    /// Number of boosting rounds
    pub n_estimators: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Learning rate (eta)
    pub learning_rate: f64,
    /// L1 penalty on leaf weights (XGBoost reg_alpha): leaves and split gains use
    /// ThresholdL1(G, alpha) = sign(G) max(|G| - alpha, 0)
    pub alpha: f64,
    /// L2 regularization on leaf weights
    pub lambda: f64,
    /// Minimum sum of hessian in a child
    pub min_child_weight: f64,
    /// Subsample ratio of training instances
    pub subsample: f64,
    /// Subsample ratio of columns per tree
    pub colsample_bytree: f64,
    /// Minimum loss reduction to make a split (gamma)
    pub gamma: f64,
    /// Number of histogram bins (max 256 for u8 storage)
    pub max_bin: usize,
    /// Objective function
    pub objective: FastXgbObjective,
    /// Random seed
    pub seed: Option<u64>,
    /// Threads for this fit (0 = rayon's global pool); results do not depend on it
    pub n_jobs: usize,
}

impl Default for FastXgbConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 6,
            learning_rate: 0.3,
            alpha: 0.0,
            lambda: 1.0,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            gamma: 0.0,
            max_bin: 256,
            objective: FastXgbObjective::SquaredError,
            seed: None,
            n_jobs: 0,
        }
    }
}

/// Objective functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FastXgbObjective {
    #[default]
    SquaredError,
    Logistic,
}

/// Pre-binned data structure for fast histogram construction
#[derive(Debug, Clone)]
struct BinnedData {
    /// Bin indices for each observation and feature (n x p), stored as u8
    bins: Vec<Vec<u8>>,
    /// Bin edges for each feature
    bin_edges: Vec<Vec<f64>>,
    /// Number of bins per feature
    n_bins: Vec<usize>,
}

impl BinnedData {
    fn new(x: ArrayView2<f64>, max_bin: usize) -> Self {
        let n = x.nrows();
        let p = x.ncols();
        let max_bin = max_bin.min(256); // u8 max

        let mut bins = vec![vec![0u8; n]; p];
        let mut bin_edges = Vec::with_capacity(p);
        let mut n_bins = Vec::with_capacity(p);

        for j in 0..p {
            let col: Vec<f64> = x.column(j).to_vec();
            let (edges, feature_bins) = compute_bins(&col, max_bin);

            for (i, &b) in feature_bins.iter().enumerate() {
                bins[j][i] = b;
            }

            n_bins.push(edges.len());
            bin_edges.push(edges);
        }

        Self {
            bins,
            bin_edges,
            n_bins,
        }
    }

    #[inline]
    fn get_bin(&self, feature: usize, sample: usize) -> usize {
        self.bins[feature][sample] as usize
    }
}

/// Compute histogram bins for a feature using quantile-based binning
fn compute_bins(values: &[f64], max_bin: usize) -> (Vec<f64>, Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Sort values with indices
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Compute quantile-based bin edges
    let n_bins = max_bin.min(n);
    let mut edges = Vec::with_capacity(n_bins);

    for i in 1..n_bins {
        let idx = (i * n) / n_bins;
        let val = indexed[idx.min(n - 1)].1;
        if edges.is_empty() || val > *edges.last().unwrap() {
            edges.push(val);
        }
    }

    // Assign bins
    let mut bins = vec![0u8; n];
    for &(orig_idx, val) in &indexed {
        let bin = edges.iter().position(|&e| val <= e).unwrap_or(edges.len());
        bins[orig_idx] = bin as u8;
    }

    (edges, bins)
}

/// Gradient histogram for a single feature
#[derive(Debug, Clone)]
struct GradientHistogram {
    grad_sum: Vec<f64>,
    hess_sum: Vec<f64>,
    count: Vec<u32>,
}

impl GradientHistogram {
    fn new(n_bins: usize) -> Self {
        Self {
            grad_sum: vec![0.0; n_bins + 1],
            hess_sum: vec![0.0; n_bins + 1],
            count: vec![0; n_bins + 1],
        }
    }

    fn clear(&mut self) {
        self.grad_sum.fill(0.0);
        self.hess_sum.fill(0.0);
        self.count.fill(0);
    }

    /// Build histogram from samples
    fn build(
        &mut self,
        binned_data: &BinnedData,
        feature: usize,
        gradients: &[f64],
        hessians: &[f64],
        samples: &[usize],
    ) {
        self.clear();
        for &idx in samples {
            let bin = binned_data.get_bin(feature, idx);
            self.grad_sum[bin] += gradients[idx];
            self.hess_sum[bin] += hessians[idx];
            self.count[bin] += 1;
        }
    }

    /// Compute histogram by subtraction (parent - sibling)
    fn subtract_from(&mut self, parent: &GradientHistogram, sibling: &GradientHistogram) {
        for i in 0..self.grad_sum.len() {
            self.grad_sum[i] = parent.grad_sum[i] - sibling.grad_sum[i];
            self.hess_sum[i] = parent.hess_sum[i] - sibling.hess_sum[i];
            self.count[i] = parent.count[i] - sibling.count[i];
        }
    }

    /// Find best split point
    fn find_best_split(
        &self,
        lambda: f64,
        alpha: f64,
        gamma: f64,
        min_child_weight: f64,
    ) -> Option<(usize, f64)> {
        let total_grad: f64 = self.grad_sum.iter().sum();
        let total_hess: f64 = self.hess_sum.iter().sum();
        let total_count: u32 = self.count.iter().sum();
        let t_total = threshold_l1(total_grad, alpha);
        let score_parent = t_total * t_total / (total_hess + lambda);

        if total_hess < min_child_weight {
            return None;
        }

        let mut best_gain = gamma; // Minimum gain threshold
        let mut best_bin = None;

        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count = 0u32;

        for bin in 0..self.grad_sum.len() - 1 {
            left_grad += self.grad_sum[bin];
            left_hess += self.hess_sum[bin];
            left_count += self.count[bin];

            // A boundary with no rows on one side is not a split, whatever gamma allows
            if left_count == 0 || left_count == total_count {
                continue;
            }

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            if left_hess < min_child_weight || right_hess < min_child_weight {
                continue;
            }

            // XGBoost CalcGain on the thresholded sums
            let t_left = threshold_l1(left_grad, alpha);
            let t_right = threshold_l1(right_grad, alpha);
            let gain = 0.5
                * ((t_left * t_left) / (left_hess + lambda)
                    + (t_right * t_right) / (right_hess + lambda)
                    - score_parent);

            if gain > best_gain {
                best_gain = gain;
                best_bin = Some(bin);
            }
        }

        best_bin.map(|b| (b, best_gain))
    }
}

/// Tree node
#[derive(Debug, Clone)]
struct FastXgbNode {
    feature: Option<usize>,
    bin_threshold: usize,
    threshold: f64,
    left: Option<usize>,
    right: Option<usize>,
    weight: f64,
    gain: f64,
}

/// Tree structure
#[derive(Debug, Clone)]
struct FastXgbTree {
    nodes: Vec<FastXgbNode>,
}

impl FastXgbTree {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn predict(&self, binned: &[u8]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.feature.is_none() {
                return node.weight;
            }

            let feat = node.feature.unwrap();
            if binned[feat] as usize <= node.bin_threshold {
                idx = node.left.unwrap();
            } else {
                idx = node.right.unwrap();
            }
        }
    }
}

/// Fast XGBoost result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastXgbResult {
    pub predictions: Vec<f64>,
    pub probabilities: Option<Vec<f64>>,
    pub feature_importances: Vec<f64>,
    pub n_trees: usize,
    pub train_loss: Vec<f64>,
    #[serde(skip)]
    trees: Vec<FastXgbTree>,
    #[serde(skip)]
    binned_data: Option<BinnedData>,
    config: FastXgbConfig,
    base_score: f64,
    /// Threads of the pool the fit ran on
    #[serde(default)]
    pub n_threads: usize,
}

impl std::fmt::Display for FastXgbResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fast XGBoost Model")?;
        writeln!(f, "==================")?;
        writeln!(f, "Trees: {}", self.n_trees)?;
        writeln!(f, "Max bins: {}", self.config.max_bin)?;
        writeln!(f, "Learning rate: {}", self.config.learning_rate)?;
        Ok(())
    }
}

/// XGBoost's ThresholdL1: the sum of gradients soft-thresholded by alpha.
#[inline]
fn threshold_l1(g: f64, alpha: f64) -> f64 {
    if g > alpha {
        g - alpha
    } else if g < -alpha {
        g + alpha
    } else {
        0.0
    }
}

/// XGBoost's CalcWeight: -ThresholdL1(G, alpha) / (H + lambda).
#[inline]
fn leaf_weight(g_sum: f64, h_sum: f64, lambda: f64, alpha: f64) -> f64 {
    -threshold_l1(g_sum, alpha) / (h_sum + lambda)
}

/// Refuse what the fit cannot use: too few rows or columns, non-finite data, a bad alpha or n_jobs.
fn validate_fast_xgb(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &FastXgbConfig,
) -> EconResult<()> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    if x.nrows() < 2 {
        return Err(invalid(format!(
            "fast_xgboost needs at least 2 rows (got {})",
            x.nrows()
        )));
    }
    if x.ncols() < 1 {
        return Err(invalid("fast_xgboost needs at least 1 column".to_string()));
    }
    if !config.alpha.is_finite() || config.alpha < 0.0 {
        return Err(invalid(format!(
            "alpha must be finite and non-negative (got {})",
            config.alpha
        )));
    }
    if config.n_jobs > FAST_XGB_MAX_JOBS {
        return Err(invalid(format!(
            "n_jobs must be at most {FAST_XGB_MAX_JOBS} (got {})",
            config.n_jobs
        )));
    }
    super::xgboost::check_finite_xy(x, y)
}

/// Train a fast XGBoost model with histogram-based splitting
pub fn fast_xgboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &FastXgbConfig,
) -> EconResult<FastXgbResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::InsufficientData {
            required: n,
            provided: y.len(),
            context: "X and y must have same number of rows".to_string(),
        });
    }
    validate_fast_xgb(x, y, config)?;

    if config.n_jobs == 0 {
        return fast_xgboost_inner(x, y, config);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.n_jobs)
        .build()
        .map_err(|e| {
            EconError::Computation(format!(
                "fast_xgboost: could not start {} worker threads: {e}",
                config.n_jobs
            ))
        })?;
    pool.install(|| fast_xgboost_inner(x, y, config))
}

/// The fit itself, on whichever rayon pool it is called from.
fn fast_xgboost_inner(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &FastXgbConfig,
) -> EconResult<FastXgbResult> {
    let n = x.nrows();
    let p = x.ncols();

    // Pre-bin the data (one-time cost)
    let binned_data = BinnedData::new(x, config.max_bin);

    // Initialize predictions
    let base_score = match config.objective {
        FastXgbObjective::SquaredError => y.mean().unwrap_or(0.0),
        FastXgbObjective::Logistic => {
            let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
            let neg = n as f64 - pos;
            (pos / neg).ln().clamp(-10.0, 10.0)
        }
    };

    let y_vec: Vec<f64> = y.to_vec();
    let mut predictions = vec![base_score; n];
    let mut trees = Vec::with_capacity(config.n_estimators);
    let mut train_loss = Vec::with_capacity(config.n_estimators);
    let mut feature_importances = vec![0.0; p];

    // Gradient and hessian buffers
    let mut gradients = vec![0.0; n];
    let mut hessians = vec![0.0; n];

    // Sample indices for subsampling
    let all_samples: Vec<usize> = (0..n).collect();

    for _round in 0..config.n_estimators {
        // Compute gradients and hessians
        compute_gradients_fast(
            &predictions,
            &y_vec,
            &mut gradients,
            &mut hessians,
            config.objective,
        );

        // Row subsampling
        let samples = if config.subsample < 1.0 {
            subsample_indices(&all_samples, config.subsample, config.seed)
        } else {
            all_samples.clone()
        };

        // Column subsampling
        let features: Vec<usize> = if config.colsample_bytree < 1.0 {
            subsample_indices(
                &(0..p).collect::<Vec<_>>(),
                config.colsample_bytree,
                config.seed,
            )
        } else {
            (0..p).collect()
        };

        // Build tree with parallel histogram construction
        let (tree, _) = build_tree_histogram(
            &binned_data,
            &gradients,
            &hessians,
            &samples,
            &features,
            config,
            &mut feature_importances,
        );

        // Update predictions
        for i in 0..n {
            let binned_row: Vec<u8> = (0..p).map(|j| binned_data.bins[j][i]).collect();
            predictions[i] += config.learning_rate * tree.predict(&binned_row);
        }

        // Compute loss
        let loss = compute_loss(&predictions, &y_vec, config.objective);
        train_loss.push(loss);

        trees.push(tree);
    }

    // Normalize feature importances
    let total_imp: f64 = feature_importances.iter().sum();
    if total_imp > 0.0 {
        for imp in &mut feature_importances {
            *imp /= total_imp;
        }
    }

    // Compute final predictions/probabilities
    let (final_preds, probs) = match config.objective {
        FastXgbObjective::SquaredError => (predictions.clone(), None),
        FastXgbObjective::Logistic => {
            let probs: Vec<f64> = predictions
                .iter()
                .map(|&p| 1.0 / (1.0 + (-p).exp()))
                .collect();
            (probs.clone(), Some(probs))
        }
    };

    Ok(FastXgbResult {
        predictions: final_preds,
        probabilities: probs,
        feature_importances,
        n_trees: trees.len(),
        train_loss,
        trees,
        binned_data: Some(binned_data),
        config: config.clone(),
        base_score,
        n_threads: rayon::current_num_threads(),
    })
}

/// Compute gradients and hessians efficiently
fn compute_gradients_fast(
    predictions: &[f64],
    targets: &[f64],
    gradients: &mut [f64],
    hessians: &mut [f64],
    objective: FastXgbObjective,
) {
    match objective {
        FastXgbObjective::SquaredError => {
            // Vectorized squared error gradients
            gradients
                .par_iter_mut()
                .zip(predictions.par_iter().zip(targets.par_iter()))
                .for_each(|(g, (&p, &t))| {
                    *g = p - t;
                });
            hessians.par_iter_mut().for_each(|h| *h = 1.0);
        }
        FastXgbObjective::Logistic => {
            gradients
                .par_iter_mut()
                .zip(hessians.par_iter_mut())
                .zip(predictions.par_iter().zip(targets.par_iter()))
                .for_each(|((g, h), (&p, &t))| {
                    let prob = 1.0 / (1.0 + (-p).exp());
                    *g = prob - t;
                    *h = (prob * (1.0 - prob)).max(1e-10);
                });
        }
    }
}

/// How many nodes' histograms a tree built directly and how many it derived by subtraction.
#[derive(Debug, Clone, Copy, Default)]
struct HistStats {
    built: usize,
    subtracted: usize,
}

/// One histogram per entry of `features`, same order.
fn build_histograms(
    binned_data: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    samples: &[usize],
    features: &[usize],
) -> Vec<GradientHistogram> {
    features
        .par_iter()
        .map(|&feat| {
            let mut hist = GradientHistogram::new(binned_data.n_bins[feat]);
            hist.build(binned_data, feat, gradients, hessians, samples);
            hist
        })
        .collect()
}

/// A node waiting on the construction stack, with its histograms if its sibling's build gave them.
struct PendingNode {
    samples: Vec<usize>,
    depth: usize,
    parent: Option<usize>,
    is_left: bool,
    histograms: Option<Vec<GradientHistogram>>,
}

/// Build tree using histogram-based splitting with parallel construction
fn build_tree_histogram(
    binned_data: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    samples: &[usize],
    features: &[usize],
    config: &FastXgbConfig,
    feature_importances: &mut [f64],
) -> (FastXgbTree, HistStats) {
    let mut tree = FastXgbTree::new();
    let mut stats = HistStats::default();

    // Stack for depth-first construction
    let mut stack = vec![PendingNode {
        samples: samples.to_vec(),
        depth: 0,
        parent: None,
        is_left: false,
        histograms: None,
    }];

    while let Some(pending) = stack.pop() {
        let PendingNode {
            samples: node_samples,
            depth,
            parent: parent_idx,
            is_left,
            histograms,
        } = pending;
        let n_samples = node_samples.len();

        // Compute node statistics
        let grad_sum: f64 = node_samples.iter().map(|&i| gradients[i]).sum();
        let hess_sum: f64 = node_samples.iter().map(|&i| hessians[i]).sum();

        // Leaf weight
        let weight = leaf_weight(grad_sum, hess_sum, config.lambda, config.alpha);

        let node_idx = tree.nodes.len();
        if let Some(parent) = parent_idx {
            if is_left {
                tree.nodes[parent].left = Some(node_idx);
            } else {
                tree.nodes[parent].right = Some(node_idx);
            }
        }
        let leaf = FastXgbNode {
            feature: None,
            bin_threshold: 0,
            threshold: 0.0,
            left: None,
            right: None,
            weight,
            gain: 0.0,
        };

        // Check stopping conditions
        if depth >= config.max_depth || n_samples < 2 || hess_sum < config.min_child_weight {
            tree.nodes.push(leaf);
            continue;
        }

        // The root's histograms, built in parallel for all features; the others came from the parent
        let histograms = histograms.unwrap_or_else(|| {
            stats.built += 1;
            build_histograms(binned_data, gradients, hessians, &node_samples, features)
        });

        // Find best split across all features
        let mut best_split: Option<(usize, usize, f64, f64)> = None; // (feature, bin, gain, threshold)

        for (&feat, hist) in features.iter().zip(&histograms) {
            if let Some((bin, gain)) = hist.find_best_split(
                config.lambda,
                config.alpha,
                config.gamma,
                config.min_child_weight,
            ) {
                if best_split.is_none() || gain > best_split.as_ref().unwrap().2 {
                    let threshold = if bin < binned_data.bin_edges[feat].len() {
                        binned_data.bin_edges[feat][bin]
                    } else {
                        f64::INFINITY
                    };
                    best_split = Some((feat, bin, gain, threshold));
                }
            }
        }

        let Some((feat, bin, gain, threshold)) = best_split else {
            // No valid split, make leaf
            tree.nodes.push(leaf);
            continue;
        };

        // Split node
        tree.nodes.push(FastXgbNode {
            feature: Some(feat),
            bin_threshold: bin,
            threshold,
            left: None,
            right: None,
            weight,
            gain,
        });

        // Update feature importance
        feature_importances[feat] += gain;

        // Partition samples
        let (left_samples, right_samples): (Vec<_>, Vec<_>) = node_samples
            .into_iter()
            .partition(|&i| binned_data.get_bin(feat, i) <= bin);

        // A child gets histograms only if it can still split: build the smaller one, subtract for
        // the larger when both can
        let can_split = |rows: &[usize]| {
            depth + 1 < config.max_depth
                && rows.len() >= 2
                && rows.iter().map(|&i| hessians[i]).sum::<f64>() >= config.min_child_weight
        };
        let build =
            |rows: &[usize]| build_histograms(binned_data, gradients, hessians, rows, features);
        let (left_hist, right_hist) = match (can_split(&left_samples), can_split(&right_samples)) {
            (true, true) => {
                stats.built += 1;
                stats.subtracted += 1;
                let left_is_smaller = left_samples.len() <= right_samples.len();
                let smaller = build(if left_is_smaller {
                    &left_samples
                } else {
                    &right_samples
                });
                let larger: Vec<GradientHistogram> = histograms
                    .par_iter()
                    .zip(smaller.par_iter())
                    .map(|(parent, sibling)| {
                        let mut hist = GradientHistogram::new(parent.grad_sum.len() - 1);
                        hist.subtract_from(parent, sibling);
                        hist
                    })
                    .collect();
                if left_is_smaller {
                    (Some(smaller), Some(larger))
                } else {
                    (Some(larger), Some(smaller))
                }
            }
            (true, false) => {
                stats.built += 1;
                (Some(build(&left_samples)), None)
            }
            (false, true) => {
                stats.built += 1;
                (None, Some(build(&right_samples)))
            }
            (false, false) => (None, None),
        };

        // Add children to stack (right first so left is processed first)
        stack.push(PendingNode {
            samples: right_samples,
            depth: depth + 1,
            parent: Some(node_idx),
            is_left: false,
            histograms: right_hist,
        });
        stack.push(PendingNode {
            samples: left_samples,
            depth: depth + 1,
            parent: Some(node_idx),
            is_left: true,
            histograms: left_hist,
        });
    }

    (tree, stats)
}

fn subsample_indices(indices: &[usize], ratio: f64, seed: Option<u64>) -> Vec<usize> {
    use rand::prelude::*;
    use rand::seq::SliceRandom;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let n_sample = ((indices.len() as f64) * ratio).ceil() as usize;
    let mut sampled = indices.to_vec();
    sampled.shuffle(&mut rng);
    sampled.truncate(n_sample);
    sampled
}

fn compute_loss(predictions: &[f64], targets: &[f64], objective: FastXgbObjective) -> f64 {
    match objective {
        FastXgbObjective::SquaredError => {
            predictions
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| (p - t).powi(2))
                .sum::<f64>()
                / predictions.len() as f64
        }
        FastXgbObjective::Logistic => {
            predictions
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| {
                    let prob = 1.0 / (1.0 + (-p).exp());
                    -(t * prob.ln() + (1.0 - t) * (1.0 - prob).ln())
                })
                .sum::<f64>()
                / predictions.len() as f64
        }
    }
}

/// Make predictions with a trained model
pub fn fast_xgboost_predict(result: &FastXgbResult, x: ArrayView2<f64>) -> Vec<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // Re-bin new data using stored bin edges
    let binned = if let Some(ref bd) = result.binned_data {
        let mut bins = vec![vec![0u8; n]; p];
        for j in 0..p {
            for i in 0..n {
                let val = x[[i, j]];
                let bin = bd.bin_edges[j]
                    .iter()
                    .position(|&e| val <= e)
                    .unwrap_or(bd.bin_edges[j].len());
                bins[j][i] = bin as u8;
            }
        }
        bins
    } else {
        return vec![result.base_score; n];
    };

    let mut predictions = vec![result.base_score; n];

    for tree in &result.trees {
        for i in 0..n {
            let binned_row: Vec<u8> = (0..p).map(|j| binned[j][i]).collect();
            predictions[i] += result.config.learning_rate * tree.predict(&binned_row);
        }
    }

    match result.config.objective {
        FastXgbObjective::SquaredError => predictions,
        FastXgbObjective::Logistic => predictions
            .iter()
            .map(|&p| 1.0 / (1.0 + (-p).exp()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use std::time::Instant;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&Array1::from_vec(v));
        }
        m
    }
    fn case_vector(ds: &crate::data::Dataset, col: &str) -> Array1<f64> {
        Array1::from_vec(numeric_column(ds.df(), col, NaPolicy::Fail).unwrap().0)
    }

    #[test]
    fn test_leaf_weight_and_gain_soft_threshold() {
        // XGBoost CalcWeight: -ThresholdL1(G, alpha) / (H + lambda)
        assert_eq!(leaf_weight(-10.0, 5.0, 1.0, 0.0), 10.0 / 6.0);
        assert_eq!(leaf_weight(-10.0, 5.0, 1.0, 5.0), 5.0 / 6.0);
        assert_eq!(leaf_weight(-10.0, 5.0, 1.0, 12.0), 0.0);
        assert_eq!(leaf_weight(7.0, 5.0, 1.0, 2.0), -5.0 / 6.0);
        // gain on thresholded sums: G_L = -8, H_L = 2, G_R = -2, H_R = 2, lambda 1
        let mut h = GradientHistogram::new(1);
        h.grad_sum = vec![-8.0, -2.0];
        h.hess_sum = vec![2.0, 2.0];
        h.count = vec![2, 2];
        let (bin, g0) = h.find_best_split(1.0, 0.0, 0.0, 1.0).unwrap(); // (lambda, alpha, gamma, mcw)
        assert_eq!(bin, 0);
        assert!((g0 - 4.0 / 3.0).abs() < 1e-13); // 0.5 (64/3 + 4/3 - 100/5)
        let (_, g1) = h.find_best_split(1.0, 1.0, 0.0, 1.0).unwrap();
        assert!((g1 - 7.0 / 30.0).abs() < 1e-13); // 0.5 (49/3 + 1/3 - 81/5)
    }

    #[test]
    fn test_validate_fast_xgboost_l1_stump() {
        let (case, ds) = crate::test_support::load_case("ml_xgboost_l1");
        let x = case_matrix(&ds, &["x1", "x2"]);
        let y = case_vector(&ds, "y");
        let mut chk = Check::new(&case);
        for (alpha, key) in [(0.0, "fast_a0_stump"), (3.0, "fast_a3_stump")] {
            let mut cfg = FastXgbConfig::default();
            cfg.n_estimators = 1;
            cfg.max_depth = 1;
            cfg.learning_rate = 0.3;
            cfg.lambda = 1.0;
            cfg.alpha = alpha;
            cfg.gamma = 0.0;
            cfg.min_child_weight = 1.0;
            let r = fast_xgboost(x.view(), y.view(), &cfg).unwrap();
            let t = &r.trees[0];
            let root = &t.nodes[0];
            let (l, rt) = (&t.nodes[root.left.unwrap()], &t.nodes[root.right.unwrap()]);
            chk.vector(
                key,
                &[
                    root.feature.unwrap() as f64,
                    root.bin_threshold as f64,
                    l.weight,
                    rt.weight,
                ],
                Tol::coef(16),
            );
        }
        chk.finish();
    }

    #[test]
    fn test_validate_fast_xgboost_l1_four_rounds() {
        let (case, ds) = crate::test_support::load_case("ml_xgboost_l1");
        let x = case_matrix(&ds, &["x1", "x2"]);
        let y = case_vector(&ds, "y");
        let mut chk = Check::new(&case);
        for (alpha, key) in [(0.0, "fast_a0_pred4"), (3.0, "fast_a3_pred4")] {
            let mut cfg = FastXgbConfig::default();
            cfg.n_estimators = 4;
            cfg.max_depth = 2;
            cfg.learning_rate = 0.3;
            cfg.lambda = 1.0;
            cfg.alpha = alpha;
            cfg.gamma = 0.0;
            cfg.min_child_weight = 1.0;
            let r = fast_xgboost(x.view(), y.view(), &cfg).unwrap();
            chk.vector(key, &r.predictions, Tol::coef(16));
        }
        chk.finish();
    }

    #[test]
    fn test_fast_xgboost_never_splits_off_an_empty_child() {
        // x takes two values: after the root split each child holds one value, every boundary leaves
        // one side empty with gain 0, and gamma < 0 used to accept that "split"
        let x = Array2::from_shape_fn((30, 1), |(i, _)| (i % 2) as f64);
        let y = Array1::from_shape_fn(30, |i| (i % 2) as f64 + 0.1 * (i % 5) as f64);
        let mut cfg = FastXgbConfig::default();
        cfg.n_estimators = 2;
        cfg.max_depth = 3;
        cfg.gamma = -1.0;
        cfg.min_child_weight = 0.0;
        let r = fast_xgboost(x.view(), y.view(), &cfg).unwrap();
        for t in &r.trees {
            for n in &t.nodes {
                assert!(
                    n.feature.is_none() || (n.left.is_some() && n.right.is_some()),
                    "{n:?}"
                );
            }
        }
    }

    fn noisy_xy(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 0.3).unwrap();
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x[[i, 0]] - x[[i, 1]] + normal.sample(&mut rng));
        (x, y)
    }

    #[test]
    fn test_fast_xgboost_n_jobs_sets_the_pool() {
        let (x, y) = noisy_xy(300, 4, 21);
        let mut cfg = FastXgbConfig::default();
        cfg.n_estimators = 15;
        let mut preds = Vec::new();
        for jobs in [1usize, 2, 3] {
            cfg.n_jobs = jobs;
            let r = fast_xgboost(x.view(), y.view(), &cfg).unwrap();
            assert_eq!(r.n_threads, jobs);
            preds.push(r.predictions);
        }
        cfg.n_jobs = 0;
        let r0 = fast_xgboost(x.view(), y.view(), &cfg).unwrap();
        assert_eq!(r0.n_threads, rayon::current_num_threads());
        // per-feature histograms and the feature-order reduction do not depend on the pool size
        assert!(preds.iter().all(|p| *p == r0.predictions));
    }

    #[test]
    fn test_histogram_subtraction_matches_direct_build() {
        let (x, y) = noisy_xy(200, 3, 22);
        let binned = BinnedData::new(x.view(), 64);
        let g: Vec<f64> = y.iter().map(|v| v - 1.0).collect();
        let h: Vec<f64> = (0..200).map(|i| 0.5 + (i % 7) as f64 / 10.0).collect();
        let all: Vec<usize> = (0..200).collect();
        let (small, large): (Vec<usize>, Vec<usize>) = all.iter().partition(|&&i| x[[i, 1]] < 0.3);
        for f in 0..3 {
            let nb = binned.n_bins[f];
            let mut parent = GradientHistogram::new(nb);
            let mut sib = GradientHistogram::new(nb);
            let mut direct = GradientHistogram::new(nb);
            let mut sub = GradientHistogram::new(nb);
            parent.build(&binned, f, &g, &h, &all);
            sib.build(&binned, f, &g, &h, &small);
            direct.build(&binned, f, &g, &h, &large);
            sub.subtract_from(&parent, &sib);
            assert_eq!(sub.count, direct.count);
            let scale: f64 = g.iter().map(|v| v.abs()).sum();
            for b in 0..=nb {
                assert!((sub.grad_sum[b] - direct.grad_sum[b]).abs() <= 1e-12 * scale);
                assert!((sub.hess_sum[b] - direct.hess_sum[b]).abs() <= 1e-12 * 200.0);
            }
        }
    }

    #[test]
    fn test_fast_xgboost_uses_histogram_subtraction() {
        let (x, y) = noisy_xy(400, 3, 23);
        let binned = BinnedData::new(x.view(), 256);
        let g: Vec<f64> = y.iter().map(|v| v - y.mean().unwrap()).collect();
        let h = vec![1.0; 400];
        let mut cfg = FastXgbConfig::default();
        cfg.max_depth = 4;
        let mut imp = vec![0.0; 3];
        let samples: Vec<usize> = (0..400).collect();
        let (tree, stats) =
            build_tree_histogram(&binned, &g, &h, &samples, &[0, 1, 2], &cfg, &mut imp);
        let splits = tree.nodes.iter().filter(|n| n.feature.is_some()).count();
        assert!(splits >= 3);
        // every split node whose two children are both split candidates derives one by subtraction
        assert!(stats.subtracted >= 1);
        assert!(stats.built >= stats.subtracted + 1); // the root and the smaller children are built directly
    }

    #[test]
    fn test_fast_xgboost_accepts_strided_and_column_major_views() {
        use ndarray::ShapeBuilder;
        let mut rng = StdRng::seed_from_u64(5);
        let data = Array2::from_shape_fn((60, 3), |_| rng.r#gen::<f64>());
        let y = data.column(2); // strided
        let mut xf = Array2::<f64>::zeros((60, 2).f()); // column-major: rows are not contiguous
        xf.assign(&data.slice(ndarray::s![.., 0..2]));
        assert!(xf.row(0).as_slice().is_none());
        let mut cfg = FastXgbConfig::default();
        cfg.n_estimators = 10;
        let r = fast_xgboost(xf.view(), y, &cfg).unwrap();
        let r2 = fast_xgboost(xf.as_standard_layout().view(), y.to_owned().view(), &cfg).unwrap();
        assert_eq!(r.predictions, r2.predictions);
        assert_eq!(fast_xgboost_predict(&r, xf.view()), r.predictions);
    }

    #[test]
    fn test_fast_xgboost_rejects_non_finite_and_bad_parameters() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut x = Array2::from_shape_fn((30, 2), |_| rng.r#gen::<f64>());
        let y = Array1::from_shape_fn(30, |_| rng.r#gen::<f64>());
        x[[3, 0]] = f64::NAN;
        let e = fast_xgboost(x.view(), y.view(), &FastXgbConfig::default()).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("row 3, column 0")),
            "{e}"
        );
        x[[3, 0]] = 0.5;
        let mut bad_alpha = FastXgbConfig::default();
        bad_alpha.alpha = -1.0;
        let mut bad_jobs = FastXgbConfig::default();
        bad_jobs.n_jobs = FAST_XGB_MAX_JOBS + 1;
        for (field, cfg) in [("alpha", bad_alpha), ("n_jobs", bad_jobs)] {
            let e = fast_xgboost(x.view(), y.view(), &cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(field)),
                "{e}"
            );
        }
        let one = Array2::<f64>::zeros((1, 2));
        assert!(
            fast_xgboost(
                one.view(),
                Array1::<f64>::zeros(1).view(),
                &FastXgbConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn test_fast_xgboost_speed() {
        let n = 1000;
        let p = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.3).unwrap();

        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |_| normal.sample(&mut rng));

        let config = FastXgbConfig {
            n_estimators: 100,
            max_depth: 6,
            max_bin: 256,
            seed: Some(42),
            ..Default::default()
        };

        let start = Instant::now();
        let result = fast_xgboost(x.view(), y.view(), &config).unwrap();
        let elapsed = start.elapsed();

        println!(
            "Fast XGBoost: n={}, p={}, trees={}",
            n, p, config.n_estimators
        );
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Final loss: {:.6}", result.train_loss.last().unwrap());

        // Performance-regression guard, asserted only in an optimized build: an
        // unoptimized build is about 100x slower, so a wall-clock budget there
        // depends on the machine and its load. The timing is always printed above.
        if !cfg!(debug_assertions) {
            assert!(
                elapsed.as_secs_f64() < 10.0,
                "Fast XGBoost too slow: {:.2}s",
                elapsed.as_secs_f64()
            );
        }
    }

    #[test]
    fn test_fast_xgboost_accuracy() {
        let n = 200;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.3).unwrap();

        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
            if j < 2 {
                rng.r#gen::<f64>()
            } else {
                normal.sample(&mut rng)
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |_| normal.sample(&mut rng));

        let config = FastXgbConfig {
            n_estimators: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = fast_xgboost(x.view(), y.view(), &config).unwrap();

        // Compute R²
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        println!("Fast XGBoost R²: {:.4}", r2);
        println!("Feature importances: {:?}", result.feature_importances);

        assert!(r2 > 0.90, "R² too low: {:.4}", r2);
        assert!(
            result.feature_importances[0] > result.feature_importances[2],
            "x1 should be more important than x3"
        );
    }
}
