//! Decision trees and Random Forest.
//!
//! Pure Rust implementations for regression and classification.
//!
//! # Performance Optimizations
//!
//! The Random Forest implementation uses several optimizations for speed:
//! - **Parallel tree building**: Trees are built concurrently via rayon
//! - **Pre-sorted feature indices**: Sorted index arrays computed once, shared read-only
//! - **Running statistics**: Split finding uses running sums instead of repeated partitioning
//! - **Zero-copy bootstrap**: Trees reference original data via index arrays
//!
//! # References
//!
//! - Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// A node in a decision tree.
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Internal node with a split
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    /// Leaf node with a prediction value
    Leaf { value: f64, n_samples: usize },
}

/// A single decision tree (CART algorithm for regression).
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: Option<TreeNode>,
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    n_features: usize,
}

impl DecisionTree {
    /// Create a new decision tree.
    pub fn new(max_depth: usize, min_samples_split: usize, max_features: Option<usize>) -> Self {
        DecisionTree {
            root: None,
            max_depth,
            min_samples_split,
            max_features,
            n_features: 0,
        }
    }

    /// Fit the tree to training data.
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: Option<&[usize]>,
        rng_state: &mut u64,
    ) {
        self.n_features = x.ncols();
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(self.build_tree(&x, &y, &indices, 0, feature_indices, rng_state));
    }

    /// Build tree recursively.
    fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        available_features: Option<&[usize]>,
        rng_state: &mut u64,
    ) -> TreeNode {
        let n_samples = indices.len();

        // Check stopping conditions
        if depth >= self.max_depth || n_samples < self.min_samples_split || n_samples <= 1 {
            return self.create_leaf(y, indices);
        }

        // Check if all targets are the same
        let target_values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let first_val = target_values[0];
        if target_values.iter().all(|&v| (v - first_val).abs() < 1e-10) {
            return self.create_leaf(y, indices);
        }

        // Select features to consider
        let features_to_try = self.select_features(available_features, rng_state);

        // Find best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices, &features_to_try)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                return self.create_leaf(y, indices);
            }

            let left = self.build_tree(
                x,
                y,
                &left_indices,
                depth + 1,
                available_features,
                rng_state,
            );
            let right = self.build_tree(
                x,
                y,
                &right_indices,
                depth + 1,
                available_features,
                rng_state,
            );

            TreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(y, indices)
        }
    }

    /// Create a leaf node with mean prediction.
    fn create_leaf(&self, y: &ArrayView1<f64>, indices: &[usize]) -> TreeNode {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let value = sum / indices.len() as f64;
        TreeNode::Leaf {
            value,
            n_samples: indices.len(),
        }
    }

    /// Select features to consider for splitting.
    fn select_features(&self, available: Option<&[usize]>, rng_state: &mut u64) -> Vec<usize> {
        let all_features: Vec<usize> = match available {
            Some(features) => features.to_vec(),
            None => (0..self.n_features).collect(),
        };

        match self.max_features {
            Some(max_f) if max_f < all_features.len() => {
                // Random subset of features
                let mut selected = Vec::with_capacity(max_f);
                let mut remaining = all_features.clone();

                for _ in 0..max_f {
                    if remaining.is_empty() {
                        break;
                    }
                    let idx = lcg_random(rng_state) % remaining.len();
                    selected.push(remaining.swap_remove(idx));
                }
                selected
            }
            _ => all_features,
        }
    }

    /// Find the best split for the given indices.
    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_mse = f64::INFINITY;
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

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let mse = self.compute_split_mse(y, &left_indices, &right_indices);

                if mse < best_mse {
                    best_mse = mse;
                    best_split = Some((feature, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    /// Compute weighted MSE for a split.
    fn compute_split_mse(&self, y: &ArrayView1<f64>, left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;

        let left_mse = compute_mse(y, left);
        let right_mse = compute_mse(y, right);

        (left.len() as f64 * left_mse + right.len() as f64 * right_mse) / n
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => 0.0,
        }
    }

    /// Traverse tree to get prediction.
    fn traverse(&self, node: &TreeNode, x: &ArrayView1<f64>) -> f64 {
        match node {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split {
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

    /// Calculate feature importances based on variance reduction.
    pub fn feature_importances(&self) -> Array1<f64> {
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

    fn accumulate_importances(&self, node: &TreeNode, importances: &mut Array1<f64>, weight: f64) {
        if let TreeNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            // Add importance for this feature
            importances[*feature_index] += weight;

            // Get child weights based on samples
            let (left_weight, right_weight) = match (left.as_ref(), right.as_ref()) {
                (TreeNode::Leaf { n_samples: ln, .. }, TreeNode::Leaf { n_samples: rn, .. }) => {
                    let total = (*ln + *rn) as f64;
                    (*ln as f64 / total, *rn as f64 / total)
                }
                _ => (0.5, 0.5),
            };

            self.accumulate_importances(left, importances, weight * left_weight);
            self.accumulate_importances(right, importances, weight * right_weight);
        }
    }

    /// Get a reference to the root node (for TreeSHAP).
    pub fn root(&self) -> Option<&TreeNode> {
        self.root.as_ref()
    }
}

/// Compute MSE for a set of indices.
fn compute_mse(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let mean = sum / indices.len() as f64;

    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len() as f64
}

use super::lcg_random;

// ---------------------------------------------------------------------------
// Optimized internal tree builder for Random Forest
// ---------------------------------------------------------------------------
//
// This builder operates on the original data arrays (no copying) using:
// - Pre-sorted feature indices (computed once, shared across all trees)
// - Boolean membership masks for node splits
// - Running sum/count statistics for O(n) split evaluation per feature
//
// The public DecisionTree API is preserved unchanged. The optimized builder
// produces TreeNode trees that are stored inside DecisionTree wrappers.

/// Pre-sorted indices for each feature, shared across all trees.
struct SortedFeatureIndices {
    /// sorted_indices[feature][rank] = original row index
    sorted_indices: Vec<Vec<usize>>,
}

impl SortedFeatureIndices {
    /// Build pre-sorted index arrays for all features.
    fn new(data: &ArrayView2<f64>) -> Self {
        let n_features = data.ncols();
        let n_samples = data.nrows();

        let sorted_indices: Vec<Vec<usize>> = (0..n_features)
            .into_par_iter()
            .map(|f| {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.sort_by(|&a, &b| {
                    data[[a, f]]
                        .partial_cmp(&data[[b, f]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                indices
            })
            .collect();

        SortedFeatureIndices { sorted_indices }
    }
}

/// Optimized split result: feature, threshold, and the partition of indices.
struct SplitResult {
    feature: usize,
    threshold: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

/// Build a tree on the original data using bootstrap sample indices.
///
/// The tree is built recursively. At each node, `node_indices` are the
/// bootstrap sample rows that reach this node. Pre-sorted feature indices
/// are used with a membership mask for efficient split finding.
fn build_tree_optimized(
    data: &ArrayView2<f64>,
    target: &ArrayView1<f64>,
    node_indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
    n_features: usize,
    sorted_feature_indices: &SortedFeatureIndices,
    rng_state: &mut u64,
) -> TreeNode {
    let n_samples = node_indices.len();

    // Stopping conditions
    if depth >= max_depth || n_samples < min_samples_split || n_samples <= 1 {
        return make_leaf(target, node_indices);
    }

    // Check if all targets are the same
    let first_val = target[node_indices[0]];
    if node_indices
        .iter()
        .all(|&i| (target[i] - first_val).abs() < 1e-10)
    {
        return make_leaf(target, node_indices);
    }

    // Select features to consider
    let features_to_try = select_features_fast(n_features, max_features, rng_state);

    // Build a membership set for the current node's samples.
    // For large n, a boolean mask is faster than a HashSet.
    let total_rows = data.nrows();
    let mut in_node = vec![0u32; total_rows];
    // Count occurrences (bootstrap can have duplicates)
    for &idx in node_indices {
        in_node[idx] += 1;
    }

    // Compute total sum and count for the node (accounting for duplicates)
    let total_sum: f64 = node_indices.iter().map(|&i| target[i]).sum();
    let total_count = n_samples;

    // Find best split using pre-sorted indices and running statistics
    if let Some(split) = find_best_split_optimized(
        data,
        target,
        node_indices,
        &in_node,
        total_sum,
        total_count,
        &features_to_try,
        sorted_feature_indices,
    ) {
        if split.left_indices.is_empty() || split.right_indices.is_empty() {
            return make_leaf(target, node_indices);
        }

        let left = build_tree_optimized(
            data,
            target,
            &split.left_indices,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            n_features,
            sorted_feature_indices,
            rng_state,
        );
        let right = build_tree_optimized(
            data,
            target,
            &split.right_indices,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            n_features,
            sorted_feature_indices,
            rng_state,
        );

        TreeNode::Split {
            feature_index: split.feature,
            threshold: split.threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    } else {
        make_leaf(target, node_indices)
    }
}

fn make_leaf(target: &ArrayView1<f64>, indices: &[usize]) -> TreeNode {
    let sum: f64 = indices.iter().map(|&i| target[i]).sum();
    let value = sum / indices.len() as f64;
    TreeNode::Leaf {
        value,
        n_samples: indices.len(),
    }
}

/// Select a random subset of feature indices.
fn select_features_fast(n_features: usize, max_features: usize, rng_state: &mut u64) -> Vec<usize> {
    if max_features >= n_features {
        return (0..n_features).collect();
    }
    let mut selected = Vec::with_capacity(max_features);
    let mut remaining: Vec<usize> = (0..n_features).collect();
    for _ in 0..max_features {
        if remaining.is_empty() {
            break;
        }
        let idx = lcg_random(rng_state) % remaining.len();
        selected.push(remaining.swap_remove(idx));
    }
    selected
}

/// Find the best split using pre-sorted indices and running statistics.
///
/// For each candidate feature, we iterate through the pre-sorted index array.
/// We skip rows not in the current node (using `in_node` mask). As we scan,
/// we maintain running sum_left/count_left and derive sum_right/count_right
/// from the totals. This gives O(N) split evaluation per feature (where N
/// is the total dataset size, but with early-out opportunities).
///
/// The weighted MSE criterion is:
///   MSE = (n_left * var_left + n_right * var_right) / n_total
/// where var = E[y^2] - (E[y])^2 = (sum_sq / n) - (sum / n)^2
///
/// We can equivalently minimize:
///   sum_sq_left - sum_left^2/n_left + sum_sq_right - sum_right^2/n_right
/// which avoids computing mean/variance explicitly.
fn find_best_split_optimized(
    data: &ArrayView2<f64>,
    target: &ArrayView1<f64>,
    node_indices: &[usize],
    in_node: &[u32],
    total_sum: f64,
    total_count: usize,
    features: &[usize],
    sorted_feature_indices: &SortedFeatureIndices,
) -> Option<SplitResult> {
    let mut best_reduction = f64::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    // Total sum of squares for the node
    let total_sum_sq: f64 = node_indices.iter().map(|&i| target[i] * target[i]).sum();
    // Baseline impurity (proportional to total_sum_sq - total_sum^2 / total_count)
    let baseline = total_sum_sq - total_sum * total_sum / total_count as f64;

    if baseline <= 0.0 {
        // Pure node, no split can improve
        return None;
    }

    for &feature in features {
        let sorted = &sorted_feature_indices.sorted_indices[feature];

        // Running statistics for the left side
        let mut sum_left = 0.0;
        let mut sum_sq_left = 0.0;
        let mut count_left: usize = 0;
        let mut prev_val = f64::NEG_INFINITY;

        for &row_idx in sorted {
            let cnt = in_node[row_idx];
            if cnt == 0 {
                continue;
            }

            let x_val = data[[row_idx, feature]];
            let y_val = target[row_idx];

            // Before adding this point, evaluate split if feature value changed
            // (split between prev_val and x_val)
            if count_left > 0 && x_val > prev_val + 1e-12 {
                let count_right = total_count - count_left;
                if count_right > 0 {
                    // Reduction = sum_left^2/n_left + sum_right^2/n_right - total_sum^2/total_count
                    let sum_right = total_sum - sum_left;
                    let reduction = sum_left * sum_left / count_left as f64
                        + sum_right * sum_right / count_right as f64;

                    if reduction > best_reduction {
                        best_reduction = reduction;
                        best_feature = feature;
                        best_threshold = (prev_val + x_val) / 2.0;
                    }
                }
            }

            // Add this point to the left side (handle bootstrap duplicates)
            let cnt_f = cnt as f64;
            sum_left += y_val * cnt_f;
            sum_sq_left += y_val * y_val * cnt_f;
            count_left += cnt as usize;
            prev_val = x_val;
        }
    }

    // Check that we found a valid split
    if best_reduction <= f64::NEG_INFINITY {
        return None;
    }

    // Partition node_indices by the best split
    let mut left_indices = Vec::with_capacity(node_indices.len() / 2);
    let mut right_indices = Vec::with_capacity(node_indices.len() / 2);
    for &idx in node_indices {
        if data[[idx, best_feature]] <= best_threshold {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    if left_indices.is_empty() || right_indices.is_empty() {
        return None;
    }

    Some(SplitResult {
        feature: best_feature,
        threshold: best_threshold,
        left_indices,
        right_indices,
    })
}

/// Random Forest result.
#[derive(Debug, Clone)]
pub struct RandomForestResult {
    /// Predictions for input data
    pub predictions: Vec<f64>,
    /// Feature importances (mean decrease impurity)
    pub feature_importances: Vec<f64>,
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Out-of-bag R-squared score (if computed)
    pub oob_score: Option<f64>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for RandomForestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Random Forest Results")?;
        writeln!(f, "=====================")?;
        writeln!(f, "Number of trees: {}", self.n_trees)?;

        if let Some(oob) = self.oob_score {
            writeln!(f, "Out-of-bag R\u{00b2} score: {:.4}", oob)?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances:")?;

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {}: {:.4}", name, importance)?;
        }

        if self.feature_importances.len() > 10 {
            writeln!(
                f,
                "  ... ({} more features)",
                self.feature_importances.len() - 10
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;

        // Show prediction statistics
        if !self.predictions.is_empty() {
            let min = self
                .predictions
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = self
                .predictions
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mean: f64 = self.predictions.iter().sum::<f64>() / self.predictions.len() as f64;
            writeln!(f, "  Min: {:.4}, Max: {:.4}, Mean: {:.4}", min, max, mean)?;
        }

        Ok(())
    }
}

/// Run Random Forest regression.
///
/// # Arguments
/// * `data` - Input feature matrix (n_samples x n_features)
/// * `target` - Target values
/// * `n_trees` - Number of trees (default: 100)
/// * `max_depth` - Maximum tree depth (default: 10)
/// * `min_samples_split` - Minimum samples to split (default: 2)
/// * `max_features` - Max features per split ("sqrt", "log2", "all", or number)
/// * `seed` - Random seed
/// * `feature_names` - Optional feature names
///
/// # Performance
///
/// Trees are built in parallel using rayon. Pre-sorted feature indices are
/// computed once and shared across all trees for O(n) split evaluation per
/// feature per node.
pub fn random_forest(
    data: ArrayView2<f64>,
    target: ArrayView1<f64>,
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    max_features: Option<&str>,
    seed: Option<u64>,
    feature_names: Option<Vec<String>>,
) -> Result<RandomForestResult, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for Random Forest".to_string());
    }
    if n_samples != target.len() {
        return Err("Data and target must have same number of samples".to_string());
    }

    let num_trees = n_trees.unwrap_or(100);
    let tree_max_depth = max_depth.unwrap_or(10);
    let tree_min_split = min_samples_split.unwrap_or(2);

    // Parse max_features
    let max_feat = parse_max_features(max_features, n_features);

    let base_seed = seed.unwrap_or(42);

    // Pre-sort feature indices once (shared read-only across all trees)
    let sorted_features = SortedFeatureIndices::new(&data);

    // Pre-generate per-tree seeds and bootstrap samples deterministically
    // so results are reproducible regardless of thread scheduling.
    let mut rng_state = base_seed;
    let tree_configs: Vec<(u64, Vec<usize>, Vec<usize>)> = (0..num_trees)
        .map(|_| {
            let tree_seed = lcg_random(&mut rng_state) as u64;
            let (bootstrap, oob) = bootstrap_sample(n_samples, &mut rng_state);
            (tree_seed, bootstrap, oob)
        })
        .collect();

    // Build trees in parallel
    let tree_results: Vec<(DecisionTree, Vec<(usize, f64)>)> = tree_configs
        .par_iter()
        .map(|(tree_seed, bootstrap_indices, oob_indices)| {
            let mut tree_rng = *tree_seed;

            // Build tree on original data using bootstrap indices (no copy)
            let root = build_tree_optimized(
                &data,
                &target,
                bootstrap_indices,
                0,
                tree_max_depth,
                tree_min_split,
                max_feat,
                n_features,
                &sorted_features,
                &mut tree_rng,
            );

            let tree = DecisionTree {
                root: Some(root),
                max_depth: tree_max_depth,
                min_samples_split: tree_min_split,
                max_features: Some(max_feat),
                n_features,
            };

            // Compute OOB predictions for this tree
            let oob_preds: Vec<(usize, f64)> = oob_indices
                .iter()
                .map(|&idx| (idx, tree.predict_one(&data.row(idx))))
                .collect();

            (tree, oob_preds)
        })
        .collect();

    // Separate trees and OOB predictions
    let mut trees = Vec::with_capacity(num_trees);
    // OOB: accumulate sum and count per sample
    let mut oob_sum = vec![0.0f64; n_samples];
    let mut oob_count = vec![0u32; n_samples];

    for (tree, oob_preds) in tree_results {
        for (idx, pred) in oob_preds {
            oob_sum[idx] += pred;
            oob_count[idx] += 1;
        }
        trees.push(tree);
    }

    // Aggregate predictions in parallel (mean across all trees)
    let predictions: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i);
            let sum: f64 = trees.iter().map(|t| t.predict_one(&row)).sum();
            sum / num_trees as f64
        })
        .collect();

    // Compute OOB score
    let oob_score = compute_oob_score_fast(&oob_sum, &oob_count, &target);

    // Aggregate feature importances
    let mut importances = Array1::zeros(n_features);
    for tree in &trees {
        importances = importances + tree.feature_importances();
    }
    importances /= num_trees as f64;

    Ok(RandomForestResult {
        predictions,
        feature_importances: importances.to_vec(),
        n_trees: num_trees,
        oob_score,
        feature_names,
    })
}

/// Random Forest model with stored trees (for SHAP computation).
///
/// Unlike `RandomForestResult`, this struct retains the individual tree
/// structures, which is necessary for TreeSHAP computation.
#[derive(Debug, Clone)]
pub struct RandomForestModel {
    /// The individual decision trees
    pub trees: Vec<DecisionTree>,
    /// Predictions for training data
    pub predictions: Vec<f64>,
    /// Feature importances (mean decrease impurity)
    pub feature_importances: Vec<f64>,
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Out-of-bag R-squared score (if computed)
    pub oob_score: Option<f64>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Base value (mean prediction, used for SHAP)
    pub base_value: f64,
}

impl std::fmt::Display for RandomForestModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Random Forest Model")?;
        writeln!(f, "===================")?;
        writeln!(f, "Number of trees: {}", self.n_trees)?;
        writeln!(f, "Base value: {:.4}", self.base_value)?;

        if let Some(oob) = self.oob_score {
            writeln!(f, "Out-of-bag R-squared: {:.4}", oob)?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances:")?;

        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {}: {:.4}", name, importance)?;
        }

        Ok(())
    }
}

/// Run Random Forest regression and retain tree structures.
///
/// This function is similar to `random_forest` but returns a `RandomForestModel`
/// that includes the individual trees, enabling TreeSHAP computation.
///
/// # Arguments
/// * `data` - Input feature matrix (n_samples x n_features)
/// * `target` - Target values
/// * `n_trees` - Number of trees (default: 100)
/// * `max_depth` - Maximum tree depth (default: 10)
/// * `min_samples_split` - Minimum samples to split (default: 2)
/// * `max_features` - Max features per split ("sqrt", "log2", "all", or number)
/// * `seed` - Random seed
/// * `feature_names` - Optional feature names
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{random_forest_with_trees, shap_values_model, ShapConfig};
///
/// let model = random_forest_with_trees(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
/// let shap = shap_values_model(&model, x.view(), &ShapConfig::default())?;
/// ```
pub fn random_forest_with_trees(
    data: ArrayView2<f64>,
    target: ArrayView1<f64>,
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    max_features: Option<&str>,
    seed: Option<u64>,
    feature_names: Option<Vec<String>>,
) -> Result<RandomForestModel, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for Random Forest".to_string());
    }
    if n_samples != target.len() {
        return Err("Data and target must have same number of samples".to_string());
    }

    let num_trees = n_trees.unwrap_or(100);
    let tree_max_depth = max_depth.unwrap_or(10);
    let tree_min_split = min_samples_split.unwrap_or(2);

    let max_feat = parse_max_features(max_features, n_features);

    let base_seed = seed.unwrap_or(42);

    // Pre-sort feature indices once (shared read-only across all trees)
    let sorted_features = SortedFeatureIndices::new(&data);

    // Pre-generate per-tree seeds and bootstrap samples deterministically
    let mut rng_state = base_seed;
    let tree_configs: Vec<(u64, Vec<usize>, Vec<usize>)> = (0..num_trees)
        .map(|_| {
            let tree_seed = lcg_random(&mut rng_state) as u64;
            let (bootstrap, oob) = bootstrap_sample(n_samples, &mut rng_state);
            (tree_seed, bootstrap, oob)
        })
        .collect();

    // Build trees in parallel
    let tree_results: Vec<(DecisionTree, Vec<(usize, f64)>)> = tree_configs
        .par_iter()
        .map(|(tree_seed, bootstrap_indices, oob_indices)| {
            let mut tree_rng = *tree_seed;

            let root = build_tree_optimized(
                &data,
                &target,
                bootstrap_indices,
                0,
                tree_max_depth,
                tree_min_split,
                max_feat,
                n_features,
                &sorted_features,
                &mut tree_rng,
            );

            let tree = DecisionTree {
                root: Some(root),
                max_depth: tree_max_depth,
                min_samples_split: tree_min_split,
                max_features: Some(max_feat),
                n_features,
            };

            let oob_preds: Vec<(usize, f64)> = oob_indices
                .iter()
                .map(|&idx| (idx, tree.predict_one(&data.row(idx))))
                .collect();

            (tree, oob_preds)
        })
        .collect();

    // Separate trees and accumulate OOB
    let mut trees = Vec::with_capacity(num_trees);
    let mut oob_sum = vec![0.0f64; n_samples];
    let mut oob_count = vec![0u32; n_samples];

    for (tree, oob_preds) in tree_results {
        for (idx, pred) in oob_preds {
            oob_sum[idx] += pred;
            oob_count[idx] += 1;
        }
        trees.push(tree);
    }

    // Aggregate predictions in parallel
    let predictions: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i);
            let sum: f64 = trees.iter().map(|t| t.predict_one(&row)).sum();
            sum / num_trees as f64
        })
        .collect();

    // Compute base value (mean prediction)
    let base_value = predictions.iter().sum::<f64>() / predictions.len() as f64;

    // Compute OOB score
    let oob_score = compute_oob_score_fast(&oob_sum, &oob_count, &target);

    // Aggregate feature importances
    let mut importances = Array1::zeros(n_features);
    for tree in &trees {
        importances = importances + tree.feature_importances();
    }
    importances /= num_trees as f64;

    Ok(RandomForestModel {
        trees,
        predictions,
        feature_importances: importances.to_vec(),
        n_trees: num_trees,
        oob_score,
        feature_names,
        base_value,
    })
}

/// Parse max_features parameter.
fn parse_max_features(max_features: Option<&str>, n_features: usize) -> usize {
    match max_features {
        Some("sqrt") => (n_features as f64).sqrt().ceil() as usize,
        Some("log2") => (n_features as f64).log2().ceil() as usize,
        Some("all") => n_features,
        Some(s) => s
            .parse()
            .unwrap_or((n_features as f64).sqrt().ceil() as usize),
        None => (n_features as f64).sqrt().ceil() as usize, // Default: sqrt
    }
}

/// Generate bootstrap sample indices.
fn bootstrap_sample(n_samples: usize, rng_state: &mut u64) -> (Vec<usize>, Vec<usize>) {
    let mut in_bag = vec![false; n_samples];
    let mut bootstrap = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let idx = lcg_random(rng_state) % n_samples;
        bootstrap.push(idx);
        in_bag[idx] = true;
    }

    let oob: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();

    (bootstrap, oob)
}

/// Compute OOB R-squared score from accumulated sum/count vectors.
fn compute_oob_score_fast(
    oob_sum: &[f64],
    oob_count: &[u32],
    target: &ArrayView1<f64>,
) -> Option<f64> {
    let n_samples = target.len();
    let mut valid_pred = Vec::new();
    let mut valid_true = Vec::new();

    for i in 0..n_samples {
        if oob_count[i] > 0 {
            valid_pred.push(oob_sum[i] / oob_count[i] as f64);
            valid_true.push(target[i]);
        }
    }

    if valid_true.is_empty() {
        return None;
    }

    let mean_y: f64 = valid_true.iter().sum::<f64>() / valid_true.len() as f64;
    let ss_tot: f64 = valid_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = valid_true
        .iter()
        .zip(valid_pred.iter())
        .map(|(&y, &pred)| (y - pred).powi(2))
        .sum();

    if ss_tot > 0.0 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    }
}

// Keep the old helper functions for backward compatibility (used by DecisionTree::fit)
/// Select rows from a 2D array.
#[allow(dead_code)]
fn select_rows(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Select elements from a 1D array.
#[allow(dead_code)]
fn select_elements(data: &ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_iter(indices.iter().map(|&i| data[i]))
}

/// Compute OOB R-squared score (legacy interface for backward compat).
#[allow(dead_code)]
fn compute_oob_score(
    oob_predictions: &ArrayView2<f64>,
    oob_counts: &ArrayView1<usize>,
    target: &ArrayView1<f64>,
) -> Option<f64> {
    let n_samples = target.len();

    // Compute OOB predictions (mean of non-NaN predictions for each sample)
    let mut oob_pred = Vec::with_capacity(n_samples);
    let mut valid_samples = Vec::new();

    for i in 0..n_samples {
        if oob_counts[i] > 0 {
            let mut sum = 0.0;
            let mut count = 0;

            for j in 0..oob_predictions.ncols() {
                let val = oob_predictions[[i, j]];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }

            if count > 0 {
                oob_pred.push(sum / count as f64);
                valid_samples.push(i);
            }
        }
    }

    if valid_samples.is_empty() {
        return None;
    }

    // Compute R-squared
    let y_true: Vec<f64> = valid_samples.iter().map(|&i| target[i]).collect();
    let mean_y: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;

    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(oob_pred.iter())
        .map(|(&y, &pred)| (y - pred).powi(2))
        .sum();

    if ss_tot > 0.0 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_decision_tree_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut tree = DecisionTree::new(5, 2, None);
        let mut rng = 42u64;
        tree.fit(x.view(), y.view(), None, &mut rng);

        // Predictions should be reasonable
        let pred = tree.predict_one(&array![3.0, 4.0].view());
        assert!(pred > 0.0 && pred < 6.0);
    }

    #[test]
    fn test_random_forest_basic() {
        // Simple linear relationship
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
            [9.0, 0.9],
            [10.0, 1.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = random_forest(
            x.view(),
            y.view(),
            Some(10), // Small for testing
            Some(5),
            Some(2),
            Some("sqrt"),
            Some(42),
            None,
        )
        .unwrap();

        assert_eq!(result.n_trees, 10);
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.feature_importances.len(), 2);

        // Feature 0 should be more important (it's the actual predictor)
        assert!(result.feature_importances[0] >= result.feature_importances[1]);
    }

    #[test]
    fn test_random_forest_feature_importance() {
        // Feature 0 is the only predictor
        let x = array![
            [1.0, 5.0, 2.0],
            [2.0, 3.0, 8.0],
            [3.0, 7.0, 1.0],
            [4.0, 2.0, 9.0],
            [5.0, 8.0, 3.0],
            [6.0, 4.0, 7.0],
            [7.0, 9.0, 4.0],
            [8.0, 1.0, 6.0],
            [9.0, 6.0, 5.0],
            [10.0, 5.0, 2.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // y = feature_0

        let result = random_forest(
            x.view(),
            y.view(),
            Some(50),
            Some(5),
            Some(2),
            Some("all"),
            Some(42),
            Some(vec![
                "predictor".to_string(),
                "noise1".to_string(),
                "noise2".to_string(),
            ]),
        )
        .unwrap();

        // First feature should have highest importance
        assert!(result.feature_importances[0] > result.feature_importances[1]);
        assert!(result.feature_importances[0] > result.feature_importances[2]);
    }

    #[test]
    fn test_bootstrap_sample() {
        let mut rng = 42u64;
        let (bootstrap, oob) = bootstrap_sample(100, &mut rng);

        // Bootstrap should have n samples
        assert_eq!(bootstrap.len(), 100);

        // OOB should have some samples (typically ~37% of n)
        assert!(!oob.is_empty());
        assert!(oob.len() < 100);

        // No overlap between bootstrap unique and oob
        let bootstrap_set: std::collections::HashSet<_> = bootstrap.iter().collect();
        for &idx in &oob {
            assert!(
                !bootstrap_set.contains(&idx)
                    || bootstrap.iter().filter(|&&x| x == idx).count() == 0
            );
        }
    }

    #[test]
    fn test_random_forest_with_trees_basic() {
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
            [9.0, 0.9],
            [10.0, 1.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let model = random_forest_with_trees(
            x.view(),
            y.view(),
            Some(10),
            Some(5),
            Some(2),
            Some("sqrt"),
            Some(42),
            None,
        )
        .unwrap();

        assert_eq!(model.n_trees, 10);
        assert_eq!(model.trees.len(), 10);
        assert_eq!(model.predictions.len(), 10);
        assert_eq!(model.feature_importances.len(), 2);

        // Each tree should have a valid root
        for tree in &model.trees {
            assert!(tree.root().is_some());
        }
    }

    #[test]
    fn test_random_forest_predictions_reasonable() {
        // Larger dataset to test prediction quality
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let v = (i as f64) + 1.0;
            x_data.push(v);
            x_data.push(v * 0.1);
            y_data.push(v + 0.1 * ((i * 7 + 3) % 5) as f64); // y ~ x + small noise
        }
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let result = random_forest(
            x.view(),
            y.view(),
            Some(50),
            Some(8),
            Some(2),
            Some("all"),
            Some(123),
            None,
        )
        .unwrap();

        // Predictions should be correlated with target
        let mean_y = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&actual, &pred)| (actual - pred).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        // R-squared should be positive (model is better than mean)
        assert!(
            r2 > 0.5,
            "R-squared should be > 0.5 for near-linear data, got {}",
            r2
        );
    }
}
