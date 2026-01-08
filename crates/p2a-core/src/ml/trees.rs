//! Decision trees and Random Forest.
//!
//! Pure Rust implementations for regression and classification.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A node in a decision tree.
#[derive(Debug, Clone)]
enum TreeNode {
    /// Internal node with a split
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    /// Leaf node with a prediction value
    Leaf {
        value: f64,
        n_samples: usize,
    },
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
    pub fn new(
        max_depth: usize,
        min_samples_split: usize,
        max_features: Option<usize>,
    ) -> Self {
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

            let left = self.build_tree(x, y, &left_indices, depth + 1, available_features, rng_state);
            let right = self.build_tree(x, y, &right_indices, depth + 1, available_features, rng_state);

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

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&i| x[[i, feature]] <= threshold);

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
    fn compute_split_mse(
        &self,
        y: &ArrayView1<f64>,
        left: &[usize],
        right: &[usize],
    ) -> f64 {
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

/// Simple Linear Congruential Generator for reproducible randomness.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
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
    /// Out-of-bag R² score (if computed)
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
            writeln!(f, "Out-of-bag R² score: {:.4}", oob)?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances:")?;

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = self.feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {}: {:.4}", name, importance)?;
        }

        if self.feature_importances.len() > 10 {
            writeln!(f, "  ... ({} more features)", self.feature_importances.len() - 10)?;
        }

        writeln!(f)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;

        // Show prediction statistics
        if !self.predictions.is_empty() {
            let min = self.predictions.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = self.predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
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

    let mut rng_state = seed.unwrap_or(42);

    // Build forest
    let mut trees = Vec::with_capacity(num_trees);
    let mut oob_predictions: Array2<f64> = Array2::from_elem((n_samples, num_trees), f64::NAN);
    let mut oob_counts: Array1<usize> = Array1::zeros(n_samples);

    for tree_idx in 0..num_trees {
        // Bootstrap sample
        let (bootstrap_indices, oob_indices) = bootstrap_sample(n_samples, &mut rng_state);

        // Extract bootstrap data
        let bootstrap_x = select_rows(&data, &bootstrap_indices);
        let bootstrap_y = select_elements(&target, &bootstrap_indices);

        // Build tree
        let mut tree = DecisionTree::new(tree_max_depth, tree_min_split, Some(max_feat));
        tree.fit(bootstrap_x.view(), bootstrap_y.view(), None, &mut rng_state);

        // OOB predictions
        for &oob_idx in &oob_indices {
            let pred = tree.predict_one(&data.row(oob_idx));
            oob_predictions[[oob_idx, tree_idx]] = pred;
            oob_counts[oob_idx] += 1;
        }

        trees.push(tree);
    }

    // Aggregate predictions (mean across all trees)
    let mut predictions = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let sum: f64 = trees.iter().map(|t| t.predict_one(&data.row(i))).sum();
        predictions.push(sum / num_trees as f64);
    }

    // Compute OOB score
    let oob_score = compute_oob_score(&oob_predictions.view(), &oob_counts.view(), &target);

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

/// Parse max_features parameter.
fn parse_max_features(max_features: Option<&str>, n_features: usize) -> usize {
    match max_features {
        Some("sqrt") => (n_features as f64).sqrt().ceil() as usize,
        Some("log2") => (n_features as f64).log2().ceil() as usize,
        Some("all") => n_features,
        Some(s) => s.parse().unwrap_or((n_features as f64).sqrt().ceil() as usize),
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

/// Select rows from a 2D array.
fn select_rows(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Select elements from a 1D array.
fn select_elements(data: &ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_iter(indices.iter().map(|&i| data[i]))
}

/// Compute OOB R² score.
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

    // Compute R²
    let y_true: Vec<f64> = valid_samples.iter().map(|&i| target[i]).collect();
    let mean_y: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;

    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = y_true.iter().zip(oob_pred.iter())
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
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ];
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
            Some(10),  // Small for testing
            Some(5),
            Some(2),
            Some("sqrt"),
            Some(42),
            None,
        ).unwrap();

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
            Some(vec!["predictor".to_string(), "noise1".to_string(), "noise2".to_string()]),
        ).unwrap();

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
            assert!(!bootstrap_set.contains(&idx) || bootstrap.iter().filter(|&&x| x == idx).count() == 0);
        }
    }
}
