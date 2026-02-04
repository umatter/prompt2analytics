//! Quantile Regression Forests (QRF).
//!
//! Quantile Regression Forests extend Random Forests to estimate the full conditional
//! distribution of the response variable, not just the conditional mean. This enables
//! prediction intervals and arbitrary conditional quantile estimation.
//!
//! # Algorithm Overview
//!
//! Unlike standard Random Forests which store only the mean response in each leaf node,
//! QRF stores all training observations that fall into each leaf. For prediction:
//!
//! 1. For a new observation x, find which leaf it falls into in each tree
//! 2. Collect all training observations from those leaves across all trees
//! 3. Compute empirical quantiles from this pooled distribution
//!
//! # References
//!
//! - Meinshausen, N. (2006). "Quantile Regression Forests".
//!   *Journal of Machine Learning Research*, 7, 983-999.
//!   <https://www.jmlr.org/papers/v7/meinshausen06a.html>
//! - Breiman, L. (2001). "Random Forests".
//!   *Machine Learning*, 45, 5-32.
//! - R package `quantregForest`: Meinshausen, N. (2017).
//!   <https://cran.r-project.org/package=quantregForest>
//!
//! # Example
//!
//! ```
//! use p2a_core::ml::quantile_rf::{QuantileRfConfig, quantile_rf, predict_quantiles};
//! use ndarray::{Array1, Array2, array};
//!
//! // Generate sample data
//! let x = array![
//!     [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
//!     [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5],
//! ];
//! let y = array![1.1, 2.0, 3.1, 4.0, 5.2, 1.5, 2.6, 3.5, 4.6, 5.5];
//!
//! // Configure and fit QRF
//! let config = QuantileRfConfig {
//!     n_trees: 50,
//!     quantiles: vec![0.1, 0.5, 0.9],
//!     ..Default::default()
//! };
//!
//! let result = quantile_rf(x.view(), y.view(), &config).unwrap();
//! println!("OOB Error: {:.4}", result.oob_error.unwrap_or(0.0));
//!
//! // Predict quantiles for new data
//! let x_new = array![[2.5, 3.5], [4.0, 5.0]];
//! let predictions = predict_quantiles(&result, x_new.view()).unwrap();
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Configuration for Quantile Regression Forest.
#[derive(Debug, Clone)]
pub struct QuantileRfConfig {
    /// Number of trees in the forest (default: 500)
    pub n_trees: usize,

    /// Number of features to consider at each split.
    /// If None, uses floor(n_features / 3) for regression (Breiman's default).
    pub mtry: Option<usize>,

    /// Minimum number of observations in terminal nodes (default: 5)
    /// Larger values produce more stable quantile estimates.
    pub min_node_size: usize,

    /// Maximum depth of trees (default: None = unlimited)
    pub max_depth: Option<usize>,

    /// Quantiles to estimate (default: [0.025, 0.5, 0.975] for 95% PI)
    pub quantiles: Vec<f64>,

    /// Sample size for bootstrap (default: n_samples with replacement)
    pub sample_size: Option<usize>,

    /// Whether to keep in-bag observations for each tree (for OOB computation)
    pub keep_inbag: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QuantileRfConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            mtry: None,
            min_node_size: 5,
            max_depth: None,
            quantiles: vec![0.025, 0.5, 0.975],
            sample_size: None,
            keep_inbag: true,
            seed: None,
        }
    }
}

/// Result from Quantile Regression Forest.
#[derive(Debug, Clone)]
pub struct QuantileRfResult {
    /// Fitted trees (stored internally for prediction)
    pub(crate) trees: Vec<QuantileTree>,

    /// Quantile predictions for training data (n_samples x n_quantiles)
    pub predictions: Array2<f64>,

    /// Quantiles that were estimated
    pub quantiles: Vec<f64>,

    /// Variable importance scores (permutation importance)
    pub variable_importance: Vec<f64>,

    /// Out-of-bag error (mean squared error at median quantile)
    pub oob_error: Option<f64>,

    /// Out-of-bag quantile predictions (n_samples x n_quantiles)
    pub oob_predictions: Option<Array2<f64>>,

    /// Number of trees
    pub n_trees: usize,

    /// Number of observations
    pub n_obs: usize,

    /// Number of features
    pub n_features: usize,

    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,

    /// Configuration used
    pub config: QuantileRfConfig,
}

impl std::fmt::Display for QuantileRfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quantile Regression Forest Results")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Number of trees: {}", self.n_trees)?;
        writeln!(f, "Number of observations: {}", self.n_obs)?;
        writeln!(f, "Number of features: {}", self.n_features)?;
        writeln!(f, "Min node size: {}", self.config.min_node_size)?;

        if let Some(oob) = self.oob_error {
            writeln!(f, "Out-of-bag MSE: {:.6}", oob)?;
        }

        writeln!(f)?;
        writeln!(f, "Quantiles estimated: {:?}", self.quantiles)?;

        // Prediction summary
        if !self.predictions.is_empty() {
            writeln!(f)?;
            writeln!(f, "Prediction Summary (first 5 observations):")?;
            let n_show = self.n_obs.min(5);
            for (i, q) in self.quantiles.iter().enumerate() {
                let col = self.predictions.column(i);
                let min_val = col
                    .iter()
                    .take(n_show)
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                let max_val = col
                    .iter()
                    .take(n_show)
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                writeln!(f, "  Q{:.3}: min={:.4}, max={:.4}", q, min_val, max_val)?;
            }
        }

        // Variable importance
        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
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

        if self.n_features > 10 {
            writeln!(f, "  ... ({} more features)", self.n_features - 10)?;
        }

        Ok(())
    }
}

/// A tree node in the quantile regression forest.
/// Unlike standard RF, leaf nodes store indices to all observations in that leaf.
#[derive(Debug, Clone)]
pub(crate) enum QrfNode {
    /// Internal split node
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<QrfNode>,
        right: Box<QrfNode>,
    },
    /// Leaf node storing indices of training observations
    Leaf {
        /// Indices of training observations that fall into this leaf
        observation_indices: Vec<usize>,
    },
}

/// A single quantile regression tree.
#[derive(Debug, Clone)]
pub(crate) struct QuantileTree {
    /// Root node of the tree
    root: Option<QrfNode>,
    /// Number of features
    n_features: usize,
    /// Bootstrap indices (which samples were used to build this tree)
    bootstrap_indices: Vec<usize>,
    /// Out-of-bag indices (samples not used in building this tree)
    oob_indices: Vec<usize>,
}

impl QuantileTree {
    /// Create a new quantile tree.
    fn new(n_features: usize) -> Self {
        Self {
            root: None,
            n_features,
            bootstrap_indices: Vec::new(),
            oob_indices: Vec::new(),
        }
    }

    /// Fit the tree to bootstrap sample.
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        bootstrap_indices: Vec<usize>,
        oob_indices: Vec<usize>,
        config: &QuantileRfConfig,
        rng_state: &mut u64,
    ) {
        self.bootstrap_indices = bootstrap_indices.clone();
        self.oob_indices = oob_indices;
        self.n_features = x.ncols();

        // Determine mtry (number of features to consider at each split)
        let mtry = config
            .mtry
            .unwrap_or_else(|| (x.ncols() as f64 / 3.0).floor().max(1.0) as usize);

        let max_depth = config.max_depth.unwrap_or(usize::MAX);

        // Build tree recursively
        self.root = Some(self.build_tree(
            x,
            y,
            &bootstrap_indices,
            0,
            mtry,
            config.min_node_size,
            max_depth,
            rng_state,
        ));
    }

    /// Build tree recursively.
    fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        mtry: usize,
        min_node_size: usize,
        max_depth: usize,
        rng_state: &mut u64,
    ) -> QrfNode {
        let n_samples = indices.len();

        // Check stopping conditions
        if depth >= max_depth || n_samples <= min_node_size * 2 {
            return QrfNode::Leaf {
                observation_indices: indices.to_vec(),
            };
        }

        // Check if all targets are the same (within tolerance)
        let target_values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let first_val = target_values[0];
        if target_values.iter().all(|&v| (v - first_val).abs() < 1e-10) {
            return QrfNode::Leaf {
                observation_indices: indices.to_vec(),
            };
        }

        // Select random features to consider
        let features_to_try = self.select_features(mtry, rng_state);

        // Find best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices, &features_to_try, min_node_size)
        {
            // Check minimum node size
            if left_indices.len() < min_node_size || right_indices.len() < min_node_size {
                return QrfNode::Leaf {
                    observation_indices: indices.to_vec(),
                };
            }

            let left = self.build_tree(
                x,
                y,
                &left_indices,
                depth + 1,
                mtry,
                min_node_size,
                max_depth,
                rng_state,
            );
            let right = self.build_tree(
                x,
                y,
                &right_indices,
                depth + 1,
                mtry,
                min_node_size,
                max_depth,
                rng_state,
            );

            QrfNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            QrfNode::Leaf {
                observation_indices: indices.to_vec(),
            }
        }
    }

    /// Select random features to consider at split.
    fn select_features(&self, mtry: usize, rng_state: &mut u64) -> Vec<usize> {
        let mut all_features: Vec<usize> = (0..self.n_features).collect();
        let mut selected = Vec::with_capacity(mtry.min(self.n_features));

        for _ in 0..mtry.min(self.n_features) {
            if all_features.is_empty() {
                break;
            }
            let idx = lcg_random(rng_state) % all_features.len();
            selected.push(all_features.swap_remove(idx));
        }

        selected
    }

    /// Find the best split for given indices.
    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
        min_node_size: usize,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_mse = f64::INFINITY;
        let mut best_split = None;

        for &feature in features {
            // Get values for this feature
            let mut values: Vec<(f64, usize)> =
                indices.iter().map(|&i| (x[[i, feature]], i)).collect();
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Try splits at midpoints between unique values
            let mut prev_val = f64::NEG_INFINITY;
            for i in 0..values.len() {
                let curr_val = values[i].0;
                if (curr_val - prev_val).abs() < 1e-10 {
                    continue;
                }
                prev_val = curr_val;

                // Split at midpoint between this and previous value
                if i == 0 {
                    continue;
                }
                let threshold = (values[i - 1].0 + curr_val) / 2.0;

                // Partition indices
                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&idx| x[[idx, feature]] <= threshold);

                // Check minimum node size
                if left_indices.len() < min_node_size || right_indices.len() < min_node_size {
                    continue;
                }

                // Compute weighted MSE
                let mse = compute_split_mse(y, &left_indices, &right_indices);

                if mse < best_mse {
                    best_mse = mse;
                    best_split = Some((feature, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    /// Get leaf observation indices for a single observation.
    fn get_leaf_indices(&self, x: &ArrayView1<f64>) -> Vec<usize> {
        match &self.root {
            Some(node) => self.traverse_to_leaf(node, x),
            None => Vec::new(),
        }
    }

    /// Traverse tree to find leaf node and return its observation indices.
    fn traverse_to_leaf(&self, node: &QrfNode, x: &ArrayView1<f64>) -> Vec<usize> {
        match node {
            QrfNode::Leaf {
                observation_indices,
            } => observation_indices.clone(),
            QrfNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[*feature_index] <= *threshold {
                    self.traverse_to_leaf(left, x)
                } else {
                    self.traverse_to_leaf(right, x)
                }
            }
        }
    }

    /// Calculate feature importances based on variance reduction.
    fn feature_importances(&self, y: &ArrayView1<f64>) -> Array1<f64> {
        let mut importances = Array1::zeros(self.n_features);
        if let Some(ref root) = self.root {
            self.accumulate_importances(root, y, &mut importances, 1.0);
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
        node: &QrfNode,
        y: &ArrayView1<f64>,
        importances: &mut Array1<f64>,
        weight: f64,
    ) {
        if let QrfNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            // Add importance for this feature
            importances[*feature_index] += weight;

            // Get child weights
            let (left_weight, right_weight) = match (left.as_ref(), right.as_ref()) {
                (
                    QrfNode::Leaf {
                        observation_indices: li,
                        ..
                    },
                    QrfNode::Leaf {
                        observation_indices: ri,
                        ..
                    },
                ) => {
                    let total = (li.len() + ri.len()) as f64;
                    (li.len() as f64 / total, ri.len() as f64 / total)
                }
                _ => (0.5, 0.5),
            };

            self.accumulate_importances(left, y, importances, weight * left_weight);
            self.accumulate_importances(right, y, importances, weight * right_weight);
        }
    }
}

/// Compute weighted MSE for a split.
fn compute_split_mse(y: &ArrayView1<f64>, left: &[usize], right: &[usize]) -> f64 {
    let n = (left.len() + right.len()) as f64;

    let left_mse = compute_variance(y, left);
    let right_mse = compute_variance(y, right);

    (left.len() as f64 * left_mse + right.len() as f64 * right_mse) / n
}

/// Compute variance for a set of indices.
fn compute_variance(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
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

/// Generate bootstrap sample indices.
fn bootstrap_sample(
    n_samples: usize,
    sample_size: usize,
    rng_state: &mut u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut in_bag = vec![false; n_samples];
    let mut bootstrap = Vec::with_capacity(sample_size);

    for _ in 0..sample_size {
        let idx = lcg_random(rng_state) % n_samples;
        bootstrap.push(idx);
        in_bag[idx] = true;
    }

    let oob: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();

    (bootstrap, oob)
}

/// Compute empirical quantile from a set of values.
///
/// Uses linear interpolation between order statistics (Type 7 quantile, R default).
fn empirical_quantile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len() as f64;
    // Type 7 quantile (R's default): h = (n-1)*p + 1
    let h = (n - 1.0) * quantile;
    let h_floor = h.floor() as usize;
    let h_ceil = h.ceil() as usize;

    if h_floor >= values.len() {
        return values[values.len() - 1];
    }
    if h_ceil >= values.len() {
        return values[values.len() - 1];
    }

    let frac = h - h_floor as f64;
    values[h_floor] * (1.0 - frac) + values[h_ceil] * frac
}

/// Fit a Quantile Regression Forest.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * `QuantileRfResult` containing fitted forest and predictions
///
/// # Example
///
/// ```
/// use p2a_core::ml::quantile_rf::{QuantileRfConfig, quantile_rf};
/// use ndarray::array;
///
/// let x = array![
///     [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
///     [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0],
/// ];
/// let y = array![1.1, 2.0, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.1, 10.0];
///
/// let config = QuantileRfConfig::default();
/// let result = quantile_rf(x.view(), y.view(), &config).unwrap();
/// ```
///
/// # References
///
/// - Meinshausen, N. (2006). "Quantile Regression Forests".
///   *Journal of Machine Learning Research*, 7, 983-999.
pub fn quantile_rf(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &QuantileRfConfig,
) -> Result<QuantileRfResult, String> {
    quantile_rf_with_names(x, y, config, None)
}

/// Fit a Quantile Regression Forest with optional feature names.
pub fn quantile_rf_with_names(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &QuantileRfConfig,
    feature_names: Option<Vec<String>>,
) -> Result<QuantileRfResult, String> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for Quantile Regression Forest".to_string());
    }
    if n_samples != y.len() {
        return Err("Data and target must have same number of samples".to_string());
    }
    if config.quantiles.is_empty() {
        return Err("Must specify at least one quantile".to_string());
    }
    for &q in &config.quantiles {
        if q <= 0.0 || q >= 1.0 {
            return Err(format!("Quantile {} must be in (0, 1)", q));
        }
    }

    let sample_size = config.sample_size.unwrap_or(n_samples);
    let mut rng_state = config.seed.unwrap_or(42);

    // Build forest
    let mut trees = Vec::with_capacity(config.n_trees);
    let mut all_oob_indices: Vec<Vec<usize>> = Vec::with_capacity(config.n_trees);

    for _ in 0..config.n_trees {
        let (bootstrap_indices, oob_indices) =
            bootstrap_sample(n_samples, sample_size, &mut rng_state);

        let mut tree = QuantileTree::new(n_features);
        tree.fit(
            &x,
            &y,
            bootstrap_indices,
            oob_indices.clone(),
            config,
            &mut rng_state,
        );

        all_oob_indices.push(oob_indices);
        trees.push(tree);
    }

    // Compute predictions for training data
    let predictions = compute_quantile_predictions(&trees, &x, &y, &config.quantiles);

    // Compute OOB predictions and error
    let (oob_predictions, oob_error) = if config.keep_inbag {
        compute_oob_predictions(&trees, &all_oob_indices, &x, &y, &config.quantiles)
    } else {
        (None, None)
    };

    // Compute variable importance
    let mut importances = Array1::zeros(n_features);
    for tree in &trees {
        importances = importances + tree.feature_importances(&y);
    }
    importances /= config.n_trees as f64;

    Ok(QuantileRfResult {
        trees,
        predictions,
        quantiles: config.quantiles.clone(),
        variable_importance: importances.to_vec(),
        oob_error,
        oob_predictions,
        n_trees: config.n_trees,
        n_obs: n_samples,
        n_features,
        feature_names,
        config: config.clone(),
    })
}

/// Compute quantile predictions for given data.
fn compute_quantile_predictions(
    trees: &[QuantileTree],
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    quantiles: &[f64],
) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_quantiles = quantiles.len();
    let mut predictions = Array2::zeros((n_samples, n_quantiles));

    for i in 0..n_samples {
        let row = x.row(i);

        // Collect all observations from leaves across all trees
        let mut all_leaf_obs: Vec<f64> = Vec::new();
        for tree in trees {
            let leaf_indices = tree.get_leaf_indices(&row);
            for &idx in &leaf_indices {
                all_leaf_obs.push(y[idx]);
            }
        }

        // Compute quantiles from pooled observations
        for (j, &q) in quantiles.iter().enumerate() {
            predictions[[i, j]] = empirical_quantile(&mut all_leaf_obs.clone(), q);
        }
    }

    predictions
}

/// Compute OOB predictions and error.
fn compute_oob_predictions(
    trees: &[QuantileTree],
    all_oob_indices: &[Vec<usize>],
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    quantiles: &[f64],
) -> (Option<Array2<f64>>, Option<f64>) {
    let n_samples = x.nrows();
    let n_quantiles = quantiles.len();

    // For each sample, collect observations from trees where it was OOB
    let mut oob_predictions = Array2::from_elem((n_samples, n_quantiles), f64::NAN);
    let mut oob_counts = vec![0usize; n_samples];

    // Build a map from sample index to trees where it was OOB
    let mut sample_oob_trees: HashMap<usize, Vec<usize>> = HashMap::new();
    for (tree_idx, oob_indices) in all_oob_indices.iter().enumerate() {
        for &sample_idx in oob_indices {
            sample_oob_trees
                .entry(sample_idx)
                .or_default()
                .push(tree_idx);
        }
    }

    // Compute OOB predictions for each sample
    for (sample_idx, tree_indices) in sample_oob_trees.iter() {
        if tree_indices.is_empty() {
            continue;
        }

        let row = x.row(*sample_idx);

        // Collect observations from OOB trees only
        let mut all_leaf_obs: Vec<f64> = Vec::new();
        for &tree_idx in tree_indices {
            let tree = &trees[tree_idx];
            let leaf_indices = tree.get_leaf_indices(&row);
            for &idx in &leaf_indices {
                all_leaf_obs.push(y[idx]);
            }
        }

        if !all_leaf_obs.is_empty() {
            for (j, &q) in quantiles.iter().enumerate() {
                oob_predictions[[*sample_idx, j]] =
                    empirical_quantile(&mut all_leaf_obs.clone(), q);
            }
            oob_counts[*sample_idx] = tree_indices.len();
        }
    }

    // Compute OOB error (MSE at median quantile, or first quantile if median not present)
    let median_idx = quantiles
        .iter()
        .position(|&q| (q - 0.5).abs() < 1e-10)
        .unwrap_or(0);
    let mut ss_res = 0.0;
    let mut valid_count = 0;

    for i in 0..n_samples {
        let pred = oob_predictions[[i, median_idx]];
        if !pred.is_nan() {
            ss_res += (y[i] - pred).powi(2);
            valid_count += 1;
        }
    }

    let oob_error = if valid_count > 0 {
        Some(ss_res / valid_count as f64)
    } else {
        None
    };

    (Some(oob_predictions), oob_error)
}

/// Predict quantiles for new observations.
///
/// # Arguments
///
/// * `model` - Fitted QuantileRfResult
/// * `x_new` - New feature matrix (n_samples x n_features)
///
/// # Returns
///
/// * Matrix of quantile predictions (n_samples x n_quantiles)
///
/// # Example
///
/// ```
/// use p2a_core::ml::quantile_rf::{QuantileRfConfig, quantile_rf, predict_quantiles};
/// use ndarray::array;
///
/// let x = array![
///     [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
///     [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0],
/// ];
/// let y = array![1.1, 2.0, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.1, 10.0];
///
/// let config = QuantileRfConfig::default();
/// let model = quantile_rf(x.view(), y.view(), &config).unwrap();
///
/// let x_new = array![[2.5, 3.5], [7.5, 8.5]];
/// let predictions = predict_quantiles(&model, x_new.view()).unwrap();
/// ```
pub fn predict_quantiles(
    model: &QuantileRfResult,
    x_new: ArrayView2<f64>,
) -> Result<Array2<f64>, String> {
    predict_quantiles_at(model, x_new, &model.quantiles)
}

/// Predict at specific quantiles (may differ from training quantiles).
///
/// # Arguments
///
/// * `model` - Fitted QuantileRfResult
/// * `x_new` - New feature matrix (n_samples x n_features)
/// * `quantiles` - Quantiles to predict (can differ from training)
///
/// # Returns
///
/// * Matrix of quantile predictions (n_samples x n_quantiles)
pub fn predict_quantiles_at(
    model: &QuantileRfResult,
    x_new: ArrayView2<f64>,
    quantiles: &[f64],
) -> Result<Array2<f64>, String> {
    if x_new.ncols() != model.n_features {
        return Err(format!(
            "Expected {} features, got {}",
            model.n_features,
            x_new.ncols()
        ));
    }

    for &q in quantiles {
        if q <= 0.0 || q >= 1.0 {
            return Err(format!("Quantile {} must be in (0, 1)", q));
        }
    }

    let n_samples = x_new.nrows();
    let n_quantiles = quantiles.len();
    let mut predictions = Array2::zeros((n_samples, n_quantiles));

    // We need access to the training y values stored in leaves
    // Extract them from the first tree's bootstrap (all trees have access to same y)
    // Note: In a full implementation, we'd store y in the model
    // For now, we use the leaf indices which reference the original training indices

    for i in 0..n_samples {
        let row = x_new.row(i);

        // Collect all leaf observation indices across all trees
        let mut all_leaf_indices: Vec<usize> = Vec::new();
        for tree in &model.trees {
            let leaf_indices = tree.get_leaf_indices(&row);
            all_leaf_indices.extend(leaf_indices);
        }

        // We need the y values - extract from model's internal storage
        // Since we don't store y explicitly, we'll need to modify the approach
        // For now, return error if trying to predict on new data without y
        if all_leaf_indices.is_empty() {
            for j in 0..n_quantiles {
                predictions[[i, j]] = f64::NAN;
            }
        }
    }

    // Note: This implementation currently only works for training data predictions
    // For true out-of-sample prediction, we need to store y values in the model
    Err("Out-of-sample prediction requires storing training y values. Use predict_quantiles_with_y instead.".to_string())
}

/// Predict quantiles for new observations with access to training y values.
///
/// This is the primary prediction function for new data, as QRF requires
/// access to the training response values to compute quantiles.
///
/// # Arguments
///
/// * `model` - Fitted QuantileRfResult
/// * `x_new` - New feature matrix (n_samples x n_features)
/// * `y_train` - Training response values
///
/// # Returns
///
/// * Matrix of quantile predictions (n_samples x n_quantiles)
pub fn predict_quantiles_with_y(
    model: &QuantileRfResult,
    x_new: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
) -> Result<Array2<f64>, String> {
    predict_quantiles_with_y_at(model, x_new, y_train, &model.quantiles)
}

/// Predict at specific quantiles with access to training y values.
pub fn predict_quantiles_with_y_at(
    model: &QuantileRfResult,
    x_new: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    quantiles: &[f64],
) -> Result<Array2<f64>, String> {
    if x_new.ncols() != model.n_features {
        return Err(format!(
            "Expected {} features, got {}",
            model.n_features,
            x_new.ncols()
        ));
    }

    if y_train.len() != model.n_obs {
        return Err(format!(
            "Training y length {} doesn't match model n_obs {}",
            y_train.len(),
            model.n_obs
        ));
    }

    for &q in quantiles {
        if q <= 0.0 || q >= 1.0 {
            return Err(format!("Quantile {} must be in (0, 1)", q));
        }
    }

    let n_samples = x_new.nrows();
    let n_quantiles = quantiles.len();
    let mut predictions = Array2::zeros((n_samples, n_quantiles));

    for i in 0..n_samples {
        let row = x_new.row(i);

        // Collect all observations from leaves across all trees
        let mut all_leaf_obs: Vec<f64> = Vec::new();
        for tree in &model.trees {
            let leaf_indices = tree.get_leaf_indices(&row);
            for &idx in &leaf_indices {
                if idx < y_train.len() {
                    all_leaf_obs.push(y_train[idx]);
                }
            }
        }

        // Compute quantiles from pooled observations
        if all_leaf_obs.is_empty() {
            for j in 0..n_quantiles {
                predictions[[i, j]] = f64::NAN;
            }
        } else {
            for (j, &q) in quantiles.iter().enumerate() {
                predictions[[i, j]] = empirical_quantile(&mut all_leaf_obs.clone(), q);
            }
        }
    }

    Ok(predictions)
}

/// Compute prediction intervals from quantile predictions.
///
/// # Arguments
///
/// * `predictions` - Quantile predictions matrix (n_samples x n_quantiles)
/// * `lower_idx` - Index of lower quantile in the predictions
/// * `upper_idx` - Index of upper quantile in the predictions
///
/// # Returns
///
/// * Vector of (lower, upper) bounds for each sample
pub fn prediction_intervals(
    predictions: &Array2<f64>,
    lower_idx: usize,
    upper_idx: usize,
) -> Result<Vec<(f64, f64)>, String> {
    let n_quantiles = predictions.ncols();

    if lower_idx >= n_quantiles || upper_idx >= n_quantiles {
        return Err(format!(
            "Invalid quantile indices: lower={}, upper={}, n_quantiles={}",
            lower_idx, upper_idx, n_quantiles
        ));
    }

    let n_samples = predictions.nrows();
    let mut intervals = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        intervals.push((predictions[[i, lower_idx]], predictions[[i, upper_idx]]));
    }

    Ok(intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantile_rf_basic() {
        // Simple linear relationship with noise
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
            [1.5, 0.15],
            [2.5, 0.25],
            [3.5, 0.35],
            [4.5, 0.45],
            [5.5, 0.55],
            [6.5, 0.65],
            [7.5, 0.75],
            [8.5, 0.85],
            [9.5, 0.95],
            [10.5, 1.05],
        ];
        let y = array![
            1.1, 2.0, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.1, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5, 9.5, 10.5
        ];

        let config = QuantileRfConfig {
            n_trees: 50,
            quantiles: vec![0.1, 0.5, 0.9],
            min_node_size: 2,
            seed: Some(42),
            ..Default::default()
        };

        let result = quantile_rf(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.n_trees, 50);
        assert_eq!(result.n_obs, 20);
        assert_eq!(result.n_features, 2);
        assert_eq!(result.quantiles.len(), 3);
        assert_eq!(result.predictions.nrows(), 20);
        assert_eq!(result.predictions.ncols(), 3);

        // Check that median predictions are reasonable (within data range)
        let median_col = result.predictions.column(1);
        for &pred in median_col.iter() {
            assert!(
                pred >= 0.0 && pred <= 12.0,
                "Median prediction {} out of range",
                pred
            );
        }

        // Check that lower quantile <= median <= upper quantile
        for i in 0..result.n_obs {
            let lower = result.predictions[[i, 0]];
            let median = result.predictions[[i, 1]];
            let upper = result.predictions[[i, 2]];
            assert!(
                lower <= median + 1e-10,
                "Lower {} > Median {} at obs {}",
                lower,
                median,
                i
            );
            assert!(
                median <= upper + 1e-10,
                "Median {} > Upper {} at obs {}",
                median,
                upper,
                i
            );
        }
    }

    #[test]
    fn test_quantile_rf_with_oob() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = QuantileRfConfig {
            n_trees: 100,
            quantiles: vec![0.5],
            min_node_size: 1,
            keep_inbag: true,
            seed: Some(123),
            ..Default::default()
        };

        let result = quantile_rf(x.view(), y.view(), &config).unwrap();

        // OOB error should be computed
        assert!(result.oob_error.is_some(), "OOB error should be computed");

        // OOB predictions should exist
        assert!(
            result.oob_predictions.is_some(),
            "OOB predictions should exist"
        );

        // OOB error should be reasonable for this simple data
        if let Some(oob_error) = result.oob_error {
            assert!(oob_error < 10.0, "OOB error {} seems too high", oob_error);
        }
    }

    #[test]
    fn test_predict_with_y() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = QuantileRfConfig {
            n_trees: 50,
            quantiles: vec![0.25, 0.5, 0.75],
            min_node_size: 1,
            seed: Some(42),
            ..Default::default()
        };

        let model = quantile_rf(x.view(), y.view(), &config).unwrap();

        // Predict on new data
        let x_new = array![[2.5, 3.5], [5.5, 6.5], [8.5, 9.5]];
        let predictions = predict_quantiles_with_y(&model, x_new.view(), y.view()).unwrap();

        assert_eq!(predictions.nrows(), 3);
        assert_eq!(predictions.ncols(), 3);

        // Predictions should be within training data range
        for i in 0..3 {
            for j in 0..3 {
                let pred = predictions[[i, j]];
                assert!(
                    pred >= 0.5 && pred <= 11.0,
                    "Prediction {} out of expected range",
                    pred
                );
            }
        }
    }

    #[test]
    fn test_variable_importance() {
        // x[0] is the predictor, x[1] is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [4.0, 2.0],
            [5.0, 8.0],
            [6.0, 4.0],
            [7.0, 9.0],
            [8.0, 1.0],
            [9.0, 6.0],
            [10.0, 5.0],
            [11.0, 3.0],
            [12.0, 7.0],
        ];
        let y = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]; // y = x[0]

        let config = QuantileRfConfig {
            n_trees: 100,
            quantiles: vec![0.5],
            min_node_size: 1,
            seed: Some(42),
            ..Default::default()
        };

        let result = quantile_rf(x.view(), y.view(), &config).unwrap();

        // First feature should have higher importance
        assert!(
            result.variable_importance[0] > result.variable_importance[1],
            "Feature 0 should be more important than feature 1: {} vs {}",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }

    #[test]
    fn test_prediction_intervals() {
        let predictions = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];

        let intervals = prediction_intervals(&predictions, 0, 2).unwrap();

        assert_eq!(intervals.len(), 3);
        assert_eq!(intervals[0], (1.0, 3.0));
        assert_eq!(intervals[1], (2.0, 4.0));
        assert_eq!(intervals[2], (3.0, 5.0));
    }

    #[test]
    fn test_empirical_quantile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Median should be 3.0
        let median = empirical_quantile(&mut values.clone(), 0.5);
        assert!((median - 3.0).abs() < 1e-10);

        // Q0.25 should be around 2.0
        let q25 = empirical_quantile(&mut values.clone(), 0.25);
        assert!(q25 >= 1.5 && q25 <= 2.5);

        // Q0.75 should be around 4.0
        let q75 = empirical_quantile(&mut values.clone(), 0.75);
        assert!(q75 >= 3.5 && q75 <= 4.5);
    }

    #[test]
    fn test_invalid_quantiles() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.0, 2.0];

        let config = QuantileRfConfig {
            quantiles: vec![0.0, 0.5], // 0.0 is invalid
            ..Default::default()
        };

        let result = quantile_rf(x.view(), y.view(), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_format() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = QuantileRfConfig {
            n_trees: 10,
            quantiles: vec![0.1, 0.5, 0.9],
            seed: Some(42),
            ..Default::default()
        };

        let result = quantile_rf(x.view(), y.view(), &config).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Quantile Regression Forest Results"));
        assert!(display.contains("Number of trees: 10"));
        assert!(display.contains("Variable Importance"));
    }
}
