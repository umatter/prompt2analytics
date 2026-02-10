//! Advanced tree-based methods.
//!
//! Implements conditional inference trees (ctree), C5.0, Quantile Random Forests,
//! Boruta feature selection, Cubist rule-based regression, and MARS.
//!
//! ## Methods
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Quantile RF** | [`quantile_rf`] | Quantile regression forests for prediction intervals |
//! | **CTree** | [`ctree`] | Conditional inference trees with permutation tests |
//! | **Boruta** | [`boruta`] | Feature selection using shadow features |
//! | **C5.0** | [`c50`] | C5.0 decision trees (information gain based) |
//! | **Cubist** | [`cubist`] | Rule-based regression with linear models in leaves |
//! | **MARS** | [`mars`] | Multivariate Adaptive Regression Splines |
//!
//! ## References
//!
//! - Meinshausen, N. (2006). "Quantile Regression Forests."
//!   *Journal of Machine Learning Research*, 7, 983-999.
//! - Hothorn, T., Hornik, K., & Zeileis, A. (2006). "Unbiased Recursive
//!   Partitioning: A Conditional Inference Framework."
//!   *Journal of Computational and Graphical Statistics*, 15(3), 651-674.
//! - Kursa, M. B., & Rudnicki, W. R. (2010). "Feature Selection with the
//!   Boruta Package." *Journal of Statistical Software*, 36(11), 1-13.
//! - Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
//! - Quinlan, J. R. (1992). "Learning with Continuous Classes." *AI'92*.
//! - Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines."
//!   *The Annals of Statistics*, 19(1), 1-67.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

// =============================================================================
// Quantile Random Forests
// =============================================================================

/// Configuration for Quantile Random Forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRfConfig {
    /// Number of trees
    pub n_trees: usize,
    /// Maximum depth
    pub max_depth: usize,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf
    pub min_samples_leaf: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for QuantileRfConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 10,
            min_samples_split: 5,
            min_samples_leaf: 2,
            seed: None,
        }
    }
}

/// Result from Quantile Random Forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRfResult {
    /// Predicted quantiles for each observation
    /// Shape: n_obs x n_quantiles
    pub quantile_predictions: Vec<Vec<f64>>,
    /// Quantiles that were predicted
    pub quantiles: Vec<f64>,
    /// Point predictions (median)
    pub predictions: Vec<f64>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Number of trees
    pub n_trees: usize,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for QuantileRfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quantile Random Forest")?;
        writeln!(f, "======================")?;
        writeln!(f, "Trees: {}", self.n_trees)?;
        writeln!(f, "Quantiles: {:?}", self.quantiles)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;
        writeln!(f)?;

        writeln!(f, "Feature Importances:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, imp) in indexed.iter().take(10) {
            let name = self
                .feature_names
                .as_ref()
                .and_then(|n| n.get(*idx).cloned())
                .unwrap_or_else(|| format!("X{}", idx));
            writeln!(f, "  {}: {:.4}", name, imp)?;
        }

        Ok(())
    }
}

/// Internal tree structure for quantile RF.
struct QrfTree {
    root: Option<QrfNode>,
    n_features: usize,
}

enum QrfNode {
    Split {
        feature: usize,
        threshold: f64,
        left: Box<QrfNode>,
        right: Box<QrfNode>,
    },
    Leaf {
        indices: Vec<usize>,
    },
}

impl QrfTree {
    fn new() -> Self {
        QrfTree {
            root: None,
            n_features: 0,
        }
    }

    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        bootstrap_indices: &[usize],
        config: &QuantileRfConfig,
        rng: &mut u64,
    ) {
        self.n_features = x.ncols();
        let max_features = ((self.n_features as f64).sqrt()).ceil() as usize;
        self.root = Some(self.build_node(x, y, bootstrap_indices, 0, config, max_features, rng));
    }

    fn build_node(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        config: &QuantileRfConfig,
        max_features: usize,
        rng: &mut u64,
    ) -> QrfNode {
        // Check stopping conditions
        if depth >= config.max_depth
            || indices.len() < config.min_samples_split
            || indices.len() <= 2 * config.min_samples_leaf
        {
            return QrfNode::Leaf {
                indices: indices.to_vec(),
            };
        }

        // Select random features
        let features = self.select_features(max_features, rng);

        // Find best split
        if let Some((feature, threshold, left_idx, right_idx)) =
            self.find_best_split(x, y, indices, &features, config)
        {
            if left_idx.len() < config.min_samples_leaf || right_idx.len() < config.min_samples_leaf
            {
                return QrfNode::Leaf {
                    indices: indices.to_vec(),
                };
            }

            let left = self.build_node(x, y, &left_idx, depth + 1, config, max_features, rng);
            let right = self.build_node(x, y, &right_idx, depth + 1, config, max_features, rng);

            QrfNode::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            QrfNode::Leaf {
                indices: indices.to_vec(),
            }
        }
    }

    fn select_features(&self, max_features: usize, rng: &mut u64) -> Vec<usize> {
        let mut features: Vec<usize> = (0..self.n_features).collect();
        for i in (1..self.n_features).rev() {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (*rng >> 33) as usize % (i + 1);
            features.swap(i, j);
        }
        features.truncate(max_features);
        features
    }

    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
        config: &QuantileRfConfig,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_mse = f64::INFINITY;
        let mut best_split = None;

        for &feature in features {
            let mut sorted: Vec<(f64, f64, usize)> = indices
                .iter()
                .map(|&i| (x[[i, feature]], y[i], i))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted.len();
            let total_sum: f64 = sorted.iter().map(|(_, yi, _)| yi).sum();
            let total_ss: f64 = sorted.iter().map(|(_, yi, _)| yi * yi).sum();

            let mut left_sum = 0.0;
            let mut left_ss = 0.0;

            for i in 0..n - 1 {
                let (xi, yi, _) = sorted[i];
                left_sum += yi;
                left_ss += yi * yi;

                let next_x = sorted[i + 1].0;
                if (next_x - xi).abs() < 1e-10 {
                    continue;
                }

                let left_n = i + 1;
                let right_n = n - left_n;

                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }

                let left_mse = left_ss - left_sum * left_sum / left_n as f64;
                let right_sum = total_sum - left_sum;
                let right_ss = total_ss - left_ss;
                let right_mse = right_ss - right_sum * right_sum / right_n as f64;

                let total_mse = left_mse + right_mse;

                if total_mse < best_mse {
                    best_mse = total_mse;
                    let threshold = (xi + next_x) / 2.0;
                    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                        indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
                    best_split = Some((feature, threshold, left_idx, right_idx));
                }
            }
        }

        best_split
    }

    fn get_leaf_indices(&self, x_row: &ArrayView1<f64>) -> &[usize] {
        fn traverse<'a>(node: &'a QrfNode, x: &ArrayView1<f64>) -> &'a [usize] {
            match node {
                QrfNode::Leaf { indices } => indices,
                QrfNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    if x[*feature] <= *threshold {
                        traverse(left, x)
                    } else {
                        traverse(right, x)
                    }
                }
            }
        }

        match &self.root {
            Some(root) => traverse(root, x_row),
            None => &[],
        }
    }
}

/// Quantile Random Forest for prediction intervals.
///
/// Instead of just predicting the mean, QRF keeps track of all training observations
/// that fall into each leaf, allowing prediction of any quantile.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values
/// * `config` - QRF configuration
/// * `quantiles` - Quantiles to predict (e.g., [0.025, 0.5, 0.975] for 95% PI)
///
/// # Returns
///
/// QuantileRfResult with quantile predictions and feature importances.
///
/// # References
///
/// Meinshausen, N. (2006). "Quantile Regression Forests."
/// *Journal of Machine Learning Research*, 7, 983-999.
pub fn quantile_rf(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &QuantileRfConfig,
    quantiles: &[f64],
) -> EconResult<QuantileRfResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_samples,
            context: "Quantile RF".to_string(),
        });
    }

    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    let mut rng = config.seed.unwrap_or(42);

    // Build forest
    let mut trees = Vec::with_capacity(config.n_trees);
    let mut importance = vec![0.0; n_features];

    for _ in 0..config.n_trees {
        // Bootstrap sample
        let bootstrap_indices: Vec<usize> = (0..n_samples)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 33) as usize % n_samples
            })
            .collect();

        let mut tree = QrfTree::new();
        tree.fit(&x, &y, &bootstrap_indices, config, &mut rng);

        // Accumulate feature importance (based on number of splits)
        accumulate_importance(&tree.root, &mut importance);

        trees.push(tree);
    }

    // Normalize importance
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for imp in &mut importance {
            *imp /= sum;
        }
    }

    // Predict quantiles for each observation
    let mut quantile_predictions = Vec::with_capacity(n_samples);
    let mut predictions = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x_row = x.row(i);

        // Collect all y values from leaves across all trees
        let mut all_y_values: Vec<f64> = Vec::new();
        for tree in &trees {
            let leaf_indices = tree.get_leaf_indices(&x_row);
            for &idx in leaf_indices {
                all_y_values.push(y[idx]);
            }
        }

        // Sort for quantile computation
        all_y_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute quantiles
        let n = all_y_values.len();
        let q_vals: Vec<f64> = quantiles
            .iter()
            .map(|&q| {
                if n == 0 {
                    f64::NAN
                } else {
                    let idx = (q * (n - 1) as f64).round() as usize;
                    all_y_values[idx.min(n - 1)]
                }
            })
            .collect();

        quantile_predictions.push(q_vals);

        // Median prediction
        let median_idx = (0.5 * (n - 1) as f64).round() as usize;
        predictions.push(if n > 0 {
            all_y_values[median_idx.min(n - 1)]
        } else {
            f64::NAN
        });
    }

    Ok(QuantileRfResult {
        quantile_predictions,
        quantiles: quantiles.to_vec(),
        predictions,
        feature_importances: importance,
        n_trees: config.n_trees,
        feature_names: None,
    })
}

fn accumulate_importance(node: &Option<QrfNode>, importance: &mut [f64]) {
    if let Some(n) = node {
        match n {
            QrfNode::Split {
                feature,
                left,
                right,
                ..
            } => {
                importance[*feature] += 1.0;
                accumulate_importance(&Some(*left.clone()), importance);
                accumulate_importance(&Some(*right.clone()), importance);
            }
            QrfNode::Leaf { .. } => {}
        }
    }
}

// Need Clone for QrfNode
impl Clone for QrfNode {
    fn clone(&self) -> Self {
        match self {
            QrfNode::Split {
                feature,
                threshold,
                left,
                right,
            } => QrfNode::Split {
                feature: *feature,
                threshold: *threshold,
                left: left.clone(),
                right: right.clone(),
            },
            QrfNode::Leaf { indices } => QrfNode::Leaf {
                indices: indices.clone(),
            },
        }
    }
}

// =============================================================================
// Conditional Inference Trees (CTree)
// =============================================================================

/// Configuration for Conditional Inference Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeConfig {
    /// Significance level for splitting (default: 0.05)
    pub alpha: f64,
    /// Maximum depth
    pub max_depth: usize,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Number of permutations for p-value computation
    pub n_permutations: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CtreeConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_depth: 10,
            min_samples_split: 20,
            n_permutations: 1000,
            seed: None,
        }
    }
}

/// Node in a conditional inference tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeNode {
    /// Node ID
    pub id: usize,
    /// Number of observations
    pub n: usize,
    /// Prediction value
    pub prediction: f64,
    /// P-value for the split (None for leaves)
    pub p_value: Option<f64>,
    /// Split feature (None for leaves)
    pub split_feature: Option<usize>,
    /// Split threshold (None for leaves)
    pub split_threshold: Option<f64>,
    /// Left child (None for leaves)
    pub left: Option<Box<CtreeNode>>,
    /// Right child (None for leaves)
    pub right: Option<Box<CtreeNode>>,
}

/// Result from Conditional Inference Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeResult {
    /// Root node
    pub root: CtreeNode,
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of terminal nodes
    pub n_terminal: usize,
    /// Maximum depth
    pub depth: usize,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for CtreeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Conditional Inference Tree")?;
        writeln!(f, "==========================")?;
        writeln!(f, "Nodes: {} ({} terminal)", self.n_nodes, self.n_terminal)?;
        writeln!(f, "Depth: {}", self.depth)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;
        Ok(())
    }
}

/// Conditional Inference Tree (CTree).
///
/// Unlike CART, CTree uses permutation tests to select split variables,
/// avoiding variable selection bias toward variables with many possible splits.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target values
/// * `config` - CTree configuration
///
/// # Returns
///
/// CtreeResult with the fitted tree.
///
/// # References
///
/// Hothorn, T., Hornik, K., & Zeileis, A. (2006).
/// "Unbiased Recursive Partitioning: A Conditional Inference Framework."
/// *J. Comp. Graph. Stat.*, 15(3), 651-674.
pub fn ctree(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &CtreeConfig,
) -> EconResult<CtreeResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < config.min_samples_split {
        return Err(EconError::InsufficientData {
            required: config.min_samples_split,
            provided: n_samples,
            context: "CTree".to_string(),
        });
    }

    let mut rng = config.seed.unwrap_or(42);
    let indices: Vec<usize> = (0..n_samples).collect();

    let root = build_ctree_node(&x, &y, &indices, 0, 1, config, n_features, &mut rng);

    let (n_nodes, n_terminal, depth) = count_ctree_stats(&root);

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| predict_ctree(&root, &x.row(i)))
        .collect();

    Ok(CtreeResult {
        root,
        n_nodes,
        n_terminal,
        depth,
        predictions,
        feature_names: None,
    })
}

fn build_ctree_node(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    depth: usize,
    node_id: usize,
    config: &CtreeConfig,
    n_features: usize,
    rng: &mut u64,
) -> CtreeNode {
    let n = indices.len();
    let prediction = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;

    // Check stopping conditions
    if depth >= config.max_depth || n < config.min_samples_split {
        return CtreeNode {
            id: node_id,
            n,
            prediction,
            p_value: None,
            split_feature: None,
            split_threshold: None,
            left: None,
            right: None,
        };
    }

    // Find best split variable using permutation test
    let mut best_feature = None;
    let mut best_p_value = 1.0;

    for feature in 0..n_features {
        // Compute test statistic (correlation between feature and response)
        let (_statistic, p_value) =
            permutation_test(x, y, indices, feature, config.n_permutations, rng);

        if p_value < best_p_value {
            best_p_value = p_value;
            best_feature = Some(feature);
        }
    }

    // Check if any variable is significant
    if best_p_value >= config.alpha || best_feature.is_none() {
        return CtreeNode {
            id: node_id,
            n,
            prediction,
            p_value: None,
            split_feature: None,
            split_threshold: None,
            left: None,
            right: None,
        };
    }

    let feature = best_feature.unwrap();

    // Find optimal split point for the selected variable
    let (threshold, left_idx, right_idx) = find_ctree_split(x, y, indices, feature);

    if left_idx.is_empty() || right_idx.is_empty() {
        return CtreeNode {
            id: node_id,
            n,
            prediction,
            p_value: Some(best_p_value),
            split_feature: None,
            split_threshold: None,
            left: None,
            right: None,
        };
    }

    let left = build_ctree_node(
        x,
        y,
        &left_idx,
        depth + 1,
        node_id * 2,
        config,
        n_features,
        rng,
    );
    let right = build_ctree_node(
        x,
        y,
        &right_idx,
        depth + 1,
        node_id * 2 + 1,
        config,
        n_features,
        rng,
    );

    CtreeNode {
        id: node_id,
        n,
        prediction,
        p_value: Some(best_p_value),
        split_feature: Some(feature),
        split_threshold: Some(threshold),
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
    }
}

fn permutation_test(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    feature: usize,
    n_permutations: usize,
    rng: &mut u64,
) -> (f64, f64) {
    let n = indices.len();

    // Compute observed statistic (absolute correlation)
    let x_vals: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
    let y_vals: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    let observed = correlation(&x_vals, &y_vals).abs();

    // Permutation distribution
    let mut count_extreme = 0;
    let mut y_perm = y_vals.clone();

    for _ in 0..n_permutations {
        // Shuffle y_perm
        for i in (1..n).rev() {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (*rng >> 33) as usize % (i + 1);
            y_perm.swap(i, j);
        }

        let perm_stat = correlation(&x_vals, &y_perm).abs();
        if perm_stat >= observed {
            count_extreme += 1;
        }
    }

    let p_value = (count_extreme + 1) as f64 / (n_permutations + 1) as f64;
    (observed, p_value)
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>()
        / n;

    let x_std: f64 = (x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>() / n).sqrt();
    let y_std: f64 = (y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / n).sqrt();

    if x_std > 1e-10 && y_std > 1e-10 {
        cov / (x_std * y_std)
    } else {
        0.0
    }
}

fn find_ctree_split(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    feature: usize,
) -> (f64, Vec<usize>, Vec<usize>) {
    let mut sorted: Vec<(f64, f64, usize)> = indices
        .iter()
        .map(|&i| (x[[i, feature]], y[i], i))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let total_sum: f64 = sorted.iter().map(|(_, yi, _)| yi).sum();

    let mut best_mse = f64::INFINITY;
    let mut best_threshold = sorted[n / 2].0;
    let mut left_sum = 0.0;

    for i in 0..n - 1 {
        let (xi, yi, _) = sorted[i];
        left_sum += yi;

        let next_x = sorted[i + 1].0;
        if (next_x - xi).abs() < 1e-10 {
            continue;
        }

        let left_n = (i + 1) as f64;
        let right_n = (n - i - 1) as f64;

        let left_mean = left_sum / left_n;
        let right_mean = (total_sum - left_sum) / right_n;

        let mse: f64 = sorted[..=i]
            .iter()
            .map(|(_, yi, _)| (yi - left_mean).powi(2))
            .sum::<f64>()
            + sorted[i + 1..]
                .iter()
                .map(|(_, yi, _)| (yi - right_mean).powi(2))
                .sum::<f64>();

        if mse < best_mse {
            best_mse = mse;
            best_threshold = (xi + next_x) / 2.0;
        }
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x[[i, feature]] <= best_threshold);

    (best_threshold, left_idx, right_idx)
}

fn count_ctree_stats(node: &CtreeNode) -> (usize, usize, usize) {
    if node.split_feature.is_none() {
        return (1, 1, 0);
    }

    let (left_n, left_t, left_d) = match &node.left {
        Some(l) => count_ctree_stats(l),
        None => (0, 0, 0),
    };

    let (right_n, right_t, right_d) = match &node.right {
        Some(r) => count_ctree_stats(r),
        None => (0, 0, 0),
    };

    (
        1 + left_n + right_n,
        left_t + right_t,
        1 + left_d.max(right_d),
    )
}

fn predict_ctree(node: &CtreeNode, x_row: &ArrayView1<f64>) -> f64 {
    match (&node.split_feature, &node.split_threshold) {
        (Some(feature), Some(threshold)) => {
            if x_row[*feature] <= *threshold {
                predict_ctree(node.left.as_ref().unwrap(), x_row)
            } else {
                predict_ctree(node.right.as_ref().unwrap(), x_row)
            }
        }
        _ => node.prediction,
    }
}

// =============================================================================
// Boruta Feature Selection
// =============================================================================

/// Configuration for Boruta feature selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorutaConfig {
    /// Maximum number of iterations
    pub max_runs: usize,
    /// P-value threshold for tentative features
    pub p_value: f64,
    /// Number of trees in the random forest
    pub n_trees: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BorutaConfig {
    fn default() -> Self {
        Self {
            max_runs: 100,
            p_value: 0.01,
            n_trees: 100,
            seed: None,
        }
    }
}

/// Feature status in Boruta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureDecision {
    /// Feature is confirmed important
    Confirmed,
    /// Feature is confirmed unimportant
    Rejected,
    /// Feature status is uncertain
    Tentative,
}

/// Result from Boruta feature selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorutaResult {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Decision for each feature
    pub decisions: Vec<FeatureDecision>,
    /// Mean importance for each feature
    pub importance_mean: Vec<f64>,
    /// Mean importance of shadow features (threshold)
    pub shadow_max_mean: f64,
    /// Number of hits (times feature beat max shadow)
    pub hits: Vec<usize>,
    /// Total number of runs
    pub n_runs: usize,
    /// Indices of confirmed features
    pub confirmed_indices: Vec<usize>,
}

impl std::fmt::Display for BorutaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Boruta Feature Selection")?;
        writeln!(f, "========================")?;
        writeln!(f, "Runs: {}", self.n_runs)?;
        writeln!(f, "Shadow max threshold: {:.4}", self.shadow_max_mean)?;
        writeln!(f)?;

        let confirmed: Vec<&String> = self
            .feature_names
            .iter()
            .zip(self.decisions.iter())
            .filter(|(_, d)| **d == FeatureDecision::Confirmed)
            .map(|(n, _)| n)
            .collect();

        writeln!(f, "Confirmed ({}):", confirmed.len())?;
        for name in &confirmed {
            writeln!(f, "  {}", name)?;
        }

        let rejected: Vec<&String> = self
            .feature_names
            .iter()
            .zip(self.decisions.iter())
            .filter(|(_, d)| **d == FeatureDecision::Rejected)
            .map(|(n, _)| n)
            .collect();

        writeln!(f)?;
        writeln!(f, "Rejected ({}):", rejected.len())?;
        for name in rejected.iter().take(10) {
            writeln!(f, "  {}", name)?;
        }

        Ok(())
    }
}

/// Boruta feature selection algorithm.
///
/// Uses shadow features (random permutations of original features) as a benchmark.
/// Features that consistently beat the best shadow feature are confirmed important.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target values
/// * `config` - Boruta configuration
/// * `feature_names` - Optional feature names
///
/// # Returns
///
/// BorutaResult with feature decisions and importance.
///
/// # References
///
/// Kursa, M. B., & Rudnicki, W. R. (2010).
/// "Feature Selection with the Boruta Package."
/// *Journal of Statistical Software*, 36(11), 1-13.
pub fn boruta(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &BorutaConfig,
    feature_names: Option<Vec<String>>,
) -> EconResult<BorutaResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_samples,
            context: "Boruta".to_string(),
        });
    }

    let mut rng = config.seed.unwrap_or(42);

    let names =
        feature_names.unwrap_or_else(|| (0..n_features).map(|i| format!("X{}", i)).collect());

    let mut hits = vec![0usize; n_features];
    let mut importance_sum = vec![0.0; n_features];
    let mut shadow_max_sum = 0.0;

    let mut decisions = vec![FeatureDecision::Tentative; n_features];
    let mut active_features: Vec<usize> = (0..n_features).collect();

    for run in 0..config.max_runs {
        // Check if all features are decided
        if active_features.is_empty() {
            break;
        }

        // Create shadow features (permuted versions of original features)
        let mut x_extended = Array2::zeros((n_samples, n_features * 2));

        // Copy original features
        for j in 0..n_features {
            for i in 0..n_samples {
                x_extended[[i, j]] = x[[i, j]];
            }
        }

        // Create shadow features by permuting rows
        for j in 0..n_features {
            let mut perm: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let k = (rng >> 33) as usize % (i + 1);
                perm.swap(i, k);
            }
            for i in 0..n_samples {
                x_extended[[i, n_features + j]] = x[[perm[i], j]];
            }
        }

        // Train random forest
        let importance = train_rf_importance(&x_extended.view(), &y, config.n_trees, &mut rng);

        // Get max shadow importance
        let shadow_max = importance[n_features..]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        shadow_max_sum += shadow_max;

        // Update hits and importance
        for &j in &active_features {
            importance_sum[j] += importance[j];
            if importance[j] > shadow_max {
                hits[j] += 1;
            }
        }

        // Make decisions using binomial test (simplified)
        let n_runs = run + 1;
        let threshold_confirm = (n_runs as f64 * (1.0 - config.p_value)).ceil() as usize;
        let threshold_reject = (n_runs as f64 * config.p_value).floor() as usize;

        let mut still_active = Vec::new();
        for &j in &active_features {
            if hits[j] >= threshold_confirm {
                decisions[j] = FeatureDecision::Confirmed;
            } else if hits[j] <= threshold_reject && n_runs >= 10 {
                decisions[j] = FeatureDecision::Rejected;
            } else {
                still_active.push(j);
            }
        }
        active_features = still_active;
    }

    let n_runs = config.max_runs;
    let importance_mean: Vec<f64> = importance_sum.iter().map(|&s| s / n_runs as f64).collect();
    let shadow_max_mean = shadow_max_sum / n_runs as f64;

    let confirmed_indices: Vec<usize> = decisions
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == FeatureDecision::Confirmed)
        .map(|(i, _)| i)
        .collect();

    Ok(BorutaResult {
        feature_names: names,
        decisions,
        importance_mean,
        shadow_max_mean,
        hits,
        n_runs,
        confirmed_indices,
    })
}

fn train_rf_importance(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    n_trees: usize,
    rng: &mut u64,
) -> Vec<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let max_depth = 5;
    let min_split = 5;

    let mut importance = vec![0.0; n_features];

    for _ in 0..n_trees {
        // Bootstrap sample
        let bootstrap: Vec<usize> = (0..n_samples)
            .map(|_| {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*rng >> 33) as usize % n_samples
            })
            .collect();

        let mut tree_imp = vec![0.0; n_features];
        build_simple_tree_for_importance(
            x,
            y,
            &bootstrap,
            0,
            max_depth,
            min_split,
            rng,
            &mut tree_imp,
        );

        for j in 0..n_features {
            importance[j] += tree_imp[j];
        }
    }

    // Normalize
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for imp in &mut importance {
            *imp /= sum;
        }
    }

    importance
}

fn build_simple_tree_for_importance(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_split: usize,
    rng: &mut u64,
    importance: &mut [f64],
) {
    if depth >= max_depth || indices.len() < min_split {
        return;
    }

    let n_features = x.ncols();
    let max_feat = ((n_features as f64).sqrt()).ceil() as usize;

    // Select random features
    let mut features: Vec<usize> = (0..n_features).collect();
    for i in (1..n_features).rev() {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (*rng >> 33) as usize % (i + 1);
        features.swap(i, j);
    }
    features.truncate(max_feat);

    // Find best split
    let mut best_mse = f64::INFINITY;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> = None;

    let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let node_mse: f64 = {
        let mean = total_sum / indices.len() as f64;
        indices.iter().map(|&i| (y[i] - mean).powi(2)).sum()
    };

    for &feature in &features {
        let mut sorted: Vec<(f64, f64, usize)> = indices
            .iter()
            .map(|&i| (x[[i, feature]], y[i], i))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_sum = 0.0;
        for i in 0..sorted.len() - 1 {
            let (xi, yi, _) = sorted[i];
            left_sum += yi;

            let next_x = sorted[i + 1].0;
            if (next_x - xi).abs() < 1e-10 {
                continue;
            }

            let left_n = (i + 1) as f64;
            let right_n = (sorted.len() - i - 1) as f64;

            let left_mean = left_sum / left_n;
            let right_mean = (total_sum - left_sum) / right_n;

            let left_mse: f64 = sorted[..=i]
                .iter()
                .map(|(_, yi, _)| (yi - left_mean).powi(2))
                .sum();
            let right_mse: f64 = sorted[i + 1..]
                .iter()
                .map(|(_, yi, _)| (yi - right_mean).powi(2))
                .sum();

            let total_mse = left_mse + right_mse;
            let improvement = node_mse - total_mse;

            if total_mse < best_mse {
                best_mse = total_mse;
                let threshold = (xi + next_x) / 2.0;
                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
                best_split = Some((feature, threshold, left_idx, right_idx, improvement));
            }
        }
    }

    if let Some((feature, _, left_idx, right_idx, improvement)) = best_split {
        if !left_idx.is_empty() && !right_idx.is_empty() {
            importance[feature] += improvement;
            build_simple_tree_for_importance(
                x,
                y,
                &left_idx,
                depth + 1,
                max_depth,
                min_split,
                rng,
                importance,
            );
            build_simple_tree_for_importance(
                x,
                y,
                &right_idx,
                depth + 1,
                max_depth,
                min_split,
                rng,
                importance,
            );
        }
    }
}

// =============================================================================
// C5.0 Decision Tree
// =============================================================================

/// Configuration for C5.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Config {
    /// Minimum cases in a leaf
    pub min_cases: usize,
    /// Confidence factor for pruning (0-1)
    pub cf: f64,
    /// Use boosting with n trials
    pub trials: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for C50Config {
    fn default() -> Self {
        Self {
            min_cases: 2,
            cf: 0.25,
            trials: 1,
            seed: None,
        }
    }
}

/// Result from C5.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Result {
    /// Predictions (class labels)
    pub predictions: Vec<i64>,
    /// Class probabilities (for each class)
    pub probabilities: Vec<Vec<f64>>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Number of rules/leaves
    pub n_rules: usize,
    /// Class labels
    pub class_labels: Vec<i64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for C50Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "C5.0 Classification")?;
        writeln!(f, "===================")?;
        writeln!(f, "Classes: {:?}", self.class_labels)?;
        writeln!(f, "Rules: {}", self.n_rules)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;
        Ok(())
    }
}

/// C5.0 decision tree for classification.
///
/// Uses information gain ratio for splitting, similar to C4.5 but with
/// additional optimizations and pruning.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target class labels (as integers)
/// * `config` - C5.0 configuration
///
/// # Returns
///
/// C50Result with predictions and probabilities.
///
/// # References
///
/// Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
pub fn c50(x: ArrayView2<f64>, y: &[i64], config: &C50Config) -> EconResult<C50Result> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same length".to_string(),
        ));
    }

    // Get unique classes
    let mut class_labels: Vec<i64> = y.to_vec();
    class_labels.sort();
    class_labels.dedup();

    let class_to_idx: std::collections::HashMap<i64, usize> = class_labels
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let n_classes = class_labels.len();
    let indices: Vec<usize> = (0..n_samples).collect();

    let mut importance = vec![0.0; n_features];
    let mut n_rules = 0;

    // Build tree
    let root = build_c50_node(
        &x,
        y,
        &indices,
        config,
        n_classes,
        &class_to_idx,
        &mut importance,
        &mut n_rules,
    );

    // Generate predictions
    let mut predictions = Vec::with_capacity(n_samples);
    let mut probabilities = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let (pred, probs) = predict_c50(&root, &x.row(i), n_classes, &class_labels);
        predictions.push(pred);
        probabilities.push(probs);
    }

    // Normalize importance
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for imp in &mut importance {
            *imp /= sum;
        }
    }

    Ok(C50Result {
        predictions,
        probabilities,
        feature_importances: importance,
        n_rules,
        class_labels,
        feature_names: None,
    })
}

enum C50Node {
    Split {
        feature: usize,
        threshold: f64,
        info_gain: f64,
        left: Box<C50Node>,
        right: Box<C50Node>,
    },
    Leaf {
        class: i64,
        class_probs: Vec<f64>,
    },
}

fn build_c50_node(
    x: &ArrayView2<f64>,
    y: &[i64],
    indices: &[usize],
    config: &C50Config,
    n_classes: usize,
    class_to_idx: &std::collections::HashMap<i64, usize>,
    importance: &mut [f64],
    n_rules: &mut usize,
) -> C50Node {
    let n = indices.len();

    // Compute class counts
    let mut counts = vec![0usize; n_classes];
    for &i in indices {
        let class_idx = class_to_idx[&y[i]];
        counts[class_idx] += 1;
    }

    let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let majority_class_idx = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Check stopping conditions
    let pure = counts.iter().filter(|&&c| c > 0).count() == 1;
    if pure || n < 2 * config.min_cases {
        *n_rules += 1;
        let class_labels: Vec<i64> = class_to_idx.keys().copied().collect();
        return C50Node::Leaf {
            class: class_labels
                .iter()
                .find(|&&c| class_to_idx[&c] == majority_class_idx)
                .cloned()
                .unwrap_or(0),
            class_probs: probs,
        };
    }

    // Find best split using information gain ratio
    let parent_entropy = entropy(&counts, n);
    let n_features = x.ncols();

    let mut best_gain_ratio = 0.0;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> = None;

    for feature in 0..n_features {
        let mut sorted: Vec<(f64, i64, usize)> = indices
            .iter()
            .map(|&i| (x[[i, feature]], y[i], i))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_counts = vec![0usize; n_classes];
        let mut left_n = 0usize;

        for i in 0..sorted.len() - 1 {
            let (xi, yi, _) = sorted[i];
            let class_idx = class_to_idx[&yi];
            left_counts[class_idx] += 1;
            left_n += 1;

            let next_x = sorted[i + 1].0;
            if (next_x - xi).abs() < 1e-10 {
                continue;
            }

            if left_n < config.min_cases || n - left_n < config.min_cases {
                continue;
            }

            let right_n = n - left_n;
            let right_counts: Vec<usize> =
                (0..n_classes).map(|j| counts[j] - left_counts[j]).collect();

            let left_entropy = entropy(&left_counts, left_n);
            let right_entropy = entropy(&right_counts, right_n);

            let weighted_entropy =
                (left_n as f64 * left_entropy + right_n as f64 * right_entropy) / n as f64;
            let info_gain = parent_entropy - weighted_entropy;

            // Split info for gain ratio
            let p_left = left_n as f64 / n as f64;
            let p_right = right_n as f64 / n as f64;
            let split_info = -p_left * p_left.ln() - p_right * p_right.ln();

            let gain_ratio = if split_info > 1e-10 {
                info_gain / split_info
            } else {
                0.0
            };

            if gain_ratio > best_gain_ratio {
                best_gain_ratio = gain_ratio;
                let threshold = (xi + next_x) / 2.0;
                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
                best_split = Some((feature, threshold, left_idx, right_idx, info_gain));
            }
        }
    }

    if let Some((feature, threshold, left_idx, right_idx, info_gain)) = best_split {
        importance[feature] += info_gain * n as f64;

        let left = build_c50_node(
            x,
            y,
            &left_idx,
            config,
            n_classes,
            class_to_idx,
            importance,
            n_rules,
        );
        let right = build_c50_node(
            x,
            y,
            &right_idx,
            config,
            n_classes,
            class_to_idx,
            importance,
            n_rules,
        );

        C50Node::Split {
            feature,
            threshold,
            info_gain,
            left: Box::new(left),
            right: Box::new(right),
        }
    } else {
        *n_rules += 1;
        let class_labels: Vec<i64> = class_to_idx.keys().copied().collect();
        C50Node::Leaf {
            class: class_labels
                .iter()
                .find(|&&c| class_to_idx[&c] == majority_class_idx)
                .cloned()
                .unwrap_or(0),
            class_probs: probs,
        }
    }
}

fn entropy(counts: &[usize], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n_f;
            -p * p.ln()
        })
        .sum()
}

fn predict_c50(
    node: &C50Node,
    x_row: &ArrayView1<f64>,
    n_classes: usize,
    class_labels: &[i64],
) -> (i64, Vec<f64>) {
    match node {
        C50Node::Leaf { class, class_probs } => (*class, class_probs.clone()),
        C50Node::Split {
            feature,
            threshold,
            left,
            right,
            ..
        } => {
            if x_row[*feature] <= *threshold {
                predict_c50(left, x_row, n_classes, class_labels)
            } else {
                predict_c50(right, x_row, n_classes, class_labels)
            }
        }
    }
}

// =============================================================================
// Cubist (Rule-based Regression)
// =============================================================================

/// Configuration for Cubist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistConfig {
    /// Number of committees (boosting)
    pub committees: usize,
    /// Maximum number of rules
    pub rules: usize,
    /// Use unbiased rules
    pub unbiased: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CubistConfig {
    fn default() -> Self {
        Self {
            committees: 1,
            rules: 100,
            unbiased: false,
            seed: None,
        }
    }
}

/// A rule in Cubist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistRule {
    /// Conditions: (feature, threshold, direction) where direction=true means <=
    pub conditions: Vec<(usize, f64, bool)>,
    /// Linear model coefficients (intercept first)
    pub coefficients: Vec<f64>,
    /// Coverage (number of training instances)
    pub coverage: usize,
}

/// Result from Cubist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistResult {
    /// Rules
    pub rules: Vec<CubistRule>,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for CubistResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cubist Rule-based Regression")?;
        writeln!(f, "============================")?;
        writeln!(f, "Rules: {}", self.rules.len())?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;
        Ok(())
    }
}

/// Cubist rule-based regression.
///
/// Builds a model-tree (tree with linear models at leaves) and extracts rules.
/// Each rule has conditions (path to leaf) and a linear model for prediction.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target values
/// * `config` - Cubist configuration
///
/// # Returns
///
/// CubistResult with rules and predictions.
///
/// # References
///
/// Quinlan, J. R. (1992). "Learning with Continuous Classes." *AI'92*.
pub fn cubist(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &CubistConfig,
) -> EconResult<CubistResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_samples,
            context: "Cubist".to_string(),
        });
    }

    let indices: Vec<usize> = (0..n_samples).collect();
    let mut rules = Vec::new();
    let mut importance = vec![0.0; n_features];

    // Build model tree and extract rules
    build_cubist_tree(
        &x,
        &y,
        &indices,
        Vec::new(),
        config,
        &mut rules,
        &mut importance,
        0,
    );

    // Limit number of rules
    if rules.len() > config.rules {
        rules.truncate(config.rules);
    }

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| predict_cubist(&rules, &x.row(i)))
        .collect();

    // Normalize importance
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for imp in &mut importance {
            *imp /= sum;
        }
    }

    Ok(CubistResult {
        rules,
        predictions,
        feature_importances: importance,
        feature_names: None,
    })
}

fn build_cubist_tree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    conditions: Vec<(usize, f64, bool)>,
    config: &CubistConfig,
    rules: &mut Vec<CubistRule>,
    importance: &mut [f64],
    depth: usize,
) {
    let n = indices.len();
    let n_features = x.ncols();

    // Stopping conditions
    if n < 10 || depth >= 10 || rules.len() >= config.rules {
        // Fit linear model and create rule
        let coefficients = fit_linear_model(x, y, indices);
        rules.push(CubistRule {
            conditions: conditions.clone(),
            coefficients,
            coverage: n,
        });
        return;
    }

    // Find best split
    let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let total_ss: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
    let node_mse = total_ss - total_sum * total_sum / n as f64;

    let mut best_mse = f64::INFINITY;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

    for feature in 0..n_features {
        let mut sorted: Vec<(f64, f64, usize)> = indices
            .iter()
            .map(|&i| (x[[i, feature]], y[i], i))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_sum = 0.0;
        let mut left_ss = 0.0;

        for i in 0..sorted.len() - 1 {
            let (xi, yi, _) = sorted[i];
            left_sum += yi;
            left_ss += yi * yi;

            let next_x = sorted[i + 1].0;
            if (next_x - xi).abs() < 1e-10 {
                continue;
            }

            let left_n = i + 1;
            let right_n = n - left_n;

            if left_n < 5 || right_n < 5 {
                continue;
            }

            let left_mse = left_ss - left_sum * left_sum / left_n as f64;
            let right_ss = total_ss - left_ss;
            let right_sum = total_sum - left_sum;
            let right_mse = right_ss - right_sum * right_sum / right_n as f64;

            let total_mse = left_mse + right_mse;

            if total_mse < best_mse {
                best_mse = total_mse;
                let threshold = (xi + next_x) / 2.0;
                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
                best_split = Some((feature, threshold, left_idx, right_idx));
            }
        }
    }

    // Check if split improves significantly
    let improvement = node_mse - best_mse;
    if improvement < node_mse * 0.01 || best_split.is_none() {
        let coefficients = fit_linear_model(x, y, indices);
        rules.push(CubistRule {
            conditions: conditions.clone(),
            coefficients,
            coverage: n,
        });
        return;
    }

    let (feature, threshold, left_idx, right_idx) = best_split.unwrap();
    importance[feature] += improvement;

    // Build left subtree
    let mut left_conditions = conditions.clone();
    left_conditions.push((feature, threshold, true));
    build_cubist_tree(
        x,
        y,
        &left_idx,
        left_conditions,
        config,
        rules,
        importance,
        depth + 1,
    );

    // Build right subtree
    let mut right_conditions = conditions.clone();
    right_conditions.push((feature, threshold, false));
    build_cubist_tree(
        x,
        y,
        &right_idx,
        right_conditions,
        config,
        rules,
        importance,
        depth + 1,
    );
}

fn fit_linear_model(x: &ArrayView2<f64>, y: &ArrayView1<f64>, indices: &[usize]) -> Vec<f64> {
    let n = indices.len();
    let p = x.ncols();

    if n < p + 1 {
        // Not enough data, just use mean
        let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        let mut coeffs = vec![0.0; p + 1];
        coeffs[0] = mean;
        return coeffs;
    }

    // Simple OLS: (X'X)^-1 X'y
    // Build X with intercept
    let mut x_mat = Array2::zeros((n, p + 1));
    let mut y_vec = Array1::zeros(n);

    for (row, &i) in indices.iter().enumerate() {
        x_mat[[row, 0]] = 1.0;
        for j in 0..p {
            x_mat[[row, j + 1]] = x[[i, j]];
        }
        y_vec[row] = y[i];
    }

    let xtx = x_mat.t().dot(&x_mat);
    let xty = x_mat.t().dot(&y_vec);

    // Add regularization
    let mut xtx_reg = xtx.clone();
    for i in 0..p + 1 {
        xtx_reg[[i, i]] += 1e-6;
    }

    // Solve using Cholesky
    match solve_system(&xtx_reg, &xty) {
        Ok(coeffs) => coeffs,
        Err(_) => {
            let mean = y_vec.sum() / n as f64;
            let mut coeffs = vec![0.0; p + 1];
            coeffs[0] = mean;
            coeffs
        }
    }
}

fn solve_system(a: &Array2<f64>, b: &Array1<f64>) -> EconResult<Vec<f64>> {
    let n = a.nrows();

    // Cholesky decomposition
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag = a[[j, j]] - sum;
                if diag <= 0.0 {
                    return Err(EconError::Computation("Not positive definite".to_string()));
                }
                l[[j, j]] = diag.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Forward substitution: Ly = b
    let mut y_sol = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * y_sol[j];
        }
        y_sol[i] = (b[i] - sum) / l[[i, i]];
    }

    // Back substitution: L'x = y
    let mut x_sol = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x_sol[j];
        }
        x_sol[i] = (y_sol[i] - sum) / l[[i, i]];
    }

    Ok(x_sol)
}

fn predict_cubist(rules: &[CubistRule], x_row: &ArrayView1<f64>) -> f64 {
    // Find matching rules and average their predictions
    let mut total_pred = 0.0;
    let mut total_weight = 0.0;

    for rule in rules {
        // Check if all conditions are satisfied
        let matches = rule.conditions.iter().all(|&(feature, threshold, is_le)| {
            if is_le {
                x_row[feature] <= threshold
            } else {
                x_row[feature] > threshold
            }
        });

        if matches {
            // Apply linear model
            let mut pred = rule.coefficients[0]; // intercept
            for (j, &coef) in rule.coefficients.iter().enumerate().skip(1) {
                if j - 1 < x_row.len() {
                    pred += coef * x_row[j - 1];
                }
            }
            total_pred += pred * rule.coverage as f64;
            total_weight += rule.coverage as f64;
        }
    }

    if total_weight > 0.0 {
        total_pred / total_weight
    } else {
        // No matching rules, use first rule as default
        if !rules.is_empty() {
            let mut pred = rules[0].coefficients[0];
            for (j, &coef) in rules[0].coefficients.iter().enumerate().skip(1) {
                if j - 1 < x_row.len() {
                    pred += coef * x_row[j - 1];
                }
            }
            pred
        } else {
            0.0
        }
    }
}

// =============================================================================
// MARS (Multivariate Adaptive Regression Splines)
// =============================================================================

/// Configuration for MARS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsConfig {
    /// Maximum number of basis functions
    pub max_terms: usize,
    /// Maximum interaction degree
    pub degree: usize,
    /// Penalty for adding terms (GCV penalty)
    pub penalty: f64,
    /// Minimum span between knots
    pub min_span: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for MarsConfig {
    fn default() -> Self {
        Self {
            max_terms: 21,
            degree: 1,
            penalty: 3.0,
            min_span: 0,
            seed: None,
        }
    }
}

/// A basis function in MARS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsBasisFunction {
    /// Feature indices involved
    pub features: Vec<usize>,
    /// Knot values
    pub knots: Vec<f64>,
    /// Directions (+1 for (x - t)+, -1 for (t - x)+)
    pub directions: Vec<i8>,
    /// Coefficient
    pub coefficient: f64,
}

/// Result from MARS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsResult {
    /// Basis functions
    pub basis_functions: Vec<MarsBasisFunction>,
    /// Intercept
    pub intercept: f64,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Generalized cross-validation score
    pub gcv: f64,
    /// R-squared
    pub r_squared: f64,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for MarsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MARS (Multivariate Adaptive Regression Splines)")?;
        writeln!(f, "================================================")?;
        writeln!(f, "Basis functions: {}", self.basis_functions.len())?;
        writeln!(f, "GCV: {:.4}", self.gcv)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;
        Ok(())
    }
}

/// MARS: Multivariate Adaptive Regression Splines.
///
/// Builds a flexible regression model using piecewise linear basis functions.
/// Uses forward selection followed by backward pruning.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target values
/// * `config` - MARS configuration
///
/// # Returns
///
/// MarsResult with basis functions and predictions.
///
/// # References
///
/// Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines."
/// *The Annals of Statistics*, 19(1), 1-67.
pub fn mars(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &MarsConfig) -> EconResult<MarsResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_samples,
            context: "MARS".to_string(),
        });
    }

    // Initialize with intercept only
    let y_mean: f64 = y.sum() / n_samples as f64;

    // Current basis functions (start empty, intercept handled separately)
    let mut basis_functions: Vec<MarsBasisFunction> = Vec::new();

    // Current residuals
    let mut residuals: Array1<f64> = y.to_owned() - y_mean;

    // Forward pass: add basis functions
    for _ in 0..config.max_terms {
        let mut best_reduction = 0.0;
        let mut best_bf: Option<(MarsBasisFunction, MarsBasisFunction)> = None;

        // Try adding a new pair of basis functions
        for feature in 0..n_features {
            // Get sorted unique values as potential knots
            let mut values: Vec<f64> = (0..n_samples).map(|i| x[[i, feature]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            let min_span = config.min_span.max(1);
            for (ki, &knot) in values.iter().enumerate() {
                if ki < min_span || ki >= values.len() - min_span {
                    continue;
                }

                // Create pair of hinge functions: (x - t)+ and (t - x)+
                let bf_plus = MarsBasisFunction {
                    features: vec![feature],
                    knots: vec![knot],
                    directions: vec![1],
                    coefficient: 0.0,
                };

                let bf_minus = MarsBasisFunction {
                    features: vec![feature],
                    knots: vec![knot],
                    directions: vec![-1],
                    coefficient: 0.0,
                };

                // Evaluate reduction in RSS
                let reduction = evaluate_mars_pair(&x, &residuals, &bf_plus, &bf_minus, n_samples);

                if reduction > best_reduction {
                    best_reduction = reduction;
                    best_bf = Some((bf_plus, bf_minus));
                }
            }
        }

        // Check if improvement is sufficient
        if best_bf.is_none() || best_reduction < 1e-8 {
            break;
        }

        let (bf_plus, bf_minus) = best_bf.unwrap();

        // Add both basis functions
        basis_functions.push(bf_plus);
        basis_functions.push(bf_minus);

        // Refit model and update residuals
        let (coefficients, intercept) = fit_mars_model(&x, &y, &basis_functions, n_samples);

        for (i, bf) in basis_functions.iter_mut().enumerate() {
            bf.coefficient = coefficients[i];
        }

        // Update residuals
        for i in 0..n_samples {
            let mut pred = intercept;
            for bf in &basis_functions {
                pred += bf.coefficient * evaluate_basis(&x.row(i), bf);
            }
            residuals[i] = y[i] - pred;
        }
    }

    // Backward pass: prune basis functions using GCV
    let mut current_gcv = compute_gcv(
        &residuals,
        n_samples,
        basis_functions.len() + 1,
        config.penalty,
    );

    loop {
        let mut best_gcv = current_gcv;
        let mut remove_idx: Option<usize> = None;

        for i in 0..basis_functions.len() {
            // Try removing this basis function
            let mut temp_bf = basis_functions.clone();
            temp_bf.remove(i);

            if temp_bf.is_empty() {
                continue;
            }

            let (temp_coeffs, temp_intercept) = fit_mars_model(&x, &y, &temp_bf, n_samples);

            // Compute residuals
            let mut temp_residuals = Array1::zeros(n_samples);
            for j in 0..n_samples {
                let mut pred = temp_intercept;
                for (k, bf) in temp_bf.iter().enumerate() {
                    pred += temp_coeffs[k] * evaluate_basis(&x.row(j), bf);
                }
                temp_residuals[j] = y[j] - pred;
            }

            let gcv = compute_gcv(
                &temp_residuals,
                n_samples,
                temp_bf.len() + 1,
                config.penalty,
            );

            if gcv < best_gcv {
                best_gcv = gcv;
                remove_idx = Some(i);
            }
        }

        if remove_idx.is_none() {
            break;
        }

        basis_functions.remove(remove_idx.unwrap());
        current_gcv = best_gcv;
    }

    // Final fit
    let (coefficients, intercept) = fit_mars_model(&x, &y, &basis_functions, n_samples);

    for (i, bf) in basis_functions.iter_mut().enumerate() {
        bf.coefficient = coefficients[i];
    }

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| {
            let mut pred = intercept;
            for bf in &basis_functions {
                pred += bf.coefficient * evaluate_basis(&x.row(i), bf);
            }
            pred
        })
        .collect();

    // Compute final statistics
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(&pi, &yi)| (yi - pi).powi(2))
        .sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    let final_residuals =
        Array1::from_iter(predictions.iter().zip(y.iter()).map(|(&pi, &yi)| yi - pi));
    let gcv = compute_gcv(
        &final_residuals,
        n_samples,
        basis_functions.len() + 1,
        config.penalty,
    );

    // Compute feature importances
    let mut importance = vec![0.0; n_features];
    for bf in &basis_functions {
        let contrib = bf.coefficient.abs();
        for &feature in &bf.features {
            importance[feature] += contrib;
        }
    }

    // Normalize
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for imp in &mut importance {
            *imp /= sum;
        }
    }

    Ok(MarsResult {
        basis_functions,
        intercept,
        predictions,
        gcv,
        r_squared,
        feature_importances: importance,
        feature_names: None,
    })
}

fn evaluate_basis(x_row: &ArrayView1<f64>, bf: &MarsBasisFunction) -> f64 {
    let mut value = 1.0;
    for ((&feature, &knot), &direction) in bf
        .features
        .iter()
        .zip(bf.knots.iter())
        .zip(bf.directions.iter())
    {
        let diff = if direction == 1 {
            (x_row[feature] - knot).max(0.0)
        } else {
            (knot - x_row[feature]).max(0.0)
        };
        value *= diff;
    }
    value
}

fn evaluate_mars_pair(
    x: &ArrayView2<f64>,
    residuals: &Array1<f64>,
    bf_plus: &MarsBasisFunction,
    bf_minus: &MarsBasisFunction,
    n: usize,
) -> f64 {
    // Compute basis function values
    let mut h_plus = vec![0.0; n];
    let mut h_minus = vec![0.0; n];

    for i in 0..n {
        h_plus[i] = evaluate_basis(&x.row(i), bf_plus);
        h_minus[i] = evaluate_basis(&x.row(i), bf_minus);
    }

    // Fit coefficients using least squares projection
    // Reduction = (r'H)(H'H)^-1(H'r) where H = [h_plus, h_minus]

    let r_hp: f64 = residuals
        .iter()
        .zip(h_plus.iter())
        .map(|(&r, &h)| r * h)
        .sum();
    let r_hm: f64 = residuals
        .iter()
        .zip(h_minus.iter())
        .map(|(&r, &h)| r * h)
        .sum();

    let hp_hp: f64 = h_plus.iter().map(|&h| h * h).sum();
    let hm_hm: f64 = h_minus.iter().map(|&h| h * h).sum();
    let hp_hm: f64 = h_plus
        .iter()
        .zip(h_minus.iter())
        .map(|(&a, &b)| a * b)
        .sum();

    // 2x2 matrix inverse
    let det = hp_hp * hm_hm - hp_hm * hp_hm;
    if det.abs() < 1e-10 {
        return 0.0;
    }

    let inv_00 = hm_hm / det;
    let inv_11 = hp_hp / det;
    let inv_01 = -hp_hm / det;

    let reduction = r_hp * (inv_00 * r_hp + inv_01 * r_hm) + r_hm * (inv_01 * r_hp + inv_11 * r_hm);

    reduction.max(0.0)
}

fn fit_mars_model(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    basis_functions: &[MarsBasisFunction],
    n: usize,
) -> (Vec<f64>, f64) {
    if basis_functions.is_empty() {
        let intercept = y.sum() / n as f64;
        return (Vec::new(), intercept);
    }

    let p = basis_functions.len();

    // Build design matrix
    let mut h = Array2::zeros((n, p + 1));
    for i in 0..n {
        h[[i, 0]] = 1.0; // intercept
        for (j, bf) in basis_functions.iter().enumerate() {
            h[[i, j + 1]] = evaluate_basis(&x.row(i), bf);
        }
    }

    // Solve (H'H)^-1 H'y
    let hth = h.t().dot(&h);
    let hty = h.t().dot(y);

    // Add regularization
    let mut hth_reg = hth.clone();
    for i in 0..p + 1 {
        hth_reg[[i, i]] += 1e-8;
    }

    match solve_system(&hth_reg, &hty) {
        Ok(coeffs) => {
            let intercept = coeffs[0];
            let bf_coeffs = coeffs[1..].to_vec();
            (bf_coeffs, intercept)
        }
        Err(_) => {
            let intercept = y.sum() / n as f64;
            (vec![0.0; p], intercept)
        }
    }
}

fn compute_gcv(residuals: &Array1<f64>, n: usize, n_params: usize, penalty: f64) -> f64 {
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let effective_params = n_params as f64 * penalty;
    let denom = (1.0 - effective_params / n as f64).powi(2);

    if denom > 0.0 {
        rss / (n as f64 * denom)
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantile_rf_basic() {
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

        let config = QuantileRfConfig {
            n_trees: 10,
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed: Some(42),
        };

        let result = quantile_rf(x.view(), y.view(), &config, &[0.25, 0.5, 0.75]).unwrap();

        assert_eq!(result.quantiles.len(), 3);
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.quantile_predictions.len(), 10);
    }

    #[test]
    fn test_ctree_basic() {
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
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0];

        let config = CtreeConfig {
            alpha: 0.05,
            max_depth: 5,
            min_samples_split: 2,
            n_permutations: 100,
            seed: Some(42),
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        assert!(result.n_nodes >= 1);
        assert_eq!(result.predictions.len(), 10);
    }

    #[test]
    fn test_boruta_basic() {
        // Feature 0 is predictive, feature 1 is noise
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
            [10.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = BorutaConfig {
            max_runs: 20,
            p_value: 0.01,
            n_trees: 50,
            seed: Some(42),
        };

        let result = boruta(x.view(), y.view(), &config, None).unwrap();

        assert_eq!(result.feature_names.len(), 2);
        assert_eq!(result.decisions.len(), 2);
    }

    #[test]
    fn test_c50_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0]
        ];
        let y = vec![0i64, 0, 0, 1, 1, 1];

        let config = C50Config::default();
        let result = c50(x.view(), &y, &config).unwrap();

        assert_eq!(result.predictions.len(), 6);
        assert_eq!(result.class_labels.len(), 2);
    }

    #[test]
    fn test_cubist_basic() {
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

        let config = CubistConfig::default();
        let result = cubist(x.view(), y.view(), &config).unwrap();

        assert!(!result.rules.is_empty());
        assert_eq!(result.predictions.len(), 10);
    }

    #[test]
    fn test_mars_basic() {
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
        // Piecewise linear: y = x for x <= 5, y = 5 + 2*(x-5) for x > 5
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

        let config = MarsConfig {
            max_terms: 10,
            degree: 1,
            penalty: 2.0,
            min_span: 1,
            seed: Some(42),
        };

        let result = mars(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.predictions.len(), 10);
        assert!(result.r_squared > 0.8);
    }
}
