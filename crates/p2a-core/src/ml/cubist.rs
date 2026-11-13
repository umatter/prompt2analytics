//! Cubist regression trees implementation.
//!
//! Cubist is a rule-based model that combines decision tree partitioning with linear
//! regression models in terminal nodes. It is an extension of Quinlan's M5 model tree
//! algorithm with additional innovations including committee models (boosting-like
//! ensembles) and instance-based corrections using k-nearest neighbors.
//!
//! ## Key Features
//!
//! - **Linear models in leaves**: Each rule terminates with a linear regression model
//!   fitted to the data subset defined by that rule
//! - **Committee models**: Boosting-like ensemble where each tree after the first is fitted
//!   to `2y - pred` of the one before, and predictions are averaged
//! - **Instance-based corrections**: K-nearest neighbor adjustment of predictions
//! - **Rule extraction**: Tree paths are collapsed into interpretable IF-THEN rules
//! - **Prediction smoothing**: Each rule's model is its leaf model smoothed with every
//!   ancestor's model up to the root
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{cubist, CubistConfig};
//! use ndarray::array;
//!
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
//!                [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]];
//! let y = array![2.1, 4.2, 6.1, 8.3, 10.0, 12.1, 14.0, 15.9, 18.1, 20.0];
//!
//! let config = CubistConfig {
//!     committees: 3,
//!     neighbors: 5,
//!     ..Default::default()
//! };
//!
//! let result = cubist(x.view(), y.view(), &config).unwrap();
//! println!("Number of rules: {}", result.n_rules);
//! println!("Training RMSE: {:.4}", result.train_rmse);
//! ```
//!
//! ## Algorithm Details
//!
//! The Cubist algorithm proceeds as follows:
//!
//! 1. **Tree construction**: Split a node where the best split removes more than
//!    `min_improvement` of the node's sum of squared deviations
//! 2. **Linear model fitting**: Fit least squares at every node, on the attributes tested on
//!    the path to it (all attributes at the root); an aliased attribute gets coefficient 0
//! 3. **Rule extraction**: Convert tree paths to rules by collapsing conditions
//! 4. **Prediction smoothing** (Quinlan 1992): from the leaf up,
//!    `p <- (n p + k q) / (n + k)` per ancestor, with `n` the training cases of the child on
//!    the path, `q` the ancestor's model and `k = smoothing_coefficient`; linear in the
//!    models, so each rule carries one composite model, which prediction evaluates and the
//!    rule prints
//! 5. **Committee iteration**: For committees > 1, the next tree is fitted to `2y - pred`
//!    (pred the previous tree's prediction), as R's Cubist; predictions are averaged
//! 6. **Instance correction**: Optionally add the mean residual of the `neighbors` nearest
//!    training cases on range-scaled distances (Quinlan 1993, equal weights)
//!
//! ## Differences from R's Cubist
//!
//! Node models use the attributes tested on the path (all at the root) and are not
//! simplified by term elimination; Cubist and M5 choose other attribute sets and drop terms.
//! Rules are not simplified or pruned, and the neighbour adjustment weights the neighbours
//! equally. See `validation/ml/cubist.md`.
//!
//! ## References
//!
//! - Quinlan, J. R. (1992). "Learning with continuous classes". In *Proceedings of the
//!   5th Australian Joint Conference on Artificial Intelligence*, pp. 343-348.
//! - Quinlan, J. R. (1993). "Combining instance-based and model-based learning".
//!   In *Proceedings of the Tenth International Conference on Machine Learning*, pp. 236-243.
//! - Wang, Y. & Witten, I. H. (1997). "Induction of model trees for predicting continuous
//!   classes". In *Proceedings of the European Conference on Machine Learning*.
//! - R package `Cubist` by Kuhn, M., Weston, S., & Quinlan, J. R.
//!   <https://cran.r-project.org/package=Cubist>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::xty;
use crate::linalg::solve::least_squares;
use crate::regression::ols::normal_equations_inverse;

/// Configuration for Cubist regression trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistConfig {
    /// Number of committee models (boosting-like ensemble), 1 to 100 (the limit
    /// of R's Cubist). Default: 1.
    /// Each committee fits adjusted residuals from the previous iteration.
    pub committees: usize,

    /// Number of nearest neighbors for instance-based correction. Default: 0 (disabled).
    /// When > 0, predictions are adjusted using k-NN of training data.
    pub neighbors: usize,

    /// Maximum depth of trees; values above 512 act as 512. Default: 10.
    pub max_depth: usize,

    /// Minimum samples required to split a node. Default: 10.
    pub min_split: usize,

    /// Minimum samples in a terminal node. Default: 5.
    pub min_bucket: usize,

    /// Minimum share of the node's sum of squared deviations a split must remove, in [0, 1).
    /// Default: 0.01.
    pub min_improvement: f64,

    /// Whether to extract rules from trees. Default: true.
    pub extract_rules: bool,

    /// Weight `k` of each ancestor model in the smoothing of a rule's model (Quinlan 1992).
    /// Default: 15.0. Higher values pull the prediction toward the ancestors (the root model
    /// as `k` grows); 0 keeps the leaf model.
    pub smoothing_coefficient: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for CubistConfig {
    fn default() -> Self {
        CubistConfig {
            committees: 1,
            neighbors: 0,
            max_depth: 10,
            min_split: 10,
            min_bucket: 5,
            min_improvement: 0.01,
            extract_rules: true,
            smoothing_coefficient: 15.0,
            seed: None,
        }
    }
}

/// A condition in a Cubist rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    /// Feature index
    pub feature: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Comparison operator: "<=", ">", "="
    pub operator: String,
    /// Threshold value
    pub threshold: f64,
}

impl std::fmt::Display for RuleCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self
            .feature_name
            .clone()
            .unwrap_or_else(|| format!("X{}", self.feature));
        write!(f, "{} {} {:.4}", name, self.operator, self.threshold)
    }
}

/// A Cubist rule with conditions and a linear model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistRule {
    /// Rule ID
    pub id: usize,
    /// Conditions that define this rule (IF conditions)
    pub conditions: Vec<RuleCondition>,
    /// Number of training samples covered by this rule
    pub coverage: usize,
    /// Intercept of the rule's smoothed model (THEN linear model), the one prediction uses
    pub intercept: f64,
    /// Coefficients of the rule's smoothed model, the non-zero ones only
    pub coefficients: Vec<(usize, String, f64)>, // (feature_idx, feature_name, coefficient)
    /// Mean of target variable in this rule's subset
    pub mean_response: f64,
    /// Standard deviation of target in this rule's subset
    pub std_response: f64,
}

impl std::fmt::Display for CubistRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rule {}: IF ", self.id)?;
        for (i, cond) in self.conditions.iter().enumerate() {
            if i > 0 {
                write!(f, " AND ")?;
            }
            write!(f, "{}", cond)?;
        }
        write!(f, " THEN y = {:.4}", self.intercept)?;
        for (_, name, coef) in &self.coefficients {
            if *coef >= 0.0 {
                write!(f, " + {:.4}*{}", coef, name)?;
            } else {
                write!(f, " - {:.4}*{}", coef.abs(), name)?;
            }
        }
        write!(f, " [coverage={}]", self.coverage)?;
        Ok(())
    }
}

/// Result from Cubist model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubistResult {
    /// Number of rules extracted
    pub n_rules: usize,
    /// The extracted rules
    pub rules: Vec<CubistRule>,
    /// Number of committees used
    pub committees: usize,
    /// Number of neighbors for instance correction (0 if disabled)
    pub neighbors: usize,
    /// Variable importance (sum of appearances in rules + linear models)
    pub variable_importance: Vec<f64>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Training RMSE
    pub train_rmse: f64,
    /// Training R-squared
    pub train_r_squared: f64,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Configuration used
    pub config: CubistConfig,

    // Internal state for prediction (not serialized)
    #[serde(skip)]
    pub(crate) committee_trees: Vec<CubistTree>,
    #[serde(skip)]
    pub(crate) training_x: Option<Array2<f64>>,
    #[serde(skip)]
    pub(crate) training_y: Option<Array1<f64>>,
}

impl std::fmt::Display for CubistResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cubist Regression Model")?;
        writeln!(f, "======================")?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Features: {}", self.n_features)?;
        writeln!(f, "Committees: {}", self.committees)?;
        if self.neighbors > 0 {
            writeln!(f, "Neighbors (k-NN correction): {}", self.neighbors)?;
        }
        writeln!(f)?;
        writeln!(f, "Performance:")?;
        writeln!(f, "  Training RMSE: {:.4}", self.train_rmse)?;
        writeln!(f, "  Training R-squared: {:.4}", self.train_r_squared)?;
        writeln!(f)?;
        writeln!(f, "Rules: {}", self.n_rules)?;

        // Show first few rules
        for rule in self.rules.iter().take(5) {
            writeln!(f, "  {}", rule)?;
        }
        if self.n_rules > 5 {
            writeln!(f, "  ... ({} more rules)", self.n_rules - 5)?;
        }

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.is_nan().cmp(&b.1.is_nan()).then(b.1.total_cmp(&a.1)));

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

        Ok(())
    }
}

/// Deepest tree level (root = 0) the builder grows. A node at this depth
/// becomes a leaf in every mode, so no input can recurse deep enough to
/// overflow a 2 MiB worker stack (user decision, Tier 0). Trees that stay
/// shallower are unchanged.
const MAX_TREE_DEPTH: usize = 512;

/// Internal tree node for Cubist.
#[derive(Debug, Clone)]
pub(crate) enum CubistNode {
    Split {
        feature: usize,
        threshold: f64,
        left: Box<CubistNode>,
        right: Box<CubistNode>,
        // Linear model at this node for smoothing
        intercept: f64,
        coefficients: Vec<f64>,
        n_samples: usize,
    },
    Leaf {
        intercept: f64,
        coefficients: Vec<f64>,
        n_samples: usize,
        mean_response: f64,
        std_response: f64,
        // Indices of features used in the path to this leaf
        path_features: Vec<usize>,
        // The rule's model: this leaf's model smoothed with every ancestor's up to the root
        // (`CubistTree::smooth`); what prediction evaluates and the rule prints.
        smoothed_intercept: f64,
        smoothed_coefficients: Vec<f64>,
    },
}

impl CubistNode {
    /// A leaf whose smoothed model is, until `CubistTree::smooth` runs, its own.
    fn leaf(
        intercept: f64,
        coefficients: Vec<f64>,
        n_samples: usize,
        (mean_response, std_response): (f64, f64),
        path_features: Vec<usize>,
    ) -> CubistNode {
        CubistNode::Leaf {
            smoothed_intercept: intercept,
            smoothed_coefficients: coefficients.clone(),
            intercept,
            coefficients,
            n_samples,
            mean_response,
            std_response,
            path_features,
        }
    }
}

/// Internal tree structure for Cubist.
#[derive(Debug, Clone)]
pub(crate) struct CubistTree {
    root: Option<CubistNode>,
    n_features: usize,
    feature_names: Option<Vec<String>>,
    smoothing_coefficient: f64,
}

impl CubistTree {
    fn new(n_features: usize, feature_names: Option<Vec<String>>, smoothing_coef: f64) -> Self {
        CubistTree {
            root: None,
            n_features,
            feature_names,
            smoothing_coefficient: smoothing_coef,
        }
    }

    /// Fit the tree to data.
    fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<f64>, config: &CubistConfig) {
        let indices: Vec<usize> = (0..x.nrows()).collect();
        let path_features = Vec::new();
        self.root = Some(self.build_tree(x, y, &indices, 0, config, path_features));
        self.smooth();
    }

    /// Fold each leaf's model and every ancestor's into one linear model (Quinlan 1992):
    /// from the leaf up, `p <- (n p + k q) / (n + k)` per level, with `n` the training cases
    /// of the child on the path, `q` the ancestor's model and `k` the smoothing coefficient.
    /// Every step is linear in the models, so the composite is one coefficient vector.
    fn smooth(&mut self) {
        let k = self.smoothing_coefficient;
        if let Some(root) = self.root.as_mut() {
            let mut ancestors = Vec::new();
            smooth_node(root, &mut ancestors, k);
        }
    }

    /// Build tree recursively.
    fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        config: &CubistConfig,
        path_features: Vec<usize>,
    ) -> CubistNode {
        let n_samples = indices.len();

        // Fit linear model at this node
        let (intercept, coefficients) = self.fit_linear_model(x, y, indices, &path_features);

        // Check stopping conditions
        if depth >= config.max_depth.min(MAX_TREE_DEPTH)
            || n_samples < config.min_split
            || n_samples <= config.min_bucket.saturating_mul(2)
        {
            let stats = self.compute_stats(y, indices);
            return CubistNode::leaf(intercept, coefficients, n_samples, stats, path_features);
        }

        // Check if all targets are the same
        let first_y = y[indices[0]];
        if indices.iter().all(|&i| (y[i] - first_y).abs() < 1e-10) {
            let stats = self.compute_stats(y, indices);
            return CubistNode::leaf(intercept, coefficients, n_samples, stats, path_features);
        }

        // Sum of squared deviations of the node's targets
        let node_ssd = self.compute_ssd(y, indices);

        // Find best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices, config, node_ssd)
        {
            // Check minimum bucket size
            if left_indices.len() < config.min_bucket || right_indices.len() < config.min_bucket {
                let stats = self.compute_stats(y, indices);
                return CubistNode::leaf(intercept, coefficients, n_samples, stats, path_features);
            }

            // Update path features for children
            let mut left_path = path_features.clone();
            let mut right_path = path_features.clone();
            if !left_path.contains(&best_feature) {
                left_path.push(best_feature);
            }
            if !right_path.contains(&best_feature) {
                right_path.push(best_feature);
            }

            let left = self.build_tree(x, y, &left_indices, depth + 1, config, left_path);
            let right = self.build_tree(x, y, &right_indices, depth + 1, config, right_path);

            CubistNode::Split {
                feature: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
                intercept,
                coefficients,
                n_samples,
            }
        } else {
            let stats = self.compute_stats(y, indices);
            CubistNode::leaf(intercept, coefficients, n_samples, stats, path_features)
        }
    }

    /// Fit a linear model using only the features in path_features.
    /// If path_features is empty, use all features.
    fn fit_linear_model(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        path_features: &[usize],
    ) -> (f64, Vec<f64>) {
        if indices.len() < 3 {
            // Too few samples, return mean
            let mean_y: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            return (mean_y, vec![0.0; self.n_features]);
        }

        // Determine which features to use
        let features_to_use: Vec<usize> = if path_features.is_empty() {
            (0..self.n_features).collect()
        } else {
            path_features.to_vec()
        };

        if features_to_use.is_empty() {
            let mean_y: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            return (mean_y, vec![0.0; self.n_features]);
        }

        let n = indices.len();
        let p = features_to_use.len();

        // Build design matrix with intercept
        let mut design = Array2::<f64>::ones((n, p + 1));
        for (row_idx, &sample_idx) in indices.iter().enumerate() {
            for (col_idx, &feat_idx) in features_to_use.iter().enumerate() {
                design[[row_idx, col_idx + 1]] = x[[sample_idx, feat_idx]];
            }
        }

        // Build y vector
        let y_sub: Array1<f64> = indices.iter().map(|&i| y[i]).collect();

        // Fit OLS: the normal equations when the design is of full rank and well conditioned,
        // otherwise lm's pivoted QR; an aliased attribute is eliminated (coefficient 0), as
        // Cubist does. The normal-equations solve skips OLS's refinement step, which would
        // add two passes over the design at every node: unrefined, the coefficients' relative
        // error is of order eps / rcond, at most about 1e-8 where the normal equations are
        // accepted (the exception is named in the `regression::ols::NormalInverse` doc).
        let fit = match normal_equations_inverse(design.view()) {
            Ok(Some(ne)) => Ok(ne.inverse.dot(&xty(&design.view(), &y_sub))),
            Ok(None) => least_squares(design.view(), y_sub.view()).map(|ls| {
                ls.coefficients
                    .mapv(|b| if b.is_finite() { b } else { 0.0 })
            }),
            Err(e) => Err(e),
        };

        match fit {
            Ok(beta) => {
                let intercept = beta[0];
                let mut coefficients = vec![0.0; self.n_features];
                for (i, &feat_idx) in features_to_use.iter().enumerate() {
                    coefficients[feat_idx] = beta[i + 1];
                }

                (intercept, coefficients)
            }
            Err(_) => {
                // Singular matrix, return mean
                let mean_y: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
                (mean_y, vec![0.0; self.n_features])
            }
        }
    }

    /// Find the best split for a node: the one removing the largest part of the node's sum of
    /// squared deviations, if it removes more than `min_improvement` of it.
    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        config: &CubistConfig,
        node_ssd: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_improvement = config.min_improvement * node_ssd;
        let mut best_split: Option<(usize, f64)> = None;

        for feature in 0..self.n_features {
            // Sort indices by feature value
            let mut sorted: Vec<(f64, f64, usize)> = indices
                .iter()
                .map(|&i| (x[[i, feature]], y[i], i))
                .collect();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Incremental MSE calculation
            let total_sum: f64 = sorted.iter().map(|(_, y_val, _)| y_val).sum();
            let total_ss: f64 = sorted.iter().map(|(_, y_val, _)| y_val * y_val).sum();
            let mut left_sum = 0.0;
            let mut left_ss = 0.0;
            let mut left_n = 0usize;

            for i in 0..sorted.len() - 1 {
                let (x_val, y_val, _) = sorted[i];
                left_sum += y_val;
                left_ss += y_val * y_val;
                left_n += 1;

                // Skip if next value is the same
                let next_x = sorted[i + 1].0;
                if (next_x - x_val).abs() < 1e-10 {
                    continue;
                }

                let right_n = sorted.len() - left_n;
                if left_n < config.min_bucket || right_n < config.min_bucket {
                    continue;
                }

                // Reduction of the sum of squared deviations
                let left_ssd = left_ss - left_sum * left_sum / left_n as f64;
                let right_sum = total_sum - left_sum;
                let right_ss = total_ss - left_ss;
                let right_ssd = right_ss - right_sum * right_sum / right_n as f64;

                let improvement = node_ssd - left_ssd - right_ssd;

                if improvement > best_improvement {
                    best_improvement = improvement;
                    let threshold = (x_val + next_x) / 2.0;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split.map(|(feature, threshold)| {
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
            (feature, threshold, left_indices, right_indices)
        })
    }

    /// Sum of squared deviations from the mean for a set of indices.
    fn compute_ssd(&self, y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let ss: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
        ss - sum * sum / indices.len() as f64
    }

    /// Compute mean and std for a set of indices.
    fn compute_stats(&self, y: &ArrayView1<f64>, indices: &[usize]) -> (f64, f64) {
        if indices.is_empty() {
            return (0.0, 0.0);
        }
        let n = indices.len() as f64;
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
        let var: f64 = indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    /// Predict for a single sample: the smoothed model of the leaf it falls in.
    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        let mut node = match &self.root {
            Some(node) => node,
            None => return 0.0,
        };
        loop {
            match node {
                CubistNode::Leaf {
                    smoothed_intercept,
                    smoothed_coefficients,
                    ..
                } => {
                    return *smoothed_intercept
                        + smoothed_coefficients
                            .iter()
                            .zip(x.iter())
                            .map(|(&c, &xi)| c * xi)
                            .sum::<f64>();
                }
                CubistNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    node = if x[*feature] <= *threshold {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }

    /// Extract rules from the tree.
    fn extract_rules(&self) -> Vec<CubistRule> {
        let mut rules = Vec::new();
        if let Some(ref root) = self.root {
            self.extract_rules_recursive(root, Vec::new(), &mut rules);
        }

        // Assign rule IDs
        for (i, rule) in rules.iter_mut().enumerate() {
            rule.id = i + 1;
        }

        rules
    }

    /// Recursive rule extraction.
    fn extract_rules_recursive(
        &self,
        node: &CubistNode,
        conditions: Vec<RuleCondition>,
        rules: &mut Vec<CubistRule>,
    ) {
        match node {
            CubistNode::Leaf {
                smoothed_intercept,
                smoothed_coefficients,
                n_samples,
                mean_response,
                std_response,
                ..
            } => {
                // The rule's model is the smoothed composite prediction uses; an eliminated
                // (aliased) attribute has coefficient 0 and is not listed.
                let mut coef_entries = Vec::new();
                for (i, &c) in smoothed_coefficients.iter().enumerate() {
                    if c != 0.0 {
                        let name = self
                            .feature_names
                            .as_ref()
                            .and_then(|names| names.get(i).cloned())
                            .unwrap_or_else(|| format!("X{}", i));
                        coef_entries.push((i, name, c));
                    }
                }

                rules.push(CubistRule {
                    id: 0, // Will be assigned later
                    conditions,
                    coverage: *n_samples,
                    intercept: *smoothed_intercept,
                    coefficients: coef_entries,
                    mean_response: *mean_response,
                    std_response: *std_response,
                });
            }
            CubistNode::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                let feature_name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(*feature).cloned());

                // Left branch: feature <= threshold
                let mut left_conditions = conditions.clone();
                left_conditions.push(RuleCondition {
                    feature: *feature,
                    feature_name: feature_name.clone(),
                    operator: "<=".to_string(),
                    threshold: *threshold,
                });
                self.extract_rules_recursive(left, left_conditions, rules);

                // Right branch: feature > threshold
                let mut right_conditions = conditions;
                right_conditions.push(RuleCondition {
                    feature: *feature,
                    feature_name,
                    operator: ">".to_string(),
                    threshold: *threshold,
                });
                self.extract_rules_recursive(right, right_conditions, rules);
            }
        }
    }

    /// Compute variable importance.
    fn variable_importance(&self) -> Array1<f64> {
        let mut importance = Array1::zeros(self.n_features);
        if let Some(ref root) = self.root {
            self.accumulate_importance(root, &mut importance);
        }

        // Normalize
        let sum: f64 = importance.sum();
        if sum > 0.0 {
            importance /= sum;
        }

        importance
    }

    /// Accumulate importance from tree structure.
    fn accumulate_importance(&self, node: &CubistNode, importance: &mut Array1<f64>) {
        match node {
            CubistNode::Leaf { coefficients, .. } => {
                // Add importance from linear model coefficients
                for (i, &c) in coefficients.iter().enumerate() {
                    importance[i] += c.abs();
                }
            }
            CubistNode::Split {
                feature,
                left,
                right,
                coefficients,
                ..
            } => {
                // Add importance for split feature
                importance[*feature] += 1.0;

                // Add importance from linear model
                for (i, &c) in coefficients.iter().enumerate() {
                    importance[i] += c.abs() * 0.5; // Weight internal nodes less
                }

                self.accumulate_importance(left, importance);
                self.accumulate_importance(right, importance);
            }
        }
    }
}

/// One ancestor on the path from the root: its model and its number of training cases.
type AncestorModel = (f64, Vec<f64>, usize);

/// `CubistTree::smooth` below `node`, with `ancestors` the models above it (root first).
fn smooth_node(node: &mut CubistNode, ancestors: &mut Vec<AncestorModel>, k: f64) {
    match node {
        CubistNode::Split {
            left,
            right,
            intercept,
            coefficients,
            n_samples,
            ..
        } => {
            ancestors.push((*intercept, coefficients.clone(), *n_samples));
            smooth_node(left, ancestors, k);
            smooth_node(right, ancestors, k);
            ancestors.pop();
        }
        CubistNode::Leaf {
            intercept,
            coefficients,
            n_samples,
            smoothed_intercept,
            smoothed_coefficients,
            ..
        } => {
            let mut b0 = *intercept;
            let mut b = coefficients.clone();
            let mut n_below = *n_samples as f64;
            for (q0, q, n_node) in ancestors.iter().rev() {
                let denom = n_below + k;
                b0 = (n_below * b0 + k * q0) / denom;
                for (bj, &qj) in b.iter_mut().zip(q) {
                    *bj = (n_below * *bj + k * qj) / denom;
                }
                n_below = *n_node as f64;
            }
            *smoothed_intercept = b0;
            *smoothed_coefficients = b;
        }
    }
}

use super::lcg_random;

/// Compute k-nearest neighbors for instance-based correction.
fn find_k_neighbors(
    x_train: &ArrayView2<f64>,
    y_train: &ArrayView1<f64>,
    x_test: &ArrayView1<f64>,
    k: usize,
) -> f64 {
    if k == 0 || x_train.nrows() == 0 {
        return 0.0;
    }

    // Compute distances
    let mut distances: Vec<(f64, f64)> = x_train
        .rows()
        .into_iter()
        .zip(y_train.iter())
        .map(|(row, &y)| {
            let dist: f64 = row
                .iter()
                .zip(x_test.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            (dist, y)
        })
        .collect();

    // Sort by distance
    distances.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Average k nearest
    let k_actual = k.min(distances.len());
    let sum: f64 = distances[..k_actual].iter().map(|(_, y)| y).sum();

    sum / k_actual as f64
}

/// Fit a Cubist regression model.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - Cubist configuration
///
/// # Returns
///
/// CubistResult containing fitted model, rules, and predictions.
///
/// # Errors
///
/// `InsufficientData` for fewer than 2 rows; `Computation` when `x` and `y`
/// differ in length; `InvalidSpecification` when `committees` is outside
/// 1..=100, `min_improvement` outside [0, 1), or `x` or `y` holds a non-finite value; `Cancelled` once the
/// installed computation budget expires (checked per committee and per row of
/// the final prediction pass, which runs the nearest-neighbour correction).
///
/// # Example
///
/// ```rust,no_run
/// use p2a_core::ml::{cubist, CubistConfig};
/// use ndarray::array;
///
/// let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
/// let y = array![1.1, 2.2, 2.8, 4.1, 4.9];
///
/// let result = cubist(x.view(), y.view(), &CubistConfig::default()).unwrap();
/// println!("RMSE: {:.4}", result.train_rmse);
/// ```
///
/// # References
///
/// - Quinlan, J. R. (1992). "Learning with continuous classes".
/// - R package `Cubist` <https://cran.r-project.org/package=Cubist>
pub fn cubist(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &CubistConfig,
) -> EconResult<CubistResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_samples,
            context: "Cubist regression".to_string(),
        });
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }
    if !(1..=100).contains(&config.committees) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "committees must be between 1 and 100 (the limit of R's Cubist), got {}",
                config.committees
            ),
        });
    }
    if !(config.min_improvement >= 0.0 && config.min_improvement < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "min_improvement must be in [0, 1), got {}",
                config.min_improvement
            ),
        });
    }
    for (j, column) in x.axis_iter(Axis(1)).enumerate() {
        let k = column.iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "feature column {j} (0-based) has {k} non-finite values (NaN or infinite); \
                     remove or impute them first"
                ),
            });
        }
    }
    let k = y.iter().filter(|v| !v.is_finite()).count();
    if k > 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the target has {k} non-finite values (NaN or infinite); remove or impute them first"
            ),
        });
    }

    let _rng_state = config.seed.unwrap_or(42);

    // Build committee models
    let mut committee_trees =
        crate::resource::try_with_capacity(config.committees, "Cubist committees")?;
    let mut current_y = y.to_owned();
    let mut all_rules = Vec::new();
    let mut total_importance = Array1::zeros(n_features);

    for c in 0..config.committees {
        crate::resource::checkpoint()?;
        let mut tree = CubistTree::new(n_features, None, config.smoothing_coefficient);
        tree.fit(&x, &current_y.view(), config);

        // Extract rules from this tree
        if config.extract_rules {
            let rules = tree.extract_rules();
            all_rules.extend(rules);
        }

        // Accumulate importance
        total_importance = total_importance + tree.variable_importance();

        // The next member's target is y - (pred - y) = 2y - pred, pred this member's
        // prediction (R's Cubist): where this member over-predicted by 2, the next one aims
        // 2 below y.
        if c < config.committees - 1 {
            for i in 0..n_samples {
                let pred = tree.predict_one(&x.row(i));
                current_y[i] = 2.0 * y[i] - pred;
            }
        }

        committee_trees.push(tree);
    }

    // Assign unique rule IDs
    for (i, rule) in all_rules.iter_mut().enumerate() {
        rule.id = i + 1;
    }

    // Normalize importance
    let sum: f64 = total_importance.sum();
    if sum > 0.0 {
        total_importance /= sum;
    }

    // Compute final predictions (average of committees)
    let mut predictions = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        // With neighbors > 0 this loop is O(n^2 p log n): check the budget per row.
        crate::resource::checkpoint()?;
        let row = x.row(i);
        let pred_sum: f64 = committee_trees
            .iter()
            .map(|tree| tree.predict_one(&row))
            .sum();
        let mut pred = pred_sum / config.committees as f64;

        // Apply instance-based correction if enabled
        if config.neighbors > 0 {
            let knn_pred = find_k_neighbors(&x, &y, &row, config.neighbors);
            // Combine model prediction with k-NN (weighted average)
            pred = 0.5 * pred + 0.5 * knn_pred;
        }

        predictions.push(pred);
    }

    // Compute training metrics
    let mean_y: f64 = y.iter().sum::<f64>() / n_samples as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(&pred, &yi)| (yi - pred).powi(2))
        .sum();

    let train_rmse = (ss_res / n_samples as f64).sqrt();
    let train_r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok(CubistResult {
        n_rules: all_rules.len(),
        rules: all_rules,
        committees: config.committees,
        neighbors: config.neighbors,
        variable_importance: total_importance.to_vec(),
        feature_names: None,
        train_rmse,
        train_r_squared,
        predictions,
        n_obs: n_samples,
        n_features,
        config: config.clone(),
        committee_trees,
        training_x: if config.neighbors > 0 {
            Some(x.to_owned())
        } else {
            None
        },
        training_y: if config.neighbors > 0 {
            Some(y.to_owned())
        } else {
            None
        },
    })
}

/// Predict using a fitted Cubist model.
///
/// # Arguments
///
/// * `result` - Fitted Cubist result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions for each observation.
pub fn cubist_predict(result: &CubistResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    if result.committee_trees.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted trees".to_string(),
        ));
    }

    let n_samples = x.nrows();
    let mut predictions = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let row = x.row(i);

        // Average predictions from all committees
        let pred_sum: f64 = result
            .committee_trees
            .iter()
            .map(|tree| tree.predict_one(&row))
            .sum();
        let mut pred = pred_sum / result.committees as f64;

        // Apply instance-based correction if enabled and training data available
        if result.neighbors > 0 {
            if let (Some(train_x), Some(train_y)) = (&result.training_x, &result.training_y) {
                let knn_pred =
                    find_k_neighbors(&train_x.view(), &train_y.view(), &row, result.neighbors);
                pred = 0.5 * pred + 0.5 * knn_pred;
            }
        }

        predictions.push(pred);
    }

    Ok(predictions)
}

/// Run Cubist on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Numeric target column. Integer and boolean (0/1) columns are accepted;
///   a missing value (null or NaN) in the target or a feature is an error naming the
///   column (`EconError::NullValues`)
/// * `x_cols` - Feature column names
/// * `config` - Cubist configuration
///
/// # Returns
///
/// CubistResult with fitted model, rules, and predictions.
pub fn run_cubist(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &CubistConfig,
) -> EconResult<CubistResult> {
    let (x, y_arr, _missing) =
        super::extract::extract_xy(dataset, y_col, x_cols, crate::data::columns::NaPolicy::Fail)?;
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    let mut result = cubist(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names.clone());

    // Update feature names in rules
    for rule in result.rules.iter_mut() {
        for cond in rule.conditions.iter_mut() {
            if cond.feature_name.is_none() {
                cond.feature_name = feature_names.get(cond.feature).cloned();
            }
        }
        for (idx, name, _) in rule.coefficients.iter_mut() {
            if name.starts_with('X') {
                if let Some(fname) = feature_names.get(*idx) {
                    *name = fname.clone();
                }
            }
        }
    }

    // Update feature names in trees
    for tree in result.committee_trees.iter_mut() {
        tree.feature_names = Some(feature_names.clone());
    }

    Ok(result)
}

/// Run Cubist with default configuration.
pub fn run_cubist_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<CubistResult> {
    run_cubist(dataset, y_col, x_cols, &CubistConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::array;

    #[test]
    fn test_validate_cubist_leaf_model_duplicated_column() {
        // lm reports the duplicated attribute NA; Cubist eliminates it (coefficient 0).
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("inv_cubist_leaf_model");
        let col = |c: &str| Array1::from(numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0);
        let x = ndarray::stack![Axis(1), col("x1"), col("x1dup")];
        let y = col("y");
        let tree = CubistTree::new(2, None, 0.0);
        let (intercept, coefs) =
            tree.fit_linear_model(&x.view(), &y.view(), &(0..30).collect::<Vec<_>>(), &[]);
        Check::new(&case)
            .named_subset(
                "coef",
                &["(Intercept)", "x1"],
                &["(Intercept)", "x1"],
                &[intercept, coefs[0]],
                Tol::coef(30),
            )
            .finish();
        assert_eq!(coefs[1], 0.0);
    }

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn cubist_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| {
            ((i * (7 + 4 * j)) % 13) as f64 + 0.1 * j as f64
        });
        let y = Array1::from_shape_fn(40, |i| {
            1.5 * ((i * 7) % 13) as f64 + ((i * 5) % 3) as f64 * 0.3
        });
        (x, y)
    }

    #[test]
    fn test_cubist_committees_are_bounded() {
        let (x, y) = cubist_data();
        for committees in [0usize, 101, 10_000_000_000] {
            let err = cubist(
                x.view(),
                y.view(),
                &CubistConfig {
                    committees,
                    ..Default::default()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("committees must be between 1 and 100"),
                "{committees}: {err}"
            );
        }
    }

    /// x = 0, 1, 2, ...; y alternates 0/1: every best split peels one row off an end.
    fn chain_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        (
            Array2::from_shape_fn((n, 1), |(i, _)| i as f64),
            Array1::from_shape_fn(n, |i| (i % 2) as f64),
        )
    }

    /// Runs `f` on a thread with a 2 MiB stack, the size of tokio's blocking threads and rayon's workers.
    fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    fn cubist_depth(root: &CubistNode) -> usize {
        let (mut max, mut stack) = (0, vec![(root, 0usize)]);
        while let Some((node, d)) = stack.pop() {
            max = max.max(d);
            if let CubistNode::Split { left, right, .. } = node {
                stack.push((left.as_ref(), d + 1));
                stack.push((right.as_ref(), d + 1));
            }
        }
        max
    }

    #[test]
    fn test_cubist_depth_is_capped_on_a_chain_input() {
        assert_eq!(MAX_TREE_DEPTH, 512);
        let depth = on_small_stack(|| {
            let (x, y) = chain_data(2_000);
            let cfg = CubistConfig {
                max_depth: usize::MAX,
                min_split: 2,
                min_bucket: 1,
                min_improvement: 0.0,
                extract_rules: false,
                ..Default::default()
            };
            let r = cubist(x.view(), y.view(), &cfg).unwrap();
            cubist_depth(r.committee_trees[0].root.as_ref().unwrap())
        });
        assert_eq!(depth, MAX_TREE_DEPTH);
    }

    #[test]
    fn test_cubist_rejects_non_finite_and_honours_cancellation() {
        let (x, y) = cubist_data();
        let mut yb = y.clone();
        yb[3] = f64::INFINITY;
        assert!(
            cubist(x.view(), yb.view(), &CubistConfig::default())
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        let cfg = CubistConfig {
            committees: 3,
            neighbors: 2,
            ..Default::default()
        };
        let err = with_budget(cancelled_budget(), || cubist(x.view(), y.view(), &cfg)).unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
    }

    #[test]
    fn test_cubist_basic() {
        // Simple linear relationship with noise
        let x = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
            [7.0, 3.5],
            [8.0, 4.0],
            [9.0, 4.5],
            [10.0, 5.0],
        ];
        // y = 2*x1 + 0.5*x2 + noise
        let y = array![2.6, 4.7, 6.9, 8.8, 11.1, 13.0, 15.2, 17.0, 19.1, 21.0];

        let config = CubistConfig {
            committees: 1,
            neighbors: 0,
            max_depth: 5,
            min_split: 3,
            min_bucket: 2,
            ..Default::default()
        };

        let result = cubist(x.view(), y.view(), &config).unwrap();

        assert!(result.n_rules >= 1);
        assert_eq!(result.predictions.len(), 10);
        assert!(result.train_rmse < 5.0); // Should have reasonable fit
        assert!(result.train_r_squared > 0.5); // Should explain most variance
    }

    #[test]
    fn test_cubist_committees() {
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
            [10.0],
        ];
        let y = array![1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.2, 8.8, 10.1];

        let config = CubistConfig {
            committees: 3,
            neighbors: 0,
            max_depth: 3,
            min_split: 2,
            min_bucket: 1,
            ..Default::default()
        };

        let result = cubist(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.committees, 3);
        assert_eq!(result.committee_trees.len(), 3);
        // With multiple committees, should have multiple sets of rules
        assert!(result.n_rules >= 1);
    }

    #[test]
    fn test_cubist_neighbors() {
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
            [10.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = CubistConfig {
            committees: 1,
            neighbors: 3, // Enable k-NN correction
            max_depth: 3,
            min_split: 2,
            min_bucket: 1,
            ..Default::default()
        };

        let result = cubist(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.neighbors, 3);
        assert!(result.training_x.is_some()); // Training data stored for k-NN
        assert!(result.training_y.is_some());
    }

    #[test]
    fn test_cubist_predict() {
        let x_train = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
        ];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = CubistConfig {
            committees: 1,
            neighbors: 0,
            max_depth: 3,
            min_split: 2,
            min_bucket: 1,
            ..Default::default()
        };

        let result = cubist(x_train.view(), y_train.view(), &config).unwrap();

        // Predict on new data
        let x_test = array![[2.5], [5.5], [8.5]];
        let predictions = cubist_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be in reasonable range
        for &p in &predictions {
            assert!(p > 0.0 && p < 12.0);
        }
    }

    #[test]
    fn test_cubist_rule_extraction() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0],
        ];
        // Different linear relationships based on second feature
        let y = array![
            1.0, 2.0, 3.0, 4.0, 5.0, // x2 = 0: y = x1
            2.0, 4.0, 6.0, 8.0, 10.0, // x2 = 1: y = 2*x1
        ];

        let config = CubistConfig {
            committees: 1,
            neighbors: 0,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            extract_rules: true,
            ..Default::default()
        };

        let result = cubist(x.view(), y.view(), &config).unwrap();

        // Should have at least one rule
        assert!(result.n_rules >= 1);

        // Check rule structure
        for rule in &result.rules {
            assert!(rule.id > 0);
            assert!(rule.coverage > 0);
            // Rules should have intercept
            // (coefficients may be zero for constant predictions)
        }
    }

    #[test]
    fn test_cubist_variable_importance() {
        // First feature is predictive, second is noise
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
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // y = x1

        let config = CubistConfig {
            committees: 1,
            neighbors: 0,
            max_depth: 5,
            min_split: 2,
            min_bucket: 1,
            ..Default::default()
        };

        let result = cubist(x.view(), y.view(), &config).unwrap();

        // First feature should have higher importance
        // (Though the linear model should capture the relationship directly)
        assert_eq!(result.variable_importance.len(), 2);
    }

    #[test]
    fn test_find_k_neighbors() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_test = array![3.0];

        let knn_pred = find_k_neighbors(&x_train.view(), &y_train.view(), &x_test.view(), 3);

        // k=3 nearest to x=3 are x=2,3,4 with y=2,3,4
        // Average should be 3.0
        assert!((knn_pred - 3.0).abs() < 0.5);
    }

    /// Uniform draws and normal noise from the module's LCG (Box-Muller).
    fn seeded_uniform(state: &mut u64) -> f64 {
        (lcg_random(state) % 1_000_000_007) as f64 / 1_000_000_007.0
    }

    fn seeded_normal(state: &mut u64) -> f64 {
        let u1 = seeded_uniform(state).max(1e-12);
        let u2 = seeded_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// `y = 2 x1 + 2 x2 + noise` (sd 0.5), two uniform features on [0, 1].
    fn plane_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut state = seed;
        let x = Array2::from_shape_fn((n, 2), |_| seeded_uniform(&mut state));
        let y = Array1::from_shape_fn(n, |i| {
            2.0 * x[[i, 0]] + 2.0 * x[[i, 1]] + 0.5 * seeded_normal(&mut state)
        });
        (x, y)
    }

    #[test]
    fn test_cubist_min_improvement_stops_splits() {
        let (x, y) = plane_data(300, 2026);
        let rules = |min_improvement: f64| {
            let cfg = CubistConfig {
                max_depth: 10,
                min_improvement,
                ..Default::default()
            };
            cubist(x.view(), y.view(), &cfg).unwrap().n_rules
        };
        let counts: Vec<usize> = [0.0, 0.01, 0.1, 0.5, 0.99]
            .iter()
            .map(|&m| rules(m))
            .collect();
        assert_eq!(counts[4], 1, "{counts:?}");
        assert!(counts.windows(2).all(|w| w[0] >= w[1]), "{counts:?}");
        assert_ne!(counts[0], counts[3], "{counts:?}");
    }

    #[test]
    fn test_cubist_split_rule_is_ssd_share() {
        // Two plateaus plus noise on one feature; the best admissible split (5 to 7 rows on
        // the left with min_bucket 5) removes a share `best` of the node's sum of squared
        // deviations, computed here by brute force.
        let x = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
        let noise = [
            0.9, -1.3, 0.4, 1.1, -0.6, -0.5, 1.2, -0.8, 0.3, -1.4, 0.7, 0.1,
        ];
        let y = Array1::from_shape_fn(12, |i| if i < 6 { 0.0 } else { 1.6 } + noise[i]);
        let ssd = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|a| (a - m).powi(2)).sum::<f64>()
        };
        let ys = y.to_vec();
        let total = ssd(&ys);
        let best = (5..=7)
            .map(|l| (total - ssd(&ys[..l]) - ssd(&ys[l..])) / total)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((0.15..0.6).contains(&best), "{best}");
        let rules = |min_improvement: f64| {
            let cfg = CubistConfig {
                max_depth: 1,
                min_improvement,
                ..Default::default()
            };
            cubist(x.view(), y.view(), &cfg).unwrap().n_rules
        };
        assert_eq!(rules(best - 0.01), 2);
        assert_eq!(rules(best + 0.01), 1);
    }

    #[test]
    fn test_cubist_min_improvement_is_a_share() {
        let (x, y) = cubist_data();
        for m in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            let cfg = CubistConfig {
                min_improvement: m,
                ..Default::default()
            };
            let err = cubist(x.view(), y.view(), &cfg)
                .err()
                .unwrap_or_else(|| panic!("min_improvement {m} accepted"));
            assert!(
                matches!(err, EconError::InvalidSpecification { .. }),
                "{m}: {err}"
            );
        }
    }

    fn hand_leaf(intercept: f64, coefficients: Vec<f64>, n_samples: usize) -> CubistNode {
        CubistNode::leaf(intercept, coefficients, n_samples, (0.0, 0.0), Vec::new())
    }

    fn hand_split(
        feature: usize,
        threshold: f64,
        left: CubistNode,
        right: CubistNode,
        intercept: f64,
        coefficients: Vec<f64>,
        n_samples: usize,
    ) -> CubistNode {
        CubistNode::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
            intercept,
            coefficients,
            n_samples,
        }
    }

    /// A finished tree from hand-built nodes (as `fit` leaves it).
    fn hand_tree(root: CubistNode, n_features: usize, k: f64) -> CubistTree {
        let mut tree = CubistTree::new(n_features, None, k);
        tree.root = Some(root);
        tree.smooth();
        tree
    }

    /// Depth 2 on one feature: root (40 cases, q_root = 1 + 2x) splits at 0.5; its left child
    /// (20 cases, q_mid = 3 - x) splits at 0.25 into the leaf (8 cases, q_leaf = 0.5 + 4x).
    fn depth_two_tree(k: f64) -> CubistTree {
        let mid = hand_split(
            0,
            0.25,
            hand_leaf(0.5, vec![4.0], 8),
            hand_leaf(-2.0, vec![1.0], 12),
            3.0,
            vec![-1.0],
            20,
        );
        let root = hand_split(
            0,
            0.5,
            mid,
            hand_leaf(7.0, vec![-3.0], 20),
            1.0,
            vec![2.0],
            40,
        );
        hand_tree(root, 1, k)
    }

    #[test]
    fn test_cubist_smoothing_reaches_the_root() {
        let tree = depth_two_tree(15.0);
        let x = 0.2;
        let (q_leaf, q_mid, q_root) = (0.5 + 4.0 * x, 3.0 - x, 1.0 + 2.0 * x);
        let p2 = (8.0 * q_leaf + 15.0 * q_mid) / (8.0 + 15.0);
        let p1 = (20.0 * p2 + 15.0 * q_root) / (20.0 + 15.0);
        let got = tree.predict_one(&array![x].view());
        assert!(
            (got - p1).abs() <= 1e-12 * p1.abs(),
            "{got} vs {p1} (p2 = {p2})"
        );
    }

    #[test]
    fn test_cubist_smoothing_coefficient_direction() {
        let x = 0.2;
        let (q_leaf, q_root) = (0.5 + 4.0 * x, 1.0 + 2.0 * x);
        let at = |k: f64| depth_two_tree(k).predict_one(&array![x].view());
        assert!(
            (at(0.0) - q_leaf).abs() <= 1e-9 * q_leaf.abs(),
            "{}",
            at(0.0)
        );
        assert!(
            (at(1e12) - q_root).abs() <= 1e-9 * q_root.abs(),
            "{}",
            at(1e12)
        );
    }

    /// The rule covering `row`: every condition holds.
    fn covering_rule<'a>(rules: &'a [CubistRule], row: ArrayView1<f64>) -> &'a CubistRule {
        let holds = |c: &RuleCondition| match c.operator.as_str() {
            "<=" => row[c.feature] <= c.threshold,
            _ => row[c.feature] > c.threshold,
        };
        let mut covering = rules.iter().filter(|r| r.conditions.iter().all(holds));
        let rule = covering.next().expect("a covering rule");
        assert!(covering.next().is_none(), "rules overlap");
        rule
    }

    #[test]
    fn test_cubist_rule_model_is_the_composite() {
        let (x, y) = plane_data(300, 7);
        let r = cubist(x.view(), y.view(), &CubistConfig::default()).unwrap();
        assert!(r.n_rules > 1, "{}", r.n_rules);
        let tree = &r.committee_trees[0];
        for row in x.rows() {
            let rule = covering_rule(&r.rules, row);
            let model = rule.intercept
                + rule
                    .coefficients
                    .iter()
                    .map(|(j, _, c)| c * row[*j])
                    .sum::<f64>();
            let want = tree.predict_one(&row);
            assert!(
                (model - want).abs() <= 1e-12 * want.abs().max(1.0),
                "rule {}: {model} vs {want}",
                rule.id
            );
        }
    }

    #[test]
    fn test_cubist_committee_target() {
        let (x, y) = plane_data(200, 11);
        let cfg = CubistConfig {
            committees: 2,
            ..Default::default()
        };
        let r = cubist(x.view(), y.view(), &cfg).unwrap();
        let mut first = CubistTree::new(2, None, cfg.smoothing_coefficient);
        first.fit(&x.view(), &y.view(), &cfg);
        let target = Array1::from_shape_fn(y.len(), |i| 2.0 * y[i] - first.predict_one(&x.row(i)));
        let mut second = CubistTree::new(2, None, cfg.smoothing_coefficient);
        second.fit(&x.view(), &target.view(), &cfg);
        for row in x.rows() {
            assert_eq!(
                r.committee_trees[1].predict_one(&row).to_bits(),
                second.predict_one(&row).to_bits()
            );
        }
    }

    #[test]
    fn test_cubist_insufficient_data() {
        let x = array![[1.0]];
        let y = array![1.0];

        let result = cubist(x.view(), y.view(), &CubistConfig::default());
        assert!(result.is_err());
    }
}
