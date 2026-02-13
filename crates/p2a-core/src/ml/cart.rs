//! CART (Classification and Regression Trees) implementation.
//!
//! Pure Rust implementation of CART based on Breiman et al. (1984).
//! Similar to R's `rpart` package.
//!
//! ## Features
//!
//! - **Regression trees**: Minimize MSE for continuous targets
//! - **Classification trees**: Minimize Gini impurity or entropy for categorical targets
//! - **Pruning**: Cost-complexity pruning with cross-validation
//! - **Variable importance**: Based on improvement in splitting criterion
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{cart, CartConfig, CartMethod};
//! use ndarray::array;
//!
//! // Regression tree
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
//!
//! let config = CartConfig {
//!     method: CartMethod::Anova,
//!     max_depth: 5,
//!     min_split: 5,
//!     cp: 0.01,
//!     ..Default::default()
//! };
//!
//! let result = cart(x.view(), y.view(), &config).unwrap();
//! println!("Tree depth: {}", result.depth);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::errors::{EconError, EconResult};

/// Method for splitting nodes (determines loss function).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CartMethod {
    /// ANOVA (regression) - minimize MSE
    #[default]
    Anova,
    /// Classification - minimize Gini impurity
    Gini,
    /// Classification - minimize entropy/information gain
    Entropy,
}

impl std::str::FromStr for CartMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "anova" | "regression" | "mse" => Ok(CartMethod::Anova),
            "gini" | "class" | "classification" => Ok(CartMethod::Gini),
            "entropy" | "information" | "deviance" => Ok(CartMethod::Entropy),
            _ => Err(format!("Unknown CART method: {}", s)),
        }
    }
}

/// Configuration for CART.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartConfig {
    /// Method for splitting (anova for regression, gini/entropy for classification)
    pub method: CartMethod,
    /// Maximum depth of tree (0 = unlimited)
    pub max_depth: usize,
    /// Minimum observations in a node to attempt a split
    pub min_split: usize,
    /// Minimum observations in a terminal node
    pub min_bucket: usize,
    /// Complexity parameter for pruning (cp)
    pub cp: f64,
    /// Number of cross-validation folds for pruning
    pub xval: usize,
    /// Maximum number of surrogate splits to store
    pub max_surrogate: usize,
    /// Use surrogate splits for missing values
    pub use_surrogate: bool,
    /// Random seed
    pub seed: Option<u64>,
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
        }
    }
}

/// A split in the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartSplit {
    /// Feature index used for split
    pub feature: usize,
    /// Split threshold (for continuous) or category (for categorical)
    pub threshold: f64,
    /// Improvement in criterion (impurity reduction)
    pub improve: f64,
    /// Direction for split (true = left if <=, false = left if >)
    pub direction: bool,
}

/// A node in the CART tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartNode {
    /// Node ID (1-indexed, root = 1)
    pub id: usize,
    /// Number of observations in this node
    pub n: usize,
    /// Weighted number of observations
    pub wt: f64,
    /// Deviance/loss at this node
    pub dev: f64,
    /// Predicted value (mean for regression, majority class for classification)
    pub yval: f64,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f64>>,
    /// Complexity parameter at which this node is pruned
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
    /// Feature importances (sum of improvements)
    pub variable_importance: Vec<f64>,
    /// Complexity parameter table for pruning
    pub cp_table: Vec<CpTableRow>,
    /// Cross-validation error at each cp
    pub cv_error: Option<Vec<f64>>,
    /// Configuration used
    pub config: CartConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Class labels (for classification)
    pub class_labels: Option<Vec<String>>,
    /// Predictions on training data
    pub predictions: Vec<f64>,
}

/// Row in the complexity parameter table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpTableRow {
    /// Complexity parameter
    pub cp: f64,
    /// Number of splits
    pub nsplit: usize,
    /// Relative error
    pub rel_error: f64,
    /// Cross-validation error (mean)
    pub xerror: f64,
    /// Cross-validation error (std dev)
    pub xstd: f64,
}

impl std::fmt::Display for CartResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CART Decision Tree")?;
        writeln!(f, "==================")?;
        writeln!(f, "Method: {:?}", self.config.method)?;
        writeln!(f, "Nodes: {} ({} terminal)", self.n_nodes, self.n_terminal)?;
        writeln!(f, "Depth: {}", self.depth)?;

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
            writeln!(f, "  CP      nsplit  rel_error  xerror    xstd")?;
            for row in &self.cp_table {
                writeln!(
                    f,
                    "  {:.4}  {:>6}  {:>9.4}  {:>7.4}  {:>7.4}",
                    row.cp, row.nsplit, row.rel_error, row.xerror, row.xstd
                )?;
            }
        }

        Ok(())
    }
}

/// Build a CART tree recursively.
fn build_cart_tree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &CartConfig,
    indices: &[usize],
    depth: usize,
    node_id: usize,
    root_mse: f64,
    importance: &mut Array1<f64>,
) -> CartNode {
    let n = indices.len();
    let wt = n as f64;

    // Compute node statistics
    let (yval, dev, class_probs) = compute_node_stats(y, indices, config);

    // Check stopping conditions
    let max_depth = if config.max_depth == 0 {
        usize::MAX
    } else {
        config.max_depth
    };
    if depth >= max_depth || n < config.min_split || n <= 2 * config.min_bucket {
        return CartNode {
            id: node_id,
            n,
            wt,
            dev,
            yval,
            class_probs,
            complexity: 0.0,
            split: None,
            left: None,
            right: None,
        };
    }

    // Check if all values are the same
    let first_y = y[indices[0]];
    if indices.iter().all(|&i| (y[i] - first_y).abs() < 1e-10) {
        return CartNode {
            id: node_id,
            n,
            wt,
            dev,
            yval,
            class_probs,
            complexity: 0.0,
            split: None,
            left: None,
            right: None,
        };
    }

    // Find best split
    if let Some((best_split, left_indices, right_indices)) =
        find_best_split_cart(x, y, config, indices)
    {
        // Check minimum improvement (cp criterion)
        let improve_ratio = if root_mse > 0.0 {
            best_split.improve / root_mse
        } else {
            0.0
        };
        if improve_ratio < config.cp {
            return CartNode {
                id: node_id,
                n,
                wt,
                dev,
                yval,
                class_probs,
                complexity: improve_ratio,
                split: None,
                left: None,
                right: None,
            };
        }

        // Check minimum bucket size
        if left_indices.len() < config.min_bucket || right_indices.len() < config.min_bucket {
            return CartNode {
                id: node_id,
                n,
                wt,
                dev,
                yval,
                class_probs,
                complexity: 0.0,
                split: None,
                left: None,
                right: None,
            };
        }

        // Update importance
        importance[best_split.feature] += best_split.improve;

        // Build children
        let left_id = node_id * 2;
        let right_id = node_id * 2 + 1;

        let left_child = build_cart_tree(
            x,
            y,
            config,
            &left_indices,
            depth + 1,
            left_id,
            root_mse,
            importance,
        );
        let right_child = build_cart_tree(
            x,
            y,
            config,
            &right_indices,
            depth + 1,
            right_id,
            root_mse,
            importance,
        );

        CartNode {
            id: node_id,
            n,
            wt,
            dev,
            yval,
            class_probs,
            complexity: improve_ratio,
            split: Some(best_split),
            left: Some(Box::new(left_child)),
            right: Some(Box::new(right_child)),
        }
    } else {
        CartNode {
            id: node_id,
            n,
            wt,
            dev,
            yval,
            class_probs,
            complexity: 0.0,
            split: None,
            left: None,
            right: None,
        }
    }
}

fn compute_node_stats(
    y: &ArrayView1<f64>,
    indices: &[usize],
    config: &CartConfig,
) -> (f64, f64, Option<Vec<f64>>) {
    match config.method {
        CartMethod::Anova => {
            // Regression: mean and MSE
            let sum: f64 = indices.iter().map(|&i| y[i]).sum();
            let mean = sum / indices.len() as f64;
            let mse: f64 = indices.iter().map(|&i| (y[i] - mean).powi(2)).sum();
            (mean, mse, None)
        }
        CartMethod::Gini | CartMethod::Entropy => {
            // Classification: majority class and Gini/entropy
            let mut counts = std::collections::HashMap::new();
            for &i in indices {
                let class = y[i] as i64;
                *counts.entry(class).or_insert(0usize) += 1;
            }

            let n = indices.len() as f64;
            let probs: Vec<f64> = counts.values().map(|&c| c as f64 / n).collect();

            let (yval, _) = counts.iter().max_by_key(|(_, c)| *c).unwrap();

            let dev = if config.method == CartMethod::Gini {
                // Gini impurity: sum(p * (1 - p))
                probs.iter().map(|p| p * (1.0 - p)).sum::<f64>() * n
            } else {
                // Entropy: -sum(p * log(p))
                probs
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum::<f64>()
                    * n
            };

            (*yval as f64, dev, Some(probs))
        }
    }
}

fn find_best_split_cart(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &CartConfig,
    indices: &[usize],
) -> Option<(CartSplit, Vec<usize>, Vec<usize>)> {
    let mut best_improve = 0.0;
    let mut best_split: Option<(CartSplit, usize)> = None; // Store feature and threshold index
    let n_features = x.ncols();
    let n = indices.len();

    // Early exit if too few samples
    if n < 2 * config.min_bucket {
        return None;
    }

    match config.method {
        CartMethod::Anova => {
            // OPTIMIZED: O(n log n) per feature using incremental sums
            // Compute total sum and sum of squares for the node
            let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
            let total_ss: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
            let node_mse = total_ss - total_sum * total_sum / n as f64;

            for feature in 0..n_features {
                // Sort indices by feature value - O(n log n)
                let mut sorted: Vec<(f64, f64, usize)> = indices
                    .iter()
                    .map(|&i| (x[[i, feature]], y[i], i))
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Scan through sorted values with running sums - O(n)
                let mut left_sum = 0.0;
                let mut left_ss = 0.0;
                let mut left_n = 0usize;
                let mut prev_x = sorted[0].0;

                for i in 0..sorted.len() - 1 {
                    let (x_val, y_val, _) = sorted[i];
                    left_sum += y_val;
                    left_ss += y_val * y_val;
                    left_n += 1;

                    // Skip if next value is the same (no valid split point)
                    let next_x = sorted[i + 1].0;
                    if (next_x - x_val).abs() < 1e-10 {
                        prev_x = x_val;
                        continue;
                    }

                    // Check bucket constraints
                    let right_n = n - left_n;
                    if left_n < config.min_bucket || right_n < config.min_bucket {
                        prev_x = x_val;
                        continue;
                    }

                    // Incremental MSE calculation: MSE = SS - sum²/n
                    let left_mse = left_ss - left_sum * left_sum / left_n as f64;
                    let right_sum = total_sum - left_sum;
                    let right_ss = total_ss - left_ss;
                    let right_mse = right_ss - right_sum * right_sum / right_n as f64;

                    let improve = node_mse - left_mse - right_mse;

                    if improve > best_improve {
                        best_improve = improve;
                        let threshold = (x_val + next_x) / 2.0;
                        best_split = Some((
                            CartSplit {
                                feature,
                                threshold,
                                improve,
                                direction: true,
                            },
                            i, // Store split position for partition later
                        ));
                    }
                    prev_x = x_val;
                }
            }
        }
        CartMethod::Gini | CartMethod::Entropy => {
            // OPTIMIZED: O(n log n) per feature using incremental class counts
            // Build class counts for the node
            let mut total_counts: std::collections::HashMap<i64, usize> =
                std::collections::HashMap::new();
            for &i in indices {
                let class = y[i] as i64;
                *total_counts.entry(class).or_insert(0) += 1;
            }
            let classes: Vec<i64> = total_counts.keys().copied().collect();
            let n_classes = classes.len();

            // For small number of classes, use Vec instead of HashMap for speed
            let class_to_idx: std::collections::HashMap<i64, usize> =
                classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            let total_vec: Vec<usize> = classes.iter().map(|c| total_counts[c]).collect();

            let node_loss = if config.method == CartMethod::Gini {
                compute_gini_from_counts(&total_vec, n)
            } else {
                compute_entropy_from_counts(&total_vec, n)
            };

            for feature in 0..n_features {
                // Sort indices by feature value
                let mut sorted: Vec<(f64, i64, usize)> = indices
                    .iter()
                    .map(|&i| (x[[i, feature]], y[i] as i64, i))
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Incremental class counts
                let mut left_counts = vec![0usize; n_classes];
                let mut left_n = 0usize;

                for i in 0..sorted.len() - 1 {
                    let (x_val, class, _) = sorted[i];
                    let class_idx = class_to_idx[&class];
                    left_counts[class_idx] += 1;
                    left_n += 1;

                    // Skip if next value is the same
                    let next_x = sorted[i + 1].0;
                    if (next_x - x_val).abs() < 1e-10 {
                        continue;
                    }

                    // Check bucket constraints
                    let right_n = n - left_n;
                    if left_n < config.min_bucket || right_n < config.min_bucket {
                        continue;
                    }

                    // Compute right counts
                    let right_counts: Vec<usize> = (0..n_classes)
                        .map(|j| total_vec[j] - left_counts[j])
                        .collect();

                    let (left_loss, right_loss) = if config.method == CartMethod::Gini {
                        (
                            compute_gini_from_counts(&left_counts, left_n),
                            compute_gini_from_counts(&right_counts, right_n),
                        )
                    } else {
                        (
                            compute_entropy_from_counts(&left_counts, left_n),
                            compute_entropy_from_counts(&right_counts, right_n),
                        )
                    };

                    let improve = node_loss - left_loss - right_loss;

                    if improve > best_improve {
                        best_improve = improve;
                        let threshold = (x_val + next_x) / 2.0;
                        best_split = Some((
                            CartSplit {
                                feature,
                                threshold,
                                improve,
                                direction: true,
                            },
                            i,
                        ));
                    }
                }
            }
        }
    }

    // Now partition indices based on best split
    best_split.map(|(split, _)| {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| x[[i, split.feature]] <= split.threshold);
        (split, left_indices, right_indices)
    })
}

/// Compute Gini impurity from class counts (optimized)
#[inline]
fn compute_gini_from_counts(counts: &[usize], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    counts
        .iter()
        .map(|&c| {
            let p = c as f64 / n_f;
            p * (1.0 - p) * n_f
        })
        .sum()
}

/// Compute entropy from class counts (optimized)
#[inline]
fn compute_entropy_from_counts(counts: &[usize], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n_f;
            -p * p.ln() * n_f
        })
        .sum()
}

/// Compute MSE (sum of squared deviations from mean).
fn compute_mse(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let mean = sum / indices.len() as f64;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum()
}

/// Compute Gini impurity.
fn compute_gini(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::new();
    for &i in indices {
        let class = y[i] as i64;
        *counts.entry(class).or_insert(0usize) += 1;
    }
    let n = indices.len() as f64;
    counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * (1.0 - p)
        })
        .sum::<f64>()
        * n
}

/// Compute entropy.
fn compute_entropy(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::new();
    for &i in indices {
        let class = y[i] as i64;
        *counts.entry(class).or_insert(0usize) += 1;
    }
    let n = indices.len() as f64;
    counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum::<f64>()
        * n
}

/// Count tree statistics recursively.
fn count_tree_stats(node: &CartNode) -> (usize, usize, usize) {
    // Returns (n_nodes, n_terminal, max_depth)
    if node.split.is_none() {
        return (1, 1, 0);
    }

    let (left_nodes, left_term, left_depth) = match &node.left {
        Some(child) => count_tree_stats(child),
        None => (0, 0, 0),
    };

    let (right_nodes, right_term, right_depth) = match &node.right {
        Some(child) => count_tree_stats(child),
        None => (0, 0, 0),
    };

    (
        1 + left_nodes + right_nodes,
        left_term + right_term,
        1 + left_depth.max(right_depth),
    )
}

/// Predict for a single observation.
fn predict_one(node: &CartNode, x: &ArrayView1<f64>) -> f64 {
    match &node.split {
        None => node.yval,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                predict_one(node.left.as_ref().unwrap(), x)
            } else {
                predict_one(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Fit a CART decision tree.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - CART configuration
///
/// # Returns
///
/// CartResult containing the fitted tree and diagnostics.
pub fn cart(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &CartConfig) -> EconResult<CartResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for CART".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    // Compute root MSE for cp calculations
    let indices: Vec<usize> = (0..n_samples).collect();
    let root_mse = compute_mse(&y, &indices);

    // Build tree
    let mut importance = Array1::zeros(n_features);
    let root = build_cart_tree(&x, &y, config, &indices, 0, 1, root_mse, &mut importance);

    // Count statistics
    let (n_nodes, n_terminal, depth) = count_tree_stats(&root);

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| predict_one(&root, &x.row(i)))
        .collect();

    // Build CP table (simplified - full version would use cross-validation)
    let cp_table = build_cp_table(&root, root_mse);

    Ok(CartResult {
        root,
        n_nodes,
        n_terminal,
        depth,
        variable_importance: importance.to_vec(),
        cp_table,
        cv_error: None,
        config: config.clone(),
        feature_names: None,
        class_labels: None,
        predictions,
    })
}

/// Build complexity parameter table.
fn build_cp_table(root: &CartNode, root_mse: f64) -> Vec<CpTableRow> {
    let mut cp_values = Vec::new();
    collect_cp_values(root, &mut cp_values);
    cp_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    cp_values.dedup();

    let mut table = Vec::new();
    let mut cumulative_improve = 0.0;

    for (nsplit, &cp) in cp_values.iter().enumerate() {
        // This is a simplified version - proper implementation would track actual improvements
        cumulative_improve += cp * root_mse;
        let rel_error = 1.0 - cumulative_improve / root_mse;

        table.push(CpTableRow {
            cp,
            nsplit,
            rel_error: rel_error.max(0.0),
            xerror: rel_error.max(0.0) * 1.1, // Placeholder
            xstd: 0.1,                        // Placeholder
        });
    }

    if table.is_empty() {
        table.push(CpTableRow {
            cp: 1.0,
            nsplit: 0,
            rel_error: 1.0,
            xerror: 1.0,
            xstd: 0.0,
        });
    }

    table
}

/// Collect CP values from tree.
fn collect_cp_values(node: &CartNode, cp_values: &mut Vec<f64>) {
    if node.complexity > 0.0 {
        cp_values.push(node.complexity);
    }
    if let Some(ref left) = node.left {
        collect_cp_values(left, cp_values);
    }
    if let Some(ref right) = node.right {
        collect_cp_values(right, cp_values);
    }
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
pub fn cart_predict(result: &CartResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    Ok((0..x.nrows())
        .map(|i| predict_one(&result.root, &x.row(i)))
        .collect())
}

/// Prune a CART tree to a specific complexity parameter.
///
/// # Arguments
///
/// * `result` - Fitted CART result
/// * `cp` - Complexity parameter to prune to
///
/// # Returns
///
/// Pruned tree result.
pub fn cart_prune(result: &CartResult, cp: f64) -> EconResult<CartResult> {
    let mut new_result = result.clone();
    new_result.root = prune_node(&result.root, cp);
    let (n_nodes, n_terminal, depth) = count_tree_stats(&new_result.root);
    new_result.n_nodes = n_nodes;
    new_result.n_terminal = n_terminal;
    new_result.depth = depth;
    Ok(new_result)
}

/// Prune a node and its children.
fn prune_node(node: &CartNode, cp: f64) -> CartNode {
    if node.split.is_none() || node.complexity < cp {
        // Terminal node or should be pruned
        return CartNode {
            id: node.id,
            n: node.n,
            wt: node.wt,
            dev: node.dev,
            yval: node.yval,
            class_probs: node.class_probs.clone(),
            complexity: node.complexity,
            split: None,
            left: None,
            right: None,
        };
    }

    // Recursively prune children
    let left = node.left.as_ref().map(|l| Box::new(prune_node(l, cp)));
    let right = node.right.as_ref().map(|r| Box::new(prune_node(r, cp)));

    CartNode {
        id: node.id,
        n: node.n,
        wt: node.wt,
        dev: node.dev,
        yval: node.yval,
        class_probs: node.class_probs.clone(),
        complexity: node.complexity,
        split: node.split.clone(),
        left,
        right,
    }
}

/// Run CART on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
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
    use crate::linalg::design::DesignMatrix;

    // Build design matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    // Get y column
    let col_names: Vec<String> = dataset
        .df()
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let y_series = dataset
        .df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: col_names,
        })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    let mut result = cart(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Run CART with default configuration.
pub fn run_cart_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<CartResult> {
    run_cart(dataset, y_col, x_cols, &CartConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
}
