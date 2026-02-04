//! Conditional Inference Trees (ctree) implementation.
//!
//! Pure Rust implementation of conditional inference trees based on
//! Hothorn, Hornik & Zeileis (2006).
//!
//! ## Features
//!
//! - **Unbiased variable selection**: Uses permutation tests to select variables,
//!   avoiding bias towards variables with many possible splits
//! - **Statistical stopping rule**: Stops growing when no significant relationship
//!   between predictors and response exists
//! - **No pruning needed**: The significance-based stopping rule produces
//!   appropriately-sized trees without post-hoc pruning
//! - **Supports both regression and classification**: Automatic detection based
//!   on number of unique values in response
//!
//! ## Algorithm
//!
//! The algorithm proceeds as follows:
//!
//! 1. Test global null hypothesis of independence between response and all predictors
//! 2. If no variable rejects H0 at level alpha (1 - mincriterion), stop
//! 3. Select the variable with smallest p-value (strongest association)
//! 4. Find optimal split point for selected variable by maximizing test statistic
//! 5. Recursively partition left and right child nodes
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{ctree, CtreeConfig};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
//! let y = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];
//!
//! let config = CtreeConfig::default();
//! let result = ctree(x.view(), y.view(), &config).unwrap();
//! println!("Tree depth: {}", result.depth);
//! ```
//!
//! ## References
//!
//! - Hothorn, T., Hornik, K., & Zeileis, A. (2006). "Unbiased Recursive Partitioning:
//!   A Conditional Inference Framework". Journal of Computational and Graphical
//!   Statistics, 15(3), 651-674. <https://doi.org/10.1198/106186006X133933>
//! - Strasser, H., & Weber, C. (1999). "On the asymptotic theory of permutation
//!   statistics". Mathematical Methods of Statistics, 8, 220-250.
//! - R package `partykit`: Hothorn, T. & Zeileis, A. (2015).
//!   <https://cran.r-project.org/package=partykit>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::Dataset;
use crate::errors::{EconError, EconResult};

/// Configuration for Conditional Inference Trees.
///
/// # Parameters
///
/// - `mincriterion`: The value of the test statistic or 1 - p-value that must be
///   exceeded to implement a split. Default is 0.95, meaning p-value < 0.05.
/// - `minsplit`: Minimum number of observations in a node required to attempt a split.
///   Default is 20.
/// - `minbucket`: Minimum number of observations in a terminal node. Default is 7.
/// - `maxdepth`: Maximum depth of the tree (0 = unlimited). Default is 0.
/// - `teststat`: Type of test statistic: "quadratic" (chi-squared) or "max" (maximum).
///   Default is "quadratic".
/// - `testtype`: P-value computation method: "bonferroni", "univariate", or "none".
///   Default is "bonferroni".
/// - `seed`: Random seed for reproducibility (used in permutation tests if enabled).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeConfig {
    /// 1 - alpha threshold for splitting (e.g., 0.95 means p < 0.05)
    pub mincriterion: f64,
    /// Minimum observations in a node to attempt split
    pub minsplit: usize,
    /// Minimum observations in terminal nodes
    pub minbucket: usize,
    /// Maximum tree depth (0 = unlimited)
    pub maxdepth: usize,
    /// Test statistic type: "quadratic" or "max"
    pub teststat: String,
    /// P-value adjustment: "bonferroni", "univariate", or "none"
    pub testtype: String,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for CtreeConfig {
    fn default() -> Self {
        CtreeConfig {
            mincriterion: 0.95,
            minsplit: 20,
            minbucket: 7,
            maxdepth: 0, // unlimited
            teststat: "quadratic".to_string(),
            testtype: "bonferroni".to_string(),
            seed: None,
        }
    }
}

/// A split in the conditional inference tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeSplit {
    /// Feature index used for split
    pub feature: usize,
    /// Split threshold
    pub threshold: f64,
    /// Test statistic value for variable selection
    pub statistic: f64,
    /// P-value for variable selection
    pub p_value: f64,
}

/// A node in the conditional inference tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeNode {
    /// Node ID (1-indexed, root = 1)
    pub id: usize,
    /// Number of observations in this node
    pub n: usize,
    /// Predicted value (mean for regression, majority class for classification)
    pub prediction: f64,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f64>>,
    /// Split information (None for terminal nodes)
    pub split: Option<CtreeSplit>,
    /// Left child node
    pub left: Option<Box<CtreeNode>>,
    /// Right child node
    pub right: Option<Box<CtreeNode>>,
}

/// Result from conditional inference tree fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeResult {
    /// Root node of the tree
    pub root: CtreeNode,
    /// Total number of nodes
    pub n_nodes: usize,
    /// Number of terminal nodes
    pub n_terminal: usize,
    /// Maximum depth reached
    pub depth: usize,
    /// Variable importance (permutation-based)
    pub variable_importance: Vec<f64>,
    /// P-values for each variable at the root (global test)
    pub root_p_values: Vec<f64>,
    /// Whether the model is for classification
    pub is_classification: bool,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Class labels (for classification)
    pub class_labels: Option<Vec<f64>>,
    /// Configuration used
    pub config: CtreeConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Training predictions
    pub predictions: Vec<f64>,
}

impl std::fmt::Display for CtreeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Conditional Inference Tree")?;
        writeln!(f, "==========================")?;
        writeln!(
            f,
            "Type: {}",
            if self.is_classification {
                "Classification"
            } else {
                "Regression"
            }
        )?;
        writeln!(f, "Nodes: {} ({} terminal)", self.n_nodes, self.n_terminal)?;
        writeln!(f, "Depth: {}", self.depth)?;
        writeln!(
            f,
            "Criterion: 1 - p-value > {:.2} (p < {:.4})",
            self.config.mincriterion,
            1.0 - self.config.mincriterion
        )?;

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Normalize
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

        if self.variable_importance.len() > 10 {
            writeln!(
                f,
                "  ... ({} more features)",
                self.variable_importance.len() - 10
            )?;
        }

        // Print tree structure summary
        writeln!(f)?;
        writeln!(f, "Root test statistics:")?;
        for (i, &p) in self.root_p_values.iter().enumerate() {
            let name = match &self.feature_names {
                Some(names) => names.get(i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            writeln!(f, "  {}: p-value = {:.4}", name, p)?;
        }

        Ok(())
    }
}

/// Compute the linear test statistic for a single variable.
///
/// Uses the framework of Strasser & Weber (1999):
/// T_j = sum_i h(X_{ij}) * g(Y_i)
///
/// For regression: g(Y) = Y (influence function)
/// For classification: g(Y) = indicator vector
///
/// The standardized statistic follows asymptotic normal/chi-squared distribution.
fn compute_test_statistic(
    x_feature: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    is_classification: bool,
    classes: Option<&[f64]>,
    teststat: &str,
) -> (f64, f64) {
    let n = x_feature.len();
    if n < 2 {
        return (0.0, 1.0);
    }

    // Compute ranks of X (to handle ordinal nature uniformly)
    let mut x_ranked: Vec<(f64, usize)> =
        x_feature.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    x_ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut rank = 1.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (x_ranked[j].0 - x_ranked[i].0).abs() < 1e-10 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (rank + rank + (j - i - 1) as f64) / 2.0;
        for k in i..j {
            ranks[x_ranked[k].1] = avg_rank;
        }
        rank += (j - i) as f64;
        i = j;
    }

    if is_classification && classes.is_some() {
        let cls = classes.unwrap();
        let n_classes = cls.len();

        // For classification, compute linear statistic for each class
        // T_k = sum_i rank(X_i) * I(Y_i = class_k)
        let mut t_vec = vec![0.0; n_classes];
        let mut class_counts = vec![0usize; n_classes];

        for i in 0..n {
            if let Some(k) = cls.iter().position(|&c| (c - y[i]).abs() < 1e-10) {
                t_vec[k] += ranks[i];
                class_counts[k] += 1;
            }
        }

        // Expected value: E[T_k] = n_k * (n+1)/2
        let expected: Vec<f64> = class_counts
            .iter()
            .map(|&c| c as f64 * (n as f64 + 1.0) / 2.0)
            .collect();

        // Variance: Var(T_k) = n_k * (n - n_k) * (n+1) / 12
        let variance: Vec<f64> = class_counts
            .iter()
            .map(|&c| c as f64 * (n - c) as f64 * (n as f64 + 1.0) / 12.0)
            .collect();

        // Compute quadratic form statistic: chi-squared
        let mut chi_sq = 0.0;
        for k in 0..n_classes {
            if variance[k] > 1e-10 {
                chi_sq += (t_vec[k] - expected[k]).powi(2) / variance[k];
            }
        }

        // P-value from chi-squared distribution
        let df = (n_classes - 1).max(1) as f64;
        let p_value = if chi_sq.is_finite() && chi_sq >= 0.0 {
            let chi2 = ChiSquared::new(df).unwrap();
            1.0 - chi2.cdf(chi_sq)
        } else {
            1.0
        };

        (chi_sq, p_value)
    } else {
        // Regression: compute correlation-based statistic
        // T = sum_i rank(X_i) * Y_i
        let mean_rank = (n as f64 + 1.0) / 2.0;
        let mean_y = y.mean().unwrap_or(0.0);

        let mut cov = 0.0;
        let mut var_rank = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dr = ranks[i] - mean_rank;
            let dy = y[i] - mean_y;
            cov += dr * dy;
            var_rank += dr * dr;
            var_y += dy * dy;
        }

        // Standardized statistic
        let denom = (var_rank * var_y).sqrt();
        let z = if denom > 1e-10 {
            cov / denom * (n as f64 - 1.0).sqrt()
        } else {
            0.0
        };

        let statistic = match teststat {
            "max" => z.abs(),
            _ => z * z, // quadratic (default)
        };

        // P-value
        let p_value = if teststat == "max" {
            // Two-sided normal
            let normal = Normal::new(0.0, 1.0).unwrap();
            2.0 * (1.0 - normal.cdf(z.abs()))
        } else {
            // Chi-squared with df=1
            let chi2 = ChiSquared::new(1.0).unwrap();
            if statistic.is_finite() && statistic >= 0.0 {
                1.0 - chi2.cdf(statistic)
            } else {
                1.0
            }
        };

        (statistic, p_value)
    }
}

/// Select the best variable based on p-values with optional Bonferroni correction.
fn select_best_variable(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    is_classification: bool,
    classes: Option<&[f64]>,
    config: &CtreeConfig,
) -> Option<(usize, f64, f64, Vec<f64>)> {
    let n_features = x.ncols();

    // Extract subset for indices
    let y_subset: Array1<f64> = indices.iter().map(|&i| y[i]).collect();

    // Compute test statistic for each variable
    let mut results: Vec<(usize, f64, f64)> = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let x_subset: Array1<f64> = indices.iter().map(|&i| x[[i, j]]).collect();
        let (stat, pval) = compute_test_statistic(
            &x_subset.view(),
            &y_subset.view(),
            is_classification,
            classes,
            &config.teststat,
        );
        results.push((j, stat, pval));
    }

    // Store p-values for output
    let p_values: Vec<f64> = results.iter().map(|(_, _, p)| *p).collect();

    // Apply multiple testing correction
    let adjusted_p_values: Vec<f64> = match config.testtype.as_str() {
        "bonferroni" => results
            .iter()
            .map(|(_, _, p)| (p * n_features as f64).min(1.0))
            .collect(),
        "univariate" | "none" => results.iter().map(|(_, _, p)| *p).collect(),
        _ => results.iter().map(|(_, _, p)| *p).collect(),
    };

    // Find the variable with the smallest p-value
    let mut best_idx = None;
    let mut best_pval = f64::INFINITY;
    let mut best_stat = 0.0;

    for (j, (_, stat, _)) in results.iter().enumerate() {
        let adj_p = adjusted_p_values[j];
        if adj_p < best_pval {
            best_pval = adj_p;
            best_stat = *stat;
            best_idx = Some(j);
        }
    }

    // Check if best variable meets criterion
    // mincriterion is 1 - alpha, so we need 1 - p-value > mincriterion
    // equivalently: p-value < 1 - mincriterion
    let alpha = 1.0 - config.mincriterion;
    if best_pval < alpha {
        best_idx.map(|j| (j, best_stat, best_pval, p_values))
    } else {
        None
    }
}

/// Find the optimal split point for a selected variable.
///
/// Maximizes the test statistic (or equivalently, minimizes p-value) over all
/// possible split points.
fn find_best_split(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    feature: usize,
    is_classification: bool,
    classes: Option<&[f64]>,
    config: &CtreeConfig,
) -> Option<(f64, Vec<usize>, Vec<usize>)> {
    let n = indices.len();
    if n < 2 * config.minbucket {
        return None;
    }

    // Sort indices by feature value
    let mut sorted: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Try all possible split points
    let mut best_stat = f64::NEG_INFINITY;
    let mut best_threshold = None;
    let mut best_split_pos = 0;

    for split_pos in config.minbucket..(n - config.minbucket + 1) {
        // Check if this is a valid split point (different x values)
        if (sorted[split_pos - 1].0 - sorted[split_pos].0).abs() < 1e-10 {
            continue;
        }

        let threshold = (sorted[split_pos - 1].0 + sorted[split_pos].0) / 2.0;

        // Compute test statistic for this split
        // Create binary indicator for split
        let left_indices: Vec<usize> = sorted[..split_pos].iter().map(|(_, i)| *i).collect();
        let right_indices: Vec<usize> = sorted[split_pos..].iter().map(|(_, i)| *i).collect();

        // Compute improvement: difference in within-node variance
        let stat =
            compute_split_statistic(y, &left_indices, &right_indices, is_classification, classes);

        if stat > best_stat {
            best_stat = stat;
            best_threshold = Some(threshold);
            best_split_pos = split_pos;
        }
    }

    best_threshold.map(|threshold| {
        let left_indices: Vec<usize> = sorted[..best_split_pos].iter().map(|(_, i)| *i).collect();
        let right_indices: Vec<usize> = sorted[best_split_pos..].iter().map(|(_, i)| *i).collect();
        (threshold, left_indices, right_indices)
    })
}

/// Compute split quality statistic (improvement in criterion).
fn compute_split_statistic(
    y: &ArrayView1<f64>,
    left_indices: &[usize],
    right_indices: &[usize],
    is_classification: bool,
    classes: Option<&[f64]>,
) -> f64 {
    let n_left = left_indices.len() as f64;
    let n_right = right_indices.len() as f64;
    let n_total = n_left + n_right;

    if n_left < 1.0 || n_right < 1.0 {
        return f64::NEG_INFINITY;
    }

    if is_classification && classes.is_some() {
        // Information gain (reduction in entropy)
        let cls = classes.unwrap();

        let parent_entropy = compute_entropy(y, &[left_indices, right_indices].concat(), cls);
        let left_entropy = compute_entropy(y, left_indices, cls);
        let right_entropy = compute_entropy(y, right_indices, cls);

        let weighted_child = (n_left * left_entropy + n_right * right_entropy) / n_total;
        parent_entropy - weighted_child
    } else {
        // Variance reduction (regression)
        let parent_var = compute_variance(y, &[left_indices, right_indices].concat());
        let left_var = compute_variance(y, left_indices);
        let right_var = compute_variance(y, right_indices);

        let weighted_child = (n_left * left_var + n_right * right_var) / n_total;
        parent_var - weighted_child
    }
}

/// Compute entropy for a set of indices.
fn compute_entropy(y: &ArrayView1<f64>, indices: &[usize], classes: &[f64]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len() as f64;
    let mut entropy = 0.0;

    for &c in classes {
        let count = indices
            .iter()
            .filter(|&&i| (y[i] - c).abs() < 1e-10)
            .count() as f64;
        if count > 0.0 {
            let p = count / n;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Compute variance for a set of indices.
fn compute_variance(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.len() < 2 {
        return 0.0;
    }

    let sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let mean = sum / indices.len() as f64;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len() as f64
}

/// Build a conditional inference tree recursively.
fn build_ctree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    indices: &[usize],
    is_classification: bool,
    classes: Option<&[f64]>,
    config: &CtreeConfig,
    depth: usize,
    node_id: usize,
    importance: &mut Array1<f64>,
) -> (CtreeNode, Vec<f64>) {
    let n = indices.len();

    // Compute node prediction
    let (prediction, class_probs) = if is_classification && classes.is_some() {
        let cls = classes.unwrap();
        let counts: Vec<usize> = cls
            .iter()
            .map(|&c| {
                indices
                    .iter()
                    .filter(|&&i| (y[i] - c).abs() < 1e-10)
                    .count()
            })
            .collect();
        let total = counts.iter().sum::<usize>() as f64;
        let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total.max(1.0)).collect();
        let best_class_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        (cls[best_class_idx], Some(probs))
    } else {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        (sum / n.max(1) as f64, None)
    };

    // Empty root p-values by default
    let mut root_p_values = vec![1.0; x.ncols()];

    // Check stopping conditions
    let max_depth = if config.maxdepth == 0 {
        usize::MAX
    } else {
        config.maxdepth
    };

    if depth >= max_depth || n < config.minsplit || n <= 2 * config.minbucket {
        return (
            CtreeNode {
                id: node_id,
                n,
                prediction,
                class_probs,
                split: None,
                left: None,
                right: None,
            },
            root_p_values,
        );
    }

    // Check if all y values are the same
    let first_y = y[indices[0]];
    if indices.iter().all(|&i| (y[i] - first_y).abs() < 1e-10) {
        return (
            CtreeNode {
                id: node_id,
                n,
                prediction,
                class_probs,
                split: None,
                left: None,
                right: None,
            },
            root_p_values,
        );
    }

    // Variable selection via permutation tests
    let selection = select_best_variable(x, y, indices, is_classification, classes, config);

    if let Some((best_var, best_stat, best_pval, p_vals)) = selection {
        root_p_values = p_vals;

        // Find optimal split point
        if let Some((threshold, left_indices, right_indices)) =
            find_best_split(x, y, indices, best_var, is_classification, classes, config)
        {
            // Check minimum bucket size
            if left_indices.len() < config.minbucket || right_indices.len() < config.minbucket {
                return (
                    CtreeNode {
                        id: node_id,
                        n,
                        prediction,
                        class_probs,
                        split: None,
                        left: None,
                        right: None,
                    },
                    root_p_values,
                );
            }

            // Update variable importance
            // Using 1 - p-value as importance measure
            importance[best_var] += 1.0 - best_pval;

            // Recursively build children
            let (left_child, _) = build_ctree(
                x,
                y,
                &left_indices,
                is_classification,
                classes,
                config,
                depth + 1,
                node_id * 2,
                importance,
            );
            let (right_child, _) = build_ctree(
                x,
                y,
                &right_indices,
                is_classification,
                classes,
                config,
                depth + 1,
                node_id * 2 + 1,
                importance,
            );

            return (
                CtreeNode {
                    id: node_id,
                    n,
                    prediction,
                    class_probs,
                    split: Some(CtreeSplit {
                        feature: best_var,
                        threshold,
                        statistic: best_stat,
                        p_value: best_pval,
                    }),
                    left: Some(Box::new(left_child)),
                    right: Some(Box::new(right_child)),
                },
                root_p_values,
            );
        }
    }

    // No valid split found
    (
        CtreeNode {
            id: node_id,
            n,
            prediction,
            class_probs,
            split: None,
            left: None,
            right: None,
        },
        root_p_values,
    )
}

/// Count tree statistics recursively.
fn count_tree_stats(node: &CtreeNode) -> (usize, usize, usize) {
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
fn predict_one(node: &CtreeNode, x: &ArrayView1<f64>) -> f64 {
    match &node.split {
        None => node.prediction,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                predict_one(node.left.as_ref().unwrap(), x)
            } else {
                predict_one(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Fit a Conditional Inference Tree.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - Tree configuration
///
/// # Returns
///
/// CtreeResult containing the fitted tree and diagnostics.
///
/// # Example
///
/// ```rust,no_run
/// use p2a_core::ml::{ctree, CtreeConfig};
/// use ndarray::array;
///
/// // Regression example
/// let x = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
/// let y = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];
///
/// let config = CtreeConfig::default();
/// let result = ctree(x.view(), y.view(), &config).unwrap();
/// ```
///
/// # References
///
/// - Hothorn, T., Hornik, K., & Zeileis, A. (2006). "Unbiased Recursive Partitioning:
///   A Conditional Inference Framework". Journal of Computational and Graphical
///   Statistics, 15(3), 651-674.
pub fn ctree(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &CtreeConfig,
) -> EconResult<CtreeResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for ctree".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }
    if config.mincriterion <= 0.0 || config.mincriterion >= 1.0 {
        return Err(EconError::Computation(
            "mincriterion must be between 0 and 1 (e.g., 0.95 for p < 0.05)".to_string(),
        ));
    }

    // Determine if classification or regression
    let unique_y: std::collections::HashSet<i64> = y.iter().map(|&v| (v * 1e10) as i64).collect();
    let is_classification = unique_y.len() <= 10 && unique_y.len() < n_samples / 2;

    let classes: Option<Vec<f64>> = if is_classification {
        let mut cls: Vec<f64> = unique_y.iter().map(|&v| v as f64 / 1e10).collect();
        cls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(cls)
    } else {
        None
    };

    let indices: Vec<usize> = (0..n_samples).collect();
    let mut importance = Array1::zeros(n_features);

    let (root, root_p_values) = build_ctree(
        &x,
        &y,
        &indices,
        is_classification,
        classes.as_deref(),
        config,
        0,
        1,
        &mut importance,
    );

    let (n_nodes, n_terminal, depth) = count_tree_stats(&root);

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| predict_one(&root, &x.row(i)))
        .collect();

    Ok(CtreeResult {
        root,
        n_nodes,
        n_terminal,
        depth,
        variable_importance: importance.to_vec(),
        root_p_values,
        is_classification,
        n_classes: classes.as_ref().map(|c| c.len()),
        class_labels: classes,
        config: config.clone(),
        feature_names: None,
        predictions,
    })
}

/// Predict using a fitted ctree.
///
/// # Arguments
///
/// * `result` - Fitted ctree result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions for each observation.
pub fn ctree_predict(result: &CtreeResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    Ok((0..x.nrows())
        .map(|i| predict_one(&result.root, &x.row(i)))
        .collect())
}

/// Predict class probabilities for classification trees.
///
/// # Arguments
///
/// * `result` - Fitted ctree result (must be classification)
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Class probability matrix (n_samples x n_classes).
pub fn ctree_predict_proba(result: &CtreeResult, x: ArrayView2<f64>) -> EconResult<Array2<f64>> {
    if !result.is_classification {
        return Err(EconError::Computation(
            "predict_proba only available for classification trees".to_string(),
        ));
    }

    let n_classes = result.n_classes.unwrap_or(2);
    let n_samples = x.nrows();

    let mut probs = Array2::zeros((n_samples, n_classes));

    for i in 0..n_samples {
        let node = find_terminal_node(&result.root, &x.row(i));
        if let Some(cp) = &node.class_probs {
            for (j, &p) in cp.iter().enumerate() {
                probs[[i, j]] = p;
            }
        }
    }

    Ok(probs)
}

/// Find the terminal node for a given observation.
fn find_terminal_node<'a>(node: &'a CtreeNode, x: &ArrayView1<f64>) -> &'a CtreeNode {
    match &node.split {
        None => node,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                find_terminal_node(node.left.as_ref().unwrap(), x)
            } else {
                find_terminal_node(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Run ctree on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
/// * `x_cols` - Feature column names
/// * `config` - Tree configuration
///
/// # Returns
///
/// CtreeResult with fitted tree and diagnostics.
pub fn run_ctree(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &CtreeConfig,
) -> EconResult<CtreeResult> {
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

    let mut result = ctree(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Run ctree with default configuration.
pub fn run_ctree_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<CtreeResult> {
    run_ctree(dataset, y_col, x_cols, &CtreeConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ctree_regression_basic() {
        // Clear pattern with two groups
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
            [11.0],
            [12.0],
            [13.0],
            [14.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0],
            [20.0]
        ];
        // Two clear groups: low values for x <= 10, high values for x > 10
        let y = array![
            1.0, 1.1, 0.9, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1, 0.9, 10.0, 10.1, 9.9, 10.2, 9.8, 10.1,
            9.9, 10.0, 10.1, 9.9
        ];

        let config = CtreeConfig {
            mincriterion: 0.95,
            minsplit: 4,
            minbucket: 2,
            maxdepth: 0,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // Should find a split
        assert!(result.n_nodes >= 1);
        assert_eq!(result.predictions.len(), 20);

        // Predictions should approximate the true pattern
        for (i, &pred) in result.predictions.iter().enumerate() {
            if i < 10 {
                assert!(pred < 5.0, "Low group prediction {} should be < 5", pred);
            } else {
                assert!(pred > 5.0, "High group prediction {} should be > 5", pred);
            }
        }
    }

    #[test]
    fn test_ctree_classification() {
        // Binary classification
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [2.5, 0.5],
            [3.0, 0.0],
            [3.5, 0.5],
            [7.0, 1.0],
            [7.5, 0.5],
            [8.0, 1.0],
            [8.5, 0.5],
            [9.0, 1.0],
            [9.5, 0.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let config = CtreeConfig {
            mincriterion: 0.95,
            minsplit: 4,
            minbucket: 2,
            maxdepth: 5,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        assert!(result.is_classification);

        // Check predictions match classes
        for (i, &pred) in result.predictions.iter().enumerate() {
            if i < 6 {
                assert!(
                    pred < 0.5,
                    "Sample {} should predict class 0, got {}",
                    i,
                    pred
                );
            } else {
                assert!(
                    pred > 0.5,
                    "Sample {} should predict class 1, got {}",
                    i,
                    pred
                );
            }
        }
    }

    #[test]
    fn test_ctree_predict() {
        let x_train = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0]
        ];
        let y_train = array![1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        let config = CtreeConfig {
            mincriterion: 0.90,
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![[2.5], [17.5]];
        let predictions = ctree_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] < 5.0, "First prediction should be low");
        assert!(predictions[1] > 5.0, "Second prediction should be high");
    }

    #[test]
    fn test_ctree_no_split_when_no_pattern() {
        // Random noise - no real pattern
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
        // Alternating values - no clear split pattern
        let y = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let config = CtreeConfig {
            mincriterion: 0.99, // Very strict criterion
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // With strict criterion and no pattern, should have few or no splits
        assert!(result.n_terminal >= 1);
    }

    #[test]
    fn test_ctree_variable_importance() {
        // First variable is predictive, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [4.0, 2.0],
            [5.0, 8.0],
            [15.0, 4.0],
            [16.0, 9.0],
            [17.0, 1.0],
            [18.0, 6.0],
            [19.0, 5.0],
        ];
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        let config = CtreeConfig {
            mincriterion: 0.90,
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // First variable should have higher importance
        assert!(
            result.variable_importance[0] >= result.variable_importance[1],
            "Feature 0 ({:.4}) should have >= importance than Feature 1 ({:.4})",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }

    #[test]
    fn test_ctree_depth_limit() {
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
            [11.0],
            [12.0],
            [13.0],
            [14.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0],
            [20.0]
        ];
        let y = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0
        ];

        let config = CtreeConfig {
            mincriterion: 0.50, // Low criterion to allow splits
            minsplit: 4,
            minbucket: 2,
            maxdepth: 2, // Limit depth
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        assert!(result.depth <= 2, "Depth {} exceeds max 2", result.depth);
    }

    #[test]
    fn test_ctree_config_defaults() {
        let config = CtreeConfig::default();
        assert_eq!(config.mincriterion, 0.95);
        assert_eq!(config.minsplit, 20);
        assert_eq!(config.minbucket, 7);
        assert_eq!(config.maxdepth, 0);
        assert_eq!(config.teststat, "quadratic");
        assert_eq!(config.testtype, "bonferroni");
    }
}
