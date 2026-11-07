//! Methods outside the other families: k-prototypes for mixed numeric and categorical data ([`kprototypes`]); R analogue `clustMixType::kproto`.

use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Seed of the random starts when `seed` is `None`.
pub const KPROTOTYPES_DEFAULT_SEED: u64 = 42;

// =============================================================================
// K-Prototypes (kprototypes)
// =============================================================================

/// Result of K-Prototypes clustering.
///
/// # References
///
/// - Huang, Z. (1998). "Extensions to the k-means algorithm for clustering
///   large data sets with categorical values." Data Mining and Knowledge
///   Discovery, 2(3), 283-304.
/// - R clustMixType::kproto documentation
///   Source: <https://www.rdocumentation.org/packages/clustMixType/versions/0.4-2/topics/kproto>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPrototypesResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Numeric prototypes for each cluster (k x d_numeric)
    #[serde(skip)]
    pub numeric_prototypes: Array2<f64>,
    /// Categorical prototypes for each cluster (k x d_categorical)
    pub categorical_prototypes: Vec<Vec<usize>>,
    /// Total cost (weighted sum of numeric distance + categorical dissimilarity)
    pub total_cost: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
    /// Number of numeric features
    pub n_numeric: usize,
    /// Number of categorical features
    pub n_categorical: usize,
    /// Gamma weight for categorical features
    pub gamma: f64,
    /// Converged flag
    pub converged: bool,
}

impl std::fmt::Display for KPrototypesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-Prototypes Clustering Results")?;
        writeln!(f, "================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Numeric features: {}", self.n_numeric)?;
        writeln!(f, "Categorical features: {}", self.n_categorical)?;
        writeln!(f, "Gamma: {:.4}", self.gamma)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Total cost: {:.4}", self.total_cost)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, size)?;
        }
        Ok(())
    }
}

/// Run K-Prototypes clustering for mixed numeric and categorical data.
///
/// # Arguments
/// * `numeric_data` - Numeric features (n x d_numeric)
/// * `categorical_data` - Categorical features encoded as integers (n x d_categorical)
/// * `n_clusters` - Number of clusters
/// * `gamma` - Weight for categorical features (default: auto-computed)
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `n_init` - Number of random initializations (default: 10)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// 1. Initialize prototypes by random selection
/// 2. Assign each point to nearest prototype using mixed distance
/// 3. Update prototypes: means for numeric, modes for categorical
/// 4. Repeat until convergence
///
/// # Distance
///
/// d(x, y) = d_numeric(x, y) + gamma * d_categorical(x, y)
/// where d_numeric is squared Euclidean and d_categorical is simple matching
///
/// # Returns
/// * `KPrototypesResult` containing cluster assignments and prototypes
pub fn kprototypes(
    numeric_data: ArrayView2<f64>,
    categorical_data: &[Vec<usize>],
    n_clusters: usize,
    gamma: Option<f64>,
    max_iterations: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<KPrototypesResult, String> {
    let n = numeric_data.nrows();
    let d_num = numeric_data.ncols();

    if categorical_data.len() != n {
        return Err(format!(
            "Categorical data length ({}) must match numeric data rows ({})",
            categorical_data.len(),
            n
        ));
    }

    let d_cat = if categorical_data.is_empty() || categorical_data[0].is_empty() {
        0
    } else {
        categorical_data[0].len()
    };

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }

    let max_iter = max_iterations.unwrap_or(100);
    let n_inits = n_init.unwrap_or(10);
    if max_iter == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    if n_inits == 0 {
        return Err("n_init must be at least 1, got 0".to_string());
    }
    if let Some(g) = gamma {
        if !g.is_finite() || g < 0.0 {
            return Err(format!("gamma must be finite and at least 0, got {g}"));
        }
    }
    crate::ml::extract::require_finite(&numeric_data, "k-prototypes")?;
    if let Some((i, row)) = categorical_data
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != d_cat)
    {
        return Err(format!(
            "categorical row {i} has {} entries, row 0 has {d_cat}",
            row.len()
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(KPROTOTYPES_DEFAULT_SEED));

    // Auto-compute gamma if not provided
    // Use ratio of average numeric variance to average categorical entropy
    let auto_gamma = if gamma.is_none() && d_num > 0 && d_cat > 0 {
        let numeric_var: f64 = (0..d_num)
            .map(|j| {
                let col_mean: f64 = numeric_data.column(j).sum() / n as f64;
                numeric_data
                    .column(j)
                    .iter()
                    .map(|&x| (x - col_mean).powi(2))
                    .sum::<f64>()
                    / n as f64
            })
            .sum::<f64>()
            / d_num as f64;

        // For categorical, use 0.5 as proxy for average dissimilarity
        numeric_var / 0.5
    } else {
        gamma.unwrap_or(1.0)
    };

    let mut best_result: Option<KPrototypesResult> = None;
    let mut best_cost = f64::INFINITY;

    for _ in 0..n_inits {
        // Initialize prototypes: random selection
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut num_proto = Array2::zeros((n_clusters, d_num));
        let mut cat_proto: Vec<Vec<usize>> = vec![vec![0; d_cat]; n_clusters];

        for (c, &idx) in indices[..n_clusters].iter().enumerate() {
            num_proto.row_mut(c).assign(&numeric_data.row(idx));
            cat_proto[c] = categorical_data[idx].clone();
        }

        let mut labels = vec![0usize; n];
        let mut converged = false;
        let mut n_iterations = 0;

        for iter in 0..max_iter {
            crate::resource::checkpoint().map_err(|e| e.to_string())?;
            n_iterations = iter + 1;
            let old_labels = labels.clone();

            // Assignment step
            for i in 0..n {
                let mut min_cost = f64::INFINITY;
                let mut best_cluster = 0;

                for c in 0..n_clusters {
                    // Numeric distance (squared Euclidean)
                    let num_dist: f64 = (0..d_num)
                        .map(|j| (numeric_data[[i, j]] - num_proto[[c, j]]).powi(2))
                        .sum();

                    // Categorical distance (simple matching)
                    let cat_dist: f64 = if d_cat > 0 {
                        (0..d_cat)
                            .filter(|&j| categorical_data[i][j] != cat_proto[c][j])
                            .count() as f64
                    } else {
                        0.0
                    };

                    let cost = num_dist + auto_gamma * cat_dist;
                    if cost < min_cost {
                        min_cost = cost;
                        best_cluster = c;
                    }
                }
                labels[i] = best_cluster;
            }

            // Check convergence
            if labels == old_labels {
                converged = true;
                break;
            }

            // Update step
            let mut cluster_counts = vec![0usize; n_clusters];
            let mut num_sums: Array2<f64> = Array2::zeros((n_clusters, d_num));
            let mut cat_counts: Vec<Vec<BTreeMap<usize, usize>>> =
                vec![vec![BTreeMap::new(); d_cat]; n_clusters];

            for i in 0..n {
                let c = labels[i];
                cluster_counts[c] += 1;

                for j in 0..d_num {
                    num_sums[[c, j]] += numeric_data[[i, j]];
                }

                for j in 0..d_cat {
                    *cat_counts[c][j].entry(categorical_data[i][j]).or_insert(0) += 1;
                }
            }

            // Update numeric prototypes (means)
            for c in 0..n_clusters {
                if cluster_counts[c] > 0 {
                    for j in 0..d_num {
                        num_proto[[c, j]] = num_sums[[c, j]] / cluster_counts[c] as f64;
                    }
                }
            }

            // Update categorical prototypes (modes)
            for c in 0..n_clusters {
                if cluster_counts[c] > 0 {
                    for j in 0..d_cat {
                        // The most frequent category; a tie goes to the smallest (as
                        // clustMixType's which.max over sorted levels).
                        let mut mode: Option<(usize, usize)> = None;
                        for (&category, &count) in &cat_counts[c][j] {
                            if mode.is_none_or(|(_, best)| count > best) {
                                mode = Some((category, count));
                            }
                        }
                        if let Some((category, _)) = mode {
                            cat_proto[c][j] = category;
                        }
                    }
                }
            }

            // Handle empty clusters
            for c in 0..n_clusters {
                if cluster_counts[c] == 0 {
                    let rand_idx = rng.gen_range(0..n);
                    num_proto.row_mut(c).assign(&numeric_data.row(rand_idx));
                    cat_proto[c] = categorical_data[rand_idx].clone();
                }
            }
        }

        // Compute total cost
        let mut total_cost = 0.0;
        let mut cluster_sizes = vec![0usize; n_clusters];
        for i in 0..n {
            let c = labels[i];
            cluster_sizes[c] += 1;

            let num_dist: f64 = (0..d_num)
                .map(|j| (numeric_data[[i, j]] - num_proto[[c, j]]).powi(2))
                .sum();

            let cat_dist: f64 = if d_cat > 0 {
                (0..d_cat)
                    .filter(|&j| categorical_data[i][j] != cat_proto[c][j])
                    .count() as f64
            } else {
                0.0
            };

            total_cost += num_dist + auto_gamma * cat_dist;
        }

        if total_cost < best_cost {
            best_cost = total_cost;
            best_result = Some(KPrototypesResult {
                labels,
                numeric_prototypes: num_proto,
                categorical_prototypes: cat_proto,
                total_cost,
                n_iterations,
                cluster_sizes,
                n_clusters,
                n,
                n_numeric: d_num,
                n_categorical: d_cat,
                gamma: auto_gamma,
                converged,
            });
        }
    }

    best_result.ok_or_else(|| "K-prototypes failed".to_string())
}

/// Convenience wrapper for kprototypes.
pub fn run_kprototypes(
    numeric_data: ArrayView2<f64>,
    categorical_data: &[Vec<usize>],
    n_clusters: usize,
    gamma: Option<f64>,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KPrototypesResult, String> {
    kprototypes(
        numeric_data,
        categorical_data,
        n_clusters,
        gamma,
        max_iterations,
        None,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_kprototypes_seeded_runs_are_identical_under_mode_ties() {
        // ML-60: modes were taken in HashMap order, so identical seeded calls could differ.
        let num = Array2::from_shape_fn((8, 1), |(i, _)| {
            if i < 4 {
                0.1 * i as f64
            } else {
                5.0 + 0.1 * i as f64
            }
        });
        let cat: Vec<Vec<usize>> = [0, 1, 0, 1, 2, 3, 2, 3].iter().map(|&c| vec![c]).collect();
        let first = kprototypes(num.view(), &cat, 2, Some(0.5), None, Some(3), Some(42)).unwrap();
        for _ in 0..20 {
            let r = kprototypes(num.view(), &cat, 2, Some(0.5), None, Some(3), Some(42)).unwrap();
            assert_eq!(
                (&r.labels, &r.categorical_prototypes, r.total_cost.to_bits()),
                (
                    &first.labels,
                    &first.categorical_prototypes,
                    first.total_cost.to_bits()
                )
            );
        }
        let mut protos = first.categorical_prototypes.clone();
        protos.sort();
        assert_eq!(protos, vec![vec![0], vec![2]]); // ties go to the smallest category
    }

    #[test]
    fn test_kprototypes_rejects_ragged_and_bad_parameters() {
        let num = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        let ragged = vec![vec![0, 1], vec![0], vec![1, 1], vec![0, 0]];
        assert!(
            kprototypes(num.view(), &ragged, 2, None, None, None, None)
                .unwrap_err()
                .contains("categorical")
        );
        let cat = vec![vec![0]; 4];
        assert!(
            kprototypes(num.view(), &cat, 2, Some(f64::NAN), None, None, None)
                .unwrap_err()
                .contains("gamma")
        );
        assert!(
            kprototypes(num.view(), &cat, 2, None, Some(0), None, None)
                .unwrap_err()
                .contains("max_iterations")
        );
        let a = kprototypes(num.view(), &cat, 2, None, None, None, None).unwrap();
        let b = kprototypes(
            num.view(),
            &cat,
            2,
            None,
            None,
            None,
            Some(KPROTOTYPES_DEFAULT_SEED),
        )
        .unwrap();
        assert_eq!(a.total_cost.to_bits(), b.total_cost.to_bits());
    }

    #[test]
    fn test_kprototypes_basic() {
        // Numeric features
        let numeric_data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        // Categorical features (e.g., encoded categories)
        let categorical_data: Vec<Vec<usize>> =
            vec![vec![0, 0], vec![0, 0], vec![1, 1], vec![1, 1]];

        let result = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            None, // auto gamma
            Some(100),
            Some(10),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n, 4);
        assert_eq!(result.n_numeric, 2);
        assert_eq!(result.n_categorical, 2);
        assert_eq!(result.labels.len(), 4);

        // First two points should be in same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        // Last two points should be in same cluster
        assert_eq!(result.labels[2], result.labels[3]);
        // The two groups should be different
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_kprototypes_numeric_only() {
        let numeric_data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        // Empty categorical features
        let categorical_data: Vec<Vec<usize>> = vec![vec![], vec![], vec![], vec![]];

        let result = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(1.0),
            Some(50),
            Some(5),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_categorical, 0);
    }

    #[test]
    fn test_kprototypes_gamma_effect() {
        let numeric_data = array![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0],];

        let categorical_data: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![0], vec![1]];

        // Low gamma: numeric features dominate
        let result_low = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(0.1),
            Some(50),
            Some(5),
            Some(42),
        )
        .unwrap();

        // High gamma: categorical features dominate
        let result_high = kprototypes(
            numeric_data.view(),
            &categorical_data,
            2,
            Some(100.0),
            Some(50),
            Some(5),
            Some(42),
        )
        .unwrap();

        // Both should produce valid results
        assert_eq!(result_low.n_clusters, 2);
        assert_eq!(result_high.n_clusters, 2);
    }
}
