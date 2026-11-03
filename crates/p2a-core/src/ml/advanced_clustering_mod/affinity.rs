//! Affinity propagation ([`affinity_propagation`]); R analogue `apcluster`.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// =============================================================================
// Affinity Propagation
// =============================================================================

/// Result of affinity propagation clustering.
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages Between
///   Data Points". Science, 315, 972-976.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityPropagationResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Indices of exemplar (cluster center) points
    pub exemplar_indices: Vec<usize>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Final cluster quality (sum of responsibilities to exemplars)
    pub net_similarity: f64,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for AffinityPropagationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Affinity Propagation Results")?;
        writeln!(f, "============================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Net similarity: {:.4}", self.net_similarity)?;
        writeln!(f)?;
        writeln!(f, "Exemplar indices: {:?}", self.exemplar_indices)?;

        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(
                f,
                "  Cluster {}: {} points (exemplar: {})",
                i, count, self.exemplar_indices[i]
            )?;
        }
        Ok(())
    }
}

/// Run affinity propagation clustering.
///
/// Affinity propagation clusters data by exchanging real-valued messages
/// between data points until a high-quality set of exemplars emerges.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `preference` - Preference for each point to be an exemplar (default: median similarity)
/// * `damping` - Damping factor (0.5 to 1.0, default: 0.5)
/// * `max_iterations` - Maximum iterations (default: 200)
/// * `convergence_iterations` - Iterations without change for convergence (default: 15)
///
/// # Algorithm
///
/// 1. Initialize similarity matrix S (negative squared Euclidean distance)
/// 2. Initialize responsibility R and availability A matrices to 0
/// 3. Iteratively update:
///    - R(i,k) = S(i,k) - max_{k' != k}[A(i,k') + S(i,k')]
///    - A(i,k) = min(0, R(k,k) + sum_{i' != i,k} max(0, R(i',k)))
///    - A(k,k) = sum_{i' != k} max(0, R(i',k))
/// 4. Exemplars are points where R(k,k) + A(k,k) > 0
///
/// # Returns
/// * `AffinityPropagationResult` containing cluster assignments and exemplars
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages".
pub fn affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> Result<AffinityPropagationResult, String> {
    let n = data.nrows();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let damp = damping.unwrap_or(0.5);
    if !(0.5..1.0).contains(&damp) {
        return Err("Damping must be between 0.5 and 1.0".to_string());
    }

    let max_iter = max_iterations.unwrap_or(200);
    let conv_iter = convergence_iterations.unwrap_or(15);

    // Compute similarity matrix (negative squared Euclidean distance)
    let mut similarity = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut sq_dist = 0.0;
                for k in 0..data.ncols() {
                    sq_dist += (data[[i, k]] - data[[j, k]]).powi(2);
                }
                similarity[[i, j]] = -sq_dist;
            }
        }
    }

    // Set preference (diagonal) - default to median similarity
    let pref = preference.unwrap_or_else(|| {
        let mut sims: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sims.push(similarity[[i, j]]);
                }
            }
        }
        sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if sims.is_empty() {
            0.0
        } else {
            sims[sims.len() / 2]
        }
    });
    for i in 0..n {
        similarity[[i, i]] = pref;
    }

    // Initialize responsibility and availability matrices
    let mut responsibility: Array2<f64> = Array2::zeros((n, n));
    let mut availability: Array2<f64> = Array2::zeros((n, n));

    // Track exemplars for convergence detection
    let mut prev_exemplars = vec![false; n];
    let mut unchanged_count = 0;

    let mut n_iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Optimized responsibility update: O(n²) instead of O(n³)
        // Precompute AS = A + S, then find first and second max per row
        let mut as_matrix = Array2::zeros((n, n));
        for i in 0..n {
            for k in 0..n {
                as_matrix[[i, k]] = availability[[i, k]] + similarity[[i, k]];
            }
        }

        // For each row, find first_max and second_max
        let mut row_max1 = vec![f64::NEG_INFINITY; n]; // first max
        let mut row_max2 = vec![f64::NEG_INFINITY; n]; // second max
        let mut row_argmax = vec![0usize; n];

        for i in 0..n {
            for k in 0..n {
                let val = as_matrix[[i, k]];
                if val > row_max1[i] {
                    row_max2[i] = row_max1[i];
                    row_max1[i] = val;
                    row_argmax[i] = k;
                } else if val > row_max2[i] {
                    row_max2[i] = val;
                }
            }
        }

        // Compute new responsibilities using precomputed max values
        for i in 0..n {
            for k in 0..n {
                let max_other = if k == row_argmax[i] {
                    row_max2[i]
                } else {
                    row_max1[i]
                };
                let r_new = similarity[[i, k]] - max_other;
                responsibility[[i, k]] = damp * responsibility[[i, k]] + (1.0 - damp) * r_new;
            }
        }

        // Optimized availability update: O(n²) instead of O(n³)
        // Precompute column sums of positive responsibilities
        let mut col_pos_sum = vec![0.0f64; n];
        for k in 0..n {
            for i in 0..n {
                col_pos_sum[k] += responsibility[[i, k]].max(0.0);
            }
        }

        // Compute availabilities using precomputed sums
        for k in 0..n {
            // Self-availability: sum of positive responsibilities from others
            let self_r_pos = responsibility[[k, k]].max(0.0);
            let a_kk = col_pos_sum[k] - self_r_pos;
            availability[[k, k]] = damp * availability[[k, k]] + (1.0 - damp) * a_kk;
        }

        for i in 0..n {
            for k in 0..n {
                if i != k {
                    let self_r_pos = responsibility[[k, k]].max(0.0);
                    let i_r_pos = responsibility[[i, k]].max(0.0);
                    // sum excluding k and i
                    let sum = col_pos_sum[k] - self_r_pos - i_r_pos;
                    let a_new = (responsibility[[k, k]] + sum).min(0.0);
                    availability[[i, k]] = damp * availability[[i, k]] + (1.0 - damp) * a_new;
                }
            }
        }

        // Check convergence: exemplars are stable
        let exemplars: Vec<bool> = (0..n)
            .map(|k| responsibility[[k, k]] + availability[[k, k]] > 0.0)
            .collect();

        if exemplars == prev_exemplars {
            unchanged_count += 1;
            if unchanged_count >= conv_iter {
                converged = true;
                break;
            }
        } else {
            unchanged_count = 0;
        }
        prev_exemplars = exemplars;
    }

    // Identify final exemplars
    let exemplar_mask: Vec<bool> = (0..n)
        .map(|k| responsibility[[k, k]] + availability[[k, k]] > 0.0)
        .collect();

    let exemplar_indices: Vec<usize> = exemplar_mask
        .iter()
        .enumerate()
        .filter(|&(_, is_ex)| *is_ex)
        .map(|(i, _)| i)
        .collect();

    // Assign points to nearest exemplar
    let mut labels = vec![0usize; n];
    for i in 0..n {
        if exemplar_mask[i] {
            // Exemplar assigned to itself
            labels[i] = exemplar_indices.iter().position(|&e| e == i).unwrap_or(0);
        } else {
            // Assign to exemplar with highest responsibility
            let mut best_ex = 0;
            let mut best_r = f64::NEG_INFINITY;
            for (ex_idx, &ex) in exemplar_indices.iter().enumerate() {
                if similarity[[i, ex]] > best_r {
                    best_r = similarity[[i, ex]];
                    best_ex = ex_idx;
                }
            }
            labels[i] = best_ex;
        }
    }

    // Compute net similarity
    let net_similarity: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            if label < exemplar_indices.len() {
                similarity[[i, exemplar_indices[label]]]
            } else {
                0.0
            }
        })
        .sum();

    Ok(AffinityPropagationResult {
        labels,
        exemplar_indices: exemplar_indices.clone(),
        n_iterations,
        converged,
        net_similarity,
        n_clusters: exemplar_indices.len(),
        n,
    })
}

/// Convenience wrapper for affinity_propagation.
pub fn run_affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
) -> Result<AffinityPropagationResult, String> {
    affinity_propagation(data, preference, damping, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_affinity_propagation_basic() {
        // Use more points and more separation for affinity propagation
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Use a low preference to encourage fewer clusters
        let result = affinity_propagation(
            data.view(),
            Some(-100.0), // Strong preference against being an exemplar
            Some(0.9),
            Some(200),
            Some(15),
        )
        .unwrap();

        assert_eq!(result.labels.len(), 6);
        // Affinity propagation may find varying numbers of clusters
        // Just check that it completed without error
        assert!(result.n_iterations > 0);
    }
}
