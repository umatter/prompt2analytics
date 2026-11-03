//! OPTICS ordering and cluster extraction ([`optics`]); R analogue `dbscan::optics`.

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::common::{
    ClusteringAlgorithm, compute_distance_matrix, compute_distance_matrix_parallel,
    select_algorithm,
};
use crate::ml::kdtree::KdTree;

// =============================================================================
// OPTICS (Ordering Points To Identify the Clustering Structure)
// =============================================================================

/// Result of OPTICS clustering.
///
/// # References
///
/// - Ankerst, M., Breunig, M.M., Kriegel, H.P., and Sander, J. (1999).
///   "OPTICS: Ordering Points To Identify the Clustering Structure".
///   ACM SIGMOD 1999.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticsResult {
    /// Ordering of points (indices in processing order)
    pub ordering: Vec<usize>,
    /// Reachability distances for each point (in ordering order)
    pub reachability: Vec<f64>,
    /// Core distances for each point
    pub core_distances: Vec<f64>,
    /// Cluster labels (-1 for noise) extracted at eps_prime
    pub labels: Vec<i32>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// MinPts parameter used
    pub min_samples: usize,
    /// Xi parameter used for cluster extraction (if any)
    pub xi: Option<f64>,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for OpticsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OPTICS Clustering Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of noise points: {}", self.n_noise)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "MinPts: {}", self.min_samples)?;
        if let Some(xi) = self.xi {
            writeln!(f, "Xi: {:.4}", xi)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Reachability plot data available in 'ordering' and 'reachability' fields"
        )?;

        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
        for &label in &self.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        let mut labels_sorted: Vec<_> = cluster_counts.keys().cloned().collect();
        labels_sorted.sort();
        for label in labels_sorted {
            if label == -1 {
                writeln!(f, "  Noise: {} points", cluster_counts[&label])?;
            } else {
                writeln!(f, "  Cluster {}: {} points", label, cluster_counts[&label])?;
            }
        }
        Ok(())
    }
}

/// Run OPTICS clustering.
///
/// OPTICS creates an ordering of points such that spatially closest points
/// become neighbors in the ordering. It produces a reachability plot that
/// can be used to extract clusters at various density levels.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `min_samples` - Minimum points for core point definition
/// * `max_eps` - Maximum epsilon neighborhood (default: infinity)
/// * `xi` - Optional steepness parameter for automatic cluster extraction
///
/// # Algorithm
///
/// 1. For each point, compute core-distance (distance to min_samples-th nearest neighbor)
/// 2. Process points in order of reachability-distance
/// 3. Update reachability-distances of unprocessed neighbors
/// 4. Extract clusters from reachability plot
///
/// # Performance
///
/// Uses KD-tree acceleration when beneficial:
/// - n < 500: O(n²) brute-force
/// - n >= 500 && d <= 15: O(n log n) core distances + O(n log n) range queries
/// - d > 15: O(n²) parallel brute-force
///
/// # Returns
/// * `OpticsResult` containing ordering and reachability distances
///
/// # References
///
/// - Ankerst et al. (1999). "OPTICS: Ordering Points To Identify the Clustering Structure".
pub fn optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    if min_samples < 1 || min_samples > n {
        return Err(format!("min_samples must be between 1 and {}", n));
    }

    let eps = max_eps.unwrap_or(f64::INFINITY);

    // Select algorithm and compute core distances
    let algorithm = select_algorithm(n, d);
    let use_kdtree = matches!(
        algorithm,
        ClusteringAlgorithm::KdTreePrim | ClusteringAlgorithm::DualTreeBoruvka
    );

    let (distances, core_distances, tree_opt) = if use_kdtree {
        let data_vecs: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
        let tree = KdTree::new(data_vecs);
        let core_dists = tree
            .compute_core_distances(min_samples)
            .into_iter()
            .map(|cd| cd.min(eps))
            .collect::<Vec<_>>();
        (None, core_dists, Some(tree))
    } else {
        // Use brute-force distance matrix
        let distances = if matches!(algorithm, ClusteringAlgorithm::BruteForceParallel) {
            compute_distance_matrix_parallel(&data)
        } else {
            compute_distance_matrix(&data)
        };

        let core_distances: Vec<f64> = (0..n)
            .map(|i| {
                let mut dists: Vec<f64> = (0..n)
                    .filter(|&j| i != j)
                    .map(|j| distances[[i, j]])
                    .collect();

                if dists.len() >= min_samples {
                    dists.select_nth_unstable_by(min_samples - 1, |a, b| {
                        a.partial_cmp(b).unwrap_or(Ordering::Equal)
                    });
                    dists[min_samples - 1].min(eps)
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        (Some(distances), core_distances, None)
    };

    // OPTICS ordering
    let mut processed = vec![false; n];
    let mut ordering = Vec::with_capacity(n);
    let mut reachability = vec![f64::INFINITY; n];

    // Priority queue: (reachability, index) - min-heap based on reachability
    #[derive(PartialEq)]
    struct OrderedPoint {
        reachability: f64,
        index: usize,
    }

    impl Eq for OrderedPoint {}

    impl Ord for OrderedPoint {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap
            other
                .reachability
                .partial_cmp(&self.reachability)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for OrderedPoint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // Helper closure to get neighbors
    let get_neighbors = |point_idx: usize, radius: f64| -> Vec<(f64, usize)> {
        if let Some(ref tree) = tree_opt {
            match tree.data().get(point_idx) {
                Some(point) => tree.radius_query(point, radius, Some(point_idx)),
                None => Vec::new(),
            }
        } else if let Some(ref dist_matrix) = distances {
            (0..n)
                .filter(|&j| j != point_idx && dist_matrix[[point_idx, j]] <= radius)
                .map(|j| (dist_matrix[[point_idx, j]], j))
                .collect()
        } else {
            Vec::new()
        }
    };

    // Process all points
    for start in 0..n {
        if processed[start] {
            continue;
        }

        // Start a new cluster from this point
        let mut seeds = BinaryHeap::new();
        processed[start] = true;
        ordering.push(start);

        // Add neighbors of start to seeds
        if core_distances[start] <= eps {
            let neighbors = get_neighbors(start, eps);
            for (dist, j) in neighbors {
                if !processed[j] {
                    let new_reach = core_distances[start].max(dist);
                    if new_reach < reachability[j] {
                        reachability[j] = new_reach;
                        seeds.push(OrderedPoint {
                            reachability: new_reach,
                            index: j,
                        });
                    }
                }
            }
        }

        // Process seeds
        while let Some(OrderedPoint {
            index: p,
            reachability: _,
        }) = seeds.pop()
        {
            if processed[p] {
                continue;
            }

            processed[p] = true;
            ordering.push(p);

            // Update neighbors if p is a core point
            if core_distances[p] <= eps {
                let neighbors = get_neighbors(p, eps);
                for (dist, j) in neighbors {
                    if !processed[j] {
                        let new_reach = core_distances[p].max(dist);
                        if new_reach < reachability[j] {
                            reachability[j] = new_reach;
                            seeds.push(OrderedPoint {
                                reachability: new_reach,
                                index: j,
                            });
                        }
                    }
                }
            }
        }
    }

    // Extract clusters using xi (steepness-based) or simple threshold
    let labels = if let Some(xi_val) = xi {
        extract_clusters_xi(&ordering, &reachability, xi_val)
    } else {
        // Use median reachability as threshold
        let mut sorted_reach: Vec<f64> = reachability
            .iter()
            .filter(|&&r| r.is_finite())
            .cloned()
            .collect();
        sorted_reach.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let threshold = if sorted_reach.is_empty() {
            f64::INFINITY
        } else {
            sorted_reach[sorted_reach.len() / 2] * 1.5
        };
        extract_clusters_threshold(&ordering, &reachability, threshold)
    };

    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    // Reorder reachability to match ordering
    let reachability_ordered: Vec<f64> = ordering.iter().map(|&i| reachability[i]).collect();

    Ok(OpticsResult {
        ordering,
        reachability: reachability_ordered,
        core_distances,
        labels,
        n_clusters,
        n_noise,
        min_samples,
        xi,
        n,
    })
}

/// Extract clusters using xi (steepness) method.
fn extract_clusters_xi(ordering: &[usize], reachability: &[f64], xi: f64) -> Vec<i32> {
    let n = ordering.len();
    let mut labels = vec![-1i32; n];

    if n == 0 {
        return labels;
    }

    // Find steep down and up areas
    let mut current_cluster = 0;
    let mut in_cluster = false;

    for i in 1..ordering.len() {
        let prev_idx = ordering[i - 1];
        let curr_idx = ordering[i];
        let prev_r = reachability[prev_idx];
        let curr_r = reachability[curr_idx];

        // Steep down: start of cluster
        if prev_r.is_finite() && curr_r.is_finite() {
            if curr_r < prev_r * (1.0 - xi) {
                // Steep down area - potential cluster start
                if !in_cluster {
                    in_cluster = true;
                }
            } else if curr_r > prev_r / (1.0 - xi) {
                // Steep up area - potential cluster end
                if in_cluster {
                    current_cluster += 1;
                    in_cluster = false;
                }
            }
        }

        if in_cluster {
            labels[curr_idx] = current_cluster;
        }
    }

    labels
}

/// Extract clusters using simple reachability threshold.
fn extract_clusters_threshold(
    ordering: &[usize],
    reachability: &[f64],
    threshold: f64,
) -> Vec<i32> {
    let n = ordering.len();
    let mut labels = vec![-1i32; n];
    let mut current_cluster = -1i32;

    for &idx in ordering {
        let r = reachability[idx];

        if r > threshold {
            // Noise or boundary
            if current_cluster >= 0 {
                current_cluster += 1;
            }
            labels[idx] = -1;
        } else {
            // Part of cluster
            if current_cluster < 0 {
                current_cluster = 0;
            }
            labels[idx] = current_cluster;
        }
    }

    labels
}

/// Convenience wrapper for optics.
pub fn run_optics(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
    xi: Option<f64>,
) -> Result<OpticsResult, String> {
    optics(data, min_samples, max_eps, xi)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_optics_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = optics(data.view(), 2, None, None).unwrap();

        assert_eq!(result.ordering.len(), 6);
        assert_eq!(result.reachability.len(), 6);
    }
}
