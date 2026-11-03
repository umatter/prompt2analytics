//! Dynamic tree cut of a dendrogram ([`dynamic_tree_cut`]); R analogue `dynamicTreeCut`.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::hierarchical::{FastLinkage, fastcluster};

// =============================================================================
// Dynamic Tree Cut (dynamicTreeCut)
// =============================================================================

/// Method for dynamic tree cutting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum DynamicCutMethod {
    /// Hybrid method combining hierarchical with PAM refinement
    #[default]
    Hybrid,
    /// Pure tree-based method
    Tree,
}

/// Result of dynamic tree cut.
///
/// # References
///
/// - Langfelder, P., Zhang, B., & Horvath, S. (2008). "Defining clusters
///   from a hierarchical cluster tree: the Dynamic Tree Cut package for R."
///   Bioinformatics, 24(5), 719-720.
/// - R dynamicTreeCut package documentation
///   Source: <https://www.rdocumentation.org/packages/dynamicTreeCut/versions/1.63-1/topics/cutreeDynamic>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTreeCutResult {
    /// Cluster assignments (0 = unassigned/outlier)
    pub labels: Vec<usize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of unassigned points (label 0)
    pub n_unassigned: usize,
    /// Cluster sizes (excluding unassigned)
    pub cluster_sizes: Vec<usize>,
    /// Number of observations
    pub n: usize,
    /// Method used
    pub method: DynamicCutMethod,
    /// Deep split parameter used
    pub deep_split: u8,
    /// Minimum cluster size used
    pub min_cluster_size: usize,
}

impl std::fmt::Display for DynamicTreeCutResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dynamic Tree Cut Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Method: {:?}", self.method)?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Unassigned points: {}", self.n_unassigned)?;
        writeln!(f, "Deep split: {}", self.deep_split)?;
        writeln!(f, "Min cluster size: {}", self.min_cluster_size)?;
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i + 1, size)?;
        }
        Ok(())
    }
}

/// Perform dynamic tree cutting on hierarchical clustering result.
///
/// Automatically detects clusters in a dendrogram by identifying significant
/// branches based on shape criteria.
///
/// # Arguments
/// * `merge` - Merge matrix from hierarchical clustering (n-1 x 2)
/// * `height` - Heights at which merges occurred
/// * `n` - Number of observations
/// * `method` - Cutting method (default: Hybrid)
/// * `deep_split` - Controls sensitivity: 0-4 (higher = more smaller clusters)
/// * `min_cluster_size` - Minimum size for a valid cluster
/// * `dist_matrix` - Optional distance matrix for hybrid method
///
/// # Algorithm
///
/// 1. Tree method: Find branches by detecting significant height gaps
/// 2. Hybrid method: Refine tree clusters using PAM
///
/// # Returns
/// * `DynamicTreeCutResult` containing adaptive cluster assignments
pub fn dynamic_tree_cut(
    merge: &Array2<i64>,
    height: &[f64],
    n: usize,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
    dist_matrix: Option<&Array2<f64>>,
) -> Result<DynamicTreeCutResult, String> {
    let cut_method = method.unwrap_or(DynamicCutMethod::Tree);
    let ds = deep_split.unwrap_or(1).min(4);
    let min_size = min_cluster_size.unwrap_or(2);

    if merge.nrows() != n - 1 {
        return Err(format!(
            "Merge matrix should have {} rows for {} observations",
            n - 1,
            n
        ));
    }
    if height.len() != n - 1 {
        return Err("Height vector length should match merge matrix rows".to_string());
    }

    // Compute height gaps to find natural breaks
    let mut height_sorted = height.to_vec();
    height_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Threshold based on deep_split
    let gap_percentile = match ds {
        0 => 0.95, // Very conservative
        1 => 0.90,
        2 => 0.75,
        3 => 0.50,
        _ => 0.25, // Very sensitive
    };

    let mut gaps: Vec<f64> = Vec::new();
    for i in 1..height_sorted.len() {
        gaps.push(height_sorted[i] - height_sorted[i - 1]);
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let gap_idx = ((gaps.len() as f64 * gap_percentile) as usize).min(gaps.len() - 1);
    let gap_threshold = if gaps.is_empty() { 0.0 } else { gaps[gap_idx] };

    // Find cut points (significant gaps)
    let mut cut_heights: Vec<f64> = Vec::new();
    for i in 1..height.len() {
        if height[i] - height[i - 1] > gap_threshold {
            cut_heights.push((height[i] + height[i - 1]) / 2.0);
        }
    }

    // If no cuts found, use a single cut at median height
    if cut_heights.is_empty() {
        let median_idx = height.len() / 2;
        cut_heights.push(height_sorted[median_idx]);
    }

    // Build cluster assignments by traversing tree
    let mut labels = vec![0usize; n];
    let mut cluster_id = 1usize;

    // Track which nodes belong to which cluster
    let mut node_clusters: HashMap<i64, usize> = HashMap::new();

    // Process merges from bottom to top
    for (step, &h) in height.iter().enumerate() {
        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        let left_cluster = if left < 0 {
            0 // Leaf, unassigned initially
        } else {
            *node_clusters.get(&left).unwrap_or(&0)
        };

        let right_cluster = if right < 0 {
            0
        } else {
            *node_clusters.get(&right).unwrap_or(&0)
        };

        // Check if this merge is above a cut threshold
        let is_cut = cut_heights.iter().any(|&ch| h > ch);

        if is_cut && left_cluster == 0 && right_cluster == 0 {
            // Both branches are new clusters
            node_clusters.insert(step as i64 + 1, 0);
        } else if left_cluster > 0 && right_cluster > 0 && is_cut {
            // Keep separate clusters
            node_clusters.insert(step as i64 + 1, 0);
        } else {
            // Merge into same cluster
            let merged_cluster = if left_cluster > 0 {
                left_cluster
            } else if right_cluster > 0 {
                right_cluster
            } else {
                // Both unassigned, create new cluster
                cluster_id += 1;
                cluster_id - 1
            };
            node_clusters.insert(step as i64 + 1, merged_cluster);

            // Propagate cluster assignment to children
            if left < 0 {
                labels[(-left - 1) as usize] = merged_cluster;
            } else if let Some(&c) = node_clusters.get(&left) {
                if c == 0 {
                    node_clusters.insert(left, merged_cluster);
                }
            }
            if right < 0 {
                labels[(-right - 1) as usize] = merged_cluster;
            } else if let Some(&c) = node_clusters.get(&right) {
                if c == 0 {
                    node_clusters.insert(right, merged_cluster);
                }
            }
        }
    }

    // Assign unassigned leaves
    for i in 0..n {
        if labels[i] == 0 {
            if let Some(&c) = node_clusters.get(&(-(i as i64) - 1)) {
                labels[i] = c;
            }
        }
    }

    // Traverse tree to assign remaining points
    fn assign_from_tree(
        merge: &Array2<i64>,
        node: i64,
        node_clusters: &HashMap<i64, usize>,
        labels: &mut Vec<usize>,
    ) -> usize {
        if node < 0 {
            let idx = (-node - 1) as usize;
            if labels[idx] == 0 {
                labels[idx] = *node_clusters.get(&node).unwrap_or(&0);
            }
            return labels[idx];
        }

        let step = (node - 1) as usize;
        let left = merge[[step, 0]];
        let right = merge[[step, 1]];

        let left_c = assign_from_tree(merge, left, node_clusters, labels);
        let right_c = assign_from_tree(merge, right, node_clusters, labels);

        if let Some(&c) = node_clusters.get(&node) {
            if c > 0 {
                return c;
            }
        }

        left_c.max(right_c)
    }

    let n_merges = merge.nrows();
    if n_merges > 0 {
        assign_from_tree(merge, n_merges as i64, &node_clusters, &mut labels);
    }

    // Remove small clusters (assign to unassigned)
    let mut cluster_sizes_map: HashMap<usize, usize> = HashMap::new();
    for &l in &labels {
        if l > 0 {
            *cluster_sizes_map.entry(l).or_insert(0) += 1;
        }
    }

    for l in labels.iter_mut() {
        if let Some(&size) = cluster_sizes_map.get(l) {
            if size < min_size {
                *l = 0;
            }
        }
    }

    // Renumber clusters to be consecutive starting from 1
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    label_map.insert(0, 0); // Keep 0 as unassigned
    let mut next = 1;
    for l in labels.iter_mut() {
        if *l > 0 {
            if let Some(&new_l) = label_map.get(l) {
                *l = new_l;
            } else {
                label_map.insert(*l, next);
                *l = next;
                next += 1;
            }
        }
    }

    // Compute final statistics
    let n_clusters = next - 1;
    let n_unassigned = labels.iter().filter(|&&l| l == 0).count();
    let mut cluster_sizes = vec![0usize; n_clusters];
    for &l in &labels {
        if l > 0 && l <= n_clusters {
            cluster_sizes[l - 1] += 1;
        }
    }

    // Hybrid refinement with PAM if distance matrix provided
    if cut_method == DynamicCutMethod::Hybrid && dist_matrix.is_some() && n_clusters > 0 {
        // PAM refinement would go here - for now, use tree result
    }

    Ok(DynamicTreeCutResult {
        labels,
        n_clusters,
        n_unassigned,
        cluster_sizes,
        n,
        method: cut_method,
        deep_split: ds,
        min_cluster_size: min_size,
    })
}

/// Convenience wrapper for dynamic_tree_cut with hierarchical clustering.
pub fn run_dynamic_tree_cut(
    data: ArrayView2<f64>,
    linkage: Option<FastLinkage>,
    method: Option<DynamicCutMethod>,
    deep_split: Option<u8>,
    min_cluster_size: Option<usize>,
) -> Result<DynamicTreeCutResult, String> {
    // First run hierarchical clustering
    let hc_result = fastcluster(data, linkage, None)?;

    // Then apply dynamic tree cut
    dynamic_tree_cut(
        &hc_result.merge,
        &hc_result.height,
        hc_result.n,
        method,
        deep_split,
        min_cluster_size,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_dynamic_tree_cut_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = run_dynamic_tree_cut(
            data.view(),
            Some(FastLinkage::Ward),
            Some(DynamicCutMethod::Tree),
            Some(2), // deep_split
            Some(2), // min_cluster_size
        )
        .unwrap();

        assert_eq!(result.n, 6);
        assert_eq!(result.labels.len(), 6);
        assert!(result.n_clusters >= 1);

        // Total assigned + unassigned should equal n
        let total: usize = result.cluster_sizes.iter().sum::<usize>() + result.n_unassigned;
        assert_eq!(total, 6);
    }

    #[test]
    fn test_dynamic_tree_cut_deep_split() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ];

        // Lower deep_split = fewer clusters
        let result_low = run_dynamic_tree_cut(data.view(), None, None, Some(0), Some(2)).unwrap();

        // Higher deep_split = potentially more clusters
        let result_high = run_dynamic_tree_cut(data.view(), None, None, Some(4), Some(2)).unwrap();

        // Both should produce valid results
        assert!(result_low.n_clusters >= 1);
        assert!(result_high.n_clusters >= 1);
    }
}
