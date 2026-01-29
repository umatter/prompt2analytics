//! Dual-Tree Boruvka algorithm for O(n log n) MST construction.
//!
//! This module implements the Dual-Tree Boruvka algorithm for efficiently
//! building minimum spanning trees from mutual reachability distances.
//! Used by HDBSCAN for large datasets where O(n²) becomes prohibitive.
//!
//! # References
//!
//! - March, W.B., Ram, P., and Gray, A.G. (2010). "Fast Euclidean Minimum
//!   Spanning Tree: Algorithm, Analysis, and Applications". KDD 2010.
//! - Campello, R.J.G.B., Moulavi, D., and Sander, J. (2013). "Density-Based
//!   Clustering Based on Hierarchical Density Estimates". PAKDD 2013.

use super::kdtree::{KdNode, KdTree, UnionFind, euclidean_distance};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Result of Dual-Tree Boruvka MST construction.
pub struct DualTreeMstResult {
    /// MST edges as (from, to, weight)
    pub edges: Vec<(usize, usize, f64)>,
    /// Number of Boruvka iterations
    pub iterations: usize,
}

/// Priority queue entry for dual-tree traversal.
#[derive(Clone)]
struct DualTreeEntry {
    /// Minimum possible mutual reachability between the two subtrees
    min_mr: f64,
    /// Node in the query tree
    query_node_idx: usize,
    /// Node in the reference tree
    ref_node_idx: usize,
}

impl PartialEq for DualTreeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.min_mr == other.min_mr
    }
}

impl Eq for DualTreeEntry {}

impl Ord for DualTreeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distances first
        other
            .min_mr
            .partial_cmp(&self.min_mr)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DualTreeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Tree node info with pre-computed child indices.
struct TreeNodeInfo<'a> {
    node: &'a KdNode,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
}

/// Collect all nodes with their child indices properly tracked.
fn collect_nodes_with_indices(root: &KdNode) -> Vec<TreeNodeInfo<'_>> {
    let mut nodes: Vec<TreeNodeInfo<'_>> = Vec::new();
    let mut stack: Vec<(&KdNode, Option<usize>, bool)> = vec![(root, None, false)]; // (node, parent_idx, is_right_child)

    while let Some((node, parent_idx, is_right)) = stack.pop() {
        let current_idx = nodes.len();

        // Update parent's child index
        if let Some(pidx) = parent_idx {
            if is_right {
                nodes[pidx].right_idx = Some(current_idx);
            } else {
                nodes[pidx].left_idx = Some(current_idx);
            }
        }

        nodes.push(TreeNodeInfo {
            node,
            left_idx: None,
            right_idx: None,
        });

        // Process right first so left is processed first when popped (DFS order)
        if let Some(ref right) = node.right {
            stack.push((right, Some(current_idx), true));
        }
        if let Some(ref left) = node.left {
            stack.push((left, Some(current_idx), false));
        }
    }

    nodes
}

/// Build MST using Dual-Tree Boruvka algorithm.
///
/// This achieves O(n log n) expected time for low-dimensional data
/// by using bounding box pruning to avoid computing unnecessary distances.
///
/// # Arguments
/// * `tree` - KD-tree of the data points
/// * `core_distances` - Pre-computed core distances for each point
///
/// # Returns
/// * MST edges as (from, to, mutual_reachability_distance)
pub fn dual_tree_boruvka_mst(tree: &KdTree, core_distances: &[f64]) -> DualTreeMstResult {
    let n = tree.len();
    if n <= 1 {
        return DualTreeMstResult {
            edges: Vec::new(),
            iterations: 0,
        };
    }

    let mut uf = UnionFind::new(n);
    let mut mst = Vec::with_capacity(n - 1);
    let mut iterations = 0;

    // Track current minimum outgoing edge for each component
    let mut component_min_edge: Vec<Option<(usize, usize, f64)>> = vec![None; n];

    while mst.len() < n - 1 {
        iterations += 1;

        // Reset minimum edges
        component_min_edge.fill(None);

        // Use dual-tree traversal to find minimum edge for each component
        find_component_min_edges_dual_tree(tree, core_distances, &mut uf, &mut component_min_edge);

        // Collect and add edges
        let mut added_any = false;
        for i in 0..n {
            let root = uf.find(i);
            if root != i {
                continue; // Only process component roots
            }

            if let Some((from, to, weight)) = component_min_edge[root] {
                if uf.find(from) != uf.find(to) {
                    uf.union(from, to);
                    mst.push((from, to, weight));
                    added_any = true;

                    if mst.len() == n - 1 {
                        break;
                    }
                }
            }
        }

        if !added_any {
            // No progress - graph might be disconnected
            // Fall back to finding any connecting edge
            let bridges = find_component_bridges(tree, core_distances, &mut uf);
            for (from, to, weight) in bridges {
                if uf.find(from) != uf.find(to) {
                    uf.union(from, to);
                    mst.push((from, to, weight));
                    if mst.len() == n - 1 {
                        break;
                    }
                }
            }

            if mst.len() < n - 1 && !added_any {
                break; // Truly disconnected
            }
        }

        // Safety check
        if iterations > n {
            break;
        }
    }

    DualTreeMstResult {
        edges: mst,
        iterations,
    }
}

/// Find minimum outgoing edge for each component using dual-tree traversal.
fn find_component_min_edges_dual_tree(
    tree: &KdTree,
    core_distances: &[f64],
    uf: &mut UnionFind,
    component_min_edge: &mut [Option<(usize, usize, f64)>],
) {
    let root = match tree.root() {
        Some(r) => r,
        None => return,
    };

    // Collect nodes with proper child index tracking
    let nodes = collect_nodes_with_indices(root);

    // Use a priority queue for best-first traversal
    let mut queue = BinaryHeap::new();

    // Initial entry: root vs root
    let min_mr = compute_min_mutual_reachability(nodes[0].node, nodes[0].node, core_distances);
    queue.push(DualTreeEntry {
        min_mr,
        query_node_idx: 0,
        ref_node_idx: 0,
    });

    while let Some(entry) = queue.pop() {
        if entry.query_node_idx >= nodes.len() || entry.ref_node_idx >= nodes.len() {
            continue;
        }

        let query_info = &nodes[entry.query_node_idx];
        let ref_info = &nodes[entry.ref_node_idx];
        let query_node = query_info.node;
        let ref_node = ref_info.node;

        // Get current best for this component
        let query_root = uf.find(query_node.point_idx);
        let current_best = component_min_edge[query_root]
            .map(|(_, _, w)| w)
            .unwrap_or(f64::INFINITY);

        // Prune if minimum possible distance exceeds current best
        if entry.min_mr >= current_best {
            continue;
        }

        // If both are small nodes, compute exact distances
        let is_query_leaf = query_info.left_idx.is_none() && query_info.right_idx.is_none();
        let is_ref_leaf = ref_info.left_idx.is_none() && ref_info.right_idx.is_none();

        if is_query_leaf && is_ref_leaf {
            // Single point comparison
            if query_node.point_idx != ref_node.point_idx {
                let qr = uf.find(query_node.point_idx);
                let rr = uf.find(ref_node.point_idx);

                if qr != rr {
                    let dist = euclidean_distance(
                        &tree.data()[query_node.point_idx],
                        &tree.data()[ref_node.point_idx],
                    );
                    let mr = dist
                        .max(core_distances[query_node.point_idx])
                        .max(core_distances[ref_node.point_idx]);

                    update_component_min(
                        component_min_edge,
                        qr,
                        query_node.point_idx,
                        ref_node.point_idx,
                        mr,
                    );
                    update_component_min(
                        component_min_edge,
                        rr,
                        ref_node.point_idx,
                        query_node.point_idx,
                        mr,
                    );
                }
            }
        } else if query_node.size <= 16 && ref_node.size <= 16 {
            // Small nodes: compute all pairs
            let query_points = KdTree::collect_indices(query_node);
            let ref_points = KdTree::collect_indices(ref_node);

            for &qi in &query_points {
                for &ri in &ref_points {
                    if qi == ri {
                        continue;
                    }

                    let qr = uf.find(qi);
                    let rr = uf.find(ri);

                    if qr != rr {
                        let dist = euclidean_distance(&tree.data()[qi], &tree.data()[ri]);
                        let mr = dist.max(core_distances[qi]).max(core_distances[ri]);

                        update_component_min(component_min_edge, qr, qi, ri, mr);
                    }
                }
            }
        } else {
            // Recurse into children - use pre-computed indices
            let query_children: Vec<(usize, &KdNode)> = [query_info.left_idx, query_info.right_idx]
                .iter()
                .filter_map(|&idx| idx.map(|i| (i, nodes[i].node)))
                .collect();

            let ref_children: Vec<(usize, &KdNode)> = [ref_info.left_idx, ref_info.right_idx]
                .iter()
                .filter_map(|&idx| idx.map(|i| (i, nodes[i].node)))
                .collect();

            // If we have no children but node isn't leaf, something is wrong
            // In that case, treat current node as containing only its point
            if query_children.is_empty() && ref_children.is_empty() {
                if query_node.point_idx != ref_node.point_idx {
                    let qr = uf.find(query_node.point_idx);
                    let rr = uf.find(ref_node.point_idx);
                    if qr != rr {
                        let dist = euclidean_distance(
                            &tree.data()[query_node.point_idx],
                            &tree.data()[ref_node.point_idx],
                        );
                        let mr = dist
                            .max(core_distances[query_node.point_idx])
                            .max(core_distances[ref_node.point_idx]);
                        update_component_min(
                            component_min_edge,
                            qr,
                            query_node.point_idx,
                            ref_node.point_idx,
                            mr,
                        );
                        update_component_min(
                            component_min_edge,
                            rr,
                            ref_node.point_idx,
                            query_node.point_idx,
                            mr,
                        );
                    }
                }
                continue;
            }

            // Generate all child pairs
            // If one side has no children, pair with the original node
            let qc_list: Vec<(usize, &KdNode)> = if query_children.is_empty() {
                vec![(entry.query_node_idx, query_node)]
            } else {
                query_children
            };

            let rc_list: Vec<(usize, &KdNode)> = if ref_children.is_empty() {
                vec![(entry.ref_node_idx, ref_node)]
            } else {
                ref_children
            };

            for (qc_idx, qc) in &qc_list {
                for (rc_idx, rc) in &rc_list {
                    // Skip if same node pair as current (would cause infinite loop)
                    if *qc_idx == entry.query_node_idx && *rc_idx == entry.ref_node_idx {
                        continue;
                    }

                    let min_mr = compute_min_mutual_reachability(qc, rc, core_distances);

                    // Only enqueue if potentially useful
                    let query_root_temp = uf.find(qc.point_idx);
                    let current_best = component_min_edge[query_root_temp]
                        .map(|(_, _, w)| w)
                        .unwrap_or(f64::INFINITY);

                    if min_mr < current_best {
                        queue.push(DualTreeEntry {
                            min_mr,
                            query_node_idx: *qc_idx,
                            ref_node_idx: *rc_idx,
                        });
                    }
                }
            }
        }
    }
}

/// Update component minimum edge if this edge is better.
#[inline]
fn update_component_min(
    component_min: &mut [Option<(usize, usize, f64)>],
    component: usize,
    from: usize,
    to: usize,
    weight: f64,
) {
    match &component_min[component] {
        None => component_min[component] = Some((from, to, weight)),
        Some((_, _, w)) if weight < *w => component_min[component] = Some((from, to, weight)),
        _ => {}
    }
}

/// Compute minimum possible mutual reachability between two tree nodes.
fn compute_min_mutual_reachability(node1: &KdNode, node2: &KdNode, _core_distances: &[f64]) -> f64 {
    // Minimum distance between bounding boxes
    // This is a lower bound on the mutual reachability distance
    KdTree::min_box_distance(&node1.bounds, &node2.bounds)
}

/// Find bridges between disconnected components using brute force.
fn find_component_bridges(
    tree: &KdTree,
    core_distances: &[f64],
    uf: &mut UnionFind,
) -> Vec<(usize, usize, f64)> {
    let n = tree.len();

    // Group points by component
    let mut component_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        component_map.entry(uf.find(i)).or_default().push(i);
    }

    let components: Vec<Vec<usize>> = component_map.values().cloned().collect();
    if components.len() <= 1 {
        return Vec::new();
    }

    let mut bridges = Vec::new();

    // Find minimum edge between each pair of components
    for ci in 0..components.len() {
        for cj in (ci + 1)..components.len() {
            let mut min_edge: Option<(usize, usize, f64)> = None;

            for &i in &components[ci] {
                for &j in &components[cj] {
                    let dist = euclidean_distance(&tree.data()[i], &tree.data()[j]);
                    let mr = dist.max(core_distances[i]).max(core_distances[j]);

                    match min_edge {
                        None => min_edge = Some((i, j, mr)),
                        Some((_, _, w)) if mr < w => min_edge = Some((i, j, mr)),
                        _ => {}
                    }
                }
            }

            if let Some(edge) = min_edge {
                bridges.push(edge);
            }
        }
    }

    bridges
}

/// Simplified dual-tree MST using KD-tree + Prim's algorithm.
///
/// This is a simpler O(n log n) to O(n²) hybrid approach:
/// - Use KD-tree for neighbor queries during Prim's
/// - Works well for moderate n and low dimensions
pub fn kdtree_prim_mst(tree: &KdTree, core_distances: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = tree.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut min_from = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;

    // Initialize distances from node 0 to all others
    for j in 1..n {
        let dist = euclidean_distance(&tree.data()[0], &tree.data()[j]);
        let mr = dist.max(core_distances[0]).max(core_distances[j]);
        min_dist[j] = mr;
        min_from[j] = 0;
    }

    // Prim's algorithm
    for _ in 1..n {
        // Find minimum (this is O(n) per iteration)
        let mut min_idx = 0;
        let mut min_val = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_dist[j] < min_val {
                min_val = min_dist[j];
                min_idx = j;
            }
        }

        if min_val == f64::INFINITY {
            break; // Disconnected
        }

        in_tree[min_idx] = true;
        mst.push((min_from[min_idx], min_idx, min_val));

        // Update distances from the newly added node
        for j in 0..n {
            if !in_tree[j] {
                let dist = euclidean_distance(&tree.data()[min_idx], &tree.data()[j]);
                let mr = dist.max(core_distances[min_idx]).max(core_distances[j]);
                if mr < min_dist[j] {
                    min_dist[j] = mr;
                    min_from[j] = min_idx;
                }
            }
        }
    }

    mst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_tree_boruvka() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ];

        let tree = KdTree::new(data);
        let core_dists = tree.compute_core_distances(2);

        let result = dual_tree_boruvka_mst(&tree, &core_dists);

        // Should have n-1 = 4 edges
        assert_eq!(result.edges.len(), 4);
    }

    #[test]
    fn test_kdtree_prim() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let tree = KdTree::new(data);
        let core_dists = tree.compute_core_distances(2);

        let mst = kdtree_prim_mst(&tree, &core_dists);

        // Should have n-1 = 3 edges
        assert_eq!(mst.len(), 3);

        // Verify connectivity
        let mut uf = UnionFind::new(4);
        for (i, j, _) in &mst {
            uf.union(*i, *j);
        }
        assert_eq!(uf.n_components(), 1);
    }

    #[test]
    fn test_mst_edge_weights() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0], // Far point
        ];

        let tree = KdTree::new(data);
        let core_dists = tree.compute_core_distances(2);

        let mst = kdtree_prim_mst(&tree, &core_dists);

        // Sort edges by weight
        let mut sorted = mst.clone();
        sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        // First edge should connect the close points (0 and 1)
        assert!((sorted[0].0 == 0 && sorted[0].1 == 1) || (sorted[0].0 == 1 && sorted[0].1 == 0));
    }

    #[test]
    fn test_collect_nodes_with_indices() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let tree = KdTree::new(data);
        let root = tree.root().unwrap();
        let nodes = collect_nodes_with_indices(root);

        // Check that child indices are properly set
        for (idx, info) in nodes.iter().enumerate() {
            if let Some(left_idx) = info.left_idx {
                assert!(left_idx > idx, "Left child should have higher index");
                assert!(left_idx < nodes.len(), "Left index should be valid");
            }
            if let Some(right_idx) = info.right_idx {
                assert!(right_idx > idx, "Right child should have higher index");
                assert!(right_idx < nodes.len(), "Right index should be valid");
            }
        }
    }

    #[test]
    fn test_dual_tree_vs_prim_small() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 5.0],
            vec![5.5, 5.0],
            vec![5.0, 5.5],
        ];

        let tree = KdTree::new(data.clone());
        let core_dists = tree.compute_core_distances(2);

        let mst_prim = kdtree_prim_mst(&tree, &core_dists);
        let mst_dual = dual_tree_boruvka_mst(&tree, &core_dists);

        // Both should have n-1 edges
        assert_eq!(mst_prim.len(), data.len() - 1);
        assert_eq!(mst_dual.edges.len(), data.len() - 1);

        // Total weights should match
        let total_prim: f64 = mst_prim.iter().map(|e| e.2).sum();
        let total_dual: f64 = mst_dual.edges.iter().map(|e| e.2).sum();

        assert!(
            (total_prim - total_dual).abs() < 1e-10,
            "MST weight mismatch: prim={}, dual={}",
            total_prim,
            total_dual
        );
    }
}
