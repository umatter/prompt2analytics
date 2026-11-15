//! Boruvka's algorithm for the mutual-reachability minimum spanning tree.
//!
//! [`dual_tree_boruvka_mst`] runs Boruvka's algorithm with k-d tree nearest-other-component
//! queries: O(log n) rounds of n pruned queries; near O(n log^2 n) for low-dimensional data,
//! O(n^2) per round in the worst case. The name is historical: the traversal is per query point,
//! not dual-tree. [`kdtree_prim_mst`] is Prim's algorithm on the dense graph.
//!
//! HDBSCAN does not use these builders: its spanning tree is Prim's algorithm with dbscan's tie
//! order ([`crate::ml::hdbscan()`]).
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

/// Result of the Boruvka MST construction.
pub struct DualTreeMstResult {
    /// MST edges as (from, to, weight)
    pub edges: Vec<(usize, usize, f64)>,
    /// Number of Boruvka rounds
    pub iterations: usize,
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

/// An edge `(weight, lower index, higher index, from, to)`; edges are ordered by
/// `(weight, lower, higher)`, a total order that keeps Boruvka correct with equal weights.
#[derive(Clone, Copy, Debug)]
struct Edge {
    w: f64,
    lo: usize,
    hi: usize,
    from: usize,
    to: usize,
}

impl Edge {
    fn new(from: usize, to: usize, w: f64) -> Self {
        Edge {
            w,
            lo: from.min(to),
            hi: from.max(to),
            from,
            to,
        }
    }

    fn cmp_key(&self, other: &Edge) -> Ordering {
        self.w
            .total_cmp(&other.w)
            .then(self.lo.cmp(&other.lo))
            .then(self.hi.cmp(&other.hi))
    }
}

/// Keep the smaller of `best` and `e` under the edge order.
fn keep_min(best: &mut Option<Edge>, e: Edge) {
    if best.is_none_or(|b| e.cmp_key(&b) == Ordering::Less) {
        *best = Some(e);
    }
}

/// Marker of a subtree whose points lie in more than one component.
const MIXED: usize = usize::MAX;

/// Build the minimum spanning tree of the mutual-reachability graph
/// (`max(d(i, j), core[i], core[j])`) by Boruvka's algorithm.
///
/// Each round finds every component's lightest outgoing edge: one k-d tree query per point for
/// its lightest edge to another component, skipping subtrees whose points all lie in the query's
/// component and subtrees whose lower bound `max(box distance, core[q], min core of the subtree)`
/// exceeds the query's best edge; then each component's lightest edge is added. Edges are
/// compared by `(weight, lower index, higher index)`, so ties cannot close a cycle and the tree
/// is minimal. O(log n) rounds of n pruned queries: near O(n log^2 n) for low-dimensional data,
/// O(n^2) per round in the worst case. The name is historical: the traversal is per query point,
/// not dual-tree.
///
/// # Arguments
/// * `tree` - KD-tree of the data points
/// * `core_distances` - Pre-computed core distances for each point
///
/// # Returns
/// * MST edges as (from, to, mutual_reachability_distance) and the number of rounds
pub fn dual_tree_boruvka_mst(tree: &KdTree, core_distances: &[f64]) -> DualTreeMstResult {
    let n = tree.len();
    let root = match tree.root() {
        Some(r) if n > 1 => r,
        _ => {
            return DualTreeMstResult {
                edges: Vec::new(),
                iterations: 0,
            };
        }
    };
    let nodes = collect_nodes_with_indices(root);

    // Minimum core distance of each subtree (fixed over the rounds); children follow their
    // parent in `nodes`, so a reverse pass sees the children first.
    let mut min_core = vec![f64::INFINITY; nodes.len()];
    for idx in (0..nodes.len()).rev() {
        let info = &nodes[idx];
        let mut m = core_distances[info.node.point_idx];
        for c in [info.left_idx, info.right_idx].into_iter().flatten() {
            m = m.min(min_core[c]);
        }
        min_core[idx] = m;
    }

    let mut uf = UnionFind::new(n);
    let mut mst = Vec::with_capacity(n - 1);
    let mut iterations = 0;
    let mut comp = vec![0usize; n];
    let mut uniform = vec![MIXED; nodes.len()];

    while mst.len() < n - 1 {
        iterations += 1;
        for (i, c) in comp.iter_mut().enumerate() {
            *c = uf.find(i);
        }
        // The one component of each subtree's points, or MIXED.
        for idx in (0..nodes.len()).rev() {
            let info = &nodes[idx];
            let mut u = comp[info.node.point_idx];
            for c in [info.left_idx, info.right_idx].into_iter().flatten() {
                if uniform[c] != u {
                    u = MIXED;
                }
            }
            uniform[idx] = u;
        }

        let per_point: Vec<Option<Edge>> = (0..n)
            .into_par_iter()
            .map(|q| {
                nearest_other_component(tree, &nodes, &min_core, &uniform, &comp, core_distances, q)
            })
            .collect();

        // Each component's lightest edge (the order is total, so the result does not depend
        // on the order of the points)
        let mut best: Vec<Option<Edge>> = vec![None; n];
        for (q, e) in per_point.into_iter().enumerate() {
            if let Some(e) = e {
                keep_min(&mut best[comp[q]], e);
            }
        }

        let mut added_any = false;
        for e in best.into_iter().flatten() {
            if uf.find(e.from) != uf.find(e.to) {
                uf.union(e.from, e.to);
                mst.push((e.from, e.to, e.w));
                added_any = true;
            }
        }
        if !added_any {
            // no finite outgoing edge (non-finite data): nothing more to join
            break;
        }
    }

    DualTreeMstResult {
        edges: mst,
        iterations,
    }
}

/// The lightest mutual-reachability edge from point `q` to a point of another component.
fn nearest_other_component(
    tree: &KdTree,
    nodes: &[TreeNodeInfo<'_>],
    min_core: &[f64],
    uniform: &[usize],
    comp: &[usize],
    core: &[f64],
    q: usize,
) -> Option<Edge> {
    let data = tree.data();
    let point = &data[q];
    let cq = comp[q];
    let mut best: Option<Edge> = None;
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if uniform[idx] == cq {
            continue;
        }
        let info = &nodes[idx];
        let bound = KdTree::min_dist_to_box(point, &info.node.bounds)
            .max(core[q])
            .max(min_core[idx]);
        // strict: an equal weight can still give a smaller edge under the index order
        if best.is_some_and(|b| bound > b.w) {
            continue;
        }
        let p = info.node.point_idx;
        if comp[p] != cq {
            let w = euclidean_distance(point, &data[p])
                .max(core[q])
                .max(core[p]);
            keep_min(&mut best, Edge::new(q, p, w));
        }
        // the nearer child is searched first (pushed last)
        match (info.left_idx, info.right_idx) {
            (Some(l), Some(r)) => {
                let dl = KdTree::min_dist_to_box(point, &nodes[l].node.bounds);
                let dr = KdTree::min_dist_to_box(point, &nodes[r].node.bounds);
                if dl <= dr {
                    stack.extend([r, l]);
                } else {
                    stack.extend([l, r]);
                }
            }
            (Some(c), None) | (None, Some(c)) => stack.push(c),
            (None, None) => {}
        }
    }
    best
}

/// Prim's algorithm on the dense mutual-reachability graph: O(n^2) distance evaluations; the tree
/// supplies only the points.
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
        sorted.sort_by(|a, b| a.2.total_cmp(&b.2));

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

    /// Kruskal on every pair: the minimum spanning tree weight.
    fn brute_force_mst_weight(data: &[Vec<f64>], core: &[f64]) -> f64 {
        let n = data.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let w = euclidean_distance(&data[i], &data[j])
                    .max(core[i])
                    .max(core[j]);
                edges.push((i, j, w));
            }
        }
        crate::ml::kdtree::kruskal_mst(&edges, n)
            .iter()
            .map(|e| e.2)
            .sum()
    }

    fn noisy(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|_| (i % 3) as f64 * 4.0 + rng.r#gen::<f64>())
                    .collect()
            })
            .collect()
    }

    fn check_spanning(edges: &[(usize, usize, f64)], n: usize) {
        assert_eq!(edges.len(), n - 1);
        let mut uf = UnionFind::new(n);
        for &(a, b, _) in edges {
            assert!(uf.union(a, b), "cycle at ({a}, {b})");
        }
        assert_eq!(uf.n_components(), 1);
    }

    #[test]
    fn test_boruvka_matches_brute_force_mst() {
        for d in [2, 5] {
            for n in [50, 200, 1000] {
                for ms in [1, 5] {
                    let data = noisy(n, d, (n * d + ms) as u64);
                    let tree = KdTree::new(data.clone());
                    let core = if ms == 1 {
                        vec![0.0; n]
                    } else {
                        tree.compute_core_distances(ms)
                    };
                    let r = dual_tree_boruvka_mst(&tree, &core);
                    check_spanning(&r.edges, n);
                    let total: f64 = r.edges.iter().map(|e| e.2).sum();
                    let truth = brute_force_mst_weight(&data, &core);
                    assert!(
                        (total - truth).abs() <= 1e-12 * truth,
                        "d {d} n {n} ms {ms}: {total} vs {truth}"
                    );
                    if ms == 1 {
                        // distinct weights: the edge set is the unique MST
                        let mut a: Vec<(usize, usize)> = r
                            .edges
                            .iter()
                            .map(|e| (e.0.min(e.1), e.0.max(e.1)))
                            .collect();
                        let prim = kdtree_prim_mst(&tree, &core);
                        let mut b: Vec<(usize, usize)> =
                            prim.iter().map(|e| (e.0.min(e.1), e.0.max(e.1))).collect();
                        a.sort_unstable();
                        b.sort_unstable();
                        assert_eq!(a, b, "d {d} n {n}");
                    }
                    assert!(r.iterations <= (n as f64).log2().ceil() as usize);
                }
            }
        }
    }

    #[test]
    fn test_boruvka_ties_give_a_minimum_tree() {
        let data: Vec<Vec<f64>> = (0..144)
            .map(|i| vec![(i % 12) as f64, (i / 12) as f64])
            .collect();
        for ms in [1, 3] {
            let tree = KdTree::new(data.clone());
            let core = if ms == 1 {
                vec![0.0; 144]
            } else {
                tree.compute_core_distances(ms)
            };
            let r = dual_tree_boruvka_mst(&tree, &core);
            check_spanning(&r.edges, 144);
            let total: f64 = r.edges.iter().map(|e| e.2).sum();
            let truth = brute_force_mst_weight(&data, &core);
            assert!((total - truth).abs() <= 1e-12 * truth, "{total} vs {truth}");
        }
    }
}
