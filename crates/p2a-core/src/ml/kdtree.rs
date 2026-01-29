//! KD-tree for efficient nearest neighbor queries.
//!
//! Provides O(log n) nearest neighbor queries for spatial clustering algorithms.
//! Enhanced with bounding boxes for dual-tree algorithms (HDBSCAN, OPTICS).

use rayon::prelude::*;
use std::cmp::Ordering;

/// A node in the KD-tree with bounding box for dual-tree pruning.
#[derive(Debug)]
pub struct KdNode {
    /// Index of the point in the original data
    pub point_idx: usize,
    /// Split dimension
    pub split_dim: usize,
    /// Split value
    pub split_val: f64,
    /// Left child (points with coord < split_val)
    pub left: Option<Box<KdNode>>,
    /// Right child (points with coord >= split_val)
    pub right: Option<Box<KdNode>>,
    /// Bounding box: (min, max) per dimension
    pub bounds: (Vec<f64>, Vec<f64>),
    /// Number of points in this subtree
    pub size: usize,
    /// All point indices in this subtree (for leaf nodes or small subtrees)
    pub point_indices: Vec<usize>,
}

/// KD-tree for efficient spatial queries.
///
/// Supports:
/// - O(log n) k-nearest neighbor queries
/// - O(log n + k) radius queries
/// - O(n log n) dual-tree traversal for MST construction
pub struct KdTree {
    root: Option<Box<KdNode>>,
    data: Vec<Vec<f64>>,
    n_dims: usize,
    n_points: usize,
}

impl KdTree {
    /// Build a KD-tree from data points.
    ///
    /// # Arguments
    /// * `data` - Data matrix as Vec of Vec (n_points x n_dims)
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let n = data.len();
        if n == 0 {
            return KdTree {
                root: None,
                data,
                n_dims: 0,
                n_points: 0,
            };
        }

        let n_dims = data[0].len();
        let indices: Vec<usize> = (0..n).collect();

        let root = Self::build_tree(&data, indices, 0, n_dims);

        KdTree {
            root,
            data,
            n_dims,
            n_points: n,
        }
    }

    /// Get the number of points in the tree.
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Get reference to the underlying data.
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    /// Get the root node.
    pub fn root(&self) -> Option<&KdNode> {
        self.root.as_ref().map(|b| b.as_ref())
    }

    /// Get the number of dimensions.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    fn build_tree(
        data: &[Vec<f64>],
        mut indices: Vec<usize>,
        depth: usize,
        n_dims: usize,
    ) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }

        // Compute bounding box for all points in this subtree
        let bounds = Self::compute_bounds(data, &indices, n_dims);
        let size = indices.len();

        let split_dim = depth % n_dims;

        // Sort indices by the split dimension
        indices.sort_by(|&a, &b| {
            data[a][split_dim]
                .partial_cmp(&data[b][split_dim])
                .unwrap_or(Ordering::Equal)
        });

        let median_idx = indices.len() / 2;
        let point_idx = indices[median_idx];
        let split_val = data[point_idx][split_dim];

        let left_indices: Vec<usize> = indices[..median_idx].to_vec();
        let right_indices: Vec<usize> = indices[median_idx + 1..].to_vec();

        // Store point_indices for small subtrees (enables efficient leaf enumeration)
        let point_indices = if size <= 32 {
            indices.clone()
        } else {
            Vec::new()
        };

        Some(Box::new(KdNode {
            point_idx,
            split_dim,
            split_val,
            left: Self::build_tree(data, left_indices, depth + 1, n_dims),
            right: Self::build_tree(data, right_indices, depth + 1, n_dims),
            bounds,
            size,
            point_indices,
        }))
    }

    /// Compute bounding box for a set of points.
    fn compute_bounds(data: &[Vec<f64>], indices: &[usize], n_dims: usize) -> (Vec<f64>, Vec<f64>) {
        let mut min_bounds = vec![f64::INFINITY; n_dims];
        let mut max_bounds = vec![f64::NEG_INFINITY; n_dims];

        for &idx in indices {
            for d in 0..n_dims {
                min_bounds[d] = min_bounds[d].min(data[idx][d]);
                max_bounds[d] = max_bounds[d].max(data[idx][d]);
            }
        }

        (min_bounds, max_bounds)
    }

    /// Compute minimum distance between a point and a bounding box.
    #[inline]
    pub fn min_dist_to_box(point: &[f64], bounds: &(Vec<f64>, Vec<f64>)) -> f64 {
        let mut sum = 0.0;
        for (d, &p) in point.iter().enumerate() {
            if p < bounds.0[d] {
                let diff = bounds.0[d] - p;
                sum += diff * diff;
            } else if p > bounds.1[d] {
                let diff = p - bounds.1[d];
                sum += diff * diff;
            }
        }
        sum.sqrt()
    }

    /// Compute minimum distance between two bounding boxes.
    #[inline]
    pub fn min_box_distance(bounds1: &(Vec<f64>, Vec<f64>), bounds2: &(Vec<f64>, Vec<f64>)) -> f64 {
        let mut sum = 0.0;
        for d in 0..bounds1.0.len() {
            let gap = if bounds1.1[d] < bounds2.0[d] {
                bounds2.0[d] - bounds1.1[d]
            } else if bounds2.1[d] < bounds1.0[d] {
                bounds1.0[d] - bounds2.1[d]
            } else {
                0.0
            };
            sum += gap * gap;
        }
        sum.sqrt()
    }

    /// Compute maximum distance between two bounding boxes.
    #[inline]
    pub fn max_box_distance(bounds1: &(Vec<f64>, Vec<f64>), bounds2: &(Vec<f64>, Vec<f64>)) -> f64 {
        let mut sum = 0.0;
        for d in 0..bounds1.0.len() {
            let dist1 = (bounds1.1[d] - bounds2.0[d]).abs();
            let dist2 = (bounds2.1[d] - bounds1.0[d]).abs();
            let max_dist = dist1.max(dist2);
            sum += max_dist * max_dist;
        }
        sum.sqrt()
    }

    /// Find k nearest neighbors of a query point.
    ///
    /// Returns vector of (distance, point_index) sorted by distance.
    pub fn k_nearest(
        &self,
        query: &[f64],
        k: usize,
        exclude_self: Option<usize>,
    ) -> Vec<(f64, usize)> {
        let mut neighbors = BoundedHeap::new(k);

        if let Some(root) = &self.root {
            self.search_knn(root, query, &mut neighbors, exclude_self);
        }

        neighbors.into_sorted_vec()
    }

    fn search_knn(
        &self,
        node: &KdNode,
        query: &[f64],
        neighbors: &mut BoundedHeap,
        exclude_self: Option<usize>,
    ) {
        // Compute distance to this node's point
        let dist = euclidean_distance(query, &self.data[node.point_idx]);

        // Add to neighbors if not excluded
        if exclude_self != Some(node.point_idx) {
            neighbors.push(dist, node.point_idx);
        }

        // Determine which subtree to search first
        let go_left = query[node.split_dim] < node.split_val;
        let (first, second) = if go_left {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the closer subtree first
        if let Some(child) = first {
            self.search_knn(child, query, neighbors, exclude_self);
        }

        // Check if we need to search the other subtree
        let split_dist = (query[node.split_dim] - node.split_val).abs();
        if neighbors.should_search(split_dist) {
            if let Some(child) = second {
                self.search_knn(child, query, neighbors, exclude_self);
            }
        }
    }

    /// Query all points within a given radius (sorted by distance).
    pub fn radius_query(
        &self,
        query: &[f64],
        radius: f64,
        exclude_self: Option<usize>,
    ) -> Vec<(f64, usize)> {
        let mut results = Vec::new();

        if let Some(root) = &self.root {
            self.search_radius(root, query, radius, &mut results, exclude_self);
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        results
    }

    /// Query all points within a given radius (unsorted, faster for DBSCAN).
    pub fn radius_query_unsorted(
        &self,
        query: &[f64],
        radius: f64,
        exclude_self: Option<usize>,
    ) -> Vec<(f64, usize)> {
        let mut results = Vec::new();

        if let Some(root) = &self.root {
            self.search_radius(root, query, radius, &mut results, exclude_self);
        }

        results
    }

    fn search_radius(
        &self,
        node: &KdNode,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(f64, usize)>,
        exclude_self: Option<usize>,
    ) {
        let dist = euclidean_distance(query, &self.data[node.point_idx]);

        if dist <= radius && exclude_self != Some(node.point_idx) {
            results.push((dist, node.point_idx));
        }

        let _split_dist = (query[node.split_dim] - node.split_val).abs();

        // Search left subtree if it could contain points within radius
        if query[node.split_dim] - radius <= node.split_val {
            if let Some(left) = &node.left {
                self.search_radius(left, query, radius, results, exclude_self);
            }
        }

        // Search right subtree if it could contain points within radius
        if query[node.split_dim] + radius >= node.split_val {
            if let Some(right) = &node.right {
                self.search_radius(right, query, radius, results, exclude_self);
            }
        }
    }
}

/// Bounded max-heap for k-nearest neighbors.
struct BoundedHeap {
    k: usize,
    items: Vec<(f64, usize)>, // (distance, index)
}

impl BoundedHeap {
    fn new(k: usize) -> Self {
        BoundedHeap {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    fn push(&mut self, dist: f64, idx: usize) {
        if self.items.len() < self.k {
            self.items.push((dist, idx));
            // Bubble up
            self.sift_up(self.items.len() - 1);
        } else if dist < self.items[0].0 {
            // Replace the max (root)
            self.items[0] = (dist, idx);
            self.sift_down(0);
        }
    }

    fn should_search(&self, split_dist: f64) -> bool {
        self.items.len() < self.k || split_dist < self.items[0].0
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.items[idx].0 > self.items[parent].0 {
                self.items.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.items.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;

            if left < len && self.items[left].0 > self.items[largest].0 {
                largest = left;
            }
            if right < len && self.items[right].0 > self.items[largest].0 {
                largest = right;
            }

            if largest != idx {
                self.items.swap(idx, largest);
                idx = largest;
            } else {
                break;
            }
        }
    }

    fn into_sorted_vec(mut self) -> Vec<(f64, usize)> {
        self.items
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        self.items
    }
}

/// Compute Euclidean distance between two points.
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// =============================================================================
// Additional methods for HDBSCAN/OPTICS
// =============================================================================

impl KdTree {
    /// Compute core distances for all points in parallel.
    ///
    /// Core distance = distance to k-th nearest neighbor.
    /// Uses O(n * k * log n) via KD-tree queries instead of O(n²).
    pub fn compute_core_distances(&self, min_samples: usize) -> Vec<f64> {
        let n = self.n_points;
        if n == 0 {
            return Vec::new();
        }

        (0..n)
            .into_par_iter()
            .map(|i| {
                let neighbors = self.k_nearest(&self.data[i], min_samples, Some(i));
                if neighbors.len() >= min_samples {
                    neighbors[min_samples - 1].0
                } else if !neighbors.is_empty() {
                    neighbors.last().map(|x| x.0).unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get all point indices in a subtree.
    pub fn collect_indices(node: &KdNode) -> Vec<usize> {
        if !node.point_indices.is_empty() {
            return node.point_indices.clone();
        }

        let mut indices = vec![node.point_idx];
        if let Some(ref left) = node.left {
            indices.extend(Self::collect_indices(left));
        }
        if let Some(ref right) = node.right {
            indices.extend(Self::collect_indices(right));
        }
        indices
    }

    /// Check if a node is a leaf (no children).
    #[inline]
    pub fn is_leaf(node: &KdNode) -> bool {
        node.left.is_none() && node.right.is_none()
    }

    /// Collect edges within a radius for sparse MST construction.
    ///
    /// Returns edges as (i, j, mutual_reachability_distance).
    pub fn collect_edges_within_radius(
        &self,
        core_distances: &[f64],
        radius_multiplier: f64,
    ) -> Vec<(usize, usize, f64)> {
        let n = self.n_points;
        if n == 0 {
            return Vec::new();
        }

        let max_core = core_distances.iter().cloned().fold(0.0f64, f64::max);
        let search_radius = radius_multiplier * max_core;

        // Collect edges in parallel
        let core_ref = core_distances;
        let edges: Vec<Vec<(usize, usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let neighbors = self.radius_query(&self.data[i], search_radius, Some(i));
                neighbors
                    .into_iter()
                    .filter(|&(_, j)| i < j)
                    .map(|(dist, j)| {
                        let mr = dist.max(core_ref[i]).max(core_ref[j]);
                        (i, j, mr)
                    })
                    .collect()
            })
            .collect();

        edges.into_iter().flatten().collect()
    }
}

/// Union-Find data structure for connected component detection.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // Path compression
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
            self.size[ry] += self.size[rx];
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
        } else {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
            self.rank[rx] += 1;
        }
        true
    }

    pub fn component_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    pub fn n_components(&mut self) -> usize {
        let n = self.parent.len();
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(self.find(i));
        }
        roots.len()
    }
}

/// Build MST using Kruskal's algorithm from a list of edges.
///
/// Returns edges in MST as (i, j, weight).
pub fn kruskal_mst(edges: &[(usize, usize, f64)], n: usize) -> Vec<(usize, usize, f64)> {
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let mut uf = UnionFind::new(n);
    let mut mst = Vec::with_capacity(n - 1);

    for (i, j, w) in sorted_edges {
        if uf.union(i, j) {
            mst.push((i, j, w));
            if mst.len() == n - 1 {
                break;
            }
        }
    }

    mst
}

/// Build a connected MST with adaptive radius expansion.
///
/// Starts with edges within a radius, then expands if graph is disconnected.
/// Falls back to brute-force bridges if needed.
pub fn build_connected_mst(tree: &KdTree, core_distances: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = tree.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut radius_mult = 2.0;
    let max_iterations = 5;

    for _ in 0..max_iterations {
        let edges = tree.collect_edges_within_radius(core_distances, radius_mult);
        let mst = kruskal_mst(&edges, n);

        if mst.len() == n - 1 {
            return mst;
        }

        // Graph is disconnected - find components and bridge them
        let mut uf = UnionFind::new(n);
        for &(i, j, _) in &mst {
            uf.union(i, j);
        }

        let n_components = uf.n_components();
        if n_components > 1 {
            // Find component representatives
            let mut component_map: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for i in 0..n {
                component_map.entry(uf.find(i)).or_default().push(i);
            }

            // Add bridges between components
            let mut all_edges = edges;
            let components: Vec<Vec<usize>> = component_map.values().cloned().collect();

            for ci in 0..components.len() {
                for cj in (ci + 1)..components.len() {
                    // Find minimum edge between components
                    let mut min_edge: Option<(usize, usize, f64)> = None;

                    for &i in &components[ci] {
                        for &j in &components[cj] {
                            let dist = euclidean_distance(&tree.data()[i], &tree.data()[j]);
                            let mr = dist.max(core_distances[i]).max(core_distances[j]);

                            match min_edge {
                                None => min_edge = Some((i.min(j), i.max(j), mr)),
                                Some((_, _, w)) if mr < w => {
                                    min_edge = Some((i.min(j), i.max(j), mr));
                                }
                                _ => {}
                            }
                        }
                    }

                    if let Some(edge) = min_edge {
                        all_edges.push(edge);
                    }
                }
            }

            let mst = kruskal_mst(&all_edges, n);
            if mst.len() == n - 1 {
                return mst;
            }
        }

        radius_mult *= 1.5;
    }

    // Fall back to full brute-force MST
    brute_force_mst(tree.data(), core_distances)
}

/// Brute-force MST construction using Prim's algorithm.
fn brute_force_mst(data: &[Vec<f64>], core_distances: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = data.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut min_from = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    for j in 1..n {
        let dist = euclidean_distance(&data[0], &data[j]);
        min_dist[j] = dist.max(core_distances[0]).max(core_distances[j]);
        min_from[j] = 0;
    }

    for _ in 1..n {
        // Find minimum
        let mut min_idx = 0;
        let mut min_val = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_dist[j] < min_val {
                min_val = min_dist[j];
                min_idx = j;
            }
        }

        in_tree[min_idx] = true;
        mst.push((min_from[min_idx], min_idx, min_val));

        // Update distances
        for j in 0..n {
            if !in_tree[j] {
                let dist = euclidean_distance(&data[min_idx], &data[j]);
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
    fn test_kdtree_knn() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let tree = KdTree::new(data);

        // Query for 2 nearest neighbors of point at origin
        let neighbors = tree.k_nearest(&[0.0, 0.0], 2, None);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].1, 0); // Self is closest (distance 0)
        // Second closest is (0.5, 0.5) at distance sqrt(0.5) ≈ 0.707
        assert_eq!(neighbors[1].1, 4);
    }

    #[test]
    fn test_kdtree_radius() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];

        let tree = KdTree::new(data);

        // Query for points within radius 1.5 of origin
        let neighbors = tree.radius_query(&[0.0, 0.0], 1.5, None);
        assert_eq!(neighbors.len(), 3); // (0,0), (1,0), (0,1)
    }

    #[test]
    fn test_bounding_boxes() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let tree = KdTree::new(data);
        let root = tree.root().unwrap();

        // Root should have bounds covering all points
        assert_eq!(root.bounds.0, vec![0.0, 0.0]);
        assert_eq!(root.bounds.1, vec![1.0, 1.0]);
        assert_eq!(root.size, 4);
    }

    #[test]
    fn test_min_box_distance() {
        let bounds1 = (vec![0.0, 0.0], vec![1.0, 1.0]);
        let bounds2 = (vec![2.0, 2.0], vec![3.0, 3.0]);

        // Diagonal distance from (1,1) to (2,2)
        let dist = KdTree::min_box_distance(&bounds1, &bounds2);
        assert!((dist - 2.0_f64.sqrt()).abs() < 1e-10);

        // Overlapping boxes
        let bounds3 = (vec![0.5, 0.5], vec![1.5, 1.5]);
        let dist_overlap = KdTree::min_box_distance(&bounds1, &bounds3);
        assert!((dist_overlap).abs() < 1e-10);
    }

    #[test]
    fn test_core_distances() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0], // Outlier
        ];

        let tree = KdTree::new(data);
        let core_dists = tree.compute_core_distances(2);

        assert_eq!(core_dists.len(), 4);
        // First 3 points are close, so their 2nd neighbor is ~1 away
        assert!(core_dists[0] < 1.5);
        assert!(core_dists[1] < 1.5);
        assert!(core_dists[2] < 1.5);
        // Outlier's 2nd neighbor is much farther
        assert!(core_dists[3] > 10.0);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        assert_eq!(uf.n_components(), 5);

        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.n_components(), 3);

        uf.union(1, 2);
        assert_eq!(uf.n_components(), 2);

        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_kruskal_mst() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 1.0)];

        let mst = kruskal_mst(&edges, 4);

        assert_eq!(mst.len(), 3);
        // Total weight should be 1 + 2 + 1 = 4 (not including the 3.0 edge)
        let total: f64 = mst.iter().map(|e| e.2).sum();
        assert!((total - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_connected_mst() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let tree = KdTree::new(data);
        let core_dists = tree.compute_core_distances(2);
        let mst = build_connected_mst(&tree, &core_dists);

        // Should have n-1 edges
        assert_eq!(mst.len(), 3);
    }
}
