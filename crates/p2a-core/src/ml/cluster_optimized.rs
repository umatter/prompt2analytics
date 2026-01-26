//! Optimized clustering algorithms using parallelization and efficient data structures.
//!
//! This module provides high-performance implementations of clustering algorithms
//! using rayon for parallelization and optimized memory access patterns.

use ndarray::{Array2, ArrayView2, Axis, Zip};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// =============================================================================
// Optimized Pairwise Distance Computation
// =============================================================================

/// Compute pairwise Euclidean distances with parallelization.
///
/// Uses rayon to parallelize over rows, achieving near-linear speedup.
pub fn compute_pairwise_distances_parallel(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();

    // Pre-allocate output matrix
    let mut distances = Array2::zeros((n, n));

    // Convert to contiguous data for better cache performance
    let data_slice: Vec<f64> = data.iter().cloned().collect();
    let data_ref = &data_slice;

    // Compute upper triangle in parallel
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).map(move |j| {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = data_ref[i * d + k] - data_ref[j * d + k];
                    sum += diff * diff;
                }
                (i, j, sum.sqrt())
            }).collect::<Vec<_>>()
        })
        .collect();

    // Fill matrix (this is fast, just memory writes)
    for (i, j, dist) in results {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }

    distances
}

/// Compute squared distances only (avoiding sqrt for methods that don't need it).
pub fn compute_pairwise_sq_distances_parallel(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();

    let mut distances = Array2::zeros((n, n));
    let data_slice: Vec<f64> = data.iter().cloned().collect();
    let data_ref = &data_slice;

    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).map(move |j| {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = data_ref[i * d + k] - data_ref[j * d + k];
                    sum += diff * diff;
                }
                (i, j, sum)
            }).collect::<Vec<_>>()
        })
        .collect();

    for (i, j, dist) in results {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }

    distances
}

// =============================================================================
// Optimized Silhouette Coefficient
// =============================================================================

/// Optimized silhouette computation with parallelization.
///
/// This version:
/// - Parallelizes pairwise distance computation
/// - Parallelizes silhouette computation for each point
/// - Uses pre-computed cluster membership for O(1) lookups
pub fn silhouette_optimized(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<(Vec<f64>, f64), String> {
    let n = data.nrows();

    if labels.len() != n {
        return Err("Labels length mismatch".to_string());
    }

    if n == 0 {
        return Err("Empty data".to_string());
    }

    let n_clusters = *labels.iter().max().unwrap_or(&0) + 1;
    if n_clusters < 2 {
        return Err("Need at least 2 clusters".to_string());
    }

    // Pre-compute cluster indices
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        cluster_indices[label].push(i);
    }

    // Compute pairwise distances in parallel
    let distances = compute_pairwise_distances_parallel(&data);

    // Compute silhouette for each point in parallel
    let silhouette_widths: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cluster_i = labels[i];
            let cluster_i_indices = &cluster_indices[cluster_i];

            // a(i): average distance to other points in same cluster
            let a_i = if cluster_i_indices.len() > 1 {
                let sum: f64 = cluster_i_indices.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| distances[[i, j]])
                    .sum();
                sum / (cluster_i_indices.len() - 1) as f64
            } else {
                0.0
            };

            // b(i): min average distance to any other cluster
            let mut b_i = f64::INFINITY;
            for c in 0..n_clusters {
                if c == cluster_i || cluster_indices[c].is_empty() {
                    continue;
                }
                let avg_dist: f64 = cluster_indices[c].iter()
                    .map(|&j| distances[[i, j]])
                    .sum::<f64>() / cluster_indices[c].len() as f64;
                if avg_dist < b_i {
                    b_i = avg_dist;
                }
            }

            // Silhouette width
            if a_i == 0.0 && b_i == 0.0 {
                0.0
            } else if a_i.is_infinite() || b_i.is_infinite() {
                0.0
            } else {
                (b_i - a_i) / a_i.max(b_i)
            }
        })
        .collect();

    let average = silhouette_widths.iter().sum::<f64>() / n as f64;

    Ok((silhouette_widths, average))
}

// =============================================================================
// Optimized K-Medoids with FastPAM
// =============================================================================

/// FastPAM-style K-Medoids with O(n) swap evaluation.
///
/// This implementation uses:
/// - Parallel distance matrix computation
/// - FastPAM swap strategy: evaluate all swaps in O(n) instead of O(n²)
/// - Parallel assignment
pub fn kmedoids_optimized(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<(Vec<usize>, Vec<usize>, f64, usize), String> {
    let n = data.nrows();

    if k == 0 || k > n {
        return Err(format!("k must be between 1 and {}", n));
    }

    let max_iter = max_iterations.unwrap_or(100);

    // Compute distances in parallel
    let distances = compute_pairwise_distances_parallel(&data);

    // BUILD phase: Initialize medoids
    let mut medoid_indices = build_medoids_fast(&distances, k, seed);

    // Pre-compute nearest and second nearest medoid for each point
    let (mut nearest, mut second_nearest, mut d_nearest, mut d_second) =
        compute_nearest_medoids(&distances, &medoid_indices);

    let mut total_cost: f64 = d_nearest.iter().sum();
    let mut n_iterations = 0;

    // SWAP phase using FastPAM strategy
    for iter in 0..max_iter {
        n_iterations = iter + 1;
        let mut best_gain = 0.0f64;
        let mut best_swap = (0usize, 0usize); // (medoid_idx, candidate)

        // Compute change in cost (TD) for each possible swap in parallel
        // FastPAM: For each non-medoid candidate, compute total change
        let candidates: Vec<usize> = (0..n)
            .filter(|i| !medoid_indices.contains(i))
            .collect();

        // Use references to avoid move issues in nested closures
        let distances_ref = &distances;
        let nearest_ref = &nearest;
        let d_nearest_ref = &d_nearest;
        let d_second_ref = &d_second;
        let medoid_indices_ref = &medoid_indices;

        let swap_gains: Vec<(usize, usize, f64)> = candidates
            .par_iter()
            .flat_map(|&candidate| {
                // For each medoid m, compute gain of swapping m with candidate
                (0..k).map(move |m_idx| {
                    let _medoid = medoid_indices_ref[m_idx];
                    let mut gain = 0.0;

                    for i in 0..n {
                        if i == candidate {
                            continue;
                        }

                        let d_candidate = distances_ref[[i, candidate]];

                        if nearest_ref[i] == m_idx {
                            // Point i is currently assigned to medoid being removed
                            // New distance = min(d_second[i], d_candidate)
                            if d_candidate < d_second_ref[i] {
                                gain += d_nearest_ref[i] - d_candidate;
                            } else {
                                gain += d_nearest_ref[i] - d_second_ref[i];
                            }
                        } else {
                            // Point i is not assigned to medoid being removed
                            if d_candidate < d_nearest_ref[i] {
                                gain += d_nearest_ref[i] - d_candidate;
                            }
                            // else: no change
                        }
                    }

                    (m_idx, candidate, gain)
                }).collect::<Vec<_>>()
            })
            .collect();

        // Find best swap
        for (m_idx, candidate, gain) in swap_gains {
            if gain > best_gain + 1e-10 {
                best_gain = gain;
                best_swap = (m_idx, candidate);
            }
        }

        if best_gain <= 1e-10 {
            break; // No improvement possible
        }

        // Apply best swap
        medoid_indices[best_swap.0] = best_swap.1;
        total_cost -= best_gain;

        // Recompute nearest medoids
        let (nn, sn, dn, ds) = compute_nearest_medoids(&distances, &medoid_indices);
        nearest = nn;
        second_nearest = sn;
        d_nearest = dn;
        d_second = ds;
    }

    // Final assignment
    let labels: Vec<usize> = nearest;

    Ok((labels, medoid_indices, total_cost, n_iterations))
}

/// BUILD phase with optimized first medoid selection.
fn build_medoids_fast(distances: &Array2<f64>, k: usize, seed: Option<u64>) -> Vec<usize> {
    let n = distances.nrows();
    let mut medoids = Vec::with_capacity(k);

    // First medoid: point minimizing sum of distances (computed in parallel)
    let sums: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| distances.row(i).sum())
        .collect();

    let first_medoid = sums.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    medoids.push(first_medoid);

    // Remaining medoids: greedy selection
    let mut min_dist_to_medoids: Vec<f64> = (0..n)
        .map(|i| distances[[i, first_medoid]])
        .collect();

    while medoids.len() < k {
        // Find point with maximum min distance to existing medoids
        let gains: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|candidate| {
                if medoids.contains(&candidate) {
                    return 0.0;
                }
                // Gain = sum of (min_dist - new_dist) for all points where new_dist < min_dist
                (0..n)
                    .map(|i| {
                        let new_dist = distances[[i, candidate]];
                        if new_dist < min_dist_to_medoids[i] {
                            min_dist_to_medoids[i] - new_dist
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect();

        let best = gains.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        medoids.push(best);

        // Update min distances
        for i in 0..n {
            min_dist_to_medoids[i] = min_dist_to_medoids[i].min(distances[[i, best]]);
        }
    }

    medoids
}

/// Compute nearest and second nearest medoid for each point.
fn compute_nearest_medoids(
    distances: &Array2<f64>,
    medoids: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let n = distances.nrows();
    let k = medoids.len();

    let results: Vec<(usize, usize, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut d1 = f64::INFINITY;
            let mut d2 = f64::INFINITY;
            let mut m1 = 0usize;
            let mut m2 = 0usize;

            for (m_idx, &medoid) in medoids.iter().enumerate() {
                let d = distances[[i, medoid]];
                if d < d1 {
                    d2 = d1;
                    m2 = m1;
                    d1 = d;
                    m1 = m_idx;
                } else if d < d2 {
                    d2 = d;
                    m2 = m_idx;
                }
            }

            (m1, m2, d1, d2)
        })
        .collect();

    let mut nearest = vec![0usize; n];
    let mut second = vec![0usize; n];
    let mut d_nearest = vec![0.0f64; n];
    let mut d_second = vec![f64::INFINITY; n];

    for (i, (m1, m2, d1, d2)) in results.into_iter().enumerate() {
        nearest[i] = m1;
        second[i] = m2;
        d_nearest[i] = d1;
        d_second[i] = d2;
    }

    (nearest, second, d_nearest, d_second)
}

// =============================================================================
// Optimized HDBSCAN
// =============================================================================

/// Optimized HDBSCAN with parallel core distance and MST computation.
pub fn hdbscan_optimized(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<(Vec<i32>, Vec<f64>, usize), String> {
    let n = data.nrows();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let mcs = min_cluster_size.unwrap_or(5).max(2);
    let ms = min_samples.unwrap_or(mcs);

    // Compute distances in parallel
    let distances = compute_pairwise_distances_parallel(&data);

    // Compute core distances in parallel
    let core_distances: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| distances[[i, j]])
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            if dists.len() >= ms {
                dists[ms - 1]
            } else {
                dists.last().cloned().unwrap_or(0.0)
            }
        })
        .collect();

    // Compute mutual reachability distances in parallel
    // Use references to avoid move issues with closures
    let distances_ref = &distances;
    let core_distances_ref = &core_distances;

    let mr_results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).map(move |j| {
                let mr = distances_ref[[i, j]]
                    .max(core_distances_ref[i])
                    .max(core_distances_ref[j]);
                (i, j, mr)
            }).collect::<Vec<_>>()
        })
        .collect();

    let mut mutual_reach = Array2::zeros((n, n));
    for (i, j, mr) in mr_results {
        mutual_reach[[i, j]] = mr;
        mutual_reach[[j, i]] = mr;
    }

    // Build MST using Prim's (optimized)
    let mst = build_mst_optimized(&mutual_reach);

    // Extract clusters
    let (labels, n_clusters) = extract_hdbscan_clusters_optimized(&mst, n, mcs);

    // Compute outlier scores
    let outlier_scores: Vec<f64> = core_distances.iter()
        .map(|&cd| cd / (cd + 1.0))
        .collect();

    Ok((labels, outlier_scores, n_clusters))
}

/// Optimized Prim's MST with better cache locality.
fn build_mst_optimized(distances: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let n = distances.nrows();
    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut min_from = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    for j in 1..n {
        min_dist[j] = distances[[0, j]];
        min_from[j] = 0;
    }

    for _ in 1..n {
        // Find minimum using parallel reduction for large n
        let (min_idx, min_val) = if n > 1000 {
            (0..n)
                .into_par_iter()
                .filter(|&j| !in_tree[j])
                .map(|j| (j, min_dist[j]))
                .reduce(|| (0, f64::INFINITY), |a, b| {
                    if a.1 < b.1 { a } else { b }
                })
        } else {
            let mut min_idx = 0;
            let mut min_val = f64::INFINITY;
            for j in 0..n {
                if !in_tree[j] && min_dist[j] < min_val {
                    min_val = min_dist[j];
                    min_idx = j;
                }
            }
            (min_idx, min_val)
        };

        in_tree[min_idx] = true;
        mst.push((min_from[min_idx], min_idx, min_val));

        // Update distances (can be parallelized for large n)
        if n > 1000 {
            let updates: Vec<(usize, f64, usize)> = (0..n)
                .into_par_iter()
                .filter(|&j| !in_tree[j] && distances[[min_idx, j]] < min_dist[j])
                .map(|j| (j, distances[[min_idx, j]], min_idx))
                .collect();

            for (j, d, from) in updates {
                min_dist[j] = d;
                min_from[j] = from;
            }
        } else {
            for j in 0..n {
                if !in_tree[j] && distances[[min_idx, j]] < min_dist[j] {
                    min_dist[j] = distances[[min_idx, j]];
                    min_from[j] = min_idx;
                }
            }
        }
    }

    mst
}

/// Extract clusters from MST using single-linkage.
fn extract_hdbscan_clusters_optimized(
    mst: &[(usize, usize, f64)],
    n: usize,
    min_cluster_size: usize,
) -> (Vec<i32>, usize) {
    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];
    let mut size: Vec<usize> = vec![1; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], size: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py { return; }

        let (smaller, larger) = if rank[px] < rank[py] { (px, py) } else { (py, px) };
        parent[smaller] = larger;
        size[larger] += size[smaller];
        if rank[px] == rank[py] {
            rank[larger] += 1;
        }
    }

    // Sort MST edges by weight
    let mut sorted_edges = mst.to_vec();
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Process edges
    for &(i, j, _) in &sorted_edges {
        union(&mut parent, &mut rank, &mut size, i, j);
    }

    // Assign cluster labels
    let mut labels = vec![-1i32; n];
    let mut root_to_cluster = std::collections::HashMap::new();
    let mut cluster_id = 0i32;

    for i in 0..n {
        let root = find(&mut parent, i);
        if size[root] >= min_cluster_size {
            let cluster = *root_to_cluster.entry(root).or_insert_with(|| {
                let id = cluster_id;
                cluster_id += 1;
                id
            });
            labels[i] = cluster;
        }
    }

    (labels, cluster_id as usize)
}

// =============================================================================
// Optimized OPTICS
// =============================================================================

/// Optimized OPTICS with parallel core distance computation.
pub fn optics_optimized(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
) -> Result<(Vec<usize>, Vec<f64>, Vec<i32>), String> {
    let n = data.nrows();

    if n == 0 {
        return Err("Empty data".to_string());
    }

    if min_samples < 1 || min_samples > n {
        return Err(format!("min_samples must be between 1 and {}", n));
    }

    let eps = max_eps.unwrap_or(f64::INFINITY);

    // Compute distances in parallel
    let distances = compute_pairwise_distances_parallel(&data);

    // Compute core distances in parallel
    let core_distances: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| distances[[i, j]])
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            if dists.len() >= min_samples {
                dists[min_samples - 1].min(eps)
            } else {
                f64::INFINITY
            }
        })
        .collect();

    // OPTICS ordering (inherently sequential but with optimized neighbor queries)
    let mut processed = vec![false; n];
    let mut ordering = Vec::with_capacity(n);
    let mut reachability = vec![f64::INFINITY; n];

    #[derive(PartialEq)]
    struct OrderedPoint {
        reachability: f64,
        index: usize,
    }

    impl Eq for OrderedPoint {}

    impl Ord for OrderedPoint {
        fn cmp(&self, other: &Self) -> Ordering {
            other.reachability.partial_cmp(&self.reachability)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for OrderedPoint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    for start in 0..n {
        if processed[start] {
            continue;
        }

        let mut seeds = BinaryHeap::new();
        processed[start] = true;
        ordering.push(start);

        if core_distances[start] <= eps {
            // Get neighbors and update reachability in parallel
            let updates: Vec<(usize, f64)> = (0..n)
                .into_par_iter()
                .filter(|&j| !processed[j] && distances[[start, j]] <= eps)
                .map(|j| {
                    let new_reach = core_distances[start].max(distances[[start, j]]);
                    (j, new_reach)
                })
                .collect();

            for (j, new_reach) in updates {
                if new_reach < reachability[j] {
                    reachability[j] = new_reach;
                    seeds.push(OrderedPoint { reachability: new_reach, index: j });
                }
            }
        }

        while let Some(OrderedPoint { index: p, .. }) = seeds.pop() {
            if processed[p] {
                continue;
            }

            processed[p] = true;
            ordering.push(p);

            if core_distances[p] <= eps {
                for j in 0..n {
                    if !processed[j] && distances[[p, j]] <= eps {
                        let new_reach = core_distances[p].max(distances[[p, j]]);
                        if new_reach < reachability[j] {
                            reachability[j] = new_reach;
                            seeds.push(OrderedPoint { reachability: new_reach, index: j });
                        }
                    }
                }
            }
        }
    }

    // Extract clusters
    let labels = extract_optics_clusters(&ordering, &reachability);

    // Reorder reachability
    let reachability_ordered: Vec<f64> = ordering.iter()
        .map(|&i| reachability[i])
        .collect();

    Ok((ordering, reachability_ordered, labels))
}

fn extract_optics_clusters(ordering: &[usize], reachability: &[f64]) -> Vec<i32> {
    let n = ordering.len();
    let mut labels = vec![-1i32; n];

    // Use median reachability as threshold
    let mut sorted_reach: Vec<f64> = reachability.iter()
        .filter(|&&r| r.is_finite())
        .cloned()
        .collect();
    sorted_reach.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let threshold = if sorted_reach.is_empty() {
        f64::INFINITY
    } else {
        sorted_reach[sorted_reach.len() / 2] * 1.5
    };

    let mut current_cluster = -1i32;
    for &idx in ordering {
        let r = reachability[idx];
        if r > threshold {
            if current_cluster >= 0 {
                current_cluster += 1;
            }
            labels[idx] = -1;
        } else {
            if current_cluster < 0 {
                current_cluster = 0;
            }
            labels[idx] = current_cluster;
        }
    }

    labels
}

// =============================================================================
// Optimized Affinity Propagation
// =============================================================================

/// Optimized Affinity Propagation with parallel message passing.
///
/// Uses parallel computation for:
/// - Similarity matrix construction
/// - Responsibility updates
/// - Availability updates
pub fn affinity_propagation_optimized(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> Result<(Vec<usize>, Vec<usize>, usize, bool), String> {
    let n = data.nrows();

    if n < 2 {
        return Err("Need at least 2 observations".to_string());
    }

    let damp = damping.unwrap_or(0.5);
    if damp < 0.5 || damp >= 1.0 {
        return Err("Damping must be between 0.5 and 1.0".to_string());
    }

    let max_iter = max_iterations.unwrap_or(200);
    let conv_iter = convergence_iterations.unwrap_or(15);

    // Compute similarity matrix in parallel (negative squared distance)
    let similarity = compute_pairwise_sq_distances_parallel(&data);
    let mut similarity = similarity.mapv(|x| -x);

    // Set preference
    let pref = preference.unwrap_or_else(|| {
        let mut sims: Vec<f64> = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sims.push(similarity[[i, j]]);
                }
            }
        }
        sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if sims.is_empty() { 0.0 } else { sims[sims.len() / 2] }
    });

    for i in 0..n {
        similarity[[i, i]] = pref;
    }

    // Initialize matrices with explicit f64 type
    let mut responsibility: Array2<f64> = Array2::zeros((n, n));
    let mut availability: Array2<f64> = Array2::zeros((n, n));

    let mut prev_exemplars = vec![false; n];
    let mut unchanged_count = 0;
    let mut n_iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // Update responsibilities in parallel
        let r_new: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n).map(|k| {
                    let max_other = (0..n)
                        .filter(|&kp| kp != k)
                        .map(|kp| availability[[i, kp]] + similarity[[i, kp]])
                        .fold(f64::NEG_INFINITY, f64::max);
                    similarity[[i, k]] - max_other
                }).collect()
            })
            .collect();

        // Apply damping
        for i in 0..n {
            for k in 0..n {
                responsibility[[i, k]] = damp * responsibility[[i, k]] + (1.0 - damp) * r_new[i][k];
            }
        }

        // Update availabilities in parallel
        let a_new: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n).map(|k| {
                    if i == k {
                        // Self-availability
                        (0..n)
                            .filter(|&ip| ip != k)
                            .map(|ip| responsibility[[ip, k]].max(0.0))
                            .sum()
                    } else {
                        let sum: f64 = (0..n)
                            .filter(|&ip| ip != i && ip != k)
                            .map(|ip| responsibility[[ip, k]].max(0.0))
                            .sum();
                        (responsibility[[k, k]] + sum).min(0.0)
                    }
                }).collect()
            })
            .collect();

        // Apply damping
        for i in 0..n {
            for k in 0..n {
                availability[[i, k]] = damp * availability[[i, k]] + (1.0 - damp) * a_new[i][k];
            }
        }

        // Check convergence
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

    // Extract exemplars and labels
    let exemplar_mask: Vec<bool> = (0..n)
        .map(|k| responsibility[[k, k]] + availability[[k, k]] > 0.0)
        .collect();

    let exemplar_indices: Vec<usize> = exemplar_mask.iter()
        .enumerate()
        .filter(|&(_, &is_ex)| is_ex)
        .map(|(i, _)| i)
        .collect();

    // Assign points to nearest exemplar in parallel
    let labels: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            if exemplar_mask[i] {
                exemplar_indices.iter().position(|&e| e == i).unwrap_or(0)
            } else {
                let mut best_ex = 0;
                let mut best_sim = f64::NEG_INFINITY;
                for (ex_idx, &ex) in exemplar_indices.iter().enumerate() {
                    if similarity[[i, ex]] > best_sim {
                        best_sim = similarity[[i, ex]];
                        best_ex = ex_idx;
                    }
                }
                best_ex
            }
        })
        .collect();

    Ok((labels, exemplar_indices, n_iterations, converged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_parallel_distances() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ];

        let dist = compute_pairwise_distances_parallel(&data.view());

        assert!((dist[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((dist[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((dist[[1, 2]] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_silhouette_optimized() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];
        let labels = vec![0, 0, 1, 1];

        let (widths, avg) = silhouette_optimized(data.view(), &labels).unwrap();

        assert_eq!(widths.len(), 4);
        assert!(avg > 0.9); // Well-separated clusters
    }

    #[test]
    fn test_kmedoids_optimized() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];

        let (labels, medoids, cost, iters) = kmedoids_optimized(data.view(), 2, Some(50), Some(42)).unwrap();

        assert_eq!(labels.len(), 4);
        assert_eq!(medoids.len(), 2);
        assert!(cost > 0.0);
    }
}
