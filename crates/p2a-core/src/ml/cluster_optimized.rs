//! Optimized clustering algorithms using parallelization and efficient data structures.
//!
//! This module provides high-performance implementations of clustering algorithms
//! using rayon for parallelization and optimized memory access patterns.

use crate::resource;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

// =============================================================================
// Optimized Pairwise Distance Computation
// =============================================================================

/// Compute pairwise Euclidean distances with parallelization.
///
/// Uses rayon to parallelize over rows, achieving near-linear speedup. Allocates n^2/2
/// `(i, j, d)` tuples besides the matrix and is not resource-guarded; the clustering functions of
/// this module use a guarded, cancellable matrix instead.
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
            ((i + 1)..n)
                .map(move |j| {
                    let mut sum = 0.0;
                    for k in 0..d {
                        let diff = data_ref[i * d + k] - data_ref[j * d + k];
                        sum += diff * diff;
                    }
                    (i, j, sum.sqrt())
                })
                .collect::<Vec<_>>()
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
            ((i + 1)..n)
                .map(move |j| {
                    let mut sum = 0.0;
                    for k in 0..d {
                        let diff = data_ref[i * d + k] - data_ref[j * d + k];
                        sum += diff * diff;
                    }
                    (i, j, sum)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, dist) in results {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }

    distances
}

/// Pairwise Euclidean distances: the `n x n` matrix allocated through the resource guards, the
/// upper triangle computed by rows in parallel (the current budget checked per row) and mirrored
/// in tiles.
fn try_pairwise_distances_parallel(
    data: &ArrayView2<f64>,
    what: &str,
) -> Result<Array2<f64>, String> {
    let (n, d) = data.dim();
    let len = resource::checked_len(&[n, n], what).map_err(|e| e.to_string())?;
    let mut buf = resource::try_vec(len, 0.0f64, what).map_err(|e| e.to_string())?;
    if n == 0 {
        return Array2::from_shape_vec((0, 0), buf).map_err(|e| e.to_string());
    }
    let rows: Vec<f64> = {
        let mut r = resource::try_vec(
            resource::checked_len(&[n, d], what).map_err(|e| e.to_string())?,
            0.0f64,
            what,
        )
        .map_err(|e| e.to_string())?;
        for (i, row) in data.rows().into_iter().enumerate() {
            for (k, &x) in row.iter().enumerate() {
                r[i * d + k] = x;
            }
        }
        r
    };
    let budget = resource::current_budget();
    buf.par_chunks_mut(n)
        .enumerate()
        .try_for_each(|(i, out)| {
            if let Some(b) = &budget {
                b.check()?;
            }
            let a = &rows[i * d..(i + 1) * d];
            for j in (i + 1)..n {
                let b = &rows[j * d..(j + 1) * d];
                let sum: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                out[j] = sum.sqrt();
            }
            Ok::<(), crate::errors::EconError>(())
        })
        .map_err(|e| e.to_string())?;
    // mirror the upper triangle in 64 x 64 tiles
    const TILE: usize = 64;
    for bi in (0..n).step_by(TILE) {
        resource::checkpoint().map_err(|e| e.to_string())?;
        for bj in (bi..n).step_by(TILE) {
            for i in bi..(bi + TILE).min(n) {
                for j in bj.max(i + 1)..(bj + TILE).min(n) {
                    buf[j * n + i] = buf[i * n + j];
                }
            }
        }
    }
    Array2::from_shape_vec((n, n), buf).map_err(|e| e.to_string())
}

// =============================================================================
// Optimized Silhouette Coefficient
// =============================================================================

/// Optimized silhouette computation with parallelization: R `cluster::silhouette` through the one core
/// of [`crate::ml::cluster_validation::silhouette()`] (singletons and `a(i) = b(i)` score 0) on a
/// parallel distance matrix.
///
/// Refuses non-finite data, labellings without 2 to n - 1 distinct clusters (where R returns `NA`), and
/// any label of n or above (labels are 0-indexed cluster indices below n).
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

    crate::ml::advanced_clustering_mod::common::check_points(&data, "silhouette_optimized")
        .map_err(|e| e.to_string())?;
    crate::ml::cluster_validation::check_silhouette_clusters(labels, n)?;

    let distances = try_pairwise_distances_parallel(&data, "silhouette_optimized distance matrix")?;
    let widths = crate::ml::cluster_validation::silhouette_core(distances.view(), labels).widths;
    let average = widths.iter().sum::<f64>() / n as f64;

    Ok((widths, average))
}

// =============================================================================
// Optimized K-Medoids (PAM)
// =============================================================================

/// K-medoids by PAM: BUILD and best-improvement SWAP, `cluster::pam`'s algorithm, with the gains
/// computed in parallel. O(k n^2) per SWAP iteration; deterministic, `seed` is ignored (kept for
/// compatibility).
///
/// Returns the labels (index into the medoids), the medoid rows, the total cost (the sum of each
/// point's distance to its nearest medoid, R's `objective["swap"] * n`) and the number of SWAP
/// iterations. A swap is made while it lowers the cost by more than `1e-12` of the cost, so the
/// stop does not depend on the data's units; the first best swap in candidate order is taken.
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
    crate::ml::advanced_clustering_mod::common::check_points(&data, "kmedoids_optimized")
        .map_err(|e| e.to_string())?;

    let max_iter = max_iterations.unwrap_or(100);

    let distances = try_pairwise_distances_parallel(&data, "kmedoids_optimized distance matrix")?;

    // BUILD phase: Initialize medoids
    let mut medoid_indices = build_medoids_fast(&distances, k, seed)?;

    // Nearest and second nearest medoid of each point
    let (mut nearest, _, mut d_nearest, mut d_second) =
        compute_nearest_medoids(&distances, &medoid_indices);

    let mut total_cost: f64 = d_nearest.iter().sum();
    let mut n_iterations = 0;

    for iter in 0..max_iter {
        resource::checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;

        let candidates: Vec<usize> = (0..n).filter(|i| !medoid_indices.contains(i)).collect();
        let distances_ref = &distances;
        let nearest_ref = &nearest;
        let d_nearest_ref = &d_nearest;
        let d_second_ref = &d_second;
        let budget = resource::current_budget();

        // Best swap per candidate: (gain, medoid slot). A point's new distance is its distance
        // to the candidate when that is smaller than its remaining nearest medoid; the
        // candidate's own distance drops to 0.
        let best_per_candidate: Vec<(f64, usize)> = candidates
            .par_iter()
            .map(|&candidate| {
                if let Some(b) = &budget {
                    b.check()?;
                }
                let col = distances_ref.column(candidate);
                let mut best = (f64::NEG_INFINITY, 0usize);
                for m_idx in 0..k {
                    let mut gain = 0.0;
                    for i in 0..n {
                        let d_candidate = col[i];
                        let remaining = if nearest_ref[i] == m_idx {
                            d_second_ref[i]
                        } else {
                            d_nearest_ref[i]
                        };
                        gain += d_nearest_ref[i] - d_candidate.min(remaining);
                    }
                    if gain > best.0 {
                        best = (gain, m_idx);
                    }
                }
                Ok(best)
            })
            .collect::<Result<_, crate::errors::EconError>>()
            .map_err(|e| e.to_string())?;

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_swap = (0usize, 0usize); // (medoid slot, candidate)
        for (&candidate, &(gain, m_idx)) in candidates.iter().zip(&best_per_candidate) {
            if gain > best_gain {
                best_gain = gain;
                best_swap = (m_idx, candidate);
            }
        }

        if best_gain.is_nan() || best_gain <= 1e-12 * total_cost {
            break; // no swap lowers the cost
        }

        medoid_indices[best_swap.0] = best_swap.1;
        let (nn, _, dn, ds) = compute_nearest_medoids(&distances, &medoid_indices);
        nearest = nn;
        d_nearest = dn;
        d_second = ds;
        total_cost = d_nearest.iter().sum();
    }

    Ok((nearest, medoid_indices, total_cost, n_iterations))
}

/// BUILD phase with optimized first medoid selection.
fn build_medoids_fast(
    distances: &Array2<f64>,
    k: usize,
    _seed: Option<u64>,
) -> Result<Vec<usize>, String> {
    let n = distances.nrows();
    let mut medoids = Vec::with_capacity(k);

    // First medoid: point minimizing sum of distances (computed in parallel)
    let sums: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| distances.row(i).sum())
        .collect();

    // the first minimum (min_by keeps the first of equal elements)
    let first_medoid = sums
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);

    medoids.push(first_medoid);

    // Remaining medoids: greedy selection
    let mut min_dist_to_medoids: Vec<f64> = (0..n).map(|i| distances[[i, first_medoid]]).collect();

    while medoids.len() < k {
        resource::checkpoint().map_err(|e| e.to_string())?;
        // The point that lowers the summed distance to the nearest medoid most
        let gains: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|candidate| {
                if medoids.contains(&candidate) {
                    return f64::NEG_INFINITY;
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

        // the first maximum (max_by would keep the last)
        let mut best = 0;
        for (i, g) in gains.iter().enumerate() {
            if g.total_cmp(&gains[best]).is_gt() {
                best = i;
            }
        }

        medoids.push(best);

        // Update min distances
        for i in 0..n {
            min_dist_to_medoids[i] = min_dist_to_medoids[i].min(distances[[i, best]]);
        }
    }

    Ok(medoids)
}

/// Compute nearest and second nearest medoid for each point.
fn compute_nearest_medoids(
    distances: &Array2<f64>,
    medoids: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let n = distances.nrows();
    let _k = medoids.len();

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

/// HDBSCAN labels, GLOSH outlier scores and the number of clusters. Delegates to
/// [`crate::ml::hdbscan()`] (R `dbscan::hdbscan`); kept for compatibility.
pub fn hdbscan_optimized(
    data: ArrayView2<f64>,
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
) -> Result<(Vec<i32>, Vec<f64>, usize), String> {
    let r = crate::ml::hdbscan(data, min_cluster_size, min_samples)?;
    Ok((r.labels, r.outlier_scores, r.n_clusters))
}

// =============================================================================
// Optimized OPTICS
// =============================================================================

/// OPTICS ordering, reachabilities (in ordering order) and labels. Delegates to
/// [`crate::ml::optics()`] (R `dbscan::optics`; labels by `extractDBSCAN(eps_cl = max_eps)` for a
/// finite `max_eps`, none otherwise); kept for compatibility.
pub fn optics_optimized(
    data: ArrayView2<f64>,
    min_samples: usize,
    max_eps: Option<f64>,
) -> Result<(Vec<usize>, Vec<f64>, Vec<i32>), String> {
    let r = crate::ml::optics(data, min_samples, max_eps, None)?;
    Ok((r.ordering, r.reachability, r.labels))
}

// =============================================================================
// Optimized Affinity Propagation
// =============================================================================

/// Affinity propagation as `(labels, exemplar_indices, n_iterations, converged)`: the apcluster
/// port [`crate::ml::advanced_clustering_mod::affinity_propagation`] with the same arguments
/// (kept for callers of the tuple form).
pub fn affinity_propagation_optimized(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> Result<(Vec<usize>, Vec<usize>, usize, bool), String> {
    crate::ml::advanced_clustering_mod::affinity_propagation(
        data,
        preference,
        damping,
        max_iterations,
        convergence_iterations,
    )
    .map(|r| (r.labels, r.exemplar_indices, r.n_iterations, r.converged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_validate_affinity_propagation_optimized_lam9_defaults() {
        // ML-5: the optimized copy declared convergence with zero exemplars after 15 iterations at
        // the default limits; it now delegates to the apcluster port.
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("apcluster_blobs");
        let col =
            |c: &str| ndarray::Array1::from(numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0);
        let x = ndarray::stack![ndarray::Axis(1), col("x"), col("y")];
        let (labels, exemplars, _iterations, converged) =
            affinity_propagation_optimized(x.view(), None, Some(0.9), None, None).unwrap();
        let ex: Vec<f64> = exemplars.iter().map(|&e| (e + 1) as f64).collect();
        let of: Vec<f64> = labels.iter().map(|&l| (exemplars[l] + 1) as f64).collect();
        Check::new(&case)
            .vector("lam9_default_exemplars", &ex, Tol::exact())
            .scalar(
                "lam9_default_n_clusters",
                exemplars.len() as f64,
                Tol::exact(),
            )
            .vector("lam9_default_exemplar_of", &of, Tol::exact())
            .scalar(
                "lam9_default_converged",
                if converged { 1.0 } else { 0.0 },
                Tol::exact(),
            )
            .finish();
        // not pinned: iteration path (the iteration count).
    }

    #[test]
    fn test_validate_hdbscan_optimized_matches_hdbscan() {
        use crate::ml::advanced_clustering_mod::common::test_data::r_codes;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hdbscan_blobs");
        let x = crate::ml::advanced_clustering_mod::common::test_data::blobs(&ds);
        let r = crate::ml::hdbscan(x.view(), Some(5), Some(5)).unwrap();
        let (labels, outlier, k) = hdbscan_optimized(x.view(), Some(5), Some(5)).unwrap();
        assert_eq!((&labels, k), (&r.labels, 3));
        assert_eq!(
            outlier.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r.outlier_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        Check::new(&case)
            .vector("mp5_cluster", &r_codes(&labels), Tol::exact())
            .vector("mp5_outlier_scores", &outlier, Tol::stat(x.nrows()))
            .finish();
    }

    /// R: extractDBSCAN(optics(blobs_noise, eps = 0.35, minPts = 5), eps_cl = 0.35) through the
    /// delegation (case optics_blobs).
    #[test]
    fn test_validate_optics_optimized_matches_optics() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("optics_blobs");
        let x = crate::ml::advanced_clustering_mod::common::test_data::blobs(&ds);
        let (ordering, reach, labels) = optics_optimized(x.view(), 5, Some(0.35)).unwrap();
        let r = crate::ml::optics(x.view(), 5, Some(0.35), None).unwrap();
        assert_eq!((ordering.clone(), labels.clone()), (r.ordering, r.labels));
        assert_eq!(reach.len(), 115);
        Check::new(&case)
            .vector(
                "eps035_order",
                &ordering.iter().map(|&p| (p + 1) as f64).collect::<Vec<_>>(),
                Tol::exact(),
            )
            .vector(
                "eps035_dbscan_cluster",
                &crate::ml::advanced_clustering_mod::common::test_data::r_codes(&labels),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_kmedoids_and_affinity_optimized_refuse_non_finite() {
        let data = array![[0.0, 0.0], [0.1, f64::NAN], [10.0, 10.0], [10.1, 10.1]];
        assert!(
            kmedoids_optimized(data.view(), 2, Some(10), Some(1))
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(
            affinity_propagation_optimized(data.view(), None, None, None, None)
                .unwrap_err()
                .contains("finite values")
        );
    }

    /// R: cluster::silhouette on blobs_noise with a singleton cluster (case silhouette_singletons).
    #[test]
    fn test_validate_silhouette_optimized_singletons() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("silhouette_singletons");
        let x = crate::ml::advanced_clustering_mod::common::test_data::blobs(&ds);
        let n = x.nrows();
        let labels: Vec<usize> = case
            .vector("labels")
            .iter()
            .map(|&v| v as usize - 1)
            .collect();
        let (widths, avg) = silhouette_optimized(x.view(), &labels).unwrap();
        Check::new(&case)
            .vector("sil_width", &widths, Tol::coef(n))
            .scalar("avg_width", avg, Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_parallel_distances() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let dist = compute_pairwise_distances_parallel(&data.view());

        assert!((dist[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((dist[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((dist[[1, 2]] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_silhouette_optimized() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];
        let labels = vec![0, 0, 1, 1];

        let (widths, avg) = silhouette_optimized(data.view(), &labels).unwrap();

        assert_eq!(widths.len(), 4);
        assert!(avg > 0.9); // Well-separated clusters
    }

    #[test]
    fn test_silhouette_optimized_refuses_label_of_n() {
        let x = array![[0.0, 0.0], [0.13, 0.08], [10.0, 10.0], [10.11, 9.94]];
        assert!(
            silhouette_optimized(x.view(), &[0, 0, 1, 4])
                .unwrap_err()
                .contains("labels must be below n = 4 (got 4)")
        );
    }

    #[test]
    fn test_kmedoids_optimized() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        let (labels, medoids, cost, _iters) =
            kmedoids_optimized(data.view(), 2, Some(50), Some(42)).unwrap();

        assert_eq!(labels.len(), 4);
        assert_eq!(medoids.len(), 2);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_validate_kmedoids_optimized_pam() {
        // R: cluster::pam(ap_blobs, k) for k = 2, 3, 4 (case pam_clara_blobs).
        use crate::ml::advanced_clustering_mod::common::test_data::blobs;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pam_clara_blobs");
        let x = blobs(&ds);
        for k in 2..=4 {
            let (_, medoids, cost, _) = kmedoids_optimized(x.view(), k, None, None).unwrap();
            let mut med: Vec<f64> = medoids.iter().map(|&m| (m + 1) as f64).collect();
            med.sort_by(f64::total_cmp);
            Check::new(&case)
                .vector(&format!("pam{k}_medoids"), &med, Tol::exact())
                .scalar(&format!("pam{k}_total"), cost, Tol::coef(115))
                .finish();
        }
    }

    fn noisy_points(n: usize, seed: u64) -> Array2<f64> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, 2), |(i, _)| {
            (i % 5) as f64 * 3.0 + rng.r#gen::<f64>() * 2.5
        })
    }

    #[test]
    fn test_kmedoids_optimized_reported_cost_is_true_cost() {
        let x = noisy_points(200, 11);
        let d = try_pairwise_distances_parallel(&x.view(), "test").unwrap();
        let n = x.nrows();
        let cost_of = |med: &[usize]| -> f64 {
            (0..n)
                .map(|i| med.iter().map(|&m| d[[i, m]]).fold(f64::INFINITY, f64::min))
                .sum()
        };
        for seed in 0..4 {
            let (labels, med, cost, _) =
                kmedoids_optimized(x.view(), 6, Some(500), Some(seed)).unwrap();
            let truth = cost_of(&med);
            assert!((cost - truth).abs() <= 1e-12 * truth, "{cost} vs {truth}");
            for (i, &l) in labels.iter().enumerate() {
                assert_eq!(
                    d[[i, med[l]]],
                    med.iter().map(|&m| d[[i, m]]).fold(f64::INFINITY, f64::min)
                );
            }
            // a local optimum: no single swap lowers the cost
            for slot in 0..med.len() {
                for c in 0..n {
                    if med.contains(&c) {
                        continue;
                    }
                    let mut m2 = med.clone();
                    m2[slot] = c;
                    assert!(cost_of(&m2) >= truth * (1.0 - 1e-12), "swap {slot}->{c}");
                }
            }
        }
    }

    #[test]
    fn test_kmedoids_optimized_scale_free() {
        let x = noisy_points(120, 3);
        let small = x.mapv(|v| v * 2f64.powi(-40));
        let (_, a, _, _) = kmedoids_optimized(x.view(), 4, None, None).unwrap();
        let (_, b, _, _) = kmedoids_optimized(small.view(), 4, None, None).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn silhouette_optimized_distance_matrix_is_fallible_and_cancellable() {
        use crate::ml::advanced_clustering_mod::common::test_data::cancelled;
        use crate::resource::{with_alloc_ceiling, with_budget};
        let x = noisy_points(400, 5);
        let labels: Vec<usize> = (0..400).map(|i| i % 5).collect();
        let ceiling = 400 * 400 * 8 / 2;
        assert!(
            with_alloc_ceiling(Some(ceiling), || silhouette_optimized(x.view(), &labels)).is_err()
        );
        assert!(
            with_alloc_ceiling(Some(ceiling), || kmedoids_optimized(
                x.view(),
                3,
                None,
                None
            ))
            .is_err()
        );
        let e = with_budget(cancelled(), || silhouette_optimized(x.view(), &labels)).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        let e =
            with_budget(cancelled(), || kmedoids_optimized(x.view(), 3, None, None)).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn test_try_pairwise_distances_match_infallible() {
        let x = noisy_points(150, 9);
        let a = compute_pairwise_distances_parallel(&x.view());
        let b = try_pairwise_distances_parallel(&x.view(), "test").unwrap();
        assert_eq!(a, b);
    }
}
