//! K-medoids: PAM ([`kmedoids`]) and CLARA for large data ([`clara`]); R analogues `cluster::pam`, `cluster::clara`.

use crate::resource::checkpoint;
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::common::{
    compute_silhouette_avg, compute_silhouette_from_distances, try_distance_matrix,
};

/// Seed of CLARA's samples when `seed` is `None`.
pub const CLARA_DEFAULT_SEED: u64 = 42;

// =============================================================================
// K-Medoids (PAM - Partitioning Around Medoids)
// =============================================================================

/// Result of K-Medoids clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data: An
///   Introduction to Cluster Analysis". Wiley, New York.
/// - R cluster::pam documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMedoidsResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// Indices of medoid points (one per cluster)
    pub medoid_indices: Vec<usize>,
    /// Total dissimilarity (objective function value)
    pub total_dissimilarity: f64,
    /// Silhouette widths for each point
    pub silhouette_widths: Option<Vec<f64>>,
    /// Average silhouette width
    pub average_silhouette: Option<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for KMedoidsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-Medoids (PAM) Clustering Results")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Total dissimilarity: {:.4}", self.total_dissimilarity)?;
        if let Some(avg_sil) = self.average_silhouette {
            writeln!(f, "Average silhouette width: {:.4}", avg_sil)?;
        }
        writeln!(f)?;
        writeln!(f, "Medoid indices: {:?}", self.medoid_indices)?;

        // Count points per cluster
        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(
                f,
                "  Cluster {}: {} points (medoid: {})",
                i, count, self.medoid_indices[i]
            )?;
        }
        Ok(())
    }
}

/// Run K-Medoids (PAM) clustering.
///
/// K-Medoids is a more robust variant of k-means that uses actual data points
/// (medoids) as cluster centers rather than means. This makes it less sensitive
/// to outliers.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `seed` - Ignored: PAM's BUILD and SWAP phases are deterministic (as `cluster::pam`); the
///   parameter is kept for compatibility
///
/// # Algorithm
///
/// PAM's SWAP here takes the first improving swap; `cluster::pam` takes the best swap per step. Both
/// reach the same medoids on the pinned data; see `validation/ml/partitioning.md`.
///
/// 1. BUILD phase: Select k medoids using a greedy algorithm
/// 2. SWAP phase: Iteratively try swapping medoids with non-medoids
///    to reduce the total dissimilarity
///
/// # Returns
/// * `KMedoidsResult` containing cluster assignments and medoid indices
///
/// # Errors
///
/// - `k == 0` or `k > n`.
/// - `"Resource limit: k-medoids distance matrix ..."` when the `n x n` `f64`
///   distance matrix exceeds the per-allocation ceiling (or its size
///   overflows).
/// - `"Computation stopped: ..."` when the budget is cancelled or expires;
///   this is checked once per row of the distance matrix.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
pub fn kmedoids(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KMedoidsResult, String> {
    let n = data.nrows();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(100);
    if max_iter == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    crate::ml::extract::require_finite(&data, "k-medoids")?;
    let _ = seed; // documented as ignored

    // Pairwise distances: n x n, so allocated through the resource guards
    // (CLARA calls this with a caller-supplied sample size)
    let distances = try_distance_matrix(&data, "k-medoids distance matrix")?;

    // BUILD phase: Initialize medoids
    let mut medoid_indices = build_medoids(&distances, k);

    // Assign points to nearest medoids
    let mut labels = assign_to_medoids(&distances, &medoid_indices);
    let mut total_dissimilarity = compute_total_dissimilarity(&distances, &labels, &medoid_indices);

    // SWAP phase
    let mut n_iterations = 0;
    for iter in 0..max_iter {
        checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;
        let mut improved = false;

        // Try swapping each medoid with each non-medoid
        for m_idx in 0..k {
            let _current_medoid = medoid_indices[m_idx];

            for candidate in 0..n {
                // Skip if candidate is already a medoid
                if medoid_indices.contains(&candidate) {
                    continue;
                }

                // Try swap
                let mut new_medoids = medoid_indices.clone();
                new_medoids[m_idx] = candidate;

                let new_labels = assign_to_medoids(&distances, &new_medoids);
                let new_cost = compute_total_dissimilarity(&distances, &new_labels, &new_medoids);

                if new_cost < total_dissimilarity - 1e-10 {
                    medoid_indices = new_medoids;
                    labels = new_labels;
                    total_dissimilarity = new_cost;
                    improved = true;
                    break; // Restart search from first medoid
                }
            }

            if improved {
                break;
            }
        }

        if !improved {
            break;
        }
    }

    // Optionally compute silhouette
    let (silhouette_widths, average_silhouette) = if k > 1 {
        let sil = compute_silhouette_from_distances(&distances, &labels, k);
        let avg = sil.iter().sum::<f64>() / n as f64;
        (Some(sil), Some(avg))
    } else {
        (None, None)
    };

    Ok(KMedoidsResult {
        labels,
        medoid_indices,
        total_dissimilarity,
        silhouette_widths,
        average_silhouette,
        n_iterations,
        n_clusters: k,
        n,
    })
}

/// BUILD phase: Greedy medoid initialization.
fn build_medoids(distances: &Array2<f64>, k: usize) -> Vec<usize> {
    let n = distances.nrows();
    let mut medoids = Vec::with_capacity(k);

    // First medoid: point that minimizes sum of distances to all others
    let mut first_medoid = 0;
    let mut min_sum = f64::INFINITY;
    for i in 0..n {
        let sum: f64 = distances.row(i).sum();
        if sum < min_sum {
            min_sum = sum;
            first_medoid = i;
        }
    }
    medoids.push(first_medoid);

    // Remaining medoids: greedily select to minimize total dissimilarity
    while medoids.len() < k {
        let mut best_candidate = 0;
        let mut best_gain = f64::NEG_INFINITY;

        for candidate in 0..n {
            if medoids.contains(&candidate) {
                continue;
            }

            // Compute gain from adding this candidate
            let mut gain = 0.0;
            for i in 0..n {
                let current_min: f64 = medoids
                    .iter()
                    .map(|&m| distances[[i, m]])
                    .fold(f64::INFINITY, |a, b| a.min(b));
                let new_dist = distances[[i, candidate]];
                if new_dist < current_min {
                    gain += current_min - new_dist;
                }
            }

            if gain > best_gain {
                best_gain = gain;
                best_candidate = candidate;
            }
        }

        medoids.push(best_candidate);
    }

    medoids
}

/// Assign each point to its nearest medoid.
fn assign_to_medoids(distances: &Array2<f64>, medoids: &[usize]) -> Vec<usize> {
    let n = distances.nrows();
    let _k = medoids.len();

    (0..n)
        .map(|i| {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;
            for (cluster, &medoid) in medoids.iter().enumerate() {
                let dist = distances[[i, medoid]];
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster;
                }
            }
            best_cluster
        })
        .collect()
}

/// Compute total dissimilarity (sum of distances to assigned medoids).
fn compute_total_dissimilarity(
    distances: &Array2<f64>,
    labels: &[usize],
    medoids: &[usize],
) -> f64 {
    labels
        .iter()
        .enumerate()
        .map(|(i, &label)| distances[[i, medoids[label]]])
        .sum()
}

/// Convenience wrapper for kmedoids.
pub fn run_kmedoids(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<KMedoidsResult, String> {
    kmedoids(data, k, max_iterations, seed)
}

// =============================================================================
// CLARA - Clustering Large Applications
// =============================================================================

/// Result of CLARA clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data:
///   An Introduction to Cluster Analysis". Wiley.
/// - R cluster::clara documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clara.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaraResult {
    /// Cluster assignments for all observations
    pub labels: Vec<usize>,
    /// Final medoid indices (in the original data)
    pub medoid_indices: Vec<usize>,
    /// Total dissimilarity
    pub objective: f64,
    /// Best sample that was used
    pub best_sample: Vec<usize>,
    /// Average silhouette width of the best sample (as R's
    /// `clara()$silinfo$avg.width`): the final labels restricted to the
    /// sample, with the sample's own dissimilarities
    pub average_silhouette: Option<f64>,
    /// Number of samples taken
    pub n_samples: usize,
    /// Sample size used
    pub sample_size: usize,
    /// Number of clusters
    pub k: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for ClaraResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CLARA Clustering Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Number of clusters: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(
            f,
            "Samples taken: {} (size: {})",
            self.n_samples, self.sample_size
        )?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        if let Some(sil) = self.average_silhouette {
            writeln!(f, "Average silhouette: {:.4}", sil)?;
        }
        writeln!(f)?;
        writeln!(f, "Medoid indices: {:?}", self.medoid_indices)?;

        // Cluster sizes
        let mut sizes = vec![0usize; self.k];
        for &l in &self.labels {
            sizes[l] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, &size) in sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {}", i, size)?;
        }

        Ok(())
    }
}

/// CLARA (Clustering Large Applications) - PAM on samples for large datasets.
///
/// CLARA extends PAM (Partitioning Around Medoids) to large datasets by
/// running PAM on multiple random samples and keeping the best result.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `samples` - Number of samples to draw (default: 5)
/// * `sample_size` - Size of each sample (default: 40 + 2*k)
/// * `max_iter` - Maximum PAM iterations per sample (default: 100)
/// * `seed` - Random seed of the samples (default: [`CLARA_DEFAULT_SEED`])
///
/// # Returns
/// * `ClaraResult` with cluster assignments and medoids
///
/// # Errors
///
/// - `k == 0` or `k > n`, `samples == 0`, or a `sample_size` (after
///   clamping to `n`) below `k`.
/// - `"Computation stopped: ..."` when the budget is cancelled or expires
///   between samples.
/// - The [`kmedoids`] errors for a sample (its `sample_size x sample_size`
///   distance matrix over the ceiling, or a cancellation while it is
///   built), prefixed `"CLARA sample of N rows (sample_size): "`.
///
/// # Algorithm
///
/// 1. Draw `samples` random samples of size `sample_size` from the data
/// 2. Run PAM on each sample to find k medoids
/// 3. Assign all observations to nearest medoid
/// 4. Compute total dissimilarity
/// 5. Keep the result with lowest dissimilarity
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn clara(
    data: ArrayView2<f64>,
    k: usize,
    samples: Option<usize>,
    sample_size: Option<usize>,
    max_iter: Option<usize>,
    seed: Option<u64>,
) -> Result<ClaraResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 || k > n {
        return Err(format!("k must be between 1 and {}", n));
    }

    let n_samples = samples.unwrap_or(5);
    let samp_size = sample_size.unwrap_or(40 + 2 * k).min(n);
    let max_iterations = max_iter.unwrap_or(100);
    if max_iterations == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    crate::ml::extract::require_finite(&data, "CLARA")?;

    if n_samples == 0 {
        return Err("samples must be at least 1 (got 0)".to_string());
    }

    if samp_size < k {
        return Err(format!("Sample size ({}) must be >= k ({})", samp_size, k));
    }

    // Initialize RNG
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(CLARA_DEFAULT_SEED));

    let mut best_objective = f64::INFINITY;
    let mut best_medoids = Vec::new();
    let mut best_sample = Vec::new();

    // Try multiple samples
    for _ in 0..n_samples {
        checkpoint().map_err(|e| e.to_string())?;

        // Draw random sample
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices: Vec<usize> = indices.into_iter().take(samp_size).collect();

        // Create sample data
        let sample_data =
            Array2::from_shape_fn((samp_size, d), |(i, j)| data[[sample_indices[i], j]]);

        // Run PAM on sample; its samp_size x samp_size distance matrix is sized
        // by sample_size, so an error names that parameter
        let pam_result = kmedoids(sample_data.view(), k, Some(max_iterations), None)
            .map_err(|e| format!("CLARA sample of {samp_size} rows (sample_size): {e}"))?;

        // Map sample medoid indices back to original data indices
        let medoids: Vec<usize> = pam_result
            .medoid_indices
            .iter()
            .map(|&idx| sample_indices[idx])
            .collect();

        // Compute objective using all data points
        let objective = compute_clara_objective(&data, &medoids);

        if objective < best_objective {
            best_objective = objective;
            best_medoids = medoids;
            best_sample = sample_indices;
        }
    }

    // Assign all points to nearest medoid
    let labels = assign_to_medoids_data(&data, &best_medoids);

    // Silhouette of the best sample, as cluster::clara reports it: the final
    // labels restricted to the sample and the sample's own distances (an
    // O(sample_size^2) computation, independent of n)
    let sample_labels: Vec<usize> = best_sample.iter().map(|&i| labels[i]).collect();
    let best_sample_data =
        Array2::from_shape_fn((best_sample.len(), d), |(i, j)| data[[best_sample[i], j]]);
    let sample_distances =
        try_distance_matrix(&best_sample_data.view(), "CLARA sample distance matrix")?;
    let avg_silhouette = Some(compute_silhouette_avg(&sample_labels, &sample_distances, k));

    Ok(ClaraResult {
        labels,
        medoid_indices: best_medoids,
        objective: best_objective,
        best_sample,
        average_silhouette: avg_silhouette,
        n_samples,
        sample_size: samp_size,
        k,
        n,
    })
}

/// Compute CLARA objective (total dissimilarity to medoids).
fn compute_clara_objective(data: &ArrayView2<f64>, medoids: &[usize]) -> f64 {
    let n = data.nrows();
    let d = data.ncols();
    let mut total = 0.0;

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        for &med in medoids {
            let mut dist = 0.0;
            for j in 0..d {
                dist += (data[[i, j]] - data[[med, j]]).powi(2);
            }
            min_dist = min_dist.min(dist.sqrt());
        }
        total += min_dist;
    }

    total
}

/// Assign points to nearest medoid using data matrix directly.
fn assign_to_medoids_data(data: &ArrayView2<f64>, medoids: &[usize]) -> Vec<usize> {
    let n = data.nrows();
    let d = data.ncols();
    let _k = medoids.len();

    (0..n)
        .map(|i| {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;

            for (j, &med) in medoids.iter().enumerate() {
                let mut dist = 0.0;
                for l in 0..d {
                    dist += (data[[i, l]] - data[[med, l]]).powi(2);
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = j;
                }
            }

            best
        })
        .collect()
}

/// Convenience wrapper for clara.
pub fn run_clara(
    data: ArrayView2<f64>,
    k: usize,
    samples: Option<usize>,
    sample_size: Option<usize>,
    seed: Option<u64>,
) -> Result<ClaraResult, String> {
    clara(data, k, samples, sample_size, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::advanced_clustering_mod::common::compute_distance_matrix;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    const ARITH: Tol = Tol::new(1e-12, 1e-12);

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        m
    }

    /// The ap_blobs data for the property tests.
    fn ap_blobs() -> Array2<f64> {
        let path = p2a_validation::repo_path("validation/datasets/generated/ap_blobs.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        case_matrix(&ds, &["x", "y"])
    }

    #[test]
    fn test_validate_pam_blobs() {
        let (case, ds) = crate::test_support::load_case("pam_clara_blobs");
        let x = case_matrix(&ds, &["x", "y"]);
        for k in 2..=4 {
            let r = kmedoids(x.view(), k, Some(100), None).unwrap();
            let mut med: Vec<f64> = r.medoid_indices.iter().map(|&m| (m + 1) as f64).collect();
            med.sort_by(f64::total_cmp);
            let of: Vec<f64> = r
                .labels
                .iter()
                .map(|&l| (r.medoid_indices[l] + 1) as f64)
                .collect();
            let t = |s: &str| format!("pam{k}_{s}");
            Check::new(&case)
                .vector(&t("medoids"), &med, Tol::exact())
                .scalar(&t("total"), r.total_dissimilarity, ARITH)
                .vector(&t("medoid_of"), &of, Tol::exact())
                .vector(
                    &t("sil_width"),
                    r.silhouette_widths.as_ref().unwrap(),
                    ARITH,
                )
                .scalar(&t("avg_sil"), r.average_silhouette.unwrap(), ARITH)
                .finish();
        }
    }

    #[test]
    fn test_validate_clara_full_sample_is_pam() {
        // clara(samples = 1, sampsize = n, pamLike = TRUE) is PAM on a shuffled copy; at k = 3 the
        // first-improvement SWAP reaches pam's medoids under every row order tried.
        let (case, ds) = crate::test_support::load_case("pam_clara_blobs");
        let x = case_matrix(&ds, &["x", "y"]);
        let n = x.nrows();
        let r = clara(x.view(), 3, Some(1), Some(n), Some(100), Some(7)).unwrap();
        let mut med: Vec<f64> = r.medoid_indices.iter().map(|&m| (m + 1) as f64).collect();
        med.sort_by(f64::total_cmp);
        let of: Vec<f64> = r
            .labels
            .iter()
            .map(|&l| (r.medoid_indices[l] + 1) as f64)
            .collect();
        Check::new(&case)
            .vector("clara3_medoids", &med, Tol::exact())
            .scalar("clara3_objective", r.objective / n as f64, ARITH) // R reports the mean dissimilarity
            .vector("clara3_medoid_of", &of, Tol::exact())
            .scalar("clara3_avg_sil", r.average_silhouette.unwrap(), ARITH)
            .finish();
    }

    #[test]
    fn test_kmedoids_result_does_not_depend_on_the_seed() {
        // BUILD and SWAP are deterministic, as cluster::pam; the seed is documented as ignored.
        let x = ap_blobs();
        let a = kmedoids(x.view(), 3, None, Some(1)).unwrap();
        let b = kmedoids(x.view(), 3, None, Some(999)).unwrap();
        let c = kmedoids(x.view(), 3, None, None).unwrap();
        assert_eq!(
            (a.medoid_indices.clone(), a.total_dissimilarity.to_bits()),
            (b.medoid_indices.clone(), b.total_dissimilarity.to_bits())
        );
        assert_eq!(a.medoid_indices, c.medoid_indices);
    }

    #[test]
    fn test_clara_default_seed_is_reproducible() {
        let x = ap_blobs();
        let a = clara(x.view(), 3, Some(3), Some(30), None, None).unwrap();
        let b = clara(
            x.view(),
            3,
            Some(3),
            Some(30),
            None,
            Some(CLARA_DEFAULT_SEED),
        )
        .unwrap();
        assert_eq!(
            (a.best_sample.clone(), a.objective.to_bits()),
            (b.best_sample, b.objective.to_bits())
        );
    }

    #[test]
    fn test_kmedoids_rejects_bad_parameters() {
        let x = array![[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 4.9]];
        assert!(
            kmedoids(x.view(), 2, Some(0), None)
                .unwrap_err()
                .contains("max_iterations")
        );
        assert!(
            clara(x.view(), 2, Some(1), Some(4), Some(0), None)
                .unwrap_err()
                .contains("max_iterations")
        );
        let nan = array![[0.0, f64::NAN], [1.0, 1.0]];
        assert!(
            kmedoids(nan.view(), 1, None, None)
                .unwrap_err()
                .contains("finite")
        );
        assert!(
            clara(nan.view(), 1, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn three_blobs(n: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, 2), |(i, j)| {
            [[0.0, 0.0], [4.0, 4.0], [0.0, 5.0]][i % 3][j] + rng.r#gen::<f64>()
        })
    }

    #[test]
    fn clara_silhouette_is_computed_on_the_best_sample() {
        let data = three_blobs(300, 8);
        let r = clara(data.view(), 3, Some(5), Some(30), None, Some(42)).unwrap();
        assert_eq!(r.best_sample.len(), 30);
        let sample = Array2::from_shape_fn((30, 2), |(i, j)| data[[r.best_sample[i], j]]);
        let sample_labels: Vec<usize> = r.best_sample.iter().map(|&i| r.labels[i]).collect();
        let want =
            compute_silhouette_avg(&sample_labels, &compute_distance_matrix(&sample.view()), 3);
        assert_eq!(r.average_silhouette.map(f64::to_bits), Some(want.to_bits()));
    }

    #[test]
    fn clara_reports_a_silhouette_for_large_n_and_validates_samples() {
        let data = three_blobs(12_000, 9);
        let r = clara(data.view(), 3, Some(2), Some(40), Some(20), Some(1)).unwrap();
        assert!(r.average_silhouette.is_some()); // was None above 10,000
        assert!(
            clara(data.view(), 3, Some(0), None, None, Some(1))
                .unwrap_err()
                .starts_with("samples")
        );
        let e = with_budget(cancelled(), || {
            clara(data.view(), 3, Some(2), Some(40), None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn kmedoids_distance_matrix_is_fallible_and_cancellable() {
        let data = three_blobs(100, 3);
        // 100 x 100 f64 = 80,000 B
        let e = with_alloc_ceiling(Some(40_000), || kmedoids(data.view(), 3, Some(10), Some(1)))
            .unwrap_err();
        assert!(e.contains("k-medoids") && e.contains("limit"), "{e}");
        let e =
            with_budget(cancelled(), || kmedoids(data.view(), 3, Some(10), Some(1))).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        // CLARA's per-sample PAM: a caller-supplied sample_size of n needs the same n x n matrix
        let e = with_alloc_ceiling(Some(40_000), || {
            clara(data.view(), 3, Some(1), Some(100), Some(10), Some(1))
        })
        .unwrap_err();
        assert!(
            e.contains("k-medoids") && e.contains("CLARA") && e.contains("sample_size"),
            "{e}"
        );
        assert!(kmedoids(data.view(), 3, Some(10), Some(1)).is_ok());
    }

    #[test]
    fn test_kmedoids_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = kmedoids(data.view(), 2, Some(50), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_clara_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = clara(data.view(), 2, Some(3), Some(4), Some(50), Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.k, 2);
        assert_eq!(result.medoid_indices.len(), 2);
        assert!(result.objective > 0.0);

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_clara_large_sample() {
        // Test with larger sample size than data (should clamp)
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],];

        let result = clara(data.view(), 2, Some(2), Some(100), None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 3);
        assert_eq!(result.sample_size, 3); // Clamped to n
    }
}
