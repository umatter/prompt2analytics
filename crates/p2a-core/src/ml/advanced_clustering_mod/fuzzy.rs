//! Fuzzy clustering: fuzzy c-means ([`fuzzy_cmeans`]; R `e1071::cmeans`) and FANNY ([`fanny`]; R `cluster::fanny`).

use crate::options::Choices;
use crate::resource::checkpoint;
use crate::traits::convergence::relative_change;
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::common::try_distance_matrix;

/// Seed of the random memberships when `seed` is `None`.
pub const FUZZY_DEFAULT_SEED: u64 = 42;

// =============================================================================
// Fuzzy C-Means Clustering
// =============================================================================

/// Result of Fuzzy C-Means clustering.
///
/// # References
///
/// - Bezdek, J.C. (1981). "Pattern Recognition with Fuzzy Objective Function
///   Algorithms". Plenum Press, New York.
/// - R e1071::cmeans documentation
///   Source: <https://www.rdocumentation.org/packages/e1071/versions/1.7-16/topics/cmeans>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyCMeansResult {
    /// Hard cluster assignments (index of maximum membership)
    pub labels: Vec<usize>,
    /// Cluster centers (k x features)
    #[serde(skip)]
    pub centers: Array2<f64>,
    /// Membership matrix (n x k) - soft assignments
    #[serde(skip)]
    pub membership: Array2<f64>,
    /// Final value of the objective function (within-cluster error)
    pub withinerror: f64,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Fuzziness parameter used
    pub m: f64,
    /// Number of observations
    pub n: usize,
    /// Did the algorithm converge?
    pub converged: bool,
}

impl std::fmt::Display for FuzzyCMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fuzzy C-Means Clustering Results")?;
        writeln!(f, "================================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Fuzziness parameter m: {:.2}", self.m)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Within-cluster error: {:.6}", self.withinerror)?;
        writeln!(f)?;

        // Count points per cluster (hard assignment)
        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }

        writeln!(f, "Cluster sizes (hard assignment):")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }

        writeln!(f)?;
        writeln!(f, "Cluster centers:")?;
        for i in 0..self.centers.nrows() {
            let center: Vec<String> = self
                .centers
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            writeln!(f, "  Cluster {}: [{}]", i, center.join(", "))?;
        }

        Ok(())
    }
}

/// Distance metric for Fuzzy C-Means.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FcmDistance {
    /// Euclidean distance (sum of squared differences)
    #[default]
    Euclidean,
    /// Manhattan distance (sum of absolute differences)
    Manhattan,
}

crate::option_enum!(
    FcmDistance,
    Choices {
        option: "distance metric",
        values: &[
            (&["euclidean", "euclid"], FcmDistance::Euclidean),
            (&["manhattan", "cityblock", "l1"], FcmDistance::Manhattan),
        ],
    }
);

/// Run Fuzzy C-Means clustering.
///
/// Not yet R-equivalent: the exponent `2/(m-1)` is applied to squared distances, so centres,
/// memberships and the objective differ from `e1071::cmeans`; the hard partition is pinned. See
/// `validation/ml/partitioning.md`.
///
/// Fuzzy C-Means is a soft clustering algorithm where each point has a degree
/// of membership to all clusters, rather than belonging to exactly one cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `m` - Fuzziness parameter (must be > 1, default: 2.0). Higher values mean softer clusters.
/// * `max_iterations` - Maximum iterations (default: 100)
/// * `tolerance` - Convergence tolerance for objective function (default: sqrt(machine epsilon))
/// * `distance` - Distance metric (euclidean or manhattan)
/// * `seed` - Optional random seed for initialization
///
/// # Algorithm
///
/// Minimizes the objective function: J = Σᵢ Σⱼ uᵢⱼᵐ dᵢⱼ
///
/// where uᵢⱼ is the membership of point i in cluster j, m is the fuzziness
/// parameter, and dᵢⱼ is the distance from point i to cluster center j.
///
/// Update rules:
/// - Membership: uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^(2/(m-1))
/// - Centers: cⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
///
/// # Returns
/// * `FuzzyCMeansResult` containing cluster assignments and membership degrees
///
/// # References
///
/// - Bezdek, J.C. (1981). "Pattern Recognition with Fuzzy Objective Function Algorithms".
pub fn fuzzy_cmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    m: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    distance: Option<FcmDistance>,
    seed: Option<u64>,
) -> Result<FuzzyCMeansResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }

    let fuzz = m.unwrap_or(2.0);
    if fuzz <= 1.0 {
        return Err("Fuzziness parameter m must be > 1".to_string());
    }

    let max_iter = max_iterations.unwrap_or(100);
    let tol = tolerance.unwrap_or(f64::EPSILON.sqrt());
    if max_iter == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!("tolerance must be finite and positive, got {tol}"));
    }
    crate::ml::extract::require_finite(&data, "fuzzy c-means")?;
    let dist_metric = distance.unwrap_or(FcmDistance::Euclidean);

    // Initialize membership matrix randomly
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(FUZZY_DEFAULT_SEED));

    let mut membership = Array2::zeros((n, n_clusters));
    for i in 0..n {
        // Random initialization, then normalize so rows sum to 1
        let mut row_sum = 0.0;
        for j in 0..n_clusters {
            membership[[i, j]] = rng.r#gen::<f64>() + 0.01;
            row_sum += membership[[i, j]];
        }
        for j in 0..n_clusters {
            membership[[i, j]] /= row_sum;
        }
    }

    // Initialize centers
    let mut centers = Array2::zeros((n_clusters, d));

    // Main FCM loop
    let mut n_iterations = 0;
    let mut prev_error = f64::INFINITY;
    let mut converged = false;
    let mut withinerror;

    for iter in 0..max_iter {
        checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;

        // Update centers: cⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
        for j in 0..n_clusters {
            let mut weight_sum = 0.0;
            let mut weighted_sum = vec![0.0; d];

            for i in 0..n {
                let w = membership[[i, j]].powf(fuzz);
                weight_sum += w;
                for k in 0..d {
                    weighted_sum[k] += w * data[[i, k]];
                }
            }

            for k in 0..d {
                centers[[j, k]] = if weight_sum > 0.0 {
                    weighted_sum[k] / weight_sum
                } else {
                    // If cluster is empty, reinitialize to a random point
                    data[[rng.gen_range(0..n), k]]
                };
            }
        }

        // Compute distances from each point to each center
        let mut distances = Array2::zeros((n, n_clusters));
        for i in 0..n {
            for j in 0..n_clusters {
                distances[[i, j]] =
                    compute_fcm_distance(&data.row(i), &centers.row(j), dist_metric);
            }
        }

        // Update membership matrix
        let exp = 2.0 / (fuzz - 1.0);
        for i in 0..n {
            // Check if point coincides with any center
            let mut at_center = None;
            for j in 0..n_clusters {
                if distances[[i, j]] == 0.0 {
                    at_center = Some(j);
                    break;
                }
            }

            if let Some(center_idx) = at_center {
                // Point is at a center - assign full membership to that cluster
                for j in 0..n_clusters {
                    membership[[i, j]] = if j == center_idx { 1.0 } else { 0.0 };
                }
            } else {
                // Standard FCM update: uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^exp
                for j in 0..n_clusters {
                    let mut sum = 0.0;
                    for k in 0..n_clusters {
                        sum += (distances[[i, j]] / distances[[i, k]]).powf(exp);
                    }
                    membership[[i, j]] = 1.0 / sum;
                }
            }
        }

        // Compute objective function: J = Σᵢ Σⱼ uᵢⱼᵐ dᵢⱼ
        withinerror = 0.0;
        for i in 0..n {
            for j in 0..n_clusters {
                withinerror += membership[[i, j]].powf(fuzz) * distances[[i, j]];
            }
        }

        // Check convergence
        if relative_change(withinerror, prev_error) < tol {
            converged = true;
            break;
        }
        prev_error = withinerror;
    }

    // Compute final distances and objective for return value
    let mut distances = Array2::zeros((n, n_clusters));
    for i in 0..n {
        for j in 0..n_clusters {
            distances[[i, j]] = compute_fcm_distance(&data.row(i), &centers.row(j), dist_metric);
        }
    }

    withinerror = 0.0;
    for i in 0..n {
        for j in 0..n_clusters {
            withinerror += membership[[i, j]].powf(fuzz) * distances[[i, j]];
        }
    }

    // Hard assignments: argmax of membership
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            let mut max_j = 0;
            let mut max_u = membership[[i, 0]];
            for j in 1..n_clusters {
                if membership[[i, j]] > max_u {
                    max_u = membership[[i, j]];
                    max_j = j;
                }
            }
            max_j
        })
        .collect();

    Ok(FuzzyCMeansResult {
        labels,
        centers,
        membership,
        withinerror,
        n_iterations,
        n_clusters,
        m: fuzz,
        n,
        converged,
    })
}

/// Compute distance for FCM.
fn compute_fcm_distance(
    a: &ndarray::ArrayView1<f64>,
    b: &ndarray::ArrayView1<f64>,
    metric: FcmDistance,
) -> f64 {
    match metric {
        FcmDistance::Euclidean => {
            // Sum of squared differences (not squared root for efficiency)
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
        }
        FcmDistance::Manhattan => {
            // Sum of absolute differences
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }
    }
}

/// Convenience wrapper for fuzzy_cmeans.
pub fn run_fuzzy_cmeans(
    data: ArrayView2<f64>,
    n_clusters: usize,
    m: Option<f64>,
    seed: Option<u64>,
) -> Result<FuzzyCMeansResult, String> {
    fuzzy_cmeans(data, n_clusters, m, None, None, None, seed)
}

// =============================================================================
// FANNY - Fuzzy Analysis Clustering
// =============================================================================

/// Result of FANNY fuzzy clustering.
///
/// # References
///
/// - Kaufman, L. and Rousseeuw, P.J. (1990). "Finding Groups in Data".
/// - R cluster::fanny documentation
///   Source: <https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/fanny.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FannyResult {
    /// Membership coefficients (n x k matrix)
    #[serde(skip)]
    pub membership: Array2<f64>,
    /// Hard cluster assignments
    pub clustering: Vec<usize>,
    /// Cluster centers (weighted by membership)
    #[serde(skip)]
    pub centers: Array2<f64>,
    /// Objective function value
    pub objective: f64,
    /// Convergence flag
    pub converged: bool,
    /// Number of iterations
    pub n_iterations: usize,
    /// Dunn's partition coefficient (1/k to 1, higher is crisper)
    pub dunn_coefficient: f64,
    /// Normalized Dunn coefficient (0 to 1)
    pub normalized_dunn: f64,
    /// Number of clusters
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Fuzziness parameter used
    pub membership_exponent: f64,
}

impl std::fmt::Display for FannyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FANNY Fuzzy Clustering Results")?;
        writeln!(f, "===============================")?;
        writeln!(f, "Number of clusters: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Fuzziness parameter: {:.2}", self.membership_exponent)?;
        writeln!(
            f,
            "Converged: {} (iterations: {})",
            self.converged, self.n_iterations
        )?;
        writeln!(f)?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        writeln!(f, "Dunn coefficient: {:.4}", self.dunn_coefficient)?;
        writeln!(f, "Normalized Dunn: {:.4}", self.normalized_dunn)?;
        writeln!(f)?;

        // Hard clustering sizes
        let mut sizes = vec![0usize; self.k];
        for &c in &self.clustering {
            sizes[c] += 1;
        }
        writeln!(f, "Hard cluster sizes: {:?}", sizes)?;

        Ok(())
    }
}

/// FANNY fuzzy clustering algorithm.
///
/// FANNY computes a fuzzy clustering of the data, where each observation
/// can belong to multiple clusters with varying degrees of membership.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
/// * `k` - Number of clusters
/// * `membership_exponent` - Fuzziness parameter m > 1 (default: 2.0)
/// * `max_iter` - Maximum iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-8)
/// * `seed` - Random seed
///
/// # Returns
/// * `FannyResult` with membership coefficients and cluster assignments
///
/// # Algorithm
///
/// FANNY minimizes the objective:
/// ∑_i ∑_v (∑_j u_iv² d(i,j) / ∑_j u_jv²) * (∑_j u_jv² / 2n_v²)
///
/// This is equivalent to fuzzy c-medoids with modified normalization.
///
/// Not yet R-equivalent: this is not `cluster::fanny`'s Kaufman-Rousseeuw iteration, so memberships,
/// coefficients and the objective differ. See `validation/ml/partitioning.md`.
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data".
pub fn fanny(
    data: ArrayView2<f64>,
    k: usize,
    membership_exponent: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FannyResult, String> {
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 || k > n {
        return Err(format!("k must be between 1 and {}", n));
    }

    let m = membership_exponent.unwrap_or(2.0);
    if m <= 1.0 {
        return Err("membership_exponent must be > 1".to_string());
    }

    let max_iterations = max_iter.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-8);
    if max_iterations == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(format!(
            "tolerance must be finite and positive, got {tolerance}"
        ));
    }
    crate::ml::extract::require_finite(&data, "FANNY")?;

    // Compute distance matrix
    let distances = try_distance_matrix(&data, "FANNY distance matrix")?;

    // Initialize RNG
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(FUZZY_DEFAULT_SEED));

    // Initialize membership matrix
    let mut membership = Array2::zeros((n, k));
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            membership[[i, j]] = rng.r#gen::<f64>() + 0.1;
            sum += membership[[i, j]];
        }
        for j in 0..k {
            membership[[i, j]] /= sum;
        }
    }

    let mut converged = false;
    let mut n_iterations = 0;
    let mut prev_objective = f64::INFINITY;

    for iter in 0..max_iterations {
        checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;

        // Update membership based on distances
        let mut new_membership = Array2::zeros((n, k));

        for i in 0..n {
            // Compute weighted distances to each cluster
            let mut cluster_dist = vec![0.0; k];
            for v in 0..k {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;
                for j in 0..n {
                    let u_jv_m = membership[[j, v]].powf(m);
                    weighted_sum += u_jv_m * distances[[i, j]];
                    weight_sum += u_jv_m;
                }
                cluster_dist[v] = if weight_sum > 1e-10 {
                    weighted_sum / weight_sum
                } else {
                    f64::INFINITY
                };
            }

            // Update membership using inverse distance weighting
            let exponent = 1.0 / (m - 1.0);
            let mut sum = 0.0;

            for v in 0..k {
                if cluster_dist[v] < 1e-10 {
                    // Point is at cluster center
                    new_membership[[i, v]] = 1.0;
                    sum = 1.0;
                    for w in 0..k {
                        if w != v {
                            new_membership[[i, w]] = 0.0;
                        }
                    }
                    break;
                }

                let mut ratio_sum = 0.0;
                for w in 0..k {
                    if cluster_dist[w] > 1e-10 {
                        ratio_sum += (cluster_dist[v] / cluster_dist[w]).powf(exponent);
                    }
                }
                new_membership[[i, v]] = if ratio_sum > 1e-10 {
                    1.0 / ratio_sum
                } else {
                    0.0
                };
                sum += new_membership[[i, v]];
            }

            // Normalize
            if sum > 1e-10 {
                for v in 0..k {
                    new_membership[[i, v]] /= sum;
                }
            }
        }

        // Compute objective
        let objective = compute_fanny_objective(&distances, &new_membership, m);

        // Check convergence
        let change: f64 = membership
            .iter()
            .zip(new_membership.iter())
            .map(|(old, new)| (*old - *new).abs())
            .sum();

        membership = new_membership;

        if change < tolerance || (prev_objective - objective).abs() < tolerance {
            converged = true;
            break;
        }
        prev_objective = objective;
    }

    // Hard cluster assignments
    let clustering: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = 0;
            for v in 1..k {
                if membership[[i, v]] > membership[[i, best]] {
                    best = v;
                }
            }
            best
        })
        .collect();

    // Compute centers (weighted by membership)
    let mut centers = Array2::zeros((k, d));
    for v in 0..k {
        let mut weight_sum = 0.0;
        for i in 0..n {
            let u_m = membership[[i, v]].powf(m);
            weight_sum += u_m;
            for j in 0..d {
                centers[[v, j]] += u_m * data[[i, j]];
            }
        }
        if weight_sum > 1e-10 {
            for j in 0..d {
                centers[[v, j]] /= weight_sum;
            }
        }
    }

    // Dunn's partition coefficient
    let dunn_coefficient: f64 = membership.iter().map(|u| u.powi(2)).sum::<f64>() / n as f64;

    // Normalized Dunn coefficient: (F - 1/k) / (1 - 1/k)
    let normalized_dunn = if k > 1 {
        (dunn_coefficient - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)
    } else {
        1.0
    };

    let objective = compute_fanny_objective(&distances, &membership, m);

    Ok(FannyResult {
        membership,
        clustering,
        centers,
        objective,
        converged,
        n_iterations,
        dunn_coefficient,
        normalized_dunn,
        k,
        n,
        membership_exponent: m,
    })
}

/// Compute FANNY objective function.
fn compute_fanny_objective(distances: &Array2<f64>, membership: &Array2<f64>, m: f64) -> f64 {
    let n = membership.nrows();
    let k = membership.ncols();

    let mut objective = 0.0;

    for v in 0..k {
        let mut numer_sum = 0.0;
        let mut denom_sum = 0.0;

        for i in 0..n {
            let u_iv_m = membership[[i, v]].powf(m);
            for j in 0..n {
                let u_jv_m = membership[[j, v]].powf(m);
                numer_sum += u_iv_m * u_jv_m * distances[[i, j]];
            }
            denom_sum += u_iv_m;
        }

        if denom_sum > 1e-10 {
            objective += numer_sum / (2.0 * denom_sum);
        }
    }

    objective
}

/// Convenience wrapper for fanny.
pub fn run_fanny(
    data: ArrayView2<f64>,
    k: usize,
    membership_exponent: Option<f64>,
    max_iter: Option<usize>,
    seed: Option<u64>,
) -> Result<FannyResult, String> {
    fanny(data, k, membership_exponent, max_iter, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_fcm_cmeans_blobs_labels() {
        // deviation ML-9: p2a applies 2/(m-1) to squared distances, so centres, memberships and
        // withinerror differ from e1071::cmeans; the hard partition agrees. Only the partition is
        // pinned.
        let (case, ds) = crate::test_support::load_case("fuzzy_reference_blobs");
        let mut x = Array2::zeros((ds.nrows(), 2));
        for (j, c) in ["x", "y"].iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            x.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        let r = fuzzy_cmeans(
            x.view(),
            3,
            Some(2.0),
            Some(10_000),
            Some(1e-12),
            None,
            Some(5),
        )
        .unwrap();
        assert!(r.converged);
        let mut o: Vec<usize> = (0..3).collect();
        o.sort_by(|&a, &b| r.centers[[a, 0]].total_cmp(&r.centers[[b, 0]]));
        let mut pos = [0usize; 3];
        for (p, &k) in o.iter().enumerate() {
            pos[k] = p;
        }
        let labels: Vec<f64> = r.labels.iter().map(|&l| (pos[l] + 1) as f64).collect();
        Check::new(&case)
            .vector("cmeans_cluster", &labels, Tol::exact())
            .finish();
    }

    #[test]
    fn test_fuzzy_cmeans_stopping_rule_is_scale_free() {
        // The old rule |dJ| / (|J| + 1e-10) stopped at once when J < 1e-10 (data in small units).
        let x = Array2::from_shape_fn((30, 2), |(i, j)| {
            [[0.0, 0.0], [3.0, 0.0], [1.5, 2.5]][i % 3][j] + 0.3 * ((i * 7 + j * 3) as f64).sin()
        });
        let base = fuzzy_cmeans(x.view(), 3, None, Some(500), None, None, Some(1)).unwrap();
        let c = 2f64.powi(-30);
        let small = fuzzy_cmeans((&x * c).view(), 3, None, Some(500), None, None, Some(1)).unwrap();
        assert_eq!(small.n_iterations, base.n_iterations);
        assert_eq!(small.labels, base.labels);
    }

    #[test]
    fn test_fcm_distance_parses_aliases_and_rejects_unknown() {
        FcmDistance::choices().validate().unwrap();
        assert_eq!(
            "Euclidean".parse::<FcmDistance>().unwrap(),
            FcmDistance::Euclidean
        );
        assert_eq!(
            "cityblock".parse::<FcmDistance>().unwrap(),
            FcmDistance::Manhattan
        );
        assert_eq!("l1".parse::<FcmDistance>().unwrap(), FcmDistance::Manhattan);
        assert!(
            "cosine"
                .parse::<FcmDistance>()
                .unwrap_err()
                .to_string()
                .contains("expected one of: euclidean, manhattan")
        );
    }

    #[test]
    fn test_fuzzy_methods_reject_bad_parameters_and_default_seeds_are_fixed() {
        let x = Array2::from_shape_fn((12, 2), |(i, j)| {
            (i % 3) as f64 * 4.0 + 0.1 * ((i + j) as f64).sin()
        });
        assert!(
            fuzzy_cmeans(x.view(), 2, None, Some(0), None, None, None)
                .unwrap_err()
                .contains("max_iterations")
        );
        assert!(
            fuzzy_cmeans(x.view(), 2, None, None, Some(0.0), None, None)
                .unwrap_err()
                .contains("tolerance")
        );
        assert!(
            fanny(x.view(), 2, None, Some(0), None, None)
                .unwrap_err()
                .contains("max_iterations")
        );
        assert!(
            fanny(x.view(), 2, None, None, Some(f64::NAN), None)
                .unwrap_err()
                .contains("tolerance")
        );
        let mut nan = x.clone();
        nan[[0, 0]] = f64::NAN;
        assert!(
            fuzzy_cmeans(nan.view(), 2, None, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
        assert!(
            fanny(nan.view(), 2, None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
        let a = fuzzy_cmeans(x.view(), 2, None, None, None, None, None).unwrap();
        let b = fuzzy_cmeans(
            x.view(),
            2,
            None,
            None,
            None,
            None,
            Some(FUZZY_DEFAULT_SEED),
        )
        .unwrap();
        assert_eq!(a.withinerror.to_bits(), b.withinerror.to_bits());
        let f1 = fanny(x.view(), 2, None, None, None, None).unwrap();
        let f2 = fanny(x.view(), 2, None, None, None, Some(FUZZY_DEFAULT_SEED)).unwrap();
        assert_eq!(f1.objective.to_bits(), f2.objective.to_bits());
    }

    #[test]
    fn test_fuzzy_cmeans_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fuzzy_cmeans(data.view(), 2, None, None, None, None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.membership.nrows(), 6);
        assert_eq!(result.membership.ncols(), 2);

        // Check membership rows sum to 1
        for i in 0..6 {
            let row_sum: f64 = (0..2).map(|j| result.membership[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} sum: {}", i, row_sum);
        }

        // Check points are separated correctly
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_fuzzy_cmeans_fuzziness() {
        let data = array![
            [0.0, 0.0],
            [5.0, 0.0], // Point equidistant from both clusters
            [10.0, 0.0],
        ];

        // With low fuzziness (near 1), should be crisper
        let result_crisp =
            fuzzy_cmeans(data.view(), 2, Some(1.5), None, None, None, Some(42)).unwrap();

        // With high fuzziness, should be fuzzier
        let result_fuzzy =
            fuzzy_cmeans(data.view(), 2, Some(3.0), None, None, None, Some(42)).unwrap();

        // The middle point should have more equal membership in fuzzy case
        let mid_idx = 1;
        let membership_crisp = result_crisp.membership.row(mid_idx);
        let membership_fuzzy = result_fuzzy.membership.row(mid_idx);

        // Variance of memberships should be lower for fuzzier clustering
        let var_crisp: f64 = membership_crisp
            .iter()
            .map(|&x| (x - 0.5).powi(2))
            .sum::<f64>()
            / 2.0;
        let var_fuzzy: f64 = membership_fuzzy
            .iter()
            .map(|&x| (x - 0.5).powi(2))
            .sum::<f64>()
            / 2.0;

        // For a point in the middle, the fuzzy version should have memberships closer to 0.5
        assert!(
            var_fuzzy <= var_crisp + 0.1,
            "Fuzzy version should be fuzzier: var_crisp={}, var_fuzzy={}",
            var_crisp,
            var_fuzzy
        );
    }

    #[test]
    fn test_fanny_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = fanny(data.view(), 2, Some(2.0), Some(100), Some(1e-6), Some(42)).unwrap();

        assert_eq!(result.clustering.len(), 6);
        assert_eq!(result.k, 2);
        assert_eq!(result.membership.nrows(), 6);
        assert_eq!(result.membership.ncols(), 2);

        // Membership should sum to 1 for each observation
        for i in 0..6 {
            let row_sum: f64 = (0..2).map(|j| result.membership[[i, j]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Membership row {} sum: {}",
                i,
                row_sum
            );
        }

        // Dunn coefficient should be in valid range
        assert!(result.dunn_coefficient >= 1.0 / result.k as f64);
        assert!(result.dunn_coefficient <= 1.0);
        assert!(result.normalized_dunn >= 0.0 && result.normalized_dunn <= 1.0);

        // Hard assignments should separate the clusters
        assert_eq!(result.clustering[0], result.clustering[1]);
        assert_eq!(result.clustering[3], result.clustering[4]);
        assert_ne!(result.clustering[0], result.clustering[3]);
    }

    #[test]
    fn test_fanny_fuzziness_parameter() {
        let data = array![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0],];

        // Higher m = fuzzier (memberships closer to 1/k)
        let result_low_m = fanny(data.view(), 2, Some(1.5), Some(100), None, Some(42)).unwrap();
        let result_high_m = fanny(data.view(), 2, Some(3.0), Some(100), None, Some(42)).unwrap();

        // Higher m should have lower Dunn coefficient (fuzzier)
        assert!(
            result_high_m.dunn_coefficient <= result_low_m.dunn_coefficient + 0.1,
            "Higher m should produce fuzzier clustering"
        );
    }
}
