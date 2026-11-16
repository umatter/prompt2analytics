//! Fuzzy clustering: fuzzy c-means ([`fuzzy_cmeans`]; R `e1071::cmeans`) and FANNY ([`fanny`]; R `cluster::fanny`).

use crate::options::Choices;
use crate::resource::{checked_len, checkpoint, try_vec};
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

/// Run Fuzzy C-Means clustering from random memberships.
///
/// The iteration is `e1071::cmeans`'s; the start is not: e1071 starts from `k` random rows as
/// centres, this function from random memberships (seeded) and then the centres they give. Use
/// [`fuzzy_cmeans_from_centers`] for `cmeans(x, centers = <matrix>)`. With Euclidean distance a
/// random start reaches e1071's fixed point (checked on `validation/ml/partitioning.md`'s data);
/// with Manhattan distance the weighted-median iteration stops at a start-dependent fixed point,
/// in R as here.
///
/// Fuzzy C-Means is a soft clustering algorithm where each point has a degree
/// of membership to all clusters, rather than belonging to exactly one cluster.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `m` - Fuzziness parameter (finite and above 1, default: 2.0). Higher values mean softer clusters.
/// * `max_iterations` - Maximum iterations (default: 100, e1071's `iter.max`)
/// * `tolerance` - Relative change of the objective at which the iteration stops (default:
///   sqrt(machine epsilon), e1071's `reltol`)
/// * `distance` - Distance metric (euclidean or manhattan)
/// * `seed` - Optional random seed for initialization
///
/// # Algorithm
///
/// Minimizes the objective J = Σᵢ wᵢ Σⱼ uᵢⱼᵐ dᵢⱼ with e1071's observation weights wᵢ = 1/n,
/// where dᵢⱼ is the *squared* Euclidean distance or the L1 distance from point i to centre j.
///
/// Update rules (e1071::cmeans):
/// - Membership: uᵢⱼ = dᵢⱼ^(-1/(m-1)) / Σₖ dᵢₖ^(-1/(m-1)), computed as 1 / Σₖ (dᵢⱼ/dᵢₖ)^(1/(m-1));
///   a point at zero distance from some centres shares its membership equally among them.
/// - Centres, Euclidean: cⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ.
/// - Centres, Manhattan: per coordinate, the lower uᵢⱼᵐ-weighted median.
///
/// `withinerror` is J, the objective with weights 1/n (e1071's `withinerror`).
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
    if n_clusters == 0 {
        return Err("n_clusters must be at least 1".to_string());
    }
    if n_clusters > n {
        return Err(format!(
            "n_clusters ({}) cannot exceed n_samples ({})",
            n_clusters, n
        ));
    }
    let cfg = FcmConfig::new(m, max_iterations, tolerance, distance)?;
    crate::ml::extract::require_finite(&data, "fuzzy c-means")?;

    // Random memberships, rows normalised to 1, then the centres they give.
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(FUZZY_DEFAULT_SEED));
    let mut membership = try_zeros2(n, n_clusters, "fuzzy c-means memberships")?;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n_clusters {
            membership[[i, j]] = rng.r#gen::<f64>() + 0.01;
            row_sum += membership[[i, j]];
        }
        for j in 0..n_clusters {
            membership[[i, j]] /= row_sum;
        }
    }
    let sorted = FcmSorted::new(&data, cfg.metric)?;
    let mut centers = try_zeros2(n_clusters, data.ncols(), "fuzzy c-means centres")?;
    update_fcm_centers(&data, &membership, cfg.m, &sorted, &mut centers);
    fcm_iterate(&data, centers, membership, &cfg, &sorted)
}

/// Fuzzy c-means from given initial centres: `e1071::cmeans(x, centers = <matrix>)`.
///
/// The memberships are first computed from `centers` (k x features; finite, distinct, with the
/// data's columns), then the iteration of [`fuzzy_cmeans`] runs. Arguments and defaults as there.
pub fn fuzzy_cmeans_from_centers(
    data: ArrayView2<f64>,
    centers: ArrayView2<f64>,
    m: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    distance: Option<FcmDistance>,
) -> Result<FuzzyCMeansResult, String> {
    let n = data.nrows();
    let k = centers.nrows();
    if k == 0 {
        return Err("centers must have at least 1 row".to_string());
    }
    if centers.ncols() != data.ncols() {
        return Err(format!(
            "centers must have the data's {} columns, got {}",
            data.ncols(),
            centers.ncols()
        ));
    }
    if k > n {
        return Err(format!("n_clusters ({k}) cannot exceed n_samples ({n})"));
    }
    let cfg = FcmConfig::new(m, max_iterations, tolerance, distance)?;
    crate::ml::extract::require_finite(&data, "fuzzy c-means")?;
    crate::ml::extract::require_finite(&centers, "fuzzy c-means centers")?;
    for a in 0..k {
        for b in 0..a {
            if centers.row(a) == centers.row(b) {
                return Err("Initial centers are not distinct.".to_string());
            }
        }
    }
    let sorted = FcmSorted::new(&data, cfg.metric)?;
    let mut distances = try_zeros2(n, k, "fuzzy c-means distances")?;
    let centers = centers.to_owned();
    fcm_distances(&data, &centers, cfg.metric, &mut distances);
    let mut membership = try_zeros2(n, k, "fuzzy c-means memberships")?;
    update_fcm_memberships(&distances, cfg.m, &mut membership);
    let mut centers = centers;
    update_fcm_centers(&data, &membership, cfg.m, &sorted, &mut centers);
    fcm_iterate(&data, centers, membership, &cfg, &sorted)
}

/// Validated parameters of the fuzzy c-means iteration.
struct FcmConfig {
    m: f64,
    max_iter: usize,
    tol: f64,
    metric: FcmDistance,
}

impl FcmConfig {
    fn new(
        m: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        distance: Option<FcmDistance>,
    ) -> Result<Self, String> {
        let m = m.unwrap_or(2.0);
        if !m.is_finite() || m <= 1.0 {
            return Err(format!("m must be finite and above 1, got {m}"));
        }
        let max_iter = max_iterations.unwrap_or(100);
        let tol = tolerance.unwrap_or(f64::EPSILON.sqrt());
        if max_iter == 0 {
            return Err("max_iterations must be at least 1, got 0".to_string());
        }
        if !tol.is_finite() || tol <= 0.0 {
            return Err(format!("tolerance must be finite and positive, got {tol}"));
        }
        Ok(FcmConfig {
            m,
            max_iter,
            tol,
            metric: distance.unwrap_or(FcmDistance::Euclidean),
        })
    }
}

/// Each column's row order (Manhattan only): the weighted medians are one pass over it.
struct FcmSorted {
    order: Vec<usize>,
}

impl FcmSorted {
    fn new(data: &ArrayView2<f64>, metric: FcmDistance) -> Result<Self, String> {
        if metric != FcmDistance::Manhattan {
            return Ok(FcmSorted { order: Vec::new() });
        }
        let (n, d) = data.dim();
        let what = "fuzzy c-means column order";
        let len = checked_len(&[n, d], what).map_err(|e| e.to_string())?;
        let mut order = try_vec(len, 0usize, what).map_err(|e| e.to_string())?;
        for (j, col) in order.chunks_mut(n.max(1)).enumerate().take(d) {
            for (i, o) in col.iter_mut().enumerate() {
                *o = i;
            }
            col.sort_by(|&a, &b| data[[a, j]].total_cmp(&data[[b, j]]));
        }
        Ok(FcmSorted { order })
    }
}

fn try_zeros2(r: usize, c: usize, what: &str) -> Result<Array2<f64>, String> {
    crate::resource::try_zeros2(r, c, what).map_err(|e| e.to_string())
}

/// The shared loop: centres are current for `membership`; iterate distances -> memberships ->
/// objective -> centres until the objective's relative change is below the tolerance.
fn fcm_iterate(
    data: &ArrayView2<f64>,
    mut centers: Array2<f64>,
    mut membership: Array2<f64>,
    cfg: &FcmConfig,
    sorted: &FcmSorted,
) -> Result<FuzzyCMeansResult, String> {
    let n = data.nrows();
    let k = centers.nrows();
    let mut distances = try_zeros2(n, k, "fuzzy c-means distances")?;
    let mut prev_error = f64::INFINITY;
    let mut withinerror = f64::NAN;
    let mut converged = false;
    let mut n_iterations = 0;
    for iter in 0..cfg.max_iter {
        checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;
        if iter > 0 {
            update_fcm_centers(data, &membership, cfg.m, sorted, &mut centers);
        }
        fcm_distances(data, &centers, cfg.metric, &mut distances);
        update_fcm_memberships(&distances, cfg.m, &mut membership);
        withinerror = fcm_objective(&distances, &membership, cfg.m);
        if relative_change(withinerror, prev_error) < cfg.tol {
            converged = true;
            break;
        }
        prev_error = withinerror;
    }

    // Hard assignments: the first cluster of maximal membership (R's which.max).
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = 0;
            for j in 1..k {
                if membership[[i, j]] > membership[[i, best]] {
                    best = j;
                }
            }
            best
        })
        .collect();

    Ok(FuzzyCMeansResult {
        labels,
        centers,
        membership,
        withinerror,
        n_iterations,
        n_clusters: k,
        m: cfg.m,
        n,
        converged,
    })
}

fn fcm_distances(
    data: &ArrayView2<f64>,
    centers: &Array2<f64>,
    metric: FcmDistance,
    out: &mut Array2<f64>,
) {
    for i in 0..data.nrows() {
        for j in 0..centers.nrows() {
            out[[i, j]] = compute_fcm_distance(&data.row(i), &centers.row(j), metric);
        }
    }
}

/// `u_ij = 1 / sum_k (d_ij / d_ik)^(1/(m-1))`; a point at zero distance from z centres gets 1/z on
/// each of them (e1071).
fn update_fcm_memberships(distances: &Array2<f64>, m: f64, membership: &mut Array2<f64>) {
    let exp = 1.0 / (m - 1.0);
    let k = distances.ncols();
    for i in 0..distances.nrows() {
        let zeros = (0..k).filter(|&j| distances[[i, j]] == 0.0).count();
        if zeros > 0 {
            let share = 1.0 / zeros as f64;
            for j in 0..k {
                membership[[i, j]] = if distances[[i, j]] == 0.0 { share } else { 0.0 };
            }
            continue;
        }
        for j in 0..k {
            let mut sum = 0.0;
            for l in 0..k {
                sum += (distances[[i, j]] / distances[[i, l]]).powf(exp);
            }
            membership[[i, j]] = 1.0 / sum;
        }
    }
}

/// `J = sum_i sum_j u_ij^m d_ij / n`.
fn fcm_objective(distances: &Array2<f64>, membership: &Array2<f64>, m: f64) -> f64 {
    let n = distances.nrows();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..distances.ncols() {
            sum += membership[[i, j]].powf(m) * distances[[i, j]];
        }
    }
    sum / n as f64
}

/// Centres from memberships: u^m-weighted means (Euclidean) or lower u^m-weighted medians
/// (Manhattan). A cluster whose memberships are all zero keeps its centre.
fn update_fcm_centers(
    data: &ArrayView2<f64>,
    membership: &Array2<f64>,
    m: f64,
    sorted: &FcmSorted,
    centers: &mut Array2<f64>,
) {
    let (n, d) = data.dim();
    for j in 0..centers.nrows() {
        let w: Vec<f64> = (0..n).map(|i| membership[[i, j]].powf(m)).collect();
        let total: f64 = w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            continue;
        }
        if sorted.order.is_empty() {
            for c in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += w[i] * data[[i, c]];
                }
                centers[[j, c]] = s / total;
            }
        } else {
            let half = total / 2.0;
            for c in 0..d {
                let order = &sorted.order[c * n..(c + 1) * n];
                let mut acc = 0.0;
                let mut med = data[[order[n - 1], c]];
                for &i in order {
                    acc += w[i];
                    if acc >= half {
                        med = data[[i, c]];
                        break;
                    }
                }
                centers[[j, c]] = med;
            }
        }
    }
}

/// Distance for FCM: squared Euclidean or L1.
fn compute_fcm_distance(
    a: &ndarray::ArrayView1<f64>,
    b: &ndarray::ArrayView1<f64>,
    metric: FcmDistance,
) -> f64 {
    match metric {
        FcmDistance::Euclidean => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
        FcmDistance::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
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

/// FANNY fuzzy clustering algorithm (Kaufman & Rousseeuw; R `cluster::fanny`).
///
/// FANNY computes a fuzzy clustering of the data, where each observation
/// can belong to multiple clusters with varying degrees of membership.
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features); Euclidean dissimilarities
/// * `k` - Number of clusters, in `1..=n/2 - 1` (R's bound)
/// * `membership_exponent` - Fuzziness parameter r > 1 (default: 2.0, R's `memb.exp`)
/// * `max_iter` - Maximum number of sweeps (default: 500, R's `maxit`)
/// * `tol` - Relative change of the objective at which the iteration stops (default: 1e-15, R's `tol`)
/// * `seed` - `None`: R's start (`iniMem.p = NULL`): membership 0.9 in the object's block cluster
///   and `0.1/(k-1)` elsewhere, with blocks of `n / k` consecutive rows and the last block taking
///   the rest. `Some(seed)`: random memberships normalised to 1 (not R's).
///
/// # Returns
/// * `FannyResult` with membership coefficients and cluster assignments
///
/// # Algorithm
///
/// FANNY minimizes the objective
/// C = ∑_v ∑_{i,j} u_iv^r u_jv^r d(i,j) / (2 ∑_j u_jv^r)
/// by `cluster::fanny`'s Gauss-Seidel iteration: objects are visited in row order, and object m's
/// memberships are set from the current values (including the rows already updated in this
/// sweep), with w = u^r,
/// a_v = ∑_{l≠m} w_lv d(m,l) / ∑_l w_lv − ∑_{l,h} w_lv w_hv d(l,h) / (2 (∑_l w_lv)²)
/// and u_mv = a_v^(−1/(r−1)) / ∑_w a_w^(−1/(r−1)) (if some a_v ≤ 0, which needs coinciding
/// objects, the memberships are shared equally among those clusters). The iteration stops after a
/// sweep when |C_old − C_new| ≤ tol · C_new.
///
/// The Dunn coefficient is F = ∑ u² / n and its normalised form (kF − 1)/(k − 1). `centers` are
/// u^r-weighted means of the data (FANNY itself has no centres).
///
/// # References
///
/// - Kaufman & Rousseeuw (1990). "Finding Groups in Data", chapter 4.
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

    if k == 0 || k > (n / 2).saturating_sub(1) {
        return Err(format!(
            "'k' (number of clusters) must be in {{1,2, .., n/2 -1}} (n = {n}, k = {k})"
        ));
    }

    let m = membership_exponent.unwrap_or(2.0);
    if !m.is_finite() || m <= 1.0 {
        return Err(format!(
            "membership_exponent must be finite and above 1, got {m}"
        ));
    }

    let max_iterations = max_iter.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-15);
    if max_iterations == 0 {
        return Err("max_iterations must be at least 1, got 0".to_string());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(format!(
            "tolerance must be finite and positive, got {tolerance}"
        ));
    }
    crate::ml::extract::require_finite(&data, "FANNY")?;

    let distances = try_distance_matrix(&data, "FANNY distance matrix")?;

    let mut membership = try_zeros2(n, k, "FANNY memberships")?;
    match seed {
        None => {
            // R's start: blocks of n %/% k rows, the last block takes the rest.
            let block = n / k;
            let other = if k > 1 { 0.1 / (k - 1) as f64 } else { 0.0 };
            for i in 0..n {
                let own = (i / block).min(k - 1);
                for v in 0..k {
                    membership[[i, v]] = if k == 1 {
                        1.0
                    } else if v == own {
                        0.9
                    } else {
                        other
                    };
                }
            }
        }
        Some(s) => {
            let mut rng = StdRng::seed_from_u64(s);
            for i in 0..n {
                let mut sum = 0.0;
                for v in 0..k {
                    membership[[i, v]] = rng.r#gen::<f64>() + 0.1;
                    sum += membership[[i, v]];
                }
                for v in 0..k {
                    membership[[i, v]] /= sum;
                }
            }
        }
    }

    let mut w = membership.mapv(|u| u.powf(m));
    let (mut esp, mut ef) = fanny_sums(&distances, &w);
    let mut objective = fanny_objective_from_sums(&esp, &ef);
    let exponent = 1.0 / (m - 1.0);
    let mut converged = false;
    let mut n_iterations = 0;
    let mut dd = vec![0.0; k];
    let mut a = vec![0.0; k];

    for iter in 0..max_iterations {
        checkpoint().map_err(|e| e.to_string())?;
        n_iterations = iter + 1;
        for obj in 0..n {
            if obj % 256 == 255 {
                checkpoint().map_err(|e| e.to_string())?;
            }
            let row = distances.row(obj);
            dd.fill(0.0);
            for (l, &dist) in row.iter().enumerate() {
                if l != obj {
                    for v in 0..k {
                        dd[v] += w[[l, v]] * dist;
                    }
                }
            }
            for v in 0..k {
                // A cluster without weight cannot take the object.
                a[v] = if esp[v] > 0.0 {
                    dd[v] / esp[v] - ef[v] / (2.0 * esp[v] * esp[v])
                } else {
                    f64::INFINITY
                };
            }
            let nonpositive = a.iter().filter(|&&x| x <= 0.0).count();
            if nonpositive > 0 {
                let share = 1.0 / nonpositive as f64;
                for v in 0..k {
                    membership[[obj, v]] = if a[v] <= 0.0 { share } else { 0.0 };
                }
            } else {
                // u_v = 1 / sum_w (a_v / a_w)^(1/(r-1)): scale-free form of a_v^(-1/(r-1)) / sum.
                for v in 0..k {
                    if a[v].is_infinite() {
                        membership[[obj, v]] = 0.0;
                        continue;
                    }
                    let mut sum = 0.0;
                    for x in &a {
                        sum += (a[v] / x).powf(exponent);
                    }
                    membership[[obj, v]] = 1.0 / sum;
                }
            }
            for v in 0..k {
                let w_new = membership[[obj, v]].powf(m);
                let delta = w_new - w[[obj, v]];
                esp[v] += delta;
                ef[v] += 2.0 * delta * dd[v];
                w[[obj, v]] = w_new;
            }
        }
        // Exact sums once per sweep: the running updates drift by rounding.
        (esp, ef) = fanny_sums(&distances, &w);
        let new_objective = fanny_objective_from_sums(&esp, &ef);
        let change = (objective - new_objective).abs();
        objective = new_objective;
        if change <= tolerance * new_objective {
            converged = true;
            break;
        }
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

    // Centres (u^r-weighted means)
    let mut centers = try_zeros2(k, d, "FANNY centres")?;
    for v in 0..k {
        let weight_sum = esp[v];
        if weight_sum > 0.0 {
            for i in 0..n {
                for j in 0..d {
                    centers[[v, j]] += w[[i, v]] * data[[i, j]];
                }
            }
            for j in 0..d {
                centers[[v, j]] /= weight_sum;
            }
        }
    }

    // Dunn's partition coefficient and its normalised form (k F - 1) / (k - 1)
    let dunn_coefficient: f64 = membership.iter().map(|u| u.powi(2)).sum::<f64>() / n as f64;
    let normalized_dunn = if k > 1 {
        (k as f64 * dunn_coefficient - 1.0) / (k as f64 - 1.0)
    } else {
        1.0
    };

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

/// `esp_v = sum_l w_lv` and `ef_v = sum_{l,h} w_lv w_hv d(l,h)` (both orders of each pair).
fn fanny_sums(distances: &Array2<f64>, w: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (n, k) = w.dim();
    let mut esp = vec![0.0; k];
    let mut ef = vec![0.0; k];
    let mut row = vec![0.0; k];
    for l in 0..n {
        row.fill(0.0);
        for h in (l + 1)..n {
            let dist = distances[[l, h]];
            for v in 0..k {
                row[v] += w[[h, v]] * dist;
            }
        }
        for v in 0..k {
            esp[v] += w[[l, v]];
            ef[v] += 2.0 * w[[l, v]] * row[v];
        }
    }
    (esp, ef)
}

/// `sum_v ef_v / (2 esp_v)`.
fn fanny_objective_from_sums(esp: &[f64], ef: &[f64]) -> f64 {
    esp.iter()
        .zip(ef)
        .filter(|(e, _)| **e > 0.0)
        .map(|(e, f)| f / (2.0 * e))
        .sum()
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

    /// ap_blobs (x, y) as an n x 2 matrix.
    fn ap_blobs(ds: &crate::data::Dataset) -> Array2<f64> {
        let mut x = Array2::zeros((ds.nrows(), 2));
        for (j, c) in ["x", "y"].iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            x.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        x
    }

    /// Cluster order by the first coordinate of the centres: `pos[label]` = rank.
    fn order_by_x(centers: &Array2<f64>) -> Vec<usize> {
        let k = centers.nrows();
        let mut o: Vec<usize> = (0..k).collect();
        o.sort_by(|&a, &b| centers[[a, 0]].total_cmp(&centers[[b, 0]]));
        let mut pos = vec![0usize; k];
        for (p, &c) in o.iter().enumerate() {
            pos[c] = p;
        }
        pos
    }

    /// Centres (x-ordered, column-major as R's `vec`), memberships (column-major, clusters in that
    /// order) and 1-based labels, as the case stores them.
    fn fcm_as_r(r: &FuzzyCMeansResult) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let pos = order_by_x(&r.centers);
        let k = r.centers.nrows();
        let mut inv = vec![0usize; k];
        for (c, &p) in pos.iter().enumerate() {
            inv[p] = c;
        }
        let centers: Vec<f64> = (0..r.centers.ncols())
            .flat_map(|j| inv.iter().map(move |&c| (c, j)))
            .map(|(c, j)| r.centers[[c, j]])
            .collect();
        let membership: Vec<f64> = inv
            .iter()
            .flat_map(|&c| r.membership.column(c).to_vec())
            .collect();
        let labels = r.labels.iter().map(|&l| (pos[l] + 1) as f64).collect();
        (centers, membership, labels)
    }

    /// ap_blobs.csv for the property tests (no R values read).
    fn ap_blobs_file() -> Array2<f64> {
        let path = p2a_validation::repo_path("validation/datasets/generated/ap_blobs.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        ap_blobs(&ds)
    }

    fn start_centers() -> Array2<f64> {
        array![[0.0, 0.0], [3.0, 0.0], [1.5, 2.5]]
    }

    #[test]
    fn test_validate_fcm_cmeans_blobs_labels() {
        // The hard partition from a random start (seed 5) is e1071's.
        let (case, ds) = crate::test_support::load_case("fuzzy_reference_blobs");
        let x = ap_blobs(&ds);
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
        let (_, _, labels) = fcm_as_r(&r);
        Check::new(&case)
            .vector("cmeans_cluster", &labels, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_fcm_cmeans_blobs() {
        // e1071::cmeans(x, centers = start, m = 2, control = list(reltol = 1e-15)).
        let (case, ds) = crate::test_support::load_case("fuzzy_reference_blobs");
        let x = ap_blobs(&ds);
        let r = fuzzy_cmeans_from_centers(
            x.view(),
            start_centers().view(),
            Some(2.0),
            Some(10_000),
            Some(1e-15),
            None,
        )
        .unwrap();
        assert!(r.converged);
        let (centers, membership, labels) = fcm_as_r(&r);
        Check::new(&case)
            .vector("cmeans_centers", &centers, Tol::coef(115))
            .vector("cmeans_membership", &membership, Tol::coef(115))
            .scalar("cmeans_withinerror", r.withinerror, Tol::coef(115))
            .vector("cmeans_cluster", &labels, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_fcm_cmeans_from_start() {
        // e1071::cmeans from the same start, Euclidean and Manhattan, m = 2 and 1.5.
        let (case, ds) = crate::test_support::load_case("fcm_cmeans_from_start");
        let x = ap_blobs(&ds);
        let mut check = Check::new(&case);
        for (dist, tag) in [
            (FcmDistance::Euclidean, "euc"),
            (FcmDistance::Manhattan, "man"),
        ] {
            for (m, mtag) in [(2.0, "m2"), (1.5, "m15")] {
                let r = fuzzy_cmeans_from_centers(
                    x.view(),
                    start_centers().view(),
                    Some(m),
                    Some(10_000),
                    Some(1e-15),
                    Some(dist),
                )
                .unwrap();
                let (centers, membership, labels) = fcm_as_r(&r);
                let key = |q: &str| format!("{tag}_{mtag}_{q}");
                check
                    .vector(&key("centers"), &centers, Tol::coef(115))
                    .vector(&key("membership"), &membership, Tol::coef(115))
                    .scalar(&key("withinerror"), r.withinerror, Tol::coef(115))
                    .vector(&key("cluster"), &labels, Tol::exact());
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_fcm_random_start_reaches_cmeans_fixed_point() {
        // In R, 20 random point starts and 20 random-membership starts reach e1071's Euclidean
        // fixed point (max centre difference 8.9e-9); a random p2a start does too.
        let (case, ds) = crate::test_support::load_case("fcm_cmeans_from_start");
        let x = ap_blobs(&ds);
        let r = fuzzy_cmeans(
            x.view(),
            3,
            Some(2.0),
            Some(10_000),
            Some(1e-15),
            None,
            Some(5),
        )
        .unwrap();
        let (centers, _, _) = fcm_as_r(&r);
        for (a, b) in centers.iter().zip(case.vector("euc_m2_centers")) {
            assert!((a - b).abs() < 1e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn test_fcm_manhattan_centres_are_weighted_medians() {
        let x = ap_blobs_file();
        for m in [2.0, 1.5] {
            let r = fuzzy_cmeans(
                x.view(),
                3,
                Some(m),
                Some(500),
                Some(1e-12),
                Some(FcmDistance::Manhattan),
                Some(3),
            )
            .unwrap();
            // Lower u^m-weighted median at the returned memberships, as R's
            // x[which(cumsum(w) >= sum(w) / 2)[1]] on sorted x.
            for c in 0..3 {
                for j in 0..2 {
                    let mut o: Vec<usize> = (0..x.nrows()).collect();
                    o.sort_by(|&a, &b| x[[a, j]].total_cmp(&x[[b, j]]));
                    let w: Vec<f64> = o.iter().map(|&i| r.membership[[i, c]].powf(m)).collect();
                    let half = w.iter().sum::<f64>() / 2.0;
                    let mut acc = 0.0;
                    let mut med = f64::NAN;
                    for (p, &wi) in w.iter().enumerate() {
                        acc += wi;
                        if acc >= half {
                            med = x[[o[p], j]];
                            break;
                        }
                    }
                    // The centres were computed from the memberships of the iteration before the
                    // last; at convergence they are the medians of the final ones.
                    assert_eq!(
                        r.centers[[c, j]].to_bits(),
                        med.to_bits(),
                        "m {m} c {c} j {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fcm_from_centers_validates_centres() {
        let x = array![[0.0, 0.1], [1.0, 0.9], [5.0, 5.2], [5.1, 4.9]];
        let e = fuzzy_cmeans_from_centers(
            x.view(),
            array![[0.0, 0.0], [0.0, 0.0]].view(),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(e.contains("Initial centers are not distinct."), "{e}");
        let e = fuzzy_cmeans_from_centers(
            x.view(),
            array![[0.0], [1.0]].view(),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(e.contains("columns"), "{e}");
        let e = fuzzy_cmeans_from_centers(
            x.view(),
            array![[0.0, f64::NAN], [1.0, 1.0]].view(),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(e.contains("finite"), "{e}");
        let e = fuzzy_cmeans_from_centers(
            x.view(),
            Array2::zeros((0, 2)).view(),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(e.contains("at least 1"), "{e}");
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
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0] {
            let e = fuzzy_cmeans(x.view(), 2, Some(bad), None, None, None, None).unwrap_err();
            assert!(e.contains("m must be finite and above 1"), "{e}");
            let e = fanny(x.view(), 2, Some(bad), None, None, None).unwrap_err();
            assert!(
                e.contains("membership_exponent must be finite and above 1"),
                "{e}"
            );
        }
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
        // fanny's `None` is R's deterministic start; `Some(seed)` a seeded random start.
        let f1 = fanny(x.view(), 2, None, None, None, None).unwrap();
        let f2 = fanny(x.view(), 2, None, None, None, None).unwrap();
        assert_eq!(f1.objective.to_bits(), f2.objective.to_bits());
        let s1 = fanny(x.view(), 2, None, None, None, Some(3)).unwrap();
        let s2 = fanny(x.view(), 2, None, None, None, Some(3)).unwrap();
        assert_eq!(s1.objective.to_bits(), s2.objective.to_bits());
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
        let data = Array2::from_shape_fn((12, 2), |(i, j)| {
            (i % 3) as f64 * 5.0 + 0.4 * ((i * 5 + j * 11) as f64).sin()
        });

        // Higher r = fuzzier (memberships closer to 1/k)
        let result_low_m = fanny(data.view(), 2, Some(1.5), Some(100), None, None).unwrap();
        let result_high_m = fanny(data.view(), 2, Some(3.0), Some(100), None, None).unwrap();
        assert!(
            result_high_m.dunn_coefficient < result_low_m.dunn_coefficient,
            "Higher r should produce fuzzier clustering"
        );
    }

    /// Memberships (column-major, clusters by mean x of the hard clusters), coefficients,
    /// objective and 1-based clustering, as the case stores them.
    fn fanny_as_r(x: &Array2<f64>, r: &FannyResult) -> (Vec<f64>, Vec<f64>) {
        let k = r.k;
        let mut mean_x = vec![0.0; k];
        let mut cnt = vec![0usize; k];
        for (i, &c) in r.clustering.iter().enumerate() {
            mean_x[c] += x[[i, 0]];
            cnt[c] += 1;
        }
        let mut o: Vec<usize> = (0..k).collect();
        o.sort_by(|&a, &b| (mean_x[a] / cnt[a] as f64).total_cmp(&(mean_x[b] / cnt[b] as f64)));
        let mut pos = vec![0usize; k];
        for (p, &c) in o.iter().enumerate() {
            pos[c] = p;
        }
        let membership = o
            .iter()
            .flat_map(|&c| r.membership.column(c).to_vec())
            .collect();
        let clustering = r.clustering.iter().map(|&c| (pos[c] + 1) as f64).collect();
        (membership, clustering)
    }

    // Memberships at Tol::iterative(1e-6): R and p2a both stop at tol 1e-15 on the objective's
    // rounding floor, where memberships still differ by up to 3.3e-8 between an incremental and a
    // recomputed objective (measured in R); 1e-6 is 30x that spread.
    #[test]
    fn test_validate_fanny_blobs() {
        let (case, ds) = crate::test_support::load_case("fuzzy_reference_blobs");
        let x = ap_blobs(&ds);
        let r = fanny(x.view(), 3, Some(2.0), Some(5000), Some(1e-15), None).unwrap();
        assert!(r.converged);
        let (membership, clustering) = fanny_as_r(&x, &r);
        Check::new(&case)
            .vector("fanny_membership", &membership, Tol::iterative(1e-6))
            .vector(
                "fanny_coeff",
                &[r.dunn_coefficient, r.normalized_dunn],
                Tol::iterative(1e-6),
            )
            .scalar("fanny_objective", r.objective, Tol::stat(115))
            .vector("fanny_clustering", &clustering, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_fanny_blobs_more() {
        let (case, ds) = crate::test_support::load_case("fanny_blobs_more");
        let x = ap_blobs(&ds);
        let mut check = Check::new(&case);
        for (k, m, tag) in [(2, 2.0, "k2_m2"), (3, 1.5, "k3_m15")] {
            let r = fanny(x.view(), k, Some(m), Some(5000), Some(1e-15), None).unwrap();
            assert!(r.converged);
            let (membership, clustering) = fanny_as_r(&x, &r);
            check
                .vector(
                    &format!("{tag}_membership"),
                    &membership,
                    Tol::iterative(1e-6),
                )
                .vector(
                    &format!("{tag}_coeff"),
                    &[r.dunn_coefficient, r.normalized_dunn],
                    Tol::iterative(1e-6),
                )
                .scalar(&format!("{tag}_objective"), r.objective, Tol::stat(115))
                .vector(&format!("{tag}_clustering"), &clustering, Tol::exact());
        }
        check.finish();
    }

    #[test]
    fn test_fanny_one_sweep_matches_r_maxit_1() {
        // R 4.3.3, cluster 2.1.6: fanny(ap_blobs, 3, memb.exp = 2, maxit = 1, tol = 1e-15)$membership[c(1, 2, 40, 100), ]
        // (one Gauss-Seidel sweep from R's start; a Jacobi sweep is 0.023 off).
        let x = ap_blobs_file();
        let r = fanny(x.view(), 3, Some(2.0), Some(1), Some(1e-15), None).unwrap();
        assert_eq!(r.n_iterations, 1);
        let expected = [
            (
                0,
                [
                    0.768847336840140505,
                    0.110585342432085301,
                    0.120567320727774041,
                ],
            ),
            (
                1,
                [
                    0.856119797375445635,
                    0.073844229018634633,
                    0.070035973605919746,
                ],
            ),
            (
                39,
                [
                    0.768714031174599355,
                    0.101047554654872515,
                    0.130238414170528005,
                ],
            ),
            (
                99,
                [
                    0.078185263772046959,
                    0.087187085462902519,
                    0.834627650765050411,
                ],
            ),
        ];
        for (row, want) in expected {
            for v in 0..3 {
                let got = r.membership[[row, v]];
                assert!(
                    (got - want[v]).abs() <= 1e-13,
                    "row {row} v {v}: {got} vs {}",
                    want[v]
                );
            }
        }
    }

    #[test]
    fn test_fanny_is_scale_free() {
        let x = ap_blobs_file();
        let base = fanny(x.view(), 3, None, None, Some(1e-10), None).unwrap();
        let small = fanny(
            (&x * 2f64.powi(-30)).view(),
            3,
            None,
            None,
            Some(1e-10),
            None,
        )
        .unwrap();
        assert_eq!(small.n_iterations, base.n_iterations);
        for (a, b) in small.membership.iter().zip(base.membership.iter()) {
            assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn test_fanny_refuses_k_above_half_n() {
        let x = Array2::from_shape_fn((9, 2), |(i, j)| {
            (i as f64) + 0.3 * ((i + 2 * j) as f64).sin()
        });
        // n = 9: n %/% 2 - 1 = 3
        assert!(fanny(x.view(), 3, None, None, None, None).is_ok());
        let e = fanny(x.view(), 4, None, None, None, None).unwrap_err();
        assert!(
            e.contains("'k' (number of clusters) must be in {1,2, .., n/2 -1}"),
            "{e}"
        );
        assert!(fanny(x.view(), 0, None, None, None, None).is_err());
    }

    #[test]
    fn test_fanny_objective_is_the_criterion_at_the_returned_memberships() {
        let x = ap_blobs_file();
        let r = fanny(x.view(), 3, Some(1.7), None, None, None).unwrap();
        let n = x.nrows();
        let mut obj = 0.0;
        for v in 0..3 {
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..n {
                let wi = r.membership[[i, v]].powf(1.7);
                den += wi;
                for j in 0..n {
                    let dist =
                        ((x[[i, 0]] - x[[j, 0]]).powi(2) + (x[[i, 1]] - x[[j, 1]]).powi(2)).sqrt();
                    num += wi * r.membership[[j, v]].powf(1.7) * dist;
                }
            }
            obj += num / (2.0 * den);
        }
        assert!(
            (obj - r.objective).abs() <= 1e-12 * obj,
            "{obj} vs {}",
            r.objective
        );
    }

    #[test]
    fn test_fuzzy_cmeans_and_fanny_are_cancellable() {
        use crate::ml::advanced_clustering_mod::common::test_data::cancelled;
        use crate::resource::with_budget;
        let x = ndarray::Array2::from_shape_fn((60, 2), |(i, j)| {
            (i % 3) as f64 * 4.0 + 0.3 * ((i * 7 + j * 3) as f64).sin() + 0.1 * j as f64
        });
        let e = with_budget(cancelled(), || {
            fuzzy_cmeans(x.view(), 3, None, None, None, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        let e =
            with_budget(cancelled(), || fanny(x.view(), 3, None, None, None, None)).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }
}
