//! Affinity propagation ([`affinity_propagation`]); a port of R's `apcluster` 1.4.14.
//!
//! Affinity propagation is defined by its message passing and stopping rule, so this is a port of
//! `apclusterC.cpp` and of the post-processing in `apcluster`'s R code, for the similarities
//! `negDistMat(x, r = 2)`. See `validation/ml/affinity_propagation.md`.

use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_zeros2};
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

// =============================================================================
// Affinity Propagation
// =============================================================================

/// Result of affinity propagation clustering.
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages Between
///   Data Points". Science, 315, 972-976.
/// - Bodenhofer, U., Kothmeier, A. and Hochreiter, S. (2011). "APCluster: an R package for
///   affinity propagation clustering". Bioinformatics, 27, 2463-2464.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityPropagationResult {
    /// Cluster assignments for each point (0-indexed; `exemplar_indices[label]` is its exemplar)
    pub labels: Vec<usize>,
    /// Indices of exemplar (cluster center) points, ascending
    pub exemplar_indices: Vec<usize>,
    /// Number of iterations run
    pub n_iterations: usize,
    /// Whether a stable, non-empty exemplar set was reached before the limit
    pub converged: bool,
    /// Sum over all points of the similarity to its exemplar, the preferences of the exemplars
    /// included (apcluster's `netsim`)
    pub net_similarity: f64,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
    /// The preference used: the given one, or the median off-diagonal similarity
    #[serde(default)]
    pub preference: f64,
    /// The damping used (apcluster's `lam`)
    #[serde(default)]
    pub damping: f64,
    /// What the fit could not do
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for AffinityPropagationResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for AffinityPropagationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Affinity Propagation Results")?;
        writeln!(f, "============================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Damping: {:.2}", self.damping)?;
        writeln!(f, "Preference: {:.4}", self.preference)?;
        writeln!(f, "Net similarity: {:.4}", self.net_similarity)?;
        writeln!(f)?;
        writeln!(f, "Exemplar indices: {:?}", self.exemplar_indices)?;
        writeln!(f)?;

        if self.n_clusters == 0 || self.exemplar_indices.len() != self.n_clusters {
            writeln!(f, "No clusters (no exemplars)")?;
        } else {
            let mut cluster_counts = vec![0usize; self.n_clusters];
            for &label in &self.labels {
                if let Some(c) = cluster_counts.get_mut(label) {
                    *c += 1;
                }
            }
            writeln!(f, "Cluster sizes:")?;
            for (i, count) in cluster_counts.iter().enumerate() {
                writeln!(
                    f,
                    "  Cluster {}: {} points (exemplar: {})",
                    i, count, self.exemplar_indices[i]
                )?;
            }
        }
        write!(f, "{}", notes_text(self))
    }
}

/// The type-7 median of the upper-triangle off-diagonal similarities, by selection. S is exactly
/// symmetric, so each value of the full off-diagonal set occurs twice and the two medians agree.
fn median_similarity(similarity: &Array2<f64>) -> EconResult<f64> {
    let n = similarity.nrows();
    let len = checked_len(&[n, n - 1], "affinity propagation preference")? / 2;
    let mut upper = try_vec(len, 0.0f64, "affinity propagation preference")?;
    let mut at = 0;
    for (i, row) in similarity.rows().into_iter().enumerate() {
        for v in row.iter().skip(i + 1) {
            upper[at] = *v;
            at += 1;
        }
    }
    // Type 7 at 0.5: positions floor((len - 1) / 2) and ceil((len - 1) / 2).
    let hi = len / 2;
    if len % 2 == 1 {
        let (_, m, _) = upper.select_nth_unstable_by(hi, f64::total_cmp);
        return Ok(*m);
    }
    let (below, m, _) = upper.select_nth_unstable_by(hi, f64::total_cmp);
    let upper_mid = *m;
    let lower_mid = below
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(upper_mid);
    Ok(quantile_sorted(
        &[lower_mid, upper_mid],
        0.5,
        QuantileType::T7,
    ))
}

/// Each point's cluster: the first `q` that maximises `S[i, exemplars[q]]`, with each exemplar in
/// its own cluster (apcluster's `max.col(ties = "first")` step).
fn assign(similarity: &Array2<f64>, exemplars: &[usize], labels: &mut [usize]) {
    for (row, label) in similarity.rows().into_iter().zip(labels.iter_mut()) {
        let mut best = 0;
        let mut best_s = row[exemplars[0]];
        for (q, &e) in exemplars.iter().enumerate().skip(1) {
            if row[e] > best_s {
                best_s = row[e];
                best = q;
            }
        }
        *label = best;
    }
    for (q, &e) in exemplars.iter().enumerate() {
        labels[e] = q;
    }
}

/// Run affinity propagation as `apcluster(negDistMat(r = 2), x, q = 0.5, nonoise = TRUE)` does.
///
/// `damping` is apcluster's `lam`, `max_iterations` its `maxits`, `convergence_iterations` its
/// `convits`. The defaults are 0.5/200/15 (scikit-learn's); apcluster's are 0.9/1000/100. p2a adds
/// no noise to the similarities (`nonoise = TRUE`).
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `preference` - Preference for each point to be an exemplar (default: the median off-diagonal
///   similarity, apcluster's `q = 0.5`)
/// * `damping` - Damping factor (in `[0.5, 1)`, default: 0.5)
/// * `max_iterations` - Maximum iterations (default: 200)
/// * `convergence_iterations` - Iterations with the same non-empty exemplar set for convergence
///   (default: 15)
///
/// # Algorithm
///
/// 1. Similarities S: negative squared Euclidean distances, the preference on the diagonal.
/// 2. Iteratively update the responsibilities and availabilities, damped.
/// 3. Stop when, over the last `convergence_iterations` iterations, each point's exemplar decision
///    `A[k,k] + R[k,k] > 0` did not change and at least one point is an exemplar.
/// 4. Assign each point to its most similar exemplar, replace each exemplar with the member that
///    maximises the within-cluster similarity sum, and assign again (apcluster's post-processing).
///
/// # Returns
/// * `AffinityPropagationResult` containing cluster assignments and exemplars. A run stopped at
///   `max_iterations` with exemplars reports `converged = false` and a warning. With no exemplar at
///   that point the result is an error (apcluster returns an empty clustering).
///
/// # References
///
/// - Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages".
pub fn affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> Result<AffinityPropagationResult, String> {
    affinity_propagation_impl(
        data,
        preference,
        damping,
        max_iterations,
        convergence_iterations,
    )
    .map_err(|e| e.to_string())
}

fn affinity_propagation_impl(
    data: ArrayView2<'_, f64>,
    preference: Option<f64>,
    damping: Option<f64>,
    max_iterations: Option<usize>,
    convergence_iterations: Option<usize>,
) -> EconResult<AffinityPropagationResult> {
    let spec = |message: String| EconError::InvalidSpecification { message };
    let n = data.nrows();
    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "affinity propagation".into(),
        });
    }
    crate::ml::extract::require_finite(&data, "affinity propagation").map_err(spec)?;
    let lam = damping.unwrap_or(0.5);
    if !lam.is_finite() || !(0.5..1.0).contains(&lam) {
        return Err(spec(format!("damping must be in [0.5, 1), got {lam}")));
    }
    let maxits = max_iterations.unwrap_or(200);
    if maxits == 0 {
        return Err(spec("max_iterations must be at least 1, got 0".into()));
    }
    let convits = convergence_iterations.unwrap_or(15);
    if convits == 0 {
        return Err(spec(
            "convergence_iterations must be at least 1, got 0".into(),
        ));
    }
    if let Some(p) = preference {
        if !p.is_finite() {
            return Err(spec(format!("preference must be finite, got {p}")));
        }
    }

    // Similarities: symmetric by construction.
    let mut s = try_zeros2(n, n, "affinity propagation similarities")?;
    for i in 0..n {
        checkpoint()?;
        let xi = data.row(i);
        for j in 0..i {
            let sq: f64 = xi
                .iter()
                .zip(data.row(j).iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            s[[i, j]] = -sq;
            s[[j, i]] = -sq;
        }
    }
    if s.iter().any(|v| !v.is_finite()) {
        return Err(spec(
            "affinity propagation: the data are too large in magnitude for their squared \
             distances to be finite"
                .into(),
        ));
    }
    let pref = match preference {
        Some(p) => p,
        None => median_similarity(&s)?,
    };
    s.diag_mut().fill(pref);

    let mut r = try_zeros2(n, n, "affinity propagation responsibilities")?;
    let mut a = try_zeros2(n, n, "affinity propagation availabilities")?;
    let ring = checked_len(&[n, convits], "affinity propagation decision buffer")?;
    let mut decisions: Vec<u8> = try_vec(ring, 0u8, "affinity propagation decision buffer")?;
    let mut counts: Vec<usize> = try_vec(n, 0usize, "affinity propagation decision counts")?;
    let mut aux: Vec<f64> = try_vec(n, 0.0f64, "affinity propagation column sums")?;

    let rest = 1.0 - lam;
    let mut iterations = 0;
    let mut converged = false;
    let mut unstable = 0usize;

    for i in 0..maxits {
        checkpoint()?;
        // Responsibilities.
        for ((row_s, row_a), mut row_r) in s.rows().into_iter().zip(a.rows()).zip(r.rows_mut()) {
            let (mut max1, mut max2, mut index) = (-f64::MAX, -f64::MAX, 0);
            for (j, (sv, av)) in row_s.iter().zip(row_a.iter()).enumerate() {
                let v = av + sv;
                if v > max1 {
                    max2 = max1;
                    max1 = v;
                    index = j;
                } else if v > max2 {
                    max2 = v;
                }
            }
            for (j, (rv, sv)) in row_r.iter_mut().zip(row_s.iter()).enumerate() {
                let other = if j == index { max2 } else { max1 };
                *rv = (rest * (sv - other) + lam * *rv).min(f64::MAX);
            }
        }
        // Availabilities, row by row: per column the additions run in the row order of
        // apclusterC's column loop, so the values are the same.
        aux.fill(0.0);
        for (j, row_r) in r.rows().into_iter().enumerate() {
            for (ii, (sum, rv)) in aux.iter_mut().zip(row_r.iter()).enumerate() {
                *sum += if *rv < 0.0 && j != ii { 0.0 } else { *rv };
            }
        }
        for (j, (row_r, mut row_a)) in r.rows().into_iter().zip(a.rows_mut()).enumerate() {
            for (ii, ((av, rv), sum)) in row_a
                .iter_mut()
                .zip(row_r.iter())
                .zip(aux.iter())
                .enumerate()
            {
                let rp = if *rv < 0.0 && j != ii { 0.0 } else { *rv };
                let mut new = sum - rp;
                if new > 0.0 && j != ii {
                    new = 0.0;
                }
                *av = rest * new + lam * *av;
            }
        }
        // Decisions: the ring buffer of the last `convits` exemplar masks.
        let slot = i % convits;
        let mut any_exemplar = false;
        unstable = 0;
        for ii in 0..n {
            let ex = usize::from(a[[ii, ii]] + r[[ii, ii]] > 0.0);
            let cell = &mut decisions[ii * convits + slot];
            counts[ii] = counts[ii] - usize::from(*cell) + ex;
            *cell = ex as u8;
            if counts[ii] > 0 && counts[ii] < convits {
                unstable += 1;
            }
            any_exemplar |= ex == 1;
        }
        iterations = i + 1;
        // ML-5: an empty exemplar set never converges.
        if iterations >= convits && unstable == 0 && any_exemplar {
            converged = true;
            break;
        }
    }

    // apcluster's post-processing.
    let mut exemplars: Vec<usize> = (0..n).filter(|&k| a[[k, k]] + r[[k, k]] > 0.0).collect();
    if exemplars.is_empty() {
        return Err(EconError::ConvergenceFailure {
            iterations,
            last_change: f64::NAN,
            suggestion: "no exemplar emerged; increase max_iterations, raise the preference, or \
                         use a damping nearer 0.5 (apcluster returns an empty clustering here)"
                .into(),
        });
    }
    let mut labels = try_vec(n, 0usize, "affinity propagation labels")?;
    assign(&s, &exemplars, &mut labels);
    // Refinement: the first member that maximises the sum of similarities within its cluster.
    for (q, exemplar) in exemplars.iter_mut().enumerate() {
        checkpoint()?;
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == q).collect();
        let mut best = members[0];
        let mut best_sum = f64::NEG_INFINITY;
        for &j in &members {
            let total: f64 = members.iter().map(|&i| s[[i, j]]).sum();
            if total > best_sum {
                best_sum = total;
                best = j;
            }
        }
        *exemplar = best;
    }
    assign(&s, &exemplars, &mut labels);
    // Ascending exemplars, as `levels(factor(idx))`.
    let mut order: Vec<usize> = (0..exemplars.len()).collect();
    order.sort_by_key(|&q| exemplars[q]);
    let mut position = vec![0usize; exemplars.len()];
    for (new, &old) in order.iter().enumerate() {
        position[old] = new;
    }
    let sorted: Vec<usize> = order.iter().map(|&q| exemplars[q]).collect();
    for l in &mut labels {
        *l = position[*l];
    }
    let net_similarity: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| s[[i, sorted[l]]])
        .sum();

    let convergence = Convergence::new(
        converged,
        iterations,
        maxits,
        &format!("points with an unstable exemplar decision over the last {convits} iterations"),
        unstable as f64,
        0.0,
    );
    let mut warnings = Vec::new();
    if let Some(note) = convergence.note("Affinity propagation") {
        warnings.push(note);
    }

    Ok(AffinityPropagationResult {
        labels,
        n_clusters: sorted.len(),
        exemplar_indices: sorted,
        n_iterations: iterations,
        converged,
        net_similarity,
        n,
        preference: pref,
        damping: lam,
        warnings,
        convergence,
    })
}

/// Convenience wrapper for affinity_propagation.
pub fn run_affinity_propagation(
    data: ArrayView2<f64>,
    preference: Option<f64>,
    damping: Option<f64>,
) -> Result<AffinityPropagationResult, String> {
    affinity_propagation(data, preference, damping, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Case, Check, Tol};

    const ARITH: Tol = Tol::new(1e-12, 1e-12);

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        m
    }

    fn check_ap(case: &Case, tag: &str, r: &AffinityPropagationResult) {
        let k = |s: &str| format!("{tag}_{s}");
        let ex: Vec<f64> = r.exemplar_indices.iter().map(|&e| (e + 1) as f64).collect();
        let of: Vec<f64> = r
            .labels
            .iter()
            .map(|&l| (r.exemplar_indices[l] + 1) as f64)
            .collect();
        Check::new(case)
            .vector(&k("exemplars"), &ex, Tol::exact())
            .scalar(&k("n_clusters"), r.n_clusters as f64, Tol::exact())
            .vector(&k("exemplar_of"), &of, Tol::exact())
            .scalar(&k("netsim"), r.net_similarity, ARITH)
            .scalar(
                &k("converged"),
                if r.converged { 1.0 } else { 0.0 },
                Tol::exact(),
            )
            .scalar("preference", r.preference, ARITH)
            .finish();
        // not pinned: iteration path (the iteration count).
    }

    #[test]
    fn test_validate_apcluster_lam9() {
        // Before: the same loop without apcluster's exemplar refinement: exemplar 20 instead of
        // 18 and a lower net similarity.
        let (case, ds) = crate::test_support::load_case("apcluster_blobs");
        let x = case_matrix(&ds, &["x", "y"]);
        let r = affinity_propagation(x.view(), None, Some(0.9), Some(2000), Some(100)).unwrap();
        check_ap(&case, "lam9", &r);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn test_affinity_propagation_reports_the_iteration_limit() {
        // A window longer than the limit cannot be filled, whatever the rounding. The case's lam5
        // non-convergence in R is not pinned: at damping 0.5 these data converge or oscillate
        // depending on the last bit of the preference (see validation/ml/affinity_propagation.md).
        let path = p2a_validation::repo_path("validation/datasets/generated/ap_blobs.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let x = case_matrix(&ds, &["x", "y"]);
        let r = affinity_propagation(x.view(), None, Some(0.5), Some(50), Some(100)).unwrap();
        assert!(!r.converged);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("did not converge after 50 iterations")),
            "{:?}",
            r.warnings
        );
        assert!(r.n_clusters > 0);
        assert!(r.labels.iter().all(|&l| l < r.exemplar_indices.len()));
        assert!(format!("{r}").contains("Converged: no"));
    }

    #[test]
    fn test_validate_apcluster_lam9_defaults() {
        // ML-5: at the default limits (200, 15) the old code declared convergence after 15
        // iterations with zero exemplars and labelled every point 0.
        let (case, ds) = crate::test_support::load_case("apcluster_blobs");
        let x = case_matrix(&ds, &["x", "y"]);
        let r = affinity_propagation(x.view(), None, Some(0.9), None, None).unwrap();
        check_ap(&case, "lam9_default", &r);
        let s = format!("{r}");
        assert!(
            s.contains("Number of clusters: 6") && s.contains("Converged: yes"),
            "{s}"
        );
    }

    #[test]
    fn test_affinity_propagation_no_exemplar_is_an_error() {
        // apcluster(maxits = 20) on these data also ends with no exemplar (an empty clustering).
        let path = p2a_validation::repo_path("validation/datasets/generated/ap_blobs.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let x = case_matrix(&ds, &["x", "y"]);
        let e = affinity_propagation(x.view(), None, Some(0.9), Some(20), Some(15)).unwrap_err();
        assert!(e.contains("no exemplar"), "{e}");
        assert!(e.contains("raise the preference"), "{e}");
    }

    #[test]
    fn test_affinity_propagation_display_without_clusters_does_not_panic() {
        let r = AffinityPropagationResult {
            labels: vec![0; 3],
            exemplar_indices: vec![],
            n_iterations: 15,
            converged: false,
            net_similarity: f64::NAN,
            n_clusters: 0,
            n: 3,
            preference: -1.0,
            damping: 0.9,
            warnings: vec![],
            convergence: Convergence::default(),
        };
        assert!(format!("{r}").contains("No clusters"));
    }

    #[test]
    fn test_median_similarity_equals_the_sorted_type_7_median() {
        // Odd and even numbers of unique pairs, compared with the full off-diagonal sort.
        for n in [2usize, 3, 4, 5, 6, 9] {
            let s = Array2::from_shape_fn((n, n), |(i, j)| {
                let (a, b) = (i.min(j) as f64, i.max(j) as f64);
                -((a * 1.3 + b * b * 0.7).sin().abs() + 0.01 * (a + b))
            });
            let mut all: Vec<f64> = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        all.push(s[[i, j]]);
                    }
                }
            }
            all.sort_by(f64::total_cmp);
            let want = quantile_sorted(&all, 0.5, QuantileType::T7);
            let got = median_similarity(&s).unwrap();
            assert!(
                (got - want).abs() <= 1e-15 * want.abs(),
                "n = {n}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn test_affinity_propagation_rejects_bad_parameters() {
        let x = array![[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 4.9]];
        let ap = |p, d, m, c| affinity_propagation(x.view(), p, d, m, c).unwrap_err();
        assert!(ap(None, Some(0.4), None, None).contains("damping"));
        assert!(ap(None, Some(1.0), None, None).contains("damping"));
        assert!(ap(None, Some(f64::NAN), None, None).contains("damping"));
        assert!(ap(None, None, Some(0), None).contains("max_iterations"));
        assert!(ap(None, None, None, Some(0)).contains("convergence_iterations"));
        assert!(ap(Some(f64::INFINITY), None, None, None).contains("preference"));
        let one = array![[1.0, 2.0]];
        assert!(
            affinity_propagation(one.view(), None, None, None, None)
                .unwrap_err()
                .contains("2")
        );
        let nan = array![[0.0, f64::NAN], [1.0, 1.0]];
        assert!(
            affinity_propagation(nan.view(), None, None, None, None)
                .unwrap_err()
                .contains("finite")
        );
    }

    #[test]
    fn test_affinity_propagation_basic() {
        // Use more points and more separation for affinity propagation
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        // Use a low preference to encourage fewer clusters
        let result = affinity_propagation(
            data.view(),
            Some(-100.0), // Strong preference against being an exemplar
            Some(0.9),
            Some(200),
            Some(15),
        )
        .unwrap();

        // apcluster(S6, p = -100, lam = 0.9, maxits = 200, convits = 15, nonoise = TRUE): exemplars
        // 2 and 5. The old loop returned no exemplar here.
        assert_eq!(result.exemplar_indices, vec![1, 4]);
        assert_eq!((result.n_clusters, result.converged), (2, true));
        assert_eq!(result.labels, vec![0, 0, 0, 1, 1, 1]);
    }
}
