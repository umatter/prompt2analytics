//! Spectral clustering on an RBF or k-nearest-neighbour affinity graph ([`spectral_clustering`]).

use crate::options::Choices;
use crate::resource::{checkpoint, try_zeros2};
use faer::prelude::ReborrowMut;
use ndarray::{Array2, ArrayView2, Axis};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::common::try_distance_matrix;

/// Seed of the k-means step when `seed` is `None`.
pub const SPECTRAL_DEFAULT_SEED: u64 = 42;

// =============================================================================
// Spectral Clustering
// =============================================================================

/// Result of spectral clustering.
///
/// # References
///
/// - Ng, A.Y., Jordan, M.I., and Weiss, Y. (2002). "On Spectral Clustering:
///   Analysis and an algorithm". NIPS 2001.
/// - von Luxburg, U. (2007). "A Tutorial on Spectral Clustering".
///   Statistics and Computing, 17(4), 395-416.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralClusteringResult {
    /// Cluster assignments for each point (0-indexed)
    pub labels: Vec<usize>,
    /// The k smallest eigenvalues of the normalized Laplacian
    /// `L_sym = I - D^-1/2 A D^-1/2`, ascending: one minus the eigenvalues of the embedding
    pub eigenvalues: Vec<f64>,
    /// Affinity matrix used
    pub affinity_type: String,
    /// Gamma parameter for RBF kernel (if used)
    pub gamma: Option<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for SpectralClusteringResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spectral Clustering Results")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Number of clusters: {}", self.n_clusters)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Affinity type: {}", self.affinity_type)?;
        if let Some(gamma) = self.gamma {
            writeln!(f, "Gamma (RBF): {:.4}", gamma)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Smallest eigenvalues of the normalized Laplacian (L_sym = I - D^-1/2 A D^-1/2):"
        )?;
        for (i, &ev) in self.eigenvalues.iter().enumerate() {
            writeln!(f, "  λ_{}: {:.6}", i, ev)?;
        }

        let mut cluster_counts = vec![0usize; self.n_clusters];
        for &label in &self.labels {
            cluster_counts[label] += 1;
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes:")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }
        Ok(())
    }
}

/// Affinity matrix type for spectral clustering.
#[derive(Debug, Clone, Copy)]
pub enum AffinityType {
    /// RBF (Gaussian) kernel: exp(-gamma * ||x - y||^2)
    Rbf { gamma: f64 },
    /// k-nearest neighbors graph: an edge of weight 1 between two points when either is among
    /// the other's k nearest (a binary graph)
    Knn { k: usize },
    /// Precomputed affinity matrix (the data): square, symmetric and nonnegative
    Precomputed,
}

/// The affinity names `AffinityType` parses (its payloads get their defaults).
#[derive(Clone, Copy, PartialEq)]
enum AffinityKind {
    Rbf,
    Knn,
    Precomputed,
}

static AFFINITY_CHOICES: Choices<AffinityKind> = Choices {
    option: "affinity type",
    values: &[
        (&["rbf", "gaussian"], AffinityKind::Rbf),
        (&["knn", "nearest_neighbors"], AffinityKind::Knn),
        (&["precomputed"], AffinityKind::Precomputed),
    ],
};

impl std::str::FromStr for AffinityType {
    type Err = crate::errors::EconError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match AFFINITY_CHOICES.parse(s)? {
            AffinityKind::Rbf => AffinityType::Rbf { gamma: 1.0 },
            AffinityKind::Knn => AffinityType::Knn { k: 10 },
            AffinityKind::Precomputed => AffinityType::Precomputed,
        })
    }
}

/// Run spectral clustering.
///
/// Spectral clustering uses the eigenvalues of the graph Laplacian to perform
/// dimensionality reduction before clustering in fewer dimensions.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `affinity` - Affinity matrix type (default: RBF with auto gamma)
/// * `seed` - Seed of the k-means step (default: [`SPECTRAL_DEFAULT_SEED`])
///
/// # Algorithm (Ng, Jordan and Weiss; `kernlab::specc`'s embedding)
///
/// 1. Construct the affinity matrix A from the data (zero diagonal for RBF and k-NN)
/// 2. M = D^(-1/2) A D^(-1/2), with D the degrees; a point of degree 0 has D^(-1/2) = 0 (R
///    would give NaN)
/// 3. The k largest eigenvalues of M and their eigenvectors, exactly to solver tolerance: the
///    graph's connected components are solved separately (dense for up to 64 points, otherwise
///    Krylov-Schur on (I + M) / 2, whose eigenvalues lie in [0, 1], with a dense fallback);
///    each component contributes its eigenvalue 1 first, then the largest remaining eigenvalues
///    over all components
/// 4. Normalize the rows of the eigenvector matrix U
/// 5. Cluster the rows with k-means (Lloyd to convergence, 10 starts)
///
/// The reported `eigenvalues` are those of L_sym = I - M, ascending.
///
/// # Returns
/// * `SpectralClusteringResult` containing cluster assignments
///
/// # References
///
/// - Ng, A.Y., Jordan, M.I., and Weiss, Y. (2002). "On Spectral Clustering".
pub fn spectral_clustering(
    data: ArrayView2<f64>,
    n_clusters: usize,
    affinity: Option<AffinityType>,
    seed: Option<u64>,
) -> Result<SpectralClusteringResult, String> {
    let n = data.nrows();

    if n_clusters == 0 || n_clusters > n {
        return Err(format!(
            "n_clusters must be between 1 and {} (n_samples)",
            n
        ));
    }

    crate::ml::extract::require_finite(&data, "spectral clustering")?;
    match affinity {
        Some(AffinityType::Rbf { gamma }) if !gamma.is_finite() || gamma <= 0.0 => {
            return Err(format!("gamma must be finite and positive, got {gamma}"));
        }
        Some(AffinityType::Knn { k: 0 }) => {
            return Err("knn affinity: k must be at least 1, got 0".to_string());
        }
        _ => {}
    }

    // Default affinity: RBF with auto gamma
    let affinity_type = affinity.unwrap_or_else(|| {
        // Auto-compute gamma as 1 / (n_features * variance)
        let variance = compute_data_variance(&data);
        let gamma = 1.0 / (data.ncols() as f64 * variance.max(1e-10));
        AffinityType::Rbf { gamma }
    });

    let (gamma_val, affinity_name) = match affinity_type {
        AffinityType::Rbf { gamma } => (Some(gamma), "rbf".to_string()),
        AffinityType::Knn { k } => (None, format!("knn(k={})", k)),
        AffinityType::Precomputed => (None, "precomputed".to_string()),
    };

    // Build affinity matrix
    let affinity_matrix = match affinity_type {
        AffinityType::Rbf { gamma } => build_rbf_affinity(&data, gamma)?,
        AffinityType::Knn { k } => build_knn_affinity(&data, k)?,
        AffinityType::Precomputed => {
            check_precomputed_affinity(&data)?;
            let mut a =
                try_zeros2(n, n, "spectral clustering affinity").map_err(|e| e.to_string())?;
            a.assign(&data);
            a
        }
    };

    // Normalized affinity and its leading eigenvectors
    let (eigenvalues, eigenvectors) = compute_laplacian_eigenvectors(affinity_matrix, n_clusters)?;

    // Cluster the eigenvector embeddings with k-means
    let labels = cluster_eigenvectors(
        &eigenvectors,
        n_clusters,
        Some(seed.unwrap_or(SPECTRAL_DEFAULT_SEED)),
    )?;

    Ok(SpectralClusteringResult {
        labels,
        eigenvalues,
        affinity_type: affinity_name,
        gamma: gamma_val,
        n_clusters,
        n,
    })
}

/// A precomputed affinity must be square, symmetric (to 1e-10 relative) and nonnegative: the
/// eigensolver relies on the eigenvalues of D^-1/2 A D^-1/2 lying in [-1, 1].
fn check_precomputed_affinity(a: &ArrayView2<f64>) -> Result<(), String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Precomputed affinity must be square".to_string());
    }
    for i in 0..n {
        checkpoint().map_err(|e| e.to_string())?;
        for j in i..n {
            let (x, y) = (a[[i, j]], a[[j, i]]);
            if x < 0.0 {
                return Err(format!(
                    "Precomputed affinity must be nonnegative (A[{i},{j}] = {x})"
                ));
            }
            if (x - y).abs() > 1e-10 * x.abs().max(y.abs()) {
                return Err(format!(
                    "Precomputed affinity must be symmetric (A[{i},{j}] = {x}, A[{j},{i}] = {y})"
                ));
            }
        }
    }
    Ok(())
}

/// Build RBF affinity matrix.
fn build_rbf_affinity(data: &ArrayView2<f64>, gamma: f64) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let mut affinity =
        try_zeros2(n, n, "spectral clustering affinity").map_err(|e| e.to_string())?;

    for i in 0..n {
        checkpoint().map_err(|e| e.to_string())?;
        for j in (i + 1)..n {
            let mut sq_dist = 0.0;
            for k in 0..data.ncols() {
                sq_dist += (data[[i, k]] - data[[j, k]]).powi(2);
            }
            let a = (-gamma * sq_dist).exp();
            affinity[[i, j]] = a;
            affinity[[j, i]] = a;
        }
    }

    Ok(affinity)
}

/// Build the k-nearest-neighbour graph: weight 1 between i and j when either is among the
/// other's k nearest points, 0 otherwise.
fn build_knn_affinity(data: &ArrayView2<f64>, k: usize) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let k = k.min(n - 1);

    // Compute all pairwise distances
    let distances = try_distance_matrix(data, "spectral clustering distances")?;

    let mut affinity =
        try_zeros2(n, n, "spectral clustering affinity").map_err(|e| e.to_string())?;

    let mut dist_idx: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        checkpoint().map_err(|e| e.to_string())?;
        dist_idx.clear();
        dist_idx.extend((0..n).filter(|&j| i != j).map(|j| (distances[[i, j]], j)));
        dist_idx.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, j) in dist_idx.iter().take(k) {
            affinity[[i, j]] = 1.0;
            affinity[[j, i]] = 1.0;
        }
    }

    Ok(affinity)
}

/// Components up to this size are solved densely.
const DENSE_COMPONENT: usize = 64;

/// The k largest eigenvalues of M = D^-1/2 A D^-1/2 (overwriting `affinity`) and their
/// eigenvectors (n x k, zero outside their component). Returns L_sym's eigenvalues 1 - mu,
/// ascending, and the eigenvectors in the same order.
fn compute_laplacian_eigenvectors(
    mut affinity: Array2<f64>,
    k: usize,
) -> Result<(Vec<f64>, Array2<f64>), String> {
    let n = affinity.nrows();
    let to_string = |e: crate::errors::EconError| e.to_string();

    // D^-1/2, then M in place
    let d_inv_sqrt: Vec<f64> = (0..n)
        .map(|i| {
            let d = affinity.row(i).sum();
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();
    for i in 0..n {
        checkpoint().map_err(to_string)?;
        for j in 0..n {
            affinity[[i, j]] *= d_inv_sqrt[i] * d_inv_sqrt[j];
        }
    }

    // Connected components of the graph (nonzero off-diagonal entries), in order of their
    // smallest point.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        checkpoint().map_err(to_string)?;
        for j in (i + 1)..n {
            if affinity[[i, j]] != 0.0 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut comp_of_root = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        if comp_of_root[r] == usize::MAX {
            comp_of_root[r] = components.len();
            components.push(Vec::new());
        }
        components[comp_of_root[r]].push(i);
    }

    // (is the component's leading eigenvalue of a connected graph, mu, component, vector)
    let mut candidates: Vec<(bool, f64, usize, Vec<f64>)> = Vec::new();
    let single = components.len() == 1;
    for (c, members) in components.iter().enumerate() {
        checkpoint().map_err(to_string)?;
        let m = members.len();
        let want = k.min(m);
        let pairs = if single {
            component_eigen(
                affinity
                    .as_slice()
                    .ok_or("spectral clustering: matrix layout")?,
                m,
                want,
                c as u64,
            )?
        } else {
            let what = "spectral clustering component";
            let len = crate::resource::checked_len(&[m, m], what).map_err(to_string)?;
            let mut block = crate::resource::try_vec(len, 0.0f64, what).map_err(to_string)?;
            for (a, &i) in members.iter().enumerate() {
                for (b, &j) in members.iter().enumerate() {
                    block[a * m + b] = affinity[[i, j]];
                }
            }
            component_eigen(&block, m, want, c as u64)?
        };
        let connected = members.iter().any(|&i| d_inv_sqrt[i] > 0.0);
        for (rank, (mu, v)) in pairs.into_iter().enumerate() {
            candidates.push((rank == 0 && connected, mu, c, v));
        }
    }
    // Leading eigenvalues (1) of the components first, in component order; then by mu.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
    candidates.truncate(k);

    let mut eigenvectors = try_zeros2(n, k, "spectral clustering embedding").map_err(to_string)?;
    let mut eigenvalues = Vec::with_capacity(k);
    for (col, (_, mu, c, v)) in candidates.iter().enumerate() {
        for (&i, &x) in components[*c].iter().zip(v) {
            eigenvectors[[i, col]] = x;
        }
        eigenvalues.push(1.0 - mu);
    }
    Ok((eigenvalues, eigenvectors))
}

/// The `want` largest eigenpairs of the symmetric m x m block M (row-major `block`), largest
/// first. Dense for small blocks; otherwise Krylov-Schur on (I + M) / 2 with a dense fallback.
fn component_eigen(
    block: &[f64],
    m: usize,
    want: usize,
    seed: u64,
) -> Result<Vec<(f64, Vec<f64>)>, String> {
    if m > DENSE_COMPONENT {
        if let Some(pairs) = krylov_eigen(block, m, want, seed)? {
            return Ok(pairs);
        }
    }
    dense_eigen(block, m, want)
}

/// Dense symmetric eigendecomposition of the block (fallible workspace).
fn dense_eigen(block: &[f64], m: usize, want: usize) -> Result<Vec<(f64, Vec<f64>)>, String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
    let to_string = |e: crate::errors::EconError| e.to_string();
    let what = "spectral clustering eigenvectors";
    crate::resource::check_alloc::<f64>(
        crate::resource::checked_len(&[m, m], what).map_err(to_string)?,
        what,
    )
    .map_err(to_string)?;
    let ws = "spectral clustering eigensolver workspace";
    let par = faer::Par::Seq;
    let scratch =
        self_adjoint_evd_scratch::<f64>(m, ComputeEigenvectors::Yes, par, Default::default());
    if scratch.layout().is_err() {
        return Err(crate::resource::overflow(ws).to_string());
    }
    crate::resource::check_alloc::<u8>(scratch.size_bytes(), ws).map_err(to_string)?;
    let mut buf = MemBuffer::try_new(scratch)
        .map_err(|_| crate::resource::reserve_failed(ws, scratch.size_bytes(), 1).to_string())?;
    let mat = faer::MatRef::from_row_major_slice(block, m, m);
    let mut vectors = faer::Mat::<f64>::zeros(m, m);
    let mut s = faer::diag::Diag::<f64>::zeros(m);
    self_adjoint_evd(
        mat,
        s.as_mut(),
        Some(vectors.as_mut()),
        par,
        MemStack::new(&mut buf),
        Default::default(),
    )
    .map_err(|_| "spectral clustering: eigendecomposition failed".to_string())?;
    let values = s.column_vector();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    Ok(order
        .iter()
        .take(want)
        .map(|&j| (values[j], (0..m).map(|i| vectors[(i, j)]).collect()))
        .collect())
}

/// The block (I + M) / 2 as a faer operator: its eigenvalues lie in [0, 1], so the largest in
/// magnitude are the largest.
#[derive(Debug)]
struct ShiftedBlock<'a> {
    m: faer::MatRef<'a, f64>,
}

impl faer::matrix_free::LinOp<f64> for ShiftedBlock<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> faer::dyn_stack::StackReq {
        faer::dyn_stack::StackReq::EMPTY
    }
    fn nrows(&self) -> usize {
        self.m.nrows()
    }
    fn ncols(&self) -> usize {
        self.m.ncols()
    }
    fn apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        _stack: &mut faer::dyn_stack::MemStack,
    ) {
        let mut out = out;
        out.copy_from(rhs);
        faer::linalg::matmul::matmul(out.rb_mut(), faer::Accum::Add, self.m, rhs, 1.0, par);
        for j in 0..out.ncols() {
            for i in 0..out.nrows() {
                out[(i, j)] *= 0.5;
            }
        }
    }
    fn conj_apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        stack: &mut faer::dyn_stack::MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

/// Krylov-Schur (faer's `partial_eigen`) on (I + M) / 2; `None` when fewer than `want` pairs
/// converged or a result is not real.
fn krylov_eigen(
    block: &[f64],
    m: usize,
    want: usize,
    seed: u64,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::matrix_free::eigen::{PartialEigenParams, partial_eigen, partial_eigen_scratch};
    let op = ShiftedBlock {
        m: faer::MatRef::from_row_major_slice(block, m, m),
    };
    let par = faer::Par::Seq;
    let params = PartialEigenParams::default();
    let ws = "spectral clustering Krylov workspace";
    let scratch = partial_eigen_scratch::<f64>(&op, want, par, params);
    if scratch.layout().is_err() {
        return Err(crate::resource::overflow(ws).to_string());
    }
    crate::resource::check_alloc::<u8>(scratch.size_bytes(), ws).map_err(|e| e.to_string())?;
    let mut buf = MemBuffer::try_new(scratch)
        .map_err(|_| crate::resource::reserve_failed(ws, scratch.size_bytes(), 1).to_string())?;
    // A fixed pseudo-random start, not the known leading eigenvector D^1/2 1 (a start inside an
    // invariant subspace stops the Krylov basis at once).
    let mut rng = StdRng::seed_from_u64(0x5eed_u64 ^ seed);
    let mut v0 = faer::Col::<f64>::zeros(m);
    let mut norm = 0.0;
    for i in 0..m {
        let x = rng.r#gen::<f64>() - 0.5;
        v0[i] = x;
        norm += x * x;
    }
    let norm = norm.sqrt();
    for i in 0..m {
        v0[i] /= norm;
    }
    let mut vecs = faer::Mat::<faer::c64>::zeros(m, want);
    let mut vals = vec![faer::c64::new(0.0, 0.0); want];
    let info = partial_eigen::<f64>(
        vecs.as_mut(),
        &mut vals,
        &op,
        v0.as_ref(),
        f64::EPSILON * 128.0,
        par,
        MemStack::new(&mut buf),
        params,
    );
    if info.n_converged_eigen < want {
        return Ok(None);
    }
    let mut pairs = Vec::with_capacity(want);
    for (j, val) in vals.iter().enumerate() {
        if !(val.re.is_finite() && val.im.abs() <= 1e-10) {
            return Ok(None);
        }
        let mut v: Vec<f64> = (0..m).map(|i| vecs[(i, j)].re).collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Ok(None);
        }
        v.iter_mut().for_each(|x| *x /= norm);
        pairs.push((2.0 * val.re - 1.0, v));
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(Some(pairs))
}

/// Cluster eigenvector embeddings using k-means.
fn cluster_eigenvectors(
    embeddings: &Array2<f64>,
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<usize>, String> {
    let n = embeddings.nrows();

    // Normalize rows (a row of zeros stays zero)
    let mut normalized = embeddings.clone();
    for i in 0..n {
        let norm: f64 = normalized.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for j in 0..embeddings.ncols() {
                normalized[[i, j]] /= norm;
            }
        }
    }

    // k-means, Lloyd to convergence
    let result = crate::ml::kmeans(normalized.view(), k, Some(100), None, Some(10), seed)?;

    Ok(result.labels)
}

/// Convenience wrapper for spectral_clustering.
pub fn run_spectral_clustering(
    data: ArrayView2<f64>,
    n_clusters: usize,
    gamma: Option<f64>,
    seed: Option<u64>,
) -> Result<SpectralClusteringResult, String> {
    let affinity = gamma.map(|g| AffinityType::Rbf { gamma: g });
    spectral_clustering(data, n_clusters, affinity, seed)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute overall data variance.
fn compute_data_variance(data: &ArrayView2<f64>) -> f64 {
    let mean = match data.mean_axis(Axis(0)) {
        Some(m) => m,
        None => return 0.0,
    };
    let n = data.nrows();

    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..data.ncols() {
            variance += (data[[i, j]] - mean[j]).powi(2);
        }
    }
    variance / (n * data.ncols()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_spectral_rejects_bad_parameters_and_default_seed_is_fixed() {
        let x = Array2::from_shape_fn((12, 2), |(i, j)| {
            (i % 2) as f64 * 5.0 + 0.1 * ((i * 3 + j) as f64).sin()
        });
        let e = |a| spectral_clustering(x.view(), 2, Some(a), None).unwrap_err();
        assert!(e(AffinityType::Rbf { gamma: 0.0 }).contains("gamma"));
        assert!(e(AffinityType::Knn { k: 0 }).contains("k must"));
        let mut nan = x.clone();
        nan[[1, 1]] = f64::NAN;
        assert!(
            spectral_clustering(nan.view(), 2, None, None)
                .unwrap_err()
                .contains("finite")
        );
        let a = spectral_clustering(x.view(), 2, None, None).unwrap();
        let b = spectral_clustering(x.view(), 2, None, Some(SPECTRAL_DEFAULT_SEED)).unwrap();
        assert_eq!(a.labels, b.labels);
        assert!("RBF".parse::<AffinityType>().is_ok());
        assert!(
            "cosine"
                .parse::<AffinityType>()
                .unwrap_err()
                .to_string()
                .contains("expected one of: rbf, knn, precomputed")
        );
        AFFINITY_CHOICES.validate().unwrap();
    }

    #[test]
    fn test_spectral_clustering_basic() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        let result = spectral_clustering(data.view(), 2, None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.n_clusters, 2);
    }

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    fn xy(ds: &crate::data::Dataset) -> Array2<f64> {
        let x = numeric_column(ds.df(), "x", NaPolicy::Fail).unwrap().0;
        let y = numeric_column(ds.df(), "y", NaPolicy::Fail).unwrap().0;
        Array2::from_shape_fn((x.len(), 2), |(i, j)| if j == 0 { x[i] } else { y[i] })
    }

    /// Labels renumbered from 1 in order of first appearance (the case's convention).
    fn by_first_appearance(labels: &[usize]) -> Vec<f64> {
        let mut map = std::collections::HashMap::new();
        labels
            .iter()
            .map(|l| {
                let next = map.len() + 1;
                *map.entry(*l).or_insert(next) as f64
            })
            .collect()
    }

    #[test]
    fn test_validate_spectral_njw_blobs() {
        // eigen(D^-1/2 A D^-1/2) on ap_blobs, A = exp(-gamma |x - y|^2) with p2a's default gamma.
        let (case, _) = crate::test_support::load_case("spectral_njw");
        let blobs = {
            let path = case.dataset_path().with_file_name("ap_blobs.csv");
            crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
        };
        let x = xy(&blobs);
        let r = spectral_clustering(x.view(), 3, None, None).unwrap();
        Check::new(&case)
            .scalar("blobs_gamma", r.gamma.unwrap(), Tol::coef(115))
            .vector("blobs_lsym", &r.eigenvalues, Tol::coef(115))
            .vector(
                "blobs_cluster",
                &by_first_appearance(&r.labels),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_spectral_njw_pair() {
        // spectral_pair: two blobs and a far tight pair, gamma 0.5 (two graph components; the
        // pair's eigenvalue near -1 must not enter the embedding).
        let (case, ds) = crate::test_support::load_case("spectral_njw");
        let x = xy(&ds);
        let mut check = Check::new(&case);
        for k in [2usize, 3] {
            let r = spectral_clustering(x.view(), k, Some(AffinityType::Rbf { gamma: 0.5 }), None)
                .unwrap();
            let lsym: Vec<f64> = case
                .vector(&format!("pair_k{k}_mu"))
                .iter()
                .map(|m| 1.0 - m)
                .collect();
            for (a, b) in r.eigenvalues.iter().zip(&lsym) {
                // Tol::coef(42): 1e-6 relative, 1e-12 absolute
                assert!((a - b).abs() <= 1e-6 * b.abs() + 1e-12, "{a} vs {b}");
            }
            check.vector(
                &format!("pair_k{k}_cluster"),
                &by_first_appearance(&r.labels),
                Tol::exact(),
            );
        }
        check.finish();
    }

    #[test]
    fn test_spectral_knn_weights_are_binary() {
        let mut rng = StdRng::seed_from_u64(4);
        let x = Array2::from_shape_fn((40, 2), |(i, _)| (i % 2) as f64 * 6.0 + rng.r#gen::<f64>());
        let a = build_knn_affinity(&x.view(), 5).unwrap();
        assert!(a.iter().all(|&v| v == 0.0 || v == 1.0));
        // The old weights exp(-0.5) on the same edges give the same normalized affinity.
        let scaled = a.mapv(|v| v * (-0.5f64).exp());
        let (e1, v1) = compute_laplacian_eigenvectors(a, 2).unwrap();
        let (e2, v2) = compute_laplacian_eigenvectors(scaled, 2).unwrap();
        for (p, q) in e1.iter().zip(&e2) {
            assert!((p - q).abs() <= 1e-12, "{p} vs {q}");
        }
        let l1 = cluster_eigenvectors(&v1, 2, Some(1)).unwrap();
        let l2 = cluster_eigenvectors(&v2, 2, Some(1)).unwrap();
        assert_eq!(by_first_appearance(&l1), by_first_appearance(&l2));
    }

    #[test]
    fn test_spectral_partial_and_dense_agree() {
        // 300 points in 4 noisy blobs, one connected RBF graph: the Krylov path against the
        // dense eigendecomposition.
        let mut rng = StdRng::seed_from_u64(9);
        let centres = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]];
        let x = Array2::from_shape_fn((300, 2), |(i, j)| {
            centres[i % 4][j] + rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.7
        });
        let a = build_rbf_affinity(&x.view(), 0.3).unwrap();
        let n = a.nrows();
        let mut m = a.clone();
        let d: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
        for i in 0..n {
            for j in 0..n {
                m[[i, j]] *= 1.0 / (d[i] * d[j]).sqrt();
            }
        }
        let block = m.as_slice().unwrap();
        let dense = dense_eigen(block, n, 4).unwrap();
        let krylov = krylov_eigen(block, n, 4, 0).unwrap().expect("converged");
        for ((a, _), (b, _)) in dense.iter().zip(&krylov) {
            assert!((a - b).abs() <= 1e-10, "{a} vs {b}");
        }
        // Same partition through the public path (Krylov) and from the dense embedding.
        let r = spectral_clustering(x.view(), 4, Some(AffinityType::Rbf { gamma: 0.3 }), Some(2))
            .unwrap();
        let mut emb = Array2::zeros((n, 4));
        for (c, (_, v)) in dense.iter().enumerate() {
            for i in 0..n {
                emb[[i, c]] = v[i];
            }
        }
        let l = cluster_eigenvectors(&emb, 4, Some(2)).unwrap();
        assert_eq!(by_first_appearance(&r.labels), by_first_appearance(&l));
    }

    #[test]
    fn test_spectral_precomputed_must_be_symmetric_and_nonnegative() {
        let bad = array![[0.0, -1.0], [-1.0, 0.0]];
        let e =
            spectral_clustering(bad.view(), 1, Some(AffinityType::Precomputed), None).unwrap_err();
        assert!(e.contains("nonnegative"), "{e}");
        let asym = array![[0.0, 1.0, 0.5], [0.9, 0.0, 0.2], [0.5, 0.2, 0.0]];
        let e =
            spectral_clustering(asym.view(), 1, Some(AffinityType::Precomputed), None).unwrap_err();
        assert!(e.contains("symmetric"), "{e}");
    }

    #[test]
    fn test_spectral_clustering_is_cancellable() {
        use crate::ml::advanced_clustering_mod::common::test_data::cancelled;
        let x = ndarray::Array2::from_shape_fn((60, 2), |(i, j)| {
            (i % 3) as f64 * 4.0 + 0.3 * ((i * 7 + j * 3) as f64).sin() + 0.1 * j as f64
        });
        let e = crate::resource::with_budget(cancelled(), || {
            spectral_clustering(x.view(), 3, None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }
}
