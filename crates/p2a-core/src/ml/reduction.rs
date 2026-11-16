//! Dimensionality reduction: PCA and t-SNE.
//!
//! Pure Rust implementations using ndarray.
//!
//! PCA uses SVD via faer for O(nm²) complexity instead of power iteration.

use super::extract::require_finite;
use crate::errors::{EconError, EconResult};
use crate::resource::{
    check_alloc, checked_len, checkpoint, try_vec, try_with_capacity, try_zeros2,
};
use crate::traits::convergence::{ResultNotes, notes_text};
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// PCA (Principal Component Analysis) result.
#[derive(Debug, Clone)]
pub struct PCAResult {
    /// Principal components (eigenvectors), shape: (n_components, n_features)
    pub components: Array2<f64>,
    /// Explained variance for each component
    pub explained_variance: Array1<f64>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<f64>,
    /// Mean of each feature (used for centering)
    pub mean: Array1<f64>,
    /// Total variance in the original data: the sum of all `min(n, p)` component
    /// variances (the trace of the sample covariance), on both computation paths,
    /// so `explained_variance_ratio` is prcomp's proportion of variance
    pub total_variance: f64,
    /// Number of components kept
    pub n_components: usize,
    /// Transformed data (if computed)
    pub transformed: Option<Array2<f64>>,
}

impl std::fmt::Display for PCAResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PCA Results")?;
        writeln!(f, "===========")?;
        writeln!(f, "Number of components: {}", self.n_components)?;
        writeln!(f, "Total variance: {:.4}", self.total_variance)?;
        writeln!(f)?;
        writeln!(f, "Explained Variance:")?;
        let mut cumulative = 0.0;
        for i in 0..self.n_components {
            cumulative += self.explained_variance_ratio[i];
            writeln!(
                f,
                "  PC{}: {:.4} ({:.2}%) [cumulative: {:.2}%]",
                i + 1,
                self.explained_variance[i],
                self.explained_variance_ratio[i] * 100.0,
                cumulative * 100.0
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Principal Components (loadings):")?;
        for i in 0..self.n_components.min(5) {
            let loadings: Vec<String> = self
                .components
                .row(i)
                .iter()
                .take(10)
                .map(|v| format!("{:.4}", v))
                .collect();
            let suffix = if self.components.ncols() > 10 {
                ", ..."
            } else {
                ""
            };
            writeln!(f, "  PC{}: [{}{}]", i + 1, loadings.join(", "), suffix)?;
        }
        if self.n_components > 5 {
            writeln!(f, "  ... ({} more components)", self.n_components - 5)?;
        }
        Ok(())
    }
}

/// Run PCA (Principal Component Analysis).
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_components` - Number of components to keep (None = keep all)
/// * `transform` - Whether to return transformed data
///
/// Refuses NaN and ±Inf, naming the first non-finite cell (R `prcomp` stops too).
pub fn pca(
    data: ArrayView2<f64>,
    n_components: Option<usize>,
    transform: bool,
) -> Result<PCAResult, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for PCA".to_string());
    }
    require_finite(&data, "PCA")?;

    // Determine number of components
    let max_components = n_samples.min(n_features);
    let n_comp = match n_components {
        Some(k) if k > max_components => {
            return Err(format!(
                "n_components ({}) cannot exceed min(n_samples, n_features) ({})",
                k, max_components
            ));
        }
        Some(k) => k,
        None => max_components,
    };

    // Center the data
    let mean = data.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;

    let mut centered = data.to_owned();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }

    // For tall-skinny data (n >> p), covariance eigendecomposition is much faster:
    // 1. Compute p×p covariance matrix C = X'X / (n-1)  — O(np²)
    // 2. Eigendecompose C (p×p)                          — O(p³)
    // For n=5000, p=5 this operates on a 5×5 matrix instead of SVD on 5000×5.
    //
    // For wide or square data, use SVD which is more numerically stable.
    let use_cov_eigen = n_samples > 4 * n_features;

    let (eigenvalues, eigenvectors) = if use_cov_eigen {
        pca_via_cov_eigen(&centered.view(), n_comp)?
    } else {
        pca_via_svd(&centered.view(), n_comp)?
    };

    // Both paths return every component variance (min(n, p) on the SVD path, p on the
    // covariance path), so the shares are of the full-spectrum total, as prcomp's
    let total_variance: f64 = eigenvalues.sum();
    let explained_variance = eigenvalues.slice(ndarray::s![..n_comp]).to_owned();
    let explained_variance_ratio = &explained_variance / total_variance;

    // Principal components (eigenvectors as rows)
    let components = eigenvectors.t().slice(ndarray::s![..n_comp, ..]).to_owned();

    // Transform data if requested
    let transformed = if transform {
        Some(centered.dot(&eigenvectors.slice(ndarray::s![.., ..n_comp])))
    } else {
        None
    };

    Ok(PCAResult {
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        total_variance,
        n_components: n_comp,
        transformed,
    })
}

/// Compute covariance matrix (sample covariance, unbiased).
#[allow(dead_code)]
fn covariance_matrix(centered: &ArrayView2<f64>) -> Array2<f64> {
    let n = centered.nrows() as f64;

    centered.t().dot(centered) / (n - 1.0)
}

/// PCA via SVD (more numerically stable than eigendecomposition of covariance matrix).
///
/// For centered data X, SVD gives X = U * S * V^T
/// - Principal components = columns of V
/// - Explained variance = S² / (n - 1)
/// - Transformed data = U * S
///
/// Optimized to compute thin SVD only once, extracting both singular values
/// and right singular vectors in a single pass. Returns all `min(n, p)`
/// variances (for the total) and the first `n_components` vectors.
fn pca_via_svd(
    centered: &ArrayView2<f64>,
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = centered.nrows();
    let p = centered.ncols();

    // Convert ndarray to faer Mat
    let mat: Mat<f64> = Mat::from_fn(n, p, |i, j| centered[[i, j]]);

    // Compute thin SVD - this gives us singular values and right singular vectors
    let svd = mat.thin_svd().map_err(|e| format!("SVD failed: {:?}", e))?;
    let v = svd.V();

    // Get singular values from the SVD as a column vector
    let s_vec = svd.S().column_vector();
    let n_singular = s_vec.nrows();

    // Number of components we can extract
    let n_comp = n_components.min(n_singular);

    // Convert singular values to explained variance
    // Variance = s² / (n - 1)
    let n_minus_1 = (n - 1) as f64;
    let mut eigenvalues = Array1::zeros(n_singular);
    for i in 0..n_singular {
        eigenvalues[i] = s_vec[i].powi(2) / n_minus_1;
    }

    // Convert V to ndarray (principal components as columns)
    let mut eigenvectors = Array2::zeros((p, n_comp));
    for j in 0..n_comp {
        for i in 0..p {
            eigenvectors[[i, j]] = v[(i, j)];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// PCA via covariance matrix eigendecomposition.
///
/// For GPU-accelerated PCA on tall matrices (n >> p):
/// 1. Compute covariance = X'X / (n-1) using GPU-dispatched xtx
/// 2. Eigendecompose the p x p covariance matrix on CPU
///
/// This leverages GPU DGEMM for the O(np²) bottleneck while keeping
/// the O(p³) eigendecomposition on CPU.
fn pca_via_cov_eigen(
    centered: &ArrayView2<f64>,
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = centered.nrows();
    let p = centered.ncols();
    let n_minus_1 = (n - 1) as f64;

    // Compute covariance matrix via GPU-dispatched xtx (O(np²) on GPU)
    let xtx = crate::linalg::xtx(centered);
    let cov = &xtx / n_minus_1;

    // Eigendecompose p x p covariance matrix on CPU (fast for small p)
    let (eigenvalues, eigenvectors) =
        crate::linalg::eig_symmetric(&cov.view()).map_err(|e| format!("Eigen failed: {}", e))?;

    // Sort by descending eigenvalue — return all p eigenvalues for total_variance
    let mut indices: Vec<usize> = (0..p).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

    let n_comp = n_components.min(p);
    let mut sorted_eigenvalues = Array1::zeros(p);
    let mut sorted_eigenvectors = Array2::zeros((p, n_comp));

    for (out_j, &orig_j) in indices.iter().enumerate() {
        sorted_eigenvalues[out_j] = eigenvalues[orig_j].max(0.0);
        if out_j < n_comp {
            for i in 0..p {
                sorted_eigenvectors[[i, out_j]] = eigenvectors[[i, orig_j]];
            }
        }
    }

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

/// Project new data onto principal components.
pub fn pca_transform(data: ArrayView2<f64>, pca_result: &PCAResult) -> EconResult<Array2<f64>> {
    let n_features = data.ncols();
    if n_features != pca_result.mean.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Data has {} features, but PCA was fit with {} features",
                n_features,
                pca_result.mean.len()
            ),
        });
    }

    // Center using the stored mean
    let mut centered = data.to_owned();
    for mut row in centered.rows_mut() {
        row -= &pca_result.mean;
    }

    // Project onto components
    let transformed = centered.dot(&pca_result.components.t());
    Ok(transformed)
}

/// Reconstruct data from principal components.
pub fn pca_inverse_transform(
    transformed: ArrayView2<f64>,
    pca_result: &PCAResult,
) -> EconResult<Array2<f64>> {
    let n_components = transformed.ncols();
    if n_components != pca_result.n_components {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Transformed data has {} components, but PCA has {} components",
                n_components, pca_result.n_components
            ),
        });
    }

    // Reconstruct: X_approx = X_transformed @ components + mean
    let mut reconstructed = transformed.dot(&pca_result.components);
    for mut row in reconstructed.rows_mut() {
        row += &pca_result.mean;
    }

    Ok(reconstructed)
}

/// t-SNE (t-distributed Stochastic Neighbor Embedding) result.
#[derive(Debug, Clone)]
pub struct TsneResult {
    /// Low-dimensional embedding, shape: (n_samples, n_components)
    pub embedding: Array2<f64>,
    /// Number of output dimensions
    pub n_components: usize,
    /// Perplexity used
    pub perplexity: f64,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// KL(P || Q) of the returned embedding, with the unexaggerated P
    pub kl_divergence: f64,
    /// Warnings: points whose bandwidth search missed the perplexity tolerance
    pub warnings: Vec<String>,
}

impl ResultNotes for TsneResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for TsneResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "t-SNE Results")?;
        writeln!(f, "=============")?;
        writeln!(f, "Number of components: {}", self.n_components)?;
        writeln!(f, "Perplexity: {:.1}", self.perplexity)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Final KL divergence: {:.6}", self.kl_divergence)?;
        writeln!(f)?;
        writeln!(
            f,
            "Embedding shape: ({}, {})",
            self.embedding.nrows(),
            self.embedding.ncols()
        )?;

        // Show first few points
        let n_show = self.embedding.nrows().min(5);
        writeln!(f, "\nFirst {} embedded points:", n_show)?;
        for i in 0..n_show {
            let point: Vec<String> = self
                .embedding
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            writeln!(f, "  Point {}: [{}]", i, point.join(", "))?;
        }
        if self.embedding.nrows() > 5 {
            writeln!(f, "  ... ({} more points)", self.embedding.nrows() - 5)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Run t-SNE (t-distributed Stochastic Neighbor Embedding).
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_components` - Number of output dimensions (default: 2)
/// * `perplexity` - Perplexity parameter (default: 30.0)
/// * `max_iterations` - Maximum number of iterations (default: 1000)
/// * `learning_rate` - Learning rate (default: 200.0)
/// * `seed` - Random seed for reproducibility
pub fn tsne(
    data: ArrayView2<f64>,
    n_components: Option<usize>,
    perplexity: Option<f64>,
    max_iterations: Option<usize>,
    learning_rate: Option<f64>,
    seed: Option<u64>,
) -> Result<TsneResult, String> {
    let n_samples = data.nrows();
    let n_comp = n_components.unwrap_or(2);
    let perp = perplexity.unwrap_or(30.0);
    let max_iter = max_iterations.unwrap_or(1000);
    let lr = learning_rate.unwrap_or(200.0);

    if n_samples < 4 {
        return Err("Need at least 4 samples for t-SNE".to_string());
    }
    if max_iter == 0 {
        return Err("max_iterations must be at least 1 (got 0)".to_string());
    }
    if n_comp == 0 {
        return Err("n_components must be at least 1 (got 0)".to_string());
    }
    if !(perp.is_finite() && perp > 0.0) {
        return Err(format!(
            "perplexity must be finite and positive (got {perp})"
        ));
    }
    if !(lr.is_finite() && lr > 0.0) {
        return Err(format!(
            "learning_rate must be finite and positive (got {lr})"
        ));
    }
    if let Some(((i, j), v)) = data.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "data must contain only finite values (row {i}, column {j} is {v})"
        ));
    }
    if perp >= (n_samples as f64) / 3.0 {
        return Err(format!(
            "Perplexity ({}) must be less than n_samples / 3 ({})",
            perp,
            n_samples / 3
        ));
    }

    let to_string = |e: EconError| e.to_string();

    // Compute joint probabilities P_ij using Gaussian kernel (row by row; no
    // distance matrix is kept)
    let (p, calibration) = compute_joint_probabilities(&data, perp)?;

    // Initialize low-dimensional embedding using PCA or random
    let mut y = initialize_embedding(data.view(), n_comp, seed)?;

    // Gradient descent parameters
    let momentum = 0.5;
    let final_momentum = 0.8;
    let momentum_switch_iter = 250;
    let early_exaggeration = 4.0;
    let early_exaggeration_iter = 100;

    let mut y_velocity = try_zeros2(n_samples, n_comp, "t-SNE velocity").map_err(to_string)?;
    let mut gains = try_zeros2(n_samples, n_comp, "t-SNE gains").map_err(to_string)?;
    gains.fill(1.0);

    // Student-t similarities, overwritten every iteration
    let mut q = try_zeros2(n_samples, n_samples, "t-SNE similarity matrix").map_err(to_string)?;

    for iter in 0..max_iter {
        checkpoint().map_err(to_string)?;

        // Early exaggeration multiplies P for the first iterations
        let exaggeration = if iter < early_exaggeration_iter {
            early_exaggeration
        } else {
            1.0
        };

        // Switch to final momentum
        let current_momentum = if iter < momentum_switch_iter {
            momentum
        } else {
            final_momentum
        };

        // Compute Q distribution (Student-t with 1 DOF)
        let sum_q = fill_q_distribution(&y.view(), &mut q);

        // Compute gradient
        let gradient = compute_gradient(&p.view(), exaggeration, &q.view(), &y.view(), sum_q);

        // Update gains for adaptive learning rate
        for i in 0..n_samples {
            for j in 0..n_comp {
                // bhtsne/Rtsne: sign(dY) != sign(uY), with sign(0) = 0
                gains[[i, j]] = if sign3(gradient[[i, j]]) != sign3(y_velocity[[i, j]]) {
                    gains[[i, j]] + 0.2
                } else {
                    f64::max(gains[[i, j]] * 0.8, 0.01)
                };
            }
        }

        // Update velocity and position
        y_velocity = &y_velocity * current_momentum - lr * &gains * &gradient;
        y = &y + &y_velocity;

        // Center embedding
        let mean = y.mean_axis(Axis(0)).unwrap();
        for mut row in y.rows_mut() {
            row -= &mean;
        }
    }

    // KL(P || Q) of the returned embedding, with the unexaggerated P
    let sum_q = fill_q_distribution(&y.view(), &mut q);
    let kl_div = compute_kl_divergence(&p.view(), &q.view(), sum_q);

    Ok(TsneResult {
        embedding: y,
        n_components: n_comp,
        perplexity: perp,
        n_iterations: max_iter,
        kl_divergence: kl_div,
        warnings: calibration.warning(n_samples, perp).into_iter().collect(),
    })
}

/// Three-valued sign (`f64::signum(0.0)` is 1).
fn sign3(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Steps of the bandwidth search per point (bhtsne's limit).
const PERPLEXITY_STEPS: usize = 200;
/// Tolerance of the bandwidth search on `|H - ln(perplexity)|` (bhtsne's).
const PERPLEXITY_TOL: f64 = 1e-5;

/// Points whose bandwidth search missed [`PERPLEXITY_TOL`], and the largest miss.
#[derive(Debug, Clone, Copy, Default)]
struct Calibration {
    missed: usize,
    worst: f64,
}

impl Calibration {
    /// The warning for points that missed the tolerance, if any.
    fn warning(&self, n: usize, perplexity: f64) -> Option<String> {
        (self.missed > 0).then(|| {
            format!(
                "the bandwidth search did not reach perplexity {perplexity} within {PERPLEXITY_TOL:e} for {} of {n} points in {PERPLEXITY_STEPS} steps (largest |H - ln(perplexity)| {:.4}); their affinities use the last bandwidth tried (exact duplicates or perplexity close to the number of distinct neighbours)",
                self.missed, self.worst
            )
        })
    }
}

/// The joint affinities `P` of t-SNE: the conditional rows of
/// [`conditional_affinities`], symmetrised as `(P + P') / sum(P + P')` with no
/// floor (bhtsne's exact path). Only the `n x n` matrix `P` is allocated.
fn compute_joint_probabilities(
    data: &ArrayView2<f64>,
    perplexity: f64,
) -> Result<(Array2<f64>, Calibration), String> {
    let (mut p, calibration) = conditional_affinities(data, perplexity)?;
    let n = p.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let val = p[[i, j]] + p[[j, i]];
            p[[i, j]] = val;
            p[[j, i]] = val;
        }
    }
    let total = p.sum();
    if total > 0.0 {
        p /= total;
    }
    Ok((p, calibration))
}

/// The conditional affinities `p_{j|i}`, row by row, calibrated as bhtsne's
/// `computeGaussianPerplexity`: for each point a search over the precision
/// `beta = 1 / (2 sigma^2)`, starting at 1, doubling (or halving) while the
/// other bound is open and bisecting once both are, for at most
/// [`PERPLEXITY_STEPS`] steps, until the row entropy
/// `H = beta sum_j d2_ij p_j / S + ln S` (`S = f64::MIN_POSITIVE + sum_j p_j`,
/// `p_j = exp(-beta d2_ij)`) is within [`PERPLEXITY_TOL`] of `ln(perplexity)`.
/// Every row is written, with the last bandwidth tried when the search misses;
/// the misses are counted. The squared distances of row `i` go into a reused
/// buffer (pair `(min(i, j), max(i, j))`); no distance matrix is kept.
fn conditional_affinities(
    data: &ArrayView2<f64>,
    perplexity: f64,
) -> Result<(Array2<f64>, Calibration), String> {
    let to_string = |e: EconError| e.to_string();
    let n = data.nrows();
    let n_features = data.ncols();
    let target_entropy = perplexity.ln();

    let mut p = try_zeros2(n, n, "t-SNE affinity matrix").map_err(to_string)?;
    let mut distances = try_vec(n, 0.0f64, "t-SNE distance row").map_err(to_string)?;
    let mut calibration = Calibration::default();

    for i in 0..n {
        checkpoint().map_err(to_string)?;

        for (j, dist) in distances.iter_mut().enumerate() {
            *dist = if i == j {
                0.0
            } else {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = data[[a, k]] - data[[b, k]];
                    dist_sq += diff * diff;
                }
                dist_sq
            };
        }

        let mut row = p.row_mut(i);
        let mut beta = 1.0f64;
        let mut min_beta: Option<f64> = None;
        let mut max_beta: Option<f64> = None;
        let mut sum_p = f64::MIN_POSITIVE;
        let mut h_diff = f64::INFINITY;
        for _ in 0..PERPLEXITY_STEPS {
            sum_p = f64::MIN_POSITIVE;
            let mut h = 0.0;
            for j in 0..n {
                if j == i {
                    row[j] = 0.0;
                    continue;
                }
                let v = (-beta * distances[j]).exp();
                row[j] = v;
                sum_p += v;
                // an underflowed term adds nothing (and 0 x an overflowed distance is not NaN)
                if v > 0.0 {
                    h += beta * (distances[j] * v);
                }
            }
            h_diff = h / sum_p + sum_p.ln() - target_entropy;
            if h_diff.abs() < PERPLEXITY_TOL {
                break;
            }
            if h_diff > 0.0 {
                // too flat: narrow the kernel
                min_beta = Some(beta);
                beta = match max_beta {
                    None => beta * 2.0,
                    Some(hi) => (beta + hi) / 2.0,
                };
            } else {
                max_beta = Some(beta);
                beta = match min_beta {
                    None => beta / 2.0,
                    Some(lo) => (beta + lo) / 2.0,
                };
            }
        }
        if !(h_diff.abs() < PERPLEXITY_TOL) {
            calibration.missed += 1;
            calibration.worst = calibration.worst.max(h_diff.abs());
        }
        row /= sum_p;
    }
    Ok((p, calibration))
}

/// Initialize low-dimensional embedding.
fn initialize_embedding(
    data: ArrayView2<f64>,
    n_components: usize,
    seed: Option<u64>,
) -> Result<Array2<f64>, String> {
    let n_samples = data.nrows();

    // Try PCA initialization first
    if let Ok(pca_result) = pca(data, Some(n_components), true) {
        if let Some(transformed) = pca_result.transformed {
            // Scale down for better optimization
            return Ok(transformed * 0.0001);
        }
    }

    // Fallback to pseudo-random initialization using seed
    let mut embedding = try_zeros2(n_samples, n_components, "t-SNE initial embedding")
        .map_err(|e| e.to_string())?;
    let seed = seed.unwrap_or(42);

    for i in 0..n_samples {
        for j in 0..n_components {
            // Simple pseudo-random based on position and seed
            let hash = ((i * 31 + j * 17 + seed as usize) as f64).sin();
            embedding[[i, j]] = hash * 0.0001;
        }
    }

    Ok(embedding)
}

/// Compute Q distribution (Student-t with 1 degree of freedom).
/// Overwrites every off-diagonal cell of `q` (the diagonal is left as is)
/// and returns the sum of all Q values.
fn fill_q_distribution(y: &ArrayView2<f64>, q: &mut Array2<f64>) -> f64 {
    let n = y.nrows();
    let mut sum_q = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            // Squared distance in low-dimensional space
            let mut dist_sq = 0.0;
            for k in 0..y.ncols() {
                let diff = y[[i, k]] - y[[j, k]];
                dist_sq += diff * diff;
            }

            // Student-t kernel: (1 + ||y_i - y_j||^2)^(-1)
            let q_ij = 1.0 / (1.0 + dist_sq);
            q[[i, j]] = q_ij;
            q[[j, i]] = q_ij;
            sum_q += 2.0 * q_ij;
        }
    }

    sum_q
}

/// Compute gradient of KL divergence for `P * exaggeration`.
fn compute_gradient(
    p: &ArrayView2<f64>,
    exaggeration: f64,
    q: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    sum_q: f64,
) -> Array2<f64> {
    let n = y.nrows();
    let n_comp = y.ncols();
    let mut gradient = Array2::zeros((n, n_comp));

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let q_ij = q[[i, j]] / sum_q;
                // Factor: 4 * (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1)
                let factor = 4.0 * (p[[i, j]] * exaggeration - q_ij) * q[[i, j]];

                for k in 0..n_comp {
                    gradient[[i, k]] += factor * (y[[i, k]] - y[[j, k]]);
                }
            }
        }
    }

    gradient
}

/// KL(P || Q) in Rtsne's form, `sum_{i != j} p_ij ln((p_ij + FLT_MIN) / (q_ij + FLT_MIN))`
/// with `q_ij = Q_ij / sum_Q` and `FLT_MIN = f32::MIN_POSITIVE`.
fn compute_kl_divergence(p: &ArrayView2<f64>, q: &ArrayView2<f64>, sum_q: f64) -> f64 {
    let n = p.nrows();
    let tiny = f64::from(f32::MIN_POSITIVE);
    let mut kl = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let p_ij = p[[i, j]];
                let q_ij = q[[i, j]] / sum_q;
                kl += p_ij * ((p_ij + tiny) / (q_ij + tiny)).ln();
            }
        }
    }

    kl
}

// =============================================================================
// Classical Multidimensional Scaling (cmdscale)
// =============================================================================

use serde::{Deserialize, Serialize};

/// Classical Multidimensional Scaling (cmdscale) result.
///
/// # References
///
/// - Torgerson, W. S. (1952). "Multidimensional scaling: I. Theory and method".
///   Psychometrika, 17, 401–419.
/// - Mardia, K. V. (1978). "Some properties of classical multidimensional scaling".
///   Communications in Statistics - Theory and Methods, A7, 1233–1241.
/// - Implementation follows R's cmdscale from the stats package.
///   Source: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmdscaleResult {
    /// Configuration matrix (n x k). Each row is a point in k-dimensional space.
    pub points: Vec<Vec<f64>>,
    /// All n eigenvalues of the double-centred matrix B, in descending
    /// algebraic order as `stats::cmdscale` returns them (negative
    /// eigenvalues, from non-Euclidean input, come last).
    pub eig: Vec<f64>,
    /// Goodness of fit, R's `GOF`: the sum of the positive eigenvalues among
    /// the first `k` over the sum of the absolute eigenvalues, and over the sum
    /// of the positive eigenvalues (`[1, 1]` when a denominator is 0, all
    /// distances 0).
    pub gof: [f64; 2],
    /// Number of dimensions in the output.
    pub k: usize,
    /// Number of points.
    pub n: usize,
    /// Whether the input was a distance matrix or data matrix.
    pub is_distance_matrix: bool,
    /// Cailliez's additive constant (R's `ac`): added to every off-diagonal
    /// distance with `add = Some(true)`; 0 otherwise.
    #[serde(default)]
    pub ac: f64,
}

impl std::fmt::Display for CmdscaleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Classical MDS (cmdscale) Results")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Number of points: {}", self.n)?;
        writeln!(f, "Output dimensions: {}", self.k)?;
        writeln!(f)?;
        writeln!(f, "Goodness of fit:")?;
        writeln!(
            f,
            "  GOF (k eigenvalues / sum |eigenvalues|): {:.4}",
            self.gof[0]
        )?;
        writeln!(
            f,
            "  GOF (k eigenvalues / sum of positive eigenvalues): {:.4}",
            self.gof[1]
        )?;
        if self.ac != 0.0 {
            writeln!(f, "Additive constant (Cailliez): {}", self.ac)?;
        }
        writeln!(f)?;

        // Show eigenvalues
        let n_show = self.eig.len().min(10);
        writeln!(f, "Eigenvalues (first {}):", n_show)?;
        for (i, &ev) in self.eig.iter().take(n_show).enumerate() {
            writeln!(f, "  λ{}: {:.6}", i + 1, ev)?;
        }
        if self.eig.len() > 10 {
            writeln!(f, "  ... ({} more)", self.eig.len() - 10)?;
        }
        writeln!(f)?;

        // Show first few points
        let n_pts = self.points.len().min(5);
        writeln!(f, "Configuration (first {} points):", n_pts)?;
        for i in 0..n_pts {
            let coords: Vec<String> = self.points[i].iter().map(|v| format!("{:.4}", v)).collect();
            writeln!(f, "  Point {}: [{}]", i + 1, coords.join(", "))?;
        }
        if self.points.len() > 5 {
            writeln!(f, "  ... ({} more points)", self.points.len() - 5)?;
        }

        Ok(())
    }
}

/// Classical Multidimensional Scaling (cmdscale).
///
/// Takes a distance matrix and returns a configuration of points in Euclidean
/// space that approximately reproduces the distances.
///
/// # Arguments
/// * `d` - Square symmetric distance matrix (n x n) as nested vectors
/// * `k` - Number of output dimensions (default: 2), clamped to `n - 1`
///   (R stops instead)
/// * `eig` - accepted for R compatibility; all `n` eigenvalues are always
///   returned
/// * `add` - `Some(true)` is R's `add = TRUE`: Cailliez's additive constant
///   `ac` (the largest real eigenvalue of R's `2n x 2n` matrix
///   `[[0, -DC(D^2)], [-I, DC(2D)]]`, `DC` the double centring) is added to
///   every off-diagonal distance before the embedding
///
/// # Returns
/// * `CmdscaleResult` containing the configuration and diagnostics
///
/// # Errors
///
/// - Fewer than 2 points; a non-square, non-symmetric or non-finite `d`, a
///   non-zero diagonal or a negative distance; `k = 0`.
/// - `"Resource limit: ..."` when the `n x n` matrix or the eigensolver's
///   workspace (about `3 n^2` `f64`) exceeds the per-allocation ceiling, and
///   with `add`, when the `2n x 2n` matrix of the additive constant or its
///   eigensolver's workspace does.
/// - `"Computation stopped: ..."` when the budget is cancelled or expires.
/// - A failed eigendecomposition.
///
/// # Algorithm
/// 1. Compute squared distance matrix D²
/// 2. Double-center: B = -½ H D² H where H = I - (1/n) 11'
/// 3. Eigendecompose B to get eigenvalues λ and eigenvectors V
/// 4. Points = V[:, 1:k] * diag(sqrt(λ[1:k]))
///
/// # References
///
/// - Torgerson, W. S. (1952). "Multidimensional scaling: I. Theory and method".
///   Psychometrika, 17, 401–419.
/// - R stats::cmdscale documentation
pub fn cmdscale(
    d: &[Vec<f64>],
    k: Option<usize>,
    eig: Option<bool>,
    add: Option<bool>,
) -> Result<CmdscaleResult, String> {
    let n = d.len();

    // Validate input
    if n < 2 {
        return Err("Need at least 2 points for MDS".to_string());
    }

    for (i, row) in d.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "Distance matrix must be square. Row {} has {} elements, expected {}",
                i,
                row.len(),
                n
            ));
        }
    }

    // Non-finite entries would pass the symmetry checks below
    for (i, row) in d.iter().enumerate() {
        if let Some(j) = row.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "Distance matrix must contain only finite values (d[{i},{j}] = {})",
                row[j]
            ));
        }
    }

    // Check symmetry and non-negative distances
    for i in 0..n {
        if d[i][i].abs() > 1e-10 {
            return Err(format!(
                "Diagonal element d[{},{}] = {} should be 0",
                i, i, d[i][i]
            ));
        }
        for j in (i + 1)..n {
            if (d[i][j] - d[j][i]).abs() > 1e-10 {
                return Err(format!(
                    "Distance matrix must be symmetric. d[{},{}]={} != d[{},{}]={}",
                    i, j, d[i][j], j, i, d[j][i]
                ));
            }
            if d[i][j] < 0.0 {
                return Err(format!(
                    "Distances must be non-negative. d[{},{}]={}",
                    i, j, d[i][j]
                ));
            }
        }
    }

    let k = cmdscale_dims(k, n)?;
    // All n eigenvalues are always returned: `eig` is accepted for R compatibility
    let _ = eig;

    if add == Some(true) {
        let ac = cailliez_constant(d)?;
        let shifted = |i: usize, j: usize| if i == j { 0.0 } else { d[i][j] + ac };
        let (points, eig_vec) = cmdscale_core(n, shifted, k)?;
        let mut result = cmdscale_result(points, eig_vec, k, n, true);
        result.ac = ac;
        return Ok(result);
    }
    let (points, eig_vec) = cmdscale_core(n, |i, j| d[i][j], k)?;
    Ok(cmdscale_result(points, eig_vec, k, n, true))
}

/// Writes `sign * DC(f)` into the `n x n` block of `z` at `(r0, c0)`, where `DC`
/// is R's `C_DoubleCentre`: the row means are subtracted, then the column means.
fn double_centre_into(
    z: &mut Mat<f64>,
    (r0, c0): (usize, usize),
    n: usize,
    f: impl Fn(usize, usize) -> f64,
    sign: f64,
) -> EconResult<()> {
    for i in 0..n {
        checkpoint()?;
        let mut sum = 0.0;
        for j in 0..n {
            let v = f(i, j);
            z[(r0 + i, c0 + j)] = v;
            sum += v;
        }
        let mean = sum / n as f64;
        for j in 0..n {
            z[(r0 + i, c0 + j)] -= mean;
        }
    }
    for j in 0..n {
        checkpoint()?;
        let mut sum = 0.0;
        for i in 0..n {
            sum += z[(r0 + i, c0 + j)];
        }
        let mean = sum / n as f64;
        for i in 0..n {
            let v = z[(r0 + i, c0 + j)] - mean;
            z[(r0 + i, c0 + j)] = sign * v;
        }
    }
    Ok(())
}

/// Cailliez's additive constant, as `stats::cmdscale(add = TRUE)`: the largest
/// real part among the eigenvalues of the `2n x 2n` matrix
/// `[[0, -DC(D^2)], [-I, DC(2D)]]`. The matrix and the general eigensolver's
/// workspace are pre-flighted against the allocation ceiling.
fn cailliez_constant(d: &[Vec<f64>]) -> Result<f64, String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, evd_real, evd_scratch};

    let to_string = |e: EconError| e.to_string();
    let n = d.len();
    let what = "cmdscale additive constant matrix";
    let m = n
        .checked_mul(2)
        .ok_or_else(|| crate::resource::overflow(what).to_string())?;
    checkpoint().map_err(to_string)?;
    check_alloc::<f64>(checked_len(&[m, m], what).map_err(to_string)?, what).map_err(to_string)?;
    let ws = "cmdscale additive constant eigensolver workspace";
    let par = faer::get_global_parallelism();
    let scratch = evd_scratch::<f64>(
        m,
        ComputeEigenvectors::No,
        ComputeEigenvectors::No,
        par,
        Default::default(),
    );
    if scratch.layout().is_err() {
        return Err(crate::resource::overflow(ws).to_string());
    }
    check_alloc::<u8>(scratch.size_bytes(), ws).map_err(to_string)?;

    let mut z = Mat::<f64>::zeros(m, m);
    double_centre_into(&mut z, (0, n), n, |i, j| d[i][j] * d[i][j], -1.0).map_err(to_string)?;
    double_centre_into(&mut z, (n, n), n, |i, j| 2.0 * d[i][j], 1.0).map_err(to_string)?;
    for i in 0..n {
        z[(n + i, i)] = -1.0;
    }

    checkpoint().map_err(to_string)?;
    let mut buf = MemBuffer::try_new(scratch)
        .map_err(|_| crate::resource::reserve_failed(ws, scratch.size_bytes(), 1).to_string())?;
    let mut re = faer::diag::Diag::<f64>::zeros(m);
    let mut im = faer::diag::Diag::<f64>::zeros(m);
    evd_real(
        z.as_ref(),
        re.as_mut(),
        im.as_mut(),
        None,
        None,
        par,
        MemStack::new(&mut buf),
        Default::default(),
    )
    .map_err(|_| "cmdscale additive constant: eigendecomposition failed".to_string())?;
    Ok(re
        .column_vector()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max))
}

/// The output dimension: `k` (default 2) clamped to `n - 1`; `k = 0` is refused.
fn cmdscale_dims(k: Option<usize>, n: usize) -> Result<usize, String> {
    match k.unwrap_or(2) {
        0 => Err("k must be at least 1 (got 0)".to_string()),
        k => Ok(k.min(n - 1)),
    }
}

/// Classical MDS on the distances `d(i, j)` of `n` points: returns the
/// `n x k` configuration and all `n` eigenvalues of the double-centred
/// matrix in descending order, as `stats::cmdscale` orders them.
///
/// Builds `B = -1/2 H D^2 H` directly from `d` (no separate `D^2` matrix)
/// and decomposes it with faer's self-adjoint eigensolver
/// (`faer::linalg::evd::self_adjoint_evd`, the routine and arguments
/// `Mat::self_adjoint_eigen` uses). Before `B` is built, the `n x n`
/// matrices (`B` and the eigenvectors) and the eigensolver's workspace, one
/// buffer the size of a few `n x n` matrices, are pre-flighted against the
/// allocation ceiling; the workspace is then allocated fallibly from the same
/// requirement that was checked.
fn cmdscale_core(
    n: usize,
    d: impl Fn(usize, usize) -> f64,
    k: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};

    let to_string = |e: EconError| e.to_string();
    checkpoint().map_err(to_string)?;
    let what = "cmdscale matrix";
    check_alloc::<f64>(checked_len(&[n, n], what).map_err(to_string)?, what).map_err(to_string)?;
    // The eigensolver's workspace: checked here, and allocated below from this
    // same requirement, so the two cannot drift apart
    let ws = "cmdscale eigensolver workspace";
    let par = faer::get_global_parallelism();
    let scratch =
        self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default());
    // faer signals an overflowing requirement as StackReq::OVERFLOW (size 0),
    // which check_alloc would pass
    if scratch.layout().is_err() {
        return Err(crate::resource::overflow(ws).to_string());
    }
    check_alloc::<u8>(scratch.size_bytes(), ws).map_err(to_string)?;

    // Row and column means of D^2, and its grand mean
    let mut row_means = vec![0.0f64; n];
    let mut col_means = vec![0.0f64; n];
    let mut total = 0.0f64;
    for i in 0..n {
        checkpoint().map_err(to_string)?;
        for j in 0..n {
            let d_ij = d(i, j);
            let sq = d_ij * d_ij;
            row_means[i] += sq;
            col_means[j] += sq;
            total += sq;
        }
    }
    for i in 0..n {
        row_means[i] /= n as f64;
        col_means[i] /= n as f64;
    }
    let grand_mean = total / (n as f64 * n as f64);

    // Double centering: B[i,j] = -0.5 * (D²[i,j] - row_mean[i] - col_mean[j] + grand_mean)
    let mut b = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        checkpoint().map_err(to_string)?;
        for j in 0..n {
            let d_ij = d(i, j);
            b[(i, j)] = -0.5 * (d_ij * d_ij - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Make B exactly symmetric (numerical stability)
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (b[(i, j)] + b[(j, i)]) / 2.0;
            b[(i, j)] = avg;
            b[(j, i)] = avg;
        }
    }

    checkpoint().map_err(to_string)?;
    let mut buf = MemBuffer::try_new(scratch)
        .map_err(|_| crate::resource::reserve_failed(ws, scratch.size_bytes(), 1).to_string())?;
    let mut vectors = Mat::<f64>::zeros(n, n);
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    self_adjoint_evd(
        b.as_ref(),
        s.as_mut(),
        Some(vectors.as_mut()),
        par,
        MemStack::new(&mut buf),
        Default::default(),
    )
    .map_err(|_| "cmdscale eigendecomposition failed".to_string())?;
    drop(buf);
    let values = s.column_vector();

    // Descending algebraic order (faer's own order is not relied on)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let eig: Vec<f64> = order.iter().map(|&j| values[j]).collect();

    // points[i, j] = eigenvectors[i, j] * sqrt(max(eigenvalues[j], 0))
    let what = "cmdscale configuration";
    checked_len(&[n, k], what).map_err(to_string)?;
    let mut points = try_with_capacity::<Vec<f64>>(n, what).map_err(to_string)?;
    for i in 0..n {
        let mut row = try_vec(k, 0.0f64, what).map_err(to_string)?;
        for (j, v) in row.iter_mut().enumerate() {
            *v = vectors[(i, order[j])] * eig[j].max(0.0).sqrt();
        }
        points.push(row);
    }

    Ok((points, eig))
}

/// Assemble a [`CmdscaleResult`], computing the goodness-of-fit measures.
fn cmdscale_result(
    points: Vec<Vec<f64>>,
    eig_vec: Vec<f64>,
    k: usize,
    n: usize,
    is_distance_matrix: bool,
) -> CmdscaleResult {
    // R's GOF: the positive eigenvalues among the first k (R drops the others)
    // over sum |eig| and over sum max(eig, 0)
    let sum_abs: f64 = eig_vec.iter().map(|&x| x.abs()).sum();
    let sum_pos: f64 = eig_vec.iter().filter(|&&x| x > 0.0).sum();
    let sum_k: f64 = eig_vec.iter().take(k).filter(|&&x| x > 0.0).sum();
    let share = |den: f64| if den > 0.0 { sum_k / den } else { 1.0 };
    let gof = [share(sum_abs), share(sum_pos)];

    CmdscaleResult {
        points,
        eig: eig_vec,
        gof,
        k,
        n,
        is_distance_matrix,
        ac: 0.0,
    }
}

/// Classical MDS from a data matrix (computes Euclidean distances first).
///
/// # Arguments
/// * `data` - Data matrix (n x p) where rows are observations
/// * `k` - Number of output dimensions (default: 2)
///
/// # Returns
/// * `CmdscaleResult` containing the configuration
///
/// # Errors
///
/// - Fewer than 2 rows, a non-finite value in `data`, or `k = 0`.
/// - `"Resource limit: ..."` when the `n x n` matrix or the eigensolver's
///   workspace (about `3 n^2` `f64`) exceeds the per-allocation ceiling.
/// - `"Computation stopped: ..."` when the budget is cancelled or expires.
/// - A failed eigendecomposition.
pub fn cmdscale_from_data(
    data: ArrayView2<f64>,
    k: Option<usize>,
) -> Result<CmdscaleResult, String> {
    let n = data.nrows();

    if n < 2 {
        return Err("Need at least 2 points for MDS".to_string());
    }
    if let Some(((i, j), v)) = data.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "data must contain only finite values (row {i}, column {j} is {v})"
        ));
    }
    let k = cmdscale_dims(k, n)?;

    // Euclidean distances computed on the fly (as R's dist), never stored
    let dist = |i: usize, j: usize| {
        if i == j {
            return 0.0;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let mut d_sq = 0.0;
        for col in 0..data.ncols() {
            let diff = data[[a, col]] - data[[b, col]];
            d_sq += diff * diff;
        }
        d_sq.sqrt()
    };

    let (points, eig_vec) = cmdscale_core(n, dist, k)?;
    Ok(cmdscale_result(points, eig_vec, k, n, false))
}

/// Run classical MDS (convenience wrapper).
pub fn run_cmdscale(
    d: &[Vec<f64>],
    k: Option<usize>,
    eig: Option<bool>,
    add: Option<bool>,
) -> Result<CmdscaleResult, String> {
    cmdscale(d, k, eig, add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    #[test]
    fn test_na_pca_refuses_non_finite_input() {
        // R prcomp: "infinite or missing values in 'x'". Before the fix: Ok with NaN
        // variances (or "SVD failed: NoConvergence", depending on the path).
        let mut x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0], [9.0, 1.0]];
        x[[4, 0]] = f64::NAN;
        let err = pca(x.view(), None, false).unwrap_err();
        assert!(
            err.starts_with("PCA: data must contain only finite values (row 4, column 0 is NaN)"),
            "{err}"
        );
        x[[4, 0]] = 9.0;
        x[[2, 1]] = f64::INFINITY;
        assert!(
            pca(x.view(), None, false)
                .unwrap_err()
                .contains("row 2, column 1 is inf")
        );
    }

    fn blobs(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, d), |(i, _)| (i % 3) as f64 * 3.0 + rng.r#gen::<f64>())
    }

    #[test]
    fn tsne_affinity_matrix_is_fallible() {
        let data = blobs(100, 3, 1);
        let e = with_alloc_ceiling(Some(8 * 100 * 100 - 8), || {
            tsne(data.view(), Some(2), Some(5.0), Some(1), None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("limit"), "{e}");
    }

    #[test]
    fn tsne_validates_parameters_data_and_cancellation() {
        let data = blobs(40, 3, 2);
        for ((c, p, it, lr), param) in [
            ((Some(2), Some(5.0), Some(0), Some(200.0)), "max_iterations"),
            ((Some(0), Some(5.0), Some(10), Some(200.0)), "n_components"),
            ((Some(2), Some(-1.0), Some(10), Some(200.0)), "perplexity"),
            (
                (Some(2), Some(f64::NAN), Some(10), Some(200.0)),
                "perplexity",
            ),
            ((Some(2), Some(5.0), Some(10), Some(0.0)), "learning_rate"),
        ] {
            let e = tsne(data.view(), c, p, it, lr, Some(1)).unwrap_err();
            assert!(e.starts_with(param), "{e}");
        }
        let mut bad = data.clone();
        bad[[3, 1]] = f64::INFINITY;
        assert!(
            tsne(bad.view(), Some(2), Some(5.0), Some(10), None, Some(1))
                .unwrap_err()
                .contains("finite")
        );
        let e = with_budget(cancelled(), || {
            tsne(data.view(), Some(2), Some(5.0), Some(10), None, Some(1))
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn cmdscale_preflights_the_eigensolver_workspace() {
        let data = blobs(60, 3, 3);
        // B and the eigenvectors (60 x 60 f64 = 28,800 B each) fit under this
        // ceiling; the eigensolver's single workspace buffer does not
        let e = with_alloc_ceiling(Some(8 * 60 * 60), || {
            cmdscale_from_data(data.view(), Some(2))
        })
        .unwrap_err();
        assert!(e.contains("eigensolver") && e.contains("limit"), "{e}");
        assert!(cmdscale_from_data(data.view(), Some(2)).is_ok());
    }

    /// Golden (characterization) test: captured on the code before the
    /// memory refactor (row-wise P, scalar exaggeration, reused Q); every
    /// floating-point operation is preserved, so the bits must not change.
    ///
    /// The constants were captured on x86_64 Linux (glibc, AVX2+FMA
    /// kernels), and only there are the bits compared
    /// (`crate::ml::golden_bits_platform`). Elsewhere libm's `exp`/`ln` (the
    /// perplexity search) and the kernels behind the PCA start can differ in
    /// the last ulp, and the gain updates, which branch on gradient signs,
    /// amplify that: moving inputs by one to three ulps (496 runs: single
    /// data entries, all of them, random subsets, the perplexity and the
    /// learning rate) moved these values by up to 8.1e-4 relative. Other
    /// targets therefore compare to a relative 1e-2.
    /// The 15 x 3 data of the t-SNE golden and step tests: three tight groups.
    fn tsne_fixture() -> Array2<f64> {
        array![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.0, 0.1],
            [0.0, 0.2, 0.1],
            [0.1, 0.1, 0.2],
            [5.0, 5.0, 5.0],
            [5.1, 5.0, 5.1],
            [5.0, 5.2, 5.0],
            [5.2, 5.1, 5.1],
            [5.0, 5.1, 5.2],
            [10.0, 0.0, 5.0],
            [10.1, 0.1, 5.1],
            [10.0, 0.2, 5.0],
            [10.2, 0.0, 5.2],
            [10.1, 0.1, 5.0],
        ]
    }

    /// Row entropies `-sum p ln p` of conditional affinity rows.
    fn row_entropies(p: &Array2<f64>) -> Vec<f64> {
        p.rows()
            .into_iter()
            .map(|r| {
                -r.iter()
                    .filter(|&&v| v > 0.0)
                    .map(|&v| v * v.ln())
                    .sum::<f64>()
            })
            .collect()
    }

    /// ML-37: every point's bandwidth reaches the perplexity, on standard-normal data in 20
    /// dimensions and on the same data rescaled (the old linear sigma search left about 220 of
    /// 300 rows all zero here).
    #[test]
    fn test_tsne_affinities_calibrate_every_row() {
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(20261118);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = Array2::from_shape_fn((300, 20), |_| normal.sample(&mut rng));
        let target = 30f64.ln();
        let base = conditional_affinities(&x.view(), 30.0).unwrap().0;
        for scale in [1.0, 10.0, 0.01] {
            let xs = &x * scale;
            let p = conditional_affinities(&xs.view(), 30.0).unwrap().0;
            let zero = p.rows().into_iter().filter(|r| r.sum() == 0.0).count();
            assert_eq!(zero, 0, "scale {scale}: {zero} of 300 rows are all zero");
            for (i, r) in p.rows().into_iter().enumerate() {
                assert!(
                    (r.sum() - 1.0).abs() <= 1e-12,
                    "scale {scale}, row {i}: {}",
                    r.sum()
                );
            }
            for (i, h) in row_entropies(&p).into_iter().enumerate() {
                assert!((h - target).abs() < 1e-5, "scale {scale}, row {i}: H = {h}");
            }
            // the search is scale-free up to its entropy tolerance: each row agrees to 1e-4 of
            // its largest entry (a relative error in beta moves an entry by beta d^2 times it)
            for (row, base_row) in p.rows().into_iter().zip(base.rows()) {
                let top = base_row.iter().copied().fold(0.0, f64::max);
                for (a, b) in row.iter().zip(base_row.iter()) {
                    assert!((a - b).abs() <= 1e-4 * top, "scale {scale}: {a} vs {b}");
                }
            }
        }
    }

    /// The search cannot reach a perplexity below the number of exact duplicates: those rows are
    /// written anyway (uniform over the duplicates) and reported.
    #[test]
    fn test_tsne_calibration_warns_when_unreached() {
        let mut rows = vec![[0.0, 0.0]; 9];
        rows.extend([[10.0, 10.0], [10.5, 10.0], [10.0, 11.0]]);
        let x = Array2::from_shape_fn((12, 2), |(i, j)| rows[i][j]);
        let r = tsne(x.view(), Some(2), Some(3.9), Some(5), None, Some(1)).unwrap();
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        let w = &r.warnings[0];
        let miss = 8f64.ln() - 3.9f64.ln();
        assert!(
            w.contains("9 of 12 points") && w.contains(&format!("{miss:.4}")),
            "{w}"
        );
        assert!(r.to_string().contains(&format!("Warning: {w}")));
        let p = conditional_affinities(&x.view(), 3.9).unwrap().0;
        for i in 0..9 {
            let row = p.row(i);
            assert!((row.sum() - 1.0).abs() <= 1e-12, "row {i}: {}", row.sum());
            assert!(
                row.iter()
                    .take(9)
                    .enumerate()
                    .all(|(j, &v)| j == i || (v - 0.125).abs() < 1e-12)
            );
        }
    }

    /// ML-37: the joint affinities carry no 1e-12 floor and sum to one (perplexity 2: at 4 the
    /// kernel must reach past the four neighbours of a group, and the cross-group entries are
    /// about 6e-11).
    #[test]
    fn test_tsne_affinities_have_no_floor() {
        let data = tsne_fixture();
        let (p, _) = compute_joint_probabilities(&data.view(), 2.0).unwrap();
        assert!((p.sum() - 1.0).abs() <= 1e-12, "{}", p.sum());
        for i in 0..15 {
            for j in 0..15 {
                if i / 5 != j / 5 {
                    assert!(p[[i, j]] < 1e-12, "P[{i},{j}] = {}", p[[i, j]]);
                }
            }
        }
    }

    /// ML-145: one iteration is bhtsne's first step. Velocity starts at 0 and sign(0) = 0, so
    /// every gain becomes 1.2 (gains + 0.2) wherever the gradient is non-zero; momentum times 0
    /// vanishes; then Y is centred.
    #[test]
    fn test_tsne_first_step_matches_bhtsne_update() {
        let data = tsne_fixture();
        let (p, _) = compute_joint_probabilities(&data.view(), 4.0).unwrap();
        let y0 = initialize_embedding(data.view(), 2, Some(42)).unwrap();
        let mut q = Array2::zeros((15, 15));
        let sq = fill_q_distribution(&y0.view(), &mut q);
        // exaggeration 4 on iteration 0
        let g = compute_gradient(&p.view(), 4.0, &q.view(), &y0.view(), sq);
        assert!(
            g.iter().any(|&v| v < 0.0),
            "the fixture must have a negative gradient entry"
        );
        let gain = g.mapv(|v| if v == 0.0 { 1.0 } else { 1.2 });
        let mut want = &y0 - &(200.0 * &gain * &g);
        let mean = want.mean_axis(Axis(0)).unwrap();
        for mut row in want.rows_mut() {
            row -= &mean;
        }
        let r = tsne(
            data.view(),
            Some(2),
            Some(4.0),
            Some(1),
            Some(200.0),
            Some(42),
        )
        .unwrap();
        for (a, b) in r.embedding.iter().zip(want.iter()) {
            // before: entries with g < 0 used gain 0.8
            assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }

    /// ML-146 (and ML-97's KL part): the reported KL is KL(P || Q(Y)) of the returned Y, with
    /// the unexaggerated P.
    #[test]
    fn test_tsne_reports_kl_of_returned_embedding() {
        let data = tsne_fixture();
        let (p, _) = compute_joint_probabilities(&data.view(), 4.0).unwrap();
        // 1 and 50: inside early exaggeration
        for iters in [1usize, 50, 120] {
            let r = tsne(
                data.view(),
                Some(2),
                Some(4.0),
                Some(iters),
                Some(200.0),
                Some(42),
            )
            .unwrap();
            let mut q = Array2::zeros((15, 15));
            let sq = fill_q_distribution(&r.embedding.view(), &mut q);
            let want = compute_kl_divergence(&p.view(), &q.view(), sq);
            assert!(
                (r.kl_divergence - want).abs() <= 1e-12 * want.abs(),
                "iters={iters}: {} vs {want}",
                r.kl_divergence
            );
        }
    }

    /// Re-captured at T3-WP42 (ML-145 changes the path from step 1, ML-146 the reported KL), and
    /// for ML-37 (bhtsne's bandwidth search, no floor on P, Rtsne's KL form).
    #[test]
    fn tsne_memory_refactor_is_bit_identical() {
        const GOLDEN: [u64; 4] = [
            13849809419206569802,
            13871516010347534690,
            13867331416230125663,
            4610700378489271804,
        ];
        let data = tsne_fixture();
        // 120 iterations cross the early-exaggeration switch at 100
        let r = tsne(
            data.view(),
            Some(2),
            Some(4.0),
            Some(120),
            Some(200.0),
            Some(42),
        )
        .unwrap();
        let got = [
            r.embedding[[0, 0]],
            r.embedding[[7, 1]],
            r.embedding[[14, 0]],
            r.kl_divergence,
        ];
        let bits = got.map(f64::to_bits);
        if crate::ml::golden_bits_platform() {
            assert_eq!(bits, GOLDEN, "got {bits:?} = {got:?}");
        } else {
            for (g, want) in got.iter().zip(GOLDEN.map(f64::from_bits)) {
                assert!(
                    (g - want).abs() <= 1e-2 * want.abs(),
                    "got {bits:?} = {got:?}"
                );
            }
        }
    }

    /// R 4.3.3:
    /// x <- matrix(c(0.12,1.05,2.31, 0.48,0.31,1.74, 2.91,3.17,0.52, 3.36,2.62,1.18,
    ///               1.07,0.74,2.95, 5.21,0.43,0.66, 4.83,1.12,1.41, 0.71,2.38,3.27), ncol=3, byrow=TRUE)
    /// cmdscale(dist(x), k=2, eig=TRUE); cmdscale(dist(x, "manhattan"), k=2, eig=TRUE)
    #[test]
    fn test_validate_cmdscale_matches_r() {
        let x = array![
            [0.12, 1.05, 2.31],
            [0.48, 0.31, 1.74],
            [2.91, 3.17, 0.52],
            [3.36, 2.62, 1.18],
            [1.07, 0.74, 2.95],
            [5.21, 0.43, 0.66],
            [4.83, 1.12, 1.41],
            [0.71, 2.38, 3.27]
        ];
        fn match_up_to_sign(got: &[Vec<f64>], want: &[[f64; 2]]) {
            for j in 0..2 {
                let dot: f64 = (0..want.len()).map(|i| got[i][j] * want[i][j]).sum();
                let s = if dot >= 0.0 { 1.0 } else { -1.0 };
                for i in 0..want.len() {
                    assert!(
                        (got[i][j] - s * want[i][j]).abs() < 1e-9,
                        "point {i} dim {j}: {} vs {}",
                        got[i][j],
                        s * want[i][j]
                    );
                }
            }
        }
        let e = cmdscale_from_data(x.view(), Some(2)).unwrap();
        match_up_to_sign(
            &e.points,
            &[
                [-2.279060593438, 0.262854553433],
                [-1.766554460130, 0.963788061328],
                [1.083580183186, -1.758390545875],
                [1.220858628800, -1.092317721778],
                [-1.660497422080, 0.740094188594],
                [3.018416293670, 1.197864000051],
                [2.416501971425, 0.565340104362],
                [-2.033244601432, -0.879232640115],
            ],
        );
        let r_eig = [
            3.282112765185e+01,
            8.358352300206e+00,
            2.785057547942e+00,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (a, b) in e.eig.iter().zip(r_eig) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }

        let d: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| (0..3).map(|c| (x[[i, c]] - x[[j, c]]).abs()).sum())
                    .collect()
            })
            .collect();
        let m = cmdscale(&d, Some(2), Some(true), None).unwrap();
        match_up_to_sign(
            &m.points,
            &[
                [-2.941543784550, 1.005679440080],
                [-2.024496365738, 1.764954247587],
                [2.297951533599, -3.012785552745],
                [2.003948448025, -2.159470877549],
                [-2.712879429698, 1.158170137280],
                [4.246989775681, 2.168300401165],
                [2.827611433079, 1.271322259857],
                [-3.697581610398, -2.196170055674],
            ],
        );
        // R orders eigenvalues algebraically: the negative ones come last
        let r_eig_m = [
            6.911178914128e+01,
            3.034895378852e+01,
            5.192720774500e+00,
            2.741845069069e+00,
            8.437653216063e-01,
            0.0,
            -2.671490215896e+00,
            -7.475446379076e+00,
        ];
        for (a, b) in m.eig.iter().zip(r_eig_m) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn cmdscale_buffers_are_fallible_and_inputs_validated() {
        let data = blobs(40, 3, 3);
        let e = with_alloc_ceiling(Some(4096), || cmdscale_from_data(data.view(), Some(2)))
            .unwrap_err();
        assert!(e.contains("limit"), "{e}");
        assert!(
            cmdscale_from_data(data.view(), Some(0))
                .unwrap_err()
                .starts_with("k ")
        );
        let e = with_budget(cancelled(), || cmdscale_from_data(data.view(), Some(2))).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        let mut d = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        d[0][2] = f64::NAN;
        d[2][0] = f64::NAN;
        assert!(
            cmdscale(&d, Some(1), None, None)
                .unwrap_err()
                .contains("finite")
        );
    }

    #[test]
    fn test_pca_basic() {
        // Simple correlated data
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0],];

        let result = pca(data.view(), Some(2), true).unwrap();

        assert_eq!(result.n_components, 2);
        assert!(result.explained_variance_ratio[0] > 0.99); // First PC should explain almost all variance
        assert!(result.transformed.is_some());
    }

    #[test]
    fn test_pca_reconstruction() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        let result = pca(data.view(), Some(2), true).unwrap();
        let transformed = result.transformed.as_ref().unwrap();
        let reconstructed = pca_inverse_transform(transformed.view(), &result).unwrap();

        // With 2 components out of 3, reconstruction won't be perfect
        // but should be reasonable
        assert_eq!(reconstructed.shape(), data.shape());
    }

    #[test]
    fn test_tsne_basic() {
        // Create clustered data
        let data = array![
            // Cluster 1
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.0, 0.1],
            [0.0, 0.2, 0.1],
            [0.1, 0.1, 0.2],
            // Cluster 2
            [5.0, 5.0, 5.0],
            [5.1, 5.0, 5.1],
            [5.0, 5.2, 5.0],
            [5.2, 5.1, 5.1],
            [5.0, 5.1, 5.2],
            // Cluster 3
            [10.0, 0.0, 5.0],
            [10.1, 0.1, 5.1],
            [10.0, 0.2, 5.0],
            [10.2, 0.0, 5.2],
            [10.1, 0.1, 5.0],
        ];

        let result = tsne(
            data.view(),
            Some(2),   // 2D embedding
            Some(4.0), // Low perplexity for small dataset (must be < n/3 = 5)
            Some(100), // Fewer iterations for test speed
            Some(200.0),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.n_components, 2);
        assert_eq!(result.embedding.nrows(), 15);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.kl_divergence >= 0.0);
    }

    #[test]
    fn test_tsne_perplexity_validation() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];

        // Perplexity too high (>= n/3)
        let result = tsne(
            data.view(),
            Some(2),
            Some(3.0), // n/3 = 2, so perplexity=3 should fail
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // cmdscale (Classical MDS) tests
    // =========================================================================

    #[test]
    fn test_cmdscale_basic() {
        // Simple 4-point example: square with side 1
        // Points at: (0,0), (1,0), (1,1), (0,1)
        // Distances: adjacent pairs = 1, diagonals = sqrt(2) ≈ 1.414
        let dist = vec![
            vec![0.0, 1.0, 1.414, 1.0],
            vec![1.0, 0.0, 1.0, 1.414],
            vec![1.414, 1.0, 0.0, 1.0],
            vec![1.0, 1.414, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), Some(true), None).unwrap();

        assert_eq!(result.n, 4);
        assert_eq!(result.k, 2);
        assert_eq!(result.points.len(), 4);
        assert_eq!(result.points[0].len(), 2);

        // GOF should be high for perfect Euclidean distances
        assert!(
            result.gof[0] > 0.9,
            "GOF1 should be high: {}",
            result.gof[0]
        );
        assert!(
            result.gof[1] > 0.9,
            "GOF2 should be high: {}",
            result.gof[1]
        );
    }

    #[test]
    fn test_cmdscale_distance_preservation() {
        // Create a simple triangle
        let dist = vec![
            vec![0.0, 3.0, 4.0],
            vec![3.0, 0.0, 5.0],
            vec![4.0, 5.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), Some(true), None).unwrap();

        // Verify distances are approximately preserved
        let pts = &result.points;
        let d01 = ((pts[0][0] - pts[1][0]).powi(2) + (pts[0][1] - pts[1][1]).powi(2)).sqrt();
        let d02 = ((pts[0][0] - pts[2][0]).powi(2) + (pts[0][1] - pts[2][1]).powi(2)).sqrt();
        let d12 = ((pts[1][0] - pts[2][0]).powi(2) + (pts[1][1] - pts[2][1]).powi(2)).sqrt();

        // This is a 3-4-5 right triangle, should be exact in 2D
        assert!((d01 - 3.0).abs() < 0.1, "d01={}, expected 3.0", d01);
        assert!((d02 - 4.0).abs() < 0.1, "d02={}, expected 4.0", d02);
        assert!((d12 - 5.0).abs() < 0.1, "d12={}, expected 5.0", d12);
    }

    #[test]
    fn test_cmdscale_from_data() {
        // 2D data points
        let data = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        let result = cmdscale_from_data(data.view(), Some(2)).unwrap();

        assert_eq!(result.n, 4);
        assert_eq!(result.k, 2);
        assert!(!result.is_distance_matrix);

        // Configuration should preserve relative positions
        // Check that we have 4 points with 2 coordinates each
        assert_eq!(result.points.len(), 4);
        for pt in &result.points {
            assert_eq!(pt.len(), 2);
        }
    }

    #[test]
    fn test_cmdscale_eigenvalues() {
        // Perfect Euclidean distances should give non-negative eigenvalues
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), Some(true), None).unwrap();

        // Check eigenvalues are returned
        assert_eq!(result.eig.len(), 3);

        // First eigenvalue should be positive (largest)
        assert!(
            result.eig[0] >= 0.0,
            "First eigenvalue should be non-negative"
        );
    }

    #[test]
    fn test_cmdscale_validation_non_square() {
        // Non-square matrix should fail
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0], // Wrong length
        ];

        let result = cmdscale(&dist, Some(2), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmdscale_validation_non_symmetric() {
        // Non-symmetric matrix should fail
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.5, 0.0, 1.0], // 1.5 != 1.0
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmdscale_validation_negative_distance() {
        // Negative distances should fail
        let dist = vec![
            vec![0.0, -1.0, 2.0],
            vec![-1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmdscale_validation_non_zero_diagonal() {
        // Non-zero diagonal should fail
        let dist = vec![
            vec![0.1, 1.0, 2.0], // Should be 0.0
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmdscale_1d() {
        // Points on a line: distances 0, 1, 2, 3
        let dist = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 0.0, 1.0, 2.0],
            vec![2.0, 1.0, 0.0, 1.0],
            vec![3.0, 2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(1), Some(true), None).unwrap();

        assert_eq!(result.k, 1);

        // In 1D, points should be collinear
        // The relative positions should be 0, 1, 2, 3 (up to sign and translation)
        let pts: Vec<f64> = result.points.iter().map(|p| p[0]).collect();

        // Check that distances between adjacent points are approximately 1
        let d01 = (pts[0] - pts[1]).abs();
        let d12 = (pts[1] - pts[2]).abs();
        let d23 = (pts[2] - pts[3]).abs();

        assert!((d01 - 1.0).abs() < 0.1, "d01={}", d01);
        assert!((d12 - 1.0).abs() < 0.1, "d12={}", d12);
        assert!((d23 - 1.0).abs() < 0.1, "d23={}", d23);
    }

    #[test]
    fn test_cmdscale_gof() {
        // Test GOF calculation with known distances
        let dist = vec![
            vec![0.0, 1.0, 1.414, 1.0],
            vec![1.0, 0.0, 1.0, 1.414],
            vec![1.414, 1.0, 0.0, 1.0],
            vec![1.0, 1.414, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), Some(true), None).unwrap();

        // GOF values should be between 0 and 1
        assert!(result.gof[0] >= 0.0 && result.gof[0] <= 1.0);
        assert!(result.gof[1] >= 0.0 && result.gof[1] <= 1.0);

        // R's pair shares a numerator; sum |eig| >= sum max(eig, 0), so GOF[0] <= GOF[1]
        assert!(result.gof[0] <= result.gof[1] + 1e-12);

        // ML-98: the text names R's two measures, and the additive constant when add was used
        let text = result.to_string();
        assert!(
            text.contains("GOF (k eigenvalues / sum |eigenvalues|)")
                && text.contains("GOF (k eigenvalues / sum of positive eigenvalues)")
                && !text.contains("Additive constant"),
            "{text}"
        );
        let added = cmdscale(&dist, Some(2), None, Some(true)).unwrap();
        assert!(added.ac > 0.0, "{}", added.ac);
        assert!(
            added.to_string().contains("Additive constant (Cailliez): "),
            "{added}"
        );
    }

    /// The 2n x 2n eigenproblem of add = TRUE is pre-flighted; without add it is never built.
    #[test]
    fn test_cmdscale_add_preflights_its_matrix() {
        let data = blobs(30, 3, 4);
        let n = data.nrows();
        let d: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..3).map(|c| (data[[i, c]] - data[[j, c]]).abs()).sum())
                    .collect()
            })
            .collect();
        let ceiling = Some(4 * n * n * 8 - 8);
        let e =
            with_alloc_ceiling(ceiling, || cmdscale(&d, Some(2), None, Some(true))).unwrap_err();
        assert!(
            e.starts_with("Resource limit") && e.contains("cmdscale additive constant"),
            "{e}"
        );
        assert!(with_alloc_ceiling(ceiling, || cmdscale(&d, Some(2), None, None)).is_ok());
        let e = with_budget(cancelled(), || cmdscale(&d, Some(2), None, Some(true))).unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn test_cmdscale_display() {
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&dist, Some(2), Some(true), None).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Classical MDS"));
        assert!(display.contains("Number of points: 3"));
        assert!(display.contains("Output dimensions: 2"));
        assert!(display.contains("Goodness of fit"));
    }

    // ========================================================================
    // R-vs-Rust Validation Tests (Phase 9)
    // ========================================================================

    fn create_validation_pca_data() -> Array2<f64> {
        // Create data with clear principal components
        // 10 samples x 4 features
        // First PC: high variance along (1, 1, 0, 0)
        // Second PC: lower variance along (0, 0, 1, 1)
        array![
            [1.0, 1.1, 0.5, 0.4],
            [2.0, 2.2, 0.3, 0.5],
            [3.0, 2.8, 0.6, 0.3],
            [4.0, 4.1, 0.2, 0.6],
            [5.0, 5.3, 0.4, 0.4],
            [1.5, 1.4, 0.8, 0.9],
            [2.5, 2.6, 1.0, 0.8],
            [3.5, 3.4, 0.9, 1.1],
            [4.5, 4.3, 1.2, 0.7],
            [5.5, 5.1, 1.1, 1.2],
        ]
    }

    #[test]
    fn test_pca_n_components_shapes() {
        // R reference: eigenvalues in decreasing order
        let data = create_validation_pca_data();
        let result = pca(data.view(), None, false).unwrap();

        // Explained variance should be in decreasing order
        for i in 1..result.n_components {
            assert!(
                result.explained_variance[i] <= result.explained_variance[i - 1] + 1e-10,
                "PC{} variance {} should be <= PC{} variance {}",
                i + 1,
                result.explained_variance[i],
                i,
                result.explained_variance[i - 1]
            );
        }

        // First PC should explain most variance
        assert!(
            result.explained_variance_ratio[0] > 0.5,
            "First PC should explain > 50% of variance"
        );
    }

    #[test]
    fn test_pca_components_orthonormal() {
        let data = create_validation_pca_data();
        let result = pca(data.view(), None, false).unwrap();

        // Components should be orthonormal
        for i in 0..result.n_components {
            // Each component should have unit length
            let norm: f64 = result
                .components
                .row(i)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "PC{} should have unit length, got {}",
                i + 1,
                norm
            );

            // Components should be orthogonal to each other
            for j in (i + 1)..result.n_components {
                let dot: f64 = result
                    .components
                    .row(i)
                    .iter()
                    .zip(result.components.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 0.01,
                    "PC{} and PC{} should be orthogonal, dot product = {}",
                    i + 1,
                    j + 1,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_pca_transformation_matches_projection() {
        let data = create_validation_pca_data();
        let result = pca(data.view(), Some(2), true).unwrap();

        let transformed = result.transformed.expect("Transformation requested");

        // Shape should be (10, 2) for 2 components
        assert_eq!(transformed.nrows(), 10);
        assert_eq!(transformed.ncols(), 2);

        // Transformed data should be centered (mean ≈ 0)
        for j in 0..2 {
            let col_mean: f64 = transformed.column(j).iter().sum::<f64>() / 10.0;
            assert!(
                col_mean.abs() < 0.01,
                "Transformed PC{} should have mean ≈ 0, got {}",
                j + 1,
                col_mean
            );
        }

        // Variance of PC1 transformation should be larger than PC2
        let var1: f64 = transformed.column(0).iter().map(|x| x * x).sum::<f64>() / 10.0;
        let var2: f64 = transformed.column(1).iter().map(|x| x * x).sum::<f64>() / 10.0;
        assert!(
            var1 >= var2 - 0.01,
            "PC1 variance {} should be >= PC2 variance {}",
            var1,
            var2
        );
    }

    #[test]
    fn test_pca_n_components_bound() {
        let data = create_validation_pca_data();

        // Request fewer components
        let result2 = pca(data.view(), Some(2), false).unwrap();
        assert_eq!(result2.n_components, 2);
        assert_eq!(result2.components.nrows(), 2);
        assert_eq!(result2.explained_variance.len(), 2);

        let result3 = pca(data.view(), Some(3), false).unwrap();
        assert_eq!(result3.n_components, 3);
    }

    #[test]
    fn test_cmdscale_preserves_euclidean_distances() {
        // MDS should preserve original distances
        let original_dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let result = cmdscale(&original_dist, Some(1), Some(true), None).unwrap();

        // Reconstruct distances from MDS coordinates
        let pts = &result.points;
        let d01 = ((pts[0][0] - pts[1][0]).powi(2)).sqrt();
        let d12 = ((pts[1][0] - pts[2][0]).powi(2)).sqrt();
        let d02 = ((pts[0][0] - pts[2][0]).powi(2)).sqrt();

        // Distances should approximately match original
        assert!((d01 - 1.0).abs() < 0.1, "d01 should be ≈ 1.0, got {}", d01);
        assert!((d12 - 1.0).abs() < 0.1, "d12 should be ≈ 1.0, got {}", d12);
        assert!((d02 - 2.0).abs() < 0.1, "d02 should be ≈ 2.0, got {}", d02);
    }

    #[test]
    fn test_tsne_preserves_local_structure() {
        // t-SNE should preserve local neighborhood structure
        // Create 3 tight clusters with enough points for perplexity requirement
        let data = array![
            // Cluster 1 (close together)
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, 0.15],
            [0.08, 0.02],
            // Cluster 2 (close together, far from cluster 1)
            [10.0, 0.0],
            [10.1, 0.1],
            [9.95, 0.15],
            [10.05, 0.08],
            // Cluster 3 (close together, far from both)
            [5.0, 10.0],
            [5.1, 10.1],
            [4.95, 10.15],
            [5.02, 10.05],
        ];

        let result = tsne(
            data.view(),
            Some(2),     // n_components
            Some(3.0),   // perplexity (must be < n_samples/3)
            Some(500),   // max_iter
            Some(200.0), // learning_rate
            Some(42),    // seed
        )
        .unwrap();

        assert_eq!(result.n_iterations, 500);
        assert_eq!(result.embedding.nrows(), 12);
        assert_eq!(result.embedding.ncols(), 2);

        // KL divergence should be finite and positive
        assert!(result.kl_divergence.is_finite());
        assert!(result.kl_divergence > 0.0);
    }

    #[test]
    fn test_pca_deterministic() {
        // PCA should be deterministic (no random state)
        let data = create_validation_pca_data();

        let result1 = pca(data.view(), Some(2), true).unwrap();
        let result2 = pca(data.view(), Some(2), true).unwrap();

        // Explained variances should be identical
        for i in 0..2 {
            assert!(
                (result1.explained_variance[i] - result2.explained_variance[i]).abs() < 1e-10,
                "PCA should be deterministic"
            );
        }
    }

    #[test]
    fn test_pca_centering() {
        let data = create_validation_pca_data();
        let result = pca(data.view(), None, false).unwrap();

        // Mean should match column means
        for j in 0..4 {
            let col_mean: f64 = data.column(j).iter().sum::<f64>() / 10.0;
            assert!(
                (result.mean[j] - col_mean).abs() < 1e-10,
                "Mean mismatch for feature {}: {} vs {}",
                j,
                result.mean[j],
                col_mean
            );
        }
    }

    use crate::ml::hclust::test_util::{case_frame, matrix};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_prcomp() {
        // prcomp(X, scale. = TRUE): sdev 1.3221524880813766 0.9819157249713941
        // 0.5364273551135557; prcomp(X) unscaled. Rotation signs are arbitrary: absolute values.
        // n = 60 > 4p: the covariance path (the SVD path: test_validate_prcomp_svd_path).
        let (case, ds) = crate::test_support::load_case("kmeans_lloyd_pca_mds");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let sd = x.std_axis(ndarray::Axis(0), 1.0);
        let z = (&x - &mean) / &sd;
        let mut c = Check::new(&case);
        for (data, pre) in [(&z, "pca_"), (&x, "pca_unscaled_")] {
            let r = pca(data.view(), None, false).unwrap();
            let sdev: Vec<f64> = r.explained_variance.iter().map(|v| v.sqrt()).collect();
            // rows = PCs: R's rotation in column-major order
            let rot: Vec<f64> = r.components.iter().map(|v| v.abs()).collect();
            c.vector(&format!("{pre}sdev"), &sdev, Tol::stat(60))
                .vector(&format!("{pre}rotation_abs"), &rot, Tol::coef(60))
                .vector(
                    &format!("{pre}var_ratio"),
                    r.explained_variance_ratio.as_slice().unwrap(),
                    Tol::stat(60),
                );
        }
        c.finish();
    }

    /// ML-35: on the SVD path (n <= 4p) with fewer components than min(n, p), the shares were
    /// taken over the kept components only (0.65135, 0.34865 here, against prcomp's 0.58270,
    /// 0.31193).
    #[test]
    fn test_validate_prcomp_svd_path() {
        let (case, ds) = crate::test_support::load_case("pca_svd_path");
        let v: Vec<String> = (1..=8).map(|i| format!("v{i}")).collect();
        let v: Vec<&str> = v.iter().map(String::as_str).collect();
        let w: Vec<String> = (1..=10).map(|i| format!("w{i}")).collect();
        let w: Vec<&str> = w.iter().map(String::as_str).collect();
        let tall = matrix(ds.df(), &v);
        let wide = matrix(&case_frame(&case, "pca_wide_6x10.csv"), &w);
        let sdev =
            |r: &PCAResult| -> Vec<f64> { r.explained_variance.iter().map(|v| v.sqrt()).collect() };
        let ratios = |r: &PCAResult| r.explained_variance_ratio.to_vec();
        // a share of the kept components equals prcomp's share of the same component
        let prefix = |r: &PCAResult, key: &str, tol: Tol| {
            let want = case.vector(key);
            for (i, &got) in r.explained_variance_ratio.iter().enumerate() {
                assert!(
                    tol.allows(got, want[i]),
                    "{key}[{i}] with k = {}: {got} vs R {}",
                    r.n_components,
                    want[i]
                );
            }
        };
        let mut c = Check::new(&case);

        let all = pca(tall.view(), None, false).unwrap();
        c.vector("tall_sdev", &sdev(&all), Tol::stat(20))
            .vector("tall_prop", &ratios(&all), Tol::stat(20))
            .scalar("tall_total_variance", all.total_variance, Tol::stat(20));
        let k2 = pca(tall.view(), Some(2), false).unwrap();
        prefix(&k2, "tall_prop", Tol::stat(20));
        // rows = PCs: R's p x 2 rotation in column-major order
        let rot: Vec<f64> = k2.components.iter().map(|v| v.abs()).collect();
        c.scalar("tall_total_variance", k2.total_variance, Tol::stat(20))
            .vector("tall_rotation_abs_k2", &rot, Tol::coef(20));

        // n < p: six singular values, the last one roundoff (R 1.45e-16)
        let all = pca(wide.view(), None, false).unwrap();
        c.vector("wide_sdev", &sdev(&all), Tol::new(Tol::stat(6).rel, 1e-12))
            .vector("wide_prop", &ratios(&all), Tol::stat(6))
            .scalar("wide_total_variance", all.total_variance, Tol::stat(6));
        let k3 = pca(wide.view(), Some(3), false).unwrap();
        prefix(&k3, "wide_prop", Tol::stat(6));
        c.scalar("wide_total_variance", k3.total_variance, Tol::stat(6));
        c.finish();
    }

    /// ML-35: the SVD path (n <= 4p) and the covariance path (n > 4p) give the same shares.
    /// Repeating every row 10 times keeps the shares and scales the covariance by 10 x 19 / 199.
    #[test]
    fn test_pca_ratio_agrees_across_paths() {
        let mut rng = StdRng::seed_from_u64(20261115);
        let z = Array2::from_shape_fn((20, 3), |_| rng.r#gen::<f64>() - 0.5);
        let load = Array2::from_shape_fn((3, 8), |_| rng.r#gen::<f64>() * 2.0 - 1.0);
        let x = z.dot(&load) + Array2::from_shape_fn((20, 8), |_| 0.3 * (rng.r#gen::<f64>() - 0.5));
        let rep = Array2::from_shape_fn((200, 8), |(i, j)| x[[i % 20, j]]);
        let svd = pca(x.view(), Some(2), false).unwrap();
        let cov = pca(rep.view(), Some(2), false).unwrap();
        for k in 0..2 {
            let (a, b) = (
                svd.explained_variance_ratio[k],
                cov.explained_variance_ratio[k],
            );
            assert!(
                (a - b).abs() <= 1e-12,
                "PC{}: {a} (SVD path) vs {b} (covariance path)",
                k + 1
            );
        }
        let scaled = svd.total_variance * 10.0 * 19.0 / 199.0;
        assert!(
            (scaled - cov.total_variance).abs() <= 1e-12 * cov.total_variance,
            "{scaled} vs {}",
            cov.total_variance
        );
    }

    /// The 21 x 21 eurodist matrix of a case, as rows.
    fn eurodist(case: &p2a_validation::Case) -> Vec<Vec<f64>> {
        let df = case_frame(case, "eurodist_matrix.csv");
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let m = matrix(&df, &names);
        m.rows().into_iter().map(|r| r.to_vec()).collect()
    }

    /// `|points|` in R's column-major order.
    fn points_abs(r: &CmdscaleResult) -> Vec<f64> {
        (0..r.k)
            .flat_map(|j| r.points.iter().map(move |p| p[j].abs()))
            .collect()
    }

    #[test]
    fn test_validate_cmdscale_eurodist() {
        // ML-36: before (power iteration) eig = 19538377, 11856555, -2251844, 1528844, ...; R
        // sorts algebraically. ML-98: GOF was [sum_pos/sum_abs, sum_k/sum_abs] = 0.8685, 0.7538.
        let (case, _) = crate::test_support::load_case("kmeans_lloyd_pca_mds");
        let r = cmdscale(&eurodist(&case), Some(2), Some(true), None).unwrap();
        Check::new(&case)
            .vector("mds_points_abs", &points_abs(&r), Tol::coef(21))
            // the null eigenvalue is roundoff (R -3.7e-09; |lambda| <~ 21 eps lambda_max ~ 1e-7):
            // abs 1e-6
            .vector("mds_eig", &r.eig, Tol::new(1e-9, 1e-6))
            .vector("mds_gof", &r.gof, Tol::coef(21))
            .finish();
    }

    /// ML-98: R's GOF pair, and add = TRUE (Cailliez's additive constant), on eurodist and on
    /// Manhattan distances (negative eigenvalues: the two GOF measures differ). Before, add was
    /// ignored.
    #[test]
    fn test_validate_cmdscale_gof_add() {
        let (case, _) = crate::test_support::load_case("cmdscale_gof_add");
        let x = matrix(&case_frame(&case, "mds_points8.csv"), &["a", "b", "c"]);
        let manh: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| (0..3).map(|c| (x[[i, c]] - x[[j, c]]).abs()).sum())
                    .collect()
            })
            .collect();
        let euro = eurodist(&case);
        let mut c = Check::new(&case);
        for (tag, d, k, add) in [
            ("euro_k2", &euro, 2, None),
            ("euro_k2_add", &euro, 2, Some(true)),
            ("manh_k2", &manh, 2, None),
            ("manh_k3_add", &manh, 3, Some(true)),
        ] {
            let n = d.len();
            let r = cmdscale(d, Some(k), Some(true), add).unwrap();
            // eurodist: null eigenvalues are roundoff of a 4e7 spectrum (abs 1e-6). With add,
            // the null eigenvalue moves with ac to first order: ac agrees with R's dgeev to about
            // 3e-13 relative (6.7e-10), which moves it by about 1e-6, so abs 1e-5 (2.4e-13 of
            // the largest eigenvalue)
            let eig_tol = match (n, add) {
                (21, None) => Tol::new(1e-9, 1e-6),
                (21, _) => Tol::new(1e-9, 1e-5),
                _ => Tol::new(1e-9, 1e-9),
            };
            c.vector(&format!("{tag}_gof"), &r.gof, Tol::coef(n))
                .scalar(&format!("{tag}_ac"), r.ac, Tol::coef(n))
                .vector(&format!("{tag}_points_abs"), &points_abs(&r), Tol::coef(n))
                .vector(&format!("{tag}_eig"), &r.eig, eig_tol);
        }
        c.finish();
    }

    #[test]
    fn test_validate_cmdscale_line_configuration() {
        // ML-36: (0, -1, 1) with k = 1 returned [[2e-24],[4e-24],[4e-24]] and eig 3e-16 before;
        // R points 0 -1 1, eig 2 0 0
        let (case, _) = crate::test_support::load_case("kmeans_lloyd_pca_mds");
        let x = array![[0.0], [-1.0], [1.0]];
        let r = cmdscale_from_data(x.view(), Some(1)).unwrap();
        Check::new(&case)
            .vector(
                "mds_line_points_abs",
                &r.points.iter().map(|p| p[0].abs()).collect::<Vec<_>>(),
                Tol::new(1e-12, 1e-12),
            )
            .vector("mds_line_eig", &r.eig, Tol::new(1e-12, 1e-12))
            .finish();
    }

    #[test]
    fn test_validate_cmdscale_is_scale_equivariant() {
        // ML-36 verification: data x 1e-5 collapsed to an all-zero configuration under the
        // absolute 1e-10 stop rule
        let (_, ds) = crate::test_support::load_case("kmeans_lloyd_pca_mds");
        let x = matrix(ds.df(), &["a", "b", "c"]);
        let s = 2f64.powi(-17);
        let a = cmdscale_from_data(x.view(), Some(2)).unwrap();
        let b = cmdscale_from_data((&x * s).view(), Some(2)).unwrap();
        for (ea, eb) in a.eig.iter().zip(&b.eig).take(3) {
            assert!((eb / (s * s) - ea).abs() <= 1e-9 * ea.abs(), "{ea} {eb}");
        }
        for (pa, pb) in a.points.iter().zip(&b.points) {
            for j in 0..2 {
                assert!((pb[j].abs() / s - pa[j].abs()).abs() <= 1e-9 * (1.0 + pa[j].abs()));
            }
        }
    }
}
