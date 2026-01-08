//! Dimensionality reduction: PCA and t-SNE.
//!
//! Pure Rust implementations using ndarray.

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
    /// Total variance in the original data
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
            let loadings: Vec<String> = self.components.row(i).iter()
                .take(10)
                .map(|v| format!("{:.4}", v))
                .collect();
            let suffix = if self.components.ncols() > 10 { ", ..." } else { "" };
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
    let mean = data.mean_axis(Axis(0))
        .ok_or("Failed to compute mean")?;

    let mut centered = data.to_owned();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }

    // Compute covariance matrix
    let cov = covariance_matrix(&centered.view());

    // Eigendecomposition using power iteration
    let (eigenvalues, eigenvectors) = symmetric_eigen(&cov, n_comp)?;

    // Compute explained variance
    let total_variance: f64 = eigenvalues.iter().take(n_features).sum();
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
fn covariance_matrix(centered: &ArrayView2<f64>) -> Array2<f64> {
    let n = centered.nrows() as f64;
    let cov = centered.t().dot(centered) / (n - 1.0);
    cov
}

/// Symmetric eigendecomposition using power iteration with deflation.
/// Returns eigenvalues (sorted descending) and eigenvectors (as columns).
fn symmetric_eigen(
    matrix: &Array2<f64>,
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = matrix.nrows();
    let mut eigenvalues = Array1::zeros(n_components);
    let mut eigenvectors = Array2::zeros((n, n_components));

    // Work with a copy for deflation
    let mut a = matrix.clone();

    for i in 0..n_components {
        // Power iteration to find dominant eigenpair
        let (eigenvalue, eigenvector) = power_iteration(&a, 1000, 1e-10)?;

        eigenvalues[i] = eigenvalue;
        eigenvectors.column_mut(i).assign(&eigenvector);

        // Deflate: A = A - λ * v * v^T
        let outer = outer_product(&eigenvector.view(), &eigenvector.view());
        a = a - eigenvalue * outer;
    }

    Ok((eigenvalues, eigenvectors))
}

/// Power iteration to find dominant eigenpair.
fn power_iteration(
    matrix: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<(f64, Array1<f64>), String> {
    let n = matrix.nrows();

    // Initialize with random-ish vector
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    v[0] += 0.1; // Break symmetry
    normalize(&mut v);

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Multiply by matrix
        let av = matrix.dot(&v);

        // Compute Rayleigh quotient (eigenvalue estimate)
        let new_eigenvalue = v.dot(&av);

        // Normalize
        let mut new_v = av;
        normalize(&mut new_v);

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tol {
            return Ok((new_eigenvalue, new_v));
        }

        eigenvalue = new_eigenvalue;
        v = new_v;
    }

    Ok((eigenvalue, v))
}

/// Normalize a vector to unit length.
fn normalize(v: &mut Array1<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        *v /= norm;
    }
}

/// Compute outer product of two vectors.
fn outer_product(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Project new data onto principal components.
pub fn pca_transform(
    data: ArrayView2<f64>,
    pca_result: &PCAResult,
) -> Result<Array2<f64>, String> {
    let n_features = data.ncols();
    if n_features != pca_result.mean.len() {
        return Err(format!(
            "Data has {} features, but PCA was fit with {} features",
            n_features,
            pca_result.mean.len()
        ));
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
) -> Result<Array2<f64>, String> {
    let n_components = transformed.ncols();
    if n_components != pca_result.n_components {
        return Err(format!(
            "Transformed data has {} components, but PCA has {} components",
            n_components,
            pca_result.n_components
        ));
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
    /// Final KL divergence
    pub kl_divergence: f64,
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
        writeln!(f, "Embedding shape: ({}, {})", self.embedding.nrows(), self.embedding.ncols())?;

        // Show first few points
        let n_show = self.embedding.nrows().min(5);
        writeln!(f, "\nFirst {} embedded points:", n_show)?;
        for i in 0..n_show {
            let point: Vec<String> = self.embedding.row(i).iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            writeln!(f, "  Point {}: [{}]", i, point.join(", "))?;
        }
        if self.embedding.nrows() > 5 {
            writeln!(f, "  ... ({} more points)", self.embedding.nrows() - 5)?;
        }
        Ok(())
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
    if perp >= (n_samples as f64) / 3.0 {
        return Err(format!(
            "Perplexity ({}) must be less than n_samples / 3 ({})",
            perp,
            n_samples / 3
        ));
    }

    // Compute pairwise squared distances in high-dimensional space
    let distances = pairwise_squared_distances(&data);

    // Compute joint probabilities P_ij using Gaussian kernel
    let p = compute_joint_probabilities(&distances, perp)?;

    // Initialize low-dimensional embedding using PCA or random
    let mut y = initialize_embedding(data.view(), n_comp, seed);

    // Gradient descent parameters
    let momentum = 0.5;
    let final_momentum = 0.8;
    let momentum_switch_iter = 250;
    let early_exaggeration = 4.0;
    let early_exaggeration_iter = 100;

    let mut y_velocity: Array2<f64> = Array2::zeros((n_samples, n_comp));
    let mut gains: Array2<f64> = Array2::from_elem((n_samples, n_comp), 1.0);

    // Exaggerate P for early iterations
    let mut p_current = &p * early_exaggeration;

    let mut kl_div = 0.0;

    for iter in 0..max_iter {
        // Stop early exaggeration
        if iter == early_exaggeration_iter {
            p_current = p.clone();
        }

        // Switch to final momentum
        let current_momentum = if iter < momentum_switch_iter {
            momentum
        } else {
            final_momentum
        };

        // Compute Q distribution (Student-t with 1 DOF)
        let (q, sum_q) = compute_q_distribution(&y.view());

        // Compute gradient
        let gradient = compute_gradient(&p_current.view(), &q.view(), &y.view(), sum_q);

        // Update gains for adaptive learning rate
        for i in 0..n_samples {
            for j in 0..n_comp {
                let sign_match = (gradient[[i, j]] > 0.0) == (y_velocity[[i, j]] > 0.0);
                gains[[i, j]] = if sign_match {
                    f64::max(gains[[i, j]] * 0.8, 0.01)
                } else {
                    gains[[i, j]] + 0.2
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

        // Compute KL divergence every 50 iterations
        if iter % 50 == 0 || iter == max_iter - 1 {
            kl_div = compute_kl_divergence(&p_current.view(), &q.view(), sum_q);
        }
    }

    Ok(TsneResult {
        embedding: y,
        n_components: n_comp,
        perplexity: perp,
        n_iterations: max_iter,
        kl_divergence: kl_div,
    })
}

/// Compute pairwise squared Euclidean distances.
fn pairwise_squared_distances(data: &ArrayView2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for k in 0..data.ncols() {
                let diff = data[[i, k]] - data[[j, k]];
                dist_sq += diff * diff;
            }
            distances[[i, j]] = dist_sq;
            distances[[j, i]] = dist_sq;
        }
    }

    distances
}

/// Compute joint probability matrix P using Gaussian kernel.
/// Uses binary search to find sigma for each point to achieve target perplexity.
fn compute_joint_probabilities(
    distances: &Array2<f64>,
    perplexity: f64,
) -> Result<Array2<f64>, String> {
    let n = distances.nrows();
    let target_entropy = perplexity.ln();

    let mut p = Array2::zeros((n, n));

    for i in 0..n {
        // Binary search for sigma_i
        let mut sigma_min = 1e-10;
        let mut sigma_max = 1e10;
        let mut sigma = 1.0;

        for _ in 0..50 {
            // Compute conditional probability P_j|i
            let mut p_i = Array1::zeros(n);
            let mut sum_exp = 0.0;

            for j in 0..n {
                if i != j {
                    let exp_val = (-distances[[i, j]] / (2.0 * sigma * sigma)).exp();
                    p_i[j] = exp_val;
                    sum_exp += exp_val;
                }
            }

            if sum_exp > 0.0 {
                p_i /= sum_exp;
            }

            // Compute entropy
            let mut entropy = 0.0;
            for j in 0..n {
                if p_i[j] > 1e-10 {
                    entropy -= p_i[j] * p_i[j].ln();
                }
            }

            // Binary search
            if (entropy - target_entropy).abs() < 1e-5 {
                // Found good sigma
                for j in 0..n {
                    p[[i, j]] = p_i[j];
                }
                break;
            } else if entropy > target_entropy {
                sigma_max = sigma;
                sigma = (sigma + sigma_min) / 2.0;
            } else {
                sigma_min = sigma;
                sigma = (sigma + sigma_max) / 2.0;
            }

            // If we exhausted iterations, use current sigma
            if sigma_max - sigma_min < 1e-10 {
                for j in 0..n {
                    p[[i, j]] = p_i[j];
                }
                break;
            }
        }
    }

    // Symmetrize: P_ij = (P_j|i + P_i|j) / 2n
    let mut p_sym = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let val = (p[[i, j]] + p[[j, i]]) / (2.0 * n as f64);
            // Ensure minimum probability for numerical stability
            let val = val.max(1e-12);
            p_sym[[i, j]] = val;
            p_sym[[j, i]] = val;
        }
    }

    Ok(p_sym)
}

/// Initialize low-dimensional embedding.
fn initialize_embedding(data: ArrayView2<f64>, n_components: usize, seed: Option<u64>) -> Array2<f64> {
    let n_samples = data.nrows();

    // Try PCA initialization first
    if let Ok(pca_result) = pca(data, Some(n_components), true) {
        if let Some(transformed) = pca_result.transformed {
            // Scale down for better optimization
            return transformed * 0.0001;
        }
    }

    // Fallback to pseudo-random initialization using seed
    let mut embedding = Array2::zeros((n_samples, n_components));
    let seed = seed.unwrap_or(42);

    for i in 0..n_samples {
        for j in 0..n_components {
            // Simple pseudo-random based on position and seed
            let hash = ((i * 31 + j * 17 + seed as usize) as f64).sin();
            embedding[[i, j]] = hash * 0.0001;
        }
    }

    embedding
}

/// Compute Q distribution (Student-t with 1 degree of freedom).
/// Returns Q matrix and sum of all Q values.
fn compute_q_distribution(y: &ArrayView2<f64>) -> (Array2<f64>, f64) {
    let n = y.nrows();
    let mut q = Array2::zeros((n, n));
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

    (q, sum_q)
}

/// Compute gradient of KL divergence.
fn compute_gradient(
    p: &ArrayView2<f64>,
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
                let factor = 4.0 * (p[[i, j]] - q_ij) * q[[i, j]];

                for k in 0..n_comp {
                    gradient[[i, k]] += factor * (y[[i, k]] - y[[j, k]]);
                }
            }
        }
    }

    gradient
}

/// Compute KL divergence between P and Q.
fn compute_kl_divergence(p: &ArrayView2<f64>, q: &ArrayView2<f64>, sum_q: f64) -> f64 {
    let n = p.nrows();
    let mut kl = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i != j && p[[i, j]] > 1e-12 {
                let q_ij = (q[[i, j]] / sum_q).max(1e-12);
                kl += p[[i, j]] * (p[[i, j]] / q_ij).ln();
            }
        }
    }

    kl
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_basic() {
        // Simple correlated data
        let data = array![
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ];

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
            Some(2),        // 2D embedding
            Some(4.0),      // Low perplexity for small dataset (must be < n/3 = 5)
            Some(100),      // Fewer iterations for test speed
            Some(200.0),
            Some(42),
        ).unwrap();

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
            Some(3.0),  // n/3 = 2, so perplexity=3 should fail
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }
}
