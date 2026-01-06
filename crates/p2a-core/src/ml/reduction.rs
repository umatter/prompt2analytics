//! Dimensionality reduction: PCA.
//!
//! Pure Rust implementation using ndarray and eigendecomposition.

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
}
