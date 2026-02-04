//! Support Vector Machine (Linear and Kernel SVM).
//!
//! Pure Rust implementation using Sequential Minimal Optimization (SMO).
//!
//! Supports multiple kernel functions:
//! - Linear: K(x, y) = x · y
//! - RBF (Gaussian): K(x, y) = exp(-gamma * ||x - y||²)
//! - Polynomial: K(x, y) = (gamma * x · y + coef0)^degree
//! - Sigmoid: K(x, y) = tanh(gamma * x · y + coef0)

use serde::{Deserialize, Serialize};

use ndarray::{Array1, ArrayView1, ArrayView2};

/// SVM result.
#[derive(Debug, Clone)]
pub struct SvmResult {
    /// Predictions (-1 or 1 for binary, or class labels if provided)
    pub predictions: Vec<i32>,
    /// Feature weights (for linear kernel)
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Number of support vectors
    pub n_support_vectors: usize,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Indices of support vectors
    pub support_vector_indices: Vec<usize>,
    /// Class labels (if mapped from original labels)
    pub class_labels: Option<(i32, i32)>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
}

impl std::fmt::Display for SvmResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Linear SVM Results")?;
        writeln!(f, "==================")?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Support vectors: {}", self.n_support_vectors)?;
        writeln!(f, "Bias: {:.6}", self.bias)?;

        writeln!(f)?;
        writeln!(f, "Feature Weights:")?;

        // Sort by absolute weight
        let mut indexed: Vec<(usize, f64)> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, weight) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {}: {:.6}", name, weight)?;
        }

        if self.weights.len() > 10 {
            writeln!(f, "  ... ({} more features)", self.weights.len() - 10)?;
        }

        if let Some((neg, pos)) = self.class_labels {
            writeln!(f)?;
            writeln!(f, "Class labels: {} (negative), {} (positive)", neg, pos)?;
        }

        writeln!(f)?;
        writeln!(f, "Predictions: {} samples", self.predictions.len())?;

        // Count predictions per class
        let neg_count = self.predictions.iter().filter(|&&p| p < 0).count();
        let pos_count = self.predictions.len() - neg_count;
        writeln!(
            f,
            "  Class distribution: {} negative, {} positive",
            neg_count, pos_count
        )?;

        Ok(())
    }
}

/// Run Linear SVM using SMO (Sequential Minimal Optimization).
///
/// # Arguments
/// * `data` - Input feature matrix (n_samples x n_features)
/// * `target` - Target values (binary: will be converted to -1/+1)
/// * `c` - Regularization parameter (default: 1.0)
/// * `max_iterations` - Maximum iterations (default: 1000)
/// * `tolerance` - Convergence tolerance (default: 1e-3)
/// * `feature_names` - Optional feature names
pub fn linear_svm(
    data: ArrayView2<f64>,
    target: ArrayView1<f64>,
    c: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    feature_names: Option<Vec<String>>,
) -> Result<SvmResult, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for SVM".to_string());
    }
    if n_samples != target.len() {
        return Err("Data and target must have same number of samples".to_string());
    }

    let c_param = c.unwrap_or(1.0);
    let max_iter = max_iterations.unwrap_or(1000);
    let tol = tolerance.unwrap_or(1e-3);

    // Convert target to -1/+1
    let (y, class_labels) = convert_to_binary(&target)?;

    // Initialize alphas and bias
    let mut alpha: Array1<f64> = Array1::zeros(n_samples);
    let mut b = 0.0;

    // Precompute kernel (linear kernel: K(i,j) = x_i · x_j)
    // For efficiency, we'll compute dot products on the fly

    let mut n_iterations = 0;
    let mut converged = false;

    // SMO main loop
    for iter in 0..max_iter {
        let mut num_changed = 0;

        for i in 0..n_samples {
            // Compute error E_i = f(x_i) - y_i
            let f_i = compute_decision(&data, &alpha.view(), &y.view(), b, i);
            let e_i = f_i - y[i];

            // Check KKT conditions
            let r_i = e_i * y[i];

            if (r_i < -tol && alpha[i] < c_param) || (r_i > tol && alpha[i] > 0.0) {
                // Select j using heuristics (max |E_i - E_j|)
                let j = select_second_alpha(i, e_i, &data, &alpha.view(), &y.view(), b, n_samples);

                if j == i {
                    continue;
                }

                let f_j = compute_decision(&data, &alpha.view(), &y.view(), b, j);
                let e_j = f_j - y[j];

                // Save old alphas
                let alpha_i_old = alpha[i];
                let alpha_j_old = alpha[j];

                // Compute bounds L and H
                let (l, h) = if (y[i] - y[j]).abs() < 1e-10 {
                    // y_i == y_j
                    (
                        f64::max(0.0, alpha[i] + alpha[j] - c_param),
                        f64::min(c_param, alpha[i] + alpha[j]),
                    )
                } else {
                    // y_i != y_j
                    (
                        f64::max(0.0, alpha[j] - alpha[i]),
                        f64::min(c_param, c_param + alpha[j] - alpha[i]),
                    )
                };

                if (l - h).abs() < 1e-10 {
                    continue;
                }

                // Compute eta = 2*K(i,j) - K(i,i) - K(j,j)
                let k_ii = dot_product(&data.row(i), &data.row(i));
                let k_jj = dot_product(&data.row(j), &data.row(j));
                let k_ij = dot_product(&data.row(i), &data.row(j));
                let eta = 2.0 * k_ij - k_ii - k_jj;

                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j
                alpha[j] = alpha_j_old - y[j] * (e_i - e_j) / eta;

                // Clip alpha_j
                alpha[j] = alpha[j].max(l).min(h);

                if (alpha[j] - alpha_j_old).abs() < 1e-5 {
                    continue;
                }

                // Update alpha_i
                alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j]);

                // Update bias
                let b1 = b
                    - e_i
                    - y[i] * (alpha[i] - alpha_i_old) * k_ii
                    - y[j] * (alpha[j] - alpha_j_old) * k_ij;
                let b2 = b
                    - e_j
                    - y[i] * (alpha[i] - alpha_i_old) * k_ij
                    - y[j] * (alpha[j] - alpha_j_old) * k_jj;

                if alpha[i] > 0.0 && alpha[i] < c_param {
                    b = b1;
                } else if alpha[j] > 0.0 && alpha[j] < c_param {
                    b = b2;
                } else {
                    b = (b1 + b2) / 2.0;
                }

                num_changed += 1;
            }
        }

        n_iterations = iter + 1;

        if num_changed == 0 {
            converged = true;
            break;
        }
    }

    // Compute weights w = sum(alpha_i * y_i * x_i)
    let mut weights = vec![0.0; n_features];
    let mut support_vector_indices = Vec::new();

    for i in 0..n_samples {
        if alpha[i] > 1e-8 {
            support_vector_indices.push(i);
            for j in 0..n_features {
                weights[j] += alpha[i] * y[i] * data[[i, j]];
            }
        }
    }

    // Make predictions
    let predictions: Vec<i32> = (0..n_samples)
        .map(|i| {
            let f_i = compute_decision(&data, &alpha.view(), &y.view(), b, i);
            if f_i >= 0.0 {
                class_labels.1
            } else {
                class_labels.0
            }
        })
        .collect();

    Ok(SvmResult {
        predictions,
        weights,
        bias: b,
        n_support_vectors: support_vector_indices.len(),
        n_iterations,
        converged,
        support_vector_indices,
        class_labels: Some(class_labels),
        feature_names,
    })
}

/// Convert target values to binary -1/+1.
fn convert_to_binary(target: &ArrayView1<f64>) -> Result<(Array1<f64>, (i32, i32)), String> {
    let mut unique_values: Vec<f64> = target.iter().cloned().collect();
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_values.dedup();

    if unique_values.len() != 2 {
        return Err(format!(
            "SVM requires exactly 2 classes, found {}",
            unique_values.len()
        ));
    }

    let neg_class = unique_values[0] as i32;
    let pos_class = unique_values[1] as i32;

    let y = target.map(|&v| {
        if (v - unique_values[0]).abs() < 1e-10 {
            -1.0
        } else {
            1.0
        }
    });

    Ok((y, (neg_class, pos_class)))
}

/// Compute decision function f(x_i) for linear SVM.
fn compute_decision(
    data: &ArrayView2<f64>,
    alpha: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    b: f64,
    i: usize,
) -> f64 {
    let n_samples = data.nrows();
    let mut result = b;

    for j in 0..n_samples {
        if alpha[j] > 1e-8 {
            result += alpha[j] * y[j] * dot_product(&data.row(j), &data.row(i));
        }
    }

    result
}

/// Select second alpha using maximum |E_i - E_j| heuristic.
fn select_second_alpha(
    i: usize,
    e_i: f64,
    data: &ArrayView2<f64>,
    alpha: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    b: f64,
    n_samples: usize,
) -> usize {
    let mut max_delta = 0.0;
    let mut j_best = i;

    // First try non-bound alphas
    for j in 0..n_samples {
        if j != i && alpha[j] > 1e-8 && alpha[j] < 1.0 - 1e-8 {
            let f_j = compute_decision(data, alpha, y, b, j);
            let e_j = f_j - y[j];
            let delta = (e_i - e_j).abs();

            if delta > max_delta {
                max_delta = delta;
                j_best = j;
            }
        }
    }

    // If no good candidate found, try all
    if j_best == i {
        for j in 0..n_samples {
            if j != i {
                let f_j = compute_decision(data, alpha, y, b, j);
                let e_j = f_j - y[j];
                let delta = (e_i - e_j).abs();

                if delta > max_delta {
                    max_delta = delta;
                    j_best = j;
                }
            }
        }
    }

    // Fallback to random different index
    if j_best == i {
        j_best = (i + 1) % n_samples;
    }

    j_best
}

/// Compute dot product of two vectors.
fn dot_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Predict class labels for new data.
pub fn svm_predict(
    data: ArrayView2<f64>,
    weights: &[f64],
    bias: f64,
    class_labels: (i32, i32),
) -> Vec<i32> {
    data.outer_iter()
        .map(|row| {
            let f: f64 = row
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + bias;
            if f >= 0.0 {
                class_labels.1
            } else {
                class_labels.0
            }
        })
        .collect()
}

// =============================================================================
// Kernel SVM
// =============================================================================

/// Kernel type for SVM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SvmKernel {
    /// Linear kernel: K(x, y) = x · y
    #[default]
    Linear,
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||²)
    Rbf,
    /// Polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
    Polynomial,
    /// Sigmoid (tanh) kernel: K(x, y) = tanh(gamma * x · y + coef0)
    Sigmoid,
}

impl std::str::FromStr for SvmKernel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(SvmKernel::Linear),
            "rbf" | "gaussian" => Ok(SvmKernel::Rbf),
            "polynomial" | "poly" => Ok(SvmKernel::Polynomial),
            "sigmoid" | "tanh" => Ok(SvmKernel::Sigmoid),
            _ => Err(format!(
                "Unknown kernel: {}. Use 'linear', 'rbf', 'polynomial', or 'sigmoid'.",
                s
            )),
        }
    }
}

/// Configuration for Kernel SVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSvmConfig {
    /// Kernel type
    pub kernel: SvmKernel,
    /// Regularization parameter C (larger = less regularization)
    pub c: f64,
    /// Kernel coefficient for RBF, polynomial, sigmoid (default: 1/n_features)
    pub gamma: Option<f64>,
    /// Degree for polynomial kernel (default: 3)
    pub degree: usize,
    /// Independent term in polynomial and sigmoid kernels (default: 0)
    pub coef0: f64,
    /// Maximum iterations for SMO
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for KernelSvmConfig {
    fn default() -> Self {
        KernelSvmConfig {
            kernel: SvmKernel::Rbf,
            c: 1.0,
            gamma: None,
            degree: 3,
            coef0: 0.0,
            max_iter: 1000,
            tolerance: 1e-3,
        }
    }
}

/// Result from Kernel SVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSvmResult {
    /// Predictions on training data
    pub predictions: Vec<i32>,
    /// Alpha coefficients for support vectors
    pub alphas: Vec<f64>,
    /// Support vector indices
    pub support_vector_indices: Vec<usize>,
    /// Bias term
    pub bias: f64,
    /// Number of support vectors
    pub n_support_vectors: usize,
    /// Number of iterations
    pub n_iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Class labels (original values mapped to -1/+1)
    pub class_labels: (i32, i32),
    /// Configuration used
    pub config: KernelSvmConfig,
    /// Training accuracy
    pub train_accuracy: f64,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Support vectors (stored for prediction)
    #[serde(skip)]
    support_vectors: Option<ndarray::Array2<f64>>,
    /// Support vector labels (y values)
    #[serde(skip)]
    support_vector_labels: Option<Vec<f64>>,
}

impl std::fmt::Display for KernelSvmResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Kernel SVM Results")?;
        writeln!(f, "==================")?;
        writeln!(f, "Kernel: {:?}", self.config.kernel)?;
        writeln!(f, "C: {}", self.config.c)?;
        if self.config.kernel != SvmKernel::Linear {
            writeln!(f, "Gamma: {:?}", self.config.gamma)?;
        }
        if self.config.kernel == SvmKernel::Polynomial {
            writeln!(f, "Degree: {}", self.config.degree)?;
        }
        writeln!(f)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Support vectors: {}", self.n_support_vectors)?;
        writeln!(f, "Bias: {:.6}", self.bias)?;
        writeln!(f, "Training accuracy: {:.2}%", self.train_accuracy * 100.0)?;
        writeln!(f)?;
        writeln!(
            f,
            "Class labels: {} (negative), {} (positive)",
            self.class_labels.0, self.class_labels.1
        )?;

        // Count predictions per class
        let neg_count = self
            .predictions
            .iter()
            .filter(|&&p| p == self.class_labels.0)
            .count();
        let pos_count = self.predictions.len() - neg_count;
        writeln!(
            f,
            "Predictions: {} negative, {} positive",
            neg_count, pos_count
        )?;

        Ok(())
    }
}

/// Compute kernel value between two vectors.
fn compute_kernel(
    x1: &ArrayView1<f64>,
    x2: &ArrayView1<f64>,
    kernel: SvmKernel,
    gamma: f64,
    degree: usize,
    coef0: f64,
) -> f64 {
    match kernel {
        SvmKernel::Linear => dot_product(x1, x2),
        SvmKernel::Rbf => {
            let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (-gamma * sq_dist).exp()
        }
        SvmKernel::Polynomial => {
            let dot = dot_product(x1, x2);
            (gamma * dot + coef0).powi(degree as i32)
        }
        SvmKernel::Sigmoid => {
            let dot = dot_product(x1, x2);
            (gamma * dot + coef0).tanh()
        }
    }
}

/// Compute decision function for kernel SVM at sample i.
fn compute_decision_kernel(
    data: &ArrayView2<f64>,
    alpha: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    b: f64,
    i: usize,
    kernel: SvmKernel,
    gamma: f64,
    degree: usize,
    coef0: f64,
) -> f64 {
    let n_samples = data.nrows();
    let mut result = b;
    let x_i = data.row(i);

    for j in 0..n_samples {
        if alpha[j] > 1e-8 {
            let k_ij = compute_kernel(&data.row(j), &x_i, kernel, gamma, degree, coef0);
            result += alpha[j] * y[j] * k_ij;
        }
    }

    result
}

/// Run Kernel SVM using SMO.
///
/// # Arguments
/// * `data` - Input feature matrix (n_samples x n_features)
/// * `target` - Binary target values
/// * `config` - SVM configuration
/// * `feature_names` - Optional feature names
pub fn kernel_svm(
    data: ArrayView2<f64>,
    target: ArrayView1<f64>,
    config: &KernelSvmConfig,
    feature_names: Option<Vec<String>>,
) -> Result<KernelSvmResult, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples < 2 {
        return Err("Need at least 2 samples for SVM".to_string());
    }
    if n_samples != target.len() {
        return Err("Data and target must have same number of samples".to_string());
    }

    // Convert target to -1/+1
    let (y, class_labels) = convert_to_binary(&target)?;

    // Set gamma if not specified
    let gamma = config.gamma.unwrap_or(1.0 / n_features as f64);

    // Initialize alphas and bias
    let mut alpha: Array1<f64> = Array1::zeros(n_samples);
    let mut b = 0.0;

    // Precompute kernel matrix for efficiency
    let mut kernel_matrix = ndarray::Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in i..n_samples {
            let k_ij = compute_kernel(
                &data.row(i),
                &data.row(j),
                config.kernel,
                gamma,
                config.degree,
                config.coef0,
            );
            kernel_matrix[[i, j]] = k_ij;
            kernel_matrix[[j, i]] = k_ij;
        }
    }

    let mut n_iterations = 0;
    let mut converged = false;

    // SMO main loop
    for iter in 0..config.max_iter {
        let mut num_changed = 0;

        for i in 0..n_samples {
            // Compute error E_i = f(x_i) - y_i using kernel matrix
            let mut f_i = b;
            for j in 0..n_samples {
                if alpha[j] > 1e-8 {
                    f_i += alpha[j] * y[j] * kernel_matrix[[j, i]];
                }
            }
            let e_i = f_i - y[i];

            // Check KKT conditions
            let r_i = e_i * y[i];

            if (r_i < -config.tolerance && alpha[i] < config.c)
                || (r_i > config.tolerance && alpha[i] > 0.0)
            {
                // Select j using maximum |E_i - E_j| heuristic
                let mut max_delta = 0.0;
                let mut j_best = (i + 1) % n_samples;

                for j in 0..n_samples {
                    if j != i && alpha[j] > 1e-8 && alpha[j] < config.c - 1e-8 {
                        let mut f_j = b;
                        for k in 0..n_samples {
                            if alpha[k] > 1e-8 {
                                f_j += alpha[k] * y[k] * kernel_matrix[[k, j]];
                            }
                        }
                        let e_j = f_j - y[j];
                        let delta = (e_i - e_j).abs();
                        if delta > max_delta {
                            max_delta = delta;
                            j_best = j;
                        }
                    }
                }

                let j = j_best;

                // Compute E_j
                let mut f_j = b;
                for k in 0..n_samples {
                    if alpha[k] > 1e-8 {
                        f_j += alpha[k] * y[k] * kernel_matrix[[k, j]];
                    }
                }
                let e_j = f_j - y[j];

                // Save old alphas
                let alpha_i_old = alpha[i];
                let alpha_j_old = alpha[j];

                // Compute bounds L and H
                let (l, h) = if (y[i] - y[j]).abs() < 1e-10 {
                    (
                        f64::max(0.0, alpha[i] + alpha[j] - config.c),
                        f64::min(config.c, alpha[i] + alpha[j]),
                    )
                } else {
                    (
                        f64::max(0.0, alpha[j] - alpha[i]),
                        f64::min(config.c, config.c + alpha[j] - alpha[i]),
                    )
                };

                if (l - h).abs() < 1e-10 {
                    continue;
                }

                // Compute eta using kernel matrix
                let k_ii = kernel_matrix[[i, i]];
                let k_jj = kernel_matrix[[j, j]];
                let k_ij = kernel_matrix[[i, j]];
                let eta = 2.0 * k_ij - k_ii - k_jj;

                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j
                alpha[j] = alpha_j_old - y[j] * (e_i - e_j) / eta;
                alpha[j] = alpha[j].max(l).min(h);

                if (alpha[j] - alpha_j_old).abs() < 1e-5 {
                    continue;
                }

                // Update alpha_i
                alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j]);

                // Update bias
                let b1 = b
                    - e_i
                    - y[i] * (alpha[i] - alpha_i_old) * k_ii
                    - y[j] * (alpha[j] - alpha_j_old) * k_ij;
                let b2 = b
                    - e_j
                    - y[i] * (alpha[i] - alpha_i_old) * k_ij
                    - y[j] * (alpha[j] - alpha_j_old) * k_jj;

                if alpha[i] > 0.0 && alpha[i] < config.c {
                    b = b1;
                } else if alpha[j] > 0.0 && alpha[j] < config.c {
                    b = b2;
                } else {
                    b = (b1 + b2) / 2.0;
                }

                num_changed += 1;
            }
        }

        n_iterations = iter + 1;

        if num_changed == 0 {
            converged = true;
            break;
        }
    }

    // Find support vectors
    let mut support_vector_indices = Vec::new();
    let mut alphas = Vec::new();

    for i in 0..n_samples {
        if alpha[i] > 1e-8 {
            support_vector_indices.push(i);
            alphas.push(alpha[i]);
        }
    }

    // Store support vectors for prediction
    let mut sv_data = ndarray::Array2::zeros((support_vector_indices.len(), n_features));
    let mut sv_labels = Vec::with_capacity(support_vector_indices.len());

    for (idx, &sv_idx) in support_vector_indices.iter().enumerate() {
        for j in 0..n_features {
            sv_data[[idx, j]] = data[[sv_idx, j]];
        }
        sv_labels.push(y[sv_idx]);
    }

    // Make predictions
    let predictions: Vec<i32> = (0..n_samples)
        .map(|i| {
            let mut f_i = b;
            for j in 0..n_samples {
                if alpha[j] > 1e-8 {
                    f_i += alpha[j] * y[j] * kernel_matrix[[j, i]];
                }
            }
            if f_i >= 0.0 {
                class_labels.1
            } else {
                class_labels.0
            }
        })
        .collect();

    // Compute training accuracy
    let correct = predictions
        .iter()
        .zip(target.iter())
        .filter(|(pred, actual)| **pred as f64 == **actual)
        .count();
    let train_accuracy = correct as f64 / n_samples as f64;

    let n_sv = support_vector_indices.len();

    Ok(KernelSvmResult {
        predictions,
        alphas,
        support_vector_indices,
        bias: b,
        n_support_vectors: n_sv,
        n_iterations,
        converged,
        class_labels,
        config: config.clone(),
        train_accuracy,
        feature_names,
        support_vectors: Some(sv_data),
        support_vector_labels: Some(sv_labels),
    })
}

/// Predict class labels for new data using a trained kernel SVM.
pub fn kernel_svm_predict(
    data: ArrayView2<f64>,
    result: &KernelSvmResult,
) -> Result<Vec<i32>, String> {
    let sv_data = result
        .support_vectors
        .as_ref()
        .ok_or("Support vectors not stored - cannot predict")?;
    let sv_labels = result
        .support_vector_labels
        .as_ref()
        .ok_or("Support vector labels not stored - cannot predict")?;

    let gamma = result.config.gamma.unwrap_or(1.0 / data.ncols() as f64);

    let predictions: Vec<i32> = data
        .outer_iter()
        .map(|x| {
            let mut f = result.bias;
            for (idx, (alpha, &y_sv)) in result.alphas.iter().zip(sv_labels.iter()).enumerate() {
                let k = compute_kernel(
                    &sv_data.row(idx),
                    &x,
                    result.config.kernel,
                    gamma,
                    result.config.degree,
                    result.config.coef0,
                );
                f += alpha * y_sv * k;
            }
            if f >= 0.0 {
                result.class_labels.1
            } else {
                result.class_labels.0
            }
        })
        .collect();

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_svm_basic() {
        // Linearly separable data
        let x = array![
            // Class 0
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            // Class 1
            [3.0, 3.0],
            [3.5, 3.5],
            [4.0, 3.0],
            [3.0, 4.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let result =
            linear_svm(x.view(), y.view(), Some(1.0), Some(100), Some(1e-3), None).unwrap();

        assert!(result.n_support_vectors > 0);
        assert_eq!(result.weights.len(), 2);

        // Should classify training data correctly (or mostly correctly)
        let correct: usize = result
            .predictions
            .iter()
            .enumerate()
            .filter(|&(i, p)| {
                let expected = if i < 4 { 0 } else { 1 };
                *p == expected
            })
            .count();
        assert!(correct >= 6); // At least 75% correct
    }

    #[test]
    fn test_svm_predict() {
        let weights = vec![1.0, 1.0];
        let bias = -3.0;
        let class_labels = (0, 1);

        let data = array![
            [1.0, 1.0], // 1+1-3 = -1 -> class 0
            [2.0, 2.0], // 2+2-3 = 1 -> class 1
            [3.0, 0.0], // 3+0-3 = 0 -> class 1
        ];

        let preds = svm_predict(data.view(), &weights, bias, class_labels);
        assert_eq!(preds, vec![0, 1, 1]);
    }

    #[test]
    fn test_svm_requires_two_classes() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];
        let y = array![0.0, 0.0, 0.0]; // Only one class

        let result = linear_svm(x.view(), y.view(), None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_svm_feature_weights() {
        // Only first feature matters
        let x = array![
            [0.0, 5.0],
            [1.0, 3.0],
            [0.5, 7.0],
            [10.0, 2.0],
            [11.0, 8.0],
            [10.5, 1.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result = linear_svm(
            x.view(),
            y.view(),
            Some(1.0),
            Some(200),
            Some(1e-4),
            Some(vec!["important".to_string(), "noise".to_string()]),
        )
        .unwrap();

        // First weight should be larger in magnitude
        assert!(result.weights[0].abs() > result.weights[1].abs());
    }
}
