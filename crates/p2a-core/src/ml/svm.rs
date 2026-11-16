//! Support Vector Machine (Linear and Kernel SVM).
//!
//! [`linear_svm`] is the linear-kernel front of LIBSVM's solver
//! ([`fast_svm`](super::svm_fast::fast_svm)), R-pinned against e1071.
//! [`kernel_svm`] keeps a simplified SMO (library only) and is not R-pinned.
//!
//! Supports multiple kernel functions:
//! - Linear: K(x, y) = x · y
//! - RBF (Gaussian): K(x, y) = exp(-gamma * ||x - y||²)
//! - Polynomial: K(x, y) = (gamma * x · y + coef0)^degree
//! - Sigmoid: K(x, y) = tanh(gamma * x · y + coef0)

use serde::{Deserialize, Serialize};

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use super::svm_fast::{FastKernel, FastSvmConfig, fast_svm};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

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
    /// The convergence record of the solver
    pub convergence: Option<Convergence>,
    /// What the caller should know about this fit
    pub warnings: Vec<String>,
}

impl ResultNotes for SvmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
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
        indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

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

        // Count predictions per class label
        let (neg, pos) = self.class_labels.unwrap_or((-1, 1));
        let neg_count = self.predictions.iter().filter(|&&p| p == neg).count();
        let pos_count = self.predictions.iter().filter(|&&p| p == pos).count();
        writeln!(
            f,
            "  Class distribution: {neg_count} predicted {neg} (negative), {pos_count} predicted {pos} (positive)"
        )?;

        write!(f, "{}", notes_text(self))
    }
}

/// Kernel cache of [`linear_svm`] in MB (e1071's `cachesize` default).
const LINEAR_SVM_CACHE_MB: usize = 40;

/// Linear C-SVC, solved by LIBSVM's algorithm: [`fast_svm`] with a linear
/// kernel. R equivalent: `e1071::svm(x, factor(y), type = 'C-classification',
/// kernel = 'linear', scale = FALSE, cost = c, tolerance = tolerance)`.
///
/// # Arguments
/// * `data` - Input feature matrix (n_samples x n_features); values must be finite
/// * `target` - Two class labels, whole numbers in the `i32` range; the larger is
///   the positive class
/// * `c` - Regularization parameter, finite and > 0 (default: 1.0)
/// * `max_iterations` - A budget of passes, at least 1 (default: 1000): LIBSVM
///   runs at most `max_iterations * n` pair updates (saturating)
/// * `tolerance` - LIBSVM's stopping tolerance `eps` on the maximal violating
///   pair, finite and > 0 (default: 1e-3, e1071's `tolerance`)
/// * `feature_names` - Optional feature names
///
/// `weights` is `sum_k alpha_k y_k x_k` over the support vectors, `bias` is
/// LIBSVM's `-rho`, and a row is predicted as the larger label when its decision
/// value is `> 0`, as LIBSVM. `n_iterations` counts LIBSVM pair updates;
/// `convergence` and `warnings` say whether the optimality condition was met
/// within the budget.
///
/// # Cost
///
/// LIBSVM's second-order working-set selection with shrinking and a kernel
/// cache of at most 40 MB (`4 n^2` bytes when that is smaller). The installed
/// computation budget is checked before the solver starts and inside it.
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

    if !(c_param.is_finite() && c_param > 0.0) {
        return Err(format!(
            "c must be a finite number greater than 0, got {c_param}"
        ));
    }
    if max_iter == 0 {
        return Err("max_iterations must be at least 1".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(format!(
            "tolerance must be a finite number greater than 0, got {tol}"
        ));
    }
    check_finite_features(&data, feature_names.as_deref())?;
    let k = target.iter().filter(|v| !v.is_finite()).count();
    if k > 0 {
        return Err(format!(
            "SVM: the target has {k} missing or non-finite values; remove or impute them first"
        ));
    }

    // The two labels must be whole numbers (the result keeps them as i32)
    let (_, class_labels) = convert_to_binary(&target)?;
    // fast_svm checks the budget only every min(n, 1000) iterations
    resource::checkpoint().map_err(|e| e.to_string())?;

    let config = FastSvmConfig {
        c: c_param,
        kernel: FastKernel::Linear,
        tolerance: tol,
        max_iter: max_iter.saturating_mul(n_samples),
        cache_size_mb: LINEAR_SVM_CACHE_MB,
        shrinking: true,
        min_working_set: 0,
    };
    // fast_svm's messages name it; the callers (ml_svm, p2a ml svm) know it as SVM
    let fast = fast_svm(data, target, &config).map_err(|e| match e.strip_prefix("FastSVM:") {
        Some(rest) => format!("SVM:{rest}"),
        None => e,
    })?;

    // w = sum_k (alpha_k y_k) x_k over the support vectors
    let mut weights = vec![0.0; n_features];
    for (&i, &a) in fast
        .support_vector_indices
        .iter()
        .zip(&fast.dual_coefficients)
    {
        for (j, w) in weights.iter_mut().enumerate() {
            *w += a * data[[i, j]];
        }
    }
    // LIBSVM's sign: a decision value > 0 is the larger label (whole numbers, checked above)
    let predictions: Vec<i32> = fast.predictions.iter().map(|&p| p as i32).collect();

    Ok(SvmResult {
        predictions,
        weights,
        bias: fast.bias,
        n_support_vectors: fast.n_support_vectors,
        n_iterations: fast.n_iterations,
        converged: fast.converged,
        support_vector_indices: fast.support_vector_indices,
        class_labels: Some(class_labels),
        feature_names,
        convergence: fast.convergence,
        warnings: fast.warnings,
    })
}

/// Refuses a feature column with a missing or non-finite value, naming it.
fn check_finite_features(
    data: &ArrayView2<f64>,
    feature_names: Option<&[String]>,
) -> Result<(), String> {
    for (j, column) in data.axis_iter(Axis(1)).enumerate() {
        let k = column.iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            let name = feature_names
                .and_then(|names| names.get(j))
                .map(|name| format!("feature '{name}'"))
                .unwrap_or_else(|| format!("column {j} (0-based)"));
            return Err(format!(
                "SVM: {name} has {k} missing or non-finite values; remove or impute them first"
            ));
        }
    }
    Ok(())
}

/// The parameter checks of [`kernel_svm`], as [`linear_svm`]'s plus the kernel's.
fn check_kernel_config(config: &KernelSvmConfig) -> Result<(), String> {
    if !(config.c.is_finite() && config.c > 0.0) {
        return Err(format!(
            "c must be a finite number greater than 0, got {}",
            config.c
        ));
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(format!(
            "tolerance must be a finite number greater than 0, got {}",
            config.tolerance
        ));
    }
    if config.max_iter == 0 {
        return Err("max_iter must be at least 1".to_string());
    }
    if let Some(g) = config.gamma {
        if !(g.is_finite() && g > 0.0) {
            return Err(format!(
                "gamma must be a finite number greater than 0, got {g}"
            ));
        }
    }
    if config.kernel == SvmKernel::Polynomial && config.degree == 0 {
        return Err("degree must be at least 1 for the polynomial kernel".to_string());
    }
    if config.degree > i32::MAX as usize {
        return Err(format!(
            "degree must be at most {}, got {}",
            i32::MAX,
            config.degree
        ));
    }
    if !config.coef0.is_finite() {
        return Err(format!("coef0 must be finite, got {}", config.coef0));
    }
    Ok(())
}

/// Two classes in a response: the smaller value is -1, the larger +1.
pub(crate) struct BinaryClasses {
    pub(crate) y: Array1<f64>,
    pub(crate) negative: f64,
    pub(crate) positive: f64,
}

/// Exactly two finite values, ordered; the one mapping from labels to -1/+1 of the SVM solvers.
pub(crate) fn binary_classes(target: &ArrayView1<f64>) -> Result<BinaryClasses, String> {
    if target.iter().any(|v| !v.is_finite()) {
        return Err("SVM: the labels contain non-finite values".to_string());
    }
    let mut unique_values: Vec<f64> = target.iter().cloned().collect();
    unique_values.sort_by(f64::total_cmp);
    unique_values.dedup();

    if unique_values.len() != 2 {
        return Err(format!(
            "SVM requires exactly 2 classes, found {}",
            unique_values.len()
        ));
    }

    let (negative, positive) = (unique_values[0], unique_values[1]);
    let y = target.map(|&v| if v == negative { -1.0 } else { 1.0 });

    Ok(BinaryClasses {
        y,
        negative,
        positive,
    })
}

/// Convert target values to binary -1/+1; the two labels must be whole numbers
/// in the `i32` range (the result types keep them as `i32`).
fn convert_to_binary(target: &ArrayView1<f64>) -> Result<(Array1<f64>, (i32, i32)), String> {
    let b = binary_classes(target)?;
    let whole = |v: f64| v.fract() == 0.0 && v >= f64::from(i32::MIN) && v <= f64::from(i32::MAX);
    if !(whole(b.negative) && whole(b.positive)) {
        return Err(format!(
            "SVM: class labels must be whole numbers between {} and {} (found {} and {}); recode the target, for example as 0/1",
            i32::MIN,
            i32::MAX,
            b.negative,
            b.positive
        ));
    }
    Ok((b.y, (b.negative as i32, b.positive as i32)))
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
            // LIBSVM's sign, as linear_svm: > 0 is the larger label
            if f > 0.0 {
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
    /// The kernel coefficient used (`config.gamma`, else `1 / n_features`)
    #[serde(default)]
    pub gamma: f64,
    /// Number of features of the training data
    #[serde(default)]
    pub n_features: usize,
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
            writeln!(f, "Gamma: {}", self.gamma)?;
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
/// * `data` - Input feature matrix (n_samples x n_features); values must be finite
/// * `target` - Two class labels, whole numbers in the `i32` range
/// * `config` - SVM configuration
/// * `feature_names` - Optional feature names
///
/// # Errors
///
/// - `c` and `tolerance` not finite and positive, `max_iter = 0`, a `gamma`
///   that is not finite and positive, `degree = 0` for the polynomial kernel,
///   a non-finite `coef0`, a non-finite feature value, fractional labels.
/// - `"Resource limit: ..."` when the `n x n` kernel matrix exceeds the
///   allocation ceiling; `"Computation stopped: ..."` when the budget is
///   cancelled or expires (checked per row of the kernel matrix and per sweep).
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
    if n_features == 0 {
        return Err("SVM: need at least one feature".to_string());
    }
    check_kernel_config(config)?;
    check_finite_features(&data, feature_names.as_deref())?;

    // Convert target to -1/+1
    let (y, class_labels) = convert_to_binary(&target)?;

    // Set gamma if not specified
    let gamma = config.gamma.unwrap_or(1.0 / n_features as f64);

    // Initialize alphas and bias
    let mut alpha: Array1<f64> = Array1::zeros(n_samples);
    let mut b = 0.0;

    // Precompute kernel matrix for efficiency
    let to_string = |e: crate::errors::EconError| e.to_string();
    let mut kernel_matrix =
        resource::try_zeros2(n_samples, n_samples, "kernel SVM matrix").map_err(to_string)?;
    for i in 0..n_samples {
        resource::checkpoint().map_err(to_string)?;
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
        resource::checkpoint().map_err(to_string)?;
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

    // Training accuracy on the internal -1/+1 labels
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|&(&pred, &yi)| (pred == class_labels.1) == (yi > 0.0))
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
        gamma,
        n_features,
        support_vectors: Some(sv_data),
        support_vector_labels: Some(sv_labels),
    })
}

/// Predict class labels for new data using a trained kernel SVM, with the
/// model's `gamma`; data of another width than the training data are refused.
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

    if data.ncols() != result.n_features {
        return Err(format!(
            "Data has {} features, but the model was fit with {}",
            data.ncols(),
            result.n_features
        ));
    }
    let gamma = result.gamma;

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
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::Array2;
    use ndarray::array;

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn overlapping_classes(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = 4.0 * next() - 2.0;
            }
            y[i] = if x[[i, 0]] + 0.5 * x[[i, 1]] + 0.8 * (next() - 0.5) > 0.0 {
                1.0
            } else {
                0.0
            };
        }
        (x, y)
    }

    /// e1071's linear C-SVC on `svm_sim` `cls_lin` (labels 1 and 2), at e1071's tolerance 1e-10.
    fn check_linear_svm_libsvm(tag: &str, c: f64) {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("svm_libsvm");
        let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        let (x1, x2) = (col("x1"), col("x2"));
        let x = Array2::from_shape_fn((x1.len(), 2), |(i, j)| if j == 0 { x1[i] } else { x2[i] });
        let y = Array1::from_vec(col("cls_lin"));
        let r = linear_svm(
            x.view(),
            y.view(),
            Some(c),
            Some(1_000_000),
            Some(1e-10),
            None,
        )
        .unwrap();
        assert!(r.converged, "{r}");
        let decision: Vec<f64> = x
            .outer_iter()
            .map(|row| row.dot(&Array1::from_vec(r.weights.clone())) + r.bias)
            .collect();
        let pred: Vec<f64> = r.predictions.iter().map(|&p| f64::from(p)).collect();
        // the fast_svm tolerance (validation/ml/svm.md)
        let tol = Tol::iterative(4e-7);
        Check::new(&case)
            .vector(&format!("{tag}_w"), &r.weights, tol)
            .scalar(&format!("{tag}_bias"), r.bias, tol)
            .vector(&format!("{tag}_decision"), &decision, tol)
            .vector(&format!("{tag}_pred"), &pred, Tol::exact())
            .finish();
    }

    /// ML-40: the old SMO searched non-bound multipliers below a hard-coded 1 and stopped after
    /// one sweep without a change, so it stopped short of the optimum at C != 1.
    #[test]
    fn test_validate_linear_svm_libsvm_c01() {
        check_linear_svm_libsvm("linear_c01", 0.1);
    }

    #[test]
    fn test_validate_linear_svm_libsvm_c10() {
        check_linear_svm_libsvm("linear_c10", 10.0);
    }

    fn fast_linear(c: f64, max_iter: usize) -> FastSvmConfig {
        FastSvmConfig {
            c,
            kernel: FastKernel::Linear,
            tolerance: 1e-3,
            max_iter,
            cache_size_mb: 40,
            shrinking: true,
            min_working_set: 0,
        }
    }

    /// linear_svm is fast_svm with a linear kernel, a 40 MB cache and a budget of
    /// max_iterations x n LIBSVM iterations.
    #[test]
    fn test_linear_svm_equals_fast_svm() {
        let (x, y) = overlapping_classes(120, 3);
        for c in [0.1, 1.0, 10.0] {
            let r = linear_svm(x.view(), y.view(), Some(c), None, None, None).unwrap();
            let f = fast_svm(x.view(), y.view(), &fast_linear(c, 1000 * 120)).unwrap();
            let mut w = [0.0; 3];
            for (&i, &a) in f.support_vector_indices.iter().zip(&f.dual_coefficients) {
                for (j, wj) in w.iter_mut().enumerate() {
                    *wj += a * x[[i, j]];
                }
            }
            assert_eq!(r.weights, w.to_vec(), "C = {c}");
            assert_eq!(r.bias, f.bias, "C = {c}");
            assert_eq!(
                r.support_vector_indices, f.support_vector_indices,
                "C = {c}"
            );
            assert_eq!(r.n_iterations, f.n_iterations, "C = {c}");
            assert_eq!((r.converged, f.converged), (true, true), "C = {c}");
            let fp: Vec<i32> = f.predictions.iter().map(|&p| p as i32).collect();
            assert_eq!(r.predictions, fp, "C = {c}");
        }
    }

    /// max_iterations counts passes of n LIBSVM pair updates; a run that stops at the budget says
    /// so in its notes.
    #[test]
    fn test_linear_svm_iteration_budget() {
        let (x, y) = overlapping_classes(200, 7);
        let full = fast_svm(x.view(), y.view(), &fast_linear(100.0, 10_000_000)).unwrap();
        assert!(
            full.converged && full.n_iterations > 200,
            "{}",
            full.n_iterations
        );
        let r = linear_svm(x.view(), y.view(), Some(100.0), Some(1), None, None).unwrap();
        assert!(!r.converged);
        assert_eq!(r.n_iterations, 200);
        let notes = crate::traits::convergence::notes_text(&r);
        assert!(
            !r.warnings.is_empty() && notes.contains("Warning: "),
            "{notes}"
        );
        assert!(r.to_string().ends_with(&notes), "{r}");
    }

    #[test]
    fn test_linear_svm_rejects_bad_arguments() {
        let (x, y) = overlapping_classes(20, 4);
        let bad = |c: Option<f64>, it: Option<usize>, tol: Option<f64>| {
            linear_svm(x.view(), y.view(), c, it, tol, None).unwrap_err()
        };
        assert!(bad(Some(0.0), None, None).contains("c must be a finite number greater than 0"));
        assert!(
            bad(Some(f64::NAN), None, None).contains("c must be a finite number greater than 0")
        );
        assert!(bad(None, Some(0), None).contains("max_iterations must be at least 1"));
        for tol in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(
                bad(None, None, Some(tol))
                    .contains("tolerance must be a finite number greater than 0"),
                "{tol}"
            );
        }
        let mut xb = x.clone();
        xb[[1, 2]] = f64::NEG_INFINITY;
        assert!(
            linear_svm(xb.view(), y.view(), None, None, None, None)
                .unwrap_err()
                .contains("non-finite")
        );
        let mut yb = y.clone();
        yb[0] = f64::NAN;
        assert!(
            linear_svm(x.view(), yb.view(), None, None, None, None)
                .unwrap_err()
                .contains("non-finite")
        );
    }

    #[test]
    fn test_linear_svm_honours_cancellation() {
        let (x, y) = overlapping_classes(40, 5);
        let err = with_budget(cancelled_budget(), || {
            linear_svm(x.view(), y.view(), None, None, None, None)
        })
        .unwrap_err();
        assert!(err.contains("cancelled"), "{err}");
    }

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
            [3.0, 0.0], // 3+0-3 = 0 -> class 0: LIBSVM's sign, > 0 is the larger label (was 1)
        ];

        let preds = svm_predict(data.view(), &weights, bias, class_labels);
        assert_eq!(preds, vec![0, 1, 0]);
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

    #[test]
    fn test_svm_linearly_separable_perfect_properties() {
        // Well-separated data should achieve 100% accuracy.
        let x = array![
            // Class 0: bottom-left cluster
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            // Class 1: top-right cluster (well separated)
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 5.0],
            [5.0, 6.0],
            [5.5, 5.0],
            [5.0, 5.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result =
            linear_svm(x.view(), y.view(), Some(10.0), Some(500), Some(1e-4), None).unwrap();

        // Should achieve perfect classification on well-separated data
        let correct: usize = result
            .predictions
            .iter()
            .enumerate()
            .filter(|&(i, p)| {
                let expected = if i < 6 { 0 } else { 1 };
                *p == expected
            })
            .count();

        assert_eq!(
            correct, 12,
            "Should achieve 100% accuracy on well-separated data, got {}/12",
            correct
        );
    }

    #[test]
    fn test_svm_support_vectors_identified_properties() {
        // There should be at least some support vectors.
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result =
            linear_svm(x.view(), y.view(), Some(1.0), Some(300), Some(1e-4), None).unwrap();

        assert!(
            result.n_support_vectors > 0,
            "Should identify at least 1 support vector, got 0"
        );
        assert!(
            result.n_support_vectors <= 6,
            "Number of support vectors should be at most n_samples"
        );
        assert_eq!(
            result.support_vector_indices.len(),
            result.n_support_vectors,
            "support_vector_indices len should match n_support_vectors"
        );

        // All support vector indices should be valid
        for &idx in &result.support_vector_indices {
            assert!(idx < 6, "Support vector index {} should be < 6", idx);
        }
    }

    #[test]
    fn test_svm_decision_boundary_sign_properties() {
        // For data where class 1 is at higher x1 values, w[0] should be positive.
        let x = array![
            [0.0, 1.0],
            [1.0, 2.0],
            [0.5, 0.5],
            [10.0, 1.0],
            [11.0, 2.0],
            [10.5, 0.5],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result =
            linear_svm(x.view(), y.view(), Some(1.0), Some(300), Some(1e-4), None).unwrap();

        // Weight for feature 0 (the separating feature) should be positive
        // because class 1 has higher x1 values.
        assert!(
            result.weights[0] > 0.0,
            "w[0] should be positive since class 1 has higher x1, got {:.6}",
            result.weights[0]
        );
    }

    #[test]
    fn test_svm_kernel_rbf_nonlinear_properties() {
        // RBF kernel should handle nonlinear separation (XOR-like pattern).
        // Class 0: (low, low) and (high, high)
        // Class 1: (low, high) and (high, low)
        let x = array![
            // Class 0
            [0.0, 0.0],
            [0.5, 0.5],
            [5.0, 5.0],
            [5.5, 5.5],
            // Class 1
            [0.0, 5.0],
            [0.5, 5.5],
            [5.0, 0.0],
            [5.5, 0.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let config = KernelSvmConfig {
            kernel: SvmKernel::Rbf,
            c: 10.0,
            gamma: Some(0.1),
            max_iter: 1000,
            tolerance: 1e-3,
            ..Default::default()
        };

        let result = kernel_svm(x.view(), y.view(), &config, None).unwrap();

        // Should converge
        assert!(result.converged, "RBF SVM should converge");

        // Should have support vectors
        assert!(result.n_support_vectors > 0, "Should have support vectors");

        // Training accuracy should be reasonable (XOR is hard with few points)
        assert!(
            result.train_accuracy >= 0.5,
            "Train accuracy should be at least 50%, got {:.1}%",
            result.train_accuracy * 100.0
        );
    }

    /// `overlapping_classes` with its 0/1 labels recoded as `neg`/`pos`.
    fn coded(n: usize, seed: u64, neg: f64, pos: f64) -> (Array2<f64>, Array1<f64>) {
        let (x, y) = overlapping_classes(n, seed);
        (x, y.mapv(|v| if v == 0.0 { neg } else { pos }))
    }

    /// ML-102: two fractional labels were truncated to integers (0.2 and 0.7 both became 0).
    #[test]
    fn test_svm_refuses_fractional_labels() {
        let cfg = KernelSvmConfig::default();
        for (neg, pos) in [(0.2, 0.7), (0.5, 1.5), (-0.5, 0.5), (0.0, 3e9)] {
            let (x, y) = coded(40, 3, neg, pos);
            let e = linear_svm(x.view(), y.view(), None, None, None, None).unwrap_err();
            assert!(
                e.contains("class labels must be whole numbers"),
                "{neg}/{pos}: {e}"
            );
            let e = kernel_svm(x.view(), y.view(), &cfg, None).unwrap_err();
            assert!(
                e.contains("class labels must be whole numbers"),
                "{neg}/{pos}: {e}"
            );
        }
        for (neg, pos) in [(0.0, 1.0), (1.0, 2.0), (-1.0, 1.0), (-3.0, 7.0)] {
            let (x, y) = coded(40, 3, neg, pos);
            let r = linear_svm(x.view(), y.view(), None, None, None, None).unwrap();
            assert_eq!(r.class_labels, Some((neg as i32, pos as i32)));
            let k = kernel_svm(x.view(), y.view(), &cfg, None).unwrap();
            assert_eq!(k.class_labels, (neg as i32, pos as i32));
        }
    }

    /// ML-101: the class distribution counts the predicted labels (it counted predictions below
    /// zero as negative, which a 0/1 coding never has).
    #[test]
    fn test_svm_display_counts_predicted_labels() {
        let (x, y) = coded(100, 6, 0.0, 1.0);
        let mut r = linear_svm(x.view(), y.view(), None, None, None, None).unwrap();
        r.predictions = (0..100).map(|i| i32::from(i >= 45)).collect();
        let text = r.to_string();
        assert!(
            text.contains(
                "Class distribution: 45 predicted 0 (negative), 55 predicted 1 (positive)"
            ),
            "{text}"
        );
    }

    /// The training accuracy does not depend on how the two classes are coded.
    #[test]
    fn test_kernel_svm_accuracy_uses_internal_labels() {
        let cfg = KernelSvmConfig::default();
        let (x, a) = coded(60, 8, 0.0, 1.0);
        let (_, b) = coded(60, 8, 1.0, 2.0);
        let ra = kernel_svm(x.view(), a.view(), &cfg, None).unwrap();
        let rb = kernel_svm(x.view(), b.view(), &cfg, None).unwrap();
        assert!(ra.train_accuracy > 0.5, "{}", ra.train_accuracy);
        assert_eq!(ra.train_accuracy, rb.train_accuracy);
    }

    /// ML-140: kernel_svm's parameters and data went into SMO unchecked (C = 0 returned an
    /// untrained model labelled converged).
    #[test]
    fn test_kernel_svm_rejects_bad_arguments() {
        let (x, y) = overlapping_classes(30, 4);
        let base = KernelSvmConfig::default();
        let bad = |cfg: KernelSvmConfig, x: &Array2<f64>| {
            kernel_svm(x.view(), y.view(), &cfg, None).unwrap_err()
        };
        for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = bad(KernelSvmConfig { c, ..base.clone() }, &x);
            assert!(e.contains("c must be"), "{c}: {e}");
        }
        for tolerance in [0.0, f64::NAN] {
            let e = bad(
                KernelSvmConfig {
                    tolerance,
                    ..base.clone()
                },
                &x,
            );
            assert!(e.contains("tolerance must be"), "{tolerance}: {e}");
        }
        let e = bad(
            KernelSvmConfig {
                max_iter: 0,
                ..base.clone()
            },
            &x,
        );
        assert!(e.contains("max_iter must be"), "{e}");
        for g in [0.0, -1.0, f64::NAN] {
            let e = bad(
                KernelSvmConfig {
                    gamma: Some(g),
                    ..base.clone()
                },
                &x,
            );
            assert!(e.contains("gamma must be"), "{g}: {e}");
        }
        let poly = KernelSvmConfig {
            kernel: SvmKernel::Polynomial,
            degree: 0,
            ..base.clone()
        };
        assert!(bad(poly, &x).contains("degree must be"));
        let e = bad(
            KernelSvmConfig {
                coef0: f64::NAN,
                ..base.clone()
            },
            &x,
        );
        assert!(e.contains("coef0 must be"), "{e}");
        for v in [f64::NAN, f64::INFINITY] {
            let mut xb = x.clone();
            xb[[2, 1]] = v;
            assert!(bad(base.clone(), &xb).contains("non-finite"), "{v}");
        }
    }

    /// ML-140: the n x n kernel matrix is allocated fallibly, and the SMO sweeps check the budget.
    #[test]
    fn test_kernel_svm_honours_cancellation_and_ceiling() {
        let (x, y) = overlapping_classes(40, 5);
        let cfg = KernelSvmConfig::default();
        let e = with_budget(cancelled_budget(), || {
            kernel_svm(x.view(), y.view(), &cfg, None)
        })
        .unwrap_err();
        assert!(e.contains("cancelled"), "{e}");
        let e = crate::resource::with_alloc_ceiling(Some(40 * 40 * 8 - 8), || {
            kernel_svm(x.view(), y.view(), &cfg, None)
        })
        .unwrap_err();
        assert!(e.contains("Resource limit"), "{e}");
    }

    /// ML-141: prediction data of another width were silently truncated, and a defaulted gamma was
    /// recomputed from them.
    #[test]
    fn test_kernel_svm_predict_checks_width_and_keeps_gamma() {
        let x = array![
            [0.0, 0.1],
            [0.3, 0.2],
            [0.1, 0.4],
            [2.0, 2.1],
            [2.2, 1.9],
            [1.8, 2.3]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let r = kernel_svm(x.view(), y.view(), &KernelSvmConfig::default(), None).unwrap();
        assert_eq!((r.gamma, r.n_features), (0.5, 2));
        assert!(r.to_string().contains("Gamma: 0.5"), "{r}");
        let e = kernel_svm_predict(array![[0.0], [1.0], [2.0]].view(), &r).unwrap_err();
        assert!(
            e.contains("has 1 features, but the model was fit with 2"),
            "{e}"
        );
        assert_eq!(kernel_svm_predict(x.view(), &r).unwrap(), r.predictions);
        let back: KernelSvmResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!((back.gamma, back.n_features), (0.5, 2));
    }

    #[test]
    fn test_binary_classes_validates_and_orders() {
        let b = binary_classes(&ndarray::array![2.0, 1.0, 2.0, 1.0].view()).unwrap();
        assert_eq!((b.negative, b.positive), (1.0, 2.0));
        assert_eq!(b.y.to_vec(), vec![1.0, -1.0, 1.0, -1.0]);
        let err = |v: ndarray::Array1<f64>| binary_classes(&v.view()).err().unwrap();
        assert!(err(ndarray::array![1.0, 1.0]).contains("exactly 2 classes, found 1"));
        assert!(err(ndarray::array![1.0, 2.0, 3.0]).contains("found 3"));
        assert!(err(ndarray::array![1.0, f64::NAN]).contains("non-finite"));
        assert_eq!(
            convert_to_binary(&ndarray::array![0.0, 1.0].view())
                .unwrap()
                .1,
            (0, 1)
        );
    }
}
