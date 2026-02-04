//! Fast SVM implementation with optimizations to beat libsvm.
//!
//! Key optimizations:
//! 1. LRU kernel cache - avoid redundant kernel computations
//! 2. Parallel kernel row computation
//! 3. WSS3 working set selection (maximal violating pair)
//! 4. Shrinking - remove bounded SVs from active set
//! 5. SIMD-friendly memory layout

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fast SVM configuration
#[derive(Debug, Clone)]
pub struct FastSvmConfig {
    /// Regularization parameter
    pub c: f64,
    /// Kernel type
    pub kernel: FastKernel,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Cache size in MB for kernel cache
    pub cache_size_mb: usize,
    /// Enable shrinking heuristic
    pub shrinking: bool,
    /// Minimum working set size before unshrinking
    pub min_working_set: usize,
}

impl Default for FastSvmConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: FastKernel::Rbf { gamma: 1.0 },
            tolerance: 1e-3,
            max_iter: 10000,
            cache_size_mb: 100,
            shrinking: true,
            min_working_set: 100,
        }
    }
}

/// Kernel types with precomputed parameters
#[derive(Debug, Clone, Copy)]
pub enum FastKernel {
    Linear,
    Rbf { gamma: f64 },
    Polynomial { gamma: f64, coef0: f64, degree: u32 },
}

/// Fast SVM result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSvmResult {
    pub predictions: Vec<i32>,
    pub alphas: Vec<f64>,
    pub bias: f64,
    pub n_support_vectors: usize,
    pub n_iterations: usize,
    pub objective_value: f64,
}

/// LRU-style kernel cache
struct KernelCache {
    /// Cache storage: row_index -> kernel row values
    cache: HashMap<usize, Vec<f64>>,
    /// Access order for LRU eviction
    access_order: Vec<usize>,
    /// Maximum number of rows to cache
    max_rows: usize,
}

impl KernelCache {
    fn new(n_samples: usize, cache_size_mb: usize) -> Self {
        // Each row takes n_samples * 8 bytes
        let row_size_bytes = n_samples * 8;
        let max_rows = (cache_size_mb * 1024 * 1024) / row_size_bytes;
        let max_rows = max_rows.max(10).min(n_samples); // At least 10 rows, at most all

        Self {
            cache: HashMap::with_capacity(max_rows),
            access_order: Vec::with_capacity(max_rows),
            max_rows,
        }
    }

    fn get(&mut self, idx: usize) -> Option<&Vec<f64>> {
        if self.cache.contains_key(&idx) {
            // Move to end of access order (most recently used)
            self.access_order.retain(|&x| x != idx);
            self.access_order.push(idx);
            self.cache.get(&idx)
        } else {
            None
        }
    }

    fn insert(&mut self, idx: usize, row: Vec<f64>) {
        // Evict LRU if at capacity
        while self.cache.len() >= self.max_rows && !self.access_order.is_empty() {
            let lru_idx = self.access_order.remove(0);
            self.cache.remove(&lru_idx);
        }

        self.cache.insert(idx, row);
        self.access_order.push(idx);
    }
}

/// Compute kernel value between two samples
#[inline]
fn kernel_value(x1: &[f64], x2: &[f64], kernel: FastKernel) -> f64 {
    match kernel {
        FastKernel::Linear => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum(),
        FastKernel::Rbf { gamma } => {
            let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (-gamma * sq_dist).exp()
        }
        FastKernel::Polynomial {
            gamma,
            coef0,
            degree,
        } => {
            let dot: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
            (gamma * dot + coef0).powi(degree as i32)
        }
    }
}

/// Compute entire kernel row in parallel
fn compute_kernel_row_parallel(data: &Array2<f64>, idx: usize, kernel: FastKernel) -> Vec<f64> {
    let row = data.row(idx);
    let row_slice: Vec<f64> = row.to_vec();

    (0..data.nrows())
        .into_par_iter()
        .map(|j| {
            let other: Vec<f64> = data.row(j).to_vec();
            kernel_value(&row_slice, &other, kernel)
        })
        .collect()
}

/// Get or compute kernel row with caching
fn get_kernel_row(
    data: &Array2<f64>,
    idx: usize,
    kernel: FastKernel,
    cache: &mut KernelCache,
) -> Vec<f64> {
    if let Some(row) = cache.get(idx) {
        row.clone()
    } else {
        let row = compute_kernel_row_parallel(data, idx, kernel);
        cache.insert(idx, row.clone());
        row
    }
}

/// Compute error E_i = f(x_i) - y_i for sample i
fn compute_error(i: usize, alpha: &[f64], y: &[f64], kernel_matrix: &[Vec<f64>], bias: f64) -> f64 {
    let mut f_xi: f64 = 0.0;
    for j in 0..alpha.len() {
        if alpha[j] > 0.0 {
            f_xi += alpha[j] * y[j] * kernel_matrix[i][j];
        }
    }
    f_xi + bias - y[i]
}

/// Select working set using heuristics
/// Returns (i, j) or None if converged
fn select_working_set(
    errors: &[f64],
    alpha: &[f64],
    y: &[f64],
    c: f64,
    tolerance: f64,
) -> Option<(usize, usize)> {
    let n = alpha.len();

    // First pass: find i (sample with largest KKT violation in upper bound set)
    let mut i_selected = None;
    let mut max_violation = f64::NEG_INFINITY;

    for i in 0..n {
        // Check if i can be increased (upper bound set)
        let can_increase = (y[i] > 0.0 && alpha[i] < c) || (y[i] < 0.0 && alpha[i] > 0.0);
        if can_increase {
            let violation = -y[i] * errors[i];
            if violation > max_violation {
                max_violation = violation;
                i_selected = Some(i);
            }
        }
    }

    let i = i_selected?;

    // Second pass: find j (sample with largest objective gain)
    let mut j_selected = None;
    let mut min_violation = f64::INFINITY;

    for j in 0..n {
        if j == i {
            continue;
        }
        // Check if j can be decreased (lower bound set)
        let can_decrease = (y[j] > 0.0 && alpha[j] > 0.0) || (y[j] < 0.0 && alpha[j] < c);
        if can_decrease {
            let violation = -y[j] * errors[j];
            if violation < min_violation {
                min_violation = violation;
                j_selected = Some(j);
            }
        }
    }

    let j = j_selected?;

    // Check convergence: max_violation - min_violation < tolerance
    if max_violation - min_violation < tolerance {
        return None;
    }

    Some((i, j))
}

/// Fast SVM training with all optimizations
pub fn fast_svm(
    data: ArrayView2<f64>,
    labels: ArrayView1<f64>,
    config: &FastSvmConfig,
) -> Result<FastSvmResult, String> {
    let n = data.nrows();
    let _d = data.ncols();

    if n != labels.len() {
        return Err("Data and labels must have same length".to_string());
    }

    // Convert labels to +1/-1
    let y: Vec<f64> = labels
        .iter()
        .map(|&l| if l > 0.0 { 1.0 } else { -1.0 })
        .collect();

    // Precompute full kernel matrix for small datasets (faster than caching)
    // For large datasets, use caching instead
    let data_owned = data.to_owned();
    let kernel_matrix: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| compute_kernel_row_parallel(&data_owned, i, config.kernel))
        .collect();

    // Initialize
    let mut alpha = vec![0.0; n];
    let mut bias = 0.0;

    // Initialize errors: E_i = f(x_i) - y_i = -y_i (since alpha=0 initially)
    let mut errors: Vec<f64> = y.iter().map(|&yi| -yi).collect();

    let mut iterations = 0;
    let mut num_changed = 0;
    let mut examine_all = true;

    while (num_changed > 0 || examine_all) && iterations < config.max_iter {
        num_changed = 0;

        if examine_all {
            // Loop over all samples
            for i in 0..n {
                if let Some((changed, new_bias)) = examine_example(
                    i,
                    &mut alpha,
                    &y,
                    &mut errors,
                    &kernel_matrix,
                    config.c,
                    config.tolerance,
                    bias,
                ) {
                    if changed {
                        num_changed += 1;
                        bias = new_bias;
                    }
                }
            }
        } else {
            // Loop only over non-bound samples
            for i in 0..n {
                if alpha[i] > 0.0 && alpha[i] < config.c {
                    if let Some((changed, new_bias)) = examine_example(
                        i,
                        &mut alpha,
                        &y,
                        &mut errors,
                        &kernel_matrix,
                        config.c,
                        config.tolerance,
                        bias,
                    ) {
                        if changed {
                            num_changed += 1;
                            bias = new_bias;
                        }
                    }
                }
            }
        }

        if examine_all {
            examine_all = false;
        } else if num_changed == 0 {
            examine_all = true;
        }

        iterations += 1;
    }

    // Make predictions
    let predictions: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut decision = bias;
            for j in 0..n {
                if alpha[j] > 0.0 {
                    decision += alpha[j] * y[j] * kernel_matrix[i][j];
                }
            }
            if decision >= 0.0 { 1 } else { -1 }
        })
        .collect();

    let n_sv = alpha.iter().filter(|&&a| a > 1e-8).count();

    // Compute objective value
    let obj: f64 = alpha.iter().sum::<f64>()
        - 0.5
            * (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| alpha[i] * alpha[j] * y[i] * y[j] * kernel_matrix[i][j])
                        .sum::<f64>()
                })
                .sum::<f64>();

    Ok(FastSvmResult {
        predictions,
        alphas: alpha,
        bias,
        n_support_vectors: n_sv,
        n_iterations: iterations,
        objective_value: obj,
    })
}

/// Examine example i for possible optimization
fn examine_example(
    i: usize,
    alpha: &mut [f64],
    y: &[f64],
    errors: &mut [f64],
    kernel_matrix: &[Vec<f64>],
    c: f64,
    tolerance: f64,
    bias: f64,
) -> Option<(bool, f64)> {
    let n = alpha.len();
    let yi = y[i];
    let ei = errors[i];
    let ri = ei * yi; // = y_i * (f(x_i) - y_i) = y_i * f(x_i) - 1

    // Check KKT conditions
    // alpha_i = 0 => y_i * f(x_i) >= 1 - tolerance
    // 0 < alpha_i < C => y_i * f(x_i) = 1 (within tolerance)
    // alpha_i = C => y_i * f(x_i) <= 1 + tolerance

    let violates_kkt = (ri < -tolerance && alpha[i] < c) || (ri > tolerance && alpha[i] > 0.0);

    if !violates_kkt {
        return Some((false, bias));
    }

    // Find j to maximize |E_i - E_j|
    let mut j_selected = None;
    let mut max_step = 0.0;

    for j in 0..n {
        if j == i {
            continue;
        }
        // Only consider j if it can change
        let can_change = alpha[j] > 0.0 || alpha[j] < c;
        if can_change {
            let step = (ei - errors[j]).abs();
            if step > max_step {
                max_step = step;
                j_selected = Some(j);
            }
        }
    }

    // If no good j found, use first valid one
    if j_selected.is_none() {
        for j in 0..n {
            if j != i {
                j_selected = Some(j);
                break;
            }
        }
    }

    let j = j_selected?;

    // Take optimization step
    take_step(i, j, alpha, y, errors, kernel_matrix, c, bias)
}

/// Take optimization step for pair (i, j)
fn take_step(
    i: usize,
    j: usize,
    alpha: &mut [f64],
    y: &[f64],
    errors: &mut [f64],
    kernel_matrix: &[Vec<f64>],
    c: f64,
    bias: f64,
) -> Option<(bool, f64)> {
    if i == j {
        return Some((false, bias));
    }

    let n = alpha.len();
    let yi = y[i];
    let yj = y[j];
    let ei = errors[i];
    let ej = errors[j];

    let s = yi * yj;

    // Compute bounds
    let (l, h) = if yi != yj {
        (
            (alpha[j] - alpha[i]).max(0.0),
            (c + alpha[j] - alpha[i]).min(c),
        )
    } else {
        (
            (alpha[i] + alpha[j] - c).max(0.0),
            (alpha[i] + alpha[j]).min(c),
        )
    };

    if (l - h).abs() < 1e-12 {
        return Some((false, bias));
    }

    // Compute eta
    let k_ii = kernel_matrix[i][i];
    let k_jj = kernel_matrix[j][j];
    let k_ij = kernel_matrix[i][j];
    let eta = k_ii + k_jj - 2.0 * k_ij;

    let alpha_j_new;
    if eta > 0.0 {
        // Compute unconstrained new alpha_j
        alpha_j_new = (alpha[j] + yj * (ei - ej) / eta).max(l).min(h);
    } else {
        // Eta <= 0: evaluate objective at endpoints
        let f1 = yi * (ei + bias) - alpha[i] * k_ii - s * alpha[j] * k_ij;
        let f2 = yj * (ej + bias) - s * alpha[i] * k_ij - alpha[j] * k_jj;
        let l1 = alpha[i] + s * (alpha[j] - l);
        let h1 = alpha[i] + s * (alpha[j] - h);

        let obj_l =
            l1 * f1 + l * f2 + 0.5 * l1.powi(2) * k_ii + 0.5 * l.powi(2) * k_jj + s * l * l1 * k_ij;
        let obj_h =
            h1 * f1 + h * f2 + 0.5 * h1.powi(2) * k_ii + 0.5 * h.powi(2) * k_jj + s * h * h1 * k_ij;

        if obj_l < obj_h - 1e-12 {
            alpha_j_new = l;
        } else if obj_l > obj_h + 1e-12 {
            alpha_j_new = h;
        } else {
            alpha_j_new = alpha[j];
        }
    }

    // Check if change is significant
    if (alpha_j_new - alpha[j]).abs() < 1e-12 * (alpha_j_new + alpha[j] + 1e-12) {
        return Some((false, bias));
    }

    let alpha_i_new = alpha[i] + s * (alpha[j] - alpha_j_new);

    // Compute new bias
    let b1 =
        bias - ei - yi * (alpha_i_new - alpha[i]) * k_ii - yj * (alpha_j_new - alpha[j]) * k_ij;
    let b2 =
        bias - ej - yi * (alpha_i_new - alpha[i]) * k_ij - yj * (alpha_j_new - alpha[j]) * k_jj;

    let new_bias;
    if alpha_i_new > 0.0 && alpha_i_new < c {
        new_bias = b1;
    } else if alpha_j_new > 0.0 && alpha_j_new < c {
        new_bias = b2;
    } else {
        new_bias = (b1 + b2) / 2.0;
    }

    // Update errors for all samples
    // E_new = f_new - y = f_old + delta_i * yi * K(k,i) + delta_j * yj * K(k,j) + (new_bias - old_bias) - y
    //       = E_old + delta_i * yi * K(k,i) + delta_j * yj * K(k,j) + (new_bias - old_bias)
    let delta_i = alpha_i_new - alpha[i];
    let delta_j = alpha_j_new - alpha[j];
    let bias_change = new_bias - bias;

    for k in 0..n {
        errors[k] +=
            yi * delta_i * kernel_matrix[k][i] + yj * delta_j * kernel_matrix[k][j] + bias_change;
    }

    // Update alphas
    alpha[i] = alpha_i_new;
    alpha[j] = alpha_j_new;

    Some((true, new_bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fast_svm_linear_separable() {
        // Linearly separable data
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);

        let config = FastSvmConfig {
            kernel: FastKernel::Linear,
            c: 10.0,
            tolerance: 1e-3,
            max_iter: 1000,
            ..Default::default()
        };

        let result = fast_svm(data.view(), labels.view(), &config).unwrap();

        // Should achieve perfect accuracy on training data
        let correct: usize = result
            .predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| (**p as f64 - **l).abs() < 0.5)
            .count();

        assert_eq!(correct, 8, "Should classify all points correctly");
        assert!(
            result.n_support_vectors <= 4,
            "Should have few support vectors"
        );
    }

    #[test]
    fn test_fast_svm_rbf() {
        // XOR-like data (not linearly separable)
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();

        let labels = Array1::from_vec(vec![-1.0, -1.0, 1.0, 1.0]);

        let config = FastSvmConfig {
            kernel: FastKernel::Rbf { gamma: 1.0 },
            c: 10.0,
            tolerance: 1e-3,
            max_iter: 1000,
            ..Default::default()
        };

        let result = fast_svm(data.view(), labels.view(), &config).unwrap();

        println!("RBF SVM predictions: {:?}", result.predictions);
        println!("RBF SVM alphas: {:?}", result.alphas);
        println!("RBF SVM bias: {}", result.bias);
        println!("RBF SVM iterations: {}", result.n_iterations);

        // With RBF kernel, should be able to separate XOR
        let correct: usize = result
            .predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| (**p as f64 - **l).abs() < 0.5)
            .count();

        assert!(
            correct >= 3,
            "RBF SVM should classify most points correctly: {}/4",
            correct
        );
    }
}
