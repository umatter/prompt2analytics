//! Support Vector Machine (Linear SVM).
//!
//! Pure Rust implementation using Sequential Minimal Optimization (SMO).

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
        let mut indexed: Vec<(usize, f64)> = self.weights
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

        for (i, weight) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("Feature_{}", i)),
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
        writeln!(f, "  Class distribution: {} negative, {} positive", neg_count, pos_count)?;

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
                    (f64::max(0.0, alpha[i] + alpha[j] - c_param), f64::min(c_param, alpha[i] + alpha[j]))
                } else {
                    // y_i != y_j
                    (f64::max(0.0, alpha[j] - alpha[i]), f64::min(c_param, c_param + alpha[j] - alpha[i]))
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
                let b1 = b - e_i - y[i] * (alpha[i] - alpha_i_old) * k_ii
                    - y[j] * (alpha[j] - alpha_j_old) * k_ij;
                let b2 = b - e_j - y[i] * (alpha[i] - alpha_i_old) * k_ij
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
            let f: f64 = row.iter().zip(weights.iter()).map(|(x, w)| x * w).sum::<f64>() + bias;
            if f >= 0.0 {
                class_labels.1
            } else {
                class_labels.0
            }
        })
        .collect()
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

        let result = linear_svm(
            x.view(),
            y.view(),
            Some(1.0),
            Some(100),
            Some(1e-3),
            None,
        ).unwrap();

        assert!(result.n_support_vectors > 0);
        assert_eq!(result.weights.len(), 2);

        // Should classify training data correctly (or mostly correctly)
        let correct: usize = result.predictions.iter()
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
            [1.0, 1.0],  // 1+1-3 = -1 -> class 0
            [2.0, 2.0],  // 2+2-3 = 1 -> class 1
            [3.0, 0.0],  // 3+0-3 = 0 -> class 1
        ];

        let preds = svm_predict(data.view(), &weights, bias, class_labels);
        assert_eq!(preds, vec![0, 1, 1]);
    }

    #[test]
    fn test_svm_requires_two_classes() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ];
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
        ).unwrap();

        // First weight should be larger in magnitude
        assert!(result.weights[0].abs() > result.weights[1].abs());
    }
}
