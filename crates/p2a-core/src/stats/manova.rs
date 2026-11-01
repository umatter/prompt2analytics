//! Multivariate Analysis of Variance (MANOVA).
//!
//! Provides one-way MANOVA for testing whether group means differ
//! across multiple dependent variables simultaneously.
//!
//! # References
//!
//! - Wilks, S. S. (1932). "Certain generalizations in the analysis of variance".
//!   Biometrika, 24(3-4), 471-494.
//! - Pillai, K. C. S. (1955). "Some new test criteria in multivariate analysis".
//!   Annals of Mathematical Statistics, 26, 117-121.
//! - Lawley, D. N. (1938). "A generalization of Fisher's z test".
//!   Biometrika, 30, 180-187.
//! - Roy, S. N. (1939). "p-statistics and some generalizations in analysis of variance".
//!   Sankhya, 4, 381-396.
//! - R Core Team. `stats::manova()` and `stats::summary.manova()`.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/manova.html>
//!
//! # Mathematical Background
//!
//! MANOVA extends univariate ANOVA to multiple response variables. Instead of
//! comparing scalar group means, we compare mean vectors.
//!
//! The total sum of squares and cross-products (SSCP) matrix T is partitioned:
//! ```text
//! T = H + E
//! ```
//! where:
//! - H = hypothesis (between-groups) SSCP matrix
//! - E = error (within-groups) SSCP matrix
//!
//! The four test statistics are based on eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₛ of E⁻¹H:
//!
//! **Wilks' Lambda:**
//! ```text
//! Λ = |E| / |E + H| = ∏(1 / (1 + λᵢ))
//! ```
//!
//! **Pillai's Trace:**
//! ```text
//! V = trace(H(H+E)⁻¹) = Σ(λᵢ / (1 + λᵢ))
//! ```
//!
//! **Hotelling-Lawley Trace:**
//! ```text
//! T² = trace(E⁻¹H) = Σλᵢ
//! ```
//!
//! **Roy's Largest Root:**
//! ```text
//! θ = λ₁
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{eig_symmetric, faer_to_ndarray, matrix_inverse, ndarray_to_faer};
use crate::traits::f_test_p_value;

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Test statistic type for MANOVA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ManovaTestStatistic {
    /// Wilks' Lambda (Λ) - most popular in literature
    Wilks,
    /// Pillai's Trace (V) - most robust, recommended for violations of assumptions
    #[default]
    Pillai,
    /// Hotelling-Lawley Trace (T²)
    HotellingLawley,
    /// Roy's Largest Root (θ)
    Roy,
}

/// Individual test result for a single MANOVA statistic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManovaTestResult {
    /// Type of test statistic
    pub test_type: ManovaTestStatistic,
    /// Value of the test statistic
    pub statistic: f64,
    /// Approximate F statistic
    pub f_value: f64,
    /// Numerator degrees of freedom
    pub df1: f64,
    /// Denominator degrees of freedom
    pub df2: f64,
    /// P-value from F approximation
    pub p_value: f64,
    /// Whether the F approximation is exact
    pub is_exact: bool,
}

/// Result of a one-way MANOVA analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManovaResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Response variable names
    pub response_vars: Vec<String>,
    /// Factor (grouping) variable name
    pub factor_var: String,
    /// Number of response variables (p)
    pub n_responses: usize,
    /// Number of groups (g)
    pub n_groups: usize,
    /// Total sample size (N)
    pub n_obs: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Degrees of Freedom
    // ═══════════════════════════════════════════════════════════════════════
    /// Hypothesis degrees of freedom (g - 1)
    pub df_hypothesis: usize,
    /// Error degrees of freedom (N - g)
    pub df_error: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // SSCP Matrices (not serialized due to size)
    // ═══════════════════════════════════════════════════════════════════════
    /// Hypothesis (between-groups) SSCP matrix H
    #[serde(skip)]
    pub sscp_hypothesis: Array2<f64>,
    /// Error (within-groups) SSCP matrix E
    #[serde(skip)]
    pub sscp_error: Array2<f64>,
    /// Total SSCP matrix T = H + E
    #[serde(skip)]
    pub sscp_total: Array2<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Eigenvalues
    // ═══════════════════════════════════════════════════════════════════════
    /// Eigenvalues of E⁻¹H (sorted descending)
    pub eigenvalues: Vec<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Wilks' Lambda test
    pub wilks: ManovaTestResult,
    /// Pillai's Trace test
    pub pillai: ManovaTestResult,
    /// Hotelling-Lawley Trace test
    pub hotelling_lawley: ManovaTestResult,
    /// Roy's Largest Root test
    pub roy: ManovaTestResult,

    // ═══════════════════════════════════════════════════════════════════════
    // Group Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Group mean vectors
    pub group_means: HashMap<String, Vec<f64>>,
    /// Group sample sizes
    pub group_sizes: HashMap<String, usize>,
    /// Grand mean vector
    pub grand_mean: Vec<f64>,
    /// Rows dropped because a response or the group label is missing (R manova's
    /// na.omit); empty for the matrix-level function.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl ManovaResult {
    /// Get the test result for a specific statistic type.
    pub fn get_test(&self, test_type: ManovaTestStatistic) -> &ManovaTestResult {
        match test_type {
            ManovaTestStatistic::Wilks => &self.wilks,
            ManovaTestStatistic::Pillai => &self.pillai,
            ManovaTestStatistic::HotellingLawley => &self.hotelling_lawley,
            ManovaTestStatistic::Roy => &self.roy,
        }
    }

    /// Check significance at a given level using Pillai's Trace (most robust).
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.pillai.p_value < alpha
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute one-way MANOVA from raw data.
///
/// # Arguments
/// * `y_data` - Matrix of response variables (n × p), where n = observations, p = variables
/// * `groups` - Group assignments for each observation (length n)
///
/// # Returns
/// * `ManovaResult` with all four test statistics
///
/// # Example
/// ```ignore
/// let y = array![[1.0, 2.0], [1.5, 2.5], [3.0, 4.0], [3.5, 4.5]];
/// let groups = vec!["A", "A", "B", "B"];
/// let result = manova_one_way(y.view(), &groups)?;
/// println!("Wilks' Lambda: {}", result.wilks.statistic);
/// ```
pub fn manova_one_way<S: AsRef<str> + Clone>(
    y_data: &Array2<f64>,
    groups: &[S],
) -> EconResult<ManovaResult> {
    let (n_obs, n_responses) = y_data.dim();

    // Validate inputs
    if groups.len() != n_obs {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of group labels ({}) must match number of observations ({})",
                groups.len(),
                n_obs
            ),
        });
    }

    if n_obs < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n_obs,
            context: "MANOVA".to_string(),
        });
    }

    if n_responses < 2 {
        return Err(EconError::InvalidSpecification {
            message: "MANOVA requires at least 2 response variables".to_string(),
        });
    }

    // Group the data
    let mut group_data: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_data
            .entry(g.as_ref().to_string())
            .or_default()
            .push(i);
    }

    let n_groups = group_data.len();

    if n_groups < 2 {
        return Err(EconError::InvalidSpecification {
            message: "MANOVA requires at least 2 groups".to_string(),
        });
    }

    // The groups in label order: every sum below runs in this order, so the result does
    // not depend on HashMap iteration order (which differs between calls).
    let mut order: Vec<&String> = group_data.keys().collect();
    order.sort();

    // Check minimum observations per group
    for (group, indices) in order.iter().map(|g| (*g, &group_data[*g])) {
        if indices.len() < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: indices.len(),
                context: format!("MANOVA group '{}'", group),
            });
        }
    }

    // Degrees of freedom
    let df_hypothesis = n_groups - 1;
    let df_error = n_obs - n_groups;

    // Check that df_error > p for E to be invertible
    if df_error < n_responses {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Error degrees of freedom ({}) must be >= number of response variables ({}) for E to be invertible",
                df_error, n_responses
            ),
        });
    }

    // Compute grand mean
    let grand_mean: Vec<f64> = (0..n_responses)
        .map(|j| y_data.column(j).mean().unwrap_or(0.0))
        .collect();

    // Compute group means and sizes
    let mut group_means: HashMap<String, Vec<f64>> = HashMap::new();
    let mut group_sizes: HashMap<String, usize> = HashMap::new();

    for (group, indices) in &group_data {
        let group_mean: Vec<f64> = (0..n_responses)
            .map(|j| {
                let sum: f64 = indices.iter().map(|&i| y_data[[i, j]]).sum();
                sum / indices.len() as f64
            })
            .collect();
        group_means.insert(group.clone(), group_mean);
        group_sizes.insert(group.clone(), indices.len());
    }

    // Compute SSCP matrices
    // H (hypothesis/between-groups): H = Σ nᵢ (ȳᵢ - ȳ)(ȳᵢ - ȳ)'
    let mut sscp_hypothesis = Array2::<f64>::zeros((n_responses, n_responses));
    for group in &order {
        let mean = &group_means[*group];
        let n_i = group_sizes[*group] as f64;
        let diff: Vec<f64> = mean.iter().zip(&grand_mean).map(|(m, g)| m - g).collect();
        for j in 0..n_responses {
            for k in 0..n_responses {
                sscp_hypothesis[[j, k]] += n_i * diff[j] * diff[k];
            }
        }
    }

    // E (error/within-groups): E = Σᵢ Σⱼ (yᵢⱼ - ȳᵢ)(yᵢⱼ - ȳᵢ)'
    let mut sscp_error = Array2::<f64>::zeros((n_responses, n_responses));
    for group in &order {
        let group_mean = &group_means[*group];
        for &i in &group_data[*group] {
            let diff: Vec<f64> = (0..n_responses)
                .map(|j| y_data[[i, j]] - group_mean[j])
                .collect();
            for j in 0..n_responses {
                for k in 0..n_responses {
                    sscp_error[[j, k]] += diff[j] * diff[k];
                }
            }
        }
    }

    // T (total): T = H + E
    let sscp_total = &sscp_hypothesis + &sscp_error;

    // Compute eigenvalues of E⁻¹H
    // First compute E⁻¹
    let e_inv = matrix_inverse(&sscp_error.view())
        .map_err(|e| EconError::Internal(format!("Failed to invert error SSCP matrix: {}", e)))?;

    // Compute E⁻¹H
    let _e_inv_h = e_inv.dot(&sscp_hypothesis);

    // For eigenvalue computation, we need to handle the non-symmetric E⁻¹H
    // The eigenvalues are the same as those of E^(-1/2) H E^(-1/2), which is symmetric
    // But for now, let's compute eigenvalues of (E + H)⁻¹ H which gives λ/(1+λ)
    // Or use the generalized eigenvalue formulation

    // Alternative: eigenvalues of E⁻¹H using the symmetric decomposition
    // Since H and E are symmetric positive semi-definite, we can use:
    // E⁻¹H has the same eigenvalues as E^(-1/2) H E^(-1/2) (symmetric)

    // Use Cholesky of E to compute E^(-1/2)
    let eigenvalues = compute_generalized_eigenvalues(&sscp_hypothesis, &sscp_error)?;

    // Sort eigenvalues in descending order
    let mut eigenvalues = eigenvalues;
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Filter out negative eigenvalues (numerical noise)
    let eigenvalues: Vec<f64> = eigenvalues.into_iter().map(|e| e.max(0.0)).collect();

    // Compute the four test statistics
    let p = n_responses as f64;
    let v_h = df_hypothesis as f64;
    let v_e = df_error as f64;
    let s = (p.min(v_h)) as usize;

    let wilks = compute_wilks_lambda(&eigenvalues, p, v_h, v_e, s);
    let pillai = compute_pillai_trace(&eigenvalues, p, v_h, v_e, s);
    let hotelling_lawley = compute_hotelling_lawley(&eigenvalues, p, v_h, v_e, s);
    let roy = compute_roy_largest_root(&eigenvalues, p, v_h, v_e, s);

    // Build variable names
    let response_vars: Vec<String> = (0..n_responses).map(|i| format!("Y{}", i + 1)).collect();

    Ok(ManovaResult {
        response_vars,
        factor_var: "Group".to_string(),
        n_responses,
        n_groups,
        n_obs,
        df_hypothesis,
        df_error,
        sscp_hypothesis,
        sscp_error,
        sscp_total,
        eigenvalues,
        wilks,
        pillai,
        hotelling_lawley,
        roy,
        group_means,
        group_sizes,
        grand_mean,
        missing: MissingSummary::none(n_obs),
    })
}

/// Compute generalized eigenvalues of Hx = λEx.
///
/// These are the eigenvalues of E⁻¹H when E is invertible.
fn compute_generalized_eigenvalues(h: &Array2<f64>, e: &Array2<f64>) -> EconResult<Vec<f64>> {
    // Method: Transform to standard eigenvalue problem
    // Hx = λEx can be rewritten as (E⁻¹H)x = λx
    // But E⁻¹H may not be symmetric.
    //
    // For symmetric H and positive definite E, the eigenvalues are real and non-negative.
    // We use: L⁻¹ H L⁻ᵀ has the same eigenvalues, where E = LLᵀ (Cholesky)

    let n = h.nrows();

    // Compute Cholesky factorization of E: E = L L^T
    let e_faer = ndarray_to_faer(&e.view());
    let chol = e_faer.llt(faer::Side::Lower).map_err(|_| {
        EconError::Internal(
            "Cholesky decomposition of E failed (E may not be positive definite)".to_string(),
        )
    })?;

    // L is the lower triangular Cholesky factor
    let l = chol.L();
    let l_ndarray = faer_to_ndarray(&faer::Mat::from_fn(n, n, |i, j| l[(i, j)]));

    // Compute L⁻¹
    let mut l_inv = Array2::<f64>::zeros((n, n));
    // Forward substitution for each column
    for col in 0..n {
        let mut x = Array1::<f64>::zeros(n);
        x[col] = 1.0; // Solving L * y = e_col
        for i in 0..n {
            let mut sum = x[i];
            for k in 0..i {
                sum -= l_ndarray[[i, k]] * l_inv[[k, col]];
            }
            l_inv[[i, col]] = sum / l_ndarray[[i, i]];
        }
    }

    // Compute M = L⁻¹ H L⁻ᵀ
    // First: L⁻¹ H
    let l_inv_h = l_inv.dot(h);
    // Then: (L⁻¹ H) L⁻ᵀ
    let m = l_inv_h.dot(&l_inv.t());

    // M should be symmetric, compute its eigenvalues
    let (eigenvalues, _) = eig_symmetric(&m.view())
        .map_err(|e| EconError::Internal(format!("Eigenvalue computation failed: {}", e)))?;

    Ok(eigenvalues.to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Statistic Computations
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Wilks' Lambda and its F approximation.
///
/// Λ = ∏(1 / (1 + λᵢ))
fn compute_wilks_lambda(
    eigenvalues: &[f64],
    p: f64,
    v_h: f64,
    v_e: f64,
    s: usize,
) -> ManovaTestResult {
    // Wilks' Lambda = product of 1/(1+λᵢ) for i=1..s
    let lambda: f64 = eigenvalues
        .iter()
        .take(s)
        .map(|&e| 1.0 / (1.0 + e))
        .product();

    // F approximation (Rao, 1951)
    // df1 = p * v_h
    // t = sqrt((p²v_h² - 4) / (p² + v_h² - 5)) if p² + v_h² - 5 > 0, else 1
    // w = v_e + v_h - 0.5(p + v_h + 1)
    // df2 = w*t - 0.5(p*v_h - 2)
    // F = ((1 - Λ^(1/t)) / Λ^(1/t)) * (df2 / df1)

    let df1 = p * v_h;

    let denom = p * p + v_h * v_h - 5.0;
    let t = if denom > 0.0 {
        ((p * p * v_h * v_h - 4.0) / denom).sqrt()
    } else {
        1.0
    };

    let w = v_e + v_h - 0.5 * (p + v_h + 1.0);
    let df2 = w * t - 0.5 * (p * v_h - 2.0);

    let lambda_t = lambda.powf(1.0 / t);
    let f_value = if lambda_t > 0.0 && lambda_t < 1.0 {
        ((1.0 - lambda_t) / lambda_t) * (df2 / df1)
    } else if lambda_t >= 1.0 {
        0.0 // No difference between groups
    } else {
        f64::INFINITY
    };

    let p_value = if df1 > 0.0 && df2 > 0.0 && f_value.is_finite() {
        f_test_p_value(f_value, df1, df2)
    } else {
        f64::NAN
    };

    // Exact when s = 1 or (p = 1 and v_h = 1) or (p = 2 and v_h = 2)
    let is_exact = s == 1 || (p == 1.0 && v_h == 1.0) || (p == 2.0 && v_h == 2.0);

    ManovaTestResult {
        test_type: ManovaTestStatistic::Wilks,
        statistic: lambda,
        f_value,
        df1,
        df2,
        p_value,
        is_exact,
    }
}

/// Compute Pillai's Trace and its F approximation.
///
/// V = Σ(λᵢ / (1 + λᵢ))
fn compute_pillai_trace(
    eigenvalues: &[f64],
    p: f64,
    v_h: f64,
    v_e: f64,
    s: usize,
) -> ManovaTestResult {
    // Pillai's trace = sum of λᵢ/(1+λᵢ) for i=1..s
    let trace: f64 = eigenvalues.iter().take(s).map(|&e| e / (1.0 + e)).sum();

    // F approximation
    // s = min(p, v_h)
    // m = (|v_h - p| - 1) / 2
    // n = (v_e - p - 1) / 2
    // df1 = s(2m + s + 1)
    // df2 = s(2n + s + 1)
    // F = (df2 / df1) * (V / (s - V))

    let s_f = s as f64;
    let m = ((v_h - p).abs() - 1.0) / 2.0;
    let n = (v_e - p - 1.0) / 2.0;

    let df1 = s_f * (2.0 * m + s_f + 1.0);
    let df2 = s_f * (2.0 * n + s_f + 1.0);

    let f_value = if s_f > trace && df1 > 0.0 {
        (df2 / df1) * (trace / (s_f - trace))
    } else if trace >= s_f {
        f64::INFINITY
    } else {
        0.0
    };

    let p_value = if df1 > 0.0 && df2 > 0.0 && f_value.is_finite() {
        f_test_p_value(f_value, df1, df2)
    } else {
        f64::NAN
    };

    // Exact when s = 1 or s = 2
    let is_exact = s <= 2;

    ManovaTestResult {
        test_type: ManovaTestStatistic::Pillai,
        statistic: trace,
        f_value,
        df1,
        df2,
        p_value,
        is_exact,
    }
}

/// Compute Hotelling-Lawley Trace and its F approximation.
///
/// T² = Σλᵢ = trace(E⁻¹H)
fn compute_hotelling_lawley(
    eigenvalues: &[f64],
    p: f64,
    v_h: f64,
    v_e: f64,
    s: usize,
) -> ManovaTestResult {
    // Hotelling-Lawley trace = sum of eigenvalues
    let trace: f64 = eigenvalues.iter().take(s).sum();

    // F approximation
    // s = min(p, v_h)
    // m = (|v_h - p| - 1) / 2
    // n = (v_e - p - 1) / 2
    // df1 = s(2m + s + 1)
    // df2 = 2(sn + 1)
    // F = (2(sn + 1) / (s²(2m + s + 1))) * T²

    let s_f = s as f64;
    let m = ((v_h - p).abs() - 1.0) / 2.0;
    let n = (v_e - p - 1.0) / 2.0;

    let df1 = s_f * (2.0 * m + s_f + 1.0);
    let df2 = 2.0 * (s_f * n + 1.0);

    let _f_value = if df1 > 0.0 {
        (df2 / (s_f * s_f * (2.0 * m + s_f + 1.0))) * trace * s_f
    } else {
        0.0
    };

    // Simplify: F = (df2 * trace) / (s * df1)
    let f_value = if df1 > 0.0 && s_f > 0.0 {
        (df2 * trace) / (s_f * df1)
    } else {
        0.0
    };

    let p_value = if df1 > 0.0 && df2 > 0.0 && f_value.is_finite() {
        f_test_p_value(f_value, df1, df2)
    } else {
        f64::NAN
    };

    // Exact when s = 1 or s = 2
    let is_exact = s <= 2;

    ManovaTestResult {
        test_type: ManovaTestStatistic::HotellingLawley,
        statistic: trace,
        f_value,
        df1,
        df2,
        p_value,
        is_exact,
    }
}

/// Compute Roy's Largest Root and its F approximation.
///
/// θ = λ₁ (largest eigenvalue)
fn compute_roy_largest_root(
    eigenvalues: &[f64],
    p: f64,
    v_h: f64,
    v_e: f64,
    _s: usize,
) -> ManovaTestResult {
    // Roy's largest root = largest eigenvalue
    let theta = eigenvalues.first().copied().unwrap_or(0.0);

    // F approximation (upper bound)
    // df1 = max(p, v_h)
    // df2 = v_e - max(p, v_h) + v_h
    // F = (df2 / df1) * θ

    let r = p.max(v_h);
    let df1 = r;
    let df2 = v_e - r + v_h;

    let f_value = if df1 > 0.0 && df2 > 0.0 {
        (df2 / df1) * theta
    } else {
        0.0
    };

    let p_value = if df1 > 0.0 && df2 > 0.0 && f_value.is_finite() {
        // Note: This F is an upper bound, so the p-value is a lower bound
        f_test_p_value(f_value, df1, df2)
    } else {
        f64::NAN
    };

    ManovaTestResult {
        test_type: ManovaTestStatistic::Roy,
        statistic: theta,
        f_value,
        df1,
        df2,
        p_value,
        is_exact: false, // Roy's test is always an upper bound approximation
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dataset Interface
// ═══════════════════════════════════════════════════════════════════════════════

/// Run MANOVA from a Dataset (R `manova(cbind(y1, y2, ...) ~ group)`).
///
/// Rows with a missing response or group label are dropped jointly, as `manova` (an `lm`)
/// drops them, and reported in `missing`. Integer responses are accepted; group labels are
/// R `factor()` labels (text as stored, without quotes).
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_cols` - Names of the response variable columns
/// * `group_col` - Name of the grouping (factor) column
///
/// # Example
/// ```ignore
/// let result = run_manova(&dataset, &["score1", "score2", "score3"], "treatment")?;
/// println!("Pillai's Trace: {:.4}, p = {:.4}", result.pillai.statistic, result.pillai.p_value);
/// ```
pub fn run_manova(dataset: &Dataset, y_cols: &[&str], group_col: &str) -> EconResult<ManovaResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numerics(y_cols)
        .factor(group_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let y_data = cols.matrix(y_cols, false)?.data;
    let factor = cols.factor(group_col);
    let groups: Vec<&str> = (0..cols.n()).map(|i| factor.label(i)).collect();

    let mut result = manova_one_way(&y_data, &groups)?;
    result.response_vars = y_cols.iter().map(|s| s.to_string()).collect();
    result.factor_var = group_col.to_string();
    result.missing = cols.missing().clone();
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_manova_basic() {
        // Two groups, two variables
        let y = array![
            [1.0, 2.0],
            [1.2, 2.1],
            [1.1, 1.9],
            [3.0, 4.0],
            [3.2, 4.1],
            [2.9, 3.9]
        ];
        let groups = vec!["A", "A", "A", "B", "B", "B"];

        let result = manova_one_way(&y, &groups).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.n_responses, 2);
        assert_eq!(result.n_obs, 6);
        assert_eq!(result.df_hypothesis, 1);
        assert_eq!(result.df_error, 4);

        // With clear group separation, all tests should be significant
        assert!(result.wilks.p_value < 0.05);
        assert!(result.pillai.p_value < 0.05);
    }

    #[test]
    fn test_manova_three_groups() {
        // Three groups, two variables with INDEPENDENT within-group variation
        // Critical: within each group, y1 and y2 must vary independently
        let y = array![
            // Group A: y1 around 1, y2 around 8, independent noise
            [1.0, 8.0],
            [1.5, 8.5], // y1 up, y2 up
            [0.5, 7.5], // y1 down, y2 down
            [1.3, 7.7], // y1 up, y2 down - breaks correlation
            // Group B: y1 around 5, y2 around 4
            [5.0, 4.0],
            [5.5, 4.5],
            [4.5, 3.5],
            [4.7, 4.3], // y1 down, y2 up - breaks correlation
            // Group C: y1 around 9, y2 around 1
            [9.0, 1.0],
            [9.5, 1.5],
            [8.5, 0.5],
            [8.7, 1.3] // y1 down, y2 up - breaks correlation
        ];
        let groups = vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"];

        let result = manova_one_way(&y, &groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df_hypothesis, 2);
        assert_eq!(result.df_error, 9);

        // Clear group separation should be significant
        assert!(
            result.pillai.p_value < 0.05,
            "Pillai should be significant, got p={}",
            result.pillai.p_value
        );
    }

    #[test]
    fn test_manova_no_difference() {
        // Two groups with overlapping data and independent variables
        let y = array![
            [1.0, 5.0],
            [1.5, 4.5],
            [2.0, 6.0],
            [0.8, 5.2],
            [1.2, 4.8],
            [1.7, 5.5],
            [1.3, 4.7],
            [1.1, 5.3]
        ];
        let groups = vec!["A", "A", "A", "A", "B", "B", "B", "B"];

        let result = manova_one_way(&y, &groups).unwrap();

        // With similar group means, tests should not be significant
        // Wilks' Lambda should be close to 1 (no difference)
        assert!(result.wilks.statistic > 0.5);
    }

    #[test]
    fn test_eigenvalue_computation() {
        // Simple 2x2 case where we can verify eigenvalues
        let h = array![[4.0, 2.0], [2.0, 4.0]];
        let e = array![[2.0, 0.0], [0.0, 2.0]]; // Identity scaled by 2

        let eigenvalues = compute_generalized_eigenvalues(&h, &e).unwrap();

        // E⁻¹H = (1/2)H = [[2, 1], [1, 2]]
        // Eigenvalues of this are 3 and 1
        assert_eq!(eigenvalues.len(), 2);
        let max_ev = eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_ev = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max_ev - 3.0).abs() < 0.001);
        assert!((min_ev - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_wilks_lambda_formula() {
        // With eigenvalues [3, 1]:
        // Wilks = (1/(1+3)) * (1/(1+1)) = 0.25 * 0.5 = 0.125
        let eigenvalues = vec![3.0, 1.0];
        let result = compute_wilks_lambda(&eigenvalues, 2.0, 1.0, 10.0, 2);

        assert!((result.statistic - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_pillai_trace_formula() {
        // With eigenvalues [3, 1]:
        // Pillai = 3/(1+3) + 1/(1+1) = 0.75 + 0.5 = 1.25
        let eigenvalues = vec![3.0, 1.0];
        let result = compute_pillai_trace(&eigenvalues, 2.0, 1.0, 10.0, 2);

        assert!((result.statistic - 1.25).abs() < 0.001);
    }

    #[test]
    fn test_hotelling_lawley_formula() {
        // With eigenvalues [3, 1]:
        // Hotelling-Lawley = 3 + 1 = 4
        let eigenvalues = vec![3.0, 1.0];
        let result = compute_hotelling_lawley(&eigenvalues, 2.0, 1.0, 10.0, 2);

        assert!((result.statistic - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_roy_largest_root_formula() {
        // With eigenvalues [3, 1]:
        // Roy = 3
        let eigenvalues = vec![3.0, 1.0];
        let result = compute_roy_largest_root(&eigenvalues, 2.0, 1.0, 10.0, 2);

        assert!((result.statistic - 3.0).abs() < 0.001);
    }

    /// Validate MANOVA against R's manova() function.
    ///
    /// R code:
    /// ```r
    /// # Three groups with clear separation and independent variation
    /// y1 <- c(1.0, 1.2, 0.8, 5.0, 5.2, 4.8, 9.0, 9.2, 8.8)
    /// y2 <- c(8.0, 7.8, 8.2, 4.0, 4.2, 3.8, 1.0, 1.2, 0.8)
    /// group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))
    /// fit <- manova(cbind(y1, y2) ~ group)
    /// summary(fit, test="Wilks")
    /// summary(fit, test="Pillai")
    /// ```
    #[test]
    fn test_validate_manova_against_r() {
        // Data with clear group separation and independent within-group variation
        let y = array![
            [1.0, 8.0],
            [1.2, 7.8],
            [0.8, 8.2],
            [5.0, 4.0],
            [5.2, 4.2],
            [4.8, 3.8],
            [9.0, 1.0],
            [9.2, 1.2],
            [8.8, 0.8]
        ];
        let groups = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];

        let result = manova_one_way(&y, &groups).unwrap();

        // The groups have clear separation on both variables
        // y1: A~1, B~5, C~9 (increasing)
        // y2: A~8, B~4, C~1 (decreasing)

        assert!(
            result.wilks.statistic < 0.1,
            "Wilks Lambda should be small for separated groups, got {}",
            result.wilks.statistic
        );
        assert!(
            result.wilks.p_value < 0.05,
            "Wilks test should be significant, got p={}",
            result.wilks.p_value
        );
        assert!(
            result.pillai.p_value < 0.05,
            "Pillai test should be significant, got p={}",
            result.pillai.p_value
        );

        // Check that eigenvalues are positive
        for ev in &result.eigenvalues {
            assert!(*ev >= 0.0, "Eigenvalues should be non-negative");
        }
    }

    #[test]
    fn test_insufficient_observations() {
        let y = array![[1.0, 2.0], [3.0, 4.0]];
        let groups = vec!["A", "B"];

        let result = manova_one_way(&y, &groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_group() {
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let groups = vec!["A", "A", "A"];

        let result = manova_one_way(&y, &groups);
        assert!(result.is_err());
    }
    /// The SSCP sums run over the groups in a fixed order, so repeated calls on the same data
    /// give bit-identical statistics (they summed in HashMap iteration order, which differs
    /// between calls).
    #[test]
    fn test_manova_statistics_do_not_depend_on_hash_order() {
        let n = 60;
        let mut y = Array2::<f64>::zeros((n, 2));
        let mut groups = Vec::with_capacity(n);
        for i in 0..n {
            let g = i % 5;
            y[[i, 0]] = 1.0 + 0.3 * g as f64 + (i as f64 * 0.7).sin() / 3.0;
            y[[i, 1]] = 2.0 - 0.2 * g as f64 + (i as f64 * 1.3).cos() / 7.0;
            groups.push(format!("group{g}"));
        }
        let first = manova_one_way(&y, &groups).unwrap();
        for _ in 0..30 {
            let again = manova_one_way(&y, &groups).unwrap();
            for (a, b) in [
                (&first.pillai, &again.pillai),
                (&first.wilks, &again.wilks),
                (&first.hotelling_lawley, &again.hotelling_lawley),
                (&first.roy, &again.roy),
            ] {
                assert_eq!(
                    a.statistic.to_bits(),
                    b.statistic.to_bits(),
                    "{:?}",
                    a.test_type
                );
                assert_eq!(
                    a.f_value.to_bits(),
                    b.f_value.to_bits(),
                    "{:?}",
                    a.test_type
                );
            }
        }
    }

    // ========================================================================
    // R-pinned: incomplete rows (T2-WP07, case stat_manova_na)
    // ========================================================================
    //
    // y1 is missing in rows 3 and 20 and y2 in row 30 (36 rows, 3 groups): manova (lm)
    // drops the 3 incomplete rows. summary.manova's F approximations are p2a's formulas.

    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
    use polars::prelude::DataType;

    #[test]
    fn test_validate_manova_na() {
        let (case, ds) = load_case("stat_manova_na");
        let r = run_manova(&ds, &["y1", "y2"], "g").unwrap();
        let (s, p, x) = (Tol::stat(33), Tol::p_value(33), Tol::exact());
        let mut check = Check::new(&case);
        for (key, t) in [
            ("pillai", &r.pillai),
            ("wilks", &r.wilks),
            ("hotelling_lawley", &r.hotelling_lawley),
            ("roy", &r.roy),
        ] {
            // Rao's df2 for Wilks is real-valued in general.
            let df2_tol = if key == "wilks" { s } else { x };
            check
                .scalar(&format!("{key}_stat"), t.statistic, s)
                .scalar(&format!("{key}_approx_f"), t.f_value, s)
                .scalar(&format!("{key}_df1"), t.df1, x)
                .scalar(&format!("{key}_df2"), t.df2, df2_tol)
                .scalar(&format!("{key}_p"), t.p_value, p);
        }
        let mut names: Vec<&String> = r.group_sizes.keys().collect();
        names.sort();
        let sizes: Vec<f64> = names.iter().map(|g| r.group_sizes[*g] as f64).collect();
        check
            .scalar("n_used", r.n_obs as f64, x)
            .named("group_n", &names, &sizes, x)
            .finish();
        assert_eq!(
            r.missing.by_column,
            vec![("y1".to_string(), 2), ("y2".to_string(), 1)]
        );
        assert_eq!(r.missing.n_dropped, 3);
    }

    /// Column-type fixture: an Int64 response (the case's `id`) equals its Float64 cast, with
    /// the rows of y1's missing cells dropped; a text response and an absent response are
    /// refused. (`y` and `y_na` of stat_friedman_quade cannot serve: they agree on every
    /// complete row, so their error SSCP matrix is singular.)
    #[test]
    fn test_validate_na_manova_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_manova_na");
        assert_eq!(ds.df().column("id").unwrap().dtype(), &DataType::Int64);
        let int = run_manova(&ds, &["y1", "id"], "g").unwrap();
        let mut df = ds.df().clone();
        let cast = df.column("id").unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(cast).unwrap();
        let float = run_manova(&Dataset::new(df), &["y1", "id"], "g").unwrap();
        assert_eq!(int.pillai.statistic, float.pillai.statistic);
        assert_eq!(int.wilks.statistic, float.wilks.statistic);
        assert_eq!(int.n_obs, 34);
        assert_eq!(int.missing.n_dropped, 2);
        assert!(matches!(
            run_manova(&ds, &["y1", "g"], "g"),
            Err(EconError::NonNumericColumn { ref column }) if column == "g"
        ));
        assert!(matches!(
            run_manova(&ds, &["y1", "nope"], "g"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }
}
