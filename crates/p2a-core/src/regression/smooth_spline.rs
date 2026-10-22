//! Smoothing spline regression.
//!
//! Implements R's `smooth.spline()` function which fits a cubic smoothing spline
//! to data by minimizing a penalized sum of squares criterion.
//!
//! # References
//!
//! - Green, P. J., & Silverman, B. W. (1994). "Nonparametric Regression and
//!   Generalized Linear Models: A Roughness Penalty Approach". Chapman & Hall.
//! - R Core Team. `stats::smooth.spline()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/smooth.spline.html>

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource;
use serde::{Deserialize, Serialize};

/// Configuration for smooth.spline fitting.
#[derive(Debug, Clone, Default)]
pub struct SmoothSplineConfig {
    /// Equivalent degrees of freedom for the spline. If specified, spar is ignored.
    pub df: Option<f64>,
    /// Smoothing parameter. If neither df nor spar is specified, uses cross-validation.
    pub spar: Option<f64>,
    /// Penalty parameter lambda (alternative to spar). spar = log(lambda) / 6.
    pub lambda: Option<f64>,
    /// Use all unique x values as knots (default: false, uses a subset for efficiency).
    pub all_knots: bool,
    /// Use generalized cross-validation (true) or ordinary leave-one-out CV (false).
    pub cv: bool,
    /// Weights for observations (optional).
    pub weights: Option<Vec<f64>>,
}

impl SmoothSplineConfig {
    /// Create a new configuration with a specified degrees of freedom.
    pub fn with_df(df: f64) -> Self {
        Self {
            df: Some(df),
            ..Default::default()
        }
    }

    /// Create a new configuration with a specified smoothing parameter.
    pub fn with_spar(spar: f64) -> Self {
        Self {
            spar: Some(spar),
            ..Default::default()
        }
    }

    /// Use cross-validation to select smoothing parameter.
    pub fn with_cv(gcv: bool) -> Self {
        Self {
            cv: gcv,
            ..Default::default()
        }
    }
}

/// Result of smooth.spline fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothSplineResult {
    /// Unique x values (sorted).
    pub x: Vec<f64>,
    /// Fitted y values at the unique x points.
    pub y: Vec<f64>,
    /// Equivalent degrees of freedom.
    pub df: f64,
    /// Smoothing parameter spar.
    pub spar: f64,
    /// Penalty parameter lambda.
    pub lambda: f64,
    /// Cross-validation score (if CV was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cv_crit: Option<f64>,
    /// Residuals at original data points.
    pub residuals: Vec<f64>,
    /// Leverage values (diagonal of smoother matrix).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leverage: Option<Vec<f64>>,
    /// Number of original observations.
    pub n_obs: usize,
    /// Number of unique x values (knots).
    pub n_knots: usize,
}

/// Fit a cubic smoothing spline to data.
///
/// The smoothing spline minimizes a penalized sum of squares:
///
/// RSS(f, λ) = Σ w_i (y_i - f(x_i))² + λ ∫ (f''(x))² dx
///
/// where λ controls the trade-off between fit and smoothness.
///
/// # Arguments
///
/// * `x` - Predictor variable values
/// * `y` - Response variable values (same length as x)
/// * `config` - Configuration for the smoothing spline
///
/// # Returns
///
/// `SmoothSplineResult` containing the fitted spline.
///
/// # Mathematical Background
///
/// The solution is a natural cubic spline with knots at the unique x values.
/// The smoothing parameter λ determines the roughness penalty:
/// - λ → 0: interpolating spline (passes through all points)
/// - λ → ∞: linear regression (maximum smoothness)
///
/// The relationship between spar and lambda is:
/// λ = r · 256^(3·spar - 1)
///
/// where r is a scaling factor based on the data range.
///
/// # References
///
/// - R function `stats::smooth.spline()`
/// - Green & Silverman (1994). "Nonparametric Regression and GLMs".
pub fn smooth_spline(
    x: &[f64],
    y: &[f64],
    config: SmoothSplineConfig,
) -> EconResult<SmoothSplineResult> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    let n = x.len();
    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "smooth.spline requires at least 4 data points".to_string(),
        });
    }

    // Check for non-finite values
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "Input contains non-finite values".to_string(),
        });
    }

    // Sort by x and handle ties by averaging y values
    let (xs, ys, ws, _original_indices) = prepare_data(x, y, config.weights.as_deref())?;
    let n_unique = xs.len();

    if n_unique < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n_unique,
            context: "smooth.spline requires at least 4 unique x values".to_string(),
        });
    }

    // Select knots (all unique x values if all_knots=true, else use a subset)
    let knots: Vec<f64> = if config.all_knots || n_unique <= 50 {
        xs.clone()
    } else {
        select_knots(&xs, n_unique.min(50))
    };

    // Compute the roughness penalty R'R (pentadiagonal, stored as bands)
    let penalty = penalty_bands(&xs)?;

    // Determine lambda
    let (lambda, cv_score) = if let Some(df) = config.df {
        // Convert df to lambda
        (df_to_lambda(&penalty, &ws, df)?, None)
    } else if let Some(spar) = config.spar {
        // Convert spar to lambda
        let range = xs[xs.len() - 1] - xs[0];
        let r = range.powi(3) / n_unique as f64;
        (r * 256.0_f64.powf(3.0 * spar - 1.0), None)
    } else if let Some(lam) = config.lambda {
        (lam, None)
    } else {
        // Use cross-validation to select lambda
        cross_validate(&penalty, &ys, &ws, config.cv)?
    };

    // Fit the smoothing spline with the selected lambda
    let (fitted, df, leverage) = fit_smoothing_spline(&penalty, &ys, &ws, lambda)?;

    // Compute spar from lambda
    let range = xs[xs.len() - 1] - xs[0];
    let r = range.powi(3) / n_unique as f64;
    let spar = (lambda.ln() / r.ln() + 1.0) / 3.0;

    // Compute residuals at original data points
    let residuals = compute_residuals_at_original(x, y, &xs, &fitted);

    Ok(SmoothSplineResult {
        x: xs,
        y: fitted,
        df,
        spar,
        lambda,
        cv_crit: cv_score,
        residuals,
        leverage: Some(leverage),
        n_obs: n,
        n_knots: knots.len(),
    })
}

/// Predict values at new x points using a fitted smooth spline.
pub fn smooth_spline_predict(result: &SmoothSplineResult, xnew: &[f64]) -> EconResult<Vec<f64>> {
    // Use cubic spline interpolation through the fitted points
    let n = result.x.len();
    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "prediction requires at least 4 fitted points".to_string(),
        });
    }

    // Compute natural spline coefficients
    let coeffs = compute_spline_coefficients(&result.x, &result.y)?;

    // Evaluate at new points
    Ok(xnew
        .iter()
        .map(|&xi| evaluate_spline(&result.x, &result.y, &coeffs, xi))
        .collect())
}

/// Convenience function to run smooth.spline on a Dataset.
pub fn run_smooth_spline(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    df: Option<f64>,
) -> EconResult<SmoothSplineResult> {
    let df_data = dataset.df();
    let available: Vec<String> = df_data
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let x_series = df_data
        .column(x_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: x_col.to_string(),
            available: available.clone(),
        })?;
    let y_series = df_data
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: available.clone(),
        })?;

    let x: Vec<f64> = x_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: x_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();
    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let config = match df {
        Some(d) => SmoothSplineConfig::with_df(d),
        None => SmoothSplineConfig::with_cv(true),
    };

    smooth_spline(&x, &y, config)
}

// ========== Internal helper functions ==========

/// Prepare data: sort by x, handle ties by averaging y values.
fn prepare_data(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
) -> EconResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let n = x.len();

    // Create index pairs and sort by x
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].total_cmp(&x[b]));

    // Default weights
    let w: Vec<f64> = match weights {
        Some(ws) => {
            if ws.len() != n {
                return Err(EconError::InvalidSpecification {
                    message: "Weights must have same length as data".to_string(),
                });
            }
            ws.to_vec()
        }
        None => vec![1.0; n],
    };

    // Aggregate duplicate x values
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws_new = Vec::new();
    let mut original_indices = Vec::new();

    let mut i = 0;
    while i < n {
        let xi = x[indices[i]];
        let mut sum_wy = 0.0;
        let mut sum_w = 0.0;
        let start_i = i;

        while i < n && (x[indices[i]] - xi).abs() < 1e-14 {
            sum_wy += w[indices[i]] * y[indices[i]];
            sum_w += w[indices[i]];
            i += 1;
        }

        xs.push(xi);
        ys.push(sum_wy / sum_w);
        ws_new.push(sum_w);
        original_indices.push(indices[start_i]);
    }

    Ok((xs, ys, ws_new, original_indices))
}

/// Select a subset of knots (uniform spacing).
fn select_knots(x: &[f64], n_knots: usize) -> Vec<f64> {
    let n = x.len();
    if n_knots >= n {
        return x.to_vec();
    }

    let step = (n - 1) as f64 / (n_knots - 1) as f64;
    (0..n_knots)
        .map(|i| {
            let idx = (i as f64 * step).round() as usize;
            x[idx.min(n - 1)]
        })
        .collect()
}

/// The symmetric pentadiagonal roughness penalty `R'R`, stored as its diagonal
/// and its first and second super-diagonals (each of length n, zero-padded).
struct PentaPenalty {
    d0: Vec<f64>,
    d1: Vec<f64>,
    d2: Vec<f64>,
}

impl PentaPenalty {
    /// Number of rows (and columns) of the penalty.
    fn n(&self) -> usize {
        self.d0.len()
    }

    /// Entry `(i, j)` of the penalty; `0.0` outside the band.
    fn at(&self, i: usize, j: usize) -> f64 {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        match hi - lo {
            0 => self.d0[lo],
            1 => self.d1[lo],
            2 => self.d2[lo],
            _ => 0.0,
        }
    }
}

/// Build the penalty `R'R` for sorted, distinct `x` in O(n).
///
/// R (the second difference operator) is (n-2) x n and tridiagonal:
/// R[k][k-1] = 1/h[k-1], R[k][k] = -1/h[k-1] - 1/h[k], R[k][k+1] = 1/h[k]
/// for k = 1..n-2, so R'R has bandwidth 2. The bands accumulate in the same
/// k order as a dense R'R would, so every entry is bit-identical to it.
fn penalty_bands(x: &[f64]) -> EconResult<PentaPenalty> {
    let n = x.len();
    let what = "smoothing spline penalty";
    let mut d0 = resource::try_vec(n, 0.0, what)?;
    let mut d1 = resource::try_vec(n, 0.0, what)?;
    let mut d2 = resource::try_vec(n, 0.0, what)?;

    for k in 1..n.saturating_sub(1) {
        let h_left = x[k] - x[k - 1];
        let h_right = x[k + 1] - x[k];
        let r_left = 1.0 / h_left;
        let r_mid = -1.0 / h_left - 1.0 / h_right;
        let r_right = 1.0 / h_right;

        // Ω[i][j] += R[k][i] * R[k][j]; R[k] is non-zero only at k-1, k, k+1
        d0[k - 1] += r_left * r_left;
        d1[k - 1] += r_left * r_mid;
        d2[k - 1] += r_left * r_right;
        d0[k] += r_mid * r_mid;
        d1[k] += r_mid * r_right;
        d0[k + 1] += r_right * r_right;
    }

    Ok(PentaPenalty { d0, d1, d2 })
}

/// Convert degrees of freedom to lambda using bisection.
fn df_to_lambda(penalty: &PentaPenalty, weights: &[f64], target_df: f64) -> EconResult<f64> {
    let n = penalty.n();

    // Binary search for lambda that gives target df
    // Using log-scale search for faster convergence
    let mut log_lambda_low: f64 = -10.0;
    let mut log_lambda_high: f64 = 10.0;

    for _ in 0..30 {
        resource::checkpoint()?;
        let log_lambda_mid = (log_lambda_low + log_lambda_high) / 2.0;
        let lambda_mid = 10.0_f64.powf(log_lambda_mid);
        let (_, df, _) = fit_smoothing_spline(penalty, &vec![0.0; n], weights, lambda_mid)?;

        if (df - target_df).abs() < 0.05 {
            return Ok(lambda_mid);
        }

        if df > target_df {
            log_lambda_low = log_lambda_mid;
        } else {
            log_lambda_high = log_lambda_mid;
        }
    }

    Ok(10.0_f64.powf((log_lambda_low + log_lambda_high) / 2.0))
}

/// Cross-validation to select lambda.
///
/// Uses a coarse-to-fine search strategy for efficiency.
fn cross_validate(
    penalty: &PentaPenalty,
    y: &[f64],
    weights: &[f64],
    gcv: bool,
) -> EconResult<(f64, Option<f64>)> {
    // Coarse search first (fewer iterations)
    let mut best_lambda = 1.0;
    let mut best_score = f64::INFINITY;

    // Coarse grid: 15 values spanning a wide range
    let coarse_lambdas: Vec<f64> = (0..15)
        .map(|i| 10.0_f64.powf(-6.0 + i as f64 * 1.0))
        .collect();

    for &lambda in &coarse_lambdas {
        resource::checkpoint()?;
        let score = compute_cv_score(penalty, y, weights, lambda, gcv)?;
        if score < best_score {
            best_score = score;
            best_lambda = lambda;
        }
    }

    // Fine search around best lambda (10 values)
    let log_best = best_lambda.log10();
    let fine_lambdas: Vec<f64> = (0..10)
        .map(|i| 10.0_f64.powf(log_best - 0.5 + i as f64 * 0.1))
        .collect();

    for &lambda in &fine_lambdas {
        resource::checkpoint()?;
        let score = compute_cv_score(penalty, y, weights, lambda, gcv)?;
        if score < best_score {
            best_score = score;
            best_lambda = lambda;
        }
    }

    Ok((best_lambda, Some(best_score)))
}

/// Compute CV score for a given lambda (helper for cross_validate).
fn compute_cv_score(
    penalty: &PentaPenalty,
    y: &[f64],
    weights: &[f64],
    lambda: f64,
    gcv: bool,
) -> EconResult<f64> {
    let n = penalty.n();
    let (fitted, df, leverage) = fit_smoothing_spline(penalty, y, weights, lambda)?;

    let score = if gcv {
        // GCV score: (1/n) Σ (y_i - f(x_i))² / (1 - df/n)²
        let rss: f64 = y
            .iter()
            .zip(fitted.iter())
            .zip(weights.iter())
            .map(|((&yi, &fi), &wi)| wi * (yi - fi).powi(2))
            .sum();
        let factor = 1.0 - df / n as f64;
        if factor > 0.0 {
            rss / (n as f64 * factor * factor)
        } else {
            f64::INFINITY
        }
    } else {
        // Leave-one-out CV: Σ (y_i - f_{-i}(x_i))² / (1 - h_ii)²
        let mut cv_sum = 0.0;
        for i in 0..n {
            let resid = y[i] - fitted[i];
            let factor = 1.0 - leverage[i];
            if factor > 0.0 {
                cv_sum += weights[i] * (resid / factor).powi(2);
            }
        }
        cv_sum / n as f64
    };

    Ok(score)
}

/// Fit smoothing spline with given lambda.
///
/// Uses banded matrix representation for O(n) complexity per solve.
/// The penalty matrix for natural cubic splines is pentadiagonal (bandwidth 2).
fn fit_smoothing_spline(
    penalty: &PentaPenalty,
    y: &[f64],
    weights: &[f64],
    lambda: f64,
) -> EconResult<(Vec<f64>, f64, Vec<f64>)> {
    let n = y.len();

    // Extract banded structure from penalty matrix
    // The penalty matrix R^T R has bandwidth 2 (pentadiagonal)
    let bandwidth = 2;

    // Build banded system matrix: W + λΩ in banded storage
    // band[k][i] = A[i][i+k-bandwidth] for k=0..2*bandwidth+1
    let band_width = 2 * bandwidth + 1;
    let mut band = vec![vec![0.0; n]; band_width];

    for i in 0..n {
        for k in 0..band_width {
            let j = i as i32 + k as i32 - bandwidth as i32;
            if j >= 0 && (j as usize) < n {
                let j = j as usize;
                band[k][i] = lambda * penalty.at(i, j);
                if i == j {
                    band[k][i] += weights[i];
                }
            }
        }
    }

    // Weighted y
    let wy: Vec<f64> = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| wi * yi)
        .collect();

    // Solve banded system using banded Cholesky - O(n * bandwidth²)
    let (fitted, leverage) = solve_banded_system(&band, &wy, weights, bandwidth)?;

    let df: f64 = leverage.iter().sum();

    Ok((fitted, df, leverage))
}

/// Solve symmetric positive definite banded system using banded Cholesky.
/// Returns solution and leverage (diagonal of smoother matrix).
fn solve_banded_system(
    band: &[Vec<f64>],
    b: &[f64],
    weights: &[f64],
    bandwidth: usize,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let n = b.len();
    let bw = bandwidth;

    // Banded Cholesky decomposition: A = L L^T
    // L is lower triangular with bandwidth = bw
    // Store L in banded format: l_band[k][i] = L[i][i-k] for k=0..bw+1
    let mut l_band = vec![vec![0.0; n]; bw + 1];

    for i in 0..n {
        // Diagonal element
        let mut sum = band[bw][i]; // A[i][i]
        for k in 1..=bw.min(i) {
            sum -= l_band[k][i] * l_band[k][i];
        }
        if sum <= 0.0 {
            l_band[0][i] = 1e-10_f64.sqrt();
        } else {
            l_band[0][i] = sum.sqrt();
        }

        // Off-diagonal elements
        for j in (i + 1)..n.min(i + bw + 1) {
            let k = j - i; // offset
            let mut sum = band[bw + k][i]; // A[j][i]
            for m in 1..=bw.min(i) {
                if k + m <= bw {
                    sum -= l_band[m][i] * l_band[k + m][j];
                }
            }
            l_band[k][j] = sum / l_band[0][i];
        }
    }

    // Forward substitution: L y = b
    let mut y_temp = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 1..=bw.min(i) {
            sum -= l_band[k][i] * y_temp[i - k];
        }
        y_temp[i] = sum / l_band[0][i];
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y_temp[i];
        for k in 1..=bw.min(n - 1 - i) {
            sum -= l_band[k][i + k] * x[i + k];
        }
        x[i] = sum / l_band[0][i];
    }

    // Compute leverage using efficient banded inverse diagonal
    // h_ii = w_i * (A^{-1})_{ii}
    // For banded matrices, we can compute diagonal of inverse efficiently
    let mut leverage = vec![0.0; n];

    // Compute diagonal of L^{-1} L^{-T} = A^{-1}
    // Use the fact that (L^{-1})_{ii} = 1/L_{ii}
    // and contributions from nearby elements
    for i in 0..n {
        let mut inv_diag = 1.0 / (l_band[0][i] * l_band[0][i]);

        // Add contributions from nearby elements (approximate but fast)
        for k in 1..=bw.min(n - 1 - i) {
            let contrib = l_band[k][i + k] / l_band[0][i + k];
            inv_diag += contrib * contrib / (l_band[0][i] * l_band[0][i]);
        }

        leverage[i] = (weights[i] * inv_diag).min(1.0).max(0.0);
    }

    Ok((x, leverage))
}

/// Compute residuals at original data points.
fn compute_residuals_at_original(
    x_orig: &[f64],
    y_orig: &[f64],
    x_fitted: &[f64],
    y_fitted: &[f64],
) -> Vec<f64> {
    // Interpolate fitted values at original x points
    let coeffs = compute_spline_coefficients(x_fitted, y_fitted)
        .unwrap_or_else(|_| vec![0.0; x_fitted.len()]);

    x_orig
        .iter()
        .zip(y_orig.iter())
        .map(|(&xi, &yi)| yi - evaluate_spline(x_fitted, y_fitted, &coeffs, xi))
        .collect()
}

/// Compute natural cubic spline coefficients (second derivatives).
fn compute_spline_coefficients(x: &[f64], y: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();
    if n < 2 {
        return Ok(vec![0.0; n]);
    }

    // Natural spline: M_0 = M_{n-1} = 0
    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x[i + 1] - x[i]);
    }

    // Build tridiagonal system
    let mut diag = vec![1.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut d = vec![0.0; n];

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Thomas algorithm
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = d[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - sub[i] * c_prime[i - 1];
        c_prime[i] = sup[i] / denom;
        d_prime[i] = (d[i] - sub[i] * d_prime[i - 1]) / denom;
    }

    let mut m = vec![0.0; n];
    m[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        m[i] = d_prime[i] - c_prime[i] * m[i + 1];
    }

    Ok(m)
}

/// Evaluate cubic spline at a point.
fn evaluate_spline(x: &[f64], y: &[f64], m: &[f64], xi: f64) -> f64 {
    let n = x.len();
    if n < 2 {
        return if n == 1 { y[0] } else { f64::NAN };
    }

    // Find the interval by binary search over the sorted knots: the first i with
    // x[i + 1] >= xi, clamped to the last interval. This is the interval a
    // linear scan picks, also for knots, out-of-range points and NaN (interval 0).
    let i = x[1..].partition_point(|&v| v < xi).min(n - 2);

    let h = x[i + 1] - x[i];
    let a = (x[i + 1] - xi) / h;
    let b = (xi - x[i]) / h;

    a * y[i] + b * y[i + 1] + ((a.powi(3) - a) * m[i] + (b.powi(3) - b) * m[i + 1]) * h * h / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// The pre-T0-WP14 `evaluate_spline` (linear interval scan), kept as the
    /// reference for the binary search.
    fn evaluate_spline_linear_scan(x: &[f64], y: &[f64], m: &[f64], xi: f64) -> f64 {
        let n = x.len();
        if n < 2 {
            return if n == 1 { y[0] } else { f64::NAN };
        }

        // Find interval
        let mut i = 0;
        for j in 0..n - 1 {
            if xi >= x[j] && xi <= x[j + 1] {
                i = j;
                break;
            }
        }
        if xi < x[0] {
            i = 0;
        } else if xi > x[n - 1] {
            i = n - 2;
        }

        let h = x[i + 1] - x[i];
        let a = (x[i + 1] - xi) / h;
        let b = (xi - x[i]) / h;

        a * y[i]
            + b * y[i + 1]
            + ((a.powi(3) - a) * m[i] + (b.powi(3) - b) * m[i + 1]) * h * h / 6.0
    }

    #[test]
    fn test_penalty_bands_match_dense_rtr() {
        let x = [0.0, 0.4, 1.0, 1.1, 2.5, 3.0, 4.2, 4.3, 6.0];
        let n = x.len();
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        // naive R'R, rows k = 1..n-2, same accumulation order
        let mut dense = vec![vec![0.0; n]; n];
        for k in 1..n - 1 {
            let r = [
                (k - 1, 1.0 / h[k - 1]),
                (k, -1.0 / h[k - 1] - 1.0 / h[k]),
                (k + 1, 1.0 / h[k]),
            ];
            for &(i, ri) in &r {
                for &(j, rj) in &r {
                    dense[i][j] += ri * rj;
                }
            }
        }
        let p = penalty_bands(&x).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(p.at(i, j), dense[i][j], "({i},{j})");
            }
        }
    }

    #[test]
    fn test_spline_interval_search_matches_linear_scan() {
        let x = [0.0, 0.5, 1.5, 2.0, 3.5, 5.0];
        let y = [1.0, 0.2, -0.4, 0.9, 1.7, 0.3];
        let m = compute_spline_coefficients(&x, &y).unwrap();
        for xi in [
            -1.0,
            0.0,
            0.25,
            0.5,
            1.5,
            1.7,
            2.0,
            4.99,
            5.0,
            7.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                evaluate_spline(&x, &y, &m, xi).to_bits(),
                evaluate_spline_linear_scan(&x, &y, &m, xi).to_bits(),
                "xi={xi}"
            );
        }
    }

    #[test]
    fn test_smooth_spline_stops_when_cancelled() {
        let x: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let e = resource::with_budget(cancelled(), || {
            smooth_spline(&x, &y, SmoothSplineConfig::with_cv(true))
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_smooth_spline_large_n_is_linear_memory() {
        // Post-fix guard. Do NOT run against the dense code: it allocates two
        // 80 GB n x n tables.
        let n = 100_000;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 1000.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 7919 % 1009) as f64 / 1009.0 - 0.5))
            .collect();
        let r = smooth_spline(&x, &y, SmoothSplineConfig::with_df(10.0)).unwrap();
        assert_eq!(r.n_obs, n);
    }

    #[test]
    fn test_smooth_spline_basic() {
        // Noisy sine curve
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| xi.sin() + 0.1 * (xi * 10.0).sin())
            .collect();

        let config = SmoothSplineConfig::with_df(10.0);
        let result = smooth_spline(&x, &y, config).unwrap();

        assert_eq!(result.n_obs, 50);
        assert!(result.df > 5.0 && result.df < 20.0);
        assert!(!result.residuals.is_empty());
    }

    #[test]
    fn test_smooth_spline_cv() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(2) + 0.1 * xi).collect();

        let config = SmoothSplineConfig::with_cv(true);
        let result = smooth_spline(&x, &y, config).unwrap();

        assert!(result.cv_crit.is_some());
        assert!(result.df > 2.0);
    }

    #[test]
    fn test_smooth_spline_too_few_points() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let result = smooth_spline(&x, &y, SmoothSplineConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_smooth_spline_predict() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(2)).collect();

        let result = smooth_spline(&x, &y, SmoothSplineConfig::with_df(5.0)).unwrap();

        let xnew = vec![5.5, 10.5, 15.5];
        let ynew = smooth_spline_predict(&result, &xnew).unwrap();

        // Should be close to x^2
        for (i, &xi) in xnew.iter().enumerate() {
            let expected = xi.powi(2);
            let diff = (ynew[i] - expected).abs();
            assert!(
                diff < 10.0,
                "At x={}, expected ~{}, got {}",
                xi,
                expected,
                ynew[i]
            );
        }
    }

    #[test]
    fn test_smooth_spline_high_df_interpolates() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0];

        // High df should be close to interpolation
        let config = SmoothSplineConfig::with_df(9.5);
        let result = smooth_spline(&x, &y, config).unwrap();

        // Fitted values should be close to original
        for (i, &yi) in y.iter().enumerate() {
            let diff = (result.y[i] - yi).abs();
            assert!(
                diff < 1.0,
                "At index {}, expected ~{}, got {}",
                i,
                yi,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_smooth_spline_low_df_smooths() {
        // Noisy data
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi + 0.5 * (if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();

        // Low df should smooth out the noise
        let config = SmoothSplineConfig::with_df(3.0);
        let result = smooth_spline(&x, &y, config).unwrap();

        // Fitted values should follow the trend, not the noise
        // The underlying trend is y = x
        let mut trend_match = 0;
        for (&xi, &fi) in x.iter().zip(result.y.iter()) {
            if (fi - xi).abs() < 1.0 {
                trend_match += 1;
            }
        }
        assert!(
            trend_match as f64 / x.len() as f64 > 0.8,
            "Low df should follow the linear trend"
        );
    }

    /// Validation test against R's smooth.spline()
    #[test]
    fn test_validate_smooth_spline_against_r() {
        // R code:
        // set.seed(42)
        // x <- 1:20
        // y <- sin(x/5) + rnorm(20, sd=0.1)
        // result <- smooth.spline(x, y, df=6)
        // result$df  # should be close to 6
        // result$lambda

        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| (xi / 5.0).sin()).collect();

        let config = SmoothSplineConfig::with_df(6.0);
        let result = smooth_spline(&x, &y, config).unwrap();

        // df should be close to requested
        assert!(
            (result.df - 6.0).abs() < 1.0,
            "Expected df ~6, got {}",
            result.df
        );

        // Residuals should be small for smooth data
        let max_resid = result.residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_resid < 0.5, "Residuals too large: max = {}", max_resid);
    }

    /// Pinned-value check. A wrong or empty `expected` prints the actual values to paste.
    fn assert_pinned(what: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{what}: pin these values: {actual:?}"
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "{what}[{i}]: got {a}, pinned {e}; all: {actual:?}"
            );
        }
    }

    /// Characterisation test recorded from the code before the T0-WP14 resource
    /// bounds: 120 points with ties (108 unique x values, so knots are
    /// subsampled). The refactor must not move these numbers.
    #[test]
    fn test_pinned_smooth_spline_outputs() {
        let x: Vec<f64> = (0..120)
            .map(|i| ((i as f64 * 0.37) % 11.0 * 10.0).round() / 10.0)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi.sin() + 0.2 * (13.7 * i as f64).sin())
            .collect();
        let mut v = Vec::new();
        for config in [
            SmoothSplineConfig::with_df(8.0),
            SmoothSplineConfig::with_cv(true),
            SmoothSplineConfig::with_cv(false),
            SmoothSplineConfig::with_spar(0.5),
        ] {
            let r = smooth_spline(&x, &y, config).unwrap();
            assert!(r.x.len() > 50 && r.x.len() < x.len());
            v.push(r.df);
            v.push(r.lambda);
            v.push(r.cv_crit.unwrap_or(-1.0));
            v.push(r.y[0]);
            v.push(r.y[r.y.len() / 2]);
            v.push(r.y[r.y.len() - 1]);
            v.push(r.residuals.iter().map(|e| e * e).sum::<f64>());
            v.extend(smooth_spline_predict(&r, &[-1.0, 0.5, 5.25, 12.0]).unwrap());
        }
        assert_pinned(
            "smooth spline",
            &v,
            &[
                7.959650339740675,
                0.31765311974565075,
                -1.0,
                0.039855882916652235,
                -0.741512866824391,
                -0.9894474775718114,
                2.208640342105104,
                -0.574606194806342,
                0.5178801934638836,
                -0.8798477928818665,
                -2.3711789722590013,
                3.385695443841225,
                1.2589254117941675,
                0.02009268620312995,
                0.07080868290266186,
                -0.7193031982347561,
                -1.0182308450943232,
                2.2761536407365695,
                -0.6673589267151805,
                0.5085642766694489,
                -0.8637937775405333,
                -1.8037473116242337,
                3.385695443841225,
                1.2589254117941675,
                0.02061709010005428,
                0.07080868290266186,
                -0.7193031982347561,
                -1.0182308450943232,
                2.2761536407365695,
                -0.6673589267151805,
                0.5085642766694489,
                -0.8637937775405333,
                -1.8037473116242337,
                0.3910712187506279,
                191.85614814814815,
                -1.0,
                0.7089811199425499,
                -0.2561670964031391,
                -0.7676214379156595,
                20.123882967936165,
                0.7986745269386725,
                0.6678437240664398,
                -0.32252981657746854,
                -1.4014086129325052,
            ],
        );
    }
}
