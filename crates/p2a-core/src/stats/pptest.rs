//! Phillips-Perron Unit Root Test.
//!
//! Tests the null hypothesis that a time series has a unit root (is non-stationary)
//! against the alternative that it is stationary. The Phillips-Perron test is similar
//! to the Augmented Dickey-Fuller test but makes a non-parametric correction to the
//! t-statistic to account for serial correlation.
//!
//! # References
//!
//! - Phillips, P. C. B. & Perron, P. (1988). "Testing for a Unit Root in Time Series
//!   Regression." *Biometrika*, 75(2), 335-346.
//! - Banerjee, A., Dolado, J. J., Galbraith, J. W., & Hendry, D. (1993).
//!   *Co-integration, Error Correction, and the Econometric Analysis of Non-Stationary Data*.
//!   Oxford University Press.
//! - R Core Team. `stats::PP.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/PP.test.html>
//!
//! # Mathematical Background
//!
//! The test uses the regression model:
//! ```text
//! Δyₜ = α + βt + γyₜ₋₁ + uₜ
//! ```
//!
//! The statistic is R's `PP.test` "Dickey-Fuller" statistic, the corrected t statistic Z(τ):
//! ```text
//! Z(τ) = τ̂ × √(σ̂²/λ²) - n³ / (4 √3 √D √λ²) × (λ² - σ̂²)
//! ```
//!
//! Where, with n = length(x) - 1:
//! - τ̂ = t-statistic of γ̂ from the OLS regression
//! - σ̂² = residual sum of squares / n
//! - λ² = Newey-West long-run variance estimate
//! - D = det(X'X) for X = [1, t, yₜ₋₁], t = 1..n, computed from centred sums
//!
//! The Newey-West estimator for long-run variance:
//! ```text
//! λ² = σ̂² + 2 × Σⱼ₌₁ᵐ wⱼ × γ̂ⱼ
//! ```
//!
//! With Bartlett weights: wⱼ = 1 - j/(m+1)
//!
//! # Critical Values
//!
//! P-values are interpolated as R's `PP.test` does (Banerjee et al. 1993, Table 4.2 as tabulated
//! in R), clamped to [0.01, 0.99].

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;

/// Result of Phillips-Perron unit root test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPTestResult {
    /// PP.test's "Dickey-Fuller" statistic: the corrected t statistic Z(t_alpha) (R does not
    /// return Z(alpha))
    pub statistic: f64,
    /// Truncation lag: trunc(4 ((n - 1)/100)^0.25), or 12 with lshort = false, as PP.test
    pub truncation_lag: usize,
    /// P-value from R's table, interpolated and clamped to [0.01, 0.99]
    pub p_value: f64,
    /// Significance level based on p-value
    pub significance: SignificanceLevel,
    /// Number of observations
    pub n_obs: usize,
    /// Whether short lag was used (lshort parameter)
    pub lshort: bool,
    /// Series name (if from dataset)
    pub series_name: Option<String>,
    /// OLS coefficient estimate for γ (rho - 1)
    pub gamma_hat: f64,
    /// OLS t-statistic before correction
    pub t_statistic: f64,
    /// Residual variance σ̂²
    pub sigma_squared: f64,
    /// Long-run variance λ²
    pub lambda_squared: f64,
}

impl std::fmt::Display for PPTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Phillips-Perron Unit Root Test")?;
        writeln!(f, "========================================")?;
        if let Some(ref name) = self.series_name {
            writeln!(f, "Series: {}", name)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Dickey-Fuller Z(τ) = {:.4}, Truncation lag = {}, p-value = {:.4} {}",
            self.statistic,
            self.truncation_lag,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Observations: {}  |  lshort: {}",
            self.n_obs, self.lshort
        )?;
        writeln!(f)?;
        writeln!(f, "H₀: The series has a unit root (non-stationary)")?;
        writeln!(f, "H₁: The series is stationary")?;
        if self.p_value < 0.05 {
            writeln!(f, "\nConclusion: Reject H₀ - evidence of stationarity.")?;
        } else {
            writeln!(
                f,
                "\nConclusion: Fail to reject H₀ - series may have a unit root."
            )?;
        }
        Ok(())
    }
}

/// Compute truncation lag parameter.
///
/// - lshort=true: trunc(4*(n/100)^0.25)
/// - lshort=false: trunc(12*(n/100)^0.25)
fn compute_truncation_lag(n: usize, lshort: bool) -> usize {
    let base = if lshort { 4.0 } else { 12.0 };
    let lag = base * (n as f64 / 100.0).powf(0.25);
    lag.floor() as usize
}

/// Compute Newey-West long-run variance estimate using Bartlett weights.
///
/// λ² = σ̂² + 2 × Σⱼ₌₁ᵐ wⱼ × γ̂ⱼ
///
/// where wⱼ = 1 - j/(m+1) (Bartlett weights)
/// and γ̂ⱼ = (1/n) × Σₜ (uₜ × uₜ₋ⱼ) (autocovariance at lag j)
fn newey_west_variance(residuals: &[f64], truncation_lag: usize) -> f64 {
    let n = residuals.len();

    // Compute residual variance: σ̂² = (1/n) × Σ uₜ²
    let sigma_sq: f64 = residuals.iter().map(|u| u * u).sum::<f64>() / n as f64;

    if truncation_lag == 0 {
        return sigma_sq;
    }

    // Compute autocovariances γ̂ⱼ for j = 1, ..., m
    let mut gamma_sum = 0.0;
    for j in 1..=truncation_lag.min(n - 1) {
        // Autocovariance at lag j
        let gamma_j: f64 = (j..n).map(|t| residuals[t] * residuals[t - j]).sum::<f64>() / n as f64;

        // Bartlett weight
        let w_j = 1.0 - j as f64 / (truncation_lag + 1) as f64;

        gamma_sum += w_j * gamma_j;
    }

    // λ² = σ̂² + 2 × Σ wⱼ × γ̂ⱼ
    sigma_sq + 2.0 * gamma_sum
}

/// The interval of ascending `xs` holding `x` and the weight of its upper end, clamped to the ends
/// as R's `approx(rule = 2)`.
fn bracket(xs: &[f64], x: f64) -> (usize, usize, f64) {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return (0, 0, 0.0);
    }
    if x >= xs[last] {
        return (last, last, 0.0);
    }
    let hi = xs.partition_point(|&v| v < x);
    (hi - 1, hi, (x - xs[hi - 1]) / (xs[hi] - xs[hi - 1]))
}

/// Linear interpolation of `(xs, ys)` at a bracket.
fn at_bracket(ys: &[f64], (lo, hi, w): (usize, usize, f64)) -> f64 {
    ys[lo] + w * (ys[hi] - ys[lo])
}

/// The p-value of `PP.test`: R's table (Banerjee et al. 1993, Table 4.2) is interpolated in the
/// sample size `m` (the regression's) for each p, and then p is interpolated in the statistic.
/// Both interpolations clamp at the table's ends. NaN stays NaN.
fn pp_p_value(stat: f64, m: usize) -> f64 {
    const TABLE_T: [f64; 6] = [25.0, 50.0, 100.0, 250.0, 500.0, 100000.0];
    const TABLE_P: [f64; 8] = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99];
    // One row per p, one column per T
    const TABLE: [[f64; 6]; 8] = [
        [-4.38, -4.15, -4.04, -3.99, -3.98, -3.96],
        [-3.95, -3.80, -3.73, -3.69, -3.68, -3.66],
        [-3.60, -3.50, -3.45, -3.43, -3.42, -3.41],
        [-3.24, -3.18, -3.15, -3.13, -3.13, -3.12],
        [-1.14, -1.19, -1.22, -1.23, -1.24, -1.25],
        [-0.80, -0.87, -0.90, -0.92, -0.93, -0.94],
        [-0.50, -0.58, -0.62, -0.64, -0.65, -0.66],
        [-0.15, -0.24, -0.28, -0.31, -0.32, -0.33],
    ];
    if stat.is_nan() {
        return f64::NAN;
    }
    let in_t = bracket(&TABLE_T, m as f64);
    let critical = TABLE.map(|row| at_bracket(&row, in_t));
    at_bracket(&TABLE_P, bracket(&critical, stat))
}

/// Perform the Phillips-Perron unit root test on a time series.
///
/// # Arguments
///
/// * `x` - Numeric vector (time series data)
/// * `lshort` - If true, use short truncation lag (default: true)
///   - lshort=true: trunc(4*((n-1)/100)^0.25)
///   - lshort=false: trunc(12*((n-1)/100)^0.25)
///
/// # Returns
///
/// `PPTestResult` with the Z(τ) test statistic, p-value, and additional diagnostics.
///
/// # Errors
///
/// Fewer than 5 observations (the regression needs a positive residual df), non-finite values,
/// a singular regression, and an exact fit, whose long-run variance is zero (`PP.test` returns NaN).
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::pptest::pp_test;
///
/// // Test random walk (should have unit root)
/// let random_walk: Vec<f64> = (0..100).scan(0.0, |state, i| {
///     *state += (i as f64 * 0.1).sin();
///     Some(*state)
/// }).collect();
///
/// let result = pp_test(&random_walk, true)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Phillips, P. C. B. & Perron, P. (1988). Biometrika, 75(2), 335-346.
pub fn pp_test(x: &[f64], lshort: bool) -> EconResult<PPTestResult> {
    let n = x.len();

    if n < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n,
            context: "Phillips-Perron: the regression on [1, t, y_{t-1}] needs a positive \
                      residual df (n - 4)"
                .to_string(),
        });
    }

    // Check for missing values
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(EconError::InvalidSpecification {
            message: "Missing or infinite values are not allowed".to_string(),
        });
    }

    let n_reg = n - 1; // Number of observations in regression

    // Compute truncation lag (R: n is length(x) - 1 here)
    let truncation_lag = compute_truncation_lag(n_reg, lshort);

    // Build design matrix for the regression:
    // Δyₜ = α + β*t + γ*yₜ₋₁ + uₜ
    //
    // We regress Δy on [1, t, y_{t-1}]

    // Build vectors for regression
    let mut delta_y = Vec::with_capacity(n_reg);
    let mut y_lag = Vec::with_capacity(n_reg);
    let mut trend = Vec::with_capacity(n_reg);

    for t in 1..n {
        delta_y.push(x[t] - x[t - 1]);
        y_lag.push(x[t - 1]);
        trend.push(t as f64);
    }

    // OLS regression: Δy = α + β*t + γ*y_{t-1} + u
    // Design matrix X = [1, t, y_{t-1}]

    // Compute X'X and X'y using explicit formulas for 3x3 case
    // X = [1, t, y_lag]
    let sum_1: f64 = n_reg as f64;
    let (mut sum_t, mut sum_y, mut sum_t2, mut sum_ty, mut sum_y2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sum_dy, mut sum_t_dy, mut sum_y_dy) = (0.0, 0.0, 0.0);
    for ((&t, &y), &dy) in trend.iter().zip(&y_lag).zip(&delta_y) {
        sum_t += t;
        sum_y += y;
        sum_t2 += t * t;
        sum_ty += t * y;
        sum_y2 += y * y;
        sum_dy += dy;
        sum_t_dy += t * dy;
        sum_y_dy += y * dy;
    }

    // X'X matrix (symmetric 3x3)
    // [ n,       sum_t,    sum_y   ]
    // [ sum_t,   sum_t2,   sum_ty  ]
    // [ sum_y,   sum_ty,   sum_y2  ]

    // X'y vector
    // [ sum_dy   ]
    // [ sum_t_dy ]
    // [ sum_y_dy ]

    // Solve using Cramer's rule or direct inverse
    // For robustness, compute determinant and check for singularity
    let det = sum_1 * (sum_t2 * sum_y2 - sum_ty * sum_ty)
        - sum_t * (sum_t * sum_y2 - sum_ty * sum_y)
        + sum_y * (sum_t * sum_ty - sum_t2 * sum_y);

    if det.abs() < 1e-14 {
        return Err(EconError::SingularMatrix {
            context: "X'X in Phillips-Perron regression".to_string(),
            suggestion: "Check for multicollinearity or constant series".to_string(),
        });
    }

    // Compute inverse of X'X (cofactor matrix / det)
    let inv_det = 1.0 / det;

    // Cofactors for row 3 (needed for γ coefficient and its variance)
    let c31 = sum_t * sum_ty - sum_t2 * sum_y;
    let c32 = sum_t * sum_y - sum_1 * sum_ty;
    let c33 = sum_1 * sum_t2 - sum_t * sum_t;

    // Compute γ̂ (coefficient on y_{t-1})
    // γ̂ = [row 3 of (X'X)^{-1}] · [X'y]
    let gamma_hat = inv_det * (c31 * sum_dy + c32 * sum_t_dy + c33 * sum_y_dy);

    // Compute all coefficients for residuals
    let c11 = sum_t2 * sum_y2 - sum_ty * sum_ty;
    let c12 = sum_ty * sum_y - sum_t * sum_y2;
    let c13 = sum_t * sum_ty - sum_t2 * sum_y;
    let c21 = c12;
    let c22 = sum_1 * sum_y2 - sum_y * sum_y;
    let c23 = sum_t * sum_y - sum_1 * sum_ty;

    let alpha_hat = inv_det * (c11 * sum_dy + c12 * sum_t_dy + c13 * sum_y_dy);
    let beta_hat = inv_det * (c21 * sum_dy + c22 * sum_t_dy + c23 * sum_y_dy);

    // Compute residuals
    let residuals: Vec<f64> = (0..n_reg)
        .map(|i| delta_y[i] - alpha_hat - beta_hat * trend[i] - gamma_hat * y_lag[i])
        .collect();

    // Residual sum of squares
    let rss: f64 = residuals.iter().map(|r| r * r).sum();

    // Degrees of freedom
    let df = n_reg - 3; // n - 1 observations, 3 parameters; positive since n >= 5

    // Residual variance σ̂² (different from Newey-West divisor)
    let sigma_sq = rss / df as f64;

    // Standard error of γ̂
    // Var(γ̂) = σ̂² × [(X'X)^{-1}]_{33}
    let var_gamma = sigma_sq * inv_det * c33;
    let se_gamma = var_gamma.sqrt();

    // OLS t-statistic for γ
    let t_stat = gamma_hat / se_gamma;

    // Compute Newey-West long-run variance
    let lambda_sq = newey_west_variance(&residuals, truncation_lag);

    // Residual variance for PP correction (using n divisor, not df)
    let sigma_sq_nw = rss / n_reg as f64;

    if lambda_sq.is_nan() || lambda_sq <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Phillips-Perron: the long-run variance is zero (the regression on \
                      [1, t, y_{t-1}] fits exactly)"
                .to_string(),
        });
    }

    // det(X'X) from centred sums. R's four-term expression gives the same number but cancels
    // terms of order m^4 sum(y^2).
    let m = n_reg as f64;
    let t_bar = sum_t / m;
    let y_bar = sum_y / m;
    let s_tt = m * (m * m - 1.0) / 12.0;
    let (mut s_yy, mut s_ty) = (0.0, 0.0);
    for (t, y) in trend.iter().zip(&y_lag) {
        let dev = y - y_bar;
        s_yy += dev * dev;
        s_ty += (t - t_bar) * dev;
    }
    let dx = m * (s_tt * s_yy - s_ty * s_ty);

    // PP.test's statistic, with the sign of λ² - σ̂² kept
    let z_tau = (sigma_sq_nw / lambda_sq).sqrt() * t_stat
        - m.powi(3) / (4.0 * 3f64.sqrt() * dx.sqrt() * lambda_sq.sqrt())
            * (lambda_sq - sigma_sq_nw);

    let p_value = pp_p_value(z_tau, n_reg);

    Ok(PPTestResult {
        statistic: z_tau,
        truncation_lag,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_obs: n,
        lshort,
        series_name: None,
        gamma_hat,
        t_statistic: t_stat,
        sigma_squared: sigma_sq_nw,
        lambda_squared: lambda_sq,
    })
}

/// Perform Phillips-Perron test from a Dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `lshort` - If true, use short truncation lag (default: true)
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::pptest::run_pp_test;
///
/// let result = run_pp_test(&dataset, "gdp_growth", true)?;
/// println!("{}", result);
/// ```
pub fn run_pp_test(dataset: &Dataset, column: &str, lshort: bool) -> EconResult<PPTestResult> {
    let (x, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    let mut result = pp_test(&x, lshort)?;
    result.series_name = Some(column.to_string());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_truncation_lag_short() {
        // R: trunc(4*(100/100)^0.25) = trunc(4*1) = 4
        assert_eq!(compute_truncation_lag(100, true), 4);

        // R: trunc(4*(1000/100)^0.25) = trunc(4*1.778) = 7
        assert_eq!(compute_truncation_lag(1000, true), 7);

        // R: trunc(4*(50/100)^0.25) = trunc(4*0.841) = 3
        assert_eq!(compute_truncation_lag(50, true), 3);
    }

    #[test]
    fn test_truncation_lag_long() {
        // R: trunc(12*(100/100)^0.25) = trunc(12*1) = 12
        assert_eq!(compute_truncation_lag(100, false), 12);

        // R: trunc(12*(1000/100)^0.25) = trunc(12*1.778) = 21
        assert_eq!(compute_truncation_lag(1000, false), 21);
    }

    #[test]
    fn test_pp_test_basic() {
        // Simple test with linear trend (should reject unit root)
        let x: Vec<f64> = (1..=100)
            .map(|i| i as f64 + (i as f64 * 0.1).sin())
            .collect();
        let result = pp_test(&x, true).unwrap();

        assert_eq!(result.n_obs, 100);
        assert!(result.truncation_lag > 0);
        assert!(!result.statistic.is_nan());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_stationary_series() {
        // Stationary series: AR(1) with phi < 1
        // x_t = 0.5 * x_{t-1} + noise
        let mut x = vec![0.0f64; 200];
        let phi = 0.5;
        for i in 1..200 {
            let noise = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (1 << 16))
                as f64
                / (1u64 << 17) as f64
                - 0.25;
            x[i] = phi * x[i - 1] + noise;
        }

        let result = pp_test(&x, true).unwrap();

        // Stationary series should have very negative statistic (reject unit root)
        assert!(
            result.statistic < -2.0,
            "Stationary AR(1) should have negative test statistic, got {}",
            result.statistic
        );
    }

    #[test]
    fn test_random_walk() {
        // Random walk (unit root): x_t = x_{t-1} + noise
        let mut x = vec![0.0f64; 200];
        for i in 1..200 {
            let noise = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (1 << 16))
                as f64
                / (1u64 << 17) as f64
                - 0.25;
            x[i] = x[i - 1] + noise * 0.1;
        }

        let result = pp_test(&x, true).unwrap();

        // Random walk typically fails to reject unit root
        // The statistic should be less negative (closer to zero or positive)
        // Note: This is probabilistic, so we use a loose bound
        assert!(!result.statistic.is_nan(), "Test statistic should be valid");
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0, 2.0, 3.0];
        let result = pp_test(&x, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_nan() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let result = pp_test(&x, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        // Use non-linear data to avoid singularity (linear trend + noise)
        let x: Vec<f64> = (1..=50)
            .map(|i| {
                let noise = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (1 << 16))
                    as f64
                    / (1u64 << 17) as f64
                    - 0.25;
                i as f64 + noise * 2.0
            })
            .collect();
        let result = pp_test(&x, true).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Phillips-Perron"));
        assert!(display.contains("Z(τ)"));
        assert!(display.contains("Truncation lag"));
        assert!(display.contains("p-value"));
    }

    /// Test lshort parameter effect
    #[test]
    fn test_lshort_parameter() {
        // Use non-linear data to avoid singularity
        let x: Vec<f64> = (1..=100)
            .map(|i| {
                let noise = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (1 << 16))
                    as f64
                    / (1u64 << 17) as f64
                    - 0.25;
                i as f64 + noise * 5.0
            })
            .collect();

        let result_short = pp_test(&x, true).unwrap();
        let result_long = pp_test(&x, false).unwrap();

        // Long truncation should use more lags
        assert!(
            result_long.truncation_lag > result_short.truncation_lag,
            "Long truncation ({}) should be > short ({})",
            result_long.truncation_lag,
            result_short.truncation_lag
        );

        // Both should be valid
        assert!(!result_short.statistic.is_nan());
        assert!(!result_long.statistic.is_nan());
    }

    #[test]
    fn test_from_dataset() {
        // Use non-linear data to avoid singularity
        let values: Vec<f64> = (1..=100)
            .map(|i| {
                let noise = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (1 << 16))
                    as f64
                    / (1u64 << 17) as f64
                    - 0.25;
                i as f64 + noise * 5.0
            })
            .collect();

        let df = df! {
            "values" => values
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_pp_test(&dataset, "values", true).unwrap();

        assert_eq!(result.n_obs, 100);
        assert_eq!(result.series_name.as_deref(), Some("values"));
    }

    #[test]
    fn test_column_not_found() {
        let df = df! {
            "values" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_pp_test(&dataset, "nonexistent", true);
        assert!(result.is_err());
    }

    /// STAT-67 (PP): the regression and Newey-West internals of PP.test, pinned where both lag rules agree (n = 200).
    #[test]
    fn test_validate_pp_test_t3_internals() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_pp_t3");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Omit).unwrap().0;
        let mut c = Check::new(&case);
        for series in ["ar", "rw"] {
            let x = col(series);
            for (tag, lshort) in [("short", true), ("long", false)] {
                let r = pp_test(&x, lshort).unwrap();
                let k = |q: &str| format!("{series}_{tag}_{q}");
                c.scalar(&k("gamma"), r.gamma_hat, Tol::coef(200))
                    .scalar(&k("t"), r.t_statistic, Tol::stat(200))
                    .scalar(&k("sigma2"), r.sigma_squared, Tol::stat(200))
                    .scalar(&k("lambda2"), r.lambda_squared, Tol::stat(200))
                    .scalar(&k("lag"), r.truncation_lag as f64, Tol::exact());
                // {series}_{tag}_stat and _p: pinned by Task 8A (STAT-6, STAT-35; sync 2026-11-01).
            }
        }
        let r = pp_test(&col("rw100"), true).unwrap();
        c.scalar("rw100_short_gamma", r.gamma_hat, Tol::coef(100))
            .scalar("rw100_short_t", r.t_statistic, Tol::stat(100))
            .scalar("rw100_short_sigma2", r.sigma_squared, Tol::stat(100))
            .finish();
        // rw100_short_lag (R 3 from n - 1 = 99) and rw100_short_lambda2: pinned by Task 8A (STAT-6).
    }

    /// R4: four observations leave 3 regression rows and 3 parameters, a residual df of 0 (sigma^2 = 0/0).
    #[test]
    fn test_pp_test_needs_positive_residual_df() {
        let e = pp_test(&[1.0, 2.5, 2.0, 3.7], true).unwrap_err();
        assert!(
            matches!(
                e,
                EconError::InsufficientData {
                    required: 5,
                    provided: 4,
                    ..
                }
            ),
            "{e}"
        );
        assert!(pp_test(&[1.0, 2.5, 2.0, 3.7, 3.1], true).is_ok());
    }

    /// STAT-6, STAT-35: PP.test's "Dickey-Fuller" statistic (the corrected t), its lag from n - 1 and its p-value
    /// from R's table, on the five fits of stat_pp_t3.
    #[test]
    fn test_validate_pp_test_t3_statistic() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_pp_t3");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Omit).unwrap().0;
        let mut c = Check::new(&case);
        for (series, tag, lshort, n) in [
            ("ar", "short", true, 200),
            ("ar", "long", false, 200),
            ("rw", "short", true, 200),
            ("rw", "long", false, 200),
            ("rw100", "short", true, 100),
        ] {
            let r = pp_test(&col(series), lshort).unwrap();
            let k = |q: &str| format!("{series}_{tag}_{q}");
            c.scalar(&k("stat"), r.statistic, Tol::stat(n))
                .scalar(&k("p"), r.p_value, Tol::p_value(n))
                .scalar(&k("lag"), r.truncation_lag as f64, Tol::exact())
                .scalar(&k("lambda2"), r.lambda_squared, Tol::stat(n));
        }
        c.finish();
    }

    /// STAT-6: the paper's series (examples.tex, Section 4.4). R: Dickey-Fuller -12.0184558378257, lag 5, p 0.01.
    #[test]
    fn test_validate_pp_test_t3_gdp() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_pp_gdp_t3");
        let short = run_pp_test(&ds, "gdp_growth", true).unwrap();
        let long = run_pp_test(&ds, "gdp_growth", false).unwrap();
        let n = short.n_obs;
        Check::new(&case)
            .scalar("gdp_short_stat", short.statistic, Tol::stat(n))
            .scalar("gdp_short_lag", short.truncation_lag as f64, Tol::exact())
            .scalar("gdp_short_p", short.p_value, Tol::p_value(n))
            .scalar("gdp_long_stat", long.statistic, Tol::stat(n))
            .scalar("gdp_long_lag", long.truncation_lag as f64, Tol::exact())
            .scalar("gdp_long_p", long.p_value, Tol::p_value(n))
            .finish();
    }

    /// STAT-6: the correction keeps its sign (p2a clipped it to 0 when lambda^2 <= sigma^2). ar_long has
    /// lambda2 1.1002914390310594 < sigma2 1.1623414925199784, so the statistic exceeds sqrt(ssqru/ssqrtl) t.
    #[test]
    fn test_pp_correction_keeps_sign() {
        use crate::data::columns::{NaPolicy, numeric_column};
        let path = p2a_validation::repo_path("validation/datasets/generated/pp_t3.csv");
        let ds = Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let x = numeric_column(ds.df(), "ar", NaPolicy::Omit).unwrap().0;
        let r = pp_test(&x, false).unwrap();
        assert!(r.lambda_squared < r.sigma_squared);
        assert!(r.statistic > (r.sigma_squared / r.lambda_squared).sqrt() * r.t_statistic);
    }

    /// STAT-35: R's table is interpolated in T, then in the statistic, both with approx(rule = 2).
    #[test]
    fn test_pp_p_value_clamps_and_is_monotone() {
        assert_eq!(pp_p_value(-50.0, 199), 0.01);
        assert_eq!(pp_p_value(5.0, 199), 0.99);
        assert_eq!(pp_p_value(-3.0, 10), pp_p_value(-3.0, 25)); // T below 25 uses the T = 25 row
        assert_eq!(pp_p_value(-3.0, 10_000_000), pp_p_value(-3.0, 100_000)); // T above 1e5 uses the last row
        let grid: Vec<f64> = (0..=60).map(|i| -5.0 + 0.1 * i as f64).collect();
        let p: Vec<f64> = grid.iter().map(|&s| pp_p_value(s, 199)).collect();
        assert!(p.windows(2).all(|w| w[0] <= w[1]));
        assert!(pp_p_value(f64::NAN, 199).is_nan());
    }
}
