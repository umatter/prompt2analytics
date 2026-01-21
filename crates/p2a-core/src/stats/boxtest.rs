//! Box-Pierce and Ljung-Box Tests for Autocorrelation.
//!
//! These tests examine the null hypothesis of independence in a time series.
//! They are often called "portmanteau" tests and are commonly used to check
//! whether residuals from a fitted ARIMA model are white noise.
//!
//! # References
//!
//! - Box, G. E. P. & Pierce, D. A. (1970). "Distribution of residual correlations
//!   in autoregressive-integrated moving average time series models."
//!   *Journal of the American Statistical Association*, 65, 1509-1526.
//! - Ljung, G. M. & Box, G. E. P. (1978). "On a measure of lack of fit in
//!   time series models." *Biometrika*, 65, 297-303.
//! - R Core Team. `stats::Box.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/box.test.html>
//!
//! # Mathematical Background
//!
//! Let n = sample size, ρ̂(k) = sample autocorrelation at lag k, m = number of lags.
//!
//! **Box-Pierce statistic:**
//! ```text
//! Q_BP = n × Σₖ₌₁ᵐ ρ̂(k)²
//! ```
//!
//! **Ljung-Box statistic:**
//! ```text
//! Q_LB = n(n+2) × Σₖ₌₁ᵐ ρ̂(k)² / (n-k)
//! ```
//!
//! Under the null hypothesis of no autocorrelation, both statistics follow
//! a chi-squared distribution with `m - fitdf` degrees of freedom.
//!
//! # Important Notes
//!
//! - When applied to residuals from an ARMA(p, q) model, set `fitdf = p + q`
//!   for a better approximation to the null distribution.
//! - The Ljung-Box test has better finite-sample properties than Box-Pierce.
//! - Missing values are not allowed.

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::{chi_squared_p_value, SignificanceLevel};

/// Type of portmanteau test to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BoxTestType {
    /// Box-Pierce test: Q = n × Σ ρ̂(k)²
    BoxPierce,
    /// Ljung-Box test: Q = n(n+2) × Σ ρ̂(k)²/(n-k) (default, better finite-sample properties)
    #[default]
    LjungBox,
}

impl std::fmt::Display for BoxTestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoxTestType::BoxPierce => write!(f, "Box-Pierce"),
            BoxTestType::LjungBox => write!(f, "Ljung-Box"),
        }
    }
}

/// Result of Box-Pierce or Ljung-Box test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxTestResult {
    /// Type of test performed
    pub test_type: BoxTestType,
    /// Test statistic (X-squared)
    pub statistic: f64,
    /// Degrees of freedom (lag - fitdf)
    pub df: usize,
    /// P-value from chi-squared distribution
    pub p_value: f64,
    /// Significance level based on p-value
    pub significance: SignificanceLevel,
    /// Number of observations
    pub n_obs: usize,
    /// Number of lags used
    pub lag: usize,
    /// Degrees of freedom subtracted (for ARMA residuals)
    pub fitdf: usize,
    /// Sample autocorrelations used in the test
    pub autocorrelations: Vec<f64>,
    /// Series name (if from dataset)
    pub series_name: Option<String>,
}

impl std::fmt::Display for BoxTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} Test for Autocorrelation", self.test_type)?;
        writeln!(f, "===========================================")?;
        if let Some(ref name) = self.series_name {
            writeln!(f, "Series: {}", name)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "X-squared = {:.4}, df = {}, p-value = {:.4} {}",
            self.statistic,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;
        writeln!(f, "Observations: {}  |  Lags: {}  |  fitdf: {}",
            self.n_obs, self.lag, self.fitdf)?;
        writeln!(f)?;
        writeln!(f, "H₀: No autocorrelation up to lag {}", self.lag)?;
        writeln!(f, "H₁: Autocorrelation exists at one or more lags")?;
        if self.p_value < 0.05 {
            writeln!(f, "\nConclusion: Reject H₀ - significant autocorrelation detected.")?;
        } else {
            writeln!(f, "\nConclusion: Fail to reject H₀ - no significant autocorrelation.")?;
        }
        Ok(())
    }
}

/// Compute sample autocorrelations for lags 1 to lag_max.
///
/// Uses a hybrid approach:
/// - Direct computation: O(n × lag) - better for small lag
/// - FFT-based computation: O(n log n) - better for large lag
///
/// Formula: ρ̂(k) = γ̂(k) / γ̂(0)
/// where γ̂(k) = (1/n) Σ (x_{t+k} - x̄)(x_t - x̄)
fn compute_autocorrelations(x: &[f64], lag_max: usize) -> Vec<f64> {
    let n = x.len();

    // FFT is O(n log n) with large constants; direct is O(n × lag)
    // FFT wins when: lag > C × log2(n) where C ≈ 3-5 (empirically determined)
    // For typical Box.test usage (lag=10-20), direct method is almost always faster
    // FFT becomes beneficial for large lag values (e.g., lag > 50 for n > 10,000)
    let log2_n = (n as f64).log2();
    let fft_threshold = (3.0 * log2_n) as usize;

    if lag_max > fft_threshold && n > 1000 {
        compute_autocorrelations_fft(x, lag_max)
    } else {
        compute_autocorrelations_direct(x, lag_max)
    }
}

/// Direct computation of autocorrelations - O(n × lag)
/// Optimized: demean once, use ndarray for SIMD-accelerated dot products
fn compute_autocorrelations_direct(x: &[f64], lag_max: usize) -> Vec<f64> {
    use ndarray::{Array1, s};

    let n = x.len();

    // Convert to ndarray for SIMD operations
    let x_arr = Array1::from_vec(x.to_vec());
    let mean = x_arr.mean().unwrap_or(0.0);

    // Demean using vectorized subtraction
    let x_centered = &x_arr - mean;

    // Compute variance using SIMD dot product
    let var = x_centered.dot(&x_centered) / n as f64;

    if var <= 0.0 {
        return vec![0.0; lag_max];
    }

    let inv_n_var = 1.0 / (n as f64 * var);

    // Compute autocorrelations using sliced dot products (SIMD accelerated)
    (1..=lag_max)
        .map(|k| {
            let a = x_centered.slice(s![k..]);
            let b = x_centered.slice(s![..n - k]);
            a.dot(&b) * inv_n_var
        })
        .collect()
}

/// FFT-based computation of autocorrelations - O(n log n)
///
/// Uses the Wiener-Khinchin theorem: the autocorrelation is the inverse FFT
/// of the power spectral density (|FFT(x)|²).
///
/// Algorithm:
/// 1. Zero-pad x to length 2n (avoids circular correlation artifacts)
/// 2. Compute FFT of padded, centered data
/// 3. Compute power spectrum: |FFT|²
/// 4. Inverse FFT to get autocorrelation
/// 5. Normalize by variance
fn compute_autocorrelations_fft(x: &[f64], lag_max: usize) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;

    // Pad to next power of 2 >= 2n for efficient FFT and to avoid circular artifacts
    let fft_len = (2 * n).next_power_of_two();

    // Create zero-padded, centered complex array
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(fft_len);
    for &xi in x {
        data.push(Complex::new(xi - mean, 0.0));
    }
    // Zero padding
    data.resize(fft_len, Complex::new(0.0, 0.0));

    // Plan and execute forward FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut data);

    // Compute power spectrum: |FFT(x)|²
    for c in &mut data {
        *c = Complex::new(c.norm_sqr(), 0.0);
    }

    // Inverse FFT to get autocorrelation
    let ifft = planner.plan_fft_inverse(fft_len);
    ifft.process(&mut data);

    // Normalize: IFFT includes factor of fft_len, and we need to divide by n for autocov
    // The autocorrelation at lag 0 is the variance
    let scale = (fft_len * n) as f64;
    let var = data[0].re / scale;

    if var <= 0.0 {
        return vec![0.0; lag_max];
    }

    // Extract autocorrelations for lags 1 to lag_max and normalize by variance
    (1..=lag_max)
        .map(|k| data[k].re / (scale * var))
        .collect()
}

/// Perform Box-Pierce or Ljung-Box test on a time series.
///
/// # Arguments
///
/// * `x` - Numeric vector (time series data)
/// * `lag` - Number of autocorrelation lags to use (default: 1)
/// * `test_type` - Type of test: BoxPierce or LjungBox (default: LjungBox)
/// * `fitdf` - Number of degrees of freedom to subtract (for ARMA residuals, use p+q)
///
/// # Returns
///
/// `BoxTestResult` with the test statistic, p-value, and autocorrelations.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::boxtest::{box_test, BoxTestType};
///
/// let x = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0];
/// let result = box_test(&x, Some(5), BoxTestType::LjungBox, 0)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Box, G. E. P. & Pierce, D. A. (1970). JASA, 65, 1509-1526.
/// - Ljung, G. M. & Box, G. E. P. (1978). Biometrika, 65, 297-303.
pub fn box_test(
    x: &[f64],
    lag: Option<usize>,
    test_type: BoxTestType,
    fitdf: usize,
) -> EconResult<BoxTestResult> {
    let n = x.len();

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "Box test requires at least 2 observations".to_string(),
        });
    }

    // Default lag = 1 (matching R's default)
    let lag = lag.unwrap_or(1);

    if lag < 1 {
        return Err(EconError::InvalidSpecification {
            message: "lag must be at least 1".to_string(),
        });
    }

    if lag >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("lag ({}) must be less than sample size ({})", lag, n),
        });
    }

    if lag <= fitdf {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "lag ({}) must be greater than fitdf ({}) for valid degrees of freedom",
                lag, fitdf
            ),
        });
    }

    // Check for missing values
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(EconError::InvalidSpecification {
            message: "Missing or infinite values are not allowed".to_string(),
        });
    }

    // Compute sample autocorrelations
    let acf = compute_autocorrelations(x, lag);

    // Compute test statistic
    let statistic = match test_type {
        BoxTestType::BoxPierce => {
            // Q_BP = n × Σₖ₌₁ᵐ ρ̂(k)²
            let sum_sq: f64 = acf.iter().map(|r| r.powi(2)).sum();
            (n as f64) * sum_sq
        }
        BoxTestType::LjungBox => {
            // Q_LB = n(n+2) × Σₖ₌₁ᵐ ρ̂(k)² / (n-k)
            let factor = (n as f64) * (n as f64 + 2.0);
            let sum: f64 = acf
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let k = i + 1; // lag is 1-indexed
                    r.powi(2) / (n - k) as f64
                })
                .sum();
            factor * sum
        }
    };

    // Degrees of freedom
    let df = lag - fitdf;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(statistic, df as f64);

    Ok(BoxTestResult {
        test_type,
        statistic,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_obs: n,
        lag,
        fitdf,
        autocorrelations: acf,
        series_name: None,
    })
}

/// Perform Box-Pierce or Ljung-Box test from a Dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `lag` - Number of autocorrelation lags (optional, default: 1)
/// * `test_type` - Type of test (default: LjungBox)
/// * `fitdf` - Degrees of freedom adjustment for ARMA residuals
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::boxtest::{run_box_test, BoxTestType};
///
/// let result = run_box_test(&dataset, "residuals", Some(10), BoxTestType::LjungBox, 2)?;
/// println!("{}", result);
/// ```
pub fn run_box_test(
    dataset: &Dataset,
    column: &str,
    lag: Option<usize>,
    test_type: BoxTestType,
    fitdf: usize,
) -> EconResult<BoxTestResult> {
    use polars::prelude::*;

    let series = dataset
        .df()
        .column(column)
        .map_err(|_| EconError::ColumnNotFound {
            column: column.to_string(),
            available: dataset
                .df()
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;

    let values = series
        .cast(&DataType::Float64)
        .map_err(|_| EconError::NonNumericColumn {
            column: column.to_string(),
        })?;

    let ca = values.f64().map_err(|_| EconError::NonNumericColumn {
        column: column.to_string(),
    })?;

    let x: Vec<f64> = ca
        .into_iter()
        .map(|opt| {
            opt.ok_or_else(|| EconError::NullValues {
                column: column.to_string(),
                count: 1,
            })
        })
        .collect::<Result<Vec<f64>, EconError>>()?;

    let mut result = box_test(&x, lag, test_type, fitdf)?;
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
    fn test_box_test_basic() {
        // Simple test with some data
        let x: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 0).unwrap();

        assert_eq!(result.n_obs, 20);
        assert_eq!(result.lag, 5);
        assert_eq!(result.df, 5);
        assert_eq!(result.fitdf, 0);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.autocorrelations.len(), 5);
    }

    #[test]
    fn test_box_pierce_vs_ljung_box() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).cos()).collect();

        let bp = box_test(&x, Some(10), BoxTestType::BoxPierce, 0).unwrap();
        let lb = box_test(&x, Some(10), BoxTestType::LjungBox, 0).unwrap();

        // Ljung-Box should have a larger statistic for same data
        // (due to the (n+2)/(n-k) factor)
        assert!(lb.statistic >= bp.statistic,
            "Ljung-Box stat ({}) should be >= Box-Pierce stat ({})",
            lb.statistic, bp.statistic);

        // Both should have same df
        assert_eq!(bp.df, lb.df);
    }

    #[test]
    fn test_white_noise_like_data() {
        // Test with data that has low autocorrelation
        // Alternating pattern + noise: should have weak autocorrelation
        let x: Vec<f64> = (0..100)
            .map(|i| {
                let base = if i % 2 == 0 { 0.5 } else { -0.5 };
                let noise = (i as f64 * 0.1).sin() * 0.1;
                base + noise
            })
            .collect();

        let result = box_test(&x, Some(10), BoxTestType::LjungBox, 0).unwrap();

        // This pattern has some structure but should still be testable
        // We're mainly checking that the function runs correctly
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df, 10);
        assert_eq!(result.n_obs, 100);
    }

    #[test]
    fn test_ar1_process() {
        // AR(1) process: x_t = 0.9 * x_{t-1} + noise
        // Should show significant autocorrelation
        let mut x = vec![0.0f64; 100];
        let phi = 0.9;
        for i in 1..100 {
            // Deterministic "noise"
            let noise = ((i * 1103515245 + 12345) % (1 << 31)) as f64 / (1u64 << 32) as f64 - 0.25;
            x[i] = phi * x[i - 1] + noise * 0.1;
        }

        let result = box_test(&x, Some(10), BoxTestType::LjungBox, 0).unwrap();

        // AR(1) with phi=0.9 should have significant autocorrelation
        assert!(result.p_value < 0.05,
            "AR(1) process should show significant autocorrelation, got p={}", result.p_value);
    }

    #[test]
    fn test_fitdf_adjustment() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();

        // Test with fitdf = 0
        let result0 = box_test(&x, Some(10), BoxTestType::LjungBox, 0).unwrap();
        assert_eq!(result0.df, 10);

        // Test with fitdf = 2 (e.g., ARMA(1,1) residuals)
        let result2 = box_test(&x, Some(10), BoxTestType::LjungBox, 2).unwrap();
        assert_eq!(result2.df, 8);

        // Same statistic, but different p-values due to df change
        assert!(approx_eq(result0.statistic, result2.statistic, 1e-10));
        // For a highly significant test (linear trend has huge statistic),
        // both p-values will be essentially zero, so we just check they're both very small
        assert!(result0.p_value < 1e-10, "Expected very small p-value for linear trend");
        assert!(result2.p_value < 1e-10, "Expected very small p-value for linear trend");
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0];
        let result = box_test(&x, Some(1), BoxTestType::LjungBox, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lag_too_large() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 0);
        assert!(result.is_err()); // lag must be < n
    }

    #[test]
    fn test_fitdf_too_large() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 5);
        assert!(result.is_err()); // fitdf must be < lag
    }

    #[test]
    fn test_display() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 0).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Ljung-Box Test"));
        assert!(display.contains("X-squared"));
        assert!(display.contains("df"));
        assert!(display.contains("p-value"));
    }

    /// Validation test against R
    ///
    /// R code:
    /// ```r
    /// x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    /// Box.test(x, lag = 5, type = "Ljung-Box")
    /// # X-squared = 11.175, df = 5, p-value = 0.04801
    ///
    /// Box.test(x, lag = 5, type = "Box-Pierce")
    /// # X-squared = 7.5444, df = 5, p-value = 0.1832
    /// ```
    #[test]
    fn test_validate_ljung_box_against_r() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 0).unwrap();

        // R: X-squared = 11.175, df = 5, p-value = 0.04801
        assert!(
            approx_eq(result.statistic, 11.175, 0.01),
            "Ljung-Box statistic should be ~11.175, got {}",
            result.statistic
        );
        assert_eq!(result.df, 5);
        assert!(
            approx_eq(result.p_value, 0.04801, 0.001),
            "p-value should be ~0.04801, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_box_pierce_against_r() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = box_test(&x, Some(5), BoxTestType::BoxPierce, 0).unwrap();

        // R: X-squared = 7.5444, df = 5, p-value = 0.1832
        assert!(
            approx_eq(result.statistic, 7.5444, 0.01),
            "Box-Pierce statistic should be ~7.5444, got {}",
            result.statistic
        );
        assert_eq!(result.df, 5);
        assert!(
            approx_eq(result.p_value, 0.1832, 0.001),
            "p-value should be ~0.1832, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_ljung_box_with_fitdf_against_r() {
        // R code:
        // x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        // Box.test(x, lag = 5, type = "Ljung-Box", fitdf = 2)
        // X-squared = 11.175, df = 3, p-value = 0.01080

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = box_test(&x, Some(5), BoxTestType::LjungBox, 2).unwrap();

        assert!(approx_eq(result.statistic, 11.175, 0.01));
        assert_eq!(result.df, 3); // 5 - 2 = 3
        assert!(approx_eq(result.p_value, 0.0108, 0.001));
    }

    #[test]
    fn test_from_dataset() {
        let df = df! {
            "values" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_box_test(&dataset, "values", Some(5), BoxTestType::LjungBox, 0).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.series_name.as_deref(), Some("values"));
        assert!(approx_eq(result.statistic, 11.175, 0.01));
    }

    /// Comprehensive validation test with multiple scenarios
    #[test]
    fn test_validate_comprehensive_against_r() {
        // Test: Linear trend (highly autocorrelated)
        // R: x <- 1:30
        // Box.test(x, lag=10, type="Ljung-Box")
        // X-squared = 104.83, df = 10, p-value < 2.2e-16

        let trend: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let result = box_test(&trend, Some(10), BoxTestType::LjungBox, 0).unwrap();

        assert!(
            approx_eq(result.statistic, 104.83, 0.1),
            "Linear trend statistic should be ~104.83, got {}",
            result.statistic
        );
        assert!(
            result.p_value < 1e-10,
            "Linear trend should have very low p-value, got {}",
            result.p_value
        );
    }
}
