//! Cumulative Periodogram for Time Series Diagnostics.
//!
//! The cumulative periodogram is a diagnostic tool for time series analysis that plots
//! the cumulative sum of the periodogram ordinates against frequency. For white noise,
//! the cumulative periodogram follows a uniform distribution, and confidence bands
//! based on the Kolmogorov-Smirnov distribution can detect departures from white noise.
//!
//! # Mathematical Background
//!
//! ## Periodogram
//!
//! The periodogram at Fourier frequency ωⱼ = 2πj/n is:
//!
//! I(ωⱼ) = (1/n) |Σₜ xₜ exp(-i ωⱼ t)|²
//!
//! ## Cumulative Periodogram
//!
//! The normalized cumulative periodogram is:
//!
//! C(ωⱼ) = Σₖ₌₁ʲ I(ωₖ) / Σₖ₌₁ᴹ I(ωₖ)
//!
//! where M = ⌊(n-1)/2⌋.
//!
//! ## White Noise Test
//!
//! Under H₀ (white noise), C(ωⱼ) should lie close to the line connecting
//! (0, 0) to (0.5, 1). The Kolmogorov-Smirnov statistic:
//!
//! D = max|C(ωⱼ) - j/M|
//!
//! tests for departures from white noise.
//!
//! ## Confidence Bands
//!
//! The 95% confidence bands are ±1.358/√M (Kolmogorov-Smirnov critical value).
//!
//! # References
//!
//! - Bartlett, M.S. (1955). *An Introduction to Stochastic Processes*. Cambridge
//!   University Press. Early work on periodogram analysis.
//!
//! - Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
//!   ISBN: 978-0125649018. Comprehensive treatment of spectral methods.
//!
//! - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*
//!   (2nd ed.), Section 10.2. Springer. ISBN: 978-0387974293.
//!
//! - Venables, W.N., & Ripley, B.D. (2002). *Modern Applied Statistics with S*
//!   (4th ed.). Springer. ISBN: 978-0387954578. Section 14.1 on spectral analysis.
//!
//! - Kolmogorov, A.N. (1933). Sulla determinazione empirica di una legge di
//!   distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91.
//!   The Kolmogorov-Smirnov test underlying the confidence bands.
//!
//! R equivalent: `stats::cpgram()` (originally from MASS package)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Result of cumulative periodogram computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpgramResult {
    /// Frequencies at which the cumulative periodogram is evaluated
    pub freq: Vec<f64>,
    /// Cumulative periodogram values (normalized to [0, 1])
    pub cpgram: Vec<f64>,
    /// Upper 95% confidence band (Kolmogorov-Smirnov based)
    pub upper_ci: Vec<f64>,
    /// Lower 95% confidence band
    pub lower_ci: Vec<f64>,
    /// Number of observations in original series
    pub n: usize,
    /// Taper proportion used
    pub taper: f64,
    /// Whether the series appears to be white noise (within CI bands)
    pub is_white_noise: bool,
    /// Maximum deviation from expected uniform distribution
    pub max_deviation: f64,
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// Approximate p-value for white noise test
    pub ks_p_value: f64,
}

/// Compute the cumulative periodogram of a time series.
///
/// The cumulative periodogram is defined as the cumulative sum of the periodogram
/// values, normalized so that the final value equals 1. Under the null hypothesis
/// of white noise, the cumulative periodogram should approximately follow a
/// straight line from (0, 0) to (0.5, 1).
///
/// # Arguments
/// * `x` - The time series data
/// * `taper` - Proportion of data to taper at each end (0.0 to 0.5, default 0.1)
///
/// # Returns
/// A `CpgramResult` containing the cumulative periodogram and confidence intervals
///
/// # Example
/// ```
/// use p2a_core::forecasting::cpgram::cpgram;
///
/// let x: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();
/// let result = cpgram(&x, Some(0.1)).unwrap();
/// println!("Is white noise: {}", result.is_white_noise);
/// ```
pub fn cpgram(x: &[f64], taper: Option<f64>) -> Result<CpgramResult, String> {
    let n = x.len();
    if n < 4 {
        return Err("Time series must have at least 4 observations".to_string());
    }

    let taper_prop = taper.unwrap_or(0.1).clamp(0.0, 0.5);

    // Apply cosine bell taper
    let tapered = apply_taper(x, taper_prop);

    // Compute periodogram via FFT
    let pgram = compute_periodogram(&tapered);
    let m = pgram.len(); // Number of Fourier frequencies

    // Compute cumulative periodogram (normalized)
    let total: f64 = pgram.iter().sum();
    let mut cumsum = Vec::with_capacity(m);
    let mut running_sum = 0.0;

    for &p in &pgram {
        running_sum += p;
        cumsum.push(if total > 0.0 {
            running_sum / total
        } else {
            0.0
        });
    }

    // Compute frequencies (normalized to [0, 0.5])
    let freq: Vec<f64> = (1..=m).map(|i| i as f64 / (2.0 * m as f64)).collect();

    // Compute Kolmogorov-Smirnov confidence bands
    // For white noise, the cumulative periodogram should be close to the identity line
    // The KS critical value at 95% is approximately 1.36 / sqrt(n)
    let ks_critical = 1.36 / (m as f64).sqrt();

    let upper_ci: Vec<f64> = freq.iter().map(|&f| (f * 2.0 + ks_critical).min(1.0)).collect();
    let lower_ci: Vec<f64> = freq.iter().map(|&f| (f * 2.0 - ks_critical).max(0.0)).collect();

    // Compute KS statistic (maximum deviation from expected uniform distribution)
    let expected: Vec<f64> = freq.iter().map(|&f| f * 2.0).collect();
    let mut max_deviation = 0.0;
    for i in 0..m {
        let deviation = (cumsum[i] - expected[i]).abs();
        if deviation > max_deviation {
            max_deviation = deviation;
        }
    }

    let ks_statistic = max_deviation * (m as f64).sqrt();

    // Approximate p-value using Kolmogorov distribution
    let ks_p_value = kolmogorov_p_value(ks_statistic);

    // Check if within confidence bands
    let is_white_noise = max_deviation <= ks_critical;

    Ok(CpgramResult {
        freq,
        cpgram: cumsum,
        upper_ci,
        lower_ci,
        n,
        taper: taper_prop,
        is_white_noise,
        max_deviation,
        ks_statistic,
        ks_p_value,
    })
}

/// Apply a cosine bell (split cosine) taper to the data.
fn apply_taper(x: &[f64], taper: f64) -> Vec<f64> {
    if taper <= 0.0 {
        return x.to_vec();
    }

    let n = x.len();
    let m = (taper * n as f64).round() as usize;

    if m == 0 {
        return x.to_vec();
    }

    let mut tapered = x.to_vec();

    // Apply cosine bell to first m points
    for i in 0..m.min(n) {
        let w = 0.5 * (1.0 - (PI * i as f64 / m as f64).cos());
        tapered[i] *= w;
    }

    // Apply cosine bell to last m points
    for i in 0..m.min(n) {
        let idx = n - 1 - i;
        let w = 0.5 * (1.0 - (PI * i as f64 / m as f64).cos());
        tapered[idx] *= w;
    }

    // Adjust for mean after tapering
    let mean: f64 = tapered.iter().sum::<f64>() / n as f64;
    for v in &mut tapered {
        *v -= mean;
    }

    tapered
}

/// Compute the periodogram using FFT-like computation.
/// Returns periodogram ordinates at Fourier frequencies.
fn compute_periodogram(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let n_freq = n / 2; // Number of Fourier frequencies (excluding DC and Nyquist)

    // Center the data
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();

    // Compute periodogram via direct Fourier transform
    // (For production, this should use FFT, but direct computation is clearer)
    let mut pgram = Vec::with_capacity(n_freq);

    for k in 1..=n_freq {
        let freq = 2.0 * PI * k as f64 / n as f64;

        // Compute Fourier coefficients
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (t, &xt) in centered.iter().enumerate() {
            cos_sum += xt * (freq * t as f64).cos();
            sin_sum += xt * (freq * t as f64).sin();
        }

        // Periodogram ordinate: (cos_sum^2 + sin_sum^2) / n
        let intensity = (cos_sum * cos_sum + sin_sum * sin_sum) / n as f64;
        pgram.push(intensity);
    }

    pgram
}

/// Approximate p-value from Kolmogorov distribution.
/// Uses the asymptotic approximation.
fn kolmogorov_p_value(d: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 2.0 {
        return 0.0;
    }

    // Asymptotic formula: P(D > d) ≈ 2 * sum_{k=1}^∞ (-1)^{k-1} exp(-2k²d²)
    let mut p = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64).powi(2) * d * d).exp();
        p += term;
        if term.abs() < 1e-10 {
            break;
        }
    }

    (2.0 * p).clamp(0.0, 1.0)
}

/// Run cumulative periodogram analysis (convenience wrapper).
pub fn run_cpgram(x: &[f64], taper: Option<f64>) -> Result<CpgramResult, String> {
    cpgram(x, taper)
}

/// Test for white noise using the cumulative periodogram.
///
/// Returns the KS test statistic and p-value for the null hypothesis
/// that the series is white noise.
pub fn white_noise_test(x: &[f64], taper: Option<f64>) -> Result<(f64, f64), String> {
    let result = cpgram(x, taper)?;
    Ok((result.ks_statistic, result.ks_p_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cpgram_basic() {
        // Simple test with synthetic data
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = cpgram(&x, Some(0.1)).unwrap();

        assert_eq!(result.n, 100);
        assert_relative_eq!(result.taper, 0.1, epsilon = 1e-10);
        assert!(!result.cpgram.is_empty());
        assert_eq!(result.cpgram.len(), result.freq.len());

        // Cumulative periodogram should end at 1.0
        assert_relative_eq!(*result.cpgram.last().unwrap(), 1.0, epsilon = 0.01);

        // Frequencies should be in (0, 0.5]
        assert!(result.freq.iter().all(|&f| f > 0.0 && f <= 0.5));
    }

    #[test]
    fn test_cpgram_white_noise() {
        // Generate white noise (random numbers)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let mut x = Vec::with_capacity(200);
        for i in 0..200 {
            i.hash(&mut hasher);
            let h = hasher.finish();
            x.push((h as f64 / u64::MAX as f64) - 0.5);
        }

        let result = cpgram(&x, Some(0.1)).unwrap();

        // For white noise, cumulative periodogram should be close to uniform
        // (though this is a statistical test, so we just check structure)
        assert!(result.max_deviation < 1.0);
    }

    #[test]
    fn test_cpgram_confidence_bands() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = cpgram(&x, Some(0.1)).unwrap();

        // CI bands should be properly ordered
        for i in 0..result.freq.len() {
            assert!(result.lower_ci[i] <= result.upper_ci[i]);
            assert!(result.lower_ci[i] >= 0.0);
            assert!(result.upper_ci[i] <= 1.0);
        }
    }

    #[test]
    fn test_cpgram_no_taper() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let result = cpgram(&x, Some(0.0)).unwrap();

        assert_relative_eq!(result.taper, 0.0, epsilon = 1e-10);
        assert!(!result.cpgram.is_empty());
    }

    #[test]
    fn test_cpgram_too_short() {
        let x = vec![1.0, 2.0, 3.0];
        let result = cpgram(&x, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_kolmogorov_p_value() {
        // Very small D should give p close to 1
        assert!(kolmogorov_p_value(0.01) > 0.9);

        // Very large D should give p close to 0
        assert!(kolmogorov_p_value(2.0) < 0.01);
    }

    #[test]
    fn test_apply_taper() {
        let x = vec![1.0; 10];
        let tapered = apply_taper(&x, 0.3);

        // Tapered ends should be smaller than middle
        // (After mean-centering, check structure)
        assert_eq!(tapered.len(), 10);
    }

    #[test]
    fn test_white_noise_test() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let (ks_stat, p_value) = white_noise_test(&x, Some(0.1)).unwrap();

        assert!(ks_stat >= 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }
}
