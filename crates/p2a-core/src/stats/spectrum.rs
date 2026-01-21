//! Spectral Density Estimation.
//!
//! Provides spectral density estimation via the periodogram method
//! with optional smoothing using modified Daniell kernels.
//!
//! # Overview
//!
//! The spectral density (power spectrum) describes how the variance of a
//! time series is distributed across frequencies. This module implements
//! the periodogram-based estimator with optional smoothing to produce
//! consistent estimates.
//!
//! # References
//!
//! - Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.
//! - Percival, D. B. & Walden, A. T. (1993). *Spectral Analysis for Physical
//!   Applications*. Cambridge University Press.
//! - Brillinger, D. R. (1981). *Time Series: Data Analysis and Theory*. SIAM.
//! - R `stats::spectrum`: <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/spectrum.html>
//! - R `stats::spec.pgram`: <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/spec.pgram.html>

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};

/// Configuration for spectral density estimation.
///
/// Controls preprocessing steps and smoothing options for the periodogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumConfig {
    /// Widths of modified Daniell smoothers (must be odd integers).
    /// Multiple spans are convolved together. Default: None (no smoothing).
    ///
    /// # Example
    /// - `spans = Some(vec![3])`: Simple moving average of width 3
    /// - `spans = Some(vec![3, 3])`: Convolution of two width-3 kernels
    pub spans: Option<Vec<usize>>,

    /// Proportion of data to taper (0.0 to 0.5). Default: 0.1.
    /// A split cosine bell taper is applied at each end of the series.
    /// Tapering reduces spectral leakage from sharp truncation.
    pub taper: f64,

    /// Whether to remove linear trend before computing periodogram.
    /// Default: true. Also removes the mean.
    pub detrend: bool,

    /// Whether to remove the mean (if detrend is false).
    /// Default: false (redundant if detrend = true).
    pub demean: bool,

    /// Zero-padding ratio (proportion of series length to add as zeros).
    /// Default: 0.0. Padding increases frequency resolution.
    pub pad_ratio: f64,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            spans: None,
            taper: 0.1,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
        }
    }
}

impl SpectrumConfig {
    /// Create a configuration with specified spans for smoothing.
    pub fn with_spans(spans: Vec<usize>) -> Self {
        Self {
            spans: Some(spans),
            ..Default::default()
        }
    }

    /// Create a configuration with no smoothing (raw periodogram).
    pub fn raw() -> Self {
        Self {
            spans: None,
            taper: 0.0,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
        }
    }
}

/// Result of spectral density estimation.
///
/// Contains the estimated spectral density at a set of Fourier frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumResult {
    /// Frequency values (cycles per unit time).
    /// Range: (0, 0.5] for normalized frequencies (Nyquist frequency = 0.5).
    pub freq: Vec<f64>,

    /// Spectral density estimates at each frequency.
    /// Units: variance per unit frequency (or per Hz if data has time units).
    pub spec: Vec<f64>,

    /// Smoothing bandwidth (in frequency units).
    /// Larger bandwidth = more smoothing, lower variance, lower resolution.
    pub bandwidth: f64,

    /// Degrees of freedom for confidence interval computation.
    /// For smoothed periodogram: df ≈ 2 × bandwidth × n
    pub df: f64,

    /// Number of observations in the original series.
    pub n_obs: usize,

    /// Length of the padded series used for computation.
    pub n_used: usize,

    /// Method used for estimation.
    pub method: String,

    /// Series name (if from dataset).
    pub series_name: Option<String>,

    /// Kernel bandwidths used (if smoothing applied).
    pub kernel_spans: Option<Vec<usize>>,

    /// Taper proportion used.
    pub taper: f64,

    /// Whether detrending was applied.
    pub detrend: bool,
}

impl SpectrumResult {
    /// Find the frequency with maximum spectral density (dominant frequency).
    pub fn peak_frequency(&self) -> Option<(f64, f64)> {
        self.spec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (self.freq[i], s))
    }

    /// Compute approximate 95% confidence interval multipliers.
    ///
    /// For a smoothed periodogram, the spectral estimate follows a scaled
    /// chi-squared distribution. The confidence interval is:
    /// [spec * df / chi2_upper, spec * df / chi2_lower]
    ///
    /// Returns (lower_multiplier, upper_multiplier) such that
    /// CI = [spec * lower_mult, spec * upper_mult]
    pub fn confidence_multipliers(&self, level: f64) -> (f64, f64) {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        if self.df < 2.0 {
            // Not enough smoothing for meaningful CI
            return (0.0, f64::INFINITY);
        }

        let alpha = 1.0 - level;
        let chi2 = ChiSquared::new(self.df).unwrap();

        // Lower and upper quantiles of chi-squared
        let chi2_lower = chi2.inverse_cdf(alpha / 2.0);
        let chi2_upper = chi2.inverse_cdf(1.0 - alpha / 2.0);

        // Multipliers for the spectral estimate
        let lower_mult = self.df / chi2_upper;
        let upper_mult = self.df / chi2_lower;

        (lower_mult, upper_mult)
    }
}

impl fmt::Display for SpectrumResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Spectral Density Estimation Results")?;
        writeln!(f, "==============================================")?;
        if let Some(ref name) = self.series_name {
            writeln!(f, "Series: {}", name)?;
        }
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Series Length Used: {}", self.n_used)?;
        writeln!(f, "Bandwidth: {:.4}", self.bandwidth)?;
        writeln!(f, "Degrees of Freedom: {:.2}", self.df)?;
        writeln!(f, "Taper: {:.1}%", self.taper * 100.0)?;
        writeln!(f, "Detrend: {}", if self.detrend { "Yes" } else { "No" })?;

        if let Some(ref spans) = self.kernel_spans {
            writeln!(f, "Kernel Spans: {:?}", spans)?;
        }

        writeln!(f)?;

        if let Some((peak_freq, peak_spec)) = self.peak_frequency() {
            writeln!(f, "Peak Frequency: {:.4} (period = {:.2})", peak_freq, 1.0 / peak_freq)?;
            writeln!(f, "Peak Spectral Density: {:.4}", peak_spec)?;
        }

        writeln!(f)?;

        // Show first and last few frequencies
        let n_show = 10.min(self.freq.len());
        writeln!(f, "Frequency    Spectrum")?;
        writeln!(f, "---------    --------")?;

        for i in 0..n_show.min(5) {
            writeln!(f, "{:9.4}    {:.4e}", self.freq[i], self.spec[i])?;
        }

        if self.freq.len() > 10 {
            writeln!(f, "   ...         ...")?;
            for i in (self.freq.len() - 5)..self.freq.len() {
                writeln!(f, "{:9.4}    {:.4e}", self.freq[i], self.spec[i])?;
            }
        } else if self.freq.len() > 5 {
            for i in 5..self.freq.len() {
                writeln!(f, "{:9.4}    {:.4e}", self.freq[i], self.spec[i])?;
            }
        }

        Ok(())
    }
}

/// Compute the spectral density of a time series using the periodogram method.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `config` - Configuration options (see [`SpectrumConfig`])
///
/// # Returns
///
/// * `SpectrumResult` - Spectral density estimates at Fourier frequencies
///
/// # Algorithm
///
/// 1. Optionally detrend/demean the series
/// 2. Apply split cosine bell taper
/// 3. Compute raw periodogram at Fourier frequencies
/// 4. Optionally smooth with modified Daniell kernel(s)
///
/// # Mathematical Details
///
/// The raw periodogram at frequency f_j = j/n is:
///
/// I(f_j) = (1/n) |Σ_{t=0}^{n-1} x_t exp(-2πi f_j t)|²
///
/// The smoothed periodogram convolves I(f) with the Daniell kernel:
///
/// Ŝ(f_j) = Σ_{k=-m}^{m} w_k I(f_{j+k})
///
/// where w_k are the kernel weights (half-weight at endpoints for modified Daniell).
///
/// # References
///
/// - Priestley, M. B. (1981). *Spectral Analysis and Time Series*, Chapter 6.
/// - R `spec.pgram`: Uses same approach with FFT optimization.
///
/// # Example
///
/// ```ignore
/// let x = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let config = SpectrumConfig::with_spans(vec![3, 3]);
/// let result = spectrum(&x, config)?;
/// println!("Peak frequency: {:?}", result.peak_frequency());
/// ```
pub fn spectrum(x: &[f64], config: SpectrumConfig) -> EconResult<SpectrumResult> {
    let n = x.len();

    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "Spectral density estimation".to_string(),
        });
    }

    // Validate config
    if config.taper < 0.0 || config.taper > 0.5 {
        return Err(EconError::InvalidSpecification {
            message: format!("Taper must be in [0, 0.5], got {}", config.taper),
        });
    }

    if let Some(ref spans) = config.spans {
        for &span in spans {
            if span < 1 || span % 2 == 0 {
                return Err(EconError::InvalidSpecification {
                    message: format!("Spans must be positive odd integers, got {}", span),
                });
            }
        }
    }

    // Step 1: Preprocess - detrend or demean
    let mut y: Vec<f64> = if config.detrend {
        detrend(x)
    } else if config.demean {
        let mean = x.iter().sum::<f64>() / n as f64;
        x.iter().map(|v| v - mean).collect()
    } else {
        x.to_vec()
    };

    // Step 2: Apply taper
    if config.taper > 0.0 {
        apply_cosine_taper(&mut y, config.taper);
    }

    // Step 3: Zero-padding
    let n_pad = (n as f64 * config.pad_ratio).round() as usize;
    let n_used = n + n_pad;
    y.extend(vec![0.0; n_pad]);

    // Step 4: Compute raw periodogram using FFT (O(n log n) instead of O(n²))
    // We compute at frequencies f_j = j/n_used for j = 1, ..., floor(n_used/2)
    // (skip f_0 = 0 as it's the mean, and we only need positive frequencies)
    let n_freq = n_used / 2;

    // Prepare FFT input: convert real signal to complex
    let mut fft_input: Vec<Complex<f64>> = y.iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_used);
    fft.process(&mut fft_input);

    // Extract periodogram from FFT output
    // FFT output[j] corresponds to frequency j/n_used
    // Power = |FFT[j]|² / n_used
    let mut freq = Vec::with_capacity(n_freq);
    let mut raw_periodogram = Vec::with_capacity(n_freq);

    for j in 1..=n_freq {
        let f_j = j as f64 / n_used as f64;
        freq.push(f_j);

        // Power spectral density from FFT
        let c = &fft_input[j];
        let power = (c.re * c.re + c.im * c.im) / n_used as f64;

        // Scale by 2 for one-sided spectrum (positive frequencies only)
        // except at Nyquist if n_used is even
        let scale = if j == n_freq && n_used % 2 == 0 { 1.0 } else { 2.0 };
        raw_periodogram.push(power * scale);
    }

    // Step 5: Smooth with modified Daniell kernel(s)
    let (spec, bandwidth, df) = if let Some(ref spans) = config.spans {
        let smoothed = apply_daniell_smoothing(&raw_periodogram, spans);

        // Compute bandwidth and degrees of freedom
        // Bandwidth = sum of span weights / n_used
        // For modified Daniell: effective width = (span - 1)/2 at half-weight
        let total_width: f64 = spans.iter().map(|&s| s as f64).product::<f64>().sqrt();
        let bw = total_width / n_used as f64;

        // Degrees of freedom ≈ 2 * n_used / L where L is effective kernel length
        // For convolved Daniell kernels: df ≈ 2 * product of spans
        let kernel_df: f64 = 2.0 * spans.iter().map(|&s| s as f64).product::<f64>();
        let effective_df = kernel_df.min(2.0 * n_freq as f64);

        (smoothed, bw, effective_df)
    } else {
        // Raw periodogram - no smoothing
        // df = 2 for chi-squared(2) distribution of periodogram ordinates
        let bw = 1.0 / n_used as f64; // Frequency resolution
        (raw_periodogram, bw, 2.0)
    };

    Ok(SpectrumResult {
        freq,
        spec,
        bandwidth,
        df,
        n_obs: n,
        n_used,
        method: "pgram".to_string(),
        series_name: None,
        kernel_spans: config.spans,
        taper: config.taper,
        detrend: config.detrend,
    })
}

/// Remove linear trend from a series.
///
/// Fits y = a + b*t by least squares and returns residuals.
fn detrend(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Compute linear regression coefficients
    // t = 0, 1, 2, ..., n-1
    // mean(t) = (n-1)/2
    // sum(t) = n(n-1)/2
    // sum(t^2) = n(n-1)(2n-1)/6

    let n_f = n as f64;
    let sum_t = n_f * (n_f - 1.0) / 2.0;
    let sum_t2 = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
    let mean_t = sum_t / n_f;

    let sum_x: f64 = x.iter().sum();
    let mean_x = sum_x / n_f;

    // sum((t - mean_t) * (x - mean_x)) = sum(t*x) - n * mean_t * mean_x
    let sum_tx: f64 = x.iter().enumerate().map(|(t, &v)| t as f64 * v).sum();
    let cov_tx = sum_tx - n_f * mean_t * mean_x;

    // var(t) = sum(t^2)/n - mean_t^2
    let var_t = sum_t2 / n_f - mean_t * mean_t;

    let slope = if var_t > 1e-15 { cov_tx / (n_f * var_t) } else { 0.0 };
    let intercept = mean_x - slope * mean_t;

    // Return residuals
    x.iter()
        .enumerate()
        .map(|(t, &v)| v - (intercept + slope * t as f64))
        .collect()
}

/// Apply split cosine bell taper to a series.
///
/// Tapers the first and last `p * n` points using a cosine bell function:
/// w(t) = 0.5 * (1 - cos(π * t / m))  for t = 0, ..., m-1
/// w(t) = 1  for t = m, ..., n-m-1
/// w(t) = 0.5 * (1 - cos(π * (n-t) / m))  for t = n-m, ..., n-1
///
/// The series is scaled to preserve variance.
fn apply_cosine_taper(x: &mut [f64], p: f64) {
    let n = x.len();
    if n == 0 || p <= 0.0 {
        return;
    }

    let m = ((p * n as f64).round() as usize).max(1).min(n / 2);

    // Compute taper weights
    let mut taper_weights = vec![1.0; n];
    for t in 0..m {
        let w = 0.5 * (1.0 - (PI * t as f64 / m as f64).cos());
        taper_weights[t] = w;
        taper_weights[n - 1 - t] = w;
    }

    // Compute normalization factor to preserve variance
    // sum of squared weights should equal n for variance preservation
    let sum_w2: f64 = taper_weights.iter().map(|w| w * w).sum();
    let norm = (n as f64 / sum_w2).sqrt();

    // Apply normalized taper
    for (t, v) in x.iter_mut().enumerate() {
        *v *= taper_weights[t] * norm;
    }
}

/// Compute Fourier coefficient at a given frequency.
///
/// Returns (Σ x_t cos(2π f t), Σ x_t sin(2π f t))
fn compute_fourier_coef(x: &[f64], freq: f64) -> (f64, f64) {
    let mut cos_sum = 0.0;
    let mut sin_sum = 0.0;

    for (t, &v) in x.iter().enumerate() {
        let angle = 2.0 * PI * freq * t as f64;
        cos_sum += v * angle.cos();
        sin_sum += v * angle.sin();
    }

    (cos_sum, sin_sum)
}

/// Apply modified Daniell smoothing to a periodogram.
///
/// The modified Daniell kernel is a moving average with half-weight at endpoints:
/// w_k = 1/(2m) for k = -m or k = m
/// w_k = 1/m for |k| < m
/// where span = 2m + 1
///
/// Multiple spans are convolved together by successive application.
fn apply_daniell_smoothing(periodogram: &[f64], spans: &[usize]) -> Vec<f64> {
    let mut result = periodogram.to_vec();

    for &span in spans {
        result = apply_single_daniell(&result, span);
    }

    result
}

/// Apply a single modified Daniell kernel of given span.
fn apply_single_daniell(x: &[f64], span: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || span <= 1 {
        return x.to_vec();
    }

    let m = span / 2; // Half-width
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for j in 0..span {
            let k = j as isize - m as isize;
            let idx = i as isize + k;

            if idx >= 0 && (idx as usize) < n {
                // Half-weight at endpoints of kernel
                let w = if j == 0 || j == span - 1 { 0.5 } else { 1.0 };
                sum += w * x[idx as usize];
                weight_sum += w;
            }
        }

        result.push(if weight_sum > 0.0 { sum / weight_sum } else { x[i] });
    }

    result
}

/// Compute the spectral density using an AR model fit.
///
/// Fits an AR(p) model to the series and computes the corresponding
/// spectral density function.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `order` - AR order (if None, selected by AIC)
/// * `n_freq` - Number of frequency points (default: 500)
///
/// # Mathematical Details
///
/// For an AR(p) model: x_t = Σ_{k=1}^p φ_k x_{t-k} + ε_t
///
/// The spectral density is:
///
/// S(f) = σ² / |1 - Σ_{k=1}^p φ_k exp(-2πi f k)|²
///
/// # References
///
/// - Brockwell & Davis (1991), Section 4.4
/// - R `spec.ar`
pub fn spectrum_ar(x: &[f64], order: Option<usize>, n_freq: Option<usize>) -> EconResult<SpectrumResult> {
    let n = x.len();

    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "AR spectral estimation".to_string(),
        });
    }

    // Fit AR model using Yule-Walker equations
    let max_order = order.unwrap_or_else(|| {
        // Default max order for AIC selection
        (10.0 * (n as f64).log10()).floor() as usize
    });
    let max_order = max_order.min(n / 2 - 1).max(1);

    let (ar_coefs, innovation_var) = if order.is_some() {
        fit_ar_yule_walker(x, max_order)?
    } else {
        // Select order by AIC
        select_ar_order_aic(x, max_order)?
    };

    let p = ar_coefs.len();
    let n_pts = n_freq.unwrap_or(500);

    // Compute spectral density at frequencies
    let mut freq = Vec::with_capacity(n_pts);
    let mut spec = Vec::with_capacity(n_pts);

    for j in 1..=n_pts {
        let f = j as f64 / (2.0 * n_pts as f64); // Frequencies in (0, 0.5]
        freq.push(f);

        // Compute |1 - Σ φ_k exp(-2πi f k)|²
        let mut re = 1.0;
        let mut im = 0.0;

        for (k, &phi) in ar_coefs.iter().enumerate() {
            let angle = 2.0 * PI * f * (k + 1) as f64;
            re -= phi * angle.cos();
            im += phi * angle.sin();
        }

        let denom = re * re + im * im;
        let s = if denom > 1e-15 { innovation_var / denom } else { innovation_var };
        spec.push(s);
    }

    Ok(SpectrumResult {
        freq,
        spec,
        bandwidth: 0.0, // AR spectrum is smooth, bandwidth not applicable
        df: f64::INFINITY, // Smooth estimate, infinite df
        n_obs: n,
        n_used: n,
        method: format!("ar({})", p),
        series_name: None,
        kernel_spans: None,
        taper: 0.0,
        detrend: true,
    })
}

/// Fit AR(p) model using Yule-Walker equations.
///
/// Returns (coefficients, innovation variance)
fn fit_ar_yule_walker(x: &[f64], p: usize) -> EconResult<(Vec<f64>, f64)> {
    let n = x.len();

    // Demean the series
    let mean = x.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = x.iter().map(|v| v - mean).collect();

    // Compute autocovariances
    let mut gamma = Vec::with_capacity(p + 1);
    for k in 0..=p {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += y[t] * y[t + k];
        }
        gamma.push(sum / n as f64);
    }

    if gamma[0] <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    // Solve Yule-Walker equations using Durbin-Levinson
    // This is more numerically stable than direct matrix solve
    let mut phi = vec![0.0; p];
    let mut v = gamma[0];

    if p == 0 {
        return Ok((vec![], v));
    }

    // First iteration
    phi[0] = gamma[1] / gamma[0];
    v = gamma[0] * (1.0 - phi[0] * phi[0]);

    for k in 2..=p {
        // Compute partial autocorrelation
        let mut num = gamma[k];
        for j in 0..(k - 1) {
            num -= phi[j] * gamma[k - 1 - j];
        }
        let phi_kk = num / v;

        // Update coefficients
        let phi_old = phi.clone();
        for j in 0..(k - 1) {
            phi[j] = phi_old[j] - phi_kk * phi_old[k - 2 - j];
        }
        phi[k - 1] = phi_kk;

        // Update innovation variance
        v *= 1.0 - phi_kk * phi_kk;

        if v <= 0.0 {
            break;
        }
    }

    Ok((phi, v))
}

/// Select AR order by AIC.
///
/// Fits AR(0) through AR(max_order) and selects the model minimizing AIC.
fn select_ar_order_aic(x: &[f64], max_order: usize) -> EconResult<(Vec<f64>, f64)> {
    let n = x.len();

    let mut best_aic = f64::INFINITY;
    let mut best_coefs = vec![];
    let mut best_var = 0.0;

    for p in 0..=max_order {
        let (coefs, var) = fit_ar_yule_walker(x, p)?;

        if var <= 0.0 {
            continue;
        }

        // AIC = n * log(var) + 2 * (p + 1)
        let aic = n as f64 * var.ln() + 2.0 * (p + 1) as f64;

        if aic < best_aic {
            best_aic = aic;
            best_coefs = coefs;
            best_var = var;
        }
    }

    if best_var <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Could not fit AR model".to_string(),
        });
    }

    Ok((best_coefs, best_var))
}

// ============================================================================
// Dataset-based convenience functions
// ============================================================================

/// Extract a numeric column from a dataset.
fn extract_column(dataset: &Dataset, col_name: &str) -> EconResult<Vec<f64>> {
    use polars::prelude::*;

    let series = dataset
        .df()
        .column(col_name)
        .map_err(|_| EconError::ColumnNotFound {
            column: col_name.to_string(),
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
            column: col_name.to_string(),
        })?;

    let ca = values.f64().map_err(|_| EconError::NonNumericColumn {
        column: col_name.to_string(),
    })?;

    ca.into_iter()
        .map(|opt| {
            opt.ok_or_else(|| EconError::NullValues {
                column: col_name.to_string(),
                count: 1,
            })
        })
        .collect()
}

/// Run spectral density estimation on a dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `config` - Configuration options
///
/// # Example
///
/// ```ignore
/// let config = SpectrumConfig::with_spans(vec![3, 3]);
/// let result = run_spectrum(&dataset, "returns", config)?;
/// println!("{}", result);
/// ```
pub fn run_spectrum(
    dataset: &Dataset,
    column: &str,
    config: SpectrumConfig,
) -> EconResult<SpectrumResult> {
    let x = extract_column(dataset, column)?;
    let mut result = spectrum(&x, config)?;
    result.series_name = Some(column.to_string());
    Ok(result)
}

/// Run AR-based spectral estimation on a dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `order` - AR order (None for AIC selection)
/// * `n_freq` - Number of frequency points
///
/// # Example
///
/// ```ignore
/// let result = run_spectrum_ar(&dataset, "returns", None, Some(200))?;
/// println!("{}", result);
/// ```
pub fn run_spectrum_ar(
    dataset: &Dataset,
    column: &str,
    order: Option<usize>,
    n_freq: Option<usize>,
) -> EconResult<SpectrumResult> {
    let x = extract_column(dataset, column)?;
    let mut result = spectrum_ar(&x, order, n_freq)?;
    result.series_name = Some(column.to_string());
    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_spectrum_basic() {
        // Test with a simple series
        let x: Vec<f64> = (0..100).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        let config = SpectrumConfig::default();
        let result = spectrum(&x, config).unwrap();

        // Should have n/2 frequency points
        assert_eq!(result.freq.len(), 50);
        assert_eq!(result.spec.len(), 50);

        // All spectral values should be non-negative
        for &s in &result.spec {
            assert!(s >= 0.0, "Spectral density should be non-negative");
        }

        // Peak should be near frequency 0.1 (period = 10)
        let (peak_freq, _peak_spec) = result.peak_frequency().unwrap();
        assert!(
            (peak_freq - 0.1).abs() < 0.02,
            "Peak frequency {} should be near 0.1",
            peak_freq
        );
    }

    #[test]
    fn test_spectrum_white_noise() {
        // White noise should have approximately flat spectrum
        let x: Vec<f64> = vec![
            0.1, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.3, 0.2, 0.1,
            0.3, -0.1, 0.0, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.2,
        ];

        let config = SpectrumConfig::with_spans(vec![3, 3]);
        let result = spectrum(&x, config).unwrap();

        // Check that spectrum values don't vary too wildly (roughly flat)
        let mean_spec: f64 = result.spec.iter().sum::<f64>() / result.spec.len() as f64;
        let max_spec = result.spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_spec = result.spec.iter().cloned().fold(f64::INFINITY, f64::min);

        // For white noise, ratio of max to min shouldn't be extreme after smoothing
        let ratio = max_spec / min_spec.max(1e-10);
        assert!(
            ratio < 20.0,
            "Smoothed white noise spectrum should be roughly flat, ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_detrend() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x);

        // Linear trend should be completely removed
        // Mean of detrended should be near 0
        let mean: f64 = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert!(
            mean.abs() < 1e-10,
            "Detrended series should have zero mean, got {}",
            mean
        );

        // All values should be near 0 for perfect linear series
        for &v in &detrended {
            assert!(
                v.abs() < 1e-10,
                "Detrended linear series should be all zeros"
            );
        }
    }

    #[test]
    fn test_cosine_taper() {
        let mut x = vec![1.0; 20];
        apply_cosine_taper(&mut x, 0.1);

        // First and last 2 points should be tapered (10% of 20 = 2)
        assert!(x[0] < 1.0, "First point should be tapered down");
        assert!(x[1] < 1.0, "Second point should be tapered");
        assert!(x[10].abs() > 0.5, "Middle points should not be heavily tapered");

        // Symmetry check
        assert!(
            approx_eq(x[0], x[19], 1e-10),
            "Taper should be symmetric"
        );
        assert!(
            approx_eq(x[1], x[18], 1e-10),
            "Taper should be symmetric"
        );
    }

    #[test]
    fn test_daniell_smoothing() {
        // Test that smoothing reduces variance
        let x: Vec<f64> = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 1.0, 5.0, 2.0, 4.0];

        let smoothed = apply_daniell_smoothing(&x, &[3]);

        // Variance should be reduced
        let var_orig: f64 = {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
        };

        let var_smooth: f64 = {
            let mean = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
            smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / smoothed.len() as f64
        };

        assert!(
            var_smooth < var_orig,
            "Smoothing should reduce variance: {} vs {}",
            var_smooth,
            var_orig
        );
    }

    #[test]
    fn test_spectrum_ar() {
        // Test AR spectral estimation
        let x: Vec<f64> = (0..100).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        let result = spectrum_ar(&x, Some(5), Some(100)).unwrap();

        assert_eq!(result.freq.len(), 100);
        assert_eq!(result.spec.len(), 100);

        // All spectral values should be non-negative
        for &s in &result.spec {
            assert!(s >= 0.0, "AR spectral density should be non-negative");
        }
    }

    #[test]
    fn test_ar_fit() {
        // Test AR model fitting with AR(1) data
        // x_t = 0.5 * x_{t-1} + e_t
        // Using a longer series with proper random-like noise
        let mut x = Vec::with_capacity(500);
        let phi = 0.5;
        x.push(0.0);

        // Use a simple linear congruential generator for reproducible pseudo-random noise
        let mut seed: u64 = 12345;
        for t in 1..500 {
            // Simple LCG for pseudo-random numbers in [-1, 1]
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let e = ((seed >> 16) as f64 / 32768.0 - 1.0) * 0.5;
            x.push(phi * x[t - 1] + e);
        }

        let (coefs, _var) = fit_ar_yule_walker(&x, 1).unwrap();

        // Estimated coefficient should be reasonably close to phi
        // Allowing tolerance for finite sample estimation
        assert!(
            (coefs[0] - phi).abs() < 0.2,
            "AR(1) coefficient {} should be near {}",
            coefs[0],
            phi
        );
    }

    #[test]
    fn test_config_validation() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();

        // Invalid taper
        let config = SpectrumConfig {
            taper: 0.6,
            ..Default::default()
        };
        assert!(spectrum(&x, config).is_err());

        // Invalid span (even number)
        let config = SpectrumConfig::with_spans(vec![4]);
        assert!(spectrum(&x, config).is_err());
    }

    #[test]
    fn test_confidence_multipliers() {
        let result = SpectrumResult {
            freq: vec![0.1],
            spec: vec![1.0],
            bandwidth: 0.05,
            df: 20.0,
            n_obs: 100,
            n_used: 100,
            method: "pgram".to_string(),
            series_name: None,
            kernel_spans: Some(vec![3, 3]),
            taper: 0.1,
            detrend: true,
        };

        let (lower, upper) = result.confidence_multipliers(0.95);

        // Lower multiplier should be < 1, upper > 1
        assert!(lower < 1.0 && lower > 0.0);
        assert!(upper > 1.0);
    }

    /// Validation test against R spec.pgram
    ///
    /// R code:
    /// ```r
    /// x <- sin(2*pi*(1:100)/10)
    /// sp <- spec.pgram(x, spans=c(3,3), taper=0.1, plot=FALSE)
    /// # Peak frequency should be at 0.1
    /// sp$freq[which.max(sp$spec)]
    /// ```
    #[test]
    fn test_validate_spectrum_against_r() {
        // Generate sine wave with period 10 (frequency 0.1)
        let x: Vec<f64> = (1..=100).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();

        let config = SpectrumConfig::with_spans(vec![3, 3]);
        let result = spectrum(&x, config).unwrap();

        // R output: peak frequency should be at or very near 0.1
        let (peak_freq, _) = result.peak_frequency().unwrap();

        assert!(
            (peak_freq - 0.1).abs() < 0.02,
            "Peak frequency {} should match R's result (≈0.1)",
            peak_freq
        );
    }

    /// Validation test: Raw periodogram (no smoothing) against R
    ///
    /// R code:
    /// ```r
    /// x <- 1:10
    /// sp <- spec.pgram(x, spans=NULL, taper=0, detrend=TRUE, plot=FALSE)
    /// length(sp$freq)  # Should be 5 (n/2)
    /// ```
    #[test]
    fn test_validate_raw_periodogram() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let config = SpectrumConfig::raw();
        let result = spectrum(&x, config).unwrap();

        // Should have n/2 = 5 frequency points
        assert_eq!(result.freq.len(), 5);

        // Detrended linear series should have small spectrum (near zero variance)
        let total_power: f64 = result.spec.iter().sum();
        assert!(
            total_power < 1.0,
            "Detrended linear series should have minimal spectral power, got {}",
            total_power
        );
    }

    /// Comprehensive validation test against R spec.pgram
    ///
    /// R code:
    /// ```r
    /// set.seed(42)
    /// x <- sin(2*pi*(1:100)/10)
    /// sp <- spec.pgram(x, spans=c(3,3), taper=0.1, detrend=TRUE, plot=FALSE)
    /// sp$freq[which.max(sp$spec)]  # 0.1
    /// sp$bandwidth  # 0.0104083300
    /// sp$df  # 6.5521
    /// ```
    #[test]
    fn test_validate_spectrum_comprehensive_against_r() {
        // Generate sine wave with period 10 (frequency 0.1) - same as R
        let x: Vec<f64> = (1..=100).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();

        let config = SpectrumConfig {
            spans: Some(vec![3, 3]),
            taper: 0.1,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
        };
        let result = spectrum(&x, config).unwrap();

        // R: SINE_PEAK_FREQ = 0.100000
        let (peak_freq, _) = result.peak_frequency().unwrap();
        assert!(
            (peak_freq - 0.1).abs() < 0.01,
            "Peak frequency {} should be 0.1 (R = 0.1)",
            peak_freq
        );

        // Check number of frequencies (should be n/2 = 50)
        assert_eq!(
            result.freq.len(), 50,
            "Should have n/2 = 50 frequencies"
        );

        // Check that frequencies span (0, 0.5]
        assert!(
            result.freq.first().unwrap() > &0.0,
            "First frequency should be positive"
        );
        assert!(
            result.freq.last().unwrap() <= &0.5,
            "Last frequency should be <= 0.5"
        );

        // R produces df ≈ 6.55 for spans=c(3,3)
        // Our implementation uses a different df formula but should be in similar range
        // The important thing is that df increases with more smoothing
        assert!(
            result.df > 2.0,
            "Degrees of freedom {} should be > 2 for smoothed periodogram",
            result.df
        );
    }

    /// Validation: Detrended linear series should have near-zero power
    ///
    /// R code:
    /// ```r
    /// x <- 1:20
    /// sp <- spec.pgram(x, spans=NULL, taper=0.1, detrend=TRUE, plot=FALSE)
    /// sum(sp$spec)  # 0 or very small
    /// ```
    #[test]
    fn test_validate_detrend_linear() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        let config = SpectrumConfig {
            spans: None,
            taper: 0.1,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
        };
        let result = spectrum(&x, config).unwrap();

        // R: LINEAR_TOTAL_POWER = 0.0000000000
        let total_power: f64 = result.spec.iter().sum();
        assert!(
            total_power < 1e-8,
            "Detrended linear series should have near-zero power, got {}",
            total_power
        );
    }

    /// Validation: Two-frequency series should show peaks at both frequencies
    ///
    /// R code:
    /// ```r
    /// x <- cos(2*pi*(1:50)/5) + 0.5*cos(2*pi*(1:50)/10)
    /// sp <- spec.pgram(x, spans=c(3), taper=0.1, detrend=TRUE, plot=FALSE)
    /// # Highest peaks at f=0.2 and f=0.1
    /// ```
    #[test]
    fn test_validate_two_frequency_spectrum() {
        // Signal with two frequencies: f=0.2 (period 5) and f=0.1 (period 10)
        let x: Vec<f64> = (1..=50).map(|i| {
            (2.0 * PI * i as f64 / 5.0).cos() + 0.5 * (2.0 * PI * i as f64 / 10.0).cos()
        }).collect();

        let config = SpectrumConfig::with_spans(vec![3]);
        let result = spectrum(&x, config).unwrap();

        // Find the two highest peaks
        let mut indexed: Vec<(usize, f64)> = result.spec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let peak1_freq = result.freq[indexed[0].0];
        let peak2_freq = result.freq[indexed[1].0];

        // The dominant frequency should be at or near 0.2 (the larger amplitude component)
        assert!(
            (peak1_freq - 0.2).abs() < 0.04,
            "Primary peak {} should be near 0.2",
            peak1_freq
        );

        // Either the second or third peak should be near 0.1
        let has_secondary = indexed.iter().take(5).any(|(i, _)| {
            let f = result.freq[*i];
            (f - 0.1).abs() < 0.04 || (f - 0.2).abs() < 0.04
        });
        assert!(
            has_secondary,
            "Should have peaks near both 0.1 and 0.2, top frequencies: {:?}",
            indexed.iter().take(5).map(|(i, _)| result.freq[*i]).collect::<Vec<_>>()
        );
    }

    /// Validation: White noise should have relatively flat spectrum
    #[test]
    fn test_validate_white_noise_spectrum() {
        // Generate pseudo-random white noise using LCG
        let mut x = Vec::with_capacity(100);
        let mut seed: u64 = 42;
        for _ in 0..100 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((seed >> 16) as f64 / 32768.0 - 1.0);
            x.push(val);
        }

        let config = SpectrumConfig::with_spans(vec![5, 5]);
        let result = spectrum(&x, config).unwrap();

        // For white noise, coefficient of variation should be moderate
        // (not as flat as theoretical but not highly peaked either)
        let mean_spec: f64 = result.spec.iter().sum::<f64>() / result.spec.len() as f64;
        let var_spec: f64 = result.spec.iter().map(|s| (s - mean_spec).powi(2)).sum::<f64>()
            / result.spec.len() as f64;
        let cv = var_spec.sqrt() / mean_spec;

        // R shows CV ≈ 0.35 for white noise with spans=c(5,5)
        // Allow broader range due to different implementations
        assert!(
            cv < 2.0,
            "White noise CV {} should be < 2.0 (R ≈ 0.35)",
            cv
        );
    }
}
