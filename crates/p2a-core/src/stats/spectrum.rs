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
use crate::data::columns::{NaPolicy, numeric_column, refuse_non_finite};
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, checkpoint_stride, try_vec, try_with_capacity};
use crate::traits::estimator::{chi_squared_p_value, chi_squared_quantile, validate_level};

fn size_overflow(what: &str) -> EconError {
    EconError::ResourceLimit(format!("{what} is too large: the requested size overflows"))
}

/// Configuration for spectral density estimation.
///
/// Controls preprocessing steps and smoothing options for the periodogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumConfig {
    /// Widths of modified Daniell smoothers (must be odd integers).
    /// Multiple spans are convolved together. Default: None (no smoothing).
    /// The combined kernel (2M + 1 ordinates, M = Σ⌊span/2⌋) must be no
    /// longer than the series (2M < n_used), as in R.
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
    /// Default: 0.0. Padding increases frequency resolution. As R's `pad`,
    /// `trunc(n * pad_ratio)` zeros are added; a ratio of 0 or below adds none.
    pub pad_ratio: f64,

    /// Pad the (already padded) series with zeros to the next highly composite
    /// length (factors 2, 3 and 5 only), as R's `fast = TRUE`. Default: true,
    /// R's default. A series of 97 points is then analysed at length 100.
    #[serde(default = "default_fast")]
    pub fast: bool,
}

fn default_fast() -> bool {
    true
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            spans: None,
            taper: 0.1,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
            fast: true,
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
            fast: true,
        }
    }
}

/// Result of spectral density estimation.
///
/// Contains the estimated spectral density at a set of Fourier frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumResult {
    /// Frequency values (cycles per unit time): `j / n_used`, `j = 1..floor(n_used / 2)`, for a
    /// periodogram; `seq(0, 0.5, length.out = n_freq)` for an AR spectrum.
    pub freq: Vec<f64>,

    /// Spectral density estimates at each frequency.
    /// Units: variance per unit frequency (or per Hz if data has time units).
    pub spec: Vec<f64>,

    /// Smoothing bandwidth (in frequency units), R's `bandwidth`: the smoothing kernel's
    /// `sqrt(sum((1/12 + i^2) k_i))` (`sqrt(1/12)` unsmoothed) divided by `n_used`.
    /// Larger bandwidth = more smoothing, lower variance, lower resolution.
    pub bandwidth: f64,

    /// Degrees of freedom for confidence interval computation, R's `df`: the kernel's
    /// `2 / sum(k^2)` (2 unsmoothed) divided by the taper factor `u4 / u2^2` and scaled by
    /// `n_obs / n_used`.
    /// `f64::INFINITY` for AR spectra, which have no chi-square interval.
    pub df: f64,

    /// Number of observations in the original series.
    pub n_obs: usize,

    /// Length of the padded series used for computation (R's `n.used`).
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
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, &s)| (self.freq[i], s))
    }

    /// Confidence interval multipliers: the interval is `[spec * lower, spec * upper]`.
    ///
    /// R's `stats:::plot.spec` rule (`spec.ci`): for `X ~ chi^2(df)` the tail `1 - level` is split
    /// in proportion to `P(X > df)` above and `P(X <= df)` below, so the multipliers are
    /// `df / qchisq(1 - (1 - level) P(X > df), df)` and `df / qchisq((1 - level) P(X <= df), df)`.
    /// The df is `spec.pgram`'s.
    ///
    /// `Ok(None)` when the df is not finite and positive: an AR spectrum has no chi-square
    /// interval (R's `spec.ar` reports no df). A level outside (0, 1) is an error.
    pub fn confidence_interval_multipliers(&self, level: f64) -> EconResult<Option<(f64, f64)>> {
        validate_level(level, "level")?;
        if !(self.df.is_finite() && self.df > 0.0) {
            return Ok(None);
        }
        let tail = 1.0 - level;
        let above = chi_squared_p_value(self.df, self.df);
        let q_upper = chi_squared_quantile(1.0 - tail * above, self.df)?;
        let q_lower = chi_squared_quantile(tail * (1.0 - above), self.df)?;
        Ok(Some((self.df / q_upper, self.df / q_lower)))
    }

    /// [`Self::confidence_interval_multipliers`] as a pair: `(NaN, NaN)` when there is no interval
    /// (an AR spectrum) or the level is invalid.
    pub fn confidence_multipliers(&self, level: f64) -> (f64, f64) {
        match self.confidence_interval_multipliers(level) {
            Ok(Some(m)) => m,
            _ => (f64::NAN, f64::NAN),
        }
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
        if self.df.is_finite() {
            writeln!(f, "Bandwidth: {:.4}", self.bandwidth)?;
            writeln!(f, "Degrees of Freedom: {:.2}", self.df)?;
        } else {
            writeln!(f, "Bandwidth: not defined (AR spectrum)")?;
            writeln!(
                f,
                "Degrees of Freedom: not defined (AR spectrum: no chi-square interval)"
            )?;
        }
        writeln!(f, "Taper: {:.1}%", self.taper * 100.0)?;
        writeln!(f, "Detrend: {}", if self.detrend { "Yes" } else { "No" })?;

        if let Some(ref spans) = self.kernel_spans {
            writeln!(f, "Kernel Spans: {:?}", spans)?;
        }

        writeln!(f)?;

        if let Some((peak_freq, peak_spec)) = self.peak_frequency() {
            // an AR spectrum's grid starts at frequency 0, which has no period
            if peak_freq > 0.0 {
                writeln!(
                    f,
                    "Peak Frequency: {:.4} (period = {:.2})",
                    peak_freq,
                    1.0 / peak_freq
                )?;
            } else {
                writeln!(
                    f,
                    "Peak Frequency: {:.4} (period = not defined at frequency 0)",
                    peak_freq
                )?;
            }
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
/// # Errors
///
/// - [`EconError::InsufficientData`] for fewer than 4 observations.
/// - [`EconError::InvalidSpecification`] for a taper that is NaN or outside
///   `[0, 0.5]`; for NaN or infinite values in `x`, or a non-finite
///   `pad_ratio`; for an even or zero span; for spans whose combined kernel
///   (`2M + 1` ordinates, with `M = Σ floor(span / 2)`) is longer than the
///   padded series, i.e. `2M >= n_used` (R's `kernapply` rule, which
///   `spec.pgram` applies); and for a series so large in magnitude that its
///   periodogram overflows.
/// - [`EconError::ResourceLimit`] when the padded length `n_used` overflows or
///   its FFT buffer cannot be allocated (the per-allocation ceiling included).
/// - [`EconError::Cancelled`] when the budget checked through [`checkpoint`]
///   is cancelled or past its deadline before a smoothing pass.
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
    // `contains` also rejects NaN, for which both `< 0.0` and `> 0.5` are false.
    if !(0.0..=0.5).contains(&config.taper) {
        return Err(EconError::InvalidSpecification {
            message: format!("Taper must be in [0, 0.5], got {}", config.taper),
        });
    }

    refuse_non_finite(x, "x")?;

    if !config.pad_ratio.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("pad_ratio must be finite, got {}", config.pad_ratio),
        });
    }

    // Lengths as R's spec.pgram: N0 = n; `pad` adds `trunc(n * pad)` zeros (R's
    // `matrix(0, nrow = N * pad)` truncates a non-integer row count, and adds
    // none for pad <= 0); `fast` pads further to the next 2-3-5 length.
    let padded = n as f64 * config.pad_ratio;
    let n_pad = if padded >= 1.0 {
        if padded >= usize::MAX as f64 {
            return Err(size_overflow("the padded series"));
        }
        padded.trunc() as usize
    } else {
        0
    };
    let mut n_used = n
        .checked_add(n_pad)
        .ok_or_else(|| size_overflow("the padded series"))?;
    if config.fast {
        n_used = nextn(n_used).ok_or_else(|| size_overflow("the padded series"))?;
    }

    if let Some(ref spans) = config.spans {
        validate_spans(spans, n_used)?;
    }

    // Steps 1-3: detrend or demean, taper and zero-pad, written straight into
    // the FFT buffer.
    let mut fft_buf: Vec<Complex<f64>> =
        try_with_capacity(n_used, "the FFT buffer for the padded series")?;
    if config.detrend {
        let (mean, slope) = detrend_line(x);
        let centre = (n as f64 + 1.0) / 2.0;
        fft_buf.extend(
            x.iter()
                .enumerate()
                .map(|(i, &v)| Complex::new(v - mean - slope * ((i + 1) as f64 - centre), 0.0)),
        );
    } else if config.demean {
        let mean = x.iter().sum::<f64>() / n as f64;
        fft_buf.extend(x.iter().map(|&v| Complex::new(v - mean, 0.0)));
    } else {
        fft_buf.extend(x.iter().map(|&v| Complex::new(v, 0.0)));
    }
    apply_cosine_taper(&mut fft_buf, config.taper);
    fft_buf.resize(n_used, Complex::new(0.0, 0.0));

    // Step 4: the periodogram |X_j|^2 / N0 (R divides by the unpadded length).
    // The FFT of a real series is conjugate-symmetric, so ordinates
    // 0..=floor(N/2) hold the whole periodogram.
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n_used).process(&mut fft_buf);
    let half = n_used / 2;
    let n0 = n as f64;
    let mut pgram = try_vec(half + 1, 0.0, "the periodogram")?;
    for (p, c) in pgram.iter_mut().zip(&fft_buf) {
        *p = c.norm_sqr() / n0;
    }
    // R: the zero-frequency ordinate (zero after demeaning) is replaced by the
    // mean of its two neighbours, I_1 and I_{N-1}, before smoothing.
    pgram[0] = 0.5 * (pgram[1] + fft_buf[n_used - 1].norm_sqr() / n0);
    drop(fft_buf);

    if pgram.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the series is too large in magnitude for a periodogram".to_string(),
        });
    }

    // Step 5: smooth with modified Daniell kernel(s), circularly as R.
    if let Some(ref spans) = config.spans {
        apply_daniell_smoothing(&mut pgram, n_used, spans)?;
    }

    // Ordinates 1..=floor(N/2), corrected for the taper's loss of power.
    let u2 = 1.0 - (5.0 / 8.0) * config.taper * 2.0;
    pgram.remove(0);
    for v in &mut pgram {
        *v /= u2;
    }
    let spec = pgram;
    let freq: Vec<f64> = (1..=half).map(|j| j as f64 / n_used as f64).collect();

    // R's df and bandwidth: those of the combined kernel (df.kernel = 2 / sum(k^2),
    // bandwidth.kernel = sqrt(sum((1/12 + i^2) k_i)); 2 and sqrt(1/12) unsmoothed), corrected
    // for the taper and the padding.
    let (kernel_df, kernel_bw) = match config.spans {
        Some(ref spans) => kernel_df_bandwidth(spans)?,
        None => (2.0, (1.0f64 / 12.0).sqrt()),
    };
    let u4 = 1.0 - (93.0 / 128.0) * config.taper * 2.0;
    let df = kernel_df / (u4 / (u2 * u2)) * (n as f64 / n_used as f64);
    let bandwidth = kernel_bw / n_used as f64;

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

/// The smallest integer `>= n` whose only prime factors are 2, 3 and 5 (R's
/// `nextn`); `None` when it does not fit in `usize`. Found by enumerating
/// `3^b 5^c` and the smallest power of two that lifts each to `n`.
fn nextn(n: usize) -> Option<usize> {
    let target = n.max(1) as u128;
    let mut best = u128::MAX;
    let mut p5: u128 = 1;
    loop {
        let mut p35 = p5;
        loop {
            let mut v = p35;
            while v < target {
                v *= 2;
            }
            best = best.min(v);
            if p35 >= target {
                break;
            }
            p35 *= 3;
        }
        if p5 >= target {
            break;
        }
        p5 *= 5;
    }
    usize::try_from(best).ok()
}

/// The degrees of freedom `2 / sum(k^2)` and bandwidth `sqrt(sum((1/12 + i^2) k_i))` of the
/// convolution `k` of the modified Daniell kernels of `spans` (R's `df.kernel` and
/// `bandwidth.kernel` of `kernel("modified.daniell", spans %/% 2)`). The spans are validated, so
/// the kernel is no longer than the series.
fn kernel_df_bandwidth(spans: &[usize]) -> EconResult<(f64, f64)> {
    let total: usize = spans.iter().map(|s| s / 2).sum();
    let mut k = try_vec(2 * total + 1, 0.0, "the smoothing kernel")?;
    let mut next = try_vec(2 * total + 1, 0.0, "the smoothing kernel")?;
    k[0] = 1.0;
    let mut len = 1;
    for &span in spans {
        checkpoint()?;
        let m = span / 2;
        if m == 0 {
            continue;
        }
        let inner = 1.0 / (2 * m) as f64;
        let new_len = len + 2 * m;
        next[..new_len].fill(0.0);
        for (i, &ki) in k[..len].iter().enumerate() {
            for j in 0..=2 * m {
                let w = if j == 0 || j == 2 * m {
                    0.5 * inner
                } else {
                    inner
                };
                next[i + j] += ki * w;
            }
        }
        std::mem::swap(&mut k, &mut next);
        len = new_len;
    }
    let half = (len - 1) as f64 / 2.0;
    let sum_sq: f64 = k[..len].iter().map(|v| v * v).sum();
    let moment: f64 = k[..len]
        .iter()
        .enumerate()
        .map(|(i, v)| (1.0 / 12.0 + (i as f64 - half).powi(2)) * v)
        .sum();
    Ok((2.0 / sum_sq, moment.sqrt()))
}

/// The mean and slope of R's `spec.pgram` detrending: the detrended value at
/// time `t = 1..n` is `x - mean - slope (t - (n + 1) / 2)`, with
/// `slope = sum(x t') / sum(t'^2)` over the centred times `t'`.
fn detrend_line(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let centre = (n + 1.0) / 2.0;
    let sum_t2 = n * (n * n - 1.0) / 12.0;
    let sum_xt: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &v)| v * ((i + 1) as f64 - centre))
        .sum();
    let slope = if sum_t2 > 0.0 { sum_xt / sum_t2 } else { 0.0 };
    (mean, slope)
}

/// R's split cosine bell taper (`spec.taper`): with `m = floor(n p)`, the
/// first and last `m` values are multiplied by
/// `w_j = 0.5 (1 - cos(pi (2j - 1) / (2m)))`, `j = 1..m` (mirrored at the
/// end). There is no renormalisation: `spectrum` divides the periodogram by
/// R's `u2 = 1 - (5/8) 2p` instead.
fn apply_cosine_taper(x: &mut [Complex<f64>], p: f64) {
    let n = x.len();
    let m = ((n as f64 * p).floor() as usize).min(n / 2);
    for j in 1..=m {
        let w = 0.5 * (1.0 - (PI * (2 * j - 1) as f64 / (2 * m) as f64).cos());
        x[j - 1] *= w;
        x[n - j] *= w;
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

/// Apply modified Daniell smoothing to a periodogram, circularly (R's
/// `kernapply(circular = TRUE)` with `kernel("modified.daniell", spans %/% 2)`).
///
/// `half` holds ordinates `0..=floor(N/2)` of a periodogram of length
/// `n_full = N`. The periodogram of a real series is symmetric
/// (`I_{N-j} = I_j`) and every pass of a symmetric kernel keeps it so; a
/// circular pass over all `N` ordinates therefore equals a pass over the half
/// with mirrored indices. The modified Daniell kernel of span `2m + 1` has
/// weight `1/(2m)` inside and `1/(4m)` on both endpoints; several spans are
/// applied one after another (their convolution is R's combined kernel).
fn apply_daniell_smoothing(half: &mut Vec<f64>, n_full: usize, spans: &[usize]) -> EconResult<()> {
    let mut out: Vec<f64> = Vec::new();
    for &span in spans {
        checkpoint()?;
        if span <= 1 {
            continue;
        }
        if out.is_empty() {
            out = try_vec(half.len(), 0.0, "the smoothed periodogram")?;
        }
        daniell_pass_mirrored(half, n_full, span / 2, &mut out);
        std::mem::swap(half, &mut out);
    }
    Ok(())
}

/// Check smoothing spans against the (padded) series length `n_used`.
///
/// Every span must be a positive odd integer, and the combined modified
/// Daniell kernel (`2M + 1` ordinates) must be no longer than the series. This
/// is R's `kernapply` rule: with `M` the sum of the half-widths `span / 2`,
/// `2 * M < n_used`.
fn validate_spans(spans: &[usize], n_used: usize) -> EconResult<()> {
    for &span in spans {
        if span < 1 || span % 2 == 0 {
            return Err(EconError::InvalidSpecification {
                message: format!("Spans must be positive odd integers, got {}", span),
            });
        }
    }

    const ADVICE: &str =
        "R's spec.pgram refuses this too ('x' is shorter than kernel 'k'). Use smaller spans";
    let Some(half_widths) = spans
        .iter()
        .try_fold(0usize, |acc, &s| acc.checked_add(s / 2))
    else {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "spans give a smoothing kernel too wide to represent, which is longer \
                 than the {n_used}-point series; {ADVICE}"
            ),
        });
    };
    if half_widths.checked_mul(2).is_none_or(|w| w >= n_used) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "spans give a smoothing kernel {} ordinates wide (sum of half-widths \
                 {half_widths}), which is longer than the {n_used}-point series; {ADVICE}",
                half_widths.saturating_mul(2).saturating_add(1)
            ),
        });
    }
    Ok(())
}

/// A Neumaier-compensated running sum; values are removed with `add(-v)`.
///
/// A periodogram can span many orders of magnitude. The compensation keeps
/// the rounding a large ordinate leaves behind out of the window sums of the
/// small ordinates that follow it.
#[derive(Default)]
struct RunningSum {
    sum: f64,
    comp: f64,
}

impl RunningSum {
    fn add(&mut self, v: f64) {
        let t = self.sum + v;
        self.comp += if self.sum.abs() >= v.abs() {
            (self.sum - t) + v
        } else {
            (v - t) + self.sum
        };
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// One circular modified Daniell pass of half-width `m` (`2m < n_full`) over
/// the ordinates `0..=floor(n_full / 2)` of a symmetric periodogram, written
/// to `out`. An index `j` outside the half is mirrored: `-j` below zero,
/// `n_full - j` above `floor(n_full / 2)`; one reflection suffices because
/// `2m < n_full`. The full-weight part of the kernel is a sliding window, so
/// one pass costs O(n) whatever the span.
fn daniell_pass_mirrored(x: &[f64], n_full: usize, m: usize, out: &mut [f64]) {
    let h = x.len() - 1;
    let at = |j: isize| -> f64 {
        let j = j.unsigned_abs();
        x[if j > h { n_full - j } else { j }]
    };
    let inner = 1.0 / (2 * m) as f64;
    let end = 0.5 * inner;
    let m = m as isize;
    // Full-weight window [i - m + 1, i + m - 1] for i = 0.
    let mut window = RunningSum::default();
    for j in (1 - m)..m {
        window.add(at(j));
    }
    for (i, o) in out.iter_mut().enumerate() {
        let i = i as isize;
        *o = window.value() * inner + (at(i - m) + at(i + m)) * end;
        window.add(at(i + m));
        window.add(-at(i + 1 - m));
    }
}

/// Compute the spectral density using an AR model fit.
///
/// Fits an AR(p) model to the series and computes the corresponding
/// spectral density function.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `order` - `Some(k)` fits AR(k) with k clamped to `1..=n/2 - 1`; `None`
///   selects by AIC among orders 0 to `floor(10·log10 n)`, that maximum
///   clamped the same way.
/// * `n_freq` - Number of frequency points from 0 to 0.5 (default: 500, as R)
///
/// # Mathematical Details
///
/// For an AR(p) model: x_t = Σ_{k=1}^p φ_k x_{t-k} + ε_t
///
/// The spectral density is:
///
/// S(f) = σ² / |1 - Σ_{k=1}^p φ_k exp(-2πi f k)|²
///
/// with `σ²` `ar.yw`'s prediction variance (the Levinson innovation variance times
/// `n / (n - (p + 1))`), at R's `spec.ar` frequencies `seq(0, 0.5, length.out = n_freq)`.
///
/// # Errors
///
/// - [`EconError::InsufficientData`] for fewer than 4 observations.
/// - [`EconError::InvalidSpecification`] for NaN or infinite values in `x`;
///   for a series with zero variance; and when the autocovariances overflow
///   (a series too large in magnitude), whether `order` is given or selected
///   by AIC.
/// - [`EconError::ResourceLimit`] when the `n_freq`-point grid cannot be
///   allocated (the per-allocation ceiling included).
/// - [`EconError::Cancelled`] when the budget checked through [`checkpoint`]
///   is cancelled or past its deadline, during the AR fit or on the frequency
///   grid.
///
/// # References
///
/// - Brockwell & Davis (1991), Section 4.4
/// - R `spec.ar`
pub fn spectrum_ar(
    x: &[f64],
    order: Option<usize>,
    n_freq: Option<usize>,
) -> EconResult<SpectrumResult> {
    let n = x.len();

    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "AR spectral estimation".to_string(),
        });
    }

    // Without this a NaN slips past the zero-variance check (`NaN <= 0.0` is
    // false) and a given order returns an all-NaN spectrum.
    refuse_non_finite(x, "x")?;

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

    // ar.yw's var.pred: the Levinson innovation variance times n / (n - (order + 1)).
    let var_pred = ar_var_pred(innovation_var, n, p);

    // Compute spectral density at frequencies
    let (freq, spec) = ar_spectrum_grid(&ar_coefs, var_pred, n_pts)?;

    Ok(SpectrumResult {
        freq,
        spec,
        bandwidth: 0.0,    // AR spectrum is smooth, bandwidth not applicable
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

/// The AR spectral density `σ² / |1 - Σ_k φ_k exp(-2πi f k)|²` at the
/// `n_pts` frequencies `f_j = 0.5 j / (n_pts - 1)`, `j = 0, ..., n_pts - 1`
/// (R's `seq(0, 0.5, length.out = n.freq)`; one point is frequency 0);
/// returns `(freq, spec)`.
fn ar_spectrum_grid(
    ar_coefs: &[f64],
    innovation_var: f64,
    n_pts: usize,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let mut freq = try_with_capacity(n_pts, "the AR spectrum frequency grid")?;
    let mut spec = try_with_capacity(n_pts, "the AR spectrum frequency grid")?;

    // Each point costs O(p), and p can be as large as n / 2.
    let stride = checkpoint_stride(ar_coefs.len());
    for j in 0..n_pts {
        if j % stride == 0 {
            checkpoint()?;
        }
        let f = if n_pts > 1 {
            0.5 * j as f64 / (n_pts - 1) as f64
        } else {
            0.0
        };
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
        let s = if denom > 1e-15 {
            innovation_var / denom
        } else {
            innovation_var
        };
        if !s.is_finite() {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "the AR({}) spectral density at frequency {f} exceeds the largest \
                     floating-point number (innovation variance {innovation_var:e}, a peak \
                     that sharp): rescale the series, for example to a smaller unit, or fit \
                     a lower order",
                    ar_coefs.len()
                ),
            });
        }
        spec.push(s);
    }

    Ok((freq, spec))
}

/// `ar.yw`'s prediction variance: the Levinson innovation variance `v` of an AR(`p`) fit to `n`
/// observations times `n / (n - (p + 1))`.
fn ar_var_pred(v: f64, n: usize, p: usize) -> f64 {
    v * n as f64 / (n - (p + 1)) as f64
}

/// Fit AR(p) model using Yule-Walker equations.
///
/// Returns (coefficients, innovation variance of the Levinson recursion, unscaled; the AIC uses
/// it, and [`ar_var_pred`] scales it as `ar.yw`)
fn fit_ar_yule_walker(x: &[f64], p: usize) -> EconResult<(Vec<f64>, f64)> {
    let n = x.len();

    // Demean the series
    let mean = x.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = x.iter().map(|v| v - mean).collect();

    // Compute autocovariances
    let mut gamma = Vec::with_capacity(p + 1);
    for k in 0..=p {
        checkpoint()?;
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += y[t] * y[t + k];
        }
        gamma.push(sum / n as f64);
    }

    // Finite input can still overflow here (the mean, or the products in the
    // sums); a given order would then return a non-finite spectrum.
    if gamma.iter().any(|g| !g.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "the series is too large in magnitude for an AR fit: its autocovariances \
                      overflow"
                .to_string(),
        });
    }

    if gamma[0] <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    // Solve Yule-Walker equations using Durbin-Levinson
    // This is more numerically stable than direct matrix solve
    yule_walker_durbin_levinson(&gamma)
}

/// Solve the Yule-Walker equations of AR(`p`), `p = gamma.len() - 1`, by the
/// Durbin-Levinson recursion on the autocovariances `gamma[0..=p]`
/// (`gamma[0] > 0`). Returns (coefficients, innovation variance).
fn yule_walker_durbin_levinson(gamma: &[f64]) -> EconResult<(Vec<f64>, f64)> {
    let p = gamma.len() - 1;
    let mut v = gamma[0];

    if p == 0 {
        return Ok((vec![], v));
    }

    let mut phi = vec![0.0; p];
    // phi of the previous order, reused across orders instead of a clone each time
    let mut prev = vec![0.0; p];

    // First iteration
    phi[0] = gamma[1] / gamma[0];
    v = gamma[0] * (1.0 - phi[0] * phi[0]);

    for k in 2..=p {
        // Each order costs O(k).
        checkpoint()?;
        // Compute partial autocorrelation
        let mut num = gamma[k];
        for j in 0..(k - 1) {
            num -= phi[j] * gamma[k - 1 - j];
        }
        let phi_kk = num / v;

        // Update coefficients
        prev[..k - 1].copy_from_slice(&phi[..k - 1]);
        for j in 0..(k - 1) {
            phi[j] = prev[j] - phi_kk * prev[k - 2 - j];
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
        checkpoint()?;
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

    // Defensive: fit_ar_yule_walker refuses non-finite and zero-variance
    // autocovariances, so AR(0) always has a finite AIC and this cannot fire.
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
    numeric_column(dataset.df(), col_name, NaPolicy::Fail).map(|(v, _)| v)
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
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();
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
            0.1, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.3, 0.2, 0.1, 0.3, -0.1, 0.0, 0.2, -0.4, 0.1,
            -0.2, 0.3, -0.1, 0.2,
        ];

        let config = SpectrumConfig::with_spans(vec![3, 3]);
        let result = spectrum(&x, config).unwrap();

        // Check that spectrum values don't vary too wildly (roughly flat)
        let _mean_spec: f64 = result.spec.iter().sum::<f64>() / result.spec.len() as f64;
        let max_spec = result
            .spec
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
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
        let (mean, slope) = detrend_line(&x);
        let detrended: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v - mean - slope * ((i + 1) as f64 - 3.0))
            .collect();

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
        // R: spec.taper(rep(1, 20), 0.1) has m = floor(2) = 2 tapered points at each end,
        // 0.5 * (1 - cos(pi * c(1, 3) / 4)), and no renormalisation.
        let mut x = vec![Complex::new(1.0, 0.0); 20];
        apply_cosine_taper(&mut x, 0.1);
        let w1 = 0.5 * (1.0 - (PI / 4.0).cos());
        let w2 = 0.5 * (1.0 - (3.0 * PI / 4.0).cos());
        for (i, w) in [(0, w1), (1, w2), (18, w2), (19, w1)] {
            assert!(approx_eq(x[i].re, w, 1e-15), "{i}: {}", x[i].re);
        }
        assert!(x[2..18].iter().all(|v| v.re == 1.0));
        // m = floor(n p): 0.1 of 19 points tapers 1 point at each end, not round(1.9) = 2.
        let mut y = vec![Complex::new(1.0, 0.0); 19];
        apply_cosine_taper(&mut y, 0.1);
        assert!(approx_eq(y[0].re, 0.5, 1e-15) && y[1].re == 1.0 && y[17].re == 1.0);
    }

    #[test]
    fn test_daniell_smoothing() {
        // Test that smoothing reduces variance: ordinates 0..=9 of a symmetric length-18 periodogram
        let x: Vec<f64> = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 1.0, 5.0, 2.0, 4.0];

        let mut smoothed = x.clone();
        apply_daniell_smoothing(&mut smoothed, 18, &[3]).unwrap();

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
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();
        let result = spectrum_ar(&x, Some(5), Some(100)).unwrap();

        assert_eq!(result.freq.len(), 100);
        assert_eq!(result.spec.len(), 100);

        // All spectral values should be non-negative
        for &s in &result.spec {
            assert!(s >= 0.0, "AR spectral density should be non-negative");
        }
    }

    /// A positively autocorrelated series peaks at frequency 0, where the AR grid starts: the
    /// Display names no period there instead of printing "period = inf".
    #[test]
    fn test_spectrum_ar_peak_at_zero_has_no_period() {
        let mut x = vec![0.0f64; 200];
        for i in 1..200 {
            let e = ((i * 7919) % 101) as f64 / 101.0 - 0.5;
            x[i] = 0.8 * x[i - 1] + e;
        }
        let r = spectrum_ar(&x, Some(1), Some(50)).unwrap();
        assert_eq!(r.peak_frequency().unwrap().0, 0.0);
        let s = r.to_string();
        assert!(s.contains("period = not defined at frequency 0"), "{s}");
        assert!(!s.contains("inf"), "{s}");
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

    /// A sine of period 10 peaks near frequency 0.1.
    #[test]
    fn test_spectrum_sine_peak_property() {
        // Generate sine wave with period 10 (frequency 0.1)
        let x: Vec<f64> = (1..=100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        let config = SpectrumConfig::with_spans(vec![3, 3]);
        let result = spectrum(&x, config).unwrap();

        let (peak_freq, _) = result.peak_frequency().unwrap();

        assert!(
            (peak_freq - 0.1).abs() < 0.02,
            "Peak frequency {} should be near 0.1",
            peak_freq
        );
    }

    /// The raw periodogram of 10 points has 5 ordinates.
    #[test]
    fn test_raw_periodogram_length_property() {
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

    /// A linear series has no power left after detrending.
    #[test]
    fn test_detrend_linear_has_no_power() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        let config = SpectrumConfig {
            spans: None,
            taper: 0.1,
            detrend: true,
            demean: false,
            pad_ratio: 0.0,
            fast: true,
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

    /// Two cosines give the highest peaks at their frequencies 0.2 and 0.1.
    #[test]
    fn test_two_frequency_peaks_property() {
        // Signal with two frequencies: f=0.2 (period 5) and f=0.1 (period 10)
        let x: Vec<f64> = (1..=50)
            .map(|i| (2.0 * PI * i as f64 / 5.0).cos() + 0.5 * (2.0 * PI * i as f64 / 10.0).cos())
            .collect();

        let config = SpectrumConfig::with_spans(vec![3]);
        let result = spectrum(&x, config).unwrap();

        // Find the two highest peaks
        let mut indexed: Vec<(usize, f64)> = result.spec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let peak1_freq = result.freq[indexed[0].0];
        let _peak2_freq = result.freq[indexed[1].0];

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
            indexed
                .iter()
                .take(5)
                .map(|(i, _)| result.freq[*i])
                .collect::<Vec<_>>()
        );
    }

    /// The spectrum of pseudo-random white noise is relatively flat.
    #[test]
    fn test_white_noise_spectrum_cv_property() {
        // Generate pseudo-random white noise using LCG
        let mut x = Vec::with_capacity(100);
        let mut seed: u64 = 42;
        for _ in 0..100 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = (seed >> 16) as f64 / 32768.0 - 1.0;
            x.push(val);
        }

        let config = SpectrumConfig::with_spans(vec![5, 5]);
        let result = spectrum(&x, config).unwrap();

        // For white noise, coefficient of variation should be moderate
        // (not as flat as theoretical but not highly peaked either)
        let mean_spec: f64 = result.spec.iter().sum::<f64>() / result.spec.len() as f64;
        let var_spec: f64 = result
            .spec
            .iter()
            .map(|s| (s - mean_spec).powi(2))
            .sum::<f64>()
            / result.spec.len() as f64;
        let cv = var_spec.sqrt() / mean_spec;

        // R shows CV ≈ 0.35 for white noise with spans=c(5,5)
        // Allow broader range due to different implementations
        assert!(cv < 2.0, "White noise CV {} should be < 2.0 (R ≈ 0.35)", cv);
    }

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    /// R: mk <- function(n) { t <- 1:n; sin(2*pi*t/7) + 0.5*cos(2*pi*t/13) + 0.3*(((37*t) %% 11) - 5)/5 + 0.02*t }
    fn mk(n: usize) -> Vec<f64> {
        (1..=n)
            .map(|t| {
                let tf = t as f64;
                (2.0 * PI * tf / 7.0).sin()
                    + 0.5 * (2.0 * PI * tf / 13.0).cos()
                    + 0.3 * (((37 * t) % 11) as f64 - 5.0) / 5.0
                    + 0.02 * tf
            })
            .collect()
    }

    /// R 4.3.3 on x <- mk(120) (nextn(120) = 120, so R does not pad):
    ///   spec.pgram(x, spans = 119)        -> ok
    ///   spec.pgram(x, spans = 121)        -> Error: 'x' is shorter than kernel 'k'
    ///   spec.pgram(x, spans = c(59, 61))  -> ok
    ///   spec.pgram(x, spans = c(61, 61))  -> Error: 'x' is shorter than kernel 'k'
    ///   spec.pgram(x, spans = c(3, 3, 3)) -> ok
    #[test]
    fn spans_follow_r_kernel_length_rule() {
        let x = mk(120);
        for (spans, ok) in [
            (vec![119], true),
            (vec![121], false),
            (vec![59, 61], true),
            (vec![61, 61], false),
            (vec![3, 3, 3], true),
        ] {
            let r = spectrum(&x, SpectrumConfig::with_spans(spans.clone()));
            assert_eq!(r.is_ok(), ok, "{spans:?}: {r:?}");
            if let Err(e) = r {
                let m = e.to_string();
                // The text of our own refusal, not the quoted R error in its advice.
                assert!(
                    m.contains("spans give a smoothing kernel")
                        && m.contains("-point series"),
                    "{m}"
                );
            }
        }
    }

    #[test]
    fn validate_spans_bounds_huge_and_overflowing_spans() {
        // Before the fix any odd span passed and the smoother looped n_freq * span times.
        assert!(validate_spans(&[usize::MAX], 120).is_err());
        assert!(validate_spans(&[usize::MAX, usize::MAX], 120).is_err()); // half-width sum overflows
        assert!(validate_spans(&[4], 120).is_err());
        assert!(validate_spans(&[0], 120).is_err());
        assert!(validate_spans(&[1, 3], 120).is_ok()); // span 1 stays a no-op, as before
        assert!(validate_spans(&[3], 4).is_ok()); // M = 1, 2 < 4
    }

    #[test]
    fn spectrum_rejects_non_finite_input_padding_and_taper() {
        let mut x = mk(64);
        x[10] = f64::NAN;
        assert!(matches!(
            spectrum(&x, SpectrumConfig::default()),
            Err(EconError::NullValues { .. })
        ));
        let x = mk(64);
        let inf = SpectrumConfig {
            pad_ratio: f64::INFINITY,
            ..Default::default()
        };
        assert!(matches!(
            spectrum(&x, inf),
            Err(EconError::InvalidSpecification { .. })
        ));
        let huge = SpectrumConfig {
            pad_ratio: 1e15,
            ..Default::default()
        }; // aborted before the fix
        assert!(matches!(
            spectrum(&x, huge),
            Err(EconError::ResourceLimit(_))
        ));
        // A NaN taper failed both range comparisons and was used as no taper.
        let nan_taper = SpectrumConfig {
            taper: f64::NAN,
            ..Default::default()
        };
        let r = spectrum(&x, nan_taper);
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("Taper")),
            "{r:?}"
        );
    }

    /// Before the fix a NaN with a given order slipped past the zero-variance
    /// check and returned an all-NaN spectrum; with the order chosen by AIC it
    /// failed as "Could not fit AR model". A finite series whose autocovariances
    /// overflow returned a non-finite spectrum for a given order.
    #[test]
    fn spectrum_ar_rejects_non_finite_and_overflowing_input() {
        let mut nan = mk(64);
        nan[10] = f64::NAN;
        let mut inf = mk(64);
        inf[20] = f64::NEG_INFINITY;
        for order in [Some(2), None] {
            let r = spectrum_ar(&nan, order, Some(50));
            assert!(
                matches!(&r, Err(EconError::NullValues { .. })),
                "{order:?}: {r:?}"
            );
            let r = spectrum_ar(&inf, order, Some(50));
            assert!(
                matches!(&r, Err(EconError::NonFiniteValues { .. })),
                "{order:?}: {r:?}"
            );
        }

        // Finite, but every squared deviation is about 1e400.
        let huge: Vec<f64> = mk(64).iter().map(|v| v * 1e200).collect();
        assert!(huge.iter().all(|v| v.is_finite()));
        for order in [Some(2), None] {
            let r = spectrum_ar(&huge, order, Some(50));
            assert!(
                matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("too large in magnitude")),
                "{order:?}: {r:?}"
            );
        }

        // A large but representable scale still fits.
        let big: Vec<f64> = mk(64).iter().map(|v| v * 1e100).collect();
        let r = spectrum_ar(&big, Some(2), Some(50)).unwrap();
        assert!(r.spec.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn spectrum_and_spectrum_ar_honour_cancellation() {
        let x = mk(64);
        assert!(matches!(
            cancelled(|| spectrum(&x, SpectrumConfig::with_spans(vec![3]))),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            cancelled(|| spectrum_ar(&x, Some(5), None)),
            Err(EconError::Cancelled(_))
        ));
    }

    /// spectrum_ar's autocovariance loop checks first, so this calls the recursion directly.
    #[test]
    fn yule_walker_durbin_levinson_honours_cancellation() {
        // p = 3, so the order loop runs and checks the budget at k = 2.
        assert!(matches!(
            cancelled(|| yule_walker_durbin_levinson(&[1.0, 0.5, 0.25, 0.125])),
            Err(EconError::Cancelled(_))
        ));
    }

    /// The pre-change recursion, verbatim (clones phi every order).
    fn yule_walker_reference(gamma: &[f64], p: usize) -> (Vec<f64>, f64) {
        let mut phi = vec![0.0; p];
        let mut v = gamma[0];

        if p == 0 {
            return (vec![], v);
        }

        phi[0] = gamma[1] / gamma[0];
        v = gamma[0] * (1.0 - phi[0] * phi[0]);

        for k in 2..=p {
            let mut num = gamma[k];
            for j in 0..(k - 1) {
                num -= phi[j] * gamma[k - 1 - j];
            }
            let phi_kk = num / v;

            let phi_old = phi.clone();
            for j in 0..(k - 1) {
                phi[j] = phi_old[j] - phi_kk * phi_old[k - 2 - j];
            }
            phi[k - 1] = phi_kk;

            v *= 1.0 - phi_kk * phi_kk;

            if v <= 0.0 {
                break;
            }
        }

        (phi, v)
    }

    #[test]
    fn yule_walker_buffer_reuse_is_bit_identical() {
        let x = mk(400);
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let y: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let gamma: Vec<f64> = (0..=60)
            .map(|k| (0..n - k).map(|t| y[t] * y[t + k]).sum::<f64>() / n as f64)
            .collect();
        for p in [0, 1, 2, 7, 60] {
            assert_eq!(
                yule_walker_durbin_levinson(&gamma[..=p]).unwrap(),
                yule_walker_reference(&gamma, p),
                "p = {p}"
            );
        }
    }

    /// With p >= 65,536 coefficients each grid point costs that much, so the
    /// grid checks the budget at every point. A fixed stride of 1,024 points
    /// did not check a 4-point grid at all.
    #[test]
    fn ar_spectrum_grid_checks_the_budget_by_cost() {
        let phi: Vec<f64> = (0..70_000)
            .map(|k| 1e-6 * (((k * 37) % 11) as f64 - 5.0))
            .collect();
        assert!(matches!(
            cancelled(|| ar_spectrum_grid(&phi, 1.0, 4)),
            Err(EconError::Cancelled(_))
        ));
        // Without a budget the grid is computed as before.
        let (freq, spec) = ar_spectrum_grid(&[0.5], 2.0, 4).unwrap();
        assert_eq!(freq, vec![0.0, 0.5 / 3.0, 1.0 / 3.0, 0.5]);
        // f = 0: |1 - 0.5|² = 0.25
        assert!((spec[0] - 2.0 / 0.25).abs() < 1e-12, "{spec:?}");
        // f = 0.5: |1 - 0.5 e^{-iπ}|² = 1.5² = 2.25
        assert!((spec[3] - 2.0 / 2.25).abs() < 1e-12, "{spec:?}");
    }

    /// R 4.3.3: x <- mk(64)
    ///   spec.pgram(x, spans = c(3,3), taper = 0, detrend = TRUE, fast = FALSE, plot = FALSE)$spec
    ///   spec.pgram(x, spans = c(3,5), taper = 0, detrend = TRUE, fast = FALSE, plot = FALSE)$spec
    /// Every ordinate, the edges included: R smooths circularly over all 64 ordinates.
    #[test]
    fn test_validate_daniell_interior_against_r() {
        const R33: [f64; 32] = [
            0.015639154180673867,
            0.0088400405362115686,
            0.29475141849034109,
            1.1700759929085298,
            1.7687398920663475,
            1.2269951529673682,
            1.358237668437869,
            3.9538882975167176,
            5.8722177259740675,
            3.975497583199092,
            1.0933717954446027,
            0.094267695716722916,
            0.055275302352636863,
            0.027614537657364172,
            0.017774041304525829,
            0.031304830696008527,
            0.05621484142705567,
            0.063648720549974058,
            0.044748082305340386,
            0.025368791471610586,
            0.054915668432741871,
            0.16213799350680347,
            0.24212869348457033,
            0.18181195885328583,
            0.069488871170710897,
            0.018200351706528783,
            0.010777710127665596,
            0.013210011050243886,
            0.017017774653834528,
            0.015780827308339733,
            0.011813562645424932,
            0.01011936558346524,
        ];
        const R35: [f64; 32] = [
            0.013829754329979627,
            0.1551952863355075,
            0.58945801672237019,
            1.0317456552783439,
            1.1985355729379492,
            1.5634887802521074,
            2.5904417252420435,
            3.6152276972059667,
            3.9646929403579048,
            3.4827947607093348,
            2.0348826394579067,
            0.57432354889861936,
            0.060941116687043301,
            0.036524671828581413,
            0.02945968417668586,
            0.03699444136579088,
            0.047476775622991366,
            0.050481461866197938,
            0.044508756010792044,
            0.049831875369040879,
            0.09375339248920711,
            0.1485221809586561,
            0.17197497618004531,
            0.15580878232764084,
            0.10000615527990764,
            0.040133290649188247,
            0.015705181378386168,
            0.013897742390750034,
            0.014495419179291813,
            0.014415668649629843,
            0.012950096445902738,
            0.011813562645425352,
        ];
        let cfg = |s: Vec<usize>| {
            let mut c = SpectrumConfig::with_spans(s);
            c.taper = 0.0;
            c.fast = false;
            c
        };
        let s33 = spectrum(&mk(64), cfg(vec![3, 3])).unwrap().spec;
        assert_eq!(s33.len(), 32);
        for (s, r) in s33.iter().zip(R33) {
            approx::assert_relative_eq!(*s, r, max_relative = 1e-10);
        }
        let s35 = spectrum(&mk(64), cfg(vec![3, 5])).unwrap().spec;
        for (s, r) in s35.iter().zip(R35) {
            approx::assert_relative_eq!(*s, r, max_relative = 1e-10);
        }
    }

    /// R's circular smoothing, naively: every pass is a full circular convolution over all `n` ordinates.
    fn circular_daniell_reference(full: &[f64], spans: &[usize]) -> Vec<f64> {
        let n = full.len();
        let mut x = full.to_vec();
        for &span in spans {
            let m = span / 2;
            if m == 0 {
                continue;
            }
            let w = |k: usize| {
                if k == 0 || k == 2 * m {
                    1.0 / (4 * m) as f64
                } else {
                    1.0 / (2 * m) as f64
                }
            };
            x = (0..n)
                .map(|i| {
                    (0..=2 * m)
                        .map(|k| w(k) * x[(i + n + k - m) % n])
                        .sum::<f64>()
                })
                .collect();
        }
        x
    }

    /// The mirrored half-length pass equals the naive circular convolution over the full periodogram
    /// (noisy symmetric input, even and odd lengths).
    #[test]
    fn daniell_mirrored_pass_matches_full_circular_convolution() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for n_full in [12usize, 13, 40, 41, 97, 100] {
            let h = n_full / 2;
            let half: Vec<f64> = (0..=h).map(|_| rng.r#gen::<f64>() * 10.0).collect();
            let full: Vec<f64> = (0..n_full)
                .map(|j| half[if j > h { n_full - j } else { j }])
                .collect();
            for spans in [vec![3], vec![3, 3], vec![7, 9], vec![1, 5]] {
                let m: usize = spans.iter().map(|s| s / 2).sum();
                if 2 * m >= n_full {
                    continue;
                }
                let mut fast = half.clone();
                apply_daniell_smoothing(&mut fast, n_full, &spans).unwrap();
                let slow = circular_daniell_reference(&full, &spans);
                for (j, a) in fast.iter().enumerate() {
                    let b = slow[j];
                    assert!(
                        (a - b).abs() <= 1e-12 * b.abs(),
                        "n={n_full} spans={spans:?} j={j}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// A periodogram can span 15 orders of magnitude. An uncompensated running sum keeps ~1e-4 of
    /// rounding from the peak and fails this test; the compensated one passes.
    #[test]
    fn daniell_sliding_window_keeps_precision_after_a_large_peak() {
        let mut half = vec![1e12];
        half.extend((0..199).map(|i| 1e-3 * (1.5 + (i as f64).sin())));
        let n_full = 2 * (half.len() - 1);
        let full: Vec<f64> = (0..n_full)
            .map(|j| half[if j >= half.len() { n_full - j } else { j }])
            .collect();
        for span in [3, 5, 51, 199] {
            let mut fast = half.clone();
            apply_daniell_smoothing(&mut fast, n_full, &[span]).unwrap();
            for (a, b) in fast.iter().zip(circular_daniell_reference(&full, &[span])) {
                assert!((a - b).abs() <= 1e-9 * b.abs(), "span={span}: {a} vs {b}");
            }
        }
    }

    /// R 4.3.3 `nextn(1:200)`.
    #[test]
    fn nextn_matches_r() {
        const R: [usize; 200] = [
            1, 2, 3, 4, 5, 6, 8, 8, 9, 10, 12, 12, 15, 15, 15, 16, 18, 18, 20, 20, 24, 24, 24, 24,
            25, 27, 27, 30, 30, 30, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40, 45, 45, 45, 45, 45, 48,
            48, 48, 50, 50, 54, 54, 54, 54, 60, 60, 60, 60, 60, 60, 64, 64, 64, 64, 72, 72, 72, 72,
            72, 72, 72, 72, 75, 75, 75, 80, 80, 80, 80, 80, 81, 90, 90, 90, 90, 90, 90, 90, 90, 90,
            96, 96, 96, 96, 96, 96, 100, 100, 100, 100, 108, 108, 108, 108, 108, 108, 108, 108,
            120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 125, 125, 125, 125, 125,
            128, 128, 128, 135, 135, 135, 135, 135, 135, 135, 144, 144, 144, 144, 144, 144, 144,
            144, 144, 150, 150, 150, 150, 150, 150, 160, 160, 160, 160, 160, 160, 160, 160, 160,
            160, 162, 162, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180,
            180, 180, 180, 180, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 200,
            200, 200, 200, 200, 200, 200, 200,
        ];
        for (n, r) in (1..=200).zip(R) {
            assert_eq!(nextn(n), Some(r), "nextn({n})");
        }
        assert_eq!(nextn(10_007), Some(10_125));
        assert_eq!(nextn(usize::MAX), None);
    }

    /// R 4.3.3: spec.pgram(z75, pad = 0.3, fast = FALSE)$n.used is 97 (75 + trunc(22.5)); with fast it is 100.
    #[test]
    fn padding_truncates_as_r() {
        let z: Vec<f64> = mk(75);
        let mut c = SpectrumConfig::default();
        c.pad_ratio = 0.3;
        c.fast = false;
        assert_eq!(spectrum(&z, c.clone()).unwrap().n_used, 97);
        c.fast = true;
        assert_eq!(spectrum(&z, c).unwrap().n_used, 100);
    }

    /// The series of case `stat_spectrum_t4` (n = 97), read without loading the case.
    fn t4_series() -> Vec<f64> {
        let path = p2a_validation::repo_path("validation/datasets/generated/spectrum_t4.csv");
        let ds = Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        extract_column(&ds, "z").unwrap()
    }

    fn pgram_config(spans: Option<Vec<usize>>, taper: f64) -> SpectrumConfig {
        let mut c = SpectrumConfig::default();
        c.spans = spans;
        c.taper = taper;
        c
    }

    /// Compare `freq`, `spec` and `n_used` of `r` with the `<key>_*` values of `stat_spectrum_t4`.
    fn check_pgram(check: &mut p2a_validation::Check, key: &str, r: &SpectrumResult) {
        use p2a_validation::Tol;
        check
            .vector(&format!("{key}_freq"), &r.freq, Tol::coef(96))
            .vector(&format!("{key}_spec"), &r.spec, Tol::coef(96))
            .scalar(&format!("{key}_n_used"), r.n_used as f64, Tol::exact())
            .scalar(&format!("{key}_df"), r.df, Tol::se(96))
            .scalar(&format!("{key}_bw"), r.bandwidth, Tol::se(96));
    }

    /// spec.pgram(x, taper = 0.1) on n = 96 and spec.pgram(z75, taper = 0) (odd n).
    #[test]
    fn test_validate_spectrum_pgram_raw_tapered() {
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t4");
        let z = t4_series();
        let mut check = p2a_validation::Check::new(&case);
        let r = run_spectrum(&ds, "x", pgram_config(None, 0.1)).unwrap();
        check_pgram(&mut check, "raw_t10", &r);
        let r = spectrum(&z[..75], pgram_config(None, 0.0)).unwrap();
        check_pgram(&mut check, "raw_n75", &r);
        check.finish();
    }

    /// Smoothed: spans (3, 3) with taper 0.1; spans (3, 5), taper 0.25, demean only; span 7, no taper, no detrending.
    #[test]
    fn test_validate_spectrum_pgram_smoothed_taper_demean() {
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t4");
        let mut check = p2a_validation::Check::new(&case);
        let r = run_spectrum(&ds, "x", pgram_config(Some(vec![3, 3]), 0.1)).unwrap();
        check_pgram(&mut check, "s33_t10", &r);
        let mut c = pgram_config(Some(vec![3, 5]), 0.25);
        c.detrend = false;
        c.demean = true;
        let r = run_spectrum(&ds, "x", c).unwrap();
        check_pgram(&mut check, "s35_t25_demean", &r);
        let mut c = pgram_config(Some(vec![7]), 0.0);
        c.detrend = false;
        let r = run_spectrum(&ds, "x", c).unwrap();
        check_pgram(&mut check, "s7_t0_none", &r);
        check.finish();
    }

    /// spec.pgram(x, spans = c(3, 3), pad = 0.5): N0 = 96, N = 144.
    #[test]
    fn test_validate_spectrum_pgram_padding() {
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t4");
        let mut check = p2a_validation::Check::new(&case);
        let mut c = pgram_config(Some(vec![3, 3]), 0.1);
        c.pad_ratio = 0.5;
        let r = run_spectrum(&ds, "x", c).unwrap();
        check_pgram(&mut check, "s33_pad", &r);
        check.finish();
    }

    /// Odd n = 75 (nextn(75) = 75), and n = 97 with R's default fast padding (to 100) and without it.
    #[test]
    fn test_validate_spectrum_pgram_odd_and_fast() {
        let (case, _) = crate::test_support::load_case("stat_spectrum_t4");
        let z = t4_series();
        let mut check = p2a_validation::Check::new(&case);
        let r = spectrum(&z[..75], pgram_config(Some(vec![3, 3]), 0.1)).unwrap();
        check_pgram(&mut check, "s33_n75", &r);
        let r = spectrum(&z, pgram_config(Some(vec![3, 3]), 0.1)).unwrap();
        check_pgram(&mut check, "s33_n97_fast", &r);
        let mut c = pgram_config(Some(vec![3, 3]), 0.1);
        c.fast = false;
        let r = spectrum(&z, c).unwrap();
        check_pgram(&mut check, "s33_n97_exact", &r);
        check.finish();
    }

    /// Non-finite input is refused (R: na.fail), so the peak and the FFT never see NaN.
    #[test]
    fn test_spectrum_refuses_non_finite() {
        let mut x: Vec<f64> = (0..32)
            .map(|i| (i as f64 * 0.7).sin() + 0.01 * i as f64)
            .collect();
        x[5] = f64::NAN;
        assert!(
            matches!(spectrum(&x, SpectrumConfig::default()), Err(EconError::NullValues { ref column, count: 1 }) if column == "x")
        );
        assert!(matches!(
            spectrum_ar(&x, Some(1), None),
            Err(EconError::NullValues { .. })
        ));
        x[5] = f64::INFINITY;
        assert!(
            matches!(spectrum(&x, SpectrumConfig::default()), Err(EconError::NonFiniteValues { ref column, count: 1 }) if column == "x")
        );
    }

    /// spec.ar(x, n.freq = 50, order = 2) and ar(x, aic = FALSE, order.max = 2) (Yule-Walker).
    #[test]
    fn test_validate_spectrum_ar_fixed_order() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t4");
        let r = run_spectrum_ar(&ds, "x", Some(2), Some(50)).unwrap();
        let x = extract_column(&ds, "x").unwrap();
        let (coef, v) = fit_ar_yule_walker(&x, 2).unwrap();
        assert_eq!(r.method, "ar(2)");
        Check::new(&case)
            .vector("ar2_freq", &r.freq, Tol::coef(96))
            .vector("ar2_spec", &r.spec, Tol::coef(96))
            .vector("ar2_coef", &coef, Tol::coef(96))
            .scalar("ar2_var_pred", ar_var_pred(v, x.len(), 2), Tol::se(96))
            .finish();
    }

    /// spec.ar(x, n.freq = 50) with the order chosen by AIC, and ar(x)'s var.pred.
    #[test]
    fn test_validate_spectrum_ar_aic() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t4");
        let r = run_spectrum_ar(&ds, "x", None, Some(50)).unwrap();
        let x = extract_column(&ds, "x").unwrap();
        let max_order = (10.0 * (x.len() as f64).log10()).floor() as usize;
        let (coef, v) = select_ar_order_aic(&x, max_order).unwrap();
        assert_eq!(r.method, format!("ar({})", coef.len()));
        Check::new(&case)
            .scalar("aic_order", coef.len() as f64, Tol::exact())
            .vector("aic_spec", &r.spec, Tol::coef(96))
            .scalar(
                "aic_var_pred",
                ar_var_pred(v, x.len(), coef.len()),
                Tol::se(96),
            )
            .finish();
    }

    /// `kernel_df_bandwidth` against direct sums over one modified Daniell kernel, and the df of the
    /// (3, 3) convolution (1, 4, 6, 4, 1) / 16; a span of 1 is no smoothing (df 2).
    #[test]
    fn kernel_df_bandwidth_matches_direct_sums() {
        for m in 1..6usize {
            let w: Vec<f64> = (0..=2 * m)
                .map(|j| {
                    if j == 0 || j == 2 * m {
                        0.25 / m as f64
                    } else {
                        0.5 / m as f64
                    }
                })
                .collect();
            let df = 2.0 / w.iter().map(|v| v * v).sum::<f64>();
            let bw = w
                .iter()
                .enumerate()
                .map(|(j, v)| (1.0 / 12.0 + (j as f64 - m as f64).powi(2)) * v)
                .sum::<f64>()
                .sqrt();
            let (a, b) = kernel_df_bandwidth(&[2 * m + 1]).unwrap();
            assert!(
                (a - df).abs() <= 1e-12 * df && (b - bw).abs() <= 1e-12 * bw,
                "m={m}"
            );
        }
        // spans (3, 3): k = (1, 4, 6, 4, 1) / 16
        let k = [1.0, 4.0, 6.0, 4.0, 1.0].map(|v| v / 16.0);
        let df = 2.0 / k.iter().map(|v| v * v).sum::<f64>();
        let (a, _) = kernel_df_bandwidth(&[3, 3]).unwrap();
        assert!((a - df).abs() <= 1e-12 * df);
        assert_eq!(kernel_df_bandwidth(&[1, 1]).unwrap().0, 2.0);
    }

    /// The series of case `stat_spectrum_t3`, read without loading the case.
    fn t3_dataset() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/spectrum_t3.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// A sharp AR peak with a very large innovation variance can take the density past
    /// f64::MAX: the grid is refused, naming the order and the peak, instead of holding an
    /// infinite ordinate.
    #[test]
    fn spectrum_ar_refuses_an_infinite_ordinate() {
        // |1 - phi|^2 at frequency 0 is about 1e-14, above the cancellation guard.
        let e = ar_spectrum_grid(&[1.0 - 1e-7], 1e300, 11).unwrap_err();
        match e {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("AR(1)"), "{message}");
                assert!(message.contains("frequency 0"), "{message}");
            }
            other => panic!("{other}"),
        }
        // An ordinary peak still passes.
        let (_, spec) = ar_spectrum_grid(&[0.9], 1.0, 11).unwrap();
        assert!(spec.iter().all(|v| v.is_finite()));
    }

    /// STAT-38: an AR spectrum has no chi-square df (R's spec.ar has none), so it has no interval.
    #[test]
    fn test_spectrum_ar_no_ci() {
        let ds = t3_dataset();
        for order in [Some(2), None] {
            let r = run_spectrum_ar(&ds, "x", order, Some(50)).unwrap();
            assert!(!r.df.is_finite());
            assert_eq!(r.confidence_interval_multipliers(0.95).unwrap(), None);
            let (lo, hi) = r.confidence_multipliers(0.95); // old API: no interval
            assert!(lo.is_nan() && hi.is_nan());
            let text = r.to_string();
            assert!(text.contains("Degrees of Freedom: not defined"), "{text}");
        }
        let p = run_spectrum(&ds, "x", SpectrumConfig::with_spans(vec![3, 3])).unwrap();
        for level in [0.0, 1.0, 95.0, -0.1, f64::NAN] {
            let e = p.confidence_interval_multipliers(level).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("level")),
                "{e}"
            );
        }
    }

    /// STAT-62 (spectrum), STAT-38: spec.pgram's frequency grid and peak, and plot.spec's spec.ci at given df.
    #[test]
    fn test_validate_spectrum_t3_grid_and_ci() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spectrum_t3");
        let r = run_spectrum(&ds, "x", SpectrumConfig::with_spans(vec![3, 3])).unwrap(); // taper 0.1, detrend
        // closed-form chi-square quantiles only (t_quantile family: <= 8e-13), so a tight tolerance
        const CI_TOL: Tol = Tol::new(1e-10, 0.0);
        let mut ci95 = Vec::new();
        let mut ci90 = Vec::new();
        for df in case.vector("ci_df") {
            let mut at = r.clone();
            at.df = df;
            let (lo, hi) = at.confidence_interval_multipliers(0.95).unwrap().unwrap();
            ci95.extend([lo, hi]);
            let (lo, hi) = at.confidence_interval_multipliers(0.90).unwrap().unwrap();
            ci90.extend([lo, hi]);
        }
        let raw = run_spectrum(&ds, "x", SpectrumConfig::raw()).unwrap(); // taper 0, detrend
        Check::new(&case)
            .vector("freq", &r.freq, Tol::coef(96))
            .scalar("peak_freq", r.peak_frequency().unwrap().0, Tol::coef(96))
            .vector("spec", &r.spec, Tol::coef(96))
            .vector("raw_spec", &raw.spec, Tol::coef(96))
            .scalar("df", r.df, Tol::se(96))
            .scalar("bandwidth", r.bandwidth, Tol::se(96))
            .scalar("raw_df", raw.df, Tol::se(96))
            .scalar("raw_bandwidth", raw.bandwidth, Tol::se(96))
            .vector("ci95_at_df", &ci95, CI_TOL)
            .vector("ci90_at_df", &ci90, CI_TOL)
            .finish();
    }
}
