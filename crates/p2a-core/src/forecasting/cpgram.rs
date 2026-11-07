//! Cumulative Periodogram for Time Series Diagnostics.
//!
//! The cumulative periodogram is a diagnostic tool for time series analysis that plots
//! the cumulative sum of the periodogram ordinates against frequency. For white noise,
//! the cumulative periodogram follows a uniform distribution, and confidence bands
//! based on the Kolmogorov-Smirnov distribution can detect departures from white noise.
//!
//! # Algorithm
//!
//! The body of R's `stats::cpgram`, which plots and returns nothing:
//!
//! 1. Centre the series, then taper with `spec.taper`: with m = floor(n p), the first and
//!    last m values are multiplied by 0.5 (1 - cos(pi (2i - 1) / (2m))), i = 1..m.
//! 2. The periodogram `|X_k|^2 / n` at the frequencies k / n, k = 1..=mp, where floor(n / 2) + 1
//!    points are R's set with (0, 0), and the last is dropped when that count is even.
//! 3. The cumulative periodogram C_k, normalised to end at 1, against the line 2 f.
//!
//! The band half-width is R's 1.358 / q with q = sqrt(mp) + 0.12 + 0.11 / sqrt(mp), the 5 % point
//! of Stephens's modified Kolmogorov statistic. The series is white noise when
//! max_k |C_k - 2 f_k| is within the band, and `ks_statistic` is that maximum times q, with the
//! asymptotic Kolmogorov survival function as `ks_p_value`.
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

#[cfg(feature = "spectral-analysis")]
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[cfg(any(test, not(feature = "spectral-analysis")))]
use crate::resource::checkpoint;
use crate::resource::try_vec;
#[cfg(feature = "spectral-analysis")]
use crate::resource::try_with_capacity;

/// Result of cumulative periodogram computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpgramResult {
    /// Frequencies k / n at which the cumulative periodogram is evaluated (R's set without 0)
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
    /// Stephens-modified Kolmogorov statistic D (sqrt(mp) + 0.12 + 0.11 / sqrt(mp)), 5 % point 1.358
    pub ks_statistic: f64,
    /// Asymptotic Kolmogorov p-value of `ks_statistic`
    pub ks_p_value: f64,
    /// R's band half-width 1.358 / (sqrt(mp) + 0.12 + 0.11 / sqrt(mp)), mp = freq.len()
    #[serde(default)]
    pub critical_value: f64,
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
/// # Errors
/// Fails on fewer than 4 observations, a non-finite value, a taper outside [0, 0.5], and a series
/// without variation once tapered.
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

    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "cpgram needs a finite series (non-finite value at position {i})"
        ));
    }

    let taper_prop = taper.unwrap_or(0.1);
    if !(0.0..=0.5).contains(&taper_prop) {
        return Err(format!(
            "taper must be between 0 and 0.5 (R spec.taper), got {taper_prop}"
        ));
    }

    // Centre BEFORE tapering, or the taper turns the mean into a low-frequency component.
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut tapered = try_vec(n, 0.0, "the cpgram series").map_err(|e| e.to_string())?;
    for (t, v) in tapered.iter_mut().zip(x) {
        *t = v - mean;
    }
    apply_taper(&mut tapered, taper_prop);

    // R's frequency set: floor(n / 2) + 1 points with (0, 0), the last dropped if that is even.
    let mut pgram = compute_periodogram(&tapered)?;
    let len = n / 2 + 1;
    let mp = if len.is_multiple_of(2) {
        len - 2
    } else {
        len - 1
    };
    pgram.truncate(mp);

    let total: f64 = pgram.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err("the tapered series has no variation".to_string());
    }
    let mut cumsum = Vec::with_capacity(mp);
    let mut running_sum = 0.0;
    for &p in &pgram {
        running_sum += p;
        cumsum.push(running_sum / total);
    }

    let freq: Vec<f64> = (1..=mp).map(|k| k as f64 / n as f64).collect();

    let sqrt_mp = (mp as f64).sqrt();
    let q = sqrt_mp + 0.12 + 0.11 / sqrt_mp;
    let critical_value = 1.358 / q;

    let upper_ci: Vec<f64> = freq
        .iter()
        .map(|&f| (f * 2.0 + critical_value).min(1.0))
        .collect();
    let lower_ci: Vec<f64> = freq
        .iter()
        .map(|&f| (f * 2.0 - critical_value).max(0.0))
        .collect();

    let max_deviation = cumsum
        .iter()
        .zip(&freq)
        .map(|(c, f)| (c - f / 0.5).abs())
        .fold(0.0, f64::max);

    let ks_statistic = max_deviation * q;
    let ks_p_value = kolmogorov_sf(ks_statistic);
    let is_white_noise = max_deviation <= critical_value;

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
        critical_value,
    })
}

/// R's `spec.taper`: a split cosine bell over the first and last floor(n p) values.
fn apply_taper(x: &mut [f64], taper: f64) {
    let n = x.len();
    let m = (n as f64 * taper).floor() as usize;
    for i in 1..=m {
        let w = 0.5 * (1.0 - (PI * (2 * i - 1) as f64 / (2 * m) as f64).cos());
        x[i - 1] *= w;
        x[n - i] *= w;
    }
}

/// Compute the periodogram ordinates `|X_k|² / n` at the Fourier frequencies
/// `k = 1..=n/2` of the centred, tapered series, with an FFT (O(n log n)) when the
/// `spectral-analysis` feature is enabled and a direct DFT otherwise.
fn compute_periodogram(x: &[f64]) -> Result<Vec<f64>, String> {
    #[cfg(feature = "spectral-analysis")]
    {
        periodogram_fft(x)
    }
    #[cfg(not(feature = "spectral-analysis"))]
    {
        periodogram_direct(x)
    }
}

/// Periodogram via rustfft.
#[cfg(feature = "spectral-analysis")]
fn periodogram_fft(x: &[f64]) -> Result<Vec<f64>, String> {
    let n = x.len();
    let mut buf: Vec<Complex<f64>> =
        try_with_capacity(n, "the periodogram FFT buffer").map_err(|e| e.to_string())?;
    buf.extend(x.iter().map(|&v| Complex::new(v, 0.0)));

    FftPlanner::new().plan_fft_forward(n).process(&mut buf);

    Ok(buf
        .iter()
        .skip(1)
        .take(n / 2)
        .map(|c| (c.re * c.re + c.im * c.im) / n as f64)
        .collect())
}

/// Periodogram via a direct DFT (O(n²)): the fallback for builds without
/// `spectral-analysis`, and the reference in tests.
#[cfg(any(test, not(feature = "spectral-analysis")))]
fn periodogram_direct(x: &[f64]) -> Result<Vec<f64>, String> {
    let n = x.len();
    let n_freq = n / 2; // Number of Fourier frequencies (excluding DC and Nyquist)

    let mut pgram = Vec::with_capacity(n_freq);

    for k in 1..=n_freq {
        checkpoint().map_err(|e| e.to_string())?;
        let freq = 2.0 * PI * k as f64 / n as f64;

        // Compute Fourier coefficients
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (t, &xt) in x.iter().enumerate() {
            cos_sum += xt * (freq * t as f64).cos();
            sin_sum += xt * (freq * t as f64).sin();
        }

        // Periodogram ordinate: (cos_sum^2 + sin_sum^2) / n
        let intensity = (cos_sum * cos_sum + sin_sum * sin_sum) / n as f64;
        pgram.push(intensity);
    }

    Ok(pgram)
}

/// Survival function of the asymptotic Kolmogorov distribution: P(K > x).
fn kolmogorov_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let p = if x < 1.0 {
        // 1 - F(x), the theta form of F, which converges fast for small x.
        let mut sum = 0.0;
        let mut k = 1.0_f64;
        loop {
            let term = (-k * k * PI * PI / (8.0 * x * x)).exp();
            sum += term;
            if term < 1e-17 * sum || term == 0.0 {
                break;
            }
            k += 2.0;
        }
        1.0 - (2.0 * PI).sqrt() / x * sum
    } else {
        // The alternating series, which keeps its digits in the tail.
        let mut sum = 0.0;
        for k in 1..=100 {
            let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
            let term = (-2.0 * (k as f64).powi(2) * x * x).exp();
            sum += sign * term;
            if term < 1e-300 {
                break;
            }
        }
        2.0 * sum
    };
    p.clamp(0.0, 1.0)
}

/// Run cumulative periodogram analysis (convenience wrapper).
pub fn run_cpgram(x: &[f64], taper: Option<f64>) -> Result<CpgramResult, String> {
    cpgram(x, taper)
}

/// Test for white noise using the cumulative periodogram.
///
/// Returns the Stephens-modified Kolmogorov statistic and its asymptotic p-value for the null
/// hypothesis that the series is white noise.
pub fn white_noise_test(x: &[f64], taper: Option<f64>) -> Result<(f64, f64), String> {
    let result = cpgram(x, taper)?;
    Ok((result.ks_statistic, result.ks_p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    use p2a_validation::{Check, Tol};

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), name, crate::data::columns::NaPolicy::Omit)
            .unwrap()
            .0
    }

    fn check_cpgram(case_id: &str, series: &str) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let x = col(&ds, series);
        let n = x.len();
        let r = cpgram(&x, Some(0.1)).unwrap();
        let k = |s: &str| format!("cpgram_{series}_{s}");
        Check::new(&case)
            .vector(&k("freq"), &r.freq, Tol::coef(n))
            .vector(&k("cpgram"), &r.cpgram, Tol::stat(n))
            .scalar(&k("crit"), r.critical_value, Tol::stat(n))
            .scalar(&k("max_dev"), r.max_deviation, Tol::stat(n))
            .scalar(&k("stat"), r.ks_statistic, Tol::stat(n))
            .scalar(&k("p"), r.ks_p_value, Tol::p_value(n))
            .scalar(
                &k("white"),
                if r.is_white_noise { 1.0 } else { 0.0 },
                Tol::exact(),
            )
            .finish();
        let lower_ok = r
            .lower_ci
            .iter()
            .zip(&r.freq)
            .all(|(l, f)| (*l - (2.0 * f - r.critical_value).max(0.0)).abs() < 1e-15);
        assert!(lower_ok);
    }

    /// N(3, 1) white noise, n = 100: 50 frequencies k/100, crit 0.18843774417144613, max dev
    /// 0.10889787598160139, statistic 0.7847860641362071, p 0.5690823493746275, white.
    #[test]
    fn test_validate_cpgram_white_noise() {
        check_cpgram("ts_filter_ar_cpt", "wn");
    }

    /// AR(1) 0.9 around 10, n = 120: 60 frequencies, crit 0.1723313593732768, max dev
    /// 0.7753420099521079, statistic 6.1098250100511695, p 7.527266358277316e-33, not white.
    #[test]
    fn test_validate_cpgram_ar1() {
        check_cpgram("ts_filter_ar_cpt", "ar1");
    }

    /// Asymptotic Kolmogorov survival function. R 4.3.3: 1 - .Call(stats:::C_pKS2, x, 1e-15) for
    /// x < 1, and 2 sum_k (-1)^(k-1) exp(-2 k^2 x^2) for x >= 1 (where 1 - pKS2 loses digits).
    #[test]
    fn test_kolmogorov_sf_matches_r() {
        for (x, p) in [
            (0.3, 0.9999906941986655),
            (0.5, 0.9639452436648751),
            (1.0, 0.26999967167735456),
            (1.358, 0.05002679733444698),
            (2.0, 0.0006709252557796953),
            (3.0, 3.045995948942526e-08),
        ] {
            assert!(
                Tol::new(1e-12, 1e-300).allows(kolmogorov_sf(x), p),
                "x = {x}: {} vs {p}",
                kolmogorov_sf(x)
            );
        }
        assert_eq!(kolmogorov_sf(0.0), 1.0);
    }

    /// R spec.taper: "'p' must be between 0 and 0.5"; the series must be finite.
    #[test]
    fn test_cpgram_rejects_bad_taper_and_non_finite() {
        let x: Vec<f64> = (0..40).map(|i| ((i * 13) % 7) as f64 + 3.0).collect();
        for t in [-0.1, 0.6, f64::NAN] {
            assert!(cpgram(&x, Some(t)).unwrap_err().contains("taper"), "{t}");
        }
        let mut y = x.clone();
        y[3] = f64::INFINITY;
        assert!(cpgram(&y, None).unwrap_err().contains("finite"));
    }
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
    fn test_apply_taper() {
        let mut x = vec![1.0; 10];
        apply_taper(&mut x, 0.3);

        // R: spec.taper(rep(1, 10), 0.3) has w = 0.5 (1 - cos(pi (2i - 1) / 6)), i = 1..3.
        let w = [0.06698729810778065, 0.5, 0.9330127018922194];
        for i in 0..3 {
            assert_relative_eq!(x[i], w[i], max_relative = 1e-14);
            assert_relative_eq!(x[9 - i], w[i], max_relative = 1e-14);
        }
        assert_eq!(&x[3..7], &[1.0; 4]);
    }

    #[test]
    fn test_white_noise_test() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let (ks_stat, p_value) = white_noise_test(&x, Some(0.1)).unwrap();

        assert!(ks_stat >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[cfg(feature = "spectral-analysis")]
    use rand::{Rng, SeedableRng};
    #[cfg(feature = "spectral-analysis")]
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

    /// R 4.3.3: x <- mk(16); (Mod(fft(x - mean(x)))^2/16)[2:9]
    #[test]
    fn test_validate_periodogram_against_r() {
        const R: [f64; 8] = [
            1.80591047303584,
            3.57672788347167,
            0.376257997633748,
            0.00111389265450212,
            0.0103007549093163,
            0.22385447313,
            0.000553155027280382,
            0.0282290746094887,
        ];
        let x = mk(16);
        let mean = x.iter().sum::<f64>() / 16.0;
        let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let p = compute_periodogram(&centred).unwrap();
        assert_eq!(p.len(), 8);
        for (a, b) in p.iter().zip(R) {
            assert_relative_eq!(*a, b, max_relative = 1e-9);
        }
    }

    /// The body of R's stats::cpgram with taper = 0 on x <- mk(65) (R 4.3.3):
    ///   xx <- spec.taper(scale(x, TRUE, FALSE), p = 0); y <- Mod(fft(xx))^2/65; y[1] <- 0
    ///   y <- y[1:33]; (cumsum(y)/sum(y))[-1]
    /// Untapered, this module's cumulative periodogram is R's without the leading 0
    /// (the tapered case: test_validate_cpgram_white_noise / _ar1).
    #[test]
    fn test_validate_cpgram_untapered_against_r() {
        const R: [f64; 32] = [
            0.100911253103486,
            0.127560049770315,
            0.140693342474136,
            0.149483378650379,
            0.362132157418225,
            0.369968644005227,
            0.380512496974472,
            0.406740755514208,
            0.860618017893972,
            0.926542914253333,
            0.936800090507442,
            0.94747044688618,
            0.950647391371401,
            0.952659066496785,
            0.954166999426409,
            0.955527892455196,
            0.957729172326381,
            0.96181462299091,
            0.962016154050691,
            0.962106138742785,
            0.962121805715227,
            0.962177610261033,
            0.964987905713569,
            0.987299832161561,
            0.990250012837924,
            0.991797741540399,
            0.992883530625223,
            0.993732346343451,
            0.994725671614886,
            0.997696480844003,
            0.998935361954203,
            1.0,
        ];
        let r = cpgram(&mk(65), Some(0.0)).unwrap();
        assert_eq!(r.cpgram.len(), 32);
        for (a, b) in r.cpgram.iter().zip(R) {
            assert_relative_eq!(*a, b, max_relative = 1e-10);
        }
    }

    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn fft_periodogram_matches_direct_dft() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        // even, odd and prime (Bluestein) lengths
        for n in [4usize, 5, 16, 97, 1000, 1009] {
            let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() - 0.5).collect();
            let mean = x.iter().sum::<f64>() / n as f64;
            let c: Vec<f64> = x.iter().map(|v| v - mean).collect();
            let fast = periodogram_fft(&c).unwrap();
            let slow = periodogram_direct(&c).unwrap();
            let scale = slow.iter().cloned().fold(0.0, f64::max);
            assert_eq!(fast.len(), n / 2);
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a - b).abs() <= 1e-9 * scale, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn direct_periodogram_honours_cancellation() {
        assert!(cancelled(|| periodogram_direct(&[1.0, -1.0, 0.5, 0.2, -0.3])).is_err());
    }
}
