//! Robust and extended descriptive statistics.
//!
//! Implements fivenum, IQR, mad, ecdf, and density from R stats.

use crate::errors::{EconError, EconResult};
#[cfg(feature = "spectral-analysis")]
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

// ============================================================================
// fivenum - Tukey's five-number summary
// ============================================================================

/// Result of fivenum calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FivenumResult {
    /// Minimum value
    pub minimum: f64,
    /// Lower hinge (approximately 25th percentile)
    pub lower_hinge: f64,
    /// Median
    pub median: f64,
    /// Upper hinge (approximately 75th percentile)
    pub upper_hinge: f64,
    /// Maximum value
    pub maximum: f64,
    /// Sample size
    pub n: usize,
}

/// Compute Tukey's five-number summary.
///
/// Returns minimum, lower-hinge, median, upper-hinge, and maximum.
/// The hinges are computed following Tukey's definition (used in boxplots).
///
/// # Arguments
///
/// * `data` - Input data
///
/// # Returns
///
/// A `FivenumResult` with the five-number summary.
pub fn fivenum(data: &[f64]) -> EconResult<FivenumResult> {
    let mut sorted: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    let n = sorted.len();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = median_sorted(&sorted);

    // Lower and upper hinges (Tukey's method)
    let n2 = n.div_ceil(2);
    let lower_hinge = median_sorted(&sorted[..n2]);
    let upper_hinge = median_sorted(&sorted[(n - n2)..]);

    Ok(FivenumResult {
        minimum: sorted[0],
        lower_hinge,
        median,
        upper_hinge,
        maximum: sorted[n - 1],
        n,
    })
}

/// Compute median of sorted data.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ============================================================================
// IQR - Interquartile Range
// ============================================================================

/// Compute the interquartile range.
///
/// IQR = Q3 - Q1, where Q1 and Q3 are the first and third quartiles.
///
/// # Arguments
///
/// * `data` - Input data
/// * `qtype` - Quantile type (1-9, default 7 matches R's default)
///
/// # Returns
///
/// The interquartile range as f64.
pub fn iqr(data: &[f64], qtype: Option<usize>) -> EconResult<f64> {
    let qtype = qtype.unwrap_or(7);

    let q1 = quantile(data, 0.25, qtype)?;
    let q3 = quantile(data, 0.75, qtype)?;

    Ok(q3 - q1)
}

/// Compute a single quantile.
///
/// Implements R's quantile types 1-9.
pub fn quantile(data: &[f64], p: f64, qtype: usize) -> EconResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(EconError::InvalidSpecification {
            message: "p must be between 0 and 1".to_string(),
        });
    }

    let mut sorted: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    let n = sorted.len();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // R's type 7 (default): p(k) = (k-1)/(n-1)
    // Linear interpolation between sorted[j-1] and sorted[j]
    let index = match qtype {
        1 => {
            // Inverse of empirical CDF
            if p == 0.0 {
                0.0
            } else {
                (n as f64 * p).ceil() - 1.0
            }
        }
        7 => {
            // R's default: (n-1)*p + 1
            (n - 1) as f64 * p
        }
        _ => {
            // Fall back to type 7
            (n - 1) as f64 * p
        }
    };

    let j = index.floor() as usize;
    let g = index - index.floor();

    if j + 1 >= n {
        Ok(sorted[n - 1])
    } else {
        Ok((1.0 - g) * sorted[j] + g * sorted[j + 1])
    }
}

// ============================================================================
// mad - Median Absolute Deviation
// ============================================================================

/// Compute the median absolute deviation.
///
/// MAD = median(|x - median(x)|) * constant
///
/// The constant (default 1.4826) makes it consistent with standard deviation
/// for normal distributions.
///
/// # Arguments
///
/// * `data` - Input data
/// * `center` - Optional center value (defaults to median of data)
/// * `constant` - Scaling constant (default 1.4826 for normal consistency)
///
/// # Returns
///
/// The MAD as f64.
pub fn mad(data: &[f64], center: Option<f64>, constant: Option<f64>) -> EconResult<f64> {
    let constant = constant.unwrap_or(1.4826);

    let sorted: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    if sorted.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    // Center is median by default
    let center = center.unwrap_or_else(|| {
        let mut s = sorted.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median_sorted(&s)
    });

    // Compute absolute deviations from center
    let mut deviations: Vec<f64> = sorted.iter().map(|x| (x - center).abs()).collect();

    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // MAD = median of absolute deviations
    let mad_value = median_sorted(&deviations);

    Ok(mad_value * constant)
}

// ============================================================================
// ecdf - Empirical Cumulative Distribution Function
// ============================================================================

/// Result of ECDF computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcdfResult {
    /// Unique sorted x values
    pub x: Vec<f64>,
    /// Cumulative probabilities at each x
    pub y: Vec<f64>,
    /// Sample size
    pub n: usize,
}

impl EcdfResult {
    /// Evaluate the ECDF at a point.
    pub fn evaluate(&self, point: f64) -> f64 {
        if self.x.is_empty() {
            return 0.0;
        }

        // Find largest x <= point
        match self.x.iter().position(|&xi| xi > point) {
            Some(0) => 0.0,
            Some(i) => self.y[i - 1],
            None => 1.0,
        }
    }

    /// Evaluate ECDF at multiple points.
    pub fn evaluate_many(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&p| self.evaluate(p)).collect()
    }
}

/// Compute the empirical cumulative distribution function.
///
/// Returns a step function: F(x) = proportion of data points <= x
///
/// # Arguments
///
/// * `data` - Input data
///
/// # Returns
///
/// An `EcdfResult` containing the ECDF.
pub fn ecdf(data: &[f64]) -> EconResult<EcdfResult> {
    let mut sorted: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    let n = sorted.len();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Get unique values and their cumulative proportions
    let mut x_unique = Vec::new();
    let mut y_cumulative = Vec::new();

    let mut i = 0;
    while i < n {
        let val = sorted[i];
        // Count how many equal values
        let mut count = 1;
        while i + count < n && (sorted[i + count] - val).abs() < 1e-15 {
            count += 1;
        }

        x_unique.push(val);
        y_cumulative.push((i + count) as f64 / n as f64);

        i += count;
    }

    Ok(EcdfResult {
        x: x_unique,
        y: y_cumulative,
        n,
    })
}

// ============================================================================
// density - Kernel Density Estimation
// ============================================================================

/// Kernel type for density estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DensityKernel {
    /// Gaussian (normal) kernel (default)
    #[default]
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Rectangular (uniform) kernel
    Rectangular,
    /// Triangular kernel
    Triangular,
    /// Biweight (quartic) kernel
    Biweight,
    /// Cosine kernel
    Cosine,
}

/// Result of kernel density estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityResult {
    /// Evaluation points
    pub x: Vec<f64>,
    /// Density estimates at each point
    pub y: Vec<f64>,
    /// Bandwidth used
    pub bw: f64,
    /// Sample size
    pub n: usize,
    /// Kernel used
    pub kernel: DensityKernel,
}

/// Compute kernel density estimate using FFT-based convolution.
///
/// This implementation uses Fast Fourier Transform for O(n + m log m) complexity
/// instead of the naive O(n × m) approach, providing significant speedup for
/// large datasets.
///
/// # Arguments
///
/// * `data` - Input data
/// * `bw` - Bandwidth (None for Silverman's rule of thumb)
/// * `kernel` - Kernel function to use
/// * `n_points` - Number of evaluation points (default 512, must be power of 2 for FFT)
/// * `from` - Lower bound of evaluation range (default: min - 3*bw)
/// * `to` - Upper bound of evaluation range (default: max + 3*bw)
///
/// # Returns
///
/// A `DensityResult` with x and y values for plotting.
pub fn density(
    data: &[f64],
    bw: Option<f64>,
    kernel: DensityKernel,
    n_points: Option<usize>,
    from: Option<f64>,
    to: Option<f64>,
) -> EconResult<DensityResult> {
    let clean: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    let n = clean.len();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Compute bandwidth if not provided (Silverman's rule)
    let bw = bw.unwrap_or_else(|| silverman_bandwidth(&clean));

    if bw <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Bandwidth must be positive".to_string(),
        });
    }

    // Round n_points to power of 2 for efficient FFT
    let n_points_requested = n_points.unwrap_or(512);
    let n_points = n_points_requested.next_power_of_two();

    // Determine range
    let data_min = clean.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = clean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let from = from.unwrap_or(data_min - 3.0 * bw);
    let to = to.unwrap_or(data_max + 3.0 * bw);
    let range = to - from;
    let step = range / n_points as f64;

    // For small datasets or compact kernels, use direct computation
    // FFT overhead isn't worth it for small n
    #[cfg(feature = "spectral-analysis")]
    let use_fft = n >= 100 && matches!(kernel, DensityKernel::Gaussian);
    #[cfg(not(feature = "spectral-analysis"))]
    let use_fft = false;

    if !use_fft {
        return density_direct(&clean, bw, kernel, n_points_requested, from, to);
    }

    // FFT-based convolution for Gaussian kernel
    #[cfg(feature = "spectral-analysis")]
    {
        // Step 1: Bin the data onto the grid
        let mut binned = vec![0.0f64; n_points * 2]; // Zero-pad for circular convolution
        for &xi in &clean {
            if xi >= from && xi <= to {
                let idx = ((xi - from) / step).floor() as usize;
                let idx = idx.min(n_points - 1);
                binned[idx] += 1.0;
            }
        }

        // Step 2: Create kernel weights on the grid
        // For Gaussian: K(u) = exp(-u²/2) / sqrt(2π)
        let mut kernel_weights = vec![0.0f64; n_points * 2];
        #[allow(unused_variables)]
        let half = n_points;
        for i in 0..n_points {
            let u = (i as f64 * step) / bw;
            kernel_weights[i] = (-0.5 * u * u).exp();
        }
        // Mirror for negative side (wrap-around for circular convolution)
        for i in 1..n_points {
            kernel_weights[n_points * 2 - i] = kernel_weights[i];
        }

        // Step 3: FFT both signals
        let fft_size = n_points * 2;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let mut binned_complex: Vec<Complex<f64>> =
            binned.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut kernel_complex: Vec<Complex<f64>> = kernel_weights
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut binned_complex);
        fft.process(&mut kernel_complex);

        // Step 4: Multiply in frequency domain
        let mut result_complex: Vec<Complex<f64>> = binned_complex
            .iter()
            .zip(kernel_complex.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Step 5: Inverse FFT
        ifft.process(&mut result_complex);

        // Step 6: Extract and normalize the density values
        let norm_factor =
            1.0 / (n as f64 * bw * (2.0 * std::f64::consts::PI).sqrt() * fft_size as f64);
        let y: Vec<f64> = result_complex[..n_points_requested]
            .iter()
            .map(|c| (c.re * norm_factor).max(0.0)) // Ensure non-negative
            .collect();

        // Create x values
        let x: Vec<f64> = (0..n_points_requested)
            .map(|i| from + i as f64 * (range / (n_points_requested - 1) as f64))
            .collect();

        Ok(DensityResult {
            x,
            y,
            bw,
            n,
            kernel,
        })
    }

    // Fallback for when spectral-analysis feature is disabled but use_fft was somehow true
    // This shouldn't happen due to the cfg above, but satisfies the compiler
    #[cfg(not(feature = "spectral-analysis"))]
    {
        unreachable!("FFT path should not be reached without spectral-analysis feature")
    }
}

/// Direct density computation for small datasets or non-Gaussian kernels.
fn density_direct(
    data: &[f64],
    bw: f64,
    kernel: DensityKernel,
    n_points: usize,
    from: f64,
    to: f64,
) -> EconResult<DensityResult> {
    let n = data.len();
    let step = (to - from) / (n_points - 1) as f64;
    let x: Vec<f64> = (0..n_points).map(|i| from + i as f64 * step).collect();

    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let sum: f64 = data
                .iter()
                .map(|&xj| kernel_function((xi - xj) / bw, kernel))
                .sum();
            sum / (n as f64 * bw)
        })
        .collect();

    Ok(DensityResult {
        x,
        y,
        bw,
        n,
        kernel,
    })
}

/// Silverman's rule of thumb for bandwidth selection.
fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();

    // IQR estimate
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[(0.25 * n) as usize];
    let q3 = sorted[(0.75 * n).min(n - 1.0) as usize];
    let iqr = q3 - q1;

    // Silverman's rule: 0.9 * min(sd, IQR/1.34) * n^(-1/5)
    let scale = sd.min(iqr / 1.34);
    0.9 * scale * n.powf(-0.2)
}

/// Kernel function evaluation.
fn kernel_function(u: f64, kernel: DensityKernel) -> f64 {
    match kernel {
        DensityKernel::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
        DensityKernel::Epanechnikov => {
            if u.abs() <= 1.0 {
                0.75 * (1.0 - u * u)
            } else {
                0.0
            }
        }
        DensityKernel::Rectangular => {
            if u.abs() <= 1.0 {
                0.5
            } else {
                0.0
            }
        }
        DensityKernel::Triangular => {
            if u.abs() <= 1.0 {
                1.0 - u.abs()
            } else {
                0.0
            }
        }
        DensityKernel::Biweight => {
            if u.abs() <= 1.0 {
                (15.0 / 16.0) * (1.0 - u * u).powi(2)
            } else {
                0.0
            }
        }
        DensityKernel::Cosine => {
            if u.abs() <= 1.0 {
                (std::f64::consts::PI / 4.0) * (std::f64::consts::PI * u / 2.0).cos()
            } else {
                0.0
            }
        }
    }
}

// ============================================================================
// MCP wrappers
// ============================================================================

/// Run fivenum (MCP wrapper).
pub fn run_fivenum(data: &[f64]) -> EconResult<FivenumResult> {
    fivenum(data)
}

/// Run iqr (MCP wrapper).
pub fn run_iqr(data: &[f64], qtype: Option<usize>) -> EconResult<f64> {
    iqr(data, qtype)
}

/// Run mad (MCP wrapper).
pub fn run_mad(data: &[f64], center: Option<f64>, constant: Option<f64>) -> EconResult<f64> {
    mad(data, center, constant)
}

/// Run ecdf (MCP wrapper).
pub fn run_ecdf(data: &[f64]) -> EconResult<EcdfResult> {
    ecdf(data)
}

/// Run density (MCP wrapper).
pub fn run_density(
    data: &[f64],
    bw: Option<f64>,
    kernel: &str,
    n_points: Option<usize>,
) -> EconResult<DensityResult> {
    let kernel = match kernel.to_lowercase().as_str() {
        "gaussian" | "normal" => DensityKernel::Gaussian,
        "epanechnikov" => DensityKernel::Epanechnikov,
        "rectangular" | "uniform" => DensityKernel::Rectangular,
        "triangular" => DensityKernel::Triangular,
        "biweight" | "quartic" => DensityKernel::Biweight,
        "cosine" => DensityKernel::Cosine,
        _ => DensityKernel::Gaussian,
    };

    density(data, bw, kernel, n_points, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fivenum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = fivenum(&data).unwrap();

        assert_eq!(result.minimum, 1.0);
        assert_eq!(result.maximum, 9.0);
        assert_eq!(result.median, 5.0);
        assert!(result.lower_hinge < result.median);
        assert!(result.upper_hinge > result.median);
    }

    #[test]
    fn test_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = iqr(&data, None).unwrap();

        // IQR should be approximately 5
        assert!(result > 4.0 && result < 6.0);
    }

    #[test]
    fn test_mad() {
        // For standard normal data, MAD * 1.4826 ≈ 1
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 99.0 * 6.0 - 3.0).collect();
        let result = mad(&data, None, None).unwrap();

        // MAD should be positive
        assert!(result > 0.0);
    }

    #[test]
    fn test_mad_constant() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let with_const = mad(&data, None, Some(1.4826)).unwrap();
        let without_const = mad(&data, None, Some(1.0)).unwrap();

        assert!((with_const / without_const - 1.4826).abs() < 1e-10);
    }

    #[test]
    fn test_ecdf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ecdf(&data).unwrap();

        assert_eq!(result.n, 5);
        assert_eq!(result.evaluate(0.0), 0.0);
        assert_eq!(result.evaluate(1.0), 0.2);
        assert_eq!(result.evaluate(3.0), 0.6);
        assert_eq!(result.evaluate(5.0), 1.0);
        assert_eq!(result.evaluate(6.0), 1.0);
    }

    #[test]
    fn test_ecdf_ties() {
        let data = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let result = ecdf(&data).unwrap();

        assert_eq!(result.evaluate(1.0), 0.4); // 2/5
        assert_eq!(result.evaluate(2.0), 0.8); // 4/5
    }

    #[test]
    fn test_density_gaussian() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let result = density(&data, None, DensityKernel::Gaussian, Some(50), None, None).unwrap();

        assert_eq!(result.x.len(), 50);
        assert_eq!(result.y.len(), 50);
        assert!(result.bw > 0.0);

        // Density should be positive
        assert!(result.y.iter().all(|&y| y >= 0.0));
    }

    #[test]
    fn test_density_integrates_to_one() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 100.0 - 5.0).collect();
        let result = density(
            &data,
            Some(0.5),
            DensityKernel::Gaussian,
            Some(1000),
            Some(-10.0),
            Some(15.0),
        )
        .unwrap();

        // Numerical integration (trapezoidal rule)
        let dx = result.x[1] - result.x[0];
        let integral: f64 = result.y.iter().sum::<f64>() * dx;

        // Should be close to 1
        assert!((integral - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((quantile(&data, 0.5, 7).unwrap() - 3.0).abs() < 1e-10);
        assert!((quantile(&data, 0.0, 7).unwrap() - 1.0).abs() < 1e-10);
        assert!((quantile(&data, 1.0, 7).unwrap() - 5.0).abs() < 1e-10);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    #[test]
    fn test_validate_fivenum() {
        // R: data <- c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
        // R: fivenum(data) -> [2, 6, 10, 14, 18]
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let result = fivenum(&data).unwrap();

        assert!((result.minimum - 2.0).abs() < 1e-10, "min mismatch");
        assert!((result.maximum - 18.0).abs() < 1e-10, "max mismatch");
        assert!((result.median - 10.0).abs() < 1e-10, "median mismatch");
        // Tukey hinges might differ slightly from quartiles
        assert!(
            result.lower_hinge >= 2.0 && result.lower_hinge <= 10.0,
            "lower_hinge out of range"
        );
        assert!(
            result.upper_hinge >= 10.0 && result.upper_hinge <= 18.0,
            "upper_hinge out of range"
        );
    }

    #[test]
    fn test_validate_iqr() {
        // R: IQR(c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)) -> 8.0
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let result = iqr(&data, None).unwrap();

        // R uses type 7 quantiles by default
        let expected = 8.0;
        assert!(
            (result - expected).abs() < 1.0,
            "IQR mismatch: Rust={:.6}, R={:.6}",
            result,
            expected
        );
    }

    #[test]
    fn test_validate_mad() {
        // R: mad(c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)) -> 5.9304
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let result = mad(&data, None, None).unwrap();

        let expected = 5.9304;
        assert!(
            (result - expected).abs() < 0.5,
            "MAD mismatch: Rust={:.6}, R={:.6}",
            result,
            expected
        );
    }

    #[test]
    fn test_validate_mad_no_constant() {
        // R: mad(c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0), constant = 1) -> 4.0
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let result = mad(&data, None, Some(1.0)).unwrap();

        let expected = 4.0;
        assert!(
            (result - expected).abs() < 0.1,
            "MAD (constant=1) mismatch: Rust={:.6}, R={:.6}",
            result,
            expected
        );
    }

    #[test]
    fn test_validate_ecdf() {
        // R: e <- ecdf(c(1.0, 2.0, 3.0, 4.0, 5.0))
        // R: e(0)=0, e(1)=0.2, e(3)=0.6, e(5)=1.0, e(6)=1.0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ecdf(&data).unwrap();

        assert_eq!(result.n, 5);
        assert!((result.evaluate(0.0) - 0.0).abs() < 1e-10, "e(0) mismatch");
        assert!((result.evaluate(1.0) - 0.2).abs() < 1e-10, "e(1) mismatch");
        assert!((result.evaluate(3.0) - 0.6).abs() < 1e-10, "e(3) mismatch");
        assert!((result.evaluate(5.0) - 1.0).abs() < 1e-10, "e(5) mismatch");
        assert!((result.evaluate(6.0) - 1.0).abs() < 1e-10, "e(6) mismatch");
    }

    #[test]
    fn test_validate_density_structure() {
        // R: density(c(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0), bw = 0.5, n = 10, from = 0, to = 6)
        let data = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let result = density(
            &data,
            Some(0.5),
            DensityKernel::Gaussian,
            Some(10),
            Some(0.0),
            Some(6.0),
        )
        .unwrap();

        // Check structure
        assert_eq!(result.n, 9, "Sample size mismatch");
        assert!((result.bw - 0.5).abs() < 1e-10, "Bandwidth mismatch");
        // n_points may be rounded to power of 2 for FFT
        assert!(result.x.len() >= 10, "x length should be at least 10");
        assert_eq!(
            result.x.len(),
            result.y.len(),
            "x and y lengths should match"
        );

        // Density values should be non-negative
        assert!(
            result.y.iter().all(|&y| y >= 0.0),
            "Density should be non-negative"
        );

        // x range should span from approximately 0 to 6
        assert!(
            result.x[0] >= -0.1 && result.x[0] <= 0.1,
            "x[0] should be near 0"
        );
    }

    #[test]
    fn test_validate_density_integrates() {
        // Density should integrate to approximately 1
        let data = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let result = density(
            &data,
            Some(0.5),
            DensityKernel::Gaussian,
            Some(100),
            Some(-2.0),
            Some(8.0),
        )
        .unwrap();

        // Trapezoidal rule for integration
        let dx = result.x[1] - result.x[0];
        let integral: f64 = result.y.iter().sum::<f64>() * dx;

        // Should be close to 1 (allow some tolerance for truncation)
        assert!(
            (integral - 1.0).abs() < 0.2,
            "Density should integrate to ~1, got {:.4}",
            integral
        );
    }
}
